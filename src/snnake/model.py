"""CNN world model for sNNake.

Predicts next game state given current state + action.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import (
    GRID_SIZE,
    NUM_CELL_TYPES,
    NUM_DIRECTIONS,
    NUM_ACTIONS,
)


class WorldModel(nn.Module):
    """CNN-based world model that predicts next Snake state.

    Architecture:
        Grid (4, 10, 10) → Conv layers → Flatten → Concat extras → MLP → Output heads

    Input:
        grid:      (B, 4, 10, 10) one-hot encoded grid
        direction: (B, 4)          one-hot
        action:    (B, 3)          one-hot
        game_over: (B, 1)          binary

    Output:
        grid_logits:      (B, 4, 10, 10) — softmax over channel dim
        direction_logits: (B, 4)          — softmax
        game_over_logits: (B, 1)          — sigmoid
    """

    def __init__(self):
        super().__init__()

        # CNN encoder for grid
        self.conv1 = nn.Conv2d(NUM_CELL_TYPES, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Flattened grid features
        self._grid_feat_dim = 32 * GRID_SIZE * GRID_SIZE  # 32 * 10 * 10 = 3200

        # Extra features: direction(4) + action(3) + game_over(1)
        self._extra_dim = NUM_DIRECTIONS + NUM_ACTIONS + 1  # 8

        # MLP decoder
        self.fc1 = nn.Linear(self._grid_feat_dim + self._extra_dim, 256)
        self.fc2 = nn.Linear(256, 256)

        # Output heads
        self.head_grid = nn.Linear(256, NUM_CELL_TYPES * GRID_SIZE * GRID_SIZE)
        self.head_direction = nn.Linear(256, NUM_DIRECTIONS)
        self.head_game_over = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, grid, direction, action, game_over):
        """Forward pass.

        Args:
            grid:      (B, 4, 10, 10) float tensor
            direction: (B, 4)          float tensor
            action:    (B, 3)          float tensor
            game_over: (B, 1)          float tensor

        Returns:
            grid_logits:      (B, 4, 10, 10)
            direction_logits: (B, 4)
            game_over_logits: (B, 1)
        """
        B = grid.shape[0]

        # Encode grid
        x = F.relu(self.bn1(self.conv1(grid)))   # (B, 16, 10, 10)
        x = F.relu(self.bn2(self.conv2(x)))       # (B, 32, 10, 10)
        x = x.view(B, -1)                          # (B, 3200)

        # Concatenate extra features
        extras = torch.cat([direction, action, game_over], dim=1)  # (B, 8)
        x = torch.cat([x, extras], dim=1)                           # (B, 3208)

        # MLP
        x = F.relu(self.fc1(x))  # (B, 256)
        x = F.relu(self.fc2(x))  # (B, 256)

        # Output heads
        grid_logits = self.head_grid(x)                    # (B, 400)
        grid_logits = grid_logits.view(B, NUM_CELL_TYPES, GRID_SIZE, GRID_SIZE)  # (B, 4, 10, 10)

        direction_logits = self.head_direction(x)  # (B, 4)
        game_over_logits = self.head_game_over(x)   # (B, 1)

        return grid_logits, direction_logits, game_over_logits

    def predict_step(self, grid, direction, game_over, action_idx):
        """Convenience: single-step prediction from numpy inputs.

        Args:
            grid:      (4, 10, 10) numpy array
            direction: (4,) numpy array
            game_over: (1,) numpy array
            action_idx: int (0=left, 1=straight, 2=right)

        Returns:
            next_grid:      (4, 10, 10) numpy array (probabilities, not argmax)
            next_direction: (4,) numpy array (probabilities)
            next_game_over: float
        """
        from .encoding import encode_action

        device = next(self.parameters()).device
        action = encode_action(action_idx)

        # Add batch dim and move to model's device
        grid_b = torch.from_numpy(grid).unsqueeze(0).to(device)
        dir_b = torch.from_numpy(direction).unsqueeze(0).to(device)
        act_b = torch.from_numpy(action).unsqueeze(0).to(device)
        go_b = torch.from_numpy(game_over).unsqueeze(0).to(device)

        with torch.no_grad():
            g_logits, d_logits, go_logits = self(grid_b, dir_b, act_b, go_b)

        # Softmax for probabilities
        g_probs = F.softmax(g_logits, dim=1).squeeze(0).cpu().numpy()
        d_probs = F.softmax(d_logits, dim=1).squeeze(0).cpu().numpy()
        go_prob = torch.sigmoid(go_logits).item()

        return g_probs, d_probs, go_prob


def compute_loss(
    grid_logits: torch.Tensor,    # (B, 4, 10, 10)
    direction_logits: torch.Tensor,  # (B, 4)
    game_over_logits: torch.Tensor,  # (B, 1)
    grid_target: torch.Tensor,    # (B, 4, 10, 10) — one-hot or integer labels
    direction_target: torch.Tensor,  # (B,) — integer class indices
    game_over_target: torch.Tensor,  # (B, 1) — binary
    lambda_grid: float = 1.0,
    lambda_dir: float = 0.5,
    lambda_go: float = 0.5,
) -> dict:
    """Compute multi-task loss.

    Args:
        grid_target: one-hot (B, 4, 10, 10) or integer labels (B, 10, 10)
        direction_target: integer class indices (B,)

    Returns:
        dict with 'total', 'grid', 'direction', 'game_over' loss values
    """
    B = grid_logits.shape[0]

    # Grid loss: CrossEntropy over 4 classes per cell
    # Flatten spatial dims: (B, 4, 10, 10) → (B*100, 4)
    g_logits = grid_logits.permute(0, 2, 3, 1).reshape(-1, NUM_CELL_TYPES)  # (B*100, 4)
    if grid_target.dtype == torch.float32 and grid_target.shape == grid_logits.shape:
        # One-hot targets: convert to integer labels
        g_target = grid_target.argmax(dim=1)  # (B, 10, 10)
    else:
        g_target = grid_target
    g_target = g_target.reshape(-1).long()  # (B*100,)
    grid_loss = F.cross_entropy(g_logits, g_target)

    # Direction loss
    dir_loss = F.cross_entropy(direction_logits, direction_target.long())

    # Game over loss (BCEWithLogits handles numerical stability)
    go_loss = F.binary_cross_entropy_with_logits(
        game_over_logits.view(-1),
        game_over_target.view(-1).float(),
    )

    total = lambda_grid * grid_loss + lambda_dir * dir_loss + lambda_go * go_loss

    return {
        "total": total,
        "grid": grid_loss,
        "direction": dir_loss,
        "game_over": go_loss,
    }


def accuracy(
    grid_logits: torch.Tensor,
    direction_logits: torch.Tensor,
    game_over_logits: torch.Tensor,
    grid_target: torch.Tensor,
    direction_target: torch.Tensor,
    game_over_target: torch.Tensor,
) -> dict:
    """Compute accuracy metrics.

    Grid accuracy: per-cell accuracy over all 100 cells.
    Direction accuracy: exact match over 4 classes.
    Game over accuracy: binary accuracy.
    """
    with torch.no_grad():
        # Grid accuracy
        g_pred = grid_logits.argmax(dim=1)  # (B, 10, 10)
        if grid_target.dtype == torch.float32:
            g_target = grid_target.argmax(dim=1)
        else:
            g_target = grid_target
        grid_acc = (g_pred == g_target).float().mean().item()

        # Direction accuracy
        d_pred = direction_logits.argmax(dim=1)
        dir_acc = (d_pred == direction_target).float().mean().item()

        # Game over accuracy
        go_pred = (torch.sigmoid(game_over_logits) > 0.5).float()
        go_acc = (go_pred.view(-1) == game_over_target.view(-1).float()).float().mean().item()

    return {
        "grid_acc": grid_acc,
        "dir_acc": dir_acc,
        "go_acc": go_acc,
    }
