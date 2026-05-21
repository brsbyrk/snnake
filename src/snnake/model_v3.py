"""Structured world model for sNNake v3 — scaled for autoregressive stability.

Key changes from v2.2:
  - Body encoder/decoder: 16→48 hidden dim (better sequence modeling)
  - State MLP: 28→256→256→256 (3 layers, 256-wide vs 64-wide)
  - Output heads: matching 256-dim hidden state
  - Total params: ~175K (vs 12K in v2.2)

Target: 1000+ step autoregressive stability with proper training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Max body length for fixed-size batching (10×10 board → 100 max)
MAX_BODY_LEN = 40
GRID_SIZE = 10

# Input dimensions
HEAD_DIM = 2
FOOD_DIM = 2
DIR_DIM = 4
ACT_DIM = 3
GO_DIM = 1

# Scaled dimensions
BODY_EMBED_DIM = 48    # up from 16
STATE_HIDDEN = 256      # up from 64
STATE_DEPTH = 3         # same depth, wider


class StructuredBodyEncoder(nn.Module):
    """Encode variable-length body sequence into a fixed-size context vector."""

    def __init__(self, segment_dim: int = 2, hidden_dim: int = BODY_EMBED_DIM):
        super().__init__()
        self.segment_proj = nn.Linear(segment_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, body_positions, body_mask):
        """Encode body sequence.

        Args:
            body_positions: (B, L, 2) normalized coordinates, zero-padded
            body_mask:      (B, L) 1 for valid positions, 0 for padding

        Returns:
            body_context: (B, hidden_dim) final hidden state of GRU
        """
        B, L, _ = body_positions.shape

        # Project each segment
        emb = F.relu(self.segment_proj(body_positions))  # (B, L, hidden)

        # Pack padded sequence so GRU ignores padding
        lengths = body_mask.sum(dim=1).long().clamp(min=1)  # (B,)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        _, h_n = self.gru(packed)  # h_n: (1, B, hidden)
        return h_n.squeeze(0)  # (B, hidden)


class StructuredBodyDecoder(nn.Module):
    """Decode shifted body sequence via GRU."""

    def __init__(self, segment_dim: int = 2, hidden_dim: int = BODY_EMBED_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.segment_proj = nn.Linear(segment_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, segment_dim)

    def forward(self, next_head, old_body_positions, body_mask, ate_food=None):
        """Decode shifted body.

        Args:
            next_head:          (B, 2) predicted next head position
            old_body_positions: (B, L, 2) old body (used as decoder input)
            body_mask:          (B, L) valid position mask
            ate_food:           not used directly — handled in loss

        Returns:
            new_body: (B, L, 2) predicted body after shift
        """
        B, L, _ = old_body_positions.shape

        # Initial hidden state: encode next_head
        h = self.segment_proj(next_head)  # (B, hidden)

        outputs = []
        # Position 0: always the new head
        h = self.gru(h, h)
        out0 = self.output_proj(h)
        outputs.append(out0)

        # Shift: for position i (1..L-1), read old_body[i-1], predict new_body[i]
        for i in range(1, L):
            seg_in = old_body_positions[:, i - 1, :]
            seg_emb = F.relu(self.segment_proj(seg_in))
            h = self.gru(seg_emb, h)
            out = self.output_proj(h)
            outputs.append(out)

        new_body = torch.stack(outputs, dim=1)  # (B, L, 2)

        # Mask out invalid positions
        new_body = new_body * body_mask.unsqueeze(-1)

        return new_body


class StructuredWorldModelV3(nn.Module):
    """Scaled coordinate-based world model for Snake.

    Parameters: ~175K
    """

    def __init__(self):
        super().__init__()

        # Body encoder/decoder
        self.body_encoder = StructuredBodyEncoder(segment_dim=2, hidden_dim=BODY_EMBED_DIM)
        self.body_decoder = StructuredBodyDecoder(segment_dim=2, hidden_dim=BODY_EMBED_DIM)

        # Combined state MLP — wider layers
        state_in_dim = HEAD_DIM + FOOD_DIM + DIR_DIM + ACT_DIM + GO_DIM + BODY_EMBED_DIM
        self.fc1 = nn.Linear(state_in_dim, STATE_HIDDEN)
        self.fc2 = nn.Linear(STATE_HIDDEN, STATE_HIDDEN)
        self.fc3 = nn.Linear(STATE_HIDDEN, STATE_HIDDEN)

        # Output heads
        self.head_next = nn.Linear(STATE_HIDDEN, HEAD_DIM)
        self.food_next = nn.Linear(STATE_HIDDEN, FOOD_DIM)
        self.ate_head = nn.Linear(STATE_HIDDEN, 1)
        self.go_head = nn.Linear(STATE_HIDDEN, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        head,             # (B, 2)
        direction,        # (B, 4)
        action,           # (B, 3)
        food,             # (B, 2)
        game_over,        # (B, 1)
        body_positions,   # (B, L_max, 2)
        body_mask,        # (B, L_max)
    ):
        B = head.shape[0]

        # Encode body
        body_context = self.body_encoder(body_positions, body_mask)  # (B, 48)

        # Concatenate all state
        state = torch.cat([head, food, direction, action, game_over, body_context], dim=1)

        # MLP — 3 hidden layers, 256-wide
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Output heads
        next_head = self.head_next(x)
        next_food = self.food_next(x)
        ate_food_logits = self.ate_head(x)
        game_over_logits = self.go_head(x)

        # Decode body
        new_body = self.body_decoder(next_head, body_positions, body_mask)

        # Derive direction deterministically from old_dir + action
        old_dir_idx = direction.argmax(dim=1)
        action_idx = action.argmax(dim=1)
        offsets = torch.tensor([-1, 0, 1], device=head.device)
        new_dir_idx = (old_dir_idx + offsets[action_idx]) % 4
        next_dir_logits = torch.zeros(B, 4, device=head.device)
        next_dir_logits[torch.arange(B), new_dir_idx] = 1.0
        next_dir_logits = next_dir_logits + 0.001

        return next_head, next_food, ate_food_logits, game_over_logits, next_dir_logits, new_body

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


def compute_game_over_weight(
    head_positions: torch.Tensor,
    target_go: torch.Tensor,
    grid_size: int = GRID_SIZE,
    edge_margin: float = 1.0,
    false_pos_weight: float = 3.0,
) -> torch.Tensor:
    """Per-sample weight for game_over loss — penalize false positives near walls."""
    hx = head_positions[:, 0] * (grid_size - 1)
    hy = head_positions[:, 1] * (grid_size - 1)
    dist_to_wall = torch.min(torch.stack([
        hx, hy,
        (grid_size - 1) - hx,
        (grid_size - 1) - hy,
    ], dim=0), dim=0).values
    near_wall = (dist_to_wall < edge_margin).float()
    not_go = (1.0 - target_go.view(-1).float())
    weight = 1.0 + false_pos_weight * near_wall * not_go
    return weight


def compute_loss(
    next_head, next_food, ate_logits, go_logits, next_dir_logits, new_body_pred,
    target_head, target_food, target_ate, target_go, target_dir, target_body,
    body_mask,
    head_positions=None,
    lambda_head=1.0, lambda_food=0.5, lambda_ate=0.5, lambda_go=0.5,
    lambda_body=1.0,
):
    """Compute multi-task loss for structured model v3."""
    # Head position: MSE
    head_loss = F.mse_loss(next_head, target_head)

    # Food position: MSE
    food_loss = F.mse_loss(next_food, target_food)

    # Ate food: BCE
    ate_loss = F.binary_cross_entropy_with_logits(
        ate_logits.view(-1), target_ate.view(-1).float()
    )

    # Game over: BCE with per-sample weighting
    go_loss = F.binary_cross_entropy_with_logits(
        go_logits.view(-1), target_go.view(-1).float(),
        weight=compute_game_over_weight(target_head, target_go),
    )

    # Direction: CE (should be near-perfect since it's derived)
    dir_loss = F.cross_entropy(next_dir_logits, target_dir.long())

    # Body: masked MSE
    body_diff = (new_body_pred - target_body) ** 2
    body_loss = (body_diff * body_mask.unsqueeze(-1)).sum() / body_mask.sum().clamp(min=1)

    total = (
        lambda_head * head_loss
        + lambda_food * food_loss
        + lambda_ate * ate_loss
        + lambda_go * go_loss
        + lambda_body * body_loss
    )

    return {
        "total": total,
        "head": head_loss,
        "food": food_loss,
        "ate": ate_loss,
        "game_over": go_loss,
        "direction": dir_loss,
        "body": body_loss,
    }


def accuracy_metrics(
    next_head, next_food, ate_logits, go_logits, new_body_pred,
    target_head, target_food, target_ate, target_go, target_body,
    body_mask,
    threshold=0.05,
):
    """Compute accuracy metrics."""
    with torch.no_grad():
        head_dist = torch.norm(next_head - target_head, dim=1)
        head_acc = (head_dist < threshold).float().mean().item()

        ate_pred = (ate_logits > 0).float()
        ate_acc = (ate_pred.view(-1) == target_ate.view(-1).float()).float().mean().item()

        go_pred = (go_logits > 0).float()
        go_acc = (go_pred.view(-1) == target_go.view(-1).float()).float().mean().item()

        body_diff = torch.norm(new_body_pred - target_body, dim=2)
        body_correct = (body_diff < threshold).float() * body_mask
        body_acc = body_correct.sum() / body_mask.sum().clamp(min=1)
        body_acc = body_acc.item()

        head_exact = (head_dist < threshold)
        body_exact = (body_diff < threshold).all(dim=1)
        full_match = (head_exact & (ate_pred.view(-1) == target_ate.view(-1).float()).bool()
                     & (go_pred.view(-1) == target_go.view(-1).float()).bool()
                     & body_exact).float().mean().item()

    return {
        "head_acc": head_acc,
        "ate_acc": ate_acc,
        "go_acc": go_acc,
        "body_acc": body_acc,
        "full_match": full_match,
    }
