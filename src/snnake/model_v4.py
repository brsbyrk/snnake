"""Discrete world model for sNNake v4 — classification over grid cells.

Key differences from v3:
  - Head, food, and body segments are predicted as grid cell indices (0-99)
  - Cross-entropy loss instead of MSE — no numerical drift
  - Single-step teacher forcing only (no BPTT needed)
  - ~170K params, same scale as v3

With discrete outputs, the model never drifts — it either predicts the
exact correct cell or it's wrong. Closed-loop playback is stable
indefinitely as long as accuracy per step stays high.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

GRID_SIZE = 10
NUM_CELLS = GRID_SIZE * GRID_SIZE  # 100
MAX_BODY_LEN = 40
EMBED_DIM = 16
BODY_HIDDEN = 32
STATE_HIDDEN = 192


class DiscreteBodyEncoder(nn.Module):
    """Encode variable-length body sequence into a fixed-size context vector.

    Body segments are given as cell indices (0-99), embedded, then GRU-encoded.
    """

    def __init__(self, num_cells: int = NUM_CELLS, embed_dim: int = EMBED_DIM, hidden_dim: int = BODY_HIDDEN):
        super().__init__()
        self.embed = nn.Embedding(num_cells, embed_dim)
        self.segment_proj = nn.Linear(embed_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, body_indices, body_mask):
        """Encode body sequence.

        Args:
            body_indices: (B, L) integer cell indices (0-99)
            body_mask:    (B, L) binary mask

        Returns:
            body_context: (B, hidden_dim)
        """
        B, L = body_indices.shape
        emb = self.embed(body_indices)  # (B, L, embed_dim)
        emb = F.relu(self.segment_proj(emb))  # (B, L, hidden)

        lengths = body_mask.sum(dim=1).long().clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        return h_n.squeeze(0)


class DiscreteBodyDecoder(nn.Module):
    """Decode shifted body sequence, predicting cell indices."""

    def __init__(self, num_cells: int = NUM_CELLS, embed_dim: int = EMBED_DIM, hidden_dim: int = BODY_HIDDEN):
        super().__init__()
        self.num_cells = num_cells
        self.embed = nn.Embedding(num_cells, embed_dim)
        self.segment_proj = nn.Linear(embed_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, num_cells)

    def forward(self, next_head_idx, old_body_indices, body_mask):
        """Decode shifted body.

        Args:
            next_head_idx: (B,) integer cell index of new head
            old_body_indices: (B, L) old body cell indices
            body_mask: (B, L) binary mask

        Returns:
            body_logits: (B, L, num_cells) logits for each body segment
        """
        B, L = old_body_indices.shape
        h = self.segment_proj(self.embed(next_head_idx))  # (B, hidden)

        outputs = []
        # Position 0: new head
        h = self.gru(h, h)
        outputs.append(self.output_proj(h))  # (B, num_cells)

        # Remaining positions
        for i in range(1, L):
            seg_emb = F.relu(self.segment_proj(self.embed(old_body_indices[:, i - 1])))  # (B, hidden)
            h = self.gru(seg_emb, h)
            outputs.append(self.output_proj(h))

        body_logits = torch.stack(outputs, dim=1)  # (B, L, num_cells)
        return body_logits


class DiscreteWorldModel(nn.Module):
    """Discrete coordinate-based world model for Snake.

    All spatial outputs are grid cell indices (0-99), predicted via
    cross-entropy classification. No numerical drift in closed-loop playback.
    """

    def __init__(self):
        super().__init__()

        # Cell index embeddings
        self.head_embed = nn.Embedding(NUM_CELLS, EMBED_DIM)
        self.food_embed = nn.Embedding(NUM_CELLS, EMBED_DIM)

        # Body encoder/decoder
        self.body_encoder = DiscreteBodyEncoder()
        self.body_decoder = DiscreteBodyDecoder()

        # State MLP
        state_in_dim = EMBED_DIM + EMBED_DIM + 4 + 3 + 1 + BODY_HIDDEN  # 16+16+4+3+1+32 = 72
        self.fc1 = nn.Linear(state_in_dim, STATE_HIDDEN)
        self.fc2 = nn.Linear(STATE_HIDDEN, STATE_HIDDEN)
        self.fc3 = nn.Linear(STATE_HIDDEN, STATE_HIDDEN)

        # Output heads — classification over grid cells
        self.head_classifier = nn.Linear(STATE_HIDDEN, NUM_CELLS)
        self.food_classifier = nn.Linear(STATE_HIDDEN, NUM_CELLS)
        self.ate_head = nn.Linear(STATE_HIDDEN, 1)
        self.go_head = nn.Linear(STATE_HIDDEN, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)

    def forward(
        self,
        head_idx,          # (B,) integer cell index
        direction,         # (B, 4) one-hot
        action,            # (B, 3) one-hot
        food_idx,          # (B,) integer cell index
        game_over,         # (B, 1) binary
        body_indices,      # (B, L) integer cell indices
        body_mask,         # (B, L) binary
    ):
        B = head_idx.shape[0]

        # Embed
        h_emb = self.head_embed(head_idx)  # (B, 16)
        f_emb = self.food_embed(food_idx)   # (B, 16)

        # Body context
        body_ctx = self.body_encoder(body_indices, body_mask)  # (B, 32)

        # State concatenation
        state = torch.cat([h_emb, f_emb, direction, action, game_over, body_ctx], dim=1)  # (B, 72)

        # MLP
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Outputs
        head_logits = self.head_classifier(x)      # (B, 100)
        food_logits = self.food_classifier(x)       # (B, 100)
        ate_logits = self.ate_head(x)               # (B, 1)
        go_logits = self.go_head(x)                 # (B, 1)

        # Derive direction deterministically
        old_dir_idx = direction.argmax(dim=1)
        action_idx = action.argmax(dim=1)
        offsets = torch.tensor([-1, 0, 1], device=head_idx.device)
        new_dir_idx = (old_dir_idx + offsets[action_idx]) % 4
        dir_logits = torch.zeros(B, 4, device=head_idx.device)
        dir_logits[torch.arange(B), new_dir_idx] = 1.0

        # Decode body
        body_logits = self.body_decoder(head_logits.argmax(dim=1), body_indices, body_mask)  # (B, L, 100)

        return head_logits, food_logits, ate_logits, go_logits, dir_logits, body_logits

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


def compute_loss(
    head_logits, food_logits, ate_logits, go_logits, dir_logits, body_logits,
    target_head, target_food, target_ate, target_go, target_dir, target_body,
    body_mask,
    lambda_head=1.0, lambda_food=0.5, lambda_ate=0.5, lambda_go=0.5, lambda_body=1.0,
):
    """Multi-task loss for discrete model — cross-entropy for spatial, BCE for binary."""
    B = head_logits.shape[0]

    # Head: CE over 100 classes
    head_loss = F.cross_entropy(head_logits, target_head.long())

    # Food: CE over 100 classes
    food_loss = F.cross_entropy(food_logits, target_food.long())

    # Ate: BCE
    ate_loss = F.binary_cross_entropy_with_logits(ate_logits.view(-1), target_ate.view(-1).float())

    # Game over: BCE
    go_loss = F.binary_cross_entropy_with_logits(go_logits.view(-1), target_go.view(-1).float())

    # Direction: CE
    dir_loss = F.cross_entropy(dir_logits, target_dir.long())

    # Body: CE over 100 classes per segment, masked
    body_loss = F.cross_entropy(
        body_logits.reshape(-1, body_logits.shape[-1]),  # (B*L, 100)
        target_body.reshape(-1).long(),                  # (B*L,)
        reduction="none",
    ).reshape(B, -1)  # (B, L)
    body_loss = (body_loss * body_mask).sum() / body_mask.sum().clamp(min=1)

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
    head_logits, food_logits, ate_logits, go_logits, dir_logits, body_logits,
    target_head, target_food, target_ate, target_go, target_dir, target_body,
    body_mask,
):
    """Accuracy metrics for discrete model."""
    with torch.no_grad():
        head_pred = head_logits.argmax(dim=1)  # (B,)
        food_pred = food_logits.argmax(dim=1)  # (B,)
        body_pred = body_logits.argmax(dim=-1)  # (B, L)
        ate_pred = (ate_logits > 0).float()
        go_pred = (go_logits > 0).float()

        head_acc = (head_pred == target_head).float().mean().item()
        food_acc = (food_pred == target_food).float().mean().item()
        ate_acc = (ate_pred.view(-1) == target_ate.view(-1).float()).float().mean().item()
        go_acc = (go_pred.view(-1) == target_go.view(-1).float()).float().mean().item()

        # Body: per-segment accuracy, masked
        body_correct = (body_pred == target_body).float() * body_mask  # (B, L)
        body_acc = body_correct.sum() / body_mask.sum().clamp(min=1)
        body_acc = body_acc.item()

        # Full state match: head + body + ate + go all correct simultaneously
        body_full = ((body_pred == target_body).float() * body_mask).sum(dim=1) / body_mask.sum(dim=1).clamp(min=1)
        body_full = (body_full > 0.999).float()  # all valid segments correct
        full_match = ((head_pred == target_head)
                      & (ate_pred.view(-1) == target_ate.view(-1).float()).bool()
                      & (go_pred.view(-1) == target_go.view(-1).float()).bool()
                      & body_full.bool()).float().mean().item()

    return {
        "head_acc": head_acc,
        "food_acc": food_acc,
        "ate_acc": ate_acc,
        "go_acc": go_acc,
        "body_acc": body_acc,
        "full_match": full_match,
    }
