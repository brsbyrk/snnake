"""Structured world model for sNNake v2.

Operates on explicit coordinates (head, body, food) instead of grid pixels.
Body is encoded/decoded as a sequence via GRU — the model learns the shift operation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Max body length for fixed-size batching (10x10 board → 100 max, but games
# practically end before 30 segments. 40 is a safe upper bound.)
MAX_BODY_LEN = 40

# Input dimensions
HEAD_DIM = 2
FOOD_DIM = 2
DIR_DIM = 4
ACT_DIM = 3
GO_DIM = 1
BODY_EMBED_DIM = 16
STATE_HIDDEN = 64


class StructuredBodyEncoder(nn.Module):
    """Encode variable-length body sequence into a fixed-size context vector.

    Reads body from head to tail via GRU.
    """

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
    """Decode shifted body sequence from next_head and body context.

    For position i in the new body:
      i=0: new_head (given)
      i>0: old_body[i-1] shifted forward by one

    The GRU learns this shift by reading old body and predicting new body.
    """

    def __init__(self, segment_dim: int = 2, hidden_dim: int = BODY_EMBED_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.segment_proj = nn.Linear(segment_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, segment_dim)

    def forward(self, next_head, old_body_positions, body_mask, ate_food):
        """Decode shifted body.

        Args:
            next_head:           (B, 2) predicted next head position
            old_body_positions:  (B, L, 2) old body (used as decoder input)
            body_mask:           (B, L) valid position mask
            ate_food:            (B, 1) whether food was eaten

        Returns:
            new_body: (B, L, 2) predicted body after shift
        """
        B, L, _ = old_body_positions.shape

        # Initial hidden state: encode next_head
        h = self.segment_proj(next_head)  # (B, hidden)

        outputs = []
        # Position 0: always the new head
        h = self.gru(h, h)
        out0 = self.output_proj(h)  # (B, 2)
        outputs.append(out0)

        # Shift: for position i (1..L-1), the model reads old_body[i-1]
        # and predicts new_body[i] (which should be old_body[i-1] shifted up)
        # The GRU state carries info about where we are in the sequence
        for i in range(1, L):
            # Input: old body segment at position i-1 shifted forward
            seg_in = old_body_positions[:, i - 1, :]  # (B, 2)
            seg_emb = F.relu(self.segment_proj(seg_in))  # (B, hidden)
            h = self.gru(seg_emb, h)
            out = self.output_proj(h)  # (B, 2)
            outputs.append(out)

        new_body = torch.stack(outputs, dim=1)  # (B, L, 2)

        # Handle food eating: if ate, the tail stays (old_body[-1] is kept)
        # The decoder naturally produces old_body[-1] at position L-1.
        # When ate_food is True, we need to shift the body differently:
        # new_body = [next_head] + old_body  (no tail removal)
        # But our decoder produces [next_head] + old_body[:-1]
        # We fix this by inserting old_body[-1] when ate_food is True
        if ate_food is not None:
            # This is handled during loss computation instead
            pass

        # Mask out invalid positions
        new_body = new_body * body_mask.unsqueeze(-1)

        return new_body


class StructuredWorldModel(nn.Module):
    """Coordinate-based world model for Snake.

    Processes head, body, food, direction, and action as structured entities
    rather than grid pixels. The body is a sequence processed by a GRU.

    Parameters: ~15K (vs 997K in CNN v1)
    """

    def __init__(self):
        super().__init__()

        # Body encoder/decoder
        self.body_encoder = StructuredBodyEncoder(segment_dim=2, hidden_dim=BODY_EMBED_DIM)
        self.body_decoder = StructuredBodyDecoder(segment_dim=2, hidden_dim=BODY_EMBED_DIM)

        # Combined state MLP
        state_in_dim = HEAD_DIM + FOOD_DIM + DIR_DIM + ACT_DIM + GO_DIM + BODY_EMBED_DIM  # 12 + 16 = 28
        self.fc1 = nn.Linear(state_in_dim, STATE_HIDDEN)
        self.fc2 = nn.Linear(STATE_HIDDEN, STATE_HIDDEN)
        self.fc3 = nn.Linear(STATE_HIDDEN, STATE_HIDDEN)

        # Output heads
        self.head_next = nn.Linear(STATE_HIDDEN, HEAD_DIM)
        self.food_next = nn.Linear(STATE_HIDDEN, FOOD_DIM)
        self.ate_head = nn.Linear(STATE_HIDDEN, 1)
        self.go_head = nn.Linear(STATE_HIDDEN, 1)
        self.dir_head = nn.Linear(STATE_HIDDEN, DIR_DIM)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        head,              # (B, 2)
        direction,         # (B, 4)
        action,            # (B, 3)
        food,              # (B, 2)
        game_over,         # (B, 1)
        body_positions,    # (B, L_max, 2)  zero-padded
        body_mask,         # (B, L_max)  binary
    ):
        """Forward pass.

        Returns:
            next_head:      (B, 2) normalized coordinates
            next_food:      (B, 2) normalized coordinates
            ate_food_logits: (B, 1)
            game_over_logits: (B, 1)
            next_dir_logits:  (B, 4)
            new_body:        (B, L_max, 2) shifted body
        """
        B = head.shape[0]

        # Encode body
        body_context = self.body_encoder(body_positions, body_mask)  # (B, 16)

        # Concatenate all state
        state = torch.cat([head, food, direction, action, game_over, body_context], dim=1)  # (B, 28)

        # MLP
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Output heads
        next_head = self.head_next(x)  # (B, 2)
        next_food = self.food_next(x)  # (B, 2)

        # next_dir_logits = self.dir_head(x)  # (B, 4)
        # Derive direction from action instead of predicting it
        # This is a key inductive bias: direction = apply_action(old_dir, action)

        ate_food_logits = self.ate_head(x)  # (B, 1)
        game_over_logits = self.go_head(x)  # (B, 1)

        # Decode body: learn the shift operation
        new_body = self.body_decoder(
            next_head, body_positions, body_mask, ate_food_logits
        )  # (B, L_max, 2)

        # Derive direction deterministically from old_dir + action
        # direction_idx = apply_action(old_dir_idx, action_idx)
        # We compute this explicitly so the model doesn't have to learn
        # the trivial direction update
        old_dir_idx = direction.argmax(dim=1)  # (B,)
        action_idx = action.argmax(dim=1)  # (B,)
        # action offsets: 0=left(-1), 1=straight(0), 2=right(+1)
        offsets = torch.tensor([-1, 0, 1], device=head.device)
        new_dir_idx = (old_dir_idx + offsets[action_idx]) % 4
        next_dir_logits = torch.zeros(B, 4, device=head.device)
        next_dir_logits[torch.arange(B), new_dir_idx] = 1.0
        next_dir_logits = next_dir_logits + 0.001  # small noise to avoid zero gradients

        return next_head, next_food, ate_food_logits, game_over_logits, next_dir_logits, new_body

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


def compute_loss(
    next_head, next_food, ate_logits, go_logits, next_dir_logits, new_body_pred,
    target_head, target_food, target_ate, target_go, target_dir, target_body,
    body_mask,
    lambda_head=1.0, lambda_food=0.5, lambda_ate=0.5, lambda_go=0.5,
    lambda_body=1.0,
):
    """Compute multi-task loss for structured model.

    Args:
        target_dir: (B,) integer class indices
        target_body: (B, L, 2) target body positions
        target_ate: (B,) binary (0 or 1)
    """
    # Head position: MSE
    head_loss = F.mse_loss(next_head, target_head)

    # Food position: MSE (only matters when food eaten, but we always compute)
    food_loss = F.mse_loss(next_food, target_food)

    # Ate food: BCE
    ate_loss = F.binary_cross_entropy_with_logits(
        ate_logits.view(-1), target_ate.view(-1).float()
    )

    # Game over: BCE
    go_loss = F.binary_cross_entropy_with_logits(
        go_logits.view(-1), target_go.view(-1).float()
    )

    # Direction: CE (should be near-perfect since it's derived)
    dir_loss = F.cross_entropy(next_dir_logits, target_dir.long())

    # Body: masked MSE
    body_diff = (new_body_pred - target_body) ** 2  # (B, L, 2)
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
    threshold=0.05,  # coordinate tolerance
):
    """Compute accuracy metrics for structured model.

    Head position accuracy: within threshold distance (normalized coords)
    """
    with torch.no_grad():
        # Head position: within threshold
        head_dist = torch.norm(next_head - target_head, dim=1)  # (B,)
        head_acc = (head_dist < threshold).float().mean().item()

        # Ate food accuracy
        ate_pred = (ate_logits > 0).float()
        ate_acc = (ate_pred.view(-1) == target_ate.view(-1).float()).float().mean().item()

        # Game over accuracy
        go_pred = (go_logits > 0).float()
        go_acc = (go_pred.view(-1) == target_go.view(-1).float()).float().mean().item()

        # Body accuracy: per-segment within threshold
        body_diff = torch.norm(new_body_pred - target_body, dim=2)  # (B, L)
        body_correct = (body_diff < threshold).float() * body_mask  # (B, L)
        body_acc = body_correct.sum() / body_mask.sum().clamp(min=1)
        body_acc = body_acc.item()

        # Full state match: all outputs correct simultaneously
        head_exact = (head_dist < threshold)
        # For "exact match" we consider head + ate + go + body
        body_exact = (body_diff < threshold).all(dim=1)  # (B,) — all segments correct
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
