"""Grid encoding/decoding for sNNake.

Encoding scheme:
  4-channel one-hot per cell:
    Channel 0: empty
    Channel 1: body
    Channel 2: head
    Channel 3: food

  Direction: 4-element one-hot [up, right, down, left]

Decoding:
  Grid → argmax over channels → integer grid (0=empty, 1=body, 2=head, 3=food)
"""

from __future__ import annotations

import numpy as np
import torch

# Cell type constants
EMPTY = 0
BODY = 1
HEAD = 2
FOOD = 3

# Grid dimensions
GRID_SIZE = 10
NUM_CELL_TYPES = 4
NUM_DIRECTIONS = 4
NUM_ACTIONS = 3

# Direction mappings
# index → (dx, dy) and reverse
DIRECTION_VECTORS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
DIRECTION_FROM_VECTOR = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3}

# Action mappings
ACTION_NAMES = ["left", "straight", "right"]

# Relative turn: given current direction index and action index, return new direction index
# direction_idx × action_idx → new_direction_idx
# left = -1 mod 4, straight = 0, right = +1 mod 4
def apply_action(direction_idx: int, action_idx: int) -> int:
    """Apply a relative action to a direction index.

    action_idx: 0=left, 1=straight, 2=right
    Returns new direction index (0-3).
    """
    offsets = [-1, 0, 1]
    return (direction_idx + offsets[action_idx]) % 4


def encode_state(
    grid: np.ndarray,        # (10, 10) int array with values 0-3
    direction: tuple | list | np.ndarray,  # (dx, dy)
    game_over: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode game state into network-friendly tensors.

    Returns:
        grid_tensor: (4, 10, 10) one-hot encoded
        dir_tensor: (4,) one-hot
        go_tensor: (1,) binary
    """
    # One-hot encode grid: (10, 10) → (4, 10, 10)
    grid_tensor = np.zeros((NUM_CELL_TYPES, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for c in range(NUM_CELL_TYPES):
        grid_tensor[c] = (grid == c).astype(np.float32)

    # One-hot direction
    dir_idx = DIRECTION_FROM_VECTOR[tuple(direction)]
    dir_tensor = np.zeros(NUM_DIRECTIONS, dtype=np.float32)
    dir_tensor[dir_idx] = 1.0

    # Game over
    go_tensor = np.array([1.0 if game_over else 0.0], dtype=np.float32)

    return grid_tensor, dir_tensor, go_tensor


def encode_action(action_idx: int) -> np.ndarray:
    """Encode action as one-hot vector."""
    a = np.zeros(NUM_ACTIONS, dtype=np.float32)
    a[action_idx] = 1.0
    return a


def decode_grid(logits: torch.Tensor | np.ndarray) -> np.ndarray:
    """Decode grid logits (4, 10, 10) into integer grid (10, 10)."""
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    return np.argmax(logits, axis=0).astype(np.int32)


def decode_direction(logits: torch.Tensor | np.ndarray) -> int:
    """Decode direction logits (4,) into direction index (0-3)."""
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    return int(np.argmax(logits))


def state_to_grid(
    grid_tensor: np.ndarray | torch.Tensor,  # (4, 10, 10) or (1, 4, 10, 10)
) -> np.ndarray:
    """Convert one-hot grid tensor back to integer grid (10, 10)."""
    if isinstance(grid_tensor, torch.Tensor):
        grid_tensor = grid_tensor.detach().cpu().numpy()
    if grid_tensor.ndim == 4:
        grid_tensor = grid_tensor[0]  # remove batch dim
    return decode_grid(grid_tensor)


def get_available_actions(direction_idx: int) -> list[int]:
    """Get list of valid action indices (all 3 are always valid in relative encoding)."""
    return [0, 1, 2]


def random_action() -> int:
    """Sample a random action index."""
    return int(np.random.randint(0, NUM_ACTIONS))


def direction_from_index(idx: int) -> tuple[int, int]:
    """Get (dx, dy) from direction index."""
    return DIRECTION_VECTORS[idx]


def index_from_direction(dx: int, dy: int) -> int:
    """Get direction index from (dx, dy) vector."""
    return DIRECTION_FROM_VECTOR[(dx, dy)]
