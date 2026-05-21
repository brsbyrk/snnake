"""Data collection for sNNake v2 — structured coordinate format.

Each transition is:
  - head:            (2,) normalized [0, 1] coordinates
  - direction:       (4,) one-hot
  - action:          (3,) one-hot
  - food:            (2,) normalized [0, 1]
  - game_over:       (1,) binary
  - body_positions:  (L, 2) normalized coordinates, head-to-tail
  - body_mask:       (L,) binary mask for valid segments

  - next_head:       (2,) 
  - next_food:       (2,)
  - next_direction:  (4,) 
  - next_game_over:  (1,)
  - ate_food:        (1,) binary
  - next_body:       (L, 2)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .engine import SnakeEngine
from .encoding import (
    DIRECTION_VECTORS, GRID_SIZE,
    encode_state, encode_action,
    index_from_direction,
    apply_action, direction_from_index,
)

GRID = GRID_SIZE
MAX_BODY_LEN = 40  # must match model_v2.py


def normalize_coord(x: int, y: int) -> np.ndarray:
    """Normalize grid coordinates to [0, 1]."""
    return np.array([x / (GRID - 1), y / (GRID - 1)], dtype=np.float32)


def denormalize_coord(arr: np.ndarray) -> tuple:
    """Convert normalized [0,1] coords back to grid coordinates."""
    x = int(round(arr[0] * (GRID - 1)))
    y = int(round(arr[1] * (GRID - 1)))
    return x, y


def body_to_array(body, max_len: int = MAX_BODY_LEN) -> tuple:
    """Convert body (list of (x,y)) to padded numpy array + mask."""
    L = len(body)
    arr = np.zeros((max_len, 2), dtype=np.float32)
    mask = np.zeros(max_len, dtype=np.float32)
    for i, (x, y) in enumerate(body):
        if i >= max_len:
            break
        arr[i] = normalize_coord(x, y)
        mask[i] = 1.0
    return arr, mask


def _safe_actions(head_x: int, head_y: int, direction_idx: int) -> list[int]:
    """Return action indices that don't cause immediate wall collision.

    An action is safe if the resulting head position (after direction
    update) is within bounds.
    """
    safe = []
    for a in range(3):
        new_dir_idx = apply_action(direction_idx, a)
        dx, dy = direction_from_index(new_dir_idx)
        nx, ny = head_x + dx, head_y + dy
        if 0 <= nx < GRID and 0 <= ny < GRID:
            safe.append(a)
    return safe


class HeuristicPlayer:
    """Greedy food-seeking bot with obstacle avoidance.

    Chooses actions that minimize Manhattan distance to food while
    avoiding wall and self collisions. Injects 20% random exploration
    for trajectory diversity.

    For body length >= 4, uses 2-step lookahead to avoid tail-chasing:
    penalizes actions that would trap the snake against its own body.
    """

    def __init__(self, rng: np.random.RandomState, grid_size: int = GRID):
        self.rng = rng
        self.grid_size = grid_size
        self.greedy_prob = 0.80

    def choose_action(self, state) -> int:
        """Return action_idx [0,1,2] that moves toward food."""
        head_x, head_y = state.body[0]
        food_x, food_y = state.food_pos
        dir_idx = state.direction_idx
        body_set = set(state.body)

        # Evaluate each action
        candidates = []  # (action_idx, distance_to_food)
        for a in range(3):
            new_dir = apply_action(dir_idx, a)
            dx, dy = direction_from_index(new_dir)
            nx, ny = head_x + dx, head_y + dy

            # Wall collision → skip
            if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
                continue

            # Immediate self collision → skip
            if (nx, ny) in body_set:
                continue

            # Distance to food
            dist = abs(nx - food_x) + abs(ny - food_y)

            # 2-step lookahead for tail chasing (body >= 4)
            if len(state.body) >= 4:
                # Check if moving here would create a trap
                next_dir2 = apply_action(new_dir, 1)  # assume straight next
                dx2, dy2 = direction_from_index(next_dir2)
                nx2, ny2 = nx + dx2, ny + dy2
                # Project what the body would look like after this move
                projected_body = [(nx, ny)] + state.body[:-1]
                if (nx2, ny2) in set(projected_body):
                    dist += 2  # penalty for setting up trap

            candidates.append((a, dist))

        if not candidates:
            return 1  # all actions deadly — straight into the wall

        # Sort by distance
        candidates.sort(key=lambda x: x[1])

        # Greedy vs exploration
        if self.rng.random() < self.greedy_prob:
            return candidates[0][0]
        else:
            safe_actions = [c[0] for c in candidates]
            return self.rng.choice(safe_actions)


def collect_data_v2(
    num_episodes: int = 50000,
    max_steps_per_episode: int = 200,
    action_change_prob: float = 0.1,
    edge_bias_prob: float = 0.0,
    edge_margin: float = 1.0,
    strategy: str = "random",
    max_body_len: int = MAX_BODY_LEN,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """Collect training data in structured coordinate format.

    When edge_bias_prob > 0, when the head is within edge_margin cells
    of a wall, bias action selection towards safe (non-collision) actions.
    This oversamples "head at edge but no collision" transitions and helps
    the model learn that edge proximity != automatic game over.

    strategy: "random" uses the original random policy.
              "heuristic" uses greedy food-seeking bot with exploration (80/20).
    """
    rng = np.random.RandomState(seed)
    engine = SnakeEngine(seed=seed)

    # Initialize player for heuristic strategy
    player = HeuristicPlayer(rng) if strategy == "heuristic" else None

    # Pre-allocate lists
    head_list = []
    dir_list = []
    act_list = []
    food_list = []
    go_list = []
    body_list = []
    body_mask_list = []

    next_head_list = []
    next_food_list = []
    next_dir_list = []
    next_go_list = []
    ate_list = []
    next_body_list = []
    next_body_mask_list = []

    iterator = range(num_episodes)
    if verbose:
        iterator = tqdm(iterator, desc="Collecting v2 data")

    for ep in iterator:
        state = engine.reset()

        for step in range(max_steps_per_episode):
            # Current state
            hx, hy = state.body[0]
            fx, fy = state.food_pos
            dir_idx = state.direction_idx
            # Sample action
            if strategy == "heuristic" and player is not None:
                action_idx = player.choose_action(state)
            else:
                # Random strategy with edge bias
                hx, hy = state.body[0]
                is_near_wall = (hx < edge_margin or hx >= GRID - edge_margin
                               or hy < edge_margin or hy >= GRID - edge_margin)
                if is_near_wall and edge_bias_prob > 0 and rng.random() < edge_bias_prob:
                    safe = _safe_actions(hx, hy, dir_idx)
                    if safe:
                        action_idx = int(rng.choice(safe))
                    else:
                        action_idx = 1
                elif rng.random() < action_change_prob:
                    action_idx = int(rng.randint(0, 3))
                else:
                    action_idx = 1  # straight

            head = normalize_coord(hx, hy)
            direction = np.eye(4, dtype=np.float32)[dir_idx]
            action = np.eye(3, dtype=np.float32)[action_idx]
            food = normalize_coord(fx, fy)
            game_over = np.array([1.0 if state.game_over else 0.0], dtype=np.float32)
            body_arr, body_mask = body_to_array(state.body, max_len=max_body_len)

            # Step
            prev_state, _, next_state, reward, done = engine.step(action_idx)

            # Next state
            nhx, nhy = next_state.body[0]
            nfx, nfy = next_state.food_pos
            ndir_idx = next_state.direction_idx

            next_head = normalize_coord(nhx, nhy)
            next_food = normalize_coord(nfx, nfy)
            next_dir = np.eye(4, dtype=np.float32)[ndir_idx]
            next_go = np.array([1.0 if next_state.game_over else 0.0], dtype=np.float32)
            ate = np.array([1.0 if reward > 0 else 0.0], dtype=np.float32)
            next_body_arr, next_body_mask = body_to_array(next_state.body, max_len=max_body_len)

            # Store
            head_list.append(head)
            dir_list.append(direction)
            act_list.append(action)
            food_list.append(food)
            go_list.append(game_over)
            body_list.append(body_arr)
            body_mask_list.append(body_mask)

            next_head_list.append(next_head)
            next_food_list.append(next_food)
            next_dir_list.append(next_dir)
            next_go_list.append(next_go)
            ate_list.append(ate)
            next_body_list.append(next_body_arr)

            if done:
                break

    if verbose:
        tqdm.write(f"Collected {len(head_list):,} transitions from {num_episodes:,} episodes")

    return {
        "head": np.array(head_list, dtype=np.float32),
        "direction": np.array(dir_list, dtype=np.float32),
        "action": np.array(act_list, dtype=np.float32),
        "food": np.array(food_list, dtype=np.float32),
        "game_over": np.array(go_list, dtype=np.float32),
        "body": np.array(body_list, dtype=np.float32),
        "body_mask": np.array(body_mask_list, dtype=np.float32),
        "next_head": np.array(next_head_list, dtype=np.float32),
        "next_food": np.array(next_food_list, dtype=np.float32),
        "next_direction": np.array(next_dir_list, dtype=np.float32),
        "next_game_over": np.array(next_go_list, dtype=np.float32),
        "ate_food": np.array(ate_list, dtype=np.float32),
        "next_body": np.array(next_body_list, dtype=np.float32),
    }


def save_data_v2(data: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    print(f"Saved {len(data['head']):,} transitions to {path} ({path.stat().st_size / 1e6:.1f} MB)")


def load_data_v2(path: str | Path) -> dict:
    path = Path(path)
    data = np.load(path)
    print(f"Loaded {len(data['head']):,} transitions from {path}")
    return dict(data)


def main():
    parser = argparse.ArgumentParser(description="Collect v2 structured Snake data")
    parser.add_argument("--episodes", type=int, default=50000, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--action-change-prob", type=float, default=0.1, help="Action change probability")
    parser.add_argument("--edge-bias-prob", type=float, default=0.0,
                        help="Oversample safe actions near walls [0.0-1.0]")
    parser.add_argument("--edge-margin", type=float, default=1.0, help="Cells from wall to consider 'near edge'")
    parser.add_argument("--strategy", type=str, default="random", choices=["random", "heuristic"],
                        help="Policy for generating episodes: random or heuristic (greedy bot)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="data/games_v2.npz", help="Output path")
    args = parser.parse_args()

    data = collect_data_v2(
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        action_change_prob=args.action_change_prob,
        edge_bias_prob=args.edge_bias_prob,
        edge_margin=args.edge_margin,
        strategy=args.strategy,
        seed=args.seed,
    )
    save_data_v2(data, args.output)


if __name__ == "__main__":
    main()
