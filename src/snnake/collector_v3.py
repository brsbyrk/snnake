"""Data collection for sNNake v3 — heuristic agent for long rollouts.

Generates training data with a food-seeking heuristic AI that produces
long games (hundreds of steps) instead of ~20 steps from random policy.

Also mixes in random episodes for diversity and edge-case coverage.
Saves episode boundaries so train_v3 can do multi-step unrolling.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .engine import SnakeEngine
from .encoding import (
    DIRECTION_VECTORS, GRID_SIZE,
    apply_action, direction_from_index,
)

GRID = GRID_SIZE
MAX_BODY_LEN = 40


# --- Coordinate utilities ---

def normalize_coord(x: int, y: int) -> np.ndarray:
    return np.array([x / (GRID - 1), y / (GRID - 1)], dtype=np.float32)


def body_to_array(body, max_len: int = MAX_BODY_LEN):
    L = len(body)
    arr = np.zeros((max_len, 2), dtype=np.float32)
    mask = np.zeros(max_len, dtype=np.float32)
    for i, (x, y) in enumerate(body):
        if i >= max_len:
            break
        arr[i] = normalize_coord(x, y)
        mask[i] = 1.0
    return arr, mask


# --- Heuristic AI ---

def _safe_actions(head_x: int, head_y: int, direction_idx: int) -> list[int]:
    """Return action indices that don't cause immediate wall collision."""
    safe = []
    for a in range(3):
        new_dir_idx = apply_action(direction_idx, a)
        dx, dy = direction_from_index(new_dir_idx)
        nx, ny = head_x + dx, head_y + dy
        if 0 <= nx < GRID and 0 <= ny < GRID:
            safe.append(a)
    return safe


def _body_occupied(body: list) -> set:
    return set(body)


def heuristic_action(
    head_x: int, head_y: int,
    direction_idx: int,
    food_x: int, food_y: int,
    body: list,
    rng: np.random.RandomState,
    exploration_prob: float = 0.15,
) -> int:
    """Food-seeking heuristic with exploration.

    Strategy:
    1. Get safe actions (no wall collision)
    2. Among safe, score each by how much it reduces distance to food
    3. Add self-collision avoidance
    4. With exploration_prob, pick random safe action
    """
    safe = _safe_actions(head_x, head_y, direction_idx)
    if not safe:
        return 1  # straight — will collide, but that's a valid transition

    if rng.random() < exploration_prob:
        return int(rng.choice(safe))

    # Score each action
    best_action = safe[0]
    best_score = -float("inf")

    for a in safe:
        new_dir_idx = apply_action(direction_idx, a)
        dx, dy = direction_from_index(new_dir_idx)
        nx, ny = head_x + dx, head_y + dy

        # Distance to food (lower is better)
        dist_to_food = math.sqrt((nx - food_x) ** 2 + (ny - food_y) ** 2)
        food_score = -dist_to_food

        # Penalty for moving toward self-body
        self_collision_penalty = -5.0 if (nx, ny) in _body_occupied(body) else 0.0

        # Bonus for continuing in same direction (pacing stability)
        straight_bonus = 0.5 if a == 1 else 0.0

        score = food_score + self_collision_penalty + straight_bonus
        if score > best_score:
            best_score = score
            best_action = a

    return best_action


# --- Data collection ---

def collect_data_v3(
    num_episodes: int = 100000,
    max_steps_per_episode: int = 500,
    heuristic_prob: float = 0.7,
    exploration_prob: float = 0.15,
    max_body_len: int = MAX_BODY_LEN,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """Collect training data with mixed heuristic + random policies.

    Args:
        heuristic_prob: probability of using heuristic (vs random) policy
        exploration_prob: exploration noise in heuristic (0 = greedy, 1 = random)

    Returns dict with transitions + episode_start_indices.
    """
    rng = np.random.RandomState(seed)
    engine = SnakeEngine(seed=seed)

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

    episode_starts = []  # indices where episodes begin

    iterator = range(num_episodes)
    if verbose:
        iterator = tqdm(iterator, desc="Collecting v3 data")

    total_steps = 0
    episode_lengths = []

    for ep in iterator:
        state = engine.reset()
        episode_starts.append(total_steps)
        ep_steps = 0

        # Choose policy for this episode
        use_heuristic = rng.random() < heuristic_prob

        for step in range(max_steps_per_episode):
            hx, hy = state.body[0]
            fx, fy = state.food_pos
            dir_idx = state.direction_idx

            # Sample action
            if use_heuristic:
                action_idx = heuristic_action(
                    hx, hy, dir_idx, fx, fy, state.body, rng, exploration_prob
                )
            else:
                # Random policy with occasional direction changes
                if rng.random() < 0.1:
                    action_idx = int(rng.randint(0, 3))
                else:
                    action_idx = 1  # straight

            head = normalize_coord(hx, hy)
            direction = np.eye(4, dtype=np.float32)[dir_idx]
            action = np.eye(3, dtype=np.float32)[action_idx]
            food = normalize_coord(fx, fy)
            game_over = np.array([1.0 if state.game_over else 0.0], dtype=np.float32)
            body_arr, body_mask_arr = body_to_array(state.body, max_len=max_body_len)

            # Step engine
            prev_state, _, next_state, reward, done = engine.step(action_idx)

            nhx, nhy = next_state.body[0]
            nfx, nfy = next_state.food_pos
            ndir_idx = next_state.direction_idx

            next_head = normalize_coord(nhx, nhy)
            next_food = normalize_coord(nfx, nfy)
            next_dir = np.eye(4, dtype=np.float32)[ndir_idx]
            next_go = np.array([1.0 if next_state.game_over else 0.0], dtype=np.float32)
            ate = np.array([1.0 if reward > 0 else 0.0], dtype=np.float32)
            next_body_arr, next_body_mask_arr = body_to_array(next_state.body, max_len=max_body_len)

            # Store
            head_list.append(head)
            dir_list.append(direction)
            act_list.append(action)
            food_list.append(food)
            go_list.append(game_over)
            body_list.append(body_arr)
            body_mask_list.append(body_mask_arr)

            next_head_list.append(next_head)
            next_food_list.append(next_food)
            next_dir_list.append(next_dir)
            next_go_list.append(next_go)
            ate_list.append(ate)
            next_body_list.append(next_body_arr)

            total_steps += 1
            ep_steps += 1

            if done:
                break

        episode_lengths.append(ep_steps)

    if verbose:
        avg_len = np.mean(episode_lengths)
        max_len = max(episode_lengths)
        tqdm.write(
            f"Collected {len(head_list):,} transitions from {num_episodes:,} episodes "
            f"(avg {avg_len:.1f}/ep, max {max_len})"
        )

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
        "episode_starts": np.array(episode_starts, dtype=np.int64),
        "episode_lengths": np.array(episode_lengths, dtype=np.int32),
    }


def save_data_v3(data: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    print(f"Saved {len(data['head']):,} transitions to {path} ({path.stat().st_size / 1e6:.1f} MB)")


def load_data_v3(path: str | Path) -> dict:
    path = Path(path)
    data = np.load(path)
    n = len(data["head"])
    print(f"Loaded {n:,} transitions from {path}")
    return dict(data)


def main():
    parser = argparse.ArgumentParser(description="Collect v3 structured Snake data with heuristic AI")
    parser.add_argument("--episodes", type=int, default=100000, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--heuristic-prob", type=float, default=0.7,
                        help="Probability of heuristic policy (vs random)")
    parser.add_argument("--exploration-prob", type=float, default=0.15,
                        help="Exploration noise in heuristic [0.0-1.0]")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="data/games_v3.npz", help="Output path")
    args = parser.parse_args()

    data = collect_data_v3(
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        heuristic_prob=args.heuristic_prob,
        exploration_prob=args.exploration_prob,
        seed=args.seed,
    )
    save_data_v3(data, args.output)


if __name__ == "__main__":
    main()
