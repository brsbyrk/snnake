"""Data collection pipeline for sNNake.

Generates training data by running many Snake games with random actions.
Saves (state, action, next_state, done) tuples as compressed numpy arrays.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .engine import SnakeEngine
from .encoding import encode_state, encode_action


def collect_data(
    num_episodes: int = 50000,
    max_steps_per_episode: int = 200,
    action_change_prob: float = 0.1,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """Run episodes and collect training data.

    Uses random actions with occasional direction changes to maximize
    state space coverage.

    Args:
        num_episodes: Number of episodes to run.
        max_steps_per_episode: Max steps before truncating an episode.
        action_change_prob: Probability of changing direction each step.
        seed: Random seed for reproducibility.

    Returns:
        dict with numpy arrays:
            grid:         (N, 4, 10, 10) — current grid
            direction:    (N, 4) — current direction one-hot
            action:       (N, 3) — action taken one-hot
            game_over:    (N, 1) — current game_over flag
            next_grid:    (N, 4, 10, 10) — grid after action
            next_dir:     (N, 4) — direction after action
            next_go:      (N, 1) — game_over after action
            done:         (N,) — whether episode ended (final step)
            reward:       (N,) — reward for each step
            episode_id:   (N,) — which episode each transition belongs to
    """
    rng = np.random.RandomState(seed)
    engine = SnakeEngine(seed=seed)

    # Pre-allocate lists (will concat at end)
    grid_list: list[np.ndarray] = []
    dir_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    go_list: list[np.ndarray] = []
    next_grid_list: list[np.ndarray] = []
    next_dir_list: list[np.ndarray] = []
    next_go_list: list[np.ndarray] = []
    done_list: list[bool] = []
    reward_list: list[float] = []
    episode_list: list[int] = []

    iterator = range(num_episodes)
    if verbose:
        iterator = tqdm(iterator, desc="Collecting data")

    for ep in iterator:
        state = engine.reset()

        for step in range(max_steps_per_episode):
            # Sample action: mostly straight, occasional turns
            if rng.random() < action_change_prob:
                action_idx = int(rng.randint(0, 3))
            else:
                action_idx = 1  # straight

            # Encode current state
            grid_t, dir_t, go_t = encode_state(state.grid, engine._dir_vector(state), state.game_over)
            act_t = encode_action(action_idx)

            # Step
            prev_state, _, next_state, reward, done = engine.step(action_idx)

            # Encode next state
            next_grid_t, next_dir_t, next_go_t = encode_state(
                next_state.grid, engine._dir_vector(next_state), next_state.game_over
            )

            # Store
            grid_list.append(grid_t)
            dir_list.append(dir_t)
            act_list.append(act_t)
            go_list.append(go_t)
            next_grid_list.append(next_grid_t)
            next_dir_list.append(next_dir_t)
            next_go_list.append(next_go_t)
            done_list.append(done)
            reward_list.append(reward)
            episode_list.append(ep)

            if done:
                break

    if verbose:
        tqdm.write(f"Collected {len(grid_list):,} transitions from {num_episodes:,} episodes")

    return {
        "grid": np.array(grid_list, dtype=np.float32),
        "direction": np.array(dir_list, dtype=np.float32),
        "action": np.array(act_list, dtype=np.float32),
        "game_over": np.array(go_list, dtype=np.float32),
        "next_grid": np.array(next_grid_list, dtype=np.float32),
        "next_direction": np.array(next_dir_list, dtype=np.float32),
        "next_game_over": np.array(next_go_list, dtype=np.float32),
        "done": np.array(done_list, dtype=bool),
        "reward": np.array(reward_list, dtype=np.float32),
        "episode_id": np.array(episode_list, dtype=np.int32),
    }



def save_data(data: dict, path: str | Path):
    """Save collected data as compressed numpy archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    print(f"Saved {len(data['grid']):,} transitions to {path} ({path.stat().st_size / 1e6:.1f} MB)")


def load_data(path: str | Path) -> dict:
    """Load collected data from numpy archive."""
    path = Path(path)
    data = np.load(path)
    print(f"Loaded {len(data['grid']):,} transitions from {path}")
    return dict(data)


def main():
    parser = argparse.ArgumentParser(description="Collect Snake training data")
    parser.add_argument("--episodes", type=int, default=50000, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--action-change-prob", type=float, default=0.1, help="Action change probability")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="data/games.npz", help="Output path")
    args = parser.parse_args()

    data = collect_data(
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        action_change_prob=args.action_change_prob,
        seed=args.seed,
    )
    save_data(data, args.output)


if __name__ == "__main__":
    main()
