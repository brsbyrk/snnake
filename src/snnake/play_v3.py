"""Autoregressive playback for sNNake v3 — closed-loop stability test.

Runs the model in closed-loop: feeds own predictions back as input.
Measures how many steps before the model diverges from ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model_v3 import ScaledWorldModel, ModelConfig
from .engine import SnakeEngine
from .encoding import DIRECTION_VECTORS
from .collector_v3 import normalize_coord, body_to_array, heuristic_action, MAX_BODY_LEN

GRID = 10
EMPTY, BODY, HEAD, FOOD = 0, 1, 2, 3


def render_grid(head_norm, body_arr, food_norm, body_mask, game_over: bool) -> str:
    """Render structured state as ASCII grid."""
    grid = np.full((GRID, GRID), EMPTY, dtype=np.int32)

    fx = int(round(food_norm[0] * (GRID - 1)))
    fy = int(round(food_norm[1] * (GRID - 1)))
    if 0 <= fx < GRID and 0 <= fy < GRID:
        grid[fy, fx] = FOOD

    for i in range(len(body_arr)):
        if i >= len(body_mask) or body_mask[i] < 0.5:
            continue
        bx, by = body_arr[i], body_arr[i + 1] if len(body_arr.shape) > 1 else 0
        # body_arr is (L, 2) — handle properly
        bx_val = float(body_arr[i, 0]) if body_arr.ndim == 2 else float(body_arr[i])  # legacy
        by_val = float(body_arr[i, 1]) if body_arr.ndim == 2 else 0
        
        gx = int(round(bx_val * (GRID - 1)))
        gy = int(round(by_val * (GRID - 1)))
        if 0 <= gx < GRID and 0 <= gy < GRID:
            grid[gy, gx] = HEAD if i == 0 else BODY

    symbols = {EMPTY: "·", BODY: "■", HEAD: "●", FOOD: "★"}
    lines = ["┌" + "─" * GRID + "┐"]
    for y in range(GRID):
        row = "│" + "".join(symbols.get(grid[y, x], "?") for x in range(GRID)) + "│"
        lines.append(row)
    lines.append("└" + "─" * GRID + "┘")
    return "\n".join(lines)


@torch.no_grad()
def autoregressive_test(
    model: ScaledWorldModel,
    num_trials: int = 50,
    max_steps: int = 1000,
    policy: str = "heuristic",
    device: str = "cpu",
    render: bool = False,
    seed: int = 42,
) -> dict:
    """Run autoregressive stability test.

    Args:
        model: Trained world model
        num_trials: Number of episodes to test
        max_steps: Max steps per episode
        policy: "random", "heuristic", or "straight"
        device: Device to run on
        render: Whether to render each step

    Returns:
        dict with stats
    """
    model.eval()
    model.to(device)

    rng = np.random.RandomState(seed)
    engine = SnakeEngine(seed=seed)

    trial_results = []

    for trial in range(num_trials):
        # Fresh engine and model state
        state = engine.reset()

        # Predicted state — start same as ground truth
        hx, hy = state.body[0]
        pred_head = normalize_coord(hx, hy)
        fx, fy = state.food_pos
        pred_food = normalize_coord(fx, fy)
        pred_dir = np.eye(4, dtype=np.float32)[state.direction_idx]
        pred_go = np.array([0.0], dtype=np.float32)
        pred_body_arr, pred_body_mask = body_to_array(state.body)

        steps = 0
        correct_steps = 0
        pred_game_over = False
        episode_done = False
        head_match = True

        while not pred_game_over and not episode_done and steps < max_steps:
            # Decide action
            if policy == "random":
                action_idx = int(rng.randint(0, 3))
            elif policy == "heuristic":
                # Use ground truth head/food position for action decision
                hx_g, hy_g = state.body[0] if not state.game_over else (pred_head[0], pred_head[1])
                fx_g, fy_g = state.food_pos if not state.game_over else (pred_food[0], pred_food[1])
                action_idx = heuristic_action(
                    int(round(hx_g)), int(round(hy_g)),
                    state.direction_idx if not state.game_over else 1,
                    int(round(fx_g)), int(round(fy_g)),
                    state.body if not state.game_over else [(0, 0)],
                    rng,
                    exploration_prob=0.05,
                )
            else:
                action_idx = 1  # straight

            # Ground truth step
            gt_valid = False
            gt_head = None
            gt_body = None
            if not state.game_over:
                _, _, next_state, reward, done = engine.step(action_idx)
                gt_head = next_state.body[0]
                gt_body = next_state.body
                gt_food = next_state.food_pos
                gt_done = done
                state = next_state
                gt_valid = True

            # Model prediction
            head_t = torch.from_numpy(pred_head).unsqueeze(0).to(device)
            dir_t = torch.from_numpy(pred_dir).unsqueeze(0).to(device)
            act_t = torch.from_numpy(np.eye(3, dtype=np.float32)[action_idx]).unsqueeze(0).to(device)
            food_t = torch.from_numpy(pred_food).unsqueeze(0).to(device)
            go_t = torch.from_numpy(pred_go).unsqueeze(0).to(device)
            body_t = torch.from_numpy(pred_body_arr).unsqueeze(0).to(device)
            body_mask_t = torch.from_numpy(pred_body_mask).unsqueeze(0).to(device)

            out = model(head_t, dir_t, act_t, food_t, go_t, body_t, body_mask_t)

            # Decode predictions
            pred_next_head = out["next_head"].squeeze(0).cpu().numpy()
            pred_next_food = out["next_food"].squeeze(0).cpu().numpy()
            pred_ate = (out["ate_logits"] > 0).item()
            pred_next_go = (out["go_logits"] > 0.5).item()
            pred_next_body = out["new_body"].squeeze(0).cpu().numpy()

            # Check if head matches ground truth
            if gt_valid and gt_head is not None:
                pred_grid_head = (
                    int(round(pred_next_head[0] * (GRID - 1))),
                    int(round(pred_next_head[1] * (GRID - 1))),
                )
                if pred_grid_head == gt_head:
                    correct_steps += 1
                else:
                    head_match = False  # first divergence

            if render:
                print(f"\n--- Step {steps} | Action: {['←','↑','→'][action_idx]} ---")
                print(f"Pred GO: {torch.sigmoid(out['go_logits']).item():.3f} | Ate: {pred_ate}")
                grid_str = render_grid(pred_next_head, pred_next_body, pred_next_food,
                                       pred_body_mask, pred_next_go)
                print(grid_str)
                if gt_head is not None:
                    print(f"  GT head: {gt_head} | Pred head: {pred_grid_head}")

            # Feed prediction back as input
            pred_head = pred_next_head
            pred_food = pred_next_food
            
            # Determine direction from action (same as model forward does)
            old_dir_idx = np.argmax(pred_dir)
            offsets = [-1, 0, 1]
            new_dir_idx = (old_dir_idx + offsets[action_idx]) % 4
            pred_dir = np.eye(4, dtype=np.float32)[new_dir_idx]
            pred_go = np.array([1.0 if pred_next_go else 0.0], dtype=np.float32)
            pred_body_arr = pred_next_body
            pred_game_over = pred_next_go

            steps += 1

            # Also stop if GT says game over (no point continuing)
            if gt_done:
                episode_done = True

        trial_results.append({
            "trial": trial,
            "steps": steps,
            "correct_steps": correct_steps,
            "head_accuracy": 100.0 * correct_steps / max(steps, 1),
            "diverged_at": steps if head_match else correct_steps,
        })

    # Aggregate
    steps_list = [r["steps"] for r in trial_results]
    acc_list = [r["head_accuracy"] for r in trial_results]
    div_list = [r["correct_steps"] for r in trial_results]
    early_div = [r["correct_steps"] for r in trial_results if r["correct_steps"] < r["steps"]]

    stats = {
        "num_trials": num_trials,
        "mean_steps": float(np.mean(steps_list)),
        "median_steps": float(np.median(steps_list)),
        "max_steps": int(np.max(steps_list)),
        "min_steps": int(np.min(steps_list)),
        "std_steps": float(np.std(steps_list)),
        "mean_head_accuracy": float(np.mean(acc_list)),
        "mean_correct_steps": float(np.mean(div_list)),
        "median_correct_steps": float(np.median(div_list)),
        "max_correct_steps": int(np.max(div_list)),
        "min_correct_steps": int(np.min(div_list)),
        "early_divergence_rate": len(early_div) / max(len(trial_results), 1),
        "policy": policy,
    }

    print("\n" + "=" * 50)
    print("AUTOREGRESSIVE TEST RESULTS")
    print("=" * 50)
    print(f"Policy: {policy} | Trials: {num_trials}")
    print(f"Steps: mean={stats['mean_steps']:.1f} median={stats['median_steps']:.0f} "
          f"max={stats['max_steps']} min={stats['min_steps']}")
    print(f"Correct steps: mean={stats['mean_correct_steps']:.1f} "
          f"max={stats['max_correct_steps']} min={stats['min_correct_steps']}")
    print(f"Head accuracy: {stats['mean_head_accuracy']:.1f}%")
    print(f"Early divergence rate: {stats['early_divergence_rate']:.2f}")
    print(f"{'✓ PLAYABLE' if stats['mean_correct_steps'] >= 1000 else '✗ Below 1000 steps'} "
          f"(target: 1000+)")
    print("=" * 50)

    return stats


def main():
    parser = argparse.ArgumentParser(description="sNNake v3 autoregressive playback")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_v3/best.pt")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--policy", type=str, default="heuristic",
                        choices=["random", "heuristic", "straight"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    
    # Handle dataclass config stored as dict in older checkpoints
    if "config" in ckpt:
        config = ckpt["config"]
        if isinstance(config, dict):
            config = ModelConfig(**config)
    else:
        config = ModelConfig()
    
    model = ScaledWorldModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded epoch {ckpt.get('epoch', '?')} (val_loss={ckpt.get('val_loss', '?'):.4f}) "
          f"— {model.get_num_params():,} params")

    stats = autoregressive_test(
        model,
        num_trials=args.trials,
        max_steps=args.steps,
        policy=args.policy,
        device=args.device,
        render=args.render,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
