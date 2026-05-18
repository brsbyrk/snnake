"""Autoregressive playback for sNNake v2 — structured world model.

Runs the model in closed-loop: feeds its own predictions back as input.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .model_v2 import StructuredWorldModel
from .engine import SnakeEngine
from .encoding import DIRECTION_VECTORS, index_from_direction
from .collector_v2 import normalize_coord, denormalize_coord, body_to_array, MAX_BODY_LEN

GRID = 10
EMPTY, BODY, HEAD, FOOD = 0, 1, 2, 3


def render_grid_from_coords(head_xy, body, food_xy, body_mask=None, game_over: bool = False) -> str:
    """Render structured state as ASCII grid."""
    grid = np.full((GRID, GRID), EMPTY, dtype=np.int32)

    fx, fy = int(round(food_xy[0] * (GRID - 1))), int(round(food_xy[1] * (GRID - 1)))
    if 0 <= fx < GRID and 0 <= fy < GRID:
        grid[fy, fx] = FOOD

    if body_mask is None:
        body_mask = np.ones(len(body))
    for i in range(len(body)):
        if i >= len(body_mask) or body_mask[i] < 0.5:
            continue
        bx, by = body[i]
        gx, gy = int(round(bx * (GRID - 1))), int(round(by * (GRID - 1)))
        if 0 <= gx < GRID and 0 <= gy < GRID:
            grid[gy, gx] = HEAD if i == 0 else BODY

    symbols = {EMPTY: "·", BODY: "■", HEAD: "●", FOOD: "★"}
    lines = ["┌" + "─" * GRID + "┐"]
    for y in range(GRID):
        row = "│" + "".join(symbols.get(grid[y, x], "?") for x in range(GRID)) + "│"
        lines.append(row)
    lines.append("└" + "─" * GRID + "┘")
    return "\n".join(lines)


def play_v2(
    model: StructuredWorldModel,
    max_steps: int = 100,
    device: str = "cpu",
    render: bool = True,
    action_policy: str = "random",
):
    """Run structured model in closed-loop with ground truth comparison."""
    model.eval()
    device = torch.device(device)

    # Ground truth engine — we seed separately so it's independent of model predictions
    engine = SnakeEngine(seed=42)
    state = engine.reset()

    # Predicted state — start from same initial state
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

    while not pred_game_over and steps < max_steps:
        # Decide action
        if action_policy == "random":
            action_idx = np.random.randint(0, 3)
        elif action_policy == "straight":
            action_idx = 1
        else:
            action_idx = np.random.randint(0, 3)

        # Ground truth step (stop comparing if GT is over)
        gt_head = None
        gt_body = None
        gt_food = None
        gt_done = False
        if not state.game_over:
            _, _, next_state, reward, done = engine.step(action_idx)
            gt_head = next_state.body[0]
            gt_body = next_state.body
            gt_food = next_state.food_pos
            gt_done = done
            state = next_state

        # Model prediction
        head_t = torch.from_numpy(pred_head).unsqueeze(0).to(device)
        dir_t = torch.from_numpy(pred_dir).unsqueeze(0).to(device)
        act_t = torch.from_numpy(np.eye(3, dtype=np.float32)[action_idx]).unsqueeze(0).to(device)
        food_t = torch.from_numpy(pred_food).unsqueeze(0).to(device)
        go_t = torch.from_numpy(pred_go).unsqueeze(0).to(device)
        body_t = torch.from_numpy(pred_body_arr).unsqueeze(0).to(device)
        body_mask_t = torch.from_numpy(pred_body_mask).unsqueeze(0).to(device)

        with torch.no_grad():
            nh_pred, nf_pred, ate_logits, go_logits, nd_logits, nb_pred = model(
                head_t, dir_t, act_t, food_t, go_t, body_t, body_mask_t
            )

        # Decode prediction
        pred_next_head = nh_pred.squeeze(0).cpu().numpy()
        pred_next_food = nf_pred.squeeze(0).cpu().numpy()
        pred_ate = (ate_logits > 0).item()
        pred_next_go = (go_logits > 0.5).item()
        pred_next_body = nb_pred.squeeze(0).cpu().numpy()
        pred_next_dir_idx = nd_logits.argmax(dim=1).item()

        # Compare with ground truth
        head_xy = (int(round(pred_next_head[0] * (GRID - 1))),
                   int(round(pred_next_head[1] * (GRID - 1))))
        step_label = "✓"
        match = False
        if gt_head is not None:
            match = (head_xy == gt_head)
            if match:
                correct_steps += 1
                step_label = "✓"
            else:
                step_label = "✗"

        if render:
            print(f"\n--- Step {steps} | Action: {['←','↑','→'][action_idx]} | {step_label} ---")
            print(f"Pred GO: {torch.sigmoid(go_logits).item():.3f} | Ate: {pred_ate}")
            pred_grid = render_grid_from_coords(
                pred_next_head, pred_next_body, pred_next_food,
                body_mask=pred_body_mask
            )
            print(pred_grid)

        # Feed prediction back as input for next step
        pred_head = pred_next_head
        pred_food = pred_next_food
        pred_dir = np.eye(4, dtype=np.float32)[pred_next_dir_idx]
        pred_go = np.array([1.0 if pred_next_go else 0.0], dtype=np.float32)
        pred_body_arr = pred_next_body
        pred_game_over = pred_next_go

        steps += 1
        if pred_game_over:
            print(f"\nModel predicted game over after {steps} steps")

    print(f"\n=== Autoregressive Results ===")
    head_acc = 100.0 * correct_steps / steps if steps > 0 else 0
    print(f"Steps: {steps} | Head accuracy: {correct_steps}/{steps} ({head_acc:.1f}%)")
    return steps


def main():
    parser = argparse.ArgumentParser(description="Run sNNake v2 autoregressive playback")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_v2/best.pt")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--policy", type=str, default="random", choices=["random", "straight"])
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = StructuredWorldModel()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)
    print(f"Loaded epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f}) — {model.get_num_params():,} params")

    play_v2(model, max_steps=args.steps, device=args.device,
            render=not args.no_render, action_policy=args.policy)


if __name__ == "__main__":
    main()
