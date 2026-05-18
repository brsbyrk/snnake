"""Autoregressive playback: run the model as a closed-loop game engine.

Feed the model's own predictions back as input and watch it simulate Snake.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .engine import SnakeEngine
from .model import WorldModel
from .encoding import (
    encode_state, encode_action, decode_grid, decode_direction,
    apply_action, direction_from_index, index_from_direction,
    DIRECTION_VECTORS, GRID_SIZE, EMPTY, BODY, HEAD, FOOD,
)


def render_grid(grid: np.ndarray) -> str:
    """Render integer grid (10, 10) as ASCII art."""
    symbols = {EMPTY: "·", BODY: "■", HEAD: "●", FOOD: "★"}
    lines = []
    lines.append("┌" + "─" * GRID_SIZE + "┐")
    for y in range(GRID_SIZE):
        row = "│"
        for x in range(GRID_SIZE):
            cell = grid[y, x]
            row += symbols.get(cell, "?")
        row += "│"
        lines.append(row)
    lines.append("└" + "─" * GRID_SIZE + "┘")
    return "\n".join(lines)


def render_side_by_side(true_grid: np.ndarray, pred_grid: np.ndarray) -> str:
    """Render true vs predicted grid side by side."""
    symbols = {EMPTY: "·", BODY: "■", HEAD: "●", FOOD: "★"}
    lines = []
    header = f"{'Ground Truth':^{GRID_SIZE + 2}}    {'Predicted':^{GRID_SIZE + 2}}"
    lines.append(header)
    lines.append("┌" + "─" * GRID_SIZE + "┐" + "    " + "┌" + "─" * GRID_SIZE + "┐")
    for y in range(GRID_SIZE):
        row_t = "│"
        row_p = "│"
        for x in range(GRID_SIZE):
            row_t += symbols.get(true_grid[y, x], "?")
            row_p += symbols.get(pred_grid[y, x], "?")
        row_t += "│"
        row_p += "│"
        lines.append(row_t + "    " + row_p)
    lines.append("└" + "─" * GRID_SIZE + "┘" + "    " + "└" + "─" * GRID_SIZE + "┘")
    return "\n".join(lines)


def play_autoregressive(
    model: WorldModel,
    max_steps: int = 100,
    device: str = "cpu",
    render: bool = True,
    compare: bool = True,
    action_policy: str = "random",
):
    """Run the model in closed-loop autoregressive mode.

    The model predicts the next state given current state + action.
    The predicted state becomes the next input. No ground truth used.

    Args:
        model: Trained WorldModel
        max_steps: Max steps to simulate
        device: Device to run on
        render: Print ASCII visualization
        compare: Also run ground truth engine for comparison
        action_policy: 'random' or 'straight' or 'human'
    """
    model.eval()
    device = torch.device(device)

    # Ground truth engine (for comparison or action generation)
    engine = SnakeEngine(seed=42)
    true_state = engine.reset()

    # Predicted state — start from same initial state
    pred_grid, pred_dir, pred_go = encode_state(
        true_state.grid, DIRECTION_VECTORS[true_state.direction_idx], true_state.game_over
    )
    pred_game_over = False

    steps = 0
    correct_steps = 0

    while not pred_game_over and steps < max_steps:
        # Decide action
        if action_policy == "random":
            action_idx = np.random.randint(0, 3)
        elif action_policy == "straight":
            action_idx = 1
        else:
            # random fallback
            action_idx = np.random.randint(0, 3)

        # Ground truth step (for comparison)
        _, _, true_next, _, true_done = engine.step(action_idx)
        true_next_grid, true_next_dir, true_next_go = encode_state(
            true_next.grid, DIRECTION_VECTORS[true_next.direction_idx], true_next.game_over
        )

        # Model prediction
        g_probs, d_probs, go_prob = model.predict_step(pred_grid, pred_dir, pred_go, action_idx)

        # Decode prediction
        pred_next_grid = decode_grid(g_probs)
        pred_next_dir_idx = decode_direction(d_probs)
        pred_next_game_over = go_prob > 0.5

        # Compare with ground truth
        comparison = ""
        step_label = "✓"
        if compare:
            true_grid = decode_grid(true_next_grid)
            match = np.array_equal(pred_next_grid, true_grid)
            if match:
                correct_steps += 1
                step_label = "✓"
            else:
                step_label = "✗"

        if render:
            print(f"\n--- Step {steps} | Action: {['←','↑','→'][action_idx]} | {step_label} ---")
            print(f"Predicted game_over: {go_prob:.3f}")
            if compare:
                print(render_side_by_side(decode_grid(true_next_grid), pred_next_grid))
            else:
                print(render_grid(pred_next_grid))

        # Feed prediction back as next input
        # For the grid, take argmax (discrete) — this is the autoregressive choice
        pred_grid = g_probs  # Use probabilities for smoother inputs? Or argmax?
        # Argmax is more realistic but may cause drift. Let's use probabilities
        # for the first version.
        pred_dir = d_probs
        pred_go = np.array([go_prob], dtype=np.float32)
        pred_game_over = pred_next_game_over

        steps += 1

        if pred_game_over:
            print(f"\nModel predicted game over after {steps} steps")

    if compare and steps > 0:
        accuracy_pct = 100.0 * correct_steps / steps
        print(f"\n=== Autoregressive Results ===")
        print(f"Steps simulated: {steps}")
        print(f"Grid accuracy vs ground truth: {correct_steps}/{steps} ({accuracy_pct:.1f}%)")

    return steps


def main():
    parser = argparse.ArgumentParser(description="Run sNNake autoregressive playback")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Model checkpoint")
    parser.add_argument("--steps", type=int, default=100, help="Max steps")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--no-compare", action="store_true", help="Disable ground truth comparison")
    parser.add_argument("--policy", type=str, default="random", choices=["random", "straight"])
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model = WorldModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    print(f"Model loaded (epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f})")

    play_autoregressive(
        model,
        max_steps=args.steps,
        device=args.device,
        render=not args.no_render,
        compare=not args.no_compare,
        action_policy=args.policy,
    )


if __name__ == "__main__":
    main()
