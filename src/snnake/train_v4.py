"""Training pipeline for sNNake v4 — discrete world model.

Simple single-step teacher forcing with cross-entropy loss.
No BPTT, no scheduled sampling, no gradual unroll.

The discrete model produces exact grid cell predictions — no drift
in closed-loop playback. AR test measures raw per-step accuracy.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .model_v4 import (
    DiscreteWorldModel,
    compute_loss,
    accuracy_metrics,
)
from .collector_v2 import load_data_v2

GRID_SIZE = 10
NUM_CELLS = GRID_SIZE * GRID_SIZE


def to_cell_index(x_norm, y_norm, grid_size=GRID_SIZE):
    """Convert normalized (x,y) coordinates to cell index (0-99)."""
    x = (x_norm * (grid_size - 1)).round().astype(np.int64)
    y = (y_norm * (grid_size - 1)).round().astype(np.int64)
    return y * grid_size + x


def convert_to_discrete(data: dict) -> dict:
    """Convert continuous coordinate data to discrete cell indices."""
    out = {}
    for key in ["head", "food"]:
        out[key] = to_cell_index(data[key][:, 0], data[key][:, 1])
    for key in ["next_head", "next_food"]:
        out[key] = to_cell_index(data[key][:, 0], data[key][:, 1])
    # Body sequences: (N, L, 2) → (N, L)
    for key, src_key in [("body", "body"), ("next_body", "next_body")]:
        x = data[src_key][:, :, 0]
        y = data[src_key][:, :, 1]
        out[key] = to_cell_index(x, y)
    # Non-spatial data passes through
    for key in ["direction", "action", "game_over", "body_mask",
                "next_direction", "next_game_over", "ate_food"]:
        out[key] = data[key]
    return out


def unzip(data: dict) -> dict:
    """Convert batched numpy arrays to scalar tensors for single-transition processing."""
    return {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in data.items()}


class DiscreteSnakeDataset(Dataset):
    """Simple dataset returning individual (state → next_state) transitions."""

    def __init__(self, data_np: dict, noise_prob: float = 0.0):
        self.data = convert_to_discrete(data_np)
        self.N = len(self.data["head"])
        self.noise_prob = noise_prob

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.data.items()}
        # Noise injection: with noise_prob, randomly corrupt the head position
        # (simulates prediction errors during closed-loop playback)
        if self.noise_prob > 0 and np.random.random() < self.noise_prob:
            # Shift head by ±1 cell in a random direction
            hx = (item["head"] % GRID_SIZE).item()
            hy = (item["head"] // GRID_SIZE).item()
            dir_idx = np.random.randint(0, 4)
            dx, dy = [(-1, 0), (1, 0), (0, -1), (0, 1)][dir_idx]
            nx, ny = np.clip(hx + dx, 0, GRID_SIZE - 1), np.clip(hy + dy, 0, GRID_SIZE - 1)
            item["head"] = ny * GRID_SIZE + nx
        return item


def collate_discrete(batch):
    """Collate a batch of discrete transitions."""
    keys = batch[0].keys()
    result = {}
    for k in keys:
        # Stack numpy arrays
        stacked = np.stack([b[k] for b in batch])
        if stacked.dtype in (np.float32, np.float64):
            result[k] = torch.from_numpy(stacked).float()
        else:
            result[k] = torch.from_numpy(stacked).long()
    return result


def autoregressive_test_discrete(
    model: torch.nn.Module,
    data_np: dict,
    num_samples: int = 200,
    max_steps: int = 256,
    seed: int = 42,
    device: torch.device | None = None,
):
    """Run closed-loop autoregressive evaluation with discrete predictions.

    Arms: snap to exact cell — no drift. Measures raw prediction accuracy
    at each step and the longest streak of correct predictions.
    """
    if device is None:
        device = next(model.parameters()).device

    data = convert_to_discrete(data_np)
    N = len(data["head"])

    # Find valid starting points (non-episode-ending, non-food eating)
    go = data["next_game_over"].ravel()
    ate = data["ate_food"].ravel()
    valid_start = np.where((go == 0) & (ate == 0))[0]
    if len(valid_start) > num_samples:
        rng = np.random.RandomState(seed)
        rng.shuffle(valid_start)
    starts = valid_start[:num_samples]

    model.eval()
    step_errors = []
    step_counts = []

    with torch.no_grad():
        for start_idx in tqdm(starts, desc="AR test", leave=False):
            correct_count = 0
            # Initialize from ground truth
            head_t = torch.tensor([data["head"][start_idx]], device=device)
            dir_t = torch.from_numpy(data["direction"][start_idx]).unsqueeze(0).float().to(device)
            food_t = torch.tensor([data["food"][start_idx]], device=device)
            body_t = torch.from_numpy(data["body"][start_idx]).unsqueeze(0).long().to(device)
            body_mask_t = torch.from_numpy(data["body_mask"][start_idx]).unsqueeze(0).float().to(device)

            for step in range(max_steps):
                step_idx = start_idx + step
                if step_idx >= N:
                    break

                # Action for this step
                action_t = torch.from_numpy(data["action"][step_idx]).unsqueeze(0).float().to(device)
                go_t = torch.zeros(1, 1, device=device)

                # Model forward
                head_logits, food_logits, ate_logits, go_logits, dir_logits, body_logits = model(
                    head_t, dir_t, action_t, food_t, go_t, body_t, body_mask_t
                )

                # Targets
                target_head = torch.tensor([data["next_head"][step_idx]], device=device)
                target_food = torch.tensor([data["next_food"][step_idx]], device=device)
                target_go = torch.tensor([data["next_game_over"][step_idx]], device=device).float()
                target_body = torch.from_numpy(data["next_body"][step_idx]).unsqueeze(0).long().to(device)
                target_body_mask = torch.from_numpy(data["body_mask"][step_idx]).unsqueeze(0).float().to(device)

                # Check head accuracy
                head_pred = head_logits.argmax(dim=1)
                head_correct = (head_pred == target_head).item()

                if not head_correct:
                    step_counts.append(correct_count)
                    break

                correct_count += 1

                # Prepare next step (model-in-loop for head and body)
                head_t = head_pred  # already (1,) — batch dim from argmax

                body_pred = body_logits.argmax(dim=-1)  # (1, L)

                # Update body with prediction
                body_t = body_pred
                body_mask_t = target_body_mask

                # Food: only update if model predicts ate_food
                ate_pred = (ate_logits > 0).float()
                if ate_pred.item() > 0.5:
                    food_t = food_logits.argmax(dim=1)
                # else food stays the same

                # Direction: derive deterministically
                old_dir_idx = dir_t.argmax(dim=1)
                act_idx = action_t.argmax(dim=1)
                offsets = torch.tensor([-1, 0, 1], device=device)
                new_dir_idx = (old_dir_idx + offsets[act_idx]) % 4
                dir_t = torch.zeros(1, 4, device=device)
                dir_t[torch.arange(1), new_dir_idx] = 1.0

                # Check game over — if predicted GO, stop
                if (go_logits > 0).item():
                    step_counts.append(correct_count)
                    break

                # Check if we reached end of episode in ground truth
                if target_go.item() > 0.5:
                    step_counts.append(correct_count)
                    break

            else:
                # Reached max_steps without divergence
                step_counts.append(correct_count)

    model.train()
    result = {
        "mean_stable_steps": float(np.mean(step_counts)),
        "max_stable_steps": int(np.max(step_counts)),
        "median_stable_steps": float(np.median(step_counts)),
        "samples_tested": len(step_counts),
    }
    return result


def train_v4(
    data_path: str | Path = "data/games_v3_combined.npz",
    checkpoint_dir: str | Path = "checkpoints_v4",
    batch_size: int = 256,
    epochs: int = 30,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-5,
    noise_prob: float = 0.02,  # probability of corrupting head position during training
    val_split: float = 0.1,
    device: str = "auto",
    seed: int = 42,
    test_interval: int = 5,
    test_samples: int = 100,
    ar_max_steps: int = 256,
):
    """Train discrete world model with teacher forcing and CE loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    # --- Load data ---
    print(f"\nLoading data from {data_path}")
    data = load_data_v2(data_path)
    N = len(data["head"])
    print(f"Total transitions: {N:,}")

    # --- Train/val split ---
    indices = np.arange(N)
    np.random.shuffle(indices)
    val_size = int(N * val_split)
    train_idx = indices[val_size:]
    val_idx = indices[:val_size]

    train_data = {k: v[train_idx] for k, v in data.items()}
    val_data = {k: v[val_idx] for k, v in data.items()}
    print(f"Train: {len(train_idx):,} | Val: {len(val_idx):,}")

    # --- Datasets ---
    train_dataset = DiscreteSnakeDataset(train_data, noise_prob=noise_prob)
    val_dataset = DiscreteSnakeDataset(val_data, noise_prob=0.0)  # no noise during val

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_discrete, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_discrete, pin_memory=True,
    )
    print(f"Batches per epoch: ~{len(train_loader)}")

    # --- Model ---
    model = DiscreteWorldModel().to(device)
    print(f"\nModel params: {model.get_num_params():,}")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # --- Training ---
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "ar_steps": []}
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        total_loss = 0.0
        total_acc = {"head_acc": 0.0, "body_acc": 0.0, "full_match": 0.0}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [train]", leave=False)
        for batch in pbar:
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Model forward
            head_logits, food_logits, ate_logits, go_logits, dir_logits, body_logits = model(
                batch["head"],
                batch["direction"].float(),
                batch["action"].float(),
                batch["food"],
                batch["game_over"].float(),
                batch["body"],
                batch["body_mask"].float(),
            )

            # Loss
            loss_dict = compute_loss(
                head_logits, food_logits, ate_logits, go_logits, dir_logits, body_logits,
                batch["next_head"], batch["next_food"],
                batch["ate_food"], batch["next_game_over"],
                batch["next_direction"].argmax(dim=1),
                batch["next_body"], batch["body_mask"].float(),
            )

            optimizer.zero_grad()
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss_dict["total"].item()

            # Accuracy
            acc = accuracy_metrics(
                head_logits, food_logits, ate_logits, go_logits, dir_logits, body_logits,
                batch["next_head"], batch["next_food"],
                batch["ate_food"], batch["next_game_over"],
                batch["next_direction"].argmax(dim=1),
                batch["next_body"], batch["body_mask"].float(),
            )
            for k in total_acc:
                total_acc[k] += acc[k]
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{loss_dict['total']:.4f}",
                "head": f"{acc['head_acc']:.3f}",
                "body": f"{acc['body_acc']:.3f}",
            })

        avg_train_loss = total_loss / n_batches
        avg_train_acc = {k: v / n_batches for k, v in total_acc.items()}

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_acc = {"head_acc": 0.0, "body_acc": 0.0, "full_match": 0.0}
        val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [val]", leave=False):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                head_logits, food_logits, ate_logits, go_logits, dir_logits, body_logits = model(
                    batch["head"],
                    batch["direction"].float(),
                    batch["action"].float(),
                    batch["food"],
                    batch["game_over"].float(),
                    batch["body"],
                    batch["body_mask"].float(),
                )

                loss_dict = compute_loss(
                    head_logits, food_logits, ate_logits, go_logits, dir_logits, body_logits,
                    batch["next_head"], batch["next_food"],
                    batch["ate_food"], batch["next_game_over"],
                    batch["next_direction"].argmax(dim=1),
                    batch["next_body"], batch["body_mask"].float(),
                )

                val_loss += loss_dict["total"].item()

                acc = accuracy_metrics(
                    head_logits, food_logits, ate_logits, go_logits, dir_logits, body_logits,
                    batch["next_head"], batch["next_food"],
                    batch["ate_food"], batch["next_game_over"],
                    batch["next_direction"].argmax(dim=1),
                    batch["next_body"], batch["body_mask"].float(),
                )
                for k in val_acc:
                    val_acc[k] += acc[k]
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        avg_val_acc = {k: v / val_batches for k, v in val_acc.items()}

        scheduler.step()

        # --- Checkpoint ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
            print(f"  ✓ New best model (val_loss={avg_val_loss:.4f})")

        # --- Autoregressive test ---
        ar_result = {}
        if (epoch + 1) % test_interval == 0 or epoch == 0:
            print("  Running autoregressive test...")
            ar_result = autoregressive_test_discrete(
                model, val_data, num_samples=test_samples,
                max_steps=ar_max_steps, seed=seed, device=device,
            )
            print(f"  AR: mean={ar_result['mean_stable_steps']:.1f}, "
                  f"median={ar_result['median_stable_steps']:.1f}, "
                  f"max={ar_result['max_stable_steps']:.0f} steps "
                  f"({ar_result['samples_tested']} samples)")
            history["ar_steps"].append({"epoch": epoch + 1, **ar_result})

        # --- Logging ---
        epoch_time = time.time() - epoch_start
        history["train_loss"].append(float(avg_train_loss))
        history["val_loss"].append(float(avg_val_loss))
        history["train_acc"].append(avg_train_acc)
        history["val_acc"].append(avg_val_acc)

        print(f"  Epoch {epoch + 1:3d}/{epochs} | "
              f"loss={avg_train_loss:.4f} | val={avg_val_loss:.4f} | "
              f"head={avg_train_acc['head_acc']:.3f}/{avg_val_acc['head_acc']:.3f} | "
              f"body={avg_train_acc['body_acc']:.3f}/{avg_val_acc['body_acc']:.3f} | "
              f"full={avg_train_acc['full_match']:.3f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | "
              f"time={epoch_time:.1f}s")

        # Save latest checkpoint and history
        torch.save(model.state_dict(), checkpoint_dir / "latest.pt")
        with open(checkpoint_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Training complete! Time: {total_time / 60:.1f} min")
    print(f"Best val loss: {best_val_loss:.4f}")

    # Final AR test
    print("\nFinal autoregressive test:")
    ar_result = autoregressive_test_discrete(
        model, val_data, num_samples=test_samples,
        max_steps=ar_max_steps, seed=seed, device=device,
    )
    print(f"  Mean:   {ar_result['mean_stable_steps']:.1f} steps")
    print(f"  Median: {ar_result['median_stable_steps']:.1f} steps")
    print(f"  Max:    {ar_result['max_stable_steps']:.0f} steps")
    history["final_ar"] = ar_result

    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train sNNake v4 — discrete world model")
    parser.add_argument("--data", type=str, default="data/games_v3_combined.npz")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_v4")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--noise-prob", type=float, default=0.02,
                        help="Probability of head noise during training (for robustness)")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-interval", type=int, default=5)
    parser.add_argument("--test-samples", type=int, default=100)
    parser.add_argument("--ar-max-steps", type=int, default=256)
    args = parser.parse_args()

    train_v4(
        data_path=args.data,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        noise_prob=args.noise_prob,
        val_split=args.val_split,
        seed=args.seed,
        test_interval=args.test_interval,
        test_samples=args.test_samples,
        ar_max_steps=args.ar_max_steps,
    )


if __name__ == "__main__":
    main()
