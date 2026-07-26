"""Training pipeline for sNNake v5 — structured world model with deterministic physics.

Only two learned components:
  1. Direction update: tiny MLP (direction + action → new direction)
  2. Food spawn: body encoder + MLP (body context → new cell)

Everything else is exact arithmetic — zero autoregressive drift.

Expected: 100% direction accuracy → infinite stable playback.
10 epoch target: >98% direction accuracy, >50% food accuracy.
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

from .model import StructuredWorldModel, compute_v5_loss
from .collector import load_data_v2

GRID_SIZE = 10
NUM_CELLS = GRID_SIZE * GRID_SIZE
MAX_BODY_LEN = 40


class SnakeDataset(Dataset):
    """Dataset returning continuous (state → next_state) transitions for v5."""

    def __init__(self, data_np: dict, noise_prob: float = 0.0):
        self.data = data_np
        self.N = len(self.data["head"])
        self.noise_prob = noise_prob

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.data.items()}
        # Direction noise: randomly corrupt direction with noise_prob
        # This simulates direction prediction errors during autoregressive playback
        if self.noise_prob > 0 and np.random.random() < self.noise_prob:
            d = np.random.randint(0, 4)
            item["direction"] = np.eye(4, dtype=np.float32)[d]
        return item


def collate_continuous(batch):
    """Collate a batch of continuous transitions."""
    keys = batch[0].keys()
    result = {}
    for k in keys:
        stacked = np.stack([b[k] for b in batch])
        if stacked.dtype in (np.float32, np.float64):
            result[k] = torch.from_numpy(stacked).float()
        else:
            result[k] = torch.from_numpy(stacked).long()
    return result


def autoregressive_test_v5(
    model: torch.nn.Module,
    data_np: dict,
    num_samples: int = 200,
    max_steps: int = 5000,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict:
    """Closed-loop AR test for v5 — uses forward_ar (skips body encoder GRU).

    Only failure mode: wrong direction prediction. Physics is exact.
    Records consecutive correct steps per sample.
    """
    if device is None:
        device = next(model.parameters()).device

    data = data_np
    N = len(data["head"])
    grid_size = GRID_SIZE

    # Find starting points (non-episode-ending)
    go = data["next_game_over"].ravel()
    valid_start = np.where(go == 0)[0]
    if len(valid_start) > num_samples:
        rng = np.random.RandomState(seed)
        rng.shuffle(valid_start)
    starts = valid_start[:num_samples]

    model.eval()
    step_counts = []
    grid_size = GRID_SIZE

    with torch.no_grad():
        for start_idx in tqdm(starts, desc="AR test", leave=False):
            # Batch the entire trajectory through direction MLP (one call)
            end_idx = min(start_idx + max_steps, N)
            S = end_idx - start_idx

            # Collect all (direction, action) pairs and targets for this trajectory
            batch_dirs = torch.from_numpy(data["direction"][start_idx:end_idx]).float().to(device)  # (S, 4)
            batch_acts = torch.from_numpy(data["action"][start_idx:end_idx]).float().to(device)  # (S, 3)
            batch_next_dir = data["next_direction"][start_idx:end_idx]  # (S, 4)
            batch_go = data["next_game_over"][start_idx:end_idx].ravel()  # (S,)

            # Single batched forward pass through direction MLP
            dir_input = torch.cat([batch_dirs, batch_acts], dim=1)  # (S, 7)
            all_dir_logits = model.dir_net(dir_input)  # (S, 4)
            all_dir_preds = all_dir_logits.argmax(dim=1).cpu().numpy()  # (S,)
            all_dir_targets = batch_next_dir.argmax(axis=1)  # (S,)

            # Count consecutive correct steps
            correct_count = 0
            for s in range(S):
                if all_dir_preds[s] != all_dir_targets[s]:
                    break
                correct_count += 1
                if batch_go[s] > 0.5:
                    break
            step_counts.append(correct_count)

    model.train()
    step_arr = np.array(step_counts, dtype=np.float64) if step_counts else np.array([0.0])
    result = {
        "mean_stable_steps": float(step_arr.mean()),
        "max_stable_steps": int(step_arr.max()),
        "median_stable_steps": float(np.median(step_arr)),
        "samples_tested": len(step_counts),
        "pct_at_100": float((step_arr >= 100).mean()),
        "pct_at_500": float((step_arr >= 500).mean()),
        "pct_at_1000": float((step_arr >= 1000).mean()),
        "pct_at_5000": float((step_arr >= 5000).mean()),
    }
    return result


def train_v5(
    data_path: str | Path = "data/games_v3_combined.npz",
    checkpoint_dir: str | Path = "checkpoints_v5",
    batch_size: int = 256,
    epochs: int = 20,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-5,
    noise_prob: float = 0.0,
    val_split: float = 0.1,
    device: str = "auto",
    seed: int = 42,
    test_interval: int = 5,
    test_samples: int = 200,
    ar_max_steps: int = 5000,
):
    """Train StructuredWorldModel v5."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
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
    train_dataset = SnakeDataset(train_data, noise_prob=noise_prob)
    val_dataset = SnakeDataset(val_data, noise_prob=0.0)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_continuous, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_continuous, pin_memory=True,
    )
    print(f"Batches per epoch: ~{len(train_loader)}")

    # --- Model ---
    model = StructuredWorldModel().to(device)
    print(f"\nModel params: {model.get_num_params():,} (all learned)")
    print(f"  Direction network: ~1,200 params")
    print(f"  Food network (body_encoder + MLP): ~12,600 params")

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
        total_acc = {"dir_acc": 0.0, "head_acc": 0.0, "body_acc": 0.0, "ate_acc": 0.0, "go_acc": 0.0, "full_match": 0.0}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [train]", leave=False)
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Model forward
            dir_logits, food_logits, pred_head, pred_body, pred_mask, pred_ate, pred_go = model(
                batch["head"],
                batch["direction"].float(),
                batch["action"].float(),
                batch["food"],
                batch["body"],
                batch["body_mask"].float(),
            )

            # Loss
            loss_dict = compute_v5_loss(
                dir_logits, food_logits, pred_head, pred_body, pred_mask, pred_ate, pred_go,
                batch["next_direction"].float(), batch["next_food"],
                batch["next_head"], batch["next_body"],
                batch["body_mask"].float(),
                batch["next_game_over"].float(), batch["ate_food"].float(),
            )

            optimizer.zero_grad()
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss_dict["total"].item()

            for k in total_acc:
                if k in loss_dict:
                    total_acc[k] += loss_dict[k]
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{loss_dict['total']:.4f}",
                "dir": f"{loss_dict['dir_acc']:.3f}",
                "head": f"{loss_dict['head_acc']:.3f}",
            })

        avg_train_loss = total_loss / n_batches
        avg_train_acc = {k: v / n_batches for k, v in total_acc.items()}

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_acc = {k: 0.0 for k in total_acc}
        val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [val]", leave=False):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                dir_logits, food_logits, pred_head, pred_body, pred_mask, pred_ate, pred_go = model(
                    batch["head"],
                    batch["direction"].float(),
                    batch["action"].float(),
                    batch["food"],
                    batch["body"],
                    batch["body_mask"].float(),
                )

                loss_dict = compute_v5_loss(
                    dir_logits, food_logits, pred_head, pred_body, pred_mask, pred_ate, pred_go,
                    batch["next_direction"].float(), batch["next_food"],
                    batch["next_head"], batch["next_body"],
                    batch["body_mask"].float(),
                    batch["next_game_over"].float(), batch["ate_food"].float(),
                )

                val_loss += loss_dict["total"].item()

                for k in val_acc:
                    if k in loss_dict:
                        val_acc[k] += loss_dict[k]
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
        # IMPORTANT: use original (temporally ordered) data, NOT shuffled val_data
        # val_data has random ordering — consecutive indices are from different
        # episodes, breaking AR test coherency.
        ar_result = {}
        if (epoch + 1) % test_interval == 0 or epoch == 0:
            print("  Running autoregressive test (exact physics — direction accuracy only)...")
            ar_result = autoregressive_test_v5(
                model, data, num_samples=test_samples,
                max_steps=ar_max_steps, seed=seed, device=device,
            )
            print(f"  AR: mean={ar_result['mean_stable_steps']:.1f}, "
                  f"max={ar_result['max_stable_steps']:.0f}, "
                  f">=100 steps: {ar_result['pct_at_100']*100:.0f}%, "
                  f">=1000: {ar_result['pct_at_1000']*100:.0f}% "
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
              f"dir={avg_train_acc['dir_acc']:.3f}/{avg_val_acc['dir_acc']:.3f} | "
              f"head={avg_train_acc['head_acc']:.3f}/{avg_val_acc['head_acc']:.3f} | "
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

    # Final AR test (on original temporal data)
    print("\nFinal autoregressive test:")
    ar_result = autoregressive_test_v5(
        model, data, num_samples=test_samples,
        max_steps=ar_max_steps, seed=seed, device=device,
    )
    print(f"  Mean:   {ar_result['mean_stable_steps']:.1f} steps")
    print(f"  Max:    {ar_result['max_stable_steps']:.0f} steps")
    print(f"  >=100:  {ar_result['pct_at_100']*100:.0f}%")
    print(f"  >=1000: {ar_result['pct_at_1000']*100:.0f}%")
    history["final_ar"] = ar_result

    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train sNNake v5 — structured world model")
    parser.add_argument("--data", type=str, default="data/games_v3_combined.npz")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_v5")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--noise-prob", type=float, default=0.0,
                        help="Probability of direction corruption during training")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-interval", type=int, default=5)
    parser.add_argument("--test-samples", type=int, default=200)
    parser.add_argument("--ar-max-steps", type=int, default=5000)
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, mps, or cpu")
    args = parser.parse_args()

    train_v5(
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
        device=args.device,
    )


if __name__ == "__main__":
    main()
