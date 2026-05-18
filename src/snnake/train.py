"""Training pipeline for sNNake world model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .model import WorldModel, compute_loss, accuracy
from .collector import load_data


class SnakeDataset(Dataset):
    """Dataset of Snake game transitions."""

    def __init__(self, data: dict):
        self.grid = torch.from_numpy(data["grid"])
        self.direction = torch.from_numpy(data["direction"])
        self.action = torch.from_numpy(data["action"])
        self.game_over = torch.from_numpy(data["game_over"])
        self.next_grid = torch.from_numpy(data["next_grid"])
        self.next_direction = torch.from_numpy(data["next_direction"])
        self.next_game_over = torch.from_numpy(data["next_game_over"])

    def __len__(self):
        return len(self.grid)

    def __getitem__(self, idx):
        return {
            "grid": self.grid[idx],
            "direction": self.direction[idx],
            "action": self.action[idx],
            "game_over": self.game_over[idx],
            "next_grid": self.next_grid[idx],
            "next_direction": self.next_direction[idx],
            "next_game_over": self.next_game_over[idx],
        }


def train(
    data_path: str | Path = "data/games.npz",
    checkpoint_dir: str | Path = "checkpoints",
    batch_size: int = 256,
    epochs: int = 50,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-5,
    val_split: float = 0.1,
    device: str = "auto",
    log_interval: int = 100,
    seed: int = 42,
):
    """Train the world model.

    Args:
        data_path: Path to collected data (.npz)
        checkpoint_dir: Directory to save checkpoints
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Adam learning rate
        weight_decay: AdamW weight decay
        val_split: Fraction of data to use for validation
        device: 'cpu', 'cuda', or 'auto'
        log_interval: Log every N batches
        seed: Random seed
    """
    # --- Setup ---
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    print(f"Loading data from {data_path}...")
    data = load_data(data_path)
    dataset = SnakeDataset(data)

    # Split
    n_total = len(dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"Train: {n_train:,} | Val: {n_val:,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 0 for stability
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # --- Model ---
    model = WorldModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # --- Training Loop ---
    best_val_loss = float("inf")
    history: list[dict] = []

    print(f"\nStarting training for {epochs} epochs...")
    print(f"{'Epoch':>6} {'Train Loss':>10} {'Val Loss':>10} {'Grid Acc':>8} {'Dir Acc':>8} {'Go Acc':>8} {'Time':>8}")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        train_losses: list[float] = []
        train_grid_accs: list[float] = []
        train_dir_accs: list[float] = []
        train_go_accs: list[float] = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            grid = batch["grid"].to(device)
            direction = batch["direction"].to(device)
            action = batch["action"].to(device)
            game_over = batch["game_over"].to(device)
            next_grid = batch["next_grid"].to(device)
            next_direction = batch["next_direction"].to(device)
            next_game_over = batch["next_game_over"].to(device)

            optimizer.zero_grad()

            # Forward
            g_logits, d_logits, go_logits = model(grid, direction, action, game_over)

            # Loss
            losses = compute_loss(
                g_logits, d_logits, go_logits,
                next_grid, next_direction.argmax(dim=1), next_game_over,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Metrics
            accs = accuracy(
                g_logits, d_logits, go_logits,
                next_grid, next_direction.argmax(dim=1), next_game_over,
            )

            train_losses.append(losses["total"].item())
            train_grid_accs.append(accs["grid_acc"])
            train_dir_accs.append(accs["dir_acc"])
            train_go_accs.append(accs["go_acc"])

            pbar.set_postfix({
                "loss": f"{losses['total'].item():.4f}",
                "g_acc": f"{accs['grid_acc']:.3f}",
            })

        scheduler.step()

        # --- Validation ---
        model.eval()
        val_losses: list[float] = []
        val_grid_accs: list[float] = []
        val_dir_accs: list[float] = []
        val_go_accs: list[float] = []

        with torch.no_grad():
            for batch in val_loader:
                grid = batch["grid"].to(device)
                direction = batch["direction"].to(device)
                action = batch["action"].to(device)
                game_over = batch["game_over"].to(device)
                next_grid = batch["next_grid"].to(device)
                next_direction = batch["next_direction"].to(device)
                next_game_over = batch["next_game_over"].to(device)

                g_logits, d_logits, go_logits = model(grid, direction, action, game_over)

                losses = compute_loss(
                    g_logits, d_logits, go_logits,
                    next_grid, next_direction.argmax(dim=1), next_game_over,
                )
                accs = accuracy(
                    g_logits, d_logits, go_logits,
                    next_grid, next_direction.argmax(dim=1), next_game_over,
                )

                val_losses.append(losses["total"].item())
                val_grid_accs.append(accs["grid_acc"])
                val_dir_accs.append(accs["dir_acc"])
                val_go_accs.append(accs["go_acc"])

        # --- Logging ---
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        avg_grid_acc = np.mean(val_grid_accs)
        avg_dir_acc = np.mean(val_dir_accs)
        avg_go_acc = np.mean(val_go_accs)
        epoch_time = time.time() - epoch_start

        print(
            f"{epoch:>6} {avg_train_loss:>10.4f} {avg_val_loss:>10.4f} "
            f"{avg_grid_acc:>8.3f} {avg_dir_acc:>8.3f} {avg_go_acc:>8.3f} "
            f"{epoch_time:>7.1f}s"
        )

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = checkpoint_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "grid_acc": avg_grid_acc,
                "dir_acc": avg_dir_acc,
                "go_acc": avg_go_acc,
            }, checkpoint_path)

        # Save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss,
        }, checkpoint_dir / "latest.pt")

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "grid_acc": avg_grid_acc,
            "dir_acc": avg_dir_acc,
            "go_acc": avg_go_acc,
            "lr": scheduler.get_last_lr()[0],
        })

    # Save history
    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model: {checkpoint_dir / 'best.pt'}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train sNNake world model")
    parser.add_argument("--data", type=str, default="data/games.npz", help="Training data path")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train(
        data_path=args.data,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
