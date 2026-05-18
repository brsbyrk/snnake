"""Training pipeline for sNNake v2 — structured world model."""

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

from .model_v2 import StructuredWorldModel, compute_loss, accuracy_metrics
from .collector_v2 import load_data_v2


class SnakeDatasetV2(Dataset):
    """Dataset of Snake transitions in structured coordinate format."""

    def __init__(self, data: dict):
        self.head = torch.from_numpy(data["head"])
        self.direction = torch.from_numpy(data["direction"])
        self.action = torch.from_numpy(data["action"])
        self.food = torch.from_numpy(data["food"])
        self.game_over = torch.from_numpy(data["game_over"])
        self.body = torch.from_numpy(data["body"])
        self.body_mask = torch.from_numpy(data["body_mask"])
        self.next_head = torch.from_numpy(data["next_head"])
        self.next_food = torch.from_numpy(data["next_food"])
        self.next_direction = torch.from_numpy(data["next_direction"])
        self.next_game_over = torch.from_numpy(data["next_game_over"])
        self.ate_food = torch.from_numpy(data["ate_food"])
        self.next_body = torch.from_numpy(data["next_body"])

    def __len__(self):
        return len(self.head)

    def __getitem__(self, idx):
        return {
            "head": self.head[idx],
            "direction": self.direction[idx],
            "action": self.action[idx],
            "food": self.food[idx],
            "game_over": self.game_over[idx],
            "body": self.body[idx],
            "body_mask": self.body_mask[idx],
            "next_head": self.next_head[idx],
            "next_food": self.next_food[idx],
            "next_direction": self.next_direction[idx],
            "next_game_over": self.next_game_over[idx],
            "ate_food": self.ate_food[idx],
            "next_body": self.next_body[idx],
        }


def train_v2(
    data_path: str | Path = "data/games_v2.npz",
    checkpoint_dir: str | Path = "checkpoints_v2",
    batch_size: int = 256,
    epochs: int = 30,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-5,
    val_split: float = 0.1,
    device: str = "auto",
    seed: int = 42,
):
    """Train the structured world model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {data_path}...")
    data = load_data_v2(data_path)
    dataset = SnakeDatasetV2(data)

    n_total = len(dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"Train: {n_train:,} | Val: {n_val:,}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    # Model
    model = StructuredWorldModel().to(device)
    print(f"Model params: {model.get_num_params():,}")
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    history = []

    print(f"\n{'Epoch':>6} {'Train Loss':>10} {'Val Loss':>10} {'Head Acc':>8} {'Body Acc':>8} {'Go Acc':>8} {'Full':>8} {'Time':>8}")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            head = batch["head"].to(device)
            direction = batch["direction"].to(device)
            action = batch["action"].to(device)
            food = batch["food"].to(device)
            game_over = batch["game_over"].to(device)
            body = batch["body"].to(device)
            body_mask = batch["body_mask"].to(device)
            next_head = batch["next_head"].to(device)
            next_food = batch["next_food"].to(device)
            next_dir = batch["next_direction"].to(device)
            next_go = batch["next_game_over"].to(device)
            ate = batch["ate_food"].to(device)
            next_body = batch["next_body"].to(device)

            optimizer.zero_grad()

            nh_pred, nf_pred, ate_logits, go_logits, nd_logits, nb_pred = model(
                head, direction, action, food, game_over, body, body_mask
            )

            losses = compute_loss(
                nh_pred, nf_pred, ate_logits, go_logits, nd_logits, nb_pred,
                next_head, next_food, ate, next_go, next_dir.argmax(dim=1), next_body,
                body_mask,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(losses["total"].item())
            pbar.set_postfix({"loss": f"{losses['total'].item():.4f}"})

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        all_accs = []

        with torch.no_grad():
            for batch in val_loader:
                head = batch["head"].to(device)
                direction = batch["direction"].to(device)
                action = batch["action"].to(device)
                food = batch["food"].to(device)
                game_over = batch["game_over"].to(device)
                body = batch["body"].to(device)
                body_mask = batch["body_mask"].to(device)
                next_head = batch["next_head"].to(device)
                next_food = batch["next_food"].to(device)
                next_dir = batch["next_direction"].to(device)
                next_go = batch["next_game_over"].to(device)
                ate = batch["ate_food"].to(device)
                next_body = batch["next_body"].to(device)

                nh_pred, nf_pred, ate_logits, go_logits, nd_logits, nb_pred = model(
                    head, direction, action, food, game_over, body, body_mask
                )

                losses = compute_loss(
                    nh_pred, nf_pred, ate_logits, go_logits, nd_logits, nb_pred,
                    next_head, next_food, ate, next_go, next_dir.argmax(dim=1), next_body,
                    body_mask,
                )
                accs = accuracy_metrics(
                    nh_pred, nf_pred, ate_logits, go_logits, nb_pred,
                    next_head, next_food, ate, next_go, next_body,
                    body_mask,
                )

                val_losses.append(losses["total"].item())
                all_accs.append(accs)

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        avg_head_acc = np.mean([a["head_acc"] for a in all_accs])
        avg_body_acc = np.mean([a["body_acc"] for a in all_accs])
        avg_go_acc = np.mean([a["go_acc"] for a in all_accs])
        avg_full = np.mean([a["full_match"] for a in all_accs])
        epoch_time = time.time() - epoch_start

        print(
            f"{epoch:>6} {avg_train_loss:>10.4f} {avg_val_loss:>10.4f} "
            f"{avg_head_acc:>8.3f} {avg_body_acc:>8.3f} {avg_go_acc:>8.3f} {avg_full:>8.3f} "
            f"{epoch_time:>7.1f}s"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "head_acc": avg_head_acc,
                "body_acc": avg_body_acc,
                "go_acc": avg_go_acc,
                "full_match": avg_full,
            }, checkpoint_dir / "best.pt")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": avg_val_loss,
        }, checkpoint_dir / "latest.pt")

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "head_acc": avg_head_acc,
            "body_acc": avg_body_acc,
            "go_acc": avg_go_acc,
            "full_match": avg_full,
            "lr": scheduler.get_last_lr()[0],
        })

    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train sNNake v2 structured model")
    parser.add_argument("--data", type=str, default="data/games_v2.npz")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_v2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_v2(
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
