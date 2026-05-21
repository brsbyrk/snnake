"""Training pipeline for sNNake v3 — scaled model + multi-step BPTT.

Features:
  - Scaled model: ~177K params (14× v2.2)
  - Multi-step unrolled training (BPTT over 8-16 steps)
  - Scheduled sampling: gradually increase model-in-loop
  - Episodic boundary detection avoids corrupting gradient
  - Periodic autoregressive validation during training
"""

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

from .model_v3 import (
    StructuredWorldModelV3 as StructuredWorldModel,
    compute_loss,
    accuracy_metrics,
)
from .collector_v2 import (
    load_data_v2,
    collect_data_v2,
    save_data_v2,
    SnakeEngine,
    normalize_coord,
    body_to_array,
    direction_from_index,
    apply_action,
)

GRID_SIZE = 10
MAX_BODY_LEN = 40


def find_episode_chunks(data: dict, unroll_steps: int) -> list[int]:
    """Find valid start indices for multi-step unrolling.

    A start is valid if unroll_steps consecutive transitions exist
    with no episode boundary crossing and no food eating (body length stable).
    """
    go = data["next_game_over"].ravel()
    ate = data["ate_food"].ravel()
    N = len(go)

    # Simple scan: any transition in the window must not be episode-end or food-eating
    valid = np.ones(N - unroll_steps, dtype=bool) if N > unroll_steps else np.array([], dtype=bool)
    for t in range(unroll_steps):
        valid &= (go[t:N - unroll_steps + t] == 0)
        valid &= (ate[t:N - unroll_steps + t] == 0)

    starts = np.where(valid)[0]
    return starts.tolist()


def build_autoregressive_test_data(
    data: dict,
    num_samples: int = 200,
    seed: int = 42,
    unroll_steps: int = 64,
) -> list[dict]:
    """Sample episode start points from data for autoregressive validation.

    Returns a list of dicts, each containing unroll_steps of consecutive
    ground-truth transitions for closed-loop playback evaluation.
    """
    rng = np.random.RandomState(seed)
    go = data["next_game_over"].ravel()
    N = len(go)

    # Find all episode starts
    ends = np.where(go == 1)[0]
    starts = np.concatenate([[0], ends + 1])
    ends = np.concatenate([ends, [N - 1]])

    # Filter episodes long enough for our test unroll
    min_len = 10
    valid_eps = [(s, e) for s, e in zip(starts, ends) if (e - s) >= min_len]

    if len(valid_eps) < num_samples:
        # Repeat with different offsets
        samples = []
        while len(samples) < num_samples:
            for s, e in valid_eps:
                if len(samples) >= num_samples:
                    break
                ep_len = e - s
                max_start = max(0, ep_len - unroll_steps)
                for offset in range(0, max_start, max(1, ep_len // 3)):
                    if len(samples) >= num_samples:
                        break
                    start = s + offset
                    end = min(start + unroll_steps, e + 1)
                    samples.append({
                        "start": start,
                        "end": end,
                        "length": end - start,
                    })
        return samples

    # Random sample
    chosen = rng.choice(len(valid_eps), size=num_samples, replace=False)
    samples = []
    for ci in chosen:
        s, e = valid_eps[ci]
        ep_len = e - s
        max_start = max(0, ep_len - unroll_steps)
        if max_start > 0:
            offset = rng.randint(0, max_start)
        else:
            offset = 0
        start = s + offset
        end = min(start + unroll_steps, e + 1)
        samples.append({
            "start": start,
            "end": end,
            "length": end - start,
        })

    return samples


def autoregressive_test(
    model: nn.Module,
    data: dict,
    samples: list[dict],
    device: torch.device,
    head_threshold: float = 0.15,  # more lenient for multi-step
) -> dict:
    """Run closed-loop autoregressive evaluation.

    For each test sample, start from ground truth step 0, then feed
    model's predictions back as input for subsequent steps.
    """
    model.eval()
    # Accept either numpy or torch tensors
    is_numpy = isinstance(next(iter(data.values())), np.ndarray)
    if is_numpy:
        device_data = {k: torch.from_numpy(v).to(device) for k, v in data.items()}
    else:
        device_data = {k: v.to(device) if isinstance(v, torch.Tensor) else torch.from_numpy(v).to(device)
                       for k, v in data.items()}

    total_steps = 0
    stable_until = []  # steps before head error exceeds threshold
    all_errors = []

    with torch.no_grad():
        for sample in tqdm(samples, desc="Autoregressive test", leave=False):
            start = sample["start"]
            end = sample["end"]
            seq_len = sample["length"]

            # Initial state (step 0) — ground truth
            head_t = device_data["head"][start].unsqueeze(0)
            food_t = device_data["food"][start].unsqueeze(0)
            dir_t = device_data["direction"][start].unsqueeze(0)
            body_t = device_data["body"][start].unsqueeze(0)
            body_mask_t = device_data["body_mask"][start].unsqueeze(0)

            stable_count = 0
            errors = []

            for t in range(seq_len):
                step_idx = start + t

                # Action for this step (from data)
                action_t = device_data["action"][step_idx].unsqueeze(0)

                # Game over (start as false, then model prediction)
                go_t = torch.zeros(1, 1, device=device)

                # Model forward
                next_head, next_food, ate_logits, go_logits, dir_logits, new_body = model(
                    head_t, dir_t, action_t, food_t, go_t, body_t, body_mask_t
                )

                # Target for this step
                target_head = device_data["next_head"][step_idx].unsqueeze(0)
                target_food = device_data["next_food"][step_idx].unsqueeze(0)
                target_body = device_data["next_body"][step_idx].unsqueeze(0)
                target_body_mask = device_data["body_mask"][step_idx].unsqueeze(0)
                target_ate = device_data["ate_food"][step_idx].unsqueeze(0)
                target_go = device_data["next_game_over"][step_idx].unsqueeze(0)
                target_dir = device_data["next_direction"][step_idx].unsqueeze(0)

                # Head error
                head_error = torch.norm(next_head - target_head, dim=1).item()
                errors.append(head_error)

                if head_error < head_threshold:
                    stable_count += 1
                else:
                    break  # diverged

                # Prepare next step input (model-in-loop)
                head_t = next_head
                body_t = new_body

                # Food: use ground truth next_food (model can't predict random food)
                food_t = target_food

                # Direction: derive deterministically
                old_dir_idx = dir_t.argmax(dim=1)
                action_idx = action_t.argmax(dim=1)
                offsets = torch.tensor([-1, 0, 1], device=device)
                new_dir_idx = (old_dir_idx + offsets[action_idx]) % 4
                dir_t = torch.zeros(1, 4, device=device)
                dir_t[torch.arange(1), new_dir_idx] = 1.0

                # Body mask: update if ate food
                ate_pred = (ate_logits > 0).float()
                body_mask_t = target_body_mask.clone()
                # If not ate, the body length stays the same
                # If ate, the body grows by 1 — target_body_mask already has the
                # right mask, so we use it directly

                total_steps += 1

            stable_until.append(stable_count)
            all_errors.extend(errors)

    model.train()
    mean_stable = np.mean(stable_until) if stable_until else 0
    max_stable = max(stable_until) if stable_until else 0
    median_stable = np.median(stable_until) if stable_until else 0

    return {
        "mean_stable_steps": mean_stable,
        "max_stable_steps": max_stable,
        "median_stable_steps": median_stable,
        "samples_tested": len(samples),
        "total_steps": total_steps,
    }


class MultiStepDataset(Dataset):
    """Dataset that returns start indices for multi-step unrolled training.

    __getitem__ returns (start_idx,) — the training loop performs
    the actual unrolling, using model predictions as subsequent inputs.
    """

    def __init__(self, data: dict, unroll_steps: int):
        self.valid_starts = find_episode_chunks(data, unroll_steps)
        self.unroll_steps = unroll_steps

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        return self.valid_starts[idx]


def train_v3(
    data_path: str | Path = "data/games_v3_combined.npz",
    checkpoint_dir: str | Path = "checkpoints_v3",
    batch_size: int = 128,
    epochs: int = 50,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-5,
    unroll_steps: int = 1,          # start with single-step teacher forcing
    scheduled_sampling_start: float = 0.0,
    scheduled_sampling_end: float = 0.5,
    noise_std: float = 0.02,
    val_split: float = 0.1,
    device: str = "auto",
    seed: int = 42,
    test_interval: int = 5,
    test_samples: int = 100,
    resume: str | None = None,      # path to checkpoint to resume from
    unroll_gradual: int = 0,        # if >0, scale unroll_steps from 2 to target over N epochs
):
    """Train sNNake v3 with multi-step BPTT and scheduled sampling."""
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
    print(f"  Body length range: {data['body_mask'].sum(axis=1).min():.0f} – {data['body_mask'].sum(axis=1).max():.0f}")

    # --- Train/val split ---
    indices = np.arange(N)
    np.random.shuffle(indices)
    val_size = int(N * val_split)
    train_idx = indices[val_size:]
    val_idx = indices[:val_size]

    train_data = {k: v[train_idx] for k, v in data.items()}
    val_data = {k: v[val_idx] for k, v in data.items()}

    # Pre-convert to tensors for efficiency in training loop
    train_tensors = {k: torch.from_numpy(v).to(device, non_blocking=True) for k, v in train_data.items()}
    val_tensors = {k: torch.from_numpy(v).to(device, non_blocking=True) for k, v in val_data.items()}

    print(f"Train: {len(train_idx):,} | Val: {len(val_idx):,}")

    # --- Build model ---
    model = StructuredWorldModel().to(device)
    start_epoch = 0
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "ar_steps": []}

    if resume:
        print(f"Resuming from {resume}")
        state_dict = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print("  Model weights loaded")

        # Load history if available
        hist_path = Path(resume).parent / "history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                history = json.load(f)
            best_val_loss = min(history.get("val_loss", [float("inf")]))
            start_epoch = len(history.get("train_loss", []))
            print(f"  Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

    print(f"\nModel params: {model.get_num_params():,}")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # --- Prepare autoregressive test data ---
    ar_test_samples = build_autoregressive_test_data(val_data, num_samples=test_samples, seed=seed, unroll_steps=64)
    print(f"Autoregressive test samples: {len(ar_test_samples)} (up to 64 steps each)")

    # --- Training loop ---
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        # Unroll schedule: gradually increase unroll_steps
        current_unroll = unroll_steps
        if unroll_gradual > 0:
            progress_in_schedule = (epoch - start_epoch) / max(unroll_gradual, 1)
            if progress_in_schedule < 1.0:
                current_unroll = max(1, int(2 + (unroll_steps - 2) * progress_in_schedule))

        # Rebuild dataloaders with current unroll_steps if changed
        if epoch == start_epoch or (unroll_gradual > 0 and epoch > start_epoch):
            train_dataset = MultiStepDataset(train_data, current_unroll)
            val_dataset = MultiStepDataset(val_data, current_unroll)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=True,
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=True,
            )
        epoch_start = time.time()

        # --- Training ---
        model.train()
        total_train_loss = 0.0
        train_batches = 0

        # Current scheduled sampling probability
        progress = epoch / max(epochs - 1, 1)
        teacher_prob = 1.0 - (scheduled_sampling_start + (scheduled_sampling_end - scheduled_sampling_start) * progress)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [train]", leave=False)
        for batch_starts in pbar:
            batch_starts = batch_starts.to(device)
            B = batch_starts.shape[0]
            U = current_unroll

            # Build indices for all steps: (B, U+1)
            offsets = torch.arange(U + 1, device=device).unsqueeze(0)
            indices = batch_starts.unsqueeze(1) + offsets  # (B, U+1)

            # Pre-gather ALL data in one shot per key
            all_head = train_tensors["head"][indices]                    # (B, U+1, 2)
            all_food = train_tensors["food"][indices]                    # (B, U+1, 2)
            all_dir = train_tensors["direction"][indices]                # (B, U+1, 4)
            all_action = train_tensors["action"][indices]                # (B, U+1, 3)
            all_body = train_tensors["body"][indices]                    # (B, U+1, L, 2)
            all_body_mask = train_tensors["body_mask"][indices]          # (B, U+1, L)
            all_next_head = train_tensors["next_head"][indices]          # (B, U+1, 2)
            all_next_food = train_tensors["next_food"][indices]          # (B, U+1, 2)
            all_next_dir = train_tensors["next_direction"][indices]      # (B, U+1, 4)
            all_next_go = train_tensors["next_game_over"][indices]       # (B, U+1, 1)
            all_next_ate = train_tensors["ate_food"][indices]            # (B, U+1, 1)
            all_next_body = train_tensors["next_body"][indices]          # (B, U+1, L, 2)
            all_next_body_mask = train_tensors["body_mask"][indices]     # (B, U+1, L)

            # Step 0 state
            head_t = all_head[:, 0]
            dir_t = all_dir[:, 0]
            food_t = all_food[:, 0]
            body_t = all_body[:, 0]
            body_mask_t = all_body_mask[:, 0]

            # Unroll loop — only model forward passes, no Python data gathering
            batch_loss = 0.0
            for t in range(U):
                action_t = all_action[:, t]
                go_t = torch.zeros(B, 1, device=device)

                # Targets for this step
                target_head = all_next_head[:, t]
                target_food = all_next_food[:, t]
                target_dir = all_next_dir[:, t]
                target_go = all_next_go[:, t]
                target_ate = all_next_ate[:, t]
                target_body = all_next_body[:, t]
                target_body_mask = all_next_body_mask[:, t]

                # Scheduled sampling: optionally replace head/body with ground truth
                if t > 0 and torch.rand(1).item() < teacher_prob:
                    head_t = all_head[:, t]
                    body_t = all_body[:, t]
                    body_mask_t = all_body_mask[:, t]

                # Model forward
                next_head, next_food, ate_logits, go_logits, dir_logits, new_body = model(
                    head_t, dir_t, action_t, food_t, go_t, body_t, body_mask_t
                )

                # Loss
                loss_dict = compute_loss(
                    next_head, next_food, ate_logits, go_logits, dir_logits, new_body,
                    target_head, target_food, target_ate, target_go,
                    target_dir.argmax(dim=1), target_body,
                    target_body_mask,
                    head_positions=head_t,
                )
                batch_loss = batch_loss + loss_dict["total"]

                # Prepare next step's input (model-in-loop)
                head_t = next_head
                body_t = new_body
                body_mask_t = target_body_mask
                food_t = target_food

                # Derive direction deterministically
                old_dir_idx = dir_t.argmax(dim=1)
                action_idx = action_t.argmax(dim=1)
                offsets_t = torch.tensor([-1, 0, 1], device=device)
                new_dir_idx = (old_dir_idx + offsets_t[action_idx]) % 4
                dir_t = torch.zeros(B, 4, device=device)
                dir_t[torch.arange(B), new_dir_idx] = 1.0

            # Backprop
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += batch_loss.item()
            train_batches += 1
            pbar.set_postfix({"loss": f"{batch_loss.item() / U:.4f}"})

        avg_train_loss = total_train_loss / max(train_batches, 1) / max(current_unroll, 1)

        # --- Validation ---
        model.eval()
        total_val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_starts in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [val]", leave=False):
                batch_starts = batch_starts.to(device)
                B = batch_starts.shape[0]
                U = current_unroll

                # Pre-gather all steps
                offsets = torch.arange(U + 1, device=device).unsqueeze(0)
                indices = batch_starts.unsqueeze(1) + offsets

                all_head = val_tensors["head"][indices]
                all_food = val_tensors["food"][indices]
                all_dir = val_tensors["direction"][indices]
                all_action = val_tensors["action"][indices]
                all_body = val_tensors["body"][indices]
                all_body_mask = val_tensors["body_mask"][indices]
                all_next_head = val_tensors["next_head"][indices]
                all_next_food = val_tensors["next_food"][indices]
                all_next_dir = val_tensors["next_direction"][indices]
                all_next_go = val_tensors["next_game_over"][indices]
                all_next_ate = val_tensors["ate_food"][indices]
                all_next_body = val_tensors["next_body"][indices]
                all_next_body_mask = val_tensors["body_mask"][indices]

                head_t = all_head[:, 0]
                dir_t = all_dir[:, 0]
                food_t = all_food[:, 0]
                body_t = all_body[:, 0]
                body_mask_t = all_body_mask[:, 0]

                batch_loss = 0.0
                for t in range(U):
                    action_t = all_action[:, t]
                    go_t = torch.zeros(B, 1, device=device)

                    target_head = all_next_head[:, t]
                    target_food = all_next_food[:, t]
                    target_dir = all_next_dir[:, t]
                    target_go = all_next_go[:, t]
                    target_ate = all_next_ate[:, t]
                    target_body = all_next_body[:, t]
                    target_body_mask = all_next_body_mask[:, t]

                    next_head, next_food, ate_logits, go_logits, dir_logits, new_body = model(
                        head_t, dir_t, action_t, food_t, go_t, body_t, body_mask_t
                    )

                    loss_dict = compute_loss(
                        next_head, next_food, ate_logits, go_logits, dir_logits, new_body,
                        target_head, target_food, target_ate, target_go,
                        target_dir.argmax(dim=1), target_body,
                        target_body_mask,
                        head_positions=head_t,
                    )

                    batch_loss = batch_loss + loss_dict["total"]

                    # Teacher forcing for val
                    head_t = target_head
                    food_t = target_food
                    body_t = target_body
                    body_mask_t = target_body_mask

                    old_dir_idx = dir_t.argmax(dim=1)
                    action_idx = action_t.argmax(dim=1)
                    offsets_t = torch.tensor([-1, 0, 1], device=device)
                    new_dir_idx = (old_dir_idx + offsets_t[action_idx]) % 4
                    dir_t = torch.zeros(B, 4, device=device)
                    dir_t[torch.arange(B), new_dir_idx] = 1.0

                total_val_loss += batch_loss.item()
                val_batches += 1

        avg_val_loss = total_val_loss / max(val_batches, 1) / max(current_unroll, 1)
        scheduler.step()

        # --- Checkpoint ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
            print(f"  ✓ New best model (val_loss={avg_val_loss:.6f})")

        # --- Autoregressive test ---
        ar_result = {}
        if (epoch + 1) % test_interval == 0 or epoch == 0:
            print("  Running autoregressive test...")
            ar_result = autoregressive_test(
                model, val_data, ar_test_samples, device,
            )
            print(f"  AR: mean={ar_result['mean_stable_steps']:.1f}, "
                  f"median={ar_result['median_stable_steps']:.1f}, "
                  f"max={ar_result['max_stable_steps']:.0f} steps")
            history["ar_steps"].append({
                "epoch": epoch + 1,
                **ar_result,
            })

        # --- Logging ---
        epoch_time = time.time() - epoch_start
        history["train_loss"].append(float(avg_train_loss))
        history["val_loss"].append(float(avg_val_loss))

        print(f"  Epoch {epoch + 1:3d}/{epochs} | "
              f"train_loss={avg_train_loss:.6f} | "
              f"val_loss={avg_val_loss:.6f} | "
              f"teacher={teacher_prob:.2f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e} | "
              f"time={epoch_time:.1f}s")

        # Save latest checkpoint
        torch.save(model.state_dict(), checkpoint_dir / "latest.pt")

        # Save history
        with open(checkpoint_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Training complete! Time: {total_time / 60:.1f} min")
    print(f"Best val loss: {best_val_loss:.6f}")

    # Final autoregressive test
    print("\nFinal autoregressive test:")
    ar_result = autoregressive_test(
        model, val_data, ar_test_samples, device,
    )
    print(f"  Mean:   {ar_result['mean_stable_steps']:.1f} steps")
    print(f"  Median: {ar_result['median_stable_steps']:.1f} steps")
    print(f"  Max:    {ar_result['max_stable_steps']:.0f} steps")
    history["final_ar"] = ar_result

    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nCheckpoints saved to {checkpoint_dir}")
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train sNNake v3 — scaled model with multi-step BPTT")
    parser.add_argument("--data", type=str, default="data/games_v3_combined.npz",
                        help="Training data path")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_v3",
                        help="Checkpoint directory")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--unroll-steps", type=int, default=1,
                        help="Multi-step BPTT unroll length (start at 1 for teacher forcing)")
    parser.add_argument("--unroll-gradual", type=int, default=0,
                        help="If >0, gradually increase unroll from 2 to target over N epochs")
    parser.add_argument("--ss-start", type=float, default=0.0,
                        help="Scheduled sampling start prob (model-in-loop)")
    parser.add_argument("--ss-end", type=float, default=0.5,
                        help="Scheduled sampling end prob (model-in-loop)")
    parser.add_argument("--noise-std", type=float, default=0.02,
                        help="Input noise std for robustness")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test-interval", type=int, default=5,
                        help="Epochs between autoregressive tests")
    parser.add_argument("--test-samples", type=int, default=100,
                        help="Samples for autoregressive test")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (.pt file)")
    parser.add_argument("--resume-best", action="store_true",
                        help="Resume from checkpoints_v3/best.pt (shortcut)")
    args = parser.parse_args()

    train_v3(
        data_path=args.data,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        unroll_steps=args.unroll_steps,
        scheduled_sampling_start=args.ss_start,
        scheduled_sampling_end=args.ss_end,
        noise_std=args.noise_std,
        val_split=args.val_split,
        seed=args.seed,
        test_interval=args.test_interval,
        test_samples=args.test_samples,
        resume=args.resume or (args.checkpoint_dir + "/best.pt" if args.resume_best else None),
        unroll_gradual=args.unroll_gradual,
    )


if __name__ == "__main__":
    main()
