"""Iterative Self-Training Engine with EMA Teacher and Curriculum Thresholding.

Combats confirmation bias and error propagation in histopathology semantic segmentation:
1. Teacher model updated via Exponential Moving Average (EMA).
2. Curriculum confidence thresholding: starts strict (e.g. 0.92) and relaxes (to 0.82).
3. Evaluates pseudo-label delta between rounds to detect automated convergence (< 1%).
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.histo_sam_lora import build_histo_sam_lora
from data.dataset import HistopathologyDataset, PseudoLabeledDataset
from training.losses import ConfidenceWeightedLoss
from training.generate_pseudo_labels import generate_pseudo_labels


def update_ema_variables(model_student: nn.Module, model_teacher: nn.Module, alpha: float):
    """Updates teacher model parameters via Exponential Moving Average (EMA)."""
    with torch.no_grad():
        for param_student, param_teacher in zip(model_student.parameters(), model_teacher.parameters()):
            if param_student.requires_grad:
                param_teacher.data.mul_(alpha).add_(param_student.data, alpha=1.0 - alpha)


def compute_pseudo_label_delta(old_dir: Path, new_dir: Path) -> float:
    """Computes the percentage of pixel label shifts between two pseudo-label rounds."""
    old_files = sorted(list(old_dir.glob("*_pseudolabel.png")))
    total_pixels = 0
    changed_pixels = 0

    for f_old in old_files:
        f_new = new_dir / f_old.name
        if not f_new.exists():
            continue

        arr_old = np.array(Image.open(str(f_old)))
        arr_new = np.array(Image.open(str(f_new)))

        diff = (arr_old != arr_new).sum()
        changed_pixels += diff
        total_pixels += arr_old.size

    if total_pixels == 0:
        return 0.0
    return float(changed_pixels / total_pixels)


def train_one_round(
    model_student: nn.Module,
    model_teacher: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: ConfidenceWeightedLoss,
    device: torch.device,
    epochs_per_round: int = 5,
    ema_alpha: float = 0.99,
) -> dict:
    """Runs student training for one self-training round."""
    model_student.train()
    model_teacher.eval()

    round_loss = 0.0
    round_conf_ratio = 0.0
    total_steps = 0

    for epoch in range(epochs_per_round):
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            confs = batch["confidence"].to(device)

            optimizer.zero_grad()
            logits = model_student(images)

            loss, loss_stats = criterion(logits, masks, confs)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model_student.parameters(), max_norm=1.0)
            optimizer.step()

            # Update EMA teacher
            update_ema_variables(model_student, model_teacher, ema_alpha)

            round_loss += loss_stats["loss_total"]
            round_conf_ratio += loss_stats["confident_ratio"]
            total_steps += 1

    avg_loss = round_loss / max(1, total_steps)
    avg_conf = round_conf_ratio / max(1, total_steps)
    return {"avg_loss": avg_loss, "avg_confident_ratio": avg_conf}


def run_iterative_self_training(
    image_paths: list[Path],
    initial_checkpoint: str | None,
    output_dir: str | Path = "results/checkpoints",
    pseudo_dir: str | Path = "data/liver_primary/pseudo_labels",
    num_rounds: int = 4,
    epochs_per_round: int = 5,
    initial_threshold: float = 0.90,
    final_threshold: float = 0.80,
    lr: float = 1e-4,
    batch_size: int = 4,
    img_size: int = 512,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    convergence_tol: float = 0.01,  # 1% change stop condition
):
    out_ckpt_dir = Path(output_dir)
    out_ckpt_dir.mkdir(parents=True, exist_ok=True)
    pseudo_base_dir = Path(pseudo_dir)
    pseudo_base_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device)
    print(f"[INFO] Initializing Iterative Self-Training on device {dev}")
    print(f"  - Images: {len(image_paths)}")
    print(f"  - Rounds: {num_rounds} | Epochs/Round: {epochs_per_round}")
    print(f"  - Curriculum Threshold: {initial_threshold:.2f} -> {final_threshold:.2f}")

    # Build Student Model
    model_student = build_histo_sam_lora(num_classes=3, lora_rank=16, decoder_type="carafe")
    if initial_checkpoint and Path(initial_checkpoint).exists():
        print(f"[INFO] Loading initial pre-trained checkpoint from {initial_checkpoint}")
        ckpt = torch.load(initial_checkpoint, map_location=dev)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model_student.load_state_dict(state_dict, strict=False)

    model_student.to(dev)

    # Build Teacher Model (EMA)
    model_teacher = copy.deepcopy(model_student)
    for p in model_teacher.parameters():
        p.requires_grad = False
    model_teacher.to(dev)

    # Dataset for pseudo-label generation
    eval_dataset = HistopathologyDataset(image_paths, img_size=img_size)

    # Initial Pseudo-Label generation
    round_pseudo_dir = pseudo_base_dir / "round_0"
    print(f"\n--- [ROUND 0] Generating Initial Pseudo-Labels ---")
    generate_pseudo_labels(model_teacher, eval_dataset, round_pseudo_dir, dev)

    best_loss = float("inf")
    history = []

    for r in range(1, num_rounds + 1):
        # Calculate curriculum threshold
        progress = (r - 1) / max(1, num_rounds - 1)
        cur_threshold = initial_threshold - progress * (initial_threshold - final_threshold)

        print(f"\n=======================================================")
        print(f" [ROUND {r}/{num_rounds}] Curriculum Threshold: {cur_threshold:.3f}")
        print(f"=======================================================")

        prev_pseudo_dir = pseudo_base_dir / f"round_{r-1}"
        cur_pseudo_dir = pseudo_base_dir / f"round_{r}"

        # Setup dataset with current pseudo-labels
        pseudo_ds = PseudoLabeledDataset(image_paths, prev_pseudo_dir, img_size=img_size)
        if len(pseudo_ds) == 0:
            print("[ERROR] No valid pseudo-labeled samples found. Aborting.")
            break

        train_loader = DataLoader(pseudo_ds, batch_size=batch_size, shuffle=True, drop_last=False)

        # Trainable parameters (LoRA + CARAFE Decoder only)
        trainable_params = [p for p in model_student.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
        criterion = ConfidenceWeightedLoss(threshold=cur_threshold, soft_weighting=False)

        # Train student for round
        train_stats = train_one_round(
            model_student=model_student,
            model_teacher=model_teacher,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=dev,
            epochs_per_round=epochs_per_round,
        )

        # Generate new pseudo-labels with Teacher
        print(f"[ROUND {r}] Refreshing Pseudo-Labels using updated Teacher...")
        generate_pseudo_labels(model_teacher, eval_dataset, cur_pseudo_dir, dev)

        # Check convergence delta
        delta = compute_pseudo_label_delta(prev_pseudo_dir, cur_pseudo_dir)
        print(f"[CONVERGENCE] Label Shift Delta vs Round {r-1}: {delta*100:.2f}% (Tolerance: {convergence_tol*100:.2f}%)")

        round_record = {
            "round": r,
            "threshold": cur_threshold,
            "train_loss": train_stats["avg_loss"],
            "confident_ratio": train_stats["avg_confident_ratio"],
            "label_delta": delta,
        }
        history.append(round_record)

        # Save checkpoint
        save_path = out_ckpt_dir / f"selftrain_round_{r}.pth"
        torch.save(
            {
                "round": r,
                "model_state_dict": model_teacher.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            },
            save_path,
        )
        print(f"[SAVED] Checkpoint saved to {save_path}")

        # Best checkpoint
        if train_stats["avg_loss"] < best_loss:
            best_loss = train_stats["avg_loss"]
            best_path = out_ckpt_dir / "selftrain_best.pth"
            torch.save({"model_state_dict": model_teacher.state_dict(), "history": history}, best_path)
            print(f"[BEST] New best model saved to {best_path}")

        # Check convergence stop condition
        if delta < convergence_tol and r > 1:
            print(f"\n[EARLY STOPPING] Pseudo-labels converged at round {r} (delta {delta*100:.3f}% < {convergence_tol*100:.2f}%).")
            break

    print("\n[COMPLETE] Iterative Self-Training Pipeline finished.")
    return history


def main():
    parser = argparse.ArgumentParser(description="Iterative Self-Training for Liver Histopathology")
    parser.add_argument("--checkpoint", type=str, default=None, help="Initial pre-trained checkpoint")
    parser.add_argument("--raw-dir", type=str, default="data/liver_primary/raw_images")
    parser.add_argument("--magnification", type=str, default="20", help="Filter magnification ('20', 'all')")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs-per-round", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    image_files = sorted(list(raw_dir.glob("*.tif")) + list(raw_dir.glob("*.tiff")))

    mag_filter = None if args.magnification.lower() == "all" else args.magnification
    dataset = HistopathologyDataset(image_files, magnification_filter=mag_filter)

    run_iterative_self_training(
        image_paths=dataset.image_paths,
        initial_checkpoint=args.checkpoint,
        num_rounds=args.rounds,
        epochs_per_round=args.epochs_per_round,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
