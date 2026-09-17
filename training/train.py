import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.histo_sam_lora import HistoSAM_LoRA
from training.losses import BoundaryAwareJointLoss
from evaluation.metrics import SegmentationMetricsMeter
from data.dataset import build_dataloaders_from_split


def set_seed(seed: int = 42):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_vram_usage(device: torch.device):

    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        print(
            f"[VRAM Monitor] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB (Limit: 12.0 GB)"
        )


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool = True,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_focal = 0.0
    total_dice = 0.0
    total_boundary = 0.0
    num_batches = len(loader)

    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with autocast():
                logits = model(images)
                loss, loss_dict = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss, loss_dict = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        total_loss += loss_dict["loss_total"]
        total_focal += loss_dict["loss_focal"]
        total_dice += loss_dict["loss_dice"]
        total_boundary += loss_dict["loss_boundary"]

    return {
        "train_loss": total_loss / max(num_batches, 1),
        "train_focal": total_focal / max(num_batches, 1),
        "train_dice": total_dice / max(num_batches, 1),
        "train_boundary": total_boundary / max(num_batches, 1),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 3,
    compute_boundary_metrics: bool = True,
) -> Tuple[Dict[str, float], float]:
    model.eval()
    val_loss = 0.0
    meter = SegmentationMetricsMeter(num_classes=num_classes)
    num_batches = len(loader)

    for batch in tqdm(loader, desc="Validation", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        logits = model(images)
        loss, _ = criterion(logits, masks)
        val_loss += loss.item()

        meter.update(logits, masks, compute_boundary_metrics=compute_boundary_metrics)

    metrics = meter.summary()
    metrics["val_loss"] = val_loss / max(num_batches, 1)
    return metrics, metrics["mDice"]


def run_training(
    split_file: str,
    data_dir: str,
    checkpoint_path: Optional[str] = None,
    output_dir: str = "results/checkpoints",
    epochs: int = 30,
    batch_size: int = 4,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    decoder_type: str = "carafe",
    lambda_focal: float = 1.0,
    lambda_dice: float = 1.0,
    lambda_boundary: float = 0.2,
    seed: int = 42,
    use_amp: bool = True,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using compute device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading split from {split_file}...")
    train_loader, val_loader = build_dataloaders_from_split(
        split_file=split_file,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=2 if device.type == "cuda" else 0,
        img_size=512,
    )

    print(f"[INFO] Initializing HistoSAM (r={lora_r}, decoder={decoder_type})...")
    model = HistoSAM_LoRA(
        checkpoint_path=checkpoint_path,
        num_classes=3,
        r=lora_r,
        lora_alpha=lora_alpha,
        decoder_type=decoder_type,
    )
    model.to(device)
    model.print_parameter_summary()
    print_vram_usage(device)

    trainable_params = model.get_trainable_parameters()
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=(use_amp and device.type == "cuda"))

    criterion = BoundaryAwareJointLoss(
        gamma=2.0,
        lambda_focal=lambda_focal,
        lambda_dice=lambda_dice,
        lambda_boundary=lambda_boundary,
        ignore_index=255,
    ).to(device)

    split_stem = Path(split_file).stem
    best_dice = 0.0
    history = []

    print("=" * 70)
    print(f" STARTING TRAINING: {split_stem} ({epochs} Epochs)")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )

        scheduler.step()

        compute_boundary = (epoch % 5 == 0) or (epoch == epochs)
        val_metrics, current_dice = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=3,
            compute_boundary_metrics=compute_boundary,
        )

        elapsed = time.time() - epoch_start
        epoch_record = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "elapsed_sec": elapsed,
            **train_metrics,
            **val_metrics,
        }
        history.append(epoch_record)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_metrics['train_loss']:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"mIoU: {val_metrics['mIoU']*100:.2f}% | "
            f"mDice: {val_metrics['mDice']*100:.2f}% | "
            f"Time: {elapsed:.1f}s"
        )
        print_vram_usage(device)

        if current_dice > best_dice:
            best_dice = current_dice
            best_model_path = out_path / f"{split_stem}_best.pth"
            model.save_trainable_weights(str(best_model_path))
            print(
                f"  --> Saved new best model (mDice: {best_dice*100:.2f}%) to {best_model_path.name}"
            )

    history_file = out_path / f"{split_stem}_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[INFO] Saved training history to {history_file}")

    return history


def run_pretraining(
    public_dirs: list[Path | str],
    checkpoint_path: Optional[str] = None,
    output_dir: str = "results/checkpoints",
    lora_r: int = 16,
    lora_alpha: float = 32.0,
    decoder_type: str = "carafe",
    lambda_focal: float = 1.0,
    lambda_dice: float = 1.0,
    lambda_boundary: float = 0.2,
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-4,
    seed: int = 42,
    use_amp: bool = True,
):

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    from data.dataset import PublicLiverDataset
    from torch.utils.data import DataLoader, random_split

    dataset = PublicLiverDataset(public_dirs, img_size=512)
    if len(dataset) == 0:
        print(
            "[ERROR] No public dataset samples found. Generate with 'python data/preprocessing/download_public_data.py --synthetic-demo'."
        )
        return

    val_len = max(1, int(len(dataset) * 0.2))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(
        dataset, [train_len, val_len], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    print(
        f"[INFO] Starting Pre-training on {len(dataset)} samples ({train_len} train, {val_len} val)..."
    )
    model = HistoSAM_LoRA(
        checkpoint_path=checkpoint_path,
        num_classes=3,
        r=lora_r,
        lora_alpha=lora_alpha,
        decoder_type=decoder_type,
    )
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = BoundaryAwareJointLoss(
        lambda_focal=lambda_focal,
        lambda_dice=lambda_dice,
        lambda_boundary=lambda_boundary,
    ).to(device)
    scaler = GradScaler(enabled=use_amp and device.type == "cuda")

    best_dice = 0.0
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, use_amp
        )
        val_metrics, _ = validate(
            model,
            val_loader,
            criterion,
            device,
            num_classes=3,
            compute_boundary_metrics=False,
        )
        scheduler.step()

        current_dice = val_metrics["mDice"]
        print(
            f"Pre-train Epoch [{epoch:02d}/{epochs:02d}] Loss: {train_metrics['train_loss']:.4f} | Val Dice: {current_dice*100:.2f}%"
        )

        if current_dice >= best_dice:
            best_dice = current_dice
            best_save = out_path / "pretrain_best.pth"
            model.save_trainable_weights(str(best_save))
            print(f"  --> Saved new best pre-trained model to {best_save.name}")

    print(
        f"[SUCCESS] Pre-training completed. Best model saved to {out_path / 'pretrain_best.pth'}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train HistoSAM-LoRA for Hepatic Pathology"
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="kfold",
        choices=["pretrain", "kfold"],
        help="Phase: pretrain or kfold",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="Path to split JSON file (for kfold)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/liver_primary/processed",
        help="Path to processed patches",
    )
    parser.add_argument(
        "--public_dirs",
        nargs="+",
        default=["data/public_pretrain/hepass", "data/public_pretrain/kmc_liver"],
        help="Paths to public dataset folders",
    )
    parser.add_argument(
        "--medsam_checkpoint",
        type=str,
        default="models/MedSAM/medsam_vit_b.pth",
        help="Path to MedSAM ViT-B checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (recommended: 4 for 12GB VRAM)",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank r (0 for frozen MedSAM baseline, 2, 4, 8, 16, 32)",
    )
    parser.add_argument(
        "--lora_alpha", type=float, default=16.0, help="LoRA scaling factor alpha"
    )
    parser.add_argument(
        "--decoder_type",
        type=str,
        default="carafe",
        choices=["carafe", "bilinear", "nearest", "conv_transpose", "pixel_shuffle"],
        help="Semantic upsampling decoder architecture",
    )
    parser.add_argument(
        "--lambda_focal", type=float, default=1.0, help="Focal loss weight"
    )
    parser.add_argument(
        "--lambda_dice", type=float, default=1.0, help="Dice loss weight"
    )
    parser.add_argument(
        "--lambda_boundary",
        type=float,
        default=0.2,
        help="Boundary Laplacian loss weight",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no_amp", action="store_true", help="Disable AMP mixed precision"
    )

    args = parser.parse_args()

    chk = args.medsam_checkpoint if Path(args.medsam_checkpoint).exists() else None
    if chk is None:
        print(
            "[WARN] MedSAM checkpoint not found; initializing model with random weights for dry-run/testing."
        )

    if args.phase == "pretrain":
        run_pretraining(
            public_dirs=args.public_dirs,
            checkpoint_path=chk,
            output_dir=args.output_dir,
            lora_r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            decoder_type=args.decoder_type,
            lambda_focal=args.lambda_focal,
            lambda_dice=args.lambda_dice,
            lambda_boundary=args.lambda_boundary,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            use_amp=not args.no_amp,
        )
    else:
        if not args.split_file:
            print("[ERROR] --split_file is required for kfold phase.")
            sys.exit(1)
        run_training(
            split_file=args.split_file,
            data_dir=args.data_dir,
            checkpoint_path=chk,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lora_r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            decoder_type=args.decoder_type,
            lambda_focal=args.lambda_focal,
            lambda_dice=args.lambda_dice,
            lambda_boundary=args.lambda_boundary,
            seed=args.seed,
            use_amp=not args.no_amp,
        )
