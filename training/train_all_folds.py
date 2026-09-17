"""Automated Cross-Validation Runner for HistoSAM-LoRA across all K-Folds.

Executes training sequentially across all splits, aggregates clinical metrics,
and generates the final Mean +- Std table required for Q1 journal submissions.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

import torch
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train import run_training


def train_all_folds(
    splits_dir: str = "data/splits",
    data_dir: str = "data/liver_primary/processed",
    medsam_checkpoint: str = "models/MedSAM/medsam_vit_b.pth",
    output_dir: str = "results/checkpoints",
    lora_r: int = 8,
    lora_alpha: float = 16.0,
    decoder_type: str = "carafe",
    lambda_focal: float = 1.0,
    lambda_dice: float = 1.0,
    lambda_boundary: float = 0.2,
    epochs: int = 30,
    batch_size: int = 4,
    lr: float = 1e-4,
    seed: int = 42,
    use_amp: bool = True,
):
    splits_path = Path(splits_dir)
    split_files = sorted(list(splits_path.glob("split_*.json")))

    if not split_files:
        print(f"[ERROR] No split files found in {splits_dir}!")
        print("Please generate splits first using:")
        print("python -c \"from data.dataset import create_patient_stratified_splits; create_patient_stratified_splits('data/liver_primary/processed/images', 'data/liver_primary/processed/masks', 'data/splits', n_splits=5, n_repeats=3)\"")
        return

    print(f"[INFO] Found {len(split_files)} split files to train in {splits_dir}.")
    print(f"[INFO] Model Configuration: r={lora_r}, decoder={decoder_type}, lambda_boundary={lambda_boundary}")

    chk = medsam_checkpoint if Path(medsam_checkpoint).exists() else None
    if chk is None:
        print("[WARN] MedSAM checkpoint not found. Model will train with randomly initialized base weights.")

    all_fold_results: List[Dict] = []

    for idx, sfile in enumerate(split_files):
        print("\n" + "#" * 70)
        print(f" TRAINING SPLIT {idx+1}/{len(split_files)}: {sfile.name}")
        print("#" * 70 + "\n")

        history = run_training(
            split_file=str(sfile),
            data_dir=data_dir,
            checkpoint_path=chk,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            decoder_type=decoder_type,
            lambda_focal=lambda_focal,
            lambda_dice=lambda_dice,
            lambda_boundary=lambda_boundary,
            seed=seed + idx,
            use_amp=use_amp,
        )

        # Extract best epoch metrics based on mDice
        best_epoch = max(history, key=lambda x: x.get("mDice", 0.0))
        best_epoch["split_file"] = sfile.name
        all_fold_results.append(best_epoch)

    # Aggregate results into final summary
    print("\n" + "=" * 70)
    print(" FINAL CROSS-VALIDATION SUMMARY (MEAN +- STD)")
    print("=" * 70)

    metrics_to_aggregate = [
        "mIoU", "mDice", "IoU_necrosis", "Dice_necrosis",
        "IoU_normal", "Dice_normal", "IoU_steatosis", "Dice_steatosis",
        "mHD95", "mASD"
    ]

    summary_stats = {}
    for m in metrics_to_aggregate:
        vals = [r[m] for r in all_fold_results if m in r and not np.isnan(r[m])]
        if vals:
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            summary_stats[m] = {"mean": mean_val, "std": std_val}

            is_percent = "IoU" in m or "Dice" in m
            scale = 100.0 if is_percent else 1.0
            unit = "%" if is_percent else "px"
            print(f" {m:<18}: {mean_val*scale:.2f} +- {std_val*scale:.2f} {unit}")

    out_summary_dir = Path("results")
    out_summary_dir.mkdir(parents=True, exist_ok=True)

    summary_json = out_summary_dir / "final_kfold_summary.json"
    with open(summary_json, "w") as f:
        json.dump({"summary": summary_stats, "all_folds": all_fold_results}, f, indent=2)

    df_folds = pd.DataFrame(all_fold_results)
    summary_csv = out_summary_dir / "final_kfold_summary.csv"
    df_folds.to_csv(summary_csv, index=False)

    print(f"\n[INFO] Complete summary saved to {summary_json} and {summary_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HistoSAM-LoRA across all K-Folds")
    parser.add_argument("--splits_dir", type=str, default="data/splits")
    parser.add_argument("--data_dir", type=str, default="data/liver_primary/processed")
    parser.add_argument("--medsam_checkpoint", type=str, default="models/MedSAM/medsam_vit_b.pth")
    parser.add_argument("--output_dir", type=str, default="results/checkpoints")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank r (0 for frozen MedSAM baseline, 2, 4, 8, 16, 32)")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA alpha scaling factor")
    parser.add_argument(
        "--decoder_type",
        type=str,
        default="carafe",
        choices=["carafe", "bilinear", "nearest", "conv_transpose", "pixel_shuffle"],
        help="Semantic upsampling decoder architecture",
    )
    parser.add_argument("--lambda_focal", type=float, default=1.0, help="Focal loss weight")
    parser.add_argument("--lambda_dice", type=float, default=1.0, help="Dice loss weight")
    parser.add_argument("--lambda_boundary", type=float, default=0.2, help="Boundary Laplacian loss weight")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")

    args = parser.parse_args()
    train_all_folds(
        splits_dir=args.splits_dir,
        data_dir=args.data_dir,
        medsam_checkpoint=args.medsam_checkpoint,
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
