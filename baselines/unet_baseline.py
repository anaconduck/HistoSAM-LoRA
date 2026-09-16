"""U-Net baseline for multi-class tissue segmentation (Q1 Benchmark Comparator).

Allows fair benchmark comparison against HistoSAM-LoRA using the exact same:
- Patient-Level Stratified K-Fold splits (split_rX_fY.json)
- Multi-class evaluation metrics (mIoU, mDice, HD95, ASD)
- Zero-augmentation protocol
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import build_dataloaders_from_split
from evaluation.metrics import SegmentationMetricsMeter


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet(nn.Module):
    """Standard U-Net architecture (Ronneberger et al., MICCAI 2015)."""

    def __init__(self, n_channels: int = 3, n_classes: int = 3):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 512))

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)

        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = self.conv1(torch.cat([x, x4], dim=1))

        x = self.up2(x)
        x = self.conv2(torch.cat([x, x3], dim=1))

        x = self.up3(x)
        x = self.conv3(torch.cat([x, x2], dim=1))

        x = self.up4(x)
        x = self.conv4(torch.cat([x, x1], dim=1))

        logits = self.outc(x)
        return logits


def train_unet_baseline(
    split_file: str,
    data_dir: str = "data/liver_primary/processed",
    output_dir: str = "results/checkpoints",
    epochs: int = 40,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    dev = torch.device(device)
    print(f"[INFO] Training U-Net baseline on {dev}...")

    train_loader, val_loader = build_dataloaders_from_split(
        split_file=split_file,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=2 if dev.type == "cuda" else 0,
        img_size=512,
    )

    model = UNet(n_channels=3, n_classes=3).to(dev)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    split_stem = Path(split_file).stem

    best_dice = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            imgs = batch["image"].to(dev)
            masks = batch["mask"].to(dev)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        meter = SegmentationMetricsMeter(num_classes=3)
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(dev)
                masks = batch["mask"].to(dev)
                logits = model(imgs)
                meter.update(logits, masks, compute_boundary_metrics=(epoch == epochs))

        metrics = meter.summary()
        avg_loss = total_loss / max(len(train_loader), 1)
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {avg_loss:.4f} | "
            f"mIoU: {metrics['mIoU']*100:.2f}% | mDice: {metrics['mDice']*100:.2f}%"
        )

        if metrics["mDice"] > best_dice:
            best_dice = metrics["mDice"]
            save_file = out_path / f"unet_baseline_{split_stem}_best.pth"
            torch.save(model.state_dict(), str(save_file))
            print(f"  --> Saved best U-Net model ({best_dice*100:.2f}%) to {save_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net baseline on histopathology")
    parser.add_argument("--split_file", type=str, required=True, help="Path to split JSON file")
    parser.add_argument("--data_dir", type=str, default="data/liver_primary/processed")
    parser.add_argument("--output_dir", type=str, default="results/checkpoints")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    train_unet_baseline(
        split_file=args.split_file,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
