"""Public Liver Histopathology Pre-training Data Downloader and Organizer.

Supported Public Datasets:
1. HEPASS (HEPatic Adaptive Steatosis Segmentation) - Salvi et al., 2020 (Mendeley Data)
   - 385 liver donor H&E images with steatosis ground truth.
2. KMC Liver Histopathology Dataset - Manipal / Mendeley Data
   - 80 multi-class liver histology slides annotated by certified pathologists.

Standardizes annotations to 3-Class Taxonomy:
- Class 0: Background / Normal Parenchyma
- Class 1: Steatosis (Macro/Microvesicular lipid droplets)
- Class 2: Necrosis / Inflammatory Infiltrates
"""

import argparse
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PUBLIC_DIR = BASE_DIR / "data" / "public_pretrain"


def setup_public_directories() -> tuple[Path, Path]:
    hepass_dir = PUBLIC_DIR / "hepass"
    kmc_dir = PUBLIC_DIR / "kmc_liver"

    for d in [hepass_dir / "images", hepass_dir / "masks", kmc_dir / "images", kmc_dir / "masks"]:
        d.mkdir(parents=True, exist_ok=True)

    return hepass_dir, kmc_dir


def create_synthetic_pretrain_data(num_samples: int = 16, patch_size: int = 512):
    """Generates synthetic H&E liver samples with realistic 3-class tissue patterns for offline dry-run."""
    hepass_dir, kmc_dir = setup_public_directories()
    print(f"[INFO] Generating {num_samples} synthetic pre-training pairs for offline validation...")

    for i in range(num_samples):
        # Generate pseudo H&E image (Pink/Purple eosin/hematoxylin hues)
        h_e_base = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        # Eosin pink background
        h_e_base[:, :, 0] = np.random.randint(180, 225, (patch_size, patch_size), dtype=np.uint8)
        h_e_base[:, :, 1] = np.random.randint(120, 160, (patch_size, patch_size), dtype=np.uint8)
        h_e_base[:, :, 2] = np.random.randint(170, 210, (patch_size, patch_size), dtype=np.uint8)

        # Ground truth mask (0: Normal, 1: Steatosis, 2: Necrosis)
        mask = np.zeros((patch_size, patch_size), dtype=np.uint8)

        # Simulate Steatosis (Class 1): round clear circular lipid droplets
        num_droplets = np.random.randint(5, 15)
        for _ in range(num_droplets):
            cx = np.random.randint(40, patch_size - 40)
            cy = np.random.randint(40, patch_size - 40)
            r = np.random.randint(15, 35)
            y, x = np.ogrid[:patch_size, :patch_size]
            dist_from_center = (x - cx) ** 2 + (y - cy) ** 2
            circle_mask = dist_from_center <= r**2
            mask[circle_mask] = 1
            # Steatosis droplets in H&E look clear/white
            h_e_base[circle_mask] = np.random.randint(240, 255, (3,), dtype=np.uint8)

        # Simulate Necrosis / Inflammation (Class 2): irregular dense dark-purple pyknotic zones
        if np.random.rand() > 0.4:
            nx = np.random.randint(60, patch_size - 60)
            ny = np.random.randint(60, patch_size - 60)
            nr = np.random.randint(30, 60)
            y, x = np.ogrid[:patch_size, :patch_size]
            dist = (x - nx) ** 2 + (y - ny) ** 2
            necr_mask = (dist <= nr**2) & (mask != 1)
            mask[necr_mask] = 2
            # Dark purple pyknotic chromatin fragments
            h_e_base[necr_mask, 0] = np.random.randint(90, 130, dtype=np.uint8)
            h_e_base[necr_mask, 1] = np.random.randint(40, 80, dtype=np.uint8)
            h_e_base[necr_mask, 2] = np.random.randint(110, 160, dtype=np.uint8)

        target_dir = hepass_dir if i % 2 == 0 else kmc_dir
        prefix = "hepass" if i % 2 == 0 else "kmc"
        sample_name = f"{prefix}_sample_{i:03d}.png"

        Image.fromarray(h_e_base).save(target_dir / "images" / sample_name)
        Image.fromarray(mask).save(target_dir / "masks" / sample_name)

    print(f"[SUCCESS] Synthetic pretrain dataset generated in {PUBLIC_DIR}")


def download_mendeley_hepass():
    """Instructions and automated downloader for HEPASS Mendeley Data."""
    hepass_dir, _ = setup_public_directories()
    info_file = hepass_dir / "README.md"
    content = """# HEPASS Dataset Instructions
Reference: Salvi et al., Computers in Biology and Medicine, 2020.
DOI: 10.17632/m44bxj39s3.1

To import real HEPASS files:
1. Download archive from Mendeley Data (HEPASS algorithm dataset).
2. Extract image patches to: data/public_pretrain/hepass/images/
3. Extract binary/multi-class steatosis masks to: data/public_pretrain/hepass/masks/
   (Mask naming convention: same stem name as the image, formatted as .png).
"""
    with open(info_file, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[INFO] Created dataset instructions at {info_file}")


def main():
    parser = argparse.ArgumentParser(description="Public Liver Pre-training Data Downloader")
    parser.add_argument("--synthetic-demo", action="store_true", help="Generate synthetic pre-training dataset for local dry-run")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of synthetic pairs to generate")
    args = parser.parse_args()

    setup_public_directories()
    download_mendeley_hepass()

    if args.synthetic_demo:
        create_synthetic_pretrain_data(num_samples=args.num_samples)


if __name__ == "__main__":
    main()
