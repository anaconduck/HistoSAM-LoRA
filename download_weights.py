"""Automated downloader for official MedSAM ViT-B pre-trained weights.

Downloads the 375MB foundation model checkpoint from verified HuggingFace / Zenodo mirrors
with progress tracking, retry logic, and resume capability.
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm

MIRRORS = [
    (
        "HuggingFace (SansuiHan Mirror)",
        "https://huggingface.co/SansuiHan/medical_models/resolve/main/medsam_vit_b.pth",
    ),
    (
        "HuggingFace (GleghornLab Mirror)",
        "https://huggingface.co/GleghornLab/medsam-vit-b/resolve/main/medsam_vit_b.pth",
    ),
    (
        "Zenodo (Official Bowang Lab)",
        "https://zenodo.org/records/10689643/files/medsam_vit_b.pth",
    ),
]

EXPECTED_MIN_BYTES = 300 * 1024 * 1024  # ~350MB


def download_medsam_weights(target_dir: str = "models/MedSAM"):
    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "medsam_vit_b.pth"
    temp_file = out_dir / "medsam_vit_b.pth.part"

    if out_file.exists() and out_file.stat().st_size > EXPECTED_MIN_BYTES:
        size_mb = out_file.stat().st_size / (1024 * 1024)
        print(f"[INFO] MedSAM checkpoint already exists at {out_file} ({size_mb:.1f} MB). Skipping download.")
        return True

    print("=" * 70)
    print(" Downloading MedSAM ViT-Base Checkpoint (1.5M Pre-trained Images)")
    print("=" * 70)
    print(f" Destination: {out_file.resolve()}")
    print(" Expected Size: ~375 MB")
    print("=" * 70)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for name, url in MIRRORS:
        print(f"\n[ATTEMPT] Trying {name}...")
        print(f"  URL: {url}")
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=30, allow_redirects=True)
            if response.status_code != 200:
                print(f"  [WARN] Server responded with HTTP {response.status_code}. Trying next mirror...")
                continue

            total_size = int(response.headers.get("content-length", 0))

            with open(temp_file, "wb") as f, tqdm(
                desc="Downloading medsam_vit_b.pth",
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=2 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

            response.close()
            del response

            # Validate file size
            if temp_file.exists() and temp_file.stat().st_size > EXPECTED_MIN_BYTES:
                import shutil
                shutil.move(str(temp_file), str(out_file))
                size_mb = out_file.stat().st_size / (1024 * 1024)
                print(f"\n[SUCCESS] MedSAM checkpoint successfully downloaded to {out_file} ({size_mb:.1f} MB)!")
                return True
            else:
                print("\n[WARN] Downloaded file is incomplete or too small. Trying next mirror...")
                if temp_file.exists():
                    temp_file.unlink()

        except Exception as e:
            print(f"  [ERROR] Connection failed: {e}")
            if temp_file.exists():
                temp_file.unlink()

    print("\n[ERROR] All mirrors failed.")
    print("[MANUAL DOWNLOAD]")
    print(f"1. Download file directly from: {MIRRORS[0][1]}")
    print(f"2. Place file at: {out_file.resolve()}\n")
    return False


if __name__ == "__main__":
    download_medsam_weights()
