"""Pseudo-Label and Confidence Map Generator for Self-Training.

Runs inference of HistoSAM-LoRA (pre-trained or previous iteration checkpoint)
on target histopathology images and saves:
1. Hard pseudo-labels (uint8 PNG: 0=Normal, 1=Steatosis, 2=Necrosis).
2. Softmax confidence maps (float32 NPY, range [0.0, 1.0]).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.histo_sam_lora import build_histo_sam_lora
from data.dataset import HistopathologyDataset, parse_magnification


@torch.no_grad()
def generate_pseudo_labels(
    model: torch.nn.Module,
    dataset: HistopathologyDataset,
    output_dir: Path | str,
    device: torch.device,
    batch_size: int = 4,
) -> dict:
    """Generates pseudo-labels and confidence maps for all images in dataset."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to(device)

    total_images = len(dataset)
    print(f"[INFO] Generating pseudo-labels for {total_images} images on device {device}...")

    conf_stats = []
    class_counts = {0: 0, 1: 0, 2: 0}

    for idx in range(total_images):
        item = dataset[idx]
        img_tensor = item["image"].unsqueeze(0).to(device)  # (1, 3, H, W)
        img_name = item["image_name"]
        stem = Path(img_name).stem

        # Forward pass
        logits = model(img_tensor)  # (1, C, H, W)
        probs = F.softmax(logits, dim=1)  # (1, C, H, W)

        # Max confidence and pseudo-label argmax
        conf, pseudo = torch.max(probs, dim=1)  # (1, H, W)

        pseudo_np = pseudo.squeeze(0).cpu().numpy().astype(np.uint8)
        conf_np = conf.squeeze(0).cpu().numpy().astype(np.float32)

        # Save pseudo-label mask
        mask_save_path = out_dir / f"{stem}_pseudolabel.png"
        Image.fromarray(pseudo_np).save(mask_save_path)

        # Save confidence map
        conf_save_path = out_dir / f"{stem}_confidence.npy"
        np.save(str(conf_save_path), conf_np)

        conf_stats.append(float(conf_np.mean()))
        for c in range(3):
            class_counts[c] += int((pseudo_np == c).sum())

        if (idx + 1) % 20 == 0 or (idx + 1) == total_images:
            print(f"[{idx+1}/{total_images}] Saved: {stem} (Mean Conf: {conf_np.mean():.4f})")

    avg_conf = float(np.mean(conf_stats))
    total_pixels = sum(class_counts.values())
    class_dist = {c: f"{(count / total_pixels) * 100:.2f}%" for c, count in class_counts.items()}

    summary = {
        "total_images": total_images,
        "mean_confidence": avg_conf,
        "class_distribution": class_dist,
        "output_directory": str(out_dir),
    }

    print("\n[SUMMARY] Pseudo-Label Generation Completed:")
    print(f"  - Mean Confidence across dataset: {avg_conf:.4f}")
    print(f"  - Pixel Distribution: {class_dist}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate Pseudo-Labels and Confidence Maps")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--raw-dir", type=str, default="data/liver_primary/raw_images", help="Path to raw images")
    parser.add_argument("--output-dir", type=str, default="data/liver_primary/pseudo_labels", help="Output directory")
    parser.add_argument("--magnification", type=str, default="20", help="Filter magnification (e.g. '20', or 'all')")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--img-size", type=int, default=512)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    image_files = sorted(list(raw_dir.glob("*.tif")) + list(raw_dir.glob("*.tiff")) + list(raw_dir.glob("*.png")))

    mag_filter = None if args.magnification.lower() == "all" else args.magnification
    dataset = HistopathologyDataset(image_files, img_size=args.img_size, magnification_filter=mag_filter)

    if len(dataset) == 0:
        print("[ERROR] No images found for pseudo-label generation.")
        return

    # Build HistoSAM-LoRA model
    model = build_histo_sam_lora(num_classes=3, lora_rank=16, decoder_type="carafe")

    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"[INFO] Loading checkpoint from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("[WARN] No checkpoint provided or not found; running with initialization weights.")

    generate_pseudo_labels(
        model=model,
        dataset=dataset,
        output_dir=args.output_dir,
        device=torch.device(args.device),
    )


if __name__ == "__main__":
    main()
