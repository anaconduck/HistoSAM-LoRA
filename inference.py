import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from models.histo_sam_lora import build_histo_sam_lora
from data.dataset import HistopathologyDataset, parse_magnification
from evaluation.visualize_results import generate_clinical_composite


@torch.no_grad()
def run_inference(
    checkpoint_path: Optional[str] = None,
    raw_images_dir: str = "data/liver_primary/raw_images",
    predictions_dir: str = "results/predictions",
    visualizations_dir: str = "results/visualizations",
    magnification: Optional[str] = "20",
    img_size: int = 512,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    dev = torch.device(device)
    raw_dir = Path(raw_images_dir)
    pred_dir = Path(predictions_dir)
    vis_dir = Path(visualizations_dir)

    pred_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        list(raw_dir.glob("*.tif"))
        + list(raw_dir.glob("*.tiff"))
        + list(raw_dir.glob("*.png"))
    )
    if not image_files:
        print(f"[ERROR] No images found in {raw_dir}")
        return

    mag_filter = (
        None
        if (magnification is None or magnification.lower() == "all")
        else magnification
    )
    dataset = HistopathologyDataset(
        image_files, img_size=img_size, magnification_filter=mag_filter
    )

    print(
        f"[INFO] Running Inference on {len(dataset)} images (Magnification: {mag_filter or 'All'})..."
    )
    print(f"  - Device: {dev}")
    print(f"  - Masks destination: {pred_dir}")
    print(f"  - Visualizations destination: {vis_dir}")

    model = build_histo_sam_lora(num_classes=3, lora_rank=16, decoder_type="carafe")
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"[INFO] Loading model checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=dev)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
    else:
        print(
            f"[WARN] Checkpoint '{checkpoint_path}' not found. Using initialized weights."
        )

    model.eval()
    model.to(dev)

    report_records = []
    total_imgs = len(dataset)

    for i in range(total_imgs):
        item = dataset[i]
        img_name = item["image_name"]
        stem = Path(img_name).stem
        orig_img_path = raw_dir / img_name

        tensor_in = item["image"].unsqueeze(0).to(dev)

        logits = model(tensor_in)
        probs = F.softmax(logits, dim=1)
        conf, preds = torch.max(probs, dim=1)

        pred_mask = preds.squeeze(0).cpu().numpy().astype(np.uint8)
        conf_map = conf.squeeze(0).cpu().numpy().astype(np.float32)

        mask_save_file = pred_dir / f"{stem}_mask.npy"
        np.save(str(mask_save_file), pred_mask)

        vis_save_file = vis_dir / f"{stem}_clinical_overlay.png"
        generate_clinical_composite(
            image_path=orig_img_path,
            mask=pred_mask,
            output_path=vis_save_file,
            panel_size=img_size,
        )

        total_px = pred_mask.size
        pct_normal = float((pred_mask == 0).sum() / total_px) * 100.0
        pct_steatosis = float((pred_mask == 1).sum() / total_px) * 100.0
        pct_necrosis = float((pred_mask == 2).sum() / total_px) * 100.0
        avg_confidence = float(conf_map.mean())

        record = {
            "image": img_name,
            "patient_id": item["patient_id"],
            "magnification": item["magnification"],
            "mean_confidence": round(avg_confidence, 4),
            "normal_parenchyma_pct": round(pct_normal, 2),
            "steatosis_pct": round(pct_steatosis, 2),
            "necrosis_pct": round(pct_necrosis, 2),
            "prediction_mask": str(mask_save_file.name),
            "visualization": str(vis_save_file.name),
        }
        report_records.append(record)

        if (i + 1) % 10 == 0 or (i + 1) == total_imgs:
            print(
                f"[{i+1}/{total_imgs}] Processed {img_name} -> Steatosis: {pct_steatosis:.1f}%, Necrosis: {pct_necrosis:.1f}%"
            )

    summary_path = pred_dir.parent / "inference_summary_report.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_processed": total_imgs,
                "magnification_filter": mag_filter,
                "samples": report_records,
            },
            f,
            indent=2,
        )

    print(f"\n[SUCCESS] Inference completed!")
    print(f"  - Total processed: {total_imgs}")
    print(f"  - Summary report saved to: {summary_path}")
    print(f"  - Pathologist deliverable ready in: {vis_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Histopathology Semantic Segmentation Inference"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/checkpoints/selftrain_best.pth",
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/liver_primary/raw_images",
        help="Raw images directory",
    )
    parser.add_argument(
        "--magnification",
        type=str,
        default="20",
        help="Filter magnification ('20', 'all')",
    )
    parser.add_argument("--img-size", type=int, default=512, help="Patch size")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        raw_images_dir=args.raw_dir,
        magnification=args.magnification,
        img_size=args.img_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
