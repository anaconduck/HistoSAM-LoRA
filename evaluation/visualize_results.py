import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLASS_COLORS = {
    0: (46, 204, 113),
    1: (241, 196, 15),
    2: (231, 76, 60),
}

CLASS_NAMES = {
    0: "Normal Parenchyma",
    1: "Steatosis (Fat Droplets)",
    2: "Necrosis / Infiltrates",
}


def create_colored_mask(
    mask: np.ndarray, alpha: float = 0.45
) -> Tuple[Image.Image, Dict[int, float]]:

    h, w = mask.shape
    rgba_img = np.zeros((h, w, 4), dtype=np.uint8)

    total_pixels = h * w
    percentages = {}

    for cls_idx, color in CLASS_COLORS.items():
        cls_mask = mask == cls_idx
        count = int(cls_mask.sum())
        percentages[cls_idx] = (count / total_pixels) * 100.0

        if cls_idx == 0:

            rgba_img[cls_mask] = (*color, int(alpha * 255 * 0.3))
        else:
            rgba_img[cls_mask] = (*color, int(alpha * 255))

    return Image.fromarray(rgba_img, "RGBA"), percentages


def generate_clinical_composite(
    image_path: Path | str,
    mask: np.ndarray,
    output_path: Path | str,
    alpha: float = 0.45,
    panel_size: int = 512,
):

    img_p = Path(image_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    resample_bilinear = getattr(Image, "Resampling", Image).BILINEAR
    resample_nearest = getattr(Image, "Resampling", Image).NEAREST

    raw_pil = Image.open(str(img_p)).convert("RGB")
    raw_resized = raw_pil.resize((panel_size, panel_size), resample_bilinear)

    if mask.shape != (panel_size, panel_size):
        pil_m = Image.fromarray(mask.astype(np.uint8))
        pil_m = pil_m.resize((panel_size, panel_size), resample_nearest)
        mask = np.array(pil_m)

    colored_overlay, percentages = create_colored_mask(mask, alpha=alpha)

    blended = raw_resized.copy().convert("RGBA")
    blended.alpha_composite(colored_overlay)
    blended_rgb = blended.convert("RGB")

    header_h = 70
    canvas_w = panel_size * 2 + 30
    canvas_h = panel_size + header_h + 50
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(25, 28, 36))
    draw = ImageDraw.Draw(canvas)

    canvas.paste(raw_resized, (10, header_h))
    canvas.paste(blended_rgb, (panel_size + 20, header_h))

    draw.text((15, 12), f"Original H&E: {img_p.name}", fill=(240, 240, 240))
    draw.text(
        (panel_size + 25, 12),
        "HistoSAM-LoRA Semantic Segmentation",
        fill=(240, 240, 240),
    )

    draw.text((15, header_h - 22), "Source Tissue", fill=(180, 180, 180))
    draw.text(
        (panel_size + 25, header_h - 22), "Diagnostic Overlay", fill=(180, 180, 180)
    )

    legend_y = canvas_h - 40
    curr_x = 20
    for cls_idx in [1, 2, 0]:
        color = CLASS_COLORS[cls_idx]
        name = CLASS_NAMES[cls_idx]
        pct = percentages[cls_idx]

        draw.rectangle(
            [curr_x, legend_y, curr_x + 18, legend_y + 18],
            fill=color,
            outline=(255, 255, 255),
        )
        draw.text(
            (curr_x + 26, legend_y + 2), f"{name}: {pct:.1f}%", fill=(230, 230, 230)
        )
        curr_x += 280

    canvas.save(str(out_p), quality=95)


def batch_visualize_predictions(
    raw_image_dir: Path | str,
    predictions_dir: Path | str,
    output_dir: Path | str,
    magnification: Optional[str] = None,
):

    raw_p = Path(raw_image_dir)
    pred_p = Path(predictions_dir)
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    images = sorted(
        list(raw_p.glob("*.tif"))
        + list(raw_p.glob("*.tiff"))
        + list(raw_p.glob("*.png"))
    )
    count = 0

    for img in images:
        stem = img.stem

        npy_pred = pred_p / f"{stem}_mask.npy"
        png_pred = pred_p / f"{stem}_pseudolabel.png"

        mask = None
        if npy_pred.exists():
            mask = np.load(str(npy_pred))
        elif png_pred.exists():
            mask = np.array(Image.open(str(png_pred)))

        if mask is not None:
            save_dest = out_p / f"{stem}_clinical_overlay.png"
            generate_clinical_composite(img, mask, save_dest)
            count += 1

    print(f"[INFO] Generated {count} clinical visualizations in {out_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical Visualization Generator")
    parser.add_argument("--image", type=str, required=False, help="Single image path")
    parser.add_argument(
        "--mask", type=str, required=False, help="Single mask path (.npy or .png)"
    )
    parser.add_argument(
        "--output", type=str, default="results/visualizations/sample_overlay.png"
    )
    args = parser.parse_args()

    if args.image and args.mask:
        m = (
            np.load(args.mask)
            if args.mask.endswith(".npy")
            else np.array(Image.open(args.mask))
        )
        generate_clinical_composite(args.image, m, args.output)
        print(f"[SUCCESS] Saved visualization to {args.output}")
    else:

        dummy_img = Image.fromarray(
            np.random.randint(150, 240, (512, 512, 3), dtype=np.uint8)
        )
        dummy_m = np.zeros((512, 512), dtype=np.uint8)
        dummy_m[100:200, 100:200] = 1
        dummy_m[250:380, 250:380] = 2

        out_test = Path("results/visualizations/test_sample_overlay.png")
        temp_img_p = Path("results/visualizations/temp_dummy.png")
        temp_img_p.parent.mkdir(parents=True, exist_ok=True)
        dummy_img.save(temp_img_p)

        generate_clinical_composite(temp_img_p, dummy_m, out_test)
        temp_img_p.unlink()
        print(f"[SUCCESS] Visualizer self-test verified: {out_test}")
