"""Tiling and annotation extraction for histopathology slides (MedSAM semantic segmentation).

Supports QuPath GeoJSON exports and tiles slides into 512x512 patches with semantic masks.
Uses pure Pillow and NumPy (no OpenCV or Shapely dependencies required).
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from stain_norm import MacenkoNormalizer


CLASS_MAP = {
    "necrosis": 0,
    "necrotic": 0,
    "nekrosis": 0,
    "normal": 1,
    "hepatocyte": 1,
    "normal_parenchyma": 1,
    "steatosis": 2,
    "fatty": 2,
    "perlemakan": 2,
    "lipid": 2,
}


def is_background_patch(
    patch_rgb: np.ndarray, bg_threshold: float = 220.0, max_bg_ratio: float = 0.85
) -> bool:
    """Detects if a patch is mostly glass slide background."""
    # Fast luminance calculation: 0.299 R + 0.587 G + 0.114 B
    gray = (
        0.299 * patch_rgb[:, :, 0]
        + 0.587 * patch_rgb[:, :, 1]
        + 0.114 * patch_rgb[:, :, 2]
    )
    bg_ratio = np.mean(gray > bg_threshold)
    return bg_ratio > max_bg_ratio


def parse_qupath_geojson(geojson_path: str) -> List[Tuple[List[Tuple[float, float]], int]]:
    """Extracts polygons and class labels from a QuPath GeoJSON export."""
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    polygons = []
    features = data.get("features", []) if isinstance(data, dict) else data

    for feat in features:
        props = feat.get("properties", {})
        classification = props.get("classification", {})
        class_name = (
            classification.get("name", "").lower()
            if isinstance(classification, dict)
            else str(classification).lower()
        )

        matched_class = None
        for key, val in CLASS_MAP.items():
            if key in class_name:
                matched_class = val
                break

        if matched_class is None:
            continue

        geom = feat.get("geometry", {})
        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])

        if gtype == "Polygon":
            if coords and len(coords[0]) >= 3:
                pts = [(pt[0], pt[1]) for pt in coords[0]]
                polygons.append((pts, matched_class))
        elif gtype == "MultiPolygon":
            for poly_coords in coords:
                if poly_coords and len(poly_coords[0]) >= 3:
                    pts = [(pt[0], pt[1]) for pt in poly_coords[0]]
                    polygons.append((pts, matched_class))

    return polygons


def polygon_overlaps_box(
    poly_pts: List[Tuple[float, float]], x0: int, y0: int, x1: int, y1: int
) -> bool:
    """Checks if polygon bounding box intersects the patch box."""
    xs = [p[0] for p in poly_pts]
    ys = [p[1] for p in poly_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    return not (max_x < x0 or min_x > x1 or max_y < y0 or min_y > y1)


def tile_image_and_annotations(
    img_path: Path,
    annotation_path: Optional[Path],
    output_img_dir: Path,
    output_lbl_dir: Path,
    patch_size: int = 512,
    normalize_stain: bool = True,
    normalizer: Optional[MacenkoNormalizer] = None,
    ignore_index: int = 255,
) -> int:
    try:
        pil_img = Image.open(str(img_path)).convert("RGB")
    except Exception as e:
        print(f"[WARN] Unable to read image {img_path}: {e}")
        return 0

    W, H = pil_img.size
    img_np = np.array(pil_img)

    polygons_with_class = []
    if annotation_path and annotation_path.exists():
        polygons_with_class = parse_qupath_geojson(str(annotation_path))

    patch_count = 0
    base_name = img_path.stem

    for y in range(0, H - patch_size + 1, patch_size):
        for x in range(0, W - patch_size + 1, patch_size):
            patch_rgb = img_np[y : y + patch_size, x : x + patch_size]

            if is_background_patch(patch_rgb):
                continue

            if normalize_stain and normalizer is not None:
                patch_rgb = normalizer.transform(patch_rgb)

            # Initialize semantic mask with ignore_index (255)
            mask_pil = Image.new("L", (patch_size, patch_size), color=ignore_index)
            draw = ImageDraw.Draw(mask_pil)

            has_annotation = False
            for poly_pts, cls_idx in polygons_with_class:
                if not polygon_overlaps_box(poly_pts, x, y, x + patch_size, y + patch_size):
                    continue

                local_pts = [(p[0] - x, p[1] - y) for p in poly_pts]
                draw.polygon(local_pts, fill=int(cls_idx))
                has_annotation = True

            # Save patch if it contains any pathology annotation
            if has_annotation:
                patch_filename = f"{base_name}_x{x}_y{y}.png"

                # Save RGB patch
                patch_img_out = output_img_dir / patch_filename
                Image.fromarray(patch_rgb).save(str(patch_img_out))

                # Save 8-bit single-channel label mask
                patch_lbl_out = output_lbl_dir / patch_filename
                mask_pil.save(str(patch_lbl_out))

                patch_count += 1

    return patch_count


def process_dataset(
    raw_images_dir: str,
    raw_annotations_dir: Optional[str],
    output_dir: str,
    patch_size: int = 512,
    normalize_stain: bool = True,
):
    raw_img_path = Path(raw_images_dir)
    raw_ann_path = Path(raw_annotations_dir) if raw_annotations_dir else None
    out_path = Path(output_dir)

    out_img_dir = out_path / "images"
    out_lbl_dir = out_path / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    normalizer = MacenkoNormalizer() if normalize_stain else None

    image_files = (
        list(raw_img_path.glob("*.png"))
        + list(raw_img_path.glob("*.jpg"))
        + list(raw_img_path.glob("*.tif*"))
    )
    print(f"[INFO] Found {len(image_files)} raw images in {raw_img_path}")

    total_patches = 0
    for img_file in tqdm(image_files, desc="Tiling images"):
        ann_file = None
        if raw_ann_path:
            cand_ann = raw_ann_path / f"{img_file.stem}.geojson"
            if cand_ann.exists():
                ann_file = cand_ann

        count = tile_image_and_annotations(
            img_file,
            ann_file,
            out_img_dir,
            out_lbl_dir,
            patch_size=patch_size,
            normalize_stain=normalize_stain,
            normalizer=normalizer,
        )
        total_patches += count

    print(f"[INFO] Successfully generated {total_patches} patches in {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tile histopathology images into patches for Semantic Segmentation (MedSAM)"
    )
    parser.add_argument(
        "--raw_images",
        type=str,
        default="data/liver_primary/raw_images",
        help="Path to raw slide images",
    )
    parser.add_argument(
        "--raw_annotations",
        type=str,
        default="data/liver_primary/raw_annotations",
        help="Path to QuPath GeoJSON annotations",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/liver_primary/processed",
        help="Output directory",
    )
    parser.add_argument("--patch_size", type=int, default=512, help="Patch size")
    parser.add_argument(
        "--no_stain_norm", action="store_true", help="Disable Macenko stain normalization"
    )

    args = parser.parse_args()
    process_dataset(
        raw_images_dir=args.raw_images,
        raw_annotations_dir=args.raw_annotations,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        normalize_stain=not args.no_stain_norm,
    )
