import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold


def parse_magnification(filename: str) -> Optional[str]:

    stem = Path(filename).stem.lower()
    m = re.search(r"(?:x(4|10|20|40)|(4|10|20|40|100|200|400)x)", stem)
    if not m:
        return None
    val = m.group(1) or m.group(2)
    if val == "200":
        return "20"
    if val == "400":
        return "40"
    if val == "100":
        return "10"
    return val


def extract_patient_id(filename: str) -> str:

    stem = Path(filename).stem

    tile_match = re.match(r"^(.*?)(?:_x\d+_y\d+)?$", stem)
    if tile_match and tile_match.group(1):
        stem = tile_match.group(1)

    pat_match = re.split(r"(?:X\d+|x\d+|-\d+|_\d+)", stem, flags=re.IGNORECASE)
    if pat_match and pat_match[0]:
        return pat_match[0]
    return stem


class HistopathologyDataset(Dataset):

    def __init__(
        self,
        image_paths: List[Path | str],
        mask_paths: Optional[List[Path | str]] = None,
        img_size: int = 512,
        normalize_type: str = "medsam",
        ignore_index: int = 255,
        magnification_filter: Optional[str] = None,
    ):
        raw_paths = [Path(p) for p in image_paths]
        if magnification_filter is not None:
            mag_str = str(magnification_filter).lower().replace("x", "")
            self.image_paths = [
                p for p in raw_paths if parse_magnification(p.name) == mag_str
            ]
            print(
                f"[INFO] Magnification filter '{mag_str}x' active: {len(self.image_paths)}/{len(raw_paths)} images retained."
            )
        else:
            self.image_paths = raw_paths

        if mask_paths is not None:
            mask_dict = {Path(m).stem: Path(m) for m in mask_paths}
            self.mask_paths = [mask_dict.get(p.stem) for p in self.image_paths]
        else:
            self.mask_paths = None

        self.img_size = img_size
        self.normalize_type = normalize_type
        self.ignore_index = ignore_index

        self.pixel_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        img_path = self.image_paths[idx]
        pil_img = Image.open(str(img_path)).convert("RGB")

        if pil_img.size != (self.img_size, self.img_size):
            pil_img = pil_img.resize((self.img_size, self.img_size), Image.BILINEAR)

        rgb = np.array(pil_img, dtype=np.float32)

        if self.normalize_type == "medsam":
            norm_rgb = (rgb - self.pixel_mean) / self.pixel_std
        else:
            norm_rgb = rgb / 255.0

        tensor_img = torch.from_numpy(norm_rgb).permute(2, 0, 1).float()

        item = {
            "image": tensor_img,
            "image_name": img_path.name,
            "patient_id": extract_patient_id(img_path.name),
            "magnification": parse_magnification(img_path.name) or "unknown",
        }

        if self.mask_paths is not None and self.mask_paths[idx] is not None:
            mask_path = self.mask_paths[idx]
            pil_mask = Image.open(str(mask_path))
            if pil_mask.size != (self.img_size, self.img_size):
                pil_mask = pil_mask.resize(
                    (self.img_size, self.img_size), Image.NEAREST
                )

            mask = np.array(pil_mask, dtype=np.int64)
            item["mask"] = torch.from_numpy(mask)

        return item


class PublicLiverDataset(Dataset):

    def __init__(
        self,
        root_dirs: List[Path | str],
        img_size: int = 512,
        normalize_type: str = "medsam",
    ):
        self.img_size = img_size
        self.normalize_type = normalize_type
        self.samples = []

        self.pixel_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        for r in root_dirs:
            r_path = Path(r)
            img_dir = r_path / "images"
            mask_dir = r_path / "masks"
            if not img_dir.exists():
                continue

            for img_p in sorted(
                list(img_dir.glob("*.png"))
                + list(img_dir.glob("*.jpg"))
                + list(img_dir.glob("*.tif"))
            ):
                mask_p = mask_dir / f"{img_p.stem}.png"
                if mask_p.exists():
                    self.samples.append((img_p, mask_p))

        print(f"[INFO] PublicLiverDataset loaded {len(self.samples)} image-mask pairs.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        img_p, mask_p = self.samples[idx]
        pil_img = Image.open(str(img_p)).convert("RGB")
        if pil_img.size != (self.img_size, self.img_size):
            pil_img = pil_img.resize((self.img_size, self.img_size), Image.BILINEAR)

        rgb = np.array(pil_img, dtype=np.float32)
        if self.normalize_type == "medsam":
            norm_rgb = (rgb - self.pixel_mean) / self.pixel_std
        else:
            norm_rgb = rgb / 255.0

        tensor_img = torch.from_numpy(norm_rgb).permute(2, 0, 1).float()

        pil_mask = Image.open(str(mask_p))
        if pil_mask.size != (self.img_size, self.img_size):
            pil_mask = pil_mask.resize((self.img_size, self.img_size), Image.NEAREST)

        mask = np.array(pil_mask, dtype=np.int64)

        return {
            "image": tensor_img,
            "mask": torch.from_numpy(mask),
            "image_name": img_p.name,
        }


class PseudoLabeledDataset(Dataset):

    def __init__(
        self,
        image_paths: List[Path | str],
        pseudo_label_dir: Path | str,
        img_size: int = 512,
        normalize_type: str = "medsam",
    ):
        self.image_paths = [Path(p) for p in image_paths]
        self.pseudo_dir = Path(pseudo_label_dir)
        self.img_size = img_size
        self.normalize_type = normalize_type

        self.pixel_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        self.valid_samples = []
        for p in self.image_paths:
            mask_p = self.pseudo_dir / f"{p.stem}_pseudolabel.png"
            conf_p = self.pseudo_dir / f"{p.stem}_confidence.npy"
            if mask_p.exists() and conf_p.exists():
                self.valid_samples.append((p, mask_p, conf_p))

        print(
            f"[INFO] PseudoLabeledDataset: {len(self.valid_samples)}/{len(self.image_paths)} pairs ready."
        )

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        img_p, mask_p, conf_p = self.valid_samples[idx]

        pil_img = Image.open(str(img_p)).convert("RGB")
        if pil_img.size != (self.img_size, self.img_size):
            pil_img = pil_img.resize((self.img_size, self.img_size), Image.BILINEAR)

        rgb = np.array(pil_img, dtype=np.float32)
        if self.normalize_type == "medsam":
            norm_rgb = (rgb - self.pixel_mean) / self.pixel_std
        else:
            norm_rgb = rgb / 255.0

        tensor_img = torch.from_numpy(norm_rgb).permute(2, 0, 1).float()

        pil_mask = Image.open(str(mask_p))
        if pil_mask.size != (self.img_size, self.img_size):
            pil_mask = pil_mask.resize((self.img_size, self.img_size), Image.NEAREST)
        tensor_mask = torch.from_numpy(np.array(pil_mask, dtype=np.int64))

        conf_map = np.load(str(conf_p))
        if conf_map.shape != (self.img_size, self.img_size):

            conf_img = Image.fromarray((conf_map * 255).astype(np.uint8))
            conf_img = conf_img.resize((self.img_size, self.img_size), Image.BILINEAR)
            conf_map = np.array(conf_img, dtype=np.float32) / 255.0

        tensor_conf = torch.from_numpy(conf_map).float()

        return {
            "image": tensor_img,
            "mask": tensor_mask,
            "confidence": tensor_conf,
            "image_name": img_p.name,
        }


def create_patient_stratified_splits(
    image_dir: Path | str,
    mask_dir: Path | str,
    output_splits_dir: Path | str,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
) -> List[Dict[str, List[str]]]:

    img_dir = Path(image_dir)
    lbl_dir = Path(mask_dir)
    out_dir = Path(output_splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        list(img_dir.glob("*.png"))
        + list(img_dir.glob("*.jpg"))
        + list(img_dir.glob("*.tif"))
    )
    valid_pairs = []

    for img_path in image_files:
        lbl_path = lbl_dir / f"{img_path.stem}.png"
        if lbl_path.exists():
            valid_pairs.append((img_path, lbl_path))

    if not valid_pairs:
        print(f"[WARN] No matched image-mask pairs found in {image_dir} and {mask_dir}")
        return []

    print(f"[INFO] Found {len(valid_pairs)} valid patch pairs for K-Fold generation.")

    patch_names = [p[0].name for p in valid_pairs]
    patients = [extract_patient_id(name) for name in patch_names]

    classes = []
    for _, lbl_p in valid_pairs:
        m = np.array(Image.open(str(lbl_p)))
        vals, counts = np.unique(m[m != 255], return_counts=True)
        dominant_cls = vals[np.argmax(counts)] if len(vals) > 0 else 0
        classes.append(dominant_cls)

    patch_names = np.array(patch_names)
    patients = np.array(patients)
    classes = np.array(classes)

    all_split_info = []

    for rep in range(n_repeats):
        current_seed = seed + rep * 100
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=current_seed
        )

        for fold, (train_idx, val_idx) in enumerate(
            sgkf.split(patch_names, classes, groups=patients)
        ):
            train_patches = patch_names[train_idx].tolist()
            val_patches = patch_names[val_idx].tolist()

            train_patients = set(patients[train_idx])
            val_patients = set(patients[val_idx])

            overlap = train_patients.intersection(val_patients)
            assert len(overlap) == 0, f"PATIENT LEAKAGE DETECTED: {overlap}!"

            split_data = {
                "repeat": rep,
                "fold": fold,
                "train_patients": sorted(list(train_patients)),
                "val_patients": sorted(list(val_patients)),
                "train_images": train_patches,
                "val_images": val_patches,
            }

            filename = out_dir / f"split_r{rep}_f{fold}.json"
            with open(filename, "w") as f:
                json.dump(split_data, f, indent=2)

            all_split_info.append(split_data)

    return all_split_info


def build_dataloaders_from_split(
    split_file: Path | str,
    data_dir: Path | str,
    batch_size: int = 4,
    num_workers: int = 0,
    img_size: int = 512,
) -> Tuple[DataLoader, DataLoader]:

    with open(split_file, "r") as f:
        split = json.load(f)

    data_path = Path(data_dir)
    img_dir = data_path / "images"
    lbl_dir = data_path / "masks"

    train_imgs = [img_dir / name for name in split["train_images"]]
    train_lbls = [lbl_dir / f"{Path(name).stem}.png" for name in split["train_images"]]

    val_imgs = [img_dir / name for name in split["val_images"]]
    val_lbls = [lbl_dir / f"{Path(name).stem}.png" for name in split["val_images"]]

    train_ds = HistopathologyDataset(train_imgs, train_lbls, img_size=img_size)
    val_ds = HistopathologyDataset(val_imgs, val_lbls, img_size=img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader
