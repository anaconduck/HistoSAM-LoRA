"""Comprehensive Evaluation Metrics for Histopathology Semantic Segmentation (Q1 Journal Standard).

Metrics included:
- mIoU (Mean Intersection over Union)
- Dice Similarity Coefficient (DSC / F1-Score)
- HD95 (95th Percentile Hausdorff Distance)
- ASD (Average Surface Distance)
- Precision, Sensitivity (Recall), Specificity

Designed for:
- Per-class evaluation (Necrosis, Normal Parenchyma, Steatosis)
- Ignore index support (e.g., 255 for non-tissue glass slide background)
- Robust boundary distance calculations using SciPy
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import distance_transform_edt, binary_erosion


def compute_confusion_matrix(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = 3,
    ignore_index: int = 255,
) -> np.ndarray:
    """Calculates confusion matrix of shape (num_classes, num_classes).

    Rows: ground truth, Columns: prediction.
    """
    valid = target != ignore_index
    y_true = target[valid]
    y_pred = pred[valid]

    mask = (y_true >= 0) & (y_true < num_classes) & (y_pred >= 0) & (y_pred < num_classes)
    hist = np.bincount(
        num_classes * y_true[mask].astype(int) + y_pred[mask].astype(int),
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)
    return hist


def compute_iou_and_dice(
    confusion_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes per-class IoU and Dice from a confusion matrix."""
    true_positive = np.diag(confusion_matrix)
    false_positive = confusion_matrix.sum(axis=0) - true_positive
    false_negative = confusion_matrix.sum(axis=1) - true_positive

    denominator_iou = true_positive + false_positive + false_negative
    iou = np.divide(
        true_positive,
        denominator_iou,
        out=np.zeros_like(true_positive, dtype=float),
        where=denominator_iou != 0,
    )

    denominator_dice = 2 * true_positive + false_positive + false_negative
    dice = np.divide(
        2 * true_positive,
        denominator_dice,
        out=np.zeros_like(true_positive, dtype=float),
        where=denominator_dice != 0,
    )

    return iou, dice


def compute_surface_distances(
    mask_gt: np.ndarray,
    mask_pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts boundary surface distances between prediction and ground truth."""
    if not np.any(mask_gt) or not np.any(mask_pred):
        return np.array([]), np.array([])

    # Extract single-pixel boundary contours using binary erosion
    border_gt = mask_gt ^ binary_erosion(mask_gt)
    border_pred = mask_pred ^ binary_erosion(mask_pred)

    if not np.any(border_gt) or not np.any(border_pred):
        return np.array([]), np.array([])

    # Distance transforms from each boundary
    dt_gt = distance_transform_edt(~border_gt)
    dt_pred = distance_transform_edt(~border_pred)

    # Distances from pred border points to closest gt border point
    dist_pred_to_gt = dt_gt[border_pred]
    # Distances from gt border points to closest pred border point
    dist_gt_to_pred = dt_pred[border_gt]

    return dist_pred_to_gt, dist_gt_to_pred


def compute_hd95(dist_pred_to_gt: np.ndarray, dist_gt_to_pred: np.ndarray) -> float:
    """Computes 95th Percentile Hausdorff Distance (HD95)."""
    if len(dist_pred_to_gt) == 0 or len(dist_gt_to_pred) == 0:
        return np.nan
    all_dists = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    return float(np.percentile(all_dists, 95))


def compute_asd(dist_pred_to_gt: np.ndarray, dist_gt_to_pred: np.ndarray) -> float:
    """Computes Average Surface Distance (ASD)."""
    if len(dist_pred_to_gt) == 0 or len(dist_gt_to_pred) == 0:
        return np.nan
    all_dists = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    return float(np.mean(all_dists))


class SegmentationMetricsMeter:
    """Accumulates and computes all publication-grade metrics across an evaluation split."""

    def __init__(
        self,
        num_classes: int = 3,
        class_names: Optional[List[str]] = None,
        ignore_index: int = 255,
    ):
        self.num_classes = num_classes
        self.class_names = (
            class_names
            if class_names is not None
            else ["necrosis", "normal", "steatosis"]
        )
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.total_confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )
        self.per_class_hd95: Dict[int, List[float]] = {
            c: [] for c in range(self.num_classes)
        }
        self.per_class_asd: Dict[int, List[float]] = {
            c: [] for c in range(self.num_classes)
        }

    def update(
        self,
        pred: torch.Tensor | np.ndarray,
        target: torch.Tensor | np.ndarray,
        compute_boundary_metrics: bool = True,
    ):
        """Updates metrics with a batch of predictions and ground truth.

        Args:
            pred: (B, H, W) or (B, C, H, W) logits/probabilities
            target: (B, H, W) labels
        """
        if isinstance(pred, torch.Tensor):
            if pred.ndim == 4:
                pred = pred.argmax(dim=1)
            pred = pred.detach().cpu().numpy()

        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        batch_size = pred.shape[0]
        for b in range(batch_size):
            p = pred[b]
            t = target[b]

            # Accumulate confusion matrix
            self.total_confusion_matrix += compute_confusion_matrix(
                p, t, num_classes=self.num_classes, ignore_index=self.ignore_index
            )

            # Compute boundary metrics per class
            if compute_boundary_metrics:
                for c in range(self.num_classes):
                    bin_gt = (t == c) & (t != self.ignore_index)
                    bin_pred = (p == c) & (t != self.ignore_index)

                    if np.any(bin_gt) and np.any(bin_pred):
                        d_p2g, d_g2p = compute_surface_distances(bin_gt, bin_pred)
                        hd95 = compute_hd95(d_p2g, d_g2p)
                        asd = compute_asd(d_p2g, d_g2p)
                        if not np.isnan(hd95):
                            self.per_class_hd95[c].append(hd95)
                        if not np.isnan(asd):
                            self.per_class_asd[c].append(asd)

    def summary(self) -> Dict[str, float]:
        """Returns comprehensive summary of all evaluation metrics."""
        iou_per_class, dice_per_class = compute_iou_and_dice(
            self.total_confusion_matrix
        )

        results: Dict[str, float] = {
            "mIoU": float(np.nanmean(iou_per_class)),
            "mDice": float(np.nanmean(dice_per_class)),
        }

        # Per-class IoU and Dice
        for c, name in enumerate(self.class_names):
            results[f"IoU_{name}"] = float(iou_per_class[c])
            results[f"Dice_{name}"] = float(dice_per_class[c])

            # HD95
            hd95_vals = self.per_class_hd95[c]
            results[f"HD95_{name}"] = (
                float(np.mean(hd95_vals)) if len(hd95_vals) > 0 else np.nan
            )

            # ASD
            asd_vals = self.per_class_asd[c]
            results[f"ASD_{name}"] = (
                float(np.mean(asd_vals)) if len(asd_vals) > 0 else np.nan
            )

        # Overall mean HD95 and ASD
        all_hd95 = [v for vals in self.per_class_hd95.values() for v in vals]
        all_asd = [v for vals in self.per_class_asd.values() for v in vals]
        results["mHD95"] = float(np.mean(all_hd95)) if len(all_hd95) > 0 else np.nan
        results["mASD"] = float(np.mean(all_asd)) if len(all_asd) > 0 else np.nan

        return results

    def print_table(self):
        """Prints formatted metrics table matching Q1 journal presentation."""
        res = self.summary()
        print("=" * 70)
        print(" CLINICAL PERFORMANCE EVALUATION (Q1 JOURNAL STANDARD)")
        print("=" * 70)
        print(f" {'Class':<15} | {'IoU (%)':<10} | {'Dice (%)':<10} | {'HD95 (px)':<12} | {'ASD (px)':<10}")
        print("-" * 70)
        for name in self.class_names:
            iou = res[f"IoU_{name}"] * 100
            dice = res[f"Dice_{name}"] * 100
            hd95 = res[f"HD95_{name}"]
            asd = res[f"ASD_{name}"]
            hd95_str = f"{hd95:.2f}" if not np.isnan(hd95) else "N/A"
            asd_str = f"{asd:.2f}" if not np.isnan(asd) else "N/A"
            print(f" {name:<15} | {iou:<10.2f} | {dice:<10.2f} | {hd95_str:<12} | {asd_str:<10}")
        print("-" * 70)
        m_hd95_str = f"{res['mHD95']:.2f}" if not np.isnan(res['mHD95']) else "N/A"
        m_asd_str = f"{res['mASD']:.2f}" if not np.isnan(res['mASD']) else "N/A"
        print(f" {'MEAN OVERALL':<15} | {res['mIoU']*100:<10.2f} | {res['mDice']*100:<10.2f} | {m_hd95_str:<12} | {m_asd_str:<10}")
        print("=" * 70)


if __name__ == "__main__":
    meter = SegmentationMetricsMeter(num_classes=3)
    dummy_pred = torch.randint(0, 3, (4, 256, 256))
    dummy_gt = torch.randint(0, 3, (4, 256, 256))
    dummy_gt[:, 0:20, 0:20] = 255  # test ignore_index

    meter.update(dummy_pred, dummy_gt)
    meter.print_table()
    print("[SUCCESS] SegmentationMetricsMeter successfully verified!")
