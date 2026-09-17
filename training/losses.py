import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class MultiClassFocalLoss(nn.Module):

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha.to(logits.device) if self.alpha is not None else None,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        pt = torch.exp(-ce_loss)

        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            valid_mask = targets != self.ignore_index
            if valid_mask.sum() > 0:
                return focal_loss[valid_mask].mean()
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class MultiClassDiceLoss(nn.Module):

    def __init__(
        self,
        smooth: float = 1e-5,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        valid_mask = targets != self.ignore_index

        targets_clamped = targets.clone()
        targets_clamped[~valid_mask] = 0
        one_hot = F.one_hot(targets_clamped, num_classes=num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()

        valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(probs)
        probs = probs * valid_mask_expanded
        one_hot = one_hot * valid_mask_expanded

        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dim=dims)
        cardinality = torch.sum(probs + one_hot, dim=dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (
            cardinality + self.smooth
        )
        dice_loss = 1.0 - dice_per_class.mean()

        return dice_loss


class BoundaryLaplacianLoss(nn.Module):

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        laplacian_kernel = (
            torch.tensor(
                [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.register_buffer("kernel", laplacian_kernel)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        targets_clean = targets.clone()
        targets_clean[targets == 255] = 0
        one_hot = (
            F.one_hot(targets_clean, num_classes=num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )

        B, C, H, W = probs.shape
        kernel = self.kernel.to(logits.device).repeat(C, 1, 1, 1)

        gt_edges = torch.abs(F.conv2d(one_hot, kernel, padding=1, groups=C))
        pred_edges = torch.abs(F.conv2d(probs, kernel, padding=1, groups=C))

        return F.l1_loss(pred_edges, gt_edges)


class BoundaryAwareJointLoss(nn.Module):

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        lambda_focal: float = 1.0,
        lambda_dice: float = 1.0,
        lambda_boundary: float = 0.2,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice
        self.lambda_boundary = lambda_boundary

        self.focal_loss = MultiClassFocalLoss(
            gamma=gamma, alpha=alpha, ignore_index=ignore_index
        )
        self.dice_loss = MultiClassDiceLoss(ignore_index=ignore_index)
        self.boundary_loss = BoundaryLaplacianLoss()

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        l_focal = self.focal_loss(logits, targets)
        l_dice = self.dice_loss(logits, targets)
        l_boundary = self.boundary_loss(logits, targets)

        total_loss = (
            self.lambda_focal * l_focal
            + self.lambda_dice * l_dice
            + self.lambda_boundary * l_boundary
        )

        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_focal": l_focal.item(),
            "loss_dice": l_dice.item(),
            "loss_boundary": l_boundary.item(),
        }
        return total_loss, loss_dict


class ConfidenceWeightedLoss(nn.Module):

    def __init__(
        self,
        threshold: float = 0.85,
        soft_weighting: bool = False,
        lambda_ce: float = 1.0,
        lambda_dice: float = 0.5,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.threshold = threshold
        self.soft_weighting = soft_weighting
        self.lambda_ce = lambda_ce
        self.lambda_dice = lambda_dice
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        pseudo_targets: torch.Tensor,
        confidence_map: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:

        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        ce_pixel = F.cross_entropy(
            logits, pseudo_targets, ignore_index=self.ignore_index, reduction="none"
        )

        confident_mask = (confidence_map >= self.threshold) & (
            pseudo_targets != self.ignore_index
        )
        num_confident = confident_mask.sum().float()

        if num_confident == 0:

            loss_ce = ce_pixel.mean() * 0.0
        else:
            if self.soft_weighting:
                weights = confidence_map * confident_mask.float()
                loss_ce = (ce_pixel * weights).sum() / (weights.sum() + 1e-6)
            else:
                loss_ce = ce_pixel[confident_mask].mean()

        one_hot = F.one_hot(
            pseudo_targets.clamp(0, num_classes - 1), num_classes=num_classes
        )
        one_hot = one_hot.permute(0, 3, 1, 2).float()

        mask_4d = confident_mask.unsqueeze(1).expand_as(probs)
        conf_probs = probs * mask_4d
        conf_targets = one_hot * mask_4d

        intersection = torch.sum(conf_probs * conf_targets, dim=(0, 2, 3))
        cardinality = torch.sum(conf_probs + conf_targets, dim=(0, 2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + 1e-5) / (cardinality + 1e-5)).mean()

        total_loss = self.lambda_ce * loss_ce + self.lambda_dice * dice_loss

        stats = {
            "loss_total": total_loss.item(),
            "loss_ce": loss_ce.item(),
            "loss_dice": dice_loss.item(),
            "confident_ratio": (num_confident / confidence_map.numel()).item(),
        }

        return total_loss, stats


if __name__ == "__main__":
    B, C, H, W = 2, 3, 128, 128
    logits = torch.randn(B, C, H, W, requires_grad=True)
    targets = torch.randint(0, C, (B, H, W))
    targets[0, 0:10, 0:10] = 255

    criterion = BoundaryAwareJointLoss()
    loss, details = criterion(logits, targets)
    loss.backward()

    print("[SUCCESS] Joint loss computed:", details)
    print("[SUCCESS] Gradients passed back successfully!")

    conf = torch.rand(B, H, W)
    self_crit = ConfidenceWeightedLoss(threshold=0.8)
    s_loss, s_details = self_crit(logits, targets, conf)
    print("[SUCCESS] ConfidenceWeightedLoss computed:", s_details)
