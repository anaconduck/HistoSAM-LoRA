import sys
import os
import math
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEDSAM_DIR = Path(__file__).resolve().parent / "MedSAM"
for p in [str(PROJECT_ROOT), str(MEDSAM_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from segment_anything import sam_model_registry
from models.carafe_module import CARAFE


class LoRA_qkv(nn.Module):

    def __init__(
        self,
        qkv_linear: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.qkv = qkv_linear
        self.dim = qkv_linear.in_features
        self.r = r
        self.scaling = lora_alpha / r
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        self.lora_a_q = nn.Linear(self.dim, r, bias=False)
        self.lora_b_q = nn.Linear(r, self.dim, bias=False)

        self.lora_a_v = nn.Linear(self.dim, r, bias=False)
        self.lora_b_v = nn.Linear(r, self.dim, bias=False)

        self.reset_parameters()

        self.qkv.weight.requires_grad = False
        if self.qkv.bias is not None:
            self.qkv.bias.requires_grad = False

    def reset_parameters(self):

        nn.init.kaiming_uniform_(self.lora_a_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_q.weight)
        nn.init.kaiming_uniform_(self.lora_a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_v.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        base_qkv = self.qkv(x)

        dropped_x = self.dropout(x)
        delta_q = self.lora_b_q(self.lora_a_q(dropped_x)) * self.scaling
        delta_v = self.lora_b_v(self.lora_a_v(dropped_x)) * self.scaling

        q = base_qkv[..., : self.dim] + delta_q
        k = base_qkv[..., self.dim : 2 * self.dim]
        v = base_qkv[..., 2 * self.dim :] + delta_v

        return torch.cat([q, k, v], dim=-1)


class PromptFreeTissueBottleneck(nn.Module):

    def __init__(self, in_channels: int = 256, num_classes: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.class_embeddings = nn.Parameter(
            torch.randn(1, num_classes, in_channels) * 0.02
        )

        self.fc_q = nn.Linear(in_channels, in_channels)
        self.fc_k = nn.Linear(in_channels, in_channels)
        self.fc_v = nn.Linear(in_channels, in_channels)
        self.proj_out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=in_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape

        feat_flat = x.flatten(2).permute(0, 2, 1)

        class_tokens = self.class_embeddings.expand(B, -1, -1)

        q = self.fc_q(feat_flat)
        k = self.fc_k(class_tokens)
        v = self.fc_v(class_tokens)

        scale = 1.0 / math.sqrt(C)
        attn = torch.bmm(q, k.transpose(1, 2)) * scale
        attn = F.softmax(attn, dim=-1)

        class_context = torch.bmm(attn, v)
        class_context = class_context.permute(0, 2, 1).view(B, C, H, W)

        out = x + self.proj_out(class_context)
        return out


class ResidualConvBlock(nn.Module):

    def __init__(self, channels: int, num_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(num_groups, channels), num_channels=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(num_groups, channels), num_channels=channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class CARAFESemanticDecoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 3,
        up_kernel: int = 5,
        encoder_kernel: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.carafe1 = CARAFE(
            in_channels=in_channels,
            out_channels=128,
            scale_factor=2,
            up_kernel=up_kernel,
            encoder_kernel=encoder_kernel,
            compressed_channels=64,
        )
        self.res1 = ResidualConvBlock(128, num_groups=16)

        self.carafe2 = CARAFE(
            in_channels=128,
            out_channels=64,
            scale_factor=2,
            up_kernel=up_kernel,
            encoder_kernel=encoder_kernel,
            compressed_channels=32,
        )
        self.res2 = ResidualConvBlock(64, num_groups=8)

        self.carafe3 = CARAFE(
            in_channels=64,
            out_channels=32,
            scale_factor=2,
            up_kernel=up_kernel,
            encoder_kernel=encoder_kernel,
            compressed_channels=16,
        )
        self.res3 = ResidualConvBlock(32, num_groups=4)

        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.GELU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(
        self, x: torch.Tensor, target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        x = self.carafe1(x)
        x = self.res1(x)

        x = self.carafe2(x)
        x = self.res2(x)

        x = self.carafe3(x)
        x = self.res3(x)

        logits = self.seg_head(x)

        if target_size is not None and (
            logits.shape[2] != target_size[0] or logits.shape[3] != target_size[1]
        ):
            logits = F.interpolate(
                logits, size=target_size, mode="bilinear", align_corners=False
            )

        return logits


class BilinearSemanticDecoder(nn.Module):

    def __init__(self, in_channels: int = 256, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False)
        self.res1 = ResidualConvBlock(128, num_groups=16)

        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.res2 = ResidualConvBlock(64, num_groups=8)

        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.res3 = ResidualConvBlock(32, num_groups=4)

        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.GELU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(
        self, x: torch.Tensor, target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.res1(self.conv1(x))

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.res2(self.conv2(x))

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.res3(self.conv3(x))

        logits = self.seg_head(x)
        if target_size is not None and (
            logits.shape[2] != target_size[0] or logits.shape[3] != target_size[1]
        ):
            logits = F.interpolate(
                logits, size=target_size, mode="bilinear", align_corners=False
            )
        return logits


class NearestSemanticDecoder(nn.Module):

    def __init__(self, in_channels: int = 256, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False)
        self.res1 = ResidualConvBlock(128, num_groups=16)

        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.res2 = ResidualConvBlock(64, num_groups=8)

        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.res3 = ResidualConvBlock(32, num_groups=4)

        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.GELU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(
        self, x: torch.Tensor, target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.res1(self.conv1(x))

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.res2(self.conv2(x))

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.res3(self.conv3(x))

        logits = self.seg_head(x)
        if target_size is not None and (
            logits.shape[2] != target_size[0] or logits.shape[3] != target_size[1]
        ):
            logits = F.interpolate(
                logits, size=target_size, mode="bilinear", align_corners=False
            )
        return logits


class ConvTransposeSemanticDecoder(nn.Module):

    def __init__(self, in_channels: int = 256, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes

        self.deconv1 = nn.ConvTranspose2d(in_channels, 128, kernel_size=2, stride=2)
        self.res1 = ResidualConvBlock(128, num_groups=16)

        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.res2 = ResidualConvBlock(64, num_groups=8)

        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.res3 = ResidualConvBlock(32, num_groups=4)

        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.GELU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(
        self, x: torch.Tensor, target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        x = self.res1(self.deconv1(x))
        x = self.res2(self.deconv2(x))
        x = self.res3(self.deconv3(x))

        logits = self.seg_head(x)
        if target_size is not None and (
            logits.shape[2] != target_size[0] or logits.shape[3] != target_size[1]
        ):
            logits = F.interpolate(
                logits, size=target_size, mode="bilinear", align_corners=False
            )
        return logits


class PixelShuffleSemanticDecoder(nn.Module):

    def __init__(self, in_channels: int = 256, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(
            in_channels, 128 * 4, kernel_size=3, padding=1, bias=False
        )
        self.ps1 = nn.PixelShuffle(2)
        self.res1 = ResidualConvBlock(128, num_groups=16)

        self.conv2 = nn.Conv2d(128, 64 * 4, kernel_size=3, padding=1, bias=False)
        self.ps2 = nn.PixelShuffle(2)
        self.res2 = ResidualConvBlock(64, num_groups=8)

        self.conv3 = nn.Conv2d(64, 32 * 4, kernel_size=3, padding=1, bias=False)
        self.ps3 = nn.PixelShuffle(2)
        self.res3 = ResidualConvBlock(32, num_groups=4)

        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.GELU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(
        self, x: torch.Tensor, target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        x = self.res1(self.ps1(self.conv1(x)))
        x = self.res2(self.ps2(self.conv2(x)))
        x = self.res3(self.ps3(self.conv3(x)))

        logits = self.seg_head(x)
        if target_size is not None and (
            logits.shape[2] != target_size[0] or logits.shape[3] != target_size[1]
        ):
            logits = F.interpolate(
                logits, size=target_size, mode="bilinear", align_corners=False
            )
        return logits


def build_semantic_decoder(
    decoder_type: str = "carafe",
    in_channels: int = 256,
    num_classes: int = 3,
) -> nn.Module:

    dtype = decoder_type.lower()
    if dtype == "carafe":
        return CARAFESemanticDecoder(in_channels=in_channels, num_classes=num_classes)
    elif dtype == "bilinear":
        return BilinearSemanticDecoder(in_channels=in_channels, num_classes=num_classes)
    elif dtype == "nearest":
        return NearestSemanticDecoder(in_channels=in_channels, num_classes=num_classes)
    elif dtype in ("conv_transpose", "deconv"):
        return ConvTransposeSemanticDecoder(
            in_channels=in_channels, num_classes=num_classes
        )
    elif dtype in ("pixel_shuffle", "pixelshuffle", "subpixel"):
        return PixelShuffleSemanticDecoder(
            in_channels=in_channels, num_classes=num_classes
        )
    else:
        raise ValueError(
            f"Unknown decoder_type: {decoder_type}. Supported: 'carafe', 'bilinear', 'nearest', 'conv_transpose', 'pixel_shuffle'"
        )


class HistoSAM_LoRA(nn.Module):

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        num_classes: int = 3,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        decoder_type: str = "carafe",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.r = r
        self.lora_alpha = lora_alpha
        self.decoder_type = decoder_type

        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        self.image_encoder = sam.image_encoder

        del sam.prompt_encoder
        del sam.mask_decoder
        del sam

        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.lora_layers: List[LoRA_qkv] = []
        if self.r > 0:
            for block in self.image_encoder.blocks:
                original_qkv = block.attn.qkv
                lora_module = LoRA_qkv(
                    original_qkv,
                    r=self.r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
                block.attn.qkv = lora_module
                self.lora_layers.append(lora_module)

        self.bottleneck = PromptFreeTissueBottleneck(
            in_channels=256, num_classes=num_classes
        )

        self.decoder = build_semantic_decoder(
            decoder_type=decoder_type,
            in_channels=256,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, _, H, W = x.shape

        if H != 1024 or W != 1024:
            x_encoder = F.interpolate(
                x, size=(1024, 1024), mode="bilinear", align_corners=False
            )
        else:
            x_encoder = x

        features = self.image_encoder(x_encoder)

        features = self.bottleneck(features)

        logits = self.decoder(features, target_size=(H, W))

        return logits

    def get_trainable_parameters(self) -> List[nn.Parameter]:

        return [p for p in self.parameters() if p.requires_grad]

    def parameter_statistics(self) -> Dict[str, int]:

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        ratio = (trainable_params / total_params) * 100.0 if total_params > 0 else 0.0

        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": frozen_params,
            "trainable_percent": ratio,
        }

    def print_parameter_summary(self):

        stats = self.parameter_statistics()
        print("=" * 60)
        print(f" HistoSAM (r={self.r}, decoder={self.decoder_type}) Parameter Summary")
        print("=" * 60)
        print(f" Total Parameters      : {stats['total']:,}")
        print(
            f" Frozen Parameters     : {stats['frozen']:,} ({100 - stats['trainable_percent']:.2f}%)"
        )
        print(
            f" Trainable Parameters  : {stats['trainable']:,} ({stats['trainable_percent']:.2f}%)"
        )
        print("=" * 60)

    def save_trainable_weights(self, save_path: str):

        trainable_state = {}
        for k, v in self.state_dict().items():
            if any(name in k for name in ["lora_", "bottleneck", "decoder"]):
                trainable_state[k] = v.cpu()

        torch.save(trainable_state, save_path)
        print(
            f"[INFO] Saved trainable weights ({len(trainable_state)} tensors) to {save_path}"
        )

    def load_trainable_weights(self, load_path: str, strict: bool = False):

        state_dict = torch.load(load_path, map_location="cpu")
        msg = self.load_state_dict(state_dict, strict=strict)
        print(f"[INFO] Loaded trainable weights from {load_path}: {msg}")


def build_histo_sam_lora(
    checkpoint_path: Optional[str] = None,
    num_classes: int = 3,
    lora_rank: int = 8,
    decoder_type: str = "carafe",
) -> HistoSAM_LoRA:

    if checkpoint_path is None:
        default_p = Path(__file__).resolve().parent / "MedSAM" / "medsam_vit_b.pth"
        if default_p.exists():
            checkpoint_path = str(default_p)

    return HistoSAM_LoRA(
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        r=lora_rank,
        decoder_type=decoder_type,
    )


if __name__ == "__main__":
    print("[INFO] Instantiating HistoSAM-LoRA (prompt-free semantic segmentation)...")
    model = HistoSAM_LoRA(checkpoint_path=None, num_classes=3, r=8)
    model.print_parameter_summary()

    dummy_input = torch.randn(2, 3, 512, 512)
    print(f"[INFO] Running forward pass on dummy batch {dummy_input.shape}...")
    output = model(dummy_input)
    print(f"[INFO] Output shape: {output.shape} (Expected: [2, 3, 512, 512])")
    assert output.shape == (2, 3, 512, 512), "Shape mismatch!"
    print("[SUCCESS] HistoSAM-LoRA pipeline validated successfully!")
