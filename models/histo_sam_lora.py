"""HistoSAM-LoRA: Parameter-Efficient Adaptation of Medical Foundation Models
with Morphology-Aware CARAFE Decoding for Zero-Augmented Few-Shot Hepatic Pathology.

Target: Q1 Medical Image Analysis / IEEE TMI Publication.
Design Constraints:
- Frozen MedSAM (ViT-Base) encoder for maximum knowledge transfer.
- Low-Rank Adaptation (LoRA) on Attention QKV projections (~1.5% trainable params).
- Prompt-Free Tissue Class Bottleneck (replacing interactive point/box prompts).
- CARAFE-Enhanced Semantic Decoder for amorphous tissue boundary reconstruction.
- Strict VRAM efficiency for NVIDIA RTX 5070 (12GB VRAM).
"""

import sys
import os
import math
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure workspace and MedSAM paths are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEDSAM_DIR = Path(__file__).resolve().parent / "MedSAM"
for p in [str(PROJECT_ROOT), str(MEDSAM_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from segment_anything import sam_model_registry
try:
    from models.carafe_module import CARAFE
except ImportError:
    from carafe_module import CARAFE


class LoRA_qkv(nn.Module):
    """Low-Rank Adaptation (LoRA) injected into ViT Attention QKV projection.

    Adapts Query (Q) and Value (V) projections while keeping Key (K) frozen.
    Preserves exact pre-trained representation at initialization by zero-initializing B.
    """

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

        # LoRA matrices for Query
        self.lora_a_q = nn.Linear(self.dim, r, bias=False)
        self.lora_b_q = nn.Linear(r, self.dim, bias=False)

        # LoRA matrices for Value
        self.lora_a_v = nn.Linear(self.dim, r, bias=False)
        self.lora_b_v = nn.Linear(r, self.dim, bias=False)

        self.reset_parameters()

        # Freeze the base projection
        self.qkv.weight.requires_grad = False
        if self.qkv.bias is not None:
            self.qkv.bias.requires_grad = False

    def reset_parameters(self):
        # Kaiming uniform for A, zero init for B -> Initial output is identical to base model
        nn.init.kaiming_uniform_(self.lora_a_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_q.weight)
        nn.init.kaiming_uniform_(self.lora_a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_v.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base frozen projection
        base_qkv = self.qkv(x)  # (B, H, W, 3 * dim)

        # LoRA adaptation for Q and V
        dropped_x = self.dropout(x)
        delta_q = self.lora_b_q(self.lora_a_q(dropped_x)) * self.scaling
        delta_v = self.lora_b_v(self.lora_a_v(dropped_x)) * self.scaling

        q = base_qkv[..., : self.dim] + delta_q
        k = base_qkv[..., self.dim : 2 * self.dim]
        v = base_qkv[..., 2 * self.dim :] + delta_v

        return torch.cat([q, k, v], dim=-1)


class PromptFreeTissueBottleneck(nn.Module):
    """Eliminates the requirement for manual bounding boxes/points in SAM.

    Injects learnable class-aware queries (e.g. for necrosis, normal, steatosis)
    via cross-feature semantic modulation before decoder upsampling.
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Learnable class embeddings for the target pathology categories
        self.class_embeddings = nn.Parameter(
            torch.randn(1, num_classes, in_channels) * 0.02
        )

        # Channel cross-attention / modulation
        self.fc_q = nn.Linear(in_channels, in_channels)
        self.fc_k = nn.Linear(in_channels, in_channels)
        self.fc_v = nn.Linear(in_channels, in_channels)
        self.proj_out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=in_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # Flatten spatial dimensions: (B, H*W, C)
        feat_flat = x.flatten(2).permute(0, 2, 1)

        # Expand class tokens to batch size: (B, num_classes, C)
        class_tokens = self.class_embeddings.expand(B, -1, -1)

        # Cross-attention: Spatial features query class tokens
        q = self.fc_q(feat_flat)  # (B, H*W, C)
        k = self.fc_k(class_tokens)  # (B, num_classes, C)
        v = self.fc_v(class_tokens)  # (B, num_classes, C)

        scale = 1.0 / math.sqrt(C)
        attn = torch.bmm(q, k.transpose(1, 2)) * scale  # (B, H*W, num_classes)
        attn = F.softmax(attn, dim=-1)

        class_context = torch.bmm(attn, v)  # (B, H*W, C)
        class_context = class_context.permute(0, 2, 1).view(B, C, H, W)

        # Modulate spatial features with residual connection
        out = x + self.proj_out(class_context)
        return out


class ResidualConvBlock(nn.Module):
    """Double convolution residual block with GroupNorm for small batch sizes."""

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
    """Morphology-Aware Semantic Decoder utilizing CARAFE upsampling.

    Replaces SAM's standard low-resolution interactive decoder.
    Progressively upsamples ViT features (64x64) back to full resolution (512x512)
    using Content-Aware ReAssembly of FEatures (CARAFE) for superior boundary definition
    on amorphous histology textures (necrosis, steatosis).
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 3,
        up_kernel: int = 5,
        encoder_kernel: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Stage 1: 64x64 -> 128x128 (256 -> 128)
        self.carafe1 = CARAFE(
            in_channels=in_channels,
            out_channels=128,
            scale_factor=2,
            up_kernel=up_kernel,
            encoder_kernel=encoder_kernel,
            compressed_channels=64,
        )
        self.res1 = ResidualConvBlock(128, num_groups=16)

        # Stage 2: 128x128 -> 256x256 (128 -> 64)
        self.carafe2 = CARAFE(
            in_channels=128,
            out_channels=64,
            scale_factor=2,
            up_kernel=up_kernel,
            encoder_kernel=encoder_kernel,
            compressed_channels=32,
        )
        self.res2 = ResidualConvBlock(64, num_groups=8)

        # Stage 3: 256x256 -> 512x512 (64 -> 32)
        self.carafe3 = CARAFE(
            in_channels=64,
            out_channels=32,
            scale_factor=2,
            up_kernel=up_kernel,
            encoder_kernel=encoder_kernel,
            compressed_channels=16,
        )
        self.res3 = ResidualConvBlock(32, num_groups=4)

        # Semantic classification head
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
        # x: (B, 256, 64, 64)
        x = self.carafe1(x)  # -> (B, 128, 128, 128)
        x = self.res1(x)

        x = self.carafe2(x)  # -> (B, 64, 256, 256)
        x = self.res2(x)

        x = self.carafe3(x)  # -> (B, 32, 512, 512)
        x = self.res3(x)

        logits = self.seg_head(x)  # -> (B, num_classes, 512, 512)

        if target_size is not None and (
            logits.shape[2] != target_size[0] or logits.shape[3] != target_size[1]
        ):
            logits = F.interpolate(
                logits, size=target_size, mode="bilinear", align_corners=False
            )

        return logits


class HistoSAM_LoRA(nn.Module):
    """Complete HistoSAM-LoRA Architecture for Hepatic Pathology.

    Integrates:
    1. Pretrained MedSAM Vision Transformer (ViT-Base) [FROZEN]
    2. LoRA injected attention projections [TRAINABLE]
    3. Prompt-Free Tissue Class Bottleneck [TRAINABLE]
    4. Morphology-Aware CARAFE Semantic Decoder [TRAINABLE]
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        num_classes: int = 3,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.r = r
        self.lora_alpha = lora_alpha

        # 1. Initialize MedSAM ViT-B
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        self.image_encoder = sam.image_encoder

        # Free memory: prompt_encoder and mask_decoder are not needed
        del sam.prompt_encoder
        del sam.mask_decoder
        del sam

        # 2. Freeze the entire MedSAM ViT encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        # 3. Inject LoRA into all attention blocks
        self.lora_layers: List[LoRA_qkv] = []
        for block in self.image_encoder.blocks:
            original_qkv = block.attn.qkv
            lora_module = LoRA_qkv(
                original_qkv,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            block.attn.qkv = lora_module
            self.lora_layers.append(lora_module)

        # 4. Prompt-Free Bottleneck
        self.bottleneck = PromptFreeTissueBottleneck(
            in_channels=256, num_classes=num_classes
        )

        # 5. CARAFE-Enhanced Semantic Decoder
        self.decoder = CARAFESemanticDecoder(
            in_channels=256,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input RGB tensor of shape (B, 3, H, W), typically normalized [0, 1].

        Returns:
            Logits tensor of shape (B, num_classes, H, W).
        """
        B, _, H, W = x.shape

        # MedSAM ViT expects 1024x1024 resolution.
        # If patch size is 512x512, interpolate to 1024x1024 for the encoder pass.
        if H != 1024 or W != 1024:
            x_encoder = F.interpolate(
                x, size=(1024, 1024), mode="bilinear", align_corners=False
            )
        else:
            x_encoder = x

        # ViT Encoder Feature Extraction
        features = self.image_encoder(x_encoder)  # (B, 256, 64, 64)

        # Prompt-Free Class Bottleneck Modulation
        features = self.bottleneck(features)  # (B, 256, 64, 64)

        # CARAFE Upsampling to target patch resolution
        logits = self.decoder(features, target_size=(H, W))  # (B, num_classes, H, W)

        return logits

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Returns all parameters with requires_grad=True."""
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_statistics(self) -> Dict[str, int]:
        """Calculates total, trainable, and frozen parameter counts."""
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
        """Prints a clean summary of model parameters."""
        stats = self.parameter_statistics()
        print("=" * 60)
        print(" HistoSAM-LoRA Parameter Summary (Target: RTX 5070 12GB VRAM)")
        print("=" * 60)
        print(f" Total Parameters      : {stats['total']:,}")
        print(f" Frozen Parameters     : {stats['frozen']:,} ({100 - stats['trainable_percent']:.2f}%)")
        print(f" Trainable Parameters  : {stats['trainable']:,} ({stats['trainable_percent']:.2f}%)")
        print("=" * 60)

    def save_trainable_weights(self, save_path: str):
        """Saves ONLY trainable parameters (LoRA + Bottleneck + Decoder).

        Saves ~5MB file instead of full 350MB+ weights.
        """
        trainable_state = {}
        for k, v in self.state_dict().items():
            if any(name in k for name in ["lora_", "bottleneck", "decoder"]):
                trainable_state[k] = v.cpu()

        torch.save(trainable_state, save_path)
        print(f"[INFO] Saved trainable weights ({len(trainable_state)} tensors) to {save_path}")

    def load_trainable_weights(self, load_path: str, strict: bool = False):
        """Loads previously trained LoRA and Decoder weights."""
        state_dict = torch.load(load_path, map_location="cpu")
        msg = self.load_state_dict(state_dict, strict=strict)
        print(f"[INFO] Loaded trainable weights from {load_path}: {msg}")


def build_histo_sam_lora(
    checkpoint_path: Optional[str] = None,
    num_classes: int = 3,
    lora_rank: int = 8,
    decoder_type: str = "carafe",
) -> HistoSAM_LoRA:
    """Convenience builder function for HistoSAM-LoRA architecture."""
    # Check default MedSAM weights location if not specified
    if checkpoint_path is None:
        default_p = Path(__file__).resolve().parent / "MedSAM" / "medsam_vit_b.pth"
        if default_p.exists():
            checkpoint_path = str(default_p)

    return HistoSAM_LoRA(
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        r=lora_rank,
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
