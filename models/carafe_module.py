import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CARAFE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        scale_factor: int = 2,
        up_kernel: int = 5,
        encoder_kernel: int = 3,
        compressed_channels: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.scale_factor = scale_factor
        self.up_kernel = up_kernel
        self.encoder_kernel = encoder_kernel

        self.compressed_channels = min(compressed_channels, in_channels)
        self.channel_compressor = nn.Conv2d(
            in_channels, self.compressed_channels, kernel_size=1, bias=False
        )

        encoder_padding = encoder_kernel // 2
        self.kernel_encoder = nn.Conv2d(
            self.compressed_channels,
            (scale_factor * up_kernel) ** 2,
            kernel_size=encoder_kernel,
            padding=encoder_padding,
            bias=False,
        )

        if self.in_channels != self.out_channels:
            self.post_conv = nn.Conv2d(
                self.in_channels, self.out_channels, kernel_size=1
            )
        else:
            self.post_conv = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in [self.channel_compressor, self.kernel_encoder]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        S = self.scale_factor
        K = self.up_kernel
        pad = K // 2

        compressed = self.channel_compressor(x)
        kernel = self.kernel_encoder(compressed)

        kernel = F.pixel_shuffle(kernel, S)
        kernel = kernel.permute(0, 2, 3, 1)
        kernel = F.softmax(kernel, dim=-1)

        x_padded = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        patches = F.unfold(x_padded, kernel_size=K, stride=1)
        patches = patches.view(B, C, K * K, H, W)

        k_reshaped = (
            kernel.view(B, H, S, W, S, K * K)
            .permute(0, 5, 2, 4, 1, 3)
            .reshape(B, K * K, S * S, H, W)
        )

        out_sub = torch.einsum("bckhw,bkmhw->bcmhw", patches, k_reshaped)

        out = (
            out_sub.view(B, C, S, S, H, W)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(B, C, H * S, W * S)
            .contiguous()
        )

        return self.post_conv(out)
