"""Lightweight 4-level U-Net for brain tumor segmentation.

Architecture:
  Encoder: 4 levels (1→64→128→256→512) with 2-conv blocks + MaxPool
  Bottleneck: 512→1024
  Decoder: 4 levels (1024+512→512, ..., 128+64→64) with skip connections
  Output: 1-channel sigmoid map (binary segmentation)

Input: single-channel 256×256 MRI slice in [-1, 1]
Output: single-channel binary probability map in [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive Conv3x3 → BatchNorm → ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    """MaxPool → DoubleConv encoder stage."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Bilinear upsample → concatenate skip → DoubleConv decoder stage."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample to match skip connection spatial size
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """4-level U-Net for single-channel brain tumor segmentation.

    Args:
        in_channels: Number of input channels (1 for grayscale MRI).
        base_features: Number of features in the first encoder block.

    Example:
        >>> net = UNet(in_channels=1, base_features=64)
        >>> x = torch.zeros(2, 1, 256, 256)
        >>> out = net(x)   # shape: (2, 1, 256, 256), values in [0, 1]
    """

    def __init__(self, in_channels: int = 1, base_features: int = 64) -> None:
        super().__init__()
        f = base_features

        # Encoder
        self.enc1 = DoubleConv(in_channels, f)       # 256 → 256, f ch
        self.enc2 = DownBlock(f, f * 2)              # 256 → 128, 2f ch
        self.enc3 = DownBlock(f * 2, f * 4)          # 128 → 64,  4f ch
        self.enc4 = DownBlock(f * 4, f * 8)          # 64  → 32,  8f ch

        # Bottleneck
        self.bottleneck = DownBlock(f * 8, f * 16)   # 32  → 16,  16f ch

        # Decoder
        self.dec4 = UpBlock(f * 16 + f * 8, f * 8)  # 16  → 32,  8f ch
        self.dec3 = UpBlock(f * 8 + f * 4, f * 4)   # 32  → 64,  4f ch
        self.dec2 = UpBlock(f * 4 + f * 2, f * 2)   # 64  → 128, 2f ch
        self.dec1 = UpBlock(f * 2 + f, f)            # 128 → 256, f ch

        self.output_conv = nn.Conv2d(f, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, 1, H, W) in [-1, 1].

        Returns:
            Segmentation probability map (B, 1, H, W) in [0, 1].
        """
        # Encoder (save skip connections)
        s1 = self.enc1(x)     # (B, f,    H,   W)
        s2 = self.enc2(s1)    # (B, 2f,   H/2, W/2)
        s3 = self.enc3(s2)    # (B, 4f,   H/4, W/4)
        s4 = self.enc4(s3)    # (B, 8f,   H/8, W/8)

        # Bottleneck
        b = self.bottleneck(s4)  # (B, 16f, H/16, W/16)

        # Decoder
        d4 = self.dec4(b, s4)    # (B, 8f,  H/8,  W/8)
        d3 = self.dec3(d4, s3)   # (B, 4f,  H/4,  W/4)
        d2 = self.dec2(d3, s2)   # (B, 2f,  H/2,  W/2)
        d1 = self.dec1(d2, s1)   # (B, f,   H,    W)

        return torch.sigmoid(self.output_conv(d1))
