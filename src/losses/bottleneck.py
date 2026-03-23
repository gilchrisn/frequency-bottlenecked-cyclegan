"""Frequency bottleneck for FB-CycleGAN.

Applies a fixed (non-learnable) Gaussian blur to the cycle path, destroying
high-frequency steganographic artifacts that generators might hide in
translated images. The blur is fully differentiable so gradients flow
through, but the zero-parameter constraint prevents the network from
learning to circumvent it.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import LossConfig


class FrequencyBottleneck(nn.Module):
    """Fixed Gaussian blur bottleneck with zero learnable parameters.

    Constructs a 2D Gaussian kernel at initialization and applies it as a
    depthwise convolution with reflection padding to preserve spatial size.
    """

    def __init__(self, kernel_size: int, sigma: float) -> None:
        """Initialize the frequency bottleneck.

        Args:
            kernel_size: Size of the Gaussian kernel (must be odd).
            sigma: Standard deviation of the Gaussian distribution.

        Raises:
            ValueError: If kernel_size is not a positive odd integer.
        """
        super().__init__()

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}."
            )

        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2

        # Build 2D Gaussian kernel
        kernel_1d = self._gaussian_kernel_1d(kernel_size, sigma)
        kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)  # outer product
        # Shape: (1, 1, K, K) for depthwise conv
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)

        # Register as buffer (not a parameter — no gradients, but moves with device)
        self.register_buffer("kernel", kernel_2d)

    @staticmethod
    def _gaussian_kernel_1d(kernel_size: int, sigma: float) -> torch.Tensor:
        """Create a 1D Gaussian kernel.

        Args:
            kernel_size: Number of elements in the kernel.
            sigma: Standard deviation.

        Returns:
            Normalized 1D Gaussian tensor of shape (kernel_size,).
        """
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel = kernel / kernel.sum()
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur via depthwise convolution.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Blurred tensor of same shape (B, C, H, W).
        """
        channels = x.shape[1]

        # Expand kernel for depthwise convolution: (C, 1, K, K)
        kernel = self.kernel.expand(channels, -1, -1, -1)

        # Reflection padding to preserve spatial dimensions
        x_padded = F.pad(x, [self.padding] * 4, mode="reflect")

        return F.conv2d(x_padded, kernel, groups=channels)


class IdentityBottleneck(nn.Module):
    """Passthrough bottleneck (no-op) for baseline CycleGAN without blur."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return input unchanged.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            The same tensor, unmodified.
        """
        return x


def create_bottleneck(config: LossConfig) -> nn.Module:
    """Factory function to create the appropriate bottleneck module.

    Args:
        config: Loss configuration with bottleneck settings.

    Returns:
        FrequencyBottleneck if use_frequency_bottleneck is True,
        otherwise IdentityBottleneck.
    """
    if config.use_frequency_bottleneck:
        return FrequencyBottleneck(
            kernel_size=config.blur_kernel_size,
            sigma=config.blur_sigma,
        )
    return IdentityBottleneck()
