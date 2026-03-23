"""70x70 PatchGAN discriminator for CycleGAN.

Classifies overlapping 70x70 patches as real or fake, capturing local
texture and style without constraining global structure.
"""

from typing import Callable, List

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.models.generator import get_norm_layer


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator that outputs a grid of real/fake predictions.

    Architecture (5 layers):
        Layer 1: Conv(4, ndf, stride=2) -> LeakyReLU(0.2)              [no norm]
        Layer 2: Conv(4, ndf*2, stride=2) -> Norm -> LeakyReLU(0.2)
        Layer 3: Conv(4, ndf*4, stride=2) -> Norm -> LeakyReLU(0.2)
        Layer 4: Conv(4, ndf*8, stride=1) -> Norm -> LeakyReLU(0.2)
        Layer 5: Conv(4, 1, stride=1)                                   [no norm, no activation]

    Receptive field: 70x70 pixels.
    """

    NEGATIVE_SLOPE: float = 0.2

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the PatchGAN discriminator from model configuration.

        Args:
            config: Model configuration dataclass with architecture hyperparameters.
        """
        super().__init__()

        input_channels: int = config.input_channels
        ndf: int = config.ndf
        norm_layer = get_norm_layer(config.norm_type)

        layers: List[nn.Module] = []

        # Layer 1: Conv -> LeakyReLU (no normalization)
        layers += [
            nn.Conv2d(
                input_channels, ndf,
                kernel_size=4, stride=2, padding=1, bias=True,
            ),
            nn.LeakyReLU(self.NEGATIVE_SLOPE, inplace=True),
        ]

        # Layer 2: Conv -> Norm -> LeakyReLU
        layers += [
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=True),
            norm_layer(ndf * 2),
            nn.LeakyReLU(self.NEGATIVE_SLOPE, inplace=True),
        ]

        # Layer 3: Conv -> Norm -> LeakyReLU
        layers += [
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=True),
            norm_layer(ndf * 4),
            nn.LeakyReLU(self.NEGATIVE_SLOPE, inplace=True),
        ]

        # Layer 4: Conv -> Norm -> LeakyReLU (stride=1)
        layers += [
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=True),
            norm_layer(ndf * 8),
            nn.LeakyReLU(self.NEGATIVE_SLOPE, inplace=True),
        ]

        # Layer 5: Conv -> output (no norm, no activation)
        layers += [
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1, bias=True),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify input patches as real or fake.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Patch-level prediction map of shape (B, 1, H', W').
        """
        return self.model(x)
