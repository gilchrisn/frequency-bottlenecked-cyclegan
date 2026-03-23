"""Cycle consistency loss for CycleGAN.

Enforces that translating an image to the other domain and back
reconstructs the original: ||F(G(x)) - x||_1 and ||G(F(y)) - y||_1.
"""

import torch
import torch.nn as nn


class CycleConsistencyLoss(nn.Module):
    """Weighted L1 loss for cycle consistency.

    Computes lambda_cycle * L1Loss(reconstructed, original) to enforce
    that the composition of generators approximates the identity mapping.
    """

    def __init__(self, lambda_cycle: float) -> None:
        """Initialize the cycle consistency loss.

        Args:
            lambda_cycle: Weight for the cycle consistency term.
        """
        super().__init__()
        self.lambda_cycle = lambda_cycle
        self.l1_loss = nn.L1Loss()

    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """Compute the weighted cycle consistency loss.

        Args:
            reconstructed: Cycle-reconstructed image, e.g. F(G(x)).
            original: Original input image x.

        Returns:
            Scalar loss: lambda_cycle * ||reconstructed - original||_1.
        """
        return self.lambda_cycle * self.l1_loss(reconstructed, original)
