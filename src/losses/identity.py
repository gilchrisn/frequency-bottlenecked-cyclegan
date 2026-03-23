"""Identity loss for CycleGAN.

Regularises generators to preserve color and structure when fed
images from their own target domain: ||G(y) - y||_1 for G: A->B.
"""

import torch
import torch.nn as nn


class IdentityLoss(nn.Module):
    """Weighted L1 loss for identity regularization.

    The effective weight is lambda_identity * lambda_cycle, following the
    standard CycleGAN convention where identity loss is specified as a
    fraction of the cycle consistency weight.
    """

    def __init__(self, lambda_identity: float, lambda_cycle: float) -> None:
        """Initialize the identity loss.

        Args:
            lambda_identity: Identity loss weight relative to cycle loss.
            lambda_cycle: Cycle consistency weight (used to scale identity).
        """
        super().__init__()
        self.weight = lambda_identity * lambda_cycle
        self.l1_loss = nn.L1Loss()

    def forward(self, identity_output: torch.Tensor, real_input: torch.Tensor) -> torch.Tensor:
        """Compute the weighted identity loss.

        Args:
            identity_output: Generator output when fed an in-domain image, e.g. G(y).
            real_input: The in-domain image y.

        Returns:
            Scalar loss: (lambda_identity * lambda_cycle) * ||G(y) - y||_1.
        """
        return self.weight * self.l1_loss(identity_output, real_input)
