"""GAN adversarial loss supporting LSGAN and vanilla BCE formulations.

LSGAN (least-squares) provides more stable gradients and avoids vanishing
gradient issues compared to vanilla BCE. Both are supported for ablation.
"""

import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """Adversarial loss for GAN training.

    Supports two modes:
        - 'lsgan': Least-squares GAN loss (MSELoss). More stable training.
        - 'vanilla': Standard GAN loss (BCEWithLogitsLoss).

    Maintains registered buffers for real and fake target tensors to ensure
    they are automatically moved to the correct device.
    """

    def __init__(self, gan_mode: str) -> None:
        """Initialize the GAN loss.

        Args:
            gan_mode: Loss formulation. One of 'lsgan' or 'vanilla'.

        Raises:
            NotImplementedError: If gan_mode is not recognized.
        """
        super().__init__()

        self.register_buffer("real_label", torch.tensor(1.0))
        self.register_buffer("fake_label", torch.tensor(0.0))

        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f"GAN mode '{gan_mode}' is not supported.")

        self.gan_mode = gan_mode

    def _get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Create a target tensor with the same shape as prediction.

        Args:
            prediction: Discriminator output tensor.
            target_is_real: Whether the target should be real (1.0) or fake (0.0).

        Returns:
            Target tensor filled with the appropriate label value.
        """
        if target_is_real:
            return self.real_label.expand_as(prediction)
        else:
            return self.fake_label.expand_as(prediction)

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Compute the GAN loss.

        Args:
            prediction: Discriminator output tensor of arbitrary shape.
            target_is_real: Whether the prediction should be classified as real.

        Returns:
            Scalar loss tensor.
        """
        target_tensor = self._get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)
