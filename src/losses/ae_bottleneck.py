"""Pretrained autoencoder bottleneck.

Projects the intermediate image onto a learned manifold of healthy brain
MRI. Anything not on the manifold (including steganographic patterns)
is destroyed by the reconstruction. Must be pretrained and frozen —
never trained jointly with the CycleGAN.
"""

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder for 256x256 single-channel images.

    Architecture:
        Encoder: 256 -> 128 -> 64 -> 32 -> latent_dim (via FC)
        Decoder: latent_dim -> 32 -> 64 -> 128 -> 256 (via FC + ConvT)

    Args:
        latent_dim: Dimension of the bottleneck latent space.
        in_channels: Number of input channels (1 for grayscale MRI).
    """

    def __init__(self, latent_dim: int = 256, in_channels: int = 1) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: (1, 256, 256) -> (128, 16, 16)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),  # -> 32x128x128
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),           # -> 64x64x64
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),          # -> 128x32x32
            nn.ReLU(True),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),         # -> 128x16x16
            nn.ReLU(True),
        )
        self.encoder_fc = nn.Linear(128 * 16 * 16, latent_dim)

        # Decoder: latent_dim -> (1, 256, 256)
        self.decoder_fc = nn.Linear(latent_dim, 128 * 16 * 16)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),  # -> 128x32x32
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # -> 64x64x64
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # -> 32x128x128
            nn.ReLU(True),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),  # -> 1x256x256
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space.

        Args:
            x: Input tensor (B, C, 256, 256).

        Returns:
            Latent vector (B, latent_dim).
        """
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        return self.encoder_fc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space.

        Args:
            z: Latent vector (B, latent_dim).

        Returns:
            Reconstructed image (B, C, 256, 256).
        """
        h = self.decoder_fc(z)
        h = h.view(h.size(0), 128, 16, 16)
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode.

        Args:
            x: Input tensor (B, C, 256, 256).

        Returns:
            Reconstructed tensor of same shape.
        """
        z = self.encode(x)
        return self.decode(z)


class AutoencoderBottleneck(nn.Module):
    """Frozen pretrained autoencoder used as a cycle-path bottleneck.

    Loads a pretrained ConvAutoencoder checkpoint, freezes all weights,
    and applies encode-decode as the bottleneck. Anything not on the
    learned healthy-brain manifold gets destroyed.

    Args:
        checkpoint_path: Path to pretrained autoencoder .pt file.
        latent_dim: Must match the pretrained model's latent_dim.
    """

    def __init__(self, checkpoint_path: str, latent_dim: int = 256) -> None:
        super().__init__()
        self.ae = ConvAutoencoder(latent_dim=latent_dim, in_channels=1)

        # Load pretrained weights
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in state:
            self.ae.load_state_dict(state["model_state_dict"])
        else:
            self.ae.load_state_dict(state)

        # Freeze all parameters
        for param in self.ae.parameters():
            param.requires_grad = False
        self.ae.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project through the autoencoder (encode -> decode).

        Args:
            x: Input tensor (B, C, 256, 256) in [-1, 1].

        Returns:
            Reconstructed tensor of same shape. Steganographic patterns
            not on the healthy manifold are destroyed.
        """
        # Keep AE in eval mode even during CycleGAN training
        self.ae.eval()
        return self.ae(x)
