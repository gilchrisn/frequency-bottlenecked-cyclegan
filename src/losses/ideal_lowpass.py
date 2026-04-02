"""Ideal low-pass filter bottleneck.

Applies a hard frequency cutoff in the FFT domain: all coefficients
above the cutoff frequency are set to exactly zero. This is the
theoretically cleanest bottleneck — it gives a precise channel capacity
bound — but may cause Gibbs phenomenon (ringing artifacts) at edges.
"""

import torch
import torch.nn as nn


class IdealLowPassBottleneck(nn.Module):
    """Hard frequency cutoff via FFT masking.

    Transforms the image to frequency domain, zeros out all coefficients
    beyond a normalized cutoff radius, and transforms back.

    Args:
        cutoff: Normalized cutoff frequency in [0, 1]. 0 = DC only,
            1 = no filtering. Frequencies with radius > cutoff * max_radius
            are set to zero.
    """

    def __init__(self, cutoff: float = 0.15) -> None:
        super().__init__()
        self.cutoff = cutoff
        self._mask_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _get_mask(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Build or retrieve cached binary frequency mask."""
        key = (h, w)
        if key not in self._mask_cache or self._mask_cache[key].device != device:
            cy, cx = h // 2, w // 2
            max_r = min(cy, cx)
            y = torch.arange(h, device=device).float() - cy
            x = torch.arange(w, device=device).float() - cx
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            dist = torch.sqrt(xx ** 2 + yy ** 2)
            mask = (dist <= self.cutoff * max_r).float()
            self._mask_cache[key] = mask
        return self._mask_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ideal low-pass filter.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Filtered tensor of same shape.
        """
        b, c, h, w = x.shape
        mask = self._get_mask(h, w, x.device)  # (H, W)

        # Process each channel independently
        result = torch.zeros_like(x)
        for ch in range(c):
            # FFT -> shift -> mask -> unshift -> IFFT
            freq = torch.fft.fft2(x[:, ch, :, :])
            freq_shifted = torch.fft.fftshift(freq)
            freq_masked = freq_shifted * mask.unsqueeze(0)
            freq_unshifted = torch.fft.ifftshift(freq_masked)
            result[:, ch, :, :] = torch.fft.ifft2(freq_unshifted).real

        return result
