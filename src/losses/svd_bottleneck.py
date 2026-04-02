"""SVD truncation bottleneck.

Compresses the intermediate image by keeping only the top-k singular
values and discarding the rest. This is a rank-k approximation that
removes fine spatial patterns (including steganographic encoding)
while preserving dominant structure.

Unlike frequency-based methods, SVD operates in the spatial domain
and compresses by importance (singular value magnitude) rather than
by frequency band.
"""

import torch
import torch.nn as nn


class SVDBottleneck(nn.Module):
    """Rank-k SVD truncation bottleneck.

    For each image in the batch, computes the SVD, keeps the top-k
    singular values, and reconstructs. Zero learnable parameters.

    Args:
        rank: Number of singular values to keep. Lower rank = more
            aggressive compression. For 256x256 images, rank=50
            preserves major structure while destroying fine detail.
    """

    def __init__(self, rank: int = 50) -> None:
        super().__init__()
        self.rank = rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SVD truncation.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Rank-k approximation of same shape.
        """
        b, c, h, w = x.shape
        result = torch.zeros_like(x)

        for ch in range(c):
            # x[:, ch] has shape (B, H, W)
            U, S, Vh = torch.linalg.svd(x[:, ch, :, :], full_matrices=False)
            # U: (B, H, K), S: (B, K), Vh: (B, K, W) where K = min(H, W)

            k = min(self.rank, S.shape[-1])

            # Truncate to rank k
            U_k = U[:, :, :k]        # (B, H, k)
            S_k = S[:, :k]            # (B, k)
            Vh_k = Vh[:, :k, :]       # (B, k, W)

            # Reconstruct: U_k @ diag(S_k) @ Vh_k
            result[:, ch, :, :] = torch.bmm(
                U_k * S_k.unsqueeze(1),  # (B, H, k) * (B, 1, k) = (B, H, k)
                Vh_k                      # (B, k, W)
            )                             # (B, H, W)

        return result
