"""Evaluation metrics for image translation quality.

Provides FID (Frechet Inception Distance) and SSIM (Structural Similarity)
computation using torchmetrics, plus a convenience function to evaluate
a full CycleGAN model on a dataloader.
"""

from typing import Any

import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure


def compute_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device,
) -> float:
    """Compute FID between real and generated image sets.

    Repeats grayscale images to 3 channels and resizes to 299x299 for
    Inception compatibility.

    Args:
        real_images: Real images tensor of shape [N, 1, H, W] in [-1, 1].
        fake_images: Generated images tensor of shape [N, 1, H, W] in [-1, 1].
        device: Torch device to run computation on.

    Returns:
        FID score (lower is better).
    """
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # Process in batches to avoid OOM
    def _prepare(images: torch.Tensor) -> torch.Tensor:
        """Prepare images for FID: repeat to 3ch, resize to 299."""
        # Repeat grayscale to 3 channels
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        # Resize to 299x299
        images = F.interpolate(
            images, size=(299, 299), mode="bilinear", align_corners=False
        )
        return images

    real_prep = _prepare(real_images.to(device))
    fake_prep = _prepare(fake_images.to(device))

    fid.update(real_prep, real=True)
    fid.update(fake_prep, real=False)

    score = fid.compute().item()
    return score


def compute_ssim(
    images1: torch.Tensor,
    images2: torch.Tensor,
) -> float:
    """Compute mean SSIM between paired image sets.

    Args:
        images1: First image set of shape [N, 1, H, W] in [-1, 1].
        images2: Second image set of shape [N, 1, H, W] in [-1, 1].

    Returns:
        Mean SSIM score (higher is better).
    """
    ssim = StructuralSimilarityIndexMeasure(data_range=2.0)
    score = ssim(images1, images2).item()
    return score


def evaluate_model(
    G_AB: torch.nn.Module,
    G_BA: torch.nn.Module,
    dataloader: Any,
    config: Any,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate CycleGAN generators on a dataloader.

    Computes FID for both translation directions and cycle-consistency SSIM.

    Args:
        G_AB: Generator translating from domain A to domain B.
        G_BA: Generator translating from domain B to domain A.
        dataloader: DataLoader yielding dicts with keys "A" and "B".
        config: Configuration object (unused, reserved for future options).
        device: Torch device.

    Returns:
        Dict with keys: "fid_AB", "fid_BA", "ssim_cycle_A", "ssim_cycle_B".
    """
    G_AB.eval()
    G_BA.eval()

    all_real_A: list[torch.Tensor] = []
    all_real_B: list[torch.Tensor] = []
    all_fake_A: list[torch.Tensor] = []
    all_fake_B: list[torch.Tensor] = []
    all_cycle_A: list[torch.Tensor] = []
    all_cycle_B: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            # Forward translations
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)

            # Cycle reconstructions
            cycle_A = G_BA(fake_B)
            cycle_B = G_AB(fake_A)

            all_real_A.append(real_A.cpu())
            all_real_B.append(real_B.cpu())
            all_fake_A.append(fake_A.cpu())
            all_fake_B.append(fake_B.cpu())
            all_cycle_A.append(cycle_A.cpu())
            all_cycle_B.append(cycle_B.cpu())

    real_A_cat = torch.cat(all_real_A, dim=0)
    real_B_cat = torch.cat(all_real_B, dim=0)
    fake_A_cat = torch.cat(all_fake_A, dim=0)
    fake_B_cat = torch.cat(all_fake_B, dim=0)
    cycle_A_cat = torch.cat(all_cycle_A, dim=0)
    cycle_B_cat = torch.cat(all_cycle_B, dim=0)

    # Compute metrics
    fid_AB = compute_fid(real_B_cat, fake_B_cat, device)
    fid_BA = compute_fid(real_A_cat, fake_A_cat, device)
    ssim_cycle_A = compute_ssim(real_A_cat, cycle_A_cat)
    ssim_cycle_B = compute_ssim(real_B_cat, cycle_B_cat)

    G_AB.train()
    G_BA.train()

    return {
        "fid_AB": fid_AB,
        "fid_BA": fid_BA,
        "ssim_cycle_A": ssim_cycle_A,
        "ssim_cycle_B": ssim_cycle_B,
    }
