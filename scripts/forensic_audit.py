"""Forensic audit of steganographic encoding in CycleGAN generators.

For each checkpoint provided, this script:
  1. Generates translated images G_AB(real_A) on the test set.
  2. Computes FFT power spectra of residuals |G_AB(real_A) - real_A| to reveal
     any structured high-frequency content characteristic of steganographic encoding.
  3. Runs a perturbation test: adds Gaussian noise to fake_B before the reverse
     cycle G_BA(fake_B + eps). A steganographic generator degrades catastrophically;
     a structurally honest one degrades gracefully.
  4. Saves FFT heatmaps and perturbation curves. Compares two checkpoints
     (typically baseline vs. FB-CycleGAN) when both are supplied.

Usage:
    # Single model
    python scripts/forensic_audit.py \
        --checkpoint outputs/checkpoints/baseline_cyclegan/final.pt \
        --data-dir data/processed --output-dir outputs/forensics/baseline

    # Comparison
    python scripts/forensic_audit.py \
        --checkpoint outputs/checkpoints/baseline_cyclegan/final.pt \
        --checkpoint-fb outputs/checkpoints/fb_cyclegan_sigma1/final.pt \
        --data-dir data/processed --output-dir outputs/forensics/comparison
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root is on path when invoked directly
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ExperimentConfig, ModelConfig
from src.data.brats_dataset import BraTSDataset
from src.data.transforms import get_val_transform
from src.data import create_dataloader
from src.models import create_generator


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_generators(checkpoint_path: str, device: torch.device):
    """Load G_AB and G_BA from a saved checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Target device.

    Returns:
        Tuple of (G_AB, G_BA) PyTorch modules in eval mode.
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct model config from checkpoint config dict if available
    saved_cfg = state.get("config", {})
    model_cfg_dict = saved_cfg.get("model", {}) if isinstance(saved_cfg, dict) else {}

    model_cfg = ModelConfig(
        input_channels=model_cfg_dict.get("input_channels", 1),
        output_channels=model_cfg_dict.get("output_channels", 1),
        ngf=model_cfg_dict.get("ngf", 64),
        ndf=model_cfg_dict.get("ndf", 64),
        norm_type=model_cfg_dict.get("norm_type", "instance"),
        no_dropout=model_cfg_dict.get("no_dropout", True),
        init_type=model_cfg_dict.get("init_type", "normal"),
        init_gain=model_cfg_dict.get("init_gain", 0.02),
    )

    G_AB = create_generator(model_cfg).to(device)
    G_BA = create_generator(model_cfg).to(device)
    G_AB.load_state_dict(state["G_AB"])
    G_BA.load_state_dict(state["G_BA"])
    G_AB.eval()
    G_BA.eval()
    return G_AB, G_BA


# ---------------------------------------------------------------------------
# FFT analysis
# ---------------------------------------------------------------------------

def compute_fft_residual_spectrum(
    generator_AB: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 200,
) -> np.ndarray:
    """Compute the mean 2D FFT power spectrum of translation residuals.

    Residual = |G_AB(real_A) - real_A|. Structured high-frequency peaks in
    the mean spectrum indicate steganographic encoding.

    Args:
        generator_AB: Trained generator (pathological -> healthy).
        loader: DataLoader yielding {"A": tensor, "B": tensor}.
        device: Torch device.
        max_batches: Maximum number of batches to process.

    Returns:
        Mean log-power spectrum as 2D numpy array, shape (H, W).
    """
    accumulated_spectrum: np.ndarray | None = None
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="FFT residuals", leave=False)):
            if i >= max_batches:
                break
            real_A = batch["A"].to(device)
            fake_B = generator_AB(real_A)

            # Residual in [-2, 2] range (both in [-1, 1])
            residual = (fake_B - real_A).abs()  # (B, 1, H, W)

            for b in range(residual.shape[0]):
                res_2d = residual[b, 0].cpu().numpy()  # (H, W)
                # Shift zero-frequency to center
                fft = np.fft.fftshift(np.fft.fft2(res_2d))
                power = np.log1p(np.abs(fft) ** 2)

                if accumulated_spectrum is None:
                    accumulated_spectrum = power
                else:
                    accumulated_spectrum += power
                count += 1

    if accumulated_spectrum is None or count == 0:
        raise RuntimeError("No images processed for FFT analysis.")

    return accumulated_spectrum / count


def radial_power_profile(spectrum_2d: np.ndarray, n_bins: int = 64) -> tuple:
    """Compute radially-averaged power spectrum (rotationally averaged FFT).

    Args:
        spectrum_2d: 2D log-power spectrum array, shape (H, W).
        n_bins: Number of radial frequency bins.

    Returns:
        Tuple of (frequencies, mean_power) arrays of length n_bins.
    """
    H, W = spectrum_2d.shape
    cy, cx = H // 2, W // 2

    # Distance from center for every pixel
    y_idx, x_idx = np.ogrid[:H, :W]
    dist = np.hypot(y_idx - cy, x_idx - cx).ravel()
    power_flat = spectrum_2d.ravel()

    max_dist = min(cy, cx)
    edges = np.linspace(0, max_dist, n_bins + 1)
    bin_means = []
    bin_centers = []

    for j in range(n_bins):
        mask = (dist >= edges[j]) & (dist < edges[j + 1])
        if mask.sum() > 0:
            bin_means.append(power_flat[mask].mean())
        else:
            bin_means.append(0.0)
        bin_centers.append((edges[j] + edges[j + 1]) / 2.0)

    return np.array(bin_centers) / max_dist, np.array(bin_means)


# ---------------------------------------------------------------------------
# Perturbation test
# ---------------------------------------------------------------------------

def perturbation_test(
    generator_AB: torch.nn.Module,
    generator_BA: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    noise_levels: list[float],
    max_batches: int = 50,
) -> dict[float, float]:
    """Measure cycle-reconstruction degradation under injected noise.

    For each noise level sigma_eps, compute:
        clean_l1  = ||G_BA(fake_B) - real_A||_1
        noisy_l1  = ||G_BA(fake_B + eps) - real_A||_1   (eps ~ N(0, sigma_eps^2))
        ratio     = noisy_l1 / clean_l1

    A steganographic generator has ratio >> 1 (catastrophic failure).
    A structurally honest generator has ratio ≈ 1 (graceful degradation).

    Args:
        generator_AB: G_AB (pathological -> healthy).
        generator_BA: G_BA (healthy -> pathological).
        loader: DataLoader.
        device: Torch device.
        noise_levels: List of sigma_eps values to test.
        max_batches: Maximum batches per noise level.

    Returns:
        Dict mapping sigma_eps -> mean ratio.
    """
    results: dict[float, float] = {}

    with torch.no_grad():
        for sigma_eps in tqdm(noise_levels, desc="Perturbation test"):
            clean_l1_sum = 0.0
            noisy_l1_sum = 0.0
            n = 0

            for i, batch in enumerate(loader):
                if i >= max_batches:
                    break
                real_A = batch["A"].to(device)
                fake_B = generator_AB(real_A)

                # Clean cycle
                rec_A_clean = generator_BA(fake_B)
                clean_l1 = F.l1_loss(rec_A_clean, real_A, reduction="mean").item()

                # Noisy cycle
                noise = torch.randn_like(fake_B) * sigma_eps
                rec_A_noisy = generator_BA(fake_B + noise)
                noisy_l1 = F.l1_loss(rec_A_noisy, real_A, reduction="mean").item()

                clean_l1_sum += clean_l1
                noisy_l1_sum += noisy_l1
                n += 1

            if n > 0 and clean_l1_sum > 1e-9:
                results[sigma_eps] = noisy_l1_sum / clean_l1_sum
            else:
                results[sigma_eps] = 1.0

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_fft_spectrum(
    spectrum: np.ndarray,
    title: str,
    save_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Save a 2D FFT power spectrum as a heatmap.

    Args:
        spectrum: 2D numpy array (log-power).
        title: Plot title.
        save_path: Output file path (.png).
        vmin: Colormap minimum (auto if None).
        vmax: Colormap maximum (auto if None).
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(spectrum, cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_radial_profiles(
    profiles: dict[str, tuple[np.ndarray, np.ndarray]],
    save_path: Path,
) -> None:
    """Plot radially-averaged power spectra for multiple models.

    Args:
        profiles: Dict mapping label -> (frequencies, mean_power).
        save_path: Output file path.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, (freqs, power) in profiles.items():
        ax.plot(freqs, power, label=label, linewidth=1.5)

    ax.set_xlabel("Normalized Radial Frequency")
    ax.set_ylabel("Mean Log Power")
    ax.set_title("Radial Power Spectrum of Translation Residuals")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_perturbation_curves(
    curves: dict[str, dict[float, float]],
    save_path: Path,
) -> None:
    """Plot perturbation test L1 ratio curves for multiple models.

    Args:
        curves: Dict mapping label -> {sigma_eps -> ratio}.
        save_path: Output file path.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, data in curves.items():
        sigmas = sorted(data.keys())
        ratios = [data[s] for s in sigmas]
        ax.plot(sigmas, ratios, marker="o", label=label, linewidth=1.5)

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.6, label="No degradation")
    ax.set_xlabel("Injected Noise σ")
    ax.set_ylabel("L1 Ratio (noisy / clean)")
    ax.set_title("Cycle Reconstruction Degradation Under Perturbation\n(steganographic models spike; honest models stay near 1.0)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_spectra(
    spec_baseline: np.ndarray,
    spec_fb: np.ndarray,
    labels: tuple[str, str],
    save_path: Path,
) -> None:
    """Side-by-side FFT heatmap comparison.

    Args:
        spec_baseline: 2D spectrum for baseline model.
        spec_fb: 2D spectrum for FB-CycleGAN model.
        labels: Tuple of (baseline_label, fb_label).
        save_path: Output file path.
    """
    vmin = min(spec_baseline.min(), spec_fb.min())
    vmax = max(spec_baseline.max(), spec_fb.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, spec, label in zip(axes, [spec_baseline, spec_fb], labels):
        im = ax.imshow(spec, cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Mean FFT Power Spectrum of Translation Residuals", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run forensic audit."""
    parser = argparse.ArgumentParser(
        description="Forensic FFT + perturbation audit for CycleGAN checkpoints."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to primary (e.g., baseline) checkpoint .pt file.",
    )
    parser.add_argument(
        "--checkpoint-fb",
        type=str,
        default=None,
        help="Optional path to FB-CycleGAN checkpoint for comparison.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Path to preprocessed data directory (containing split.json).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to audit.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/forensics",
        help="Directory to save audit results.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for image generation.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=200,
        help="Maximum batches to process for FFT analysis.",
    )
    parser.add_argument(
        "--perturbation-batches",
        type=int,
        default=50,
        help="Maximum batches per noise level for perturbation test.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="Baseline CycleGAN",
        help="Display label for primary checkpoint.",
    )
    parser.add_argument(
        "--label-fb",
        type=str,
        default="FB-CycleGAN",
        help="Display label for FB-CycleGAN checkpoint.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    # Build a minimal config for the dataloader
    cfg = ExperimentConfig()
    cfg.train.batch_size = args.batch_size
    cfg.data.flip = False  # no augmentation at eval time

    dataset = BraTSDataset(
        config=cfg.data,
        split=args.split,
        transform=get_val_transform(cfg.data),
        processed_dir=args.data_dir,
    )
    loader = create_dataloader(dataset, cfg.data, cfg.train, split=args.split)
    print(f"Loaded {args.split} split: {len(dataset)} samples")

    # --- Noise levels for perturbation test ---
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]

    # -----------------------------------------------------------------------
    # Audit primary checkpoint
    # -----------------------------------------------------------------------
    print(f"\n=== Auditing: {args.checkpoint} ===")
    G_AB_base, G_BA_base = load_generators(args.checkpoint, device)

    spec_base = compute_fft_residual_spectrum(G_AB_base, loader, device, args.max_batches)
    radial_base = radial_power_profile(spec_base)
    pert_base = perturbation_test(
        G_AB_base, G_BA_base, loader, device, noise_levels, args.perturbation_batches
    )

    # Save individual FFT heatmap
    plot_fft_spectrum(
        spec_base,
        title=f"FFT Residual Spectrum — {args.label}",
        save_path=output_dir / "fft_spectrum_baseline.png",
    )
    print(f"  Saved FFT spectrum: {output_dir / 'fft_spectrum_baseline.png'}")

    # -----------------------------------------------------------------------
    # Audit FB-CycleGAN checkpoint (optional)
    # -----------------------------------------------------------------------
    spec_fb = None
    radial_fb = None
    pert_fb = None

    if args.checkpoint_fb is not None:
        print(f"\n=== Auditing: {args.checkpoint_fb} ===")
        G_AB_fb, G_BA_fb = load_generators(args.checkpoint_fb, device)

        spec_fb = compute_fft_residual_spectrum(G_AB_fb, loader, device, args.max_batches)
        radial_fb = radial_power_profile(spec_fb)
        pert_fb = perturbation_test(
            G_AB_fb, G_BA_fb, loader, device, noise_levels, args.perturbation_batches
        )

        plot_fft_spectrum(
            spec_fb,
            title=f"FFT Residual Spectrum — {args.label_fb}",
            save_path=output_dir / "fft_spectrum_fb.png",
        )
        print(f"  Saved FFT spectrum: {output_dir / 'fft_spectrum_fb.png'}")

        # Side-by-side comparison
        plot_comparison_spectra(
            spec_base,
            spec_fb,
            labels=(args.label, args.label_fb),
            save_path=output_dir / "fft_comparison.png",
        )
        print(f"  Saved comparison: {output_dir / 'fft_comparison.png'}")

    # -----------------------------------------------------------------------
    # Radial power profiles
    # -----------------------------------------------------------------------
    profiles: dict[str, tuple] = {args.label: radial_base}
    if radial_fb is not None:
        profiles[args.label_fb] = radial_fb

    plot_radial_profiles(profiles, save_path=output_dir / "radial_power_profiles.png")
    print(f"Saved radial profiles: {output_dir / 'radial_power_profiles.png'}")

    # -----------------------------------------------------------------------
    # Perturbation curves
    # -----------------------------------------------------------------------
    pert_curves: dict[str, dict[float, float]] = {args.label: pert_base}
    if pert_fb is not None:
        pert_curves[args.label_fb] = pert_fb

    plot_perturbation_curves(pert_curves, save_path=output_dir / "perturbation_curves.png")
    print(f"Saved perturbation curves: {output_dir / 'perturbation_curves.png'}")

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    print("\n--- Perturbation Test Summary ---")
    print(f"{'Noise σ':>10}", end="")
    for label in pert_curves:
        print(f"  {label:>25}", end="")
    print()
    for sigma in noise_levels:
        print(f"{sigma:>10.3f}", end="")
        for label in pert_curves:
            ratio = pert_curves[label].get(sigma, float("nan"))
            print(f"  {ratio:>25.4f}", end="")
        print()

    # Check H1 hypothesis: ratio at sigma=0.1 vs sigma=0.0
    sigma_test = 0.1
    base_ratio = pert_base.get(sigma_test, 1.0)
    print(f"\nH1 check — {args.label} ratio at σ={sigma_test}: {base_ratio:.4f}")
    if base_ratio > 1.5:
        print("  ✓ Ratio > 1.5: consistent with steganographic encoding (H1 confirmed)")
    else:
        print("  ✗ Ratio ≤ 1.5: no strong evidence of steganographic encoding")

    if pert_fb is not None:
        fb_ratio = pert_fb.get(sigma_test, 1.0)
        print(f"H2 check — {args.label_fb} ratio at σ={sigma_test}: {fb_ratio:.4f}")
        if fb_ratio < base_ratio:
            print("  ✓ FB-CycleGAN more robust than baseline (H2 consistent)")
        else:
            print("  ✗ FB-CycleGAN not more robust (unexpected)")

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
