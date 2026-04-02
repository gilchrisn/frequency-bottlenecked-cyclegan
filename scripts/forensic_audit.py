"""Steganographic forensic audit: FFT residual analysis + perturbation robustness.

Produces:
1. FFT power spectrum of residuals |G(x) - x| — structured high-freq = steganography
2. Radially averaged power spectrum (1D curve) — bumps at high freq = hidden data
3. Perturbation robustness test — inject noise before reverse cycle, measure degradation

Usage:
    python scripts/forensic_audit.py --checkpoint outputs/checkpoints/baseline/final.pt
    python scripts/forensic_audit.py --checkpoint outputs/checkpoints/fb_sigma1/final.pt
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import ExperimentConfig, TrainConfig, DataConfig
from src.data import create_dataset
from src.training import CycleGANTrainer


def compute_fft_power(residual: np.ndarray) -> np.ndarray:
    """Compute centered 2D FFT power spectrum of a residual image."""
    # Hann window to reduce spectral leakage
    h, w = residual.shape
    win_h = np.hanning(h)
    win_w = np.hanning(w)
    window = np.outer(win_h, win_w)
    windowed = residual * window

    fft = np.fft.fft2(windowed)
    fft_shifted = np.fft.fftshift(fft)
    power = np.abs(fft_shifted) ** 2
    return power


def radial_average(power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute radially averaged power spectrum.

    Returns:
        (frequencies, power_values) arrays for 1D plotting.
    """
    h, w = power.shape
    cy, cx = h // 2, w // 2

    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    max_r = min(cy, cx)
    radial_sum = np.zeros(max_r)
    radial_count = np.zeros(max_r)

    for ri in range(max_r):
        mask = r == ri
        radial_sum[ri] = power[mask].sum()
        radial_count[ri] = mask.sum()

    radial_count[radial_count == 0] = 1
    radial_mean = radial_sum / radial_count

    freqs = np.arange(max_r) / max_r  # normalized frequency [0, 1]
    return freqs, radial_mean


def run_fft_audit(trainer: CycleGANTrainer, loader: DataLoader,
                  device: str, n_samples: int = 100) -> dict:
    """Run FFT residual analysis on generated images.

    Returns dict with averaged power spectra and 2D power maps.
    """
    trainer.G_AB.eval()
    trainer.G_BA.eval()

    all_power_AB = []
    all_power_BA = []
    sample_residual_AB = None

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_samples:
                break
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            fake_B = trainer.G_AB(real_A)
            fake_A = trainer.G_BA(real_B)

            # Residuals
            res_AB = (fake_B - real_A).abs().cpu().numpy()
            res_BA = (fake_A - real_B).abs().cpu().numpy()

            for b in range(res_AB.shape[0]):
                power_AB = compute_fft_power(res_AB[b, 0])
                power_BA = compute_fft_power(res_BA[b, 0])
                all_power_AB.append(power_AB)
                all_power_BA.append(power_BA)

                if sample_residual_AB is None:
                    sample_residual_AB = res_AB[b, 0]

    # Average power spectra
    avg_power_AB = np.mean(all_power_AB, axis=0)
    avg_power_BA = np.mean(all_power_BA, axis=0)

    freqs_AB, radial_AB = radial_average(avg_power_AB)
    freqs_BA, radial_BA = radial_average(avg_power_BA)

    return {
        "avg_power_AB": avg_power_AB,
        "avg_power_BA": avg_power_BA,
        "freqs_AB": freqs_AB,
        "radial_AB": radial_AB,
        "freqs_BA": freqs_BA,
        "radial_BA": radial_BA,
        "sample_residual": sample_residual_AB,
        "n_samples": len(all_power_AB),
    }


def run_perturbation_test(trainer: CycleGANTrainer, loader: DataLoader,
                          device: str, noise_levels: list[float],
                          n_samples: int = 50) -> dict:
    """Test robustness to noise injection in the cycle path.

    Steganographic model: catastrophic degradation.
    Honest model: graceful degradation.
    """
    trainer.G_AB.eval()
    trainer.G_BA.eval()

    results = {sigma: [] for sigma in noise_levels}

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_samples:
                break
            real_A = batch["A"].to(device)

            fake_B = trainer.G_AB(real_A)

            for sigma in noise_levels:
                noise = torch.randn_like(fake_B) * sigma
                perturbed = fake_B + noise
                rec_A = trainer.G_BA(perturbed)
                l1_error = (rec_A - real_A).abs().mean().item()
                results[sigma].append(l1_error)

    # Also compute clean reconstruction error as baseline
    clean_errors = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_samples:
                break
            real_A = batch["A"].to(device)
            fake_B = trainer.G_AB(real_A)
            rec_A = trainer.G_BA(fake_B)
            clean_errors.append((rec_A - real_A).abs().mean().item())

    return {
        "noise_levels": noise_levels,
        "mean_errors": {s: np.mean(v) for s, v in results.items()},
        "std_errors": {s: np.std(v) for s, v in results.items()},
        "clean_error": np.mean(clean_errors),
    }


def plot_audit(fft_results: dict, perturb_results: dict,
               output_dir: Path, name: str) -> None:
    """Generate all forensic audit plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"Steganographic Forensic Audit — {name}", fontsize=14, y=1.02)

    # 1. Sample residual image
    ax = axes[0, 0]
    ax.imshow(fft_results["sample_residual"], cmap="hot", vmin=0)
    ax.set_title("Sample Residual |G(x) - x|")
    ax.axis("off")

    # 2. 2D FFT power spectrum (A->B)
    ax = axes[0, 1]
    power = fft_results["avg_power_AB"]
    ax.imshow(np.log1p(power), cmap="inferno")
    ax.set_title(f"Avg FFT Power Spectrum (Path->Healthy)\nn={fft_results['n_samples']} samples")
    ax.axis("off")

    # 3. 2D FFT power spectrum (B->A)
    ax = axes[0, 2]
    power = fft_results["avg_power_BA"]
    ax.imshow(np.log1p(power), cmap="inferno")
    ax.set_title("Avg FFT Power Spectrum (Healthy->Path)")
    ax.axis("off")

    # 4. Radial average (A->B) — log-log
    ax = axes[1, 0]
    freqs = fft_results["freqs_AB"]
    radial = fft_results["radial_AB"]
    mask = freqs > 0
    ax.loglog(freqs[mask], radial[mask], linewidth=1.5)
    ax.set_xlabel("Normalized Frequency")
    ax.set_ylabel("Power")
    ax.set_title("Radial Power Spectrum (Path->Healthy)\nBumps at high freq = steganography")
    ax.grid(True, alpha=0.3)

    # 5. Radial average (B->A) — log-log
    ax = axes[1, 1]
    freqs = fft_results["freqs_BA"]
    radial = fft_results["radial_BA"]
    mask = freqs > 0
    ax.loglog(freqs[mask], radial[mask], linewidth=1.5)
    ax.set_xlabel("Normalized Frequency")
    ax.set_ylabel("Power")
    ax.set_title("Radial Power Spectrum (Healthy->Path)")
    ax.grid(True, alpha=0.3)

    # 6. Perturbation robustness
    ax = axes[1, 2]
    sigmas = perturb_results["noise_levels"]
    means = [perturb_results["mean_errors"][s] for s in sigmas]
    stds = [perturb_results["std_errors"][s] for s in sigmas]
    clean = perturb_results["clean_error"]

    ax.errorbar(sigmas, means, yerr=stds, marker="o", capsize=4, linewidth=2)
    ax.axhline(y=clean, color="green", linestyle="--", label=f"Clean: {clean:.4f}")
    ax.set_xlabel("Noise sigma")
    ax.set_ylabel("Mean L1 Reconstruction Error")
    ax.set_title("Perturbation Robustness\nSteep rise = steganographic")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / f"forensic_audit_{name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Steganographic forensic audit.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="outputs/plots")
    parser.add_argument("--name", type=str, default=None,
                        help="Name for plots (default: inferred from checkpoint path)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Infer name from checkpoint path
    name = args.name or Path(args.checkpoint).parent.name

    cfg = ExperimentConfig(
        name=f"audit_{name}",
        train=TrainConfig(compile_models=False),
        data=DataConfig(num_workers=0),
        use_wandb=False,
        device=device,
    )

    trainer = CycleGANTrainer(cfg)
    trainer.load_checkpoint(args.checkpoint)

    ds = create_dataset(cfg.data, "val")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    print(f"Running FFT audit on {args.n_samples} samples...")
    fft_results = run_fft_audit(trainer, loader, device, args.n_samples)
    print(f"  Analyzed {fft_results['n_samples']} images")

    print("Running perturbation robustness test...")
    noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    perturb_results = run_perturbation_test(trainer, loader, device, noise_levels)
    print(f"  Clean L1 error: {perturb_results['clean_error']:.4f}")
    for sigma in noise_levels:
        print(f"  sigma={sigma}: L1={perturb_results['mean_errors'][sigma]:.4f}")

    print("Generating plots...")
    plot_audit(fft_results, perturb_results, Path(args.output_dir), name)
    print("Done.")


if __name__ == "__main__":
    main()
