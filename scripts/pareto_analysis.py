"""Pareto analysis: sweep sigma and compare honesty vs quality metrics.

Generates a multi-panel figure comparing baseline and all FB-CycleGAN variants.

Usage:
    python -m scripts.pareto_analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.config import ExperimentConfig, TrainConfig, DataConfig, LossConfig
from src.training import CycleGANTrainer
from src.data import create_dataset
from torch.utils.data import DataLoader
from scripts.forensic_audit import run_fft_audit


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = {
        "Baseline": {"path": "outputs/checkpoints/baseline/final.pt", "sigma": 0.0, "bn": False},
        "sigma=0.5": {"path": "outputs/checkpoints/fb/fb_sigma0.5.pt", "sigma": 0.5, "bn": True},
        "sigma=1.0": {"path": "outputs/checkpoints/fb/fb_sigma1.0.pt", "sigma": 1.0, "bn": True},
        "sigma=1.5": {"path": "outputs/checkpoints/fb/fb_sigma1.5.pt", "sigma": 1.5, "bn": True},
        "sigma=2.0": {"path": "outputs/checkpoints/fb/fb_sigma2.pt", "sigma": 2.0, "bn": True},
    }

    noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    n_eval = 50
    colors = ["red", "orange", "blue", "green", "purple"]

    all_results = {}
    for name, info in models.items():
        print(f"Evaluating {name}...")
        ks = max(3, int(info["sigma"] * 4) | 1) if info["sigma"] > 0 else 5
        cfg = ExperimentConfig(
            name=name, train=TrainConfig(compile_models=False),
            loss=LossConfig(use_frequency_bottleneck=info["bn"],
                            blur_sigma=info["sigma"], blur_kernel_size=ks),
            data=DataConfig(num_workers=0), use_wandb=False, device=device,
        )
        trainer = CycleGANTrainer(cfg)
        trainer.load_checkpoint(info["path"])

        ds = create_dataset(cfg.data, "val")
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

        # Perturbation test + cycle L1
        perturb = {s: [] for s in noise_levels}
        cycle_l1s = []

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= n_eval:
                    break
                real_A = batch["A"].to(device)
                fake_B = trainer.G_AB(real_A)

                # Clean cycle
                if info["bn"]:
                    rec = trainer.G_BA(trainer.bottleneck(fake_B))
                else:
                    rec = trainer.G_BA(fake_B)
                cycle_l1s.append((rec - real_A).abs().mean().item())

                # Perturbation
                for s in noise_levels:
                    noise = torch.randn_like(fake_B) * s if s > 0 else torch.zeros_like(fake_B)
                    rec_n = trainer.G_BA(fake_B + noise)
                    perturb[s].append((rec_n - real_A).abs().mean().item())

        # FFT
        loader2 = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        fft_res = run_fft_audit(trainer, loader2, device, n_eval)

        clean = max(np.mean(perturb[0.0]), 1e-6)
        all_results[name] = {
            "sigma": info["sigma"],
            "cycle_l1": np.mean(cycle_l1s),
            "cycle_l1_std": np.std(cycle_l1s),
            "perturb": {s: np.mean(v) for s, v in perturb.items()},
            "perturb_std": {s: np.std(v) for s, v in perturb.items()},
            "degradation_ratio": np.mean(perturb[0.001]) / clean,
            "freqs": fft_res["freqs_AB"],
            "radial": fft_res["radial_AB"],
        }
        print(f"  Cycle L1: {all_results[name]['cycle_l1']:.4f}, "
              f"Degradation: {all_results[name]['degradation_ratio']:.2f}x")

        del trainer
        torch.cuda.empty_cache()

    # ===========================
    # PLOTTING
    # ===========================
    names = list(all_results.keys())

    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.suptitle("FB-CycleGAN Pareto Analysis: Effect of Blur Sigma", fontsize=16, y=1.02)

    # 1. Perturbation curves
    ax = axes[0, 0]
    for (name, r), color in zip(all_results.items(), colors):
        means = [r["perturb"][s] for s in noise_levels]
        stds = [r["perturb_std"][s] for s in noise_levels]
        ax.errorbar(noise_levels, means, yerr=stds, marker="o", capsize=3,
                    linewidth=2, label=name, color=color)
    ax.set_xlabel("Noise Sigma", fontsize=12)
    ax.set_ylabel("L1 Reconstruction Error", fontsize=12)
    ax.set_title("Perturbation Robustness\n(flat = honest, steep = steganographic)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Radial power spectra
    ax = axes[0, 1]
    for (name, r), color in zip(all_results.items(), colors):
        f, p = r["freqs"], r["radial"]
        mask = f > 0
        ax.loglog(f[mask], p[mask], linewidth=2, label=name, color=color)
    ax.set_xlabel("Normalized Frequency", fontsize=12)
    ax.set_ylabel("Power", fontsize=12)
    ax.set_title("Radial Power Spectrum of Residuals\n(lower at high freq = less steganography)",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. Cycle L1 vs sigma
    ax = axes[0, 2]
    sigs = [r["sigma"] for r in all_results.values()]
    l1s = [r["cycle_l1"] for r in all_results.values()]
    l1_stds = [r["cycle_l1_std"] for r in all_results.values()]
    ax.errorbar(sigs, l1s, yerr=l1_stds, marker="s", capsize=5, linewidth=2,
                color="black", markersize=8)
    for s, l1, n in zip(sigs, l1s, names):
        ax.annotate(n, (s, l1), textcoords="offset points", xytext=(5, 8), fontsize=8)
    ax.set_xlabel("Blur Sigma", fontsize=12)
    ax.set_ylabel("Cycle Reconstruction L1", fontsize=12)
    ax.set_title("Honesty Cost: Cycle L1 vs Sigma\n(higher = more honest, lossy translation)",
                 fontsize=12)
    ax.grid(True, alpha=0.3)

    # 4. Degradation ratio bar chart
    ax = axes[1, 0]
    ratios = [r["degradation_ratio"] for r in all_results.values()]
    ax.bar(range(len(names)), ratios, color=colors, edgecolor="black")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, fontsize=9, ha="right")
    ax.set_ylabel("L1 Ratio (noise=0.001 / clean)", fontsize=12)
    ax.set_title("Steganographic Sensitivity\n(>1.5 = steganographic, ~1.0 = honest)", fontsize=12)
    ax.axhline(y=1.5, color="red", linestyle="--", alpha=0.5, label="Steg. threshold")
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Perfect robustness")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # 5. Summary table
    ax = axes[1, 1]
    ax.axis("off")
    header = f"{'Model':>15}  {'Cycle L1':>10}  {'Degrad':>8}  {'Verdict':>15}"
    rows = [header, "-" * 55]
    for name, r in all_results.items():
        verdict = "STEGANOGRAPHIC" if r["degradation_ratio"] > 1.5 else "HONEST"
        rows.append(f"{name:>15}  {r['cycle_l1']:>10.4f}  "
                    f"{r['degradation_ratio']:>7.2f}x  {verdict:>15}")
    ax.text(0.5, 0.5, "\n".join(rows), transform=ax.transAxes, fontsize=11,
            fontfamily="monospace", va="center", ha="center",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.set_title("Summary", fontsize=12)

    # 6. Pareto frontier
    ax = axes[1, 2]
    for (name, r), color in zip(all_results.items(), colors):
        ax.scatter(r["cycle_l1"], r["degradation_ratio"], color=color, s=150,
                   zorder=5, edgecolor="black")
        ax.annotate(name, (r["cycle_l1"], r["degradation_ratio"]),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)
    ax.set_xlabel("Cycle L1 (higher = more honest)", fontsize=12)
    ax.set_ylabel("Degradation Ratio (lower = less steganographic)", fontsize=12)
    ax.set_title("Pareto Frontier: Honesty vs Steganography\n(bottom-right = ideal)", fontsize=12)
    ax.axhline(y=1.5, color="red", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/plots/pareto_analysis.png", dpi=150, bbox_inches="tight")
    print("Saved outputs/plots/pareto_analysis.png")


if __name__ == "__main__":
    main()
