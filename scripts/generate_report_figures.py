"""Generate all figures for the full report.

Creates 7 publication-quality figures in outputs/plots/.

Usage:
    python -m scripts.generate_report_figures
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import ExperimentConfig, TrainConfig, DataConfig, LossConfig
from src.training import CycleGANTrainer
from src.data import create_dataset
from scripts.forensic_audit import compute_fft_power, radial_average

MODELS = {
    "Baseline": ("outputs/checkpoints/baseline/final.pt", False, 1.0),
    "FB s=0.5": ("outputs/checkpoints/fb/fb_sigma0.5.pt", True, 0.5),
    "FB s=1.0": ("outputs/checkpoints/fb/fb_sigma1.0.pt", True, 1.0),
    "FB s=1.5": ("outputs/checkpoints/fb/fb_sigma1.5.pt", True, 1.5),
    "FB s=2.0": ("outputs/checkpoints/fb/fb_sigma2.pt", True, 2.0),
}
COLORS = {
    "Baseline": "#e74c3c", "FB s=0.5": "#e67e22", "FB s=1.0": "#3498db",
    "FB s=1.5": "#2ecc71", "FB s=2.0": "#9b59b6",
}
NAMES = list(MODELS.keys())


def load_trainers(device):
    trainers = {}
    for name, (path, bn, sigma) in MODELS.items():
        if not Path(path).exists():
            print(f"  Skipping {name}: {path} not found")
            continue
        ks = max(3, int(sigma * 4) | 1) if sigma > 0 else 5
        cfg = ExperimentConfig(
            name=name, train=TrainConfig(compile_models=False),
            loss=LossConfig(use_frequency_bottleneck=bn, blur_sigma=sigma, blur_kernel_size=ks),
            data=DataConfig(num_workers=0), use_wandb=False, device=device,
        )
        t = CycleGANTrainer(cfg)
        t.load_checkpoint(path)
        trainers[name] = t
    return trainers


def fig1_translation_samples(trainers, loader, device):
    """Translation samples across all models."""
    print("Figure 1: Translation samples...", flush=True)
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle("Figure 1: Translation Samples (Pathological -> Healthy)", fontsize=16, y=1.01)

    for col, name in enumerate(NAMES):
        axes[0, col].set_title(name, fontsize=13, fontweight="bold")

    for r, label in enumerate(["Input (Pathological)", "Generated Healthy G(x)",
                                "Cycle Reconstruction", "Residual |G(x)-x| (5x)"]):
        axes[r, 0].set_ylabel(label, fontsize=10, rotation=90, labelpad=15)

    batch = next(iter(loader))
    real_A = batch["A"].to(device)

    with torch.no_grad():
        for col, (name, t) in enumerate(trainers.items()):
            t.G_AB.eval()
            t.G_BA.eval()
            fake_B = t.G_AB(real_A)
            if MODELS[name][1]:
                rec_A = t.G_BA(t.bottleneck(fake_B))
            else:
                rec_A = t.G_BA(fake_B)
            res = (fake_B - real_A).abs() * 5

            for r, img in enumerate([real_A, fake_B, rec_A, res]):
                im = img[0, 0].cpu().numpy()
                cmap = "hot" if r == 3 else "gray"
                vmin = 0 if r == 3 else -1
                vmax = 1
                axes[r, col].imshow(im, cmap=cmap, vmin=vmin, vmax=vmax)
                axes[r, col].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/plots/fig1_translation_samples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig1", flush=True)


def fig2_perturbation(trainers, loader, device):
    """Perturbation robustness test."""
    print("Figure 2: Perturbation robustness...", flush=True)
    noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    perturb_results = {}

    for name, t in trainers.items():
        t.G_AB.eval()
        t.G_BA.eval()
        results = {s: [] for s in noise_levels}
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= 50:
                    break
                real_A = batch["A"].to(device)
                fake_B = t.G_AB(real_A)
                for s in noise_levels:
                    noise = torch.randn_like(fake_B) * s if s > 0 else torch.zeros_like(fake_B)
                    rec = t.G_BA(fake_B + noise)
                    results[s].append((rec - real_A).abs().mean().item())
        perturb_results[name] = {s: (np.mean(v), np.std(v)) for s, v in results.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Figure 2: Perturbation Robustness Test", fontsize=16)

    for name in NAMES:
        if name not in perturb_results:
            continue
        means = [perturb_results[name][s][0] for s in noise_levels]
        stds = [perturb_results[name][s][1] for s in noise_levels]
        ax1.errorbar(noise_levels, means, yerr=stds, marker="o", capsize=3,
                     linewidth=2.5, label=name, color=COLORS[name], markersize=6)

    ax1.set_xlabel("Noise Sigma", fontsize=12)
    ax1.set_ylabel("L1 Reconstruction Error", fontsize=12)
    ax1.set_title("Reconstruction Error vs Noise\n(steep rise = steganographic)", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ratios = []
    for name in NAMES:
        if name not in perturb_results:
            ratios.append(0)
            continue
        clean = perturb_results[name][0.0][0]
        noisy = perturb_results[name][0.001][0]
        ratios.append(noisy / clean if clean > 0 else 0)

    bars = ax2.bar(range(len(NAMES)), ratios, color=[COLORS[n] for n in NAMES], edgecolor="black")
    ax2.axhline(y=1.5, color="red", linestyle="--", alpha=0.7, label="Steganography threshold")
    ax2.axhline(y=1.0, color="green", linestyle="--", alpha=0.7, label="Perfect robustness")
    ax2.set_xticks(range(len(NAMES)))
    ax2.set_xticklabels(NAMES, rotation=20, ha="right", fontsize=11)
    ax2.set_ylabel("Degradation Ratio", fontsize=12)
    ax2.set_title("Steganographic Sensitivity\n(>1.5 = steganographic)", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")
    for i, r in enumerate(ratios):
        ax2.text(i, r + 0.05, f"{r:.2f}x", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig("outputs/plots/fig2_perturbation_robustness.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig2", flush=True)


def fig3_fft(trainers, loader, device):
    """FFT spectral analysis."""
    print("Figure 3: FFT spectra...", flush=True)
    fft_data = {}
    for name, t in trainers.items():
        t.G_AB.eval()
        powers = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= 50:
                    break
                real_A = batch["A"].to(device)
                fake_B = t.G_AB(real_A)
                res = (fake_B - real_A).abs().cpu().numpy()
                for b in range(res.shape[0]):
                    powers.append(compute_fft_power(res[b, 0]))
        avg_power = np.mean(powers, axis=0)
        freqs, radial = radial_average(avg_power)
        fft_data[name] = {"avg_power": avg_power, "freqs": freqs, "radial": radial}

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Figure 3: FFT Spectral Analysis of Translation Residuals", fontsize=16)

    axes[0].imshow(np.log1p(fft_data["Baseline"]["avg_power"]), cmap="inferno")
    axes[0].set_title("Baseline: 2D Power Spectrum", fontsize=12)
    axes[0].axis("off")

    if "FB s=1.0" in fft_data:
        axes[1].imshow(np.log1p(fft_data["FB s=1.0"]["avg_power"]), cmap="inferno")
    axes[1].set_title("FB s=1.0: 2D Power Spectrum", fontsize=12)
    axes[1].axis("off")

    for name in NAMES:
        if name not in fft_data:
            continue
        f, r = fft_data[name]["freqs"], fft_data[name]["radial"]
        mask = f > 0
        axes[2].loglog(f[mask], r[mask], linewidth=2.5, label=name, color=COLORS[name])
    axes[2].set_xlabel("Normalized Frequency", fontsize=12)
    axes[2].set_ylabel("Power", fontsize=12)
    axes[2].set_title("Radial Power Spectrum (all models)", fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/plots/fig3_fft_spectral.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig3", flush=True)


def fig4_quality():
    """Quality metrics from pre-computed values."""
    print("Figure 4: Quality metrics...", flush=True)
    fid_ab = [101.3, 105.6, 107.5, 104.4, 93.8]
    ssim_ab = [0.289, 0.292, 0.283, 0.292, 0.295]
    ssim_cyc = [0.777, 0.807, 0.787, 0.779, 0.771]
    sigmas = [0, 0.5, 1.0, 1.5, 2.0]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Figure 4: Translation Quality Metrics", fontsize=16)

    x = np.arange(len(NAMES))
    axes[0].bar(x, fid_ab, color=[COLORS[n] for n in NAMES], edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(NAMES, rotation=20, ha="right", fontsize=10)
    axes[0].set_ylabel("FID (lower = better)", fontsize=12)
    axes[0].set_title("FID: Path -> Healthy", fontsize=13)
    axes[0].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(fid_ab):
        axes[0].text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9)

    axes[1].bar(x, ssim_ab, color=[COLORS[n] for n in NAMES], edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(NAMES, rotation=20, ha="right", fontsize=10)
    axes[1].set_ylabel("SSIM (higher = better)", fontsize=12)
    axes[1].set_title("SSIM: Generated vs Real Target", fontsize=13)
    axes[1].set_ylim(0.25, 0.32)
    axes[1].grid(True, alpha=0.3, axis="y")

    axes[2].plot(sigmas, ssim_cyc, "ko-", linewidth=2.5, markersize=8)
    for s, v in zip(sigmas, ssim_cyc):
        axes[2].annotate(f"{v:.3f}", (s, v), textcoords="offset points", xytext=(5, 8), fontsize=10)
    axes[2].set_xlabel("Blur Sigma (0 = baseline)", fontsize=12)
    axes[2].set_ylabel("Cycle SSIM", fontsize=12)
    axes[2].set_title("Cycle Reconstruction vs Sigma", fontsize=13)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/plots/fig4_quality_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig4", flush=True)


def fig5_classifier():
    """Classifier leakage results."""
    print("Figure 5: Classifier leakage...", flush=True)
    quad_accs = [66.9, 76.0, 74.1, 77.1, 79.2]
    size_accs = [53.9, 58.0, 63.6, 55.4, 61.2]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Figure 5: Classifier Leakage Test", fontsize=16)

    for ax_idx, (task, accs, chance) in enumerate([("Quadrant", quad_accs, 25.0), ("Size", size_accs, 33.3)]):
        ax = axes[ax_idx]
        ax.bar(range(len(NAMES)), accs, color=[COLORS[n] for n in NAMES], edgecolor="black")
        ax.axhline(y=chance, color="gray", linestyle="--", linewidth=2, label=f"Chance ({chance:.0f}%)")
        ax.axhline(y=chance * 1.5, color="orange", linestyle="--", linewidth=1.5, label=f"Threshold ({chance*1.5:.0f}%)")
        ax.set_xticks(range(len(NAMES)))
        ax.set_xticklabels(NAMES, rotation=20, ha="right", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title(f"Predict Tumor {task}", fontsize=13)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis="y")
        for i, a in enumerate(accs):
            ax.text(i, a + 1.5, f"{a:.1f}%", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig("outputs/plots/fig5_classifier_leakage.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig5", flush=True)


def fig6_controls():
    """Control experiment results."""
    print("Figure 6: Control experiments...", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Figure 6: Classifier Control Experiments", fontsize=16)

    # 6a: Blur-then-classify
    ax = axes[0]
    models_btc = ["Baseline", "FB s=1.0"]
    clean = [70.1, 75.2]
    blurred = [72.3, 73.0]
    x = np.arange(len(models_btc))
    w = 0.3
    ax.bar(x - w/2, clean, w, label="Clean", color=["#e74c3c", "#3498db"], edgecolor="black")
    ax.bar(x + w/2, blurred, w, label="After blur (s=2.0)", color=["#e74c3c", "#3498db"],
           edgecolor="black", alpha=0.5, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(models_btc, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Blur-then-Classify\n(no drop = visible leakage)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(50, 85)
    ax.grid(True, alpha=0.3, axis="y")

    # 6b: Real healthy control
    ax = axes[1]
    tasks = ["Quadrant", "Size"]
    gen_accs = [66.9, 53.9]
    real_accs = [35.5, 37.2]
    chances = [25.0, 33.3]
    x = np.arange(len(tasks))
    w = 0.25
    ax.bar(x - w, gen_accs, w, label="Generated (Baseline)", color="#e74c3c", edgecolor="black")
    ax.bar(x, real_accs, w, label="Real healthy (random labels)", color="#95a5a6", edgecolor="black")
    ax.bar(x + w, chances, w, label="Random chance", color="white", edgecolor="black", hatch="xx")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Dataset Bias Control\n(real near chance = no bias)", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 80)
    ax.grid(True, alpha=0.3, axis="y")

    # 6c: Pareto
    ax = axes[2]
    quad_accs = [66.9, 76.0, 74.1, 77.1, 79.2]
    degrad = [2.4, 1.03, 1.0, 1.0, 1.0]
    for i, name in enumerate(NAMES):
        ax.scatter(quad_accs[i], degrad[i], color=COLORS[name], s=200, edgecolor="black", zorder=5)
        ax.annotate(name, (quad_accs[i], degrad[i]), textcoords="offset points", xytext=(8, 5), fontsize=10)
    ax.axhline(y=1.5, color="red", linestyle="--", alpha=0.4)
    ax.set_xlabel("Classifier Leakage (% accuracy)", fontsize=12)
    ax.set_ylabel("Perturbation Degradation Ratio", fontsize=12)
    ax.set_title("Steganography vs Structural Leakage\n(bottom-left = ideal)", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/plots/fig6_control_experiments.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig6", flush=True)


def fig7_summary():
    """Summary table as figure."""
    print("Figure 7: Summary table...", flush=True)

    fig, ax = plt.subplots(1, 1, figsize=(18, 4))
    ax.axis("off")

    data = [
        ["Baseline", "2.4x", "0.051", "101.3", "0.289", "0.777", "66.9%", "72.3%", "35.5%"],
        ["FB s=0.5", "1.0x", "0.052", "105.6", "0.292", "0.807", "76.0%", "-", "-"],
        ["FB s=1.0", "1.0x", "0.054", "107.5", "0.283", "0.787", "74.1%", "73.0%", "-"],
        ["FB s=1.5", "1.0x", "0.055", "104.4", "0.292", "0.779", "77.1%", "-", "-"],
        ["FB s=2.0", "1.0x", "0.058", "93.8", "0.295", "0.771", "79.2%", "-", "-"],
    ]
    cols = ["Model", "Perturb\nRatio", "Cycle\nL1", "FID\nA->B", "SSIM\nA->B",
            "SSIM\ncycle", "Classifier\nQuad", "Blur-then\nClassify", "Real Healthy\nControl"]

    table = ax.table(cellText=data, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    for i in range(len(data)):
        cell = table[i + 1, 1]
        if "2.4" in data[i][1]:
            cell.set_facecolor("#ffcccc")
        else:
            cell.set_facecolor("#ccffcc")

    ax.set_title("Figure 7: Comprehensive Results Summary", fontsize=16, pad=20)
    plt.savefig("outputs/plots/fig7_summary_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig7", flush=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path("outputs/plots").mkdir(parents=True, exist_ok=True)

    print("Loading models...", flush=True)
    trainers = load_trainers(device)

    ds = create_dataset(DataConfig(num_workers=0), "val")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    fig1_translation_samples(trainers, loader, device)

    loader2 = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    fig2_perturbation(trainers, loader2, device)

    loader3 = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    fig3_fft(trainers, loader3, device)

    fig4_quality()
    fig5_classifier()
    fig6_controls()
    fig7_summary()

    for t in trainers.values():
        del t
    torch.cuda.empty_cache()

    print("\nAll 7 figures saved to outputs/plots/", flush=True)


if __name__ == "__main__":
    main()
