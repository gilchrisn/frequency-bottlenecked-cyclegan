"""Compute FID and SSIM for all checkpoints.

Usage:
    python -m scripts.eval_fid_ssim
"""

import torch
import numpy as np
from pathlib import Path
from src.config import ExperimentConfig, TrainConfig, DataConfig, LossConfig
from src.training import CycleGANTrainer
from src.data import create_dataset
from torch.utils.data import DataLoader
from src.evaluation.metrics import compute_ssim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def evaluate_checkpoint(checkpoint_path: str, name: str, device: str,
                        n_samples: int = 200) -> dict:
    """Evaluate FID and SSIM for one checkpoint."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_saved = state.get("config", {})
    if isinstance(cfg_saved, dict) and "loss" in cfg_saved:
        loss_cfg = cfg_saved["loss"]
        bn = getattr(loss_cfg, "use_frequency_bottleneck",
                     loss_cfg.get("use_frequency_bottleneck", False) if isinstance(loss_cfg, dict) else False)
        sigma = getattr(loss_cfg, "blur_sigma",
                        loss_cfg.get("blur_sigma", 1.0) if isinstance(loss_cfg, dict) else 1.0)
    else:
        bn, sigma = False, 1.0

    ks = max(3, int(sigma * 4) | 1) if sigma > 0 else 5
    cfg = ExperimentConfig(
        name=name, train=TrainConfig(compile_models=False),
        loss=LossConfig(use_frequency_bottleneck=bn, blur_sigma=sigma, blur_kernel_size=ks),
        data=DataConfig(num_workers=0), use_wandb=False, device=device,
    )
    trainer = CycleGANTrainer(cfg)
    trainer.load_checkpoint(checkpoint_path)

    ds = create_dataset(cfg.data, "val")
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    real_A_all, real_B_all = [], []
    fake_A_all, fake_B_all = [], []
    cycle_A_all, cycle_B_all = [], []

    trainer.G_AB.eval()
    trainer.G_BA.eval()
    collected = 0

    with torch.no_grad():
        for batch in loader:
            if collected >= n_samples:
                break
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            fake_B = trainer.G_AB(real_A)
            fake_A = trainer.G_BA(real_B)

            if bn:
                cycle_A = trainer.G_BA(trainer.bottleneck(fake_B))
                cycle_B = trainer.G_AB(trainer.bottleneck(fake_A))
            else:
                cycle_A = trainer.G_BA(fake_B)
                cycle_B = trainer.G_AB(fake_A)

            real_A_all.append(real_A.cpu())
            real_B_all.append(real_B.cpu())
            fake_A_all.append(fake_A.cpu())
            fake_B_all.append(fake_B.cpu())
            cycle_A_all.append(cycle_A.cpu())
            cycle_B_all.append(cycle_B.cpu())
            collected += real_A.shape[0]

    real_A_cat = torch.cat(real_A_all)[:n_samples]
    real_B_cat = torch.cat(real_B_all)[:n_samples]
    fake_A_cat = torch.cat(fake_A_all)[:n_samples]
    fake_B_cat = torch.cat(fake_B_all)[:n_samples]
    cycle_A_cat = torch.cat(cycle_A_all)[:n_samples]
    cycle_B_cat = torch.cat(cycle_B_all)[:n_samples]

    # SSIM
    ssim_AB = compute_ssim(real_B_cat, fake_B_cat)
    ssim_BA = compute_ssim(real_A_cat, fake_A_cat)
    ssim_cycle_A = compute_ssim(real_A_cat, cycle_A_cat)
    ssim_cycle_B = compute_ssim(real_B_cat, cycle_B_cat)

    # FID (expensive — use torchmetrics)
    try:
        from src.evaluation.metrics import compute_fid
        fid_AB = compute_fid(real_B_cat, fake_B_cat, torch.device(device))
        fid_BA = compute_fid(real_A_cat, fake_A_cat, torch.device(device))
    except Exception as e:
        print(f"  FID failed: {e}")
        fid_AB, fid_BA = float("nan"), float("nan")

    results = {
        "name": name,
        "sigma": sigma,
        "bottleneck": bn,
        "ssim_AB": ssim_AB,
        "ssim_BA": ssim_BA,
        "ssim_cycle_A": ssim_cycle_A,
        "ssim_cycle_B": ssim_cycle_B,
        "fid_AB": fid_AB,
        "fid_BA": fid_BA,
        "n_samples": min(collected, n_samples),
    }

    del trainer
    torch.cuda.empty_cache()
    return results


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoints = {
        "Baseline": "outputs/checkpoints/baseline/final.pt",
        "FB s=0.5": "outputs/checkpoints/fb/fb_sigma0.5.pt",
        "FB s=1.0": "outputs/checkpoints/fb/fb_sigma1.0.pt",
        "FB s=1.5": "outputs/checkpoints/fb/fb_sigma1.5.pt",
        "FB s=2.0": "outputs/checkpoints/fb/fb_sigma2.pt",
    }

    all_results = []
    for name, path in checkpoints.items():
        if not Path(path).exists():
            print(f"Skipping {name}: {path} not found")
            continue
        print(f"\nEvaluating {name}...")
        r = evaluate_checkpoint(path, name, device, n_samples=200)
        all_results.append(r)
        print(f"  SSIM(A->B): {r['ssim_AB']:.4f}  SSIM(B->A): {r['ssim_BA']:.4f}")
        print(f"  SSIM cycle A: {r['ssim_cycle_A']:.4f}  SSIM cycle B: {r['ssim_cycle_B']:.4f}")
        print(f"  FID(A->B): {r['fid_AB']:.1f}  FID(B->A): {r['fid_BA']:.1f}")

    # Print table
    print(f"\n{'='*80}")
    print(f"{'Model':>12} {'s':>5} {'SSIM_AB':>8} {'SSIM_BA':>8} {'SSIM_cyc':>9} {'FID_AB':>8} {'FID_BA':>8}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['name']:>12} {r['sigma']:>5.1f} {r['ssim_AB']:>8.4f} {r['ssim_BA']:>8.4f} "
              f"{r['ssim_cycle_A']:>9.4f} {r['fid_AB']:>8.1f} {r['fid_BA']:>8.1f}")

    # Save results
    import json
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    with open("outputs/metrics/fid_ssim_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Plot
    if len(all_results) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Translation Quality Metrics Across Sigma Values", fontsize=14)

        sigmas = [r["sigma"] for r in all_results]
        names = [r["name"] for r in all_results]

        # FID
        ax = axes[0]
        fids = [r["fid_AB"] for r in all_results]
        ax.bar(range(len(names)), fids, color=["red", "orange", "blue", "green", "purple"],
               edgecolor="black")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("FID (lower = better)")
        ax.set_title("FID: Path -> Healthy")
        ax.grid(True, alpha=0.3, axis="y")

        # SSIM translation
        ax = axes[1]
        ssims = [r["ssim_AB"] for r in all_results]
        ax.bar(range(len(names)), ssims, color=["red", "orange", "blue", "green", "purple"],
               edgecolor="black")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("SSIM (higher = better)")
        ax.set_title("SSIM: Generated vs Real Healthy")
        ax.grid(True, alpha=0.3, axis="y")

        # SSIM cycle
        ax = axes[2]
        ssim_cyc = [r["ssim_cycle_A"] for r in all_results]
        ax.bar(range(len(names)), ssim_cyc, color=["red", "orange", "blue", "green", "purple"],
               edgecolor="black")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("SSIM (higher = better)")
        ax.set_title("SSIM: Cycle Reconstruction")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig("outputs/plots/fid_ssim_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSaved plot: outputs/plots/fid_ssim_comparison.png")


if __name__ == "__main__":
    main()
