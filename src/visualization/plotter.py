"""Visualization utilities for FB-CycleGAN training and evaluation results.

All functions save figures to disk and return the save path. They do NOT
call plt.show(), so they are safe to use in non-interactive (server/Colab)
environments.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves(
    loss_history: dict[str, list[float]],
    save_path: str | Path,
    title: str = "Training Loss Curves",
    smoothing: float = 0.0,
) -> Path:
    """Plot training loss curves from a history dict.

    Args:
        loss_history: Dict mapping metric name -> list of scalar values
            (one per logged step). Typical keys:
            "G_total", "G_gan_AB", "G_gan_BA", "G_cycle", "G_identity",
            "D_A", "D_B".
        save_path: Output file path.
        title: Figure title.
        smoothing: Exponential moving average smoothing factor in [0, 1).
            0 means no smoothing.

    Returns:
        Resolved Path to the saved figure.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Separate generator and discriminator losses for cleaner layout
    gen_keys = [k for k in loss_history if k.startswith("G")]
    dis_keys = [k for k in loss_history if k.startswith("D")]
    other_keys = [k for k in loss_history if k not in gen_keys and k not in dis_keys]

    groups = []
    if gen_keys:
        groups.append(("Generator Losses", gen_keys))
    if dis_keys:
        groups.append(("Discriminator Losses", dis_keys))
    if other_keys:
        groups.append(("Other", other_keys))

    n_plots = max(len(groups), 1)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    def _smooth(values: list[float], alpha: float) -> list[float]:
        if alpha <= 0:
            return values
        smoothed = []
        ema = values[0]
        for v in values:
            ema = alpha * ema + (1 - alpha) * v
            smoothed.append(ema)
        return smoothed

    for ax, (group_title, keys) in zip(axes, groups):
        for key in keys:
            vals = _smooth(loss_history[key], smoothing)
            ax.plot(vals, label=key, linewidth=1.2)
        ax.set_title(group_title)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Metric comparison bar chart
# ---------------------------------------------------------------------------

def plot_metric_comparison(
    metrics: dict[str, dict[str, float]],
    save_path: str | Path,
    title: str = "Metric Comparison",
    higher_is_better: Optional[dict[str, bool]] = None,
) -> Path:
    """Grouped bar chart comparing metrics across multiple models.

    Args:
        metrics: Dict mapping model_name -> {metric_name: value}.
            Example: {"Baseline": {"FID": 120.3, "SSIM": 0.72},
                      "FB σ=1":   {"FID": 98.1,  "SSIM": 0.79}}
        save_path: Output file path.
        title: Figure title.
        higher_is_better: Dict mapping metric name -> bool.
            Metrics where lower is better are annotated differently.

    Returns:
        Resolved Path to the saved figure.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model_names = list(metrics.keys())
    if not model_names:
        return save_path

    metric_names = list(next(iter(metrics.values())).keys())
    n_metrics = len(metric_names)
    n_models = len(model_names)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4), squeeze=False)

    x = np.arange(n_models)
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    for j, metric in enumerate(metric_names):
        ax = axes[0, j]
        values = [metrics[m].get(metric, 0.0) for m in model_names]
        bars = ax.bar(x, values, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=8)
        ax.set_title(metric)
        ax.set_ylabel("Value")
        ax.grid(True, axis="y", alpha=0.3)

        # Annotate bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max(values),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        if higher_is_better is not None and not higher_is_better.get(metric, True):
            ax.set_title(f"{metric} (lower ↓)")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Pareto curve (FID vs SSIM vs sigma)
# ---------------------------------------------------------------------------

def plot_pareto_curve(
    sigma_results: dict[float, dict[str, float]],
    save_path: str | Path,
    x_metric: str = "fid_AB",
    y_metric: str = "ssim_cycle_A",
    title: str = "Honesty–Quality Pareto Frontier",
) -> Path:
    """Scatter plot of FID vs SSIM across sigma values.

    Each point is one FB-CycleGAN variant. The baseline (sigma=0 / no blur)
    can be included by passing sigma=0.0. A Pareto-optimal frontier emerges
    as the desired operating point for the project.

    Args:
        sigma_results: Dict mapping sigma -> {metric_name: value}.
            Must include x_metric and y_metric keys.
        save_path: Output file path.
        x_metric: Metric for x-axis (default: FID, lower is better).
        y_metric: Metric for y-axis (default: SSIM, higher is better).
        title: Figure title.

    Returns:
        Resolved Path to the saved figure.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    sigmas = sorted(sigma_results.keys())
    x_vals = [sigma_results[s].get(x_metric, 0.0) for s in sigmas]
    y_vals = [sigma_results[s].get(y_metric, 0.0) for s in sigmas]

    fig, ax = plt.subplots(figsize=(7, 5))

    scatter = ax.scatter(x_vals, y_vals, c=sigmas, cmap="viridis", s=80, zorder=5)
    plt.colorbar(scatter, ax=ax, label="σ (blur strength)")

    # Annotate each point with its sigma value
    for sigma, x, y in zip(sigmas, x_vals, y_vals):
        label = "Baseline" if sigma == 0.0 else f"σ={sigma}"
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )

    # Connect points in sigma order (trajectory)
    ax.plot(x_vals, y_vals, "--", color="gray", linewidth=1.0, alpha=0.6)

    ax.set_xlabel(f"{x_metric} (lower is better →)")
    ax.set_ylabel(f"{y_metric} (higher is better ↑)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Sample image grid
# ---------------------------------------------------------------------------

def plot_sample_grid(
    images: dict[str, torch.Tensor],
    save_path: str | Path,
    n_cols: int = 4,
    title: str = "Translation Samples",
    normalize: bool = True,
) -> Path:
    """Save a grid of images labeled by row.

    Args:
        images: Ordered dict mapping row_label -> tensor of shape (N, 1, H, W)
            in [-1, 1]. First n_cols images from each row label are used.
        save_path: Output file path.
        n_cols: Number of columns (images per row).
        title: Figure title.
        normalize: If True, linearly scale each image to [0, 1] for display.

    Returns:
        Resolved Path to the saved figure.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    row_labels = list(images.keys())
    n_rows = len(row_labels)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for r, label in enumerate(row_labels):
        tensor = images[label]  # (N, 1, H, W)
        for c in range(n_cols):
            ax = axes[r, c]
            if c < tensor.shape[0]:
                img = tensor[c, 0].cpu().float().numpy()
                if normalize:
                    lo, hi = img.min(), img.max()
                    if hi > lo:
                        img = (img - lo) / (hi - lo)
                    else:
                        img = img - lo
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=9, rotation=0, labelpad=40, va="center")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# FFT heatmap
# ---------------------------------------------------------------------------

def plot_fft_heatmap(
    spectrum: np.ndarray,
    save_path: str | Path,
    title: str = "FFT Power Spectrum",
    cmap: str = "inferno",
) -> Path:
    """Save a 2D FFT power spectrum as a heatmap.

    Args:
        spectrum: 2D numpy array of log-power values.
        save_path: Output file path.
        title: Figure title.
        cmap: Matplotlib colormap name.

    Returns:
        Resolved Path to the saved figure.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(spectrum, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
