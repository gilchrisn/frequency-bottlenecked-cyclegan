"""Plot training results and evaluation metrics for FB-CycleGAN experiments.

Reads W&B run history CSVs or local JSON files and produces:
  - Training loss curves per experiment
  - FID / SSIM metric comparison bar charts
  - Pareto frontier of FID vs SSIM across sigma values

Usage:
    # Plot loss curves from a W&B exported CSV
    python scripts/plot_results.py --run baseline_cyclegan \
        --loss-csv outputs/baseline_cyclegan/loss_history.csv \
        --output-dir outputs/figures

    # Plot metric comparison from sweep JSON
    python scripts/plot_results.py \
        --metrics-json outputs/sweep_results.json \
        --output-dir outputs/figures

    # Plot Pareto curve from sweep JSON
    python scripts/plot_results.py \
        --metrics-json outputs/sweep_results.json \
        --pareto \
        --output-dir outputs/figures
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.visualization.plotter import (
    plot_loss_curves,
    plot_metric_comparison,
    plot_pareto_curve,
)


def load_loss_csv(csv_path: str) -> dict[str, list[float]]:
    """Load a loss history CSV into a dict of lists.

    Expects a CSV with columns like 'loss/G_total', 'loss/D_A', etc.
    W&B exports use this format.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Dict mapping metric_name -> list of float values.
    """
    df = pd.read_csv(csv_path)
    # Strip W&B 'loss/' prefix for cleaner labels
    rename = {col: col.replace("loss/", "") for col in df.columns}
    df = df.rename(columns=rename)

    history = {}
    for col in df.columns:
        if col.lower() in ("step", "epoch", "_step", "global_step"):
            continue
        vals = df[col].dropna().tolist()
        if vals:
            history[col] = [float(v) for v in vals]
    return history


def load_loss_json(json_path: str) -> dict[str, list[float]]:
    """Load a loss history from a JSON file.

    Expected format: {"loss_name": [v1, v2, ...], ...}
    """
    with open(json_path) as f:
        data = json.load(f)
    return {k: [float(x) for x in v] for k, v in data.items() if isinstance(v, list)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot FB-CycleGAN experiment results.")
    parser.add_argument("--run", type=str, default=None, help="Experiment name (for plot titles).")
    parser.add_argument(
        "--loss-csv",
        type=str,
        default=None,
        help="Path to CSV file with loss history (W&B export or custom).",
    )
    parser.add_argument(
        "--loss-json",
        type=str,
        default=None,
        help="Path to JSON file with loss history.",
    )
    parser.add_argument(
        "--metrics-json",
        type=str,
        default=None,
        help=(
            "Path to JSON file with per-model metrics. "
            "Format: {model_name: {metric: value}}. "
            "For sigma sweep: {sigma: {metric: value}} (use --pareto)."
        ),
    )
    parser.add_argument(
        "--pareto",
        action="store_true",
        help="Interpret metrics-json as sigma sweep and plot Pareto curve.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/figures")
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.9,
        help="EMA smoothing factor for loss curves (0 = no smoothing).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run or "experiment"

    # ------------------------------------------------------------------
    # Loss curves
    # ------------------------------------------------------------------
    if args.loss_csv is not None or args.loss_json is not None:
        if args.loss_csv:
            history = load_loss_csv(args.loss_csv)
        else:
            history = load_loss_json(args.loss_json)

        if history:
            save_path = output_dir / f"{run_name}_loss_curves.png"
            plot_loss_curves(
                history,
                save_path=save_path,
                title=f"Loss Curves — {run_name}",
                smoothing=args.smoothing,
            )
            print(f"Saved loss curves: {save_path}")
        else:
            print("Warning: empty loss history, skipping loss curve plot.")

    # ------------------------------------------------------------------
    # Metrics comparison / Pareto
    # ------------------------------------------------------------------
    if args.metrics_json is not None:
        with open(args.metrics_json) as f:
            raw = json.load(f)

        if args.pareto:
            # sigma sweep: keys are sigma values (possibly as strings)
            sigma_results = {}
            for k, v in raw.items():
                try:
                    sigma_results[float(k)] = v
                except ValueError:
                    pass  # skip non-numeric keys

            if sigma_results:
                save_path = output_dir / "pareto_curve.png"
                plot_pareto_curve(
                    sigma_results,
                    save_path=save_path,
                    x_metric="fid_AB",
                    y_metric="ssim_cycle_A",
                    title="Honesty–Quality Pareto Frontier (FB-CycleGAN σ sweep)",
                )
                print(f"Saved Pareto curve: {save_path}")

                # Also print tabular summary
                print("\nSigma sweep results:")
                sigmas = sorted(sigma_results.keys())
                metrics = list(next(iter(sigma_results.values())).keys())
                header = f"{'sigma':>8}  " + "  ".join(f"{m:>15}" for m in metrics)
                print(header)
                print("-" * len(header))
                for s in sigmas:
                    row = f"{s:>8.2f}  " + "  ".join(
                        f"{sigma_results[s].get(m, float('nan')):>15.4f}" for m in metrics
                    )
                    print(row)
            else:
                print("No valid sigma keys found in metrics JSON.")

        else:
            # Model comparison: keys are model names
            save_path = output_dir / "metric_comparison.png"
            plot_metric_comparison(
                raw,
                save_path=save_path,
                title="FID / SSIM Comparison Across Models",
                higher_is_better={"fid_AB": False, "fid_BA": False,
                                   "ssim_cycle_A": True, "ssim_cycle_B": True},
            )
            print(f"Saved metric comparison: {save_path}")

    if args.loss_csv is None and args.loss_json is None and args.metrics_json is None:
        print("Nothing to plot. Provide --loss-csv, --loss-json, or --metrics-json.")
        parser.print_help()


if __name__ == "__main__":
    main()
