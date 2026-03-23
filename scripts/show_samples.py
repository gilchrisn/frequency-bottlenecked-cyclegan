"""Show sample images from the preprocessed BraTS dataset.

Displays a grid of randomly selected pathological and healthy slices.
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Load and display sample preprocessed slices."""
    parser = argparse.ArgumentParser(
        description="Show sample images from preprocessed BraTS dataset."
    )
    parser.add_argument(
        "--n", type=int, default=8, help="Number of samples per class."
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Path to preprocessed data directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/plots/dataset_samples.png",
        help="Output path for the sample grid image.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    processed_dir = Path(args.processed_dir)

    # Collect sample paths
    path_files = sorted((processed_dir / "pathological").glob("*.npy"))
    health_files = sorted((processed_dir / "healthy").glob("*.npy"))

    n = min(args.n, len(path_files), len(health_files))
    if n == 0:
        print("No samples found. Run preprocess_brats.py first.")
        return

    path_samples = rng.sample(path_files, n)
    health_samples = rng.sample(health_files, n)

    # Create figure: 2 rows (pathological on top, healthy on bottom)
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 5))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col in range(n):
        # Pathological (top row)
        img_p = np.load(path_samples[col])
        axes[0, col].imshow(img_p, cmap="gray", vmin=-2, vmax=2)
        axes[0, col].set_title("Pathological", fontsize=8)
        axes[0, col].axis("off")

        # Healthy (bottom row)
        img_h = np.load(health_samples[col])
        axes[1, col].imshow(img_h, cmap="gray", vmin=-2, vmax=2)
        axes[1, col].set_title("Healthy", fontsize=8)
        axes[1, col].axis("off")

    plt.suptitle("BraTS Dataset Samples", fontsize=12)
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved sample grid to {output_path}")


if __name__ == "__main__":
    main()
