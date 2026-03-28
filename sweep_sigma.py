"""Sweep over Gaussian blur sigma values for Pareto analysis.

Trains an FB-CycleGAN for each sigma value to find the optimal
trade-off between steganography suppression and translation quality.
After all training runs, scores every checkpoint and writes the
consolidated results to outputs/sweep_results.json.
"""

import argparse
import json
from pathlib import Path

import torch

from src.config import ExperimentConfig, LossConfig, TrainConfig
from src.utils import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for sigma sweep."""
    parser = argparse.ArgumentParser(
        description="Sweep over blur sigma values for FB-CycleGAN."
    )
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0, 3.0],
        help="List of sigma values to sweep (default: 0.5 1.0 1.5 2.0 3.0).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs per sigma (default: 100).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/sweep_results.json",
        help="Path to save consolidated sweep scores (default: outputs/sweep_results.json).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Path to preprocessed data directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for post-sweep scoring (default: 4).",
    )
    return parser.parse_args()


def main() -> None:
    """Run training for each sigma value, then score all checkpoints."""
    args = parse_args()

    from src.data import create_dataset, create_dataloader
    from src.training import CycleGANTrainer
    from scripts.score import score_checkpoint

    sweep_results = {}

    for sigma in args.sigmas:
        kernel_size = max(3, int(sigma * 4) | 1)
        name = f"fb_sigma_{sigma}"

        config = ExperimentConfig(
            name=name,
            loss=LossConfig(
                use_frequency_bottleneck=True,
                blur_kernel_size=kernel_size,
                blur_sigma=sigma,
            ),
            train=TrainConfig(epochs=args.epochs),
            device=args.device,
        )

        print(f"\n{'='*60}")
        print(f"Sweep: sigma={sigma}, kernel_size={kernel_size}, name={name}")
        print(f"{'='*60}\n")

        set_seed(config.seed)

        train_dataset = create_dataset(config.data, split="train")
        val_dataset = create_dataset(config.data, split="val")
        train_loader = create_dataloader(train_dataset, config.data, config.train, split="train")
        val_loader = create_dataloader(val_dataset, config.data, config.train, split="val")

        trainer = CycleGANTrainer(config)
        trainer.train(train_loader, val_loader)

        # Score the finished checkpoint immediately
        checkpoint_path = f"outputs/checkpoints/{name}/final.pt"
        if Path(checkpoint_path).exists():
            print(f"\nScoring {checkpoint_path} ...")
            scores = score_checkpoint(
                checkpoint_path=checkpoint_path,
                data_dir=args.data_dir,
                split="test",
                batch_size=args.batch_size,
                device=torch.device(args.device),
            )
            scores["sigma"] = sigma
            sweep_results[str(sigma)] = scores
            print(f"  sigma={sigma}: {scores}")
        else:
            print(f"  WARNING: checkpoint not found at {checkpoint_path}, skipping scoring.")

    # Write consolidated sweep results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSweep results saved to: {output_path}")


if __name__ == "__main__":
    main()
