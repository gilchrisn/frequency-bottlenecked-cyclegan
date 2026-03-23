"""Sweep over Gaussian blur sigma values for Pareto analysis.

Trains an FB-CycleGAN for each sigma value to find the optimal
trade-off between steganography suppression and translation quality.
"""

import argparse

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
        default=200,
        help="Number of training epochs per sigma (default: 200).",
    )
    return parser.parse_args()


def main() -> None:
    """Run training for each sigma value."""
    args = parse_args()

    from src.data import create_dataset, create_dataloader
    from src.training import CycleGANTrainer

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
        )

        print(f"\n{'='*60}")
        print(f"Sweep: sigma={sigma}, kernel_size={kernel_size}, name={name}")
        print(f"{'='*60}\n")

        set_seed(config.seed)

        train_dataset = create_dataset(config.data, split="train")
        val_dataset = create_dataset(config.data, split="val")
        train_loader = create_dataloader(train_dataset, config.train, shuffle=True)
        val_loader = create_dataloader(val_dataset, config.train, shuffle=False)

        trainer = CycleGANTrainer(config)
        trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
