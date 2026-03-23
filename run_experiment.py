"""Entry point for running FB-CycleGAN experiments.

Supports both training and evaluation modes via command-line arguments.
Uses lazy imports to avoid loading heavy modules unless needed.
"""

import argparse
import sys


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    from src.config import PRESETS

    parser = argparse.ArgumentParser(
        description="Run FB-CycleGAN training or evaluation."
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(PRESETS.keys()),
        default="baseline_cyclegan",
        help="Experiment preset name (default: baseline_cyclegan).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only (requires --checkpoint).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file for evaluation or resuming training.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point dispatching to training or evaluation."""
    args = parse_args()

    from src.config import get_config
    from src.utils import set_seed

    config = get_config(args.preset)
    set_seed(config.seed)

    if args.eval_only:
        if args.checkpoint is None:
            print("Error: --checkpoint is required with --eval-only.")
            sys.exit(1)

        from src.training import CycleGANTrainer
        from src.evaluation import evaluate_model

        trainer = CycleGANTrainer(config)
        trainer.load_checkpoint(args.checkpoint)
        results = evaluate_model(trainer, config)
        print("Evaluation results:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")
    else:
        from src.data import create_dataset, create_dataloader
        from src.training import CycleGANTrainer

        train_dataset = create_dataset(config.data, split="train")
        val_dataset = create_dataset(config.data, split="val")
        train_loader = create_dataloader(train_dataset, config.train, shuffle=True)
        val_loader = create_dataloader(val_dataset, config.train, shuffle=False)

        trainer = CycleGANTrainer(config)
        trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
