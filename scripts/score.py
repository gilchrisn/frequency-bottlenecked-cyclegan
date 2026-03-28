"""Compute FID and SSIM scores for a trained CycleGAN checkpoint.

Loads a checkpoint, runs inference on the test split, and reports:
  - FID_AB: FID between generated healthy (fake_B) and real healthy (real_B)
  - FID_BA: FID between generated pathological (fake_A) and real pathological (real_A)
  - SSIM_cycle_A: SSIM of cycle-reconstructed real_A vs original real_A
  - SSIM_cycle_B: SSIM of cycle-reconstructed real_B vs original real_B

Results are printed to stdout and optionally saved as a JSON file.

Usage:
    python scripts/score.py \
        --checkpoint outputs/checkpoints/baseline_cyclegan/final.pt \
        --data-dir data/processed \
        --output outputs/scores/baseline.json

    # Score multiple checkpoints and collect into sweep_results.json
    python scripts/score.py \
        --checkpoint outputs/checkpoints/fb_cyclegan_sigma1/final.pt \
        --data-dir data/processed \
        --output outputs/scores/fb_sigma1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ExperimentConfig, ModelConfig
from src.data.brats_dataset import BraTSDataset
from src.data.transforms import get_val_transform
from src.data import create_dataloader
from src.evaluation.metrics import compute_fid, compute_ssim
from src.models import create_generator


def load_generators(checkpoint_path: str, device: torch.device):
    """Load G_AB and G_BA from a checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint.
        device: Target device.

    Returns:
        Tuple of (G_AB, G_BA) in eval mode.
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_cfg = state.get("config", {})
    model_cfg_dict = saved_cfg.get("model", {}) if isinstance(saved_cfg, dict) else {}

    model_cfg = ModelConfig(
        input_channels=model_cfg_dict.get("input_channels", 1),
        output_channels=model_cfg_dict.get("output_channels", 1),
        ngf=model_cfg_dict.get("ngf", 64),
        norm_type=model_cfg_dict.get("norm_type", "instance"),
        no_dropout=model_cfg_dict.get("no_dropout", True),
    )

    G_AB = create_generator(model_cfg).to(device)
    G_BA = create_generator(model_cfg).to(device)
    G_AB.load_state_dict(state["G_AB"])
    G_BA.load_state_dict(state["G_BA"])
    G_AB.eval()
    G_BA.eval()
    return G_AB, G_BA


def score_checkpoint(
    checkpoint_path: str,
    data_dir: str,
    split: str,
    batch_size: int,
    device: torch.device,
    max_samples: int = 0,
) -> dict[str, float]:
    """Run FID and SSIM evaluation on a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        data_dir: Path to preprocessed data directory.
        split: Dataset split to evaluate on ("test", "val", or "train").
        batch_size: Batch size for inference.
        device: Torch device.
        max_samples: Maximum number of samples to use (0 = all).

    Returns:
        Dict with keys: fid_AB, fid_BA, ssim_cycle_A, ssim_cycle_B.
    """
    G_AB, G_BA = load_generators(checkpoint_path, device)

    cfg = ExperimentConfig()
    cfg.train.batch_size = batch_size
    cfg.data.flip = False

    dataset = BraTSDataset(
        config=cfg.data,
        split=split,
        transform=get_val_transform(cfg.data),
        processed_dir=data_dir,
    )
    loader = create_dataloader(dataset, cfg.data, cfg.train, split=split)

    all_real_A = []
    all_real_B = []
    all_fake_A = []
    all_fake_B = []
    all_cycle_A = []
    all_cycle_B = []
    total = 0

    with torch.no_grad():
        for batch in loader:
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)
            cycle_A = G_BA(fake_B)
            cycle_B = G_AB(fake_A)

            all_real_A.append(real_A.cpu())
            all_real_B.append(real_B.cpu())
            all_fake_A.append(fake_A.cpu())
            all_fake_B.append(fake_B.cpu())
            all_cycle_A.append(cycle_A.cpu())
            all_cycle_B.append(cycle_B.cpu())
            total += real_A.size(0)

            if max_samples > 0 and total >= max_samples:
                break

    import torch as _torch
    real_A_cat = _torch.cat(all_real_A, dim=0)
    real_B_cat = _torch.cat(all_real_B, dim=0)
    fake_A_cat = _torch.cat(all_fake_A, dim=0)
    fake_B_cat = _torch.cat(all_fake_B, dim=0)
    cycle_A_cat = _torch.cat(all_cycle_A, dim=0)
    cycle_B_cat = _torch.cat(all_cycle_B, dim=0)

    print(f"  Computing FID (A→B)... ({len(real_B_cat)} samples)")
    fid_AB = compute_fid(real_B_cat, fake_B_cat, device)

    print(f"  Computing FID (B→A)... ({len(real_A_cat)} samples)")
    fid_BA = compute_fid(real_A_cat, fake_A_cat, device)

    print("  Computing SSIM (cycle A)...")
    ssim_cycle_A = compute_ssim(real_A_cat, cycle_A_cat)

    print("  Computing SSIM (cycle B)...")
    ssim_cycle_B = compute_ssim(real_B_cat, cycle_B_cat)

    return {
        "fid_AB": fid_AB,
        "fid_BA": fid_BA,
        "ssim_cycle_A": ssim_cycle_A,
        "ssim_cycle_B": ssim_cycle_B,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a CycleGAN checkpoint with FID and SSIM, or merge score JSONs."
    )
    parser.add_argument(
        "--merge",
        type=str,
        nargs="+",
        metavar="NAME=PATH",
        default=None,
        help=(
            "Merge multiple score JSON files into one comparison file. "
            "Provide pairs of NAME=PATH, e.g. "
            "--merge baseline=outputs/scores/baseline.json fb=outputs/scores/fb_sigma1.json"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .pt file.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Path to preprocessed data directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum samples to evaluate (0 = all).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save results as JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    # --merge mode: combine existing score JSONs into one comparison file
    if args.merge is not None:
        merged = {}
        for entry in args.merge:
            if "=" not in entry:
                parser.error(f"--merge entries must be NAME=PATH, got: {entry!r}")
            name, path = entry.split("=", 1)
            with open(path, "r") as f:
                merged[name] = json.load(f)
        output_path = Path(args.output) if args.output else Path("outputs/scores/comparison.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"Merged {len(merged)} score files → {output_path}")
        return

    if args.checkpoint is None:
        parser.error("--checkpoint is required unless --merge is used.")

    device = torch.device(args.device)
    print(f"Scoring: {args.checkpoint}")
    print(f"Split:   {args.split} | Device: {device}")

    results = score_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        device=device,
        max_samples=args.max_samples,
    )

    print("\n=== Evaluation Results ===")
    for metric, value in results.items():
        print(f"  {metric:20s}: {value:.4f}")

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
