"""Classifier Leakage Test for steganographic detection.

Train a lightweight classifier on generated "healthy" images to predict
properties of the original pathological input (tumor quadrant, size bin).
If the classifier succeeds, the generated images contain hidden information.

Usage:
    python -m scripts.classifier_leakage --checkpoint outputs/checkpoints/baseline/final.pt
    python -m scripts.classifier_leakage --checkpoint outputs/checkpoints/fb/fb_sigma1.0.pt
    python -m scripts.classifier_leakage --compare baseline=outputs/checkpoints/baseline/final.pt fb_s1=outputs/checkpoints/fb/fb_sigma1.0.pt
"""

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import resnet18

from src.config import ExperimentConfig, TrainConfig, DataConfig, LossConfig
from src.training import CycleGANTrainer
from src.data import create_dataset


# =========================================================================
# Step 1: Extract tumor metadata from segmentation masks
# =========================================================================

def extract_tumor_labels(processed_dir: str = "data/processed",
                         raw_dir: str = "data/raw") -> dict:
    """Extract tumor quadrant and size labels for each pathological slice.

    For each slice file in data/processed/pathological/, go back to the
    original segmentation mask and compute:
    - tumor_quadrant: which quadrant of the brain has the most tumor (0-3)
    - tumor_size: small (0), medium (1), large (2) based on area terciles

    Returns:
        Dict mapping slice filename -> {"quadrant": int, "size": int}
    """
    import nibabel as nib
    from scipy.ndimage import zoom as scipy_zoom

    processed = Path(processed_dir)
    raw = Path(raw_dir)
    split_path = processed / "split.json"

    with open(split_path) as f:
        splits = json.load(f)

    # Find all pathological slices
    path_slices = sorted((processed / "pathological").glob("*.npy"))

    # Group slices by patient
    patient_slices = {}
    for p in path_slices:
        stem = p.stem
        parts = stem.rsplit("_slice", 1)
        patient_id = parts[0]
        slice_idx = int(parts[1])
        patient_slices.setdefault(patient_id, []).append((p, slice_idx))

    labels = {}
    tumor_areas = []  # Collect all areas for tercile binning

    # First pass: compute tumor areas
    for patient_id, slices in patient_slices.items():
        # Find seg file
        patient_dirs = list(raw.rglob(f"{patient_id}"))
        if not patient_dirs:
            continue
        pdir = patient_dirs[0]
        seg_files = list(pdir.glob("*_seg.nii.gz")) + list(pdir.glob("*_seg.nii"))
        if not seg_files:
            continue

        seg_vol = np.asarray(nib.load(str(seg_files[0])).dataobj, dtype=np.float32)

        for npy_path, z in slices:
            seg_slice = seg_vol[:, :, z]
            tumor_mask = seg_slice > 0
            tumor_area = tumor_mask.sum()
            tumor_areas.append((npy_path.name, tumor_area, tumor_mask))

    if not tumor_areas:
        print("WARNING: No tumor labels could be extracted.")
        return {}

    # Compute terciles for size binning
    areas = [a for _, a, _ in tumor_areas]
    t1 = np.percentile(areas, 33)
    t2 = np.percentile(areas, 67)

    # Second pass: assign labels
    for patient_id, slices in patient_slices.items():
        patient_dirs = list(raw.rglob(f"{patient_id}"))
        if not patient_dirs:
            continue
        pdir = patient_dirs[0]
        seg_files = list(pdir.glob("*_seg.nii.gz")) + list(pdir.glob("*_seg.nii"))
        if not seg_files:
            continue

        seg_vol = np.asarray(nib.load(str(seg_files[0])).dataobj, dtype=np.float32)

        for npy_path, z in slices:
            seg_slice = seg_vol[:, :, z]
            tumor_mask = seg_slice > 0
            tumor_area = tumor_mask.sum()

            # Size bin
            if tumor_area < t1:
                size_bin = 0  # small
            elif tumor_area < t2:
                size_bin = 1  # medium
            else:
                size_bin = 2  # large

            # Quadrant: split brain into 4 quadrants, find where most tumor is
            h, w = tumor_mask.shape
            q_counts = [
                tumor_mask[:h//2, :w//2].sum(),   # top-left
                tumor_mask[:h//2, w//2:].sum(),    # top-right
                tumor_mask[h//2:, :w//2].sum(),    # bottom-left
                tumor_mask[h//2:, w//2:].sum(),    # bottom-right
            ]
            quadrant = int(np.argmax(q_counts))

            labels[npy_path.name] = {
                "quadrant": quadrant,
                "size": size_bin,
                "area": int(tumor_area),
            }

    print(f"Extracted labels for {len(labels)} pathological slices")
    size_counts = [0, 0, 0]
    quad_counts = [0, 0, 0, 0]
    for v in labels.values():
        size_counts[v["size"]] += 1
        quad_counts[v["quadrant"]] += 1
    print(f"  Size distribution: small={size_counts[0]}, medium={size_counts[1]}, large={size_counts[2]}")
    print(f"  Quadrant distribution: {quad_counts}")

    return labels


# =========================================================================
# Step 2: Dataset that generates "healthy" images and pairs with labels
# =========================================================================

class LeakageDataset(Dataset):
    """Dataset of generated 'healthy' images with tumor property labels.

    For each pathological input x, generates G(x) and labels it with
    the tumor quadrant and size of the original.
    """

    def __init__(self, generated_images: list[torch.Tensor],
                 quadrant_labels: list[int],
                 size_labels: list[int]) -> None:
        self.images = generated_images
        self.quadrants = quadrant_labels
        self.sizes = size_labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        img = self.images[idx]
        # Repeat to 3 channels for ResNet
        img_3ch = img.repeat(3, 1, 1)
        return {
            "image": img_3ch,
            "quadrant": self.quadrants[idx],
            "size": self.sizes[idx],
        }


# =========================================================================
# Step 3: Generate translations and build labelled dataset
# =========================================================================

def generate_leakage_dataset(trainer: CycleGANTrainer,
                             labels: dict,
                             split: str = "val",
                             device: str = "cuda",
                             batch_size: int = 16) -> LeakageDataset:
    """Generate 'healthy' translations and pair with tumor labels (batched)."""
    ds = create_dataset(DataConfig(num_workers=0), split)

    # Filter to slices that have labels
    valid_indices = []
    valid_labels_q = []
    valid_labels_s = []
    for idx, path in enumerate(ds.paths_A):
        if path.name in labels:
            valid_indices.append(idx)
            valid_labels_q.append(labels[path.name]["quadrant"])
            valid_labels_s.append(labels[path.name]["size"])

    images = []
    trainer.G_AB.eval()

    with torch.no_grad():
        # Process in batches
        for start in range(0, len(valid_indices), batch_size):
            end = min(start + batch_size, len(valid_indices))
            batch_tensors = []
            for idx in valid_indices[start:end]:
                img = ds._load_slice(ds.paths_A[idx])
                if ds.transform:
                    img = ds.transform(img)
                batch_tensors.append(img)

            batch = torch.stack(batch_tensors).to(device)
            fake_B = trainer.G_AB(batch)
            images.extend(fake_B.cpu().unbind(0))

            if (start // batch_size) % 20 == 0:
                print(f"    Generated {len(images)}/{len(valid_indices)} translations...", flush=True)

    print(f"Generated {len(images)} labelled translations for split '{split}'")
    return LeakageDataset(images, valid_labels_q, valid_labels_s)


# =========================================================================
# Step 4: Train and evaluate the leakage classifier
# =========================================================================

def train_leakage_classifier(train_dataset: LeakageDataset,
                             val_dataset: LeakageDataset,
                             task: str = "quadrant",
                             epochs: int = 20,
                             lr: float = 1e-3,
                             device: str = "cuda") -> dict:
    """Train a ResNet-18 classifier on generated images.

    Args:
        train_dataset: Training set of generated images + labels.
        val_dataset: Validation set.
        task: Which label to predict — "quadrant" (4 classes) or "size" (3 classes).
        epochs: Number of training epochs.
        lr: Learning rate.
        device: Device.

    Returns:
        Dict with train/val accuracy and random baseline.
    """
    n_classes = 4 if task == "quadrant" else 3
    random_chance = 1.0 / n_classes

    # Build classifier
    model = resnet18(weights=None, num_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        correct, total = 0, 0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            targets = batch[task].to(device)

            logits = model(imgs)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        train_acc = correct / max(total, 1)

        # Validate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                targets = batch[task].to(device)
                logits = model(imgs)
                preds = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        val_acc = correct / max(total, 1)
        best_val_acc = max(best_val_acc, val_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>2}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

    return {
        "task": task,
        "n_classes": n_classes,
        "random_chance": random_chance,
        "best_val_acc": best_val_acc,
        "final_train_acc": train_acc,
        "final_val_acc": val_acc,
    }


# =========================================================================
# Step 5: Main — run leakage test on one or more checkpoints
# =========================================================================

def run_leakage_test(checkpoint_path: str, name: str,
                     labels: dict, device: str) -> dict:
    """Run the full leakage test on one checkpoint."""
    print(f"\n{'='*60}")
    print(f"Classifier Leakage Test: {name}")
    print(f"{'='*60}")

    # Determine if this is an FB model
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

    # Generate datasets
    print("Generating translations for train split...")
    train_ds = generate_leakage_dataset(trainer, labels, split="train", device=device)
    print("Generating translations for val split...")
    val_ds = generate_leakage_dataset(trainer, labels, split="val", device=device)

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("ERROR: No labelled data. Skipping.")
        return {}

    # Run classifier for both tasks
    results = {}
    for task in ["quadrant", "size"]:
        print(f"\nTraining classifier for: {task}")
        r = train_leakage_classifier(train_ds, val_ds, task=task,
                                      epochs=20, device=device)
        results[task] = r
        verdict = "LEAKING" if r["best_val_acc"] > r["random_chance"] * 1.5 else "CLEAN"
        print(f"  Result: best_val_acc={r['best_val_acc']:.3f} "
              f"(chance={r['random_chance']:.3f}) -> {verdict}")

    del trainer
    torch.cuda.empty_cache()
    return results


def main() -> None:
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Classifier Leakage Test")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Single checkpoint to test")
    parser.add_argument("--compare", nargs="+", default=None,
                        help="Multiple name=path pairs to compare")
    parser.add_argument("--output", type=str, default="outputs/plots/classifier_leakage.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Extract tumor labels (one-time, reuse across models)
    print("Extracting tumor labels from segmentation masks...")
    labels = extract_tumor_labels()

    if not labels:
        print("Failed to extract labels. Check data/raw/ has segmentation files.")
        return

    # Determine which checkpoints to test
    checkpoints = {}
    if args.compare:
        for item in args.compare:
            name, path = item.split("=", 1)
            checkpoints[name] = path
    elif args.checkpoint:
        name = Path(args.checkpoint).parent.name
        checkpoints[name] = args.checkpoint
    else:
        # Default: test all available
        checkpoints = {
            "Baseline": "outputs/checkpoints/baseline/final.pt",
            "FB s=0.5": "outputs/checkpoints/fb/fb_sigma0.5.pt",
            "FB s=1.0": "outputs/checkpoints/fb/fb_sigma1.0.pt",
            "FB s=1.5": "outputs/checkpoints/fb/fb_sigma1.5.pt",
            "FB s=2.0": "outputs/checkpoints/fb/fb_sigma2.pt",
        }

    # Run tests
    all_results = {}
    for name, path in checkpoints.items():
        if Path(path).exists():
            all_results[name] = run_leakage_test(path, name, labels, device)

    # Plot comparison
    if len(all_results) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Classifier Leakage Test: Can a Classifier Recover Tumor Info\n"
                     "from Generated 'Healthy' Images?", fontsize=14)

        for ax_idx, task in enumerate(["quadrant", "size"]):
            ax = axes[ax_idx]
            names = []
            accs = []
            chance = None

            for name, results in all_results.items():
                if task in results:
                    names.append(name)
                    accs.append(results[task]["best_val_acc"])
                    chance = results[task]["random_chance"]

            colors = ["red" if a > chance * 1.5 else "green" for a in accs]
            bars = ax.bar(range(len(names)), accs, color=colors, edgecolor="black")
            ax.axhline(y=chance, color="gray", linestyle="--", linewidth=2,
                       label=f"Random chance ({chance:.2f})")
            ax.axhline(y=chance * 1.5, color="orange", linestyle="--", linewidth=1,
                       label=f"Leakage threshold ({chance*1.5:.2f})")
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=25, ha="right", fontsize=10)
            ax.set_ylabel("Best Validation Accuracy", fontsize=12)
            ax.set_title(f"Task: Predict Tumor {task.capitalize()}\n"
                         f"(red = LEAKING, green = CLEAN)", fontsize=12)
            ax.legend(fontsize=9)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSaved comparison plot: {output_path}")

    # Print final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, results in all_results.items():
        print(f"\n{name}:")
        for task, r in results.items():
            verdict = "LEAKING" if r["best_val_acc"] > r["random_chance"] * 1.5 else "CLEAN"
            print(f"  {task}: acc={r['best_val_acc']:.3f} "
                  f"(chance={r['random_chance']:.3f}) -> {verdict}")


if __name__ == "__main__":
    main()
