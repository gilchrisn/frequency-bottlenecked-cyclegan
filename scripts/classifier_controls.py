"""Follow-up classifier experiments to distinguish steganographic vs structural leakage.

Three experiments:
1. Blur-then-classify: blur generated images before classifier to destroy high-freq encoding
2. Real healthy control: run classifier on real healthy images to check for dataset bias
3. GradCAM attention maps: visualize where the classifier looks

Usage:
    python -m scripts.classifier_controls
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import ExperimentConfig, TrainConfig, DataConfig, LossConfig
from src.training import CycleGANTrainer
from src.data import create_dataset
from src.losses.bottleneck import FrequencyBottleneck
from scripts.classifier_leakage import (
    extract_tumor_labels, LeakageDataset, generate_leakage_dataset,
    train_leakage_classifier,
)


# =========================================================================
# Experiment 1: Blur-then-classify
# =========================================================================

def blur_dataset(dataset: LeakageDataset, sigma: float, kernel_size: int) -> LeakageDataset:
    """Apply Gaussian blur to all images in a LeakageDataset."""
    blur = FrequencyBottleneck(kernel_size, sigma)
    blurred_images = []
    for img in dataset.images:
        # img is [1, H, W], blur expects [B, C, H, W]
        blurred = blur(img.unsqueeze(0)).squeeze(0)
        blurred_images.append(blurred)
    return LeakageDataset(blurred_images, dataset.quadrants, dataset.sizes)


def run_blur_then_classify(labels: dict, device: str) -> dict:
    """Experiment 1: Blur generated images before classification.

    If baseline accuracy drops but FB stays same -> baseline used high-freq encoding.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Blur-then-Classify")
    print("=" * 60)

    checkpoints = {
        "Baseline": ("outputs/checkpoints/baseline/final.pt", False, 1.0),
        "FB s=1.0": ("outputs/checkpoints/fb/fb_sigma1.0.pt", True, 1.0),
    }

    blur_sigma = 2.0
    blur_ks = 9
    results = {}

    for name, (path, bn, sigma) in checkpoints.items():
        if not Path(path).exists():
            continue

        print(f"\n--- {name} ---")
        ks = max(3, int(sigma * 4) | 1) if sigma > 0 else 5
        cfg = ExperimentConfig(
            name=name, train=TrainConfig(compile_models=False),
            loss=LossConfig(use_frequency_bottleneck=bn, blur_sigma=sigma, blur_kernel_size=ks),
            data=DataConfig(num_workers=0), use_wandb=False, device=device,
        )
        trainer = CycleGANTrainer(cfg)
        trainer.load_checkpoint(path)

        # Generate translations
        print("  Generating val translations...", flush=True)
        train_ds = generate_leakage_dataset(trainer, labels, "train", device, batch_size=16)
        val_ds = generate_leakage_dataset(trainer, labels, "val", device, batch_size=16)

        del trainer
        torch.cuda.empty_cache()

        # Unblurred classifier
        print("  Training classifier (unblurred)...", flush=True)
        r_clean = train_leakage_classifier(train_ds, val_ds, "quadrant", epochs=15, device=device)

        # Blurred classifier
        print(f"  Blurring images with sigma={blur_sigma}...", flush=True)
        train_ds_blur = blur_dataset(train_ds, blur_sigma, blur_ks)
        val_ds_blur = blur_dataset(val_ds, blur_sigma, blur_ks)

        print("  Training classifier (blurred)...", flush=True)
        r_blur = train_leakage_classifier(train_ds_blur, val_ds_blur, "quadrant", epochs=15, device=device)

        results[name] = {
            "clean_acc": r_clean["best_val_acc"],
            "blurred_acc": r_blur["best_val_acc"],
            "drop": r_clean["best_val_acc"] - r_blur["best_val_acc"],
        }
        print(f"  Clean: {r_clean['best_val_acc']:.3f} -> Blurred: {r_blur['best_val_acc']:.3f} "
              f"(drop: {results[name]['drop']:.3f})", flush=True)

    return results


# =========================================================================
# Experiment 2: Real healthy control (dataset bias check)
# =========================================================================

class RealHealthyDataset(Dataset):
    """Real healthy images with random quadrant/size labels as control."""

    def __init__(self, config: DataConfig, split: str, labels: dict) -> None:
        ds = create_dataset(config, split)
        self.images = []
        self.quadrants = []
        self.sizes = []

        # Collect all real healthy images
        for path in ds.paths_B:
            arr = np.load(path).astype(np.float32)
            arr = np.clip(arr, -3.0, 3.0) / 3.0
            tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]
            self.images.append(tensor)

        # Assign random labels (from the pathological distribution)
        all_quads = [v["quadrant"] for v in labels.values()]
        all_sizes = [v["size"] for v in labels.values()]
        rng = random.Random(42)
        for _ in range(len(self.images)):
            self.quadrants.append(rng.choice(all_quads))
            self.sizes.append(rng.choice(all_sizes))

        print(f"RealHealthyDataset [{split}]: {len(self.images)} images", flush=True)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        img = self.images[idx].repeat(3, 1, 1)
        return {
            "image": img,
            "quadrant": self.quadrants[idx],
            "size": self.sizes[idx],
        }


def run_real_healthy_control(labels: dict, device: str) -> dict:
    """Experiment 2: Can the classifier predict random labels from real healthy images?

    If yes -> dataset bias exists. If no -> leakage comes from the generator.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Real Healthy Control (Dataset Bias Check)")
    print("=" * 60)

    cfg = DataConfig(num_workers=0)
    train_ds = RealHealthyDataset(cfg, "train", labels)
    val_ds = RealHealthyDataset(cfg, "val", labels)

    results = {}
    for task in ["quadrant", "size"]:
        print(f"\n  Training classifier for: {task}", flush=True)
        r = train_leakage_classifier(train_ds, val_ds, task, epochs=15, device=device)
        results[task] = r
        print(f"  Result: {r['best_val_acc']:.3f} (chance={r['random_chance']:.3f})", flush=True)

    return results


# =========================================================================
# Experiment 3: GradCAM attention maps
# =========================================================================

def compute_gradcam(model: nn.Module, images: torch.Tensor,
                    targets: torch.Tensor, device: str) -> np.ndarray:
    """Compute GradCAM heatmaps for a batch of images.

    Uses the last conv layer of ResNet-18 (layer4).
    """
    model.eval()
    images = images.to(device)
    targets = targets.to(device)

    # Hook into last conv layer
    activations = []
    gradients = []

    def fwd_hook(module, input, output):
        activations.append(output.detach())

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    # ResNet-18 last conv block
    target_layer = model.layer4[-1].conv2
    fwd_handle = target_layer.register_forward_hook(fwd_hook)
    bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

    # Forward
    logits = model(images)
    loss = F.cross_entropy(logits, targets)
    model.zero_grad()
    loss.backward()

    fwd_handle.remove()
    bwd_handle.remove()

    # Compute GradCAM
    act = activations[0]   # [B, C, H', W']
    grad = gradients[0]    # [B, C, H', W']
    weights = grad.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
    cam = (weights * act).sum(dim=1, keepdim=True)  # [B, 1, H', W']
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze(1).cpu().numpy()  # [B, H, W]

    # Normalize per image
    for i in range(cam.shape[0]):
        cam_min, cam_max = cam[i].min(), cam[i].max()
        if cam_max - cam_min > 1e-8:
            cam[i] = (cam[i] - cam_min) / (cam_max - cam_min)

    return cam


def run_gradcam(labels: dict, device: str) -> None:
    """Experiment 3: GradCAM attention maps comparing baseline vs FB classifier focus."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: GradCAM Attention Maps")
    print("=" * 60)

    checkpoints = {
        "Baseline": ("outputs/checkpoints/baseline/final.pt", False, 1.0),
        "FB s=1.0": ("outputs/checkpoints/fb/fb_sigma1.0.pt", True, 1.0),
    }

    n_samples = 6
    fig, axes = plt.subplots(len(checkpoints), n_samples * 2, figsize=(4 * n_samples, 4 * len(checkpoints)))
    fig.suptitle("GradCAM: Where Does the Classifier Look to Predict Tumor Quadrant?", fontsize=14, y=1.02)

    for row, (name, (path, bn, sigma)) in enumerate(checkpoints.items()):
        if not Path(path).exists():
            continue

        print(f"\n--- {name} ---", flush=True)
        ks = max(3, int(sigma * 4) | 1) if sigma > 0 else 5
        cfg = ExperimentConfig(
            name=name, train=TrainConfig(compile_models=False),
            loss=LossConfig(use_frequency_bottleneck=bn, blur_sigma=sigma, blur_kernel_size=ks),
            data=DataConfig(num_workers=0), use_wandb=False, device=device,
        )
        trainer = CycleGANTrainer(cfg)
        trainer.load_checkpoint(path)

        # Generate val translations
        val_ds = generate_leakage_dataset(trainer, labels, "val", device, batch_size=16)
        del trainer
        torch.cuda.empty_cache()

        # Train a quick classifier
        print("  Training classifier for GradCAM...", flush=True)
        train_ds = generate_leakage_dataset(
            CycleGANTrainer(cfg).load_checkpoint(path) or CycleGANTrainer(cfg),
            labels, "train", device, batch_size=16
        ) if False else val_ds  # Use val as both for speed in GradCAM

        model = resnet18(weights=None, num_classes=4).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(val_ds, batch_size=32, shuffle=True, num_workers=0)

        # Quick train (10 epochs)
        for epoch in range(10):
            model.train()
            for batch in loader:
                imgs = batch["image"].to(device)
                tgts = batch["quadrant"].to(device)
                loss = criterion(model(imgs), tgts)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Get GradCAM for first n_samples
        sample_loader = DataLoader(val_ds, batch_size=n_samples, shuffle=False, num_workers=0)
        batch = next(iter(sample_loader))
        images = batch["image"]
        targets = torch.tensor(batch["quadrant"])

        cams = compute_gradcam(model, images, targets, device)

        for col in range(n_samples):
            # Original image (grayscale channel of the 3ch repeat)
            img = images[col, 0].numpy()
            axes[row, col * 2].imshow(img, cmap="gray", vmin=-1, vmax=1)
            axes[row, col * 2].set_title(f"Q={batch['quadrant'][col]}", fontsize=9)
            axes[row, col * 2].axis("off")

            # GradCAM overlay
            axes[row, col * 2 + 1].imshow(img, cmap="gray", vmin=-1, vmax=1)
            axes[row, col * 2 + 1].imshow(cams[col], cmap="jet", alpha=0.5, vmin=0, vmax=1)
            axes[row, col * 2 + 1].set_title("GradCAM", fontsize=9)
            axes[row, col * 2 + 1].axis("off")

        axes[row, 0].set_ylabel(name, fontsize=12, rotation=90, labelpad=15)

        del model
        torch.cuda.empty_cache()

    plt.tight_layout()
    plt.savefig("outputs/plots/gradcam_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: outputs/plots/gradcam_comparison.png", flush=True)


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Extracting tumor labels...", flush=True)
    labels = extract_tumor_labels()

    # Experiment 1: Blur-then-classify
    blur_results = run_blur_then_classify(labels, device)

    # Experiment 2: Real healthy control
    control_results = run_real_healthy_control(labels, device)

    # Experiment 3: GradCAM
    run_gradcam(labels, device)

    # Final summary
    print("\n" + "=" * 60)
    print("FULL SUMMARY")
    print("=" * 60)

    print("\nExp 1 — Blur-then-Classify (quadrant):")
    for name, r in blur_results.items():
        print(f"  {name}: clean={r['clean_acc']:.3f} -> blurred={r['blurred_acc']:.3f} "
              f"(drop={r['drop']:.3f})")

    print("\nExp 2 — Real Healthy Control:")
    for task, r in control_results.items():
        verdict = "BIAS" if r["best_val_acc"] > r["random_chance"] * 1.5 else "NO BIAS"
        print(f"  {task}: acc={r['best_val_acc']:.3f} (chance={r['random_chance']:.3f}) -> {verdict}")

    print("\nExp 3 — GradCAM saved to outputs/plots/gradcam_comparison.png")

    # Save results
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    with open("outputs/metrics/classifier_controls.json", "w") as f:
        json.dump({
            "blur_then_classify": blur_results,
            "real_healthy_control": {k: {kk: vv for kk, vv in v.items()} for k, v in control_results.items()},
        }, f, indent=2, default=str)
    print("\nSaved: outputs/metrics/classifier_controls.json", flush=True)


if __name__ == "__main__":
    main()
