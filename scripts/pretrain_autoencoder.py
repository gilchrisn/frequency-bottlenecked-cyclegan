"""Pretrain a convolutional autoencoder on healthy brain slices.

The trained AE learns the manifold of "what healthy brains look like"
and is then used as a frozen bottleneck in the CycleGAN cycle path.

Usage:
    python -m scripts.pretrain_autoencoder
    python -m scripts.pretrain_autoencoder --latent-dim 128 --epochs 50
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses.ae_bottleneck import ConvAutoencoder


class HealthySliceDataset(Dataset):
    """Simple dataset loading healthy .npy slices for AE pretraining."""

    def __init__(self, processed_dir: str, split: str, split_file: str) -> None:
        import json
        processed = Path(processed_dir)
        with open(split_file) as f:
            splits = json.load(f)
        patient_ids = set(splits[split])

        self.paths = []
        for p in sorted((processed / "healthy").glob("*.npy")):
            pid = p.stem.rsplit("_slice", 1)[0]
            if pid in patient_ids:
                self.paths.append(p)

        print(f"HealthySliceDataset [{split}]: {len(self.paths)} slices", flush=True)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        arr = np.load(self.paths[idx]).astype(np.float32)
        arr = np.clip(arr, -3.0, 3.0) / 3.0
        return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain autoencoder on healthy brain slices.")
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--output", type=str, default="outputs/checkpoints/ae_pretrained.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    split_file = str(Path(args.processed_dir) / "split.json")

    train_ds = HealthySliceDataset(args.processed_dir, "train", split_file)
    val_ds = HealthySliceDataset(args.processed_dir, "val", split_file)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = ConvAutoencoder(latent_dim=args.latent_dim, in_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ConvAutoencoder(latent_dim={args.latent_dim}), {n_params:,} params", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Training for {args.epochs} epochs...", flush=True)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            x = batch.to(device)
            recon = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                recon = model(x)
                val_loss += criterion(recon, x).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}: train={train_loss:.6f}, val={val_loss:.6f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "latent_dim": args.latent_dim,
                "epoch": epoch,
                "val_loss": val_loss,
            }, args.output)

    print(f"\nBest val loss: {best_val_loss:.6f}", flush=True)
    print(f"Saved: {args.output}", flush=True)

    # Save loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Autoencoder Pretraining (latent_dim={args.latent_dim})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = str(Path(args.output).with_suffix(".png"))
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Loss curve: {plot_path}", flush=True)

    # Save sample reconstructions
    model.eval()
    batch = next(iter(val_loader))[:8].to(device)
    with torch.no_grad():
        recon = model(batch)

    fig, axes = plt.subplots(2, 8, figsize=(20, 5))
    for i in range(8):
        axes[0, i].imshow(batch[i, 0].cpu(), cmap="gray", vmin=-1, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i, 0].cpu(), cmap="gray", vmin=-1, vmax=1)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=11)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=11)
    plt.suptitle(f"AE Reconstruction Samples (latent_dim={args.latent_dim})", fontsize=13)
    plt.tight_layout()
    samples_path = str(Path(args.output).with_name("ae_samples.png"))
    plt.savefig(samples_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Samples: {samples_path}", flush=True)


if __name__ == "__main__":
    main()
