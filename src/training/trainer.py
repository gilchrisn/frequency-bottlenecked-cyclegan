"""Main CycleGAN trainer with optional frequency bottleneck.

Orchestrates the full training loop: generator and discriminator optimization,
cycle-consistency enforcement, identity regularization, checkpointing,
validation, and optional W&B logging.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.config import ExperimentConfig
from src.models import create_generator, create_discriminator
from src.losses import GANLoss, CycleConsistencyLoss, IdentityLoss, create_bottleneck
from src.training.replay_buffer import ReplayBuffer
from src.training.scheduler import create_scheduler
from src.utils import ensure_dir, get_logger


def set_requires_grad(nets: list[nn.Module], requires_grad: bool) -> None:
    """Set requires_grad for all parameters in a list of networks.

    Args:
        nets: List of PyTorch modules.
        requires_grad: Whether to enable or disable gradient computation.
    """
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


class CycleGANTrainer:
    """Trainer for CycleGAN with optional frequency bottleneck.

    Manages model creation, optimization, training loops, validation,
    checkpointing, and logging.

    Args:
        config: Full experiment configuration.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.logger = get_logger("trainer", log_dir=f"{config.output_dir}/logs")

        # --- Models ---
        self.G_AB = create_generator(config.model).to(self.device)
        self.G_BA = create_generator(config.model).to(self.device)
        self.D_A = create_discriminator(config.model).to(self.device)
        self.D_B = create_discriminator(config.model).to(self.device)

        # --- Losses ---
        self.criterion_gan = GANLoss(config.loss.gan_mode).to(self.device)
        self.criterion_cycle = CycleConsistencyLoss(config.loss.lambda_cycle)
        self.criterion_identity = IdentityLoss(config.loss.lambda_identity, config.loss.lambda_cycle)

        # --- Frequency bottleneck ---
        self.bottleneck = create_bottleneck(config.loss).to(self.device)

        # --- Optimizers ---
        self.optimizer_G = torch.optim.Adam(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()),
            lr=config.train.lr_g,
            betas=(config.train.beta1, config.train.beta2),
        )
        self.optimizer_D = torch.optim.Adam(
            list(self.D_A.parameters()) + list(self.D_B.parameters()),
            lr=config.train.lr_d,
            betas=(config.train.beta1, config.train.beta2),
        )

        # --- Schedulers ---
        self.scheduler_G = create_scheduler(self.optimizer_G, config.train)
        self.scheduler_D = create_scheduler(self.optimizer_D, config.train)

        # --- Replay buffers ---
        self.buffer_A = ReplayBuffer(config.train.pool_size)
        self.buffer_B = ReplayBuffer(config.train.pool_size)

        # --- Directories ---
        self.checkpoint_dir = ensure_dir(
            Path(config.output_dir) / "checkpoints" / config.name
        )
        self.sample_dir = ensure_dir(
            Path(config.output_dir) / "samples" / config.name
        )

        # --- W&B ---
        self.wandb_run = None
        if config.use_wandb:
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project="fb-cyclegan",
                    name=config.name,
                    config={
                        "model": vars(config.model),
                        "loss": vars(config.loss),
                        "train": vars(config.train),
                        "data": vars(config.data),
                        "seed": config.seed,
                    },
                )
                self.logger.info("W&B initialized: %s", self.wandb_run.url)
            except Exception as e:
                self.logger.warning("W&B init failed, continuing without: %s", e)
                self.wandb_run = None

        self.logger.info("CycleGANTrainer initialized for '%s'", config.name)

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Run the full training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
        """
        self.logger.info(
            "Starting training for %d epochs", self.config.train.epochs
        )

        for epoch in range(1, self.config.train.epochs + 1):
            self._train_epoch(train_loader, epoch)

            self.scheduler_G.step()
            self.scheduler_D.step()

            if epoch % self.config.train.val_freq == 0:
                self._validate(val_loader, epoch)

            if epoch % self.config.train.save_freq == 0:
                self._save_checkpoint(epoch)

        self._save_checkpoint(self.config.train.epochs, is_final=True)
        self.logger.info("Training complete.")

        if self.wandb_run is not None:
            import wandb
            wandb.finish()

    def _train_epoch(self, loader: DataLoader, epoch: int) -> None:
        """Run one training epoch.

        Args:
            loader: Training data loader yielding dicts with 'A' and 'B' keys.
            epoch: Current epoch number (1-indexed).
        """
        self.G_AB.train()
        self.G_BA.train()
        self.D_A.train()
        self.D_B.train()

        progress = tqdm(
            loader,
            desc=f"Epoch {epoch}/{self.config.train.epochs}",
            leave=False,
        )

        for step, batch in enumerate(progress):
            real_A = batch["A"].to(self.device)
            real_B = batch["B"].to(self.device)

            # =====================
            # Generator optimization
            # =====================
            set_requires_grad([self.D_A, self.D_B], False)
            self.optimizer_G.zero_grad()

            # Forward pass
            fake_B = self.G_AB(real_A)
            fake_A = self.G_BA(real_B)

            # CRITICAL: bottleneck on cycle path only
            rec_A = self.G_BA(self.bottleneck(fake_B))
            rec_B = self.G_AB(self.bottleneck(fake_A))

            # Identity mapping
            idt_A = self.G_BA(real_A)
            idt_B = self.G_AB(real_B)

            # Generator losses
            loss_gan_AB = self.criterion_gan(self.D_B(fake_B), target_is_real=True)
            loss_gan_BA = self.criterion_gan(self.D_A(fake_A), target_is_real=True)
            loss_cycle_A = self.criterion_cycle(rec_A, real_A)
            loss_cycle_B = self.criterion_cycle(rec_B, real_B)
            loss_cycle = loss_cycle_A + loss_cycle_B
            loss_idt_A = self.criterion_identity(idt_A, real_A)
            loss_idt_B = self.criterion_identity(idt_B, real_B)
            loss_idt = loss_idt_A + loss_idt_B

            loss_G = loss_gan_AB + loss_gan_BA + loss_cycle + loss_idt
            loss_G.backward()
            self.optimizer_G.step()

            # ========================
            # Discriminator optimization
            # ========================
            set_requires_grad([self.D_A, self.D_B], True)
            self.optimizer_D.zero_grad()

            # D_A: distinguish real_A from fake_A
            fake_A_buf = self.buffer_A.query(fake_A.detach())
            loss_D_A_real = self.criterion_gan(
                self.D_A(real_A), target_is_real=True
            )
            loss_D_A_fake = self.criterion_gan(
                self.D_A(fake_A_buf), target_is_real=False
            )
            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

            # D_B: distinguish real_B from fake_B
            fake_B_buf = self.buffer_B.query(fake_B.detach())
            loss_D_B_real = self.criterion_gan(
                self.D_B(real_B), target_is_real=True
            )
            loss_D_B_fake = self.criterion_gan(
                self.D_B(fake_B_buf), target_is_real=False
            )
            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

            loss_D = loss_D_A + loss_D_B
            loss_D.backward()
            self.optimizer_D.step()

            # --- Logging ---
            global_step = (epoch - 1) * len(loader) + step
            if global_step % self.config.train.log_freq == 0:
                lr = self.optimizer_G.param_groups[0]["lr"]
                progress.set_postfix(
                    G=f"{loss_G.item():.4f}",
                    D=f"{loss_D.item():.4f}",
                    lr=f"{lr:.6f}",
                )

                if self.wandb_run is not None:
                    import wandb

                    wandb.log(
                        {
                            "loss/G_total": loss_G.item(),
                            "loss/G_gan_AB": loss_gan_AB.item(),
                            "loss/G_gan_BA": loss_gan_BA.item(),
                            "loss/G_cycle": loss_cycle.item(),
                            "loss/G_identity": loss_idt.item(),
                            "loss/D_A": loss_D_A.item(),
                            "loss/D_B": loss_D_B.item(),
                            "lr": lr,
                            "epoch": epoch,
                        },
                        step=global_step,
                    )

    @torch.no_grad()
    def _validate(self, loader: DataLoader, epoch: int) -> None:
        """Generate translation samples and save a visual grid.

        Args:
            loader: Validation data loader.
            epoch: Current epoch number.
        """
        self.G_AB.eval()
        self.G_BA.eval()

        images = []
        for i, batch in enumerate(loader):
            if i >= 4:
                break
            real_A = batch["A"].to(self.device)
            real_B = batch["B"].to(self.device)

            fake_B = self.G_AB(real_A)
            fake_A = self.G_BA(real_B)

            images.extend([real_A, fake_B, real_B, fake_A])

        if images:
            grid = torch.cat(images, dim=0)
            save_path = self.sample_dir / f"epoch_{epoch:04d}.png"
            save_image(grid, save_path, nrow=4, normalize=True, value_range=(-1, 1))
            self.logger.info("Saved validation samples to %s", save_path)

            if self.wandb_run is not None:
                import wandb

                wandb.log(
                    {"val/samples": wandb.Image(str(save_path))},
                    step=epoch,
                )

    def _save_checkpoint(self, epoch: int, is_final: bool = False) -> None:
        """Save model and optimizer states to disk.

        Args:
            epoch: Current epoch number.
            is_final: If True, save as 'final.pt' in addition to epoch file.
        """
        state = {
            "epoch": epoch,
            "G_AB": self.G_AB.state_dict(),
            "G_BA": self.G_BA.state_dict(),
            "D_A": self.D_A.state_dict(),
            "D_B": self.D_B.state_dict(),
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
            "scheduler_G": self.scheduler_G.state_dict(),
            "scheduler_D": self.scheduler_D.state_dict(),
            "config": vars(self.config),
        }

        path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        torch.save(state, path)
        self.logger.info("Checkpoint saved: %s", path)

        if is_final:
            final_path = self.checkpoint_dir / "final.pt"
            torch.save(state, final_path)
            self.logger.info("Final checkpoint saved: %s", final_path)

    def load_checkpoint(self, path: str) -> int:
        """Restore all model and optimizer states from a checkpoint.

        Args:
            path: Path to the checkpoint file.

        Returns:
            The epoch number from the checkpoint.
        """
        self.logger.info("Loading checkpoint: %s", path)
        state = torch.load(path, map_location=self.device, weights_only=False)

        self.G_AB.load_state_dict(state["G_AB"])
        self.G_BA.load_state_dict(state["G_BA"])
        self.D_A.load_state_dict(state["D_A"])
        self.D_B.load_state_dict(state["D_B"])
        self.optimizer_G.load_state_dict(state["optimizer_G"])
        self.optimizer_D.load_state_dict(state["optimizer_D"])
        self.scheduler_G.load_state_dict(state["scheduler_G"])
        self.scheduler_D.load_state_dict(state["scheduler_D"])

        epoch = state["epoch"]
        self.logger.info("Restored from epoch %d", epoch)
        return epoch
