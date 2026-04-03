"""Learned spectral bottleneck for FB-CycleGAN.

Replaces the fixed Gaussian blur with a soft, learnable frequency mask M
trained jointly with the generators but under a separate optimizer.

Why a separate optimizer?
--------------------------
The mask's gradient comes from two competing sources:
  - Cycle loss pushes M → 1 everywhere (let all info through to minimise
    reconstruction error).
  - Sparsity penalty pushes M → 0 everywhere (zero out frequencies).

If the mask shares the generator optimizer, the cycle loss dominates and M
collapses to all-ones — a useless passthrough. A dedicated Adam instance
with a lower learning rate (lr_mask << lr_g) lets the sparsity penalty win
at a controlled rate while generators converge normally.

Usage in trainer
----------------
    # __init__
    self.bottleneck = create_bottleneck(config.loss).to(self.device)
    self.optimizer_mask = build_mask_optimizer(self.bottleneck, config)

    # _train_epoch — after loss_G.backward(), before optimizer_G.step()
    loss_sparsity = sparsity_loss(self.bottleneck, config.loss.mask_sparsity_gamma)
    self.optimizer_mask.zero_grad()
    loss_sparsity.backward()
    self.optimizer_mask.step()

    # _save_checkpoint
    state["bottleneck"] = self.bottleneck.state_dict()
    state["optimizer_mask"] = self.optimizer_mask.state_dict()   # if it exists

    # load_checkpoint
    if "bottleneck" in state:
        self.bottleneck.load_state_dict(state["bottleneck"])
"""

import torch
import torch.nn as nn


class LearnedSpectralBottleneck(nn.Module):
    """Differentiable frequency-domain bottleneck with a learnable soft mask.

    Pipeline per forward pass:
        1. rfft2(x)          — real FFT → complex spectrum (B, C, H, W//2+1)
        2. spectrum * M      — element-wise gate; M broadcast over B and C
        3. irfft2(...)       — back to image space (B, C, H, W)

    The mask M = sigmoid(logits) lives in (0, 1). It is initialised to
    logits = 3.0 so sigmoid(3) ≈ 0.95, making M an approximate passthrough
    at the start of training. This avoids a cold-start where the bottleneck
    immediately destroys the signal before the generators have learned anything.

    Args:
        height: Spatial height of the images (default 256).
        width:  Spatial width  of the images (default 256).
    """

    def __init__(self, height: int = 256, width: int = 256) -> None:
        super().__init__()
        freq_w = width // 2 + 1
        # logits = 3.0  →  sigmoid ≈ 0.95  (near-passthrough at init)
        self.logits = nn.Parameter(torch.full((1, 1, height, freq_w), 3.0))

    # ------------------------------------------------------------------
    # Public helpers (called by trainer)
    # ------------------------------------------------------------------

    def mask(self) -> torch.Tensor:
        """Return the current soft mask M = sigmoid(logits).

        Shape: (1, 1, H, W//2+1) — broadcast-compatible with rfft2 output.
        """
        return torch.sigmoid(self.logits)

    def l1_norm(self) -> torch.Tensor:
        """Mean L1 value of the mask — used as the sparsity penalty term.

        Returns a scalar tensor. Multiply by gamma in the trainer:
            loss_sparsity = gamma * bottleneck.l1_norm()
        """
        return self.mask().mean()

    def bandwidth(self) -> float:
        """Fraction of frequency bins with mask value > 0.5 (no-grad).

        Useful for logging: tracks how many frequencies the mask keeps open
        over training. Should decrease as sparsity penalty takes effect.
        """
        with torch.no_grad():
            return (self.mask() > 0.5).float().mean().item()

    # ------------------------------------------------------------------
    # nn.Module interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the learned spectral mask to x.

        Args:
            x: Image tensor of shape (B, C, H, W).

        Returns:
            Filtered tensor of the same shape (B, C, H, W).
        """
        M = self.mask()                                  # (1, 1, H, W//2+1)
        X_f = torch.fft.rfft2(x)                        # (B, C, H, W//2+1) complex
        X_f_masked = X_f * M                            # broadcast over B, C
        return torch.fft.irfft2(X_f_masked, s=x.shape[-2:])  # (B, C, H, W)


# ---------------------------------------------------------------------------
# Trainer-level helpers — import these in trainer.py
# ---------------------------------------------------------------------------

def build_mask_optimizer(
    bottleneck: nn.Module,
    config,  # ExperimentConfig
) -> torch.optim.Optimizer | None:
    """Create a dedicated Adam optimizer for the spectral mask.

    Returns None if the bottleneck has no learnable parameters (e.g.
    FrequencyBottleneck or IdentityBottleneck), so the trainer can call
    this unconditionally and guard with ``if optimizer_mask is not None``.

    The mask lr is set to config.train.lr_mask if that field exists,
    otherwise falls back to lr_g * 0.1 — intentionally slower than the
    generators so the sparsity penalty converges gradually.

    Args:
        bottleneck: The bottleneck module (any type).
        config:     Full ExperimentConfig.

    Returns:
        Adam optimizer for mask parameters, or None.
    """
    params = list(bottleneck.parameters())
    if not params:
        return None

    lr_mask = getattr(config.train, "lr_mask", config.train.lr_g * 0.1)
    return torch.optim.Adam(params, lr=lr_mask, betas=(0.5, 0.999))


def sparsity_loss(bottleneck: nn.Module, gamma: float) -> torch.Tensor:
    """Return gamma * mask.l1_norm() if bottleneck supports it, else 0.

    Safe to call for any bottleneck type — returns a zero scalar for
    IdentityBottleneck and FrequencyBottleneck.

    Args:
        bottleneck: Any bottleneck module.
        gamma:      Sparsity weight (config.loss.mask_sparsity_gamma).

    Returns:
        Scalar loss tensor (differentiable w.r.t. mask logits).
    """
    if isinstance(bottleneck, LearnedSpectralBottleneck):
        return gamma * bottleneck.l1_norm()
    return torch.tensor(0.0, requires_grad=False)
