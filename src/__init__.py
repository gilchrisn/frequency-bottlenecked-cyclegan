"""Public API for src.losses.

Exports all loss classes, bottleneck modules, and trainer-level helpers so
the rest of the codebase can import from ``src.losses`` without knowing the
internal module layout.
"""

from src.losses.adversarial import GANLoss
from src.losses.cycle import CycleConsistencyLoss
from src.losses.identity import IdentityLoss
from src.losses.bottleneck import (
    FrequencyBottleneck,
    IdentityBottleneck,
    create_bottleneck,
)
from src.losses.learned_mask import (
    LearnedSpectralBottleneck,
    build_mask_optimizer,
    sparsity_loss,
)

__all__ = [
    # Standard losses
    "GANLoss",
    "CycleConsistencyLoss",
    "IdentityLoss",
    # Bottleneck modules
    "FrequencyBottleneck",
    "IdentityBottleneck",
    "LearnedSpectralBottleneck",
    # Factory + trainer helpers
    "create_bottleneck",
    "build_mask_optimizer",
    "sparsity_loss",
]
