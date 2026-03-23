"""Loss functions for FB-CycleGAN training.

Re-exports all loss modules for convenient access:
    - GANLoss: Adversarial loss (LSGAN or vanilla BCE)
    - CycleConsistencyLoss: L1 cycle reconstruction loss
    - IdentityLoss: L1 identity regularization loss
    - FrequencyBottleneck: Fixed Gaussian blur bottleneck
    - IdentityBottleneck: No-op passthrough bottleneck
    - create_bottleneck: Factory for bottleneck selection
"""

from src.losses.adversarial import GANLoss
from src.losses.bottleneck import (
    FrequencyBottleneck,
    IdentityBottleneck,
    create_bottleneck,
)
from src.losses.cycle import CycleConsistencyLoss
from src.losses.identity import IdentityLoss

__all__ = [
    "GANLoss",
    "CycleConsistencyLoss",
    "IdentityLoss",
    "FrequencyBottleneck",
    "IdentityBottleneck",
    "create_bottleneck",
]
