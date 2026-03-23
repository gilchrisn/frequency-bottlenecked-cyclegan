"""Learning rate scheduler factory for CycleGAN training.

Supports linear decay policy: constant LR for the first N epochs, then
linearly decay to zero over the remaining epochs.
"""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

from src.config import TrainConfig


def create_scheduler(optimizer: Optimizer, config: TrainConfig) -> _LRScheduler:
    """Create a learning rate scheduler based on the training config.

    Args:
        optimizer: The optimizer whose LR will be scheduled.
        config: Training configuration specifying lr_policy and parameters.

    Returns:
        A PyTorch LR scheduler instance.

    Raises:
        ValueError: If config.lr_policy is not recognized.
    """
    if config.lr_policy == "linear":
        return _create_linear_decay(
            optimizer,
            decay_start=config.lr_decay_start,
            total_epochs=config.epochs,
        )
    else:
        raise ValueError(
            f"Unknown lr_policy '{config.lr_policy}'. Supported: 'linear'."
        )


def _create_linear_decay(
    optimizer: Optimizer,
    decay_start: int,
    total_epochs: int,
) -> LambdaLR:
    """Create a linear decay scheduler.

    Learning rate remains constant for the first `decay_start` epochs,
    then decays linearly to zero over the remaining epochs.

    Args:
        optimizer: The optimizer whose LR will be scheduled.
        decay_start: Epoch at which decay begins.
        total_epochs: Total number of training epochs.

    Returns:
        LambdaLR scheduler implementing the linear decay policy.
    """
    decay_epochs = total_epochs - decay_start

    def lr_lambda(epoch: int) -> float:
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / decay_epochs)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
