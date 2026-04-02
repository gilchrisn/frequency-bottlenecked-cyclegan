"""Shared utility functions for the FB-CycleGAN project."""

import logging
import os
import random
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_logger(
    name: str,
    log_dir: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """Create a logger with console output and optional file handler.

    Args:
        name: Logger name (typically module name).
        log_dir: If provided, also write logs to a file in this directory.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s %(name)s %(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / f"{name}.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: PyTorch module.

    Returns:
        Total number of parameters with requires_grad=True.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory (and parents) if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The Path object for the created directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
