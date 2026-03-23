"""Picklable transform classes for BraTS data augmentation.

Uses classes (not closures) for Windows multiprocessing compatibility.
All transforms operate on torch tensors of shape [C, H, W].
"""

import random

import torch
import torch.nn.functional as F

from src.config import DataConfig


class TrainTransform:
    """Training-time augmentation: resize, random crop, random horizontal flip.

    Attributes:
        load_size: Size to resize images to before cropping.
        crop_size: Final crop size.
    """

    def __init__(self, config: DataConfig) -> None:
        """Initialize training transform.

        Args:
            config: Data configuration with load_size and crop_size fields.
        """
        self.load_size = config.load_size
        self.crop_size = config.crop_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply training augmentation.

        Args:
            img: Input tensor of shape [C, H, W].

        Returns:
            Augmented tensor of shape [C, crop_size, crop_size].
        """
        # Resize to load_size
        img = F.interpolate(
            img.unsqueeze(0),
            size=(self.load_size, self.load_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Random crop to crop_size
        _, h, w = img.shape
        if h > self.crop_size and w > self.crop_size:
            top = random.randint(0, h - self.crop_size)
            left = random.randint(0, w - self.crop_size)
            img = img[:, top : top + self.crop_size, left : left + self.crop_size]
        elif h != self.crop_size or w != self.crop_size:
            img = F.interpolate(
                img.unsqueeze(0),
                size=(self.crop_size, self.crop_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        # Random horizontal flip
        if random.random() > 0.5:
            img = torch.flip(img, dims=[2])

        return img


class ValTransform:
    """Validation-time transform: resize only (no augmentation).

    Attributes:
        crop_size: Target output size.
    """

    def __init__(self, config: DataConfig) -> None:
        """Initialize validation transform.

        Args:
            config: Data configuration with crop_size field.
        """
        self.crop_size = config.crop_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply validation transform (resize only).

        Args:
            img: Input tensor of shape [C, H, W].

        Returns:
            Resized tensor of shape [C, crop_size, crop_size].
        """
        img = F.interpolate(
            img.unsqueeze(0),
            size=(self.crop_size, self.crop_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return img


def get_train_transform(config: DataConfig) -> TrainTransform:
    """Factory function for training transform.

    Args:
        config: Data configuration object.

    Returns:
        TrainTransform instance.
    """
    return TrainTransform(config)


def get_val_transform(config: DataConfig) -> ValTransform:
    """Factory function for validation transform.

    Args:
        config: Data configuration object.

    Returns:
        ValTransform instance.
    """
    return ValTransform(config)
