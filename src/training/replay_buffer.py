"""Image replay buffer for stabilizing GAN discriminator training.

Stores previously generated images and returns a mix of new and buffered
images, following Shrivastava et al. (2017).
"""
from __future__ import annotations

import random

import torch
from torch import Tensor


class ReplayBuffer:
    """Fixed-size buffer that returns a mix of new and previously stored images.

    For each image in a query batch, there is a 50% chance it is swapped
    with a randomly chosen buffered image. The new image then replaces
    the buffered one.

    Args:
        pool_size: Maximum number of images to store. If 0, the buffer
            acts as a passthrough (no buffering).
    """

    def __init__(self, pool_size: int = 50) -> None:
        self.pool_size = pool_size
        self.num_images: int = 0
        self.images: list[Tensor] = []

    def query(self, images: Tensor) -> Tensor:
        """Return a batch with some images potentially swapped from the buffer.

        Args:
            images: Batch of generated images, shape (B, C, H, W).

        Returns:
            Tensor of same shape with some entries replaced by buffered images.
        """
        if self.pool_size == 0:
            return images

        result = []
        for image in images:
            image = image.unsqueeze(0)

            if self.num_images < self.pool_size:
                self.images.append(image.clone())
                self.num_images += 1
                result.append(image)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.pool_size - 1)
                    buffered = self.images[idx].clone()
                    self.images[idx] = image.clone()
                    result.append(buffered)
                else:
                    result.append(image)

        return torch.cat(result, dim=0)
