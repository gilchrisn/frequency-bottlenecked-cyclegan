"""Downstream segmentation evaluation for FB-CycleGAN."""

from src.downstream.unet import UNet
from src.downstream.seg_dataset import SegmentationDataset

__all__ = ["UNet", "SegmentationDataset"]
