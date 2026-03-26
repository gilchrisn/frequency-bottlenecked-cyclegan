"""Segmentation dataset: paired (MRI slice, tumor mask) for U-Net training.

Loads preprocessed .npy slices from `data/processed/pathological/` and their
corresponding binary mask .npy files from `data/processed/masks/`.

The masks are created by `scripts/preprocess_masks.py` which mirrors the
preprocessing logic in `scripts/preprocess_brats.py`.

Optionally, a pre-generated synthetic image directory can be supplied to
replace or augment the real images while keeping the same masks, which is
the key mechanism for the downstream evaluation experiment:
  - Train on real pathological images → baseline Dice
  - Train on synthetic images from baseline CycleGAN → Dice_base
  - Train on synthetic images from FB-CycleGAN → Dice_fb
  - Higher Dice_fb confirms FB-CycleGAN produces clinically useful images (H4).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """Paired (image, mask) dataset for brain tumor segmentation.

    Args:
        processed_dir: Root processed data directory (contains split.json,
            pathological/, and masks/ sub-directories).
        split: Dataset split — "train", "val", or "test".
        synthetic_image_dir: Optional directory containing synthetic .npy image
            files with the **same filename stems** as the real pathological
            slices. When provided, synthetic images replace real ones while
            their original masks are kept.
        transform: Optional callable applied independently to both image and
            mask tensors (must be deterministic / shared-seed safe).
        augment: If True, apply random horizontal flip during training.
    """

    def __init__(
        self,
        processed_dir: str = "data/processed",
        split: str = "train",
        synthetic_image_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.synthetic_image_dir = (
            Path(synthetic_image_dir) if synthetic_image_dir is not None else None
        )
        self.transform = transform
        self.augment = augment

        # Load patient split
        split_path = self.processed_dir / "split.json"
        with open(split_path, "r") as f:
            split_data = json.load(f)
        patient_ids = set(split_data[split])

        # Collect paired (image_path, mask_path) for patients in this split
        self.pairs = self._collect_pairs(patient_ids)

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No paired (image, mask) slices found for split '{split}'. "
                "Run `scripts/preprocess_masks.py` to generate mask files."
            )

        print(
            f"SegmentationDataset [{split}]: {len(self.pairs)} paired slices"
            + (f" (synthetic images from {self.synthetic_image_dir})" if self.synthetic_image_dir else "")
        )

    def _collect_pairs(self, patient_ids: set) -> list[tuple[Path, Path]]:
        """Return sorted list of (image_path, mask_path) tuples.

        Skips slices whose mask file is missing.
        """
        path_dir = self.processed_dir / "pathological"
        mask_dir = self.processed_dir / "masks"

        if not path_dir.exists():
            return []

        pairs = []
        for img_path in sorted(path_dir.glob("*.npy")):
            stem = img_path.stem
            parts = stem.rsplit("_slice", 1)
            if len(parts) != 2:
                continue
            pid = parts[0]
            if pid not in patient_ids:
                continue

            mask_path = mask_dir / img_path.name
            if not mask_path.exists():
                continue  # silently skip if mask not yet preprocessed

            pairs.append((img_path, mask_path))

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return a (image, mask) pair.

        Returns:
            Dict with:
              "image": float tensor (1, H, W) in [-1, 1]
              "mask":  float tensor (1, H, W) in {0, 1}
        """
        img_path, mask_path = self.pairs[index]

        # Optionally swap to synthetic image (same mask)
        if self.synthetic_image_dir is not None:
            syn_path = self.synthetic_image_dir / img_path.name
            load_path = syn_path if syn_path.exists() else img_path
        else:
            load_path = img_path

        img_arr = np.load(load_path).astype(np.float32)
        mask_arr = np.load(mask_path).astype(np.float32)

        # Clip and scale image to [-1, 1]
        img_arr = np.clip(img_arr, -3.0, 3.0) / 3.0

        # Binarize mask
        mask_arr = (mask_arr > 0.5).astype(np.float32)

        img = torch.from_numpy(img_arr).unsqueeze(0)   # (1, H, W)
        mask = torch.from_numpy(mask_arr).unsqueeze(0)  # (1, H, W)

        # Random horizontal flip (same seed for both)
        if self.augment and random.random() < 0.5:
            img = img.flip(-1)
            mask = mask.flip(-1)

        if self.transform is not None:
            img = self.transform(img)

        return {"image": img, "mask": mask}
