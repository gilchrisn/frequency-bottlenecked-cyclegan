"""BraTS dataset for unpaired pathological <-> healthy brain MRI translation.

Loads preprocessed 2D .npy slices and pairs them in an unpaired fashion
suitable for CycleGAN training.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import DataConfig


class BraTSDataset(Dataset):
    """Unpaired dataset of pathological (A) and healthy (B) brain MRI slices.

    Attributes:
        paths_A: List of paths to pathological slices.
        paths_B: List of paths to healthy slices.
        transform: Optional transform to apply to loaded tensors.
    """

    def __init__(
        self,
        config: DataConfig,
        split: str = "train",
        transform: Optional[Callable] = None,
        processed_dir: str = "data/processed",
    ) -> None:
        """Initialize the BraTS dataset.

        Args:
            config: Data configuration object.
            split: One of "train", "val", or "test".
            transform: Optional callable transform for data augmentation.
            processed_dir: Path to the preprocessed data directory.
        """
        super().__init__()
        self.config = config
        self.split = split
        self.transform = transform
        self.processed_dir = Path(processed_dir)

        # Load patient-level split
        split_path = self.processed_dir / "split.json"
        with open(split_path, "r") as f:
            split_data = json.load(f)

        patient_ids = set(split_data[split])

        # Collect paths filtered by patient IDs in this split
        self.paths_A = self._filter_paths(
            self.processed_dir / "pathological", patient_ids
        )
        self.paths_B = self._filter_paths(
            self.processed_dir / "healthy", patient_ids
        )

        if len(self.paths_A) == 0:
            raise RuntimeError(
                f"No pathological slices found for split '{split}' "
                f"in {self.processed_dir / 'pathological'}"
            )
        if len(self.paths_B) == 0:
            raise RuntimeError(
                f"No healthy slices found for split '{split}' "
                f"in {self.processed_dir / 'healthy'}"
            )

        print(
            f"BraTSDataset [{split}]: {len(self.paths_A)} pathological, "
            f"{len(self.paths_B)} healthy slices"
        )

    def _filter_paths(
        self, directory: Path, patient_ids: set[str]
    ) -> list[Path]:
        """Filter .npy files by patient ID prefix.

        Args:
            directory: Directory containing .npy slice files.
            patient_ids: Set of patient IDs to include.

        Returns:
            Sorted list of matching file paths.
        """
        if not directory.exists():
            return []

        paths = []
        for npy_path in sorted(directory.glob("*.npy")):
            # Filename format: {patient_id}_slice{NNN}.npy
            # Patient ID is everything before the last _sliceNNN part
            stem = npy_path.stem
            parts = stem.rsplit("_slice", 1)
            if len(parts) == 2:
                pid = parts[0]
                if pid in patient_ids:
                    paths.append(npy_path)
        return paths

    def __len__(self) -> int:
        """Return dataset length as max of both domains."""
        return max(len(self.paths_A), len(self.paths_B))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get an unpaired sample.

        Domain A (pathological) is indexed with wraparound.
        Domain B (healthy) is sampled randomly (unpaired).

        Args:
            index: Sample index.

        Returns:
            Dict with keys "A" and "B", each a float tensor of shape [1, H, W].
        """
        # Domain A: wraparound indexing
        idx_A = index % len(self.paths_A)
        img_A = self._load_slice(self.paths_A[idx_A])

        # Domain B: random sampling (unpaired)
        idx_B = random.randint(0, len(self.paths_B) - 1)
        img_B = self._load_slice(self.paths_B[idx_B])

        # Apply transform
        if self.transform is not None:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def _load_slice(self, path: Path) -> torch.Tensor:
        """Load a .npy slice and convert to tensor.

        Clips to [-3, 3], scales to [-1, 1], and adds a channel dimension.

        Args:
            path: Path to the .npy file.

        Returns:
            Float tensor of shape [1, H, W].
        """
        arr = np.load(path).astype(np.float32)

        # Clip outliers and scale to [-1, 1]
        arr = np.clip(arr, -3.0, 3.0)
        arr = arr / 3.0  # Now in [-1, 1]

        # Convert to tensor with channel dimension [1, H, W]
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor
