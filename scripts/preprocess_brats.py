"""One-time BraTS preprocessing: NIfTI -> 2D .npy slices.

Extracts axial FLAIR slices from BraTS 2020 volumes and classifies them as
pathological (tumor present) or healthy (no tumor). Saves preprocessed slices
as individual .npy files for efficient data loading during training.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def load_nifti(path: str | Path) -> np.ndarray:
    """Load a NIfTI volume (.nii or .nii.gz).

    Args:
        path: Path to the NIfTI file.

    Returns:
        3D numpy array of voxel intensities.
    """
    img = nib.load(str(path))
    return img.get_fdata().astype(np.float32)


def find_patient_dirs(raw_dir: str | Path) -> list[Path]:
    """Find all patient directories containing FLAIR sequences.

    Args:
        raw_dir: Root directory of the raw BraTS dataset.

    Returns:
        Sorted list of unique parent directories containing FLAIR files.
    """
    raw_dir = Path(raw_dir)
    flair_files = list(raw_dir.rglob("*_flair.nii.gz")) + list(
        raw_dir.rglob("*_flair.nii")
    )
    patient_dirs = sorted(set(f.parent for f in flair_files))
    return patient_dirs


def get_brain_bbox(
    slice_2d: np.ndarray, margin: int = 5
) -> Optional[tuple[int, int, int, int]]:
    """Compute bounding box of non-zero region in a 2D slice.

    Args:
        slice_2d: 2D numpy array.
        margin: Pixels of padding around the bounding box.

    Returns:
        Tuple of (row_min, row_max, col_min, col_max) or None if empty.
    """
    nonzero = np.nonzero(slice_2d)
    if len(nonzero[0]) == 0:
        return None

    row_min = max(0, nonzero[0].min() - margin)
    row_max = min(slice_2d.shape[0], nonzero[0].max() + margin + 1)
    col_min = max(0, nonzero[1].min() - margin)
    col_max = min(slice_2d.shape[1], nonzero[1].max() + margin + 1)

    return (row_min, row_max, col_min, col_max)


def preprocess_slice(
    slice_2d: np.ndarray,
    vol_mean: float,
    vol_std: float,
    target_size: int = 256,
) -> np.ndarray:
    """Center-crop brain region, resize, and Z-score normalize.

    Args:
        slice_2d: Raw 2D axial slice.
        vol_mean: Mean intensity of the brain region in the volume.
        vol_std: Standard deviation of brain intensities in the volume.
        target_size: Output spatial dimension (square).

    Returns:
        Preprocessed 2D numpy array of shape (target_size, target_size).
    """
    bbox = get_brain_bbox(slice_2d)
    if bbox is None:
        return np.zeros((target_size, target_size), dtype=np.float32)

    row_min, row_max, col_min, col_max = bbox
    cropped = slice_2d[row_min:row_max, col_min:col_max]

    # Resize to target_size x target_size
    zoom_h = target_size / cropped.shape[0]
    zoom_w = target_size / cropped.shape[1]
    resized = zoom(cropped, (zoom_h, zoom_w), order=1)

    # Z-score normalize using volume-level statistics
    if vol_std > 1e-8:
        normalized = (resized - vol_mean) / vol_std
    else:
        normalized = resized - vol_mean

    return normalized.astype(np.float32)


def process_patient(
    patient_dir: Path,
    output_dir: Path,
    target_size: int = 256,
    tumor_threshold: float = 0.01,
    min_brain_area: int = 1000,
    healthy_margin: int = 15,
    max_bright_fraction: float = 0.02,
) -> dict[str, int]:
    """Process a single patient: extract and classify axial slices.

    Iterates over axial slices (axis 2) and classifies each as:
    - pathological: tumor occupies > tumor_threshold fraction of brain
    - healthy: tumor_area == 0 AND at least healthy_margin slices away
      from the nearest tumor-containing slice
    - skip: otherwise

    Args:
        patient_dir: Directory containing the patient's NIfTI files.
        output_dir: Root output directory (data/processed).
        target_size: Output spatial dimension.
        tumor_threshold: Minimum tumor-to-brain ratio for pathological label.
        min_brain_area: Minimum non-zero pixels to keep a slice.
        healthy_margin: Minimum number of slices between a healthy slice and
            the nearest tumor-containing slice.

    Returns:
        Dict with counts: {"pathological": N, "healthy": M, "skipped": K}.
    """
    patient_id = patient_dir.name

    # Find FLAIR file
    flair_files = list(patient_dir.glob("*_flair.nii.gz")) + list(
        patient_dir.glob("*_flair.nii")
    )
    if not flair_files:
        print(f"  No FLAIR file found for {patient_id}, skipping.")
        return {"pathological": 0, "healthy": 0, "skipped": 0}
    flair_vol = load_nifti(flair_files[0])

    # Find segmentation file
    seg_files = list(patient_dir.glob("*_seg.nii.gz")) + list(
        patient_dir.glob("*_seg.nii")
    )
    seg_vol = load_nifti(seg_files[0]) if seg_files else None

    # Create output directories
    path_dir = output_dir / "pathological"
    health_dir = output_dir / "healthy"
    path_dir.mkdir(parents=True, exist_ok=True)
    health_dir.mkdir(parents=True, exist_ok=True)

    counts = {"pathological": 0, "healthy": 0, "skipped": 0}

    # Iterate over axial slices (axis 2)
    n_slices = flair_vol.shape[2]

    # Pre-compute the z-range of tumor presence with a safety margin buffer.
    # Using min/max of tumor slices is sufficient since tumors are contiguous.
    tumor_z_min, tumor_z_max = n_slices, -1
    if seg_vol is not None:
        for i in range(n_slices):
            if np.any(seg_vol[:, :, i] > 0):
                tumor_z_min = min(tumor_z_min, i)
                tumor_z_max = max(tumor_z_max, i)
    has_tumor = tumor_z_max >= 0
    safe_z_min = tumor_z_min - healthy_margin  # healthy must be below this
    safe_z_max = tumor_z_max + healthy_margin  # healthy must be above this

    # Compute intensity statistics from tumor-free slices only so that
    # bright tumor voxels don't inflate vol_std and loosen the filter.
    if has_tumor:
        safe_slices = (
            list(range(0, max(0, safe_z_min))) +
            list(range(min(n_slices, safe_z_max + 1), n_slices))
        )
    else:
        safe_slices = list(range(n_slices))

    safe_voxels = np.concatenate([
        flair_vol[:, :, i][flair_vol[:, :, i] > 0].ravel()
        for i in safe_slices
    ]) if safe_slices else np.array([])

    if safe_voxels.size > 0:
        vol_mean = float(safe_voxels.mean())
        vol_std = float(safe_voxels.std()) if safe_voxels.std() > 1e-8 else 1.0
    else:
        # Fallback to whole-volume stats if no safe slices exist
        brain_voxels = flair_vol[flair_vol > 0]
        vol_mean = float(brain_voxels.mean()) if brain_voxels.size > 0 else 0.0
        vol_std = float(brain_voxels.std()) if brain_voxels.size > 0 else 1.0

    for i in range(n_slices):
        flair_slice = flair_vol[:, :, i]
        brain_area = np.sum(flair_slice > 0)

        if brain_area < min_brain_area:
            counts["skipped"] += 1
            continue

        # Classify slice
        if seg_vol is not None:
            seg_slice = seg_vol[:, :, i]
            tumor_area = np.sum(seg_slice > 0)
            tumor_ratio = tumor_area / brain_area if brain_area > 0 else 0.0

            if tumor_ratio > tumor_threshold:
                label = "pathological"
            elif tumor_area == 0:
                # Require at least healthy_margin slices from the tumor z-range
                if has_tumor and safe_z_min <= i <= safe_z_max:
                    counts["skipped"] += 1
                    continue
                # Reject slices with too many bright outlier pixels
                # (CSF artifacts, vascular signal) that mimic tumor on FLAIR
                brain_pixels = flair_slice[flair_slice > 0]
                bright_fraction = np.mean(brain_pixels > vol_mean + 3.0 * vol_std)
                if bright_fraction > max_bright_fraction:
                    counts["skipped"] += 1
                    continue
                label = "healthy"
            else:
                counts["skipped"] += 1
                continue
        else:
            label = "healthy"

        # Preprocess and save
        processed = preprocess_slice(flair_slice, vol_mean, vol_std, target_size)
        filename = f"{patient_id}_slice{i:03d}.npy"
        save_dir = path_dir if label == "pathological" else health_dir
        np.save(save_dir / filename, processed)
        counts[label] += 1

    return counts


def create_split(
    patient_ids: list[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Create patient-level train/val/test split.

    Args:
        patient_ids: List of patient ID strings.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys "train", "val", "test" mapping to patient ID lists.
    """
    rng = random.Random(seed)
    ids = sorted(patient_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    split = {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }
    return split


def main() -> None:
    """Main entry point for BraTS preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess BraTS NIfTI volumes into 2D .npy slices."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Path to raw BraTS data directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Path to output directory for processed slices.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=256,
        help="Target spatial dimension for output slices.",
    )
    parser.add_argument(
        "--tumor-threshold",
        type=float,
        default=0.01,
        help="Minimum tumor-to-brain ratio for pathological label.",
    )
    parser.add_argument(
        "--min-brain-area",
        type=int,
        default=1000,
        help="Minimum non-zero pixels to keep a slice.",
    )
    parser.add_argument(
        "--healthy-margin",
        type=int,
        default=15,
        help="Minimum slices between a healthy slice and the nearest tumor slice.",
    )
    parser.add_argument(
        "--max-bright-fraction",
        type=float,
        default=0.02,
        help="Max fraction of brain pixels above 3 sigma before a healthy slice is rejected.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    print(f"Finding patient directories in {raw_dir}...")
    patient_dirs = find_patient_dirs(raw_dir)
    print(f"Found {len(patient_dirs)} patients.")

    total_counts = {"pathological": 0, "healthy": 0, "skipped": 0}

    for idx, patient_dir in enumerate(patient_dirs):
        patient_id = patient_dir.name
        print(f"[{idx + 1}/{len(patient_dirs)}] Processing {patient_id}...")
        counts = process_patient(
            patient_dir,
            output_dir,
            target_size=args.target_size,
            tumor_threshold=args.tumor_threshold,
            min_brain_area=args.min_brain_area,
            healthy_margin=args.healthy_margin,
            max_bright_fraction=args.max_bright_fraction,
        )
        for key in total_counts:
            total_counts[key] += counts[key]

    print(f"\nTotal slices: {total_counts}")

    # Create patient-level split
    patient_ids = [d.name for d in patient_dirs]
    split = create_split(patient_ids)
    split_path = output_dir / "split.json"
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"Split saved to {split_path}")
    for key, ids in split.items():
        print(f"  {key}: {len(ids)} patients")


if __name__ == "__main__":
    main()
