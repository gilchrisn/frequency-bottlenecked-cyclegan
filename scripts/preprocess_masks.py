"""Extract and save tumor segmentation masks for pathological slices.

This is a companion to `scripts/preprocess_brats.py`. It re-reads the raw
BraTS NIfTI segmentation volumes and extracts the 2D axial slices that were
classified as pathological, applying the same bounding-box crop and resize
so mask .npy files are spatially aligned with their corresponding image
.npy files in `data/processed/pathological/`.

The masks are stored in `data/processed/masks/` with identical filenames
to their image counterparts. A binary mask value of 1 indicates any tumor
label (enhancing tumor, edema, or necrotic core); 0 indicates background.

Usage:
    python scripts/preprocess_masks.py \
        --raw-dir data/raw \
        --processed-dir data/processed \
        --target-size 256 \
        --tumor-threshold 0.05 \
        --min-brain-area 1000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def load_nifti(path: str | Path) -> np.ndarray:
    img = nib.load(str(path))
    return img.get_fdata().astype(np.float32)


def get_brain_bbox(
    slice_2d: np.ndarray, margin: int = 5
) -> tuple[int, int, int, int] | None:
    """Bounding box of non-zero region, matching preprocess_brats.py exactly."""
    nonzero = np.nonzero(slice_2d)
    if len(nonzero[0]) == 0:
        return None
    row_min = max(0, nonzero[0].min() - margin)
    row_max = min(slice_2d.shape[0], nonzero[0].max() + margin + 1)
    col_min = max(0, nonzero[1].min() - margin)
    col_max = min(slice_2d.shape[1], nonzero[1].max() + margin + 1)
    return (row_min, row_max, col_min, col_max)


def resize_mask_slice(
    mask_slice: np.ndarray,
    flair_slice: np.ndarray,
    target_size: int = 256,
) -> np.ndarray:
    """Crop mask using FLAIR brain bounding box, resize, binarize.

    We use the FLAIR slice to compute the bounding box (not the mask itself)
    so the spatial alignment with the preprocessed FLAIR image is exact.

    Args:
        mask_slice: Raw 2D segmentation slice (any tumor label > 0).
        flair_slice: Corresponding raw 2D FLAIR slice (used for bbox).
        target_size: Output spatial dimension.

    Returns:
        Binary mask of shape (target_size, target_size), dtype float32.
    """
    bbox = get_brain_bbox(flair_slice)
    if bbox is None:
        return np.zeros((target_size, target_size), dtype=np.float32)

    row_min, row_max, col_min, col_max = bbox
    cropped = mask_slice[row_min:row_max, col_min:col_max]

    zoom_h = target_size / cropped.shape[0]
    zoom_w = target_size / cropped.shape[1]
    # order=0 (nearest-neighbor) preserves binary labels
    resized = zoom(cropped, (zoom_h, zoom_w), order=0)

    return (resized > 0).astype(np.float32)


def process_patient_masks(
    patient_dir: Path,
    processed_dir: Path,
    target_size: int = 256,
    tumor_threshold: float = 0.05,
    min_brain_area: int = 1000,
) -> int:
    """Extract mask .npy files for the pathological slices of one patient.

    Mirrors the slice-selection logic in `preprocess_brats.py` so only
    slices that were classified as pathological get a mask file.

    Args:
        patient_dir: Raw patient directory containing NIfTI files.
        processed_dir: Root of the processed data (data/processed).
        target_size: Spatial dimension matching the preprocessed images.
        tumor_threshold: Minimum tumor-to-brain ratio for pathological label.
        min_brain_area: Minimum brain pixels to process a slice.

    Returns:
        Number of mask files saved.
    """
    patient_id = patient_dir.name

    flair_files = list(patient_dir.glob("*_flair.nii.gz")) + list(
        patient_dir.glob("*_flair.nii")
    )
    seg_files = list(patient_dir.glob("*_seg.nii.gz")) + list(
        patient_dir.glob("*_seg.nii")
    )

    if not flair_files or not seg_files:
        return 0

    flair_vol = load_nifti(flair_files[0])
    seg_vol = load_nifti(seg_files[0])

    mask_dir = processed_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    n_slices = flair_vol.shape[2]

    for i in range(n_slices):
        flair_slice = flair_vol[:, :, i]
        seg_slice = seg_vol[:, :, i]
        brain_area = np.sum(flair_slice > 0)

        if brain_area < min_brain_area:
            continue

        tumor_area = np.sum(seg_slice > 0)
        tumor_ratio = tumor_area / brain_area if brain_area > 0 else 0.0

        # Only save mask if slice would have been labeled pathological
        if tumor_ratio <= tumor_threshold:
            continue

        mask_processed = resize_mask_slice(seg_slice, flair_slice, target_size)
        filename = f"{patient_id}_slice{i:03d}.npy"
        save_path = mask_dir / filename
        np.save(save_path, mask_processed)
        saved += 1

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract mask .npy files aligned with preprocessed pathological slices."
    )
    parser.add_argument(
        "--raw-dir", type=str, default="data/raw",
        help="Root directory of raw BraTS NIfTI files.",
    )
    parser.add_argument(
        "--processed-dir", type=str, default="data/processed",
        help="Root directory of preprocessed slices (must contain split.json).",
    )
    parser.add_argument(
        "--target-size", type=int, default=256,
        help="Spatial dimension matching preprocessed images.",
    )
    parser.add_argument(
        "--tumor-threshold", type=float, default=0.05,
        help="Minimum tumor-to-brain ratio for pathological label.",
    )
    parser.add_argument(
        "--min-brain-area", type=int, default=1000,
        help="Minimum non-zero pixels to process a slice.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)

    # Find patient directories (same logic as preprocess_brats.py)
    flair_files = list(raw_dir.rglob("*_flair.nii.gz")) + list(
        raw_dir.rglob("*_flair.nii")
    )
    patient_dirs = sorted(set(f.parent for f in flair_files))
    print(f"Found {len(patient_dirs)} patients in {raw_dir}")

    total = 0
    for idx, patient_dir in enumerate(patient_dirs):
        print(f"[{idx + 1}/{len(patient_dirs)}] {patient_dir.name}...", end=" ")
        saved = process_patient_masks(
            patient_dir,
            processed_dir,
            target_size=args.target_size,
            tumor_threshold=args.tumor_threshold,
            min_brain_area=args.min_brain_area,
        )
        print(f"saved {saved} masks")
        total += saved

    print(f"\nTotal mask files saved: {total}")
    print(f"Masks directory: {processed_dir / 'masks'}")


if __name__ == "__main__":
    main()
