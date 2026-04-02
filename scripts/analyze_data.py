"""BraTS data exploration.

Finds NIfTI files in the raw data directory, groups by patient, lists
available sequences, and prints summary statistics for a sample volume.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib


def main() -> None:
    """Explore the raw BraTS dataset structure and statistics."""
    raw_dir = Path("data/raw")

    # Find all NIfTI files
    nifti_files = sorted(
        list(raw_dir.rglob("*.nii.gz")) + list(raw_dir.rglob("*.nii"))
    )
    print(f"Found {len(nifti_files)} NIfTI files in {raw_dir}/\n")

    # Group by patient directory
    patients: dict[str, list[Path]] = {}
    for f in nifti_files:
        patient_id = f.parent.name
        patients.setdefault(patient_id, []).append(f)

    print(f"Found {len(patients)} patients.\n")

    # List sequences per patient (first 5)
    for i, (patient_id, files) in enumerate(sorted(patients.items())):
        sequences = [f.name for f in files]
        print(f"  {patient_id}: {sequences}")
        if i >= 4:
            print(f"  ... and {len(patients) - 5} more patients")
            break

    # Load a sample volume and print statistics
    if nifti_files:
        sample_path = nifti_files[0]
        print(f"\nSample volume: {sample_path}")

        img = nib.load(str(sample_path))
        data = img.get_fdata()

        print(f"  Shape:      {data.shape}")
        print(f"  Dtype:      {data.dtype}")
        print(f"  Voxel size: {img.header.get_zooms()}")
        print(f"  Value range: [{data.min():.2f}, {data.max():.2f}]")
        print(f"  Mean:       {data.mean():.2f}")
        print(f"  Std:        {data.std():.2f}")

        # Brain-only stats
        brain_mask = data > 0
        brain_voxels = data[brain_mask]
        if brain_voxels.size > 0:
            print(f"\n  Brain voxels: {brain_voxels.size} "
                  f"({100 * brain_voxels.size / data.size:.1f}% of volume)")
            print(f"  Brain mean:  {brain_voxels.mean():.2f}")
            print(f"  Brain std:   {brain_voxels.std():.2f}")
    else:
        print("\nNo NIfTI files found. Run download_data.py first.")


if __name__ == "__main__":
    main()
