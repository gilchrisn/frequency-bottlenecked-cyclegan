"""Download BraTS 2020 dataset from Kaggle.

Requires the Kaggle CLI to be installed and configured with API credentials.
See https://github.com/Kaggle/kaggle-api for setup instructions.
"""

import subprocess
import sys
from pathlib import Path

DATASET_SLUG = "awsaf49/brats20-dataset-training-validation"
DEST = "data/raw"


def main() -> None:
    """Download BraTS 2020 training and validation data."""
    dest_path = Path(DEST)
    dest_path.mkdir(parents=True, exist_ok=True)

    print(f"-> Downloading {DATASET_SLUG} to {dest_path}/")
    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", DATASET_SLUG,
                "-p", str(dest_path),
                "--unzip",
            ],
            check=True,
        )
        print(f"-> Download complete. Data saved to {dest_path}/")
    except FileNotFoundError:
        print("-> Error: kaggle CLI not found. Install with: pip install kaggle")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"-> Download failed with exit code {e.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
