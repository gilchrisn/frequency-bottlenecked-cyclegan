"""Compute evaluation scores between real and generated images (stub)."""

import argparse


def main() -> None:
    """Placeholder for scoring."""
    parser = argparse.ArgumentParser(
        description="Score generated images against real images."
    )
    parser.add_argument(
        "--real", type=str, required=True, help="Path to real images."
    )
    parser.add_argument(
        "--generated", type=str, required=True, help="Path to generated images."
    )
    parser.parse_args()
    print("Not yet implemented")


if __name__ == "__main__":
    main()
