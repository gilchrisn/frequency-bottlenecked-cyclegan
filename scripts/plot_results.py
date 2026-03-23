"""Plot training results and metrics (stub)."""

import argparse


def main() -> None:
    """Placeholder for result plotting."""
    parser = argparse.ArgumentParser(description="Plot training results.")
    parser.add_argument(
        "--run", type=str, required=True, help="Name of the experiment run."
    )
    parser.parse_args()
    print("Not yet implemented")


if __name__ == "__main__":
    main()
