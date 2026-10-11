"""Kept for compatibility: the corpus build, prediction and analytics now run in one idempotent command.

Usage: python -m stratlens.build   (this script calls it)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stratlens import build  # noqa: E402


def main() -> None:
    build.main([])


if __name__ == "__main__":
    main()
