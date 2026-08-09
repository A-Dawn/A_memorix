"""CLI wrapper for the generic LPMM converter."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from a_memorix.core.utils.lpmm_converter import main


if __name__ == "__main__":
    sys.exit(main())
