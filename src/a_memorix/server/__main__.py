"""Backward-compatible module entry for the unified CLI."""

from __future__ import annotations

import sys

from a_memorix.cli import main as cli_main


def main() -> None:
    raise SystemExit(cli_main(["serve", *sys.argv[1:]]))


if __name__ == "__main__":
    main()
