#!/usr/bin/env python3
"""Standalone delete tool by source."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure standalone package imports work when running as:
# `python scripts/delete_knowledge.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from amemorix.bootstrap import build_context
from amemorix.common.logging import get_logger
from amemorix.services import DeleteService
from amemorix.settings import AppSettings

logger = get_logger("A_Memorix.DeleteKnowledge")


async def run(args: argparse.Namespace) -> int:
    settings = AppSettings.load(args.config)
    ctx = build_context(settings)
    deleter = DeleteService(ctx)
    try:
        if args.list:
            sources = ctx.metadata_store.get_all_sources()
            if not sources:
                print("No sources found.")
                return 0
            print("Sources:")
            for item in sources:
                print(f"- {item.get('source')} ({item.get('count')})")
            return 0

        if not args.source:
            print("Use --list or --source <name>")
            return 2

        paragraphs = ctx.metadata_store.get_paragraphs_by_source(args.source)
        if not paragraphs:
            print(f"Source not found: {args.source}")
            return 1
        if not args.yes:
            print(f"Will delete {len(paragraphs)} paragraphs from source '{args.source}'.")
            print("Re-run with -y/--yes to confirm.")
            return 0

        success = 0
        for para in paragraphs:
            try:
                await deleter.paragraph(str(para.get("hash")))
                success += 1
            except Exception as exc:
                logger.error("Delete paragraph failed: %s", exc)
        print(f"Deleted {success}/{len(paragraphs)} paragraphs from source '{args.source}'.")
        return 0
    finally:
        await ctx.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone source delete tool for A_Memorix")
    parser.add_argument("--config", type=str, default=None, help="Path to config.toml")
    parser.add_argument("--list", action="store_true", help="List all sources")
    parser.add_argument("--source", type=str, default="", help="Delete all paragraphs from source")
    parser.add_argument("-y", "--yes", action="store_true", help="Confirm destructive delete")
    return parser.parse_args()


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
