#!/usr/bin/env python3
"""Import LPMM OpenIE JSON into standalone A_Memorix."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure standalone package imports work when running as:
# `python scripts/import_lpmm_json.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from amemorix.bootstrap import build_context
from amemorix.common.logging import get_logger
from amemorix.services import ImportService
from amemorix.settings import AppSettings

logger = get_logger("A_Memorix.ImportLPMM")


def convert_lpmm_payload(lpmm_data: Dict[str, Any], source_name: str) -> Dict[str, Any]:
    paragraphs: List[Dict[str, Any]] = []
    relations: List[Dict[str, Any]] = []

    for doc in lpmm_data.get("docs", []) if isinstance(lpmm_data, dict) else []:
        content = str(doc.get("passage", "") or "").strip()
        if not content:
            continue
        paragraphs.append({"content": content, "source": source_name})

        for triple in doc.get("extracted_triples", []):
            if isinstance(triple, list) and len(triple) == 3:
                relations.append(
                    {
                        "subject": str(triple[0] or "").strip(),
                        "predicate": str(triple[1] or "").strip(),
                        "object": str(triple[2] or "").strip(),
                    }
                )
            elif isinstance(triple, dict):
                relations.append(
                    {
                        "subject": str(triple.get("subject", "")).strip(),
                        "predicate": str(triple.get("predicate", "")).strip(),
                        "object": str(triple.get("object", "")).strip(),
                    }
                )

    return {"paragraphs": paragraphs, "relations": relations}


async def run(args: argparse.Namespace) -> int:
    settings = AppSettings.load(args.config)
    ctx = build_context(settings)
    importer = ImportService(ctx)
    try:
        target = Path(args.path).expanduser()
        if not target.is_absolute():
            target = (Path.cwd() / target).resolve()
        if not target.exists():
            print(f"Path not found: {target}")
            return 2

        files: List[Path]
        if target.is_dir():
            files = sorted(target.glob("*-openie.json")) or sorted(target.glob("*.json"))
        else:
            files = [target]
        if not files:
            print("No JSON files found.")
            return 0

        ok_count = 0
        for file_path in files:
            try:
                payload = json.loads(file_path.read_text(encoding="utf-8"))
                converted = convert_lpmm_payload(payload, source_name=f"{args.source_prefix}:{file_path.name}")
                await importer.import_json(converted)
                ok_count += 1
                logger.info("Imported %s", file_path.name)
            except Exception as exc:
                logger.error("Import failed for %s: %s", file_path, exc)

        print(f"Imported {ok_count}/{len(files)} files.")
        return 0
    finally:
        await ctx.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import LPMM OpenIE JSON into A_Memorix standalone")
    parser.add_argument("path", type=str, help="JSON file or directory")
    parser.add_argument("--config", type=str, default=None, help="Path to config.toml")
    parser.add_argument("--source-prefix", type=str, default="lpmm", help="source prefix in metadata")
    return parser.parse_args()


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
