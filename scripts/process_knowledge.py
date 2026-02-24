#!/usr/bin/env python3
"""Standalone batch import script."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict

# Ensure standalone package imports work when running as:
# `python scripts/process_knowledge.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from amemorix.bootstrap import build_context
from amemorix.common.logging import get_logger
from amemorix.services import ImportService
from amemorix.settings import AppSettings

logger = get_logger("A_Memorix.ProcessKnowledge")


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_manifest(path: Path, data: Dict[str, Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


async def run(args: argparse.Namespace) -> int:
    settings = AppSettings.load(args.config)
    ctx = build_context(settings)
    service = ImportService(ctx)

    source_dir = Path(args.source_dir).expanduser()
    if not source_dir.is_absolute():
        source_dir = (Path.cwd() / source_dir).resolve()
    if not source_dir.exists():
        logger.error("Source dir not found: %s", source_dir)
        await ctx.close()
        return 2

    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = (Path.cwd() / manifest_path).resolve()
    manifest = _load_manifest(manifest_path)

    files = sorted(
        [*source_dir.rglob("*.txt"), *source_dir.rglob("*.md"), *source_dir.rglob("*.json")]
    )
    if not files:
        logger.info("No import files found in %s", source_dir)
        await ctx.close()
        return 0

    imported = 0
    skipped = 0
    for file_path in files:
        rel = str(file_path.relative_to(source_dir))
        digest = _file_hash(file_path)
        record = manifest.get(rel, {})
        if not args.force and record.get("hash") == digest and record.get("imported") == "true":
            skipped += 1
            continue

        try:
            if file_path.suffix.lower() == ".json":
                payload = file_path.read_text(encoding="utf-8")
                await service.import_json(payload)
            else:
                text = file_path.read_text(encoding="utf-8")
                await service.import_text(text=text, source=f"file:{rel}")
            manifest[rel] = {"hash": digest, "imported": "true"}
            imported += 1
            logger.info("Imported %s", rel)
        except Exception as exc:
            logger.error("Import failed for %s: %s", rel, exc)

    _save_manifest(manifest_path, manifest)
    logger.info("Done. imported=%s skipped=%s total=%s", imported, skipped, len(files))
    await ctx.close()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone batch import for A_Memorix")
    parser.add_argument("--config", type=str, default=None, help="Path to config.toml")
    parser.add_argument("--source-dir", type=str, default="./data/raw", help="Input file directory")
    parser.add_argument("--manifest", type=str, default="./data/import_manifest.json", help="Import manifest path")
    parser.add_argument("--force", action="store_true", help="Force re-import even if hash unchanged")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
