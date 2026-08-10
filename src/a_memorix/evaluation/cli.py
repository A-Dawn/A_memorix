"""Command-line entry point for public A_memorix evaluations."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Callable, Sequence

import asyncio
import json

from .common import (
    CachedEmbeddingProvider,
    EmbeddingConfig,
    OpenAICompatibleEmbeddingProvider,
    evaluation_run_lock,
)
from .comparison import compare_summaries, load_summary
from . import longmemeval, swebench


DEFAULT_ROOT = Path("data/public-benchmarks")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="a-memorix-eval",
        description="Run reproducible public retrieval evaluations",
    )
    parser.add_argument("--config", type=Path, default=Path("config.txt"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    suites = parser.add_subparsers(dest="suite", required=True)

    long_parser = suites.add_parser(
        "longmemeval", help="Cross-session and long-context memory retrieval"
    )
    _add_suite_commands(long_parser, suite="longmemeval")

    swe_parser = suites.add_parser(
        "swebench", help="Repository issue-to-source-file retrieval"
    )
    _add_suite_commands(swe_parser, suite="swebench")

    compare = suites.add_parser(
        "compare", help="Compare compatible baseline and candidate summaries"
    )
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--quality-metric", action="append", default=[])
    compare.add_argument("--max-metric-drop", type=float, default=0.02)
    compare.add_argument("--max-ingest-p95-ratio", type=float, default=1.25)
    compare.add_argument("--max-search-p95-ratio", type=float, default=1.50)
    compare.add_argument("--max-total-p95-ratio", type=float, default=1.25)
    compare.add_argument("--quality-only", action="store_true")
    return parser


def _add_suite_commands(parser: ArgumentParser, *, suite: str) -> None:
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="Download and validate the pinned dataset")
    commands.add_parser("validate", help="Validate the local pinned dataset")
    run = commands.add_parser("run", help="Run retrieval evaluation")
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--top-k", type=int, default=50 if suite == "longmemeval" else 20)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--work-dir", type=Path)
    run.add_argument("--embedding-cache", type=Path)
    run.add_argument("--embedding-batch-size", type=int, default=16)
    run.add_argument("--embedding-concurrency", type=int, default=3)
    run.add_argument("--keep-case-data", action="store_true")
    run.add_argument("--resume", action="store_true")
    if suite == "longmemeval":
        run.add_argument("--granularity", choices=("session", "turn"), default="turn")
        run.add_argument("--question-id", action="append", default=[])
        run.add_argument("--question-type", action="append", default=[])
        run.add_argument("--include-unscored", action="store_true")
    else:
        run.add_argument("--instance-id", action="append", default=[])
        run.add_argument("--repo-cache-dir", type=Path)
        run.add_argument("--chunk-chars", type=int, default=8_000)
        run.add_argument("--chunk-overlap-chars", type=int, default=800)


def _paths(args: Namespace) -> dict[str, Path]:
    root = Path(args.data_root).resolve()
    if args.suite == "longmemeval":
        suite_root = root / "longmemeval"
        dataset = suite_root / "longmemeval_s_cleaned.json"
    else:
        suite_root = root / "swebench-lite"
        dataset = swebench.resolve_dataset_path(suite_root)
    return {
        "root": root,
        "suite": suite_root,
        "dataset": dataset,
        "output": Path(args.output_dir).resolve()
        if getattr(args, "output_dir", None)
        else suite_root / "results",
        "work": Path(args.work_dir).resolve()
        if getattr(args, "work_dir", None)
        else suite_root / "work",
        "cache": Path(args.embedding_cache).resolve()
        if getattr(args, "embedding_cache", None)
        else root / "cache" / "embeddings.sqlite3",
    }


async def _run(args: Namespace, paths: dict[str, Path]) -> dict[str, object]:
    config = EmbeddingConfig.from_file(args.config)
    provider = CachedEmbeddingProvider(
        OpenAICompatibleEmbeddingProvider(config),
        paths["cache"],
    )
    try:
        if args.suite == "longmemeval":
            return await longmemeval.run_benchmark(
                provider,
                longmemeval.RunOptions(
                    dataset_path=paths["dataset"],
                    output_dir=paths["output"],
                    work_dir=paths["work"],
                    granularity=args.granularity,
                    top_k=args.top_k,
                    limit=max(0, args.limit),
                    question_ids=tuple(args.question_id),
                    question_types=tuple(args.question_type),
                    scored_only=not args.include_unscored,
                    keep_case_data=args.keep_case_data,
                    resume=args.resume,
                    embedding_batch_size=max(1, args.embedding_batch_size),
                    embedding_concurrency=max(1, args.embedding_concurrency),
                ),
            )
        repo_cache = (
            Path(args.repo_cache_dir).resolve()
            if args.repo_cache_dir
            else paths["suite"] / "repositories"
        )
        return await swebench.run_benchmark(
            provider,
            swebench.RunOptions(
                dataset_path=paths["dataset"],
                output_dir=paths["output"],
                work_dir=paths["work"],
                repo_cache_dir=repo_cache,
                top_k=args.top_k,
                limit=max(0, args.limit),
                instance_ids=tuple(args.instance_id),
                chunk_chars=args.chunk_chars,
                chunk_overlap_chars=args.chunk_overlap_chars,
                keep_case_data=args.keep_case_data,
                resume=args.resume,
                embedding_batch_size=max(1, args.embedding_batch_size),
                embedding_concurrency=max(1, args.embedding_concurrency),
            ),
        )
    finally:
        provider.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.suite == "compare":
        metrics = tuple(args.quality_metric) if args.quality_metric else ()
        if metrics:
            payload = compare_summaries(
                load_summary(args.baseline),
                load_summary(args.candidate),
                quality_metrics=metrics,
                max_metric_drop=args.max_metric_drop,
                max_ingest_p95_ratio=args.max_ingest_p95_ratio,
                max_search_p95_ratio=args.max_search_p95_ratio,
                max_total_p95_ratio=args.max_total_p95_ratio,
                quality_only=args.quality_only,
            )
        else:
            payload = compare_summaries(
                load_summary(args.baseline),
                load_summary(args.candidate),
                max_metric_drop=args.max_metric_drop,
                max_ingest_p95_ratio=args.max_ingest_p95_ratio,
                max_search_p95_ratio=args.max_search_p95_ratio,
                max_total_p95_ratio=args.max_total_p95_ratio,
                quality_only=args.quality_only,
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if payload["status"] == "passed" else 1
    paths = _paths(args)
    download: Callable[[str | Path], dict[str, Any]]
    validate: Callable[[str | Path], dict[str, Any]]
    if args.suite == "longmemeval":
        download = longmemeval.download_dataset
        validate = longmemeval.validate_dataset
    else:
        download = swebench.download_dataset
        validate = swebench.validate_dataset
    if args.command == "download":
        payload = download(paths["dataset"])
    elif args.command == "validate":
        payload = validate(paths["dataset"])
    elif args.command == "run":
        with evaluation_run_lock(paths["output"]):
            payload = asyncio.run(_run(args, paths))
    else:
        raise RuntimeError(f"unsupported evaluation command: {args.command}")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
