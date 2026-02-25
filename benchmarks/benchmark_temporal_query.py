from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.storage.metadata_store import MetadataStore


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def run_benchmark(
    paragraphs: int,
    queries: int,
    query_window_minutes: int,
    query_limit: int,
    person_filter_ratio: float,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    base_dt = datetime(2026, 1, 1, 0, 0)

    with tempfile.TemporaryDirectory(prefix="amemorix_bench_") as tmp:
        data_dir = Path(tmp) / "metadata"
        store = MetadataStore(data_dir=data_dir)
        store.connect()

        try:
            ingest_started = time.perf_counter()
            person_pool = [f"user_{idx:03d}" for idx in range(80)]

            for idx in range(paragraphs):
                offset_minutes = rng.randint(0, 60 * 24 * 45)
                event_dt = base_dt + timedelta(minutes=offset_minutes)
                source = f"bench:source:{idx % 12}"
                para_hash = store.add_paragraph(
                    content=f"benchmark paragraph {idx}",
                    source=source,
                    knowledge_type="factual",
                    time_meta={"event_time": event_dt.strftime("%Y/%m/%d %H:%M")},
                )
                if idx % 4 == 0:
                    person_name = person_pool[idx % len(person_pool)]
                    store.add_entity(name=person_name, source_paragraph=para_hash)

            ingest_elapsed_ms = (time.perf_counter() - ingest_started) * 1000.0

            # Warmup to stabilize sqlite caches.
            warmup = min(30, max(5, queries // 10))
            max_offset = 60 * 24 * 45 - query_window_minutes - 1
            for _ in range(warmup):
                start_offset = rng.randint(0, max(1, max_offset))
                start_ts = (base_dt + timedelta(minutes=start_offset)).timestamp()
                end_ts = (base_dt + timedelta(minutes=start_offset + query_window_minutes)).timestamp()
                store.query_paragraphs_temporal(
                    start_ts=start_ts,
                    end_ts=end_ts,
                    source=None,
                    person=None,
                    limit=query_limit,
                    allow_created_fallback=False,
                )

            query_latencies_ms: list[float] = []
            total_hits = 0
            query_started = time.perf_counter()
            for _ in range(queries):
                start_offset = rng.randint(0, max(1, max_offset))
                start_ts = (base_dt + timedelta(minutes=start_offset)).timestamp()
                end_ts = (base_dt + timedelta(minutes=start_offset + query_window_minutes)).timestamp()

                person = None
                if rng.random() < person_filter_ratio:
                    person = person_pool[rng.randint(0, len(person_pool) - 1)]

                t0 = time.perf_counter()
                rows = store.query_paragraphs_temporal(
                    start_ts=start_ts,
                    end_ts=end_ts,
                    source=None,
                    person=person,
                    limit=query_limit,
                    allow_created_fallback=False,
                )
                elapsed = (time.perf_counter() - t0) * 1000.0
                query_latencies_ms.append(elapsed)
                total_hits += len(rows)

            query_elapsed_s = max(1e-9, time.perf_counter() - query_started)
            qps = queries / query_elapsed_s

            return {
                "seed": seed,
                "paragraphs": paragraphs,
                "queries": queries,
                "query_window_minutes": query_window_minutes,
                "query_limit": query_limit,
                "person_filter_ratio": person_filter_ratio,
                "ingest": {
                    "total_ms": round(ingest_elapsed_ms, 3),
                    "paragraphs_per_sec": round(paragraphs / max(1e-9, ingest_elapsed_ms / 1000.0), 3),
                },
                "query": {
                    "total_sec": round(query_elapsed_s, 6),
                    "qps": round(qps, 3),
                    "mean_ms": round(mean(query_latencies_ms), 3),
                    "p50_ms": round(percentile(query_latencies_ms, 50), 3),
                    "p95_ms": round(percentile(query_latencies_ms, 95), 3),
                    "p99_ms": round(percentile(query_latencies_ms, 99), 3),
                    "avg_hits": round(total_hits / max(1, queries), 3),
                },
            }
        finally:
            store.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark metadata temporal query path for A_Memorix."
    )
    parser.add_argument("--paragraphs", type=int, default=5000, help="number of synthetic paragraphs")
    parser.add_argument("--queries", type=int, default=300, help="number of benchmark queries")
    parser.add_argument("--query-window-minutes", type=int, default=120, help="time window in minutes")
    parser.add_argument("--query-limit", type=int, default=50, help="max rows per temporal query")
    parser.add_argument(
        "--person-filter-ratio",
        type=float,
        default=0.25,
        help="fraction of queries that include person filter (0.0-1.0)",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_benchmark(
        paragraphs=max(100, args.paragraphs),
        queries=max(10, args.queries),
        query_window_minutes=max(1, args.query_window_minutes),
        query_limit=max(1, args.query_limit),
        person_filter_ratio=max(0.0, min(1.0, args.person_filter_ratio)),
        seed=args.seed,
    )

    print("=== A_Memorix Temporal Query Benchmark ===")
    print(f"Run at: {datetime.now().isoformat(timespec='seconds')}")
    print(
        "Ingest: "
        f"{result['ingest']['paragraphs_per_sec']} paragraphs/s, "
        f"total={result['ingest']['total_ms']} ms"
    )
    print(
        "Query: "
        f"qps={result['query']['qps']}, "
        f"mean={result['query']['mean_ms']} ms, "
        f"p50={result['query']['p50_ms']} ms, "
        f"p95={result['query']['p95_ms']} ms, "
        f"p99={result['query']['p99_ms']} ms, "
        f"avg_hits={result['query']['avg_hits']}"
    )
    print()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
