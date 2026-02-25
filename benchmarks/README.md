# Benchmarks

## Temporal Query Benchmark

This benchmark measures the SQLite-backed temporal query path in `MetadataStore`.

Run:

```bash
python benchmarks/benchmark_temporal_query.py --paragraphs 5000 --queries 300
```

Useful options:

- `--query-window-minutes`: temporal window size for each query.
- `--query-limit`: max returned rows per query.
- `--person-filter-ratio`: fraction of queries that include person filter.
- `--seed`: random seed for reproducibility.

The script reports:

- ingest throughput (`paragraphs_per_sec`);
- query throughput (`qps`);
- latency distribution (`mean`, `p50`, `p95`, `p99`);
- average hit count per query.

## `/v1/query/time` End-to-End Benchmark

This benchmark starts a local `amemorix` service with synthetic temporal data and
measures HTTP end-to-end latency/throughput for `POST /v1/query/time`.

Run:

```bash
python benchmarks/benchmark_v1_time_e2e.py --paragraphs 3000 --queries 200 --person-filter-ratio 0.25
```

It reports:

- ingest throughput used to prepare synthetic dataset;
- HTTP query throughput (`qps`);
- latency distribution (`mean`, `p50`, `p95`, `p99`);
- average returned hit count (`avg_hits`).
