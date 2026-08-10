"""Comparable baseline checks for public evaluation summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import json


DEFAULT_QUALITY_METRICS = (
    "mrr",
    "ndcg@10",
    "recall_any@10",
    "recall_fraction@10",
)
COMPATIBILITY_FIELDS = (
    "evaluation_schema_version",
    "benchmark",
    "selected_case_ids",
    "top_k",
    "dataset.sha256",
    "embedding.fingerprint.provider",
    "embedding.fingerprint.model",
    "embedding.fingerprint.dimension",
    "embedding.fingerprint.endpoint_sha256",
    "embedding_prewarm.batch_size",
    "embedding_prewarm.max_concurrent",
)
PERFORMANCE_ENVIRONMENT_FIELDS = (
    "runtime.python_version",
    "runtime.system",
    "runtime.machine",
    "runtime.packages.numpy",
    "runtime.packages.scipy",
    "runtime.packages.faiss-cpu",
)
SCHEMA_V1_LEGACY_DEFAULTS = {
    "embedding_prewarm.batch_size": 16,
    "embedding_prewarm.max_concurrent": 3,
}


def load_summary(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evaluation summary must be a JSON object: {source}")
    return payload


def compare_summaries(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    quality_metrics: Sequence[str] = DEFAULT_QUALITY_METRICS,
    max_metric_drop: float = 0.02,
    max_ingest_p95_ratio: float = 1.25,
    max_search_p95_ratio: float = 1.50,
    max_total_p95_ratio: float = 1.25,
    quality_only: bool = False,
) -> dict[str, Any]:
    metric_drop = max(0.0, float(max_metric_drop))
    incompatibilities: list[dict[str, object]] = []
    fields = list(COMPATIBILITY_FIELDS)
    for suite_field in (
        "granularity",
        "chunk_chars",
        "chunk_overlap_chars",
    ):
        if (
            _read_path(baseline, suite_field) is not None
            or _read_path(candidate, suite_field) is not None
        ):
            fields.append(suite_field)
    if not quality_only:
        fields.extend(PERFORMANCE_ENVIRONMENT_FIELDS)
    for field in fields:
        baseline_value = _compatibility_value(baseline, field)
        candidate_value = _compatibility_value(candidate, field)
        if baseline_value != candidate_value:
            incompatibilities.append(
                {
                    "field": field,
                    "baseline": baseline_value,
                    "candidate": candidate_value,
                }
            )

    failures: list[str] = []
    baseline_completed = _number(baseline, "results.completed_count")
    baseline_failed = _number(baseline, "results.failed_count")
    baseline_selected = _number(baseline, "selected_case_count")
    if baseline_failed != 0 or baseline_completed != baseline_selected:
        failures.append("baseline_has_incomplete_cases")
    candidate_completed = _number(candidate, "results.completed_count")
    candidate_failed = _number(candidate, "results.failed_count")
    candidate_selected = _number(candidate, "selected_case_count")
    if candidate_failed != 0 or candidate_completed != candidate_selected:
        failures.append("candidate_has_incomplete_cases")

    quality: list[dict[str, object]] = []
    for metric in dict.fromkeys(str(item) for item in quality_metrics if str(item)):
        baseline_value = _number(baseline, f"results.metrics.{metric}")
        candidate_value = _number(candidate, f"results.metrics.{metric}")
        drop = baseline_value - candidate_value
        passed = drop <= metric_drop
        quality.append(
            {
                "metric": metric,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "drop": round(drop, 6),
                "max_drop": metric_drop,
                "passed": passed,
            }
        )
        if not passed:
            failures.append(f"quality_regression:{metric}")

    performance: list[dict[str, object]] = []
    if not quality_only:
        if (
            _number(baseline, "resumed_case_count") != 0
            or _number(candidate, "resumed_case_count") != 0
        ):
            failures.append("performance_comparison_requires_single_run")
        if (
            _number(baseline, "embedding.cache_misses") != 0
            or _number(candidate, "embedding.cache_misses") != 0
        ):
            failures.append("performance_comparison_requires_warm_cache")
        latency_limits = (
            ("ingest_p95", max_ingest_p95_ratio),
            ("search_p95", max_search_p95_ratio),
            ("total_p95", max_total_p95_ratio),
        )
        for timing, raw_limit in latency_limits:
            limit = max(1.0, float(raw_limit))
            baseline_value = _number(baseline, f"results.timing_ms.{timing}")
            candidate_value = _number(candidate, f"results.timing_ms.{timing}")
            ratio = candidate_value / baseline_value if baseline_value > 0 else None
            passed = ratio is not None and ratio <= limit
            performance.append(
                {
                    "timing": timing,
                    "baseline_ms": baseline_value,
                    "candidate_ms": candidate_value,
                    "ratio": round(ratio, 6) if ratio is not None else None,
                    "max_ratio": limit,
                    "passed": passed,
                }
            )
            if not passed:
                failures.append(f"latency_regression:{timing}")

    if incompatibilities:
        failures.insert(0, "incompatible_summaries")
    return {
        "status": "passed" if not failures else "failed",
        "quality_only": bool(quality_only),
        "incompatibilities": incompatibilities,
        "quality": quality,
        "performance": performance,
        "failures": list(dict.fromkeys(failures)),
    }


def _read_path(payload: Mapping[str, Any], path: str) -> object:
    value: object = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _compatibility_value(payload: Mapping[str, Any], path: str) -> object:
    value = _read_path(payload, path)
    if value is not None:
        return value
    if _read_path(payload, "evaluation_schema_version") == 1:
        return SCHEMA_V1_LEGACY_DEFAULTS.get(path)
    return None


def _number(payload: Mapping[str, Any], path: str) -> float:
    value = _read_path(payload, path)
    if not isinstance(value, (int, float)):
        raise ValueError(f"evaluation summary field must be numeric: {path}")
    return float(value)
