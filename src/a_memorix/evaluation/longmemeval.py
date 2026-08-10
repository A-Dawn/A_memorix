"""LongMemEval-S Cleaned retrieval evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import json
import os
import re
import time

from .common import (
    CachedEmbeddingProvider,
    EVALUATION_SCHEMA_VERSION,
    aggregate_results,
    append_jsonl,
    prepare_result_log,
    runtime_fingerprint,
    write_json,
    write_jsonl,
)
from .engine_backend import (
    AMemorixEvaluationBackend,
    BackendOptions,
    RetrievalCase,
    RetrievalDocument,
)


DATASET_URLS = (
    (
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/"
        "resolve/main/longmemeval_s_cleaned.json"
    ),
    (
        "https://hf-mirror.com/datasets/xiaowu0162/longmemeval-cleaned/"
        "resolve/main/longmemeval_s_cleaned.json"
    ),
)
DATASET_BYTES = 277_383_467
DATASET_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
DATASET_CASES = 500
DATASET_SCORED_CASES = 470
DATASET_ABSTENTION_CASES = 30
DATASET_NO_TARGET_CASES = 0
DATASET_TYPE_COUNTS = {
    "knowledge-update": 78,
    "multi-session": 133,
    "single-session-assistant": 56,
    "single-session-preference": 30,
    "single-session-user": 70,
    "temporal-reasoning": 133,
}
DATASET_REPOSITORY = "https://github.com/xiaowu0162/LongMemEval"
DATASET_REPOSITORY_COMMIT = "9e0b455f4ef0e2ab8f2e582289761153549043fc"

Granularity = Literal["session", "turn"]


@dataclass(frozen=True)
class LongMemEvalCase:
    question_id: str
    question_type: str
    question: str
    question_date: str
    answer_session_ids: tuple[str, ...]
    haystack_dates: tuple[str, ...]
    haystack_session_ids: tuple[str, ...]
    haystack_sessions: tuple[tuple[Mapping[str, object], ...], ...]

    @property
    def is_abstention(self) -> bool:
        return self.question_id.endswith("_abs")

    @property
    def has_retrieval_target(self) -> bool:
        return any(
            bool(turn.get("has_answer", False))
            for session in self.haystack_sessions
            for turn in session
        )

    @property
    def retrieval_scored(self) -> bool:
        return not self.is_abstention and self.has_retrieval_target

    def as_retrieval_case(self, granularity: Granularity) -> RetrievalCase:
        documents: list[RetrievalDocument] = []
        gold_ids: list[str] = []
        answer_session_ids = set(self.answer_session_ids)
        for session_id, date_text, turns in zip(
            self.haystack_session_ids,
            self.haystack_dates,
            self.haystack_sessions,
            strict=True,
        ):
            timestamp = parse_datetime(date_text)
            if granularity == "session":
                text = "\n".join(_render_turn(turn) for turn in turns).strip()
                if text:
                    documents.append(
                        RetrievalDocument(
                            document_id=session_id,
                            text=text,
                            timestamp=timestamp,
                            metadata={"session_id": session_id},
                        )
                    )
                    if session_id in answer_session_ids:
                        gold_ids.append(session_id)
                continue
            if granularity != "turn":
                raise ValueError(f"unsupported LongMemEval granularity: {granularity}")
            for turn_index, turn in enumerate(turns, start=1):
                text = _render_turn(turn)
                if not text:
                    continue
                corpus_id = f"{session_id}_{turn_index}"
                documents.append(
                    RetrievalDocument(
                        document_id=corpus_id,
                        text=text,
                        timestamp=timestamp,
                        metadata={
                            "session_id": session_id,
                            "turn_index": turn_index,
                            "role": str(turn.get("role", "")),
                        },
                    )
                )
                if bool(turn.get("has_answer", False)):
                    gold_ids.append(corpus_id)
        return RetrievalCase(
            case_id=self.question_id,
            case_type=self.question_type,
            query=self.question,
            documents=tuple(documents),
            gold_ids=tuple(gold_ids),
        )


def _render_turn(turn: Mapping[str, object]) -> str:
    role = str(turn.get("role", "")).strip().capitalize()
    content = str(turn.get("content", "")).strip()
    return f"{role}: {content}" if content else ""


@dataclass(frozen=True)
class RunOptions:
    dataset_path: Path
    output_dir: Path
    work_dir: Path
    granularity: Granularity = "turn"
    top_k: int = 50
    limit: int = 0
    question_ids: tuple[str, ...] = ()
    question_types: tuple[str, ...] = ()
    scored_only: bool = True
    keep_case_data: bool = False
    resume: bool = False
    embedding_batch_size: int = 16
    embedding_concurrency: int = 3


def parse_datetime(value: str) -> float:
    try:
        normalized = re.sub(r"\s+\([^)]+\)", "", value.strip())
        return (
            datetime.strptime(normalized, "%Y/%m/%d %H:%M")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
    except ValueError as exc:
        raise ValueError(f"invalid LongMemEval date: {value}") from exc


def parse_case(payload: Mapping[str, Any]) -> LongMemEvalCase:
    question_id = _required_text(payload, "question_id")
    question_type = _required_text(payload, "question_type")
    question = _required_text(payload, "question")
    question_date = _required_text(payload, "question_date")
    answer_session_ids = _required_list(payload, "answer_session_ids")
    haystack_dates = _required_list(payload, "haystack_dates")
    haystack_session_ids = _required_list(payload, "haystack_session_ids")
    raw_sessions = _required_list(payload, "haystack_sessions")
    if not (len(haystack_dates) == len(haystack_session_ids) == len(raw_sessions)):
        raise ValueError(f"LongMemEval haystack arrays differ in length: {question_id}")
    normalized_sessions: list[tuple[Mapping[str, object], ...]] = []
    for raw_session in raw_sessions:
        if not isinstance(raw_session, list) or not raw_session:
            raise ValueError(
                f"LongMemEval session must be a non-empty list: {question_id}"
            )
        normalized: list[Mapping[str, object]] = []
        for turn in raw_session:
            if not isinstance(turn, dict):
                raise ValueError(f"LongMemEval turn must be an object: {question_id}")
            role = turn.get("role")
            content = turn.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                raise ValueError(
                    f"LongMemEval turn role/content is invalid: {question_id}"
                )
            item: dict[str, object] = {"role": role, "content": content}
            if "has_answer" in turn:
                if not isinstance(turn["has_answer"], bool):
                    raise ValueError(
                        f"LongMemEval has_answer must be boolean: {question_id}"
                    )
                item["has_answer"] = turn["has_answer"]
            normalized.append(item)
        normalized_sessions.append(tuple(normalized))
    parse_datetime(question_date)
    for date_text in haystack_dates:
        parse_datetime(str(date_text))
    return LongMemEvalCase(
        question_id=question_id,
        question_type=question_type,
        question=question,
        question_date=question_date,
        answer_session_ids=tuple(str(item) for item in answer_session_ids),
        haystack_dates=tuple(str(item) for item in haystack_dates),
        haystack_session_ids=tuple(str(item) for item in haystack_session_ids),
        haystack_sessions=tuple(normalized_sessions),
    )


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"LongMemEval field {key} must be non-empty text")
    return value.strip()


def _required_list(payload: Mapping[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"LongMemEval field {key} must be a list")
    return value


def iter_json_array(
    path: str | Path, chunk_chars: int = 1 << 20
) -> Iterator[dict[str, Any]]:
    """Stream a large top-level JSON array without retaining the whole file."""

    dataset_path = Path(path).resolve()
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    finished = False
    with dataset_path.open("r", encoding="utf-8") as handle:
        eof = False
        while not finished:
            if not eof and len(buffer) - position < chunk_chars // 2:
                buffer = buffer[position:] + handle.read(chunk_chars)
                position = 0
                eof = len(buffer) < chunk_chars
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer):
                    if eof:
                        raise ValueError("LongMemEval dataset is empty")
                    continue
                if buffer[position] != "[":
                    raise ValueError("LongMemEval top level must be an array")
                started = True
                position += 1
                continue
            while position < len(buffer) and (
                buffer[position].isspace() or buffer[position] == ","
            ):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                finished = True
                position += 1
                continue
            if position >= len(buffer):
                if eof:
                    raise ValueError("LongMemEval dataset is truncated")
                continue
            try:
                item, next_position = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError as exc:
                if eof:
                    raise ValueError(f"LongMemEval JSON is incomplete: {exc}") from exc
                next_chunk = handle.read(chunk_chars)
                if next_chunk:
                    buffer += next_chunk
                else:
                    eof = True
                continue
            if not isinstance(item, dict):
                raise ValueError("LongMemEval cases must be objects")
            position = next_position
            yield item
        trailing = buffer[position:] + handle.read()
        if trailing.strip():
            raise ValueError("LongMemEval dataset contains trailing content")


def iter_cases(path: str | Path) -> Iterator[LongMemEvalCase]:
    for payload in iter_json_array(path):
        yield parse_case(payload)


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(4 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(path: str | Path, *, verify_hash: bool = True) -> dict[str, Any]:
    dataset_path = Path(path).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    actual_bytes = dataset_path.stat().st_size
    if actual_bytes != DATASET_BYTES:
        raise ValueError(
            f"LongMemEval size mismatch: expected={DATASET_BYTES}, actual={actual_bytes}"
        )
    actual_hash = file_sha256(dataset_path) if verify_hash else ""
    if verify_hash and actual_hash != DATASET_SHA256:
        raise ValueError(
            f"LongMemEval SHA-256 mismatch: expected={DATASET_SHA256}, actual={actual_hash}"
        )
    type_counts: Counter[str] = Counter()
    case_count = 0
    scored_count = 0
    abstention_count = 0
    no_target_count = 0
    for case in iter_cases(dataset_path):
        case_count += 1
        type_counts[case.question_type] += 1
        scored_count += int(case.retrieval_scored)
        abstention_count += int(case.is_abstention)
        no_target_count += int(not case.is_abstention and not case.has_retrieval_target)
    actual_types = dict(sorted(type_counts.items()))
    expected = (
        case_count == DATASET_CASES
        and scored_count == DATASET_SCORED_CASES
        and abstention_count == DATASET_ABSTENTION_CASES
        and no_target_count == DATASET_NO_TARGET_CASES
        and actual_types == DATASET_TYPE_COUNTS
    )
    if not expected:
        raise ValueError(
            "LongMemEval case distribution does not match the pinned dataset"
        )
    return {
        "path": str(dataset_path),
        "bytes": actual_bytes,
        "sha256": actual_hash,
        "case_count": case_count,
        "scored_case_count": scored_count,
        "abstention_case_count": abstention_count,
        "no_target_case_count": no_target_count,
        "question_type_counts": actual_types,
        "source": DATASET_REPOSITORY,
        "source_commit": DATASET_REPOSITORY_COMMIT,
        "license": "MIT",
    }


def download_dataset(path: str | Path, *, retries: int = 6) -> dict[str, Any]:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    if target.exists() and target.stat().st_size == DATASET_BYTES:
        return validate_dataset(target)
    errors: list[str] = []
    for attempt in range(max(1, int(retries))):
        current_size = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "A_memorix-evaluation/1.0"}
        if current_size:
            headers["Range"] = f"bytes={current_size}-"
        url = DATASET_URLS[attempt % len(DATASET_URLS)]
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=90.0) as response:
                status = int(getattr(response, "status", response.getcode()))
                mode = "ab" if current_size and status == 206 else "wb"
                with partial.open(mode) as output:
                    while chunk := response.read(4 << 20):
                        output.write(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                        if output.tell() > DATASET_BYTES:
                            raise RuntimeError(
                                "LongMemEval download exceeded pinned size"
                            )
            if partial.stat().st_size != DATASET_BYTES:
                raise RuntimeError(
                    f"LongMemEval download incomplete: {partial.stat().st_size}/{DATASET_BYTES}"
                )
            partial.replace(target)
            return validate_dataset(target)
        except (HTTPError, URLError, OSError, RuntimeError, ValueError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            time.sleep(min(10.0, 1.5 * (attempt + 1)))
    raise RuntimeError(
        "LongMemEval download failed; partial file was retained: "
        + " | ".join(errors[-3:])
    )


async def run_benchmark(
    provider: CachedEmbeddingProvider,
    options: RunOptions,
) -> dict[str, Any]:
    validation = validate_dataset(options.dataset_path)
    id_filter = set(options.question_ids)
    type_filter = set(options.question_types)
    selected_ids: list[str] = []
    for case in iter_cases(options.dataset_path):
        if id_filter and case.question_id not in id_filter:
            continue
        if type_filter and case.question_type not in type_filter:
            continue
        if options.scored_only and not case.retrieval_scored:
            continue
        selected_ids.append(case.question_id)
        if options.limit > 0 and len(selected_ids) >= options.limit:
            break
    if not selected_ids:
        raise ValueError("LongMemEval selection contains no cases")
    await provider.initialize()
    runtime = runtime_fingerprint()
    dataset_report = dict(validation)
    dataset_report.pop("path", None)
    embedding_prewarm = {
        "batch_size": max(1, int(options.embedding_batch_size)),
        "max_concurrent": max(1, int(options.embedding_concurrency)),
    }
    manifest = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "benchmark": "LongMemEval-S Cleaned retrieval",
        "granularity": options.granularity,
        "top_k": options.top_k,
        "selected_case_ids": selected_ids,
        "dataset_sha256": validation["sha256"],
        "embedding_fingerprint": dict(provider.fingerprint()),
        "embedding_prewarm": embedding_prewarm,
        "runtime": runtime,
    }
    output_dir = Path(options.output_dir).resolve()
    rows = prepare_result_log(output_dir, manifest, resume=options.resume)
    resumed_case_count = len(rows)
    completed_ids = {str(row["case_id"]) for row in rows}
    selected_id_set = set(selected_ids)
    backend = AMemorixEvaluationBackend(
        provider,
        BackendOptions(
            work_root=options.work_dir,
            top_k=options.top_k,
            embedding_batch_size=options.embedding_batch_size,
            embedding_concurrency=options.embedding_concurrency,
            keep_case_data=options.keep_case_data,
        ),
    )
    for case in iter_cases(options.dataset_path):
        if case.question_id not in selected_id_set or case.question_id in completed_ids:
            continue
        row = await backend.run_case(case.as_retrieval_case(options.granularity))
        rows.append(row)
        append_jsonl(output_dir / "results.jsonl", row)
    order = {case_id: index for index, case_id in enumerate(selected_ids)}
    rows.sort(key=lambda row: order[str(row.get("case_id", ""))])
    write_jsonl(output_dir / "results.jsonl", rows)
    embedding_report = dict(provider.stats())
    embedding_report.pop("cache_path", None)
    summary = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "benchmark": "LongMemEval-S Cleaned retrieval",
        "granularity": options.granularity,
        "top_k": options.top_k,
        "selected_case_count": len(selected_ids),
        "selected_case_ids": selected_ids,
        "resumed_case_count": resumed_case_count,
        "dataset": dataset_report,
        "embedding": embedding_report,
        "embedding_prewarm": embedding_prewarm,
        "runtime": runtime,
        "results": aggregate_results(rows),
    }
    write_json(output_dir / "summary.json", summary)
    return summary
