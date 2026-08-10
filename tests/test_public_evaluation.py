from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import json
import re
import tarfile
import zipfile

import pytest

from a_memorix.evaluation.common import (
    CachedEmbeddingProvider,
    EmbeddingConfig,
    append_jsonl,
    evaluation_run_lock,
    evaluate_ranking,
    prepare_result_log,
)
from a_memorix.evaluation.cli import build_parser
from a_memorix.evaluation.comparison import compare_summaries
from a_memorix.evaluation.engine_backend import (
    AMemorixEvaluationBackend,
    AMemorixSharedNamespaceBackend,
    BackendOptions,
    RetrievalCase,
    RetrievalDocument,
)
from a_memorix.evaluation.longmemeval import iter_json_array, parse_case
from a_memorix.evaluation.swebench import (
    RepositoryCache,
    SWEBenchCase,
    _is_test_path,
    chunk_text,
    iter_cases as iter_swebench_cases,
)
from a_memorix.core.runtime.services.runtime_dependency_service import (
    MemoryRuntimeDependencyService,
)
import a_memorix.evaluation.swebench as swebench_module
import a_memorix.evaluation.common as common_module


class DeterministicEmbeddingProvider:
    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension
        self._token_indices: dict[str, int] = {}

    async def initialize(self) -> int:
        return self.dimension

    async def embed(
        self,
        texts: Sequence[str],
        *,
        dimensions: int | None = None,
    ) -> Sequence[Sequence[float]]:
        assert dimensions in {None, self.dimension}
        return [self._vector(text) for text in texts]

    async def prewarm(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 16,
        max_concurrent: int = 3,
    ) -> int:
        del batch_size, max_concurrent
        unique = list(dict.fromkeys(texts))
        await self.embed(unique)
        return len(unique)

    def _vector(self, text: str) -> list[float]:
        values = [0.0] * self.dimension
        for token in re.findall(r"[a-z0-9_]+", text.lower()):
            index = self._token_indices.setdefault(token, len(self._token_indices))
            if index >= self.dimension:
                raise ValueError("deterministic test vocabulary exceeded vector size")
            values[index] += 1.0
        norm = sum(value * value for value in values) ** 0.5 or 1.0
        return [value / norm for value in values]

    def fingerprint(self) -> Mapping[str, object]:
        return {
            "provider": "deterministic-test",
            "model": "token-hash",
            "dimension": self.dimension,
        }

    def stats(self) -> Mapping[str, object]:
        return self.fingerprint()


def test_evaluation_run_lock_rejects_concurrent_output_directory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"

    with evaluation_run_lock(output):
        with pytest.raises(RuntimeError, match="another evaluation"):
            with evaluation_run_lock(output):
                pass

    with evaluation_run_lock(output):
        pass


class CountingEmbeddingProvider:
    def __init__(self) -> None:
        self.request_count = 0
        self.batch_sizes: list[int] = []
        self.failures_remaining = 0

    async def embed(
        self,
        texts: Sequence[str],
        *,
        dimensions: int | None = None,
    ) -> Sequence[Sequence[float]]:
        assert dimensions is None
        self.request_count += 1
        self.batch_sizes.append(len(texts))
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("temporary embedding failure")
        return [[float(index + 1) for index in range(4)] for _ in texts]

    def fingerprint(self) -> Mapping[str, object]:
        return {"provider": "counting", "model": "counting-model"}


def test_evaluation_cli_accepts_explicit_embedding_throughput_options() -> None:
    args = build_parser().parse_args(
        [
            "swebench",
            "run",
            "--embedding-batch-size",
            "32",
            "--embedding-concurrency",
            "9",
        ]
    )

    assert args.embedding_batch_size == 32
    assert args.embedding_concurrency == 9


def test_evaluation_cli_accepts_full_namespace_mode() -> None:
    args = build_parser().parse_args(["longmemeval", "run", "--namespace-mode", "full"])

    assert args.namespace_mode == "full"


def test_embedding_config_supports_positional_file_without_exposing_key(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.txt"
    path.write_text(
        "https://embedding.example/v1\nsecret-value\nmodel-name\n",
        encoding="utf-8",
    )

    config = EmbeddingConfig.from_file(path)

    assert config.embeddings_url == "https://embedding.example/v1/embeddings"
    assert config.model == "model-name"
    assert "secret-value" not in repr(config)
    assert "secret-value" not in str(config.public_fingerprint())


def test_embedding_config_supports_key_value_file(tmp_path: Path) -> None:
    path = tmp_path / "config.txt"
    path.write_text(
        "endpoint=https://embedding.example\napi_key=secret-value\nmodel=model-name\n",
        encoding="utf-8",
    )

    config = EmbeddingConfig.from_file(path)

    assert config.embeddings_url == "https://embedding.example/v1/embeddings"
    assert config.model == "model-name"


@pytest.mark.asyncio
async def test_embedding_cache_prewarms_unique_texts_in_batches(
    tmp_path: Path,
) -> None:
    raw = CountingEmbeddingProvider()
    provider = CachedEmbeddingProvider(  # type: ignore[arg-type]
        raw,
        tmp_path / "embeddings.sqlite3",
    )
    try:
        count = await provider.prewarm(
            ["one", "two", "three", "four", "five", "one"],
            batch_size=2,
            max_concurrent=2,
        )
        first_request_count = raw.request_count
        await provider.prewarm(
            ["one", "two", "three", "four", "five"],
            batch_size=2,
            max_concurrent=2,
        )
    finally:
        provider.close()

    assert count == 5
    assert first_request_count == 4
    assert raw.request_count == first_request_count
    assert raw.batch_sizes[0] == 1
    assert sorted(raw.batch_sizes[1:]) == [1, 2, 2]

    second_raw = CountingEmbeddingProvider()
    second = CachedEmbeddingProvider(  # type: ignore[arg-type]
        second_raw,
        tmp_path / "embeddings.sqlite3",
    )
    try:
        assert await second.initialize() == 4
    finally:
        second.close()
    assert second_raw.request_count == 0


@pytest.mark.asyncio
async def test_embedding_cache_prewarms_retry_transient_failures(
    tmp_path: Path,
) -> None:
    raw = CountingEmbeddingProvider()
    provider = CachedEmbeddingProvider(  # type: ignore[arg-type]
        raw,
        tmp_path / "embeddings.sqlite3",
    )
    try:
        await provider.initialize()
        raw.failures_remaining = 1
        await provider.prewarm(["one"], batch_size=1, max_concurrent=1)
    finally:
        provider.close()

    assert raw.request_count == 3


@pytest.mark.asyncio
async def test_embedding_cache_tolerates_sustained_transient_overload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw = CountingEmbeddingProvider()
    provider = CachedEmbeddingProvider(  # type: ignore[arg-type]
        raw,
        tmp_path / "embeddings.sqlite3",
    )
    try:
        await provider.initialize()
        raw.failures_remaining = 4

        async def no_sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr(common_module.asyncio, "sleep", no_sleep)
        await provider.prewarm(["one"], batch_size=1, max_concurrent=1)
    finally:
        provider.close()

    assert raw.request_count == 6


def test_longmemeval_json_array_streams_across_small_chunks(tmp_path: Path) -> None:
    path = tmp_path / "public.json"
    payload = [{"question_id": "one"}, {"question_id": "two"}]
    path.write_text(json.dumps(payload), encoding="utf-8")

    items = list(iter_json_array(path, chunk_chars=17))

    assert [item["question_id"] for item in items] == ["one", "two"]


def test_ranking_metrics_cover_fraction_mrr_and_ndcg() -> None:
    metrics = evaluate_ranking(
        ["noise", "gold-a", "gold-b"],
        ["gold-a", "gold-b"],
        cutoffs=(1, 2, 3),
    )

    assert metrics["recall_any@1"] == 0.0
    assert metrics["recall_fraction@2"] == 0.5
    assert metrics["recall_all@3"] == 1.0
    assert metrics["mrr"] == 0.5
    assert 0.0 < metrics["ndcg@3"] < 1.0


def test_result_log_resume_keeps_completed_cases_and_drops_failures(
    tmp_path: Path,
) -> None:
    manifest = {
        "evaluation_schema_version": 1,
        "selected_case_ids": ["completed", "failed"],
    }
    output = tmp_path / "run"

    assert prepare_result_log(output, manifest, resume=False) == []
    append_jsonl(
        output / "results.jsonl",
        {"case_id": "completed", "status": "completed"},
    )
    append_jsonl(
        output / "results.jsonl",
        {"case_id": "failed", "status": "failed"},
    )

    rows = prepare_result_log(output, manifest, resume=True)

    assert rows == [{"case_id": "completed", "status": "completed"}]
    persisted = (output / "results.jsonl").read_text(encoding="utf-8")
    assert '"completed"' in persisted
    assert '"failed"' not in persisted
    with pytest.raises(ValueError, match="does not match"):
        prepare_result_log(
            output,
            {**manifest, "evaluation_schema_version": 2},
            resume=True,
        )


def test_summary_comparison_checks_compatibility_quality_and_warm_latency() -> None:
    baseline = _comparison_summary()
    candidate = _comparison_summary()

    passed = compare_summaries(baseline, candidate)

    assert passed["status"] == "passed"

    candidate["dataset"]["sha256"] = "different"
    candidate["results"]["metrics"]["mrr"] = 0.8
    candidate["results"]["timing_ms"]["search_p95"] = 20.0

    failed = compare_summaries(baseline, candidate)

    assert failed["status"] == "failed"
    assert failed["failures"] == [
        "incompatible_summaries",
        "quality_regression:mrr",
        "latency_regression:search_p95",
    ]


def test_summary_comparison_rejects_resumed_performance_run() -> None:
    baseline = _comparison_summary()
    candidate = _comparison_summary()
    candidate["resumed_case_count"] = 1

    performance = compare_summaries(baseline, candidate)
    quality = compare_summaries(baseline, candidate, quality_only=True)

    assert performance["status"] == "failed"
    assert performance["failures"] == ["performance_comparison_requires_single_run"]
    assert quality["status"] == "passed"


def test_summary_comparison_rejects_different_embedding_prewarm_settings() -> None:
    baseline = _comparison_summary()
    candidate = _comparison_summary()
    candidate["embedding_prewarm"]["max_concurrent"] = 9

    comparison = compare_summaries(baseline, candidate)

    assert comparison["status"] == "failed"
    assert comparison["failures"] == ["incompatible_summaries"]
    assert comparison["incompatibilities"] == [
        {
            "field": "embedding_prewarm.max_concurrent",
            "baseline": 3,
            "candidate": 9,
        }
    ]


def test_summary_comparison_rejects_different_namespace_modes() -> None:
    baseline = _comparison_summary()
    candidate = _comparison_summary()
    candidate["namespace_mode"] = "full"

    comparison = compare_summaries(baseline, candidate, quality_only=True)

    assert comparison["status"] == "failed"
    assert comparison["incompatibilities"] == [
        {
            "field": "namespace_mode",
            "baseline": "isolated",
            "candidate": "full",
        }
    ]


def test_summary_comparison_uses_full_corpus_ingest_timing() -> None:
    baseline = _comparison_summary()
    candidate = _comparison_summary()
    for summary in (baseline, candidate):
        summary["namespace_mode"] = "full"
        summary["corpus"] = {"timing_ms": {"ingest": 100.0}}
        del summary["results"]["timing_ms"]["ingest_p95"]

    comparison = compare_summaries(baseline, candidate)

    assert comparison["status"] == "passed"
    assert [item["timing"] for item in comparison["performance"]] == [
        "corpus_ingest",
        "search_p95",
        "total_p95",
    ]


def test_summary_comparison_accepts_schema_v1_legacy_prewarm_defaults() -> None:
    legacy = _comparison_summary()
    candidate = _comparison_summary()
    del legacy["embedding_prewarm"]

    comparison = compare_summaries(legacy, candidate)

    assert comparison["status"] == "passed"
    assert comparison["incompatibilities"] == []


def _comparison_summary() -> dict[str, object]:
    return {
        "evaluation_schema_version": 1,
        "benchmark": "public",
        "granularity": "turn",
        "top_k": 10,
        "selected_case_count": 1,
        "resumed_case_count": 0,
        "selected_case_ids": ["case-1"],
        "dataset": {"sha256": "dataset"},
        "embedding": {
            "cache_misses": 0,
            "fingerprint": {
                "provider": "test",
                "model": "test-model",
                "dimension": 4,
                "endpoint_sha256": "endpoint",
            },
        },
        "embedding_prewarm": {"batch_size": 16, "max_concurrent": 3},
        "runtime": {
            "python_version": "3.12.0",
            "system": "TestOS",
            "machine": "test-machine",
            "packages": {
                "numpy": "1.0",
                "scipy": "1.0",
                "faiss-cpu": "1.0",
            },
        },
        "results": {
            "completed_count": 1,
            "failed_count": 0,
            "metrics": {
                "mrr": 1.0,
                "ndcg@10": 1.0,
                "recall_any@10": 1.0,
                "recall_fraction@10": 1.0,
            },
            "timing_ms": {
                "ingest_p95": 10.0,
                "search_p95": 10.0,
                "total_p95": 20.0,
            },
        },
    }


def test_longmemeval_case_builds_session_and_turn_gold_ids() -> None:
    case = parse_case(
        {
            "question_id": "public-case",
            "question_type": "multi-session",
            "question": "Where is the public artifact?",
            "question_date": "2024/01/03 (Wed) 12:00",
            "answer_session_ids": ["answer_session"],
            "haystack_dates": [
                "2024/01/01 (Mon) 12:00",
                "2024/01/02 (Tue) 12:00",
            ],
            "haystack_session_ids": ["history_session", "answer_session"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "Unrelated public note."},
                    {"role": "assistant", "content": "Acknowledged."},
                ],
                [
                    {
                        "role": "user",
                        "content": "The public artifact is in cabinet seven.",
                        "has_answer": True,
                    },
                    {"role": "assistant", "content": "Recorded."},
                ],
            ],
        }
    )

    session_case = case.as_retrieval_case("session")
    turn_case = case.as_retrieval_case("turn")

    assert session_case.gold_ids == ("answer_session",)
    assert turn_case.gold_ids == ("answer_session_1",)
    assert len(session_case.documents) == 2
    assert len(turn_case.documents) == 4
    assert "Assistant: Recorded." in session_case.documents[1].text


def test_longmemeval_turn_corpus_keeps_assistant_evidence() -> None:
    case = parse_case(
        {
            "question_id": "assistant-evidence",
            "question_type": "single-session-assistant",
            "question": "Which public code did the assistant provide?",
            "question_date": "2024/01/03 (Wed) 12:00",
            "answer_session_ids": ["evidence-session"],
            "haystack_dates": ["2024/01/01 (Mon) 12:00"],
            "haystack_session_ids": ["evidence-session"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "Please provide a public code."},
                    {
                        "role": "assistant",
                        "content": "The public code is delta-seven.",
                        "has_answer": True,
                    },
                ]
            ],
        }
    )

    retrieval = case.as_retrieval_case("turn")

    assert case.retrieval_scored is True
    assert retrieval.gold_ids == ("evidence-session_2",)
    assert retrieval.documents[1].text == "Assistant: The public code is delta-seven."


def test_swebench_patch_files_and_chunks_use_file_level_metric_ids() -> None:
    case = SWEBenchCase(
        instance_id="owner__repo-1",
        repo="owner/repo",
        base_commit="a" * 40,
        problem_statement="Update parser behavior",
        patch=(
            "--- a/src/parser.py\n+++ b/src/parser.py\n@@ -1 +1 @@\n-old\n+new\n"
            "--- a/src/helpers.py\n+++ b/src/helpers.py\n@@ -1 +1 @@\n-old\n+new\n"
        ),
    )

    chunks = list(chunk_text("line\n" * 400, max_chars=512, overlap_chars=64))

    assert case.gold_files == ("src/parser.py", "src/helpers.py")
    assert len(chunks) > 1
    assert _is_test_path("tests/test_parser.py") is True
    assert _is_test_path("src/parser.py") is False


def test_swebench_json_zip_transport_uses_the_same_case_model(
    tmp_path: Path,
) -> None:
    path = tmp_path / "public.zip"
    payload = {
        "instance_id": "owner__repo-1",
        "repo": "owner/repo",
        "base_commit": "a" * 40,
        "problem_statement": "Update public parser behavior",
        "patch": "--- a/src/parser.py\n+++ b/src/parser.py\n",
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("test/owner__repo-1.json", json.dumps(payload))

    cases = list(iter_swebench_cases(path))

    assert len(cases) == 1
    assert cases[0].instance_id == "owner__repo-1"
    assert cases[0].gold_files == ("src/parser.py",)


def test_swebench_repository_cache_skips_fetch_for_local_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "owner__repo"
    (target / ".git").mkdir(parents=True)
    commands: list[tuple[str, ...]] = []
    monkeypatch.setattr(swebench_module, "_git_succeeds", lambda *args: True)
    monkeypatch.setattr(
        swebench_module,
        "_run_git",
        lambda *args: commands.append(tuple(args)),
    )

    result = RepositoryCache(tmp_path).checkout("owner/repo", "a" * 40)

    assert result == target
    assert all("fetch" not in command for command in commands)
    assert [command[2] for command in commands] == ["checkout", "clean"]


def test_swebench_repository_cache_replaces_partial_clone_only_after_full_clone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "owner__repo"
    (target / ".git").mkdir(parents=True)
    (target / "partial-marker").write_text("partial", encoding="utf-8")
    clone_modes: list[bool] = []

    def clone(replacement: Path, _slug: str, *, full: bool) -> None:
        clone_modes.append(full)
        (replacement / ".git").mkdir(parents=True)
        (replacement / "full-marker").write_text("full", encoding="utf-8")

    monkeypatch.setattr(swebench_module, "_is_partial_clone", lambda _path: True)
    monkeypatch.setattr(swebench_module, "_clone_repository", clone)
    monkeypatch.setattr(swebench_module, "_git_succeeds", lambda *args: True)
    monkeypatch.setattr(swebench_module, "_run_git", lambda *args: None)

    result = RepositoryCache(
        tmp_path,
        full_repositories={"owner/repo"},
    ).checkout("owner/repo", "a" * 40)

    assert result == target
    assert clone_modes == [True]
    assert not (target / "partial-marker").exists()
    assert (target / "full-marker").read_text(encoding="utf-8") == "full"


def test_swebench_repository_cache_prefetches_missing_selected_commits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "owner__repo"
    (target / ".git").mkdir(parents=True)
    fetched = False
    commands: list[tuple[str, ...]] = []

    def git_succeeds(*args: str) -> bool:
        if "cat-file" in args:
            return fetched
        return True

    def run_git(*args: str) -> None:
        nonlocal fetched
        commands.append(tuple(args))
        if "fetch" in args:
            fetched = True

    first = "a" * 40
    second = "b" * 40
    monkeypatch.setattr(swebench_module, "_is_partial_clone", lambda _path: False)
    monkeypatch.setattr(swebench_module, "_git_succeeds", git_succeeds)
    monkeypatch.setattr(swebench_module, "_run_git", run_git)

    RepositoryCache(
        tmp_path,
        full_repositories={"owner/repo"},
        commits_by_repo={"owner/repo": (first, second)},
    ).checkout("owner/repo", first)

    fetch = next(command for command in commands if "fetch" in command)
    assert "--no-filter" in fetch
    assert first in fetch
    assert second in fetch


def test_swebench_repository_cache_falls_back_to_codeload_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commit = "a" * 40
    fixture_root = tmp_path / "fixture" / f"owner__repo-{commit[:7]}"
    fixture_root.mkdir(parents=True)
    (fixture_root / "public.py").write_text("PUBLIC = True\n", encoding="utf-8")
    fixture_archive = tmp_path / "fixture.tar.gz"
    with tarfile.open(fixture_archive, mode="w:gz") as archive:
        archive.add(fixture_root, arcname=fixture_root.name)
    downloads = 0

    def clone_failure(_target: Path, _slug: str, *, full: bool) -> None:
        del full
        raise RuntimeError("simulated Git connection failure")

    def download(target: Path, *, slug: str, commit: str, attempts: int = 3) -> None:
        nonlocal downloads
        del slug, commit, attempts
        downloads += 1
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(fixture_archive.read_bytes())

    monkeypatch.setattr(swebench_module, "_clone_repository", clone_failure)
    monkeypatch.setattr(swebench_module, "_download_repository_archive", download)
    cache_root = tmp_path / "repositories"
    cache = RepositoryCache(cache_root, full_repositories={"owner/repo"})

    first = cache.checkout("owner/repo", commit)
    second = RepositoryCache(cache_root).checkout("owner/repo", commit)

    assert first == second == cache_root / "owner__repo"
    assert (first / "public.py").read_text(encoding="utf-8") == "PUBLIC = True\n"
    assert downloads == 1


def test_swebench_git_retry_recovers_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def flaky_git(*_args: str) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("temporary network failure")

    monkeypatch.setattr(swebench_module, "_run_git", flaky_git)
    monkeypatch.setattr(swebench_module.time, "sleep", lambda _seconds: None)

    swebench_module._retry_git("checkout")

    assert attempts == 3


def test_swebench_repository_cleanup_retries_windows_file_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "repo"
    target.mkdir()
    original_rmtree = swebench_module.shutil.rmtree
    attempts = 0

    def transient_lock(path: Path, **_kwargs: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(5, "simulated Windows file lock", str(path))
        original_rmtree(path)

    monkeypatch.setattr(swebench_module.shutil, "rmtree", transient_lock)
    monkeypatch.setattr(swebench_module.time, "sleep", lambda _seconds: None)

    swebench_module._remove_repository_dir(target)

    assert attempts == 3
    assert not target.exists()


def test_swebench_repository_move_retries_windows_file_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    original_rename = Path.rename
    attempts = 0

    def transient_lock(path: Path, destination: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(5, "simulated Windows file lock", str(path))
        return original_rename(path, destination)

    monkeypatch.setattr(Path, "rename", transient_lock)
    monkeypatch.setattr(swebench_module.time, "sleep", lambda _seconds: None)

    swebench_module._move_repository_dir(source, target)

    assert attempts == 3
    assert target.is_dir()
    assert not source.exists()


@pytest.mark.asyncio
async def test_public_evaluation_backend_scores_real_application_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    persist_calls = 0
    original_persist = MemoryRuntimeDependencyService._persist

    def tracked_persist(
        service: MemoryRuntimeDependencyService,
        *,
        force_vectors: bool = False,
    ) -> None:
        nonlocal persist_calls
        persist_calls += 1
        original_persist(service, force_vectors=force_vectors)

    monkeypatch.setattr(MemoryRuntimeDependencyService, "_persist", tracked_persist)
    backend = AMemorixEvaluationBackend(
        DeterministicEmbeddingProvider(),  # type: ignore[arg-type]
        BackendOptions(work_root=tmp_path / "work", top_k=2),
    )
    case = RetrievalCase(
        case_id="offline-public-case",
        case_type="cross-session-smoke",
        query="orbital beacon",
        documents=(
            RetrievalDocument(
                document_id="relevant",
                text="The orbital beacon calibration procedure uses a reference clock.",
                metadata={"session_id": "session-01"},
            ),
            RetrievalDocument(
                document_id="noise-01",
                text="The garden irrigation schedule starts before sunrise.",
                metadata={"session_id": "session-02"},
            ),
            RetrievalDocument(
                document_id="noise-02",
                text="The accounting archive contains quarterly invoice records.",
                metadata={"session_id": "session-03"},
            ),
            RetrievalDocument(
                document_id="noise-03",
                text="The workshop inventory lists replacement hinges and handles.",
                metadata={"session_id": "session-04"},
            ),
            RetrievalDocument(
                document_id="noise-04",
                text="The workshop inventory lists replacement hinges and handles.",
                metadata={"session_id": "session-05"},
            ),
        ),
        gold_ids=("relevant",),
    )

    result = await backend.run_case(case)

    assert result["status"] == "completed", result
    assert result["metrics"]["recall_any@1"] == 1.0, json.dumps(
        result,
        indent=2,
    )
    assert result["ranked_ids"][0] == "relevant"
    assert persist_calls == 2
    assert list((tmp_path / "work").iterdir()) == []


@pytest.mark.asyncio
async def test_public_evaluation_backend_scores_code_file_metric_ids(
    tmp_path: Path,
) -> None:
    backend = AMemorixEvaluationBackend(
        DeterministicEmbeddingProvider(),  # type: ignore[arg-type]
        BackendOptions(work_root=tmp_path / "work", top_k=3),
    )
    case = RetrievalCase(
        case_id="offline-code-case",
        case_type="code-smoke",
        query="parser malformed token recovery",
        documents=(
            RetrievalDocument(
                document_id="src/parser.py#chunk-0",
                metric_id="src/parser.py",
                text="Parser malformed token recovery keeps the next valid expression.",
            ),
            RetrievalDocument(
                document_id="src/parser.py#chunk-1",
                metric_id="src/parser.py",
                text="Parser helpers normalize source positions.",
            ),
            RetrievalDocument(
                document_id="src/cache.py#chunk-0",
                metric_id="src/cache.py",
                text="Cache entries expire after the configured interval.",
            ),
        ),
        gold_ids=("src/parser.py",),
    )

    result = await backend.run_case(case)

    assert result["status"] == "completed", result
    assert result["metrics"]["recall_any@1"] == 1.0
    assert result["ranked_ids"][0] == "src/parser.py"
    assert result["ranked_ids"].count("src/parser.py") == 1


@pytest.mark.asyncio
async def test_shared_namespace_backend_builds_one_deduplicated_corpus(
    tmp_path: Path,
) -> None:
    shared_document = RetrievalDocument(
        document_id="shared-session",
        text="The orbital beacon uses a reference clock.",
        timestamp=200.0,
    )
    cases = (
        RetrievalCase(
            case_id="case-one",
            case_type="cross-session-smoke",
            query="orbital beacon reference clock",
            documents=(
                shared_document,
                RetrievalDocument(
                    document_id="noise-one",
                    text="The garden irrigation begins before sunrise.",
                ),
            ),
            gold_ids=("shared-session",),
        ),
        RetrievalCase(
            case_id="case-two",
            case_type="cross-session-smoke",
            query="accounting invoice archive",
            documents=(
                RetrievalDocument(
                    document_id="shared-session",
                    text=shared_document.text,
                    timestamp=100.0,
                ),
                RetrievalDocument(
                    document_id="invoice-session",
                    text="The accounting invoice archive is stored offsite.",
                ),
            ),
            gold_ids=("invoice-session",),
        ),
    )
    backend = AMemorixSharedNamespaceBackend(
        DeterministicEmbeddingProvider(),  # type: ignore[arg-type]
        BackendOptions(work_root=tmp_path / "work", top_k=3),
    )
    documents, _ = backend._build_corpus(cases)

    try:
        corpus = await backend.prepare(cases)
        first = await backend.run_case(cases[0])
        second = await backend.run_case(cases[1])
    finally:
        await backend.close()

    assert corpus["occurrence_document_count"] == 4
    assert corpus["logical_document_count"] == 3
    assert corpus["unique_memory_count"] == 3
    assert corpus["duplicate_occurrence_count"] == 1
    assert corpus["date_variant_document_count"] == 1
    canonical_shared = next(
        item for item in documents if item.document_id == "shared-session"
    )
    assert canonical_shared.timestamp == 100.0
    assert canonical_shared.metadata["evaluation_date_variant_count"] == 2
    assert first["metrics"]["recall_any@1"] == 1.0
    assert second["metrics"]["recall_any@1"] == 1.0
    assert "ingest" not in first["timing_ms"]
    assert list((tmp_path / "work").iterdir()) == []


def test_shared_namespace_corpus_rejects_conflicting_document_content() -> None:
    cases = (
        RetrievalCase(
            case_id="case-one",
            case_type="smoke",
            query="first",
            documents=(RetrievalDocument(document_id="same", text="first"),),
            gold_ids=("same",),
        ),
        RetrievalCase(
            case_id="case-two",
            case_type="smoke",
            query="second",
            documents=(RetrievalDocument(document_id="same", text="second"),),
            gold_ids=("same",),
        ),
    )

    with pytest.raises(ValueError, match="conflicting content"):
        AMemorixSharedNamespaceBackend._build_corpus(cases)
