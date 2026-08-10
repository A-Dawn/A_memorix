"""SWE-bench Lite issue-to-source-file retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import time
import zipfile

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
    "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/resolve/"
    "main/data/test-00000-of-00001.parquet",
    "https://hf-mirror.com/datasets/princeton-nlp/SWE-bench_Lite/resolve/"
    "main/data/test-00000-of-00001.parquet",
)
DATASET_BYTES = 1_119_540
DATASET_SHA256 = "7a21f37b8bc179c7db5beeb14e88ac538ba283455c776e6b2535bbfb6e3551b4"
DATASET_ROWS = 300
DATASET_SOURCE = "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite"
DATASET_MIRROR_FILENAME = "test-json-v1.2.15.zip"
DATASET_MIRROR_URL = (
    "https://raw.githubusercontent.com/harvard-cns/orla/v1.2.15/"
    "examples/swe_bench_lite/dataset.zip"
)
DATASET_MIRROR_BYTES = 363_320
DATASET_MIRROR_SHA256 = (
    "10444a1a74cc53ac204036366e85f5fedf71a12c1d67d5fc6078f0806da2fb11"
)
DATASET_MIRROR_SOURCE = "https://github.com/harvard-cns/orla/tree/v1.2.15"


@dataclass(frozen=True)
class SWEBenchCase:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str

    @property
    def gold_files(self) -> tuple[str, ...]:
        files: list[str] = []
        for match in re.finditer(r"^---\s+a/(.+)$", self.patch, re.MULTILINE):
            path = match.group(1).strip()
            if path != "/dev/null" and path not in files:
                files.append(path)
        return tuple(files)


@dataclass(frozen=True)
class RunOptions:
    dataset_path: Path
    output_dir: Path
    work_dir: Path
    repo_cache_dir: Path
    top_k: int = 20
    limit: int = 0
    instance_ids: tuple[str, ...] = ()
    chunk_chars: int = 8_000
    chunk_overlap_chars: int = 800
    keep_case_data: bool = False
    resume: bool = False
    embedding_batch_size: int = 16
    embedding_concurrency: int = 3


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_dataset_path(suite_root: str | Path) -> Path:
    root = Path(suite_root).resolve()
    official = root / "test.parquet"
    mirror = root / DATASET_MIRROR_FILENAME
    if official.is_file():
        return official
    if mirror.is_file():
        return mirror
    return official


def download_dataset(path: str | Path, *, retries: int = 2) -> dict[str, Any]:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.name == DATASET_MIRROR_FILENAME:
        return _download_mirror(target, retries=retries)
    partial = target.with_suffix(target.suffix + ".partial")
    if target.exists() and target.stat().st_size == DATASET_BYTES:
        return validate_dataset(target)
    for attempt in range(max(1, int(retries))):
        url = DATASET_URLS[attempt % len(DATASET_URLS)]
        current_size = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "A_memorix-evaluation/1.0"}
        if current_size:
            headers["Range"] = f"bytes={current_size}-"
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=30.0) as response:
                status = int(getattr(response, "status", response.getcode()))
                mode = "ab" if current_size and status == 206 else "wb"
                with partial.open(mode) as output:
                    while chunk := response.read(1 << 20):
                        output.write(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                        if output.tell() > DATASET_BYTES:
                            raise RuntimeError(
                                "SWE-bench download exceeded pinned size"
                            )
            if partial.stat().st_size != DATASET_BYTES:
                raise RuntimeError(
                    f"SWE-bench download incomplete: {partial.stat().st_size}/{DATASET_BYTES}"
                )
            partial.replace(target)
            return validate_dataset(target)
        except (HTTPError, URLError, OSError, RuntimeError, ValueError):
            time.sleep(min(10.0, 1.5 * (attempt + 1)))
    return _download_mirror(
        target.with_name(DATASET_MIRROR_FILENAME),
        retries=retries,
    )


def _download_mirror(path: Path, *, retries: int) -> dict[str, Any]:
    if path.is_file():
        return validate_dataset(path)
    partial = path.with_suffix(path.suffix + ".partial")
    errors: list[str] = []
    for attempt in range(max(1, int(retries))):
        current_size = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "A_memorix-evaluation/1.0"}
        if current_size:
            headers["Range"] = f"bytes={current_size}-"
        try:
            request = Request(DATASET_MIRROR_URL, headers=headers)
            with urlopen(request, timeout=60.0) as response:
                status = int(getattr(response, "status", response.getcode()))
                mode = "ab" if current_size and status == 206 else "wb"
                with partial.open(mode) as output:
                    while chunk := response.read(1 << 20):
                        output.write(chunk)
                        if output.tell() > DATASET_MIRROR_BYTES:
                            raise RuntimeError(
                                "SWE-bench mirror download exceeded pinned size"
                            )
            if partial.stat().st_size != DATASET_MIRROR_BYTES:
                raise RuntimeError("SWE-bench mirror download is incomplete")
            partial.replace(path)
            return validate_dataset(path)
        except (HTTPError, URLError, OSError, RuntimeError, ValueError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            time.sleep(min(5.0, attempt + 1.0))
    raise RuntimeError(
        "SWE-bench Lite download failed; partial files were retained: "
        + " | ".join(errors[-2:])
    )


def validate_dataset(path: str | Path) -> dict[str, Any]:
    dataset_path = Path(path).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    actual_bytes = dataset_path.stat().st_size
    actual_hash = file_sha256(dataset_path)
    is_mirror = dataset_path.suffix.lower() == ".zip"
    expected_bytes = DATASET_MIRROR_BYTES if is_mirror else DATASET_BYTES
    expected_hash = DATASET_MIRROR_SHA256 if is_mirror else DATASET_SHA256
    if actual_bytes != expected_bytes or actual_hash != expected_hash:
        raise ValueError(
            "SWE-bench Lite file does not match the pinned artifact: "
            f"bytes={actual_bytes}, sha256={actual_hash}"
        )
    cases = list(iter_cases(dataset_path))
    if len(cases) != DATASET_ROWS:
        raise ValueError(
            f"SWE-bench Lite row count mismatch: {len(cases)} != {DATASET_ROWS}"
        )
    missing_gold = [case.instance_id for case in cases if not case.gold_files]
    if missing_gold:
        raise ValueError(
            f"SWE-bench Lite cases without patch files: {missing_gold[:5]}"
        )
    if len({case.instance_id for case in cases}) != DATASET_ROWS:
        raise ValueError("SWE-bench Lite contains duplicate instance IDs")
    return {
        "path": str(dataset_path),
        "bytes": actual_bytes,
        "sha256": actual_hash,
        "case_count": len(cases),
        "repository_count": len({case.repo for case in cases}),
        "source": DATASET_SOURCE,
        "transport": "pinned-json-mirror" if is_mirror else "official-parquet",
        "transport_source": DATASET_MIRROR_SOURCE if is_mirror else DATASET_SOURCE,
        "benchmark_code_license": "MIT",
        "repository_source_licenses": "upstream",
    }


def iter_cases(path: str | Path) -> Iterator[SWEBenchCase]:
    dataset_path = Path(path).resolve()
    if dataset_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(dataset_path) as archive:
            names = sorted(
                name
                for name in archive.namelist()
                if name.startswith("test/") and name.endswith(".json")
            )
            for name in names:
                payload = json.loads(archive.read(name))
                if not isinstance(payload, dict):
                    raise ValueError(f"SWE-bench mirror row is not an object: {name}")
                case = _parse_case(payload)
                if Path(name).stem != case.instance_id:
                    raise ValueError(f"SWE-bench mirror filename mismatch: {name}")
                yield case
        return
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "SWE-bench evaluation requires the evaluation extra: "
            "pip install -e '.[evaluation]'"
        ) from exc
    table = parquet.read_table(dataset_path)
    required = {
        "instance_id",
        "repo",
        "base_commit",
        "problem_statement",
        "patch",
    }
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(f"SWE-bench dataset is missing columns: {sorted(missing)}")
    for row in table.select(sorted(required)).to_pylist():
        yield _parse_case(row)


def _parse_case(payload: dict[str, Any]) -> SWEBenchCase:
    values: dict[str, str] = {}
    for key in ("instance_id", "repo", "base_commit", "problem_statement", "patch"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"SWE-bench field must be non-empty text: {key}")
        values[key] = value.strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", values["base_commit"]):
        raise ValueError(f"SWE-bench base commit is invalid: {values['instance_id']}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", values["repo"]):
        raise ValueError(f"SWE-bench repository is invalid: {values['repo']}")
    return SWEBenchCase(**values)


class RepositoryCache:
    """Own generated SWE-bench mirrors and switch them to pinned commits."""

    def __init__(
        self,
        root: str | Path,
        *,
        full_repositories: Iterable[str] = (),
        commits_by_repo: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.full_repositories = frozenset(full_repositories)
        self.commits_by_repo = {
            repo: tuple(dict.fromkeys(commits))
            for repo, commits in (commits_by_repo or {}).items()
        }
        self._prepared_repositories: set[str] = set()

    def checkout(self, repo: str, commit: str) -> Path:
        slug = repo.replace("/", "__")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+", slug):
            raise ValueError(f"invalid SWE-bench repository name: {repo}")
        target = (self.root / slug).resolve()
        if target.parent != self.root:
            raise RuntimeError("SWE-bench repository path escaped the cache root")
        if _archive_state_commit(self.root, slug):
            return _checkout_repository_archive(
                self.root,
                target,
                slug=slug,
                commit=commit,
            )
        full_clone = repo in self.full_repositories
        if (target / ".git").is_dir() and full_clone and _is_partial_clone(target):
            replacement = target.with_name(f".{target.name}.full")
            try:
                _clone_repository(replacement, slug, full=True)
            except RuntimeError as git_error:
                return _archive_after_git_failure(
                    self.root,
                    target,
                    slug=slug,
                    commit=commit,
                    git_error=git_error,
                )
            _remove_repository_dir(target)
            replacement.replace(target)
        if not (target / ".git").is_dir():
            try:
                _clone_repository(target, slug, full=full_clone)
            except RuntimeError as git_error:
                return _archive_after_git_failure(
                    self.root,
                    target,
                    slug=slug,
                    commit=commit,
                    git_error=git_error,
                )
        if full_clone and repo not in self._prepared_repositories:
            _prefetch_commits(target, self.commits_by_repo.get(repo, ()))
            self._prepared_repositories.add(repo)
        if not _git_succeeds(
            "-C", str(target), "cat-file", "-e", f"{commit}^{{commit}}"
        ):
            _retry_git("-C", str(target), "fetch", "origin", commit, "--depth=1")
        _retry_git("-C", str(target), "checkout", "--detach", "--force", commit)
        _run_git("-C", str(target), "clean", "-fdx")
        return target


def _archive_after_git_failure(
    root: Path,
    target: Path,
    *,
    slug: str,
    commit: str,
    git_error: RuntimeError,
) -> Path:
    try:
        return _checkout_repository_archive(
            root,
            target,
            slug=slug,
            commit=commit,
        )
    except Exception as archive_error:
        raise RuntimeError(
            f"SWE-bench repository preparation failed via Git and codeload archive; "
            f"git_error={git_error}; archive_error={archive_error}"
        ) from archive_error


def _archive_state_path(root: Path, slug: str) -> Path:
    return root / ".archive-state" / f"{slug}.json"


def _archive_state_commit(root: Path, slug: str) -> str:
    state_path = _archive_state_path(root, slug)
    if not state_path.is_file():
        return ""
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict) or payload.get("slug") != slug:
        return ""
    commit = str(payload.get("commit", ""))
    return commit if re.fullmatch(r"[0-9a-fA-F]{40}", commit) else ""


def _checkout_repository_archive(
    root: Path,
    target: Path,
    *,
    slug: str,
    commit: str,
) -> Path:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError(f"invalid SWE-bench archive commit: {commit}")
    if _archive_state_commit(root, slug) == commit and target.is_dir():
        return target

    archive_dir = (root / ".archives" / slug).resolve()
    if archive_dir.parent.parent != root:
        raise RuntimeError("SWE-bench archive path escaped the cache root")
    archive_path = archive_dir / f"{commit}.tar.gz"
    if not archive_path.is_file():
        _download_repository_archive(archive_path, slug=slug, commit=commit)

    extract_root = (root / f".{slug}.archive-extract").resolve()
    staged = (root / f".{slug}.archive-ready").resolve()
    if extract_root.parent != root or staged.parent != root:
        raise RuntimeError("SWE-bench archive staging path escaped the cache root")
    _remove_repository_dir(extract_root)
    _remove_repository_dir(staged)
    extract_root.mkdir(parents=True)
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) > 250_000:
                raise RuntimeError("SWE-bench archive contains too many entries")
            expanded_bytes = sum(max(0, int(member.size)) for member in members)
            if expanded_bytes > 2 * 1024**3:
                raise RuntimeError("SWE-bench archive expands beyond 2 GiB")
            archive.extractall(extract_root, members=members, filter="data")
        children = list(extract_root.iterdir())
        if len(children) != 1 or not children[0].is_dir():
            raise RuntimeError("SWE-bench archive must contain one repository root")
        _move_repository_dir(children[0], staged)
    finally:
        _remove_repository_dir(extract_root)

    if target.exists():
        _remove_repository_dir(target)
    _move_repository_dir(staged, target)
    write_json(
        _archive_state_path(root, slug),
        {"slug": slug, "commit": commit, "source": "github-codeload"},
    )
    return target


def _download_repository_archive(
    target: Path,
    *,
    slug: str,
    commit: str,
    attempts: int = 3,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.download")
    url = f"https://codeload.github.com/swe-bench-repos/{slug}/tar.gz/{commit}"
    last_error: BaseException | None = None
    for attempt in range(max(1, int(attempts))):
        temporary.unlink(missing_ok=True)
        try:
            request = Request(url, headers={"User-Agent": "A_memorix-evaluation/1.0"})
            with urlopen(request, timeout=120.0) as response, temporary.open("wb") as output:
                total = 0
                while chunk := response.read(1 << 20):
                    total += len(chunk)
                    if total > 1024**3:
                        raise RuntimeError("SWE-bench archive download exceeds 1 GiB")
                    output.write(chunk)
            if total == 0:
                raise RuntimeError("SWE-bench archive download is empty")
            temporary.replace(target)
            return
        except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as exc:
            last_error = exc
            time.sleep(min(5.0, attempt + 1.0))
    temporary.unlink(missing_ok=True)
    raise RuntimeError(
        f"SWE-bench codeload archive failed for {slug}@{commit}"
    ) from last_error


def _clone_repository(
    target: Path,
    slug: str,
    *,
    full: bool,
    attempts: int = 3,
) -> None:
    last_error: RuntimeError | None = None
    for attempt in range(max(1, int(attempts))):
        try:
            if target.exists():
                _remove_repository_dir(target)
            clone_args = ["clone"]
            if not full:
                clone_args.append("--filter=blob:none")
            clone_args.extend(
                (
                    "--no-checkout",
                    f"https://github.com/swe-bench-repos/{slug}.git",
                    str(target),
                )
            )
            _run_git(*clone_args)
            return
        except RuntimeError as exc:
            last_error = exc
            time.sleep(min(5.0, attempt + 1.0))
    if target.exists():
        try:
            _remove_repository_dir(target)
        except RuntimeError:
            pass
    assert last_error is not None
    raise last_error


def _remove_repository_dir(target: Path, *, attempts: int = 10) -> None:
    last_error: OSError | None = None
    for attempt in range(max(1, int(attempts))):
        if not target.exists():
            return
        try:
            shutil.rmtree(target, onexc=_clear_readonly_and_retry)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(
        f"SWE-bench repository cache could not be removed: {target}"
    ) from last_error


def _move_repository_dir(source: Path, target: Path, *, attempts: int = 10) -> None:
    if target.exists():
        raise RuntimeError(f"SWE-bench repository move target already exists: {target}")
    last_error: OSError | None = None
    for attempt in range(max(1, int(attempts))):
        try:
            source.rename(target)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(
        f"SWE-bench repository cache could not be moved: {source} -> {target}"
    ) from last_error


def _clear_readonly_and_retry(
    operation: Callable[..., object],
    path: str,
    error: BaseException,
) -> None:
    if not isinstance(error, PermissionError):
        raise error
    os.chmod(path, os.stat(path).st_mode | stat.S_IWRITE)
    operation(path)


def _is_partial_clone(target: Path) -> bool:
    completed = subprocess.run(
        _git_command(
            "git",
            "-C",
            str(target),
            "config",
            "--bool",
            "--get",
            "remote.origin.promisor",
        ),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60.0,
        env=_git_environment(),
    )
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def _retry_git(*args: str, attempts: int = 3) -> None:
    last_error: RuntimeError | None = None
    for attempt in range(max(1, int(attempts))):
        try:
            _run_git(*args)
            return
        except RuntimeError as exc:
            last_error = exc
            time.sleep(min(5.0, attempt + 1.0))
    assert last_error is not None
    raise last_error


def _prefetch_commits(target: Path, commits: Iterable[str]) -> None:
    missing = [
        commit
        for commit in dict.fromkeys(commits)
        if not _git_succeeds(
            "-C", str(target), "cat-file", "-e", f"{commit}^{{commit}}"
        )
    ]
    for offset in range(0, len(missing), 32):
        batch = missing[offset : offset + 32]
        _retry_git(
            "-C",
            str(target),
            "fetch",
            "--no-filter",
            "--depth=1",
            "origin",
            *batch,
        )
    unresolved = [
        commit
        for commit in missing
        if not _git_succeeds(
            "-C", str(target), "cat-file", "-e", f"{commit}^{{commit}}"
        )
    ]
    if unresolved:
        raise RuntimeError(
            f"SWE-bench repository cache is missing {len(unresolved)} commits"
        )


def _run_git(*args: str) -> None:
    network_operation = "clone" in args or "fetch" in args
    try:
        completed = subprocess.run(
            _git_command("git", *args),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600.0 if network_operation else 60.0,
            env=_git_environment(),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"git {' '.join(args[:3])} timed out"
        ) from exc
    if completed.returncode:
        detail = (completed.stderr or completed.stdout)[-2000:]
        raise RuntimeError(
            f"git {' '.join(args[:3])} failed with exit "
            f"{completed.returncode}: {detail}"
        )


def _git_succeeds(*args: str) -> bool:
    try:
        completed = subprocess.run(
            _git_command("git", *args),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60.0,
            env=_git_environment(),
        )
    except subprocess.TimeoutExpired:
        return False
    return completed.returncode == 0


def _git_command(executable: str, *args: str) -> list[str]:
    return [
        executable,
        "-c",
        "http.lowSpeedLimit=1024",
        "-c",
        "http.lowSpeedTime=30",
        *args,
    ]


def _git_environment() -> dict[str, str]:
    return {**os.environ, "GIT_TERMINAL_PROMPT": "0"}


def build_retrieval_case(
    case: SWEBenchCase,
    repo_root: Path,
    *,
    chunk_chars: int,
    overlap_chars: int,
) -> RetrievalCase:
    documents: list[RetrievalDocument] = []
    source_files: set[str] = set()
    for path in sorted(repo_root.rglob("*.py")):
        relative = path.relative_to(repo_root).as_posix()
        if _is_test_path(relative) or ".git" in path.parts:
            continue
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue
        source_files.add(relative)
        for index, chunk in enumerate(
            chunk_text(text, max_chars=chunk_chars, overlap_chars=overlap_chars)
        ):
            documents.append(
                RetrievalDocument(
                    document_id=f"{relative}#chunk-{index}",
                    metric_id=relative,
                    text=f"File: {relative}\n{chunk}",
                    metadata={"repository": case.repo, "file_path": relative},
                )
            )
    gold = tuple(path for path in case.gold_files if path in source_files)
    if not gold:
        raise ValueError(
            "no gold source file exists at the SWE-bench base commit; "
            f"patch files={case.gold_files}"
        )
    return RetrievalCase(
        case_id=case.instance_id,
        case_type=case.repo,
        query=case.problem_statement,
        documents=tuple(documents),
        gold_ids=gold,
    )


def _is_test_path(path: str) -> bool:
    words = set(re.split(r"[\s_./-]+", path.lower()))
    return bool(words & {"test", "tests", "testing"})


def chunk_text(
    text: str,
    *,
    max_chars: int,
    overlap_chars: int,
) -> Iterator[str]:
    size = max(512, int(max_chars))
    overlap = max(0, min(size // 2, int(overlap_chars)))
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            newline = text.rfind("\n", start + size // 2, end)
            if newline > start:
                end = newline + 1
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)


async def run_benchmark(
    provider: CachedEmbeddingProvider,
    options: RunOptions,
) -> dict[str, Any]:
    validation = validate_dataset(options.dataset_path)
    id_filter = set(options.instance_ids)
    selected: list[SWEBenchCase] = []
    for case in iter_cases(options.dataset_path):
        if id_filter and case.instance_id not in id_filter:
            continue
        selected.append(case)
        if options.limit > 0 and len(selected) >= options.limit:
            break
    if not selected:
        raise ValueError("SWE-bench selection contains no cases")
    await provider.initialize()
    runtime = runtime_fingerprint()
    selected_ids = [case.instance_id for case in selected]
    dataset_report = dict(validation)
    dataset_report.pop("path", None)
    embedding_prewarm = {
        "batch_size": max(1, int(options.embedding_batch_size)),
        "max_concurrent": max(1, int(options.embedding_concurrency)),
    }
    manifest = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "benchmark": "SWE-bench Lite issue-to-source-file retrieval",
        "top_k": options.top_k,
        "selected_case_ids": selected_ids,
        "chunk_chars": options.chunk_chars,
        "chunk_overlap_chars": options.chunk_overlap_chars,
        "dataset_sha256": validation["sha256"],
        "embedding_fingerprint": dict(provider.fingerprint()),
        "embedding_prewarm": embedding_prewarm,
        "runtime": runtime,
    }
    output_dir = Path(options.output_dir).resolve()
    rows = prepare_result_log(output_dir, manifest, resume=options.resume)
    resumed_case_count = len(rows)
    completed_ids = {str(row["case_id"]) for row in rows}
    repo_counts: dict[str, int] = {}
    commits_by_repo: dict[str, list[str]] = {}
    for case in selected:
        repo_counts[case.repo] = repo_counts.get(case.repo, 0) + 1
        commits_by_repo.setdefault(case.repo, []).append(case.base_commit)
    repository_cache = RepositoryCache(
        options.repo_cache_dir,
        full_repositories=(repo for repo, count in repo_counts.items() if count > 1),
        commits_by_repo=commits_by_repo,
    )
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
    for case in selected:
        if case.instance_id in completed_ids:
            continue
        try:
            repo_root = repository_cache.checkout(case.repo, case.base_commit)
            retrieval_case = build_retrieval_case(
                case,
                repo_root,
                chunk_chars=options.chunk_chars,
                overlap_chars=options.chunk_overlap_chars,
            )
            row = await backend.run_case(retrieval_case)
            row["repo"] = case.repo
            row["base_commit"] = case.base_commit
            row["patch_file_count"] = len(case.gold_files)
        except Exception as exc:
            row = {
                "status": "failed",
                "case_id": case.instance_id,
                "case_type": case.repo,
                "repo": case.repo,
                "base_commit": case.base_commit,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        rows.append(row)
        append_jsonl(output_dir / "results.jsonl", row)
    order = {case_id: index for index, case_id in enumerate(selected_ids)}
    rows.sort(key=lambda row: order[str(row.get("case_id", ""))])
    write_jsonl(output_dir / "results.jsonl", rows)
    embedding_report = dict(provider.stats())
    embedding_report.pop("cache_path", None)
    summary = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "benchmark": "SWE-bench Lite issue-to-source-file retrieval",
        "top_k": options.top_k,
        "selected_case_count": len(selected),
        "selected_case_ids": selected_ids,
        "resumed_case_count": resumed_case_count,
        "chunk_chars": options.chunk_chars,
        "chunk_overlap_chars": options.chunk_overlap_chars,
        "dataset": dataset_report,
        "embedding": embedding_report,
        "embedding_prewarm": embedding_prewarm,
        "runtime": runtime,
        "results": aggregate_results(rows),
    }
    write_json(output_dir / "summary.json", summary)
    return summary
