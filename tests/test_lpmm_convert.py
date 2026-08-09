from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from a_memorix.core.storage import MetadataStore, VectorStore
from a_memorix.core.utils.web_import_manager import ImportTaskManager

REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERT_SCRIPT = REPO_ROOT / "src" / "a_memorix" / "scripts" / "convert_lpmm.py"


def _fingerprint(dimension: int = 2) -> dict[str, object]:
    return {
        "version": 1,
        "hash": "sha256:lpmm-contract-test",
        "provider": "test",
        "model": "lpmm-compatible",
        "dimension": dimension,
        "dimension_request_mode": "explicit",
        "source": "observed",
    }


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _conversion_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_dir = tmp_path / "a-memorix"
    return (
        data_dir,
        data_dir / "imports" / "source" / "lpmm" / "dataset",
        data_dir / "imports" / "converted" / "dataset",
    )


def _run_convert(
    data_dir: Path,
    input_dir: Path,
    output_dir: Path,
    *,
    dimension: int = 2,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(CONVERT_SCRIPT),
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--data-dir",
            str(data_dir),
            "--dim",
            str(dimension),
            "--embedding-fingerprint-json",
            json.dumps(_fingerprint(dimension)),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _build_verify_manager(data_dir: Path) -> ImportTaskManager:
    runtime = SimpleNamespace(
        data_dir=data_dir,
        get_config=lambda key, default=None: default,
    )
    return ImportTaskManager(runtime)


def test_lpmm_converter_rejects_paths_outside_import_root(tmp_path: Path) -> None:
    data_dir, _, output_dir = _conversion_paths(tmp_path)
    outside_input = tmp_path / "outside"
    outside_input.mkdir()

    result = _run_convert(data_dir, outside_input, output_dir)

    assert result.returncode != 0
    assert "lpmm_path_outside_root" in f"{result.stdout}\n{result.stderr}"


def test_lpmm_converter_writes_loadable_dual_pools_and_refuses_overwrite(tmp_path: Path) -> None:
    data_dir, input_dir, output_dir = _conversion_paths(tmp_path)
    _write_parquet(
        input_dir / "paragraph.parquet",
        [{"hash": "lpmm-paragraph", "str": "旧版段落", "embedding": [1.0, 0.0]}],
    )
    _write_parquet(
        input_dir / "entity.parquet",
        [{"hash": "lpmm-entity", "str": "旧版实体", "embedding": [0.0, 1.0]}],
    )
    _write_parquet(
        input_dir / "relation.parquet",
        [
            {
                "hash": "lpmm-relation",
                "subject": "旧版实体",
                "predicate": "关联",
                "object": "旧版目标",
                "embedding": [0.5, 0.5],
            }
        ],
    )

    result = _run_convert(data_dir, input_dir, output_dir)

    assert result.returncode == 0, result.stderr or result.stdout
    manifest_path = output_dir / "vectors" / "dual_ready.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "ready"
    assert manifest["paragraph_vectors"] == 1
    assert manifest["graph_vectors"] == 2
    assert manifest["stats"] == {
        "paragraphs": {"done": 1, "failed": 0},
        "entities": {"done": 1, "failed": 0},
        "relations": {"done": 1, "failed": 0},
    }

    paragraph_store = VectorStore(dimension=2, data_dir=output_dir / "vectors" / "paragraph")
    graph_store = VectorStore(dimension=2, data_dir=output_dir / "vectors" / "graph")
    paragraph_store.load(expected_embedding_fingerprint=manifest["embedding_fingerprint"])
    graph_store.load(expected_embedding_fingerprint=manifest["embedding_fingerprint"])
    metadata_store = MetadataStore(data_dir=output_dir / "metadata")
    metadata_store.connect()
    try:
        paragraph_hash = str(metadata_store.query("SELECT hash FROM paragraphs")[0]["hash"])
        entity_hash = str(metadata_store.query("SELECT hash FROM entities")[0]["hash"])
        relation_hash = str(metadata_store.query("SELECT hash FROM relations")[0]["hash"])
    finally:
        metadata_store.close()
    assert paragraph_hash in paragraph_store
    assert f"entity:{entity_hash}" in graph_store
    assert f"relation:{relation_hash}" in graph_store
    assert _build_verify_manager(data_dir)._verify_convert_output(output_dir)["ok"] is True

    committed_manifest = manifest_path.read_bytes()
    second = _run_convert(data_dir, input_dir, output_dir)
    assert second.returncode != 0
    assert "lpmm_output_not_empty" in second.stderr
    assert manifest_path.read_bytes() == committed_manifest


def test_lpmm_converter_deduplicates_semantic_ids_within_batch(tmp_path: Path) -> None:
    data_dir, input_dir, output_dir = _conversion_paths(tmp_path)
    _write_parquet(
        input_dir / "paragraph.parquet",
        [
            {"hash": "paragraph-a", "str": "重复段落", "embedding": [1.0, 0.0]},
            {"hash": "paragraph-b", "str": "重复段落", "embedding": [1.0, 0.0]},
        ],
    )
    _write_parquet(
        input_dir / "entity.parquet",
        [
            {"hash": "entity-a", "str": "重复实体", "embedding": [0.0, 1.0]},
            {"hash": "entity-b", "str": "重复实体", "embedding": [0.0, 1.0]},
        ],
    )
    relation = {
        "subject": "重复实体",
        "predicate": "关联",
        "object": "重复目标",
        "embedding": [0.5, 0.5],
    }
    _write_parquet(
        input_dir / "relation.parquet",
        [{"hash": "relation-a", **relation}, {"hash": "relation-b", **relation}],
    )

    result = _run_convert(data_dir, input_dir, output_dir)

    assert result.returncode == 0, result.stderr or result.stdout
    manifest = json.loads((output_dir / "vectors" / "dual_ready.json").read_text(encoding="utf-8"))
    assert manifest["stats"] == {
        "paragraphs": {"done": 1, "failed": 0},
        "entities": {"done": 1, "failed": 0},
        "relations": {"done": 1, "failed": 0},
    }


@pytest.mark.parametrize(
    ("rows", "expected_error"),
    [
        (
            [{"hash": "bad", "str": "错误维度", "embedding": [1.0, 0.0, 0.5]}],
            "lpmm_vector_dimension_mismatch",
        ),
        ([], "lpmm_no_usable_vectors"),
    ],
)
def test_lpmm_converter_failure_does_not_publish_ready_manifest(
    tmp_path: Path,
    rows: list[dict[str, object]],
    expected_error: str,
) -> None:
    data_dir, input_dir, output_dir = _conversion_paths(tmp_path)
    _write_parquet(input_dir / "paragraph.parquet", rows)

    result = _run_convert(data_dir, input_dir, output_dir)

    assert result.returncode != 0
    assert expected_error in result.stderr
    assert not (output_dir / "vectors" / "dual_ready.json").exists()


def test_lpmm_output_verification_rejects_metadata_without_graph_vector(tmp_path: Path) -> None:
    data_dir, input_dir, output_dir = _conversion_paths(tmp_path)
    _write_parquet(
        input_dir / "paragraph.parquet",
        [{"hash": "lpmm-paragraph", "str": "可用段落", "embedding": [1.0, 0.0]}],
    )
    result = _run_convert(data_dir, input_dir, output_dir)
    assert result.returncode == 0, result.stderr or result.stdout

    metadata_store = MetadataStore(data_dir=output_dir / "metadata")
    metadata_store.connect()
    metadata_store.add_entity(name="缺少向量的实体")
    metadata_store.close()

    verify = _build_verify_manager(data_dir)._verify_convert_output(output_dir)
    assert verify["stores_opened"] is True
    assert verify["references_valid"] is False
    assert verify["ok"] is False


@pytest.mark.asyncio
async def test_web_lpmm_convert_rejects_nonempty_target_before_queueing(tmp_path: Path) -> None:
    data_dir = tmp_path / "runtime"
    source_dir = data_dir / "imports" / "source" / "lpmm" / "dataset"
    source_dir.mkdir(parents=True)
    target_dir = data_dir / "imports" / "converted" / "dataset"
    target_dir.mkdir(parents=True)
    (target_dir / "existing.txt").write_text("keep", encoding="utf-8")
    runtime = SimpleNamespace(
        data_dir=data_dir,
        get_config=lambda key, default=None: default,
    )
    manager = ImportTaskManager(runtime)

    with pytest.raises(ValueError, match="目标目录必须为空"):
        await manager.create_lpmm_convert_task(
            {
                "alias": "lpmm",
                "relative_path": "dataset",
                "target_alias": "converted",
                "target_relative_path": "dataset",
                "dimension": 2,
            }
        )

    assert manager._tasks == {}
    assert (target_dir / "existing.txt").read_text(encoding="utf-8") == "keep"
