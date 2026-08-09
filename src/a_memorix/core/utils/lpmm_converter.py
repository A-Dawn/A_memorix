"""Convert LPMM parquet exports into A_memorix dual-pool storage."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..storage import GraphStore, MetadataStore, QuantizationType, SparseMatrixFormat, VectorStore
from .io import atomic_write

logger = logging.getLogger("A_Memorix.LPMMConverter")


def _resolve_bounded_path(raw_path: str, root: Path, label: str) -> Path:
    candidate = Path(str(raw_path or "").strip()).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        raise ValueError(
            f"lpmm_path_outside_root: {label}必须位于导入目录: {root.resolve()}"
        ) from None
    return resolved


def _parse_fingerprint(raw_value: str, dimension: int) -> dict[str, Any]:
    try:
        value = json.loads(str(raw_value or ""))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Embedding 指纹不是有效 JSON: {exc}") from exc
    if not isinstance(value, dict) or not str(value.get("hash", "")).strip():
        raise ValueError("Embedding 指纹必须包含非空 hash")
    if int(value.get("dimension", 0) or 0) != int(dimension):
        raise ValueError("Embedding 指纹维度与转换维度不一致")
    if str(value.get("source", "")).strip().lower() != "observed":
        raise ValueError("LPMM 转换只接受已观测的 Embedding 指纹")
    return dict(value)


class LPMMConverter:
    """One-shot converter for trusted LPMM exports."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        *,
        dimension: int,
        batch_size: int,
        embedding_fingerprint: dict[str, Any],
        allow_unsafe_pickle: bool = False,
    ) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.dimension = max(1, int(dimension))
        self.batch_size = max(1, int(batch_size))
        self.embedding_fingerprint = dict(embedding_fingerprint)
        self.allow_unsafe_pickle = bool(allow_unsafe_pickle)
        self.vector_root = output_dir / "vectors"
        self.paragraph_vector_store: VectorStore | None = None
        self.graph_vector_store: VectorStore | None = None
        self.graph_store: GraphStore | None = None
        self.metadata_store: MetadataStore | None = None
        self.vector_stats = {"paragraph": 0, "entity": 0, "relation": 0}
        self.id_mapping: dict[str, str] = {}

    def _validate_input(self) -> None:
        if not self.input_dir.is_dir():
            raise ValueError(f"输入路径必须为目录: {self.input_dir}")
        required = (self.input_dir / "paragraph.parquet", self.input_dir / "entity.parquet")
        if not any(path.is_file() for path in required):
            raise ValueError("输入目录至少需要 paragraph.parquet 或 entity.parquet")

    def _initialize_stores(self) -> None:
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise RuntimeError(
                f"lpmm_output_not_empty: 输出目录必须为空，已拒绝覆盖: {self.output_dir}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.paragraph_vector_store = VectorStore(
            dimension=self.dimension,
            quantization_type=QuantizationType.INT8,
            data_dir=self.vector_root / "paragraph",
        )
        self.graph_vector_store = VectorStore(
            dimension=self.dimension,
            quantization_type=QuantizationType.INT8,
            data_dir=self.vector_root / "graph",
        )
        self.graph_store = GraphStore(
            matrix_format=SparseMatrixFormat.CSR,
            data_dir=self.output_dir / "graph",
        )
        self.metadata_store = MetadataStore(data_dir=self.output_dir / "metadata")
        self.metadata_store.connect()

    def _register_id(self, raw_id: Any, mapped_id: str, item_type: str) -> None:
        token = str(raw_id or "").strip()
        if not token:
            return
        self.id_mapping[token] = mapped_id
        prefix = f"{item_type}-"
        self.id_mapping[token.removeprefix(prefix)] = mapped_id
        self.id_mapping[f"{prefix}{token.removeprefix(prefix)}"] = mapped_id

    @staticmethod
    def _parse_relation_text(value: Any) -> tuple[str, str, str]:
        text = str(value or "").strip()
        for separator in ("|", "->"):
            parts = [part.strip() for part in text.split(separator) if part.strip()]
            if len(parts) >= 3:
                return parts[0], parts[1], parts[2]
        parts = text.split()
        return (parts[0], parts[1], " ".join(parts[2:])) if len(parts) >= 3 else ("", "", "")

    def _convert_vectors(self) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("LPMM 转换需要安装 pyarrow") from exc

        assert self.metadata_store is not None
        assert self.paragraph_vector_store is not None
        assert self.graph_vector_store is not None
        assert self.graph_store is not None

        for item_type in ("paragraph", "entity", "relation"):
            path = self.input_dir / f"{item_type}.parquet"
            if not path.is_file():
                continue
            parquet = pq.ParquetFile(path)
            if parquet.metadata.num_rows == 0:
                continue
            columns = set(parquet.schema_arrow.names)
            content_column = "str" if "str" in columns else "content" if "content" in columns else ""
            has_triple = {"subject", "predicate", "object"}.issubset(columns)
            if "embedding" not in columns:
                raise ValueError(f"{path} 缺少 embedding 列")
            if item_type != "relation" and not content_column:
                raise ValueError(f"{path} 缺少 str 或 content 列")
            if item_type == "relation" and not has_triple and not content_column:
                raise ValueError(f"{path} 缺少关系三元组或可解析文本")

            selected = ["embedding"]
            for column in ("hash", content_column, "subject", "predicate", "object"):
                if column and column in columns and column not in selected:
                    selected.append(column)

            row_index = 0
            for batch in parquet.iter_batches(batch_size=self.batch_size, columns=selected):
                vectors: list[np.ndarray] = []
                vector_ids: list[str] = []
                relation_edges: list[tuple[str, str]] = []
                relation_hashes: list[str] = []
                seen_ids: set[str] = set()
                for row in batch.to_pylist():
                    row_index += 1
                    vector = np.asarray(row["embedding"], dtype=np.float32)
                    if vector.shape != (self.dimension,):
                        raise ValueError(
                            f"lpmm_vector_dimension_mismatch: "
                            f"{item_type} 第 {row_index} 行向量维度不匹配: "
                            f"{vector.shape} vs ({self.dimension},)"
                        )
                    if not np.all(np.isfinite(vector)):
                        raise ValueError(f"{item_type} 第 {row_index} 行向量包含非有限值")

                    if item_type == "relation":
                        if has_triple:
                            subject = str(row.get("subject") or "").strip()
                            predicate = str(row.get("predicate") or "").strip()
                            obj = str(row.get("object") or "").strip()
                        else:
                            subject, predicate, obj = self._parse_relation_text(row.get(content_column))
                        if not (subject and predicate and obj):
                            raise ValueError(f"relation 第 {row_index} 行无法解析为完整三元组")
                        stored_id = self.metadata_store.add_relation(
                            subject=subject,
                            predicate=predicate,
                            obj=obj,
                            source_paragraph=None,
                        )
                        vector_id = f"relation:{stored_id}"
                    else:
                        content = str(row.get(content_column) or "").strip()
                        if not content:
                            raise ValueError(f"{item_type} 第 {row_index} 行内容为空")
                        if item_type == "paragraph":
                            stored_id = self.metadata_store.add_paragraph(
                                content=content,
                                source="lpmm_import",
                                knowledge_type="factual",
                            )
                            vector_id = stored_id
                        else:
                            stored_id = self.metadata_store.add_entity(name=content)
                            vector_id = f"entity:{stored_id}"

                    self._register_id(row.get("hash"), stored_id, item_type)
                    if vector_id in seen_ids:
                        continue
                    seen_ids.add(vector_id)
                    vectors.append(vector)
                    vector_ids.append(vector_id)
                    if item_type == "relation":
                        relation_edges.append((subject, obj))
                        relation_hashes.append(stored_id)

                if not vectors:
                    continue
                target = self.paragraph_vector_store if item_type == "paragraph" else self.graph_vector_store
                self.vector_stats[item_type] += int(target.add(np.stack(vectors), vector_ids))
                if relation_edges:
                    self.graph_store.add_edges(relation_edges, relation_hashes=relation_hashes)
                    for relation_hash in relation_hashes:
                        self.metadata_store.set_relation_vector_state(relation_hash, "ready")

        if sum(self.vector_stats.values()) <= 0:
            raise RuntimeError(
                "lpmm_no_usable_vectors: LPMM 输入没有产生任何可用向量，拒绝发布 ready 标志"
            )

    def _convert_graph(self) -> None:
        assert self.graph_store is not None
        candidates = [self.input_dir / "rag-graph.graphml", self.input_dir / "graph.graphml"]
        if self.allow_unsafe_pickle:
            candidates.append(self.input_dir / "graph_structure.pkl")
        graph_path = next((path for path in candidates if path.is_file()), None)
        if graph_path is None:
            return
        try:
            import networkx as nx
        except ImportError as exc:
            raise RuntimeError("读取 LPMM 图文件需要安装 networkx") from exc
        if graph_path.suffix == ".pkl":
            with graph_path.open("rb") as handle:
                raw_graph = pickle.load(handle)
            graph = raw_graph.graph if hasattr(raw_graph, "graph") else raw_graph
        else:
            graph = nx.read_graphml(graph_path)
        nodes = [self.id_mapping.get(str(node), str(node)) for node in graph.nodes]
        if nodes:
            self.graph_store.add_nodes(nodes)
        edges = [
            (self.id_mapping.get(str(source), str(source)), self.id_mapping.get(str(target), str(target)))
            for source, target in graph.edges
        ]
        if edges:
            self.graph_store.add_edges(edges)

    def _write_manifest(self) -> None:
        assert self.paragraph_vector_store is not None
        assert self.graph_vector_store is not None
        plural_names = {
            "paragraph": "paragraphs",
            "entity": "entities",
            "relation": "relations",
        }
        payload = {
            "status": "ready",
            "version": 1,
            "mode": "dual",
            "dimension": self.dimension,
            "created_at": time.time(),
            "paragraph_vectors": int(self.paragraph_vector_store.num_vectors),
            "graph_vectors": int(self.graph_vector_store.num_vectors),
            "stats": {
                plural_names[item_type]: {"done": int(count), "failed": 0}
                for item_type, count in self.vector_stats.items()
            },
            "migration": {
                plural_names[item_type]: {"copied": int(count), "encoded": 0, "missing": 0}
                for item_type, count in self.vector_stats.items()
            },
            "embedding_fingerprint": self.embedding_fingerprint,
            "generation_reason": "lpmm_conversion",
        }
        with atomic_write(self.vector_root / "dual_ready.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")

    def run(self) -> None:
        self._validate_input()
        self._initialize_stores()
        try:
            self._convert_vectors()
            self._convert_graph()
            assert self.paragraph_vector_store is not None
            assert self.graph_vector_store is not None
            assert self.graph_store is not None
            self.paragraph_vector_store.save(embedding_fingerprint=self.embedding_fingerprint)
            self.graph_vector_store.save(embedding_fingerprint=self.embedding_fingerprint)
            self.graph_store.save()
            assert self.metadata_store is not None
            self.metadata_store.close()
            self.metadata_store = None
            self._write_manifest()
        finally:
            if self.metadata_store is not None:
                self.metadata_store.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将 LPMM 数据转换为 A_memorix 双向量池")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--embedding-fingerprint-json", required=True)
    parser.add_argument("--allow-unsafe-pickle", action="store_true")
    parser.add_argument("--skip-relation-vector-rebuild", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args(argv)
    try:
        data_dir = Path(args.data_dir).expanduser().resolve()
        input_dir = _resolve_bounded_path(args.input, data_dir / "imports" / "source" / "lpmm", "LPMM 输入")
        output_dir = _resolve_bounded_path(args.output, data_dir / "imports" / "converted", "LPMM 输出")
        fingerprint = _parse_fingerprint(args.embedding_fingerprint_json, args.dim)
        LPMMConverter(
            input_dir,
            output_dir,
            dimension=args.dim,
            batch_size=args.batch_size,
            embedding_fingerprint=fingerprint,
            allow_unsafe_pickle=args.allow_unsafe_pickle,
        ).run()
    except Exception as exc:
        logger.error(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
