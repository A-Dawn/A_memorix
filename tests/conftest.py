from __future__ import annotations

import pytest

from core.storage import GraphStore, MetadataStore, QuantizationType, SparseMatrixFormat, VectorStore


@pytest.fixture
def stores(tmp_path):
    metadata = MetadataStore(data_dir=tmp_path / "metadata")
    metadata.connect()
    graph = GraphStore(matrix_format=SparseMatrixFormat.CSR, data_dir=tmp_path / "graph")
    vector = VectorStore(
        dimension=4,
        quantization_type=QuantizationType.INT8,
        data_dir=tmp_path / "vectors",
    )
    vector.min_train_threshold = 1000
    try:
        yield metadata, graph, vector
    finally:
        metadata.close()
