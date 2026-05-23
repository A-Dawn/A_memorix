from __future__ import annotations

import sqlite3

import pytest

from core.storage.metadata_store import MetadataStore, SCHEMA_VERSION


def _create_legacy_metadata_db(data_dir, *, version: int | None = None) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(data_dir / "metadata.db")
    cursor = conn.cursor()
    cursor.executescript(
        """
        CREATE TABLE paragraphs (
            hash TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            vector_index INTEGER,
            created_at REAL,
            updated_at REAL,
            metadata TEXT,
            source TEXT,
            word_count INTEGER
        );
        CREATE TABLE entities (
            hash TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            vector_index INTEGER,
            appearance_count INTEGER DEFAULT 1,
            created_at REAL,
            metadata TEXT
        );
        CREATE TABLE relations (
            hash TEXT PRIMARY KEY,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            vector_index INTEGER,
            confidence REAL DEFAULT 1.0,
            created_at REAL,
            source_paragraph TEXT,
            metadata TEXT
        );
        """
    )
    if version is not None:
        cursor.execute(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at REAL NOT NULL)"
        )
        cursor.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (?, 1.0)",
            (int(version),),
        )
    conn.commit()
    conn.close()


def _column_names(store: MetadataStore, table_name: str) -> set[str]:
    cursor = store._conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    return {str(row[1]) for row in cursor.fetchall()}


@pytest.mark.parametrize("legacy_version", [None, SCHEMA_VERSION - 1])
def test_metadata_store_auto_migrates_legacy_schema(tmp_path, legacy_version):
    data_dir = tmp_path / "metadata"
    _create_legacy_metadata_db(data_dir, version=legacy_version)

    store = MetadataStore(data_dir=data_dir)
    try:
        store.connect()

        assert store.get_schema_version() == SCHEMA_VERSION
        assert "knowledge_type" in _column_names(store, "paragraphs")
        assert "vector_state" in _column_names(store, "relations")
        assert store.has_table("paragraph_relations")
        assert store.has_table("paragraph_entities")
        assert store.has_table("async_tasks")
        assert store.has_table("transcript_sessions")
        assert store.has_table("person_registry")
        assert store.has_table("person_profile_switches")

        store.set_person_profile_switch("stream", "user", True)
        assert store.get_person_profile_switch("stream", "user") is True
    finally:
        store.close()


def test_metadata_store_refuses_future_schema_version(tmp_path):
    data_dir = tmp_path / "metadata"
    store = MetadataStore(data_dir=data_dir)
    store.connect()
    store.close()

    conn = sqlite3.connect(data_dir / "metadata.db")
    conn.execute(
        "INSERT INTO schema_migrations(version, applied_at) VALUES (?, 1.0)",
        (SCHEMA_VERSION + 1,),
    )
    conn.commit()
    conn.close()

    future_store = MetadataStore(data_dir=data_dir)
    with pytest.raises(RuntimeError, match="current=.*expected="):
        future_store.connect()
    future_store.close()
