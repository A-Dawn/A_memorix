from __future__ import annotations

from pathlib import Path
from typing import Any

import asyncio
import hashlib
import sqlite3

import pytest

from a_memorix import (
    AMemorixEngine,
    BatchIngestTextRequest,
    CreateNamespaceRequest,
    DeleteBySourceRequest,
    DeleteMemoryRequest,
    GetMemoryRequest,
    IngestTextInput,
    IngestTextRequest,
    JobStatus,
    NamespaceConfig,
    NamespaceConflictError,
    NamespaceFeatureConfig,
    NamespaceStateError,
    ProviderReference,
    RequestContext,
    UpdateNamespaceConfigRequest,
)


class ApplicationRuntime:
    def __init__(
        self,
        namespace,
        records: dict[str, dict[str, dict[str, object]]],
        calls: dict[str, int],
    ) -> None:
        self.namespace = namespace
        self._records = records.setdefault(namespace.namespace_id, {})
        self._calls = calls
        self.ready = False

    async def initialize(self) -> None:
        self.ready = True

    async def shutdown(self) -> None:
        self.ready = False

    def is_runtime_ready(self) -> bool:
        return self.ready

    def runtime_capability_status(self) -> dict[str, object]:
        return {
            "capabilities": {"metadata": True, "sparse": True, "llm": False},
            "degraded": False,
        }

    async def contains_external_memory(self, external_id: str) -> bool:
        return any(
            str(record.get("external_id", "")) == external_id
            for record in self._records.values()
        )

    async def ingest_text(self, **kwargs: Any) -> dict[str, object]:
        namespace_id = self.namespace.namespace_id
        self._calls[namespace_id] = self._calls.get(namespace_id, 0) + 1
        external_id = str(kwargs.get("external_id", "") or "")
        text = str(kwargs["text"])
        memory_id = hashlib.sha256(
            (external_id or text).encode("utf-8")
        ).hexdigest()
        if memory_id in self._records:
            return {"stored_ids": [], "skipped_ids": [memory_id], "reason": "exists"}
        source_type = str(kwargs["source_type"])
        conversation_id = str(kwargs.get("chat_id", "") or "")
        source = f"{source_type}:{conversation_id}" if conversation_id else source_type
        self._records[memory_id] = {
            "memory_id": memory_id,
            "external_id": external_id,
            "source_type": source_type,
            "source": source,
            "content": text,
            "metadata": dict(kwargs.get("metadata") or {}),
        }
        return {"stored_ids": [memory_id], "skipped_ids": []}

    async def get_memory_record(
        self,
        *,
        memory_id: str = "",
        external_id: str = "",
    ) -> dict[str, object] | None:
        if memory_id:
            return self._records.get(memory_id)
        return next(
            (
                record
                for record in self._records.values()
                if record.get("external_id") == external_id
            ),
            None,
        )

    async def memory_delete_admin(self, **kwargs: Any) -> dict[str, object]:
        memory_id = str((kwargs.get("selector") or {}).get("hash", ""))
        deleted = self._records.pop(memory_id, None)
        return {
            "operation_id": "delete-1" if deleted else "",
            "deleted_paragraph_count": int(deleted is not None),
            "error": "" if deleted else "memory not found",
        }

    async def memory_source_admin(self, **kwargs: Any) -> dict[str, object]:
        await asyncio.sleep(0)
        source = str(kwargs.get("source", ""))
        deleted_ids = [
            memory_id
            for memory_id, record in self._records.items()
            if record.get("source") == source
        ]
        for memory_id in deleted_ids:
            self._records.pop(memory_id)
        return {
            "operation_id": "source-delete-1" if deleted_ids else "",
            "deleted_count": len(deleted_ids),
            "deleted_paragraph_count": len(deleted_ids),
        }


class ApplicationRuntimeFactory:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, dict[str, object]]] = {}
        self.calls: dict[str, int] = {}
        self.namespaces: list[object] = []

    def __call__(self, namespace, _data_dir: Path) -> ApplicationRuntime:
        self.namespaces.append(namespace)
        return ApplicationRuntime(namespace, self.records, self.calls)


def _context(namespace_id: str, *, key: str | None = None) -> RequestContext:
    return RequestContext(
        namespace_id=namespace_id,
        agent_id="test-agent",
        conversation_id="conversation-1",
        idempotency_key=key,
    )


@pytest.mark.asyncio
async def test_control_schema_v2_migrates_without_rebuilding_namespaces(
    tmp_path: Path,
) -> None:
    control_root = tmp_path / "control"
    namespace_root = tmp_path / "namespaces"
    storage_key = "1" * 32
    control_root.mkdir(parents=True)
    (namespace_root / storage_key).mkdir(parents=True)
    database = control_root / "namespaces.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE namespaces (
                namespace_id TEXT PRIMARY KEY COLLATE BINARY,
                storage_key TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                last_active_at REAL,
                version INTEGER NOT NULL,
                quota_json TEXT NOT NULL,
                purge_after REAL
            );
            CREATE TABLE api_keys (
                key_id TEXT PRIMARY KEY COLLATE BINARY,
                namespace_id TEXT NOT NULL COLLATE BINARY,
                secret_hash BLOB NOT NULL UNIQUE,
                label TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL,
                revoked_at REAL,
                last_used_at REAL,
                FOREIGN KEY(namespace_id) REFERENCES namespaces(namespace_id)
                    ON DELETE CASCADE
            );
            PRAGMA user_version = 2;
            """
        )
        connection.execute(
            """
            INSERT INTO namespaces (
                namespace_id, storage_key, status, created_at, updated_at,
                last_active_at, version, quota_json, purge_after
            ) VALUES ('legacy', ?, 'inactive', 1, 1, NULL, 1, '{}', NULL)
            """,
            (storage_key,),
        )

    engine = AMemorixEngine(
        data_dir=tmp_path,
        runtime_factory=ApplicationRuntimeFactory(),
        idle_timeout_seconds=0,
    )
    await engine.initialize()
    try:
        namespace = await engine.get_namespace("legacy")
        assert namespace.config == NamespaceConfig()
        assert namespace.config_version == 1
        with sqlite3.connect(database) as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
    finally:
        await engine.shutdown()


@pytest.mark.asyncio
async def test_ingest_idempotency_survives_engine_restart(tmp_path: Path) -> None:
    factory = ApplicationRuntimeFactory()
    engine = AMemorixEngine(
        data_dir=tmp_path,
        runtime_factory=factory,
        idle_timeout_seconds=0,
    )
    await engine.initialize()
    await engine.create_namespace(CreateNamespaceRequest(namespace_id="tenant-a"))
    request = IngestTextRequest(
        context=_context("tenant-a", key="write-1"),
        external_id="external-1",
        source_type="note",
        text="durable response",
    )
    first = await engine.ingest_text(request)
    await engine.shutdown()

    await engine.initialize()
    try:
        replayed = await engine.ingest_text(
            request.model_copy(
                update={
                    "context": request.context.model_copy(
                        update={"request_id": "new-request", "trace_id": "new-trace"}
                    )
                }
            )
        )
        assert replayed == first
        assert factory.calls["tenant-a"] == 1
        with pytest.raises(NamespaceConflictError):
            await engine.ingest_text(request.model_copy(update={"text": "different"}))
    finally:
        await engine.shutdown()


@pytest.mark.asyncio
async def test_namespace_config_requires_inactive_versioned_update(tmp_path: Path) -> None:
    factory = ApplicationRuntimeFactory()
    engine = AMemorixEngine(
        data_dir=tmp_path,
        runtime_factory=factory,
        idle_timeout_seconds=0,
    )
    await engine.initialize()
    try:
        await engine.create_namespace(CreateNamespaceRequest(namespace_id="tenant-a"))
        config = NamespaceConfig(
            llm=ProviderReference(
                provider_id="openai-compatible",
                model_id="reasoning-model",
                secret_ref="secret://tenant-a/llm",
            ),
            features=NamespaceFeatureConfig(episodes=False, person_profiles=False),
        )
        request = UpdateNamespaceConfigRequest(
            namespace_id="tenant-a",
            config=config,
            expected_config_version=1,
        )
        with pytest.raises(NamespaceStateError):
            await engine.update_namespace_config(request)
        await engine.disable_namespace("tenant-a")
        updated = await engine.update_namespace_config(request)
        assert updated.config_version == 2
        assert updated.config.llm is not None
        assert updated.config.llm.secret_ref == "secret://tenant-a/llm"
        with pytest.raises(NamespaceConflictError):
            await engine.update_namespace_config(request)
        await engine.enable_namespace("tenant-a")
        capabilities = await engine.get_namespace_capabilities("tenant-a")
        assert capabilities.config_version == 2
        assert "episode" not in capabilities.search_modes
        assert factory.namespaces[-1].config == config
    finally:
        await engine.shutdown()


@pytest.mark.asyncio
async def test_batch_get_delete_and_source_job(tmp_path: Path) -> None:
    engine = AMemorixEngine(
        data_dir=tmp_path,
        runtime_factory=ApplicationRuntimeFactory(),
        idle_timeout_seconds=0,
    )
    await engine.initialize()
    try:
        await engine.create_namespace(CreateNamespaceRequest(namespace_id="tenant-a"))
        context = _context("tenant-a", key="batch-1")
        response = await engine.batch_ingest_text(
            BatchIngestTextRequest(
                context=context,
                items=(
                    IngestTextInput(
                        external_id="one",
                        source_type="document",
                        text="first",
                    ),
                    IngestTextInput(
                        external_id="two",
                        source_type="document",
                        text="second",
                    ),
                ),
            )
        )
        assert response.succeeded == 2
        assert response.failed == 0

        fetched = await engine.get_memory(
            GetMemoryRequest(context=_context("tenant-a"), external_id="one")
        )
        assert fetched.memory.content == "first"
        deleted = await engine.delete_memory(
            DeleteMemoryRequest(context=_context("tenant-a"), external_id="one")
        )
        assert deleted.deleted_memory_ids == (fetched.memory.memory_id,)

        job = await engine.submit_delete_by_source(
            DeleteBySourceRequest(
                context=_context("tenant-a"),
                source="document:conversation-1",
            )
        )
        for _ in range(100):
            job = await engine.get_job("tenant-a", job.job_id)
            if job.status not in {JobStatus.PENDING, JobStatus.RUNNING}:
                break
            await asyncio.sleep(0.01)
        assert job.status is JobStatus.SUCCEEDED
        assert job.result["deleted_memory_count"] == 1
        assert (await engine.list_jobs("tenant-a")) == [job]
    finally:
        await engine.shutdown()
