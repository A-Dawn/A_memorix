from __future__ import annotations

from pathlib import Path
from typing import Sequence

import asyncio
import sqlite3

import pytest
from pydantic import ValidationError

from a_memorix import (
    AMemorixEngine,
    CreateNamespaceRequest,
    DeleteMemoryRequest,
    GetMemoryRequest,
    IngestTextRequest,
    InvalidArgumentError,
    MigrationRequiredError,
    NamespaceCapacityError,
    NamespaceHostPorts,
    NamespaceNotFoundError,
    NamespaceQuota,
    NamespaceRuntimeError,
    NamespaceStatus,
    RequestContext,
    SDKMemoryKernel,
)


class MutableClock:
    def __init__(self, value: float = 1_800_000_000.0) -> None:
        self.value = value

    def time(self) -> float:
        return self.value

    def monotonic(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeRuntime:
    def __init__(
        self,
        data_dir: Path,
        *,
        fail_initialize: bool = False,
        fail_shutdown_once: bool = False,
        delay: float = 0.0,
    ) -> None:
        self.data_dir = data_dir
        self.fail_initialize = fail_initialize
        self.fail_shutdown_once = fail_shutdown_once
        self.delay = delay
        self.initialize_calls = 0
        self.shutdown_calls = 0
        self.ready = False

    async def initialize(self) -> None:
        self.initialize_calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail_initialize:
            raise RuntimeError("injected runtime failure")
        self.ready = True
        (self.data_dir / "runtime.marker").write_text("ready", encoding="utf-8")

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.fail_shutdown_once and self.shutdown_calls == 1:
            raise RuntimeError("injected shutdown failure")
        self.ready = False

    def is_runtime_ready(self) -> bool:
        return self.ready


class FakeRuntimeFactory:
    def __init__(
        self,
        *,
        failing: set[str] | None = None,
        fail_shutdown_once: set[str] | None = None,
        delay: float = 0.0,
    ) -> None:
        self.failing = failing or set()
        self.fail_shutdown_once = fail_shutdown_once or set()
        self.delay = delay
        self.calls: dict[str, int] = {}
        self.runtimes: dict[str, list[FakeRuntime]] = {}

    def __call__(self, namespace, data_dir: Path) -> FakeRuntime:
        namespace_id = namespace.namespace_id
        self.calls[namespace_id] = self.calls.get(namespace_id, 0) + 1
        runtime = FakeRuntime(
            data_dir,
            fail_initialize=namespace_id in self.failing,
            fail_shutdown_once=namespace_id in self.fail_shutdown_once,
            delay=self.delay,
        )
        self.runtimes.setdefault(namespace_id, []).append(runtime)
        return runtime


class DeterministicEmbeddingProvider:
    async def embed(
        self,
        texts: Sequence[str],
        *,
        dimensions: int | None = None,
    ) -> Sequence[Sequence[float]]:
        dimension = int(dimensions or 8)
        return [
            [float((sum(text.encode("utf-8")) + index) % 31 + 1) for index in range(dimension)]
            for text in texts
        ]

    def fingerprint(self) -> dict[str, object]:
        return {"provider": "test", "model": "namespace-isolation"}


def _request(namespace_id: str) -> CreateNamespaceRequest:
    return CreateNamespaceRequest(namespace_id=namespace_id)


def _context(namespace_id: str) -> RequestContext:
    return RequestContext(
        namespace_id=namespace_id,
        agent_id="same-agent",
        user_id="same-user",
        conversation_id="same-conversation",
    )


def test_namespace_id_is_a_lowercase_transport_safe_identifier() -> None:
    assert RequestContext(namespace_id="agent-1.prod").namespace_id == "agent-1.prod"
    for invalid in ("Agent-1", "agent/1", "../agent", "agent 1", ""):
        with pytest.raises(ValidationError):
            RequestContext(namespace_id=invalid)


@pytest.mark.asyncio
async def test_namespace_admin_uses_the_same_identifier_validation(tmp_path: Path) -> None:
    engine = AMemorixEngine(
        data_dir=tmp_path,
        runtime_factory=FakeRuntimeFactory(),
        idle_timeout_seconds=0,
    )
    await engine.initialize()
    try:
        with pytest.raises(InvalidArgumentError):
            await engine.get_namespace("Invalid/ID")
    finally:
        await engine.shutdown()


@pytest.mark.asyncio
async def test_newer_control_schema_requires_an_explicit_migration(tmp_path: Path) -> None:
    control_root = tmp_path / "control"
    control_root.mkdir()
    with sqlite3.connect(control_root / "namespaces.db") as connection:
        connection.execute("PRAGMA user_version = 99")

    engine = AMemorixEngine(
        data_dir=tmp_path,
        runtime_factory=FakeRuntimeFactory(),
        idle_timeout_seconds=0,
    )
    with pytest.raises(MigrationRequiredError):
        await engine.initialize()
    with sqlite3.connect(control_root / "namespaces.db") as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 99


@pytest.mark.asyncio
async def test_namespace_delete_quarantine_restore_and_delayed_purge(tmp_path: Path) -> None:
    clock = MutableClock()
    factory = FakeRuntimeFactory()
    engine = AMemorixEngine(
        data_dir=tmp_path,
        runtime_factory=factory,
        clock=clock,
        idle_timeout_seconds=0,
    )
    await engine.initialize()
    try:
        created = await engine.create_namespace(_request("tenant-a"))
        assert created.status is NamespaceStatus.ACTIVE
        async with engine.runtime(_context("tenant-a")) as runtime:
            assert runtime.is_runtime_ready()

        with sqlite3.connect(tmp_path / "control" / "namespaces.db") as connection:
            storage_key = connection.execute(
                "SELECT storage_key FROM namespaces WHERE namespace_id = ?",
                ("tenant-a",),
            ).fetchone()[0]
        assert storage_key != "tenant-a"
        assert (tmp_path / "namespaces" / storage_key / "runtime.marker").is_file()

        deleted = await engine.delete_namespace("tenant-a")
        assert deleted.status is NamespaceStatus.QUARANTINED
        assert not (tmp_path / "namespaces" / storage_key).exists()
        assert (tmp_path / "quarantine" / storage_key / "runtime.marker").is_file()

        restored = await engine.restore_namespace("tenant-a")
        assert restored.status is NamespaceStatus.ACTIVE
        assert (tmp_path / "namespaces" / storage_key / "runtime.marker").is_file()

        await engine.delete_namespace("tenant-a")
        clock.advance(7 * 24 * 60 * 60 - 1)
        assert await engine.purge_expired_namespaces() == []
        clock.advance(1)
        assert await engine.purge_expired_namespaces() == ["tenant-a"]
        assert not (tmp_path / "quarantine" / storage_key).exists()
        with pytest.raises(NamespaceNotFoundError):
            await engine.get_namespace("tenant-a")
    finally:
        await engine.shutdown()


@pytest.mark.asyncio
async def test_control_plane_recovers_interrupted_directory_moves(tmp_path: Path) -> None:
    clock = MutableClock()
    factory = FakeRuntimeFactory()
    engine = AMemorixEngine(
        data_dir=tmp_path,
        runtime_factory=factory,
        clock=clock,
        idle_timeout_seconds=0,
    )
    await engine.initialize()
    await engine.create_namespace(_request("recoverable"))
    await engine.shutdown()

    db_path = tmp_path / "control" / "namespaces.db"
    with sqlite3.connect(db_path) as connection:
        storage_key = connection.execute(
            "SELECT storage_key FROM namespaces WHERE namespace_id = 'recoverable'"
        ).fetchone()[0]
        connection.execute(
            "UPDATE namespaces SET status = 'quarantined', purge_after = ? WHERE namespace_id = 'recoverable'",
            (clock.time() + 7 * 24 * 60 * 60,),
        )
    assert (tmp_path / "namespaces" / storage_key).is_dir()

    await engine.initialize()
    assert not (tmp_path / "namespaces" / storage_key).exists()
    assert (tmp_path / "quarantine" / storage_key).is_dir()
    await engine.shutdown()

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE namespaces SET status = 'active', purge_after = NULL WHERE namespace_id = 'recoverable'"
        )
    await engine.initialize()
    try:
        assert (tmp_path / "namespaces" / storage_key).is_dir()
        assert not (tmp_path / "quarantine" / storage_key).exists()
    finally:
        await engine.shutdown()


@pytest.mark.asyncio
async def test_concurrent_initialization_is_deduplicated_and_lru_closes_idle_runtime(
    tmp_path: Path,
) -> None:
    factory = FakeRuntimeFactory(delay=0.02)
    engine = AMemorixEngine(
        data_dir=tmp_path,
        runtime_factory=factory,
        max_active_namespaces=1,
        idle_timeout_seconds=0,
    )
    await engine.initialize()
    try:
        await engine.create_namespace(_request("alpha"))
        await engine.create_namespace(_request("beta"))

        async def use_alpha() -> int:
            async with engine.runtime(_context("alpha")) as runtime:
                await asyncio.sleep(0)
                return id(runtime)

        runtime_ids = await asyncio.gather(*(use_alpha() for _ in range(8)))
        assert len(set(runtime_ids)) == 1
        assert factory.calls["alpha"] == 1

        async with engine.runtime(_context("beta")):
            pass
        assert factory.runtimes["alpha"][0].shutdown_calls == 1
        assert (await engine.namespace_health("alpha")).runtime_state.value == "closed"
    finally:
        await engine.shutdown()


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_leave_namespace_loading(tmp_path: Path) -> None:
    factory = FakeRuntimeFactory(delay=0.03)
    engine = AMemorixEngine(
        data_dir=tmp_path,
        runtime_factory=factory,
        idle_timeout_seconds=0,
    )
    await engine.initialize()
    try:
        await engine.create_namespace(_request("alpha"))

        async def wait_for_runtime() -> None:
            async with engine.runtime(_context("alpha")):
                pytest.fail("cancelled waiter must not enter the runtime")

        waiter = asyncio.create_task(wait_for_runtime())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        await asyncio.sleep(0.05)

        health = await engine.namespace_health("alpha")
        assert health.runtime_state.value == "ready"
        assert health.resource_usage.active_requests == 0
        assert factory.calls["alpha"] == 1
    finally:
        await engine.shutdown()


@pytest.mark.asyncio
async def test_busy_runtime_is_not_evicted_or_closed(tmp_path: Path) -> None:
    factory = FakeRuntimeFactory()
    engine = AMemorixEngine(
        data_dir=tmp_path,
        runtime_factory=factory,
        max_active_namespaces=1,
        idle_timeout_seconds=0,
    )
    await engine.initialize()
    try:
        await engine.create_namespace(_request("alpha"))
        await engine.create_namespace(_request("beta"))
        async with engine.runtime(_context("alpha")):
            with pytest.raises(NamespaceCapacityError):
                async with engine.runtime(_context("beta")):
                    pytest.fail("busy runtime capacity should reject a second namespace")

            disable_task = asyncio.create_task(engine.disable_namespace("alpha"))
            await asyncio.sleep(0)
            assert not disable_task.done()
        disabled = await disable_task
        assert disabled.status is NamespaceStatus.INACTIVE
        assert factory.runtimes["alpha"][0].shutdown_calls == 1
    finally:
        await engine.shutdown()


@pytest.mark.asyncio
async def test_namespace_quota_limits_concurrent_requests_and_reports_storage(
    tmp_path: Path,
) -> None:
    engine = AMemorixEngine(
        data_dir=tmp_path,
        runtime_factory=FakeRuntimeFactory(),
        idle_timeout_seconds=0,
    )
    await engine.initialize()
    try:
        await engine.create_namespace(
            CreateNamespaceRequest(
                namespace_id="limited",
                quota=NamespaceQuota(
                    max_concurrent_requests=1,
                    max_storage_bytes=1,
                ),
            )
        )
        async with engine.runtime(_context("limited")):
            with pytest.raises(NamespaceCapacityError):
                async with engine.runtime(_context("limited")):
                    pytest.fail("namespace request quota should reject a second lease")
            health = await engine.namespace_health("limited")
            assert health.resource_usage.active_requests == 1
            assert health.resource_usage.storage_bytes > 1
            assert not health.healthy
            assert "storage quota exceeded" in str(health.last_error)
    finally:
        await engine.shutdown()


@pytest.mark.asyncio
async def test_runtime_initialization_failure_is_isolated_by_namespace(tmp_path: Path) -> None:
    factory = FakeRuntimeFactory(
        failing={"broken"},
        fail_shutdown_once={"broken"},
    )
    engine = AMemorixEngine(
        data_dir=tmp_path,
        runtime_factory=factory,
        idle_timeout_seconds=0,
    )
    await engine.initialize()
    try:
        await engine.create_namespace(_request("broken"))
        await engine.create_namespace(_request("healthy"))
        with pytest.raises(NamespaceRuntimeError):
            async with engine.runtime(_context("broken")):
                pytest.fail("broken runtime should not be leased")
        async with engine.runtime(_context("healthy")) as runtime:
            assert runtime.is_runtime_ready()
        broken_health = await engine.namespace_health("broken")
        healthy_health = await engine.namespace_health("healthy")
        assert broken_health.runtime_state.value == "failed"
        assert not broken_health.healthy
        assert healthy_health.runtime_state.value == "ready"
        assert healthy_health.healthy
    finally:
        await engine.shutdown()
    assert factory.runtimes["broken"][0].shutdown_calls == 2


@pytest.mark.asyncio
async def test_engine_preserves_runtime_references_when_shutdown_must_be_retried(
    tmp_path: Path,
) -> None:
    factory = FakeRuntimeFactory(fail_shutdown_once={"alpha"})
    engine = AMemorixEngine(
        data_dir=tmp_path,
        runtime_factory=factory,
        idle_timeout_seconds=0,
    )
    await engine.initialize()
    await engine.create_namespace(_request("alpha"))
    async with engine.runtime(_context("alpha")):
        pass

    with pytest.raises(NamespaceRuntimeError):
        await engine.shutdown()
    assert (await engine.get_namespace("alpha")).status is NamespaceStatus.ACTIVE

    await engine.shutdown()
    assert factory.runtimes["alpha"][0].shutdown_calls == 2


@pytest.mark.asyncio
async def test_real_kernels_isolate_identical_external_and_user_ids(tmp_path: Path) -> None:
    def host_ports(_namespace) -> NamespaceHostPorts:
        return NamespaceHostPorts(embedding_provider=DeterministicEmbeddingProvider())

    engine = AMemorixEngine(
        data_dir=tmp_path,
        host_port_factory=host_ports,
        config_factory=lambda _namespace: {
            "embedding": {"dimension": 8, "dimension_request_mode": "always"},
            "retrieval": {"sparse": {"enabled": False}},
        },
        max_active_namespaces=2,
        idle_timeout_seconds=0,
    )
    await engine.initialize()
    try:
        await engine.create_namespace(_request("tenant-a"))
        await engine.create_namespace(_request("tenant-b"))
        results = []
        for namespace_id in ("tenant-a", "tenant-b"):
            async with engine.runtime(_context(namespace_id)) as runtime:
                assert isinstance(runtime, SDKMemoryKernel)
                result = await runtime.ingest_text(
                    external_id="document:same",
                    source_type="document",
                    text="The same text belongs to two independent namespaces.",
                    user_id="same-user",
                )
                results.append(result)
                assert runtime.metadata_store is not None
                assert runtime.metadata_store.get_external_memory_ref("document:same") is not None
        assert all(result["stored_ids"] for result in results)

        async with engine.runtime(_context("tenant-a")) as first:
            await first.ingest_text(
                external_id="document:first-only",
                source_type="document",
                text="Only tenant A stores this text.",
                user_id="same-user",
            )
            assert first.metadata_store is not None
            assert first.metadata_store.get_external_memory_ref("document:first-only") is not None
        async with engine.runtime(_context("tenant-b")) as second:
            assert second.metadata_store is not None
            assert second.metadata_store.get_external_memory_ref("document:first-only") is None

        application_context = _context("tenant-a")
        ingested = await engine.ingest_text(
            IngestTextRequest(
                context=application_context,
                external_id="document:managed",
                source_type="document",
                text="Managed through the generic application API.",
            )
        )
        fetched = await engine.get_memory(
            GetMemoryRequest(
                context=application_context,
                external_id="document:managed",
            )
        )
        assert fetched.memory.memory_id == ingested.stored_ids[0]
        deleted = await engine.delete_memory(
            DeleteMemoryRequest(
                context=application_context,
                external_id="document:managed",
            )
        )
        assert deleted.deleted_memory_ids == (fetched.memory.memory_id,)
    finally:
        await engine.shutdown()
