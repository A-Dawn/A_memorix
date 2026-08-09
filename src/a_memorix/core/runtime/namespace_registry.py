"""Concurrent lifecycle management for namespace runtimes."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import asyncio

from a_memorix.contracts import (
    AMemorixError,
    CreateNamespaceRequest,
    NamespaceCapacityError,
    NamespaceConflictError,
    NamespaceHealth,
    NamespaceInfo,
    NamespaceResourceUsage,
    NamespaceRuntimeError,
    NamespaceRuntimeState,
    NamespaceStateError,
    NamespaceStatus,
    RequestContext,
    UpdateNamespaceConfigRequest,
)
from a_memorix.ports import Clock, SystemClock
from a_memorix.contracts.context import validate_namespace_id

from .namespace_control import NamespaceControlStore
from .namespace_runtime import NamespaceRuntime, NamespaceRuntimeFactory
from .namespace_storage import NamespaceStorageLayout


@dataclass
class _RuntimeEntry:
    runtime: NamespaceRuntime
    namespace: NamespaceInfo
    last_used: float
    active_requests: int = 0
    accepting_requests: bool = True
    zero_requests: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        self.zero_requests.set()


class NamespaceRuntimeRegistry:
    """Own namespace state transitions and lazily loaded runtimes."""

    def __init__(
        self,
        *,
        control_store: NamespaceControlStore,
        storage_layout: NamespaceStorageLayout,
        runtime_factory: NamespaceRuntimeFactory,
        max_active_namespaces: int = 8,
        max_concurrent_requests_per_namespace: int = 64,
        idle_timeout_seconds: float = 900.0,
        quarantine_retention_seconds: float = 7 * 24 * 60 * 60,
        clock: Clock | None = None,
    ) -> None:
        if max_active_namespaces < 1:
            raise ValueError("max_active_namespaces must be at least 1")
        if max_concurrent_requests_per_namespace < 1:
            raise ValueError("max_concurrent_requests_per_namespace must be at least 1")
        if idle_timeout_seconds < 0:
            raise ValueError("idle_timeout_seconds cannot be negative")
        if quarantine_retention_seconds < 0:
            raise ValueError("quarantine_retention_seconds cannot be negative")
        self._store = control_store
        self._layout = storage_layout
        self._runtime_factory = runtime_factory
        self._max_active_namespaces = max_active_namespaces
        self._max_concurrent_requests = max_concurrent_requests_per_namespace
        self._idle_timeout_seconds = idle_timeout_seconds
        self._quarantine_retention_seconds = quarantine_retention_seconds
        self._clock = clock or SystemClock()
        self._entries: dict[str, _RuntimeEntry] = {}
        self._initializing: dict[str, asyncio.Task[None]] = {}
        self._failures: dict[str, str] = {}
        self._namespace_locks: dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()
        self._capacity_lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()
        self._reaper_task: asyncio.Task[None] | None = None
        self._started = False
        self._shutting_down = False
        self._closed = False

    async def start(self) -> None:
        if self._started:
            if self._closed:
                raise RuntimeError(
                    "namespace registry cannot be restarted after shutdown"
                )
            return
        self._layout.initialize()
        self._started = True
        records = self._store.list_records()
        self._layout.recover_pending_restores(records)
        for record in records:
            try:
                self._layout.reconcile(record)
                if record.info.status is NamespaceStatus.CREATING:
                    self._store.transition(
                        record.info.namespace_id,
                        expected={NamespaceStatus.CREATING},
                        target=NamespaceStatus.ACTIVE,
                        now=self._clock.time(),
                    )
                elif record.info.status is NamespaceStatus.PURGING:
                    self._store.remove_purging(record.info.namespace_id)
            except BaseException as exc:
                self._failures[record.info.namespace_id] = (
                    f"control-plane recovery failed: {exc}"
                )
        if self._idle_timeout_seconds > 0:
            self._reaper_task = asyncio.create_task(
                self._idle_reaper(),
                name="a-memorix-namespace-lru-reaper",
            )

    async def shutdown(self) -> None:
        async with self._shutdown_lock:
            if not self._started or self._closed:
                return
            self._shutting_down = True
            if self._reaper_task is not None:
                self._reaper_task.cancel()
                await asyncio.gather(self._reaper_task, return_exceptions=True)
                self._reaper_task = None
            async with self._registry_lock:
                initialization_tasks = list(self._initializing.values())
            if initialization_tasks:
                await asyncio.gather(*initialization_tasks, return_exceptions=True)
            errors: list[BaseException] = []
            async with self._capacity_lock:
                async with self._registry_lock:
                    namespace_ids = list(self._entries)
                    for entry in self._entries.values():
                        entry.accepting_requests = False
                for namespace_id in namespace_ids:
                    entry = self._entries.get(namespace_id)
                    if entry is None:
                        continue
                    try:
                        await self._shutdown_entry(namespace_id, entry)
                    except BaseException as exc:
                        errors.append(exc)
            if errors:
                self._shutting_down = False
                if self._idle_timeout_seconds > 0:
                    self._reaper_task = asyncio.create_task(
                        self._idle_reaper(),
                        name="a-memorix-namespace-lru-reaper",
                    )
                raise NamespaceRuntimeError(
                    "one or more namespace runtimes failed to shut down"
                ) from errors[0]
            self._closed = True
            self._shutting_down = False

    async def create_namespace(self, request: CreateNamespaceRequest) -> NamespaceInfo:
        self._require_running()
        async with self._namespace_lock(request.namespace_id):
            storage_key = uuid4().hex
            record = self._store.create(
                request, storage_key=storage_key, now=self._clock.time()
            )
            created_directory = False
            try:
                self._layout.create_active(record.storage_key)
                created_directory = True
                record = self._store.transition(
                    request.namespace_id,
                    expected={NamespaceStatus.CREATING},
                    target=NamespaceStatus.ACTIVE,
                    now=self._clock.time(),
                )
            except BaseException:
                try:
                    self._store.transition(
                        request.namespace_id,
                        expected={NamespaceStatus.CREATING},
                        target=NamespaceStatus.PURGING,
                        now=self._clock.time(),
                    )
                    if created_directory:
                        self._layout.purge(storage_key)
                    self._store.remove_purging(request.namespace_id)
                except BaseException as cleanup_error:
                    self._failures[request.namespace_id] = (
                        f"namespace creation cleanup failed: {cleanup_error}"
                    )
                raise
            self._failures.pop(request.namespace_id, None)
            return record.info

    @asynccontextmanager
    async def inactive_storage(
        self,
        namespace_id: str,
    ) -> AsyncIterator[tuple[NamespaceInfo, Path]]:
        self._require_running()
        namespace_id = validate_namespace_id(namespace_id)
        async with self._namespace_lock(namespace_id):
            record = self._store.get_record(namespace_id)
            if record.info.status is not NamespaceStatus.INACTIVE:
                raise NamespaceStateError(
                    "namespace backup requires an inactive namespace",
                    details={
                        "namespace_id": namespace_id,
                        "status": record.info.status.value,
                    },
                )
            async with self._registry_lock:
                if namespace_id in self._entries:
                    raise NamespaceRuntimeError(
                        f"namespace runtime has not fully shut down: {namespace_id}"
                    )
            yield record.info, self._layout.validate_active(record.storage_key)

    async def restore_inactive_namespace(
        self,
        request: CreateNamespaceRequest,
        *,
        populate: Callable[[Path], None],
    ) -> NamespaceInfo:
        self._require_running()
        namespace_id = validate_namespace_id(request.namespace_id)
        async with self._namespace_lock(namespace_id):
            if self._store.namespace_exists(namespace_id):
                raise NamespaceConflictError(
                    f"namespace already exists: {namespace_id}",
                    details={"namespace_id": namespace_id},
                )
            storage_key = uuid4().hex
            staging_created = False
            committed = False
            try:
                staging_path = self._layout.create_restore_staging(storage_key)
                staging_created = True
                await asyncio.to_thread(populate, staging_path)
                self._layout.mark_restore_pending(storage_key, namespace_id)
                self._layout.commit_restore_staging(storage_key)
                staging_created = False
                committed = True
                record = self._store.create_restored(
                    request,
                    storage_key=storage_key,
                    now=self._clock.time(),
                )
            except BaseException:
                if staging_created:
                    self._layout.discard_restore_staging(storage_key)
                if committed:
                    self._layout.purge(storage_key)
                self._layout.clear_restore_pending(storage_key)
                raise
            self._failures.pop(namespace_id, None)
            try:
                self._layout.clear_restore_pending(storage_key)
            except OSError as exc:
                self._failures[namespace_id] = (
                    f"namespace restore marker cleanup failed: {exc}"
                )
            return record.info

    def get_namespace(self, namespace_id: str) -> NamespaceInfo:
        self._require_running()
        namespace_id = validate_namespace_id(namespace_id)
        return self._store.get_record(namespace_id).info

    def list_namespaces(self) -> list[NamespaceInfo]:
        self._require_running()
        return [record.info for record in self._store.list_records()]

    async def update_namespace_config(
        self,
        request: UpdateNamespaceConfigRequest,
    ) -> NamespaceInfo:
        self._require_running()
        namespace_id = validate_namespace_id(request.namespace_id)
        async with self._namespace_lock(namespace_id):
            record = self._store.update_config(
                namespace_id,
                config=request.config,
                expected_config_version=request.expected_config_version,
                now=self._clock.time(),
            )
            self._failures.pop(namespace_id, None)
            return record.info

    async def disable_namespace(self, namespace_id: str) -> NamespaceInfo:
        self._require_running()
        namespace_id = validate_namespace_id(namespace_id)
        async with self._namespace_lock(namespace_id):
            entry = await self._block_new_requests(namespace_id)
            try:
                record = self._store.transition(
                    namespace_id,
                    expected={NamespaceStatus.ACTIVE},
                    target=NamespaceStatus.INACTIVE,
                    now=self._clock.time(),
                )
            except BaseException:
                await self._resume_requests(namespace_id, entry)
                raise
            if entry is not None:
                await self._shutdown_entry(namespace_id, entry)
            return record.info

    async def enable_namespace(self, namespace_id: str) -> NamespaceInfo:
        self._require_running()
        namespace_id = validate_namespace_id(namespace_id)
        async with self._namespace_lock(namespace_id):
            async with self._registry_lock:
                if namespace_id in self._entries:
                    raise NamespaceRuntimeError(
                        f"namespace runtime has not fully shut down: {namespace_id}"
                    )
            record = self._store.get_record(namespace_id)
            self._layout.validate_active(record.storage_key)
            enabled = self._store.transition(
                namespace_id,
                expected={NamespaceStatus.INACTIVE},
                target=NamespaceStatus.ACTIVE,
                now=self._clock.time(),
            )
            self._failures.pop(namespace_id, None)
            return enabled.info

    async def delete_namespace(self, namespace_id: str) -> NamespaceInfo:
        self._require_running()
        namespace_id = validate_namespace_id(namespace_id)
        async with self._namespace_lock(namespace_id):
            record = self._store.get_record(namespace_id)
            if record.info.status not in {
                NamespaceStatus.ACTIVE,
                NamespaceStatus.INACTIVE,
                NamespaceStatus.QUARANTINED,
            }:
                raise NamespaceStateError(
                    f"cannot delete namespace in state {record.info.status.value}: {namespace_id}"
                )
            entry = await self._block_new_requests(namespace_id)
            if record.info.status is not NamespaceStatus.QUARANTINED:
                try:
                    record = self._store.transition(
                        namespace_id,
                        expected={NamespaceStatus.ACTIVE, NamespaceStatus.INACTIVE},
                        target=NamespaceStatus.QUARANTINED,
                        now=self._clock.time(),
                        purge_after=self._clock.time()
                        + self._quarantine_retention_seconds,
                    )
                except BaseException:
                    await self._resume_requests(namespace_id, entry)
                    raise
            if entry is not None:
                await self._shutdown_entry(namespace_id, entry)
            self._layout.move_to_quarantine(record.storage_key)
            self._failures.pop(namespace_id, None)
            return self._store.get_record(namespace_id).info

    async def restore_namespace(self, namespace_id: str) -> NamespaceInfo:
        self._require_running()
        namespace_id = validate_namespace_id(namespace_id)
        async with self._namespace_lock(namespace_id):
            record = self._store.get_record(namespace_id)
            purge_after = record.info.purge_after
            restored = self._store.transition(
                namespace_id,
                expected={NamespaceStatus.QUARANTINED},
                target=NamespaceStatus.ACTIVE,
                now=self._clock.time(),
            )
            try:
                self._layout.restore_from_quarantine(record.storage_key)
            except BaseException:
                self._store.transition(
                    namespace_id,
                    expected={NamespaceStatus.ACTIVE},
                    target=NamespaceStatus.QUARANTINED,
                    now=self._clock.time(),
                    purge_after=purge_after.timestamp()
                    if purge_after is not None
                    else None,
                )
                raise
            self._failures.pop(namespace_id, None)
            return restored.info

    async def purge_namespace(self, namespace_id: str, *, force: bool = True) -> bool:
        self._require_running()
        namespace_id = validate_namespace_id(namespace_id)
        async with self._namespace_lock(namespace_id):
            record = self._store.get_record(namespace_id)
            if record.info.status is NamespaceStatus.QUARANTINED:
                purge_after = record.info.purge_after
                if (
                    not force
                    and purge_after is not None
                    and purge_after.timestamp() > self._clock.time()
                ):
                    return False
                record = self._store.transition(
                    namespace_id,
                    expected={NamespaceStatus.QUARANTINED},
                    target=NamespaceStatus.PURGING,
                    now=self._clock.time(),
                )
            elif record.info.status is not NamespaceStatus.PURGING:
                raise NamespaceStateError(
                    f"namespace must be quarantined before purge: {namespace_id}"
                )
            entry = await self._block_new_requests(namespace_id)
            if entry is not None:
                await self._shutdown_entry(namespace_id, entry)
            self._layout.purge(record.storage_key)
            self._store.remove_purging(namespace_id)
            self._failures.pop(namespace_id, None)
            return True

    async def purge_expired_namespaces(self) -> list[str]:
        purged: list[str] = []
        for record in self._store.list_records():
            if record.info.status is not NamespaceStatus.QUARANTINED:
                continue
            if await self.purge_namespace(record.info.namespace_id, force=False):
                purged.append(record.info.namespace_id)
        return purged

    @asynccontextmanager
    async def lease(self, context: RequestContext) -> AsyncIterator[NamespaceRuntime]:
        entry: _RuntimeEntry | None = None
        try:
            try:
                entry = await self._acquire(context.namespace_id)
            except AMemorixError as exc:
                if not exc.request_id:
                    exc.request_id = context.request_id
                if not exc.trace_id:
                    exc.trace_id = context.trace_id
                raise
            yield entry.runtime
        finally:
            if entry is not None:
                await self._release(context.namespace_id, entry)

    async def get_health(self, namespace_id: str) -> NamespaceHealth:
        self._require_running()
        namespace_id = validate_namespace_id(namespace_id)
        record = self._store.get_record(namespace_id)
        last_error = self._failures.get(namespace_id)
        try:
            storage_bytes = self._layout.storage_bytes(record.storage_key)
        except AMemorixError as exc:
            storage_bytes = 0
            last_error = str(exc)
        async with self._registry_lock:
            entry = self._entries.get(namespace_id)
            initializing = namespace_id in self._initializing
            active_requests = entry.active_requests if entry is not None else 0
        if initializing:
            runtime_state = NamespaceRuntimeState.LOADING
        elif last_error:
            runtime_state = NamespaceRuntimeState.FAILED
        elif entry is None:
            runtime_state = NamespaceRuntimeState.CLOSED
        else:
            try:
                runtime_state = (
                    NamespaceRuntimeState.READY
                    if entry.runtime.is_runtime_ready()
                    else NamespaceRuntimeState.DEGRADED
                )
            except BaseException as exc:
                runtime_state = NamespaceRuntimeState.FAILED
                last_error = f"runtime health check failed: {exc}"
        quota = record.info.quota
        over_storage_quota = bool(
            quota.max_storage_bytes is not None
            and storage_bytes > quota.max_storage_bytes
        )
        if over_storage_quota:
            runtime_state = NamespaceRuntimeState.DEGRADED
            last_error = (
                f"storage quota exceeded: {storage_bytes} > {quota.max_storage_bytes}"
            )
        healthy = bool(
            not last_error
            and record.info.status is not NamespaceStatus.PURGING
            and runtime_state
            in {
                NamespaceRuntimeState.CLOSED,
                NamespaceRuntimeState.READY,
            }
        )
        return NamespaceHealth(
            namespace=record.info,
            runtime_state=runtime_state,
            healthy=healthy,
            resource_usage=NamespaceResourceUsage(
                active_requests=active_requests,
                storage_bytes=storage_bytes,
            ),
            last_error=last_error,
        )

    async def close_idle_runtimes(self) -> list[str]:
        self._require_running()
        cutoff = self._clock.monotonic() - self._idle_timeout_seconds
        async with self._capacity_lock:
            async with self._registry_lock:
                candidates = sorted(
                    (
                        (namespace_id, entry)
                        for namespace_id, entry in self._entries.items()
                        if entry.active_requests == 0
                        and entry.accepting_requests
                        and entry.last_used <= cutoff
                    ),
                    key=lambda item: item[1].last_used,
                )
            closed: list[str] = []
            for namespace_id, entry in candidates:
                async with self._namespace_lock(namespace_id):
                    async with self._registry_lock:
                        current = self._entries.get(namespace_id)
                        if current is not entry or current.active_requests != 0:
                            continue
                        current.accepting_requests = False
                    try:
                        await self._shutdown_entry(namespace_id, entry)
                    except AMemorixError:
                        continue
                    closed.append(namespace_id)
            return closed

    async def _acquire(self, namespace_id: str) -> _RuntimeEntry:
        self._require_running()
        while True:
            await self._registry_lock.acquire()
            try:
                entry = self._entries.get(namespace_id)
                if entry is not None:
                    if not entry.accepting_requests:
                        raise NamespaceStateError(
                            f"namespace is not accepting requests: {namespace_id}"
                        )
                    request_limit = (
                        entry.namespace.quota.max_concurrent_requests
                        or self._max_concurrent_requests
                    )
                    if entry.active_requests >= request_limit:
                        raise NamespaceCapacityError(
                            f"namespace request limit reached: {namespace_id}",
                            details={"limit": request_limit},
                        )
                    entry.active_requests += 1
                    entry.zero_requests.clear()
                    return entry
                task = self._initializing.get(namespace_id)
                if task is None:
                    task = asyncio.create_task(
                        self._run_initialization(namespace_id),
                        name=f"a-memorix-init-{namespace_id}",
                    )
                    self._initializing[namespace_id] = task
            finally:
                self._registry_lock.release()
            await asyncio.shield(task)

    async def _run_initialization(self, namespace_id: str) -> None:
        current_task = asyncio.current_task()
        try:
            await self._initialize_runtime(namespace_id)
        finally:
            async with self._registry_lock:
                if self._initializing.get(namespace_id) is current_task:
                    self._initializing.pop(namespace_id, None)

    async def _initialize_runtime(self, namespace_id: str) -> None:
        async with self._capacity_lock:
            async with self._namespace_lock(namespace_id):
                self._require_running()
                record = self._store.get_record(namespace_id)
                if record.info.status is not NamespaceStatus.ACTIVE:
                    raise NamespaceStateError(
                        f"namespace is not active: {namespace_id}",
                        details={"status": record.info.status.value},
                    )
                data_dir = self._layout.validate_active(record.storage_key)
                await self._ensure_capacity(namespace_id)
                runtime: NamespaceRuntime | None = None
                try:
                    created = self._runtime_factory(record.info, data_dir)
                    candidate = (
                        await created if hasattr(created, "__await__") else created
                    )
                    if not isinstance(candidate, NamespaceRuntime):
                        raise TypeError(
                            "runtime factory returned an incompatible object"
                        )
                    runtime = candidate
                    await runtime.initialize()
                    if self._closed or self._shutting_down:
                        await runtime.shutdown()
                        runtime = None
                        raise RuntimeError(
                            "namespace registry shut down during initialization"
                        )
                except BaseException as exc:
                    failure = f"runtime initialization failed: {exc}"
                    if runtime is not None:
                        try:
                            await runtime.shutdown()
                        except BaseException as cleanup_error:
                            failure = f"{failure}; failed runtime cleanup also failed: {cleanup_error}"
                            failed_entry = _RuntimeEntry(
                                runtime=runtime,
                                namespace=record.info,
                                last_used=self._clock.monotonic(),
                                accepting_requests=False,
                            )
                            async with self._registry_lock:
                                self._entries[namespace_id] = failed_entry
                    self._failures[namespace_id] = failure
                    raise NamespaceRuntimeError(
                        f"namespace runtime failed to initialize: {namespace_id}",
                        details={"namespace_id": namespace_id},
                    ) from exc
                entry = _RuntimeEntry(
                    runtime=runtime,
                    namespace=record.info,
                    last_used=self._clock.monotonic(),
                )
                async with self._registry_lock:
                    self._entries[namespace_id] = entry
                self._store.touch(namespace_id, now=self._clock.time())
                self._failures.pop(namespace_id, None)

    async def _ensure_capacity(self, loading_namespace_id: str) -> None:
        while True:
            async with self._registry_lock:
                if len(self._entries) < self._max_active_namespaces:
                    return
                candidates = sorted(
                    (
                        (namespace_id, entry)
                        for namespace_id, entry in self._entries.items()
                        if namespace_id != loading_namespace_id
                        and entry.active_requests == 0
                        and entry.accepting_requests
                    ),
                    key=lambda item: item[1].last_used,
                )
                if not candidates:
                    raise NamespaceCapacityError(
                        "all active namespace runtimes are busy",
                        details={"max_active_namespaces": self._max_active_namespaces},
                    )
                victim_id, victim = candidates[0]
                victim.accepting_requests = False
            async with self._namespace_lock(victim_id):
                await self._shutdown_entry(victim_id, victim)

    async def _release(self, namespace_id: str, entry: _RuntimeEntry) -> None:
        async with self._registry_lock:
            current = self._entries.get(namespace_id)
            if current is not entry:
                return
            entry.active_requests -= 1
            entry.last_used = self._clock.monotonic()
            if entry.active_requests == 0:
                entry.zero_requests.set()
        self._store.touch(namespace_id, now=self._clock.time())

    async def _block_new_requests(self, namespace_id: str) -> _RuntimeEntry | None:
        async with self._registry_lock:
            entry = self._entries.get(namespace_id)
            if entry is not None:
                entry.accepting_requests = False
            return entry

    async def _resume_requests(
        self, namespace_id: str, entry: _RuntimeEntry | None
    ) -> None:
        if entry is None:
            return
        async with self._registry_lock:
            if self._entries.get(namespace_id) is entry:
                entry.accepting_requests = True

    async def _shutdown_entry(self, namespace_id: str, entry: _RuntimeEntry) -> None:
        await entry.zero_requests.wait()
        try:
            await entry.runtime.shutdown()
        except BaseException as exc:
            self._failures[namespace_id] = f"runtime shutdown failed: {exc}"
            raise NamespaceRuntimeError(
                f"namespace runtime failed to shut down: {namespace_id}",
                details={"namespace_id": namespace_id},
            ) from exc
        async with self._registry_lock:
            if self._entries.get(namespace_id) is entry:
                self._entries.pop(namespace_id, None)

    async def _idle_reaper(self) -> None:
        interval = min(max(self._idle_timeout_seconds / 2, 0.1), 60.0)
        try:
            while True:
                await asyncio.sleep(interval)
                await self.close_idle_runtimes()
                await self.purge_expired_namespaces()
        except asyncio.CancelledError:
            raise
        except BaseException:
            if not self._closed:
                self._reaper_task = asyncio.create_task(
                    self._idle_reaper(),
                    name="a-memorix-namespace-lru-reaper",
                )

    def _namespace_lock(self, namespace_id: str) -> asyncio.Lock:
        lock = self._namespace_locks.get(namespace_id)
        if lock is None:
            lock = asyncio.Lock()
            self._namespace_locks[namespace_id] = lock
        return lock

    def _require_running(self) -> None:
        if not self._started:
            raise RuntimeError("namespace registry has not been started")
        if self._shutting_down:
            raise RuntimeError("namespace registry is shutting down")
        if self._closed:
            raise RuntimeError("namespace registry has been shut down")
