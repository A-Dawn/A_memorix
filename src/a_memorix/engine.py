"""Public multi-namespace in-process engine."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from a_memorix.contracts import (
    CreateNamespaceRequest,
    NamespaceHealth,
    NamespaceInfo,
    RequestContext,
)
from a_memorix.ports import Clock

from .core.runtime.namespace_control import NamespaceControlStore
from .core.runtime.namespace_registry import NamespaceRuntimeRegistry
from .core.runtime.namespace_runtime import (
    NamespaceConfigFactory,
    NamespaceHostPortFactory,
    NamespaceRuntime,
    NamespaceRuntimeFactory,
    SDKKernelRuntimeFactory,
)
from .core.runtime.namespace_storage import NamespaceStorageLayout


class AMemorixEngine:
    """Manage isolated namespace runtimes under one service data root."""

    def __init__(
        self,
        *,
        data_dir: str | Path,
        runtime_factory: NamespaceRuntimeFactory | None = None,
        config_factory: NamespaceConfigFactory | None = None,
        host_port_factory: NamespaceHostPortFactory | None = None,
        max_active_namespaces: int = 8,
        max_concurrent_requests_per_namespace: int = 64,
        idle_timeout_seconds: float = 900.0,
        quarantine_retention_days: float = 7.0,
        clock: Clock | None = None,
    ) -> None:
        if runtime_factory is not None and (
            config_factory is not None or host_port_factory is not None
        ):
            raise ValueError(
                "config_factory and host_port_factory cannot be combined with runtime_factory"
            )
        self._layout = NamespaceStorageLayout(data_dir)
        self._runtime_factory = runtime_factory or SDKKernelRuntimeFactory(
            config_factory=config_factory,
            host_port_factory=host_port_factory,
        )
        self._max_active_namespaces = max_active_namespaces
        self._max_concurrent_requests = max_concurrent_requests_per_namespace
        self._idle_timeout_seconds = idle_timeout_seconds
        self._quarantine_retention_seconds = quarantine_retention_days * 24 * 60 * 60
        self._clock = clock
        self._store: NamespaceControlStore | None = None
        self._registry: NamespaceRuntimeRegistry | None = None

    @property
    def data_dir(self) -> Path:
        return self._layout.data_root

    async def initialize(self) -> None:
        if self._registry is not None:
            return
        self._layout.initialize()
        store = NamespaceControlStore(self._layout.control_db_path)
        registry = NamespaceRuntimeRegistry(
            control_store=store,
            storage_layout=self._layout,
            runtime_factory=self._runtime_factory,
            max_active_namespaces=self._max_active_namespaces,
            max_concurrent_requests_per_namespace=self._max_concurrent_requests,
            idle_timeout_seconds=self._idle_timeout_seconds,
            quarantine_retention_seconds=self._quarantine_retention_seconds,
            clock=self._clock,
        )
        try:
            await registry.start()
        except BaseException:
            store.close()
            raise
        self._store = store
        self._registry = registry

    async def shutdown(self) -> None:
        registry = self._registry
        store = self._store
        if registry is None:
            return
        await registry.shutdown()
        if store is not None:
            store.close()
        self._registry = None
        self._store = None

    async def __aenter__(self) -> "AMemorixEngine":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        await self.shutdown()

    async def create_namespace(self, request: CreateNamespaceRequest) -> NamespaceInfo:
        return await self._require_registry().create_namespace(request)

    async def get_namespace(self, namespace_id: str) -> NamespaceInfo:
        return self._require_registry().get_namespace(namespace_id)

    async def list_namespaces(self) -> list[NamespaceInfo]:
        return self._require_registry().list_namespaces()

    async def disable_namespace(self, namespace_id: str) -> NamespaceInfo:
        return await self._require_registry().disable_namespace(namespace_id)

    async def enable_namespace(self, namespace_id: str) -> NamespaceInfo:
        return await self._require_registry().enable_namespace(namespace_id)

    async def delete_namespace(self, namespace_id: str) -> NamespaceInfo:
        return await self._require_registry().delete_namespace(namespace_id)

    async def restore_namespace(self, namespace_id: str) -> NamespaceInfo:
        return await self._require_registry().restore_namespace(namespace_id)

    async def purge_namespace(self, namespace_id: str) -> None:
        await self._require_registry().purge_namespace(namespace_id, force=True)

    async def purge_expired_namespaces(self) -> list[str]:
        return await self._require_registry().purge_expired_namespaces()

    async def namespace_health(self, namespace_id: str) -> NamespaceHealth:
        return await self._require_registry().get_health(namespace_id)

    async def close_idle_runtimes(self) -> list[str]:
        return await self._require_registry().close_idle_runtimes()

    @asynccontextmanager
    async def runtime(self, context: RequestContext) -> AsyncIterator[NamespaceRuntime]:
        async with self._require_registry().lease(context) as runtime:
            yield runtime

    def _require_registry(self) -> NamespaceRuntimeRegistry:
        if self._registry is None:
            raise RuntimeError("AMemorixEngine has not been initialized")
        return self._registry
