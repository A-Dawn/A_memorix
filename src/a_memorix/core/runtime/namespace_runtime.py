"""Runtime contracts and the default SDK kernel factory."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import inspect

from a_memorix.contracts import NamespaceInfo
from a_memorix.ports import NamespaceHostPorts


@runtime_checkable
class NamespaceRuntime(Protocol):
    async def initialize(self) -> None:
        """Open storage and background resources."""

    async def shutdown(self) -> None:
        """Flush and release all namespace-owned resources."""

    def is_runtime_ready(self) -> bool:
        """Report whether the runtime can currently serve requests."""


NamespaceRuntimeFactory = Callable[
    [NamespaceInfo, Path],
    NamespaceRuntime | Awaitable[NamespaceRuntime],
]
NamespaceConfigFactory = Callable[
    [NamespaceInfo],
    Mapping[str, object] | Awaitable[Mapping[str, object]],
]
NamespaceHostPortFactory = Callable[
    [NamespaceInfo],
    NamespaceHostPorts | Awaitable[NamespaceHostPorts],
]


class SDKKernelRuntimeFactory:
    """Create one SDKMemoryKernel with namespace-specific config and ports."""

    def __init__(
        self,
        *,
        config_factory: NamespaceConfigFactory | None = None,
        host_port_factory: NamespaceHostPortFactory | None = None,
    ) -> None:
        self._config_factory = config_factory
        self._host_port_factory = host_port_factory

    async def __call__(
        self, namespace: NamespaceInfo, data_dir: Path
    ) -> NamespaceRuntime:
        from .sdk_memory_kernel import SDKMemoryKernel

        config: Mapping[str, object] = {}
        if self._config_factory is not None:
            config = await _resolve(self._config_factory(namespace))
        effective_config = _merge_config(
            deepcopy(dict(config)),
            _namespace_feature_config(namespace),
        )
        ports = NamespaceHostPorts()
        if self._host_port_factory is not None:
            ports = await _resolve(self._host_port_factory(namespace))
        return SDKMemoryKernel(
            data_dir=data_dir,
            config=effective_config,
            embedding_provider=ports.embedding_provider,
            llm_provider=ports.llm_provider,
            identity_resolver=ports.identity_resolver,
            message_source=ports.message_source,
        )


async def _resolve(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _namespace_feature_config(namespace: NamespaceInfo) -> dict[str, object]:
    features = namespace.config.features
    return {
        "episode": {
            "enabled": features.episodes,
            "generation_enabled": features.episodes,
        },
        "person_profile": {"enabled": features.person_profiles},
        "retrieval": {
            "sparse": {"enabled": features.sparse_retrieval},
            "relation_vectorization": {"enabled": features.relation_vectors},
        },
        "embedding": {
            "fallback": {
                "allow_metadata_only_write": features.allow_metadata_only_write,
            }
        },
    }


def _merge_config(
    base: dict[str, object],
    patch: Mapping[str, object],
) -> dict[str, object]:
    for key, value in patch.items():
        current = base.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            base[key] = _merge_config(current, value)
        else:
            base[key] = deepcopy(value)
    return base
