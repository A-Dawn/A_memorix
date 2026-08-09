"""Explicit runtime capabilities shared with internal application services."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, Protocol, Sequence

from ...ports import LLMProvider


class RuntimeServices(Protocol):
    """Narrow interface for components that collaborate with one memory runtime."""

    data_dir: Path
    config: Dict[str, Any]
    llm_provider: LLMProvider | None
    vector_store: Any
    paragraph_vector_store: Any
    graph_vector_store: Any
    graph_store: Any
    metadata_store: Any
    embedding_manager: Any
    sparse_index: Any
    relation_write_service: Any

    def get_config(self, key: str, default: Any = None) -> Any: ...

    def is_runtime_ready(self) -> bool: ...

    def is_chat_enabled(
        self,
        stream_id: str,
        group_id: str | None = None,
        user_id: str | None = None,
    ) -> bool: ...

    def dual_vector_pools_enabled(self) -> bool: ...

    def is_embedding_degraded(self) -> bool: ...

    def allow_metadata_only_write(self) -> bool: ...

    async def ensure_runtime_self_check(self, *, force: bool = False) -> Dict[str, Any]: ...

    async def execute_request_with_dedup(
        self,
        request_key: str,
        executor: Callable[[], Coroutine[Any, Any, Dict[str, Any]]],
    ) -> tuple[bool, Dict[str, Any]]: ...

    async def apply_retrieval_tuning_profile(
        self,
        profile: Dict[str, Any],
        *,
        validate: bool = True,
    ) -> Dict[str, Any]: ...

    async def ingest_text(self, **kwargs: Any) -> Dict[str, Any]: ...

    async def write_paragraph_vector_or_enqueue(
        self,
        *,
        paragraph_hash: str,
        content: str,
        context: str = "",
    ) -> Dict[str, Any]: ...

    def enqueue_paragraph_vector_backfill(
        self,
        paragraph_hash: str,
        *,
        error: str = "",
    ) -> None: ...

    async def reinforce_access(self, relation_hashes: Sequence[str]) -> None: ...
