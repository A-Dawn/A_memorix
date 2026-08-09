from pathlib import Path
from typing import Any

import pytest

from a_memorix.core.runtime.sdk_memory_kernel import SDKMemoryKernel


class _DummyImportTaskManager:
    def __init__(self) -> None:
        self.sources: list[str] = []

    async def invalidate_manifest_for_sources(self, sources: list[str]) -> dict[str, Any]:
        self.sources.extend(sources)
        return {"removed_count": len(sources), "removed_keys": [f"key:{source}" for source in sources]}


@pytest.mark.asyncio
async def test_memory_delete_admin_execute_invalidates_import_manifest(monkeypatch) -> None:
    kernel = SDKMemoryKernel(data_dir=Path.cwd(), config={})
    manager = _DummyImportTaskManager()
    kernel.import_task_manager = manager  # type: ignore[assignment]

    async def fake_initialize() -> None:
        return None

    async def fake_execute_delete_action(**kwargs):
        assert kwargs["mode"] == "source"
        assert kwargs["selector"] == {"sources": ["web_import:demo.txt"]}
        return {"success": True, "sources": ["web_import:demo.txt"], "deleted_source_count": 1}

    monkeypatch.setattr(kernel, "initialize", fake_initialize)
    monkeypatch.setattr(kernel, "_execute_delete_action", fake_execute_delete_action)

    result = await kernel.memory_delete_admin(
        action="execute",
        mode="source",
        selector={"sources": ["web_import:demo.txt"]},
    )

    assert manager.sources == ["web_import:demo.txt"]
    assert result["manifest_invalidation"]["removed_count"] == 1
