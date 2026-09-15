"""在提供 src 宿主依赖的环境运行：python -m pytest tests/test_empty_summary.py。"""

from importlib import import_module
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


# 插件可以安装在任意目录，使用固定测试包名加载实际实现。
package = ModuleType("memorix_under_test")
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[package.__name__] = package

SDKMemoryKernel = import_module("memorix_under_test.core.runtime.sdk_memory_kernel").SDKMemoryKernel
module = import_module("memorix_under_test.core.utils.summary_importer")


@pytest.fixture
def importer(monkeypatch):
    instance = module.SummaryImporter(
        vector_store=Mock(), graph_store=Mock(), metadata_store=Mock(),
        embedding_manager=Mock(), plugin_config={},
    )
    monkeypatch.setattr(instance, "_ensure_runtime_self_check", AsyncMock(return_value=(True, "")))
    monkeypatch.setattr(instance, "_resolve_summary_model_config", lambda: SimpleNamespace(model_list=["synthetic"]))
    monkeypatch.setattr(module.message_api, "get_messages_by_time_in_chat", lambda **_: [SimpleNamespace(time=1)])
    monkeypatch.setattr(module.message_api, "build_readable_messages", lambda _: "用户：你好。")
    monkeypatch.setattr(module.llm_api, "resolve_task_name_from_model_config", lambda _: "utils", raising=False)
    monkeypatch.setattr(instance, "_execute_import", AsyncMock())
    return instance


@pytest.mark.asyncio
@pytest.mark.parametrize("response,success,skipped", [
    ('{"summary":"","entities":[],"relations":[]}', True, True),
    ('```json\n{"summary":"  ","entities":[],"relations":[]}\n```', True, True),
    ('{"summary":"","entities":["用户"],"relations":[]}', False, False),
    ('{"summary":"","entities":[],"relations":{}}', False, False),
    ('{"summary":null,"entities":[],"relations":[]}', False, False),
    ('{"summary":""}', False, False),
    ('{}', False, False),
    ('not json', False, False),
    (json.dumps({"summary": "用户喜欢猫。", "entities": [], "relations": []}), True, False),
])
async def test_import_result_distinguishes_empty_from_invalid(importer, monkeypatch, response, success, skipped):
    monkeypatch.setattr(module.llm_api, "generate", AsyncMock(return_value=SimpleNamespace(
        success=True, completion=SimpleNamespace(response=response),
    )))
    result = await importer.import_from_stream("synthetic-chat", include_personality=False)
    assert tuple(result) == (result.success, result.detail)
    assert result.success is success
    assert result.skipped is skipped
    assert importer._execute_import.await_count == int(success and not skipped)
    assert importer.vector_store.save.call_count == int(success and not skipped)
    assert importer.graph_store.save.call_count == int(success and not skipped)


@pytest.mark.asyncio
async def test_empty_summary_does_not_rebuild_or_persist(tmp_path, monkeypatch):
    kernel = SDKMemoryKernel(plugin_root=tmp_path, config={})
    monkeypatch.setattr(kernel, "initialize", AsyncMock())
    monkeypatch.setattr(kernel, "_is_chat_filtered", lambda **_: False)
    monkeypatch.setattr(kernel, "_persist", Mock())
    monkeypatch.setattr(kernel, "rebuild_episodes_for_sources", AsyncMock())
    kernel.summary_importer = SimpleNamespace(import_from_stream=AsyncMock(
        return_value=module.SummaryImportResult(True, "本批没有可写入的记忆", skipped=True),
    ))
    result = await kernel.ingest_summary(external_id="synthetic-batch", chat_id="synthetic-chat", text="")
    assert result["success"] is True
    assert result["skipped_ids"] == ["synthetic-batch"]
    assert not result.get("stored_ids")
    kernel._persist.assert_not_called()
    kernel.rebuild_episodes_for_sources.assert_not_awaited()
