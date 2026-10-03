from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import asyncio
import hashlib
import json
import shutil
import time

from src.common.logger import get_logger

from .base import KernelServiceBase

logger = get_logger("A_Memorix.VectorSpaces")


class MemoryVectorSpaceService(KernelServiceBase):
    """按模型指纹保存独立向量库，切换时只重新编码发生变化的输入。"""

    @staticmethod
    def input_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def space_id(fingerprint: Dict[str, Any]) -> str:
        return hashlib.sha256(str(fingerprint["hash"]).encode("utf-8")).hexdigest()

    def _catalog_root(self) -> Path:
        return self.data_dir / "vectors" / "spaces"

    def _read_active_space(self) -> str:
        path = self.data_dir / "vectors" / "active.json"
        if not path.exists():
            return ""
        return str(json.loads(path.read_text(encoding="utf-8"))["space_id"])

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        temporary.replace(path)

    def _read_inputs(self) -> Dict[str, str]:
        path = self._vectors_root() / "embedding_inputs.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _save_inputs(self) -> None:
        self._write_json(self._vectors_root() / "embedding_inputs.json", self._vector_space_inputs)

    def record_input(self, item_type: str, item_id: str, text: str) -> None:
        self._vector_space_inputs[f"{item_type}:{item_id}"] = self.input_hash(text)

    def _register_current_space(self, fingerprint: Dict[str, Any]) -> None:
        space_id = self._active_vector_space_id
        root = self._catalog_root() / space_id
        path = root / "space.json"
        previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self._write_json(path, {
            "space_id": space_id,
            "embedding_fingerprint": fingerprint,
            "created_at": previous.get("created_at", time.time()),
            "last_used_at": time.time(),
        })
        self._write_json(self.data_dir / "vectors" / "active.json", {"space_id": space_id})
        self._save_inputs()

    def _migrate_legacy_space(self) -> None:
        """给原来固定路径的向量库登记模型身份，不修改其向量内容。"""
        if self._active_vector_space_id:
            return
        fingerprint = self._stored_embedding_fingerprint()
        if fingerprint is None:
            return
        old_root = self.data_dir / "vectors"
        space_id = self.space_id(fingerprint)
        target = self._catalog_root() / space_id
        target.mkdir(parents=True, exist_ok=True)
        for name in ("paragraph", "graph", "dual_ready.json", "embedding_inputs.json",
                     "vectors.bin", "vectors_ids.bin", "vectors.index", "vectors_metadata.json", "vectors_metadata.pkl"):
            source = old_root / name
            if source.exists():
                shutil.move(str(source), str(target / name))
        self._active_vector_space_id = space_id
        self._vector_space_inputs = self._read_inputs()
        self._register_current_space(fingerprint)

    async def synchronize(self, *, force: bool = False) -> Dict[str, Any]:
        fingerprint = self._current_embedding_fingerprint_for_validation()
        if fingerprint is None:
            return {"success": False, "error": "embedding_fingerprint_unavailable"}
        space_id = self.space_id(fingerprint)
        if not force and space_id == self._active_vector_space_id and self._runtime_capabilities["vector_read"]:
            return {"success": True, "changed": False}
        async with self._vector_rebuild_lock:
            if not force and space_id == self._active_vector_space_id and self._runtime_capabilities["vector_read"]:
                return {"success": True, "changed": False}
            self._target_vector_space_id = space_id
            self._set_embedding_degraded(active=True, reason="正在同步当前模型向量库")
            self._set_runtime_capability("vector_read", False)
            self._set_runtime_capability("vector_write", False)
            original_inputs = dict(self._vector_space_inputs)
            try:
                # 先保存仍属于旧模型的内存向量，再改动当前库引用和目录。
                old_fingerprint = self._stored_embedding_fingerprint()
                if old_fingerprint is not None:
                    for store in (self.vector_store, self.paragraph_vector_store, self.graph_vector_store):
                        if store is not None:
                            await asyncio.to_thread(store.save, embedding_fingerprint=old_fingerprint)
                    await asyncio.to_thread(self._save_inputs)
                await asyncio.to_thread(self._migrate_legacy_space)
                self._active_vector_space_id = space_id
                self._vector_space_inputs = await asyncio.to_thread(self._read_inputs)
                original_inputs = dict(self._vector_space_inputs)
                self._vector_space_sources = {}
                self._dual_vector_pools_ready = False
                self._legacy_vector_view = None
                self.embedding_dimension = self._current_embedding_status_dimension()
                self.vector_store = self._make_vector_store(self._vectors_root())
                self.paragraph_vector_store = None
                self.graph_vector_store = None
                await asyncio.to_thread(self.metadata_store.reset_vector_projection_state)
                # 已保存的目标库先加载为复制来源；全量计划只对缺失或变化的输入调用模型。
                await asyncio.to_thread(self._embedding_state_service._load_vector_stores_for_runtime)
                if self._dual_vector_pools_enabled():
                    self._vector_space_sources = {
                        "paragraph": self.paragraph_vector_store,
                        "graph": self.graph_vector_store,
                    }
                elif self.vector_store.has_data():
                    self._vector_space_sources = {"single": self.vector_store}
                result = await self._vector_runtime_service._rebuild_all_vectors_locked(reuse_space=True)
                if not result.get("success"):
                    raise RuntimeError(str(result.get("errors") or result.get("error") or "向量库同步失败"))
                await asyncio.to_thread(self._register_current_space, fingerprint)
                logger.info(f"模型向量库已启用: model={fingerprint.get('model')}, space={space_id}")
                return {**result, "changed": True, "space_id": space_id}
            except Exception as exc:
                self._vector_space_inputs = original_inputs
                self._set_vector_health(state="unavailable", error_code="vector_space_sync_failed", reason=str(exc))
                self._set_embedding_degraded(active=True, reason=str(exc))
                raise
            finally:
                self._vector_space_sources = {}
                self._target_vector_space_id = ""
