"""Memory maintenance API service."""

from __future__ import annotations

import datetime
from typing import Any, Dict, List

from core.utils.search_execution_service import SearchExecutionRequest, SearchExecutionService

from amemorix.context import AppContext


class MemoryService:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx

    async def status(self) -> Dict[str, Any]:
        cursor = self.ctx.metadata_store._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM relations WHERE is_inactive = 0")
        active_count = int(cursor.fetchone()[0] or 0)
        cursor.execute("SELECT COUNT(*) FROM relations WHERE is_inactive = 1")
        inactive_count = int(cursor.fetchone()[0] or 0)
        cursor.execute("SELECT COUNT(*) FROM deleted_relations")
        deleted_count = int(cursor.fetchone()[0] or 0)
        now = datetime.datetime.now().timestamp()
        cursor.execute("SELECT COUNT(*) FROM relations WHERE is_pinned = 1")
        pinned_count = int(cursor.fetchone()[0] or 0)
        cursor.execute("SELECT COUNT(*) FROM relations WHERE protected_until > ?", (now,))
        ttl_count = int(cursor.fetchone()[0] or 0)

        return {
            "active_relations": active_count,
            "inactive_relations": inactive_count,
            "recycle_bin_relations": deleted_count,
            "pinned_relations": pinned_count,
            "ttl_protected_relations": ttl_count,
            "config": self.ctx.get_config("memory", {}),
        }

    async def protect(self, query_or_hash: str, hours: float = 24.0) -> Dict[str, Any]:
        hashes = await self._resolve_relations(query_or_hash)
        if not hashes:
            return {"success": False, "message": "未找到关系"}
        now = datetime.datetime.now().timestamp()
        if float(hours) <= 0:
            self.ctx.metadata_store.update_relations_protection(hashes, is_pinned=True)
            mode = "pin"
            until = None
        else:
            until = now + float(hours) * 3600
            self.ctx.metadata_store.update_relations_protection(hashes, protected_until=until, is_pinned=False)
            mode = "ttl"
        return {"success": True, "mode": mode, "count": len(hashes), "protected_until": until}

    async def reinforce(self, query_or_hash: str) -> Dict[str, Any]:
        hashes = await self._resolve_relations(query_or_hash)
        if not hashes:
            return {"success": False, "message": "未找到关系"}

        status_map = self.ctx.metadata_store.get_relation_status_batch(hashes)
        cursor = self.ctx.metadata_store._conn.cursor()
        placeholders = ",".join(["?"] * len(hashes))
        cursor.execute(
            f"SELECT hash, subject, object FROM relations WHERE hash IN ({placeholders})",
            hashes,
        )
        revived = 0
        now = datetime.datetime.now().timestamp()
        for row in cursor.fetchall():
            h, s, o = row
            st = status_map.get(h)
            if st and st.get("is_inactive"):
                revived += 1
            self.ctx.graph_store.update_edge_weight(str(s), str(o), 1.0, max_weight=float(self.ctx.get_config("memory.max_weight", 10.0)))

        self.ctx.metadata_store.reinforce_relations(hashes)
        self.ctx.metadata_store.update_relations_protection(
            hashes,
            last_reinforced=now,
            protected_until=now + float(self.ctx.get_config("memory.auto_protect_ttl_hours", 24.0)) * 3600,
        )
        self.ctx.graph_store.save()
        return {"success": True, "count": len(hashes), "revived": revived}

    async def restore(self, hash_value: str, restore_type: str = "relation") -> Dict[str, Any]:
        h = str(hash_value or "").strip().lower()
        if not h:
            raise ValueError("hash is empty")

        if restore_type == "entity":
            cursor = self.ctx.metadata_store._conn.cursor()
            cursor.execute("UPDATE entities SET is_deleted=0, deleted_at=NULL WHERE hash=?", (h,))
            self.ctx.metadata_store._conn.commit()
            return {"success": True, "type": "entity", "hash": h}

        record = self.ctx.metadata_store.restore_relation(h)
        if not record:
            return {"success": False, "message": "relation not found in recycle bin"}
        s = str(record["subject"])
        o = str(record["object"])
        self.ctx.metadata_store.revive_entities_by_names([s, o])
        self.ctx.graph_store.add_nodes([s, o])
        self.ctx.graph_store.add_edges([(s, o)], weights=[float(record.get("confidence", 1.0) or 1.0)], relation_hashes=[h])
        self.ctx.graph_store.save()
        return {"success": True, "type": "relation", "hash": h}

    async def _resolve_relations(self, query: str) -> List[str]:
        value = str(query or "").strip()
        if not value:
            return []

        if len(value) in {32, 64} and all(c in "0123456789abcdefABCDEF" for c in value):
            v = value.lower()
            if len(v) == 64:
                st = self.ctx.metadata_store.get_relation_status_batch([v])
                if st:
                    return [v]
            cursor = self.ctx.metadata_store._conn.cursor()
            cursor.execute("SELECT hash FROM relations WHERE hash LIKE ? LIMIT 5", (f"{v}%",))
            hits = [str(row[0]) for row in cursor.fetchall()]
            if hits:
                return hits

        # Semantic relation fallback
        search = await SearchExecutionService.execute(
            retriever=self.ctx.retriever,
            threshold_filter=None,
            plugin_config={
                **self.ctx.config,
                "plugin_instance": self.ctx,
                "graph_store": self.ctx.graph_store,
                "metadata_store": self.ctx.metadata_store,
            },
            request=SearchExecutionRequest(
                caller="v1.memory.resolve",
                query_type="search",
                query=value,
                top_k=10,
                use_threshold=False,
                enable_ppr=True,
            ),
            enforce_chat_filter=False,
            reinforce_access=False,
        )
        if search.success:
            hashes = [r.hash_value for r in search.results if getattr(r, "result_type", "") == "relation"]
            if hashes:
                return hashes[:5]

        cursor = self.ctx.metadata_store._conn.cursor()
        cursor.execute(
            "SELECT hash FROM relations WHERE subject LIKE ? OR object LIKE ? LIMIT 5",
            (f"%{value}%", f"%{value}%"),
        )
        return [str(row[0]) for row in cursor.fetchall()]
