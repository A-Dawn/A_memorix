"""Delete orchestration service."""

from __future__ import annotations

import re
from typing import Any, Dict

from core.utils.hash import compute_hash, normalize_text

from amemorix.context import AppContext


class DeleteService:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx

    @staticmethod
    def _looks_like_hash(text: str) -> bool:
        return bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(text or "").strip()))

    async def paragraph(self, paragraph_spec: str) -> Dict[str, Any]:
        query = str(paragraph_spec or "").strip()
        if not query:
            raise ValueError("paragraph_spec is empty")

        target = None
        if self._looks_like_hash(query):
            target = self.ctx.metadata_store.get_paragraph(query)
            if not target:
                raise ValueError(f"paragraph not found: {query}")
        else:
            matches = self.ctx.metadata_store.search_paragraphs_by_content(query)
            if not matches:
                raise ValueError("paragraph not found")
            if len(matches) > 1:
                query_norm = normalize_text(query)
                exact = [m for m in matches if normalize_text(str(m.get("content", ""))) == query_norm]
                if len(exact) != 1:
                    raise ValueError("multiple paragraphs matched, use hash")
                target = exact[0]
            else:
                target = matches[0]

        paragraph_hash = str(target["hash"])
        plan = self.ctx.metadata_store.delete_paragraph_atomic(paragraph_hash)
        relation_prune_ops = plan.get("relation_prune_ops", []) or []
        edges_to_remove = plan.get("edges_to_remove", []) or []

        if relation_prune_ops and hasattr(self.ctx.graph_store, "prune_relation_hashes"):
            self.ctx.graph_store.prune_relation_hashes(relation_prune_ops)
        elif edges_to_remove:
            self.ctx.graph_store.delete_edges(edges_to_remove)

        vector_ids = []
        vid = plan.get("vector_id_to_remove")
        if vid:
            vector_ids.append(str(vid))
        for op in relation_prune_ops:
            if len(op) >= 3 and op[2]:
                vector_ids.append(str(op[2]))
        deleted_vectors = self.ctx.vector_store.delete(list(dict.fromkeys(vector_ids))) if vector_ids else 0

        self.ctx.vector_store.save()
        self.ctx.graph_store.save()
        return {
            "success": True,
            "paragraph_hash": paragraph_hash,
            "relation_prune_count": len(relation_prune_ops),
            "deleted_vectors": deleted_vectors,
        }

    async def entity(self, entity_name: str) -> Dict[str, Any]:
        target = str(entity_name or "").strip()
        if not target:
            raise ValueError("entity_name is empty")
        canonical = target.lower()
        if not self.ctx.graph_store.has_node(canonical):
            raise ValueError(f"entity not found: {canonical}")

        neighbors = self.ctx.graph_store.get_neighbors(canonical)
        rel_hashes = {
            str(r["hash"])
            for r in (
                self.ctx.metadata_store.get_relations(subject=canonical)
                + self.ctx.metadata_store.get_relations(object=canonical)
            )
            if r.get("hash")
        }

        self.ctx.graph_store.delete_nodes([canonical])
        self.ctx.metadata_store.delete_entity(canonical)
        vector_ids = [compute_hash(canonical)] + list(rel_hashes)
        deleted_vectors = self.ctx.vector_store.delete(vector_ids)

        self.ctx.vector_store.save()
        self.ctx.graph_store.save()
        return {
            "success": True,
            "entity_name": canonical,
            "deleted_edges": len(neighbors),
            "deleted_vectors": deleted_vectors,
        }

    async def relation(self, relation_spec: str) -> Dict[str, Any]:
        query = str(relation_spec or "").strip()
        if not query:
            raise ValueError("relation_spec is empty")

        relation = None
        if self._looks_like_hash(query):
            rel_hash = query.lower()
            relation = self.ctx.metadata_store.get_relation(rel_hash)
            if not relation:
                raise ValueError(f"relation not found: {rel_hash}")
        else:
            if "|" in query:
                parts = [p.strip() for p in query.split("|")]
                if len(parts) != 3:
                    raise ValueError("relation format should be subject|predicate|object")
                s, p, o = parts
            else:
                parts = query.split(maxsplit=2)
                if len(parts) != 3:
                    raise ValueError("relation format should be subject predicate object")
                s, p, o = parts
            rel_hash = compute_hash(f"{s.lower()}|{p.lower()}|{o.lower()}")
            relation = self.ctx.metadata_store.get_relation(rel_hash)
            if not relation:
                raise ValueError("relation not found")

        # Keep relation restore path functional: delete into recycle-bin first.
        deleted_count = self.ctx.metadata_store.backup_and_delete_relations([rel_hash])
        if int(deleted_count) <= 0:
            raise RuntimeError("delete relation failed")

        subject = str(relation.get("subject", ""))
        obj = str(relation.get("object", ""))
        if hasattr(self.ctx.graph_store, "prune_relation_hashes"):
            self.ctx.graph_store.prune_relation_hashes([(subject, obj, rel_hash)])
        else:
            self.ctx.graph_store.delete_edges([(subject, obj)])

        deleted_vectors = self.ctx.vector_store.delete([rel_hash])
        self.ctx.vector_store.save()
        self.ctx.graph_store.save()
        return {
            "success": True,
            "relation_hash": rel_hash,
            "subject": subject,
            "predicate": str(relation.get("predicate", "")),
            "object": obj,
            "deleted_vectors": deleted_vectors,
        }

    async def clear(self) -> Dict[str, Any]:
        counts = {
            "paragraphs": self.ctx.metadata_store.count_paragraphs(),
            "relations": self.ctx.metadata_store.count_relations(),
            "entities": self.ctx.metadata_store.count_entities(),
            "vectors": self.ctx.vector_store.num_vectors,
        }
        self.ctx.vector_store.clear()
        self.ctx.graph_store.clear()
        self.ctx.metadata_store.clear_all()
        self.ctx.vector_store.save()
        self.ctx.graph_store.save()
        return {"success": True, "deleted": counts}
