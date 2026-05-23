"""Query orchestration service."""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.utils.aggregate_query_service import AggregateQueryService
from core.utils.search_execution_service import (
    SearchExecutionRequest,
    SearchExecutionService,
)
from core.utils.time_parser import parse_query_time_range

from amemorix.context import AppContext


class QueryService:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx

    def _plugin_config(self) -> Dict[str, Any]:
        cfg = dict(self.ctx.config)
        cfg["plugin_instance"] = self.ctx
        cfg["graph_store"] = self.ctx.graph_store
        cfg["metadata_store"] = self.ctx.metadata_store
        return cfg

    async def search(self, *, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        req = SearchExecutionRequest(
            caller="v1.search",
            query_type="search",
            query=str(query or ""),
            top_k=top_k,
            use_threshold=True,
            enable_ppr=bool(self.ctx.get_config("retrieval.enable_ppr", True)),
        )
        result = await SearchExecutionService.execute(
            retriever=self.ctx.retriever,
            threshold_filter=self.ctx.threshold_filter,
            plugin_config=self._plugin_config(),
            request=req,
            enforce_chat_filter=False,
            reinforce_access=True,
        )
        if not result.success:
            raise ValueError(result.error)
        return {
            "query_type": result.query_type,
            "query": result.query,
            "top_k": result.top_k,
            "count": result.count,
            "elapsed_ms": result.elapsed_ms,
            "results": SearchExecutionService.to_serializable_results(result.results),
        }

    async def time_search(
        self,
        *,
        query: str = "",
        time_from: Optional[str] = None,
        time_to: Optional[str] = None,
        person: Optional[str] = None,
        source: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        # Validate format eagerly to provide deterministic error text.
        parse_query_time_range(time_from, time_to)

        req = SearchExecutionRequest(
            caller="v1.time",
            query_type="time",
            query=str(query or ""),
            top_k=top_k,
            time_from=time_from,
            time_to=time_to,
            person=person,
            source=source,
            use_threshold=True,
            enable_ppr=bool(self.ctx.get_config("retrieval.enable_ppr", True)),
        )
        result = await SearchExecutionService.execute(
            retriever=self.ctx.retriever,
            threshold_filter=self.ctx.threshold_filter,
            plugin_config=self._plugin_config(),
            request=req,
            enforce_chat_filter=False,
            reinforce_access=True,
        )
        if not result.success:
            raise ValueError(result.error)
        return {
            "query_type": result.query_type,
            "query": result.query,
            "time_from": time_from,
            "time_to": time_to,
            "top_k": result.top_k,
            "count": result.count,
            "elapsed_ms": result.elapsed_ms,
            "results": SearchExecutionService.to_serializable_results(result.results),
        }

    async def entity(self, *, entity_name: str) -> Dict[str, Any]:
        target = str(entity_name or "").strip()
        if not target:
            raise ValueError("entity_name is empty")
        if not self.ctx.graph_store.has_node(target):
            raise ValueError(f"entity not found: {target}")
        neighbors = self.ctx.graph_store.get_neighbors(target)
        paragraphs = self.ctx.metadata_store.get_paragraphs_by_entity(target)
        relations = (
            self.ctx.metadata_store.get_relations(subject=target)
            + self.ctx.metadata_store.get_relations(object=target)
        )
        return {
            "entity_name": target,
            "neighbors": neighbors,
            "paragraphs": paragraphs,
            "relations": relations,
        }

    async def relation(self, *, subject: str = "", predicate: str = "", obj: str = "") -> Dict[str, Any]:
        rels = self.ctx.metadata_store.get_relations(
            subject=subject or None,
            predicate=predicate or None,
            object=obj or None,
        )
        return {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "count": len(rels),
            "relations": rels,
        }

    async def episode(
        self,
        *,
        query: str = "",
        time_from: Optional[str] = None,
        time_to: Optional[str] = None,
        person: Optional[str] = None,
        source: Optional[str] = None,
        top_k: Optional[int] = None,
        include_paragraphs: bool = False,
    ) -> Dict[str, Any]:
        ts_from, ts_to = parse_query_time_range(time_from, time_to) if (time_from or time_to) else (None, None)
        safe_top_k = max(1, min(50, int(top_k or self.ctx.get_config("retrieval.temporal.default_top_k", 10))))
        results = await self.ctx.episode_retrieval_service.query(
            query=query,
            top_k=safe_top_k,
            time_from=ts_from,
            time_to=ts_to,
            person=person,
            source=source,
            include_paragraphs=include_paragraphs,
        )
        return {
            "query_type": "episode",
            "query": query,
            "time_from": time_from,
            "time_to": time_to,
            "top_k": safe_top_k,
            "count": len(results),
            "results": results,
        }

    async def aggregate(
        self,
        *,
        query: str = "",
        time_from: Optional[str] = None,
        time_to: Optional[str] = None,
        person: Optional[str] = None,
        source: Optional[str] = None,
        top_k: Optional[int] = None,
        mix: bool = True,
        mix_top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        safe_top_k = max(1, min(50, int(top_k or self.ctx.get_config("retrieval.temporal.default_top_k", 10))))

        async def _search_runner() -> Dict[str, Any]:
            payload = await self.search(query=query, top_k=safe_top_k)
            payload["success"] = True
            return payload

        async def _time_runner() -> Dict[str, Any]:
            payload = await self.time_search(
                query=query,
                time_from=time_from,
                time_to=time_to,
                person=person,
                source=source,
                top_k=safe_top_k,
            )
            payload["success"] = True
            return payload

        async def _episode_runner() -> Dict[str, Any]:
            payload = await self.episode(
                query=query,
                time_from=time_from,
                time_to=time_to,
                person=person,
                source=source,
                top_k=safe_top_k,
            )
            payload["success"] = True
            return payload

        return await AggregateQueryService(self.ctx).execute(
            query=query,
            top_k=safe_top_k,
            mix=bool(mix),
            mix_top_k=mix_top_k,
            time_from=time_from,
            time_to=time_to,
            search_runner=_search_runner,
            time_runner=_time_runner if (time_from or time_to) else None,
            episode_runner=_episode_runner,
        )

    async def stats(self) -> Dict[str, Any]:
        vector_stats = {"num_vectors": self.ctx.vector_store.num_vectors, "dimension": self.ctx.vector_store.dimension}
        graph_stats = {"num_nodes": self.ctx.graph_store.num_nodes, "num_edges": self.ctx.graph_store.num_edges}
        metadata_stats = self.ctx.metadata_store.get_statistics()
        return {
            "vector_store": vector_stats,
            "graph_store": graph_stats,
            "metadata_store": metadata_stats,
            "retriever": self.ctx.retriever.get_statistics(),
            "sparse": self.ctx.sparse_index.stats() if self.ctx.sparse_index is not None else None,
        }
