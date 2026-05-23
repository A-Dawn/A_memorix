"""Summary import service wrappers."""

from __future__ import annotations

from typing import Any, Dict, List

from core.utils.summary_importer import SummaryImporter

from amemorix.context import AppContext


class SummaryService:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self.llm_client = self.ctx.llm_client
        plugin_config = dict(self.ctx.config)
        plugin_config["relation_write_service"] = self.ctx.relation_write_service
        self.importer = SummaryImporter(
            vector_store=self.ctx.vector_store,
            graph_store=self.ctx.graph_store,
            metadata_store=self.ctx.metadata_store,
            embedding_manager=self.ctx.embedding_manager,
            plugin_config=plugin_config,
            llm_client=self.llm_client,
        )

    async def import_from_transcript(
        self,
        *,
        session_id: str,
        messages: List[Dict[str, Any]],
        source: str = "",
        context_length: int = 50,
    ) -> Dict[str, Any]:
        ok, msg = await self.importer.import_from_transcript(
            session_id=session_id,
            messages=messages,
            source=source,
            context_length=context_length,
        )
        return {"success": ok, "message": msg}
