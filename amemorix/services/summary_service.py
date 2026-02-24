"""Summary import service wrappers."""

from __future__ import annotations

from typing import Any, Dict, List

from core.utils.summary_importer import SummaryImporter

from amemorix.context import AppContext
from amemorix.llm_client import LLMClient
from amemorix.settings import resolve_openapi_endpoint_config


class SummaryService:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        endpoint_cfg = resolve_openapi_endpoint_config(self.ctx.config, section="embedding")
        summarization_model = str(self.ctx.get_config("summarization.model_name", "") or "").strip()
        if summarization_model.lower() == "auto":
            summarization_model = ""
        self.llm_client = LLMClient(
            base_url=str(endpoint_cfg.get("base_url", "")),
            api_key=str(endpoint_cfg.get("api_key", "")),
            model=str(
                summarization_model
                or endpoint_cfg.get("chat_model", "")
                or endpoint_cfg.get("model", "")
                or "gpt-4o-mini"
            ),
            timeout_seconds=float(endpoint_cfg.get("timeout_seconds", 60) or 60),
            max_retries=int(endpoint_cfg.get("max_retries", 3) or 3),
        )
        self.importer = SummaryImporter(
            vector_store=self.ctx.vector_store,
            graph_store=self.ctx.graph_store,
            metadata_store=self.ctx.metadata_store,
            embedding_manager=self.ctx.embedding_manager,
            plugin_config=self.ctx.config,
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
