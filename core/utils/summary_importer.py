"""
聊天总结与知识导入工具（独立版）。
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from amemorix.common.logging import get_logger
from amemorix.llm_client import LLMClient

from ..embedding.api_adapter import EmbeddingAPIAdapter
from ..storage import (
    GraphStore,
    KnowledgeType,
    MetadataStore,
    VectorStore,
    get_knowledge_type_from_string,
)

logger = get_logger("A_Memorix.SummaryImporter")

SUMMARY_PROMPT_TEMPLATE = """
你需要对聊天记录进行结构化总结，并抽取关键实体与关系。

聊天记录：
{chat_history}

请输出严格 JSON：
{{
  "summary": "对话总结",
  "entities": ["实体1", "实体2"],
  "relations": [
    {{"subject":"实体1","predicate":"关系","object":"实体2"}}
  ]
}}

要求：
1. 总结简洁、客观、可作为长期记忆。
2. 实体与关系尽量使用原文措辞。
3. 如果没有关系，relations 返回空数组。
"""


class SummaryImporter:
    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        metadata_store: MetadataStore,
        embedding_manager: EmbeddingAPIAdapter,
        plugin_config: dict,
        llm_client: Optional[LLMClient] = None,
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.metadata_store = metadata_store
        self.embedding_manager = embedding_manager
        self.plugin_config = plugin_config or {}
        self.llm_client = llm_client

    def _cfg(self, key: str, default: Any = None) -> Any:
        current: Any = self.plugin_config if isinstance(self.plugin_config, dict) else {}
        for part in key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def _build_chat_text(self, messages: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for item in messages:
            role = str(item.get("role", "user") or "user")
            content = str(item.get("content", "") or "").strip()
            if not content:
                continue
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _fallback_summary(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged = " ".join(str(m.get("content", "") or "").strip() for m in messages if str(m.get("content", "")).strip())
        merged = merged[:500]
        return {"summary": merged or "暂无可总结内容", "entities": [], "relations": []}

    async def _generate_summary_payload(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not messages:
            return self._fallback_summary(messages)

        history = self._build_chat_text(messages)
        prompt = SUMMARY_PROMPT_TEMPLATE.format(chat_history=history)

        if self.llm_client is None:
            return self._fallback_summary(messages)

        try:
            ok, payload, raw = await self.llm_client.complete_json(prompt, temperature=0.2, max_tokens=1200)
            if ok and isinstance(payload, dict):
                return payload
            logger.warning("Summary LLM returned non-JSON, fallback parser used.")
            if raw:
                start = raw.find("{")
                end = raw.rfind("}")
                if start >= 0 and end > start:
                    try:
                        parsed = json.loads(raw[start : end + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        pass
        except Exception as exc:
            logger.warning("Summary LLM call failed: %s", exc)

        return self._fallback_summary(messages)

    async def import_from_transcript(
        self,
        *,
        session_id: str,
        messages: List[Dict[str, Any]],
        source: str = "",
        context_length: Optional[int] = None,
    ) -> Tuple[bool, str]:
        try:
            session = self.metadata_store.upsert_transcript_session(
                session_id=session_id,
                source=source or f"transcript:{session_id}",
                metadata={"imported_at": time.time()},
            )
            self.metadata_store.append_transcript_messages(session_id=session["session_id"], messages=messages)

            limit = int(context_length) if context_length is not None else int(self._cfg("summarization.context_length", 50))
            transcript_messages = self.metadata_store.get_transcript_messages(session["session_id"], limit=max(1, limit))
            payload = await self._generate_summary_payload(transcript_messages)

            summary = str(payload.get("summary", "") or "").strip()
            entities = payload.get("entities", [])
            relations = payload.get("relations", [])
            if not summary:
                return False, "总结为空"

            await self._execute_import(
                summary=summary,
                entities=entities if isinstance(entities, list) else [],
                relations=relations if isinstance(relations, list) else [],
                stream_id=session["session_id"],
            )

            self.vector_store.save()
            self.graph_store.save()
            return True, f"总结导入成功: session={session['session_id']}"
        except Exception as exc:
            logger.error("Summary transcript import failed: %s", exc, exc_info=True)
            return False, str(exc)

    async def import_from_stream(
        self,
        stream_id: str,
        context_length: Optional[int] = None,
        include_personality: Optional[bool] = None,
    ) -> Tuple[bool, str]:
        del include_personality
        limit = int(context_length) if context_length is not None else int(self._cfg("summarization.context_length", 50))
        messages = self.metadata_store.get_transcript_messages(stream_id, limit=max(1, limit))
        if not messages:
            return False, "未找到可总结的聊天记录（请先写入 transcript）"
        return await self.import_from_transcript(
            session_id=stream_id,
            messages=messages,
            source=f"chat_summary:{stream_id}",
            context_length=limit,
        )

    async def _execute_import(
        self,
        summary: str,
        entities: List[str],
        relations: List[Dict[str, str]],
        stream_id: str,
        time_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        type_str = self._cfg("summarization.default_knowledge_type", "narrative")
        knowledge_type = get_knowledge_type_from_string(type_str) or KnowledgeType.NARRATIVE

        hash_value = self.metadata_store.add_paragraph(
            content=summary,
            source=f"chat_summary:{stream_id}",
            knowledge_type=knowledge_type.value,
            time_meta=time_meta,
        )

        embedding = await self.embedding_manager.encode(summary)
        self.vector_store.add(vectors=embedding.reshape(1, -1), ids=[hash_value])

        if entities:
            self.graph_store.add_nodes([str(e) for e in entities if str(e).strip()])

        for rel in relations:
            s = str(rel.get("subject", "")).strip()
            p = str(rel.get("predicate", "")).strip()
            o = str(rel.get("object", "")).strip()
            if not (s and p and o):
                continue
            rel_hash = self.metadata_store.add_relation(
                subject=s,
                predicate=p,
                obj=o,
                confidence=1.0,
                source_paragraph=hash_value,
            )
            self.graph_store.add_edges([(s, o)], relation_hashes=[rel_hash])

        logger.info("Summary imported: %s", hash_value[:8])
