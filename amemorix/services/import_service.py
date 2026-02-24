"""Import orchestration service."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.storage import KnowledgeType, detect_knowledge_type
from core.utils.time_parser import normalize_time_meta

from amemorix.common.logging import get_logger
from amemorix.context import AppContext

logger = get_logger("A_Memorix.ImportService")


class ImportService:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx

    async def run_import(self, mode: str, payload: Any, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        mode = str(mode or "").strip().lower() or "text"
        options = options or {}

        if mode == "text":
            text = payload if isinstance(payload, str) else str(payload or "")
            return await self.import_text(text=text, source=str(options.get("source", "v1_import")))
        if mode == "paragraph":
            if isinstance(payload, dict):
                return await self.import_paragraph(
                    content=str(payload.get("content", "")),
                    source=str(payload.get("source", options.get("source", "v1_import"))),
                    knowledge_type=str(payload.get("knowledge_type", "")),
                    time_meta=payload.get("time_meta"),
                )
            return await self.import_paragraph(content=str(payload or ""), source=str(options.get("source", "v1_import")))
        if mode == "relation":
            if isinstance(payload, dict):
                return await self.import_relation(
                    subject=str(payload.get("subject", "")),
                    predicate=str(payload.get("predicate", "")),
                    obj=str(payload.get("object", "")),
                    confidence=float(payload.get("confidence", 1.0) or 1.0),
                    source_paragraph=str(payload.get("source_paragraph", "")),
                )
            return await self.import_relation_from_text(str(payload or ""))
        if mode == "json":
            return await self.import_json(payload)
        if mode == "file":
            return await self.import_file(str(payload or ""))
        raise ValueError(f"unsupported import mode: {mode}")

    async def import_text(self, *, text: str, source: str = "v1_import") -> Dict[str, Any]:
        paragraphs = self._split_text(text)
        imported = 0
        hashes: List[str] = []
        for para in paragraphs:
            result = await self.import_paragraph(content=para, source=source)
            imported += 1
            hashes.append(str(result["hash"]))
        self.ctx.vector_store.save()
        self.ctx.graph_store.save()
        return {"mode": "text", "paragraphs": imported, "hashes": hashes}

    async def import_paragraph(
        self,
        *,
        content: str,
        source: str = "v1_import",
        knowledge_type: str = "",
        time_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        text = str(content or "").strip()
        if not text:
            raise ValueError("paragraph content is empty")

        if knowledge_type:
            resolved_kt = str(knowledge_type).strip().lower()
        else:
            resolved_kt = detect_knowledge_type(text).value

        hash_value = self.ctx.metadata_store.add_paragraph(
            content=text,
            source=source,
            knowledge_type=resolved_kt if resolved_kt else KnowledgeType.MIXED.value,
            time_meta=normalize_time_meta(time_meta or {}),
        )
        embedding = await self.ctx.embedding_manager.encode(text)
        self.ctx.vector_store.add(vectors=embedding.reshape(1, -1), ids=[hash_value])
        return {"mode": "paragraph", "hash": hash_value, "knowledge_type": resolved_kt}

    async def import_relation_from_text(self, text: str) -> Dict[str, Any]:
        if "|" in text:
            parts = [p.strip() for p in text.split("|")]
            if len(parts) != 3:
                raise ValueError("relation must be 'subject|predicate|object'")
            return await self.import_relation(subject=parts[0], predicate=parts[1], obj=parts[2])

        parts = text.split(maxsplit=2)
        if len(parts) != 3:
            raise ValueError("relation must be 'subject predicate object'")
        return await self.import_relation(subject=parts[0], predicate=parts[1], obj=parts[2])

    async def import_relation(
        self,
        *,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source_paragraph: str = "",
    ) -> Dict[str, Any]:
        s = str(subject or "").strip()
        p = str(predicate or "").strip()
        o = str(obj or "").strip()
        if not (s and p and o):
            raise ValueError("relation subject/predicate/object cannot be empty")

        self.ctx.graph_store.add_nodes([s, o])
        rel_hash = self.ctx.metadata_store.add_relation(
            subject=s,
            predicate=p,
            obj=o,
            confidence=float(confidence),
            source_paragraph=source_paragraph,
        )
        self.ctx.graph_store.add_edges([(s, o)], weights=[float(confidence)], relation_hashes=[rel_hash])

        relation_text = f"{s} {p} {o}"
        rel_embedding = await self.ctx.embedding_manager.encode(relation_text)
        self.ctx.vector_store.add(vectors=rel_embedding.reshape(1, -1), ids=[rel_hash])
        return {"mode": "relation", "hash": rel_hash, "subject": s, "predicate": p, "object": o}

    async def import_json(self, payload: Any) -> Dict[str, Any]:
        if isinstance(payload, str):
            text = payload.strip()
            if text.startswith("{") or text.startswith("["):
                data = json.loads(text)
            else:
                path = Path(text)
                if not path.exists():
                    raise FileNotFoundError(f"json file not found: {path}")
                data = json.loads(path.read_text(encoding="utf-8"))
        elif isinstance(payload, dict):
            data = payload
        else:
            raise ValueError("json payload must be dict or json text/file path")

        paragraph_count = 0
        relation_count = 0
        hashes: List[str] = []

        for item in data.get("paragraphs", []) if isinstance(data, dict) else []:
            if isinstance(item, str):
                result = await self.import_paragraph(content=item, source="v1_json")
            else:
                result = await self.import_paragraph(
                    content=str(item.get("content", "")),
                    source=str(item.get("source", "v1_json")),
                    knowledge_type=str(item.get("knowledge_type", "")),
                    time_meta=item.get("time_meta")
                    or {
                        "event_time": item.get("event_time"),
                        "event_time_start": item.get("event_time_start"),
                        "event_time_end": item.get("event_time_end"),
                        "time_range": item.get("time_range"),
                        "time_granularity": item.get("time_granularity"),
                        "time_confidence": item.get("time_confidence"),
                    },
                )
            paragraph_count += 1
            hashes.append(str(result["hash"]))

        for rel in data.get("relations", []) if isinstance(data, dict) else []:
            await self.import_relation(
                subject=str(rel.get("subject", "")),
                predicate=str(rel.get("predicate", "")),
                obj=str(rel.get("object", "")),
                confidence=float(rel.get("confidence", 1.0) or 1.0),
                source_paragraph=str(rel.get("source_paragraph", "")),
            )
            relation_count += 1

        self.ctx.vector_store.save()
        self.ctx.graph_store.save()
        return {
            "mode": "json",
            "paragraphs": paragraph_count,
            "relations": relation_count,
            "hashes": hashes,
        }

    async def import_file(self, path_str: str) -> Dict[str, Any]:
        path = Path(path_str).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"file not found: {path}")

        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return await self.import_text(text=path.read_text(encoding="utf-8"), source=f"file:{path.name}")
        if suffix == ".json":
            return await self.import_json(path.read_text(encoding="utf-8"))
        raise ValueError(f"unsupported file suffix: {suffix}")

    def _split_text(self, text: str, max_length: int = 500) -> List[str]:
        paragraphs = str(text or "").split("\n\n")
        out: List[str] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) <= max_length:
                out.append(para)
                continue

            sentences = re.split(r"[。！？.!?]", para)
            current = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(current) + len(sentence) + 1 <= max_length:
                    current = f"{current}{sentence}。"
                else:
                    if current:
                        out.append(current.strip())
                    current = f"{sentence}。"
            if current:
                out.append(current.strip())
        return out

