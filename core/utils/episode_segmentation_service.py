"""Standalone Episode semantic segmentation service.

The service can use an OpenAI-compatible ``LLMClient`` supplied through
``plugin_config["llm_client"]``. If no client is available, callers should fall
back to deterministic rule-based episode generation.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from amemorix.common.logging import get_logger

logger = get_logger("A_Memorix.EpisodeSegmentationService")


class EpisodeSegmentationService:
    """LLM-backed episode segmentation with strict JSON normalization."""

    SEGMENTATION_VERSION = "episode_mvp_v1"

    def __init__(self, plugin_config: Optional[dict] = None):
        self.plugin_config = plugin_config or {}

    def _cfg(self, key: str, default: Any = None) -> Any:
        current: Any = self.plugin_config
        for part in key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    @staticmethod
    def _clamp_score(value: Any, default: float = 0.0) -> float:
        try:
            num = float(value)
        except Exception:
            num = default
        return max(0.0, min(1.0, num))

    @staticmethod
    def _safe_json_loads(text: str) -> Dict[str, Any]:
        raw = str(text or "").strip()
        if not raw:
            raise ValueError("empty_response")

        if "```" in raw:
            raw = raw.replace("```json", "```").replace("```JSON", "```")
            for part in raw.split("```"):
                part = part.strip()
                if part.startswith("{") and part.endswith("}"):
                    raw = part
                    break

        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(raw[start : end + 1])
            if isinstance(data, dict):
                return data

        raise ValueError("invalid_json_response")

    def _build_prompt(
        self,
        *,
        source: str,
        window_start: Optional[float],
        window_end: Optional[float],
        paragraphs: List[Dict[str, Any]],
    ) -> str:
        rows: List[str] = []
        for idx, item in enumerate(paragraphs, 1):
            p_hash = str(item.get("hash", "") or "").strip()
            content = str(item.get("content", "") or "").strip().replace("\r\n", "\n")
            rows.append(
                (
                    f"[{idx}] hash={p_hash}\n"
                    f"event_time={item.get('event_time')}\n"
                    f"event_time_start={item.get('event_time_start')}\n"
                    f"event_time_end={item.get('event_time_end')}\n"
                    f"content={content[:800]}"
                )
            )

        source_text = str(source or "").strip() or "unknown"
        return (
            "You are an episode segmentation engine.\n"
            "Group the given paragraphs into one or more coherent episodes.\n"
            "Return JSON ONLY. No markdown, no explanation.\n\n"
            "Hard JSON schema:\n"
            "{\n"
            '  "episodes": [\n'
            "    {\n"
            '      "title": "string",\n'
            '      "summary": "string",\n'
            '      "paragraph_hashes": ["hash1", "hash2"],\n'
            '      "participants": ["person1", "person2"],\n'
            '      "keywords": ["kw1", "kw2"],\n'
            '      "time_confidence": 0.0,\n'
            '      "llm_confidence": 0.0\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "1) paragraph_hashes must come from input only.\n"
            "2) title and summary must be non-empty.\n"
            "3) keep participants/keywords concise and deduplicated.\n"
            "4) if uncertain, still provide best effort confidence values.\n\n"
            f"source={source_text}\n"
            f"window_start={window_start}\n"
            f"window_end={window_end}\n"
            "paragraphs:\n"
            + "\n\n".join(rows)
        )

    def _normalize_episodes(
        self,
        *,
        payload: Dict[str, Any],
        input_hashes: List[str],
    ) -> List[Dict[str, Any]]:
        raw_episodes = payload.get("episodes")
        if not isinstance(raw_episodes, list):
            raise ValueError("episodes_missing_or_not_list")

        valid_hashes = set(input_hashes)
        normalized: List[Dict[str, Any]] = []
        for item in raw_episodes:
            if not isinstance(item, dict):
                continue

            title = str(item.get("title", "") or "").strip()
            summary = str(item.get("summary", "") or "").strip()
            raw_hashes = item.get("paragraph_hashes")
            if not title or not summary or not isinstance(raw_hashes, list):
                continue

            paragraph_hashes: List[str] = []
            seen = set()
            for raw_hash in raw_hashes:
                token = str(raw_hash or "").strip()
                if token and token in valid_hashes and token not in seen:
                    seen.add(token)
                    paragraph_hashes.append(token)
            if not paragraph_hashes:
                continue

            participants = [str(x).strip() for x in (item.get("participants") or []) if str(x).strip()]
            keywords = [str(x).strip() for x in (item.get("keywords") or []) if str(x).strip()]
            normalized.append(
                {
                    "title": title,
                    "summary": summary,
                    "paragraph_hashes": paragraph_hashes,
                    "participants": participants[:16],
                    "keywords": keywords[:20],
                    "time_confidence": self._clamp_score(item.get("time_confidence"), default=1.0),
                    "llm_confidence": self._clamp_score(item.get("llm_confidence"), default=0.5),
                }
            )

        if not normalized:
            raise ValueError("episodes_all_invalid")
        return normalized

    async def segment(
        self,
        *,
        source: str,
        window_start: Optional[float],
        window_end: Optional[float],
        paragraphs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not paragraphs:
            raise ValueError("paragraphs_empty")

        llm_client = self.plugin_config.get("llm_client") if isinstance(self.plugin_config, dict) else None
        if llm_client is None:
            raise RuntimeError("episode segmentation llm client unavailable")

        prompt = self._build_prompt(
            source=source,
            window_start=window_start,
            window_end=window_end,
            paragraphs=paragraphs,
        )
        raw = await llm_client.complete(
            prompt,
            temperature=float(self._cfg("episode.segmentation_temperature", 0.2)),
            max_tokens=int(self._cfg("episode.segmentation_max_tokens", 1500)),
        )
        payload = self._safe_json_loads(str(raw))
        input_hashes = [str(p.get("hash", "") or "").strip() for p in paragraphs]
        episodes = self._normalize_episodes(payload=payload, input_hashes=input_hashes)

        return {
            "episodes": episodes,
            "segmentation_model": getattr(llm_client, "model", "standalone_llm"),
            "segmentation_version": self.SEGMENTATION_VERSION,
        }
