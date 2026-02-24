"""Configuration loading for standalone runtime."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import tomllib

from .common.logging import get_logger

logger = get_logger("A_Memorix.Settings")


DEFAULT_CONFIG: Dict[str, Any] = {
    "server": {"host": "0.0.0.0", "port": 8082, "workers": 1},
    "auth": {
        "enabled": True,
        "write_tokens": [],
        "read_tokens": [],
        "protect_read_endpoints": False,
    },
    "cors": {"allow_origins": []},
    "storage": {"data_dir": "./data"},
    "embedding": {
        "dimension": 1024,
        "quantization_type": "int8",
        "batch_size": 32,
        "max_concurrent": 5,
        "model_name": "auto",
        "retry": {"max_attempts": 5, "max_wait_seconds": 30, "min_wait_seconds": 2},
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key": "",
            "model": "text-embedding-3-large",
            "timeout_seconds": 30,
            "max_retries": 3,
        },
    },
    "retrieval": {
        "top_k_relations": 10,
        "top_k_paragraphs": 20,
        "top_k_final": 10,
        "alpha": 0.5,
        "enable_ppr": True,
        "ppr_alpha": 0.85,
        "ppr_concurrency_limit": 4,
        "enable_parallel": True,
        "relation_semantic_fallback": True,
        "relation_fallback_min_score": 0.3,
        "temporal": {
            "enabled": True,
            "allow_created_fallback": True,
            "candidate_multiplier": 8,
            "default_top_k": 10,
            "max_scan": 1000,
        },
        "search": {
            "smart_fallback": {"enabled": True, "threshold": 0.6},
            "safe_content_dedup": {"enabled": True},
        },
        "time": {"skip_threshold_when_query_empty": True},
        "sparse": {
            "enabled": True,
            "backend": "fts5",
            "lazy_load": True,
            "mode": "auto",
            "tokenizer_mode": "jieba",
            "jieba_user_dict": "",
            "char_ngram_n": 2,
            "candidate_k": 80,
            "max_doc_len": 2000,
            "enable_ngram_fallback_index": True,
            "enable_like_fallback": False,
            "enable_relation_sparse_fallback": True,
            "relation_candidate_k": 60,
            "relation_max_doc_len": 512,
            "unload_on_disable": True,
            "shrink_memory_on_unload": True,
        },
        "fusion": {
            "method": "weighted_rrf",
            "rrf_k": 60,
            "vector_weight": 0.7,
            "bm25_weight": 0.3,
            "normalize_score": True,
            "normalize_method": "minmax",
        },
    },
    "threshold": {
        "min_threshold": 0.3,
        "max_threshold": 0.95,
        "percentile": 75.0,
        "std_multiplier": 1.5,
        "min_results": 3,
        "enable_auto_adjust": True,
    },
    "graph": {"sparse_matrix_format": "csr"},
    "advanced": {
        "enable_auto_save": True,
        "auto_save_interval_minutes": 5,
        "debug": False,
    },
    "memory": {
        "enabled": True,
        "half_life_hours": 24.0,
        "base_decay_interval_hours": 1.0,
        "prune_threshold": 0.1,
        "freeze_duration_hours": 24.0,
        "enable_auto_reinforce": True,
        "reinforce_buffer_max_size": 1000,
        "reinforce_cooldown_hours": 1.0,
        "max_weight": 10.0,
        "revive_boost_weight": 0.5,
        "auto_protect_ttl_hours": 24.0,
    },
    "summarization": {
        "enabled": True,
        "model_name": "auto",
        "context_length": 50,
        "include_personality": True,
        "default_knowledge_type": "narrative",
    },
    "person_profile": {
        "enabled": True,
        "opt_in_required": True,
        "default_injection_enabled": False,
        "profile_ttl_minutes": 360.0,
        "refresh_interval_minutes": 30,
        "active_window_hours": 72.0,
        "max_refresh_per_cycle": 50,
        "top_k_evidence": 12,
        "registry": {
            "page_size_default": 20,
            "page_size_max": 100,
            "match_strategy": "contains",
        },
    },
    "filter": {"enabled": True, "mode": "whitelist", "chats": []},
    "routing": {
        "search_owner": "action",
        "tool_search_mode": "forward",
        "enable_request_dedup": True,
        "request_dedup_ttl_seconds": 2,
    },
    "tasks": {
        "import_workers": 1,
        "summary_workers": 1,
        "queue_maxsize": 1024,
        "summary_poll_interval_seconds": 1,
    },
}


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _parse_env_value(raw: str) -> Any:
    text = raw.strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return text


def _set_nested(config: Dict[str, Any], path: list[str], value: Any) -> None:
    cur: Dict[str, Any] = config
    for key in path[:-1]:
        existing = cur.get(key)
        if not isinstance(existing, dict):
            existing = {}
            cur[key] = existing
        cur = existing
    cur[path[-1]] = value


def _apply_env_overrides(config: Dict[str, Any], prefix: str = "AMEMORIX__") -> Dict[str, Any]:
    out = copy.deepcopy(config)
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        tail = env_key[len(prefix) :].strip()
        if not tail:
            continue
        parts = [p.lower() for p in tail.split("__") if p.strip()]
        if not parts:
            continue
        _set_nested(out, parts, _parse_env_value(env_value))
    return out


def _overlay_non_empty(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _overlay_non_empty(out[key], value)
            continue
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        out[key] = value
    return out


def _first_non_empty_env(keys: list[str]) -> Optional[str]:
    for key in keys:
        value = str(os.getenv(key, "") or "").strip()
        if value:
            return value
    return None


def resolve_openapi_endpoint_config(config: Dict[str, Any], *, section: str = "embedding") -> Dict[str, Any]:
    """
    Resolve OpenAI-compatible endpoint config.

    Compatibility rules:
    - Preferred: `[embedding.openapi]`
    - Legacy compatible: `[embedding.openai]`
    - Env aliases: `OPENAPI_*` first, then `OPENAI_*`
    """
    root = config.get(section, {}) if isinstance(config, dict) else {}
    if not isinstance(root, dict):
        root = {}

    legacy_cfg = root.get("openai", {})
    if not isinstance(legacy_cfg, dict):
        legacy_cfg = {}

    openapi_cfg = root.get("openapi", {})
    if not isinstance(openapi_cfg, dict):
        openapi_cfg = {}

    # Start from legacy config, then apply non-empty openapi overrides.
    merged = _overlay_non_empty(legacy_cfg, openapi_cfg)

    env_aliases: Dict[str, list[str]] = {
        "base_url": ["OPENAPI_BASE_URL", "OPENAI_BASE_URL"],
        "api_key": ["OPENAPI_API_KEY", "OPENAI_API_KEY"],
        "model": [
            "OPENAPI_EMBEDDING_MODEL",
            "OPENAI_EMBEDDING_MODEL",
            "OPENAPI_MODEL",
            "OPENAI_MODEL",
        ],
        "chat_model": [
            "OPENAPI_CHAT_MODEL",
            "OPENAI_CHAT_MODEL",
            "OPENAPI_MODEL",
            "OPENAI_MODEL",
        ],
        "timeout_seconds": ["OPENAPI_TIMEOUT_SECONDS", "OPENAI_TIMEOUT_SECONDS"],
        "max_retries": ["OPENAPI_MAX_RETRIES", "OPENAI_MAX_RETRIES"],
    }

    for field, keys in env_aliases.items():
        current = merged.get(field)
        missing = current is None or (isinstance(current, str) and not current.strip())
        if not missing:
            continue
        env_value = _first_non_empty_env(keys)
        if env_value is not None:
            merged[field] = _parse_env_value(env_value)

    # Sensible defaults for OpenAI-compatible providers.
    if not str(merged.get("base_url", "") or "").strip():
        merged["base_url"] = "https://api.openai.com/v1"
    if "timeout_seconds" not in merged:
        merged["timeout_seconds"] = 30
    if "max_retries" not in merged:
        merged["max_retries"] = 3
    return merged


def mask_sensitive(config: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(config)

    def _mask(value: Any) -> str:
        text = str(value or "")
        if len(text) <= 4:
            return "*" * len(text)
        return f"{text[:2]}***{text[-2:]}"

    auth = out.get("auth", {})
    if isinstance(auth, dict):
        for key in ("write_tokens", "read_tokens"):
            values = auth.get(key)
            if isinstance(values, list):
                auth[key] = [_mask(v) for v in values]
    emb = out.get("embedding", {})
    if isinstance(emb, dict):
        for key in ("openai", "openapi"):
            endpoint_cfg = emb.get(key, {})
            if isinstance(endpoint_cfg, dict) and "api_key" in endpoint_cfg:
                endpoint_cfg["api_key"] = _mask(endpoint_cfg["api_key"])
    return out


@dataclass(slots=True)
class AppSettings:
    """Resolved runtime settings."""

    config: Dict[str, Any]
    config_path: Optional[Path] = None

    @classmethod
    def load(cls, path: Optional[str] = None) -> "AppSettings":
        base = copy.deepcopy(DEFAULT_CONFIG)
        resolved_path: Optional[Path] = None

        if path:
            resolved_path = Path(path).expanduser().resolve()
            if not resolved_path.exists():
                raise FileNotFoundError(f"Config file not found: {resolved_path}")
            with resolved_path.open("rb") as f:
                parsed = tomllib.load(f)
            base = _deep_merge(base, parsed)
        else:
            default_path = Path.cwd() / "config.toml"
            if default_path.exists():
                resolved_path = default_path
                with default_path.open("rb") as f:
                    parsed = tomllib.load(f)
                base = _deep_merge(base, parsed)

        resolved = _apply_env_overrides(base)
        logger.info(
            "Settings loaded: config_path=%s",
            str(resolved_path) if resolved_path else "<defaults/env>",
        )
        return cls(config=resolved, config_path=resolved_path)

    def get(self, key: str, default: Any = None) -> Any:
        current: Any = self.config
        for part in key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def get_openapi_endpoint_config(self) -> Dict[str, Any]:
        return resolve_openapi_endpoint_config(self.config)

    @property
    def host(self) -> str:
        return str(self.get("server.host", "0.0.0.0"))

    @property
    def port(self) -> int:
        try:
            return int(self.get("server.port", 8082))
        except (TypeError, ValueError):
            return 8082

    @property
    def workers(self) -> int:
        return 1

    @property
    def data_dir(self) -> Path:
        raw = str(self.get("storage.data_dir", "./data"))
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path
