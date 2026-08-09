"""Feedback-correction policy derived from an explicit configuration mapping."""

from __future__ import annotations

from typing import Any, Mapping


def _cfg(config: Mapping[str, Any] | None, key: str, default: Any) -> Any:
    if not isinstance(config, Mapping):
        return default
    section = config.get("feedback_correction", config)
    if isinstance(section, Mapping) and key in section:
        return section[key]
    legacy_key = f"feedback_correction_{key}"
    return config.get(legacy_key, default)


def _fuzzy_cfg(config: Mapping[str, Any] | None, key: str, default: Any) -> Any:
    if not isinstance(config, Mapping):
        return default
    section = config.get("fuzzy_modify", config)
    if isinstance(section, Mapping) and key in section:
        return section[key]
    return config.get(f"fuzzy_modify_{key}", default)


def feedback_signal_tokens() -> tuple[str, ...]:
    return ("不对", "错了", "你记错", "记错了", "不是", "并不是", "纠正", "更正", "改成", "应该是", "实际是", "说反了")


def feedback_contains_signal(text: str) -> bool:
    content = str(text or "").strip().lower()
    return bool(content) and any(token in content for token in feedback_signal_tokens())


def feedback_noise(text: str) -> bool:
    content = str(text or "").strip()
    if not content:
        return True
    if feedback_contains_signal(content):
        return False
    markers = ("哈哈", "好的", "收到", "谢谢", "嗯嗯", "晚安", "早安", "拜拜", "在吗")
    return len(content) <= 2 or (len(content) <= 8 and any(marker in content for marker in markers))


def feedback_cfg_enabled(config: Mapping[str, Any] | None = None) -> bool:
    return bool(_cfg(config, "enabled", False))


def feedback_cfg_window_hours(config: Mapping[str, Any] | None = None) -> float:
    return max(0.1, float(_cfg(config, "window_hours", 12.0) or 12.0))


def feedback_cfg_check_interval_seconds(config: Mapping[str, Any] | None = None) -> float:
    return float(max(1, int(_cfg(config, "check_interval_minutes", 30) or 30))) * 60.0


def feedback_cfg_batch_size(config: Mapping[str, Any] | None = None) -> int:
    return max(1, int(_cfg(config, "batch_size", 20) or 20))


def feedback_cfg_auto_apply_threshold(config: Mapping[str, Any] | None = None) -> float:
    return min(1.0, max(0.0, float(_cfg(config, "auto_apply_threshold", 0.85))))


def feedback_cfg_max_messages(config: Mapping[str, Any] | None = None) -> int:
    return max(1, int(_cfg(config, "max_feedback_messages", 30) or 30))


def feedback_cfg_prefilter_enabled(config: Mapping[str, Any] | None = None) -> bool:
    return bool(_cfg(config, "prefilter_enabled", True))


def feedback_cfg_paragraph_mark_enabled(config: Mapping[str, Any] | None = None) -> bool:
    return bool(_cfg(config, "paragraph_mark_enabled", True))


def feedback_cfg_paragraph_hard_filter_enabled(config: Mapping[str, Any] | None = None) -> bool:
    return bool(_cfg(config, "paragraph_hard_filter_enabled", True))


def feedback_cfg_profile_refresh_enabled(config: Mapping[str, Any] | None = None) -> bool:
    return bool(_cfg(config, "profile_refresh_enabled", True))


def feedback_cfg_profile_force_refresh_on_read(config: Mapping[str, Any] | None = None) -> bool:
    return bool(_cfg(config, "profile_force_refresh_on_read", True))


def feedback_cfg_episode_rebuild_enabled(config: Mapping[str, Any] | None = None) -> bool:
    return bool(_cfg(config, "episode_rebuild_enabled", True))


def feedback_cfg_episode_query_block_enabled(config: Mapping[str, Any] | None = None) -> bool:
    return bool(_cfg(config, "episode_query_block_enabled", True))


def feedback_cfg_reconcile_interval_seconds(config: Mapping[str, Any] | None = None) -> float:
    return float(max(1, int(_cfg(config, "reconcile_interval_minutes", 5) or 5))) * 60.0


def feedback_cfg_reconcile_batch_size(config: Mapping[str, Any] | None = None) -> int:
    return max(1, int(_cfg(config, "reconcile_batch_size", 20) or 20))


def feedback_cfg_window_label(config: Mapping[str, Any] | None = None) -> str:
    hours = feedback_cfg_window_hours(config)
    return f"{int(round(hours))}h" if abs(hours - round(hours)) < 1e-9 else f"{hours:.2f}h"


def fuzzy_modify_cfg_enabled(config: Mapping[str, Any] | None = None) -> bool:
    return bool(_fuzzy_cfg(config, "enabled", True))


def fuzzy_modify_cfg_auto_execute_enabled(config: Mapping[str, Any] | None = None) -> bool:
    return bool(_fuzzy_cfg(config, "auto_execute_enabled", False))


def fuzzy_modify_cfg_confirm_threshold(config: Mapping[str, Any] | None = None) -> float:
    return float(_fuzzy_cfg(config, "confirm_threshold", 0.85))


def fuzzy_modify_cfg_candidate_limit(config: Mapping[str, Any] | None = None) -> int:
    return max(1, int(_fuzzy_cfg(config, "candidate_limit", 20) or 20))


def fuzzy_modify_cfg_max_targets(config: Mapping[str, Any] | None = None) -> int:
    return max(1, int(_fuzzy_cfg(config, "max_targets", 5) or 5))


def fuzzy_modify_cfg_allow_global_scope(config: Mapping[str, Any] | None = None) -> bool:
    return bool(_fuzzy_cfg(config, "allow_global_scope", False))
