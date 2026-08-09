from pathlib import Path

from a_memorix.core.runtime.sdk_memory_kernel import SDKMemoryKernel
from a_memorix.core.utils import feedback_policy


def test_feedback_policy_signal_and_noise_detection() -> None:
    assert feedback_policy.feedback_contains_signal("你记错了，实际是绿色")
    assert not feedback_policy.feedback_contains_signal("好的收到")

    assert feedback_policy.feedback_noise("")
    assert feedback_policy.feedback_noise("好的")
    assert not feedback_policy.feedback_noise("不是绿色，是蓝色")


def test_feedback_policy_config_accessors_clamp_values() -> None:
    config = {
        "feedback_correction": {
            "enabled": True,
            "window_hours": -5,
            "check_interval_minutes": -2,
            "batch_size": -3,
            "auto_apply_threshold": 9,
            "max_feedback_messages": -4,
            "prefilter_enabled": False,
            "paragraph_mark_enabled": False,
            "paragraph_hard_filter_enabled": False,
            "profile_refresh_enabled": False,
            "profile_force_refresh_on_read": False,
            "episode_rebuild_enabled": False,
            "episode_query_block_enabled": False,
            "reconcile_interval_minutes": -5,
            "reconcile_batch_size": -6,
        },
        "fuzzy_modify": {
            "enabled": False,
            "auto_execute_enabled": True,
            "confirm_threshold": 0.72,
            "candidate_limit": -7,
            "max_targets": -8,
            "allow_global_scope": True,
        },
    }

    assert feedback_policy.feedback_cfg_enabled(config)
    assert feedback_policy.feedback_cfg_window_hours(config) == 0.1
    assert feedback_policy.feedback_cfg_check_interval_seconds(config) == 60.0
    assert feedback_policy.feedback_cfg_batch_size(config) == 1
    assert feedback_policy.feedback_cfg_auto_apply_threshold(config) == 1.0
    assert feedback_policy.feedback_cfg_max_messages(config) == 1
    assert not feedback_policy.feedback_cfg_prefilter_enabled(config)
    assert not feedback_policy.feedback_cfg_paragraph_mark_enabled(config)
    assert not feedback_policy.feedback_cfg_paragraph_hard_filter_enabled(config)
    assert not feedback_policy.feedback_cfg_profile_refresh_enabled(config)
    assert not feedback_policy.feedback_cfg_profile_force_refresh_on_read(config)
    assert not feedback_policy.feedback_cfg_episode_rebuild_enabled(config)
    assert not feedback_policy.feedback_cfg_episode_query_block_enabled(config)
    assert feedback_policy.feedback_cfg_reconcile_interval_seconds(config) == 60.0
    assert feedback_policy.feedback_cfg_reconcile_batch_size(config) == 1

    assert not feedback_policy.fuzzy_modify_cfg_enabled(config)
    assert feedback_policy.fuzzy_modify_cfg_auto_execute_enabled(config)
    assert feedback_policy.fuzzy_modify_cfg_confirm_threshold(config) == 0.72
    assert feedback_policy.fuzzy_modify_cfg_candidate_limit(config) == 1
    assert feedback_policy.fuzzy_modify_cfg_max_targets(config) == 1
    assert feedback_policy.fuzzy_modify_cfg_allow_global_scope(config)


def test_feedback_policy_window_label_uses_compact_hour_format() -> None:
    assert feedback_policy.feedback_cfg_window_label({"feedback_correction": {"window_hours": 2}}) == "2h"
    assert feedback_policy.feedback_cfg_window_label({"feedback_correction": {"window_hours": 1.5}}) == "1.50h"


def test_feedback_policy_kernel_and_service_compatibility_wrappers() -> None:
    kernel = SDKMemoryKernel(
        data_dir=Path.cwd(),
        config={
            "feedback_correction": {"enabled": True, "window_hours": 3, "batch_size": 7},
            "fuzzy_modify": {"enabled": True, "candidate_limit": 11, "max_targets": 4},
        },
    )

    assert kernel._feedback_contains_signal("不对，应该是红色")
    assert kernel._feedback_cfg_enabled()
    assert kernel._feedback_cfg_window_hours() == 3.0
    assert kernel._feedback_cfg_batch_size() == 7
    assert kernel._feedback_cfg_window_label() == "3h"
    assert kernel._fuzzy_modify_cfg_candidate_limit() == 11
    assert kernel._fuzzy_modify_cfg_max_targets() == 4

    correction_service = kernel._correction_admin_service
    assert correction_service._fuzzy_modify_cfg_enabled()
    assert correction_service._fuzzy_modify_cfg_candidate_limit() == 11
    assert correction_service._fuzzy_modify_cfg_max_targets() == 4
