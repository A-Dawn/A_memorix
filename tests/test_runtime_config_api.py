from __future__ import annotations

import tomllib
from types import SimpleNamespace

from fastapi.testclient import TestClient

from server import MemorixServer


class ConfigPlugin:
    def __init__(self, *, auth_enabled: bool = True, write_tokens=None, config_path=None):
        self.config = {
            "auth": {
                "enabled": bool(auth_enabled),
                "write_tokens": list(write_tokens or []),
                "read_tokens": [],
                "protect_read_endpoints": False,
            },
            "advanced": {
                "enable_auto_save": True,
                "auto_save_interval_minutes": 5,
                "debug": False,
            },
            "retrieval": {
                "top_k_final": 10,
                "alpha": 0.5,
                "enable_ppr": True,
            },
            "tasks": {"queue_maxsize": 1024},
            "memory": {"enabled": True, "half_life_hours": 24, "prune_threshold": 0.1},
            "episode": {"enabled": True, "generation_enabled": True},
            "person_profile": {"enabled": True, "top_k_evidence": 12},
        }
        self._runtime_auto_save = True
        self.settings = SimpleNamespace(config_path=config_path)

    def get_config(self, key: str, default=None):
        value = self.config
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value


def test_runtime_config_update_applies_whitelisted_values():
    client = TestClient(MemorixServer(ConfigPlugin()).app)

    response = client.patch(
        "/api/config/runtime",
        json={
            "updates": {
                "advanced.enable_auto_save": False,
                "retrieval.alpha": 0.7,
                "tasks.queue_maxsize": 2048,
            }
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["runtime_only"] is True
    assert payload["auto_save_enabled"] is False
    assert payload["config"]["retrieval"]["alpha"] == 0.7
    assert payload["config"]["tasks"]["queue_maxsize"] == 2048


def test_runtime_config_update_rejects_unknown_and_out_of_range_keys():
    client = TestClient(MemorixServer(ConfigPlugin()).app)

    unknown = client.patch("/api/config/runtime", json={"updates": {"auth.enabled": False}})
    out_of_range = client.patch("/api/config/runtime", json={"updates": {"retrieval.alpha": 2}})

    assert unknown.status_code == 400
    assert "Unsupported runtime config key" in unknown.json()["detail"]
    assert out_of_range.status_code == 400
    assert "retrieval.alpha" in out_of_range.json()["detail"]


def test_runtime_config_persist_writes_whitelisted_updates_to_config_file(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[auth]
enabled = true
write_tokens = ["test-token"]

[retrieval]
alpha = 0.2
top_k_final = 8
""".strip()
        + "\n",
        encoding="utf-8",
    )
    plugin = ConfigPlugin(write_tokens=["test-token"], config_path=config_path)
    client = TestClient(MemorixServer(plugin).app)

    response = client.patch(
        "/api/config/runtime",
        json={"persist": True, "updates": {"retrieval.alpha": 0.75, "tasks.queue_maxsize": 2048}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["persisted"] is True
    assert payload["runtime_only"] is False
    assert payload["config_path"] == str(config_path)

    with config_path.open("rb") as handle:
        persisted = tomllib.load(handle)
    assert persisted["auth"]["write_tokens"] == ["test-token"]
    assert persisted["retrieval"]["alpha"] == 0.75
    assert persisted["retrieval"]["top_k_final"] == 8
    assert persisted["tasks"]["queue_maxsize"] == 2048


def test_runtime_config_persist_requires_server_token_auth(tmp_path):
    plugin = ConfigPlugin(auth_enabled=False, write_tokens=["test-token"], config_path=tmp_path / "config.toml")
    client = TestClient(MemorixServer(plugin).app)

    response = client.patch(
        "/api/config/runtime",
        json={"persist": True, "updates": {"retrieval.alpha": 0.75}},
    )

    assert response.status_code == 403
    assert "Token" in response.json()["detail"]
    assert plugin.config["retrieval"]["alpha"] == 0.5
