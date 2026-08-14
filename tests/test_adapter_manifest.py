from __future__ import annotations

from pathlib import Path

import json

import pytest
from pydantic import ValidationError

from a_memorix import (
    AdapterCompatibilityError,
    AdapterManifest,
    adapter_manifest_json_schema,
    ensure_adapter_compatible,
    load_adapter_manifest,
)
from a_memorix.cli import main


REPO_ROOT = Path(__file__).resolve().parents[1]


def _remote_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "id": "community.test-agent",
        "name": "Test Agent Adapter",
        "version": "1.2.3",
        "runtime": "remote",
        "package": "a-memorix-test-agent",
        "core_version": ">=2.0.0a2,<3.0",
        "adapter_protocol": "1",
        "transports": ["grpc"],
        "host_ports": [],
        "license": "AGPL-3.0-only",
        "source": "https://example.com/test-agent",
        "permissions": {
            "api": ["memory.read", "memory.write"],
            "network": ["a-memorix"],
            "filesystem": [],
            "environment": [],
            "subprocess": False,
        },
    }


def test_documented_adapter_manifests_are_valid_and_compatible() -> None:
    for filename in ("adapter-remote.toml", "adapter-in-process.toml"):
        manifest = load_adapter_manifest(REPO_ROOT / "docs" / "examples" / filename)
        ensure_adapter_compatible(manifest, core_version="2.0.0a2")
        assert manifest.schema_version == 1
        assert manifest.adapter_protocol == "1"


def test_remote_adapter_rejects_in_process_capabilities() -> None:
    payload = _remote_manifest()
    payload["entrypoint"] = "test_agent.adapter:create_adapter"
    payload["host_ports"] = ["embedding"]

    with pytest.raises(ValidationError) as error:
        AdapterManifest.model_validate(payload)

    assert "remote adapters cannot declare a Python entrypoint" in str(error.value)


def test_in_process_adapter_requires_package_entrypoint_and_single_transport() -> None:
    payload = _remote_manifest()
    payload.update(
        {
            "runtime": "in_process",
            "package": None,
            "transports": ["in_process", "grpc"],
        }
    )

    with pytest.raises(ValidationError) as error:
        AdapterManifest.model_validate(payload)

    assert "must use only the in_process transport" in str(error.value)


@pytest.mark.parametrize(
    ("field_path", "value", "message"),
    [
        (("version",), "1.0.0-01", "leading zeros"),
        (("transports",), ["grpc", "grpc"], "must not contain duplicates"),
        (
            ("permissions", "network"),
            ["a-memorix", "https://*.example.com"],
            "without credentials or paths",
        ),
        (
            ("permissions", "environment"),
            ["lowercase_secret"],
            "uppercase variable names",
        ),
    ],
)
def test_manifest_rejects_ambiguous_or_unsafe_declarations(
    field_path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    payload = _remote_manifest()
    target = payload
    for field in field_path[:-1]:
        target = target[field]  # type: ignore[assignment,index]
    target[field_path[-1]] = value  # type: ignore[index]

    with pytest.raises(ValidationError) as error:
        AdapterManifest.model_validate(payload)

    assert message in str(error.value)


def test_remote_network_transport_requires_a_memorix_origin() -> None:
    payload = _remote_manifest()
    permissions = payload["permissions"]
    assert isinstance(permissions, dict)
    permissions["network"] = []

    with pytest.raises(ValidationError) as error:
        AdapterManifest.model_validate(payload)

    assert "must declare the a-memorix network origin" in str(error.value)


def test_provider_network_placeholders_are_valid_permissions() -> None:
    payload = _remote_manifest()
    permissions = payload["permissions"]
    assert isinstance(permissions, dict)
    permissions["network"] = [
        "a-memorix",
        "embedding-provider",
        "llm-provider",
    ]

    manifest = AdapterManifest.model_validate(payload)

    assert manifest.permissions.network == (
        "a-memorix",
        "embedding-provider",
        "llm-provider",
    )


def test_adapter_compatibility_uses_the_declared_core_range() -> None:
    manifest = AdapterManifest.model_validate(_remote_manifest())

    with pytest.raises(AdapterCompatibilityError) as error:
        ensure_adapter_compatible(manifest, core_version="3.0.0")

    assert "requires A_memorix >=2.0.0a2,<3.0" in str(error.value)


def test_schema_exposes_required_security_and_runtime_fields() -> None:
    schema = adapter_manifest_json_schema()

    assert schema["title"] == "A_memorix Adapter Manifest v1"
    assert set(schema["required"]) >= {
        "schema_version",
        "runtime",
        "transports",
        "host_ports",
        "permissions",
    }
    assert "AdapterPermissions" in schema["$defs"]


def test_adapter_cli_does_not_load_service_configuration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("A_MEMORIX_CONFIG", "missing-service-config.toml")
    manifest = REPO_ROOT / "docs" / "examples" / "adapter-remote.toml"

    exit_code = main(["adapter", "validate", str(manifest)])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["manifest"]["runtime"] == "remote"


def test_adapter_cli_reports_version_mismatch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = REPO_ROOT / "docs" / "examples" / "adapter-remote.toml"

    exit_code = main(
        ["adapter", "validate", str(manifest), "--core-version", "3.0.0"]
    )

    assert exit_code == 2
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "invalid_argument"
    assert "selected version is 3.0.0" in error["message"]
