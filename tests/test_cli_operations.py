from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import urlopen

import asyncio
import json
import logging
import os
import socket
import subprocess
import sys

import grpc
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from pydantic import ValidationError

from a_memorix.cli import _parser, _server_config, main
from a_memorix.client import AMemorixClient
from a_memorix.config import ObservabilityConfig, ServerTLSConfig, load_config
from a_memorix.engine import AMemorixEngine
from a_memorix.logging import configure_logging
from a_memorix.observability import ObservabilityRuntime
from a_memorix.providers import build_configured_providers
from a_memorix.server import AMemorixGrpcServer


def test_config_precedence_relative_paths_and_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config" / "a-memorix.toml"
    config_path.parent.mkdir()
    config_path.write_text(
        """
[server]
data_dir = "../state"
port = 51000
admin_token_file = "admin-token"

[client]
target = "file.example:50051"

[observability]
log_format = "text"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("A_MEMORIX_CONFIG", str(config_path))
    monkeypatch.setenv("A_MEMORIX_GRPC_PORT", "52000")
    monkeypatch.setenv("A_MEMORIX_CLIENT_TARGET", "env.example:50051")
    monkeypatch.setenv("A_MEMORIX_CLIENT_TLS_ENABLED", "true")
    monkeypatch.setenv("A_MEMORIX_ADMIN_TOKEN", "secret-value-that-must-not-be-rendered")

    config = load_config()

    assert config.server.port == 52000
    assert config.server.data_dir == (tmp_path / "state").resolve()
    assert config.server.admin_token_file == (config_path.parent / "admin-token").resolve()
    assert config.client.target == "env.example:50051"
    assert config.client.tls.enabled is True
    assert "secret-value-that-must-not-be-rendered" not in json.dumps(config.redacted())

    args = _parser().parse_args(["serve", "--port", "70000"])
    with pytest.raises(ValidationError):
        _server_config(config.server, args)


def test_config_rejects_unknown_fields_and_incomplete_tls(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.toml"
    config_path.write_text("[server]\nunknown = true\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(config_path, environ={})
    with pytest.raises(ValidationError):
        ServerTLSConfig(client_ca=tmp_path / "ca.pem")


def test_provider_config_precedence_relative_secrets_and_redaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config" / "a-memorix.toml"
    config_path.parent.mkdir()
    config_path.write_text(
        """
[providers.embedding]
endpoint = "https://file.example/v1"
model = "embedding-file"
api_key_file = "embedding-key"
dimension = 768

[providers.llm]
endpoint = "https://llm.example/v1"
model = "llm-file"
api_key_file = "llm-key"

[mcp]
mode = "degraded"
probe_llm = false
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("A_MEMORIX_EMBEDDING_ENDPOINT", "https://env.example/v1")
    monkeypatch.setenv("A_MEMORIX_EMBEDDING_MODEL", "embedding-env")
    monkeypatch.setenv("A_MEMORIX_EMBEDDING_DIMENSION", "1024")
    monkeypatch.setenv("A_MEMORIX_EMBEDDING_API_KEY", "embedding-secret")
    monkeypatch.setenv("A_MEMORIX_LLM_API_KEY", "llm-secret")
    monkeypatch.setenv("A_MEMORIX_MCP_MODE", "standard")
    monkeypatch.setenv("A_MEMORIX_MCP_PROBE_LLM", "true")

    config = load_config(config_path)
    providers = build_configured_providers(config.providers)

    assert config.providers.embedding.endpoint == "https://env.example/v1"
    assert config.providers.embedding.model == "embedding-env"
    assert config.providers.embedding.dimension == 1024
    assert config.providers.embedding.api_key_file == (
        config_path.parent / "embedding-key"
    ).resolve()
    assert config.providers.llm.api_key_file == (
        config_path.parent / "llm-key"
    ).resolve()
    assert config.mcp.mode == "standard"
    assert config.mcp.probe_llm is True
    assert providers.embedding is not None
    assert providers.llm is not None
    rendered = json.dumps(config.redacted()) + json.dumps(
        providers.embedding.fingerprint()
    )
    assert "embedding-secret" not in rendered
    assert "llm-secret" not in rendered


def test_cli_can_disable_non_secret_environment_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "a-memorix.toml"
    config_path.write_text(
        """
[providers.embedding]
endpoint = "https://file.example/v1"
model = "embedding-file"
dimension = 768
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("A_MEMORIX_EMBEDDING_ENDPOINT", "https://env.example/v1")
    monkeypatch.setenv("A_MEMORIX_EMBEDDING_MODEL", "embedding-env")
    monkeypatch.setenv("A_MEMORIX_EMBEDDING_DIMENSION", "1024")
    monkeypatch.setenv("A_MEMORIX_EMBEDDING_API_KEY", "embedding-secret")

    result = main(
        [
            "--config",
            str(config_path),
            "--no-environment-overrides",
            "config",
        ]
    )

    assert result == 0
    rendered = capsys.readouterr().out
    payload = json.loads(rendered)
    assert payload["providers"]["embedding"]["endpoint"] == "https://file.example/v1"
    assert payload["providers"]["embedding"]["model"] == "embedding-file"
    assert payload["providers"]["embedding"]["dimension"] == 768
    assert "embedding-secret" not in rendered


def test_mcp_parser_supports_explicit_degraded_mode() -> None:
    args = _parser().parse_args(
        [
            "mcp",
            "--namespace",
            "agent",
            "--mode",
            "degraded",
            "--no-probe-llm",
        ]
    )

    assert args.mode == "degraded"
    assert args.probe_llm is False


def test_standard_mcp_rejects_missing_providers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from a_memorix.cli import _run_mcp
    from a_memorix.core.storage import vector_store

    monkeypatch.setattr(vector_store, "HAS_FAISS", True)
    config = load_config(environ={})
    args = _parser().parse_args(
        [
            "mcp",
            "--namespace",
            "agent",
            "--data-dir",
            str(tmp_path),
        ]
    )

    with pytest.raises(RuntimeError, match="requires an Embedding endpoint"):
        _run_mcp(config, args)


def test_json_logging_is_machine_readable(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", "json")
    logging.getLogger("a_memorix.test").info(
        "request complete",
        extra={"namespace_id": "tenant-a", "duration_ms": 1.25},
    )

    payload = json.loads(capsys.readouterr().err)
    assert payload["level"] == "INFO"
    assert payload["namespace_id"] == "tenant-a"
    assert payload["duration_ms"] == 1.25


@pytest.mark.asyncio
async def test_standard_health_and_prometheus_metrics(tmp_path: Path) -> None:
    metrics_port = _free_port()
    telemetry = ObservabilityRuntime(
        ObservabilityConfig(
            access_log=False,
            metrics_port=metrics_port,
        )
    )
    server = AMemorixGrpcServer(
        AMemorixEngine(data_dir=tmp_path / "data"),
        port=0,
        allow_unauthenticated=True,
        observability=telemetry,
    )
    await server.start()
    try:
        async with AMemorixClient(server.target) as client:
            assert await client.check_health() == "SERVING"
        metrics = await asyncio.to_thread(
            lambda: urlopen(
                f"http://127.0.0.1:{metrics_port}/metrics",
                timeout=5,
            ).read().decode("utf-8")
        )
        assert "a_memorix_rpc_requests_total" in metrics
        assert "grpc.health.v1.Health/Check" in metrics
    finally:
        await server.stop(0)


@pytest.mark.asyncio
async def test_grpc_mutual_tls_health_check(tmp_path: Path) -> None:
    ca_cert, ca_key = _certificate_authority()
    server_cert, server_key = _issued_certificate(
        ca_cert,
        ca_key,
        "localhost",
        server=True,
    )
    client_cert, client_key = _issued_certificate(
        ca_cert,
        ca_key,
        "a-memorix-client",
        server=False,
    )
    ca_pem = ca_cert.public_bytes(serialization.Encoding.PEM)
    credentials = grpc.ssl_server_credentials(
        ((_private_key_pem(server_key), server_cert.public_bytes(serialization.Encoding.PEM)),),
        root_certificates=ca_pem,
        require_client_auth=True,
    )
    server = AMemorixGrpcServer(
        AMemorixEngine(data_dir=tmp_path / "tls-data"),
        port=0,
        admin_token="admin-token-for-mutual-tls-test-0001",
        credentials=credentials,
    )
    await server.start()
    try:
        channel_credentials = grpc.ssl_channel_credentials(
            root_certificates=ca_pem,
            private_key=_private_key_pem(client_key),
            certificate_chain=client_cert.public_bytes(serialization.Encoding.PEM),
        )
        async with AMemorixClient(
            server.target,
            credentials=channel_credentials,
            tls_server_name="localhost",
        ) as client:
            assert await client.check_health() == "SERVING"
    finally:
        await server.stop(0)


@pytest.mark.asyncio
async def test_cli_rpc_workflow_including_backup_restore(tmp_path: Path) -> None:
    server = AMemorixGrpcServer(
        AMemorixEngine(data_dir=tmp_path / "service-data"),
        port=0,
        allow_unauthenticated=True,
    )
    await server.start()
    backup_file = tmp_path / "tenant.amxbackup"
    try:
        created = await _run_cli(
            server.target,
            "namespace",
            "create",
            "cli-tenant",
            "--allow-metadata-only-write",
            "--sparse-retrieval",
        )
        assert created["namespace"]["namespace_id"] == "cli-tenant"

        ingested = await _run_cli(
            server.target,
            "memory",
            "ingest",
            "cli-tenant",
            "--text",
            "A_memorix CLI stores isolated memories",
            "--source-type",
            "document",
            "--external-id",
            "document:cli",
        )
        assert ingested["stored_ids"]
        searched = await _run_cli(
            server.target,
            "memory",
            "search",
            "cli-tenant",
            "--query",
            "isolated memories",
        )
        assert any("isolated memories" in hit["content"] for hit in searched["hits"])

        doctor = await _run_cli(server.target, "doctor", "--health-only")
        assert doctor["healthy"] is True
        await _run_cli(server.target, "namespace", "disable", "cli-tenant")
        backup = await _run_cli(server.target, "backup", "create", "cli-tenant")
        backup_id = backup["backup"]["backup_id"]
        downloaded = await _run_cli(
            server.target,
            "backup",
            "download",
            backup_id,
            str(backup_file),
        )
        assert Path(downloaded["destination"]) == backup_file.resolve()
        assert backup_file.is_file()
        uploaded = await _run_cli(
            server.target,
            "backup",
            "upload",
            str(backup_file),
        )
        assert uploaded["backup"]["sha256"] == backup["backup"]["sha256"]
        restored = await _run_cli(
            server.target,
            "backup",
            "restore",
            backup_id,
            "cli-restored",
        )
        assert restored["namespace"]["namespace_id"] == "cli-restored"
    finally:
        await server.stop(0)


async def _run_cli(target: str, group: str, *arguments: str) -> dict[str, object]:
    environment = dict(os.environ)
    source_root = str(Path(__file__).parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_root, environment.get("PYTHONPATH", "")) if part
    )
    command = [
        sys.executable,
        "-m",
        "a_memorix.cli",
        group,
        "--target",
        target,
        *arguments,
    ]
    process = await asyncio.to_thread(
        subprocess.run,
        command,
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _certificate_authority() -> tuple[x509.Certificate, rsa.RSAPrivateKey]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "A_memorix test CA")])
    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    return certificate, key


def _issued_certificate(
    ca_certificate: x509.Certificate,
    ca_key: rsa.RSAPrivateKey,
    common_name: str,
    *,
    server: bool,
) -> tuple[x509.Certificate, rsa.RSAPrivateKey]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    now = datetime.now(timezone.utc)
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_certificate.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
    )
    if server:
        builder = builder.add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]),
            critical=False,
        )
    certificate = builder.sign(ca_key, hashes.SHA256())
    return certificate, key


def _private_key_pem(key: rsa.RSAPrivateKey) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
