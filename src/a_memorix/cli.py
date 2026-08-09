"""Unified command line interface for service and remote administration."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import argparse
import asyncio
import hashlib
import json
import os
import signal
import sys

from pydantic import ValidationError

from a_memorix.config import (
    AMemorixConfig,
    ClientConfig,
    ServerConfig,
    load_config,
    read_secret,
)
from a_memorix.contracts import AMemorixError, ErrorCode
from a_memorix import (
    __version__,
    adapter_manifest_json_schema,
    ensure_adapter_compatible,
    load_adapter_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "adapter":
            result = _adapter_command(args)
            _print_value(result, pretty=args.pretty)
            return 0
        config = load_config(args.config)
        if args.command == "config":
            _print_value(config.redacted(), pretty=args.pretty)
            return 0
        if args.command == "serve":
            return _run_serve(config, args)
        if args.command == "mcp":
            return _run_mcp(config, args)
        result = asyncio.run(_run_remote(config, args))
        _print_value(result, pretty=args.pretty)
        return 0
    except KeyboardInterrupt:
        return 130
    except (AMemorixError, OSError, RuntimeError, ValidationError, ValueError) as exc:
        _print_error(exc)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="a-memorix",
        description="Run and administer A_memorix",
    )
    parser.add_argument(
        "--config",
        help="explicit TOML configuration file; A_MEMORIX_CONFIG is the fallback",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="indent JSON output",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="run the Python gRPC service")
    serve.add_argument("--data-dir")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument(
        "--allow-unauthenticated",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    serve.add_argument("--admin-token-file")
    serve.add_argument("--tls-certificate")
    serve.add_argument("--tls-private-key")
    serve.add_argument("--tls-client-ca")
    serve.add_argument(
        "--tls-require-client-auth",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    serve.add_argument("--log-level")
    serve.add_argument("--log-format", choices=("json", "text"))
    serve.add_argument("--metrics-host")
    serve.add_argument("--metrics-port", type=int)
    serve.add_argument("--otlp-endpoint")

    mcp = commands.add_parser("mcp", help="run a fixed-namespace MCP server")
    mcp.add_argument("--namespace", required=True)
    mcp.add_argument("--data-dir")
    mcp.add_argument("--create-namespace", action="store_true")
    mcp.add_argument("--transport", default="stdio", choices=("stdio",))

    commands.add_parser("config", help="print the effective redacted configuration")

    adapter = commands.add_parser("adapter", help="validate adapter metadata")
    adapter_commands = adapter.add_subparsers(dest="adapter_command", required=True)
    adapter_validate = adapter_commands.add_parser("validate")
    adapter_validate.add_argument("manifest", type=Path)
    adapter_validate.add_argument("--core-version", default=__version__)
    adapter_commands.add_parser("schema")

    namespace = commands.add_parser("namespace", help="manage namespaces")
    _add_client_options(namespace)
    namespace_commands = namespace.add_subparsers(dest="namespace_command", required=True)
    create = namespace_commands.add_parser("create")
    create.add_argument("namespace_id")
    create.add_argument("--max-concurrent-requests", type=int)
    create.add_argument("--max-storage-bytes", type=int)
    create.add_argument("--namespace-config", type=Path)
    create.add_argument(
        "--allow-metadata-only-write",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    create.add_argument(
        "--sparse-retrieval",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    namespace_commands.add_parser("list")
    for name in ("get", "disable", "enable", "delete", "restore-deleted", "purge"):
        command = namespace_commands.add_parser(name)
        command.add_argument("namespace_id")

    api_key = commands.add_parser("api-key", help="manage namespace API keys")
    _add_client_options(api_key)
    key_commands = api_key.add_subparsers(dest="api_key_command", required=True)
    key_create = key_commands.add_parser("create")
    key_create.add_argument("namespace_id")
    key_create.add_argument("--label", default="")
    key_list = key_commands.add_parser("list")
    key_list.add_argument("namespace_id")
    key_revoke = key_commands.add_parser("revoke")
    key_revoke.add_argument("namespace_id")
    key_revoke.add_argument("key_id")

    memory = commands.add_parser("memory", help="manage memories")
    _add_client_options(memory)
    memory_commands = memory.add_subparsers(dest="memory_command", required=True)
    ingest = memory_commands.add_parser("ingest")
    ingest.add_argument("namespace_id")
    ingest.add_argument("--text", required=True)
    ingest.add_argument("--source-type", required=True)
    ingest.add_argument("--external-id", default="")
    ingest.add_argument("--conversation-id", default="")
    ingest.add_argument("--idempotency-key", default="")
    search = memory_commands.add_parser("search")
    search.add_argument("namespace_id")
    search.add_argument("--query", default="")
    search.add_argument("--limit", type=int, default=5)
    search.add_argument(
        "--mode",
        choices=("search", "time", "hybrid", "episode", "aggregate"),
        default="search",
    )
    search.add_argument("--conversation-id", default="")
    for name in ("get", "delete"):
        command = memory_commands.add_parser(name)
        command.add_argument("namespace_id")
        selectors = command.add_mutually_exclusive_group(required=True)
        selectors.add_argument("--memory-id")
        selectors.add_argument("--external-id")

    backup = commands.add_parser("backup", help="manage namespace backups")
    _add_client_options(backup)
    backup_commands = backup.add_subparsers(dest="backup_command", required=True)
    backup_create = backup_commands.add_parser("create")
    backup_create.add_argument("namespace_id")
    backup_list = backup_commands.add_parser("list")
    backup_list.add_argument("--source-namespace-id", default="")
    backup_download = backup_commands.add_parser("download")
    backup_download.add_argument("backup_id")
    backup_download.add_argument("destination", type=Path)
    backup_download.add_argument("--force", action="store_true")
    backup_upload = backup_commands.add_parser("upload")
    backup_upload.add_argument("source", type=Path)
    backup_restore = backup_commands.add_parser("restore")
    backup_restore.add_argument("backup_id")
    backup_restore.add_argument("target_namespace_id")
    backup_delete = backup_commands.add_parser("delete")
    backup_delete.add_argument("backup_id")

    doctor = commands.add_parser("doctor", help="check configuration and service health")
    _add_client_options(doctor)
    doctor.add_argument("--health-only", action="store_true")
    return parser


def _adapter_command(args: argparse.Namespace) -> object:
    if args.adapter_command == "schema":
        return adapter_manifest_json_schema()
    manifest = load_adapter_manifest(args.manifest)
    ensure_adapter_compatible(manifest, core_version=args.core_version)
    return {
        "valid": True,
        "core_version": args.core_version,
        "manifest": manifest.model_dump(mode="json"),
    }


def _add_client_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target")
    parser.add_argument("--token-file")
    parser.add_argument("--ca-certificate")
    parser.add_argument("--client-certificate")
    parser.add_argument("--client-private-key")
    parser.add_argument("--server-name")
    parser.add_argument("--tls", action="store_true", default=None)
    parser.add_argument("--timeout", type=float)


def _run_serve(config: AMemorixConfig, args: argparse.Namespace) -> int:
    from a_memorix.logging import configure_logging

    server_config = _server_config(config.server, args)
    observability = _validated_update(
        config.observability,
        _defined_updates(
            log_level=args.log_level,
            log_format=args.log_format,
            metrics_host=args.metrics_host,
            metrics_port=args.metrics_port,
            otlp_endpoint=args.otlp_endpoint,
        )
    )
    configure_logging(observability.log_level, observability.log_format)
    try:
        asyncio.run(_serve(server_config, observability))
    except KeyboardInterrupt:
        pass
    return 0


async def _serve(server_config: ServerConfig, observability_config: Any) -> None:
    from a_memorix.engine import AMemorixEngine
    from a_memorix.observability import ObservabilityRuntime
    from a_memorix.server import AMemorixGrpcServer

    admin_token = read_secret(
        environment_name="A_MEMORIX_ADMIN_TOKEN",
        file_path=server_config.admin_token_file,
    )
    engine = AMemorixEngine(
        data_dir=server_config.data_dir,
        max_active_namespaces=server_config.max_active_namespaces,
        max_concurrent_requests_per_namespace=(
            server_config.max_concurrent_requests_per_namespace
        ),
        idle_timeout_seconds=server_config.idle_timeout_seconds,
        quarantine_retention_days=server_config.quarantine_retention_days,
        idempotency_retention_seconds=(
            server_config.idempotency_retention_seconds
        ),
    )
    telemetry = ObservabilityRuntime(observability_config)
    server = AMemorixGrpcServer(
        engine,
        host=server_config.host,
        port=server_config.port,
        admin_token=admin_token,
        allow_unauthenticated=server_config.allow_unauthenticated,
        credentials=_server_credentials(server_config),
        maximum_message_bytes=server_config.maximum_message_bytes,
        observability=telemetry,
    )
    await server.start()
    try:
        await _wait_for_server_shutdown(server)
    finally:
        await server.stop()


async def _wait_for_server_shutdown(server: Any) -> None:
    loop = asyncio.get_running_loop()
    stop_requested = asyncio.Event()
    registered_signals: list[signal.Signals] = []
    for event in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(event, stop_requested.set)
        except (NotImplementedError, RuntimeError):
            continue
        registered_signals.append(event)
    if not registered_signals:
        await server.wait_for_termination()
        return
    server_wait = asyncio.create_task(server.wait_for_termination())
    signal_wait = asyncio.create_task(stop_requested.wait())
    try:
        _, pending = await asyncio.wait(
            (server_wait, signal_wait),
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if server_wait.done() and not server_wait.cancelled():
            server_wait.result()
    finally:
        for event in registered_signals:
            loop.remove_signal_handler(event)


def _run_mcp(config: AMemorixConfig, args: argparse.Namespace) -> int:
    from a_memorix import AMemorixEngine, create_fixed_namespace_mcp
    from a_memorix.logging import configure_logging

    configure_logging(
        config.observability.log_level,
        config.observability.log_format,
    )
    data_dir = Path(args.data_dir) if args.data_dir else config.server.data_dir
    server = create_fixed_namespace_mcp(
        AMemorixEngine(data_dir=data_dir),
        args.namespace,
        create_namespace=args.create_namespace,
    )
    server.run(transport=args.transport)
    return 0


async def _run_remote(config: AMemorixConfig, args: argparse.Namespace) -> object:
    from a_memorix.client import AMemorixClient

    client_config = _client_config(config.client, args)
    token = os.environ.get("A_MEMORIX_API_KEY", "").strip() or read_secret(
        environment_name="A_MEMORIX_ADMIN_TOKEN",
        file_path=client_config.token_file,
    )
    async with AMemorixClient(
        client_config.target,
        api_key=token,
        credentials=_client_credentials(client_config),
        timeout=client_config.timeout_seconds,
        maximum_message_bytes=client_config.maximum_message_bytes,
        tls_server_name=client_config.tls.server_name,
    ) as client:
        if args.command == "namespace":
            return await _namespace_command(client, args)
        if args.command == "api-key":
            return await _api_key_command(client, args)
        if args.command == "memory":
            return await _memory_command(client, args)
        if args.command == "backup":
            return await _backup_command(client, args)
        if args.command == "doctor":
            return await _doctor(client, config, args)
    raise RuntimeError(f"unsupported command: {args.command}")


async def _namespace_command(client: Any, args: argparse.Namespace) -> object:
    from a_memorix.api.v1 import namespace_pb2

    command = args.namespace_command
    if command == "create":
        quota = namespace_pb2.NamespaceQuota()
        if args.max_concurrent_requests is not None:
            quota.max_concurrent_requests = args.max_concurrent_requests
        if args.max_storage_bytes is not None:
            quota.max_storage_bytes = args.max_storage_bytes
        namespace_config = namespace_pb2.NamespaceConfig()
        if args.namespace_config is not None:
            payload = json.loads(args.namespace_config.read_text(encoding="utf-8"))
            from google.protobuf import json_format

            json_format.ParseDict(payload, namespace_config, ignore_unknown_fields=False)
        if args.allow_metadata_only_write is not None:
            namespace_config.features.allow_metadata_only_write = (
                args.allow_metadata_only_write
            )
        if args.sparse_retrieval is not None:
            namespace_config.features.sparse_retrieval = args.sparse_retrieval
        return await client.create_namespace(
            namespace_pb2.CreateNamespaceRequest(
                namespace_id=args.namespace_id,
                quota=quota,
                config=namespace_config,
            )
        )
    if command == "list":
        return await client.list_namespaces()
    request = namespace_pb2.GetNamespaceRequest(namespace_id=args.namespace_id)
    if command == "get":
        return await client.get_namespace(request)
    method_name = {
        "disable": "disable_namespace",
        "enable": "enable_namespace",
        "delete": "delete_namespace",
        "restore-deleted": "restore_namespace",
        "purge": "purge_namespace",
    }[command]
    request_type = {
        "disable": namespace_pb2.DisableNamespaceRequest,
        "enable": namespace_pb2.EnableNamespaceRequest,
        "delete": namespace_pb2.DeleteNamespaceRequest,
        "restore-deleted": namespace_pb2.RestoreNamespaceRequest,
        "purge": namespace_pb2.PurgeNamespaceRequest,
    }[command]
    return await getattr(client, method_name)(
        request_type(namespace_id=args.namespace_id)
    )


async def _api_key_command(client: Any, args: argparse.Namespace) -> object:
    from a_memorix.api.v1 import auth_pb2

    if args.api_key_command == "create":
        return await client.create_api_key(
            auth_pb2.CreateApiKeyRequest(
                namespace_id=args.namespace_id,
                label=args.label,
            )
        )
    if args.api_key_command == "list":
        return await client.list_api_keys(
            auth_pb2.ListApiKeysRequest(namespace_id=args.namespace_id)
        )
    return await client.revoke_api_key(
        auth_pb2.RevokeApiKeyRequest(
            namespace_id=args.namespace_id,
            key_id=args.key_id,
        )
    )


async def _memory_command(client: Any, args: argparse.Namespace) -> object:
    from a_memorix.api.v1 import common_pb2, memory_pb2

    context = common_pb2.RequestContext(namespace_id=args.namespace_id)
    if args.memory_command == "ingest":
        context.conversation_id = args.conversation_id
        return await client.ingest_text(
            memory_pb2.IngestTextRequest(
                context=context,
                external_id=args.external_id,
                source_type=args.source_type,
                text=args.text,
            ),
            idempotency_key=args.idempotency_key,
        )
    if args.memory_command == "search":
        context.conversation_id = args.conversation_id
        modes = {
            "search": memory_pb2.SEARCH_MODE_SEARCH,
            "time": memory_pb2.SEARCH_MODE_TIME,
            "hybrid": memory_pb2.SEARCH_MODE_HYBRID,
            "episode": memory_pb2.SEARCH_MODE_EPISODE,
            "aggregate": memory_pb2.SEARCH_MODE_AGGREGATE,
        }
        return await client.search_memory(
            memory_pb2.SearchMemoryRequest(
                context=context,
                query=args.query,
                limit=args.limit,
                mode=modes[args.mode],
            )
        )
    request_type = (
        memory_pb2.GetMemoryRequest
        if args.memory_command == "get"
        else memory_pb2.DeleteMemoryRequest
    )
    request = request_type(context=context)
    if args.memory_id:
        request.memory_id = args.memory_id
    else:
        request.external_id = args.external_id
    method = client.get_memory if args.memory_command == "get" else client.delete_memory
    return await method(request)


async def _backup_command(client: Any, args: argparse.Namespace) -> object:
    from a_memorix.api.v1 import backup_pb2

    if args.backup_command == "create":
        return await client.create_namespace_backup(
            backup_pb2.CreateNamespaceBackupRequest(
                namespace_id=args.namespace_id
            )
        )
    if args.backup_command == "list":
        return await client.list_namespace_backups(
            backup_pb2.ListNamespaceBackupsRequest(
                source_namespace_id=args.source_namespace_id
            )
        )
    if args.backup_command == "download":
        return await _download_backup(client, args.backup_id, args.destination, args.force)
    if args.backup_command == "upload":
        return await _upload_backup(client, args.source)
    if args.backup_command == "restore":
        return await client.restore_namespace_from_backup(
            backup_pb2.RestoreNamespaceFromBackupRequest(
                backup_id=args.backup_id,
                target_namespace_id=args.target_namespace_id,
            )
        )
    return await client.delete_namespace_backup(
        backup_pb2.DeleteNamespaceBackupRequest(backup_id=args.backup_id)
    )


async def _download_backup(
    client: Any,
    backup_id: str,
    destination: Path,
    force: bool,
) -> dict[str, object]:
    from a_memorix.api.v1 import backup_pb2

    destination = destination.expanduser().resolve()
    if destination.exists() and not force:
        raise FileExistsError(f"backup destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.part")
    digest = hashlib.sha256()
    offset = 0
    backup: Any | None = None
    try:
        with temporary.open("xb") as output:
            while True:
                response = await client.download_namespace_backup(
                    backup_pb2.DownloadNamespaceBackupRequest(
                        backup_id=backup_id,
                        offset=offset,
                        max_bytes=1024 * 1024,
                    )
                )
                backup = response.backup
                output.write(response.data)
                digest.update(response.data)
                offset = response.next_offset
                if response.complete:
                    break
        if backup is None or digest.hexdigest() != backup.sha256:
            raise RuntimeError("downloaded backup failed SHA-256 verification")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "backup": _json_value(backup),
        "destination": str(destination),
    }


async def _upload_backup(client: Any, source: Path) -> object:
    from a_memorix.api.v1 import backup_pb2

    source = source.expanduser().resolve()
    digest = hashlib.sha256()
    upload = await client.begin_namespace_backup_upload()
    offset = 0
    try:
        with source.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                response = await client.upload_namespace_backup_chunk(
                    backup_pb2.UploadNamespaceBackupChunkRequest(
                        upload_id=upload.upload_id,
                        offset=offset,
                        data=chunk,
                    )
                )
                offset = response.next_offset
        return await client.complete_namespace_backup_upload(
            backup_pb2.CompleteNamespaceBackupUploadRequest(
                upload_id=upload.upload_id,
                expected_sha256=digest.hexdigest(),
            )
        )
    except BaseException as error:
        try:
            await client.abort_namespace_backup_upload(
                backup_pb2.AbortNamespaceBackupUploadRequest(
                    upload_id=upload.upload_id
                )
            )
        except BaseException as cleanup_error:
            error.add_note(f"backup upload cleanup failed: {cleanup_error}")
        raise


async def _doctor(
    client: Any,
    config: AMemorixConfig,
    args: argparse.Namespace,
) -> dict[str, object]:
    health = await client.check_health()
    result: dict[str, object] = {
        "healthy": health == "SERVING",
        "grpc_health": health,
        "target": _client_config(config.client, args).target,
    }
    if not args.health_only:
        data_path = config.server.data_dir.expanduser().resolve()
        existing_parent = data_path
        while not existing_parent.exists() and existing_parent != existing_parent.parent:
            existing_parent = existing_parent.parent
        result["configuration"] = config.redacted()
        result["data_parent_exists"] = existing_parent.exists()
        result["data_parent_writable"] = os.access(existing_parent, os.W_OK)
    if not result["healthy"]:
        raise RuntimeError(f"gRPC health check returned {health}")
    return result


def _server_config(config: ServerConfig, args: argparse.Namespace) -> ServerConfig:
    tls = _validated_update(
        config.tls,
        _defined_updates(
            certificate=_optional_path(args.tls_certificate),
            private_key=_optional_path(args.tls_private_key),
            client_ca=_optional_path(args.tls_client_ca),
            require_client_auth=args.tls_require_client_auth,
        )
    )
    return _validated_update(
        config,
        {
            **_defined_updates(
                data_dir=_optional_path(args.data_dir),
                host=args.host,
                port=args.port,
                allow_unauthenticated=args.allow_unauthenticated,
                admin_token_file=_optional_path(args.admin_token_file),
            ),
            "tls": tls,
        }
    )


def _client_config(config: ClientConfig, args: argparse.Namespace) -> ClientConfig:
    tls = _validated_update(
        config.tls,
        _defined_updates(
            enabled=getattr(args, "tls", None),
            ca_certificate=_optional_path(getattr(args, "ca_certificate", None)),
            certificate=_optional_path(getattr(args, "client_certificate", None)),
            private_key=_optional_path(getattr(args, "client_private_key", None)),
            server_name=getattr(args, "server_name", None),
        )
    )
    return _validated_update(
        config,
        {
            **_defined_updates(
                target=getattr(args, "target", None),
                token_file=_optional_path(getattr(args, "token_file", None)),
                timeout_seconds=getattr(args, "timeout", None),
            ),
            "tls": tls,
        }
    )


def _server_credentials(config: ServerConfig) -> Any:
    if config.tls.certificate is None:
        return None
    import grpc

    private_key = config.tls.private_key.read_bytes()
    certificate = config.tls.certificate.read_bytes()
    client_ca = config.tls.client_ca.read_bytes() if config.tls.client_ca else None
    return grpc.ssl_server_credentials(
        ((private_key, certificate),),
        root_certificates=client_ca,
        require_client_auth=config.tls.require_client_auth,
    )


def _client_credentials(config: ClientConfig) -> Any:
    tls = config.tls
    if (
        tls.ca_certificate is None
        and tls.certificate is None
        and not tls.server_name
        and not tls.enabled
    ):
        return None
    import grpc

    return grpc.ssl_channel_credentials(
        root_certificates=(
            tls.ca_certificate.read_bytes() if tls.ca_certificate else None
        ),
        private_key=tls.private_key.read_bytes() if tls.private_key else None,
        certificate_chain=tls.certificate.read_bytes() if tls.certificate else None,
    )


def _defined_updates(**values: object) -> dict[str, object]:
    return {key: value for key, value in values.items() if value is not None}


def _validated_update(model: Any, updates: dict[str, object]) -> Any:
    values = model.model_dump(mode="python")
    values.update(updates)
    return type(model).model_validate(values)


def _optional_path(value: str | None) -> Path | None:
    return Path(value).expanduser() if value else None


def _json_value(value: object) -> object:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (dict, list, tuple, str, int, float, bool, type(None))):
        return value
    try:
        from google.protobuf import json_format, message
    except ModuleNotFoundError:
        return str(value)
    if isinstance(value, message.Message):
        return json.loads(
            json_format.MessageToJson(
                value,
                preserving_proto_field_name=True,
                sort_keys=True,
            )
        )
    return value


def _print_value(value: object, *, pretty: bool) -> None:
    print(
        json.dumps(
            _json_value(value),
            ensure_ascii=False,
            indent=2 if pretty else None,
            sort_keys=True,
            separators=None if pretty else (",", ":"),
        )
    )


def _print_error(error: BaseException) -> None:
    if isinstance(error, AMemorixError):
        payload = error.to_envelope().model_dump(mode="json")
    else:
        notes = list(getattr(error, "__notes__", ()))
        payload = {
            "code": ErrorCode.INVALID_ARGUMENT.value
            if isinstance(error, (OSError, ValidationError, ValueError))
            else ErrorCode.INTERNAL_ERROR.value,
            "message": str(error),
            "retryable": False,
            "details": {"notes": notes} if notes else {},
        }
    print(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        file=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
