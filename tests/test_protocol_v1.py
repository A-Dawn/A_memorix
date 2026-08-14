from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import asyncio
import json
import os
import shutil
import socket
import subprocess
import urllib.error
import urllib.request

import pytest
from google.api import annotations_pb2
from mcp import Client

from a_memorix import (
    AMemorixClient,
    AMemorixEngine,
    CreateNamespaceRequest,
    ErrorCode,
    InvalidArgumentError,
    RelationExtractionMode,
    RemoteAMemorixError,
    create_fixed_namespace_mcp,
)
from a_memorix.api.v1 import (
    auth_pb2,
    backup_pb2,
    common_pb2,
    job_pb2,
    memory_pb2,
    namespace_pb2,
)
from a_memorix.server import AMemorixGrpcServer
from a_memorix.server.mapping import ingest_input_from_proto, ingest_request_from_proto


ADMIN_TOKEN = "admin-token-for-tests-with-at-least-32-characters"


def test_unknown_relation_extraction_enum_inherits_default() -> None:
    context = common_pb2.RequestContext(
        namespace_id="tenant-a",
        agent_id="test-agent",
    )
    request = memory_pb2.IngestTextRequest(
        context=context,
        source_type="document",
        text="unknown enum",
        relation_extraction=99,
    )
    item = memory_pb2.IngestTextInput(
        source_type="document",
        text="unknown enum",
        relation_extraction=99,
    )

    assert (
        ingest_request_from_proto(request).relation_extraction
        is RelationExtractionMode.INHERIT
    )
    assert (
        ingest_input_from_proto(item).relation_extraction
        is RelationExtractionMode.INHERIT
    )


class MemoryRuntime:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.ready = False
        self.memories: dict[str, dict[str, object]] = {}

    async def initialize(self) -> None:
        self.ready = True

    async def shutdown(self) -> None:
        self.ready = False

    def is_runtime_ready(self) -> bool:
        return self.ready

    def runtime_capability_status(self) -> dict[str, object]:
        return {
            "capabilities": {"metadata": True, "sparse": True},
            "degraded": False,
        }

    async def contains_external_memory(self, external_id: str) -> bool:
        return external_id in self.memories

    async def execute_request_with_dedup(self, request_key: str, executor):
        del request_key
        return False, await executor()

    async def ingest_text(self, **kwargs: Any) -> dict[str, object]:
        external_id = str(kwargs["external_id"])
        if external_id in self.memories:
            return {
                "stored_ids": [],
                "skipped_ids": [str(self.memories[external_id]["memory_id"])],
                "reason": "exists",
            }
        memory_id = f"memory:{external_id}"
        self.memories[external_id] = {
            "memory_id": memory_id,
            "external_id": external_id,
            "source_type": str(kwargs["source_type"]),
            "content": str(kwargs["text"]),
            "source": str(kwargs["source_type"]),
            "metadata": dict(kwargs.get("metadata", {})),
        }
        return {"stored_ids": [memory_id], "skipped_ids": []}

    async def get_memory_record(
        self,
        *,
        memory_id: str = "",
        external_id: str = "",
    ) -> dict[str, object] | None:
        if external_id:
            return self.memories.get(external_id)
        return next(
            (
                value
                for value in self.memories.values()
                if value["memory_id"] == memory_id
            ),
            None,
        )

    async def memory_delete_admin(self, **kwargs: Any) -> dict[str, object]:
        memory_id = str((kwargs.get("selector") or {}).get("hash", ""))
        external_id = next(
            (
                key
                for key, value in self.memories.items()
                if value["memory_id"] == memory_id
            ),
            "",
        )
        deleted = self.memories.pop(external_id, None) if external_id else None
        return {
            "operation_id": "delete-memory" if deleted else "",
            "deleted_paragraph_count": int(deleted is not None),
            "error": "" if deleted else "memory not found",
        }

    async def memory_source_admin(self, **kwargs: Any) -> dict[str, object]:
        source = str(kwargs.get("source", ""))
        targets = [
            key for key, value in self.memories.items() if value["source"] == source
        ]
        for key in targets:
            self.memories.pop(key)
        return {
            "operation_id": "delete-source" if targets else "",
            "deleted_count": len(targets),
            "deleted_paragraph_count": len(targets),
        }

    async def search_memory(self, request) -> dict[str, object]:
        hits = [
            {
                "hash": value["memory_id"],
                "type": "paragraph",
                "content": value["content"],
                "score": 1.0,
                "source": value["source"],
                "metadata": value["metadata"],
            }
            for value in self.memories.values()
            if not request.query
            or request.query.casefold() in str(value["content"]).casefold()
        ][: request.limit]
        return {
            "summary": "\n".join(str(item["content"]) for item in hits),
            "hits": hits,
            "degraded": False,
            "retrieval_ready": True,
            "retrieval_mode": "test",
            "available_channels": ["test"],
            "unavailable_channels": [],
        }


class MemoryRuntimeFactory:
    def __init__(self) -> None:
        self.runtimes: dict[str, MemoryRuntime] = {}

    def __call__(self, namespace, data_dir: Path) -> MemoryRuntime:
        runtime = MemoryRuntime(data_dir)
        self.runtimes[namespace.namespace_id] = runtime
        return runtime


def test_proto_is_the_source_of_http_routes() -> None:
    create = namespace_pb2.DESCRIPTOR.services_by_name["NamespaceService"].methods_by_name[
        "CreateNamespace"
    ]
    ingest = memory_pb2.DESCRIPTOR.services_by_name["MemoryService"].methods_by_name[
        "IngestText"
    ]
    create_rule = create.GetOptions().Extensions[annotations_pb2.http]
    ingest_rule = ingest.GetOptions().Extensions[annotations_pb2.http]
    backup = backup_pb2.DESCRIPTOR.services_by_name["BackupService"].methods_by_name[
        "CreateNamespaceBackup"
    ]
    backup_rule = backup.GetOptions().Extensions[annotations_pb2.http]

    assert create_rule.post == "/v1/namespaces"
    assert ingest_rule.post == "/v1/namespaces/{context.namespace_id}/memories:ingest"
    assert backup_rule.post == "/v1/namespaces/{namespace_id}/backups"


@pytest.mark.asyncio
async def test_api_keys_are_hashed_scoped_and_revocable(tmp_path: Path) -> None:
    engine = AMemorixEngine(
        data_dir=tmp_path,
        runtime_factory=MemoryRuntimeFactory(),
    )
    await engine.initialize()
    try:
        await engine.create_namespace(CreateNamespaceRequest(namespace_id="tenant-a"))
        await engine.create_namespace(CreateNamespaceRequest(namespace_id="tenant-b"))
        created = await engine.create_api_key("tenant-a", label="agent")

        authenticated = engine.authenticate_api_key(created.secret)
        assert authenticated is not None
        assert authenticated.namespace_id == "tenant-a"
        assert engine.authenticate_api_key("invalid") is None

        database_bytes = b"".join(
            path.read_bytes()
            for path in tmp_path.glob("namespaces.db*")
            if path.is_file()
        )
        assert created.secret.encode() not in database_bytes

        revoked = await engine.revoke_api_key("tenant-a", created.api_key.key_id)
        assert revoked.revoked_at is not None
        assert engine.authenticate_api_key(created.secret) is None

        with pytest.raises(InvalidArgumentError):
            await engine.create_api_key(
                "tenant-b",
                expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
            )
    finally:
        await engine.shutdown()


@pytest.mark.asyncio
async def test_grpc_auth_errors_and_memory_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("A_MEMORIX_ADMIN_TOKEN", ADMIN_TOKEN)
    engine = AMemorixEngine(
        data_dir=tmp_path,
        runtime_factory=MemoryRuntimeFactory(),
    )
    server = AMemorixGrpcServer(
        engine,
        port=0,
    )
    async with server:
        async with AMemorixClient(server.target, api_key=ADMIN_TOKEN) as admin:
            await admin.create_namespace(
                namespace_pb2.CreateNamespaceRequest(namespace_id="tenant-a")
            )
            await admin.create_namespace(
                namespace_pb2.CreateNamespaceRequest(namespace_id="tenant-b")
            )
            first_page = await admin.list_namespaces(
                namespace_pb2.ListNamespacesRequest(page_size=1)
            )
            assert len(first_page.namespaces) == 1
            assert first_page.next_page_token

            await admin.disable_namespace(
                namespace_pb2.DisableNamespaceRequest(namespace_id="tenant-a")
            )
            configured = await admin.update_namespace_config(
                namespace_pb2.UpdateNamespaceConfigRequest(
                    namespace_id="tenant-a",
                    expected_config_version=1,
                    config=namespace_pb2.NamespaceConfig(
                        llm=namespace_pb2.ProviderReference(
                            provider_id="openai-compatible",
                            model_id="test-model",
                            secret_ref="secret://tenant-a/llm",
                        ),
                        features=namespace_pb2.NamespaceFeatureConfig(
                            episodes=False,
                            person_profiles=False,
                            sparse_retrieval=True,
                            relation_vectors=False,
                            allow_metadata_only_write=True,
                        ),
                        relation_extraction=namespace_pb2.RelationExtractionConfig(
                            enabled=True,
                            default_enabled=False,
                            profile="agent-memory-v1",
                            max_entities=48,
                            max_relations=40,
                            max_chunk_chars=6000,
                            chunk_overlap_chars=400,
                        ),
                    ),
                )
            )
            assert configured.namespace.config_version == 2
            assert configured.namespace.config.llm.secret_ref == "secret://tenant-a/llm"
            assert configured.namespace.config.relation_extraction.enabled is True
            assert (
                configured.namespace.config.relation_extraction.profile
                == "agent-memory-v1"
            )
            assert configured.namespace.config.relation_extraction.max_chunk_chars == 6000
            await admin.enable_namespace(
                namespace_pb2.EnableNamespaceRequest(namespace_id="tenant-a")
            )
            created = await admin.create_api_key(
                auth_pb2.CreateApiKeyRequest(namespace_id="tenant-a", label="agent")
            )

            async with AMemorixClient(server.target, api_key=created.secret) as tenant:
                own = await tenant.get_namespace(
                    namespace_pb2.GetNamespaceRequest(namespace_id="tenant-a")
                )
                assert own.namespace.namespace_id == "tenant-a"

                with pytest.raises(RemoteAMemorixError) as forbidden:
                    await tenant.get_namespace(
                        namespace_pb2.GetNamespaceRequest(namespace_id="tenant-b")
                    )
                assert forbidden.value.code is ErrorCode.FORBIDDEN

                context = common_pb2.RequestContext(
                    namespace_id="tenant-a",
                    request_id="grpc-request-1",
                    trace_id="grpc-trace-1",
                )
                ingested = await tenant.ingest_text(
                    memory_pb2.IngestTextRequest(
                        context=context,
                        external_id="document:1",
                        source_type="document",
                        text="Protocol parity memory",
                    ),
                    idempotency_key="ingest-1",
                )
                assert list(ingested.stored_ids) == ["memory:document:1"]

                result = await tenant.search_memory(
                    memory_pb2.SearchMemoryRequest(
                        context=context,
                        query="protocol parity",
                        limit=5,
                    )
                )
                assert [item.content for item in result.hits] == [
                    "Protocol parity memory"
                ]

                capabilities = await tenant.get_namespace_capabilities(
                    namespace_pb2.GetNamespaceCapabilitiesRequest(
                        namespace_id="tenant-a"
                    )
                )
                assert capabilities.capabilities.capabilities["get_memory"]

                batch = await tenant.batch_ingest_text(
                    memory_pb2.BatchIngestTextRequest(
                        context=context,
                        items=[
                            memory_pb2.IngestTextInput(
                                external_id="document:batch",
                                source_type="document",
                                text="Batch protocol memory",
                            )
                        ],
                    ),
                    idempotency_key="batch-1",
                )
                assert batch.succeeded == 1
                fetched = await tenant.get_memory(
                    memory_pb2.GetMemoryRequest(
                        context=context,
                        external_id="document:1",
                    )
                )
                assert fetched.memory.content == "Protocol parity memory"
                deleted = await tenant.delete_memory(
                    memory_pb2.DeleteMemoryRequest(
                        context=context,
                        external_id="document:1",
                    )
                )
                assert list(deleted.deleted_memory_ids) == ["memory:document:1"]

                submitted = await tenant.submit_delete_by_source(
                    job_pb2.SubmitDeleteBySourceRequest(
                        context=context,
                        source="document",
                    )
                )
                for _ in range(100):
                    job = await tenant.get_job(
                        job_pb2.GetJobRequest(
                            namespace_id="tenant-a",
                            job_id=submitted.job.job_id,
                        )
                    )
                    if job.job.status not in {
                        job_pb2.JOB_STATUS_PENDING,
                        job_pb2.JOB_STATUS_RUNNING,
                    }:
                        break
                    await asyncio.sleep(0.01)
                assert job.job.status == job_pb2.JOB_STATUS_SUCCEEDED

                with pytest.raises(RemoteAMemorixError) as invalid_limit:
                    await tenant.search_memory(
                        memory_pb2.SearchMemoryRequest(
                            context=context,
                            query="protocol parity",
                            limit=0,
                        )
                    )
                assert invalid_limit.value.code is ErrorCode.INVALID_ARGUMENT

                conflicting = memory_pb2.IngestTextRequest(
                    context=common_pb2.RequestContext(
                        namespace_id="tenant-a",
                        idempotency_key="body-key",
                        request_id="grpc-request-2",
                    ),
                    external_id="document:2",
                    source_type="document",
                    text="conflict",
                )
                with pytest.raises(RemoteAMemorixError) as invalid:
                    await tenant.ingest_text(
                        conflicting,
                        idempotency_key="header-key",
                    )
                assert invalid.value.code is ErrorCode.INVALID_ARGUMENT
                assert invalid.value.request_id == "grpc-request-2"

            await admin.disable_namespace(
                namespace_pb2.DisableNamespaceRequest(namespace_id="tenant-a")
            )
            async with AMemorixClient(server.target, api_key=created.secret) as tenant:
                with pytest.raises(RemoteAMemorixError) as forbidden_backup:
                    await tenant.create_namespace_backup(
                        backup_pb2.CreateNamespaceBackupRequest(
                            namespace_id="tenant-a"
                        )
                    )
                assert forbidden_backup.value.code is ErrorCode.FORBIDDEN

            created_backup = await admin.create_namespace_backup(
                backup_pb2.CreateNamespaceBackupRequest(namespace_id="tenant-a")
            )
            listed_backups = await admin.list_namespace_backups(
                backup_pb2.ListNamespaceBackupsRequest(source_namespace_id="tenant-a")
            )
            assert [item.backup_id for item in listed_backups.backups] == [
                created_backup.backup.backup_id
            ]
            downloaded = bytearray()
            offset = 0
            while True:
                chunk = await admin.download_namespace_backup(
                    backup_pb2.DownloadNamespaceBackupRequest(
                        backup_id=created_backup.backup.backup_id,
                        offset=offset,
                        max_bytes=257,
                    )
                )
                downloaded.extend(chunk.data)
                offset = chunk.next_offset
                if chunk.complete:
                    break
            upload = await admin.begin_namespace_backup_upload()
            uploaded = await admin.upload_namespace_backup_chunk(
                backup_pb2.UploadNamespaceBackupChunkRequest(
                    upload_id=upload.upload_id,
                    data=bytes(downloaded),
                )
            )
            assert uploaded.next_offset == len(downloaded)
            completed = await admin.complete_namespace_backup_upload(
                backup_pb2.CompleteNamespaceBackupUploadRequest(
                    upload_id=upload.upload_id,
                    expected_sha256=created_backup.backup.sha256,
                )
            )
            assert completed.backup.backup_id == created_backup.backup.backup_id
            restored = await admin.restore_namespace_from_backup(
                backup_pb2.RestoreNamespaceFromBackupRequest(
                    backup_id=created_backup.backup.backup_id,
                    target_namespace_id="tenant-restored",
                )
            )
            assert restored.namespace.status == namespace_pb2.NAMESPACE_STATUS_INACTIVE

            await admin.revoke_api_key(
                auth_pb2.RevokeApiKeyRequest(
                    namespace_id="tenant-a",
                    key_id=created.api_key.key_id,
                )
            )
            async with AMemorixClient(server.target, api_key=created.secret) as revoked:
                with pytest.raises(RemoteAMemorixError) as unauthorized:
                    await revoked.get_namespace(
                        namespace_pb2.GetNamespaceRequest(namespace_id="tenant-a")
                    )
                assert unauthorized.value.code is ErrorCode.UNAUTHORIZED


@pytest.mark.asyncio
async def test_gateway_http_json_matches_grpc(tmp_path: Path) -> None:
    go = shutil.which("go")
    if go is None:
        pytest.skip("Go is required for the gRPC-Gateway integration test")

    repository = Path(__file__).resolve().parents[1]
    executable = tmp_path / ("a-memorix-gateway.exe" if os.name == "nt" else "a-memorix-gateway")
    environment = {**os.environ, "GOPROXY": "https://goproxy.cn,direct"}
    subprocess.run(
        [go, "build", "-o", str(executable), "./cmd/a-memorix-gateway"],
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    engine = AMemorixEngine(
        data_dir=tmp_path / "data",
        runtime_factory=MemoryRuntimeFactory(),
    )
    server = AMemorixGrpcServer(
        engine,
        port=0,
        admin_token=ADMIN_TOKEN,
    )
    http_port = _free_port()
    gateway: subprocess.Popen[str] | None = None
    async with server:
        gateway = subprocess.Popen(
            [
                str(executable),
                "--listen",
                f"127.0.0.1:{http_port}",
                "--grpc-target",
                server.target,
            ],
            cwd=repository,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            base_url = f"http://127.0.0.1:{http_port}"
            await _wait_for_gateway(base_url, gateway)
            status, created_namespace = await asyncio.to_thread(
                _http_json,
                "POST",
                f"{base_url}/v1/namespaces",
                {"namespaceId": "http-tenant"},
                ADMIN_TOKEN,
            )
            assert status == 200
            assert created_namespace["namespace"]["namespaceId"] == "http-tenant"

            status, created_key = await asyncio.to_thread(
                _http_json,
                "POST",
                f"{base_url}/v1/namespaces/http-tenant/api-keys",
                {"namespaceId": "http-tenant", "label": "http-agent"},
                ADMIN_TOKEN,
            )
            assert status == 200
            namespace_key = str(created_key["secret"])

            context = {"namespaceId": "http-tenant", "requestId": "http-request-1"}
            status, ingested = await asyncio.to_thread(
                _http_json,
                "POST",
                f"{base_url}/v1/namespaces/http-tenant/memories:ingest",
                {
                    "context": context,
                    "externalId": "document:http",
                    "sourceType": "document",
                    "text": "Shared HTTP and gRPC result",
                },
                namespace_key,
            )
            assert status == 200
            assert ingested["storedIds"] == ["memory:document:http"]

            status, http_search = await asyncio.to_thread(
                _http_json,
                "POST",
                f"{base_url}/v1/namespaces/http-tenant/memories:search",
                {
                    "context": {"namespaceId": "http-tenant"},
                    "query": "shared http",
                    "limit": 5,
                },
                namespace_key,
                "http-request-1",
            )
            assert status == 200, http_search

            status, fetched = await asyncio.to_thread(
                _http_json,
                "POST",
                f"{base_url}/v1/namespaces/http-tenant/memories:get",
                {
                    "context": {"namespaceId": "http-tenant"},
                    "externalId": "document:http",
                },
                namespace_key,
            )
            assert status == 200
            assert fetched["memory"]["content"] == "Shared HTTP and gRPC result"

            status, batch = await asyncio.to_thread(
                _http_json,
                "POST",
                f"{base_url}/v1/namespaces/http-tenant/memories:batchIngest",
                {
                    "context": {
                        "namespaceId": "http-tenant",
                        "idempotencyKey": "http-batch-1",
                    },
                    "items": [
                        {
                            "externalId": "document:http-batch",
                            "sourceType": "document",
                            "text": "HTTP batch result",
                        }
                    ],
                },
                namespace_key,
            )
            assert status == 200
            assert batch["succeeded"] == 1

            status, invalid_limit = await asyncio.to_thread(
                _http_json,
                "POST",
                f"{base_url}/v1/namespaces/http-tenant/memories:search",
                {
                    "context": {"namespaceId": "http-tenant"},
                    "query": "shared http",
                    "limit": 0,
                },
                namespace_key,
                "http-invalid-limit",
            )
            assert status == 400
            assert invalid_limit["code"] == "invalid_argument"
            assert invalid_limit["requestId"] == "http-invalid-limit"

            async with AMemorixClient(server.target, api_key=namespace_key) as client:
                grpc_search = await client.search_memory(
                    memory_pb2.SearchMemoryRequest(
                        context=common_pb2.RequestContext(
                            namespace_id="http-tenant",
                            request_id="http-request-1",
                        ),
                        query="shared http",
                        limit=5,
                    )
                )
            assert [item["content"] for item in http_search["hits"]] == [
                item.content for item in grpc_search.hits
            ]

            status, _ = await asyncio.to_thread(
                _http_json,
                "POST",
                f"{base_url}/v1/namespaces/http-tenant:disable",
                None,
                ADMIN_TOKEN,
            )
            assert status == 200
            status, http_backup = await asyncio.to_thread(
                _http_json,
                "POST",
                f"{base_url}/v1/namespaces/http-tenant/backups",
                None,
                ADMIN_TOKEN,
            )
            assert status == 200, http_backup
            backup_id = str(http_backup["backup"]["backupId"])
            status, backup_content = await asyncio.to_thread(
                _http_json,
                "GET",
                f"{base_url}/v1/backups/{backup_id}/content?maxBytes=1048576",
                None,
                ADMIN_TOKEN,
            )
            assert status == 200, backup_content
            assert backup_content["backup"]["backupId"] == backup_id
            assert backup_content["data"]
            status, upload_started = await asyncio.to_thread(
                _http_json,
                "POST",
                f"{base_url}/v1/backup-uploads",
                None,
                ADMIN_TOKEN,
            )
            assert status == 200, upload_started
            upload_id = str(upload_started["uploadId"])
            status, upload_progress = await asyncio.to_thread(
                _http_json,
                "PUT",
                f"{base_url}/v1/backup-uploads/{upload_id}",
                {"offset": "0", "data": backup_content["data"]},
                ADMIN_TOKEN,
            )
            assert status == 200, upload_progress
            status, upload_completed = await asyncio.to_thread(
                _http_json,
                "POST",
                f"{base_url}/v1/backup-uploads/{upload_id}:complete",
                {"expectedSha256": http_backup["backup"]["sha256"]},
                ADMIN_TOKEN,
            )
            assert status == 200, upload_completed
            assert upload_completed["backup"]["backupId"] == backup_id
            status, restored_backup = await asyncio.to_thread(
                _http_json,
                "POST",
                f"{base_url}/v1/backups/{backup_id}:restore",
                {"targetNamespaceId": "http-restored"},
                ADMIN_TOKEN,
            )
            assert status == 200, restored_backup
            assert restored_backup["namespace"]["namespaceId"] == "http-restored"
            assert (
                restored_backup["namespace"]["status"]
                == "NAMESPACE_STATUS_INACTIVE"
            )
        finally:
            if gateway is not None:
                gateway.terminate()
                await asyncio.to_thread(gateway.wait, 10)


@pytest.mark.asyncio
async def test_mcp_adapter_is_bound_to_one_namespace(tmp_path: Path) -> None:
    factory = MemoryRuntimeFactory()
    engine = AMemorixEngine(data_dir=tmp_path, runtime_factory=factory)
    server = create_fixed_namespace_mcp(
        engine,
        "mcp-tenant",
        create_namespace=True,
    )

    async with Client(server) as client:
        tools = await client.list_tools()
        schemas = {tool.name: tool.input_schema for tool in tools.tools}
        assert set(schemas) == {
            "batch_ingest_text",
            "delete_by_source",
            "delete_memory",
            "get_job",
            "get_memory",
            "ingest_text",
            "list_jobs",
            "namespace_health",
            "search_memory",
        }
        assert all(
            "namespace_id" not in schema.get("properties", {})
            for schema in schemas.values()
        )

        ingested = await client.call_tool(
            "ingest_text",
            {
                "text": "MCP fixed namespace memory",
                "source_type": "document",
                "external_id": "document:mcp",
            },
        )
        assert not ingested.is_error
        assert ingested.structured_content["stored_ids"] == ["memory:document:mcp"]

        searched = await client.call_tool(
            "search_memory",
            {"query": "fixed namespace", "limit": 5},
        )
        assert not searched.is_error
        assert [item["content"] for item in searched.structured_content["hits"]] == [
            "MCP fixed namespace memory"
        ]
        assert set(factory.runtimes) == {"mcp-tenant"}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _wait_for_gateway(base_url: str, process: subprocess.Popen[str]) -> None:
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError(f"gRPC-Gateway exited with code {process.returncode}")
        try:
            response = await asyncio.to_thread(
                urllib.request.urlopen,
                f"{base_url}/healthz",
                timeout=5,
            )
            if response.status == 200 and response.read() == b"serving\n":
                return
        except OSError:
            pass
        await asyncio.sleep(0.05)
    raise TimeoutError("gRPC-Gateway did not become ready")


def _http_json(
    method: str,
    url: str,
    body: dict[str, object] | None,
    token: str,
    request_id: str = "",
) -> tuple[int, dict[str, object]]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Accept", "application/json")
    if body is not None:
        request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    if request_id:
        request.add_header("X-Request-ID", request_id)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")
