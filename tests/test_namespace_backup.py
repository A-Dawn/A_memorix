from __future__ import annotations

from pathlib import Path

import hashlib
import json
import zipfile

import pytest

from a_memorix import (
    AMemorixEngine,
    CreateNamespaceRequest,
    MigrationRequiredError,
    NamespaceConfig,
    NamespaceConflictError,
    NamespaceFeatureConfig,
    NamespaceIntegrityError,
    NamespaceQuota,
    NamespaceStateError,
    NamespaceStatus,
    ProviderReference,
    RestoreNamespaceBackupRequest,
)
from a_memorix.core.runtime.namespace_storage import NamespaceStorageLayout


class BackupRuntime:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.ready = False

    async def initialize(self) -> None:
        self.ready = True

    async def shutdown(self) -> None:
        self.ready = False

    def is_runtime_ready(self) -> bool:
        return self.ready


class BackupRuntimeFactory:
    def __call__(self, _namespace, data_dir: Path) -> BackupRuntime:
        return BackupRuntime(data_dir)


def _namespace_data_roots(data_root: Path) -> list[Path]:
    return sorted(
        (path for path in (data_root / "namespaces").iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )


@pytest.mark.asyncio
async def test_backup_round_trip_preserves_data_config_and_quota(
    tmp_path: Path,
) -> None:
    engine = AMemorixEngine(
        data_dir=tmp_path,
        runtime_factory=BackupRuntimeFactory(),
        idle_timeout_seconds=0,
    )
    await engine.initialize()
    try:
        config = NamespaceConfig(
            llm=ProviderReference(
                provider_id="openai-compatible",
                model_id="test-model",
                secret_ref="secret://source/llm",
            ),
            features=NamespaceFeatureConfig(episodes=False),
        )
        quota = NamespaceQuota(
            max_concurrent_requests=3,
            max_storage_bytes=1024 * 1024,
        )
        await engine.create_namespace(
            CreateNamespaceRequest(
                namespace_id="source",
                quota=quota,
                config=config,
            )
        )
        source_root = _namespace_data_roots(tmp_path)[0]
        (source_root / "metadata").mkdir()
        payload = b"portable namespace data\x00\x01"
        (source_root / "metadata" / "memory.db").write_bytes(payload)
        created_key = await engine.create_api_key("source", label="not-exported")

        with pytest.raises(NamespaceStateError):
            await engine.create_namespace_backup("source")

        await engine.disable_namespace("source")
        backup = await engine.create_namespace_backup("source")
        assert backup.source_namespace_id == "source"
        assert backup.data_size_bytes == len(payload)
        assert backup.file_count == 1
        assert backup.source_config_version == 1

        archive_path = tmp_path / "backups" / f"{backup.backup_id}.amxbackup"
        with zipfile.ZipFile(archive_path) as archive:
            assert set(archive.namelist()) == {
                "manifest.json",
                "data/metadata/memory.db",
            }
            assert archive.read("data/metadata/memory.db") == payload
            archive_bytes = archive_path.read_bytes()
            assert created_key.secret.encode() not in archive_bytes
            assert b"idempotency_records" not in archive_bytes
            assert b"api_keys" not in archive_bytes
            assert b"jobs" not in archive_bytes

        restored = await engine.restore_namespace_from_backup(
            RestoreNamespaceBackupRequest(
                backup_id=backup.backup_id,
                target_namespace_id="restored",
            )
        )
        assert restored.status is NamespaceStatus.INACTIVE
        assert restored.config == config
        assert restored.quota == quota
        assert restored.config_version == 1

        restored_files = [
            path
            for root in _namespace_data_roots(tmp_path)
            if root != source_root
            for path in root.rglob("memory.db")
        ]
        assert len(restored_files) == 1
        assert restored_files[0].read_bytes() == payload
        assert await engine.list_api_keys("restored") == []

        with pytest.raises(NamespaceConflictError):
            await engine.restore_namespace_from_backup(
                RestoreNamespaceBackupRequest(
                    backup_id=backup.backup_id,
                    target_namespace_id="restored",
                )
            )
    finally:
        await engine.shutdown()


@pytest.mark.asyncio
async def test_backup_chunk_download_and_upload_are_restart_safe(
    tmp_path: Path,
) -> None:
    engine = AMemorixEngine(
        data_dir=tmp_path,
        runtime_factory=BackupRuntimeFactory(),
        idle_timeout_seconds=0,
    )
    await engine.initialize()
    await engine.create_namespace(CreateNamespaceRequest(namespace_id="source"))
    source_root = _namespace_data_roots(tmp_path)[0]
    (source_root / "payload.bin").write_bytes(b"chunked-backup" * 1024)
    await engine.disable_namespace("source")
    backup = await engine.create_namespace_backup("source")

    downloaded = bytearray()
    offset = 0
    while True:
        chunk = await engine.download_namespace_backup(
            backup.backup_id,
            offset=offset,
            max_bytes=257,
        )
        assert chunk.offset == offset
        downloaded.extend(chunk.data)
        offset = chunk.next_offset
        if chunk.complete:
            break
    assert len(downloaded) == backup.archive_size_bytes
    assert hashlib.sha256(downloaded).hexdigest() == backup.sha256

    upload = await engine.begin_namespace_backup_upload()
    split = len(downloaded) // 2
    first = await engine.upload_namespace_backup_chunk(
        upload.upload_id,
        offset=0,
        data=bytes(downloaded[:split]),
    )
    with pytest.raises(NamespaceConflictError):
        await engine.upload_namespace_backup_chunk(
            upload.upload_id,
            offset=0,
            data=b"wrong-offset",
        )
    await engine.upload_namespace_backup_chunk(
        upload.upload_id,
        offset=first.next_offset,
        data=bytes(downloaded[split:]),
    )
    uploaded = await engine.complete_namespace_backup_upload(
        upload.upload_id,
        expected_sha256=backup.sha256,
    )
    assert uploaded == backup

    await engine.shutdown()
    await engine.initialize()
    try:
        assert await engine.get_namespace_backup(backup.backup_id) == backup
    finally:
        await engine.shutdown()


@pytest.mark.asyncio
async def test_restore_rejects_a_tampered_archive_and_cleans_staging(
    tmp_path: Path,
) -> None:
    engine = AMemorixEngine(
        data_dir=tmp_path,
        runtime_factory=BackupRuntimeFactory(),
        idle_timeout_seconds=0,
    )
    await engine.initialize()
    try:
        await engine.create_namespace(CreateNamespaceRequest(namespace_id="source"))
        source_root = _namespace_data_roots(tmp_path)[0]
        (source_root / "payload.bin").write_bytes(b"original")
        await engine.disable_namespace("source")
        backup = await engine.create_namespace_backup("source")
        archive_path = tmp_path / "backups" / f"{backup.backup_id}.amxbackup"
        archive_bytes = bytearray(archive_path.read_bytes())
        archive_bytes[len(archive_bytes) // 2] ^= 0xFF
        archive_path.write_bytes(archive_bytes)

        with pytest.raises(NamespaceIntegrityError):
            await engine.restore_namespace_from_backup(
                RestoreNamespaceBackupRequest(
                    backup_id=backup.backup_id,
                    target_namespace_id="tampered",
                )
            )
        assert list((tmp_path / "restore-staging").iterdir()) == []
        assert [item.namespace_id for item in await engine.list_namespaces()] == [
            "source"
        ]
    finally:
        await engine.shutdown()


@pytest.mark.asyncio
async def test_startup_recovers_interrupted_restore_switches(tmp_path: Path) -> None:
    engine = AMemorixEngine(
        data_dir=tmp_path,
        runtime_factory=BackupRuntimeFactory(),
        idle_timeout_seconds=0,
    )
    await engine.initialize()
    await engine.create_namespace(CreateNamespaceRequest(namespace_id="registered"))
    await engine.disable_namespace("registered")
    registered_root = _namespace_data_roots(tmp_path)[0]
    (registered_root / "kept.bin").write_bytes(b"registered")
    await engine.shutdown()

    layout = NamespaceStorageLayout(tmp_path)
    layout.initialize()
    layout.mark_restore_pending(registered_root.name, "registered")

    orphan_key = "f" * 32
    orphan_root = layout.active_path(orphan_key)
    orphan_root.mkdir()
    (orphan_root / "orphan.bin").write_bytes(b"orphan")
    layout.mark_restore_pending(orphan_key, "orphan")

    staging_key = "e" * 32
    staging_root = layout.create_restore_staging(staging_key)
    (staging_root / "partial.bin").write_bytes(b"partial")

    await engine.initialize()
    try:
        assert registered_root.is_dir()
        assert (registered_root / "kept.bin").read_bytes() == b"registered"
        assert not orphan_root.exists()
        assert not staging_root.exists()
        assert list(layout.restore_pending_root.iterdir()) == []
    finally:
        await engine.shutdown()


@pytest.mark.asyncio
async def test_upload_rejects_an_unsupported_backup_format_version(
    tmp_path: Path,
) -> None:
    engine = AMemorixEngine(
        data_dir=tmp_path,
        runtime_factory=BackupRuntimeFactory(),
        idle_timeout_seconds=0,
    )
    await engine.initialize()
    try:
        await engine.create_namespace(CreateNamespaceRequest(namespace_id="source"))
        await engine.disable_namespace("source")
        backup = await engine.create_namespace_backup("source")
        archive_path = tmp_path / "backups" / f"{backup.backup_id}.amxbackup"
        unsupported_path = tmp_path / "unsupported.amxbackup"
        with zipfile.ZipFile(archive_path, "r") as source:
            entries = {name: source.read(name) for name in source.namelist()}
        manifest = json.loads(entries["manifest.json"])
        manifest["format_version"] = 2
        entries["manifest.json"] = json.dumps(manifest).encode("utf-8")
        with zipfile.ZipFile(unsupported_path, "w") as target:
            for name, payload in entries.items():
                target.writestr(name, payload)

        upload = await engine.begin_namespace_backup_upload()
        await engine.upload_namespace_backup_chunk(
            upload.upload_id,
            offset=0,
            data=unsupported_path.read_bytes(),
        )
        with pytest.raises(MigrationRequiredError):
            await engine.complete_namespace_backup_upload(upload.upload_id)
        await engine.abort_namespace_backup_upload(upload.upload_id)
    finally:
        await engine.shutdown()
