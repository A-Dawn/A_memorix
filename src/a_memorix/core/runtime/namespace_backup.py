"""Versioned, portable namespace backup archives."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import RLock
from uuid import uuid4

import hashlib
import json
import os
import re
import zipfile
import zlib

from pydantic import ValidationError

from a_memorix.contracts import (
    InvalidArgumentError,
    MigrationRequiredError,
    NAMESPACE_BACKUP_FORMAT,
    NAMESPACE_BACKUP_FORMAT_VERSION,
    NamespaceBackupFile,
    NamespaceBackupInfo,
    NamespaceBackupManifest,
    NamespaceBackupUpload,
    NamespaceConflictError,
    NamespaceInfo,
    NamespaceIntegrityError,
    NotFoundError,
)


BACKUP_ARCHIVE_SUFFIX = ".amxbackup"
MAX_BACKUP_CHUNK_BYTES = 1024 * 1024
_MANIFEST_NAME = "manifest.json"
_DATA_PREFIX = "data/"
_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COPY_BUFFER_BYTES = 1024 * 1024
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024


class NamespaceBackupStore:
    def __init__(self, backup_root: Path, upload_root: Path) -> None:
        self.backup_root = Path(backup_root)
        self.upload_root = Path(upload_root)
        self._lock = RLock()
        self._initialized = False

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            for path in (self.backup_root, self.upload_root):
                path.mkdir(parents=True, exist_ok=True)
                if path.is_symlink() or path.is_junction():
                    raise NamespaceIntegrityError(
                        f"backup storage cannot use a symbolic link: {path}"
                    )
            for temporary_path in self.backup_root.glob(".*.tmp"):
                temporary_path.unlink()
            for temporary_path in self.backup_root.glob("*.json.tmp"):
                temporary_path.unlink()
            self._initialized = True

    def create_backup(
        self,
        namespace: NamespaceInfo,
        source: Path,
        *,
        created_at: datetime | None = None,
    ) -> NamespaceBackupInfo:
        self.initialize()
        backup_id = uuid4().hex
        timestamp = created_at or datetime.now(timezone.utc)
        temporary_path = self.backup_root / f".{backup_id}.tmp"
        final_path = self._archive_path(backup_id)
        try:
            manifest = self._write_archive(
                temporary_path,
                backup_id=backup_id,
                namespace=namespace,
                source=Path(source),
                created_at=timestamp,
            )
            archive_sha256 = _hash_file(temporary_path)
            info = _backup_info(
                manifest,
                archive_size_bytes=temporary_path.stat().st_size,
                sha256=archive_sha256,
            )
            with self._lock:
                if final_path.exists():
                    raise NamespaceConflictError(
                        f"namespace backup already exists: {backup_id}",
                        details={"backup_id": backup_id},
                    )
                os.replace(temporary_path, final_path)
                self._write_metadata(info)
            return info
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    def get_backup(self, backup_id: str) -> NamespaceBackupInfo:
        archive_path = self._archive_path(backup_id)
        if not archive_path.is_file():
            raise NotFoundError(
                f"namespace backup not found: {backup_id}",
                details={"backup_id": backup_id},
            )
        _reject_link(archive_path, kind="namespace backup archive")
        metadata_path = self._metadata_path(backup_id)
        if metadata_path.is_file():
            _reject_link(metadata_path, kind="namespace backup metadata")
            try:
                info = NamespaceBackupInfo.model_validate_json(
                    metadata_path.read_text(encoding="utf-8")
                )
            except (OSError, ValidationError, ValueError) as exc:
                raise NamespaceIntegrityError(
                    f"namespace backup metadata is invalid: {backup_id}",
                    details={"backup_id": backup_id},
                ) from exc
            if info.backup_id != backup_id:
                raise NamespaceIntegrityError(
                    f"namespace backup metadata ID does not match: {backup_id}",
                    details={"backup_id": backup_id},
                )
            if archive_path.stat().st_size != info.archive_size_bytes:
                raise NamespaceIntegrityError(
                    f"namespace backup size does not match its metadata: {backup_id}",
                    details={"backup_id": backup_id},
                )
            return info
        _, info = self.inspect_archive(archive_path, expected_backup_id=backup_id)
        self._write_metadata(info)
        return info

    def list_backups(self, source_namespace_id: str = "") -> list[NamespaceBackupInfo]:
        self.initialize()
        items = [
            self.get_backup(path.name.removesuffix(BACKUP_ARCHIVE_SUFFIX))
            for path in self.backup_root.glob(f"*{BACKUP_ARCHIVE_SUFFIX}")
            if path.is_file()
        ]
        if source_namespace_id:
            items = [
                item
                for item in items
                if item.source_namespace_id == source_namespace_id
            ]
        return sorted(items, key=lambda item: (item.created_at, item.backup_id))

    def delete_backup(self, backup_id: str) -> None:
        info = self.get_backup(backup_id)
        del info
        with self._lock:
            self._archive_path(backup_id).unlink()
            self._metadata_path(backup_id).unlink(missing_ok=True)

    def read_chunk(
        self,
        backup_id: str,
        *,
        offset: int = 0,
        max_bytes: int = 256 * 1024,
    ) -> tuple[NamespaceBackupInfo, bytes, int, bool]:
        if offset < 0:
            raise InvalidArgumentError("backup download offset cannot be negative")
        if not (1 <= max_bytes <= MAX_BACKUP_CHUNK_BYTES):
            raise InvalidArgumentError(
                f"backup chunk size must be between 1 and {MAX_BACKUP_CHUNK_BYTES} bytes"
            )
        info = self.get_backup(backup_id)
        if offset > info.archive_size_bytes:
            raise InvalidArgumentError(
                "backup download offset exceeds archive size",
                details={
                    "backup_id": backup_id,
                    "offset": offset,
                    "archive_size_bytes": info.archive_size_bytes,
                },
            )
        with self._archive_path(backup_id).open("rb") as handle:
            handle.seek(offset)
            data = handle.read(max_bytes)
        next_offset = offset + len(data)
        return info, data, next_offset, next_offset == info.archive_size_bytes

    def begin_upload(self) -> NamespaceBackupUpload:
        self.initialize()
        upload_id = uuid4().hex
        path = self._upload_path(upload_id)
        with self._lock:
            path.touch(exist_ok=False)
        return NamespaceBackupUpload(upload_id=upload_id)

    def append_upload(
        self,
        upload_id: str,
        *,
        offset: int,
        data: bytes,
    ) -> NamespaceBackupUpload:
        if offset < 0:
            raise InvalidArgumentError("backup upload offset cannot be negative")
        if not data:
            raise InvalidArgumentError("backup upload chunk cannot be empty")
        if len(data) > MAX_BACKUP_CHUNK_BYTES:
            raise InvalidArgumentError(
                f"backup upload chunk exceeds {MAX_BACKUP_CHUNK_BYTES} bytes"
            )
        path = self._require_upload(upload_id)
        with self._lock:
            current_offset = path.stat().st_size
            if offset != current_offset:
                raise NamespaceConflictError(
                    "backup upload offset does not match",
                    details={
                        "upload_id": upload_id,
                        "expected_offset": current_offset,
                        "received_offset": offset,
                    },
                )
            with path.open("ab") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            return NamespaceBackupUpload(
                upload_id=upload_id,
                next_offset=current_offset + len(data),
            )

    def complete_upload(
        self,
        upload_id: str,
        *,
        expected_sha256: str = "",
    ) -> NamespaceBackupInfo:
        path = self._require_upload(upload_id)
        normalized_sha256 = str(expected_sha256 or "").strip().lower()
        if normalized_sha256 and not _SHA256_PATTERN.fullmatch(normalized_sha256):
            raise InvalidArgumentError("expected backup SHA-256 is invalid")
        actual_sha256 = _hash_file(path)
        if normalized_sha256 and normalized_sha256 != actual_sha256:
            raise NamespaceIntegrityError(
                "uploaded backup SHA-256 does not match",
                details={
                    "upload_id": upload_id,
                    "expected_sha256": normalized_sha256,
                    "actual_sha256": actual_sha256,
                },
            )
        manifest, info = self.inspect_archive(path, known_sha256=actual_sha256)
        final_path = self._archive_path(manifest.backup_id)
        with self._lock:
            if final_path.exists():
                _, existing = self.inspect_backup(manifest.backup_id)
                if existing.sha256 != actual_sha256:
                    raise NamespaceConflictError(
                        "uploaded backup ID conflicts with an existing archive",
                        details={"backup_id": manifest.backup_id},
                    )
                path.unlink(missing_ok=True)
                return existing
            os.replace(path, final_path)
            self._write_metadata(info)
        return info

    def abort_upload(self, upload_id: str) -> None:
        path = self._require_upload(upload_id)
        with self._lock:
            path.unlink()

    def inspect_backup(
        self,
        backup_id: str,
    ) -> tuple[NamespaceBackupManifest, NamespaceBackupInfo]:
        path = self._archive_path(backup_id)
        if not path.is_file():
            raise NotFoundError(
                f"namespace backup not found: {backup_id}",
                details={"backup_id": backup_id},
            )
        _reject_link(path, kind="namespace backup archive")
        return self.inspect_archive(path, expected_backup_id=backup_id)

    def inspect_archive(
        self,
        path: Path,
        *,
        expected_backup_id: str = "",
        known_sha256: str = "",
    ) -> tuple[NamespaceBackupManifest, NamespaceBackupInfo]:
        archive_path = Path(path)
        try:
            with zipfile.ZipFile(archive_path, mode="r", allowZip64=True) as archive:
                manifest = _read_manifest(archive)
                if expected_backup_id and manifest.backup_id != expected_backup_id:
                    raise NamespaceIntegrityError(
                        "backup archive ID does not match its managed resource",
                        details={
                            "expected_backup_id": expected_backup_id,
                            "manifest_backup_id": manifest.backup_id,
                        },
                    )
                _validate_archive_members(archive, manifest)
                _verify_archive_data(archive, manifest)
        except NamespaceIntegrityError:
            raise
        except (
            OSError,
            zipfile.BadZipFile,
            ValidationError,
            ValueError,
            zlib.error,
        ) as exc:
            raise NamespaceIntegrityError(
                "namespace backup archive is invalid"
            ) from exc
        archive_sha256 = known_sha256 or _hash_file(archive_path)
        return manifest, _backup_info(
            manifest,
            archive_size_bytes=archive_path.stat().st_size,
            sha256=archive_sha256,
        )

    def extract_backup(self, backup_id: str, destination: Path) -> None:
        manifest, _ = self.inspect_backup(backup_id)
        destination = Path(destination)
        if not destination.is_dir() or any(destination.iterdir()):
            raise NamespaceIntegrityError(
                "namespace restore destination must be an empty directory"
            )
        try:
            with zipfile.ZipFile(
                self._archive_path(backup_id), mode="r", allowZip64=True
            ) as archive:
                for item in manifest.files:
                    target = destination.joinpath(*item.path.split("/"))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    size_bytes = 0
                    with archive.open(f"{_DATA_PREFIX}{item.path}", "r") as source:
                        with target.open("xb") as output:
                            while chunk := source.read(_COPY_BUFFER_BYTES):
                                output.write(chunk)
                                digest.update(chunk)
                                size_bytes += len(chunk)
                    if (
                        size_bytes != item.size_bytes
                        or digest.hexdigest() != item.sha256
                    ):
                        raise NamespaceIntegrityError(
                            f"backup file failed verification during restore: {item.path}"
                        )
        except NamespaceIntegrityError:
            raise
        except (OSError, zipfile.BadZipFile, zlib.error) as exc:
            raise NamespaceIntegrityError("namespace backup extraction failed") from exc

    def _write_archive(
        self,
        path: Path,
        *,
        backup_id: str,
        namespace: NamespaceInfo,
        source: Path,
        created_at: datetime,
    ) -> NamespaceBackupManifest:
        if not source.is_dir():
            raise NamespaceIntegrityError("namespace data directory is missing")
        files = _source_files(source)
        manifest_files: list[NamespaceBackupFile] = []
        with zipfile.ZipFile(
            path,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for source_path in files:
                relative_path = source_path.relative_to(source).as_posix()
                digest = hashlib.sha256()
                size_bytes = 0
                entry = _zip_info(f"{_DATA_PREFIX}{relative_path}")
                with source_path.open("rb") as input_file:
                    with archive.open(entry, mode="w", force_zip64=True) as output:
                        while chunk := input_file.read(_COPY_BUFFER_BYTES):
                            output.write(chunk)
                            digest.update(chunk)
                            size_bytes += len(chunk)
                manifest_files.append(
                    NamespaceBackupFile(
                        path=relative_path,
                        size_bytes=size_bytes,
                        sha256=digest.hexdigest(),
                    )
                )
            manifest = NamespaceBackupManifest(
                backup_id=backup_id,
                source_namespace_id=namespace.namespace_id,
                created_at=created_at,
                producer_version=_producer_version(),
                source_config_version=namespace.config_version,
                quota=namespace.quota,
                config=namespace.config,
                data_size_bytes=sum(item.size_bytes for item in manifest_files),
                files=tuple(manifest_files),
            )
            archive.writestr(
                _zip_info(_MANIFEST_NAME),
                manifest.model_dump_json().encode("utf-8"),
            )
        return manifest

    def _write_metadata(self, info: NamespaceBackupInfo) -> None:
        path = self._metadata_path(info.backup_id)
        temporary_path = path.with_suffix(".json.tmp")
        temporary_path.write_text(info.model_dump_json(), encoding="utf-8")
        os.replace(temporary_path, path)

    def _archive_path(self, backup_id: str) -> Path:
        normalized = _validate_id(backup_id, kind="backup")
        return self.backup_root / f"{normalized}{BACKUP_ARCHIVE_SUFFIX}"

    def _metadata_path(self, backup_id: str) -> Path:
        normalized = _validate_id(backup_id, kind="backup")
        return self.backup_root / f"{normalized}.json"

    def _upload_path(self, upload_id: str) -> Path:
        normalized = _validate_id(upload_id, kind="backup upload")
        return self.upload_root / f"{normalized}.part"

    def _require_upload(self, upload_id: str) -> Path:
        path = self._upload_path(upload_id)
        if not path.is_file():
            raise NotFoundError(
                f"namespace backup upload not found: {upload_id}",
                details={"upload_id": upload_id},
            )
        _reject_link(path, kind="namespace backup upload")
        return path


def _source_files(source: Path) -> list[Path]:
    files: list[Path] = []
    for path in source.rglob("*"):
        if path.is_symlink() or path.is_junction():
            raise NamespaceIntegrityError(
                f"namespace backup cannot contain a symbolic link: {path}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise NamespaceIntegrityError(
                f"namespace backup contains a non-regular file: {path}"
            )
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(source).as_posix())


def _read_manifest(archive: zipfile.ZipFile) -> NamespaceBackupManifest:
    entries = [item for item in archive.infolist() if item.filename == _MANIFEST_NAME]
    if len(entries) != 1:
        raise NamespaceIntegrityError("backup archive must contain one manifest")
    entry = entries[0]
    if entry.file_size > _MAX_MANIFEST_BYTES:
        raise NamespaceIntegrityError("backup manifest exceeds the supported size")
    try:
        payload = archive.read(entry)
        raw_manifest = json.loads(payload)
    except (KeyError, UnicodeDecodeError, ValidationError, ValueError) as exc:
        raise NamespaceIntegrityError("backup manifest is invalid") from exc
    if not isinstance(raw_manifest, dict):
        raise NamespaceIntegrityError("backup manifest is invalid")
    if raw_manifest.get("format") != NAMESPACE_BACKUP_FORMAT:
        raise NamespaceIntegrityError("backup archive format is invalid")
    format_version = raw_manifest.get("format_version")
    if format_version != NAMESPACE_BACKUP_FORMAT_VERSION:
        raise MigrationRequiredError(
            "namespace backup format version is not supported",
            details={
                "archive_format_version": format_version,
                "supported_format_version": NAMESPACE_BACKUP_FORMAT_VERSION,
            },
        )
    try:
        return NamespaceBackupManifest.model_validate(raw_manifest)
    except ValidationError as exc:
        raise NamespaceIntegrityError("backup manifest is invalid") from exc


def _validate_archive_members(
    archive: zipfile.ZipFile,
    manifest: NamespaceBackupManifest,
) -> None:
    entries = archive.infolist()
    names = [item.filename for item in entries]
    if len(names) != len(set(names)):
        raise NamespaceIntegrityError("backup archive contains duplicate entries")
    expected = {
        _MANIFEST_NAME,
        *(f"{_DATA_PREFIX}{item.path}" for item in manifest.files),
    }
    if set(names) != expected:
        raise NamespaceIntegrityError(
            "backup archive entries do not match the manifest"
        )
    files_by_name = {item.filename: item for item in entries}
    for item in entries:
        unix_mode = (item.external_attr >> 16) & 0o170000
        if item.is_dir() or unix_mode == 0o120000 or item.flag_bits & 0x1:
            raise NamespaceIntegrityError(
                f"backup archive contains an unsupported entry: {item.filename}"
            )
    for file_record in manifest.files:
        entry = files_by_name[f"{_DATA_PREFIX}{file_record.path}"]
        if entry.file_size != file_record.size_bytes:
            raise NamespaceIntegrityError(
                f"backup entry size does not match the manifest: {file_record.path}"
            )


def _verify_archive_data(
    archive: zipfile.ZipFile,
    manifest: NamespaceBackupManifest,
) -> None:
    for item in manifest.files:
        digest = hashlib.sha256()
        size_bytes = 0
        with archive.open(f"{_DATA_PREFIX}{item.path}", "r") as source:
            while chunk := source.read(_COPY_BUFFER_BYTES):
                digest.update(chunk)
                size_bytes += len(chunk)
        if size_bytes != item.size_bytes or digest.hexdigest() != item.sha256:
            raise NamespaceIntegrityError(
                f"backup file checksum does not match: {item.path}"
            )


def _backup_info(
    manifest: NamespaceBackupManifest,
    *,
    archive_size_bytes: int,
    sha256: str,
) -> NamespaceBackupInfo:
    return NamespaceBackupInfo(
        backup_id=manifest.backup_id,
        source_namespace_id=manifest.source_namespace_id,
        created_at=manifest.created_at,
        format_version=manifest.format_version,
        producer_version=manifest.producer_version,
        source_config_version=manifest.source_config_version,
        archive_size_bytes=archive_size_bytes,
        data_size_bytes=manifest.data_size_bytes,
        file_count=len(manifest.files),
        sha256=sha256,
    )


def _zip_info(filename: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=filename, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100600 << 16
    return info


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_COPY_BUFFER_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_id(value: str, *, kind: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _ID_PATTERN.fullmatch(normalized):
        raise InvalidArgumentError(f"invalid {kind} ID")
    return normalized


def _reject_link(path: Path, *, kind: str) -> None:
    if path.is_symlink() or path.is_junction():
        raise NamespaceIntegrityError(f"{kind} cannot use a symbolic link: {path}")


def _producer_version() -> str:
    try:
        return version("a-memorix")
    except PackageNotFoundError:
        return "2.0.0a1"
