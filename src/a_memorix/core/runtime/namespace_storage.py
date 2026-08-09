"""Safe physical layout for namespace-owned files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import json
import os
import re
import shutil

from a_memorix.contracts import NamespaceIntegrityError, NamespaceStatus
from a_memorix.paths import resolve_data_dir

from .namespace_control import _NamespaceRecord


_STORAGE_KEY_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class NamespaceStorageLayout:
    def __init__(self, data_root: str | Path) -> None:
        self.data_root = resolve_data_dir(data_root)
        self.control_root = self.data_root / "control"
        self.namespaces_root = self.data_root / "namespaces"
        self.quarantine_root = self.data_root / "quarantine"
        self.restore_staging_root = self.data_root / "restore-staging"
        self.restore_pending_root = self.data_root / "restore-pending"
        self.backups_root = self.data_root / "backups"
        self.backup_uploads_root = self.data_root / "backup-uploads"

    @property
    def control_db_path(self) -> Path:
        return self.control_root / "namespaces.db"

    def initialize(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        for path in (
            self.control_root,
            self.namespaces_root,
            self.quarantine_root,
            self.restore_staging_root,
            self.restore_pending_root,
            self.backups_root,
            self.backup_uploads_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
            self._assert_within(path, self.data_root)
            if _is_link_or_junction(path):
                raise NamespaceIntegrityError(f"namespace layout cannot use a symbolic link: {path}")

    def active_path(self, storage_key: str) -> Path:
        return self._storage_path(self.namespaces_root, storage_key)

    def quarantine_path(self, storage_key: str) -> Path:
        return self._storage_path(self.quarantine_root, storage_key)

    def create_active(self, storage_key: str) -> Path:
        path = self.active_path(storage_key)
        path.mkdir(parents=False, exist_ok=False)
        return path

    def create_restore_staging(self, storage_key: str) -> Path:
        path = self._storage_path(self.restore_staging_root, storage_key)
        path.mkdir(parents=False, exist_ok=False)
        return path

    def commit_restore_staging(self, storage_key: str) -> Path:
        source = self._storage_path(self.restore_staging_root, storage_key)
        target = self.active_path(storage_key)
        if target.exists():
            raise NamespaceIntegrityError(
                f"namespace restore target already exists: {storage_key}"
            )
        if not source.is_dir():
            raise NamespaceIntegrityError(
                f"namespace restore staging directory is missing: {storage_key}"
            )
        self._assert_tree_has_no_symlinks(source)
        os.replace(source, target)
        return target

    def discard_restore_staging(self, storage_key: str) -> None:
        path = self._storage_path(self.restore_staging_root, storage_key)
        if not path.exists():
            return
        self._assert_tree_has_no_symlinks(path)
        shutil.rmtree(path)

    def mark_restore_pending(self, storage_key: str, namespace_id: str) -> None:
        path = self._restore_pending_path(storage_key)
        temporary_path = path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(
                {
                    "storage_key": storage_key,
                    "namespace_id": namespace_id,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)

    def clear_restore_pending(self, storage_key: str) -> None:
        self._restore_pending_path(storage_key).unlink(missing_ok=True)

    def recover_pending_restores(self, records: Iterable[_NamespaceRecord]) -> None:
        records_by_storage_key = {record.storage_key: record for record in records}
        for child in self.restore_staging_root.iterdir():
            if child.is_symlink() or child.is_junction():
                raise NamespaceIntegrityError(
                    f"restore staging cannot use a symbolic link: {child}"
                )
            if not _STORAGE_KEY_PATTERN.fullmatch(child.name) or not child.is_dir():
                raise NamespaceIntegrityError(
                    f"restore staging contains an invalid entry: {child}"
                )
            self._assert_tree_has_no_symlinks(child)
            shutil.rmtree(child)
        for temporary_path in self.restore_pending_root.glob("*.json.tmp"):
            temporary_path.unlink()
        for marker_path in self.restore_pending_root.glob("*.json"):
            storage_key = marker_path.stem
            if not _STORAGE_KEY_PATTERN.fullmatch(storage_key):
                raise NamespaceIntegrityError(
                    f"restore pending marker has an invalid name: {marker_path}"
                )
            try:
                payload = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise NamespaceIntegrityError(
                    f"restore pending marker is invalid: {marker_path}"
                ) from exc
            if not isinstance(payload, dict) or payload != {
                "storage_key": storage_key,
                "namespace_id": payload.get("namespace_id"),
            }:
                raise NamespaceIntegrityError(
                    f"restore pending marker is invalid: {marker_path}"
                )
            namespace_id = payload.get("namespace_id")
            if not isinstance(namespace_id, str) or not namespace_id:
                raise NamespaceIntegrityError(
                    f"restore pending marker is invalid: {marker_path}"
                )
            record = records_by_storage_key.get(storage_key)
            if record is None:
                self.purge(storage_key)
            elif (
                record.info.namespace_id != namespace_id
                or record.info.status is not NamespaceStatus.INACTIVE
            ):
                raise NamespaceIntegrityError(
                    f"restore pending marker conflicts with namespace control state: {namespace_id}"
                )
            else:
                self.validate_active(storage_key)
            marker_path.unlink()

    def validate_active(self, storage_key: str) -> Path:
        path = self.active_path(storage_key)
        if not path.is_dir():
            raise NamespaceIntegrityError(f"namespace data directory is missing: {storage_key}")
        self._assert_tree_has_no_symlinks(path)
        return path

    def move_to_quarantine(self, storage_key: str) -> Path:
        source = self.active_path(storage_key)
        target = self.quarantine_path(storage_key)
        if target.exists():
            if source.exists():
                raise NamespaceIntegrityError(f"namespace has both active and quarantined data: {storage_key}")
            self._assert_tree_has_no_symlinks(target)
            return target
        if not source.is_dir():
            raise NamespaceIntegrityError(f"namespace data directory is missing: {storage_key}")
        self._assert_tree_has_no_symlinks(source)
        os.replace(source, target)
        return target

    def restore_from_quarantine(self, storage_key: str) -> Path:
        source = self.quarantine_path(storage_key)
        target = self.active_path(storage_key)
        if target.exists():
            if source.exists():
                raise NamespaceIntegrityError(f"namespace has both active and quarantined data: {storage_key}")
            self._assert_tree_has_no_symlinks(target)
            return target
        if not source.is_dir():
            raise NamespaceIntegrityError(f"namespace quarantine directory is missing: {storage_key}")
        self._assert_tree_has_no_symlinks(source)
        os.replace(source, target)
        return target

    def purge(self, storage_key: str) -> None:
        for path in (self.active_path(storage_key), self.quarantine_path(storage_key)):
            if not path.exists():
                continue
            self._assert_tree_has_no_symlinks(path)
            shutil.rmtree(path)

    def reconcile(self, record: _NamespaceRecord) -> None:
        status = record.info.status
        key = record.storage_key
        active = self.active_path(key)
        quarantined = self.quarantine_path(key)
        if active.exists() and quarantined.exists():
            raise NamespaceIntegrityError(f"namespace has two physical data roots: {record.info.namespace_id}")
        if status is NamespaceStatus.CREATING:
            if quarantined.exists():
                raise NamespaceIntegrityError(
                    f"creating namespace unexpectedly has quarantined data: {record.info.namespace_id}"
                )
            if not active.exists():
                active.mkdir(parents=False, exist_ok=False)
            self._assert_tree_has_no_symlinks(active)
            return
        if status in {NamespaceStatus.ACTIVE, NamespaceStatus.INACTIVE}:
            if not active.exists() and quarantined.exists():
                self.restore_from_quarantine(key)
            self.validate_active(key)
            return
        if status is NamespaceStatus.QUARANTINED:
            self.move_to_quarantine(key)
            return
        if status is NamespaceStatus.PURGING:
            self.purge(key)

    def storage_bytes(self, storage_key: str) -> int:
        path = self.active_path(storage_key)
        if not path.exists():
            path = self.quarantine_path(storage_key)
        if not path.exists():
            return 0
        self._assert_tree_has_no_symlinks(path)
        total = 0
        for root, _, files in os.walk(path, followlinks=False):
            for filename in files:
                total += (Path(root) / filename).stat().st_size
        return total

    def _storage_path(self, parent: Path, storage_key: str) -> Path:
        if not _STORAGE_KEY_PATTERN.fullmatch(storage_key):
            raise NamespaceIntegrityError(f"invalid internal namespace storage key: {storage_key}")
        path = parent / storage_key
        self._assert_within(path, parent)
        if _is_link_or_junction(path):
            raise NamespaceIntegrityError(f"namespace data root cannot be a symbolic link: {path}")
        return path

    def _restore_pending_path(self, storage_key: str) -> Path:
        if not _STORAGE_KEY_PATTERN.fullmatch(storage_key):
            raise NamespaceIntegrityError(
                f"invalid internal namespace storage key: {storage_key}"
            )
        path = self.restore_pending_root / f"{storage_key}.json"
        self._assert_within(path, self.restore_pending_root)
        return path

    @staticmethod
    def _assert_within(path: Path, parent: Path) -> None:
        resolved_parent = parent.resolve()
        resolved_path = path.resolve(strict=False)
        try:
            resolved_path.relative_to(resolved_parent)
        except ValueError as exc:
            raise NamespaceIntegrityError(f"namespace path escapes its storage root: {path}") from exc

    @staticmethod
    def _assert_tree_has_no_symlinks(path: Path) -> None:
        if _is_link_or_junction(path):
            raise NamespaceIntegrityError(f"namespace data cannot use symbolic links: {path}")
        for child in path.rglob("*"):
            if _is_link_or_junction(child):
                raise NamespaceIntegrityError(f"namespace data cannot use symbolic links: {child}")


def _is_link_or_junction(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()
