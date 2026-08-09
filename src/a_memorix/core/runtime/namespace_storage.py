"""Safe physical layout for namespace-owned files."""

from __future__ import annotations

from pathlib import Path

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

    @property
    def control_db_path(self) -> Path:
        return self.control_root / "namespaces.db"

    def initialize(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        for path in (self.control_root, self.namespaces_root, self.quarantine_root):
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
