"""SQLite-backed namespace control plane."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import json
import sqlite3

from a_memorix.contracts import (
    ApiKeyInfo,
    CreateNamespaceRequest,
    MigrationRequiredError,
    NamespaceConflictError,
    NamespaceInfo,
    NamespaceNotFoundError,
    NamespaceQuota,
    NamespaceStateError,
    NamespaceStatus,
    NotFoundError,
)

from ..storage.sqlite_connection import SQLiteConnectionManager


CONTROL_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class _NamespaceRecord:
    info: NamespaceInfo
    storage_key: str


class NamespaceControlStore:
    """Store namespace metadata without mixing in memory-domain data."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connections = SQLiteConnectionManager(self.db_path)
        try:
            self._initialize_schema()
        except BaseException:
            self._connections.close_all()
            raise

    def _initialize_schema(self) -> None:
        connection = self._connections.connection()
        current_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if current_version > CONTROL_SCHEMA_VERSION:
            raise MigrationRequiredError(
                "namespace control database is newer than this A_memorix version",
                details={
                    "current_schema_version": current_version,
                    "supported_schema_version": CONTROL_SCHEMA_VERSION,
                },
            )
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS namespaces (
                namespace_id TEXT PRIMARY KEY COLLATE BINARY,
                storage_key TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                last_active_at REAL,
                version INTEGER NOT NULL,
                quota_json TEXT NOT NULL,
                purge_after REAL
            );
            CREATE INDEX IF NOT EXISTS idx_namespaces_status
                ON namespaces(status);
            CREATE INDEX IF NOT EXISTS idx_namespaces_purge_after
                ON namespaces(purge_after);
            CREATE TABLE IF NOT EXISTS api_keys (
                key_id TEXT PRIMARY KEY COLLATE BINARY,
                namespace_id TEXT NOT NULL COLLATE BINARY,
                secret_hash BLOB NOT NULL UNIQUE,
                label TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL,
                revoked_at REAL,
                last_used_at REAL,
                FOREIGN KEY(namespace_id) REFERENCES namespaces(namespace_id)
                    ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_api_keys_namespace
                ON api_keys(namespace_id, created_at);
            """
        )
        if current_version < CONTROL_SCHEMA_VERSION:
            connection.execute(f"PRAGMA user_version = {CONTROL_SCHEMA_VERSION}")
        connection.commit()

    def close(self) -> None:
        self._connections.close_all()

    def create(
        self,
        request: CreateNamespaceRequest,
        *,
        storage_key: str,
        now: float,
    ) -> _NamespaceRecord:
        quota_json = request.quota.model_dump_json()
        try:
            with self._connections.transaction(immediate=True) as connection:
                connection.execute(
                    """
                    INSERT INTO namespaces (
                        namespace_id, storage_key, status, created_at, updated_at,
                        last_active_at, version, quota_json, purge_after
                    ) VALUES (?, ?, ?, ?, ?, NULL, 1, ?, NULL)
                    """,
                    (
                        request.namespace_id,
                        storage_key,
                        NamespaceStatus.CREATING.value,
                        now,
                        now,
                        quota_json,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise NamespaceConflictError(
                f"namespace already exists: {request.namespace_id}",
                details={"namespace_id": request.namespace_id},
            ) from exc
        return self.get_record(request.namespace_id)

    def get_record(self, namespace_id: str) -> _NamespaceRecord:
        row = self._connections.connection().execute(
            "SELECT * FROM namespaces WHERE namespace_id = ?",
            (namespace_id,),
        ).fetchone()
        if row is None:
            raise NamespaceNotFoundError(
                f"namespace not found: {namespace_id}",
                details={"namespace_id": namespace_id},
            )
        return self._from_row(row)

    def list_records(self) -> list[_NamespaceRecord]:
        rows = self._connections.connection().execute(
            "SELECT * FROM namespaces ORDER BY namespace_id"
        ).fetchall()
        return [self._from_row(row) for row in rows]

    def transition(
        self,
        namespace_id: str,
        *,
        expected: Iterable[NamespaceStatus],
        target: NamespaceStatus,
        now: float,
        purge_after: float | None = None,
    ) -> _NamespaceRecord:
        expected_statuses = frozenset(expected)
        with self._connections.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM namespaces WHERE namespace_id = ?",
                (namespace_id,),
            ).fetchone()
            if row is None:
                raise NamespaceNotFoundError(
                    f"namespace not found: {namespace_id}",
                    details={"namespace_id": namespace_id},
                )
            current = NamespaceStatus(str(row["status"]))
            if current not in expected_statuses:
                raise NamespaceStateError(
                    f"cannot change namespace {namespace_id} from {current.value} to {target.value}",
                    details={
                        "namespace_id": namespace_id,
                        "current_status": current.value,
                        "target_status": target.value,
                    },
                )
            connection.execute(
                """
                UPDATE namespaces
                SET status = ?, updated_at = ?, version = version + 1,
                    purge_after = ?
                WHERE namespace_id = ?
                """,
                (target.value, now, purge_after, namespace_id),
            )
        return self.get_record(namespace_id)

    def touch(self, namespace_id: str, *, now: float) -> None:
        with self._connections.transaction(immediate=True) as connection:
            connection.execute(
                """
                UPDATE namespaces
                SET last_active_at = ?, updated_at = ?
                WHERE namespace_id = ? AND status = ?
                """,
                (now, now, namespace_id, NamespaceStatus.ACTIVE.value),
            )

    def remove_purging(self, namespace_id: str) -> None:
        with self._connections.transaction(immediate=True) as connection:
            cursor = connection.execute(
                "DELETE FROM namespaces WHERE namespace_id = ? AND status = ?",
                (namespace_id, NamespaceStatus.PURGING.value),
            )
            if cursor.rowcount != 1:
                raise NamespaceStateError(
                    f"namespace is not ready to purge: {namespace_id}",
                    details={"namespace_id": namespace_id},
                )

    def create_api_key(
        self,
        *,
        key_id: str,
        namespace_id: str,
        secret_hash: bytes,
        label: str,
        created_at: float,
        expires_at: float | None,
    ) -> ApiKeyInfo:
        try:
            with self._connections.transaction(immediate=True) as connection:
                connection.execute(
                    """
                    INSERT INTO api_keys (
                        key_id, namespace_id, secret_hash, label, created_at,
                        expires_at, revoked_at, last_used_at
                    ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)
                    """,
                    (
                        key_id,
                        namespace_id,
                        secret_hash,
                        label,
                        created_at,
                        expires_at,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise NamespaceConflictError(
                "cannot create API key",
                details={"namespace_id": namespace_id, "key_id": key_id},
            ) from exc
        return self.get_api_key(namespace_id, key_id)

    def get_api_key(self, namespace_id: str, key_id: str) -> ApiKeyInfo:
        row = self._connections.connection().execute(
            """
            SELECT key_id, namespace_id, label, created_at, expires_at,
                   revoked_at, last_used_at
            FROM api_keys
            WHERE namespace_id = ? AND key_id = ?
            """,
            (namespace_id, key_id),
        ).fetchone()
        if row is None:
            raise NotFoundError(
                f"API key not found: {key_id}",
                details={"namespace_id": namespace_id, "key_id": key_id},
            )
        return self._api_key_from_row(row)

    def list_api_keys(self, namespace_id: str) -> list[ApiKeyInfo]:
        rows = self._connections.connection().execute(
            """
            SELECT key_id, namespace_id, label, created_at, expires_at,
                   revoked_at, last_used_at
            FROM api_keys
            WHERE namespace_id = ?
            ORDER BY created_at, key_id
            """,
            (namespace_id,),
        ).fetchall()
        return [self._api_key_from_row(row) for row in rows]

    def authenticate_api_key(
        self,
        secret_hash: bytes,
        *,
        now: float,
    ) -> ApiKeyInfo | None:
        row = self._connections.connection().execute(
            """
            SELECT key_id, namespace_id, label, created_at, expires_at,
                   revoked_at, last_used_at
            FROM api_keys
            WHERE secret_hash = ?
              AND revoked_at IS NULL
              AND (expires_at IS NULL OR expires_at > ?)
            """,
            (secret_hash, now),
        ).fetchone()
        if row is None:
            return None
        last_used_at = row["last_used_at"]
        if last_used_at is None or now - float(last_used_at) >= 60.0:
            with self._connections.transaction(immediate=True) as connection:
                connection.execute(
                    """
                    UPDATE api_keys
                    SET last_used_at = ?
                    WHERE key_id = ? AND revoked_at IS NULL
                    """,
                    (now, str(row["key_id"])),
                )
            row = self._connections.connection().execute(
                """
                SELECT key_id, namespace_id, label, created_at, expires_at,
                       revoked_at, last_used_at
                FROM api_keys
                WHERE key_id = ?
                  AND revoked_at IS NULL
                  AND (expires_at IS NULL OR expires_at > ?)
                """,
                (str(row["key_id"]), now),
            ).fetchone()
            if row is None:
                return None
        return self._api_key_from_row(row)

    def revoke_api_key(
        self,
        namespace_id: str,
        key_id: str,
        *,
        now: float,
    ) -> ApiKeyInfo:
        current = self.get_api_key(namespace_id, key_id)
        if current.revoked_at is not None:
            return current
        with self._connections.transaction(immediate=True) as connection:
            connection.execute(
                """
                UPDATE api_keys
                SET revoked_at = ?
                WHERE namespace_id = ? AND key_id = ? AND revoked_at IS NULL
                """,
                (now, namespace_id, key_id),
            )
        return self.get_api_key(namespace_id, key_id)

    @staticmethod
    def _from_row(row: sqlite3.Row) -> _NamespaceRecord:
        quota_data = json.loads(str(row["quota_json"] or "{}"))
        info = NamespaceInfo(
            namespace_id=str(row["namespace_id"]),
            status=NamespaceStatus(str(row["status"])),
            created_at=_utc_datetime(float(row["created_at"])),
            updated_at=_utc_datetime(float(row["updated_at"])),
            last_active_at=(
                _utc_datetime(float(row["last_active_at"]))
                if row["last_active_at"] is not None
                else None
            ),
            version=int(row["version"]),
            quota=NamespaceQuota.model_validate(quota_data),
            purge_after=(
                _utc_datetime(float(row["purge_after"]))
                if row["purge_after"] is not None
                else None
            ),
        )
        return _NamespaceRecord(info=info, storage_key=str(row["storage_key"]))

    @staticmethod
    def _api_key_from_row(row: sqlite3.Row) -> ApiKeyInfo:
        return ApiKeyInfo(
            key_id=str(row["key_id"]),
            namespace_id=str(row["namespace_id"]),
            label=str(row["label"]),
            created_at=_utc_datetime(float(row["created_at"])),
            expires_at=(
                _utc_datetime(float(row["expires_at"]))
                if row["expires_at"] is not None
                else None
            ),
            revoked_at=(
                _utc_datetime(float(row["revoked_at"]))
                if row["revoked_at"] is not None
                else None
            ),
            last_used_at=(
                _utc_datetime(float(row["last_used_at"]))
                if row["last_used_at"] is not None
                else None
            ),
        )


def _utc_datetime(timestamp: float) -> datetime:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)
