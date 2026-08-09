"""SQLite-backed namespace control plane."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import json
import sqlite3

from a_memorix.contracts import (
    ApiKeyInfo,
    CreateNamespaceRequest,
    ErrorCode,
    ErrorEnvelope,
    JobInfo,
    JobStatus,
    JobType,
    MigrationRequiredError,
    NamespaceConfig,
    NamespaceConflictError,
    NamespaceInfo,
    NamespaceNotFoundError,
    NamespaceQuota,
    NamespaceStateError,
    NamespaceStatus,
    NotFoundError,
)

from ..storage.sqlite_connection import SQLiteConnectionManager


CONTROL_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class _NamespaceRecord:
    info: NamespaceInfo
    storage_key: str


@dataclass(frozen=True)
class IdempotencyDecision:
    execute: bool
    response: Mapping[str, object] | None = None


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
                config_json TEXT NOT NULL,
                config_version INTEGER NOT NULL,
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
            CREATE TABLE IF NOT EXISTS idempotency_records (
                namespace_id TEXT NOT NULL COLLATE BINARY,
                operation TEXT NOT NULL COLLATE BINARY,
                idempotency_key TEXT NOT NULL COLLATE BINARY,
                request_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                response_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                PRIMARY KEY(namespace_id, operation, idempotency_key),
                FOREIGN KEY(namespace_id) REFERENCES namespaces(namespace_id)
                    ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_idempotency_expires_at
                ON idempotency_records(expires_at);
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY COLLATE BINARY,
                namespace_id TEXT NOT NULL COLLATE BINARY,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                progress REAL NOT NULL,
                payload_json TEXT NOT NULL,
                result_json TEXT,
                error_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                started_at REAL,
                completed_at REAL,
                FOREIGN KEY(namespace_id) REFERENCES namespaces(namespace_id)
                    ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_namespace_created
                ON jobs(namespace_id, created_at, job_id);
            CREATE INDEX IF NOT EXISTS idx_jobs_status
                ON jobs(status);
            """
        )
        namespace_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(namespaces)").fetchall()
        }
        if "config_json" not in namespace_columns:
            connection.execute(
                "ALTER TABLE namespaces ADD COLUMN config_json TEXT NOT NULL DEFAULT '{}'"
            )
        if "config_version" not in namespace_columns:
            connection.execute(
                "ALTER TABLE namespaces ADD COLUMN config_version INTEGER NOT NULL DEFAULT 1"
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
        return self._insert_namespace(
            request,
            storage_key=storage_key,
            status=NamespaceStatus.CREATING,
            now=now,
        )

    def create_restored(
        self,
        request: CreateNamespaceRequest,
        *,
        storage_key: str,
        now: float,
    ) -> _NamespaceRecord:
        return self._insert_namespace(
            request,
            storage_key=storage_key,
            status=NamespaceStatus.INACTIVE,
            now=now,
        )

    def namespace_exists(self, namespace_id: str) -> bool:
        row = (
            self._connections.connection()
            .execute(
                "SELECT 1 FROM namespaces WHERE namespace_id = ?",
                (namespace_id,),
            )
            .fetchone()
        )
        return row is not None

    def _insert_namespace(
        self,
        request: CreateNamespaceRequest,
        *,
        storage_key: str,
        status: NamespaceStatus,
        now: float,
    ) -> _NamespaceRecord:
        quota_json = request.quota.model_dump_json()
        config_json = request.config.model_dump_json()
        try:
            with self._connections.transaction(immediate=True) as connection:
                connection.execute(
                    """
                    INSERT INTO namespaces (
                        namespace_id, storage_key, status, created_at, updated_at,
                        last_active_at, version, quota_json, config_json,
                        config_version, purge_after
                    ) VALUES (?, ?, ?, ?, ?, NULL, 1, ?, ?, 1, NULL)
                    """,
                    (
                        request.namespace_id,
                        storage_key,
                        status.value,
                        now,
                        now,
                        quota_json,
                        config_json,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise NamespaceConflictError(
                f"namespace already exists: {request.namespace_id}",
                details={"namespace_id": request.namespace_id},
            ) from exc
        return self.get_record(request.namespace_id)

    def update_config(
        self,
        namespace_id: str,
        *,
        config: NamespaceConfig,
        expected_config_version: int | None,
        now: float,
    ) -> _NamespaceRecord:
        with self._connections.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT status, config_version FROM namespaces WHERE namespace_id = ?",
                (namespace_id,),
            ).fetchone()
            if row is None:
                raise NamespaceNotFoundError(
                    f"namespace not found: {namespace_id}",
                    details={"namespace_id": namespace_id},
                )
            status = NamespaceStatus(str(row["status"]))
            if status is not NamespaceStatus.INACTIVE:
                raise NamespaceStateError(
                    "namespace configuration can only be changed while inactive",
                    details={"namespace_id": namespace_id, "status": status.value},
                )
            current_version = int(row["config_version"])
            if (
                expected_config_version is not None
                and expected_config_version != current_version
            ):
                raise NamespaceConflictError(
                    "namespace configuration version does not match",
                    details={
                        "namespace_id": namespace_id,
                        "expected_config_version": expected_config_version,
                        "current_config_version": current_version,
                    },
                )
            connection.execute(
                """
                UPDATE namespaces
                SET config_json = ?, config_version = config_version + 1,
                    version = version + 1, updated_at = ?
                WHERE namespace_id = ?
                """,
                (config.model_dump_json(), now, namespace_id),
            )
        return self.get_record(namespace_id)

    def get_record(self, namespace_id: str) -> _NamespaceRecord:
        row = (
            self._connections.connection()
            .execute(
                "SELECT * FROM namespaces WHERE namespace_id = ?",
                (namespace_id,),
            )
            .fetchone()
        )
        if row is None:
            raise NamespaceNotFoundError(
                f"namespace not found: {namespace_id}",
                details={"namespace_id": namespace_id},
            )
        return self._from_row(row)

    def list_records(self) -> list[_NamespaceRecord]:
        rows = (
            self._connections.connection()
            .execute("SELECT * FROM namespaces ORDER BY namespace_id")
            .fetchall()
        )
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
        row = (
            self._connections.connection()
            .execute(
                """
            SELECT key_id, namespace_id, label, created_at, expires_at,
                   revoked_at, last_used_at
            FROM api_keys
            WHERE namespace_id = ? AND key_id = ?
            """,
                (namespace_id, key_id),
            )
            .fetchone()
        )
        if row is None:
            raise NotFoundError(
                f"API key not found: {key_id}",
                details={"namespace_id": namespace_id, "key_id": key_id},
            )
        return self._api_key_from_row(row)

    def list_api_keys(self, namespace_id: str) -> list[ApiKeyInfo]:
        rows = (
            self._connections.connection()
            .execute(
                """
            SELECT key_id, namespace_id, label, created_at, expires_at,
                   revoked_at, last_used_at
            FROM api_keys
            WHERE namespace_id = ?
            ORDER BY created_at, key_id
            """,
                (namespace_id,),
            )
            .fetchall()
        )
        return [self._api_key_from_row(row) for row in rows]

    def authenticate_api_key(
        self,
        secret_hash: bytes,
        *,
        now: float,
    ) -> ApiKeyInfo | None:
        row = (
            self._connections.connection()
            .execute(
                """
            SELECT key_id, namespace_id, label, created_at, expires_at,
                   revoked_at, last_used_at
            FROM api_keys
            WHERE secret_hash = ?
              AND revoked_at IS NULL
              AND (expires_at IS NULL OR expires_at > ?)
            """,
                (secret_hash, now),
            )
            .fetchone()
        )
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
            row = (
                self._connections.connection()
                .execute(
                    """
                SELECT key_id, namespace_id, label, created_at, expires_at,
                       revoked_at, last_used_at
                FROM api_keys
                WHERE key_id = ?
                  AND revoked_at IS NULL
                  AND (expires_at IS NULL OR expires_at > ?)
                """,
                    (str(row["key_id"]), now),
                )
                .fetchone()
            )
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

    def claim_idempotency(
        self,
        *,
        namespace_id: str,
        operation: str,
        idempotency_key: str,
        request_hash: str,
        now: float,
        expires_at: float,
    ) -> IdempotencyDecision:
        with self._connections.transaction(immediate=True) as connection:
            connection.execute(
                "DELETE FROM idempotency_records WHERE expires_at <= ?",
                (now,),
            )
            row = connection.execute(
                """
                SELECT request_hash, status, response_json
                FROM idempotency_records
                WHERE namespace_id = ? AND operation = ? AND idempotency_key = ?
                """,
                (namespace_id, operation, idempotency_key),
            ).fetchone()
            if row is not None:
                if str(row["request_hash"]) != request_hash:
                    raise NamespaceConflictError(
                        "idempotency key was already used for a different request",
                        details={
                            "namespace_id": namespace_id,
                            "operation": operation,
                            "idempotency_key": idempotency_key,
                        },
                    )
                if str(row["status"]) == "completed":
                    raw_response = json.loads(str(row["response_json"] or "{}"))
                    response = raw_response if isinstance(raw_response, dict) else {}
                    return IdempotencyDecision(execute=False, response=response)
                connection.execute(
                    """
                    UPDATE idempotency_records
                    SET updated_at = ?, expires_at = ?
                    WHERE namespace_id = ? AND operation = ? AND idempotency_key = ?
                    """,
                    (now, expires_at, namespace_id, operation, idempotency_key),
                )
                return IdempotencyDecision(execute=True)
            connection.execute(
                """
                INSERT INTO idempotency_records (
                    namespace_id, operation, idempotency_key, request_hash,
                    status, response_json, created_at, updated_at, expires_at
                ) VALUES (?, ?, ?, ?, 'running', NULL, ?, ?, ?)
                """,
                (
                    namespace_id,
                    operation,
                    idempotency_key,
                    request_hash,
                    now,
                    now,
                    expires_at,
                ),
            )
        return IdempotencyDecision(execute=True)

    def complete_idempotency(
        self,
        *,
        namespace_id: str,
        operation: str,
        idempotency_key: str,
        request_hash: str,
        response: Mapping[str, object],
        now: float,
        expires_at: float,
    ) -> None:
        with self._connections.transaction(immediate=True) as connection:
            cursor = connection.execute(
                """
                UPDATE idempotency_records
                SET status = 'completed', response_json = ?, updated_at = ?, expires_at = ?
                WHERE namespace_id = ? AND operation = ? AND idempotency_key = ?
                  AND request_hash = ?
                """,
                (
                    json.dumps(
                        dict(response), ensure_ascii=False, separators=(",", ":")
                    ),
                    now,
                    expires_at,
                    namespace_id,
                    operation,
                    idempotency_key,
                    request_hash,
                ),
            )
            if cursor.rowcount != 1:
                raise NamespaceConflictError(
                    "idempotency claim no longer exists",
                    details={
                        "namespace_id": namespace_id,
                        "operation": operation,
                        "idempotency_key": idempotency_key,
                    },
                )

    def abandon_idempotency(
        self,
        *,
        namespace_id: str,
        operation: str,
        idempotency_key: str,
        request_hash: str,
    ) -> None:
        with self._connections.transaction(immediate=True) as connection:
            connection.execute(
                """
                DELETE FROM idempotency_records
                WHERE namespace_id = ? AND operation = ? AND idempotency_key = ?
                  AND request_hash = ? AND status = 'running'
                """,
                (namespace_id, operation, idempotency_key, request_hash),
            )

    def create_job(
        self,
        *,
        job_id: str,
        namespace_id: str,
        job_type: JobType,
        payload: Mapping[str, object],
        now: float,
    ) -> JobInfo:
        with self._connections.transaction(immediate=True) as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, namespace_id, job_type, status, progress,
                    payload_json, result_json, error_json, created_at,
                    updated_at, started_at, completed_at
                ) VALUES (?, ?, ?, ?, 0, ?, NULL, NULL, ?, ?, NULL, NULL)
                """,
                (
                    job_id,
                    namespace_id,
                    job_type.value,
                    JobStatus.PENDING.value,
                    json.dumps(
                        dict(payload), ensure_ascii=False, separators=(",", ":")
                    ),
                    now,
                    now,
                ),
            )
        return self.get_job(namespace_id, job_id)

    def get_job(self, namespace_id: str, job_id: str) -> JobInfo:
        row = (
            self._connections.connection()
            .execute(
                "SELECT * FROM jobs WHERE namespace_id = ? AND job_id = ?",
                (namespace_id, job_id),
            )
            .fetchone()
        )
        if row is None:
            raise NotFoundError(
                f"job not found: {job_id}",
                details={"namespace_id": namespace_id, "job_id": job_id},
            )
        return self._job_from_row(row)

    def list_jobs(self, namespace_id: str) -> list[JobInfo]:
        rows = (
            self._connections.connection()
            .execute(
                """
            SELECT * FROM jobs
            WHERE namespace_id = ?
            ORDER BY created_at, job_id
            """,
                (namespace_id,),
            )
            .fetchall()
        )
        return [self._job_from_row(row) for row in rows]

    def start_job(self, namespace_id: str, job_id: str, *, now: float) -> JobInfo:
        with self._connections.transaction(immediate=True) as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = ?, started_at = ?, updated_at = ?
                WHERE namespace_id = ? AND job_id = ? AND status = ?
                """,
                (
                    JobStatus.RUNNING.value,
                    now,
                    now,
                    namespace_id,
                    job_id,
                    JobStatus.PENDING.value,
                ),
            )
            if cursor.rowcount != 1:
                return self.get_job(namespace_id, job_id)
        return self.get_job(namespace_id, job_id)

    def complete_job(
        self,
        namespace_id: str,
        job_id: str,
        *,
        result: Mapping[str, object],
        now: float,
    ) -> JobInfo:
        with self._connections.transaction(immediate=True) as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, progress = 1, result_json = ?, error_json = NULL,
                    updated_at = ?, completed_at = ?
                WHERE namespace_id = ? AND job_id = ? AND status = ?
                """,
                (
                    JobStatus.SUCCEEDED.value,
                    json.dumps(dict(result), ensure_ascii=False, separators=(",", ":")),
                    now,
                    now,
                    namespace_id,
                    job_id,
                    JobStatus.RUNNING.value,
                ),
            )
        return self.get_job(namespace_id, job_id)

    def fail_job(
        self,
        namespace_id: str,
        job_id: str,
        *,
        error: ErrorEnvelope,
        now: float,
    ) -> JobInfo:
        with self._connections.transaction(immediate=True) as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, error_json = ?, updated_at = ?, completed_at = ?
                WHERE namespace_id = ? AND job_id = ?
                  AND status IN (?, ?)
                """,
                (
                    JobStatus.FAILED.value,
                    error.model_dump_json(),
                    now,
                    now,
                    namespace_id,
                    job_id,
                    JobStatus.PENDING.value,
                    JobStatus.RUNNING.value,
                ),
            )
        return self.get_job(namespace_id, job_id)

    def cancel_job(self, namespace_id: str, job_id: str, *, now: float) -> JobInfo:
        with self._connections.transaction(immediate=True) as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, updated_at = ?, completed_at = ?
                WHERE namespace_id = ? AND job_id = ? AND status = ?
                """,
                (
                    JobStatus.CANCELLED.value,
                    now,
                    now,
                    namespace_id,
                    job_id,
                    JobStatus.PENDING.value,
                ),
            )
        return self.get_job(namespace_id, job_id)

    def fail_interrupted_jobs(self, *, now: float) -> int:
        error = ErrorEnvelope(
            code=ErrorCode.CAPABILITY_UNAVAILABLE,
            message="job was interrupted by a service restart",
            retryable=True,
        )
        with self._connections.transaction(immediate=True) as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = ?, error_json = ?, updated_at = ?, completed_at = ?
                WHERE status IN (?, ?)
                """,
                (
                    JobStatus.FAILED.value,
                    error.model_dump_json(),
                    now,
                    now,
                    JobStatus.PENDING.value,
                    JobStatus.RUNNING.value,
                ),
            )
        return max(0, cursor.rowcount)

    @staticmethod
    def _from_row(row: sqlite3.Row) -> _NamespaceRecord:
        quota_data = json.loads(str(row["quota_json"] or "{}"))
        config_data = json.loads(str(row["config_json"] or "{}"))
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
            config_version=int(row["config_version"]),
            quota=NamespaceQuota.model_validate(quota_data),
            config=NamespaceConfig.model_validate(config_data),
            purge_after=(
                _utc_datetime(float(row["purge_after"]))
                if row["purge_after"] is not None
                else None
            ),
        )
        return _NamespaceRecord(info=info, storage_key=str(row["storage_key"]))

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> JobInfo:
        result_data = json.loads(str(row["result_json"] or "{}"))
        raw_error = str(row["error_json"] or "")
        error = ErrorEnvelope.model_validate_json(raw_error) if raw_error else None
        return JobInfo(
            job_id=str(row["job_id"]),
            namespace_id=str(row["namespace_id"]),
            job_type=JobType(str(row["job_type"])),
            status=JobStatus(str(row["status"])),
            progress=float(row["progress"]),
            created_at=_utc_datetime(float(row["created_at"])),
            updated_at=_utc_datetime(float(row["updated_at"])),
            started_at=(
                _utc_datetime(float(row["started_at"]))
                if row["started_at"] is not None
                else None
            ),
            completed_at=(
                _utc_datetime(float(row["completed_at"]))
                if row["completed_at"] is not None
                else None
            ),
            result=result_data if isinstance(result_data, dict) else {},
            error=error,
        )

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
