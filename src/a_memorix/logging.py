"""Library-local logging access without host application coupling."""

from __future__ import annotations

from datetime import datetime, timezone

import json
import logging


def get_logger(name: str) -> logging.Logger:
    """Return a standard logger without changing application logging policy."""

    return logging.getLogger(name)


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                tz=timezone.utc,
            ).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in (
            "request_id",
            "trace_id",
            "namespace_id",
            "rpc_method",
            "rpc_status",
            "duration_ms",
        ):
            value = getattr(record, field, None)
            if value not in {None, ""}:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def configure_logging(level: str = "INFO", log_format: str = "json") -> None:
    normalized_level = str(level or "INFO").upper()
    if normalized_level not in logging.getLevelNamesMapping():
        raise ValueError(f"invalid log level: {level}")
    handler = logging.StreamHandler()
    if log_format == "json":
        handler.setFormatter(JSONFormatter())
    elif log_format == "text":
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    else:
        raise ValueError(f"invalid log format: {log_format}")
    logging.basicConfig(
        level=normalized_level,
        handlers=[handler],
        force=True,
    )
