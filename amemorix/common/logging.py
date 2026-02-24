"""Local logging helpers for standalone runtime."""

from __future__ import annotations

import logging
import os
from typing import Optional

_CONFIGURED = False


def setup_logging(level: Optional[str] = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved = (level or os.getenv("AMEMORIX_LOG_LEVEL") or "INFO").upper().strip()
    if resolved not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
        resolved = "INFO"

    logging.basicConfig(
        level=getattr(logging, resolved, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)

