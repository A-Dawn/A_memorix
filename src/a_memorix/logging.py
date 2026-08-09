"""Library-local logging access without host application coupling."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a standard logger without changing application logging policy."""

    return logging.getLogger(name)
