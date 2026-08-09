"""Clock contract used by lifecycle and resource-management code."""

from __future__ import annotations

from time import monotonic, time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    def time(self) -> float:
        """Return Unix time in seconds."""

    def monotonic(self) -> float:
        """Return a monotonic timestamp for elapsed-time calculations."""


class SystemClock:
    def time(self) -> float:
        return time()

    def monotonic(self) -> float:
        return monotonic()
