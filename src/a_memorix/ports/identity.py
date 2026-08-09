"""External identity lookup contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class IdentityRecord:
    person_id: str
    display_name: str = ""
    aliases: tuple[str, ...] = ()


@runtime_checkable
class IdentityResolver(Protocol):
    """Resolve host identities without exposing host database models."""

    def resolve(self, key: str) -> IdentityRecord | None:
        """Resolve an external ID, display name or alias."""

    def aliases(self, person_id: str) -> Sequence[str]:
        """Return known aliases for one stable person ID."""
