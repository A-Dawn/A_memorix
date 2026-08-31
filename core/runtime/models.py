from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


_ALLOWED_SECURITY_DOMAINS = {"normal", "kami"}


@dataclass(frozen=True)
class MemoryAccessScope:
    """Trusted retrieval/write scope for logical memory isolation.

    ``force_all_memory_access`` is intentionally not part of user-facing payloads;
    hosts must only construct it after authenticating a trusted Kami session.
    """

    allowed_memory_space_ids: tuple[str, ...] = ()
    allowed_partition_ids: tuple[str, ...] = ()
    security_domain: str = "normal"
    force_all_memory_access: bool = False
    access_trace_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_memory_space_ids", _clean(self.allowed_memory_space_ids))
        object.__setattr__(self, "allowed_partition_ids", _clean(self.allowed_partition_ids))
        domain = str(self.security_domain or "normal").strip().lower()
        if domain not in _ALLOWED_SECURITY_DOMAINS:
            raise ValueError(f"unsupported security_domain: {domain}")
        if domain == "normal" and self.force_all_memory_access:
            raise ValueError("force_all_memory_access requires security_domain='kami'")
        if domain == "normal" and any(item == "*" for item in self.allowed_memory_space_ids + self.allowed_partition_ids):
            raise ValueError("normal scope cannot use wildcard access")
        object.__setattr__(self, "security_domain", domain)
        object.__setattr__(self, "force_all_memory_access", bool(self.force_all_memory_access))
        object.__setattr__(self, "access_trace_id", str(self.access_trace_id or "").strip())

    @classmethod
    def normal(cls) -> "MemoryAccessScope":
        return cls()

    @classmethod
    def trusted_kami(cls, *, access_trace_id: str = "") -> "MemoryAccessScope":
        return cls(security_domain="kami", force_all_memory_access=True, access_trace_id=access_trace_id)

    def allows(self, *, memory_space_id: str, partition_id: str, security_domain: str = "normal") -> bool:
        if self.force_all_memory_access and self.security_domain == "kami":
            return str(security_domain or "normal").lower() == "kami"
        if str(security_domain or "normal").lower() != self.security_domain:
            return False
        spaces = set(self.allowed_memory_space_ids)
        partitions = set(self.allowed_partition_ids)
        return (not spaces or memory_space_id in spaces) and (not partitions or partition_id in partitions)


def _clean(values: Optional[Iterable[str]]) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(dict.fromkeys(str(value or "").strip() for value in values if str(value or "").strip()))
