"""Person profile API orchestration service."""

from __future__ import annotations

from typing import Any, Dict, Optional

from amemorix.context import AppContext


class PersonProfileApiService:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx

    async def query(
        self,
        *,
        person_id: str = "",
        person_keyword: str = "",
        top_k: int = 12,
        force_refresh: bool = False,
        source_note: str = "v1:person_query",
    ) -> Dict[str, Any]:
        ttl_minutes = float(self.ctx.get_config("person_profile.profile_ttl_minutes", 360))
        return await self.ctx.person_profile_service.query_person_profile(
            person_id=str(person_id or "").strip(),
            person_keyword=str(person_keyword or "").strip(),
            top_k=max(4, int(top_k or 12)),
            ttl_seconds=max(60.0, ttl_minutes * 60.0),
            force_refresh=bool(force_refresh),
            source_note=source_note,
        )

    async def set_override(self, *, person_id: str, override_text: str, updated_by: str = "v1") -> Dict[str, Any]:
        pid = self.ctx.person_profile_service.resolve_person_id(person_id) or str(person_id or "").strip()
        if not pid:
            raise ValueError("person_id is empty")
        override = self.ctx.metadata_store.set_person_profile_override(
            person_id=pid,
            override_text=str(override_text or ""),
            updated_by=str(updated_by or "v1"),
            source="v1",
        )
        profile = await self.query(person_id=pid, source_note="v1:override")
        return {"success": True, "person_id": pid, "override": override, "profile": profile}

    async def delete_override(self, *, person_id: str) -> Dict[str, Any]:
        pid = self.ctx.person_profile_service.resolve_person_id(person_id) or str(person_id or "").strip()
        if not pid:
            raise ValueError("person_id is empty")
        deleted = self.ctx.metadata_store.delete_person_profile_override(pid)
        return {"success": True, "person_id": pid, "deleted": bool(deleted)}

    async def upsert_registry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.ctx.metadata_store.upsert_person_registry(
            person_id=str(payload.get("person_id", "")).strip(),
            person_name=str(payload.get("person_name", "")).strip(),
            nickname=str(payload.get("nickname", "")).strip(),
            user_id=str(payload.get("user_id", "")).strip(),
            platform=str(payload.get("platform", "")).strip(),
            group_nick_name=payload.get("group_nick_name"),
            memory_points=payload.get("memory_points"),
            last_know=payload.get("last_know"),
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        )

    async def list_registry(self, *, keyword: str = "", page: int = 1, page_size: Optional[int] = None) -> Dict[str, Any]:
        default_size = int(self.ctx.get_config("person_profile.registry.page_size_default", 20))
        max_size = int(self.ctx.get_config("person_profile.registry.page_size_max", 100))
        size = page_size if page_size is not None else default_size
        size = max(1, min(max_size, int(size)))
        return self.ctx.metadata_store.list_person_registry(keyword=keyword, page=page, page_size=size)

