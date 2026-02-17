"""
人物画像开关命令

/person_profile on
/person_profile off
/person_profile status
"""

from typing import Optional, Tuple

from src.common.logger import get_logger
from src.plugin_system.base.base_command import BaseCommand
from src.chat.message_receive.message import MessageRecv

logger = get_logger("A_Memorix.PersonProfileCommand")


class PersonProfileCommand(BaseCommand):
    command_name = "person_profile"
    command_description = "控制人物画像自动注入开关（按 stream_id + user_id）"
    command_pattern = r"^\/person_profile(?:\s+(?P<action>\w+))?$"

    def __init__(self, message: MessageRecv, plugin_config: Optional[dict] = None):
        super().__init__(message, plugin_config)
        self.metadata_store = self.plugin_config.get("metadata_store")
        if self.metadata_store is None:
            try:
                from ...plugin import A_MemorixPlugin

                instances = A_MemorixPlugin.get_storage_instances()
                self.metadata_store = instances.get("metadata_store")
            except Exception:
                self.metadata_store = None

    def _resolve_scope(self) -> Tuple[str, str]:
        stream_id = ""
        user_id = ""
        if self.message and self.message.chat_stream:
            stream_id = str(getattr(self.message.chat_stream, "stream_id", "") or "").strip()
            chat_user_id = str(getattr(getattr(self.message.chat_stream, "user_info", None), "user_id", "") or "").strip()
            user_id = chat_user_id
        try:
            msg_user_id = str(getattr(self.message.message_info.user_info, "user_id", "") or "").strip()
            if msg_user_id:
                user_id = msg_user_id
        except Exception:
            pass
        return stream_id, user_id

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        action = str(self.matched_groups.get("action", "status") or "status").strip().lower()
        if action not in {"on", "off", "status"}:
            return True, "用法: /person_profile on|off|status", 1

        if not bool(self.get_config("person_profile.enabled", True)):
            return False, "❌ 人物画像功能未启用（person_profile.enabled=false）", 1

        if self.metadata_store is None:
            return False, "❌ 元数据存储不可用，无法设置人物画像开关", 1

        stream_id, user_id = self._resolve_scope()
        if not stream_id or not user_id:
            return False, "❌ 无法识别当前会话范围（stream_id/user_id）", 1

        opt_in_required = bool(self.get_config("person_profile.opt_in_required", True))
        default_enabled = bool(self.get_config("person_profile.default_injection_enabled", False))

        if action == "status":
            stored_enabled = self.metadata_store.get_person_profile_switch(stream_id, user_id, default=default_enabled)
            effective_enabled = stored_enabled if opt_in_required else default_enabled
            lines = [
                "人物画像注入状态：",
                f"- stream_id: {stream_id}",
                f"- user_id: {user_id}",
                f"- opt_in_required: {opt_in_required}",
                f"- default_injection_enabled: {default_enabled}",
                f"- switch(enabled): {stored_enabled}",
                f"- effective_injection: {effective_enabled}",
            ]
            return True, "\n".join(lines), 1

        enabled = action == "on"
        self.metadata_store.set_person_profile_switch(stream_id=stream_id, user_id=user_id, enabled=enabled)
        status_text = "已开启" if enabled else "已关闭"
        return True, f"✅ 人物画像自动注入{status_text}（stream={stream_id}, user={user_id}）", 1

