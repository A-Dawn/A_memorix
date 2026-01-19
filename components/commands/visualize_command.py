"""
可视化知识图谱Command组件

生成交互式HTML知识图谱可视化文件。
"""

import time
from typing import Tuple, Optional
from pathlib import Path

from src.common.logger import get_logger
from src.plugin_system.base.base_command import BaseCommand
from src.chat.message_receive.message import MessageRecv

# 导入核心模块
from ...core import GraphStore, MetadataStore

logger = get_logger("A_Memorix.VisualizeCommand")


class VisualizeCommand(BaseCommand):
    """可视化知识图谱Command

    功能：
    - 返回可视化服务器的访问地址 (不再生成静态文件)
    """

    # Command基本信息
    command_name = "visualize"
    command_description = "获取知识图谱可视化编辑器的访问地址"
    command_pattern = r"^\/visualize(?:\s+(?P<output_path>.+))?$"

    def __init__(self, message: MessageRecv, plugin_config: Optional[dict] = None):
        """初始化可视化Command"""
        super().__init__(message, plugin_config)

        # 获取存储实例
        self.graph_store: Optional[GraphStore] = self.plugin_config.get("graph_store")
        self.metadata_store: Optional[MetadataStore] = self.plugin_config.get("metadata_store")

        # 兜底逻辑
        if not all([
            self.graph_store is not None,
            self.metadata_store is not None,
        ]):
            from ...plugin import A_MemorixPlugin
            instances = A_MemorixPlugin.get_storage_instances()
            if instances:
                self.graph_store = self.graph_store or instances.get("graph_store")
                self.metadata_store = self.metadata_store or instances.get("metadata_store")

        # 设置日志前缀
        if self.message and self.message.chat_stream:
            self.log_prefix = f"[VisualizeCommand-{self.message.chat_stream.stream_id}]"
        else:
            self.log_prefix = "[VisualizeCommand]"

    @property
    def debug_enabled(self) -> bool:
        """检查是否启用了调试模式"""
        advanced = self.plugin_config.get("advanced", {})
        if isinstance(advanced, dict):
            return advanced.get("debug", False)
        return self.plugin_config.get("debug", False)

    def _ensure_stores(self):
        """确保存储实例已加载"""
        # 再次尝试获取
        if not all([self.graph_store, self.metadata_store]):
            from ...plugin import A_MemorixPlugin
            instances = A_MemorixPlugin.get_storage_instances()
            if instances:
                self.graph_store = self.graph_store or instances.get("graph_store")
                self.metadata_store = self.metadata_store or instances.get("metadata_store")

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        """执行可视化命令

        Returns:
            Tuple[bool, Optional[str], int]: (是否成功, 回复消息, 拦截级别)
        """
        self._ensure_stores()
        
        try:
            # 尝试获取全局实例的配置
            from ...plugin import _get_global_instance
            plugin = _get_global_instance()
            
            if plugin:
                enabled = plugin.get_config("web.enabled", True)
                host = plugin.get_config("web.host", "0.0.0.0")
                port = plugin.get_config("web.port", 8082)
                
                # 处理 host 显示
                display_host = "localhost" if host == "0.0.0.0" else host
                url = f"http://{display_host}:{port}"
                
                if not enabled:
                    msg = "❌ 可视化服务器未启用，请在 config.toml 中设置 [web] enabled = true"
                    await self.send_text(msg)
                    return False, msg, 2
                
                # 检查服务器是否已启动，如果未启动则尝试启动
                if not plugin.server:
                    try:
                        logger.info(f"{self.log_prefix} 可视化服务器未运行，正在尝试启动...")
                        from ...server import MemorixServer
                        plugin.server = MemorixServer(plugin, host=host, port=port)
                        plugin.server.start()
                        # 给一点启动时间
                        import asyncio
                        await asyncio.sleep(1)
                    except Exception as e:
                        logger.error(f"{self.log_prefix} 启动可视化服务器失败: {e}")
                        msg = f"❌ 启动可视化服务器失败: {str(e)}"
                        await self.send_text(msg)
                        return False, msg, 2
                    
                result_msg = (
                    f"✅ 可视化编辑器已启动\n"
                    f"🔗 访问地址: {url}\n\n"
                )
                
                # 直接发送消息
                await self.send_text(result_msg)
                
                # 返回拦截等级 2 (不触发后续思考/回复)
                return True, result_msg, 2
                
            msg = "❌ 无法获取插件实例，请稍后重试"
            await self.send_text(msg)
            return False, msg, 2

        except Exception as e:
            logger.error(f"{self.log_prefix} 获取 Web 配置失败: {e}")
            msg = f"❌ 执行失败: {str(e)}"
            await self.send_text(msg)
            return False, msg, 2
