"""
总结导入Action组件
"""
from typing import Tuple, Optional, Dict, Any
from src.common.logger import get_logger
from src.plugin_system.base.base_action import BaseAction
from src.plugin_system.base.component_types import ActionActivationType
from ...core.utils.summary_importer import SummaryImporter

logger = get_logger("A_Memorix.SummaryImportAction")

class SummaryImportAction(BaseAction):
    """总结导入Action
    
    允许机器人调用来总结当前对话并将其存入知识库。
    """

    # Action基本信息
    action_name = "summary_import"
    action_description = "总结当前对话的历史记录并将其作为知识导入知识库"
    
    # 激活配置
    activation_type = ActionActivationType.ALWAYS
    parallel_action = True

    # Action参数
    action_parameters = {
        "context_length": {
            "type": "integer",
            "description": "总结的历史消息数量（可选，不传则使用默认配置）",
            "default": 0,
        }
    }

    # Action依赖
    action_require = ["vector_store", "graph_store", "metadata_store", "embedding_manager"]

    def __init__(self, *args, **kwargs):
        """初始化总结导入Action"""
        super().__init__(*args, **kwargs)

        # 初始化导入器
        self.importer: Optional[SummaryImporter] = None
        self._initialize_importer()

    def _initialize_importer(self) -> None:
        """从插件配置或实例中获取存储并初始化导入器"""
        try:
            # 优先从配置获取，兜底从插件实例获取
            vector_store = self.plugin_config.get("vector_store")
            graph_store = self.plugin_config.get("graph_store")
            metadata_store = self.plugin_config.get("metadata_store")
            embedding_manager = self.plugin_config.get("embedding_manager")

            if not all([vector_store, graph_store, metadata_store, embedding_manager]):
                from ...plugin import A_MemorixPlugin
                instances = A_MemorixPlugin.get_storage_instances()
                if instances:
                    vector_store = vector_store or instances.get("vector_store")
                    graph_store = graph_store or instances.get("graph_store")
                    metadata_store = metadata_store or instances.get("metadata_store")
                    embedding_manager = embedding_manager or instances.get("embedding_manager")

            if not all([vector_store, graph_store, metadata_store, embedding_manager]):
                logger.warning(f"{self.log_prefix} 存储组件未完全初始化，概括导入功能暂不可用")
                return

            self.importer = SummaryImporter(
                vector_store=vector_store,
                graph_store=graph_store,
                metadata_store=metadata_store,
                embedding_manager=embedding_manager,
                plugin_config=self.plugin_config
            )
            logger.info(f"{self.log_prefix} 总结导入器初始化完成")

        except Exception as e:
            logger.error(f"{self.log_prefix} 总结导入器初始化失败: {e}")

    async def execute(self) -> Tuple[bool, str]:
        """执行总结导入动作"""
        if not self.importer:
            return False, "总结导入器未初始化"

        # 检查功能开关
        if not self.plugin_config.get("summarization", {}).get("enabled", True):
            return False, "总结导入功能已被禁用"

        # 获取参数
        context_length = self.action_data.get("context_length", 0)
        if context_length <= 0:
            context_length = None

        if not self.chat_stream:
            return False, "无法获取当前聊天上下文"

        stream_id = self.chat_stream.stream_id
        group_id = self.group_id
        user_id = self.user_id
        
        # 检查过滤配置
        from ...plugin import A_MemorixPlugin
        plugin_instance = A_MemorixPlugin.get_global_instance()
        if plugin_instance:
            if not plugin_instance.is_chat_enabled(stream_id, group_id, user_id):
                return False, "当前聊天流已被禁用记忆导入功能"
        
        logger.info(f"{self.log_prefix} 触发手动总结导入: stream={stream_id}")

        success, message = await self.importer.import_from_stream(
            stream_id=stream_id,
            context_length=context_length
        )
        
        return success, message
