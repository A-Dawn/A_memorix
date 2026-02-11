"""
A_Memorix 插件主入口

完全独立的轻量级知识库插件，提供低资源环境下的高效知识存储与检索。
"""

import sys
from pathlib import Path
from typing import List, Tuple, Type, Optional, Dict, Union, Any
from src.plugin_system import (
    BasePlugin,
    BaseAction,
    BaseCommand,
    BaseTool,
    ActionInfo,
    CommandInfo,
    ToolInfo,
    ActionActivationType,
    ConfigField,
    register_plugin,
)
from src.common.logger import get_logger
import asyncio
import uuid
import time
import json
import datetime
from .core.utils.io import atomic_write

# deleted imports

from .core import (
    VectorStore,
    GraphStore,
    MetadataStore,
    EmbeddingAPIAdapter,
    create_embedding_api_adapter,
)

logger = get_logger("A_Memorix")


# 插件实例全局引用（由组件兜底使用）
# 使用 sys.modules 存储以解决由于不同路径导入导致的多个模块副本问题
def _set_global_instance(instance):
    sys.modules["A_MEMORIX_GLOBAL_INSTANCE"] = instance

def _get_global_instance():
    return sys.modules.get("A_MEMORIX_GLOBAL_INSTANCE")

@register_plugin
class A_MemorixPlugin(BasePlugin):
    """
    A_Memorix 轻量级知识库插件

    核心特性：
    - 完全独立的数据存储（plugins/A_memorix/data/）
    - 内存优化：目标512MB以内支持10万级数据
    - 向量量化：int8量化节省75%空间
    - 稀疏矩阵图：CSR格式存储知识图谱
    - 双路检索：关系+段落并行检索
    - Personalized PageRank排序
    """

    # 插件基本信息（PluginBase要求的抽象属性）
    plugin_name = "A_Memorix"
    plugin_version = "0.3.2"
    plugin_description = "轻量级知识库插件 - 完全独立的记忆增强系统"
    plugin_author = "A_Dawn"
    enable_plugin = False  # 默认禁用，需要在config.toml中启用
    dependencies: list[str] = []
    python_dependencies: list[str] = ["numpy", "scipy", "nest-asyncio", "faiss-cpu", "fastapi", "uvicorn", "pydantic"]  # 插件所需Python依赖
    config_file_name: str = "config.toml"

    # 配置节描述
    config_section_descriptions = {
        "plugin": "插件基本信息",
        "storage": "存储配置",
        "embedding": "嵌入模型配置",
        "retrieval": "检索配置",
        "threshold": "阈值策略配置",
        "graph": "知识图谱配置",
        "web": "可视化服务器配置",
        "advanced": "高级配置",
        "summarization": "总结与导入配置",
        "schedule": "定时任务配置",
        "filter": "消息过滤配置",
        "memory": "记忆衰减与强化配置",
    }

    # 配置Schema定义
    config_schema: dict = {
        "plugin": {
            "config_version": ConfigField(
                type=str,
                default="3.0.0",
                description="配置文件版本"
            ),
            "enabled": ConfigField(
                type=bool,
                default=False,
                description="是否启用插件"
            ),
        },
        "storage": {
            "data_dir": ConfigField(
                type=str,
                default="./data",  # Changed to relative path default
                description="数据目录（默认为插件目录下的 data）"
            ),
        },
        "embedding": {
            "dimension": ConfigField(
                type=int,
                default=1024,
                description="向量维度 (对于支持动态维度的模型，将尝试请求此维度)"
            ),
            "quantization_type": ConfigField(
                type=str,
                default="int8",
                description="量化类型: float32, int8, pq"
            ),
            "batch_size": ConfigField(
                type=int,
                default=32,
                description="批量生成嵌入的批次大小"
            ),
            "max_concurrent": ConfigField(
                type=int,
                default=5,
                description="嵌入API最大并发请求数"
            ),
            "model_name": ConfigField(
                type=str,
                default="auto",
                description="指定嵌入模型名称 (对应 model_config.toml 中的 name)"
            ),
            "retry": ConfigField(
                type=dict,
                default={
                    "max_attempts": 5,
                    "max_wait_seconds": 30,
                    "min_wait_seconds": 2,
                },
                description="嵌入重试配置: max_attempts, max_wait_seconds, min_wait_seconds"
            ),
        },
        "retrieval": {
            "top_k_relations": ConfigField(
                type=int,
                default=10,
                description="关系检索返回数量"
            ),
            "top_k_paragraphs": ConfigField(
                type=int,
                default=20,
                description="段落检索返回数量"
            ),
            "alpha": ConfigField(
                type=float,
                default=0.5,
                description="双路检索融合权重 (0:仅关系, 1:仅段落)"
            ),
            "enable_ppr": ConfigField(
                type=bool,
                default=True,
                description="是否启用 Personalized PageRank 重排序"
            ),
            "ppr_alpha": ConfigField(
                type=float,
                default=0.85,
                description="PPR的alpha参数"
            ),
            "ppr_concurrency_limit": ConfigField(
                type=int,
                default=4,
                description="PPR计算的最大并发数"
            ),
            "enable_parallel": ConfigField(
                type=bool,
                default=True,
                description="是否启用并行检索"
            ),
            "relation_semantic_fallback": ConfigField(
                type=bool,
                default=True,
                description="是否启用关系查询的语义回退（支持自然语言查询）"
            ),
            "relation_fallback_min_score": ConfigField(
                type=float,
                default=0.3,
                description="关系语义回退的最小相似度阈值"
            ),
        },
        "threshold": {
            "min_threshold": ConfigField(
                type=float,
                default=0.3,
                description="搜索结果的最小阈值"
            ),
            "max_threshold": ConfigField(
                type=float,
                default=0.95,
                description="搜索结果的最大阈值"
            ),
            "percentile": ConfigField(
                type=float,
                default=75.0,
                description="动态阈值的分位数"
            ),
            "std_multiplier": ConfigField(
                type=float,
                default=1.5,
                description="标准差倍数（用于异常值过滤）"
            ),
            "min_results": ConfigField(
                type=int,
                default=3,
                description="即使未达标也强制返回的最小结果数"
            ),
            "enable_auto_adjust": ConfigField(
                type=bool,
                default=True,
                description="是否根据结果分布自动调整阈值"
            ),
        },
        "graph": {
            "sparse_matrix_format": ConfigField(
                type=str,
                default="csr",
                description="稀疏矩阵存储格式: csr, csc"
            ),
        },
        "web": {
            "enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用可视化编辑服务器"
            ),
            "port": ConfigField(
                type=int,
                default=8082,
                description="服务器端口"
            ),
            "host": ConfigField(
                type=str,
                default="0.0.0.0",
                description="服务器绑定地址"
            ),
        },
        "advanced": {
            "enable_auto_save": ConfigField(
                type=bool,
                default=True,
                description="启用自动保存（原子化统一持久化）"
            ),
            "auto_save_interval_minutes": ConfigField(
                type=int,
                default=5,
                description="自动保存间隔（分钟）"
            ),
            "debug": ConfigField(
                type=bool,
                default=False,
                description="启用详细调试日志"
            ),
            "extraction_model": ConfigField(
                type=str,
                default="auto",
                description="指定知识抽取模型名称 (对应 model_config.toml 中的 name)"
            ),
        },
        "summarization": {
            "enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用总结导入功能"
            ),
            "model_name": ConfigField(
                type=str,
                default="auto",
                description="总结使用的模型选择器（支持 auto/任务名/模型名；也可在配置文件中使用数组）"
            ),
            "context_length": ConfigField(
                type=int,
                default=50,
                description="总结消息的上下文条数"
            ),
            "include_personality": ConfigField(
                type=bool,
                default=True,
                description="总结提示词是否包含机器人人设"
            ),
            "default_knowledge_type": ConfigField(
                type=str,
                default="narrative",
                description="总结导入时的默认知识类型"
            ),
        },
        "schedule": {
            "enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用定时自动导入"
            ),
            "import_times": ConfigField(
                type=list,
                default=["04:00"],
                description="每日自动导入的时间点列表 (24小时制, 如 ['04:00', '16:00'])"
            ),
        },
        "filter": {
            "enabled": ConfigField(
                type=bool,
                default=True,
                description="是否启用聊天流过滤"
            ),
            "mode": ConfigField(
                type=str,
                default="whitelist",
                description="过滤模式：whitelist(白名单) 或 blacklist(黑名单)"
            ),
            "chats": ConfigField(
                type=list,
                default=[],
                description="聊天流 ID 列表。支持填写: 1. 群号 (group_id, 如: 123456); 2. 私聊用户ID (user_id, 如: 10001); 3. 聊天流唯一标识 (stream_id, MD5格式)。"
            ),
        },
        "memory": {
             "half_life_hours": ConfigField(type=float, default=24.0, description="记忆强度半衰期 (小时)"),
             "base_decay_interval_hours": ConfigField(type=float, default=1.0, description="衰减任务执行间隔 (小时)"),
             "prune_threshold": ConfigField(type=float, default=0.1, description="记忆遗忘(冷冻)阈值"),
             "freeze_duration_hours": ConfigField(type=float, default=0.01, description="记忆冷冻保留期 (小时), 过期物理删除"), # 默认可以设小点便于测试，或者保留24h
             "enable_auto_reinforce": ConfigField(type=bool, default=True, description="是否启用自动强化"),
             "reinforce_buffer_max_size": ConfigField(type=int, default=1000, description="强化缓冲区最大大小"),
             "reinforce_cooldown_hours": ConfigField(type=float, default=1.0, description="同一记忆强化的冷却时间"),
             "max_weight": ConfigField(type=float, default=10.0, description="记忆最大权重限制"),
             "revive_boost_weight": ConfigField(type=float, default=0.5, description="复活时的基础增强权重"),
             "auto_protect_ttl_hours": ConfigField(type=float, default=24.0, description="复活/强化后的自动保护时长"),
             "min_active_weight_protected": ConfigField(type=float, default=0.5, description="保护期内记忆的最低权重地板"),
             "enabled": ConfigField(type=bool, default=True, description="V5记忆系统总开关"),
             "orphan": {
                 "enable_soft_delete": ConfigField(type=bool, default=True, description="是否启用软删除"),
                 "entity_retention_days": ConfigField(type=float, default=7.0, description="实体保留期(天)"),
                 "paragraph_retention_days": ConfigField(type=float, default=7.0, description="段落保留期(天)"),
                 "sweep_grace_hours": ConfigField(type=float, default=24.0, description="软删除宽限期(小时)"),
             }
        }
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _set_global_instance(self)
        self._initialized = False

        # 核心存储组件
        self.vector_store: Optional[VectorStore] = None
        self.graph_store: Optional[GraphStore] = None
        self.metadata_store: Optional[MetadataStore] = None
        self.embedding_manager: Optional[EmbeddingAPIAdapter] = None

        # 插件配置字典（传递给组件）
        self._plugin_config: dict = {}

        # 独立 Web 服务器实例
        self.server = None
        
        # 运行时自动保存开关（可通过WebUI修改）
        self._runtime_auto_save: Optional[bool] = None

        # V5 记忆系统
        self.reinforce_buffer: Set[str] = set() # 存储待强化的关系哈希
        self._memory_lock = asyncio.Lock()
        self._scheduled_import_task: Optional[asyncio.Task] = None
        self._auto_save_task: Optional[asyncio.Task] = None
        self._memory_maintenance_task: Optional[asyncio.Task] = None

    @property
    def debug_enabled(self) -> bool:
        return self.get_config("advanced.debug", False)

    def log_debug(self, message: str):
        """输出调试日志（仅在debug模式下）"""
        if self.debug_enabled:
            logger.info(f"[DEBUG] {message}")

    def get_plugin_components(
        self,
    ) -> List[
        Tuple[
            ActionInfo | CommandInfo | ToolInfo,
            Type[BaseAction | BaseCommand | BaseTool],
        ]
    ]:
        """获取插件包含的组件列表

        Returns:
            组件信息和组件类的列表
        """
        # 延迟导入以避免循环依赖
        from .components import (
            KnowledgeSearchAction,
            ImportCommand,
            QueryCommand,
            DeleteCommand,
            VisualizeCommand,
            KnowledgeQueryTool,
            MemoryModifierTool,
            SummaryImportAction,
        )

        components = []

        # KnowledgeSearchAction
        components.append(
            (
                ActionInfo(
                    name="knowledge_search",
                    component_type="action",
                    description="在知识库中搜索相关内容，支持段落和关系的双路检索",
                    activation_type=ActionActivationType.ALWAYS,
                    activation_keywords=[],
                    keyword_case_sensitive=False,
                    parallel_action=True,
                    random_activation_probability=0.0,
                    action_parameters={
                        "query": {
                            "type": "string",
                            "description": "搜索查询文本",
                            "required": True,
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "返回结果数量",
                            "default": 10,
                        },
                        "use_threshold": {
                            "type": "boolean",
                            "description": "是否使用动态阈值过滤",
                            "default": True,
                        },
                        "enable_ppr": {
                            "type": "boolean",
                            "description": "是否启用PPR重排序",
                            "default": True,
                        },
                    },
                    action_require=[
                        "vector_store",
                        "graph_store",
                        "metadata_store",
                        "embedding_manager",
                    ],
                    associated_types=[],
                ),
                KnowledgeSearchAction,
            )
        )

        # SummaryImportAction
        components.append(
            (
                ActionInfo(
                    name="summary_import",
                    component_type="action",
                    description="总结当前对话的历史记录并将其作为知识导入知识库",
                    activation_type=ActionActivationType.ALWAYS,
                    parallel_action=True,
                    action_parameters={
                        "context_length": {
                            "type": "integer",
                            "description": "总结的历史消息数量（可选）",
                            "default": 0,
                        }
                    },
                    action_require=[
                        "vector_store",
                        "graph_store",
                        "metadata_store",
                        "embedding_manager",
                    ],
                ),
                SummaryImportAction,
            )
        )

        # ImportCommand
        components.append(
            (
                CommandInfo(
                    name="import",
                    component_type="command",
                    description="导入知识到知识库，支持文本、段落、实体和关系的导入",
                    command_pattern=r"^\/import(?:\s+(?P<mode>\w+))?(?:\s+(?P<content>.+))?$",
                ),
                ImportCommand,
            )
        )

        # QueryCommand
        components.append(
            (
                CommandInfo(
                    name="query",
                    component_type="command",
                    description="查询知识库，支持检索、实体、关系和统计信息",
                    command_pattern=r"^\/query(?:\s+(?P<mode>\w+))?(?:\s+(?P<content>.+))?$",
                ),
                QueryCommand,
            )
        )

        # DeleteCommand
        components.append(
            (
                CommandInfo(
                    name="delete",
                    component_type="command",
                    description="删除知识库内容，支持段落、实体和关系的删除",
                    command_pattern=r"^\/delete(?:\s+(?P<mode>\w+))?(?:\s+(?P<content>.+))?$",
                ),
                DeleteCommand,
            )
        )

        # VisualizeCommand
        components.append(
            (
                CommandInfo(
                    name="visualize",
                    component_type="command",
                    description="生成知识图谱的交互式HTML可视化文件",
                    command_pattern=r"^\/visualize(?:\s+(?P<output_path>.+))?$",
                ),
                VisualizeCommand,
            )
        )

        # KnowledgeQueryTool
        components.append(
            (
                ToolInfo(
                    name="knowledge_query",
                    component_type="tool",
                    tool_description="查询A_Memorix知识库，支持检索、实体查询、关系查询和统计信息",
                    enabled=True,
                    tool_parameters=[
                        (
                            "query_type",
                            "string",
                            "查询类型：search(检索)、entity(实体)、relation(关系)、stats(统计)",
                            True,
                            ["search", "entity", "relation", "stats"],
                        ),
                        (
                            "query",
                            "string",
                            "查询内容（检索文本/实体名称/关系规格），stats模式不需要",
                            False,
                            None,
                        ),
                        (
                            "top_k",
                            "integer",
                            "返回结果数量（仅search模式）",
                            False,
                            None,
                        ),
                        (
                            "use_threshold",
                            "boolean",
                            "是否使用动态阈值过滤（仅search模式）",
                            False,
                            None,
                        ),
                    ],
                ),
                KnowledgeQueryTool,
            )
        )

        # MemoryModifierTool
        components.append(
            (
                ToolInfo(
                    name="memory_modifier",
                    component_type="tool",
                    tool_description="修改记忆的权重（强化/弱化）或设置永久性",
                    enabled=True,
                    tool_parameters=[
                         (
                            "action",
                            "string",
                            "动作: reinforce(强化), weaken(弱化), remember_forever(永久记忆), forget(遗忘)",
                            True,
                            ["reinforce", "weaken", "remember_forever", "forget"],
                        ),
                        (
                            "query",
                            "string",
                            "目标记忆的查询内容",
                            True,
                            None,
                        ),
                        (
                            "target_type",
                            "string",
                            "目标类型: relation(关系), entity(实体), paragraph(段落)",
                            False,
                            ["relation", "entity", "paragraph"],
                        ),
                        (
                            "strength",
                            "number",
                            "调整强度 (0.1 - 5.0)，默认为1.0",
                            False,
                            None,
                        ),
                    ],
                ),
                MemoryModifierTool,
            )
        )

        # MemoryMaintenanceCommand
        from .components.commands.memory_command import MemoryMaintenanceCommand
        components.append(
            (
                CommandInfo(
                    name="memory",
                    component_type="command",
                    description="记忆维护指令 (Status, Protect, Reinforce, Restore)",
                    command_pattern=r"^\/memory(?:\s+(?P<action>\w+))?(?:\s+(?P<args>.+))?$",
                ),
                MemoryMaintenanceCommand,
            )
        )

        # DebugServerCommand (临时)
        from .components.commands.debug_server_command import DebugServerCommand
        components.append(
            (
                CommandInfo(
                    name="debug_server",
                    component_type="command",
                    description="调试启动 Web Server",
                    command_pattern=r"^/debug_server$",
                ),
                DebugServerCommand,
            )
        )

        return components


    def register_plugin(self) -> bool:
        """注册插件并同步初始化存储"""
        self._sync_initialize()
        return super().register_plugin()

    def _sync_initialize(self):
        """同步初始化存储组件"""
        if not self._initialized:
            try:
                logger.info("A_Memorix 插件正在开始同步初始化存储组件...")
                self._initialize_storage()
                self._initialized = True
                
                # --- V5 迁移：如果缺失则重建边映射 ---
                if self.graph_store and self.metadata_store:
                    # 检查映射是否为空但元数据存在（迁移场景）
                    if not self.graph_store._edge_hash_map:
                         # 确保元数据存储已连接/就绪（在 _initialize_storage 后应当如此）
                         if self.metadata_store.has_data():
                             logger.info("[V5 Migration] Detecting missing Edge Map. Attempting rebuild from Metadata...")
                             try:
                                 triples = self.metadata_store.get_all_triples()
                                 if triples:
                                     cnt = self.graph_store.rebuild_edge_hash_map(triples)
                                     logger.info(f"[V5 Migration] Rebuilt {cnt} edge mappings.")
                                     # Save immediately to persist the migration
                                     self.graph_store.save() 
                                 else:
                                     logger.info("[V5 Migration] No triples found in MetadataStore.")
                             except Exception as e:
                                 logger.error(f"[V5 Migration] Failed to rebuild edge map: {e}")
                
                logger.info("A_Memorix 插件同步初始化成功")

                # 更新插件配置
                self._update_plugin_config()

            except Exception as e:
                logger.error(f"A_Memorix 插件同步初始化失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.debug("A_Memorix 存储组件已初始化，跳过")

    async def on_enable(self):
        """插件启用时调用"""
        logger.info("A_Memorix 插件已启用")
        self._sync_initialize()

        # 启动独立 Web 服务器
        if self.get_config("web.enabled", True):
            try:
                from .server import MemorixServer
                host = self.get_config("web.host", "0.0.0.0")
                port = self.get_config("web.port", 8082)
                
                if not self.server:
                    logger.info(f"正在启动 A_Memorix 可视化服务器 ({host}:{port})...")
                    self.server = MemorixServer(self, host=host, port=port)
                    self.server.start()
            except Exception as e:
                logger.error(f"启动 A_Memorix 可视化服务器失败: {e}")

        self._start_background_tasks()

    async def on_disable(self):
        """插件禁用时调用"""
        logger.info("A_Memorix 插件正在禁用...")

        await self._cancel_background_tasks()

        # 关闭独立 Web 服务器
        if self.server:
            try:
                self.server.stop()
                self.server = None
                logger.info("A_Memorix 可视化服务器已关闭")
            except Exception as e:
                logger.error(f"关闭 A_Memorix 可视化服务器失败: {e}")

        # 关闭存储组件
        if self.metadata_store:
            try:
                self.metadata_store.close()
                logger.info("元数据存储已关闭")
            except Exception as e:
                logger.error(f"关闭元数据存储时出错: {e}")

    def _start_background_tasks(self):
        """启动后台任务（幂等，避免重复创建）。"""
        if (
            self.get_config("summarization.enabled", True)
            and self.get_config("schedule.enabled", True)
            and (self._scheduled_import_task is None or self._scheduled_import_task.done())
        ):
            self._scheduled_import_task = asyncio.create_task(self._scheduled_import_loop())

        if (
            self.get_config("advanced.enable_auto_save", True)
            and (self._auto_save_task is None or self._auto_save_task.done())
        ):
            self._auto_save_task = asyncio.create_task(self._auto_save_loop())

        if self._memory_maintenance_task is None or self._memory_maintenance_task.done():
            self._memory_maintenance_task = asyncio.create_task(self._memory_maintenance_loop())

    async def _cancel_background_tasks(self):
        """停止后台任务并等待收敛。"""
        tasks = [
            ("scheduled_import", self._scheduled_import_task),
            ("auto_save", self._auto_save_task),
            ("memory_maintenance", self._memory_maintenance_task),
        ]
        for _, task in tasks:
            if task and not task.done():
                task.cancel()

        for name, task in tasks:
            if not task:
                continue
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"后台任务 {name} 退出异常: {e}")

        self._scheduled_import_task = None
        self._auto_save_task = None
        self._memory_maintenance_task = None

    async def on_unload(self):
        """插件卸载时调用"""
        logger.info("A_Memorix 插件已卸载")

    def _update_plugin_config(self):
        """更新插件配置字典供组件使用"""
        storage_instances = {
            "vector_store": self.vector_store,
            "graph_store": self.graph_store,
            "metadata_store": self.metadata_store,
            "embedding_manager": self.embedding_manager,
        }
        
        # 同时更新私有配置和主配置，确保命令可以通过其获取实例
        self._plugin_config.update(storage_instances)
        # 即使 self.config 是 DotDict，update 也应该正常工作
        self.config.update(storage_instances)

        logger.info(f"A_Memorix 配置已注入存储实例: {list(storage_instances.keys())}")

    @staticmethod
    def get_global_instance() -> Optional['A_MemorixPlugin']:
        """获取全局插件实例（供组件使用）"""
        return _get_global_instance()

    @classmethod
    def get_storage_instances(cls) -> Dict[str, Any]:
        """获取存储实例（供组件兜底使用）"""
        logger.info("get_storage_instances() 被调用")
        
        instance = _get_global_instance()
        logger.info(f"  _get_global_instance() 返回: {instance is not None}")
        
        if instance:
            result = {
                "vector_store": instance.vector_store,
                "graph_store": instance.graph_store,
                "metadata_store": instance.metadata_store,
                "embedding_manager": instance.embedding_manager,
            }
            logger.info(f"  从全局实例获取: vector_store={result['vector_store'] is not None}, "
                       f"graph_store={result['graph_store'] is not None}, "
                       f"metadata_store={result['metadata_store'] is not None}, "
                       f"embedding_manager={result['embedding_manager'] is not None}")
            return result
        
        # 如果单例不存在，尝试从 PluginManager 获取
        logger.warning("  全局实例不存在，尝试从 PluginManager 获取...")
        try:
            from src.plugin_system.core.plugin_manager import plugin_manager
            plugin = plugin_manager.get_plugin_instance("A_Memorix")
            logger.info(f"  plugin_manager.get_plugin_instance('A_Memorix') 返回: {plugin is not None}")
            
            if plugin and hasattr(plugin, "vector_store"):
                result = {
                    "vector_store": getattr(plugin, "vector_store"),
                    "graph_store": getattr(plugin, "graph_store"),
                    "metadata_store": getattr(plugin, "metadata_store"),
                    "embedding_manager": getattr(plugin, "embedding_manager"),
                }
                logger.info(f"  从 PluginManager 获取: vector_store={result['vector_store'] is not None}, "
                           f"graph_store={result['graph_store'] is not None}, "
                           f"metadata_store={result['metadata_store'] is not None}, "
                           f"embedding_manager={result['embedding_manager'] is not None}")
                return result
        except Exception as e:
            logger.error(f"通过 PluginManager 获取存储实例失败: {e}")
            import traceback
            traceback.print_exc()
            
        logger.error("  所有获取方式都失败，返回空字典")
        return {}

    async def _initialize_storage_async(self):
        """异步初始化存储组件（用于嵌入维度检测）"""
        # 从config.toml获取配置
        data_dir_str = self.get_config("storage.data_dir", "./data")
        
        # 处理相对路径：如果是相对路径，则相对于插件目录
        if data_dir_str.startswith("."):
            # 获取当前文件(plugin.py)所在目录
            plugin_dir = Path(__file__).resolve().parent
            data_dir = (plugin_dir / data_dir_str).resolve()
        else:
            data_dir = Path(data_dir_str)
            
        logger.info(f"A_Memorix 数据存储路径: {data_dir}")

        # 创建数据目录
        data_dir.mkdir(parents=True, exist_ok=True)

        # 初始化嵌入 API 适配器
        self.embedding_manager = create_embedding_api_adapter(
            batch_size=self.get_config("embedding.batch_size", 32),
            max_concurrent=self.get_config("embedding.max_concurrent", 5),
            default_dimension=self.get_config("embedding.dimension", 1024),
            model_name=self.get_config("embedding.model_name", "auto"),
            retry_config=self.get_config("embedding.retry", {}),
        )
        logger.info("嵌入 API 适配器初始化完成")

        # 异步检测嵌入维度
        try:
            detected_dimension = await self.embedding_manager._detect_dimension()
            logger.info(f"嵌入维度检测成功: {detected_dimension}")
        except Exception as e:
            logger.warning(f"嵌入维度检测失败: {e}，使用默认值")
            detected_dimension = self.embedding_manager.default_dimension

        # 获取量化类型
        quantization_str = self.get_config("embedding.quantization_type", "int8")
        from .core.storage import QuantizationType
        quantization_map = {
            "float32": QuantizationType.FLOAT32,
            "int8": QuantizationType.INT8,
            "pq": QuantizationType.PQ,
        }
        quantization_type = quantization_map.get(quantization_str, QuantizationType.INT8)

        # 初始化向量存储（使用检测到的维度）
        self.vector_store = VectorStore(
            dimension=detected_dimension,
            quantization_type=quantization_type,
            data_dir=data_dir / "vectors",
        )
        self.vector_store.min_train_threshold = self.get_config("embedding.min_train_threshold", 40)
        logger.info(f"向量存储初始化完成（维度: {detected_dimension}, 训练阈值: {self.vector_store.min_train_threshold}）")

        # 获取稀疏矩阵格式
        matrix_format_str = self.get_config("graph.sparse_matrix_format", "csr")
        from .core.storage import SparseMatrixFormat
        matrix_format_map = {
            "csr": SparseMatrixFormat.CSR,
            "csc": SparseMatrixFormat.CSC,
        }
        matrix_format = matrix_format_map.get(matrix_format_str, SparseMatrixFormat.CSR)

        # 初始化图存储
        self.graph_store = GraphStore(
            matrix_format=matrix_format,
            data_dir=data_dir / "graph",
        )
        logger.info("图存储初始化完成")

        # 初始化元数据存储
        self.metadata_store = MetadataStore(data_dir=data_dir / "metadata")
        self.metadata_store.connect()
        logger.info("元数据存储初始化完成")

        # 加载现有数据（如果存在）
        if self.vector_store.has_data():
            try:
                self.vector_store.load()
                logger.info(f"向量数据已加载，共 {self.vector_store.num_vectors} 个向量")
            except Exception as e:
                logger.warning(f"加载向量数据失败: {e}")

        if self.graph_store.has_data():
            try:
                self.graph_store.load()
                logger.info(f"图数据已加载，共 {self.graph_store.num_nodes} 个节点")
            except Exception as e:
                logger.warning(f"加载图数据失败: {e}")
        
        logger.info(f"知识库数据目录: {data_dir}")

    def _initialize_storage(self):
        """同步初始化存储组件（包装异步方法）"""
        import asyncio
        
        # 获取或创建事件循环
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果循环正在运行，创建新任务
                logger.warning("事件循环正在运行，使用 asyncio.create_task")
                # 这种情况下我们不能直接 await，需要特殊处理
                # 暂时使用同步方式，后续可以优化
                import nest_asyncio
                nest_asyncio.apply()
                loop.run_until_complete(self._initialize_storage_async())
            else:
                # 循环未运行，直接运行
                loop.run_until_complete(self._initialize_storage_async())
        except RuntimeError:
            # 没有事件循环，创建新的
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._initialize_storage_async())
            finally:
                loop.close()

    async def _scheduled_import_loop(self):
        """定时总结导入循环"""
        import asyncio
        import datetime
        
        logger.info("A_Memorix 定时总结导入任务已启动")
        
        # 记录上次检查的时间，用于跨越时间点检测
        last_check_now = datetime.datetime.now()
        
        while True:
            try:
                # 每分钟检查一次
                await asyncio.sleep(60)
                
                # 检查总开关和定时开关
                if not self.get_config("summarization.enabled", True) or not self.get_config("schedule.enabled", True):
                    continue
                
                now = datetime.datetime.now()
                import_times = self.get_config("schedule.import_times", ["04:00"])
                
                for t_str in import_times:
                    try:
                        # 解析配置的时间点 (HH:MM)
                        h, m = map(int, t_str.split(":"))
                        # 构造今天的该时间点
                        target_time = now.replace(hour=h, minute=m, second=0, microsecond=0)
                        
                        # 如果当前时间刚跨过目标时间点
                        if last_check_now < target_time <= now:
                            logger.info(f"触发 A_Memorix 定时导入任务: {t_str}")
                            await self._perform_bulk_summary_import()
                    except (ValueError, Exception) as e:
                        logger.error(f"解析定时配置 '{t_str}' 出错: {e}")
                
                last_check_now = now
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"定时导入循环发生未知错误: {e}")
                await asyncio.sleep(60)

    def is_chat_enabled(self, stream_id: str, group_id: str = None, user_id: str = None) -> bool:
        """检查聊天流是否启用记忆功能
        
        基于 filter 配置进行判断。支持以下格式:
        1. 纯 ID (如 "123456"): 匹配 stream_id, group_id 或 user_id (兼容模式)
        2. 带前缀 ID:
           - "group:123456": 仅匹配群号
           - "user:10001" 或 "private:10001": 仅匹配用户 ID
           - "stream:abcd...": 仅匹配聊天流 MD5
        """
        filter_config = self.get_config("filter", {})
        enabled = filter_config.get("enabled", True)
        
        if not enabled:
            return True
            
        mode = filter_config.get("mode", "whitelist")
        chats = filter_config.get("chats", [])
        
        if not chats:
            # 如果配置为空，白名单模式下兜底放行
            return mode == "whitelist"
            
        # 统一转为字符串并清理空格
        stream_id = str(stream_id) if stream_id else ""
        group_id = str(group_id) if group_id else ""
        user_id = str(user_id) if user_id else ""
        
        is_matched = False
        for pattern in chats:
            pattern = str(pattern).strip()
            if not pattern:
                continue
                
            if ":" in pattern:
                prefix, value = pattern.split(":", 1)
                prefix = prefix.lower()
                if prefix == "group" and value == group_id:
                    is_matched = True
                elif prefix in ["user", "private"] and value == user_id:
                    is_matched = True
                elif prefix == "stream" and value == stream_id:
                    is_matched = True
            else:
                # 兼容模式：匹配任意字段
                if pattern in [stream_id, group_id, user_id]:
                    is_matched = True
                    
            if is_matched:
                break
            
        if mode == "whitelist":
            return is_matched
        else:
            # 黑名单模式：匹配到的被禁用
            return not is_matched

    async def _perform_bulk_summary_import(self):
        """为所有活跃聊天执行总结导入"""
        import asyncio
        from .core.utils.summary_importer import SummaryImporter
        from src.common.database.database_model import ChatStreams
        
        # 实例化导入器
        importer = SummaryImporter(
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            metadata_store=self.metadata_store,
            embedding_manager=self.embedding_manager,
            plugin_config=self.config
        )
        
        # 获取所有已知的聊天流 ID, Group ID 和 User ID
        def _get_all_streams():
            try:
                # 获取 stream_id, group_id, user_id
                query = ChatStreams.select(ChatStreams.stream_id, ChatStreams.group_id, ChatStreams.user_id)
                return [{
                    "stream_id": s.stream_id, 
                    "group_id": s.group_id,
                    "user_id": s.user_id
                } for s in query]
            except Exception as e:
                logger.error(f"获取聊天流列表失败: {e}")
                return []
            
        streams = await asyncio.to_thread(_get_all_streams)
        
        if not streams:
            logger.info("未发现可总结的聊天流")
            return
            
        logger.info(f"开始为 {len(streams)} 个聊天流执行批量总结检查...")
        
        success_count = 0
        skipped_count = 0
        
        for s in streams:
            s_id = s["stream_id"]
            g_id = s.get("group_id")
            u_id = s.get("user_id")
            
            # 过滤检查
            if not self.is_chat_enabled(stream_id=s_id, group_id=g_id, user_id=u_id):
                skipped_count += 1
                continue
                
            try:
                # 执行总结导入 (SummaryImporter 内部会处理无新消息的情况)
                success, msg = await importer.import_from_stream(s_id)
                if success:
                    success_count += 1
                    logger.info(f"聊天流 {s_id} 自动总结成功")
            except Exception as e:
                logger.error(f"处理聊天流 {s_id} 自动总结时出错: {e}")
                
        logger.info(f"批量总结任务完成，成功: {success_count}，跳过: {skipped_count}")



        logger.info(f"批量总结任务完成，成功: {success_count}，跳过: {skipped_count}")

    async def save_all(self):
        """统一保存所有数据 (Unified Persistence)"""
        if not self.vector_store or not self.graph_store:
            return

        commit_id = str(uuid.uuid4())
        logger.info(f"开始统一保存 (Commit ID: {commit_id})...")
        
        try:
            # 并行保存各组件
            # VectorStore 和 GraphStore 的 save 方法现在已经是线程安全的(或使用原子写)
            # 但为了减少IO阻塞，最好在线程池运行
            await asyncio.gather(
                asyncio.to_thread(self.vector_store.save),
                asyncio.to_thread(self.graph_store.save)
                # MetadataStore 是 SQLite，通常实时写入，无需显式 save
            )
            
            # 更新 Manifest，标志着一次完整的持久化状态
            await self._update_manifest(commit_id)
            logger.info(f"统一保存完成 (Commit ID: {commit_id})")
            
        except Exception as e:
            logger.error(f"统一保存失败: {e}")

    async def _update_manifest(self, commit_id: str):
        """更新持久化清单"""
        manifest = {
            "last_commit_id": commit_id,
            "timestamp": time.time(),
            "iso_timestamp": datetime.datetime.now().isoformat(),
            "version": self.plugin_version
        }
        
        data_dir = Path(self.get_config("storage.data_dir", "./plugins/A_memorix/data"))
        manifest_path = data_dir / "persistence_manifest.json"
        
        try:
            # 使用原子写入更新 Manifest
            with atomic_write(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            logger.error(f"更新 Manifest 失败: {e}")

    async def _auto_save_loop(self):
        """自动保存循环"""
        logger.info("自动保存任务已启动")
        try:
            while True:
                # 获取配置的间隔时间 (分钟)
                interval = self.get_config("advanced.auto_save_interval_minutes", 5)
                if interval <= 0:
                    interval = 5
                
                await asyncio.sleep(interval * 60)
                
                if self.get_config("advanced.enable_auto_save", True):
                    await self.save_all()
                    
        except asyncio.CancelledError:
            logger.info("自动保存任务已取消")
        except Exception as e:
            logger.error(f"自动保存循环发生错误: {e}")

    # =========================================================================
    # V5 Memory System Logic
    # =========================================================================

    async def reinforce_access(self, relation_hashes: List[str]):
        """
        触发记忆强化 (Thread-safe push to buffer)
        """
        if not self.get_config("memory.enable_auto_reinforce", True):
            return
            
        async with self._memory_lock:
            self.reinforce_buffer.update(relation_hashes)
            
            # 如果缓冲区过大，可以考虑触发立即处理（可选，目前依赖定时循环即可） (TODO)

    async def _memory_maintenance_loop(self):
        """
        记忆维护循环 (Decay, Reinforce, Freeze, Prune)
        """
        logger.info("A_Memorix 记忆维护循环已启动 (V5)")
        
        while True:
            try:
                # 获取间隔 (默认1小时)
                interval_hours = self.get_config("memory.base_decay_interval_hours", 1.0)
                interval_seconds = max(60, int(interval_hours * 3600))
                
                await asyncio.sleep(interval_seconds)
                
                if not self.metadata_store or not self.graph_store:
                    continue
                    
                # Master Switch Check
                if not self.get_config("memory.enabled", True):
                    continue
                    
                async with self._memory_lock:
                    # 1. Process Reinforce Buffer
                    current_buffer = list(self.reinforce_buffer)
                    self.reinforce_buffer.clear()
                    
                    if current_buffer:
                        await self._process_reinforce_batch(current_buffer)
                        
                    # 2. 全局衰减 (Global Decay)
                    half_life = self.get_config("memory.half_life_hours", 24.0)
                    if half_life > 0:
                        # factor = (1/2) ^ (dt / half_life)
                        factor = 0.5 ** (interval_hours / half_life)
                        # 保护地板值由 prune 逻辑处理，decay 只负责乘法
                        self.graph_store.decay(factor)
                        logger.debug(f"执行记忆衰减: factor={factor:.4f}")
                        
                    # 3. 冷冻与修剪 (Freeze & Prune) (检查候选记忆)
                    await self._process_freeze_and_prune()
                    
                    # 4. 孤儿节点回收 (Orphan GC) (标记与清除)
                    await self._orphan_gc_phase()

            except asyncio.CancelledError:
                logger.info("记忆维护循环已取消")
                break
            except Exception as e:
                logger.error(f"记忆维护循环发生错误: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(60)

    async def _process_reinforce_batch(self, hashes: List[str]):
        """处理强化批次"""
        try:
            # 获取当前状态
            status_map = self.metadata_store.get_relation_status_batch(hashes)
            
            now = datetime.datetime.now().timestamp()
            cooldown = self.get_config("memory.reinforce_cooldown_hours", 1.0) * 3600
            max_weight = self.get_config("memory.max_weight", 10.0)
            revive_boost = self.get_config("memory.revive_boost_weight", 0.5)
            auto_protect = self.get_config("memory.auto_protect_ttl_hours", 24.0) * 3600
            
            hashes_to_update = []
            hashes_to_revive = []
            updates_protect = []
            
            # 需要查出 subject, object, predicate 来更新 GraphStore (因为 update_edge_weight 需要 u, v)
            # 这里稍微有点低效，因为 status_map 没包含 s, o。
            # 为了准确性，我们需要查询。
            cursor = self.metadata_store._conn.cursor()
            placeholders = ",".join(["?"] * len(hashes))
            cursor.execute(f"SELECT hash, subject, object FROM relations WHERE hash IN ({placeholders})", hashes)
            relation_info = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
            
            for h in hashes:
                if h not in status_map: continue
                s = status_map[h]
                info = relation_info.get(h)
                if not info: continue
                
                src, tgt = info
                
                # 冷却检查 (Cooldown Check)
                last_re = s.get("last_reinforced") or 0
                if (now - last_re) < cooldown and not s["is_inactive"]:
                    continue # 如果仍在冷却中且处于活跃状态，则跳过
                    
                # 计算增量权重 (Calculate Delta)
                current_w = s["weight"]
                # Delta = amount * (1 - w/max)
                delta = 1.0 * (1.0 - (current_w / max_weight))
                if delta < 0: delta = 0
                
                # 逻辑:
                # 1. 更新图权重 (Update Graph Weight)
                self.graph_store.update_edge_weight(src, tgt, delta, max_weight=max_weight)
                
                # 2. 元数据更新 (Metadata Updates)
                # 如果是不活跃状态，复活需要进行显式处理？
                # 实际上 update_edge_weight 会添加缺失的边，但我们需要更新元数据标志。
                if s["is_inactive"]:
                    hashes_to_revive.append(h)
                else:
                    hashes_to_update.append(h)
                    
            # 批量更新元数据 (Batch update Metadata)
            if hashes_to_revive:
                self.metadata_store.mark_relations_active(hashes_to_revive, boost_weight=revive_boost)
                self.metadata_store.update_relations_protection(
                    hashes_to_revive, 
                    protected_until=now + auto_protect, 
                    last_reinforced=now
                )
                logger.info(f"复活记忆: {len(hashes_to_revive)} 条")
                
            if hashes_to_update:
                self.metadata_store.update_relations_protection(
                    hashes_to_update, 
                    protected_until=now + auto_protect, 
                    last_reinforced=now
                )
                
        except Exception as e:
            logger.error(f"处理强化批次失败: {e}")

    async def _process_freeze_and_prune(self):
        """处理冷冻与修剪"""
        try:
            prune_threshold = self.get_config("memory.prune_threshold", 0.1)
            freeze_duration = self.get_config("memory.freeze_duration_hours", 24.0) * 3600
            now = datetime.datetime.now().timestamp()
            
            # 1. 冷冻阶段 (FREEZE PASS) (不活跃逻辑)
            # 策略：如果一条边权重过低，且其下所有关系均无保护，则冻结该边。
            # "冻结" = 在元数据中标记为不活跃 + 从邻接矩阵中移除 (但保留在 Map 中)。
            # 只有当边被移除，该记忆才不会参与 PageRank，符合 "不活跃" 定义。
            
            # 从图中获取低权重边 (邻居矩阵)
            low_edges = self.graph_store.get_low_weight_edges(prune_threshold)
            
            hashes_to_freeze = [] # 元数据更新列表
            edges_to_deactivate = [] # 图邻域更新列表
            
            for src, tgt in low_edges:
                src_canon = self.graph_store._canonicalize(src)
                tgt_canon = self.graph_store._canonicalize(tgt)
                if src_canon in self.graph_store._node_to_idx and tgt_canon in self.graph_store._node_to_idx:
                    s_idx = self.graph_store._node_to_idx[src_canon]
                    t_idx = self.graph_store._node_to_idx[tgt_canon]
                    
                    associated_hashes = self.graph_store._edge_hash_map.get((s_idx, t_idx), set())
                    if not associated_hashes: continue
                    
                    # 检查保护状态 (Check Protection)
                    statuses = self.metadata_store.get_relation_status_batch(list(associated_hashes))
                    
                    is_edge_protected = False
                    current_edge_hashes = []
                    
                    for h, st in statuses.items():
                        # 保护规则: 已置顶 (Pinned) 或 TTL 有效
                        if st["is_pinned"] or (st["protected_until"] or 0) > now:
                            is_edge_protected = True
                            break
                        # 如果已是不活跃状态则跳过 (虽然在已停用的低权重边中不应出现，但为了安全进行检查)
                        if st["is_inactive"]:
                            pass 
                        current_edge_hashes.append(h)
                            
                    if not is_edge_protected and current_edge_hashes:
                        # Freeze the whole edge
                        hashes_to_freeze.extend(current_edge_hashes)
                        edges_to_deactivate.append((src, tgt))
                        
            if hashes_to_freeze:
                self.metadata_store.mark_relations_inactive(hashes_to_freeze, inactive_since=now)
                # 仅从矩阵中移除 (保留在 Map 中)
                self.graph_store.deactivate_edges(edges_to_deactivate)
                logger.info(f"冷冻记忆: {len(hashes_to_freeze)} 条关系, 冻结 {len(edges_to_deactivate)} 条边")

            # 2. 修剪阶段 (PRUNE PASS) (删除逻辑)
            # 从元数据和 Map 中移除过期的不活跃关系。
            cutoff = now - freeze_duration
            expired_hashes = self.metadata_store.get_prune_candidates(cutoff)
            
            if expired_hashes:
                cursor = self.metadata_store._conn.cursor()
                placeholders = ",".join(["?"] * len(expired_hashes))
                cursor.execute(f"SELECT hash, subject, object FROM relations WHERE hash IN ({placeholders})", expired_hashes)
                
                ops_to_prune = [] # List[(src, tgt, hash)] for GraphStore
                actually_deleted_hashes = []
                
                for r in cursor.fetchall():
                    h, s, o = r[0], r[1], r[2]
                    # We need to remove this specific hash from map
                    ops_to_prune.append((s, o, h))
                    actually_deleted_hashes.append(h)
                
                # Update GraphStore (Map -> if empty -> Matrix)
                # Note: Matrix entry should be already gone via Freeze, but prune_relation_hashes handles that safety.
                if ops_to_prune:
                    self.graph_store.prune_relation_hashes(ops_to_prune)
                    
                # 从元数据中备份并删除 (Backup and Delete in Metadata)
                count = self.metadata_store.backup_and_delete_relations(actually_deleted_hashes)
                logger.info(f"物理修剪: {count} 条记忆 (已清理映射)")

        except Exception as e:
            logger.error(f"处理冷冻与修剪失败: {e}")

    async def _orphan_gc_phase(self):
        """
        孤儿节点回收阶段 (Orphan GC Phase)
        策略: Mark & Sweep (标记-清除)
        逻辑:
        1. Mark: 找出孤儿(Active Degree=0 & 未冻结)，同时满足 Retention 要求的，标记为 is_deleted=1.
        2. Sweep: 找出 is_deleted=1 且 deleted_at < now - grace 的，物理删除.
        """
        # Feature Toggle
        orphan_config = self.get_config("memory.orphan", {})
        if not orphan_config.get("enable_soft_delete", True):
            return

        try:
            logger.debug("开始孤儿节点回收阶段 (GC Phase)...")
            
            # Configs
            entity_retention = orphan_config.get("entity_retention_days", 7.0) * 86400
            para_retention = orphan_config.get("paragraph_retention_days", 7.0) * 86400
            grace_period = orphan_config.get("sweep_grace_hours", 24.0) * 3600
            
            # ==========================================================
            # 1. MARK PHASE (标记)
            # ==========================================================
            
            # 1.1 标记实体 (Mark Entities)
            # 从图中获取孤儿候选者 (活跃但孤立)
            # 注意: include_inactive=True (默认) 会排除掉那些虽然度为 0 但参与了冻结边的节点 -> 保护冻结节点不被删除
            isolated_candidates = self.graph_store.get_isolated_nodes(include_inactive=True)
            
            if isolated_candidates:
                # 通过元数据过滤 (保留时长与引用检查)
                final_entity_candidates = self.metadata_store.get_entity_gc_candidates(
                    isolated_candidates, 
                    retention_seconds=entity_retention
                )
                
                if final_entity_candidates:
                    cnt = self.metadata_store.mark_as_deleted(final_entity_candidates, "entity")
                    if cnt > 0:
                        logger.info(f"[GC-Mark] 标记删除实体: {cnt} 个")

            # 1.2 标记段落 (Mark Paragraphs)
            # 通过元数据过滤 (保留时长 & 无关系 & 无实体)
            para_candidates = self.metadata_store.get_paragraph_gc_candidates(retention_seconds=para_retention)
            if para_candidates:
                cnt = self.metadata_store.mark_as_deleted(para_candidates, "paragraph")
                if cnt > 0:
                    logger.info(f"[GC-Mark] 标记删除段落: {cnt} 个")
                    
            # ==========================================================
            # 2. SWEEP PHASE (物理清理)
            # ==========================================================
            
            # 2.1 清理段落 (Sweep Paragraphs)
            dead_paragraphs_tuples = self.metadata_store.sweep_deleted_items("paragraph", grace_period)
            if dead_paragraphs_tuples:
                dead_para_hashes = [t[0] for t in dead_paragraphs_tuples]
                count = self.metadata_store.physically_delete_paragraphs(dead_para_hashes)
                if count > 0:
                    logger.info(f"[GC-Sweep] 物理删除段落: {count} 个")

            # 2.2 清理实体 (Sweep Entities)
            dead_entities_tuples = self.metadata_store.sweep_deleted_items("entity", grace_period)
            if dead_entities_tuples:
                dead_entity_hashes = [t[0] for t in dead_entities_tuples]
                dead_entity_names = [t[1] for t in dead_entities_tuples]
                
                # 关键顺序：先从图存储中删除 (内存/矩阵)，然后再从元数据中删除。
                
                # 1. 图存储删除 (需要名称) (GraphStore Delete)
                self.graph_store.delete_nodes(dead_entity_names)
                
                # 2. 元数据存储删除 (需要哈希) (MetadataStore Delete)
                count = self.metadata_store.physically_delete_entities(dead_entity_hashes)
                if count > 0:
                   logger.info(f"[GC-Sweep] 物理删除实体: {count} 个")

        except Exception as e:
            logger.error(f"孤儿节点回收失败: {e}")
            import traceback
            traceback.print_exc()



# 插件导出
__plugin__ = A_MemorixPlugin
