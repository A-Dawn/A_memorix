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
    plugin_version = "0.1.0"
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
        "graph": "知识图谱配置",
        "import": "导入配置",
        "advanced": "高级配置",
    }

    # 配置Schema定义
    config_schema: dict = {
        "plugin": {
            "config_version": ConfigField(
                type=str,
                default="1.0.0",
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
                default="./plugins/A_memorix/data",
                description="数据目录（完全独立于原LPMM系统）"
            ),
            "max_memory_mb": ConfigField(
                type=int,
                default=512,
                description="最大内存占用限制（MB）"
            ),
            "enable_memory_mapping": ConfigField(
                type=bool,
                default=True,
                description="启用内存映射文件存储"
            ),
        },
        "embedding": {
            "model_preset": ConfigField(
                type=str,
                default="light",
                description="模型预设档位: ultra_light, light, balanced, performance"
            ),
            "dimension": ConfigField(
                type=int,
                default=384,
                description="向量维度"
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
            "ppr_alpha": ConfigField(
                type=float,
                default=0.85,
                description="PPR的alpha参数"
            ),
            "ppr_iterations": ConfigField(
                type=int,
                default=50,
                description="PPR迭代次数"
            ),
            "enable_dynamic_threshold": ConfigField(
                type=bool,
                default=True,
                description="启用动态阈值过滤"
            ),
        },
        "graph": {
            "enable_synonym_connection": ConfigField(
                type=bool,
                default=True,
                description="启用同义词自动连接"
            ),
            "synonym_threshold": ConfigField(
                type=float,
                default=0.85,
                description="同义词相似度阈值"
            ),
            "sparse_matrix_format": ConfigField(
                type=str,
                default="csr",
                description="稀疏矩阵存储格式: csr, csc"
            ),
        },
        "import": {
            "chunk_size": ConfigField(
                type=int,
                default=1000,
                description="批量导入的块大小"
            ),
            "enable_openie": ConfigField(
                type=bool,
                default=True,
                description="启用OpenIE信息抽取"
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
            "enable_multiprocessing": ConfigField(
                type=bool,
                default=False,
                description="启用多进程处理（单核环境应关闭）"
            ),
            "num_workers": ConfigField(
                type=int,
                default=1,
                description="工作进程数量"
            ),
            "log_level": ConfigField(
                type=str,
                default="INFO",
                description="日志级别: DEBUG, INFO, WARNING, ERROR"
            ),
            "enable_auto_save": ConfigField(
                type=bool,
                default=True,
                description="启用自动保存"
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
        },
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

    async def on_disable(self):
        """插件禁用时调用"""
        logger.info("A_Memorix 插件正在禁用...")

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
        data_dir_str = self.get_config("storage.data_dir", "./plugins/A_memorix/data")
        data_dir = Path(data_dir_str)

        # 创建数据目录
        data_dir.mkdir(parents=True, exist_ok=True)

        # 初始化嵌入 API 适配器
        self.embedding_manager = create_embedding_api_adapter(
            batch_size=self.get_config("embedding.batch_size", 32),
            max_concurrent=self.get_config("embedding.max_concurrent", 5),
            default_dimension=self.get_config("embedding.dimension", 1024),
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
        logger.info(f"向量存储初始化完成（维度: {detected_dimension}）")

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



# 插件导出
__plugin__ = A_MemorixPlugin
