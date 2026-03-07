"""
A_Memorix 插件主入口

完全独立的轻量级知识库插件，提供低资源环境下的高效知识存储与检索。
"""

import inspect
from typing import List, Tuple, Type, Optional, Dict, Union, Any, Set, Callable, Awaitable
from src.plugin_system import (
    BasePlugin,
    BaseAction,
    BaseCommand,
    BaseTool,
    BaseEventHandler,
    ActionInfo,
    CommandInfo,
    ToolInfo,
    EventHandlerInfo,
    ActionActivationType,
    EventType,
    MaiMessages,
    CustomEventHandlerResult,
    register_plugin,
)
from src.common.logger import get_logger
import asyncio

try:
    # 旧版的麦麦有这个工具函数，新版的没有了，做个兼容
    from src.common.toml_utils import to_builtin_data as _host_to_builtin_data
except Exception:
    _host_to_builtin_data = None

# deleted imports

from .core import (
    VectorStore,
    GraphStore,
    MetadataStore,
    EmbeddingAPIAdapter,
    SparseBM25Index,
    RelationWriteService,
)
from .core.config.plugin_config_schema import (
    config_schema as PLUGIN_CONFIG_SCHEMA,
    config_section_descriptions as PLUGIN_CONFIG_SECTION_DESCRIPTIONS,
)
from .core.runtime import RequestRouter
from .core.runtime.lifecycle_orchestrator import (
    cancel_background_tasks as cancel_background_tasks_runtime,
    ensure_initialized as ensure_initialized_runtime,
    initialize_storage_async as initialize_storage_async_runtime,
    start_background_tasks as start_background_tasks_runtime,
)
from .core.runtime.maintenance_tasks import (
    auto_save_loop as auto_save_loop_runtime,
    cleanup_orphan_relation_vectors as cleanup_orphan_relation_vectors_runtime,
    episode_generation_loop as episode_generation_loop_runtime,
    get_relation_vector_stats as get_relation_vector_stats_runtime,
    memory_maintenance_loop as memory_maintenance_loop_runtime,
    orphan_gc_phase as orphan_gc_phase_runtime,
    perform_bulk_summary_import as perform_bulk_summary_import_runtime,
    person_profile_refresh_loop as person_profile_refresh_loop_runtime,
    process_freeze_and_prune as process_freeze_and_prune_runtime,
    process_reinforce_batch as process_reinforce_batch_runtime,
    refresh_person_profiles_for_enabled_switches as refresh_profiles_runtime,
    relation_vector_backfill_loop as relation_vector_backfill_loop_runtime,
    reinforce_access as reinforce_access_runtime,
    save_all as save_all_runtime,
    scheduled_import_loop as scheduled_import_loop_runtime,
    update_manifest as update_manifest_runtime,
)
from .core.utils import PluginIdPolicy

logger = get_logger("A_Memorix")


def _to_builtin_data_compat(value: Any) -> Any:
    """
    将 tomlkit 节点转换为原生 Python 结构。

    优先复用宿主实现；若宿主未提供，则使用兼容实现避免插件在导入阶段失败。
    """
    if _host_to_builtin_data is not None:
        try:
            return _host_to_builtin_data(value)
        except Exception:
            # 兜底到本地兼容实现，避免宿主实现异常影响插件加载。
            pass

    try:
        unwrap = getattr(value, "unwrap", None)
        if callable(unwrap):
            value = unwrap()
    except Exception:
        pass

    if isinstance(value, dict):
        return {str(k): _to_builtin_data_compat(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_builtin_data_compat(v) for v in value]

    items = getattr(value, "items", None)
    if callable(items):
        try:
            return {str(k): _to_builtin_data_compat(v) for k, v in items()}
        except Exception:
            pass
    return value


def _resolve_loaded_plugin_instance(plugin_name: str) -> Optional["A_MemorixPlugin"]:
    """Resolve plugin instance via plugin_manager only (no global singleton fallback)."""
    try:
        from src.plugin_system.core.plugin_manager import plugin_manager

        direct = plugin_manager.get_plugin_instance(plugin_name)
        if direct is not None:
            return direct

        for loaded_name in plugin_manager.list_loaded_plugins():
            if PluginIdPolicy.is_target_plugin_id(loaded_name):
                instance = plugin_manager.get_plugin_instance(loaded_name)
                if instance is not None:
                    return instance
    except Exception:
        return None
    return None


def _patch_webui_a_memorix_routes_for_tomlkit_serialization() -> None:
    """
    运行时补丁：仅修正 A_Memorix 插件配置接口返回中的 tomlkit 节点序列化问题。

    限制：
    - 仅补丁 `/api/webui/plugins/config/{plugin_id}` 及其 schema 路由
    - 仅在 plugin_id == "A_Memorix" 时做返回值原生化
    - 不修改核心源码文件，仅在插件加载后动态包裹路由回调
    """
    target_paths = {
        "/api/webui/plugins/config/{plugin_id}",
        "/api/webui/plugins/config/{plugin_id}/schema",
        "/plugins/config/{plugin_id}",
        "/plugins/config/{plugin_id}/schema",
        "/config/{plugin_id}",
        "/config/{plugin_id}/schema",
    }
    patched_paths: list[str] = []
    route_collections: list[list[Any]] = []

    # 1) WebUI app 已创建时，直接补丁实际生效的 app.routes。
    try:
        from src.webui import webui_server as webui_server_module

        webui_server = getattr(webui_server_module, "_webui_server", None)
        app = getattr(webui_server, "app", None) if webui_server is not None else None
        if app is not None:
            route_collections.append(list(getattr(app, "routes", [])))
    except Exception as e:
        logger.debug(f"读取 WebUI app 路由失败，将尝试补丁 router 定义: {e}")

    # 2) app 未创建时，补丁 router.routes，确保后续 include_router 也使用已包裹 endpoint。
    try:
        from src.webui.routers import plugin as plugin_router_module

        plugin_router = getattr(plugin_router_module, "router", None)
        if plugin_router is not None:
            route_collections.append(list(getattr(plugin_router, "routes", [])))
    except Exception as e:
        logger.debug(f"读取插件路由定义失败: {e}")

    if not route_collections:
        logger.debug("未获取到可补丁的路由集合，跳过配置接口补丁")
        return

    for routes in route_collections:
        for route in routes:
            path = getattr(route, "path", "")
            methods = getattr(route, "methods", set()) or set()
            dependant = getattr(route, "dependant", None)
            if path not in target_paths or "GET" not in methods or dependant is None:
                continue

            original_call = getattr(dependant, "call", None)
            if original_call is None:
                continue
            if getattr(original_call, "_a_memorix_tomlkit_patch_applied", False):
                continue

            original_signature = inspect.signature(original_call)

            async def _patched_call(
                *args,
                __original_call=original_call,
                __original_signature=original_signature,
                **kwargs,
            ):
                result = await __original_call(*args, **kwargs)
                plugin_id = None
                try:
                    bound = __original_signature.bind_partial(*args, **kwargs)
                    plugin_id = bound.arguments.get("plugin_id")
                except Exception:
                    plugin_id = kwargs.get("plugin_id")

                if not PluginIdPolicy.is_target_plugin_id(plugin_id) or not isinstance(result, dict):
                    return result

                patched = dict(result)
                if "config" in patched:
                    patched["config"] = _to_builtin_data_compat(patched.get("config"))
                if "schema" in patched:
                    patched["schema"] = _to_builtin_data_compat(patched.get("schema"))
                return patched

            _patched_call.__name__ = getattr(original_call, "__name__", "_patched_call")
            _patched_call.__doc__ = getattr(original_call, "__doc__", None)
            setattr(_patched_call, "_a_memorix_tomlkit_patch_applied", True)

            dependant.call = _patched_call
            if hasattr(route, "endpoint"):
                route.endpoint = _patched_call
            patched_paths.append(path)

    if patched_paths:
        unique_paths = ", ".join(sorted(set(patched_paths)))
        logger.info(f"A_Memorix 已应用插件配置接口序列化补丁: {unique_paths}")
    else:
        logger.debug("A_Memorix 配置接口补丁未命中目标路由（可能等待 WebUI 路由注册）")

    # 补充说明：此补丁仅针对 A_Memorix 插件的配置接口进行结果原生化处理，确保返回给前端的数据结构不包含 tomlkit 的特殊类型，从而避免前端解析错误。
    # 其他插件不受影响，且仅在路径和方法完全匹配时才会进行处理，最大程度地减少了对Core的影响(迫真)(讨厌monkey patching但实在没有更优雅的方案了)


class A_MemorixStartHandler(BaseEventHandler):
    """在系统启动时调用插件 on_enable，拉起后台任务。"""

    event_type = EventType.ON_START
    handler_name = "a_memorix_start_handler"
    handler_description = "A_Memorix 启动生命周期处理器"

    async def execute(
        self, message: MaiMessages | None
    ) -> Tuple[bool, bool, Optional[str], Optional[CustomEventHandlerResult], Optional[MaiMessages]]:
        plugin = _resolve_loaded_plugin_instance(self.plugin_name)
        if plugin is None:
            logger.warning("A_Memorix ON_START: 未找到插件实例，跳过 on_enable")
            return True, True, "A_Memorix 实例缺失", None, None

        try:
            await plugin.on_enable()
            return True, True, "A_Memorix on_enable 已执行", None, None
        except Exception as e:
            logger.error(f"A_Memorix ON_START 执行失败: {e}", exc_info=True)
            return False, True, f"A_Memorix ON_START 失败: {e}", None, None


class A_MemorixStopHandler(BaseEventHandler):
    """在系统停止时调用插件 on_disable，收敛后台任务。"""

    event_type = EventType.ON_STOP
    handler_name = "a_memorix_stop_handler"
    handler_description = "A_Memorix 停止生命周期处理器"

    async def execute(
        self, message: MaiMessages | None
    ) -> Tuple[bool, bool, Optional[str], Optional[CustomEventHandlerResult], Optional[MaiMessages]]:
        plugin = _resolve_loaded_plugin_instance(self.plugin_name)
        if plugin is None:
            logger.warning("A_Memorix ON_STOP: 未找到插件实例，跳过 on_disable")
            return True, True, "A_Memorix 实例缺失", None, None

        try:
            await plugin.on_disable()
            return True, True, "A_Memorix on_disable 已执行", None, None
        except Exception as e:
            logger.error(f"A_Memorix ON_STOP 执行失败: {e}", exc_info=True)
            return False, True, f"A_Memorix ON_STOP 失败: {e}", None, None


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
    plugin_version = "1.0.1"
    plugin_description = "轻量级知识库插件 - 含人物画像能力的独立记忆增强系统"
    plugin_author = "A_Dawn"
    enable_plugin = False  # 默认禁用，需要在config.toml中启用
    dependencies: list[str] = []
    python_dependencies: list[str] = [
        "numpy",
        "scipy",
        "networkx",
        "pyarrow",
        "pandas",
        "nest-asyncio",
        "faiss-cpu",
        "fastapi",
        "uvicorn",
        "pydantic",
        "python-multipart",
        "jieba",
    ]  # 插件所需Python依赖
    config_file_name: str = "config.toml"

    # 配置节描述
    config_section_descriptions = PLUGIN_CONFIG_SECTION_DESCRIPTIONS

    # 配置Schema定义
    config_schema: dict = PLUGIN_CONFIG_SCHEMA

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _patch_webui_a_memorix_routes_for_tomlkit_serialization()
        self._initialized = False
        self._runtime_ready = False
        self._init_lock = asyncio.Lock()

        # 核心存储组件
        self.vector_store: Optional[VectorStore] = None
        self.graph_store: Optional[GraphStore] = None
        self.metadata_store: Optional[MetadataStore] = None
        self.embedding_manager: Optional[EmbeddingAPIAdapter] = None
        self.sparse_index: Optional[SparseBM25Index] = None
        self.relation_write_service: Optional[RelationWriteService] = None

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
        self._person_profile_refresh_task: Optional[asyncio.Task] = None
        self._memory_maintenance_task: Optional[asyncio.Task] = None
        self._relation_vector_backfill_task: Optional[asyncio.Task] = None
        self._episode_generation_task: Optional[asyncio.Task] = None

        # 检索请求去重（短 TTL + in-flight 合并）
        self._request_dedup_cache: Dict[str, Dict[str, Any]] = {}
        self._request_dedup_inflight: Dict[str, asyncio.Future] = {}
        self._request_dedup_lock = asyncio.Lock()
        self._request_router = RequestRouter(self)

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
            ActionInfo | CommandInfo | ToolInfo | EventHandlerInfo,
            Type[BaseAction | BaseCommand | BaseTool | BaseEventHandler],
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
            PersonProfileCommand,
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
                    description="主责 search/time 检索链路：在知识库中搜索相关内容，支持段落和关系的双路检索",
                    activation_type=ActionActivationType.ALWAYS,
                    activation_keywords=[],
                    keyword_case_sensitive=False,
                    parallel_action=True,
                    random_activation_probability=0.0,
                    action_parameters={
                        "query_type": {
                            "type": "string",
                            "description": "查询模式：semantic(仅语义)/time(仅时间)/hybrid(语义+时间)",
                            "required": False,
                        },
                        "query": {
                            "type": "string",
                            "description": "查询文本（semantic/hybrid必填）",
                            "required": False,
                        },
                        "time_from": {
                            "type": "string",
                            "description": "开始时间（仅支持 YYYY/MM/DD 或 YYYY/MM/DD HH:mm；日期自动按 00:00 展开，其他格式报错）",
                            "required": False,
                        },
                        "time_to": {
                            "type": "string",
                            "description": "结束时间（仅支持 YYYY/MM/DD 或 YYYY/MM/DD HH:mm；日期自动按 23:59 展开，其他格式报错）",
                            "required": False,
                        },
                        "person": {
                            "type": "string",
                            "description": "人物过滤（可选）",
                            "required": False,
                        },
                        "source": {
                            "type": "string",
                            "description": "来源过滤（可选）",
                            "required": False,
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

        # PersonProfileCommand
        components.append(
            (
                CommandInfo(
                    name="person_profile",
                    component_type="command",
                    description="控制人物画像自动注入开关（on/off/status）",
                    command_pattern=r"^\/person_profile(?:\s+(?P<action>\w+))?$",
                ),
                PersonProfileCommand,
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
                    tool_description="查询A_Memorix知识库（entity/relation/person/stats 主责；search/time 可按 routing 转发或兼容）",
                    enabled=True,
                    tool_parameters=[
                        (
                            "query_type",
                            "string",
                            "查询类型：search(检索)、time(时序检索)、episode(情景记忆)、aggregate(聚合)、entity(实体)、relation(关系)、person(人物画像)、stats(统计)",
                            True,
                            ["search", "time", "episode", "aggregate", "entity", "relation", "person", "stats"],
                        ),
                        (
                            "query",
                            "string",
                            "查询内容（检索文本/实体名称/关系规格），stats模式不需要",
                            False,
                            None,
                        ),
                        (
                            "person_id",
                            "string",
                            "人物ID（person模式可选；为空时自动解析）",
                            False,
                            None,
                        ),
                        (
                            "top_k",
                            "integer",
                            "返回结果数量（search/time/episode模式）",
                            False,
                            None,
                        ),
                        (
                            "use_threshold",
                            "boolean",
                            "是否使用动态阈值过滤（search/time模式）",
                            False,
                            None,
                        ),
                        (
                            "mix",
                            "boolean",
                            "aggregate 模式是否输出混合融合结果（Weighted RRF）",
                            False,
                            None,
                        ),
                        (
                            "mix_top_k",
                            "integer",
                            "aggregate 模式混合结果返回数量（默认回落 top_k）",
                            False,
                            None,
                        ),
                        (
                            "time_from",
                            "string",
                            "开始时间（time模式，仅支持 YYYY/MM/DD 或 YYYY/MM/DD HH:mm；日期按 00:00 展开）",
                            False,
                            None,
                        ),
                        (
                            "time_to",
                            "string",
                            "结束时间（time模式，仅支持 YYYY/MM/DD 或 YYYY/MM/DD HH:mm；日期按 23:59 展开）",
                            False,
                            None,
                        ),
                        (
                            "person",
                            "string",
                            "人物过滤（time模式可选）",
                            False,
                            None,
                        ),
                        (
                            "source",
                            "string",
                            "来源过滤（time/episode模式可选）",
                            False,
                            None,
                        ),
                        (
                            "include_paragraphs",
                            "boolean",
                            "episode 模式是否附带关联段落明细",
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

        # Lifecycle EventHandlers
        components.append((A_MemorixStartHandler.get_handler_info(), A_MemorixStartHandler))
        components.append((A_MemorixStopHandler.get_handler_info(), A_MemorixStopHandler))

        return components


    def register_plugin(self) -> bool:
        """注册插件（初始化在 on_enable 异步执行）"""
        return super().register_plugin()

    def _validate_runtime_config(self) -> None:
        mode = self._get_routing_mode_value("tool_search_mode", "forward")
        if mode not in {"forward", "disabled"}:
            raise ValueError(
                "routing.tool_search_mode 仅允许 forward|disabled。"
                " 请执行 scripts/release_vnext_migrate.py migrate 进行配置迁移。"
            )

        summary_model_cfg = self.get_config("summarization.model_name", ["auto"])
        if not isinstance(summary_model_cfg, list):
            raise ValueError(
                "summarization.model_name 在新版本必须为数组(List[str])。"
                " 请执行 scripts/release_vnext_migrate.py migrate。"
            )

        q_type = str(self.get_config("embedding.quantization_type", "int8") or "int8").strip().lower()
        if q_type != "int8":
            raise ValueError(
                "embedding.quantization_type 在新版本仅支持 int8(SQ8)。"
                " 请执行 scripts/release_vnext_migrate.py migrate。"
            )

    def _check_storage_ready(self) -> bool:
        return all(
            getattr(self, attr, None) is not None
            for attr in ("metadata_store", "vector_store", "graph_store", "embedding_manager")
        )

    def is_runtime_ready(self) -> bool:
        return bool(self._runtime_ready and self._check_storage_ready())

    async def _ensure_initialized(self) -> None:
        await ensure_initialized_runtime(self)

    async def on_enable(self):
        """插件启用时调用"""
        logger.info("A_Memorix 插件已启用")
        await self._ensure_initialized()
        if not self.is_runtime_ready():
            raise RuntimeError("A_Memorix 未完成就绪，拒绝启动 Web/后台任务")

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
        self._runtime_ready = False

        await self._cancel_background_tasks()

        # 关闭独立 Web 服务器
        if self.server:
            try:
                self.server.stop()
                self.server = None
                logger.info("A_Memorix 可视化服务器已关闭")
            except Exception as e:
                logger.error(f"关闭 A_Memorix 可视化服务器失败: {e}")

        # 卸载稀疏检索组件（释放连接与缓存）
        if self.sparse_index:
            try:
                sparse_cfg = self.get_config("retrieval.sparse", {}) or {}
                unload_on_disable = bool(sparse_cfg.get("unload_on_disable", True)) if isinstance(sparse_cfg, dict) else True
                if unload_on_disable:
                    self.sparse_index.unload()
                    logger.info("稀疏检索组件已卸载")
            except Exception as e:
                logger.error(f"卸载稀疏检索组件失败: {e}")

        # 关闭存储组件
        if self.metadata_store:
            try:
                self.metadata_store.close()
                logger.info("元数据存储已关闭")
            except Exception as e:
                logger.error(f"关闭元数据存储时出错: {e}")

    def _start_background_tasks(self):
        """启动后台任务（幂等，避免重复创建）。"""
        start_background_tasks_runtime(self)

    async def _cancel_background_tasks(self):
        """停止后台任务并等待收敛。"""
        await cancel_background_tasks_runtime(self)

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
            "sparse_index": self.sparse_index,
            "relation_write_service": self.relation_write_service,
            "plugin_instance": self,
        }
        
        # 同时更新私有配置和主配置，确保命令可以通过其获取实例
        self._plugin_config.update(storage_instances)
        # 即使 self.config 是 DotDict，update 也应该正常工作
        self.config.update(storage_instances)

        logger.info(f"A_Memorix 配置已注入存储实例: {list(storage_instances.keys())}")

    @staticmethod
    def get_global_instance() -> Optional['A_MemorixPlugin']:
        """获取插件实例（通过 PluginManager 解析，不使用全局单例）。"""
        return _resolve_loaded_plugin_instance("A_Memorix")

    @classmethod
    def get_storage_instances(cls) -> Dict[str, Any]:
        """获取存储实例（组件兜底路径）。"""
        instance = cls.get_global_instance()
        if instance is None:
            return {}
        return {
            "vector_store": instance.vector_store,
            "graph_store": instance.graph_store,
            "metadata_store": instance.metadata_store,
            "embedding_manager": instance.embedding_manager,
            "sparse_index": instance.sparse_index,
            "relation_write_service": instance.relation_write_service,
            "plugin_instance": instance,
        }

    async def _initialize_storage_async(self):
        """异步初始化存储组件（用于嵌入维度检测）"""
        await initialize_storage_async_runtime(self)

    def _initialize_storage(self):
        """已弃用：初始化必须走异步生命周期。"""
        raise RuntimeError("同步初始化路径已移除，请使用 await _ensure_initialized()")

    async def _scheduled_import_loop(self):
        """定时总结导入循环"""
        await scheduled_import_loop_runtime(self)

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
            # 空列表策略：白名单全拦截，黑名单全放行
            return mode == "blacklist"
            
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

    def _get_request_router(self) -> RequestRouter:
        router = getattr(self, "_request_router", None)
        if router is None:
            router = RequestRouter(self)
            self._request_router = router
        return router

    def _get_routing_mode_value(self, key: str, default: str) -> str:
        return self._get_request_router().get_routing_mode_value(key, default)

    def get_search_owner(self) -> str:
        return self._get_request_router().get_search_owner()

    def get_tool_search_mode(self) -> str:
        return self._get_request_router().get_tool_search_mode()

    def _is_request_dedup_enabled(self) -> bool:
        return self._get_request_router().is_request_dedup_enabled()

    def _get_request_dedup_ttl_seconds(self) -> float:
        return self._get_request_router().get_request_dedup_ttl_seconds()

    def _cleanup_request_dedup_cache_locked(self, now_ts: Optional[float] = None) -> None:
        self._get_request_router().cleanup_request_dedup_cache_locked(now_ts)

    async def execute_request_with_dedup(
        self,
        request_key: str,
        executor: Callable[[], Awaitable[Any]],
    ) -> Tuple[bool, Any]:
        """执行短时请求去重。"""
        return await self._get_request_router().execute_request_with_dedup(
            request_key=request_key,
            executor=executor,
        )

    def is_person_profile_injection_enabled(self, stream_id: str, user_id: str) -> bool:
        """检查人物画像自动注入是否开启（按 stream_id + user_id）。"""
        if not bool(self.get_config("person_profile.enabled", True)):
            return False

        opt_in_required = bool(self.get_config("person_profile.opt_in_required", True))
        default_enabled = bool(self.get_config("person_profile.default_injection_enabled", False))

        if not opt_in_required:
            return default_enabled

        if not stream_id or not user_id or self.metadata_store is None:
            return False

        try:
            return bool(self.metadata_store.get_person_profile_switch(stream_id, user_id, default=default_enabled))
        except Exception as e:
            logger.warning(f"读取人物画像开关失败: {e}")
            return False

    async def _person_profile_refresh_loop(self):
        """按需刷新人物画像快照（仅针对已开启范围内活跃人物）。"""
        await person_profile_refresh_loop_runtime(self)

    async def _refresh_person_profiles_for_enabled_switches(self):
        """刷新已开启范围内活跃人物画像。"""
        await refresh_profiles_runtime(self)

    async def _perform_bulk_summary_import(self):
        """为所有活跃聊天执行总结导入"""
        await perform_bulk_summary_import_runtime(self)

    def is_relation_vectorization_enabled(self) -> bool:
        cfg = self.get_config("retrieval.relation_vectorization", {}) or {}
        if not isinstance(cfg, dict):
            return False
        return bool(cfg.get("enabled", False))

    def should_write_relation_vector_on_import(self) -> bool:
        cfg = self.get_config("retrieval.relation_vectorization", {}) or {}
        if not isinstance(cfg, dict):
            return False
        return bool(cfg.get("enabled", False)) and bool(cfg.get("write_on_import", True))

    async def _relation_vector_backfill_loop(self):
        """后台分批回填关系向量。"""
        await relation_vector_backfill_loop_runtime(self)

    async def _episode_generation_loop(self):
        """后台异步生成 Episode。"""
        await episode_generation_loop_runtime(self)

    def _cleanup_orphan_relation_vectors(self, limit: int = 200) -> int:
        """清理关系孤儿向量（deleted_relations 存在且 relations 不存在）。"""
        return cleanup_orphan_relation_vectors_runtime(self, limit=limit)

    def get_relation_vector_stats(self) -> Dict[str, Any]:
        """返回关系向量状态与覆盖统计。"""
        return get_relation_vector_stats_runtime(self)

    async def save_all(self):
        """统一保存所有数据(Unified Persistence)"""
        await save_all_runtime(self)

    async def _update_manifest(self, commit_id: str):
        """更新持久化清单"""
        await update_manifest_runtime(self, commit_id=commit_id)

    async def _auto_save_loop(self):
        """自动保存循环"""
        await auto_save_loop_runtime(self)

    # =========================================================================
    # V5 Memory System Logic
    # =========================================================================

    async def reinforce_access(self, relation_hashes: List[str]):
        """触发记忆强化 (Thread-safe push to buffer)"""
        await reinforce_access_runtime(self, relation_hashes=relation_hashes)
            

    async def _memory_maintenance_loop(self):
        """记忆维护循环 (Decay, Reinforce, Freeze, Prune)"""
        await memory_maintenance_loop_runtime(self)

    async def _process_reinforce_batch(self, hashes: List[str]):
        """处理强化批次"""
        await process_reinforce_batch_runtime(self, hashes=hashes)

    async def _process_freeze_and_prune(self):
        """处理冷冻与修剪"""
        await process_freeze_and_prune_runtime(self)

    async def _orphan_gc_phase(self):
        """孤儿节点回收阶段 (Orphan GC Phase)."""
        await orphan_gc_phase_runtime(self)



# 插件导出
__plugin__ = A_MemorixPlugin
