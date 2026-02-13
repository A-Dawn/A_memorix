"""
知识检索Action组件

提供基于双路检索的知识搜索功能。
"""

import time
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path

from src.common.logger import get_logger
from src.plugin_system.base.base_action import BaseAction
from src.plugin_system.base.component_types import ActionActivationType
from src.chat.message_receive.chat_stream import ChatStream

# 导入核心模块
from ...core import (
    DualPathRetriever,
    RetrievalStrategy,
    DualPathRetrieverConfig,
    TemporalQueryOptions,
    SparseBM25Config,
    FusionConfig,
)
from ...core.utils.time_parser import parse_query_time_range

logger = get_logger("A_Memorix.KnowledgeSearchAction")


class KnowledgeSearchAction(BaseAction):
    """知识检索Action

    功能：
    - 双路检索（段落+关系）
    - 智能结果融合
    - PPR重排序
    - 动态阈值过滤
    """

    # Action基本信息
    action_name = "knowledge_search"
    action_description = "在知识库中搜索相关内容，支持段落和关系的双路检索"

    # 激活配置
    activation_type = ActionActivationType.ALWAYS
    parallel_action = True

    # Action参数
    action_parameters = {
        "query_type": {
            "type": "string",
            "description": "查询模式: semantic(语义)、time(时间)、hybrid(语义+时间)",
            "required": False,
            "enum": ["semantic", "time", "hybrid"],
            "default": "semantic",
        },
        "query": {
            "type": "string",
            "description": "搜索查询文本（semantic/hybrid必填，time可选）",
            "required": False,
        },
        "time_from": {
            "type": "string",
            "description": "开始时间，仅支持 YYYY/MM/DD 或 YYYY/MM/DD HH:mm（日期自动按 00:00 展开）",
            "required": False,
        },
        "time_to": {
            "type": "string",
            "description": "结束时间，仅支持 YYYY/MM/DD 或 YYYY/MM/DD HH:mm（日期自动按 23:59 展开）",
            "required": False,
        },
        "person": {
            "type": "string",
            "description": "按人物过滤（可选）",
            "required": False,
        },
        "source": {
            "type": "string",
            "description": "按来源过滤（可选）",
            "required": False,
        },
        "top_k": {
            "type": "integer",
            "description": "返回结果数量",
            "default": 10,
            "min": 1,
            "max": 50,
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
    }

    # Action依赖
    action_require = ["vector_store", "graph_store", "metadata_store", "embedding_manager"]

    def __init__(self, *args, **kwargs):
        """初始化知识检索Action"""
        super().__init__(*args, **kwargs)

        # 初始化检索器
        self.retriever: Optional[DualPathRetriever] = None
        self._initialize_retriever()
 
    @property
    def debug_enabled(self) -> bool:
        """检查是否启用了调试模式"""
        advanced = self.plugin_config.get("advanced", {})
        if isinstance(advanced, dict):
            return advanced.get("debug", False)
        return self.plugin_config.get("debug", False)
 
    def _initialize_retriever(self) -> None:
        """初始化检索器"""
        try:
            # 从插件配置获取存储实例 (优先从配置获取，兜底从插件实例获取)
            vector_store = self.plugin_config.get("vector_store")
            graph_store = self.plugin_config.get("graph_store")
            metadata_store = self.plugin_config.get("metadata_store")
            embedding_manager = self.plugin_config.get("embedding_manager")
            sparse_index = self.plugin_config.get("sparse_index")

            # 兜底逻辑：如果配置中没有存储实例，尝试直接从插件系统获取
            # 使用 is not None 检查，因为空对象可能布尔值为 False
            if not all([
                vector_store is not None,
                graph_store is not None,
                metadata_store is not None,
                embedding_manager is not None
            ]):
                from ...plugin import A_MemorixPlugin
                instances = A_MemorixPlugin.get_storage_instances()
                if instances:
                    vector_store = vector_store or instances.get("vector_store")
                    graph_store = graph_store or instances.get("graph_store")
                    metadata_store = metadata_store or instances.get("metadata_store")
                    embedding_manager = embedding_manager or instances.get("embedding_manager")
                    sparse_index = sparse_index or instances.get("sparse_index")


            # 最终检查 (使用 is not None 而非布尔值，因为空对象可能为 False)
            if not all([
                vector_store is not None,
                graph_store is not None,
                metadata_store is not None,
                embedding_manager is not None
            ]):
                logger.warning(f"{self.log_prefix} 存储组件未完全初始化，无法使用检索功能")
                return

            # 创建检索器配置
            sparse_cfg_raw = self.get_config("retrieval.sparse", {}) or {}
            if not isinstance(sparse_cfg_raw, dict):
                sparse_cfg_raw = {}
            fusion_cfg_raw = self.get_config("retrieval.fusion", {}) or {}
            if not isinstance(fusion_cfg_raw, dict):
                fusion_cfg_raw = {}
            try:
                sparse_cfg = SparseBM25Config(**sparse_cfg_raw)
            except Exception as e:
                logger.warning(f"{self.log_prefix} sparse 配置非法，回退默认: {e}")
                sparse_cfg = SparseBM25Config()
            try:
                fusion_cfg = FusionConfig(**fusion_cfg_raw)
            except Exception as e:
                logger.warning(f"{self.log_prefix} fusion 配置非法，回退默认: {e}")
                fusion_cfg = FusionConfig()

            config = DualPathRetrieverConfig(
                top_k_paragraphs=self.get_config("retrieval.top_k_paragraphs", 20),
                top_k_relations=self.get_config("retrieval.top_k_relations", 10),
                top_k_final=self.get_config("retrieval.top_k_final", 10),
                alpha=self.get_config("retrieval.alpha", 0.5),
                enable_ppr=self.get_config("retrieval.enable_ppr", True),
                ppr_alpha=self.get_config("retrieval.ppr_alpha", 0.85),
                ppr_concurrency_limit=self.get_config("retrieval.ppr_concurrency_limit", 4),
                enable_parallel=self.get_config("retrieval.enable_parallel", True),
                retrieval_strategy=RetrievalStrategy.DUAL_PATH,
                debug=self.debug_enabled,
                sparse=sparse_cfg,
                fusion=fusion_cfg,
            )

            # 创建检索器
            self.retriever = DualPathRetriever(
                vector_store=vector_store,
                graph_store=graph_store,
                metadata_store=metadata_store,
                embedding_manager=embedding_manager,
                sparse_index=sparse_index,
                config=config,
            )

            logger.info(f"{self.log_prefix} 知识检索器初始化完成")

        except Exception as e:
            logger.error(f"{self.log_prefix} 检索器初始化失败: {e}")
            self.retriever = None

    async def execute(self) -> Tuple[bool, str]:
        """执行知识检索

        Returns:
            Tuple[bool, str]: (是否成功, 结果文本)
        """
        # 检查检索器是否可用
        if not self.retriever:
            return False, "知识检索器未初始化"

        # 获取查询参数
        query = str(self.action_data.get("query", "") or "").strip()
        query_type = str(self.action_data.get("query_type", "") or "").strip().lower()
        time_from_raw = self.action_data.get("time_from")
        time_to_raw = self.action_data.get("time_to")
        person = self.action_data.get("person")
        source = self.action_data.get("source")
        top_k_raw = self.action_data.get("top_k")
        use_threshold = self.action_data.get("use_threshold", True)
        enable_ppr = self.action_data.get("enable_ppr", True)
        if not query_type:
            if time_from_raw or time_to_raw:
                query_type = "hybrid" if query else "time"
            else:
                query_type = "semantic"

        if query_type not in {"semantic", "time", "hybrid"}:
            return False, f"query_type 无效: {query_type}（仅支持 semantic/time/hybrid）"

        if query_type in {"semantic", "hybrid"} and not query:
            return False, "semantic/hybrid 模式必须提供 query"

        temporal_enabled = bool(self.get_config("retrieval.temporal.enabled", True))
        if query_type in {"time", "hybrid"} and not temporal_enabled:
            return False, "时序检索已禁用（retrieval.temporal.enabled=false）"

        temporal_default_top_k = int(
            self.get_config("retrieval.temporal.default_top_k", 10)
        )
        default_top_k = temporal_default_top_k if query_type in {"time", "hybrid"} else 10
        if top_k_raw is None:
            top_k = default_top_k
        else:
            try:
                top_k = int(top_k_raw)
            except (TypeError, ValueError):
                return False, "top_k 参数必须为整数"
        top_k = max(1, min(50, top_k))

        temporal: Optional[TemporalQueryOptions] = None
        if query_type in {"time", "hybrid"}:
            if not time_from_raw and not time_to_raw:
                return False, "time/hybrid 模式至少需要 time_from 或 time_to"
            try:
                ts_from, ts_to = parse_query_time_range(
                    str(time_from_raw) if time_from_raw is not None else None,
                    str(time_to_raw) if time_to_raw is not None else None,
                )
            except ValueError as e:
                return False, f"时间参数错误: {e}"
            temporal = TemporalQueryOptions(
                time_from=ts_from,
                time_to=ts_to,
                person=str(person).strip() if person else None,
                source=str(source).strip() if source else None,
                allow_created_fallback=self.get_config(
                    "retrieval.temporal.allow_created_fallback",
                    True,
                ),
                candidate_multiplier=int(
                    self.get_config("retrieval.temporal.candidate_multiplier", 8)
                ),
                max_scan=int(self.get_config("retrieval.temporal.max_scan", 1000)),
            )

        # 0. 检查是否在允许的聊天流中
        from ...plugin import A_MemorixPlugin
        plugin_instance = A_MemorixPlugin.get_global_instance()
        if plugin_instance:
             # 获取当前聊天流ID, 群组ID, 用户ID
            stream_id = self.chat_id
            group_id = self.group_id
            user_id = self.user_id
            
            if not plugin_instance.is_chat_enabled(stream_id, group_id, user_id):
                # 如果未启用，安静地返回（视为成功但无结果，避免打扰）
                logger.info(f"{self.log_prefix} 聊天流已被过滤配置禁用，跳过检索")
                return True, ""

        logger.info(
            f"{self.log_prefix} 开始知识检索: query_type={query_type}, query='{query}', top_k={top_k}"
        )

        try:
            # 记录开始时间
            start_time = time.time()

            # 执行检索
            results = await self._search_knowledge(
                query=query,
                top_k=top_k,
                use_threshold=use_threshold,
                enable_ppr=enable_ppr,
                temporal=temporal,
            )

            # 计算耗时
            elapsed_ms = (time.time() - start_time) * 1000

            # 格式化结果
            if not results:
                response = f"未找到相关内容（检索耗时: {elapsed_ms:.1f}ms）"
                logger.info(f"{self.log_prefix} {response}")
                return True, response

            # 构建响应
            query_display = query if query else "N/A"
            response = self._format_results(results, query_display, elapsed_ms)

            logger.info(
                f"{self.log_prefix} 检索完成: 返回{len(results)}条结果, 耗时{elapsed_ms:.1f}ms"
            )

            return True, response

        except Exception as e:
            error_msg = f"知识检索失败: {str(e)}"
            logger.error(f"{self.log_prefix} {error_msg}")
            return False, error_msg

    async def _search_knowledge(
        self,
        query: str,
        top_k: int,
        use_threshold: bool,
        enable_ppr: bool,
        temporal: Optional[TemporalQueryOptions] = None,
    ) -> List[Any]:
        """执行知识检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            use_threshold: 是否使用阈值过滤
            enable_ppr: 是否启用PPR

        Returns:
            检索结果列表
        """
        # 临时配置PPR
        original_ppr_setting = self.retriever.config.enable_ppr
        self.retriever.config.enable_ppr = enable_ppr

        try:
            # 执行检索 (异步)
            results = await self.retriever.retrieve(
                query,
                top_k=top_k,
                temporal=temporal,
            )

            # 应用阈值过滤
            if use_threshold and hasattr(self.retriever, "threshold_filter"):
                threshold_filter = self.retriever.threshold_filter
                if threshold_filter:
                    results = threshold_filter.filter(results)

            # V5: Hook for Memory Reinforcement
            from ...plugin import A_MemorixPlugin
            plugin_instance = A_MemorixPlugin.get_global_instance()
            if plugin_instance:
                 # Reinforce only relations that survived filtering
                 relation_hashes = [
                     r.hash_value for r in results if r.result_type == "relation"
                 ]
                 if relation_hashes:
                     # Fire and forget (it pushes to buffer)
                     await plugin_instance.reinforce_access(relation_hashes)

            return results

        finally:
            # 恢复原始配置
            self.retriever.config.enable_ppr = original_ppr_setting

    def _format_results(self, results: List[Any], query: str, elapsed_ms: float) -> str:
        """格式化检索结果

        Args:
            results: 检索结果列表
            query: 原始查询
            elapsed_ms: 检索耗时

        Returns:
            格式化的结果文本
        """
        lines = []
        lines.append(f"🔍 知识检索结果（查询: '{query}'，耗时: {elapsed_ms:.1f}ms）")
        lines.append("")

        # 按类型分组
        paragraphs = []
        relations = []

        for result in results:
            if result.result_type == "paragraph":
                paragraphs.append(result)
            elif result.result_type == "relation":
                relations.append(result)

        # 添加段落结果
        if paragraphs:
            lines.append("📄 匹配的段落：")
            for i, result in enumerate(paragraphs, 1):
                score_pct = result.score * 100
                summary = result.content[:100] + ("..." if len(result.content) > 100 else "")
                lines.append(f"  {i}. [{score_pct:.1f}%] {summary}")
                time_meta = result.metadata.get("time_meta", {})
                if time_meta:
                    basis = time_meta.get("match_basis", "none")
                    s_text = time_meta.get("effective_start_text") or "N/A"
                    e_text = time_meta.get("effective_end_text") or "N/A"
                    lines.append(f"     ⏱️ {s_text} ~ {e_text} ({basis})")
            lines.append("")

        # 添加关系结果
        if relations:
            lines.append("🔗 匹配的关系：")
            for i, result in enumerate(relations, 1):
                score_pct = result.score * 100
                subject = result.metadata.get("subject", "")
                predicate = result.metadata.get("predicate", "")
                obj = result.metadata.get("object", "")
                lines.append(f"  {i}. [{score_pct:.1f}%] {subject} {predicate} {obj}")
                time_meta = result.metadata.get("time_meta", {})
                if time_meta:
                    basis = time_meta.get("match_basis", "none")
                    s_text = time_meta.get("effective_start_text") or "N/A"
                    e_text = time_meta.get("effective_end_text") or "N/A"
                    lines.append(f"     ⏱️ {s_text} ~ {e_text} ({basis})")
            lines.append("")

        # 添加统计信息
        lines.append(f"📊 统计: 共{len(results)}条结果（段落: {len(paragraphs)}, 关系: {len(relations)}）")

        return "\n".join(lines)

    async def search_batch(
        self,
        queries: List[str],
        top_k: int = 10,
    ) -> Dict[str, List[Any]]:
        """批量检索知识

        Args:
            queries: 查询列表
            top_k: 每个查询返回的结果数

        Returns:
            查询到结果的映射 {query: results}
        """
        if not self.retriever:
            logger.error(f"{self.log_prefix} 检索器未初始化")
            return {}

        results_map = {}

        for query in queries:
            try:
                results = self.retriever.retrieve(query, top_k=top_k)
                results_map[query] = results
                logger.info(
                    f"{self.log_prefix} 批量检索: '{query}' -> {len(results)}条结果"
                )
            except Exception as e:
                logger.error(f"{self.log_prefix} 批量检索失败 '{query}': {e}")
                results_map[query] = []

        return results_map

    def get_statistics(self) -> Dict[str, Any]:
        """获取检索统计信息

        Returns:
            统计信息字典
        """
        if not self.retriever:
            return {
                "status": "not_initialized",
            }

        stats = {
            "status": "active",
            "config": {
                "top_k_paragraphs": self.retriever.config.top_k_paragraphs,
                "top_k_relations": self.retriever.config.top_k_relations,
                "alpha": self.retriever.config.alpha,
                "enable_ppr": self.retriever.config.enable_ppr,
                "enable_parallel": self.retriever.config.enable_parallel,
                "retrieval_strategy": self.retriever.config.retrieval_strategy.value,
            },
        }

        # 添加存储统计
        if hasattr(self.retriever, "vector_store"):
            stats["vector_store"] = {
                "num_vectors": self.retriever.vector_store.num_vectors,
                "dimension": self.retriever.vector_store.dimension,
            }

        if hasattr(self.retriever, "graph_store"):
            stats["graph_store"] = {
                "num_nodes": self.retriever.graph_store.num_nodes,
                "num_edges": self.retriever.graph_store.num_edges,
            }

        return stats

    def __repr__(self) -> str:
        return (
            f"KnowledgeSearchAction("
            f"retriever_initialized={self.retriever is not None})"
        )
