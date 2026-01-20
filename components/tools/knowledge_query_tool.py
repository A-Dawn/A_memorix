"""
知识查询Tool组件

提供LLM可调用的知识查询工具。
"""

import time
from typing import Any, List, Tuple, Optional, Dict
from pathlib import Path

from src.common.logger import get_logger
from src.plugin_system.base.base_tool import BaseTool
from src.plugin_system.base.component_types import ToolParamType
from src.chat.message_receive.chat_stream import ChatStream

# 导入核心模块
from ...core import (
    DualPathRetriever,
    RetrievalStrategy,
    DualPathRetrieverConfig,
    DynamicThresholdFilter,
    ThresholdMethod,
    ThresholdConfig,
)

logger = get_logger("A_Memorix.KnowledgeQueryTool")


class KnowledgeQueryTool(BaseTool):
    """知识查询Tool

    功能：
    - 双路检索查询
    - 实体查询
    - 关系查询
    - 统计信息
    - LLM可直接调用
    """

    # Tool基本信息
    name = "knowledge_query"
    description = "查询A_Memorix知识库，支持检索、实体查询、关系查询和统计信息"

    # Tool参数定义
    parameters: List[Tuple[str, ToolParamType, str, bool, List[str] | None]] = [
        (
            "query_type",
            ToolParamType.STRING,
            "查询类型：search(检索)、entity(实体)、relation(关系)、stats(统计)",
            True,
            ["search", "entity", "relation", "stats"],
        ),
        (
            "query",
            ToolParamType.STRING,
            "查询内容（检索文本/实体名称/关系规格），stats模式不需要",
            False,
            None,
        ),
        (
            "top_k",
            ToolParamType.INTEGER,
            "返回结果数量（仅search模式）",
            False,
            None,
        ),
        (
            "use_threshold",
            ToolParamType.BOOLEAN,
            "是否使用动态阈值过滤（仅search模式）",
            False,
            None,
        ),
    ]

    # LLM可用
    available_for_llm = True

    def __init__(self, plugin_config: Optional[dict] = None, chat_stream: Optional["ChatStream"] = None):
        """初始化知识查询Tool"""
        super().__init__(plugin_config, chat_stream)

        # 获取存储实例
        self.vector_store = self.plugin_config.get("vector_store")
        self.graph_store = self.plugin_config.get("graph_store")
        self.metadata_store = self.plugin_config.get("metadata_store")
        self.embedding_manager = self.plugin_config.get("embedding_manager")

        # 初始化检索器
        self.retriever: Optional[DualPathRetriever] = None
        self.threshold_filter: Optional[DynamicThresholdFilter] = None

        # 设置日志前缀
        chat_id = self.chat_id if self.chat_id else "unknown"
        self.log_prefix = f"[KnowledgeQueryTool-{chat_id}]"

        # 初始化组件
        self._initialize_components()

    @property
    def debug_enabled(self) -> bool:
        """检查是否启用了调试模式"""
        advanced = self.plugin_config.get("advanced", {})
        if isinstance(advanced, dict):
            return advanced.get("debug", False)
        return self.plugin_config.get("debug", False)

    def _initialize_components(self) -> None:
        """初始化检索和过滤组件"""
        try:
            # 检查存储是否可用 (优先从配置获取，兜底从插件实例获取)
            vector_store = self.vector_store
            graph_store = self.graph_store
            metadata_store = self.metadata_store
            embedding_manager = self.embedding_manager

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
                    
                    # 同步回实例属性
                    self.vector_store = vector_store
                    self.graph_store = graph_store
                    self.metadata_store = metadata_store
                    self.embedding_manager = embedding_manager


            # 最终检查 (使用 is not None 而非布尔值，因为空对象可能为 False)
            if not all([
                vector_store is not None,
                graph_store is not None,
                metadata_store is not None,
                embedding_manager is not None
            ]):
                logger.warning(f"{self.log_prefix} 存储组件未完全初始化")
                return

            # 创建检索器配置
            config = DualPathRetrieverConfig(
                top_k_paragraphs=self.get_config("retrieval.top_k_paragraphs", 20),
                top_k_relations=self.get_config("retrieval.top_k_relations", 10),
                top_k_final=self.get_config("retrieval.top_k_final", 10),
                alpha=self.get_config("retrieval.alpha", 0.5),
                enable_ppr=self.get_config("retrieval.enable_ppr", True),
                ppr_alpha=self.get_config("retrieval.ppr_alpha", 0.85),
                enable_parallel=self.get_config("retrieval.enable_parallel", True),
                retrieval_strategy=RetrievalStrategy.DUAL_PATH,
            )

            # 创建检索器
            self.retriever = DualPathRetriever(
                vector_store=self.vector_store,
                graph_store=self.graph_store,
                metadata_store=self.metadata_store,
                embedding_manager=self.embedding_manager,
                config=config,
            )

            # 创建阈值过滤器
            threshold_config = ThresholdConfig(
                method=ThresholdMethod.ADAPTIVE,
                min_threshold=self.get_config("threshold.min_threshold", 0.3),
                max_threshold=self.get_config("threshold.max_threshold", 0.95),
                percentile=self.get_config("threshold.percentile", 75.0),
                std_multiplier=self.get_config("threshold.std_multiplier", 1.5),
                min_results=self.get_config("threshold.min_results", 3),
                enable_auto_adjust=self.get_config("threshold.enable_auto_adjust", True),
            )

            self.threshold_filter = DynamicThresholdFilter(threshold_config)

            logger.info(f"{self.log_prefix} 知识查询Tool初始化完成")

        except Exception as e:
            logger.error(f"{self.log_prefix} 组件初始化失败: {e}")

    async def execute(self, function_args: dict[str, Any]) -> dict[str, Any]:
        """执行工具函数（供LLM调用）

        Args:
            function_args: 工具调用参数
                - query_type: 查询类型
                - query: 查询内容
                - top_k: 返回结果数量
                - use_threshold: 是否使用阈值过滤

        Returns:
            dict: 工具执行结果
        """
        # 检查组件是否初始化
        if not self.retriever:
            return {
                "success": False,
                "error": "知识查询Tool未初始化",
                "content": "❌ 知识查询Tool未初始化",
                "results": [],
            }

        # 解析参数
        query_type = function_args.get("query_type", "search")
        query = function_args.get("query", "")
        top_k = function_args.get("top_k", 10)
        use_threshold = function_args.get("use_threshold", True)

        logger.info(
            f"{self.log_prefix} LLM调用: query_type={query_type}, "
            f"query='{query}', top_k={top_k}"
        )

        if self.debug_enabled:
            logger.info(f"{self.log_prefix} [DEBUG] 工具完整参数: {function_args}")

        try:
            # 根据查询类型执行
            if query_type == "search":
                result = await self._search(query, top_k, use_threshold)
            elif query_type == "entity":
                result = await self._query_entity(query)
            elif query_type == "relation":
                result = await self._query_relation(query)
            elif query_type == "stats":
                result = self._get_stats()
            else:
                result = {
                    "success": False,
                    "error": f"未知的查询类型: {query_type}",
                    "content": f"❌ 未知的查询类型: {query_type}",
                    "results": [],
                }

            return result

        except Exception as e:
            error_msg = f"查询失败: {str(e)}"
            logger.error(f"{self.log_prefix} {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "content": f"❌ 查询发生错误: {error_msg}",
                "results": [],
            }

    async def direct_execute(
        self,
        query_type: str = "search",
        query: str = "",
        top_k: int = 10,
        use_threshold: bool = True,
    ) -> Dict[str, Any]:
        """直接执行工具函数（供插件调用）

        Args:
            query_type: 查询类型
            query: 查询内容
            top_k: 返回结果数量
            use_threshold: 是否使用阈值过滤

        Returns:
            Dict: 执行结果
        """
        function_args = {
            "query_type": query_type,
            "query": query,
            "top_k": top_k,
            "use_threshold": use_threshold,
        }

        return await self.execute(function_args)

    async def _search(
        self,
        query: str,
        top_k: int,
        use_threshold: bool,
    ) -> Dict[str, Any]:
        """执行检索查询

        Args:
            query: 查询文本
            top_k: 返回结果数量
            use_threshold: 是否使用阈值过滤

        Returns:
            查询结果字典
        """
        if not query:
            return {
                "success": False,
                "error": "查询内容不能为空",
                "content": "⚠️ 查询内容不能为空",
                "results": [],
            }

        start_time = time.time()

        # 执行检索（异步调用）
        results = await self.retriever.retrieve(query, top_k=top_k)

        # 应用阈值过滤
        if use_threshold and self.threshold_filter:
            results = self.threshold_filter.filter(results)
            if self.debug_enabled:
                logger.info(f"{self.log_prefix} [DEBUG] 过滤后结果数量 (Tool): {len(results)}")

        elapsed = time.time() - start_time

        # 格式化结果
        formatted_results = []
        try:
            for i, result in enumerate(results):
                # DEBUG: Check result type
                if self.debug_enabled:
                    logger.info(f"{self.log_prefix} Result {i} type: {type(result)}")
                    
                formatted_results.append({
                    "type": result.result_type,
                    "score": float(result.score),
                    "content": result.content,
                    "metadata": result.metadata,
                })
        except Exception as e:
            logger.error(f"{self.log_prefix} Error formatting results: {e}")
            raise

        # 生成 content 摘要
        if formatted_results:
            summary_lines = [f"找到 {len(formatted_results)} 条结果："]
            for i, res in enumerate(formatted_results[:5]):
                type_icon = "📄" if res['type'] == 'paragraph' else "🔗"
                try:
                    summary_lines.append(f"{i+1}. {type_icon} {res.get('content', 'N/A')} ({res.get('score', 0.0):.2f})")
                except Exception as e:
                     logger.error(f"{self.log_prefix} Error generating summary for index {i}: {e}")
                     # Defensively continue
                     summary_lines.append(f"{i+1}. {type_icon} [Error accessing content] ({res.get('score', 0.0):.2f})")
                     
            content = "\n".join(summary_lines)
        else:
            content = "未找到相关结果。"

        return {
            "success": True,
            "query_type": "search",
            "query": query,
            "results": formatted_results,
            "count": len(formatted_results),
            "elapsed_ms": elapsed * 1000,
            "content": content,
        }

    async def _query_entity(self, entity_name: str) -> Dict[str, Any]:
        """查询实体信息

        Args:
            entity_name: 实体名称

        Returns:
            查询结果字典
        """
        if not entity_name:

            return {
                "success": False,
                "error": "实体名称不能为空",
                "content": "⚠️ 实体名称不能为空",
                "results": [],
            }

        # 检查实体是否存在
        if not self.graph_store.has_node(entity_name):

            return {
                "success": False,
                "error": f"实体不存在: {entity_name}",
                "content": f"❌ 实体 '{entity_name}' 不存在",
                "results": [],
            }

        # 获取邻居节点
        neighbors = self.graph_store.get_neighbors(entity_name)

        # 获取相关段落
        paragraphs = self.metadata_store.get_paragraphs_by_entity(entity_name)

        # 格式化段落
        formatted_paragraphs = [
            {
                "hash": para["hash"],
                "content": para["content"],
                "created_at": para.get("created_at"),
            }
            for para in paragraphs
        ]


        # 生成 content 摘要
        content_lines = [f"实体 '{entity_name}' 信息："]
        content_lines.append(f"- 邻居节点 ({len(neighbors)}): {', '.join(neighbors[:10])}{'...' if len(neighbors)>10 else ''}")
        content_lines.append(f"- 相关段落 ({len(paragraphs)}):")
        for i, para in enumerate(formatted_paragraphs[:3]):
             content_lines.append(f"  {i+1}. {para['content'][:50]}...")
        
        content = "\n".join(content_lines)

        return {
            "success": True,
            "query_type": "entity",
            "entity": entity_name,
            "neighbors": neighbors,
            "related_paragraphs": formatted_paragraphs,
            "neighbor_count": len(neighbors),
            "paragraph_count": len(paragraphs),
            "content": content,
        }

    async def _query_relation(self, relation_spec: str) -> Dict[str, Any]:
        """查询关系信息

        Args:
            relation_spec: 关系规格

        Returns:
            查询结果字典
        """
        if not relation_spec:

            return {
                "success": False,
                "error": "关系规格不能为空",
                "content": "⚠️ 关系规格不能为空",
                "results": [],
            }

        # 解析关系规格
        if "|" in relation_spec:
            parts = relation_spec.split("|")
            if len(parts) < 2:
                return {
                    "success": False,
                    "error": "关系格式错误",
                    "content": "❌ 关系格式错误",
                    "results": [],
                }
            subject = parts[0].strip()
            predicate = parts[1].strip()
            obj = parts[2].strip() if len(parts) > 2 else None
        else:
            parts = relation_spec.split(maxsplit=1)
            if len(parts) < 2:
                return {
                    "success": False,
                    "error": "关系格式错误",
                    "content": "❌ 关系格式错误",
                    "results": [],
                }
            subject = parts[0].strip()
            predicate = parts[1].strip()
            obj = None

        # 查询关系
        relations = self.metadata_store.get_relations(
            subject=subject if subject else None,
            predicate=predicate if predicate else None,
            object=obj if obj else None,
        )

        # 格式化关系
        formatted_relations = []
        for rel in relations:
            formatted_relations.append({
                "hash": rel["hash"],
                "subject": rel["subject"],
                "predicate": rel["predicate"],
                "object": rel["object"],  # 数据库列名就是 'object'
                "confidence": rel.get("confidence", 1.0),
            })


        # 生成 content 摘要
        if formatted_relations:
            lines = [f"找到 {len(formatted_relations)} 条关系："]
            for i, rel in enumerate(formatted_relations[:10]):
                lines.append(f"{i+1}. {rel['subject']} {rel['predicate']} {rel['object']}")
            content = "\n".join(lines)
        else:
            content = "未找到符合条件的关系。"

        return {
            "success": True,
            "query_type": "relation",
            "spec": {"subject": subject, "predicate": predicate, "object": obj},
            "results": formatted_relations,
            "count": len(formatted_relations),
            "content": content,
        }

    def _get_stats(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        stats = {
            "vector_store": {
                "num_vectors": self.vector_store.num_vectors if self.vector_store else 0,
                "dimension": self.vector_store.dimension if self.vector_store else 0,
            },
            "graph_store": {
                "num_nodes": self.graph_store.num_nodes if self.graph_store else 0,
                "num_edges": self.graph_store.num_edges if self.graph_store else 0,
            },
            "metadata_store": {
                "num_paragraphs": self.metadata_store.count_paragraphs() if self.metadata_store else 0,
                "num_relations": self.metadata_store.count_relations() if self.metadata_store else 0,
                "num_entities": self.metadata_store.count_entities() if self.metadata_store else 0,
            },
        }

        # Format a human-readable summary
        content = (
            f"📊 知识库统计信息\n\n"
            f"📦 向量存储:\n"
            f"  - 向量数量: {stats['vector_store']['num_vectors']}\n"
            f"  - 维度: {stats['vector_store']['dimension']}\n\n"
            f"🕸️ 图存储:\n"
            f"  - 节点数: {stats['graph_store']['num_nodes']}\n"
            f"  - 边数: {stats['graph_store']['num_edges']}\n\n"
            f"📝 元数据存储:\n"
            f"  - 段落数: {stats['metadata_store']['num_paragraphs']}\n"
            f"  - 关系数: {stats['metadata_store']['num_relations']}\n"
            f"  - 实体数: {stats['metadata_store']['num_entities']}"
        )

        return {
            "success": True,
            "query_type": "stats",
            "content": content,
            "statistics": stats,
        }

    def get_tool_info_summary(self) -> str:
        """获取工具信息摘要

        Returns:
            工具信息摘要文本
        """
        if not self.retriever:
            return "❌ 知识查询Tool未初始化"

        lines = [
            "🔧 知识查询Tool信息",
            "",
            "📋 基本信息:",
            f"  - 名称: {self.name}",
            f"  - 描述: {self.description}",
            f"  - LLM可用: {'是' if self.available_for_llm else '否'}",
            "",
            "⚙️ 检索配置:",
            f"  - Top-K段落: {self.retriever.config.top_k_paragraphs}",
            f"  - Top-K关系: {self.retriever.config.top_k_relations}",
            f"  - 融合系数(alpha): {self.retriever.config.alpha}",
            f"  - PPR启用: {'是' if self.retriever.config.enable_ppr else '否'}",
            f"  - 并行检索: {'是' if self.retriever.config.enable_parallel else '否'}",
            "",
            "📊 存储统计:",
            f"  - 向量数量: {self.vector_store.num_vectors if self.vector_store else 0}",
            f"  - 节点数量: {self.graph_store.num_nodes if self.graph_store else 0}",
            f"  - 边数量: {self.graph_store.num_edges if self.graph_store else 0}",
            f"  - 段落数量: {self.metadata_store.count_paragraphs() if self.metadata_store else 0}",
        ]

        return "\n".join(lines)
