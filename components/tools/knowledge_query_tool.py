"""
知识查询Tool组件

提供LLM可调用的知识查询工具。
"""

import time
from typing import Any, List, Tuple, Optional, Dict
from pathlib import Path

from src.common.logger import get_logger
from src.plugin_system.apis import person_api
from src.plugin_system.base.base_tool import BaseTool
from src.plugin_system.base.component_types import ToolParamType
from src.chat.message_receive.chat_stream import ChatStream

# 导入核心模块
from ...core import (
    DualPathRetriever,
    RetrievalStrategy,
    DualPathRetrieverConfig,
    TemporalQueryOptions,
    DynamicThresholdFilter,
    ThresholdMethod,
    ThresholdConfig,
    SparseBM25Config,
    FusionConfig,
)
from ...core.utils.time_parser import parse_query_time_range
from ...core.utils.person_profile_service import PersonProfileService

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
            "查询类型：search(检索)、time(时序检索)、entity(实体)、relation(关系)、person(人物画像)、stats(统计)",
            True,
            ["search", "time", "entity", "relation", "person", "stats"],
        ),
        (
            "query",
            ToolParamType.STRING,
            "查询内容（检索文本/实体名称/关系规格），stats模式不需要",
            False,
            None,
        ),
        (
            "person_id",
            ToolParamType.STRING,
            "人物ID（person模式可选；为空时会尝试通过query或会话上下文解析）",
            False,
            None,
        ),
        (
            "top_k",
            ToolParamType.INTEGER,
            "返回结果数量（search/time模式）",
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
        (
            "time_from",
            ToolParamType.STRING,
            "开始时间（time模式，仅支持 YYYY/MM/DD 或 YYYY/MM/DD HH:mm；日期按 00:00 展开）",
            False,
            None,
        ),
        (
            "time_to",
            ToolParamType.STRING,
            "结束时间（time模式，仅支持 YYYY/MM/DD 或 YYYY/MM/DD HH:mm；日期按 23:59 展开）",
            False,
            None,
        ),
        (
            "person",
            ToolParamType.STRING,
            "人物过滤（time模式可选）",
            False,
            None,
        ),
        (
            "source",
            ToolParamType.STRING,
            "来源过滤（time模式可选）",
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
        self.sparse_index = self.plugin_config.get("sparse_index")

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
            sparse_index = self.sparse_index

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
                    
                    # 同步回实例属性
                    self.vector_store = vector_store
                    self.graph_store = graph_store
                    self.metadata_store = metadata_store
                    self.embedding_manager = embedding_manager
                    self.sparse_index = sparse_index


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
                vector_store=self.vector_store,
                graph_store=self.graph_store,
                metadata_store=self.metadata_store,
                embedding_manager=self.embedding_manager,
                sparse_index=self.sparse_index,
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
        query_type = str(function_args.get("query_type", "search") or "search").strip().lower()
        query = function_args.get("query", "")
        top_k_raw = function_args.get("top_k")
        default_top_k = (
            int(self.get_config("retrieval.temporal.default_top_k", 10))
            if query_type == "time"
            else 10
        )
        if top_k_raw is None:
            top_k = default_top_k
        else:
            try:
                top_k = max(1, int(top_k_raw))
            except (TypeError, ValueError):
                return {
                    "success": False,
                    "error": "top_k 必须是整数",
                    "content": "❌ top_k 必须是整数",
                    "results": [],
                }
        use_threshold = function_args.get("use_threshold", True)
        time_from = function_args.get("time_from")
        time_to = function_args.get("time_to")
        person = function_args.get("person")
        source = function_args.get("source")
        person_id = function_args.get("person_id")
        for_injection = bool(function_args.get("for_injection", False))
        force_refresh = bool(function_args.get("force_refresh", False))
        stream_id = function_args.get("stream_id")
        user_id = function_args.get("user_id")

        logger.info(
            f"{self.log_prefix} LLM调用: query_type={query_type}, "
            f"query='{query}', top_k={top_k}, time_from={time_from}, time_to={time_to}"
        )

        if self.debug_enabled:
            logger.info(f"{self.log_prefix} [DEBUG] 工具完整参数: {function_args}")

        try:
            # 根据查询类型执行
            if query_type == "search":
                result = await self._search(query, top_k, use_threshold)
            elif query_type == "time":
                result = await self._query_time(
                    query=query,
                    top_k=top_k,
                    time_from=time_from,
                    time_to=time_to,
                    person=person,
                    source=source,
                    use_threshold=use_threshold,
                )
            elif query_type == "entity":
                result = await self._query_entity(query)
            elif query_type == "relation":
                result = await self._query_relation(query)
            elif query_type == "person":
                result = await self._query_person(
                    query=query,
                    person_id=person_id,
                    top_k=top_k,
                    for_injection=for_injection,
                    force_refresh=force_refresh,
                    stream_id=stream_id,
                    user_id=user_id,
                )
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
        time_from: Optional[str] = None,
        time_to: Optional[str] = None,
        person: Optional[str] = None,
        source: Optional[str] = None,
        person_id: Optional[str] = None,
        for_injection: bool = False,
        force_refresh: bool = False,
        stream_id: Optional[str] = None,
        user_id: Optional[str] = None,
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
            "time_from": time_from,
            "time_to": time_to,
            "person": person,
            "source": source,
            "person_id": person_id,
            "for_injection": for_injection,
            "force_refresh": force_refresh,
            "stream_id": stream_id,
            "user_id": user_id,
        }

        return await self.execute(function_args)

    def _is_person_profile_injection_enabled(self, stream_id: Optional[str], user_id: Optional[str]) -> bool:
        if not bool(self.get_config("person_profile.enabled", True)):
            return False

        opt_in_required = bool(self.get_config("person_profile.opt_in_required", True))
        default_enabled = bool(self.get_config("person_profile.default_injection_enabled", False))

        if not opt_in_required:
            return default_enabled

        s_id = str(stream_id or "").strip()
        u_id = str(user_id or "").strip()
        if not s_id or not u_id or self.metadata_store is None:
            return False
        return bool(self.metadata_store.get_person_profile_switch(s_id, u_id, default=default_enabled))

    async def _query_person(
        self,
        query: str,
        person_id: Optional[str],
        top_k: int,
        for_injection: bool = False,
        force_refresh: bool = False,
        stream_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """查询人物画像。"""
        if not bool(self.get_config("person_profile.enabled", True)):
            if for_injection:
                return {
                    "success": True,
                    "query_type": "person",
                    "content": "",
                    "results": [],
                    "disabled_reason": "person_profile_module_disabled",
                }
            return {
                "success": False,
                "query_type": "person",
                "error": "人物画像功能未启用（person_profile.enabled=false）",
                "content": "❌ 人物画像功能未启用（person_profile.enabled=false）",
                "results": [],
            }

        resolved_stream_id = str(stream_id or self.chat_id or "").strip()
        resolved_user_id = str(user_id or "").strip()
        if not resolved_user_id and self.chat_stream and getattr(self.chat_stream, "user_info", None):
            resolved_user_id = str(getattr(self.chat_stream.user_info, "user_id", "") or "").strip()

        if for_injection and not self._is_person_profile_injection_enabled(resolved_stream_id, resolved_user_id):
            return {
                "success": True,
                "query_type": "person",
                "content": "",
                "results": [],
                "disabled_reason": "person_profile_not_opted_in",
            }

        service = PersonProfileService(
            metadata_store=self.metadata_store,
            graph_store=self.graph_store,
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            sparse_index=self.sparse_index,
            plugin_config=self.plugin_config,
            retriever=self.retriever,
        )

        pid = str(person_id or "").strip()
        if not pid and resolved_user_id and self.platform:
            try:
                pid = person_api.get_person_id(self.platform, resolved_user_id)
            except Exception:
                pid = ""
        if not pid and query:
            pid = service.resolve_person_id(str(query))

        if not pid:
            if for_injection:
                return {
                    "success": True,
                    "query_type": "person",
                    "content": "",
                    "results": [],
                    "disabled_reason": "person_id_unresolved",
                }
            return {
                "success": False,
                "query_type": "person",
                "error": "未能解析 person_id，请提供 person_id 或有效的人名/别名",
                "content": "❌ 未能解析 person_id，请提供 person_id 或有效的人名/别名",
                "results": [],
            }

        ttl_minutes = float(self.get_config("person_profile.profile_ttl_minutes", 360))
        ttl_seconds = max(60.0, ttl_minutes * 60.0)

        profile = await service.query_person_profile(
            person_id=pid,
            person_keyword=str(query or "").strip(),
            top_k=max(4, top_k),
            ttl_seconds=ttl_seconds,
            force_refresh=bool(force_refresh),
            source_note="knowledge_query:person",
        )

        if not profile.get("success", False):
            if for_injection:
                return {
                    "success": True,
                    "query_type": "person",
                    "content": "",
                    "results": [],
                    "error": profile.get("error", "unknown"),
                }
            return {
                "success": False,
                "query_type": "person",
                "error": profile.get("error", "unknown"),
                "content": "❌ 人物画像查询失败",
                "results": [],
            }

        if resolved_stream_id and resolved_user_id and self.metadata_store is not None:
            try:
                self.metadata_store.mark_person_profile_active(resolved_stream_id, resolved_user_id, pid)
            except Exception as e:
                logger.warning(f"{self.log_prefix} 记录活跃人物失败: {e}")

        persona_block = PersonProfileService.format_persona_profile_block(profile)
        if not persona_block and not for_injection:
            persona_block = "暂无足够证据形成该人物画像。"

        return {
            "success": True,
            "query_type": "person",
            "person_id": pid,
            "person_name": profile.get("person_name", ""),
            "profile_version": profile.get("profile_version"),
            "updated_at": profile.get("updated_at"),
            "expires_at": profile.get("expires_at"),
            "evidence_ids": profile.get("evidence_ids", []),
            "aliases": profile.get("aliases", []),
            "relation_edges": profile.get("relation_edges", []),
            "vector_evidence": profile.get("vector_evidence", []),
            "profile_source": profile.get("profile_source", "auto_snapshot"),
            "has_manual_override": bool(profile.get("has_manual_override", False)),
            "manual_override_text": profile.get("manual_override_text", ""),
            "auto_profile_text": profile.get("auto_profile_text", profile.get("profile_text", "")),
            "override_updated_at": profile.get("override_updated_at"),
            "override_updated_by": profile.get("override_updated_by", ""),
            "profile_text": profile.get("profile_text", ""),
            "content": persona_block,
            "results": [],
        }

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

        # 3. Smart Fallback if results are weak
        # 如果最高分 < 0.6，尝试提取实体并进行 Path Search
        max_score = 0.0
        if results:
            max_score = results[0].score

        fallback_triggered = False
        path_results = []
        
        # 可配置阈值 (TODO: 移至 Config)
        SMART_FALLBACK_THRESHOLD = 0.6
        
        if max_score < SMART_FALLBACK_THRESHOLD:
            # 尝试提取实体
            entities = self._extract_entities_from_query(query)
            if len(entities) == 2:
                if self.debug_enabled:
                    logger.info(f"{self.log_prefix} [Smart Fallback] Triggering Path Search for {entities}")
                
                path_data = self._path_search(query)
                if path_data and path_data.get("results"):
                    # 转换 path results 为 search result 格式
                    for p in path_data["results"]:
                        # 构造一个伪 RetrievalResult 类似的结构
                        path_results.append({
                            "type": "relation_path",
                            "score": 0.95, # 给赋予较高置信度，因为它基于图
                            "content": f"[Indirect Relation] {p['description']}",
                            "metadata": {"source": "graph_path", "nodes": p['nodes']}
                        })
                    fallback_triggered = True

        # 4. 合并结果 (Path Results 优先)
        # Convert original results to dict format first
        formatted_results = []
        try:
            for result in results:
                formatted_results.append({
                    "type": result.result_type,
                    "score": float(result.score),
                    "content": result.content,
                    "metadata": result.metadata,
                })
        except Exception as e:
            logger.error(f"{self.log_prefix} Error formatting results: {e}")

        # 如果触发了 Fallback，将 Path 结果加到前面
        if fallback_triggered:
            formatted_results = path_results + formatted_results

        # 5. Deduplication (Safe Mode)
        # 去重，但保留至少 1 条 (如果原结果不为空)
        original_count = len(formatted_results)
        formatted_results = self._deduplicate_results(formatted_results)
        
        if self.debug_enabled:
            logger.info(f"{self.log_prefix} Deduplication: {original_count} -> {len(formatted_results)}")

        # 6. 生成 content 摘要 (Clean Output)
        if formatted_results:
            summary_lines = [f"找到 {len(formatted_results)} 条相关信息："]
            for i, res in enumerate(formatted_results[:5]): # Top 5 for context
                type_icon = "📄" if res['type'] == 'paragraph' else "🔗"
                if res['type'] == 'relation_path': type_icon = "🛤️"
                
                content_text = res.get('content', 'N/A')
                # Remove score from LLM output to avoid bias
                # But keep it in logs/debug
                summary_lines.append(f"{i+1}. {type_icon} {content_text}")
                
            content = "\n".join(summary_lines)
            logger.info(f"{self.log_prefix} Returning {len(formatted_results)} results to LLM context")
        else:
            content = "未找到相关结果。"

        return {
            "success": True,
            "query_type": "search",
            "query": query,
            "results": formatted_results, # 包含分数的完整数据返回给程序
            "count": len(formatted_results),
            "elapsed_ms": elapsed * 1000,
            "content": content, # 给 LLM 看的摘要 (无分数)
        }

    async def _query_time(
        self,
        query: str,
        top_k: int,
        time_from: Optional[str],
        time_to: Optional[str],
        person: Optional[str],
        source: Optional[str],
        use_threshold: bool = True,
    ) -> Dict[str, Any]:
        """执行时序检索（可选语义query）。"""
        if not bool(self.get_config("retrieval.temporal.enabled", True)):
            return {
                "success": False,
                "error": "时序检索已禁用（retrieval.temporal.enabled=false）",
                "content": "❌ 时序检索已禁用（retrieval.temporal.enabled=false）",
                "results": [],
            }

        if not time_from and not time_to:
            return {
                "success": False,
                "error": "time模式至少需要time_from或time_to",
                "content": "❌ time模式至少需要time_from或time_to",
                "results": [],
            }

        try:
            ts_from, ts_to = parse_query_time_range(
                str(time_from) if time_from is not None else None,
                str(time_to) if time_to is not None else None,
            )
        except ValueError as e:
            return {
                "success": False,
                "error": f"时间参数错误: {e}",
                "content": f"❌ 时间参数错误: {e}",
                "results": [],
            }

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

        start_time = time.time()
        results = await self.retriever.retrieve(
            query=query,
            top_k=top_k,
            temporal=temporal,
        )
        if query and use_threshold and self.threshold_filter:
            results = self.threshold_filter.filter(results)
        elapsed = time.time() - start_time

        formatted_results = []
        for result in results:
            metadata = dict(result.metadata or {})
            if "time_meta" not in metadata:
                metadata["time_meta"] = {}
            formatted_results.append(
                {
                    "hash": result.hash_value,
                    "type": result.result_type,
                    "score": float(result.score),
                    "content": result.content,
                    "metadata": metadata,
                }
            )

        if formatted_results:
            lines = [f"找到 {len(formatted_results)} 条时间相关信息："]
            for i, item in enumerate(formatted_results[:5], 1):
                time_meta = item["metadata"].get("time_meta", {})
                s_text = time_meta.get("effective_start_text", "N/A")
                e_text = time_meta.get("effective_end_text", "N/A")
                basis = time_meta.get("match_basis", "none")
                lines.append(f"{i}. {item['content']}")
                lines.append(f"   时间: {s_text} ~ {e_text} ({basis})")
            content = "\n".join(lines)
        else:
            content = "未找到符合时间条件的结果。"

        return {
            "success": True,
            "query_type": "time",
            "query": query,
            "time_from": time_from,
            "time_to": time_to,
            "person": person,
            "source": source,
            "results": formatted_results,
            "count": len(formatted_results),
            "elapsed_ms": elapsed * 1000,
            "content": content,
        }

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对结果进行去重
        - 基于内容哈希/相似度
        - 保留分数最高的
        - 安全修正: 保证不因为去重导致结果为空
        """
        if not results:
            return []
            
        unique_results = []
        seen_hashes = set()
        seen_contents = set() # 为了处理不同 Hash 但内容极相似的情况 (简单前缀/包含检查)
        
        # 1. Path Results 总是保留 (只要不完全重复)
        # 2. Others based on content
        
        for res in results:
            # Simple content normalization
            content = res.get("content", "").strip()
            if not content:
                continue
                
            # Check exact hash (if available) or content hash
            res_md = res.get("metadata", {})
            h = res_md.get("hash") or str(hash(content))
            
            if h in seen_hashes:
                continue
                
            # Soft dedup: check if content is substring of already seen (or vice versa)
            # This is O(N^2) but N is small (TopK=10-20)
            is_dup = False
            for seen in seen_contents:
                # 如果极其相似 (比如只是前缀不同)，视为重复
                # 这里简单从严：如果 A 包含 B，保留 A (通常 A 信息量大) ?
                # 或者保留分数高的。
                # 简单策略: 如果 content 几乎一样 (Levenshtein costly)，这里用包含关系
                if content in seen or seen in content:
                    # 如果当前分数显著更高 (>0.1 diff)，则保留当前(这很难，因为我们是在 append)
                    # 假设 results 已经按 score 排序
                    is_dup = True
                    break
            
            if is_dup:
                continue
                
            seen_hashes.add(h)
            seen_contents.add(content)
            unique_results.append(res)
            
        # 安全修正: 如果去重后空了 (极不可能，因为第一条肯定进)，或者去得太狠
        # 这里只要 original results 有东西，unique_results 至少会有 1 条
        if not unique_results and results:
             unique_results.append(results[0])
             
        return unique_results

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
        # 获取配置
        enable_fallback = self.get_config("retrieval.relation_semantic_fallback", True)
        fallback_min_score = self.get_config("retrieval.relation_fallback_min_score", 0.3)
        
        # Path Search 配置
        enable_path_search = self.get_config("retrieval.relation_enable_path_search", True)
        path_trigger_threshold = self.get_config("retrieval.relation_path_trigger_threshold", 0.4)

        # 1. 结构化检测
        # 如果包含明确的分隔符，视为结构化查询
        is_structured = "|" in relation_spec or "->" in relation_spec

        # 2. 自然语言优先处理
        # 如果不是明确的结构化查询，且启用了回退（意味着支持语义模式），则直接使用语义检索
        if not is_structured and enable_fallback:
            return await self._semantic_search_relation(relation_spec, fallback_min_score)

        # 3. 结构化查询处理 (精确匹配)
        subject, predicate, obj = None, None, None

        if "|" in relation_spec:
            parts = relation_spec.split("|")
            if len(parts) >= 2:
                subject = parts[0].strip()
                predicate = parts[1].strip()
                obj = parts[2].strip() if len(parts) > 2 else None
        elif "->" in relation_spec:
             parts = relation_spec.split("->")
             if len(parts) >= 2:
                subject = parts[0].strip()
                predicate = parts[1].strip() # 简化处理，假设 -> 就是谓语的一部分或者分隔
                obj = parts[1].strip() # 这里 split 只有两部分，中间作为谓语处理有点模糊，暂且维持原逻辑或作为 binary
                # 实际上原逻辑没处理 ->, 这里仅做简单兼容，或者退回到 split()
                # 考虑到兼容性，这里仅以此作为"结构化"标志，解析还是尝试空格
                pass

        if not subject: # 尝试空格解析 (Legacy)
            parts = relation_spec.split(maxsplit=1)
            if len(parts) >= 2:
                subject = parts[0].strip()
                predicate = parts[1].strip()
                obj = None
            else:
                 # 无法解析为结构化，且没走 NL 路径 (说明 enable_fallback=False)
                 return {
                    "success": False,
                    "error": "关系格式错误 (请使用 S|P|O 或开启语义回退)",
                    "content": "❌ 关系格式错误: 请使用 'Subject|Predicate|Object' 格式",
                    "results": [],
                }

        # 执行精确查询
        relations = self.metadata_store.get_relations(
            subject=subject if subject else None,
            predicate=predicate if predicate else None,
            object=obj if obj else None,
        )

        # 4. 结构化查询失败的回退
        # 如果精确匹配无结果，且启用了回退，尝试语义检索
        if not relations and enable_fallback:
             # 使用原始查询字符串进行语义检索
             semantic_result = await self._semantic_search_relation(relation_spec, fallback_min_score)
             
             # 检查是否触发 Path Search
             # 触发条件: 启用且 (无结果 或 最高分低于阈值)
             hits_count = semantic_result.get("count", 0)
             max_score = 0.0
             if hits_count > 0 and semantic_result.get("results"):
                 max_score = semantic_result["results"][0].get("similarity", 0.0)
                 
             if enable_path_search and (hits_count == 0 or max_score < path_trigger_threshold):
                 if self.debug_enabled:
                     logger.info(f"{self.log_prefix} 触发路径搜索 (Hits={hits_count}, MaxScore={max_score:.2f})")
                     
                 path_result = self._path_search(relation_spec)
                 if path_result:
                     return path_result
             
             return semantic_result

        # 格式化精确匹配结果
        formatted_relations = []
        for rel in relations:
            formatted_relations.append({
                "hash": rel["hash"],
                "subject": rel["subject"],
                "predicate": rel["predicate"],
                "object": rel["object"],
                "confidence": rel.get("confidence", 1.0),
                "is_semantic": False,
            })

        # 生成 content 摘要
        if formatted_relations:
            lines = [f"找到 {len(formatted_relations)} 条精确匹配关系："]
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

    async def _semantic_search_relation(
        self,
        query: str,
        min_score: float,
    ) -> Dict[str, Any]:
        """执行语义关系检索

        Args:
            query: 查询文本
            min_score: 最小相似度阈值

        Returns:
            查询结果
        """
        if not self.retriever:
             return {
                "success": False,
                "error": "检索器未初始化",
                "content": "❌ 检索器未初始化",
                "results": [],
            }

        # 执行检索 (策略: REL_ONLY, TopK: 5)
        # 护栏 B: TopK 小一点
        results = await self.retriever.retrieve(
            query,
            top_k=5,
            strategy=RetrievalStrategy.REL_ONLY
        )

        formatted_results = []
        seen_relations = set()

        for res in results:
            # 护栏 B: 阈值过滤
            if res.score < min_score:
                continue
            
            # 护栏 D: 类型过滤 (retrieve REL_ONLY 应该只返回 relation，但防御性检查)
            if res.result_type != "relation":
                continue

            # 获取元数据
            meta = res.metadata
            subj = meta.get("subject", "?")
            pred = meta.get("predicate", "?")
            obj = meta.get("object", "?")
            
            # 护栏 D: 去重
            rel_key = (subj, pred, obj)
            if rel_key in seen_relations:
                continue
            seen_relations.add(rel_key)

            formatted_results.append({
                "hash": res.hash_value,
                "subject": subj,
                "predicate": pred,
                "object": obj,
                "confidence": meta.get("confidence", 1.0),
                "similarity": res.score,
                "is_semantic": True, # 标记为语义结果
            })

        # 护栏 C: 明确标注
        if formatted_results:
            lines = [f"找到 {len(formatted_results)} 条 [语义候选] 关系："]
            for i, rel in enumerate(formatted_results):
                lines.append(
                    f"{i+1}. {rel['subject']} {rel['predicate']} {rel['object']} "
                    f"(相似度: {rel['similarity']:.2f})"
                )
            
            lines.append("")
            lines.append("💡 若需精确过滤，请使用 'Subject|Predicate|Object' 格式")
            content = "\n".join(lines)
        else:
            content = (
                f"未找到相关的关系 (语义相似度均低于 {min_score})。\n"
                "💡 请尝试更具体的关系描述，或使用 'S|P|O' 格式进行精确查询。"
            )

        return {
            "success": True,
            "query_type": "relation",
            "search_mode": "semantic",
            "query": query,
            "results": formatted_results,
            "count": len(formatted_results),
            "content": content,
        }

    def _path_search(self, query: str) -> Optional[Dict[str, Any]]:
        """执行路径搜索 (多跳关系)"""
        # 1. 提取实体
        entities = self._extract_entities_from_query(query)
        if len(entities) != 2:
            if self.debug_enabled:
                logger.debug(f"{self.log_prefix} PathSearch Abort: Requires exactly 2 entities, found {len(entities)}: {entities}")
            return None
            
        start_node, end_node = entities[0], entities[1]
        
        # 2. 查找路径
        paths = self.graph_store.find_paths(
            start_node, 
            end_node, 
            max_depth=3, # Configurable?
            max_paths=5
        )
        
        if not paths:
            return None
            
        # 3. 丰富路径信息 (查找边上的关系谓语)
        formatted_paths = []
        edge_cache = {} # (u, v) -> relation_str
        
        for path_nodes in paths:
            path_desc = []
            valid_path = True
            
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i+1]
                
                # Check cache
                cache_key = tuple(sorted((u, v))) # Undirected cache key
                if cache_key in edge_cache:
                    rel_info = edge_cache[cache_key]
                else:
                    # Query metadata for relation u->v or v->u
                    # 优先找 u->v
                    rels = self.metadata_store.get_relations(subject=u, object=v)
                    direction = "->"
                    if not rels:
                        # 尝试 v->u
                        rels = self.metadata_store.get_relations(subject=v, object=u)
                        direction = "<-"
                    
                    if rels:
                        # Pick best confidence or first
                        best_rel = max(rels, key=lambda x: x.get("confidence", 1.0))
                        pred = best_rel.get("predicate", "related")
                        rel_info = (pred, direction, u, v) if direction == "->" else (pred, direction, v, u)
                    else:
                        rel_info = ("?", "->", u, v) # Should not happen if graph consistent
                        
                    edge_cache[cache_key] = rel_info
                
                pred, direction, src, tgt = rel_info
                if direction == "->":
                    step_str = f"-[{pred}]->"
                else:
                    step_str = f"<-[{pred}]-"
                path_desc.append(step_str)
            
            # Reconstruct full string: Node1 -[pred]-> Node2 ...
            full_path_str = path_nodes[0]
            for i, step in enumerate(path_desc):
                full_path_str += f" {step} {path_nodes[i+1]}"
            
            formatted_paths.append({
                "nodes": path_nodes,
                "description": full_path_str
            })

        # Generate content
        lines = [f"Found {len(formatted_paths)} indirect connection paths between '{start_node}' and '{end_node}':"]
        for i, p in enumerate(formatted_paths):
            lines.append(f"{i+1}. {p['description']}")
            
        content = "\n".join(lines)
        
        return {
            "success": True,
            "query_type": "relation",
            "search_mode": "path",
            "query": query,
            "results": formatted_paths,
            "count": len(formatted_paths),
            "content": content
        }

    def _extract_entities_from_query(self, query: str) -> List[str]:
        """从查询中提取已知的图节点实体 (简易启发式)"""
        if not self.graph_store:
            return []
            
        # 1. 简单的 N-gram 匹配 (N=1..4)
        tokens = query.replace("?", " ").replace("!", " ").replace(".", " ").split()
        found_entities = set()
        
        # 优化: 仅检查 query 中的 potential matches
        # 由于无法遍历所有 node，我们生成 query 的所有子串 check existence
        # O(L^2) where L is query length (small)
        
        n = len(tokens)
        # Max n-gram size: 4 or length of query
        max_n = min(4, n)
        
        # Greedy search: prioritize longer matches
        skip_indices = set()
        
        for size in range(max_n, 0, -1):
            for i in range(n - size + 1):
                # Check if this span is already covered
                if any(idx in skip_indices for idx in range(i, i+size)):
                    continue
                    
                span = " ".join(tokens[i : i+size])
                # Check original case first, then exact match only (kv store usually case sensitive-ish)
                # But user query might be lower/upper.
                # Use ignore_case=True to support "system" matches "System"
                matched_node = self.graph_store.find_node(span, ignore_case=True)
                if matched_node:
                    found_entities.add(matched_node)
                    # Mark indices as covered
                    for idx in range(i, i+size):
                        skip_indices.add(idx)
                else:
                    pass
                    
        return list(found_entities)

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
            "sparse": self.sparse_index.stats() if self.sparse_index else None,
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
        sparse_stats = stats.get("sparse")
        if sparse_stats:
            content += (
                f"\n\n🧩 稀疏检索:\n"
                f"  - 启用: {'是' if sparse_stats.get('enabled') else '否'}\n"
                f"  - 已加载: {'是' if sparse_stats.get('loaded') else '否'}\n"
                f"  - Tokenizer: {sparse_stats.get('tokenizer_mode', 'N/A')}\n"
                f"  - FTS文档数: {sparse_stats.get('doc_count', 0)}"
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
            f"  - 融合方法: {self.retriever.config.fusion.method}",
            f"  - PPR启用: {'是' if self.retriever.config.enable_ppr else '否'}",
            f"  - 并行检索: {'是' if self.retriever.config.enable_parallel else '否'}",
            "",
            "📊 存储统计:",
            f"  - 向量数量: {self.vector_store.num_vectors if self.vector_store else 0}",
            f"  - 节点数量: {self.graph_store.num_nodes if self.graph_store else 0}",
            f"  - 边数量: {self.graph_store.num_edges if self.graph_store else 0}",
            f"  - 段落数量: {self.metadata_store.count_paragraphs() if self.metadata_store else 0}",
        ]
        if self.sparse_index:
            sparse_stats = self.sparse_index.stats()
            lines.extend([
                "",
                "🧩 稀疏检索:",
                f"  - 启用: {'是' if sparse_stats.get('enabled') else '否'}",
                f"  - 已加载: {'是' if sparse_stats.get('loaded') else '否'}",
                f"  - Tokenizer: {sparse_stats.get('tokenizer_mode', 'N/A')}",
            ])

        return "\n".join(lines)
