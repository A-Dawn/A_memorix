"""
双路检索器

同时检索关系和段落，实现知识图谱增强的检索。
"""

import asyncio
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Union
from enum import Enum

import numpy as np

from src.common.logger import get_logger
from ..storage import VectorStore, GraphStore, MetadataStore
from ..embedding import EmbeddingManager
from ..utils.matcher import AhoCorasick
from .pagerank import PersonalizedPageRank, PageRankConfig

logger = get_logger("A_Memorix.DualPathRetriever")


class RetrievalStrategy(Enum):
    """检索策略"""

    PARA_ONLY = "paragraph_only"  # 仅段落检索
    REL_ONLY = "relation_only"   # 仅关系检索
    DUAL_PATH = "dual_path"      # 双路检索（推荐）


@dataclass
class RetrievalResult:
    """
    检索结果

    属性：
        hash_value: 哈希值
        content: 内容（段落或关系）
        score: 相似度分数
        result_type: 结果类型（paragraph/relation）
        source: 来源（paragraph_search/relation_search/fusion）
        metadata: 额外元数据
    """

    hash_value: str
    content: str
    score: float
    result_type: str  # "paragraph" or "relation"
    source: str  # "paragraph_search", "relation_search", "fusion"
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "hash": self.hash_value,
            "content": self.content,
            "score": self.score,
            "type": self.result_type,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class DualPathRetrieverConfig:
    """
    双路检索器配置

    属性：
        top_k_paragraphs: 段落检索数量
        top_k_relations: 关系检索数量
        top_k_final: 最终返回数量
        alpha: 段落和关系的融合权重（0-1）
            - 0: 仅使用关系分数
            - 1: 仅使用段落分数
            - 0.5: 平均融合
        enable_ppr: 是否启用PageRank重排序
        ppr_alpha: PageRank的alpha参数
        enable_parallel: 是否并行检索
        retrieval_strategy: 检索策略
    """

    top_k_paragraphs: int = 20
    top_k_relations: int = 10
    top_k_final: int = 10
    alpha: float = 0.5  # 融合权重
    enable_ppr: bool = True
    ppr_alpha: float = 0.85
    enable_parallel: bool = True
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.DUAL_PATH

    def __post_init__(self):
        """验证配置"""
        if not 0 <= self.alpha <= 1:
            raise ValueError(f"alpha必须在[0, 1]之间: {self.alpha}")

        if self.top_k_paragraphs <= 0:
            raise ValueError(f"top_k_paragraphs必须大于0: {self.top_k_paragraphs}")

        if self.top_k_relations <= 0:
            raise ValueError(f"top_k_relations必须大于0: {self.top_k_relations}")

        if self.top_k_final <= 0:
            raise ValueError(f"top_k_final必须大于0: {self.top_k_final}")


class DualPathRetriever:
    """
    双路检索器

    功能：
    - 并行检索段落和关系
    - 结果融合与排序
    - PageRank重排序
    - 实体识别与加权

    参数：
        vector_store: 向量存储
        graph_store: 图存储
        metadata_store: 元数据存储
        embedding_manager: 嵌入管理器
        config: 检索配置
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        metadata_store: MetadataStore,
        embedding_manager: EmbeddingManager,
        config: Optional[DualPathRetrieverConfig] = None,
    ):
        """
        初始化双路检索器

        Args:
            vector_store: 向量存储
            graph_store: 图存储
            metadata_store: 元数据存储
            embedding_manager: 嵌入管理器
            config: 检索配置
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.metadata_store = metadata_store
        self.embedding_manager = embedding_manager
        self.config = config or DualPathRetrieverConfig()

        # PageRank计算器
        ppr_config = PageRankConfig(alpha=self.config.ppr_alpha)
        self._ppr = PersonalizedPageRank(
            graph_store=graph_store,
            config=ppr_config,
        )

        logger.info(
            f"DualPathRetriever 初始化: "
            f"strategy={self.config.retrieval_strategy.value}, "
            f"top_k_para={self.config.top_k_paragraphs}, "
            f"top_k_rel={self.config.top_k_relations}"
        )

        # 缓存 Aho-Corasick 匹配器
        self._ac_matcher: Optional[AhoCorasick] = None
        self._ac_nodes_count = 0

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        strategy: Optional[RetrievalStrategy] = None,
    ) -> List[RetrievalResult]:
        """
        执行检索（异步方法）

        Args:
            query: 查询文本
            top_k: 返回结果数量（默认使用配置值）
            strategy: 检索策略（默认使用配置值）

        Returns:
            检索结果列表
        """
        top_k = top_k or self.config.top_k_final
        strategy = strategy or self.config.retrieval_strategy

        logger.info(f"执行检索: query='{query[:50]}...', strategy={strategy.value}")

        # 根据策略执行检索
        if strategy == RetrievalStrategy.PARA_ONLY:
            results = await self._retrieve_paragraphs_only(query, top_k)
        elif strategy == RetrievalStrategy.REL_ONLY:
            results = await self._retrieve_relations_only(query, top_k)
        else:  # DUAL_PATH
            results = await self._retrieve_dual_path(query, top_k)

        logger.info(f"检索完成: 返回 {len(results)} 条结果")
        return results

    async def _retrieve_paragraphs_only(
        self,
        query: str,
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        仅检索段落（异步方法）

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        # 生成查询嵌入（异步调用）
        query_emb = await self.embedding_manager.encode(query)

        # 检索段落
        para_ids, para_scores = self.vector_store.search(
            query_emb,
            k=top_k * 2,  # 检索更多，后续过滤
        )

        # 获取段落内容
        results = []
        for hash_value, score in zip(para_ids, para_scores):
            paragraph = self.metadata_store.get_paragraph(hash_value)
            if paragraph is None:
                continue

            results.append(RetrievalResult(
                hash_value=hash_value,
                content=paragraph["content"],
                score=float(score),
                result_type="paragraph",
                source="paragraph_search",
                metadata={"word_count": paragraph.get("word_count", 0)},
            ))

        return results[:top_k]

    async def _retrieve_relations_only(
        self,
        query: str,
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        仅检索关系（异步方法）

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        # 生成查询嵌入（异步调用）
        query_emb = await self.embedding_manager.encode(query)

        # 检索关系
        rel_ids, rel_scores = self.vector_store.search(
            query_emb,
            k=top_k * 2,
        )

        # 获取关系内容
        results = []
        for hash_value, score in zip(rel_ids, rel_scores):
            relation = self.metadata_store.get_relation(hash_value)
            if relation is None:
                continue

            # 格式化关系文本
            content = f"{relation['subject']} {relation['predicate']} {relation['object']}"

            results.append(RetrievalResult(
                hash_value=hash_value,
                content=content,
                score=float(score),
                result_type="relation",
                source="relation_search",
                metadata={
                    "subject": relation["subject"],
                    "predicate": relation["predicate"],
                    "object": relation["object"],
                    "confidence": relation.get("confidence", 1.0),
                },
            ))

        return results[:top_k]

    async def _retrieve_dual_path(
        self,
        query: str,
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        双路检索（段落+关系）（异步方法）

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            融合后的检索结果列表
        """
        # 生成查询嵌入（异步调用）
        query_emb = await self.embedding_manager.encode(query)

        # 并行检索（使用 asyncio）
        if self.config.enable_parallel:
            para_results, rel_results = await self._parallel_retrieve(query_emb)
        else:
            para_results, rel_results = self._sequential_retrieve(query_emb)

        # 融合结果
        fused_results = self._fuse_results(
            para_results,
            rel_results,
            query_emb,
        )

        # PageRank重排序
        if self.config.enable_ppr:
            fused_results = self._rerank_with_ppr(
                fused_results,
                query,
            )

        return fused_results[:top_k]

    async def _parallel_retrieve(
        self,
        query_emb: np.ndarray,
    ) -> Tuple[List[RetrievalResult], List[RetrievalResult]]:
        """
        并行检索段落和关系（异步方法）

        Args:
            query_emb: 查询嵌入

        Returns:
            (段落结果, 关系结果)
        """
        # 使用 asyncio.gather 并发执行两个搜索任务
        # 由于 _search_paragraphs 和 _search_relations 是 CPU 密集型同步函数，
        # 使用 asyncio.to_thread 在线程池中执行
        try:
            para_task = asyncio.to_thread(
                self._search_paragraphs,
                query_emb,
                self.config.top_k_paragraphs,
            )
            rel_task = asyncio.to_thread(
                self._search_relations,
                query_emb,
                self.config.top_k_relations,
            )
            
            para_results, rel_results = await asyncio.gather(
                para_task, rel_task, return_exceptions=True
            )
            
            # 处理异常
            if isinstance(para_results, Exception):
                logger.error(f"段落检索失败: {para_results}")
                para_results = []
            if isinstance(rel_results, Exception):
                logger.error(f"关系检索失败: {rel_results}")
                rel_results = []
                
            return para_results, rel_results
            
        except Exception as e:
            logger.error(f"并行检索失败: {e}")
            return [], []

    def _sequential_retrieve(
        self,
        query_emb: np.ndarray,
    ) -> Tuple[List[RetrievalResult], List[RetrievalResult]]:
        """
        顺序检索段落和关系

        Args:
            query_emb: 查询嵌入

        Returns:
            (段落结果, 关系结果)
        """
        para_results = self._search_paragraphs(
            query_emb,
            self.config.top_k_paragraphs,
        )

        rel_results = self._search_relations(
            query_emb,
            self.config.top_k_relations,
        )

        return para_results, rel_results

    def _search_paragraphs(
        self,
        query_emb: np.ndarray,
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        搜索段落

        Args:
            query_emb: 查询嵌入
            top_k: 返回数量

        Returns:
            段落结果列表
        """
        para_ids, para_scores = self.vector_store.search(query_emb, k=top_k)

        results = []
        for hash_value, score in zip(para_ids, para_scores):
            paragraph = self.metadata_store.get_paragraph(hash_value)
            if paragraph is None:
                continue

            results.append(RetrievalResult(
                hash_value=hash_value,
                content=paragraph["content"],
                score=float(score),
                result_type="paragraph",
                source="paragraph_search",
                metadata={"word_count": paragraph.get("word_count", 0)},
            ))

        return results

    def _search_relations(
        self,
        query_emb: np.ndarray,
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        搜索关系

        Args:
            query_emb: 查询嵌入
            top_k: 返回数量

        Returns:
            关系结果列表
        """
        rel_ids, rel_scores = self.vector_store.search(query_emb, k=top_k)

        results = []
        for hash_value, score in zip(rel_ids, rel_scores):
            relation = self.metadata_store.get_relation(hash_value)
            if relation is None:
                continue

            content = f"{relation['subject']} {relation['predicate']} {relation['object']}"

            results.append(RetrievalResult(
                hash_value=hash_value,
                content=content,
                score=float(score),
                result_type="relation",
                source="relation_search",
                metadata={
                    "subject": relation["subject"],
                    "predicate": relation["predicate"],
                    "object": relation["object"],
                    "confidence": relation.get("confidence", 1.0),
                },
            ))

        return results

    def _fuse_results(
        self,
        para_results: List[RetrievalResult],
        rel_results: List[RetrievalResult],
        query_emb: np.ndarray,
    ) -> List[RetrievalResult]:
        """
        融合段落和关系结果

        融合策略：
        1. 计算加权分数
        2. 去重（基于段落和关系的关联）
        3. 排序

        Args:
            para_results: 段落结果
            rel_results: 关系结果
            query_emb: 查询嵌入

        Returns:
            融合后的结果列表
        """
        alpha = self.config.alpha

        # 为段落结果计算加权分数
        for result in para_results:
            result.score = result.score * alpha
            result.source = "fusion"

        # 为关系结果计算加权分数
        for result in rel_results:
            result.score = result.score * (1 - alpha)
            result.source = "fusion"

        # 合并结果
        all_results = para_results + rel_results

        # 去重：如果段落有关联的关系，只保留分数更高的
        seen_paragraphs = set()
        deduplicated_results = []

        for result in all_results:
            if result.result_type == "paragraph":
                hash_val = result.hash_value
                if hash_val not in seen_paragraphs:
                    seen_paragraphs.add(hash_val)
                    deduplicated_results.append(result)
            else:  # relation
                # 检查关系关联的段落是否已存在
                relation = self.metadata_store.get_relation(result.hash_value)
                if relation:
                    # 获取关联的段落
                    para_rels = self.metadata_store.query("""
                        SELECT paragraph_hash FROM paragraph_relations
                        WHERE relation_hash = ?
                    """, (result.hash_value,))

                    if para_rels:
                        # 检查段落是否已在结果中
                        for para_rel in para_rels:
                            if para_rel["paragraph_hash"] in seen_paragraphs:
                                # 段落已存在，跳过此关系
                                break
                        else:
                            # 所有段落都不存在，添加关系
                            deduplicated_results.append(result)
                    else:
                        # 没有关联段落，直接添加
                        deduplicated_results.append(result)
                else:
                    deduplicated_results.append(result)

        # 按分数排序
        deduplicated_results.sort(key=lambda x: x.score, reverse=True)

        return deduplicated_results

    def _rerank_with_ppr(
        self,
        results: List[RetrievalResult],
        query: str,
    ) -> List[RetrievalResult]:
        """
        使用PageRank重排序结果

        Args:
            results: 检索结果
            query: 查询文本

        Returns:
            重排序后的结果
        """
        # 从查询中提取实体
        entities = self._extract_entities(query)

        if not entities:
            logger.debug("未识别到实体，跳过PPR重排序")
            return results

        # 计算PPR分数
        ppr_scores = self._ppr.compute(
            personalization=entities,
            normalize=True,
        )

        # 调整结果分数
        for result in results:
            if result.result_type == "paragraph":
                # 获取段落的实体
                para_entities = self.metadata_store.get_paragraph_entities(
                    result.hash_value
                )

                # 计算实体的平均PPR分数
                if para_entities:
                    entity_scores = []
                    for ent in para_entities:
                        ent_hash = ent["hash"]
                        if ent_hash in ppr_scores:
                            entity_scores.append(ppr_scores[ent_hash])

                    if entity_scores:
                        avg_ppr = np.mean(entity_scores)
                        # 融合原始分数和PPR分数
                        result.score = result.score * 0.7 + avg_ppr * 0.3

        # 重新排序
        results.sort(key=lambda x: x.score, reverse=True)

        return results

    def _extract_entities(self, text: str) -> Dict[str, float]:
        """
        从文本中提取实体（简化版本）

        Args:
            text: 输入文本

        Returns:
            实体字典 {实体名: 权重}
        """
        # 获取所有实体
        all_entities = self.graph_store.get_nodes()
        if not all_entities:
            return {}

        # 检查是否需要更新 Aho-Corasick 匹配器
        if self._ac_matcher is None or self._ac_nodes_count != len(all_entities):
            self._ac_matcher = AhoCorasick()
            for entity in all_entities:
                self._ac_matcher.add_pattern(entity.lower())
            self._ac_matcher.build()
            self._ac_nodes_count = len(all_entities)

        # 执行匹配
        text_lower = text.lower()
        stats = self._ac_matcher.find_all(text_lower)

        # 映射回原始名称并使用出现次数作为权重
        node_map = {node.lower(): node for node in all_entities}
        entities = {node_map[low_name]: float(count) for low_name, count in stats.items()}

        return entities

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取检索统计信息

        Returns:
            统计信息字典
        """
        return {
            "config": {
                "top_k_paragraphs": self.config.top_k_paragraphs,
                "top_k_relations": self.config.top_k_relations,
                "top_k_final": self.config.top_k_final,
                "alpha": self.config.alpha,
                "enable_ppr": self.config.enable_ppr,
                "enable_parallel": self.config.enable_parallel,
                "strategy": self.config.retrieval_strategy.value,
            },
            "vector_store": {
                "size": self.vector_store.size,
            },
            "graph_store": {
                "num_nodes": self.graph_store.num_nodes,
                "num_edges": self.graph_store.num_edges,
            },
            "metadata_store": self.metadata_store.get_statistics(),
        }

    def __repr__(self) -> str:
        return (
            f"DualPathRetriever("
            f"strategy={self.config.retrieval_strategy.value}, "
            f"para_k={self.config.top_k_paragraphs}, "
            f"rel_k={self.config.top_k_relations})"
        )
