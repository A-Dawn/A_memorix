"""
双路检索器

同时检索关系和段落，实现知识图谱增强的检索。
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union
from enum import Enum

import numpy as np

from src.common.logger import get_logger
from ..storage import VectorStore, GraphStore, MetadataStore
from ..embedding import EmbeddingManager
from ..utils.matcher import AhoCorasick
from ..utils.time_parser import format_timestamp
from .pagerank import PersonalizedPageRank, PageRankConfig
from .sparse_bm25 import SparseBM25Config, SparseBM25Index

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
        ppr_concurrency_limit: PPR计算的最大并发数
        enable_parallel: 是否并行检索
        retrieval_strategy: 检索策略
        debug: 是否启用调试模式（打印搜索结果原文）
    """
 
    top_k_paragraphs: int = 20
    top_k_relations: int = 10
    top_k_final: int = 10
    alpha: float = 0.5  # 融合权重
    enable_ppr: bool = True
    ppr_alpha: float = 0.85
    ppr_concurrency_limit: int = 4
    enable_parallel: bool = True
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.DUAL_PATH
    debug: bool = False
    sparse: SparseBM25Config = field(default_factory=SparseBM25Config)
    fusion: "FusionConfig" = field(default_factory=lambda: FusionConfig())

    def __post_init__(self):
        """验证配置"""
        if isinstance(self.sparse, dict):
            self.sparse = SparseBM25Config(**self.sparse)
        if isinstance(self.fusion, dict):
            self.fusion = FusionConfig(**self.fusion)

        if not 0 <= self.alpha <= 1:
            raise ValueError(f"alpha必须在[0, 1]之间: {self.alpha}")

        if self.top_k_paragraphs <= 0:
            raise ValueError(f"top_k_paragraphs必须大于0: {self.top_k_paragraphs}")

        if self.top_k_relations <= 0:
            raise ValueError(f"top_k_relations必须大于0: {self.top_k_relations}")

        if self.top_k_final <= 0:
            raise ValueError(f"top_k_final必须大于0: {self.top_k_final}")


@dataclass
class TemporalQueryOptions:
    """时序查询选项。"""

    time_from: Optional[float] = None
    time_to: Optional[float] = None
    person: Optional[str] = None
    source: Optional[str] = None
    allow_created_fallback: bool = True
    candidate_multiplier: int = 8
    max_scan: int = 1000


@dataclass
class FusionConfig:
    """融合配置。"""

    method: str = "weighted_rrf"  # weighted_rrf | alpha_legacy
    rrf_k: int = 60
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    normalize_score: bool = True
    normalize_method: str = "minmax"

    def __post_init__(self):
        self.method = str(self.method or "weighted_rrf").strip().lower()
        self.normalize_method = str(self.normalize_method or "minmax").strip().lower()
        self.rrf_k = max(1, int(self.rrf_k))
        self.vector_weight = max(0.0, float(self.vector_weight))
        self.bm25_weight = max(0.0, float(self.bm25_weight))
        s = self.vector_weight + self.bm25_weight
        if s <= 0:
            self.vector_weight = 0.7
            self.bm25_weight = 0.3
        elif abs(s - 1.0) > 1e-8:
            self.vector_weight /= s
            self.bm25_weight /= s


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
        sparse_index: Optional[SparseBM25Index] = None,
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
        self.sparse_index = sparse_index

        # PageRank计算器
        ppr_config = PageRankConfig(alpha=self.config.ppr_alpha)
        self._ppr = PersonalizedPageRank(
            graph_store=graph_store,
            config=ppr_config,
        )
        self._ppr_semaphore = asyncio.Semaphore(self.config.ppr_concurrency_limit)

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
        temporal: Optional[TemporalQueryOptions] = None,
    ) -> List[RetrievalResult]:
        """
        执行检索（异步方法）

        Args:
            query: 查询文本
            top_k: 返回结果数量（默认使用配置值）
            strategy: 检索策略（默认使用配置值）
            temporal: 时序查询选项（可选）

        Returns:
            检索结果列表
        """
        top_k = top_k or self.config.top_k_final
        strategy = strategy or self.config.retrieval_strategy

        logger.info(f"执行检索: query='{query[:50]}...', strategy={strategy.value}")

        if temporal and not (query or "").strip():
            return self._retrieve_temporal_only(temporal, top_k)

        # 根据策略执行检索
        if strategy == RetrievalStrategy.PARA_ONLY:
            results = await self._retrieve_paragraphs_only(query, top_k, temporal=temporal)
        elif strategy == RetrievalStrategy.REL_ONLY:
            results = await self._retrieve_relations_only(query, top_k, temporal=temporal)
        else:  # DUAL_PATH
            results = await self._retrieve_dual_path(query, top_k, temporal=temporal)

        logger.info(f"检索完成: 返回 {len(results)} 条结果")

        # 调试模式：打印结果原文
        if self.config.debug:
            logger.info(f"[DEBUG] 检索结果内容原文:")
            for i, res in enumerate(results):
                logger.info(f"  {i+1}. [{res.result_type}] (Score: {res.score:.4f}) {res.content}")

        return results

    def _cap_temporal_scan_k(
        self,
        candidate_k: int,
        temporal: Optional[TemporalQueryOptions],
    ) -> int:
        """对 temporal 模式候选召回数应用 max_scan 上限。"""
        k = max(1, int(candidate_k))
        if temporal and temporal.max_scan and temporal.max_scan > 0:
            k = min(k, int(temporal.max_scan))
        return max(1, k)

    def _is_valid_embedding(self, emb: Optional[np.ndarray]) -> bool:
        if emb is None:
            return False
        arr = np.asarray(emb, dtype=np.float32)
        if arr.ndim == 0 or arr.size == 0:
            return False
        return bool(np.all(np.isfinite(arr)))

    def _should_use_sparse(
        self,
        embedding_ok: bool,
        vector_results: Optional[List[RetrievalResult]] = None,
    ) -> bool:
        if not self.config.sparse.enabled or self.sparse_index is None:
            return False

        mode = self.config.sparse.mode
        if mode == "hybrid":
            return True
        if mode == "fallback_only":
            return not embedding_ok
        # auto
        if not embedding_ok:
            return True
        if not vector_results:
            return True
        best = max((float(r.score) for r in vector_results), default=0.0)
        return best < 0.45

    def _should_use_sparse_relations(
        self,
        embedding_ok: bool,
        relation_results: Optional[List[RetrievalResult]] = None,
    ) -> bool:
        if not self.config.sparse.enable_relation_sparse_fallback:
            return False
        return self._should_use_sparse(embedding_ok, relation_results)

    def _normalize_scores_minmax(self, results: List[RetrievalResult]) -> None:
        if not results:
            return
        vals = [float(r.score) for r in results]
        lo = min(vals)
        hi = max(vals)
        if hi - lo < 1e-12:
            for r in results:
                r.score = 1.0
            return
        for r in results:
            r.score = (float(r.score) - lo) / (hi - lo)

    def _fuse_ranked_lists_weighted_rrf(
        self,
        vector_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """按 weighted RRF 融合两路段落召回。"""
        if not vector_results:
            out = sparse_results[:]
            if self.config.fusion.normalize_score:
                self._normalize_scores_minmax(out)
            return out
        if not sparse_results:
            out = vector_results[:]
            if self.config.fusion.normalize_score:
                self._normalize_scores_minmax(out)
            return out

        k = self.config.fusion.rrf_k
        w_vec = self.config.fusion.vector_weight
        w_sparse = self.config.fusion.bm25_weight
        merged: Dict[str, RetrievalResult] = {}
        score_map: Dict[str, float] = {}

        for rank, item in enumerate(vector_results, start=1):
            h = item.hash_value
            if h not in merged:
                merged[h] = item
                merged[h].source = "fusion_rrf"
            score_map[h] = score_map.get(h, 0.0) + w_vec * (1.0 / (k + rank))

        for rank, item in enumerate(sparse_results, start=1):
            h = item.hash_value
            if h not in merged:
                merged[h] = item
                merged[h].source = "fusion_rrf"
            score_map[h] = score_map.get(h, 0.0) + w_sparse * (1.0 / (k + rank))

        out = list(merged.values())
        for item in out:
            item.score = float(score_map.get(item.hash_value, 0.0))

        out.sort(key=lambda x: x.score, reverse=True)
        if self.config.fusion.normalize_score and self.config.fusion.normalize_method == "minmax":
            self._normalize_scores_minmax(out)
        return out

    def _search_paragraphs_sparse(
        self,
        query: str,
        top_k: int,
        temporal: Optional[TemporalQueryOptions] = None,
    ) -> List[RetrievalResult]:
        """BM25 段落召回。"""
        if not self.sparse_index or not self.config.sparse.enabled:
            return []

        candidate_k = max(top_k, self.config.sparse.candidate_k)
        candidate_k = self._cap_temporal_scan_k(candidate_k, temporal)
        sparse_rows = self.sparse_index.search(query=query, k=candidate_k)
        results: List[RetrievalResult] = []
        for row in sparse_rows:
            hash_value = row["hash"]
            paragraph = self.metadata_store.get_paragraph(hash_value)
            if paragraph is None:
                continue
            time_meta = self._build_time_meta_from_paragraph(paragraph, temporal=temporal)
            results.append(
                RetrievalResult(
                    hash_value=hash_value,
                    content=paragraph["content"],
                    score=float(row.get("score", 0.0)),
                    result_type="paragraph",
                    source="sparse_bm25",
                    metadata={
                        "word_count": paragraph.get("word_count", 0),
                        "time_meta": time_meta,
                        "bm25_score": float(row.get("bm25_score", 0.0)),
                    },
                )
            )
        results = self._apply_temporal_filter_to_paragraphs(results, temporal)
        if self.config.fusion.normalize_score and self.config.fusion.normalize_method == "minmax":
            self._normalize_scores_minmax(results)
        return results

    def _search_relations_sparse(
        self,
        query: str,
        top_k: int,
        temporal: Optional[TemporalQueryOptions] = None,
    ) -> List[RetrievalResult]:
        """关系 BM25 召回。"""
        if not self.sparse_index or not self.config.sparse.enabled:
            return []
        if not self.config.sparse.enable_relation_sparse_fallback:
            return []

        candidate_k = max(top_k, self.config.sparse.relation_candidate_k)
        candidate_k = self._cap_temporal_scan_k(candidate_k, temporal)
        rows = self.sparse_index.search_relations(query=query, k=candidate_k)
        results: List[RetrievalResult] = []
        for row in rows:
            hash_value = row["hash"]
            relation = self.metadata_store.get_relation(hash_value)
            if relation is None:
                continue

            relation_time_meta = None
            if temporal:
                relation_time_meta = self._best_supporting_time_meta(hash_value, temporal)
                if relation_time_meta is None:
                    continue

            content = f"{relation['subject']} {relation['predicate']} {relation['object']}"
            results.append(
                RetrievalResult(
                    hash_value=hash_value,
                    content=content,
                    score=float(row.get("score", 0.0)),
                    result_type="relation",
                    source="sparse_relation_bm25",
                    metadata={
                        "subject": relation["subject"],
                        "predicate": relation["predicate"],
                        "object": relation["object"],
                        "confidence": relation.get("confidence", 1.0),
                        "time_meta": relation_time_meta,
                        "bm25_score": float(row.get("bm25_score", 0.0)),
                    },
                )
            )

        if self.config.fusion.normalize_score and self.config.fusion.normalize_method == "minmax":
            self._normalize_scores_minmax(results)
        return self._apply_temporal_filter_to_relations(results, temporal)

    def _merge_relation_results(
        self,
        vector_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """合并关系候选，按 hash 去重并保留更高分。"""
        merged: Dict[str, RetrievalResult] = {}
        for item in vector_results:
            merged[item.hash_value] = item
        for item in sparse_results:
            old = merged.get(item.hash_value)
            if old is None or float(item.score) > float(old.score):
                merged[item.hash_value] = item
            elif old is not None and old.source != item.source:
                old.source = "relation_fusion"
        out = list(merged.values())
        out.sort(key=lambda x: x.score, reverse=True)
        return out

    async def _retrieve_paragraphs_only(
        self,
        query: str,
        top_k: int,
        temporal: Optional[TemporalQueryOptions] = None,
    ) -> List[RetrievalResult]:
        """
        仅检索段落（异步方法）

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        query_emb = None
        embedding_ok = False
        vector_results: List[RetrievalResult] = []

        try:
            query_emb = await self.embedding_manager.encode(query)
            embedding_ok = self._is_valid_embedding(query_emb)
        except Exception as e:
            logger.warning(f"段落检索 embedding 生成失败，将尝试 sparse 回退: {e}")

        if embedding_ok:
            multiplier = max(1, temporal.candidate_multiplier) if temporal else 1
            candidate_k = self._cap_temporal_scan_k(top_k * 2 * multiplier, temporal)
            para_ids, para_scores = self.vector_store.search(
                query_emb,  # type: ignore[arg-type]
                k=candidate_k,
            )

            for hash_value, score in zip(para_ids, para_scores):
                paragraph = self.metadata_store.get_paragraph(hash_value)
                if paragraph is None:
                    continue
                time_meta = self._build_time_meta_from_paragraph(paragraph, temporal=temporal)
                vector_results.append(
                    RetrievalResult(
                        hash_value=hash_value,
                        content=paragraph["content"],
                        score=float(score),
                        result_type="paragraph",
                        source="paragraph_search",
                        metadata={
                            "word_count": paragraph.get("word_count", 0),
                            "time_meta": time_meta,
                        },
                    )
                )
            vector_results = self._apply_temporal_filter_to_paragraphs(vector_results, temporal)

        sparse_results: List[RetrievalResult] = []
        if self._should_use_sparse(embedding_ok, vector_results):
            sparse_results = self._search_paragraphs_sparse(query, top_k, temporal=temporal)

        if self.config.fusion.method == "weighted_rrf" and (vector_results and sparse_results):
            results = self._fuse_ranked_lists_weighted_rrf(vector_results, sparse_results)
        elif vector_results and sparse_results:
            results = vector_results + sparse_results
            results.sort(key=lambda x: x.score, reverse=True)
        else:
            results = vector_results if vector_results else sparse_results

        return results[:top_k]

    async def _retrieve_relations_only(
        self,
        query: str,
        top_k: int,
        temporal: Optional[TemporalQueryOptions] = None,
    ) -> List[RetrievalResult]:
        """
        仅检索关系 (通过实体枢纽 Entity-Pivot)
        
        策略:
        1. 检索向量库中的 Top-K 实体 (Entity)
        2. 通过图结构/元数据扩展出与实体关联的关系 (Relation)
        3. 以实体相似度作为基础分返回关系

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            检索结果列表
        """
        query_emb = None
        embedding_ok = False
        vector_results: List[RetrievalResult] = []
        try:
            query_emb = await self.embedding_manager.encode(query)
            embedding_ok = self._is_valid_embedding(query_emb)
        except Exception as e:
            logger.warning(f"关系检索 embedding 生成失败，将尝试 sparse 回退: {e}")

        if embedding_ok:
            # 1. 检索向量 (混合了段落和实体，所以扩大检索范围以召回足够多实体)
            multiplier = max(1, temporal.candidate_multiplier) if temporal else 1
            candidate_k = self._cap_temporal_scan_k(top_k * 3 * multiplier, temporal)
            ids, scores = self.vector_store.search(
                query_emb,  # type: ignore[arg-type]
                k=candidate_k,
            )

            seen_relations = set()
            for hash_value, score in zip(ids, scores):
                entity = self.metadata_store.get_entity(hash_value)
                if not entity:
                    continue
                entity_name = entity["name"]

                related_rels = []
                related_rels.extend(self.metadata_store.get_relations(subject=entity_name))
                related_rels.extend(self.metadata_store.get_relations(object=entity_name))

                for rel in related_rels:
                    if rel["hash"] in seen_relations:
                        continue
                    seen_relations.add(rel["hash"])

                    relation_time_meta = None
                    if temporal:
                        relation_time_meta = self._best_supporting_time_meta(rel["hash"], temporal)
                        if relation_time_meta is None:
                            continue

                    content = f"{rel['subject']} {rel['predicate']} {rel['object']}"
                    vector_results.append(
                        RetrievalResult(
                            hash_value=rel["hash"],
                            content=content,
                            score=float(score),
                            result_type="relation",
                            source="relation_search (via entity)",
                            metadata={
                                "subject": rel["subject"],
                                "predicate": rel["predicate"],
                                "object": rel["object"],
                                "confidence": rel.get("confidence", 1.0),
                                "pivot_entity": entity_name,
                                "time_meta": relation_time_meta,
                            },
                        )
                    )

            vector_results = self._apply_temporal_filter_to_relations(vector_results, temporal)

        sparse_results: List[RetrievalResult] = []
        if self._should_use_sparse_relations(embedding_ok, vector_results):
            sparse_results = self._search_relations_sparse(query=query, top_k=top_k, temporal=temporal)

        if vector_results and sparse_results:
            results = self._merge_relation_results(vector_results, sparse_results)
        else:
            results = vector_results if vector_results else sparse_results

        return results[:top_k]

    async def _retrieve_dual_path(
        self,
        query: str,
        top_k: int,
        temporal: Optional[TemporalQueryOptions] = None,
    ) -> List[RetrievalResult]:
        """
        双路检索（段落+关系）（异步方法）

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            融合后的检索结果列表
        """
        query_emb = None
        embedding_ok = False
        try:
            query_emb = await self.embedding_manager.encode(query)
            embedding_ok = self._is_valid_embedding(query_emb)
        except Exception as e:
            logger.warning(f"双路检索 embedding 生成失败，将尝试 sparse 回退: {e}")

        para_results: List[RetrievalResult] = []
        rel_results: List[RetrievalResult] = []
        if embedding_ok:
            # 并行检索（使用 asyncio）
            if self.config.enable_parallel:
                para_results, rel_results = await self._parallel_retrieve(query_emb, temporal=temporal)  # type: ignore[arg-type]
            else:
                para_results, rel_results = self._sequential_retrieve(query_emb, temporal=temporal)  # type: ignore[arg-type]
        else:
            logger.warning("embedding 不可用，跳过向量段落/关系召回")

        sparse_para_results: List[RetrievalResult] = []
        if self._should_use_sparse(embedding_ok, para_results):
            sparse_para_results = self._search_paragraphs_sparse(
                query=query,
                top_k=max(top_k * 2, self.config.sparse.candidate_k),
                temporal=temporal,
            )
        sparse_rel_results: List[RetrievalResult] = []
        if self._should_use_sparse_relations(embedding_ok, rel_results):
            sparse_rel_results = self._search_relations_sparse(
                query=query,
                top_k=max(top_k, self.config.sparse.relation_candidate_k),
                temporal=temporal,
            )

        if self.config.fusion.method == "weighted_rrf" and para_results and sparse_para_results:
            para_results = self._fuse_ranked_lists_weighted_rrf(para_results, sparse_para_results)
        elif para_results and sparse_para_results:
            para_results = para_results + sparse_para_results
            para_results.sort(key=lambda x: x.score, reverse=True)
        elif sparse_para_results and (not para_results or not embedding_ok):
            para_results = sparse_para_results

        if rel_results and sparse_rel_results:
            rel_results = self._merge_relation_results(rel_results, sparse_rel_results)
        elif sparse_rel_results and (not rel_results or not embedding_ok):
            rel_results = sparse_rel_results

        # 融合结果
        fused_results = self._fuse_results(
            para_results,
            rel_results,
            query_emb,
        )

        # PageRank重排序
        if self.config.enable_ppr:
            fused_results = await self._rerank_with_ppr(
                fused_results,
                query,
            )

        if temporal:
            fused_results = self._sort_results_with_temporal(fused_results, temporal)

        return fused_results[:top_k]

    async def _parallel_retrieve(
        self,
        query_emb: np.ndarray,
        temporal: Optional[TemporalQueryOptions] = None,
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
                temporal,
            )
            rel_task = asyncio.to_thread(
                self._search_relations,
                query_emb,
                self.config.top_k_relations,
                temporal,
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
        temporal: Optional[TemporalQueryOptions] = None,
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
            temporal,
        )

        rel_results = self._search_relations(
            query_emb,
            self.config.top_k_relations,
            temporal,
        )

        return para_results, rel_results

    def _search_paragraphs(
        self,
        query_emb: np.ndarray,
        top_k: int,
        temporal: Optional[TemporalQueryOptions] = None,
    ) -> List[RetrievalResult]:
        """
        搜索段落

        Args:
            query_emb: 查询嵌入
            top_k: 返回数量

        Returns:
            段落结果列表
        """
        multiplier = max(1, temporal.candidate_multiplier) if temporal else 1
        candidate_k = self._cap_temporal_scan_k(top_k * multiplier, temporal)
        para_ids, para_scores = self.vector_store.search(query_emb, k=candidate_k)

        results = []
        for hash_value, score in zip(para_ids, para_scores):
            paragraph = self.metadata_store.get_paragraph(hash_value)
            if paragraph is None:
                continue

            time_meta = self._build_time_meta_from_paragraph(
                paragraph,
                temporal=temporal,
            )
            results.append(RetrievalResult(
                hash_value=hash_value,
                content=paragraph["content"],
                score=float(score),
                result_type="paragraph",
                source="paragraph_search",
                metadata={
                    "word_count": paragraph.get("word_count", 0),
                    "time_meta": time_meta,
                },
            ))

        return self._apply_temporal_filter_to_paragraphs(results, temporal)

    def _search_relations(
        self,
        query_emb: np.ndarray,
        top_k: int,
        temporal: Optional[TemporalQueryOptions] = None,
    ) -> List[RetrievalResult]:
        """
        搜索关系

        Args:
            query_emb: 查询嵌入
            top_k: 返回数量

        Returns:
            关系结果列表
        """
        multiplier = max(1, temporal.candidate_multiplier) if temporal else 1
        candidate_k = self._cap_temporal_scan_k(top_k * multiplier, temporal)
        rel_ids, rel_scores = self.vector_store.search(query_emb, k=candidate_k)

        results = []
        for hash_value, score in zip(rel_ids, rel_scores):
            relation = self.metadata_store.get_relation(hash_value)
            if relation is None:
                continue

            relation_time_meta = None
            if temporal:
                relation_time_meta = self._best_supporting_time_meta(hash_value, temporal)
                if relation_time_meta is None:
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
                    "time_meta": relation_time_meta,
                },
            ))

        return self._apply_temporal_filter_to_relations(results, temporal)

    def _fuse_results(
        self,
        para_results: List[RetrievalResult],
        rel_results: List[RetrievalResult],
        query_emb: Optional[np.ndarray] = None,
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
            query_emb: 查询嵌入（兼容参数，当前未使用）

        Returns:
            融合后的结果列表
        """
        del query_emb  # 参数保留用于兼容
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

    async def _rerank_with_ppr(
        self,
        results: List[RetrievalResult],
        query: str,
    ) -> List[RetrievalResult]:
        """
        使用PageRank重排序结果 (异步 + 线程池)

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

        # 计算PPR分数 (放入线程池运行，避免阻塞主循环)
        async with self._ppr_semaphore:
            ppr_scores = await asyncio.to_thread(
                self._ppr.compute,
                personalization=entities,
                normalize=True,
            )

        # 调整结果分数
        ppr_scores_by_name = {
            str(name).strip().lower(): float(score)
            for name, score in ppr_scores.items()
        }
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
                        ent_name = str(ent.get("name", "")).strip().lower()
                        if ent_name in ppr_scores_by_name:
                            entity_scores.append(ppr_scores_by_name[ent_name])

                    if entity_scores:
                        avg_ppr = np.mean(entity_scores)
                        # 融合原始分数和PPR分数
                        result.score = result.score * 0.7 + avg_ppr * 0.3

        # 重新排序
        results.sort(key=lambda x: x.score, reverse=True)

        return results

    def _retrieve_temporal_only(
        self,
        temporal: TemporalQueryOptions,
        top_k: int,
    ) -> List[RetrievalResult]:
        """无语义 query 时，直接走时序索引查询。"""
        limit = self._cap_temporal_scan_k(
            top_k * max(1, temporal.candidate_multiplier),
            temporal,
        )
        paragraphs = self.metadata_store.query_paragraphs_temporal(
            start_ts=temporal.time_from,
            end_ts=temporal.time_to,
            person=temporal.person,
            source=temporal.source,
            limit=limit,
            allow_created_fallback=temporal.allow_created_fallback,
        )
        results: List[RetrievalResult] = []
        for para in paragraphs:
            time_meta = self._build_time_meta_from_paragraph(para, temporal=temporal)
            results.append(
                RetrievalResult(
                    hash_value=para["hash"],
                    content=para["content"],
                    score=1.0,
                    result_type="paragraph",
                    source="temporal_scan",
                    metadata={
                        "word_count": para.get("word_count", 0),
                        "time_meta": time_meta,
                    },
                )
            )

        results = self._sort_results_with_temporal(results, temporal)
        return results[:top_k]

    def _extract_effective_time(
        self,
        paragraph: Dict[str, Any],
        temporal: Optional[TemporalQueryOptions] = None,
    ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """提取段落有效时间区间与命中依据。"""
        event_time = paragraph.get("event_time")
        event_start = paragraph.get("event_time_start")
        event_end = paragraph.get("event_time_end")

        if event_start is not None or event_end is not None:
            effective_start = event_start if event_start is not None else (
                event_time if event_time is not None else event_end
            )
            effective_end = event_end if event_end is not None else (
                event_time if event_time is not None else event_start
            )
            return effective_start, effective_end, "event_time_range"

        if event_time is not None:
            return event_time, event_time, "event_time"

        allow_fallback = True
        if temporal is not None:
            allow_fallback = temporal.allow_created_fallback

        created_at = paragraph.get("created_at")
        if allow_fallback and created_at is not None:
            return created_at, created_at, "created_at_fallback"

        return None, None, None

    def _build_time_meta_from_paragraph(
        self,
        paragraph: Dict[str, Any],
        temporal: Optional[TemporalQueryOptions] = None,
    ) -> Dict[str, Any]:
        """构建统一 time_meta 结构。"""
        effective_start, effective_end, match_basis = self._extract_effective_time(
            paragraph,
            temporal=temporal,
        )
        return {
            "event_time": paragraph.get("event_time"),
            "event_time_start": paragraph.get("event_time_start"),
            "event_time_end": paragraph.get("event_time_end"),
            "ingest_time": paragraph.get("created_at"),
            "time_granularity": paragraph.get("time_granularity"),
            "time_confidence": paragraph.get("time_confidence", 1.0),
            "effective_start": effective_start,
            "effective_end": effective_end,
            "effective_start_text": format_timestamp(effective_start),
            "effective_end_text": format_timestamp(effective_end),
            "match_basis": match_basis or "none",
        }

    def _matches_person_filter(self, paragraph_hash: str, person: Optional[str]) -> bool:
        if not person:
            return True
        target = person.strip().lower()
        if not target:
            return True
        para_entities = self.metadata_store.get_paragraph_entities(paragraph_hash)
        for ent in para_entities:
            name = str(ent.get("name", "")).strip().lower()
            if target in name:
                return True
        return False

    def _is_temporal_match(
        self,
        paragraph: Dict[str, Any],
        temporal: TemporalQueryOptions,
    ) -> bool:
        """判断段落是否命中时序筛选。"""
        if temporal.source and paragraph.get("source") != temporal.source:
            return False

        if not self._matches_person_filter(paragraph.get("hash", ""), temporal.person):
            return False

        effective_start, effective_end, _ = self._extract_effective_time(paragraph, temporal=temporal)
        if effective_start is None or effective_end is None:
            return False

        if temporal.time_from is not None and temporal.time_to is not None:
            return effective_end >= temporal.time_from and effective_start <= temporal.time_to
        if temporal.time_from is not None:
            return effective_end >= temporal.time_from
        if temporal.time_to is not None:
            return effective_start <= temporal.time_to
        return True

    def _apply_temporal_filter_to_paragraphs(
        self,
        results: List[RetrievalResult],
        temporal: Optional[TemporalQueryOptions],
    ) -> List[RetrievalResult]:
        if not temporal:
            return results

        filtered: List[RetrievalResult] = []
        for result in results:
            paragraph = self.metadata_store.get_paragraph(result.hash_value)
            if not paragraph:
                continue
            if not self._is_temporal_match(paragraph, temporal):
                continue
            result.metadata["time_meta"] = self._build_time_meta_from_paragraph(paragraph, temporal=temporal)
            filtered.append(result)

        return self._sort_results_with_temporal(filtered, temporal)

    def _best_supporting_time_meta(
        self,
        relation_hash: str,
        temporal: TemporalQueryOptions,
    ) -> Optional[Dict[str, Any]]:
        """获取关系在时序窗口内最优支撑段落的 time_meta。"""
        supports = self.metadata_store.get_paragraphs_by_relation(relation_hash)
        if not supports:
            return None

        best_meta: Optional[Dict[str, Any]] = None
        best_time = float("-inf")
        for para in supports:
            if not self._is_temporal_match(para, temporal):
                continue
            meta = self._build_time_meta_from_paragraph(para, temporal=temporal)
            eff = meta.get("effective_end")
            score = float(eff) if eff is not None else float("-inf")
            if score >= best_time:
                best_time = score
                best_meta = meta

        return best_meta

    def _apply_temporal_filter_to_relations(
        self,
        results: List[RetrievalResult],
        temporal: Optional[TemporalQueryOptions],
    ) -> List[RetrievalResult]:
        if not temporal:
            return results

        filtered: List[RetrievalResult] = []
        for result in results:
            meta = result.metadata.get("time_meta")
            if meta is None:
                meta = self._best_supporting_time_meta(result.hash_value, temporal)
                if meta is None:
                    continue
                result.metadata["time_meta"] = meta
            filtered.append(result)

        return self._sort_results_with_temporal(filtered, temporal)

    def _sort_results_with_temporal(
        self,
        results: List[RetrievalResult],
        temporal: TemporalQueryOptions,
    ) -> List[RetrievalResult]:
        """语义优先，时间次排序（新到旧）。"""
        del temporal  # temporal 保留给未来扩展，目前只使用结果内 time_meta

        def _temporal_key(item: RetrievalResult) -> float:
            time_meta = item.metadata.get("time_meta", {})
            effective = time_meta.get("effective_end")
            if effective is None:
                effective = time_meta.get("effective_start")
            if effective is None:
                return float("-inf")
            return float(effective)

        results.sort(key=lambda x: (x.score, _temporal_key(x)), reverse=True)
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
        vector_size = getattr(self.vector_store, "size", None)
        if vector_size is None:
            vector_size = getattr(self.vector_store, "num_vectors", 0)

        return {
            "config": {
                "top_k_paragraphs": self.config.top_k_paragraphs,
                "top_k_relations": self.config.top_k_relations,
                "top_k_final": self.config.top_k_final,
                "alpha": self.config.alpha,
                "enable_ppr": self.config.enable_ppr,
                "enable_parallel": self.config.enable_parallel,
                "strategy": self.config.retrieval_strategy.value,
                "sparse_mode": self.config.sparse.mode,
                "fusion_method": self.config.fusion.method,
            },
            "vector_store": {
                "size": int(vector_size),
            },
            "graph_store": {
                "num_nodes": self.graph_store.num_nodes,
                "num_edges": self.graph_store.num_edges,
            },
            "metadata_store": self.metadata_store.get_statistics(),
            "sparse": self.sparse_index.stats() if self.sparse_index else None,
        }

    def __repr__(self) -> str:
        return (
            f"DualPathRetriever("
            f"strategy={self.config.retrieval_strategy.value}, "
            f"para_k={self.config.top_k_paragraphs}, "
            f"rel_k={self.config.top_k_relations})"
        )
