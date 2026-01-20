"""
图存储模块

基于SciPy稀疏矩阵的知识图谱存储与计算。
"""

import pickle
from enum import Enum
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict, Set, Any

import numpy as np

class SparseMatrixFormat(Enum):
    """稀疏矩阵格式"""
    CSR = "csr"
    CSC = "csc"

try:
    from scipy.sparse import csr_matrix, csc_matrix, triu, save_npz, load_npz
    from scipy.sparse.linalg import norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

import contextlib
from src.common.logger import get_logger
from ..utils.hash import compute_hash
from ..utils.io import atomic_write

logger = get_logger("A_Memorix.GraphStore")


class GraphModificationMode(Enum):
    """图修改模式"""
    BATCH = "batch"             # 批量模式 (默认, 适合一次性加载)
    INCREMENTAL = "incremental" # 增量模式 (适合频繁随机写入, 使用LIL)
    READ_ONLY = "read_only"     # 只读模式 (适合计算, CSR/CSC)


class GraphStore:
    """
    图存储类

    功能：
    - CSR稀疏矩阵存储图结构
    - 节点和边的CRUD操作
    - Personalized PageRank计算
    - 同义词自动连接
    - 图持久化

    参数：
        matrix_format: 稀疏矩阵格式（csr/csc）
        data_dir: 数据目录
    """

    def __init__(
        self,
        matrix_format: str = "csr",
        data_dir: Optional[Union[str, Path]] = None,
    ):
        """
        初始化图存储

        Args:
            matrix_format: 稀疏矩阵格式（csr/csc）
            data_dir: 数据目录
        """
        if not HAS_SCIPY:
            raise ImportError("SciPy 未安装，请安装: pip install scipy")

        if isinstance(matrix_format, SparseMatrixFormat):
            self.matrix_format = matrix_format.value
        else:
            self.matrix_format = str(matrix_format).lower()
        self.data_dir = Path(data_dir) if data_dir else None

        # 节点管理
        self._nodes: List[str] = []  # 节点列表
        self._node_to_idx: Dict[str, int] = {}  # 节点名到索引的映射
        self._node_attrs: Dict[str, Dict[str, Any]] = {}  # 节点属性

        # 边管理（邻接矩阵）
        self._adjacency: Optional[Union[csr_matrix, csc_matrix]] = None

        # 统计信息
        self._total_nodes_added = 0
        self._total_edges_added = 0
        self._total_nodes_deleted = 0
        self._total_edges_deleted = 0
        
        # 状态管理
        self._modification_mode = GraphModificationMode.BATCH

        logger.info(f"GraphStore 初始化: format={matrix_format}")
        
    @contextlib.contextmanager
    def batch_update(self):
        """
        批量更新上下文管理器
        
        进入时切换到 LIL 格式以优化随机/增量更新
        退出时恢复到 CSR/CSC 格式以优化存储和计算
        """
        original_mode = self._modification_mode
        self._switch_mode(GraphModificationMode.INCREMENTAL)
        try:
            yield
        finally:
            self._switch_mode(original_mode)
            
    def _switch_mode(self, new_mode: GraphModificationMode):
        """切换修改模式并转换矩阵格式"""
        if new_mode == self._modification_mode:
            return
            
        if self._adjacency is None:
            self._modification_mode = new_mode
            return

        logger.debug(f"切换图模式: {self._modification_mode.value} -> {new_mode.value}")
        
        # 转换逻辑
        if new_mode == GraphModificationMode.INCREMENTAL:
            # 转换为 LIL
            if not isinstance(self._adjacency, (list, dict)): # crudely check if not lil
                 try:
                     self._adjacency = self._adjacency.tolil()
                     logger.debug("已转换为 LIL 格式")
                 except Exception as e:
                     logger.warning(f"转换为 LIL 失败: {e}")
        
        elif new_mode in [GraphModificationMode.BATCH, GraphModificationMode.READ_ONLY]:
            # 转换回配置的格式 (CSR/CSC)
            if self.matrix_format == "csr":
                self._adjacency = self._adjacency.tocsr()
            elif self.matrix_format == "csc":
                self._adjacency = self._adjacency.tocsc()
            logger.debug(f"已恢复为 {self.matrix_format.upper()} 格式")
            
        self._modification_mode = new_mode

    def add_nodes(
        self,
        nodes: List[str],
        attributes: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> int:
        """
        添加节点

        Args:
            nodes: 节点名称列表
            attributes: 节点属性字典 {node_name: {attr: value}}

        Returns:
            成功添加的节点数量
        """
        added = 0
        for node in nodes:
            if node in self._node_to_idx:
                logger.debug(f"节点已存在，跳过: {node}")
                continue

            # 添加到节点列表
            idx = len(self._nodes)
            self._nodes.append(node)
            self._node_to_idx[node] = idx

            # 添加属性
            if attributes and node in attributes:
                self._node_attrs[node] = attributes[node]
            else:
                self._node_attrs[node] = {}

            added += 1
            self._total_nodes_added += 1

        # 扩展邻接矩阵
        if added > 0:
            self._expand_adjacency_matrix(added)

        logger.debug(f"添加 {added} 个节点")
        return added

    def add_edges(
        self,
        edges: List[Tuple[str, str]],
        weights: Optional[List[float]] = None,
    ) -> int:
        """
        添加边

        Args:
            edges: 边列表 [(source, target), ...]
            weights: 边权重列表（默认为1.0）

        Returns:
            成功添加的边数量
        """
        if not edges:
            return 0

        # 确保所有节点存在
        nodes_to_add = set()
        for src, tgt in edges:
            if src not in self._node_to_idx:
                nodes_to_add.add(src)
            if tgt not in self._node_to_idx:
                nodes_to_add.add(tgt)

        if nodes_to_add:
            self.add_nodes(list(nodes_to_add))

        # 处理权重
        if weights is None:
            weights = [1.0] * len(edges)

        if len(weights) != len(edges):
            raise ValueError(f"边数量与权重数量不匹配: {len(edges)} vs {len(weights)}")

        # 如果仅仅是添加边且处于增量模式 (LIL)，直接更新
        if self._modification_mode == GraphModificationMode.INCREMENTAL:
             if self._adjacency is None:
                 # 初始化为空 LIL
                 n = len(self._nodes)
                 from scipy.sparse import lil_matrix
                 self._adjacency = lil_matrix((n, n), dtype=np.float32)

             # 尝试直接使用 LIL 索引更新
             try:
                 # 批量获取索引
                 rows = [self._node_to_idx[src] for src, _ in edges]
                 cols = [self._node_to_idx[tgt] for _, tgt in edges]
                 
                 # 确保矩阵足够大 (如果 add_nodes 没有扩展它) - 通常 add_nodes 会处理
                 # 这里直接赋值
                 self._adjacency[rows, cols] = weights
                 
                 self._total_edges_added += len(edges)
                 logger.debug(f"增量添加 {len(edges)} 条边 (LIL)")
                 return len(edges)
             except Exception as e:
                 logger.warning(f"LIL 增量更新失败，回退到通用方法: {e}")
                 # Fallback to general method below

        # 通用方法 (构建 COO 然后合并)
        # 构建边的三元组
        row_indices = []
        col_indices = []
        data_values = []

        for (src, tgt), weight in zip(edges, weights):
            src_idx = self._node_to_idx[src]
            tgt_idx = self._node_to_idx[tgt]

            row_indices.append(src_idx)
            col_indices.append(tgt_idx)
            data_values.append(weight)

        # 创建新的边的矩阵
        n = len(self._nodes)
        new_edges = csr_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(n, n),
        )

        # 合并到邻接矩阵
        if self._adjacency is None:
            self._adjacency = new_edges
        else:
            self._adjacency = self._adjacency + new_edges

        # 转换为指定格式
        if self.matrix_format == "csc" and isinstance(self._adjacency, csr_matrix):
            self._adjacency = self._adjacency.tocsc()
        elif self.matrix_format == "csr" and isinstance(self._adjacency, csc_matrix):
            self._adjacency = self._adjacency.tocsr()

        self._total_edges_added += len(edges)
        logger.debug(f"添加 {len(edges)} 条边")
        return len(edges)

    def update_edge_weight(
        self,
        source: str,
        target: str,
        delta: float,
        min_weight: float = 0.1,
        max_weight: float = 10.0,
    ) -> float:
        """
        更新边权重 (增量/强化/弱化)

        Args:
            source: 源节点
            target: 目标节点
            delta: 权重变化量 (+/-)
            min_weight: 最小权重限制
            max_weight: 最大权重限制

        Returns:
            更新后的权重
        """
        if source not in self._node_to_idx or target not in self._node_to_idx:
            logger.warning(f"节点不存在，无法更新权重: {source} -> {target}")
            return 0.0

        current_weight = self.get_edge_weight(source, target)
        if current_weight == 0.0 and delta <= 0:
            # 边不存在且试图减少权重，忽略
            return 0.0
        
        # 如果边不存在但 delta > 0，相当于添加新边 (默认基础权重0 + delta)
        # 但为了逻辑清晰，我们假设只更新存在的边，或者确实添加
        
        new_weight = current_weight + delta
        new_weight = max(min_weight, min(max_weight, new_weight))
        
        # 使用 batch_update 上下文自动处理格式转换
        # 这里我们临时切换到 incremental 模式进行单次更新
        with self.batch_update():
            # add_edges 会覆盖或添加，我们需要覆盖
            self.add_edges([(source, target)], [new_weight])
            
        logger.debug(f"更新权重 {source}->{target}: {current_weight:.2f} -> {new_weight:.2f}")
        return new_weight

    def delete_nodes(self, nodes: List[str]) -> int:
        """
        删除节点（及相关的边）

        Args:
            nodes: 要删除的节点列表

        Returns:
            成功删除的节点数量
        """
        if not nodes:
            return 0

        # 检查哪些节点存在
        existing_nodes = [node for node in nodes if node in self._node_to_idx]
        if not existing_nodes:
            logger.warning("所有节点都不存在，无法删除")
            return 0

        # 获取要删除的索引
        indices_to_delete = [self._node_to_idx[node] for node in existing_nodes]
        indices_to_keep = [
            i for i in range(len(self._nodes))
            if i not in indices_to_delete
        ]

        # 创建索引映射
        old_to_new = {old: new for new, old in enumerate(indices_to_keep)}

        # 重建节点列表
        self._nodes = [self._nodes[i] for i in indices_to_keep]
        self._node_to_idx = {node: old_to_new[idx] for node, idx in self._node_to_idx.items() if idx in old_to_new}

        # 删除节点属性
        for node in existing_nodes:
            self._node_attrs.pop(node, None)

        # 重建邻接矩阵
        if self._adjacency is not None:
            self._adjacency = self._adjacency[indices_to_keep, :][:, indices_to_keep]

        deleted_count = len(existing_nodes)
        self._total_nodes_deleted += deleted_count

        logger.info(f"删除 {deleted_count} 个节点")
        return deleted_count

    def remove_nodes(self, nodes: List[str]) -> int:
        """兼容性别名：删除节点"""
        return self.delete_nodes(nodes)

    def delete_edges(
        self,
        edges: List[Tuple[str, str]],
    ) -> int:
        """
        删除边

        Args:
            edges: 要删除的边列表 [(source, target), ...]

        Returns:
            成功删除的边数量
        """
        if not edges or self._adjacency is None:
            return 0

        deleted = 0
        # 转换为COO格式便于修改
        adj_coo = self._adjacency.tocoo()

        # 构建要删除的边的索引集合
        edges_to_delete = set()
        for src, tgt in edges:
            if src in self._node_to_idx and tgt in self._node_to_idx:
                src_idx = self._node_to_idx[src]
                tgt_idx = self._node_to_idx[tgt]
                edges_to_delete.add((src_idx, tgt_idx))

        # 过滤要删除的边
        new_row = []
        new_col = []
        new_data = []

        for i, j, val in zip(adj_coo.row, adj_coo.col, adj_coo.data):
            if (i, j) not in edges_to_delete:
                new_row.append(i)
                new_col.append(j)
                new_data.append(val)
            else:
                deleted += 1

        # 重建邻接矩阵
        n = len(self._nodes)
        self._adjacency = csr_matrix((new_data, (new_row, new_col)), shape=(n, n))

        # 转换回指定格式
        if self.matrix_format == "csc":
            self._adjacency = self._adjacency.tocsc()

        self._total_edges_deleted += deleted
        logger.info(f"删除 {deleted} 条边")
        return deleted

    def remove_edges(self, edges: List[Tuple[str, str]]) -> int:
        """兼容性别名：删除边"""
        return self.delete_edges(edges)

    def get_nodes(self) -> List[str]:
        """
        获取所有节点

        Returns:
            节点列表
        """
        return self._nodes.copy()

    def has_node(self, node: str) -> bool:
        """
        检查节点是否存在

        Args:
            node: 节点名称

        Returns:
            节点是否存在
        """
        return node in self._node_to_idx

    def get_node_attributes(self, node: str) -> Optional[Dict[str, Any]]:
        """
        获取节点属性

        Args:
            node: 节点名称

        Returns:
            节点属性字典，不存在则返回None
        """
        return self._node_attrs.get(node)

    def get_neighbors(self, node: str) -> List[str]:
        """
        获取节点的邻居

        Args:
            node: 节点名称

        Returns:
            邻居节点列表
        """
        if node not in self._node_to_idx or self._adjacency is None:
            return []

        idx = self._node_to_idx[node]

        # 获取邻接行
        if self.matrix_format == "csr":
            row = self._adjacency.getrow(idx).toarray().flatten()
        else:
            row = self._adjacency[:, idx].toarray().flatten()

        # 找非零元素
        neighbor_indices = np.where(row > 0)[0]
        neighbors = [self._nodes[i] for i in neighbor_indices]

        return neighbors

    def get_edge_weight(self, source: str, target: str) -> float:
        """
        获取边的权重

        Args:
            source: 源节点
            target: 目标节点

        Returns:
            边权重，不存在则返回0.0
        """
        if source not in self._node_to_idx or target not in self._node_to_idx:
            return 0.0

        if self._adjacency is None:
            return 0.0

        src_idx = self._node_to_idx[source]
        tgt_idx = self._node_to_idx[target]

        return float(self._adjacency[src_idx, tgt_idx])

    def compute_pagerank(
        self,
        personalization: Optional[Dict[str, float]] = None,
        alpha: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> Dict[str, float]:
        """
        计算Personalized PageRank

        Args:
            personalization: 个性化向量 {node: weight}，默认为均匀分布
            alpha: 阻尼系数（0-1之间）
            max_iter: 最大迭代次数
            tol: 收敛阈值

        Returns:
            节点PageRank值字典 {node: score}
        """
        if self._adjacency is None or len(self._nodes) == 0:
            logger.warning("图为空，无法计算PageRank")
            return {}

        n = len(self._nodes)

        # 构建列归一化的转移矩阵
        adj = self._adjacency.astype(np.float32)

        # 计算出度
        out_degrees = np.array(adj.sum(axis=1)).flatten()

        # 处理悬挂节点（出度为0）
        dangling = (out_degrees == 0)
        out_degrees[dangling] = 1.0  # 避免除零

        # 归一化
        D_inv = csr_matrix(np.diag(1.0 / out_degrees))
        M = adj.T @ D_inv  # 转移矩阵

        # 初始化个性化向量
        if personalization is None:
            # 均匀分布
            p = np.ones(n) / n
        else:
            # 使用指定的个性化向量
            p = np.zeros(n)
            total_weight = sum(personalization.values())
            for node, weight in personalization.items():
                if node in self._node_to_idx:
                    idx = self._node_to_idx[node]
                    p[idx] = weight / total_weight

            # 确保和为1
            if p.sum() == 0:
                p = np.ones(n) / n
            else:
                p = p / p.sum()

        # 幂迭代法
        for i in range(max_iter):
            p_new = (1 - alpha) * p + alpha * (M @ p)

            # 检查收敛
            diff = np.linalg.norm(p_new - p, 1)
            if diff < tol:
                logger.debug(f"PageRank在 {i+1} 次迭代后收敛")
                p = p_new
                break

            p = p_new
        else:
            logger.warning(f"PageRank未在 {max_iter} 次迭代内收敛")

        # 转换为字典
        scores = {node: float(p[idx]) for node, idx in self._node_to_idx.items()}

        return scores

    def connect_synonyms(
        self,
        similarity_matrix: np.ndarray,
        node_list: List[str],
        threshold: float = 0.85,
    ) -> int:
        """
        连接相似节点（同义词）

        Args:
            similarity_matrix: 相似度矩阵 (N x N)
            node_list: 对应的节点列表（长度为N）
            threshold: 相似度阈值

        Returns:
            添加的边数量
        """
        if len(node_list) != similarity_matrix.shape[0]:
            raise ValueError(
                f"节点列表长度与相似度矩阵维度不匹配: "
                f"{len(node_list)} vs {similarity_matrix.shape[0]}"
            )

        # 找到相似的节点对（上三角，排除对角线）
        similar_pairs = np.argwhere(
            (triu(similarity_matrix, k=1) >= threshold) &
            (triu(similarity_matrix, k=1) < 1.0)  # 排除完全相同的
        )

        # 添加边
        edges = []
        for i, j in similar_pairs:
            if i < len(node_list) and j < len(node_list):
                src = node_list[i]
                tgt = node_list[j]
                # 使用相似度作为权重
                weight = float(similarity_matrix[i, j])
                edges.append((src, tgt, weight))

        if edges:
            edge_pairs = [(src, tgt) for src, tgt, _ in edges]
            weights = [w for _, _, w in edges]
            count = self.add_edges(edge_pairs, weights)
            logger.info(f"连接 {count} 对相似节点（阈值={threshold}）")
            return count
        return 0

    def clear(self) -> None:
        """清空所有数据"""
        self._nodes.clear()
        self._node_to_idx.clear()
        self._node_attrs.clear()
        self._adjacency = None
        self._total_nodes_added = 0
        self._total_edges_added = 0
        self._total_nodes_deleted = 0
        self._total_edges_deleted = 0
        logger.info("图存储已清空")

    def save(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        """
        保存到磁盘

        Args:
            data_dir: 数据目录（默认使用初始化时的目录）
        """
        if data_dir is None:
            data_dir = self.data_dir

        if data_dir is None:
            raise ValueError("未指定数据目录")

        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        # 保存邻接矩阵
        if self._adjacency is not None:
            matrix_path = data_dir / "graph_adjacency.npz"
            with atomic_write(matrix_path, "wb") as f:
                save_npz(f, self._adjacency)
            logger.debug(f"保存邻接矩阵: {matrix_path}")

        # 保存元数据
        metadata = {
            "nodes": self._nodes,
            "node_to_idx": self._node_to_idx,
            "node_attrs": self._node_attrs,
            "matrix_format": self.matrix_format,
            "total_nodes_added": self._total_nodes_added,
            "total_edges_added": self._total_edges_added,
            "total_nodes_deleted": self._total_nodes_deleted,
            "total_edges_deleted": self._total_edges_deleted,
        }

        metadata_path = data_dir / "graph_metadata.pkl"
        with atomic_write(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        logger.debug(f"保存元数据: {metadata_path}")

        logger.info(f"图存储已保存到: {data_dir}")

    def load(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        """
        从磁盘加载

        Args:
            data_dir: 数据目录（默认使用初始化时的目录）
        """
        if data_dir is None:
            data_dir = self.data_dir

        if data_dir is None:
            raise ValueError("未指定数据目录")

        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        # 加载元数据
        metadata_path = data_dir / "graph_metadata.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")

        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        # 恢复状态
        self._nodes = metadata["nodes"]
        self._node_to_idx = metadata["node_to_idx"]
        self._node_attrs = metadata["node_attrs"]
        self.matrix_format = metadata["matrix_format"]
        self._total_nodes_added = metadata["total_nodes_added"]
        self._total_edges_added = metadata["total_edges_added"]
        self._total_nodes_deleted = metadata["total_nodes_deleted"]
        self._total_edges_deleted = metadata["total_edges_deleted"]

        # 加载邻接矩阵
        matrix_path = data_dir / "graph_adjacency.npz"
        if matrix_path.exists():
            self._adjacency = load_npz(str(matrix_path))

            # 确保格式正确
            if self.matrix_format == "csc" and isinstance(self._adjacency, csr_matrix):
                self._adjacency = self._adjacency.tocsc()
            elif self.matrix_format == "csr" and isinstance(self._adjacency, csc_matrix):
                self._adjacency = self._adjacency.tocsr()

            logger.debug(f"加载邻接矩阵: {matrix_path}, shape={self._adjacency.shape}")

        logger.info(
            f"图存储已加载: {len(self._nodes)} 个节点, "
            f"{self._adjacency.nnz if self._adjacency is not None else 0} 条边"
        )

    def _expand_adjacency_matrix(self, added_nodes: int) -> None:
        """
        扩展邻接矩阵以容纳新节点

        Args:
            added_nodes: 新增节点数量
        """
        if self._adjacency is None:
            n = len(self._nodes)
            self._adjacency = csr_matrix((n, n), dtype=np.float32)
            return

        old_n = self._adjacency.shape[0]
        new_n = old_n + added_nodes

        # 创建扩展后的矩阵
        if self.matrix_format == "csr":
            new_adjacency = csr_matrix((new_n, new_n), dtype=np.float32)
            new_adjacency[:old_n, :old_n] = self._adjacency
        else:
            new_adjacency = csc_matrix((new_n, new_n), dtype=np.float32)
            new_adjacency[:old_n, :old_n] = self._adjacency

        self._adjacency = new_adjacency
        
        # 如果都在增量模式，确保是LIL
        if self._modification_mode == GraphModificationMode.INCREMENTAL:
             try:
                 self._adjacency = self._adjacency.tolil()
             except:
                 pass

    @property
    def num_nodes(self) -> int:
        """节点数量"""
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        """边数量"""
        if self._adjacency is None:
            return 0
        return int(self._adjacency.nnz)

    @property
    def density(self) -> float:
        """
        图密度（实际边数 / 可能的最大边数）

        有向图: E / (V * (V - 1))
        无向图: 2E / (V * (V - 1))

        这里按有向图计算
        """
        if self.num_nodes < 2:
            return 0.0

        max_edges = self.num_nodes * (self.num_nodes - 1)
        return self.num_edges / max_edges if max_edges > 0 else 0.0

    def __len__(self) -> int:
        """节点数量"""
        return self.num_nodes

    def has_data(self) -> bool:
        """检查磁盘上是否存在现有数据"""
        if self.data_dir is None:
            return False
        return (self.data_dir / "graph_metadata.pkl").exists()

    def __repr__(self) -> str:
        return (
            f"GraphStore(nodes={self.num_nodes}, edges={self.num_edges}, "
            f"density={self.density:.4f}, format={self.matrix_format})"
        )
