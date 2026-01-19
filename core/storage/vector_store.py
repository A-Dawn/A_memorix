"""
向量存储模块

基于Faiss的高效向量存储与检索，支持int8量化和内存映射。
"""

import os
import pickle
import shutil
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict, Any

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from src.common.logger import get_logger
from ..utils.quantization import QuantizationType

logger = get_logger("A_Memorix.VectorStore")


class VectorStore:
    """
    向量存储类

    功能：
    - Faiss索引（IndexFlatIP / IndexIVFFlat）
    - int8标量量化（可选）
    - 内存映射存储
    - 标记删除 + 定期重建
    - 增量缓冲区
    - 持久化

    参数：
        dimension: 向量维度
        quantization_type: 量化类型（float32/int8）
        index_type: Faiss索引类型（flat/ivf）
        data_dir: 数据目录
        use_mmap: 是否使用内存映射
        buffer_size: 缓冲区大小
    """

    def __init__(
        self,
        dimension: int,
        quantization_type: QuantizationType = QuantizationType.INT8,
        index_type: str = "flat",
        data_dir: Optional[Union[str, Path]] = None,
        use_mmap: bool = True,
        buffer_size: int = 1000,
    ):
        """
        初始化向量存储

        Args:
            dimension: 向量维度
            quantization_type: 量化类型
            index_type: 索引类型（flat/ivf）
            data_dir: 数据目录
            use_mmap: 是否使用内存映射
            buffer_size: 缓冲区大小
        """
        if not HAS_FAISS:
            raise ImportError("Faiss 未安装，请安装: pip install faiss-cpu")

        self.dimension = dimension
        self.quantization_type = quantization_type
        self.index_type = index_type.lower()
        self.data_dir = Path(data_dir) if data_dir else None
        self.use_mmap = use_mmap
        self.buffer_size = buffer_size

        # 内部状态
        self._index: Optional[faiss.Index] = None
        self._vectors: Optional[np.ndarray] = None
        self._ids: List[str] = []
        self._id_to_idx: Dict[str, int] = {}
        self._deleted_ids: set = set()
        self._buffer_ids: List[str] = []
        self._buffer_vectors: List[np.ndarray] = []

        # 量化参数
        self._quant_min: Optional[float] = None
        self._quant_max: Optional[float] = None

        # 统计信息
        self._total_added = 0
        self._total_deleted = 0

        logger.info(
            f"VectorStore 初始化: dim={dimension}, quant={quantization_type.value}, "
            f"index={index_type}"
        )

    def add(self, vectors: np.ndarray, ids: List[str]) -> int:
        """
        添加向量

        Args:
            vectors: 向量数组 (N x D)
            ids: 向量ID列表

        Returns:
            成功添加的向量数量

        Raises:
            ValueError: 向量维度不匹配或ID重复
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"向量维度不匹配: 期望 {self.dimension}, 实际 {vectors.shape[1]}"
            )

        if len(vectors) != len(ids):
            raise ValueError(f"向量数量与ID数量不匹配: {len(vectors)} vs {len(ids)}")

        # 检查ID重复
        duplicate_ids = set(ids) & set(self._ids)
        if duplicate_ids:
            raise ValueError(f"ID已存在: {duplicate_ids}")

        # 量化向量
        if self.quantization_type == QuantizationType.INT8:
            vectors, params = self._quantize_with_params(vectors)
            # 保存量化参数（首次添加时）
            if self._quant_min is None:
                self._quant_min = params["min"]
                self._quant_max = params["max"]

        # 添加到缓冲区
        self._buffer_ids.extend(ids)
        self._buffer_vectors.append(vectors)

        # 检查是否需要flush
        if len(self._buffer_ids) >= self.buffer_size:
            self._flush_buffer()

        self._total_added += len(ids)
        logger.debug(f"添加 {len(ids)} 个向量到缓冲区")
        return len(ids)

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        filter_deleted: bool = True,
    ) -> Tuple[List[str], List[float]]:
        """
        搜索最相似的向量

        Args:
            query: 查询向量 (D,) 或 (N x D)
            k: 返回结果数量
            filter_deleted: 是否过滤已删除的向量

        Returns:
            (ID列表, 分数列表)
        """
        # 检查是否有数据（索引或缓冲区）
        has_index = self._index is not None and self._index.ntotal > 0
        has_buffer = len(self._buffer_ids) > 0
        
        if not has_index and not has_buffer:
            logger.warning("索引和缓冲区均为空，无法搜索")
            return [], []

        # 确保查询是2D数组
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # 量化查询（如果需要）
        if self.quantization_type == QuantizationType.INT8:
            query, _ = self._quantize_with_params(query)

        # 搜索数量（考虑过滤）
        search_k = k * 3 if filter_deleted else k

        # 1. 在主索引中搜索
        distances = np.array([])
        indices = np.array([])
        
        if has_index:
            try:
                distances, indices = self._index.search(query.astype(np.float32), search_k)
                distances = distances[0]
                indices = indices[0]
            except Exception as e:
                logger.error(f"主索引搜索失败: {e}")

        # 2. 在缓冲区中搜索（线性）
        # 即使主索引搜索失败，也尝试搜索缓冲区
        buffer_distances, buffer_indices = self._search_buffer(query, k)

        # 3. 合并结果
        results = self._merge_search_results(
            distances, indices, buffer_distances, buffer_indices, k
        )

        # 过滤已删除的
        if filter_deleted:
            results = [(id_, score) for id_, score in results if id_ not in self._deleted_ids]

        # 分离ID和分数
        if not results:
            return [], []

        ids, scores = zip(*results)
        return list(ids), list(scores)

    def delete(self, ids: List[str]) -> int:
        """
        删除向量（标记删除）

        Args:
            ids: 要删除的ID列表

        Returns:
            成功删除的数量
        """
        deleted = 0
        for id_ in ids:
            if id_ in self._id_to_idx and id_ not in self._deleted_ids:
                self._deleted_ids.add(id_)
                deleted += 1
                self._total_deleted += 1

        logger.info(f"标记删除 {deleted} 个向量（累计: {len(self._deleted_ids)}）")

        # 检查是否需要重建索引
        self._check_rebuild_needed()

        return deleted

    def remove(self, ids: List[str]) -> int:
        """兼容性别名：删除向量"""
        return self.delete(ids)

    def get(self, ids: List[str]) -> List[Optional[np.ndarray]]:
        """
        获取向量

        Args:
            ids: ID列表

        Returns:
            向量列表（不存在的ID对应None）
        """
        vectors = []
        for id_ in ids:
            if id_ in self._deleted_ids:
                vectors.append(None)
                continue

            # 先在缓冲区查找
            if id_ in self._buffer_ids:
                total_idx = self._buffer_ids.index(id_)
                # 遍历缓冲区找到对应的向量
                current_base = 0
                for buf_arr in self._buffer_vectors:
                    if current_base + len(buf_arr) > total_idx:
                        vec = buf_arr[total_idx - current_base]
                        vectors.append(self._dequantize_vector(vec))
                        break
                    current_base += len(buf_arr)
                continue

            # 在主存储查找
            if id_ in self._id_to_idx:
                idx = self._id_to_idx[id_]
                vec = self._vectors[idx].copy()
                vectors.append(self._dequantize_vector(vec))
            else:
                vectors.append(None)

        return vectors

    def rebuild_index(self) -> None:
        """
        重建索引（删除已标记的向量）

        操作：
        1. 过滤已删除的向量
        2. 重新构建索引
        3. 清空删除标记
        """
        if not self._deleted_ids:
            logger.info("没有需要删除的向量，跳过重建")
            return

        logger.info(f"开始重建索引，当前向量数: {len(self._ids)}")

        # 过滤向量
        valid_indices = [
            idx for idx, id_ in enumerate(self._ids) if id_ not in self._deleted_ids
        ]

        if not valid_indices:
            logger.warning("所有向量都被删除，清空索引")
            self._ids.clear()
            self._id_to_idx.clear()
            self._vectors = None
            self._index = None
            self._deleted_ids.clear()
            return

        # 重建数据
        new_vectors = self._vectors[valid_indices]
        new_ids = [self._ids[i] for i in valid_indices]

        # 重建索引
        self._ids = new_ids
        self._id_to_idx = {id_: idx for idx, id_ in enumerate(new_ids)}
        self._vectors = new_vectors
        self._build_index(self._vectors)

        # 清空删除标记
        deleted_count = len(self._deleted_ids)
        self._deleted_ids.clear()

        logger.info(
            f"索引重建完成: 删除 {deleted_count} 个向量，剩余 {len(new_ids)} 个"
        )

    def clear(self) -> None:
        """清空所有数据"""
        self._ids.clear()
        self._id_to_idx.clear()
        self._vectors = None
        self._index = None
        self._deleted_ids.clear()
        self._buffer_ids.clear()
        self._buffer_vectors.clear()
        self._total_added = 0
        self._total_deleted = 0
        logger.info("向量存储已清空")

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

        # 先flush缓冲区
        self._flush_buffer()

        # 保存索引
        if self._index is not None:
            index_path = data_dir / "vectors.index"
            faiss.write_index(self._index, str(index_path))
            logger.debug(f"保存索引: {index_path}")

        # 保存向量（如果使用内存映射）
        if self._vectors is not None:
            vectors_path = data_dir / "vectors.npy"
            
            # Windows 兼容性处理：如果向量数据是当前文件的只读内存映射，
            # 尝试写入同名文件会触发 [Errno 22] Invalid argument。
            # 仅当数据已修改（不再是 memmap）或是不同路径时才执行保存。
            is_same_file_mmap = False
            if isinstance(self._vectors, np.memmap):
                try:
                    # 检查是否指向同一个物理文件
                    if os.path.abspath(self._vectors.filename) == os.path.abspath(str(vectors_path)):
                        is_same_file_mmap = True
                except Exception:
                    pass
            
            if not is_same_file_mmap:
                try:
                    # 使用 str() 转换路径以提高跨版本兼容性
                    if self.use_mmap:
                        np.save(str(vectors_path), self._vectors)
                    else:
                        np.save(str(vectors_path), self._vectors, allow_pickle=False)
                    logger.debug(f"保存向量: {vectors_path}")
                except Exception as e:
                    # 如果是因为锁定（Errno 22/13），在 Windows 上尝试优雅处理
                    if "Errno 22" in str(e) or "Errno 13" in str(e):
                        logger.warning(f"保存向量文件受阻 (通常由于文件处于内存映射状态): {e}。数据将保留在内存中，在关闭或下次尝试时重试。")
                    else:
                        logger.error(f"保存向量文件失败: {e}")
                        raise
            else:
                logger.debug("数据为原始内存映射且无变化，跳过 vectors.npy 写入以避开 Windows 锁定")

        # 保存元数据
        metadata = {
            "dimension": self.dimension,
            "quantization_type": self.quantization_type.value,
            "index_type": self.index_type,
            "ids": self._ids,
            "deleted_ids": list(self._deleted_ids),
            "total_added": self._total_added,
            "total_deleted": self._total_deleted,
            "quant_min": self._quant_min,
            "quant_max": self._quant_max,
        }

        metadata_path = data_dir / "vectors_metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        logger.debug(f"保存元数据: {metadata_path}")

        logger.info(f"向量存储已保存到: {data_dir}")

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
        metadata_path = data_dir / "vectors_metadata.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")

        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        # 恢复状态
        self.dimension = metadata["dimension"]
        self.quantization_type = QuantizationType(metadata["quantization_type"])
        self.index_type = metadata["index_type"]
        self._ids = metadata["ids"]
        self._deleted_ids = set(metadata["deleted_ids"])
        self._total_added = metadata["total_added"]
        self._total_deleted = metadata["total_deleted"]
        self._quant_min = metadata.get("quant_min")
        self._quant_max = metadata.get("quant_max")

        # 重建ID映射
        self._id_to_idx = {id_: idx for idx, id_ in enumerate(self._ids)}

        # 加载向量
        vectors_path = data_dir / "vectors.npy"
        if vectors_path.exists():
            if self.use_mmap:
                self._vectors = np.load(vectors_path, mmap_mode="r")
            else:
                self._vectors = np.load(vectors_path)
            logger.debug(f"加载向量: {vectors_path}, shape={self._vectors.shape}")

        # 加载索引
        index_path = data_dir / "vectors.index"
        if index_path.exists():
            self._index = faiss.read_index(str(index_path))
            logger.debug(f"加载索引: {index_path}, ntotal={self._index.ntotal}")

        logger.info(
            f"向量存储已加载: {len(self._ids)} 个向量, "
            f"{len(self._deleted_ids)} 个已删除"
        )

    def _quantize_with_params(
        self, vectors: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        量化向量并返回参数

        Args:
            vectors: 输入向量

        Returns:
            (量化后的向量, 参数字典)
        """
        if self.quantization_type == QuantizationType.FLOAT32:
            return vectors.astype(np.float32), {}

        # int8量化
        min_val = np.min(vectors)
        max_val = np.max(vectors)

        if max_val == min_val:
            return np.zeros_like(vectors, dtype=np.int8), {"min": 0.0, "max": 0.0}

        # 归一化到 [0, 255]
        normalized = (vectors - min_val) / (max_val - min_val) * 255
        quantized = np.round(normalized).astype(np.int8)

        return quantized, {"min": float(min_val), "max": float(max_val)}

    def _dequantize_vector(self, vectors: np.ndarray) -> np.ndarray:
        """
        反量化向量

        Args:
            vectors: 量化后的向量

        Returns:
            反量化后的向量
        """
        if self.quantization_type == QuantizationType.FLOAT32:
            return vectors

        # int8反量化
        if self._quant_min is None or self._quant_max is None:
            # 默认范围 [0, 255] -> [-1, 1]
            return (vectors.astype(np.float32) + 128.0) / 255.0 * 2.0 - 1.0

        min_val = self._quant_min
        max_val = self._quant_max
        # 将 np.int8 [-128, 127] 映射回 [0, 255]
        normalized = (vectors.astype(np.float32) + 128.0) / 255.0
        return normalized * (max_val - min_val) + min_val

    def _build_index(self, vectors: np.ndarray) -> None:
        """
        构建Faiss索引

        Args:
            vectors: 向量数组
        """
        if self.index_type == "flat":
            # 精确搜索（内积）
            self._index = faiss.IndexFlatIP(self.dimension)
            self._index.add(vectors.astype(np.float32))

        elif self.index_type == "ivf":
            # IVF索引（倒排文件）
            nlist = min(100, len(vectors) // 10)  # 聚类中心数量
            if nlist < 1:
                nlist = 1

            quantizer = faiss.IndexFlatIP(self.dimension)
            self._index = faiss.IndexIVFFlat(
                quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT
            )

            # 训练
            if not self._index.is_trained:
                self._index.train(vectors.astype(np.float32))

            # 添加向量
            self._index.add(vectors.astype(np.float32))

        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")

        logger.info(
            f"构建 {self.index_type} 索引完成: {self._index.ntotal} 个向量"
        )

    def _flush_buffer(self) -> None:
        """刷新缓冲区到主存储"""
        if not self._buffer_ids:
            return

        logger.debug(f"刷新缓冲区: {len(self._buffer_ids)} 个向量")

        # 合并缓冲区向量
        if self._buffer_vectors:
            all_vectors = np.concatenate(self._buffer_vectors, axis=0)
        else:
            all_vectors = np.array([]).reshape(0, self.dimension)

        # 扩展主存储
        if self._vectors is None:
            self._vectors = all_vectors
        else:
            self._vectors = np.concatenate([self._vectors, all_vectors], axis=0)

        # 扩展ID列表
        start_idx = len(self._ids)
        self._ids.extend(self._buffer_ids)
        for idx, id_ in enumerate(self._buffer_ids):
            self._id_to_idx[id_] = start_idx + idx

        # 清空缓冲区
        self._buffer_ids.clear()
        self._buffer_vectors.clear()

        # 重建索引
        self._build_index(self._vectors)

    def _search_buffer(
        self, query: np.ndarray, k: int
    ) -> Tuple[List[float], List[str]]:
        """
        在缓冲区中线性搜索

        Args:
            query: 查询向量
            k: 返回数量

        Returns:
            (分数列表, ID列表)
        """
        if not self._buffer_ids:
            return [], []

        # 计算相似度
        scores = []
        for vec_buffer in self._buffer_vectors:
            # 内积
            sim = np.dot(vec_buffer, query.T).flatten()
            scores.extend(sim)

        # 排序取top-k
        indices = np.argsort(scores)[::-1][:k]
        top_scores = [scores[i] for i in indices]
        top_ids = [self._buffer_ids[i] for i in indices]

        return top_scores, top_ids

    def _merge_search_results(
        self,
        main_distances: np.ndarray,
        main_indices: np.ndarray,
        buffer_scores: List[float],
        buffer_ids: List[str],
        k: int,
    ) -> List[Tuple[str, float]]:
        """
        合并主索引和缓冲区的搜索结果

        Args:
            main_distances: 主索引距离
            main_indices: 主索引索引
            buffer_scores: 缓冲区分数
            buffer_ids: 缓冲区ID
            k: 返回数量

        Returns:
            [(ID, 分数)] 列表
        """
        results = []

        # 主索引结果
        for idx, dist in zip(main_indices, main_distances):
            if idx < 0 or idx >= len(self._ids):
                continue
            id_ = self._ids[idx]
            results.append((id_, float(dist)))

        # 缓冲区结果
        for id_, score in zip(buffer_ids, buffer_scores):
            results.append((id_, score))

        # 排序
        results.sort(key=lambda x: x[1], reverse=True)

        # 返回top-k
        return results[:k]

    def _check_rebuild_needed(self) -> None:
        """检查是否需要重建索引"""
        if not self._ids:
            return

        delete_ratio = len(self._deleted_ids) / len(self._ids)
        delete_count = len(self._deleted_ids)

        # 删除比例超过30%或数量超过1000时重建
        if delete_ratio > 0.3 or delete_count > 1000:
            logger.info(
                f"触发索引重建: 删除比例={delete_ratio:.1%}, "
                f"删除数量={delete_count}"
            )
            self.rebuild_index()

    @property
    def size(self) -> int:
        """当前向量数量（不包括已删除的）"""
        return len(self._ids) - len(self._deleted_ids)

    @property
    def total_size(self) -> int:
        """总向量数量（包括已删除的）"""
        return len(self._ids)

    @property
    def num_vectors(self) -> int:
        """兼容性别名：当前向量数量"""
        return self.size

    @property
    def deleted_count(self) -> int:
        """已删除向量数量"""
        return len(self._deleted_ids)

    def __len__(self) -> int:
        """向量数量（不包括已删除的）"""
        return self.size

    def has_data(self) -> bool:
        """检查磁盘上是否存在现有数据"""
        if self.data_dir is None:
            return False
        return (self.data_dir / "vectors.index").exists() or (self.data_dir / "vectors_metadata.pkl").exists()

    def __repr__(self) -> str:
        return (
            f"VectorStore(dim={self.dimension}, "
            f"size={self.size}, deleted={self.deleted_count}, "
            f"quant={self.quantization_type.value})"
        )
