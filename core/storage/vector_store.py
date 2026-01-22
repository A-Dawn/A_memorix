"""
向量存储模块

基于Faiss的高效向量存储与检索，支持SQ8量化、Append-Only磁盘存储和内存映射。
"""

import os
import pickle
import hashlib
import shutil
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict, Set
import random
import threading  # Added threading import

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from src.common.logger import get_logger
from ..utils.quantization import QuantizationType
from ..utils.io import atomic_write, atomic_save_path

logger = get_logger("A_Memorix.VectorStore")


class VectorStore:
    """
    向量存储类 (SQ8 + Append-Only Disk)

    特性：
    - 索引: IndexIDMap2(IndexScalarQuantizer(QT_8bit))
    - 存储: float16 on-disk binary (vectors.bin)
    - 内存: 仅索引常驻 RAM (<512MB for 100k vectors)
    - ID: SHA1-based stable int64 IDs
    - 一致性: 强制 L2 Normalization (IP == Cosine)
    """

    # 默认训练触发阈值 (40 样本，过大可能导致小数据集不生效，过小可能量化退化)
    DEFAULT_MIN_TRAIN = 40
    # 强制训练样本量
    TRAIN_SIZE = 10000
    # 储水池采样上限 (流式处理前 50k 数据)
    RESERVOIR_CAPACITY = 10000
    RESERVOIR_SAMPLE_SCOPE = 50000

    def __init__(
        self,
        dimension: int,
        quantization_type: QuantizationType = QuantizationType.INT8,
        index_type: str = "sq8",
        data_dir: Optional[Union[str, Path]] = None,
        use_mmap: bool = True,
        buffer_size: int = 1024,
    ):
        if not HAS_FAISS:
            raise ImportError("Faiss 未安装，请安装: pip install faiss-cpu")

        self.dimension = dimension
        self.data_dir = Path(data_dir) if data_dir else None
        self.quantization_type = QuantizationType.INT8 
        self.index_type = "sq8" 
        self.buffer_size = buffer_size

        self._index: Optional[faiss.IndexIDMap2] = None
        self._init_index()

        self._is_trained = False
        self._vector_norm = "l2"
        
        # Fallback Index (Flat) - 用于在 SQ8 训练完成前提供检索能力
        # 必须使用 IndexIDMap2 以保证 ID 与主索引一致
        self._fallback_index: Optional[faiss.IndexIDMap2] = None
        self._init_fallback_index()
        
        self._known_hashes: Set[str] = set()
        self._deleted_ids: Set[int] = set()
        
        self._reservoir_buffer: List[np.ndarray] = []
        self._seen_count_for_reservoir = 0

        self._write_buffer_vecs: List[np.ndarray] = []
        self._write_buffer_ids: List[int] = []

        self._total_added = 0
        self._total_deleted = 0
        self._bin_count = 0 
        
        # Thread safety lock
        self._lock = threading.RLock()

        logger.info(f"VectorStore Init: dim={dimension}, SQ8 Mode, Append-Only Storage")

    def _init_index(self):
        """初始化空的 Faiss 索引"""
        quantizer = faiss.IndexScalarQuantizer(
            self.dimension, 
            faiss.ScalarQuantizer.QT_8bit, 
            faiss.METRIC_INNER_PRODUCT
        )
        self._index = faiss.IndexIDMap2(quantizer)
        self._is_trained = False

    def _init_fallback_index(self):
        """初始化 Flat 回退索引"""
        flat_index = faiss.IndexFlatIP(self.dimension)
        self._fallback_index = faiss.IndexIDMap2(flat_index)
        logger.debug("Fallback index (Flat) initialized.")

    @staticmethod
    def _generate_id(key: str) -> int:
        """生成稳定的 int64 ID (SHA1 截断)"""
        h = hashlib.sha1(key.encode("utf-8")).digest()
        val = int.from_bytes(h[:8], byteorder="big", signed=False)
        return val & 0x7FFFFFFFFFFFFFFF

    @property
    def _bin_path(self) -> Path:
        return self.data_dir / "vectors.bin"
    
    @property
    def _ids_bin_path(self) -> Path:
        return self.data_dir / "vectors_ids.bin"

    @property
    def _int_to_str_map(self) -> Dict[int, str]:
        """Lazy build volatile map from known hashes"""
        # Note: This is read-heavy and cached, might need lock if _known_hashes updates concurrently
        # But add/delete are now locked, so checking len mismatch is somewhat safe-ish for quick dirty cache
        if not hasattr(self, "_cached_map") or len(self._cached_map) != len(self._known_hashes):
            with self._lock: # Protect cache rebuild
                 self._cached_map = {self._generate_id(k): k for k in self._known_hashes}
        return self._cached_map

    def add(self, vectors: np.ndarray, ids: List[str]) -> int:
        with self._lock:
            if vectors.shape[1] != self.dimension:
                raise ValueError(f"Dimension mismatch: {vectors.shape[1]} vs {self.dimension}")

            vectors = np.ascontiguousarray(vectors, dtype=np.float32)
            faiss.normalize_L2(vectors)

            processed_vecs = []
            processed_int_ids = []
            
            for i, str_id in enumerate(ids):
                if str_id in self._known_hashes:
                    continue
                
                int_id = self._generate_id(str_id)
                self._known_hashes.add(str_id)
                
                processed_vecs.append(vectors[i])
                processed_int_ids.append(int_id)

            if not processed_vecs:
                return 0

            batch_vecs = np.array(processed_vecs, dtype=np.float32)
            batch_ids = np.array(processed_int_ids, dtype=np.int64)

            self._write_buffer_vecs.append(batch_vecs)
            self._write_buffer_ids.extend(processed_int_ids)

            if len(self._write_buffer_ids) >= self.buffer_size:
                self._flush_write_buffer()

            if not self._is_trained:
                # 双写到回退索引
                self._fallback_index.add_with_ids(batch_vecs, batch_ids)
                
                self._update_reservoir(batch_vecs)
                # 这里的 TRAIN_SIZE 取默认 10k，或者根据当前数据量动态判断
                if len(self._reservoir_buffer) >= 10000:
                    logger.info(f"训练样本达到上限，开始训练...")
                    self._train_and_replay()

            self._total_added += len(batch_ids)
            return len(batch_ids)
    
    def _flush_write_buffer(self):
        # Assumes caller holds lock or is internal
        if not self._write_buffer_vecs:
            return

        batch_vecs = np.concatenate(self._write_buffer_vecs, axis=0)
        batch_ids = np.array(self._write_buffer_ids, dtype=np.int64)

        vecs_fp16 = batch_vecs.astype(np.float16)
        
        with open(self._bin_path, "ab") as f:
            f.write(vecs_fp16.tobytes())
        
        ids_bytes = batch_ids.astype('>i8').tobytes()
        with open(self._ids_bin_path, "ab") as f:
            f.write(ids_bytes)
            
        self._bin_count += len(batch_ids)

        if self._is_trained and self._index.is_trained:
            self._index.add_with_ids(batch_vecs, batch_ids)
        else:
            # 即使在 flush 时，如果未训练，也要同步到 fallback
            self._fallback_index.add_with_ids(batch_vecs, batch_ids)

        self._write_buffer_vecs.clear()
        self._write_buffer_ids.clear()

    def _update_reservoir(self, vectors: np.ndarray):
        for vec in vectors:
            self._seen_count_for_reservoir += 1
            if len(self._reservoir_buffer) < self.RESERVOIR_CAPACITY:
                self._reservoir_buffer.append(vec)
            else:
                if self._seen_count_for_reservoir <= self.RESERVOIR_SAMPLE_SCOPE:
                    r = random.randint(0, self._seen_count_for_reservoir - 1)
                    if r < self.RESERVOIR_CAPACITY:
                        self._reservoir_buffer[r] = vec

    def _train_and_replay(self):
        # Assumes caller holds lock
        if not self._reservoir_buffer:
            logger.warning("No training data available.")
            return

        train_data = np.array(self._reservoir_buffer, dtype=np.float32)
        logger.info(f"Training Index with {len(train_data)} samples...")
        
        try:
            self._index.train(train_data)
        except Exception as e:
            logger.error(f"SQ8 Training failed: {e}. Staying in fallback mode.")
            return

        self._is_trained = True
        self._reservoir_buffer = []

        logger.info("Replaying data from disk to populate index...")
        try:
            replay_count = self._replay_vectors_to_index()
            # 只有当 replay 成功且数据量一致时，才释放回退索引
            if self._index.ntotal >= self._bin_count:
                logger.info(f"Replay successful ({self._index.ntotal}/{self._bin_count}). Releasing fallback index.")
                self._fallback_index.reset()
            else:
                logger.warning(f"Replay count mismatch: {self._index.ntotal} vs {self._bin_count}. Keeping fallback index.")
        except Exception as e:
            logger.error(f"Replay failed: {e}. Keeping fallback index as backup.")

    def _replay_vectors_to_index(self) -> int:
        """从 vectors.bin 读取并添加到 index"""
        if not self._bin_path.exists() or not self._ids_bin_path.exists():
            return 0

        vec_item_size = self.dimension * 2
        id_item_size = 8
        chunk_size = 10000 
        
        with open(self._bin_path, "rb") as f_vec, open(self._ids_bin_path, "rb") as f_id:
            while True:
                vec_data = f_vec.read(chunk_size * vec_item_size)
                id_data = f_id.read(chunk_size * id_item_size)
                
                if not vec_data:
                    break
                
                batch_fp16 = np.frombuffer(vec_data, dtype=np.float16).reshape(-1, self.dimension)
                batch_fp32 = batch_fp16.astype(np.float32)
                faiss.normalize_L2(batch_fp32)
                
                batch_ids = np.frombuffer(id_data, dtype='>i8').astype(np.int64)
                
                valid_mask = [id_ not in self._deleted_ids for id_ in batch_ids]
                if not all(valid_mask):
                    batch_fp32 = batch_fp32[valid_mask]
                    batch_ids = batch_ids[valid_mask]
                
                if len(batch_ids) > 0:
                    self._index.add_with_ids(batch_fp32, batch_ids)

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        filter_deleted: bool = True,
    ) -> Tuple[List[str], List[float]]:
        
        # 核心逻辑：主索引未就绪时，检查回退索引
        # 使用 Double-Checked Locking 模式确保线程安全
        if not self._is_trained or self._index.ntotal == 0:
            with self._lock:
                # 再次检查，防止多线程竞争导致的重复初始化
                if not self._is_trained or self._index.ntotal == 0:
                    self._flush_write_buffer()
                    
                    # 判断回退索引是否需要从磁盘自举 (重启动作)
                    if self._bin_count > 0 and self._fallback_index.ntotal == 0:
                        logger.info(f"Detecting data on disk ({self._bin_count}) but index not ready. Bootstrapping fallback index...")
                        self._bootstrap_fallback_from_disk()
                    
                    # 尝试根据阈值强制训练
                    min_train = getattr(self, "min_train_threshold", self.DEFAULT_MIN_TRAIN)
                    if self._bin_count > 0 and self._bin_count >= min(min_train, self._bin_count) and not self._is_trained:
                         # 只有达到 40 个或更多时才尝试训练，否则继续使用 Flat
                         if self._bin_count >= min_train:
                            self._force_train_small_data()
            
        # 最终搜索源选择 (Read is safe assuming _index ptr swap is atomic in Python, but structure mutation is protected)
        search_index = self._index if (self._is_trained and self._index.ntotal > 0) else self._fallback_index
        
        if search_index.ntotal == 0:
            logger.warning("Indices are empty. No data to search.")
            return [], []

        if query.ndim == 1:
            query = query.reshape(1, -1)
            
        query = np.ascontiguousarray(query, dtype=np.float32)
        faiss.normalize_L2(query)
        
        # 执行检索
        dists, ids = search_index.search(query, k * 2)
        
        # Faiss search 返回的是 (1, K) 的数组，取第一行
        dists = dists[0]
        ids = ids[0]
        
        results = []
        for id_val, score in zip(ids, dists):
            if id_val == -1: continue
            if filter_deleted and id_val in self._deleted_ids:
                continue
            
            str_id = self._int_to_str_map.get(id_val)
            if str_id:
                results.append((str_id, float(score)))

        # Sort and trim just in case filtering reduced count
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:k]
        
        if not results:
            return [], []
            
        return [r[0] for r in results], [r[1] for r in results]

    def _bootstrap_fallback_from_disk(self):
        """重启后自举：从磁盘 vectors.bin 加载数据到 fallback 索引"""
        # Internal method, lock should be held by caller
        if not self._bin_path.exists() or not self._ids_bin_path.exists():
            return

        logger.info("Replaying all disk vectors to fallback index...")
        vec_item_size = self.dimension * 2
        id_item_size = 8
        chunk_size = 10000 
        
        with open(self._bin_path, "rb") as f_vec, open(self._ids_bin_path, "rb") as f_id:
            while True:
                vec_data = f_vec.read(chunk_size * vec_item_size)
                id_data = f_id.read(chunk_size * id_item_size)
                if not vec_data: break
                
                batch_fp16 = np.frombuffer(vec_data, dtype=np.float16).reshape(-1, self.dimension)
                batch_fp32 = batch_fp16.astype(np.float32)
                faiss.normalize_L2(batch_fp32)
                batch_ids = np.frombuffer(id_data, dtype='>i8').astype(np.int64)
                
                valid_mask = [id_ not in self._deleted_ids for id_ in batch_ids]
                if any(valid_mask):
                    self._fallback_index.add_with_ids(batch_fp32[valid_mask], batch_ids[valid_mask])
        
        logger.info(f"Fallback index self-bootstrapped with {self._fallback_index.ntotal} items.")

    def _force_train_small_data(self):
        # Internal method, lock should be held by caller
        logger.info("Forcing training on small dataset...")
        self._reservoir_buffer = [] 
        
        chunk_size = 10000
        vec_item_size = self.dimension * 2
        
        with open(self._bin_path, "rb") as f:
            while len(self._reservoir_buffer) < self.TRAIN_SIZE:
                data = f.read(chunk_size * vec_item_size)
                if not data: break
                fp16 = np.frombuffer(data, dtype=np.float16).reshape(-1, self.dimension)
                fp32 = fp16.astype(np.float32)
                faiss.normalize_L2(fp32)
                
                for vec in fp32:
                    self._reservoir_buffer.append(vec)
                    if len(self._reservoir_buffer) >= self.TRAIN_SIZE:
                        break
        
        self._train_and_replay()

    def delete(self, ids: List[str]) -> int:
        with self._lock:
            count = 0
            for str_id in ids:
                if str_id not in self._known_hashes:
                    continue
                int_id = self._generate_id(str_id)
                if int_id not in self._deleted_ids:
                    self._deleted_ids.add(int_id)
                    if self._index.is_trained:
                         self._index.remove_ids(np.array([int_id], dtype=np.int64))
                    # 同步从 fallback 移除
                    if self._fallback_index.ntotal > 0:
                         self._fallback_index.remove_ids(np.array([int_id], dtype=np.int64))
                    count += 1
            self._total_deleted += count
            
            # Check GC
            self._check_rebuild_needed()
            return count

    def _check_rebuild_needed(self):
        """GC Excution Check"""
        # Internal check, lock held by delete
        if self._bin_count == 0: return
        ratio = len(self._deleted_ids) / self._bin_count
        if ratio > 0.3 and len(self._deleted_ids) > 1000:
            logger.info(f"Triggering GC/Rebuild (deleted ratio: {ratio:.2f})")
            self.rebuild_index()

    def rebuild_index(self):
        """GC: 重建索引，压缩 bin 文件"""
        # TODO: This logic is complex and should be protected by lock
        # Caller holds lock (from delete -> check -> rebuild), so it is safe.
        logger.info("Starting Compaction (GC)...")
        
        tmp_bin = self.data_dir / "vectors.bin.tmp"
        tmp_ids = self.data_dir / "vectors_ids.bin.tmp"
        
        vec_item_size = self.dimension * 2
        id_item_size = 8
        chunk_size = 10000
        
        new_count = 0
        
        # 1. Compact Files
        with open(self._bin_path, "rb") as f_vec, open(self._ids_bin_path, "rb") as f_id, \
             open(tmp_bin, "wb") as w_vec, open(tmp_ids, "wb") as w_id:
            while True:
                vec_data = f_vec.read(chunk_size * vec_item_size)
                id_data = f_id.read(chunk_size * id_item_size)
                if not vec_data: break
                
                batch_fp16 = np.frombuffer(vec_data, dtype=np.float16).reshape(-1, self.dimension)
                batch_ids = np.frombuffer(id_data, dtype='>i8').astype(np.int64)
                
                keep_mask = [id_ not in self._deleted_ids for id_ in batch_ids]
                
                if any(keep_mask):
                    keep_vecs = batch_fp16[keep_mask]
                    keep_ids = batch_ids[keep_mask]
                    
                    w_vec.write(keep_vecs.tobytes())
                    w_id.write(keep_ids.astype('>i8').tobytes())
                    new_count += len(keep_ids)

        # 2. Reset State & Atomic Swap
        self._bin_count = new_count
        
        # Close current index
        self._index.reset()
        if self._fallback_index: self._fallback_index.reset() # Also clear fallback
        self._is_trained = False
        
        # Swap files
        shutil.move(str(tmp_bin), str(self._bin_path))
        shutil.move(str(tmp_ids), str(self._ids_bin_path))
        
        # Reset Tombstones (Critical)
        self._deleted_ids.clear()
        
        # 3. Reload/Rebuild Index (Fresh Train)
        # We need to re-train because data distribution might have changed significantly after deletion
        self._init_index()
        self._init_fallback_index() # Re-init fallback too
        self._force_train_small_data() # This will train and replay from the NEW compact file
        
        logger.info("Compaction Complete.")

    def save(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        with self._lock:
            if not data_dir: data_dir = self.data_dir
            if not data_dir: raise ValueError("No data_dir")
            
            data_dir = Path(data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)
            
            self._flush_write_buffer()
            
            if self._is_trained:
                index_path = data_dir / "vectors.index"
                with atomic_save_path(index_path) as tmp:
                    faiss.write_index(self._index, tmp)

            meta = {
                "dimension": self.dimension,
                "quantization_type": self.quantization_type.value,
                "is_trained": self._is_trained,
                "vector_norm": self._vector_norm,
                "deleted_ids": list(self._deleted_ids),
                "known_hashes": list(self._known_hashes),
            }
            
            with atomic_write(data_dir / "vectors_metadata.pkl", "wb") as f:
                pickle.dump(meta, f)
                
            logger.info("VectorStore saved.")

    def load(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        with self._lock:
            if not data_dir: data_dir = self.data_dir
            data_dir = Path(data_dir)
            
            npy_path = data_dir / "vectors.npy"
            idx_path = data_dir / "vectors.index"
            bin_path = data_dir / "vectors.bin"
            
            if npy_path.exists() and not bin_path.exists():
                logger.info("Found legacy .npy, migrating to .bin...")
                self._migrate_from_npy(npy_path, idx_path, data_dir)
                return

            meta_path = data_dir / "vectors_metadata.pkl"
            if not meta_path.exists():
                logger.warning("No metadata found, initialized empty.")
                return
                
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
                
            if meta.get("vector_norm") != "l2":
                logger.warning("Index IDMap2 version mismatch (L2 Norm), forcing rebuild...")
                self._known_hashes = set(meta.get("ids", [])) | set(meta.get("known_hashes", []))
                self._deleted_ids = set(meta.get("deleted_ids", []))
                self._init_index()
                self._force_train_small_data()
                return

            self._is_trained = meta.get("is_trained", False)
            self._vector_norm = meta.get("vector_norm", "l2")
            self._deleted_ids = set(meta.get("deleted_ids", []))
            self._known_hashes = set(meta.get("known_hashes", []))
            
            if self._is_trained:
                if idx_path.exists():
                    try:
                        self._index = faiss.read_index(str(idx_path))
                        if not isinstance(self._index, faiss.IndexIDMap2):
                            logger.warning("Loaded index type mismatch. Rebuilding...")
                            self._init_index()
                            self._force_train_small_data()
                    except Exception as e:
                         logger.error(f"Failed to load index: {e}. Rebuilding...")
                         self._init_index()
                         self._force_train_small_data()
                else:
                    logger.warning("Index file missing despite metadata indicating trained. Rebuilding from bin...")
                    self._init_index()
                    self._force_train_small_data()
            
            if bin_path.exists():
                self._bin_count = bin_path.stat().st_size // (self.dimension * 2)

    def _migrate_from_npy(self, npy_path, idx_path, data_dir):
        # Assumed safe to run without lock as it is called from inside load() which is locked (if load is used correctly)
        # But for safety, recursive lock helps.
        
        try:
            arr = np.load(npy_path, mmap_mode="r")
        except Exception:
            arr = np.load(npy_path)
            
        meta_path = data_dir / "vectors_metadata.pkl"
        old_ids = []
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                m = pickle.load(f)
                old_ids = m.get("ids", [])
                
        if len(arr) != len(old_ids):
            logger.error(f"Migration mismatch: arr {len(arr)} != ids {len(old_ids)}")
            return

        logger.info(f"Migrating {len(arr)} vectors...")
        
        chunk = 1000
        for i in range(0, len(arr), chunk):
            sub_arr = arr[i : i+chunk]
            sub_ids = old_ids[i : i+chunk]
            self.add(sub_arr, sub_ids)
            
        if not self._is_trained:
            self._force_train_small_data()

        shutil.move(str(npy_path), str(npy_path) + ".bak")
        if idx_path.exists():
            shutil.move(str(idx_path), str(idx_path) + ".bak")
            
        logger.info("Migration complete.")

    def clear(self) -> None:
        with self._lock:
            self._ids_bin_path.unlink(missing_ok=True)
            self._bin_path.unlink(missing_ok=True)
            self._init_index()
            self._known_hashes.clear()
            self._deleted_ids.clear()
            self._bin_count = 0
            logger.info("VectorStore cleared.")

    def has_data(self) -> bool:
        return (self.data_dir / "vectors_metadata.pkl").exists()

    @property
    def num_vectors(self) -> int:
        return len(self._known_hashes) - len(self._deleted_ids)

