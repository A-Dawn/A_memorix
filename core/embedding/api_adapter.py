"""
嵌入 API 适配器

将主程序的嵌入 API 适配为与 EmbeddingManager 兼容的接口。
"""

import asyncio
import time
from typing import List, Union, Optional
import numpy as np

from src.common.logger import get_logger
from src.chat.utils.utils import get_embedding

logger = get_logger("A_Memorix.EmbeddingAPIAdapter")


class EmbeddingAPIAdapter:
    """
    嵌入 API 适配器
    
    功能：
    - 包装主程序的 get_embedding() API
    - 提供与 EmbeddingManager 兼容的接口
    - 支持批量编码和并发控制
    - 自动检测嵌入维度
    - 错误处理和降级机制
    
    参数：
        batch_size: 批量处理大小
        max_concurrent: 最大并发请求数
        default_dimension: 默认维度（检测失败时使用）
        enable_cache: 是否启用缓存（暂未实现）
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        max_concurrent: int = 5,
        default_dimension: int = 1024,
        enable_cache: bool = False,
        model_name: str = "auto",
    ):
        """初始化嵌入 API 适配器"""
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.default_dimension = default_dimension
        self.enable_cache = enable_cache
        self.model_name = model_name
        
        # 嵌入维度（延迟初始化）
        self._dimension: Optional[int] = None
        self._dimension_detected = False
        
        # 统计信息
        self._total_encoded = 0
        self._total_errors = 0
        self._total_time = 0.0
        
        logger.info(
            f"EmbeddingAPIAdapter 初始化: batch_size={batch_size}, "
            f"max_concurrent={max_concurrent}, default_dim={default_dimension}"
        )
    
    async def _detect_dimension(self) -> int:
        """
        自动检测嵌入维度
        
        Returns:
            嵌入向量维度
        """
        if self._dimension_detected and self._dimension is not None:
            return self._dimension
        
        logger.info("正在检测嵌入模型维度...")
        
        try:
            # 使用测试文本获取嵌入
            kwargs = {}
            if self.model_name and self.model_name != "auto":
                kwargs["model"] = self.model_name
                
            test_embedding = await get_embedding("test", **kwargs)
            
            if test_embedding and isinstance(test_embedding, list):
                detected_dim = len(test_embedding)
                self._dimension = detected_dim
                self._dimension_detected = True
                logger.info(f"嵌入维度检测成功: {detected_dim}")
                return detected_dim
            else:
                logger.warning(f"嵌入维度检测失败，使用默认值: {self.default_dimension}")
                self._dimension = self.default_dimension
                self._dimension_detected = True
                return self.default_dimension
                
        except Exception as e:
            logger.error(f"嵌入维度检测异常: {e}，使用默认值: {self.default_dimension}")
            self._dimension = self.default_dimension
            self._dimension_detected = True
            return self.default_dimension
    
    async def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        生成文本嵌入
        
        Args:
            texts: 文本或文本列表
            batch_size: 批次大小（默认使用初始化时的值）
            show_progress: 是否显示进度条（暂未实现）
            normalize: 是否归一化（主程序 API 自动处理）
        
        Returns:
            嵌入向量 (N x D)
        """
        start_time = time.time()
        
        # 确保维度已检测
        if not self._dimension_detected:
            await self._detect_dimension()
        
        # 标准化输入
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        if not texts:
            return np.zeros((0, self._dimension or self.default_dimension), dtype=np.float32)
        
        # 使用配置的批次大小
        if batch_size is None:
            batch_size = self.batch_size
        
        # 批量编码
        try:
            embeddings = await self._encode_batch_internal(texts, batch_size)
            
            # 确保是2D数组
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            
            self._total_encoded += len(texts)
            elapsed = time.time() - start_time
            self._total_time += elapsed
            
            logger.debug(
                f"编码完成: {len(texts)} 个文本, "
                f"耗时 {elapsed:.2f}s, "
                f"平均 {elapsed/len(texts):.3f}s/文本"
            )
            
            # 如果是单个输入，返回1D数组
            if single_input:
                return embeddings[0]
            
            return embeddings
            
        except Exception as e:
            self._total_errors += 1
            logger.error(f"编码失败: {e}")
            
            # 降级处理：返回零向量
            logger.warning(f"返回零向量作为降级处理")
            dim = self._dimension or self.default_dimension
            fallback = np.zeros((len(texts), dim), dtype=np.float32)
            
            if single_input:
                return fallback[0]
            return fallback
    
    async def _encode_batch_internal(
        self,
        texts: List[str],
        batch_size: int,
    ) -> np.ndarray:
        """
        内部批量编码实现（带并发控制）
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
        
        Returns:
            嵌入向量数组
        """
        all_embeddings = []
        
        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # 并发请求（控制并发数）
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def encode_with_semaphore(text: str, index: int):
                async with semaphore:
                    try:
                        kwargs = {}
                        if self.model_name and self.model_name != "auto":
                            kwargs["model"] = self.model_name
                            
                        embedding = await get_embedding(text, **kwargs)
                        if embedding is None:
                            # API 返回 None，使用零向量
                            dim = self._dimension or self.default_dimension
                            embedding = [0.0] * dim
                            logger.warning(f"文本 {index} 编码返回 None，使用零向量")
                        return index, np.array(embedding, dtype=np.float32)
                    except Exception as e:
                        logger.error(f"文本 {index} 编码失败: {e}")
                        # 返回零向量
                        dim = self._dimension or self.default_dimension
                        return index, np.zeros(dim, dtype=np.float32)
            
            # 并发执行
            tasks = [
                encode_with_semaphore(text, i + idx)
                for idx, text in enumerate(batch)
            ]
            results = await asyncio.gather(*tasks)
            
            # 按索引排序并提取嵌入
            results.sort(key=lambda x: x[0])
            batch_embeddings = [emb for _, emb in results]
            all_embeddings.extend(batch_embeddings)
        
        # 转换为 numpy 数组
        return np.array(all_embeddings, dtype=np.float32)
    
    async def encode_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        批量生成嵌入（与 encode 相同，保持接口兼容）
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
            num_workers: 工作线程数（映射到 max_concurrent）
            show_progress: 是否显示进度条
        
        Returns:
            嵌入向量 (N x D)
        """
        # num_workers 映射到并发控制
        if num_workers is not None:
            old_concurrent = self.max_concurrent
            self.max_concurrent = num_workers
            try:
                result = await self.encode(texts, batch_size, show_progress)
                return result
            finally:
                self.max_concurrent = old_concurrent
        else:
            return await self.encode(texts, batch_size, show_progress)
    
    def get_embedding_dimension(self) -> int:
        """
        获取嵌入维度
        
        Returns:
            嵌入维度
        """
        if self._dimension is not None:
            return self._dimension
        
        # 如果还未检测，返回默认值
        logger.warning(f"维度尚未检测，返回默认值: {self.default_dimension}")
        return self.default_dimension
    
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            "model_name": "main_program_embedding_api",
            "dimension": self._dimension or self.default_dimension,
            "dimension_detected": self._dimension_detected,
            "batch_size": self.batch_size,
            "max_concurrent": self.max_concurrent,
            "total_encoded": self._total_encoded,
            "total_errors": self._total_errors,
            "avg_time_per_text": (
                self._total_time / self._total_encoded 
                if self._total_encoded > 0 else 0.0
            ),
        }
    
    @property
    def is_model_loaded(self) -> bool:
        """模型是否已加载（API 始终可用）"""
        return True
    
    def __repr__(self) -> str:
        return (
            f"EmbeddingAPIAdapter("
            f"dim={self._dimension or self.default_dimension}, "
            f"detected={self._dimension_detected}, "
            f"encoded={self._total_encoded})"
        )


def create_embedding_api_adapter(
    batch_size: int = 32,
    max_concurrent: int = 5,
    default_dimension: int = 1024,
    model_name: str = "auto",
) -> EmbeddingAPIAdapter:
    """
    创建嵌入 API 适配器
    
    Args:
        batch_size: 批量处理大小
        max_concurrent: 最大并发请求数
        default_dimension: 默认维度
        model_name: 指定模型名称
    
    Returns:
        嵌入 API 适配器实例
    """
    return EmbeddingAPIAdapter(
        batch_size=batch_size,
        max_concurrent=max_concurrent,
        default_dimension=default_dimension,
        model_name=model_name,
    )
