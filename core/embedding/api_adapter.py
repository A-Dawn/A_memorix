"""
嵌入 API 适配器

将主程序的嵌入 API 适配为与 EmbeddingManager 兼容的接口。
"""

import asyncio
import time
from typing import List, Union, Optional
import numpy as np
import openai
import aiohttp

from src.common.logger import get_logger
from src.chat.utils.utils import get_embedding
from src.config.config import model_config
from src.llm_models.model_client.base_client import client_registry

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
    - 支持手动设置请求维度
    
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
        retry_config: Optional[dict] = None,
    ):
        """初始化嵌入 API 适配器"""
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.default_dimension = default_dimension
        self.enable_cache = enable_cache
        self.model_name = model_name
        
        # 重试配置
        self.retry_config = retry_config or {}
        self.max_attempts = self.retry_config.get("max_attempts", 3)
        self.max_wait_seconds = self.retry_config.get("max_wait_seconds", 10)
        self.min_wait_seconds = self.retry_config.get("min_wait_seconds", 2)
        
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

    async def _request_with_retry(self, client, model_info, text: str, extra_params: dict):
        """
        统一重试逻辑，避免在模块导入期依赖第三方重试框架。
        """
        retriable_exceptions = (openai.APIConnectionError, aiohttp.ClientError, asyncio.TimeoutError)
        max_attempts = max(1, int(self.max_attempts))
        base_wait = max(0.1, float(self.min_wait_seconds))
        max_wait = max(base_wait, float(self.max_wait_seconds))

        last_exc = None
        for attempt in range(1, max_attempts + 1):
            try:
                return await client.get_embedding(
                    model_info=model_info,
                    embedding_input=text,
                    extra_params=extra_params,
                )
            except retriable_exceptions as e:
                last_exc = e
                if attempt >= max_attempts:
                    raise
                wait_seconds = min(max_wait, base_wait * (2 ** (attempt - 1)))
                logger.warning(
                    f"Embedding 请求失败，重试 {attempt}/{max_attempts - 1}，"
                    f"{wait_seconds:.1f}s 后重试: {e}"
                )
                await asyncio.sleep(wait_seconds)
            except Exception:
                raise

        if last_exc:
            raise last_exc
        raise RuntimeError("Embedding 请求失败：未知错误")

    async def _get_embedding_direct(self, text: str, dimensions: Optional[int] = None) -> Optional[List[float]]:
        """
        直接通过 Client 获取 Embedding，支持传递 dimensions 参数
        
        Args:
            text: 输入文本
            dimensions: 请求的维度 (仅部分模型支持，如 OpenAI text-embedding-3)
            
        Returns:
            嵌入向量列表 或 None
        """
        try:
            # 1. 确定模型
            # 默认使用配置中的 embedding 任务模型
            task_config = model_config.model_task_config.embedding
            
            # 如果指定了 model_name，尝试查找对应的模型信息
            if self.model_name and self.model_name != "auto":
                model_identifier = self.model_name
                # 这里简化处理：我们假设 self.model_name 是 model_config 中的一个 key
                # 或者它就是一个直接的模型标识符。
                # 为了保持与 main program 一致，我们最好通过 task_config 来选择
                # 但这里我们需要更底层的控制。
                
                # 尝试从 model_config 获取模型信息
                try:
                     model_info = model_config.get_model_info(model_identifier)
                except Exception:
                     # 如果找不到，可能它不是配置名而是直接的模型名，这在使用 LLMRequest 时通常不支持
                     # 但我们这里尝试回退到默认 embedding 模型
                     model_info = model_config.get_model_info(task_config.model_list[0])
            else:
                 # 使用 embedding 任务配置的第一个模型 (简单起见，不复现复杂的负载均衡逻辑)
                 # 如果需要负载均衡，可以复用 LLMRequest，但 LLMRequest 不支持 kwargs。
                 # 所以这里这是一个权衡：支持 dimensions vs 支持负载均衡。
                 # 鉴于这通常用于特定高级模型，我们优先支持特性。
                 model_name_to_use = task_config.model_list[0]
                 model_info = model_config.get_model_info(model_name_to_use)
            
            # 2. 获取 Provider 和 Client
            api_provider = model_config.get_provider(model_info.api_provider)
            # 强制新建客户端以避免潜在的 event loop 问题 (参考 LLMRequest)
            client = client_registry.get_client_class_instance(api_provider, force_new=True)
            
            # 3. 构造参数
            extra_params = {}
            if dimensions is not None:
                extra_params["dimensions"] = dimensions
                
            # 4. 调用 API (内置指数退避重试)
            response = await self._request_with_retry(
                client=client,
                model_info=model_info,
                text=text,
                extra_params=extra_params,
            )
            
            return response.embedding
            
        except Exception as e:
            logger.error(f"通过直接 Client 获取 Embedding 失败: {e}")
            return None
    
    async def _detect_dimension(self) -> int:
        """
        自动检测嵌入维度
        
        Returns:
            嵌入向量维度
        """
        if self._dimension_detected and self._dimension is not None:
            return self._dimension
        
        logger.info("正在检测嵌入模型维度...")
        
        # 策略：优先尝试请求 default_dimension
        # 如果模型支持（如 text-embedding-3），将返回该维度的向量
        # 如果模型不支持（抛出异常或忽略参数），则回退到不带参数的探测
        
        # 1. 尝试使用 default_dimension 请求
        try:
            target_dim = self.default_dimension
            logger.debug(f"尝试请求指定维度: {target_dim}")
            
            test_embedding = await self._get_embedding_direct("test", dimensions=target_dim)
            
            if test_embedding and isinstance(test_embedding, list):
                detected_dim = len(test_embedding)
                
                # 检查是否真的返回了请求的维度
                if detected_dim == target_dim:
                    logger.info(f"嵌入维度检测成功 (匹配配置): {detected_dim}")
                    self._dimension = detected_dim
                    self._dimension_detected = True
                    return detected_dim
                else:
                    logger.warning(f"请求维度 {target_dim} 但模型返回 {detected_dim}，将使用模型返回的自然维度")
                    self._dimension = detected_dim
                    self._dimension_detected = True
                    return detected_dim
            else:
                # 返回 None，可能是临时错误或不支持，尝试不带参数
                logger.debug("带参数探测返回空，尝试不带参数探测...")
                pass
                
        except Exception as e:
            # 某些模型如果收到不支持的参数可能会报错
            logger.debug(f"带维度参数探测失败: {e}，尝试不带参数探测...")
            pass
            
        # 2. 回退：尝试不带 dimensions 参数探测自然维度
        try:
            test_embedding = await self._get_embedding_direct("test", dimensions=None)
            
            if test_embedding and isinstance(test_embedding, list):
                detected_dim = len(test_embedding)
                self._dimension = detected_dim
                self._dimension_detected = True
                logger.info(f"嵌入维度检测成功 (自然维度): {detected_dim}")
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
        dimensions: Optional[int] = None,
    ) -> np.ndarray:
        """
        生成文本嵌入
        
        Args:
            texts: 文本或文本列表
            batch_size: 批次大小（默认使用初始化时的值）
            show_progress: 是否显示进度条（暂未实现）
            normalize: 是否归一化（主程序 API 自动处理）
            dimensions: 请求的嵌入维度 (可选, 仅部分模型支持)
        
        Returns:
            嵌入向量 (N x D)
        """
        start_time = time.time()
        
        # 确保维度已检测 (如果未指定 dimensions)
        # 如果指定了 dimensions，我们信任用户，或者用它更新 self._dimension?
        if dimensions is not None:
             target_dim = dimensions
        else:
            if not self._dimension_detected:
                await self._detect_dimension()
            target_dim = self._dimension or self.default_dimension

        
        # 标准化输入
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        if not texts:
            return np.zeros((0, target_dim), dtype=np.float32)
        
        # 使用配置的批次大小
        if batch_size is None:
            batch_size = self.batch_size
        
        # 批量编码
        try:
            embeddings = await self._encode_batch_internal(texts, batch_size, dimensions=dimensions)
            
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
            
            # 失败处理：返回 NaN 向量以供上层识别跳过
            logger.warning(f"返回 NaN 向量以供跳过处理")
            fallback = np.full((len(texts), target_dim), np.nan, dtype=np.float32)
            
            if single_input:
                return fallback[0]
            return fallback
    
    async def _encode_batch_internal(
        self,
        texts: List[str],
        batch_size: int,
        dimensions: Optional[int] = None,
    ) -> np.ndarray:
        """
        内部批量编码实现（带并发控制）
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
            dimensions: 维度参数
        
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
                        # 使用直接接口以支持 dimensions
                        embedding = await self._get_embedding_direct(text, dimensions=dimensions)
                        
                        if embedding is None:
                            # API 返回 None，使用 NaN 向量
                            dim = dimensions or self._dimension or self.default_dimension
                            embedding = [np.nan] * dim
                            logger.warning(f"文本 {index} 编码返回 None，使用 NaN 向量")
                        return index, np.array(embedding, dtype=np.float32)
                    except Exception as e:
                        logger.error(f"文本 {index} 编码失败: {e}")
                        # 返回 NaN 向量
                        dim = dimensions or self._dimension or self.default_dimension
                        return index, np.full(dim, np.nan, dtype=np.float32)
            
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
        dimensions: Optional[int] = None,
    ) -> np.ndarray:
        """
        批量生成嵌入（与 encode 相同，保持接口兼容）
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
            num_workers: 工作线程数（映射到 max_concurrent）
            show_progress: 是否显示进度条
            dimensions: 请求的维度
        
        Returns:
            嵌入向量 (N x D)
        """
        # num_workers 映射到并发控制
        if num_workers is not None:
            old_concurrent = self.max_concurrent
            self.max_concurrent = num_workers
            try:
                result = await self.encode(texts, batch_size, show_progress, dimensions=dimensions)
                return result
            finally:
                self.max_concurrent = old_concurrent
        else:
            return await self.encode(texts, batch_size, show_progress, dimensions=dimensions)
    
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
            "model_name": self.model_name,
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
    retry_config: Optional[dict] = None,
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
        retry_config=retry_config,
    )
