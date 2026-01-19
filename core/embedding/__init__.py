"""嵌入模块 - 向量生成与量化"""

# 新的 API 适配器（主程序嵌入 API）
from .api_adapter import (
    EmbeddingAPIAdapter,
    create_embedding_api_adapter,
)

# 保留旧的导入以兼容（标记为弃用）
try:
    from .manager import (
        EmbeddingManager,
        create_embedding_manager_from_config,
    )
    from .presets import (
        EmbeddingModelConfig,
        get_custom_config,
        validate_config_compatibility,
        are_models_compatible,
    )
    _HAS_LOCAL_MANAGER = True
except ImportError:
    _HAS_LOCAL_MANAGER = False

from ..utils.quantization import QuantizationType

__all__ = [
    # 新的 API 适配器（推荐使用）
    "EmbeddingAPIAdapter",
    "create_embedding_api_adapter",
    # 量化
    "QuantizationType",
]

# 兼容性导出（如果本地管理器存在）
if _HAS_LOCAL_MANAGER:
    __all__.extend([
        "EmbeddingManager",
        "create_embedding_manager_from_config",
        "EmbeddingModelConfig",
        "get_custom_config",
        "validate_config_compatibility",
        "are_models_compatible",
    ])

