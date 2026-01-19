"""
知识类型定义

定义不同类型知识的枚举和相关工具函数。
"""

from enum import Enum
from typing import Optional


class KnowledgeType(Enum):
    """知识类型枚举
    
    用于区分不同类型的知识内容，以便采用差异化的处理策略。
    """
    
    STRUCTURED = "structured"  # 结构化知识（三元组、关系）
    NARRATIVE = "narrative"    # 叙事性文本（剧情、故事）
    FACTUAL = "factual"       # 事实陈述（客观描述）
    MIXED = "mixed"           # 混合类型
    AUTO = "auto"             # 自动识别


def get_knowledge_type_from_string(type_str: str) -> Optional[KnowledgeType]:
    """从字符串获取知识类型
    
    Args:
        type_str: 类型字符串
        
    Returns:
        对应的KnowledgeType，无效则返回None
    """
    type_str = type_str.lower().strip()
    try:
        return KnowledgeType(type_str)
    except ValueError:
        return None


def should_extract_relations(knowledge_type: KnowledgeType) -> bool:
    """判断是否应该提取关系
    
    Args:
        knowledge_type: 知识类型
        
    Returns:
        是否应该提取关系
    """
    # 叙事性文本不强制提取关系
    return knowledge_type in [
        KnowledgeType.STRUCTURED,
        KnowledgeType.FACTUAL,
        KnowledgeType.MIXED,
    ]


def get_default_chunk_size(knowledge_type: KnowledgeType) -> int:
    """获取默认的分块大小
    
    Args:
        knowledge_type: 知识类型
        
    Returns:
        推荐的分块大小（字符数）
    """
    chunk_sizes = {
        KnowledgeType.STRUCTURED: 300,  # 结构化知识通常较短
        KnowledgeType.NARRATIVE: 800,   # 叙事文本可以更长以保持上下文
        KnowledgeType.FACTUAL: 500,     # 事实陈述中等长度
        KnowledgeType.MIXED: 500,       # 混合类型使用默认值
        KnowledgeType.AUTO: 500,        # 自动识别时使用默认值
    }
    return chunk_sizes.get(knowledge_type, 500)


def get_type_display_name(knowledge_type: KnowledgeType) -> str:
    """获取知识类型的显示名称
    
    Args:
        knowledge_type: 知识类型
        
    Returns:
        中文显示名称
    """
    display_names = {
        KnowledgeType.STRUCTURED: "结构化知识",
        KnowledgeType.NARRATIVE: "叙事性文本",
        KnowledgeType.FACTUAL: "事实陈述",
        KnowledgeType.MIXED: "混合类型",
        KnowledgeType.AUTO: "自动识别",
    }
    return display_names.get(knowledge_type, "未知类型")
