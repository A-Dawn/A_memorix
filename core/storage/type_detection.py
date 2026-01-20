"""
知识类型自动检测

基于启发式规则自动识别知识类型。
"""

import re
from typing import Optional
from .knowledge_types import KnowledgeType


def detect_knowledge_type(content: str) -> KnowledgeType:
    """自动检测知识类型
    
    采用启发式规则进行判断：
    - 包含 | 分隔符 → STRUCTURED
    - 长度 > 200 且包含连续叙述 → NARRATIVE
    - 短句且包含关系动词 → FACTUAL
    - 其他 → MIXED
    
    Args:
        content: 文本内容
        
    Returns:
        检测到的知识类型
    """
    content = content.strip()
    length = len(content)
    
    # 规则1：明确的结构化标记（包含三元组分隔符）
    if "|" in content and content.count("|") >= 2:
        parts = content.split("|")
        if len(parts) == 3 and all(p.strip() for p in parts):
            return KnowledgeType.STRUCTURED
    
    # 规则2：叙事性文本（长文本 + 叙述特征）
    if length > 200:
        # 检查叙述性标记
        narrative_markers = [
            r'然后', r'接着', r'于是', r'后来', r'最后', r'突然',
            r'一天', r'曾经', r'有一次', r'从前',
            r'说道', r'问道', r'想着', r'觉得',
        ]
        narrative_score = sum(1 for marker in narrative_markers if re.search(marker, content))
        
        # 检查对话标记
        has_dialogue = bool(re.search(r'["「『].*?["」』]', content))
        
        if narrative_score >= 2 or has_dialogue:
            return KnowledgeType.NARRATIVE
    
    # 规则3：事实陈述（短句 + 关系动词）
    if length < 200:
        factual_verbs = [
            r'是', r'有', r'在', r'为', r'属于', r'位于',
            r'包含', r'拥有', r'成立于', r'出生于',
        ]
        factual_score = sum(1 for verb in factual_verbs if re.search(r'\s*' + verb + r'\s*', content))
        
        if factual_score >= 1:
            return KnowledgeType.FACTUAL
    
    # 默认返回混合类型
    return KnowledgeType.MIXED


def get_type_from_user_input(type_hint: Optional[str], content: str) -> KnowledgeType:
    """根据用户输入和内容获取知识类型
    
    Args:
        type_hint: 用户指定的类型提示（可选）
        content: 文本内容
        
    Returns:
        确定的知识类型
    """
    from .knowledge_types import get_knowledge_type_from_string
    
    # 优先使用用户指定的类型
    if type_hint:
        user_type = get_knowledge_type_from_string(type_hint)
        if user_type and user_type != KnowledgeType.AUTO:
            return user_type
    
    # 否则自动检测
    return detect_knowledge_type(content)
