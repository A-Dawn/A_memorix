from typing import Any, List, Tuple, Optional, Dict
from src.plugin_system.base.base_tool import BaseTool
from src.plugin_system.base.component_types import ToolParamType
from src.chat.message_receive.chat_stream import ChatStream
from src.common.logger import get_logger

logger = get_logger("A_Memorix.MemoryModifierTool")

class MemoryModifierTool(BaseTool):
    """
    记忆修改工具
    
    允许Bot自我调整记忆的权重或持久性：
    - reinforce: 强化相关记忆 (增加连接权重)
    - weaken: 弱化相关记忆 (减少连接权重)
    - remember_forever: 将相关记忆标记为永久 (防止遗忘)
    - forget: 遗忘 (大幅降低权重或标记为非永久)
    """
    
    name = "memory_modifier"
    description = "用于修改记忆的权重（强化/弱化）或设置永久性。当用户强调某事重要或纠正错误时使用。"
    
    parameters: List[Tuple[str, ToolParamType, str, bool, List[str] | None]] = [
         (
            "action",
            ToolParamType.STRING,
            "动作: reinforce(强化), weaken(弱化), remember_forever(永久记忆), forget(遗忘)",
            True,
            ["reinforce", "weaken", "remember_forever", "forget"],
        ),
        (
            "query",
            ToolParamType.STRING,
            "目标记忆的查询内容",
            True,
            None,
        ),
        (
            "target_type",
            ToolParamType.STRING,
            "目标类型: relation(关系), entity(实体), paragraph(段落)",
            False,
            ["relation", "entity", "paragraph"],
        ),
        (
            "strength",
            ToolParamType.FLOAT,
            "调整强度 (0.1 - 5.0)，默认为1.0",
            False,
            None,
        ),
    ]

    available_for_llm = True
    
    def __init__(self, plugin_config: Optional[dict] = None, chat_stream: Optional["ChatStream"] = None):
        super().__init__(plugin_config, chat_stream)
        
        # 获取存储实例
        self.vector_store = self.plugin_config.get("vector_store")
        self.graph_store = self.plugin_config.get("graph_store")
        self.metadata_store = self.plugin_config.get("metadata_store")
        
        self._ensure_stores()

    def _ensure_stores(self):
        """确保存储实例已加载"""
        if not all([self.vector_store, self.graph_store, self.metadata_store]):
            from ...plugin import A_MemorixPlugin
            instances = A_MemorixPlugin.get_storage_instances()
            if instances:
                self.vector_store = self.vector_store or instances.get("vector_store")
                self.graph_store = self.graph_store or instances.get("graph_store")
                self.metadata_store = self.metadata_store or instances.get("metadata_store")

    async def execute(self, function_args: dict[str, Any]) -> dict[str, Any]:
        """
        执行记忆修改

        Args:
            function_args:
                - action: 动作 
                - query: 查询
                - target_type: 目标类型
                - strength: 强度
        
        Returns:
            执行结果
        """
        self._ensure_stores()
        
        if not self.graph_store or not self.metadata_store:
             return {
                 "success": False,
                 "error": "存储组件未初始化",
                 "results": []
             }

        action = function_args.get("action")
        query = function_args.get("query")
        target_type = function_args.get("target_type", "relation")
        strength = float(function_args.get("strength", 1.0))
        
        # 1. Find the target items
        found_items = []
        
        if target_type == "relation":
             # 简单实现：使用 query 模糊搜索 relations
             results = self.metadata_store.query(
                 "SELECT * FROM relations WHERE subject LIKE ? OR object LIKE ? LIMIT 5",
                 (f"%{query}%", f"%{query}%")
             )
             found_items = results
        
        if not found_items:
            return {
                "success": False,
                "error": f"未找到与 '{query}' 相关的记忆 ({target_type})",
                "results": []
            }
            
        count = 0
        details = []
        
        # 2. Apply Action
        if action in ["reinforce", "weaken", "forget"]:
            # Adjust Weights
            delta = 0.5 * strength if action == "reinforce" else -0.5 * strength
            if action == "forget":
                delta = -2.0 * strength # Heavy penalty
                
            for item in found_items:
                if "subject" in item and "object" in item:
                    src = item["subject"]
                    tgt = item["object"]
                    
                    # Update A->B
                    w1 = self.graph_store.update_edge_weight(src, tgt, delta)
                    count += 1
                    details.append(f"{src}->{tgt} ({w1:.2f})")
            
            verb = "强化" if delta > 0 else "弱化"
            return {
                "success": True,
                "content": f"✅ 已{verb} {count} 条相关记忆 (Query: {query})",
                "results": details
            }

        elif action == "remember_forever":
            # Set Permanence
            is_perm = True
            for item in found_items:
                self.metadata_store.set_permanence(item["hash"], "relation", is_perm)
                count += 1
                details.append(f"{item['subject']}->{item['object']} (Permanent)")
                
            return {
                "success": True, 
                "content": f"✅ 已标记 {count} 条记忆为永久 (Query: {query})",
                "results": details
            }
            
        else:
            return {
                "success": False,
                "error": f"未知动作: {action}",
                "results": []
            }
