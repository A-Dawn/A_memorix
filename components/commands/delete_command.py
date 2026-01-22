"""
删除知识Command组件

提供知识库删除功能，支持段落、实体和关系的删除。
"""

import time
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path

from src.common.logger import get_logger
from src.plugin_system.base.base_command import BaseCommand
from ...core.utils.hash import compute_hash

# ... (existing imports)

class DeleteCommand(BaseCommand):
# ... (existing code)

    async def _delete_entity(self, entity_name: str) -> Tuple[bool, str]:
        """删除实体
        
        Args:
            entity_name: 实体名称

        Returns:
            Tuple[bool, str]: (是否成功, 结果消息)
        """
        start_time = time.time()

        # 规范化实体名称
        entity_name = entity_name.strip().lower()

        # 检查实体是否存在
        if not self.graph_store.has_node(entity_name):
            return False, f"❌ 实体不存在: {entity_name}"

        # 获取相关关系统计
        neighbors = self.graph_store.get_neighbors(entity_name)
        edge_count = len(neighbors)

        # 获取相关段落 (已自动处理 canonical lookup)
        related_paragraphs = self.metadata_store.get_paragraphs_by_entity(entity_name)
        
        # 计算hash并从向量库删除 (确保一致性)
        try:
            # 逻辑需与 MetadataStore.add_entity 保持一致
            # entity_name 已经是 canonicalized
            entity_hash = compute_hash(entity_name)
            self.vector_store.remove([entity_hash])
        except Exception as e:
            logger.warning(f"{self.log_prefix} 删除实体向量失败 {entity_name}: {e}")

        # 删除实体
        success = self.graph_store.remove_nodes([entity_name])

        if success:
            # 从元数据中删除实体
            self.metadata_store.delete_entity(entity_name)

            elapsed = time.time() - start_time

            result_lines = [
                "✅ 实体删除完成",
                f"🏷️ 实体名称: {entity_name}",
                f"🔗 关联边数: {edge_count}",
                f"📄 相关段落: {len(related_paragraphs)}",
                f"⏱️ 耗时: {elapsed*1000:.1f}ms",
                "",
                "⚠️ 注意: 相关段落未删除，如需删除请使用 /delete paragraph",
            ]

            return True, "\n".join(result_lines)
        else:
            return False, f"❌ 实体删除失败: {entity_name}"

    async def _delete_relation(self, relation_spec: str) -> Tuple[bool, str]:
        """删除关系

        Args:
            relation_spec: 关系规格 (格式: subject|predicate|object 或 hash)

        Returns:
            Tuple[bool, str]: (是否成功, 结果消息)
        """
        start_time = time.time()

        # 检查是否为hash
        if len(relation_spec) == 64:  # SHA256 hash长度
            hash_value = relation_spec
            relation = self.metadata_store.get_relation_by_hash(hash_value)

            if not relation:
                return False, f"❌ 未找到关系: {hash_value[:16]}..."

            subject = relation.get("subject", "")
            predicate = relation.get("predicate", "")
            obj = relation.get("object", "")
        else:
            # 解析关系规格
            if "|" in relation_spec:
                parts = relation_spec.split("|")
                if len(parts) != 3:
                    return False, "❌ 关系格式错误，应使用: subject|predicate|object"
                subject, predicate, obj = parts
            else:
                parts = relation_spec.split(maxsplit=2)
                if len(parts) != 3:
                    return False, "❌ 关系格式错误，应使用: subject predicate object"
                subject, predicate, obj = parts

            # 查找关系 (此时需要规范化参数以匹配数据库中的存储)
            # 注意: MetadataStore.get_relations 目前执行的是部分匹配 (LIKE)
            # 如果我们要精确删除，最好自己算 Hash 然后 get_relation_by_hash
            # 或者修改 get_relations 支持精确匹配?
            # 为了稳妥，我们计算 canonical hash 然后直接查
            
            s_canon = subject.strip().lower()
            p_canon = predicate.strip().lower()
            o_canon = obj.strip().lower()
            
            relation_key = f"{s_canon}|{p_canon}|{o_canon}"
            hash_value = compute_hash(relation_key)
            
            relation = self.metadata_store.get_relation(hash_value)
            
            if not relation:
                 # 也许用户只是想模糊删除? 但 /delete relation 在语义上应该是删除具体某一个
                 return False, f"❌ 未找到关系 (或 Hash 不匹配): {subject} {predicate} {obj}"

            # 兼容旧逻辑变量名
            relations = [relation] 

            if not relations:
                return False, f"❌ 未找到关系: {subject} {predicate} {obj}"

            if len(relations) > 1:
                return False, f"⚠️ 找到 {len(relations)} 个匹配的关系，请使用hash精确删除"

            relation = relations[0]
            hash_value = relation["hash"]

        # 删除关系
        success = self.metadata_store.delete_relation(hash_value)

        if success:
            # 从图中删除边
            subject = relation.get("subject", "")
            obj = relation.get("object", "")
            self.graph_store.remove_edges([(subject, obj)])

            elapsed = time.time() - start_time

            result_lines = [
                "✅ 关系删除完成",
                f"🔗 Hash: {hash_value[:16]}...",
                f"📌 {subject} {relation.get('predicate', '')} {obj}",
                f"⏱️ 耗时: {elapsed*1000:.1f}ms",
            ]

            return True, "\n".join(result_lines)
        else:
            return False, f"❌ 关系删除失败: {hash_value[:16]}..."

    async def _clear_knowledge_base(self) -> Tuple[bool, str]:
        """清空知识库

        Returns:
            Tuple[bool, str]: (是否成功, 结果消息)
        """
        # ⚠️ 危险操作，需要额外确认
        # 这里简单实现，实际应用中应该要求二次确认

        start_time = time.time()

        try:
            # 获取当前统计
            num_paragraphs = self.metadata_store.count_paragraphs()
            num_relations = self.metadata_store.count_relations()
            num_entities = self.metadata_store.count_entities()
            num_vectors = self.vector_store.num_vectors

            # 清空向量存储
            self.vector_store.clear()

            # 清空图存储
            self.graph_store.clear()

            # 清空元数据存储
            self.metadata_store.clear_all()

            elapsed = time.time() - start_time

            result_lines = [
                "⚠️ 知识库已清空",
                "",
                "📊 已删除内容:",
                f"  - 段落: {num_paragraphs}",
                f"  - 关系: {num_relations}",
                f"  - 实体: {num_entities}",
                f"  - 向量: {num_vectors}",
                "",
                f"⏱️ 耗时: {elapsed*1000:.1f}ms",
                "",
                "⚠️ 此操作不可撤销！",
            ]

            return True, "\n".join(result_lines)

        except Exception as e:
            return False, f"❌ 清空知识库失败: {str(e)}"

    def _get_deletion_stats(self) -> Tuple[bool, str]:
        """获取删除统计信息

        Returns:
            Tuple[bool, str]: (是否成功, 统计信息)
        """
        # 获取软删除统计
        deleted_paragraphs = self.metadata_store.count_paragraphs(include_deleted=True, only_deleted=True)
        deleted_relations = self.metadata_store.count_relations(include_deleted=True, only_deleted=True)

        # 获取当前统计
        current_paragraphs = self.metadata_store.count_paragraphs()
        current_relations = self.metadata_store.count_relations()
        current_entities = self.metadata_store.count_entities()

        # 构建响应
        lines = [
            "📊 删除统计信息",
            "",
            "🗑️ 已删除（软删除）:",
            f"  - 段落: {deleted_paragraphs}",
            f"  - 关系: {deleted_relations}",
            "",
            "📦 当前内容:",
            f"  - 段落: {current_paragraphs}",
            f"  - 关系: {current_relations}",
            f"  - 实体: {current_entities}",
            "",
            "💡 提示:",
            "  - 段落和关系使用软删除，可通过重建索引彻底清除",
            "  - 使用 /delete clear 清空整个知识库",
        ]

        return True, "\n".join(lines)

    def _get_help_message(self) -> str:
        """获取帮助消息

        Returns:
            帮助消息文本
        """
        return """📖 删除命令帮助

用法:
  /delete paragraph <hash或内容>  - 删除段落（软删除）
  /delete entity <实体名称>       - 删除实体
  /delete relation <关系规格>     - 删除关系
  /delete clear                  - 清空知识库（危险操作！）
  /delete stats                  - 显示删除统计
  /delete help                   - 显示此帮助

快捷模式:
  /delete p <hash或内容>         - 删除段落（paragraph的简写）
  /delete e <实体名称>           - 删除实体（entity的简写）
  /delete r <关系规格>           - 删除关系（relation的简写）

示例:
  /delete paragraph a1b2c3d4...
  /delete paragraph 人工智能的定义
  /delete entity Apple
  /delete relation Apple|founded|Steve Jobs
  /delete relation founded by Steve Jobs
  /delete stats

关系格式:
  - subject|predicate|object（使用|分隔）
  - subject predicate object（使用空格分隔）
  - 完整的64位hash值（精确删除）

注意事项:
  - 段落删除采用软删除，不会立即物理删除
  - 删除实体不会删除相关段落，仅删除实体节点
  - 删除关系会同时删除图中的边
  - clear操作不可撤销，请谨慎使用
"""
