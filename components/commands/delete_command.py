"""
删除知识Command组件

提供知识库删除功能，支持段落、实体和关系的删除。
"""

import time
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path

from src.common.logger import get_logger
from src.plugin_system.base.base_command import BaseCommand
from src.chat.message_receive.message import MessageRecv

# 导入核心模块
from ...core import VectorStore, GraphStore, MetadataStore

logger = get_logger("A_Memorix.DeleteCommand")


class DeleteCommand(BaseCommand):
    """删除知识Command

    功能：
    - 删除段落（软删除）
    - 删除实体
    - 删除关系
    - 批量删除
    - 清空知识库
    """

    # Command基本信息
    command_name = "delete"
    command_description = "删除知识库内容，支持段落、实体和关系的删除"
    command_pattern = r"^\/delete(?:\s+(?P<mode>\w+))?(?:\s+(?P<content>.+))?$"

    def __init__(self, message: MessageRecv, plugin_config: Optional[dict] = None):
        """初始化删除Command"""
        super().__init__(message, plugin_config)

        # 获取存储实例 (优先从配置获取，兜底从插件实例获取)
        self.vector_store: Optional[VectorStore] = self.plugin_config.get("vector_store")
        self.graph_store: Optional[GraphStore] = self.plugin_config.get("graph_store")
        self.metadata_store: Optional[MetadataStore] = self.plugin_config.get("metadata_store")

        # 兜底逻辑：如果配置中没有存储实例，尝试直接从插件系统获取
        # 使用 is not None 检查，因为空对象可能布尔值为 False
        if not all([
            self.vector_store is not None,
            self.graph_store is not None,
            self.metadata_store is not None
        ]):
            from ...plugin import A_MemorixPlugin
            instances = A_MemorixPlugin.get_storage_instances()
            if instances:
                self.vector_store = self.vector_store or instances.get("vector_store")
                self.graph_store = self.graph_store or instances.get("graph_store")
                self.metadata_store = self.metadata_store or instances.get("metadata_store")

        # 设置日志前缀
        if self.message and self.message.chat_stream:
            self.log_prefix = f"[DeleteCommand-{self.message.chat_stream.stream_id}]"
        else:
            self.log_prefix = "[DeleteCommand]"

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        """执行删除命令

        Returns:
            Tuple[bool, Optional[str], int]: (是否成功, 回复消息, 拦截级别)
        """
        # 检查存储是否初始化 (使用 is not None 而非布尔值，因为空对象可能为 False)
        if not all([
            self.vector_store is not None,
            self.graph_store is not None,
            self.metadata_store is not None
        ]):
            error_msg = "❌ 知识库未初始化"
            return False, error_msg, 0

        # 获取匹配的参数
        mode = self.matched_groups.get("mode", "help")
        content = self.matched_groups.get("content", "")

        # 如果没有内容，显示帮助
        if not content and mode not in ["clear", "stats", "help"]:
            help_msg = self._get_help_message()
            return True, help_msg, 0

        logger.info(f"{self.log_prefix} 执行删除: mode={mode}, content='{content}'")

        try:
            # 根据模式执行删除
            if mode == "paragraph" or mode == "p":
                success, result = await self._delete_paragraph(content)
            elif mode == "entity" or mode == "e":
                success, result = await self._delete_entity(content)
            elif mode == "relation" or mode == "r":
                success, result = await self._delete_relation(content)
            elif mode == "clear":
                # 清空需要确认
                success, result = await self._clear_knowledge_base()
            elif mode == "stats":
                success, result = self._get_deletion_stats()
            elif mode == "help":
                success, result = True, self._get_help_message()
            else:
                success, result = False, f"❌ 未知的删除模式: {mode}"

            return success, result, 0

        except Exception as e:
            error_msg = f"❌ 删除失败: {str(e)}"
            logger.error(f"{self.log_prefix} {error_msg}")
            return False, error_msg, 0

    async def _delete_paragraph(self, hash_or_content: str) -> Tuple[bool, str]:
        """删除段落

        Args:
            hash_or_content: 段落hash或内容

        Returns:
            Tuple[bool, str]: (是否成功, 结果消息)
        """
        start_time = time.time()

        # 尝试作为hash查找
        paragraph = self.metadata_store.get_paragraph(hash_or_content)

        if not paragraph:
            # 尝试作为内容查找
            paragraphs = self.metadata_store.search_paragraphs_by_content(hash_or_content)

            if not paragraphs:
                return False, f"❌ 未找到段落: {hash_or_content[:50]}..."

            if len(paragraphs) > 1:
                # 多个匹配，列出选项
                lines = [
                    f"⚠️ 找到 {len(paragraphs)} 个匹配的段落:",
                    "",
                ]
                for i, para in enumerate(paragraphs[:5], 1):
                    content = para["content"][:60] + "..." if len(para["content"]) > 60 else para["content"]
                    hash_val = para["hash"][:16] + "..."
                    lines.append(f"  {i}. [{hash_val}] {content}")

                if len(paragraphs) > 5:
                    lines.append(f"  ... 还有 {len(paragraphs) - 5} 个")

                lines.append("")
                lines.append("💡 请使用完整的hash值精确删除")

                return True, "\n".join(lines)

            # 使用第一个匹配
            paragraph = paragraphs[0]

        hash_value = paragraph["hash"]

        # 删除段落（会级联删除相关关系和实体关联）
        success = self.metadata_store.delete_paragraph(hash_value)

        if success:
            # 从向量存储中删除
            self.vector_store.remove([hash_value])

            elapsed = time.time() - start_time
            result_lines = [
                "✅ 段落删除完成",
                f"📝 Hash: {hash_value[:16]}...",
                f"📄 内容: {paragraph['content'][:50]}...",
                f"⏱️ 耗时: {elapsed*1000:.1f}ms",
            ]
            return True, "\n".join(result_lines)
        else:
            return False, f"❌ 段落删除失败: {hash_value[:16]}..."

    async def _delete_entity(self, entity_name: str) -> Tuple[bool, str]:
        """删除实体

        Args:
            entity_name: 实体名称

        Returns:
            Tuple[bool, str]: (是否成功, 结果消息)
        """
        start_time = time.time()

        # 检查实体是否存在
        if not self.graph_store.has_node(entity_name):
            return False, f"❌ 实体不存在: {entity_name}"

        # 获取相关关系统计
        neighbors = self.graph_store.get_neighbors(entity_name)
        edge_count = len(neighbors)

        # 获取相关段落
        related_paragraphs = self.metadata_store.get_paragraphs_by_entity(entity_name)

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

            # 查找关系
            relations = self.metadata_store.get_relations(
                subject=subject.strip(),
                predicate=predicate.strip(),
                object=obj.strip(),
            )

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
