"""
聊天总结与知识导入工具

该模块负责从聊天记录中提取信息，生成总结，并将总结内容及提取的实体/关系
导入到 A_memorix 的存储组件中。
"""

import time
import json
import re
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from src.common.logger import get_logger
from src.plugin_system.apis import llm_api, message_api
from src.chat.utils.prompt_builder import global_prompt_manager, Prompt
from src.config.config import global_config

from ..storage import (
    VectorStore,
    GraphStore,
    MetadataStore,
    KnowledgeType,
    get_knowledge_type_from_string
)
from ..embedding import EmbeddingAPIAdapter

logger = get_logger("A_Memorix.SummaryImporter")

# 默认总结提示词模版
SUMMARY_PROMPT_TEMPLATE = """
你是 {bot_name}。{personality_context}
现在你需要对以下一段聊天记录进行总结，并提取其中的重要知识。

聊天记录内容：
{chat_history}

请完成以下任务：
1. **生成总结**：以第三人称或机器人的视角，简洁明了地总结这段对话的主要内容、发生的事件或讨论的主题。
2. **提取实体与关系**：识别并提取对话中提到的重要实体以及它们之间的关系。

请严格以 JSON 格式输出，格式如下：
{{
  "summary": "总结文本内容",
  "entities": ["张三", "李四"],
  "relations": [
    {{"subject": "张三", "predicate": "认识", "object": "李四"}}
  ]
}}

注意：总结应具有叙事性，能够作为长程记忆的一部分。直接使用实体的实际名称，不要使用 e1/e2 等代号。
"""

class SummaryImporter:
    """总结并导入知识的工具类"""

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        metadata_store: MetadataStore,
        embedding_manager: EmbeddingAPIAdapter,
        plugin_config: dict
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.metadata_store = metadata_store
        self.embedding_manager = embedding_manager
        self.plugin_config = plugin_config

    async def import_from_stream(
        self,
        stream_id: str,
        context_length: Optional[int] = None,
        include_personality: Optional[bool] = None
    ) -> Tuple[bool, str]:
        """
        从指定的聊天流中提取记录并执行总结导入

        Args:
            stream_id: 聊天流 ID
            context_length: 总结的历史消息条数
            include_personality: 是否包含人设

        Returns:
            Tuple[bool, str]: (是否成功, 结果消息)
        """
        try:
            # 1. 获取配置
            if context_length is None:
                context_length = self.plugin_config.get("summarization", {}).get("context_length", 50)
            
            if include_personality is None:
                include_personality = self.plugin_config.get("summarization", {}).get("include_personality", True)

            # 2. 获取历史消息
            # 获取当前时间之前的消息
            now = time.time()
            messages = message_api.get_messages_before_time_in_chat(
                chat_id=stream_id,
                timestamp=now,
                limit=context_length
            )

            if not messages:
                return False, "未找到有效的聊天记录进行总结"

            # 转换为可读文本
            chat_history_text = message_api.build_readable_messages_to_str(messages)
            
            # 3. 准备提示词内容
            bot_name = global_config.bot.nickname or "机器人"
            personality_context = ""
            if include_personality:
                personality = getattr(global_config.bot, "personality", "")
                if personality:
                    personality_context = f"你的性格设定是：{personality}"

            # 4. 调用 LLM
            prompt = SUMMARY_PROMPT_TEMPLATE.format(
                bot_name=bot_name,
                personality_context=personality_context,
                chat_history=chat_history_text
            )

            model_name = self.plugin_config.get("summarization", {}).get("model_name", "auto")
            
            # 获取可用模型并匹配
            available_models = llm_api.get_available_models()
            model_config_to_use = None
            if model_name in available_models:
                model_config_to_use = available_models[model_name]
            elif "balanced" in available_models:
                model_config_to_use = available_models["balanced"]
            elif available_models:
                model_config_to_use = list(available_models.values())[0]

            logger.info(f"正在为流 {stream_id} 执行总结，消息条数: {len(messages)}")

            success, response, _, _ = await llm_api.generate_with_model(
                prompt=prompt,
                model_config=model_config_to_use,
                request_type="A_Memorix.ChatSummarization"
            )

            if not success or not response:
                return False, "LLM 生成总结失败"

            # 5. 解析结果
            data = self._parse_llm_response(response)
            if not data or "summary" not in data:
                return False, "解析 LLM 响应失败或总结为空"

            summary_text = data["summary"]
            entities = data.get("entities", [])
            relations = data.get("relations", [])

            # 6. 执行导入
            await self._execute_import(summary_text, entities, relations, stream_id)

            # 7. 持久化
            self.vector_store.save()
            self.graph_store.save()

            result_msg = (
                f"✅ 总结导入成功\n"
                f"📝 总结长度: {len(summary_text)}\n"
                f"📌 提取实体: {len(entities)}\n"
                f"🔗 提取关系: {len(relations)}"
            )
            return True, result_msg

        except Exception as e:
            logger.error(f"总结导入过程中出错: {e}", exc_info=True)
            return False, f"错误: {str(e)}"

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """解析 LLM 返回的 JSON"""
        try:
            # 尝试查找 JSON
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {}
        except Exception as e:
            logger.warning(f"解析总结 JSON 失败: {e}")
            return {}

    async def _execute_import(
        self,
        summary: str,
        entities: List[str],
        relations: List[Dict[str, str]],
        stream_id: str
    ):
        """将数据写入存储"""
        # 获取默认知识类型
        type_str = self.plugin_config.get("summarization", {}).get("default_knowledge_type", "narrative")
        knowledge_type = get_knowledge_type_from_string(type_str) or KnowledgeType.NARRATIVE

        # 导入总结文本
        hash_value = self.metadata_store.add_paragraph(
            content=summary,
            source=f"chat_summary:{stream_id}",
            knowledge_type=knowledge_type.value
        )

        embedding = await self.embedding_manager.encode(summary)
        self.vector_store.add(
            vectors=embedding.reshape(1, -1),
            ids=[hash_value]
        )

        # 导入实体
        if entities:
            self.graph_store.add_nodes(entities)

        # 导入关系
        for rel in relations:
            s, p, o = rel.get("subject"), rel.get("predicate"), rel.get("object")
            if all([s, p, o]):
                # 写入元数据
                rel_hash = self.metadata_store.add_relation(
                    subject=s,
                    predicate=p,
                    obj=o,
                    confidence=1.0,
                    source_paragraph=summary
                )
                # 写入图数据库
                self.graph_store.add_edges([(s, o)])
                
        logger.info(f"总结导入完成: hash={hash_value[:8]}")
