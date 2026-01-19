#!/usr/bin/env python3
"""
知识库自动导入脚本

功能：
1. 扫描 plugins/A_memorix/data/raw 下的 .txt 文件
2. 检查 data/import_manifest.json 确认是否已导入
3. 调用 LLM 处理未导入文件生成 JSON
4. 将生成的数据直接存入 VectorStore/GraphStore/MetadataStore
5. 更新 manifest

用法：无需参数，直接运行
"""

import sys
import os
import json
import asyncio
import time
import random
import hashlib
import tomlkit
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# 路径设置
current_dir = Path(__file__).resolve().parent
# 假设脚本在 plugins/A_memorix/scripts
plugin_root = current_dir.parent
project_root = plugin_root.parent.parent
sys.path.insert(0, str(project_root))

# 数据目录
DATA_DIR = plugin_root / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_PATH = DATA_DIR / "import_manifest.json"

try:
    print(f"Project root: {project_root}")
    print(f"Sys path: {sys.path[:3]}...")
    
    import src
    print("✅ src imported")
    
    import plugins
    print("✅ plugins imported")
    
    import plugins.A_memorix
    print("✅ plugins.A_memorix imported")
    
    from src.common.logger import get_logger
    from src.plugin_system.apis import llm_api
    from src.config.config import global_config, model_config
    
    # 导入核心组件
    from plugins.A_memorix.core import (
        VectorStore,
        GraphStore,
        MetadataStore,
        create_embedding_api_adapter,
        PersonalizedPageRank,
        KnowledgeType,
    )
    from plugins.A_memorix.core.storage import (
        QuantizationType, 
        SparseMatrixFormat,
        detect_knowledge_type
    )
    
except ImportError as e:
    print(f"❌ 无法导入模块: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

logger = get_logger("A_Memorix.AutoImport")

class AutoImporter:
    def __init__(self, force: bool = False, clear_manifest: bool = False, target_type: str = "auto"):
        self.vector_store: Optional[VectorStore] = None
        self.graph_store: Optional[GraphStore] = None
        self.metadata_store: Optional[MetadataStore] = None
        self.embedding_manager = None
        self.plugin_config = {}
        self.manifest = {}
        self.force = force
        self.clear_manifest = clear_manifest
        self.target_type = target_type

    async def initialize(self):
        """初始化配置和存储"""
        logger.info("正在初始化...")
        
        # 1. 确保目录存在
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        
        # 2. 加载 Manifest
        if self.clear_manifest:
            logger.info("🧹 清理 Manifest (--clear-manifest activated)")
            self.manifest = {}
            self._save_manifest()
        elif MANIFEST_PATH.exists():
            try:
                with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                    self.manifest = json.load(f)
            except Exception as e:
                logger.error(f"加载 Mainfest 失败: {e}")
                self.manifest = {}
        
        # 3. 加载插件配置
        config_path = plugin_root / "config.toml"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.plugin_config = tomlkit.load(f)
        except Exception as e:
            logger.error(f"加载插件配置失败: {e}")
            return False

        # 4. 初始化存储组件
        try:
            await self._init_stores()
        except Exception as e:
            logger.error(f"初始化存储失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        return True

    async def _init_stores(self):
        """初始化存储组件 (参考 A_MemorixPlugin)"""
        # 嵌入API
        self.embedding_manager = create_embedding_api_adapter(
            batch_size=self.plugin_config.get("embedding", {}).get("batch_size", 32),
            default_dimension=self.plugin_config.get("embedding", {}).get("dimension", 384),
            model_name=self.plugin_config.get("embedding", {}).get("model_name", "auto"),
        )
        
        # 检测维度
        try:
            dim = await self.embedding_manager._detect_dimension()
        except:
            dim = self.embedding_manager.default_dimension
            
        # 向量存储
        q_type_str = self.plugin_config.get("embedding", {}).get("quantization_type", "int8")
        q_map = {"float32": QuantizationType.FLOAT32, "int8": QuantizationType.INT8, "pq": QuantizationType.PQ}
        
        self.vector_store = VectorStore(
            dimension=dim,
            quantization_type=q_map.get(q_type_str, QuantizationType.INT8),
            data_dir=DATA_DIR / "vectors"
        )
        
        # 图存储
        m_fmt_str = self.plugin_config.get("graph", {}).get("sparse_matrix_format", "csr")
        m_map = {"csr": SparseMatrixFormat.CSR, "csc": SparseMatrixFormat.CSC}
        
        self.graph_store = GraphStore(
            matrix_format=m_map.get(m_fmt_str, SparseMatrixFormat.CSR),
            data_dir=DATA_DIR / "graph"
        )
        
        # 元数据存储
        self.metadata_store = MetadataStore(data_dir=DATA_DIR / "metadata")
        self.metadata_store.connect()
        
        # 加载数据
        if self.vector_store.has_data():
            self.vector_store.load()
        if self.graph_store.has_data():
            self.graph_store.load()

    def load_file(self, file_path: Path) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def get_file_hash(self, content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    async def process_and_import(self):
        """主处理循环"""
        if not await self.initialize():
            return

        files = list(RAW_DIR.glob("*.txt"))
        logger.info(f"扫描到 {len(files)} 个文件 in {RAW_DIR}")

        processed_count = 0
        
        for file_path in files:
            filename = file_path.name
            content = self.load_file(file_path)
            file_hash = self.get_file_hash(content)
            
            # 检查是否已处理
            if not self.force and filename in self.manifest:
                record = self.manifest[filename]
                if record.get("hash") == file_hash and record.get("imported"):
                    logger.info(f"跳过已导入文件: {filename}")
                    continue
            
            if self.force:
                logger.info(f"强制重新导入: {filename}")
            
            logger.info(f"=== 开始处理: {filename} ===")
            
            # 1. LLM 处理生成 JSON
            json_data = await self._process_text_to_json(content, filename)
            
            # 保存中间结果
            json_path = PROCESSED_DIR / f"{file_path.stem}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
                
            # 2. 导入到数据库
            await self._import_to_db(json_data)
            
            # 3. 更新 Manifest
            self.manifest[filename] = {
                "hash": file_hash,
                "timestamp": time.time(),
                "imported": True
            }
            self._save_manifest()
            
            logger.info(f"✅ 文件 {filename} 处理并导入完成")
            processed_count += 1
            
            # 保存数据库状态
            self.vector_store.save()
            self.graph_store.save()

        if processed_count == 0:
            logger.info("没有新文件需要处理")
        else:
            logger.info(f"本次共处理 {processed_count} 个文件")

    async def _select_model(self) -> Any:
        """精确选择最适合知识抽取的模型 (仅限明确配置和任务匹配)"""
        models = llm_api.get_available_models()
        if not models:
            raise ValueError("没有可用的 LLM 模型配置")

        # 1. 优先级最高：插件配置强制指定
        config_model = self.plugin_config.get("advanced", {}).get("extraction_model", "auto")
        if config_model != "auto" and config_model in models:
            logger.info(f"  使用插件配置指定的模型: {config_model}")
            return models[config_model]

        # 2. 优先级第二：主程序任务配置匹配 (lpmm_entity_extract)
        try:
            from src.config.config import model_config as host_model_config
            task_configs = getattr(host_model_config, "model_task_config", {})
            
            # 按优先级尝试两种相关的任务配置
            for task_key in ["lpmm_entity_extract", "lpmm_rdf_build"]:
                if task_key in task_configs:
                    task_models = task_configs[task_key].get("model_list", [])
                    for m in task_models:
                        if m in models:
                            logger.info(f"  通过主程序任务配置 [{task_key}] 匹配到模型: {m}")
                            return models[m]
        except Exception as e:
            logger.debug(f"读取主程序任务配置失败: {e}")

        # 3. 兜底策略：如果以上均未匹配，抛出错误引导用户配置
        logger.error("❌ 未能在主程序配置中找到合适的 [lpmm_entity_extract] 任务模型")
        logger.warning("请在 model_config.toml 的 [model_task_config.lpmm_entity_extract] 中指定模型，")
        logger.warning("或者在插件 config.toml 的 [advanced] 中设置 extraction_model")
        
        # 为了兼容性，返回首个可用模型但给出强烈警告
        first_model = list(models.keys())[0]
        logger.warning(f"由于未匹配到专用模型，被迫使用首个可用模型: {first_model}")
        return models[first_model]

    async def _process_text_to_json(self, text: str, filename: str) -> Dict:
        """调用 LLM 处理文本"""
        chunks = self._split_text(text)
        logger.info(f"  分块数量: {len(chunks)}")
        
        all_data = {"paragraphs": [], "entities": [], "relations": []}
        
        # 智能选择模型配置
        model_config = await self._select_model()
        
        for i, chunk in enumerate(chunks):
            # 添加段落
            all_data["paragraphs"].append({"content": chunk, "source": filename})
            
            # 提取信息
            result = await self._extract_info(chunk, model_config)
            
            if result.get("entities"):
                all_data["entities"].extend(result["entities"])
            if result.get("relations"):
                all_data["relations"].extend(result["relations"])
                
            logger.info(f"  已处理块 {i+1}/{len(chunks)}")
            await asyncio.sleep(0.5)
            
        # 去重
        all_data["entities"] = list(set(all_data["entities"]))
        return all_data

    async def _extract_info(self, chunk: str, model_config: Any) -> Dict:
        prompt = f"""请分析以下文本，提取其中的实体（Entities）和关系（Relations）。
仅提取关键信息。
JSON格式: {{ "entities": ["e1"], "relations": [{{"subject": "s", "predicate": "p", "object": "o"}}] }}
文本:
{chunk[:2000]}
"""
        success, response, _, _ = await llm_api.generate_with_model(
            prompt=prompt,
            model_config=model_config,
            request_type="Script.ProcessKnowledge"
        )
        if success:
            try:
                # 简单清理
                txt = response.strip()
                if "```" in txt:
                    txt = txt.split("```json")[-1].split("```")[0].strip()
                    if txt.startswith("json"): txt = txt[4:].strip()
                return json.loads(txt)
            except:
                pass
        return {}

    def _split_text(self, text: str, size=800) -> List[str]:
        # 简单按行分块
        lines = text.split("\n")
        chunks = []
        cur = ""
        for line in lines:
            if len(cur) + len(line) > size:
                chunks.append(cur)
                cur = line + "\n"
            else:
                cur += line + "\n"
        if cur: chunks.append(cur)
        return chunks

    async def _import_to_db(self, data: Dict):
        """将JSON数据导入存储"""
        # 1. 导入段落
        for item in data.get("paragraphs", []):
            content = item["content"] if isinstance(item, dict) else item
            
            # 元数据判定
            if self.target_type and self.target_type != "auto":
                from plugins.A_memorix.core.storage import get_knowledge_type_from_string
                k_type = get_knowledge_type_from_string(self.target_type) or detect_knowledge_type(content)
            else:
                k_type = detect_knowledge_type(content)
                
            h_val = self.metadata_store.add_paragraph(content, "auto_import", k_type.value)
            
            # 向量
            emb = await self.embedding_manager.encode(content)
            self.vector_store.add(emb.reshape(1, -1), [h_val])
            
        # 2. 导入实体
        entities = data.get("entities", [])
        if entities:
            self.graph_store.add_nodes(entities)
            
        # 3. 导入关系
        for rel in data.get("relations", []):
            s, p, o = rel.get("subject"), rel.get("predicate"), rel.get("object")
            if s and p and o:
                # 这里的add_edges会自动add_nodes，但为了安全先保证nodes存在
                self.graph_store.add_nodes([s, o])
                
                # 添加到图
                self.graph_store.add_edges([(s, o)])
                
                # 添加到元数据（如果需要关系元数据支持）
                self.metadata_store.add_relation(s, p, o)

    def _save_manifest(self):
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)

async def main():
    quotes = [
        "記憶の中に居た,温もりが側にいて",
        "幸福の切れ端を,繋いでいた。願っていた"
    ]
    logger.info(random.choice(quotes))  # Runtime Easter Egg
    
    parser = argparse.ArgumentParser(description="A_Memorix 知识库自动导入工具")
    parser.add_argument("--force", action="store_true", help="强制重新导入所有文件，忽略已导入记录")
    parser.add_argument("--clear-manifest", action="store_true", help="处理前清空导入历史记录")
    parser.add_argument("--type", "-t", choices=["structured", "narrative", "factual", "mixed", "auto"], default="auto", help="强制指定所有导入文件的知识类型")
    args = parser.parse_args()

    if not global_config or not model_config:
        logger.error("全局配置未加载")
        return
        
    importer = AutoImporter(force=args.force, clear_manifest=args.clear_manifest, target_type=args.type)
    await importer.process_and_import()

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
