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
    
    # 动态计算插件名称 (假设脚本位于 plugins/<plugin_name>/scripts/)
    script_path = Path(__file__).resolve()
    plugin_dir = script_path.parent.parent
    plugin_name = plugin_dir.name
    
    import importlib
    
    # 确保 plugins 包已加载
    try:
        if f"plugins.{plugin_name}" not in sys.modules:
            importlib.import_module(f"plugins.{plugin_name}")
        print(f"✅ plugins.{plugin_name} imported")
    except ImportError as e:
        print(f"⚠️ Could not import plugins.{plugin_name}: {e}")

    from src.common.logger import get_logger
    from src.plugin_system.apis import llm_api
    from src.config.config import global_config, model_config
    
    # 动态导入核心组件
    core_module = importlib.import_module(f"plugins.{plugin_name}.core")
    VectorStore = core_module.VectorStore
    GraphStore = core_module.GraphStore
    MetadataStore = core_module.MetadataStore
    create_embedding_api_adapter = core_module.create_embedding_api_adapter
    PersonalizedPageRank = core_module.PersonalizedPageRank
    KnowledgeType = core_module.KnowledgeType

    storage_module = importlib.import_module(f"plugins.{plugin_name}.core.storage")
    QuantizationType = storage_module.QuantizationType
    SparseMatrixFormat = storage_module.SparseMatrixFormat
    detect_knowledge_type = storage_module.detect_knowledge_type
    
except ImportError as e:
    print(f"❌ 无法导入模块: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

logger = get_logger("A_Memorix.AutoImport")

class AutoImporter:
    def __init__(self, force: bool = False, clear_manifest: bool = False, target_type: str = "auto", concurrency: int = 5):
        self.vector_store: Optional[VectorStore] = None
        self.graph_store: Optional[GraphStore] = None
        self.metadata_store: Optional[MetadataStore] = None
        self.embedding_manager = None
        self.plugin_config = {}
        self.manifest = {}
        self.force = force
        self.clear_manifest = clear_manifest
        self.target_type = target_type
        
        # 并发控制
        self.concurrency_limit = concurrency
        self.semaphore = None
        self.storage_lock = None

    async def initialize(self):
        """初始化配置和存储"""
        logger.info(f"正在初始化... (并发数: {self.concurrency_limit})")
        
        # 初始化并发原语
        self.semaphore = asyncio.Semaphore(self.concurrency_limit)
        self.storage_lock = asyncio.Lock()
        
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

    # ... (store initialization remains same) ...

    async def process_and_import(self):
        """主处理循环 (并行版)"""
        if not await self.initialize():
            return

        files = list(RAW_DIR.glob("*.txt"))
        logger.info(f"扫描到 {len(files)} 个文件 in {RAW_DIR}")

        if not files:
            logger.info("没有新文件需要处理")
            return

        # 创建任务列表
        tasks = []
        for file_path in files:
            task = asyncio.create_task(self._process_single_file(file_path))
            tasks.append(task)
            
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 统计结果
        success_count = 0
        for res in results:
            if res is True:
                success_count += 1
            elif isinstance(res, Exception):
                logger.error(f"任务异常: {res}")
        
        logger.info(f"本次主处理完成，共成功处理 {success_count}/{len(files)} 个文件")
        
        # 最后再一次保存确保安全
        if self.vector_store: self.vector_store.save()
        if self.graph_store: self.graph_store.save()

    async def _process_single_file(self, file_path: Path) -> bool:
        """处理单个文件的流程 (受信号量控制)"""
        filename = file_path.name
        
        # 1. 获取信号量 (限制并发 LLM 调用)
        async with self.semaphore:
            try:
                content = self.load_file(file_path)
                file_hash = self.get_file_hash(content)
                
                # 检查是否已处理 (快速检查，无需锁)
                if not self.force and filename in self.manifest:
                    record = self.manifest[filename]
                    if record.get("hash") == file_hash and record.get("imported"):
                        logger.info(f"跳过已导入文件: {filename}")
                        return False
                
                if self.force:
                    logger.info(f"强制重新导入: {filename}")
                
                logger.info(f">>> 开始处理: {filename}")
                
                # 2. LLM 处理生成 JSON (耗时操作，并发执行)
                # 注意：这里可能会有大量的 LLM 请求
                json_data = await self._process_text_to_json(content, filename)
                
                # HACK: 将文件内容嵌入到 json_data 中，以便 _import_to_db 使用 (如果需要)
                # 实际上 _import_to_db 主要用 content 存段落，json_data["paragraphs"] 里已经有了
                
                # 保存中间结果 (IO操作，相对快，暂不锁)
                json_path = PROCESSED_DIR / f"{file_path.stem}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)

                # 3. 导入到数据库 (写操作，必须加锁串行化)
                async with self.storage_lock:
                    logger.info(f"🔒 正在写入数据库: {filename}")
                    try:
                        await self._import_to_db(json_data)

                        # 更新 Manifest
                        self.manifest[filename] = {
                            "hash": file_hash,
                            "timestamp": time.time(),
                            "imported": True
                        }
                        self._save_manifest()
                        
                        # 每次成功处理后保存一次，避免崩溃丢失全部
                        # 考虑到性能，可以改为每N个保存一次，或者就保持这样安全性高
                        self.vector_store.save()
                        self.graph_store.save()
                        
                        logger.info(f"✅ 文件 {filename} 处理并导入完成")
                        return True

                    except Exception as e:
                        logger.error(f"❌ 导入数据库失败 {filename}: {e}")
                        import traceback
                        traceback.print_exc()

                        self.manifest[filename] = {
                            "hash": file_hash,
                            "timestamp": time.time(),
                            "imported": False,
                            "error": str(e)
                        }
                        self._save_manifest()
                        # 即使失败也算处理结束
                        return False

            except Exception as e:
                logger.error(f"处理文件 {filename} 时发生未捕获异常: {e}")
                import traceback
                traceback.print_exc()
                return False

    async def _select_model(self) -> Any:
        """精确选择最适合知识抽取的模型 (返回 TaskConfig)"""
        models = llm_api.get_available_models()
        if not models:
            raise ValueError("没有可用的 LLM 模型配置")

        # 1. 优先级最高：插件配置强制指定（支持任务名称）
        config_model = self.plugin_config.get("advanced", {}).get("extraction_model", "auto")

        # 如果指定了任务名称（如 "lpmm_entity_extract"），直接使用
        if config_model != "auto" and config_model in models:
            logger.info(f"  使用插件配置指定的任务: {config_model}")
            return models[config_model]

        # 2. 优先级第二：默认使用 lpmm_entity_extract 任务
        for task_key in ["lpmm_entity_extract", "lpmm_rdf_build", "embedding"]:
            if task_key in models:
                logger.info(f"  使用主程序任务配置: {task_key}")
                task_cfg = models[task_key]
                logger.info(f"    模型列表: {task_cfg.model_list}")
                return models[task_key]

        # 3. 兜底策略：使用第一个可用任务
        first_task = list(models.keys())[0]
        logger.warning(f"⚠️ 未找到实体抽取专用任务，使用任务: {first_task}")
        return models[first_task]

    async def _process_text_to_json(self, text: str, filename: str) -> Dict:
        """调用 LLM 处理文本"""
        chunks = self._split_text(text)
        logger.info(f"  分块数量: {len(chunks)}")
        
        all_data = {"paragraphs": [], "entities": [], "relations": []}
        
        # 智能选择模型配置
        model_config = await self._select_model()
        
        for i, chunk in enumerate(chunks):
            # 提取信息
            result = await self._extract_info(chunk, model_config)
            
            # 记录段落及其关联的知识
            paragraph_item = {
                "content": chunk,
                "source": filename,
                "entities": result.get("entities", []),
                "relations": result.get("relations", [])
            }
            all_data["paragraphs"].append(paragraph_item)
            
            # 同时也维护平铺的实体列表以便去重
            if result.get("entities"):
                all_data["entities"].extend(result["entities"])
                
            logger.info(f"  已处理块 {i+1}/{len(chunks)}")
            await asyncio.sleep(0.2)

        # 去重实体（支持字符串和字典格式）
        def dedupe_entities(entities):
            seen = set()
            unique = []
            for e in entities:
                key = e if isinstance(e, str) else json.dumps(e, sort_keys=True, ensure_ascii=False)
                if key not in seen:
                    seen.add(key)
                    unique.append(e)
            return unique

        all_data["entities"] = dedupe_entities(all_data["entities"])
        return all_data

    async def _extract_info(self, chunk: str, model_config: Any) -> Dict:
        prompt = f"""请分析以下文本，提取其中的实体（Entities）和关系（Relations）。
仅提取关键或重要的信息。实体名称应尽可能完整。
不要使用 e1, e2 等占位符作为实体名，直接使用实体的实际名称。

JSON格式示例:
{{
  "entities": ["梅露可", "图图"],
  "relations": [
    {{"subject": "梅露可", "predicate": "伙伴", "object": "图图"}}
  ]
}}

文本内容:
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

    async def _add_entity_with_vector(self, name: str, source_paragraph: Optional[str] = None) -> str:
        """添加实体并在向量库中生成索引"""
        # 1. 存入元数据和图存储
        hash_value = self.metadata_store.add_entity(name, source_paragraph=source_paragraph)
        self.graph_store.add_nodes([name])

        # 2. 生成向量并存入向量库
        try:
            emb = await self.embedding_manager.encode(name)
            try:
                self.vector_store.add(emb.reshape(1, -1), [hash_value])
            except ValueError:
                # 忽略已存在的ID
                pass
        except Exception as e:
            logger.warning(f"  [Error] Failed to vectorize entity {name}: {e}")

        return hash_value

    async def _import_to_db(self, data: Dict):
        """将JSON数据导入存储"""
        # 使用批量更新模式优化图存储性能 (避免 CSR 警告)
        # 注意: batch_update 是同步上下文管理器，不影响 async await
        with self.graph_store.batch_update():
            # 1. 按段落导入及其关联的知识
            for item in data.get("paragraphs", []):
                content = item["content"] if isinstance(item, dict) else item
                source = item.get("source", "script") if isinstance(item, dict) else "script"
                
                # 元数据判定
                if self.target_type and self.target_type != "auto":
                    # 动态导入 get_knowledge_type_from_string
                    plugin_name = self.plugin_config.get("plugin", {}).get("name", "A_memorix") # Fallback name from config or path
                    # Better to reuse the plugin_name calculated at module level, but we are in a method. 
                    # Let's re-calculate or assume module level variable is available if we made it global, 
                    # but here we can just use relative import logic since we know the structure or importlib again.
                    # Actually, `storage_module` from global scope is not easily accessible here unless passed.
                    
                    # Re-calculate cleanly
                    script_path = Path(__file__).resolve()
                    plugin_name = script_path.parent.parent.name
                    storage_mod = importlib.import_module(f"plugins.{plugin_name}.core.storage")
                    get_knowledge_type_from_string = storage_mod.get_knowledge_type_from_string

                    k_type = get_knowledge_type_from_string(self.target_type) or detect_knowledge_type(content)
                else:
                    k_type = detect_knowledge_type(content)
                    
                h_val = self.metadata_store.add_paragraph(content, source, k_type.value)
                
                # 向量
                emb = await self.embedding_manager.encode(content)
                self.vector_store.add(emb.reshape(1, -1), [h_val])
                
                # 导入该段落关联的实体 (确保存在)
                para_entities = item.get("entities", []) if isinstance(item, dict) else []
                for entity in para_entities:
                    await self._add_entity_with_vector(entity, source_paragraph=h_val)
                    
                # 导入该段落关联的关系 (关键：传入 h_val)
                para_relations = item.get("relations", []) if isinstance(item, dict) else []
                for rel in para_relations:
                    s, p, o = rel.get("subject"), rel.get("predicate"), rel.get("object")
                    if s and p and o:
                        await self._add_entity_with_vector(s, source_paragraph=h_val)
                        await self._add_entity_with_vector(o, source_paragraph=h_val)
                        
                        self.graph_store.add_edges([(s, o)])
                        # 传入 source_paragraph 哈希
                        self.metadata_store.add_relation(s, p, o, source_paragraph=h_val)

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
    parser.add_argument("--concurrency", "-c", type=int, default=5, help="LLM 并发请求数量限制 (默认: 5)")
    args = parser.parse_args()

    if not global_config or not model_config:
        logger.error("全局配置未加载")
        return
        
    importer = AutoImporter(
        force=args.force, 
        clear_manifest=args.clear_manifest, 
        target_type=args.type,
        concurrency=args.concurrency
    )
    await importer.process_and_import()

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
