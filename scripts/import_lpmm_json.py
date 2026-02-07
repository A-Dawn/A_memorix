#!/usr/bin/env python3
"""
LPMM OpenIE JSON 导入工具

功能：
1. 读取符合 LPMM 规范的 OpenIE JSON 文件 (通常命名为 *-openie.json)
2. 将其转换为 A_memorix 插件所需的中间格式
   - passage -> content
   - extracted_triples [s, p, o] -> relations {subject, predicate, object}
   - idx -> hash (如果匹配) 或重新计算
3. 调用 AutoImporter 直接入库

用法：
    python import_lpmm_json.py <json_file_or_dir>
"""

import sys
import os
import json
import argparse
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from rich.console import Console

console = Console()

# 设置环境路径以复用 process_knowledge.py 中的 AutoImporter
current_dir = Path(__file__).resolve().parent
plugin_root = current_dir.parent
sys.path.insert(0, str(plugin_root))
# 为了引用更上层的 src
project_root = plugin_root.parent.parent
sys.path.insert(0, str(project_root))

try:
    from scripts.process_knowledge import AutoImporter
    # 尝试导入 hash 工具
    sys.path.insert(0, str(plugin_root))
    from core.utils.hash import compute_paragraph_hash
    from src.common.logger import get_logger
except ImportError as e:
    print(f"导入模块失败，请确保在 plugins/A_memorix/scripts 目录下运行或设置了正确的 PYTHONPATH: {e}")
    sys.exit(1)

logger = get_logger("A_Memorix.LPMM_Import")


class LPMMConverter:
    def __init__(self):
        pass

    def convert_lpmm_to_memorix(self, lpmm_data: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """
        将 LPMM 格式转换为 A_memorix 格式
        LPMM Root: { "docs": [ ... ], "avg_...": ... }
        Memorix: { "paragraphs": [ ... ], "entities": [ ... ] }
        """
        memorix_data = {
            "paragraphs": [],
            "entities": [] # 用于去重
        }

        docs = lpmm_data.get("docs", [])
        if not docs:
            logger.warning(f"文件中未找到 'docs' 字段: {filename}")
            return memorix_data

        for doc in docs:
            content = doc.get("passage", "")
            if not content:
                continue

            # 处理 ID / Hash
            # LPMM 的 idx 通常就是 hash，但为了保险起见，我们重新计算或者校验
            # 这里我们优先信任 content 计算出的 hash，为了和 A_memorix 自身的逻辑保持一致
            p_hash = compute_paragraph_hash(content)
            
            # 处理关系: [s, p, o] -> {s, p, o}
            triples = doc.get("extracted_triples", [])
            relations = []
            for triple in triples:
                if isinstance(triple, list) and len(triple) == 3:
                    relations.append({
                        "subject": triple[0],
                        "predicate": triple[1],
                        "object": triple[2]
                    })
            
            entities = doc.get("extracted_entities", [])

            paragraph_item = {
                "hash": p_hash,
                "content": content,
                "source": filename, # 标记来源文件
                "entities": entities,
                "relations": relations
            }
            
            memorix_data["paragraphs"].append(paragraph_item)
            memorix_data["entities"].extend(entities)

        # 简单去重 entities
        memorix_data["entities"] = list(set(memorix_data["entities"]))
        
        return memorix_data


async def main():
    parser = argparse.ArgumentParser(description="将 LPMM OpenIE JSON 导入 A_memorix")
    parser.add_argument("path", help="LPMM JSON 文件路径或包含这些文件的目录")
    parser.add_argument("--force", action="store_true", help="强制重新导入")
    parser.add_argument("--concurrency", "-c", type=int, default=5, help="并发数")
    
    args = parser.parse_args()
    
    target_path = Path(args.path)
    if not target_path.exists():
        logger.error(f"路径不存在: {target_path}")
        return

    files_to_process = []
    if target_path.is_dir():
        files_to_process = list(target_path.glob("*-openie.json"))
        # 如果找不到特定后缀，尝试所有 .json
        if not files_to_process:
             files_to_process = list(target_path.glob("*.json"))
    else:
        files_to_process = [target_path]

    if not files_to_process:
        logger.error("未找到可处理的 JSON 文件")
        return

    logger.info(f"找到 {len(files_to_process)} 个文件待处理...")

    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

    # 初始化导入器
    importer = AutoImporter(force=args.force, concurrency=args.concurrency)
    # 初始化存储 (这一步很重要)
    if not await importer.initialize():
        logger.error("初始化存储失败")
        return

    converter = LPMMConverter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console, # 确保复用全局 console
        transient=False  # 完成后保留
    ) as progress:
        # 主任务（针对文件列表）并不是很有意义，我们针对具体条目
        # 总进度条 (如果多个文件，这里可以是一个总览，或者我们为每个文件建一个 Task)
        
        for json_file in files_to_process:
            logger.info(f"正在转换并导入: {json_file.name}")
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    lpmm_data = json.load(f)
                
                # 1. 转换格式
                memorix_data = converter.convert_lpmm_to_memorix(lpmm_data, json_file.name)
                
                total_items = len(memorix_data.get("paragraphs", []))
                if not total_items:
                    logger.warning(f"  转换结果为空: {json_file.name}")
                    continue

                # 创建该文件的进度任务
                task_id = progress.add_task(f"Importing {json_file.name}", total=total_items)

                # 回调函数：每完成一个item调用一次
                def update_progress(n=1):
                    progress.advance(task_id, advance=n)

                # 2. 调用导入器接口
                await importer.import_json_data(
                    memorix_data, 
                    filename=f"lpmm_{json_file.name}",
                    progress_callback=update_progress
                )
                
            except Exception as e:
                logger.error(f"处理文件 {json_file.name} 失败: {e}")
                import traceback
                traceback.print_exc()

    logger.info("全部处理完成")
    await importer.close()

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
