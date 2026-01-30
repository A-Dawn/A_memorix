#!/usr/bin/env python3
"""
知识库批量删除工具

功能：
1. 列出当前所有知识来源（文件）
2. 按来源（文件名）批量删除相关知识（级联删除段落、实体、关系）

用法：
    python delete_knowledge.py --list
    python delete_knowledge.py --source "filename.txt"
"""

import sys
import os
import argparse
import asyncio
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

# 路径设置
current_dir = Path(__file__).resolve().parent
plugin_root = current_dir.parent
project_root = plugin_root.parent.parent
sys.path.insert(0, str(project_root))

try:
    import src
    import plugins
    
    # 动态导入核心组件
    plugin_name = plugin_root.name
    import importlib
    
    core_module = importlib.import_module(f"plugins.{plugin_name}.core")
    VectorStore = core_module.VectorStore
    GraphStore = core_module.GraphStore
    MetadataStore = core_module.MetadataStore
    
    from src.common.logger import get_logger
    from src.config.config import global_config
    
except ImportError as e:
    console.print(f"[bold red]无法导入模块:[/bold red] {e}")
    sys.exit(1)

logger = get_logger("A_Memorix.DeleteTool")

class KnowledgeDeleter:
    def __init__(self):
        self.data_dir = plugin_root / "data"
        self.metadata_store = None
        self.vector_store = None
        self.graph_store = None
        
    def initialize(self):
        """初始化存储"""
        console.print("[dim]正在初始化存储组件...[/dim]")
        
        # 1. MetadataStore
        self.metadata_store = MetadataStore(data_dir=self.data_dir / "metadata")
        self.metadata_store.connect()
        
        # 2. VectorStore
        # 我们需要加载它以便能够按 ID 删除
        self.vector_store = VectorStore(
            dimension=1, # 维度不重要，只要能加载 ID 映射即可
            data_dir=self.data_dir / "vectors"
        )
        if self.vector_store.has_data():
            self.vector_store.load()
            
        # 3. GraphStore
        self.graph_store = GraphStore(
            data_dir=self.data_dir / "graph"
        )
        if self.graph_store.has_data():
            self.graph_store.load()
            
    def list_sources(self):
        """列出所有来源"""
        sources = self.metadata_store.get_all_sources()
        
        if not sources:
            console.print("[yellow]知识库中没有任何来源记录。[/yellow]")
            return
            
        table = Table(title="已导入知识来源")
        table.add_column("来源 (文件)", style="cyan")
        table.add_column("段落数量", justify="right", style="green")
        table.add_column("最后更新时间", style="magenta")
        
        for s in sources:
            import datetime
            ts = s.get('last_updated')
            date_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') if ts else "N/A"
            table.add_row(
                str(s['source']), 
                str(s['count']), 
                date_str
            )
            
        console.print(table)
        
    def delete_source(self, source_name: str, skip_confirm: bool = False):
        """删除指定来源"""
        # 1. 检查是否存在
        paragraphs = self.metadata_store.get_paragraphs_by_source(source_name)
        if not paragraphs:
            console.print(f"[red]未找到来源 '{source_name}' 的任何数据。[/red]")
            return
            
        count = len(paragraphs)
        console.print(f"找到 [bold green]{count}[/bold green] 个相关段落。")
        
        if not skip_confirm:
            if input(f"⚠️  确定要删除 '{source_name}' 及其所有关联数据吗？ (y/N): ").lower() != 'y':
                console.print("操作已取消。")
                return

        with console.status(f"正在删除 '{source_name}'...", spinner="dots"):
            deleted_count = 0
            errors = []
            
            for p in paragraphs:
                try:
                    # 原子删除
                    cleanup_plan = self.metadata_store.delete_paragraph_atomic(p['hash'])
                    
                    # 清理向量
                    vec_id = cleanup_plan.get("vector_id_to_remove")
                    if vec_id:
                        try:
                            self.vector_store.delete([vec_id])
                        except Exception:
                            pass
                            
                    # 清理图边
                    edges = cleanup_plan.get("edges_to_remove", [])
                    if edges:
                        try:
                            self.graph_store.delete_edges(edges)
                        except Exception:
                            pass
                            
                    deleted_count += 1
                except Exception as e:
                    errors.append(str(e))
            
            # 保存
            self.vector_store.save()
            self.graph_store.save()
            
        if errors:
            console.print(f"[yellow]删除完成，但有 {len(errors)} 个错误。[/yellow]")
        else:
            console.print(f"[bold green]✅ 成功删除 '{source_name}' 相关的所有 {deleted_count} 条数据。[/bold green]")

    def close(self):
        if self.metadata_store:
            self.metadata_store.close()

def main():
    parser = argparse.ArgumentParser(description="A_Memorix 知识库批量删除工具")
    parser.add_argument("--list", action="store_true", help="列出所有知识来源文件")
    parser.add_argument("--source", type=str, help="指定要删除的来源名称 (文件名)")
    parser.add_argument("--yes", "-y", action="store_true", help="跳过确认提示")
    
    args = parser.parse_args()
    
    deleter = KnowledgeDeleter()
    try:
        deleter.initialize()
        
        if args.list:
            deleter.list_sources()
        elif args.source:
            deleter.delete_source(args.source, args.yes)
        else:
            parser.print_help()
            
    finally:
        deleter.close()

if __name__ == "__main__":
    main()
