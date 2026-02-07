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
import pickle
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
        self.vector_dimension = 384

    def _detect_vector_dimension(self) -> int:
        """从向量元数据读取真实维度，避免用错误维度写回。"""
        meta_path = self.data_dir / "vectors" / "vectors_metadata.pkl"
        if not meta_path.exists():
            return 384

        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            dim = int(meta.get("dimension", 384))
            if dim > 0:
                return dim
        except Exception:
            pass
        return 384

    def _collect_paragraph_entities(self, paragraph_hash: str):
        """收集段落实体（hash -> name），用于后续保守孤儿清理。"""
        candidates = {}
        try:
            entities = self.metadata_store.get_paragraph_entities(paragraph_hash)
        except Exception:
            return candidates

        for ent in entities:
            h = ent.get("hash")
            n = ent.get("name")
            if h and n:
                candidates[h] = n
        return candidates

    def _is_entity_still_referenced(self, entity_hash: str, entity_name: str) -> bool:
        """判断实体是否仍被任何有效数据引用。"""
        # 1) 段落实体关联仍存在
        if self.metadata_store.query(
            "SELECT 1 FROM paragraph_entities WHERE entity_hash = ? LIMIT 1",
            (entity_hash,),
        ):
            return True

        # 2) 关系仍引用 (subject/object) - 使用大小写不敏感匹配
        canon_name = str(entity_name).strip().lower()
        if canon_name and self.metadata_store.query(
            """
            SELECT 1
            FROM relations
            WHERE LOWER(TRIM(subject)) = ? OR LOWER(TRIM(object)) = ?
            LIMIT 1
            """,
            (canon_name, canon_name),
        ):
            return True

        # 3) 图中仍有邻居
        try:
            if self.graph_store.get_neighbors(entity_name):
                return True
        except Exception:
            pass

        return False

    def _cleanup_orphan_entities(self, candidate_entities):
        """
        清理孤儿实体（保守策略）
        仅处理本次删除波及的候选实体，且必须确认完全无引用才删除。
        """
        removed = 0
        skipped = 0

        for entity_hash, entity_name in candidate_entities.items():
            if self._is_entity_still_referenced(entity_hash, entity_name):
                skipped += 1
                continue

            try:
                self.metadata_store.delete_entity(entity_hash)
            except Exception:
                skipped += 1
                continue

            try:
                self.vector_store.delete([entity_hash])
            except Exception:
                pass

            try:
                self.graph_store.delete_nodes([entity_name])
            except Exception:
                pass

            removed += 1

        return removed, skipped
        
    def initialize(self):
        """初始化存储"""
        console.print("[dim]正在初始化存储组件...[/dim]")
        
        # 1. MetadataStore
        self.metadata_store = MetadataStore(data_dir=self.data_dir / "metadata")
        self.metadata_store.connect()
        
        # 2. VectorStore
        # 我们需要加载它以便能够按 ID 删除
        self.vector_dimension = self._detect_vector_dimension()
        self.vector_store = VectorStore(
            dimension=self.vector_dimension,
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
            candidate_entities = {}
            relation_prune_ops = []
            fallback_edges = set()
            
            for p in paragraphs:
                try:
                    # 先收集候选实体，避免段落删除后无法回溯。
                    candidate_entities.update(self._collect_paragraph_entities(p["hash"]))

                    # 原子删除
                    cleanup_plan = self.metadata_store.delete_paragraph_atomic(p['hash'])
                    
                    # 清理向量
                    vec_id = cleanup_plan.get("vector_id_to_remove")
                    if vec_id:
                        try:
                            self.vector_store.delete([vec_id])
                        except Exception:
                            pass
                    
                    # 累积图清理计划（优先 relation hash 精准裁剪）
                    for op in cleanup_plan.get("relation_prune_ops", []):
                        relation_prune_ops.append(op)

                    for edge in cleanup_plan.get("edges_to_remove", []):
                        fallback_edges.add(tuple(edge))
                            
                    deleted_count += 1
                except Exception as e:
                    errors.append(str(e))

            # 图清理：优先精准裁剪，回退到边级删除
            try:
                if relation_prune_ops and hasattr(self.graph_store, "prune_relation_hashes"):
                    self.graph_store.prune_relation_hashes(relation_prune_ops)
                elif fallback_edges:
                    self.graph_store.delete_edges(list(fallback_edges))
            except Exception:
                pass

            # 保守清理孤儿实体
            removed_entities, skipped_entities = self._cleanup_orphan_entities(candidate_entities)
            
            # 保存
            self.vector_store.save()
            self.graph_store.save()
            
        if errors:
            console.print(f"[yellow]删除完成，但有 {len(errors)} 个错误。[/yellow]")
        else:
            console.print(
                f"[bold green]✅ 成功删除 '{source_name}' 相关的 {deleted_count} 条段落，"
                f"清理孤儿实体 {removed_entities} 个（保留 {skipped_entities} 个仍被引用实体）。[/bold green]"
            )

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
