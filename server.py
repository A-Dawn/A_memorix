
import asyncio
import threading
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from src.common.logger import get_logger

logger = get_logger("A_Memorix.Server")

class EdgeWeightUpdate(BaseModel):
    source: str
    target: str
    weight: float

class NodeDelete(BaseModel):
    node_id: str

class EdgeDelete(BaseModel):
    source: str
    target: str

class NodeCreate(BaseModel):
    node_id: str
    label: Optional[str] = None

class EdgeCreate(BaseModel):
    source: str
    target: str
    weight: float = 1.0
    predicate: Optional[str] = None

class NodeRename(BaseModel):
    old_id: str
    new_id: str

class AutoSaveConfig(BaseModel):
    enabled: bool

class SourceListRequest(BaseModel):
    node_id: Optional[str] = None
    edge_source: Optional[str] = None
    edge_target: Optional[str] = None

class SourceDeleteRequest(BaseModel):
    paragraph_hash: str

class BatchSourceDeleteRequest(BaseModel):
    source: str

class MemorixServer:
    def __init__(self, plugin_instance, host="0.0.0.0", port=8082):
        self.plugin = plugin_instance
        self.host = host
        self.port = port
        self.app = FastAPI(title="A_Memorix 可视化编辑器")
        self.server_thread = None
        self._server = None
        self.should_exit = False
        
        # 配置 CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()

    def _setup_routes(self):
        
        @self.app.get("/api/graph")
        async def get_graph(exclude_leaf: bool = False, source: Optional[str] = None, density: float = 1.0):
            """获取图谱数据，支持过滤叶子节点、来源及信息密度控制"""
            
            # --- 分支 1: 按来源过滤 (Batch Filtering) ---
            if source:
                if self.plugin.metadata_store is None:
                    raise HTTPException(status_code=503, detail="Metadata store not initialized")
                
                try:
                    # 1. 获取该来源的所有段落
                    paragraphs = self.plugin.metadata_store.get_paragraphs_by_source(source)
                    
                    found_nodes = set()
                    found_edges = []
                    processed_edge_keys = set()
                    
                    # 2. 遍历段落收集实体和关系
                    node_map = {} # lowercase_id -> display_label
                    
                    for p in paragraphs:
                        # 收集实体
                        p_entities = self.plugin.metadata_store.get_paragraph_entities(p['hash'])
                        for e in p_entities:
                            raw_name = e['name']
                            lower_id = raw_name.strip().lower()
                            node_map[lower_id] = raw_name # Use the name from entity table as preferred label
                            found_nodes.add(lower_id)
                            
                        # 收集关系
                        p_relations = self.plugin.metadata_store.get_paragraph_relations(p['hash'])
                        for r in p_relations:
                            s_raw, t_raw = r['subject'], r['object']
                            s_id, t_id = s_raw.strip().lower(), t_raw.strip().lower()
                            
                            # Update labels if not present (prefer entity table, but relation raw text is fallback)
                            if s_id not in node_map: node_map[s_id] = s_raw
                            if t_id not in node_map: node_map[t_id] = t_raw
                            
                            found_nodes.add(s_id)
                            found_nodes.add(t_id)
                            
                            key = (s_id, t_id)
                            if key not in processed_edge_keys:
                                found_edges.append({
                                    "id": f"{s_id}_{t_id}",
                                    "from": s_id,
                                    "to": t_id,
                                    "value": float(r['confidence']),
                                    "label": r['predicate'],
                                    "arrows": "to"
                                })
                                processed_edge_keys.add(key)
                    
                    # 3. 转换为前端格式
                    nodes = [{"id": nid, "label": node_map.get(nid, nid)} for nid in found_nodes]
                    edges = found_edges
                    
                    # 4. (修正) 应用叶子节点过滤 (之前此处有且逻辑错误，会导致无法进入此分支)
                    if exclude_leaf:
                       # 重新计算局部度数 (针对当前来源过滤出的子图)
                       degrees = {}
                       for e in edges:
                           degrees[e['from']] = degrees.get(e['from'], 0) + 1
                           degrees[e['to']] = degrees.get(e['to'], 0) + 1
                       
                       # 过滤掉局部度数为 1 的节点
                       nodes = [n for n in nodes if degrees.get(n['id'], 0) != 1]
                       node_ids = set(n['id'] for n in nodes)
                       # 只保留连接两个已存在节点的边
                       edges = [e for e in edges if e['from'] in node_ids and e['to'] in node_ids]

                    return {
                        "nodes": nodes, 
                        "edges": edges, 
                        "debug": {
                            "source": source,
                            "nodes": len(nodes),
                            "edges": len(edges)
                        }
                    }
                    
                except Exception as e:
                    logger.error(f"Get graph by source failed: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

            # --- 分支 2: 全量图谱 (现有逻辑) ---
            if self.plugin.graph_store is None:
                raise HTTPException(status_code=503, detail="Graph store not initialized")
            
            node_names = self.plugin.graph_store.get_nodes()
            
            # --- 智能显著性过滤 (Saliency Filtering) ---
            if exclude_leaf:
                # 1. 获取 PageRank 得分
                scores = self.plugin.graph_store.get_saliency_scores()
                if not scores:
                    filtered_nodes = node_names
                else:
                    # 2. 确定筛选阈值
                    # 使用基于 density 的分位数或线性阈值
                    # density=1.0 展示全部; density=0 仅展示最核心部分
                    sorted_scores = sorted(scores.values())
                    n = len(sorted_scores)
                    # 我们过滤掉后 (1.0 - density) 比例的节点
                    # 但即使 density 很低，也至少保留前 5 个节点或 10% 节点
                    threshold_idx = min(int(n * (1.0 - density)), n - 5)
                    threshold_idx = max(0, threshold_idx)
                    min_score = sorted_scores[threshold_idx] if sorted_scores else 0
                    
                    # 3. 筛选与保护
                    # 识别核心节点 (Hubs) - PageRank 前 10%
                    hub_threshold = sorted_scores[int(n * 0.9)] if n > 10 else 0
                    hubs = {node for node, score in scores.items() if score >= hub_threshold}
                    
                    filtered_nodes = [] # 最终显示的节点 ID 列表
                    node_status = {} # nodeId -> score/ghost status
                    
                    # 确定幽灵密度 (Ghosting) - 阈值以下的 20% 节点作为幽灵显示
                    ghost_threshold_idx = max(0, threshold_idx - int(n * 0.2))
                    ghost_min_score = sorted_scores[ghost_threshold_idx] if sorted_scores else 0

                    for name in node_names:
                        score = scores.get(name, 0)
                        is_hub_neighbor = any(self.plugin.graph_store.get_edge_weight(name, hub) > 0 for hub in hubs) or \
                                          any(self.plugin.graph_store.get_edge_weight(hub, name) > 0 for hub in hubs)
                        
                        if score >= min_score or is_hub_neighbor:
                            # 正常保留
                            filtered_nodes.append(name)
                            node_status[name] = {"is_ghost": False}
                        elif score >= ghost_min_score:
                            # 作为幽灵保留 (Ghosting)
                            filtered_nodes.append(name)
                            node_status[name] = {"is_ghost": True}
            else:
                filtered_nodes = node_names
                node_status = {name: {"is_ghost": False} for name in node_names}

            # 转换为 Set 以提高查找性能
            filtered_node_set = set(filtered_nodes)
            nodes = [
                {
                    "id": name, 
                    "label": name, 
                    "is_ghost": node_status.get(name, {}).get("is_ghost", False)
                } for name in filtered_nodes
            ]
            edges = []
            processed_edges = set()
            
            # 获取所有边 - 遍历每个节点的邻居
            processed_edges = set()
            
            # 预加载所有关系谓语 (MetadataStore)
            # 为了优化性能，一次性全部查出，构建内存查找表
            edge_predicates = {} # (source, target) -> [predicate1, predicate2...]
            
            if self.plugin.metadata_store:
                try:
                    all_relations = self.plugin.metadata_store.get_relations()
                    logger.info(f"[DEBUG] Fetched {len(all_relations)} relations from MetadataStore")
                    for rel in all_relations:
                        s, p, o = rel['subject'], rel['predicate'], rel['object']
                        key = (s, o)
                        if key not in edge_predicates:
                            edge_predicates[key] = []
                        edge_predicates[key].append(p)
                    
                    if all_relations:
                        logger.info(f"[DEBUG] Sample edge_predicates key: {list(edge_predicates.keys())[0]}")
                        
                except Exception as e:
                    logger.error(f"Error fetching relations for graph: {e}")
            else:
                logger.warning("[DEBUG] MetadataStore is NOT initialized or available")

            for source in filtered_nodes: # 关键修复：只从过滤后的节点开始搜索
                neighbors = self.plugin.graph_store.get_neighbors(source)
                for target in neighbors:
                    # 关键修复：确保目标节点也在过滤后的列表中
                    if target not in filtered_node_set:
                        continue
                        
                    edge_key = (source, target)
                    if edge_key not in processed_edges:
                        weight = self.plugin.graph_store.get_edge_weight(source, target)
                        # 获取谓语描述
                        # 尝试精确匹配
                        predicates = edge_predicates.get((source, target), [])
                        
                        # 如果没有找到，尝试不区分大小写的匹配 (slow path, but helpful for debugging)
                        if not predicates:
                            for (ks, ko), preds in edge_predicates.items():
                                if ks.lower() == source.lower() and ko.lower() == target.lower():
                                    predicates = preds
                                    logger.info(f"[DEBUG] Found case-insensitive match for {source}->{target}: {preds}")
                                    break
                        
                        # 如果有谓语，优先显示谓语；否则显示权重
                        if predicates:
                            # 限制长度，防止 label 太长
                            display_label = ", ".join(predicates[:3])
                            if len(predicates) > 3:
                                display_label += "..."
                        else:
                            display_label = f"{weight:.2f}"
                        
                        edges.append({
                            "id": f"{source}_{target}",
                            "from": source, 
                            "to": target, 
                            "value": float(weight),
                            "label": display_label,
                            "predicates": predicates,
                            "arrows": "to"
                        })
                        processed_edges.add(edge_key)
            
            debug_info = {
                "relation_count": len(all_relations) if self.plugin.metadata_store else -1,
                "sample_key": list(edge_predicates.keys())[0] if edge_predicates else None,
                "edge_count": len(edges),
                "exclude_leaf": exclude_leaf
            }
                
            return {"nodes": nodes, "edges": edges, "debug": debug_info}

        @self.app.post("/api/edge/weight")
        async def update_edge_weight(data: EdgeWeightUpdate):
            """更新边权重"""
            if self.plugin.graph_store is None:
                raise HTTPException(status_code=503, detail="Graph store not initialized")
            
            try:
                # 计算增量 (因为 update_edge_weight 是基于增量的)
                # 或者我们需要一个直接设置权重的方法。
                # 查看 GraphStore源码，update_edge_weight 是 add weight.
                # 如果我们要 set weight，我们需要先获取当前权重。
                
                current_weight = self.plugin.graph_store.get_edge_weight(data.source, data.target)
                delta = data.weight - current_weight
                
                new_weight = self.plugin.graph_store.update_edge_weight(data.source, data.target, delta)
                # 持久化保存
                self.plugin.graph_store.save()
                return {"success": True, "new_weight": new_weight}
            except Exception as e:
                logger.error(f"Update weight failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/api/node")
        async def delete_node(data: NodeDelete):
            """删除节点"""
            if self.plugin.graph_store is None:
                raise HTTPException(status_code=503, detail="Graph store not initialized")
                
            try:
                # 使用 GraphStore.delete_nodes 方法
                deleted_count = self.plugin.graph_store.delete_nodes([data.node_id])
                
                # 同时从 MetadataStore 删除实体
                if self.plugin.metadata_store:
                    self.plugin.metadata_store.delete_entity(data.node_id)
                
                # 持久化保存
                self.plugin.graph_store.save()
                return {"success": True, "deleted_count": deleted_count}
            except Exception as e:
                logger.error(f"Delete node failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/api/edge")
        async def delete_edge(data: EdgeDelete):
            """删除边"""
            if self.plugin.graph_store is None:
                raise HTTPException(status_code=503, detail="Graph store not initialized")
            
            try:
                # 将权重设为 0 或移除
                # 简单做法：update_edge_weight 减去当前权重
                current_weight = self.plugin.graph_store.get_edge_weight(data.source, data.target)
                self.plugin.graph_store.update_edge_weight(data.source, data.target, -current_weight)
                
                # 持久化保存
                self.plugin.graph_store.save()
                return {"success": True}
            except Exception as e:
                logger.error(f"Delete edge failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/node")
        async def create_node(data: NodeCreate):
            """创建节点"""
            print(f"DEBUG: graph_store={self.plugin.graph_store}")
            if self.plugin.graph_store is None:
                raise HTTPException(status_code=503, detail="Graph store not initialized")
            
            try:
                # 1. 使用 GraphStore.add_nodes 方法建立物理节点
                added_count = self.plugin.graph_store.add_nodes([data.node_id])
                
                # 2. 同时在 MetadataStore 注册实体，保证元数据一致性
                if self.plugin.metadata_store:
                    self.plugin.metadata_store.add_entity(name=data.node_id)
                
                # 持久化保存
                self.plugin.graph_store.save()
                return {"success": True, "added_count": added_count, "node_id": data.node_id}
            except Exception as e:
                logger.error(f"Create node failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/edge")
        async def create_edge(data: EdgeCreate):
            """创建边 (支持语义关系)"""
            if self.plugin.graph_store is None:
                raise HTTPException(status_code=503, detail="Graph store not initialized")
            
            try:
                # 确保节点存在
                self.plugin.graph_store.add_nodes([data.source, data.target])
                
                # 1. 如果有语义关系，先存入 MetadataStore
                if data.predicate and self.plugin.metadata_store:
                   self.plugin.metadata_store.add_relation(
                       subject=data.source, 
                       predicate=data.predicate, 
                       obj=data.target,
                       confidence=data.weight
                   )

                # 2. 使用 GraphStore.add_edges 方法建立物理连接
                added_count = self.plugin.graph_store.add_edges(
                    [(data.source, data.target)],
                    weights=[data.weight]
                )
                
                # 持久化保存
                self.plugin.graph_store.save()
                return {"success": True, "added_count": added_count, "predicate": data.predicate}
            except Exception as e:
                logger.error(f"Create edge failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.put("/api/node/rename")
        async def rename_node(data: NodeRename):
            """重命名节点 (实际上是创建新节点，复制边，删除旧节点)"""
            if self.plugin.graph_store is None:
                raise HTTPException(status_code=503, detail="Graph store not initialized")
            
            try:
                # 检查旧节点是否存在
                if not self.plugin.graph_store.has_node(data.old_id):
                    raise HTTPException(status_code=404, detail=f"Node '{data.old_id}' not found")
                
                # 获取旧节点的所有边
                neighbors = self.plugin.graph_store.get_neighbors(data.old_id)
                
                # 添加新节点
                self.plugin.graph_store.add_nodes([data.new_id])
                
                # 复制边到新节点
                for neighbor in neighbors:
                    weight = self.plugin.graph_store.get_edge_weight(data.old_id, neighbor)
                    if weight > 0:
                        self.plugin.graph_store.add_edges([(data.new_id, neighbor)], weights=[weight])
                
                # 获取指向旧节点的边 (反向边)
                all_nodes = self.plugin.graph_store.get_nodes()
                for node in all_nodes:
                    if node != data.old_id and node != data.new_id:
                        weight = self.plugin.graph_store.get_edge_weight(node, data.old_id)
                        if weight > 0:
                            self.plugin.graph_store.add_edges([(node, data.new_id)], weights=[weight])
                
                # 删除旧节点
                self.plugin.graph_store.delete_nodes([data.old_id])
                
                # 持久化保存
                self.plugin.graph_store.save()
                return {"success": True, "old_id": data.old_id, "new_id": data.new_id}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Rename node failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/source/list")
        async def list_sources(data: SourceListRequest):
            """获取来源段落列表"""
            if self.plugin.metadata_store is None:
                 raise HTTPException(status_code=503, detail="Metadata store not initialized")
            
            paragraphs = []
            seen_hashes = set()
            
            try:
                # 0. 如果无任何参数，则返回文件列表 (Summary Mode)
                if not data.node_id and not data.edge_source and not data.edge_target:
                    sources = self.plugin.metadata_store.get_all_sources()
                    return {"mode": "summary", "sources": sources}
                # 1. 如果是查节点来源 (By Entity)
                if data.node_id:
                    # 注意: WebUI 传来的 node_id 通常是实体名称 (Node Name)
                    # MetadataStore.get_paragraphs_by_entity 接受 entity_name
                    entity_paras = self.plugin.metadata_store.get_paragraphs_by_entity(data.node_id)
                    for p in entity_paras:
                        if p['hash'] not in seen_hashes:
                            paragraphs.append(p)
                            seen_hashes.add(p['hash'])
                            
                # 2. 如果是查边来源 (By Relation)
                if data.edge_source and data.edge_target:
                    # 查出两点间的所有关系
                    relations = self.plugin.metadata_store.get_relations(
                        subject=data.edge_source, 
                        object=data.edge_target
                    )
                    for rel in relations:
                        rel_paras = self.plugin.metadata_store.get_paragraphs_by_relation(rel['hash'])
                        for p in rel_paras:
                            if p['hash'] not in seen_hashes:
                                paragraphs.append(p)
                                seen_hashes.add(p['hash'])
                                
                # 简化返回结构
                result = []
                for p in paragraphs:
                    result.append({
                        "hash": p["hash"],
                        "content": p["content"], # 全文或截断
                        "created_at": p.get("created_at"),
                        "source": p.get("source", "unknown")
                    })
                    
                return {"sources": result}
                
            except Exception as e:
                logger.error(f"List sources failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/source/batch_delete")
        async def batch_delete_source(data: BatchSourceDeleteRequest):
            """按来源批量删除（文件删除）"""
            if not self.plugin.metadata_store or not self.plugin.vector_store or not self.plugin.graph_store:
                 raise HTTPException(status_code=503, detail="Stores not fully initialized")
                 
            try:
                # 1. 找出所有相关段落
                paragraphs = self.plugin.metadata_store.get_paragraphs_by_source(data.source)
                if not paragraphs:
                    return {"success": True, "message": "No paragraphs found for this source", "count": 0}
                
                deleted_count = 0
                errors = []
                
                # 2. 逐个删除 (复用原子删除逻辑)
                # 考虑到性能，这里是简单的循环。如果有成千上万条，可能需要优化为批量事务。
                for p in paragraphs:
                    try:
                        # Phase 1: DB Transaction
                        cleanup_plan = self.plugin.metadata_store.delete_paragraph_atomic(p['hash'])
                        
                        # Phase 2: Memory Store Cleanup
                        vec_id = cleanup_plan.get("vector_id_to_remove")
                        if vec_id:
                            try:
                                self.plugin.vector_store.delete([vec_id])
                            except Exception:
                                pass # ignore missing vector
                                
                        edges_to_remove = cleanup_plan.get("edges_to_remove", [])
                        if edges_to_remove:
                            try:
                                self.plugin.graph_store.delete_edges(edges_to_remove)
                            except Exception:
                                pass
                                
                        deleted_count += 1
                        
                    except Exception as pe:
                        logger.error(f"Failed to delete paragraph {p['hash']}: {pe}")
                        errors.append(f"{p['hash']}: {pe}")
                
                # 3. 保存变更
                try:
                    self.plugin.vector_store.save()
                    self.plugin.graph_store.save()
                except Exception as se:
                    logger.warning(f"Auto-save after batch delete failed: {se}")
                    
                msg = f"Successfully deleted {deleted_count} paragraphs from source '{data.source}'"
                if errors:
                    msg += f". Errors: {len(errors)} occurred."
                    
                return {"success": True, "message": msg, "count": deleted_count, "errors": errors}
                
            except Exception as e:
                logger.error(f"Batch source delete failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        @self.app.delete("/api/source")
        async def delete_source(data: SourceDeleteRequest):
            """删除来源段落（两阶段提交）"""
            if not self.plugin.metadata_store or not self.plugin.vector_store or not self.plugin.graph_store:
                 raise HTTPException(status_code=503, detail="Stores not fully initialized")
                 
            try:
                # === Phase 1: DB Transaction & Plan Generation ===
                # 调用我们在 MetadataStore 实现的原子方法
                cleanup_plan = self.plugin.metadata_store.delete_paragraph_atomic(data.paragraph_hash)
                
                # === Phase 2: Post-Commit Cleanup (In-Memory Stores) ===
                # 这一步失败不会回滚 DB，但保证了 DB 的一致性
                errors = []
                
                # 1. 清理向量 (使用稳定 ID)
                vec_id = cleanup_plan.get("vector_id_to_remove")
                if vec_id:
                    try:
                        # VectorStore.delete 接受 ID 列表
                        self.plugin.vector_store.delete([vec_id])
                    except Exception as ve:
                        logger.error(f"Vector cleanup failed for {vec_id}: {ve}")
                        errors.append(f"Vector cleanup error: {ve}")
                        
                # 2. 清理图边 (批量删除)
                edges_to_remove = cleanup_plan.get("edges_to_remove", [])
                if edges_to_remove:
                    try:
                        self.plugin.graph_store.delete_edges(edges_to_remove)
                    except Exception as ge:
                        logger.error(f"Graph cleanup failed: {ge}")
                        errors.append(f"Graph cleanup error: {ge}")
                
                # 如果有非致命错误，记录并在响应中提示
                msg = "Source deleted successfully"
                if errors:
                    msg += f", but with cleanup warnings: {'; '.join(errors)}"
                    
                # 触发保存以持久化内存变更
                try:
                    self.plugin.vector_store.save()
                    self.plugin.graph_store.save()
                except Exception as se:
                    logger.warning(f"Auto-save after delete failed: {se}")
                
                return {"success": True, "message": msg, "details": cleanup_plan}
                
            except Exception as e:
                logger.error(f"Delete source failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/save")
        async def manual_save():
            """手动保存所有数据到磁盘"""
            try:
                saved_components = []
                if self.plugin.graph_store is not None:
                    self.plugin.graph_store.save()
                    saved_components.append("graph_store")
                if self.plugin.vector_store is not None:
                    self.plugin.vector_store.save()
                    saved_components.append("vector_store")
                logger.info(f"手动保存完成: {saved_components}")
                return {"success": True, "saved": saved_components}
            except Exception as e:
                logger.error(f"Manual save failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/config")
        async def get_config():
            """获取配置"""
            return {
                "auto_save_enabled": self.plugin.get_config("advanced.enable_auto_save", True),
                "auto_save_interval": self.plugin.get_config("advanced.auto_save_interval_minutes", 5)
            }

        @self.app.post("/api/config/auto_save")
        async def set_auto_save(data: AutoSaveConfig):
            """设置自动保存开关（仅运行时生效）"""
            self.plugin._runtime_auto_save = data.enabled
            logger.info(f"自动保存已{'启用' if data.enabled else '禁用'}（运行时）")
            return {"success": True, "auto_save_enabled": data.enabled}

        @self.app.get("/")
        async def index():
            """返回主页"""
            html_path = Path(__file__).parent / "web" / "index.html"
            if html_path.exists():
                return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
            return HTMLResponse(content="<h1>UI Not Found</h1>")

    def run(self):
        """运行服务器 (阻塞)"""
        logger.info(f"Starting A_Memorix WebUI on {self.host}:{self.port}")
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
        self._server = uvicorn.Server(config)
        self._server.run()

    def start(self):
        """在独立线程启动"""
        if self.server_thread and self.server_thread.is_alive():
            return
            
        self.server_thread = threading.Thread(target=self.run, daemon=True)
        self.server_thread.start()
        
    def stop(self):
        """停止服务器"""
        if self._server:
            self._server.should_exit = True
        if self.server_thread:
            self.server_thread.join(timeout=2)
            logger.info("A_Memorix WebUI Stopped")
