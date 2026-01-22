
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
        async def get_graph():
            """获取全量图谱数据"""
            if not self.plugin.graph_store:
                raise HTTPException(status_code=503, detail="Graph store not initialized")
            
            nodes = []
            edges = []
            
            # 使用正确的 GraphStore API
            node_names = self.plugin.graph_store.get_nodes()
            
            # 构建节点列表
            for name in node_names:
                nodes.append({"id": name, "label": name})
                
            # 获取所有边 - 遍历每个节点的邻居
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

            for source in node_names:
                neighbors = self.plugin.graph_store.get_neighbors(source)
                for target in neighbors:
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
                "edge_count": len(edges)
            }
                
            return {"nodes": nodes, "edges": edges, "debug": debug_info}

        @self.app.post("/api/edge/weight")
        async def update_edge_weight(data: EdgeWeightUpdate):
            """更新边权重"""
            if not self.plugin.graph_store:
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
            if not self.plugin.graph_store:
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
            if not self.plugin.graph_store:
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
            if not self.plugin.graph_store:
                raise HTTPException(status_code=503, detail="Graph store not initialized")
            
            try:
                # 使用 GraphStore.add_nodes 方法
                added_count = self.plugin.graph_store.add_nodes([data.node_id])
                # 持久化保存
                self.plugin.graph_store.save()
                return {"success": True, "added_count": added_count, "node_id": data.node_id}
            except Exception as e:
                logger.error(f"Create node failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/edge")
        async def create_edge(data: EdgeCreate):
            """创建边 (支持语义关系)"""
            if not self.plugin.graph_store:
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
            if not self.plugin.graph_store:
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
            if not self.plugin.metadata_store:
                 raise HTTPException(status_code=503, detail="Metadata store not initialized")
            
            paragraphs = []
            seen_hashes = set()
            
            try:
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
                if self.plugin.graph_store:
                    self.plugin.graph_store.save()
                    saved_components.append("graph_store")
                if self.plugin.vector_store:
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
