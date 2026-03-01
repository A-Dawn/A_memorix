
import asyncio
import threading
import json
import uvicorn
from fastapi import FastAPI, HTTPException, Body, Query, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from src.common.logger import get_logger
from src.common.database.database_model import PersonInfo
from .core.utils.hash import compute_hash

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

class PersonProfileQueryRequest(BaseModel):
    person_id: Optional[str] = None
    person_keyword: Optional[str] = None
    top_k: int = 12
    force_refresh: bool = False

class PersonProfileOverrideUpsertRequest(BaseModel):
    person_id: str
    override_text: str
    updated_by: Optional[str] = None

class PersonProfileOverrideDeleteRequest(BaseModel):
    person_id: str

class ImportPasteRequest(BaseModel):
    input_mode: str = "text"
    content: str
    name: Optional[str] = None
    file_concurrency: Optional[int] = None
    chunk_concurrency: Optional[int] = None
    llm_enabled: Optional[bool] = True
    strategy_override: Optional[str] = "auto"
    chat_log: Optional[bool] = False
    chat_reference_time: Optional[str] = None
    force: Optional[bool] = False
    clear_manifest: Optional[bool] = False
    dedupe_policy: Optional[str] = None

class ImportRetryRequest(BaseModel):
    file_concurrency: Optional[int] = None
    chunk_concurrency: Optional[int] = None
    llm_enabled: Optional[bool] = None
    strategy_override: Optional[str] = None
    chat_log: Optional[bool] = None
    chat_reference_time: Optional[str] = None
    force: Optional[bool] = None
    clear_manifest: Optional[bool] = None
    dedupe_policy: Optional[str] = None

class ImportPathResolveRequest(BaseModel):
    alias: str
    relative_path: Optional[str] = ""
    must_exist: Optional[bool] = True

class ImportRawScanRequest(BaseModel):
    alias: Optional[str] = "raw"
    relative_path: Optional[str] = ""
    glob: Optional[str] = "*"
    recursive: Optional[bool] = True
    input_mode: Optional[str] = "text"
    file_concurrency: Optional[int] = None
    chunk_concurrency: Optional[int] = None
    llm_enabled: Optional[bool] = True
    strategy_override: Optional[str] = "auto"
    chat_log: Optional[bool] = False
    chat_reference_time: Optional[str] = None
    force: Optional[bool] = False
    clear_manifest: Optional[bool] = False
    dedupe_policy: Optional[str] = None

class ImportLpmmOpenieRequest(BaseModel):
    alias: Optional[str] = "lpmm"
    relative_path: Optional[str] = ""
    include_all_json: Optional[bool] = False
    file_concurrency: Optional[int] = None
    chunk_concurrency: Optional[int] = None
    llm_enabled: Optional[bool] = True
    strategy_override: Optional[str] = "auto"
    chat_log: Optional[bool] = False
    chat_reference_time: Optional[str] = None
    force: Optional[bool] = False
    clear_manifest: Optional[bool] = False
    dedupe_policy: Optional[str] = None

class ImportLpmmConvertRequest(BaseModel):
    alias: Optional[str] = "lpmm"
    relative_path: Optional[str] = ""
    target_alias: Optional[str] = "plugin_data"
    target_relative_path: Optional[str] = ""
    dimension: Optional[int] = None
    batch_size: Optional[int] = None

class ImportTemporalBackfillRequest(BaseModel):
    alias: Optional[str] = "plugin_data"
    relative_path: Optional[str] = ""
    dry_run: Optional[bool] = False
    no_created_fallback: Optional[bool] = False
    limit: Optional[int] = 100000


class ImportMaiBotMigrationRequest(BaseModel):
    source_db: Optional[str] = None
    time_from: Optional[str] = None
    time_to: Optional[str] = None
    stream_ids: Optional[List[str]] = None
    group_ids: Optional[List[str]] = None
    user_ids: Optional[List[str]] = None
    start_id: Optional[int] = None
    end_id: Optional[int] = None
    no_resume: Optional[bool] = None
    reset_state: Optional[bool] = None
    read_batch_size: Optional[int] = None
    commit_window_rows: Optional[int] = None
    embed_batch_size: Optional[int] = None
    entity_embed_batch_size: Optional[int] = None
    embed_workers: Optional[int] = None
    max_errors: Optional[int] = None
    log_every: Optional[int] = None
    preview_limit: Optional[int] = None
    dry_run: Optional[bool] = None
    verify_only: Optional[bool] = None

class MemorixServer:
    def __init__(self, plugin_instance, host="0.0.0.0", port=8082):
        self.plugin = plugin_instance
        self.host = host
        self.port = port
        self.app = FastAPI(title="A_Memorix 可视化编辑器")
        self.server_thread = None
        self._server = None
        self.should_exit = False
        
        # 缓存 relations predicate map
        self._relation_cache = None
        self._relation_cache_timestamp = 0
        self._relation_cache_snapshot = None

        # 兜底：确保核心存储在 Web 服务启动前可用（例如 debug_server 直启场景）
        if any(
            getattr(self.plugin, attr, None) is None
            for attr in ("metadata_store", "vector_store", "graph_store", "embedding_manager")
        ):
            sync_init = getattr(self.plugin, "_sync_initialize", None)
            if callable(sync_init):
                try:
                    sync_init()
                except Exception as e:
                    logger.warning(f"MemorixServer pre-init storage failed: {e}")

        # 导入任务管理器
        from .core.utils.web_import_manager import ImportTaskManager
        self.import_manager = ImportTaskManager(plugin_instance)
        self.import_manager.set_write_changed_callback(self._on_import_write_changed)
        
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
        def _build_person_profile_service():
            from .core.utils.person_profile_service import PersonProfileService

            return PersonProfileService(
                metadata_store=self.plugin.metadata_store,
                graph_store=self.plugin.graph_store,
                vector_store=self.plugin.vector_store,
                embedding_manager=self.plugin.embedding_manager,
                sparse_index=getattr(self.plugin, "sparse_index", None),
                plugin_config=getattr(self.plugin, "config", {}) or {},
            )

        def _resolve_person_id_for_web(service, raw_value: str) -> str:
            value = str(raw_value or "").strip()
            if not value:
                return ""
            if len(value) == 32 and all(ch in "0123456789abcdefABCDEF" for ch in value):
                return value.lower()
            resolved = service.resolve_person_id(value)
            return resolved or ""

        def _parse_group_nicks(raw_value: Any) -> List[str]:
            if not raw_value:
                return []
            try:
                data = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
            except Exception:
                return []
            if not isinstance(data, list):
                return []
            out: List[str] = []
            for item in data:
                if isinstance(item, dict):
                    nick = str(item.get("group_nick_name", "")).strip()
                    if nick:
                        out.append(nick)
                elif isinstance(item, str):
                    nick = item.strip()
                    if nick:
                        out.append(nick)
            return out

        def _is_import_enabled() -> bool:
            return bool(self.plugin.get_config("web.import.enabled", True))

        def _ensure_write_allowed() -> None:
            if self.import_manager and self.import_manager.is_write_blocked():
                raise HTTPException(
                    status_code=409,
                    detail="导入任务运行中，写操作已临时禁用",
                )

        def _ensure_import_token(token_header: Optional[str]) -> None:
            configured = str(self.plugin.get_config("web.import.token", "") or "").strip()
            if not configured:
                return
            if not token_header:
                raise HTTPException(status_code=401, detail="Missing import token")
            if token_header.strip() != configured:
                raise HTTPException(status_code=403, detail="Invalid import token")

        def _relation_db_snapshot() -> tuple[int, float, str]:
            if not self.plugin.metadata_store:
                return (0, 0.0, "")
            try:
                rows = self.plugin.metadata_store.query(
                    """
                    SELECT
                        COUNT(*) AS relation_count,
                        COALESCE(MAX(created_at), 0) AS max_created_at,
                        COALESCE(MAX(hash), '') AS max_hash
                    FROM relations
                    """
                )
                if not rows:
                    return (0, 0.0, "")
                row = rows[0]
                if isinstance(row, dict):
                    return (
                        int(row.get("relation_count") or 0),
                        float(row.get("max_created_at") or 0.0),
                        str(row.get("max_hash") or ""),
                    )
                return (int(row[0] or 0), float(row[1] or 0.0), str(row[2] or ""))
            except Exception as e:
                logger.warning(f"Failed to read relation snapshot: {e}")
                return (0, 0.0, "")

        def _fallback_predicates_for_edge(source: str, target: str) -> List[str]:
            if not self.plugin.metadata_store:
                return []
            try:
                rows = self.plugin.metadata_store.get_relations(subject=source, object=target)
            except Exception as e:
                logger.warning(f"Fallback relation lookup failed for {source}->{target}: {e}")
                return []

            predicates: List[str] = []
            seen = set()
            for row in rows:
                if not isinstance(row, dict):
                    continue
                pred = str(row.get("predicate", "") or "").strip()
                if not pred:
                    continue
                key = pred.lower()
                if key in seen:
                    continue
                seen.add(key)
                predicates.append(pred)
            return predicates

        def _empty_manifest_cleanup(requested_sources: Optional[List[str]] = None) -> Dict[str, Any]:
            return {
                "requested_sources": list(requested_sources or []),
                "removed_count": 0,
                "removed_keys": [],
                "remaining_count": 0,
                "unmatched_sources": [],
                "warnings": [],
            }

        async def _cleanup_manifest_for_sources(sources: List[str]) -> Dict[str, Any]:
            requested_sources = [str(s or "").strip() for s in (sources or []) if str(s or "").strip()]
            cleanup = _empty_manifest_cleanup(requested_sources)
            if not self.import_manager:
                if requested_sources:
                    cleanup["unmatched_sources"] = list(requested_sources)
                cleanup["warnings"].append("import_manager_unavailable")
                return cleanup

            try:
                payload = await self.import_manager.invalidate_manifest_for_sources(requested_sources)
                if not isinstance(payload, dict):
                    if requested_sources:
                        cleanup["unmatched_sources"] = list(requested_sources)
                    cleanup["warnings"].append("manifest_cleanup_invalid_response")
                    return cleanup

                cleanup["requested_sources"] = list(payload.get("requested_sources") or requested_sources)
                cleanup["removed_count"] = int(payload.get("removed_count") or 0)
                cleanup["removed_keys"] = [str(k) for k in (payload.get("removed_keys") or [])]
                cleanup["remaining_count"] = int(payload.get("remaining_count") or 0)
                cleanup["unmatched_sources"] = [str(s) for s in (payload.get("unmatched_sources") or [])]
                cleanup["warnings"] = [str(w) for w in (payload.get("warnings") or [])]
                return cleanup
            except Exception as e:
                logger.warning(f"Manifest cleanup failed for sources={requested_sources}: {e}")
                if requested_sources:
                    cleanup["unmatched_sources"] = list(requested_sources)
                cleanup["warnings"].append(f"manifest_cleanup_failed: {e}")
                return cleanup

        def _load_import_guide_text() -> Dict[str, Any]:
            # 使用插件工作目录内的相对路径文档，不依赖远端网络。
            local_path = (Path(__file__).resolve().parent / "IMPORT_GUIDE.md").resolve()
            if not local_path.exists():
                raise HTTPException(status_code=404, detail=f"导入文档不存在: {local_path}")
            try:
                text = local_path.read_text(encoding="utf-8")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"读取导入文档失败: {e}")
            if not text.strip():
                raise HTTPException(status_code=404, detail=f"导入文档为空: {local_path}")
            return {
                "source": "local",
                "url": "",
                "path": str(local_path),
                "content": text,
            }

        def _collect_paragraph_entities(paragraph_hash: str) -> Dict[str, str]:
            candidates: Dict[str, str] = {}
            if not self.plugin.metadata_store:
                return candidates
            try:
                entities = self.plugin.metadata_store.get_paragraph_entities(paragraph_hash)
            except Exception as e:
                logger.warning(f"Collect paragraph entities failed: {e}")
                return candidates

            for ent in entities:
                entity_hash = str(ent.get("hash", "") or "").strip()
                entity_name = str(ent.get("name", "") or "").strip()
                if entity_hash and entity_name:
                    candidates[entity_hash] = entity_name
            return candidates

        def _is_entity_still_referenced(entity_hash: str, entity_name: str) -> bool:
            if not self.plugin.metadata_store:
                return False

            if self.plugin.metadata_store.query(
                "SELECT 1 FROM paragraph_entities WHERE entity_hash = ? LIMIT 1",
                (entity_hash,),
            ):
                return True

            canon_name = str(entity_name or "").strip().lower()
            if canon_name and self.plugin.metadata_store.query(
                """
                SELECT 1
                FROM relations
                WHERE LOWER(TRIM(subject)) = ? OR LOWER(TRIM(object)) = ?
                LIMIT 1
                """,
                (canon_name, canon_name),
            ):
                return True

            if self.plugin.graph_store:
                try:
                    if self.plugin.graph_store.get_neighbors(entity_name):
                        return True
                except Exception:
                    pass

            return False

        def _cleanup_orphan_entities(candidate_entities: Dict[str, str]) -> tuple[int, int]:
            removed = 0
            skipped = 0

            if (
                not candidate_entities
                or not self.plugin.metadata_store
            ):
                return removed, skipped

            for entity_hash, entity_name in candidate_entities.items():
                if _is_entity_still_referenced(entity_hash, entity_name):
                    skipped += 1
                    continue

                try:
                    deleted = self.plugin.metadata_store.delete_entity(entity_hash)
                except Exception as e:
                    logger.warning(f"Delete orphan entity failed: {entity_hash[:8]}... ({e})")
                    skipped += 1
                    continue

                if not deleted:
                    skipped += 1
                    continue

                if self.plugin.vector_store:
                    try:
                        self.plugin.vector_store.delete([entity_hash])
                    except Exception:
                        pass

                if self.plugin.graph_store:
                    try:
                        self.plugin.graph_store.delete_nodes([entity_name])
                    except Exception:
                        pass

                removed += 1

            return removed, skipped

        @self.app.on_event("shutdown")
        async def _on_shutdown():
            try:
                await self.import_manager.shutdown()
            except Exception as e:
                logger.warning(f"Import manager shutdown failed: {e}")

        
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
                            node_map[lower_id] = raw_name # 优先使用实体表中的名称作为显示标签
                            found_nodes.add(lower_id)
                            
                        # 收集关系
                        p_relations = self.plugin.metadata_store.get_paragraph_relations(p['hash'])
                        for r in p_relations:
                            s_raw, t_raw = r['subject'], r['object']
                            s_id, t_id = s_raw.strip().lower(), t_raw.strip().lower()
                            
                            # 如果不存在则更新标签（优先使用实体表，关系原始文本作为备选）
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
                       original_nodes = nodes
                       original_edges = edges
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

                       # 兜底：若过滤后为空，回退到原始来源子图，避免聚焦视图“全空白”。
                       if not nodes and (original_nodes or original_edges):
                           nodes = original_nodes
                           edges = original_edges

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
            # 使用缓存优化性能
            edge_predicates = {}
            relation_count = 0
            cache_rebuilt = False
            relation_snapshot = (0, 0.0, "")
            
            if self.plugin.metadata_store:
                try:
                    relation_snapshot = _relation_db_snapshot()
                    if self._relation_cache is None or self._relation_cache_snapshot != relation_snapshot:
                        # 重新构建缓存
                        import time
                        start_t = time.time()
                        raw_triples = self.plugin.metadata_store.get_all_triples()
                        cache = {}
                        count = 0
                        for s, p, o, _ in raw_triples: # _ 用于忽略 hash 字段
                            key = (s, o)
                            if key not in cache: cache[key] = []
                            cache[key].append(p)
                            count += 1
                        self._relation_cache = cache
                        self._relation_cache_snapshot = relation_snapshot
                        self._relation_cache_timestamp = time.time()
                        cache_rebuilt = True
                        logger.info(f"[Cache] 重新构建关系缓存，共 {count} 条关系，耗时 {time.time() - start_t:.4f}s")
                    
                    edge_predicates = self._relation_cache or {}
                    relation_count = int(relation_snapshot[0])
                        
                except Exception as e:
                    logger.error(f"Error fetching relations for graph: {e}")
            else:
                logger.warning("[DEBUG] MetadataStore 未初始化或不可用")

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
                        
                        # 如果没有找到，尝试不区分大小写的匹配 (慢速路径，但有助于调试)
                        if not predicates:
                            for (ks, ko), preds in edge_predicates.items():
                                if ks.lower() == source.lower() and ko.lower() == target.lower():
                                    predicates = preds
                                    logger.info(f"[DEBUG] Found case-insensitive match for {source}->{target}: {preds}")
                                    break

                        # 缓存未命中时，回查 MetadataStore（兜底），避免仅显示权重。
                        if not predicates:
                            predicates = _fallback_predicates_for_edge(source, target)
                            if predicates and isinstance(self._relation_cache, dict):
                                self._relation_cache[(source, target)] = predicates
                                edge_predicates[(source, target)] = predicates
                        
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

            # --- V5: 恢复非活跃边 (已冷冻/已衰减) ---
            # 遍历持久化存储中的所有边，找出虽然权重为 0（非活跃）但连接着两个可见节点的边
            if self.plugin.graph_store:
                gst = self.plugin.graph_store
                idx_to_node = gst._nodes
                
                # O(E) 遍历 - 对于可视化端点是可以接受的
                for (s_idx, t_idx), hashes in gst._edge_hash_map.items():
                    if not hashes: continue
                    
                    # 确保索引有效
                    if s_idx < len(idx_to_node) and t_idx < len(idx_to_node):
                        s_name = idx_to_node[s_idx]
                        t_name = idx_to_node[t_idx]
                        
                        # 仅当两个节点都在当前过滤视图中时显示
                        if s_name in filtered_node_set and t_name in filtered_node_set:
                            edge_key = (s_name, t_name)
                            if edge_key not in processed_edges:
                                # 找到一条非活跃边
                                predicates = edge_predicates.get(edge_key, [])
                                display_label = ", ".join(predicates[:3]) if predicates else "(冷冻)"
                                
                                edges.append({
                                    "id": f"{s_name}_{t_name}",
                                    "from": s_name, 
                                    "to": t_name, 
                                    "value": 0.05, # 最小视觉权重
                                    "physics": False, # 不影响布局
                                    "label": display_label,
                                    "predicates": predicates,
                                    "arrows": "to",
                                    "is_active": False,
                                    "dashes": True, # 视觉提示
                                    "color": {"color": "rgba(203, 213, 225, 0.4)"} # 默认 Slate-300
                                })
                                processed_edges.add(edge_key)
            
            # --- V5: 注入节点状态 (软删除) ---
            if self.plugin.metadata_store and nodes:
                try:
                    # 1. 为所有可见节点计算哈希
                    # 映射 hash -> node_index/node_id
                    node_hash_map = {}
                    node_hashes = []
                    
                    # 我们需要规范化的哈希。GraphStore 知道如何规范化？
                    # 通常应与 MetadataStore.compute_hash(node_id) 相同？
                    # 如果可用，让我们使用 MetadataStore.compute_hash 逻辑，或者直接使用 GraphStore 逻辑。
                    # GraphStore 使用 _canonicalize (lower().strip())。MetadataStore 使用 compute_hash(name)。
                    # 它们应该匹配。
                    
                    for i, n in enumerate(nodes):
                        # 注意：在某些分支中 node['id'] 是显示名称，或者是规范化的 ID？
                        # 在分支 2 (filtered_nodes) 中，'id' 是来自 GraphStore 的名称。
                        nid = n['id']
                        # MetadataStore 期望规范化的哈希
                        # 假设 compute_hash 封装了简单的逻辑？
                        # 我们可以导入或重用逻辑。
                        # 安全的做法：如果可用则使用 GraphStore 规范化，然后哈希。
                        # 或者直接尝试按原样对 ID 进行哈希，假设它就是名称。
                        
                        # GraphStore 节点名称保留了大小写，但为了键值进行了规范化。
                        # MetadataStore 的删除基于规范化哈希。
                        # 所以我们应该对规范化名称进行哈希。
                        
                        # 如果不容易获取规范化器，则将其转换为小写。
                        canon_name = nid.strip().lower()
                        h = compute_hash(canon_name)

                        node_hashes.append(h)
                        node_hash_map[h] = i

                    # 2. 批量查询
                    if node_hashes:
                        node_status_map = self.plugin.metadata_store.get_entity_status_batch(node_hashes)

                        # 3. 应用到节点
                        for h, status in node_status_map.items():
                            if h in node_hash_map:
                                idx = node_hash_map[h]
                                node_ref = nodes[idx]
                                if status.get('is_deleted'):
                                    node_ref['is_deleted'] = True
                                    node_ref['color'] = {'background': '#ef4444', 'border': '#fee2e2'} # 红色警告
                                    node_ref['shape'] = 'box' # 不同的形状？
                                    node_ref['label'] += ' (已删除)'
                except Exception as e:
                    logger.warning(f"Failed to inject node status: {e}")

            # --- V5: 注入记忆状态 (批量) ---
            if self.plugin.metadata_store:
                try:
                    import datetime
                    now = datetime.datetime.now().timestamp()
                    
                    # 元数据查询收集器
                    # 我们需要查询所有边的关系状态。
                    # 如果一条一条查询会很重。理想情况下我们需要批量查询。
                    # MetadataStore.get_relation_status_batch 接收 [hashes]。
                    # 但这里我们只有边 (s,t)。我们需要先将 (s,t) 映射到哈希。
                    
                    # GraphStore 拥有 `_edge_hash_map`。
                    # 让我们遍历边并收集哈希。
                    
                    all_graph_hashes = []
                    edge_hash_mapping = {} # 边索引 -> [hashes]
                    
                    gst = self.plugin.graph_store
                    
                    for i, edge in enumerate(edges):
                        s, t = edge['from'], edge['to']
                        # 从 GraphStore 映射中获取哈希
                        # 使用内部方法还是需要公开 API？
                        # _edge_hash_map 使用索引。
                        s_canon = gst._canonicalize(s)
                        t_canon = gst._canonicalize(t)
                        if s_canon in gst._node_to_idx and t_canon in gst._node_to_idx:
                            s_idx = gst._node_to_idx[s_canon]
                            t_idx = gst._node_to_idx[t_canon]
                            hashes = gst._edge_hash_map.get((s_idx, t_idx), set())
                            if hashes:
                                h_list = list(hashes)
                                all_graph_hashes.extend(h_list)
                                edge_hash_mapping[i] = h_list
 
                    if all_graph_hashes:
                        # 批量查询元数据
                        status_map = self.plugin.metadata_store.get_relation_status_batch(all_graph_hashes)
                        
                        # 应用到边
                        for i, h_list in edge_hash_mapping.items():
                            # 聚合边的状态
                            # 规则:
                            # - 置顶 (Pinned): 如果任一哈希已置顶 -> 边置顶
                            # - 保护 (Protected): 最大 protected_until
                            # - 非活跃 (Inactive): 如果所有哈希皆非活跃 -> 边非活跃 (仅视觉显示，逻辑上图应该处理此情况)
                            # - 健康 (Health): 平均值还是最小值？让我们使用最大安全性。
                            
                            is_pinned = False
                            max_protected = 0
                            all_inactive = True
                            
                            for h in h_list:
                                st = status_map.get(h)
                                if st:
                                    if st.get('is_pinned'): is_pinned = True
                                    p_until = st.get('protected_until') or 0
                                    if p_until > max_protected: max_protected = p_until
                                    if not st.get('is_inactive'): all_inactive = False
                                    
                            edge_ref = edges[i]
                            edge_ref['is_pinned'] = is_pinned
                            edge_ref['protected_until'] = max_protected
                            edge_ref['is_protected'] = (max_protected > now)
                            edge_ref['is_active'] = not all_inactive
                            
                            # 非活跃/已冷冻的视觉线索
                            if all_inactive:
                                edge_ref['color'] = {'color': 'rgba(203, 213, 225, 0.4)'} # Slate-300
                                edge_ref['dashes'] = True
                                
                            # 已保护的视觉线索
                            if is_pinned or (max_protected > now):
                                edge_ref['shadow'] = {'enabled': True, 'color': 'rgba(251, 191, 36, 0.6)', 'size': 5} # 琥珀色阴影

                except Exception as e:
                    logger.warning(f"Failed to inject V5 metadata: {e}")

            debug_info = {
                "relation_count": relation_count,
                "sample_key": list(edge_predicates.keys())[0] if edge_predicates else None,
                "edge_count": len(edges),
                "exclude_leaf": exclude_leaf,
                "cache_rebuilt": cache_rebuilt,
                "relation_snapshot": {
                    "count": int(relation_snapshot[0]),
                    "max_created_at": float(relation_snapshot[1]),
                    "max_hash": str(relation_snapshot[2]),
                },
            }
                
            return {"nodes": nodes, "edges": edges, "debug": debug_info}

        @self.app.post("/api/edge/weight")
        async def update_edge_weight(data: EdgeWeightUpdate):
            """更新边权重"""
            _ensure_write_allowed()
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
                # 持久化保存到磁盘
                self.plugin.graph_store.save()
                return {"success": True, "new_weight": new_weight}
            except Exception as e:
                logger.error(f"Update weight failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/api/node")
        async def delete_node(data: NodeDelete):
            """删除节点"""
            _ensure_write_allowed()
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
                self._invalidate_relation_cache("delete_node")
                return {"success": True, "deleted_count": deleted_count}
            except Exception as e:
                logger.error(f"Delete node failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/api/edge")
        async def delete_edge(data: EdgeDelete):
            """删除边"""
            _ensure_write_allowed()
            if self.plugin.graph_store is None:
                raise HTTPException(status_code=503, detail="Graph store not initialized")
            
            try:
                # 将权重设为 0 或移除
                # 简单做法：update_edge_weight 减去当前权重
                current_weight = self.plugin.graph_store.get_edge_weight(data.source, data.target)
                self.plugin.graph_store.update_edge_weight(data.source, data.target, -current_weight)
                
                # 持久化保存
                self.plugin.graph_store.save()
                self._invalidate_relation_cache("delete_edge")
                return {"success": True}
            except Exception as e:
                logger.error(f"Delete edge failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/node")
        async def create_node(data: NodeCreate):
            """创建节点"""
            _ensure_write_allowed()
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
            _ensure_write_allowed()
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
                self._invalidate_relation_cache("create_edge")
                return {"success": True, "added_count": added_count, "predicate": data.predicate}
            except Exception as e:
                logger.error(f"Create edge failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.put("/api/node/rename")
        async def rename_node(data: NodeRename):
            """重命名节点 (实际上是创建新节点，复制边，删除旧节点)"""
            _ensure_write_allowed()
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
                self._invalidate_relation_cache("rename_node")
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
            _ensure_write_allowed()
            if not self.plugin.metadata_store or not self.plugin.vector_store or not self.plugin.graph_store:
                 raise HTTPException(status_code=503, detail="Stores not fully initialized")
                 
            try:
                # 1. 找出所有相关段落
                paragraphs = self.plugin.metadata_store.get_paragraphs_by_source(data.source)
                if not paragraphs:
                    manifest_cleanup = await _cleanup_manifest_for_sources([data.source])
                    logger.info(
                        f"[ManifestCleanup] batch_delete source='{data.source}' removed="
                        f"{manifest_cleanup.get('removed_count', 0)} unmatched="
                        f"{len(manifest_cleanup.get('unmatched_sources', []))}"
                    )
                    return {
                        "success": True,
                        "message": "No paragraphs found for this source",
                        "count": 0,
                        "manifest_cleanup": manifest_cleanup,
                    }
                
                deleted_count = 0
                errors = []
                candidate_entities: Dict[str, str] = {}
                relation_prune_ops: List[tuple[str, str, str]] = []
                fallback_edges = set()
                
                # 2. 逐个删除 (复用原子删除逻辑)
                # 考虑到性能，这里是简单的循环。如果有成千上万条，可能需要优化为批量事务。
                for p in paragraphs:
                    try:
                        candidate_entities.update(_collect_paragraph_entities(p["hash"]))

                        # Phase 1: DB Transaction
                        cleanup_plan = self.plugin.metadata_store.delete_paragraph_atomic(p['hash'])
                        
                        # Phase 2: Memory Store Cleanup
                        vec_id = cleanup_plan.get("vector_id_to_remove")
                        if vec_id:
                            try:
                                self.plugin.vector_store.delete([vec_id])
                            except Exception:
                                pass # ignore missing vector

                        for op in cleanup_plan.get("relation_prune_ops", []):
                            relation_prune_ops.append(op)

                        for edge in cleanup_plan.get("edges_to_remove", []):
                            fallback_edges.add(tuple(edge))
                                
                        deleted_count += 1
                        
                    except Exception as pe:
                        logger.error(f"Failed to delete paragraph {p['hash']}: {pe}")
                        errors.append(f"{p['hash']}: {pe}")

                # 2.1 图清理：优先使用 relation hash 精准裁剪，避免残留 edge_hash_map。
                try:
                    if relation_prune_ops and hasattr(self.plugin.graph_store, "prune_relation_hashes"):
                        self.plugin.graph_store.prune_relation_hashes(relation_prune_ops)
                    elif fallback_edges:
                        self.plugin.graph_store.delete_edges(list(fallback_edges))
                except Exception as ge:
                    logger.error(f"Batch graph cleanup failed: {ge}")
                    errors.append(f"graph_cleanup: {ge}")

                # 2.2 孤儿实体清理：避免批删后残留无引用节点。
                removed_entities, skipped_entities = _cleanup_orphan_entities(candidate_entities)
                
                # 3. 保存变更
                try:
                    self.plugin.vector_store.save()
                    self.plugin.graph_store.save()
                except Exception as se:
                    logger.warning(f"Auto-save after batch delete failed: {se}")
                    
                msg = f"Successfully deleted {deleted_count} paragraphs from source '{data.source}'"
                if errors:
                    msg += f". Errors: {len(errors)} occurred."
                msg += f" Orphan entities removed={removed_entities}, skipped={skipped_entities}."

                manifest_cleanup = await _cleanup_manifest_for_sources([data.source])
                logger.info(
                    f"[ManifestCleanup] batch_delete source='{data.source}' removed="
                    f"{manifest_cleanup.get('removed_count', 0)} unmatched="
                    f"{len(manifest_cleanup.get('unmatched_sources', []))}"
                )

                self._invalidate_relation_cache("batch_delete_source")
                return {
                    "success": True,
                    "message": msg,
                    "count": deleted_count,
                    "errors": errors,
                    "manifest_cleanup": manifest_cleanup,
                }
                
            except Exception as e:
                logger.error(f"Batch source delete failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        @self.app.delete("/api/source")
        async def delete_source(data: SourceDeleteRequest):
            """删除来源段落（两阶段提交）"""
            _ensure_write_allowed()
            if not self.plugin.metadata_store or not self.plugin.vector_store or not self.plugin.graph_store:
                 raise HTTPException(status_code=503, detail="Stores not fully initialized")

            try:
                # === Phase 1: DB Transaction & Plan Generation ===
                # 调用我们在 MetadataStore 实现的原子方法
                paragraph_before_delete = self.plugin.metadata_store.get_paragraph(data.paragraph_hash)
                source_to_cleanup = ""
                if isinstance(paragraph_before_delete, dict):
                    source_to_cleanup = str(paragraph_before_delete.get("source") or "").strip()

                candidate_entities = _collect_paragraph_entities(data.paragraph_hash)
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
                relation_prune_ops = cleanup_plan.get("relation_prune_ops", []) or []
                edges_to_remove = cleanup_plan.get("edges_to_remove", [])
                if relation_prune_ops:
                    try:
                        self.plugin.graph_store.prune_relation_hashes(relation_prune_ops)
                    except Exception as ge:
                        logger.error(f"Graph cleanup failed: {ge}")
                        errors.append(f"Graph cleanup error: {ge}")
                elif edges_to_remove:
                    try:
                        self.plugin.graph_store.delete_edges(edges_to_remove)
                    except Exception as ge:
                        logger.error(f"Graph cleanup failed: {ge}")
                        errors.append(f"Graph cleanup error: {ge}")

                removed_entities, skipped_entities = _cleanup_orphan_entities(candidate_entities)
                
                # 如果有非致命错误，记录并在响应中提示
                msg = "来源删除成功"
                if errors:
                    msg += f"，但带有清理警告: {'; '.join(errors)}"
                msg += f"；孤儿实体清理: 删除 {removed_entities}，跳过 {skipped_entities}"
                    
                # 触发保存以持久化内存中的变更
                try:
                    self.plugin.vector_store.save()
                    self.plugin.graph_store.save()
                except Exception as se:
                    logger.warning(f"删除来源后的自动保存失败: {se}")

                manifest_cleanup = await _cleanup_manifest_for_sources([])
                manifest_cleanup["skipped"] = False
                if source_to_cleanup:
                    try:
                        remaining_rows = self.plugin.metadata_store.get_paragraphs_by_source(source_to_cleanup)
                    except Exception as source_err:
                        manifest_cleanup["requested_sources"] = [source_to_cleanup]
                        manifest_cleanup["skipped"] = True
                        manifest_cleanup["unmatched_sources"] = [source_to_cleanup]
                        manifest_cleanup["warnings"] = list(manifest_cleanup.get("warnings") or [])
                        manifest_cleanup["warnings"].append(f"source_check_failed: {source_err}")
                    else:
                        if remaining_rows:
                            manifest_cleanup["requested_sources"] = [source_to_cleanup]
                            manifest_cleanup["removed_count"] = 0
                            manifest_cleanup["removed_keys"] = []
                            manifest_cleanup["unmatched_sources"] = []
                            manifest_cleanup["skipped"] = True
                            manifest_cleanup["warnings"] = list(manifest_cleanup.get("warnings") or [])
                            manifest_cleanup["warnings"].append("source_still_exists")
                        else:
                            manifest_cleanup = await _cleanup_manifest_for_sources([source_to_cleanup])
                            manifest_cleanup["skipped"] = False
                else:
                    manifest_cleanup["requested_sources"] = []
                    manifest_cleanup["skipped"] = True
                    manifest_cleanup["warnings"] = list(manifest_cleanup.get("warnings") or [])
                    manifest_cleanup["warnings"].append("source_missing_or_not_found")

                logger.info(
                    f"[ManifestCleanup] single_delete source='{source_to_cleanup or '-'}' removed="
                    f"{manifest_cleanup.get('removed_count', 0)} unmatched="
                    f"{len(manifest_cleanup.get('unmatched_sources', []))} skipped="
                    f"{bool(manifest_cleanup.get('skipped', False))}"
                )

                self._invalidate_relation_cache("delete_source")
                return {
                    "success": True,
                    "message": msg,
                    "details": cleanup_plan,
                    "manifest_cleanup": manifest_cleanup,
                }

            except Exception as e:
                logger.error(f"Delete source failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # --- V5 记忆管理端点 ---
        
        class MemoryProtectRequest(BaseModel):
            id: str # 边 ID "s_t"
            type: str # "pin" (置顶) 或 "ttl" (时间限制)
            duration: Optional[float] = 0.0 # TTL 的小时数
            
        class MemoryActionRequest(BaseModel):
            id: str # 边 ID "s_t"
            
        class MemoryRestoreRequest(BaseModel):
            hash: str
            type: Optional[str] = "relation" # relation (关系) | entity (实体)
        
        @self.app.get("/api/memory/recycle_bin")
        async def get_recycle_bin(limit: int = 50):
            """获取回收站中的记忆 (Entities + Relations)"""
            if not self.plugin.metadata_store:
                raise HTTPException(status_code=503, detail="Metadata store missing")
            
            try:
                # 1. 关系
                deleted_rels = self.plugin.metadata_store.get_deleted_relations(limit)
                for x in deleted_rels: x['type'] = 'relation'
                
                # 2. 实体
                deleted_ents = self.plugin.metadata_store.get_deleted_entities(limit)
                
                # 3. 合并
                combined = deleted_rels + deleted_ents
                combined.sort(key=lambda x: x.get('deleted_at', 0) or 0, reverse=True)
                
                return {"items": combined[:limit]}
            except Exception as e:
                logger.error(f"Recycle bin fetch failed: {e}")
                return {"items": [], "error": str(e)}

        @self.app.post("/api/memory/restore")
        async def restore_memory(data: MemoryRestoreRequest):
            """从回收站恢复记忆"""
            _ensure_write_allowed()
            if not self.plugin.metadata_store or not self.plugin.graph_store:
                raise HTTPException(status_code=503, detail="Stores missing")

            try:
                if data.type == "entity":
                    # 复活实体
                    cursor = self.plugin.metadata_store._conn.cursor()
                    cursor.execute("UPDATE entities SET is_deleted=0, deleted_at=NULL WHERE hash=?", (data.hash,))
                    self.plugin.metadata_store._conn.commit()
                    return {"success": True, "type": "entity", "hash": data.hash}

                # relation: 先从回收站恢复元数据，再回灌图边
                record = self.plugin.metadata_store.restore_relation(data.hash)
                if not record:
                    raise HTTPException(status_code=404, detail="回收站中未找到该记忆")

                s, t = record["subject"], record["object"]

                # 若实体处于软删除状态，先复活再补图。
                self.plugin.metadata_store.revive_entities_by_names([s, t])
                self.plugin.graph_store.add_nodes([s, t])
                self.plugin.graph_store.add_edges(
                    [(s, t)],
                    weights=[record["confidence"]],
                    relation_hashes=[data.hash],
                )
                self.plugin.graph_store.save()
                self._invalidate_relation_cache("memory_restore")

                return {"success": True, "type": "relation", "hash": data.hash}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Restore failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/memory/reinforce")
        async def reinforce_memory(data: MemoryActionRequest):
            """强化记忆 (Reset decay)"""
            _ensure_write_allowed()
            if "_" not in data.id: raise HTTPException(400, "Invalid ID format")
            s, t = data.id.split("_", 1) 
            
            if not self.plugin.graph_store: raise HTTPException(503, "Graph store missing")
            
            try:
                gst = self.plugin.graph_store
                s_idx = gst._node_to_idx.get(gst._canonicalize(s))
                t_idx = gst._node_to_idx.get(gst._canonicalize(t))
                
                if s_idx is not None and t_idx is not None:
                     hashes = gst._edge_hash_map.get((s_idx, t_idx), set())
                     if hashes:
                         self.plugin.metadata_store.reinforce_relations(list(hashes))
                
                # 稍微提升权重
                self.plugin.graph_store.update_edge_weight(s, t, 0.1) 
                self.plugin.graph_store.save()
                
                return {"success": True}
            except Exception as e:
                logger.error(f"Reinforce failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/memory/freeze")
        async def freeze_memory(data: MemoryActionRequest):
            """手动冷冻记忆"""
            _ensure_write_allowed()
            if "_" not in data.id: raise HTTPException(400, "Invalid ID format")
            s, t = data.id.split("_", 1)
            
            if not self.plugin.graph_store: raise HTTPException(503, "Graph store missing")

            try:
                gst = self.plugin.graph_store
                s_idx = gst._node_to_idx.get(gst._canonicalize(s))
                t_idx = gst._node_to_idx.get(gst._canonicalize(t))
                
                # 1. 在元数据中标记为不活跃
                if s_idx is not None and t_idx is not None:
                     hashes = gst._edge_hash_map.get((s_idx, t_idx), set())
                     if hashes:
                         self.plugin.metadata_store.mark_relations_inactive(list(hashes))
                
                # 2. 在图中停用 (移除边但保留映射)
                gst.deactivate_edges([(s, t)])
                gst.save()
                
                return {"success": True}
            except Exception as e:
                 logger.error(f"Freeze failed: {e}")
                 raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/memory/protect")
        async def protect_memory(data: MemoryProtectRequest):
            """设置保护 (Pin/TTL)"""
            _ensure_write_allowed()
            if "_" not in data.id: raise HTTPException(400, "Invalid ID format")
            s, t = data.id.split("_", 1)
            
            if not self.plugin.graph_store: raise HTTPException(503, "Graph store missing")

            try:
                gst = self.plugin.graph_store
                s_idx = gst._node_to_idx.get(gst._canonicalize(s))
                t_idx = gst._node_to_idx.get(gst._canonicalize(t))
                
                if s_idx is not None and t_idx is not None:
                     hashes = gst._edge_hash_map.get((s_idx, t_idx), set())
                     if hashes:
                         h_list = list(hashes)
                         is_pinned = (data.type == "pin")
                         ttl = data.duration * 3600 if data.type == "ttl" else 0
                         
                         self.plugin.metadata_store.protect_relations(h_list, is_pinned=is_pinned, ttl_seconds=ttl)
                
                return {"success": True}
            except Exception as e:
                 logger.error(f"Protect failed: {e}")
                 raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/person_profile/query")
        async def query_person_profile(data: PersonProfileQueryRequest):
            """查询人物画像（自动画像 + 手工覆盖结果）。"""
            _ensure_write_allowed()
            if not bool(self.plugin.get_config("person_profile.enabled", True)):
                raise HTTPException(status_code=400, detail="人物画像功能未启用")
            if self.plugin.metadata_store is None:
                raise HTTPException(status_code=503, detail="Metadata store not initialized")
            try:
                service = _build_person_profile_service()
                ttl_minutes = float(self.plugin.get_config("person_profile.profile_ttl_minutes", 360))
                ttl_seconds = max(60.0, ttl_minutes * 60.0)
                result = await service.query_person_profile(
                    person_id=str(data.person_id or "").strip(),
                    person_keyword=str(data.person_keyword or "").strip(),
                    top_k=max(4, int(data.top_k or 12)),
                    ttl_seconds=ttl_seconds,
                    force_refresh=bool(data.force_refresh),
                    source_note="webui:person_profile_query",
                )
                if not result.get("success", False):
                    raise HTTPException(status_code=400, detail=result.get("error", "人物画像查询失败"))
                return result
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Person profile query failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/person_profile/list")
        async def list_person_profile_candidates(
            keyword: str = Query("", description="关键词（匹配 person_name/nickname/user_id/person_id/group_nick_name）"),
            page: int = Query(1, ge=1, description="页码，从1开始"),
            page_size: int = Query(20, ge=1, le=100, description="每页数量"),
        ):
            """获取人物列表（支持关键词与分页）。"""
            try:
                query = PersonInfo.select(
                    PersonInfo.person_id,
                    PersonInfo.person_name,
                    PersonInfo.nickname,
                    PersonInfo.user_id,
                    PersonInfo.platform,
                    PersonInfo.group_nick_name,
                    PersonInfo.last_know,
                )

                kw = str(keyword or "").strip()
                if kw:
                    query = query.where(
                        (PersonInfo.person_name.contains(kw))
                        | (PersonInfo.nickname.contains(kw))
                        | (PersonInfo.user_id.contains(kw))
                        | (PersonInfo.person_id.contains(kw))
                        | (PersonInfo.group_nick_name.contains(kw))
                    )

                query = query.order_by(PersonInfo.last_know.desc(), PersonInfo.id.desc())
                total = query.count()
                offset = (int(page) - 1) * int(page_size)
                rows = list(query.offset(offset).limit(int(page_size)))

                items: List[Dict[str, Any]] = []
                for row in rows:
                    pid = str(getattr(row, "person_id", "") or "").strip()
                    person_name = str(getattr(row, "person_name", "") or "").strip()
                    nickname = str(getattr(row, "nickname", "") or "").strip()
                    user_id = str(getattr(row, "user_id", "") or "").strip()
                    aliases = _parse_group_nicks(getattr(row, "group_nick_name", None))

                    has_snapshot = False
                    has_override = False
                    latest_profile_updated_at = None
                    if self.plugin.metadata_store is not None and pid:
                        snapshot = self.plugin.metadata_store.get_latest_person_profile_snapshot(pid)
                        override = self.plugin.metadata_store.get_person_profile_override(pid)
                        has_snapshot = snapshot is not None
                        has_override = override is not None and bool(str(override.get("override_text", "")).strip())
                        if has_override:
                            latest_profile_updated_at = override.get("updated_at")
                        elif has_snapshot:
                            latest_profile_updated_at = snapshot.get("updated_at")

                    display_name = person_name or nickname or user_id or pid
                    items.append(
                        {
                            "person_id": pid,
                            "display_name": display_name,
                            "person_name": person_name,
                            "nickname": nickname,
                            "user_id": user_id,
                            "platform": str(getattr(row, "platform", "") or ""),
                            "aliases": aliases,
                            "last_know": getattr(row, "last_know", None),
                            "has_snapshot": has_snapshot,
                            "has_override": has_override,
                            "latest_profile_updated_at": latest_profile_updated_at,
                        }
                    )

                return {
                    "success": True,
                    "keyword": kw,
                    "page": int(page),
                    "page_size": int(page_size),
                    "total": int(total),
                    "items": items,
                }
            except Exception as e:
                logger.error(f"List person profile candidates failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/person_profile/override")
        async def save_person_profile_override(data: PersonProfileOverrideUpsertRequest):
            """保存/更新人物画像手工覆盖。"""
            _ensure_write_allowed()
            if not bool(self.plugin.get_config("person_profile.enabled", True)):
                raise HTTPException(status_code=400, detail="人物画像功能未启用")
            if self.plugin.metadata_store is None:
                raise HTTPException(status_code=503, detail="Metadata store not initialized")
            try:
                service = _build_person_profile_service()
                resolved_pid = _resolve_person_id_for_web(service, data.person_id)
                if not resolved_pid:
                    raise HTTPException(status_code=400, detail="person_id 不能为空")

                override = self.plugin.metadata_store.set_person_profile_override(
                    person_id=resolved_pid,
                    override_text=str(data.override_text or ""),
                    updated_by=str(data.updated_by or "webui"),
                    source="webui",
                )
                ttl_minutes = float(self.plugin.get_config("person_profile.profile_ttl_minutes", 360))
                ttl_seconds = max(60.0, ttl_minutes * 60.0)
                merged = await service.query_person_profile(
                    person_id=resolved_pid,
                    top_k=12,
                    ttl_seconds=ttl_seconds,
                    force_refresh=False,
                    source_note="webui:person_profile_override",
                )
                return {
                    "success": True,
                    "person_id": resolved_pid,
                    "override": override,
                    "profile": merged,
                }
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Save person profile override failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/api/person_profile/override")
        async def delete_person_profile_override(data: PersonProfileOverrideDeleteRequest):
            """清除人物画像手工覆盖。"""
            _ensure_write_allowed()
            if self.plugin.metadata_store is None:
                raise HTTPException(status_code=503, detail="Metadata store not initialized")
            try:
                service = _build_person_profile_service()
                resolved_pid = _resolve_person_id_for_web(service, data.person_id)
                if not resolved_pid:
                    raise HTTPException(status_code=400, detail="person_id 不能为空")

                deleted = self.plugin.metadata_store.delete_person_profile_override(resolved_pid)
                return {"success": True, "person_id": resolved_pid, "deleted": bool(deleted)}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Delete person profile override failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))


        @self.app.post("/api/save")
        async def manual_save():
            """手动保存所有数据到磁盘"""
            _ensure_write_allowed()
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
            _ensure_write_allowed()
            self.plugin._runtime_auto_save = data.enabled
            logger.info(f"自动保存已{'启用' if data.enabled else '禁用'}（运行时）")
            return {"success": True, "auto_save_enabled": data.enabled}

        @self.app.post("/api/import/tasks/upload")
        async def create_import_task_upload(
            files: Optional[List[UploadFile]] = File(default=None),
            files_array: Optional[List[UploadFile]] = File(default=None, alias="files[]"),
            payload: str = Form("{}"),
            x_memorix_import_token: Optional[str] = Header(default=None, alias="X-Memorix-Import-Token"),
        ):
            if not _is_import_enabled():
                raise HTTPException(status_code=404, detail="导入功能未启用")
            _ensure_import_token(x_memorix_import_token)
            merged_files = list(files or []) + list(files_array or [])
            if not merged_files:
                raise HTTPException(status_code=400, detail="至少需要上传一个文件")
            try:
                payload_obj = json.loads(payload or "{}")
            except Exception:
                raise HTTPException(status_code=400, detail="payload 必须为合法 JSON")
            if not isinstance(payload_obj, dict):
                raise HTTPException(status_code=400, detail="payload 必须为 JSON 对象")
            try:
                task = await self.import_manager.create_upload_task(merged_files, payload_obj)
                return {"success": True, "task": task}
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Create import upload task failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/import/tasks/paste")
        async def create_import_task_paste(
            data: ImportPasteRequest,
            x_memorix_import_token: Optional[str] = Header(default=None, alias="X-Memorix-Import-Token"),
        ):
            if not _is_import_enabled():
                raise HTTPException(status_code=404, detail="导入功能未启用")
            _ensure_import_token(x_memorix_import_token)
            try:
                task = await self.import_manager.create_paste_task(data.model_dump())
                return {"success": True, "task": task}
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Create import paste task failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/import/tasks/maibot_migration")
        async def create_import_task_maibot_migration(
            data: ImportMaiBotMigrationRequest,
            x_memorix_import_token: Optional[str] = Header(default=None, alias="X-Memorix-Import-Token"),
        ):
            if not _is_import_enabled():
                raise HTTPException(status_code=404, detail="导入功能未启用")
            _ensure_import_token(x_memorix_import_token)
            try:
                task = await self.import_manager.create_maibot_migration_task(data.model_dump())
                return {"success": True, "task": task}
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Create maibot migration task failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/import/path_aliases")
        async def get_import_path_aliases(
            x_memorix_import_token: Optional[str] = Header(default=None, alias="X-Memorix-Import-Token"),
        ):
            if not _is_import_enabled():
                raise HTTPException(status_code=404, detail="导入功能未启用")
            _ensure_import_token(x_memorix_import_token)
            return {"success": True, "items": self.import_manager.get_path_aliases()}

        @self.app.post("/api/import/path_resolve")
        async def resolve_import_path(
            data: ImportPathResolveRequest,
            x_memorix_import_token: Optional[str] = Header(default=None, alias="X-Memorix-Import-Token"),
        ):
            if not _is_import_enabled():
                raise HTTPException(status_code=404, detail="导入功能未启用")
            _ensure_import_token(x_memorix_import_token)
            try:
                resolved = await self.import_manager.resolve_path_request(data.model_dump())
                return {"success": True, **resolved}
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.post("/api/import/tasks/raw_scan")
        async def create_import_task_raw_scan(
            data: ImportRawScanRequest,
            x_memorix_import_token: Optional[str] = Header(default=None, alias="X-Memorix-Import-Token"),
        ):
            if not _is_import_enabled():
                raise HTTPException(status_code=404, detail="导入功能未启用")
            _ensure_import_token(x_memorix_import_token)
            try:
                task = await self.import_manager.create_raw_scan_task(data.model_dump())
                return {"success": True, "task": task}
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Create raw scan task failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/import/tasks/lpmm_openie")
        async def create_import_task_lpmm_openie(
            data: ImportLpmmOpenieRequest,
            x_memorix_import_token: Optional[str] = Header(default=None, alias="X-Memorix-Import-Token"),
        ):
            if not _is_import_enabled():
                raise HTTPException(status_code=404, detail="导入功能未启用")
            _ensure_import_token(x_memorix_import_token)
            try:
                task = await self.import_manager.create_lpmm_openie_task(data.model_dump())
                return {"success": True, "task": task}
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Create lpmm openie task failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/import/tasks/lpmm_convert")
        async def create_import_task_lpmm_convert(
            data: ImportLpmmConvertRequest,
            x_memorix_import_token: Optional[str] = Header(default=None, alias="X-Memorix-Import-Token"),
        ):
            if not _is_import_enabled():
                raise HTTPException(status_code=404, detail="导入功能未启用")
            _ensure_import_token(x_memorix_import_token)
            try:
                task = await self.import_manager.create_lpmm_convert_task(data.model_dump())
                return {"success": True, "task": task}
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Create lpmm convert task failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/import/tasks/temporal_backfill")
        async def create_import_task_temporal_backfill(
            data: ImportTemporalBackfillRequest,
            x_memorix_import_token: Optional[str] = Header(default=None, alias="X-Memorix-Import-Token"),
        ):
            if not _is_import_enabled():
                raise HTTPException(status_code=404, detail="导入功能未启用")
            _ensure_import_token(x_memorix_import_token)
            try:
                task = await self.import_manager.create_temporal_backfill_task(data.model_dump())
                return {"success": True, "task": task}
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Create temporal backfill task failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/import/tasks")
        async def list_import_tasks(
            limit: int = Query(50, ge=1, le=200),
            x_memorix_import_token: Optional[str] = Header(default=None, alias="X-Memorix-Import-Token"),
        ):
            if not _is_import_enabled():
                raise HTTPException(status_code=404, detail="导入功能未启用")
            _ensure_import_token(x_memorix_import_token)
            items = await self.import_manager.list_tasks(limit=limit)
            settings = await self.import_manager.get_runtime_settings()
            return {"success": True, "items": items, "settings": settings}

        @self.app.get("/api/import/guide")
        async def get_import_guide(
            x_memorix_import_token: Optional[str] = Header(default=None, alias="X-Memorix-Import-Token"),
        ):
            if not _is_import_enabled():
                raise HTTPException(status_code=404, detail="导入功能未启用")
            _ensure_import_token(x_memorix_import_token)
            guide = _load_import_guide_text()
            return {"success": True, **guide}

        @self.app.get("/api/import/tasks/{task_id}")
        async def get_import_task(
            task_id: str,
            include_chunks: bool = Query(False),
            x_memorix_import_token: Optional[str] = Header(default=None, alias="X-Memorix-Import-Token"),
        ):
            if not _is_import_enabled():
                raise HTTPException(status_code=404, detail="导入功能未启用")
            _ensure_import_token(x_memorix_import_token)
            task = await self.import_manager.get_task(task_id, include_chunks=include_chunks)
            if not task:
                raise HTTPException(status_code=404, detail="任务不存在")
            return {"success": True, "task": task}

        @self.app.get("/api/import/tasks/{task_id}/files/{file_id}/chunks")
        async def get_import_task_chunks(
            task_id: str,
            file_id: str,
            offset: int = Query(0, ge=0),
            limit: int = Query(100, ge=1, le=500),
            x_memorix_import_token: Optional[str] = Header(default=None, alias="X-Memorix-Import-Token"),
        ):
            if not _is_import_enabled():
                raise HTTPException(status_code=404, detail="导入功能未启用")
            _ensure_import_token(x_memorix_import_token)
            data = await self.import_manager.get_chunks(task_id, file_id, offset=offset, limit=limit)
            if not data:
                raise HTTPException(status_code=404, detail="任务或文件不存在")
            return {"success": True, **data}

        @self.app.post("/api/import/tasks/{task_id}/cancel")
        async def cancel_import_task(
            task_id: str,
            x_memorix_import_token: Optional[str] = Header(default=None, alias="X-Memorix-Import-Token"),
        ):
            if not _is_import_enabled():
                raise HTTPException(status_code=404, detail="导入功能未启用")
            _ensure_import_token(x_memorix_import_token)
            task = await self.import_manager.cancel_task(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="任务不存在")
            return {"success": True, "task": task}

        @self.app.post("/api/import/tasks/{task_id}/retry_failed")
        async def retry_import_task_failed(
            task_id: str,
            data: Optional[ImportRetryRequest] = Body(default=None),
            x_memorix_import_token: Optional[str] = Header(default=None, alias="X-Memorix-Import-Token"),
        ):
            if not _is_import_enabled():
                raise HTTPException(status_code=404, detail="导入功能未启用")
            _ensure_import_token(x_memorix_import_token)
            try:
                overrides = data.model_dump(exclude_none=True) if data else {}
                task = await self.import_manager.retry_failed(task_id, overrides=overrides)
                if not task:
                    raise HTTPException(status_code=404, detail="任务不存在")
                retry_summary = {}
                if isinstance(task, dict):
                    retry_summary = dict(task.get("retry_summary") or {})
                return {"success": True, "task": task, "retry_summary": retry_summary}
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Retry failed import task error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/import")
        async def import_page():
            """返回导入中心页面"""
            if not _is_import_enabled():
                raise HTTPException(status_code=404, detail="导入功能未启用")
            html_path = Path(__file__).parent / "web" / "import.html"
            if html_path.exists():
                return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
            return HTMLResponse(content="<h1>Import UI Not Found</h1>")

        @self.app.get("/")
        async def index():
            """返回主页"""
            html_path = Path(__file__).parent / "web" / "index.html"
            if html_path.exists():
                return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
            return HTMLResponse(content="<h1>UI Not Found</h1>")

    def _invalidate_relation_cache(self, reason: str = "") -> None:
        self._relation_cache = None
        self._relation_cache_timestamp = 0
        self._relation_cache_snapshot = None
        if reason:
            logger.debug(f"关系谓词缓存已失效: {reason}")

    async def _on_import_write_changed(self, payload: Dict[str, Any]) -> None:
        task_kind = str(payload.get("task_kind") or "").strip() or "unknown"
        task_id = str(payload.get("task_id") or "").strip() or "-"
        self._invalidate_relation_cache(f"import_task:{task_kind}:{task_id}")

    def run(self):
        """运行服务器 (阻塞)"""
        logger.info(f"正在启动 A_Memorix WebUI，地址：{self.host}:{self.port}")
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
            logger.info("A_Memorix WebUI 已停止")
