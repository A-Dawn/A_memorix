# A_Memorix 配置参数详解（config.toml）

适用版本：`plugins/A_memorix/config.toml`（`config_version = "4.0.1"`，插件代码 `v0.5.1`）。

---

## ⚠️ 先看这 5 条

- `embedding.quantization_type` 当前**基本不生效**：虽然配置支持 `float32/int8/pq`，但 `VectorStore` 内部目前固定走 SQ8（`int8`）实现（后期预期不会走其他实现）。
- `retrieval.sparse` 与 `retrieval.fusion` 是新增检索增强配置：可在 embedding 异常时自动回退 BM25，并通过 weighted RRF 融合候选。
- `routing.search_owner=action` + `routing.tool_search_mode=forward` 是默认推荐：Action 主责 `search/time`，Tool 在 `search/time` 上走统一转发链路。
- `memory.reinforce_buffer_max_size`、`memory.min_active_weight_protected` 当前代码里**未实际使用**。
- `filter.chats = []` 时采用安全兜底：`whitelist`=全部拒绝，`blacklist`=全部放行。
- `retrieval.sparse.enable_relation_sparse_fallback = false` 会关闭关系 sparse 召回，但当前段落 sparse 查询路径仍会幂等检查 `relations_fts` schema/backfill（有轻微额外开销）。

---

## `[plugin]` 插件基础

- `plugin.config_version`
  - 功能：配置版本号。
  - 生效：用于插件配置迁移；当版本不一致时按 `config_schema` 自动迁移并回写配置。
- `plugin.enabled`
  - 功能：插件开关。
  - 生效：加载配置后会回写到 `enable_plugin`，决定插件管理器是否启用该插件。

## `[storage]` 存储

- `storage.data_dir`
  - 功能：插件数据目录（向量/图/元数据都在其子目录下）。
  - 生效：
    - 以 `.` 开头时，按 `plugins/A_memorix/plugin.py` 所在目录解析相对路径。
    - 其他值按绝对/当前工作目录路径处理。

## `[embedding]` 嵌入

- `embedding.dimension`
  - 功能：期望嵌入维度。
  - 生效：作为嵌入探测的目标维度与失败兜底维度。
- `embedding.quantization_type`
  - 功能：期望向量量化类型（`float32/int8/pq`）。
  - 生效：会传入初始化流程，但当前 `VectorStore` 固定 SQ8，实际仍是 `int8` 路径。
- `embedding.batch_size`
  - 功能：批量编码大小。
  - 生效：`EmbeddingAPIAdapter.encode()` 的默认批次大小。
- `embedding.max_concurrent`
  - 功能：嵌入请求最大并发。
  - 生效：适配器内部并发控制上限。
- `embedding.model_name`
  - 功能：指定嵌入模型（或 `auto`）。
  - 生效：适配器优先按该名称查 `model_config`，失败再回退任务默认模型。
- `embedding.retry.max_attempts`
  - 功能：最大重试次数。
  - 生效：嵌入请求失败后的重试上限。
- `embedding.retry.max_wait_seconds`
  - 功能：最大退避等待秒数。
  - 生效：指数退避等待时间上限。
- `embedding.retry.min_wait_seconds`
  - 功能：最小退避等待秒数。
  - 生效：指数退避初始等待时间。

## `[retrieval]` 检索

- `retrieval.top_k_relations`
  - 功能：关系通道召回数量。
  - 生效：DualPath 关系检索分支的候选数量基线。
- `retrieval.top_k_paragraphs`
  - 功能：段落通道召回数量。
  - 生效：DualPath 段落检索分支的候选数量基线。
- `retrieval.alpha`
  - 功能：双路融合权重（0 偏关系，1 偏段落）。
  - 生效：融合阶段分数加权。
- `retrieval.enable_ppr`
  - 功能：是否开启 Personalized PageRank 重排。
  - 生效：DualPath 检索后重排开关。
- `retrieval.ppr_alpha`
  - 功能：PPR 的阻尼系数。
  - 生效：传给 `PageRankConfig(alpha=...)`。
- `retrieval.ppr_concurrency_limit`
  - 功能：PPR 计算并发上限。
  - 生效：用于限制 PPR 线程池重排并发。
- `retrieval.enable_parallel`
  - 功能：是否并行执行段落/关系检索。
  - 生效：DualPath 内部并发执行开关。
- `retrieval.relation_semantic_fallback`
  - 功能：关系查询失败时是否回退语义检索。
  - 生效：`/query relation` 与 `knowledge_query` relation 模式的回退开关。
- `retrieval.relation_fallback_min_score`
  - 功能：关系语义回退最低分数阈值。
  - 生效：过滤低分语义关系候选。
- `retrieval.search.smart_fallback.enabled`
  - 功能：统一链路 search 低分时是否启用路径回退。
  - 生效：search 最高分低于阈值时，尝试基于图路径补充间接关系结果。
- `retrieval.search.smart_fallback.threshold`
  - 功能：统一链路 search 的低分触发阈值。
  - 生效：当 search 最高分小于该值时触发路径回退（默认 `0.6`）。
- `retrieval.search.safe_content_dedup.enabled`
  - 功能：统一链路结果安全去重开关。
  - 生效：按 hash/内容相似度去重，并保证至少保留一条结果。

### `[retrieval.sparse]` 稀疏检索（FTS5 + BM25）

- `retrieval.sparse.enabled`
  - 功能：是否启用稀疏检索路径。
  - 生效：关闭后不会触发 BM25 检索。
- `retrieval.sparse.backend`
  - 功能：稀疏后端类型。
  - 生效：当前仅支持 `fts5`。
- `retrieval.sparse.lazy_load`
  - 功能：是否懒加载索引连接。
  - 生效：开启后首次命中稀疏检索时才加载。
- `retrieval.sparse.mode`
  - 功能：稀疏检索模式（`auto/fallback_only/hybrid`）。
  - 生效：控制 embedding 正常/异常时是否启用 BM25。
- `retrieval.sparse.tokenizer_mode`
  - 功能：分词模式（`jieba/mixed/char_2gram`）。
  - 生效：FTS MATCH 查询构造时使用。
- `retrieval.sparse.jieba_user_dict`
  - 功能：jieba 用户词典路径。
  - 生效：`tokenizer_mode` 包含 jieba 时加载。
- `retrieval.sparse.char_ngram_n`
  - 功能：字符 n-gram 的 n。
  - 生效：`char_2gram`/`mixed` 分词路径。
- `retrieval.sparse.candidate_k`
  - 功能：稀疏检索候选上限。
  - 生效：BM25 召回数量的默认上限。
- `retrieval.sparse.max_doc_len`
  - 功能：BM25 返回内容最大长度。
  - 生效：返回段落内容截断长度。
- `retrieval.sparse.enable_ngram_fallback_index`
  - 功能：是否启用 ngram 倒排回退索引。
  - 生效：FTS 未命中时优先走 `paragraph_ngrams` 倒排召回，避免 LIKE 全表扫描。
- `retrieval.sparse.enable_like_fallback`
  - 功能：是否启用 LIKE 全表扫描兜底。
  - 生效：仅在 ngram 回退为空时触发；默认关闭以降低扫描开销。
- `retrieval.sparse.enable_relation_sparse_fallback`
  - 功能：是否启用关系通道的稀疏回退。
  - 生效：在 relation-only / dual-path 的关系分支按 `auto/fallback_only/hybrid` 决策触发。
- `retrieval.sparse.relation_candidate_k`
  - 功能：关系稀疏检索候选上限。
  - 生效：关系 BM25 路每次召回候选数量上限。
- `retrieval.sparse.relation_max_doc_len`
  - 功能：关系稀疏检索返回内容最大长度。
  - 生效：关系内容（subject/predicate/object 拼接）截断长度。
- `retrieval.sparse.unload_on_disable`
  - 功能：插件关闭时是否卸载稀疏检索连接。
  - 生效：`on_disable` 阶段执行卸载。
- `retrieval.sparse.shrink_memory_on_unload`
  - 功能：卸载时是否调用 SQLite `shrink_memory`。
  - 生效：用于释放连接缓存占用。

### 稀疏检索触发规则（DualPath 实现细节）

- `sparse.mode = auto`
  - 触发条件（满足任一）：
    - embedding 生成失败或非法（空/NaN/Inf）
    - 向量候选为空
    - 向量最高分 `< 0.45`
- `sparse.mode = fallback_only`
  - 仅在 embedding 不可用时触发 sparse。
- `sparse.mode = hybrid`
  - 总是启用 sparse；若 embedding 同时可用，则并行形成双候选路。
- 额外约束：
  - 候选数会受 `retrieval.temporal.max_scan` 二次裁剪（时序检索模式）。

### 稀疏回退链路（段落）

- 查询 token 构造：最多取前 64 个 token，按 `OR` 组装 FTS `MATCH`。
- 主路径：`FTS5 bm25`。
- FTS miss 后：
  - 若 `enable_ngram_fallback_index=true`：走 `paragraph_ngrams` 倒排召回（优先）。
  - 若仍为空且 `enable_like_fallback=true`：才走 LIKE 子串扫描兜底。
- 冷启动成本：
  - 首次 `ensure_loaded()` 可能触发 `paragraphs_fts/relations_fts/paragraph_ngrams` 的 schema 检查与回填。

### `[retrieval.fusion]` 融合

- `retrieval.fusion.method`
  - 功能：融合方法（默认 `weighted_rrf`）。
  - 生效：段落向量召回与 BM25 候选融合策略。
- `retrieval.fusion.rrf_k`
  - 功能：RRF 平滑参数。
  - 生效：`1 / (rrf_k + rank)` 计算项。
- `retrieval.fusion.vector_weight`
  - 功能：向量路权重。
  - 生效：weighted RRF 中向量候选贡献比例。
- `retrieval.fusion.bm25_weight`
  - 功能：BM25 路权重。
  - 生效：weighted RRF 中稀疏候选贡献比例。
- `retrieval.fusion.normalize_score`
  - 功能：融合后是否做归一化。
  - 生效：在阈值过滤前将分数标准化。
- `retrieval.fusion.normalize_method`
  - 功能：归一化方法。
  - 生效：当前支持 `minmax`。

### 融合实现细节

- `fusion.method = weighted_rrf`
  - 段落双路融合公式：`score = w_vec/(k+rank_vec) + w_bm25/(k+rank_sparse)`。
  - 其中 `k = retrieval.fusion.rrf_k`。
- 权重修正：
  - `vector_weight + bm25_weight != 1` 时会自动归一化；
  - 若二者都为 0，自动回退为 `0.7/0.3`。
- 非 `weighted_rrf`：
  - 当前实现会退化为“拼接后按分数排序”的 legacy 行为。
- `normalize_score=true` 且 `normalize_method=minmax`：
  - 若分数全相同，归一化后统一为 `1.0`。

### `[retrieval.temporal]` 时序检索

- `retrieval.temporal.enabled`
  - 功能：时序检索总开关。
  - 生效：禁用后 `/query time`、`knowledge_query(time)`、`knowledge_search(time/hybrid)` 均直接返回禁用提示。
- `retrieval.temporal.allow_created_fallback`
  - 功能：无事件时间时是否回退使用 `created_at`。
  - 生效：时序筛选计算有效时间区间时使用。
- `retrieval.temporal.candidate_multiplier`
  - 功能：时序模式候选放大倍率。
  - 生效：先扩大召回再做时间过滤，提升召回率。
- `retrieval.temporal.default_top_k`
  - 功能：时序查询默认返回条数。
  - 生效：time/hybrid 模式未显式传 `top_k` 时作为默认值。
- `retrieval.temporal.max_scan`
  - 功能：时序模式最大扫描候选上限。
  - 生效：对放大后的候选数量做硬上限裁剪。
- `retrieval.time.skip_threshold_when_query_empty`
  - 功能：time 模式在 query 为空时是否跳过阈值过滤。
  - 生效：为 `true` 时与 legacy 行为对齐（仅按时序过滤，不做阈值筛除）。

### 时序参数格式约束（Action/Tool/Command）

- 适用入口：
  - `knowledge_search` action：`query_type=time|hybrid`
  - `knowledge_query` tool：`query_type=time`
  - `/query time`（或 `/query t`）
- 时间参数：
  - 仅支持 `YYYY/MM/DD` 或 `YYYY/MM/DD HH:mm`
  - `YYYY-MM-DD`、`2025/1/2`、自然语言（如“上周三晚上”）会被判为参数错误
- 日期展开规则：
  - `time_from`/`from` 为日期时自动展开到 `00:00`
  - `time_to`/`to` 为日期时自动展开到 `23:59`
- 说明：
  - 后端不做相对时间词解析；复杂自然语言时间需在模型侧先转换为绝对时间再传参。

## `[threshold]` 动态阈值

- `threshold.min_threshold`
  - 功能：最小阈值下界。
  - 生效：动态阈值结果会被 `clip` 到该下界以上。
- `threshold.max_threshold`
  - 功能：最大阈值上界。
  - 生效：动态阈值结果会被 `clip` 到该上界以下。
- `threshold.percentile`
  - 功能：百分位阈值计算参数。
  - 生效：`percentile` 与 `adaptive` 计算路径会用到。
- `threshold.std_multiplier`
  - 功能：标准差阈值系数。
  - 生效：`std_dev` 与 `adaptive` 计算路径会用到。
- `threshold.min_results`
  - 功能：最少保留结果数。
  - 生效：过滤后不足该值时，按分数补齐到该数量。
- `threshold.enable_auto_adjust`
  - 功能：自动阈值校准开关。
  - 生效：开启后再走一层 `_auto_adjust_threshold`。

## `[graph]` 图存储

- `graph.sparse_matrix_format`
  - 功能：图矩阵格式（`csr` 或 `csc`）。
  - 生效：`GraphStore` 初始化与后续格式切换策略。

## `[web]` 可视化服务

- `web.enabled`
  - 功能：Web 可视化服务开关。
  - 生效：插件启用时决定是否启动 FastAPI 服务；`/visualize` 也会检查此项。
- `web.port`
  - 功能：服务端口。
  - 生效：Web 服务监听端口。
- `web.host`
  - 功能：服务监听地址。
  - 生效：Web 服务绑定地址。

## `[advanced]` 高级

- `advanced.enable_auto_save`
  - 功能：自动保存总开关。
  - 生效：决定是否创建自动保存任务，以及循环内是否实际执行 `save_all()`。
- `advanced.auto_save_interval_minutes`
  - 功能：自动保存间隔（分钟）。
  - 生效：自动保存任务每轮 sleep 时使用；<=0 会回退为 5 分钟。
- `advanced.debug`
  - 功能：调试日志开关。
  - 生效：影响插件与命令/Action/Tool 的 debug 日志输出。
- `advanced.extraction_model`
  - 功能：知识抽取模型选择。
  - 生效：`/import text` 的 LLM 抽取模型优先使用该配置；`auto` 时按任务配置与兜底策略选型。

## `[summarization]` 总结导入

- `summarization.enabled`
  - 功能：总结导入总开关。
  - 生效：`summary_import` Action 与定时总结都会先检查此项。
- `summarization.model_name`
  - 功能：总结模型选择器。
  - 生效：支持 `auto`、任务名、模型名、数组、多选择器字符串（逗号分隔），由 `SummaryImporter` 解析成候选模型列表。
- `summarization.context_length`
  - 功能：总结读取历史消息条数。
  - 生效：拉取聊天记录时 `limit` 值。
- `summarization.include_personality`
  - 功能：总结提示词是否注入 bot 人设。
  - 生效：构造总结 prompt 时决定是否拼接 personality 文本。
- `summarization.default_knowledge_type`
  - 功能：总结入库时默认知识类型。
  - 生效：写入段落时转换为 `KnowledgeType`（无法识别时回退 `narrative`）。

## `[schedule]` 定时任务

- `schedule.enabled`
  - 功能：定时总结开关。
  - 生效：与 `summarization.enabled` 共同决定是否启动/执行定时导入循环。
- `schedule.import_times`
  - 功能：每日触发时间点列表（`HH:MM`）。
  - 生效：循环每分钟检查“是否刚跨过目标时间点”；匹配时触发批量总结导入。

## `[filter]` 聊天流过滤

- `filter.enabled`
  - 功能：过滤功能开关。
  - 生效：关闭时直接放行所有聊天流。
- `filter.mode`
  - 功能：`whitelist` 或 `blacklist`。
  - 生效：命中规则后在白/黑名单语义下分别放行或拒绝。
- `filter.chats`
  - 功能：过滤目标列表。
  - 生效：支持 `group:123`、`user:10001`、`private:10001`、`stream:<md5>` 或纯 ID（兼容匹配 stream/group/user）。
  - 注意：当列表为空时，`whitelist`=全部拒绝，`blacklist`=全部放行。

## `[routing]` 检索路由与兼容

- `routing.search_owner`
  - 功能：`search/time` 主责入口（`action|tool|dual`）。
  - 生效：
    - `action`：Action 主责（推荐）。
    - `tool`：Action 侧检索链路跳过，由 Tool 侧负责。
    - `dual`：Action 与 Tool 都可触发（依赖去重抑制重复执行）。
- `routing.tool_search_mode`
  - 功能：Tool 在 `search/time` 上的执行模式（`forward|disabled`）。
  - 生效：
    - `forward`：Tool 转发到统一检索执行服务（与 Action 同链路）。
    - `disabled`：Tool 的 `search/time` 直接拒绝并提示改走 Action。
  - 兼容：历史配置值 `legacy` 仍可读取，但会被按 `forward` 处理并输出废弃告警。
- `routing.enable_request_dedup`
  - 功能：是否开启短时请求去重。
  - 生效：同键请求在 TTL 内复用结果，并复用进行中的同键请求，避免 Action+Tool 同轮重复检索与重复强化。
- `routing.request_dedup_ttl_seconds`
  - 功能：去重窗口（秒）。
  - 生效：命中窗口内请求直接复用缓存结果；超时后重新执行检索。

## `[person_profile]` 人物画像

- `person_profile.enabled`
  - 功能：人物画像模块总开关。
  - 生效：关闭后 `/query person`、`knowledge_query(query_type=person)` 与自动注入都会直接禁用。
- `person_profile.opt_in_required`
  - 功能：是否要求显式开启注入。
  - 生效：为 `true` 时，只有执行 `/person_profile on` 的 `stream_id + user_id` 组合才会注入画像。
- `person_profile.default_injection_enabled`
  - 功能：无开关记录时的默认注入状态。
  - 生效：`opt_in_required=false` 或缺省记录时作为默认值。
- `person_profile.profile_ttl_minutes`
  - 功能：画像快照 TTL（分钟）。
  - 生效：画像查询优先复用 TTL 内快照，过期后重建并写入新版本。
- `person_profile.refresh_interval_minutes`
  - 功能：后台刷新周期（分钟）。
  - 生效：定时任务每轮 sleep 周期；最小有效周期由实现限制。
- `person_profile.active_window_hours`
  - 功能：活跃人物窗口（小时）。
  - 生效：仅刷新最近活跃窗口内、且开关启用范围内的人物。
- `person_profile.max_refresh_per_cycle`
  - 功能：每轮最大刷新人数。
  - 生效：限制单轮刷新负载，防止后台任务抢占过多资源。
- `person_profile.top_k_evidence`
  - 功能：画像构建时的证据数量上限。
  - 生效：控制向量/关系证据采样规模与输出稳定性。

## `[memory]` 记忆系统（V5）

- `memory.half_life_hours`
  - 功能：记忆半衰期（小时）。
  - 生效：维护循环按 `factor = 0.5^(interval/half_life)` 做全图权重衰减。
- `memory.base_decay_interval_hours`
  - 功能：维护循环间隔（小时）。
  - 生效：每轮 `sleep` 间隔（最小 60 秒）。
- `memory.prune_threshold`
  - 功能：冷冻候选阈值。
  - 生效：低于阈值的边进入 freeze/prune 流程候选。
- `memory.freeze_duration_hours`
  - 功能：冷冻保留时长（小时）。
  - 生效：超过时长的 inactive 关系会被物理修剪。
- `memory.enable_auto_reinforce`
  - 功能：自动强化开关。
  - 生效：关闭后检索命中关系不会进入强化缓冲。
- `memory.reinforce_buffer_max_size`
  - 功能：强化缓冲区上限（设计参数）。
  - 生效：当前未实际用于截断/限流（代码中为 TODO）。
- `memory.reinforce_cooldown_hours`
  - 功能：同一关系强化冷却期。
  - 生效：冷却期内且仍活跃时跳过重复强化。
- `memory.max_weight`
  - 功能：关系权重上限。
  - 生效：强化更新边权时的上限裁剪。
- `memory.revive_boost_weight`
  - 功能：inactive 关系复活时元数据增强值。
  - 生效：`mark_relations_active(..., boost_weight=...)`。
- `memory.auto_protect_ttl_hours`
  - 功能：强化/复活后的自动保护时长。
  - 生效：更新 `protected_until`（自动强化与手动强化都使用）。
- `memory.min_active_weight_protected`
  - 功能：保护期最低权重地板（设计参数）。
  - 生效：当前未在衰减或修剪逻辑中实际引用。
- `memory.enabled`
  - 功能：记忆维护主开关。
  - 生效：关闭后维护循环直接跳过衰减/强化处理。

---

## 附：代码支持但 `config.toml` 当前未显式列出的可选项

- `embedding.min_train_threshold`：SQ8 强制训练阈值（默认 40）。
- `retrieval.top_k_final`：DualPath 最终返回条数（默认 10）。
- `retrieval.relation_enable_path_search`：relation 语义回退后是否触发路径搜索（默认 true）。
- `retrieval.relation_path_trigger_threshold`：触发路径搜索分数阈值（默认 0.4）。
- `memory.orphan.enable_soft_delete` / `entity_retention_days` / `paragraph_retention_days` / `sweep_grace_hours`：孤儿节点 GC 的标记-清扫参数。
