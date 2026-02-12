# 更新日志 (Changelog)

## [0.4.0] - 2026-02-12

本次更新为 **时序检索增强正式版**，完成分钟级时间检索能力、Action/Tool/Command 三入口统一，以及文档与配置 schema 同步。

### 🚀 主要新增

#### 时序检索能力（分钟级）

- 新增统一时序查询入口：
  - `/query time`（别名 `/query t`）
  - `knowledge_query(query_type=time)`
  - `knowledge_search(query_type=time|hybrid)`
- 查询时间参数统一支持：
  - `YYYY/MM/DD`
  - `YYYY/MM/DD HH:mm`
- 日期参数自动展开边界：
  - `from/time_from` -> `00:00`
  - `to/time_to` -> `23:59`
- 查询结果统一回传 `metadata.time_meta`，包含命中时间窗口与命中依据（事件时间或 `created_at` 回退）。

#### 存储与检索链路

- 段落存储层支持时序字段：
  - `event_time`
  - `event_time_start`
  - `event_time_end`
  - `time_granularity`
  - `time_confidence`
- 时序命中采用区间相交逻辑，并遵循“双层时间语义”：
  - 优先 `event_time/event_time_range`
  - 缺失时回退 `created_at`（可配置关闭）
- 检索排序规则保持：语义优先，时间次排序（新到旧）。
- `process_knowledge.py` 新增 `--chat-log` 参数：
  - 启用后强制使用 `narrative` 策略；
  - 使用 LLM 对聊天文本进行语义时间抽取（支持相对时间转绝对时间），写入 `event_time/event_time_start/event_time_end`。
  - 新增 `--chat-reference-time`，用于指定相对时间语义解析的参考时间点。

#### Schema 与文档同步

- `_manifest.json` 同步补齐 `retrieval.temporal` 配置 schema。
- 配置 schema 版本升级：`config_version` 从 `3.0.0` 提升到 `3.1.0`（`plugin.py` / `config.toml` / 配置文档同步）。
- 更新 `README.md`、`CONFIG_REFERENCE.md`、`IMPORT_GUIDE.md`，补充时序检索入口、参数格式与导入时间字段说明。

### 🔖 版本信息

- 插件版本：`0.3.3` → `0.4.0`
- 配置版本：`3.0.0` → `3.1.0`

## [0.3.3] - 2026-02-11

本次更新为 **语言一致性补丁版本**，重点收敛知识抽取时的语言漂移问题，要求输出严格贴合原文语言，不做翻译改写。

### 🛠️ 关键修复

#### 抽取语言约束

- `BaseStrategy`:
  - 移除按 `zh/en/mixed` 分支的语言类型判定逻辑；
  - 统一为单一约束：抽取值保持原文语言、保留原始术语、禁止翻译。
- `NarrativeStrategy` / `FactualStrategy`:
  - 抽取提示词统一接入上述语言约束；
  - 明确要求 JSON 键名固定、抽取值遵循原文语言表达。

#### 导入链路一致性

- `ImportCommand` 的 LLM 抽取提示词同步强化“优先原文语言、不要翻译”要求，避免脚本与指令导入行为不一致。

#### 测试与文档

- 更新 `test_strategies.py`，将语言判定测试调整为统一语言约束测试，并验证提示词中包含禁止翻译约束。
- 同步更新注释与文档描述，确保实现与说明一致。

### 🔖 版本信息

- 插件版本：`0.3.2` → `0.3.3`

## [0.3.2] - 2026-02-11

本次更新为 **V5 稳定性与兼容性修复版本**，在保持原有业务设计（强化→衰减→冷冻→修剪→回收）的前提下，修复关键链路断裂与误判问题。

### 🛠️ 关键修复

#### V5 记忆系统契约与链路

- `MetadataStore`:
  - 统一 `mark_relations_inactive(hashes, inactive_since=None)` 调用契约，兼容不同调用方；
  - 补充 `has_table(table_name)`；
  - 增加 `restore_relation(hash)` 兼容别名，修复服务层恢复调用断裂；
  - 修正 `get_entity_gc_candidates` 对孤立节点参数的处理（支持节点名映射到实体 hash）。
- `GraphStore`:
  - 清理 `deactivate_edges` 重复定义并统一返回冻结数量，保证上层日志与断言稳定。
- `server.py`:
  - 修复 `/api/memory/restore` relation 恢复链路；
  - 清理不可达分支并统一异常路径；
  - 回收站查询在表检测场景下不再出现错误退空。

#### 命令与模型选择

- `/memory` 命令修复 hash 长度判定：以 64 位 `sha256` 为标准，同时兼容历史 32 位输入。
- 总结模型选择修复：
  - 解决 `summarization.model_name = auto` 误命中 `embedding` 问题；
  - 支持数组与选择器语法（`task:model` / task / model）；
  - 兼容逗号分隔字符串写法（如 `"utils:model1","utils:model2",replyer`）。

#### 生命周期与脚本稳定性

- `plugin.py` 修复后台任务生命周期管理：
  - 增加 `_scheduled_import_task` / `_auto_save_task` / `_memory_maintenance_task` 句柄；
  - 避免重复启动；
  - 插件停用时统一 cancel + await 收敛。
- `process_knowledge.py` 修复 tenacity 重试日志级别类型错误（`"WARNING"` → `logging.WARNING`），避免 `KeyError: 'WARNING'`。

### 🔖 版本信息

- 插件版本：`0.3.1` → `0.3.2`

## [0.3.1] - 2026-02-07

本次更新为 **稳定性补丁版本**，主要修复脚本导入链路、删除安全性与 LPMM 转换一致性问题。

### 🛠️ 关键修复

#### 新增功能

- 新增 `scripts/convert_lpmm.py`：
  - 支持将 LPMM 的 `parquet + graph` 数据直接转换为 A_Memorix 存储结构；
  - 提供 LPMM ID 到 A_Memorix ID 的映射能力，用于图节点/边重写；
  - 当前实现优先保证检索一致性，关系向量采用安全策略（不直接导入）。

#### 导入链路

- 修复 `import_lpmm_json.py` 依赖的 `AutoImporter.import_json_data` 公共入口缺失/不稳定问题，确保外部脚本可稳定调用 JSON 直导入流程。

#### 删除安全

- 修复按来源删除时“同一 `(subject, object)` 存在多关系”场景下的误删风险：
  - `MetadataStore.delete_paragraph_atomic` 新增 `relation_prune_ops`；
  - 仅在无兄弟关系时才回退删除整条边。
- `delete_knowledge.py` 新增保守孤儿实体清理（仅对本次候选实体执行，且需同时满足无段落引用、无关系引用、图无邻居）。
- `delete_knowledge.py` 改为读取向量元数据中的真实维度，避免 `dimension=1` 写回污染。

#### LPMM 转换修复

- 修复 `convert_lpmm.py` 中向量 ID 与 `MetadataStore` 哈希不一致导致的检索反查失败问题。
- 为避免脏召回，转换阶段暂时跳过 `relation.parquet` 的直接向量导入（待关系元数据一一映射能力完善后再恢复）。

### 🔖 版本信息

- 插件版本：`0.3.0` → `0.3.1`

## [0.3.0] - 2026-01-30

本次更新引入了 **V5 动态记忆系统**，实现了符合生物学特性的记忆衰减、强化与全声明周期管理，并提供了配套的指令与工具。

### 🧠 记忆系统 (V5)

#### 核心机制

- **记忆衰减 (Decay)**: 引入"遗忘曲线"，随时间推移自动降低图谱连接权重。
- **访问强化 (Reinforcement)**: "越用越强"，每次检索命中都会刷新记忆活跃度并增强权重。
- **生命周期 (Lifecycle)**:
  - **活跃 (Active)**: 正常参与计算与检索。
  - **冷冻 (Inactive)**: 权重过低被冻结，不再参与 PPR 计算，但保留语义映射 (Mapping)。
  - **修剪 (Prune)**: 过期且无保护的冷冻记忆将被移入回收站。
- **多重保护**: 支持 **永久锁定 (Pin)** 与 **限时保护 (TTL)**，防止关键记忆被误删。

#### GraphStore

- **多关系映射**: 实现 `(u,v) -> Set[Hash]` 映射，确保同一通道下的多重语义关系互不干扰。
- **原子化操作**: 新增 `decay`, `deactivate_edges` (软删), `prune_relation_hashes` (硬删) 等原子操作。

### 🛠️ 指令与工具

#### Memory Command (`/memory`)

新增全套记忆维护指令：

- `/memory status`: 查看记忆系统健康状态（活跃/冷冻/回收站计数）。
- `/memory protect <query> [hours]`: 保护记忆。不填时间为永久锁定(Pin)，填时间为临时保护(TTL)。
- `/memory reinforce <query>`: 手动强化记忆（绕过冷却时间）。
- `/memory restore <hash>`: 从回收站恢复误删记忆（仅当节点存在时重建连接）。

#### MemoryModifierTool

- **LLM 能力增强**: 更新工具逻辑，支持 LLM 自主触发 `reinforce`, `weaken`, `remember_forever`, `forget` 操作，并自动映射到 V5 底层逻辑。

### ⚙️ 配置 (`config.toml`)

新增 `[memory]` 配置节：

- `half_life_hours`: 记忆半衰期 (默认 24h)。
- `enable_auto_reinforce`: 是否开启检索自动强化。
- `prune_threshold`: 冷冻/修剪阈值 (默认 0.1)。

### 💻 WebUI (v1.4)

实现了与 V5 记忆系统深度集成的全生命周期管理界面：

- **可视化增强**:
  - **冷冻状态**: 非活跃记忆以 **虚线 + 灰色 (Slate-300)** 显示。
  - **保护状态**: 被 Pin 或保护的记忆带有 **金色 (Amber) 光晕**。
- **交互升级**:
  - **记忆回收站**: 新增 Dock 入口与专用面板，支持浏览删除记录并一键恢复。
  - **快捷操作**: 边属性面板新增 **强化 (Reinforce)**、**保护 (Protect/Pin)**、**冷冻 (Freeze)** 按钮。
  - **实时反馈**: 操作后自动刷新图谱布局与样式。

---

## [0.2.3] - 2026-01-30

本次更新主要集中在 **WebUI 交互体验优化** 与 **文档/配置的规范化**。

### 🎨 WebUI (v1.3)

#### 加载与同步体验升级

- **沉浸式加载**: 全新设计的加载遮罩，采用磨砂玻璃背景 (`backdrop-filter`) 与呼吸灯文字动效，提升视觉质感。
- **精准状态反馈**: 优化加载逻辑，明确区分“网络同步”与“拓扑计算”阶段，解决数据加载时的闪烁问题。
- **新手引导**: 在加载界面新增基础操作提示，降低新用户上手门槛。

#### 全功能帮助面板

- **操作指南重构**: 全面翻新“操作指南”面板，新增 Dock 栏功能详解、编辑管理操作及视图配置说明。

### 🛠️ 工程与规范

#### plugin.py

- **配置描述补全**: 修复了 `config_section_descriptions` 中缺失 `summarization`, `schedule`, `filter` 节导致的问题。
- **版本号**: `0.2.2` → `0.2.3`

### ⚙️ 核心与服务

#### Core

- **量化逻辑修正**: 修正了 `_scalar_quantize_int8` 函数，确保向量值正确映射到 `[-128, 127]` 区间，提高量化精度。

#### Server

- **缓存一致性**: 在执行删除节点/边等修改操作后，显式清除 `_relation_cache`，确保前端获取的关系数据实时更新。

### 🤖 脚本与数据处理

#### process_knowledge.py

- **策略模式重构**: 引入了 `Strategy-Aware` 架构，支持通过 `Narrative` (叙事), `Factual` (事实), `Quote` (引用) 三种策略差异化处理文本(准确说是确认实装)（默认采用 Narrative模式）。
- **智能分块纠错**: 新增“分块拯救” (`Chunk Rescue`) 机制，可在长叙事文本中自动识别并提取内嵌的歌词或诗句。

#### import_lpmm_json.py

- **LPMM 迁移工具**: 增加了对 LPMM OpenIE JSON 格式的完整支持，能够自动计算 Hash 并迁移实体/关系数据，确保与 A_Memorix 存储格式兼容。

#### Project

- **构建清理**: 优化 `.gitignore` 规则

---

## [0.2.2] - 2026-01-27

本次更新专注于提高 **网络请求的鲁棒性**，特别是针对嵌入服务的调用。

### 🛠️ 稳定性与工程改进

#### EmbeddingAPI

- **可配置重试机制**: 新增 `[embedding.retry]` 配置项，允许自定义最大重试次数和等待时间。默认重试次数从 3 次增加到 10 次，以更好应对网络波动。
- **配置项**:
  - `max_attempts`: 最大重试次数 (默认: 10)
  - `max_wait_seconds`: 最大等待时间 (默认: 30s)
  - `min_wait_seconds`: 最小等待时间 (默认: 2s)

#### plugin.py

- **版本号**: `0.2.1` → `0.2.2`

---

## [0.2.1] - 2026-01-26

本次更新重点在于 **可视化交互的全方位重构** 以及 **底层鲁棒性的进一步增强**。

### 🎨 可视化与交互重构

#### WebUI (Glassmorphism)

- **全新视觉设计**: 采用深色磨砂玻璃 (Glassmorphism) 风格，配合动态渐变背景。
- **Dock 菜单栏**: 底部新增 macOS 风格 Dock 栏，聚合所有常用功能。
- **显著性视图 (Saliency View)**: 基于 **PageRank** 算法的“信息密度”滑块，支持以此过滤叶子节点，仅展示核心骨干或全量细节。
- **功能面板**:
  - **❓ 操作指南**: 内置交互说明与特性介绍。
  - **🔍 悬浮搜索**: 支持按拼音/ID 实时过滤节点。
  - **📂 记忆溯源**: 支持按源文件批量查看和删除记忆数据。
  - **📖 内容字典**: 列表化展示所有实体与关系，支持排序与筛选。

### 🛠️ 稳定性与工程改进

#### EmbeddingAPI

- **鲁棒性增强**: 引入 `tenacity` 实现指数退避重试机制。
- **错误处理**: 失败时返回 `NaN` 向量而非零向量，允许上层逻辑安全跳过。

#### MetadataStore

- **自动修复**: 自动检测并修复 `vector_index` 列错位（文件名误存）的历史数据问题。
- **数据统计**: 新增 `get_all_sources` 接口支持来源统计。

#### 脚本与工具

- **用户体验**: 引入 `rich` 库优化终端输出进度条与状态显示。
- **接口开放**: `process_knowledge.py` 新增 `import_json_data` 供外部调用。
- **LPMM 迁移**: 新增 `import_lpmm_json.py`，支持导入符合 LPMM 规范的 OpenIE JSON 数据。

#### plugin.py

- **版本号**: `0.2.0` → `0.2.1`

---

## [0.2.0] - 2026-01-22

> [!CAUTION]
> **不完全兼容变更**：v0.2.0 版本重构了底层存储架构。由于数据结构的重大调整，**旧版本的导入数据无法在新版本中完全无损兼容**。
> 虽然部分组件支持自动迁移，但为确保数据一致性和检索质量，**强烈建议在升级后重新使用 `process_knowledge.py` 导入原始数据**。

本次更新为**重大版本升级**，包含向量存储架构重写、检索逻辑强化及多项稳定性改进。

### 🚀 核心架构重写

#### VectorStore: SQ8 量化 + Append-Only 存储

- **全新存储格式**: 从 `.npy` 迁移至 `vectors.bin`（float16 增量追加）和 `vectors_ids.bin`，大幅减少内存占用。
- **原生 SQ8 量化**: 使用 Faiss `IndexScalarQuantizer(QT_8bit)`，替代手动 int8 量化逻辑。
- **L2 Normalization 强制化**: 所有向量在存储和检索时统一执行 L2 归一化，确保 Inner Product 等价于 Cosine 相似度。
- **Fallback 索引机制**: 新增 `IndexFlatIP` 回退索引，在 SQ8 训练完成前提供检索能力，避免冷启动无结果问题。
- **Reservoir Sampling 训练采样**: 使用蓄水池采样收集训练数据（上限 10k），保证小数据集和流式导入场景下的训练样本多样性。
- **线程安全**: 新增 `threading.RLock` 保护并发读写操作。
- **自动迁移**: 支持从旧版 `.npy` 格式自动迁移至新 `.bin` 格式。

### ✨ 检索功能增强

#### KnowledgeQueryTool: 智能回退与多跳路径搜索

- **Smart Fallback (智能回退)**: 当向量检索置信度低于阈值 (默认 0.6) 时，自动尝试提取查询中的实体进行多跳路径搜索（`_path_search`），增强对间接关系的召回能力。
- **结果去重 (`_deduplicate_results`)**: 新增基于内容相似度的安全去重逻辑，防止冗余结果污染 LLM 上下文，同时确保至少保留一条结果。
- **语义关系检索 (`_semantic_search_relation`)**: 支持自然语言查询关系（无需 `S|P|O` 格式），内部使用 `REL_ONLY` 策略进行向量检索。
- **路径搜索 (`_path_search`)**: 新增 `GraphStore.find_paths` 调用，支持查找两个实体间的间接连接路径（最大深度 3，最多 5 条路径）。
- **Clean Output**: LLM 上下文中不再包含原始相似度分数，避免模型偏见。

#### DualPathRetriever: 并发控制与调试模式

- **PPR 并发限制 (`ppr_concurrency_limit`)**: 新增 Semaphore 控制 PageRank 计算并发数，防止 CPU 峰值过载。
- **Debug 模式**: 新增 `debug` 配置项，启用时打印检索结果原文到日志。
- **Entity-Pivot 关系检索**: 优化 `_retrieve_relations_only` 策略，通过检索实体后扩展其关联关系，替代直接检索关系向量。

### ⚙️ 配置与 Schema 扩展

#### plugin.py

- **版本号**: `0.1.3` → `0.2.0`
- **默认配置版本**: `config_version` 默认值更新为 `2.0.0`
- **新增配置项**:
  - `retrieval.relation_semantic_fallback` (bool): 是否启用关系查询的语义回退。
  - `retrieval.relation_fallback_min_score` (float): 语义回退的最小相似度阈值。
- **相对路径支持**: `storage.data_dir` 现在支持相对路径（相对于插件目录），默认值改为 `./data`。
- **全局实例获取**: 新增 `A_MemorixPlugin.get_global_instance()` 静态方法，供组件可靠获取插件实例。

#### config.toml / \_manifest.json

- **新增 `ppr_concurrency_limit`**: 控制 PPR 算法并发数。
- **新增训练阈值配置**: `embedding.min_train_threshold` 控制触发 SQ8 训练的最小样本数。

### 🛠️ 稳定性与工程改进

#### GraphStore

- **`find_paths` 方法**: 新增多跳路径查找功能，支持 BFS 搜索指定深度内的实体间路径。
- **`find_node` 方法**: 新增大小写不敏感的节点查找。

#### MetadataStore

- **Schema 迁移**: 自动添加缺失的 `is_permanent`, `last_accessed`, `access_count` 字段。

#### 脚本与工具

- **新增脚本**:
  - `scripts/diagnose_relations_source.py`: 诊断关系溯源问题。
  - `scripts/verify_search_robustness.py`: 验证检索鲁棒性。
  - `scripts/run_stress_test.py`, `stress_test_data.py`: 压力测试套件。
  - `scripts/migrate_canonicalization.py`, `migrate_paragraph_relations.py`: 数据迁移工具。
- **目录整理**: 将大量旧版测试脚本移动至 `deprecated/` 目录。

### 🗑️ 移除与废弃

- 废弃 `vectors.npy` 存储格式（自动迁移至 `.bin`）。

---

## [0.1.3] - 上一个稳定版本

- 初始发布，包含基础双路检索功能。
- 手动 Int8 向量量化。
- 基于 `.npy` 的向量存储。
