# A_Memorix API 特化版配置参考（config.toml）

适用运行方式：

- `python -m amemorix serve --config ./config.toml`
- `amemorix serve --config ./config.toml`

当前默认配置基线：`amemorix/settings.py::DEFAULT_CONFIG`。

## 最小可用示例

```toml
[server]
host = "0.0.0.0"
port = 8082
workers = 1

[auth]
enabled = true
write_tokens = ["replace-with-your-token"]
read_tokens = []
protect_read_endpoints = false

[storage]
data_dir = "./data"

[embedding]
dimension = 1024
batch_size = 32
max_concurrent = 5
quantization_type = "int8"

[embedding.openapi]
base_url = "https://your-openai-compatible-endpoint/v1"
api_key = "sk-..."
model = "text-embedding-3-large"
timeout_seconds = 30
max_retries = 3

[tasks]
import_workers = 1
summary_workers = 1
queue_maxsize = 1024
```

## 核心配置

### `[server]`

- `host`: 监听地址
- `port`: 服务端口
- `workers`: 独立模式固定单进程，建议保持 `1`（CLI 会强制 `1`）

### `[auth]`

- `enabled`: 鉴权总开关
- `write_tokens`: 写接口 token 列表
- `read_tokens`: 读接口 token 列表（为空时读接口回退使用 `write_tokens`）
- `protect_read_endpoints`: 是否让读接口也强制鉴权

默认策略：写接口强制鉴权，读接口放行。

### `[cors]`

- `allow_origins`: CORS 白名单，默认 `[]`（不放开通配）

### `[storage]`

- `data_dir`: 数据根目录（包含 `vectors/graph/metadata`）

### `[embedding]`

- `dimension`: 目标维度
- `batch_size`: 批大小
- `max_concurrent`: 最大并发
- `quantization_type`: `float32/int8/pq`（当前实现以 `int8` 路径为主）
- `retry.max_attempts`
- `retry.max_wait_seconds`
- `retry.min_wait_seconds`

### `[embedding.openapi]`（推荐）

- `base_url`: 任意 OpenAI-compatible 端点
- `api_key`: 密钥
- `model`: embedding 模型
- `chat_model`: 可选；summary/person 的 chat 调用可用
- `timeout_seconds`: 超时
- `max_retries`: 请求重试

### `[embedding.openai]`（兼容旧键）

- 仍可使用；会与 `[embedding.openapi]` 合并
- `openapi` 的非空字段优先覆盖 `openai`

### `[tasks]`

- `import_workers`: 导入 worker 数
- `summary_workers`: 总结 worker 数
- `queue_maxsize`: 任务队列上限
- `summary_poll_interval_seconds`: 总结轮询间隔

### `[retrieval]`

常用项：

- `top_k_relations`
- `top_k_paragraphs`
- `top_k_final`
- `alpha`
- `enable_ppr`
- `ppr_alpha`
- `ppr_concurrency_limit`
- `enable_parallel`

#### `[retrieval.temporal]`

- `enabled`
- `allow_created_fallback`
- `candidate_multiplier`
- `default_top_k`
- `max_scan`

#### `[retrieval.search]`

- `smart_fallback.enabled`
- `smart_fallback.threshold`
- `safe_content_dedup.enabled`

#### `[retrieval.time]`

- `skip_threshold_when_query_empty`

#### `[retrieval.sparse]`

- `enabled`
- `backend`（当前为 `fts5`）
- `lazy_load`
- `mode`：`auto/fallback_only/hybrid`
- `tokenizer_mode`：`jieba/mixed/char_2gram`
- `enable_ngram_fallback_index`
- `enable_like_fallback`
- `enable_relation_sparse_fallback`
- 其余候选上限与长度控制参数

#### `[retrieval.fusion]`

- `method`（默认 `weighted_rrf`）
- `rrf_k`
- `vector_weight`
- `bm25_weight`
- `normalize_score`
- `normalize_method`

### `[threshold]`

- `min_threshold`
- `max_threshold`
- `percentile`
- `std_multiplier`
- `min_results`
- `enable_auto_adjust`

### `[memory]`

- `enabled`
- `half_life_hours`
- `base_decay_interval_hours`
- `prune_threshold`
- `freeze_duration_hours`
- `enable_auto_reinforce`
- `reinforce_cooldown_hours`
- `max_weight`
- `revive_boost_weight`
- `auto_protect_ttl_hours`

### `[person_profile]`

- `enabled`
- `profile_ttl_minutes`
- `refresh_interval_minutes`
- `active_window_hours`
- `max_refresh_per_cycle`
- `top_k_evidence`

#### `[person_profile.registry]`

- `page_size_default`
- `page_size_max`
- `match_strategy`

### `[summarization]`

- `enabled`
- `model_name`
- `context_length`
- `include_personality`
- `default_knowledge_type`

## 环境变量覆盖

支持两类：

- 通用嵌套覆盖：`AMEMORIX__SECTION__KEY=value`
- OpenAPI 别名：`OPENAPI_*` 与 `OPENAI_*`

例如：

- `OPENAPI_BASE_URL`
- `OPENAPI_API_KEY`
- `OPENAPI_EMBEDDING_MODEL`
- `OPENAPI_CHAT_MODEL`
- `OPENAPI_TIMEOUT_SECONDS`
- `OPENAPI_MAX_RETRIES`

## 兼容与遗留键

以下键仅用于读取旧配置文件时兼容，独立 API 运行可忽略：

- `[web]`
- `[schedule]`

API 特化版不依赖 MaiBot 运行时。
