# A_Memorix 端点与开发者文档

本文档用于开发者快速理解本项目的运行方式、端点与开发方式。

## 1. 项目定位

- 独立运行的 API 服务（不依赖 MaiBot 运行时）
- 启动命令：`python -m amemorix serve --config ./config.toml`
- 双端点体系：
  - 新接口：`/v1/*`（推荐给新接入方）
  - 兼容接口：`/api/*`（兼容历史 WebUI/调用方）

## 2. 运行与鉴权

### 2.1 安装与启动

```powershell
pip install -r requirements.txt
python -m amemorix serve --config ./config.toml
```

可选覆盖：

```powershell
python -m amemorix serve --config ./config.toml --host 127.0.0.1 --port 8082
```

### 2.2 鉴权规则

鉴权由 `amemorix/auth.py` 中间件统一处理：

- 默认开启：`auth.enabled=true`
- 写请求（`POST/PUT/PATCH/DELETE`）默认需要 Bearer Token
- 读请求默认放行；若 `auth.protect_read_endpoints=true` 则读请求也需鉴权
- `GET /healthz`、`GET /readyz` 永远放行

常见状态码：

- `401 Unauthorized`：缺失 token 或 token 错误
- `503 Authentication tokens are not configured.`：鉴权开启但 token 列表为空

请求头示例：

```http
Authorization: Bearer <your-token>
```

### 2.3 在线接口文档

FastAPI 默认可用：

- `GET /docs`
- `GET /openapi.json`

## 3. 端点清单

### 3.1 系统端点

| Method | Path | 鉴权 | 说明 |
| --- | --- | --- | --- |
| GET | `/healthz` | 否 | 存活探针 |
| GET | `/readyz` | 否 | 就绪探针 |
| GET | `/` | 读接口策略 | 返回 `web/index.html` |

### 3.2 `/v1/*` 操作接口（推荐）

| Method | Path | 鉴权 | 说明 |
| --- | --- | --- | --- |
| POST | `/v1/import/tasks` | 是 | 创建导入异步任务 |
| GET | `/v1/import/tasks/{task_id}` | 读接口策略 | 查询导入任务状态 |
| POST | `/v1/query/search` | 是 | 语义检索 |
| POST | `/v1/query/time` | 是 | 时序检索 |
| POST | `/v1/query/entity` | 是 | 实体查询 |
| POST | `/v1/query/relation` | 是 | 关系查询 |
| GET | `/v1/query/stats` | 读接口策略 | 统计信息 |
| POST | `/v1/delete/paragraph` | 是 | 删除段落 |
| POST | `/v1/delete/entity` | 是 | 删除实体 |
| POST | `/v1/delete/relation` | 是 | 删除关系 |
| POST | `/v1/delete/clear` | 是 | 清空数据 |
| POST | `/v1/memory/status` | 是 | 记忆状态 |
| POST | `/v1/memory/protect` | 是 | 记忆保护 |
| POST | `/v1/memory/reinforce` | 是 | 记忆强化 |
| POST | `/v1/memory/restore` | 是 | 回收站恢复 |
| POST | `/v1/person/query` | 是 | 人物画像查询 |
| POST | `/v1/person/override` | 是 | 设置画像覆盖 |
| DELETE | `/v1/person/override` | 是 | 删除画像覆盖 |
| POST | `/v1/person/registry/upsert` | 是 | 人物登记更新 |
| GET | `/v1/person/registry/list` | 读接口策略 | 人物登记分页 |
| POST | `/v1/summary/tasks` | 是 | 创建总结异步任务 |
| GET | `/v1/summary/tasks/{task_id}` | 读接口策略 | 查询总结任务状态 |

### 3.2.1 导入任务请求体

`POST /v1/import/tasks`

```json
{
  "mode": "paragraph",
  "payload": {
    "content": "Alice maintains release checklist.",
    "source": "demo",
    "time_meta": {
      "event_time": "2026/02/24 10:00"
    }
  },
  "options": {}
}
```

`mode` 支持：

- `text`
- `paragraph`
- `relation`
- `json`
- `file`

任务状态枚举（`task.status`）：

- `queued`
- `running`
- `succeeded`
- `failed`
- `canceled`

### 3.2.2 时间查询格式约束

`POST /v1/query/time` 的 `time_from` / `time_to` 仅支持：

- `YYYY/MM/DD`
- `YYYY/MM/DD HH:mm`

不接受 `YYYY-MM-DD`（会返回 `400`）。

### 3.2.3 删除/记忆接口要点

- `/v1/delete/paragraph` 字段名为 `paragraph_hash`，实现中也接受“段落文本匹配”
- `/v1/delete/relation` 的 `relation` 支持：
  - 关系 hash
  - `subject|predicate|object`
  - `subject predicate object`
- `/v1/memory/protect` / `reinforce` 的 `id` 可是关系 hash，也可以是查询文本（内部会解析）

### 3.2.4 人物接口要点

- `/v1/person/registry/upsert` 用于本地 `person_registry` 维护
- `/v1/person/query` 支持 `person_id` 或 `person_keyword` 解析
- `/v1/person/override` 的覆盖文本优先于自动画像文本

### 3.2.5 总结接口要点

`POST /v1/summary/tasks` 示例：

```json
{
  "session_id": "sess_demo_001",
  "source": "summary:demo",
  "messages": [
    {"role": "user", "content": "今天完成发布清单梳理。"},
    {"role": "assistant", "content": "已记录关键风险和截止时间。"}
  ],
  "context_length": 20
}
```

### 3.3 `/api/*` 兼容接口（历史契约）

| Method | Path | 鉴权 | 说明 |
| --- | --- | --- | --- |
| GET | `/api/graph` | 读接口策略 | 获取图谱数据 |
| POST | `/api/edge/weight` | 是 | 修改边权重 |
| DELETE | `/api/node` | 是 | 删除节点 |
| DELETE | `/api/edge` | 是 | 删除边 |
| POST | `/api/node` | 是 | 创建节点 |
| POST | `/api/edge` | 是 | 创建边 |
| PUT | `/api/node/rename` | 是 | 重命名节点 |
| POST | `/api/source/list` | 是 | 查询来源段落 |
| POST | `/api/source/batch_delete` | 是 | 按来源批量删除 |
| DELETE | `/api/source` | 是 | 删除单段落来源 |
| GET | `/api/memory/recycle_bin` | 读接口策略 | 回收站查询 |
| POST | `/api/memory/restore` | 是 | 回收站恢复 |
| POST | `/api/memory/reinforce` | 是 | 强化记忆 |
| POST | `/api/memory/freeze` | 是 | 冷冻记忆 |
| POST | `/api/memory/protect` | 是 | 保护记忆 |
| POST | `/api/person_profile/query` | 是 | 人物画像查询 |
| GET | `/api/person_profile/list` | 读接口策略 | 人物列表 |
| POST | `/api/person_profile/override` | 是 | 画像覆盖 |
| DELETE | `/api/person_profile/override` | 是 | 删除覆盖 |
| POST | `/api/save` | 是 | 手动保存 |
| GET | `/api/config` | 读接口策略 | 配置只读视图（脱敏） |
| POST | `/api/config/auto_save` | 是 | 运行时自动保存开关 |

兼容接口中的 `DELETE` 请求包含 JSON body，部分客户端默认不支持，调用时需显式带请求体。

## 4. 开发者指南

### 4.1 目录结构

```text
Http_fix/
  amemorix/
    __main__.py        # CLI 入口
    app.py             # FastAPI app factory
    settings.py        # 默认配置 + TOML + env 覆盖
    auth.py            # Bearer 鉴权中间件
    bootstrap.py       # 上下文构建（存储/检索/嵌入）
    context.py         # 运行时上下文
    task_manager.py    # 后台任务
    routers/v1_router.py
    services/          # API 编排层
  core/
    storage/           # vector/graph/metadata
    retrieval/         # 检索链路
    embedding/         # OpenAI-compatible embedding adapter
    utils/             # time parser / person profile / summary importer
  server.py            # /api/* 兼容路由
  scripts/             # 导入/删除/迁移脚本
```

### 4.2 请求处理链路

1. `amemorix/__main__.py` 解析 CLI 参数并调用 `create_app`
2. `amemorix/app.py` 构建 `AppContext`、注册鉴权中间件与路由
3. `/v1/*` 走 `routers -> services -> core`
4. `/api/*` 走 `server.py` 兼容路由
5. 任务型接口进入 `TaskManager` 异步 worker 执行

### 4.3 数据与任务模型

默认数据目录：

- `data/vectors`
- `data/graph`
- `data/metadata`

`metadata` 中包含任务/人物/总结相关表（由 `MetadataStore` 管理）：

- `async_tasks`
- `person_registry`
- `person_profile_snapshots`
- `person_profile_overrides`
- `transcript_sessions`
- `transcript_messages`

### 4.4 配置加载优先级

1. `amemorix/settings.py` 内置 `DEFAULT_CONFIG`
2. `config.toml`（通过 `--config` 指定或默认当前目录）
3. `AMEMORIX__...` 环境变量覆盖
4. OpenAI-compatible 别名环境变量补位：`OPENAPI_*`、`OPENAI_*`

端点配置合并规则：

- 推荐使用 `[embedding.openapi]`
- 兼容 `[embedding.openai]`
- 两者存在时，`openapi` 的非空字段优先

### 4.5 新增功能开发流程

1. 在 `amemorix/services/` 增加或扩展服务编排逻辑
2. 在 `amemorix/routers/v1_router.py` 定义请求模型与路由
3. 若为异步任务，扩展 `amemorix/task_manager.py` 与 `MetadataStore` 任务字段
4. 若涉及检索/存储算法，修改 `core/*`
5. 更新本文档与 `README.md`
6. 执行回归验证后再合并

### 4.6 本地验证建议

当前仓库未内置 `tests/` 套件，建议至少执行：

```powershell
python -m compileall amemorix core server.py scripts
python -m amemorix --help
python -m amemorix serve --config ./config.toml
```

再按端点做最小 E2E：

1. `POST /v1/import/tasks`
2. `GET /v1/import/tasks/{task_id}`
3. `POST /v1/query/search`
4. `POST /v1/delete/paragraph` 或 `POST /v1/delete/clear`

### 4.7 常见问题

- `401 Unauthorized`：检查 `Authorization: Bearer <token>`
- `503 Authentication tokens are not configured.`：`auth.enabled=true` 但 token 列表为空
- `400 时间格式错误`：`/v1/query/time` 使用 `YYYY/MM/DD` 或 `YYYY/MM/DD HH:mm`
- `/api/*` 的 `DELETE` 请求失败：确认客户端支持带 body 的 DELETE
