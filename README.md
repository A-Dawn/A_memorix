# A_Memorix API 特化版

A_Memorix 已调整为 **API-first 独立服务**。核心目标是：

- 单进程 CLI 启动：`python -m amemorix serve --config ./config.toml`
- 保留历史兼容接口：`/api/*`
- 提供统一操作接口：`/v1/*`
- 支持任意 **OpenAI-compatible** 端点（不限定 OpenAI 官方）

> 本分支已移除 `plugin.py` 与 `components/*`，仅保留独立 API 服务形态。

## 主要能力

- 双路检索（向量 + 图谱）
- 时序检索（分钟级）
- 记忆维护（protect / reinforce / restore）
- 人物画像（registry / query / override）
- 总结导入（异步任务）
- Web 可视化（`/`）

## 安装

```powershell
pip install -r requirements.txt
```

或：

```powershell
pip install -e .
```

## 启动

```powershell
python -m amemorix serve --config ./config.toml
```

安装为脚本后：

```powershell
amemorix serve --config ./config.toml
```

## OpenAPI 兼容端点配置

推荐使用：

```toml
[embedding.openapi]
base_url = "https://your-openai-compatible-endpoint/v1"
api_key = "sk-..."
model = "text-embedding-3-large"
timeout_seconds = 30
max_retries = 3
```

兼容：

- 旧键 `[embedding.openai]` 仍可用
- 环境变量支持 `OPENAPI_*` 与 `OPENAI_*`

## 鉴权策略

默认启用 Bearer Token：

- 写接口（`POST/PUT/PATCH/DELETE`）默认强制鉴权
- 读接口默认放行
- 可通过 `auth.protect_read_endpoints=true` 开启全接口鉴权

## API 一览

- 兼容层：`/api/*`
- 新接口：`/v1/*`
  - 导入任务：`POST /v1/import/tasks`、`GET /v1/import/tasks/{task_id}`
  - 检索：`/v1/query/*`
  - 删除：`/v1/delete/*`
  - 记忆：`/v1/memory/*`
  - 人物：`/v1/person/*`
  - 总结任务：`POST /v1/summary/tasks`、`GET /v1/summary/tasks/{task_id}`
- 探活：`GET /healthz`、`GET /readyz`

## 数据兼容

默认沿用：

- `data/vectors`
- `data/graph`
- `data/metadata`

支持旧数据目录直接启动，并在启动阶段执行兼容迁移（非破坏追加）。

## 常用脚本

- 批量导入：`python scripts/process_knowledge.py --config ./config.toml`
- LPMM OpenIE 导入：`python scripts/import_lpmm_json.py <path> --config ./config.toml`
- 按来源删除：`python scripts/delete_knowledge.py --list --config ./config.toml`

## 文档

- 快速开始：[QUICK_START.md](QUICK_START.md)
- 端点与开发者文档：[DEVELOPER_API_GUIDE.md](DEVELOPER_API_GUIDE.md)
- 配置说明：[CONFIG_REFERENCE.md](CONFIG_REFERENCE.md)
- 导入指南：[IMPORT_GUIDE.md](IMPORT_GUIDE.md)
- 更新日志：[CHANGELOG.md](CHANGELOG.md)

## 许可证

AGPL-3.0
