# A_Memorix API 特化版快速入门

## 1. 安装依赖

```powershell
pip install -r requirements.txt
```

## 2. 准备配置

复制示例：

```powershell
Copy-Item .\config.toml.example .\config.toml
```

至少确认以下配置：

```toml
[server]
host = "0.0.0.0"
port = 8082
workers = 1

[auth]
enabled = true
write_tokens = ["replace-with-your-token"]
protect_read_endpoints = false

[embedding.openapi]
base_url = "https://your-openai-compatible-endpoint/v1"
api_key = "sk-..."
model = "text-embedding-3-large"
```

## 3. 启动服务

```powershell
python -m amemorix serve --config ./config.toml
```

## 4. 健康检查

```powershell
curl http://127.0.0.1:8082/healthz
curl http://127.0.0.1:8082/readyz
```

## 5. 鉴权快速验证

未带 token 的写请求应返回 `401`：

```powershell
curl -X POST http://127.0.0.1:8082/v1/query/search -H "Content-Type: application/json" -d '{"query":"test"}'
```

带 token 请求：

```powershell
$token = "replace-with-your-token"
curl -X POST http://127.0.0.1:8082/v1/query/search `
  -H "Authorization: Bearer $token" `
  -H "Content-Type: application/json" `
  -d '{"query":"test","top_k":5}'
```

## 6. 导入任务（异步）

创建任务：

```powershell
$token = "replace-with-your-token"
$body = '{"mode":"paragraph","payload":{"content":"Alice maintains release checklist.","source":"quickstart"}}'
curl -X POST http://127.0.0.1:8082/v1/import/tasks `
  -H "Authorization: Bearer $token" `
  -H "Content-Type: application/json" `
  -d $body
```

轮询任务：

```powershell
curl -H "Authorization: Bearer $token" http://127.0.0.1:8082/v1/import/tasks/<task_id>
```

## 7. 常用端点

- 兼容接口：`/api/*`
- 新接口：`/v1/*`
- Web UI：首页 `http://127.0.0.1:8082/`

## 8. 兼容说明

- 推荐 `[embedding.openapi]`（OpenAI-compatible 统一入口）
- 旧键 `[embedding.openai]` 仍兼容
- 环境变量支持 `OPENAPI_*` 与 `OPENAI_*`
