# A_Memorix Benchmark Report

更新日期：2026-02-24

## 1. 目的与范围

本报告记录当前仓库内可复现的四类验证结果：

1. **in-tree tests**（`pytest`）  
2. **时序查询 micro-benchmark**（`benchmarks/benchmark_temporal_query.py`，针对 `MetadataStore.query_paragraphs_temporal`）
3. **`/v1/query/time` HTTP 端到端 benchmark**（`benchmarks/benchmark_v1_time_e2e.py`）
4. **Docker 启动与接口冒烟测试**（容器化部署可用性验证）

## 2. 运行环境

- OS: `Microsoft Windows 11 家庭中文版 10.0.26200`
- CPU: `13th Gen Intel(R) Core(TM) i7-13650HX`
- Memory: `31.69 GB`
- Python: `3.12.7`
- 运行时间（本轮报告）：`2026-02-24 23:09:14 +08:00`

## 3. 测试结果（in-tree tests）

命令：

```powershell
python -m pytest -q
```

结果：

- `12 passed, 24 warnings in 23.07s`

覆盖范围（当前）：

- 时间解析：`tests/test_time_parser.py`
- 时序查询过滤语义：`tests/test_metadata_store_temporal.py`
- 总结导入流程：`tests/test_summary_importer.py`
- 统一检索执行流程：`tests/test_search_execution_service.py`

原始输出：`benchmarks/results/pytest_q.txt`

## 4. Benchmark 方法

脚本：

- `benchmarks/benchmark_temporal_query.py`

核心参数：

- `paragraphs`: 合成段落数量
- `queries`: 查询次数
- `query_window_minutes`: 时间窗口
- `query_limit`: 每次查询最大返回
- `person_filter_ratio`: 含 person 过滤的查询占比

统一说明：

- 每组实验都包含 warmup。
- 数据为脚本临时目录生成的合成数据。
- 指标包含 ingest 吞吐、query 吞吐、延迟分位（mean/p50/p95/p99）。

## 5. Benchmark 结果

### 5.1 结果表

| Case | paragraphs | queries | person_filter_ratio | ingest (para/s) | qps | mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | avg_hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small_seed7 | 1200 | 120 | 0.30 | 2256.737 | 3046.783 | 0.319 | 0.307 | 0.454 | 0.560 | 1.192 |
| medium_seed42 | 5000 | 300 | 0.25 | 2021.807 | 826.801 | 1.196 | 1.242 | 1.515 | 1.670 | 6.627 |
| medium_noperson_seed42 | 5000 | 300 | 0.00 | 2416.743 | 1081.195 | 0.917 | 0.840 | 1.275 | 1.328 | 9.313 |
| large_seed42 | 10000 | 500 | 0.25 | 1853.630 | 425.640 | 2.332 | 2.421 | 2.976 | 3.225 | 14.262 |

### 5.2 观察

1. **规模提升时延迟增加**：从 `1200` 到 `10000` 段落，p95 从 `0.454ms` 上升到 `2.976ms`。  
2. **person 过滤有可观成本**：在 `5000` 段落设置下，`person_filter_ratio=0.25` 相比 `0.0`，qps 从 `1081.195` 降到 `826.801`。  
3. **当前量级下仍保持低毫秒级**：`5000` 段落 + person 过滤条件下，p95 约 `1.515ms`。  

## 6. `/v1/query/time` 端到端 Benchmark（HTTP E2E）

### 6.1 方法说明

脚本：`benchmarks/benchmark_v1_time_e2e.py`

流程：

1. 生成合成时序数据（写入临时 `metadata.db`）。
2. 启动本地 `amemorix` 服务（临时配置，关闭 embedding 维度自动探测）。
3. 对 `POST /v1/query/time` 发起 HTTP 压测并统计延迟分位。

### 6.2 结果表

| Case | paragraphs | queries | person_filter_ratio | ingest (para/s) | qps | mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | avg_hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2e_small_seed7 | 800 | 80 | 0.30 | 2433.804 | 156.131 | 6.363 | 2.958 | 24.160 | 27.345 | 1.137 |
| e2e_medium_seed42 | 3000 | 200 | 0.25 | 2285.562 | 132.924 | 7.473 | 4.446 | 25.086 | 27.529 | 5.190 |
| e2e_medium_noperson_seed42 | 3000 | 200 | 0.00 | 2296.405 | 138.895 | 7.149 | 4.693 | 23.149 | 28.898 | 6.565 |
| e2e_large_seed42 | 5000 | 300 | 0.25 | 2273.851 | 121.208 | 8.197 | 5.627 | 26.166 | 29.620 | 8.033 |

### 6.3 观察

1. **E2E 延迟显著高于存储层 micro-benchmark**：反映了 HTTP、序列化、业务层编排等额外开销。  
2. **person 过滤在 E2E 路径同样有成本**：`3000` 段落下，`ratio=0.25` 的 qps 低于 `ratio=0.0`。  
3. **当前 E2E 水平**：在 `5000` 段落 + `0.25` person 过滤条件下，qps 为 `121.208`，p95 为 `26.166ms`。  

## 7. Docker 启动与接口冒烟测试

测试目标：

- 验证镜像可启动并通过探活（`/healthz`、`/readyz`）。
- 验证鉴权行为（未授权写请求返回 `401`，带 token 写请求成功）。
- 验证 `/v1/query/time` 在容器内可正常响应。

镜像与配置：

- image: `amemorix:0.6.1`
- 配置：临时挂载 `config.toml`，`auth.write_tokens=["test-token"]`
- 端口：随机映射（本次为 `127.0.0.1:58915 -> 8082`）

结果（2026-02-24）：

| item | value |
| --- | --- |
| `healthz_status` | `ok` |
| `readyz_ready` | `true` |
| `api_config_auth_write_tokens_len` | `1` |
| `unauthorized_write_status` | `401` |
| `authorized_upsert_success` | `true` |
| `registry_list_total` | `1` |
| `time_query_type` | `time` |
| `time_query_count` | `0` |

说明：

- 当前鉴权实现按 HTTP method 统一判断写操作，因此 `POST /v1/query/time` 也需带 Bearer token。
- 本项为容器可用性与接口行为冒烟验证，不替代性能 benchmark。

## 8. 可复现实验命令

```powershell
python -m pytest -q

python benchmarks/benchmark_temporal_query.py --paragraphs 1200 --queries 120 --query-window-minutes 90 --query-limit 30 --person-filter-ratio 0.3 --seed 7
python benchmarks/benchmark_temporal_query.py --paragraphs 5000 --queries 300 --query-window-minutes 120 --query-limit 50 --person-filter-ratio 0.25 --seed 42
python benchmarks/benchmark_temporal_query.py --paragraphs 5000 --queries 300 --query-window-minutes 120 --query-limit 50 --person-filter-ratio 0.0 --seed 42
python benchmarks/benchmark_temporal_query.py --paragraphs 10000 --queries 500 --query-window-minutes 120 --query-limit 50 --person-filter-ratio 0.25 --seed 42

python benchmarks/benchmark_v1_time_e2e.py --paragraphs 800 --queries 80 --query-window-minutes 90 --query-limit 30 --person-filter-ratio 0.3 --seed 7
python benchmarks/benchmark_v1_time_e2e.py --paragraphs 3000 --queries 200 --query-window-minutes 120 --query-limit 50 --person-filter-ratio 0.25 --seed 42
python benchmarks/benchmark_v1_time_e2e.py --paragraphs 3000 --queries 200 --query-window-minutes 120 --query-limit 50 --person-filter-ratio 0.0 --seed 42
python benchmarks/benchmark_v1_time_e2e.py --paragraphs 5000 --queries 300 --query-window-minutes 120 --query-limit 50 --person-filter-ratio 0.25 --seed 42
```

## 9. 原始结果文件

- `benchmarks/results/small_seed7.txt`
- `benchmarks/results/medium_seed42.txt`
- `benchmarks/results/medium_noperson_seed42.txt`
- `benchmarks/results/large_seed42.txt`
- `benchmarks/results/summary.json`
- `benchmarks/results/pytest_q.txt`
- `benchmarks/results/e2e_small_seed7.txt`
- `benchmarks/results/e2e_medium_seed42.txt`
- `benchmarks/results/e2e_medium_noperson_seed42.txt`
- `benchmarks/results/e2e_large_seed42.txt`
- `benchmarks/results/e2e_summary.json`
- `benchmarks/results/docker_smoke_20260224.json`
