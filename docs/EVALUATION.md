# 公共量化评测

A_memorix 的普通测试用于验证功能、数据安全和协议兼容性。公共量化评测回答另一类问题：底层检索改动是否降低了跨 session 记忆、长上下文记忆和代码库检索能力。

评测代码不会读取 `tests/data/benchmarks` 或 `tests/data/real_dialogues`。这些目录只用于本地研究并受 `.gitignore` 保护。公共评测下载的数据、Embedding cache、临时 Namespace 和结果统一保存在 `data/public-benchmarks`，不会进入 Git。

## 数据集

### LongMemEval-S Cleaned

[LongMemEval](https://github.com/xiaowu0162/LongMemEval) 是面向长期交互记忆的公开 benchmark。A_memorix 使用 MIT 许可的 `longmemeval_s_cleaned.json`，固定文件大小、SHA-256、500条案例和问题类型分布。默认评价470条非 abstention 题目，user 与 assistant evidence 都进入 corpus 和评分。

评测覆盖 single-session、multi-session、knowledge-update、temporal-reasoning 和 preference。默认以单条 user 或 assistant turn 为检索单元，gold 直接使用官方 `has_answer`；session 粒度包含完整对话，gold 直接使用 `answer_session_ids`。

### SWE-bench Lite

[SWE-bench](https://www.swebench.com/SWE-bench/guides/datasets/) 使用真实 GitHub issue 和对应修复。代码库记忆评测固定使用 SWE-bench Lite 300题 test split，并按官方 retrieval baseline 的口径：

- query 为 issue 的 problem statement
- corpus 为 base commit 下的 Python 源文件
- 测试目录不进入 corpus
- Gold 为修复 patch 修改的非测试源文件
- 大文件按重叠字符窗口切分，评分前按文件路径合并 chunk

数据优先从官方 Hugging Face parquet 下载。网络无法访问 Hugging Face 时，下载器会使用 Harvard CNS `orla` 仓库 `v1.2.15` 中的300个 JSON；两种格式都固定大小和 SHA-256，字段语义相同。benchmark 代码采用 MIT 许可，实际源码 corpus 继续受各上游仓库许可约束。源码优先通过 Git 从 `swe-bench-repos` mirror 获取；Git 服务不可用时，评测器改用 GitHub codeload 下载固定 commit 的归档。两条路径都只写入本地 cache，不修改用户自己的源码仓库。

LoCoMo 使用 CC BY-NC 4.0，不作为通用 release gate。CoIR 主要评价代码片段检索，可在后续作为补充，但不能替代 SWE-bench 的 issue-to-repository 场景。

## Embedding 配置

评测只调用 OpenAI-compatible Embedding API，不调用 LLM。根目录 `config.txt` 支持三行格式：

```text
<endpoint>
<api-key>
<model-id>
```

也支持 `endpoint=...`、`api_key=...`、`model=...`。`config.txt` 已由 `.gitignore` 明确排除。报告只记录模型 ID、向量维度和 endpoint 的 SHA-256，不记录 endpoint 原文或凭据。

安装评测依赖：

```bash
python -m pip install -e ".[test,evaluation,vector]"
```

## 运行

下载并校验 LongMemEval：

```bash
a-memorix-eval longmemeval download
a-memorix-eval longmemeval validate
```

先运行一题，确认 Provider、向量维度和本地 Faiss 环境：

```bash
a-memorix-eval longmemeval run --limit 1 --granularity turn
```

下列 smoke sample 从数据集顺序中为每种 question type 选择第一个 scored case，不按检索成绩挑选：

```powershell
a-memorix-eval longmemeval run `
  --question-id 6a1eabeb `
  --question-id 0a995998 `
  --question-id 7161e7e2 `
  --question-id 8a2466db `
  --question-id e47becba `
  --question-id gpt4_59149c77 `
  --granularity session `
  --top-k 50 `
  --output-dir data/public-benchmarks/longmemeval/results/smoke
```

运行全部可评分题：

```bash
a-memorix-eval longmemeval run --granularity turn
```

下载并运行 SWE-bench Lite：

```bash
a-memorix-eval swebench download
a-memorix-eval swebench run --limit 1
```

固定的代码检索 smoke case：

```powershell
a-memorix-eval swebench run `
  --instance-id pallets__flask-4045 `
  --top-k 20 `
  --output-dir data/public-benchmarks/swebench-lite/results/smoke
```

完整代码库评测会下载多个仓库并产生较多 Embedding 请求。SQLite cache 会复用相同模型下未变化的源码 chunk，中断后重新运行不会再次请求已缓存文本。Git clone、fetch 和需要远端 blob 的 checkout 会有限重试，网络持续不可用时仍会将对应 case 记为失败，留给下次 resume 重做。

Embedding 预热默认每批16条、最多3个并发请求。冷 cache 预热可以按 Provider 容量调整，但正式对照必须使用相同参数：

```powershell
a-memorix-eval swebench run `
  --embedding-batch-size 16 `
  --embedding-concurrency 3
```

服务端返回短暂过载时，每个批次最多尝试5次，并采用1、2、4、8秒退避。增加并发只适合补 cache，不应直接拿来和默认参数下的性能结果比较。

长任务可以使用 `--resume`。工具会在每个 case 完成后同步追加 `results.jsonl`，并通过 `run-state.json` 检查 schema、case、数据、模型、参数和运行环境。resume 会保留已完成 case，失败 case 会重新运行。同一输出目录由进程锁保护，第二个评测进程会直接退出，不能同时改写结果和临时 Namespace。多次进程拼接的 summary 可用于质量统计，不能作为性能 baseline；compare 的性能模式会拒绝 `resumed_case_count` 大于0的结果。

## 指标与发布门槛

两套 benchmark 都输出 `results.jsonl` 和 `summary.json`，包含：

- `recall_any@K`、`recall_all@K`、`recall_fraction@K`
- `precision@K`、`MRR`、`nDCG@K`
- embedding prewarm、初始化、写入、检索和单案例总耗时的 mean、median、P95
- Embedding 请求数、cache hit、cache miss 和公开指纹
- 评测 schema 版本、精确 case ID、数据 SHA-256、`top_k`、Embedding 批量与并发参数、粒度或 chunk 参数
- Python、操作系统、CPU 架构和关键依赖版本，不记录主机名或本地路径

评测层先填充 SQLite embedding cache，再通过公开 `batch_ingest_text` API 写入，每个写入批次最多100条。`embedding_prewarm` 记录向量读取或生成时间，`ingest` 记录本地写入时间，`total` 包含两者。同一套 case 至少运行两次。第一次补齐 cache，用于确认端到端流程；第二次要求 `cache_misses=0`，用于记录本地写入和检索性能。冷 cache 与热 cache 的耗时不能直接比较。

比较 baseline 与 candidate：

```powershell
a-memorix-eval compare `
  data/public-benchmarks/baseline/summary.json `
  data/public-benchmarks/candidate/summary.json
```

compare 会先核对 benchmark、case ID、数据 SHA-256、Embedding 指纹、预热批量与并发、`top_k`、粒度或 chunk 参数。Schema v1 的早期 summary 没有保存预热参数，当时命令行也不能修改它们，因此比较器按固定默认值16和3读取。性能比较还要求运行环境相同、两边都没有 cache miss。默认质量指标为 `MRR`、`nDCG@10`、`Recall-any@10`、`Recall-fraction@10`，默认允许绝对下降0.02；默认 ingest、search、total P95 比例上限分别为1.25、1.50、1.25。阈值都能通过命令行调整。只比较质量时可使用 `--quality-only`，但不能据此判断性能是否退化。

2026-08-10 的本地 smoke run 已完成：LongMemEval 6种 question type 各1题，SWE-bench Lite 1题，全部成功，至少一个目标均在Top-1命中。这部分结果只用于确认评测链路。

同日完成的 LongMemEval session 粒度完整热 cache 基线包含470个可评分 case，全部成功且 ID 唯一。整体 MRR 为0.932080、`nDCG@10` 为0.926632、`Recall-any@10` 为0.995745、`Recall-fraction@10` 为0.977163。分类型 MRR 中，single-session-assistant 为1.000、knowledge-update 为0.980、multi-session 为0.960、single-session-user 为0.930、temporal-reasoning 为0.890、single-session-preference 为0.780。写入耗时中位数477毫秒、P95 559毫秒；检索耗时中位数38毫秒、P95 47毫秒；单 case 总耗时中位数618毫秒、P95 714毫秒。该轮 `cache_misses=0`、`request_count=0`，可作为同机性能对照。

SWE-bench Lite 热 cache 基线包含全部300个 case，全部成功。整体 MRR 为0.689214、`nDCG@10` 为0.735735、`Recall-any@10` 与 `Recall-fraction@10` 均为0.876667。写入耗时中位数4.857秒、P95 19.047秒；检索耗时中位数153毫秒、P95 292毫秒；单 case 总耗时中位数5.417秒、P95 20.417秒。该轮共读取846,228条缓存向量，没有远程请求。按仓库看，`psf/requests`、`pydata/xarray`、`matplotlib/matplotlib` 的 MRR 较低，后续代码检索优化应优先检查这些场景。

性能分析还使用同一 Astropy case 做过修改前后的受控复跑。批量写入、延后向量与图持久化、取消无限配额下的重复目录扫描，以及修正向量恢复时不必要的写缓冲刷新后，写入从77.595秒降到6.002秒，排名与 MRR 保持一致，提升约12.93倍。

两套结果衡量的是 Embedding 检索，不包含 LLM 生成答案的正确率。发布仍按当前决定暂停；后续改动应使用同一数据、模型、参数和机器与本基线比较，不能把不同 Provider、粒度或 chunk 参数的分数直接放在一起。
