# A_Memorix 快速入门（面向 MaiBot）

这份文档只做一件事：让你最快把内容导入并把插件跑起来。

## 0. 先确认这 4 件事

1. 插件目录存在：`plugins/A_memorix/`
2. 主项目版本满足插件最低要求：`>= 0.12.1`
3. 你已经配置好 MaiBot 的模型（至少要有可用的 `embedding` 任务模型）
4. 你在项目根目录执行命令（即 `MaiBot/` 目录），并优先使用主程序的虚拟环境

---

## 1. 最快路径：批量导入 + 启用插件（推荐）

### 第一步：先激活主程序虚拟环境（强烈推荐）

根据官方部署文档（Windows / Linux / macOS / Docker），依赖应安装在隔离环境中，不建议直接在系统 Python 全局执行 `pip install`。
参考：<https://docs.mai-mai.org/manual/deployment/>

推荐做法：在主程序运行所用的同一个虚拟环境里安装 A_Memorix 依赖。

常见激活命令速查（覆盖主流工具）（注意！请根据你的实际使用环境选择命令！）：

#### 1) `venv` / `virtualenv`

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows CMD
.\.venv\Scripts\activate.bat

# Windows Git Bash / MSYS2
source .venv/Scripts/activate

# Linux / macOS (bash/zsh)
source .venv/bin/activate

# Linux / macOS (fish)
source .venv/bin/activate.fish

# Linux / macOS (csh/tcsh)
source .venv/bin/activate.csh
```

#### 2) `conda`

```powershell
# conda
conda activate <主程序环境名>
```

#### 3) `uv`（本质仍是 `.venv`）

```powershell
# 创建环境
uv venv

# 激活（命令与 venv 相同）
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
.\.venv\Scripts\activate.bat        # Windows CMD
source .venv/bin/activate           # Linux/macOS

# 或不激活，直接用 uv run
uv run python --version
```

说明：
- 若 PowerShell 执行脚本受限，可先执行：`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

激活后执行：

```shell
pip install -r requirements.txt --upgrade
pip install -r plugins/A_memorix/requirements.txt --upgrade
```

---

如果你不是本地虚拟环境部署，再按下列方式执行：

#### A. `uv` 环境（与主程序一致时使用）

```shell
uv venv
uv pip install -r requirements.txt --upgrade
uv pip install -r plugins/A_memorix/requirements.txt --upgrade
```

#### B. Docker 部署

在容器内安装（`core` 服务）：

```powershell
docker compose exec core pip install -r /MaiMBot/plugins/A_memorix/requirements.txt --upgrade
```

说明：
- Docker 方案走容器环境，不要在宿主机 Python 上装插件依赖。
- 官方 Linux 部署文档明确提示：`venv` 和 `uv` 不要混用。
- 如果你会频繁重建容器，建议把依赖固化到镜像或持久化方案中。

### 第二步：准备要导入的文本

把你的 `.txt` 文件放到：

`plugins/A_memorix/data/raw/`

建议用 UTF-8 编码。

### 第三步：离线批量导入（最快）

按你上一步选择的环境执行：

#### A. 主程序虚拟环境（推荐）

```shell
python plugins/A_memorix/scripts/process_knowledge.py
```

#### B. `uv` 环境

```shell
uv run python plugins/A_memorix/scripts/process_knowledge.py
```

#### C. Docker 环境

```shell
docker compose exec core python /MaiMBot/plugins/A_memorix/scripts/process_knowledge.py
```

常用参数：

```powershell
# 本地环境（uv/venv/conda）强制重导
python plugins/A_memorix/scripts/process_knowledge.py --force

# 本地环境聊天记录导入模式（会做时间语义抽取）
python plugins/A_memorix/scripts/process_knowledge.py --chat-log

# 本地环境聊天记录导入时，指定“相对时间”的参考点
python plugins/A_memorix/scripts/process_knowledge.py --chat-log --chat-reference-time "2026/02/12 10:30"

# Docker 示例（路径需使用容器内路径）
docker compose exec core python /MaiMBot/plugins/A_memorix/scripts/process_knowledge.py --force
```

### 第四步：启用插件

编辑 `plugins/A_memorix/config.toml`，确认：

```toml
[plugin]
enabled = true
```

### 第五步：重启 MaiBot

根据部署方式重启：

- 本地部署（uv/venv/conda）：重启主程序进程
- Docker：`docker compose restart core`

重启后，插件管理器会扫描 `plugins/` 目录并加载 A_Memorix。

### 第六步：验证是否成功

在聊天里发送：

```text
/query stats
```

如果看到段落/实体/关系数量，说明插件已生效。

---

## 2. 不走脚本也能快速导入（聊天命令）

适合临时补充少量内容：

```text
/import text 原神是一款由米哈游自主研发的...
/import paragraph 这是一条单段记忆
/import relation ARC|创建了|A_memorix
/import file ./劲爆小文件.txt
/import json {"paragraphs":[{"content":"2026年1月1日项目启动","event_time":"2026/01/01"}]}
```

导入后可立刻查询：

```text
/query s 游戏
```

---

## 3. 查询与可视化（最常用）

### 3.1 语义检索

```text
/query search 你要找的内容
# 简写
/query s 你要找的内容
```

### 3.2 时间检索（v0.5.0）

```text
/query time q="项目进展" from=2026/01/01 to="2026/01/31 18:30"
# 简写
/query t q=会议 from=2026/02/01 to=2026/02/07
```

时间格式只支持：
- `YYYY/MM/DD`
- `YYYY/MM/DD HH:mm`

### 3.3 人物画像（v0.5.0）

查询人物画像：

```text
/query person <person_id|别名>
# 简写
/query p <person_id|别名>
```

控制画像注入开关（按当前 `stream_id + user_id`）：

```text
/person_profile status
/person_profile on
/person_profile off
```

### 3.4 可视化编辑器

```text
/visualize
```

默认地址：

`http://localhost:8082`

可在 `plugins/A_memorix/config.toml` 的 `[web]` 里改端口和绑定地址。

### 3.5 Web Import 导入中心（v0.6.0 新增）

在可视化服务启动后，直接访问：

`http://localhost:8082/import`

导入中心支持：

- 上传文件 / 粘贴导入
- 本地扫描（alias + relative_path）
- LPMM OpenIE 导入
- LPMM 二进制转换（staging + switch）
- 时序回填
- MaiBot 迁移

并可查看任务/文件/分块三级状态，支持取消与“重试失败项（分块优先）”。

---

## 4. 建议你先改的 3 个配置

文件：`plugins/A_memorix/config.toml`

1. `embedding.model_name`
- 默认 `auto`。
- 想固定模型时，填你 `model_config.toml` 里可用的模型名。

2. `retrieval.sparse.mode`
- 默认 `auto`（推荐）。
- embedding 异常时会自动回退 BM25，稳定性更好。

3. `filter`（聊天流过滤）
- 默认是白名单模式 `whitelist` 且 `chats=[]`，表示全部放行。
- 如需只给某些群启用：

```toml
[filter]
enabled = true
mode = "whitelist"
chats = ["group:123456789"]
```

---

## 5. 运维命令速查

```text
/query stats                    # 看库状态
/delete paragraph <hash>        # 删段落
/delete relation <hash或S|P|O>  # 删关系
/delete clear                   # 清空知识库（危险）
/memory status                  # 记忆系统状态
/memory protect <query> 24      # 保护24小时
/memory reinforce <query>       # 手动强化
/person_profile status          # 查看人物画像注入状态
/person_profile on              # 开启人物画像注入
/person_profile off             # 关闭人物画像注入
```

---

## 6. 常见问题

### Q1: `/query` 没结果

按顺序检查：
1. `plugins/A_memorix/config.toml` 里 `[plugin].enabled` 是否为 `true`
2. 是否已重启 MaiBot
3. `/query stats` 是否显示段落数量 > 0
4. embedding 模型是否可用（模型配置是否正确）

### Q2: 批量导入脚本报模型相关错误

原因通常是主项目 `config/model_config.toml` 里没有可用模型或密钥不可用。先让主项目 LLM/embedding 正常，再跑导入脚本。

### Q3: 可视化页面打不开

检查：
1. `config.toml` 的 `[web].enabled = true`
2. 端口是否被占用（默认 `8082`）
3. 用 `/visualize` 触发一次服务启动

---

## 7. 进阶导入（可选）

1. LPMM OpenIE JSON 迁移：

```powershell
python plugins/A_memorix/scripts/import_lpmm_json.py <json文件或目录>
```

2. LPMM 存储直转（尽量不消耗 token）（注意！这一步的维度必须完全相同，否则会出现严重错误）：

```powershell
python plugins/A_memorix/scripts/convert_lpmm.py -i <lpmm数据目录> -o plugins/A_memorix/data
```

3. 回填旧数据的时间字段：

```powershell
python plugins/A_memorix/scripts/backfill_temporal_metadata.py --dry-run
python plugins/A_memorix/scripts/backfill_temporal_metadata.py --limit 50000
```
