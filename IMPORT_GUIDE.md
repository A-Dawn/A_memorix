# A_Memorix 导入指南与最佳实践 (Import Guide)

本文档旨在详细说明 A_Memorix 支持的各类导入文件格式、内部处理逻辑以及最佳实践，帮助用户构建高质量的知识库。

---

## 策略模式 (Strategies)

A_Memorix 采用 **策略模式 (Strategy-Aware)** 来处理不同类型的文本。导入脚本 (`process_knowledge.py`) 会尝试自动识别文本类型，也支持用户手动指定。

目前支持以下三种策略：

| 策略类型             | 适用场景                           | 核心逻辑                                                | 自动识别特征                         |
| :------------------- | :--------------------------------- | :------------------------------------------------------ | :----------------------------------- |
| **Narrative** (叙事) | 小说、同人文、剧本、长篇故事       | 按场景/章节切分，使用滑动窗口；提取“事件”和“角色关系”。 | `#`, `Chapter`, `***` 等章节标记     |
| **Factual** (事实)   | 设定集、维基百科、百科全书、说明书 | 按语义块切分，保留列表/表格结构；提取“SPO三元组”。      | 包含列表符号、定义格式 (`Term: Def`) |
| **Quote** (引用)     | 歌词、诗歌、名人名言、经典台词     | 按双换行符 (Stanza) 切分；原文即知识，不做提取。        | 平均行长短 (<20字符)，行数多         |

---

## 文件格式与最佳实践

### 1. 叙事文本 (.txt) - `Narrative`

适用于导入具有时间线和情节发展的内容。系统会重点提取**人物关系变化**和**关键事件**。

**最佳实践：**

- **明确章节分隔**：使用 Markdown 标题 (`# 第一章`) 或标准标识 (`Chapter 1`, `***`) 来分隔场景。这能帮助系统更准确地划分上下文。
- **段落清晰**：保持自然的段落分隔。系统使用滑动窗口 (Window 800 / Overlap 200) 处理长文本，自然的换行有助于保持语义完整。
- **人物称呼统一**：尽量在文中统一角色的称呼，有助于实体对齐。

**示例：**

```text
# 第一章：初遇

这是一个阳光明媚的早晨... (正文)

***

第二天晚上... (新场景)
```

> **💡 智能分块援救 (Chunk Rescue)**
> 如果你在叙事文本中嵌入了大段歌词或诗句（如角色在唱歌），系统会自动检测到这些“短行密集”的区块，并自动切换为 `Quote` 策略进行处理，防止它们被错误地概括或拆散。（不建议过度依赖本功能）

### 2. 事实文本 (.txt) - `Factual`

适用于构建世界观设定、物品介绍或规则说明。系统会重点提取**三元组信息 (Subject-Predicate-Object)**。

**最佳实践：**

- **结构化排版**：善用列表 (`-`, `1.`) 和定义格式 (`术语：解释`)。系统检测到这些结构时会尽量避免在中间切分。
- **信息密度**：单个段落尽量聚焦一个主题。
- **避免过度修饰**：尽量使用陈述句，减少文学修饰，有助于提高三元组提取的准确率。

**示例：**

```text
# 魔法系统设定

- **魔力源**：来自大气中的以太。
- **施法者**：必须具备“灵视”天赋。

## 禁忌
1. 禁止在闹市区施法。
2. 禁止进行人体炼成。
```

### 3. 引用文本 (.txt) - `Quote`

适用于无需概括、需要原文背诵或引用的内容。

**最佳实践：**

- **按节分块**：使用**双换行符**分隔不同的段落/小节（Stanza）。每个小节会被作为一个独立的知识块存储。
- **短行排版**：保留原文的换行格式。

**示例：**

```text
静夜思
床前明月光
疑是地上霜

举头望明月
低头思故乡
```

---

## LPMM 迁移导入

如果您拥有符合 LPMM (Large Scale Pre-trained Multimodal Model) 规范的 OpenIE JSON 数据 (`*-openie.json`)，可直接迁移。

**文件规范：**

- **格式**：JSON
- **必需字段**：
  - `docs`: 列表
    - `passage`: 原始段落文本
    - `extracted_triples`: 三元组列表 `[[s, p, o], ...]`
    - `extracted_entities`: 实体列表 (可选)

**命令：**

```bash
python plugins/A_memorix/scripts/import_lpmm_json.py <包含json的目录或文件路径>
```

此脚本会自动计算 Hash 并去重，将数据无缝转换到 A_Memorix 的存储格式中。

---

---

## 4. LPMM 二进制直转 (无需 Token)

如果您希望**完全保留** LPMM 的原始 Embedding 向量和图结构，且**不消耗任何 Token**，可以使用直接转换脚本。

> **⚠️ 注意**：这要求 A_memorix 配置的 Embedding 维度与原 LPMM 项目完全一致。

**命令：**

```bash
python plugins/A_memorix/scripts/convert_lpmm.py --input <LPMM数据目录> --output <A_memorix数据目录>
```

**示例：**

Assume LPMM data is in `data/lpmm_storage` and you want to output to `plugins/A_memorix/data`.

```bash
python plugins/A_memorix/scripts/convert_lpmm.py -i data/lpmm_storage -o plugins/A_memorix/data
```

此脚本会：

1. 直接读取 `.parquet` 文件并转换为 A_memorix 的二进制向量格式。
2. 直接读取 `.graphml` 或 `.pkl` 文件并转换为稀疏矩阵图。
3. 自动重建元数据。

## 常用命令速查

### 时间元数据导入（时序检索）

如果希望后续可按时间窗口（含分钟）精确检索，建议在导入时为段落提供时间字段。

#### 1. `/import json` 支持的段落时间字段

在 `paragraphs[*]` 中可直接传：

- `event_time`
- `event_time_start`
- `event_time_end`
- `time_range`（`[start, end]`）
- `time_granularity`（可选，未传会自动推断 `day/minute`）
- `time_confidence`（可选）

示例：

```json
{
  "paragraphs": [
    {
      "content": "2025年1月1日上午项目例会确定了里程碑。",
      "event_time_start": "2025/01/01 09:00",
      "event_time_end": "2025/01/01 10:30",
      "time_granularity": "minute",
      "time_confidence": 0.95
    }
  ]
}
```

#### 2. 脚本导入 (`process_knowledge.py`) 的时间输入

- 脚本在处理 JSON payload 时支持 `paragraphs[*].time_meta`；
- `time_meta` 可传 timestamp（秒）或时间字符串；
- 若未提供 `event_time*`，系统仍可回退 `created_at` 参与时序检索（取决于 `retrieval.temporal.allow_created_fallback`）。

#### 3. 查询时间参数（与导入不同）

注意：查询入口（Action/Tool/`/query time`）时间格式更严格，仅接受：

- `YYYY/MM/DD`
- `YYYY/MM/DD HH:mm`

### 自动导入 (推荐)

将 `.txt` 文件放入 `plugins/A_memorix/data/raw/` 后运行：

```bash
python plugins/A_memorix/scripts/process_knowledge.py
```

_主要参数：_

- `--force`: 强制重新处理所有文件
- `--type [narrative|factual|quote]`: 强制指定策略（不使用自动检测）
- `--chat-log`: 聊天记录导入模式。强制使用 narrative 策略，并通过 LLM 语义抽取 `time_meta`（`event_time` 或 `event_time_start/end`）
- `--chat-reference-time <datetime>`: 聊天记录模式下相对时间参考点（如 `2026/02/12 10:30`）；不传默认当前本地时间

### 清空知识库

如果需要重置所有数据（注意！此操作不可逆！）：

```bash
# 在聊天窗口输入
/delete clear
```

或直接删除 `plugins/A_memorix/data/` 下的 `vectors`, `graph`, `metadata` 目录。

---

## 简单实现示例 (Sample Templates)

以下示例可直接复制保存为对应的文件进行测试，或交由LLM进行样例学习

### 1. 叙事文本 (`plugins/A_memorix/data/raw/story_demo.txt`)

> 系统会自动识别 `#` 开头的章节，并提取其中的事件脉络。

```text
# 第一章：星之子

艾瑞克在废墟中醒来，手中的星盘发出微弱的蓝光。他并不记得自己是如何来到这里的，只依稀记得莉莉丝最后的警告：“千万不要回头。”

远处传来了机械守卫的轰鸣声。艾瑞克迅速收起星盘，向着北方的废弃都市奔去。他知道，那里有反抗军唯一的据点。

***

# 第二章：重逢

在反抗军的地下掩体中，艾瑞克见到了那个熟悉的身影。莉莉丝正站在全息地图前，眉头紧锁。

“你还是来了。”莉莉丝没有回头，但声音中带着一丝颤抖。
“我必须来，”艾瑞克握紧了拳头，“为了解开星盘的秘密，也为了你。”
```

### 2. 事实文本 (`plugins/A_memorix/data/raw/rules_demo.txt`)

> 系统会识别列表和定义，提取高精度的 S-P-O 三元组。

```text
# 联邦安全协议 v2.0

## 核心法则
1. **第一公理**：任何人工智能不得伤害人类个体，或因不作为而使人类个体受到伤害。
2. **第二公理**：人工智能必须服从人类的命令，除非该命令与第一公理冲突。

## 术语定义
- **以太网络**：覆盖全联邦的高速量子通讯网络。
- **黑色障壁**：用于隔离高危 AI 的物理防火墙设施。
```

### 3. 引用文本 (`plugins/A_memorix/data/raw/poem_demo.txt`)

> 系统会按双换行符切分，保留原文格式主要用于背诵或咏唱。

```text
致橡树

我如果爱你——
绝不像攀援的凌霄花，
借你的高枝炫耀自己；

我如果爱你——
绝不学痴情的鸟儿，
为绿荫重复单调的歌曲；

也不止像泉源，
常年送来清凉的慰籍；
也不止像险峰，
增加你的高度，衬托你的威仪。
```

### 4. LPMM JSON 数据 (`lpmm_data-openie.json`)

> 符合 LPMM 规范的中间格式。

```json
{
  "docs": [
    {
      "passage": "艾瑞克手中的星盘是打开遗迹的唯一钥匙。",
      "extracted_triples": [
        ["星盘", "是", "唯一的钥匙"],
        ["星盘", "属于", "艾瑞克"],
        ["钥匙", "用于", "遗迹"]
      ],
      "extracted_entities": ["星盘", "艾瑞克", "遗迹", "钥匙"]
    },
    {
      "passage": "莉莉丝是反抗军的现任领袖。",
      "extracted_triples": [
        ["莉莉丝", "是", "领袖"],
        ["领袖", "所属", "反抗军"]
      ]
    }
  ]
}
```
