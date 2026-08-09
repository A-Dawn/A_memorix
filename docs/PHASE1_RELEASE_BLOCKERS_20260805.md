# 阶段1：影响发布的问题修复记录

记录日期：2026-08-05
实施仓库：`D:/Dev/rdev/MaiBot`
实施分支：`dev`
基线快照：`snapshot/a-memorix-phase0-20260805`
修复提交：`2a5a9a9b83b1616dc4d058343c07e4071a9726b5`

## 问题范围

阶段0基线中有3个真实存储测试稳定失败。共同表现是 Runtime 重启后向量 ID 消失，日志显示完整的 V2 向量版本被判定为 `v2_fingerprint_mismatch`，随后被移动到隔离目录并替换为空版本。

根因不在向量二进制文件。Embedding Adapter 在启动时尚未完成真实请求，只能根据候选模型配置生成临时指纹。如果上一次运行实际使用了备用模型，临时指纹便可能与磁盘中记录的实际模型不同。旧状态机把这种尚未确认的兼容性问题当成存储损坏处理，造成了不必要的数据隔离。

## 修复后的状态机

向量故障现在分为三类：

| 类别 | 典型错误 | 处理方式 |
| --- | --- | --- |
| 待验证 | `embedding_fingerprint_unavailable` | 保留原文件、关闭向量读写、等待真实 embedding 探测 |
| 空间不兼容 | 维度不匹配、指纹缺失、指纹不匹配 | 保留原文件、关闭向量读写、标记需要显式重建 |
| 存储损坏 | V2 commit 损坏、向量文件对缺失或截断等 | 沿用恢复日志、隔离目录和空双池恢复流程 |

启动校验只接受来源为 `observed` 的模型指纹。没有 `source` 字段、结果确定的本地 Adapter 仍按已确认指纹处理，以兼容现有实现。来源为 `configured` 的指纹不能再触发 V2 向量加载、隔离或覆盖。

真实 embedding 探测现在可以在向量通道关闭时运行。探测完成后：

- 实际模型指纹与原版本一致时，Runtime 重新加载单池或双池，恢复检索和写入功能。
- 实际模型维度或指纹不一致时，原版本留在原路径，Runtime 进入 `incompatible` 状态。
- 用户执行 `rebuild_all_vectors` 后，Runtime 按当前模型重建向量，成功时恢复健康状态和向量功能。

缺失双池 ready manifest 的自愈路径也采用相同校验规则。实际模型未观测前，不会加载已有双池或根据临时指纹重建 manifest。

## 代码与测试

修复提交包含7个实现文件和7个测试文件。主要改动位于：

- `core/runtime/services/embedding_state_service.py`
- `core/runtime/services/runtime_lifecycle_service.py`
- `core/runtime/services/dual_vector_state_service.py`
- `core/runtime/services/vector_recovery_service.py`
- `core/runtime/services/vector_runtime_service.py`
- `core/utils/runtime_self_check.py`

新增和调整的回归场景包括：

- 候选模型发生回退后的进程重启
- 候选模型顺序变化
- 模型尚未观测时保留完整 V2 世代
- 指纹缺失时禁止自动隔离
- 维度和指纹不兼容时等待显式重建
- 显式重建后恢复向量读写能力
- 单池、双池、删除恢复 Outbox 和硬退出后的持久化完整流程

## 验证结果

在 `D:/Dev/rdev/MaiBot` 执行：

```powershell
$env:PYTHONPATH='.'
uv run pytest pytests/A_memorix_test -q --tb=short
```

结果：

- 756项通过
- 3项跳过
- 0项失败
- 27条既有警告，主要来自 SWIG 类型和 Pydantic 2.11 弃用提示
- 总耗时约104秒

受影响文件的 Ruff 检查全部通过，`git diff --check` 通过。阶段0记录的3个真实存储失败用例已单独复跑并全部通过。

## 本阶段范围

本阶段只修复 MaiBot 内嵌 A_memorix 的数据正确性问题，没有为了抽取通用层改变 MaiBot 接口、配置方式、运行结构或性能策略。MaiBot 工作区原有的未提交文件没有被纳入修复提交。

修复暂时保留在 MaiBot，作为阶段2迁入 A_memorix 2.0 时的已验证来源。主仓库不直接复制当前 MaiBot 的内部依赖，迁入时仍需按 Host Port、Namespace 和标准包结构拆分。

## 退出条件

阶段1退出条件已经满足：

1. 3个真实存储失败用例恢复通过。
2. 完整 A_memorix 测试集无失败。
3. embedding 空间不兼容不再被归类为存储损坏。
4. 未观测模型不会触发完整 V2 向量隔离。
5. 候选模型回退、顺序变化、指纹缺失和显式重建已有回归测试。

下一阶段建立独立通用开发主线、标准 Python 包结构和可迁移测试基线。
