# 阶段2.1运行时边界收紧记录

记录日期：2026-08-08
实施仓库：`D:/Dev/rdev/A_memorix`
实施分支：`refactor/generic-v2`
对照来源：`D:/Dev/rdev/MaiBot@2a5a9a9b83b1616dc4d058343c07e4071a9726b5`

## 已确认约束

- namespace 是独立的数据、资源和生命周期边界，不与单个 Agent 强制一一绑定。
- Host Port 可以按 namespace 创建，禁止通过全局插件对象向核心传递能力。
- namespace 显式创建，删除采用停用、隔离、延迟清理的生命周期。
- 通用主线不包含 MaiBot 专属迁移、来源前缀和宿主管理动作。
- 默认许可证为 AGPL-3.0-only。协议变更或其他许可安排须发送邮件至 `contact@luminarc.tech` 申请。

## 运行时改造

内部应用服务改为依赖显式 `RuntimeServices` 协议。导入、检索、摘要和调优不再通过配置字典中的 `plugin_instance`、全局运行时注册表或任意宿主对象取回运行组件。配置仍作为普通数据传递，运行能力则通过类型化接口传递。

MaiBot数据库迁移任务、`maibot`目录别名、管理动作和`maibot.chat_history:`来源兼容已从通用层移除。未来的`integration/maibot`负责把旧数据转换为通用的写入请求和`chat_stream:`来源。

LPMM属于可复用格式转换，不归入MaiBot适配。阶段2.1补齐了随Wheel分发的转换器、`lpmm`可选依赖和真实Parquet测试。转换过程必须接收已观测的Embedding指纹，失败时不得发布`dual_ready.json`。

## 测试迁移台账

| MaiBot测试文件 | 阶段2.1处理 | 后续归属 |
| --- | --- | --- |
| `test_lpmm_convert.py` | 已迁为通用转换、边界、幂等、覆盖保护和引用校验测试 | main |
| `test_retrieval_type_filter_config.py` | 检索过滤行为已由`test_memory_graph_search_kernel.py`覆盖；MaiBot配置Schema不迁入 | 阶段3类型化配置契约 |
| `test_memory_service.py` | 内容是MaiBot调用包装、超时和旧动作别名，不属于内核 | `integration/maibot` |
| `test_host_service_config_update.py` | 内容是宿主配置文件备份、回滚和重载 | 阶段3控制面配置服务，MaiBot映射留在集成分支 |
| `test_host_service_shared_memory_groups.py` | 含共享群组、启动队列和宿主调度；不能在namespace定义前照搬 | 阶段3 RequestContext、namespace隔离和启动队列契约 |
| `test_maibot_migration_script.py` | 专门读取MaiBot数据库，不进入通用包 | `integration/maibot` |
| `test_memory_flow_service.py` | 内核摘要、事实账本已有测试；消息触发、人物选择和自动启动属于宿主编排 | 通用语义随阶段3继续收敛，MaiBot触发逻辑留在集成分支 |
| `test_chat_summary_writeback_integration.py` | 依赖MaiBot消息数据库和会话流，未直接搬运 | 阶段3以`MessageSource`和RequestContext重写通用集成测试 |
| `test_person_memory_writeback.py` | 依赖MaiBot人物模块；事实写入与账本已有通用测试 | 身份与画像语义通用化后重写，MaiBot选择逻辑留在集成分支 |

这份台账不把延期项视为已迁移。阶段3开始时，应先把表中标记为阶段3的行为转成不依赖任何Agent实现的contract，再建立namespace注册表和协议层。

## 验证结果

- `ruff check src/a_memorix tests`通过。
- `pytest -q`结果为635项通过、2项按环境变量默认跳过、0项失败。
- `python -m build`成功生成Wheel和sdist，Wheel包含通用LPMM转换器。
- 在不继承系统包的全新虚拟环境中安装基础Wheel后，可以从仓库外导入`SDKMemoryKernel`和`KernelSearchRequest`。
- 在同一干净环境按Wheel元数据安装`lpmm` extra后，可以导入`pyarrow`并运行`python -m a_memorix.scripts.convert_lpmm --help`。
- 静态扫描确认源码和测试中不存在MaiBot专属标识、`plugin_config`、`runtime_registry`或`self.plugin`。`plugin_instance`仅保留在禁止其回归的负向断言中。
