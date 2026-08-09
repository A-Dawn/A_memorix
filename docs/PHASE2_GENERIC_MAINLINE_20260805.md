# 阶段2通用开发主线记录

记录日期：2026-08-05
实施仓库：`D:/Dev/rdev/A_memorix`
实施分支：`refactor/generic-v2`
迁移来源：`D:/Dev/rdev/MaiBot@2a5a9a9b83b1616dc4d058343c07e4071a9726b5`

## 实施结果

阶段2把经过阶段1验证的 MaiBot 内嵌实现迁回独立仓库，并建立了 A_memorix 2.x 通用开发主线。新主线采用标准 Python 包结构，不再把 MaiBot 插件目录当作安装入口。

核心工程契约如下：

| 项目 | 约定 |
| --- | --- |
| 分发名 | `a-memorix` |
| 导入名 | `a_memorix` |
| 当前版本 | `2.0.0a1` |
| Python | 3.12及以上 |
| 构建后端 | setuptools |
| 调用模型 | async-first |
| 数据目录 | 构造时显式注入 |
| 默认许可证 | AGPL-3.0-only |

## 结构迁移

源代码迁入 `src/a_memorix/`，测试迁入顶层 `tests/`。仓库增加 `pyproject.toml`，依赖拆为基础依赖、`vector`、`test` 和 `build` extras。

旧的插件入口、MaiBot 命令和工具组件、Web 页面、宿主脚本及1.x专用文档已经从2.x开发主线删除。1.x代码仍可通过 `legacy-v1.0.1` 标签和 `legacy/plugin-v1` 分支追溯。

## 宿主解耦

通用核心新增四类最小宿主接口：

- `EmbeddingProvider`
- `LLMProvider`
- `IdentityResolver`
- `MessageSource`

Embedding 适配器只依赖批量向量和模型指纹。模型路由、Episode 切分、画像、摘要及反馈修正通过注入接口调用外部模型和消息源。人物身份不再读取 MaiBot 数据库类型，日志也不再使用 MaiBot 日志模块。

静态扫描确认 `src/a_memorix` 和 `tests` 中没有 `src.*`、MaiBot 全局配置、聊天管理器、旧模型客户端或人物数据库模型导入。

## 数据目录契约

`SDKMemoryKernel` 现在必须显式接收 `data_dir`。元数据、图、向量、导入状态和运行时写者锁都从该目录派生，配置中的旧 `storage.data_dir` 不再覆盖构造参数。

测试中的重启、双池迁移、损坏恢复和导入任务也统一使用该契约。这样可以在进入阶段3前先建立明确的物理隔离边界，避免宿主路径推导重新渗入核心。

## 延后范围

Episode、画像、摘要等能力已完成依赖层面的通用化，但其领域行为仍带有长期对话场景的既有假设。本阶段保留这些经过验证的实现，不把它们定义为稳定外部协议。更完整的通用语义会结合阶段3的 RequestContext、namespace 和类型化 contract 继续收敛。

阶段2完成时，Web 导入管理器中的 MaiBot 数据迁移路径曾作为迁移期内部能力保留，不进入顶层公开 API。该临时路径已在阶段2.1移除。HTTP、MCP、RPC、Python SDK 稳定接口和社区扩展清单不属于本阶段。

## 验证结果

在 `D:/Dev/rdev/A_memorix` 执行：

```powershell
pytest -q
```

结果为629项通过、2项跳过、0项失败。跳过项是需要显式设置 `A_MEMORIX_RUN_LARGE_MIGRATION_TEST=1` 的大规模格式迁移压测。

发行验证结果如下：

- `ruff check src/a_memorix tests` 通过。
- `python -m build` 成功生成 `a_memorix-2.0.0a1-py3-none-any.whl` 和 `a_memorix-2.0.0a1.tar.gz`。
- Wheel 以 `--no-deps --target` 安装到系统临时目录后，可以在离开仓库源码目录的情况下导入顶层公开 API。
- 安装后的分发元数据版本和包内 `__version__` 均为 `2.0.0a1`。

## 退出条件

阶段2退出条件已经满足：

1. `refactor/generic-v2` 分支已经建立。
2. 标准 `src/` 包结构和 `pyproject.toml` 已落地。
3. 包名、版本、Python 下限和许可证元数据已统一。
4. 数据目录和外部能力改为显式注入。
5. 通用核心不再导入 MaiBot。
6. 迁移后的核心测试可在独立仓库运行。

下一阶段实现 Host Port 的稳定 contract、RequestContext 和强隔离 Namespace Runtime。
