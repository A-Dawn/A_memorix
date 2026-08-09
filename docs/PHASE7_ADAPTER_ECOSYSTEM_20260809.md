# 阶段7：Adapter 生态基础记录

日期：2026-08-09  
版本：2.0.0a2  
状态：主仓库契约与扩展索引基础设施已完成

## 本次完成范围

主仓库发布 Adapter Protocol v1和 Manifest Schema v1。适配器分为`remote`与`in_process`两种运行形态，二者不能混用私有能力。远程适配器只能依赖公开 gRPC、gRPC-Gateway HTTP/JSON或固定 namespace MCP；进程内适配器可以使用顶层公开 Python API和 Host Ports，但不能导入`a_memorix.core`。

Manifest 使用 TOML，包含稳定 ID、SemVer 版本、核心 PEP 440兼容范围、协议版本、transport、Host Port、许可证、源码地址和显式权限声明。权限覆盖公开 API、网络 origin、文件系统、环境变量和子进程。Manifest 不保存服务地址、namespace ID或任何密钥值。

CLI 提供`a-memorix adapter validate`和`a-memorix adapter schema`。校验器拒绝未知字段、重复声明、运行形态冲突、不安全网络 origin、错误环境变量名和不兼容核心版本。该命令不读取服务配置，也不连接运行中的 A_memorix 服务。

## 没有加入的能力

本阶段没有在主服务中增加第三方插件发现、自动安装、热加载或远程代码执行。进程内入口点由 Agent 宿主管理，协议 v1不臆造所有 Agent 共用的生命周期 ABI。Manifest 权限是审计声明，执行仍依赖 API Key、进程隔离、容器权限、文件系统 ACL和网络策略。

Episode、画像、摘要等领域接口仍未进入 Adapter Protocol。它们需要先形成 Agent 无关的公共 contract。

## 独立仓库状态

[A_memorix-extensions](https://github.com/A-Dawn/A_memorix-extensions)已作为公开仓库建立。首版实现了 manifest 不可变路径、ID与版本一致性、包名与入口点所有权唯一性、信任索引完整性、内容哈希检查以及固定核心校验器版本的 CI。官方、社区已验证、社区未验证等级由索引维护，不能由第三方 manifest 自行声明。

干净环境安装、公共协议契约、双 namespace 隔离、权限行为比对、依赖与许可证检查将在适配器申请已验证等级时执行，自动化仍需继续建设。官方扩展签名和撤回机制仍等待 ADR。私密安全报告统一使用`security@luminarc.tech`；扩展仓库在接受外部代码贡献前还需要确定贡献授权方式。

## 验收结果

远程与进程内示例均通过真实 CLI 校验，Schema 输出可以被标准 JSON 解析。新增12项契约测试覆盖运行形态互斥、版本范围、权限重复、网络通配符、环境变量格式和本地命令配置隔离。合并既有测试后共679项通过，2项可选大规模迁移压测按设计跳过。
