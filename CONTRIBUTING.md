# 开发指南

克隆后的仓库目录就是工程根目录。本机现有目录为 `D:\RSI\framework`。先读取 [工程约定](AGENTS.md)。训练端接手时，再读取 [Codex 接手说明](docs/CODEX_HANDOFF.md)。

## 模块职责

源码按配置、数据、筛选、模型、存储、评估、隔离执行、流程和命令分为 9 个子包。新增功能先确定归属，再选择相关接口。目录图和扩展步骤见 [代码结构与维护](docs/代码结构与维护.md)。

| 模块 | 职责 | 修改时重点检查 |
| --- | --- | --- |
| `rsi/configuration/validation.py` | 基础解析、规范序列化和哈希 | 类型、非有限数值、重复字段 |
| `rsi/configuration/settings.py` | 训练、硬件、评分版本和验收的字段类型 | 可空字段含义、字段兼容 |
| `rsi/configuration/contracts.py` | 实验配置与版本约束 | 未知字段、预算、研究模式前置条件 |
| `rsi/datasets/data.py` | 轨迹、监督掩码和数据隔离 | token 位置、数据角色、分数版本 |
| `rsi/curation/inspection.py` | 分数检查与结构校验报告 | 与选择器共用可选资格 |
| `rsi/curation/selection.py` | 策略排序与统一组批 | 确定性、配额、监督预算 |
| `rsi/datasets/labels.py` | 独立标签、技能表与轨迹快照关联 | 严格解析、未知质量、模拟权限、不可覆盖旧字段 |
| `rsi/curation/policy.py` | 独立策略与派生合同 | 冻结预算、版本一致、不可变性 |
| `rsi/curation/policy_selection.py` | 标签资格检查与解释报告 | 当前评分、上下文身份、预算与配额失败区分 |
| `rsi/curation/feedback.py`、`rsi/curation/history.py` | 模拟反馈与显式批次历史 | 计数分母、证据类型、成员引用 |
| `rsi/curation/dispatch.py` | 调度上下文与纯补丁应用 | 显式白名单、完整重校验、身份与证据一致 |
| `rsi/curation/selection_io.py` | 输入读取与本地产物发布 | 单写者、拒绝覆盖、清单校验、失败清理 |
| `rsi/curation/selection_demo.py`、`rsi/commands/selection_cli.py` | 合成演示与命令适配 | 固定种子、无模型依赖、NOT_RUN 标记 |
| `rsi/curation/comparison.py` | 不依赖数据库的对照计划计算 | 等监督量、总预算、计划字段 |
| `rsi/storage/schema.py` | 数据库建表与版本检查 | 先检查已有格式，禁止隐式修复 |
| `rsi/storage/state.py` | SQLite 快照、事件和成本 | 事务、幂等、恢复、并发预算预留 |
| `rsi/models/backends.py` | 训练与评估接口 | 输入产物、实际成本、失败传播 |
| `rsi/commands/cli.py` | 命令参数与输出 | 退出码、错误信息、可重复调用 |
| `rsi/datasets/tokenization.py` | 消息到 token 的映射 | 模板边界、错误动作排除、往返一致 |
| `rsi/models/scoring.py` | 评分类型与参考百分位 | 参考组隔离、相同值的中点秩 |
| `rsi/models/hf_backend.py` | 单卡模型执行 | 目标移位、实际监督量、产物保存 |
| `rsi/workflows/pilot.py` | 工程流程编排 | 两组独立起点、事件与成本对应 |
| `rsi/storage/artifacts.py` | 内容哈希与产物校验 | 拒绝覆盖、文件变更、路径范围 |
| `rsi/models/devices.py` | UUID 绑定与设备核验 | 拒绝数字绑定、错误型号和错误 UUID |
| `rsi/storage/checkpoints.py` | 步骤意图、检查点发布与恢复核对 | 不完整步骤、旧检查点、输入变更 |
| `rsi/storage/leases.py` | 同一运行目录的单写入者锁 | 并发拒绝、进程退出后释放 |
| `rsi/evaluation/coding_tasks.py` | 独立编程任务格式 | 非空测试、任务分组、最终测试隔离 |
| `rsi/evaluation/bounded_python.py` | 受限整数表达式解释 | 语法白名单、整数范围、任务失败 |
| `rsi/evaluation/coding_evaluation.py` | 生成与逐任务执行 | 完整分母、逐任务证据、失败保留 |
| `rsi/workflows/evaluation_run.py` | 基线与两组候选的独立评估 | 模型身份、调用预算、任务清单 |
| `rsi/evaluation/acceptance.py` | 配对验收纯函数 | 相同任务、样本量、增益与能力保持 |
| `rsi/datasets/mbpp.py` | 官方格式、划分与重复分组 | 划分边界、跨划分重复、代码只解析不执行 |
| `rsi/datasets/dataset_import.py` | 来源冻结与参考解检查 | 先选子集、失败保留、最终测试不执行 |
| `rsi/datasets/data_compile.py` | 数据包到现有轨迹接口 | 文件哈希、真实分词、禁止静默截断 |
| `rsi/models/model_inputs.py` | 模型与 tokenizer 内容绑定 | 与 pilot 共用同一版本规则 |
| `rsi/evaluation/python_tasks.py` | 通用 Python 任务格式 | 非空断言测试、角色隔离 |
| `rsi/execution/sandbox.py` | 宿主侧隔离编排 | 启动探针、超时、输出上限、进程清理 |
| `rsi/execution/sandbox_worker.py` | 沙箱内可信启动器 | 先设上限与系统调用过滤，再执行代码 |
| `rsi/evaluation/isolated_evaluation.py` | 通用代码逐任务评估 | 测试调用记录、完整任务分母、基础设施错误 |
| `rsi/curation/packing.py` | 完整轨迹精确组批 | 与穷举结果一致、重复输入、预算无解、规模上限 |
| `rsi/workflows/real_run.py` | 真实对照准备 | 产物哈希、评分预算、模型起点、禁止最终测试角色 |
| `rsi/workflows/training_run.py` | 两种入口共用的训练执行与恢复 | 计划不重算、检查点与账本一致、已完成分支不重跑 |
| `rsi/workflows/scoring_run.py` | 评分调用与成本记录 | 调用意图、异常留痕、实际消耗 |
| `rsi/datasets/python_diagnostics.py` | 静态关键字诊断标注 | 注释和字符串排除、字符位置、训练掩码不变 |
| `rsi/models/diagnostics.py` | 固定诊断集合聚合 | token 加权均值、缺失不等于零、技能分母 |
| `rsi/workflows/reporting.py` | 开发报告生成 | 报告与账本一致、逐任务重算判定、成本范围 |
| `rsi/storage/provenance.py` | 递归源码身份 | 子包覆盖、相对路径、内容变更 |
| `rsi/_compat.py` | 旧模块导入兼容 | 模块对象一致、补丁行为、命令入口 |
| `rsi/datasets/training_inputs.py`、`rsi/workflows/training_compile.py` | 标准来源编译与显式标签重绑 | 消息映射、真实 tokenizer、证据与掩码交集 |
| `rsi/models/model_resolution.py`、`rsi/workflows/execution_plan.py` | 模型定位和单批冻结计划 | 内容身份、原顺序、LoRA 兼容与禁止重新选数 |
| `rsi/workflows/attempt_run.py`、`rsi/models/training_mock.py` | 单次训练与显式测试后端 | 单计划事务、使用量、失败成本、完整步骤恢复 |
| `rsi/evaluation/fit_evaluation.py` | 整条轨迹与技能片段拟合 | 目标移位、完整词表熵、token 加权、缺失值 |
| `rsi/evaluation/task_adapters.py`、`rsi/evaluation/execution_evaluation.py` | 版本化任务与单模型执行 | 隔离、隐藏测试、技能重叠、完整分母 |
| `rsi/datasets/content_identity.py`、`rsi/evaluation/agent_tasks.py` | 内容身份与多步 Agent 环境协议 | 实际内容绑定、环境初态、隐藏判定器、动作轨迹与资源释放 |
| `rsi/evaluation/round_feedback.py`、`rsi/workflows/validation_run.py` | 四项测量和可复算反馈 | 来源、证据哈希、阶段缓存、完整测量后验收 |
| `rsi/workflows/benchmark_run.py` | 独立最终测试 | 禁止进入反馈和验收、复用执行器 |
| `rsi/storage/execution_artifacts.py`、`rsi/commands/training_cli.py`、`rsi/workflows/training_demo.py` | 执行清单、命令和演示 | 源码与依赖身份、拒绝覆盖、mock 不冒充实测 |

配置与数据模块不调用训练后端。选择策略不直接写数据库。命令行层调用业务接口，不复制业务规则。接入真实后端时，外部执行与状态提交必须保持明确边界。

测试按配置、数据、选择、状态、命令和回归主题分文件。`tests/helpers.py` 读取公共合成样本。`tests/test_cli_process.py` 用独立进程检查完整命令流程。不要让业务代码依赖测试模块。

轨迹解析按质量、分词、监督、诊断片段、使用历史和评分分为命名函数。修改一类校验时，先查看对应函数及该模块测试。

解析后的配置、评分和批次使用不可变字段。技能权重与预算使用只读映射。`TrajectoryRecord.raw` 和 `to_dict()` 返回独立副本。修改副本不会更新已校验对象。修改输入后必须重新解析。

`comparison.build_comparison` 只负责计算。`StateStore.prepare_comparison` 负责检查运行状态和原子保存。现阶段组批计算仍在保存事务内进行，以保持状态检查与提交一致。真实训练不得放在数据库事务内。

## 本地开发流程

1. 在 PowerShell 中进入工程目录。

   ```powershell
   Set-Location D:\RSI\framework
   ```

2. 使用 Python 3.11 或更高版本修改相关模块。当前运行源码无需安装第三方依赖。
3. 为行为变化补充测试。文档或路径调整无需编写重复测试。
4. 运行测试。

   ```text
   python -m unittest discover -s tests -v
   ```

5. 检查演示输入。

   ```text
   python -m rsi validate --config configs/smoke.json --data examples/trajectories.demo.jsonl --require-current-scores
   ```

6. 如果修改命令或状态流程，按 [运行手册](docs/第一阶段运行手册.md) 使用新 run-id 检查完整流程。
7. 更新接口说明和验证记录。记录实际运行的命令及结果。

## 风格与兼容

Python 代码使用四个空格缩进。使用明确的英文标识符。中文说明遵循本仓库 AGENTS.md 的表达规范。`.editorconfig` 为支持它的编辑器提供基础格式设置。

当前没有配置自动格式检查、静态类型检查或持续集成。类型标注要求属于开发约定。不要将它写成已经通过的自动检查。引入这些工具时，需固定配置并验证现有代码。

本次目录迁移保持包名 `rsi`、命令参数和数据格式不变。后续接口不兼容的改动需提供版本说明。数据库迁移需先验证备份与恢复方法。

0.7.0 的新代码使用完整子包路径，例如 `from rsi.curation.selection import build_batch`。旧的 `rsi.selection` 指向同一实现模块。不要在新实现中继续使用旧路径。子包不集中导出所有模块，避免导入顺序耦合。

`tests/test_package_layout.py` 检查兼容导入、旧路径补丁、命令入口、递归源码身份和沙箱启动器位置。训练恢复仍要求源码身份一致。0.6.0 运行需要保留当时的代码目录，不要用 0.7.0 恢复。重构不迁移已有运行产物。

0.2.0 将部分内部 Python 字典接口改为字段类型。JSON 格式仍为 1.0，数据库版本仍为 1。转换方法见 [接口说明](docs/数据与接口说明.md) 的 Python 接口一节。

完整研究方案由项目负责人另行提供。仓库的执行目标与当前边界见 [Codex 接手说明](docs/CODEX_HANDOFF.md)。

## 可选模型依赖

模型模块在执行时才导入 torch、transformers 和 peft。不要让基础配置和数据库命令依赖 GPU 包。

pyproject.toml 的 hf 依赖组记录本次服务器验证使用的版本。CUDA 构建版本和设备信息另外保存到每次运行的 environment.json。远程手册使用已有环境，不要求重复安装大型依赖。

tests/test_hf_optional.py 在安装依赖后执行离线 CPU 小模型测试。缺少依赖时，该项明确跳过。其他测试不需要下载模型。报告测试结果时，分别说明通过数和跳过数。

状态库中的训练使用与成本在同一事务中写入。后端完成一个反向传播步骤后调用账本接口。0.4.0 在优化器更新后保存完整检查点。完整步骤可以恢复。步骤内中断、账本差异或最终保存阶段中断会被拒绝。不能擦除账本后重跑。

测试检查连续训练与恢复训练的参数、随机状态和学习率计划。模型测试依赖可选环境。Windows 沙箱的系统临时目录不允许目录重命名，因此检查点测试在工程根目录创建临时目录，并在测试结束后清理。

Linux 隔离集成测试通过 `RSI_RUN_SANDBOX_TESTS=1` 显式启用。默认测试不依赖宿主安装 bwrap。集成测试启用后，隔离依赖不可用应失败，不能静默跳过或退回普通进程。

`sandbox_worker.py` 是唯一执行外部 Python 代码的位置。它不是宿主执行入口。新增代码格式或数据适配器不得在宿主使用 `exec`、`eval` 或导入待测代码。系统调用列表必须配套正常任务测试和拒绝行为测试。

0.6.0 增加合同 1.1，要求明确选择组批规则。合同 1.0 保持原贪心行为。数据库版本不变。恢复身份包含整个源码清单；升级代码后不要在新版本目录恢复旧运行。

修改训练损失或诊断逻辑时，运行可选模型测试。该测试同时检查变长监督量、诊断开关不改变模型参数、完整恢复参数一致和失败后成本保留。

真实数据开发对照使用 `real_run` 准备输入，使用 `training_run` 执行冻结计划。不要把这些流程复制到新数据集适配器。新适配器应输出已有来源格式或已校验轨迹。

独立标签入口使用 `policy_selection.validate_selection` 和 `select_with_policy`。排名与解释均调用 `selection.score_terms`。预算与配额失败分别使用 `BudgetError` 和 `QuotaError`，两者仍继承 `ValidationError`。输入错误继续抛出 `ValidationError`。旧调用方无需修改异常处理。

调度应用使用 `dispatch.apply_dispatch`。该纯函数返回新策略、实际语义差异和元数据差异。持久化由 `selection_io` 和命令适配层负责。它们不写训练数据库，不执行模型切换。操作说明见 [筛选链路演示运行手册](docs/筛选链路演示运行手册.md)。

`validate_selection` 返回继承 SelectionContext 的 PolicySelectionContext。该上下文在直接传给共用组批器时也必须保留标签和历史身份。复核范围及回归结果见 [标签策略与调度代码复核记录](docs/标签策略与调度代码复核记录.md)。
