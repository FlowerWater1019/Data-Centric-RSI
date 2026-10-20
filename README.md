# RSI 实验内核

当前版本为 0.6.0。框架已接入真实 MBPP 数据的单卡开发对照：导入、编译、评分、精确组批、LoRA 训练、完整步骤恢复、诊断和 Python 隔离评估。

pilot 命令执行真实分词、模型评分和 LoRA 更新。resume-pilot 从完整步骤边界恢复同一次训练。evaluate-pilot 执行独立的受限纯函数任务，并保存工程验收判定。三个命令都使用合成数据检查程序。

import-mbpp 导入固定版本的真实数据，并在隔离环境中检查参考解。compile-data 将来源编译为现有轨迹格式。evaluate-pilot 也可使用 Linux 隔离后端执行完整 Python 标准库代码。

run-comparison 执行真实数据评分和两组等监督 token 训练。resume-comparison 恢复完整训练步骤。report-comparison 核验证据并分别报告普通微调收益和固定选择相对随机选择的差值。

当前主要运行说明为 [真实数据训练对照运行手册](docs/真实数据训练对照运行手册.md)。模块与边界见 [真实数据训练对照接口](docs/真实数据训练对照接口.md)。自动模型晋级、多轮 Agent 决策、按需评分和分布式训练仍未接入。

当前基础命令只检查数据与计划。它们不会自行启动训练。演示数据和工程验证结果不属于研究效果证据。

新增独立的标签、筛选策略与模拟调度接口。该接口使用 sidecar JSONL 关联旧轨迹，复用现有固定评分和完整轨迹组批。旧合同、数据库格式和命令默认行为保持兼容。

离线演示生成 10 条候选和 2 条参考轨迹。演示执行两次筛选，中间应用预设调度补丁。训练和任务执行均为 `NOT_RUN`。

```text
python -m rsi make-selection-fixture --output runs/selection-fixture --seed 42 --settings examples/selection_fixture.settings.json
python -m rsi selection-demo --fixture runs/selection-fixture --output runs/selection-demo
```

输出目录必须不存在。字段定义见 [标签策略与调度接口](docs/标签策略与调度接口.md)。独立命令、产物和实测结果见 [筛选链路演示运行手册](docs/筛选链路演示运行手册.md)。这些模拟结果不提供真实标签有效性、模型能力或 RSI 效果证据。

GitHub 仓库为 [FlowerWater1019/Data-Centric-RSI](https://github.com/FlowerWater1019/Data-Centric-RSI)。克隆后的仓库目录就是工程根目录。本机现有工作目录为 `D:\RSI\framework`。

训练端的 Codex 先读取 [CODEX_HANDOFF.md](docs/CODEX_HANDOFF.md)。该文件给出接手顺序、模型适配边界、数据获取方式和回传要求。

仓库包含源码、配置、测试、小规模合成示例和工程文档。真实数据、模型权重、状态库、运行日志和历史证据保留在运行机器，不上传 GitHub。历史工程文档中的设备和路径占位值需要替换为目标机器的实际值。

运行步骤见 [第一阶段运行手册](docs/第一阶段运行手册.md)。字段定义见 [数据与接口说明](docs/数据与接口说明.md)。实现边界见 [阶段一交付与后续接入](docs/阶段一交付与后续接入.md)。开发要求见 [工程约定](AGENTS.md) 和 [开发指南](CONTRIBUTING.md)。完整研究方案由项目负责人另行提供，不包含在本仓库。

## 快速检查

先进入克隆后的仓库根目录，再使用 Python 3.11 或更高版本运行以下命令。运行当前命令不需要安装第三方依赖。

```text
python -m unittest discover -s tests -v
python -m rsi validate --config configs/smoke.json --data examples/trajectories.demo.jsonl --require-current-scores
```

第一条命令运行自动测试。第二条命令检查演示输入。两条命令都不调用 GPU。

## 目录

| 路径 | 内容 |
| --- | --- |
| rsi/contracts.py | 实验配置、模型引用与固定策略 |
| rsi/settings.py | 训练、硬件和验收的明确字段类型 |
| rsi/data.py | 轨迹、掩码、诊断位置和数据隔离校验 |
| rsi/inspection.py | 共用的分数检查与校验报告 |
| rsi/selection.py | 统一选择接口、随机排序、固定排序与约束组批 |
| rsi/comparison.py | 对照计划的纯计算 |
| rsi/schema.py | 数据库版本检查与新库建表 |
| rsi/state.py | 输入快照、计划、事件、预算预留与实际消耗账本 |
| rsi/backends.py | 训练与评估的字段接口 |
| rsi/tokenization.py | 消息内容位置、真实分词与监督映射 |
| rsi/scoring.py | 评分类型、参考分布与百分位 |
| rsi/hf_backend.py | 本地模型加载、完整词表评分与单卡 LoRA |
| rsi/pilot.py | 工程验证流程与实际训练记录 |
| rsi/real_run.py、rsi/training_run.py | 真实对照准备与共用训练执行 |
| rsi/packing.py | 完整轨迹的精确监督预算组批 |
| rsi/scoring_run.py、rsi/diagnostics.py | 评分调用记录与固定集合诊断 |
| rsi/python_diagnostics.py | 可选的 Python 关键字片段标注 |
| rsi/reporting.py | 账本与逐任务证据复核、开发报告 |
| rsi/artifacts.py | 模型与训练产物的内容校验 |
| rsi/checkpoints.py、rsi/leases.py | 完整步骤检查点与运行目录单写入者锁 |
| rsi/coding_tasks.py、rsi/bounded_python.py | 独立任务格式与受限整数解释器 |
| rsi/coding_evaluation.py、rsi/evaluation_run.py | 逐任务证据与三组评估流程 |
| rsi/acceptance.py | 任务级配对判定与能力保持检查 |
| rsi/cli.py | 校验、初始化、计划、状态与导出命令 |
| configs/smoke.json | CPU 演示配置 |
| configs/h200_8gpu.draft.json | 8×H200 目标配置草稿 |
| examples/trajectories.demo.jsonl | 7 条合成轨迹及演示分数 |
| examples/pilot_messages.jsonl | 8 条消息级合成编程轨迹 |
| configs/pilot.json | 单卡工程验证参数 |
| configs/comparison.development.json | 真实数据开发对照参数 |
| tests/ | 按模块划分的测试及独立进程命令检查 |

## 目标硬件

用户提供的训练目标是 8 张 H200，每张显存 144 GB。代码记录该声明，但没有探测远程设备。草稿配置使用 8 个目标训练进程，并将 bf16 记为候选精度。具体训练参数需在模型与数据确定后验证。

本地基础命令只依赖 Python 标准库。模型执行使用可选依赖。服务器已使用现有环境执行工程验证，未安装新环境，也未下载新模型。分布式训练尚未执行。

工程检查的历史记录见 [工程检查与调整记录](docs/工程检查与调整记录-2026-10-18.md)。新流程见 [小模型工程验证运行手册](docs/小模型工程验证运行手册.md)。当前接口补充见 [真实后端接口与限制](docs/真实后端接口与限制.md)。

两张 5880 的实际执行结果见 [服务器工程验证记录](docs/服务器工程验证记录-2026-10-18.md)。

0.4.0 的运行步骤见 [恢复与任务评估运行手册](docs/恢复与任务评估运行手册.md)。新增接口见 [恢复与验收接口说明](docs/恢复与验收接口说明.md)。历史验证记录保留原版本结论。

本轮实测结果见 [恢复与任务评估验证记录](docs/恢复与任务评估验证记录-2026-10-18.md)。

0.5.0 的接口和边界见 [真实数据与隔离执行接口](docs/真实数据与隔离执行接口.md)。运行步骤见 [真实数据与隔离执行运行手册](docs/真实数据与隔离执行运行手册.md)。

本轮真实数据与隔离环境的实测结果见 [0.5.0 验证记录](docs/真实数据与隔离执行验证记录-2026-10-18.md)。

0.6.0 已完成真实数据训练、暂停恢复和隔离评估。实测范围见 [0.6.0 验证记录](docs/真实数据训练对照验证记录-2026-10-18.md)。下一阶段需要确定的信息见 [开发训练实验交接单](docs/开发训练实验交接单.md)。
