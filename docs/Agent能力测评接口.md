# Agent 能力测评接口

训练输入是带参考轨迹的数据。能力测评输入是任务描述、初始环境状态和评分标准。Agent 自由选择动作并执行。程序不提供参考动作序列，也不比较生成轨迹是否与训练轨迹一致。

拟合验证是另一条独立链路。它只使用本次冻结训练批次，在父候选模型上计算全部有效 SFT loss 位置的 NLL 和完整词表 entropy。关键技能片段从同一次前向的逐 token 结果中映射，不额外前向，不改变训练掩码。用户、工具和模板内容仍不进入该统计。

## 任务输入

使用 `execution-suite-v2` 和 `content_identity_rule=task-content-v1`。任务声明 ID、来源问题 ID、隔离分组、技能标签、适配器版本与实际内容指纹。Agent 任务使用 `adapter_id=agent_environment_v1`。payload 字段如下：

| 字段 | 含义 | Agent 是否可见 |
| --- | --- | --- |
| `prompt` | 任务描述与目标 | 是 |
| `initial_state` | 交给环境 reset 的初始状态对象 | Agent 接收环境返回的观察；不直接暴露全部内部状态 |
| `verifier.id` | 环境判定器的明确版本 | 否 |
| `verifier.config` | 成功条件与评分配置 | 否 |
| `max_steps` | 该任务的动作次数上限 | 属于执行协议 |

技能标签用于结果分组。它们不进入 Agent 的任务提示，也不被当作失败原因。没有技能标签的任务仍进入总体分母，但无法给出相应技能结论。框架不会猜测任务技能。

示例见 `examples/training_validation/agent_feedback_tasks.json` 和 `agent_final_tasks.json`。这些是合成计数器任务，需要相应环境实现，不是 SWE-bench 等官方套件。

## 执行接口

`rsi.evaluation.agent_tasks` 定义以下协议：

- `AgentPolicy.act`：接收公开任务、当前观察、历史动作与观察、任务 seed 和当前步号，返回动作。
- `AgentEnvironment.reset`：根据初始状态创建一个任务环境，返回初始观察。
- `AgentEnvironment.step`：执行动作，返回观察和环境终止标记。
- `AgentEnvironment.verify`：根据判定器 ID 和配置检查最终状态，返回是否完成、可选的 0～1 分数和原因。
- `AgentEnvironment.close`：释放该任务的进程、容器、文件或其他资源。

`AgentTaskRunner` 每题创建独立环境。策略只能看到公开任务和观察，拿不到私有判定配置。策略返回 `finish`、环境返回 terminated，或达到 max_steps 时结束执行。结束后始终调用环境判定器。达到动作上限不自动等于成功；如果目标仍未完成，该题为 TASK_FAILED。

环境或策略异常为 INFRA_ERROR。程序保留调用意图和已有动作，停止后续任务，并释放环境。程序不会重放未知动作。环境后端必须自行保证其沙箱、权限、初始状态和判定器可信；此协议不自动建立容器或宿主隔离。

Agent 模型通过 runner.model 绑定 ModelRef。父候选 runner 使用各自模型引用，但应使用相同的 Agent 协议、环境和评分后端版本。runner.identity 必须明确绑定外部策略协议、环境与判定器的配置和实现版本；不把父候选权重差异当成环境差异。

## 与 benchmark 共用核心

两者都调用 `rsi.evaluation.execution_evaluation.evaluate_execution`。对 Agent 任务，传入 agent_runner；generate 可以为 None。代码生成适配器与 Agent 环境适配器是独立分支。Agent 任务不会降级为一次代码生成。

业务接口示意如下。policy、environment_factory、model 和 identity 由接入方提供，下面的名称不是可直接运行的后端。

```python
from rsi.evaluation.agent_tasks import AgentTaskRunner
from rsi.evaluation.execution_evaluation import evaluate_execution

runner = AgentTaskRunner(policy, environment_factory, identity, model)
report = evaluate_execution(
    model, suite, None, output, call_budget,
    source="measured", agent_runner=runner,
)
```

单次训练后的能力比较使用 `run_dual_validation(..., parent_runner=..., candidate_runner=...)`。独立最终测评使用 `evaluate_benchmark(..., agent_runner=...)`。这两个入口共用 Agent 执行、逐题判分、结果类型和内容隔离。训练反馈额外进行父候选配对及 target/retention 汇总。最终报告不进入反馈或验收。

当前 CLI 没有配置具体 Agent SDK、工具路由、仓库环境或容器后端。直接给 CLI 传 Agent 套件而未接入 runner，会明确报错。真实 Agent 后端仍需接入。假环境单元测试通过不代表真实 Agent 能力测评完成。

## 统计与证据

报告使用 `execution-evaluation-v2`。逐题保存任务公共输入、seed、调用意图、Agent 初始观察、动作、环境观察、终止原因和判定结果。

总体、by_group、by_skill 和 by_group_and_skill 都保存 planned、attempted、completed、passed、infrastructure_error、not_run 和 pass_rate。每题只计一次。多动作、多断言和多技能不会扩大独立任务数量。

pass_rate 是完成目标的任务数除以预定任务数。它不是动作正确率。score_sum 和 mean_score 汇总判定器给出的规范化分数。判定器未提供分数时，完成任务取 1，失败任务取 0。评分标准决定“准确率”的具体含义，框架不会把不同任务的任意指标混为一个 accuracy。

不完整执行时保留已知分数和全部任务分母，但 pass_rate、mean_score 为 null，禁止正式机械验收。能力比较是否提升应从实际父候选结果判断，不预设候选更好。

evaluation_calls 记录任务尝试次数；Agent 内部的动作数由轨迹记录，并受 max_steps 限制。这个计数不是外部 LLM API 调用次数或 token 费用。具体 Agent 后端若需额外费用预算，应在后端接入时显式记录和限制。

## 内容隔离与缓存

内容指纹由实际 prompt、测试、环境初始化和判定器内容计算。Python 代码按 AST 规范化，测试顺序与注释不改变身份。问题公开文本另用统一 Unicode 与空白规则计算身份，并与训练来源中的 user 消息比较。

最终套件不得与反馈套件重复相同的规范化公开问题。更换任务 ID、声明另一个指纹、调整测试或只修改排版，都不能把已知同题变成独立最终题。该检查不证明没有语义改写或未知来源污染。

执行缓存必须逐题匹配实际冻结套件的数量、顺序、ID、分组、技能、seed 和公共输入，并核对同目录调用、生成、判分及 Agent 轨迹证据。受限表达式判分可纯函数重算；真实环境不能为校验缓存而重新执行动作。

旧 `execution-suite-v1` 的任意指纹没有内容绑定保证，现明确拒绝。迁移时在新的输入文件中设置 v2、内容规则，并调用 `rsi.datasets.content_identity.task_fingerprint` 计算每题指纹，再重新验证隔离。不要改写已有 runs。旧执行报告和缓存也不能直接当作 v2 结果复用。
