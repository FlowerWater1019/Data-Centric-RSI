"""Agent 环境任务协议。外部策略、环境和判定器必须显式接入。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from rsi.configuration.validation import boolean, canonical, digest, fail, integer, loads, number, obj, read_json, string, to_json_value
from rsi.datasets.labels import fields
from rsi.storage.artifacts import write_json
from rsi.configuration.contracts import ModelRef


@dataclass(frozen=True)
class AgentTask:
    prompt: str
    initial_state_json: str
    verifier_id: str
    verifier_config_json: str
    max_steps: int

    @classmethod
    def parse(cls, raw: Any) -> "AgentTask":
        v = fields(raw, "agent_task", "prompt initial_state verifier max_steps")
        verifier = fields(v["verifier"], "agent_task.verifier", "id config")
        if not isinstance(v["initial_state"], dict) or not isinstance(verifier["config"], dict):
            fail("agent_task", "初始状态和判定器配置必须是 JSON 对象。")
        return cls(string(v["prompt"], "agent_task.prompt"), canonical(v["initial_state"]),
                   string(verifier["id"], "agent_task.verifier.id"), canonical(verifier["config"]),
                   integer(v["max_steps"], "agent_task.max_steps", 1))

    def to_dict(self) -> dict[str, Any]:
        return {"prompt": self.prompt, "initial_state": loads(self.initial_state_json),
                "verifier": {"id": self.verifier_id, "config": loads(self.verifier_config_json)}, "max_steps": self.max_steps}


@dataclass(frozen=True)
class AgentAction:
    name: str
    arguments_json: str


@dataclass(frozen=True)
class EnvironmentStep:
    observation_json: str
    terminated: bool


@dataclass(frozen=True)
class AgentVerdict:
    passed: bool
    score: float | None
    reason: str


@dataclass(frozen=True)
class AgentTransition:
    step: int
    action: AgentAction
    observation_json: str
    terminated: bool


@dataclass(frozen=True)
class AgentOutcome:
    generated: str
    verdict: AgentVerdict
    steps: int
    stop_reason: str
    trace: tuple[AgentTransition, ...]


class AgentPolicy(Protocol):
    def act(self, prompt: str, observation_json: str, history: tuple[AgentTransition, ...], seed: int, step: int) -> AgentAction: ...


class AgentEnvironment(Protocol):
    def reset(self, initial_state_json: str) -> str: ...
    def step(self, action: AgentAction) -> EnvironmentStep: ...
    def verify(self, verifier_id: str, config_json: str) -> AgentVerdict: ...
    def close(self) -> None: ...


class AgentTaskRunner:
    """逐任务创建环境。策略只看公开任务、观察和轨迹，判定配置只交给环境。"""

    def __init__(self, policy: AgentPolicy, environment_factory: Callable[[], AgentEnvironment], identity: str, model: ModelRef) -> None:
        self.policy, self.environment_factory = policy, environment_factory
        self.model = ModelRef.parse(to_json_value(model))
        self.identity = string(identity, "agent_runner.identity")

    def run(self, task: AgentTask, seed: int, output: Path) -> AgentOutcome:
        AgentTask.parse(task.to_dict())
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "request.json", {"task": task.to_dict(), "seed": seed, "model": to_json_value(self.model), "runner_identity": self.identity})
        environment = self.environment_factory()
        trace = []
        try:
            write_json(output / "reset.intent.json", {"initial_state": loads(task.initial_state_json)})
            observation = environment.reset(task.initial_state_json)
            if not isinstance(loads(observation), dict):
                fail("agent.observation", "环境观察必须是 JSON 对象。")
            write_json(output / "initial_observation.json", loads(observation))
            stop_reason = "STEP_LIMIT"
            for step in range(1, task.max_steps + 1):
                write_json(output / f"{step:06d}.intent.json", {"step": step, "observation": loads(observation)})
                action = self.policy.act(task.prompt, observation, tuple(trace), seed, step)
                string(action.name, "agent.action.name")
                arguments = loads(action.arguments_json)
                if not isinstance(arguments, dict):
                    fail("agent.action.arguments", "动作参数必须是 JSON 对象。")
                action = AgentAction(action.name, canonical(arguments))
                write_json(output / f"{step:06d}.action.json", {"name": action.name, "arguments": arguments})
                if action.name == "finish":
                    next_step = EnvironmentStep(observation, True)
                    stop_reason = "AGENT_FINISHED"
                else:
                    next_step = environment.step(action)
                    boolean(next_step.terminated, "agent.step.terminated")
                    if not isinstance(loads(next_step.observation_json), dict):
                        fail("agent.observation", "环境观察必须是 JSON 对象。")
                    if next_step.terminated:
                        stop_reason = "ENVIRONMENT_TERMINATED"
                observation = next_step.observation_json
                item = AgentTransition(step, action, observation, next_step.terminated)
                write_json(output / f"{step:06d}.result.json", to_json_value(item))
                trace.append(item)
                if next_step.terminated:
                    break
            verdict = environment.verify(task.verifier_id, task.verifier_config_json)
            boolean(verdict.passed, "agent.verdict.passed")
            string(verdict.reason, "agent.verdict.reason")
            if verdict.score is not None:
                number(verdict.score, "agent.verdict.score", 0, 1)
            outcome = AgentOutcome(canonical(to_json_value(trace[-1].action)), verdict, len(trace), stop_reason, tuple(trace))
            write_json(output / "outcome.json", to_json_value(outcome))
            return outcome
        finally:
            environment.close()


def verify_agent_evidence(root: Path, task: AgentTask, seed: int, model: ModelRef, runner_identity: str,
                          generated: str, checks: list[dict[str, Any]]) -> None:
    """核验已经完成的 Agent 轨迹，不重置或重放真实环境。"""
    if read_json(root / "request.json") != {"task": task.to_dict(), "seed": seed, "model": to_json_value(model), "runner_identity": runner_identity}:
        fail("agent.cache.request", "Agent 请求不属于当前任务、模型或后端。")
    if read_json(root / "reset.intent.json") != {"initial_state": loads(task.initial_state_json)}:
        fail("agent.cache.initial_state", "初始环境状态不匹配。")
    observation = read_json(root / "initial_observation.json")
    if not isinstance(observation, dict):
        fail("agent.cache.observation", "初始观察必须为对象。")
    outcome = fields(read_json(root / "outcome.json"), "agent.outcome", "generated verdict steps stop_reason trace")
    steps = integer(outcome["steps"], "agent.outcome.steps", 1)
    if not isinstance(outcome["trace"], list) or steps != len(outcome["trace"]) or steps > task.max_steps or outcome["generated"] != generated:
        fail("agent.cache.trace", "步数、最大步数或最终输出不一致。")
    last = None
    for index, item in enumerate(outcome["trace"], 1):
        item = fields(item, "agent.transition", "step action observation_json terminated")
        action = fields(item["action"], "agent.action", "name arguments_json")
        if item["step"] != index or last is not None and last["terminated"]:
            fail("agent.cache.trace", "步骤不连续，或终止后仍有动作。")
        boolean(item["terminated"], "agent.transition.terminated")
        if read_json(root / f"{index:06d}.intent.json") != {"step": index, "observation": observation}:
            fail("agent.cache.intent", "动作之前的观察与轨迹不一致。")
        if read_json(root / f"{index:06d}.action.json") != {"name": action["name"], "arguments": loads(action["arguments_json"])}:
            fail("agent.cache.action", "动作文件与轨迹不一致。")
        if read_json(root / f"{index:06d}.result.json") != item:
            fail("agent.cache.result", "环境步骤事实与轨迹不一致。")
        observation = loads(item["observation_json"])
        if not isinstance(observation, dict):
            fail("agent.cache.observation", "环境观察必须为对象。")
        last = item
    stop = outcome["stop_reason"]
    if stop not in {"STEP_LIMIT", "AGENT_FINISHED", "ENVIRONMENT_TERMINATED"} or (
        stop == "STEP_LIMIT" and (steps != task.max_steps or last["terminated"])) or (
        stop != "STEP_LIMIT" and not last["terminated"]) or (
        stop == "AGENT_FINISHED" and last["action"]["name"] != "finish"):
        fail("agent.cache.stop_reason", "结束原因与轨迹终止状态不一致。")
    if generated != canonical(last["action"]):
        fail("agent.cache.generated", "最终输出与最后动作不一致。")
    verdict = fields(outcome["verdict"], "agent.verdict", "passed score reason")
    boolean(verdict["passed"], "agent.verdict.passed")
    string(verdict["reason"], "agent.verdict.reason")
    if verdict["score"] is not None:
        number(verdict["score"], "agent.verdict.score", 0, 1)
    expected = [{"status": "passed" if verdict["passed"] else "failed", "score": verdict["score"],
                 "reason": verdict["reason"], "steps": steps, "stop_reason": stop}]
    if checks != expected:
        fail("agent.cache.verdict", "环境判定与任务报告不一致。")
