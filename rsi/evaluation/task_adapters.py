"""任务套件封装及已支持本地适配器。隐藏测试不交给生成器。"""

from dataclasses import dataclass
from typing import Any

from rsi.evaluation.coding_tasks import CodingTask, TestCase
from rsi.datasets.labels import SkillOntology, fields
from rsi.evaluation.python_tasks import PythonTask
from rsi.evaluation.agent_tasks import AgentTask
from rsi.datasets.content_identity import CONTENT_RULE, problem_fingerprint, task_fingerprint
from rsi.configuration.validation import array, choice, digest, fail, integer, obj, string, strings, to_json_value


@dataclass(frozen=True)
class GenerationConfig:
    strategy: str
    attempts: int
    max_new_tokens: int
    max_context_length: int
    seed: int

    @classmethod
    def parse(cls, raw: Any) -> "GenerationConfig":
        v = fields(raw, "generation", "strategy attempts max_new_tokens max_context_length seed")
        choice(v["strategy"], "generation.strategy", {"greedy"})
        if integer(v["attempts"], "generation.attempts", 1) != 1:
            fail("generation.attempts", "仅支持一次贪心完成，不自动重试。")
        return cls(v["strategy"], 1, integer(v["max_new_tokens"], "generation.max_new_tokens", 1),
                   integer(v["max_context_length"], "generation.max_context_length", 2), integer(v["seed"], "generation.seed"))


@dataclass(frozen=True)
class ExecutionTask:
    task_id: str
    source_problem_id: str
    split_group: str
    duplicate_group: str
    group: str
    skills: tuple[str, ...]
    label_source: str
    content_fingerprint: str
    adapter_id: str
    adapter_version: str
    payload: CodingTask | PythonTask | AgentTask


@dataclass(frozen=True)
class TaskSuiteV2:
    schema_version: str
    suite_id: str
    revision: str
    purpose: str
    domain: str
    ontology_version: str
    provenance: str
    environment_id: str
    generation: GenerationConfig
    tasks: tuple[ExecutionTask, ...]
    content_identity_rule: str

    def to_dict(self) -> dict[str, Any]:
        value = to_json_value(self)
        for row, task in zip(value["tasks"], self.tasks):
            if task.adapter_id == "agent_environment_v1":
                row["payload"] = task.payload.to_dict()
                continue
            keys = ("prompt", "tests") if task.adapter_id == "bounded_integer_expression_v1" else ("prompt", "entry_point", "setup", "tests")
            row["payload"] = {key: row["payload"][key] for key in keys}
        return value

    @classmethod
    def parse(cls, raw: Any, ontology: SkillOntology) -> "TaskSuiteV2":
        v = obj(raw, "suite", set(cls.__dataclass_fields__))
        choice(v["schema_version"], "suite.schema_version", {"execution-suite-v2"})
        choice(v["content_identity_rule"], "suite.content_identity_rule", {CONTENT_RULE})
        purpose = choice(v["purpose"], "suite.purpose", {"feedback", "final_test"})
        if v["domain"] != ontology.domain or v["ontology_version"] != ontology.ontology_version:
            fail("suite.ontology_version", "领域或技能表版本不一致。")
        tasks = []
        for item in array(v["tasks"], "suite.tasks", True):
            item = obj(item, "suite.task", set(ExecutionTask.__dataclass_fields__))
            ids = {key: string(item[key], "suite.task." + key) for key in
                   ("task_id", "source_problem_id", "split_group", "duplicate_group", "content_fingerprint")}
            group = choice(item["group"], "suite.task.group", {"target", "retention"} if purpose == "feedback" else {"final_test"})
            skills = strings(item["skills"], "suite.task.skills")
            if set(skills) - set(ontology.ids):
                fail("suite.task.skills", "包含未定义技能。")
            adapter = choice(item["adapter_id"], "suite.task.adapter_id", {"bounded_integer_expression_v1", "bubblewrap_seccomp_v1", "agent_environment_v1"})
            choice(item["adapter_version"], "suite.task.adapter_version", {"1"})
            if adapter == "bounded_integer_expression_v1":
                payload = fields(item["payload"], "suite.task.payload", "prompt tests")
                tests = tuple(TestCase.parse(test) for test in array(payload["tests"], "suite.task.payload.tests", True))
                prompt = string(payload["prompt"], "suite.task.payload.prompt")
                if len(prompt) > 4096 or len(tests) > 100:
                    fail("suite.task.payload", "提示或测试数量超限。")
                task = CodingTask(ids["task_id"], group, ids["split_group"], ids["duplicate_group"], prompt, tests)
            elif adapter == "bubblewrap_seccomp_v1":
                payload = fields(item["payload"], "suite.task.payload", "prompt entry_point setup tests")
                task = PythonTask.parse({**payload, **{k: ids[k] for k in ("task_id", "split_group", "duplicate_group")}, "group": group},
                                        allowed_groups=frozenset({"target", "retention", "final_test"}))
            else:
                task = AgentTask.parse(item["payload"])
            if ids["content_fingerprint"] != task_fingerprint(adapter, item["payload"]):
                fail("suite.task.content_fingerprint", "指纹必须由实际任务、环境与判定内容计算，不能使用任意编号。")
            tasks.append(ExecutionTask(**ids, group=group, skills=skills,
                 label_source=choice(item["label_source"], "suite.task.label_source", {"human", "dataset", "code", "synthetic_fixture"}),
                 adapter_id=adapter, adapter_version="1", payload=task))
        for key in ("task_id", "source_problem_id", "split_group", "duplicate_group", "content_fingerprint"):
            if len({getattr(t, key) for t in tasks}) != len(tasks):
                fail("suite.tasks." + key, "任务必须独立，不得重复。")
        owners: dict[str, int] = {}
        group_owners: dict[str, int] = {}
        for index, task in enumerate(tasks):
            for lookup, values in ((owners, (task.task_id, task.source_problem_id)),
                                   (group_owners, (task.split_group, task.duplicate_group))):
                for identity in values:
                    if lookup.setdefault(identity, index) != index:
                        fail("suite.tasks.isolation", "不同任务的身份或隔离分组存在交叉别名。")
        if purpose == "feedback" and {t.group for t in tasks} != {"target", "retention"}:
            fail("suite.tasks.group", "反馈必须同时包含 target 和 retention。")
        if len({problem_fingerprint(t.payload.prompt) for t in tasks}) != len(tasks):
            fail("suite.tasks.content", "套件包含重复的公开问题，不能把同题不同测试当独立任务。")
        return cls(v["schema_version"], string(v["suite_id"], "suite.suite_id"), string(v["revision"], "suite.revision"),
                   purpose, ontology.domain, ontology.ontology_version, string(v["provenance"], "suite.provenance"),
                   string(v["environment_id"], "suite.environment_id"), GenerationConfig.parse(v["generation"]), tuple(tasks), CONTENT_RULE)


def verify_suite_isolation(left: TaskSuiteV2, right: TaskSuiteV2) -> None:
    """按已声明的任务/问题/划分/重复组/指纹检查跨套件重叠。"""
    left_ids = {x for t in left.tasks for x in (t.task_id, t.source_problem_id)}
    right_ids = {x for t in right.tasks for x in (t.task_id, t.source_problem_id)}
    left_groups = {x for t in left.tasks for x in (t.split_group, t.duplicate_group)}
    right_groups = {x for t in right.tasks for x in (t.split_group, t.duplicate_group)}
    left_payloads = left.to_dict()["tasks"]
    right_payloads = right.to_dict()["tasks"]
    for row in (*left_payloads, *right_payloads):
        if row["content_fingerprint"] != task_fingerprint(row["adapter_id"], row["payload"]):
            fail("suite.content_fingerprint", "任务数据类的指纹与实际内容不一致。")
    if (left_ids & right_ids or left_groups & right_groups or
        {problem_fingerprint(t.payload.prompt) for t in left.tasks} & {problem_fingerprint(t.payload.prompt) for t in right.tasks} or
        {t.content_fingerprint for t in left.tasks} & {t.content_fingerprint for t in right.tasks}):
        fail("suite.isolation", "最终套件与反馈套件存在已知重叠。")
