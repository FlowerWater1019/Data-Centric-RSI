"""预设反馈格式。数值只用于接口演示，不代表真实测量。"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from rsi.datasets.labels import SkillOntology, fields
from rsi.configuration.validation import choice, fail, integer, number, obj, string, to_json_value


@dataclass(frozen=True)
class FitMetrics:
    nll: float
    entropy: float


@dataclass(frozen=True)
class FitFixture:
    scope: str
    supervised_tokens: int
    before: FitMetrics
    after: FitMetrics


@dataclass(frozen=True)
class ExecutionCounts:
    task_count: int
    before_passed: int
    after_passed: int

    def rates(self) -> dict[str, float | str | None]:
        return {"status": "AVAILABLE" if self.task_count else "MISSING",
                "before": self.before_passed / self.task_count if self.task_count else None,
                "after": self.after_passed / self.task_count if self.task_count else None}


@dataclass(frozen=True)
class ExecutionFixture:
    target: ExecutionCounts
    retention: ExecutionCounts
    by_skill: Mapping[str, ExecutionCounts]


@dataclass(frozen=True)
class AcceptanceFixture:
    status: str
    source: str
    rule_version: str
    reason: str


@dataclass(frozen=True)
class Evidence:
    kind: str
    skill_id: str | None


@dataclass(frozen=True)
class FeedbackFixture:
    schema_version: str
    feedback_id: str
    round_id: str
    batch_id: str
    parent_checkpoint_id: str
    candidate_checkpoint_id: str
    source: str
    data_role: str
    fit: FitFixture
    execution: ExecutionFixture
    acceptance: AcceptanceFixture
    evidence: Mapping[str, Evidence]

    def to_dict(self) -> dict[str, Any]:
        return to_json_value(self)


def parse_counts(raw: Any, path: str) -> ExecutionCounts:
    v = fields(raw, path, "task_count before_passed after_passed")
    result = ExecutionCounts(**{k: integer(x, path + "." + k) for k, x in v.items()})
    if max(result.before_passed, result.after_passed) > result.task_count:
        fail(path, "通过数超过任务数。")
    return result


def parse_feedback(raw: Any, ontology: SkillOntology) -> FeedbackFixture:
    """接收模拟反馈。真实验收需要另一种明确版本的接口。"""
    v = obj(raw, "feedback", set(FeedbackFixture.__dataclass_fields__))
    choice(v["schema_version"], "feedback.schema_version", {"feedback-fixture-v1"})
    choice(v["source"], "feedback.source", {"synthetic_fixture"})
    choice(v["data_role"], "feedback.data_role", {"feedback"})
    f = fields(v["fit"], "feedback.fit", "scope supervised_tokens before after")
    metrics = []
    for key in ("before", "after"):
        m = fields(f[key], "feedback.fit." + key, "nll entropy")
        metrics.append(FitMetrics(**{k: number(x, f"feedback.fit.{key}.{k}") for k, x in m.items()}))
    fit = FitFixture(choice(f["scope"], "feedback.fit.scope", {"selected_training_batch"}),
                     integer(f["supervised_tokens"], "feedback.fit.supervised_tokens", 1), *metrics)
    ex = fields(v["execution"], "feedback.execution", "target retention by_skill")
    groups = obj(ex["by_skill"], "feedback.execution.by_skill", set(), set(ontology.ids))
    execution = ExecutionFixture(parse_counts(ex["target"], "feedback.execution.target"),
                                 parse_counts(ex["retention"], "feedback.execution.retention"),
                                 MappingProxyType({k: parse_counts(x, "feedback.execution.by_skill." + k) for k, x in groups.items()}))
    ac = fields(v["acceptance"], "feedback.acceptance", "status source rule_version reason")
    acceptance = AcceptanceFixture(choice(ac["status"], "feedback.acceptance.status", {"accepted", "rejected", "inconclusive"}),
                                   choice(ac["source"], "feedback.acceptance.source", {"fixture"}),
                                   choice(ac["rule_version"], "feedback.acceptance.rule_version", {"fixture-v1"}),
                                   string(ac["reason"], "feedback.acceptance.reason"))
    if not isinstance(v["evidence"], dict):
        fail("feedback.evidence", "必须是 ID 注册表。")
    evidence = {}
    for key, raw_item in v["evidence"].items():
        path = "feedback.evidence." + string(key, "feedback.evidence.id")
        item = fields(raw_item, path, "kind skill_id")
        kind = choice(item["kind"], path + ".kind", {"fit_nll", "fit_entropy", "target_execution", "retention_execution", "skill_execution", "acceptance_result"})
        skill = item["skill_id"]
        if kind == "skill_execution":
            choice(skill, path + ".skill_id", set(execution.by_skill))
        elif skill is not None:
            fail(path + ".skill_id", "非技能证据必须为 null。")
        evidence[key] = Evidence(kind, skill)
    identities = {k: string(v[k], "feedback." + k) for k in
                  ("feedback_id", "round_id", "batch_id", "parent_checkpoint_id", "candidate_checkpoint_id")}
    return FeedbackFixture(v["schema_version"], **identities, source=v["source"], data_role=v["data_role"],
                           fit=fit, execution=execution, acceptance=acceptance, evidence=MappingProxyType(evidence))
