"""可复算的单次反馈。不调用控制器，不执行模型晋级。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .acceptance import AcceptanceDecision, AcceptanceRule, TaskOutcome, decide
from .artifacts import file_sha256
from .backends import ArtifactRef
from .contracts import ModelRef
from .execution_evaluation import ExecutionEvaluation, parse_execution_evaluation
from .fit_evaluation import FitComparison, FitEvaluation, compare_fit, parse_fit_evaluation
from .labels import fields
from .validation import array, choice, digest, fail, integer, read_json, string, to_json_value


@dataclass(frozen=True)
class EvidenceRef:
    kind: str
    artifact: ArtifactRef
    model_hash: str
    input_hash: str


@dataclass(frozen=True)
class CostSummary:
    training_tokens: int
    scoring_tokens: int
    evaluation_calls: int
    unknown_consumption_possible: bool


@dataclass(frozen=True)
class MechanicalAssessment:
    status: str
    reason: str
    rule: AcceptanceRule | None
    decision: AcceptanceDecision | None


@dataclass(frozen=True)
class PairedExecution:
    before: ExecutionEvaluation
    after: ExecutionEvaluation
    paired_changes: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class RoundFeedback:
    schema_version: str
    feedback_id: str
    round_id: str
    attempt_id: str
    batch_id: str
    source: str
    data_role: str
    execution_mode: str
    data_kind: str
    label_mode: str
    parent_model: ModelRef
    candidate_model: ModelRef
    plan_hash: str
    fit: FitComparison
    execution: PairedExecution
    acceptance: MechanicalAssessment
    evidence: tuple[EvidenceRef, ...]
    costs: CostSummary
    status: str
    controller_status: str = "NOT_RUN"
    model_promotion: str = "NOT_APPLIED"

    def to_dict(self) -> dict[str, Any]:
        return to_json_value(self)


def build_round_feedback(*, feedback_id: str, round_id: str, attempt_id: str, batch_id: str, plan_hash: str,
                         fit_parent: FitEvaluation, fit_candidate: FitEvaluation,
                         execution_parent: ExecutionEvaluation, execution_candidate: ExecutionEvaluation,
                         data_kind: str, label_mode: str, costs: CostSummary,
                         evidence: tuple[EvidenceRef, ...], rule: AcceptanceRule | None = None) -> RoundFeedback:
    """只接受 feedback 任务及可核验原始产物。完整配对后才计算机械验收。"""
    for key, value in (("feedback_id", feedback_id), ("round_id", round_id), ("attempt_id", attempt_id), ("batch_id", batch_id), ("plan_hash", plan_hash)):
        string(value, key)
    left, right = execution_parent, execution_candidate
    if left.purpose != "feedback" or right.purpose != "feedback":
        fail("round_feedback.data_role", "final_test 不得进入训练反馈或验收。")
    # 数据类构造不替代边界校验。纯 Python 调用也必须提供自洽的逐条事实。
    parse_fit_evaluation(fit_parent.to_dict())
    parse_fit_evaluation(fit_candidate.to_dict())
    parse_execution_evaluation(left.to_dict())
    parse_execution_evaluation(right.to_dict())
    if (left.suite_hash, left.generation_hash, left.environment_hash, tuple((r.task_id, r.group, r.skills, r.seed) for r in left.rows)) != (
        right.suite_hash, right.generation_hash, right.environment_hash, tuple((r.task_id, r.group, r.skills, r.seed) for r in right.rows)):
        fail("round_feedback.execution", "任务、环境、技能、生成规则或随机性不一致。")
    if fit_parent.model != left.model or fit_candidate.model != right.model:
        fail("round_feedback.model", "两路测量使用不同模型。")
    if len({fit_parent.source, fit_candidate.source, left.source, right.source}) != 1:
        fail("round_feedback.source", "不能混用 mock 与真实测量。")
    source = choice(fit_parent.source, "round_feedback.source", {"measured", "synthetic_fixture"})
    fit = compare_fit(fit_parent, fit_candidate)
    values = {"fit_parent": (fit_parent, fit_parent.ordered_batch_hash), "fit_candidate": (fit_candidate, fit_candidate.ordered_batch_hash),
              "execution_parent": (left, left.suite_hash), "execution_candidate": (right, right.suite_hash)}
    if len(evidence) != 4 or {item.kind for item in evidence} != values.keys():
        fail("round_feedback.evidence", "必须分别提供四项测量证据。")
    for item in evidence:
        value, input_hash = values[item.kind]
        if file_sha256(item.artifact.path) != item.artifact.sha256 or read_json(item.artifact.path) != to_json_value(value):
            fail("round_feedback.evidence." + item.kind, "产物内容或哈希与测量不一致。")
        if item.model_hash != digest(value.model) or item.input_hash != input_hash:
            fail("round_feedback.evidence." + item.kind, "适用模型或输入身份不一致。")
    complete = left.status == right.status == "COMPLETED" and fit_parent.overall.status == fit_candidate.overall.status == "COMPLETED"
    decision = None
    if complete and rule is not None:
        decision = decide(tuple(TaskOutcome(r.task_id, r.group, r.status == "PASSED") for r in left.rows),
                          tuple(TaskOutcome(r.task_id, r.group, r.status == "PASSED") for r in right.rows), rule)
    assessment = MechanicalAssessment(decision.status if decision else "NOT_RUN", decision.reason if decision else
                  "未提供规则。" if rule is None else "拟合或任务执行不完整，禁止验收。", rule, decision)
    changes = tuple((l.task_id, int(r.status == "PASSED") - int(l.status == "PASSED")) for l, r in zip(left.rows, right.rows)
                    if l.status in {"PASSED", "TASK_FAILED"} and r.status in {"PASSED", "TASK_FAILED"})
    return RoundFeedback("round-feedback-v1", feedback_id, round_id, attempt_id, batch_id, source, "feedback",
                "hf_local" if source == "measured" else "mock", choice(data_kind, "data_kind", {"synthetic", "real_public", "private"}),
                choice(label_mode, "label_mode", {"synthetic_fixture", "annotated", "mixed"}), left.model, right.model,
                plan_hash, fit, PairedExecution(left, right, changes), assessment, evidence, costs,
                "COMPLETED" if complete else "PARTIAL")


def parse_round_feedback(raw: Any) -> RoundFeedback:
    """严格解析并从引用证据重算；不接受字符串占位对象或伪造摘要。"""
    from .validation import boolean
    v = fields(raw, "round_feedback", " ".join(RoundFeedback.__dataclass_fields__))
    choice(v["schema_version"], "round_feedback.schema_version", {"round-feedback-v1"})
    choice(v["data_role"], "round_feedback.data_role", {"feedback"})
    choice(v["controller_status"], "round_feedback.controller_status", {"NOT_RUN"})
    choice(v["model_promotion"], "round_feedback.model_promotion", {"NOT_APPLIED"})
    refs, values = [], {}
    for item in array(v["evidence"], "round_feedback.evidence", True):
        item = fields(item, "evidence", "kind artifact model_hash input_hash")
        kind = choice(item["kind"], "evidence.kind", {"fit_parent", "fit_candidate", "execution_parent", "execution_candidate"})
        a = fields(item["artifact"], "evidence.artifact", "path sha256")
        artifact = ArtifactRef(string(a["path"], "artifact.path"), string(a["sha256"], "artifact.sha256"))
        if file_sha256(artifact.path) != artifact.sha256:
            fail("round_feedback.evidence", "证据哈希已变化。")
        values[kind] = (parse_fit_evaluation if kind.startswith("fit_") else parse_execution_evaluation)(read_json(artifact.path))
        refs.append(EvidenceRef(kind, artifact, string(item["model_hash"], "evidence.model_hash"), string(item["input_hash"], "evidence.input_hash")))
    if len(values) != 4:
        fail("round_feedback.evidence", "证据缺失或重复。")
    cost = fields(v["costs"], "round_feedback.costs", "training_tokens scoring_tokens evaluation_calls unknown_consumption_possible")
    costs = CostSummary(*(integer(cost[k], "costs." + k) for k in ("training_tokens", "scoring_tokens", "evaluation_calls")),
                        boolean(cost["unknown_consumption_possible"], "costs.unknown_consumption_possible"))
    assessment = fields(v["acceptance"], "round_feedback.acceptance", "status reason rule decision")
    rule = None
    if assessment["rule"] is not None:
        r = fields(assessment["rule"], "acceptance.rule", "minimum_gain retention_tolerance minimum_tasks_per_group alpha")
        rule = AcceptanceRule.parse({**r, "schema_version": "acceptance-1", "statistical_rule": "paired_hoeffding_two_groups_v1", "max_retests": 0})
    rebuilt = build_round_feedback(**{k: v[k] for k in ("feedback_id", "round_id", "attempt_id", "batch_id", "plan_hash", "data_kind", "label_mode")},
                **values, costs=costs, evidence=tuple(refs), rule=rule)
    if rebuilt.to_dict() != v:
        fail("round_feedback", "反馈摘要、来源或模型身份与证据不一致。")
    return rebuilt
