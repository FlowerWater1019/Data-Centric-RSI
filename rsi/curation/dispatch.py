"""调度声明与显式白名单补丁。自然语言只保存，不执行。"""

from dataclasses import dataclass
from typing import Any

from rsi.curation.feedback import FeedbackFixture
from rsi.curation.history import BatchHistory
from rsi.datasets.labels import SkillOntology, fields
from rsi.curation.policy import SelectionPolicy, parse_policy
from rsi.configuration.validation import (array, canonical, choice, digest, fail, loads, obj,
                         string, strings, to_json_value)


@dataclass(frozen=True)
class Controller:
    type: str
    checkpoint_id: str
    prompt_version: str


@dataclass(frozen=True)
class DispatchInputs:
    parent_checkpoint_id: str
    candidate_checkpoint_id: str
    current_policy_id: str
    batch_id: str
    feedback_id: str
    history_dispatch_ids: tuple[str, ...]


@dataclass(frozen=True)
class Observation:
    evidence_ref: str
    statement: str


@dataclass(frozen=True)
class Hypothesis:
    statement: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class Diagnosis:
    summary: str
    observations: tuple[Observation, ...]
    hypotheses: tuple[Hypothesis, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class ModelAction:
    recommendation: str
    reason: str
    acceptance_evidence_ref: str


@dataclass(frozen=True)
class SetOperation:
    path: str
    value_json: str


@dataclass(frozen=True)
class ListOperation:
    path: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class PolicyUpdate:
    base_policy_id: str
    proposed_policy_id: str
    sets: tuple[SetOperation, ...]
    adds: tuple[ListOperation, ...]
    removes: tuple[ListOperation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"base_policy_id": self.base_policy_id, "proposed_policy_id": self.proposed_policy_id,
                "set": {x.path: loads(x.value_json) for x in self.sets},
                "add": {x.path: list(x.values) for x in self.adds},
                "remove": {x.path: list(x.values) for x in self.removes}}


@dataclass(frozen=True)
class DispatchDecision:
    schema_version: str
    dispatch_id: str
    round_id: str
    controller: Controller
    inputs: DispatchInputs
    diagnosis: Diagnosis
    model_action: ModelAction
    natural_language_dispatch: str
    policy_update: PolicyUpdate
    next_action: str

    def to_dict(self) -> dict[str, Any]:
        result = to_json_value(self)
        result["policy_update"] = self.policy_update.to_dict()
        return result


def parse_set_operation(path: str, value: Any) -> SetOperation:
    try:
        encoded = canonical(value)
    except (ValueError, TypeError, OverflowError) as exc:
        fail("dispatch.policy_update.set." + path, "补丁值必须是有限 JSON 数据：" + str(exc))
    return SetOperation(path, encoded)


def parse_dispatch(raw: Any) -> DispatchDecision:
    """解析调度格式。补丁路径和跨文件身份在应用前校验。"""
    v = obj(raw, "dispatch", set(DispatchDecision.__dataclass_fields__))
    choice(v["schema_version"], "dispatch.schema_version", {"dispatch-decision-v1"})
    c = fields(v["controller"], "dispatch.controller", "type checkpoint_id prompt_version")
    controller = Controller(choice(c["type"], "dispatch.controller.type", {"fixture", "model"}),
                            string(c["checkpoint_id"], "dispatch.controller.checkpoint_id"),
                            string(c["prompt_version"], "dispatch.controller.prompt_version"))
    inp = obj(v["inputs"], "dispatch.inputs", set(DispatchInputs.__dataclass_fields__))
    inputs = DispatchInputs(**{k: strings(x, "dispatch.inputs." + k) if k == "history_dispatch_ids"
                               else string(x, "dispatch.inputs." + k) for k, x in inp.items()})
    d = fields(v["diagnosis"], "dispatch.diagnosis", "summary observations hypotheses limitations")
    observations = []
    for i, item in enumerate(array(d["observations"], "dispatch.diagnosis.observations")):
        path = f"dispatch.diagnosis.observations[{i}]"
        item = fields(item, path, "evidence_ref statement")
        observations.append(Observation(**{k: string(x, path + "." + k) for k, x in item.items()}))
    hypotheses = []
    for i, item in enumerate(array(d["hypotheses"], "dispatch.diagnosis.hypotheses")):
        path = f"dispatch.diagnosis.hypotheses[{i}]"
        item = fields(item, path, "statement evidence_refs")
        hypotheses.append(Hypothesis(string(item["statement"], path + ".statement"), strings(item["evidence_refs"], path + ".evidence_refs")))
    diagnosis = Diagnosis(string(d["summary"], "dispatch.diagnosis.summary"), tuple(observations), tuple(hypotheses),
                          strings(d["limitations"], "dispatch.diagnosis.limitations"))
    m = fields(v["model_action"], "dispatch.model_action", "recommendation reason acceptance_evidence_ref")
    model_action = ModelAction(choice(m["recommendation"], "dispatch.model_action.recommendation", {"accept", "reject", "hold"}),
                               string(m["reason"], "dispatch.model_action.reason"),
                               string(m["acceptance_evidence_ref"], "dispatch.model_action.acceptance_evidence_ref"))
    p = fields(v["policy_update"], "dispatch.policy_update", "base_policy_id proposed_policy_id set add remove")
    operations = {}
    for key in ("set", "add", "remove"):
        if not isinstance(p[key], dict):
            fail("dispatch.policy_update." + key, "必须是操作对象。")
        for path in p[key]:
            string(path, "dispatch.policy_update." + key)
        operations[key] = p[key]
    update = PolicyUpdate(string(p["base_policy_id"], "dispatch.policy_update.base_policy_id"),
                          string(p["proposed_policy_id"], "dispatch.policy_update.proposed_policy_id"),
                          tuple(parse_set_operation(k, x) for k, x in sorted(operations["set"].items())),
                          tuple(ListOperation(k, strings(x, "dispatch.policy_update.add." + k)) for k, x in sorted(operations["add"].items())),
                          tuple(ListOperation(k, strings(x, "dispatch.policy_update.remove." + k)) for k, x in sorted(operations["remove"].items())))
    return DispatchDecision(v["schema_version"], string(v["dispatch_id"], "dispatch.dispatch_id"),
                            string(v["round_id"], "dispatch.round_id"), controller, inputs, diagnosis, model_action,
                            string(v["natural_language_dispatch"], "dispatch.natural_language_dispatch"), update,
                            choice(v["next_action"], "dispatch.next_action", {"select_next_batch", "stop"}))


def validate_dispatch_context(policy: SelectionPolicy, dispatch: DispatchDecision, feedback: FeedbackFixture,
                              history: BatchHistory, history_dispatches: tuple[DispatchDecision, ...] = (),
                              history_feedback: tuple[FeedbackFixture, ...] = ()) -> None:
    """校验父策略、反馈身份和可解析证据。不执行模型操作。"""
    inputs, update = dispatch.inputs, dispatch.policy_update
    if inputs.current_policy_id != policy.policy_id or update.base_policy_id != policy.policy_id:
        fail("dispatch.policy_update.base_policy_id", "与实际父策略不一致。")
    if update.proposed_policy_id == policy.policy_id:
        fail("dispatch.policy_update.proposed_policy_id", "不能等于父策略 ID。")
    for key in ("batch_id", "parent_checkpoint_id", "candidate_checkpoint_id", "feedback_id"):
        if getattr(inputs, key) != getattr(feedback, key):
            fail("dispatch.inputs." + key, "与反馈身份不一致。")
    if dispatch.round_id != feedback.round_id:
        fail("dispatch.round_id", "与反馈轮次不一致。")
    if dispatch.controller.type != "fixture":
        fail("dispatch.controller.type", "本阶段只接收 fixture 控制器。")
    if dispatch.controller.checkpoint_id != inputs.parent_checkpoint_id:
        fail("dispatch.controller.checkpoint_id", "必须等于本轮父检查点。")
    batches = {b.batch_id: b for b in history.batches}
    if inputs.batch_id not in batches or batches[inputs.batch_id].policy_hash != digest(policy):
        fail("dispatch.inputs.batch_id", "批次历史不存在，或父策略哈希不一致。")
    prior = {d.dispatch_id: d for d in history_dispatches}
    if len(prior) != len(history_dispatches) or dispatch.dispatch_id in prior:
        fail("dispatch.inputs.history_dispatch_ids", "历史 ID 重复或引用当前调度。")
    known: set[str] = set()
    for d in history_dispatches:
        if set(d.inputs.history_dispatch_ids) - known:
            fail("dispatch_history", "历史顺序包含未来或未知调度。")
        known.add(d.dispatch_id)
    if set(inputs.history_dispatch_ids) - prior.keys():
        fail("dispatch.inputs.history_dispatch_ids", "存在未来或未知调度。")
    evidence = dict(feedback.evidence)
    feedback_by_id = {f.feedback_id: f for f in history_feedback}
    if len(feedback_by_id) != len(history_feedback):
        fail("feedback_history", "历史反馈 ID 重复。")
    for key in inputs.history_dispatch_ids:
        old = prior[key]
        if old.inputs.feedback_id not in feedback_by_id:
            fail("feedback_history", "历史调度缺少明确反馈。")
        old_feedback = feedback_by_id[old.inputs.feedback_id]
        if old.round_id != old_feedback.round_id or any(getattr(old.inputs, k) != getattr(old_feedback, k)
                for k in ("batch_id", "parent_checkpoint_id", "candidate_checkpoint_id", "feedback_id")):
            fail("feedback_history", "历史反馈与调度身份不一致。")
        for name, item in old_feedback.evidence.items():
            qualified_name = old.dispatch_id + ":" + name
            if qualified_name in evidence:
                fail("dispatch.diagnosis.evidence_refs", "当前或历史证据 ID 冲突：" + qualified_name)
            evidence[qualified_name] = item
    refs = [x.evidence_ref for x in dispatch.diagnosis.observations]
    refs += [ref for h in dispatch.diagnosis.hypotheses for ref in h.evidence_refs]
    refs.append(dispatch.model_action.acceptance_evidence_ref)
    if set(refs) - evidence.keys():
        fail("dispatch.diagnosis.evidence_refs", "证据引用无法解析。")
    # 模型建议只允许由当前反馈中的验收证据支持。
    acceptance_ref = feedback.evidence.get(dispatch.model_action.acceptance_evidence_ref)
    if acceptance_ref is None or acceptance_ref.kind != "acceptance_result":
        fail("dispatch.model_action.acceptance_evidence_ref", "必须引用当前反馈的验收结果。")
    expected = {"accept": "accepted", "reject": "rejected"}.get(dispatch.model_action.recommendation)
    if expected is not None and feedback.acceptance.status != expected:
        fail("dispatch.model_action.recommendation", "与 fixture 验收状态不一致。")


@dataclass(frozen=True)
class PolicyDiff:
    path: str
    before_json: str
    after_json: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "before": loads(self.before_json), "after": loads(self.after_json)}


@dataclass(frozen=True)
class DispatchApplication:
    policy: SelectionPolicy
    diff: tuple[PolicyDiff, ...]
    request_hash: str
    metadata_diff: tuple[PolicyDiff, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "dispatch-application-v1", "request_hash": self.request_hash,
                "policy_hash": digest(self.policy), "policy_id": self.policy.policy_id,
                "diff": [d.to_dict() for d in self.diff], "model_operation": "recommendation_only",
                "metadata_diff": [d.to_dict() for d in self.metadata_diff],
                "training_status": "NOT_RUN", "task_execution_status": "NOT_RUN"}


def apply_dispatch(policy: SelectionPolicy, dispatch: DispatchDecision, feedback: FeedbackFixture,
                   ontology: SkillOntology, history: BatchHistory,
                   history_dispatches: tuple[DispatchDecision, ...] = (),
                   history_feedback: tuple[FeedbackFixture, ...] = ()) -> DispatchApplication:
    """纯补丁应用。失败时保留父对象。完整重校验后返回新策略。"""
    validate_dispatch_context(policy, dispatch, feedback, history, history_dispatches, history_feedback)
    v = policy.to_dict()
    update = dispatch.policy_update
    diffs = []
    for operation in update.sets:
        path, value = operation.path, loads(operation.value_json)
        if path == "scoring.nll_preference":
            target, key = v["scoring"], "nll_preference"
        elif path == "scoring.entropy_preference":
            target, key = v["scoring"], "entropy_preference"
        elif path == "scoring.weights":
            target, key = v["scoring"], "weights"
        elif path in {"scoring.skill_preferences." + k for k in ontology.ids}:
            target, key = v["scoring"]["skill_preferences"], path.removeprefix("scoring.skill_preferences.")
        elif path in {"batch_constraints.min_skill_token_shares." + k for k in ontology.ids}:
            target, key = v["batch_constraints"]["min_skill_token_shares"], path.removeprefix("batch_constraints.min_skill_token_shares.")
        elif path in {"batch_constraints.min_task_type_token_shares." + k for k in policy.eligibility.allowed_task_types}:
            target, key = v["batch_constraints"]["min_task_type_token_shares"], path.removeprefix("batch_constraints.min_task_type_token_shares.")
        elif path in {"batch_constraints.max_task_type_token_shares." + k for k in policy.eligibility.allowed_task_types}:
            target, key = v["batch_constraints"]["max_task_type_token_shares"], path.removeprefix("batch_constraints.max_task_type_token_shares.")
        else:
            fail("dispatch.policy_update.set." + path, "路径不在白名单中。")
        old = target.get(key)
        if old != value:
            diffs.append(PolicyDiff(path, canonical(old), canonical(value)))
        target[key] = value
    allowed_lists = {"eligibility.excluded_data_ids", "eligibility.deferred_batch_ids"}
    adds, removes = {x.path: x.values for x in update.adds}, {x.path: x.values for x in update.removes}
    for path in sorted(adds.keys() | removes.keys()):
        if path not in allowed_lists:
            fail("dispatch.policy_update.add/remove." + path, "路径不在白名单中。")
        if set(adds.get(path, ())) & set(removes.get(path, ())):
            fail("dispatch.policy_update." + path, "同一元素不能同时 add/remove。")
        key = "excluded_data_ids" if path == "eligibility.excluded_data_ids" else "deferred_batch_ids"
        before = v["eligibility"][key]
        after = sorted((set(before) | set(adds.get(path, ()))) - set(removes.get(path, ())))
        if before != after:
            diffs.append(PolicyDiff(path, canonical(before), canonical(after)))
        v["eligibility"][key] = after
    v["policy_id"] = update.proposed_policy_id
    v["parent_policy_id"] = policy.policy_id
    v["created_by"] = {"type": "fixture", "controller_checkpoint_id": dispatch.controller.checkpoint_id,
                        "dispatch_id": dispatch.dispatch_id}
    candidate = parse_policy(v, ontology)
    history.resolve(candidate.eligibility.deferred_batch_ids)
    request_hash = digest({"parent": policy.to_dict(), "dispatch": dispatch.to_dict(), "feedback": feedback,
                           "ontology": ontology, "history": history,
                           "history_dispatches": [d.to_dict() for d in history_dispatches], "history_feedback": history_feedback})
    before, after = policy.to_dict(), candidate.to_dict()
    metadata = tuple(PolicyDiff(key, canonical(before[key]), canonical(after[key]))
                     for key in ("policy_id", "parent_policy_id", "created_by") if before[key] != after[key])
    return DispatchApplication(candidate, tuple(diffs), request_hash, metadata)
