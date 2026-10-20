"""独立选择策略及纯合同适配。实验预算是冻结上限。"""

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping

from .contracts import ExperimentContract, FixedStrategy, Preference
from .labels import DOMAINS, FLAGS, MODES, TRAJECTORY_TYPES, SkillOntology, fields, nullable_id
from .validation import boolean, choice, fail, integer, number, obj, string, strings, to_json_value


def shares(raw: Any, path: str, keys: set[str]) -> Mapping[str, float]:
    value = obj(raw, path, set(), keys)
    return MappingProxyType({k: number(v, path + "." + k, 0, 1) for k, v in value.items()})


@dataclass(frozen=True)
class CreatedBy:
    type: str
    controller_checkpoint_id: str | None
    dispatch_id: str | None


@dataclass(frozen=True)
class Eligibility:
    allowed_label_modes: tuple[str, ...]
    required_true_flags: tuple[str, ...]
    unknown_flag_action: str
    allowed_task_types: tuple[str, ...]
    allowed_trajectory_types: tuple[str, ...]
    allowed_harness_protocols: tuple[str, ...]
    allowed_languages: tuple[str, ...]
    excluded_data_ids: tuple[str, ...]
    deferred_batch_ids: tuple[str, ...]


@dataclass(frozen=True)
class PolicyScoring:
    formula_version: str
    score_requirement: str
    reference_version: str
    cdf_version: str
    nll_preference: Preference
    entropy_preference: Preference
    skill_preferences: Mapping[str, float]
    skill_aggregation: str
    weights: Mapping[str, float]


@dataclass(frozen=True)
class BatchConstraints:
    supervised_token_budget: int
    allowed_shortfall: int
    max_context_tokens_per_sample: int
    min_skill_token_shares: Mapping[str, float]
    min_task_type_token_shares: Mapping[str, float]
    max_task_type_token_shares: Mapping[str, float]
    duplicate_rule: str
    allow_within_batch_repetition: bool


@dataclass(frozen=True)
class Execution:
    packing_rule: str
    quota_handling: str
    tie_breaker: str
    seed: int


@dataclass(frozen=True)
class SelectionPolicy:
    schema_version: str
    policy_id: str
    parent_policy_id: str | None
    domain: str
    skill_ontology_version: str
    created_by: CreatedBy
    eligibility: Eligibility
    scoring: PolicyScoring
    batch_constraints: BatchConstraints
    execution: Execution
    description: str

    def to_dict(self) -> dict[str, Any]:
        return to_json_value(self)


def parse_policy(raw: Any, ontology: SkillOntology) -> SelectionPolicy:
    """严格解析策略。任务类型由 allowed_task_types 显式注册。"""
    v = obj(raw, "policy", set(SelectionPolicy.__dataclass_fields__))
    choice(v["schema_version"], "policy.schema_version", {"selection-policy-v1"})
    if v["domain"] != ontology.domain:
        fail("policy.domain", "与技能表领域不一致。")
    if v["skill_ontology_version"] != ontology.ontology_version:
        fail("policy.skill_ontology_version", "与技能表版本不一致。")
    c = fields(v["created_by"], "policy.created_by", "type controller_checkpoint_id dispatch_id")
    creator = CreatedBy(choice(c["type"], "policy.created_by.type", {"human", "fixture", "controller"}),
                        nullable_id(c["controller_checkpoint_id"], "policy.created_by.controller_checkpoint_id"),
                        nullable_id(c["dispatch_id"], "policy.created_by.dispatch_id"))
    if creator.type != "human" and (creator.controller_checkpoint_id is None or creator.dispatch_id is None):
        fail("policy.created_by", "调度策略需要控制检查点和调度 ID。")
    if creator.type == "human" and (creator.controller_checkpoint_id is not None or creator.dispatch_id is not None):
        fail("policy.created_by", "人工根策略的控制检查点与调度 ID 必须为 null。")
    policy_id = string(v["policy_id"], "policy.policy_id")
    parent_id = nullable_id(v["parent_policy_id"], "policy.parent_policy_id")
    if policy_id == parent_id:
        fail("policy.parent_policy_id", "策略不能以自身为父策略。")
    e = obj(v["eligibility"], "policy.eligibility", set(Eligibility.__dataclass_fields__))
    values = {k: strings(x, "policy.eligibility." + k) for k, x in e.items() if k != "unknown_flag_action"}
    for key, options in (("allowed_label_modes", MODES), ("required_true_flags", FLAGS),
                         ("allowed_trajectory_types", TRAJECTORY_TYPES)):
        if set(values[key]) - options:
            fail("policy.eligibility." + key, "包含不支持的值。")
    eligibility = Eligibility(**values, unknown_flag_action=choice(e["unknown_flag_action"],
                              "policy.eligibility.unknown_flag_action", {"exclude", "error"}))
    s = obj(v["scoring"], "policy.scoring", set(PolicyScoring.__dataclass_fields__))
    fixed = FixedStrategy.parse({"strategy_id": v["policy_id"], "nll": s["nll_preference"],
                                "entropy": s["entropy_preference"], "weights": s["weights"],
                                "skill_preferences": s["skill_preferences"]}, ontology.ids)
    scoring = PolicyScoring(choice(s["formula_version"], "policy.scoring.formula_version", {"weighted_utility_v1"}),
                            choice(s["score_requirement"], "policy.scoring.score_requirement", {"current_checkpoint"}),
                            string(s["reference_version"], "policy.scoring.reference_version"),
                            string(s["cdf_version"], "policy.scoring.cdf_version"), fixed.nll, fixed.entropy,
                            fixed.skill_preferences, choice(s["skill_aggregation"], "policy.scoring.skill_aggregation", {"max"}), fixed.weights)
    b = obj(v["batch_constraints"], "policy.batch_constraints", set(BatchConstraints.__dataclass_fields__))
    budget = integer(b["supervised_token_budget"], "policy.batch_constraints.supervised_token_budget", 1)
    shortfall = integer(b["allowed_shortfall"], "policy.batch_constraints.allowed_shortfall")
    if shortfall >= budget:
        fail("policy.batch_constraints.allowed_shortfall", "必须小于预算。")
    minimum = shares(b["min_task_type_token_shares"], "policy.batch_constraints.min_task_type_token_shares", set(eligibility.allowed_task_types))
    maximum = shares(b["max_task_type_token_shares"], "policy.batch_constraints.max_task_type_token_shares", set(eligibility.allowed_task_types))
    if sum(minimum.values()) > 1 + 1e-9 or any(x > maximum.get(k, 1) for k, x in minimum.items()):
        fail("policy.batch_constraints.min_task_type_token_shares", "任务类型最低比例总和超过 1，或最低比例超过最高比例。")
    if boolean(b["allow_within_batch_repetition"], "policy.batch_constraints.allow_within_batch_repetition"):
        fail("policy.batch_constraints.allow_within_batch_repetition", "不支持重复采样。")
    constraints = BatchConstraints(budget, shortfall,
                    integer(b["max_context_tokens_per_sample"], "policy.batch_constraints.max_context_tokens_per_sample", 1),
                    shares(b["min_skill_token_shares"], "policy.batch_constraints.min_skill_token_shares", set(ontology.ids)),
                    minimum, maximum, choice(b["duplicate_rule"], "policy.batch_constraints.duplicate_rule", {"one_per_duplicate_group"}), False)
    ex = obj(v["execution"], "policy.execution", set(Execution.__dataclass_fields__))
    execution = Execution(choice(ex["packing_rule"], "policy.execution.packing_rule", {"exact_ranked_v1", "greedy_v1"}),
                          choice(ex["quota_handling"], "policy.execution.quota_handling", {"postcheck_fail"}),
                          choice(ex["tie_breaker"], "policy.execution.tie_breaker", {"data_id_ascending"}),
                          integer(ex["seed"], "policy.execution.seed"))
    if execution.packing_rule == "exact_ranked_v1" and shortfall:
        fail("policy.batch_constraints.allowed_shortfall", "精确组批要求偏差为 0。")
    return SelectionPolicy(v["schema_version"], policy_id, parent_id, ontology.domain,
                           ontology.ontology_version, creator, eligibility, scoring, constraints, execution,
                           string(v["description"], "policy.description"))


def adapt_policy(contract: ExperimentContract, policy: SelectionPolicy) -> ExperimentContract:
    """返回独立派生合同。保留原模型、训练配置与实验上限。"""
    b, s = policy.batch_constraints, policy.scoring
    if b.supervised_token_budget > min(contract.token_budget, contract.budgets["training_tokens"]):
        fail("policy.batch_constraints.supervised_token_budget", "超过冻结实验预算。")
    if s.reference_version != contract.scoring.reference_version or s.cdf_version != contract.scoring.cdf_version:
        fail("policy.scoring", "评分版本与合同不匹配。")
    return replace(contract, contract_id=contract.contract_id + ":" + policy.policy_id,
                   seed=policy.execution.seed, token_budget=b.supervised_token_budget,
                   allowed_shortfall=b.allowed_shortfall, min_skill_shares=b.min_skill_token_shares,
                   packing_rule=policy.execution.packing_rule,
                   fixed_strategy=FixedStrategy(policy.policy_id, s.nll_preference, s.entropy_preference,
                                                s.weights, s.skill_preferences))
