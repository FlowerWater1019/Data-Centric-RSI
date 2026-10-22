"""标签筛选与可解释报告。复用已有排名和完整轨迹组批。"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from rsi.configuration.contracts import ExperimentContract
from rsi.datasets.data import ScoreSnapshot, TrajectoryRecord
from rsi.curation.history import BatchHistory
from rsi.datasets.labels import DataLabels, SkillOntology, bind_labels
from rsi.curation.policy import SelectionPolicy, adapt_policy
from rsi.curation.selection import (BatchManifest, FixedSelector, ScoreTerms, SelectionContext,
                        build_batch, check_task_type_shares, score_terms)
from rsi.configuration.validation import BudgetError, QuotaError, canonical, digest, fail, loads, to_json_value


@dataclass(frozen=True)
class PolicySelectionContext(SelectionContext):
    """携带完整输入身份，可直接传给旧排名器和组批器。"""
    identities: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "identities", MappingProxyType(dict(self.identities)))

    @property
    def context_id(self) -> str:
        return digest(self.identities)


@dataclass(frozen=True)
class CandidateReport:
    data_id: str
    eligibility_pass: bool
    exclusion_reasons: tuple[str, ...]
    entered_scoring: bool
    raw_scores: ScoreSnapshot | None
    score: ScoreTerms | None
    ranking_position: int | None
    selected_position: int | None
    packing_reason: str | None


@dataclass(frozen=True)
class SelectionResult:
    """报告以规范 JSON 保存，返回副本。批次清单保持既有类型。"""
    batch: BatchManifest | None
    _report_json: str

    def to_dict(self) -> dict[str, Any]:
        return loads(self._report_json)


def validate_selection(contract: ExperimentContract, records: tuple[TrajectoryRecord, ...],
                       labels: tuple[DataLabels, ...], ontology: SkillOntology,
                       policy: SelectionPolicy, history: BatchHistory = BatchHistory(),
                       *, allow_synthetic_labels: bool = False) -> PolicySelectionContext:
    """校验全部输入及合格候选评分。不组批、不训练。"""
    bound = bind_labels(records, labels, ontology, contract, allow_synthetic_labels=allow_synthetic_labels)
    if policy.domain != ontology.domain or policy.skill_ontology_version != ontology.ontology_version:
        fail("policy", "领域或技能表版本不匹配。")
    derived = adapt_policy(contract, policy)
    deferred = history.resolve(policy.eligibility.deferred_batch_ids)
    eligible = tuple(r for r in records if r.role == "candidate" and not exclusion_reasons(r, bound[r.data_id], policy, deferred))
    for record in eligible:
        record.require_current_scores(derived)
    ordered = sorted(records, key=lambda record: record.data_id)
    dataset_hash = digest([r.raw for r in ordered])
    identities = {"contract_hash": digest(contract), "derived_contract_hash": digest(derived),
                  "model_hash": digest(contract.model), "dataset_hash": dataset_hash,
                  "labels_hash": digest(sorted(labels, key=lambda x: x.data_id)), "ontology_hash": digest(ontology),
                  "policy_hash": digest(policy), "history_hash": digest(history),
                  "scoring_hash": digest({"versions": contract.scoring, "snapshots": [r.scores for r in ordered]})}
    return PolicySelectionContext(derived, eligible, identities["derived_contract_hash"], dataset_hash, identities)


def exclusion_reasons(record: TrajectoryRecord, label: DataLabels, policy: SelectionPolicy,
                      deferred: frozenset[str]) -> tuple[str, ...]:
    reasons = []
    e = policy.eligibility
    if not record.is_eligible:
        reasons.append("old_base_ineligible")
    if label.provenance.label_mode not in e.allowed_label_modes:
        reasons.append("label_mode_not_allowed")
    for flag in e.required_true_flags:
        value = getattr(label.quality_flags, flag)
        if value is False:
            reasons.append("quality_false:" + flag)
        elif value is None:
            if e.unknown_flag_action == "error":
                fail("labels." + record.data_id + ".quality_flags." + flag, "策略不允许未知质量。")
            reasons.append("quality_unknown:" + flag)
    for attr, allowed in (("task_type", e.allowed_task_types), ("trajectory_type", e.allowed_trajectory_types),
                          ("harness_protocol", e.allowed_harness_protocols), ("language", e.allowed_languages)):
        if getattr(label.task, attr) not in allowed:
            reasons.append(attr + "_not_allowed")
    if record.data_id in e.excluded_data_ids:
        reasons.append("explicit_exclusion")
    if record.data_id in deferred:
        reasons.append("deferred_batch")
    if record.token_count > policy.batch_constraints.max_context_tokens_per_sample:
        reasons.append("context_length_exceeded")
    return tuple(reasons)


def select_with_policy(contract: ExperimentContract, records: tuple[TrajectoryRecord, ...],
                       labels: tuple[DataLabels, ...], ontology: SkillOntology,
                       policy: SelectionPolicy, history: BatchHistory = BatchHistory(),
                       *, allow_synthetic_labels: bool = False) -> SelectionResult:
    """输入与评分错误抛异常。预算和配额失败返回明确失败报告及空批次。"""
    context = validate_selection(contract, records, labels, ontology, policy, history,
                                 allow_synthetic_labels=allow_synthetic_labels)
    bound = {label.data_id: label for label in labels}
    identities = dict(context.identities)
    context_id = context.context_id
    ranked = FixedSelector().rank(context)
    ranks = {r.data_id: i for i, r in enumerate(ranked, 1)}
    batch = None
    status, reason = "SELECTED", None
    task_shares: dict[str, float] = {}
    try:
        batch = build_batch(context, FixedSelector())
        selected_ids = {item.data_id for item in batch.items}
        for task in policy.eligibility.allowed_task_types:
            task_shares[task] = sum(r.supervised_tokens for r in ranked if r.data_id in selected_ids
                                    and bound[r.data_id].task.task_type == task) / batch.supervised_tokens
        check_task_type_shares(task_shares, policy.batch_constraints.min_task_type_token_shares,
                               policy.batch_constraints.max_task_type_token_shares)
    except (BudgetError, QuotaError) as exc:
        reason = str(exc)
        status = "QUOTA_FAILED" if isinstance(exc, QuotaError) else "BUDGET_FAILED"
        batch = None
    positions = {item.data_id: i for i, item in enumerate(batch.items, 1)} if batch else {}
    candidates = []
    deferred = history.resolve(policy.eligibility.deferred_batch_ids)
    for record in sorted(records, key=lambda r: r.data_id):
        if record.role != "candidate":
            continue
        excluded = exclusion_reasons(record, bound[record.data_id], policy, deferred)
        entered = not excluded
        candidates.append(CandidateReport(record.data_id, entered, excluded, entered,
                          record.scores if entered else None,
                          score_terms(record, context.contract) if entered else None,
                          ranks.get(record.data_id), positions.get(record.data_id),
                          "not_selected_by_packing" if entered and batch and record.data_id not in positions else None))
    report = {"schema_version": "selection-report-v1", "status": status, "failure_reason": reason,
              "run_mode": "fixture" if any(x.provenance.label_mode == "synthetic_fixture" for x in labels) else "annotated",
              "training_status": "NOT_RUN", "task_execution_status": "NOT_RUN", "context_id": context_id,
              "identities": identities, "derived_contract": to_json_value(context.contract),
              "candidates": to_json_value(candidates), "batch": batch.to_dict() if batch else None,
              "task_type_shares": task_shares,
              "counts": {"records": len(records), "candidates": len(candidates),
                         "base_eligible": sum(r.is_eligible for r in records),
                         "label_eligible": sum(not tuple(reason for reason in row.exclusion_reasons
                                                          if reason != "context_length_exceeded") for row in candidates),
                         "length_eligible": len(ranked), "eligible": len(ranked),
                         "scored": len(ranked), "selected": len(positions)},
              "packing_check": "PASSED" if status in {"SELECTED", "QUOTA_FAILED"} else "FAILED",
              "quota_check": "PASSED" if status == "SELECTED" else "FAILED" if status == "QUOTA_FAILED" else "NOT_RUN"}
    return SelectionResult(batch, canonical(report))
