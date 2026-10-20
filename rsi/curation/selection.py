"""随机与固定排序共用约束组批器。组批失败不表示已证明无解。"""

import random
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from rsi.configuration.contracts import ExperimentContract, ModelRef, Preference
from rsi.datasets.data import TrajectoryRecord
from rsi.curation.packing import exact_ranked
from rsi.configuration.validation import BudgetError, QuotaError, digest, fail, to_json_value


@dataclass(frozen=True)
class SelectionContext:
    contract: ExperimentContract
    records: tuple[TrajectoryRecord, ...]
    contract_hash: str
    dataset_hash: str

    def eligible(self) -> list[TrajectoryRecord]:
        return sorted((r for r in self.records if r.is_eligible), key=lambda r: r.data_id)

    @property
    def context_id(self) -> str:
        return digest({"contract": self.contract_hash, "dataset": self.dataset_hash,
                       "eligible": [r.data_id for r in self.eligible()]})


class Selector(Protocol):
    """返回完整候选集合的排序。不得修改情境或候选内容。"""
    name: str

    def rank(self, context: SelectionContext) -> list[TrajectoryRecord]: ...


class RandomSelector:
    name = "random"

    def rank(self, context: SelectionContext) -> list[TrajectoryRecord]:
        records = context.eligible()
        random.Random(context.contract.seed).shuffle(records)
        return records


def utility(value: float, preference: Preference) -> float:
    a, d = preference.support
    b, c = preference.preferred
    if value <= a or value >= d:
        return 0.0
    if b <= value <= c:
        return 1.0
    return (value - a) / (b - a) if value < b else (d - value) / (d - c)


@dataclass(frozen=True)
class ScoreTerms:
    nll_utility: float
    entropy_utility: float
    skill_match: float
    weighted_nll: float
    weighted_entropy: float
    weighted_skill: float
    total: float


def score_terms(record: TrajectoryRecord, contract: ExperimentContract) -> ScoreTerms:
    """排名与解释共用评分。调用方先核对当前评分版本。"""
    strategy = contract.fixed_strategy
    scores = record.scores
    if scores is None:
        fail(record.data_id, "固定策略缺少当前分数。")
    nll = utility(scores.nll_percentile, strategy.nll)
    entropy = utility(scores.entropy_percentile, strategy.entropy)
    skill = max((strategy.skill_preferences.get(k, 0) for k in record.skills), default=0)
    wn, we, ws = strategy.weights["nll"] * nll, strategy.weights["entropy"] * entropy, strategy.weights["skill"] * skill
    return ScoreTerms(nll, entropy, skill, wn, we, ws, wn + we + ws)


class FixedSelector:
    name = "fixed"

    def rank(self, context: SelectionContext) -> list[TrajectoryRecord]:
        records = context.eligible()
        for record in records:
            record.require_current_scores(context.contract)

        return sorted(records, key=lambda r: (-score_terms(r, context.contract).total, r.data_id))


def check_task_type_shares(shares: Mapping[str, float], minimum: Mapping[str, float],
                           maximum: Mapping[str, float]) -> None:
    """组批后核对任务类型配额。不重排或放宽约束。"""
    for task, value in minimum.items():
        if shares.get(task, 0) < value:
            raise QuotaError("batch.task_type.min." + task + ": 配额检查失败；没有自动放宽配额。")
    for task, value in maximum.items():
        if shares.get(task, 0) > value:
            raise QuotaError("batch.task_type.max." + task + ": 配额检查失败；没有自动放宽配额。")


@dataclass(frozen=True)
class BatchItem:
    data_id: str
    version: str
    mask_version: str
    content_hash: str
    repetitions: int
    supervised_tokens: int


@dataclass(frozen=True)
class BatchManifest:
    selector: str
    context_id: str
    model: ModelRef
    seed: int
    items: tuple[BatchItem, ...]
    supervised_tokens: int
    context_tokens: int
    skill_shares: Mapping[str, float]
    fingerprint: str
    training_status: str = "NOT_RUN"

    def __post_init__(self) -> None:
        object.__setattr__(self, "skill_shares", MappingProxyType(dict(self.skill_shares)))

    def to_dict(self) -> dict[str, object]:
        """返回独立的可序列化副本。JSON 字段保持 1.0 格式。"""
        return to_json_value(self)


def build_batch(context: SelectionContext, selector: Selector) -> BatchManifest:
    """按完整轨迹组批。预算或配额不满足时抛出 ValidationError。"""
    originals = {r.data_id: r for r in context.eligible()}
    ranked = selector.rank(context)
    eligible_ids = set(originals)
    if len(ranked) != len(eligible_ids) or {r.data_id for r in ranked} != eligible_ids:
        fail("selector", "排序器必须完整返回可选集合，且不能重复。")
    if any(record != originals[record.data_id] for record in ranked):
        fail("selector", "排序器不能修改轨迹、分数或监督量。")
    selected = []
    duplicate_groups = set()
    content_hashes = set()
    total = 0
    candidates = exact_ranked(ranked, context.contract.token_budget) if context.contract.packing_rule == "exact_ranked_v1" else ranked
    for record in candidates:
        if record.duplicate_group in duplicate_groups or record.content_hash in content_hashes:
            continue
        if total + record.supervised_tokens > context.contract.token_budget:
            continue
        selected.append(record)
        total += record.supervised_tokens
        duplicate_groups.add(record.duplicate_group)
        content_hashes.add(record.content_hash)
    if not selected or total < context.contract.token_budget - context.contract.allowed_shortfall:
        raise BudgetError("batch: 贪心组批未找到预算内合法批次；未证明约束无解。")
    shares = {k: sum(r.supervised_tokens for r in selected if k in r.skills) / total for k in context.contract.skills}
    for skill, minimum in context.contract.min_skill_shares.items():
        if shares[skill] < minimum:
            raise QuotaError(f"batch: 贪心组批未满足 {skill} 覆盖要求；没有自动放宽配额。")
    items = tuple(BatchItem(r.data_id, r.version, r.train_mask_version, r.content_hash, 1, r.supervised_tokens)
                  for r in selected)
    fingerprint = digest(sorted(items, key=lambda item: item.data_id))
    return BatchManifest(selector.name, context.context_id, context.contract.model, context.contract.seed,
                         items, total, sum(r.token_count for r in selected), shares, fingerprint)
