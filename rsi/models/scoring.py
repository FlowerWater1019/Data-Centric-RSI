"""评分统计与参考分布。最终测试数据不进入评分或百分位计算。"""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from rsi.configuration.contracts import ExperimentContract
from rsi.datasets.data import TrajectoryRecord
from rsi.configuration.validation import fail, number, to_json_value


@dataclass(frozen=True)
class SpanMetric:
    skill: str
    token_count: int
    ce_sum: float
    mean_ce: float


@dataclass(frozen=True)
class TokenMetrics:
    data_id: str
    supervised_tokens: int
    context_tokens: int
    nll: float
    entropy: float
    spans: tuple[SpanMetric, ...]


def empirical_percentile(value: float, reference: list[float]) -> float:
    """使用参考组经验分布。相同值使用中点秩。"""
    number(value, "score")
    if not reference:
        fail("reference", "参考组不能为空。")
    ordered = sorted(number(item, "reference") for item in reference)
    return (bisect_left(ordered, value) + bisect_right(ordered, value)) / (2 * len(ordered))


def attach_scores(
    records: tuple[TrajectoryRecord, ...], metrics: dict[str, TokenMetrics], contract: ExperimentContract,
) -> list[dict]:
    references = [r for r in records if r.role == "reference"]
    if not references or any(r.data_id not in metrics for r in references):
        fail("reference", "缺少当前模型的完整参考组评分。")
    nll = [metrics[r.data_id].nll for r in references]
    entropy = [metrics[r.data_id].entropy for r in references]
    result = []
    for record in records:
        raw = record.raw
        if record.role == "candidate":
            if record.data_id not in metrics:
                fail(record.data_id, "缺少候选评分。")
            metric = metrics[record.data_id]
            raw["scores"] = {
                "checkpoint_id": contract.model.checkpoint_id, "trajectory_version": record.version,
                "mask_version": record.train_mask_version, **to_json_value(contract.scoring),
                "nll": metric.nll, "entropy": metric.entropy,
                "nll_percentile": empirical_percentile(metric.nll, nll),
                "entropy_percentile": empirical_percentile(metric.entropy, entropy),
            }
        result.append(raw)
    return result
