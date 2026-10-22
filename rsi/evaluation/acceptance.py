"""固定任务集上的配对验收。该模块不修改已接受模型。"""

from dataclasses import dataclass
import math
from typing import Any

from rsi.configuration.validation import boolean, choice, fail, integer, number, obj, string


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    group: str
    passed: bool

    @classmethod
    def parse(cls, value: Any) -> "TaskOutcome":
        raw = obj(value, "outcome", {"task_id", "group", "passed"})
        return cls(string(raw["task_id"], "outcome.task_id"),
                   choice(raw["group"], "outcome.group", {"target", "retention"}),
                   boolean(raw["passed"], "outcome.passed"))


@dataclass(frozen=True)
class AcceptanceRule:
    minimum_gain: float
    retention_tolerance: float
    minimum_tasks_per_group: int
    alpha: float

    @classmethod
    def parse(cls, value: Any) -> "AcceptanceRule":
        raw = obj(value, "acceptance", {"schema_version", "statistical_rule", "minimum_gain",
                  "retention_tolerance", "minimum_tasks_per_group", "alpha", "max_retests"})
        choice(raw["schema_version"], "acceptance.schema_version", {"acceptance-1"})
        choice(raw["statistical_rule"], "acceptance.statistical_rule", {"paired_hoeffding_two_groups_v1"})
        if integer(raw["max_retests"], "acceptance.max_retests") != 0:
            fail("acceptance", "本版本不支持重复检验或自适应停止。")
        alpha = number(raw["alpha"], "acceptance.alpha", 0, 1)
        if not 0 < alpha < 1:
            fail("acceptance.alpha", "必须严格位于 0 与 1 之间。")
        return cls(number(raw["minimum_gain"], "acceptance.minimum_gain", 0, 1),
                   number(raw["retention_tolerance"], "acceptance.retention_tolerance", 0, 1),
                   integer(raw["minimum_tasks_per_group"], "acceptance.minimum_tasks_per_group", 1), alpha)


@dataclass(frozen=True)
class GainInterval:
    sample_count: int
    baseline_pass_rate: float
    candidate_pass_rate: float
    mean_gain: float
    lower: float
    upper: float


@dataclass(frozen=True)
class AcceptanceDecision:
    status: str
    reason: str
    target: GainInterval
    retention: GainInterval


def decide(baseline: tuple[TaskOutcome, ...], candidate: tuple[TaskOutcome, ...],
           rule: AcceptanceRule) -> AcceptanceDecision:
    """按 task_id 配对。任务内测试只决定通过与否，不增加样本量。"""
    number(rule.minimum_gain, "rule.minimum_gain", 0, 1)
    number(rule.retention_tolerance, "rule.retention_tolerance", 0, 1)
    integer(rule.minimum_tasks_per_group, "rule.minimum_tasks_per_group", 1)
    if not 0 < number(rule.alpha, "rule.alpha", 0, 1) < 1:
        fail("rule.alpha", "必须严格位于 0 与 1 之间。")
    # 公开接口也校验直接构造的类型，防止非法布尔值和重复编号绕过 JSON 边界。
    for outcomes in (baseline, candidate):
        for item in outcomes:
            TaskOutcome.parse({"task_id": item.task_id, "group": item.group, "passed": item.passed})
        if len({r.task_id for r in outcomes}) != len(outcomes):
            fail("acceptance", "任务编号重复。")
    left, right = {r.task_id: r for r in baseline}, {r.task_id: r for r in candidate}
    if not left or left.keys() != right.keys() or any(left[k].group != right[k].group for k in left):
        fail("acceptance", "两次评估必须使用相同任务编号和分组。")
    intervals = {}
    for group in ("target", "retention"):
        keys = [k for k in left if left[k].group == group]
        if not keys:
            fail("acceptance", "目标组和能力保持组均不能为空。")
        n = len(keys)
        old, new = sum(left[k].passed for k in keys) / n, sum(right[k].passed for k in keys) / n
        mean = new - old
        # 每个配对差值位于 [-1, 1]。双侧界再对两个组作联合界。
        radius = math.sqrt(2 * math.log(4 / rule.alpha) / n)
        intervals[group] = GainInterval(n, old, new, mean, max(-1, mean - radius), min(1, mean + radius))
    target, retention = intervals["target"], intervals["retention"]
    if min(target.sample_count, retention.sample_count) < rule.minimum_tasks_per_group:
        status, reason = "inconclusive", "任务数量低于预先冻结的下限。"
    elif target.lower >= rule.minimum_gain and retention.lower >= -rule.retention_tolerance:
        status, reason = "accepted", "两个组的保守区间均满足阈值。"
    elif target.upper < rule.minimum_gain or retention.upper < -rule.retention_tolerance:
        status, reason = "rejected", "至少一个组的区间上界不能满足阈值。"
    else:
        status, reason = "inconclusive", "区间跨越阈值，当前证据不能确定。"
    return AcceptanceDecision(status, reason, target, retention)
