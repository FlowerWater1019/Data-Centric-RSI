"""固定参考上下文的拟合测量。支持整条轨迹与技能证据监督片段。"""

from dataclasses import dataclass
from typing import Any, Protocol

from .contracts import ModelRef
from .data import TrajectoryRecord
from .labels import fields
from .scoring import TokenScore
from .validation import array, choice, digest, fail, integer, number, string, strings, to_json_value


class TokenScorer(Protocol):
    def score_tokens(self, record: TrajectoryRecord) -> tuple[TokenScore, ...]: ...


@dataclass(frozen=True)
class SkillEvidence:
    data_id: str
    trajectory_version: str
    skill_id: str
    source: str
    token_positions: tuple[int, ...]

    @classmethod
    def parse(cls, raw: Any) -> "SkillEvidence":
        v = fields(raw, "skill_evidence", "data_id trajectory_version skill_id source token_positions")
        positions = tuple(integer(x, "skill_evidence.token_positions") for x in array(v["token_positions"], "skill_evidence.token_positions"))
        # 重复引用的监督位置按集合去重；不能改变原轨迹的训练掩码。
        return cls(*(string(v[k], "skill_evidence." + k) for k in ("data_id", "trajectory_version", "skill_id")),
                   choice(v["source"], "skill_evidence.source", {"human", "dataset", "code", "synthetic_fixture"}), tuple(sorted(set(positions))))


@dataclass(frozen=True)
class FitStats:
    status: str
    token_count: int
    ce_sum: float
    entropy_sum: float
    nll: float | None
    entropy: float | None


def aggregate(values: tuple[TokenScore, ...]) -> FitStats:
    """按有效 token 聚合，空集合使用 MISSING/null。"""
    ce = sum((number(v.ce, "fit.ce", 0) for v in values), 0.0)
    entropy = sum((number(v.entropy, "fit.entropy", 0) for v in values), 0.0)
    count = len(values)
    number(ce, "fit.ce_sum")
    number(entropy, "fit.entropy_sum")
    return FitStats("COMPLETED" if count else "MISSING", count, ce, entropy,
                    ce / count if count else None, entropy / count if count else None)


@dataclass(frozen=True)
class SkillFit:
    skill_id: str
    stats: FitStats


@dataclass(frozen=True)
class RecordFit:
    data_id: str
    trajectory_version: str
    mask_version: str
    token_scores: tuple[TokenScore, ...]
    whole_record: FitStats
    by_skill: tuple[SkillFit, ...]


@dataclass(frozen=True)
class FitEvaluation:
    schema_version: str
    model: ModelRef
    source: str
    scope: str
    ordered_batch_hash: str
    evidence_hash: str
    scoring_rule_version: str
    records: tuple[RecordFit, ...]
    overall: FitStats
    by_skill: tuple[SkillFit, ...]

    def to_dict(self) -> dict[str, Any]:
        return to_json_value(self)


def evaluate_fit(model: ModelRef, records: tuple[TrajectoryRecord, ...], scorer: TokenScorer,
                 skills: tuple[str, ...], evidence: tuple[SkillEvidence, ...] = (),
                 *, source: str = "measured", scope: str = "selected_training_batch") -> FitEvaluation:
    """同一次前向支持整条数据与技能片段统计。片段只是旁路，不改变 SFT。"""
    choice(source, "fit.source", {"measured", "synthetic_fixture"})
    choice(scope, "fit.scope", {"selected_training_batch", "fixed_diagnostic_set"})
    skills = strings(list(skills), "fit.skills")
    if any(set(r.skills) - set(skills) for r in records):
        fail("fit.skills", "技能表必须覆盖输入轨迹的技能。")
    if not records or len({r.data_id for r in records}) != len(records):
        fail("fit.records", "记录必须非空且不重复。")
    if any(r.role != ("candidate" if scope == "selected_training_batch" else "feedback_diagnostic") for r in records):
        fail("fit.records.role", "拟合范围与数据角色不一致；禁止 final_test。")
    lookup = {r.data_id: r for r in records}
    coverage: dict[tuple[str, str], set[int]] = {}
    for item in evidence:
        SkillEvidence.parse(to_json_value(item))
        r = lookup.get(item.data_id)
        if r is None or item.trajectory_version != r.version or item.skill_id not in skills or item.skill_id not in r.skills:
            fail("fit.evidence", "轨迹版本或技能引用无效。")
        if any(p < 0 or p >= r.token_count for p in item.token_positions):
            fail("fit.evidence.token_positions", "位置越界。")
        coverage.setdefault((r.data_id, item.skill_id), set()).update(item.token_positions)
    rows, all_values = [], []
    by_skill: dict[str, list[TokenScore]] = {s: [] for s in skills}
    for record in records:
        if record.raw["tokenization"]["status"] != "verified":
            fail("fit.tokenization", "真实拟合输入必须已编译；mock 也使用明确验证的测试分词器。")
        values = scorer.score_tokens(record) if record.supervised_tokens else ()
        expected = {i for i, enabled in enumerate(record.raw["supervision"]["loss_mask"]) if enabled}
        if len(values) != len(expected) or {integer(v.position, "fit.position", 1) for v in values} != expected:
            fail("fit.token_scores", "评分位置必须恰好覆盖原有效监督位置，不能重复或遗漏。")
        skill_rows = []
        for skill in skills:
            subset = tuple(v for v in values if v.position in coverage.get((record.data_id, skill), set()))
            by_skill[skill].extend(subset)
            skill_rows.append(SkillFit(skill, aggregate(subset)))
        rows.append(RecordFit(record.data_id, record.version, record.train_mask_version, values, aggregate(values), tuple(skill_rows)))
        all_values.extend(values)
    order_hash = digest([{"data_id": r.data_id, "version": r.version, "mask": r.train_mask_version,
                          "raw_hash": digest(r.raw)} for r in records])
    return FitEvaluation("fit-evaluation-v1", model, source, scope, order_hash, digest(evidence),
                         "supervised_full_vocab_nats_v1", tuple(rows), aggregate(tuple(all_values)),
                         tuple(SkillFit(s, aggregate(tuple(by_skill[s]))) for s in skills))


@dataclass(frozen=True)
class FitDifference:
    before: FitStats
    after: FitStats
    delta_nll: float | None
    delta_entropy: float | None


def difference(before: FitStats, after: FitStats) -> FitDifference:
    if before.token_count != after.token_count:
        fail("fit.pair.token_count", "父候选统计位置数不一致。")
    return FitDifference(before, after, after.nll - before.nll if before.nll is not None and after.nll is not None else None,
                         after.entropy - before.entropy if before.entropy is not None and after.entropy is not None else None)


@dataclass(frozen=True)
class RecordDifference:
    data_id: str
    whole_record: FitDifference
    by_skill: tuple[tuple[str, FitDifference], ...]


@dataclass(frozen=True)
class FitComparison:
    before_hash: str
    after_hash: str
    overall: FitDifference
    per_record: tuple[RecordDifference, ...]
    by_skill: tuple[tuple[str, FitDifference], ...]


def compare_fit(before: FitEvaluation, after: FitEvaluation) -> FitComparison:
    """比较整条轨迹和技能证据片段。两次必须使用同一输入和证据。"""
    if (before.ordered_batch_hash, before.evidence_hash, before.scope, before.scoring_rule_version, before.source) != (
        after.ordered_batch_hash, after.evidence_hash, after.scope, after.scoring_rule_version, after.source):
        fail("fit.pair", "输入、证据、范围或评分规则不一致。")
    if tuple(s.skill_id for s in before.by_skill) != tuple(s.skill_id for s in after.by_skill):
        fail("fit.pair.skills", "技能表不一致。")
    rows = []
    for left, right in zip(before.records, after.records):
        if tuple(s.skill_id for s in left.by_skill) != tuple(s.skill_id for s in right.by_skill):
            fail("fit.pair.records.skills", "逐条技能字段不一致。")
        if (left.data_id, left.trajectory_version, left.mask_version, tuple(v.position for v in left.token_scores)) != (
            right.data_id, right.trajectory_version, right.mask_version, tuple(v.position for v in right.token_scores)):
            fail("fit.pair.records", "逐条输入或统计位置不一致。")
        rows.append(RecordDifference(left.data_id, difference(left.whole_record, right.whole_record),
                    tuple((l.skill_id, difference(l.stats, r.stats)) for l, r in zip(left.by_skill, right.by_skill))))
    if len(before.records) != len(after.records):
        fail("fit.pair.records", "记录数量不一致。")
    return FitComparison(digest(before), digest(after), difference(before.overall, after.overall), tuple(rows),
                         tuple((l.skill_id, difference(l.stats, r.stats)) for l, r in zip(before.by_skill, after.by_skill)))


def parse_stats(raw: Any) -> FitStats:
    v = fields(raw, "fit.stats", "status token_count ce_sum entropy_sum nll entropy")
    count = integer(v["token_count"], "fit.stats.token_count")
    ce, entropy = number(v["ce_sum"], "fit.stats.ce_sum"), number(v["entropy_sum"], "fit.stats.entropy_sum")
    status = choice(v["status"], "fit.stats.status", {"COMPLETED", "MISSING"})
    if count == 0:
        if status != "MISSING" or ce != 0 or entropy != 0 or v["nll"] is not None or v["entropy"] is not None:
            fail("fit.stats", "缺失统计必须为零计数和 null 均值。")
    elif status != "COMPLETED" or abs(number(v["nll"], "fit.stats.nll") - ce / count) > 1e-10 or abs(number(v["entropy"], "fit.stats.entropy") - entropy / count) > 1e-10:
        fail("fit.stats", "总和、计数、状态或均值不一致。")
    return FitStats(status, count, ce, entropy, v["nll"], v["entropy"])


def parse_fit_evaluation(raw: Any) -> FitEvaluation:
    v = fields(raw, "fit", "schema_version model source scope ordered_batch_hash evidence_hash scoring_rule_version records overall by_skill")
    choice(v["schema_version"], "fit.schema_version", {"fit-evaluation-v1"})
    choice(v["scoring_rule_version"], "fit.scoring_rule_version", {"supervised_full_vocab_nats_v1"})

    def skill_rows(value: Any) -> tuple[SkillFit, ...]:
        result = []
        for item in array(value, "fit.by_skill"):
            item = fields(item, "fit.skill", "skill_id stats")
            result.append(SkillFit(string(item["skill_id"], "fit.skill_id"), parse_stats(item["stats"])))
        if len({r.skill_id for r in result}) != len(result):
            fail("fit.skill_id", "技能重复。")
        return tuple(result)

    records = []
    for item in array(v["records"], "fit.records", True):
        item = fields(item, "fit.record", "data_id trajectory_version mask_version token_scores whole_record by_skill")
        tokens = []
        for token in array(item["token_scores"], "fit.token_scores"):
            token = fields(token, "fit.token", "position ce entropy")
            tokens.append(TokenScore(integer(token["position"], "fit.position", 1), number(token["ce"], "fit.ce"), number(token["entropy"], "fit.entropy")))
        if len({x.position for x in tokens}) != len(tokens):
            fail("fit.position", "统计位置重复。")
        whole = parse_stats(item["whole_record"])
        if aggregate(tuple(tokens)) != whole:
            fail("fit.whole_record", "逐 token 证据与汇总不一致。")
        records.append(RecordFit(*(string(item[k], "fit.record." + k) for k in ("data_id", "trajectory_version", "mask_version")),
                                 tuple(tokens), whole, skill_rows(item["by_skill"])))
    overall = parse_stats(v["overall"])
    if aggregate(tuple(t for r in records for t in r.token_scores)) != overall:
        fail("fit.overall", "总体未按 token 总和聚合。")
    skills = skill_rows(v["by_skill"])
    if len({r.data_id for r in records}) != len(records):
        fail("fit.records", "记录重复。")
    for record in records:
        if tuple(s.skill_id for s in record.by_skill) != tuple(s.skill_id for s in skills):
            fail("fit.by_skill", "逐条与总体技能字段不一致。")
        if any(s.stats.token_count > record.whole_record.token_count for s in record.by_skill):
            fail("fit.by_skill", "技能 token 数不能超过整条监督量。")
    for index, skill in enumerate(skills):
        rows = [r.by_skill[index].stats for r in records]
        if skill.stats.token_count != sum(s.token_count for s in rows) or any(
            abs(getattr(skill.stats, key) - sum(getattr(s, key) for s in rows)) > 1e-8 for key in ("ce_sum", "entropy_sum")):
            fail("fit.by_skill", "技能总体未按逐条 token 总和聚合。")
    return FitEvaluation(v["schema_version"], ModelRef.parse(v["model"]), choice(v["source"], "fit.source", {"measured", "synthetic_fixture"}),
              choice(v["scope"], "fit.scope", {"selected_training_batch", "fixed_diagnostic_set"}),
              *(string(v[k], "fit." + k) for k in ("ordered_batch_hash", "evidence_hash", "scoring_rule_version")),
              tuple(records), overall, skills)
