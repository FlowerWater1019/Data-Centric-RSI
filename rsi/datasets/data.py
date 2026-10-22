"""轨迹格式、监督范围及数据角色隔离。"""

from dataclasses import dataclass
from typing import Any

from rsi.configuration.contracts import ExperimentContract
from rsi.configuration.validation import array, boolean, canonical, choice, fail, integer, loads, number, obj, string, strings, text_digest, to_json_value


ROLES = {"candidate", "score_probe", "reference", "feedback_diagnostic", "feedback_execution", "final_test"}


@dataclass(frozen=True)
class ScoreSnapshot:
    checkpoint_id: str
    trajectory_version: str
    mask_version: str
    rule_version: str
    reference_version: str
    cdf_version: str
    nll: float
    entropy: float
    nll_percentile: float
    entropy_percentile: float


@dataclass(frozen=True)
class TrajectoryRecord:
    data_id: str
    version: str
    role: str
    split_group: str
    duplicate_group: str
    content_hash: str
    skills: tuple[str, ...]
    supervised_tokens: int
    token_count: int
    train_mask_version: str
    quality_pass: bool
    in_accepted_lineage: bool
    scores: ScoreSnapshot | None
    _raw_json: str

    @property
    def raw(self) -> dict[str, Any]:
        """返回独立的原始数据副本。修改副本不会改变已校验记录。"""
        return loads(self._raw_json)

    @property
    def is_eligible(self) -> bool:
        """只判断基础资格。评分有效性与组批约束另行检查。"""
        return self.role == "candidate" and self.quality_pass and not self.in_accepted_lineage and self.supervised_tokens > 0

    @classmethod
    def parse(cls, raw: Any, contract: ExperimentContract, path: str) -> "TrajectoryRecord":
        value = obj(raw, path, {"schema_version", "data_id", "version", "role", "split_group", "duplicate_group",
                    "text", "text_sha256", "skills", "quality", "tokenization", "supervision", "diagnostic_spans",
                    "usage", "scores"})
        if value["schema_version"] != "1.0":
            fail(path, "数据版本必须是 1.0。")
        text = string(value["text"], path + ".text")
        if value["text_sha256"] != text_digest(text):
            fail(path + ".text_sha256", "文本内容与 SHA256 不一致。")
        role = choice(value["role"], path + ".role", ROLES)
        skills = strings(value["skills"], path + ".skills")
        if set(skills) - set(contract.skills):
            fail(path + ".skills", "包含未定义技能。")
        quality_pass = _validate_quality(value["quality"], path)
        tokens, count = _validate_tokenization(value["tokenization"], text, contract, path)
        mask, mask_version = _validate_supervision(value["supervision"], tokens, count, path)
        _validate_diagnostic_spans(value["diagnostic_spans"], skills, tokens, mask, count, path)
        lineage = _validate_usage(value["usage"], role, path)
        scores = _parse_scores(value["scores"], path)
        return cls(string(value["data_id"], path), string(value["version"], path), role,
                   string(value["split_group"], path), string(value["duplicate_group"], path), value["text_sha256"],
                   skills, sum(mask), count, mask_version, quality_pass, lineage, scores, canonical(value))

    def require_current_scores(self, contract: ExperimentContract) -> None:
        if self.scores is None:
            fail(self.data_id, "固定策略缺少当前分数。")
        expected = {"checkpoint_id": contract.model.checkpoint_id, "trajectory_version": self.version,
                    "mask_version": self.train_mask_version, **to_json_value(contract.scoring)}
        for key, expected_value in expected.items():
            if getattr(self.scores, key) != expected_value:
                fail(f"{self.data_id}.scores.{key}", "分数版本过期或不相容。")


def validate_dataset(raw: list[dict[str, Any]], contract: ExperimentContract) -> tuple[TrajectoryRecord, ...]:
    array(raw, "dataset", nonempty=True)
    records = tuple(TrajectoryRecord.parse(item, contract, f"data[{i}]") for i, item in enumerate(raw))
    if len({r.data_id for r in records}) != len(records):
        fail("dataset", "data_id 重复。")
    allowed = [{"score_probe", "reference"}]
    if contract.allow_feedback_overlap:
        allowed.append({"feedback_diagnostic", "feedback_execution"})
    for attribute in ("split_group", "duplicate_group", "content_hash"):
        groups: dict[str, set[str]] = {}
        for record in records:
            groups.setdefault(getattr(record, attribute), set()).add(record.role)
        for group, roles in groups.items():
            if len(roles) > 1 and not any(roles <= shared for shared in allowed):
                fail(f"dataset.{attribute}.{group}", "数据角色存在泄漏或非法重叠。")
    return records



def _validate_quality(raw: Any, path: str) -> bool:
    quality = obj(raw, path + ".quality", {"verification", "evidence_id", "integrity_valid", "leakage"})
    verification = choice(quality["verification"], path, {"pass", "fail", "unknown"})
    leakage = choice(quality["leakage"], path, {"clear", "flagged", "unknown"})
    integrity = boolean(quality["integrity_valid"], path)
    if verification == "pass":
        string(quality["evidence_id"], path + ".quality.evidence_id")
    elif quality["evidence_id"] is not None:
        string(quality["evidence_id"], path + ".quality.evidence_id")
    return verification == "pass" and leakage == "clear" and integrity


def _validate_tokenization(
    raw: Any, text: str, contract: ExperimentContract, path: str,
) -> tuple[dict[str, Any], int]:
    tokens = obj(raw, path + ".tokenization", {"status", "tokenizer_revision", "template_revision",
                 "token_ids", "offsets", "roles", "correctness"})
    status = choice(tokens["status"], path, {"demo", "verified"})
    if status == "demo" and contract.purpose != "smoke":
        fail(path, "演示分词只能用于 smoke 配置。")
    if tokens["tokenizer_revision"] != contract.model.tokenizer_revision or tokens["template_revision"] != contract.model.template_revision:
        fail(path + ".tokenization", "分词器或模板版本不匹配。")
    ids = array(tokens["token_ids"], path, True)
    for token in ids:
        integer(token, path + ".token_ids")
    count = len(ids)
    for field in ("offsets", "roles", "correctness"):
        if len(array(tokens[field], path + "." + field)) != count:
            fail(path, "分词字段长度不一致。")
    end_previous = 0
    for i, pair in enumerate(tokens["offsets"]):
        pair = array(pair, path + ".offsets")
        if len(pair) != 2:
            fail(path, "位置区间需要两个端点。")
        start, end = [integer(x, path + ".offsets") for x in pair]
        if not end_previous <= start <= end <= len(text):
            fail(path, "字符位置必须有序且位于文本内。")
        end_previous = end
        choice(tokens["roles"][i], path + ".roles", {"system", "user", "assistant", "tool"})
        choice(tokens["correctness"][i], path + ".correctness", {"valid", "incorrect", "unknown", "context"})
    return tokens, count


def _validate_supervision(
    raw: Any, tokens: dict[str, Any], count: int, path: str,
) -> tuple[list[bool], str]:
    supervision = obj(raw, path + ".supervision", {"mask_version", "loss_mask"})
    mask = array(supervision["loss_mask"], path + ".loss_mask")
    if len(mask) != count:
        fail(path, "训练掩码与 token 数量不一致。")
    if mask[0] is True:
        fail(path, "首个 token 没有前序位置，不能计入因果语言模型监督量。")
    for i, enabled in enumerate(mask):
        boolean(enabled, path + ".loss_mask")
        if enabled and (tokens["roles"][i] != "assistant" or tokens["correctness"][i] != "valid"):
            fail(path, "正向监督只能覆盖有有效证据的 assistant token。")
    return mask, string(supervision["mask_version"], path + ".supervision.mask_version")


def _validate_diagnostic_spans(
    raw: Any, skills: tuple[str, ...], tokens: dict[str, Any],
    mask: list[bool], count: int, path: str,
) -> None:
    for i, raw_span in enumerate(array(raw, path + ".diagnostic_spans")):
        span = obj(raw_span, path + f".diagnostic_spans[{i}]", {"skill", "token_start", "token_end",
                   "char_start", "char_end", "evidence_id", "mask_version"})
        if span["skill"] not in skills:
            fail(path, "诊断技能不在本轨迹技能标签中。")
        start = integer(span["token_start"], path)
        end = integer(span["token_end"], path, 1)
        if not 0 <= start < end <= count:
            fail(path, "诊断 token 区间越界。")
        char_start = integer(span["char_start"], path)
        char_end = integer(span["char_end"], path, 1)
        if [char_start, char_end] != [tokens["offsets"][start][0], tokens["offsets"][end - 1][1]]:
            fail(path, "诊断字符位置与 token 位置不一致。")
        if not any(mask[start:end]):
            fail(path, "诊断片段没有有效监督 token。")
        string(span["evidence_id"], path)
        string(span["mask_version"], path)


def _validate_usage(raw: Any, role: str, path: str) -> bool:
    usage = obj(raw, path + ".usage", {"ever_used_in_training_attempt", "in_accepted_lineage",
                "training_attempt_ids", "cumulative_supervised_tokens"})
    ever_used = boolean(usage["ever_used_in_training_attempt"], path)
    lineage = boolean(usage["in_accepted_lineage"], path)
    attempts = strings(usage["training_attempt_ids"], path)
    cumulative = integer(usage["cumulative_supervised_tokens"], path)
    if ever_used != bool(attempts) or ever_used != (cumulative > 0) or (lineage and not ever_used):
        fail(path + ".usage", "训练使用记录相互矛盾。")
    if role != "candidate" and ever_used:
        fail(path + ".usage", "评分或验证保留数据不得带有梯度训练使用记录。")
    return lineage


def _parse_scores(raw: Any, path: str) -> ScoreSnapshot | None:
    scores = raw
    if scores is not None:
        scores = obj(scores, path + ".scores", {"checkpoint_id", "trajectory_version", "mask_version",
                     "rule_version", "reference_version", "cdf_version", "nll", "entropy", "nll_percentile", "entropy_percentile"})
        for key in ("checkpoint_id", "trajectory_version", "mask_version", "rule_version", "reference_version", "cdf_version"):
            string(scores[key], path + ".scores." + key)
        number(scores["nll"], path + ".nll")
        number(scores["entropy"], path + ".entropy")
        number(scores["nll_percentile"], path, 0, 1)
        number(scores["entropy_percentile"], path, 0, 1)
    return ScoreSnapshot(**scores) if scores is not None else None
