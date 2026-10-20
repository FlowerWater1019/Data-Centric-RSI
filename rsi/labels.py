"""独立标签快照。标签只能收紧轨迹资格，不能改写轨迹。"""

from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import ExperimentContract
from .data import ROLES, TrajectoryRecord
from .validation import (array, boolean, choice, fail, integer, number, obj,
                         string, strings, to_json_value)


SOURCES = {"seeded_random", "human", "teacher", "code", "dataset", "synthetic_fixture"}
FLAGS = {"trajectory_complete", "result_verified", "protocol_valid", "quality_pass", "no_answer_leakage"}
MODES = {"synthetic_fixture", "annotated"}
DOMAINS = {"coding", "general"}
TRAJECTORY_TYPES = {"single_turn", "multi_turn", "tool_interaction"}


def fields(raw: Any, path: str, names: str) -> dict[str, Any]:
    return obj(raw, path, set(names.split()))


def nullable_id(value: Any, path: str) -> str | None:
    return None if value is None else string(value, path)


def references(value: Any, path: str, evidence_ids: frozenset[str]) -> tuple[str, ...]:
    result = strings(value, path)
    if set(result) - evidence_ids:
        fail(path, "证据引用无法解析。")
    return result


@dataclass(frozen=True)
class Skill:
    id: str
    name: str
    description: str


@dataclass(frozen=True)
class SkillOntology:
    schema_version: str
    ontology_version: str
    domain: str
    skills: tuple[Skill, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.skills)

    def to_dict(self) -> dict[str, Any]:
        return to_json_value(self)

    @classmethod
    def parse(cls, raw: Any) -> "SkillOntology":
        value = fields(raw, "ontology", "schema_version ontology_version domain skills")
        choice(value["schema_version"], "ontology.schema_version", {"skill-ontology-v1"})
        items = []
        for i, item in enumerate(array(value["skills"], "ontology.skills", True)):
            path = f"ontology.skills[{i}]"
            item = fields(item, path, "id name description")
            items.append(Skill(**{k: string(v, f"{path}.{k}") for k, v in item.items()}))
        if len({s.id for s in items}) != len(items):
            fail("ontology.skills", "技能 ID 重复。")
        return cls(value["schema_version"], string(value["ontology_version"], "ontology.ontology_version"),
                   choice(value["domain"], "ontology.domain", DOMAINS), tuple(items))


@dataclass(frozen=True)
class Provenance:
    dataset_id: str
    dataset_revision: str
    source_record_id: str
    label_mode: str
    annotation_seed: int | None


@dataclass(frozen=True)
class Partition:
    role: str
    split_group: str
    duplicate_group: str


@dataclass(frozen=True)
class Task:
    task_type: str
    trajectory_type: str
    harness_protocol: str
    language: str


@dataclass(frozen=True)
class TaskRef:
    task_id: str
    repository: str | None
    base_commit: str | None
    environment_id: str | None
    evaluation_spec_id: str | None


@dataclass(frozen=True)
class ProtocolRef:
    harness_id: str
    harness_version: str
    tool_schema_version: str | None


@dataclass(frozen=True)
class QualityFlags:
    trajectory_complete: bool | None
    result_verified: bool | None
    protocol_valid: bool | None
    quality_pass: bool | None
    no_answer_leakage: bool | None


@dataclass(frozen=True)
class Annotation:
    source: str
    rule_version: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class SkillLabel:
    skill_id: str
    source: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class SkillLabels:
    ontology_version: str
    items: tuple[SkillLabel, ...]


@dataclass(frozen=True)
class Cost:
    tokenization_version: str
    context_tokens: int
    supervised_tokens: int
    length_bucket: str


@dataclass(frozen=True)
class MissingScores:
    status: str = "missing"


@dataclass(frozen=True)
class AvailableScores:
    status: str
    checkpoint_id: str
    trajectory_version: str
    mask_version: str
    scoring_rule_version: str
    reference_version: str
    cdf_version: str
    normalization_group: str
    nll: float
    entropy: float
    nll_percentile: float
    entropy_percentile: float


@dataclass(frozen=True)
class Usage:
    ever_used_in_training_attempt: bool
    in_accepted_lineage: bool
    training_attempt_ids: tuple[str, ...]
    cumulative_supervised_tokens: int


@dataclass(frozen=True)
class DataLabels:
    schema_version: str
    data_id: str
    trajectory_version: str
    label_version: str
    domain: str
    provenance: Provenance
    partition: Partition
    task: Task
    task_ref: TaskRef
    protocol_ref: ProtocolRef
    quality_flags: QualityFlags
    quality_annotation: Annotation
    skills: SkillLabels
    cost: Cost
    model_scores: MissingScores | AvailableScores
    usage: Usage

    def to_dict(self) -> dict[str, Any]:
        return to_json_value(self)


def parse_labels(raw: Any, ontology: SkillOntology,
                 evidence_ids: frozenset[str] = frozenset()) -> DataLabels:
    """解析一条标签。非空证据必须在调用方提供的注册表中。"""
    p = "labels"
    v = obj(raw, p, set(DataLabels.__dataclass_fields__))
    choice(v["schema_version"], p + ".schema_version", {"data-labels-v1"})
    domain = choice(v["domain"], p + ".domain", DOMAINS)
    if domain != ontology.domain:
        fail(p + ".domain", "与技能表领域不一致。")
    pro = fields(v["provenance"], p + ".provenance", "dataset_id dataset_revision source_record_id label_mode annotation_seed")
    mode = choice(pro["label_mode"], p + ".provenance.label_mode", MODES)
    seed = pro["annotation_seed"]
    if seed is not None or mode == "synthetic_fixture":
        seed = integer(seed, p + ".provenance.annotation_seed")
    provenance = Provenance(*(string(pro[k], p + ".provenance." + k) for k in
                              ("dataset_id", "dataset_revision", "source_record_id")), mode, seed)
    part = fields(v["partition"], p + ".partition", "role split_group duplicate_group")
    partition = Partition(choice(part["role"], p + ".partition.role", ROLES),
                          string(part["split_group"], p + ".partition.split_group"),
                          string(part["duplicate_group"], p + ".partition.duplicate_group"))
    task_raw = fields(v["task"], p + ".task", "task_type trajectory_type harness_protocol language")
    task = Task(string(task_raw["task_type"], p + ".task.task_type"),
                choice(task_raw["trajectory_type"], p + ".task.trajectory_type", TRAJECTORY_TYPES),
                string(task_raw["harness_protocol"], p + ".task.harness_protocol"),
                string(task_raw["language"], p + ".task.language"))
    tr = fields(v["task_ref"], p + ".task_ref", "task_id repository base_commit environment_id evaluation_spec_id")
    task_ref = TaskRef(string(tr["task_id"], p + ".task_ref.task_id"),
                       **{k: nullable_id(tr[k], p + ".task_ref." + k) for k in tr if k != "task_id"})
    pr = fields(v["protocol_ref"], p + ".protocol_ref", "harness_id harness_version tool_schema_version")
    protocol_ref = ProtocolRef(string(pr["harness_id"], p + ".protocol_ref.harness_id"),
                               string(pr["harness_version"], p + ".protocol_ref.harness_version"),
                               nullable_id(pr["tool_schema_version"], p + ".protocol_ref.tool_schema_version"))
    flags = obj(v["quality_flags"], p + ".quality_flags", FLAGS)
    quality_flags = QualityFlags(**{k: None if x is None else boolean(x, p + ".quality_flags." + k) for k, x in flags.items()})
    ann = fields(v["quality_annotation"], p + ".quality_annotation", "source rule_version evidence_refs")
    annotation = Annotation(choice(ann["source"], p + ".quality_annotation.source", SOURCES),
                            string(ann["rule_version"], p + ".quality_annotation.rule_version"),
                            references(ann["evidence_refs"], p + ".quality_annotation.evidence_refs", evidence_ids))
    sk = fields(v["skills"], p + ".skills", "ontology_version items")
    if sk["ontology_version"] != ontology.ontology_version:
        fail(p + ".skills.ontology_version", "与技能表版本不一致。")
    items = []
    for i, item in enumerate(array(sk["items"], p + ".skills.items")):
        path = f"{p}.skills.items[{i}]"
        item = fields(item, path, "skill_id source evidence_refs")
        items.append(SkillLabel(choice(item["skill_id"], path + ".skill_id", set(ontology.ids)),
                                choice(item["source"], path + ".source", SOURCES),
                                references(item["evidence_refs"], path + ".evidence_refs", evidence_ids)))
    if len({s.skill_id for s in items}) != len(items):
        fail(p + ".skills.items", "技能重复。")
    cost_raw = fields(v["cost"], p + ".cost", "tokenization_version context_tokens supervised_tokens length_bucket")
    cost = Cost(string(cost_raw["tokenization_version"], p + ".cost.tokenization_version"),
                integer(cost_raw["context_tokens"], p + ".cost.context_tokens"),
                integer(cost_raw["supervised_tokens"], p + ".cost.supervised_tokens"),
                string(cost_raw["length_bucket"], p + ".cost.length_bucket"))
    if cost.supervised_tokens > cost.context_tokens:
        fail(p + ".cost.supervised_tokens", "不能超过总输入长度。")
    scores = parse_label_scores(v["model_scores"])
    use = fields(v["usage"], p + ".usage", "ever_used_in_training_attempt in_accepted_lineage training_attempt_ids cumulative_supervised_tokens")
    usage = Usage(boolean(use["ever_used_in_training_attempt"], p + ".usage.ever_used_in_training_attempt"),
                  boolean(use["in_accepted_lineage"], p + ".usage.in_accepted_lineage"),
                  strings(use["training_attempt_ids"], p + ".usage.training_attempt_ids"),
                  integer(use["cumulative_supervised_tokens"], p + ".usage.cumulative_supervised_tokens"))
    if usage.ever_used_in_training_attempt != bool(usage.training_attempt_ids) or (
        usage.ever_used_in_training_attempt != (usage.cumulative_supervised_tokens > 0)) or (
        usage.in_accepted_lineage and not usage.ever_used_in_training_attempt) or (
        not usage.ever_used_in_training_attempt and usage.cumulative_supervised_tokens):
        fail(p + ".usage", "使用历史矛盾。")
    return DataLabels(v["schema_version"], string(v["data_id"], p + ".data_id"),
                      string(v["trajectory_version"], p + ".trajectory_version"),
                      string(v["label_version"], p + ".label_version"), domain, provenance, partition,
                      task, task_ref, protocol_ref, quality_flags, annotation,
                      SkillLabels(sk["ontology_version"], tuple(items)), cost, scores, usage)


def parse_label_scores(raw: Any) -> MissingScores | AvailableScores:
    path = "labels.model_scores"
    if isinstance(raw, dict) and raw.get("status") == "missing":
        fields(raw, path, "status")
        return MissingScores()
    v = obj(raw, path, set(AvailableScores.__dataclass_fields__))
    choice(v["status"], path + ".status", {"available"})
    choice(v["normalization_group"], path + ".normalization_group", {"all_candidates"})
    numeric = {"nll", "entropy", "nll_percentile", "entropy_percentile"}
    return AvailableScores(**{k: number(x, path + "." + k, 0, 1 if k.endswith("percentile") else None)
                              if k in numeric else string(x, path + "." + k) for k, x in v.items()})


def scores_snapshot(record: TrajectoryRecord) -> dict[str, Any]:
    if record.scores is None:
        return {"status": "missing"}
    result = to_json_value(record.scores)
    result["scoring_rule_version"] = result.pop("rule_version")
    return {"status": "available", "normalization_group": "all_candidates", **result}


def bind_labels(records: tuple[TrajectoryRecord, ...], labels: tuple[DataLabels, ...],
                ontology: SkillOntology, contract: ExperimentContract,
                *, allow_synthetic_labels: bool = False) -> Mapping[str, DataLabels]:
    """候选与标签一对一关联。参考和探针不需要标签。拒绝最终测试数据。"""
    from types import MappingProxyType
    if any(r.role == "final_test" for r in records):
        fail("data.role", "新选择入口禁止 final_test。")
    if set(ontology.ids) != set(contract.skills):
        fail("ontology.skills", "与合同技能表不一致。")
    candidates = {r.data_id: r for r in records if r.role == "candidate"}
    bound = {label.data_id: label for label in labels}
    if len(bound) != len(labels):
        fail("labels.data_id", "标签重复。")
    if bound.keys() != candidates.keys():
        fail("labels.data_id", f"缺失标签 {sorted(candidates.keys() - bound.keys())}；孤儿标签 {sorted(bound.keys() - candidates.keys())}。")
    for data_id, label in bound.items():
        r = candidates[data_id]
        path = "labels." + data_id
        if label.domain != ontology.domain or label.skills.ontology_version != ontology.ontology_version:
            fail(path, "领域或技能表版本不匹配。")
        if label.provenance.label_mode == "synthetic_fixture" and not (allow_synthetic_labels and contract.purpose == "smoke"):
            fail(path + ".provenance.label_mode", "模拟标签需要显式 allow-synthetic-labels 和 smoke 配置。")
        expected = {"trajectory_version": r.version, "partition": Partition(r.role, r.split_group, r.duplicate_group),
                    "usage": r.raw["usage"], "model_scores": scores_snapshot(r)}
        actual = label.to_dict()
        for key, item in expected.items():
            if actual[key] != to_json_value(item):
                fail(path + "." + key, "与轨迹快照不一致。")
        if tuple(s.skill_id for s in label.skills.items) != r.skills:
            fail(path + ".skills", "与轨迹快照不一致。")
        if (label.cost.context_tokens, label.cost.supervised_tokens) != (r.token_count, r.supervised_tokens):
            fail(path + ".cost", "与轨迹 token 数不一致。")
        if label.cost.tokenization_version != r.raw["tokenization"]["tokenizer_revision"]:
            fail(path + ".cost.tokenization_version", "与轨迹分词版本不一致。")
    return MappingProxyType(bound)
