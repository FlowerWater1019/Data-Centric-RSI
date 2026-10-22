"""标准来源编译与消息证据映射。静态标签与编译快照分开。"""

from dataclasses import dataclass
from typing import Any, Protocol

from rsi.configuration.contracts import ExperimentContract
from rsi.datasets.data import TrajectoryRecord, validate_dataset
from rsi.evaluation.fit_evaluation import SkillEvidence
from rsi.datasets.labels import DataLabels, SkillOntology, fields, parse_labels, scores_snapshot
from rsi.datasets.tokenization import ChatTokenizer, compile_trajectory, render_with_ranges
from rsi.configuration.validation import array, choice, digest, fail, integer, string, to_json_value


class SourceAdapter(Protocol):
    def normalize(self, value: dict[str, Any]) -> dict[str, Any]: ...


class StandardSourceAdapter:
    """source-1 保留全部已声明字段，未知工具结构交由编译器拒绝。"""
    def normalize(self, value: dict[str, Any]) -> dict[str, Any]:
        return dict(fields(value, "source", "schema_version data_id role split_group duplicate_group skills quality messages"))


@dataclass(frozen=True)
class MessageMapping:
    message_index: int
    role: str
    char_start: int
    char_end: int
    token_positions: tuple[int, ...]


@dataclass(frozen=True)
class SourceMapping:
    data_id: str
    trajectory_version: str
    source_hash: str
    messages: tuple[MessageMapping, ...]


@dataclass(frozen=True)
class CompiledTrainingBundle:
    records: tuple[TrajectoryRecord, ...]
    mappings: tuple[SourceMapping, ...]
    sources_hash: str


@dataclass(frozen=True)
class TrainingProvenance:
    schema_version: str
    data_kind: str
    execution_mode: str
    source: str
    sources_hash: str
    static_labels_hash: str
    tokenizer_execution_mode: str

    @classmethod
    def parse(cls, raw: Any) -> "TrainingProvenance":
        value = fields(raw, "compiled.provenance", " ".join(cls.__dataclass_fields__))
        choice(value["schema_version"], "compiled.provenance.schema_version", {"training-provenance-v1"})
        choice(value["data_kind"], "compiled.provenance.data_kind", {"synthetic", "real_public", "private"})
        mode = choice(value["execution_mode"], "compiled.provenance.execution_mode", {"mock", "hf_local"})
        if value["tokenizer_execution_mode"] != mode or value["source"] != ("measured" if mode == "hf_local" else "synthetic_fixture"):
            fail("compiled.provenance", "执行模式、分词方式与事实来源不一致。")
        string(value["sources_hash"], "compiled.provenance.sources_hash")
        string(value["static_labels_hash"], "compiled.provenance.static_labels_hash")
        return cls(**value)


def compile_training_sources(sources: list[dict[str, Any]], tokenizer: ChatTokenizer,
                             contract: ExperimentContract, max_length: int,
                             adapter: SourceAdapter | None = None) -> CompiledTrainingBundle:
    """完整编译，不截断。保留来源与消息位置供旁路技能拟合使用。"""
    integer(max_length, "max_length", 2)
    adapter = adapter or StandardSourceAdapter()
    normalized = [adapter.normalize(row) for row in sources]
    if any(row["role"] not in {"candidate", "reference", "score_probe", "feedback_diagnostic"} for row in normalized):
        fail("sources.role", "训练来源禁止 final_test 或反馈执行任务。")
    compiled, mappings = [], []
    for row in normalized:
        raw = compile_trajectory(row, tokenizer, contract, max_length)
        _, ranges = render_with_ranges(tokenizer, row["messages"])
        offsets = raw["tokenization"]["offsets"]
        messages = tuple(MessageMapping(i, row["messages"][i]["role"], left, right,
                     tuple(j for j, (start, end) in enumerate(offsets) if left <= start < end <= right))
                     for i, (left, right) in enumerate(ranges))
        mappings.append(SourceMapping(raw["data_id"], raw["version"], digest(row), messages))
        compiled.append(raw)
    return CompiledTrainingBundle(validate_dataset(compiled, contract), tuple(mappings), digest(normalized))


def rebind_label_snapshots(records: tuple[TrajectoryRecord, ...], metadata: tuple[DataLabels, ...],
                           ontology: SkillOntology) -> tuple[DataLabels, ...]:
    """显式更新编译/评分快照；保留静态标签及来源，不自动重标技能。"""
    lookup = {item.data_id: item for item in metadata}
    candidates = {r.data_id: r for r in records if r.role == "candidate"}
    if len(lookup) != len(metadata) or lookup.keys() != candidates.keys():
        fail("label_metadata", "候选静态标签必须一对一提供。")
    result = []
    for data_id, record in candidates.items():
        label = lookup[data_id]
        if tuple(x.skill_id for x in label.skills.items) != record.skills:
            fail("label_metadata.skills", "静态技能与来源不一致，禁止自动重标。")
        raw = label.to_dict()
        raw.update(trajectory_version=record.version,
                   partition={"role": record.role, "split_group": record.split_group, "duplicate_group": record.duplicate_group},
                   usage=record.raw["usage"], model_scores=scores_snapshot(record))
        raw["cost"].update(tokenization_version=record.raw["tokenization"]["tokenizer_revision"],
                           context_tokens=record.token_count, supervised_tokens=record.supervised_tokens)
        result.append(parse_labels(raw, ontology))
    return tuple(result)


def bind_message_evidence(raw: Any, mappings: tuple[SourceMapping, ...]) -> tuple[SkillEvidence, ...]:
    """message-evidence-v1 固定使用 source-1 零起点消息索引，不猜测文本。"""
    value = fields(raw, "message_evidence", "schema_version index_rule items")
    choice(value["schema_version"], "message_evidence.schema_version", {"message-evidence-v1"})
    choice(value["index_rule"], "message_evidence.index_rule", {"source_message_index_zero_v1"})
    lookup = {m.data_id: m for m in mappings}
    result = []
    for item in array(value["items"], "message_evidence.items"):
        item = fields(item, "message_evidence.item", "data_id trajectory_version skill_id source message_index")
        data_id = string(item["data_id"], "message_evidence.data_id")
        mapping = lookup.get(data_id)
        index = integer(item["message_index"], "message_evidence.message_index")
        if mapping is None or item["trajectory_version"] != mapping.trajectory_version or index >= len(mapping.messages):
            fail("message_evidence.item", "来源、版本或消息索引无效。")
        result.append(SkillEvidence.parse({k: item[k] for k in ("data_id", "trajectory_version", "skill_id", "source")} |
                      {"token_positions": list(mapping.messages[index].token_positions)}))
    return tuple(result)


def parse_mappings(raw: Any) -> tuple[SourceMapping, ...]:
    result = []
    for row in array(raw, "mappings"):
        v = fields(row, "mapping", "data_id trajectory_version source_hash messages")
        messages = []
        for m in array(v["messages"], "mapping.messages"):
            m = fields(m, "mapping.message", "message_index role char_start char_end token_positions")
            messages.append(MessageMapping(integer(m["message_index"], "mapping.message_index"),
                    choice(m["role"], "mapping.role", {"system", "user", "assistant", "tool"}),
                    integer(m["char_start"], "mapping.char_start"), integer(m["char_end"], "mapping.char_end"),
                    tuple(integer(p, "mapping.token_positions") for p in array(m["token_positions"], "mapping.token_positions"))))
        if [m.message_index for m in messages] != list(range(len(messages))) or any(m.char_start > m.char_end for m in messages):
            fail("mapping", "消息索引或区间无效。")
        result.append(SourceMapping(*(string(v[k], "mapping." + k) for k in ("data_id", "trajectory_version", "source_hash")), tuple(messages)))
    if len({m.data_id for m in result}) != len(result):
        fail("mappings.data_id", "重复映射。")
    return tuple(result)
