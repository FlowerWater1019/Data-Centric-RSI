"""冻结选择输出为独立训练计划。只验证成员，不重新选择。"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from rsi.configuration.contracts import ExperimentContract, ModelRef
from rsi.models.model_resolution import LoraConfig, LoadConfig
from rsi.curation.policy_selection import validate_selection
from rsi.curation.selection import BatchItem, BatchManifest, check_task_type_shares
from rsi.curation.selection_io import SelectionInputs
from rsi.configuration.settings import TrainingConfig
from rsi.configuration.validation import array, choice, digest, fail, integer, number, obj, string, to_json_value
from rsi.datasets.labels import fields


def parse_batch(raw: Any) -> BatchManifest:
    v = obj(raw, "batch", set(BatchManifest.__dataclass_fields__))
    items = []
    for item in array(v["items"], "batch.items", True):
        item = obj(item, "batch.item", set(BatchItem.__dataclass_fields__))
        items.append(BatchItem(*(string(item[k], "batch.item." + k) for k in ("data_id", "version", "mask_version", "content_hash")),
                               integer(item["repetitions"], "batch.item.repetitions", 1), integer(item["supervised_tokens"], "batch.item.supervised_tokens", 1)))
    if not isinstance(v["skill_shares"], dict):
        fail("batch.skill_shares", "必须是映射。")
    return BatchManifest(string(v["selector"], "batch.selector"), string(v["context_id"], "batch.context_id"),
           ModelRef.parse(v["model"]), integer(v["seed"], "batch.seed"), tuple(items),
           integer(v["supervised_tokens"], "batch.supervised_tokens", 1), integer(v["context_tokens"], "batch.context_tokens", 1),
           {string(k, "batch.skill_shares"): number(x, "batch.skill_shares." + k, 0, 1) for k, x in v["skill_shares"].items()},
           string(v["fingerprint"], "batch.fingerprint"), choice(v["training_status"], "batch.training_status", {"NOT_RUN"}))


@dataclass(frozen=True)
class TrainingPlan:
    schema_version: str
    plan_id: str
    run_id: str
    attempt_id: str
    batch: BatchManifest
    ordered_items_hash: str
    selection_manifest_id: str
    input_hashes: Mapping[str, str]
    training: TrainingConfig
    lora: LoraConfig
    load: LoadConfig
    model_location_hash: str
    mode: str
    data_kind: str
    label_mode: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_hashes", MappingProxyType(dict(self.input_hashes)))

    def to_dict(self) -> dict[str, Any]:
        return to_json_value(self)


def freeze_training_plan(inputs: SelectionInputs, report: dict[str, Any], batch: BatchManifest,
                         selection_id: str, settings: dict[str, Any], location_hash: str,
                         run_id: str, attempt_id: str, *, data_kind: str) -> TrainingPlan:
    """核验第一阶段所有身份及批次断言。此函数不调用 rank/build_batch。"""
    settings = fields(settings, "attempt_settings", "schema_version mode training lora load")
    choice(settings["schema_version"], "attempt_settings.schema_version", {"training-attempt-v1"})
    mode = choice(settings["mode"], "attempt_settings.mode", {"mock", "hf_local"})
    if mode == "mock" and inputs.contract.purpose != "smoke":
        fail("attempt_settings.mode", "mock 只能用于 smoke。")
    context = validate_selection(inputs.contract, inputs.records, inputs.labels, inputs.ontology, inputs.policy,
                                 inputs.history, allow_synthetic_labels=inputs.contract.purpose == "smoke")
    if report.get("schema_version") != "selection-report-v1" or report.get("status") != "SELECTED":
        fail("selection.status", "需要完整 SELECTED 报告。")
    if report.get("identities") != dict(context.identities) or report.get("context_id") != context.context_id:
        fail("selection.identities", "选择输入或上下文已变化。")
    if report.get("derived_contract") != to_json_value(context.contract) or report.get("batch") != batch.to_dict():
        fail("selection.batch", "派生合同或批次与报告不一致。")
    if batch.model != inputs.contract.model or batch.context_id != context.context_id or batch.seed != inputs.policy.execution.seed:
        fail("batch.model", "批次模型、上下文或种子不匹配。")
    records = {r.data_id: r for r in context.records}
    chosen = []
    for item in batch.items:
        r = records.get(item.data_id)
        if r is None or item != BatchItem(r.data_id, r.version, r.train_mask_version, r.content_hash, 1, r.supervised_tokens):
            fail("batch.items", "成员并非原选择上下文中的合格完整轨迹。")
        if r.raw["tokenization"]["status"] != "verified":
            fail("batch.tokenization", "不能将第一阶段 demo token 当作实际编译数据。")
        chosen.append(r)
    for key in ("data_id", "duplicate_group", "content_hash"):
        if len({getattr(r, key) for r in chosen}) != len(chosen):
            fail("batch." + key, "训练批次存在重复。")
    if batch.fingerprint != digest(sorted(batch.items, key=lambda item: item.data_id)):
        fail("batch.fingerprint", "成员身份不一致。")
    total = sum(r.supervised_tokens for r in chosen)
    if total != batch.supervised_tokens or not context.contract.token_budget - context.contract.allowed_shortfall <= total <= context.contract.token_budget:
        fail("batch.supervised_tokens", "监督预算不一致。")
    if batch.context_tokens != sum(r.token_count for r in chosen):
        fail("batch.context_tokens", "输入总量不一致。")
    shares = {k: sum(r.supervised_tokens for r in chosen if k in r.skills) / total for k in inputs.ontology.ids}
    if dict(batch.skill_shares) != shares or any(shares[k] < minimum for k, minimum in context.contract.min_skill_shares.items()):
        fail("batch.skill_shares", "技能占比或配额不一致。")
    label_lookup = {x.data_id: x for x in inputs.labels}
    task_shares = {k: sum(r.supervised_tokens for r in chosen if label_lookup[r.data_id].task.task_type == k) / total
                   for k in inputs.policy.eligibility.allowed_task_types}
    check_task_type_shares(task_shares, inputs.policy.batch_constraints.min_task_type_token_shares,
                           inputs.policy.batch_constraints.max_task_type_token_shares)
    # 复用原合同的训练字段校验，避免扩散第二套参数规则。
    from rsi.configuration.training_config import contract_dict
    training_contract = contract_dict(inputs.contract)
    training_contract["training"] = settings["training"]
    training = ExperimentContract.parse(training_contract).training
    if (training.backend, training.world_size, training.per_device_batch_size, training.gradient_accumulation_steps) != ("hf_lora", 1, 1, 1):
        fail("training", "只支持单设备、单轨迹、无梯度累积的 hf_lora 执行语义。")
    if training.learning_rate is None or training.lora_rank is None or training.max_sequence_length is None:
        fail("training", "训练参数必须完整。")
    if any(r.token_count > training.max_sequence_length for r in chosen):
        fail("training.max_sequence_length", "批次存在超长轨迹。")
    load = LoadConfig.parse(settings["load"])
    if load.precision != training.precision:
        fail("load.precision", "训练与加载精度不同。")
    label_modes = {x.provenance.label_mode for x in inputs.labels}
    return TrainingPlan("training-plan-v1", "plan_" + digest({"run": run_id, "attempt": attempt_id}),
                string(run_id, "run_id"), string(attempt_id, "attempt_id"), batch, digest(batch.items),
                string(selection_id, "selection_manifest_id"), context.identities, training,
                LoraConfig.parse(settings["lora"]), load, string(location_hash, "model_location_hash"), mode,
                choice(data_kind, "data_kind", {"synthetic", "real_public", "private"}),
                next(iter(label_modes)) if len(label_modes) == 1 else "mixed")
