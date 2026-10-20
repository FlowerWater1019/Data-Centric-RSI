"""实验配置与不可变模型引用。硬件声明不代表设备已经核实。"""

from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from rsi.configuration.settings import AcceptanceConfig, HardwareTarget, ScoreVersions, TrainingConfig
from rsi.configuration.validation import array, boolean, choice, fail, integer, number, obj, string, strings, to_json_value


@dataclass(frozen=True)
class ModelRef:
    checkpoint_id: str
    base_revision: str
    adapter_revision: str
    tokenizer_revision: str
    template_revision: str

    @classmethod
    def parse(cls, raw: Any, path: str = "model") -> "ModelRef":
        fields = set(cls.__dataclass_fields__)
        value = obj(raw, path, fields)
        return cls(**{key: string(value[key], f"{path}.{key}") for key in fields})


@dataclass(frozen=True)
class Preference:
    support: tuple[float, float]
    preferred: tuple[float, float]

    @classmethod
    def parse(cls, raw: Any, path: str) -> "Preference":
        value = obj(raw, path, {"support", "preferred"})
        pairs = []
        for key in ("support", "preferred"):
            items = array(value[key], f"{path}.{key}")
            if len(items) != 2:
                fail(path, "区间必须包含两个端点。")
            pairs.append(tuple(number(x, path, 0, 1) for x in items))
        (a, d), (b, c) = pairs
        if not a < b <= c < d:
            fail(path, "区间必须满足 support_left < preferred_left <= preferred_right < support_right。")
        return cls(pairs[0], pairs[1])


@dataclass(frozen=True)
class FixedStrategy:
    strategy_id: str
    nll: Preference
    entropy: Preference
    weights: Mapping[str, float]
    skill_preferences: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", MappingProxyType(dict(self.weights)))
        object.__setattr__(self, "skill_preferences", MappingProxyType(dict(self.skill_preferences)))

    @classmethod
    def parse(cls, raw: Any, skills: tuple[str, ...]) -> "FixedStrategy":
        value = obj(raw, "fixed_strategy", {"strategy_id", "nll", "entropy", "weights", "skill_preferences"})
        weights = obj(value["weights"], "weights", {"nll", "entropy", "skill"})
        weights = {k: number(v, f"weights.{k}", 0, 1) for k, v in weights.items()}
        if abs(sum(weights.values()) - 1) > 1e-9:
            fail("weights", "权重总和必须为 1。")
        preferences = obj(value["skill_preferences"], "skill_preferences", set(), set(skills))
        preferences = {k: number(v, f"skill_preferences.{k}", 0, 1) for k, v in preferences.items()}
        return cls(string(value["strategy_id"], "strategy_id"), Preference.parse(value["nll"], "nll"),
                   Preference.parse(value["entropy"], "entropy"), weights, preferences)


@dataclass(frozen=True)
class ExperimentContract:
    contract_id: str
    purpose: str
    seed: int
    model: ModelRef
    skills: tuple[str, ...]
    token_budget: int
    allowed_shortfall: int
    min_skill_shares: Mapping[str, float]
    scoring: ScoreVersions
    fixed_strategy: FixedStrategy
    hardware: HardwareTarget
    training: TrainingConfig
    acceptance: AcceptanceConfig
    budgets: Mapping[str, int]
    allow_feedback_overlap: bool
    packing_rule: str = "greedy_v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_skill_shares", MappingProxyType(dict(self.min_skill_shares)))
        object.__setattr__(self, "budgets", MappingProxyType(dict(self.budgets)))

    @classmethod
    def parse(cls, raw: Any) -> "ExperimentContract":
        value = obj(raw, "contract", {"schema_version", "contract_id", "purpose", "seed", "model", "skills",
                    "selection", "scoring", "fixed_strategy", "hardware", "training", "acceptance", "budgets",
                    "allow_feedback_overlap"})
        schema = choice(value["schema_version"], "schema_version", {"1.0", "1.1"})
        purpose = choice(value["purpose"], "purpose", {"smoke", "draft", "research"})
        skills = strings(value["skills"], "skills", True)
        fields = {"supervised_token_budget", "allowed_shortfall", "min_skill_shares"}
        selection = obj(value["selection"], "selection", fields | ({"packing_rule"} if schema == "1.1" else set()))
        packing = choice(selection.get("packing_rule", "greedy_v1"), "selection.packing_rule",
                         {"greedy_v1", "exact_ranked_v1"})
        budget = integer(selection["supervised_token_budget"], "supervised_token_budget", 1)
        shortfall = integer(selection["allowed_shortfall"], "allowed_shortfall")
        if shortfall >= budget:
            fail("allowed_shortfall", "允许偏差必须小于预算。")
        if packing == "exact_ranked_v1" and shortfall != 0:
            fail("allowed_shortfall", "精确组批要求偏差为 0。")
        shares = obj(selection["min_skill_shares"], "min_skill_shares", set(), set(skills))
        shares = {k: number(v, f"min_skill_shares.{k}", 0, 1) for k, v in shares.items()}
        scoring = obj(value["scoring"], "scoring", {"rule_version", "reference_version", "cdf_version"})
        for key, item in scoring.items():
            string(item, f"scoring.{key}")
        hardware = obj(value["hardware"], "hardware", {"accelerator", "gpu_count", "memory_gb_per_gpu", "evidence"})
        string(hardware["accelerator"], "hardware.accelerator")
        integer(hardware["gpu_count"], "gpu_count", 1)
        integer(hardware["memory_gb_per_gpu"], "memory_gb_per_gpu", 1)
        choice(hardware["evidence"], "hardware.evidence", {"user_reported", "verified"})
        training = obj(value["training"], "training", {"backend", "world_size", "precision", "learning_rate",
                       "lora_rank", "max_sequence_length", "per_device_batch_size", "gradient_accumulation_steps",
                       "optimizer_state", "scheduler_state"})
        choice(training["backend"], "training.backend", {"unimplemented", "hf_lora"})
        world = integer(training["world_size"], "world_size", 1)
        if world > hardware["gpu_count"]:
            fail("world_size", "训练进程数超过目标 GPU 数。")
        choice(training["precision"], "precision", {"bf16", "fp32"})
        for key in ("lora_rank", "max_sequence_length", "per_device_batch_size", "gradient_accumulation_steps"):
            if training[key] is not None:
                integer(training[key], f"training.{key}", 1)
        if training["learning_rate"] is not None:
            rate = number(training["learning_rate"], "learning_rate")
            if rate == 0:
                fail("learning_rate", "学习率必须大于 0。")
        for key in ("optimizer_state", "scheduler_state"):
            choice(training[key], key, {"fresh_per_attempt"})
        acceptance = obj(value["acceptance"], "acceptance", {"primary_metric", "minimum_gain", "retention_tolerance",
                         "sample_size", "statistical_rule", "max_retests"})
        for key in ("primary_metric", "statistical_rule"):
            if acceptance[key] is not None:
                string(acceptance[key], f"acceptance.{key}")
        for key in ("minimum_gain", "retention_tolerance"):
            if acceptance[key] is not None:
                number(acceptance[key], f"acceptance.{key}", 0, 1)
        for key in ("sample_size", "max_retests"):
            if acceptance[key] is not None:
                integer(acceptance[key], f"acceptance.{key}", 1 if key == "sample_size" else 0)
        budgets = obj(value["budgets"], "budgets", {"training_tokens", "scoring_tokens", "evaluation_calls"})
        budgets = {k: integer(v, f"budgets.{k}") for k, v in budgets.items()}
        result = cls(string(value["contract_id"], "contract_id"), purpose, integer(value["seed"], "seed"),
                     ModelRef.parse(value["model"]), skills, budget, shortfall, shares, ScoreVersions(**scoring),
                     FixedStrategy.parse(value["fixed_strategy"], skills), HardwareTarget(**hardware), TrainingConfig(**training),
                     AcceptanceConfig(**acceptance), budgets, boolean(value["allow_feedback_overlap"], "allow_feedback_overlap"), packing)
        if purpose == "research":
            result.require_research_ready()
        return result

    def require_research_ready(self) -> None:
        missing = [f"training.{k}" for k, v in to_json_value(self.training).items() if v is None]
        missing += [f"acceptance.{k}" for k, v in to_json_value(self.acceptance).items() if v is None]
        if missing:
            fail("research", "正式实验配置未完成：" + ", ".join(missing))
        fail("research", "正式研究验收流程尚未实现。当前只支持工程验证。")
