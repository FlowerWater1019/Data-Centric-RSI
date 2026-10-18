"""经过配置解析器校验的字段类型。构造类型本身不替代输入校验。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreVersions:
    rule_version: str
    reference_version: str
    cdf_version: str


@dataclass(frozen=True)
class HardwareTarget:
    accelerator: str
    gpu_count: int
    memory_gb_per_gpu: int
    evidence: str


@dataclass(frozen=True)
class TrainingConfig:
    """空参数表示待配置。后端接入后仍需检查执行前置条件。"""

    backend: str
    world_size: int
    precision: str
    learning_rate: float | None
    lora_rank: int | None
    max_sequence_length: int | None
    per_device_batch_size: int | None
    gradient_accumulation_steps: int | None
    optimizer_state: str
    scheduler_state: str


@dataclass(frozen=True)
class AcceptanceConfig:
    primary_metric: str | None
    minimum_gain: float | None
    retention_tolerance: float | None
    sample_size: int | None
    statistical_rule: str | None
    max_retests: int | None
