"""训练与评估的字段接口。执行实现位于各后端模块。"""

from dataclasses import dataclass
from typing import Protocol

from rsi.configuration.contracts import ModelRef
from rsi.curation.selection import BatchManifest
from rsi.configuration.settings import TrainingConfig


@dataclass(frozen=True)
class ArtifactRef:
    """产物位置与内容校验值。执行适配器必须核验实际文件。"""

    path: str
    sha256: str


@dataclass(frozen=True)
class TrainingRequest:
    """一次训练请求。新尝试不复用优化器；同次恢复传入完整状态产物。"""
    run_id: str
    attempt_id: str
    contract_hash: str
    starting_model: ModelRef
    batch: BatchManifest
    training_config: TrainingConfig
    resume_artifact: ArtifactRef | None = None


@dataclass(frozen=True)
class TrainingResult:
    """训练完成后的产物和实耗。返回该类型不代表模型已获验收。"""
    candidate_model: ModelRef
    artifact: ArtifactRef
    actual_supervised_tokens: int
    diagnostic_artifact: ArtifactRef | None


@dataclass(frozen=True)
class TrainingUse:
    """一次完成反向传播的数据使用。与 token 消耗在同一事务中记录。"""

    attempt_id: str
    step: int
    data_id: str
    version: str
    mask_version: str


@dataclass(frozen=True)
class EvaluationRequest:
    model: ModelRef
    task_manifest: ArtifactRef
    seed: int


@dataclass(frozen=True)
class MetricResult:
    """聚合指标。定义与聚合单位必须说明分母或样本单位。"""

    name: str
    value: float
    sample_count: int
    definition: str


@dataclass(frozen=True)
class EvaluationResult:
    metrics: tuple[MetricResult, ...]
    evidence: ArtifactRef


class TrainingBackend(Protocol):
    """执行训练并返回完整产物。失败必须抛出异常，不能返回成功结果。"""
    def train(self, request: TrainingRequest) -> TrainingResult: ...


class EvaluationBackend(Protocol):
    """执行独立任务评估。指标必须能追溯到逐任务证据产物。"""

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult: ...


class UnimplementedTrainingBackend:
    def train(self, request: TrainingRequest) -> TrainingResult:
        raise NotImplementedError("真实训练后端尚未实现。没有执行 SFT，也没有生成候选模型。")
