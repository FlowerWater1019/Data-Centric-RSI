"""输入检查服务。命令行和后续调用方使用同一资格规则。"""

from dataclasses import dataclass

from .contracts import ExperimentContract
from .data import TrajectoryRecord
from .settings import HardwareTarget
from .validation import ValidationError, fail, to_json_value


@dataclass(frozen=True)
class ValidationReport:
    records: int
    current_score_records: int
    stale_or_missing_score_ids: tuple[str, ...]
    purpose: str
    hardware_target: HardwareTarget
    status: str = "VALIDATED"
    validation_level: str = "structural"
    training_status: str = "NOT_RUN"

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)


def inspect_scores(
    contract: ExperimentContract,
    records: tuple[TrajectoryRecord, ...],
    *,
    require_current_scores: bool = False,
) -> ValidationReport:
    """检查已有分数。严格模式还要求所有可选候选具有当前分数。"""
    current = 0
    stale: list[str] = []
    for record in records:
        if record.scores is None and not (require_current_scores and record.is_eligible):
            continue
        try:
            record.require_current_scores(contract)
            current += 1
        except ValidationError:
            stale.append(record.data_id)
    if require_current_scores and stale:
        fail("scores", "分数缺失或版本过期：" + ", ".join(stale))
    return ValidationReport(len(records), current, tuple(stale), contract.purpose, contract.hardware)
