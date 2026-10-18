"""计算受控对照计划。此模块不读写数据库，也不启动训练。"""

from dataclasses import dataclass

from .selection import BatchManifest, FixedSelector, RandomSelector, SelectionContext, build_batch
from .settings import HardwareTarget, TrainingConfig
from .validation import fail, to_json_value


@dataclass(frozen=True)
class ComparisonArms:
    random: BatchManifest
    fixed: BatchManifest


@dataclass(frozen=True)
class ComparisonPlan:
    run_id: str
    comparison_id: str
    context_id: str
    contract_hash: str
    dataset_hash: str
    training_config: TrainingConfig
    hardware_target: HardwareTarget
    same_batch: bool
    arms: ComparisonArms
    schema_version: str = "1.0"
    training_status: str = "NOT_RUN"
    metrics: None = None

    def to_dict(self) -> dict[str, object]:
        return to_json_value(self)


def build_comparison(context: SelectionContext, run_id: str, comparison_id: str) -> ComparisonPlan:
    """生成等监督量的两个分支。预算不足或组批失败时抛出 ValidationError。"""
    random_batch = build_batch(context, RandomSelector())
    fixed_batch = build_batch(context, FixedSelector())
    total = random_batch.supervised_tokens + fixed_batch.supervised_tokens
    if total > context.contract.budgets["training_tokens"]:
        fail("budgets.training_tokens", "总预算不足以覆盖两个分支的计划监督量。")
    if random_batch.supervised_tokens != fixed_batch.supervised_tokens:
        fail("comparison", "两组实际计划监督量不同，不能登记为等 token 对照。")
    return ComparisonPlan(
        run_id=run_id,
        comparison_id=comparison_id,
        context_id=context.context_id,
        contract_hash=context.contract_hash,
        dataset_hash=context.dataset_hash,
        training_config=context.contract.training,
        hardware_target=context.contract.hardware,
        same_batch=random_batch.fingerprint == fixed_batch.fingerprint,
        arms=ComparisonArms(random=random_batch, fixed=fixed_batch),
    )
