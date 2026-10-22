"""旧模块导入兼容。新代码使用功能子包，兼容入口共享真实模块对象。"""

from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import find_spec, spec_from_loader
import sys
from types import CodeType, MappingProxyType, ModuleType
from typing import Sequence


LEGACY_MODULES = MappingProxyType({
    "rsi.acceptance": "rsi.evaluation.acceptance",
    "rsi.artifacts": "rsi.storage.artifacts",
    "rsi.backends": "rsi.models.backends",
    "rsi.bounded_python": "rsi.evaluation.bounded_python",
    "rsi.checkpoints": "rsi.storage.checkpoints",
    "rsi.cli": "rsi.commands.cli",
    "rsi.coding_evaluation": "rsi.evaluation.coding_evaluation",
    "rsi.coding_tasks": "rsi.evaluation.coding_tasks",
    "rsi.comparison": "rsi.curation.comparison",
    "rsi.contracts": "rsi.configuration.contracts",
    "rsi.data": "rsi.datasets.data",
    "rsi.data_compile": "rsi.datasets.data_compile",
    "rsi.dataset_import": "rsi.datasets.dataset_import",
    "rsi.devices": "rsi.models.devices",
    "rsi.diagnostics": "rsi.models.diagnostics",
    "rsi.dispatch": "rsi.curation.dispatch",
    "rsi.evaluation_run": "rsi.workflows.evaluation_run",
    "rsi.feedback": "rsi.curation.feedback",
    "rsi.hf_backend": "rsi.models.hf_backend",
    "rsi.history": "rsi.curation.history",
    "rsi.inspection": "rsi.curation.inspection",
    "rsi.isolated_evaluation": "rsi.evaluation.isolated_evaluation",
    "rsi.labels": "rsi.datasets.labels",
    "rsi.leases": "rsi.storage.leases",
    "rsi.mbpp": "rsi.datasets.mbpp",
    "rsi.model_inputs": "rsi.models.model_inputs",
    "rsi.packing": "rsi.curation.packing",
    "rsi.pilot": "rsi.workflows.pilot",
    "rsi.policy": "rsi.curation.policy",
    "rsi.policy_selection": "rsi.curation.policy_selection",
    "rsi.python_diagnostics": "rsi.datasets.python_diagnostics",
    "rsi.python_tasks": "rsi.evaluation.python_tasks",
    "rsi.real_run": "rsi.workflows.real_run",
    "rsi.reporting": "rsi.workflows.reporting",
    "rsi.sandbox": "rsi.execution.sandbox",
    "rsi.schema": "rsi.storage.schema",
    "rsi.scoring": "rsi.models.scoring",
    "rsi.scoring_run": "rsi.workflows.scoring_run",
    "rsi.selection": "rsi.curation.selection",
    "rsi.selection_cli": "rsi.commands.selection_cli",
    "rsi.selection_demo": "rsi.curation.selection_demo",
    "rsi.selection_io": "rsi.curation.selection_io",
    "rsi.settings": "rsi.configuration.settings",
    "rsi.state": "rsi.storage.state",
    "rsi.tokenization": "rsi.datasets.tokenization",
    "rsi.training_run": "rsi.workflows.training_run",
    "rsi.validation": "rsi.configuration.validation",
    'rsi.attempt_run': 'rsi.workflows.attempt_run',
    'rsi.benchmark_run': 'rsi.workflows.benchmark_run',
    'rsi.execution_artifacts': 'rsi.storage.execution_artifacts',
    'rsi.execution_evaluation': 'rsi.evaluation.execution_evaluation',
    'rsi.execution_plan': 'rsi.workflows.execution_plan',
    'rsi.fit_evaluation': 'rsi.evaluation.fit_evaluation',
    'rsi.model_resolution': 'rsi.models.model_resolution',
    'rsi.round_feedback': 'rsi.evaluation.round_feedback',
    'rsi.task_adapters': 'rsi.evaluation.task_adapters',
    'rsi.training_cli': 'rsi.commands.training_cli',
    'rsi.training_compile': 'rsi.workflows.training_compile',
    'rsi.training_config': 'rsi.configuration.training_config',
    'rsi.training_demo': 'rsi.workflows.training_demo',
    'rsi.training_inputs': 'rsi.datasets.training_inputs',
    'rsi.training_mock': 'rsi.models.training_mock',
    'rsi.validation_run': 'rsi.workflows.validation_run',
})


class _LegacyLoader(Loader):
    def __init__(self, canonical: str, canonical_spec: ModuleSpec) -> None:
        self.canonical = canonical
        self.canonical_spec = canonical_spec

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        return import_module(self.canonical)

    def exec_module(self, module: ModuleType) -> None:
        # 导入器会先设置旧路径的 spec。恢复真实身份，供反射和脚本入口使用。
        module.__spec__ = self.canonical_spec

    def get_code(self, fullname: str) -> CodeType | None:
        # 保留 python -m rsi.cli 等旧脚本入口。
        loader = self.canonical_spec.loader
        if loader is None or not hasattr(loader, "get_code"):
            raise ImportError(f"模块没有可执行源码：{self.canonical}")
        return loader.get_code(self.canonical)


class _LegacyFinder(MetaPathFinder):
    def find_spec(self, fullname: str, path: Sequence[str] | None = None,
                  target: ModuleType | None = None) -> ModuleSpec | None:
        canonical = LEGACY_MODULES.get(fullname)
        if canonical is None:
            return None
        canonical_spec = find_spec(canonical)
        if canonical_spec is None:
            raise ModuleNotFoundError(canonical)
        return spec_from_loader(fullname, _LegacyLoader(canonical, canonical_spec),
                                origin=canonical_spec.origin)


def install_legacy_imports() -> None:
    """仅安装本包明确列出的旧路径。不加载模型依赖或沙箱启动器。"""
    if not any(isinstance(finder, _LegacyFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _LegacyFinder())
