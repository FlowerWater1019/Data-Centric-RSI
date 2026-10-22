"""独立最终测试。此入口不构造反馈，也不调用验收规则。"""

from pathlib import Path
from typing import Any

from rsi.storage.artifacts import write_json
from rsi.configuration.contracts import ExperimentContract
from rsi.datasets.data import validate_dataset
from rsi.storage.execution_artifacts import complete_execution, verify_execution
from rsi.evaluation.execution_evaluation import evaluate_execution
from rsi.evaluation.agent_tasks import AgentTaskRunner
from rsi.models.hf_backend import HfRuntime, load_tokenizer
from rsi.datasets.labels import SkillOntology
from rsi.storage.leases import run_lease
from rsi.models.model_resolution import ModelLocation, check_device
from rsi.execution.sandbox import BubblewrapExecutor, SandboxLimits
from rsi.storage.state import StateStore
from rsi.evaluation.task_adapters import TaskSuiteV2, verify_suite_isolation
from rsi.configuration.validation import choice, fail, integer, read_json, read_jsonl, to_json_value


def evaluate_benchmark(bundle: str | Path, model_location: str | Path, suite_path: str | Path,
                       output: str | Path, *, mode: str, device: str, call_budget: int,
                       feedback_suite_path: str | Path,
                       sandbox_path: str | Path | None = None, expected_gpu_name: str | None = None,
                       agent_runner: AgentTaskRunner | None = None) -> dict[str, Any]:
    """输入编译快照只用于隔离与成本配置，不要求训练尝试或比较产物。"""
    choice(mode, "mode", {"mock", "hf_local"})
    integer(call_budget, "benchmark.call_budget", 1)
    source, root = Path(bundle).resolve(), Path(output).resolve()
    manifest = verify_execution(source)
    if manifest["status"] != "COMPILED_AND_SCORED":
        fail("benchmark.bundle.status", "隔离输入必须是已完成编译和评分的快照。")
    location_file = Path(model_location).resolve()
    location = ModelLocation.parse(read_json(location_file), location_file.parent)
    ontology = SkillOntology.parse(read_json(source / "ontology.json"))
    suite = TaskSuiteV2.parse(read_json(suite_path), ontology)
    if suite.purpose != "final_test":
        fail("benchmark.suite.purpose", "最终测试入口要求 final_test。")
    feedback_suite = TaskSuiteV2.parse(read_json(feedback_suite_path), ontology)
    if feedback_suite.purpose != "feedback":
        fail("benchmark.feedback_suite.purpose", "隔离参照必须为 feedback 套件。")
    verify_suite_isolation(suite, feedback_suite)
    agent_tasks = [t for t in suite.tasks if t.adapter_id == "agent_environment_v1"]
    if agent_tasks and (agent_runner is None or len(agent_tasks) != len(suite.tasks)):
        fail("benchmark.agent_runner", "Agent 最终测试需要显式接入 AgentTaskRunner，不能退回代码生成。")
    config = read_json(source / "config.json")
    config["model"] = to_json_value(location.model_ref)
    contract = ExperimentContract.parse(config)
    records = validate_dataset(read_jsonl(source / "trajectories.jsonl"), contract)
    if mode == "mock" and (device != "cpu" or contract.purpose != "smoke"):
        fail("benchmark.mode", "mock 仅允许 smoke 与 CPU。")
    hardware = check_device(device, expected_gpu_name) if mode == "hf_local" else {"device": "cpu", "mode": "mock"}
    if mode == "hf_local":
        location.verify(load_tokenizer(location.base_model_path))
    if root.exists():
        fail("benchmark.output", "输出已存在；最终测试不自动重复执行。")
    if call_budget < 1:
        fail("benchmark.call_budget", "至少为 1。")
    root.mkdir(parents=True)
    with run_lease(root), StateStore(root / "state.sqlite3") as state:
        state.initialize("benchmark", "controlled", config, [r.raw for r in records])
        write_json(root / "model_location.json", location.to_dict())
        write_json(root / "feedback_suite_for_isolation.json", feedback_suite.to_dict())
        write_json(root / "config.json", config)
        state.reserve("benchmark", "execution", "evaluation_calls", min(call_budget, len(suite.tasks)))
        from rsi.configuration.validation import digest
        write_json(root / "environment.json", {"hardware": hardware, "mode": mode, "compiled_manifest_id": manifest["content_id"],
                   "feedback_suite_hash_for_isolation": digest(feedback_suite)})
        executor = BubblewrapExecutor(SandboxLimits.parse(read_json(sandbox_path))) if sandbox_path else None
        kwargs = dict(source="measured" if mode == "hf_local" else "synthetic_fixture", state=state,
                      run_id="benchmark", executor=executor, records=records, sources=read_jsonl(source / "sources.jsonl"))
        try:
            if agent_tasks:
                result = evaluate_execution(location.model_ref, suite, None, root / "execution", call_budget,
                                             agent_runner=agent_runner, **kwargs)
            elif mode == "hf_local":
                tokenizer = load_tokenizer(location.base_model_path)
                with HfRuntime(location.base_model_path, device, location.load.precision, location.adapter_path,
                               attention_implementation=location.load.attention_implementation) as runtime:
                    def generate(prompt: str, limit: int, seed: int) -> str:
                        runtime.torch.manual_seed(seed)
                        return runtime.generate_code(tokenizer, prompt, limit, suite.generation.max_context_length)
                    result = evaluate_execution(location.model_ref, suite, generate, root / "execution", call_budget, **kwargs)
            else:
                result = evaluate_execution(location.model_ref, suite, lambda p, n, s: "def solve(x, y):\n    return x + y\n",
                                            root / "execution", call_budget, **kwargs)
        except BaseException as exc:
            write_json(root / "failure.json", {"error": str(exc), "state": state.snapshot("benchmark")})
            raise
        state.release("benchmark", "execution")
        write_json(root / "costs.json", state.snapshot("benchmark"))
        report = {"schema_version": "benchmark-report-v1", "status": result.status, "purpose": "final_test",
                  "model": to_json_value(location.model_ref), "source": result.source, "execution": result.to_dict(),
                  "feedback_status": "NOT_APPLICABLE", "promotion_status": "NOT_APPLIED"}
        write_json(root / "benchmark_report.json", report)
        (root / "benchmark_report.md").write_text("# 独立最终测试\n\n来源：" + result.source +
            "\n\n完成状态：" + result.status + "\n\n通过率：" + str(result.overall.pass_rate) +
            "\n\n结果仅用于最终报告。不得用于筛选、反馈或模型验收。\n", encoding="utf-8")
        complete_execution(root, result.status)
        return report
