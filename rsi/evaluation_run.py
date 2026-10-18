"""独立工程任务评估。基线和两组候选共用冻结任务与验收规则。"""

from dataclasses import replace
import os
from pathlib import Path
from typing import Any

from .acceptance import AcceptanceRule, TaskOutcome, decide
from .artifacts import file_sha256, model_manifest, verify_training_artifact, write_json
from .backends import ArtifactRef, EvaluationRequest
from .coding_evaluation import CodingEvaluationBackend
from .coding_tasks import TaskSuite
from .python_tasks import PythonTaskSuite
from .isolated_evaluation import IsolatedEvaluationBackend
from .sandbox import BubblewrapExecutor, SandboxLimits
from .contracts import ModelRef
from .devices import require_uuid_binding, verify_device
from .hf_backend import HfRuntime, load_dependencies, load_tokenizer
from .scoring_run import progress
from .state import StateStore
from .validation import fail, integer, obj, read_json, to_json_value


def run_evaluation(pilot_path: str | Path, tasks_path: str | Path, rule_path: str | Path,
                   model_path: str | Path, output_path: str | Path, device: str,
                   sandbox_path: str | Path | None = None, max_context_length: int | None = None) -> dict[str, Any]:
    """评估已完成 pilot。新输出目录保存任务、规则、逐任务结果和调用成本。"""
    pilot, output = Path(pilot_path).resolve(), Path(output_path).resolve()
    report = read_json(pilot / "report.json")
    if report.get("training_status") != "COMPLETED" or report.get("purpose") != "engineering_validation":
        fail("pilot", "需要已完成的工程训练报告。")
    tasks_raw, rule_raw = read_json(tasks_path), read_json(rule_path)
    isolated = tasks_raw.get("schema_version") == "python-tasks-1"
    suite = PythonTaskSuite.parse(tasks_raw) if isolated else TaskSuite.parse(tasks_raw)
    rule = AcceptanceRule.parse(rule_raw)
    executor, preflight = None, None
    if isolated:
        if sandbox_path is None:
            fail("sandbox", "真实 Python 任务必须提供 --sandbox-config，不能退回宿主执行。")
        executor = BubblewrapExecutor(SandboxLimits.parse(read_json(sandbox_path)))
        preflight = executor.preflight()
    elif sandbox_path is not None:
        fail("sandbox", "受限表达式任务不使用通用隔离配置。")
    binding = require_uuid_binding(device, os.environ.get("CUDA_VISIBLE_DEVICES"))
    torch, _, _ = load_dependencies()
    if torch.cuda.device_count() != 1:
        fail("device", "评估必须只看到一个 CUDA 设备。")
    hardware = verify_device(torch.cuda.get_device_properties(device), report["hardware"]["name"], binding)
    identity = model_manifest(model_path)
    if identity["content_id"] != report["model_identity"]:
        fail("model", "评估基座与训练基座不同。")
    with StateStore(pilot / "state.sqlite3", create=False, read_only=True) as source_state:
        context = source_state.context("training")
        training_state = source_state.snapshot("training")
        if training_state["training_status"] != "COMPLETED":
            fail("pilot", "报告与训练账本不一致。")
    forbidden_ids = {r.data_id for r in context.records}
    forbidden_splits = {r.raw["split_group"] for r in context.records}
    forbidden_duplicates = {r.raw["duplicate_group"] for r in context.records}
    forbidden_texts = {r.raw["text"] for r in context.records}
    for task in suite.tasks:
        if (task.task_id in forbidden_ids or task.split_group in forbidden_splits
                or task.duplicate_group in forbidden_duplicates or task.prompt in forbidden_texts):
            fail("tasks", "评估任务与训练、评分或最终测试数据的标识重叠。")
    models = {"baseline": context.contract.model}
    for arm in ("random", "fixed"):
        result = report["arms"][arm]["result"]
        recorded = next((event["payload"]["result"] for event in training_state["events"]
                         if event["event_id"] == f"training:{arm}:completed"), None)
        if result != recorded:
            fail("pilot", "报告候选与账本中的完成记录不同。")
        artifact = ArtifactRef(**obj(result["artifact"], "artifact", {"path", "sha256"}))
        verify_training_artifact(artifact)
        if Path(artifact.path).resolve() != (pilot / arm / "manifest.json").resolve():
            fail("artifact", "候选产物必须属于该 pilot 目录。")
        model = ModelRef(**result["candidate_model"])
        if model.adapter_revision != artifact.sha256 or replace(model, checkpoint_id=context.contract.model.checkpoint_id,
                adapter_revision="none") != context.contract.model:
            fail("model", "候选模型引用与产物或基座不一致。")
        models[arm] = model
    output.mkdir(parents=True, exist_ok=False)
    task_ref = write_json(output / "tasks.json", tasks_raw)
    rule_ref = write_json(output / "acceptance_rule.json", rule_raw)
    write_json(output / "provenance.json", {"pilot_report_sha256": file_sha256(pilot / "report.json"),
               "hardware": hardware, "model_identity": identity["content_id"],
               "code": {p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}})
    context_length = context.contract.training.max_sequence_length if max_context_length is None else integer(max_context_length, "max_context_length", 2)
    write_json(output / "generation_config.json", {"max_context_length": context_length, "max_new_tokens": suite.max_new_tokens,
               "seed": context.contract.seed, "sampling": "greedy"})
    if isolated:
        write_json(output / "sandbox.json", read_json(sandbox_path))
        write_json(output / "preflight.json", preflight)
    tokenizer = load_tokenizer(model_path)
    contract_raw = read_json(pilot / "contract.json")
    contract_raw["contract_id"] += "_coding_evaluation"
    contract_raw["budgets"]["evaluation_calls"] = len(suite.tasks) * len(models)
    results = {}
    with StateStore(output / "state.sqlite3") as state:
        state.initialize("evaluation", "controlled", contract_raw, [r.raw for r in context.records])
        for label, model in models.items():
            progress("evaluate_" + label)
            state.reserve("evaluation", label, "evaluation_calls", len(suite.tasks))
            def on_call(task_id: str) -> None:
                # 在调用前登记尝试。失败调用仍占用预算，不作为成功评估。
                state.consume("evaluation", label + ":" + task_id, label, 1)
            try:
                torch.manual_seed(context.contract.seed)
                adapter = None if label == "baseline" else pilot / label / "adapter"
                with HfRuntime(model_path, device, context.contract.training.precision, adapter) as runtime:
                    def generate(prompt: str, limit: int) -> str:
                        return runtime.generate_code(tokenizer, prompt, limit, context_length)
                    backend = (IsolatedEvaluationBackend(generate, output / label, on_call, executor, preflight)
                               if isolated else CodingEvaluationBackend(generate, output / label, on_call))
                    results[label] = backend.evaluate(EvaluationRequest(model, task_ref, context.contract.seed))
            except BaseException as exc:
                state.record_error("evaluation", label, str(exc) or type(exc).__name__)
                write_json(output / "failure.json", {"status": "execution_error", "stage": label,
                           "error": str(exc) or type(exc).__name__, "costs": state.snapshot("evaluation")["budgets"]})
                raise
            finally:
                state.release("evaluation", label)
        snapshot = state.snapshot("evaluation")
    def outcomes(label: str) -> tuple[TaskOutcome, ...]:
        result = results[label]
        if file_sha256(result.evidence.path) != result.evidence.sha256:
            fail("evaluation", "评估证据已经变化。")
        evidence = read_json(result.evidence.path)
        if evidence["task_manifest_sha256"] != task_ref.sha256:
            fail("evaluation", "基线与候选任务清单不同。")
        return tuple(TaskOutcome.parse(row["outcome"]) for row in evidence["tasks"])
    baseline = outcomes("baseline")
    decisions = {arm: to_json_value(decide(baseline, outcomes(arm), rule)) for arm in ("random", "fixed")}
    selection_comparison = to_json_value(decide(outcomes("random"), outcomes("fixed"), rule))
    summary = {"schema_version": "evaluation-report-1", "purpose": "engineering_validation",
               "task_execution": "COMPLETED", "data_kind": "real_public" if isolated else "synthetic", "hardware": hardware,
               "executor": "bubblewrap_seccomp_v1" if isolated else "bounded_integer_expression_v1",
               "tasks": to_json_value(task_ref), "acceptance_rule": to_json_value(rule_ref),
               "evaluations": to_json_value(results), "decisions": decisions, "state": snapshot,
               "fixed_vs_random": selection_comparison,
               "comparison_scope": "fixed_candidate_pool_model_budget_and_seed",
               "uncertainty_scope": "per_comparison_two_groups; not adjusted across multiple comparisons or seeds",
               "model_promotion": "NOT_APPLIED", "selection_improvement_evidence": "NOT_ESTABLISHED"}
    write_json(output / "report.json", summary)
    return {"task_execution": "COMPLETED", "report": str(output / "report.json"),
            "decisions": decisions, "model_promotion": "NOT_APPLIED"}
