"""四项测量的独立编排。阶段成功可复用，未知失败不自动重跑。"""

from dataclasses import replace
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rsi.evaluation.acceptance import AcceptanceRule
from rsi.storage.artifacts import file_sha256, verify_training_artifact, write_json
from rsi.workflows.attempt_run import _result, load_attempt, verify_training_result
from rsi.models.backends import ArtifactRef, TrainingResult
from rsi.storage.execution_artifacts import complete_execution, implementation_identity, verify_execution
from rsi.evaluation.execution_evaluation import evaluate_execution, parse_execution_evaluation
from rsi.evaluation.execution_evaluation import verify_execution_evidence
from rsi.evaluation.agent_tasks import AgentTaskRunner
from rsi.evaluation.execution_evaluation import ExecutionEvaluation
from rsi.workflows.execution_plan import TrainingPlan
from rsi.curation.selection_io import SelectionInputs
from rsi.evaluation.fit_evaluation import FitEvaluation, SkillEvidence, evaluate_fit, parse_fit_evaluation
from rsi.models.hf_backend import HfRuntime, load_tokenizer
from rsi.storage.leases import run_lease
from rsi.models.model_resolution import ModelLocation, check_device
from rsi.evaluation.round_feedback import CostSummary, EvidenceRef, RoundFeedback, build_round_feedback, parse_round_feedback
from rsi.execution.sandbox import BubblewrapExecutor, SandboxLimits
from rsi.storage.state import StateStore
from rsi.evaluation.task_adapters import TaskSuiteV2
from rsi.models.training_mock import MockTokenScorer
from rsi.configuration.validation import digest, fail, integer, read_json, read_jsonl, to_json_value


def completed_candidate(root: Path) -> TrainingResult:
    plan, _, _ = load_attempt(root)
    with StateStore(root / "state.sqlite3", create=False, read_only=True) as state:
        event = next((e for e in state.snapshot(plan.run_id)["events"] if e["event_id"] == f"training:{plan.attempt_id}:completed"), None)
    if event is None:
        fail("validation.candidate", "训练尚未完成，禁止伪造候选模型。")
    result = _result(event["payload"]["result"])
    verify_training_result(result, plan)
    return result


def model_for_stage(root: Path, candidate: bool, device: str, expected_gpu_name: str | None) -> tuple[TrainingPlan, SelectionInputs, ModelLocation, dict[str, Any]]:
    plan, inputs, location = load_attempt(root)
    if candidate:
        result = completed_candidate(root)
        location = replace(location, model_ref=result.candidate_model, adapter_manifest=result.artifact)
    if plan.mode == "hf_local":
        hardware = check_device(device, expected_gpu_name)
        location.verify(load_tokenizer(location.base_model_path))
    else:
        if device != "cpu":
            fail("mock.device", "mock 仅允许 CPU。")
        hardware = {"device": "cpu", "mode": "mock"}
    return plan, inputs, location, hardware


def evaluate_fit_stage(attempt: str | Path, output: str | Path, *, candidate: bool, device: str,
                        expected_gpu_name: str | None = None, _locked: bool = False) -> FitEvaluation:
    """独立测量入口。内部编排已持有同一目录锁时显式复用锁。"""
    with nullcontext() if _locked else run_lease(Path(attempt).resolve()):
        return _evaluate_fit_stage(attempt, output, candidate=candidate, device=device, expected_gpu_name=expected_gpu_name)


def _evaluate_fit_stage(attempt: str | Path, output: str | Path, *, candidate: bool, device: str,
                         expected_gpu_name: str | None = None) -> FitEvaluation:
    root, stage = Path(attempt).resolve(), Path(output).resolve()
    plan, inputs, location, hardware = model_for_stage(root, candidate, device, expected_gpu_name)
    selected = tuple(next(r for r in inputs.records if r.data_id == item.data_id) for item in plan.batch.items)
    raw_evidence = read_json(root / "inputs/bundle/skill_evidence.json")
    selected_ids = {r.data_id for r in selected}
    evidence = tuple(SkillEvidence.parse(row) for row in raw_evidence if row["data_id"] in selected_ids)
    source = "measured" if plan.mode == "hf_local" else "synthetic_fixture"
    request = {"plan_hash": digest(plan), "ordered_items_hash": plan.ordered_items_hash,
               "model_location_hash": location.identity, "evidence_hash": digest(evidence),
               "implementation": implementation_identity(), "hardware": hardware, "source": source}
    if stage.exists():
        manifest = verify_execution(stage)
        if manifest["status"] != "COMPLETED":
            fail("fit.cache.status", "拟合缓存必须完整完成。")
        if read_json(stage / "request.json") != request:
            fail("fit.cache", "完整缓存的模型、输入、代码或设备已变化。")
        cached = parse_fit_evaluation(read_json(stage / "fit.json"))
        class SavedScorer:
            def score_tokens(self, record: Any) -> Any:
                row = next((r for r in cached.records if r.data_id == record.data_id), None)
                if row is None:
                    fail("fit.cache.records", "缓存缺少冻结轨迹。")
                return row.token_scores
        expected = evaluate_fit(location.model_ref, selected, SavedScorer(), inputs.ontology.ids, evidence, source=source)
        if cached != expected:
            fail("fit.cache", "缓存模型、输入身份或技能摘要与逐 token 证据不一致。")
        return cached
    stage.mkdir(parents=True)
    write_json(stage / "request.json", request)
    with StateStore(root / "state.sqlite3", create=False) as state:
        training_done = any(e["event_id"] == f"training:{plan.attempt_id}:completed" for e in state.snapshot(plan.run_id)["events"])
        write_json(stage / "measurement.json", {"started_at": datetime.now(timezone.utc).isoformat(),
                   "timing": "after_training" if training_done else "before_training", "source": source})
        reservation = "fit_" + digest(str(stage))[:16]
        state.reserve(plan.run_id, reservation, "scoring_tokens", sum(r.token_count for r in selected))

        class ChargedScorer:
            def __init__(self, scorer: Any) -> None:
                self.scorer = scorer

            def score_tokens(self, record: Any) -> Any:
                write_json(stage / "calls" / (digest(record.data_id) + ".intent.json"), {"data_id": record.data_id, "model": to_json_value(location.model_ref)})
                value = self.scorer.score_tokens(record)
                state.consume(plan.run_id, reservation + ":" + record.data_id, reservation, record.token_count)
                write_json(stage / "calls" / (digest(record.data_id) + ".result.json"), to_json_value(value))
                return value

        try:
            if plan.mode == "hf_local":
                with HfRuntime(location.base_model_path, device, location.load.precision, location.adapter_path,
                               attention_implementation=location.load.attention_implementation) as scorer:
                    result = evaluate_fit(location.model_ref, selected, ChargedScorer(scorer), inputs.ontology.ids, evidence, source=source)
            else:
                result = evaluate_fit(location.model_ref, selected, ChargedScorer(MockTokenScorer(candidate)), inputs.ontology.ids, evidence, source=source)
        except BaseException as exc:
            write_json(stage / "failure.json", {"error": str(exc), "unknown_consumption_possible": True})
            raise
        state.release(plan.run_id, reservation)
        write_json(stage / "costs.json", state.snapshot(plan.run_id))
    write_json(stage / "fit.json", result.to_dict())
    complete_execution(stage, "COMPLETED")
    return result


def evaluate_execution_stage(attempt: str | Path, suite_path: str | Path, output: str | Path, *, candidate: bool,
                              device: str, call_budget: int, sandbox_path: str | Path | None = None,
                              expected_gpu_name: str | None = None, _locked: bool = False,
                              agent_runner: AgentTaskRunner | None = None) -> ExecutionEvaluation:
    """串行执行同一训练尝试的评估；已完成阶段必须通过完整身份核验。"""
    with nullcontext() if _locked else run_lease(Path(attempt).resolve()):
        return _evaluate_execution_stage(attempt, suite_path, output, candidate=candidate, device=device, call_budget=call_budget,
                                         sandbox_path=sandbox_path, expected_gpu_name=expected_gpu_name, agent_runner=agent_runner)


def _evaluate_execution_stage(attempt: str | Path, suite_path: str | Path, output: str | Path, *, candidate: bool,
                              device: str, call_budget: int, sandbox_path: str | Path | None = None,
                              expected_gpu_name: str | None = None, agent_runner: AgentTaskRunner | None = None) -> ExecutionEvaluation:
    root, stage = Path(attempt).resolve(), Path(output).resolve()
    integer(call_budget, "execution.call_budget", 1)
    plan, inputs, location, hardware = model_for_stage(root, candidate, device, expected_gpu_name)
    suite = TaskSuiteV2.parse(read_json(suite_path), inputs.ontology)
    if suite.purpose != "feedback":
        fail("suite.purpose", "此入口仅用于反馈，final_test 使用 evaluate-benchmark。")
    agent_tasks = [t for t in suite.tasks if t.adapter_id == "agent_environment_v1"]
    if agent_tasks and (agent_runner is None or len(agent_tasks) != len(suite.tasks)):
        fail("execution.agent_runner", "Agent 套件需要显式接入 AgentTaskRunner；流程入口不混用代码生成与 Agent 后端。")
    executor = BubblewrapExecutor(SandboxLimits.parse(read_json(sandbox_path))) if sandbox_path else None
    source = "measured" if plan.mode == "hf_local" else "synthetic_fixture"
    request = {"plan_hash": digest(plan), "suite_hash": digest(suite), "model_location_hash": location.identity,
               "implementation": implementation_identity(), "hardware": hardware, "source": source,
               "sandbox_hash": digest(read_json(sandbox_path)) if sandbox_path else None, "call_budget": call_budget,
               "agent_runner": agent_runner.identity if agent_runner else None}
    if stage.exists():
        manifest = verify_execution(stage)
        if read_json(stage / "stage_request.json") != request:
            fail("execution.cache", "模型、任务、环境、代码或预算不匹配。")
        cached = parse_execution_evaluation(read_json(stage / "evaluation.json"))
        if (manifest["status"] != cached.status or cached.model != location.model_ref or cached.source != source or
            cached.suite_hash != digest(suite) or cached.generation_hash != digest(suite.generation)):
            fail("execution.cache", "缓存结果与阶段请求的模型、任务、规则或来源不一致。")
        verify_execution_evidence(stage, cached, suite, location.model_ref, source, call_budget, executor, agent_runner)
        return cached
    with StateStore(root / "state.sqlite3", create=False) as state:
        reservation = "execution_" + digest(str(stage))[:16]
        if call_budget < 1:
            fail("execution.call_budget", "预算至少为 1。")
        state.reserve(plan.run_id, reservation, "evaluation_calls", min(call_budget, len(suite.tasks)))
        sources = read_jsonl(root / "inputs/bundle/sources.jsonl")
        if agent_tasks:
            result = evaluate_execution(location.model_ref, suite, None, stage, call_budget, source=source,
                      state=state, run_id=plan.run_id, reservation=reservation, records=inputs.records, sources=sources,
                      stage_request=request, agent_runner=agent_runner)
        elif plan.mode == "hf_local":
            tokenizer = load_tokenizer(location.base_model_path)
            with HfRuntime(location.base_model_path, device, location.load.precision, location.adapter_path,
                           attention_implementation=location.load.attention_implementation) as runtime:
                def generate(prompt: str, limit: int, seed: int) -> str:
                    runtime.torch.manual_seed(seed)
                    return runtime.generate_code(tokenizer, prompt, limit, suite.generation.max_context_length)
                result = evaluate_execution(location.model_ref, suite, generate, stage, call_budget, source=source,
                        state=state, run_id=plan.run_id, reservation=reservation, executor=executor, records=inputs.records, sources=sources, stage_request=request)
        else:
            result = evaluate_execution(location.model_ref, suite, lambda prompt, limit, seed: "def solve(x, y):\n    return x + y\n", stage, call_budget,
                       source=source, state=state, run_id=plan.run_id, reservation=reservation, executor=executor, records=inputs.records, sources=sources, stage_request=request)
        state.release(plan.run_id, reservation)
        # 阶段清单保持不可变。后续账本快照保存在阶段目录之外。
        write_json(stage.parent / (stage.name + ".costs.json"), state.snapshot(plan.run_id))
    return result


def run_dual_validation(attempt: str | Path, suite_path: str | Path, *, device: str,
                         call_budget: int, rule_path: str | Path | None = None,
                         sandbox_path: str | Path | None = None, expected_gpu_name: str | None = None,
                         parent_runner: AgentTaskRunner | None = None, candidate_runner: AgentTaskRunner | None = None) -> RoundFeedback:
    root = Path(attempt).resolve()
    with run_lease(root):
        plan, _, _ = load_attempt(root)
        completed_candidate(root)
        stage = root / "validation"
        fits, executions = [], []
        for candidate, name in ((False, "parent"), (True, "candidate")):
            fits.append(evaluate_fit_stage(root, stage / ("fit-" + name), candidate=candidate, device=device, expected_gpu_name=expected_gpu_name, _locked=True))
            executions.append(evaluate_execution_stage(root, suite_path, stage / ("execution-" + name), candidate=candidate, device=device,
                               call_budget=call_budget, sandbox_path=sandbox_path, expected_gpu_name=expected_gpu_name, _locked=True,
                               agent_runner=candidate_runner if candidate else parent_runner))
        evidence = []
        for kind, value, path in (("fit_parent", fits[0], stage / "fit-parent/fit.json"),
                ("fit_candidate", fits[1], stage / "fit-candidate/fit.json"),
                ("execution_parent", executions[0], stage / "execution-parent/evaluation.json"),
                ("execution_candidate", executions[1], stage / "execution-candidate/evaluation.json")):
            evidence.append(EvidenceRef(kind, ArtifactRef(str(path), file_sha256(path)), digest(value.model),
                                       value.ordered_batch_hash if kind.startswith("fit") else value.suite_hash))
        with StateStore(root / "state.sqlite3", create=False, read_only=True) as state:
            snapshot = state.snapshot(plan.run_id)
        budget = snapshot["budgets"]
        costs = CostSummary(*(budget[k]["used"] for k in ("training_tokens", "scoring_tokens", "evaluation_calls")),
                            any(e.status == "PARTIAL" for e in executions))
        feedback = build_round_feedback(feedback_id="feedback_" + digest(plan), round_id="round_" + plan.attempt_id,
                   attempt_id=plan.attempt_id, batch_id=plan.batch.context_id + ":" + plan.batch.fingerprint, plan_hash=digest(plan),
                   fit_parent=fits[0], fit_candidate=fits[1], execution_parent=executions[0], execution_candidate=executions[1],
                   data_kind=plan.data_kind, label_mode=plan.label_mode, costs=costs, evidence=tuple(evidence),
                   rule=AcceptanceRule.parse(read_json(rule_path)) if rule_path else None)
        path = stage / "round_feedback.json"
        if path.exists():
            if parse_round_feedback(read_json(path)) != feedback:
                fail("validation", "已发布反馈与本次结果不同。")
        else:
            write_json(path, feedback.to_dict())
        report_text = ("# 单次训练双路验证\n\n来源：" + feedback.source +
                "\n\n训练数据性质：" + feedback.data_kind + "\n\n标签性质：" + feedback.label_mode +
                "\n\n拟合 NLL 差值：" + str(feedback.fit.overall.delta_nll) +
                "\n\n整条轨迹和技能片段差值见 round_feedback.json 的 fit.per_record 与 fit.by_skill。\n\n执行状态：" + feedback.status +
                "\n\n机械验收：" + feedback.acceptance.status + "。控制器 NOT_RUN，模型晋级 NOT_APPLIED。\n")
        report_path = stage / "validation_report.md"
        if report_path.exists():
            if report_path.read_text(encoding="utf-8") != report_text:
                fail("validation.report", "已有报告与证据不一致，拒绝覆盖。")
        else:
            with report_path.open("x", encoding="utf-8") as report_file:
                report_file.write(report_text)
        if feedback.status == "COMPLETED":
            if (root / "completion_manifest.json").exists():
                verify_execution(root)
            else:
                complete_execution(root, "VALIDATION_COMPLETED")
        return feedback
