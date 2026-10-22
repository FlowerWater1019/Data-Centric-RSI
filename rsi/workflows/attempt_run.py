"""独立单次训练。准备产物可复制，实际失败现场与成本必须保留。"""

from dataclasses import replace
from pathlib import Path
import shutil
from typing import Any

from rsi.storage.artifacts import file_sha256, verify_training_artifact, write_json
from rsi.models.backends import ArtifactRef, TrainingRequest, TrainingResult, TrainingUse
from rsi.storage.checkpoints import StepJournal, TrainingPaused, request_identity
from rsi.configuration.contracts import ModelRef
from rsi.storage.execution_artifacts import verify_execution, implementation_snapshot
from rsi.workflows.execution_plan import TrainingPlan, freeze_training_plan, parse_batch
from rsi.models.hf_backend import HfLoraBackend, load_tokenizer
from rsi.storage.leases import run_lease
from rsi.models.model_resolution import ModelLocation, check_device
from rsi.curation.selection_io import SelectionInputs, load_inputs, verify_manifest
from rsi.curation.policy import adapt_policy
from rsi.configuration.training_config import contract_dict
from rsi.datasets.training_inputs import TrainingProvenance
from rsi.storage.state import StateStore
from rsi.configuration.validation import digest, fail, integer, read_json, string, to_json_value
from rsi.datasets.labels import fields


def load_attempt(root: Path) -> tuple[TrainingPlan, SelectionInputs, ModelLocation]:
    source, selection = root / "inputs/bundle", root / "inputs/selection"
    compiled = verify_execution(source)
    if compiled["status"] != "COMPILED_AND_SCORED":
        fail("compiled.status", "训练输入必须是已完成编译和评分的快照。")
    manifest = verify_manifest(selection)
    inputs = load_inputs(source / "config.json", source / "trajectories.jsonl", source / "labels.jsonl",
                         source / "ontology.json", source / "policy.json", source / "batch_history.json")
    request = fields(read_json(root / "request.json"), "attempt.request", "run_id attempt_id")
    provenance = TrainingProvenance.parse(read_json(source / "provenance.json"))
    location = ModelLocation.parse(read_json(root / "model_location.json"), root)
    if location.model_ref != inputs.contract.model:
        fail("model_location.model_ref", "父模型必须与冻结选择相同。")
    plan = freeze_training_plan(inputs, read_json(selection / "selection_report.json"), parse_batch(read_json(selection / "batch.json")),
             manifest["content_id"], read_json(root / "settings.json"), location.identity, request["run_id"], request["attempt_id"],
             data_kind=provenance.data_kind)
    if plan.mode != provenance.execution_mode:
        fail("compiled.provenance.execution_mode", "训练模式必须与实际编译分词模式相同，不能把 mock token 用于真实训练。")
    plan = replace(plan, input_hashes={**plan.input_hashes, "compiled_manifest_id": compiled["content_id"]})
    if plan.load != location.load:
        fail("model_location.load", "冻结训练加载配置与模型定位不一致。")
    if plan.to_dict() != read_json(root / "plan.json"):
        fail("training_plan", "请求、顺序、模型定位或输入快照已变化。")
    with StateStore(root / "state.sqlite3", create=False, read_only=True) as state:
        snapshot = state.snapshot(plan.run_id)
    expected_snapshot = digest({"mode": "controlled", "contract": contract_dict(adapt_policy(inputs.contract, inputs.policy)),
                                "dataset": [r.raw for r in inputs.records]})
    event = next((e for e in snapshot["events"] if e["event_id"] == "training_plan:" + plan.attempt_id), None)
    if snapshot["snapshot_hash"] != expected_snapshot or event is None or event["payload"] != {
        "schema_version": "single-training-plan-v1", "attempt_id": plan.attempt_id, "batch": plan.batch.to_dict(), "plan_hash": digest(plan)}:
        fail("training_plan.state", "账本快照或已登记单批计划不一致。准备可能未完成。")
    return plan, inputs, location


def prepare_training_attempt(bundle: str | Path, selection: str | Path, selection_id: str,
                             model_location: str | Path, settings_path: str | Path,
                             output: str | Path, run_id: str, attempt_id: str) -> TrainingPlan:
    source, selected, root = Path(bundle).resolve(), Path(selection).resolve(), Path(output).resolve()
    compiled_manifest = verify_execution(source)
    if compiled_manifest["status"] != "COMPILED_AND_SCORED":
        fail("compiled.status", "准备训练要求已完成编译和评分的快照。")
    selection_manifest = verify_manifest(selected)
    if selection_manifest["content_id"] != selection_id:
        fail("selection.content_id", "选择产物身份不匹配。")
    inputs = load_inputs(source / "config.json", source / "trajectories.jsonl", source / "labels.jsonl",
                         source / "ontology.json", source / "policy.json", source / "batch_history.json")
    location_path = Path(model_location).resolve()
    location = ModelLocation.parse(read_json(location_path), location_path.parent)
    if location.model_ref != inputs.contract.model:
        fail("model_location.model_ref", "父模型不属于本次选择。")
    settings = read_json(settings_path)
    provenance = TrainingProvenance.parse(read_json(source / "provenance.json"))
    plan = freeze_training_plan(inputs, read_json(selected / "selection_report.json"), parse_batch(read_json(selected / "batch.json")),
               selection_id, settings, location.identity, run_id, attempt_id,
               data_kind=provenance.data_kind)
    if plan.mode != provenance.execution_mode:
        fail("compiled.provenance.execution_mode", "训练模式必须与实际编译分词模式相同，不能把 mock token 用于真实训练。")
    plan = replace(plan, input_hashes={**plan.input_hashes, "compiled_manifest_id": compiled_manifest["content_id"]})
    if plan.load != location.load:
        fail("model_location.load", "训练加载配置与模型定位不一致。")
    if root.exists():
        existing, _, _ = load_attempt(root)
        if existing != plan:
            fail("output", "已有尝试与请求不同。")
        return existing
    if root.is_relative_to(source) or root.is_relative_to(selected):
        fail("output", "执行输出不能放在输入快照内部。")
    if plan.mode == "hf_local":
        location.verify(load_tokenizer(location.base_model_path))
    root.mkdir(parents=True)
    with run_lease(root):
        shutil.copytree(source, root / "inputs/bundle")
        shutil.copytree(selected, root / "inputs/selection")
        if verify_execution(root / "inputs/bundle") != compiled_manifest or verify_manifest(root / "inputs/selection") != selection_manifest:
            fail("inputs", "准备期间输入发生变化。")
        write_json(root / "request.json", {"run_id": run_id, "attempt_id": attempt_id})
        write_json(root / "settings.json", settings)
        write_json(root / "model_location.json", location.to_dict())
        write_json(root / "plan.json", plan.to_dict())
        with StateStore(root / "state.sqlite3") as state:
            state.initialize(run_id, "controlled", contract_dict(adapt_policy(inputs.contract, inputs.policy)), [r.raw for r in inputs.records])
            # 普通评分已在编译中执行。导入已有消耗，避免为同一轮重置评分预算。
            scoring_used = integer(read_json(source / "costs.json")["budgets"]["scoring_tokens"]["used"], "compiled.costs.scoring_tokens.used", 1)
            expected_scoring = sum(r.token_count for r in inputs.records if r.role in {"candidate", "reference", "score_probe"})
            if scoring_used != expected_scoring:
                fail("compiled.costs", "编译评分消耗与实际输入不一致。")
            state.reserve(run_id, "compiled_scoring", "scoring_tokens", scoring_used)
            state.consume(run_id, "compiled_scoring:" + compiled_manifest["content_id"], "compiled_scoring", scoring_used)
            state.release(run_id, "compiled_scoring")
            state.prepare_training_plan(run_id, attempt_id, plan.batch, digest(plan))
    return plan


def _result(raw: dict[str, Any]) -> TrainingResult:
    value = fields(raw, "training_result", "candidate_model artifact actual_supervised_tokens diagnostic_artifact")
    def artifact(raw_ref: Any) -> ArtifactRef:
        ref = fields(raw_ref, "training_result.artifact", "path sha256")
        return ArtifactRef(string(ref["path"], "training_result.artifact.path"), string(ref["sha256"], "training_result.artifact.sha256"))
    return TrainingResult(ModelRef.parse(value["candidate_model"]), artifact(value["artifact"]),
                          integer(value["actual_supervised_tokens"], "training_result.actual_supervised_tokens", 1),
                          artifact(value["diagnostic_artifact"]) if value["diagnostic_artifact"] is not None else None)


def verify_training_result(result: TrainingResult, plan: TrainingPlan) -> None:
    """校验外部后端结果与冻结父模型。错误结果不能登记完成。"""
    _result(to_json_value(result))
    parent, candidate = plan.batch.model, result.candidate_model
    if any(getattr(parent, key) != getattr(candidate, key) for key in ("base_revision", "tokenizer_revision", "template_revision")):
        fail("training_result.candidate_model", "候选必须沿用冻结父模型的基座、分词器和模板。")
    if candidate.adapter_revision != result.artifact.sha256 or candidate == parent:
        fail("training_result.candidate_model.adapter_revision", "候选引用必须绑定返回的 adapter 清单。")
    if result.actual_supervised_tokens != plan.batch.supervised_tokens:
        fail("training_result.actual_supervised_tokens", "实际监督量与冻结批次不一致。")
    verify_training_artifact(result.artifact)
    if result.diagnostic_artifact is not None and file_sha256(result.diagnostic_artifact.path) != result.diagnostic_artifact.sha256:
        fail("training_result.diagnostic_artifact", "训练诊断文件哈希不匹配。")


def run_training_attempt(output: str | Path, device: str, *, resume_artifact: ArtifactRef | None = None,
                         stop_after_step: int | None = None, expected_gpu_name: str | None = None) -> TrainingResult:
    root = Path(output).resolve()
    with run_lease(root):
        plan, inputs, location = load_attempt(root)
        if stop_after_step is not None and integer(stop_after_step, "stop_after_step", 1) > len(plan.batch.items):
            fail("stop_after_step", "暂停步骤超过冻结批次。")
        selected = tuple(next(r for r in inputs.records if r.data_id == item.data_id) for item in plan.batch.items)
        request = TrainingRequest(plan.run_id, plan.attempt_id, plan.input_hashes["contract_hash"], plan.batch.model,
                   plan.batch, plan.training, resume_artifact, plan.lora, plan.load, location.identity)
        if plan.mode == "hf_local":
            hardware = check_device(device, expected_gpu_name)
            location.verify(load_tokenizer(location.base_model_path))
        else:
            if device != "cpu":
                fail("mock.device", "mock 仅允许 CPU。")
            hardware = {"device": "cpu", "mode": "mock"}
        with StateStore(root / "state.sqlite3", create=False) as state:
            events = state.snapshot(plan.run_id)["events"]
            completed = next((e["payload"]["result"] for e in events if e["event_id"] == f"training:{plan.attempt_id}:completed"), None)
            if completed:
                result = _result(completed)
                verify_training_result(result, plan)
                if resume_artifact is not None:
                    fail("resume", "已完成的尝试不能恢复旧检查点。")
                state.release(plan.run_id, plan.attempt_id)
                if not (root / "training_costs.json").exists():
                    write_json(root / "training_costs.json", state.snapshot(plan.run_id))
                return result
            started = any(e["event_id"] == f"training:{plan.attempt_id}:started" for e in events)
            if resume_artifact:
                _, step = StepJournal(root / "training", request_identity(request, list(selected), device)).validate_resume(resume_artifact)
                state.check_training_resume(plan.run_id, plan.attempt_id, plan.attempt_id, plan.batch.fingerprint, step)
                state.training_boundary(plan.run_id, plan.attempt_id, step, "resumed", resume_artifact.path, resume_artifact.sha256)
            elif started:
                fail("attempt", "已有未完成步骤；必须明确恢复完整检查点，不能盲目重跑。")
            if not (root / "environment.json").exists():
                write_json(root / "environment.json", {"hardware": hardware, "mode": plan.mode,
                           "source": "measured" if plan.mode == "hf_local" else "synthetic_fixture", "software": implementation_snapshot(),
                           "model_config": read_json(Path(location.base_model_path) / "config.json") if plan.mode == "hf_local" else None})
            elif read_json(root / "environment.json")["hardware"] != hardware:
                fail("resume.device", "执行设备已变化。")
            if not started:
                state.reserve(plan.run_id, plan.attempt_id, "training_tokens", plan.batch.supervised_tokens)
                state.training_event(plan.run_id, plan.attempt_id, "started", batch_fingerprint=plan.batch.fingerprint)

            def on_step(step: int, record: Any, count: int) -> None:
                state.consume(plan.run_id, f"{plan.attempt_id}:step:{step}", plan.attempt_id, count,
                              TrainingUse(plan.attempt_id, step, record.data_id, record.version, record.train_mask_version))

            try:
                if plan.mode == "mock":
                    from rsi.models.training_mock import MockTrainingBackend
                    backend = MockTrainingBackend(root / "training", device, selected, on_step, stop_after_step)
                else:
                    backend = HfLoraBackend(location.base_model_path, location.model_ref, inputs.records, root / "training",
                              device, on_step, stop_after_step, location.adapter_manifest)
                result = backend.train(request)
                verify_training_result(result, plan)
                state.training_event(plan.run_id, plan.attempt_id, "completed", batch_fingerprint=plan.batch.fingerprint, result=result)
            except TrainingPaused as exc:
                state.training_boundary(plan.run_id, plan.attempt_id, read_json(exc.artifact.path)["step"], "paused", exc.artifact.path, exc.artifact.sha256)
                raise
            except BaseException as exc:
                state.training_event(plan.run_id, plan.attempt_id, "failed", batch_fingerprint=plan.batch.fingerprint, error=str(exc) or type(exc).__name__)
                state.release(plan.run_id, plan.attempt_id)
                write_json(root / "failures" / ("training-" + digest(str(exc)) + ".json"), {"error": str(exc), "state": state.snapshot(plan.run_id)})
                raise
            state.release(plan.run_id, plan.attempt_id)
            write_json(root / "training_costs.json", state.snapshot(plan.run_id))
            return result


def resume_training_attempt(output: str | Path, device: str, checkpoint: ArtifactRef,
                            *, expected_gpu_name: str | None = None) -> TrainingResult:
    return run_training_attempt(output, device, resume_artifact=checkpoint, expected_gpu_name=expected_gpu_name)
