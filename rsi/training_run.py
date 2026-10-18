"""执行已冻结的单卡对照计划。新尝试和完整步骤恢复共用同一后端。"""

import os
from pathlib import Path
from typing import Any

from .artifacts import file_sha256, model_manifest, verify_training_artifact, write_json
from .backends import ArtifactRef, TrainingRequest, TrainingUse
from .checkpoints import StepJournal, TrainingPaused, request_identity
from .comparison import build_comparison
from .data import TrajectoryRecord
from .devices import require_uuid_binding, verify_device
from .diagnostics import summarize_metrics
from .hf_backend import HfLoraBackend, HfRuntime, load_dependencies
from .leases import run_lease
from .scoring_run import score_and_record, progress
from .state import StateStore
from .validation import digest, fail, read_json, to_json_value


def finish_pilot(output: Path, model_path: str | Path, device: str, *,
                 stop_after_step: int | None = None, resume_artifact: ArtifactRef | None = None,
                 resume_arm: str | None = None) -> dict[str, Any]:
    """继续冻结的工程对照。只在已有计划上训练，不重算分数或改变数据。"""
    with run_lease(output):
        return _finish_pilot(output, model_path, device, stop_after_step, resume_artifact, resume_arm)


def _finish_pilot(output: Path, model_path: str | Path, device: str, stop_after_step: int | None,
                  resume_artifact: ArtifactRef | None, resume_arm: str | None) -> dict[str, Any]:
    torch, _, _ = load_dependencies()
    environment = read_json(output / "environment.json")
    binding = require_uuid_binding(device, os.environ.get("CUDA_VISIBLE_DEVICES"))
    if torch.cuda.device_count() != 1:
        fail("device", "恢复必须只看到一个 CUDA 设备。")
    hardware = verify_device(torch.cuda.get_device_properties(device), environment["hardware"]["name"], binding)
    if hardware != environment["hardware"]:
        fail("resume", "恢复设备与原运行不同。")
    code = read_json(output / "code_identity.json")
    implementation = [{"name": p.name, "sha256": file_sha256(p)} for p in sorted(Path(__file__).parent.glob("*.py"))]
    if digest(implementation) != code["content_id"]:
        fail("resume", "源码已变化。请保留原目录与原运行。")
    identity = model_manifest(model_path)
    if identity["content_id"] != read_json(output / "model_identity.json")["content_id"]:
        fail("resume", "本地模型内容已变化。")
    if (output / "report.json").exists():
        fail("resume", "该工程运行已经完成。")
    base_metrics = read_json(output / "baseline_scores.json")
    results = {}
    with StateStore(output / "state.sqlite3", create=False) as state:
        context = state.context("training")
        contract = context.contract
        plan = build_comparison(context, "training", "random_vs_fixed")
        feedback = tuple(r for r in context.records if r.role == "feedback_diagnostic")
        for arm, batch in (("random", plan.arms.random), ("fixed", plan.arms.fixed)):
            events = state.snapshot("training")["events"]
            finished = next((e["payload"]["result"] for e in events if e["event_id"] == f"training:{arm}:completed"), None)
            if finished is None:
                progress("train_" + arm)
                checkpoint = resume_artifact if arm == resume_arm else None
                started = any(e["event_id"] == f"training:{arm}:started" for e in events)
                if started and checkpoint is None:
                    fail("resume", "已有未完成尝试。必须明确提供该组的完整检查点。")
                if checkpoint is not None:
                    verify_training_artifact(checkpoint)
                    state.check_training_resume("training", arm, arm, batch.fingerprint, read_json(checkpoint.path)["step"])
                else:
                    state.reserve("training", arm, "training_tokens", batch.supervised_tokens)
                    state.training_event("training", arm, "started", batch_fingerprint=batch.fingerprint)

                def on_step(step: int, record: TrajectoryRecord, count: int) -> None:
                    state.consume("training", f"{arm}:step:{step}", arm, count,
                                  TrainingUse(arm, step, record.data_id, record.version, record.train_mask_version))

                backend = HfLoraBackend(model_path, contract.model, context.records, output / arm, device,
                                        on_step, stop_after_step if arm == "random" else None)
                request = TrainingRequest("training", arm, context.contract_hash, contract.model, batch,
                                          contract.training, checkpoint)
                if checkpoint is not None:
                    selected = [next(r for r in context.records if r.data_id == item.data_id) for item in batch.items]
                    _, step = StepJournal(output / arm, request_identity(request, selected, device)).validate_resume(checkpoint)
                    state.training_boundary("training", arm, step, "resumed", checkpoint.path, checkpoint.sha256)
                try:
                    result = backend.train(request)
                    verify_training_artifact(result.artifact)
                    state.training_event("training", arm, "completed", batch_fingerprint=batch.fingerprint, result=result)
                    finished = to_json_value(result)
                except TrainingPaused as exc:
                    # 保留剩余预算。恢复继续使用原尝试编号和原步骤编号。
                    state.training_boundary("training", arm, read_json(exc.artifact.path)["step"], "paused",
                                            exc.artifact.path, exc.artifact.sha256)
                    raise
                except BaseException as exc:
                    state.training_event("training", arm, "failed", batch_fingerprint=batch.fingerprint,
                                         error=str(exc) or type(exc).__name__)
                    state.release("training", arm)
                    raise
                state.release("training", arm)
            artifact = ArtifactRef(**finished["artifact"])
            verify_training_artifact(artifact)
            after_path = output / f"after_{arm}.json"
            if not after_path.exists():
                if any(e["event_id"] == "reserve:after_" + arm for e in state.snapshot("scoring")["events"]):
                    fail("resume", "诊断调用已开始，但完整产物缺失。拒绝自动重复调用。")
                progress("reload_and_diagnose_" + arm)
                with HfRuntime(model_path, device, contract.training.precision, output / arm / "adapter") as runtime:
                    after = score_and_record(runtime, feedback, state, "after_" + arm, output / "score_calls")
                write_json(after_path, to_json_value(after))
            results[arm] = {"result": finished, "diagnostics_after_reload": read_json(after_path),
                            "diagnostic_model": finished["candidate_model"]}
        final_state, score_state = state.snapshot("training"), state.snapshot("scoring")
    report = {"schema_version": "pilot-report-2", "purpose": "engineering_validation", "data_kind": code.get("data_kind", "synthetic"),
              "training_status": final_state["training_status"], "same_batch": plan.same_batch,
              "model_identity": identity["content_id"], "hardware": hardware, "code_identity": code["content_id"],
              "baseline_diagnostics": {r.data_id: base_metrics[r.data_id] for r in feedback},
              "diagnostic_summary": {"baseline": summarize_metrics({r.data_id: base_metrics[r.data_id] for r in feedback}, contract.skills),
                                     **{arm: summarize_metrics(result["diagnostics_after_reload"], contract.skills) for arm, result in results.items()}},
              "arms": results, "training_state": final_state, "scoring_state": score_state,
              "task_execution": "NOT_RUN", "acceptance": "NOT_RUN",
              "resume_validation": "COMPLETE_STEP_RESUMED" if resume_artifact else "NOT_RUN",
              "distributed_validation": "NOT_RUN", "selection_improvement_evidence": "NOT_ESTABLISHED",
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    write_json(output / "report.json", report)
    progress("completed")
    return {"training_status": report["training_status"], "purpose": report["purpose"],
            "report": str(output / "report.json"), "same_batch": plan.same_batch, "task_execution": "NOT_RUN"}


def resume_pilot(output_path: str | Path, model_path: str | Path, device: str,
                 arm: str, checkpoint_path: str | Path, checkpoint_sha256: str) -> dict[str, Any]:
    """恢复用户明确指定的完整检查点。清单哈希必须由原暂停输出提供。"""
    return finish_pilot(Path(output_path).resolve(), model_path, device,
                        resume_arm=arm, resume_artifact=ArtifactRef(str(checkpoint_path), checkpoint_sha256))
