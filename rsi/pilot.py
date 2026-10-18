"""单卡工程验证流程。合成轨迹和短训练不构成研究效果证据。"""

import importlib.metadata
import os
from pathlib import Path
import sys
from typing import Any

from .artifacts import file_sha256, model_manifest, write_json
from .checkpoints import TrainingPaused
from .comparison import build_comparison
from .contracts import ExperimentContract
from .data import TrajectoryRecord, validate_dataset
from .devices import require_uuid_binding, verify_device
from .hf_backend import HfRuntime, load_dependencies, load_tokenizer
from .model_inputs import bound_model
from .scoring import attach_scores
from .scoring_run import score_and_record, progress
from .training_run import finish_pilot, resume_pilot
from .state import StateStore
from .tokenization import compile_trajectory
from .validation import choice, digest, fail, integer, number, obj, read_json, read_jsonl, to_json_value


def run_pilot(config_path: str | Path, model_path: str | Path, output_path: str | Path, device: str,
              stop_after_step: int | None = None) -> dict[str, Any]:
    """在新目录执行真实单卡流程。拒绝覆盖或隐式恢复已有运行。"""
    path = Path(config_path).resolve()
    spec = obj(read_json(path), "pilot", {"schema_version", "source", "contract_template", "seed",
               "examples_per_arm", "precision", "learning_rate", "lora_rank", "max_sequence_length", "expected_device_name"})
    if spec["schema_version"] != "pilot-1":
        fail("pilot", "仅支持 pilot-1 工程验证配置。")
    seed = integer(spec["seed"], "pilot.seed")
    examples_per_arm = integer(spec["examples_per_arm"], "pilot.examples_per_arm", 1)
    if stop_after_step is not None and not 1 <= integer(stop_after_step, "stop_after_step", 1) <= examples_per_arm:
        fail("stop_after_step", "暂停步骤必须位于本组的训练步骤范围内。")
    max_length = integer(spec["max_sequence_length"], "pilot.max_sequence_length", 2)
    rank = integer(spec["lora_rank"], "pilot.lora_rank", 1)
    learning_rate = number(spec["learning_rate"], "pilot.learning_rate")
    if learning_rate == 0:
        fail("pilot.learning_rate", "必须大于 0。")
    precision = choice(spec["precision"], "pilot.precision", {"bf16", "fp32"})
    require_uuid_binding(device, os.environ.get("CUDA_VISIBLE_DEVICES"))
    output = Path(output_path).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        return _run(spec, path, model_path, output, device, seed, examples_per_arm, max_length,
                    rank, learning_rate, precision, stop_after_step)
    except TrainingPaused as exc:
        write_json(output / "pause.json", {"training_status": "PAUSED", "checkpoint": to_json_value(exc.artifact),
                   "arm": "random", "resume_scope": "complete_step_boundary"})
        return {"training_status": "PAUSED", "checkpoint": to_json_value(exc.artifact), "output": str(output)}
    except BaseException as exc:
        write_json(output / "failure.json", {"status": "FAILED", "error": str(exc) or type(exc).__name__,
                   "cost_evidence": "inspect state.sqlite3; interrupted kernel cost can be unknown",
                   "resume_supported": False})
        raise


def _run(
    spec: dict[str, Any], config_path: Path, model_path: str | Path, output: Path, device: str,
    seed: int, examples_per_arm: int, max_length: int, rank: int, learning_rate: float, precision: str,
    stop_after_step: int | None,
) -> dict[str, Any]:
    torch, _, _ = load_dependencies()
    binding = require_uuid_binding(device, os.environ.get("CUDA_VISIBLE_DEVICES"))
    if torch.cuda.device_count() != 1:
        fail("device", "工程验证必须只看到一个 CUDA 设备。")
    hardware = verify_device(torch.cuda.get_device_properties(device), spec["expected_device_name"], binding)
    torch.cuda.reset_peak_memory_stats(device)
    implementation = [{"name": p.name, "sha256": file_sha256(p)} for p in sorted(Path(__file__).parent.glob("*.py"))]
    code_identity = digest(implementation)
    write_json(output / "code_identity.json", {"content_id": code_identity, "files": implementation,
               "pilot_config_sha256": file_sha256(config_path),
               "source_sha256": file_sha256(config_path.parent / spec["source"]),
               "contract_template_sha256": file_sha256(config_path.parent / spec["contract_template"])})
    progress("hash_local_model")
    identity = model_manifest(model_path)
    write_json(output / "model_identity.json", identity)
    tokenizer = load_tokenizer(model_path)
    raw = read_json(config_path.parent / spec["contract_template"])
    raw["purpose"], raw["contract_id"], raw["seed"] = "draft", "local_engineering_pilot_v1", seed
    raw["model"] = bound_model(identity, tokenizer)
    raw["scoring"] = {"rule_version": "masked_full_vocab_v1", "reference_version": "pending", "cdf_version": "pending"}
    raw["training"].update(backend="hf_lora", world_size=1, precision=precision, learning_rate=learning_rate,
                           lora_rank=rank, max_sequence_length=max_length, per_device_batch_size=1,
                           gradient_accumulation_steps=1)
    # 该配置只用于工程验证。研究用覆盖配额与预算必须在真实数据实验中另行冻结。
    raw["selection"]["min_skill_shares"] = {}
    sources = read_jsonl(config_path.parent / spec["source"])
    contract = ExperimentContract.parse(raw)
    progress("tokenize_sources")
    compiled = [compile_trajectory(source, tokenizer, contract, max_length) for source in sources]
    records = validate_dataset(compiled, contract)
    counts = {r.supervised_tokens for r in records if r.is_eligible}
    if len(counts) != 1 or next(iter(counts), 0) == 0:
        fail("pilot", "合成候选必须具有相同监督长度，以检查严格等 token 对照。")
    per_arm = examples_per_arm * next(iter(counts))
    raw["selection"].update(supervised_token_budget=per_arm, allowed_shortfall=0)
    raw["budgets"] = {"training_tokens": 2 * per_arm, "scoring_tokens": 100000, "evaluation_calls": 0}
    references = [r.raw for r in records if r.role == "reference"]
    raw["scoring"]["reference_version"] = digest(references)
    raw["scoring"]["cdf_version"] = digest({"reference": raw["scoring"]["reference_version"],
                                           "model": raw["model"], "rule": "midrank_ecdf_v1"})
    raw["hardware"] = {"accelerator": hardware["name"], "gpu_count": 1,
                       "memory_gb_per_gpu": max(1, int((hardware["memory_bytes"] or 0) / 1024 ** 3)),
                       "evidence": "verified"}
    contract = ExperimentContract.parse(raw)
    write_json(output / "contract.json", raw)
    write_json(output / "tokenized.json", compiled)
    write_json(output / "environment.json", {"python": sys.version, "hardware": hardware,
               "packages": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "peft", "accelerate", "safetensors"]}})
    feedback = tuple(r for r in records if r.role == "feedback_diagnostic")
    score_records = tuple(r for r in records if r.role in {"candidate", "reference", "feedback_diagnostic"})
    if not feedback:
        fail("pilot", "需要固定的反馈诊断轨迹。")
    with StateStore(output / "state.sqlite3") as state:
        state.initialize("scoring", "controlled", raw, compiled)
        progress("baseline_scoring")
        with HfRuntime(model_path, device, precision) as runtime:
            base_metrics = score_and_record(runtime, score_records, state, "baseline")
        write_json(output / "baseline_scores.json", to_json_value(base_metrics))
        scored = attach_scores(records, base_metrics, contract)
        write_json(output / "scored.json", scored)
        state.initialize("training", "controlled", raw, scored)
        stored_plan = state.prepare_comparison("training", "random_vs_fixed")
        context = state.context("training")
        plan = build_comparison(context, "training", "random_vs_fixed")
        if plan.to_dict() != stored_plan:
            fail("plan", "纯计算计划与已保存计划不一致。")
    return finish_pilot(output, model_path, device, stop_after_step=stop_after_step)
