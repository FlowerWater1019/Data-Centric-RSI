"""真实数据的单卡开发对照。评分、训练、恢复复用既有执行组件。"""

import importlib.metadata
import os
from pathlib import Path
import sys
from typing import Any

from rsi.storage.provenance import source_manifest
from rsi.storage.artifacts import file_sha256, model_manifest, verify_training_artifact, write_json
from rsi.models.backends import ArtifactRef
from rsi.storage.checkpoints import TrainingPaused
from rsi.configuration.contracts import ExperimentContract
from rsi.datasets.data import validate_dataset
from rsi.models.devices import require_uuid_binding, verify_device
from rsi.models.hf_backend import HfRuntime, load_dependencies
from rsi.models.scoring import attach_scores
from rsi.workflows.scoring_run import progress, score_and_record
from rsi.storage.state import StateStore
from rsi.workflows.training_run import finish_pilot
from rsi.configuration.validation import choice, digest, fail, integer, obj, read_json, read_jsonl, string, to_json_value


def comparison_inputs(compiled_path: str | Path, compiled_sha256: str,
                      config_path: str | Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """核验编译产物与运行参数。此操作不加载模型，不启动评分或训练。"""
    root = Path(compiled_path).resolve()
    verify_training_artifact(ArtifactRef(str(root / "report.json"), compiled_sha256))
    report = read_json(root / "report.json")
    if report.get("schema_version") != "compiled-data-1" or report.get("data_kind") != "real_public":
        fail("compiled", "需要完整的真实数据编译产物。")
    if not {"contract.json", "trajectories.jsonl", "model_identity.json"} <= {p["name"] for p in report["files"]}:
        fail("compiled", "编译产物缺少必要文件。")
    spec = obj(read_json(config_path), "comparison", {"schema_version", "purpose", "seed", "selection",
               "training", "scoring_token_budget", "expected_device_name"})
    choice(spec["schema_version"], "comparison.schema_version", {"real-comparison-1"})
    choice(spec["purpose"], "comparison.purpose", {"development"})
    string(spec["expected_device_name"], "expected_device_name")
    config = read_json(root / "contract.json")
    if config["purpose"] != "draft":
        fail("compiled.purpose", "当前真实对照使用 draft 合同。")
    config.update(schema_version="1.1", contract_id="real_data_comparison_v1",
                  seed=integer(spec["seed"], "comparison.seed"), selection=spec["selection"], training=spec["training"])
    budget = integer(spec["selection"].get("supervised_token_budget"), "selection.supervised_token_budget", 1)
    config["budgets"] = {"training_tokens": 2 * budget,
                         "scoring_tokens": integer(spec["scoring_token_budget"], "scoring_token_budget", 1),
                         "evaluation_calls": 0}
    records_raw = read_jsonl(root / "trajectories.jsonl")
    contract = ExperimentContract.parse(config)
    if contract.model.adapter_revision != "none":
        fail("model.adapter_revision", "当前真实对照要求没有已有适配器的起点。")
    if contract.packing_rule != "exact_ranked_v1" or contract.allowed_shortfall != 0:
        fail("selection", "真实对照要求完整轨迹精确组批。")
    training = contract.training
    if (training.backend, training.world_size, training.per_device_batch_size, training.gradient_accumulation_steps) != ("hf_lora", 1, 1, 1):
        fail("training", "当前真实对照只支持单卡、单样本、无梯度累积的 hf_lora。")
    if training.learning_rate is None or training.lora_rank is None or training.max_sequence_length is None:
        fail("training", "学习率、LoRA 秩和最大长度必须明确。")
    records = validate_dataset(records_raw, contract)
    if any(record.role not in {"candidate", "reference", "feedback_diagnostic"} for record in records):
        fail("compiled.roles", "当前对照只接收候选、固定参考和固定反馈诊断。")
    if not all(any(r.role == role for r in records) for role in ("candidate", "reference", "feedback_diagnostic")):
        fail("compiled.roles", "候选、参考和反馈诊断均不能为空。")
    if any(r.scores is not None for r in records):
        fail("compiled.scores", "需要未评分的原始编译产物。")
    if any(r.token_count > training.max_sequence_length for r in records):
        fail("training.max_sequence_length", "存在超长轨迹。程序不会截断。")
    cost = sum(r.supervised_tokens for r in records) + 2 * sum(r.supervised_tokens for r in records if r.role == "feedback_diagnostic")
    if cost > config["budgets"]["scoring_tokens"]:
        fail("scoring_token_budget", "预算不能覆盖基线全量评分和两组固定诊断。")
    # 在昂贵模型调用前检查预算可达性。最终选择仍使用真实评分后的排名。
    from rsi.curation.packing import exact_ranked
    exact_ranked([r for r in records if r.is_eligible], budget)
    config["scoring"] = {"rule_version": "masked_full_vocab_v1",
                         "reference_version": digest([r.raw for r in records if r.role == "reference"]),
                         "cdf_version": digest({"model": to_json_value(contract.model),
                                                "references": [r.raw for r in records if r.role == "reference"],
                                                "rule": "midrank_ecdf_v1"})}
    return root, config, records_raw, spec


def preflight_comparison(compiled_path: str | Path, compiled_sha256: str,
                         config_path: str | Path) -> dict[str, Any]:
    """检查输入与预算。固定策略是否满足配额仍需真实评分后判断。"""
    _, config, rows, _ = comparison_inputs(compiled_path, compiled_sha256, config_path)
    records = validate_dataset(rows, ExperimentContract.parse(config))
    return {"status": "INPUTS_VALID", "training": "NOT_RUN", "scoring": "NOT_RUN",
            "compiled_sha256": compiled_sha256, "config_sha256": file_sha256(config_path),
            "candidate_records": sum(r.is_eligible for r in records),
            "candidate_supervised_tokens": sum(r.supervised_tokens for r in records if r.is_eligible),
            "per_arm_tokens": config["selection"]["supervised_token_budget"],
            "packing_rule": config["selection"]["packing_rule"], "quotas": "CHECK_AFTER_SCORING"}


def run_comparison(compiled_path: str | Path, compiled_sha256: str, config_path: str | Path,
                   model_path: str | Path, output_path: str | Path, device: str,
                   stop_after_step: int | None = None) -> dict[str, Any]:
    """冻结真实输入并运行两个训练分支。输出目录必须尚不存在。"""
    root, config, rows, spec = comparison_inputs(compiled_path, compiled_sha256, config_path)
    if stop_after_step is not None:
        integer(stop_after_step, "stop_after_step", 1)
    binding = require_uuid_binding(device, os.environ.get("CUDA_VISIBLE_DEVICES"))
    torch, _, _ = load_dependencies()
    if torch.cuda.device_count() != 1:
        fail("device", "当前对照必须只看到一个 CUDA 设备。")
    hardware = verify_device(torch.cuda.get_device_properties(device), spec["expected_device_name"], binding)
    identity = model_manifest(model_path)
    if identity["content_id"] != read_json(root / "model_identity.json")["content_id"]:
        fail("model", "当前本地模型与编译时模型不同。")
    config["hardware"] = {"accelerator": hardware["name"], "gpu_count": 1,
                          "memory_gb_per_gpu": max(1, int(hardware["memory_bytes"] / 1024**3)), "evidence": "verified"}
    contract = ExperimentContract.parse(config)
    records = validate_dataset(rows, contract)
    output = Path(output_path).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        implementation = source_manifest()
        write_json(output / "code_identity.json", {"content_id": digest(implementation), "files": implementation,
                   "data_kind": "real_public", "compiled_sha256": compiled_sha256,
                   "comparison_config_sha256": file_sha256(config_path)})
        write_json(output / "input_report.json", read_json(root / "report.json"))
        write_json(output / "comparison_config.json", spec)
        write_json(output / "model_identity.json", identity)
        write_json(output / "environment.json", {"python": sys.version, "hardware": hardware,
                   "packages": {name: importlib.metadata.version(name) for name in
                                ("torch", "transformers", "peft", "accelerate", "safetensors")}})
        write_json(output / "contract.json", config)
        write_json(output / "tokenized.json", rows)
        with StateStore(output / "state.sqlite3") as state:
            state.initialize("scoring", "controlled", config, rows)
            progress("real_baseline_scoring")
            with HfRuntime(model_path, device, contract.training.precision) as runtime:
                metrics = score_and_record(runtime, records, state, "baseline", output / "score_calls")
            write_json(output / "baseline_scores.json", to_json_value(metrics))
            scored = attach_scores(records, metrics, contract)
            write_json(output / "scored.json", scored)
            state.initialize("training", "controlled", config, scored)
            plan = state.prepare_comparison("training", "random_vs_fixed")
            write_json(output / "plan.json", plan)
            if stop_after_step is not None and stop_after_step > len(plan["arms"]["random"]["items"]):
                fail("stop_after_step", "暂停步骤超过随机分支的计划步骤数。")
        return finish_pilot(output, model_path, device, stop_after_step=stop_after_step)
    except TrainingPaused as exc:
        write_json(output / "pause.json", {"training_status": "PAUSED", "arm": "random",
                   "checkpoint": to_json_value(exc.artifact), "resume_scope": "complete_step_boundary"})
        return {"training_status": "PAUSED", "output": str(output), "checkpoint": to_json_value(exc.artifact)}
    except BaseException as exc:
        write_json(output / "failure.json", {"status": "FAILED", "error": str(exc) or type(exc).__name__,
                   "resume_supported": False, "cost_evidence": "state.sqlite3 and score_calls; incomplete calls can have unknown cost"})
        raise
