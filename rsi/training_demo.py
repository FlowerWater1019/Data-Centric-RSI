"""可复现的单次编排演示。mock 数字仅用于测试，不表示模型效果。"""

from dataclasses import replace
from pathlib import Path
from typing import Any

from .artifacts import write_json
from .attempt_run import prepare_training_attempt, run_training_attempt
from .benchmark_run import evaluate_benchmark
from .contracts import ExperimentContract
from .hf_backend import load_tokenizer
from .model_resolution import LoadConfig, ModelLocation, check_device
from .selection_demo import fixture_values, save_selection, select_inputs, write_jsonl
from .selection_io import finish_manifest, load_inputs, publish_directory
from .training_compile import MockChatTokenizer, compile_training_data
from .training_inputs import compile_training_sources
from .validation import digest, fail, read_json, to_json_value
from .validation_run import evaluate_fit_stage, run_dual_validation


def example_values() -> dict[str, Any]:
    """标准 source-1 示例与两类独立任务。技能标签仍明确为合成。"""
    values = fixture_values(labels_per_sample=2)
    sources = []
    evidence = []
    for row in values["data"]:
        source = {k: row[k] for k in ("data_id", "role", "split_group", "duplicate_group", "skills", "quality")}
        source["schema_version"] = "source-1"
        source["messages"] = [
            {"role": "user", "content": "Add two integers. Record " + row["data_id"], "correctness": "context", "supervise": False, "evidence_id": None, "diagnostics": []},
            {"role": "assistant", "content": "def solve(x, y):\n    return x + y\n", "correctness": "valid", "supervise": True,
             "evidence_id": "synthetic_" + row["data_id"], "diagnostics": []}]
        sources.append(source)
        for skill in row["skills"]:
            evidence.append({"data_id": row["data_id"], "trajectory_version": digest(source), "skill_id": skill,
                             "source": "synthetic_fixture", "message_index": 1})
    values["sources"] = sources
    values["evidence"] = {"schema_version": "message-evidence-v1", "index_rule": "source_message_index_zero_v1", "items": evidence}
    values["settings"] = {"schema_version": "training-attempt-v1", "mode": "mock", "training": {
        "backend": "hf_lora", "world_size": 1, "precision": "fp32", "learning_rate": 0.001, "lora_rank": 2,
        "max_sequence_length": 512, "per_device_batch_size": 1, "gradient_accumulation_steps": 1,
        "optimizer_state": "fresh_per_attempt", "scheduler_state": "fresh_per_attempt"},
        "lora": {"targets": ["q_proj", "v_proj"], "alpha": 4.0, "dropout": 0.0}, "load": to_json_value(LoadConfig())}
    return values


def example_suite(ontology: dict[str, Any], *, final_test: bool = False) -> dict[str, Any]:
    prefix = "benchmark" if final_test else "feedback"
    tasks = []
    for i, group in enumerate(("final_test", "final_test") if final_test else ("target", "retention")):
        tasks.append({"task_id": f"{prefix}_{i}", "source_problem_id": f"{prefix}_problem_{i}",
            "split_group": f"{prefix}_split_{i}", "duplicate_group": f"{prefix}_dup_{i}", "group": group,
            "skills": [s["id"] for s in ontology["skills"][:2]], "label_source": "synthetic_fixture",
            "content_fingerprint": digest({"prefix": prefix, "index": i}), "adapter_id": "bounded_integer_expression_v1",
            "adapter_version": "1", "payload": {"prompt": "Write solve(x, y) to add two integers.",
                            "tests": [{"arguments": [i + 1, 4], "expected": i + 5}, {"arguments": [-2, 7], "expected": 5}]}})
    return {"schema_version": "execution-suite-v1", "suite_id": prefix + "_smoke", "revision": "synthetic-v1",
        "purpose": "final_test" if final_test else "feedback", "domain": ontology["domain"],
        "ontology_version": ontology["ontology_version"], "provenance": "synthetic_fixture", "environment_id": "bounded-expression-v1",
        "generation": {"strategy": "greedy", "attempts": 1, "max_new_tokens": 64, "max_context_length": 512, "seed": 42}, "tasks": tasks}


def run_demo(output: str | Path, *, mode: str = "mock", location_path: str | Path | None = None,
             device: str = "cpu", expected_gpu_name: str | None = None) -> dict[str, Any]:
    root = Path(output).resolve()
    if root.exists():
        fail("demo.output", "演示输出已存在，拒绝覆盖。")
    if mode not in {"mock", "hf_local"}:
        fail("demo.mode", "只支持 mock 或 hf_local。")
    values = example_values()
    config, settings = values["config"], values["settings"]
    config["training"] = settings["training"].copy()
    config["budgets"] = {"training_tokens": 100000, "scoring_tokens": 1000000, "evaluation_calls": 100}
    if mode == "hf_local":
        if location_path is None:
            fail("demo.model_location", "真实模型演示必须提供本机定位配置。")
        file = Path(location_path).resolve()
        location = ModelLocation.parse(read_json(file), file.parent)
        check_device(device, expected_gpu_name)
        tokenizer = load_tokenizer(location.base_model_path)
        location.verify(tokenizer)
        settings["load"] = to_json_value(location.load)
        settings["training"]["precision"] = location.load.precision
        config["training"]["precision"] = location.load.precision
    else:
        config["model"]["adapter_revision"] = "none"
        config["model"]["checkpoint_id"] = "mock_parent_explicit_fixture"
        location = ModelLocation.parse({"schema_version": "model-location-v1", "model_ref": config["model"],
                    "base_model_path": ".", "adapter_manifest": None, "load": settings["load"]}, root)
        tokenizer = MockChatTokenizer()
    settings["mode"] = mode
    config["model"] = to_json_value(location.model_ref)
    contract = ExperimentContract.parse(config)
    compiled = compile_training_sources(values["sources"], tokenizer, contract, 512)
    budget = sum(r.supervised_tokens for r in compiled.records[:4])
    config["selection"]["supervised_token_budget"] = budget
    values["policy"]["batch_constraints"]["supervised_token_budget"] = budget
    root.mkdir(parents=True)
    source = root / "source"
    source.mkdir()
    for name, value in (("config", config), ("ontology", values["ontology"]), ("policy", values["policy"]),
                         ("settings", settings), ("message_evidence", values["evidence"]), ("model_location", location.to_dict()),
                         ("feedback_tasks", example_suite(values["ontology"])), ("final_tasks", example_suite(values["ontology"], final_test=True))):
        write_json(source / (name + ".json"), value)
    write_jsonl(source / "sources.jsonl", values["sources"])
    write_jsonl(source / "metadata.jsonl", values["labels"])
    compilation = compile_training_data(source / "sources.jsonl", source / "config.json", source / "metadata.jsonl",
                   source / "ontology.json", source / "policy.json", source / "model_location.json", root / "compiled",
                   mode=mode, device=device, max_length=512, data_kind="synthetic", message_evidence_path=source / "message_evidence.json",
                   expected_gpu_name=expected_gpu_name)
    bundle = root / "compiled"
    inputs = load_inputs(*(bundle / name for name in ("config.json", "trajectories.jsonl", "labels.jsonl", "ontology.json", "policy.json", "batch_history.json")))
    selected = select_inputs(inputs, allow_synthetic_labels=True)
    if selected.batch is None:
        fail("demo.selection", selected.to_dict()["failure_reason"])
    with publish_directory(root / "selection") as selected_root:
        save_selection(selected_root, selected)
        selection_manifest = finish_manifest(selected_root, "selection")
    attempt = root / "attempt"
    plan = prepare_training_attempt(bundle, root / "selection", selection_manifest["content_id"], source / "model_location.json",
                                    source / "settings.json", attempt, "single_demo", "attempt_001")
    # 训练前实际测量父模型。双路编排会核验身份并复用此完整结果。
    evaluate_fit_stage(attempt, attempt / "validation/fit-parent", candidate=False, device=device, expected_gpu_name=expected_gpu_name)
    trained = run_training_attempt(attempt, device, expected_gpu_name=expected_gpu_name)
    feedback = run_dual_validation(attempt, source / "feedback_tasks.json", device=device, call_budget=2, expected_gpu_name=expected_gpu_name)
    candidate = replace(location, model_ref=trained.candidate_model, adapter_manifest=trained.artifact)
    write_json(root / "candidate_location.json", candidate.to_dict())
    benchmark = evaluate_benchmark(bundle, root / "candidate_location.json", source / "final_tasks.json", root / "benchmark",
                                   mode=mode, device=device, call_budget=2, feedback_suite_path=source / "feedback_tasks.json", expected_gpu_name=expected_gpu_name)
    report = {"status": feedback.status, "mode": mode, "source": feedback.source, "data_kind": "synthetic",
        "label_mode": plan.label_mode, "compiled_content_id": compilation["content_id"], "training_tokens": trained.actual_supervised_tokens,
        "fit_delta_nll": feedback.fit.overall.delta_nll, "benchmark_status": benchmark["status"],
        "controller_status": "NOT_RUN", "promotion_status": "NOT_APPLIED", "multi_round_status": "NOT_RUN"}
    write_json(root / "demo_report.json", report)
    return report
