"""第二阶段命令参数。业务规则保存在各职责模块。"""

import argparse
from typing import Any

from rsi.configuration.validation import to_json_value


COMMANDS = {"compile-training-data", "prepare-training", "train-attempt", "resume-training", "evaluate-fit",
            "evaluate-execution", "validate-training", "evaluate-benchmark", "training-validation-demo"}


def add_commands(commands: argparse._SubParsersAction) -> None:
    for name in sorted(COMMANDS):
        cmd = commands.add_parser(name, help="独立单次训练与验证；不自动晋级。")
        if name == "compile-training-data":
            for arg in ("sources", "config", "metadata", "ontology", "policy", "model-location", "output", "mode", "device"):
                cmd.add_argument("--" + arg, required=True)
            cmd.add_argument("--max-length", type=int, required=True)
            cmd.add_argument("--data-kind", choices=["synthetic", "real_public", "private"], required=True)
            cmd.add_argument("--message-evidence")
        elif name == "prepare-training":
            for arg in ("bundle", "selection", "selection-id", "model-location", "settings", "output", "run-id", "attempt-id"):
                cmd.add_argument("--" + arg, required=True)
        elif name == "training-validation-demo":
            cmd.add_argument("--output", required=True)
            cmd.add_argument("--mode", choices=["mock", "hf_local"], default="mock")
            cmd.add_argument("--model-location")
            cmd.add_argument("--device", default="cpu")
        elif name == "evaluate-benchmark":
            for arg in ("bundle", "model-location", "tasks", "output", "mode", "device"):
                cmd.add_argument("--" + arg, required=True)
            cmd.add_argument("--call-budget", type=int, required=True)
            cmd.add_argument("--feedback-tasks", required=True)
            cmd.add_argument("--sandbox-config")
        else:
            cmd.add_argument("--attempt", required=True)
            cmd.add_argument("--device", required=True)
            if name == "train-attempt":
                cmd.add_argument("--stop-after-step", type=int)
            if name == "resume-training":
                cmd.add_argument("--checkpoint", required=True)
                cmd.add_argument("--checkpoint-sha256", required=True)
            if name in {"evaluate-fit", "evaluate-execution"}:
                cmd.add_argument("--model", choices=["parent", "candidate"], required=True)
                cmd.add_argument("--output", required=True)
            if name in {"evaluate-execution", "validate-training"}:
                cmd.add_argument("--tasks", required=True)
                cmd.add_argument("--call-budget", type=int, required=True)
                cmd.add_argument("--sandbox-config")
            if name == "validate-training":
                cmd.add_argument("--rule")
        cmd.add_argument("--expected-gpu-name")


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    from rsi.workflows.attempt_run import prepare_training_attempt, run_training_attempt, resume_training_attempt
    from rsi.models.backends import ArtifactRef
    from rsi.workflows.benchmark_run import evaluate_benchmark
    from rsi.storage.checkpoints import TrainingPaused
    from rsi.workflows.training_compile import compile_training_data
    from rsi.workflows.validation_run import evaluate_fit_stage, evaluate_execution_stage, run_dual_validation
    gpu = args.expected_gpu_name
    if args.command == "compile-training-data":
        return compile_training_data(args.sources, args.config, args.metadata, args.ontology, args.policy, args.model_location,
                args.output, mode=args.mode, device=args.device, max_length=args.max_length, data_kind=args.data_kind,
                message_evidence_path=args.message_evidence, expected_gpu_name=gpu)
    if args.command == "prepare-training":
        return prepare_training_attempt(args.bundle, args.selection, args.selection_id, args.model_location,
                                        args.settings, args.output, args.run_id, args.attempt_id).to_dict()
    if args.command == "training-validation-demo":
        from rsi.workflows.training_demo import run_demo
        return run_demo(args.output, mode=args.mode, location_path=args.model_location, device=args.device, expected_gpu_name=gpu)
    if args.command == "evaluate-benchmark":
        return evaluate_benchmark(args.bundle, args.model_location, args.tasks, args.output, mode=args.mode, device=args.device,
                                 call_budget=args.call_budget, feedback_suite_path=args.feedback_tasks, sandbox_path=args.sandbox_config, expected_gpu_name=gpu)
    if args.command in {"train-attempt", "resume-training"}:
        try:
            if args.command == "resume-training":
                result = resume_training_attempt(args.attempt, args.device, ArtifactRef(args.checkpoint, args.checkpoint_sha256), expected_gpu_name=gpu)
            else:
                result = run_training_attempt(args.attempt, args.device, stop_after_step=args.stop_after_step, expected_gpu_name=gpu)
        except TrainingPaused as exc:
            return {"status": "PAUSED", "checkpoint": to_json_value(exc.artifact)}
        return {"status": "TRAINED", "result": to_json_value(result), "promotion_status": "NOT_APPLIED"}
    if args.command == "evaluate-fit":
        return evaluate_fit_stage(args.attempt, args.output, candidate=args.model == "candidate", device=args.device, expected_gpu_name=gpu).to_dict()
    if args.command == "evaluate-execution":
        return evaluate_execution_stage(args.attempt, args.tasks, args.output, candidate=args.model == "candidate", device=args.device,
                       call_budget=args.call_budget, sandbox_path=args.sandbox_config, expected_gpu_name=gpu).to_dict()
    return run_dual_validation(args.attempt, args.tasks, device=args.device, call_budget=args.call_budget,
                              rule_path=args.rule, sandbox_path=args.sandbox_config, expected_gpu_name=gpu).to_dict()
