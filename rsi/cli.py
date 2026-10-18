"""命令入口。演示计划不执行训练。"""

import argparse
import json
from pathlib import Path
import sqlite3
import sys

from .contracts import ExperimentContract
from .data import validate_dataset
from .inspection import inspect_scores
from .state import StateStore
from .sandbox import SandboxUnavailable
from .validation import ValidationError, read_json, read_jsonl


def emit(value: dict[str, object]) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="RSI 实验内核。支持结构检查、计划、状态和单卡工程验证。")
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="校验配置和轨迹，不启动训练。")
    validate.add_argument("--config", required=True)
    validate.add_argument("--data", required=True)
    validate.add_argument("--require-current-scores", action="store_true")
    initialize = commands.add_parser("init", help="冻结输入并创建运行。")
    initialize.add_argument("--config", required=True)
    initialize.add_argument("--data", required=True)
    initialize.add_argument("--mode", choices=["online", "controlled"], default="controlled")
    for name in ("plan-comparison", "status", "export"):
        commands.add_parser(name)
    for name in ("init", "plan-comparison", "status", "export"):
        command = commands.choices[name]
        command.add_argument("--db", required=True)
        command.add_argument("--run-id", required=True)
    commands.choices["plan-comparison"].add_argument("--comparison-id", required=True)
    commands.choices["export"].add_argument("--output", required=True)
    commands.add_parser("train", help="正式研究入口仍未实现；工程验证使用 pilot。")
    pilot = commands.add_parser("pilot", help="使用本地小模型运行单卡工程验证。")
    pilot.add_argument("--config", required=True)
    pilot.add_argument("--model-path", required=True)
    pilot.add_argument("--output", required=True)
    pilot.add_argument("--device", required=True)
    pilot.add_argument("--stop-after-step", type=int, help="工程恢复检查：在随机组指定步骤后暂停。")
    resume = commands.add_parser("resume-pilot", help="从完整步骤边界恢复原工程尝试。")
    resume.add_argument("--output", required=True)
    resume.add_argument("--model-path", required=True)
    resume.add_argument("--device", required=True)
    resume.add_argument("--arm", choices=["random", "fixed"], required=True)
    resume.add_argument("--checkpoint", required=True)
    resume.add_argument("--checkpoint-sha256", required=True)
    evaluate = commands.add_parser("evaluate-pilot", help="执行独立纯函数任务并生成工程验收判定。")
    evaluate.add_argument("--pilot", required=True)
    evaluate.add_argument("--tasks", required=True)
    evaluate.add_argument("--rule", required=True)
    evaluate.add_argument("--model-path", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", required=True)
    evaluate.add_argument("--sandbox-config")
    evaluate.add_argument("--max-context-length", type=int)
    check = commands.add_parser("sandbox-check", help="检查 Linux 隔离机制，不加载模型。")
    check.add_argument("--config", required=True)
    check.add_argument("--output", required=True)
    data_import = commands.add_parser("import-mbpp", help="导入官方 MBPP，隔离检查参考解并冻结划分。")
    data_import.add_argument("--source", required=True)
    data_import.add_argument("--origin", required=True)
    data_import.add_argument("--sandbox-config", required=True)
    data_import.add_argument("--output", required=True)
    data_import.add_argument("--limit-per-role", type=int)
    compile_data = commands.add_parser("compile-data", help="把真实数据包编译为既有轨迹接口。")
    compile_data.add_argument("--bundle", required=True)
    compile_data.add_argument("--bundle-sha256", required=True)
    compile_data.add_argument("--config", required=True)
    compile_data.add_argument("--model-path", required=True)
    compile_data.add_argument("--output", required=True)
    compile_data.add_argument("--max-length", type=int, required=True)
    compile_data.add_argument("--diagnostics", choices=["none", "python_keywords_v1"], default="none")
    for name in ("check-comparison", "run-comparison"):
        command = commands.add_parser(name, help="检查或执行真实数据单卡开发对照。")
        command.add_argument("--compiled", required=True)
        command.add_argument("--compiled-sha256", required=True)
        command.add_argument("--config", required=True)
    real = commands.choices["run-comparison"]
    real.add_argument("--model-path", required=True)
    real.add_argument("--output", required=True)
    real.add_argument("--device", required=True)
    real.add_argument("--stop-after-step", type=int)
    real_resume = commands.add_parser("resume-comparison", help="恢复真实数据对照的完整训练步骤。")
    for name in ("output", "model-path", "device", "checkpoint", "checkpoint-sha256"):
        real_resume.add_argument("--" + name, required=True)
    real_resume.add_argument("--arm", required=True, choices=["random", "fixed"])
    report = commands.add_parser("report-comparison", help="核验运行证据并生成开发对照摘要。")
    for name in ("training", "evaluation", "output"):
        report.add_argument("--" + name, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "report-comparison":
            from .reporting import report_comparison
            emit(report_comparison(args.training, args.evaluation, args.output))
            return 0
        if args.command == "check-comparison":
            from .real_run import preflight_comparison
            emit(preflight_comparison(args.compiled, args.compiled_sha256, args.config))
            return 0
        if args.command == "run-comparison":
            from .real_run import run_comparison
            emit(run_comparison(args.compiled, args.compiled_sha256, args.config, args.model_path,
                                args.output, args.device, args.stop_after_step))
            return 0
        if args.command == "resume-comparison":
            from .training_run import resume_pilot
            emit(resume_pilot(args.output, args.model_path, args.device, args.arm, args.checkpoint, args.checkpoint_sha256))
            return 0
        if args.command == "pilot":
            from .pilot import run_pilot
            emit(run_pilot(args.config, args.model_path, args.output, args.device, args.stop_after_step))
            return 0
        if args.command == "resume-pilot":
            from .pilot import resume_pilot
            emit(resume_pilot(args.output, args.model_path, args.device, args.arm, args.checkpoint, args.checkpoint_sha256))
            return 0
        if args.command == "evaluate-pilot":
            from .evaluation_run import run_evaluation
            emit(run_evaluation(args.pilot, args.tasks, args.rule, args.model_path, args.output, args.device,
                                args.sandbox_config, args.max_context_length))
            return 0
        if args.command == "sandbox-check":
            from .sandbox import BubblewrapExecutor, SandboxLimits
            from .artifacts import write_json
            result = BubblewrapExecutor(SandboxLimits.parse(read_json(args.config))).preflight()
            write_json(args.output, result)
            emit({"status": "VERIFIED", "output": args.output})
            return 0
        if args.command == "import-mbpp":
            from .dataset_import import import_mbpp
            emit(import_mbpp(args.source, args.origin, args.sandbox_config, args.output, args.limit_per_role))
            return 0
        if args.command == "compile-data":
            from .data_compile import compile_bundle
            emit(compile_bundle(args.bundle, args.bundle_sha256, args.config, args.model_path, args.output, args.max_length, args.diagnostics))
            return 0
        if args.command == "train":
            raise ValidationError("正式训练入口尚未实现。小模型工程验证请使用 pilot 命令。")
        if args.command == "validate":
            contract = ExperimentContract.parse(read_json(args.config))
            records = validate_dataset(read_jsonl(args.data), contract)
            report = inspect_scores(contract, records, require_current_scores=args.require_current_scores)
            emit(report.to_dict())
            return 0
        if args.command != "init" and not Path(args.db).is_file():
            raise ValidationError("数据库不存在。请先执行 init。")
        with StateStore(args.db, create=args.command == "init",
                        read_only=args.command in {"status", "export"}) as state:
            if args.command == "init":
                created = state.initialize(args.run_id, args.mode, read_json(args.config), read_jsonl(args.data))
                emit({"created": created, "run_id": args.run_id, "training_status": "NOT_RUN"})
            elif args.command == "plan-comparison":
                emit(state.prepare_comparison(args.run_id, args.comparison_id))
            elif args.command == "status":
                emit(state.snapshot(args.run_id))
            elif args.command == "export":
                snapshot = state.snapshot(args.run_id)
                output = Path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open("x", encoding="utf-8") as handle:
                    json.dump(snapshot, handle, ensure_ascii=False, indent=2, allow_nan=False)
                emit({"output": str(output), "training_status": snapshot["training_status"]})
        return 0
    except (ValidationError, OSError, sqlite3.Error, SandboxUnavailable) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
