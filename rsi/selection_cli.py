"""标签策略命令参数与业务调用。"""

import argparse
from pathlib import Path
from typing import Any

from .artifacts import write_json
from .dispatch import apply_dispatch, parse_dispatch
from .feedback import parse_feedback
from .history import BatchHistory
from .labels import SkillOntology, fields
from .policy import parse_policy
from .policy_selection import validate_selection
from .selection_demo import make_fixture, run_demo, save_selection, select_inputs
from .selection_io import finish_manifest, load_inputs, publish_directory, verify_manifest
from .validation import array, fail, integer, read_json, strings


COMMANDS = {"make-selection-fixture", "validate-selection", "select-with-policy", "apply-dispatch", "selection-demo"}


def add_commands(commands: argparse._SubParsersAction) -> None:
    fixture = commands.add_parser("make-selection-fixture", help="生成十条合成候选及独立标签。")
    fixture.add_argument("--output", required=True)
    fixture.add_argument("--seed", type=int, default=42)
    fixture.add_argument("--settings", help="技能列表与每条标签数。")
    for name in ("validate-selection", "select-with-policy"):
        cmd = commands.add_parser(name, help="校验或执行独立标签策略。")
        for field in ("config", "data", "labels", "ontology", "policy"):
            cmd.add_argument("--" + field, required=True)
        cmd.add_argument("--batch-history")
        cmd.add_argument("--allow-synthetic-labels", action="store_true")
        if name == "select-with-policy":
            cmd.add_argument("--output", required=True)
    apply = commands.add_parser("apply-dispatch", help="校验模拟调度并发布新策略，不切换模型。")
    for field in ("policy", "dispatch", "feedback", "ontology", "batch-history", "output"):
        apply.add_argument("--" + field, required=True)
    apply.add_argument("--dispatch-history", help="按时间顺序的历史调度 JSON 数组。")
    apply.add_argument("--feedback-history", help="历史反馈 JSON 数组。")
    demo = commands.add_parser("selection-demo", help="执行纯模拟筛选、调度和再次筛选。")
    demo.add_argument("--fixture", required=True)
    demo.add_argument("--output", required=True)


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "make-selection-fixture":
        if not args.settings:
            return make_fixture(args.output, args.seed)
        values = fields(read_json(args.settings), "fixture.settings", "skills labels_per_sample")
        return make_fixture(args.output, args.seed, strings(values["skills"], "fixture.settings.skills", True),
                            integer(values["labels_per_sample"], "fixture.settings.labels_per_sample", 1))
    if args.command == "selection-demo":
        return run_demo(args.fixture, args.output)
    if args.command == "apply-dispatch":
        return apply_command(args)
    inputs = load_inputs(args.config, args.data, args.labels, args.ontology, args.policy, args.batch_history)
    if args.command == "validate-selection":
        context = validate_selection(inputs.contract, inputs.records, inputs.labels, inputs.ontology,
                                     inputs.policy, inputs.history, allow_synthetic_labels=args.allow_synthetic_labels)
        return {"status": "VALIDATED", "eligible": len(context.records), "training_status": "NOT_RUN", "task_execution_status": "NOT_RUN"}
    result = select_inputs(inputs, allow_synthetic_labels=args.allow_synthetic_labels)
    if result.batch is None:
        # 完整失败报告可由纯接口获取。CLI 失败时不发布成功目录。
        fail("selection." + result.to_dict()["status"], result.to_dict()["failure_reason"])
    with publish_directory(args.output) as root:
        save_selection(root, result)
        finish_manifest(root, "selection")
    return result.to_dict()


def apply_command(args: argparse.Namespace) -> dict[str, Any]:
    ontology = SkillOntology.parse(read_json(args.ontology))
    policy = parse_policy(read_json(args.policy), ontology)
    dispatch = parse_dispatch(read_json(args.dispatch))
    feedback = parse_feedback(read_json(args.feedback), ontology)
    history = BatchHistory.parse(read_json(args.batch_history))
    prior = tuple(parse_dispatch(v) for v in array(read_json(args.dispatch_history), "dispatch_history")) if args.dispatch_history else ()
    old_feedback = tuple(parse_feedback(v, ontology) for v in array(read_json(args.feedback_history), "feedback_history")) if args.feedback_history else ()
    application = apply_dispatch(policy, dispatch, feedback, ontology, history, prior, old_feedback)
    target = Path(args.output)
    if target.exists():
        verify_manifest(target)
        receipt = read_json(target / "dispatch_application.json")
        old_dispatch = read_json(target / "dispatch.json")
        if old_dispatch["dispatch_id"] != dispatch.dispatch_id or receipt["request_hash"] != application.request_hash:
            fail("output.dispatch_id", "输出已存在且请求内容冲突。")
        return {**application.to_dict(), "status": "already_applied"}
    with publish_directory(target) as root:
        write_json(root / "policy.parent.json", policy.to_dict())
        write_json(root / "policy.after.json", application.policy.to_dict())
        write_json(root / "dispatch.json", dispatch.to_dict())
        write_json(root / "feedback.json", feedback.to_dict())
        write_json(root / "ontology.json", ontology.to_dict())
        write_json(root / "batch_history.json", history.to_dict())
        write_json(root / "dispatch_history.json", [d.to_dict() for d in prior])
        write_json(root / "feedback_history.json", [f.to_dict() for f in old_feedback])
        write_json(root / "dispatch_application.json", application.to_dict())
        finish_manifest(root, "dispatch")
    return {**application.to_dict(), "status": "applied"}
