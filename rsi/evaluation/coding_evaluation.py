"""生成代码并保存逐任务证据。执行器和生成器通过接口注入。"""

from collections.abc import Callable
from pathlib import Path

from rsi.evaluation.acceptance import TaskOutcome
from rsi.storage.artifacts import file_sha256, write_json
from rsi.models.backends import EvaluationRequest, EvaluationResult, MetricResult
from rsi.evaluation.bounded_python import execute_task
from rsi.evaluation.coding_tasks import TaskSuite
from rsi.configuration.validation import fail, read_json, to_json_value


TASK_INSTRUCTION = (
    "Write Python code only. Define solve(x, y) with one return expression. "
    "Use integers, x, y, +, -, *, //, %, comparisons, conditional expressions, abs, min, max only. "
    "Do not use imports, loops, assignments, annotations, decorators, or other calls. "
    "All integer magnitudes must be at most 1000000000000."
)


class CodingEvaluationBackend:
    """一次生成对应一个任务。调用者负责模型绑定、预算和输入隔离。"""

    def __init__(self, generate: Callable[[str, int], str], output: Path,
                 on_call: Callable[[str], None]) -> None:
        self.generate, self.output, self.on_call = generate, output, on_call

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        if file_sha256(request.task_manifest.path) != request.task_manifest.sha256:
            fail("tasks", "任务清单哈希不匹配。")
        suite = TaskSuite.parse(read_json(request.task_manifest.path))
        self.output.mkdir(parents=True, exist_ok=False)
        rows, outcomes = [], []
        for task in suite.tasks:
            self.on_call(task.task_id)
            code = self.generate(TASK_INSTRUCTION + "\n\n" + task.prompt, suite.max_new_tokens)
            execution = execute_task(code, task)
            outcome = TaskOutcome(task.task_id, task.group, execution.status == "passed")
            row = {"outcome": to_json_value(outcome), "code": code, "execution": to_json_value(execution)}
            # 逐任务落盘。后续生成失败时，已经完成的证据仍然可读。
            write_json(self.output / f"task-{len(rows):06d}.json", row)
            rows.append(row)
            outcomes.append(outcome)
        metrics = tuple(MetricResult("pass_at_1_" + group,
                         sum(r.passed for r in outcomes if r.group == group) / sum(r.group == group for r in outcomes),
                         sum(r.group == group for r in outcomes), "一次生成通过全部测试的任务数 / 该组全部任务数；不支持的代码计为失败。")
                        for group in ("target", "retention"))
        evidence = write_json(self.output / "evaluation.json", {
            "schema_version": "coding-evaluation-1", "purpose": "engineering_validation",
            "request": to_json_value(request), "task_manifest_sha256": request.task_manifest.sha256,
            "generation": "greedy_one_completion", "prompt_prefix": TASK_INSTRUCTION,
            "executor": "bounded_integer_expression_v1", "metrics": to_json_value(metrics), "tasks": rows,
        })
        return EvaluationResult(metrics, evidence)
