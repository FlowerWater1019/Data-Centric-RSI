"""完整 Python 代码评估。生成器不接收测试答案或参考解。"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .acceptance import TaskOutcome
from .artifacts import file_sha256, write_json
from .backends import EvaluationRequest, EvaluationResult, MetricResult
from .bounded_python import UnsupportedCode, extract_code
from .python_tasks import PythonTaskSuite
from .sandbox import BubblewrapExecutor
from .validation import fail, read_json, to_json_value


PYTHON_INSTRUCTION = (
    "Return complete Python code only, optionally in one python code block. "
    "Use the requested function signature and the Python standard library. "
    "Do not read stdin, write files, access the network, or start processes or threads."
)


class IsolatedEvaluationBackend:
    """每次生成对应一个任务。每个测试使用全新的隔离进程。"""

    def __init__(self, generate: Callable[[str, int], str], output: Path,
                 on_call: Callable[[str], None], executor: BubblewrapExecutor,
                 preflight: dict[str, Any]) -> None:
        self.generate, self.output, self.on_call = generate, output, on_call
        self.executor, self.preflight = executor, preflight

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        if file_sha256(request.task_manifest.path) != request.task_manifest.sha256:
            fail("tasks", "任务清单哈希不匹配。")
        suite = PythonTaskSuite.parse(read_json(request.task_manifest.path))
        self.output.mkdir(parents=True, exist_ok=False)
        write_json(self.output / "preflight.json", self.preflight)
        rows, outcomes = [], []
        for index, task in enumerate(suite.tasks):
            self.on_call(task.task_id)
            generated = self.generate(PYTHON_INSTRUCTION + "\n\n" + task.prompt, suite.max_new_tokens)
            write_json(self.output / f"task-{index:06d}.generated.json", {"task_id": task.task_id, "code": generated})
            checks, format_error = [], None
            try:
                code = extract_code(generated)
            except UnsupportedCode as exc:
                format_error = str(exc)
            else:
                for test_index, test in enumerate(task.tests):
                    write_json(self.output / f"task-{index:06d}-test-{test_index:03d}.started.json", {"task_id": task.task_id})
                    check = self.executor.run(code, test, task.setup)
                    write_json(self.output / f"task-{index:06d}-test-{test_index:03d}.result.json", to_json_value(check))
                    checks.append(check)
            passed = format_error is None and len(checks) == len(task.tests) and all(check.status == "passed" for check in checks)
            outcome = TaskOutcome(task.task_id, task.group, passed)
            row = {"outcome": to_json_value(outcome), "code": generated, "format_error": format_error,
                   "test_results": to_json_value(checks), "test_count": len(task.tests)}
            write_json(self.output / f"task-{index:06d}.json", row)
            rows.append(row)
            outcomes.append(outcome)
        metrics = tuple(MetricResult("pass_at_1_" + group,
                         sum(row.passed for row in outcomes if row.group == group) / sum(row.group == group for row in outcomes),
                         sum(row.group == group for row in outcomes), "一次生成通过全部隔离测试的任务数 / 该组全部任务数。")
                        for group in ("target", "retention"))
        evidence = write_json(self.output / "evaluation.json", {
            "schema_version": "isolated-evaluation-1", "purpose": "engineering_validation", "data_kind": "real_public",
            "request": to_json_value(request), "task_manifest_sha256": request.task_manifest.sha256,
            "generation": "greedy_one_completion", "prompt_prefix": PYTHON_INSTRUCTION,
            "executor": "bubblewrap_seccomp_v1", "metrics": to_json_value(metrics), "tasks": rows,
            "test_calls_started": sum(len(row["test_results"]) for row in rows), "provenance": suite.provenance})
        return EvaluationResult(metrics, evidence)
