"""单模型自由执行与完整分母。失败现场保留，不自动重跑未知调用。"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rsi.storage.artifacts import write_json
from rsi.evaluation.bounded_python import UnsupportedCode, execute_task, extract_code
from rsi.evaluation.coding_evaluation import TASK_INSTRUCTION
from rsi.configuration.contracts import ModelRef
from rsi.datasets.data import TrajectoryRecord
from rsi.storage.execution_artifacts import complete_execution, verify_execution
from rsi.evaluation.isolated_evaluation import PYTHON_INSTRUCTION
from rsi.execution.sandbox import BubblewrapExecutor
from rsi.storage.state import StateStore
from rsi.evaluation.task_adapters import TaskSuiteV2
from rsi.configuration.validation import choice, digest, fail, read_json, to_json_value


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    group: str
    skills: tuple[str, ...]
    status: str
    public_input: str
    generated: str | None
    seed: int
    error: str | None
    test_results_json: str


@dataclass(frozen=True)
class ExecutionSummary:
    planned: int
    attempted: int
    completed: int
    passed: int
    infrastructure_error: int
    not_run: int
    pass_rate: float | None


def summarize(rows: tuple[TaskResult, ...]) -> ExecutionSummary:
    attempted = sum(r.status not in {"NOT_RUN", "BUDGET_EXHAUSTED"} for r in rows)
    completed = sum(r.status in {"PASSED", "TASK_FAILED"} for r in rows)
    passed = sum(r.status == "PASSED" for r in rows)
    return ExecutionSummary(len(rows), attempted, completed, passed, sum(r.status == "INFRA_ERROR" for r in rows),
                            len(rows) - attempted, passed / len(rows) if rows and completed == len(rows) else None)


@dataclass(frozen=True)
class ExecutionEvaluation:
    schema_version: str
    model: ModelRef
    source: str
    purpose: str
    suite_hash: str
    generation_hash: str
    environment_hash: str
    status: str
    rows: tuple[TaskResult, ...]
    overall: ExecutionSummary
    by_group: tuple[tuple[str, ExecutionSummary], ...]
    by_skill: tuple[tuple[str, ExecutionSummary], ...]
    by_group_and_skill: tuple[tuple[str, str, ExecutionSummary], ...]

    def to_dict(self) -> dict[str, Any]:
        return to_json_value(self)


def verify_isolation(suite: TaskSuiteV2, records: tuple[TrajectoryRecord, ...], sources: list[dict[str, Any]] = ()) -> None:
    """检查已知身份重叠，不声称证明不存在语义污染。"""
    ids = {r.data_id for r in records} | {row["data_id"] for row in sources}
    groups = {r.split_group for r in records} | {r.duplicate_group for r in records}
    hashes = {r.content_hash for r in records}
    for task in suite.tasks:
        if task.task_id in ids or task.source_problem_id in ids or task.split_group in groups or task.duplicate_group in groups or task.content_fingerprint in hashes:
            fail("suite.isolation", "任务与训练/评分来源存在已知重叠。")


def evaluate_execution(model: ModelRef, suite: TaskSuiteV2, generate: Callable[[str, int, int], str], output: str | Path,
                        call_budget: int, *, source: str, state: StateStore | None = None, run_id: str = "",
                        reservation: str = "execution", executor: BubblewrapExecutor | None = None,
                        records: tuple[TrajectoryRecord, ...] = (), sources: list[dict[str, Any]] = (),
                        stage_request: dict[str, Any] | None = None) -> ExecutionEvaluation:
    """一次贪心生成。隐藏测试仅送解释器/隔离判分器，不送 generate。"""
    from rsi.configuration.validation import canonical, integer
    choice(source, "execution.source", {"measured", "synthetic_fixture"})
    integer(call_budget, "execution.call_budget")
    verify_isolation(suite, records, sources)
    root = Path(output)
    identity = {"model": to_json_value(model), "suite": digest(suite), "source": source,
                "environment": digest({"id": suite.environment_id, "limits": to_json_value(executor.limits) if executor else None})}
    if root.exists():
        fail("execution.output", "已存在执行目录，使用完整阶段复用入口；禁止自动重复调用。")
    root.mkdir(parents=True)
    write_json(root / "request.json", identity)
    write_json(root / "suite.json", suite.to_dict())
    if stage_request is not None:
        write_json(root / "stage_request.json", stage_request)
    preflight = None
    if any(t.adapter_id == "bubblewrap_seccomp_v1" for t in suite.tasks):
        if executor is None:
            fail("execution.executor", "Python 任务必须提供隔离执行器，不能退回宿主执行。")
        try:
            preflight = executor.preflight()
            write_json(root / "preflight.json", preflight)
        except BaseException as exc:
            write_json(root / "infrastructure_failure.json", {"error": str(exc), "status": "INFRA_ERROR"})
            raise
    rows = []
    blocked = False
    for index, task in enumerate(suite.tasks):
        seed = int(digest({"seed": suite.generation.seed, "task_id": task.task_id})[:8], 16)
        prompt = (TASK_INSTRUCTION if task.adapter_id == "bounded_integer_expression_v1" else PYTHON_INSTRUCTION) + "\n\n" + task.payload.prompt
        status, generated, error, checks = "NOT_RUN", None, None, []
        if not blocked and index >= call_budget:
            status = "BUDGET_EXHAUSTED"
        elif not blocked:
            write_json(root / f"{index:06d}.intent.json", {"task_id": task.task_id, "seed": seed, "public_input": prompt, **identity})
            try:
                if state:
                    # 调用意图已落盘。即使模型随后失败，调用次数也不归零。
                    state.consume(run_id, reservation + ":" + task.task_id, reservation, 1)
                generated = generate(prompt, suite.generation.max_new_tokens, seed)
                if not isinstance(generated, str):
                    generated = None
                    fail("generation", "生成器必须返回文本。")
                write_json(root / f"{index:06d}.generated.json", {"task_id": task.task_id, "generated": generated})
                if task.adapter_id == "bounded_integer_expression_v1":
                    result = execute_task(generated, task.payload)
                    checks.append(to_json_value(result))
                    status = "PASSED" if result.status == "passed" else "TASK_FAILED"
                else:
                    try:
                        code = extract_code(generated)
                    except UnsupportedCode as exc:
                        status, error = "TASK_FAILED", str(exc)
                    else:
                        for test in task.payload.tests:
                            result = executor.run(code, test, task.payload.setup)
                            checks.append(to_json_value(result))
                        status = "PASSED" if all(x["status"] == "passed" for x in checks) else "TASK_FAILED"
            except Exception as exc:
                # 保留失败事实和已发生调用；不生成成功壳子，不重试未知消耗。
                status, error, blocked = "INFRA_ERROR", str(exc) or type(exc).__name__, True
        row = TaskResult(task.task_id, task.group, task.skills, status, prompt, generated, seed, error, canonical(checks))
        write_json(root / f"{index:06d}.result.json", to_json_value(row))
        rows.append(row)
    rows = tuple(rows)
    overall = summarize(rows)
    groups = sorted({t.group for t in suite.tasks})
    skills = sorted({s for t in suite.tasks for s in t.skills})
    report = ExecutionEvaluation("execution-evaluation-v1", model, source, suite.purpose, digest(suite), digest(suite.generation),
                identity["environment"], "COMPLETED" if overall.completed == overall.planned else "PARTIAL", rows, overall,
                tuple((g, summarize(tuple(r for r in rows if r.group == g))) for g in groups),
                tuple((s, summarize(tuple(r for r in rows if s in r.skills))) for s in skills),
                tuple((g, s, summarize(tuple(r for r in rows if r.group == g and s in r.skills))) for g in groups for s in skills))
    write_json(root / "evaluation.json", report.to_dict())
    complete_execution(root, report.status)
    return report


def parse_execution_evaluation(raw: Any) -> ExecutionEvaluation:
    """从逐题事实重算完整分母，拒绝占位字段和矛盾摘要。"""
    from rsi.datasets.labels import fields
    from rsi.configuration.validation import array, integer, strings
    v = fields(raw, "execution", "schema_version model source purpose suite_hash generation_hash environment_hash status rows overall by_group by_skill by_group_and_skill")
    choice(v["schema_version"], "execution.schema_version", {"execution-evaluation-v1"})
    purpose = choice(v["purpose"], "execution.purpose", {"feedback", "final_test"})
    rows = []
    for item in array(v["rows"], "execution.rows", True):
        item = fields(item, "execution.row", "task_id group skills status public_input generated seed error test_results_json")
        from rsi.configuration.validation import string, loads
        for key in ("task_id", "group", "public_input", "test_results_json"):
            string(item[key], "execution.row." + key)
        for key in ("generated", "error"):
            if item[key] is not None:
                if not isinstance(item[key], str):
                    fail("execution.row." + key, "必须为文本或 null。")
        choice(item["group"], "execution.row.group", {"target", "retention"} if purpose == "feedback" else {"final_test"})
        checks = array(loads(item["test_results_json"]), "execution.test_results")
        for check in checks:
            if not isinstance(check, dict):
                fail("execution.test_results", "判分事实必须为对象。")
            string(check.get("status"), "execution.test_results.status")
        if item["status"] in {"PASSED", "TASK_FAILED"} and not isinstance(item["generated"], str):
            fail("execution.row.generated", "完成判分的任务必须保存生成文本。")
        if item["status"] == "PASSED" and (not checks or any(c["status"] != "passed" for c in checks)):
            fail("execution.row.status", "PASSED 必须具有全部通过的非空判分事实。")
        if item["status"] == "PASSED" and item["error"] is not None:
            fail("execution.row.error", "通过任务不能同时报告错误。")
        if item["status"] == "TASK_FAILED" and checks and all(c["status"] == "passed" for c in checks):
            fail("execution.row.status", "全部判分事实通过的任务不能标为失败。")
        if item["status"] in {"NOT_RUN", "BUDGET_EXHAUSTED"} and (item["generated"] is not None or item["error"] is not None or checks):
            fail("execution.row.status", "未调用的任务不能包含生成、错误或判分事实。")
        if item["status"] == "INFRA_ERROR":
            string(item["error"], "execution.row.error")
        rows.append(TaskResult(item["task_id"], item["group"], strings(item["skills"], "execution.skills"),
                    choice(item["status"], "execution.status", {"PASSED", "TASK_FAILED", "INFRA_ERROR", "BUDGET_EXHAUSTED", "NOT_RUN"}),
                    item["public_input"], item["generated"], integer(item["seed"], "execution.seed"), item["error"], item["test_results_json"]))
    if len({r.task_id for r in rows}) != len(rows):
        fail("execution.task_id", "任务重复。")
    rows = tuple(rows)
    groups, skills = sorted({r.group for r in rows}), sorted({s for r in rows for s in r.skills})
    if purpose == "feedback" and set(groups) != {"target", "retention"}:
        fail("execution.rows.group", "反馈报告必须保留 target 和 retention 的完整分母。")
    overall = summarize(rows)
    report = ExecutionEvaluation(v["schema_version"], ModelRef.parse(v["model"]),
                choice(v["source"], "execution.source", {"measured", "synthetic_fixture"}),
                choice(v["purpose"], "execution.purpose", {"feedback", "final_test"}),
                *(string(v[k], "execution." + k) for k in ("suite_hash", "generation_hash", "environment_hash")),
                "COMPLETED" if overall.completed == overall.planned else "PARTIAL", rows, overall,
                tuple((g, summarize(tuple(r for r in rows if r.group == g))) for g in groups),
                tuple((s, summarize(tuple(r for r in rows if s in r.skills))) for s in skills),
                tuple((g, s, summarize(tuple(r for r in rows if r.group == g and s in r.skills))) for g in groups for s in skills))
    if report.to_dict() != v:
        fail("execution.summary", "逐题事实与总体/分组摘要不一致。")
    return report
