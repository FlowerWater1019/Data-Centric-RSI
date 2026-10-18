"""通用 Python 计算任务格式。测试代码只在隔离执行器内运行。"""

import ast
from dataclasses import dataclass
from typing import Any

from .validation import array, choice, fail, integer, obj, string


@dataclass(frozen=True)
class PythonTask:
    task_id: str
    group: str
    split_group: str
    duplicate_group: str
    prompt: str
    entry_point: str
    setup: str
    tests: tuple[str, ...]

    @classmethod
    def parse(cls, value: Any) -> "PythonTask":
        raw = obj(value, "task", {"task_id", "group", "split_group", "duplicate_group", "prompt", "entry_point", "setup", "tests"})
        texts = {key: string(raw[key], "task." + key) for key in
                 ("task_id", "split_group", "duplicate_group", "prompt", "entry_point")}
        if not texts["entry_point"].isidentifier():
            fail("task.entry_point", "必须是 Python 标识符。")
        if not isinstance(raw["setup"], str):
            fail("task.setup", "必须是字符串，可以为空。")
        tests = tuple(string(test, "task.tests") for test in array(raw["tests"], "task.tests", True))
        if len(tests) > 100 or any(len(text.encode()) > 65536 for text in (*tests, raw["setup"], texts["prompt"])):
            fail("task", "单段文本最多 64 KiB，每个任务最多 100 个测试。")
        for test in tests:
            try:
                tree = ast.parse(test)
            except (SyntaxError, ValueError, RecursionError) as exc:
                fail("task.tests", "测试语法无效：" + str(exc))
            if not tree.body or not all(isinstance(node, ast.Assert) for node in tree.body):
                fail("task.tests", "每个测试必须只包含 assert 语句。")
        return cls(texts["task_id"], choice(raw["group"], "task.group", {"target", "retention"}),
                   texts["split_group"], texts["duplicate_group"], texts["prompt"], texts["entry_point"], raw["setup"], tests)


@dataclass(frozen=True)
class PythonTaskSuite:
    tasks: tuple[PythonTask, ...]
    max_new_tokens: int
    provenance: dict[str, Any]

    @classmethod
    def parse(cls, value: Any) -> "PythonTaskSuite":
        raw = obj(value, "tasks", {"schema_version", "purpose", "role", "executor", "max_new_tokens", "provenance", "tasks"})
        choice(raw["schema_version"], "tasks.schema_version", {"python-tasks-1"})
        choice(raw["purpose"], "tasks.purpose", {"engineering_validation"})
        choice(raw["role"], "tasks.role", {"feedback_execution"})
        choice(raw["executor"], "tasks.executor", {"bubblewrap_seccomp_v1"})
        limit = integer(raw["max_new_tokens"], "tasks.max_new_tokens", 1)
        if limit > 2048:
            fail("tasks.max_new_tokens", "最多为 2048。")
        provenance = obj(raw["provenance"], "tasks.provenance", {"dataset", "revision", "sha256", "license", "split_policy"})
        for key in provenance:
            string(provenance[key], "tasks.provenance." + key)
        tasks = tuple(PythonTask.parse(item) for item in array(raw["tasks"], "tasks.tasks", True))
        for key in ("task_id", "split_group", "duplicate_group"):
            if len({getattr(task, key) for task in tasks}) != len(tasks):
                fail("tasks." + key, "任务编号或分组重复。")
        if {task.group for task in tasks} != {"target", "retention"}:
            fail("tasks", "必须同时具有目标组和能力保持组。")
        return cls(tasks, limit, dict(provenance))
