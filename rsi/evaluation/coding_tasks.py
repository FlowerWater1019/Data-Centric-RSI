"""独立纯函数任务的冻结格式。当前格式仅用于工程验证。"""

from dataclasses import dataclass
from typing import Any

from rsi.configuration.validation import array, choice, fail, integer, obj, string


@dataclass(frozen=True)
class TestCase:
    arguments: tuple[int, int]
    expected: int


@dataclass(frozen=True)
class CodingTask:
    task_id: str
    group: str
    split_group: str
    duplicate_group: str
    prompt: str
    tests: tuple[TestCase, ...]


@dataclass(frozen=True)
class TaskSuite:
    tasks: tuple[CodingTask, ...]
    max_new_tokens: int

    @classmethod
    def parse(cls, value: Any) -> "TaskSuite":
        raw = obj(value, "tasks", {"schema_version", "purpose", "role", "executor", "max_new_tokens", "tasks"})
        choice(raw["schema_version"], "tasks.schema_version", {"coding-tasks-1"})
        choice(raw["purpose"], "tasks.purpose", {"engineering_validation"})
        choice(raw["role"], "tasks.role", {"feedback_execution"})
        choice(raw["executor"], "tasks.executor", {"bounded_integer_expression_v1"})
        limit = integer(raw["max_new_tokens"], "tasks.max_new_tokens", 1)
        if limit > 512:
            fail("tasks.max_new_tokens", "工程任务最多生成 512 个 token。")
        tasks = []
        for index, item in enumerate(array(raw["tasks"], "tasks.tasks", True)):
            path = f"tasks.tasks[{index}]"
            item = obj(item, path, {"task_id", "group", "split_group", "duplicate_group", "prompt", "tests"})
            tests = []
            for test in array(item["tests"], path + ".tests", True):
                test = obj(test, path + ".tests", {"arguments", "expected"})
                args = array(test["arguments"], path + ".arguments")
                if len(args) != 2:
                    fail(path, "每个测试必须提供两个整数参数。")
                values = [*args, test["expected"]]
                if any(type(x) is not int or abs(x) > 10**12 for x in values):
                    fail(path, "测试值必须是绝对值不大于 10^12 的整数。")
                tests.append(TestCase(tuple(args), test["expected"]))
            prompt = string(item["prompt"], path + ".prompt")
            if len(prompt) > 4096 or len(tests) > 100:
                fail(path, "提示最多 4096 字符，每任务最多 100 个测试。")
            tasks.append(CodingTask(string(item["task_id"], path + ".task_id"),
                                   choice(item["group"], path + ".group", {"target", "retention"}),
                                   string(item["split_group"], path + ".split_group"),
                                   string(item["duplicate_group"], path + ".duplicate_group"), prompt, tuple(tests)))
        if len(tasks) > 10000:
            fail("tasks", "工程任务数量超过 10000。")
        for field in ("task_id", "split_group", "duplicate_group", "prompt"):
            if len({getattr(t, field) for t in tasks}) != len(tasks):
                fail("tasks." + field, "任务必须独立，字段值不能重复。")
        if {t.group for t in tasks} != {"target", "retention"}:
            fail("tasks", "必须同时提供目标任务与能力保持任务。")
        return cls(tuple(tasks), limit)
