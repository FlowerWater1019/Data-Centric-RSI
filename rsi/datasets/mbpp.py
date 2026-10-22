"""MBPP 原始 JSONL 适配。官方划分先于工程角色分配。"""

import ast
from dataclasses import dataclass
import re
from typing import Any

from rsi.configuration.validation import array, fail, integer, obj, string, text_digest


@dataclass(frozen=True)
class MbppRecord:
    task_id: int
    prompt: str
    code: str
    setup: str
    tests: tuple[str, ...]
    challenge_tests: tuple[str, ...]

    @classmethod
    def parse(cls, value: Any) -> "MbppRecord":
        raw = obj(value, "mbpp", {"task_id", "text", "code", "test_setup_code", "test_list", "challenge_test_list"})
        task_id = integer(raw["task_id"], "mbpp.task_id", 1)
        if task_id > 974:
            fail("mbpp.task_id", "此版本只支持官方原始 MBPP 的 1 至 974。")
        if not isinstance(raw["test_setup_code"], str):
            fail("mbpp.test_setup_code", "必须是字符串。")
        prompt, code = string(raw["text"], "mbpp.text"), string(raw["code"], "mbpp.code")
        tests = tuple(string(test, "mbpp.test_list") for test in array(raw["test_list"], "mbpp.test_list", True))
        challenges = tuple(string(test, "mbpp.challenge_test_list") for test in array(raw["challenge_test_list"], "mbpp.challenge_test_list"))
        if any(len(text.encode()) > 65536 for text in (prompt, code, raw["test_setup_code"], *tests, *challenges)):
            fail("mbpp", "单段内容超过 64 KiB。")
        return cls(task_id, prompt, code, raw["test_setup_code"], tests, challenges)

    @property
    def official_split(self) -> str:
        if self.task_id <= 10:
            return "prompting"
        if self.task_id <= 510:
            return "test"
        if self.task_id <= 600:
            return "validation"
        return "train"

    def signature(self) -> tuple[str, str]:
        try:
            code = ast.parse(self.code)
            calls = {node.func.id for test in self.tests for node in ast.walk(ast.parse(test))
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        except (SyntaxError, ValueError, RecursionError) as exc:
            fail("mbpp.signature", "无法解析参考代码或测试：" + str(exc))
        functions = [node for node in code.body if isinstance(node, ast.FunctionDef) and node.name in calls]
        if len(functions) != 1:
            fail("mbpp.signature", "不能确定唯一测试入口。")
        function = functions[0]
        return function.name, f"def {function.name}({ast.unparse(function.args)}):"


def duplicate_groups(records: tuple[MbppRecord, ...]) -> dict[int, str]:
    """连接相同规范化提示或相同代码语法树。该规则不等同于语义去重。"""
    parent = {record.task_id: record.task_id for record in records}
    def find(task_id: int) -> int:
        while parent[task_id] != task_id:
            parent[task_id] = parent[parent[task_id]]
            task_id = parent[task_id]
        return task_id
    seen = {}
    for record in records:
        prompt_key = "prompt:" + text_digest(re.sub(r"\s+", " ", record.prompt.strip()).casefold())
        try:
            normalized = ast.dump(ast.parse(record.code), include_attributes=False)
        except (SyntaxError, ValueError, RecursionError):
            normalized = record.code.strip()
        keys = (prompt_key, "code:" + text_digest(normalized))
        for key in keys:
            if key in seen:
                first, second = find(record.task_id), find(seen[key])
                parent[max(first, second)] = min(first, second)
            seen[key] = record.task_id
    return {record.task_id: "mbpp_group_" + str(find(record.task_id)) for record in records}


def engineering_role(record: MbppRecord, group: str) -> str:
    if record.official_split in {"test", "prompting"}:
        return "final_test" if record.official_split == "test" else "excluded_prompting"
    bucket = int(text_digest(group)[:8], 16)
    if record.official_split == "validation":
        return "target" if bucket % 2 == 0 else "retention"
    return {0: "reference", 1: "feedback_diagnostic"}.get(bucket % 10, "candidate")
