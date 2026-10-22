"""跨训练来源与测评任务的内容身份。只检查已知文本重叠，不推断语义等价。"""

import ast
import unicodedata
from typing import Any

from rsi.configuration.validation import canonical, choice, digest, fail, string


CONTENT_RULE = "task-content-v1"


def normalized_text(value: str) -> str:
    """统一 Unicode、换行与空白。保留非空白字符和顺序。"""
    return " ".join(unicodedata.normalize("NFC", value).split())


def problem_fingerprint(prompt: str) -> str:
    return digest({"rule": "problem-public-text-v1", "prompt": normalized_text(string(prompt, "content.prompt"))})


def normalized_python(value: str) -> str:
    """规范化 Python 语法树，不执行代码。注释和排版不改变身份。"""
    try:
        return ast.dump(ast.parse(value), annotate_fields=True, include_attributes=False)
    except (SyntaxError, ValueError, RecursionError) as exc:
        fail("content.python", "无法规范化环境或测试代码：" + str(exc))


def task_fingerprint(adapter_id: str, payload: dict[str, Any]) -> str:
    """绑定真实公开任务、初始环境和判定器；编号与技能标签不参与内容身份。"""
    choice(adapter_id, "content.adapter_id", {"bounded_integer_expression_v1", "bubblewrap_seccomp_v1", "agent_environment_v1"})
    prompt = normalized_text(string(payload["prompt"], "content.prompt"))
    if adapter_id == "bounded_integer_expression_v1":
        tests = sorted({canonical(t) for t in payload["tests"]})
        content = {"prompt": prompt, "tests": tests}
    elif adapter_id == "bubblewrap_seccomp_v1":
        content = {"prompt": prompt, "entry_point": payload["entry_point"], "setup": normalized_python(payload["setup"]),
                   "tests": sorted({normalized_python(t) for t in payload["tests"]})}
    else:
        content = {**payload, "prompt": prompt}
    return digest({"rule": CONTENT_RULE, "adapter_id": adapter_id, "adapter_version": "1", "content": content})


def source_problem_fingerprints(source: dict[str, Any]) -> set[str]:
    messages = source.get("messages", [])
    prompts = [m["content"] for m in messages if m.get("role") == "user" and normalized_text(m.get("content", ""))]
    return {problem_fingerprint(p) for p in prompts} | ({problem_fingerprint("\n".join(prompts))} if prompts else set())
