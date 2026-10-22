"""静态标记 Python 关键字片段。标签表示语法位置，不证明具体能力。"""

from copy import deepcopy
import io
import tokenize
from typing import Any

from rsi.configuration.validation import fail


SKILLS = ("python_control_keyword", "python_return_keyword")
CONTROL = {"if", "elif", "else", "for", "while", "try", "except", "finally", "raise", "with", "break", "continue"}


def annotate_keywords(source: dict[str, Any]) -> dict[str, Any]:
    """返回来源副本。只标记受监督 assistant 的代码，不改文本或训练掩码。"""
    result = deepcopy(source)
    for message in result["messages"]:
        if message["role"] != "assistant" or not message["supervise"]:
            continue
        if message["diagnostics"]:
            fail("diagnostics", "自动关键字标注要求原诊断片段为空。")
        code = message["content"]
        lines = code.splitlines(keepends=True)
        starts, total = [], 0
        for line in lines:
            starts.append(total)
            total += len(line)
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
        except (tokenize.TokenError, IndentationError) as exc:
            fail("diagnostics", "Python 词法分析失败：" + str(exc))
        for token in tokens:
            if token.type != tokenize.NAME or token.string not in CONTROL | {"return"}:
                continue
            skill = SKILLS[1] if token.string == "return" else SKILLS[0]
            start = starts[token.start[0] - 1] + token.start[1]
            end = starts[token.end[0] - 1] + token.end[1]
            # 保留前导空格，允许 BPE 的空格前缀 token 完整落入诊断范围。
            while start > 0 and code[start - 1] in " \t":
                start -= 1
            message["diagnostics"].append({"skill": skill, "start": start, "end": end})
            if skill not in result["skills"]:
                result["skills"].append(skill)
    return result
