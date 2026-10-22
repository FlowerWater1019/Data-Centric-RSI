"""解释受限整数表达式。模型文本不会传入 exec、eval 或外部进程。"""

import ast
from dataclasses import dataclass
import operator

from rsi.evaluation.coding_tasks import CodingTask


class UnsupportedCode(ValueError):
    """生成内容超出任务预先声明的语言范围。"""


@dataclass(frozen=True)
class ExecutionOutcome:
    status: str
    passed_cases: int
    total_cases: int
    detail: str | None


def extract_code(text: str) -> str:
    if len(text) > 8192:
        raise UnsupportedCode("代码超过 8192 字符。")
    if "```" not in text:
        return text.strip()
    parts = text.split("```")
    if len(parts) != 3 or parts[0].strip() or parts[2].strip():
        raise UnsupportedCode("只允许一个代码块，且块外不能有文字。")
    lines = parts[1].splitlines()
    if not lines or lines[0].strip() not in {"", "python"}:
        raise UnsupportedCode("代码块语言必须是 python。")
    return "\n".join(lines[1:]).strip()


def parse_expression(text: str) -> ast.expr:
    tree = ast.parse(extract_code(text))
    if len(list(ast.walk(tree))) > 256 or len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise UnsupportedCode("只允许一个有限长度的函数。")
    function = tree.body[0]
    args = function.args
    if (function.name != "solve" or function.decorator_list or function.returns or args.defaults
            or args.kw_defaults or args.kwonlyargs or args.posonlyargs or args.vararg or args.kwarg
            or [a.arg for a in args.args] != ["x", "y"] or any(a.annotation for a in args.args)):
        raise UnsupportedCode("函数签名必须为 solve(x, y)，且没有注解或装饰器。")
    body = function.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        raise UnsupportedCode("函数体必须只有一条 return，可带文档字符串。")
    expression = body[0].value
    # 整棵树先检查，避免未执行分支携带不支持的语法。
    allowed = (ast.Constant, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp, ast.Compare, ast.IfExp,
               ast.Call, ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd,
               ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
    for node in ast.walk(expression):
        if not isinstance(node, allowed):
            raise UnsupportedCode("表达式包含不支持的语法。")
        if isinstance(node, ast.Name) and node.id not in {"x", "y", "abs", "min", "max"}:
            raise UnsupportedCode("表达式引用了未允许的名称。")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name)
                or node.func.id not in {"abs", "min", "max"} or node.keywords
                or len(node.args) != (1 if node.func.id == "abs" else 2)):
            raise UnsupportedCode("只允许 abs(一个参数)、min/max(两个参数)。")
        if isinstance(node, ast.Constant):
            bounded(node.value)
    return expression


def bounded(value: object) -> int:
    if type(value) is not int or abs(value) > 10**12:
        raise UnsupportedCode("整数超出范围或类型不受支持。")
    return value


def interpret(node: ast.expr, variables: dict[str, int], depth: int = 0) -> int | bool:
    if depth > 32:
        raise UnsupportedCode("表达式嵌套超过 32 层。")
    def child(value: ast.expr) -> int | bool:
        return interpret(value, variables, depth + 1)
    if isinstance(node, ast.Constant):
        return bounded(node.value)
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise UnsupportedCode("函数名不能作为返回值或数值。")
        return variables[node.id]
    if isinstance(node, ast.BinOp):
        operation = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                     ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}[type(node.op)]
        return bounded(operation(bounded(child(node.left)), bounded(child(node.right))))
    if isinstance(node, ast.UnaryOp):
        value = bounded(child(node.operand))
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.Compare):
        left = bounded(child(node.left))
        operations = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
                      ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
        for op, right_node in zip(node.ops, node.comparators):
            right = bounded(child(right_node))
            if not operations[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        return child(node.body if child(node.test) else node.orelse)
    if isinstance(node, ast.Call):
        values = [bounded(child(argument)) for argument in node.args]
        return bounded({"abs": abs, "min": min, "max": max}[node.func.id](*values))
    raise UnsupportedCode("表达式不受支持。")


def execute_task(code: str, task: CodingTask) -> ExecutionOutcome:
    """在整数解释器中执行全部测试。失败任务仍属于 pass@1 的分母。"""
    passed = 0
    try:
        expression = parse_expression(code)
        for test in task.tests:
            value = interpret(expression, dict(zip(("x", "y"), test.arguments)))
            passed += type(value) is int and value == test.expected
    except (UnsupportedCode, SyntaxError, RecursionError, ValueError) as exc:
        return ExecutionOutcome("unsupported", passed, len(task.tests), str(exc))
    except ArithmeticError as exc:
        return ExecutionOutcome("runtime_error", passed, len(task.tests), str(exc))
    return ExecutionOutcome("passed" if passed == len(task.tests) else "failed", passed, len(task.tests), None)
