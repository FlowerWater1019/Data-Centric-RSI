"""严格解析输入。错误消息包含字段位置。"""

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, NoReturn


class ValidationError(ValueError):
    """输入不符合实验约定。"""


class BudgetError(ValidationError):
    """完整轨迹组批未达到监督预算。"""


class QuotaError(ValidationError):
    """已组批的成员未通过配额检查。"""


def fail(path: str, message: str) -> NoReturn:
    raise ValidationError(f"{path}: {message}")


def obj(value: Any, path: str, required: set[str], optional: set[str] | None = None) -> dict:
    if not isinstance(value, dict):
        fail(path, "必须是对象。")
    missing = required - value.keys()
    extra = value.keys() - required - (optional or set())
    if missing:
        fail(path, f"缺少字段：{', '.join(sorted(missing))}。")
    if extra:
        fail(path, f"存在未知字段：{', '.join(sorted(extra))}。")
    return value


def string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        fail(path, "必须是非空字符串。")
    return value


def choice(value: Any, path: str, options: set[str]) -> str:
    value = string(value, path)
    if value not in options:
        fail(path, f"允许值：{', '.join(sorted(options))}。")
    return value


def integer(value: Any, path: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        fail(path, f"必须是大于或等于 {minimum} 的整数。")
    return value


def number(value: Any, path: str, minimum: float = 0, maximum: float | None = None) -> float:
    if type(value) not in (int, float):
        fail(path, "必须是有限数值。")
    try:
        converted = float(value)
    except OverflowError:
        fail(path, "数值超过浮点表示范围。")
    if not math.isfinite(converted):
        fail(path, "必须是有限数值。")
    if value < minimum or (maximum is not None and value > maximum):
        fail(path, "数值超出允许范围。")
    return converted


def boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        fail(path, "必须是布尔值。")
    return value


def array(value: Any, path: str, nonempty: bool = False) -> list:
    if not isinstance(value, list) or (nonempty and not value):
        fail(path, "必须是列表，且满足非空要求。")
    return value


def strings(value: Any, path: str, nonempty: bool = False) -> tuple[str, ...]:
    result = tuple(string(x, f"{path}[{i}]") for i, x in enumerate(array(value, path, nonempty)))
    if len(result) != len(set(result)):
        fail(path, "列表不能包含重复项。")
    return result


def to_json_value(value: Any) -> Any:
    """生成独立的 JSON 数据树。支持只读映射和字段明确的数据类。"""
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    return value


def canonical(value: Any) -> str:
    return json.dumps(to_json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            fail("JSON", f"字段重复：{key}。")
        result[key] = value
    return result


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        fail("JSON.number", "指数数值超出有限浮点范围。")
    return value


def loads(text: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_pairs, parse_float=_finite_float,
                          parse_constant=lambda x: fail("JSON", f"禁止数值 {x}。"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"JSON 第 {exc.lineno} 行：{exc.msg}") from exc
    except ValidationError:
        raise
    except ValueError as exc:
        raise ValidationError(f"JSON.number: 数值解析失败：{exc}") from exc


def read_json(path: str | Path) -> Any:
    return loads(_read_text(path))


def _read_text(path: str | Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8-sig")
    except UnicodeError as exc:
        raise ValidationError(f"{path}: 文件必须使用 UTF-8 编码。") from exc


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(_read_text(path).splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(loads(line))
        except ValidationError as exc:
            raise ValidationError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        fail(str(path), "数据文件为空。")
    return records
