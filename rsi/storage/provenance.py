"""递归记录源码内容。相对路径参与身份，避免子包重名与漏记。"""

from pathlib import Path

from rsi._paths import PACKAGE_ROOT
from rsi.storage.artifacts import file_sha256


def source_manifest(root: Path | None = None) -> list[dict[str, str]]:
    """返回全部 Python 源码的稳定清单。默认根目录为当前 rsi 包。"""
    source_root = (root or PACKAGE_ROOT).resolve()
    return [{"name": path.relative_to(source_root).as_posix(), "sha256": file_sha256(path)}
            for path in sorted(source_root.rglob("*.py"))
            if "__pycache__" not in path.relative_to(source_root).parts]
