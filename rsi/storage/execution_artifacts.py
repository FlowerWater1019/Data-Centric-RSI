"""有实际消耗的执行产物。失败现场与账本不进行整体删除。"""

from pathlib import Path
from typing import Any

from rsi.storage.artifacts import file_sha256, verify_training_artifact, write_json
from rsi.models.backends import ArtifactRef
from rsi.configuration.validation import digest, fail, read_json
from rsi.storage.provenance import source_manifest


EXECUTION_STATUSES = {"COMPILED_AND_SCORED", "COMPLETED", "PARTIAL", "VALIDATION_COMPLETED"}


def execution_files(root: Path) -> tuple[Path, ...]:
    """枚举必须纳入快照的文件。账本和进程锁另行校验。"""
    return tuple(p for p in sorted(root.rglob("*")) if p.is_file() and
        p.name not in {".writer.lock", "state.sqlite3", "state.sqlite3-wal", "state.sqlite3-shm"} and
        p != root / "completion_manifest.json")


def complete_execution(root: Path, status: str, files: tuple[Path, ...] | None = None) -> ArtifactRef:
    """只在阶段成功后写不可覆盖清单。SQLite 和锁不作为可变文件快照。"""
    from rsi.configuration.validation import choice
    choice(status, "execution_manifest.status", EXECUTION_STATUSES)
    paths = files if files is not None else execution_files(root)
    if set(paths) != set(execution_files(root)) or len(paths) != len(set(paths)):
        fail("execution_manifest.files", "完成清单必须覆盖全部运行快照文件，且不能重复。")
    entries = [{"name": p.relative_to(root).as_posix(), "sha256": file_sha256(p)} for p in paths]
    return write_json(root / "completion_manifest.json", {"schema_version": "execution-artifacts-v1", "status": status,
                       "content_id": digest(entries), "files": entries})


def verify_execution(root: str | Path, expected_id: str | None = None) -> dict[str, Any]:
    root = Path(root)
    path = root / "completion_manifest.json"
    from rsi.datasets.labels import fields
    from rsi.configuration.validation import array, choice, string
    value = fields(read_json(path), "execution_manifest", "schema_version status content_id files")
    choice(value["status"], "execution_manifest.status", EXECUTION_STATUSES)
    entries = array(value["files"], "execution_manifest.files", True)
    names = []
    for entry in entries:
        entry = fields(entry, "execution_manifest.file", "name sha256")
        names.append(string(entry["name"], "execution_manifest.file.name"))
        string(entry["sha256"], "execution_manifest.file.sha256")
    if len(set(names)) != len(names):
        fail("execution_manifest.files", "文件不能重复。")
    expected = {p.relative_to(root).as_posix() for p in execution_files(root)}
    if set(names) != expected:
        fail("execution_manifest.files", "清单遗漏文件，文件缺失，或目录含有未登记文件。")
    verify_training_artifact(ArtifactRef(str(path), file_sha256(path)))
    if value["schema_version"] != "execution-artifacts-v1" or value["content_id"] != digest(entries):
        fail("execution_manifest", "执行清单格式或身份无效。")
    if expected_id is not None and value["content_id"] != expected_id:
        fail("execution_manifest.content_id", "内容身份不匹配。")
    return value


def implementation_snapshot() -> dict[str, Any]:
    """记录源码哈希与实际依赖版本，不导入 GPU 包。"""
    import importlib.metadata
    versions = {}
    for name in ("torch", "transformers", "peft", "tokenizers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    import platform
    return {"files": {entry["name"]: entry["sha256"] for entry in source_manifest()},
            "versions": versions, "python": platform.python_version(), "platform": platform.platform()}


def implementation_identity() -> str:
    """阶段缓存绑定源码与实际软件环境。"""
    return digest(implementation_snapshot())
