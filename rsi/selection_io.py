"""选择产物的单写者发布。清单最后写入，再重命名目录。"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any, Iterator
from uuid import uuid4

from .artifacts import file_sha256, write_json
from .contracts import ExperimentContract
from .data import TrajectoryRecord, validate_dataset
from .history import BatchHistory
from .labels import DataLabels, SkillOntology, parse_labels
from .leases import run_lease
from .policy import SelectionPolicy, parse_policy
from .validation import digest, fail, read_json, read_jsonl


@dataclass(frozen=True)
class SelectionInputs:
    contract: ExperimentContract
    records: tuple[TrajectoryRecord, ...]
    labels: tuple[DataLabels, ...]
    ontology: SkillOntology
    policy: SelectionPolicy
    history: BatchHistory


def load_inputs(config: str | Path, data: str | Path, labels: str | Path, ontology: str | Path,
                policy: str | Path, history: str | Path | None = None) -> SelectionInputs:
    """读取独立输入。所有旧轨迹仍通过原解析器。"""
    contract = ExperimentContract.parse(read_json(config))
    skills = SkillOntology.parse(read_json(ontology))
    return SelectionInputs(contract, validate_dataset(read_jsonl(data), contract),
                           tuple(parse_labels(raw, skills) for raw in read_jsonl(labels)), skills,
                           parse_policy(read_json(policy), skills),
                           BatchHistory.parse(read_json(history)) if history else BatchHistory())


def load_fixture(root: str | Path) -> SelectionInputs:
    root = Path(root)
    verify_manifest(root)
    return load_inputs(root / "config.json", root / "trajectories.jsonl", root / "labels.jsonl",
                       root / "ontology.json", root / "policy.json", root / "batch_history.json")


def finish_manifest(root: Path, kind: str) -> dict[str, Any]:
    files = {p.relative_to(root).as_posix(): file_sha256(p) for p in sorted(root.rglob("*"))
             if p.is_file() and p != root / "manifest.json"}
    result = {"schema_version": "selection-artifacts-v1", "kind": kind, "files": files,
              "content_id": digest(files), "training_status": "NOT_RUN", "task_execution_status": "NOT_RUN"}
    write_json(root / "manifest.json", result)
    return result


def verify_manifest(root: Path) -> dict[str, Any]:
    """核验所有清单内容。拒绝目录外路径和不完整发布。"""
    from .labels import fields
    from .validation import choice, string
    value = fields(read_json(root / "manifest.json"), "manifest",
                   "schema_version kind files content_id training_status task_execution_status")
    choice(value["schema_version"], "manifest.schema_version", {"selection-artifacts-v1"})
    choice(value["training_status"], "manifest.training_status", {"NOT_RUN"})
    choice(value["task_execution_status"], "manifest.task_execution_status", {"NOT_RUN"})
    string(value["kind"], "manifest.kind")
    if not isinstance(value["files"], dict) or digest(value["files"]) != value["content_id"]:
        fail("manifest.files", "文件清单身份不一致。")
    for name, expected in value["files"].items():
        target = (root / name).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file() or file_sha256(target) != expected:
            fail("manifest.files." + name, "文件越界、缺失或哈希不匹配。")
    if any(p.is_symlink() for p in root.rglob("*")):
        fail("manifest.files", "输入目录不允许符号链接。")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != root / "manifest.json"}
    if actual != set(value["files"]):
        fail("manifest.files", "存在未登记文件。")
    return value


@contextmanager
def publish_directory(output: str | Path) -> Iterator[Path]:
    """只发布完整新目录。失败清理仅限本次创建的同级临时目录。"""
    target = Path(output).resolve()
    if target.exists():
        fail("output", "输出目录已存在，拒绝覆盖。")
    target.parent.mkdir(parents=True, exist_ok=True)
    with run_lease(target.parent):
        if target.exists():
            fail("output", "输出目录已存在，拒绝覆盖。")
        stage = target.parent / ("." + target.name + ".pending-" + uuid4().hex)
        stage.mkdir()
        try:
            yield stage
            if not (stage / "manifest.json").is_file():
                fail("output.manifest", "发布前必须完成清单。")
            verify_manifest(stage)
            stage.rename(target)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
