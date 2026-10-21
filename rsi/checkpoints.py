"""训练步骤日志。只恢复已发布检查点；不重放状态不明的步骤。"""

import os
import time
import importlib.metadata
from collections.abc import Callable
from pathlib import Path

from .artifacts import file_sha256, verify_training_artifact, write_json
from .backends import ArtifactRef, TrainingRequest
from .data import TrajectoryRecord
from .validation import digest, fail, read_json, to_json_value


class TrainingPaused(Exception):
    """训练已在完整步骤后停止。检查点和已记成本均保留。"""

    def __init__(self, artifact: ArtifactRef) -> None:
        super().__init__("训练已在完整检查点处暂停。")
        self.artifact = artifact


def request_identity(request: TrainingRequest, records: list[TrajectoryRecord], device: str) -> str:
    """冻结恢复所需的输入、代码和设备绑定。恢复产物本身不参与标识。"""
    value = to_json_value(request)
    value.pop("resume_artifact")
    versions = {}
    for name in ("torch", "transformers", "peft", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return digest({"request": value, "records": [r.raw for r in records], "device": device,
                   "versions": versions,
                   "cuda_binding": os.environ.get("CUDA_VISIBLE_DEVICES") if device != "cpu" else None,
                   "implementation": {p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}})


class StepJournal:
    """步骤意图先落盘，完整检查点后发布。目录发布只支持同一文件系统。"""

    def __init__(self, root: Path, identity: str) -> None:
        self.root, self.identity = root, identity

    def begin(self, step: int) -> None:
        write_json(self.root / "intents" / f"{step:06d}.json", {"step": step, "identity": self.identity})

    def publish(self, step: int, save_state: Callable[[Path], None]) -> ArtifactRef:
        """调用保存函数写入临时目录。返回后，检查点目录完整可读。"""
        parent = self.root / "checkpoints"
        parent.mkdir(exist_ok=True)
        pending, final = parent / f".{step:06d}.pending", parent / f"{step:06d}"
        pending.mkdir(exist_ok=False)
        save_state(pending / "state.pt")
        write_json(pending / "manifest.json", {"schema_version": "checkpoint-1", "step": step,
                   "identity": self.identity, "files": [{"name": "state.pt", "sha256": file_sha256(pending / "state.pt")}]})
        # Windows 文件扫描可能短暂锁住已关闭目录。只重试发布，不重做训练或保存。
        for attempt in range(4):
            if final.exists():
                fail("checkpoint.publish", "目标检查点已存在，禁止覆盖。")
            try:
                pending.rename(final)
                break
            except PermissionError as exc:
                if getattr(exc, "winerror", None) not in {5, 32} or attempt == 3:
                    raise
                time.sleep(0.05 * (attempt + 1))
        return ArtifactRef(str((final / "manifest.json").resolve()), file_sha256(final / "manifest.json"))

    def validate_resume(self, artifact: ArtifactRef) -> tuple[Path, int]:
        verify_training_artifact(artifact)
        path = Path(artifact.path).resolve()
        manifest = read_json(path)
        step = manifest.get("step")
        if (manifest.get("schema_version") != "checkpoint-1" or type(step) is not int or step < 1
                or manifest.get("identity") != self.identity):
            fail("resume", "检查点格式或冻结输入不一致。")
        if path != (self.root / "checkpoints" / f"{step:06d}" / "manifest.json").resolve():
            fail("resume", "检查点必须属于本次训练目录。")
        intents = sorted((self.root / "intents").glob("*.json"))
        expected = [f"{i:06d}.json" for i in range(1, step + 1)]
        if [p.name for p in intents] != expected:
            fail("resume", "存在未完成或缺失的步骤意图。成本可能未知；拒绝自动重放。")
        if any(read_json(p) != {"step": i, "identity": self.identity} for i, p in enumerate(intents, 1)):
            fail("resume", "步骤意图与检查点不一致。")
        checkpoints = sorted(p.name for p in (self.root / "checkpoints").iterdir())
        if checkpoints != [f"{i:06d}" for i in range(1, step + 1)]:
            fail("resume", "检查点不是最新完整状态，或存在未发布状态。")
        if (self.root / "manifest.json").exists() or (self.root / "training.json").exists():
            fail("resume", "训练已进入最终产物保存阶段。不能自动覆盖或恢复。")
        return path.parent / "state.pt", step
