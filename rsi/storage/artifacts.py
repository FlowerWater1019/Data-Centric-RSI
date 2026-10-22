"""计算本地产物的内容标识，并写入不可覆盖的清单。"""

import hashlib
import json
from pathlib import Path
from typing import Any

from rsi.models.backends import ArtifactRef
from rsi.configuration.validation import digest, fail, read_json


def file_sha256(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: str | Path, value: Any) -> ArtifactRef:
    """创建新文件。目标已存在时拒绝覆盖。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return ArtifactRef(str(target.resolve()), file_sha256(target))


def model_manifest(path: str | Path) -> dict[str, Any]:
    """核验本地模型文件。标识只代表本地内容，不声称对应官方提交。"""
    root = Path(path).resolve()
    if not root.is_dir():
        fail("model_path", "需要已有的本地模型目录。程序不下载模型。")
    files = sorted(p for p in root.iterdir() if p.is_file() and (
        p.suffix in {".safetensors", ".json", ".jinja", ".model", ".txt"}
    ))
    if not (root / "config.json").is_file() or not any(p.suffix == ".safetensors" for p in files):
        fail("model_path", "需要 config.json 和 safetensors 权重。")
    entries = [{"name": p.name, "bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in files]
    return {"path": str(root), "files": entries, "content_id": digest(entries),
            "identity_evidence": "local_file_hashes", "official_revision": None}


def verify_training_artifact(artifact: ArtifactRef) -> None:
    """重新核验训练清单与所有文件。拒绝清单中的越界路径。"""
    path = Path(artifact.path).resolve()
    if file_sha256(path) != artifact.sha256:
        fail("artifact", "训练产物清单校验值不匹配。")
    manifest = read_json(path)
    from rsi.configuration.validation import array, obj, string
    if not isinstance(manifest, dict) or "files" not in manifest:
        fail("artifact.files", "产物清单必须提供文件列表。")
    entries = array(manifest["files"], "artifact.files", True)
    seen: set[Path] = set()
    for entry in entries:
        entry = obj(entry, "artifact.file", {"name", "sha256"})
        string(entry["name"], "artifact.file.name")
        string(entry["sha256"], "artifact.file.sha256")
        target = (path.parent / entry["name"]).resolve()
        if target in seen:
            fail("artifact.files", "多个条目指向相同文件。")
        seen.add(target)
        if not target.is_relative_to(path.parent) or file_sha256(target) != entry["sha256"]:
            fail("artifact", "训练产物文件越界或校验值不匹配。")
