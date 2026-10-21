"""本地模型内容、加载配置和 LoRA 配置。路径不替代内容身份。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os

from .artifacts import model_manifest, verify_training_artifact
from .backends import ArtifactRef
from .contracts import ModelRef
from .devices import require_uuid_binding, verify_device
from .labels import fields
from .model_inputs import bound_model
from .validation import boolean, choice, digest, fail, number, read_json, string, strings, to_json_value


@dataclass(frozen=True)
class LoadConfig:
    precision: str = "fp32"
    attention_implementation: str = "eager"
    local_files_only: bool = True
    trust_remote_code: bool = False

    @classmethod
    def parse(cls, raw: Any) -> "LoadConfig":
        v = fields(raw, "load", "precision attention_implementation local_files_only trust_remote_code")
        choice(v["precision"], "load.precision", {"fp32", "bf16"})
        choice(v["attention_implementation"], "load.attention_implementation", {"eager", "sdpa"})
        if not boolean(v["local_files_only"], "load.local_files_only") or boolean(v["trust_remote_code"], "load.trust_remote_code"):
            fail("load", "必须仅加载本地文件，且禁用 remote code。")
        return cls(**v)


@dataclass(frozen=True)
class LoraConfig:
    targets: tuple[str, ...] = ("q_proj", "v_proj")
    alpha: float = 4.0
    dropout: float = 0.0

    @classmethod
    def parse(cls, raw: Any) -> "LoraConfig":
        v = fields(raw, "lora", "targets alpha dropout")
        alpha = number(v["alpha"], "lora.alpha")
        if alpha <= 0:
            fail("lora.alpha", "必须大于 0。")
        dropout = number(v["dropout"], "lora.dropout", 0, 1)
        if dropout == 1:
            fail("lora.dropout", "必须小于 1。")
        return cls(strings(v["targets"], "lora.targets", True), alpha, dropout)


@dataclass(frozen=True)
class ModelLocation:
    schema_version: str
    model_ref: ModelRef
    base_model_path: str
    adapter_manifest: ArtifactRef | None
    load: LoadConfig

    @classmethod
    def parse(cls, raw: Any, relative_to: Path = Path(".")) -> "ModelLocation":
        v = fields(raw, "model_location", "schema_version model_ref base_model_path adapter_manifest load")
        choice(v["schema_version"], "model_location.schema_version", {"model-location-v1"})
        base = Path(string(v["base_model_path"], "model_location.base_model_path"))
        base = (relative_to / base).resolve() if not base.is_absolute() else base.resolve()
        adapter = None
        if v["adapter_manifest"] is not None:
            a = fields(v["adapter_manifest"], "adapter_manifest", "path sha256")
            path = Path(string(a["path"], "adapter_manifest.path"))
            path = (relative_to / path).resolve() if not path.is_absolute() else path.resolve()
            adapter = ArtifactRef(str(path), string(a["sha256"], "adapter_manifest.sha256"))
        return cls(v["schema_version"], ModelRef.parse(v["model_ref"]), str(base), adapter, LoadConfig.parse(v["load"]))

    @property
    def identity(self) -> str:
        return digest({"model_ref": self.model_ref, "load": self.load,
                       "adapter_hash": self.adapter_manifest.sha256 if self.adapter_manifest else None})

    @property
    def adapter_path(self) -> Path | None:
        return Path(self.adapter_manifest.path).parent / "adapter" if self.adapter_manifest else None

    def verify(self, tokenizer: Any | None = None) -> None:
        identity = model_manifest(self.base_model_path)
        if identity["content_id"] != self.model_ref.base_revision:
            fail("model_location.model_ref.base_revision", "基座内容变化或引用不匹配。")
        if tokenizer is not None:
            expected = bound_model(identity, tokenizer)
            for key in ("tokenizer_revision", "template_revision"):
                if getattr(self.model_ref, key) != expected[key]:
                    fail("model_location.model_ref." + key, "分词器或模板内容不匹配。")
        if self.adapter_manifest:
            verify_training_artifact(self.adapter_manifest)
            if self.adapter_manifest.sha256 != self.model_ref.adapter_revision or not self.adapter_path.is_dir():
                fail("adapter_manifest", "父 adapter 内容与模型引用不一致。")
        elif self.model_ref.adapter_revision != "none":
            fail("adapter_manifest", "已有 adapter 的模型必须提供完整产物引用。")

    def to_dict(self) -> dict[str, Any]:
        return to_json_value(self)


def check_device(device: str, expected_name: str | None = None) -> dict[str, Any]:
    """真实模型加载前核验物理设备。CPU 不要求 GPU 包。"""
    if device == "cpu":
        return {"device": "cpu", "uuid": None}
    from .hf_backend import load_dependencies
    binding = require_uuid_binding(device, os.environ.get("CUDA_VISIBLE_DEVICES"))
    if expected_name is None:
        fail("device.expected_name", "GPU 执行必须声明型号。")
    torch, _, _ = load_dependencies()
    if torch.cuda.device_count() != 1:
        fail("device", "必须只暴露一张 GPU。")
    return verify_device(torch.cuda.get_device_properties(device), expected_name, binding)
