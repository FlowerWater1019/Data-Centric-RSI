"""单卡设备绑定校验。数字编号不能作为物理设备身份。"""

import re
from typing import Any

from .validation import fail


def require_uuid_binding(device: str, visible_devices: str | None) -> str:
    """要求进程只暴露一个完整 GPU UUID，并使用逻辑 cuda:0。"""
    if device != "cuda:0" or visible_devices is None or re.fullmatch(
        r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", visible_devices
    ) is None:
        fail("device", "pilot 要求 CUDA_VISIBLE_DEVICES 为单个完整 GPU UUID，并使用 cuda:0。禁止数字编号。")
    return visible_devices


def verify_device(properties: Any, expected_name: str, expected_uuid: str) -> dict[str, Any]:
    """模型加载前核对 CUDA 返回的真实型号与 UUID。"""
    observed_uuid = str(getattr(properties, "uuid", ""))
    if not observed_uuid.startswith("GPU-"):
        observed_uuid = "GPU-" + observed_uuid
    if properties.name != expected_name or observed_uuid.lower() != expected_uuid.lower():
        fail("device", f"实际设备与声明不符：{properties.name}, {observed_uuid}。未加载模型。")
    return {"device": "cuda:0", "cuda_visible_devices": expected_uuid, "uuid": observed_uuid,
            "name": properties.name, "memory_bytes": properties.total_memory}
