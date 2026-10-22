"""将已验证数据包编译为现有轨迹接口。不生成模型评分或演示分数。"""

from pathlib import Path
from typing import Any

from rsi.storage.artifacts import file_sha256, model_manifest, verify_training_artifact, write_json
from rsi.models.backends import ArtifactRef
from rsi.configuration.contracts import ExperimentContract
from rsi.datasets.data import validate_dataset
from rsi.datasets.dataset_import import write_jsonl
from rsi.models.hf_backend import load_tokenizer
from rsi.models.model_inputs import bound_model
from rsi.datasets.python_diagnostics import SKILLS, annotate_keywords
from rsi.datasets.tokenization import compile_trajectory
from rsi.configuration.validation import fail, integer, read_json, read_jsonl, to_json_value


def verified_bundle(path: str | Path, expected_sha256: str) -> tuple[Path, dict[str, Any]]:
    root = Path(path).resolve()
    artifact = ArtifactRef(str(root / "manifest.json"), expected_sha256)
    verify_training_artifact(artifact)
    manifest = read_json(artifact.path)
    if manifest.get("schema_version") != "import-bundle-1" or manifest.get("data_kind") != "real_public":
        fail("bundle", "需要已完成的真实数据导入包。")
    names = {entry["name"] for entry in manifest["files"]}
    if not {"sources.jsonl", "feedback_tasks.json", "origin.json"} <= names:
        fail("bundle", "数据包缺少必要产物。")
    return root, manifest


def compile_bundle(bundle_path: str | Path, bundle_sha256: str, config_path: str | Path,
                   model_path: str | Path, output_path: str | Path, max_length: int,
                   diagnostics: str = "none") -> dict[str, Any]:
    """仅加载本地 tokenizer。超长或无效轨迹导致明确失败，不静默截断。"""
    integer(max_length, "max_length", 2)
    root, manifest = verified_bundle(bundle_path, bundle_sha256)
    output = Path(output_path).resolve()
    if output.exists():
        fail("output", "编译目录必须尚不存在。")
    identity = model_manifest(model_path)
    tokenizer = load_tokenizer(model_path)
    config = read_json(config_path)
    config["model"] = bound_model(identity, tokenizer)
    if diagnostics not in {"none", "python_keywords_v1"}:
        fail("diagnostics", "只支持 none 或 python_keywords_v1。")
    if diagnostics != "none":
        config["skills"] = list(dict.fromkeys(config["skills"] + list(SKILLS)))
    if config["purpose"] != "draft":
        fail("config.purpose", "真实数据编译要求显式 draft 配置。")
    contract = ExperimentContract.parse(config)
    sources = read_jsonl(root / "sources.jsonl")
    if any(row["role"] not in {"candidate", "reference", "score_probe", "feedback_diagnostic"} for row in sources):
        fail("sources", "训练来源不能包含反馈执行或最终测试任务。")
    if diagnostics != "none":
        sources = [annotate_keywords(source) for source in sources]
    compiled = [compile_trajectory(source, tokenizer, contract, max_length) for source in sources]
    records = validate_dataset(compiled, contract)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "contract.json", config)
    write_jsonl(output / "trajectories.jsonl", compiled)
    write_json(output / "model_identity.json", identity)
    write_jsonl(output / "compiled_sources.jsonl", sources)
    report = {"schema_version": "compiled-data-1", "status": "COMPILED", "purpose": "engineering_validation",
              "data_kind": manifest["data_kind"], "bundle": str(root), "bundle_sha256": bundle_sha256,
              "source_config_sha256": file_sha256(config_path), "model": to_json_value(contract.model),
              "records": len(records), "candidate_records": sum(record.is_eligible for record in records),
              "supervised_tokens": sum(record.supervised_tokens for record in records),
              "max_context_tokens": max(record.token_count for record in records),
              "max_length": max_length, "scores": "NOT_COMPUTED", "training": "NOT_RUN",
              "diagnostic_annotation": diagnostics,
              "files": [{"name": p.name, "sha256": file_sha256(p)} for p in sorted(output.iterdir())]}
    write_json(output / "report.json", report)
    return report
