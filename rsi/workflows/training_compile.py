"""独立标准来源编译与显式评分，保留失败调用和成本。"""

from pathlib import Path
from typing import Any

from rsi.storage.artifacts import file_sha256, write_json
from rsi.configuration.contracts import ExperimentContract
from rsi.datasets.data import validate_dataset
from rsi.storage.execution_artifacts import complete_execution, implementation_snapshot
from rsi.models.hf_backend import HfRuntime, load_tokenizer
from rsi.datasets.labels import SkillOntology, parse_labels
from rsi.storage.leases import run_lease
from rsi.models.model_resolution import ModelLocation, check_device
from rsi.curation.policy import parse_policy
from rsi.models.scoring import TokenMetrics, attach_scores
from rsi.curation.selection_demo import write_jsonl
from rsi.curation.selection_io import load_inputs
from rsi.storage.state import StateStore
from rsi.datasets.training_inputs import bind_message_evidence, compile_training_sources, rebind_label_snapshots
from rsi.configuration.validation import choice, digest, fail, read_json, read_jsonl, to_json_value


class MockChatTokenizer:
    """明确的字符测试分词器，仅供 mock 编排，不可作为真实模型 tokenizer。"""
    is_fast = True
    chat_template = "mock-role-lines-v1"

    def apply_chat_template(self, messages: list[dict[str, str]], *, tokenize: bool = False, **kwargs: Any) -> str:
        return "".join("<" + m["role"] + ">" + m["content"] + "\n" for m in messages)

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def decode(self, ids: list[int], **kwargs: Any) -> str:
        return "".join(chr(i) for i in ids)


def compile_training_data(sources_path: str | Path, config_path: str | Path, metadata_path: str | Path,
                           ontology_path: str | Path, policy_path: str | Path, location_path: str | Path,
                           output: str | Path, *, mode: str, device: str, max_length: int, data_kind: str,
                           message_evidence_path: str | Path | None = None, expected_gpu_name: str | None = None) -> dict[str, Any]:
    """编译并普通前向评分，不执行训练，不自动改变外部预算。"""
    choice(mode, "mode", {"mock", "hf_local"})
    choice(data_kind, "data_kind", {"synthetic", "real_public", "private"})
    root = Path(output).resolve()
    if root.exists():
        fail("output", "编译输出已存在，拒绝覆盖或重复评分。")
    location_file = Path(location_path).resolve()
    location = ModelLocation.parse(read_json(location_file), location_file.parent)
    config = read_json(config_path)
    config["model"] = to_json_value(location.model_ref)
    contract = ExperimentContract.parse(config)
    if mode == "mock" and contract.purpose != "smoke":
        fail("mode", "mock 分词和评分只能使用 smoke。")
    ontology = SkillOntology.parse(read_json(ontology_path))
    policy = parse_policy(read_json(policy_path), ontology)
    metadata = tuple(parse_labels(row, ontology) for row in read_jsonl(metadata_path))
    sources = read_jsonl(sources_path)
    if mode == "hf_local":
        hardware = check_device(device, expected_gpu_name)
        tokenizer = load_tokenizer(location.base_model_path)
        location.verify(tokenizer)
    else:
        if device != "cpu":
            fail("mode", "mock 编排仅允许 CPU。")
        tokenizer, hardware = MockChatTokenizer(), {"device": "cpu", "mode": "mock"}
    bundle = compile_training_sources(sources, tokenizer, contract, max_length)
    if mode == "hf_local" and config["training"]["precision"] != location.load.precision:
        fail("precision", "模型加载与合同精度不一致。")
    root.mkdir(parents=True)
    with run_lease(root):
        write_json(root / "request.json", {"sources_hash": digest(sources), "model_identity": location.identity,
                    "mode": mode, "source_config_sha256": file_sha256(config_path)})
        write_json(root / "environment.json", {"hardware": hardware, "mode": mode, "software": implementation_snapshot(),
                   "model_config": read_json(Path(location.base_model_path) / "config.json") if mode == "hf_local" else None})
        write_jsonl(root / "sources.jsonl", sources)
        write_json(root / "message_mappings.json", to_json_value(bundle.mappings))
        raw_evidence = read_json(message_evidence_path) if message_evidence_path else {
                        "schema_version": "message-evidence-v1", "index_rule": "source_message_index_zero_v1", "items": []}
        evidence = bind_message_evidence(raw_evidence, bundle.mappings)
        lookup = {r.data_id: r for r in bundle.records}
        if any(item.skill_id not in ontology.ids or item.skill_id not in lookup[item.data_id].skills for item in evidence):
            fail("message_evidence.skill_id", "证据技能必须属于该轨迹的静态技能和 ontology。")
        write_json(root / "message_evidence.json", raw_evidence)
        write_json(root / "skill_evidence.json", to_json_value(evidence))
        metrics: dict[str, TokenMetrics] = {}
        with StateStore(root / "state.sqlite3") as state:
            state.initialize("compile", "controlled", config, [r.raw for r in bundle.records])
            selected = tuple(r for r in bundle.records if r.role in {"candidate", "reference", "score_probe"})
            state.reserve("compile", "score", "scoring_tokens", sum(r.token_count for r in selected))

            def score_all(scorer: Any) -> None:
                for index, record in enumerate(selected):
                    write_json(root / "score_calls" / f"{index:06d}.intent.json", {"data_id": record.data_id, "model": to_json_value(location.model_ref)})
                    value = scorer.score(record) if mode == "hf_local" else None
                    if value is None:
                        tokens = scorer.score_tokens(record)
                        from rsi.evaluation.fit_evaluation import aggregate
                        stats = aggregate(tokens)
                        value = TokenMetrics(record.data_id, stats.token_count, record.token_count, stats.nll, stats.entropy, (), stats.ce_sum, stats.entropy_sum)
                    metrics[record.data_id] = value
                    state.consume("compile", "score:" + record.data_id, "score", record.token_count)
                    write_json(root / "score_calls" / f"{index:06d}.result.json", to_json_value(value))

            try:
                if mode == "hf_local":
                    with HfRuntime(location.base_model_path, device, location.load.precision, location.adapter_path,
                                   attention_implementation=location.load.attention_implementation) as scorer:
                        score_all(scorer)
                else:
                    from rsi.models.training_mock import MockTokenScorer
                    score_all(MockTokenScorer())
            except BaseException as exc:
                state.record_error("compile", "scoring_failure", str(exc) or type(exc).__name__)
                write_json(root / "failure.json", {"error": str(exc), "unknown_consumption_possible": True})
                raise
            state.release("compile", "score")
            write_json(root / "costs.json", state.snapshot("compile"))
        scored = attach_scores(bundle.records, metrics, contract)
        records = validate_dataset(scored, contract)
        labels = rebind_label_snapshots(records, metadata, ontology)
        write_json(root / "config.json", config)
        write_json(root / "ontology.json", ontology.to_dict())
        write_json(root / "policy.json", policy.to_dict())
        write_json(root / "batch_history.json", {"schema_version": "batch-history-v1", "batches": []})
        write_json(root / "model_location.json", location.to_dict())
        write_jsonl(root / "trajectories.jsonl", scored)
        write_jsonl(root / "labels.jsonl", [label.to_dict() for label in labels])
        provenance = {"schema_version": "training-provenance-v1", "data_kind": data_kind, "execution_mode": mode,
                      "source": "measured" if mode == "hf_local" else "synthetic_fixture", "sources_hash": bundle.sources_hash,
                      "static_labels_hash": digest(metadata), "tokenizer_execution_mode": mode}
        write_json(root / "provenance.json", provenance)
        artifact = complete_execution(root, "COMPILED_AND_SCORED")
    return {"status": "COMPILED_AND_SCORED", "artifact": to_json_value(artifact), "content_id": read_json(artifact.path)["content_id"],
            "training_status": "NOT_RUN", "provenance": provenance}
