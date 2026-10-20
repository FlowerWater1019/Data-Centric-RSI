"""十条候选的离线演示。所有标签和反馈都是合成值。"""

from dataclasses import replace
from pathlib import Path
import random
from typing import Any

from rsi._paths import PROJECT_ROOT
from rsi.storage.artifacts import write_json
from rsi.configuration.contracts import ExperimentContract
from rsi.datasets.data import validate_dataset
from rsi.curation.dispatch import apply_dispatch, parse_dispatch
from rsi.curation.feedback import parse_feedback
from rsi.curation.history import BatchHistory, HistoricalBatch
from rsi.datasets.labels import FLAGS, SkillOntology, parse_labels, scores_snapshot
from rsi.curation.policy import parse_policy
from rsi.curation.policy_selection import SelectionResult, select_with_policy, validate_selection
from rsi.models.scoring import TokenMetrics, attach_scores
from rsi.curation.selection_io import (SelectionInputs, finish_manifest, load_fixture, publish_directory,
                           verify_manifest)
from rsi.configuration.validation import canonical, digest, fail, integer, read_json, string, strings, text_digest


NOTICE = "synthetic labels / synthetic feedback / no training / no task execution"


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for value in values:
            handle.write(canonical(value) + "\n")


def fixture_values(seed: int = 42, skills: tuple[str, ...] = ("code_generation", "error_analysis", "regression_check"),
                   labels_per_sample: int = 1) -> dict[str, Any]:
    """生成新合成轨迹。局部随机数只分配技能，不修改现有数据。"""
    integer(seed, "fixture.seed")
    skills = strings(list(skills), "fixture.skills", True)
    integer(labels_per_sample, "fixture.labels_per_sample", 1)
    if labels_per_sample > len(skills) or any("." in s for s in skills):
        fail("fixture.skills", "标签数不能超过技能数，技能 ID 不能包含点。")
    config = read_json(PROJECT_ROOT / "configs/smoke.json")
    config["schema_version"] = "1.1"
    config["contract_id"] = "selection_fixture_v1"
    config["seed"] = seed
    config["skills"] = list(skills)
    config["selection"] = {"supervised_token_budget": 16, "allowed_shortfall": 0,
                           "min_skill_shares": {}, "packing_rule": "exact_ranked_v1"}
    config["fixed_strategy"]["skill_preferences"] = {k: 0 for k in skills}
    contract = ExperimentContract.parse(config)
    ontology_version = "demo-coding-skills-v1" if skills == ("code_generation", "error_analysis", "regression_check") else "demo-coding-skills-" + digest(skills)
    ontology = {"schema_version": "skill-ontology-v1", "ontology_version": ontology_version, "domain": "coding",
                "skills": [{"id": k, "name": k, "description": "合成演示技能，不表示真实能力。"} for k in skills]}
    rng = random.Random(seed)
    data = []
    for i in range(12):
        data_id = f"sample_{i + 1:03d}" if i < 10 else f"reference_{i - 9:03d}"
        text = f"task-{data_id}\nfix\n"
        count = len(text)
        selected_skills = rng.sample(list(skills), labels_per_sample) if i < 10 else []
        data.append({"schema_version": "1.0", "data_id": data_id, "version": text_digest(text),
            "role": "candidate" if i < 10 else "reference", "split_group": "split_" + data_id,
            "duplicate_group": "duplicate_" + data_id, "text": text, "text_sha256": text_digest(text),
            "skills": selected_skills,
            "quality": {"verification": "pass", "evidence_id": "synthetic_" + data_id, "integrity_valid": True, "leakage": "clear"},
            "tokenization": {"status": "demo", "tokenizer_revision": contract.model.tokenizer_revision,
                "template_revision": contract.model.template_revision, "token_ids": [ord(c) for c in text],
                "offsets": [[j, j + 1] for j in range(count)],
                "roles": ["user"] * (count - 4) + ["assistant"] * 4,
                "correctness": ["context"] * (count - 4) + ["valid"] * 4},
            "supervision": {"mask_version": "mask_fixture_v1", "loss_mask": [False] * (count - 4) + [True] * 4},
            "diagnostic_spans": [], "usage": {"ever_used_in_training_attempt": False, "in_accepted_lineage": False,
                "training_attempt_ids": [], "cumulative_supervised_tokens": 0}, "scores": None})
    records = validate_dataset(data, contract)
    metrics = {r.data_id: TokenMetrics(r.data_id, r.supervised_tokens, r.token_count,
               1.5 if r.role == "candidate" else float(i - 9),
               2.5 if r.role == "candidate" else float(i - 8), ()) for i, r in enumerate(records)}
    data = attach_scores(records, metrics, contract)
    records = validate_dataset(data, contract)
    labels = []
    for record in records[:10]:
        labels.append({"schema_version": "data-labels-v1", "data_id": record.data_id,
            "trajectory_version": record.version, "label_version": "labels-v1", "domain": "coding",
            "provenance": {"dataset_id": "demo_coding", "dataset_revision": "demo-v1", "source_record_id": record.data_id,
                "label_mode": "synthetic_fixture", "annotation_seed": seed},
            "partition": {"role": record.role, "split_group": record.split_group, "duplicate_group": record.duplicate_group},
            "task": {"task_type": "code_generation", "trajectory_type": "single_turn", "harness_protocol": "python_function_v1", "language": "python"},
            "task_ref": {"task_id": record.data_id, "repository": None, "base_commit": None, "environment_id": None, "evaluation_spec_id": None},
            "protocol_ref": {"harness_id": "fixture_python", "harness_version": "fixture-v1", "tool_schema_version": None},
            "quality_flags": {k: True for k in sorted(FLAGS)},
            "quality_annotation": {"source": "synthetic_fixture", "rule_version": "demo-quality-v1", "evidence_refs": []},
            "skills": {"ontology_version": ontology["ontology_version"], "items": [{"skill_id": k, "source": "seeded_random", "evidence_refs": []} for k in record.skills]},
            "cost": {"tokenization_version": contract.model.tokenizer_revision, "context_tokens": record.token_count,
                "supervised_tokens": record.supervised_tokens, "length_bucket": "short"},
            "model_scores": scores_snapshot(record), "usage": record.raw["usage"]})
    # 配额与预算来自已生成的监督量，不把文档占位值当作分词结果。
    budget = sum(r.supervised_tokens for r in records[:4])
    config["selection"]["supervised_token_budget"] = budget
    preferences = {k: max(0.1, 0.7 - i * 0.3) for i, k in enumerate(skills)}
    policy = {"schema_version": "selection-policy-v1", "policy_id": "policy_001", "parent_policy_id": None,
        "domain": "coding", "skill_ontology_version": ontology["ontology_version"],
        "created_by": {"type": "human", "controller_checkpoint_id": None, "dispatch_id": None},
        "eligibility": {"allowed_label_modes": ["synthetic_fixture"], "required_true_flags": sorted(FLAGS),
            "unknown_flag_action": "exclude", "allowed_task_types": ["code_generation"], "allowed_trajectory_types": ["single_turn"],
            "allowed_harness_protocols": ["python_function_v1"], "allowed_languages": ["python"], "excluded_data_ids": [], "deferred_batch_ids": []},
        "scoring": {"formula_version": "weighted_utility_v1", "score_requirement": "current_checkpoint",
            "reference_version": contract.scoring.reference_version, "cdf_version": contract.scoring.cdf_version,
            "nll_preference": {"support": [0.2, 0.95], "preferred": [0.45, 0.75]},
            "entropy_preference": {"support": [0.1, 0.9], "preferred": [0.3, 0.65]},
            "skill_preferences": preferences, "skill_aggregation": "max", "weights": {"nll": 0.3, "entropy": 0.15, "skill": 0.55}},
        "batch_constraints": {"supervised_token_budget": budget, "allowed_shortfall": 0, "max_context_tokens_per_sample": 4096,
            "min_skill_token_shares": {}, "min_task_type_token_shares": {}, "max_task_type_token_shares": {},
            "duplicate_rule": "one_per_duplicate_group", "allow_within_batch_repetition": False},
        "execution": {"packing_rule": "exact_ranked_v1", "quota_handling": "postcheck_fail", "tie_breaker": "data_id_ascending", "seed": seed},
        "description": "合成演示策略，不表达真实技能结论。"}
    target = "regression_check" if "regression_check" in skills else skills[-1]
    target_present = any(target in record.skills for record in records if record.role == "candidate")
    feedback = {"schema_version": "feedback-fixture-v1", "feedback_id": "feedback_001", "round_id": "round_001",
        "batch_id": "unbound_fixture_batch", "parent_checkpoint_id": contract.model.checkpoint_id,
        "candidate_checkpoint_id": "candidate_fixture_001", "source": "synthetic_fixture", "data_role": "feedback",
        "fit": {"scope": "selected_training_batch", "supervised_tokens": budget,
            "before": {"nll": 1.5, "entropy": 2.5}, "after": {"nll": 1.2, "entropy": 2.2}},
        "execution": {"target": {"task_count": 4, "before_passed": 2, "after_passed": 2},
            "retention": {"task_count": 4, "before_passed": 3, "after_passed": 2},
            "by_skill": {target: {"task_count": 4, "before_passed": 3, "after_passed": 2}}},
        "acceptance": {"status": "rejected", "source": "fixture", "rule_version": "fixture-v1", "reason": "预设拒绝值，未运行验收。"},
        "evidence": {"fit_nll": {"kind": "fit_nll", "skill_id": None},
            "regression_execution": {"kind": "skill_execution", "skill_id": target},
            "acceptance_result": {"kind": "acceptance_result", "skill_id": None}}}
    dispatch = {"schema_version": "dispatch-decision-v1", "dispatch_id": "dispatch_001", "round_id": "round_001",
        "controller": {"type": "fixture", "checkpoint_id": contract.model.checkpoint_id, "prompt_version": "dispatch-prompt-v1"},
        "inputs": {"parent_checkpoint_id": contract.model.checkpoint_id, "candidate_checkpoint_id": feedback["candidate_checkpoint_id"],
            "current_policy_id": policy["policy_id"], "batch_id": feedback["batch_id"], "feedback_id": feedback["feedback_id"], "history_dispatch_ids": []},
        "diagnosis": {"summary": "模拟反馈只检查调度链路。",
            "observations": [{"evidence_ref": "fit_nll", "statement": "预设拟合 NLL 下降。"},
                {"evidence_ref": "regression_execution", "statement": "预设技能组通过数下降。"}],
            "hypotheses": [{"statement": "提高偏好可能改变批次组成。", "evidence_refs": ["regression_execution"]}],
            "limitations": [NOTICE]},
        "model_action": {"recommendation": "reject", "reason": "模拟拒绝，本阶段只记录建议。", "acceptance_evidence_ref": "acceptance_result"},
        "natural_language_dispatch": "提高指定技能的偏好和最低覆盖。保留父模型。",
        "policy_update": {"base_policy_id": policy["policy_id"], "proposed_policy_id": "policy_002",
            "set": {"scoring.skill_preferences." + target: 1.0,
                    "batch_constraints.min_skill_token_shares." + target: 0.25 if target_present else 0.0}, "add": {}, "remove": {}},
        "next_action": "select_next_batch"}
    if not target_present:
        dispatch["natural_language_dispatch"] = "当前合成候选不含目标技能。仅提高该技能偏好，最低覆盖设为 0。"
    return {"config": config, "data": data, "labels": labels, "ontology": ontology,
            "policy": policy, "feedback": feedback, "dispatch": dispatch,
            "score_fixture": {"source": "synthetic_fixture", "metrics": {k: {"nll": v.nll, "entropy": v.entropy} for k, v in metrics.items()}}}


def make_fixture(output: str | Path, seed: int = 42, skills: tuple[str, ...] = ("code_generation", "error_analysis", "regression_check"),
                 labels_per_sample: int = 1) -> dict[str, Any]:
    values = fixture_values(seed, skills, labels_per_sample)
    with publish_directory(output) as root:
        for key, name in (("config", "config.json"), ("ontology", "ontology.json"), ("policy", "policy.json"),
                          ("feedback", "feedback.fixture.json"), ("dispatch", "dispatch.fixture.json"), ("score_fixture", "scores.fixture.json")):
            write_json(root / name, values[key])
        write_json(root / "batch_history.json", BatchHistory().to_dict())
        write_jsonl(root / "trajectories.jsonl", values["data"])
        write_jsonl(root / "labels.jsonl", values["labels"])
        inconclusive = {**values["feedback"], "feedback_id": "feedback_inconclusive",
                        "acceptance": {**values["feedback"]["acceptance"], "status": "inconclusive", "reason": "预设证据不足。"}}
        write_json(root / "feedback.inconclusive.fixture.json", inconclusive)
        # 发布前通过实际入口校验生成结果。
        ontology = SkillOntology.parse(values["ontology"])
        contract = ExperimentContract.parse(values["config"])
        validate_selection(contract, validate_dataset(values["data"], contract),
                           tuple(parse_labels(x, ontology) for x in values["labels"]), ontology,
                           parse_policy(values["policy"], ontology), allow_synthetic_labels=True)
        parse_feedback(values["feedback"], ontology)
        parse_feedback(inconclusive, ontology)
        parse_dispatch(values["dispatch"])
        manifest = finish_manifest(root, "fixture")
    return {"status": "CREATED", "content_id": manifest["content_id"], "candidate_count": 10,
            "record_count": len(values["data"]), "training_status": "NOT_RUN", "task_execution_status": "NOT_RUN"}


def select_inputs(inputs: SelectionInputs, *, allow_synthetic_labels: bool = False) -> SelectionResult:
    return select_with_policy(inputs.contract, inputs.records, inputs.labels, inputs.ontology, inputs.policy,
                              inputs.history, allow_synthetic_labels=allow_synthetic_labels)


def save_selection(root: Path, result: SelectionResult) -> None:
    write_json(root / "selection_report.json", result.to_dict())
    if result.batch is None:
        fail("selection." + result.to_dict()["status"], result.to_dict()["failure_reason"])
    write_json(root / "batch.json", result.batch.to_dict())


def ranking_ids(result: SelectionResult) -> list[str]:
    """只返回实际参与排名的候选。被排除候选保留在完整报告中。"""
    candidates = (row for row in result.to_dict()["candidates"] if row["ranking_position"] is not None)
    return [row["data_id"] for row in sorted(candidates, key=lambda row: row["ranking_position"])]


def run_demo(fixture: str | Path, output: str | Path) -> dict[str, Any]:
    """绑定真实批次身份并落盘，再执行纯调度和第二次选择。"""
    if Path(output).resolve().is_relative_to(Path(fixture).resolve()):
        fail("output", "演示输出不能位于 fixture 内部。")
    fixture_manifest = verify_manifest(Path(fixture))
    inputs = load_fixture(fixture)
    before = select_inputs(inputs, allow_synthetic_labels=True)
    if before.batch is None:
        fail("demo.before", before.to_dict()["failure_reason"])
    ids = before.to_dict()["identities"]
    batch_id = "batch_" + digest({"context": before.batch.context_id, "fingerprint": before.batch.fingerprint})
    history = BatchHistory(batches=inputs.history.batches + (HistoricalBatch(batch_id, ids["policy_hash"], ids["labels_hash"],
               ids["dataset_hash"], tuple(x.data_id for x in before.batch.items)),))
    feedback_raw = read_json(Path(fixture) / "feedback.fixture.json")
    dispatch_raw = read_json(Path(fixture) / "dispatch.fixture.json")
    feedback_raw["batch_id"] = batch_id
    dispatch_raw["inputs"]["batch_id"] = batch_id
    feedback = parse_feedback(feedback_raw, inputs.ontology)
    if feedback.parent_checkpoint_id != inputs.contract.model.checkpoint_id:
        fail("feedback.parent_checkpoint_id", "与本次筛选的实际父模型不一致。")
    if feedback.fit.supervised_tokens != before.batch.supervised_tokens:
        fail("feedback.fit.supervised_tokens", "与实际批次不一致。")
    dispatch = parse_dispatch(dispatch_raw)
    if dispatch.next_action != "select_next_batch":
        fail("dispatch.next_action", "两次筛选演示要求 select_next_batch；stop 请使用独立 apply-dispatch 保存。")
    applied = apply_dispatch(inputs.policy, dispatch, feedback, inputs.ontology, history)
    # 历史随轮次增加。数据、标签、分数和 seed 保持相同。
    after = select_inputs(replace(inputs, policy=applied.policy, history=history), allow_synthetic_labels=True)
    if after.batch is None:
        fail("demo.after", after.to_dict()["failure_reason"])
    before_ids, after_ids = [x.data_id for x in before.batch.items], [x.data_id for x in after.batch.items]
    comparison = {"notice": NOTICE, "before_ids": before_ids, "after_ids": after_ids,
        "membership_changed": set(before_ids) != set(after_ids), "policy_diff": applied.to_dict()["diff"],
        "policy_metadata_diff": applied.to_dict()["metadata_diff"],
        "before_ranking": ranking_ids(before),
        "after_ranking": ranking_ids(after),
        "same_inputs": {k: before.to_dict()["identities"][k] == after.to_dict()["identities"][k]
                        for k in ("dataset_hash", "labels_hash", "scoring_hash", "contract_hash", "ontology_hash")},
        "same_seed": before.batch.seed == after.batch.seed,
        "training_status": "NOT_RUN", "task_execution_status": "NOT_RUN"}
    with publish_directory(output) as root:
        source = Path(fixture)
        # 复制已核验的全部输入。绑定后的反馈和调度另存于根目录。
        import shutil
        shutil.copytree(source, root / "inputs")
        if verify_manifest(root / "inputs") != fixture_manifest:
            fail("fixture", "输入快照在执行期间发生变化，拒绝发布。")
        for name, result in (("before", before), ("after", after)):
            (root / name).mkdir()
            save_selection(root / name, result)
        write_json(root / "batch_history.json", history.to_dict())
        write_json(root / "feedback.fixture.json", feedback_raw)
        write_json(root / "dispatch.fixture.json", dispatch_raw)
        write_json(root / "policy.after.json", applied.policy.to_dict())
        write_json(root / "dispatch_application.json", applied.to_dict())
        write_json(root / "comparison.json", comparison)
        rates = {k: value.rates() for k, value in feedback.execution.by_skill.items()}
        write_json(root / "feedback_rates.json", rates)
        (root / "report.md").write_text("# 筛选链路演示\n\n" + NOTICE + "\n\n"
            + "训练状态：NOT_RUN。任务执行状态：NOT_RUN。模型操作：recommendation_only。\n\n"
            + "更新前：" + ", ".join(before_ids) + "\n\n更新后：" + ", ".join(after_ids)
            + "\n\n成员变化：" + str(comparison["membership_changed"]) + "\n\n实际策略差异：\n\n```json\n"
            + canonical(comparison["policy_diff"]) + "\n```\n", encoding="utf-8")
        manifest = finish_manifest(root, "demo")
    return {**comparison, "content_id": manifest["content_id"]}
