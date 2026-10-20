"""标签严格解析、快照关联及隔离。"""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import random

from rsi.labels import FLAGS, SkillOntology, parse_labels
from rsi.selection_demo import fixture_values
from rsi.validation import ValidationError, canonical, loads
from tests.selection_helpers import SelectionFixture


class LabelsTests(SelectionFixture):
    def test_strict_fields_versions_ids_and_numeric_types(self) -> None:
        raw = self.values["labels"][0]
        variants = [dict(raw, extra=True), {k: v for k, v in raw.items() if k != "cost"},
                    dict(raw, schema_version="unknown"), dict(raw, data_id=" ")]
        for field, value in (("context_tokens", True), ("supervised_tokens", -1)):
            changed = deepcopy(raw)
            changed["cost"][field] = value
            variants.append(changed)
        for changed in variants:
            with self.subTest(changed=changed), self.assertRaises(ValidationError):
                parse_labels(changed, self.ontology)
        for text in ('{"data_id":"a","data_id":"b"}', '{"nll":NaN}', '{"nll":Infinity}'):
            with self.assertRaises(ValidationError):
                loads(text)

    def test_quality_flags_only_boolean_or_null(self) -> None:
        for value in (False, None):
            self.values["labels"][0]["quality_flags"]["quality_pass"] = value
            report = self.select().to_dict()["candidates"][0]
            self.assertFalse(report["eligibility_pass"])
            self.assertIsNone(report["score"])
            self.assertIsNone(report["ranking_position"])
        self.values["labels"][0]["quality_flags"]["quality_pass"] = 1
        with self.assertRaisesRegex(ValidationError, "quality_flags"):
            self.inputs()

    def test_unknown_error_action(self) -> None:
        self.values["labels"][0]["quality_flags"]["protocol_valid"] = None
        self.values["policy"]["eligibility"]["unknown_flag_action"] = "error"
        with self.assertRaisesRegex(ValidationError, "protocol_valid"):
            self.select()

    def test_old_quality_lineage_and_zero_supervision_remain_excluded(self) -> None:
        self.values["data"][0]["quality"]["verification"] = "fail"
        use = {"ever_used_in_training_attempt": True, "in_accepted_lineage": True,
               "training_attempt_ids": ["old"], "cumulative_supervised_tokens": 4}
        self.values["data"][1]["usage"] = use
        self.values["labels"][1]["usage"] = deepcopy(use)
        self.values["data"][2]["supervision"]["loss_mask"] = [False] * len(self.values["data"][2]["text"])
        self.values["labels"][2]["cost"]["supervised_tokens"] = 0
        report = self.select().to_dict()
        self.assertEqual(report["counts"]["eligible"], 7)
        self.assertTrue(all("old_base_ineligible" in row["exclusion_reasons"] for row in report["candidates"][:3]))

    def test_illegal_trajectory_is_validated_before_labels(self) -> None:
        self.values["data"][0]["tokenization"]["roles"][-1] = "tool"
        with self.assertRaisesRegex(ValidationError, "正向监督"):
            self.select()

    def test_final_test_is_rejected_and_references_need_no_labels(self) -> None:
        self.assertEqual(len(self.inputs()[1]), 12)
        self.values["data"][-1]["role"] = "final_test"
        with self.assertRaisesRegex(ValidationError, "final_test"):
            self.select()

    def test_candidate_only_one_to_one_binding(self) -> None:
        original = deepcopy(self.values["labels"])
        variants = [original[:-1], original + [original[0]], original + [dict(original[0], data_id="orphan")],
                    [dict(original[0], trajectory_version="old"), *original[1:]]]
        for labels in variants:
            self.values["labels"] = labels
            with self.subTest(labels=len(labels)), self.assertRaises(ValidationError):
                self.select()

    def test_snapshot_fields_cannot_override_records(self) -> None:
        for section, key, value in (("partition", "split_group", "other"), ("cost", "context_tokens", 100),
                ("model_scores", "nll", 8), ("model_scores", "normalization_group", "other"),
                ("usage", "training_attempt_ids", ["attempt"])):
            original = deepcopy(self.values["labels"][0])
            self.values["labels"][0][section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValidationError):
                self.select()
            self.values["labels"][0] = original
        self.values["labels"][0]["skills"]["items"] = []
        with self.assertRaisesRegex(ValidationError, "skills"):
            self.select()

    def test_skill_and_ontology_validation(self) -> None:
        raw = self.values["labels"][0]
        for key, value in (("domain", "general"), ("skills", {"ontology_version": "old", "items": []})):
            with self.subTest(key=key), self.assertRaises(ValidationError):
                parse_labels(dict(raw, **{key: value}), self.ontology)
        for items in ([raw["skills"]["items"][0]] * 2,
                      [{"skill_id": "undefined", "source": "human", "evidence_refs": []}]):
            with self.assertRaises(ValidationError):
                parse_labels(dict(raw, skills={"ontology_version": self.ontology.ontology_version, "items": items}), self.ontology)
        ontology = self.ontology.to_dict()
        ontology["skills"].append(ontology["skills"][0])
        with self.assertRaises(ValidationError):
            SkillOntology.parse(ontology)

    def test_evidence_registry_required(self) -> None:
        self.values["labels"][0]["quality_annotation"]["evidence_refs"] = ["proof"]
        with self.assertRaisesRegex(ValidationError, "evidence_refs"):
            self.inputs()
        label = parse_labels(self.values["labels"][0], self.ontology, frozenset({"proof"}))
        self.assertEqual(label.quality_annotation.evidence_refs, ("proof",))

    def test_synthetic_permission_requires_flag_and_smoke(self) -> None:
        with self.assertRaisesRegex(ValidationError, "allow-synthetic-labels"):
            self.select(synthetic=False)
        self.select()
        # verified 的声明仅用来使轨迹解析达到标签权限检查。
        self.values["config"]["purpose"] = "draft"
        for record in self.values["data"]:
            record["tokenization"]["status"] = "verified"
        with self.assertRaisesRegex(ValidationError, "smoke"):
            self.select()

    def test_annotated_mode_does_not_enable_quality(self) -> None:
        for label in self.values["labels"]:
            label["provenance"].update(label_mode="annotated", annotation_seed=None)
        self.values["policy"]["eligibility"]["allowed_label_modes"] = ["annotated"]
        self.values["labels"][0]["quality_flags"]["quality_pass"] = None
        result = self.select(synthetic=False).to_dict()
        self.assertEqual(result["run_mode"], "annotated")
        self.assertFalse(result["candidates"][0]["eligibility_pass"])

    def test_missing_scores_union_and_invalid_numbers(self) -> None:
        self.values["labels"][0]["model_scores"] = {"status": "missing"}
        self.values["data"][0]["scores"] = None
        with self.assertRaisesRegex(ValidationError, "缺少当前分数"):
            self.select()
        self.values["labels"][0]["model_scores"]["nll"] = 0
        with self.assertRaisesRegex(ValidationError, "未知字段"):
            self.inputs()
        for key, value in (("nll", float("nan")), ("entropy", -1), ("nll_percentile", 1.1), ("entropy_percentile", True)):
            raw = deepcopy(fixture_values()["labels"][0])
            raw["model_scores"][key] = value
            with self.subTest(key=key), self.assertRaises(ValidationError):
                parse_labels(raw, self.ontology)

    def test_fixture_is_deterministic_and_does_not_touch_global_rng(self) -> None:
        state = random.getstate()
        self.assertEqual(fixture_values(42), self.values)
        self.assertNotEqual(fixture_values(42)["labels"], fixture_values(43)["labels"])
        self.assertEqual(random.getstate(), state)
        self.assertEqual(len(self.values["labels"]), 10)
        self.assertTrue(all(all(x["quality_flags"][k] is True for k in FLAGS) for x in self.values["labels"]))
        self.values["data"].reverse()
        self.values["labels"].reverse()
        result = self.select().to_dict()
        self.values["data"].reverse()
        self.values["labels"].reverse()
        self.assertEqual(self.select().to_dict(), result)

    def test_labels_are_immutable_and_exports_are_independent(self) -> None:
        label = self.inputs()[2][0]
        before = canonical(label)
        exported = label.to_dict()
        exported["skills"]["items"].clear()
        with self.assertRaises(FrozenInstanceError):
            label.quality_flags.quality_pass = False
        self.assertEqual(canonical(label), before)
