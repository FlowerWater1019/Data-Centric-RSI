"""策略适配、排名、预算、配额和上下文身份。"""

from copy import deepcopy
from dataclasses import replace

from rsi.contracts import Preference
from rsi.history import BatchHistory, HistoricalBatch
from rsi.policy import adapt_policy, parse_policy
from rsi.policy_selection import validate_selection
from rsi.selection import FixedSelector, SelectionContext, build_batch, score_terms, utility
from rsi.validation import ValidationError, digest
from tests.selection_helpers import SelectionFixture


class PolicySelectionTests(SelectionFixture):
    def test_validated_context_keeps_label_and_history_identity_when_reused_by_batch_builder(self) -> None:
        first = validate_selection(*self.inputs(), allow_synthetic_labels=True)
        self.assertEqual(first.context_id, self.select().batch.context_id)
        self.values["labels"][0]["label_version"] = "labels-v2"
        second = validate_selection(*self.inputs(), allow_synthetic_labels=True)
        history = BatchHistory(batches=(HistoricalBatch("old", "p", "l", "d", ("sample_010",)),))
        third = validate_selection(*self.inputs(), history, allow_synthetic_labels=True)
        self.assertEqual(len({x.context_id for x in (first, second, third)}), 3)
        batches = [build_batch(context, FixedSelector()) for context in (first, second, third)]
        self.assertEqual(len({batch.context_id for batch in batches}), 3)
        self.assertEqual(len({batch.fingerprint for batch in batches}), 1)

    def test_trapezoid_boundary_values(self) -> None:
        preference = Preference((.2, 1), (.4, .8))
        self.assertEqual([utility(x, preference) for x in (0, .2, .4, .6, .8, 1)], [0, 0, 1, 1, 1, 0])
        self.assertAlmostEqual(utility(.3, preference), .5)
        self.assertAlmostEqual(utility(.9, preference), .5)

    def test_adapter_leaves_original_contract_and_inputs_unchanged(self) -> None:
        contract, records, labels, ontology, policy = self.inputs()
        original = digest(contract)
        config = deepcopy(self.values["config"])
        derived = adapt_policy(contract, policy)
        self.assertNotEqual(derived.contract_id, contract.contract_id)
        self.assertEqual(digest(contract), original)
        self.assertEqual(self.values["config"], config)
        self.assertIs(derived.model, contract.model)
        self.assertEqual(derived.seed, policy.execution.seed)

    def test_policy_validation_rejects_wrong_weights_intervals_and_flags(self) -> None:
        base = deepcopy(self.values["policy"])
        for section, key, value in (("scoring", "weights", {"nll": .5, "entropy": .5, "skill": True}),
              ("scoring", "weights", {"nll": .5, "entropy": .5, "skill": .5}),
              ("scoring", "nll_preference", {"support": [0, 1], "preferred": [0, .8]}),
              ("scoring", "skill_preferences", {"undefined": 1}),
              ("execution", "seed", True), ("execution", "quota_handling", "optimize"),
              ("batch_constraints", "supervised_token_budget", True),
              ("batch_constraints", "allow_within_batch_repetition", True)):
            value_raw = deepcopy(base)
            value_raw[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValidationError):
                parse_policy(value_raw, self.ontology)

    def test_frozen_budget_and_scoring_versions(self) -> None:
        self.values["policy"]["batch_constraints"]["supervised_token_budget"] = 20
        with self.assertRaisesRegex(ValidationError, "冻结实验预算"):
            self.select()
        self.values["policy"]["batch_constraints"]["supervised_token_budget"] = 16
        self.values["policy"]["scoring"]["cdf_version"] = "old"
        with self.assertRaisesRegex(ValidationError, "评分版本"):
            self.select()

    def test_stale_score_is_rejected_even_with_zero_numeric_weight(self) -> None:
        self.values["policy"]["scoring"]["weights"] = {"nll": 0, "entropy": 0, "skill": 1}
        self.values["data"][0]["scores"]["checkpoint_id"] = "old"
        self.values["labels"][0]["model_scores"]["checkpoint_id"] = "old"
        with self.assertRaisesRegex(ValidationError, "过期"):
            self.select()

    def test_filtered_candidate_does_not_require_scores(self) -> None:
        self.values["data"][0]["scores"] = None
        self.values["labels"][0]["model_scores"] = {"status": "missing"}
        self.values["policy"]["eligibility"]["excluded_data_ids"] = ["sample_001"]
        report = self.select().to_dict()["candidates"][0]
        self.assertIsNone(report["score"])
        self.assertFalse(report["entered_scoring"])

    def test_max_skill_matching_empty_skill_zero_and_stable_ties(self) -> None:
        self.values["data"][0]["skills"] = ["code_generation", "error_analysis"]
        item = self.values["labels"][0]["skills"]["items"][0]
        self.values["labels"][0]["skills"]["items"] = [{**item, "skill_id": k} for k in self.values["data"][0]["skills"]]
        self.values["data"][1]["skills"] = []
        self.values["labels"][1]["skills"]["items"] = []
        self.values["policy"]["scoring"]["weights"] = {"nll": 0, "entropy": 0, "skill": 1}
        report = self.select().to_dict()
        self.assertEqual(report["candidates"][0]["score"]["total"], .7)
        self.assertEqual(report["candidates"][1]["score"]["total"], 0)
        self.values["policy"]["scoring"]["skill_preferences"] = {}
        ranked = sorted(self.select().to_dict()["candidates"], key=lambda r: r["ranking_position"])
        self.assertEqual([r["data_id"] for r in ranked], sorted(r["data_id"] for r in ranked))

    def test_whole_trajectory_budget_and_unreachable_failure(self) -> None:
        result = self.select()
        self.assertEqual(result.batch.supervised_tokens, 16)
        self.assertEqual(len(result.batch.items), 4)
        self.assertTrue(all(item.supervised_tokens == 4 and item.repetitions == 1 for item in result.batch.items))
        self.values["policy"]["batch_constraints"]["supervised_token_budget"] = 15
        failed = self.select()
        self.assertIsNone(failed.batch)
        self.assertEqual(failed.to_dict()["status"], "BUDGET_FAILED")
        self.assertIn("没有截断", failed.to_dict()["failure_reason"])

    def test_duplicate_pool_is_input_error(self) -> None:
        self.values["data"][1]["duplicate_group"] = self.values["data"][0]["duplicate_group"]
        self.values["labels"][1]["partition"]["duplicate_group"] = self.values["data"][0]["duplicate_group"]
        with self.assertRaisesRegex(ValidationError, "去重"):
            self.select()

    def test_skill_quota_postcheck_does_not_reorder(self) -> None:
        self.values["policy"]["batch_constraints"]["min_skill_token_shares"] = {"regression_check": .25}
        report = self.select().to_dict()
        self.assertEqual(report["status"], "QUOTA_FAILED")
        self.assertIsNone(report["batch"])
        self.assertIn("没有自动放宽", report["failure_reason"])

    def test_task_type_min_max_and_overlapping_skill_shares(self) -> None:
        # 两个合法技能同时覆盖整个批次，share 总和允许超过 1。
        for raw, label in zip(self.values["data"][:10], self.values["labels"]):
            raw["skills"] = ["code_generation", "regression_check"]
            item = label["skills"]["items"][0]
            label["skills"]["items"] = [{**item, "skill_id": k} for k in raw["skills"]]
        self.values["policy"]["batch_constraints"]["min_skill_token_shares"] = {"code_generation": 1, "regression_check": 1}
        result = self.select()
        self.assertEqual(sum(result.batch.skill_shares.values()), 2)
        self.values["policy"]["batch_constraints"]["max_task_type_token_shares"] = {"code_generation": .5}
        self.assertEqual(self.select().to_dict()["status"], "QUOTA_FAILED")
        self.values["policy"]["batch_constraints"]["min_task_type_token_shares"] = {"code_generation": .6}
        with self.assertRaisesRegex(ValidationError, "最低比例"):
            self.select()

    def test_task_minimum_total_and_registered_types(self) -> None:
        self.values["policy"]["eligibility"]["allowed_task_types"].append("repair")
        b = self.values["policy"]["batch_constraints"]
        b["min_task_type_token_shares"] = {"code_generation": .6, "repair": .6}
        with self.assertRaisesRegex(ValidationError, "总和"):
            self.select()
        b["min_task_type_token_shares"] = {"unregistered": .1}
        with self.assertRaisesRegex(ValidationError, "未知字段"):
            self.select()

    def test_reports_all_exclusion_reasons(self) -> None:
        self.values["data"][0]["quality"]["verification"] = "fail"
        self.values["labels"][0]["quality_flags"].update(quality_pass=False, result_verified=None)
        self.values["labels"][0]["task"].update(language="java", harness_protocol="other", task_type="repair", trajectory_type="tool_interaction")
        self.values["policy"]["eligibility"]["excluded_data_ids"] = ["sample_001"]
        self.values["policy"]["batch_constraints"]["max_context_tokens_per_sample"] = 1
        report = self.select().to_dict()["candidates"][0]
        self.assertEqual(len(report["exclusion_reasons"]), 9)
        self.assertIsNone(report["raw_scores"])

    def test_deferred_history_resolution_and_release(self) -> None:
        history = BatchHistory(batches=(HistoricalBatch("old", "p", "l", "d", ("sample_002",)),))
        self.values["policy"]["eligibility"]["deferred_batch_ids"] = ["missing"]
        with self.assertRaisesRegex(ValidationError, "历史批次"):
            self.select(history)
        self.values["policy"]["eligibility"]["deferred_batch_ids"] = ["old"]
        self.assertIn("deferred_batch", self.select(history).to_dict()["candidates"][1]["exclusion_reasons"])
        self.values["policy"]["eligibility"]["deferred_batch_ids"] = []
        self.assertTrue(self.select(history).to_dict()["candidates"][1]["eligibility_pass"])

    def test_context_hash_covers_labels_policy_history_and_member_fingerprint(self) -> None:
        first = self.select()
        self.values["labels"][0]["label_version"] = "labels-v2"
        second = self.select()
        self.values["policy"]["description"] = "新的描述。"
        third = self.select()
        history = BatchHistory(batches=(HistoricalBatch("old", "p", "l", "d", ("sample_010",)),))
        fourth = self.select(history)
        self.assertEqual(len({r.batch.context_id for r in (first, second, third, fourth)}), 4)
        self.assertEqual(len({r.batch.fingerprint for r in (first, second, third, fourth)}), 1)

    def test_legacy_manifest_members_equal_adapter_result(self) -> None:
        contract, records, labels, ontology, policy = self.inputs()
        derived = adapt_policy(contract, policy)
        legacy = build_batch(SelectionContext(derived, records, digest(derived), digest(records)), FixedSelector())
        result = self.select()
        self.assertEqual(legacy.items, result.batch.items)
        self.assertEqual(legacy.fingerprint, result.batch.fingerprint)
        exported = result.to_dict()
        exported["candidates"].clear()
        self.assertEqual(len(result.to_dict()["candidates"]), 10)
