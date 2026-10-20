"""调度白名单、证据、不可变性和模拟反馈边界。"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

from rsi.dispatch import apply_dispatch, parse_dispatch
from rsi.feedback import parse_feedback
from rsi.policy import parse_policy
from rsi.validation import ValidationError, canonical, digest
from tests.selection_helpers import SelectionFixture


class DispatchTests(SelectionFixture):
    def test_valid_patch_and_repeat_are_pure_and_deterministic(self) -> None:
        context = self.dispatch_context()
        original = canonical(context[0])
        first = apply_dispatch(*context)
        second = apply_dispatch(*context)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.policy.to_dict(), second.policy.to_dict())
        self.assertEqual(canonical(context[0]), original)
        self.assertEqual(first.policy.parent_policy_id, context[0].policy_id)
        self.assertEqual(first.policy.created_by.dispatch_id, context[1].dispatch_id)
        self.assertEqual(first.to_dict()["model_operation"], "recommendation_only")
        self.assertEqual(len(first.diff), 2)

    def test_complete_preferences_and_weights_replacements(self) -> None:
        self.values["dispatch"]["policy_update"]["set"] = {
            "scoring.nll_preference": {"support": [.1, 1], "preferred": [.3, .8]},
            "scoring.entropy_preference": {"support": [0, 1], "preferred": [.2, .7]},
            "scoring.weights": {"nll": .5, "entropy": .4, "skill": .1},
            "batch_constraints.min_task_type_token_shares.code_generation": 1,
            "batch_constraints.max_task_type_token_shares.code_generation": 1}
        result = self.apply()
        self.assertEqual(result.policy.scoring.weights["nll"], .5)
        self.assertEqual(result.policy.scoring.nll_preference.support, (.1, 1))
        self.assertEqual(len(result.diff), 5)

    def test_frozen_and_arbitrary_paths_rejected(self) -> None:
        for path in ("domain", "execution.seed", "execution.packing_rule", "batch_constraints.supervised_token_budget",
                "eligibility.allowed_label_modes", "scoring.cdf_version", "scoring.weights.nll",
                "scoring.nll_preference.support.0", "model.checkpoint_id", "__class__", "scoring.skill_preferences.undefined"):
            self.values["dispatch"]["policy_update"]["set"] = {path: 1}
            with self.subTest(path=path), self.assertRaisesRegex(ValidationError, "白名单"):
                self.apply()

    def test_invalid_patch_rolls_back_entire_parent(self) -> None:
        self.values["dispatch"]["policy_update"]["set"]["scoring.weights"] = {"nll": .9, "entropy": .9, "skill": .9}
        context = self.dispatch_context()
        before = canonical(context[0])
        with self.assertRaisesRegex(ValidationError, "总和"):
            apply_dispatch(*context)
        self.assertEqual(canonical(context[0]), before)

    def test_partial_replacements_and_nonfinite_values_rejected(self) -> None:
        for value in ({"scoring.weights": {"nll": 1}}, {"scoring.nll_preference": {"preferred": [.3, .8]}},
                      {"scoring.skill_preferences.regression_check": True}, {"scoring.skill_preferences.regression_check": float("nan")}):
            self.values["dispatch"]["policy_update"]["set"] = value
            with self.subTest(value=value), self.assertRaises(ValidationError):
                self.apply()

    def test_add_remove_are_sorted_set_operations(self) -> None:
        update = self.values["dispatch"]["policy_update"]
        update["add"] = {"eligibility.excluded_data_ids": ["z", "a"]}
        update["remove"] = {"eligibility.excluded_data_ids": ["absent"]}
        first = self.apply()
        self.assertEqual(first.policy.eligibility.excluded_data_ids, ("a", "z"))
        self.values["policy"] = first.policy.to_dict()
        self.values["dispatch"]["inputs"]["current_policy_id"] = first.policy.policy_id
        update.update(base_policy_id=first.policy.policy_id, proposed_policy_id="policy_003")
        second = self.apply()
        self.assertEqual(second.policy.eligibility.excluded_data_ids, ("a", "z"))
        self.assertFalse(any(x.path == "eligibility.excluded_data_ids" for x in second.diff))

    def test_list_conflicts_duplicates_and_unknown_batch_are_rejected(self) -> None:
        update = self.values["dispatch"]["policy_update"]
        for adds, removes in (({"eligibility.excluded_data_ids": ["a"]}, {"eligibility.excluded_data_ids": ["a"]}),
                ({"eligibility.excluded_data_ids": ["a", "a"]}, {}),
                ({"eligibility.deferred_batch_ids": ["unknown"]}, {}),
                ({"description": ["a"]}, {})):
            update["add"], update["remove"] = adds, removes
            with self.subTest(adds=adds), self.assertRaises(ValidationError):
                self.apply()

    def test_parent_and_proposed_id_checks(self) -> None:
        update = self.values["dispatch"]["policy_update"]
        update["base_policy_id"] = "wrong"
        with self.assertRaisesRegex(ValidationError, "父策略"):
            self.apply()
        update["base_policy_id"] = "policy_001"
        update["proposed_policy_id"] = "policy_001"
        with self.assertRaisesRegex(ValidationError, "父策略"):
            self.apply()

    def test_feedback_identity_and_controller_match(self) -> None:
        context = self.dispatch_context()
        for field in ("feedback_id", "round_id", "batch_id", "parent_checkpoint_id", "candidate_checkpoint_id"):
            feedback = replace(context[2], **{field: "wrong"})
            with self.subTest(field=field), self.assertRaises(ValidationError):
                apply_dispatch(context[0], context[1], feedback, *context[3:])
        dispatch = replace(context[1], controller=replace(context[1].controller, checkpoint_id="wrong"))
        with self.assertRaisesRegex(ValidationError, "父检查点"):
            apply_dispatch(context[0], dispatch, *context[2:])

    def test_evidence_reference_and_evidence_type_checks(self) -> None:
        self.values["dispatch"]["diagnosis"]["observations"][0]["evidence_ref"] = "missing"
        with self.assertRaisesRegex(ValidationError, "证据引用"):
            self.apply()
        self.values["dispatch"]["diagnosis"]["observations"][0]["evidence_ref"] = "fit_nll"
        self.values["dispatch"]["model_action"]["acceptance_evidence_ref"] = "fit_nll"
        with self.assertRaisesRegex(ValidationError, "验收结果"):
            self.apply()
        self.values["feedback"]["data_role"] = "final_test"
        with self.assertRaisesRegex(ValidationError, "data_role"):
            self.apply()

    def test_inconclusive_cannot_accept_and_reject_must_match(self) -> None:
        self.values["feedback"]["acceptance"]["status"] = "inconclusive"
        for action in ("accept", "reject"):
            self.values["dispatch"]["model_action"]["recommendation"] = action
            with self.subTest(action=action), self.assertRaisesRegex(ValidationError, "验收状态"):
                self.apply()
        self.values["dispatch"]["model_action"]["recommendation"] = "hold"
        self.apply()
        self.values["feedback"]["acceptance"]["status"] = "accepted"
        self.values["dispatch"]["model_action"]["recommendation"] = "accept"
        self.assertEqual(self.apply().to_dict()["model_operation"], "recommendation_only")

    def test_unknown_history_and_future_references_are_rejected(self) -> None:
        self.values["dispatch"]["inputs"]["history_dispatch_ids"] = ["future"]
        with self.assertRaisesRegex(ValidationError, "未来或未知"):
            self.apply()

    def test_explicit_history_evidence_resolves_without_guessing(self) -> None:
        context = self.dispatch_context()
        old_dispatch = replace(context[1], dispatch_id="old")
        new_diagnosis = replace(context[1].diagnosis,
                                observations=(replace(context[1].diagnosis.observations[0], evidence_ref="old:fit_nll"),))
        dispatch = replace(context[1], inputs=replace(context[1].inputs, history_dispatch_ids=("old",)), diagnosis=new_diagnosis)
        result = apply_dispatch(context[0], dispatch, *context[2:], (old_dispatch,), (context[2],))
        self.assertEqual(result.policy.parent_policy_id, context[0].policy_id)
        with self.assertRaisesRegex(ValidationError, "历史调度缺少"):
            apply_dispatch(context[0], dispatch, *context[2:], (old_dispatch,))

    def test_empty_patch_and_natural_language_do_not_change_selection(self) -> None:
        self.values["dispatch"]["policy_update"]["set"] = {}
        self.values["dispatch"]["natural_language_dispatch"] = "这只是待人审阅的说明。"
        before = self.select()
        result = self.apply()
        self.assertEqual(result.diff, ())
        self.values["policy"] = result.policy.to_dict()
        after = self.select()
        self.assertEqual(before.batch.items, after.batch.items)

    def test_feedback_counts_and_missing_rates(self) -> None:
        raw = deepcopy(self.values["feedback"])
        raw["execution"]["target"] = {"task_count": 0, "before_passed": 0, "after_passed": 0}
        feedback = parse_feedback(raw, self.ontology)
        self.assertEqual(feedback.execution.target.rates(), {"status": "MISSING", "before": None, "after": None})
        for field, value in (("task_count", True), ("before_passed", 5), ("after_passed", -1)):
            variant = deepcopy(self.values["feedback"])
            variant["execution"]["target"][field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                parse_feedback(variant, self.ontology)

    def test_feedback_and_dispatch_reject_extra_fields_and_immutable_exports(self) -> None:
        for key, parser in (("feedback", lambda x: parse_feedback(x, self.ontology)), ("dispatch", parse_dispatch)):
            with self.subTest(key=key), self.assertRaises(ValidationError):
                parser(dict(self.values[key], unknown=True))
        context = self.dispatch_context()
        exported = context[1].to_dict()
        exported["policy_update"]["set"].clear()
        self.assertEqual(len(context[1].policy_update.sets), 2)
        with self.assertRaises(TypeError):
            context[2].execution.by_skill["regression_check"] = None
        with self.assertRaises(FrozenInstanceError):
            context[1].controller.checkpoint_id = "wrong"
