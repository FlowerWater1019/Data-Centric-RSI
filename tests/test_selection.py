"""按模块检查可观察行为。"""

from copy import deepcopy

from rsi.selection import FixedSelector, RandomSelector, build_batch
from rsi.validation import ValidationError


from tests.helpers import Fixture


class SelectionTests(Fixture):
    def test_random_is_reproducible_and_order_independent(self) -> None:
        one = build_batch(self.context(), RandomSelector())
        two = build_batch(self.context(list(reversed(self.data))), RandomSelector())
        self.assertEqual(one.items, two.items)
        self.assertEqual(one.fingerprint, two.fingerprint)

    def test_fixed_weighted_formula(self) -> None:
        result = build_batch(self.context(), FixedSelector())
        self.assertEqual([r.data_id for r in result.items], ["c", "b"])
        self.assertEqual(result.supervised_tokens, 8)

    def test_stale_checkpoint_mask_and_cdf_are_rejected(self) -> None:
        for field in ("checkpoint_id", "mask_version", "cdf_version", "trajectory_version", "reference_version"):
            data = deepcopy(self.data)
            data[0]["scores"][field] = "outdated"
            with self.subTest(field=field), self.assertRaisesRegex(ValidationError, "过期"):
                build_batch(self.context(data), FixedSelector())

    def test_missing_scores_allowed_for_random_but_not_fixed(self) -> None:
        self.data[0]["scores"] = None
        build_batch(self.context(), RandomSelector())
        with self.assertRaisesRegex(ValidationError, "缺少"):
            build_batch(self.context(), FixedSelector())

    def test_all_holdout_roles_are_excluded(self) -> None:
        self.assertEqual({r.data_id for r in self.context().eligible()}, {"a", "b", "c", "d"})

    def test_accepted_lineage_is_excluded(self) -> None:
        self.data[2]["usage"] = {"ever_used_in_training_attempt": True, "in_accepted_lineage": True,
                                 "training_attempt_ids": ["old"], "cumulative_supervised_tokens": 4}
        self.assertNotIn("c", [r.data_id for r in self.context().eligible()])

    def test_failed_quality_is_excluded(self) -> None:
        self.data[2]["quality"]["verification"] = "fail"
        self.assertNotIn("c", [r.data_id for r in self.context().eligible()])

    def test_quota_failure_is_not_silently_relaxed(self) -> None:
        self.config["selection"]["min_skill_shares"] = {"failure_recovery": 1.0}
        with self.assertRaisesRegex(ValidationError, "没有自动放宽"):
            build_batch(self.context(), FixedSelector())

    def test_unreachable_budget_reports_search_failure(self) -> None:
        self.config["selection"]["supervised_token_budget"] = 7
        with self.assertRaisesRegex(ValidationError, "未证明"):
            build_batch(self.context(), FixedSelector())
