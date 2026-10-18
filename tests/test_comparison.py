"""对照计划的纯计算与输入检查服务。"""

from rsi.comparison import build_comparison
from rsi.inspection import inspect_scores
from rsi.validation import ValidationError, canonical, loads
from tests.helpers import Fixture


class ComparisonTests(Fixture):
    def test_plan_needs_no_database_and_preserves_json_format(self) -> None:
        plan = build_comparison(self.context(), "run", "pair")
        payload = plan.to_dict()
        self.assertEqual(payload, loads(canonical(plan)))
        self.assertEqual(payload["arms"]["fixed"]["supervised_tokens"], 8)
        self.assertEqual(payload["arms"]["random"]["supervised_tokens"], 8)
        self.assertIsNone(payload["metrics"])
        self.assertEqual(payload["training_status"], "NOT_RUN")

    def test_insufficient_total_budget_rejects_plan(self) -> None:
        self.config["budgets"]["training_tokens"] = 15
        with self.assertRaisesRegex(ValidationError, "总预算不足"):
            build_comparison(self.context(), "run", "pair")

    def test_strict_inspection_reuses_eligibility_rule(self) -> None:
        self.data[0]["scores"] = None
        self.data[0]["quality"]["verification"] = "fail"
        context = self.context()
        report = inspect_scores(context.contract, context.records, require_current_scores=True)
        self.assertEqual(report.current_score_records, 3)
        self.assertEqual(report.stale_or_missing_score_ids, ())

    def test_inspection_reports_stale_scores_then_strict_mode_rejects(self) -> None:
        self.data[0]["scores"]["checkpoint_id"] = "old"
        context = self.context()
        report = inspect_scores(context.contract, context.records)
        self.assertEqual(report.stale_or_missing_score_ids, ("a",))
        with self.assertRaisesRegex(ValidationError, "过期"):
            inspect_scores(context.contract, context.records, require_current_scores=True)
