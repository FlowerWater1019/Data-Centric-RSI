"""已校验对象不能因调用方修改输入或输出而改变。"""

from dataclasses import FrozenInstanceError, replace

from rsi.data import TrajectoryRecord
from rsi.selection import FixedSelector, SelectionContext, build_batch
from rsi.validation import ValidationError
from tests.helpers import Fixture


class ImmutabilityTests(Fixture):
    def test_input_scores_do_not_alias_validated_record(self) -> None:
        context = self.context()
        before = build_batch(context, FixedSelector()).to_dict()
        self.data[0]["scores"]["nll_percentile"] = 0.5
        self.data[0]["supervision"]["loss_mask"][-1] = False
        self.assertEqual(build_batch(context, FixedSelector()).to_dict(), before)
        self.assertTrue(context.records[0].raw["supervision"]["loss_mask"][-1])

    def test_raw_property_returns_independent_copy(self) -> None:
        record = self.context().records[0]
        raw = record.raw
        raw["scores"]["checkpoint_id"] = "changed"
        self.assertEqual(record.raw["scores"]["checkpoint_id"], self.contract.model.checkpoint_id)
        record.require_current_scores(self.contract)

    def test_config_maps_and_structures_cannot_be_mutated(self) -> None:
        for mapping, key in (
            (self.contract.budgets, "training_tokens"),
            (self.contract.fixed_strategy.weights, "nll"),
            (self.contract.min_skill_shares, "regression_check"),
        ):
            with self.subTest(key=key), self.assertRaises(TypeError):
                mapping[key] = 0
        with self.assertRaises(FrozenInstanceError):
            self.contract.training.world_size = 1

    def test_score_fields_cannot_be_mutated(self) -> None:
        score = self.context().records[0].scores
        with self.assertRaises(FrozenInstanceError):
            score.nll_percentile = 0.5

    def test_exported_batch_can_be_modified_without_changing_manifest(self) -> None:
        batch = build_batch(self.context(), FixedSelector())
        original = batch.to_dict()
        changed = batch.to_dict()
        changed["items"][0]["supervised_tokens"] = 100
        changed["model"]["checkpoint_id"] = "changed"
        self.assertEqual(batch.to_dict(), original)
        with self.assertRaises(FrozenInstanceError):
            batch.items[0].supervised_tokens = 100

    def test_selector_cannot_replace_validated_record(self) -> None:
        class AlteredSelector:
            name = "altered"

            def rank(self, context: SelectionContext) -> list[TrajectoryRecord]:
                records = context.eligible()
                records[0] = replace(records[0], supervised_tokens=1)
                return records

        with self.assertRaisesRegex(ValidationError, "不能修改"):
            build_batch(self.context(), AlteredSelector())

    def test_selector_cannot_drop_or_repeat_records(self) -> None:
        class DuplicateSelector:
            name = "duplicate"

            def rank(self, context: SelectionContext) -> list[TrajectoryRecord]:
                records = context.eligible()
                return [records[0]] * len(records)

        with self.assertRaisesRegex(ValidationError, "完整返回"):
            build_batch(self.context(), DuplicateSelector())
