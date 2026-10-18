"""完整轨迹精确预算、排名优先和输入边界。"""

from dataclasses import replace
import itertools

from rsi.comparison import build_comparison
from rsi.data import TrajectoryRecord
from rsi.contracts import ExperimentContract
from rsi.packing import exact_ranked
from rsi.validation import ValidationError
from tests.helpers import Fixture


class ExactPackingTests(Fixture):
    def records(self, lengths: list[int] | tuple[int, ...]) -> list[TrajectoryRecord]:
        return [replace(record, supervised_tokens=length) for record, length in zip(self.context().eligible(), lengths)]

    def test_skips_preferred_item_only_when_it_blocks_exact_budget(self) -> None:
        records = self.records([6, 5, 4, 3])
        result = exact_ranked(records, 8)
        self.assertEqual([r.data_id for r in result], [records[1].data_id, records[3].data_id])
        self.assertEqual(sum(r.supervised_tokens for r in result), 8)

    def test_small_cases_match_exhaustive_lexicographic_optimum(self) -> None:
        for lengths in itertools.product(range(1, 5), repeat=4):
            records = self.records(lengths)
            for budget in (3, 5, 7):
                feasible = [bits for bits in itertools.product((0, 1), repeat=4)
                            if sum(bit * length for bit, length in zip(bits, lengths)) == budget]
                if not feasible:
                    with self.assertRaises(ValidationError):
                        exact_ranked(records, budget)
                    continue
                result = exact_ranked(records, budget)
                self.assertEqual(tuple(int(r in result) for r in records), max(feasible))

    def test_duplicate_groups_and_resource_excess_are_rejected(self) -> None:
        records = self.records([1, 2, 3, 4])
        with self.assertRaisesRegex(ValidationError, "去重"):
            exact_ranked([records[0], replace(records[1], duplicate_group=records[0].duplicate_group)], 3)
        with self.assertRaisesRegex(ValidationError, "规模"):
            exact_ranked(records, 2_000_000)

    def test_schema_11_is_explicit_and_old_contracts_stay_greedy(self) -> None:
        self.assertEqual(self.contract.packing_rule, "greedy_v1")
        self.config["selection"]["packing_rule"] = "exact_ranked_v1"
        with self.assertRaisesRegex(ValidationError, "未知字段"):
            ExperimentContract.parse(self.config)
        self.config["schema_version"] = "1.1"
        self.config["selection"]["allowed_shortfall"] = 0
        plan = build_comparison(self.context(), "run", "compare")
        self.assertEqual(plan.arms.random.supervised_tokens, plan.arms.fixed.supervised_tokens)
        self.config["selection"]["allowed_shortfall"] = 1
        with self.assertRaisesRegex(ValidationError, "偏差"):
            ExperimentContract.parse(self.config)
