"""经验百分位与因果目标位置的检查。"""

from rsi.hf_backend import masked_targets
from rsi.scoring import empirical_percentile
from rsi.validation import ValidationError
from tests.helpers import Fixture


class ScoringTests(Fixture):
    def test_midrank_ties_and_endpoints(self) -> None:
        self.assertEqual(empirical_percentile(2, [1,2,2,3]), 0.5)
        self.assertEqual(empirical_percentile(0, [1,2,3]), 0.0)
        self.assertEqual(empirical_percentile(4, [1,2,3]), 1.0)

    def test_reference_must_be_nonempty_and_finite(self) -> None:
        for reference in ([], [float('nan')]):
            with self.assertRaises(ValidationError):
                empirical_percentile(1, reference)

    def test_causal_targets_shift_once(self) -> None:
        record = self.context().records[0]
        positions, targets = masked_targets(record)
        self.assertEqual(positions, [10,11,12,13])
        self.assertEqual(targets, [ord(c) for c in 'fix\n'])
