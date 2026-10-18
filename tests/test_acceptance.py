"""验收使用任务级配对证据。合成输入仅用于检查规则。"""

from dataclasses import replace
import unittest

from rsi.acceptance import AcceptanceRule, TaskOutcome, decide
from rsi.validation import ValidationError, read_json
from tests.helpers import ROOT


class AcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rule = AcceptanceRule.parse(read_json(ROOT / 'configs/acceptance.engineering.json'))

    def outcomes(self, target: bool, retention: bool, n: int = 10000) -> tuple[TaskOutcome, ...]:
        return tuple(TaskOutcome(f'{group}_{i}', group, passed)
                     for group, passed in (('target', target), ('retention', retention)) for i in range(n))

    def test_clear_gain_and_retention_are_accepted(self) -> None:
        result = decide(self.outcomes(False, True), self.outcomes(True, True), self.rule)
        self.assertEqual(result.status, 'accepted')
        self.assertEqual(result.target.sample_count, 10000)

    def test_clear_damage_is_rejected(self) -> None:
        result = decide(self.outcomes(False, True), self.outcomes(True, False), self.rule)
        self.assertEqual(result.status, 'rejected')

    def test_tiny_perfect_gain_still_is_inconclusive(self) -> None:
        result = decide(self.outcomes(False, True, 2), self.outcomes(True, True, 2), self.rule)
        self.assertEqual(result.status, 'inconclusive')
        self.assertEqual(result.target.lower, -1)

    def test_unchanged_model_is_not_automatically_accepted(self) -> None:
        baseline = self.outcomes(True, True, 16)
        self.assertEqual(decide(baseline, baseline, self.rule).status, 'inconclusive')

    def test_pairing_is_by_task_id(self) -> None:
        baseline, candidate = self.outcomes(False, True), self.outcomes(True, True)
        self.assertEqual(decide(baseline, candidate, self.rule), decide(baseline, candidate[::-1], self.rule))

    def test_mismatched_or_duplicated_tasks_are_rejected(self) -> None:
        baseline = self.outcomes(True, True, 2)
        for candidate in (baseline[:-1], baseline + baseline[:1],
                          (replace(baseline[0], group='retention'), *baseline[1:])):
            with self.assertRaises(ValidationError):
                decide(baseline, candidate, self.rule)

    def test_retests_and_invalid_alpha_are_rejected(self) -> None:
        for key, value in [('alpha', 0), ('alpha', 1), ('max_retests', 1), ('minimum_gain', -0.1)]:
            raw = read_json(ROOT / 'configs/acceptance.engineering.json')
            raw[key] = value
            with self.assertRaises(ValidationError):
                AcceptanceRule.parse(raw)
