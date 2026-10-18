"""按模块检查可观察行为。"""


from rsi.contracts import ExperimentContract
from rsi.validation import ValidationError, loads


from tests.helpers import Fixture


class ContractTests(Fixture):
    def test_hardware_target_is_user_reported(self) -> None:
        self.assertEqual(self.contract.hardware.gpu_count, 8)
        self.assertEqual(self.contract.hardware.memory_gb_per_gpu, 144)
        self.assertEqual(self.contract.hardware.evidence, "user_reported")

    def test_unknown_fields_are_rejected(self) -> None:
        self.config["unlimited_budget"] = True
        with self.assertRaisesRegex(ValidationError, "未知字段"):
            ExperimentContract.parse(self.config)

    def test_boolean_is_not_a_budget(self) -> None:
        self.config["budgets"]["training_tokens"] = True
        with self.assertRaises(ValidationError):
            ExperimentContract.parse(self.config)

    def test_nonfinite_weights_are_rejected(self) -> None:
        self.config["fixed_strategy"]["weights"]["nll"] = float("nan")
        with self.assertRaises(ValidationError):
            ExperimentContract.parse(self.config)

    def test_invalid_weight_sum(self) -> None:
        self.config["fixed_strategy"]["weights"]["nll"] = .9
        with self.assertRaisesRegex(ValidationError, "总和"):
            ExperimentContract.parse(self.config)

    def test_invalid_preference_interval(self) -> None:
        self.config["fixed_strategy"]["nll"]["preferred"] = [.1, .75]
        with self.assertRaisesRegex(ValidationError, "区间"):
            ExperimentContract.parse(self.config)

    def test_overlapping_skill_quotas_are_allowed(self) -> None:
        self.config["selection"]["min_skill_shares"] = {"regression_check": .8, "failure_recovery": .8}
        ExperimentContract.parse(self.config)

    def test_world_size_cannot_exceed_hardware(self) -> None:
        self.config["training"]["world_size"] = 9
        with self.assertRaisesRegex(ValidationError, "GPU"):
            ExperimentContract.parse(self.config)

    def test_formal_training_fails_closed(self) -> None:
        self.config["purpose"] = "research"
        with self.assertRaisesRegex(ValidationError, "未完成"):
            ExperimentContract.parse(self.config)

    def test_json_duplicates_and_nan_are_rejected(self) -> None:
        for text in ('{"seed":1,"seed":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.subTest(text=text), self.assertRaises(ValidationError):
                loads(text)
