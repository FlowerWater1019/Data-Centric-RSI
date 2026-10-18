"""按模块检查可观察行为。"""

from copy import deepcopy

from rsi.contracts import ExperimentContract
from rsi.data import validate_dataset
from rsi.validation import ValidationError


from tests.helpers import Fixture


class DataTests(Fixture):
    def test_valid_masks_exclude_failed_actions(self) -> None:
        record = validate_dataset(self.data, self.contract)[0]
        self.assertEqual(record.supervised_tokens, 4)
        self.assertTrue(any(x == "incorrect" for x in record.raw["tokenization"]["correctness"]))

    def test_error_actions_cannot_be_supervised(self) -> None:
        self.data[0]["supervision"]["loss_mask"][7] = True
        with self.assertRaisesRegex(ValidationError, "正向监督"):
            validate_dataset(self.data, self.contract)

    def test_tool_tokens_cannot_be_supervised(self) -> None:
        self.data[0]["tokenization"]["roles"][-1] = "tool"
        with self.assertRaisesRegex(ValidationError, "正向监督"):
            validate_dataset(self.data, self.contract)

    def test_unknown_actions_cannot_be_supervised(self) -> None:
        self.data[0]["tokenization"]["correctness"][-1] = "unknown"
        with self.assertRaisesRegex(ValidationError, "正向监督"):
            validate_dataset(self.data, self.contract)

    def test_mask_length_is_validated(self) -> None:
        self.data[0]["supervision"]["loss_mask"].pop()
        with self.assertRaisesRegex(ValidationError, "数量"):
            validate_dataset(self.data, self.contract)

    def test_first_token_cannot_inflate_supervised_budget(self) -> None:
        self.data[0]["supervision"]["loss_mask"][0] = True
        self.data[0]["tokenization"]["roles"][0] = "assistant"
        self.data[0]["tokenization"]["correctness"][0] = "valid"
        with self.assertRaisesRegex(ValidationError, "首个 token"):
            validate_dataset(self.data, self.contract)

    def test_span_offsets_are_validated(self) -> None:
        self.data[0]["diagnostic_spans"][0]["char_start"] = 0
        with self.assertRaisesRegex(ValidationError, "位置"):
            validate_dataset(self.data, self.contract)

    def test_text_hash_is_validated(self) -> None:
        self.data[0]["text"] += "changed"
        with self.assertRaisesRegex(ValidationError, "SHA256"):
            validate_dataset(self.data, self.contract)

    def test_split_leakage_is_rejected(self) -> None:
        self.data[-1]["split_group"] = self.data[0]["split_group"]
        with self.assertRaisesRegex(ValidationError, "泄漏"):
            validate_dataset(self.data, self.contract)

    def test_hidden_content_duplicate_is_rejected(self) -> None:
        duplicate = deepcopy(self.data[0])
        duplicate.update(data_id="hidden", role="final_test", split_group="different", duplicate_group="different")
        self.data.append(duplicate)
        with self.assertRaisesRegex(ValidationError, "content_hash"):
            validate_dataset(self.data, self.contract)

    def test_feedback_overlap_requires_contract_permission(self) -> None:
        duplicate = deepcopy(self.data[-2])
        duplicate.update(data_id="feedback_exec", role="feedback_execution")
        self.data.append(duplicate)
        with self.assertRaises(ValidationError):
            validate_dataset(self.data, self.contract)
        self.config["allow_feedback_overlap"] = True
        validate_dataset(self.data, ExperimentContract.parse(self.config))

    def test_probe_reference_overlap_is_allowed(self) -> None:
        duplicate = deepcopy(self.data[4])
        duplicate.update(data_id="reference", role="reference")
        self.data.append(duplicate)
        validate_dataset(self.data, self.contract)

    def test_usage_history_must_be_consistent(self) -> None:
        self.data[0]["usage"]["in_accepted_lineage"] = True
        with self.assertRaisesRegex(ValidationError, "矛盾"):
            validate_dataset(self.data, self.contract)

    def test_demo_tokenization_cannot_enter_draft_run(self) -> None:
        self.config["purpose"] = "draft"
        with self.assertRaisesRegex(ValidationError, "演示分词"):
            validate_dataset(self.data, ExperimentContract.parse(self.config))
