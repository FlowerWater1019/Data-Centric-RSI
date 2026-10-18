"""真实对照的输入检查不加载模型。测试只使用合成编译产物。"""

from pathlib import Path
from typing import Any
import tempfile
from unittest.mock import patch

from rsi.artifacts import file_sha256, write_json
from rsi.dataset_import import write_jsonl
from rsi.real_run import preflight_comparison
from rsi.validation import ValidationError, read_json
from tests.helpers import Fixture, ROOT


class RealRunInputTests(Fixture):
    def setUp(self) -> None:
        super().setUp()
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.bundle = self.root / "compiled"
        self.bundle.mkdir()
        self.config["purpose"] = "draft"
        self.config["model"]["adapter_revision"] = "none"
        self.data[4]["role"] = "reference"
        self.rows = [r for r in self.data if r["role"] in {"candidate", "reference", "feedback_diagnostic"}]
        for row in self.rows:
            row["scores"] = None
            row["tokenization"]["status"] = "verified"
        self.spec = read_json(ROOT / "configs/comparison.development.json")
        self.spec["selection"]["supervised_token_budget"] = 8

    def freeze(self) -> None:
        write_json(self.bundle / "contract.json", self.config)
        write_jsonl(self.bundle / "trajectories.jsonl", self.rows)
        write_json(self.bundle / "model_identity.json", {"content_id": "fixture"})
        self.ref = write_json(self.bundle / "report.json", {"schema_version": "compiled-data-1", "data_kind": "real_public",
            "files": [{"name": p.name, "sha256": file_sha256(p)} for p in self.bundle.iterdir()]})
        write_json(self.root / "config.json", self.spec)

    def check(self) -> dict[str, Any]:
        return preflight_comparison(self.bundle, self.ref.sha256, self.root / "config.json")

    def test_valid_inputs_require_no_model_call(self) -> None:
        self.freeze()
        with patch("rsi.real_run.load_dependencies") as dependencies:
            result = self.check()
            dependencies.assert_not_called()
        self.assertEqual(result["per_arm_tokens"], 8)
        self.assertEqual(result["training"], "NOT_RUN")

    def test_tampered_inputs_rejected(self) -> None:
        self.freeze()
        (self.bundle / "trajectories.jsonl").write_text("[]")
        with self.assertRaisesRegex(ValidationError, "校验值"):
            self.check()

    def test_final_test_role_is_rejected_before_model_call(self) -> None:
        self.rows[-1]["role"] = "final_test"
        self.freeze()
        with self.assertRaisesRegex(ValidationError, "只接收"):
            self.check()

    def test_insufficient_score_budget_is_rejected(self) -> None:
        self.spec["scoring_token_budget"] = 1
        self.freeze()
        with self.assertRaisesRegex(ValidationError, "评分"):
            self.check()

    def test_unreachable_budget_is_rejected(self) -> None:
        self.spec["selection"]["supervised_token_budget"] = 7
        self.freeze()
        with self.assertRaisesRegex(ValidationError, "无法达到"):
            self.check()

    def test_overlength_is_rejected_without_truncation(self) -> None:
        self.spec["training"]["max_sequence_length"] = 1
        self.freeze()
        with self.assertRaisesRegex(ValidationError, "超长"):
            self.check()
