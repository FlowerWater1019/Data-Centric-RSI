"""按模块检查可观察行为。"""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile

from rsi.cli import main
from rsi.validation import read_json


from tests.helpers import Fixture, ROOT


class CliTests(Fixture):
    def call(self, arguments: list[str]) -> tuple[int, str, str]:
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            code = main(arguments)
        return code, output.getvalue(), error.getvalue()

    def test_train_is_explicitly_unavailable(self) -> None:
        code, output, error = self.call(["train"])
        self.assertEqual(code, 2)
        self.assertIn("尚未实现", error)
        self.assertEqual(output, "")

    def test_documented_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            db, report = str(Path(folder) / "state.db"), str(Path(folder) / "report.json")
            shared = ["--db", db, "--run-id", "smoke"]
            inputs = ["--config", str(ROOT / "configs/smoke.json"), "--data", str(ROOT / "examples/trajectories.demo.jsonl")]
            commands = [["validate", *inputs, "--require-current-scores"], ["init", *inputs, *shared],
                        ["plan-comparison", *shared, "--comparison-id", "pair1"], ["status", *shared],
                        ["export", *shared, "--output", report]]
            for command in commands:
                code, output, error = self.call(command)
                self.assertEqual(code, 0, error)
                self.assertEqual(json.loads(output)["training_status"], "NOT_RUN")
            self.assertEqual(read_json(report)["status"], "planned")
            self.assertEqual(self.call(["export", *shared, "--output", report])[0], 2)

    def test_missing_database_is_not_created_by_status(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            db = Path(folder) / "absent.db"
            code, _, _ = self.call(["status", "--db", str(db), "--run-id", "missing"])
            self.assertEqual(code, 2)
            self.assertFalse(db.exists())
