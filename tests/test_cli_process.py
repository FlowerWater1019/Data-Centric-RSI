"""通过独立进程检查公开命令，避免依赖测试进程内的状态。"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.helpers import ROOT


class CommandProcessTests(unittest.TestCase):
    def test_full_workflow_and_repeated_requests(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            db = Path(folder) / "state.sqlite3"
            report = Path(folder) / "report.json"
            shared = ["--db", str(db), "--run-id", "process-test"]
            inputs = ["--config", "configs/smoke.json", "--data", "examples/trajectories.demo.jsonl"]

            def call(arguments: list[str], expected_code: int = 0) -> dict[str, object]:
                result = subprocess.run(
                    [sys.executable, "-m", "rsi", *arguments],
                    cwd=ROOT, env=dict(os.environ, PYTHONUTF8="1"),
                    capture_output=True, text=True, encoding="utf-8", timeout=30,
                )
                self.assertEqual(result.returncode, expected_code, result.stderr)
                if expected_code:
                    self.assertIn("尚未实现", result.stderr)
                    self.assertEqual(result.stdout, "")
                    return {}
                payload = json.loads(result.stdout)
                self.assertEqual(payload["training_status"], "NOT_RUN")
                return payload

            call(["validate", *inputs, "--require-current-scores"])
            self.assertTrue(call(["init", *inputs, *shared])["created"])
            self.assertFalse(call(["init", *inputs, *shared])["created"])
            plan_args = ["plan-comparison", *shared, "--comparison-id", "pair"]
            self.assertEqual(call(plan_args), call(plan_args))
            state = call(["status", *shared])
            self.assertEqual(state["status"], "planned")
            self.assertEqual(len(state["events"]), 2)
            self.assertEqual(len(state["comparisons"]), 1)
            call(["export", *shared, "--output", str(report)])
            self.assertEqual(json.loads(report.read_text(encoding="utf-8")), state)
            call(["train"], expected_code=2)
