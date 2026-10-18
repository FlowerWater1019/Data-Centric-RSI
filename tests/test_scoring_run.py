"""评分调用中断必须保留已知成本和未完成调用意图。"""

from pathlib import Path
import tempfile
from unittest.mock import Mock

from rsi.artifacts import file_sha256
from rsi.scoring import TokenMetrics
from rsi.scoring_run import score_and_record
from rsi.state import StateStore
from tests.helpers import Fixture


class ScoreRunTests(Fixture):
    def test_scoring_failure_keeps_cost_and_incomplete_intent(self) -> None:
        records = tuple(self.context().eligible()[:2])
        runtime = Mock()
        runtime.score.side_effect = [TokenMetrics(records[0].data_id, 4, records[0].token_count, 1.0, 2.0, ()),
                                     RuntimeError("injected scoring interruption")]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with StateStore(root / "state.sqlite3") as state:
                state.initialize("scoring", "controlled", self.config, self.data)
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    score_and_record(runtime, records, state, "baseline", root / "calls")
                snapshot = state.snapshot("scoring")
                self.assertEqual(snapshot["budgets"]["scoring_tokens"]["used"], 4)
                self.assertEqual(snapshot["budgets"]["scoring_tokens"]["reserved"], 0)
                self.assertEqual(snapshot["status"], "execution_error")
            self.assertTrue((root / "calls/baseline/000000.result.json").is_file())
            self.assertTrue((root / "calls/baseline/000001.started.json").is_file())
            self.assertFalse((root / "calls/baseline/000001.result.json").exists())

    def test_completed_scoring_cannot_be_silently_repeated(self) -> None:
        records = tuple(self.context().eligible()[:1])
        runtime = Mock()
        runtime.score.return_value = TokenMetrics(records[0].data_id, 4, records[0].token_count, 1.0, 2.0, ())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with StateStore(root / "state.sqlite3") as state:
                state.initialize("scoring", "controlled", self.config, self.data)
                score_and_record(runtime, records, state, "baseline", root / "calls")
                result_path = root / "calls/baseline/000000.result.json"
                before = file_sha256(result_path)
                with self.assertRaises(FileExistsError):
                    score_and_record(runtime, records, state, "baseline", root / "calls")
                self.assertEqual(runtime.score.call_count, 1)
                self.assertEqual(file_sha256(result_path), before)
                self.assertEqual(state.snapshot("scoring")["budgets"]["scoring_tokens"]["used"], 4)
