"""按模块检查可观察行为。"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import tempfile
import threading

from rsi.state import StateStore
from rsi.validation import ValidationError


from tests.helpers import Fixture


class StateTests(Fixture):
    def setUp(self) -> None:
        super().setUp()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.db = Path(self.directory.name) / "state.sqlite3"
        self.state = StateStore(self.db)
        self.addCleanup(self.state.close)
        self.state.initialize("run", "controlled", self.config, self.data)

    def test_initialization_is_idempotent(self) -> None:
        self.assertFalse(self.state.initialize("run", "controlled", self.config, self.data))
        self.assertEqual(len(self.state.snapshot("run")["events"]), 1)

    def test_same_run_cannot_change_inputs_or_mode(self) -> None:
        config = deepcopy(self.config)
        config["seed"] += 1
        for mode, value in (("controlled", config), ("online", self.config)):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValidationError, "输入不同"):
                self.state.initialize("run", mode, value, self.data)

    def test_plans_survive_restart_without_training(self) -> None:
        first = self.state.prepare_comparison("run", "pair1")
        with StateStore(self.db) as reopened:
            self.assertEqual(first, reopened.prepare_comparison("run", "pair1"))
            snapshot = reopened.snapshot("run")
        self.assertEqual(snapshot["status"], "planned")
        self.assertEqual(snapshot["training_status"], "NOT_RUN")
        self.assertIsNone(first["metrics"])
        self.assertEqual(first["arms"]["random"]["context_id"], first["arms"]["fixed"]["context_id"])
        self.assertEqual(len(snapshot["events"]), 2)
        self.assertEqual(snapshot["budgets"]["training_tokens"]["used"], 0)

    def test_snapshot_is_independent_of_external_mutations(self) -> None:
        self.config["seed"] = 1000
        self.data.clear()
        self.assertEqual(self.state.context("run").contract.seed, 17)
        self.assertEqual(len(self.state.context("run").records), 7)

    def test_online_run_cannot_write_controlled_plans(self) -> None:
        self.state.initialize("online", "online", self.config, self.data)
        with self.assertRaisesRegex(ValidationError, "controlled"):
            self.state.prepare_comparison("online", "pair1")

    def test_failed_planning_leaves_no_partial_plan(self) -> None:
        data = deepcopy(self.data)
        data[0]["scores"]["checkpoint_id"] = "stale"
        self.state.initialize("bad", "controlled", self.config, data)
        with self.assertRaises(ValidationError):
            self.state.prepare_comparison("bad", "pair1")
        snapshot = self.state.snapshot("bad")
        self.assertEqual(snapshot["status"], "validated")
        self.assertEqual(snapshot["comparisons"], [])

    def test_transaction_rolls_back_all_events(self) -> None:
        with self.assertRaises(RuntimeError):
            with self.state.transaction():
                self.state._event("run", "partial", "test", {})
                raise RuntimeError("injected failure")
        self.assertEqual(len(self.state.snapshot("run")["events"]), 1)

    def test_budget_is_idempotent_and_release_keeps_used_cost(self) -> None:
        self.assertTrue(self.state.reserve("run", "job", "scoring_tokens", 10))
        self.assertFalse(self.state.reserve("run", "job", "scoring_tokens", 10))
        self.assertTrue(self.state.consume("run", "chunk1", "job", 4))
        self.assertFalse(self.state.consume("run", "chunk1", "job", 4))
        self.state.release("run", "job")
        self.assertFalse(self.state.release("run", "job"))
        cost = self.state.snapshot("run")["budgets"]["scoring_tokens"]
        self.assertEqual(cost, {"limit": 1000, "used": 4, "reserved": 0, "available": 996})

    def test_conflicting_cost_id_fails_without_changing_cost(self) -> None:
        self.state.reserve("run", "job", "scoring_tokens", 10)
        self.state.consume("run", "chunk", "job", 4)
        with self.assertRaises(ValidationError):
            self.state.consume("run", "chunk", "job", 5)
        self.assertEqual(self.state.snapshot("run")["budgets"]["scoring_tokens"]["used"], 4)

    def test_actual_overrun_is_preserved(self) -> None:
        self.state.reserve("run", "job", "scoring_tokens", 2)
        self.state.consume("run", "chunk", "job", 4)
        snapshot = self.state.snapshot("run")
        self.assertEqual(snapshot["status"], "execution_error")
        self.assertEqual(snapshot["budgets"]["scoring_tokens"]["used"], 4)
        with self.assertRaises(ValidationError):
            self.state.reserve("run", "next", "scoring_tokens", 1)

    def test_failed_attempt_does_not_erase_cost_or_model(self) -> None:
        self.state.reserve("run", "job", "scoring_tokens", 10)
        self.state.consume("run", "chunk", "job", 4)
        self.state.record_error("run", "failed", "演示执行错误")
        self.state.release("run", "job")
        snapshot = self.state.snapshot("run")
        self.assertEqual(snapshot["accepted_model"], self.config["model"])
        self.assertEqual(snapshot["budgets"]["scoring_tokens"]["used"], 4)

    def test_two_connections_cannot_overreserve(self) -> None:
        barrier = threading.Barrier(2)

        def reserve(identity: str) -> bool:
            with StateStore(self.db) as store:
                barrier.wait(timeout=10)
                try:
                    store.reserve("run", identity, "scoring_tokens", 600)
                    return True
                except ValidationError:
                    return False

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(reserve, ["job1", "job2"]))
        self.assertEqual(sorted(results), [False, True])
        self.assertEqual(self.state.snapshot("run")["budgets"]["scoring_tokens"]["reserved"], 600)
