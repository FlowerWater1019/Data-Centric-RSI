"""第二阶段迁入子包后，缓存与真实恢复入口必须覆盖嵌套源码。"""

from pathlib import Path
import unittest
from unittest.mock import patch

from rsi.configuration.validation import ValidationError
from rsi.storage.execution_artifacts import implementation_identity, implementation_snapshot
from rsi.storage.state import StateStore
from rsi.storage.checkpoints import TrainingPaused
from rsi.workflows.attempt_run import resume_training_attempt, run_training_attempt
from tests import test_training_validation as fixtures


class TrainingPackageLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixtures.TrainingValidationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls) -> None:
        fixtures.TrainingValidationTests.tearDownClass.__func__(cls)

    setUp = fixtures.TrainingValidationTests.setUp
    tearDown = fixtures.TrainingValidationTests.tearDown
    prepare = fixtures.TrainingValidationTests.prepare

    def test_stage2_cache_identity_uses_shared_recursive_manifest(self) -> None:
        source = self.work / "source/nested"
        source.mkdir(parents=True)
        path = source / "module.py"
        path.write_text("version = 1\n", encoding="utf-8")
        with patch("rsi.storage.provenance.PACKAGE_ROOT", source.parent):
            before = implementation_identity()
            self.assertEqual(set(implementation_snapshot()["files"]), {"nested/module.py"})
            path.write_text("version = 2\n", encoding="utf-8")
            self.assertNotEqual(implementation_identity(), before)

    def test_stage2_resume_rejects_nested_change_without_recharging(self) -> None:
        root = self.prepare()
        source = self.work / "source/nested"
        source.mkdir(parents=True)
        path = source / "module.py"
        path.write_text("version = 1\n", encoding="utf-8")
        with patch("rsi.storage.provenance.PACKAGE_ROOT", source.parent):
            with self.assertRaises(TrainingPaused) as paused:
                run_training_attempt(root, "cpu", stop_after_step=1)
            checkpoint = paused.exception.artifact
            path.write_text("version = 2\n", encoding="utf-8")
            with self.assertRaises(ValidationError):
                resume_training_attempt(root, "cpu", checkpoint)
        with StateStore(root / "state.sqlite3", create=False, read_only=True) as state:
            self.assertEqual(state.snapshot("new_run")["budgets"]["training_tokens"]["used"], self.plan.batch.items[0].supervised_tokens)
