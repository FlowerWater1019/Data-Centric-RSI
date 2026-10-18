"""检查点边界与写入者锁。测试不需要 torch。"""

from pathlib import Path
import tempfile
import unittest

from rsi.checkpoints import StepJournal
from rsi.leases import run_lease
from rsi.validation import ValidationError
from tests.helpers import ROOT


class CheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.journal = StepJournal(self.root, 'identity')

    def publish(self, step: int = 1):
        self.journal.begin(step)
        return self.journal.publish(step, lambda path: path.write_bytes(b'test-state'))

    def test_complete_boundary_can_be_reopened(self) -> None:
        artifact = self.publish()
        path, step = self.journal.validate_resume(artifact)
        self.assertEqual((path.read_bytes(), step), (b'test-state', 1))

    def test_inflight_step_prevents_replay(self) -> None:
        artifact = self.publish()
        self.journal.begin(2)
        with self.assertRaisesRegex(ValidationError, '成本可能未知'):
            self.journal.validate_resume(artifact)

    def test_old_checkpoint_cannot_roll_back_cost(self) -> None:
        old = self.publish()
        self.publish(2)
        with self.assertRaises(ValidationError):
            self.journal.validate_resume(old)

    def test_changed_identity_and_bytes_are_rejected(self) -> None:
        artifact = self.publish()
        with self.assertRaises(ValidationError):
            StepJournal(self.root, 'changed').validate_resume(artifact)
        (Path(artifact.path).parent / 'state.pt').write_bytes(b'changed')
        with self.assertRaises(ValidationError):
            self.journal.validate_resume(artifact)

    def test_unpublished_directory_prevents_resume(self) -> None:
        artifact = self.publish()
        (self.root / 'checkpoints/.000002.pending').mkdir()
        with self.assertRaises(ValidationError):
            self.journal.validate_resume(artifact)

    def test_completed_output_cannot_be_overwritten(self) -> None:
        artifact = self.publish()
        (self.root / 'training.json').write_text('{}')
        with self.assertRaises(ValidationError):
            self.journal.validate_resume(artifact)

    def test_writer_lock_releases_after_exception(self) -> None:
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with run_lease(self.root):
                with self.assertRaises(ValidationError):
                    with run_lease(self.root):
                        self.fail('second writer entered')
                raise RuntimeError('injected')
        with run_lease(self.root):
            pass
