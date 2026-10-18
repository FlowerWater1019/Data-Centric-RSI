"""实际训练状态与数据使用账本。测试不调用真实设备。"""

from dataclasses import replace
from pathlib import Path
import tempfile

from rsi.backends import ArtifactRef, TrainingResult, TrainingUse
from rsi.state import StateStore
from rsi.validation import ValidationError
from tests.helpers import Fixture


class TrainingJournalTests(Fixture):
    def setUp(self) -> None:
        super().setUp()
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.state = StateStore(Path(folder.name) / 'state.db')
        self.addCleanup(self.state.close)
        self.state.initialize('run', 'controlled', self.config, self.data)
        plan = self.state.prepare_comparison('run','pair')
        self.fingerprint = plan['arms']['fixed']['fingerprint']
        self.record = self.state.context('run').records[2]
        self.use = TrainingUse('fixed',1,self.record.data_id,self.record.version,self.record.train_mask_version)
        self.state.reserve('run','fixed','training_tokens',8)

    def start(self) -> None:
        self.state.training_event('run','fixed','started',batch_fingerprint=self.fingerprint)

    def test_training_requires_saved_batch(self) -> None:
        with self.assertRaisesRegex(ValidationError, '不属于'):
            self.state.training_event('run','fixed','started',batch_fingerprint='missing')
        self.assertEqual(self.state.snapshot('run')['training_status'],'NOT_RUN')

    def test_usage_requires_started_attempt(self) -> None:
        with self.assertRaises(ValidationError):
            self.state.consume('run','step1','fixed',4,self.use)
        self.assertEqual(self.state.snapshot('run')['budgets']['training_tokens']['used'],0)

    def test_cost_and_usage_are_idempotent_and_preserved_on_failure(self) -> None:
        self.start()
        self.assertTrue(self.state.consume('run','step1','fixed',4,self.use))
        self.assertFalse(self.state.consume('run','step1','fixed',4,self.use))
        with self.assertRaises(ValidationError):
            self.state.consume('run','step1','fixed',4,replace(self.use,step=2))
        self.state.training_event('run','fixed','failed',batch_fingerprint=self.fingerprint,error='injected')
        snapshot = self.state.snapshot('run')
        self.assertEqual(snapshot['training_status'],'FAILED')
        self.assertEqual(snapshot['budgets']['training_tokens']['used'],4)
        self.assertEqual(snapshot['accepted_model'],self.config['model'])

    def test_completion_must_match_ledger(self) -> None:
        self.start()
        self.state.consume('run','step1','fixed',4,self.use)
        result = TrainingResult(self.contract.model,ArtifactRef('demo','demo'),8,None)
        with self.assertRaisesRegex(ValidationError, '账本'):
            self.state.training_event('run','fixed','completed',batch_fingerprint=self.fingerprint,result=result)
        self.assertEqual(self.state.snapshot('run')['training_status'],'RUNNING')

    def test_usage_outside_selected_batch_is_rejected(self) -> None:
        self.start()
        record = self.state.context('run').records[0]
        use = TrainingUse('fixed',1,record.data_id,record.version,record.train_mask_version)
        with self.assertRaisesRegex(ValidationError, '冻结批次'):
            self.state.consume('run','outside','fixed',4,use)
        self.assertEqual(self.state.snapshot('run')['budgets']['training_tokens']['used'],0)

    def test_completed_attempt_requires_entire_frozen_batch(self) -> None:
        self.start()
        self.state.consume('run','step1','fixed',4,self.use)
        result = TrainingResult(self.contract.model,ArtifactRef('demo','demo'),4,None)
        with self.assertRaisesRegex(ValidationError, '计划监督量'):
            self.state.training_event('run','fixed','completed',batch_fingerprint=self.fingerprint,result=result)

    def test_successful_training_keeps_accepted_model_unchanged(self) -> None:
        self.start()
        self.state.consume('run','step1','fixed',4,self.use)
        record = self.state.context('run').records[1]
        self.state.consume('run','step2','fixed',4,TrainingUse('fixed',2,record.data_id,record.version,record.train_mask_version))
        result = TrainingResult(replace(self.contract.model,checkpoint_id='candidate'),ArtifactRef('demo','demo'),8,None)
        self.state.training_event('run','fixed','completed',batch_fingerprint=self.fingerprint,result=result)
        self.assertEqual(self.state.snapshot('run')['training_status'],'COMPLETED')
        self.assertEqual(self.state.snapshot('run')['accepted_model'],self.config['model'])

    def test_equal_token_total_does_not_hide_repeated_wrong_data(self) -> None:
        self.start()
        self.state.consume('run','step1','fixed',4,self.use)
        self.state.consume('run','step2','fixed',4,replace(self.use,step=2))
        result = TrainingResult(self.contract.model,ArtifactRef('demo','demo'),8,None)
        with self.assertRaisesRegex(ValidationError, '实际数据使用'):
            self.state.training_event('run','fixed','completed',batch_fingerprint=self.fingerprint,result=result)
        self.assertEqual(self.state.snapshot('run')['budgets']['training_tokens']['used'],8)

    def test_resume_requires_matching_cost_boundary(self) -> None:
        self.start()
        with self.assertRaises(ValidationError):
            self.state.check_training_resume('run', 'fixed', 'fixed', self.fingerprint, 1)
        self.state.consume('run', 'step1', 'fixed', 4, self.use)
        self.state.check_training_resume('run', 'fixed', 'fixed', self.fingerprint, 1)
        self.state.consume('run', 'step2', 'fixed', 4, replace(self.use, step=2))
        with self.assertRaises(ValidationError):
            self.state.check_training_resume('run', 'fixed', 'fixed', self.fingerprint, 1)
        self.assertEqual(self.state.snapshot('run')['budgets']['training_tokens']['used'], 8)

    def test_closed_reservation_prevents_resume(self) -> None:
        self.start()
        self.state.consume('run', 'step1', 'fixed', 4, self.use)
        self.state.release('run', 'fixed')
        with self.assertRaises(ValidationError):
            self.state.check_training_resume('run', 'fixed', 'fixed', self.fingerprint, 1)

    def test_pause_and_resume_are_visible_without_cost_rollback(self) -> None:
        self.start()
        self.state.consume('run', 'step1', 'fixed', 4, self.use)
        self.state.training_boundary('run', 'fixed', 1, 'paused', 'checkpoint', 'hash')
        self.assertEqual(self.state.snapshot('run')['training_status'], 'PAUSED')
        self.state.training_boundary('run', 'fixed', 1, 'resumed', 'checkpoint', 'hash')
        self.assertEqual(self.state.snapshot('run')['training_status'], 'RUNNING')
        self.assertEqual(self.state.snapshot('run')['budgets']['training_tokens']['used'], 4)
