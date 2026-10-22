"""第二阶段故障注入与异常输入回归。共用合成输入，不依赖 GPU。"""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from rsi.storage.artifacts import write_json
from rsi.workflows.attempt_run import run_training_attempt
from rsi.storage.execution_artifacts import complete_execution, verify_execution
from rsi.evaluation.execution_evaluation import evaluate_execution, parse_execution_evaluation, verify_isolation
from rsi.evaluation.fit_evaluation import evaluate_fit, parse_fit_evaluation
from rsi.evaluation.task_adapters import TaskSuiteV2
from rsi.models.training_mock import MockTokenScorer, MockTrainingBackend
from rsi.configuration.validation import ValidationError, canonical, digest, read_json
from rsi.workflows.validation_run import evaluate_fit_stage, evaluate_execution_stage, run_dual_validation
from tests import test_training_validation as fixtures


class TrainingHardeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixtures.TrainingValidationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls) -> None:
        fixtures.TrainingValidationTests.tearDownClass.__func__(cls)

    setUp = fixtures.TrainingValidationTests.setUp
    tearDown = fixtures.TrainingValidationTests.tearDown
    prepare = fixtures.TrainingValidationTests.prepare
    fit = fixtures.TrainingValidationTests.fit
    suite = fixtures.TrainingValidationTests.suite

    def test_manifest_cannot_omit_changed_file(self) -> None:
        root = self.work / "stage"
        write_json(root / "a.json", {"value": 1})
        write_json(root / "b.json", {"value": 2})
        complete_execution(root, "COMPLETED")
        manifest = read_json(root / "completion_manifest.json")
        manifest["files"] = manifest["files"][:1]
        manifest["content_id"] = digest(manifest["files"])
        (root / "completion_manifest.json").write_text(canonical(manifest), encoding="utf-8")
        (root / "b.json").write_text("changed", encoding="utf-8")
        with self.assertRaises(ValidationError):
            verify_execution(root)

    def test_manifest_rejects_invalid_status(self) -> None:
        root = self.work / "stage"
        write_json(root / "a.json", {"value": 1})
        complete_execution(root, "COMPLETED")
        manifest = read_json(root / "completion_manifest.json")
        manifest["status"] = "NOT_RUN"
        (root / "completion_manifest.json").write_text(canonical(manifest), encoding="utf-8")
        with self.assertRaises(ValidationError):
            verify_execution(root)

    def test_candidate_must_keep_parent_base_and_tokenizer(self) -> None:
        original = MockTrainingBackend.train
        for field in ("base_revision", "tokenizer_revision", "template_revision"):
            with self.subTest(field=field):
                output = self.prepare()
                def wrong_model(backend, request):
                    result = original(backend, request)
                    return replace(result, candidate_model=replace(result.candidate_model, **{field: "wrong"}))
                with patch.object(MockTrainingBackend, "train", wrong_model), self.assertRaises(ValidationError):
                    run_training_attempt(output, "cpu")
                # 下一次子案例使用独立输出，保留失败现场。
                output.rename(self.work / field)

    def test_candidate_adapter_revision_matches_artifact(self) -> None:
        output = self.prepare()
        original = MockTrainingBackend.train
        def wrong_adapter(backend, request):
            result = original(backend, request)
            return replace(result, candidate_model=replace(result.candidate_model, adapter_revision="wrong"))
        with patch.object(MockTrainingBackend, "train", wrong_adapter), self.assertRaises(ValidationError):
            run_training_attempt(output, "cpu")

    def test_duplicate_skills_rejected_before_scoring(self) -> None:
        with self.assertRaises(ValidationError):
            evaluate_fit(self.location.model_ref, self.selected, MockTokenScorer(), (self.ontology.ids[0],) * 2)

    def test_fit_rejects_unknown_scoring_rule(self) -> None:
        raw = self.fit().to_dict()
        raw["scoring_rule_version"] = "unimplemented_rule"
        with self.assertRaises(ValidationError):
            parse_fit_evaluation(raw)

    def test_task_ids_cannot_alias_other_problem_ids(self) -> None:
        raw = self.suite().to_dict()
        raw["tasks"][1]["source_problem_id"] = raw["tasks"][0]["task_id"]
        with self.assertRaises(ValidationError):
            TaskSuiteV2.parse(raw, self.ontology)

    def test_isolation_compares_split_and_duplicate_namespaces(self) -> None:
        suite = self.suite()
        task = replace(suite.tasks[0], split_group=self.selected[0].duplicate_group)
        with self.assertRaises(ValidationError):
            verify_isolation(replace(suite, tasks=(task, *suite.tasks[1:])), self.selected)

    def test_execution_parser_rejects_impossible_pass(self) -> None:
        result = evaluate_execution(self.location.model_ref, self.suite(), lambda p, n, s: "def solve(x,y): return x+y",
                                    self.work / "execution", 2, source="synthetic_fixture")
        for field, value in (("generated", None), ("test_results_json", "[]"), ("group", "final_test")):
            raw = result.to_dict()
            raw["rows"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                parse_execution_evaluation(raw)

    def test_non_text_generation_has_parseable_partial_report(self) -> None:
        result = evaluate_execution(self.location.model_ref, self.suite(), lambda p, n, s: 123,
                                    self.work / "execution", 2, source="synthetic_fixture")
        self.assertEqual(result.status, "PARTIAL")
        self.assertEqual(parse_execution_evaluation(result.to_dict()), result)

    def test_keyboard_interrupt_keeps_intent_and_no_completion(self) -> None:
        def interrupted(p, n, s):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            evaluate_execution(self.location.model_ref, self.suite(), interrupted, self.work / "execution", 2, source="synthetic_fixture")
        self.assertTrue((self.work / "execution/000000.intent.json").exists())
        self.assertFalse((self.work / "execution/completion_manifest.json").exists())

    def test_publication_failure_can_finish_without_repeating_measurements(self) -> None:
        root = self.prepare()
        run_training_attempt(root, "cpu")
        def fail_final_manifest(path, status, files=None):
            if path == root:
                raise OSError("injected final publication failure")
            return complete_execution(path, status, files)
        with patch("rsi.workflows.validation_run.complete_execution", side_effect=fail_final_manifest), self.assertRaises(OSError):
            run_dual_validation(root, self.demo / "source/feedback_tasks.json", device="cpu", call_budget=2)
        self.assertTrue((root / "validation/round_feedback.json").exists())
        before = read_json(root / "validation/round_feedback.json")["costs"]
        result = run_dual_validation(root, self.demo / "source/feedback_tasks.json", device="cpu", call_budget=2)
        self.assertEqual(result.status, "COMPLETED")
        self.assertTrue((root / "completion_manifest.json").exists())
        self.assertEqual(result.to_dict()["costs"], before)

    def test_invalid_budgets_rejected_before_generation(self) -> None:
        for index, budget in enumerate((True, False, -1, 1.5, None, float("nan"))):
            generate = unittest.mock.Mock(return_value="def solve(x,y): return x+y")
            output = self.work / str(index)
            with self.subTest(budget=budget), self.assertRaises(ValidationError):
                evaluate_execution(self.location.model_ref, self.suite(), generate, output, budget, source="synthetic_fixture")
            generate.assert_not_called()
            self.assertFalse(output.exists())

    def test_cache_cannot_accept_boolean_as_integer_budget(self) -> None:
        root = self.prepare()
        stage = self.work / "execution"
        evaluate_execution_stage(root, self.demo / "source/feedback_tasks.json", stage, candidate=False, device="cpu", call_budget=1)
        with self.assertRaises(ValidationError):
            evaluate_execution_stage(root, self.demo / "source/feedback_tasks.json", stage, candidate=False, device="cpu", call_budget=True)

    def test_fit_cache_recomputes_skill_summary_from_saved_token_facts(self) -> None:
        root = self.prepare()
        stage = self.work / "fit"
        evaluate_fit_stage(root, stage, candidate=False, device="cpu")
        raw = read_json(stage / "fit.json")
        # 将全部技能伪装为缺失，同时保持总体与逐条 token 事实不变。
        for row in raw["records"]:
            for skill in row["by_skill"]:
                skill["stats"] = {"status": "MISSING", "token_count": 0, "ce_sum": 0.0, "entropy_sum": 0.0, "nll": None, "entropy": None}
        for skill in raw["by_skill"]:
            skill["stats"] = {"status": "MISSING", "token_count": 0, "ce_sum": 0.0, "entropy_sum": 0.0, "nll": None, "entropy": None}
        (stage / "fit.json").write_text(canonical(raw), encoding="utf-8")
        # 重建合法文件清单，确认拒绝来自业务核验，不只依赖文件哈希。
        (stage / "completion_manifest.json").unlink()
        complete_execution(stage, "COMPLETED")
        with self.assertRaisesRegex(ValidationError, "fit.cache"):
            evaluate_fit_stage(root, stage, candidate=False, device="cpu")

    def test_known_scoring_failure_keeps_cost_and_blocks_unknown_replay(self) -> None:
        root = self.prepare()
        count = 0
        original = MockTokenScorer.score_tokens
        def fail_second(scorer, record):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError("injected scoring OOM")
            return original(scorer, record)
        with patch.object(MockTokenScorer, "score_tokens", fail_second), self.assertRaises(RuntimeError):
            evaluate_fit_stage(root, self.work / "fit", candidate=False, device="cpu")
        from rsi.storage.state import StateStore
        with StateStore(root / "state.sqlite3", create=False) as state:
            budget = state.snapshot("new_run")["budgets"]["scoring_tokens"]
            compilation = read_json(self.demo / "compiled/costs.json")["budgets"]["scoring_tokens"]["used"]
            self.assertEqual(budget["used"], compilation + self.selected[0].token_count)
            self.assertGreater(budget["reserved"], 0)
        with self.assertRaises(OSError):
            evaluate_fit_stage(root, self.work / "fit", candidate=False, device="cpu")
        self.assertTrue((self.work / "fit/failure.json").exists())

    def test_checkpoint_publish_retries_only_transient_rename(self) -> None:
        from rsi.storage.checkpoints import StepJournal
        journal = StepJournal(self.work, "identity")
        journal.begin(1)
        real_rename = Path.rename
        attempts = []
        def transient(path, destination):
            attempts.append(path)
            if len(attempts) == 1:
                error = PermissionError("transient Windows directory lock")
                error.winerror = 5
                raise error
            return real_rename(path, destination)
        save = unittest.mock.Mock(side_effect=lambda p: p.write_bytes(b"complete state"))
        with patch.object(Path, "rename", transient), patch("rsi.storage.checkpoints.time.sleep"):
            artifact = journal.publish(1, save)
        self.assertEqual(len(attempts), 2)
        save.assert_called_once()
        self.assertEqual(journal.validate_resume(artifact)[1], 1)

    def test_checkpoint_permanent_failure_preserves_pending_state(self) -> None:
        from rsi.storage.checkpoints import StepJournal
        journal = StepJournal(self.work, "identity")
        journal.begin(1)
        error = PermissionError("permanent Windows directory lock")
        error.winerror = 5
        with patch.object(Path, "rename", side_effect=error) as rename, patch("rsi.storage.checkpoints.time.sleep"), self.assertRaises(PermissionError):
            journal.publish(1, lambda p: p.write_bytes(b"complete state"))
        self.assertEqual(rename.call_count, 4)
        self.assertTrue((self.work / "checkpoints/.000001.pending/state.pt").exists())
        self.assertFalse((self.work / "checkpoints/000001").exists())

    def test_mock_compilation_cannot_feed_real_training(self) -> None:
        from rsi.workflows.attempt_run import prepare_training_attempt
        settings = read_json(self.demo / "source/settings.json")
        settings["mode"] = "hf_local"
        write_json(self.work / "real-settings.json", settings)
        with patch("rsi.workflows.attempt_run.load_tokenizer", side_effect=AssertionError("must reject before loading")), self.assertRaisesRegex(ValidationError, "compiled.provenance.execution_mode"):
            prepare_training_attempt(self.demo / "compiled", self.demo / "selection", self.plan.selection_manifest_id,
                self.demo / "source/model_location.json", self.work / "real-settings.json", self.work / "bad", "r", "a")
        self.assertFalse((self.work / "bad").exists())

    def test_provenance_rejects_mixed_sources_and_unknown_schema(self) -> None:
        from rsi.datasets.training_inputs import TrainingProvenance
        raw = read_json(self.demo / "compiled/provenance.json")
        for key, value in (("source", "measured"), ("tokenizer_execution_mode", "hf_local"), ("schema_version", "unknown"), ("sources_hash", None)):
            with self.subTest(key=key), self.assertRaises(ValidationError):
                TrainingProvenance.parse({**raw, key: value})

    def test_zero_supervision_is_missing_without_forward_pass(self) -> None:
        from rsi.datasets.data import TrajectoryRecord
        raw = self.selected[0].raw
        raw["supervision"]["loss_mask"] = [False] * self.selected[0].token_count
        raw["scores"] = None
        record = TrajectoryRecord.parse(raw, self.inputs.contract, "zero")
        scorer = unittest.mock.Mock()
        result = evaluate_fit(self.location.model_ref, (record,), scorer, self.ontology.ids, source="synthetic_fixture")
        scorer.score_tokens.assert_not_called()
        self.assertEqual(result.overall.status, "MISSING")
        self.assertIsNone(result.overall.nll)
        self.assertEqual(parse_fit_evaluation(result.to_dict()), result)

    def test_finite_token_values_cannot_overflow_aggregate(self) -> None:
        from rsi.evaluation.fit_evaluation import aggregate
        from rsi.models.scoring import TokenScore
        with self.assertRaises(ValidationError):
            aggregate((TokenScore(1, 1e308, 1), TokenScore(2, 1e308, 1)))

    def test_malformed_artifact_reports_field_error(self) -> None:
        from rsi.storage.artifacts import verify_training_artifact
        for index, value in enumerate((None, [], {}, {"files": None}, {"files": []}, {"files": [None]}, {"files": [{"name": "x"}]})):
            reference = write_json(self.work / (str(index) + ".json"), value)
            with self.subTest(value=value), self.assertRaises(ValidationError):
                verify_training_artifact(reference)
