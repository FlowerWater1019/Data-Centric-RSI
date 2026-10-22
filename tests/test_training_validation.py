"""第二阶段公开行为、隔离、恢复和失败回归测试。不要求模型依赖。"""

import copy
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import importlib.util
import math
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from rsi.storage.artifacts import file_sha256, write_json
from rsi.workflows.attempt_run import load_attempt, prepare_training_attempt, run_training_attempt, resume_training_attempt
from rsi.models.backends import ArtifactRef, TrainingUse, TrainingResult
from rsi.storage.checkpoints import TrainingPaused
from rsi.configuration.contracts import ExperimentContract
from rsi.datasets.data import TrajectoryRecord
from rsi.storage.execution_artifacts import verify_execution
from rsi.evaluation.execution_evaluation import evaluate_execution, parse_execution_evaluation, verify_isolation
from rsi.workflows.execution_plan import freeze_training_plan
from rsi.evaluation.fit_evaluation import SkillEvidence, evaluate_fit, compare_fit, parse_fit_evaluation
from rsi.datasets.labels import SkillOntology
from rsi.evaluation.round_feedback import build_round_feedback, parse_round_feedback
from rsi.models.scoring import TokenScore
from rsi.storage.state import StateStore
from rsi.evaluation.task_adapters import TaskSuiteV2, verify_suite_isolation
from rsi.workflows.training_compile import MockChatTokenizer
from rsi.workflows.training_demo import run_demo, example_values, example_suite
from rsi.datasets.training_inputs import bind_message_evidence, compile_training_sources
from rsi.models.training_mock import MockTokenScorer
from rsi.datasets.content_identity import task_fingerprint
from rsi.configuration.validation import ValidationError, canonical, digest, read_json, to_json_value
from rsi.workflows.validation_run import evaluate_fit_stage, evaluate_execution_stage, run_dual_validation


class TrainingValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = tempfile.TemporaryDirectory()
        cls.root = Path(cls.workspace.name)
        cls.demo = cls.root / "demo"
        cls.report = run_demo(cls.demo)
        cls.plan, cls.inputs, cls.location = load_attempt(cls.demo / "attempt")
        cls.ontology = cls.inputs.ontology
        cls.selected = tuple(next(r for r in cls.inputs.records if r.data_id == i.data_id) for i in cls.plan.batch.items)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.workspace.cleanup()

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=self.root)
        self.work = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def prepare(self) -> Path:
        root = self.work / "attempt"
        prepare_training_attempt(self.demo / "compiled", self.demo / "selection", self.plan.selection_manifest_id,
             self.demo / "source/model_location.json", self.demo / "source/settings.json", root, "new_run", "new_attempt")
        return root

    def fit(self, *, candidate: bool = False, evidence: tuple[SkillEvidence, ...] = ()):
        return evaluate_fit(self.location.model_ref, self.selected, MockTokenScorer(candidate), self.ontology.ids, evidence,
                            source="synthetic_fixture")

    def suite(self, final: bool = False) -> TaskSuiteV2:
        return TaskSuiteV2.parse(example_suite(self.ontology.to_dict(), final_test=final), self.ontology)

    def test_complete_demo_keeps_fixture_and_no_promotion(self) -> None:
        self.assertEqual(self.report["status"], "COMPLETED")
        self.assertEqual(self.report["source"], "synthetic_fixture")
        self.assertEqual(self.report["training_tokens"], self.plan.batch.supervised_tokens)
        self.assertEqual(self.report["promotion_status"], "NOT_APPLIED")
        self.assertEqual(self.report["controller_status"], "NOT_RUN")
        verify_execution(self.demo / "attempt")
        feedback = parse_round_feedback(read_json(self.demo / "attempt/validation/round_feedback.json"))
        self.assertEqual(feedback.acceptance.status, "NOT_RUN")

    def test_old_selection_manifest_remains_not_run(self) -> None:
        raw = read_json(self.demo / "selection/manifest.json")
        self.assertEqual(raw["training_status"], "NOT_RUN")

    def test_prepare_is_idempotent_and_has_no_comparison(self) -> None:
        root = self.prepare()
        before = file_sha256(root / "plan.json")
        prepare_training_attempt(self.demo / "compiled", self.demo / "selection", self.plan.selection_manifest_id,
             self.demo / "source/model_location.json", self.demo / "source/settings.json", root, "new_run", "new_attempt")
        with StateStore(root / "state.sqlite3", create=False) as state:
            self.assertEqual(state.connection.execute("SELECT COUNT(*) FROM comparisons").fetchone()[0], 0)
            self.assertEqual(sum(e["kind"] == "single_training_plan" for e in state.snapshot("new_run")["events"]), 1)
        self.assertEqual(file_sha256(root / "plan.json"), before)

    def test_training_does_not_call_selector_and_reentry_does_not_charge(self) -> None:
        root = self.prepare()
        with patch("rsi.curation.selection.build_batch", side_effect=AssertionError("must not select")), patch("rsi.curation.policy_selection.select_with_policy", side_effect=AssertionError("must not rank")):
            first = run_training_attempt(root, "cpu")
            second = run_training_attempt(root, "cpu")
        self.assertEqual(first, second)
        with StateStore(root / "state.sqlite3", create=False) as state:
            self.assertEqual(state.snapshot("new_run")["budgets"]["training_tokens"]["used"], self.plan.batch.supervised_tokens)

    def test_pause_resume_charges_once_and_finishes(self) -> None:
        root = self.prepare()
        with self.assertRaises(TrainingPaused) as caught:
            run_training_attempt(root, "cpu", stop_after_step=1)
        self.assertFalse((root / "completion_manifest.json").exists())
        with self.assertRaises(ValidationError):
            run_training_attempt(root, "cpu")
        result = resume_training_attempt(root, "cpu", caught.exception.artifact)
        self.assertEqual(result.actual_supervised_tokens, self.plan.batch.supervised_tokens)
        with StateStore(root / "state.sqlite3", create=False) as state:
            self.assertEqual(state.snapshot("new_run")["budgets"]["training_tokens"]["used"], result.actual_supervised_tokens)
        with self.assertRaises(ValidationError):
            resume_training_attempt(root, "cpu", caught.exception.artifact)

    def test_unknown_step_blocks_resume_and_retains_cost(self) -> None:
        root = self.prepare()
        with self.assertRaises(TrainingPaused) as caught:
            run_training_attempt(root, "cpu", stop_after_step=1)
        checkpoint = caught.exception.artifact
        write_json(root / "training/intents/000002.json", {"unknown": True})
        with self.assertRaises(ValidationError):
            resume_training_attempt(root, "cpu", checkpoint)
        with StateStore(root / "state.sqlite3", create=False) as state:
            self.assertEqual(state.snapshot("new_run")["budgets"]["training_tokens"]["used"], self.plan.batch.items[0].supervised_tokens)

    def test_backend_failure_preserves_actual_cost_and_failure(self) -> None:
        root = self.prepare()
        from rsi.models.training_mock import MockTrainingBackend
        original = MockTrainingBackend.train
        def fail_after_step(backend, request):
            backend.stop = 1
            try:
                return original(backend, request)
            except TrainingPaused:
                raise RuntimeError("injected persistence failure")
        with patch.object(MockTrainingBackend, "train", fail_after_step), self.assertRaises(RuntimeError):
            run_training_attempt(root, "cpu")
        self.assertTrue(list((root / "failures").glob("*.json")))
        with StateStore(root / "state.sqlite3", create=False) as state:
            snapshot = state.snapshot("new_run")
            self.assertEqual(snapshot["budgets"]["training_tokens"]["used"], self.plan.batch.items[0].supervised_tokens)
        with self.assertRaises(ValidationError):
            run_training_attempt(root, "cpu")

    def test_input_tamper_rejected(self) -> None:
        for relative in ("plan.json", "inputs/bundle/labels.jsonl", "inputs/bundle/trajectories.jsonl", "inputs/selection/batch.json", "inputs/bundle/skill_evidence.json"):
            with self.subTest(relative=relative):
                target = self.work / digest(relative)
                shutil.copytree(self.demo / "attempt", target)
                with (target / relative).open("a", encoding="utf-8") as stream:
                    stream.write(" ")
                # plan JSON 仅多一个空白不会改变语义；其余文件清单必须拒绝。
                if relative == "plan.json":
                    raw = read_json(target / relative)
                    raw["batch"]["items"].reverse()
                    (target / relative).write_text(canonical(raw), encoding="utf-8")
                with self.assertRaises(ValidationError):
                    load_attempt(target)

    def test_wrong_selection_hash_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            prepare_training_attempt(self.demo / "compiled", self.demo / "selection", "wrong",
                self.demo / "source/model_location.json", self.demo / "source/settings.json", self.work / "bad", "r", "a")

    def test_incomplete_prepare_ledger_and_consumed_budget_cannot_be_reset(self) -> None:
        root = self.prepare()
        with StateStore(root / "state.sqlite3", create=False) as state:
            compile_used = read_json(self.demo / "compiled/costs.json")["budgets"]["scoring_tokens"]["used"]
            self.assertEqual(state.snapshot("new_run")["budgets"]["scoring_tokens"]["used"], compile_used)
            with state.transaction():
                state.connection.execute("DELETE FROM events WHERE run_id=? AND event_id=?", ("new_run", "training_plan:new_attempt"))
        with self.assertRaisesRegex(ValidationError, "training_plan.state"):
            load_attempt(root)

    def test_order_has_separate_identity_without_reselecting(self) -> None:
        report = read_json(self.demo / "selection/selection_report.json")
        settings = read_json(self.demo / "source/settings.json")
        reversed_batch = replace(self.plan.batch, items=tuple(reversed(self.plan.batch.items)))
        report["batch"] = reversed_batch.to_dict()
        other = freeze_training_plan(self.inputs, report, reversed_batch, self.plan.selection_manifest_id, settings,
                                      self.location.identity, "r", "a", data_kind="synthetic")
        self.assertEqual(other.batch.fingerprint, self.plan.batch.fingerprint)
        self.assertNotEqual(other.ordered_items_hash, self.plan.ordered_items_hash)

    def test_non_batch_training_use_is_rejected_transactionally(self) -> None:
        root = self.prepare()
        plan, inputs, _ = load_attempt(root)
        outsider = next(r for r in inputs.records if r.is_eligible and r.data_id not in {i.data_id for i in plan.batch.items})
        with StateStore(root / "state.sqlite3", create=False) as state:
            state.reserve("new_run", "new_attempt", "training_tokens", plan.batch.supervised_tokens)
            state.training_event("new_run", "new_attempt", "started", batch_fingerprint=plan.batch.fingerprint)
            with self.assertRaises(ValidationError):
                state.consume("new_run", "bad", "new_attempt", outsider.supervised_tokens,
                    TrainingUse("new_attempt", 1, outsider.data_id, outsider.version, outsider.train_mask_version))
            self.assertEqual(state.snapshot("new_run")["budgets"]["training_tokens"]["used"], 0)

    def test_whole_record_and_skill_intersection_and_missing(self) -> None:
        record = self.selected[0]
        skill = record.skills[0]
        mask = record.raw["supervision"]["loss_mask"]
        positions = [i for i, bit in enumerate(mask) if bit]
        user = next(i for i, role in enumerate(record.raw["tokenization"]["roles"]) if role == "user")
        evidence = (SkillEvidence(record.data_id, record.version, skill, "synthetic_fixture", (user, *positions[:2])),) * 2
        before = self.fit(evidence=evidence)
        after = self.fit(candidate=True, evidence=evidence)
        pair = compare_fit(before, after)
        self.assertEqual(before.records[0].by_skill[self.ontology.ids.index(skill)].stats.token_count, 2)
        self.assertEqual(before.overall.token_count, sum(r.supervised_tokens for r in self.selected))
        self.assertAlmostEqual(pair.overall.delta_nll, -0.1)
        self.assertAlmostEqual(dict(pair.by_skill)[skill].delta_nll, -0.1)
        missing = next(s.stats for s in before.records[1].by_skill)
        self.assertIsNone(missing.nll)
        self.assertEqual(missing.status, "MISSING")
        self.assertEqual(mask, record.raw["supervision"]["loss_mask"])

    def test_user_and_tool_evidence_remains_missing(self) -> None:
        values = example_values()
        source = values["sources"][0]
        source["messages"].insert(1, {"role": "tool", "content": "result", "correctness": "context", "supervise": False, "evidence_id": None, "diagnostics": []})
        bundle = compile_training_sources([source], MockChatTokenizer(), self.inputs.contract, 512)
        evidence_raw = {"schema_version": "message-evidence-v1", "index_rule": "source_message_index_zero_v1", "items": [
            {"data_id": source["data_id"], "trajectory_version": bundle.records[0].version, "skill_id": source["skills"][0],
             "source": "synthetic_fixture", "message_index": i} for i in (0, 1)]}
        evidence = bind_message_evidence(evidence_raw, bundle.mappings)
        result = evaluate_fit(self.location.model_ref, bundle.records, MockTokenScorer(), self.ontology.ids, evidence, source="synthetic_fixture")
        self.assertTrue(all(s.stats.nll is None for s in result.by_skill))
        self.assertGreater(result.overall.token_count, 0)

    def test_token_weighted_aggregation_for_unequal_lengths(self) -> None:
        short = self.selected[1].raw
        positions = [i for i, bit in enumerate(short["supervision"]["loss_mask"]) if bit]
        short["supervision"]["loss_mask"] = [i == positions[0] for i in range(len(short["supervision"]["loss_mask"]))]
        short["scores"] = None
        short_record = TrajectoryRecord.parse(short, self.inputs.contract, "short")
        first = self.selected[0]
        class Scorer:
            def score_tokens(self, r):
                return tuple(TokenScore(i, 1.0 if r.data_id == first.data_id else 9.0, 2.0)
                             for i, bit in enumerate(r.raw["supervision"]["loss_mask"]) if bit)
        result = evaluate_fit(self.location.model_ref, (first, short_record), Scorer(), self.ontology.ids)
        self.assertAlmostEqual(result.overall.nll, (first.supervised_tokens + 9) / (first.supervised_tokens + 1))
        self.assertNotEqual(result.overall.nll, 5)

    def test_fit_rejects_wrong_position_version_and_scope(self) -> None:
        r = self.selected[0]
        bad = SkillEvidence(r.data_id, "old", r.skills[0], "code", (1,))
        with self.assertRaises(ValidationError):
            self.fit(evidence=(bad,))
        with self.assertRaises(ValidationError):
            self.fit(evidence=(replace(bad, trajectory_version=r.version, token_positions=(-1,)),))
        with self.assertRaises(ValidationError):
            evaluate_fit(self.location.model_ref, (replace(r, role="final_test"),), MockTokenScorer(), self.ontology.ids)
        before = self.fit()
        with self.assertRaises(ValidationError):
            compare_fit(before, replace(before, evidence_hash="changed"))
        raw = before.to_dict()
        raw["overall"]["ce_sum"] += 1
        with self.assertRaises(ValidationError):
            parse_fit_evaluation(raw)

    def test_private_tests_not_sent_and_overlapping_skill_denominators(self) -> None:
        suite = self.suite()
        prompts = []
        result = evaluate_execution(self.location.model_ref, suite,
            lambda p, n, s: prompts.append(p) or ("def solve(x,y):\n return x*y" if "multiply" in p else "def solve(x,y):\n return x+y"), self.work / "execution", 2, source="synthetic_fixture")
        self.assertEqual(result.overall.planned, 2)
        self.assertEqual(result.overall.passed, 2)
        self.assertTrue(all(stats.planned == 2 for _, stats in result.by_skill))
        self.assertTrue(all("expected" not in p and "synthetic_fixture" not in p for p in prompts))
        self.assertEqual(parse_execution_evaluation(result.to_dict()), result)
        self.assertEqual(TaskSuiteV2.parse(suite.to_dict(), self.ontology), suite)

    def test_partial_infrastructure_preserves_full_denominator(self) -> None:
        def fail_generation(p, n, s):
            raise RuntimeError("OOM injected")
        result = evaluate_execution(self.location.model_ref, self.suite(), fail_generation, self.work / "execution", 2, source="synthetic_fixture")
        self.assertEqual(result.status, "PARTIAL")
        self.assertEqual(result.overall.planned, 2)
        self.assertEqual(result.overall.infrastructure_error, 1)
        self.assertEqual(result.overall.not_run, 1)
        self.assertIsNone(result.overall.pass_rate)
        self.assertTrue((self.work / "execution/000000.intent.json").exists())

    def test_budget_and_bad_model_output_are_distinct(self) -> None:
        result = evaluate_execution(self.location.model_ref, self.suite(), lambda p, n, s: "import os",
                                    self.work / "execution", 1, source="synthetic_fixture")
        self.assertEqual([r.status for r in result.rows], ["TASK_FAILED", "BUDGET_EXHAUSTED"])
        self.assertEqual(result.overall.attempted, 1)

    def test_isolation_checks_source_id_group_and_fingerprint(self) -> None:
        suite = self.suite()
        for field, value in (("source_problem_id", self.selected[0].data_id), ("split_group", self.selected[0].split_group),
                             ("content_fingerprint", self.selected[0].content_hash)):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                verify_isolation(replace(suite, tasks=(replace(suite.tasks[0], **{field: value}), *suite.tasks[1:])), self.selected)

    def test_final_feedback_cross_suite_isolation(self) -> None:
        feedback, final = self.suite(), self.suite(True)
        verify_suite_isolation(feedback, final)
        for field in ("task_id", "source_problem_id", "split_group", "duplicate_group", "content_fingerprint"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                verify_suite_isolation(feedback, replace(final, tasks=(replace(final.tasks[0], **{field: getattr(feedback.tasks[0], field)}), *final.tasks[1:])))

    def test_duplicate_and_out_of_order_step_rejected(self) -> None:
        root = self.prepare()
        plan, _, _ = load_attempt(root)
        first, second = plan.batch.items[:2]
        with StateStore(root / "state.sqlite3", create=False) as state:
            state.reserve("new_run", "new_attempt", "training_tokens", plan.batch.supervised_tokens)
            state.training_event("new_run", "new_attempt", "started", batch_fingerprint=plan.batch.fingerprint)
            with self.assertRaises(ValidationError):
                state.consume("new_run", "second", "new_attempt", second.supervised_tokens,
                    TrainingUse("new_attempt", 2, second.data_id, second.version, second.mask_version))
            state.consume("new_run", "first", "new_attempt", first.supervised_tokens,
                TrainingUse("new_attempt", 1, first.data_id, first.version, first.mask_version))
            with self.assertRaises(ValidationError):
                state.consume("new_run", "duplicate", "new_attempt", first.supervised_tokens,
                    TrainingUse("new_attempt", 1, first.data_id, first.version, first.mask_version))

    def test_partial_validation_blocks_mechanical_acceptance(self) -> None:
        root = self.prepare()
        run_training_attempt(root, "cpu")
        rule = self.work / "rule.json"
        write_json(rule, {"schema_version": "acceptance-1", "statistical_rule": "paired_hoeffding_two_groups_v1",
            "minimum_gain": 0.1, "retention_tolerance": 0.1, "minimum_tasks_per_group": 2, "alpha": 0.05, "max_retests": 0})
        feedback = run_dual_validation(root, self.demo / "source/feedback_tasks.json", device="cpu", call_budget=1, rule_path=rule)
        self.assertEqual(feedback.status, "PARTIAL")
        self.assertEqual(feedback.acceptance.status, "NOT_RUN")
        self.assertFalse((root / "completion_manifest.json").exists())
        self.assertEqual(parse_round_feedback(feedback.to_dict()), feedback)

    def test_paired_execution_environment_mismatch_rejected(self) -> None:
        root = self.prepare()
        run_training_attempt(root, "cpu")
        original = evaluate_execution_stage
        def different_environment(*args, **kwargs):
            result = original(*args, **kwargs)
            return replace(result, environment_hash="other") if kwargs["candidate"] else result
        with patch("rsi.workflows.validation_run.evaluate_execution_stage", side_effect=different_environment), self.assertRaises(ValidationError):
            run_dual_validation(root, self.demo / "source/feedback_tasks.json", device="cpu", call_budget=2)
        self.assertFalse((root / "completion_manifest.json").exists())

    def test_python_adapter_never_falls_back_to_host_execution(self) -> None:
        raw = self.suite().to_dict()
        raw["tasks"][0].update(adapter_id="bubblewrap_seccomp_v1", payload={"prompt": "Write solve(x,y).", "entry_point": "solve",
                       "setup": "", "tests": ["assert solve(1,2) == 3"]})
        raw["tasks"][0]["content_fingerprint"] = task_fingerprint("bubblewrap_seccomp_v1", raw["tasks"][0]["payload"])
        suite = TaskSuiteV2.parse(raw, self.ontology)
        with self.assertRaises(ValidationError):
            evaluate_execution(self.location.model_ref, suite, lambda p, n, s: "def solve(x,y): return x+y",
                               self.work / "execution", 2, source="synthetic_fixture")
        self.assertFalse((self.work / "execution/completion_manifest.json").exists())

    def test_adapter_load_failure_releases_already_loaded_base(self) -> None:
        import weakref
        from types import SimpleNamespace
        from rsi.models.hf_backend import HfRuntime
        references = []
        class Model:
            def __init__(self):
                self.config = SimpleNamespace(use_cache=True)
                references.append(weakref.ref(self))
            def to(self, device):
                return self
        fake_torch = SimpleNamespace(float32="fp32", bfloat16="bf16")
        fake_transformers = SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **kw: Model()))
        def fail_adapter(*args, **kwargs):
            raise RuntimeError("adapter load failure")
        fake_peft = SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=fail_adapter))
        with patch("rsi.models.hf_backend.load_dependencies", return_value=(fake_torch, fake_transformers, fake_peft)), self.assertRaises(RuntimeError):
            HfRuntime("unused", "cpu", "fp32", "missing-adapter")
        self.assertIsNone(references[0]())

    def test_all_independent_cli_commands_and_exit_status(self) -> None:
        from rsi.commands.cli import main
        def command(*args):
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main([str(a) for a in args])
            self.assertEqual(code, 0, stderr.getvalue())
            return json.loads(stdout.getvalue())
        source = self.demo / "source"
        bundle, selection, attempt = self.work / "bundle", self.work / "selection", self.work / "cli-attempt"
        command("compile-training-data", "--sources", source / "sources.jsonl", "--metadata", source / "metadata.jsonl",
                "--config", source / "config.json", "--ontology", source / "ontology.json", "--policy", source / "policy.json",
                "--model-location", source / "model_location.json", "--output", bundle, "--mode", "mock", "--device", "cpu",
                "--max-length", "512", "--data-kind", "synthetic", "--message-evidence", source / "message_evidence.json")
        command("select-with-policy", "--config", bundle / "config.json", "--data", bundle / "trajectories.jsonl",
                "--labels", bundle / "labels.jsonl", "--ontology", bundle / "ontology.json", "--policy", bundle / "policy.json",
                "--output", selection, "--allow-synthetic-labels")
        command("prepare-training", "--bundle", bundle, "--selection", selection,
                "--selection-id", read_json(selection / "manifest.json")["content_id"], "--model-location", source / "model_location.json",
                "--settings", source / "settings.json", "--output", attempt, "--run-id", "cli", "--attempt-id", "cli_001")
        paused = command("train-attempt", "--attempt", attempt, "--device", "cpu", "--stop-after-step", "1")
        self.assertEqual(paused["status"], "PAUSED")
        command("resume-training", "--attempt", attempt, "--device", "cpu", "--checkpoint", paused["checkpoint"]["path"],
                "--checkpoint-sha256", paused["checkpoint"]["sha256"])
        command("evaluate-fit", "--attempt", attempt, "--device", "cpu", "--model", "parent", "--output", attempt / "validation/fit-parent")
        command("evaluate-execution", "--attempt", attempt, "--device", "cpu", "--model", "parent", "--output", attempt / "validation/execution-parent",
                "--tasks", source / "feedback_tasks.json", "--call-budget", "2")
        result = command("validate-training", "--attempt", attempt, "--device", "cpu", "--tasks", source / "feedback_tasks.json", "--call-budget", "2")
        self.assertEqual(result["source"], "synthetic_fixture")
        command("evaluate-benchmark", "--bundle", bundle, "--model-location", source / "model_location.json", "--mode", "mock",
                "--device", "cpu", "--tasks", source / "final_tasks.json", "--feedback-tasks", source / "feedback_tasks.json",
                "--call-budget", "2", "--output", self.work / "final")
        with redirect_stderr(io.StringIO()):
            code = main(["train-attempt", "--attempt", str(self.work / "missing"), "--device", "cpu"])
        self.assertNotEqual(code, 0)

    def test_feedback_refuses_final_test_in_business_layer(self) -> None:
        feedback = parse_round_feedback(read_json(self.demo / "attempt/validation/round_feedback.json"))
        with self.assertRaisesRegex(ValidationError, "final_test"):
            build_round_feedback(feedback_id="x", round_id="x", attempt_id="x", batch_id="x", plan_hash="x",
               fit_parent=self.fit(), fit_candidate=self.fit(), execution_parent=replace(feedback.execution.before, purpose="final_test"),
               execution_candidate=feedback.execution.after, data_kind="synthetic", label_mode="synthetic_fixture",
               costs=feedback.costs, evidence=feedback.evidence)
        report = read_json(self.demo / "benchmark/benchmark_report.json")
        self.assertFalse(any("acceptance" in key for key in report))

    def test_feedback_parser_rejects_forged_measured_and_placeholders(self) -> None:
        raw = read_json(self.demo / "attempt/validation/round_feedback.json")
        for field, value in (("source", "measured"), ("fit", "TYPED_FIT_COMPARISON_REQUIRED"), ("model_promotion", "APPLIED")):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                parse_round_feedback({**raw, field: value})

    def test_cache_reuse_no_charge_and_identity_mismatch_rejected(self) -> None:
        root = self.prepare()
        output = root / "validation/fit-parent"
        first = evaluate_fit_stage(root, output, candidate=False, device="cpu")
        second = evaluate_fit_stage(root, output, candidate=False, device="cpu")
        self.assertEqual(first, second)
        with patch("rsi.workflows.validation_run.implementation_identity", return_value="changed"), self.assertRaises(ValidationError):
            evaluate_fit_stage(root, output, candidate=False, device="cpu")
        with StateStore(root / "state.sqlite3", create=False) as state:
            compilation = read_json(self.demo / "compiled/costs.json")["budgets"]["scoring_tokens"]["used"]
            self.assertEqual(state.snapshot("new_run")["budgets"]["scoring_tokens"]["used"], compilation + self.plan.batch.context_tokens)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "缺少 torch，无法运行手算 logits 检查。")
    def test_hand_computed_logits_shift_mask_entropy_and_legacy_span_count(self) -> None:
        import torch
        from types import SimpleNamespace
        from rsi.models.hf_backend import HfRuntime
        raw = self.selected[0].raw
        count = len(raw["tokenization"]["token_ids"])
        raw["tokenization"]["token_ids"] = [0] * count
        enabled = [i for i, bit in enumerate(raw["supervision"]["loss_mask"]) if bit]
        raw["tokenization"]["token_ids"][enabled[0]] = 1
        raw["supervision"]["loss_mask"] = [i in enabled[:2] for i in range(count)]
        raw["scores"] = None
        offset = raw["tokenization"]["offsets"][enabled[0]]
        raw["diagnostic_spans"] = [{"skill": self.selected[0].skills[0], "token_start": enabled[0], "token_end": enabled[0] + 1,
            "char_start": offset[0], "char_end": offset[1], "evidence_id": "hand-computed", "mask_version": "content_spans_v1"}]
        record = TrajectoryRecord.parse(raw, self.inputs.contract, "logits")
        logits = torch.zeros((1, count, 2))
        logits[0, enabled[0] - 1] = torch.tensor([0.0, math.log(3)])
        class Model:
            def eval(self):
                pass
            def __call__(self, **kwargs):
                return SimpleNamespace(logits=logits)
        runtime = HfRuntime.__new__(HfRuntime)
        runtime.torch, runtime.device, runtime.model = torch, "cpu", Model()
        values = runtime.score_tokens(record)
        self.assertEqual([v.position for v in values], enabled[:2])
        self.assertAlmostEqual(values[0].ce, -math.log(0.75), places=6)
        self.assertAlmostEqual(values[0].entropy, -(0.25 * math.log(0.25) + 0.75 * math.log(0.75)), places=6)
        self.assertAlmostEqual(values[1].ce, math.log(2), places=6)
        metric = runtime.score(record)
        self.assertEqual(metric.supervised_tokens, 2)
        self.assertEqual(metric.spans[0].token_count, 1)
        self.assertAlmostEqual(metric.ce_sum, sum(v.ce for v in values))


if __name__ == "__main__":
    unittest.main()
