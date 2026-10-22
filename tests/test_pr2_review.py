"""PR #2 独立审查的业务复现。重建清单不能绕过冻结内容核验。"""

from dataclasses import replace
from pathlib import Path
import copy
import unittest
from unittest.mock import patch

from rsi.configuration.validation import ValidationError, canonical, read_json
from rsi.storage.execution_artifacts import complete_execution
from rsi.evaluation.task_adapters import TaskSuiteV2, verify_suite_isolation
from rsi.workflows.validation_run import evaluate_execution_stage
from rsi.workflows.validation_run import evaluate_fit_stage
from rsi.workflows.attempt_run import run_training_attempt
from rsi.evaluation.execution_evaluation import summarize, verify_isolation
from rsi.datasets.content_identity import task_fingerprint
from rsi.models.training_mock import MockTokenScorer
from tests import test_training_validation as fixtures


class Pr2ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixtures.TrainingValidationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls) -> None:
        fixtures.TrainingValidationTests.tearDownClass.__func__(cls)

    setUp = fixtures.TrainingValidationTests.setUp
    tearDown = fixtures.TrainingValidationTests.tearDown
    prepare = fixtures.TrainingValidationTests.prepare
    suite = fixtures.TrainingValidationTests.suite

    def test_default_final_questions_are_different(self) -> None:
        feedback, final = self.suite(), self.suite(True)
        self.assertFalse({t.payload.prompt for t in feedback.tasks} & {t.payload.prompt for t in final.tasks})

    def test_fingerprint_must_be_derived_from_payload(self) -> None:
        raw = self.suite().to_dict()
        raw["tasks"][0]["content_fingerprint"] = "arbitrary_nonempty_string"
        with self.assertRaises(ValidationError):
            TaskSuiteV2.parse(raw, self.ontology)

    def test_same_content_with_different_ids_and_claimed_hashes_is_rejected(self) -> None:
        feedback, final = self.suite(), self.suite(True)
        raw = final.to_dict()
        raw["tasks"][0]["payload"] = feedback.to_dict()["tasks"][0]["payload"]
        with self.assertRaises(ValidationError):
            altered = TaskSuiteV2.parse(raw, self.ontology)
            verify_suite_isolation(feedback, altered)

    def assert_cache_change_rejected(self, field: str, value: object) -> None:
        root = self.prepare()
        stage = self.work / "execution"
        evaluate_execution_stage(root, self.demo / "source/feedback_tasks.json", stage, candidate=False, device="cpu", call_budget=2)
        raw = read_json(stage / "evaluation.json")
        raw["rows"][0][field] = value
        (stage / "evaluation.json").write_text(canonical(raw), encoding="utf-8")
        (stage / "completion_manifest.json").unlink()
        complete_execution(stage, "COMPLETED")
        with self.assertRaises(ValidationError):
            evaluate_execution_stage(root, self.demo / "source/feedback_tasks.json", stage, candidate=False, device="cpu", call_budget=2)

    def test_execution_cache_rejects_unknown_task(self) -> None:
        self.assert_cache_change_rejected("task_id", "NOT_IN_FROZEN_SUITE")

    def test_execution_cache_rejects_changed_seed(self) -> None:
        self.assert_cache_change_rejected("seed", 987654321)

    def test_same_public_problem_with_changed_tests_is_still_overlap(self) -> None:
        feedback, final = self.suite(), self.suite(True)
        raw = final.to_dict()
        raw["tasks"][0]["payload"]["prompt"] = " \n" + feedback.tasks[0].payload.prompt.replace(" ", "  ") + "\r\n"
        raw["tasks"][0]["content_fingerprint"] = task_fingerprint(raw["tasks"][0]["adapter_id"], raw["tasks"][0]["payload"])
        altered = TaskSuiteV2.parse(raw, self.ontology)
        with self.assertRaises(ValidationError):
            verify_suite_isolation(feedback, altered)

    def test_training_source_uses_comparable_public_problem_identity(self) -> None:
        suite = self.suite()
        source = {"data_id": "independent_id", "messages": [
            {"role": "user", "content": "\n" + suite.tasks[0].payload.prompt + "  "}]}
        with self.assertRaises(ValidationError):
            verify_isolation(suite, self.selected, [source])

    def test_python_fingerprint_ignores_layout_comments_and_test_order(self) -> None:
        payload = {"prompt": "Repair the function", "entry_point": "solve", "setup": "offset=1\n",
                   "tests": ["assert solve(1,2)==3", "assert solve(0,0)==0"]}
        other = {**payload, "prompt": " Repair  the function\r\n", "setup": "# environment\noffset = 1\n",
                 "tests": ["assert solve(0, 0) == 0", "# private\nassert solve(1, 2) == 3"]}
        self.assertEqual(task_fingerprint("bubblewrap_seccomp_v1", payload), task_fingerprint("bubblewrap_seccomp_v1", other))

    def test_old_unbound_suite_version_is_explicitly_rejected(self) -> None:
        raw = self.suite().to_dict()
        raw["schema_version"] = "execution-suite-v1"
        with self.assertRaises(ValidationError):
            TaskSuiteV2.parse(raw, self.ontology)

    def test_cache_rejects_self_consistent_task_count_order_groups_and_skills(self) -> None:
        root = self.prepare()
        raw = self.suite().to_dict()
        extra = self.suite(True).to_dict()["tasks"][0]
        extra["group"] = "retention"
        raw["tasks"].append(extra)
        tasks = self.work / "tasks.json"
        from rsi.storage.artifacts import write_json
        write_json(tasks, raw)
        mutations = {
            "omission": lambda rows: rows[:2],
            "extra": lambda rows: (*rows, replace(rows[0], task_id="extra_task")),
            "order": lambda rows: tuple(reversed(rows)),
            "groups": lambda rows: (replace(rows[0], group="retention"), replace(rows[1], group="target"), rows[2]),
            "skills": lambda rows: (replace(rows[0], skills=()), *rows[1:]),
            "prompt": lambda rows: (replace(rows[0], public_input="modified public task"), *rows[1:]),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                stage = self.work / name
                report = evaluate_execution_stage(root, tasks, stage, candidate=False, device="cpu", call_budget=3)
                rows = mutate(report.rows)
                groups, skills = sorted({r.group for r in rows}), sorted({s for r in rows for s in r.skills})
                modified = replace(report, rows=rows, overall=summarize(rows),
                    by_group=tuple((g, summarize(tuple(r for r in rows if r.group == g))) for g in groups),
                    by_skill=tuple((s, summarize(tuple(r for r in rows if s in r.skills))) for s in skills),
                    by_group_and_skill=tuple((g,s,summarize(tuple(r for r in rows if r.group == g and s in r.skills))) for g in groups for s in skills))
                (stage / "evaluation.json").write_text(canonical(modified.to_dict()), encoding="utf-8")
                for index, row in enumerate(rows):
                    (stage / f"{index:06d}.result.json").write_text(canonical(row), encoding="utf-8")
                (stage / "completion_manifest.json").unlink()
                complete_execution(stage, "COMPLETED")
                with self.assertRaises(ValidationError):
                    evaluate_execution_stage(root, tasks, stage, candidate=False, device="cpu", call_budget=3)

    def test_cache_rejects_changed_call_generation_and_judgement_facts(self) -> None:
        root = self.prepare()
        for name in ("intent", "generated", "judgement", "result"):
            with self.subTest(name=name):
                stage = self.work / name
                evaluate_execution_stage(root, self.demo / "source/feedback_tasks.json", stage, candidate=False, device="cpu", call_budget=2)
                path = stage / ("000000." + name + ".json")
                value = read_json(path)
                value["task_id"] = "wrong_evidence_task"
                path.write_text(canonical(value), encoding="utf-8")
                (stage / "completion_manifest.json").unlink()
                complete_execution(stage, "COMPLETED")
                with self.assertRaises(ValidationError):
                    evaluate_execution_stage(root, self.demo / "source/feedback_tasks.json", stage, candidate=False, device="cpu", call_budget=2)

    def test_fit_uses_exact_training_batch_and_one_token_pass_for_all_skills(self) -> None:
        root = self.prepare()
        captured = []
        original = MockTokenScorer.score_tokens
        def observe(scorer, record):
            captured.append((scorer.candidate, record.data_id, record.raw))
            return original(scorer, record)
        with patch.object(MockTokenScorer, "score_tokens", observe):
            before = evaluate_fit_stage(root, self.work / "fit-parent", candidate=False, device="cpu")
            run_training_attempt(root, "cpu")
            after = evaluate_fit_stage(root, self.work / "fit-candidate", candidate=True, device="cpu")
        ids = [item.data_id for item in self.plan.batch.items]
        self.assertEqual([item[1] for item in captured], ids + ids)
        self.assertEqual([item[2] for item in captured[:len(ids)]], [item[2] for item in captured[len(ids):]])
        self.assertEqual(before.overall.token_count, self.plan.batch.supervised_tokens)
        self.assertEqual(after.overall.token_count, self.plan.batch.supervised_tokens)
        for row in before.records:
            raw = next(item[2] for item in captured if item[1] == row.data_id)
            self.assertEqual({v.position for v in row.token_scores}, {i for i, bit in enumerate(raw["supervision"]["loss_mask"]) if bit})
            self.assertTrue(any(s.stats.token_count for s in row.by_skill))
