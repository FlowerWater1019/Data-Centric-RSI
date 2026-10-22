"""多步环境测试。策略和环境均为显式测试实现，不代表真实 Agent 能力。"""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import Mock

from rsi.configuration.validation import ValidationError, canonical, loads, read_json
from rsi.datasets.content_identity import task_fingerprint
from rsi.evaluation.agent_tasks import AgentAction, AgentTaskRunner, AgentVerdict, EnvironmentStep
from rsi.evaluation.execution_evaluation import evaluate_execution, parse_execution_evaluation
from rsi.evaluation.task_adapters import TaskSuiteV2
from rsi.storage.artifacts import write_json
from rsi.storage.execution_artifacts import complete_execution
from rsi.configuration.validation import ValidationError
from rsi.workflows.attempt_run import run_training_attempt
from rsi.workflows.benchmark_run import evaluate_benchmark
from rsi.workflows.validation_run import evaluate_execution_stage, run_dual_validation
from tests import test_training_validation as fixtures


class CounterPolicy:
    def __init__(self, weak: bool = False, stride: int = 1) -> None:
        self.weak, self.stride = weak, stride
        self.seen = []

    def act(self, prompt, observation_json, history, seed, step):
        self.seen.append((prompt, observation_json, history))
        target = int(prompt.split()[-1])
        value = loads(observation_json)["value"]
        if self.weak or value >= target:
            return AgentAction("finish", "{}")
        return AgentAction("increment", canonical({"amount": min(self.stride, target - value)}))


class CounterEnvironment:
    def __init__(self, owner: list, fail: bool = False) -> None:
        self.closed = False
        self.fail = fail
        owner.append(self)

    def reset(self, state):
        self.value = loads(state)["value"]
        return canonical({"value": self.value})

    def step(self, action):
        if self.fail:
            raise RuntimeError("injected environment failure")
        if action.name != "increment":
            raise ValueError("unsupported action")
        self.value += loads(action.arguments_json)["amount"]
        return EnvironmentStep(canonical({"value": self.value}), False)

    def verify(self, verifier_id, config):
        passed = self.value == loads(config)["target"]
        return AgentVerdict(passed, float(passed), "environment goal checked")

    def close(self):
        self.closed = True


class AgentEvaluationTests(unittest.TestCase):
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

    def agent_suite(self, final=False, max_steps=10):
        raw = self.suite(final).to_dict()
        for task, target in zip(raw["tasks"], (2, 4) if final else (3, 5)):
            task["adapter_id"] = "agent_environment_v1"
            task["payload"] = {"prompt": "Increase the counter until it reaches " + str(target), "initial_state": {"value": 0},
                "verifier": {"id": "counter-goal-v1", "config": {"target": target, "private_marker": "DO_NOT_SHOW_POLICY"}}, "max_steps": max_steps}
            task["content_fingerprint"] = task_fingerprint(task["adapter_id"], task["payload"])
        return TaskSuiteV2.parse(raw, self.ontology)

    def runner(self, policy=None, model=None, fail=False):
        environments = []
        policy = policy or CounterPolicy()
        runner = AgentTaskRunner(policy, lambda: CounterEnvironment(environments, fail), "counter-runtime-v1", model or self.location.model_ref)
        return runner, environments, policy

    def test_agent_acts_multiple_steps_without_reference_trajectory(self) -> None:
        runner, environments, policy = self.runner()
        generate = Mock(side_effect=AssertionError("must not use code generation"))
        result = evaluate_execution(self.location.model_ref, self.agent_suite(), generate, self.work / "execution", 2,
                                    source="synthetic_fixture", agent_runner=runner)
        generate.assert_not_called()
        self.assertEqual(result.overall.passed, 2)
        self.assertEqual(result.overall.mean_score, 1)
        self.assertTrue(all(env.closed for env in environments))
        self.assertTrue(all(stats.pass_rate == 1 for _, stats in result.by_skill))
        self.assertTrue(all("DO_NOT_SHOW_POLICY" not in repr(item) for item in policy.seen))
        self.assertGreater(read_json(self.work / "execution/000000.agent/outcome.json")["steps"], 1)
        self.assertEqual(parse_execution_evaluation(result.to_dict()), result)

    def test_different_valid_action_paths_both_pass(self) -> None:
        counts = []
        for stride in (1, 2):
            runner, _, _ = self.runner(CounterPolicy(stride=stride))
            root = self.work / str(stride)
            result = evaluate_execution(self.location.model_ref, self.agent_suite(), None, root, 2,
                                        source="synthetic_fixture", agent_runner=runner)
            self.assertEqual(result.overall.passed, 2)
            counts.append(read_json(root / "000001.agent/outcome.json")["steps"])
        self.assertNotEqual(*counts)

    def test_step_limit_is_task_failure_not_fake_pass(self) -> None:
        runner, environments, _ = self.runner()
        result = evaluate_execution(self.location.model_ref, self.agent_suite(max_steps=1), None, self.work / "execution", 2,
                                    source="synthetic_fixture", agent_runner=runner)
        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.overall.passed, 0)
        self.assertEqual(result.overall.mean_score, 0)
        self.assertTrue(all(env.closed for env in environments))

    def test_environment_can_terminate_without_agent_finish(self) -> None:
        class TerminatingEnvironment(CounterEnvironment):
            def step(self, action: AgentAction) -> EnvironmentStep:
                result = super().step(action)
                return EnvironmentStep(result.observation_json, True)

        environments = []
        runner = AgentTaskRunner(CounterPolicy(stride=100), lambda: TerminatingEnvironment(environments),
                                 "counter-terminates-v1", self.location.model_ref)
        root = self.work / "execution"
        result = evaluate_execution(self.location.model_ref, self.agent_suite(), None, root, 2,
                                    source="synthetic_fixture", agent_runner=runner)
        self.assertEqual(result.overall.passed, 2)
        outcome = read_json(root / "000000.agent/outcome.json")
        self.assertEqual(outcome["stop_reason"], "ENVIRONMENT_TERMINATED")
        self.assertEqual(outcome["steps"], 1)
        self.assertTrue(all(env.closed for env in environments))

    def test_fractional_grade_and_untagged_tasks_keep_full_denominator(self) -> None:
        class PartialCreditEnvironment(CounterEnvironment):
            def verify(self, verifier_id: str, config: str) -> AgentVerdict:
                return AgentVerdict(False, 0.25, "partial completion")

        raw = self.agent_suite().to_dict()
        raw["tasks"][1]["skills"] = []
        suite = TaskSuiteV2.parse(raw, self.ontology)
        environments = []
        runner = AgentTaskRunner(CounterPolicy(weak=True), lambda: PartialCreditEnvironment(environments),
                                 "counter-partial-v1", self.location.model_ref)
        result = evaluate_execution(self.location.model_ref, suite, None, self.work / "execution", 2,
                                    source="synthetic_fixture", agent_runner=runner)
        self.assertEqual(result.overall.planned, 2)
        self.assertEqual(result.overall.pass_rate, 0)
        self.assertEqual(result.overall.mean_score, 0.25)
        self.assertTrue(all(stats.planned == 1 for _, stats in result.by_skill))
        self.assertTrue(all(env.closed for env in environments))

    def test_invalid_verifier_score_keeps_failure_and_releases_environment(self) -> None:
        class InvalidScoreEnvironment(CounterEnvironment):
            def verify(self, verifier_id: str, config: str) -> AgentVerdict:
                return AgentVerdict(True, 1.5, "invalid normalized score")

        environments = []
        runner = AgentTaskRunner(CounterPolicy(), lambda: InvalidScoreEnvironment(environments),
                                 "counter-invalid-v1", self.location.model_ref)
        result = evaluate_execution(self.location.model_ref, self.agent_suite(), None, self.work / "execution", 2,
                                    source="synthetic_fixture", agent_runner=runner)
        self.assertEqual([row.status for row in result.rows], ["INFRA_ERROR", "NOT_RUN"])
        self.assertIsNone(result.overall.pass_rate)
        self.assertIn("agent.verdict.score", result.rows[0].error)
        self.assertTrue(all(env.closed for env in environments))

    def test_agent_model_mismatch_rejects_before_environment_creation(self) -> None:
        runner, environments, policy = self.runner()
        other = replace(self.location.model_ref, checkpoint_id="different_checkpoint")
        with self.assertRaisesRegex(ValidationError, "agent_runner.model"):
            evaluate_execution(other, self.agent_suite(), None, self.work / "execution", 2,
                               source="synthetic_fixture", agent_runner=runner)
        self.assertEqual(environments, [])
        self.assertEqual(policy.seen, [])
        self.assertFalse((self.work / "execution").exists())

    def test_environment_failure_keeps_trace_and_closes_resource(self) -> None:
        runner, environments, _ = self.runner(fail=True)
        result = evaluate_execution(self.location.model_ref, self.agent_suite(), None, self.work / "execution", 2,
                                    source="synthetic_fixture", agent_runner=runner)
        self.assertEqual([r.status for r in result.rows], ["INFRA_ERROR", "NOT_RUN"])
        self.assertIsNone(result.overall.mean_score)
        self.assertTrue(environments[0].closed)
        self.assertTrue((self.work / "execution/000000.agent/000001.action.json").exists())

    def test_missing_backend_rejects_without_generating_code(self) -> None:
        generate = Mock()
        with self.assertRaisesRegex(ValidationError, "agent_runner"):
            evaluate_execution(self.location.model_ref, self.agent_suite(), generate, self.work / "execution", 2, source="synthetic_fixture")
        generate.assert_not_called()
        self.assertFalse((self.work / "execution").exists())

    def test_feedback_and_final_benchmark_share_agent_execution(self) -> None:
        root = self.prepare()
        trained = run_training_attempt(root, "cpu")
        parent, _, _ = self.runner(CounterPolicy(weak=True))
        candidate, _, _ = self.runner(CounterPolicy(stride=2), trained.candidate_model)
        tasks = self.work / "feedback.json"
        final = self.work / "final.json"
        write_json(tasks, self.agent_suite().to_dict())
        write_json(final, self.agent_suite(True).to_dict())
        feedback = run_dual_validation(root, tasks, device="cpu", call_budget=2, parent_runner=parent, candidate_runner=candidate)
        self.assertEqual(feedback.execution.before.overall.pass_rate, 0)
        self.assertEqual(feedback.execution.after.overall.pass_rate, 1)
        self.assertEqual(feedback.source, "synthetic_fixture")
        report = evaluate_benchmark(self.demo / "compiled", self.demo / "source/model_location.json", final, self.work / "benchmark",
                     mode="mock", device="cpu", call_budget=2, feedback_suite_path=tasks, agent_runner=parent)
        self.assertEqual(report["purpose"], "final_test")
        self.assertFalse(any("acceptance" in k for k in report))

    def test_agent_cache_does_not_replay_and_rejects_changed_trace_after_rehash(self) -> None:
        root = self.prepare()
        runner, environments, policy = self.runner()
        tasks, stage = self.work / "tasks.json", self.work / "execution"
        write_json(tasks, self.agent_suite().to_dict())

        def evaluate():
            return evaluate_execution_stage(root, tasks, stage, candidate=False, device="cpu",
                                            call_budget=2, agent_runner=runner)

        first = evaluate()
        action_count = len(policy.seen)
        self.assertEqual(first, evaluate())
        self.assertEqual(len(environments), 2)
        self.assertEqual(len(policy.seen), action_count)
        self.assertTrue(all(env.closed for env in environments))
        action_file = stage / "000000.agent/000001.action.json"
        action = read_json(action_file)
        action["arguments"]["amount"] = 999
        action_file.write_text(canonical(action), encoding="utf-8")
        (stage / "completion_manifest.json").unlink()
        complete_execution(stage, "COMPLETED")
        with self.assertRaises(ValidationError):
            evaluate()
        self.assertEqual(len(policy.seen), action_count)
