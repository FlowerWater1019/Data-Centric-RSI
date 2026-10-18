"""测试代码执行范围、任务分母和逐任务证据。"""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from rsi.artifacts import write_json
from rsi.backends import EvaluationRequest
from rsi.bounded_python import execute_task
from rsi.coding_evaluation import CodingEvaluationBackend
from rsi.coding_tasks import TaskSuite
from rsi.validation import ValidationError, read_json
from tests.helpers import Fixture, ROOT


class CodingTasksTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = read_json(ROOT / 'examples/coding_tasks.synthetic.json')
        self.task = TaskSuite.parse(self.raw).tasks[0]

    def test_correct_code_and_conditional_expression(self) -> None:
        for code in ('def solve(x, y):\n    return abs(x-y)',
                     '```python\ndef solve(x, y):\n    return x-y if x>y else y-x\n```'):
            self.assertEqual(execute_task(code, self.task).status, 'passed')

    def test_wrong_answer_remains_failed(self) -> None:
        result = execute_task('def solve(x, y):\n    return x-y', self.task)
        self.assertEqual((result.status, result.passed_cases, result.total_cases), ('failed', 1, 3))

    def test_unsupported_programs_never_execute(self) -> None:
        examples = [
            "import os\ndef solve(x, y):\n    return os.system('echo bad')",
            "def solve(x, y):\n    return __import__('os')",
            "def solve(x, y):\n    return 0 if x else open('some-file')",
            "def solve(x, y):\n    while True: pass",
            "@print('bad')\ndef solve(x, y):\n    return 0",
            "def solve(x=print('bad'), y=0):\n    return 0",
            "def solve(x, y):\n    return x ** 999999999",
            "def solve(x, y):\n    return [0] * 999999999",
            "def solve(x, y):\n    return x.__class__",
            "def solve(x, y):\n    return abs",
            "def solve(x, y):\n    return True",
            'x' * 8193,
        ]
        for code in examples:
            with self.subTest(code=code[:50]):
                self.assertEqual(execute_task(code, self.task).status, 'unsupported')

    def test_arithmetic_error_is_task_failure(self) -> None:
        result = execute_task('def solve(x, y):\n    return x // (y-y)', self.task)
        self.assertEqual(result.status, 'runtime_error')

    def test_large_intermediate_is_bounded(self) -> None:
        result = execute_task('def solve(x, y):\n    return 1000000000000*1000000000000', self.task)
        self.assertEqual(result.status, 'unsupported')

    def test_empty_tests_duplicate_tasks_and_final_test_are_rejected(self) -> None:
        for mutate in (lambda r: r.update(role='final_test'),
                       lambda r: r['tasks'][0].update(tests=[]),
                       lambda r: r['tasks'].append(r['tasks'][0]),
                       lambda r: r['tasks'][0].update(group='final_test')):
            raw = read_json(ROOT / 'examples/coding_tasks.synthetic.json')
            mutate(raw)
            with self.assertRaises(ValidationError):
                TaskSuite.parse(raw)


class EvaluationEvidenceTests(Fixture):
    def test_every_task_counts_once_and_evidence_contains_code(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = write_json(root / 'tasks.json', read_json(ROOT / 'examples/coding_tasks.synthetic.json'))
            calls = []
            backend = CodingEvaluationBackend(lambda prompt, limit: 'def solve(x, y):\n    return abs(x-y)',
                                              root / 'evaluation', calls.append)
            result = backend.evaluate(EvaluationRequest(self.contract.model, manifest, 17))
            self.assertEqual(len(calls), 4)
            self.assertEqual([(m.value, m.sample_count) for m in result.metrics], [(0.5, 2), (0.0, 2)])
            evidence = read_json(result.evidence.path)
            self.assertEqual(len(evidence['tasks']), 4)
            self.assertIn('code', evidence['tasks'][0])
            with self.assertRaises(FileExistsError):
                backend.evaluate(EvaluationRequest(self.contract.model, manifest, 17))

    def test_changed_manifest_is_rejected_before_generation(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = write_json(root / 'tasks.json', read_json(ROOT / 'examples/coding_tasks.synthetic.json'))
            Path(manifest.path).write_text('{}')
            backend = CodingEvaluationBackend(lambda p, n: self.fail('must not generate'), root / 'output', lambda _: None)
            with self.assertRaises(ValidationError):
                backend.evaluate(EvaluationRequest(self.contract.model, manifest, 17))
            self.assertFalse((root / 'output').exists())
