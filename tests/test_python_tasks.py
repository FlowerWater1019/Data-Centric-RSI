"""隔离任务格式与通用评估接口。默认测试使用可注入的执行器。"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from rsi.artifacts import write_json
from rsi.backends import EvaluationRequest
from rsi.isolated_evaluation import IsolatedEvaluationBackend
from rsi.python_tasks import PythonTaskSuite
from rsi.sandbox import SandboxResult, SandboxUnavailable
from rsi.validation import ValidationError, read_json
from tests.helpers import Fixture


def suite_raw() -> dict:
    return {'schema_version':'python-tasks-1','purpose':'engineering_validation','role':'feedback_execution',
            'executor':'bubblewrap_seccomp_v1','max_new_tokens':128,
            'provenance':{'dataset':'synthetic-test','revision':'fixture','sha256':'fixture','license':'fixture','split_policy':'fixture'},
            'tasks':[{'task_id':name,'group':group,'split_group':name,'duplicate_group':name,'prompt':'Return x + 1.',
                      'entry_point':'solve','setup':'','tests':['assert solve(1)==2','assert solve(2)==3']}
                     for name,group in [('a','target'),('b','retention')]]}


class PythonTaskTests(unittest.TestCase):
    def test_rejects_final_test_empty_assertions_and_duplicates(self) -> None:
        for change in (lambda raw: raw.update(role='final_test'),
                       lambda raw: raw['tasks'][0].update(tests=[]),
                       lambda raw: raw['tasks'][0].update(tests=['pass']),
                       lambda raw: raw['tasks'][1].update(duplicate_group='a')):
            raw=suite_raw()
            change(raw)
            with self.assertRaises(ValidationError):
                PythonTaskSuite.parse(raw)

    def test_test_parsing_does_not_execute(self) -> None:
        raw=suite_raw()
        raw['tasks'][0]['tests']=["assert __import__('os').system('must-not-run') == 0"]
        self.assertEqual(len(PythonTaskSuite.parse(raw).tasks),2)


class IsolatedEvidenceTests(Fixture):
    def test_all_cases_and_failed_tasks_remain_in_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            manifest=write_json(root/'tasks.json',suite_raw())
            executor=Mock()
            executor.run.side_effect=[SandboxResult(status,None,0,0.01,'') for status in ['passed','passed','passed','failed']]
            calls=[]
            backend=IsolatedEvaluationBackend(lambda prompt,n:'def solve(x):\n    return x+1',root/'out',calls.append,executor,{})
            result=backend.evaluate(EvaluationRequest(self.contract.model,manifest,17))
            self.assertEqual(calls,['a','b'])
            self.assertEqual(executor.run.call_count,4)
            self.assertEqual([metric.value for metric in result.metrics],[1.0,0.0])
            self.assertEqual(read_json(result.evidence.path)['test_calls_started'],4)

    def test_infrastructure_failure_cannot_become_model_failure(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            manifest=write_json(root/'tasks.json',suite_raw())
            executor=Mock()
            executor.run.side_effect=SandboxUnavailable('injected unavailable')
            backend=IsolatedEvaluationBackend(lambda prompt,n:'def solve(x):\n    return x+1',root/'out',lambda _:None,executor,{})
            with self.assertRaises(SandboxUnavailable):
                backend.evaluate(EvaluationRequest(self.contract.model,manifest,17))
            self.assertTrue((root/'out/task-000000-test-000.started.json').is_file())
            self.assertFalse((root/'out/evaluation.json').exists())

    def test_invalid_generated_format_is_counted_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            manifest=write_json(root/'tasks.json',suite_raw())
            executor=Mock()
            backend=IsolatedEvaluationBackend(lambda prompt,n:'Explanation\n```python\npass\n```',root/'out',lambda _:None,executor,{})
            result=backend.evaluate(EvaluationRequest(self.contract.model,manifest,17))
            executor.run.assert_not_called()
            self.assertEqual([metric.value for metric in result.metrics],[0.0,0.0])
