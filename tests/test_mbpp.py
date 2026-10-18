"""真实数据格式测试。所有测试行由本文件合成，不依赖下载。"""

import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from rsi.artifacts import file_sha256, verify_training_artifact
from rsi.backends import ArtifactRef
from rsi.dataset_import import import_mbpp
from rsi.mbpp import MbppRecord, duplicate_groups, engineering_role
from rsi.python_tasks import PythonTaskSuite
from rsi.sandbox import SandboxResult
from rsi.validation import ValidationError, read_json, read_jsonl
from tests.helpers import Fixture, ROOT


def row(task_id: int) -> dict:
    return {"task_id": task_id, "text": f"Synthetic task {task_id}",
            "code": f"def f{task_id}(x):\n    return x + {task_id}", "test_setup_code": "",
            "test_list": [f"assert f{task_id}(0) == {task_id}"], "challenge_test_list": []}


class MbppTests(Fixture):
    def test_official_boundaries_precede_role_assignment(self) -> None:
        for task_id, split in [(1, 'prompting'), (10, 'prompting'), (11, 'test'), (510, 'test'),
                               (511, 'validation'), (600, 'validation'), (601, 'train'), (974, 'train')]:
            record = MbppRecord.parse(row(task_id))
            self.assertEqual(record.official_split, split)
            if split == 'test':
                self.assertEqual(engineering_role(record, 'group'), 'final_test')

    def test_duplicate_chain_connects_prompt_and_code(self) -> None:
        a, b, c = row(601), row(602), row(603)
        b['text'] = a['text'].upper() + '  '
        c['code'] = b['code']
        groups = duplicate_groups(tuple(MbppRecord.parse(item) for item in [a, b, c]))
        self.assertEqual(len(set(groups.values())), 1)

    def test_signature_does_not_execute_reference_code(self) -> None:
        raw = row(601)
        raw['code'] = "raise RuntimeError('must never execute on host')\n" + raw['code']
        self.assertEqual(MbppRecord.parse(raw).signature(), ('f601', 'def f601(x):'))

    def test_import_freezes_subset_and_never_executes_final_test(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'input.jsonl'
            records = [row(i) for i in range(1, 975)]
            # 训练数据与官方测试数据重复时，训练副本必须排除。
            records[600]['code'] = records[10]['code']
            source.write_text('\n'.join(json.dumps(value) for value in records), encoding='utf-8')
            revision = 'a' * 40
            origin = {'dataset':'google-research/mbpp', 'revision':revision, 'license':'CC-BY-4.0',
                      'sha256':file_sha256(source),
                      'url':f'https://raw.githubusercontent.com/google-research/google-research/{revision}/mbpp/mbpp.jsonl'}
            (root / 'origin.json').write_text(json.dumps(origin))
            calls = []
            def execute(code: str, test: str, setup: str) -> SandboxResult:
                calls.append(test)
                return SandboxResult('passed', None, 0, 0.01, '')
            with patch('rsi.dataset_import.BubblewrapExecutor') as factory:
                factory.return_value.preflight.return_value = {'test_fixture':True}
                factory.return_value.run.side_effect = execute
                result = import_mbpp(source, root/'origin.json', ROOT/'configs/sandbox.json', root/'bundle', 2)
            self.assertEqual(result['sources'], 6)
            self.assertEqual(result['feedback_tasks'], 4)
            self.assertEqual(len(calls), 10)
            self.assertTrue(all('f601(' not in test for test in calls))
            for test in calls:
                task_id = int(test.split('f')[1].split('(')[0])
                self.assertGreaterEqual(task_id, 511)
            selection = read_json(root/'bundle/selection.json')
            self.assertEqual(sum(row['reason']=='final_test' for row in selection['excluded']), 500)
            sources = read_jsonl(root/'bundle/sources.jsonl')
            self.assertTrue(all(int(item['data_id'].split('_')[1])>=601 for item in sources))
            suite = PythonTaskSuite.parse(read_json(root/'bundle/feedback_tasks.json'))
            self.assertEqual(len(suite.tasks), 4)
            verify_training_artifact(ArtifactRef(**result['manifest']))

    def test_wrong_hash_fails_before_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'input').write_text('{}')
            revision='a'*40
            origin={'dataset':'google-research/mbpp','revision':revision,'license':'CC-BY-4.0','sha256':'wrong',
                    'url':f'https://raw.githubusercontent.com/google-research/google-research/{revision}/mbpp/mbpp.jsonl'}
            (root/'origin').write_text(json.dumps(origin))
            with patch('rsi.dataset_import.BubblewrapExecutor') as factory:
                with self.assertRaisesRegex(ValidationError,'哈希'):
                    import_mbpp(root/'input',root/'origin',ROOT/'configs/sandbox.json',root/'out',1)
                factory.assert_not_called()
