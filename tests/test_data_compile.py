"""真实数据编译接口测试。使用合成数据包和可注入 tokenizer。"""

from pathlib import Path
import tempfile
from unittest.mock import patch

from rsi.artifacts import file_sha256, write_json
from rsi.data_compile import compile_bundle
from rsi.dataset_import import write_jsonl
from rsi.validation import ValidationError, read_jsonl
from tests.helpers import Fixture, ROOT
from tests.test_tokenization import CharacterTokenizer


class DataCompileTests(Fixture):
    def setUp(self) -> None:
        super().setUp()
        folder=tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root=Path(folder.name)
        self.bundle=self.root/'bundle'
        self.bundle.mkdir()
        source=read_jsonl(ROOT/'examples/pilot_messages.jsonl')[0]
        write_jsonl(self.bundle/'sources.jsonl',[source])
        write_json(self.bundle/'feedback_tasks.json',{'fixture':True})
        write_json(self.bundle/'origin.json',{'fixture':True})
        self.artifact=write_json(self.bundle/'manifest.json',{'schema_version':'import-bundle-1','data_kind':'real_public',
            'files':[{'name':path.name,'sha256':file_sha256(path)} for path in self.bundle.iterdir()]})
        self.config['purpose']='draft'
        write_json(self.root/'config.json',self.config)
        self.identity={'content_id':'local-test-model','files':[]}

    def compile(self, limit: int = 1024) -> dict:
        with patch('rsi.data_compile.model_manifest',return_value=self.identity), \
             patch('rsi.data_compile.load_tokenizer',return_value=CharacterTokenizer()):
            return compile_bundle(self.bundle,self.artifact.sha256,self.root/'config.json','local-test-model',self.root/'compiled',limit)

    def test_compiled_data_has_real_tokenization_and_no_fabricated_scores(self) -> None:
        report=self.compile()
        self.assertEqual(report['records'],1)
        self.assertEqual(report['scores'],'NOT_COMPUTED')
        records=read_jsonl(self.root/'compiled/trajectories.jsonl')
        self.assertIsNone(records[0]['scores'])
        self.assertEqual(records[0]['tokenization']['status'],'verified')

    def test_overlength_cannot_silently_create_partial_output(self) -> None:
        with self.assertRaisesRegex(ValidationError,'截断'):
            self.compile(8)
        self.assertFalse((self.root/'compiled').exists())

    def test_bundle_tampering_rejected_before_tokenizer_load(self) -> None:
        (self.bundle/'sources.jsonl').write_text('{}')
        with patch('rsi.data_compile.load_tokenizer') as tokenizer:
            with self.assertRaises(ValidationError):
                compile_bundle(self.bundle,self.artifact.sha256,self.root/'config.json','unused',self.root/'compiled',1024)
            tokenizer.assert_not_called()
