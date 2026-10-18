"""产物内容校验与拒绝覆盖。"""

from pathlib import Path
import tempfile
import unittest

from rsi.artifacts import file_sha256, verify_training_artifact, write_json
from rsi.validation import ValidationError


class ArtifactTests(unittest.TestCase):
    def test_changed_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            payload = root / 'adapter.txt'
            payload.write_text('original', encoding='utf-8')
            ref = write_json(root / 'manifest.json', {'files':[{'name':'adapter.txt','sha256':file_sha256(payload)}]})
            verify_training_artifact(ref)
            payload.write_text('changed', encoding='utf-8')
            with self.assertRaises(ValidationError):
                verify_training_artifact(ref)

    def test_existing_artifact_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'artifact.json'
            write_json(path, {'version':1})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_json(path, {'version':2})
            self.assertEqual(path.read_bytes(), original)
