"""输入编码与数值边界的回归检查。"""

from pathlib import Path
import tempfile
import unittest

from rsi.validation import ValidationError, number, read_json, read_jsonl


class InputBoundaryTests(unittest.TestCase):
    def test_oversized_number_is_validation_error(self) -> None:
        with self.assertRaises(ValidationError):
            number(10 ** 1000, "learning_rate")

    def test_non_utf8_files_are_validation_errors(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            path.write_bytes(b'\xff')
            for reader in (read_json, read_jsonl):
                with self.subTest(reader=reader.__name__), self.assertRaisesRegex(ValidationError, "UTF-8"):
                    reader(path)
