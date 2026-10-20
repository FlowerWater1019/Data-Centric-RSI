"""输入编码与数值边界的回归检查。"""

from pathlib import Path
import tempfile
import unittest

from rsi.validation import ValidationError, loads, number, read_json, read_jsonl


class InputBoundaryTests(unittest.TestCase):
    def test_json_rejects_overflowing_exponents_and_oversized_integer_literals(self) -> None:
        for text in ('{"value":1e999}', '{"value":-1e999}', '{"value":' + '9' * 5000 + '}'):
            with self.subTest(prefix=text[:25]), self.assertRaisesRegex(ValidationError, "JSON"):
                loads(text)
        self.assertEqual(loads('{"value":1.5e2}'), {"value": 150.0})

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
