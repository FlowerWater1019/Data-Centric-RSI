"""功能子包迁移的公开行为：兼容导入、源码身份和执行路径。"""

import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

from rsi._compat import LEGACY_MODULES
from rsi._paths import PACKAGE_ROOT
from rsi.models.backends import TrainingRequest
from rsi.storage.checkpoints import StepJournal, request_identity
from rsi.storage.provenance import source_manifest
from rsi.configuration.validation import ValidationError, digest
from rsi.curation.selection import FixedSelector, build_batch
from tests.helpers import Fixture, ROOT


class PackageLayoutTests(Fixture):
    def test_old_and_new_imports_share_one_module_object(self) -> None:
        for old, canonical in LEGACY_MODULES.items():
            with self.subTest(old=old):
                module = importlib.import_module(old)
                self.assertIs(module, importlib.import_module(canonical))
                self.assertEqual(module.__spec__.name, canonical)

    def test_old_path_patch_changes_the_real_implementation(self) -> None:
        canonical = importlib.import_module("rsi.datasets.data_compile")
        with patch("rsi.data_compile.load_tokenizer") as tokenizer:
            self.assertIs(canonical.load_tokenizer, tokenizer)
            canonical.load_tokenizer("fixture")
            tokenizer.assert_called_once_with("fixture")

    def test_basic_and_legacy_imports_do_not_load_model_dependencies_or_worker(self) -> None:
        script = (
            "import importlib,sys; from rsi._compat import LEGACY_MODULES; "
            "[importlib.import_module(name) for name in LEGACY_MODULES]; "
            "assert not any(name.split('.')[0] in {'torch','transformers','peft'} for name in sys.modules); "
            "assert 'rsi.execution.sandbox_worker' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_new_and_legacy_command_entries_work(self) -> None:
        for module in ("rsi", "rsi.commands.cli", "rsi.cli"):
            with self.subTest(module=module):
                result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("selection-demo", result.stdout)
                self.assertIn("run-comparison", result.stdout)

    def test_manifest_preserves_relative_paths_and_detects_nested_changes(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            for name in ("one", "two", "__pycache__"):
                (root / name).mkdir()
                (root / name / "same.py").write_text("value = 1\n", encoding="utf-8")
            before = source_manifest(root)
            self.assertEqual([entry["name"] for entry in before], ["one/same.py", "two/same.py"])
            (root / "two/same.py").write_text("value = 2\n", encoding="utf-8")
            after = source_manifest(root)
            self.assertEqual(before[0], after[0])
            self.assertNotEqual(digest(before), digest(after))

    def test_nested_source_change_rejects_old_training_checkpoint(self) -> None:
        context = self.context()
        batch = build_batch(context, FixedSelector())
        request = TrainingRequest("run", "attempt", context.contract_hash, context.contract.model,
                                  batch, context.contract.training)
        selected = [next(record for record in context.records if record.data_id == item.data_id)
                    for item in batch.items]
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            source = root / "source/nested"
            source.mkdir(parents=True)
            (source / "module.py").write_text("value = 1\n", encoding="utf-8")
            output = root / "attempt"
            output.mkdir()
            with patch("rsi.storage.provenance.PACKAGE_ROOT", root / "source"):
                identity = request_identity(request, selected, "cpu")
                journal = StepJournal(output, identity)
                journal.begin(1)
                artifact = journal.publish(1, lambda path: path.write_bytes(b"fixture state"))
                journal.validate_resume(artifact)
                (source / "module.py").write_text("value = 2\n", encoding="utf-8")
                changed = request_identity(request, selected, "cpu")
                self.assertNotEqual(identity, changed)
                with self.assertRaisesRegex(ValidationError, "冻结输入"):
                    StepJournal(output, changed).validate_resume(artifact)

    def test_real_manifest_covers_the_package_and_compatibility_code(self) -> None:
        names = {entry["name"] for entry in source_manifest()}
        self.assertIn("_compat.py", names)
        self.assertIn("models/hf_backend.py", names)
        self.assertIn("curation/dispatch.py", names)
        self.assertIn("execution/sandbox_worker.py", names)
        self.assertEqual(names, {p.relative_to(PACKAGE_ROOT).as_posix() for p in PACKAGE_ROOT.rglob("*.py")
                                 if "__pycache__" not in p.relative_to(PACKAGE_ROOT).parts})

    def test_sandbox_worker_remains_an_external_file(self) -> None:
        from rsi.execution.sandbox import BubblewrapExecutor, SandboxLimits

        with patch("rsi.execution.sandbox.platform.system", return_value="Linux"), \
             patch("rsi.execution.sandbox.platform.machine", return_value="x86_64"), \
             patch("rsi.execution.sandbox.shutil.which", return_value="/usr/bin/bwrap"), \
             patch("rsi.execution.sandbox.Path.is_file", return_value=True):
            executor = BubblewrapExecutor(SandboxLimits(5, 2, 256, 65536))
        expected = PACKAGE_ROOT / "execution/sandbox_worker.py"
        self.assertEqual(executor.worker, expected)
        self.assertTrue(expected.is_file())
        self.assertIn(str(expected), executor.command)
