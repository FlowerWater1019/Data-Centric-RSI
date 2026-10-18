"""默认测试检查边界。Linux 集成测试需显式开启，不依赖网络或 GPU。"""

from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsi.sandbox import BubblewrapExecutor, SandboxLimits, SandboxUnavailable, sandbox_command
from rsi.validation import ValidationError, read_json
from tests.helpers import ROOT


class SandboxConfigTests(unittest.TestCase):
    def test_invalid_resource_bounds_rejected(self) -> None:
        for key, value in [('memory_mb', 4096), ('cpu_seconds', 0), ('wall_seconds', 60), ('output_bytes', 0)]:
            raw = read_json(ROOT/'configs/sandbox.json')
            raw[key] = value
            with self.assertRaises(ValidationError):
                SandboxLimits.parse(raw)

    def test_unsupported_host_cannot_fall_back(self) -> None:
        with patch('rsi.sandbox.platform.system', return_value='Windows'):
            with self.assertRaises(SandboxUnavailable):
                sandbox_command(Path('worker.py'))

    def test_command_does_not_mount_home_or_project(self) -> None:
        with patch('rsi.sandbox.platform.system',return_value='Linux'), \
             patch('rsi.sandbox.platform.machine',return_value='x86_64'), \
             patch('rsi.sandbox.shutil.which',return_value='/usr/bin/bwrap'), \
             patch('rsi.sandbox.Path.is_file',return_value=True):
            command=sandbox_command(Path('/project/rsi/sandbox_worker.py'))
        self.assertIn('--unshare-net',command)
        self.assertIn('--unshare-pid',command)
        self.assertIn('--clearenv',command)
        self.assertNotIn('--bind',command)
        self.assertNotIn('/home',command)
        self.assertNotIn('/data',command)
        self.assertEqual(command[-3:],['-I','-B','/runner.py'])


@unittest.skipUnless(os.environ.get('RSI_RUN_SANDBOX_TESTS')=='1', '未启用 Linux 隔离集成测试。')
class SandboxIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.limits=SandboxLimits.parse(read_json(ROOT/'configs/sandbox.json'))
        self.executor=BubblewrapExecutor(self.limits)

    def test_preflight_and_standard_library(self) -> None:
        result=self.executor.preflight()
        self.assertEqual(len(result['probes']),6)
        code='import collections\ndef count(xs):\n    return dict(collections.Counter(xs))'
        self.assertEqual(self.executor.run(code,"assert count([1,1,2]) == {1:2,2:1}").status,'passed')

    def test_private_host_file_is_not_visible(self) -> None:
        with tempfile.NamedTemporaryFile(dir=ROOT) as handle:
            code=f'import os\nexists=os.path.exists({str(Path(handle.name).resolve())!r})'
            self.assertEqual(self.executor.run(code,'assert not exists').status,'passed')

    def test_wall_timeout_kills_worker(self) -> None:
        executor=BubblewrapExecutor(replace(self.limits, wall_seconds=2))
        result=executor.run('import time\ntime.sleep(30)','assert True')
        self.assertEqual(result.status,'timeout')
        self.assertLess(result.elapsed_seconds,5)

    def test_output_flood_is_bounded(self) -> None:
        result=self.executor.run("while True:\n    print('x'*4096, flush=True)",'assert True')
        self.assertEqual(result.status,'output_limit')
        self.assertLessEqual(len(result.output.encode()),self.limits.output_bytes)

    def test_memory_limit(self) -> None:
        result=self.executor.run('value=bytearray(1024*1024*512)','assert True')
        self.assertEqual(result.status,'memory_limit')

    def test_early_exit_cannot_report_success(self) -> None:
        result=self.executor.run('import os\nos._exit(0)','assert False')
        self.assertEqual(result.status,'runtime_error')

    def test_exec_and_threads_are_denied(self) -> None:
        code="import os\nblocked=False\ntry: os.execv('/usr/bin/true',['true'])\nexcept PermissionError: blocked=True"
        self.assertEqual(self.executor.run(code,'assert blocked').status,'passed')
        code="import threading\nblocked=False\ntry: threading.Thread(target=lambda:None).start()\nexcept RuntimeError: blocked=True"
        self.assertEqual(self.executor.run(code,'assert blocked').status,'passed')

    def test_assertion_failure_counts_as_failure(self) -> None:
        self.assertEqual(self.executor.run('value=1','assert value==2').status,'failed')
