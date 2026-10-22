"""Linux Python 隔离后端。依赖不可用时拒绝执行，禁止退回宿主进程。"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import selectors
import shutil
import signal
import subprocess
import time
from typing import Any
import uuid

from rsi.storage.artifacts import file_sha256
from rsi.configuration.validation import choice, fail, integer, obj, to_json_value


class SandboxUnavailable(RuntimeError):
    """隔离机制未建立。该错误不能计为模型任务失败。"""


@dataclass(frozen=True)
class SandboxLimits:
    wall_seconds: int
    cpu_seconds: int
    memory_mb: int
    output_bytes: int

    @classmethod
    def parse(cls, value: Any) -> "SandboxLimits":
        raw = obj(value, "sandbox", {"schema_version", "backend", "wall_seconds", "cpu_seconds", "memory_mb", "output_bytes"})
        choice(raw["schema_version"], "sandbox.schema_version", {"sandbox-1"})
        choice(raw["backend"], "sandbox.backend", {"bubblewrap_seccomp_v1"})
        bounds = {"wall_seconds": (1, 30), "cpu_seconds": (1, 10), "memory_mb": (64, 1024), "output_bytes": (1024, 262144)}
        for key, (low, high) in bounds.items():
            if integer(raw[key], "sandbox." + key, low) > high:
                fail("sandbox." + key, f"不能超过 {high}。")
        if raw["cpu_seconds"] > raw["wall_seconds"]:
            fail("sandbox", "CPU 时间不能超过墙钟时间。")
        return cls(*(raw[key] for key in bounds))


@dataclass(frozen=True)
class SandboxResult:
    status: str
    detail: str | None
    return_code: int
    elapsed_seconds: float
    output: str


def sandbox_command(worker: Path) -> list[str]:
    """只挂载系统运行库与启动器。主目录、工程目录和凭据均不挂载。"""
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SandboxUnavailable("当前隔离后端要求 Linux x86_64。")
    binary = shutil.which("bwrap")
    if binary is None or not Path("/usr/bin/python3").is_file():
        raise SandboxUnavailable("需要已有的 bwrap 和 /usr/bin/python3。")
    return [binary, "--unshare-user", "--unshare-ipc", "--unshare-pid", "--unshare-net", "--unshare-uts",
            "--die-with-parent", "--new-session", "--cap-drop", "ALL",
            "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
            "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
            "--proc", "/proc", "--dev", "/dev", "--dir", "/work",
            "--ro-bind", str(worker), "/runner.py", "--remount-ro", "/", "--chdir", "/work",
            "--clearenv", "--setenv", "PATH", "/usr/bin", "--setenv", "LANG", "C.UTF-8",
            "/usr/bin/python3", "-I", "-B", "/runner.py"]


class BubblewrapExecutor:
    """每个测试启动独立沙箱。只支持 Python 标准库计算任务。"""

    def __init__(self, limits: SandboxLimits) -> None:
        self.limits = limits
        self.worker = Path(__file__).with_name("sandbox_worker.py").resolve()
        self.command = sandbox_command(self.worker)

    def run(self, code: str, test: str, setup: str = "") -> SandboxResult:
        """执行一次测试。墙钟和输出限制由沙箱外的父进程执行。"""
        if any(not isinstance(value, str) for value in (code, test, setup)):
            fail("sandbox", "代码、测试和初始化内容必须是字符串。")
        nonce = uuid.uuid4().hex
        payload = json.dumps({"code": code, "test": test, "setup": setup, "nonce": nonce,
                              "limits": to_json_value(self.limits)}).encode()
        if len(payload) > 262144:
            fail("sandbox", "单次输入超过 256 KiB。")
        started = time.monotonic()
        output = bytearray()
        forced = None
        with subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              start_new_session=True, env={"PATH": "/usr/bin:/bin"}) as process:
            try:
                # 输入也进入非阻塞循环，避免启动失败或不读 stdin 时卡住父进程。
                os.set_blocking(process.stdin.fileno(), False)
                os.set_blocking(process.stdout.fileno(), False)
                remaining = memoryview(payload)
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdin, selectors.EVENT_WRITE)
                    selector.register(process.stdout, selectors.EVENT_READ)
                    while selector.get_map():
                        if time.monotonic() - started > self.limits.wall_seconds:
                            forced = "timeout"
                            break
                        for key, _ in selector.select(0.05):
                            if key.fileobj is process.stdin:
                                try:
                                    sent = os.write(process.stdin.fileno(), remaining)
                                    remaining = remaining[sent:]
                                except BrokenPipeError:
                                    remaining = remaining[:0]
                                if not remaining:
                                    selector.unregister(process.stdin)
                                    process.stdin.close()
                            else:
                                block = os.read(process.stdout.fileno(), 8192)
                                if not block:
                                    selector.unregister(process.stdout)
                                else:
                                    output.extend(block)
                                    if len(output) > self.limits.output_bytes:
                                        forced = "output_limit"
                                        break
                        if forced:
                            break
                if forced:
                    self._kill(process)
                elif process.poll() is None:
                    try:
                        process.wait(timeout=max(0.1, self.limits.wall_seconds - (time.monotonic() - started)))
                    except subprocess.TimeoutExpired:
                        forced = "timeout"
                        self._kill(process)
                return_code = process.wait(timeout=5)
            except BaseException:
                self._kill(process)
                process.wait(timeout=5)
                raise
        text = output[:self.limits.output_bytes].decode("utf-8", errors="replace")
        marker = "RSI_" + nonce + "="
        messages = []
        for line in text.splitlines():
            if line.startswith(marker):
                try:
                    messages.append(json.loads(line[len(marker):]))
                except json.JSONDecodeError:
                    continue
        if not any(isinstance(message, dict) and message.get("stage") == "ready" for message in messages):
            raise SandboxUnavailable("隔离启动或资源限制失败：" + text[-1500:])
        result = next((m for m in reversed(messages) if isinstance(m, dict) and m.get("stage") == "result"), None)
        if forced:
            status, detail = forced, "parent_enforced_limit"
        elif return_code != 0 or result is None:
            status, detail = "runtime_error", "worker_exited_without_success"
        elif result.get("status") not in {"passed", "failed", "memory_limit", "runtime_error"}:
            raise SandboxUnavailable("隔离结果格式无效。")
        else:
            status, detail = result["status"], result.get("detail")
        return SandboxResult(status, detail, return_code, time.monotonic() - started, text)

    @staticmethod
    def _kill(process: subprocess.Popen) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def preflight(self) -> dict[str, Any]:
        """执行可信探针。任一保护未生效时，不启动候选代码。"""
        probes = {
            "stdlib": ("import math\nvalue = math.factorial(6)", "assert value == 720"),
            "filesystem_hidden": ("import os", "assert not os.path.exists('/data') and not os.path.exists('/home')"),
            "network_denied": ("import socket\nblocked=False\ntry: socket.socket()\nexcept PermissionError: blocked=True", "assert blocked"),
            "process_denied": ("import os\nblocked=False\ntry: os.fork()\nexcept PermissionError: blocked=True", "assert blocked"),
            "write_denied": ("blocked=False\ntry: open('/work/probe', 'w')\nexcept OSError: blocked=True", "assert blocked"),
            "limit_raise_denied": ("import resource\nblocked=False\ntry: resource.setrlimit(resource.RLIMIT_CPU,(100,100))\nexcept (ValueError,OSError): blocked=True", "assert blocked"),
        }
        evidence = {}
        for name, (code, test) in probes.items():
            result = self.run(code, test)
            if result.status != "passed":
                raise SandboxUnavailable("隔离探针失败：" + name + " " + result.output[-500:])
            evidence[name] = to_json_value(result)
        return {"backend": "bubblewrap_seccomp_v1", "limits": to_json_value(self.limits),
                "worker_sha256": file_sha256(self.worker), "probes": evidence,
                "bwrap_version": subprocess.check_output([self.command[0], "--version"], text=True).strip(),
                "python_version": subprocess.check_output(["/usr/bin/python3", "--version"], text=True).strip(),
                "kernel": platform.release()}
