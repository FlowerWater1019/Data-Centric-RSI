"""为本机运行目录提供单写入者锁。进程退出后由操作系统释放锁。"""

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Iterator

from rsi.configuration.validation import fail


@contextmanager
def run_lease(root: Path) -> Iterator[None]:
    """同一目录不能由两个恢复进程同时修改。不支持网络文件系统。"""
    with (root / ".writer.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fail("run", "运行目录已有写入者，或文件系统不支持锁。")
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
