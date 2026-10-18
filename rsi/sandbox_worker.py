"""隔离环境内的可信启动器。此文件只能由 Bubblewrap 后端启动。"""

import ctypes
import errno
import json
import os
import resource
import sys


def install_filter() -> None:
    """仅允许计算、只读文件访问和现有输出流。禁止新进程与网络。"""
    library = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int

    class Comparison(ctypes.Structure):
        _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int),
                    ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]

    library.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
                                                ctypes.c_uint, ctypes.POINTER(Comparison)]
    context = library.seccomp_init(0x00050000 | errno.EPERM)
    if not context:
        raise RuntimeError("seccomp_init failed")
    try:
        names = (
            "read write readv writev close fstat newfstatat stat lstat statx lseek pread64 "
            "access faccessat faccessat2 readlink readlinkat getdents64 getcwd "
            "mmap mprotect munmap mremap madvise brk rt_sigaction rt_sigprocmask rt_sigreturn "
            "sigaltstack futex clock_gettime clock_nanosleep nanosleep gettimeofday time select pselect6 poll ppoll "
            "getpid getppid gettid getuid geteuid getgid getegid getgroups getrandom "
            "getrusage times uname sysinfo sched_getaffinity sched_yield "
            "exit exit_group restart_syscall"
        ).split()
        for name in names:
            syscall = library.seccomp_syscall_resolve_name(name.encode())
            if syscall >= 0 and library.seccomp_rule_add_array(context, 0x7FFF0000, syscall, 0, None) != 0:
                raise RuntimeError("cannot allow syscall: " + name)
        # MASKED_EQ 检查打开标志。没有任何创建、截断、追加或写入权限。
        forbidden = (os.O_ACCMODE | os.O_CREAT | os.O_TRUNC | os.O_APPEND
                     | (getattr(os, "O_TMPFILE", 0) & ~os.O_DIRECTORY))
        for name, argument in (("open", 1), ("openat", 2)):
            syscall = library.seccomp_syscall_resolve_name(name.encode())
            if syscall < 0:
                continue
            comparison = Comparison(argument, 7, forbidden, 0)
            if library.seccomp_rule_add_array(context, 0x7FFF0000, syscall, 1, ctypes.byref(comparison)) != 0:
                raise RuntimeError("cannot restrict file opens")
        if library.seccomp_load(context) != 0:
            raise RuntimeError("seccomp_load failed")
    finally:
        library.seccomp_release(context)


def main() -> None:
    payload = json.loads(sys.stdin.buffer.read(262145))
    limits, nonce = payload["limits"], payload["nonce"]
    resource.setrlimit(resource.RLIMIT_AS, (limits["memory_mb"] * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu_seconds"], limits["cpu_seconds"] + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
    install_filter()
    emit = lambda value: print("\nRSI_" + nonce + "=" + json.dumps(value), flush=True)
    emit({"stage": "ready"})
    namespace = {"__name__": "__candidate__"}
    try:
        exec(compile(payload["setup"], "<setup>", "exec"), namespace)
        exec(compile(payload["code"], "<candidate>", "exec"), namespace)
        exec(compile(payload["test"], "<test>", "exec"), namespace)
    except AssertionError:
        emit({"stage": "result", "status": "failed", "detail": "assertion_failed"})
    except MemoryError:
        emit({"stage": "result", "status": "memory_limit", "detail": "MemoryError"})
    except BaseException as exc:
        emit({"stage": "result", "status": "runtime_error", "detail": type(exc).__name__})
    else:
        emit({"stage": "result", "status": "passed", "detail": None})


if __name__ == "__main__":
    main()
