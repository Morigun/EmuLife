from __future__ import annotations

import os
import sys


def _half_core_count() -> int:
    total = os.cpu_count() or 2
    return max(1, total // 2)


def limit_numba_threads() -> int:
    n = _half_core_count()
    os.environ["NUMBA_NUM_THREADS"] = str(n)
    return n


def set_affinity(n_cores: int | None = None) -> int:
    if n_cores is None:
        n_cores = _half_core_count()

    if sys.platform == "win32":
        import ctypes
        mask = (1 << n_cores) - 1
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetProcessAffinityMask(handle, mask)
    else:
        try:
            os.sched_setaffinity(0, set(range(n_cores)))
        except (AttributeError, OSError):
            pass

    return n_cores


def set_affinity_for_process(pid: int, n_cores: int) -> None:
    if sys.platform == "win32":
        import ctypes
        mask = ((1 << n_cores) - 1)
        total = os.cpu_count() or 2
        half = total // 2
        shifted_mask = mask << half
        handle = ctypes.windll.kernel32.OpenProcess(0x0200, False, pid)
        if handle:
            ctypes.windll.kernel32.SetProcessAffinityMask(handle, shifted_mask)
            ctypes.windll.kernel32.CloseHandle(handle)
