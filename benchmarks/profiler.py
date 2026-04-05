"""Resource profiler: wall time + peak memory via tracemalloc.

Note: tracemalloc tracks Python-level allocations only and underestimates
memory by ~20-40% for numpy-heavy code (C-extensions allocate outside Python
heap). Results should be treated as a lower bound.
"""

import tracemalloc
import time


class ResourceProfiler:
    """Context manager that measures elapsed wall time and peak memory usage.

    Usage::

        with ResourceProfiler() as prof:
            do_work()
        print(f"{prof.elapsed_s:.2f}s, {prof.peak_mem_mb:.1f} MB")
    """

    elapsed_s: float = 0.0
    peak_mem_mb: float = 0.0

    def __enter__(self) -> 'ResourceProfiler':
        tracemalloc.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed_s = time.perf_counter() - self._t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_mem_mb = peak / (1024 ** 2)
