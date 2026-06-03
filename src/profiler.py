"""
Per-Stage Latency Profiler
--------------------------
Lightweight context-manager profiler. Wraps pipeline stages and prints
a summary table at the end of each run. Useful for M3 MPS benchmarking.
"""

import time
from collections import OrderedDict


class Profiler:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._timings: OrderedDict = OrderedDict()

    class _Stage:
        def __init__(self, profiler, name):
            self.profiler = profiler
            self.name = name
            self._start = None

        def __enter__(self):
            self._start = time.perf_counter()
            return self

        def __exit__(self, *args):
            elapsed_ms = (time.perf_counter() - self._start) * 1000
            self.profiler._timings[self.name] = elapsed_ms

    def measure(self, stage_name: str) -> "_Stage":
        """Usage: `with profiler.measure('stage_name'):` """
        if not self.enabled:
            return self._noop_ctx()
        return self._Stage(self, stage_name)

    class _NoopCtx:
        def __enter__(self): return self
        def __exit__(self, *args): pass

    def _noop_ctx(self):
        return self._NoopCtx()

    def print_summary(self):
        if not self._timings:
            return

        print("\n╔══════════════════════════════════════╗")
        print("║       [Profiler] Stage Latencies      ║")
        print("╠══════════════════════════════════════╣")

        for stage, ms in self._timings.items():
            label = stage.ljust(26)
            bar = "█" * max(1, int(ms / 50))  # 1 block per 50ms
            print(f"║  {label}: {ms:7.1f}ms  {bar}")

        total = sum(self._timings.values())
        print("╠══════════════════════════════════════╣")
        print(f"║  {'TOTAL'.ljust(26)}: {total:7.1f}ms")
        print("╚══════════════════════════════════════╝\n")

    def get(self, stage_name: str) -> float:
        """Returns latency in ms for a specific stage."""
        return self._timings.get(stage_name, 0.0)
