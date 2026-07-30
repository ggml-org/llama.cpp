#!/usr/bin/env python3
"""Micro-benchmark for ``tools/tessera/_accelerate.py``.

Compares the public vDSP wrapper against a hand-rolled pure-Python
implementation on the same inputs. Two modes:

* ``--mode buffer`` (default)    pre-builds ``array.array('f')`` buffers
  once and reuses them across all timed iterations. This is closer to
  how the quantizer's inner loops actually run (the data is already in
  a contiguous buffer). The vDSP advantage is much larger here.

* ``--mode list``    feeds Python lists through the public API. This
  is the smoke-test mode and includes the list-to-ctypes marshaling
  cost on every call. Real quant code would not do this - it would
  hand numpy arrays to the C++ shim - so the numbers are a conservative
  lower bound on the vDSP advantage.

The benchmark prints a fixed-width table and exits 0 on completion. It
does not enforce a pass/fail threshold; use the smoke test for that.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from array import array
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _accelerate as acc  # noqa: E402


# ---------------------------------------------------------------------------
# Pure-Python reference implementations
# ---------------------------------------------------------------------------


def py_meanv(x): return sum(x) / len(x)
def py_measqv(x): return sum(v * v for v in x) / len(x)
def py_maxv(x): return max(x)
def py_minv(x): return min(x)
def py_sve(x): return float(sum(x))
def py_vsmul(x, s): return [v * s for v in x]
def py_vadd(a, b): return [av + bv for av, bv in zip(a, b)]
def py_vmul(a, b): return [av * bv for av, bv in zip(a, b)]
def py_dotpr(a, b): return float(sum(av * bv for av, bv in zip(a, b)))


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_ns(fn, repeat=5, min_iters=3, max_iters=10000):
    """Run ``fn`` enough times to get a stable per-call measurement.

    Uses an adaptive scheme: starts with ``min_iters``, doubles up to
    ``max_iters`` until the per-call time exceeds 50 ms, then takes the
    median of ``repeat`` measurements.
    """
    timings = []
    for _ in range(repeat):
        iters = min_iters
        # Warm-up
        for _ in range(2):
            fn()
        # Coarse-grained: ensure each measurement covers >= 5 ms
        while iters <= max_iters:
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            elapsed_s = time.perf_counter() - t0
            per_call_ns = elapsed_s / iters * 1e9
            if elapsed_s >= 0.005 or iters == max_iters:
                timings.append(per_call_ns)
                break
            iters *= 2
    return statistics.median(timings)


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------


def _make_vectors(n, seed=1234):
    """Deterministic pair of float32 vectors of length n."""
    a = [0.0] * n
    b = [0.0] * n
    # Simple LCG so the bench is reproducible without importing random.
    state = seed
    for i in range(n):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        a[i] = (state / 0x7FFFFFFF) * 2.0 - 1.0
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        b[i] = (state / 0x7FFFFFFF) * 2.0 - 1.0
    return a, b


# ---------------------------------------------------------------------------
# The benchmark harness
# ---------------------------------------------------------------------------


def _row(name, n, py_ns, vdsp_ns, unit="ns"):
    speedup = py_ns / vdsp_ns if vdsp_ns > 0 else float("inf")
    return (
        f"  {name:<22} n={n:>9,}  "
        f"py={py_ns/1000:>10,.2f} us  "
        f"vDSP={vdsp_ns/1000:>10,.2f} us  "
        f"speedup={speedup:>7,.2f}x"
    )


def _bench_reductions(n, mode, a):
    rows = []
    a_buf = array("f", a) if mode == "buffer" else a

    def run_py_mean(): py_meanv(a)
    def run_py_measq(): py_measqv(a)
    def run_py_max(): py_maxv(a)
    def run_py_min(): py_minv(a)
    def run_py_sve(): py_sve(a)
    def run_vdsp_mean(): acc.vDSP_meanv(a_buf)
    def run_vdsp_measq(): acc.vDSP_measqv(a_buf)
    def run_vdsp_max(): acc.vDSP_maxv(a_buf)
    def run_vdsp_min(): acc.vDSP_minv(a_buf)
    def run_vdsp_sve(): acc.vDSP_sve(a_buf)

    rows.append(("vDSP_meanv", n,
                 _time_ns(run_py_mean), _time_ns(run_vdsp_mean)))
    rows.append(("vDSP_measqv", n,
                 _time_ns(run_py_measq), _time_ns(run_vdsp_measq)))
    rows.append(("vDSP_maxv", n,
                 _time_ns(run_py_max), _time_ns(run_vdsp_max)))
    rows.append(("vDSP_minv", n,
                 _time_ns(run_py_min), _time_ns(run_vdsp_min)))
    rows.append(("vDSP_sve", n,
                 _time_ns(run_py_sve), _time_ns(run_vdsp_sve)))
    return rows


def _bench_elementwise(n, mode, a, b):
    rows = []
    a_buf = array("f", a) if mode == "buffer" else a
    b_buf = array("f", b) if mode == "buffer" else b

    def run_py_vsmul(): py_vsmul(a, 0.5)
    def run_py_vadd(): py_vadd(a, b)
    def run_py_vmul(): py_vmul(a, b)
    def run_py_dot(): py_dotpr(a, b)
    # Elementwise ops return a fresh list. We discard the result inside
    # the timed closure so the list allocation is paid, matching the pure-
    # Python reference which also produces a fresh list each call.
    def run_vdsp_vsmul(): acc.vDSP_vsmul(a_buf, 0.5)
    def run_vdsp_vadd(): acc.vDSP_vadd(a_buf, b_buf)
    def run_vdsp_vmul(): acc.vDSP_vmul(a_buf, b_buf)
    def run_vdsp_dot(): acc.vDSP_dotpr(a_buf, b_buf)

    rows.append(("vDSP_vsmul", n,
                 _time_ns(run_py_vsmul), _time_ns(run_vdsp_vsmul)))
    rows.append(("vDSP_vadd", n,
                 _time_ns(run_py_vadd), _time_ns(run_vdsp_vadd)))
    rows.append(("vDSP_vmul", n,
                 _time_ns(run_py_vmul), _time_ns(run_vdsp_vmul)))
    rows.append(("vDSP_dotpr", n,
                 _time_ns(run_py_dot), _time_ns(run_vdsp_dot)))
    return rows


def run_bench(mode: str, sizes: list[int]):
    print(f"_accelerate: {acc.status_line()}")
    print(f"mode: {mode}")
    if not acc.is_available():
        print()
        print("vDSP unavailable - benchmark will run the pure-Python fallback")
        print("on both sides. Numbers will be roughly equal.")
    print()

    for n in sizes:
        a, b = _make_vectors(n)
        print(f"--- workload n = {n:,} ---")
        for name, sz, py_ns, vdsp_ns in _bench_reductions(n, mode, a):
            print(_row(name, sz, py_ns, vdsp_ns))
        for name, sz, py_ns, vdsp_ns in _bench_elementwise(n, mode, a, b):
            print(_row(name, sz, py_ns, vdsp_ns))
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("list", "buffer"), default="buffer",
                        help="How to feed data into the wrapper. 'buffer' "
                             "pre-builds array.array and reuses it; 'list' "
                             "marshals a fresh list per call.")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[1024, 16_384, 262_144, 1_048_576],
                        help="Workload sizes (number of elements).")
    args = parser.parse_args()
    run_bench(args.mode, args.sizes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
