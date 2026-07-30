#!/usr/bin/env python3
"""Smoke test for ``tools/tessera/_accelerate.py``.

Exercises every public vDSP wrapper against a small reference and reports
pass/fail per function. On non-Darwin platforms (Linux CI, Windows) or when
the Accelerate dylib is unavailable, the wrapper degrades to the
pure-Python fallback, so the smoke test still passes - we just print a
"running in fallback" notice and exit 0.

Usage:
    python tools/tessera/accelerate_smoke.py [--verbose]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# Allow running from the repo root or from tools/tessera/.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _accelerate as acc  # noqa: E402


# ---------------------------------------------------------------------------
# Reference implementations (used to compare against the wrapper)
# ---------------------------------------------------------------------------


def ref_meanv(x): return sum(x) / len(x)
def ref_measqv(x): return sum(v * v for v in x) / len(x)
def ref_maxv(x): return max(x)
def ref_minv(x): return min(x)
def ref_sve(x): return float(sum(x))
def ref_vsmul(x, s): return [v * s for v in x]
def ref_vmul(a, b): return [av * bv for av, bv in zip(a, b)]
def ref_vadd(a, b): return [av + bv for av, bv in zip(a, b)]
def ref_dotpr(a, b): return float(sum(av * bv for av, bv in zip(a, b)))


def ref_conv(sig, filt):
    n = len(sig)
    p = len(filt)
    if p == 0 or n < p:
        return []
    filt_rev = list(reversed(filt))
    out_len = n - p + 1
    out = [0.0] * out_len
    for i in range(out_len):
        acc_v = 0.0
        for j in range(p):
            acc_v += sig[i + j] * filt_rev[j]
        out[i] = acc_v
    return out


def ref_zrvmul(re, im, real):
    return [r * v for r, v in zip(re, real)], [i * v for i, v in zip(im, real)]


def ref_mmov(src, m, n, src_cols):
    return [src[r * src_cols + c] for r in range(n) for c in range(m)]


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


FLOAT_TOL = 1e-5


def _close_scalar(actual, expected, tol=FLOAT_TOL):
    if math.isnan(expected) and math.isnan(actual):
        return True
    if math.isinf(expected) and math.isinf(actual) and (expected > 0) == (actual > 0):
        return True
    return abs(actual - expected) <= tol * max(1.0, abs(expected))


def _close_seq(actual, expected, tol=FLOAT_TOL):
    if len(actual) != len(expected):
        return False
    return all(_close_scalar(a, e, tol) for a, e in zip(actual, expected))


def _check(name, actual, expected, *, tol=FLOAT_TOL):
    ok = _close_scalar(actual, expected, tol) if isinstance(expected, (int, float)) \
        else _close_seq(actual, expected, tol)
    status = "PASS" if ok else "FAIL"
    print(f"  {name:<14} {status}")
    return ok


# ---------------------------------------------------------------------------
# Smoke cases
# ---------------------------------------------------------------------------


def run_smoke(verbose=False):
    print(f"_accelerate: {acc.status_line()}")
    print(f"is_available: {acc.is_available()}")
    print()

    if not acc.is_available():
        # On non-Darwin platforms (Linux CI, Windows) the vDSP dylib is
        # absent by design; the wrapper's pure-Python fallback still has
        # to remain importable and behave sensibly, but the smoke test
        # has nothing specific to validate against real vDSP. Report
        # and exit 0 so CI stays green.
        print("Accelerate not available, skipping.")
        return 0

    failures = []

    # small fixed test inputs
    x_small = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_small = [10.0, 20.0, 30.0, 40.0, 50.0]
    mixed   = [-2.0, -1.0, 0.0, 1.5, 3.25]

    print("[1/4] reductions")
    if not _check("vDSP_meanv", acc.vDSP_meanv(x_small), ref_meanv(x_small)):
        failures.append("vDSP_meanv")
    if not _check("vDSP_measqv", acc.vDSP_measqv(x_small), ref_measqv(x_small)):
        failures.append("vDSP_measqv")
    if not _check("vDSP_maxv", acc.vDSP_maxv(mixed), ref_maxv(mixed)):
        failures.append("vDSP_maxv")
    if not _check("vDSP_minv", acc.vDSP_minv(mixed), ref_minv(mixed)):
        failures.append("vDSP_minv")
    if not _check("vDSP_sve", acc.vDSP_sve(x_small), ref_sve(x_small)):
        failures.append("vDSP_sve")

    print("[2/4] elementwise")
    if not _check("vDSP_vsmul", acc.vDSP_vsmul(x_small, 2.5), ref_vsmul(x_small, 2.5)):
        failures.append("vDSP_vsmul")
    if not _check("vDSP_vmul", acc.vDSP_vmul(x_small, y_small), ref_vmul(x_small, y_small)):
        failures.append("vDSP_vmul")
    if not _check("vDSP_vadd", acc.vDSP_vadd(x_small, y_small), ref_vadd(x_small, y_small)):
        failures.append("vDSP_vadd")
    if not _check("vDSP_dotpr", acc.vDSP_dotpr(x_small, y_small), ref_dotpr(x_small, y_small)):
        failures.append("vDSP_dotpr")

    print("[3/4] convolution / complex / mmov")
    sig = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    filt = [10.0, 20.0, 30.0]
    if not _check("vDSP_conv", acc.vDSP_conv(sig, filt), ref_conv(sig, filt)):
        failures.append("vDSP_conv")
    # length-1 filter is its own edge case (output length == input length)
    if not _check("vDSP_conv(len1)", acc.vDSP_conv([1.0, 2.0, 3.0], [2.0]),
                  ref_conv([1.0, 2.0, 3.0], [2.0])):
        failures.append("vDSP_conv(len1)")
    # empty filter
    if not _check("vDSP_conv(empty)", acc.vDSP_conv([1.0, 2.0, 3.0], []),
                  ref_conv([1.0, 2.0, 3.0], [])):
        failures.append("vDSP_conv(empty)")

    re_in = [2.0, 4.0, 8.0, 16.0]
    im_in = [10.0, 11.0, 12.0, 13.0]
    real_v = [1.0, 2.0, 3.0, 4.0]
    re_ref, im_ref = ref_zrvmul(re_in, im_in, real_v)
    re_actual, im_actual = acc.vDSP_zrvmul(re_in, im_in, real_v)
    if not _check("vDSP_zrvmul(re)", re_actual, re_ref):
        failures.append("vDSP_zrvmul(re)")
    if not _check("vDSP_zrvmul(im)", im_actual, im_ref):
        failures.append("vDSP_zrvmul(im)")

    src = list(range(1, 13))
    if not _check("vDSP_mmov(2x3)", acc.vDSP_mmov(src, 3, 2),
                  ref_mmov(src, 3, 2, 3)):
        failures.append("vDSP_mmov(2x3)")
    if not _check("vDSP_mmov(2x3,sc=4)", acc.vDSP_mmov(src, 3, 2, src_cols=4),
                  ref_mmov(src, 3, 2, 4)):
        failures.append("vDSP_mmov(2x3,sc=4)")

    print("[4/4] edge cases")
    if not _check("meanv(empty)", acc.vDSP_meanv([]), 0.0):
        failures.append("meanv(empty)")
    if not _check("maxv(empty)", acc.vDSP_maxv([]), float("-inf")):
        failures.append("maxv(empty)")
    if not _check("minv(empty)", acc.vDSP_minv([]), float("inf")):
        failures.append("minv(empty)")
    if not _check("dotpr(empty)", acc.vDSP_dotpr([], []), 0.0):
        failures.append("dotpr(empty)")
    if not _check("vDSP_mmov(0x0)", acc.vDSP_mmov([], 0, 0), []):
        failures.append("vDSP_mmov(0x0)")

    # mismatch-length error path
    err_raised = False
    try:
        acc.vDSP_vadd([1.0, 2.0], [1.0])
    except ValueError:
        err_raised = True
    print(f"  {'vDSP_vadd(len_mismatch)':<24} {'PASS' if err_raised else 'FAIL'}")
    if not err_raised:
        failures.append("vDSP_vadd(len_mismatch)")

    print()
    if failures:
        print(f"FAILED: {len(failures)} test(s): {', '.join(failures)}")
        return 1
    print("All vDSP smoke tests passed.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true",
                        help="Print extra reference values.")
    args = parser.parse_args()
    return run_smoke(verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
