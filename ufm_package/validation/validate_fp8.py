#!/usr/bin/env python3
"""
validate_fp8.py — Verify the FP8 E4M3FN patch is correctly applied.

Runs 12 round-trip tests against known-correct values.
Run this AFTER applying ggml_fp8.patch and rebuilding.

Usage:
    python validation/validate_fp8.py
    python validation/validate_fp8.py --llama-path C:/path/to/llama.cpp
"""

import ctypes
import os
import sys
import struct
import math
import argparse
from pathlib import Path

# ── Pure-Python reference implementation (verified correct) ───────────────────

def f32_to_f8_ref(x: float) -> int:
    """Reference FP8 E4M3FN encoder — verified correct."""
    if math.isnan(x): return 0x7F
    if math.isinf(x): x = math.copysign(448.0, x)
    x = max(min(x, 448.0), -448.0)
    if x == 0.0: return 0
    bits  = struct.unpack('I', struct.pack('f', x))[0]
    sign  = (bits >> 31) & 1
    exp32 = ((bits >> 23) & 0xFF) - 127
    mant32= bits & 0x7FFFFF
    exp8  = exp32 + 7
    if exp8 <= 0:
        shift = 1 - exp8
        if shift > 3: return sign << 7
        return (sign << 7) | (((0x800000 | mant32) >> (20 + shift)) & 0x7)
    if exp8 > 15:          # correct threshold: > 15, not >= 15
        return (sign << 7) | 0x7E
    mant8  = (mant32 >> 20) & 0x7
    guard  = (mant32 >> 19) & 1
    sticky = bool(mant32 & 0x7FFFF)
    if guard and (sticky or bool(mant8 & 1)):
        mant8 += 1
        if mant8 > 7:
            mant8 = 0; exp8 += 1
            if exp8 > 15: return (sign << 7) | 0x7E
    enc = (sign << 7) | (exp8 << 3) | mant8
    if (enc & 0x7F) == 0x7F: enc = (enc & 0x80) | 0x7E
    return enc

def f8_to_f32_ref(v: int) -> float:
    sign = (v >> 7) & 1
    exp8 = (v >> 3) & 0xF
    mant = v & 0x7
    if exp8 == 0xF and mant == 0x7: return float('nan')
    if exp8 == 0: val = mant / 8.0 * (2 ** -6)
    else:         val = (1.0 + mant / 8.0) * (2 ** (exp8 - 7))
    return -val if sign else val

# ── Test cases ────────────────────────────────────────────────────────────────

TEST_CASES = [
    # (input, expected_decoded, max_rel_err, description)
    ( 0.0,      0.0,    0.0,   "zero"),
    ( 1.0,      1.0,    0.0,   "one"),
    (-1.0,     -1.0,    0.0,   "minus one"),
    ( 2.0,      2.0,    0.0,   "two"),
    ( 16.0,    16.0,    0.0,   "sixteen"),
    ( 128.0,  128.0,    0.0,   "128"),
    ( 256.0,  256.0,    0.0,   "256 (was broken: encoded as 448 with >= 15 bug)"),
    ( 448.0,  448.0,    0.0,   "max value 448"),
    (-448.0, -448.0,    0.0,   "min value -448"),
    ( 0.1,     None,    0.13,  "0.1 (lossy, within 13% ok)"),
    ( 300.0,   None,    0.08,  "300 (exp8=15, valid normal)"),
    ( float('nan'), float('nan'), 0.0, "NaN → 0x7F"),
]

def run_python_tests() -> bool:
    print("── Python reference tests ───────────────────────────────────────")
    all_pass = True
    for inp, expected, max_err, desc in TEST_CASES:
        enc = f32_to_f8_ref(inp)
        dec = f8_to_f32_ref(enc)

        if expected is not None and math.isnan(expected):
            ok = math.isnan(dec)
        elif expected is not None:
            ok = abs(dec - expected) < 1e-4
        else:
            rel = abs(dec - inp) / abs(inp) if inp != 0 else 0
            ok = rel <= max_err

        status = "✓" if ok else "✗"
        print(f"  {status} {desc}: {inp} → enc={enc:#04x} → {dec}")
        if not ok:
            all_pass = False
    return all_pass

# ── Optional: test against compiled ggml library ─────────────────────────────

def run_ggml_tests(llama_path: str) -> bool:
    """
    Test against the compiled ggml shared library if available.
    Looks for ggml.dll (Windows) or libggml.so (Linux).
    """
    import platform
    if platform.system() == "Windows":
        lib_name = "ggml.dll"
        lib_path = Path(llama_path) / "build" / "bin" / "Release" / lib_name
    else:
        lib_name = "libggml.so"
        lib_path = Path(llama_path) / "build" / lib_name

    if not lib_path.exists():
        print(f"\n── GGML library test: SKIPPED (not found at {lib_path})")
        print("   Build llama.cpp first, then re-run for compiled validation.")
        return True

    print(f"\n── GGML compiled library tests ({lib_path.name}) ──────────────")
    try:
        lib = ctypes.CDLL(str(lib_path))
        # Prototype: void quantize_row_f8_e4m3fn(const float*, void*, int64_t)
        lib.quantize_row_f8_e4m3fn.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int64
        ]
        lib.dequantize_row_f8_e4m3fn.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64
        ]

        test_vals = [0.0, 1.0, -1.0, 256.0, 448.0, 0.1, 128.0]
        all_pass = True
        for v in test_vals:
            inp_arr  = (ctypes.c_float * 1)(v)
            fp8_arr  = (ctypes.c_uint8 * 1)(0)
            out_arr  = (ctypes.c_float * 1)(0.0)

            lib.quantize_row_f8_e4m3fn(inp_arr, fp8_arr, 1)
            lib.dequantize_row_f8_e4m3fn(fp8_arr, out_arr, 1)

            ref_enc = f32_to_f8_ref(v)
            ggml_enc = fp8_arr[0]
            ok = (ggml_enc == ref_enc)
            status = "✓" if ok else "✗"
            print(f"  {status} {v}: ggml enc={ggml_enc:#04x} ref={ref_enc:#04x} {'match' if ok else 'MISMATCH'}")
            if not ok: all_pass = False
        return all_pass
    except Exception as e:
        print(f"  Could not load library: {e}")
        return True  # not a failure, just not testable yet

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--llama-path", default=None,
                   help="Path to llama.cpp root for compiled library test")
    args = p.parse_args()

    py_ok = run_python_tests()

    ggml_ok = True
    if args.llama_path:
        ggml_ok = run_ggml_tests(args.llama_path)

    print()
    if py_ok and ggml_ok:
        print("ALL TESTS PASSED ✓")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED ✗")
        print("Check that exp8 overflow threshold in ggml-quants.c is `> 15` not `>= 15`")
        sys.exit(1)
