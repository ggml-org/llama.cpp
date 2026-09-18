# SPDX-License-Identifier: MIT
"""Independent stdlib-only audit of architecture_compare raw quantization dumps.
RNE, NOT the old Vulkan half-away reference. Never mutates raw evidence.
"""
import argparse
import array
import json
import math
import struct
import sys
from pathlib import Path

def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]
def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]
def load(path, code):
    out = array.array(code)
    out.frombytes(path.read_bytes())
    if sys.byteorder != "little" and out.itemsize > 1:
        out.byteswap()
    return out

def inspect(folder, plane, limit=32):
    x = load(folder / (plane + ".f32"), "f")
    q = load(folder / (plane + ".q8"), "b")
    scales = load(folder / (plane + ".scales.f32"), "f")
    if not x or len(x) != len(q) or len(x) % 32 or len(scales) != len(x)//32:
        raise ValueError("raw dump size mismatch")
    if not all(math.isfinite(v) for v in x):
        raise ValueError("nonfinite input needs invalid-mask investigation")
    detail = []
    code_diff = scale_diff = bad_codes = bad_scales = max_ulp = boundary = 0
    normal_min = f32(2**-126)
    for g, d in enumerate(scales):
        m = max(abs(v) for v in x[g*32:(g+1)*32])
        ref = 1. if m == 0 else max(f32(m/127), normal_min)
        scale_diff += bits(d) != bits(ref)
        if not math.isfinite(d) or d <= 0:
            bad_scales += 1
            continue
        max_ulp = max(max_ulp, abs(bits(d)-bits(ref)))
        if (m == 0 and d != 1.) or abs(d-ref)/ref > 5e-6:
            bad_scales += 1
        for j in range(32):
            i = 32*g+j
            zr = max(-127., min(127., f32(x[i]/ref)))
            qr = round(zr)
            z = x[i]/d
            lo = round(max(-127., min(127., z-2e-4)))
            hi = round(max(-127., min(127., z+2e-4)))
            if q[i] == -128 or not lo <= q[i] <= hi:
                bad_codes += 1
            elif q[i] != round(max(-127., min(127., f32(z)))):
                boundary += 1
            if q[i] != qr:
                code_diff += 1
                if len(detail) < limit:
                    detail.append({"index": i, "x": x[i], "cpu_scale":ref, "actual_scale":d,
                                   "cpu_rne":qr, "actual_code":q[i], "actual_normalized_double":z,
                                   "cpu_division_f32":zr,
                                   "reciprocal_multiply_f32":f32(x[i]*f32(1/d))})
    return {"codes":len(q),"groups":len(scales),"cpu_code_differences":code_diff,
            "cpu_scale_differences":scale_diff,"max_scale_ulp":max_ulp,"boundary_codes":boundary,
            "bad_codes":bad_codes,"bad_scales":bad_scales,"pass":bad_codes == bad_scales == 0,
            "difference_examples":detail}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    result = {"contract":"XMX-v0.4 RNE; explicit scale and normalized-error audit, not bit-exact division",
              "input":inspect(a.snapshot,"input"),"hidden":inspect(a.snapshot,"hidden")}
    output = a.out or a.snapshot / "independent-quant-audit.json"
    if output.exists():
        ap.error("audit already exists; choose a new --out instead of overwriting evidence")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0 if result["input"]["pass"] and result["hidden"]["pass"] else 1
if __name__ == "__main__":
    sys.exit(main())
