#!/usr/bin/env python3
"""
validate_shaders.py — Pre-compile checks + Q4K math verification.

Run BEFORE building to catch issues early.

Usage:
    python validation/validate_shaders.py
    python validation/validate_shaders.py --glslc "C:/VulkanSDK/1.3.x/Bin/glslc.exe"
"""

import argparse
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

SHADER_DIR = Path(__file__).parent.parent / "shaders"

# ── Static analysis ───────────────────────────────────────────────────────────

CHECKS = [
    # (pattern_that_should_exist, pattern_that_must_NOT_exist, shader_glob, message)
    # Q4K sub-block size must be 32
    (r'(?:w_in_block|w_offset)\s*/\s*32u', r'w_in_block\s*/\s*16u',
     "linear_coop_q4k*.glsl", "Q4K sub-block must be /32u not /16u"),

    # Q4K nibble group must use /64u
    (r'w\s*/\s*64u', None,
     "linear_coop_q4k*.glsl", "Q4K nibble group must use /64u"),

    # Block size must be 144
    (r'144u', None,
     "linear_coop_q4k*.glsl", "Q4K block size must be 144u"),

    # barrier() after LDS stores before coopMatLoad
    # (approximate: barrier exists somewhere in inner loop body)
    (r'barrier\(\)', r'subgroupBarrier\(\).*coopMatLoad',
     "linear_coop_32.glsl", "32-tile shader must use barrier() not subgroupBarrier before store"),

    # INT32 staging for Q8 coopmat store
    (r'int32_t.*s_int32', None,
     "linear_coop_q8.glsl", "Q8 shader must have int32_t staging array"),

    # kvcache_update_q8 must use uint for packHalf2x16 result
    (r'uint\s+scale_bits\s*=\s*packHalf2x16', None,
     "kvcache_update_q8.glsl", "scale_bits must be uint not uint16_t"),

    # No buffer reference parameters in silu shader
    (None, r'void\s+\w+\([^)]*uint8_t\s*\[\]',
     "linear_coop_q4k_silu.glsl", "silu shader must not have buffer params in functions"),

    # FP8 decoder must handle exponent=15 as valid (not overflow)
    # Check: exponent == 0xFu && mantissa == 0x7u is the NaN check (correct)
    (r'exponent == 0xFu && mantissa == 0x7u', None,
     "linear_coop_fp8.glsl", "FP8 decoder must check exp==0xF AND mant==7 for NaN"),
    # Wave32 variants must use local_size_x = 32
    (r'local_size_x\s*=\s*32', None,
     "linear_coop_q4k_w32.glsl", "Wave32 shader must use local_size_x = 32"),
    (r'local_size_x\s*=\s*32', None,
     "linear_coop_q8_w32.glsl",  "Wave32 shader must use local_size_x = 32"),
    # Wave32 inner loops must use 8 elements per thread (not 4)
    (r'tid\s*\*\s*8u', None,
     "linear_coop_q4k_w32.glsl", "Wave32 shader must loop over 8 elements per thread"),
    (r'tid\s*\*\s*8u', None,
     "linear_coop_q8_w32.glsl",  "Wave32 shader must loop over 8 elements per thread"),
]

def run_static_checks() -> int:
    failures = 0
    print("── Static shader analysis ───────────────────────────────────────")
    import glob

    for must_have, must_not_have, pattern, msg in CHECKS:
        matches = list(SHADER_DIR.glob(pattern))
        if not matches:
            print(f"  SKIP {pattern} (no files match)")
            continue
        for shader_path in matches:
            content = shader_path.read_text()
            name    = shader_path.name

            if must_have and not re.search(must_have, content):
                print(f"  ✗ {name}: MISSING — {msg}")
                failures += 1
                continue

            if must_not_have and re.search(must_not_have, content):
                print(f"  ✗ {name}: FORBIDDEN pattern found — {msg}")
                failures += 1
                continue

            print(f"  ✓ {name}: {msg}")

    return failures

# ── Q4K math verification ────────────────────────────────────────────────────

def verify_q4k_math() -> int:
    print("\n── Q4K dequant math verification ────────────────────────────────")
    import random
    random.seed(2026)

    def unpack_half(lo, hi):
        return struct.unpack('e', struct.pack('H', (hi << 8) | lo))[0]

    def get_scale_min(scales, j):
        if j < 4:
            return float(scales[j] & 0x3F), float(scales[j+4] & 0x3F)
        sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
        m  = (scales[j+4] >> 4)   | ((scales[j  ] >> 6) << 4)
        return float(sc), float(m)

    def our_nibble(qs, w):
        g = w // 64; p = w % 32; s = (w % 64) // 32
        bval = qs[g * 32 + p]
        return bval & 0xF if s == 0 else bval >> 4

    def ggml_dequant(block, w):
        d    = unpack_half(block[0], block[1])
        dmin = unpack_half(block[2], block[3])
        scales = block[4:16]; qs = block[16:144]
        # ggml loop: j steps by 64, is = sub-block
        g = w // 64; within = w % 64
        is_sub = g * 2
        if within < 32:
            sc, m = get_scale_min(scales, is_sub)
            nibble = float(qs[g * 32 + within] & 0xF)
        else:
            sc, m = get_scale_min(scales, is_sub + 1)
            nibble = float(qs[g * 32 + (within - 32)] >> 4)
        return d * sc * nibble - dmin * m

    def our_dequant(block, w):
        d    = unpack_half(block[0], block[1])
        dmin = unpack_half(block[2], block[3])
        scales = block[4:16]; qs = block[16:144]
        j = w // 32
        sc, m = get_scale_min(scales, j)
        nibble = float(our_nibble(qs, w))
        return d * sc * nibble - dmin * m

    block = bytearray(144)
    d_bits = struct.unpack('H', struct.pack('e', 0.1))[0]
    block[0] = d_bits & 0xFF; block[1] = d_bits >> 8
    dm_bits = struct.unpack('H', struct.pack('e', 0.02))[0]
    block[2] = dm_bits & 0xFF; block[3] = dm_bits >> 8
    for i in range(12): block[4+i] = random.randint(0, 255)
    for i in range(128): block[16+i] = random.randint(0, 255)

    mismatches = 0
    for w in range(256):
        a = our_dequant(block, w)
        b = ggml_dequant(block, w)
        if abs(a - b) > 1e-5:
            print(f"  ✗ weight {w}: ours={a:.6f} ggml={b:.6f}")
            mismatches += 1

    if mismatches == 0:
        print(f"  ✓ All 256 Q4K weights match ggml reference exactly")
    return mismatches

# ── glslc compilation test ────────────────────────────────────────────────────

def run_compile_tests(glslc: str) -> int:
    if not glslc or not Path(glslc).exists():
        print(f"\n── Shader compilation: SKIPPED (glslc not found at {glslc})")
        print("   Install Vulkan SDK and pass --glslc path to test compilation.")
        return 0

    print(f"\n── Shader compilation (glslc) ───────────────────────────────────")
    failures = 0
    for shader in sorted(SHADER_DIR.glob("*.glsl")):
        cmd = [glslc, "--target-env=vulkan1.2", "-O",
               "-fshader-stage=compute", str(shader), "-o", os.devnull]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  ✓ {shader.name}")
        else:
            print(f"  ✗ {shader.name}")
            for line in r.stderr.splitlines():
                print(f"      {line}")
            failures += 1
    return failures

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--glslc", default=None,
                   help="Path to glslc.exe for compilation tests")
    args = p.parse_args()

    f1 = run_static_checks()
    f2 = verify_q4k_math()
    f3 = run_compile_tests(args.glslc)

    total = f1 + f2 + f3
    print(f"\n{'ALL PASS ✓' if total == 0 else f'{total} FAILURES ✗'}")
    sys.exit(0 if total == 0 else 1)
