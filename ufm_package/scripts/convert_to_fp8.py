#!/usr/bin/env python3
"""
convert_to_fp8.py — Convert a GGUF model to FP8 E4M3FN format.

Usage:
    python scripts/convert_to_fp8.py model_q4km.gguf model_fp8.gguf
    python scripts/convert_to_fp8.py model.gguf model_fp8.gguf --tensors ffn

Requirements:
    pip install gguf numpy
    gguf package: pip install -e llama.cpp/gguf-py

What this produces:
  For each weight tensor converted:
    - <name>             stored as GGML_TYPE_F8_E4M3FN (uint8, 1 byte/weight)
    - <name>.fp8_scale   stored as F32 scalar (1 element)
  Non-weight tensors (norms, embeddings) kept as F16.

The GPU shader (linear_coop_fp8.glsl) reads both tensors:
    float weight = fp8_decode(raw_byte) * scale

FP8 vs Q4_K size comparison (7B model):
    F16:    ~14.5 GB
    FP8:    ~7.7  GB  (this format)
    Q4_K_M: ~4.1 GB

FP8 is NOT a compression win over Q4_K. The win is 4x WMMA throughput
on RDNA4 when the driver maps uint8 coopmat to hardware FP8 WMMA units.
Use FP8 for speed on models that already fit in VRAM, not for size reduction.
"""

import argparse
import math
import struct
import sys
from pathlib import Path
import numpy as np

try:
    import gguf
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("Install: pip install -e llama.cpp/gguf-py  (or pip install gguf)")

GGML_TYPE_F8_E4M3FN = 40
FP8_MAX             = 448.0

# Tensors to keep as-is (too small or too sensitive to FP8 precision)
SKIP_PATTERNS = [
    "token_embd",   # embedding table
    "output_norm",  # RMS norm weights
    "attn_norm",
    "ffn_norm",
    ".bias",
]

# ── FP8 E4M3FN encoder (verified: correct for all values 0..448) ──────────────
# Critical fix: overflow threshold is exp8 > 15, NOT >= 15.
# E4M3FN allows exp=15 with mant=0..6 as valid normals (256.0..416.0).
# Only exp=15 + mant=7 is NaN (encodings 0x7F and 0xFF).

def f32_to_f8_array(x: np.ndarray) -> np.ndarray:
    """Vectorised FP8 E4M3FN encoder. Returns uint8 array."""
    x   = np.asarray(x, dtype=np.float32)
    out = np.zeros(x.shape, dtype=np.uint8)

    # Clamp and handle special values
    x = np.where(np.isnan(x),  0.0, x)          # NaN → 0 for weights
    x = np.where(np.isposinf(x),  FP8_MAX, x)
    x = np.where(np.isneginf(x), -FP8_MAX, x)
    x = np.clip(x, -FP8_MAX, FP8_MAX)

    zero_mask = (x == 0.0)
    bits    = x.view(np.uint32)
    sign    = (bits >> 31) & 0x1
    exp32   = ((bits >> 23) & 0xFF).astype(np.int32) - 127
    mant32  = bits & 0x7FFFFF
    exp8    = exp32 + 7

    # --- Denormal (exp8 <= 0) ---
    denorm = ~zero_mask & (exp8 <= 0)
    if np.any(denorm):
        shift  = np.clip(1 - exp8[denorm], 0, 3)
        mant_d = (((0x800000 | mant32[denorm]) >> (20 + shift)) & 0x7).astype(np.uint8)
        out[denorm] = ((sign[denorm] << 7) | mant_d).astype(np.uint8)

    # --- Overflow (exp8 > 15) — FIX: was >= 15, must be > 15 ---
    overflow = ~zero_mask & (exp8 > 15)
    if np.any(overflow):
        out[overflow] = ((sign[overflow] << 7) | 0x7E).astype(np.uint8)

    # --- Normal ---
    normal = ~zero_mask & (exp8 > 0) & (exp8 <= 15)
    if np.any(normal):
        m8     = ((mant32[normal] >> 20) & 0x7).astype(np.uint32)
        guard  = (mant32[normal] >> 19) & 1
        sticky = (mant32[normal] & 0x7FFFF) != 0
        rup    = guard.astype(bool) & (sticky.astype(bool) | (m8 & 1).astype(bool))
        m8[rup] += 1

        exp8_n   = exp8[normal].copy()
        ovf_mant = m8 > 7
        m8[ovf_mant]      = 0
        exp8_n[ovf_mant] += 1

        still_overflow = exp8_n > 15
        enc = ((sign[normal] << 7) | (exp8_n << 3) | m8).astype(np.uint8)
        enc[still_overflow] = ((sign[normal][still_overflow] << 7) | 0x7E).astype(np.uint8)

        # Block NaN encoding 0x7F / 0xFF
        is_nan_enc = (enc & 0x7F) == 0x7F
        enc[is_nan_enc] = ((enc[is_nan_enc] & 0x80) | 0x7E).astype(np.uint8)
        out[normal] = enc

    return out


def f8_to_f32_array(v: np.ndarray) -> np.ndarray:
    """Decode FP8 E4M3FN uint8 → float32 (for verification)."""
    v    = v.astype(np.uint32)
    sign = (v >> 7) & 0x1
    exp8 = (v >> 3) & 0xF
    mant =  v       & 0x7
    out  = np.zeros(v.shape, dtype=np.float32)

    nan_m  = (exp8 == 15) & (mant == 7)
    out[nan_m] = np.nan

    denorm_m = (exp8 == 0) & ~nan_m
    if np.any(denorm_m):
        val = mant[denorm_m].astype(np.float32) / 8.0 * (2.0 ** -6)
        out[denorm_m] = np.where(sign[denorm_m] == 1, -val, val)

    normal_m = ~nan_m & ~denorm_m
    if np.any(normal_m):
        val = (1.0 + mant[normal_m].astype(np.float32) / 8.0) * \
              (2.0 ** (exp8[normal_m].astype(np.float32) - 7.0))
        out[normal_m] = np.where(sign[normal_m] == 1, -val, val)

    return out


def should_convert(name: str, shape: tuple, filter_strs, skip_small: int) -> bool:
    for pat in SKIP_PATTERNS:
        if pat in name:
            return False
    if filter_strs and not any(f in name for f in filter_strs):
        return False
    if len(shape) < 2:
        return False
    total = 1
    for d in shape: total *= d
    return total >= skip_small


def convert(inp: str, out: str, tensor_filter, skip_small: int):
    print(f"Reading  : {inp}")
    reader = GGUFReader(inp)
    tensors = {t.name: t for t in reader.tensors}
    print(f"Tensors  : {len(tensors)}")

    to_convert = {
        n: t for n, t in tensors.items()
        if should_convert(n, tuple(t.shape), tensor_filter, skip_small)
    }
    print(f"→ FP8    : {len(to_convert)}")
    print(f"→ F16    : {len(tensors) - len(to_convert)}")

    # Get architecture for writer
    arch = "llama"
    if "general.architecture" in reader.fields:
        arch = str(reader.fields["general.architecture"].parts[-1])

    writer = GGUFWriter(out, arch)

    # Copy all metadata key-values
    for key, field in reader.fields.items():
        if key.startswith("GGUF."): continue
        try:
            writer.add_key_value(key, field.parts[-1].tolist(), field.types[0])
        except Exception:
            pass  # skip any that can't be copied verbatim

    writer.add_string("quantize.fp8.method", "e4m3fn_per_tensor")
    writer.add_uint32("quantize.fp8.ggml_type", GGML_TYPE_F8_E4M3FN)

    n_conv = 0
    for name, tensor in tensors.items():
        # Dequantise to F32 whatever the source type
        try:
            w_f32 = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
        except Exception:
            w_f32 = tensor.data.astype(np.float32)

        shape = tuple(tensor.shape)

        if name in to_convert:
            mx = float(np.max(np.abs(w_f32)))
            if mx == 0.0: mx = 1.0
            scale   = mx / FP8_MAX
            fp8_data = f32_to_f8_array(w_f32 / scale)

            # Verify round-trip max error
            decoded = f8_to_f32_array(fp8_data) * scale
            max_rel = float(np.max(np.abs(decoded - w_f32) / (np.abs(w_f32) + 1e-8)))

            writer.add_tensor(name, fp8_data.reshape(shape), GGML_TYPE_F8_E4M3FN)
            writer.add_tensor(f"{name}.fp8_scale",
                              np.array([scale], dtype=np.float32),
                              GGMLQuantizationType.F32)
            n_conv += 1
            print(f"  [FP8] {name:<60s}  scale={scale:.4f}  max_rel_err={max_rel:.3f}")
        else:
            f16 = w_f32.astype(np.float16)
            writer.add_tensor(name, f16.reshape(shape), GGMLQuantizationType.F16)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    in_sz  = Path(inp).stat().st_size / 1e9
    out_sz = Path(out).stat().st_size / 1e9
    print(f"\nDone: {in_sz:.2f} GB → {out_sz:.2f} GB  "
          f"({100*(1-out_sz/in_sz):.1f}% smaller than input)")
    print(f"Output: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert GGUF weights to FP8 E4M3FN")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--tensors", nargs="+",
                   help="Only convert tensors whose names contain these strings "
                        "(e.g. --tensors ffn_gate ffn_up ffn_down)")
    p.add_argument("--skip-small", type=int, default=4096,
                   help="Skip tensors with fewer elements (default 4096)")
    args = p.parse_args()
    convert(args.input, args.output, args.tensors, args.skip_small)
