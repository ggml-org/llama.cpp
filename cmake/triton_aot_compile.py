
import sys
try:
    import triton
    import triton.language as tl
    from triton.compiler.compiler import ASTSource
except ImportError:
    pass
try:
    from triton.backends.amd.compiler import HIPBackend
except ImportError:
    pass
try:
    from triton.backends.nvidia.compiler import CUDABackend
except ImportError:
    pass
#!/usr/bin/env python3
"""AOT compilation of fused RotorQuant Triton kernels to .hsaco (HIP) or .cubin (CUDA).

Source kernels extracted from carlosfundora/sglang-1-bit-turbo @ 4786c5de
These are standalone versions of the kernels originally defined inside a Python
class/function scope (closure variables replaced with explicit parameters).

Usage:
    python3 cmake/triton_aot_compile.py --output-dir build/triton-kernels --target hip
    python3 cmake/triton_aot_compile.py --output-dir build/triton-kernels --target cuda
    python3 cmake/triton_aot_compile.py --output-dir build/triton-kernels --target hip --arch gfx1031
"""

from __future__ import annotations

import argparse
import os
import sys

import triton
import triton.language as tl
from triton.compiler.compiler import compile as tc_compile, ASTSource

# ---------------------------------------------------------------------------
# Standalone Triton kernel definitions (extracted from sglang RotorQuant engine)
# ---------------------------------------------------------------------------


@triton.jit
def _planar2_quantize_kernel(
    input_ptr, indices_ptr,
    rot2_ptr, centroids_ptr,
    batch_size, emb_dim,
    n_groups: tl.constexpr,
    n_levels: tl.constexpr,
    stride_in_b, stride_in_d,
    stride_idx_b, stride_idx_d,
    BLOCK_G: tl.constexpr,
):
    """Givens rotate → nearest centroid → store int8 index (3-bit path)."""
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    g_mask = g_offs < n_groups

    cos_t = tl.load(rot2_ptr + g_offs * 2 + 0, mask=g_mask, other=1.0)
    sin_t = tl.load(rot2_ptr + g_offs * 2 + 1, mask=g_mask, other=0.0)

    d0 = g_offs * 2
    v0 = tl.load(input_ptr + pid_b * stride_in_b + d0 * stride_in_d,
                 mask=g_mask & (d0 < emb_dim), other=0.0)
    v1 = tl.load(input_ptr + pid_b * stride_in_b + (d0 + 1) * stride_in_d,
                 mask=g_mask & ((d0 + 1) < emb_dim), other=0.0)

    r0 = cos_t * v0 - sin_t * v1
    r1 = sin_t * v0 + cos_t * v1

    best_idx0 = tl.zeros_like(r0).to(tl.int32)
    best_dist0 = tl.abs(r0 - tl.load(centroids_ptr))
    for i in tl.static_range(1, n_levels):
        c = tl.load(centroids_ptr + i)
        dd = tl.abs(r0 - c)
        mask = dd < best_dist0
        best_dist0 = tl.where(mask, dd, best_dist0)
        best_idx0 = tl.where(mask, i, best_idx0)

    best_idx1 = tl.zeros_like(r1).to(tl.int32)
    best_dist1 = tl.abs(r1 - tl.load(centroids_ptr))
    for i in tl.static_range(1, n_levels):
        c = tl.load(centroids_ptr + i)
        dd = tl.abs(r1 - c)
        mask = dd < best_dist1
        best_dist1 = tl.where(mask, dd, best_dist1)
        best_idx1 = tl.where(mask, i, best_idx1)

    tl.store(indices_ptr + pid_b * stride_idx_b + d0 * stride_idx_d,
             best_idx0.to(tl.int8), mask=g_mask & (d0 < emb_dim))
    tl.store(indices_ptr + pid_b * stride_idx_b + (d0 + 1) * stride_idx_d,
             best_idx1.to(tl.int8), mask=g_mask & ((d0 + 1) < emb_dim))


@triton.jit
def _planar2_dequantize_kernel(
    indices_ptr, output_ptr,
    rot2_ptr, centroids_ptr,
    batch_size, emb_dim,
    n_groups: tl.constexpr,
    n_levels: tl.constexpr,
    stride_idx_b, stride_idx_d,
    stride_out_b, stride_out_d,
    BLOCK_G: tl.constexpr,
):
    """Load int8 index → centroid lookup → inverse Givens rotate → fp16."""
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    g_mask = g_offs < n_groups

    cos_t = tl.load(rot2_ptr + g_offs * 2 + 0, mask=g_mask, other=1.0)
    sin_t = tl.load(rot2_ptr + g_offs * 2 + 1, mask=g_mask, other=0.0)

    d0 = g_offs * 2
    idx0 = tl.load(indices_ptr + pid_b * stride_idx_b + d0 * stride_idx_d,
                   mask=g_mask & (d0 < emb_dim), other=0).to(tl.int32)
    idx1 = tl.load(indices_ptr + pid_b * stride_idx_b + (d0 + 1) * stride_idx_d,
                   mask=g_mask & ((d0 + 1) < emb_dim), other=0).to(tl.int32)

    q0 = tl.load(centroids_ptr + idx0, mask=g_mask, other=0.0)
    q1 = tl.load(centroids_ptr + idx1, mask=g_mask, other=0.0)

    # Inverse Givens: R^T = [[cos, sin], [-sin, cos]]
    f0 = cos_t * q0 + sin_t * q1
    f1 = -sin_t * q0 + cos_t * q1

    tl.store(output_ptr + pid_b * stride_out_b + d0 * stride_out_d,
             f0, mask=g_mask & (d0 < emb_dim))
    tl.store(output_ptr + pid_b * stride_out_b + (d0 + 1) * stride_out_d,
             f1, mask=g_mask & ((d0 + 1) < emb_dim))


@triton.jit
def _fused_planar4_quant_pack_kernel(
    input_ptr, packed_ptr,
    rot2_ptr, centroids_ptr,
    batch_size, emb_dim,
    n_groups: tl.constexpr,
    n_levels: tl.constexpr,
    stride_in_b, stride_in_d,
    stride_pack_b, stride_pack_g,
    BLOCK_G: tl.constexpr,
):
    """Fused: Givens rotate → nearest centroid → 4-bit pack (lo|hi<<4 per byte)."""
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    g_mask = g_offs < n_groups

    cos_t = tl.load(rot2_ptr + g_offs * 2 + 0, mask=g_mask, other=1.0)
    sin_t = tl.load(rot2_ptr + g_offs * 2 + 1, mask=g_mask, other=0.0)

    d0 = g_offs * 2
    v0 = tl.load(input_ptr + pid_b * stride_in_b + d0 * stride_in_d,
                 mask=g_mask & (d0 < emb_dim), other=0.0)
    v1 = tl.load(input_ptr + pid_b * stride_in_b + (d0 + 1) * stride_in_d,
                 mask=g_mask & ((d0 + 1) < emb_dim), other=0.0)

    r0 = cos_t * v0 - sin_t * v1
    r1 = sin_t * v0 + cos_t * v1

    best_idx0 = tl.zeros_like(r0).to(tl.int32)
    best_dist0 = tl.abs(r0 - tl.load(centroids_ptr))
    for i in tl.static_range(1, n_levels):
        c = tl.load(centroids_ptr + i)
        dd = tl.abs(r0 - c)
        mask = dd < best_dist0
        best_dist0 = tl.where(mask, dd, best_dist0)
        best_idx0 = tl.where(mask, i, best_idx0)

    best_idx1 = tl.zeros_like(r1).to(tl.int32)
    best_dist1 = tl.abs(r1 - tl.load(centroids_ptr))
    for i in tl.static_range(1, n_levels):
        c = tl.load(centroids_ptr + i)
        dd = tl.abs(r1 - c)
        mask = dd < best_dist1
        best_dist1 = tl.where(mask, dd, best_dist1)
        best_idx1 = tl.where(mask, i, best_idx1)

    packed_byte = (best_idx0 & 0x0F) | ((best_idx1 & 0x0F) << 4)
    tl.store(packed_ptr + pid_b * stride_pack_b + g_offs * stride_pack_g,
             packed_byte.to(tl.int8), mask=g_mask)


@triton.jit
def _fused_planar4_unpack_dequant_kernel(
    packed_ptr, output_ptr,
    rot2_ptr, centroids_ptr, norms_ptr,
    batch_size, emb_dim,
    n_groups: tl.constexpr,
    stride_pack_b, stride_pack_g,
    stride_out_b, stride_out_d,
    BLOCK_G: tl.constexpr,
):
    """Fused: unpack 4-bit → centroid lookup → inv Givens → rescale → fp16."""
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    g_mask = g_offs < n_groups

    packed = tl.load(packed_ptr + pid_b * stride_pack_b + g_offs * stride_pack_g,
                     mask=g_mask, other=0).to(tl.int32)
    idx0 = packed & 0x0F
    idx1 = (packed >> 4) & 0x0F

    q0 = tl.load(centroids_ptr + idx0, mask=g_mask, other=0.0)
    q1 = tl.load(centroids_ptr + idx1, mask=g_mask, other=0.0)

    cos_t = tl.load(rot2_ptr + g_offs * 2 + 0, mask=g_mask, other=1.0)
    sin_t = tl.load(rot2_ptr + g_offs * 2 + 1, mask=g_mask, other=0.0)

    f0 = cos_t * q0 + sin_t * q1
    f1 = -sin_t * q0 + cos_t * q1

    norm = tl.load(norms_ptr + pid_b).to(tl.float32)
    f0 = f0 * norm
    f1 = f1 * norm

    d0 = g_offs * 2
    tl.store(output_ptr + pid_b * stride_out_b + d0 * stride_out_d,
             f0.to(tl.float16), mask=g_mask & (d0 < emb_dim))
    tl.store(output_ptr + pid_b * stride_out_b + (d0 + 1) * stride_out_d,
             f1.to(tl.float16), mask=g_mask & ((d0 + 1) < emb_dim))


@triton.jit
def _fused_iso4_quant_pack_kernel(
    input_ptr, packed_ptr,
    qL_ptr, qR_ptr, centroids_ptr,
    batch_size, d_padded,
    n_groups: tl.constexpr,
    n_levels: tl.constexpr,
    stride_in_b, stride_in_d,
    stride_pack_b, stride_pack_e,
    BLOCK_G: tl.constexpr,
):
    """Fused: quaternion sandwich → nearest centroid → 4-bit pack (2 bytes/group)."""
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    g_mask = g_offs < n_groups

    d0 = g_offs * 4
    v0 = tl.load(input_ptr + pid_b * stride_in_b + d0 * stride_in_d, mask=g_mask, other=0.0)
    v1 = tl.load(input_ptr + pid_b * stride_in_b + (d0 + 1) * stride_in_d, mask=g_mask, other=0.0)
    v2 = tl.load(input_ptr + pid_b * stride_in_b + (d0 + 2) * stride_in_d, mask=g_mask, other=0.0)
    v3 = tl.load(input_ptr + pid_b * stride_in_b + (d0 + 3) * stride_in_d, mask=g_mask, other=0.0)

    aw = tl.load(qL_ptr + g_offs * 4 + 0, mask=g_mask, other=1.0)
    ax = tl.load(qL_ptr + g_offs * 4 + 1, mask=g_mask, other=0.0)
    ay = tl.load(qL_ptr + g_offs * 4 + 2, mask=g_mask, other=0.0)
    az = tl.load(qL_ptr + g_offs * 4 + 3, mask=g_mask, other=0.0)

    bw = tl.load(qR_ptr + g_offs * 4 + 0, mask=g_mask, other=1.0)
    bx = tl.load(qR_ptr + g_offs * 4 + 1, mask=g_mask, other=0.0)
    by = tl.load(qR_ptr + g_offs * 4 + 2, mask=g_mask, other=0.0)
    bz = tl.load(qR_ptr + g_offs * 4 + 3, mask=g_mask, other=0.0)

    # Hamilton product: temp = q_L * v (treating v as pure quaternion)
    tw = aw * v0 - ax * v1 - ay * v2 - az * v3
    tx = aw * v1 + ax * v0 + ay * v3 - az * v2
    ty = aw * v2 - ax * v3 + ay * v0 + az * v1
    tz = aw * v3 + ax * v2 - ay * v1 + az * v0

    # Hamilton product: result = temp * conj(q_R)
    rw = tw * bw + tx * bx + ty * by + tz * bz
    rx = -tw * bx + tx * bw - ty * bz + tz * by
    ry = -tw * by + tx * bz + ty * bw - tz * bx
    rz = -tw * bz - tx * by + ty * bx + tz * bw

    # Nearest centroid × 4 components
    best_iw = tl.zeros_like(rw).to(tl.int32)
    best_dw = tl.abs(rw - tl.load(centroids_ptr))
    for i in tl.static_range(1, n_levels):
        c = tl.load(centroids_ptr + i)
        m = tl.abs(rw - c) < best_dw
        best_dw = tl.where(m, tl.abs(rw - c), best_dw)
        best_iw = tl.where(m, i, best_iw)

    best_ix = tl.zeros_like(rx).to(tl.int32)
    best_dx = tl.abs(rx - tl.load(centroids_ptr))
    for i in tl.static_range(1, n_levels):
        c = tl.load(centroids_ptr + i)
        m = tl.abs(rx - c) < best_dx
        best_dx = tl.where(m, tl.abs(rx - c), best_dx)
        best_ix = tl.where(m, i, best_ix)

    best_iy = tl.zeros_like(ry).to(tl.int32)
    best_dy = tl.abs(ry - tl.load(centroids_ptr))
    for i in tl.static_range(1, n_levels):
        c = tl.load(centroids_ptr + i)
        m = tl.abs(ry - c) < best_dy
        best_dy = tl.where(m, tl.abs(ry - c), best_dy)
        best_iy = tl.where(m, i, best_iy)

    best_iz = tl.zeros_like(rz).to(tl.int32)
    best_dz = tl.abs(rz - tl.load(centroids_ptr))
    for i in tl.static_range(1, n_levels):
        c = tl.load(centroids_ptr + i)
        m = tl.abs(rz - c) < best_dz
        best_dz = tl.where(m, tl.abs(rz - c), best_dz)
        best_iz = tl.where(m, i, best_iz)

    # Pack: 4 indices → 2 bytes (byte0 = iw|ix<<4, byte1 = iy|iz<<4)
    byte0 = (best_iw & 0x0F) | ((best_ix & 0x0F) << 4)
    byte1 = (best_iy & 0x0F) | ((best_iz & 0x0F) << 4)

    tl.store(packed_ptr + pid_b * stride_pack_b + (g_offs * 2) * stride_pack_e,
             byte0.to(tl.int8), mask=g_mask)
    tl.store(packed_ptr + pid_b * stride_pack_b + (g_offs * 2 + 1) * stride_pack_e,
             byte1.to(tl.int8), mask=g_mask)


@triton.jit
def _fused_iso4_unpack_dequant_kernel(
    packed_ptr, output_ptr,
    qL_ptr, qR_ptr, centroids_ptr, norms_ptr,
    batch_size, d_padded, head_dim,
    n_groups: tl.constexpr,
    stride_pack_b, stride_pack_e,
    stride_out_b, stride_out_d,
    BLOCK_G: tl.constexpr,
):
    """Fused: unpack 4-bit → centroid → inv quat sandwich → rescale → fp16."""
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    g_mask = g_offs < n_groups

    byte0 = tl.load(packed_ptr + pid_b * stride_pack_b + (g_offs * 2) * stride_pack_e,
                    mask=g_mask, other=0).to(tl.int32)
    byte1 = tl.load(packed_ptr + pid_b * stride_pack_b + (g_offs * 2 + 1) * stride_pack_e,
                    mask=g_mask, other=0).to(tl.int32)

    iw = byte0 & 0x0F
    ix = (byte0 >> 4) & 0x0F
    iy = byte1 & 0x0F
    iz = (byte1 >> 4) & 0x0F

    v0 = tl.load(centroids_ptr + iw, mask=g_mask, other=0.0)
    v1 = tl.load(centroids_ptr + ix, mask=g_mask, other=0.0)
    v2 = tl.load(centroids_ptr + iy, mask=g_mask, other=0.0)
    v3 = tl.load(centroids_ptr + iz, mask=g_mask, other=0.0)

    aw = tl.load(qL_ptr + g_offs * 4 + 0, mask=g_mask, other=1.0)
    ax = tl.load(qL_ptr + g_offs * 4 + 1, mask=g_mask, other=0.0)
    ay = tl.load(qL_ptr + g_offs * 4 + 2, mask=g_mask, other=0.0)
    az = tl.load(qL_ptr + g_offs * 4 + 3, mask=g_mask, other=0.0)

    bw = tl.load(qR_ptr + g_offs * 4 + 0, mask=g_mask, other=1.0)
    bx = tl.load(qR_ptr + g_offs * 4 + 1, mask=g_mask, other=0.0)
    by = tl.load(qR_ptr + g_offs * 4 + 2, mask=g_mask, other=0.0)
    bz = tl.load(qR_ptr + g_offs * 4 + 3, mask=g_mask, other=0.0)

    # Inverse sandwich: conj(q_L) * v * q_R
    tw = aw * v0 + ax * v1 + ay * v2 + az * v3
    tx = aw * v1 - ax * v0 - ay * v3 + az * v2
    ty = aw * v2 + ax * v3 - ay * v0 - az * v1
    tz = aw * v3 - ax * v2 + ay * v1 - az * v0

    rw = tw * bw - tx * bx - ty * by - tz * bz
    rx = tw * bx + tx * bw + ty * bz - tz * by
    ry = tw * by - tx * bz + ty * bw + tz * bx
    rz = tw * bz + tx * by - ty * bx + tz * bw

    norm = tl.load(norms_ptr + pid_b).to(tl.float32)
    rw = rw * norm
    rx = rx * norm
    ry = ry * norm
    rz = rz * norm

    d0 = g_offs * 4
    tl.store(output_ptr + pid_b * stride_out_b + d0 * stride_out_d,
             rw.to(tl.float16), mask=g_mask & (d0 < head_dim))
    tl.store(output_ptr + pid_b * stride_out_b + (d0 + 1) * stride_out_d,
             rx.to(tl.float16), mask=g_mask & ((d0 + 1) < head_dim))
    tl.store(output_ptr + pid_b * stride_out_b + (d0 + 2) * stride_out_d,
             ry.to(tl.float16), mask=g_mask & ((d0 + 2) < head_dim))
    tl.store(output_ptr + pid_b * stride_out_b + (d0 + 3) * stride_out_d,
             rz.to(tl.float16), mask=g_mask & ((d0 + 3) < head_dim))


# ---------------------------------------------------------------------------
# Compilation configurations: (kernel_fn, signature, constexprs, suffix)
# n_groups=64 for PlanarQuant (128 dims / 2), n_groups=32 for IsoQuant (128 dims / 4)
# n_levels=8 for 3-bit, n_levels=16 for 4-bit
# BLOCK_G=8 is a balanced tile size for gfx1031 wave size 64
# ---------------------------------------------------------------------------

KERNELS = [
    # 3-bit PlanarQuant (planar2): quantize
    (
        _planar2_quantize_kernel,
        {
            "input_ptr": "*fp16",
            "indices_ptr": "*i8",
            "rot2_ptr": "*fp32",
            "centroids_ptr": "*fp32",
            "batch_size": "i32",
            "emb_dim": "i32",
            "stride_in_b": "i32",
            "stride_in_d": "i32",
            "stride_idx_b": "i32",
            "stride_idx_d": "i32",
        },
        {"n_groups": 64, "n_levels": 8, "BLOCK_G": 8},
        "ng64_nl8",
    ),
    # 3-bit PlanarQuant (planar2): dequantize
    (
        _planar2_dequantize_kernel,
        {
            "indices_ptr": "*i8",
            "output_ptr": "*fp16",
            "rot2_ptr": "*fp32",
            "centroids_ptr": "*fp32",
            "batch_size": "i32",
            "emb_dim": "i32",
            "stride_idx_b": "i32",
            "stride_idx_d": "i32",
            "stride_out_b": "i32",
            "stride_out_d": "i32",
        },
        {"n_groups": 64, "n_levels": 8, "BLOCK_G": 8},
        "ng64_nl8",
    ),
    # 4-bit fused PlanarQuant: quantize + pack
    (
        _fused_planar4_quant_pack_kernel,
        {
            "input_ptr": "*fp16",
            "packed_ptr": "*i8",
            "rot2_ptr": "*fp32",
            "centroids_ptr": "*fp32",
            "batch_size": "i32",
            "emb_dim": "i32",
            "stride_in_b": "i32",
            "stride_in_d": "i32",
            "stride_pack_b": "i32",
            "stride_pack_g": "i32",
        },
        {"n_groups": 64, "n_levels": 16, "BLOCK_G": 8},
        "ng64_nl16",
    ),
    # 4-bit fused PlanarQuant: unpack + dequantize
    (
        _fused_planar4_unpack_dequant_kernel,
        {
            "packed_ptr": "*i8",
            "output_ptr": "*fp16",
            "rot2_ptr": "*fp32",
            "centroids_ptr": "*fp32",
            "norms_ptr": "*fp32",
            "batch_size": "i32",
            "emb_dim": "i32",
            "stride_pack_b": "i32",
            "stride_pack_g": "i32",
            "stride_out_b": "i32",
            "stride_out_d": "i32",
        },
        {"n_groups": 64, "BLOCK_G": 8},
        "ng64",
    ),
    # 4-bit fused IsoQuant: quantize + pack
    (
        _fused_iso4_quant_pack_kernel,
        {
            "input_ptr": "*fp16",
            "packed_ptr": "*i8",
            "qL_ptr": "*fp32",
            "qR_ptr": "*fp32",
            "centroids_ptr": "*fp32",
            "batch_size": "i32",
            "d_padded": "i32",
            "stride_in_b": "i32",
            "stride_in_d": "i32",
            "stride_pack_b": "i32",
            "stride_pack_e": "i32",
        },
        {"n_groups": 32, "n_levels": 16, "BLOCK_G": 8},
        "ng32_nl16",
    ),
    # 4-bit fused IsoQuant: unpack + dequantize
    (
        _fused_iso4_unpack_dequant_kernel,
        {
            "packed_ptr": "*i8",
            "output_ptr": "*fp16",
            "qL_ptr": "*fp32",
            "qR_ptr": "*fp32",
            "centroids_ptr": "*fp32",
            "norms_ptr": "*fp32",
            "batch_size": "i32",
            "d_padded": "i32",
            "head_dim": "i32",
            "stride_pack_b": "i32",
            "stride_pack_e": "i32",
            "stride_out_b": "i32",
            "stride_out_d": "i32",
        },
        {"n_groups": 32, "BLOCK_G": 8},
        "ng32",
    ),
]


def get_target(target_name: str, arch: str):
    """Build a Triton backend target object."""
    if target_name == "hip":
        from triton.backends.amd.compiler import GPUTarget
        warp_size = 64
        return GPUTarget("hip", arch, warp_size)
    elif target_name == "cuda":
        from triton.backends.nvidia.compiler import GPUTarget
        warp_size = 32
        return GPUTarget("cuda", arch, warp_size)
    else:
        raise ValueError(f"Unknown target: {target_name!r}, expected 'hip' or 'cuda'")


def compile_all(output_dir: str, target_name: str, arch: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    ext = "hsaco" if target_name == "hip" else "cubin"
    asm_key = "hsaco" if target_name == "hip" else "cubin"

    target = get_target(target_name, arch)
    import logging; logging.warning(f"Target: {target}")

    errors = []
    for fn, sig, consts, suffix in KERNELS:
        name = fn.__name__
        out_name = f"{name}_{suffix}.{ext}"
        out_path = os.path.join(output_dir, out_name)

        import logging; logging.warning(f"  Compiling {name} ({suffix}) ...", end=" ", flush=True)
        try:
            src = ASTSource(fn, signature=sig, constexprs=consts)
            compiled = tc_compile(src, target=target)
            binary = compiled.asm[asm_key]
            # binary may be bytes or str; normalise to bytes
            if isinstance(binary, str):
                binary = binary.encode()
            with open(out_path, "wb") as f:
                f.write(binary)
            import logging; logging.warning(f"ok → {out_name} ({len(binary)} bytes)")
        except Exception as exc:
            import logging; logging.warning(f"FAILED: {exc}")
            errors.append((name, exc))

    if errors:
        import logging; logging.warning(f"\n{len(errors)} kernel(s) failed to compile:", file=sys.stderr)
        for name, exc in errors:
            import logging; logging.warning(f"  {name}: {exc}", file=sys.stderr)
        sys.exit(1)
    else:
        import logging; logging.warning(f"\nAll kernels compiled to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="AOT-compile fused RotorQuant Triton kernels"
    )
    parser.add_argument(
        "--output-dir", default="build/triton-kernels",
        help="Directory to write .hsaco/.cubin files (default: build/triton-kernels)",
    )
    parser.add_argument(
        "--target", default="hip", choices=["hip", "cuda"],
        help="Backend target: 'hip' (AMD) or 'cuda' (NVIDIA) (default: hip)",
    )
    parser.add_argument(
        "--arch", default=None,
        help=(
            "GPU architecture string, e.g. 'gfx1031' for HIP or 'sm_89' for CUDA. "
            "Defaults to 'gfx1031' for HIP and 'sm_80' for CUDA."
        ),
    )
    args = parser.parse_args()

    if args.arch is None:
        args.arch = "gfx1031" if args.target == "hip" else "sm_80"

    import logging; logging.warning(f"RotorQuant Triton AOT compiler")
    import logging; logging.warning(f"  target : {args.target}")
    import logging; logging.warning(f"  arch   : {args.arch}")
    import logging; logging.warning(f"  output : {args.output_dir}")
    import logging; logging.warning()

    compile_all(args.output_dir, args.target, args.arch)


if __name__ == "__main__":
    main()
