# IFAIRY64 OpenCL Implementation Plan

Status: Draft (2026-05-28)

This document defines the OpenCL-side preparation work for `GGML_TYPE_IFAIRY64`.
The first implementation target is correctness-preserving raw decode matmul.
LUT/prepacked kernels can be layered on later without changing the external
fallback contract.

## Scope

First supported OpenCL op:

- `GGML_OP_MUL_MAT`
- `src0->type == GGML_TYPE_IFAIRY64`
- `src1->type == GGML_TYPE_F32`
- `dst->type == GGML_TYPE_F32`
- raw `block_ifairy64` model weights
- contiguous `src0` and `src1`

Unsupported shapes and all iFairy custom ops must keep returning
`supports_op=false` so the scheduler falls back to CPU.

The semantic invariant is unchanged:

```text
w = wr + i*wi
x = xr + i*xi
w * conj(x) = (wr*xr + wi*xi) + i*(wi*xr - wr*xi)
```

## Buffer Layout Decision

Use a SoA raw-weight layout as the first GPU-readable layout:

```text
raw block_ifairy64:
  qs[16], d_real(fp16), d_imag(fp16)

OpenCL IF64 extra:
  q: row-major packed 2-bit codes, 16 bytes per 64-value block
  d: row-major fp16 scale pairs, 4 bytes per 64-value block
```

Rationale:

- avoids the uncoalesced 20-byte AoS block stride in kernels
- keeps the first kernel independent from the CPU LUT packed format
- uses the same total payload size as raw `block_ifairy64`, aside from buffer
  alignment padding
- keeps `get_tensor` able to reconstruct exact raw `block_ifairy64` bytes for
  test-backend roundtrips

Do not start with OpenCL LUT packed weights. That path is more likely to be
fast eventually, but it couples the first kernel to activation LUT preprocess
and packed tile scheduling. The raw SoA path gives a smaller correctness target
and a useful baseline for deciding whether LUT packing is worth the extra GPU
memory and preprocessing cost.

## Preparation Checklist

Before implementing kernels:

- add `ggml_tensor_extra_cl_ifairy64`
- allocate separate `q` and `d` sub-buffers with OpenCL alignment
- override OpenCL alloc size for `GGML_TYPE_IFAIRY64`
- pack raw `block_ifairy64` bytes into `q` and `d` in `set_tensor`
- reconstruct raw bytes from `q` and `d` in `get_tensor`
- keep `GGML_OPENCL_IFAIRY64` and kernel-ready gate disabled for compute until
  the matmul kernel is implemented

## Kernel Follow-Up

The first kernel should consume the SoA layout directly:

- `q[block]` supplies the packed phase codes
- `d[block]` supplies `(d_real, d_imag)` scales
- activation is read as `F32` initially, with an implementation-local conversion
  equivalent to the CPU `F32 -> GGML_TYPE_IFAIRY_Q16` path
- output stays `F32`

Only after this path is correct should we add an optional OpenCL LUT/prepacked
path for decode and larger batch shapes.
