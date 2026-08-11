# gfx1030 wave reductions and INT16 dot2

## Current implementation points

- `ggml/src/ggml-cuda/common.cuh` is the common device-intrinsic layer.
  - `ggml_cuda_warp_reduce_sum_gfx1030<use_native, width>` and
    `ggml_cuda_warp_reduce_max_gfx1030<use_native, width>` use the HIP
    `__builtin_amdgcn_wave_reduce_*` builtins for native full-wave gfx1030
    reductions and preserve the existing shuffle implementation otherwise.
  - `ggml_hip_sdot2_i16` exposes the RDNA2 `__builtin_amdgcn_sdot2` operation
    with a scalar fallback. It is currently a capability helper only.
- `ggml/src/ggml-cuda/fattn-tile.cuh` is the first production consumer for
  native wave reductions. The KQ max and softmax sum reductions already carry
  `use_gfx1030_native`, so they can select native or stock at compile time.
- The HIP compiler probe confirmed native wave reduction emits gfx1030 DPP
  instructions (`v_add_*_dpp`, `v_max_*_dpp`) plus the required cross-row
  `ds_swizzle_b32`.

## Why INT16 needs a separate consumer

The current CUDA backend has no numeric `GGML_TYPE_I16` matmul/vector-dot
consumer. Existing `int16_t` uses in quantized loaders are packed bit storage,
not signed-16 values; routing Q4/Q8 through `sdot2` would be mathematically
wrong and would discard the already validated `sdot4`/`sdot8` paths.

The safe next step is a dedicated I16 microkernel/probe (or a real I16 GGML
operation with host dispatch) before using `ggml_hip_sdot2_i16` in inference.
Do not claim an INT16 LLM speedup until that consumer exists and is profiled.

Both production and diagnostic HIP/RCCL builds compile with the new helpers;
no GPU correctness or benchmark run was performed for this follow-up yet.