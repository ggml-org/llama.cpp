# PLAN: NVIDIA native-BF16 Flash Attention MMA kernel

## Goal

Add a native BF16 Flash Attention tensor-core (MMA) kernel for NVIDIA GPUs (sm_80+) by
adapting the existing F16 MMA kernel `fattn-mma-f16.cuh`. This eliminates the implicit
BF16->F16 KV conversion pass and uses FP32 VKQ accumulation for better precision at deep
contexts (the F16 kernel's half2 VKQ accumulation at `mma.cuh:970-1019` is the PPL-drift
root cause). Mirrors the intent of AMD PR #26856, which explicitly names an NVIDIA
bf16-MMA kernel as separate follow-up work.

## Constraints

- No `fma.rn.bf16x2` (element-wise) usage; no scalar `cvt`+`fma` emulation.
- Do not modify AMD paths (`fattn-tile.cu` / `fattn-tile.cuh`) or the F16 MMA kernel
  behavior. The only edit to `fattn-mma-f16.cuh` is adding `#pragma once` at line 1.
- `V_DOT2_F32_BF16_AVAILABLE` stays undefined for NVIDIA.
- PTX core: `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`. A/B operands are
  `nv_bfloat162`, accumulators FP32.

## Design decisions

- **P stays in registers**: the MMA kernel never stages P in SRAM; convert the FP32 `KQ_C`
  accumulator to a `nv_bfloat162` B-tile exactly once before the VKQ mma.
- **VKQ in FP32 accumulators**: `T_C_VKQ = tile<16,8,float>` (ncols==8) /
  `tile<16,16,float>` (wide). KQ already accumulates in FP32 in the F16 kernel.
- **VKQ combine SRAM staging as `nv_bfloat162`** (not f32): keeps the wide-case combine
  buffer the same size as F16 (e.g. ~67.5KB for 256/256/64). SRAM sizes are unchanged
  because `nv_bfloat162` == `half2` == 4 bytes; the case-function byte math uses
  `sizeof(half2)`.
- **Load path is fully reusable - no copy needed**: `nv_bfloat162` and `half2` are both
  4 bytes and the loads are pure byte copies. The bf16 kernel keeps `extern __shared__
  half2 tile_Q[]` / `half2 * tile_K/V` (identical SRAM), reuses
  `flash_attn_ext_f16_load_tile` (363) and `flash_attn_ext_f16_load_mask` (352) unchanged,
  and only reinterprets the pointers as `nv_bfloat162` at the `load_ldmatrix` /
  `load_ldmatrix_trans` call sites (both T-templated and bit-agnostic).
- **Selection gate**: kernel chosen only when `K->type == GGML_TYPE_BF16 &&
  V->type == GGML_TYPE_BF16` and `bf16_mma_hardware_available(cc)`. The routing check is
  inserted **inside** the `turing_mma_available(cc)` branch of
  `ggml_cuda_get_best_fattn_kernel` (`fattn.cu:461`), which is already NVIDIA-only - so
  AMD is never routed here even though `bf16_mma_hardware_available` is true for AMD.
  `need_f16_K/V == false` so no `to_fp16` conversion kernels run; BF16 K/V are read raw
  from `K->data` / `V->data`. `get_alloc_size` for MMA_BF16 then returns zero extra
  (matches vec-for-BF16).
- **Register footprint exactly doubles** (both narrow and wide): bf16 narrow VKQ_C is
  `tile<16,8,float>` = 4 f32 regs/thread vs f16 `tile<16,4,half2>` = 2 regs; wide is
  `tile<16,16,float>` = 8 regs vs 4. For DV=256 that is ~64 vs ~32 regs for VKQ_C.
  Config (occupancy/nthreads) may need a retune pass for the bf16 path.
- **cols_per_warp**: `get_cols_per_warp(cc)` returns **16** on Ampere
  (`fattn-mma-f16.cuh:338`), so the narrow (cols_per_warp==8) path triggers only for
  ncols==8; the wide path is the default for ncols>=16.

## Steps

### 1. `ggml/src/ggml-cuda/fattn-mma-f16.cuh` (1-line edit)

- Add `#pragma once` at the top of the file. The file currently has **no include guard**
  (unlike `common.cuh`, `mma.cuh`, `fattn-common.cuh`). This is required so the bf16 header
  can `#include` it and so `fattn.cu` including both headers does not redefine the f16
  config getters / `mma_tile_sizes`.

### 2. `ggml/src/ggml-cuda/mma.cuh`

- Add an `#elif defined(AMPERE_MMA_AVAILABLE)` CUDA branch to the wide BF16 mma at
  `mma.cuh:1257-1259` (currently AMD-only):
  `mma(tile<16,16,float,dl_d>&, tile<16,8,nv_bfloat162,dl_ab>&,
  tile<16,8,nv_bfloat162,dl_ab>&)`, mirroring the wide f16 mma at `1196-1209`
  (2x m16n8k16 f32.bf16.bf16.f32: first with `Bxi[0]`/`Bxi[2]`, then `Bxi[1]`/`Bxi[3]`).
  Keep the existing AMD `#if` and the final `#else NO_DEVICE_CODE`.
- Add `get_bf16(tile<I,J,float>) -> tile<I,J/2,nv_bfloat162>` mirroring `get_half2` at
  `713`, using `__floats2bfloat162_rn`; guard with `AMPERE_MMA_AVAILABLE`.
- Add `get_transposed(tile<16,4,nv_bfloat162>) -> tile<8,8,nv_bfloat162>` mirroring `722`
  using `ggml_cuda_movmatrix`; guard with `AMPERE_MMA_AVAILABLE`.

Existing narrow BF16 mma at `mma.cuh:1181` (
`mma(tile<16,8,float>&, tile<16,8,nv_bfloat162>&, tile<8,8,nv_bfloat162>&)`) is reused for
both KQ and VKQ in the ncols==8 case.

### 3. New file `ggml/src/ggml-cuda/fattn-mma-bf16.cuh`

Adaptation of `fattn-mma-f16.cuh`. **`#include "fattn-mma-f16.cuh"`** (now guarded) to reuse
the config getters, `get_cols_per_warp`, `flash_attn_ext_f16_load_tile`, and
`flash_attn_ext_f16_load_mask`.

- **Tile types**: rename the tile-size struct to **`mma_tile_sizes_bf16`** - `mma_tile_sizes`
  at `fattn-mma-f16.cuh:1028` is at global scope, so a same-named struct would be a
  redefinition error in `fattn.cu` (AMPERE implies TURING on sm_80+). Types:
  `T_A_KQ/T_B_KQ/T_A_VKQ/T_B_VKQ = nv_bfloat162`;
  `T_C_KQ = tile<16,8,float>` (ncols==8) / `tile<16,16,float>` (wide);
  `T_C_VKQ = tile<16,8,float>` (ncols==8) / `tile<16,16,float>` (wide).
- **SRAM**: keep `extern __shared__ half2 tile_Q[]` and `half2 * tile_K/V` (same as f16);
  SRAM sizing is identical since nv_bfloat162 == half2 == 4 bytes.
- **Loads**: reuse `flash_attn_ext_f16_load_tile` / `flash_attn_ext_f16_load_mask`
  unchanged on the half2-typed buffers; reinterpret pointers as `nv_bfloat162` only at the
  `load_ldmatrix` / `load_ldmatrix_trans` sites. Q fill: multiply scale in f32, then
  `ggml_cuda_cast<nv_bfloat162>(float2)` (writes 4 bytes into tile_Q, same as the f16
  `scale_h2 * make_half2(...)` store).
- **P conversion**: `B[k] = get_bf16(KQ_C[k])` (wide) /
  `get_transposed(get_bf16(KQ_C[k]))` (ncols==8).
- **VKQ mma**: narrow reuses the mma at `1181`; wide uses the new CUDA mma (swapped A/B,
  as in the f16 kernel at `991`).
- **KQ_max_scale** applied to VKQ_C in plain f32 (the float branch at `899-906`, NOT the
  TURING half2 branch at `865-887`).
- **Combine write**: FP32 VKQ_C -> `nv_bfloat162` SRAM (`get_bf16`/`get_transposed` for
  ncols==8; direct scalar `__float2bfloat16` store for the wide path). Meta block
  (`KQ_cmr`, row sums) written via `float2*` cast as in f16 (the existing padding already
  covers it). Read-back via `ggml_cuda_cast<float2>(nv_bfloat162)` (which uses
  `__bfloat1622float2` on CUDA, `convert.cuh:48-49`).
- **Structure mirrors f16 for link safety**: the kernel `flash_attn_ext_bf16` is defined
  **unconditionally** with the body guarded `#if defined(AMPERE_MMA_AVAILABLE)` and a
  `NO_DEVICE_CODE;` fallback (mirrors the f16 kernel at `1703`). The case function
  `ggml_cuda_flash_attn_ext_mma_bf16_case` and the `DECL_FATTN_MMA_BF16_CASE` extern
  declarations (via `..._ALL_NCOLS2`) are **unguarded at global scope** (mirrors
  `1967-2033`), so every arch pass in `fattn.cu` links; the actual explicit instantiations
  live in the instance .cu files.
- **Launcher**: case function calls
  `launch_fattn<DV, ncols1, ncols2>(..., nbatch_fa, /*need_f16_K*/false,
  /*need_f16_V*/false, /*stream_k*/true, warp_size_host)`.
  Reuse the `GGML_CUDA_FATTN_MMA_CONFIG_CASE` Ampere config.

### 4. `ggml/src/ggml-cuda/fattn.cu`

- Add `BEST_FATTN_KERNEL_MMA_BF16` to the `best_fattn_kernel` enum (331-336).
- Routing in `ggml_cuda_get_best_fattn_kernel`: inside the existing
  `if (turing_mma_available(cc) && ...)` branch (already NVIDIA-only), add a bf16 check
  before `return BEST_FATTN_KERNEL_MMA_F16` (482): when `K->type == GGML_TYPE_BF16 &&
  V->type == GGML_TYPE_BF16 && bf16_mma_hardware_available(cc)` and head sizes are
  MMA-supported, return MMA_BF16.
- Add BF16 switch functions mirroring the f16 ones but with **no Turing/Volta branches**:
  `switch_ncols1` / `switch_ncols2` and the per-DKQ ncols2 dispatcher at `121-241`,
  including all special cases:
  - 192: ncols1 switch with ncols2 in {8, 16} (GLM-4.7 `gqa_ratio==20` path).
  - 320: ncols2 == 32 only.
  - 512: ncols2 in {2, 4, 8}.
  - 576: ncols2 in {4, 16, 32}.
  - 40 / 72: excluded from MMA dispatch entirely.
- Add an MMA_BF16 case to `ggml_cuda_flash_attn_ext` dispatch.
- `get_alloc_size`: MMA_BF16 with `need_f16_K/V = false` (zero extra, like vec-for-BF16).

### 5. Instance files

- Extend `template-instances/generate_cu_files.py`: add a `SOURCE_FATTN_MMA_BF16` template
  (includes `../fattn-mma-bf16.cuh`) plus `DECL_FATTN_MMA_BF16_CASE`; emit
  `fattn-mma-bf16-instance-ncols1_*-ncols2_*.cu` for the same (DKQ, DV, ncols1, ncols2)
  combos as the F16 instances, **mirroring the exact skip logic** at `generate_cu_files.py`
  lines 86-101 (head 40/72 skipped; 192->ncols2 in {8,16}; 320->32; 512->{2,4,8};
  576->{4,16,32}; 64/80/96/112/128/256 -> ncols2 in {1,2,4,8}).
- Run the generator. **No CMake change** required: `ggml/src/ggml-cuda/CMakeLists.txt`
  line 108 GLOBs `template-instances/fattn-mma*.cu`.

### 6. Validation (not run until explicitly requested)

- Build for sm_80/86/89/90/100/120a (incremental ninja of `ggml-cuda`).
- `test-backend-ops` `FLASH_ATTN_EXT` with K/V = BF16 for head sizes
  40/64/72/80/96/112/128/192/256/320/512/576.
- `llama-bench` pp1024/pp4096 BF16 vs F16 cache.
- `llama-perplexity -c 32768` BF16 vs F32/F16 baselines.
- Run on both RTX 3090 (sm_86) and RTX PRO 5000 Blackwell (sm_120a).

## Relevant files

- `ggml/src/ggml-cuda/fattn-mma-f16.cuh` - base to adapt; add `#pragma once` (line 1).
  Config struct (10-24), config macro (26), load_tile (363), load_mask (352),
  mma_tile_sizes (1027-1043), process_tile (1116), combine write (1567-1625), read-back
  (~1670), case fn (1896), DECL macro (1967-2033), kernel body guard (1703/1141).
- `ggml/src/ggml-cuda/mma.cuh` - add CUDA branch to wide BF16 mma (1257-1303), add
  `get_bf16` / `get_transposed` BF16 overloads. Existing: narrow BF16 mma (1181),
  f16-accum VKQ mma (970-1005), wide f16 mma (1196), get_half2 (713), get_transposed
  (722), load_ldmatrix / load_ldmatrix_trans (786/801/830/884).
- `ggml/src/ggml-cuda/fattn.cu` - enum (331-336), routing (461-482), switch_ncols1/2
  (8-34), per-DKQ dispatch (121-241), dispatch + get_alloc_size (550-562).
- `ggml/src/ggml-cuda/fattn-common.cuh` - `launch_fattn` (973, `need_f16_K/V`),
  f16_extra_data (47-85), `fattn_kernel_t`.
- `ggml/src/ggml-cuda/common.cuh` - `AMPERE_MMA_AVAILABLE` (282-284),
  `bf16_mma_hardware_available` (322), `ampere_mma_available` (352), `cp_async_available`
  (356).
- `ggml/src/ggml-cuda/convert.cuh` - `ggml_cuda_cast` (44-60), `to_bf16_cuda_t`.
- `ggml/src/ggml-cuda/template-instances/generate_cu_files.py` - extend to emit BF16
  instance files; mirror skip logic (86-101).
- `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-*.cu` - pattern to mirror.
- `ggml/src/ggml-cuda/CMakeLists.txt` - line 108 GLOB; no edit needed.
- `ggml/src/ggml-cuda/fattn-tile.cu` / `fattn-tile.cuh` - AMD BF16 path, do NOT modify.