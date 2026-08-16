# PLAN v2: NVIDIA native-BF16 Flash Attention MMA kernel

## Status

Second review round against the codebase (git HEAD). Supersedes `PLAN-bf16-fattn.md`.
Round 1 findings (include-guard, `mma_tile_sizes_bf16` rename, load-path reuse, NVIDIA-only
routing, unconditional case fn) are retained. Round 2 adds **two corrections that the v1
plan would get wrong** (VKQ_C sizing, KQ_max_scale scaling) plus several dispatch
clarifications. Line numbers refer to `fattn-mma-f16.cuh`, `mma.cuh`, `fattn.cu`,
`fattn-common.cuh`, `common.cuh`, `convert.cuh` at review time.

## Goal

Add a native BF16 Flash Attention tensor-core (MMA) kernel for NVIDIA GPUs (sm_80+) by
adapting the existing F16 MMA kernel. Eliminates the implicit BF16->F16 KV conversion and
uses FP32 VKQ accumulation (the F16 kernel's half2 VKQ accumulators at `mma.cuh:971` and
`mma.cuh:996` are the PPL-drift root cause). Mirrors AMD PR #26856, which names the NVIDIA
bf16-MMA kernel as follow-up work.

## Constraints (unchanged from v1)

- No `fma.rn.bf16x2` (element-wise); no scalar `cvt`+`fma` emulation.
- Do not modify AMD paths (`fattn-tile.cu`/`fattn-tile.cuh`) or the F16 MMA kernel
  behavior. The only edit to `fattn-mma-f16.cuh` is adding `#pragma once` at line 1.
- `V_DOT2_F32_BF16_AVAILABLE` stays undefined for NVIDIA.
- PTX core: `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`; A/B `nv_bfloat162`,
  accumulators FP32.

## Round 2 corrections (NEW - not in v1)

1. **VKQ_C array size** (`fattn-mma-f16.cuh:1182-1183`). The f16 TURING formula is
   `VKQ_C[cols_per_warp == 8 ? DV/T_C_VKQ::I : DV/(2*T_C_VKQ::J)]`. It relies on the
   invariant `2*T_C_VKQ::J == T_A_VKQ::I` (f16 wide: 2*8==16). For bf16 wide
   (`T_C_VKQ = tile<16,16,float>`, J=16) that formula gives `DV/32`, but the VKQ mma loop
   (`:974`) indexes `VKQ_C[i_VKQ_0/T_A_VKQ::I]` = `DV/16` tiles -> out of bounds.
   **The bf16 kernel must declare `T_C_VKQ VKQ_C[DV/T_C_VKQ::I]` for BOTH narrow and
   wide** (== DV/16 in both cases, since T_C_VKQ::I==16).
   Register impact for DV=256: wide = 16 tiles x 8 f32 = **128 regs** (f16: 64); narrow =
   16 tiles x 4 f32 = 64 regs (f16: 32). Doubling confirmed for both paths.
2. **KQ_max_scale VKQ scaling appears TWICE**, both half2-based under
   `TURING_MMA_AVAILABLE`: the normal path (`:865-887`) and the attention-sinks fixup path
   (`:1370-1392`). Neither the TURING half2 branch nor the AMD float branch
   (`:888-907`, `:1393-1412`) is directly reusable:
   - AMD float branch uses `KQ_max_scale[0]` (single col/thread on AMD) and bound
     `DV/T_C_VKQ::J` (wrong for bf16).
   - bf16 needs a f32 loop with per-KQ-column scale (`l%2` maps to col 0/1 on NVIDIA,
     `cols_per_thread==2`, see the `KQ_idx = l % 2` pattern at `:721`) and loop bound
     **`DV/T_C_VKQ::I`** in both places.
3. **Dispatch clarifications** (verified against `fattn.cu` / `generate_cu_files.py`):
   - 192 has **no extern declarations** in the f16 header (`:1978-2033`) - it is covered by
     implicit instantiation in `fattn.cu` (harmless; instance files provide the
     definitions). The `(ncols/16,16)` externs for ncols==8 (e.g. `(0,16)`) are
     dangling-but-harmless. Mirror the f16 extern list for bf16; 192 externs optional.
   - The 576 `gqa_ratio==20` dispatch (`fattn.cu:193-231`) has cc-dependent sub-branches.
     On Ampere (sm_80x, cc < ADA_LOVELACE) the `cc >= GGML_CUDA_CC_TURING` branch applies
     and can select **ncols2==32** when `Q->ne[1] <= 4 && K->ne[1] > 16384`. All referenced
     (ncols1,ncols2) combos exist in the instance matrix. Keep the same structure in the
     bf16 dispatcher.
   - `bf16_mma_hardware_available(cc)` is true on AMD CDNA/RDNA3+ and MThreads
     (`common.cuh:322-326`) but the kernel is NVIDIA-only
     (`AMPERE_MMA_AVAILABLE`, `common.cuh:282-284`). Routing MUST stay inside the
     `turing_mma_available(cc)` branch (NVIDIA-only, `common.cuh:348-350`).

## Verified facts (round 1 + 2)

- `fattn-mma-f16.cuh` has **no include guard**; `common.cuh`, `mma.cuh`,
  `fattn-common.cuh`, `convert.cuh` all have `#pragma once`.
- `mma_tile_sizes` is global-scope under `#if defined(TURING_MMA_AVAILABLE)`
  (`:1027-1043`). AMPERE implies TURING (800>=750) so a same-named struct in the bf16
  header included alongside = redefinition. Use `mma_tile_sizes_bf16`.
- Config getters (`ggml_cuda_fattn_mma_get_config*` `:38-246`, `get_cols_per_thread`
  `:329` returns 2, `get_cols_per_warp` `:337-344` returns 16 for Turing+) are global-scope
  static; available to the bf16 header via inclusion. `ggml_cuda_fattn_mma_get_config`
  host dispatcher (`:231-246`) picks ampere on cc>=800.
- `launch_fattn<DV,ncols1,ncols2>(ctx, dst, kernel, nwarps, nbytes_shared, nbatch_fa,
  need_f16_K, need_f16_V, stream_k, warp_size)` (`fattn-common.cuh:973-976`); f16 case
  passes `(..., true, true, true, warp_size_host)` (`:1962-1963`). bf16 passes
  `(..., false, false, true, warp_size_host)`. `get_f16_extra_data(dst,false,false)`
  returns zero extra (`fattn-common.cuh:47-85`); `launch_fattn` reads raw BF16
  `K->data`/`V->data` when `need_f16_K/V` false (`fattn-common.cuh:1022-1084`).
- `flash_attn_ext_f16_load_tile` (`:363-448`) is NOT type-templated (takes `const half2*`)
  but is a pure byte copy (`cp_async_cg_16` / `ggml_cuda_memcpy_1<16>`); `nv_bfloat162`
  and `half2` are both 4 bytes. Reuse unchanged on half2-typed SRAM.
- `flash_attn_ext_f16_load_mask` (`:450-528`) is mask/half-based, agnostic to K/V type.
  Reuse unchanged. All mask/slope/softmax/sinks code (`:694-755`, `:778+`, `:1340-1424`)
  operates on float/half and is K/V-type-agnostic.
- `load_ldmatrix`/`load_ldmatrix_trans` are templated on T (`mma.cuh:786,801,830,885`);
  they emit `ldmatrix.sync` which is bit-agnostic over the 16-bit lanes -> work for
  `nv_bfloat162` unchanged. Call sites must reinterpret the half2-typed SRAM pointers.
- mma overloads resolve by tile type, so the copied kernel body's mma calls need no
  explicit change: narrow KQ (`:628`) and VKQ (`:983`) -> existing narrow bf16 mma
  (`mma.cuh:1181-1194`); wide KQ swapped (`:636`) and wide VKQ swapped (`:991`) -> new
  CUDA wide bf16 mma.
- Wide f16 mma pattern to mirror for the CUDA bf16 branch: 2x `m16n8k16` using `Bxi[0]/`
  `Bxi[2]` then `Bxi[1]/Bxi[3]` (`mma.cuh:1196-1209`).
- `get_half2`/`get_transposed` (`mma.cuh:711-728`) - bf16 analogues needed:
  `get_bf16(tile<I,J,float>) -> tile<I,J/2,nv_bfloat162>` (via `__floats2bfloat162_rn`)
  and `get_transposed(tile<16,4,nv_bfloat162>) -> tile<8,8,nv_bfloat162>` (via
  `ggml_cuda_movmatrix`, bit-agnostic).
- `ggml_cuda_cast<nv_bfloat162>(float2)` = `{x.x, x.y}` (RN via `nv_bfloat16(float)`
  ctor) and `ggml_cuda_cast<float2>(nv_bfloat162)` = `__bfloat1622float2`
  (`convert.cuh:44-60`).
- `ggml_cuda_fattn_kv_type_supported` already includes GGML_TYPE_BF16
  (`fattn.cu:351`), so BF16 K/V already reach the MMA_F16 branch today (converted to F16);
  the bf16 routing is a clean insertion.
- `cp_async_available` (`common.cuh:356-358`) is NVIDIA Ampere+; `nstages` via
  `ggml_cuda_fattn_mma_get_nstages` (`fattn-mma-f16.cuh:348-359`).
- Kernel `flash_attn_ext_f16` (`:1703-1893`) is defined unconditionally with body guarded
  `#if defined(FLASH_ATTN_AVAILABLE) && (VOLTA||TURING||AMD...)` and `NO_DEVICE_CODE`
  fallback. Process_tile (`:1116-1701`) and iter (`:530-1025`) likewise. The bf16 kernel
  mirrors this with `#if defined(FLASH_ATTN_AVAILABLE) && defined(AMPERE_MMA_AVAILABLE)`.
- Case fn (`:1896-1964`) + DECL macros (`:1967-2033`) are global-scope, unguarded. bf16
  mirrors with `DECL_FATTN_MMA_BF16_CASE` / `..._ALL_NCOLS2`; instance files provide
  explicit instantiations.

## Design decisions (retained from v1, updated)

- **P stays in registers**: convert FP32 `KQ_C` -> `nv_bfloat162` B-tile once before the
  VKQ mma (`:921-933`): wide `B[k] = get_bf16(KQ_C[k])`, narrow
  `B[k] = get_transposed(get_bf16(KQ_C[k]))`.
- **VKQ in FP32**: `T_C_VKQ = tile<16,8,float>` (narrow) / `tile<16,16,float>` (wide).
  `T_C_KQ` same types. All other tiles (`T_A_KQ/T_B_KQ/T_A_VKQ/T_B_VKQ`) = `nv_bfloat162`.
- **SRAM is byte-identical to F16**: keep `extern __shared__ half2 tile_Q[]`,
  `half2 * tile_K/V`, `half` tile_mask. Reuse f16 load_tile/load_mask. Only reinterpret
  pointers as `nv_bfloat162` at ldmatrix / Q-fill / read-back sites.
- **Q fill** (`:1229-1237`): store `((nv_bfloat162*)tile_Q)[jc*stride_tile_Q + k] =
  ggml_cuda_cast<nv_bfloat162>(make_float2(scale*tmp.x, scale*tmp.y))` (scale in f32,
  slightly more precise than f16's half2 multiply - acceptable); zero-fill via
  `__floats2bfloat162_rn(0.0f, 0.0f)`.
- **Combine write** (`:1567-1624`): narrow -> `B = get_transposed(get_bf16(VKQ_C[...]))`
  stored as nv_bfloat162 into `tile_Q` (cast); wide -> f16 float branch (`:1611-1624`)
  with `half` -> `nv_bfloat16` scalar stores via `__float2bfloat16` (indexing
  `(k00+k1)/(T_C_VKQ::J/2)` tops out at DV/16, matching the bf16 VKQ_C size - verified).
  Meta block (`KQ_cmr` via `((float2*)tile_Q)[...]`, `:1433-1491`) unchanged.
- **Read-back** (`:1670`): `ggml_cuda_cast<float2>(((const nv_bfloat162*)tile_Q)[...])`.
- **KQ_max_scale**: f32 loops in BOTH the normal path and the sinks fixup path, bound
  `DV/T_C_VKQ::I`, per-column scale index derived from the f16 half2 semantics
  (col = `l%2`, `KQ_max_scale` array per thread).
- **Selection**: BF16 K/V + NVIDIA cc>=800; check inserted inside the
  `turing_mma_available(cc)` branch of `ggml_cuda_get_best_fattn_kernel`
  (`fattn.cu:461-483`), before `return BEST_FATTN_KERNEL_MMA_F16` (`:482`).
  `get_alloc_size` returns zero extra (like vec-for-BF16).
- **Config**: reuse `ggml_cuda_fattn_mma_get_config_ampere` (via the f16 header). The
  doubled VKQ register footprint (esp. wide DV=256 at 128 regs for VKQ_C) may push the
  256/512/576 configs' occupancy down - plan a retune pass.

## Steps

### 1. `ggml/src/ggml-cuda/fattn-mma-f16.cuh`
- Add `#pragma once` at line 1 (required so the bf16 header can include it).

### 2. `ggml/src/ggml-cuda/mma.cuh`
- Add `#elif defined(AMPERE_MMA_AVAILABLE)` CUDA branch to the wide BF16 mma
  (`:1257-1303`, currently AMD-only), mirroring the wide f16 mma (`:1196-1209`):
  `mma(tile<16,16,float,dl_d>&, tile<16,8,nv_bfloat162,dl_ab>&,
  tile<16,8,nv_bfloat162,dl_ab>&)` as 2x `m16n8k16 f32.bf16.bf16.f32` with
  `Bxi[0]/Bxi[2]` then `Bxi[1]/Bxi[3]`. Keep the AMD `#if` and final `#else NO_DEVICE_CODE`.
- Add `get_bf16(tile<I,J,float>) -> tile<I,J/2,nv_bfloat162>` via
  `__floats2bfloat162_rn`, and `get_transposed(tile<16,4,nv_bfloat162>) ->
  tile<8,8,nv_bfloat162>` via `ggml_cuda_movmatrix`; guard with
  `AMPERE_MMA_AVAILABLE` (references only from the AMPERE-guarded bf16 kernel).

### 3. New file `ggml/src/ggml-cuda/fattn-mma-bf16.cuh`
`#include "fattn-mma-f16.cuh"` (+ common/cp-async/mma/fattn-common transitively).
Global-scope structure mirrors f16:
- `mma_tile_sizes_bf16` struct (NOT `mma_tile_sizes`); types as in Design decisions.
- `flash_attn_ext_bf16_process_tile` / `flash_attn_ext_bf16_iter`: copies of the f16
  versions (`:530-1701`) with the substitutions from Design decisions; both bodies guarded
  `#if defined(AMPERE_MMA_AVAILABLE)` + `NO_DEVICE_CODE` fallback.
- `flash_attn_ext_bf16` kernel: copy of `:1703-1893` with `#if defined(FLASH_ATTN_AVAILABLE)
  && defined(AMPERE_MMA_AVAILABLE)` body guard; `__launch_bounds__` reuses the device
  config getters.
- `ggml_cuda_flash_attn_ext_mma_bf16_case` (unguarded, mirrors `:1896-1964`) calling
  `launch_fattn<DV,ncols1,ncols2>(..., nbatch_fa, false, false, true, warp_size_host)`;
  byte math stays `sizeof(half2)`; reuses host config getters + `GGML_CUDA_FATTN_MMA_CONFIG_CASE`.
- `DECL_FATTN_MMA_BF16_CASE` / `..._ALL_NCOLS2` + extern declarations mirroring
  `:1967-2033`.

### 4. `ggml/src/ggml-cuda/fattn.cu`
- Add `BEST_FATTN_KERNEL_MMA_BF16` to the enum (`:331-336`).
- Routing inside the turing branch (`:461-483`), before `return BEST_FATTN_KERNEL_MMA_F16`:
  `if (K->type == GGML_TYPE_BF16 && V->type == GGML_TYPE_BF16 && bf16_mma_hardware_available(cc))`
  return MMA_BF16.
- `ggml_cuda_flash_attn_ext_mma_bf16` dispatch (mirror `:113-242`) with:
  - switch_ncols2: GQA logic only, NO Volta block (`:67-89`), NO Turing-only-build/AMD
    condition (`:27`); the ncols1 chain is `<= 8/ncols2`, `<= 16/ncols2`, `<= 32/ncols2`,
    else `64/ncols2`.
  - per-DKQ dispatch with all special cases: 192 -> ncols1 switch, ncols2=16 if
    `gqa_ratio%16==0` else 8; 320 -> ncols2=32; 512 -> switch_ncols2; 576 -> the full
    cc-dependent gqa_ratio==20 logic + the `gqa_ratio%16==0` / else fallbacks.
  - 40/72 excluded by routing; the kernel's own `DKQ==192 && ncols2 not in {8,16}` guard
    (`:1743`) is copied.
- `ggml_cuda_flash_attn_ext` (`:570-585`): add MMA_BF16 case.
- `get_alloc_size` (`:536-568`): `case BEST_FATTN_KERNEL_MMA_BF16: break;` (need_f16 stays
  false).

### 5. Instance files
- Extend `generate_cu_files.py`: add `SOURCE_FATTN_MMA_BF16` (includes
  `../fattn-mma-bf16.cuh`) + `DECL_FATTN_MMA_BF16_CASE`; emit
  `fattn-mma-bf16-instance-ncols1_*-ncols2_*.cu` with the **same** skip logic as f16
  (`:86-101`: 40/72 skipped; 192->ncols2 in {8,16}; 320->32; 512->{2,4,8};
  576->{4,16,32}; others -> ncols2 in {1,2,4,8}).
- Run the generator. No CMake change (`CMakeLists.txt:108` GLOBs `fattn-mma*.cu`).

### 6. Validation (not run until explicitly requested)
- Build sm_80/86/89/90/100/120a (incremental ninja of `ggml-cuda`).
- `test-backend-ops` FLASH_ATTN_EXT BF16 K/V, head sizes 40/64/72/80/96/112/128/192/256/320/512/576.
- `llama-bench` pp1024/pp4096 BF16 vs F16 cache.
- `llama-perplexity -c 32768` BF16 vs F32/F16 baselines.
- Both RTX 3090 (sm_86) and RTX PRO 5000 Blackwell (sm_120a).
- If occupancy drops (esp. wide DV=256/512/576 VKQ_C), retune the bf16 config.

## Relevant files
- `fattn-mma-f16.cuh` - add `#pragma once`; base to adapt.
- `mma.cuh` - wide BF16 mma CUDA branch (1257-1303), `get_bf16`/`get_transposed`; existing
  narrow bf16 mma (1181).
- `fattn.cu` - enum (331), routing (461-483), dispatch (113-242), alloc size (536), ext (570).
- `fattn-common.cuh` - launch_fattn (973), f16_extra_data (47-85).
- `common.cuh` - AMPERE_MMA_AVAILABLE (282), bf16_mma_hardware_available (322),
  turing_mma_available (348), cp_async_available (356).
- `convert.cuh` - cast (44-60).
- `template-instances/generate_cu_files.py` - add bf16 instances.
- `fattn-tile.cu`/`fattn-tile.cuh` - AMD BF16 path, do NOT modify.