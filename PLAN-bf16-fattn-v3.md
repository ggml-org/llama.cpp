# PLAN v3: Native BF16 Flash Attention MMA Kernel for NVIDIA (sm_80+)

Supersedes `PLAN-bf16-fattn-v2.md` and `PLAN-bf16-fattn.md`.

Status: 3 verification rounds done. This document is the authoritative implementation plan.

## 1. Goal

Add a native BF16 Flash Attention Tensor-Core kernel (`flash_attn_ext_bf16`, MMA-based) for
NVIDIA GPUs (sm_80+ / AMPERE) by adapting `fattn-mma-f16.cuh`. It eliminates the implicit
BF16->F16 KV conversion and accumulates VKQ in FP32.

The F16 kernel's PPL drift is rooted in the FP16 VKQ accumulators
(`mma(tile<16,8,half2>& D, ...)` at `mma.cuh:995` and `mma(tile<16,4,half2>& D, ...)` at
`mma.cuh:970`). The BF16 kernel keeps KQ in FP32 (`tile<16,8,float>` / `tile<16,16,float>`),
the same as F16, but accumulates VKQ in FP32 too, using `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.

## 2. CORRECTIONS vs v2 (round-3 findings) - READ FIRST

These three items override v2:

### 2.1 Wide VKQ scaling column mapping is `(l/2) % 2`, NOT `l % 2`

The KQ_max_scale VKQ scaling loop (both the normal path at `fattn-mma-f16.cuh:865-887` and the
sinks-fixup path at `:1370-1421`) must use:

- narrow (`cols_per_warp == 8`, `T_C_VKQ = tile<16, 8, float>`):
  `VKQ_C[i].x[l] *= KQ_max_scale[l % 2]`
- wide (`cols_per_warp == 16`, `T_C_VKQ = tile<16, 16, float>`):
  `VKQ_C[i].x[l] *= KQ_max_scale[(l/2) % 2]`

Both loops use the bound `DV/T_C_VKQ::I` and are scalar f32 loops (replace the half2 code).

Why: each BF16 float element `l` of VKQ_C corresponds to one F16 half2 lane `l` of the
equivalent F16 tile. Narrow F16 scales lane 0 -> scale[0], lane 1 -> scale[1], so element
`l -> scale[l%2]`. Wide F16 scales half2 element `l0+col` by scale[col] on both lanes, so
lane `l` (in half2 element `l/2`) -> scale[(l/2)%2]. This is consistent with the softmax
`KQ_idx = (l/2) % 2` used for the wide KQ_C tile (`fattn-mma-f16.cuh:798,840`) and with
`get_i(l) = (((l/2)%2)*8) + threadIdx.x/4` for `tile<16,16,float>`: the KQ column of a row is
determined by the row's 8-row block, i.e. `(l/2)%2`.

### 2.2 No manual accumulator zeroing needed

`KQ_C` and `VKQ_C` are automatically zeroed by the tile struct member initializer
`T x[ne] = {0}` (`mma.cuh:228`, NVIDIA `DATA_LAYOUT_I_MAJOR` branch). The `mma` overloads use
`"+r"` accumulate semantics, and the arrays `T_C_KQ KQ_C[...]` / `T_C_VKQ VKQ_C[...]` are
value-initialized through this member initializer. No zeroing loops are required in the BF16
kernel (confirmed: the F16 kernel has none either).

### 2.3 Wide BF16 mma guard placement and structure

The wide BF16 mma overload `mma(tile<16,16,float>& D, tile<16,8,nv_bfloat162>& A,
tile<16,8,nv_bfloat162>& B)` exists at `mma.cuh:1257` but is AMD-only (`#else NO_DEVICE_CODE`).
Add a CUDA branch at the TOP of that overload, mirroring the F16 wide mma at `mma.cuh:1196-1255`:

```cpp
#if defined(AMPERE_MMA_AVAILABLE)
    // 2x m16n8k16 f32.bf16.bf16.f32 with Bxi[0]/Bxi[2], then Bxi[1]/Bxi[3] (same Axi).
    // (no Turing m8n8k8 fallback needed; BF16 mma requires sm_80+)
#elif defined(AMD_WMMA_AVAILABLE)
    ... existing RDNA code ...
#elif defined(AMD_MFMA_AVAILABLE)
    ... existing CDNA code ...
#else
    GGML_UNUSED_VARS(D, A, B);
    NO_DEVICE_CODE;
#endif
```

Use `AMPERE_MMA_AVAILABLE` (not `TURING_MMA_AVAILABLE`) to match the narrow BF16 mma guard
(`mma.cuh:1183`).

## 3. Complete implementation steps

### Step 1 - fattn-mma-f16.cuh

Add `#pragma once` as the first line. It currently has NO include guard. The BF16 header
includes it to reuse: config getters (`ggml_cuda_fattn_mma_get_*`, `:38-246`),
`get_cols_per_warp` (16 on Turing+), `get_cols_per_thread` (2),
`flash_attn_ext_f16_load_tile` (`:364`, half2-typed byte copy, type-agnostic),
`flash_attn_ext_f16_load_mask` (`:450`, half-based, type-agnostic), the nbatch helpers and the
combine helpers. NO other change to this file.

### Step 2 - mma.cuh

All additions are additive; existing code untouched:

1. Wide BF16 mma CUDA branch at `:1257` (see 2.3).
2. `ggml_cuda_movmatrix(const nv_bfloat162 x)` overload, mirroring the half2 wrapper at
   `:63-67`:
   ```cpp
   static __device__ __forceinline__ nv_bfloat162 ggml_cuda_movmatrix(const nv_bfloat162 x) {
       nv_bfloat162 ret;
       *((int *) &ret) = ggml_cuda_movmatrix(*((const int *) &x));
       return ret;
   }
   ```
3. `get_bf16` (mirror `get_half2` at `:711-720`, same guard region):
   ```cpp
   template <int I, int J>
   static __device__ __forceinline__ tile<I, J/2, nv_bfloat162> get_bf16(const tile<I, J, float> & tile_float) {
       tile<I, J/2, nv_bfloat162> ret;
   #pragma unroll
       for (int l0 = 0; l0 < tile_float.ne; l0 += 2) {
           ret.x[l0/2] = ggml_cuda_cast<nv_bfloat162>(make_float2(tile_float.x[l0 + 0], tile_float.x[l0 + 1]));
       }
       return ret;
   }
   ```
   (`ggml_cuda_cast<nv_bfloat162>(float2)` = `{x.x, x.y}` = RN conversion, `convert.cuh:54-60`.)
4. `get_transposed` overload for BF16 (mirror `:722-728`):
   ```cpp
   static __device__ __forceinline__ tile<8, 8, nv_bfloat162> get_transposed(const tile<16, 4, nv_bfloat162> & t) {
       tile<8, 8, nv_bfloat162> ret;
       ret.x[0] = ggml_cuda_movmatrix(t.x[0]);
       ret.x[1] = ggml_cuda_movmatrix(t.x[1]);
       return ret;
   }
   ```

No other mma.cuh changes. The narrow BF16 mma (`mma.cuh:1181`) and the narrow/tile helpers
already exist.

### Step 3 - new file fattn-mma-bf16.cuh

Guard with `#pragma once`, include `mma.cuh`, `fattn-mma-f16.cuh`, `cp-async.cuh`,
`fattn-common.cuh`. Only compiled for CUDA.

Define `mma_tile_sizes_bf16<DV, ncols>` mirroring `mma_tile_sizes` (`fattn-mma-f16.cuh:1027-1101`)
but with `nv_bfloat162` in place of `half2` for the A/B tiles (guarded
`#if defined(AMPERE_MMA_AVAILABLE)`):

- narrow (`ncols < 16`): `T_A_KQ = tile<16,8,nv_bfloat162, I_MAJOR_MIRRORED>`,
  `T_B_KQ = tile<8,8,nv_bfloat162, I_MAJOR_MIRRORED>`, `T_C_KQ = tile<16,8,float, I_MAJOR>`,
  `T_A_VKQ = tile<16,8,nv_bfloat162, I_MAJOR_MIRRORED>`,
  `T_B_VKQ = tile<8,8,nv_bfloat162, I_MAJOR_MIRRORED>`, `T_C_VKQ = tile<16,8,float, I_MAJOR>`
- wide (`ncols == 16/32`): `T_A_KQ = T_B_KQ = T_A_VKQ = T_B_VKQ = tile<16,8,nv_bfloat162, I_MAJOR_MIRRORED>`,
  `T_C_KQ = T_C_VKQ = tile<16,16,float, I_MAJOR>`

Then `flash_attn_ext_bf16_iter` and `flash_attn_ext_bf16_process_tile` and
`flash_attn_ext_bf16` kernel: the full F16 kernel bodies adapted per the rules below.

Adaptation rules (BF16 vs F16) - ALL of:

1. The mma calls resolve to the BF16 overloads (`mma.cuh:1181` narrow, `:1257` wide).
   The CUDA wide KQ mma uses the swapped form `mma(KQ_C[...], Q_B[...], K_A)` (CUDA column-major),
   identical structure to `fattn-mma-f16.cuh:636/663/991`.
2. `B[k]` conversion (softmax P -> VKQ B operand, `fattn-mma-f16.cuh:922-931`):
   - narrow: `B[k] = get_transposed(get_bf16(KQ_C[k]));`
   - wide:   `B[k] = get_bf16(KQ_C[k]);`
3. `KQ_max_scale` VKQ scaling: replace the half2 branches (`:865-887` normal, `:1370-1421`
   sinks) with scalar f32 loops using the mapping from section 2.1 and bound `DV/T_C_VKQ::I`.
4. `KQ_C` declaration (iter, `:576-582`): reuse the TURING formula unchanged
   `T_C_KQ KQ_C[nbatch_fa/(np*(cols_per_warp == 8 ? T_C_KQ::I : T_C_KQ::J))]` (type-agnostic).
5. `VKQ_C` declaration (process_tile, `:1182-1190`):
   `T_C_VKQ VKQ_C[DV/T_C_VKQ::I]` for BOTH narrow and wide (round-2 finding; the F16 formula
   `DV/(2*T_C_VKQ::J)` relies on `2*T_C_VKQ::J == T_A_VKQ::I` which does not hold for the BF16
   wide tile and would under-allocate DV/32 vs the actual DV/16). This doubles VKQ register
   usage: narrow = 64 f32 regs (F16: 32), wide = 128 (F16: 64) for DV=256. Config retune
   flagged (step 5).
6. Softmax (max/exp/rowsum, mask add, `:687-863`): unchanged - all scalar f32 ops on
   `tile<16,8,float>` / `tile<16,16,float>` KQ_C, including `KQ_idx = l % 2` (narrow) and
   `(l/2) % 2` (wide), and the shfl_xor reductions (narrow offsets 16..4, wide 2..1). The wide
   mask add reads `half2` from `tile_mask` (`:778`) - unchanged (mask stays half-based).
7. Load paths: `tile_Q/tile_K/tile_V` stay `extern __shared__ half2[]` (SRAM type-agnostic).
   K/V load via `flash_attn_ext_f16_load_tile` (byte copy). Q is loaded as half2
   (`make_half2(tmp.x, tmp.y) * scale_h2`) at `:1202-1231`; for BF16 reinterpret the same
   bytes as `nv_bfloat162` when reading with ldmatrix / filling Q_B. `load_ldmatrix` is
   T-templated and bit-agnostic. OOB zero-fill uses `make_half2(0.0f, 0.0f)` - unchanged.
8. Q_in_reg load (`:1245-1252`): `load_ldmatrix(Q_B[k0/T_B_KQ::J], ...)` with
   `T_B_KQ = tile<16,8,nv_bfloat162,...>` - unchanged code, new type.
9. Combine-write (`:1566-1625`):
   - narrow (`cols_per_warp == 8`): replace
     `get_transposed(VKQ_C[...])` (F16, VKQ_C is already half2) with
     `get_transposed(get_bf16(VKQ_C[...]))` and write `B.x[l]` (nv_bfloat162) into `tile_Q`
     reinterpreted as `nv_bfloat162*`. The `(k00+k1)/T_B_KQ::J` indexing gives DV/16, matching
     the BF16 VKQ_C size.
   - wide: use the float branch (`:1611-1624`) but write with explicit conversion:
     `tile_Q_h[k] = __float2bfloat16(VKQ_C[...].x[l])` where `tile_Q_h = (nv_bfloat16 *) tile_Q`.
     The indexing `(k00+k1)/(T_C_VKQ::J/2)` tops out at DV/16, matching the BF16 VKQ_C size.
     Note: `float->half` works implicitly, `float->nv_bfloat16` does NOT - the explicit cast
     is required.
10. Meta write / combine (`:1422-1560`): all float2/scalar ops, unchanged.
11. Read-back and final write: reuse the F16 kernel's structure; final half2 -> float2 output
    conversion via `ggml_cuda_cast<float2>(half2)` becomes
    `ggml_cuda_cast<float2>(nv_bfloat162)` (`convert.cuh:44-45`, `__bfloat1622float2`).

Kernel/case/extern scaffolding (mirror `fattn-mma-f16.cuh:1703-2033`):

- `flash_attn_ext_bf16` kernel defined unconditionally (no `#if`) with the body guarded
  `#if defined(FLASH_ATTN_AVAILABLE) && defined(AMPERE_MMA_AVAILABLE)` and a
  `NO_DEVICE_CODE` fallback, so it compiles on all backends with no link errors.
- `flash_attn_ext_bf16` case function (`case_fattn_mma_bf16`) defined unconditionally; inside,
  guard the per-case dispatch with the same condition, calling the kernel.
- `DECL_FATTN_MMA_BF16_CASE(DKQ, DV, ncols1, ncols2, nwarps)` and
  `DECL_FATTN_MMA_BF16_ALL_NCOLS2(...)` macros + extern declarations at global scope,
  mirroring `:1967-2033`.
- 192 has no externs (implicit instantiation from fattn.cu); the `(ncols/16,16)` externs for
  ncols==8 are dangling but harmless (linker discards).

### Step 4 - fattn.cu

1. enum `GGML_CUDA_FATTN_KERNEL_MMA_BF16` in the middle of the list (`:331-336`).
2. Routing (`:461-483`): insert AFTER the `turing_mma_available(cc)` check (NVIDIA-only gate)
   and BEFORE the `return GGML_CUDA_FATTN_KERNEL_MMA_F16;`:
   ```cpp
   if (ggml_cuda_should_use_bf16_kernel(...)) {
       return GGML_CUDA_FATTN_KERNEL_MMA_BF16;
   }
   ```
   This is intentionally inside the NVIDIA-only branch because
   `bf16_mma_hardware_available` (`common.cuh:322`) is also true on AMD CDNA/RDNA3+ and would
   otherwise misroute AMD to a CUDA-only kernel.
3. `switch_ncols1` / `switch_ncols2` mirror the F16 versions minus the Volta block (`:67-89`)
   and the Turing-build/AMD condition (`:27`); special DKQ cases identical to F16
   (192 -> ncols2 in {8,16}, 320 -> 32, 512 -> {2,4,8}, 576 -> {4,16,32}; 40/72 excluded).
4. Dispatch per DKQ (`:113-242`): call the BF16 case function / kernel for all four KV
   configurations exactly like the F16 dispatch. 576 with gqa_ratio==20 is cc-dependent: on
   Ampere (`cc >= GGML_CUDA_CC_TURING`) the ncols2==32 sub-branch applies.
5. `get_alloc_size` (`:536-568`): MMA_BF16 -> zero extra (like vec-for-BF16).
6. `launch_fattn<DV,ncols1,ncols2>(..., nbatch_fa, /*need_f16_K*/false, /*need_f16_V*/false,
   /*stream_k*/true, warp_size_host)` - mirrors the MMA_F16 call.
7. Add `#include "fattn-mma-bf16.cuh"` next to the F16 include.

### Step 5 - generate_cu_files.py and config retune

Extend `template-instances/generate_cu_files.py` (skip logic at `:86-101`) to emit
`fattn-mma-bf16-instance-*.cu` for the same (DKQ, DV, ncols1, ncols2) matrix as the F16
instances, skipping non-AMPERE architectures. No CMake change needed: `fattn-mma*.cu` is
GLOBbed (`CMakeLists.txt:108`).

Register pressure doubles for VKQ_C; retune `mma_tile_sizes_bf16`/`get_nstages`/`get_nbatch_*`
if needed (report benchmark, do not guess).

## 4. Scope / constraints

- Do NOT modify AMD paths (`fattn-tile.cu`/`fattn-tile.cuh`) or the F16 kernel behavior.
- `V_DOT2_F32_BF16_AVAILABLE` stays undefined for NVIDIA (no element-wise fma fallback).
- Only supported types: KV converted to BF16 (f32 K/V dequantize path). The BF16 kernel is
  selected only when `k->type == GGML_TYPE_F32` (or quantized K with the bf16 vec/MMA gate)
  and KV conversion target is BF16.
- Validation (test-backend-ops, llama-bench, llama-perplexity) only when explicitly requested.

## 5. Reference map (verified line numbers)

- `fattn-mma-f16.cuh`: pragma once (1), config getters (38-246), load_tile (364-448),
  load_mask (450-528), iter (533-1022), mma_tile_sizes (1027-1101), process_tile (1116-1701),
  kernel (1703-1893), case fn (1896-1964), DECL/externs (1967-2033).
- KQ_max_scale VKQ scaling sites: normal (865-887), sinks (1370-1421).
- Softmax KQ_idx: narrow `l%2` (721,747), wide `(l/2)%2` (798,840); shfl offsets (730-735,
  806-830, 1323-1346).
- B-array conversion (922-931); VKQ mma (983-1008); Q fill (1202-1231); Q_B load (1245-1252);
  combine-write (1566-1625); meta (1422-1560).
- `mma.cuh`: movmatrix (28-67), get_half2/get_transposed (711-728), load_ldmatrix (785-900),
  narrow BF16 mma (1181-1194), wide F16 mma (1196-1255), wide BF16 mma AMD-only (1257-1303).
- `fattn.cu`: enum (331-336), routing (461-483), switch fns (8-34/36-111), dispatch (113-242),
  get_alloc_size (536-568), ext (570-585).
- `fattn-common.cuh`: launch_fattn (973-976), f16_extra_data (47-85).
- `common.cuh`: AMPERE_MMA_AVAILABLE (282-284), bf16_mma_hardware_available (322),
  turing_mma_available (348-350), tile `T x[ne] = {0}` (228).
- `convert.cuh`: bf16 casts (44-60).
- `generate_cu_files.py`: skip logic (86-101).

## 6. Test matrix

- sm_86 (RTX 3090): narrow (ncols<=8) and wide (ncols 16/32) paths, 192/256/320/512/576 head
  dims, GQA, sinks, multi-stage (nstages>1), MLA (V_is_K_view).
- sm_120a (RTX PRO 5000 Blackwell): same, confirms AMPERE path.
- Compare vs MMA_F16 and vec kernels: correctness (test-backend-ops) and PPL
  (llama-perplexity) - BF16 MMA must match BF16 vec precision (FP32 VKQ).