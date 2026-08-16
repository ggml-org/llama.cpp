# PLAN v5: Native BF16 Flash Attention MMA Kernel for NVIDIA (sm_80+)

Delta over `PLAN-bf16-fattn-v3.md` + `PLAN-bf16-fattn-v4.md`. This file records the round-5
deep-dive results: one CRITICAL routing correction, the confirmation that no
`fattn-common.cuh` change is needed, and the exact host-side wiring. v3+v4+v5 supersede v2/v1.

## 1. CRITICAL correction - MMA_BF16 is only valid for natively-BF16 K/V

The KV conversion infrastructure produces F16 buffers ONLY. In `launch_fattn`
(`fattn-common.cuh:1005-1084`):

- `ggml_cuda_flash_attn_ext_get_f16_extra_data(KQV, need_f16_K, need_f16_V)` allocates
  buffers only when `need_f16_X && X->type != GGML_TYPE_F16`.
- Conversion uses `ggml_get_to_fp16_cuda` / `ggml_get_to_fp16_nc_cuda` - F16 targets only.
- There is NO `to_bf16` path in the FA launch. `need_f16_K == false` means the kernel
  receives the RAW K data (whatever its type); it does NOT get a BF16-converted copy.

Conclusion: **MMA_BF16 must be routed ONLY when `K->type == GGML_TYPE_BF16 && V->type ==
GGML_TYPE_BF16`** (native BF16 cache). This matches the vec kernel's existing BF16 behavior
(native BF16 is consumed directly; F32 is converted to F16, never to BF16). With native
BF16, `need_f16_K = need_f16_V = false` and the kernel reads the raw BF16 data directly,
so `fattn-common.cuh` needs NO changes.

Note: `GGML_TYPE_BF16` is already accepted by `ggml_cuda_fattn_kv_type_supported`
(`fattn.cu:351`), so BF16 caches already reach the turing branch today and currently get
MMA_F16 (bf16->f16 conversion + half2 accumulation). Our gate intercepts them.

## 2. Host-side wiring (all in `fattn.cu`, exact locations)

### 2.1 Enum (`:331-336`)

```cpp
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
    BEST_FATTN_KERNEL_MMA_BF16 = 500,
};
```

### 2.2 Routing (`:461-483`) - the corrected gate

Insert inside the `turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72` branch
(`:461`), AFTER the vec block (`:462-481`), immediately before `return
BEST_FATTN_KERNEL_MMA_F16;` (`:482`):

```cpp
if (K->type == GGML_TYPE_BF16 && V->type == GGML_TYPE_BF16 && bf16_mma_hardware_available(cc)) {
    return BEST_FATTN_KERNEL_MMA_BF16;
}
```

- `bf16_mma_hardware_available(cc)` (`common.cuh:322-326`) = NVIDIA && cc >= AMPERE. Within
  the turing branch the NVIDIA part is redundant (turing_mma_available already requires it);
  the function documents intent and matches mmq usage.
- Placement preserves the existing preference order: VEC still wins for its conditions
  (Ada+ single-prompt, etc.) on BF16 caches; MMA_BF16 replaces MMA_F16 for everything else.
- 40/72 head sizes are already excluded by the branch condition.

### 2.3 get_alloc_size (`:550-562`)

```cpp
switch (kernel) {
    case BEST_FATTN_KERNEL_TILE:
    case BEST_FATTN_KERNEL_MMA_F16:
        need_f16_K = true;
        need_f16_V = true;
        break;
    case BEST_FATTN_KERNEL_MMA_BF16:
        break; // native BF16 K/V: no conversion buffers
    case BEST_FATTN_KERNEL_VEC:
        ...
}
```

### 2.4 Main entry (`:570-585`)

```cpp
case BEST_FATTN_KERNEL_MMA_BF16:
    ggml_cuda_flash_attn_ext_mma_bf16(ctx, dst);
    break;
```

plus `#include "fattn-mma-bf16.cuh"` at the top (file is in the same dir, no path change).

### 2.5 Case function (new `fattn-mma-bf16.cuh`, mirror of `fattn-mma-f16.cuh:1895-1964`)

- Same `switch (cc)`, same per-cc config (reuse `fattn_mma_config` tables unchanged).
- `nbytes_shared_*` in `sizeof(half2)` units - byte-identical for bf16 (half2 and
  nv_bfloat162 are both 4 bytes).
- `V_is_K_view = (DKQ == 576)` unchanged.
- `launch_fattn<DV,ncols1,ncols2>(ctx, dst, kernel, nwarps, nbytes_shared_total, nbatch_fa,
  false, false, true, warp_size_host)` - `need_f16_K=false, need_f16_V=false, stream_k=true`.

## 3. Confirmations (round-5, line-verified)

- **`launch_fattn` tail is type-agnostic** (`:972-1189`): grid/occupancy computation
  (`:1111-1185`), stream-k selection (`:1120-1150`), mask-scan (`:1094-1109`, mask is always
  F16), KV_max kernel - no half2/bf16 assumptions. `GGML_ASSERT(K->nb[0] ==
  ggml_element_size(K))` (`:994`) holds for contiguous BF16 (nb[0] = 2).
- **Kernel signature is type-agnostic** (`fattn-mma-f16.cuh:1703-1704`): `char *`/byte
  pointers; `stride_K = nb11 / sizeof(half2)` is in 4-byte units, valid for nv_bfloat162 too.
- **Kernel body guard for BF16** (mirror of `:1703-1710`): use
  `#if defined(FLASH_ATTN_AVAILABLE) && defined(AMPERE_MMA_AVAILABLE)` (NOT the f16 turing
  guard) with the `#else NO_DEVICE_CODE` fallback so the always-compiled `__global__`
  template links on all builds.
- **SRAM** stays `extern __shared__ half2[]`; reinterpret as `nv_bfloat162 *` only at the
  three touch points (Q fill, ldmatrix KQ/VKQ operands, read-back).
- **DECL macros** (mirror of `:1967-2033`): `DECL_FATTN_MMA_BF16_CASE` (explicit
  instantiation) + `DECL_FATTN_MMA_BF16_CASE_ALL_NCOLS2` (extern declarations); the
  generated instance .cu files provide the definitions.
- **switch_ncols1/switch_ncols2** (mirror of `fattn.cu:8-111`): same `#if defined(...)`
  guards minus the Volta block (`:67-89`) and minus the `TURING_MMA_AVAILABLE`-only build
  condition (bf16 instances use `AMPERE_MMA_AVAILABLE`); `DECL_*` sets match the f16 shape
  (ncols2 = 1/2/4/8, 8/16, 32, 16/32 for 192, 32 for 320, 2/4/8 for 512, 4/16/32 for 576).

## 4. generate_cu_files.py (`:78-103`) - extend, do not restructure

Mirror the mma-f16 emission block with new `SOURCE_FATTN_MMA_BF16_START` and
`SOURCE_FATTN_MMA_BF16_CASE` templates and the SAME rules:

- ncols in {8, 16, 32, 64}; ncols2 in {1, 2, 4, 8, 16, 32}; skip ncols2 > ncols.
- Skip head sizes 40, 72.
- 192 -> ncols2 in (8, 16); 320 -> 32; 512 -> (2, 4, 8); 576 -> (4, 16, 32);
  ncols2 in (16, 32) only for those niche sizes.
- CMake needs NO change (`fattn-mma*.cu` glob at `CMakeLists.txt:108` picks up the new files).

## 5. Everything else stands (v3 + v4, authoritative)

- mma.cuh additions: wide-BF16 AMPERE branch (v3 2.3), `ggml_cuda_movmatrix(nv_bfloat162)`,
  `get_bf16`, `get_transposed(tile<16,4,nv_bfloat162>)`.
- FP32 KQ_C/VKQ_C; `VKQ_C[DV/T_C_VKQ::I]` both narrow and wide; no manual zeroing.
- Wide KQ_max_scale mapping `(l/2) % 2`, narrow `l % 2`.
- Conversion-point checklist (v4 section 4): Q fill, B-array, KQ_max_scale, combine-write,
  read-back; softmax/mask/meta/fixup unchanged.
- `#pragma once` added at `fattn-mma-f16.cuh:1`.
- Constraints: no AMD-path changes, no fma.rn.bf16x2, no scalar cvt emulation,
  `V_DOT2_F32_BF16_AVAILABLE` stays undefined for NVIDIA.

## 6. Implementation steps

1. `fattn-mma-f16.cuh:1`: add `#pragma once`.
2. `mma.cuh`: add the three helpers + wide-BF16 AMPERE branch (v3 2.3).
3. Create `fattn-mma-bf16.cuh` (clone of f16 with all v3/v4/v5 deltas applied).
4. `fattn.cu`: enum, routing gate, get_alloc_size, main entry, include.
5. `generate_cu_files.py`: add BF16 templates + emit; run the script.
6. Build (NVIDIA, incl. sm_86 + sm_120a) and iterate on compile errors.

Validation (NOT actioned until explicitly requested): test-backend-ops with BF16 KV types,
llama-bench / llama-perplexity with `--cache-type-k bf16 --cache-type-v bf16` vs the F16
baseline on RTX 3090 (sm_86) and RTX PRO 5000 (sm_120a).