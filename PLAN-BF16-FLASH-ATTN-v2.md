# PLAN v2: Native BF16 Flash Attention MMA Kernel for NVIDIA (sm_80+)

Complete, self-contained implementation plan (supersedes `PLAN-BF16-FLASH-ATTN.md`). All
line numbers verified against the current tree in a final deep-dive round.

Changes vs V1 are marked [NEW]:

- [NEW] routing gate uses `ampere_mma_available(cc)` (compiled-arch based), not
  `bf16_mma_hardware_available(cc)` (hardware based), to avoid dispatching to an empty
  (NO_DEVICE_CODE) kernel on a restricted-arch build.
- [NEW] exact `switch_ncols1` guard structure for bf16.
- [NEW] exact kernel-body skip checks to keep / drop for bf16.
- [NEW] case function mirrors the `cudaFuncSetAttribute` shared-memory-limit raise.
- [NEW] DECL macro instance list matched exactly to the f16 externs (including the 192
  implicit-instantiation detail and the "dangling" ncols1=0 externs).
- [NEW] quantified register-pressure risk for the wide DV=512 path (spilling).

## 1. Goal and motivation

Add `flash_attn_ext_bf16`, an MMA-based Flash Attention kernel that consumes a native BF16
KV cache directly and accumulates the VKQ result in FP32.

The existing F16 MMA kernel (`fattn-mma-f16.cuh`) drifts on PPL because both VKQ
accumulators are FP16 (`mma(tile<16,8,half2>& D, ...)` and `mma(tile<16,4,half2>& D, ...)`,
m16n8k16.f16.f16.f16.f16). Today a `--cache-type-k bf16 --cache-type-v bf16` cache is
converted BF16->F16 and then accumulated in FP16, losing precision twice. The BF16 kernel
keeps KQ in FP32 (same as F16) and accumulates VKQ in FP32 using
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.

## 2. Scope: natively-BF16 KV cache ONLY

The KV conversion infrastructure in `launch_fattn` (`fattn-common.cuh:1005-1084`) produces
F16 buffers ONLY:

- `ggml_cuda_flash_attn_ext_get_f16_extra_data(KQV, need_f16_K, need_f16_V)` allocates
  buffers only when `need_f16_X && X->type != GGML_TYPE_F16`.
- Conversion uses `ggml_get_to_fp16_cuda` / `ggml_get_to_fp16_nc_cuda` - F16 targets only.
- There is NO `to_bf16` path in the FA launch. `need_f16_K == false` means the kernel
  receives the RAW K data (whatever its type); it does NOT get a BF16-converted copy.

Therefore the kernel is routed ONLY when `K->type == GGML_TYPE_BF16 && V->type ==
GGML_TYPE_BF16` (a native BF16 cache). This matches the vec kernel's existing BF16 behavior
(native BF16 consumed directly; F32 is converted to F16, never to BF16). Consequently
`fattn-common.cuh` needs NO changes: the case function calls
`launch_fattn(..., need_f16_K=false, need_f16_V=false, stream_k=true)` and the kernel reads
the raw BF16 data.

`GGML_TYPE_BF16` is already accepted by `ggml_cuda_fattn_kv_type_supported` (`fattn.cu:351`),
so BF16 caches already reach the tensor-core branch today and currently get MMA_F16
(BF16->F16 conversion + FP16 accumulation). The new gate intercepts them.

Constraints:

- Do NOT modify AMD paths (`fattn-tile.cu` / `fattn-tile.cuh`) or the F16 kernel behavior.
- No element-wise `fma.rn.bf16x2`, no scalar cvt + fma emulation. `V_DOT2_F32_BF16_AVAILABLE`
  stays undefined for NVIDIA.
- `mma.cuh` changes are purely additive.
- The only edit to `fattn-mma-f16.cuh` is adding `#pragma once` as the first line (it has no
  include guard today).

## 3. Tile types (verified against `mma_tile_sizes`, `fattn-mma-f16.cuh:1027-1043`)

Mirror the F16 Turing/Ampere sets exactly, replacing `half2` with `nv_bfloat162` for the A/B
input tiles and with FP32 tiles for the VKQ accumulator. The data_layout enums are UNCHANGED
(plain `DATA_LAYOUT_I_MAJOR` default; the MIRRORED layouts belong only to Volta/RDNA3).

| set | f16 reference | bf16 (new) |
|---|---|---|
| narrow T_A_KQ  | tile<16, 8, half2>            | tile<16, 8, nv_bfloat162> |
| narrow T_B_KQ  | tile< 8, 8, half2>            | tile< 8, 8, nv_bfloat162> |
| narrow T_C_KQ  | tile<16, 8, float>            | tile<16, 8, float> (same) |
| narrow T_A_VKQ | tile<16, 8, half2>            | tile<16, 8, nv_bfloat162> |
| narrow T_B_VKQ | tile< 8, 8, half2>            | tile< 8, 8, nv_bfloat162> |
| narrow T_C_VKQ | tile<16, 4, half2> (fp16 acc) | tile<16, 8, float> (fp32 acc) |
| wide T_A_KQ    | tile<16, 8, half2>            | tile<16, 8, nv_bfloat162> |
| wide T_B_KQ    | tile<16, 8, half2>            | tile<16, 8, nv_bfloat162> |
| wide T_C_KQ    | tile<16,16, float>            | tile<16,16, float> (same) |
| wide T_A_VKQ   | tile<16, 8, half2>            | tile<16, 8, nv_bfloat162> |
| wide T_B_VKQ   | tile<16, 8, half2>            | tile<16, 8, nv_bfloat162> |
| wide T_C_VKQ   | tile<16, 8, half2> (fp16 acc) | tile<16,16, float> (fp32 acc) |

Consequences:

- The bf16 narrow `T_C_VKQ = tile<16,8,float>` (4 f32/thread) maps EXACTLY onto the existing
  narrow bf16 mma overload (`mma.cuh:1181`). No new narrow VKQ mma needed.
- The bf16 wide `T_C_VKQ = tile<16,16,float>` (8 f32/thread) needs the new wide bf16 mma
  (section 5).
- [NEW] Register pressure: narrow is UNCHANGED (4 f32 vs 4 half2 per tile). Wide DOUBLES the
  VKQ_C footprint (8 f32 vs 4 half2 per tile). For DV=512 (DKQ=576) wide, VKQ_C alone is
  DV/16 = 32 tiles x 8 = 256 f32 registers/thread, exceeding the 255-register limit ->
  guaranteed spills. Narrow DV=256 is 16 x 4 = 64 (fine). This is a known risk; see
  section 9 step 7 for the benchmark-driven response. It affects performance, not
  correctness.

## 4. KQ_max_scale VKQ scaling (derived from tile get_j semantics)

Each element `l` of the FP32 VKQ tile covers a specific KQ column; the scale must match the
KQ softmax column indexing. Derived from `get_j` in `mma.cuh:256-271`:

- narrow (`T_C_VKQ = tile<16,8,float>`, `get_j = (tid%4)*2 + (l%2)`): column parity == `l%2`
  -> `VKQ_C[i].x[l] *= KQ_max_scale[l % 2]`
- wide (`T_C_VKQ = tile<16,16,float>`, `get_j = (l/4)*8 + (tid%4)*2 + (l%2)`): columns 0-7
  get scale[0], columns 8-15 get scale[1] -> `VKQ_C[i].x[l] *= KQ_max_scale[(l/2) % 2]`

Both are scalar f32 loops (replacing the half2 code) with bound `DV/T_C_VKQ::I` (DV/16 for
all four cases). Apply at BOTH sites: normal (`fattn-mma-f16.cuh:865-887`) and sinks
(`:1370-1421`). Consistent with the KQ softmax `KQ_idx` mapping (narrow `l%2` at
`:721,:747`; wide `(l/2)%2` at `:798,:840`).

No manual zeroing of KQ_C / VKQ_C is needed: `tile::x[ne] = {0}` (`mma.cuh:228`, NVIDIA
I_MAJOR branch) value-initializes the arrays; the mma overloads use `"+r"` accumulate.

## 5. mma overload mapping (verified against mma.cuh)

| call site | bf16 operands | mma overload |
|---|---|---|
| KQ narrow (:628)     | D=tile<16,8,float>, A=tile<16,8,nv_bfloat162>, B=tile<8,8,nv_bfloat162> | :1181 EXISTS |
| KQ wide CUDA (:636)  | D=tile<16,16,float>, A=tile<16,8,nv_bfloat162>, B=tile<16,8,nv_bfloat162> | NEW AMPERE branch |
| VKQ narrow (:983)    | D=tile<16,8,float>, A=tile<16,8,nv_bfloat162>, B=tile<8,8,nv_bfloat162> | :1181 EXISTS |
| VKQ wide CUDA (:991) | D=tile<16,16,float>, A=tile<16,8,nv_bfloat162>, B=tile<16,8,nv_bfloat162> | NEW AMPERE branch |

The wide bf16 mma overload exists at `mma.cuh:1257` but is AMD-only
(`#else NO_DEVICE_CODE`). Add a CUDA branch at the TOP, mirroring the wide F16 mma
(`mma.cuh:1196-1255`): `#if defined(AMPERE_MMA_AVAILABLE)`, two
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` asm calls with Bxi[0],Bxi[2] into
Dxi[0-3] then Bxi[1],Bxi[3] into Dxi[4-7] (same Axi[0-3]). Guard with `AMPERE_MMA_AVAILABLE`
(not TURING), matching the narrow bf16 mma at `mma.cuh:1183`. No Turing m8n8k8 fallback
exists for bf16 (bf16 mma requires sm_80+) and none is needed: the bf16 kernel only compiles
under `AMPERE_MMA_AVAILABLE`.

## 6. Changes by file

### 6.1 fattn-mma-f16.cuh

Add `#pragma once` as the first line. The bf16 header includes this file to reuse: config
getters (`:38-246`), `get_cols_per_warp`/`get_cols_per_thread` (`:329-344`),
`flash_attn_ext_f16_load_tile` (`:364`, byte copy, type-agnostic),
`flash_attn_ext_f16_load_mask` (`:450`, half-based, type-agnostic), the nbatch helpers and
the combine helpers. NO other change.

### 6.2 mma.cuh (all additive)

1. Wide bf16 AMPERE branch at the top of `:1257` (section 5).
2. `ggml_cuda_movmatrix(const nv_bfloat162 x)` wrapper, mirroring the half2 wrapper at
   `:63-67` (reinterpret through int, call the int overload).
3. `get_bf16`, mirroring `get_half2` at `:711-720`:
   `template <int I, int J> tile<I, J/2, nv_bfloat162> get_bf16(const tile<I, J, float> &)`,
   converting each float pair with `ggml_cuda_cast<nv_bfloat162>(make_float2(...))`.
4. `get_transposed(const tile<16, 4, nv_bfloat162> &) -> tile<8, 8, nv_bfloat162>`,
   mirroring `:722-728` (two `ggml_cuda_movmatrix` calls).

Casts confirmed in `convert.cuh`: `ggml_cuda_cast<float2>(nv_bfloat162)` =
`__bfloat1622float2` (`:44-53`); `ggml_cuda_cast<nv_bfloat162>(float2)` = aggregate init
`{x.x, x.y}` = RN conversion (`:54-60`).

### 6.3 new file: fattn-mma-bf16.cuh

`#pragma once`; include `mma.cuh`, `fattn-mma-f16.cuh`, `cp-async.cuh`, `fattn-common.cuh`.
CUDA-only.

Define `mma_tile_sizes_bf16<DV, ncols>` per section 3, guarded
`#if defined(AMPERE_MMA_AVAILABLE)`. Then clone the F16 kernel (iter, process_tile, kernel,
case function, DECL macros) applying the adaptation rules below.

Adaptation rules (BF16 vs F16) - ALL of:

1. mma calls resolve to the bf16 overloads per section 5. The CUDA wide KQ/VKQ mma keeps the
   swapped form (`mma(KQ_C[...], Q_B[...], K_A)` / `mma(VKQ_C[...], B[...], A)`), identical
   structure to `:636/:663/:991`.
2. B-array conversion (softmax P -> VKQ B operand, `:922-931`):
   - narrow: `B[k] = get_transposed(get_bf16(KQ_C[k]));`
   - wide:   `B[k] = get_bf16(KQ_C[k]);`
3. KQ_max_scale VKQ scaling: scalar f32 loops per section 4, at both sites.
4. `KQ_C` declaration (`:576-582`): reuse the F16 formula unchanged
   `T_C_KQ KQ_C[nbatch_fa/(np*(cols_per_warp == 8 ? T_C_KQ::I : T_C_KQ::J))]`
   (type-agnostic; narrow and wide both give /16).
5. `VKQ_C` declaration (process_tile, `:1182-1190`): `T_C_VKQ VKQ_C[DV/T_C_VKQ::I]` for BOTH
   narrow and wide. The F16 formula `DV/(2*T_C_VKQ::J)` works for F16 only by coincidence
   (f16 wide J=8 gives DV/16); for bf16 wide J=16 it would give DV/32, under-allocating by
   half. Do NOT copy the F16 formula.
6. Softmax (max/exp/rowsum, mask add, `:687-863`): unchanged - all scalar f32 ops on the FP32
   KQ_C tiles. The wide mask add reads half2 from `tile_mask` (`:778`) - unchanged (mask
   stays half-based).
7. Q fill (`:1202-1241`): write bf16 via `((nv_bfloat162 *) tile_Q)[jc*stride_tile_Q + k] =
   ggml_cuda_cast<nv_bfloat162>(scale_f2 * tmp)` with `scale_f2 = make_float2(scale, scale)`
   (float multiply, single RN rounding; there is no `__hmul2` for nv_bfloat162 in this
   codebase). OOB zero-fill: `ggml_cuda_cast<nv_bfloat162>(make_float2(0.0f, 0.0f))`.
8. Q/K/V loads: keep `tile_Q/tile_K/tile_V` as `extern __shared__ half2[]` (SRAM is
   type-agnostic) and pass `half2*` to the existing `load_tile`/`load_mask` unchanged.
   `load_ldmatrix` / `load_ldmatrix_trans` are T-templated and bit-agnostic
   (`mma.cuh:785-914`); cast the SRAM pointer to `nv_bfloat162*` at the call sites
   (section 7).
9. Combine-write (`:1566-1625`):
   - narrow: `const T_B_KQ B = get_transposed(get_bf16(VKQ_C[(k00 + k1)/T_B_KQ::J]));` then
     write `B.x[l]` (nv_bfloat162) into `tile_Q` reinterpreted as `nv_bfloat162*`. The
     `(k00+k1)/T_B_KQ::J` indexing tops out at DV/16, matching the bf16 VKQ_C size. The
     `static_assert` on T_C_VKQ::x becomes `float[T_C_VKQ::ne]`.
   - wide: use the EXISTING float branch (`:1611-1624`, currently AMD-only by usage) with
     `nv_bfloat16 * tile_Q_h = (nv_bfloat16 *) tile_Q;` and
     `tile_Q_h[j*(2*tile_stride) + k] = __float2bfloat16(VKQ_C[(k00 + k1)/(T_C_VKQ::J/2)].x[l]);`.
     The explicit cast is REQUIRED (`float->half` is implicit, `float->nv_bfloat16` is not).
     Indexing stays in bounds: max k = 2*nbatch_combine < 2*tile_stride.
10. Meta write / combine (`:1422-1560`): all float2/scalar ops, unchanged.
11. Read-back (`:1667-1693`): `ggml_cuda_cast<float2>(nv_bfloat162)` instead of
    `__half22float2(half2)`. Everything after is pure float arithmetic.

Kernel scaffolding:

- `flash_attn_ext_bf16` kernel: `static __global__`, identical char*/byte signature to the
  f16 kernel (`:1703-1726`), `__launch_bounds__(...)` with the same
  `ggml_cuda_fattn_mma_get_nthreads/occupancy` getters.
- `ggml_cuda_pdl_sync()` stays OUTSIDE the body guard (mirror `:1727`).
- Body guard: `#if defined(FLASH_ATTN_AVAILABLE) && defined(AMPERE_MMA_AVAILABLE)` with a
  `NO_DEVICE_CODE` fallback so the always-compiled `__global__` template links on all builds.
- [NEW] Skip checks inside the guard: KEEP the softcap-skip
  (`if (use_logit_softcap && !(DKQ == 128 || DKQ == 256 || DKQ == 512))`, `:1739-1742`) and
  the DKQ==192 check (`:1743-1746`). DROP the `__CUDA_ARCH__ == GGML_CUDA_CC_TURING`
  ncols>32 check (`:1754-1759`) and the AMD_WMMA / AMD_MFMA checks (`:1761-1773`).
- `stride_K = nb11 / sizeof(half2)` (`:1785`) and `stride_V` (`:1788`) are in 4-byte units,
  valid for nv_bfloat162. K_h2/V_h2 stay `const half2 *` (byte math; ldmatrix sites cast).
- `flash_attn_ext_bf16_process_tile` and `flash_attn_ext_bf16_iter` mirror the f16
  signatures (half2-typed SRAM pointers, float accumulators).

Case function (mirror `:1895-1964`):

- Same per-cc `fattn_mma_config` getters, `nbytes_shared_*` in `sizeof(half2)` units
  (byte-identical for bf16), `cols_per_warp = std::min(ncols, get_cols_per_warp(cc))`,
  `V_is_K_view = (DKQ == 576)`.
- [NEW] Mirror the `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`
  shared-memory-limit raise (`:1942-1960`) with the bf16 kernel pointer (guarded
  `#if !defined(GGML_USE_MUSA)`), for both `use_logit_softcap` values.
- `launch_fattn<DV, ncols1, ncols2>(ctx, dst, fattn_kernel, nwarps, nbytes_shared_total,
  nbatch_fa, false, false, true, warp_size_host)`.

DECL macros (mirror `:1967-2033`):

- `DECL_FATTN_MMA_BF16_CASE(DKQ, DV, ncols1, ncols2)` = explicit instantiation of the bf16
  case function; `DECL_FATTN_MMA_BF16_CASE_ALL_NCOLS2(DKQ, DV, ncols)` = extern decls for
  ncols2 in {1,2,4,8,16}.
- [NEW] Instance list matches the f16 externs exactly:
  `ALL_NCOLS2( 64,64, 8/16/32/64)`, `ALL_NCOLS2( 80,80, ...)`, `( 96,96, ...)`,
  `(112,112, ...)`, `(128,128, ...)`, `(256,256, ...)`;
  512->512: the 12 (ncols1,ncols2) externs (4/2,8/2,16/2,32/2, 2/4,4/4,8/4,16/4, 1/8,2/8,4/8,8/8);
  576->512: (1,16),(2,16),(4,16),(4,4),(8,4),(16,4),(1,32),(2,32);
  320->256: (1,32),(2,32).
- [NEW] NO 192 externs: the case function is fully defined in the header, so 192 is
  implicitly instantiated in fattn.cu's TU (same as f16). The generated instance files do
  NOT define 192 either; the DECL_ALL_NCOLS2 macro for ncols=8 emits "dangling" externs with
  ncols1=0 (8/16), which are never referenced and harmless.

### 6.4 fattn.cu

1. enum (`:331-336`): add `BEST_FATTN_KERNEL_MMA_BF16 = 500`.
2. Routing (`:461-483`): inside the `turing_mma_available(cc) && Q->ne[0] != 40 &&
   Q->ne[0] != 72` branch, AFTER the vec block, immediately before
   `return BEST_FATTN_KERNEL_MMA_F16;`:
   ```cpp
   if (K->type == GGML_TYPE_BF16 && V->type == GGML_TYPE_BF16 && ampere_mma_available(cc)) {
       return BEST_FATTN_KERNEL_MMA_BF16;
   }
   ```
   [NEW] `ampere_mma_available(cc)` (`common.cuh:352-354`) = NVIDIA && highest compiled arch
   >= AMPERE, matching the kernel's compile-time `AMPERE_MMA_AVAILABLE` guard exactly. This
   is the SAME pattern as the f16 routing's `turing_mma_available(cc)` (also compiled-arch
   based). Do NOT use `bf16_mma_hardware_available(cc)` here: it is hardware-based (cc >=
   800) and would route to MMA_BF16 on an sm_75-only build running Ampere hardware, where
   the kernel body compiles to NO_DEVICE_CODE (empty) -> silent wrong output. Placing the
   gate inside the turing branch keeps it NVIDIA-only and preserves VEC priority (a BF16
   cache with a single-prompt small batch still goes VEC on Ada+).
3. `switch_ncols1` (bf16 mirror of `:8-34`): [NEW] the ncols2<=8 guard uses
   `ampere_mma_available(cc)` (mirror of f16's `turing_mma_available` at `:14`); the f16
   line-27 Turing/AMD conditions are DROPPED (bf16 has no Turing or AMD instances):
   ```cpp
   if constexpr (ncols2 <= 8) {
       if (ampere_mma_available(cc) && Q->ne[1] <= 8/ncols2) { case<8/ncols2, ncols2>; return; }
   }
   if constexpr (ncols2 <= 16) {
       if (Q->ne[1] <= 16/ncols2) { case<16/ncols2, ncols2>; return; }
   }
   if (Q->ne[1] <= 32/ncols2) { case<32/ncols2, ncols2>; return; }
   case<64/ncols2, ncols2>;
   ```
4. `switch_ncols2` (bf16 mirror of `:36-111`): drop the Volta block (`:67-89`); keep the
   gqa_ratio-based selection (`:91-110`) unchanged.
5. Per-DKQ dispatch `ggml_cuda_flash_attn_ext_mma_bf16(ctx, dst)` (mirror of `:113-242`):
   same DKQ switch and ncols2 selection (192 -> {8,16}, 320 -> 32, 512 -> {2,4,8}, 576 ->
   {4,16,32}; 40/72 excluded; 576 with gqa_ratio==20 uses the ncols2==32 sub-branch),
   calling the bf16 case functions.
6. `get_alloc_size` (`:550-562`): `case BEST_FATTN_KERNEL_MMA_BF16: break;` (native BF16
   K/V: no conversion buffers).
7. Main entry (`:570-585`): `case BEST_FATTN_KERNEL_MMA_BF16:
   ggml_cuda_flash_attn_ext_mma_bf16(ctx, dst); break;`
8. Add `#include "fattn-mma-bf16.cuh"` next to the F16 include.

### 6.5 template-instances/generate_cu_files.py

Mirror the mma-f16 emission block (`:78-103`) with new `SOURCE_FATTN_MMA_BF16_START` and
`SOURCE_FATTN_MMA_BF16_CASE` templates and the SAME rules: ncols in {8,16,32,64}, ncols2 in
{1,2,4,8,16,32} with ncols2 <= ncols; skip 40 and 72; 192 -> {8,16}; 320 -> 32; 512 ->
{2,4,8}; 576 -> {4,16,32}; ncols2 in {16,32} only for those niche sizes. Then run the
script. No CMake change needed: `fattn-mma*.cu` is GLOBbed (`CMakeLists.txt:108`).

## 7. SRAM touch-point checklist (the ONLY nv_bfloat162 conversions)

Keep SRAM as `half2`; reinterpret as nv_bfloat162 exactly here:

1. Q fill `:1202-1241` (rule 7).
2. KQ ldmatrix `:626/:652`: `load_ldmatrix(K_A, (const nv_bfloat162 *)(tile_K + ...), stride)`.
3. Q_B ldmatrix `:645/:1250`: same cast pattern on tile_Q.
4. VKQ ldmatrix_trans `:981`: `load_ldmatrix_trans(A, (const nv_bfloat162 *)(tile_V_i + ...), stride)`.
5. Combine-write narrow `:1570-1581` (rule 9).
6. Combine-write wide `:1611-1624` (rule 9).
7. Read-back `:1670` (rule 11).

Everything else (mask tile, softmax, meta, fixup, sinks arithmetic) stays half/float.
`GGML_ASSERT(K->nb[0] == ggml_element_size(K))` (`fattn-common.cuh:994`) holds for
contiguous BF16 K/V (nb[0] = 2).

## 8. Reference map (verified line numbers)

- `fattn-mma-f16.cuh`: pragma once (1), config getters (38-246), load_tile (364-448),
  load_mask (450-528), iter (533-1022), mma_tile_sizes (1027-1113), process_tile (1116-1701),
  kernel (1703-1893), case fn (1895-1964), DECL/externs (1967-2033).
- KQ_max_scale VKQ scaling sites: normal (865-887), sinks (1370-1421).
- Softmax KQ_idx: narrow `l%2` (721,747), wide `(l/2)%2` (798,840).
- B-array conversion (922-931); VKQ mma (953-1016); Q fill (1202-1241); Q_B load (1245-1252);
  combine-write (1566-1625); meta (1422-1560); read-back (1667-1693).
- `mma.cuh`: movmatrix (28-67), get_half2/get_transposed (711-728), load_ldmatrix (785-914),
  narrow bf16 mma (1181-1194), wide f16 mma (1196-1255), wide bf16 mma AMD-only (1257-1303),
  tile `T x[ne] = {0}` (228).
- `fattn.cu`: enum (331-336), routing (461-483), switch_ncols1 (8-34), switch_ncols2 (36-111),
  dispatch (113-242), get_alloc_size (536-568), ext (570-585).
- `fattn-common.cuh`: launch_fattn (972-1189), f16_extra_data (47-85).
- `common.cuh`: AMPERE_MMA_AVAILABLE (282-284), ampere_mma_available (352-354),
  bf16_mma_hardware_available (322-326), turing_mma_available (348-350).
- `convert.cuh`: bf16 casts (38-66).
- `generate_cu_files.py`: mma-f16 emission (78-103).

## 9. Implementation checklist

1. `fattn-mma-f16.cuh:1`: add `#pragma once`.
2. `mma.cuh`: add the 4 items in section 6.2.
3. Create `fattn-mma-bf16.cuh` (sections 6.3 + 7).
4. `fattn.cu`: enum, routing gate, switch/dispatch, get_alloc_size, main entry, include.
5. `generate_cu_files.py`: add bf16 templates and emit; run the script.
6. Build (NVIDIA, incl. sm_86 + sm_120a) and iterate on compile errors.
7. Register pressure: narrow is unchanged. Wide VKQ_C doubles to 8 f32/thread; DV=512 wide
   exceeds the 255-register limit and will spill. If benchmarks show an unacceptable
   slowdown on those paths, retune `mma_tile_sizes_bf16` / `get_nstages` / `get_nbatch_*`
   based on measurement (report numbers, do not guess).

## 10. Validation plan (NOT actioned until explicitly requested)

- Correctness: `test-backend-ops` filtered to FLASH_ATTN_EXT with F32/F16/BF16 Q and BF16
  K/V; compare against the F16 MMA and vec kernels.
- Precision: `llama-perplexity` with `--cache-type-k bf16 --cache-type-v bf16`; the bf16 MMA
  kernel must match the bf16 vec kernel's precision (FP32 VKQ).
- Performance: `llama-bench -b 2048 -n 64 --cache-type-k bf16 --cache-type-v bf16` vs the
  F16-cache baseline, covering narrow (ncols<=8) and wide (ncols 16/32/64) paths, head dims
  192/256/320/512/576, GQA, sinks, multi-stage (nstages>1), MLA (V_is_K_view).
- Hardware: RTX 3090 (sm_86) and RTX PRO 5000 (sm_120a, Blackwell).