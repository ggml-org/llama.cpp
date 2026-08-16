# PLAN v4: Native BF16 Flash Attention MMA Kernel for NVIDIA (sm_80+)

Delta over `PLAN-bf16-fattn-v3.md` (v3 remains the base plan). This file records the
round-4 verification results and the two conversion-point refinements it uncovered.
v3+v4 together supersede v2/v1.

## 1. Round-4 refinements (NEW specification - not in v3)

### 1.1 Q fill and scaling (process_tile, `fattn-mma-f16.cuh:1202-1241`)

The F16 kernel stores Q to `tile_Q` as `scale_h2 * make_half2(tmp.x, tmp.y)` (half-precision
multiply). There is no `__hmul2`-style intrinsic and no arithmetic operator for
`nv_bfloat162` in this codebase (grep confirmed: only `half2` usages exist), so the BF16
kernel must scale in float and round once:

```cpp
const float2 scale_f2 = make_float2(scale, scale);
...
((nv_bfloat162 *) tile_Q)[jc*stride_tile_Q + k] = ggml_cuda_cast<nv_bfloat162>(scale_f2 * tmp);
```

- `scale_f2 * tmp` is a float2 element-wise multiply.
- `ggml_cuda_cast<nv_bfloat162>(float2)` = `nv_bfloat162{__float2bfloat16(x.x), __float2bfloat16(x.y)}`
  (RN), `convert.cuh:54-60`.
- OOB zero-fill (`:1237`): `ggml_cuda_cast<nv_bfloat162>(make_float2(0.0f, 0.0f))`.
- This is a single rounding of a float multiply (more accurate than the F16 double-rounding);
  matches the BF16 vec kernel's precision behavior.

### 1.2 Final read-back (process_tile, `fattn-mma-f16.cuh:1670`)

```cpp
// F16:
const float2 dstk_val_add = __half22float2(tile_Q[(jc_tile_K + ip*cols_per_warp) * tile_stride + k]);
// BF16:
const float2 dstk_val_add = ggml_cuda_cast<float2>(((const nv_bfloat162 *) tile_Q)[(jc_tile_K + ip*cols_per_warp) * tile_stride + k]);
```

`ggml_cuda_cast<float2>(nv_bfloat162)` = `__bfloat1622float2` (`convert.cuh:44-45`).
Everything after (KQ_crs scaling, rowsum division, dstk/dstk_fixup writes) is pure float
arithmetic, unchanged.

## 2. Round-4 confirmations (things v3 asserted, now line-verified)

- **Config getters are type-agnostic.** `fattn_mma_config` values (nthreads, occupancy,
  nbatch_fa, nbatch_K2, nbatch_V2, nbatch_combine, nstages_target, Q_in_reg) are plain
  int/bool constants (`fattn-mma-f16.cuh:38-263`) with no dependency on half2/bf16 types.
  The BF16 kernel reuses them unchanged, including the AMPERE table (`:38-88`).
- **`get_cols_per_thread() == 2`** (non-AMD, `:329-335`) and **`get_cols_per_warp(cc) == 16`**
  for Turing+ (`:337-344`). These match the BF16 tile geometries
  (`T_B_KQ::I == 8` narrow / `16` wide). Reuse as-is.
- **`load_tile` is a 16-byte copy** (`flash_attn_ext_f16_load_tile`, `:364-448`) via
  `cp.async` or `ggml_cuda_memcpy_1<16>`; the `half2` typing is only for alignment/stride
  math. Reusable for BF16 K/V unchanged.
- **`load_ldmatrix` / `load_ldmatrix_trans` are T-templated and bit-agnostic** over 32-bit
  registers. The exact overloads needed exist: `tile<8,8,T>` (`mma.cuh:786`),
  `tile<16,4,T>` (`:801`), `tile<16,8,T>` (`:830`), `tile<16,8,T,dl>` trans (`:885`).
- **VKQ mma region** (`fattn-mma-f16.cuh:953-1016`): narrow `mma(VKQ_C, A, B)` with
  `A=tile<16,8,nv_bfloat162>`, `B=tile<8,8,nv_bfloat162>`, `D=tile<16,8,float>` ->
  `mma.cuh:1181` (exists); wide CUDA `mma(VKQ_C, B[k..], A)` with `A=B=tile<16,8,nv_bfloat162>`,
  `D=tile<16,16,float>` -> `mma.cuh:1257` (AMD branch to be extended, see v3 2.3). Both use
  `load_ldmatrix_trans` for the V operand; swap for CUDA wide only.
- **Mask handling**: narrow add `:704` (`slope * __half2float(tile_mask[...])`) and wide add
  `:778-780` (`__half22float2` from `tile_mask`) read the half-based mask - NO change for BF16.
- **Softmax, meta write, combine** (`:687-863`, `:1422-1560`): all scalar float ops - NO change.

## 3. Nothing else changed vs v3

All of v3's corrections/constraints remain authoritative:
- VKQ scaling mapping: narrow `l % 2`, wide `(l/2) % 2` (v3 2.1).
- No manual accumulator zeroing (`tile::x[ne] = {0}`, `mma.cuh:228`).
- `VKQ_C[DV/T_C_VKQ::I]` for both narrow and wide.
- mma.cuh additions: wide AMPERE branch (2.3), `ggml_cuda_movmatrix(nv_bfloat162)`,
  `get_bf16`, `get_transposed(tile<16,4,nv_bfloat162>)`.
- Steps 1-5, scope/constraints, reference map, test matrix as in v3.

## 4. Complete conversion-point checklist for the BF16 kernel (all of them)

In `flash_attn_ext_bf16_process_tile` / `iter`, the ONLY half2<->bf16 touch points are:

1. Q fill `:1202-1241` (v4 1.1).
2. B-array conversion `:922-931`: `get_transposed(get_bf16(KQ_C[k]))` (narrow),
   `get_bf16(KQ_C[k])` (wide).
3. KQ_max_scale VKQ scaling `:865-887` and `:1370-1421`: scalar f32 loops, `l%2` /
   `(l/2)%2`, bound `DV/T_C_VKQ::I`.
4. Combine-write `:1566-1625`: narrow `get_transposed(get_bf16(...))` write nv_bfloat162;
   wide float-branch write `__float2bfloat16(...)` (no implicit float->bf16 conversion).
5. Read-back `:1670` (v4 1.2).
6. mma calls resolve to bf16 overloads; all else (loads, ldmatrix, softmax, mask, meta,
   fixup, sinks arithmetic) is shared verbatim.