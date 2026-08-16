# PLAN v6: Native BF16 Flash Attention MMA Kernel for NVIDIA (sm_80+)

Delta over `PLAN-bf16-fattn-v3.md` + `v4` + `v5`. This file records the round-6 deep-dive
results: full verification of the `iter` body, the mma overload mapping, the tile layouts,
and the exact SRAM typing strategy. v3+v4+v5+v6 supersede all earlier versions.

## 1. Tile types for the BF16 kernel (verified against `mma_tile_sizes`, `fattn-mma-f16.cuh:1027-1043`)

Turing/Ampere CUDA sets. Only the `half2` types change to `nv_bfloat162`; the fp32
accumulators and all data_layouts are unchanged:

| set | f16 (reference) | bf16 (new) |
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

KEY FINDINGS:
- The f16 NARROW T_C_VKQ is `tile<16,4,half2>` (m16n8k16.f16.f16.f16.f16, fp16 accumulate)
  and the f16 WIDE T_C_VKQ is `tile<16,8,half2>` (also fp16 accumulate) - both are the
  PPL-drift source. The bf16 kernel replaces them with fp32 tiles.
- The bf16 narrow T_C_VKQ = tile<16,8,float> is bit-identical in register count to the
  narrow KQ_C (4 f32/thread) and maps EXACTLY onto the existing narrow bf16 mma
  (`mma.cuh:1181`). NO new narrow VKQ mma overload needed.
- The bf16 wide T_C_VKQ = tile<16,16,float> (8 f32/thread, +4 regs vs f16 wide) maps onto
  the NEW wide bf16 mma (v3 2.3). VKQ_C register pressure: narrow 4 regs (same), wide 8
  regs (vs f16 4 half2 = 4 regs).

## 2. mma overload mapping (all confirmed against `mma.cuh`)

| call site | bf16 operands | mma overload |
|---|---|---|
| KQ narrow (:628)     | D=tile<16,8,float>, A=tile<16,8,nv_bfloat162>, B=tile<8,8,nv_bfloat162> | :1181 EXISTS |
| KQ wide CUDA (:636)  | D=tile<16,16,float>, A=tile<16,8,nv_bfloat162>, B=tile<16,8,nv_bfloat162> | NEW AMPERE branch |
| VKQ narrow (:983)    | D=tile<16,8,float>, A=tile<16,8,nv_bfloat162>, B=tile<8,8,nv_bfloat162> | :1181 EXISTS |
| VKQ wide CUDA (:991) | D=tile<16,16,float>, A=tile<16,8,nv_bfloat162>, B=tile<16,8,nv_bfloat162> | NEW AMPERE branch |

The NEW wide bf16 mma goes at the TOP of the existing AMD wide bf16 mma (`mma.cuh:1257`),
mirroring the wide f16 mma (`:1196-1255`): `#ifdef AMPERE_MMA_AVAILABLE`, 2x
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` with Bxi[0],Bxi[2] then Bxi[1],Bxi[3],
D in Dxi[0-3] then Dxi[4-7]. AMD branches (`#elif`) stay below.

## 3. KQ_max_scale mappings - now DERIVED from tile get_j semantics (not just asserted)

For T_C_VKQ = tile<16,8,float> (narrow, `mma.cuh:261-262`): element l covers column
`(threadIdx.x%4)*2 + (l%2)`, so column parity = l%2. Scale = `KQ_max_scale[l % 2]`.
For T_C_VKQ = tile<16,16,float> (wide, `mma.cuh:263-264`): element l covers column
`(l/4)*8 + (threadIdx.x%4)*2 + (l%2)`; columns 0-7 get scale[0], 8-15 get scale[1]
(consistent with KQ softmax KQ_idx `(l/2)%2` at :798/:840). Scale = `KQ_max_scale[(l/2) % 2]`.

Both scalar f32 loops, bound `DV/T_C_VKQ::I` (= DV/16 for ALL four cases), at BOTH sites:
normal (`:865-887`) and sinks (`:1370-1421`). Confirmed consistent with the KQ_C softmax
KQ_idx mapping (narrow `l%2` :721, wide `(l/2)%2` :798).

## 4. VKQ_C sizing - the f16 formula is a COINCIDENCE, do not copy it

f16 declaration (`:1183`): `VKQ_C[cols_per_warp == 8 ? DV/T_C_VKQ::I : DV/(2*T_C_VKQ::J)]`.
For f16 wide T_C_VKQ::J = 8, so `DV/(2*8) = DV/16` - correct by coincidence.
For bf16 wide T_C_VKQ::J = 16, the same formula gives DV/32 - WRONG (half the tiles).
bf16 declaration (both narrow and wide): `T_C_VKQ VKQ_C[DV/T_C_VKQ::I];` (= DV/16).
KQ_C (`:577`) and B (`:922`) formulas are type-agnostic (narrow/wide both give /16).

## 5. SRAM typing strategy - keep load_tile/load_mask 100% untouched

Declare `extern __shared__ half2 tile_Q[]` in the bf16 kernel exactly like f16, and pass
`half2 *` tile_K/tile_V to the existing load_tile/load_mask unchanged. Reinterpret-cast to
nv_bfloat162 ONLY at the 6 nv_bfloat162 touch sites:

1. Q fill (:1202-1241): `((nv_bfloat162 *) tile_Q)[jc*stride_tile_Q + k] = ggml_cuda_cast<nv_bfloat162>(scale_f2 * tmp);` (v4 1.1); OOB zero-fill `ggml_cuda_cast<nv_bfloat162>(make_float2(0.0f, 0.0f))`.
2. KQ ldmatrix (:626/:652): `load_ldmatrix(K_A, (const nv_bfloat162 *)(tile_K + i_KQ_0*stride_tile_K + (k_KQ_0 - k0_start)), stride_tile_K);` (K_A = tile<16,8,nv_bfloat162>).
3. Q_B ldmatrix (:645/:1250): same cast pattern on tile_Q.
4. VKQ ldmatrix_trans (:981): `load_ldmatrix_trans(A, (const nv_bfloat162 *)(tile_V_i + 2*k0*stride_tile_V + (i_VKQ_0 - i0_start)/2), stride_tile_V);` (A = tile<16,8,nv_bfloat162>).
5. Combine-write narrow (:1570-1581): `const T_B_KQ B = get_transposed(get_bf16(VKQ_C[(k00 + k1)/T_B_KQ::J]));` then `((nv_bfloat162 *) tile_Q)[jc_cwd*tile_stride + k] = B.x[l];`; static_assert becomes `float[T_C_VKQ::ne]`.
6. Combine-write wide (:1611-1624): the EXISTING float branch (currently AMD-only by usage) with `nv_bfloat16 * tile_Q_h = (nv_bfloat16 *) tile_Q;` and `tile_Q_h[j*(2*tile_stride) + k] = __float2bfloat16(VKQ_C[(k00 + k1)/(T_C_VKQ::J/2)].x[l]);`. Indexing (k = 2*k1 + get_j(l)) stays in-bounds: max k = 2*nbatch_combine < 2*tile_stride.
7. Read-back (:1670): `ggml_cuda_cast<float2>(((const nv_bfloat162 *) tile_Q)[(jc_tile_K + ip*cols_per_warp) * tile_stride + k]);` (v4 1.2).

B-array conversion (:922-931): narrow `B[k] = get_transposed(get_bf16(KQ_C[k]));`, wide
`B[k] = get_bf16(KQ_C[k]);` where get_bf16 = `template<int I,int J> tile<I,J/2,nv_bfloat162> get_bf16(const tile<I,J,float>&)` (mirror get_half2 :713-720, guards the whole loop with `if constexpr (cols_per_warp == 8)`), and get_transposed(tile<16,4,nv_bfloat162>) -> tile<8,8,nv_bfloat162> (mirror :722-728, uses ggml_cuda_movmatrix(nv_bfloat162) new wrapper at :63-67 mirror).

## 6. Nothing else changes

Mask/softmax/meta/fixup/sinks arithmetic is float or half-based (mask tile stays half).
`tile_V_i` pointer math (:970), stride math, occupancy, launch, case fn, host wiring,
generate_cu_files.py: all as in v5. Wide Q-fill and read-back use the same float branch and
read-back path as narrow (the 6 sites above are the ONLY nv_bfloat162 conversions).

## 7. Updated implementation checklist (v3 step 3, expanded)

1. `fattn-mma-f16.cuh:1`: `#pragma once`.
2. `mma.cuh`: `ggml_cuda_movmatrix(nv_bfloat162)` (:63-67 mirror); `get_bf16` (:713-720 mirror); `get_transposed(tile<16,4,nv_bfloat162>)` (:722-728 mirror); wide bf16 AMPERE branch at top of :1257.
3. `fattn-mma-bf16.cuh`: full clone with: tile types per section 1; mma mapping per section 2; scaling per section 3; VKQ_C sizing per section 4; SRAM casts per section 5; AMPERE kernel guard; bf16 DECL macros.
4. `fattn.cu`: enum + routing gate (v5 2.1/2.2) + get_alloc_size (v5 2.3) + main entry (v5 2.4) + include.
5. `generate_cu_files.py`: add SOURCE_FATTN_MMA_BF16_START/CASE, same skip rules; run.
6. Build (sm_86, sm_120a) and iterate.

Validation (NOT actioned until requested): test-backend-ops / llama-bench / llama-perplexity
with `--cache-type-k bf16 --cache-type-v bf16` vs F16 baseline on RTX 3090 + RTX PRO 5000.