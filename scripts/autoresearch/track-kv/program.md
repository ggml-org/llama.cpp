# TurboQuant KV Cache Kernel Optimization

## Goal
Maximize decode t/s for TurboQuant KV cache types (turbo2, turbo3, turbo4) on CUDA
(RTX 5090, Blackwell sm_120). Focus on the VEC flash attention decode kernel
(`fattn-vec.cuh`) which dominates decode-time compute.

Current baseline: ~187 t/s with turbo3 KV on Qwen3.5-35B-A3B (Q4_K_M weights).
Target: close the gap to q8_0 KV (~200+ t/s).

## Target File
`ggml/src/ggml-cuda/fattn-vec.cuh` — ONLY modify this file.

## Model & Benchmark
- Model: `/mnt/ai/models/huggingface/qwen3.5-35b-a3b-GGUF/Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf`
- Benchmark: `llama-bench -ngl 99 -p 512 -n 128 -r 3 --cache-type-k turbo3 --cache-type-v turbo3`
- Correctness: PPL must stay within 0.1 of baseline
- Also test: `--cache-type-k turbo4 --cache-type-v turbo4` and `--cache-type-k turbo2 --cache-type-v turbo2`

## Architecture Overview
TurboQuant KV cache compresses K and V tensors using PolarQuant (WHT rotation +
Lloyd-Max quantization). Block size = 128, with norm + 2/3/4-bit quantized values.

### VEC Flash Attention Decode Kernel
The VEC kernel handles single-token decode (n_tokens ≤ 2). Each warp computes
attention for one head. The kernel has two main phases:

**KQ scoring (Q × K^T):**
- Q is pre-rotated and quantized to q8_1 format
- K is stored in turbo format (128-element blocks with norms + quantized values)
- Uses shared-memory LUT: precompute Q×centroid products, then score via LUT lookup
- turbo3: 8-entry LUT per Q block; turbo2: 4-entry LUT

**V aggregation (softmax(KQ) × V):**
- V is stored in turbo format
- Dequant V values, multiply by attention weight, accumulate
- Sparse V optimization: skip dequant for negligible attention weights

### Key Performance Features Already Implemented
- Shared-memory Q×centroid LUT (eliminates multiply in KQ inner loop)
- q8_1 Q quantization path (int8 Q values for turbo KQ scoring)
- __expf fast-math softmax
- L2 prefetch for K+V blocks
- Sparse V thresholds (skip V dequant for low attention weights)
- __launch_bounds__ occupancy 3
- nthreads_KQ=8 for turbo types

## Already Tried — Do NOT Re-explore
| Approach | Result | Why it failed |
|----------|--------|--------------|
| Larger LUT (16-entry for turbo3) | No improvement | 8-entry already covers 3-bit |
| Different occupancy (1, 2, 4) | 3 is optimal | Lower occupancy = less latency hiding |
| V dequant loop unroll | No improvement | Compiler already unrolling |
| `expf` → `__expf` fast-math | Already applied | +0.1%, already in current code |
| Sparse V threshold tuning | Already at 1e-3 | Hill-climbed 1e-6→1e-4→5e-4→1e-3→2e-3, diminishing returns. Do NOT keep bumping this — higher thresholds risk PPL regression at long context. The current value is already aggressive. |
| L2 prefetch for next K/V blocks | +0.1% | Already tried, marginal gain |
| L1 vs L2 prefetch | No difference | Tried both, within noise |
| `__launch_bounds__` occupancy 1→2→3 | Occupancy 2 marginally best | Already applied |

## Promising Directions to Explore
Focus on STRUCTURAL changes to the kernel, not parameter tuning.

### From community discussion (ggml-org/llama.cpp#20969)

- **Fused K tile loader (dusterbloom/Madreag approach)**: Keep K in compressed TBQ3
  format in the MMA kernel, fuse dequant into the tile loader. Zero temp buffer for K.
  This is how Madreag's optimized fork achieves near-parity with q8_0 on prefill.
- **cp.async pipeline for V tiles**: Bulk dequant V → fp16, then use cp.async.cg
  for V tile loads into shared memory. Overlaps V dequant with K scoring compute.
- **Hybrid prefill architecture**: Different code paths for prefill (MMA with fused
  tile loaders) vs decode (VEC with current approach). Prefill benefits most from
  tile-level fusion.
- **Precomputed scaled centroids per V block**: Instead of `centroid[idx] * norm`
  per element, precompute `scaled_centroid[idx] = centroid[idx] * norm` once per
  block (4 or 8 entries × 1 float each). Eliminates one multiply per V element.
- **Cross-head WHT (AmesianX)**: For models with head_dim=64, apply WHT across
  multiple KV heads via Kronecker decomposition (H_512 = H_8 ⊗ H_64). Claims
  better decorrelation for small head dims.

### Kernel-level ideas

- **KQ scoring with dp4a**: Q is already q8_1. If K centroids can be mapped to
  int8 per-block (like we proved with TQ4_0), dp4a for KQ dot product.
- **Warp specialization**: Dedicate some warps to K prefetch, others to V prefetch.
- **Double buffering**: Prefetch next KV block while processing current one
  using cp.async or separate warp.
- **Register pressure reduction**: Profile register usage, reduce if spilling.
- **Shared memory V cache**: Cache frequently-accessed V blocks in shmem.
- **Half2 accumulation**: Use fp16 for intermediate attention weight accumulation.
- **Fused softmax + V aggregation**: Combine the two passes into one.
- **Vectorized memory loads**: Use `float4` or `uint4` loads for K/V data.
- **Loop interchange**: Change iteration order (heads vs KV positions) for
  better cache locality.
- **Reduce warp reduction overhead**: The `__shfl_xor_sync` reduction at end
  of KQ scoring runs 5 stages — can we accumulate differently?

## Constraints
- Must not change the turbo block format ABI (shared with Metal/CPU)
- Must not modify any file other than fattn-vec.cuh
- Must maintain correct attention output (PPL gate catches corruption)
- Must work on Blackwell (sm_120) and Ampere (sm_86)
- The kernel is templated — changes affect all turbo type instantiations
