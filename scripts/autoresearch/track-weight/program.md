# TQ4_1S Weight Kernel Optimization

## Goal
Maximize decode t/s for TQ4_1S `mul_mat_vec` on CUDA (RTX 5090, Blackwell sm_120).
Current baseline: ~69 t/s. Target: close the gap to q4_0 (267 t/s).

## Target File
`ggml/src/ggml-cuda/mmvq-tq.cu` — ONLY modify this file.

## Model & Benchmark
- Model: `/tmp/qwen2.5-7b-instruct-tq4_1s.gguf`
- Benchmark: `llama-bench -ngl 99 -p 0 -n 128 -r 3` (decode only)
- Correctness: PPL on wikitext-2 must stay within 0.1 of baseline (7.599)

## Architecture Overview
The TQ4_1S format stores WHT-rotated 4-bit weights with non-linear Lloyd-Max centroids.
Block size = 32 elements, dual half-block scales (d0 for [0..15], d1 for [16..31]).
20 bytes per block = 5.0 bits/value.

Dequant per element: `centroid_table[4bit_index] * half_block_scale`
Then inverse WHT (Walsh-Hadamard Transform) to recover original weight space.

The fused mmvq kernel avoids per-block inverse WHT by pre-rotating the activation
vector (WHT forward) once, then the inner loop is just:
```
sum += rotated_activation[lane] * centroid[idx] * d
```

### Current Kernel (V8)
- 8 warps per CUDA block (MMVQ_TQ_NWARPS = 8), each warp handles one output row
- 32 lanes per warp, each lane handles element `lane` within every block
- Activation pre-rotated to float scratch buffer via warp shuffle WHT
- Inner loop: 1 float FMA per element per lane
- Warp reduction via `__shfl_xor_sync`

### Block Layout (block_tq4_1s)
```c
struct block_tq4_1s {
    half d0;           // 2 bytes: scale for elements [0..15]
    half d1;           // 2 bytes: scale for elements [16..31]
    uint8_t qs[16];    // 16 bytes: 4-bit indices, consecutive pair packing
                       // qs[j/2] >> ((j&1)*4) & 0xF = centroid index for element j
};
```

### Centroid Table (constant memory)
16 Lloyd-Max optimal values for N(0,1):
```
[-2.733, -2.069, -1.618, -1.256, -0.942, -0.657, -0.388, -0.128,
  0.128,  0.388,  0.657,  0.942,  1.256,  1.618,  2.069,  2.733]
```

## Confirmed Bottleneck
The centroid lookup itself is NOT the bottleneck — confirmed via ablation
(replacing `centroid[idx]` with `(idx-8)` gives identical 69 t/s).

The real bottleneck is:
1. **Float32 activation bandwidth**: 4 bytes/element vs q8_1's 1 byte. Each warp
   reads the full activation vector from global memory per row.
2. **Float FMA arithmetic density**: 1 MAC per instruction vs dp4a's 4 MACs.
   q4_0 processes 8 elements per dp4a pair; V8 processes 1 element per FMA.

## Already Tried — Do NOT Re-explore
| Version | Approach | Result | Why it failed |
|---------|----------|--------|--------------|
| V9-V11 | Multi-row NR0=2,4 (shmem or registers) | Regressed | __syncthreads overhead or register spill |
| V12 | Shmem activation broadcast | ~67 t/s | No help on 5090 (128MB L2 already caching) |
| V13 | Loop unroll ×4 | ~69 t/s | Compiler already optimal |
| V14 | WMMA tensor cores | 6 t/s | Setup overhead >> throughput for matvec |
| V15 | L2 prefetch hints | ~69 t/s | No measurable effect |
| V16 | __launch_bounds__ tuning | ~69 t/s | Occupancy changes no effect |
| V18 | Per-block int8 LUT + dp4a | 46 t/s | LUT build + pack overhead > dp4a gain |
| V19 | ILP 4× unroll | 70 t/s | Negligible improvement |

## Promising Directions to Explore

### From community discussion (ggml-org/llama.cpp#20969)

- **Entropy-coded weight compression (karambaso idea)**: With only 16 centroid
  values, 4-bit indices have low entropy. Runtime Huffman/ANS decompression in
  shmem could reduce effective bandwidth 30-50%. Decode is memory-bound, so extra
  compute for decompression may be free. This reframes the problem: instead of
  faster dequant, read less data.
- **Fused tile loader pattern (from Madreag's KV work)**: Load multiple weight
  blocks into shmem, dequant in-register from shmem. Amortizes global memory
  latency across a tile of blocks.
- **F32 vs fp16 activation precision**: AmesianX notes WHT amplifies q8_1
  quantization error ~16x. Our V8 uses f32 activation which avoids this.
  But fp16 activation would halve bandwidth. Worth testing if the quality
  tradeoff is acceptable for weights (less sensitive than KV cache).

### Kernel-level ideas

- **half2 packed FMA**: Process 2 elements per `__hmul2`/`__hfma2` instruction.
  Centroids in fp16 constant memory, activation in fp16. 2x arithmetic density.
- **Warp-cooperative coalesced loading**: Reorganize memory access so weight loads
  are fully coalesced (currently scattered due to per-lane block access).
- **Register blocking across blocks**: Each lane accumulates across multiple
  blocks before reducing, keeping partial sums in registers.
- **Async memory copy (cp.async)**: Prefetch next block's weight data while
  computing current block.
- **Different warp configurations**: Try 4 or 16 warps instead of 8.
- **Two-level tiling**: Load a tile of blocks to shmem, process tile, repeat.
  This is the shmem activation variant (V12) but for weights instead.
- **Vectorized weight loads**: Load 4 bytes (8 nibbles) per lane per iteration
  instead of extracting one nibble at a time.
- **Activation compression**: Quantize pre-rotated activation to fp16 or int8
  to reduce bandwidth (loses some precision but may be worth it).
- **Stream-K style decomposition**: Different work partitioning across warps.
- **Per-block centroid pre-scale**: Pre-multiply centroid × d_half into a
  16-entry fp16 LUT in registers (not shmem). Then inner loop is just
  `lut[idx] * activation` — one FMA instead of two multiplies.

## Constraints
- Must not change the block_tq4_1s ABI (format is shared with Metal/CPU)
- Must not modify any file other than mmvq-tq.cu
- Output must be bit-exact for the same input (deterministic)
- Must work on both Blackwell (sm_120) and Ampere (sm_86)
