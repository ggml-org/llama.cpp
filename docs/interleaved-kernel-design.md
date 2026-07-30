# Interleaved Intra-Kernel Workload Design

Status: DRAFT
Date: 2026-07-30
Depends: kernel_TILE640_MATMUL (ggml-metal.metal:11360)

## 1. Motivation

kernel_TILE640_MATMUL is memory-bound. The base-3 decode of a 640-weight
page is ~40-60 ALU cycles on SIMD group 0 while SIMD groups 1-3 wait at
the threadgroup barrier. With T640_TOKEN_TILE=4, three of four SIMD groups
are idle during every page decode phase. Over a full row (nt pages), this
is a substantial fraction of wall time spent doing nothing.

Two workloads can fill this idle time:

1. Speculative drafter (dflash/dspark) small GEMMs - compute-bound, uses
   idle ALUs during memory-bound decode windows.
2. KV cache operations (quantization, eviction scoring) - low-register,
   embarrassingly parallel, fills remaining idle time after drafter
   saturates (diminishing returns past 2-3 draft tokens).

## 2. Idle Window Map

Note: the occupancy-optimized kernel_TILE640_MATMUL uses cooperative decode
(all SIMD groups reconstruct the page in parallel), which eliminates Window A.
The interleaved variant deliberately reverts to si==0-only decode to recreate
Window A as usable idle time for drafter/KV work. This is a throughput trade:
~3-4x longer decode phase, but the idle SIMD groups produce drafter tokens
and KV quantization for free. The runtime selects the kernel variant based
on whether speculative decoding or KV management is active.

Window A - Page decode (PRIMARY, interleaved variant only):
  Location: lines 11404-11450, si==0 decodes, si!=0 idle
  Duration: ~40-60 cycles per page, nt pages per row
  Available: SIMD groups 1..min(3, token_count-1)
  Constraint: must finish before barrier at 11451

Window B - Post-dot-product barrier:
  Location: line 11466, all groups sync before next page
  Duration: small (barrier latency only)
  Available: early-finishing SIMD groups
  Constraint: too small for meaningful work

Window C - Outlier addback divergence:
  Location: lines 11471-11489
  Duration: proportional to K_i variance across tokens
  Available: SIMD groups whose token has K_i=0
  Constraint: only fires when outlier counts differ across tokens

Primary target: Window A. Secondary: Window C.

## 3. Three-Tier Temporal Priority Schedule

P0: Tile640 dequant + matmul (the actual inference, never preempted)
P1: Speculative drafter small GEMM (compute-bound, fills memory-load stalls)
P2: KV cache ops (fills remaining stalls after drafter, from threadgroup)

Scheduling is TEMPORAL, not spatial. All SIMD groups cooperatively decode
the page (matching the base kernel's occupancy optimization). During the
dot-product loop, each thread issues its activation memory load (~200-400
cycle latency) and then executes one P1 or P2 instruction while waiting.
The GPU's out-of-order execution engine overlaps the independent compute
with the memory latency, so P1/P2 work is effectively free.

P1 takes priority over P2. When drafter_enabled is set, every dot-product
loop iteration includes one drafter FMA step. When the drafter is done (or
disabled), P2 KV quantization takes over.

KV lines are prefetched into threadgroup memory during the decode phase
(device memory -> threadgroup), then consumed from threadgroup during the
dot product. This avoids competing for HBM bandwidth with activation loads.

Drafter weights are read from device memory but the working set is small
(~1-5 MB for a 1-2 layer draft model) and fits in L2 cache.

## 4. Drafter Integration (P1)

### 4.1 Architecture

dflash/dspark are small draft models (typically 1-2 layers, hidden_dim
256-512). Their GEMM is a small matmul: [n_draft_tokens x hidden] *
[hidden x vocab_slice]. This is compute-bound and maps well to idle
SIMD groups.

### 4.2 Buffer Layout

Drafter weights are bound as additional kernel buffers:
  device const half* drafter_packed     // small Tile640 or dense
  device const half* drafter_bias
  device float*      drafter_hidden     // scratch for intermediate

Drafter state (current draft position, token count) via constant buffer:
  constant drafter_args & dargs [[buffer(N)]]

### 4.3 Scheduling

During Window A, SIMD groups 1-3 check a drafter_work_remaining flag.
If true, they execute one drafter GEMM tile per page decode cycle.
The drafter GEMM is decomposed into page-sized chunks that fit within
the decode window.

Pseudo-code for si != 0 during Window A:
  if (dargs.enabled && drafter_tiles_remaining > 0) {
      // P1: drafter GEMM tile
      drafter_matmul_tile(drafter_packed, drafter_hidden, ...);
      drafter_tiles_remaining--;
  } else if (kv_work_remaining > 0) {
      // P2: KV cache op
      kv_quant_tile(kv_cache, kv_scales, ...);
      kv_work_remaining--;
  }
  // else: idle (no work available)

### 4.4 Synchronization

Drafter results are NOT consumed by the current Tile640 kernel. They
are written to device memory for the NEXT kernel dispatch (the verify
step). No cross-threadgroup sync needed. Only constraint: finish
before the barrier at 11451.

## 5. KV Cache Operations (P2)

### 5.1 Target Operations

KV quantization (FP16 -> INT8 in-place):
  - Read 16 FP16 values from KV cache line
  - Compute max(abs) across the line -> scale
  - Quantize: round(val / scale) -> int8
  - Write back: int8 values + fp16 scale
  - Register cost: ~6 registers per lane
  - No cross-threadgroup sync

KV eviction scoring:
  - Read attention weight vector for candidate entries
  - Weighted sum -> importance score
  - Write score to eviction buffer
  - Register cost: ~4 registers per lane
  - Small dot product, structurally similar to drafter

KV compaction index remap:
  - After eviction, rewrite page table entries
  - Scalar ALU + scattered memory
  - Register cost: ~3 registers per lane

### 5.2 Buffer Layout

  device half*  kv_cache        // [n_layers * n_heads * seq_len * head_dim]
  device uchar* kv_quantized    // [n_layers * n_heads * seq_len * head_dim] INT8
  device half*  kv_scales       // [n_layers * n_heads * seq_len] per-line scale
  device float* kv_scores       // [seq_len] eviction importance scores

KV work descriptor via constant buffer:
  constant kv_work_args & kargs [[buffer(N+1)]]
  struct kv_work_args {
      uint32_t op_type;       // 0=quant, 1=evict_score, 2=compact
      uint32_t layer;
      uint32_t head;
      uint32_t seq_start;
      uint32_t seq_count;
      uint32_t head_dim;
  };

### 5.3 Scheduling

KV ops are chunked into tiles that fit within Window A. Each tile
processes one cache line (head_dim values, typically 64-128). A SIMD
group processes one tile per page decode cycle.

Priority within P2: quant > evict_score > compact (quant has the
highest throughput benefit per cycle).

### 5.4 Correctness Constraints

- KV quantization mutates the cache DURING inference. The current
  layer's KV entries must not be quantized while they are being read
  by the attention kernel. Solution: only quantize layers < current
  layer (already computed, no longer needed in FP16).
- Eviction scoring reads attention weights from the CURRENT layer's
  attention output. This is available after the attention kernel
  completes, before the FFN (where Tile640 runs). So eviction scoring
  is safe during FFN's Tile640 kernel.
- Compaction must not run concurrently with any KV read. Restrict to
  the last Tile640 layer's FFN, or use a separate dispatch.

## 6. Register Budget

Tile640 baseline: ~24 registers per thread (decoded_page in threadgroup,
acc + loop vars in registers).

Drafter tile: +8 registers (small GEMM accumulator + indices).
KV quant tile: +6 registers (max accumulator + quantize temps).
KV score tile: +4 registers (dot product accumulator).

Combined worst case (Tile640 + drafter): ~32 registers.
Apple GPU register file: 65536 registers per threadgroup (M-series).
At 64 threads: 1024 registers/thread max. We are well within budget.

Occupancy concern: threadgroup memory. decoded_page[640] = 2560 bytes.
Drafter scratch would add ~512 bytes. KV staging ~256 bytes. Total
~3328 bytes per threadgroup. At 32KB threadgroup memory limit, this
allows ~9 concurrent threadgroups per SM, which is fine.

## 7. Function Constants

FC_TILE640_INTERLEAVE + 0: drafter_enabled (bool)
FC_TILE640_INTERLEAVE + 1: drafter_hidden_dim (int)
FC_TILE640_INTERLEAVE + 2: drafter_max_tiles (int, duty cycle cap)
FC_TILE640_INTERLEAVE + 3: kv_ops_enabled (bool)
FC_TILE640_INTERLEAVE + 4: kv_op_type (int, 0=quant 1=evict 2=compact)

## 8. Acceptance Criteria

1. Drafter: tokens/sec with interleaved drafter vs sequential speculative
   decoding at matched acceptance rate. Target: >10% throughput gain from
   eliminated drafter dispatch overhead.
2. KV quant: KV cache memory reduction achieved with zero additional
   kernel dispatches. Measure: bytes saved per token generated.
3. Combined: tokens/sec with drafter + KV vs baseline Tile640. The KV
   ops must not reduce tokens/sec by more than 1% (they should be free).
4. Correctness: output logits bit-identical to non-interleaved kernel
   (drafter/KV results are side outputs, not consumed by P0).

## 9. Phasing

Phase 1: Drafter interleaving only (P0 + P1).
  - New kernel variant: kernel_TILE640_MATMUL_INTERLEAVED
  - Drafter buffers + scheduling in Window A
  - Benchmark vs sequential spec decode

Phase 2: KV quantization (P0 + P1 + P2-quant).
  - Add KV buffers + kv_quant_tile to idle scheduler
  - Layer-safety constraint (only quantize layers < current)
  - Benchmark: memory savings + throughput neutrality

Phase 3: KV eviction + compaction (P0 + P1 + P2-full).
  - Add eviction scoring + compaction
  - Requires attention-weight buffer from attention kernel
  - Benchmark: end-to-end generation with eviction active

## 10. Novelty Assessment

Intra-kernel speculative drafting: novel. Published speculative decoding
(Leviathan et al., Chen et al.) uses separate model forward passes. No
prior work injects draft-token GEMMs into a production GEMM kernel's
barrier windows.

Intra-kernel KV cache quantization: novel. FlashAttention (Dao et al.)
optimizes KV access patterns. vLLM/PagedAttention (Kwon et al.) manages
KV paging at the scheduler level. KVQuant (Hooper et al.) quantizes KV
caches but as a separate kernel pass. Nobody quantizes KV entries inside
the matmul kernel using stolen ALU cycles.

Three-tier priority scheduling within a single kernel dispatch: novel.
The closest prior art is persistent kernels (e.g., StreamK) that schedule
GEMM tiles across SMs, but these do not mix heterogeneous workloads
(inference + drafting + cache management) within a single threadgroup.
