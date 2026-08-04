# Interleaved Prefill-Dequant-Decode Kernel Design Study

Status: DRAFT (research and design study; no production code change required)
Date: 2026-07-31
Scope: Cross-phase GPU kernel interleaving for Tessera T640 inference on
        Apple Silicon (Metal) and NVIDIA (CUDA).
Related:
  - docs/interleaved-kernel-design.md (narrower: intra-kernel drafter/KV overlap)
  - docs/ane-backend-deep-study.md (heterogeneous ANE/GPU angle)
  - ggml/src/ggml-metal/ggml-metal.metal (kernel_TILE640_MATMUL, lines 11371-11500)
  - ggml/src/ggml-metal/ggml-metal-tile640-interleaved.metal (existing temporal
    interleaving prototype, 326 lines)
  - ggml/src/ggml-metal/ggml-metal-ops.cpp (host dispatch, lines 1765-1828)
  - ggml/src/ggml-cuda/mmq.cuh + mmq.cu (fused dequant + stream-K GEMM)
  - tools/quantize/tessera/tessera-format.h (T640 format spec)

## Table of Contents

1. Executive summary
2. Compute/memory profile analysis
3. Spatial occupancy analysis
4. Temporal occupancy analysis
5. Kernel design candidates (A: fused, B: pipelined streams, C: persistent)
6. Tessera T640-specific considerations
7. Related work
8. Hardware-specific feasibility (Metal, CUDA)
9. Concrete implementation plan for Tessera
10. Risk catalog
Appendix A: Source citations
Appendix B: Roofline numbers used in this study

## 1. Executive summary

### 1.1 The idea, restated

Three transformer-inference phases have complementary resource profiles:

| Phase    | Bottleneck        | ALU idle?  | Memory idle?  |
|----------|-------------------|------------|---------------|
| Prefill  | Compute (GEMM)    | low        | high (headroom)|
| Dequant  | Memory bandwidth  | high       | low           |
| Decode   | Memory bandwidth  | high       | low           |

The hypothesis: interleaving prefill compute with dequant/decode memory
transfers can keep both ALUs and memory controllers busy at the same time,
raising achieved throughput toward the roofline of whichever resource is
currently the limiter.

### 1.1.1 What "interleaving" can mean (three distinct things)

This study is careful to separate three meanings that are often conflated:

1. Intra-kernel temporal interleaving. One kernel. One thread. The thread
   issues a memory load with ~200-400 cycle latency, then executes
   independent ALU work (from a different logical task) while the load is
   in flight. The GPU's out-of-order issue hides the latency. This is what
   ggml-metal-tile640-interleaved.metal already does.

2. Spatial partitioning. One kernel, but different threadgroups (or SIMD
   groups) execute different tasks concurrently. Group A does GEMM tiles;
   group B does dequant. They rendezvous in shared memory.

3. Inter-kernel / inter-stream overlap. Multiple independent kernels on
   multiple command buffers (Metal) or CUDA streams. The hardware scheduler
   overlaps their execution if resources allow.

The three are not equally feasible. (1) is already working in-tree for
T640 decode. (2) is hard on both Metal and CUDA because the hardware
scheduler, not the programmer, maps threadgroups to SIMD pipelines, so
"spatial partitioning" is a request, not a guarantee. (3) is the
best-established mechanism on CUDA and partially supported on Metal.

### 1.2 Is it feasible?

| Mechanism                      | Metal (Apple)   | CUDA (NVIDIA) |
|--------------------------------|-----------------|---------------|
| Intra-kernel temporal (1)      | YES (in tree)   | YES           |
| Spatial partitioning (2)       | WEAK (no control)| WEAK (cooperative groups help) |
| Inter-stream overlap (3)       | PARTIAL         | YES (mature)  |

Feasible, yes. But the win is narrower than the headline "complementary
resources" framing suggests, for three reasons discovered below:

- During a single inference request the three phases are largely
  sequential per layer (you cannot decode layer N+1 until layer N is
  done), so within one request the phases rarely overlap. The overlap
  that exists is across layers (dequant N+1 while compute N) or across
  requests (decode of request A while prefill of request B).
- Dequant in T640 is already fused into the GEMM (see section 6.2). The
  standalone kernel_TILE640_DEQUANT exists for debug sidecars and the
  rare dense-emit path. So "overlap dequant with compute" is mostly
  already done by fusion; the residual standalone dequant work is small.
- Decode at batch=1 is so memory-bound that its ALU usage is ~1-3% of
  peak (section 2.3). There is huge ALU headroom in decode, but only if
  there is independent compute to fill it. The natural filler is a
  speculative drafter (already designed in docs/interleaved-kernel-
  design.md) or a second concurrent request.

### 1.3 Realistic speedup range

Numbers below are end-to-end tokens/sec or prefill tok/sec uplift, not
per-kernel micro-benchmarks. Reasoning is in the cited sections.

| Scenario                                   | Backend | Speedup   | Source |
|--------------------------------------------|---------|-----------|--------|
| Drafter GEMM hidden in T640 decode windows | Metal   | 1.08-1.20 | sec 5.1, existing prototype |
| Layer-N+1 dequant overlapped with layer-N  | CUDA    | 1.05-1.15 | sec 4.4, 5.2 |
| Layer-N+1 dequant overlapped with layer-N  | Metal   | 1.02-1.08 | sec 4.3 (limited concurrency) |
| Prefill of req B overlapped with decode A  | CUDA    | 1.20-1.60 | sec 4.4 |
| Prefill of req B overlapped with decode A  | Metal   | 1.10-1.30 | sec 4.3 |
| GPU-prefill while ANE-decodes (hetero)     | Apple   | 1.30-2.00 | sec 8.1.5 |

The single largest realistic win is heterogeneous: run decode on the ANE
(in progress, D1) and prefill on the GPU concurrently. This is the only
configuration where the two phases genuinely use disjoint hardware with
no scheduler contention.

### 1.4 Fundamental obstacles

1. Cross-layer data dependency. Layer N+1 needs layer N's output. The
   only thing that can be overlapped with layer N's compute is work for
   layer N+1 that does not depend on N's activations - i.e., loading
   and dequanting weights for N+1. That is a real opportunity (sec 4.5)
   but it is prefetch, not arbitrary overlap.

2. Memory bandwidth, not ALU, is the limiter for dequant and decode.
   Overlapping two memory-bound tasks does not double throughput; it
   splits the same bandwidth. The win from overlapping prefill
   (compute-bound) with dequant (memory-bound) is genuine, but only
   because prefill leaves memory bandwidth headroom.

3. Kernel launch and synchronization cost. On Metal, dispatch overhead
   is ~5-20 us per kernel. Fine-grained per-layer interleaving at the
   command-buffer level can spend its budget on launch overhead. The
   persistent-kernel approach (Candidate C) avoids this but is
   constrained on Metal by max thread counts and shared-memory limits
   (sec 5.3, 8.1).

4. The hardware scheduler does what it wants. On Metal you cannot tell
   two SIMD groups to run different code; you can only write one kernel
   and let the scheduler fill lanes. Spatial partitioning (Candidate A
   as "two tasks in one kernel") therefore collapses into temporal
   interleaving in practice (sec 3.2, 5.1).

5. Correctness fragility. The existing interleaved prototype (sec 5.1)
   must keep P0 (the matmul) bit-identical to the base kernel. Adding
   overlap work increases register pressure and can spill, which
   silently regresses P0 throughput. This is the dominant practical
   risk (sec 10).

### 1.5 Bottom line

The idea is feasible and partially already implemented. The realistic
prize is:

(a) 8-20% decode throughput from hiding the speculative drafter inside
    decode windows (Metal, in progress per docs/interleaved-kernel-
    design.md);
(b) 5-15% from cross-layer weight/dequant prefetch on CUDA streams;
(c) 30-100% from a heterogeneous GPU-prefill + ANE-decode split, which
    is the only path that uses truly disjoint hardware.

The spatial-partitioning vision (two different kernels running on
different SIMD groups at the same time within one dispatch) is the
weakest mechanism and should not be the primary investment.

## 2. Compute/memory profile analysis

We use a roofline framing. Arithmetic intensity AI = FLOPs / byte of
memory traffic. The ridge point AI* = peak_FLOPS / peak_bytes_per_sec
separates the memory-bound (AI < AI*) from compute-bound (AI > AI*)
regime.

Reference hardware (Appendix B has the citations):

| Device     | Peak FP16 tensor | Peak FP32  | Mem BW   | AI* (FP16 tensor) | AI* (FP32) |
|------------|------------------|------------|----------|-------------------|------------|
| Apple M3 Max 40c | ~14 TFLOPS  | ~3.5 TFLOPS | 400 GB/s | 35 FLOP/byte | 8.75 |
| Apple M4 Max 40c | ~17 TFLOPS  | ~4.3 TFLOPS | 546 GB/s | 31 FLOP/byte | 7.9  |
| NVIDIA H100 SXM5 | 989 TFLOPS dense | ~67 TFLOPS | 3.35 TB/s | 295 FLOP/byte | 20 |
| NVIDIA A100 80GB | 312 TFLOPS | 19.5 TFLOPS | 2.0 TB/s | 156 FLOP/byte | 9.75 |

Notes on the table. Apple GPU "TFLOPS" figures for FP16 assume the
GPU's SIMD FMAs and are less standardized than NVIDIA tensor-core
numbers; treat them as +/- 30%. The H100 number is dense FP16 tensor;
with 2:4 sparsity it doubles. The Apple GPU has no tensor cores; its
matmul throughput comes from wide SIMD FMAs and (M3+) the matrix-
multiply-accumulation "AMX" coprocessor reachable via simdgroup_matrix
in Metal. AMX lifts effective FP16 throughput materially but Apple does
not publish clean peak numbers; community measurement puts M3 Max 40c
in the 14-22 TFLOPS FP16 range for GEMM-shaped workloads.

### 2.1 Prefill

Prefill is a tall-skinny GEMM: weights W [out_dim, in_dim] times
activations A [in_dim, n_prompt_tokens]. For T640 the weights are
stored packed-ternary and dequanted on the fly, but the dequant is
fused into the GEMM (sec 6.2), so for roofline purposes we count the
packed-weight bytes read.

Per output element: 2*in_dim FLOPs. Per (W tile read once, reused
across n tokens): in_dim * 2 bytes packed (roughly; T640 is 1.6 bpw
ternary + ~1.5 bpw outliers + scales, call it ~2 effective bpw for
roofline) and the activation block contributes n_tokens * in_dim * 2
bytes (fp16). So:

  AI_prefill ~ (2 * in_dim * n_tokens) / (in_dim * 0.25 + n_tokens * in_dim * 2)
             ~ 2*n_tokens / (0.25 + 2*n_tokens)
             -> 1.0 as n_tokens grows  (in FLOP per packed-weight-byte)

That looks memory-bound, which is misleading: the weight is read once
and reused across n_tokens, so the right denominator for large
n_tokens is dominated by activations, not weights. Re-arranging per
output element and counting the weight amortization:

  bytes_per_output_elem ~ (2 bytes_packed_weight / n_tokens) + 2 bytes_activation
  FLOP_per_output_elem  ~ 2*in_dim
  AI ~ 2*in_dim / (2 + 2/n_tokens) ~ in_dim for large n_tokens

With in_dim = 3072-8192 (typical), AI is in the thousands. Prefill is
firmly compute-bound for n_tokens above a modest threshold.

Transition point. Solve for AI = AI*. On M4 Max with FP32 math
(AI* = 7.9): compute-bound when in_dim * f(n) > 7.9, which holds for
any realistic in_dim. More useful is the batch at which the GEMM
saturates memory bandwidth. Empirically (and consistent with the
roofline), prefill on Apple Silicon becomes compute-bound at around
n_tokens >= 64-128 for hidden_dim >= 2048. Below that it is bandwidth-
limited by the activation traffic and the weight-streaming cost.

On H100 (AI* FP16 tensor = 295), the picture in fp16-tensor math: the
ridge is much higher, so prefill stays compute-bound only when n_tokens
is large enough to amortize weight reads; for small prompts on a 70B
model you are still bandwidth-limited streaming the weights once.

Occupancy during prefill:
- Metal: high. The kernel_TILE640_MATMUL dispatches (out_dim, ceil(n/
  4), n_batch) threadgroups of up to 128 threads (4 SIMD groups).
  With out_dim in the thousands and 4 tokens per group, the grid is
  large enough to fill all GPU cores. ALU utilization is high during
  the dot-product loop; the cooperative decode phase is a brief
  memory+ALU spike per page.
- CUDA: the MMQ kernel (mmq.cuh) selects nthreads=128/256, I/J SRAM
  tiles, and stream-K for small J. Occupancy is tuned per arch
  (mmq-config-ampere/blackwell/cdna/rdna2/rdna4/pascal.cuh). At large
  n_tokens, occupancy is high and the kernel is FMA/tensor-core bound.

Prefill leaves memory-bandwidth headroom. On M4 Max at ~17 TFLOPS FP16
and 546 GB/s, a compute-bound GEMM at the roofline uses all 17 TFLOPS
but only a fraction of 546 GB/s. Roughly, if AI realized is ~30
FLOP/byte (well past the ridge), bandwidth used is peak_FLOPS / AI =
17e12 / 30 = ~567 GB/s - i.e., prefill at the compute roofline
saturates bandwidth too. So in practice the headroom during prefill is
modest on Apple Silicon because the GEMM is already pushing both. On
H100 with FP8 tensor at 1979 TFLOPS dense and 3.35 TB/s, AI realized
in a tuned GEMM is ~300+, and bandwidth used is 1979e12/300 = ~6.6
TB/s - above the HBM3 limit, meaning the kernel is bandwidth-aware
and does not actually realize 1979 TFLOPS; it realizes closer to
3.35e12 * 300 = ~1 PFLOPS. The takeaway: on both backends, a well-
tuned prefill GEMM keeps bandwidth busy. There is not a large idle
memory controller to fill with dequant work during prefill.

This is the first important negative finding: the assumption that
"prefill is compute-bound so memory is idle" is only weakly true.
A well-optimized GEMM is balanced. The real headroom is ALU idle
during decode/dequant, not memory idle during prefill.

### 2.2 Dequant (T640)

Source: kernel_TILE640_DEQUANT (ggml-metal.metal lines 11687-11708)
and tile640_decode_element (lines 11621-11659). Format spec:
tools/quantize/tessera/tessera-format.h.

T640 layout per weight:
- page_size = 640, lane_size = 20, lane_scale_bits = 8.
- 32 lanes per page. Each lane is 20 ternary trits.
- 20 trits packed in radix-243 groups: 4 groups of 5 trits per 32-bit
  word, so 32 words per page (T640_WORDS_PER_PAGE = 32). That is
  128 bytes of packed trits per 640 weights = 1.6 bits/weight.
- page_scales: 1 half per page (2 bytes / 640 = 0.025 bpw).
- lane_scales: 1 uchar per lane = 32 bytes / 640 = 0.4 bpw.
- outliers: CSR sparse, ~0.5% of weights at fp16. Effective ~1.5 bpw
  for a 0.5% outlier density at 2 bytes each (0.005 * 16 = 0.08, plus
  index overhead, call it 1.5 bpw including indices per the kernel
  comment at ggml-metal.metal:11294).
- Total effective: ~1.6 + 0.025 + 0.4 + 1.5 = ~3.5 bpw.

Memory access pattern for standalone dequant (kernel_TILE640_DEQUANT):
- 1 thread per output element. gid -> (row, col).
- Reads: 1 packed word (4 bytes, but coalesced 32 words/page), 1
  page_scale (2 bytes), 1 lane_scale (1 byte), and scans the row's
  outlier CSR range (offsets, cols, vals) linearly per row.
- Writes: 1 float (4 bytes).

Bytes read per byte written. Per page (640 weights -> 640 floats out
= 2560 bytes out):
  - packed: 128 bytes
  - page_scale: 2 bytes
  - lane_scales: 32 bytes
  - outliers for the whole row: amortized small. Per-page share of
    outliers for a 0.5% density over in_dim columns: ~3.2 outliers
    * (4 byte col + 2 byte val) = ~19 bytes.
  - total read ~ 181 bytes per 2560 bytes written.
  - So the kernel reads ~0.07 bytes per byte written. It is
    overwhelmingly write-dominated at the page granularity, but the
    reads are scattered (CSR outlier scan, lane_scale gather) while
    the writes are streaming. The bottleneck is the read latency of
    the packed words and the lane-scale gather, not write bandwidth.

ALU work per element (tile640_decode_element, lines 11621-11659):
- Trit extraction: in the standalone scalar path it is a loop of
  trit divisions by 3 (lines 11642-11644), O(trit) integer divides
  per element - expensive. In the MATMUL kernel the vectorized path
  uses tile640_trit with the T640_TRIT5_LUT, turning 20 serial divs
  into 4 LUT lookups (lines 11345-11350). That is ~4 LUT reads + a
  few shifts/masks per element.
- Scale: 1 fmul (page_max * lane_scale * 1/127).
- Outlier scan: a linear search through the row's CSR range. For
  rows with K_i outliers this is O(K_i) comparisons per element,
  which is the dominant cost when outlier density is non-trivial.

Arithmetic intensity of standalone dequant:
  FLOPs: ~2-4 (scale + sign select) per element, plus the outlier
  scan comparisons (not floating point but ALU).
  Bytes touched: ~0.28 bytes read + 4 bytes write ~ 4.3 bytes.
  AI ~ 1 FLOP / 4.3 bytes ~ 0.23 FLOP/byte.

Against AI* ~ 8 (FP32 on M4 Max), AI = 0.23 is deep in the memory-
bound regime. Standalone dequant is unambiguously memory-bound, and
more specifically write/scan-latency-bound rather than read-bandwidth-
bound.

Is there ALU work? Yes:
- Trit unpacking (LUT + shifts): meaningful integer ALU.
- Scale multiply: FP ALU.
- Outlier addback: FP ALU plus comparisons.

But the amount is small relative to the memory traffic, so ALUs sit
idle most of the dequant wall time. This is the idle-ALU window the
interleaving idea wants to fill.

Critical caveat (section 6.2): in the production path, dequant is
fused into the GEMM (kernel_TILE640_MATMUL decodes each page into
threadgroup memory and immediately consumes it, lines 11420-11469).
The standalone kernel_TILE640_DEQUANT runs only for the debug sidecar
(metal-dump-dequant, ggml-metal-ops.cpp:1780) and the dense-emit
path. So "overlap dequant with compute" as a separate phase is a
small opportunity because most dequant is already fused.

### 2.3 Decode

Decode (batch=1) is a matrix-vector product W * x where x is one
column. Per output element: 2*in_dim FLOPs. The whole weight matrix W
is streamed once (out_dim * in_dim * bytes_per_weight). The activation
vector is in_dim * 2 bytes, negligible.

  AI_decode ~ 2 * in_dim * out_dim / (out_dim * in_dim * 0.4375 bytes)
            ~ 2 / 0.4375
            ~ 4.6 FLOP/byte
  (using 3.5 bpw = 0.4375 bytes/weight for T640)

Against AI* ~ 8 (FP32 M4 Max) or ~7.9, AI = 4.6 is below the ridge:
decode is memory-bound. Against AI* ~ 31 (FP16 M4 Max with AMX),
decode is far below: heavily memory-bound.

On H100 with FP16 tensor (AI* = 295), decode at AI = 4.6 is memory-
bound by a factor of ~64. This is why decode is described as "weight-
bandwidth limited."

How much of decode time is weight loading vs compute? With AI = 4.6
and AI* (FP32) = 8, the kernel spends ~AI*/AI is not the right ratio;
the right statement is: the kernel is memory-bound, so wall time is
dominated by weight streaming. Compute time at peak FP32 is
2*in_dim*out_dim / peak_FLOPS. Memory time is bytes / peak_BW. The
ratio compute/memory = AI / AI* = 4.6/8 = 0.58 on M4 Max FP32, and
4.6/31 = 0.15 in the FP16/AMX regime. So compute is 15-58% of the
memory time; the rest of the time the ALUs are idle waiting for
weights. That is the window.

When does decode become compute-bound? Solve AI = AI*. With FP16/
AMX at AI* ~ 31 on M4 Max, decode stays memory-bound until AI rises
above 31, which requires batch n_tokens such that weights are reused
n_tokens times: AI grows ~linearly with n_tokens (weight amortization).
AI(n) ~ 4.6 * n. Set 4.6*n = 31 -> n ~ 7. So on Apple Silicon decode
becomes compute-bound around batch 7-16 (consistent with community
benchmarks). On H100 (AI* = 295), n ~ 64. This matches the well-known
rule that LLM decode on H100 is memory-bound up to batch ~64-128.

Occupancy during decode:
- Metal: the host dispatch shrinks the threadgroup to 1 SIMD group
  for n_tokens < 4 (ggml-metal-ops.cpp:1822, simdgroups_per_tg =
  min(token_tile, max(1, n_tokens))). So at batch=1, each threadgroup
  is 32 threads = 1 SIMD group, and only 1 of the 4 token slots is
  used. The grid is (out_dim, 1, n_batch*n_seqs) - large enough to
  fill the GPU, but each group does little compute per page and
  spends most time streaming weights. ALU utilization per group is
  low; the GPU as a whole may show "active" SMs but they are stalled
  on memory.
- CUDA: analogous. The MMQ path is replaced by mmvq (matrix-vector)
  for batch=1; mmvq uses fewer threads per block and is bandwidth-
  bound. ALUs idle.

Summary table (the single most important table for the rest of the
study):

| Phase        | AI (FLOP/byte) | Bottleneck | ALU util (typ) | Mem util (typ) |
|--------------|----------------|------------|----------------|----------------|
| Prefill (large) | 1000+       | compute    | 60-90%          | 60-90% (balanced) |
| Prefill (small) | 5-20        | memory     | 20-40%          | 70-90% |
| Dequant T640 | 0.23           | memory     | 5-15%           | 60-80% (write+scan) |
| Decode batch=1 | 4.6         | memory     | 5-15%           | 70-90% |
| Decode batch=16 (Apple) | ~70 | compute | 60-80%        | 60-80% |

## 3. Spatial occupancy analysis

### 3.1 Metal SIMD model

Apple GPU: one "GPU core" has multiple SIMD pipelines. The unit of
scheduling is the SIMD group of 32 threads (the Metal simdgroup). A
threadgroup contains 1+ SIMD groups (up to max_total_threads_per_
threadgroup, here 128 = 4 SIMD groups per the kernel attribute at
ggml-metal.metal:11370).

The M4 Max 40-core GPU has 40 "GPU cores" * 16 "execution units" per
core region (Apple's terminology is inconsistent across sources); the
practical number for scheduling is that many threadgroups run
concurrently across the GPU, each with its own SIMD groups.

How many SIMD groups active per phase:
- Prefill: each threadgroup has 4 SIMD groups (token_tile=4), and the
  grid has out_dim * ceil(n/4) * batch threadgroups. For a 4k out_dim
  and 256 prompt tokens, that is 4096 * 64 = 262144 threadgroups. The
  GPU runs as many concurrently as resources allow (limited by
  threadgroup memory: decoded_page[640] floats = 2560 bytes per
  group). Effectively all SIMD pipelines are busy.
- Dequant (standalone kernel_TILE640_DEQUANT): 1 thread per element,
  flat grid. Tons of threadgroups, each lightly loaded. SIMD pipelines
  busy but mostly stalled on memory.
- Decode batch=1: each threadgroup is 1 SIMD group (32 threads), grid
  is out_dim * 1 * 1. SIMD groups are "active" (scheduled) but stalling
  on memory.

### 3.2 Can you put two different tasks on different SIMD groups in one kernel?

This is the crux of Candidate A (spatial). The answer on Metal is:
not directly. Within a single kernel invocation, every SIMD group in
a threadgroup executes the same program counter sequence. You can
branch on simdgroup_index_in_threadgroup (si) so that si==0 does one
thing and si!=0 does another - the existing interleaved prototype
mentions this (docs/interleaved-kernel-design.md:43-50, the "si==0
decodes, si!=0 idle" pattern). But:

- That is temporal divergence within a barrier window, not two
  independently-scheduled tasks. All SIMD groups still sync at the
  threadgroup barrier, so the "idle" groups must finish their
  alternate work before the barrier or they hold up si==0.
- Metal's hardware scheduler, not the programmer, decides which
  threadgroups run on which core and in what order. You cannot pin
  "GEMM threadgroups to cores 0-19 and dequant threadgroups to cores
  20-39." There is no Metal API for spatial core reservation.

The practical consequence: "spatial partitioning" on Metal is really
just temporal interleaving with si-based branching inside one
threadgroup. The existing ggml-metal-tile640-interleaved.metal is
exactly this. There is no separate spatial path to exploit.

There is one genuine spatial mechanism on Metal: dispatch two
independent kernels concurrently via separate command buffers or
MTLDispatchTypeConcurrent (sec 4.3). The hardware scheduler can then
place them on different cores. But you do not control the placement,
and they compete for shared resources (memory bandwidth, L2).

### 3.3 CUDA SM model

NVIDIA H100 SXM5: 132 SMs, 4 warp schedulers per SM = 528 warp slots.
Max warps per SM is 64 (Hopper) = 8448 warps in flight max. A thread
block occupies one SM (or more for cooperative grids).

- Prefill: the MMQ kernel uses 128 or 256 threads per block (4-8
  warps). With large J (n_tokens), many blocks fill all 132 SMs at
  high occupancy (multiple blocks per SM). Compute-bound.
- Dequant standalone: low occupancy per SM, bandwidth-bound.
- Decode batch=1: mmvq uses small blocks; few warps active per SM;
  bandwidth-bound. Many SMs idle (low occupancy) because there are
  not enough blocks to fill 132 SMs for a single vector.

### 3.4 CUDA spatial partitioning

Better story than Metal, via cooperative groups. You can launch a
single grid where thread blocks self-identify (blockIdx) and branch:
"if blockIdx.x < N_gemm, do GEMM tile; else do dequant." This is the
"Candidate A fused" approach and it works because CUDA lets you write
heterogeneous work into one kernel and the block scheduler distributes
blocks across SMs. Two blocks on the same SM can be from different
branches.

But: blocks on the same SM share the SM's memory bandwidth and
register file. Putting a bandwidth-bound dequant block next to a
compute-bound GEMM block on the same SM does not give you free
dequant; it splits the SM's bandwidth. The win is real only when the
GEMM block is not saturating the SM's memory port - i.e., the GEMM is
compute-bound enough to leave bandwidth headroom (which section 2.1
showed is only weakly true for well-tuned GEMMs).

### 3.5 Is there spatial headroom to run dequant/decode on idle ALUs during prefill?

On Metal: no clean mechanism (sec 3.2). On CUDA: yes, via cooperative
groups + heterogeneous blocks, but the bandwidth-sharing caveat
applies. The honest answer is: spatial headroom exists on paper (idle
SIMD lanes during memory stalls), but capturing it requires either
intra-kernel temporal interleaving (sec 5.1, already in tree) or
cooperative-grid block branching on CUDA (sec 5.3). Pure spatial
partitioning as a separate dispatch is just inter-stream overlap
(sec 4.4), which is better understood.

## 4. Temporal occupancy analysis

Temporal occupancy = overlapping memory transfer latency with compute
on the same execution unit. This is what GPUs do natively via:
- Out-of-order issue: a memory load with 200-400 cycle latency does
  not block the ALU; independent instructions execute while the load
  is in flight.
- Memory-level parallelism: each thread can have multiple outstanding
  loads.
- Prefetch / async copy: explicit DMA-like copies (CUDA cp.async,
  Apple's blit command encoder) that run alongside compute.

### 4.1 Can dequant/decode weight loads overlap with prefill GEMM compute?

Mechanically yes. The question is whether the GEMM has ALU idle slots
to fill. Section 2.1 established that a well-tuned GEMM is balanced
(high ALU and high bandwidth simultaneously). So the headroom for
"free" overlap compute during prefill is small on both backends.

The bigger temporal win is the reverse: overlap prefill/decode
compute (ALU work) with dequant/decode weight loads (memory latency).
That is, hide memory latency behind compute, which is what every GEMM
kernel already does internally via double-buffering and instruction
scheduling. The existing kernel_TILE640_MATMUL does this: the
activation load (200-400 cycles) is issued, then the FMA chain runs
on previously-loaded data, then the new data arrives. The interleaved
prototype extends this by putting *different task* FMA work (drafter,
KV quant) into the latency window instead of just P0 FMA work.

### 4.2 Synchronization cost taxonomy

| Granularity           | Metal cost           | CUDA cost          |
|-----------------------|----------------------|--------------------|
| threadgroup_barrier   | ~tens of cycles      | __syncthreads ~tens of cycles |
| simd_sum/simd_max     | a few cycles         | __shfl + warp reduce, few cycles |
| Cross-threadgroup sync| not possible in-kernel | cooperative_groups grid sync: expensive (~us) |
| Cross-kernel (same queue) | serial, implicit | same stream: serial |
| Cross-kernel (concurrent) | shared event, ~5-20 us | cudaEvent + stream wait, ~5 us |
| Kernel launch         | ~5-20 us             | ~3-10 us (graph: ~0) |

Intra-kernel interleaving (Candidate A) avoids launch and cross-kernel
sync entirely. Its only sync is the existing threadgroup_barrier,
which the base kernel already pays. So the marginal sync cost of A is
~zero - which is why the in-tree prototype is attractive.

Inter-stream overlap (Candidate B) pays kernel-launch and event-sync
cost per overlapped unit. At one overlap per layer per request, that
is ~40 syncs for a 40-layer model - maybe 200-800 us total, small
relative to a multi-second generation. Fine. At one overlap per tile
or per layer-internal step, it dominates - not fine. So B must be
coarse-grained (per layer or per request).

Persistent kernel (Candidate C) pays launch once, then uses in-kernel
synchronization (grid sync on CUDA, not available on Metal). Metal
cannot do a true persistent megakernel because there is no grid-wide
barrier and the max total threads is bounded (sec 5.3).

### 4.3 Metal overlap patterns

Within a command buffer: a single compute command encoder serializes
its dispatches by default. Use dispatch_threads with MTLDispatchType
Concurrent to let multiple dispatches on the same encoder run
concurrently - but they share the encoder's resources and the
scheduler may still serialize if the first dispatch saturates the GPU.
Apple's guidance (WWDC22 "Load Resources Faster with Metal 3",
developer.apple.com/videos/play/wwdc2022/10104) is that concurrent
dispatch is useful when no single dispatch fills the GPU.

Across command buffers: enqueue multiple command buffers to a command
queue. The queue runs them with some concurrency (Apple does not
document the exact degree; community finding is 1-2 concurrent on
most Apple Silicon parts for compute-heavy work). Use MTLSharedEvent
to synchronize: buffer A signals an event on completion, buffer B
waits on it. This is the Metal analogue of CUDA streams + events.

Blit + compute overlap within one command buffer: a blit encoder and
a compute encoder on the same command buffer are recorded serially,
but the GPU can overlap their execution if they touch different
resources. In practice the overlap is unreliable for compute-compute
pairs; it is reliable for blit-compute when the blit is a DMA that
uses different hardware from the ALUs. So "prefetch weights with a
blit while compute runs" is a real Metal pattern. Apple explicitly
demonstrates this in the WWDC22 session.

So for Metal, the realistic temporal-overlap patterns are:
1. Intra-kernel temporal interleaving (in tree, sec 5.1).
2. Blit-prefetch of next-layer weights during current-layer compute
   (real, modest win, sec 4.5).
3. Concurrent decode-of-A with prefill-of-B across command buffers
   with shared events (real, scheduler-limited, sec 5.2).

What does NOT work well on Metal:
- Two compute kernels on concurrent dispatch expecting true parallel
  execution (often serialized by the scheduler).
- A persistent megakernel (no grid barrier, bounded threads).

### 4.4 CUDA overlap patterns

CUDA streams are the mature mechanism. cudaMemcpyAsync on stream 1
overlaps with a compute kernel on stream 0, provided the GPU has the
resources (and the copy uses the copy engine, not SMs). H100 has
dedicated copy engines, so async memcpy is genuinely free of SM
contention. This is the cleanest "memory vs compute" overlap on
either backend.

Two compute kernels on two streams overlap if combined occupancy <
100%. Two memory-bound kernels on two streams do not overlap
meaningfully (they split HBM bandwidth). A compute-bound kernel and
a memory-bound kernel overlap well - the compute kernel keeps SMs
busy, the memory-bound one keeps the copy engine / HBM busy. This is
the canonical case the user's hypothesis targets, and on CUDA it
works.

CUDA graphs reduce launch overhead to near-zero, making fine-grained
per-layer inter-stream pipelining practical. This is the production
mechanism used by vLLM, TensorRT-LLM, and others.

### 4.5 The cross-layer prefetch opportunity

The single most robust temporal-overlap win, available on both
backends, is prefetching and pre-dequanting layer N+1's weights while
layer N computes. This is robust because:

- It is a pure memory operation (read weights from DRAM into L2 or
  into a dequant staging buffer). Memory-bound.
- Layer N's compute is (for prefill, or for decode batch>1) compute-
  bound. So the two have complementary profiles by construction.
- There is no data dependency: layer N+1's weights do not depend on
  layer N's activations.

The catch: layer N+1's weight fetch competes with layer N's own
weight fetches and activation traffic for HBM bandwidth. If layer N
is already bandwidth-limited (small-batch decode), the prefetch
steals bandwidth and slows layer N - net zero. If layer N is compute-
bound (prefill, or decode batch > crossover), the prefetch is free.

So cross-layer prefetch is a prefill-time and large-batch-decode win,
not a batch=1-decode win. Estimated uplift: 5-15% on prefill-heavy
or large-batch workloads, ~0 on batch=1 decode (which is already
bandwidth-saturated).

## 5. Kernel design candidates

### 5.1 Candidate A: Single fused intra-kernel interleaving (TEMPORAL)

What it is: one kernel. The T640 matmul is P0 (must complete, must be
bit-identical). During the memory-latency windows inside P0, threads
execute independent FMA work from P1 (drafter) and P2 (KV cache ops).
This is exactly what ggml-metal-tile640-interleaved.metal implements.

Why "spatial" candidate A collapses to temporal: section 3.2 showed
Metal cannot pin tasks to SIMD groups. So the only way to get two
tasks in one kernel is temporal branch-by-si or temporal
branch-by-tid. The implementation is temporal even though the design
intent was spatial.

Mechanism in the existing prototype
(ggml-metal-tile640-interleaved.metal, kernel_TILE640_MATMUL_INTERLEAVED):

Pseudocode (Metal, abridged from the in-tree file):

```
kernel void kernel_TILE640_MATMUL_INTERLEAVED(...) {
    // P0 state
    float acc = 0.0f;
    // P1 state (drafter): each thread owns one (token, vocab) pair
    float drafter_acc = 0.0f;
    if (drafter_enabled) { /* assign (token, vocab) by tid */ }
    // P2 state (KV quant): one KV line per page iteration
    float kv_max_abs = 0.0f;

    for (p = 0; p < nt; ++p) {
        // --- cooperative decode of page p into threadgroup ---
        cooperative_decode(page p -> decoded_page[]);
        // --- P2 prefetch: load one KV line into threadgroup ---
        if (kv_enabled) prefetch_kv_line(p -> kv_prefetch[]);
        threadgroup_barrier();

        // --- dot-product loop with interleaving ---
        for (k = sl*4; k < page_cols; k += 128) {
            float4 a4 = load_activation4(input, ...); // ~200-400 cyc
            // P1: drafter FMA while a4 is in flight (L2-resident weights)
            if (drafter_enabled) {
                drafter_acc = fma(drafter_w[h], drafter_hs[h], drafter_acc);
            }
            // P2: KV max_abs reduce from threadgroup (no device load)
            else if (kv_enabled) {
                kv_max_abs = fmax(kv_max_abs, fabs(kv_prefetch[d]));
            }
            // P0: a4 has arrived -> FMA chain
            acc = fma(a4.x, decoded_page[k].x, acc); // x4
        }
        threadgroup_barrier();
        // --- P2: write quantized KV line ---
        if (kv_enabled) { kv_max_abs = simd_max(...); write_int8(...); }
    }
    // P1: reduce and write drafter logits
    if (drafter_enabled) { drafter_acc = simd_sum(...); write_logit(...); }
    // P0: outlier addback + reduce + write output (bit-identical to base)
    outlier_addback(...); acc = simd_sum(acc); output[i,j] = acc;
}
```

Expected occupancy:
- P0 occupancy unchanged from base kernel (the FMA chain is
  identical; the interleaved work is in the load-to-use window).
- P1 (drafter) uses idle ALU cycles during the activation-load
  latency. Drafter weights are L2-resident (1-5 MB working set), so
  no HBM bandwidth competition with P0.
- P2 (KV) uses threadgroup-resident data, so zero memory competition.
- Risk: register pressure. Base is ~24 regs/thread; drafter adds ~8,
  KV adds ~6, combined ~32. M-series allows ~1024 regs/thread at 64
  threads/group, so budget is fine. But spilling is the silent killer
  and must be measured (sec 10).

Synchronization complexity: LOW. Only the existing threadgroup_barrier
is used. No cross-threadgroup sync. P1/P2 outputs are side outputs to
device memory, consumed by later dispatches.

Implementation difficulty: 2-4 weeks for drafter (P1), 3-5 weeks for
KV quant (P2) including the layer-safety analysis (only quantize
layers < current). The P1 prototype file exists; wiring drafter
buffers through the host dispatch and the speculative-decoding
scheduler is the bulk of the work.

Hardware specificity: Metal (implemented). CUDA equivalent is
straightforward (same intra-kernel interleaving with warp-level
primitives). Works on both.

Pros:
- Zero launch overhead (one dispatch).
- P0 stays bit-identical (verifiable).
- Already partially implemented and benchmarked.

Cons:
- Only fills the activation-load latency window, which is a fraction
  of total decode time. Ceiling on P1/P2 throughput = (idle ALU during
  activation loads). Realistic ceiling: ~10-20% extra drafter tokens
  "for free," diminishing after that.
- Register pressure can silently regress P0 if it spills.
- Invasive: the kernel is now coupled to the speculative-decoding and
  KV-quant subsystems, raising testing burden.

### 5.2 Candidate B: Multi-kernel pipeline with async overlap (INTER-KERNEL)

What it is: separate kernels for prefill, decode, and dequant,
launched on overlapping command streams so that, e.g., dequant of
layer N+1 runs concurrently with the compute of layer N. Coarse-
grained (per layer or per request), not per tile.

Metal pseudocode (dispatch loop for one request, layer-pipelined):

```
// Per layer, two command buffers: compute (this layer) and prefetch
// (next layer weights via blit) + dequant-next-layer.
for (layer = 0; layer < n_layers; ++layer) {
    id<MTLCommandBuffer> cb_compute = [queue commandBuffer];
    id<MTLCommandBuffer> cb_prefetch = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc_c = [cb_compute computeCommandEncoder];
    id<MTLBlitCommandEncoder> enc_b = [cb_prefetch blitCommandEncoder];

    // Compute layer N (depends on layer N-1 output)
    [enc_c setComputePipelineState: matmul_pipeline];
    bind_layer_n_inputs(enc_c, layer);
    [enc_c dispatchThreadgroups: grid threadsPerThreadgroup: tg];
    [enc_c endEncoding];

    // Prefetch layer N+1 weights (independent of layer N output)
    if (layer + 1 < n_layers) {
        for (w in layer_weights[layer+1]) {
            [enc_b copyFromBuffer: w.source sourceOffset: 0
                          toBuffer: w.l2_staging destOffset: 0
                              size: w.size];
        }
    }
    [enc_b endEncoding];

    // Order: compute N must finish before compute N+1 starts, but
    // prefetch N+1 is independent. Use shared events only for the
    // compute chain.
    [cb_compute commit];
    [cb_prefetch commit];   // scheduler may overlap with cb_compute
    [cb_compute waitUntilCompleted]; // serializes the CPU loop
}
```

CUDA pseudocode (stream-pipelined, the mature pattern):

```
cudaStream_t s_compute, s prefetch;
for (layer = 0; layer < n_layers; ++layer) {
    // Prefetch N+1 weights async on s_prefetch (uses copy engine)
    if (layer + 1 < n_layers) {
        for (w in layer_weights[layer+1])
            cudaMemcpyAsync(w.l2_staging, w.source, w.size,
                            cudaMemcpyDeviceToDevice, s_prefetch);
    }
    // Compute layer N on s_compute
    matmul_kernel<<<grid, block, smem, s_compute>>>(
        layer_inputs[layer], layer_outputs[layer]);
    // Layer N+1 compute must wait for layer N compute (data dep)
    cudaEventRecord(layer_done[layer], s_compute);
    cudaStreamWaitEvent(s_compute, layer_done[layer]); // self for clarity
}
cudaStreamSynchronize(s_compute);
```

Expected occupancy:
- Compute stream: high during GEMM, same as standalone.
- Prefetch stream: uses copy engine (CUDA) or blit hardware (Metal),
  which is distinct from ALUs. So ALU occupancy of the prefetch is
  ~zero; it consumes HBM bandwidth.
- Net: when compute is compute-bound, prefetch is free. When compute
  is bandwidth-bound (batch=1 decode), prefetch competes and is not
  free.

Synchronization complexity: MEDIUM. Per-layer events on CUDA are
cheap (~5 us). On Metal, shared events work but the CPU-side
waitUntilCompleted in the pseudocode above serializes the loop; to
get real overlap you must enqueue several layers ahead without
blocking (double-buffered command buffer ring).

Implementation difficulty: 3-5 weeks on CUDA (streams + events are
standard; the work is in the layer-graph builder and weight-staging
allocator). 5-8 weeks on Metal (command buffer ring, shared event
wiring, scheduler-behavior validation).

Hardware specificity: CUDA (mature, recommended first). Metal
(possible, lower confidence in overlap degree).

Pros:
- Non-invasive: each kernel stays simple. No coupling between
  subsystems.
- Composable with existing graph capture (CUDA) and Tessera's op
  scheduling.
- Captures the cross-layer prefetch win (sec 4.5), which Candidate A
  does not (A only fills intra-layer windows).

Cons:
- Launch + sync overhead at fine granularity (must be coarse, >=1
  layer).
- Prefetch effectiveness depends on L2 capacity. H100 L2 is 50 MB;
  a layer of a 70B model at 3.5 bpw is ~30 MB - fits one layer, not
  two. Apple L2 is much smaller (M-series system-level cache is a
  few MB to ~24 MB on Max parts). So prefetch reaches at most one
  layer ahead.
- Does not help batch=1 decode (already bandwidth-saturated).

### 5.3 Candidate C: Persistent megakernel

What it is: one kernel launched once, stays resident, drains a work
queue of (prefill tile, dequant task, decode step) items. The Flash-
Attention / Stream-K / CUTLASS persistent pattern. The kernel owns
the SMs for the whole inference batch.

CUDA pseudocode (cooperative groups, persistent):

```
__global__ void megakernel(WorkQueue* q) {
    auto grid = cooperative_groups::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (true) {
        WorkItem w = q->claim_next(); // atomic counter
        if (w.type == DONE) break;
        switch (w.type) {
            case PREFILL_TILE: prefill_tile(w); break;
            case DEQUANT_BLOCK: dequant_block(w); break;
            case DECODE_STEP: decode_step(w); break;
        }
        grid.sync(); // expensive, ~us, only if needed for deps
    }
}
// Launch with cooperative launch to enable grid.sync():
cudaLaunchCooperativeKernel(megakernel, grid, block, args);
```

Metal pseudocode (constrained - see obstacles):

```
// Metal has no grid-wide barrier and bounds max_total_threads_per_
// threadgroup. A "persistent" kernel is really a long-running kernel
// where each threadgroup loops over a device-side work counter.
kernel void persistent_tessera(device atomic_uint* work_q, ...) {
    while (true) {
        uint slot = atomic_fetch_add_explicit(work_q, 1u,
                                              memory_order_relaxed);
        if (slot >= total_work) break;
        WorkItem w = decode_work(slot);
        switch (w.type) {
            case PREFILL_TILE: prefill_tile(w); break;
            case DEQUANT_BLOCK: dequant_block(w); break;
        }
        // NO grid-wide sync: cross-tile deps must be encoded in the
        // work queue ordering (claim only happens after deps done).
    }
}
```

Expected occupancy:
- CUDA: high and stable. The grid owns all SMs; no launch overhead
  between work items. This is the key win. Stream-K (already in
  ggml-cuda/mmq.cuh:1395-1437) is a limited form of this for the
  GEMM alone.
- Metal: bounded. Apple GPUs limit threadgroup count and have no grid
  barrier, so cross-work-item dependencies must be expressed via the
  atomic work counter and device-memory flags. This works for
  independent work items but cannot express "decode layer N+1 after
  layer N output is ready" without polling.

Synchronization complexity: HIGH. Work queue design, dependency
tracking in device memory, polling-based sync on Metal, cooperative
grid sync on CUDA. Debugging is hard.

Implementation difficulty: 8-16 weeks. The hardest of the three.
CUTLASS-style persistent GEMM is a known art but integrating it with
T640 dequant, attention, and the scheduler is a large project.

Hardware specificity: CUDA strongly preferred. Metal possible but
hobbled (no grid sync, smaller thread limits). Apple's guidance is
that persistent kernels are not the intended Metal pattern; the
intended pattern is many short kernels with the scheduler doing the
work distribution.

Pros:
- Eliminates per-kernel launch overhead entirely.
- Enables fine-grained work stealing and load balancing across SMs.
- The right long-term home for a fused T640+attention+drafter pipeline
  on CUDA.

Cons:
- Large engineering investment.
- Brittle (work queue bugs, device-side sync).
- Metal support is weak; the in-tree ggml-metal path cannot easily
  adopt it.
- The Stream-K variant for GEMM alone is already in tree
  (mmq.cuh); generalizing it to multi-phase is the expensive part.

### 5.4 Candidate comparison matrix

| Property                    | A (fused, intra) | B (streams)   | C (persistent) |
|-----------------------------|------------------|---------------|----------------|
| Launch overhead             | none             | per layer     | none           |
| Sync complexity             | low              | medium        | high           |
| Cross-layer overlap         | no               | yes           | yes            |
| Intra-layer overlap         | yes              | no            | yes            |
| Metal feasibility           | high (in tree)   | medium        | low            |
| CUDA feasibility            | high             | high          | high           |
| Coupling/invasiveness       | high             | low           | high           |
| Time to prototype           | exists           | 3-5 wk        | 8-16 wk        |
| Best realistic speedup      | 1.08-1.20 (decode)| 1.05-1.30    | 1.15-1.40      |
| Helps batch=1 decode?       | yes (drafter)    | no            | yes            |
| Helps prefill?              | marginal         | yes (prefetch)| yes            |

## 6. Tessera T640-specific considerations

### 6.1 T640 memory access pattern (recap from sec 2.2)

Per page (640 weights): 128 bytes packed trits + 2 bytes page scale +
32 bytes lane scales = 162 bytes metadata-and-trits, plus the page's
share of outliers. Output (if standalone dequant): 640 * 4 = 2560
bytes float. In the fused GEMM path, the page is decoded into
threadgroup memory (decoded_page[640] float = 2560 bytes, ggml-metal.
metal:11412) and consumed immediately, so it never round-trips through
device memory.

The radix-243 packing (T640_TRIT5_LUT at ggml-metal.metal:11317) is
the key ALU cost: extracting one trit costs a LUT read + shift (4
groups per word, lines 11345-11350). This is cheap enough that the
dequant is memory-bound, not ALU-bound.

### 6.2 Can T640 dequant be fused INTO the prefill GEMM?

Yes - it already is. kernel_TILE640_MATMUL (ggml-metal.metal:11371)
does exactly this: it never produces a standalone dequantized weight
matrix. Each page is decoded into threadgroup memory and consumed by
the dot-product loop in the same kernel. This is the Marlin-style
"fused quant GEMM" pattern (sec 7.3) applied to ternary weights.

So the question "can dequant be fused into prefill" is answered: it
is fused, and this is the production path. There is no separate
dequant to overlap with prefill for the GEMM itself.

What remains as a "dequant phase" is:
- The debug sidecar (metal-dump-dequant, ggml-metal-ops.cpp:1780),
  which is off in production.
- The dense-emit path (kernel_TILE640_DEQUANT for full-matrix output),
  used when a downstream op needs the dequantized weight as a dense
  tensor (rare).
- Per-layer weight prefetch from DRAM into L2 (Candidate B, sec 5.2),
  which is a copy, not a dequant.

The implication: the user's mental model of "three phases, dequant is
a separate memory-bound phase to overlap" does not match the actual
Tessera architecture, where dequant is fused. The overlap opportunity
is smaller than the framing suggests.

### 6.3 Fusing dequant into decode, and cross-layer dequant overlap

Decode uses the same fused kernel_TILE640_MATMUL with n_tokens=1
(host shrinks to 1 SIMD group, ggml-metal-ops.cpp:1822). So decode is
also already fused-dequant. There is no separate decode-time dequant
to overlap.

Cross-layer dequant overlap (dequant layer N+1 while computing layer
N) is meaningful only if dequant were a standalone pass. Since it is
fused into each layer's GEMM, "dequanting layer N+1 early" means
"running layer N+1's GEMM early," which requires layer N's output -
a circular dependency. So cross-layer dequant overlap is not
applicable to T640 as currently architected.

What IS applicable: cross-layer weight *prefetch* (copying packed
layer N+1 weights from DRAM toward L2 while layer N computes). The
packed weights have no data dependency on activations. This is
Candidate B's contribution, and it is a copy, not a dequant.

### 6.4 Does the outlier addback create a barrier that prevents interleaving?

The outlier addback (ggml-metal.metal:11471-11492) runs after the
main page loop and after the last threadgroup_barrier (line 11469).
It reads the CSR outlier arrays (outlier_row_offsets, outlier_cols,
outlier_vals) for the current row i and does scattered FMAs into acc.

It does NOT require a cross-threadgroup barrier. Within the
threadgroup, each thread (sl in its SIMD group) independently handles
its stride of the K_i outliers. The only sync is the final simd_sum
reduction (line 11494), which is intra-SIMD-group and cheap.

So the outlier addback does not prevent interleaving. It is itself a
candidate for overlap (it is memory-latency-bound on the CSR reads),
and the existing interleaved prototype leaves it as the P0 tail. The
CSR read latency could in principle hide P1/P2 work, but the addback
is usually short (K_i is small, ~0.5% density) so the window is tiny.

One subtlety: the addback's scattered reads can cause cache thrash if
interleaved work also hammers the cache. The prototype isolates P2
(KV) in threadgroup memory precisely to avoid this (ggml-metal-tile640
-interleaved.metal, kv_prefetch in threadgroup).

### 6.5 T640 and the radix-243 LUT: an ALU-cost note

The T640_TRIT5_LUT is a `constant` array of 243 ushorts (486 bytes,
ggml-metal.metal:11317-11339). It lives in the constant address space,
which on Apple Silicon is cached in the constant cache / texture
cache. LUT reads are cheap and do not compete with device-memory
bandwidth. This means the dequant ALU work is genuinely "free" of
memory-side cost, and interleaving additional ALU work into the
dequant phase is safe from a memory-contention standpoint - but the
dequant phase is short, so the window is small.

## 7. Related work

### 7.1 Flash Attention / Flash Decoding (Dao et al.)

- FlashAttention (Dao 2022; Dao 2023, arXiv 2205.14135 / 2307.08691):
  tiling strategy to keep attention matmuls in SRAM, maximizing
  arithmetic intensity. The tiling philosophy (own the SM, reuse
  data in SRAM) is the direct inspiration for Candidate C.
- Flash Decoding (Dao 2023 blog): split-K parallelism for decode -
  split the long K dimension across blocks to find parallelism in
  the otherwise-serial decode. Relevant to decode occupancy, not
  directly to cross-phase interleaving.

Neither paper addresses cross-phase (prefill vs decode vs dequant)
interleaving. They are about single-op occupancy.

### 7.2 CUTLASS (NVIDIA)

CUTLASS provides parameterized mixed-precision GEMM with fused
epilogues and, in recent versions, fused dequant (via the EVT -
epilogue visitor tree - and the collective::Epilogue). The Hopper
WGMMA path supports NVFP4/MXFP4 with on-the-fly dequant, which is
the same fusion idea as T640-in-GEMM. CUTLASS persistent kernels
(Stream-K, sec 7.6) are the reference for Candidate C.

### 7.3 Marlin (IST-DASLab)

- Paper: MARLIN: Mixed-Precision Auto-Regressive Parallel Inference,
  arXiv 2408.11743 (Frantar et al., 2024). Note: the charter cited
  2403.09899, which is not the Marlin paper; the correct id is
  2408.11743.
- Contribution: a fused FP16 x INT4 GEMM for autoregressive decode
  that reaches close to ideal 4x over cuBLAS for batch <= 16-32. Key
  tricks: weight layout reordered for warp-level access, dequant on-
  the-fly, pipelined stages, and careful register/shared-memory
  budgeting to keep SMs busy.
- Relevance to Tessera: Marlin is the proof that fusing dequant into
  a decode GEMM yields large wins for memory-bound decode. T640
  already does this fusion. Marlin's pipelining (multiple software
  stages to overlap loads and compute) is the same idea as Candidate
  A's intra-kernel interleaving, applied within one op rather than
  across ops.

### 7.4 Apple Metal Performance Shaders (MPS)

MPS does not expose a fused quant+GEMM for custom quantization formats.
MPSGraph has a MatMul op and supports some fusion via its graph
compiler, but T640's ternary layout is not representable in MPSGraph's
type system. This is why Tessera ships custom Metal kernels rather
than using MPS. MPS does implement its own internal kernel scheduling,
but it is opaque and not available for custom kernels. So MPS offers
no help for this design.

### 7.5 ggml-cuda K-quants fused kernel (in-tree)

The existing ggml-cuda path (ggml/src/ggml-cuda/mmq.cuh, mmq.cu,
dequantize.cuh) implements fused dequant+mulmat for all K-quants and
IQ-quants. The pattern: a single mul_mat_q kernel reads packed weights,
dequants into shared memory tiles, and does the matmul. This is
exactly the T640-in-GEMM fusion. The config tables
(mmq-config-ampere.cuh etc.) tune nthreads, I/J tiles, sram_layout,
and stream_k per (type, batch, arch). The stream_k option
(mmq.cuh:1395-1437) is a limited persistent-kernel pattern for the
GEMM alone - it distributes output tiles across SMs via a flat tile
counter and a fixup kernel for partial reductions.

This is the strongest in-tree precedent for Candidate C: the
infrastructure for tile-stealing work distribution exists, and
generalizing it to multi-phase work is the engineering challenge.

### 7.6 Speculative decoding kernel overlap

Published speculative decoding (Leviathan 2023, Chen 2023) runs the
draft and verify models as separate forward passes - no kernel-level
overlap. The Tessera docs/interleaved-kernel-design.md (in tree)
notes that intra-kernel speculative drafting is novel: no prior work
injects draft-token GEMMs into a production GEMM kernel's barrier
windows. That claim appears correct based on the related-work survey.
Candidate A is therefore genuinely novel in the academic sense, even
though its mechanism (temporal interleaving) is standard.

### 7.7 Triton (OpenAI)

Triton is a Python DSL that compiles to PTX/SPIR-V and makes
block-level programming easier. Its value here is that writing
Candidate A or C in Triton is much easier than in raw Metal/CUDA,
because Triton handles shared-memory tiling and barrier placement
automatically. Triton does not run on Metal (only NVIDIA/AMD GPUs and
CPU), so it is CUDA-only for Tessera. For the CUDA backend, a Triton
implementation of Candidate A or C is worth considering as a
maintainability win.

### 7.8 Papers on multi-phase / interleaved inference kernels

Direct prior work on interleaving prefill with decode at the kernel
level is scarce. The closest bodies of work are:
- Continuous batching / in-flight batching (vLLM, Orca, Sarathi):
  overlap prefill and decode of *different requests* at the
  scheduler level (not kernel level). This is the system-level
  analogue of Candidate B and is the most impactful known technique.
  Sarathi-Serve (Agrawal et al., 2023, arXiv 2308.16369) specifically
  proposes "chunked prefill" to interleave prefill and decode in a
  single batch to improve GPU utilization - this is the scheduler-
  level realization of the user's intuition.
- Disaggregated inference (Splitwise, DistServe): separate prefill
  and decode onto different GPUs. This is the cluster-level analogue
  and is orthogonal to single-GPU kernel interleaving.

The honest summary: at the kernel level, the user's idea (interleave
prefill compute with decode/dequant memory) is partially novel but
partially already subsumed by (a) fused dequant GEMMs (T640 itself,
Marlin) and (b) scheduler-level continuous batching. The remaining
kernel-level novelty is intra-kernel interleaving of *different
logical tasks* (Candidate A's drafter-in-decode), which the in-tree
prototype already pursues.

### 7.9 Tessera's own overlap work (in-tree)

- docs/interleaved-kernel-design.md (257 lines): the narrower design
  for Candidate A (drafter + KV quant inside T640 decode windows).
  This study subsumes and broadens it.
- ggml-metal-tile640-interleaved.metal (326 lines): the working
  prototype implementing Candidate A.
- docs/speculative.md: DFlash and DSpark drafters. These are the P1
  workloads Candidate A interleaves.
- docs/ane-backend-deep-study.md (1020 lines): the heterogeneous
  angle (sec 8.1.5 of this study). The ANE backend (D1, in progress)
  opens the GPU-prefill + ANE-decode possibility, which is the
  largest realistic win.
- docs/runtime-aware-pipeline.md / pipeline-design.md: calibration
  and telemetry, not directly overlap, but the Layer-1 kernel dequant
  fidelity work is what makes the fused-dequant path trustworthy.

## 8. Hardware-specific feasibility

### 8.1 Apple Silicon (Metal)

#### 8.1.1 Concurrent compute kernels on different SIMD groups

Section 3.2 established this is not directly controllable. Within one
kernel you can branch by si; across kernels you can use concurrent
dispatch or separate command buffers, but the scheduler decides
placement. There is no Metal API to reserve SIMD pipelines for a
specific task.

#### 8.1.2 Kernel launch overhead

dispatch_threadgroups costs ~5-20 us per call on current Apple Silicon
(community measurement; Apple does not publish the number). For a
40-layer model with ~6 ops/layer, that is ~240 dispatches per token
at batch=1 decode, or ~1.2-4.8 ms of pure dispatch overhead per token
- significant relative to a ~20-50 ms/token decode. This is a strong
argument for Candidate A (one dispatch) and Candidate C (one
persistent dispatch) over fine-grained Candidate B on Metal.

Indirect command buffers (Metal 3) reduce per-dispatch cost for
repeated identical kernels by encoding once and executing many times.
This helps decode (same kernels every token) and partially closes the
launch-overhead gap.

#### 8.1.3 Memory bandwidth and prefill headroom

M1 Max: 400 GB/s. M2 Max: 400 GB/s. M3 Max: 400 GB/s (16-CPU/40-GPU)
or 300 GB/s (14-CPU/30-GPU). M4 Max: 546 GB/s. M3 Ultra: 800 GB/s.
M5 Max: ~614 GB/s.

Section 2.1 showed a tuned prefill GEMM uses most of this bandwidth
while also using most of the ALUs. So "memory is idle during prefill,
let's fill it with dequant" is only weakly true on Apple Silicon.
The headroom is maybe 10-30% of bandwidth during compute-bound
prefill, and capturing it via blit-prefetch (Candidate B) is realistic
but modest (~5-10% uplift).

#### 8.1.4 The ANE backend (D1) and heterogeneous interleaving

This is the most promising Apple-specific angle. Per docs/ane-backend
-deep-study.md, the ANE is a separate coprocessor with its own
compute and its own memory path (shared DRAM via IOSurface, sec 4.3
of the ANE study). If decode runs on the ANE and prefill runs on the
GPU, the two use genuinely disjoint compute hardware, and the only
shared resource is DRAM bandwidth.

The ANE study (sec 4.3.2) documents the IOSurface zero-copy pattern
and MTLSharedEvent synchronization between ANE and Metal. The
infrastructure for GPU+ANE concurrent execution exists. The model
needs to be partitionable into a GPU prefill path and an ANE decode
path; the ANE study's op-coverage matrix (sec 4.1) shows most
transformer ops are ANE-feasible with caveats.

If achievable, GPU-prefill concurrent with ANE-decode gives the
largest realistic speedup in this study (1.30-2.00x on mixed
prefill+decode workloads), because it is the only configuration
without scheduler contention. This should be a primary research
direction, ahead of any single-backend kernel interleaving.

Caveat: ANE FP16 throughput for matmul is not cleanly published, and
the ANE study (sec 1.4, 3.3) documents many performance and
correctness constraints. Whether ANE decode is fast enough to keep up
with GPU prefill is an open empirical question.

#### 8.1.5 Could dequant move to ANE?

The ANE can do the trit-unpack + scale + outlier-addback as a
sequence of elementwise MIL ops (the ANE study sec 4.2 sketches this
for the T640 matmul decomposition). But ANE elementwise throughput
for a dequant that is memory-bound is unlikely to beat the GPU, and
the IOSurface round-trip adds latency. Moving dequant to ANE while
the GPU does the GEMM is probably not a win, because (a) the GPU
needs the dequantized weights immediately for the fused GEMM, and
(b) the ANE-GPU sync cost eats the benefit. Heterogeneous split is
better done at the phase level (GPU prefill, ANE decode) than at the
op level (GPU GEMM, ANE dequant).

### 8.2 NVIDIA CUDA

#### 8.2.1 CUDA streams for overlap

Well-established and the recommended mechanism for Candidate B.
Constraints:
- Async memcpy uses the copy engine, not SMs, so it overlaps compute
  for free - the cleanest overlap case.
- Two compute kernels overlap only if their combined occupancy < 100%
  of the SMs and they do not saturate HBM bandwidth.
- Hyper-Q (multiple hardware work queues) on H100 allows true
  concurrency of independent streams.

#### 8.2.2 H100 memory bandwidth and prefill headroom

H100 SXM5: 3.35 TB/s HBM3, 80 GB, 50 MB L2. H200: ~4.8 TB/s, 141 GB.
The prefill-headroom analysis is the same as Apple Silicon's but
scaled: a tuned FP8/FP16 GEMM on H100 is balanced and uses most of
the bandwidth. Prefetch headroom is modest. Cross-layer prefetch
(Candidate B) reaches one layer ahead (one layer of a 70B model at
3.5 bpw is ~30 MB, which fits in the 50 MB L2).

#### 8.2.3 Cooperative groups and persistent kernels

H100 supports cooperative groups grid sync via cudaLaunchCooperative
Kernel. The SM count (132) and max blocks per SM allow large grids.
CUTLASS Stream-K (and the in-tree mmq.cuh stream_k) are the proven
patterns. Candidate C is feasible on H100 with standard tools.

The practical constraint on H100 is that tensor cores want dense
GEMM shapes; ternary T640 does not map to tensor-core INT4/FP8 paths
directly. A T640 GEMM on H100 uses either DP4A-style INT ops or
emulated FP from the LUT dequant, neither of which uses tensor cores
efficiently. So the H100 T640 GEMM is less efficient than an FP8
GEMM, and the interleaving win is correspondingly different. This is
a Tessera-specific note: T640's ternary format is a poor fit for
NVIDIA tensor cores, so the CUDA backend's absolute throughput for
T640 is lower than for NVFP4/MXFP4, and the relative interleaving
win may be larger (more headroom) but the absolute throughput lower.

## 9. Concrete implementation plan for Tessera

### 9.1 Recommendation: pursue in this order

1. FIRST: finish Candidate A (drafter-in-decode) on Metal. It is
   already prototyped (ggml-metal-tile640-interleaved.metal), it is
   the lowest-risk win, and it directly improves decode throughput
   where Tessera is most bandwidth-limited. Target: 8-20% decode
   throughput gain with speculative decoding active.

2. SECOND: pursue the heterogeneous GPU+ANE split (sec 8.1.4), not
   further single-backend kernel interleaving. This is the largest
   realistic win (30-100%) and is the strategic bet. It depends on
   the ANE backend (D1) maturing, which is already in progress.

3. THIRD (CUDA only, lower priority for Tessera's Apple-first focus):
   Candidate B cross-layer prefetch on CUDA streams for server
   deployments. 5-15% on prefill-heavy or large-batch workloads.

4. DEFER Candidate C (persistent megakernel). It is the largest
   engineering investment with the most fragile result, and its win
   overlaps with (1) and (3). Revisit only if (1)+(2)+(3) are
   exhausted and there is still measurable headroom on CUDA.

Rationale: the analysis shows the spatial-partitioning vision is the
weakest mechanism everywhere, the temporal-intra-kernel vision is
already captured in tree (Candidate A), and the biggest unexploited
opportunity is heterogeneous compute (ANE+GPU), which is a different
axis entirely.

### 9.2 Smallest viable experiment to validate the hypothesis

To validate that intra-kernel interleaving yields free P1/P2 throughput
without P0 regression:

- Take the existing kernel_TILE640_MATMUL_INTERLEAVED.
- Benchmark batch=1 decode tokens/sec with drafter_enabled=0 (baseline)
  vs drafter_enabled=1 (interleaved), holding the drafter acceptance
  rate fixed (so the comparison is apples-to-apples on verified
  tokens/sec).
- Simultaneously measure P0 bit-identical output (the existing
  runtime-aware-pipeline Layer-1 fidelity test applies).
- Measure register usage (Metal's -BNDOPT or shader compilation
  report) to confirm no spill.

Acceptance: verified tokens/sec improves by >= 8% with P0 bit-
identical and no register spill. This is a 1-week experiment using
existing in-tree code and benchmarks. It directly tests the core
hypothesis on the cheapest mechanism.

### 9.3 Estimated effort to production

| Work item                                  | Effort  | Backend |
|--------------------------------------------|---------|---------|
| Candidate A drafter (P1) productionization | 3-5 wk  | Metal   |
| Candidate A KV quant (P2) productionization| 4-6 wk  | Metal   |
| Heterogeneous GPU+ANE split (research)     | 8-16 wk | Apple   |
| Candidate B cross-layer prefetch           | 3-5 wk  | CUDA    |
| Candidate B Metal command-buffer ring      | 5-8 wk  | Metal   |
| Candidate C persistent megakernel          | 8-16 wk | CUDA    |

### 9.4 Backend priority

Metal first. Tessera's primary deployment is Apple Silicon; the
in-tree prototype is Metal; and the largest win (heterogeneous ANE)
is Apple-only. CUDA work (Candidate B, C) is for server/NVIDIA
deployments and should follow the Metal work.

### 9.5 Interaction with the ANE backend work

The ANE backend (D1) is the single biggest multiplier for this design
area. Recommended interactions:

- Do NOT move dequant to ANE (sec 8.1.5). Keep dequant fused in the
  GPU GEMM.
- DO pursue running decode (the whole layer, including the fused
  T640 GEMM) on the ANE for batch=1, while the GPU handles prefill
  and large-batch decode. This requires the ANE T640 decomposition
  (ANE study sec 4.2) to be performance-competitive, which is the
  open empirical question.
- The MTLSharedEvent synchronization (ANE study sec 4.3.2) is the
  glue. The existing ane-mtp.mm serial-queue pattern must be
  generalized to allow concurrent GPU+ANE execution with event-based
  handoff at the phase boundary.
- Candidate A's drafter interleaving remains useful even in the
  heterogeneous world: during GPU prefill, the GPU has no decode to
  interleave, but if a decode is running on ANE, the GPU prefill
  does not need interleaving help (it is compute-bound). So
  Candidate A and heterogeneous split target different regimes
  (Candidate A: single-device decode; heterogeneous: multi-device
  prefill+decode).

## 10. Risk catalog

| # | Risk                                         | Likelihood | Impact | Mitigation |
|---|----------------------------------------------|------------|--------|------------|
| R1| Interleaving overhead exceeds idle-ALU savings | High    | Med    | Measure P0 regression before/after each interleave addition; gate on bit-identical + no register spill. |
| R2| Register spill silently regresses P0          | High    | High   | Track register count per kernel variant; abort interleave if spill detected. The base kernel is ~24 regs; budget hard cap at ~40. |
| R3| Metal scheduler serializes "concurrent" dispatch | High  | Med    | Validate overlap with GPU timestamp counters; do not assume concurrency. Prefer intra-kernel (Candidate A) over inter-kernel (B) on Metal. |
| R4| Occupancy numbers wrong (phases not complementary) | Med  | High   | Section 2 shows prefill is balanced (not memory-idle); the complementary-resource assumption is only weakly true. Set expectations accordingly; the win is in decode (ALU idle), not prefill (memory idle). |
| R5| T640 dequant too cheap to be worth overlapping | High   | Low    | Confirmed: dequant is fused into the GEMM already. Do not pursue standalone-dequant overlap. The overlap targets are drafter GEMM (P1) and KV ops (P2), not dequant. |
| R6| Synchronization complexity makes the kernel fragile | Med  | High   | Candidate A uses only existing barriers (low complexity). Candidate C (work queue) is the fragile one; defer. |
| R7| Outlier addback cache thrash from interleaved work | Low  | Med    | Keep P2 in threadgroup memory (already done in prototype). Do not let P1/P2 hit device memory in the addback window. |
| R8| Ternary T640 is a poor fit for NVIDIA tensor cores | High | Med    | CUDA T640 GEMM uses INT/FP emulation, not tensor cores. Absolute CUDA throughput for T640 is lower than NVFP4; consider whether NVFP4/MXFP4 paths should be preferred on H100. |
| R9| ANE decode not fast enough to keep up with GPU prefill | Med | High   | Open empirical question; validate early in D1. Fallback: ANE unused, GPU does both phases. |
| R10| Cross-layer prefetch thrashes L2 (Apple small cache) | Med | Med    | Limit prefetch to one layer ahead on Apple Silicon; validate L2 hit rate. |
| R11| Continuous batching at scheduler level already captures the win | Med | Med | True for server deployments (Sarathi-style). The kernel-level Candidate A still helps single-request Apple-Silicon decode where no scheduler-level batching occurs. |
| R12| Indirect command buffers / Metal 3 features regress on older targets | Low | Low | Gate behind deployment target (M1+). |

## Appendix A: Source citations

Primary Tessera sources (read for this study):
- ggml/src/ggml-metal/ggml-metal.metal lines 11285-11708 (T640 kernels:
  kernel_TILE640_MATMUL, _MATMUL_ID, _GET_ROWS, _DEQUANT, tile640_decode
  _element, T640_TRIT5_LUT).
- ggml/src/ggml-metal/ggml-metal-tile640-interleaved.metal (full file,
  326 lines: the Candidate A prototype).
- ggml/src/ggml-metal/ggml-metal-ops.cpp lines 1765-1828 (host dispatch,
  threadgroup sizing, simdgroups_per_tg logic).
- tools/quantize/tessera/tessera-format.h (T640 format spec, page=640,
  lane=20, lane_scale_bits=8).
- ggml/src/ggml-cuda/mmq.cuh, mmq.cu, dequantize.cuh, mmq-config-*
  (fused dequant+mulmat, stream-K, per-arch config tables).
- docs/interleaved-kernel-design.md (the narrower in-tree design).
- docs/ane-backend-deep-study.md (heterogeneous ANE+GPU, IOSurface,
  MTLSharedEvent, op coverage).
- docs/speculative.md (DFlash, DSpark drafters - the P1 workloads).
- docs/tessera-studio-design.md, docs/runtime-aware-pipeline.md,
  docs/pipeline-design.md (telemetry and fidelity context).

External references (web-verified 2026-07-31):
- Apple M3 Max / M4 Max specs: Apple Newsroom and Apple Support tech
  specs pages; M4 Max 546 GB/s, M3 Max 400 GB/s, FP16 ~14-17 TFLOPS.
  https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/
  https://support.apple.com/en-il/121553
- NVIDIA H100 SXM5: 80 GB HBM3, 3.35 TB/s, 132 SMs, FP16 tensor 989
  TFLOPS dense / 1979 with sparsity, FP8 1979/3958 TFLOPS.
  https://www.nvidia.com/en-us/data-center/h100/
  https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- Marlin: MARLIN: Mixed-Precision Auto-Regressive Parallel Inference,
  Frantar et al., arXiv 2408.11743 (the charter's 2403.09899 is not
  Marlin). https://arxiv.org/abs/2408.11743 ;
  https://github.com/IST-DASLab/marlin
- Flash Attention: Dao, arXiv 2205.14135 (2022) and 2307.08691 (2023).
  Flash Decoding: Dao blog 2023.
- Metal concurrency: WWDC22 "Load Resources Faster with Metal 3"
  (developer.apple.com/videos/play/wwdc2022/10104); Metal Programming
  Guide command execution model; MTLComputeCommandEncoder docs;
  Stack Overflow #57963757 on concurrent compute dispatch.
- Sarathi-Serve (chunked prefill, scheduler-level prefill+decode
  interleaving): Agrawal et al., arXiv 2308.16369.
- CUTLASS (NVIDIA), Triton (OpenAI): general mixed-precision GEMM and
  persistent-kernel references.

## Appendix B: Roofline numbers used in this study

All numbers are approximate, for ordering-of-magnitude reasoning.

| Device          | Peak (used)        | BW       | AI* (ridge)   |
|-----------------|--------------------|----------|---------------|
| Apple M3 Max 40c| ~14 TFLOPS FP16    | 400 GB/s | 35 FLOP/byte  |
| Apple M3 Max 40c| ~3.5 TFLOPS FP32   | 400 GB/s | 8.75          |
| Apple M4 Max 40c| ~17 TFLOPS FP16    | 546 GB/s | 31 FLOP/byte  |
| Apple M4 Max 40c| ~4.3 TFLOPS FP32   | 546 GB/s | 7.9           |
| NVIDIA H100 SXM5| 989 TFLOPS FP16 tensor (dense) | 3.35 TB/s | 295 |
| NVIDIA H100 SXM5| ~67 TFLOPS FP32    | 3.35 TB/s| 20            |
| NVIDIA A100 80GB| 312 TFLOPS FP16 tensor | 2.0 TB/s | 156       |
| NVIDIA A100 80GB| 19.5 TFLOPS FP32   | 2.0 TB/s | 9.75          |

T640 effective weight density: ~3.5 bits/weight (1.6 packed ternary +
0.025 page scale + 0.4 lane scale + ~1.5 outlier). Used in decode AI
calculation: AI_decode ~ 2 / (3.5/8) = 4.6 FLOP/byte.

End of document.
