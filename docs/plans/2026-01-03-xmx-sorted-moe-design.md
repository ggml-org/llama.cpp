# XMX Sorted MoE Kernel Design

> **Goal:** Create a unified ESIMD+XMX MoE kernel that beats oneDNN performance for both prefill (large batch) AND decode (small batch) on Intel Arc B580.

**Target Hardware:** Intel Arc B580 (BMG/Xe2) - 20 XeCores, 160 XMX units, 64KB SLM per XeCore

**Current Performance (GPT-OSS 20B Q8_0):**
| Metric | Master (oneDNN) | Current ESIMD | Target |
|--------|-----------------|---------------|--------|
| pp512  | 282 t/s         | 679 t/s*      | 800+ t/s |
| tg128  | 14 t/s          | 31 t/s        | 35+ t/s |

*After batch threshold fix - falls back to oneDNN batching for large batches.

---

## Architecture Overview

The design combines two key optimizations from different approaches:

1. **GPU-Side Token Sorting** (from oneDNN's batching approach)
2. **XMX Tiled GEMM** (using joint_matrix/dpas instructions)
3. **SLM Double-Buffering** (memory latency hiding)

```
Phase 1: GPU-Side Token Sorting
┌─────────────────────────────────────────────────────┐
│ Input: tokens[N] × experts[n_ids] routing decisions │
│                                                     │
│ For each expert e in 0..n_experts:                  │
│   1. Count tokens routed to expert e                │
│   2. Gather those tokens into contiguous buffer     │
│   3. Record original positions for scatter-back     │
└─────────────────────────────────────────────────────┘
                          ↓
Phase 2: XMX Tiled GEMM
┌─────────────────────────────────────────────────────┐
│ For each expert with batched_tokens > 0:            │
│   - Use joint_matrix with 8×16×32 tiles             │
│   - Process all tokens for this expert in one call  │
│   - Leverage XMX systolic arrays for throughput     │
│   - SLM double-buffering for latency hiding         │
└─────────────────────────────────────────────────────┘
                          ↓
Phase 3: Scatter Results
┌─────────────────────────────────────────────────────┐
│ Using recorded positions, scatter results back to   │
│ original token positions                            │
└─────────────────────────────────────────────────────┘
```

---

## Hardware-Queried XMX Configuration

Instead of hardcoding XMX dimensions, we query them from the hardware at runtime:

```cpp
#include <sycl/ext/oneapi/matrix/matrix.hpp>

struct XMXCapabilities {
    bool supported = false;

    // Queried from hardware
    size_t M = 0, N = 0, K = 0;  // Tile dimensions
    bool supports_int8 = false;
    bool supports_fp16 = false;
    bool supports_bf16 = false;

    // Derived optimal config
    int optimal_tiles_m = 1;
    int optimal_tiles_n = 1;
    size_t slm_per_xecore = 0;
};

XMXCapabilities query_xmx_capabilities(sycl::device& dev) {
    XMXCapabilities caps;

    if (!dev.has(sycl::aspect::ext_intel_matrix)) {
        return caps;  // XMX not available
    }
    caps.supported = true;

    // Query supported matrix combinations from hardware
    using namespace sycl::ext::oneapi::experimental;
    auto combinations = dev.get_info<info::device::matrix_combinations>();

    for (const auto& combo : combinations) {
        // Find int8 configuration (for Q8_0)
        if (combo.atype == matrix_type::sint8 &&
            combo.btype == matrix_type::sint8) {
            caps.supports_int8 = true;
            caps.M = combo.msize;  // Hardware-reported M
            caps.N = combo.nsize;  // Hardware-reported N
            caps.K = combo.ksize;  // Hardware-reported K

            GGML_SYCL_DEBUG("[XMX] Hardware reports: M=%zu, N=%zu, K=%zu (int8)\n",
                           caps.M, caps.N, caps.K);
        }

        if (combo.atype == matrix_type::fp16 &&
            combo.btype == matrix_type::fp16) {
            caps.supports_fp16 = true;
        }
    }

    // Query SLM size
    caps.slm_per_xecore = dev.get_info<sycl::info::device::local_mem_size>();

    // Compute optimal tile counts based on queried dimensions
    caps.optimal_tiles_m = std::min(4, (int)(32 / caps.M));
    caps.optimal_tiles_n = std::min(4, (int)(64 / caps.N));

    return caps;
}
```

**Expected values for Arc B580:**
- M = 8, N = 16, K = 32 (matches Q8_0 block size perfectly)
- SLM = 64KB per XeCore

---

## SLM Caching Strategy

**Arc B580 Memory Hierarchy:**
- 64KB SLM per XeCore (fast, ~10 cycles latency)
- 192KB L1 cache per XeCore
- ~12GB VRAM (slow, ~300+ cycles latency)

**SLM Budget Allocation (per work-group):**

```
┌─────────────────────────────────────────────────────────┐
│              64KB SLM Budget Allocation                 │
├─────────────────────────────────────────────────────────┤
│ Weight Tile A:     8KB  (double-buffer slot 1)          │
│ Weight Tile B:     8KB  (double-buffer slot 2)          │
│ Token Tile:        4KB  (batched token activations)     │
│ Accumulator:       4KB  (partial results)               │
│ Expert Routing:    1KB  (token→expert mapping)          │
│ Reserved:         39KB  (headroom for compiler)         │
└─────────────────────────────────────────────────────────┘
```

**Double-Buffering Pipeline:**

```cpp
slm_buffer<half, 8*1024> weight_buf[2];  // A/B buffers
slm_buffer<half, 4*1024> token_buf;

for (tile_k = 0; tile_k < K; tile_k += TILE_K) {
    int buf_idx = tile_k % 2;

    // Async prefetch NEXT tile while computing CURRENT
    if (tile_k + TILE_K < K) {
        async_copy(weight_buf[1-buf_idx], &weights[tile_k + TILE_K]);
    }

    // XMX GEMM on current tile from SLM
    joint_matrix_mad(acc, weight_buf[buf_idx], token_buf, acc);

    barrier();  // Ensure prefetch complete before swap
}
```

---

## Dynamic Model-Adaptive Configuration

**Layer 1: Hardware-Fixed (compile-time)**

```cpp
// Fixed by Intel XMX hardware - queried at runtime, validated
// Expected: M=8, N=16, K=32 for int8 on Arc B580
```

**Layer 2: Model-Adaptive (runtime)**

```cpp
struct MoEKernelConfig {
    // Derived from model at load time
    int64_t model_dim;        // e.g., 4096
    int64_t expert_dim;       // e.g., 14336
    int64_t n_experts;        // e.g., 32
    int64_t n_experts_active; // e.g., 8
    ggml_type weight_type;    // Q8_0, Q4_0, etc.

    // Computed optimal configuration
    int tiles_m;              // Tiles per WG (M dim)
    int tiles_n;              // Tiles per WG (N dim)
    int wg_size;              // Work-group size
    int slm_weight_kb;        // SLM for weight tiles
    int slm_token_kb;         // SLM for token buffer
    bool use_double_buffer;   // Memory latency hiding

    // Dispatch strategy thresholds
    int xmx_min_batch;        // Below this → ESIMD kernel
    int xmx_max_batch;        // Above this → oneDNN fallback (if needed)
};

MoEKernelConfig compute_moe_config(const ggml_tensor* weights) {
    MoEKernelConfig cfg;
    cfg.model_dim = weights->ne[0];
    cfg.expert_dim = weights->ne[1];
    cfg.n_experts = weights->ne[2];
    cfg.weight_type = weights->type;

    // Heuristics based on dimensions
    if (cfg.expert_dim >= 14336) {
        // Large FFN (GPT-OSS 20B style)
        cfg.tiles_m = 4;  cfg.tiles_n = 4;
        cfg.wg_size = 256;
        cfg.slm_weight_kb = 16;
    } else if (cfg.expert_dim >= 4096) {
        // Medium FFN (Mistral MoE style)
        cfg.tiles_m = 2;  cfg.tiles_n = 4;
        cfg.wg_size = 128;
        cfg.slm_weight_kb = 8;
    } else {
        // Small experts
        cfg.tiles_m = 1;  cfg.tiles_n = 2;
        cfg.wg_size = 64;
        cfg.slm_weight_kb = 4;
    }

    // Batch thresholds - tune per quantization type
    cfg.xmx_min_batch = (cfg.weight_type == GGML_TYPE_Q8_0) ? 4 : 8;
    cfg.xmx_max_batch = 2048;

    return cfg;
}
```

**Layer 3: Runtime Dispatch (per-call)**

```cpp
void ggml_sycl_mul_mat_id_moe(/* ... */, int64_t n_tokens) {
    static MoEKernelConfig cfg = compute_moe_config(weights);

    if (n_tokens < cfg.xmx_min_batch) {
        // Decode path: use current ESIMD kernel (optimized for n=1)
        launch_esimd_moe_kernel(/* ... */);
    } else if (n_tokens <= cfg.xmx_max_batch) {
        // Prefill path: use new XMX sorted kernel
        launch_xmx_sorted_moe_kernel(cfg, /* ... */);
    } else {
        // Huge batch: fall back to oneDNN (if ever needed)
        launch_onednn_batched_moe(/* ... */);
    }
}
```

---

## Phase 1: GPU-Side Token Sorting

```cpp
struct ExpertBatch {
    int32_t start_idx;      // Start in sorted token buffer
    int32_t count;          // Number of tokens for this expert
};

// Kernel 1a: Count tokens per expert (parallel histogram)
template<int N_EXPERTS>
void k_count_tokens_per_expert(
    const int32_t* __restrict__ expert_ids,  // [n_tokens, n_ids]
    int32_t* __restrict__ expert_counts,     // [N_EXPERTS]
    int64_t n_tokens,
    int64_t n_ids,
    sycl::nd_item<1> item)
{
    int tid = item.get_global_id(0);
    if (tid < n_tokens * n_ids) {
        int expert = expert_ids[tid];
        if (expert >= 0 && expert < N_EXPERTS) {
            sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                            sycl::memory_scope::device>(
                expert_counts[expert]).fetch_add(1);
        }
    }
}

// Kernel 1b: Exclusive prefix sum to get write offsets
// (Using SYCL group algorithms or manual scan)

// Kernel 1c: Scatter tokens to sorted positions
void k_sort_tokens_by_expert(
    const half* __restrict__ tokens_in,      // [n_tokens, model_dim]
    half* __restrict__ tokens_sorted,        // [n_tokens, model_dim]
    const int32_t* __restrict__ expert_ids,  // [n_tokens, n_ids]
    const int32_t* __restrict__ expert_offsets, // [N_EXPERTS] prefix sum
    int32_t* __restrict__ token_mapping,     // [n_tokens*n_ids] original position
    int64_t n_tokens, int64_t n_ids, int64_t model_dim,
    sycl::nd_item<2> item)
{
    int token_idx = item.get_global_id(0);
    int dim_idx = item.get_global_id(1);

    if (token_idx < n_tokens * n_ids && dim_idx < model_dim) {
        int expert = expert_ids[token_idx];

        // Atomically claim a slot in this expert's batch
        int write_pos = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device>(
            expert_offsets[expert]).fetch_add(1);

        // Copy token to sorted position
        tokens_sorted[write_pos * model_dim + dim_idx] =
            tokens_in[(token_idx / n_ids) * model_dim + dim_idx];

        // Record mapping for scatter-back
        if (dim_idx == 0) {
            token_mapping[write_pos] = token_idx;
        }
    }
}
```

---

## Phase 2: XMX Tiled GEMM

```cpp
template<typename T, int TILES_M, int TILES_N>
void k_xmx_moe_gemm(
    const int8_t* __restrict__ weights,     // [expert_dim, model_dim] Q8_0 qs
    const half* __restrict__ weight_scales, // [expert_dim, model_dim/32] Q8_0 d
    const half* __restrict__ tokens,        // [batch, model_dim] sorted tokens
    half* __restrict__ output,              // [batch, expert_dim]
    int64_t batch, int64_t expert_dim, int64_t model_dim,
    const XMXCapabilities& caps,
    sycl::nd_item<3> item)
{
    using namespace sycl::ext::oneapi::experimental::matrix;

    auto sg = item.get_sub_group();

    // Work-group processes TILES_M×TILES_N output tiles
    const int wg_row = item.get_group(1) * TILES_M * caps.M;
    const int wg_col = item.get_group(0) * TILES_N * caps.N;

    // SLM buffers for double-buffering
    auto slm = sycl::ext::oneapi::group_local_memory_for_overwrite<
        int8_t[2][TILES_M * caps.M * caps.K]>(item.get_group());
    auto slm_tokens = sycl::ext::oneapi::group_local_memory_for_overwrite<
        int8_t[TILES_N * caps.N * caps.K]>(item.get_group());

    // Accumulators (one per sub-group lane)
    joint_matrix<sub_group, int32_t, use::accumulator, caps.M, caps.N>
        acc[TILES_M][TILES_N];

    // Initialize accumulators to zero
    for (int tm = 0; tm < TILES_M; tm++)
        for (int tn = 0; tn < TILES_N; tn++)
            joint_matrix_fill(sg, acc[tm][tn], 0);

    // K-dimension loop with double-buffering
    int buf_idx = 0;
    for (int k = 0; k < model_dim; k += caps.K) {
        // Prefetch next weight tile to alternate buffer
        if (k + caps.K < model_dim) {
            async_work_group_copy(
                slm[1 - buf_idx],
                &weights[(wg_row) * model_dim + k + caps.K],
                TILES_M * caps.M * caps.K);
        }

        // Load current tile from SLM into joint_matrix
        joint_matrix<sub_group, int8_t, use::a, caps.M, caps.K> mat_a;
        joint_matrix<sub_group, int8_t, use::b, caps.K, caps.N> mat_b;

        for (int tm = 0; tm < TILES_M; tm++) {
            for (int tn = 0; tn < TILES_N; tn++) {
                joint_matrix_load(sg, mat_a,
                    &slm[buf_idx][tm * caps.M * caps.K], caps.K);
                joint_matrix_load(sg, mat_b,
                    &slm_tokens[tn * caps.N * caps.K], caps.K);

                // XMX multiply-accumulate (systolic array operation)
                joint_matrix_mad(sg, acc[tm][tn], mat_a, mat_b, acc[tm][tn]);
            }
        }

        sycl::group_barrier(item.get_group());
        buf_idx = 1 - buf_idx;
    }

    // Apply Q8_0 scales and store results
    for (int tm = 0; tm < TILES_M; tm++) {
        for (int tn = 0; tn < TILES_N; tn++) {
            // Convert int32 accumulator to fp16 with scale
            joint_matrix_store(sg, acc[tm][tn],
                &output[(wg_row + tm * caps.M) * expert_dim + wg_col + tn * caps.N],
                expert_dim, layout::row_major);
        }
    }
}
```

**Q8_0 Dequant Strategy:**

XMX processes raw int8, apply scale AFTER accumulation (~32x more efficient than dequant-before-multiply):

```cpp
// Q8_0 structure: 32 int8 values + 1 fp16 scale
struct block_q8_0 {
    sycl::half d;        // Scale factor
    int8_t qs[QK8_0];    // 32 quantized values
};

// acc_fp32 = acc_int32 * scale_a[k_blocks] * scale_b[k_blocks]
```

---

## Phase 3: Scatter Results

```cpp
void k_scatter_results(
    const half* __restrict__ sorted_output,  // [n_tokens*n_ids, expert_dim]
    half* __restrict__ final_output,         // [n_tokens, n_ids, expert_dim]
    const int32_t* __restrict__ token_mapping, // [n_tokens*n_ids]
    int64_t total_sorted, int64_t expert_dim,
    sycl::nd_item<2> item)
{
    int sorted_idx = item.get_global_id(0);
    int dim_idx = item.get_global_id(1);

    if (sorted_idx < total_sorted && dim_idx < expert_dim) {
        int original_pos = token_mapping[sorted_idx];
        final_output[original_pos * expert_dim + dim_idx] =
            sorted_output[sorted_idx * expert_dim + dim_idx];
    }
}
```

---

## Main Dispatcher

```cpp
void ggml_sycl_mul_mat_id_xmx_sorted(
    ggml_backend_sycl_context& ctx,
    const ggml_tensor* src0,   // Expert weights [n_experts, expert_dim, model_dim]
    const ggml_tensor* src1,   // Token activations [n_tokens, model_dim]
    const ggml_tensor* ids,    // Expert routing [n_tokens, n_ids]
    ggml_tensor* dst,          // Output [n_tokens, n_ids, expert_dim]
    sycl::queue* stream)
{
    const XMXCapabilities& caps = ctx.device_info.xmx_caps;

    int64_t n_tokens = src1->ne[1];
    int64_t n_ids = ids->ne[1];
    int64_t model_dim = src0->ne[0];
    int64_t expert_dim = src0->ne[1];
    int64_t n_experts = src0->ne[2];

    // Allocate temporary buffers (from pool)
    auto* tokens_sorted = ctx.pool.alloc<half>(n_tokens * n_ids * model_dim);
    auto* token_mapping = ctx.pool.alloc<int32_t>(n_tokens * n_ids);
    auto* expert_counts = ctx.pool.alloc<int32_t>(n_experts);
    auto* expert_offsets = ctx.pool.alloc<int32_t>(n_experts);
    auto* sorted_output = ctx.pool.alloc<half>(n_tokens * n_ids * expert_dim);

    // Phase 1: Sort tokens by expert
    stream->memset(expert_counts, 0, n_experts * sizeof(int32_t));
    stream->parallel_for(/*...*/,
        [=](auto item) { k_count_tokens_per_expert<32>(/*...*/); });

    compute_prefix_sum(expert_counts, expert_offsets, n_experts, stream);

    stream->parallel_for(/*...*/,
        [=](auto item) { k_sort_tokens_by_expert(/*...*/); });

    // Phase 2: XMX GEMM per expert (only for experts with tokens)
    for (int e = 0; e < n_experts; e++) {
        int32_t count = expert_counts[e];
        if (count == 0) continue;

        int32_t offset = expert_offsets[e];
        const int8_t* expert_weights = get_expert_weights(src0, e);

        auto grid = sycl::range<3>(
            ceil_div(expert_dim, caps.optimal_tiles_n * caps.N),
            ceil_div(count, caps.optimal_tiles_m * caps.M),
            1);

        stream->parallel_for(/*...*/,
            [=](auto item) {
                k_xmx_moe_gemm<half, 4, 4>(
                    expert_weights, /*...*/,
                    &tokens_sorted[offset * model_dim],
                    &sorted_output[offset * expert_dim],
                    count, expert_dim, model_dim, caps, item);
            });
    }

    // Phase 3: Scatter back to original positions
    stream->parallel_for(/*...*/,
        [=](auto item) { k_scatter_results(/*...*/); });
}
```

---

## Performance Expectations

| Phase | Kernel | Expected Cost |
|-------|--------|---------------|
| Sort | Count + Scan + Scatter | ~5% of total |
| XMX GEMM | Per-expert batched | ~90% of total |
| Scatter | Result redistribution | ~5% of total |

**Key Wins Over Current ESIMD Kernel:**
1. **Batched GEMM** instead of per-token processing
2. **XMX systolic arrays** instead of scalar ESIMD
3. **SLM double-buffering** hides memory latency
4. **Hardware-queried dimensions** ensure correctness

**XMX Theoretical Performance:**
- XMX peak: ~330 TOPS int8 on B580
- At 80% efficiency: ~264 TOPS
- For pp512 with 8 experts active: should exceed 500 t/s

---

## Implementation Plan

### Task 1: Add XMX Capability Query
- File: `ggml/src/ggml-sycl/common.cpp`
- Add `query_xmx_capabilities()` function
- Cache in device info structure

### Task 2: Implement Token Sorting Kernels
- File: `ggml/src/ggml-sycl/moe-sort.hpp` (new)
- Kernels: count, prefix sum, scatter

### Task 3: Implement XMX MoE GEMM Kernel
- File: `ggml/src/ggml-sycl/moe-xmx.hpp` (new)
- Tiled GEMM with SLM double-buffering

### Task 4: Implement Scatter-Back Kernel
- File: `ggml/src/ggml-sycl/moe-xmx.hpp`
- Result redistribution kernel

### Task 5: Integrate Dispatcher
- File: `ggml/src/ggml-sycl/ggml-sycl.cpp`
- Add dispatch logic to `ggml_sycl_mul_mat_id()`

### Task 6: Benchmark and Tune
- Test on GPT-OSS 20B Q8_0
- Compare pp512 and tg128 performance
- Tune tile sizes and thresholds

---

## Risk Assessment

- **Low risk**: Fallback to existing ESIMD kernel if XMX fails
- **Medium risk**: Joint matrix API is experimental (may change)
- **Mitigation**: Hardware query validates assumptions at runtime

## References

- [Intel SYCL joint_matrix specification](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc)
- [Programming Intel XMX Using SYCL](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/joint-matrix.html)
- Existing code: `ggml/src/ggml-sycl/fattn-xmx-f16.hpp`, `mmq_xmx.cpp`
