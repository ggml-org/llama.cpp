# Unified XMX Fused MoE Kernel Design

## Overview

This design combines the best approaches from two previous plans:
- **Token sorting and scatter-back** from the sorted MoE design
- **Fused kernel architecture** with persistent work-groups from the fused MoE plan
- **SLM token caching** for reduced memory bandwidth

## Problem Statement

The current XMX MoE implementation launches **one kernel per expert** (16 per layer), resulting in:
- ~1,152 kernel launches per forward pass
- 42 t/s vs ESIMD's 675 t/s (16x slower)
- GPU pipeline stalls between each kernel
- Kernel launch overhead dominates compute time

## Target Use Cases

Optimized equally for:
- **Decode** (n_tokens=1): Single token generation
- **Prefill** (n_tokens=512+): Prompt processing

## Hardware Capabilities (Runtime Detection)

The kernel uses `XMXCapabilities` queried at device initialization to adapt to hardware:

```cpp
struct XMXCapabilities {
    bool supported = false;

    // XMX tile dimensions (queried from hardware)
    size_t M = 0;  // Expected: 8 (rows per matrix tile)
    size_t N = 0;  // Expected: 16 (columns per matrix tile)
    size_t K = 0;  // Expected: 32 (reduction dimension)

    // Supported data types
    bool supports_int8 = false;   // For Q8_0 quantization
    bool supports_fp16 = false;   // For MXFP4 scale computation

    // Device memory
    size_t slm_size = 0;          // Shared Local Memory per work-group (typically 64KB)

    // Derived optimal config
    int optimal_tiles_m = 1;      // Tiles in M dimension
    int optimal_tiles_n = 1;      // Tiles in N dimension
};
```

### Hardware-Specific Parameters

| Parameter | Detection Method | Arc B580 Value |
|-----------|-----------------|----------------|
| XeCore count | `nsm` (max_compute_units) | 40 |
| WGs per XeCore | Empirical | 2 |
| Total persistent WGs | `nsm * 2` | 80 |
| SLM size | `xmx_caps.slm_size` | 65536 (64KB) |
| XMX tile M | `xmx_caps.M` | 8 |
| XMX tile N | `xmx_caps.N` | 16 |
| XMX tile K | `xmx_caps.K` | 32 |
| Max work-group size | `max_work_group_sizes[id]` | 1024 |

### Hardware-Adaptive Configuration

```cpp
// Query from cached device info
const auto& dev_info = ggml_sycl_info().devices[device_id];
const auto& xmx = dev_info.xmx_caps;

// Calculate persistent work-group count based on actual hardware
const int NUM_PERSISTENT_WGS = dev_info.nsm * 2;  // 2 WGs per XeCore

// SLM budget allocation
const size_t SLM_BUDGET = xmx.slm_size;  // 64KB on Arc B580
const size_t slm_for_tokens = hidden_dim + num_k_blocks * sizeof(sycl::half);

// Work-group size selection
const int WG_SIZE = std::min(256, ggml_sycl_info().max_work_group_sizes[device_id]);
```

## Architecture Overview

All phases are **mandatory** and execute in sequence:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 1: Token Preprocessing                  │
│                         (always executes)                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Input: fp16 tokens [num_tokens, hidden_dim]                │ │
│  │  Output: int8 q_tokens + half scales                        │ │
│  │  Kernel: quantize_tokens_to_int8()                          │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 2: Token Sorting                        │
│                         (always executes)                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Sort tokens by expert_id for contiguous memory access      │ │
│  │  Record original positions for scatter-back                 │ │
│  │  Kernels: count → prefix_sum → sort_by_expert               │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Phase 3: Fused XMX MoE GEMM                     │
│                         (always executes)                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Persistent work-groups (nsm * 2 WGs)                       │ │
│  │  SLM caching for quantized token input                      │ │
│  │  XMX joint_matrix GEMM tiles                                │ │
│  │  Output: sorted results [num_tokens * n_ids, out_dim]       │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Phase 4: Scatter-back                          │
│                         (always executes)                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Restore results to original token order                    │ │
│  │  Uses token_mapping from Phase 2                            │ │
│  │  Kernel: moe_scatter_results()                              │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Phase 1: Token Preprocessing

Quantize fp16 tokens to int8 with per-block scales.

```cpp
// Dimensions match XMX K dimension (32 elements per block)
constexpr int QUANT_BLOCK_SIZE = 32;  // == xmx_caps.K

void quantize_tokens_to_int8(
    const sycl::half* tokens,      // [num_tokens, hidden_dim]
    int8_t* q_tokens,              // [num_tokens, hidden_dim]
    sycl::half* scales,            // [num_tokens, num_k_blocks]
    int num_tokens,
    int hidden_dim,
    sycl::queue& queue
) {
    int num_k_blocks = hidden_dim / QUANT_BLOCK_SIZE;

    queue.parallel_for(
        sycl::nd_range<2>(
            {num_tokens, num_k_blocks * QUANT_BLOCK_SIZE},
            {1, QUANT_BLOCK_SIZE}
        ),
        [=](sycl::nd_item<2> item) {
            int token = item.get_global_id(0);
            int block = item.get_group(1);
            int lane = item.get_local_id(1);

            // Find max absolute value in block
            float val = sycl::fabs(static_cast<float>(
                tokens[token * hidden_dim + block * 32 + lane]));
            float max_val = sycl::reduce_over_group(
                item.get_group(), val, sycl::maximum<float>());

            // Compute scale and quantize
            float scale = max_val / 127.0f;
            if (lane == 0) {
                scales[token * num_k_blocks + block] = sycl::half(scale);
            }

            float inv_scale = (scale > 0) ? 127.0f / max_val : 0.0f;
            int8_t q = static_cast<int8_t>(sycl::round(
                static_cast<float>(tokens[token * hidden_dim + block * 32 + lane])
                * inv_scale));

            q_tokens[token * hidden_dim + block * 32 + lane] = q;
        }
    );
}
```

## Phase 2: Token Sorting

Sort tokens by expert ID for contiguous memory access during GEMM.

### Step 2a: Count tokens per expert

```cpp
void moe_count_tokens_per_expert(
    const int32_t* expert_ids,    // [num_tokens, n_ids]
    int32_t* expert_counts,       // [n_experts]
    int num_tokens,
    int n_ids,
    int n_experts,
    sycl::queue& queue
) {
    // Zero counts
    queue.memset(expert_counts, 0, n_experts * sizeof(int32_t));

    queue.parallel_for(
        sycl::range<1>(num_tokens * n_ids),
        [=](sycl::id<1> idx) {
            int expert = expert_ids[idx];
            if (expert >= 0 && expert < n_experts) {
                sycl::atomic_ref<int32_t,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>
                    count_ref(expert_counts[expert]);
                count_ref.fetch_add(1);
            }
        }
    );
}
```

### Step 2b: Compute expert offsets (prefix sum)

```cpp
void moe_compute_expert_offsets(
    const int32_t* expert_counts,  // [n_experts]
    int32_t* expert_offsets,       // [n_experts + 1]
    int n_experts,
    sycl::queue& queue
) {
    // Simple CPU-side prefix sum (n_experts is small, typically 16)
    std::vector<int32_t> counts(n_experts);
    queue.copy(expert_counts, counts.data(), n_experts).wait();

    std::vector<int32_t> offsets(n_experts + 1);
    offsets[0] = 0;
    for (int i = 0; i < n_experts; i++) {
        offsets[i + 1] = offsets[i] + counts[i];
    }

    queue.copy(offsets.data(), expert_offsets, n_experts + 1);
}
```

### Step 2c: Sort tokens by expert

```cpp
void moe_sort_tokens_by_expert(
    const int32_t* expert_ids,     // [num_tokens, n_ids]
    int32_t* expert_offsets,       // [n_experts + 1] (modified atomically)
    int32_t* sorted_token_ids,     // [num_tokens * n_ids] output
    int32_t* token_mapping,        // [num_tokens * n_ids] original position
    int num_tokens,
    int n_ids,
    int n_experts,
    sycl::queue& queue
) {
    queue.parallel_for(
        sycl::range<1>(num_tokens * n_ids),
        [=](sycl::id<1> idx) {
            int token_idx = idx / n_ids;
            int id_idx = idx % n_ids;
            int expert = expert_ids[idx];

            if (expert >= 0 && expert < n_experts) {
                // Atomically get write position
                sycl::atomic_ref<int32_t,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>
                    offset_ref(expert_offsets[expert]);
                int write_pos = offset_ref.fetch_add(1);

                // Write sorted token reference
                sorted_token_ids[write_pos] = token_idx;
                token_mapping[write_pos] = idx;  // Original flat index
            }
        }
    );
}
```

## Phase 3: Fused XMX MoE GEMM

### Persistent Work-Group Pattern

```cpp
template<int TILES_M, int TILES_N>
void fused_xmx_moe_gemm(
    // Sorted token indices
    const int32_t* sorted_token_ids,  // [total_sorted_tokens]
    const int32_t* expert_offsets,    // [n_experts + 1]

    // Pre-quantized tokens
    const int8_t* q_tokens,           // [num_tokens, hidden_dim]
    const sycl::half* token_scales,   // [num_tokens, num_k_blocks]

    // Expert weights (Q8_0 or MXFP4 depending on model)
    const void* expert_weights,       // Layout depends on quantization
    const void* expert_scales,

    // Output
    sycl::half* output,               // [total_sorted_tokens, out_dim]

    // Dimensions
    int num_tokens,
    int n_experts,
    int out_dim,
    int hidden_dim,
    int64_t expert_stride,

    // Hardware caps
    const XMXCapabilities& xmx,
    int num_persistent_wgs,

    sycl::queue& queue
) {
    constexpr int XMX_M = 8, XMX_N = 16, XMX_K = 32;
    constexpr int TILE_M = TILES_M * XMX_M;  // 32
    constexpr int TILE_N = TILES_N * XMX_N;  // 64
    constexpr int WG_SIZE = 256;

    int n_output_tiles = (out_dim + TILE_N - 1) / TILE_N;
    int num_k_blocks = hidden_dim / XMX_K;

    // SLM sizes
    size_t slm_token_size = hidden_dim;
    size_t slm_scales_size = num_k_blocks * sizeof(sycl::half);

    queue.submit([&](sycl::handler& h) {
        // SLM allocations
        sycl::local_accessor<int8_t, 1> slm_token(hidden_dim, h);
        sycl::local_accessor<sycl::half, 1> slm_scales(num_k_blocks, h);

        h.parallel_for(
            sycl::nd_range<1>(num_persistent_wgs * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
                int group_id = item.get_group_linear_id();
                int tid = item.get_local_linear_id();
                auto sg = item.get_sub_group();

                // Process all experts
                for (int expert = 0; expert < n_experts; expert++) {
                    int expert_start = expert_offsets[expert];
                    int expert_end = expert_offsets[expert + 1];
                    int expert_token_count = expert_end - expert_start;

                    if (expert_token_count == 0) continue;

                    // Total work items for this expert
                    int64_t total_work = expert_token_count * n_output_tiles;

                    // Persistent loop over work items
                    for (int64_t work_idx = group_id;
                         work_idx < total_work;
                         work_idx += num_persistent_wgs) {

                        int tile_idx = work_idx % n_output_tiles;
                        int local_token_idx = work_idx / n_output_tiles;
                        int sorted_idx = expert_start + local_token_idx;
                        int token_idx = sorted_token_ids[sorted_idx];

                        // Load token into SLM (collaborative)
                        for (int i = tid; i < hidden_dim; i += WG_SIZE) {
                            slm_token[i] = q_tokens[token_idx * hidden_dim + i];
                        }
                        for (int i = tid; i < num_k_blocks; i += WG_SIZE) {
                            slm_scales[i] = token_scales[token_idx * num_k_blocks + i];
                        }
                        sycl::group_barrier(item.get_group());

                        // XMX GEMM for this tile
                        int col_start = tile_idx * TILE_N;

                        // Get expert weight pointer
                        const int8_t* w_ptr = static_cast<const int8_t*>(expert_weights)
                            + expert * expert_stride;

                        // Accumulator tiles
                        using namespace sycl::ext::oneapi::experimental::matrix;
                        joint_matrix<sycl::sub_group, int32_t, use::accumulator,
                                     XMX_M, XMX_N> acc[TILES_M][TILES_N];

                        // Initialize accumulators
                        for (int tm = 0; tm < TILES_M; tm++) {
                            for (int tn = 0; tn < TILES_N; tn++) {
                                joint_matrix_fill(sg, acc[tm][tn], 0);
                            }
                        }

                        // K-dimension reduction
                        for (int k = 0; k < hidden_dim; k += XMX_K) {
                            joint_matrix<sycl::sub_group, int8_t, use::a,
                                         XMX_M, XMX_K, layout::row_major> mat_a;
                            joint_matrix<sycl::sub_group, int8_t, use::b,
                                         XMX_K, XMX_N, layout::row_major> mat_b;

                            // Load from SLM
                            joint_matrix_load(sg, mat_a, &slm_token[k], XMX_K);

                            for (int tm = 0; tm < TILES_M; tm++) {
                                for (int tn = 0; tn < TILES_N; tn++) {
                                    int row = tm * XMX_M;
                                    int col = col_start + tn * XMX_N;

                                    // Load weight tile
                                    joint_matrix_load(sg, mat_b,
                                        w_ptr + (row * hidden_dim + k) * sizeof(int8_t),
                                        hidden_dim);

                                    // Multiply-accumulate
                                    joint_matrix_mad(sg, acc[tm][tn], mat_a, mat_b, acc[tm][tn]);
                                }
                            }
                        }

                        // Store results (with scale application)
                        sycl::half* out_ptr = output + sorted_idx * out_dim + col_start;
                        for (int tm = 0; tm < TILES_M; tm++) {
                            for (int tn = 0; tn < TILES_N; tn++) {
                                joint_matrix_store(sg, acc[tm][tn],
                                    out_ptr + tm * XMX_M * out_dim + tn * XMX_N,
                                    out_dim, layout::row_major);
                            }
                        }

                        sycl::group_barrier(item.get_group());
                    }
                }
            }
        );
    });
}
```

## Phase 4: Scatter-back

Restore results to original token order.

```cpp
void moe_scatter_results(
    const sycl::half* sorted_output,   // [num_tokens * n_ids, out_dim]
    const int32_t* token_mapping,      // [num_tokens * n_ids] original positions
    sycl::half* final_output,          // [num_tokens, n_ids, out_dim]
    int total_sorted,
    int out_dim,
    sycl::queue& queue
) {
    queue.parallel_for(
        sycl::nd_range<2>(
            {total_sorted, out_dim},
            {1, std::min(256, out_dim)}
        ),
        [=](sycl::nd_item<2> item) {
            int sorted_idx = item.get_global_id(0);
            int col = item.get_global_id(1);

            int original_idx = token_mapping[sorted_idx];
            final_output[original_idx * out_dim + col] =
                sorted_output[sorted_idx * out_dim + col];
        }
    );
}
```

## Quantization Format Support

### Q8_0 (Primary Target)

- Block size: 32 (matches XMX K=32 perfectly)
- Storage: int8_t values + fp16 scale per block
- Direct XMX int8 multiply-accumulate

### MXFP4 (Secondary Target)

- Block size: 32 (4-bit values packed as nibbles)
- E8M0 exponent format for scales
- Requires unpacking to int8 before XMX

## Memory Layout Support

The kernel supports all three weight layouts via `GGML_SYCL_REORDER_MODE`:

| Mode | Access Pattern | Performance |
|------|---------------|-------------|
| `coalesced` | Warp-coalesced 128-byte reads | Best (+35%) |
| `soa` | Structure-of-Arrays | Good (+32%) |
| `none` | Traditional AoS | Baseline |

## Expected Performance

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Kernel launches/layer | 16 | 1 | 16x reduction |
| pp512 t/s | 42 | 400+ | ~10x |
| GPU utilization | ~5% | ~80% | 16x |

## Implementation Files

New files to create:
- `ggml/src/ggml-sycl/moe-xmx-fused.hpp` - Fused kernel implementation

Files to modify:
- `ggml/src/ggml-sycl/ggml-sycl.cpp` - Dispatch logic
- `ggml/src/ggml-sycl/moe-sort.hpp` - Token sorting (integrate existing)

## Validation Criteria

1. **Correctness**: Output matches ESIMD reference path
2. **Determinism**: Fixed seed + temp=0 produces consistent output
3. **Performance**: pp512 >= 300 t/s (7x improvement over current)

### Test Command

```bash
GGML_SYCL_XMX_MOE=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -ngl 99 --flash-attn on \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0

# Expected: "1, 2, 3, 4, 5, 6, 7, 8, 9, 10..."
```
