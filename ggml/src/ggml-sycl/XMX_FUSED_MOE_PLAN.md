# XMX Fused MoE Kernel Redesign Plan

## Problem Statement

The current XMX MoE implementation launches **one kernel per expert** (16 per layer), resulting in:
- ~1,152 kernel launches per forward pass
- 42 t/s vs ESIMD's 675 t/s (16x slower)
- GPU pipeline stalls between each kernel
- Kernel launch overhead dominates compute time

## Root Cause Analysis

### Current XMX Architecture (Broken)
```
for each expert e with tokens:
    launch_xmx_moe_gemm_*(weights[e], tokens[e], output[e])  // Individual kernel
stream->wait()
```

Each kernel launch has:
- SYCL runtime overhead (~10-50μs)
- GPU command queue submission
- Work-group scheduling delay
- Memory allocation for intermediate buffers

### ESIMD Architecture (Reference - 675 t/s)
```
launch fused_moe_kernel(all_weights, all_tokens, expert_ids, output)  // Single kernel
```

Key features:
1. **Single kernel launch** for entire MoE layer
2. **Grid dispatch**: `(num_tokens * n_ids, nrows)`
3. **Expert indexing inside kernel** via `expert_ids[token, id]`
4. **SLM caching** for input (shared across all experts for a token)
5. **Persistent work-groups** (40 WGs loop over all work)

## Design: Fused XMX MoE Kernel

### Phase 1: Basic Fused Kernel (Priority: HIGH)

Transform XMX to match ESIMD's dispatch pattern:

```cpp
template<int TILES_M, int TILES_N>
void fused_xmx_moe_gemm_mxfp4_soa(
    // All expert weights in flat SoA format
    const uint8_t* all_expert_qs,    // [n_experts, nblocks, 16] packed nibbles
    const uint8_t* all_expert_e,     // [n_experts, nblocks] E8M0 exponents

    // Pre-quantized tokens (shared across all experts)
    const int8_t* q_tokens,          // [num_tokens, hidden_dim]
    const sycl::half* token_scales,  // [num_tokens, num_k_blocks]

    // Expert routing
    const int32_t* expert_ids,       // [num_tokens, n_ids] which expert per slot

    // Output
    sycl::half* output,              // [num_tokens, n_ids, out_dim]

    // Dimensions
    int num_tokens,
    int n_ids,                       // experts per token (e.g., 2)
    int out_dim,
    int hidden_dim,
    int n_experts,
    int64_t expert_stride,           // bytes between experts in weight tensor

    sycl::queue& queue
) {
    constexpr int XMX_M = 8, XMX_N = 16, XMX_K = 32;
    constexpr int TILE_M = TILES_M * XMX_M;  // 32
    constexpr int TILE_N = TILES_N * XMX_N;  // 64

    int num_k_blocks = hidden_dim / 32;

    // Grid: (num_tokens * n_ids, out_dim / TILE_N, ceil(1 / TILE_M))
    // Each work-group handles one (token, expert_slot, output_tile) combination
    int total_batches = num_tokens * n_ids;
    int n_tiles = (out_dim + TILE_N - 1) / TILE_N;

    auto event = queue.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(total_batches, n_tiles, 32),  // Global
                sycl::range<3>(1, 1, 32)                     // Local (sub-group)
            ),
            [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(32)]] {
                int batch_idx = item.get_group(0);
                int tile_idx = item.get_group(1);

                // Decompose batch into (token, expert_slot)
                int token_idx = batch_idx / n_ids;
                int id_idx = batch_idx % n_ids;

                // Read expert ID for this slot
                int expert_id = expert_ids[token_idx * n_ids + id_idx];
                if (expert_id < 0 || expert_id >= n_experts) return;

                // Calculate expert weight pointers (flat SoA)
                int nblocks_per_expert = out_dim * num_k_blocks;
                const uint8_t* expert_qs = all_expert_qs + expert_id * nblocks_per_expert * 16;
                const uint8_t* expert_e = all_expert_e + expert_id * nblocks_per_expert;

                // Output tile position
                int col_start = tile_idx * TILE_N;

                // Token input (single row, batch=1 for decode)
                const int8_t* token_row = q_tokens + token_idx * hidden_dim;
                const sycl::half* token_scale_row = token_scales + token_idx * num_k_blocks;

                // Output pointer
                sycl::half* out_row = output + (token_idx * n_ids + id_idx) * out_dim + col_start;

                // XMX GEMM for this tile (reuse existing tile logic)
                // ... joint_matrix operations ...
            }
        );
    });
    // NO .wait() here - caller handles synchronization
}
```

### Phase 2: SLM Input Caching (Priority: MEDIUM)

Cache token input in SLM to reduce global memory traffic:

```cpp
// Allocate SLM for token input caching
sycl::local_accessor<int8_t, 1> slm_tokens(hidden_dim, h);
sycl::local_accessor<sycl::half, 1> slm_scales(num_k_blocks, h);

// Collaborative load into SLM
for (int i = tid; i < hidden_dim; i += WG_SIZE) {
    slm_tokens[i] = token_row[i];
}
for (int i = tid; i < num_k_blocks; i += WG_SIZE) {
    slm_scales[i] = token_scale_row[i];
}
sycl::group_barrier(item.get_group());

// All sub-groups in WG now read from SLM instead of global memory
```

### Phase 3: Persistent Kernel (Priority: LOW)

Match ESIMD's persistent work-group pattern for maximum efficiency:

```cpp
constexpr int NUM_PERSISTENT_GROUPS = 80;  // 2 per XeCore, Arc B580 has 40 XeCores

// Single 1D dispatch
queue.parallel_for(
    sycl::nd_range<1>(NUM_PERSISTENT_GROUPS * WG_SIZE, WG_SIZE),
    [=](sycl::nd_item<1> item) {
        int group_id = item.get_group_linear_id();
        int64_t total_work = num_tokens * n_ids * n_output_tiles;

        // Persistent loop - each WG processes multiple work items
        for (int64_t work_idx = group_id; work_idx < total_work; work_idx += NUM_PERSISTENT_GROUPS) {
            int token_idx = work_idx / (n_ids * n_output_tiles);
            int id_idx = (work_idx / n_output_tiles) % n_ids;
            int tile_idx = work_idx % n_output_tiles;

            // Process this (token, expert_slot, tile) combination
            // ...
        }
    }
);
```

## Implementation Steps

### Step 1: Create Fused Kernel Template
- [ ] New file: `moe-xmx-fused.hpp`
- [ ] `fused_xmx_moe_gemm_mxfp4_soa()` with single kernel launch
- [ ] Reuse existing XMX tile logic from `moe-xmx.hpp`
- [ ] Expert ID lookup inside kernel

### Step 2: Update Dispatch Logic
- [ ] New function `try_fused_xmx_moe()` in `ggml-sycl.cpp`
- [ ] Grid dimensions: `(num_tokens * n_ids, n_tiles, 32)`
- [ ] Remove per-expert loop from host code
- [ ] Single `stream->wait()` at end

### Step 3: Token Preprocessing
- [ ] Pre-quantize all tokens to int8 before kernel launch
- [ ] Store scales per token-block
- [ ] Pass as unified buffer to fused kernel

### Step 4: Memory Layout Adaptation
- [ ] Support all three layouts: AoS, SoA, Coalesced
- [ ] Calculate per-expert offsets inside kernel
- [ ] Flat SoA: `expert_qs = base_qs + expert_id * blocks_per_expert * 16`

### Step 5: Validation
- [ ] Compare output vs ESIMD path
- [ ] Test with different batch sizes (1, 8, 32)
- [ ] Verify correctness on Q8_0 and MXFP4

### Step 6: Performance Tuning
- [ ] Add SLM caching for tokens
- [ ] Experiment with tile sizes (TILES_M=4,8 TILES_N=4,8)
- [ ] Consider persistent kernel for high throughput

## Expected Performance

| Configuration | Current | Target | Rationale |
|--------------|---------|--------|-----------|
| pp512 t/s | 42 | 400+ | Single kernel eliminates launch overhead |
| Kernel launches/layer | 16 | 1 | Fused architecture |
| GPU utilization | ~5% | ~80% | No pipeline stalls |

## Risks and Mitigations

1. **Register Pressure**: XMX joint_matrix uses many registers
   - Mitigation: Tune TILES_M/TILES_N to balance occupancy

2. **Work Imbalance**: Some experts may have more tokens
   - Mitigation: 2D grid naturally load-balances across tiles

3. **Memory Bandwidth**: Expert weights scattered in memory
   - Mitigation: SLM caching for tokens reduces bandwidth

4. **Compilation Time**: Large templated kernel
   - Mitigation: Precompile common configurations

## Dependencies

- Existing XMX tile GEMM logic in `moe-xmx.hpp`
- Token preprocessing from current implementation
- Flat SoA layout already working for MXFP4

## Timeline Estimate

- Phase 1 (Basic Fused): 4-6 hours
- Phase 2 (SLM Caching): 2-3 hours
- Phase 3 (Persistent): 4-6 hours
- Validation & Tuning: 4-8 hours

Total: 14-23 hours of focused development

## Success Criteria

1. Single kernel launch for entire MoE layer
2. pp512 performance >= 300 t/s (7x improvement over current)
3. Correct output matching ESIMD reference
4. No regressions on Q8_0 models
