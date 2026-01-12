# XMX MoE Graph-Compatible Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate host synchronization from XMX MoE kernels to enable SYCL command graph compatibility

**Architecture:** Replace host-side expert iteration loop with GPU-side work-group self-assignment. Each work-group uses binary search on `expert_tile_offsets` to determine which expert to process, enabling single-kernel dispatch without host intervention.

**Tech Stack:** SYCL, Intel XMX (joint_matrix), oneAPI Level Zero

---

## Problem Summary

Current XMX MoE achieves ~24 t/s pp512 vs ESIMD+graphs ~671 t/s. The bottleneck is host synchronization:
- Lines 11578-11580: `moe_count_tokens_per_expert` and `moe_compute_expert_offsets` have internal `.wait()` calls
- Line 11596: `stream->memcpy(expert_write_pos, ...).wait()`
- Lines 11633-11635: `stream->memcpy(h_counts, h_offsets, ...).wait()` to copy data to HOST vectors
- Lines 11793-11894: HOST for-loop iterating over experts, launching separate kernels

These patterns are incompatible with SYCL graph recording which requires continuous GPU execution without host intervention.

---

## Task 1: Add GPU Tile Mapping Function to moe-sort.hpp

**Files:**
- Modify: `ggml/src/ggml-sycl/moe-sort.hpp:450` (append before `#endif`)

**Step 1: Write the function signature and documentation**

Add at line 450 (before `#endif`):

```cpp
// Compute tile mapping for fused XMX kernel
// Converts expert counts to tile counts and computes cumulative tile offsets
// This enables work-groups to self-assign to experts via binary search
//
// Output:
//   expert_tile_offsets[i] = cumulative tiles for experts 0..i-1
//   expert_tile_offsets[n_experts] = total_tiles (for grid launch size)
//
// Formula: tiles_for_expert[e] = ceil(expert_counts[e] / tile_M)
inline sycl::event moe_compute_tile_mapping(
    const int32_t * expert_counts,       // [n_experts] token counts per expert
    int32_t *       expert_tile_offsets, // [n_experts + 1] output tile offsets
    int32_t *       total_tiles_out,     // [1] output: total tiles for grid launch
    int64_t         n_experts,
    int64_t         tile_M,              // XMX tile size in M dimension (typically 32)
    sycl::queue &   queue,
    sycl::event     dep_event = {}) {

    return queue.submit([&](sycl::handler & cgh) {
        if (dep_event.get_info<sycl::info::event::command_execution_status>() !=
            sycl::info::event_command_status::complete) {
            cgh.depends_on(dep_event);
        }

        // Single work-item computes prefix sum of tile counts
        cgh.single_task([=]() {
            int32_t cumulative = 0;
            expert_tile_offsets[0] = 0;

            for (int64_t e = 0; e < n_experts; e++) {
                int32_t count = expert_counts[e];
                int32_t tiles = (count + tile_M - 1) / tile_M;  // ceil division
                cumulative += tiles;
                expert_tile_offsets[e + 1] = cumulative;
            }

            *total_tiles_out = cumulative;
        });
    });
}
```

**Step 2: Verify file compiles**

Run:
```bash
source /opt/intel/oneapi/setvars.sh --force
./scripts/quick-rebuild.sh ggml-sycl.cpp
```

Expected: Build succeeds (no syntax errors)

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/moe-sort.hpp
git commit -m "feat(sycl): add GPU tile mapping function for fused XMX MoE"
```

---

## Task 2: Add Binary Search Device Helper to moe-xmx.hpp

**Files:**
- Modify: `ggml/src/ggml-sycl/moe-xmx.hpp:18` (inside namespace moe_xmx, after `using namespace` line)

**Step 1: Add the device function**

Insert after line 20 (`using namespace sycl::ext::oneapi::experimental::matrix;`):

```cpp
// Device function: binary search to find expert from work-group ID
// Given wg_id and expert_tile_offsets[], returns the expert index
// such that expert_tile_offsets[expert] <= wg_id < expert_tile_offsets[expert + 1]
inline int find_expert_for_workgroup(
    int             wg_id,
    const int32_t * expert_tile_offsets,
    int             n_experts) {
    int lo = 0, hi = n_experts;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (expert_tile_offsets[mid + 1] <= wg_id) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Compute local tile index within expert's tile range
inline int get_local_tile_index(
    int             wg_id,
    int             expert_idx,
    const int32_t * expert_tile_offsets) {
    return wg_id - expert_tile_offsets[expert_idx];
}
```

**Step 2: Verify file compiles**

Run:
```bash
./scripts/quick-rebuild.sh ggml-sycl.cpp
```

Expected: Build succeeds

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/moe-xmx.hpp
git commit -m "feat(sycl): add binary search device helper for XMX MoE work assignment"
```

---

## Task 3: Add Fused XMX Kernel with GPU Work Assignment

**Files:**
- Modify: `ggml/src/ggml-sycl/moe-xmx.hpp` (append new function after `launch_xmx_moe_gemm_q8_0_soa`)

**Step 1: Add fused kernel template**

This is a large function. Add after the closing brace of `launch_xmx_moe_gemm_q8_0_soa` (around line 690):

```cpp
// Fused XMX MoE kernel - GPU-side expert assignment for graph compatibility
// Each work-group determines its expert via binary search, eliminating host iteration
//
// Key differences from per-expert kernel:
// 1. Launches total_tiles work-groups (covers all experts)
// 2. Each WG binary-searches to find its expert
// 3. No host synchronization needed
//
// Parameters:
//   expert_offsets: [n_experts + 1] token offsets (cumulative sum of counts)
//   expert_tile_offsets: [n_experts + 1] tile offsets (cumulative sum of ceil(count/tile_M))
//   total_tiles: total work-groups to launch
template <int TILES_M = 4, int TILES_N = 4>
sycl::event launch_fused_xmx_moe_q8_0_soa(
    sycl::queue &          queue,
    sycl::event            dep_event,
    const int8_t *         all_expert_qs,      // [n_experts * out_dim * in_dim] SoA qs
    const sycl::half *     all_expert_d,       // [n_experts * out_dim * in_dim/32] SoA scales
    const int8_t *         q_tokens,           // [total_pairs, in_dim] pre-quantized
    const sycl::half *     token_scales,       // [total_pairs, in_dim/32]
    sycl::half *           sorted_output,      // [total_pairs, out_dim]
    const int32_t *        expert_offsets,     // [n_experts + 1] token offsets
    const int32_t *        expert_tile_offsets,// [n_experts + 1] tile offsets
    int32_t                total_tiles,
    int64_t                n_experts,
    int64_t                out_dim,
    int64_t                in_dim,
    int64_t                qs_stride_per_expert, // bytes of qs data per expert
    const MoEXMXConfig &   cfg) {

    constexpr int XMX_M   = 8;
    constexpr int XMX_N   = 16;
    constexpr int XMX_K   = 32;
    constexpr int SG_SIZE = 16;

    int wg_out_rows = TILES_M * XMX_M;  // 32 output rows per WG
    int wg_out_cols = TILES_N * XMX_N;  // 64 output cols per WG

    // Calculate n_col_wgs (number of work-groups per tile row for N dimension)
    int n_col_wgs = (out_dim + wg_out_cols - 1) / wg_out_cols;

    // Launch grid: total_tiles * n_col_wgs work-groups
    // Each tile processes wg_out_rows tokens, each col_wg covers wg_out_cols output dims
    sycl::range<2> global{ static_cast<size_t>(total_tiles * cfg.wg_size),
                           static_cast<size_t>(n_col_wgs) };
    sycl::range<2> local{ static_cast<size_t>(cfg.wg_size), 1 };

    const int     num_sgs       = cfg.wg_size / SG_SIZE;
    const int64_t num_k_blocks  = in_dim / XMX_K;
    const int64_t nblocks_per_expert = out_dim * num_k_blocks;

    return queue.submit([&](sycl::handler & cgh) {
        if (dep_event.get_info<sycl::info::event::command_execution_status>() !=
            sycl::info::event_command_status::complete) {
            cgh.depends_on(dep_event);
        }

        // SLM allocations (same as per-expert kernel)
        constexpr int slm_weights_size = TILES_N * XMX_N * XMX_K;
        sycl::local_accessor<int8_t, 1> slm_weights(sycl::range<1>(slm_weights_size), cgh);

        constexpr int slm_acc_per_sg = XMX_M * XMX_N * sizeof(int32_t);
        const int     slm_acc_size   = num_sgs * slm_acc_per_sg;
        sycl::local_accessor<int8_t, 1> slm_acc(sycl::range<1>(slm_acc_size), cgh);

        sycl::local_accessor<int8_t, 1> slm_tokens(sycl::range<1>(TILES_M * XMX_M * XMX_K), cgh);
        sycl::local_accessor<float, 1>  slm_token_scales(sycl::range<1>(TILES_M * XMX_M), cgh);
        sycl::local_accessor<float, 1>  slm_weight_scales(sycl::range<1>(TILES_N * XMX_N), cgh);

        cgh.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                auto sg    = item.get_sub_group();
                int  sg_id = sg.get_group_linear_id();
                int  lane  = sg.get_local_linear_id();

                // GPU-side work assignment via binary search
                int tile_idx = item.get_group(0);  // Which tile (0 to total_tiles-1)
                int col_wg   = item.get_group(1);  // Which column work-group

                // Find expert for this tile via binary search
                int expert_idx = find_expert_for_workgroup(tile_idx, expert_tile_offsets, n_experts);

                // Get expert's token range
                int expert_token_start = expert_offsets[expert_idx];
                int expert_token_count = expert_offsets[expert_idx + 1] - expert_token_start;

                // Get local tile index within this expert
                int local_tile = get_local_tile_index(tile_idx, expert_idx, expert_tile_offsets);

                // Calculate which rows this WG handles within the expert's batch
                int wg_row = local_tile * wg_out_rows;  // Row offset within expert batch
                int wg_col = col_wg * wg_out_cols;      // Column offset in output

                // Skip if no work for this WG
                if (wg_row >= expert_token_count) {
                    return;
                }

                // Calculate global row in sorted token array
                int global_row_start = expert_token_start + wg_row;

                // Get expert weight pointers (SoA layout)
                const int8_t *     expert_qs = all_expert_qs + expert_idx * qs_stride_per_expert;
                const sycl::half * expert_d  = all_expert_d + expert_idx * nblocks_per_expert;

                // Initialize accumulators
                joint_matrix<sycl::sub_group, int32_t, use::accumulator, XMX_M, XMX_N> acc[TILES_M][TILES_N];
                for (int tm = 0; tm < TILES_M; tm++) {
                    for (int tn = 0; tn < TILES_N; tn++) {
                        joint_matrix_fill(sg, acc[tm][tn], 0);
                    }
                }

                float float_acc[TILES_M][TILES_N][XMX_M * XMX_N] = { { { 0.0f } } };

                // K-dimension reduction loop (same logic as per-expert kernel)
                for (int64_t k_block = 0; k_block < num_k_blocks; k_block++) {
                    int64_t k = k_block * XMX_K;

                    // Cooperative token loading to SLM
                    constexpr int slm_tokens_size = TILES_M * XMX_M * XMX_K;
                    int items_per_sg = slm_tokens_size / num_sgs;
                    int sg_offset    = sg_id * items_per_sg;

                    for (int i = 0; i < items_per_sg; i += SG_SIZE) {
                        int idx = sg_offset + i + lane;
                        if (idx < slm_tokens_size) {
                            int tile_row   = idx / XMX_K;
                            int tile_k     = idx % XMX_K;
                            int local_row  = wg_row + tile_row;

                            if (local_row < expert_token_count) {
                                int global_row = global_row_start + tile_row;
                                slm_tokens[idx] = q_tokens[global_row * in_dim + k + tile_k];
                            } else {
                                slm_tokens[idx] = 0;
                            }
                        }
                    }

                    // Load token scales for this K-block
                    if (sg_id < (TILES_M * XMX_M + SG_SIZE - 1) / SG_SIZE) {
                        int row_idx = sg_id * SG_SIZE + lane;
                        if (row_idx < TILES_M * XMX_M) {
                            int local_row = wg_row + row_idx;
                            if (local_row < expert_token_count) {
                                int global_row = global_row_start + row_idx;
                                slm_token_scales[row_idx] =
                                    static_cast<float>(token_scales[global_row * num_k_blocks + k_block]);
                            } else {
                                slm_token_scales[row_idx] = 0.0f;
                            }
                        }
                    }

                    // Load weights from SoA format
                    int weights_per_sg = slm_weights_size / num_sgs;
                    int w_sg_offset    = sg_id * weights_per_sg;

                    for (int i = 0; i < weights_per_sg; i += SG_SIZE) {
                        int idx = w_sg_offset + i + lane;
                        if (idx < slm_weights_size) {
                            int out_col_local = idx / XMX_K;
                            int k_elem        = idx % XMX_K;
                            int global_col    = wg_col + out_col_local;

                            int8_t val = 0;
                            if (global_col < out_dim) {
                                // SoA block indexing: block_idx = out_col * num_k_blocks + k_block
                                int64_t block_idx = global_col * num_k_blocks + k_block;
                                val = expert_qs[block_idx * XMX_K + k_elem];
                            }
                            slm_weights[k_elem + out_col_local * XMX_K] = val;
                        }
                    }

                    // Load weight scales
                    if (sg_id < (TILES_N * XMX_N + SG_SIZE - 1) / SG_SIZE) {
                        int col_idx = sg_id * SG_SIZE + lane;
                        if (col_idx < TILES_N * XMX_N) {
                            int global_col = wg_col + col_idx;
                            if (global_col < out_dim) {
                                int64_t block_idx = global_col * num_k_blocks + k_block;
                                slm_weight_scales[col_idx] = static_cast<float>(expert_d[block_idx]);
                            } else {
                                slm_weight_scales[col_idx] = 0.0f;
                            }
                        }
                    }

                    item.barrier(sycl::access::fence_space::local_space);

                    // XMX computation (only sg_id == 0)
                    if (sg_id == 0) {
                        joint_matrix<sycl::sub_group, int8_t, use::a, XMX_M, XMX_K, layout::row_major> mat_a;
                        joint_matrix<sycl::sub_group, int8_t, use::b, XMX_K, XMX_N, layout::col_major> mat_b;

                        for (int tm = 0; tm < TILES_M; tm++) {
                            auto slm_tokens_ptr = sycl::address_space_cast<
                                sycl::access::address_space::local_space, sycl::access::decorated::no>(
                                &slm_tokens[tm * XMX_M * XMX_K]);
                            joint_matrix_load(sg, mat_a, slm_tokens_ptr, XMX_K);

                            for (int tn = 0; tn < TILES_N; tn++) {
                                int local_row = wg_row + tm * XMX_M;
                                int col       = wg_col + tn * XMX_N;

                                if (local_row < expert_token_count && col < out_dim) {
                                    auto slm_weights_ptr = sycl::address_space_cast<
                                        sycl::access::address_space::local_space, sycl::access::decorated::no>(
                                        &slm_weights[tn * XMX_N * XMX_K]);
                                    joint_matrix_load(sg, mat_b, slm_weights_ptr, XMX_K);

                                    joint_matrix_mad(sg, acc[tm][tn], mat_a, mat_b, acc[tm][tn]);

                                    int32_t * acc_slm_raw = reinterpret_cast<int32_t *>(&slm_acc[0]);
                                    auto acc_slm_ptr = sycl::address_space_cast<
                                        sycl::access::address_space::local_space, sycl::access::decorated::no>(
                                        acc_slm_raw);
                                    joint_matrix_store(sg, acc[tm][tn], acc_slm_ptr, XMX_N, layout::row_major);

                                    sycl::group_barrier(sg);

                                    for (int i = lane; i < XMX_M * XMX_N; i += SG_SIZE) {
                                        int tile_row = i / XMX_N;
                                        int tile_col = i % XMX_N;
                                        if (local_row + tile_row < expert_token_count && col + tile_col < out_dim) {
                                            float t_scale = slm_token_scales[tm * XMX_M + tile_row];
                                            float w_scale = slm_weight_scales[tn * XMX_N + tile_col];
                                            float_acc[tm][tn][i] += acc_slm_raw[i] * t_scale * w_scale;
                                        }
                                    }

                                    joint_matrix_fill(sg, acc[tm][tn], 0);
                                }
                            }
                        }
                    }

                    item.barrier(sycl::access::fence_space::local_space);
                }

                // Final output store (only sg_id == 0)
                if (sg_id == 0) {
                    for (int tm = 0; tm < TILES_M; tm++) {
                        for (int tn = 0; tn < TILES_N; tn++) {
                            int local_row = wg_row + tm * XMX_M;
                            int col       = wg_col + tn * XMX_N;

                            if (local_row < expert_token_count && col < out_dim) {
                                for (int i = lane; i < XMX_M * XMX_N; i += SG_SIZE) {
                                    int tile_row = i / XMX_N;
                                    int tile_col = i % XMX_N;
                                    if (local_row + tile_row < expert_token_count && col + tile_col < out_dim) {
                                        int global_row = global_row_start + tm * XMX_M + tile_row;
                                        sorted_output[global_row * out_dim + col + tile_col] =
                                            sycl::half(float_acc[tm][tn][i]);
                                    }
                                }
                            }
                            (void) acc[tm][tn];
                        }
                    }
                }
            });
    });
}
```

**Step 2: Verify file compiles**

Run:
```bash
./scripts/quick-rebuild.sh ggml-sycl.cpp
```

Expected: Build succeeds

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/moe-xmx.hpp
git commit -m "feat(sycl): add fused XMX MoE kernel with GPU work assignment"
```

---

## Task 4: Add Pre-allocated Buffers to Device Context

**Files:**
- Modify: `ggml/src/ggml-sycl/common.hpp` (find `xmx_moe_buffers` struct, extend it)

**Step 1: Find and extend xmx_moe_buffers**

Locate the existing `xmx_moe_buffers` struct in common.hpp. Add new members:

```cpp
// XMX MoE buffers for graph-compatible execution
struct xmx_moe_buffers {
    // Existing members...
    int32_t * expert_tile_offsets = nullptr;  // [max_experts + 1]
    int32_t * total_tiles         = nullptr;  // [1] scalar on device

    // Pre-allocate for up to 64 experts
    static constexpr int MAX_EXPERTS = 64;

    void allocate_tile_mapping(sycl::queue & q) {
        if (!expert_tile_offsets) {
            expert_tile_offsets = sycl::malloc_device<int32_t>(MAX_EXPERTS + 1, q);
            total_tiles = sycl::malloc_device<int32_t>(1, q);
        }
    }

    void free_tile_mapping(sycl::queue & q) {
        if (expert_tile_offsets) {
            sycl::free(expert_tile_offsets, q);
            expert_tile_offsets = nullptr;
        }
        if (total_tiles) {
            sycl::free(total_tiles, q);
            total_tiles = nullptr;
        }
    }
};
```

**Step 2: Verify file compiles**

Run:
```bash
./scripts/quick-rebuild.sh ggml-sycl.cpp
```

Expected: Build succeeds

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/common.hpp
git commit -m "feat(sycl): add tile mapping buffers to xmx_moe_buffers"
```

---

## Task 5: Replace Host Iteration with Fused Dispatch in ggml-sycl.cpp

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp:11498-11894` (refactor `try_xmx_sorted_moe`)

**Step 1: Remove early bailout for graphs**

Delete lines 11498-11504 (the graph check that prevents XMX):

```cpp
// DELETE THIS BLOCK:
// XMX MoE uses synchronous memcpy for expert counts - incompatible with SYCL graphs
// When graphs are enabled, consistently use ESIMD for both warmup and recording
// to ensure oneDNN primitive cache is warmed correctly
if (!g_ggml_sycl_disable_graph) {
    GGML_SYCL_DEBUG("[XMX MoE] Graphs enabled, falling back to ESIMD for consistency\n");
    return false;
}
```

**Step 2: Replace synchronous sort/count functions with async versions**

Replace lines 11578-11596 with async event-chained versions:

```cpp
// Phase 1: Sort tokens by expert (async, graph-compatible)
const int32_t * expert_ids = static_cast<const int32_t *>(ids->data);

// Async token counting
sycl::event count_event = moe_count_tokens_per_expert_async<64>(
    expert_ids, expert_counts, n_tokens, n_ids, *stream);

// Async GPU-side prefix sum for expert offsets
sycl::event offset_event = moe_compute_expert_offsets_gpu_simple(
    expert_counts, expert_offsets, n_experts, *stream, count_event);

// Compute tile mapping for fused kernel dispatch
constexpr int TILE_M = 32;  // TILES_M * XMX_M = 4 * 8 = 32
int32_t * expert_tile_offsets = sycl::malloc_device<int32_t>(n_experts + 1, *stream);
int32_t * total_tiles_dev     = sycl::malloc_device<int32_t>(1, *stream);

sycl::event tile_map_event = moe_compute_tile_mapping(
    expert_counts, expert_tile_offsets, total_tiles_dev,
    n_experts, TILE_M, *stream, offset_event);

// Copy offsets for atomic writes during sorting (async)
int32_t * expert_write_pos = sycl::malloc_device<int32_t>(n_experts, *stream);
sycl::event copy_event = stream->memcpy(expert_write_pos, expert_offsets,
                                         n_experts * sizeof(int32_t), {offset_event});
```

**Step 3: Replace host for-loop with fused kernel dispatch**

Replace lines 11631-11894 (the host iteration logic) with:

```cpp
// Phase 2: Fused XMX GEMM (single kernel for all experts)
auto xmx_cfg = moe_xmx::MoEXMXConfig::from_capabilities(caps);

// Read total_tiles from device (single sync point - required for grid launch size)
int32_t total_tiles_host = 0;
stream->memcpy(&total_tiles_host, total_tiles_dev, sizeof(int32_t), {tile_map_event}).wait();

if (total_tiles_host == 0) {
    GGML_SYCL_DEBUG("[XMX MoE] No tokens assigned to any expert, skipping\n");
    // Free buffers and return
    goto cleanup;
}

GGML_SYCL_DEBUG("[XMX MoE] Fused kernel: total_tiles=%d\n", total_tiles_host);

// Pre-quantize all sorted tokens (async)
sycl::event quant_event = /* async quantization - chain from sort_event */;

if (src0->type == GGML_TYPE_Q8_0 && is_soa) {
    // Launch fused kernel with GPU-side expert assignment
    const int8_t *     base_qs = static_cast<const int8_t *>(src0->data);
    const sycl::half * base_d  = reinterpret_cast<const sycl::half *>(
        static_cast<const char *>(src0->data) + soa_total_qs_q8);

    sycl::event gemm_event = moe_xmx::launch_fused_xmx_moe_q8_0_soa<4, 4>(
        *stream, quant_event,
        base_qs, base_d,
        q_tokens, token_scales,
        sorted_output,
        expert_offsets, expert_tile_offsets,
        total_tiles_host,
        n_experts, out_dim, in_dim,
        q8_qs_per_expert,
        xmx_cfg);

    // Chain scatter to GEMM completion
    scatter_event = moe_scatter_results_f16_to_f32_async(
        sorted_output, final_output, token_map,
        /* actual_pairs read from expert_offsets[n_experts] */ actual_pairs,
        out_dim, n_ids, out_nb1, out_nb2, *stream, gemm_event);
}
```

**Step 4: Verify build succeeds**

Run:
```bash
./scripts/quick-rebuild.sh ggml-sycl.cpp
```

Expected: Build succeeds

**Step 5: Commit**

```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "refactor(sycl): replace host iteration with fused XMX MoE dispatch"
```

---

## Task 6: Test with Graphs Enabled

**Step 1: Run basic correctness test**

```bash
source /opt/intel/oneapi/setvars.sh --force
GGML_SYCL_XMX_MOE=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -ngl 99 --flash-attn on \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0
```

Expected output: `1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20`

**Step 2: Run with debug to verify graph recording**

```bash
GGML_SYCL_DEBUG=1 GGML_SYCL_XMX_MOE=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -ngl 99 --flash-attn on \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0 2>&1 | grep -E "(graph|XMX)"
```

Expected: Should see "graph recording" messages, NOT "falling back to ESIMD"

**Step 3: Commit test results**

Document test results in a follow-up commit message if all tests pass.

---

## Task 7: Benchmark Performance

**Step 1: Run ESIMD baseline (graphs enabled)**

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -p 512 -n 128 -ngl 99 -fa 1
```

Record: ESIMD+graphs pp512 t/s (target: ~671 t/s)

**Step 2: Run XMX+graphs benchmark**

```bash
GGML_SYCL_XMX_MOE=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
  -p 512 -n 128 -ngl 99 -fa 1
```

Expected: Within 10% of ESIMD baseline (~600+ t/s)

**Step 3: Document results**

If performance meets target, create summary commit:

```bash
git commit --allow-empty -m "perf(sycl): XMX MoE graph-compatible - achieved Xt/s vs ESIMD Yt/s"
```

---

## Success Criteria

1. ✅ XMX MoE works with `GGML_SYCL_DISABLE_GRAPH=0` (default)
2. ✅ Zero `.wait()` calls in hot path (except single total_tiles read)
3. ✅ Performance within 10% of ESIMD+graphs baseline
4. ✅ Correct output verified against reference model

## Files Modified Summary

| File | Changes |
|------|---------|
| `ggml/src/ggml-sycl/moe-sort.hpp` | Add `moe_compute_tile_mapping()` |
| `ggml/src/ggml-sycl/moe-xmx.hpp` | Add `find_expert_for_workgroup()`, `launch_fused_xmx_moe_q8_0_soa()` |
| `ggml/src/ggml-sycl/common.hpp` | Extend `xmx_moe_buffers` with tile mapping |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | Replace host iteration with fused dispatch |
