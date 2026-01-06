# XMX MXFP4 MoE Kernel Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add XMX tile-aligned memory layout for MXFP4 MoE weights to improve memory access patterns and enable INT8 XMX acceleration

**Architecture:** New tile-aligned layout groups scales and qs by XMX tiles (not by block), enabling coalesced loads. Layout converter runs at model load, fused kernel uses tile-aligned data.

**Tech Stack:** Intel SYCL, XMX joint_matrix, MXFP4 quantization (type 39)

---

## Task 1: Add MXFPXMXConfig Structure

**Files:**
- Modify: `ggml/src/ggml-sycl/moe-xmx-fused.hpp:24-43`

**Step 1: Add the MXFP4 XMX config struct after FusedMoEConfig**

Add this code after line 43 (after the closing brace of `FusedMoEConfig::from_device`):

```cpp
// MXFP4-specific XMX configuration
struct MXFPXMXConfig {
    // Fixed XMX dimensions (Intel spec)
    static constexpr int XMX_M = 8;
    static constexpr int XMX_N = 16;
    static constexpr int XMX_K = 32;  // Matches QK_MXFP4

    // Dynamic from hardware
    int tiles_n;              // From xmx_caps.optimal_tiles_n
    int num_persistent_wgs;   // From dev_info.nsm * 2
    size_t slm_budget;        // From xmx_caps.slm_size

    // Derived
    int tile_n_total;         // XMX_N * tiles_n
    bool use_double_buffer;   // If SLM budget permits

    static MXFPXMXConfig from_device(int device_id) {
        const auto& dev = ggml_sycl_info().devices[device_id];
        const auto& xmx = dev.xmx_caps;

        MXFPXMXConfig cfg;
        cfg.tiles_n = xmx.optimal_tiles_n > 0 ? xmx.optimal_tiles_n : 4;
        cfg.num_persistent_wgs = dev.nsm * 2;
        cfg.slm_budget = xmx.slm_size > 0 ? xmx.slm_size : 65536;
        cfg.tile_n_total = XMX_N * cfg.tiles_n;

        // Double buffer if SLM can hold 2× weight tiles + LUT + tokens
        size_t weight_tile_bytes = cfg.tile_n_total * XMX_K / 2;  // MXFP4: 4 bits per element
        size_t lut_bytes = 16;
        size_t token_tile_bytes = XMX_M * XMX_K * sizeof(int8_t);
        cfg.use_double_buffer = (2 * weight_tile_bytes + lut_bytes + token_tile_bytes) < cfg.slm_budget;

        return cfg;
    }
};
```

**Step 2: Verify it compiles**

Run: `./scripts/quick-rebuild.sh moe-xmx-fused.hpp`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/moe-xmx-fused.hpp
git commit -m "feat(sycl): add MXFPXMXConfig for hardware-aware MXFP4 XMX config"
```

---

## Task 2: Add XMX Tile-Aligned Layout Info Structure

**Files:**
- Modify: `ggml/src/ggml-sycl/moe-xmx-fused.hpp` (after MXFPXMXConfig)

**Step 1: Add layout info struct**

Add after MXFPXMXConfig:

```cpp
// XMX tile-aligned layout metadata
// Layout: [tile_groups...] where each tile_group contains:
//   scales[tiles_k][tile_n_total] followed by qs[tiles_k][tile_n_total][16]
struct MXFPXMXLayoutInfo {
    int64_t n_rows;           // out_dim
    int64_t n_cols;           // in_dim
    int64_t n_tile_groups_k;  // ceil(in_dim / (XMX_K * tiles_k_per_group))
    int64_t n_tile_groups_n;  // ceil(out_dim / tile_n_total)
    int64_t tile_n_total;     // XMX_N * tiles_n (from hardware)
    int64_t tiles_k_per_group; // Number of K blocks per tile group
    int64_t total_bytes;      // Size of converted buffer

    // Compute layout info for a weight tensor
    static MXFPXMXLayoutInfo compute(int64_t out_dim, int64_t in_dim, const MXFPXMXConfig& cfg) {
        MXFPXMXLayoutInfo info;
        info.n_rows = out_dim;
        info.n_cols = in_dim;
        info.tile_n_total = cfg.tile_n_total;
        info.tiles_k_per_group = 1;  // One K block per group for simplicity

        constexpr int XMX_K = 32;
        int64_t n_k_blocks = in_dim / XMX_K;

        info.n_tile_groups_k = n_k_blocks;  // One tile group per K block
        info.n_tile_groups_n = (out_dim + cfg.tile_n_total - 1) / cfg.tile_n_total;

        // Per tile group: scales + packed qs
        // scales: [tile_n_total] uint8
        // qs: [tile_n_total][16] uint8 (16 bytes = 32 nibbles per block)
        int64_t bytes_per_tile_group = info.tile_n_total * (1 + 16);  // 1 scale + 16 qs per column

        info.total_bytes = info.n_tile_groups_k * info.n_tile_groups_n * bytes_per_tile_group;

        return info;
    }
};
```

**Step 2: Verify it compiles**

Run: `./scripts/quick-rebuild.sh moe-xmx-fused.hpp`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/moe-xmx-fused.hpp
git commit -m "feat(sycl): add MXFPXMXLayoutInfo for tile-aligned MXFP4 layout"
```

---

## Task 3: Add Layout Converter Function (Host Implementation)

**Files:**
- Modify: `ggml/src/ggml-sycl/moe-xmx-fused.hpp` (after MXFPXMXLayoutInfo)

**Step 1: Add the layout converter function**

```cpp
// Convert MXFP4 weights from SoA layout to XMX tile-aligned layout
// SoA input: qs[nblocks * 16], e[nblocks] where nblocks = out_dim * (in_dim/32)
// XMX output: [tile_groups...] with scales and qs grouped by tile
//
// This runs on host at model load time (not in hot path)
inline void reorder_mxfp4_to_xmx_layout(
    const uint8_t* src_qs,        // SoA packed nibbles [nblocks * 16]
    const uint8_t* src_e,         // SoA exponents [nblocks]
    uint8_t* dst,                 // XMX tile-aligned output
    const MXFPXMXLayoutInfo& info) {

    constexpr int XMX_K = 32;
    constexpr int PACKED_BYTES = 16;  // 32 nibbles packed into 16 bytes

    const int64_t n_k_blocks = info.n_cols / XMX_K;

    // Iterate over tile groups
    uint8_t* dst_ptr = dst;

    for (int64_t tg_k = 0; tg_k < info.n_tile_groups_k; tg_k++) {
        for (int64_t tg_n = 0; tg_n < info.n_tile_groups_n; tg_n++) {
            // Write scales for this tile group [tile_n_total]
            for (int64_t tn = 0; tn < info.tile_n_total; tn++) {
                int64_t out_col = tg_n * info.tile_n_total + tn;
                if (out_col < info.n_rows) {
                    // SoA block index: out_col * n_k_blocks + k_block
                    int64_t src_block_idx = out_col * n_k_blocks + tg_k;
                    *dst_ptr++ = src_e[src_block_idx];
                } else {
                    *dst_ptr++ = 0;  // Padding for out-of-bounds
                }
            }

            // Write packed qs for this tile group [tile_n_total][16]
            for (int64_t tn = 0; tn < info.tile_n_total; tn++) {
                int64_t out_col = tg_n * info.tile_n_total + tn;
                if (out_col < info.n_rows) {
                    int64_t src_block_idx = out_col * n_k_blocks + tg_k;
                    const uint8_t* src_qs_block = src_qs + src_block_idx * PACKED_BYTES;
                    for (int b = 0; b < PACKED_BYTES; b++) {
                        *dst_ptr++ = src_qs_block[b];
                    }
                } else {
                    // Zero padding
                    for (int b = 0; b < PACKED_BYTES; b++) {
                        *dst_ptr++ = 0;
                    }
                }
            }
        }
    }
}
```

**Step 2: Verify it compiles**

Run: `./scripts/quick-rebuild.sh moe-xmx-fused.hpp`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/moe-xmx-fused.hpp
git commit -m "feat(sycl): add reorder_mxfp4_to_xmx_layout converter"
```

---

## Task 4: Add XMX Tile-Aligned Kernel

**Files:**
- Modify: `ggml/src/ggml-sycl/moe-xmx-fused.hpp` (after reorder function)

**Step 1: Add the new kernel that reads tile-aligned layout**

```cpp
// Fused XMX MoE GEMM for MXFP4 weights (XMX tile-aligned layout)
// This kernel reads the tile-aligned layout created by reorder_mxfp4_to_xmx_layout
template <int TILES_M = 4, int TILES_N = 4>
sycl::event fused_xmx_moe_gemm_mxfp4_tiled(
    sycl::event dep_event,
    const uint8_t* all_expert_weights_tiled,  // XMX tile-aligned layout per expert
    const int8_t* q_tokens,                   // [num_tokens, in_dim]
    const sycl::half* token_scales,           // [num_tokens, in_dim/32]
    const int32_t* sorted_token_ids,
    const int32_t* expert_offsets,
    sycl::half* output,
    int num_tokens,
    int n_experts,
    int64_t out_dim,
    int64_t in_dim,
    int64_t expert_tiled_stride,              // Bytes between expert tiled weight buffers
    const MXFPXMXConfig& cfg,
    sycl::queue& queue) {

    constexpr int XMX_M = 8, XMX_N = 16, XMX_K = 32;
    constexpr int SG_SIZE = 16;
    constexpr int TILE_N = TILES_N * XMX_N;

    const int num_k_blocks = in_dim / XMX_K;
    const int n_output_tiles = (out_dim + TILE_N - 1) / TILE_N;

    // Bytes per tile group in XMX layout: scales[TILE_N] + qs[TILE_N][16]
    const int64_t bytes_per_tile_group = TILE_N * (1 + 16);

    (void)num_tokens;

    return queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(dep_event);

        // SLM allocations
        sycl::local_accessor<int8_t, 1> slm_token(sycl::range<1>(XMX_M * XMX_K), cgh);
        sycl::local_accessor<float, 1> slm_token_scale(sycl::range<1>(1), cgh);
        sycl::local_accessor<int8_t, 1> slm_weights(sycl::range<1>(TILE_N * XMX_K), cgh);
        sycl::local_accessor<float, 1> slm_weight_scales(sycl::range<1>(TILE_N), cgh);
        sycl::local_accessor<int8_t, 1> slm_kvalues(sycl::range<1>(16), cgh);
        sycl::local_accessor<int32_t, 1> slm_acc(sycl::range<1>(cfg.num_persistent_wgs / SG_SIZE * XMX_M * XMX_N), cgh);

        const int num_persistent_wgs = cfg.num_persistent_wgs;

        cgh.parallel_for(
            sycl::nd_range<1>(num_persistent_wgs * 256, 256),
            [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
                auto sg = item.get_sub_group();
                int group_id = item.get_group_linear_id();
                int sg_id = sg.get_group_linear_id();
                int lane = sg.get_local_linear_id();

                // Load LUT
                if (sg_id == 0 && lane < 16) {
                    slm_kvalues[lane] = kvalues_mxfp4[lane];
                }
                sycl::group_barrier(item.get_group());

                if (sg_id != 0) return;

                for (int expert = 0; expert < n_experts; expert++) {
                    int expert_start = expert_offsets[expert];
                    int expert_end = expert_offsets[expert + 1];
                    int expert_tokens = expert_end - expert_start;
                    if (expert_tokens == 0) continue;

                    int64_t expert_work = static_cast<int64_t>(expert_tokens) * n_output_tiles;
                    const uint8_t* expert_tiled = all_expert_weights_tiled + expert * expert_tiled_stride;

                    for (int64_t local_work = group_id; local_work < expert_work; local_work += num_persistent_wgs) {
                        int tile_idx = local_work % n_output_tiles;
                        int local_token_idx = local_work / n_output_tiles;
                        int sorted_idx = expert_start + local_token_idx;

                        float float_acc[TILES_N * XMX_N] = {0.0f};

                        joint_matrix<sycl::sub_group, int32_t, use::accumulator, XMX_M, XMX_N> acc[TILES_N];
                        for (int tn = 0; tn < TILES_N; tn++) {
                            joint_matrix_fill(sg, acc[tn], 0);
                        }

                        for (int k_block = 0; k_block < num_k_blocks; k_block++) {
                            // Load token
                            for (int i = lane; i < XMX_M * XMX_K; i += SG_SIZE) {
                                int row = i / XMX_K;
                                int col = i % XMX_K;
                                slm_token[i] = (row == 0) ? q_tokens[sorted_idx * in_dim + k_block * XMX_K + col] : 0;
                            }
                            if (lane == 0) {
                                slm_token_scale[0] = static_cast<float>(token_scales[sorted_idx * num_k_blocks + k_block]);
                            }

                            // Load weights from tile-aligned layout
                            // Tile group offset: (k_block * n_tile_groups_n + tile_idx) * bytes_per_tile_group
                            const uint8_t* tile_group = expert_tiled +
                                (k_block * n_output_tiles + tile_idx) * bytes_per_tile_group;
                            const uint8_t* scales_ptr = tile_group;
                            const uint8_t* qs_ptr = tile_group + TILE_N;

                            for (int i = lane; i < TILE_N; i += SG_SIZE) {
                                // Load scale (E8M0 -> float)
                                slm_weight_scales[i] = sycl_e8m0_to_fp32_half(scales_ptr[i]);

                                // Unpack nibbles
                                const uint8_t* packed = qs_ptr + i * 16;
                                for (int k = 0; k < 16; k++) {
                                    uint8_t byte = packed[k];
                                    slm_weights[i * XMX_K + k] = slm_kvalues[byte & 0xF];
                                    slm_weights[i * XMX_K + k + 16] = slm_kvalues[byte >> 4];
                                }
                            }
                            sycl::group_barrier(sg);

                            // XMX compute
                            joint_matrix<sycl::sub_group, int8_t, use::a, XMX_M, XMX_K, layout::row_major> mat_a;
                            joint_matrix<sycl::sub_group, int8_t, use::b, XMX_K, XMX_N, layout::col_major> mat_b;

                            auto slm_token_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                sycl::access::decorated::no>(&slm_token[0]);
                            joint_matrix_load(sg, mat_a, slm_token_ptr, XMX_K);

                            for (int tn = 0; tn < TILES_N; tn++) {
                                auto slm_weights_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                    sycl::access::decorated::no>(&slm_weights[tn * XMX_N * XMX_K]);
                                joint_matrix_load(sg, mat_b, slm_weights_ptr, XMX_K);
                                joint_matrix_mad(sg, acc[tn], mat_a, mat_b, acc[tn]);
                            }

                            // Extract and accumulate with scales
                            int32_t* sg_acc_ptr = &slm_acc[sg_id * XMX_M * XMX_N];
                            float t_scale = slm_token_scale[0];

                            for (int tn = 0; tn < TILES_N; tn++) {
                                auto acc_slm_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                    sycl::access::decorated::no>(sg_acc_ptr);
                                joint_matrix_store(sg, acc[tn], acc_slm_ptr, XMX_N, layout::row_major);
                                sycl::group_barrier(sg);

                                for (int i = lane; i < XMX_N; i += SG_SIZE) {
                                    float w_scale = slm_weight_scales[tn * XMX_N + i];
                                    float_acc[tn * XMX_N + i] += sg_acc_ptr[i] * t_scale * w_scale;
                                }
                                joint_matrix_fill(sg, acc[tn], 0);
                            }
                            sycl::group_barrier(sg);
                        }

                        // Store output
                        for (int tn = 0; tn < TILES_N; tn++) {
                            for (int i = lane; i < XMX_N; i += SG_SIZE) {
                                int out_col = tile_idx * TILE_N + tn * XMX_N + i;
                                if (out_col < static_cast<int>(out_dim)) {
                                    output[sorted_idx * out_dim + out_col] = sycl::half(float_acc[tn * XMX_N + i]);
                                }
                            }
                        }
                        sycl::group_barrier(sg);
                    }
                }
            });
    });
}
```

**Step 2: Verify it compiles**

Run: `./scripts/quick-rebuild.sh moe-xmx-fused.hpp`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/moe-xmx-fused.hpp
git commit -m "feat(sycl): add fused_xmx_moe_gemm_mxfp4_tiled kernel"
```

---

## Task 5: Add Entry Point Function

**Files:**
- Modify: `ggml/src/ggml-sycl/moe-xmx-fused.hpp` (after kernel, before `#endif`)

**Step 1: Add the entry point**

```cpp
// Entry point for fused XMX MoE dispatch (MXFP4 XMX tile-aligned layout)
inline std::pair<bool, sycl::event> try_fused_xmx_moe_mxfp4_tiled(
    sycl::event dep_event,
    const uint8_t* all_expert_weights_tiled,
    const int8_t* q_tokens,
    const sycl::half* token_scales,
    const int32_t* sorted_token_ids,
    const int32_t* expert_offsets,
    sycl::half* output,
    int num_tokens,
    int n_experts,
    int64_t out_dim,
    int64_t in_dim,
    int64_t expert_tiled_stride,
    int device_id,
    sycl::queue& queue) {

    const auto& dev_info = ggml_sycl_info().devices[device_id];
    if (!dev_info.xmx_caps.supported) {
        return {false, sycl::event{}};
    }

    MXFPXMXConfig cfg = MXFPXMXConfig::from_device(device_id);

    GGML_SYCL_DEBUG(
        "[MoE-Fused] Launching XMX MXFP4 tiled kernel: "
        "tokens=%d experts=%d out=%ld in=%ld wgs=%d tiled_stride=%ld\n",
        num_tokens, n_experts, out_dim, in_dim, cfg.num_persistent_wgs, expert_tiled_stride);

    sycl::event evt = moe_xmx_fused::fused_xmx_moe_gemm_mxfp4_tiled<4, 4>(
        dep_event, all_expert_weights_tiled, q_tokens, token_scales,
        sorted_token_ids, expert_offsets, output, num_tokens, n_experts,
        out_dim, in_dim, expert_tiled_stride, cfg, queue);

    return {true, evt};
}
```

**Step 2: Verify it compiles**

Run: `./scripts/quick-rebuild.sh moe-xmx-fused.hpp`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/moe-xmx-fused.hpp
git commit -m "feat(sycl): add try_fused_xmx_moe_mxfp4_tiled entry point"
```

---

## Task 6: Add Environment Variable Control

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (near other env var declarations, ~line 100-200)

**Step 1: Search for existing env var pattern**

Run: `grep -n "GGML_SYCL_XMX_MOE" ggml/src/ggml-sycl/ggml-sycl.cpp | head -5`

**Step 2: Add tiled layout env var near existing XMX_MOE var**

Find the existing `GGML_SYCL_XMX_MOE` declaration and add below it:

```cpp
// Enable XMX tile-aligned layout for MXFP4 MoE (requires XMX_MOE=1)
static bool g_ggml_sycl_xmx_moe_tiled = []() {
    const char* val = getenv("GGML_SYCL_XMX_MOE_TILED");
    return val && atoi(val) != 0;
}();
```

**Step 3: Verify it compiles**

Run: `cmake --build build -j 16 2>&1 | tail -20`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "feat(sycl): add GGML_SYCL_XMX_MOE_TILED env var"
```

---

## Task 7: Add Tiled Layout Cache to Tensor Extra

**Files:**
- Modify: `ggml/src/ggml-sycl/common.hpp` (in ggml_tensor_extra_gpu struct)

**Step 1: Find ggml_tensor_extra_gpu struct**

Run: `grep -n "struct ggml_tensor_extra_gpu" ggml/src/ggml-sycl/common.hpp`

**Step 2: Add tiled layout pointer to the struct**

Add after existing device buffer members:

```cpp
    // XMX tile-aligned MXFP4 layout (cached at first use)
    void* xmx_mxfp4_tiled[GGML_SYCL_MAX_DEVICES] = {nullptr};
    size_t xmx_mxfp4_tiled_size = 0;
```

**Step 3: Verify it compiles**

Run: `cmake --build build -j 16 2>&1 | tail -20`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add ggml/src/ggml-sycl/common.hpp
git commit -m "feat(sycl): add xmx_mxfp4_tiled cache to tensor extra"
```

---

## Task 8: Add Dispatch Integration

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (in MoE dispatch section ~line 11700-11900)

**Step 1: Find the MXFP4 MoE dispatch location**

Run: `grep -n "try_fused_xmx_moe_mxfp4_soa" ggml/src/ggml-sycl/ggml-sycl.cpp | head -3`

**Step 2: Add tiled path dispatch before SoA path**

Insert before the existing `try_fused_xmx_moe_mxfp4_soa` call:

```cpp
// Try XMX tile-aligned path first (if enabled)
if (g_ggml_sycl_xmx_moe_tiled && dev_info.xmx_caps.supported) {
    // Get or create tiled layout
    auto* extra = static_cast<ggml_tensor_extra_gpu*>(expert_weights->extra);
    if (extra && extra->xmx_mxfp4_tiled[device_id] == nullptr) {
        // Create tiled layout on first use
        MXFPXMXConfig cfg = MXFPXMXConfig::from_device(device_id);
        MXFPXMXLayoutInfo info = MXFPXMXLayoutInfo::compute(out_dim, in_dim, cfg);

        void* tiled_buf = sycl::malloc_device(info.total_bytes * n_experts, *stream);
        // TODO: Convert each expert's weights to tiled layout
        // For now, this is a placeholder - actual conversion needs SoA pointers

        extra->xmx_mxfp4_tiled[device_id] = tiled_buf;
        extra->xmx_mxfp4_tiled_size = info.total_bytes * n_experts;
    }

    if (extra && extra->xmx_mxfp4_tiled[device_id]) {
        auto [ok, evt] = try_fused_xmx_moe_mxfp4_tiled(
            dep_event,
            static_cast<const uint8_t*>(extra->xmx_mxfp4_tiled[device_id]),
            q_tokens, token_scales, sorted_token_ids, expert_offsets,
            output, num_tokens, n_experts, out_dim, in_dim,
            info.total_bytes,  // expert_tiled_stride
            device_id, *stream);
        if (ok) {
            return evt;
        }
    }
}
```

**Step 3: Verify it compiles**

Run: `cmake --build build -j 16 2>&1 | tail -20`
Expected: Build succeeds (may have warnings about unused variables)

**Step 4: Commit**

```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "wip(sycl): add XMX MXFP4 tiled dispatch skeleton"
```

---

## Task 9: Implement Full Layout Conversion in Dispatch

**Files:**
- Modify: `ggml/src/ggml-sycl/ggml-sycl.cpp` (expand Task 8 skeleton)

**Step 1: Replace the TODO with actual conversion**

Find the "TODO: Convert each expert's weights" comment and replace with:

```cpp
        // Convert each expert's SoA layout to tiled layout
        uint8_t* tiled_ptr = static_cast<uint8_t*>(tiled_buf);
        for (int e = 0; e < n_experts; e++) {
            const uint8_t* expert_qs = all_expert_qs + e * expert_qs_stride;
            const uint8_t* expert_e = all_expert_e + e * expert_e_stride;

            reorder_mxfp4_to_xmx_layout(
                expert_qs, expert_e,
                tiled_ptr + e * info.total_bytes,
                info);
        }
        // Sync after host-side conversion
        stream->wait();
```

**Step 2: Verify it compiles**

Run: `cmake --build build -j 16 2>&1 | tail -20`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add ggml/src/ggml-sycl/ggml-sycl.cpp
git commit -m "feat(sycl): implement MXFP4 tiled layout conversion in dispatch"
```

---

## Task 10: Test Correctness

**Step 1: Build with debug**

Run: `cmake --build build -j 16`

**Step 2: Test baseline (existing SoA path)**

```bash
GGML_SYCL_DEBUG=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf -ngl 99 --flash-attn on \
  --no-conversation -p 'Count from 1 to 5:' -n 15 --seed 42 --temp 0
```
Expected: "1, 2, 3, 4, 5, 6, 7, 8, 9, 10..." with MXFP4 kernel messages

**Step 3: Test tiled path**

```bash
GGML_SYCL_XMX_MOE=1 GGML_SYCL_XMX_MOE_TILED=1 GGML_SYCL_DEBUG=1 \
  ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf -ngl 99 --flash-attn on \
  --no-conversation -p 'Count from 1 to 5:' -n 15 --seed 42 --temp 0
```
Expected: Same output as baseline, with "XMX MXFP4 tiled" in debug messages

**Step 4: Commit test verification**

```bash
git commit --allow-empty -m "test: verify XMX MXFP4 tiled kernel correctness"
```

---

## Task 11: Benchmark Performance

**Step 1: Benchmark baseline**

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf -p 512 -n 128 -ngl 99 -fa 1
```

Record: pp512 t/s, tg128 t/s

**Step 2: Benchmark tiled path**

```bash
GGML_SYCL_XMX_MOE=1 GGML_SYCL_XMX_MOE_TILED=1 \
  ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-bench \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf -p 512 -n 128 -ngl 99 -fa 1
```

Compare: Should be within 10% of baseline (ideally faster)

**Step 3: Document results**

Add benchmark results as comment in commit message.

---

## Task 12: Final Cleanup and Documentation

**Step 1: Update CLAUDE.md with new env var**

Add to the SYCL Environment Variables table:

```markdown
| `GGML_SYCL_XMX_MOE_TILED` | 0 | Enable XMX tile-aligned MXFP4 layout |
```

**Step 2: Commit documentation update**

```bash
git add CLAUDE.md
git commit -m "docs: add GGML_SYCL_XMX_MOE_TILED to env var table"
```

**Step 3: Final verification**

```bash
# Verify graphs work with tiled path
GGML_SYCL_XMX_MOE=1 GGML_SYCL_XMX_MOE_TILED=1 \
  ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf -ngl 99 --flash-attn on \
  --no-conversation -p 'Count from 1 to 5:' -n 15 --seed 42 --temp 0
```
Expected: "graphs reused = N" where N > 0
