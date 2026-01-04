# XMX MoE GEMM Kernel Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement working XMX-accelerated GEMM kernels for MoE expert dispatch, replacing the current skeleton implementations that produce zeros.

**Architecture:** Two-phase approach with pre-quantization (Phase 0) followed by per-expert XMX GEMM (Phase 1). Uses Intel XMX joint_matrix API with device-aware SLM allocation.

**Tech Stack:** Intel SYCL, XMX joint_matrix API, Intel Arc B580 GPU (M=8, N=16, K=32 tiles)

---

## Design Decisions

### 1. Token Quantization Strategy: Pre-Quantize Before Kernel

**Decision:** Approach A - Pre-quantize fp16 tokens to int8 in a separate kernel before XMX GEMM.

**Rationale:**
- XMX hardware requires int8 operands; fp16 tokens can't be used directly
- Pre-quantization amortizes overhead across all experts
- Simpler kernel logic (no quantization inside GEMM loop)
- Quantized tokens can be reused for multiple experts

**Implementation:**
```cpp
// Phase 0: Pre-quantization kernel (one pass)
// Input: sorted_tokens[batch, in_dim] fp16
// Output: q_tokens[batch, in_dim] int8, scales[batch, in_dim/32] fp16
void preprocess_tokens_q8(
    const sycl::half* tokens,
    int8_t* q_tokens,
    sycl::half* scales,
    int64_t batch,
    int64_t in_dim,
    sycl::queue& queue);
```

### 2. Weight Handling Strategy: Direct LUT Mapping for MXFP4

**Decision:** Use kvalues_mxfp4 directly as int8 values without intermediate float conversion.

**Rationale:**
- kvalues_mxfp4 = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}
- All values fit in int8 range (-128 to 127)
- XMX can process int8 weights directly
- E8M0 exponent becomes part of output scale factor

**MXFP4 Block Layout:**
```
[16 bytes packed values][1 byte E8M0 exponent] = 17 bytes per 32 elements
Each byte: [high_nibble << 4 | low_nibble]
```

**Unpacking Pattern:**
```cpp
const int8_t kvalues_mxfp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// For each packed byte:
int8_t low  = kvalues_mxfp4[packed_byte & 0x0F];
int8_t high = kvalues_mxfp4[packed_byte >> 4];
```

### 3. Scale Accumulation Strategy: Per-K-Block Application

**Decision:** Apply scales after each K-block (32 elements), accumulate in float.

**Rationale:**
- XMX outputs int32 accumulator (8-bit × 8-bit → 32-bit)
- Must apply quantization scales before accumulator overflow
- Token scale, weight scale, and MXFP4 exponent combine per K-block
- Float accumulation preserves precision across blocks

**Pattern:**
```cpp
for (int k_block = 0; k_block < K_blocks; k_block++) {
    // XMX GEMM for this K-block: C_int32 += A_int8 * B_int8
    joint_matrix_mad(sg, C, A, B, C);

    // Extract and apply scales
    int32_t raw = joint_matrix_get(C);
    float token_scale = token_scales[k_block];
    float weight_scale = weight_scales[k_block];  // or pow2(e8m0_exp - 127) for MXFP4

    float_accum += raw * token_scale * weight_scale * 0.5f;  // 0.5f for MXFP4 LUT

    // Reset accumulator for next K-block
    joint_matrix_fill(sg, C, 0);
}
```

### 4. Q8_0 Weight Handling

**Q8_0 Block Layout:**
```
[2 bytes fp16 scale][32 bytes int8 values] = 34 bytes per 32 elements
```

**Processing:**
- Load 32 int8 values directly for XMX
- Extract fp16 scale for output scaling
- Simpler than MXFP4 (no nibble unpacking needed)

---

## Kernel Architecture

### Phase 0: Token Pre-Quantization

```cpp
// One kernel launch before all expert GEMMs
// Work distribution: one work-group per row, one sub-group per 32-element block

void preprocess_tokens_q8(
    const sycl::half* tokens,      // [batch, in_dim]
    int8_t* q_tokens,              // [batch, in_dim]
    sycl::half* scales,            // [batch, in_dim/32]
    int64_t batch,
    int64_t in_dim,
    sycl::queue& queue)
{
    constexpr int QK = 32;  // Quantization block size
    constexpr int SG_SIZE = 16;

    queue.parallel_for(
        nd_range<1>(batch * (in_dim/QK) * SG_SIZE, SG_SIZE),
        [=](nd_item<1> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {
            auto sg = item.get_sub_group();
            int block_id = item.get_group(0);
            int row = block_id / (in_dim/QK);
            int k_block = block_id % (in_dim/QK);

            // Each lane loads 2 values (32 total per sub-group)
            int lane = sg.get_local_linear_id();
            float v0 = tokens[row * in_dim + k_block * QK + lane * 2];
            float v1 = tokens[row * in_dim + k_block * QK + lane * 2 + 1];

            // Find max absolute value via sub-group reduction
            float amax = sycl::reduce_over_group(sg,
                sycl::max(sycl::fabs(v0), sycl::fabs(v1)), sycl::maximum<float>());

            // Compute scale and quantize
            float scale = amax / 127.0f;
            float inv_scale = (amax != 0.0f) ? 127.0f / amax : 0.0f;

            int8_t q0 = sycl::round(v0 * inv_scale);
            int8_t q1 = sycl::round(v1 * inv_scale);

            // Store quantized values
            q_tokens[row * in_dim + k_block * QK + lane * 2] = q0;
            q_tokens[row * in_dim + k_block * QK + lane * 2 + 1] = q1;

            // Store scale (one per block)
            if (lane == 0) {
                scales[row * (in_dim/QK) + k_block] = sycl::half(scale);
            }
        });
}
```

### Phase 1: XMX GEMM Kernel

**Grid Dimensions:**
```cpp
// Per-expert kernel launch
int work_groups_m = (batch + TILES_M * XMX_M - 1) / (TILES_M * XMX_M);
int work_groups_n = (out_dim + TILES_N * XMX_N - 1) / (TILES_N * XMX_N);
// Total work-groups: work_groups_m * work_groups_n

int threads_per_wg = 256;  // 16 sub-groups of 16 threads
```

**SLM Allocation (Device-Aware):**
```cpp
struct MoEXMXConfig {
    int tiles_m = 4;
    int tiles_n = 4;
    size_t slm_token_bytes;   // tokens for one K-block
    size_t slm_weight_bytes;  // weights for one K-block (double-buffered)
    size_t slm_scale_bytes;   // scales storage

    static MoEXMXConfig from_capabilities(const XMXCapabilities& caps) {
        MoEXMXConfig cfg;

        // Use 80% of SLM as budget
        size_t slm_budget = caps.slm_size * 8 / 10;

        // Token tiles: TILES_M * M * K = 4 * 8 * 32 = 1024 bytes (int8)
        cfg.slm_token_bytes = cfg.tiles_m * caps.M * caps.K;

        // Weight tiles: TILES_N * N * K = 4 * 16 * 32 = 2048 bytes (int8)
        // Double-buffered: 4096 bytes
        cfg.slm_weight_bytes = cfg.tiles_n * caps.N * caps.K * 2;

        // Scales: 256 bytes for token + weight scales
        cfg.slm_scale_bytes = 256;

        // Total: ~5.3KB per work-group (well within 64KB)
        assert(cfg.slm_token_bytes + cfg.slm_weight_bytes + cfg.slm_scale_bytes <= slm_budget);

        return cfg;
    }
};
```

**Kernel Flow:**
```cpp
template<int TILES_M = 4, int TILES_N = 4>
void launch_xmx_moe_gemm_q8_0(
    const void* weights_qs,       // [out_dim, in_dim] Q8_0
    const int8_t* q_tokens,       // [batch, in_dim] pre-quantized
    const sycl::half* token_scales, // [batch, in_dim/32]
    sycl::half* output,           // [batch, out_dim]
    int64_t batch, int64_t out_dim, int64_t in_dim,
    const MoEXMXConfig& cfg,
    sycl::queue& queue)
{
    constexpr int XMX_M = 8, XMX_N = 16, XMX_K = 32;
    constexpr int SG_SIZE = 16;
    constexpr int NUM_SG = 16;  // Sub-groups per work-group

    int wg_m = (batch + TILES_M * XMX_M - 1) / (TILES_M * XMX_M);
    int wg_n = (out_dim + TILES_N * XMX_N - 1) / (TILES_N * XMX_N);
    int K_blocks = in_dim / XMX_K;

    queue.submit([&](sycl::handler& cgh) {
        // SLM allocation
        sycl::local_accessor<int8_t, 1> slm_tokens(cfg.slm_token_bytes, cgh);
        sycl::local_accessor<int8_t, 1> slm_weights(cfg.slm_weight_bytes, cgh);
        sycl::local_accessor<sycl::half, 1> slm_scales(cfg.slm_scale_bytes / 2, cgh);

        cgh.parallel_for(
            nd_range<2>({wg_m * NUM_SG, wg_n * SG_SIZE}, {NUM_SG, SG_SIZE}),
            [=](nd_item<2> item) [[intel::reqd_sub_group_size(SG_SIZE)]] {

                auto sg = item.get_sub_group();
                int wg_id_m = item.get_group(0);
                int wg_id_n = item.get_group(1);
                int sg_id = item.get_local_id(0);

                // Each sub-group owns one output tile
                int tile_m = sg_id / TILES_N;
                int tile_n = sg_id % TILES_N;

                // Output tile position
                int out_row = wg_id_m * TILES_M * XMX_M + tile_m * XMX_M;
                int out_col = wg_id_n * TILES_N * XMX_N + tile_n * XMX_N;

                // Float accumulator for precision
                float acc[XMX_M][XMX_N] = {0};

                // K-loop with SLM staging
                for (int k = 0; k < K_blocks; k++) {
                    // === Load token tile to SLM ===
                    // Cooperative load across all sub-groups
                    // ...

                    // === Load weight tile to SLM ===
                    // Extract int8 values from Q8_0 blocks
                    // ...

                    item.barrier(sycl::access::fence_space::local_space);

                    // === XMX GEMM ===
                    joint_matrix<sub_group, int8_t, use::a, XMX_M, XMX_K, layout::row_major> A;
                    joint_matrix<sub_group, int8_t, use::b, XMX_K, XMX_N, layout::row_major> B;
                    joint_matrix<sub_group, int32_t, use::accumulator, XMX_M, XMX_N> C;

                    // Load from SLM
                    auto slm_tokens_ptr = address_space_cast<access::address_space::local_space>(
                        slm_tokens.get_pointer());
                    auto slm_weights_ptr = address_space_cast<access::address_space::local_space>(
                        slm_weights.get_pointer());

                    joint_matrix_load(sg, A, slm_tokens_ptr + tile_m * XMX_M * XMX_K, XMX_K);
                    joint_matrix_load(sg, B, slm_weights_ptr + tile_n * XMX_N * XMX_K, XMX_N);
                    joint_matrix_fill(sg, C, 0);
                    joint_matrix_mad(sg, C, A, B, C);

                    // === Apply scales and accumulate ===
                    float token_scale = slm_scales[tile_m];  // Loaded cooperatively
                    float weight_scale = slm_scales[TILES_M + tile_n];

                    // Extract C and accumulate with scales
                    // (Implementation detail: iterate over C elements)
                    for (int m = 0; m < XMX_M; m++) {
                        for (int n = 0; n < XMX_N; n++) {
                            int32_t raw = /* extract C[m][n] */;
                            acc[m][n] += raw * token_scale * weight_scale;
                        }
                    }

                    item.barrier(sycl::access::fence_space::local_space);
                }

                // === Write output ===
                for (int m = 0; m < XMX_M; m++) {
                    for (int n = 0; n < XMX_N; n++) {
                        if (out_row + m < batch && out_col + n < out_dim) {
                            output[(out_row + m) * out_dim + out_col + n] =
                                sycl::half(acc[m][n]);
                        }
                    }
                }
            });
    });
}
```

---

## Summary

| Component | Decision | Rationale |
|-----------|----------|-----------|
| Token quantization | Pre-quantize (Phase 0) | Amortizes across experts, simpler GEMM |
| MXFP4 weights | Direct LUT to int8 | kvalues fit in int8, no float needed |
| Q8_0 weights | Direct int8 load | Already int8 format |
| Scale handling | Per-K-block accumulate | Prevents int32 overflow, preserves precision |
| SLM allocation | Device-aware via XMXCapabilities | Optimal tile count based on slm_size |

## Risk Assessment

- **Medium risk**: XMX joint_matrix API usage matches mmq_xmx.cpp reference
- **Implementation gaps**: Token pre-quantization kernel not yet tested in isolation
- **Performance unknown**: Need benchmarking vs fused ESIMD approach
- **Correctness first**: Start with functional implementation, optimize later
