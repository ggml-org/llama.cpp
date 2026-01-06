# XMX MXFP4 MoE Kernel Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement XMX-accelerated MXFP4 MoE inference with SYCL command graph compatibility

**Status:** ✅ Partially Implemented (Graph compatibility achieved via ESIMD fused path)

**Architecture:** MXFP4 weights converted to INT8 via LUT, processed through INT8×INT8 XMX DPAS operations in a fused persistent work-group kernel

**Tech Stack:** Intel SYCL, XMX matrix extensions, MXFP4 quantization

## Implementation Status (2026-01-06)

The SYCL command graph compatibility goal was achieved through a simpler approach than originally designed:

1. **MXFP4 SoA fused path** auto-enabled during graph recording
2. **Batch size limits** relaxed during graph recording to prevent fallback to host-side routing
3. **Event chaining** used throughout to avoid `.wait()` calls

**Results:**
- ✅ Correctness verified: Output matches expected "1, 2, 3, 4, 5"
- ✅ Graphs working: "graphs reused = 13"
- ✅ Performance: **680.60 t/s pp512** (exceeds target of ~671 t/s)

The XMX DPAS kernel design below remains valid for future optimization work to further accelerate MXFP4 MoE operations beyond the current ESIMD implementation.

---

## 1. Architecture Overview

The implementation converts MXFP4 weights to INT8 on-the-fly using a 16-entry lookup table, then leverages INT8×INT8 XMX DPAS for maximum throughput.

### Data Flow

```
Expert Weights (MXFP4 AoS)
         │
         ▼ [One-time at model load]
XMX-Optimized Layout (tile-aligned)
         │
         ▼ [Per inference - in kernel]
┌─────────────────────────────────────┐
│     Fused XMX MoE Kernel            │
│                                     │
│  1. Load MXFP4 block from XMX layout│
│  2. LUT lookup: nibble → INT8       │
│     {0,±1,±2,±3,±4,±6,±8,±12}      │
│  3. Scale INT8 by block scale       │
│  4. XMX DPAS: INT8 × INT8 → INT32   │
│  5. Accumulate, convert to FP16     │
└─────────────────────────────────────┘
         │
         ▼
    Output (FP16)
```

### Key Insight

MXFP4 block size (QK_MXFP4=32) matches XMX K dimension exactly, allowing one MXFP4 block to feed one XMX accumulation step without padding or regrouping.

---

## 2. Hardware-Aware Configuration

All configuration parameters are derived from runtime hardware queries, not hardcoded values.

### Configuration Structure

```cpp
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
        size_t weight_tile_bytes = cfg.tile_n_total * XMX_K / 2;  // MXFP4
        size_t lut_bytes = 16;
        size_t token_tile_bytes = XMX_M * XMX_K * sizeof(sycl::half);
        cfg.use_double_buffer = (2 * weight_tile_bytes + lut_bytes + token_tile_bytes)
                                 < cfg.slm_budget;
        return cfg;
    }
};
```

### SLM Allocation Strategy

| Component | Size | Purpose |
|-----------|------|---------|
| LUT | 16 bytes | MXFP4→INT8 conversion table |
| Weight tile A | TILE_N × 32 / 2 bytes | Current K-block weights |
| Weight tile B | TILE_N × 32 / 2 bytes | Prefetched next K-block (if double-buffering) |
| Token tile | 8 × K_CHUNK × 2 bytes | Input tokens for TILE_M rows |

---

## 3. XMX-Optimized Memory Layout

### Layout Structure

```
Original MXFP4 AoS: [block0_qs(16B) + block0_scale(1B)] [block1...] ...

XMX-Optimized Layout:
┌─────────────────────────────────────────────────────────────┐
│ Header: n_rows, n_cols, n_blocks_per_row, conversion_flags  │
├─────────────────────────────────────────────────────────────┤
│ Tile Group 0 (K_TILE_GROUPS × N_TILE_GROUPS tiles):         │
│   scales[TILE_K/32][TILE_N]  ← All scales for this group    │
│   qs[TILE_K/32][TILE_N][16]  ← All quantized bytes          │
├─────────────────────────────────────────────────────────────┤
│ Tile Group 1...                                             │
└─────────────────────────────────────────────────────────────┘
```

### Benefits

1. **Coalesced scale loads**: Scales grouped → single vector load per tile
2. **XMX-aligned nibbles**: 32 nibbles (16 bytes) = one K-step worth of weights
3. **Tile-major ordering**: All data for a tile group is contiguous
4. **Prefetch-friendly**: Next tile group can be prefetched while current processes

---

## 4. Fused Kernel Implementation

```cpp
template <typename AccT = float>
void fused_xmx_moe_gemm_mxfp4(
    sycl::nd_item<1> item,
    const sycl::half* sorted_tokens,
    const int32_t* expert_offsets,
    const int32_t* expert_tile_offsets,
    const void* expert_weights_xmx,
    sycl::half* output,
    const MXFPXMXConfig& cfg,
    sycl::local_accessor<int8_t, 1> slm)
{
    // 1. Load LUT to SLM (once per work-group)
    if (item.get_local_id(0) < 16) {
        slm_lut[lid] = kvalues_mxfp4_int8[lid];
    }

    // 2. Binary search: wg_id → expert assignment
    int expert_idx = find_expert_for_workgroup(wg_id, expert_tile_offsets, n_experts);
    int local_tile = wg_id - expert_tile_offsets[expert_idx];

    // 3. Per-tile processing loop
    for each output_tile in assigned_tiles:
        // Load token tile to SLM
        load_tokens_to_slm(sorted_tokens, expert_offsets[expert_idx], tile_row);

        // Accumulate across K dimension
        AccT acc[TILE_M][TILE_N] = {0};
        for (int k = 0; k < hidden_dim; k += TILE_K):
            // Load MXFP4 weights, convert to INT8 via LUT
            load_and_convert_mxfp4_to_int8(weights_xmx, expert_idx, k, slm_weights);

            // XMX DPAS: acc += tokens_int8 × weights_int8
            xmx_dpas_int8(slm_tokens, slm_weights, acc);

        // Write output tile
        store_output_tile(output, expert_offsets[expert_idx], tile_row, acc);
}
```

---

## 5. Layout Converter

```cpp
struct MXFPXMXLayoutInfo {
    int64_t n_rows, n_cols;
    int64_t n_tile_groups_k, n_tile_groups_n;
    int64_t tile_k, tile_n;
    int64_t total_bytes;
};

sycl::event reorder_mxfp4_to_xmx_layout(
    sycl::queue& q,
    const void* src_aos,
    void* dst_xmx,
    int64_t n_rows,
    int64_t n_cols,
    const XMXCapabilities& xmx_caps,
    sycl::event dep);
```

### Conversion Logic

```cpp
// Each work-item handles one MXFP4 block
int block_row = gid / blocks_per_row;
int block_col = gid % blocks_per_row;

// Compute destination tile group
int tile_group_k = (block_col * QK_MXFP4) / tile_k;
int tile_group_n = block_row / tile_n;

// Read from AoS source
uint8_t scale = src_aos[block_idx * 17 + 16];
uint8_t qs[16]; memcpy(qs, &src_aos[block_idx * 17], 16);

// Write to XMX layout destination
dst_scales[tile_group][local_k][local_n] = scale;
dst_qs[tile_group][local_k][local_n][0..15] = qs[0..15];
```

---

## 6. Dispatch Integration

### Decision Tree

```cpp
if (expert_weight_type == GGML_TYPE_BLOCK_MXFP4) {
    const auto& xmx_caps = ggml_sycl_info().devices[device_id].xmx_caps;

    if (xmx_caps.supported &&
        xmx_caps.supports_int8 &&
        hidden_dim % xmx_caps.K == 0) {

        void* weights_xmx = get_or_create_xmx_layout(expert_weights, xmx_caps);

        return launch_fused_xmx_moe_mxfp4(
            *stream, dep_event,
            sorted_tokens, expert_offsets, expert_tile_offsets,
            weights_xmx, output, cfg);
    }

    // Fallback: existing ESIMD fused kernel
    return try_fused_moe_esimd_mxfp4(...);
}
```

### Environment Control

```cpp
// GGML_SYCL_XMX_MOE=1 enables XMX path
static bool use_xmx_moe = getenv("GGML_SYCL_XMX_MOE")
                          && atoi(getenv("GGML_SYCL_XMX_MOE")) != 0;
```

### Files to Modify

- `ggml/src/ggml-sycl/ggml-sycl.cpp` - Dispatch logic
- `ggml/src/ggml-sycl/moe-xmx.hpp` - Add MXFP4 kernel and converter
- `ggml/src/ggml-sycl/common.hpp` - Add xmx_mxfp4_layout to tensor extra

---

## 7. Testing Strategy

### Correctness Testing

```bash
# Baseline: ESIMD path
GGML_SYCL_DISABLE_GRAPH=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf -ngl 99 --flash-attn on \
  --no-conversation -p 'Count from 1 to 5:' -n 15 --seed 42 --temp 0

# XMX MXFP4 path
GGML_SYCL_XMX_MOE=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf -ngl 99 --flash-attn on \
  --no-conversation -p 'Count from 1 to 5:' -n 15 --seed 42 --temp 0
```

### Performance Benchmarking

```bash
GGML_SYCL_XMX_MOE=0 ./build/bin/llama-bench -m gpt-oss-20b-Q8_0.gguf -p 512 -n 128 -ngl 99 -fa 1
GGML_SYCL_XMX_MOE=1 ./build/bin/llama-bench -m gpt-oss-20b-Q8_0.gguf -p 512 -n 128 -ngl 99 -fa 1
```

### Success Criteria

1. Output matches ESIMD baseline exactly
2. Graph reuse confirmed (N > 0)
3. Performance within 10% of ESIMD (ideally faster)
4. Works on Arc A770 (64KB SLM) and B580 (128KB SLM)
