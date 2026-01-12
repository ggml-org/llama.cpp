# XMX MoE Coalesced Layout Support Design

**Date**: 2026-01-04
**Status**: Approved
**Branch**: sycl-xmx-flash-attention

## Problem Statement

The XMX-accelerated MoE GEMM kernels currently reject coalesced memory layout (the branch default), forcing users to set `GGML_SYCL_LAYOUT_OVERRIDE=aos` to use XMX acceleration. This creates a performance trade-off: either use XMX with slower AoS memory access, or use coalesced layout without XMX.

**Current Coverage:**
| Layout | Q8_0 | MXFP4 |
|--------|------|-------|
| AoS | ✅ | ✅ |
| SoA | ✅ | ❌ |
| Coalesced | ❌ | ❌ |

**Target Coverage:** All 6 combinations supported.

## Design Goals

1. **Complete Coverage**: Support all layout × quantization combinations
2. **Native Unpacking**: No runtime conversion overhead - unpack directly in XMX kernels
3. **Hardware-Optimal Tiling**: Use XMXCapabilities for dynamic tile configuration
4. **Transparent Integration**: Work with existing optimization state tracking

## Architecture Overview

### Component Interaction
```
Model Load → detect_moe_expert_tensor() → convert_to_xmx_coalesced()
                                                    ↓
                                         Set XMX_COALESCED state
                                                    ↓
Inference → try_xmx_sorted_moe() → dispatch by (quant_type, layout)
                                                    ↓
                                   launch_xmx_moe_gemm_*_coalesced()
```

### New Kernels Required
- `launch_xmx_moe_gemm_q8_0_coalesced` - Q8_0 with coalesced layout
- `launch_xmx_moe_gemm_mxfp4_soa` - MXFP4 with SoA layout
- `launch_xmx_moe_gemm_mxfp4_coalesced` - MXFP4 with coalesced layout

## XMX Coalesced Layout Format

### Q8_0 XMX Coalesced (per K_TILE=32 columns)

**Standard Coalesced** (MMVQ-optimized, 8 words × 4 bytes):
```
Word 0: block0.qs[0:3], block1.qs[0:3], ..., blockN.qs[0:3]
Word 1: block0.qs[4:7], block1.qs[4:7], ..., blockN.qs[4:7]
...
Word 7: block0.qs[28:31], block1.qs[28:31], ..., blockN.qs[28:31]
Scales: d0, d1, d2, ..., dN (fp16, contiguous)
```

**XMX Coalesced** (K_TILE=32 aligned):
```
[32 int8 values from row 0, col 0-31]  // One K_TILE worth
[32 int8 values from row 1, col 0-31]
...
[32 int8 values from row M-1, col 0-31]
[Scales: d[0], d[1], ..., d[K/32-1]] per row, contiguous
```

This layout enables:
- Coalesced 128-byte loads (32 int8 × 4 rows = 128 bytes)
- Direct feeding to joint_matrix without shuffle
- Scale broadcast across K_TILE

### MXFP4 XMX Coalesced

**Per K_TILE=32 columns (16 bytes packed + 1 exponent per block):**
```
[16 packed bytes from row 0, col 0-31] [exp0]
[16 packed bytes from row 1, col 0-31] [exp1]
...
[Exponents grouped for broadcast]
```

## Kernel Unpacking Logic

### Warp-Coalesced Global Loads

Each warp (32 threads) cooperatively loads a tile:

```cpp
// Thread i loads 4 consecutive bytes from global memory
// Warp collectively loads 128 bytes (one cache line)
int8_t local_qs[4];
uint32_t* load_ptr = (uint32_t*)(weights + tile_offset + thread_id * 4);
*(uint32_t*)local_qs = *load_ptr;  // Coalesced 4-byte load

// Store to SLM for XMX consumption
slm_tile[thread_id * 4 + 0] = local_qs[0];
slm_tile[thread_id * 4 + 1] = local_qs[1];
slm_tile[thread_id * 4 + 2] = local_qs[2];
slm_tile[thread_id * 4 + 3] = local_qs[3];

item.barrier(sycl::access::fence_space::local_space);

// joint_matrix_load from SLM
joint_matrix_load(sg, B_tile, slm_tile, K_TILE);
```

### Scale Handling

Scales are loaded once per K_TILE and broadcast:
```cpp
// One thread loads scale, broadcasts via SLM or shuffle
sycl::half scale = scales[k_tile_idx];
// Apply after int8 accumulation: result = acc * scale_A * scale_B
```

## Load-Time Conversion

### MoE Expert Tensor Detection

In `ggml_backend_sycl_buffer_set_tensor`:

```cpp
bool is_moe_expert_tensor(const ggml_tensor* tensor) {
    // 3D tensor: [in_features, out_features, n_experts]
    if (tensor->n_dims != 3) return false;

    // Quantized type
    if (tensor->type != GGML_TYPE_Q8_0 &&
        tensor->type != GGML_TYPE_MXFP4) return false;

    // Name pattern (optional, for robustness)
    // e.g., "blk.*.ffn_gate_exps.weight"

    return true;
}
```

### Conversion Flow

```cpp
void convert_to_xmx_coalesced(ggml_tensor* tensor, sycl::queue& q) {
    auto& caps = XMXCapabilities::getInstance();
    if (!caps.isSupported()) return;

    const int K_TILE = caps.getKTile();  // 32 for int8
    const int M_TILE = caps.getMTile();  // 8
    const int N_TILE = caps.getNTile();  // 16

    size_t converted_size = calculate_xmx_coalesced_size(tensor, K_TILE);
    void* converted_data = sycl::malloc_device(converted_size, q);

    // Launch conversion kernel per expert
    int n_experts = tensor->ne[2];
    for (int e = 0; e < n_experts; e++) {
        launch_convert_to_xmx_coalesced_kernel(
            tensor->data, converted_data,
            tensor->ne[0], tensor->ne[1], e,
            K_TILE, tensor->type, q);
    }

    // Swap buffers
    sycl::free(tensor->data, q);
    tensor->data = converted_data;

    // Mark optimization state
    auto* extra = (ggml_tensor_extra_gpu*)tensor->extra;
    extra->optimized_feature.state = reorder_mode::XMX_COALESCED;
}
```

### Optimization State Tracking

Add new state to `reorder_mode` enum:
```cpp
enum class reorder_mode {
    NONE,           // Original AoS layout
    SOA,            // Structure-of-Arrays
    COALESCED,      // MMVQ-optimized coalesced
    XMX_COALESCED   // XMX-optimized coalesced (new)
};
```

## Dispatch Logic Update

In `try_xmx_sorted_moe`:

```cpp
// Get layout from optimization state
auto* extra = (ggml_tensor_extra_gpu*)expert_weights->extra;
reorder_mode layout = extra->optimized_feature.state;

// Dispatch based on quantization type and layout
if (expert_weights->type == GGML_TYPE_Q8_0) {
    switch (layout) {
        case reorder_mode::XMX_COALESCED:
            return launch_xmx_moe_gemm_q8_0_coalesced(...);
        case reorder_mode::SOA:
            return launch_xmx_moe_gemm_q8_0_soa(...);
        case reorder_mode::NONE:
            return launch_xmx_moe_gemm_q8_0(...);  // AoS
        case reorder_mode::COALESCED:
            // Standard MMVQ coalesced - convert or fallback
            GGML_SYCL_DEBUG("[XMX MoE] MMVQ coalesced not XMX-optimized\n");
            return false;
    }
} else if (expert_weights->type == GGML_TYPE_MXFP4) {
    // Similar dispatch for MXFP4
}
```

## Error Handling

### Fallback Chain
```
XMX Coalesced → XMX SoA → XMX AoS → Fused MoE → Standard DMMV
```

### Validation Points
1. **Hardware**: `XMXCapabilities::isSupported()`
2. **Quantization**: Weight tensor is Q8_0 or MXFP4
3. **Layout**: `extra->optimized_feature.state` matches expected
4. **Dimensions**: Expert dimensions compatible with XMX tiles

### Debug Logging
```cpp
GGML_SYCL_DEBUG("[XMX MoE] Using Q8_0 coalesced kernel for expert %d\n", e);
GGML_SYCL_DEBUG("[XMX MoE] Layout %d not supported, falling back\n", layout);
```

## Testing Strategy

### Unit Tests
1. Layout conversion correctness (coalesced → XMX coalesced)
2. Kernel output vs CPU reference
3. Dimension boundary cases (aligned/unaligned to XMX tiles)

### Integration Tests
```bash
# Default coalesced mode with XMX MoE
GGML_SYCL_XMX_MOE=1 ONEAPI_DEVICE_SELECTOR=level_zero:1 \
  ./build/bin/llama-completion \
  -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf -ngl 99 --flash-attn on \
  -p 'Count from 1 to 5:' -n 15 --seed 42 --temp 0
# Expected: "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15"
```

### Performance Validation
```bash
for mode in aos soa coalesced; do
  echo "=== Mode: $mode ==="
  GGML_SYCL_LAYOUT_OVERRIDE=$mode GGML_SYCL_XMX_MOE=1 \
    ./build/bin/llama-bench \
    -m /Storage/GenAI/models/gpt-oss-20b-Q8_0.gguf \
    -p 512 -n 128 -ngl 99 -fa 1
done
```

## Implementation Tasks

1. [ ] Add `XMX_COALESCED` to `reorder_mode` enum
2. [ ] Implement `launch_xmx_moe_gemm_q8_0_coalesced` kernel
3. [ ] Implement `launch_xmx_moe_gemm_mxfp4_soa` kernel
4. [ ] Implement `launch_xmx_moe_gemm_mxfp4_coalesced` kernel
5. [ ] Add `convert_to_xmx_coalesced` load-time conversion
6. [ ] Add `is_moe_expert_tensor` detection logic
7. [ ] Update `try_xmx_sorted_moe` dispatch for new layouts
8. [ ] Add unit tests for layout conversion
9. [ ] Add integration tests with MoE models
10. [ ] Benchmark all layout × quantization combinations

## References

- Existing kernels: `ggml/src/ggml-sycl/moe-xmx.hpp`
- Coalesced layout: `ggml/src/ggml-sycl/mmvq.cpp` lines 1741-1749
- XMX dispatch: `ggml/src/ggml-sycl/ggml-sycl.cpp` lines 11411-11692
- Optimization tracking: `ggml/src/ggml-sycl/common.hpp` (`optimize_feature` class)
