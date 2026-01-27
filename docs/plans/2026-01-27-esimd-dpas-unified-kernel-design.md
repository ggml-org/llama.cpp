# ESIMD dpas Integration for Unified Kernel

## Overview

Replace the current `joint_matrix` XMX path in the unified kernel with ESIMD dpas instructions for better hardware control and dual FP16/INT8 support.

**Epic:** `llama.cpp-a3t` (XMX-Optimized Unified Kernel Architecture)
**Related Task:** `llama.cpp-a3t.5` (XMX kernel optimization for high utilization)

## Problem Statement

The current unified kernel XMX path uses `joint_matrix` API which:
1. Provides limited control over memory access patterns
2. Only supports FP16 dpas (K-step=16)
3. Shows 27% regression vs scalar baseline (18.78 t/s vs 25.73 t/s)
4. VTune shows 92% XVE stall indicating poor utilization

## Solution

Implement ESIMD dpas with:
1. **Dual-path support**: FP16 (default) and INT8 (experimental) via compile/runtime switch
2. **Hardware-queried configuration**: No hardcoded tile sizes
3. **Direct dpas API**: `xmx::dpas<8, Repeat, TAcc, TAcc, TB, TA>(acc, b, a)`
4. **Double-buffering option**: Overlap memory loads with compute when SLM permits

## Architecture

### File Changes

| File | Change |
|------|--------|
| `unified-kernel.hpp` | Add `XMXConfig` struct, ESIMD headers, path selection helpers |
| `unified-kernel.cpp` | Add ESIMD kernel implementations, update `launch_unified_matmul` |
| `CMakeLists.txt` | Add `GGML_SYCL_XMX_INT8` option |

### Path Selection

```
launch_unified_matmul()
├── ESIMD path (args.use_xmx && xmx.supported)
│   ├── INT8 (GGML_SYCL_XMX_INT8=1 && xmx.supports_int8)
│   └── FP16 (default)
├── joint_matrix path (fallback, existing code)
└── Scalar path (no XMX support)
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GGML_SYCL_XMX_UNIFIED` | 0 | Enable XMX path in unified kernel |
| `GGML_SYCL_XMX_INT8` | 0 | Use INT8 dpas (experimental) |
| `GGML_SYCL_UNIFIED_DEBUG` | 0 | Enable debug logging |

## Hardware Configuration

### XMXConfig Struct

```cpp
struct XMXConfig {
    // Hardware-queried XMX dimensions
    size_t xmx_m;           // Output M per dpas (expected: 8)
    size_t xmx_n;           // Output N per dpas (expected: 16)
    size_t xmx_k;           // K-step per dpas (FP16: 16, INT8: 32)

    // Hardware resources
    size_t slm_size;        // SLM bytes per work-group
    int nsm;                // Compute units

    // Capability flags
    bool supports_int8;
    bool supports_fp16;

    static XMXConfig from_device(int device_id);
};
```

Configuration is queried from `ggml_sycl_info().devices[device_id].xmx_caps` at runtime, following the pattern established in `moe-xmx-fused.hpp`.

## ESIMD dpas API

### Template Parameters

```cpp
xmx::dpas<SystolicDepth, Repeat, AccType, AccType, BType, AType>(acc, b, a)
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| SystolicDepth | 8 | Fixed for Intel XMX |
| Repeat | 1-8 | Output rows (M dimension) |
| AccType | float/int32 | Accumulator type |
| BType | half/int8 | B matrix (weights) element type |
| AType | half/int8 | A matrix (activations) element type |

### Vector Sizes

| Path | K-step | A vector | B vector | Output |
|------|--------|----------|----------|--------|
| FP16 | 16 | Repeat×16 | 16×16=256 | Repeat×16 |
| INT8 | 32 | Repeat×32 | 32×16=512 | Repeat×16 |

Note: INT8 K-step (32) aligns perfectly with Q4_0 block size (32 weights per block).

## Kernel Implementation

### Work Distribution

```
Grid: ceil(M/TILE_M) × ceil(N/TILE_N) work-items
Work-group: 1 work-item (ESIMD single-threaded model)
Each work-item: computes one [TILE_M × TILE_N] output tile
```

TILE_M and TILE_N are queried from hardware (expected: 8×16).

### FP16 Path

```cpp
template <int TILE_M, int TILE_N>
SYCL_ESIMD_FUNCTION void esimd_matmul_fp16_impl(
    const block_q4_0_unified* weights,
    const float* activations,
    float* output,
    int64_t M, int64_t N, int64_t K,
    int64_t m_start, int64_t n_start,
    bool use_double_buffer)
{
    constexpr int K_TILE = 16;  // FP16 dpas K-step

    simd<float, TILE_M * TILE_N> acc = 0.0f;

    for (int64_t kt = 0; kt < K / K_TILE; kt++) {
        // Load and dequantize Q4_0 weights → FP16
        simd<sycl::half, TILE_N * K_TILE> b_vec;
        load_dequant_weights_fp16(b_vec, weights, ...);

        // Load activations FP32 → FP16
        simd<sycl::half, TILE_M * K_TILE> a_vec;
        load_activations_fp16(a_vec, activations, ...);

        // dpas: FP16 × FP16 → FP32
        acc = xmx::dpas<8, TILE_M, float, float, sycl::half, sycl::half>(
            acc, b_vec, a_vec);
    }

    // Store output with boundary checking
    store_output(acc, output, m_start, n_start, M, N);
}
```

### INT8 Path

```cpp
template <int TILE_M, int TILE_N>
SYCL_ESIMD_FUNCTION void esimd_matmul_int8_impl(
    const block_q4_0_unified* weights,
    const float* activations,
    float* output,
    int64_t M, int64_t N, int64_t K,
    int64_t m_start, int64_t n_start,
    bool use_double_buffer)
{
    constexpr int K_TILE = 32;  // INT8 dpas K-step = Q4_0 block size

    simd<float, TILE_M * TILE_N> acc_fp = 0.0f;

    for (int64_t kt = 0; kt < K / K_TILE; kt++) {
        // Unpack Q4_0 → INT8 (no scale applied)
        simd<int8_t, TILE_N * K_TILE> b_vec;
        simd<sycl::half, TILE_N> weight_scales;
        unpack_q4_0_to_int8(b_vec, weight_scales, weights, ...);

        // Quantize activations FP32 → INT8 with per-row scale
        simd<int8_t, TILE_M * K_TILE> a_vec;
        simd<float, TILE_M> act_scales;
        quantize_activations_int8(a_vec, act_scales, activations, ...);

        // dpas: INT8 × INT8 → INT32
        simd<int32_t, TILE_M * TILE_N> acc_i32;
        acc_i32 = xmx::dpas<8, TILE_M, int32_t, int32_t, int8_t, int8_t>(
            simd<int32_t, TILE_M * TILE_N>(0), b_vec, a_vec);

        // Apply scales: acc_fp += acc_i32 * weight_scale * act_scale
        apply_scales(acc_fp, acc_i32, weight_scales, act_scales);
    }

    store_output(acc_fp, output, m_start, n_start, M, N);
}
```

### Q4_0 to INT8 Conversion

Q4_0 block: 16 bytes containing 32 4-bit weights (0-15) + FP16 scale.

```cpp
// Unpack 16 bytes → 32 INT8 values
for (int i = 0; i < 16; i++) {
    int8_t lo = (qs[i] & 0x0F) - 8;   // Low nibble: 0-15 → -8 to +7
    int8_t hi = (qs[i] >> 4) - 8;      // High nibble: 0-15 → -8 to +7
    weights_i8[i] = lo;
    weights_i8[i + 16] = hi;
}
// Scale stored separately, applied after dpas
```

### Activation Quantization (INT8 path only)

Per-row dynamic quantization:

```cpp
// Find max abs value in row's K-tile
float max_abs = reduce_max_abs(activations, m_row, k_start, K_TILE);

// Compute scale
float scale = max_abs / 127.0f;
float scale_inv = (scale > 0) ? (1.0f / scale) : 0.0f;

// Quantize each element
int8_t q = clamp(round(val * scale_inv), -127, 127);
```

## SLM Layout

```
Per K-tile (single buffer):
┌─────────────────────────────────────────┐
│ Weights: [TILE_N × K_TILE] elements     │
├─────────────────────────────────────────┤
│ Activations: [TILE_M × K_TILE] elements │
├─────────────────────────────────────────┤
│ Scales (INT8 only): [TILE_N + TILE_M]   │
└─────────────────────────────────────────┘

FP16 path (TILE_M=8, TILE_N=16, K_TILE=16):
- Weights: 16 × 16 × 2 = 512 bytes
- Activations: 8 × 16 × 2 = 256 bytes
- Total: 768 bytes

INT8 path (TILE_M=8, TILE_N=16, K_TILE=32):
- Weights: 16 × 32 × 1 = 512 bytes
- Activations: 8 × 32 × 1 = 256 bytes
- Scales: (16 + 8) × 4 = 96 bytes
- Total: 864 bytes

Double-buffer: 2× above sizes
Decision: use_double_buffer = (2 * buffer_size) < (slm_size / 2)
```

## Boundary Handling

Non-aligned dimensions handled with per-element bounds checking:

```cpp
// Load with bounds check
if (n_global < N && k_off < k_len) {
    w = dequant_q4_0_half(&weights[block_idx], idx_in_block);
} else {
    w = 0.0f;  // Zero-pad out-of-bounds
}

// Store with bounds check
if (m_global < M && n_global < N) {
    output[m_global * N + n_global] = acc[m_off * TILE_N + n_off];
}
```

## Testing

### Correctness Test

```bash
# Compare FP16 ESIMD vs scalar baseline
GGML_SYCL_XMX_UNIFIED=1 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0 -ngl 99 > esimd_fp16.txt

GGML_SYCL_XMX_UNIFIED=0 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0 -ngl 99 > scalar.txt

diff esimd_fp16.txt scalar.txt  # Must be identical

# Compare INT8 ESIMD vs scalar baseline
GGML_SYCL_XMX_UNIFIED=1 GGML_SYCL_XMX_INT8=1 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0 -ngl 99 > esimd_int8.txt

diff esimd_int8.txt scalar.txt  # Must be identical
```

### Performance Benchmark

```bash
# Baseline (scalar)
GGML_SYCL_XMX_UNIFIED=0 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128 -ngl 99

# FP16 ESIMD
GGML_SYCL_XMX_UNIFIED=1 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128 -ngl 99

# INT8 ESIMD
GGML_SYCL_XMX_UNIFIED=1 GGML_SYCL_XMX_INT8=1 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128 -ngl 99
```

### Batch Size Sweep

```bash
for pp in 32 128 512 2048; do
  echo "=== PP${pp} Scalar ==="
  GGML_SYCL_XMX_UNIFIED=0 ./build/bin/llama-bench -m model.gguf -p $pp -n 32 -ngl 99

  echo "=== PP${pp} FP16 ESIMD ==="
  GGML_SYCL_XMX_UNIFIED=1 ./build/bin/llama-bench -m model.gguf -p $pp -n 32 -ngl 99

  echo "=== PP${pp} INT8 ESIMD ==="
  GGML_SYCL_XMX_UNIFIED=1 GGML_SYCL_XMX_INT8=1 ./build/bin/llama-bench -m model.gguf -p $pp -n 32 -ngl 99
done
```

## Success Criteria

1. **Correctness**: Bit-exact output vs scalar baseline for both FP16 and INT8 paths
2. **Performance**: PP512 >= 27.70 t/s (current scalar baseline)
3. **No regression**: TG (token generation) performance maintained
4. **Hardware utilization**: VTune XVE stall < 50% (from current 92%)

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| INT8 quantization overhead exceeds benefit | FP16 path is default; INT8 is opt-in |
| Non-aligned dimensions cause correctness issues | Extensive boundary checking, TDD approach |
| ESIMD compilation issues | Fallback to existing joint_matrix path |
| Hardware query returns unexpected values | Fallback defaults in XMXConfig |

## Implementation Phases (Detailed)

### Phase 1: XMXConfig Struct and Hardware Query Integration

**Goal**: Establish hardware-queried configuration infrastructure

**Files to modify**:
- `ggml/src/ggml-sycl/unified-kernel.hpp`

**Implementation details**:

```cpp
// Add to unified-kernel.hpp after existing includes

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

// Runtime environment variable checks
inline bool use_esimd_dpas() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("GGML_SYCL_XMX_ESIMD");
        enabled = (env && std::string(env) == "1") ? 1 : 0;
    }
    return enabled != 0;
}

inline bool use_int8_dpas() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("GGML_SYCL_XMX_INT8");
        enabled = (env && std::string(env) == "1") ? 1 : 0;
    }
    return enabled != 0;
}

struct XMXConfig {
    // Hardware-queried XMX tile dimensions
    // Intel ESIMD dpas: SystolicDepth=8 (fixed), Repeat=1-8, ExecutionSize=16 (Arc)
    size_t xmx_m = 8;       // RepeatCount determines M (1-8, we use 8)
    size_t xmx_n = 16;      // ExecutionSize: 8 for DG2, 16 for PVC/Arc
    size_t xmx_k_fp16 = 16; // K for FP16: SystolicDepth(8) × OpsPerChannel(2) = 16
    size_t xmx_k_int8 = 32; // K for INT8: SystolicDepth(8) × OpsPerChannel(4) = 32

    // Hardware resources
    size_t slm_size = 65536;  // SLM bytes per work-group (queried)
    int nsm = 20;             // Compute units (queried)

    // Capability flags from xmx_caps
    bool supported = false;
    bool supports_int8 = false;
    bool supports_fp16 = false;

    // Derived configuration
    bool use_double_buffer = false;
    int tiles_per_workitem = 1;

    // Query hardware and populate config
    static XMXConfig from_device(int device_id) {
        XMXConfig cfg;

        if (device_id < 0 || device_id >= ggml_sycl_info().device_count) {
            return cfg;  // Return defaults
        }

        const auto& dev = ggml_sycl_info().devices[device_id];
        const auto& xmx = dev.xmx_caps;

        cfg.supported = xmx.supported;
        cfg.supports_int8 = xmx.supports_int8;
        cfg.supports_fp16 = xmx.supports_fp16;

        // Use queried values with fallback defaults
        // M dimension: from xmx.M or default 8
        cfg.xmx_m = xmx.M > 0 ? xmx.M : 8;
        // N dimension: from xmx.N or default 16
        cfg.xmx_n = xmx.N > 0 ? xmx.N : 16;
        // K dimension: from xmx.K (type-dependent) or defaults
        // Note: xmx.K is queried for INT8 (=32), FP16 uses K=16
        cfg.xmx_k_int8 = xmx.K > 0 ? xmx.K : 32;
        cfg.xmx_k_fp16 = 16;  // FP16 K is always 16 per Intel spec

        cfg.slm_size = xmx.slm_size > 0 ? xmx.slm_size : 65536;
        cfg.nsm = dev.nsm;

        // Compute double-buffer feasibility
        // FP16: weights(16×16×2) + activations(8×16×2) = 768 bytes
        // INT8: weights(16×32×1) + activations(8×32×1) + scales = 864 bytes
        size_t single_buffer = 864;  // Use larger INT8 size
        cfg.use_double_buffer = (2 * single_buffer) < (cfg.slm_size / 2);

        return cfg;
    }
};
```

**Verification**:
```bash
# Build and verify hardware query works
ninja -C build && GGML_SYCL_DEBUG=1 ./build/bin/llama-bench \
  -m model.gguf -p 32 -n 1 -ngl 99 2>&1 | grep -i xmx
# Should show: [XMX] int8: M=8, N=16, K=32
```

---

### Phase 2: FP16 ESIMD dpas Path

**Goal**: Replace joint_matrix with ESIMD dpas for FP16, achieving baseline correctness

**Files to modify**:
- `ggml/src/ggml-sycl/unified-kernel.cpp`

**Intel ESIMD dpas API** (from official documentation):
```cpp
// Signature: xmx::dpas<SystolicDepth, RepeatCount, T, CT, BT, AT>(C, B, A)
// - SystolicDepth: Always 8 for Intel XMX
// - RepeatCount: 1-8, determines output rows
// - T: Result type
// - CT: Accumulator input type (same as T)
// - BT: B matrix element type (weights)
// - AT: A matrix element type (activations)
//
// For FP16: dpas<8, 8, float, float, half, half>(acc, B, A)
// - Output: 8×16 (RepeatCount × ExecutionSize)
// - K consumed: 16 (8 × 2 ops/channel for FP16)
// - A size: 8×16 = 128 half elements
// - B size: 16×16 = 256 half elements
```

**Implementation details**:

```cpp
// Add new ESIMD kernel function in unified-kernel.cpp

#if defined(SYCL_EXT_INTEL_ESIMD)

template <int TILE_M, int TILE_N>
SYCL_ESIMD_FUNCTION void esimd_matmul_fp16_kernel(
    const block_q4_0_unified* __restrict weights,
    const float* __restrict activations,
    float* __restrict output,
    int64_t M, int64_t N, int64_t K,
    int64_t m_start, int64_t n_start,
    int64_t k_blocks_per_row)
{
    using namespace sycl::ext::intel::esimd;

    // FP16 dpas constants
    constexpr int K_TILE = 16;      // FP16 K-step
    constexpr int EXEC_N = 16;      // Execution size for Arc
    constexpr int REPEAT = TILE_M;  // 8

    // Vector sizes per dpas call
    constexpr int A_SIZE = REPEAT * K_TILE;   // 8×16 = 128
    constexpr int B_SIZE = K_TILE * EXEC_N;   // 16×16 = 256
    constexpr int C_SIZE = REPEAT * EXEC_N;   // 8×16 = 128

    // Initialize accumulator
    simd<float, C_SIZE> acc = 0.0f;

    const int64_t k_tiles = K / K_TILE;

    // K-loop: process K_TILE elements per iteration
    for (int64_t kt = 0; kt < k_tiles; kt++) {
        const int64_t k_start = kt * K_TILE;

        // Load weights: dequantize Q4_0 → FP16
        // Layout: B[k, n] col-major for dpas (transpose of weight[n, k])
        simd<sycl::half, B_SIZE> b_vec;

        #pragma unroll
        for (int n_off = 0; n_off < TILE_N; n_off++) {
            const int64_t n_global = n_start + n_off;

            #pragma unroll
            for (int k_off = 0; k_off < K_TILE; k_off++) {
                sycl::half w = sycl::half(0.0f);

                if (n_global < N) {
                    const int64_t k_global = k_start + k_off;
                    const int64_t block_idx = n_global * k_blocks_per_row + k_global / 32;
                    const int idx_in_block = static_cast<int>(k_global % 32);
                    w = dequant_q4_0_half(&weights[block_idx], idx_in_block);
                }
                // Col-major: B[k, n] = b_vec[k * EXEC_N + n]
                b_vec[k_off * EXEC_N + n_off] = w;
            }
        }

        // Load activations: FP32 → FP16
        // Layout: A[m, k] row-major
        simd<sycl::half, A_SIZE> a_vec;

        #pragma unroll
        for (int m_off = 0; m_off < TILE_M; m_off++) {
            const int64_t m_global = m_start + m_off;

            #pragma unroll
            for (int k_off = 0; k_off < K_TILE; k_off++) {
                sycl::half a = sycl::half(0.0f);

                if (m_global < M) {
                    const int64_t k_global = k_start + k_off;
                    a = static_cast<sycl::half>(activations[m_global * K + k_global]);
                }
                // Row-major: A[m, k] = a_vec[m * K_TILE + k]
                a_vec[m_off * K_TILE + k_off] = a;
            }
        }

        // ESIMD dpas: acc = A × B + acc
        // dpas<SystolicDepth=8, Repeat=8, float, float, half, half>
        acc = xmx::dpas<8, REPEAT, float, float, sycl::half, sycl::half>(
            acc, b_vec, a_vec);
    }

    // Store output with boundary checking
    #pragma unroll
    for (int m_off = 0; m_off < TILE_M; m_off++) {
        const int64_t m_global = m_start + m_off;
        if (m_global >= M) continue;

        #pragma unroll
        for (int n_off = 0; n_off < TILE_N; n_off++) {
            const int64_t n_global = n_start + n_off;
            if (n_global < N) {
                output[m_global * N + n_global] = acc[m_off * EXEC_N + n_off];
            }
        }
    }
}

#endif // SYCL_EXT_INTEL_ESIMD
```

**Kernel launch code**:
```cpp
// Add to launch_unified_matmul() in unified-kernel.cpp

#if defined(SYCL_EXT_INTEL_ESIMD)
if (use_esimd_dpas() && cfg.supported && cfg.supports_fp16) {
    constexpr int TILE_M = 8;
    constexpr int TILE_N = 16;

    const int64_t grid_m = (args.M + TILE_M - 1) / TILE_M;
    const int64_t grid_n = (args.N + TILE_N - 1) / TILE_N;
    const int64_t total_tiles = grid_m * grid_n;
    const int64_t k_blocks_per_row = args.K / 32;

    const block_q4_0_unified* weights =
        static_cast<const block_q4_0_unified*>(args.weights);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<class esimd_fp16_kernel>(
            sycl::nd_range<1>(sycl::range<1>(total_tiles), sycl::range<1>(1)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t tile_idx = item.get_global_id(0);
                const int64_t tile_m = tile_idx / grid_n;
                const int64_t tile_n = tile_idx % grid_n;
                const int64_t m_start = tile_m * TILE_M;
                const int64_t n_start = tile_n * TILE_N;

                esimd_matmul_fp16_kernel<TILE_M, TILE_N>(
                    weights, args.activations, args.output,
                    args.M, args.N, args.K,
                    m_start, n_start, k_blocks_per_row);
            });
    });
    return;
}
#endif
```

**Verification**:
```bash
# Test correctness
GGML_SYCL_XMX_ESIMD=1 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0 -ngl 99 > esimd.txt

./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0 -ngl 99 > baseline.txt

diff esimd.txt baseline.txt  # Must be identical

# Benchmark
GGML_SYCL_XMX_ESIMD=1 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128 -ngl 99
```

---

### Phase 3: INT8 ESIMD dpas Path

**Goal**: Add INT8 dpas path for 2× theoretical throughput

**Intel ESIMD dpas API for INT8**:
```cpp
// For INT8: dpas<8, 8, int32_t, int32_t, int8_t, int8_t>(acc, B, A)
// - Output: 8×16 INT32 accumulator
// - K consumed: 32 (8 × 4 ops/channel for INT8)
// - A size: 8×32 = 256 int8 elements
// - B size: 32×16 = 512 int8 elements
```

**Implementation details**:

```cpp
template <int TILE_M, int TILE_N>
SYCL_ESIMD_FUNCTION void esimd_matmul_int8_kernel(
    const block_q4_0_unified* __restrict weights,
    const float* __restrict activations,
    float* __restrict output,
    int64_t M, int64_t N, int64_t K,
    int64_t m_start, int64_t n_start,
    int64_t k_blocks_per_row)
{
    using namespace sycl::ext::intel::esimd;

    // INT8 dpas constants - K=32 matches Q4_0 block size exactly!
    constexpr int K_TILE = 32;
    constexpr int EXEC_N = 16;
    constexpr int REPEAT = TILE_M;

    constexpr int A_SIZE = REPEAT * K_TILE;   // 8×32 = 256
    constexpr int B_SIZE = K_TILE * EXEC_N;   // 32×16 = 512
    constexpr int C_SIZE = REPEAT * EXEC_N;   // 8×16 = 128

    // FP32 accumulator for scaled results
    simd<float, C_SIZE> acc_fp = 0.0f;

    const int64_t k_tiles = K / K_TILE;  // Each K-tile = one Q4_0 block per N

    // K-loop: each iteration processes exactly one Q4_0 block
    for (int64_t kt = 0; kt < k_tiles; kt++) {
        // Weight scales for this K-block (one per N column)
        simd<float, TILE_N> weight_scales;
        simd<int8_t, B_SIZE> b_vec;

        // Unpack Q4_0 → INT8 (no scale multiplication yet)
        #pragma unroll
        for (int n_off = 0; n_off < TILE_N; n_off++) {
            const int64_t n_global = n_start + n_off;

            if (n_global < N) {
                const int64_t block_idx = n_global * k_blocks_per_row + kt;
                const block_q4_0_unified* blk = &weights[block_idx];

                // Store scale for later
                weight_scales[n_off] = static_cast<float>(blk->d);

                // Unpack 16 bytes → 32 INT8 values
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int8_t lo = static_cast<int8_t>((blk->qs[i] & 0x0F) - 8);
                    int8_t hi = static_cast<int8_t>((blk->qs[i] >> 4) - 8);
                    b_vec[i * EXEC_N + n_off] = lo;
                    b_vec[(i + 16) * EXEC_N + n_off] = hi;
                }
            } else {
                weight_scales[n_off] = 0.0f;
                #pragma unroll
                for (int k = 0; k < K_TILE; k++) {
                    b_vec[k * EXEC_N + n_off] = 0;
                }
            }
        }

        // Activation scales (one per M row)
        simd<float, TILE_M> act_scales;
        simd<int8_t, A_SIZE> a_vec;

        // Quantize activations FP32 → INT8
        #pragma unroll
        for (int m_off = 0; m_off < TILE_M; m_off++) {
            const int64_t m_global = m_start + m_off;

            if (m_global < M) {
                // Find max abs in this row's K-tile
                float max_abs = 0.0f;
                #pragma unroll
                for (int k = 0; k < K_TILE; k++) {
                    float val = activations[m_global * K + kt * K_TILE + k];
                    max_abs = sycl::fmax(max_abs, sycl::fabs(val));
                }

                float scale = max_abs / 127.0f;
                float scale_inv = (scale > 1e-10f) ? (1.0f / scale) : 0.0f;
                act_scales[m_off] = scale;

                // Quantize
                #pragma unroll
                for (int k = 0; k < K_TILE; k++) {
                    float val = activations[m_global * K + kt * K_TILE + k];
                    int8_t q = static_cast<int8_t>(
                        sycl::clamp(sycl::round(val * scale_inv), -127.0f, 127.0f));
                    a_vec[m_off * K_TILE + k] = q;
                }
            } else {
                act_scales[m_off] = 0.0f;
                #pragma unroll
                for (int k = 0; k < K_TILE; k++) {
                    a_vec[m_off * K_TILE + k] = 0;
                }
            }
        }

        // ESIMD dpas: INT8 × INT8 → INT32
        simd<int32_t, C_SIZE> acc_i32 =
            xmx::dpas<8, REPEAT, int32_t, int32_t, int8_t, int8_t>(
                simd<int32_t, C_SIZE>(0), b_vec, a_vec);

        // Apply scales: acc_fp += acc_i32 * weight_scale * act_scale
        #pragma unroll
        for (int m_off = 0; m_off < TILE_M; m_off++) {
            float a_scale = act_scales[m_off];
            #pragma unroll
            for (int n_off = 0; n_off < TILE_N; n_off++) {
                float w_scale = weight_scales[n_off];
                int32_t i32_val = acc_i32[m_off * EXEC_N + n_off];
                acc_fp[m_off * EXEC_N + n_off] +=
                    static_cast<float>(i32_val) * w_scale * a_scale;
            }
        }
    }

    // Store output
    #pragma unroll
    for (int m_off = 0; m_off < TILE_M; m_off++) {
        const int64_t m_global = m_start + m_off;
        if (m_global >= M) continue;

        #pragma unroll
        for (int n_off = 0; n_off < TILE_N; n_off++) {
            const int64_t n_global = n_start + n_off;
            if (n_global < N) {
                output[m_global * N + n_global] = acc_fp[m_off * EXEC_N + n_off];
            }
        }
    }
}
```

**Verification**:
```bash
# INT8 correctness test
GGML_SYCL_XMX_ESIMD=1 GGML_SYCL_XMX_INT8=1 ./build/bin/llama-completion \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0 -ngl 99 > int8.txt

diff int8.txt baseline.txt  # Should be identical or very close

# Benchmark comparison
for path in "FP16" "INT8"; do
  if [ "$path" == "FP16" ]; then
    export GGML_SYCL_XMX_INT8=0
  else
    export GGML_SYCL_XMX_INT8=1
  fi
  echo "=== $path ==="
  GGML_SYCL_XMX_ESIMD=1 ./build/bin/llama-bench \
    -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128 -ngl 99
done
```

---

### Phase 4: Double-Buffering Optimization

**Goal**: Overlap memory loads with compute to hide latency

**Reference**: `fattn-xmx-f16.hpp` double-buffering pattern

**Implementation details**:

```cpp
// Add SLM management for double-buffering
template <int TILE_M, int TILE_N, int K_TILE>
SYCL_ESIMD_FUNCTION void esimd_matmul_double_buffered(
    /* params */)
{
    using namespace sycl::ext::intel::esimd;

    // SLM layout for double-buffering
    constexpr size_t WEIGHTS_SIZE = TILE_N * K_TILE * sizeof(sycl::half);
    constexpr size_t ACTS_SIZE = TILE_M * K_TILE * sizeof(sycl::half);
    constexpr size_t BUFFER_SIZE = WEIGHTS_SIZE + ACTS_SIZE;
    constexpr size_t SLM_TOTAL = 2 * BUFFER_SIZE;  // Two buffers

    slm_init<SLM_TOTAL>();

    // Buffer offsets
    constexpr uint32_t BUF0_WEIGHTS = 0;
    constexpr uint32_t BUF0_ACTS = WEIGHTS_SIZE;
    constexpr uint32_t BUF1_WEIGHTS = BUFFER_SIZE;
    constexpr uint32_t BUF1_ACTS = BUFFER_SIZE + WEIGHTS_SIZE;

    simd<float, TILE_M * TILE_N> acc = 0.0f;

    // Pre-load first K-tile into buffer 0
    load_tile_to_slm(/* K-tile 0 → BUF0 */);

    int buf_compute = 0;

    for (int64_t kt = 0; kt < k_tiles; kt++) {
        // Determine buffer offsets
        uint32_t compute_w_off = (buf_compute == 0) ? BUF0_WEIGHTS : BUF1_WEIGHTS;
        uint32_t compute_a_off = (buf_compute == 0) ? BUF0_ACTS : BUF1_ACTS;
        uint32_t load_w_off = (buf_compute == 0) ? BUF1_WEIGHTS : BUF0_WEIGHTS;
        uint32_t load_a_off = (buf_compute == 0) ? BUF1_ACTS : BUF0_ACTS;

        // Load from SLM to registers
        simd<sycl::half, TILE_N * K_TILE> b_vec = slm_block_load<sycl::half, TILE_N * K_TILE>(compute_w_off);
        simd<sycl::half, TILE_M * K_TILE> a_vec = slm_block_load<sycl::half, TILE_M * K_TILE>(compute_a_off);

        // Start loading next K-tile into other buffer (async)
        if (kt + 1 < k_tiles) {
            load_tile_to_slm(/* K-tile kt+1 → load_*_off */);
        }

        // Compute dpas
        acc = xmx::dpas<8, TILE_M, float, float, sycl::half, sycl::half>(acc, b_vec, a_vec);

        // Swap buffers
        buf_compute = 1 - buf_compute;
    }

    // Store output
    store_output(acc, output, m_start, n_start, M, N);
}
```

**SLM helper functions using ESIMD primitives**:
```cpp
// Block load with cache hints
template <typename T, int N>
SYCL_ESIMD_FUNCTION simd<T, N> slm_load_streaming(uint32_t offset) {
    return slm_block_load<T, N>(offset);
}

// Block store
template <typename T, int N>
SYCL_ESIMD_FUNCTION void slm_store(uint32_t offset, simd<T, N> data) {
    slm_block_store<T, N>(offset, data);
}

// Global load with prefetch
template <typename T, int N>
SYCL_ESIMD_FUNCTION simd<T, N> global_load_prefetch(const T* ptr, const T* prefetch_ptr) {
    using namespace sycl::ext::intel::esimd;

    // Issue prefetch for next iteration
    if (prefetch_ptr) {
        lsc_prefetch<T, 1, lsc_data_size::default_size, cache_hint::streaming, cache_hint::cached>(
            prefetch_ptr);
    }

    // Load current data with streaming hint
    return lsc_block_load<T, N>(ptr, properties{cache_hint_L1<cache_hint::streaming>});
}
```

---

### Phase 5: Performance Tuning and VTune Analysis

**Goal**: Achieve PP512 >= 27.70 t/s, XVE stall < 50%

**Tuning parameters**:

1. **Register pressure**: Use `[[intel::num_registers(128)]]` or `[[intel::num_registers(256)]]`
2. **Tile sizes**: Experiment with TILE_M=4,8 and larger TILE_N
3. **Prefetch distance**: Tune based on memory latency
4. **Work distribution**: Persistent threads vs one-tile-per-workitem

**VTune analysis commands**:
```bash
# GPU hotspots analysis
source /opt/intel/oneapi/setvars.sh
vtune -collect gpu-hotspots -- \
  env GGML_SYCL_XMX_ESIMD=1 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128 -ngl 99

# GPU compute extended (for XVE stall analysis)
vtune -collect gpu-compute-extended -- \
  env GGML_SYCL_XMX_ESIMD=1 ./build/bin/llama-bench \
  -m /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf -p 512 -n 128 -ngl 99
```

**Key metrics to check**:
- XVE Active vs XVE Stall percentage
- Memory bandwidth utilization
- EU Array Active percentage
- XMX utilization (if available)

**Optimization iterations**:
1. Baseline ESIMD vs joint_matrix comparison
2. Add SLM caching → measure improvement
3. Add double-buffering → measure improvement
4. Tune tile sizes → find optimal configuration
5. Add LSC cache hints → measure improvement

## References

- [Intel XMX Programming Guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/programming-intel-xmx-using-sycl-joint-matrix.html)
- [ESIMD Extension Specification](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_esimd/sycl_ext_intel_esimd.md)
- [Intel XMX Performance Guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/boost-matrix-multiplication-performance-with-intel.html)

- Working ESIMD dpas benchmark: `tools/sycl-kernel-bench/kernels/dpas_exploration/dpas_common.hpp`
- Hardware query pattern: `ggml/src/ggml-sycl/moe-xmx-fused.hpp`
- XMX capabilities: `ggml/src/ggml-sycl/common.cpp:query_xmx_capabilities()`
- Current unified kernel: `ggml/src/ggml-sycl/unified-kernel.cpp`
