//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Unified Kernel Implementation for SYCL Matmul
//
// This file implements the unified matmul kernel supporting:
// - Q4_0 quantization (scalar and XMX paths)
// - Tile-based computation with SLM staging
// - Boundary handling for non-aligned dimensions
// - XMX dpas acceleration via Intel joint_matrix extensions
//
// XMX Path:
// - Enabled when args.use_xmx=true and hardware supports it
// - Uses 8x16x8 tile dimensions for half precision dpas
// - Falls back to scalar path when XMX is unavailable or dimensions don't fit
//

#include "unified-kernel.hpp"

#include <array>
#include <chrono>

// =============================================================================
// Standalone Test Mode Support
// =============================================================================
// When UNIFIED_KERNEL_TEST_STANDALONE is defined, provide stub implementations
// for symbols normally provided by common.cpp. This allows standalone testing
// without linking the entire ggml-sycl library.
#ifdef UNIFIED_KERNEL_TEST_STANDALONE

// Stub global debug flags
bool g_ggml_sycl_debug = false;
bool g_ggml_sycl_debug_forced_off = false;

// Stub device info structure (minimal for testing)
struct ggml_sycl_device_info_stub {
    int nsm = 20;
    const char * device_name = "Intel(R) Arc(TM) B580 Graphics";
    struct {
        bool supported = true;
        bool supports_int8 = true;
        bool supports_fp16 = true;
        int slm_size = 65536;
        int max_wg_size = 1024;
        int sg_size_preferred = 16;
        int M = 8;   // XMX tile M dimension
        int N = 16;  // XMX tile N dimension
        int K = 32;  // XMX tile K dimension (for INT8)
    } xmx_caps;
};

struct ggml_sycl_info_stub {
    int device_count = 1;
    ggml_sycl_device_info_stub devices[1] = {};
};

static ggml_sycl_info_stub g_stub_sycl_info;

const ggml_sycl_info_stub & ggml_sycl_info() {
    return g_stub_sycl_info;
}

// Stub GGML_SYCL_DEBUG macro (no-op in standalone mode)
#define GGML_SYCL_DEBUG(...) do {} while(0)

// Stub GGML_LOG macros (normally from ggml-impl.h)
#define GGML_LOG_ERROR(...) fprintf(stderr, __VA_ARGS__)
#define GGML_LOG_WARN(...)  fprintf(stderr, __VA_ARGS__)

#else  // !UNIFIED_KERNEL_TEST_STANDALONE

#include "common.hpp"  // For ggml_sycl_info() and GGML_SYCL_DEBUG

#endif  // UNIFIED_KERNEL_TEST_STANDALONE

#include <algorithm>
#include <cstdlib>
#include <cstdio>

namespace ggml_sycl_unified {

// =============================================================================
// GPU Family Detection Helper
// =============================================================================
// Case-insensitive substring search for device name matching

static bool name_contains(const char * name, const char * substr) {
    if (!name || !substr) return false;

    // Convert both to lowercase and search
    std::string lower_name   = name;
    std::string lower_substr = substr;
    for (char & c : lower_name) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    for (char & c : lower_substr) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return lower_name.find(lower_substr) != std::string::npos;
}

// GPU family enumeration for hardware capability detection
enum class GPUFamily {
    UNKNOWN,
    ARC_ALCHEMIST,   // A-series (A770, A750, A580, A380, A310)
    ARC_BATTLEMAGE,  // B-series (B580, B570)
    DATA_CENTER_MAX, // PVC (Ponte Vecchio)
    DATA_CENTER_FLEX // Arctic Sound (Flex series)
};

// Detect GPU family from device name
static GPUFamily detect_gpu_family_from_name(const char * name) {
    if (!name) return GPUFamily::UNKNOWN;

    // Arc Battlemage (B-series): B580, B570, etc.
    if (name_contains(name, "B580") || name_contains(name, "B570") || name_contains(name, "B50") ||
        (name_contains(name, "Arc") && name_contains(name, "Battlemage"))) {
        return GPUFamily::ARC_BATTLEMAGE;
    }

    // Arc Alchemist (A-series): A770, A750, A580, A380, A310, etc.
    if (name_contains(name, "A770") || name_contains(name, "A750") || name_contains(name, "A580") ||
        name_contains(name, "A380") || name_contains(name, "A310") ||
        (name_contains(name, "Arc") && name_contains(name, "Graphics"))) {
        return GPUFamily::ARC_ALCHEMIST;
    }

    // Data Center GPU Max (PVC/Ponte Vecchio)
    if (name_contains(name, "Max") || name_contains(name, "PVC") || name_contains(name, "Ponte")) {
        return GPUFamily::DATA_CENTER_MAX;
    }

    // Data Center GPU Flex (Arctic Sound)
    if (name_contains(name, "Flex") || name_contains(name, "Arctic")) {
        return GPUFamily::DATA_CENTER_FLEX;
    }

    return GPUFamily::UNKNOWN;
}

// Determine max ESIMD work-group size from GPU family
// ESIMD has stricter limits than regular SYCL kernels:
// - Arc (Alchemist/Battlemage): max 64 work-items
// - PVC (Ponte Vecchio/Data Center Max): up to 1024 work-items
static int get_max_esimd_workgroup(GPUFamily family) {
    switch (family) {
        case GPUFamily::DATA_CENTER_MAX:
            return 1024;  // Xe-HPC architecture
        case GPUFamily::ARC_ALCHEMIST:
        case GPUFamily::ARC_BATTLEMAGE:
        case GPUFamily::DATA_CENTER_FLEX:
        case GPUFamily::UNKNOWN:
        default:
            return 64;    // Conservative default
    }
}

// Check if GPU family supports named barriers (nbarrier intrinsics)
// Named barriers are advanced ESIMD features for fine-grained synchronization.
// Only available on PVC (Xe-HPC), NOT on Arc (XeLPG/XeHPG).
// NOTE: This is now informational only - kernels use SPIR-V split barriers for Arc compatibility.
static bool supports_named_barriers(GPUFamily family) {
    // Only Data Center Max (PVC) supports named barriers
    return family == GPUFamily::DATA_CENTER_MAX;
}

// Check if GPU family supports ESIMD xmx::dpas intrinsics with ExecutionSize=16
//
// According to Intel Graphics Compiler documentation (documentation/visa/instructions/DPAS.md):
//   - Pre-PVC (XeHP/XeHPG/Arc Alchemist): ExecutionSize = 8 only
//   - PVC and later (Xe-HPC, Xe2/Battlemage): ExecutionSize = 16
//
// Our ESIMD kernels use ESIMD_EXEC_SIZE=16, so they require PVC or Xe2 class hardware.
//
// NOTE: XeLPG (Meteor Lake iGPU) does NOT have XMX hardware at all - that's a different
// architecture from Arc discrete GPUs. The "XeLPG" error message is misleading.
static bool gpu_family_supports_esimd_dpas(GPUFamily family) {
    switch (family) {
        case GPUFamily::DATA_CENTER_MAX:   // PVC (Xe-HPC) - ExecutionSize=16 supported
        case GPUFamily::ARC_BATTLEMAGE:    // Xe2 (B580, B570) - ExecutionSize=16 supported
            return true;
        case GPUFamily::ARC_ALCHEMIST:     // XeHPG (A770, A750) - ExecutionSize=8 only
        case GPUFamily::DATA_CENTER_FLEX:  // XeHPG-based - ExecutionSize=8 only
        case GPUFamily::UNKNOWN:
        default:
            return false;
    }
}

// =============================================================================
// XMXConfig::from_device() Implementation
// =============================================================================
// Queries hardware-specific XMX capabilities with robust edge case handling.

XMXConfig XMXConfig::from_device(int device_id) {
    XMXConfig cfg;  // Start with safe defaults

    // Edge case: device_id < 0 returns default config
    if (device_id < 0) {
        GGML_SYCL_DEBUG("[XMXConfig] device_id=%d < 0, returning default config\n", device_id);
        return cfg;
    }

    // Edge case: device_id >= device_count returns default config
    // Note: ggml_sycl_info() is in global namespace (defined in common.hpp)
    const auto & info = ::ggml_sycl_info();
    if (device_id >= info.device_count) {
        GGML_SYCL_DEBUG("[XMXConfig] device_id=%d >= device_count=%d, returning default config\n",
                        device_id, info.device_count);
        return cfg;
    }

    // Safe to access device info now
    const auto & dev = info.devices[device_id];
    const auto & xmx = dev.xmx_caps;

    // Copy hardware capability flags
    cfg.supported     = xmx.supported;
    cfg.supports_int8 = xmx.supports_int8;
    cfg.supports_fp16 = xmx.supports_fp16;

    // Copy nsm (compute units)
    cfg.nsm = dev.nsm > 0 ? dev.nsm : 20;  // Fallback to 20 if 0

    // Edge case: slm_size = 0 should use default
    cfg.slm_size = xmx.slm_size > 0 ? xmx.slm_size : 65536;

    // Detect GPU family for hardware capability settings
    GPUFamily family = detect_gpu_family_from_name(dev.device_name);

    // Query max work-group size for ESIMD kernels using device family
    // ESIMD kernels have stricter limits than regular SYCL kernels:
    // - Standard SYCL query returns 1024 for Arc B580, but ESIMD is limited to 64
    // - PVC (Data Center Max) can use up to 1024 work-items for ESIMD
    cfg.max_esimd_workgroup = get_max_esimd_workgroup(family);

    // Named barriers (nbarrier) are only available on PVC, not on Arc
    // This is now informational - kernels use SPIR-V split barriers which work on Arc
    cfg.supports_named_barrier = supports_named_barriers(family);

    // ESIMD dpas intrinsics with ExecutionSize=16 work on PVC and Xe2 (Battlemage)
    // Arc Alchemist (XeHPG) only supports ExecutionSize=8, requiring different kernel config
    cfg.supports_esimd_dpas = gpu_family_supports_esimd_dpas(family);

    // Edge case: M/N/K = 0 should use fallback defaults
    // XMX dimensions: Use queried values if valid, otherwise defaults
    cfg.xmx_m = (xmx.M > 0) ? xmx.M : 8;
    cfg.xmx_n = (xmx.N > 0) ? xmx.N : 16;

    // K dimension depends on data type:
    // - For INT8: Use queried K if valid (expected: 32)
    // - For FP16: Always 16 (SystolicDepth(8) x OpsPerChannel(2))
    cfg.xmx_k_int8 = (xmx.K > 0) ? xmx.K : 32;
    cfg.xmx_k_fp16 = 16;  // Fixed for FP16

    // Derived: double buffer feasibility
    // Double buffer if SLM can hold 2x tile buffers (conservative: 50% of SLM)
    // Tile buffer = M x K x sizeof(half) for activations + N x K x sizeof(half) for weights
    size_t tile_size = cfg.xmx_m * cfg.xmx_k_int8 * sizeof(sycl::half) +
                       cfg.xmx_n * cfg.xmx_k_int8 * sizeof(sycl::half);
    cfg.use_double_buffer = (2 * tile_size) < (cfg.slm_size / 2);

    // Default tiles per work-item (can be tuned later)
    cfg.tiles_per_workitem = 1;

    GGML_SYCL_DEBUG("[XMXConfig] device=%d: M=%zu N=%zu K_INT8=%zu K_FP16=%zu SLM=%zu nsm=%d "
                    "supported=%d int8=%d fp16=%d double_buf=%d max_esimd_wg=%d nbarrier=%d esimd_dpas=%d\n",
                    device_id, cfg.xmx_m, cfg.xmx_n, cfg.xmx_k_int8, cfg.xmx_k_fp16,
                    cfg.slm_size, cfg.nsm, cfg.supported, cfg.supports_int8,
                    cfg.supports_fp16, cfg.use_double_buffer, cfg.max_esimd_workgroup,
                    cfg.supports_named_barrier, cfg.supports_esimd_dpas);

    return cfg;
}

static bool ggml_sycl_unified_debug_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_SYCL_UNIFIED_DEBUG");
        enabled = (env && std::atoi(env) != 0) ? 1 : 0;
    }
    return enabled != 0;
}

// =============================================================================
// Kernel Class Names for Profiling
// =============================================================================

template <int TILE_M, int TILE_N, int TILE_K, bool USE_XMX>
class unified_matmul_kernel_name;

// Separate name for fallback to avoid ODR violation
class unified_matmul_kernel_fallback;

// DMMV-like kernel for batch=1 optimization
class unified_dmmv_kernel_name;

// =============================================================================
// Q4_0 Dequantization Helper
// =============================================================================

/**
 * Dequantize a single Q4_0 weight value.
 *
 * Q4_0 block layout:
 * - qs[0..15]: 16 bytes containing 32 4-bit values
 * - For index i < 16: value = (qs[i] & 0xF) - 8
 * - For index i >= 16: value = (qs[i-16] >> 4) - 8
 *
 * @param block Pointer to Q4_0 block
 * @param i     Index within block (0..31)
 * @return Dequantized float value
 */
SYCL_EXTERNAL inline float dequant_q4_0(const block_q4_0_unified * block, int i) {
    const float d = static_cast<float>(block->d);
    int         qs_val;
    if (i < 16) {
        qs_val = block->qs[i] & 0x0F;
    } else {
        qs_val = block->qs[i - 16] >> 4;
    }
    return static_cast<float>(qs_val - 8) * d;
}

/**
 * Dequantize Q4_0 weight to half precision for XMX.
 *
 * @param block Pointer to Q4_0 block
 * @param i     Index within block (0..31)
 * @return Dequantized half value
 */
SYCL_EXTERNAL inline sycl::half dequant_q4_0_half(const block_q4_0_unified * block, int i) {
    const sycl::half d = block->d;
    int              qs_val;
    if (i < 16) {
        qs_val = block->qs[i] & 0x0F;
    } else {
        qs_val = block->qs[i - 16] >> 4;
    }
    return static_cast<sycl::half>(qs_val - 8) * d;
}

/**
 * Dequantize Q4_0 weight from SoA layout to half precision for XMX.
 *
 * SoA Layout: [qs: N rows × K/32 blocks × 16 bytes/block] [d: N rows × K/32 blocks × sizeof(half)]
 *
 * @param qs_base     Base pointer to quantized values
 * @param d_base      Base pointer to scale factors
 * @param row         Row index (N dimension)
 * @param k_blocks    Number of K blocks per row (K / 32)
 * @param block_idx   Block index within the row (0 to k_blocks-1)
 * @param idx_in_blk  Index within block (0..31)
 * @return Dequantized half value
 */
SYCL_EXTERNAL inline sycl::half dequant_q4_0_half_soa(
    const uint8_t * qs_base,
    const sycl::half * d_base,
    int64_t row,
    int k_blocks,
    int block_idx,
    int idx_in_blk) {
    // Each row has k_blocks * 16 bytes of quantized values
    const int row_qs_bytes = k_blocks * 16;
    const uint8_t * qs_row = qs_base + row * row_qs_bytes;
    const sycl::half * d_row = d_base + row * k_blocks;

    const sycl::half d = d_row[block_idx];
    const uint8_t * qs = qs_row + block_idx * 16;

    int qs_val;
    if (idx_in_blk < 16) {
        qs_val = qs[idx_in_blk] & 0x0F;
    } else {
        qs_val = qs[idx_in_blk - 16] >> 4;
    }
    return static_cast<sycl::half>(qs_val - 8) * d;
}

// =============================================================================
// XMX Kernel Class Names
// =============================================================================

#if GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE
template <int TILE_M, int TILE_N, int TILE_K>
class unified_matmul_xmx_kernel_name;
#endif

// =============================================================================
// Unified Matmul Kernel - Optimized XMX Path
// =============================================================================

#if GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE

// Namespace alias for brevity
namespace sycl_xmx = sycl::ext::oneapi::experimental::matrix;

/**
 * Optimized XMX-accelerated matmul kernel using joint_matrix.
 *
 * GGML Convention: dst[m,n] = sum_k(src0[n,k] * src1[m,k])
 * Computes: output[M,N] = activations[M,K] @ weights[N,K]^T
 *
 * OPTIMIZATION: Direct joint_matrix accumulation
 * ==============================================
 * Key optimizations over the previous version:
 * 1. Larger work-groups with multiple sub-groups for better occupancy
 * 2. Direct joint_matrix_store to output without intermediate SLM extraction
 * 3. Streaming loads with reduced synchronization
 * 4. Full K-dimension processed with single accumulator
 *
 * Work distribution:
 * - Each work-group covers a TILE_M x TILE_N output region
 * - Sub-groups process XMX tiles within the work-group tile
 * - All sub-groups cooperate on loading, compute is distributed
 *
 * @tparam TILE_M  M tile size (must be multiple of 8)
 * @tparam TILE_N  N tile size (must be multiple of 16)
 * @tparam TILE_K  K tile size (must be multiple of XMX_TILE_K=16)
 */
template <int TILE_M, int TILE_N, int TILE_K>
SYCL_EXTERNAL void unified_matmul_xmx_kernel_impl(sycl::nd_item<2>                    item,
                                                   const UnifiedKernelArgs             args,
                                                   sycl::local_accessor<sycl::half, 1> slm_weights,
                                                   sycl::local_accessor<sycl::half, 1> slm_activations,
                                                   sycl::local_accessor<float, 1>      slm_acc_out) {
    auto sg = item.get_sub_group();

    // Tile coordinates
    const int tile_row = item.get_group(0);  // M dimension (output rows)
    const int tile_col = item.get_group(1);  // N dimension (output columns)

    // Thread coordinates within work-group
    const int local_row = item.get_local_id(0);
    const int local_col = item.get_local_id(1);
    const int local_size_row = item.get_local_range(0);
    const int local_size_col = item.get_local_range(1);
    const int local_linear = local_row * local_size_col + local_col;
    const int local_total = local_size_row * local_size_col;

    // Sub-group info
    const int sg_id = sg.get_group_linear_id();
    const int lane = sg.get_local_linear_id();

    // Global output coordinates for this work-group
    const int64_t m_start = tile_row * TILE_M;  // Starting output row
    const int64_t n_start = tile_col * TILE_N;  // Starting output column

    // Number of K tiles
    const int k_tiles = (args.K + TILE_K - 1) / TILE_K;
    const int k_blocks_per_row = args.K / UNIFIED_QK4_0;

    // Cast weight pointer (AoS layout)
    const block_q4_0_unified * weights = static_cast<const block_q4_0_unified *>(args.weights);

    // SoA layout pointers
    // SoA layout: [qs: N rows × K/32 blocks × 16 bytes/block] [d: N rows × K/32 blocks × sizeof(half)]
    const bool use_soa = (args.layout == LayoutMode::SOA);
    const int64_t total_blocks = args.N * k_blocks_per_row;
    const int64_t d_offset = total_blocks * (UNIFIED_QK4_0 / 2);  // Byte offset to scale values
    const uint8_t * qs_base = static_cast<const uint8_t *>(args.weights);
    const sycl::half * d_base = reinterpret_cast<const sycl::half *>(
        static_cast<const char *>(args.weights) + d_offset);

    // XMX tile dimensions and counts
    constexpr int NUM_TILES_M = TILE_M / XMX_TILE_M;
    constexpr int NUM_TILES_N = TILE_N / XMX_TILE_N;
    constexpr int NUM_K_STEPS = TILE_K / XMX_TILE_K;

    // Each sub-group handles one (tm, tn) output tile
    // Assign sub-groups to output tiles in row-major order
    const int num_output_tiles = NUM_TILES_M * NUM_TILES_N;
    const int num_subgroups = (local_total + XMX_SUBGROUP_SIZE - 1) / XMX_SUBGROUP_SIZE;

    // Joint matrix declarations
    sycl_xmx::joint_matrix<sycl::sub_group, sycl::half,
                           sycl_xmx::use::a, XMX_TILE_M, XMX_TILE_K,
                           sycl_xmx::layout::row_major> mat_a;
    sycl_xmx::joint_matrix<sycl::sub_group, sycl::half,
                           sycl_xmx::use::b, XMX_TILE_K, XMX_TILE_N,
                           sycl_xmx::layout::col_major> mat_b;
    sycl_xmx::joint_matrix<sycl::sub_group, float,
                           sycl_xmx::use::accumulator, XMX_TILE_M, XMX_TILE_N> acc;

    // Initialize accumulator to zero
    sycl_xmx::joint_matrix_fill(sg, acc, 0.0f);

    // Determine which output tile this sub-group handles
    // Each sub-group processes one XMX output tile (8x16)
    const int my_tile_idx = sg_id % num_output_tiles;
    const int my_tm = my_tile_idx / NUM_TILES_N;
    const int my_tn = my_tile_idx % NUM_TILES_N;
    const int m_base = my_tm * XMX_TILE_M;
    const int n_base = my_tn * XMX_TILE_N;

    // K-loop: iterate over K dimension in tiles
    for (int kt = 0; kt < k_tiles; kt++) {
        const int64_t k_start = kt * TILE_K;
        const int64_t k_end = sycl::min(k_start + TILE_K, args.K);
        const int     k_len = static_cast<int>(k_end - k_start);

        // ==== Cooperative Load: All threads load data to SLM ====
        // Load weights [TILE_N x TILE_K], but only up to k_len valid K elements
        for (int idx = local_linear; idx < TILE_N * TILE_K; idx += local_total) {
            const int n_off = idx / TILE_K;
            const int k_off = idx % TILE_K;
            const int64_t n_global = n_start + n_off;

            sycl::half w = sycl::half(0.0f);
            // Load only valid N/K combinations:
            // - n_off < TILE_N (always true due to loop structure)
            // - n_global < args.N (boundary check on N)
            // - k_off < k_len (only load actual K data for this tile)
            if (n_global < args.N && k_off < k_len) {
                const int64_t k_global = k_start + k_off;
                const int block_in_row = static_cast<int>(k_global / UNIFIED_QK4_0);
                const int idx_in_block = static_cast<int>(k_global % UNIFIED_QK4_0);
                if (use_soa) {
                    // SoA layout: separate qs and d arrays
                    w = dequant_q4_0_half_soa(qs_base, d_base, n_global, k_blocks_per_row, block_in_row, idx_in_block);
                } else {
                    // AoS layout: contiguous block_q4_0_unified structs
                    const int block_idx = static_cast<int>(n_global * k_blocks_per_row + block_in_row);
                    w = dequant_q4_0_half(&weights[block_idx], idx_in_block);
                }
            }
            slm_weights[n_off * TILE_K + k_off] = w;
        }

        // Load activations [TILE_M x TILE_K], but only up to k_len valid K elements
        for (int idx = local_linear; idx < TILE_M * TILE_K; idx += local_total) {
            const int m_off = idx / TILE_K;
            const int k_off = idx % TILE_K;
            const int64_t m_global = m_start + m_off;

            sycl::half a = sycl::half(0.0f);
            // Load only valid M/K combinations:
            // - m_off < TILE_M (always true due to loop structure)
            // - m_global < args.M (boundary check on M)
            // - k_off < k_len (only load actual K data for this tile)
            if (m_global < args.M && k_off < k_len) {
                const int64_t k_global = k_start + k_off;
                a = static_cast<sycl::half>(args.activations[m_global * args.K + k_global]);
            }
            slm_activations[m_off * TILE_K + k_off] = a;
        }

        // Barrier after loading
        item.barrier(sycl::access::fence_space::local_space);

        // ==== XMX Compute: Each sub-group computes its assigned tile ====
        if (sg_id < num_output_tiles) {
            // K-dimension loop within this K-tile
            // NOTE: TILE_K is always a full tile (32 for Q4_0)
            // Partial K only happens at last k_tile, handled by k_len check during load
            constexpr int NUM_K_TILE_STEPS = TILE_K / XMX_TILE_K;
            for (int tk = 0; tk < NUM_K_TILE_STEPS; tk++) {
                const int k_base = tk * XMX_TILE_K;

                // Load activations tile (row-major: activations[m, k])
                auto a_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                       sycl::access::decorated::no>(
                    const_cast<sycl::half*>(&slm_activations[m_base * TILE_K + k_base]));
                sycl_xmx::joint_matrix_load(sg, mat_a, a_ptr, TILE_K);

                // Load weights tile (col-major for transposed access)
                auto b_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                       sycl::access::decorated::no>(
                    const_cast<sycl::half*>(&slm_weights[n_base * TILE_K + k_base]));
                sycl_xmx::joint_matrix_load(sg, mat_b, b_ptr, TILE_K);

                // Compute: acc += A * B
                sycl_xmx::joint_matrix_mad(sg, acc, mat_a, mat_b, acc);
            }
        }

        // Barrier before next K-tile (only if there are more tiles)
        if (kt + 1 < k_tiles) {
            item.barrier(sycl::access::fence_space::local_space);
        }
    }

    // ==== Write output: Each sub-group stores its result ====
    if (sg_id < num_output_tiles) {
        const int64_t m_global_base = m_start + m_base;
        const int64_t n_global_base = n_start + n_base;

        // Check if ANY part of this tile is within bounds
        if (m_global_base < args.M && n_global_base < args.N) {
            // Store result directly to global memory
            // Need to handle boundary cases where tile extends beyond matrix
            const bool fully_in_bounds =
                (m_global_base + XMX_TILE_M <= args.M) &&
                (n_global_base + XMX_TILE_N <= args.N);

            if (fully_in_bounds && (args.N % XMX_TILE_N == 0)) {
                // Direct store to global memory for fully-in-bounds tiles
                // NOTE: Only use direct store when N is a multiple of XMX_TILE_N (16)
                // because joint_matrix_store with non-aligned stride causes data corruption
                auto out_ptr = sycl::address_space_cast<sycl::access::address_space::global_space,
                                                         sycl::access::decorated::no>(
                    &args.output[m_global_base * args.N + n_global_base]);
                sycl_xmx::joint_matrix_store(sg, acc, out_ptr, args.N, sycl_xmx::layout::row_major);
            } else {
                // Boundary case: Store to dedicated float SLM buffer,
                // then write valid elements to global memory with per-element bounds checking
                constexpr int ACC_SIZE = XMX_TILE_M * XMX_TILE_N;

                // Use the dedicated float SLM accessor
                auto slm_acc_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                            sycl::access::decorated::no>(&slm_acc_out[0]);

                sycl_xmx::joint_matrix_store(sg, acc, slm_acc_ptr, XMX_TILE_N, sycl_xmx::layout::row_major);
                sycl::group_barrier(sg);

                // Write valid elements with explicit bounds checking
                for (int i = lane; i < ACC_SIZE; i += XMX_SUBGROUP_SIZE) {
                    const int row = i / XMX_TILE_N;
                    const int col = i % XMX_TILE_N;
                    const int64_t m_global = m_global_base + row;
                    const int64_t n_global = n_global_base + col;

                    // Only write if BOTH indices are within bounds
                    if (m_global < args.M && n_global < args.N) {
                        args.output[m_global * args.N + n_global] = slm_acc_out[i];
                    }
                }
            }
        }
    }
}

#endif  // GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE

// =============================================================================
// ESIMD dpas Kernel - FP16 Path (Phase 2)
// =============================================================================
// Uses Intel ESIMD xmx::dpas for explicit SIMD control on XMX hardware.
// This path provides better XVE utilization than joint_matrix for some cases.
//
// ESIMD dpas characteristics:
// - dpas operand order: dpas(accumulator, B_tile, A_tile) - B before A!
// - K-tile for FP16 = 16 (SystolicDepth=8 x OpsPerChannel=2)
// - RepeatCount = 1-8, determines M dimension (we use 8)
// - ExecutionSize = 16 for Arc (determines N dimension)
// - Output tile: 8x16 (M x N)

#if GGML_SYCL_ESIMD_AVAILABLE

// Kernel class names for SYCL naming and profiling
template <int TILE_M, int TILE_N>
class esimd_fp16_kernel;

template <int TILE_M, int TILE_N>
class esimd_fp16_double_buffered_kernel;

template <int TILE_M, int TILE_N>
class esimd_int8_kernel;

#if GGML_SYCL_COOPERATIVE_KERNEL_ENABLED
// Cooperative ESIMD kernel: multi-work-item with work-group barrier
// Uses split barriers (SPV_INTEL_split_barrier) for efficient synchronization on Arc.
template <int WG_SIZE>
class esimd_fp16_cooperative_kernel;
#endif

#if GGML_SYCL_LARGE_TILE_KERNEL_ENABLED
// Large-tile ESIMD kernel: 64 work-items for 32x32 output tiles
// Uses split barriers (SPV_INTEL_split_barrier) for efficient synchronization on Arc.
template <int WG_SIZE>
class esimd_fp16_large_tile_kernel;
#endif

// =============================================================================
// ESIMD dpas Constants - FP16
// =============================================================================
// Hardware-defined parameters for FP16 dpas on Intel Arc
constexpr int ESIMD_SYSTOLIC_DEPTH = 8;   // Always 8 for dpas
constexpr int ESIMD_REPEAT_COUNT = 8;     // M dimension = RepeatCount
constexpr int ESIMD_EXEC_SIZE = 16;       // N dimension = ExecutionSize
constexpr int ESIMD_K_PER_DPAS = 16;      // K elements per dpas for FP16

// Operand sizes for FP16 dpas
// A (activations): Repeat * K_per = 8 * 16 = 128 half elements
// B (weights): K_per * ExecSize = 16 * 16 = 256 half elements
// Acc (output): Repeat * ExecSize = 8 * 16 = 128 float elements
constexpr int ESIMD_A_SIZE = ESIMD_REPEAT_COUNT * ESIMD_K_PER_DPAS;   // 128
constexpr int ESIMD_B_SIZE = ESIMD_K_PER_DPAS * ESIMD_EXEC_SIZE;     // 256
constexpr int ESIMD_ACC_SIZE = ESIMD_REPEAT_COUNT * ESIMD_EXEC_SIZE; // 128

// =============================================================================
// ESIMD dpas Constants - INT8
// =============================================================================
// Hardware-defined parameters for INT8 dpas on Intel Arc
// INT8 has K=32 per dpas (SystolicDepth=8 x OpsPerChannel=4)
constexpr int ESIMD_K_PER_DPAS_INT8 = 32;  // K elements per dpas for INT8

// Operand sizes for INT8 dpas
// A (activations): Repeat * K_per = 8 * 32 = 256 int8 elements
// B (weights): K_per * ExecSize = 32 * 16 = 512 int8 elements
// Acc (output): Repeat * ExecSize = 8 * 16 = 128 int32 elements
constexpr int ESIMD_A_SIZE_INT8 = ESIMD_REPEAT_COUNT * ESIMD_K_PER_DPAS_INT8;   // 256
constexpr int ESIMD_B_SIZE_INT8 = ESIMD_K_PER_DPAS_INT8 * ESIMD_EXEC_SIZE;     // 512
constexpr int ESIMD_ACC_SIZE_INT8 = ESIMD_REPEAT_COUNT * ESIMD_EXEC_SIZE;      // 128

// =============================================================================
// SLM Double-Buffering Constants for ESIMD FP16 Path (Phase 4)
// =============================================================================
// SLM layout for double-buffering memory/compute overlap.
// Each buffer holds one K-tile of weights and activations in FP16.
//
// Buffer sizes (FP16):
// - Weights: TILE_N × K_TILE × sizeof(half) = 16 × 16 × 2 = 512 bytes
// - Activations: TILE_M × K_TILE × sizeof(half) = 8 × 16 × 2 = 256 bytes
// - Total per buffer: 768 bytes
// - Double-buffer total: 1536 bytes
//
// Layout in SLM:
// Buffer 0: [0, 512): weights_0 [16×16 half]
//           [512, 768): activations_0 [8×16 half]
// Buffer 1: [768, 1280): weights_1 [16×16 half]
//           [1280, 1536): activations_1 [8×16 half]

constexpr int ESIMD_SLM_WEIGHTS_SIZE = ESIMD_EXEC_SIZE * ESIMD_K_PER_DPAS;      // 16 × 16 = 256 half elements
constexpr int ESIMD_SLM_ACTS_SIZE = ESIMD_REPEAT_COUNT * ESIMD_K_PER_DPAS;      // 8 × 16 = 128 half elements
constexpr int ESIMD_SLM_BUFFER_SIZE = ESIMD_SLM_WEIGHTS_SIZE + ESIMD_SLM_ACTS_SIZE;  // 384 half elements
constexpr int ESIMD_SLM_TOTAL_SIZE = 2 * ESIMD_SLM_BUFFER_SIZE;                 // 768 half elements for double-buffer

// SLM byte offsets for double-buffering
constexpr uint32_t ESIMD_SLM_BUF0_WEIGHTS = 0;
constexpr uint32_t ESIMD_SLM_BUF0_ACTS = ESIMD_SLM_WEIGHTS_SIZE * sizeof(sycl::half);                          // 512 bytes
constexpr uint32_t ESIMD_SLM_BUF1_WEIGHTS = ESIMD_SLM_BUFFER_SIZE * sizeof(sycl::half);                        // 768 bytes
constexpr uint32_t ESIMD_SLM_BUF1_ACTS = (ESIMD_SLM_BUFFER_SIZE + ESIMD_SLM_WEIGHTS_SIZE) * sizeof(sycl::half); // 1280 bytes
constexpr uint32_t ESIMD_SLM_TOTAL_BYTES = ESIMD_SLM_TOTAL_SIZE * sizeof(sycl::half);                          // 1536 bytes

// =============================================================================
// Cooperative ESIMD Constants (Multi-work-item with named barriers)
// =============================================================================
// Work-group configuration for cooperative loading:
// - 32 work-items per work-group (2 sub-groups of 16)
// - Each sub-group owns one 8x16 output tile
// - All work-items cooperate on loading larger tiles to SLM
//
// SLM layout for cooperative kernel:
// - Weights: [16] x [16] half values (raw, row-major)
// - Activations: [16] x [16] half values (raw, row-major)
// - Each sub-group reads its portion and packs to VNNI in registers
//
// Work-group size constraints:
// - Fixed at 32 (2 sub-groups of 16 work-items)
// - TODO: Add WG_SIZE=64 support in future (4 sub-groups)

constexpr int COOP_SUBGROUP_SIZE = 16;       // Sub-group size for XMX (fixed by hardware)

// Default constants for WG_SIZE=32 (compile-time constants for kernel instantiation)
constexpr int COOP_WG_SIZE_DEFAULT = 32;     // Default work-group size
constexpr int COOP_NUM_SUBGROUPS_DEFAULT = COOP_WG_SIZE_DEFAULT / COOP_SUBGROUP_SIZE;  // 2 sub-groups

// Output tile dimensions per work-group (for WG_SIZE=32)
constexpr int COOP_WG_TILES_M = 2;           // 2 M-tiles per work-group
constexpr int COOP_WG_TILES_N = 1;           // 1 N-tile per work-group
constexpr int COOP_WG_M = COOP_WG_TILES_M * ESIMD_REPEAT_COUNT;   // 16 output rows
constexpr int COOP_WG_N = COOP_WG_TILES_N * ESIMD_EXEC_SIZE;      // 16 output columns

// SLM sizes for cooperative kernel (single buffer for simplicity)
// Note: SLM usage stays under 64KB (actual: 1024 bytes) for all supported WG sizes
constexpr int COOP_SLM_WEIGHTS_SIZE = COOP_WG_N * ESIMD_K_PER_DPAS;  // 16 * 16 = 256 half
constexpr int COOP_SLM_ACTS_SIZE = COOP_WG_M * ESIMD_K_PER_DPAS;     // 16 * 16 = 256 half
constexpr int COOP_SLM_TOTAL_HALF = COOP_SLM_WEIGHTS_SIZE + COOP_SLM_ACTS_SIZE;  // 512 half
constexpr uint32_t COOP_SLM_TOTAL_BYTES = COOP_SLM_TOTAL_HALF * sizeof(sycl::half);  // 1024 bytes

// SLM byte offsets for cooperative kernel
constexpr uint32_t COOP_SLM_WEIGHTS_OFFSET = 0;
constexpr uint32_t COOP_SLM_ACTS_OFFSET = COOP_SLM_WEIGHTS_SIZE * sizeof(sycl::half);  // 512 bytes

// Legacy constant for backwards compatibility
constexpr int COOP_WG_SIZE = COOP_WG_SIZE_DEFAULT;

// =============================================================================
// Large-Tile ESIMD Constants (32x64 output tiles with 128 work-items)
// =============================================================================
// Work-group configuration for large-tile prompt processing:
// - 128 work-items per work-group (8 sub-groups of 16)
// - Each sub-group computes 2 stacked 8x16 dpas tiles = 16x16 output
// - 4 sub-groups per row covering 64 columns (4 × 16 = 64)
// - 2 rows of sub-groups covering 32 rows (2 × 16 = 32)
// - All work-items cooperate on loading larger tiles to SLM
//
// Sub-group layout (8 sub-groups in 4×2 grid):
//   sg_row = sg_id / 4  (0 or 1 for rows of sub-groups)
//   sg_col = sg_id % 4  (0-3 for columns of sub-groups)
//
// Output tile ownership:
//   M range: [sg_row * 16 .. sg_row * 16 + 15] (16 rows per sg_row)
//   N range: [sg_col * 16 .. sg_col * 16 + 15] (16 columns per sub-group)
//
// Note: Each sub-group actually computes 2 dpas tiles (8×16 each) stacked
// vertically, for a total of 16 rows per sub-group row.

constexpr int LARGE_WG_SIZE = 64;                           // Work-items per work-group (ESIMD limit)
constexpr int LARGE_NUM_SUBGROUPS = LARGE_WG_SIZE / COOP_SUBGROUP_SIZE;  // 4 sub-groups
constexpr int LARGE_SG_COLS = 2;                            // Sub-groups per row (covering N)
constexpr int LARGE_SG_ROWS = 2;                            // Rows of sub-groups (covering M)

// Output dimensions match header constants
// LARGE_TILE_M = 32, LARGE_TILE_N = 32, LARGE_TILE_K = 32 (defined in hpp)

// Each sub-group row handles 16 M-rows (2 dpas tiles of 8 rows each)
constexpr int LARGE_SG_M = LARGE_TILE_M / LARGE_SG_ROWS;    // 16 rows per sub-group row

// SLM sizes for large-tile kernel
// Weights: 32 rows × 32 cols = 1024 half = 2048 bytes
// Activations: 32 rows × 32 cols = 1024 half = 2048 bytes
// Total: 2048 half = 4096 bytes (well under 64KB limit)
constexpr int LARGE_SLM_WEIGHTS_SIZE = LARGE_TILE_N * LARGE_TILE_K;  // 32 × 32 = 1024 half
constexpr int LARGE_SLM_ACTS_SIZE = LARGE_TILE_M * LARGE_TILE_K;     // 32 × 32 = 1024 half
constexpr int LARGE_SLM_TOTAL_HALF = LARGE_SLM_WEIGHTS_SIZE + LARGE_SLM_ACTS_SIZE;  // 2048 half
constexpr uint32_t LARGE_SLM_TOTAL_BYTES = LARGE_SLM_TOTAL_HALF * sizeof(sycl::half);  // 4096 bytes

// SLM byte offsets for large-tile kernel
constexpr uint32_t LARGE_SLM_WEIGHTS_OFFSET = 0;
constexpr uint32_t LARGE_SLM_ACTS_OFFSET = LARGE_SLM_WEIGHTS_SIZE * sizeof(sycl::half);  // 2048 bytes

// =============================================================================
// Vectorized Q4_0 Dequantization using ESIMD SIMD Operations
// =============================================================================
// These functions provide high-throughput dequantization for Q4_0 weights
// using ESIMD vector operations instead of scalar loops.
//
// Q4_0 block layout (18 bytes total):
// - d: sycl::half scale factor (2 bytes)
// - qs[16]: 16 packed bytes containing 32 nibbles (16 bytes)
//   - Low nibble:  qs[i] & 0x0F  -> value - 8 for signed range [-8, +7]
//   - High nibble: qs[i] >> 4   -> value - 8 for signed range [-8, +7]
//
// Weight order in memory (after unpacking):
// - Positions 0-15:  low nibbles from qs[0..15]
// - Positions 16-31: high nibbles from qs[0..15]
// =============================================================================

/**
 * Vectorized dequantization of a full Q4_0 block (32 weights) to FP16.
 *
 * Loads 16 packed bytes and unpacks to 32 half-precision values using
 * ESIMD SIMD operations. This eliminates scalar loops in the hot path.
 *
 * @param block  Pointer to Q4_0 block (must be valid, no bounds check)
 * @return simd<sycl::half, 32> containing dequantized weights
 *
 * Performance: ~16 bytes loaded, 32 weights output = 2:1 expansion ratio
 * Target throughput: >100 GB/s for Q4_0 on Intel Arc
 */
SYCL_ESIMD_FUNCTION esimd::simd<sycl::half, UNIFIED_QK4_0>
dequant_q4_0_block_vectorized(const block_q4_0_unified * block) {
    // Load scale factor
    const sycl::half d = block->d;

    // Load all 16 packed bytes at once using pointer cast
    // Note: block->qs is 16 bytes, we load them into a simd vector
    esimd::simd<uint8_t, 16> packed;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        packed[i] = block->qs[i];
    }

    // Extract low nibbles: (packed & 0x0F) - 8
    // Use bitwise AND and subtraction on simd vectors
    esimd::simd<int32_t, 16> lo_nibbles = esimd::simd<int32_t, 16>(packed & 0x0F) - 8;

    // Extract high nibbles: (packed >> 4) - 8
    esimd::simd<int32_t, 16> hi_nibbles = esimd::simd<int32_t, 16>(packed >> 4) - 8;

    // Convert to half precision and apply scale
    // Result layout: [lo_0, lo_1, ..., lo_15, hi_0, hi_1, ..., hi_15]
    esimd::simd<sycl::half, UNIFIED_QK4_0> result;

    // Low nibbles go to positions 0-15
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        result[i] = static_cast<sycl::half>(lo_nibbles[i]) * d;
    }

    // High nibbles go to positions 16-31
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        result[i + 16] = static_cast<sycl::half>(hi_nibbles[i]) * d;
    }

    return result;
}

/**
 * Vectorized dequantization of a partial Q4_0 block to FP16.
 *
 * For tiles that don't align to full 32-weight blocks, this function
 * handles partial block extraction efficiently.
 *
 * @param block       Pointer to Q4_0 block
 * @param start_idx   Starting index within block (0..31)
 * @param count       Number of weights to extract (1..32-start_idx)
 * @param output      Output array to fill with dequantized weights
 *
 * Note: For best performance, prefer dequant_q4_0_block_vectorized()
 * when processing full blocks.
 */
template <int MAX_COUNT>
SYCL_ESIMD_FUNCTION void dequant_q4_0_partial_vectorized(
    const block_q4_0_unified * block,
    int                        start_idx,
    int                        count,
    esimd::simd<sycl::half, MAX_COUNT> & output) {

    // Get full block dequantization
    esimd::simd<sycl::half, UNIFIED_QK4_0> full = dequant_q4_0_block_vectorized(block);

    // Copy requested portion
    #pragma unroll
    for (int i = 0; i < MAX_COUNT; i++) {
        if (i < count) {
            output[i] = full[start_idx + i];
        }
    }
}

/**
 * Vectorized dequantization for a tile of weights spanning multiple blocks.
 *
 * This is the main entry point for tile-based dequantization. Handles:
 * - Full blocks within the tile (vectorized)
 * - Partial blocks at tile boundaries (vectorized with masking)
 * - VNNI format output for dpas compatibility
 *
 * @tparam TILE_K  K dimension tile size (should be multiple of ESIMD_K_PER_DPAS)
 * @tparam TILE_N  N dimension tile size (typically 16 for dpas)
 * @param weights  Pointer to Q4_0 weight blocks
 * @param n_global Global N index for this weight row
 * @param k_start  Starting K index for this tile
 * @param K        Total K dimension
 * @param k_blocks_per_row Number of Q4_0 blocks per weight row
 * @param k_len    Valid K elements in this tile
 * @return simd containing dequantized weights in row-major order
 */
template <int TILE_K>
SYCL_ESIMD_FUNCTION esimd::simd<sycl::half, TILE_K>
dequant_q4_0_tile_vectorized(
    const block_q4_0_unified * weights,
    int64_t                    n_global,
    int64_t                    k_start,
    int64_t                    K,
    int                        k_blocks_per_row,
    int                        k_len) {

    esimd::simd<sycl::half, TILE_K> result = sycl::half(0.0f);

    // For TILE_K=16 (ESIMD_K_PER_DPAS), we may span at most 2 Q4_0 blocks
    // since each block has 32 weights and TILE_K=16

    // Calculate which block(s) we need
    const int first_block_idx = static_cast<int>(k_start / UNIFIED_QK4_0);
    const int start_in_block = static_cast<int>(k_start % UNIFIED_QK4_0);

    // Get the first block
    const int global_block_idx = static_cast<int>(n_global * k_blocks_per_row + first_block_idx);
    const block_q4_0_unified * blk = &weights[global_block_idx];

    // Dequantize the full block
    esimd::simd<sycl::half, UNIFIED_QK4_0> full_block = dequant_q4_0_block_vectorized(blk);

    // Copy weights from the block to result
    // Handle the case where we start mid-block
    // Note: Using ternary instead of sycl::min which is not supported in ESIMD context
    const int remaining_in_block = UNIFIED_QK4_0 - start_in_block;
    const int weights_from_first_block = (remaining_in_block < k_len) ? remaining_in_block : k_len;

    #pragma unroll
    for (int i = 0; i < TILE_K; i++) {
        if (i < weights_from_first_block) {
            result[i] = full_block[start_in_block + i];
        }
    }

    // If we need weights from a second block (tile spans block boundary)
    if (weights_from_first_block < k_len) {
        const int second_block_idx = first_block_idx + 1;
        if (second_block_idx < k_blocks_per_row) {
            const int global_block_idx_2 = static_cast<int>(n_global * k_blocks_per_row + second_block_idx);
            const block_q4_0_unified * blk2 = &weights[global_block_idx_2];

            esimd::simd<sycl::half, UNIFIED_QK4_0> full_block_2 = dequant_q4_0_block_vectorized(blk2);

            const int weights_from_second = k_len - weights_from_first_block;
            #pragma unroll
            for (int i = 0; i < TILE_K; i++) {
                if (i >= weights_from_first_block && i < k_len) {
                    result[i] = full_block_2[i - weights_from_first_block];
                }
            }
        }
    }

    return result;
}

// =============================================================================
// Vectorized MXFP4 Dequantization using ESIMD SIMD Operations
// =============================================================================
// MXFP4 uses a shared E8M0 exponent with E2M1 mantissa values.
//
// MXFP4 block layout (17 bytes total):
// - e: uint8_t E8M0 shared exponent (1 byte), represents 2^(e-127)
// - qs[16]: 16 packed bytes containing 32 nibbles (16 bytes)
//   - Low nibble:  qs[i] & 0x0F  -> lookup in kvalues_mxfp4
//   - High nibble: qs[i] >> 4   -> lookup in kvalues_mxfp4
//
// The kvalues_mxfp4 lookup table maps 4-bit codes to signed integers
// that are doubled - multiply by 0.5 during dequantization.
// =============================================================================

/**
 * Convert E8M0 exponent to float scale factor (halved for MXFP4).
 *
 * E8M0 is an 8-bit unsigned exponent representing 2^(e-127).
 * For MXFP4, we pre-apply the 0.5 factor here since kvalues are doubled.
 *
 * @param e E8M0 exponent byte
 * @return Float scale factor (already halved)
 */
SYCL_ESIMD_FUNCTION float e8m0_to_scale_esimd(uint8_t e) {
    uint32_t bits;
    if (e == 0) {
        // Denormal case: return 2^(-127) * 0.5 = 2^(-128)
        bits = 0x00400000;  // Small positive float
    } else {
        // Normal case: 2^(e-127) * 0.5 = 2^(e-128)
        bits = static_cast<uint32_t>(e - 1) << 23;
    }
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

/**
 * Vectorized dequantization of a full MXFP4 block (32 weights) to FP16.
 *
 * Loads 16 packed bytes, looks up E2M1 values, and applies E8M0 scale.
 * This eliminates scalar loops in the hot path.
 *
 * @param block  Pointer to MXFP4 block (must be valid, no bounds check)
 * @return simd<sycl::half, 32> containing dequantized weights
 *
 * Performance: ~17 bytes loaded, 32 weights output
 * Target throughput: >100 GB/s for MXFP4 on Intel Arc
 */
SYCL_ESIMD_FUNCTION esimd::simd<sycl::half, UNIFIED_QK_MXFP4>
dequant_mxfp4_block_vectorized(const block_mxfp4_unified * block) {
    // Get scale factor (already halved for MXFP4 kvalues)
    const float scale = e8m0_to_scale_esimd(block->e);

    // Load all 16 packed bytes at once
    esimd::simd<uint8_t, 16> packed;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        packed[i] = block->qs[i];
    }

    // Dequantize using kvalues lookup table
    // kvalues_mxfp4_unified: maps 4-bit codes to doubled signed integers
    esimd::simd<sycl::half, UNIFIED_QK_MXFP4> result;

    // Low nibbles go to positions 0-15
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        const int8_t kval = kvalues_mxfp4_unified[packed[i] & 0x0F];
        result[i] = static_cast<sycl::half>(static_cast<float>(kval) * scale);
    }

    // High nibbles go to positions 16-31
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        const int8_t kval = kvalues_mxfp4_unified[packed[i] >> 4];
        result[i + 16] = static_cast<sycl::half>(static_cast<float>(kval) * scale);
    }

    return result;
}

/**
 * Vectorized dequantization for a tile of MXFP4 weights spanning multiple blocks.
 *
 * This is the main entry point for tile-based MXFP4 dequantization.
 *
 * @tparam TILE_K  K dimension tile size (should be multiple of ESIMD_K_PER_DPAS)
 * @param weights  Pointer to MXFP4 weight blocks
 * @param n_global Global N index for this weight row
 * @param k_start  Starting K index for this tile
 * @param K        Total K dimension
 * @param k_blocks_per_row Number of MXFP4 blocks per weight row
 * @param k_len    Valid K elements in this tile
 * @return simd containing dequantized weights in row-major order
 */
template <int TILE_K>
SYCL_ESIMD_FUNCTION esimd::simd<sycl::half, TILE_K>
dequant_mxfp4_tile_vectorized(
    const block_mxfp4_unified * weights,
    int64_t                     n_global,
    int64_t                     k_start,
    int64_t                     K,
    int                         k_blocks_per_row,
    int                         k_len) {

    esimd::simd<sycl::half, TILE_K> result = sycl::half(0.0f);

    // For TILE_K=16 (ESIMD_K_PER_DPAS), we may span at most 2 MXFP4 blocks
    // since each block has 32 weights and TILE_K=16

    // Calculate which block(s) we need
    const int first_block_idx = static_cast<int>(k_start / UNIFIED_QK_MXFP4);
    const int start_in_block = static_cast<int>(k_start % UNIFIED_QK_MXFP4);

    // Get the first block
    const int global_block_idx = static_cast<int>(n_global * k_blocks_per_row + first_block_idx);
    const block_mxfp4_unified * blk = &weights[global_block_idx];

    // Dequantize the full block
    esimd::simd<sycl::half, UNIFIED_QK_MXFP4> full_block = dequant_mxfp4_block_vectorized(blk);

    // Copy weights from the block to result
    const int remaining_in_block = UNIFIED_QK_MXFP4 - start_in_block;
    const int weights_from_first_block = (remaining_in_block < k_len) ? remaining_in_block : k_len;

    #pragma unroll
    for (int i = 0; i < TILE_K; i++) {
        if (i < weights_from_first_block) {
            result[i] = full_block[start_in_block + i];
        }
    }

    // If we need weights from a second block (tile spans block boundary)
    if (weights_from_first_block < k_len) {
        const int second_block_idx = first_block_idx + 1;
        if (second_block_idx < k_blocks_per_row) {
            const int global_block_idx_2 = static_cast<int>(n_global * k_blocks_per_row + second_block_idx);
            const block_mxfp4_unified * blk2 = &weights[global_block_idx_2];

            esimd::simd<sycl::half, UNIFIED_QK_MXFP4> full_block_2 = dequant_mxfp4_block_vectorized(blk2);

            #pragma unroll
            for (int i = 0; i < TILE_K; i++) {
                if (i >= weights_from_first_block && i < k_len) {
                    result[i] = full_block_2[i - weights_from_first_block];
                }
            }
        }
    }

    // Suppress unused warning
    (void) K;

    return result;
}

// =============================================================================
// Vectorized Q8_0 Dequantization using ESIMD SIMD Operations
// =============================================================================
// Q8_0 is simpler than Q4_0 since each weight is a full byte (no nibble packing).
//
// Q8_0 block layout (34 bytes total):
// - d: sycl::half scale factor (2 bytes)
// - qs[32]: 32 signed int8 values (32 bytes)
//
// No unpacking needed - just load, cast to half, and multiply by scale.
// =============================================================================

/**
 * Vectorized dequantization of a full Q8_0 block (32 weights) to FP16.
 *
 * Q8_0 is simpler than Q4_0 - no nibble unpacking required.
 * Just load 32 int8 values, convert to half, and multiply by scale.
 *
 * @param qs     Pointer to 32 int8 quantized values
 * @param scale  Scale factor to apply
 * @return simd<sycl::half, 32> containing dequantized weights
 *
 * Target throughput: >200 GB/s for Q8_0 on Intel Arc (simpler path)
 */
SYCL_ESIMD_FUNCTION esimd::simd<sycl::half, 32>
dequant_q8_0_block_vectorized(const int8_t * qs, sycl::half scale) {
    // Load all 32 int8 values
    esimd::simd<int32_t, 32> weights_int;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        weights_int[i] = static_cast<int32_t>(qs[i]);
    }

    // Convert to half and apply scale
    esimd::simd<sycl::half, 32> result;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        result[i] = static_cast<sycl::half>(weights_int[i]) * scale;
    }

    return result;
}

/**
 * Vectorized dequantization for a K-tile of Q8_0 weights.
 *
 * @tparam TILE_K  K dimension tile size
 * @param qs       Pointer to int8 quantized values
 * @param scale    Scale factor to apply
 * @param k_start  Starting offset within the block
 * @param k_len    Number of weights to extract
 * @return simd containing dequantized weights
 */
template <int TILE_K>
SYCL_ESIMD_FUNCTION esimd::simd<sycl::half, TILE_K>
dequant_q8_0_tile_vectorized(
    const int8_t * qs,
    sycl::half     scale,
    int            k_start,
    int            k_len) {

    esimd::simd<sycl::half, TILE_K> result = sycl::half(0.0f);

    // For Q8_0, weights are stored directly as int8
    // Just load, convert, and scale
    #pragma unroll
    for (int i = 0; i < TILE_K; i++) {
        if (i < k_len) {
            const int8_t q = qs[k_start + i];
            result[i] = static_cast<sycl::half>(static_cast<int32_t>(q)) * scale;
        }
    }

    return result;
}

// =============================================================================
// Prefetch Support for Memory/Compute Overlap (Phase 4 - Task llama.cpp-attk)
// =============================================================================
// Prefetching future K-tiles while computing current tiles improves throughput.
//
// Cache hint strategy:
// - Weights: use once per output element, minimal caching benefit
// - Activations: reused across N columns, benefits from caching
//
// Note: ESIMD prefetch uses different APIs depending on compiler version.
// We use esimd::prefetch for basic prefetch functionality.
// The prefetch distance is passed in via args.prefetch_depth from the host.

/**
 * Prefetch Q4_0 weights for a future K-tile using LSC prefetch with cache hints.
 *
 * Prefetches weight data asynchronously using Intel LSC (Load/Store Cache)
 * prefetch intrinsics. Data will be in cache when the load is executed
 * later in the K-loop.
 *
 * Cache hint strategy for weights:
 * - L1: streaming - weights are used once per output element, don't pollute L1
 * - L2: uncached  - weights have poor temporal locality, don't cache in L2
 *
 * @param weights          Pointer to Q4_0 weight blocks
 * @param n_global         Global N index for this weight row
 * @param k_tile_start     Starting K index for the tile to prefetch
 * @param K                Total K dimension
 * @param k_blocks_per_row Number of Q4_0 blocks per weight row
 * @param N                Total N dimension (for bounds checking)
 */
template <int K_TILE_SIZE>
SYCL_ESIMD_FUNCTION void prefetch_weights_block(
    const block_q4_0_unified * weights,
    int64_t                    n_global,
    int64_t                    k_tile_start,
    int64_t                    K,
    int                        k_blocks_per_row,
    int64_t                    N) {

    // Bounds check: don't prefetch beyond array
    if (n_global >= N || k_tile_start >= K) {
        return;
    }

    // Calculate block address for this K-tile
    // Q4_0 blocks have 32 weights each
    const int k_block_idx = static_cast<int>(k_tile_start / UNIFIED_QK4_0);
    const int global_block_idx = static_cast<int>(n_global * k_blocks_per_row + k_block_idx);

    // Prefetch the Q4_0 block using LSC prefetch with streaming cache hints
    // This brings the block into cache ahead of the actual load without
    // occupying registers or polluting cache with data that's only used once.
    const block_q4_0_unified * block_ptr = &weights[global_block_idx];

    // Use LSC prefetch with streaming hints for weights:
    // - L1 streaming: evict-first policy to minimize cache pollution
    // - L2 uncached: don't cache in L2 since weights have poor temporal locality
    //
    // NOTE: Alignment is critical for LSC prefetch:
    // - block_q4_0_unified is 18 bytes (2 bytes scale + 16 bytes quants)
    // - Array of 18-byte blocks is at offsets 0, 18, 36, ... (≡ 2 mod 4)
    // - This violates DWORD (4-byte) alignment required by uint32_t prefetch
    //
    // Solution: Align to 16-byte boundary and prefetch that aligned portion.
    // This covers the entire 18-byte block within a 32-byte range.
    // Calculate 16-byte aligned address (align down)
    constexpr int ALIGN_SIZE = 16;  // 16-byte alignment for proper DWORD-aligned access
    const uint8_t * byte_ptr = reinterpret_cast<const uint8_t *>(block_ptr);
    const uint64_t addr = reinterpret_cast<uint64_t>(byte_ptr);
    const uint64_t aligned_addr = (addr / ALIGN_SIZE) * ALIGN_SIZE;
    const uint32_t * aligned_ptr = reinterpret_cast<const uint32_t *>(aligned_addr);

    constexpr auto props = esimd::properties{
        esimd::cache_hint_L1<esimd::cache_hint::streaming>,
        esimd::cache_hint_L2<esimd::cache_hint::uncached>
    };
    // Prefetch 4 uint32_t values (16 bytes, covers the entire 18-byte block within aligned boundary)
    esimd::prefetch<uint32_t, 4>(
        aligned_ptr, 0, esimd::simd_mask<1>(1), props);
}

/**
 * Prefetch MXFP4 weights for a future K-tile.
 *
 * @param weights          Pointer to MXFP4 weight blocks
 * @param n_global         Global N index for this weight row
 * @param k_tile_start     Starting K index for the tile to prefetch
 * @param K                Total K dimension
 * @param k_blocks_per_row Number of blocks per weight row
 * @param N                Total N dimension (for bounds checking)
 */
template <int K_TILE_SIZE>
SYCL_ESIMD_FUNCTION void prefetch_weights_block_mxfp4(
    const block_mxfp4_unified * weights,
    int64_t                     n_global,
    int64_t                     k_tile_start,
    int64_t                     K,
    int                         k_blocks_per_row,
    int64_t                     N) {

    // Bounds check: don't prefetch beyond array
    if (n_global >= N || k_tile_start >= K) {
        return;
    }

    // Calculate block address for this K-tile
    // MXFP4 blocks have 32 weights each
    const int k_block_idx = static_cast<int>(k_tile_start / UNIFIED_QK_MXFP4);
    const int global_block_idx = static_cast<int>(n_global * k_blocks_per_row + k_block_idx);

    // Prefetch the MXFP4 block using LSC prefetch
    const block_mxfp4_unified * block_ptr = &weights[global_block_idx];

    // Align to 16-byte boundary
    constexpr int ALIGN_SIZE = 16;
    const uint8_t * byte_ptr = reinterpret_cast<const uint8_t *>(block_ptr);
    const uint64_t addr = reinterpret_cast<uint64_t>(byte_ptr);
    const uint64_t aligned_addr = (addr / ALIGN_SIZE) * ALIGN_SIZE;
    const uint32_t * aligned_ptr = reinterpret_cast<const uint32_t *>(aligned_addr);

    constexpr auto props = esimd::properties{
        esimd::cache_hint_L1<esimd::cache_hint::streaming>,
        esimd::cache_hint_L2<esimd::cache_hint::uncached>
    };
    // Prefetch 4 uint32_t values (16 bytes, covers the entire 17-byte block within aligned boundary)
    esimd::prefetch<uint32_t, 4>(
        aligned_ptr, 0, esimd::simd_mask<1>(1), props);
}

/**
 * Generic prefetch dispatcher for weights.
 *
 * @param weights          Pointer to weight blocks (void*)
 * @param n_global         Global N index for this weight row
 * @param k_tile_start     Starting K index for the tile to prefetch
 * @param K                Total K dimension
 * @param k_blocks_per_row Number of blocks per weight row
 * @param N                Total N dimension (for bounds checking)
 * @param quant_type       Quantization type
 */
template <int K_TILE_SIZE>
SYCL_ESIMD_FUNCTION void prefetch_weights_block_generic(
    const void * weights,
    int64_t      n_global,
    int64_t      k_tile_start,
    int64_t      K,
    int          k_blocks_per_row,
    int64_t      N,
    int          quant_type) {

    if (quant_type == QUANT_TYPE_MXFP4) {
        prefetch_weights_block_mxfp4<K_TILE_SIZE>(
            static_cast<const block_mxfp4_unified *>(weights),
            n_global, k_tile_start, K, k_blocks_per_row, N);
    } else {
        // Default: Q4_0
        prefetch_weights_block<K_TILE_SIZE>(
            static_cast<const block_q4_0_unified *>(weights),
            n_global, k_tile_start, K, k_blocks_per_row, N);
    }
}

/**
 * Prefetch activations for a future K-tile using LSC prefetch with cache hints.
 *
 * Prefetches activation data asynchronously using Intel LSC (Load/Store Cache)
 * prefetch intrinsics. Activations benefit more from caching as they are
 * reused across N columns.
 *
 * Cache hint strategy for activations:
 * - L1: cached - activations are reused across multiple output columns
 * - L2: cached - activations have good temporal locality
 *
 * @param activations      Pointer to activation matrix
 * @param m_global         Global M index for this activation row
 * @param k_tile_start     Starting K index for the tile to prefetch
 * @param K                Total K dimension (for indexing and bounds)
 * @param M                Total M dimension (for bounds checking)
 */
template <int K_TILE_SIZE>
SYCL_ESIMD_FUNCTION void prefetch_activations_block(
    const float * activations,
    int64_t       m_global,
    int64_t       k_tile_start,
    int64_t       K,
    int64_t       M) {

    // Bounds check: don't prefetch beyond array
    if (m_global >= M || k_tile_start >= K) {
        return;
    }

    // Calculate address for this K-tile
    const float * tile_ptr = activations + m_global * K + k_tile_start;

    // Use LSC prefetch with cached hints for activations:
    // - L1 cached: activations are reused across N columns
    // - L2 cached: activations have good temporal locality
    // Prefetch K_TILE_SIZE floats (typically 16 floats = 64 bytes)
    constexpr auto props = esimd::properties{
        esimd::cache_hint_L1<esimd::cache_hint::cached>,
        esimd::cache_hint_L2<esimd::cache_hint::cached>
    };
    esimd::prefetch<float, K_TILE_SIZE>(tile_ptr, 0, esimd::simd_mask<1>(1), props);
}

/**
 * Prefetch weights for a K-tile (cooperative version).
 *
 * Called by work-items in cooperative kernel. Each work-item prefetches
 * one weight row if local_id < TILE_N.
 *
 * @param weights          Pointer to Q4_0 weight blocks
 * @param n_wg_start       Starting N index for this work-group
 * @param k_tile_start     Starting K index for the tile to prefetch
 * @param args             Kernel arguments (for dimensions)
 * @param k_blocks_per_row Number of Q4_0 blocks per weight row
 * @param local_id         Work-item local ID [0..WG_SIZE)
 * @param tile_n           N tile size (COOP_WG_N)
 */
template <int K_TILE_SIZE, int TILE_N>
SYCL_ESIMD_FUNCTION void prefetch_weights_cooperative(
    const block_q4_0_unified *  weights,
    int64_t                     n_wg_start,
    int64_t                     k_tile_start,
    const UnifiedKernelArgs &   args,
    int                         k_blocks_per_row,
    int                         local_id,
    int                         tile_n) {

    // Only work-items [0..TILE_N) prefetch weights
    if (local_id < tile_n) {
        const int64_t n_global = n_wg_start + local_id;
        prefetch_weights_block<K_TILE_SIZE>(
            weights, n_global, k_tile_start, args.K, k_blocks_per_row, args.N);
    }
}

/**
 * Prefetch activations for a K-tile (cooperative version).
 *
 * Called by work-items in cooperative kernel. Each work-item prefetches
 * one activation row if local_id >= TILE_N.
 *
 * @param activations      Pointer to activation matrix
 * @param m_wg_start       Starting M index for this work-group
 * @param k_tile_start     Starting K index for the tile to prefetch
 * @param args             Kernel arguments (for dimensions)
 * @param local_id         Work-item local ID [0..WG_SIZE)
 * @param tile_n           N tile size (COOP_WG_N) - work-items >= this prefetch activations
 * @param tile_m           M tile size (COOP_WG_M)
 */
template <int K_TILE_SIZE, int TILE_M>
SYCL_ESIMD_FUNCTION void prefetch_activations_cooperative(
    const float *             activations,
    int64_t                   m_wg_start,
    int64_t                   k_tile_start,
    const UnifiedKernelArgs & args,
    int                       local_id,
    int                       tile_n,
    int                       tile_m) {

    // Work-items [TILE_N..TILE_N+TILE_M) prefetch activations
    const int act_id = local_id - tile_n;
    if (act_id >= 0 && act_id < tile_m) {
        const int64_t m_global = m_wg_start + act_id;
        prefetch_activations_block<K_TILE_SIZE>(
            activations, m_global, k_tile_start, args.K, args.M);
    }
}

/**
 * Check if cooperative ESIMD dpas path is enabled via environment.
 *
 * Cooperative path uses multiple work-items with named barriers for
 * work-group level loading. Enabled by default while optimizing;
 * set GGML_SYCL_XMX_COOPERATIVE=0 to disable.
 *
 * @return true if cooperative ESIMD path is enabled
 */
inline bool use_cooperative_esimd() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_SYCL_XMX_COOPERATIVE");
        if (!env) {
            enabled = 1;
        } else {
            enabled = (std::string(env) == "0") ? 0 : 1;
        }
    }
    return enabled != 0;
}

/**
 * Get the configured work-group size for cooperative ESIMD kernel.
 *
 * Currently only WG_SIZE=32 is supported (2 sub-groups of 16 work-items).
 * TODO: Add WG_SIZE=64 support in future (4 sub-groups covering 4 output tiles)
 *
 * @return Work-group size (always 32)
 */
inline int get_cooperative_wg_size() {
    return 32;
}

/**
 * Check if cooperative ESIMD dpas path can be used for given dimensions.
 *
 * Cooperative ESIMD dpas requires:
 * - ESIMD enabled via GGML_SYCL_XMX_ESIMD=1
 * - Cooperative enabled via GGML_SYCL_XMX_COOPERATIVE=1
 * - M >= 8 (at least one dpas M-tile)
 * - N >= 16 (at least one dpas N-tile)
 * - K aligned to Q4_0 block size (32)
 *
 * @param M  Output rows
 * @param N  Output columns
 * @param K  Reduction dimension
 * @return true if cooperative ESIMD dpas can be used
 */
inline bool can_use_cooperative_esimd(int64_t M, int64_t N, int64_t K) {
    // Both ESIMD and cooperative must be enabled
    if (!use_esimd_dpas() || !use_cooperative_esimd()) {
        return false;
    }
    // K must be multiple of Q4_0 block size for proper dequantization
    if (K % UNIFIED_QK4_0 != 0) {
        return false;
    }
    // Need enough work for cooperative loading to be beneficial
    // At least 8 M-rows for one dpas tile per sub-group
    return M >= ESIMD_REPEAT_COUNT && N >= ESIMD_EXEC_SIZE;
}

/**
 * Load weights for a K-tile to SLM with VNNI packing for ESIMD dpas.
 *
 * Loads and dequantizes Q4_0 weights from global memory to SLM in VNNI format.
 * VNNI layout for FP16: b[(k/2) * N * 2 + n * 2 + (k%2)]
 *
 * @param weights       Pointer to Q4_0 weight blocks
 * @param slm_offset    SLM byte offset for weights buffer
 * @param n_start       Starting N index
 * @param k_start       Starting K index for this tile
 * @param N             Total N dimension
 * @param K             Total K dimension
 * @param k_blocks_per_row Number of blocks per weight row
 * @param k_len         Valid K elements in this tile (may be < ESIMD_K_PER_DPAS)
 */
template <int TILE_M, int TILE_N>
SYCL_ESIMD_FUNCTION void load_weights_to_slm_vnni(
    const block_q4_0_unified * weights,
    uint32_t                   slm_offset,
    int64_t                    n_start,
    int64_t                    k_start,
    int64_t                    N,
    int64_t                    K,
    int                        k_blocks_per_row,
    int                        k_len) {

    // Load and dequantize weights with VNNI packing using vectorized dequantization
    esimd::simd<sycl::half, ESIMD_SLM_WEIGHTS_SIZE> w_vec = sycl::half(0.0f);

    #pragma unroll
    for (int n = 0; n < TILE_N; n++) {
        const int64_t n_global = n_start + n;
        if (n_global >= N) continue;

        // Use vectorized tile dequantization for this row
        esimd::simd<sycl::half, ESIMD_K_PER_DPAS> row_weights =
            dequant_q4_0_tile_vectorized<ESIMD_K_PER_DPAS>(
                weights, n_global, k_start, K, k_blocks_per_row, k_len);

        // Repack to VNNI layout: b[(k/2) * N * 2 + n * 2 + (k%2)]
        #pragma unroll
        for (int k = 0; k < ESIMD_K_PER_DPAS; k++) {
            const int vnni_idx = (k / 2) * (TILE_N * 2) + n * 2 + (k % 2);
            w_vec[vnni_idx] = (k < k_len) ? row_weights[k] : sycl::half(0.0f);
        }
    }

    // Store to SLM
    esimd::slm_block_store<sycl::half, ESIMD_SLM_WEIGHTS_SIZE>(slm_offset, w_vec);
}

/**
 * Load MXFP4 weights for a K-tile to SLM with VNNI packing for ESIMD dpas.
 *
 * @param weights       Pointer to MXFP4 weight blocks
 * @param slm_offset    SLM byte offset for weights buffer
 * @param n_start       Starting N index
 * @param k_start       Starting K index for this tile
 * @param N             Total N dimension
 * @param K             Total K dimension
 * @param k_blocks_per_row Number of blocks per weight row
 * @param k_len         Valid K elements in this tile (may be < ESIMD_K_PER_DPAS)
 */
template <int TILE_M, int TILE_N>
SYCL_ESIMD_FUNCTION void load_weights_to_slm_vnni_mxfp4(
    const block_mxfp4_unified * weights,
    uint32_t                    slm_offset,
    int64_t                     n_start,
    int64_t                     k_start,
    int64_t                     N,
    int64_t                     K,
    int                         k_blocks_per_row,
    int                         k_len) {

    // Load and dequantize MXFP4 weights with VNNI packing
    esimd::simd<sycl::half, ESIMD_SLM_WEIGHTS_SIZE> w_vec = sycl::half(0.0f);

    #pragma unroll
    for (int n = 0; n < TILE_N; n++) {
        const int64_t n_global = n_start + n;
        if (n_global >= N) continue;

        // Use vectorized tile dequantization for this row
        esimd::simd<sycl::half, ESIMD_K_PER_DPAS> row_weights =
            dequant_mxfp4_tile_vectorized<ESIMD_K_PER_DPAS>(
                weights, n_global, k_start, K, k_blocks_per_row, k_len);

        // Repack to VNNI layout: b[(k/2) * N * 2 + n * 2 + (k%2)]
        #pragma unroll
        for (int k = 0; k < ESIMD_K_PER_DPAS; k++) {
            const int vnni_idx = (k / 2) * (TILE_N * 2) + n * 2 + (k % 2);
            w_vec[vnni_idx] = (k < k_len) ? row_weights[k] : sycl::half(0.0f);
        }
    }

    // Store to SLM
    esimd::slm_block_store<sycl::half, ESIMD_SLM_WEIGHTS_SIZE>(slm_offset, w_vec);
}

/**
 * Generic weight loader dispatcher for ESIMD VNNI path.
 *
 * Dispatches to the appropriate weight loader based on quantization type.
 *
 * @param weights       Pointer to weight blocks (void*, cast internally)
 * @param slm_offset    SLM byte offset for weights buffer
 * @param n_start       Starting N index
 * @param k_start       Starting K index for this tile
 * @param N             Total N dimension
 * @param K             Total K dimension
 * @param k_blocks_per_row Number of blocks per weight row
 * @param k_len         Valid K elements in this tile
 * @param quant_type    Quantization type (QUANT_TYPE_Q4_0 or QUANT_TYPE_MXFP4)
 */
template <int TILE_M, int TILE_N>
SYCL_ESIMD_FUNCTION void load_weights_to_slm_vnni_generic(
    const void * weights,
    uint32_t     slm_offset,
    int64_t      n_start,
    int64_t      k_start,
    int64_t      N,
    int64_t      K,
    int          k_blocks_per_row,
    int          k_len,
    int          quant_type) {

    if (quant_type == QUANT_TYPE_MXFP4) {
        load_weights_to_slm_vnni_mxfp4<TILE_M, TILE_N>(
            static_cast<const block_mxfp4_unified *>(weights),
            slm_offset, n_start, k_start, N, K, k_blocks_per_row, k_len);
    } else {
        // Default: Q4_0
        load_weights_to_slm_vnni<TILE_M, TILE_N>(
            static_cast<const block_q4_0_unified *>(weights),
            slm_offset, n_start, k_start, N, K, k_blocks_per_row, k_len);
    }
}

/**
 * Load activations for a K-tile to SLM in row-major format.
 *
 * @param activations   Pointer to activation matrix
 * @param slm_offset    SLM byte offset for activations buffer
 * @param m_start       Starting M index
 * @param k_start       Starting K index for this tile
 * @param M             Total M dimension
 * @param K             Total K dimension
 * @param k_len         Valid K elements in this tile (may be < ESIMD_K_PER_DPAS)
 */
template <int TILE_M, int TILE_N>
SYCL_ESIMD_FUNCTION void load_activations_to_slm(
    const float * activations,
    uint32_t      slm_offset,
    int64_t       m_start,
    int64_t       k_start,
    int64_t       M,
    int64_t       K,   // Total K dimension (used to index activations)
    int           k_len) {

    esimd::simd<sycl::half, ESIMD_SLM_ACTS_SIZE> a_vec = sycl::half(0.0f);

    #pragma unroll
    for (int m = 0; m < TILE_M; m++) {
        const int64_t m_global = m_start + m;
        if (m_global >= M) continue;

        #pragma unroll
        for (int k = 0; k < ESIMD_K_PER_DPAS; k++) {
            if (k >= k_len) break;

            const int64_t k_global = k_start + k;
            const float act_f32 = activations[m_global * K + k_global];
            // Row-major: a[m * K_per + k]
            a_vec[m * ESIMD_K_PER_DPAS + k] = static_cast<sycl::half>(act_f32);
        }
    }

    // Store to SLM
    esimd::slm_block_store<sycl::half, ESIMD_SLM_ACTS_SIZE>(slm_offset, a_vec);
}

/**
 * ESIMD FP16 matmul kernel with double-buffering (Phase 4).
 *
 * GGML Convention: dst[m,n] = sum_k(src0[n,k] * src1[m,k])
 * Computes: output[M,N] = activations[M,K] @ weights[N,K]^T
 *
 * Uses double-buffering to overlap memory loads with compute:
 * 1. Pre-load first K-tile into buffer 0
 * 2. For each K-tile:
 *    - Load next K-tile into alternate buffer (if not last)
 *    - Execute dpas on current buffer
 *    - Swap buffers
 * 3. Write final accumulator to output
 *
 * @tparam TILE_M  M tile size (must be 8 for dpas)
 * @tparam TILE_N  N tile size (must be 16 for dpas)
 * @param args     Kernel arguments
 * @param m_start  Starting M index for this work-item's tile
 * @param n_start  Starting N index for this work-item's tile
 * @param cfg      XMX configuration (use_double_buffer flag)
 */
template <int TILE_M, int TILE_N>
SYCL_ESIMD_FUNCTION void esimd_matmul_fp16_double_buffered_impl(
    const UnifiedKernelArgs args,
    int64_t                 m_start,
    int64_t                 n_start,
    const XMXConfig &       /* cfg - reserved for future tuning */) {

    // Validate tile sizes match dpas requirements
    static_assert(TILE_M == ESIMD_REPEAT_COUNT, "TILE_M must be 8 for dpas");
    static_assert(TILE_N == ESIMD_EXEC_SIZE, "TILE_N must be 16 for dpas");

    // Boundary checking: return early if entire tile is out of bounds
    if (m_start >= args.M || n_start >= args.N) {
        return;
    }

    // Number of blocks per weight row (both Q4_0 and MXFP4 have 32 elements/block)
    constexpr int QK = 32;
    const int k_blocks_per_row = static_cast<int>(args.K / QK);

    // Initialize SLM for double-buffering
    esimd::slm_init<ESIMD_SLM_TOTAL_BYTES>();

    // Initialize accumulator: [8 x 16] float
    esimd::simd<float, ESIMD_ACC_SIZE> acc = 0.0f;

    // Number of K tiles (each dpas processes 16 K elements)
    const int64_t k_tiles = (args.K + ESIMD_K_PER_DPAS - 1) / ESIMD_K_PER_DPAS;

    // Edge case: Single K-tile (K <= K_TILE)
    // No overlap opportunity, but still works correctly
    if (k_tiles <= 1) {
        // Just load and compute directly without double-buffering
        const int64_t k_start = 0;
        const int64_t k_remaining = args.K;
        const int k_len = static_cast<int>(k_remaining < ESIMD_K_PER_DPAS ? k_remaining : ESIMD_K_PER_DPAS);

        // Load weights and activations to buffer 0
        load_weights_to_slm_vnni_generic<TILE_M, TILE_N>(
            args.weights, ESIMD_SLM_BUF0_WEIGHTS,
            n_start, k_start, args.N, args.K, k_blocks_per_row, k_len, args.quant_type);
        load_activations_to_slm<TILE_M, TILE_N>(
            args.activations, ESIMD_SLM_BUF0_ACTS,
            m_start, k_start, args.M, args.K, k_len);

        // Fence to ensure SLM writes complete before reads
        esimd::fence<esimd::fence_mask::local_barrier>();

        // Load from SLM to registers
        esimd::simd<sycl::half, ESIMD_SLM_WEIGHTS_SIZE> b_vec =
            esimd::slm_block_load<sycl::half, ESIMD_SLM_WEIGHTS_SIZE>(ESIMD_SLM_BUF0_WEIGHTS);
        esimd::simd<sycl::half, ESIMD_SLM_ACTS_SIZE> a_vec =
            esimd::slm_block_load<sycl::half, ESIMD_SLM_ACTS_SIZE>(ESIMD_SLM_BUF0_ACTS);

        // Execute dpas
        acc = xmx::dpas<ESIMD_SYSTOLIC_DEPTH, ESIMD_REPEAT_COUNT,
                        float, float, sycl::half, sycl::half>(acc, b_vec, a_vec);
    } else {
        // ========================================================================
        // Double-buffered K-loop for memory/compute overlap
        // ========================================================================

        // Pre-load first K-tile into buffer 0
        {
            const int64_t k_start = 0;
            const int64_t k_remaining = args.K;
            const int k_len = static_cast<int>(k_remaining < ESIMD_K_PER_DPAS ? k_remaining : ESIMD_K_PER_DPAS);

            load_weights_to_slm_vnni_generic<TILE_M, TILE_N>(
                args.weights, ESIMD_SLM_BUF0_WEIGHTS,
                n_start, k_start, args.N, args.K, k_blocks_per_row, k_len, args.quant_type);
            load_activations_to_slm<TILE_M, TILE_N>(
                args.activations, ESIMD_SLM_BUF0_ACTS,
                m_start, k_start, args.M, args.K, k_len);
        }

        // Fence after pre-loading buffer 0
        esimd::fence<esimd::fence_mask::local_barrier>();

        int buf_compute = 0;  // Start with buffer 0

        // Get prefetch distance from kernel args (set by host-side launch_unified_matmul)
        // Double-buffering loads kt+1, so prefetch targets kt+prefetch_depth
        const int prefetch_distance = args.prefetch_depth;

        // Main K-loop with double-buffering
        for (int64_t kt = 0; kt < k_tiles; kt++) {
            // ================================================================
            // Prefetch: Look ahead beyond double-buffer distance
            // ================================================================
            // Double-buffering loads kt+1. Prefetch targets kt+prefetch_depth.
            // This gets data into cache before the load is needed.
            if (prefetch_distance > 1 && kt + prefetch_distance < k_tiles) {
                const int64_t prefetch_k_start = (kt + prefetch_distance) * ESIMD_K_PER_DPAS;

                // Prefetch weights for future K-tile
                #pragma unroll
                for (int n = 0; n < TILE_N; n++) {
                    prefetch_weights_block_generic<ESIMD_K_PER_DPAS>(
                        args.weights, n_start + n, prefetch_k_start, args.K, k_blocks_per_row, args.N, args.quant_type);
                }

                // Prefetch activations for future K-tile
                #pragma unroll
                for (int m = 0; m < TILE_M; m++) {
                    prefetch_activations_block<ESIMD_K_PER_DPAS>(
                        args.activations, m_start + m, prefetch_k_start, args.K, args.M);
                }
            }

            // Determine current buffer offsets
            const uint32_t compute_w_off = (buf_compute == 0) ? ESIMD_SLM_BUF0_WEIGHTS : ESIMD_SLM_BUF1_WEIGHTS;
            const uint32_t compute_a_off = (buf_compute == 0) ? ESIMD_SLM_BUF0_ACTS : ESIMD_SLM_BUF1_ACTS;

            // Determine load buffer offsets (alternate buffer)
            const uint32_t load_w_off = (buf_compute == 0) ? ESIMD_SLM_BUF1_WEIGHTS : ESIMD_SLM_BUF0_WEIGHTS;
            const uint32_t load_a_off = (buf_compute == 0) ? ESIMD_SLM_BUF1_ACTS : ESIMD_SLM_BUF0_ACTS;

            // Load from SLM to registers (from compute buffer)
            esimd::simd<sycl::half, ESIMD_SLM_WEIGHTS_SIZE> b_vec =
                esimd::slm_block_load<sycl::half, ESIMD_SLM_WEIGHTS_SIZE>(compute_w_off);
            esimd::simd<sycl::half, ESIMD_SLM_ACTS_SIZE> a_vec =
                esimd::slm_block_load<sycl::half, ESIMD_SLM_ACTS_SIZE>(compute_a_off);

            // Prefetch/load next K-tile into alternate buffer (if not last iteration)
            if (kt + 1 < k_tiles) {
                const int64_t next_k_start = (kt + 1) * ESIMD_K_PER_DPAS;
                const int64_t next_k_remaining = args.K - next_k_start;
                const int next_k_len = static_cast<int>(next_k_remaining < ESIMD_K_PER_DPAS ? next_k_remaining : ESIMD_K_PER_DPAS);

                load_weights_to_slm_vnni_generic<TILE_M, TILE_N>(
                    args.weights, load_w_off,
                    n_start, next_k_start, args.N, args.K, k_blocks_per_row, next_k_len, args.quant_type);
                load_activations_to_slm<TILE_M, TILE_N>(
                    args.activations, load_a_off,
                    m_start, next_k_start, args.M, args.K, next_k_len);
            }

            // Execute dpas on current buffer's data
            acc = xmx::dpas<ESIMD_SYSTOLIC_DEPTH, ESIMD_REPEAT_COUNT,
                            float, float, sycl::half, sycl::half>(acc, b_vec, a_vec);

            // Fence after SLM writes (ensure next iteration's loads are visible)
            if (kt + 1 < k_tiles) {
                esimd::fence<esimd::fence_mask::local_barrier>();
            }

            // Swap buffers
            buf_compute = 1 - buf_compute;
        }
    }

    // Write output with boundary checking
    // Output layout: acc[m * TILE_N + n] = [8 x 16] row-major
    #pragma unroll
    for (int m = 0; m < TILE_M; m++) {
        const int64_t m_global = m_start + m;
        if (m_global >= args.M) continue;

        #pragma unroll
        for (int n = 0; n < TILE_N; n++) {
            const int64_t n_global = n_start + n;
            if (n_global >= args.N) continue;

            args.output[m_global * args.N + n_global] = acc[m * TILE_N + n];
        }
    }
}

/**
 * ESIMD FP16 matmul kernel using xmx::dpas instruction.
 *
 * GGML Convention: dst[m,n] = sum_k(src0[n,k] * src1[m,k])
 * Computes: output[M,N] = activations[M,K] @ weights[N,K]^T
 *
 * Uses ESIMD xmx::dpas for hardware-accelerated matrix multiplication.
 * Each work-item processes one 8x16 output tile using dpas instruction.
 *
 * dpas layout requirements:
 * - A matrix (activations): [Repeat x K] = [8 x 16] half, row-major packed
 * - B matrix (weights): [K x ExecSize] = [16 x 16] half, VNNI-like layout
 * - Output: [Repeat x ExecSize] = [8 x 16] float accumulator
 *
 * Work distribution:
 * - 2D grid: [ceil(M/8), ceil(N/16)]
 * - Each work-item handles one 8x16 output tile
 *
 * @tparam TILE_M  M tile size (must be 8 for dpas)
 * @tparam TILE_N  N tile size (must be 16 for dpas)
 */
template <int TILE_M, int TILE_N>
SYCL_ESIMD_FUNCTION void esimd_matmul_fp16_kernel_impl(
    const UnifiedKernelArgs args,
    int64_t                 m_start,
    int64_t                 n_start) {

    // Validate tile sizes match dpas requirements
    static_assert(TILE_M == ESIMD_REPEAT_COUNT, "TILE_M must be 8 for dpas");
    static_assert(TILE_N == ESIMD_EXEC_SIZE, "TILE_N must be 16 for dpas");

    // Boundary checking: return early if entire tile is out of bounds
    if (m_start >= args.M || n_start >= args.N) {
        return;
    }

    // Number of Q4_0 blocks per weight row
    const int k_blocks_per_row = static_cast<int>(args.K / UNIFIED_QK4_0);

    // Cast weight pointer
    const block_q4_0_unified * weights = static_cast<const block_q4_0_unified *>(args.weights);

    // Initialize accumulator: [8 x 16] float
    esimd::simd<float, ESIMD_ACC_SIZE> acc = 0.0f;

    // Number of K tiles (each dpas processes 16 K elements)
    const int64_t k_tiles = (args.K + ESIMD_K_PER_DPAS - 1) / ESIMD_K_PER_DPAS;

    // K-loop: iterate over K dimension in tiles of 16
    for (int64_t kt = 0; kt < k_tiles; kt++) {
        const int64_t k_start = kt * ESIMD_K_PER_DPAS;
        // Calculate remaining K elements (avoid sycl::min which is not supported in ESIMD)
        const int64_t k_remaining = args.K - k_start;
        const int k_len = static_cast<int>(k_remaining < ESIMD_K_PER_DPAS ? k_remaining : ESIMD_K_PER_DPAS);

        // ============================================================
        // Load and dequantize weights into B matrix with VNNI packing
        // Using vectorized dequantization for improved throughput.
        //
        // dpas computes: C[m,n] += sum_k(A[m,k] * B[k,n])
        // GGML wants: dst[m,n] = sum_k(activations[m,k] * weights[n,k])
        // So B[k,n] = weights[n,k] (transpose)
        //
        // For FP16 dpas, B matrix needs VNNI-like layout:
        // B_vnni[k/2 * N * 2 + n * 2 + k%2] = B[k,n]
        // This groups consecutive K values together for systolic array
        // ============================================================
        esimd::simd<sycl::half, ESIMD_B_SIZE> b_vec = sycl::half(0.0f);

        // Use vectorized dequantization for each weight row, then repack to VNNI
        #pragma unroll
        for (int n = 0; n < TILE_N; n++) {
            const int64_t n_global = n_start + n;
            if (n_global >= args.N) continue;

            // Vectorized tile dequantization for this row
            esimd::simd<sycl::half, ESIMD_K_PER_DPAS> row_weights =
                dequant_q4_0_tile_vectorized<ESIMD_K_PER_DPAS>(
                    weights, n_global, k_start, args.K, k_blocks_per_row, k_len);

            // Repack to VNNI layout: b[(k/2) * N * 2 + n * 2 + (k%2)]
            #pragma unroll
            for (int k = 0; k < ESIMD_K_PER_DPAS; k++) {
                const int vnni_idx = (k / 2) * (TILE_N * 2) + n * 2 + (k % 2);
                b_vec[vnni_idx] = (k < k_len) ? row_weights[k] : sycl::half(0.0f);
            }
        }

        // ============================================================
        // Load activations into A matrix [Repeat x K] = [8 x 16]
        // A is stored row-major: a[m * K_per + k]
        // ============================================================
        esimd::simd<sycl::half, ESIMD_A_SIZE> a_vec = sycl::half(0.0f);

        #pragma unroll
        for (int m = 0; m < TILE_M; m++) {
            const int64_t m_global = m_start + m;
            if (m_global >= args.M) continue;

            #pragma unroll
            for (int k = 0; k < ESIMD_K_PER_DPAS; k++) {
                if (k >= k_len) break;

                const int64_t k_global = k_start + k;
                const float act_f32 = args.activations[m_global * args.K + k_global];
                // A layout for dpas: a[m * K_per + k]
                a_vec[m * ESIMD_K_PER_DPAS + k] = static_cast<sycl::half>(act_f32);
            }
        }

        // ============================================================
        // Execute dpas: acc += A @ B (computes C[m,n] += sum_k(A[m,k] * B[k,n]))
        //
        // dpas<SystolicDepth, RepeatCount, AccType, CType, BType, AType>
        // Note: operand order is dpas(acc, B, A) - B before A!
        //
        // dpas computes: C[m,n] += sum_k(A[m,k] * B[k,n])
        // A layout: row-major, a[m * K + k] where m=0..7, k=0..15
        // B layout: VNNI-packed, b[(k/2) * N * 2 + n * 2 + (k%2)] for k=0..15, n=0..15
        // ============================================================
        acc = xmx::dpas<ESIMD_SYSTOLIC_DEPTH, ESIMD_REPEAT_COUNT,
                        float, float, sycl::half, sycl::half>(acc, b_vec, a_vec);
    }

    // Write output with boundary checking
    // Output layout: acc[m * TILE_N + n] = [8 x 16] row-major
    #pragma unroll
    for (int m = 0; m < TILE_M; m++) {
        const int64_t m_global = m_start + m;
        if (m_global >= args.M) continue;

        #pragma unroll
        for (int n = 0; n < TILE_N; n++) {
            const int64_t n_global = n_start + n;
            if (n_global >= args.N) continue;

            args.output[m_global * args.N + n_global] = acc[m * TILE_N + n];
        }
    }
}

// =============================================================================
// ESIMD dpas Kernel - INT8 Path (Phase 3)
// =============================================================================
// Uses INT8 dpas with dynamic quantization for both weights and activations.
//
// Quantization approach:
// 1. Dequantize Q4_0 weights to FP values: w_fp = (nibble - 8) * d
// 2. Find max-abs per N column per K-tile for weights
// 3. Quantize weights to INT8: w_int8 = w_fp * 127 / w_max_abs
// 4. Find max-abs per M row for activations (across all K)
// 5. Quantize activations to INT8: a_int8 = a * 127 / a_max_abs
// 6. Execute dpas: int32_acc = sum_k(w_int8 * a_int8)
// 7. Dequantize result: fp_result = int32_acc * w_scale * a_scale / (127 * 127)
//
// Key differences from FP16:
// - K-tile = 32 (not 16)
// - dpas outputs INT32 accumulator
// - Both weights and activations dynamically quantized
//
// IMPORTANT: INT8 is LOSSY - not bit-exact with FP16/FP32 path!

/**
 * ESIMD INT8 matmul kernel using xmx::dpas instruction.
 *
 * GGML Convention: dst[m,n] = sum_k(src0[n,k] * src1[m,k])
 * Computes: output[M,N] = activations[M,K] @ weights[N,K]^T
 *
 * Work distribution:
 * - 2D grid: [ceil(M/8), ceil(N/16)]
 * - Each work-item handles one 8x16 output tile
 *
 * @tparam TILE_M  M tile size (must be 8 for dpas)
 * @tparam TILE_N  N tile size (must be 16 for dpas)
 */
template <int TILE_M, int TILE_N>
SYCL_ESIMD_FUNCTION void esimd_matmul_int8_kernel_impl(
    const UnifiedKernelArgs args,
    int64_t                 m_start,
    int64_t                 n_start) {

    // Validate tile sizes match dpas requirements
    static_assert(TILE_M == ESIMD_REPEAT_COUNT, "TILE_M must be 8 for dpas");
    static_assert(TILE_N == ESIMD_EXEC_SIZE, "TILE_N must be 16 for dpas");

    // Boundary checking: return early if entire tile is out of bounds
    if (m_start >= args.M || n_start >= args.N) {
        return;
    }

    // Number of Q4_0 blocks per weight row
    const int k_blocks_per_row = static_cast<int>(args.K / UNIFIED_QK4_0);

    // Cast weight pointer
    const block_q4_0_unified * weights = static_cast<const block_q4_0_unified *>(args.weights);

    // Initialize FP32 accumulator for final result: [8 x 16]
    // We accumulate in FP32 because each K-tile has different scales
    esimd::simd<float, ESIMD_ACC_SIZE> fp_acc = 0.0f;

    // Number of K tiles (each dpas processes 32 K elements for INT8)
    const int64_t k_tiles = (args.K + ESIMD_K_PER_DPAS_INT8 - 1) / ESIMD_K_PER_DPAS_INT8;

    // ========================================================================
    // Step 1: Compute per-row activation scales (max-abs for each M row)
    // This needs to scan all K elements for each of the 8 rows in this tile
    // ========================================================================
    esimd::simd<float, TILE_M> act_max_abs = 0.0f;

    #pragma unroll
    for (int m = 0; m < TILE_M; m++) {
        const int64_t m_global = m_start + m;
        if (m_global >= args.M) {
            act_max_abs[m] = 1.0f;  // Dummy scale for out-of-bounds rows
            continue;
        }

        float max_abs = 0.0f;
        for (int64_t k = 0; k < args.K; k++) {
            float val = args.activations[m_global * args.K + k];
            float abs_val = (val >= 0.0f) ? val : -val;
            if (abs_val > max_abs) {
                max_abs = abs_val;
            }
        }
        act_max_abs[m] = (max_abs > 1e-10f) ? max_abs : 1.0f;
    }

    // ========================================================================
    // K-loop: iterate over K dimension in tiles of 32
    // ========================================================================
    for (int64_t kt = 0; kt < k_tiles; kt++) {
        const int64_t k_start_local = kt * ESIMD_K_PER_DPAS_INT8;
        const int64_t k_remaining = args.K - k_start_local;
        const int k_len = static_cast<int>(k_remaining < ESIMD_K_PER_DPAS_INT8 ? k_remaining : ESIMD_K_PER_DPAS_INT8);

        // ====================================================================
        // Step 2a: Dequantize Q4_0 weights to FP and find max-abs per N column
        // ====================================================================
        // Temporary storage for dequantized weights [TILE_N x K_TILE]
        float w_fp[TILE_N][ESIMD_K_PER_DPAS_INT8];
        esimd::simd<float, TILE_N> w_max_abs = 0.0f;

        #pragma unroll
        for (int n = 0; n < TILE_N; n++) {
            const int64_t n_global = n_start + n;

            #pragma unroll
            for (int k = 0; k < ESIMD_K_PER_DPAS_INT8; k++) {
                if (n_global >= args.N || k >= k_len) {
                    w_fp[n][k] = 0.0f;
                    continue;
                }

                const int64_t k_global = k_start_local + k;
                const int k_block_idx = static_cast<int>(k_global / UNIFIED_QK4_0);
                const int idx_in_block = static_cast<int>(k_global % UNIFIED_QK4_0);

                const int block_idx = static_cast<int>(n_global * k_blocks_per_row + k_block_idx);
                const block_q4_0_unified * blk = &weights[block_idx];
                const float d = static_cast<float>(blk->d);

                int qs_val;
                if (idx_in_block < 16) {
                    qs_val = blk->qs[idx_in_block] & 0x0F;
                } else {
                    qs_val = blk->qs[idx_in_block - 16] >> 4;
                }

                // Dequantize to FP: w = (nibble - 8) * d
                const float w_val = static_cast<float>(qs_val - 8) * d;
                w_fp[n][k] = w_val;

                // Track max-abs for this column
                float abs_w = (w_val >= 0.0f) ? w_val : -w_val;
                if (abs_w > w_max_abs[n]) {
                    w_max_abs[n] = abs_w;
                }
            }
        }

        // Ensure no divide by zero
        #pragma unroll
        for (int n = 0; n < TILE_N; n++) {
            if (w_max_abs[n] < 1e-10f) {
                w_max_abs[n] = 1.0f;
            }
        }

        // ====================================================================
        // Step 2b: Quantize weights to INT8 with VNNI packing
        // ====================================================================
        esimd::simd<int8_t, ESIMD_B_SIZE_INT8> b_vec = 0;

        #pragma unroll
        for (int n = 0; n < TILE_N; n++) {
            #pragma unroll
            for (int k = 0; k < ESIMD_K_PER_DPAS_INT8; k++) {
                // Quantize: w_int8 = w_fp * 127 / max_abs
                float qval = w_fp[n][k] * 127.0f / w_max_abs[n];
                if (qval > 127.0f) qval = 127.0f;
                if (qval < -127.0f) qval = -127.0f;
                const int8_t w_int8 = static_cast<int8_t>(qval);

                // VNNI layout for INT8: b[(k/4) * N * 4 + n * 4 + (k%4)]
                const int vnni_idx = (k / 4) * (TILE_N * 4) + n * 4 + (k % 4);
                b_vec[vnni_idx] = w_int8;
            }
        }

        // ====================================================================
        // Step 3: Quantize activations to INT8
        // ====================================================================
        esimd::simd<int8_t, ESIMD_A_SIZE_INT8> a_vec = 0;

        #pragma unroll
        for (int m = 0; m < TILE_M; m++) {
            const int64_t m_global = m_start + m;
            if (m_global >= args.M) continue;

            const float scale_inv = 127.0f / act_max_abs[m];

            #pragma unroll
            for (int k = 0; k < ESIMD_K_PER_DPAS_INT8; k++) {
                if (k >= k_len) break;

                const int64_t k_global = k_start_local + k;
                const float act_f32 = args.activations[m_global * args.K + k_global];

                float qval = act_f32 * scale_inv;
                if (qval > 127.0f) qval = 127.0f;
                if (qval < -127.0f) qval = -127.0f;

                const int8_t a_int8 = static_cast<int8_t>(qval);
                a_vec[m * ESIMD_K_PER_DPAS_INT8 + k] = a_int8;
            }
        }

        // ====================================================================
        // Step 4: Execute dpas: int32_acc = sum_k(w_int8 * a_int8)
        // ====================================================================
        esimd::simd<int32_t, ESIMD_ACC_SIZE_INT8> int32_acc = 0;
        int32_acc = xmx::dpas<ESIMD_SYSTOLIC_DEPTH, ESIMD_REPEAT_COUNT,
                              int32_t, int32_t, int8_t, int8_t>(int32_acc, b_vec, a_vec);

        // ====================================================================
        // Step 5: Dequantize and accumulate to FP32
        // fp_result += int32_acc * (w_max_abs / 127) * (a_max_abs / 127)
        //            = int32_acc * w_max_abs * a_max_abs / (127 * 127)
        // ====================================================================
        constexpr float scale_denom = 127.0f * 127.0f;

        #pragma unroll
        for (int m = 0; m < TILE_M; m++) {
            const float a_scale = act_max_abs[m] / scale_denom;

            #pragma unroll
            for (int n = 0; n < TILE_N; n++) {
                const int idx = m * TILE_N + n;
                const float w_scale = w_max_abs[n];
                const float combined_scale = w_scale * a_scale;

                fp_acc[idx] += static_cast<float>(int32_acc[idx]) * combined_scale;
            }
        }
    }

    // ========================================================================
    // Write output with boundary checking
    // ========================================================================
    #pragma unroll
    for (int m = 0; m < TILE_M; m++) {
        const int64_t m_global = m_start + m;
        if (m_global >= args.M) continue;

        #pragma unroll
        for (int n = 0; n < TILE_N; n++) {
            const int64_t n_global = n_start + n;
            if (n_global >= args.N) continue;

            args.output[m_global * args.N + n_global] = fp_acc[m * TILE_N + n];
        }
    }
}

#if GGML_SYCL_COOPERATIVE_KERNEL_ENABLED
// =============================================================================
// Cooperative ESIMD FP16 Kernel Implementation
// =============================================================================
// Uses multiple work-items with split barriers for work-group level loading.
// Each sub-group (16 work-items) owns one 8x16 output tile.
// All work-items cooperate on loading data to SLM using strided pattern.
//
// Key differences from single work-item kernel:
// 1. Work-group size: 32 (2 sub-groups) vs 1
// 2. Cooperative loading: All work-items load together, then barrier
// 3. Each sub-group computes its own output tile after loading
// 4. Uses SPIR-V split barriers for work-group synchronization (Arc-compatible)

/**
 * Cooperative ESIMD FP16 matmul kernel using xmx::dpas with work-group level loading.
 *
 * GGML Convention: dst[m,n] = sum_k(src0[n,k] * src1[m,k])
 * Computes: output[M,N] = activations[M,K] @ weights[N,K]^T
 *
 * Work distribution:
 * - 2D grid of work-groups, each handles COOP_WG_M x COOP_WG_N output region
 * - Each work-group has 32 work-items (2 sub-groups of 16)
 * - Sub-group 0: handles M-rows [0..7], sub-group 1: handles M-rows [8..15]
 * - All work-items cooperate on loading weights [16 x K_TILE] and activations [16 x K_TILE]
 *
 * Named barrier usage:
 * - barrier 0: Synchronize after cooperative loading
 * - esimd::fence ensures memory visibility before barrier
 *
 * Block operations strategy (for performance):
 * - Loading phase: Each work-item loads one row (16 half) and uses slm_block_store
 * - Compute phase: Use slm_block_load to fetch entire 8x16 or 16x16 tiles
 *
 * @param item        ND-item for work distribution
 * @param args        Kernel arguments
 * @param local_id    Linear work-item ID within work-group [0..31]
 * @param sg_id       Sub-group ID within work-group [0..1]
 * @param lane        Lane ID within sub-group [0..15]
 */
template <int WG_SIZE>
SYCL_ESIMD_FUNCTION void esimd_matmul_fp16_cooperative_impl(
    sycl::nd_item<2>            item,
    const UnifiedKernelArgs     args,
    int                         local_id,
    int                         sg_id,
    int                         lane) {

    // Compile-time validation of WG_SIZE constraints
    static_assert(WG_SIZE % 16 == 0, "WG_SIZE must be multiple of 16 for sub-group size");
    static_assert(WG_SIZE >= 32, "WG_SIZE must be at least 32 (2 sub-groups)");

    // Work-group coordinates (each work-group handles COOP_WG_M x COOP_WG_N output)
    const int wg_row = item.get_group(0);  // M dimension
    const int wg_col = item.get_group(1);  // N dimension

    // Global output coordinates for this work-group
    const int64_t m_wg_start = wg_row * COOP_WG_M;  // Starting output row for work-group
    const int64_t n_wg_start = wg_col * COOP_WG_N;  // Starting output column for work-group

    // Boundary check: skip if entire work-group is out of bounds
    if (m_wg_start >= args.M || n_wg_start >= args.N) {
        return;
    }

    // This sub-group's output tile coordinates
    // Sub-group 0: M-rows [0..7], Sub-group 1: M-rows [8..15]
    const int64_t m_sg_start = m_wg_start + sg_id * ESIMD_REPEAT_COUNT;
    const int64_t n_sg_start = n_wg_start;  // All sub-groups handle same N range

    // Check if this sub-group has work (boundary check)
    const bool sg_has_work = (m_sg_start < args.M && n_sg_start < args.N);

    // Number of Q4_0 blocks per weight row
    const int k_blocks_per_row = static_cast<int>(args.K / UNIFIED_QK4_0);

    // Cast weight pointer
    const block_q4_0_unified * weights = static_cast<const block_q4_0_unified *>(args.weights);

    // Initialize SLM for this work-group
    // Layout: [weights: 16 x 16 half][activations: 16 x 16 half]
    esimd::slm_init<COOP_SLM_TOTAL_BYTES>();

    // Initialize accumulator for this sub-group's output tile: [8 x 16] float
    esimd::simd<float, ESIMD_ACC_SIZE> acc = 0.0f;

    // Number of K tiles
    const int64_t k_tiles = (args.K + ESIMD_K_PER_DPAS - 1) / ESIMD_K_PER_DPAS;

    // Get prefetch distance from kernel args (set by host-side launch_unified_matmul)
    // args.prefetch_depth is set from get_prefetch_distance() on the host side
    const int prefetch_distance = args.prefetch_depth;

    // K-loop: iterate over K dimension in tiles of 16
    for (int64_t kt = 0; kt < k_tiles; kt++) {
        const int64_t k_start = kt * ESIMD_K_PER_DPAS;
        const int64_t k_remaining = args.K - k_start;
        const int k_len = static_cast<int>(k_remaining < ESIMD_K_PER_DPAS ? k_remaining : ESIMD_K_PER_DPAS);

        // ================================================================
        // Phase 0: LSC Prefetch for Future K-Tiles (Memory/Compute Overlap)
        // ================================================================
        // Prefetch future tile data while loading current tile.
        // Uses cache hints based on data reuse patterns:
        // - Weights: streaming (used once per output element)
        // - Activations: cached (reused across N columns)
        //
        // Bounds checking: Only prefetch if future tile exists
        // Guard: if (kt + prefetch_distance < k_tiles) prefetch(...)
        if (prefetch_distance > 0 && kt + prefetch_distance < k_tiles) {
            const int64_t prefetch_k_start = (kt + prefetch_distance) * ESIMD_K_PER_DPAS;

            // Prefetch weights for future K-tile (work-items 0..15)
            prefetch_weights_cooperative<ESIMD_K_PER_DPAS, COOP_WG_N>(
                weights, n_wg_start, prefetch_k_start, args,
                k_blocks_per_row, local_id, COOP_WG_N);

            // Prefetch activations for future K-tile (work-items 16..31)
            prefetch_activations_cooperative<ESIMD_K_PER_DPAS, COOP_WG_M>(
                args.activations, m_wg_start, prefetch_k_start, args,
                local_id, COOP_WG_N, COOP_WG_M);
        }

        // ================================================================
        // Phase 1: Cooperative Loading with Block Operations
        // ================================================================
        // SLM layout: [weights: 16 rows x 16 cols][activations: 16 rows x 16 cols]
        // 32 work-items load 32 rows total (16 weight rows + 16 activation rows)
        // Each work-item loads one row (16 half values) using block store
        //
        // Work-item assignment:
        // - local_id 0-15: Load weight rows 0-15
        // - local_id 16-31: Load activation rows 0-15

        // Each work-item loads exactly one row of 16 half values
        constexpr int ELEMS_PER_ROW = ESIMD_K_PER_DPAS;  // 16

        if (local_id < COOP_WG_N) {
            // Work-items 0-15: Load weight row local_id using vectorized dequantization
            const int n_off = local_id;
            const int64_t n_global = n_wg_start + n_off;

            // Use vectorized tile dequantization for this row
            esimd::simd<sycl::half, ELEMS_PER_ROW> w_row = sycl::half(0.0f);

            if (n_global < args.N) {
                // Vectorized dequantization: load and unpack entire tile at once
                w_row = dequant_q4_0_tile_vectorized<ELEMS_PER_ROW>(
                    weights, n_global, k_start, args.K, k_blocks_per_row, k_len);

                // Zero out elements beyond k_len for boundary handling
                #pragma unroll
                for (int k = 0; k < ELEMS_PER_ROW; k++) {
                    if (k >= k_len) {
                        w_row[k] = sycl::half(0.0f);
                    }
                }
            }

            // Block store entire row to SLM
            const uint32_t slm_off = COOP_SLM_WEIGHTS_OFFSET + n_off * ELEMS_PER_ROW * sizeof(sycl::half);
            esimd::slm_block_store<sycl::half, ELEMS_PER_ROW>(slm_off, w_row);

        } else {
            // Work-items 16-31: Load activation row (local_id - 16)
            const int m_off = local_id - COOP_WG_N;  // 0-15
            const int64_t m_global = m_wg_start + m_off;

            // Build row vector in registers, then block store
            esimd::simd<sycl::half, ELEMS_PER_ROW> a_row = sycl::half(0.0f);

            if (m_global < args.M) {
                #pragma unroll
                for (int k = 0; k < ELEMS_PER_ROW; k++) {
                    if (k >= k_len) break;
                    const int64_t k_global = k_start + k;
                    a_row[k] = static_cast<sycl::half>(args.activations[m_global * args.K + k_global]);
                }
            }

            // Block store entire row to SLM
            const uint32_t slm_off = COOP_SLM_ACTS_OFFSET + m_off * ELEMS_PER_ROW * sizeof(sycl::half);
            esimd::slm_block_store<sycl::half, ELEMS_PER_ROW>(slm_off, a_row);
        }

        // Split barrier: signal that loading is complete, then wait for all loaders
        // Using split barriers (SPV_INTEL_split_barrier) for better performance on Arc
        esimd::fence<esimd::fence_mask::local_barrier>();
        split_barrier_arrive(ScopeWorkgroup, SemanticsWGMem);
        split_barrier_wait(ScopeWorkgroup, SemanticsWGMem);

        // ================================================================
        // Phase 2: Compute with Block Loads
        // ================================================================
        if (sg_has_work) {
            // Block load activations for this sub-group's M-rows [sg_id*8 .. sg_id*8+7]
            // SLM layout is row-major: row m at offset m * 16 * sizeof(half)
            // A layout for dpas: a[m * K_per + k] for m in [0..7], k in [0..15]
            const int m_base = sg_id * ESIMD_REPEAT_COUNT;  // 0 or 8
            const uint32_t a_slm_base = COOP_SLM_ACTS_OFFSET + m_base * ELEMS_PER_ROW * sizeof(sycl::half);

            // Block load all 8 rows (8 x 16 = 128 half values)
            esimd::simd<sycl::half, ESIMD_A_SIZE> a_vec =
                esimd::slm_block_load<sycl::half, ESIMD_A_SIZE>(a_slm_base);

            // Handle boundary: zero out rows beyond M
            #pragma unroll
            for (int m = 0; m < ESIMD_REPEAT_COUNT; m++) {
                const int64_t m_global = m_wg_start + m_base + m;
                if (m_global >= args.M) {
                    #pragma unroll
                    for (int k = 0; k < ESIMD_K_PER_DPAS; k++) {
                        a_vec[m * ESIMD_K_PER_DPAS + k] = sycl::half(0.0f);
                    }
                }
            }

            // Block load weights (all 16 rows = 256 half values)
            // Then repack to VNNI format for dpas
            esimd::simd<sycl::half, COOP_SLM_WEIGHTS_SIZE> w_raw =
                esimd::slm_block_load<sycl::half, COOP_SLM_WEIGHTS_SIZE>(COOP_SLM_WEIGHTS_OFFSET);

            // Repack weights from row-major to VNNI format
            // Row-major: w_raw[n * 16 + k] for n in [0..15], k in [0..15]
            // VNNI: b_vec[(k/2) * N * 2 + n * 2 + (k%2)]
            esimd::simd<sycl::half, ESIMD_B_SIZE> b_vec;
            #pragma unroll
            for (int n = 0; n < ESIMD_EXEC_SIZE; n++) {
                #pragma unroll
                for (int k = 0; k < ESIMD_K_PER_DPAS; k++) {
                    const int vnni_idx = (k / 2) * (ESIMD_EXEC_SIZE * 2) + n * 2 + (k % 2);
                    b_vec[vnni_idx] = w_raw[n * ESIMD_K_PER_DPAS + k];
                }
            }

            // Handle boundary: zero out weights beyond N
            #pragma unroll
            for (int n = 0; n < ESIMD_EXEC_SIZE; n++) {
                const int64_t n_global = n_wg_start + n;
                if (n_global >= args.N) {
                    #pragma unroll
                    for (int k = 0; k < ESIMD_K_PER_DPAS; k++) {
                        const int vnni_idx = (k / 2) * (ESIMD_EXEC_SIZE * 2) + n * 2 + (k % 2);
                        b_vec[vnni_idx] = sycl::half(0.0f);
                    }
                }
            }

            // Execute dpas: acc += A @ B
            acc = xmx::dpas<ESIMD_SYSTOLIC_DEPTH, ESIMD_REPEAT_COUNT,
                            float, float, sycl::half, sycl::half>(acc, b_vec, a_vec);
        }

        // Split barrier before next K-tile (ensure all sub-groups done before overwriting SLM)
        if (kt + 1 < k_tiles) {
            esimd::fence<esimd::fence_mask::local_barrier>();
            split_barrier_arrive(ScopeWorkgroup, SemanticsWGMem);
            split_barrier_wait(ScopeWorkgroup, SemanticsWGMem);
        }
    }

    // ================================================================
    // Phase 3: Write output (each sub-group writes its tile)
    // ================================================================
    if (sg_has_work) {
        #pragma unroll
        for (int m = 0; m < ESIMD_REPEAT_COUNT; m++) {
            const int64_t m_global = m_sg_start + m;
            if (m_global >= args.M) continue;

            #pragma unroll
            for (int n = 0; n < ESIMD_EXEC_SIZE; n++) {
                const int64_t n_global = n_sg_start + n;
                if (n_global >= args.N) continue;

                args.output[m_global * args.N + n_global] = acc[m * ESIMD_EXEC_SIZE + n];
            }
        }
    }
}
#endif  // GGML_SYCL_COOPERATIVE_KERNEL_ENABLED

#if GGML_SYCL_LARGE_TILE_KERNEL_ENABLED
/**
 * Large-tile ESIMD kernel implementation for 32×32 output tiles.
 *
 * This kernel processes larger output tiles (32×32) using 64 work-items
 * (4 sub-groups of 16). Designed for prompt processing where larger batches
 * benefit from increased parallelism and better SLM utilization.
 *
 * Architecture:
 * - 64 work-items = 4 sub-groups of 16 lanes each
 * - 2×2 sub-group grid: 2 columns (covering 32 N) × 2 rows (covering 32 M)
 * - Each sub-group computes 2 stacked 8×16 dpas tiles = 16×16 output
 * - K-dimension tiled at 32 for efficient dpas chaining
 *
 * SLM Layout:
 * - Weights: [32 rows × 32 cols] = 1024 half = 2048 bytes
 * - Activations: [32 rows × 32 cols] = 1024 half = 2048 bytes
 * - Total: 4096 bytes (well under 64KB limit)
 *
 * Cooperative Loading (64 work-items):
 * - Work-items 0-31: Load weight rows (one row of 32 half each)
 * - Work-items 32-63: Load activation rows (one row of 32 half each)
 *
 * @tparam WG_SIZE  Work-group size (must be 64)
 * @param item      ND-item for work distribution
 * @param args      Kernel arguments
 * @param local_id  Linear work-item ID within work-group [0..63]
 * @param sg_id     Sub-group ID within work-group [0..3]
 * @param lane      Lane ID within sub-group [0..15]
 */
template <int WG_SIZE>
SYCL_ESIMD_FUNCTION void large_tile_esimd_kernel_impl(
    sycl::nd_item<2>            item,
    const UnifiedKernelArgs     args,
    int                         local_id,
    int                         sg_id,
    int                         /* lane */) {  // lane unused in this kernel

    // Compile-time validation of WG_SIZE constraints
    static_assert(WG_SIZE == 64, "Large-tile kernel requires WG_SIZE=64 (ESIMD hw limit)");
    static_assert(WG_SIZE % 16 == 0, "WG_SIZE must be multiple of 16 for sub-group size");

    // Work-group coordinates (each work-group handles LARGE_TILE_M × LARGE_TILE_N output)
    const int wg_row = item.get_group(0);  // M dimension
    const int wg_col = item.get_group(1);  // N dimension

    // Global output coordinates for this work-group
    const int64_t m_wg_start = wg_row * LARGE_TILE_M;  // Starting output row (32 rows per WG)
    const int64_t n_wg_start = wg_col * LARGE_TILE_N;  // Starting output column (64 cols per WG)

    // Boundary check: skip if entire work-group is out of bounds
    if (m_wg_start >= args.M || n_wg_start >= args.N) {
        return;
    }

    // Sub-group grid position (2 columns × 2 rows of sub-groups)
    // sg_id 0-1: row 0, sg_id 2-3: row 1
    const int sg_row = sg_id / LARGE_SG_COLS;  // 0 or 1
    const int sg_col = sg_id % LARGE_SG_COLS;  // 0 or 1

    // This sub-group's output tile coordinates
    // Each sub-group computes 2 stacked 8×16 tiles = 16×16 total output
    const int64_t m_sg_start = m_wg_start + sg_row * LARGE_SG_M;  // 16 rows per sg_row
    const int64_t n_sg_start = n_wg_start + sg_col * ESIMD_EXEC_SIZE;  // 16 cols per sub-group

    // Check if this sub-group has work (boundary check)
    const bool sg_has_work = (m_sg_start < args.M && n_sg_start < args.N);

    // Number of Q4_0 blocks per weight row
    const int k_blocks_per_row = static_cast<int>(args.K / UNIFIED_QK4_0);

    // Cast weight pointer
    const block_q4_0_unified * weights = static_cast<const block_q4_0_unified *>(args.weights);

    // Initialize SLM for this work-group
    esimd::slm_init<LARGE_SLM_TOTAL_BYTES>();

    // Initialize accumulators for this sub-group's output tiles
    // Each sub-group computes 2 stacked 8×16 tiles = 16×16 = 256 outputs
    // Use 2 separate accumulators for the 2 dpas tiles
    esimd::simd<float, ESIMD_ACC_SIZE> acc_lo = 0.0f;  // Rows [0..7]
    esimd::simd<float, ESIMD_ACC_SIZE> acc_hi = 0.0f;  // Rows [8..15]

    // Number of K tiles (LARGE_TILE_K = 32, process 2 dpas K-tiles of 16 each)
    const int64_t k_tiles = (args.K + LARGE_TILE_K - 1) / LARGE_TILE_K;

    // K-loop: iterate over K dimension in tiles of 32
    for (int64_t kt = 0; kt < k_tiles; kt++) {
        const int64_t k_start = kt * LARGE_TILE_K;
        const int64_t k_remaining = args.K - k_start;
        const int k_len = static_cast<int>(k_remaining < LARGE_TILE_K ? k_remaining : LARGE_TILE_K);

        // ================================================================
        // Phase 1: Cooperative Loading with Block Operations
        // ================================================================
        // SLM layout: [weights: 32 rows x 32 cols][activations: 32 rows x 32 cols]
        // 64 work-items load cooperatively:
        // - Work-items 0-31: Load weight rows 0-31 (one row each, 32 half values)
        // - Work-items 32-63: Load activation rows 0-31 (one row each, 32 half values)

        constexpr int ELEMS_PER_ROW = LARGE_TILE_K;  // 32 half values per row

        if (local_id < LARGE_TILE_N) {
            // Work-items 0-31: Load weight row local_id
            const int n_off = local_id;
            const int64_t n_global = n_wg_start + n_off;

            // Build row vector with dequantized weights
            esimd::simd<sycl::half, ELEMS_PER_ROW> w_row = sycl::half(0.0f);

            if (n_global < args.N) {
                // Dequantize 32 elements (may span 1-2 Q4_0 blocks since Q4_0 has 32 elements)
                // K_start should be aligned to 32 for LARGE_TILE_K=32

                #pragma unroll
                for (int k = 0; k < ELEMS_PER_ROW; k++) {
                    if (k >= k_len) {
                        w_row[k] = sycl::half(0.0f);
                        continue;
                    }
                    const int64_t k_global = k_start + k;
                    const int64_t block_idx = n_global * k_blocks_per_row + k_global / UNIFIED_QK4_0;
                    const int idx_in_block = static_cast<int>(k_global % UNIFIED_QK4_0);

                    const block_q4_0_unified * blk = &weights[block_idx];
                    const sycl::half d = blk->d;
                    const int byte_idx = idx_in_block / 2;
                    const int nibble_sel = idx_in_block % 2;
                    const uint8_t packed = blk->qs[byte_idx];
                    const int q_val = nibble_sel ? ((packed >> 4) & 0xF) : (packed & 0xF);
                    w_row[k] = static_cast<sycl::half>(static_cast<float>(d) * (q_val - 8));
                }
            }

            // Block store entire row to SLM
            const uint32_t slm_off = LARGE_SLM_WEIGHTS_OFFSET + n_off * ELEMS_PER_ROW * sizeof(sycl::half);
            esimd::slm_block_store<sycl::half, ELEMS_PER_ROW>(slm_off, w_row);

        } else if (local_id < LARGE_TILE_N + LARGE_TILE_M) {
            // Work-items 32-63: Load activation rows 0-31
            const int m_off = local_id - LARGE_TILE_N;  // 0-31
            const int64_t m_global = m_wg_start + m_off;

            // Build row vector in registers, then block store
            esimd::simd<sycl::half, ELEMS_PER_ROW> a_row = sycl::half(0.0f);

            if (m_global < args.M) {
                #pragma unroll
                for (int k = 0; k < ELEMS_PER_ROW; k++) {
                    if (k >= k_len) break;
                    const int64_t k_global = k_start + k;
                    a_row[k] = static_cast<sycl::half>(args.activations[m_global * args.K + k_global]);
                }
            }

            // Block store entire row to SLM
            const uint32_t slm_off = LARGE_SLM_ACTS_OFFSET + m_off * ELEMS_PER_ROW * sizeof(sycl::half);
            esimd::slm_block_store<sycl::half, ELEMS_PER_ROW>(slm_off, a_row);
        }
        // All 64 work-items participate in loading (32 weights + 32 activations)

        // Split barrier: signal that loading is complete, then wait for all loaders
        // Using split barriers (SPV_INTEL_split_barrier) for better performance on Arc
        esimd::fence<esimd::fence_mask::local_barrier>();
        split_barrier_arrive(ScopeWorkgroup, SemanticsWGMem);
        split_barrier_wait(ScopeWorkgroup, SemanticsWGMem);

        // ================================================================
        // Phase 2: Compute with Block Loads
        // ================================================================
        // Each sub-group computes its 16×16 output tile using 2 dpas operations
        // (2 stacked 8×16 tiles, one for rows 0-7 and one for rows 8-15)
        //
        // We process K=32 in two K-tile iterations of 16 each

        if (sg_has_work) {
            // Process two K-subtiles of 16 each within the K=32 tile
            #pragma unroll
            for (int k_sub = 0; k_sub < 2; k_sub++) {
                const int k_sub_start = k_sub * ESIMD_K_PER_DPAS;  // 0 or 16
                const int k_sub_len = (k_sub_start + ESIMD_K_PER_DPAS <= k_len) ?
                                      ESIMD_K_PER_DPAS :
                                      (k_len > k_sub_start ? k_len - k_sub_start : 0);

                if (k_sub_len == 0) continue;

                // ---- Load weights for this sub-group's N-column ----
                // SLM weight offset for this sub-group's 16 columns
                const int n_local_base = sg_col * ESIMD_EXEC_SIZE;  // 0, 16, 32, or 48

                // Load 16 weight rows × 16 K elements = 256 half
                // Row-major in SLM: row n at offset [n * 32 + k_sub_start]
                esimd::simd<sycl::half, ESIMD_B_SIZE> w_raw;  // 256 elements
                #pragma unroll
                for (int n = 0; n < ESIMD_EXEC_SIZE; n++) {
                    const uint32_t row_offset = LARGE_SLM_WEIGHTS_OFFSET +
                        (n_local_base + n) * LARGE_TILE_K * sizeof(sycl::half) +
                        k_sub_start * sizeof(sycl::half);

                    // Load 16 half values (one row slice)
                    esimd::simd<sycl::half, ESIMD_K_PER_DPAS> row_slice =
                        esimd::slm_block_load<sycl::half, ESIMD_K_PER_DPAS>(row_offset);

                    #pragma unroll
                    for (int k = 0; k < ESIMD_K_PER_DPAS; k++) {
                        w_raw[n * ESIMD_K_PER_DPAS + k] = row_slice[k];
                    }
                }

                // Repack weights from row-major to VNNI format for dpas
                esimd::simd<sycl::half, ESIMD_B_SIZE> b_vec;
                #pragma unroll
                for (int n = 0; n < ESIMD_EXEC_SIZE; n++) {
                    #pragma unroll
                    for (int k = 0; k < ESIMD_K_PER_DPAS; k++) {
                        const int vnni_idx = (k / 2) * (ESIMD_EXEC_SIZE * 2) + n * 2 + (k % 2);
                        b_vec[vnni_idx] = w_raw[n * ESIMD_K_PER_DPAS + k];
                    }
                }

                // Handle boundary: zero out weights beyond N
                #pragma unroll
                for (int n = 0; n < ESIMD_EXEC_SIZE; n++) {
                    const int64_t n_global = n_sg_start + n;
                    if (n_global >= args.N) {
                        #pragma unroll
                        for (int k = 0; k < ESIMD_K_PER_DPAS; k++) {
                            const int vnni_idx = (k / 2) * (ESIMD_EXEC_SIZE * 2) + n * 2 + (k % 2);
                            b_vec[vnni_idx] = sycl::half(0.0f);
                        }
                    }
                }

                // ---- Process lower 8 rows (acc_lo) ----
                {
                    const int m_local_base = sg_row * LARGE_SG_M;  // 0 or 16

                    // Load 8 activation rows × 16 K elements = 128 half
                    esimd::simd<sycl::half, ESIMD_A_SIZE> a_vec;
                    #pragma unroll
                    for (int m = 0; m < ESIMD_REPEAT_COUNT; m++) {
                        const uint32_t row_offset = LARGE_SLM_ACTS_OFFSET +
                            (m_local_base + m) * LARGE_TILE_K * sizeof(sycl::half) +
                            k_sub_start * sizeof(sycl::half);

                        esimd::simd<sycl::half, ESIMD_K_PER_DPAS> row_slice =
                            esimd::slm_block_load<sycl::half, ESIMD_K_PER_DPAS>(row_offset);

                        #pragma unroll
                        for (int k = 0; k < ESIMD_K_PER_DPAS; k++) {
                            a_vec[m * ESIMD_K_PER_DPAS + k] = row_slice[k];
                        }
                    }

                    // Handle boundary: zero out rows beyond M
                    #pragma unroll
                    for (int m = 0; m < ESIMD_REPEAT_COUNT; m++) {
                        const int64_t m_global = m_sg_start + m;
                        if (m_global >= args.M) {
                            #pragma unroll
                            for (int k = 0; k < ESIMD_K_PER_DPAS; k++) {
                                a_vec[m * ESIMD_K_PER_DPAS + k] = sycl::half(0.0f);
                            }
                        }
                    }

                    // Execute dpas: acc_lo += A @ B
                    acc_lo = xmx::dpas<ESIMD_SYSTOLIC_DEPTH, ESIMD_REPEAT_COUNT,
                                       float, float, sycl::half, sycl::half>(acc_lo, b_vec, a_vec);
                }

                // ---- Process upper 8 rows (acc_hi) ----
                {
                    const int m_local_base = sg_row * LARGE_SG_M + ESIMD_REPEAT_COUNT;  // 8 or 24

                    // Load 8 activation rows × 16 K elements = 128 half
                    esimd::simd<sycl::half, ESIMD_A_SIZE> a_vec;
                    #pragma unroll
                    for (int m = 0; m < ESIMD_REPEAT_COUNT; m++) {
                        const uint32_t row_offset = LARGE_SLM_ACTS_OFFSET +
                            (m_local_base + m) * LARGE_TILE_K * sizeof(sycl::half) +
                            k_sub_start * sizeof(sycl::half);

                        esimd::simd<sycl::half, ESIMD_K_PER_DPAS> row_slice =
                            esimd::slm_block_load<sycl::half, ESIMD_K_PER_DPAS>(row_offset);

                        #pragma unroll
                        for (int k = 0; k < ESIMD_K_PER_DPAS; k++) {
                            a_vec[m * ESIMD_K_PER_DPAS + k] = row_slice[k];
                        }
                    }

                    // Handle boundary: zero out rows beyond M
                    #pragma unroll
                    for (int m = 0; m < ESIMD_REPEAT_COUNT; m++) {
                        const int64_t m_global = m_sg_start + ESIMD_REPEAT_COUNT + m;
                        if (m_global >= args.M) {
                            #pragma unroll
                            for (int k = 0; k < ESIMD_K_PER_DPAS; k++) {
                                a_vec[m * ESIMD_K_PER_DPAS + k] = sycl::half(0.0f);
                            }
                        }
                    }

                    // Execute dpas: acc_hi += A @ B
                    acc_hi = xmx::dpas<ESIMD_SYSTOLIC_DEPTH, ESIMD_REPEAT_COUNT,
                                       float, float, sycl::half, sycl::half>(acc_hi, b_vec, a_vec);
                }
            }
        }

        // Split barrier before next K-tile (ensure all sub-groups done before overwriting SLM)
        if (kt + 1 < k_tiles) {
            esimd::fence<esimd::fence_mask::local_barrier>();
            split_barrier_arrive(ScopeWorkgroup, SemanticsWGMem);
            split_barrier_wait(ScopeWorkgroup, SemanticsWGMem);
        }
    }

    // ================================================================
    // Phase 3: Write output (each sub-group writes its 16×16 tile)
    // ================================================================
    if (sg_has_work) {
        // Write lower 8 rows
        #pragma unroll
        for (int m = 0; m < ESIMD_REPEAT_COUNT; m++) {
            const int64_t m_global = m_sg_start + m;
            if (m_global >= args.M) continue;

            #pragma unroll
            for (int n = 0; n < ESIMD_EXEC_SIZE; n++) {
                const int64_t n_global = n_sg_start + n;
                if (n_global >= args.N) continue;

                args.output[m_global * args.N + n_global] = acc_lo[m * ESIMD_EXEC_SIZE + n];
            }
        }

        // Write upper 8 rows
        #pragma unroll
        for (int m = 0; m < ESIMD_REPEAT_COUNT; m++) {
            const int64_t m_global = m_sg_start + ESIMD_REPEAT_COUNT + m;
            if (m_global >= args.M) continue;

            #pragma unroll
            for (int n = 0; n < ESIMD_EXEC_SIZE; n++) {
                const int64_t n_global = n_sg_start + n;
                if (n_global >= args.N) continue;

                args.output[m_global * args.N + n_global] = acc_hi[m * ESIMD_EXEC_SIZE + n];
            }
        }
    }
}
#endif  // GGML_SYCL_LARGE_TILE_KERNEL_ENABLED

/**
 * Check if INT8 ESIMD dpas path can be used for given dimensions.
 *
 * INT8 ESIMD dpas requires:
 * - ESIMD enabled via GGML_SYCL_XMX_ESIMD=1
 * - INT8 enabled via GGML_SYCL_XMX_INT8=1
 * - K aligned to Q4_0 block size (32)
 *
 * @param M  Output rows
 * @param N  Output columns
 * @param K  Reduction dimension
 * @return true if INT8 ESIMD dpas can be used
 */
inline bool can_use_esimd_int8_dpas(int64_t M, int64_t N, int64_t K) {
    // Both ESIMD and INT8 must be enabled
    if (!use_esimd_dpas() || !use_int8_dpas()) {
        return false;
    }
    // K must be multiple of Q4_0 block size for proper dequantization
    if (K % UNIFIED_QK4_0 != 0) {
        return false;
    }
    // Must have at least some work to do
    return M >= 1 && N >= 1 && K >= 1;
}

/**
 * Check if ESIMD dpas path can be used for given dimensions.
 *
 * ESIMD dpas requires:
 * - ESIMD enabled (default on; disable with GGML_SYCL_XMX_ESIMD=0)
 * - M >= 1 (we handle partial tiles)
 * - N >= 1 (we handle partial tiles)
 * - K aligned to Q4_0 block size (32)
 *
 * @param M  Output rows
 * @param N  Output columns
 * @param K  Reduction dimension
 * @return true if ESIMD dpas can be used
 */
inline bool can_use_esimd_dpas(int64_t M, int64_t N, int64_t K) {
    // ESIMD path enabled by default; disable with GGML_SYCL_XMX_ESIMD=0
    if (!use_esimd_dpas()) {
        return false;
    }
    // K must be multiple of Q4_0 block size for proper dequantization
    if (K % UNIFIED_QK4_0 != 0) {
        return false;
    }
    // Must have at least some work to do
    return M >= 1 && N >= 1 && K >= 1;
}

#endif  // GGML_SYCL_ESIMD_AVAILABLE

// =============================================================================
// Unified Matmul Kernel - Scalar Path
// =============================================================================

/**
 * Unified matmul kernel template.
 *
 * GGML Convention: dst[m,n] = sum_k(src0[n,k] * src1[m,k])
 * Computes: output[M,N] = activations[M,K] @ weights[N,K]^T
 *
 * Where:
 * - weights (src0) has shape [N, K] - indexed by output column n
 * - activations (src1) has shape [M, K] - indexed by output row m
 * - output (dst) has shape [M, N]
 *
 * Template parameters control tile sizes and compute path.
 * Implements scalar path; XMX path is in unified_matmul_xmx_kernel_impl.
 *
 * Work distribution:
 * - 2D grid: [ceil(M/TILE_M), ceil(N/TILE_N)]
 * - Each work-group computes one output tile of size TILE_M x TILE_N
 * - K dimension is iterated within each work-group
 *
 * Memory access pattern:
 * - Weights: load tile_n rows of K weights (one per output column), dequantize on-the-fly
 * - Activations: load tile_m rows of K activations (one per output row)
 * - Output: write tile_m x tile_n results
 */
template <int TILE_M, int TILE_N, int TILE_K, bool USE_XMX>
void unified_matmul_kernel_impl(sycl::nd_item<2>                   item,
                                const UnifiedKernelArgs            args,
                                sycl::local_accessor<float, 1>     slm_weights,
                                sycl::local_accessor<float, 1>     slm_activations) {
    // Tile coordinates
    const int tile_row = item.get_group(0);  // M dimension (output rows)
    const int tile_col = item.get_group(1);  // N dimension (output columns)

    // Thread coordinates within work-group
    const int local_row = item.get_local_id(0);
    const int local_col = item.get_local_id(1);
    const int local_size_row = item.get_local_range(0);
    const int local_size_col = item.get_local_range(1);

    // Global output coordinates for this work-group
    const int64_t m_start = tile_row * TILE_M;  // Starting output row
    const int64_t n_start = tile_col * TILE_N;  // Starting output column

    // Number of K tiles
    const int k_tiles = (args.K + TILE_K - 1) / TILE_K;
    const int k_blocks_per_row = args.K / UNIFIED_QK4_0;

    // Cast weight pointer
    const block_q4_0_unified * weights = static_cast<const block_q4_0_unified *>(args.weights);

    // Initialize accumulator for each thread's output elements
    // Each thread computes multiple output elements
    float acc[4][4] = {{0.0f}};  // Thread computes up to 4x4 outputs

    // Determine how many outputs each thread computes
    const int outputs_per_thread_m = (TILE_M + local_size_row - 1) / local_size_row;
    const int outputs_per_thread_n = (TILE_N + local_size_col - 1) / local_size_col;

    // K-loop: iterate over K dimension in tiles
    for (int kt = 0; kt < k_tiles; kt++) {
        const int64_t k_start = kt * TILE_K;
        const int64_t k_end = sycl::min(k_start + TILE_K, args.K);
        const int     k_len = static_cast<int>(k_end - k_start);

        // Barrier before loading new tile data
        item.barrier(sycl::access::fence_space::local_space);

        // ==== Load weights to SLM ====
        // GGML: weights[N, K] - indexed by output column n
        // Each thread loads multiple weight elements
        // SLM layout: [TILE_N x TILE_K] indexed as [n_off * TILE_K + k_off]
        for (int n_off = local_row; n_off < TILE_N; n_off += local_size_row) {
            const int64_t n_global = n_start + n_off;  // Output column = weight row
            if (n_global >= args.N) continue;

            for (int k_off = local_col; k_off < k_len; k_off += local_size_col) {
                const int64_t k_global = k_start + k_off;

                // GGML: weights[n_global, k_global]
                const int block_idx = static_cast<int>(n_global * k_blocks_per_row + k_global / UNIFIED_QK4_0);
                const int idx_in_block = static_cast<int>(k_global % UNIFIED_QK4_0);

                // Dequantize and store to SLM
                float w = dequant_q4_0(&weights[block_idx], idx_in_block);
                slm_weights[n_off * TILE_K + k_off] = w;
            }
        }

        // ==== Load activations to SLM ====
        // GGML: activations[M, K] - indexed by output row m
        // SLM layout: [TILE_M x TILE_K] indexed as [m_off * TILE_K + k_off]
        for (int m_off = local_row; m_off < TILE_M; m_off += local_size_row) {
            const int64_t m_global = m_start + m_off;  // Output row
            if (m_global >= args.M) continue;

            for (int k_off = local_col; k_off < k_len; k_off += local_size_col) {
                const int64_t k_global = k_start + k_off;

                // GGML: activations[m_global, k_global]
                float a = args.activations[m_global * args.K + k_global];
                slm_activations[m_off * TILE_K + k_off] = a;
            }
        }

        // Barrier after loading
        item.barrier(sycl::access::fence_space::local_space);

        // ==== Compute: accumulate partial products ====
        // GGML: dst[m,n] = sum_k(weights[n,k] * activations[m,k])
        // Each thread computes its assigned outputs
        for (int mo = 0; mo < outputs_per_thread_m; mo++) {
            const int m_off = local_row + mo * local_size_row;
            if (m_off >= TILE_M) continue;
            if (m_start + m_off >= args.M) continue;

            for (int no = 0; no < outputs_per_thread_n; no++) {
                const int n_off = local_col + no * local_size_col;
                if (n_off >= TILE_N) continue;
                if (n_start + n_off >= args.N) continue;

                // Dot product over K tile
                // dst[m,n] = sum_k(weights[n,k] * activations[m,k])
                float sum = 0.0f;
                for (int k = 0; k < k_len; k++) {
                    float w = slm_weights[n_off * TILE_K + k];     // weights[n, k]
                    float a = slm_activations[m_off * TILE_K + k]; // activations[m, k]
                    sum += w * a;
                }

                // Accumulate to thread-local storage
                if (mo < 4 && no < 4) {
                    acc[mo][no] += sum;
                }
            }
        }
    }

    // ==== Write output ====
    // Barrier not strictly needed here since we're writing to global memory
    for (int mo = 0; mo < outputs_per_thread_m && mo < 4; mo++) {
        const int m_off = local_row + mo * local_size_row;
        if (m_off >= TILE_M) continue;
        const int64_t m_global = m_start + m_off;
        if (m_global >= args.M) continue;

        for (int no = 0; no < outputs_per_thread_n && no < 4; no++) {
            const int n_off = local_col + no * local_size_col;
            if (n_off >= TILE_N) continue;
            const int64_t n_global = n_start + n_off;
            if (n_global >= args.N) continue;

            args.output[m_global * args.N + n_global] = acc[mo][no];
        }
    }
}

// =============================================================================
// Kernel Launcher
// =============================================================================

void launch_unified_matmul(sycl::queue & q, const UnifiedKernelArgs & args_in) {
    // Make a mutable copy of args to set prefetch_depth from host configuration
    UnifiedKernelArgs args = args_in;

    // Set prefetch distance from host-side configuration if not already set
    // This ensures the kernel gets the correct value without calling getenv from device
    if (args.prefetch_depth <= 0) {
        args.prefetch_depth = get_prefetch_distance();
    }

    // Validate arguments
    if (!validate_args(args)) {
        fprintf(stderr, "[unified-kernel] Invalid arguments\n");
        return;
    }

    // Check for supported quantization types
    if (args.quant_type != QUANT_TYPE_Q4_0 && args.quant_type != QUANT_TYPE_MXFP4) {
        fprintf(stderr, "[unified-kernel] Unsupported quantization type=%d (supported: Q4_0=%d, MXFP4=%d)\n",
                args.quant_type, QUANT_TYPE_Q4_0, QUANT_TYPE_MXFP4);
        return;
    }

    // ==========================================================================
    // Batch-Size Gating: Select optimal kernel path
    // ==========================================================================
    // Query XMX config for dispatch decision (cached in XMXConfig::from_device)
    // Use device 0 by default (single-GPU typical case)
    XMXConfig cfg = XMXConfig::from_device(0);

    // Batch size is the M dimension (number of tokens being processed)
    const int batch_size = static_cast<int>(args.M);

    // Select kernel path based on batch size and hardware capabilities
    KernelPath selected_path = select_kernel_path(
        batch_size, args.M, args.N, args.K, args.quant_type, cfg);

    // Debug: Print dispatch decision (controlled by GGML_SYCL_DEBUG=1)
    if (ggml_sycl_unified_debug_enabled()) {
        fprintf(stderr, "[unified-kernel] DISPATCH: batch=%d path=%s M=%lld N=%lld K=%lld "
                "min_batch=%d xmx_supported=%d force_mmvq=%d force_esimd=%d\n",
                batch_size, kernel_path_name(selected_path),
                static_cast<long long>(args.M), static_cast<long long>(args.N),
                static_cast<long long>(args.K),
                get_esimd_min_batch(), cfg.supported ? 1 : 0,
                env_force_mmvq() ? 1 : 0, env_force_esimd() ? 1 : 0);
        fflush(stderr);
    }

    // Early return for DMMV and MMVQ paths - these are handled by existing kernels
    // in ggml-sycl.cpp. The unified kernel only handles the ESIMD_DPAS path here.
    // The caller (ggml_sycl_mul_mat) should check the path and dispatch accordingly.
    //
    // NOTE: This function is the unified kernel launcher. For now, we continue
    // with ESIMD path selection below if ESIMD is available. The DMMV/MMVQ
    // fallback is implicit: if ESIMD paths don't match, scalar path is used.

    // Debug: Print launch parameters (opt-in to avoid log spam in production)
    if (ggml_sycl_unified_debug_enabled()) {
        fprintf(stderr, "[unified-kernel] LAUNCH: M=%lld N=%lld K=%lld type=%d tile=(%d,%d,%d) xmx=%d\n",
                static_cast<long long>(args.M), static_cast<long long>(args.N),
                static_cast<long long>(args.K), args.quant_type,
                args.tile_m, args.tile_n, args.tile_k, args.use_xmx ? 1 : 0);
        fflush(stderr);
    }

    // Calculate grid dimensions
    const int tile_m = args.tile_m;
    const int tile_n = args.tile_n;
    const int tile_k = args.tile_k;

    const int grid_m = (static_cast<int>(args.M) + tile_m - 1) / tile_m;
    const int grid_n = (static_cast<int>(args.N) + tile_n - 1) / tile_n;

    // Work-group size: use square-ish shape
    // Limit to reasonable size that divides well
    const int wg_size_m = std::min(tile_m, 8);
    const int wg_size_n = std::min(tile_n, 16);

    // Determine if XMX path should be used
    const bool prefer_esimd_small = ggml_sycl_unified::prefer_esimd_small();
    const int  esimd_min_batch    = ggml_sycl_unified::get_esimd_min_batch();
    const int  prefer_esimd_max_m = ggml_sycl_unified::prefer_esimd_max_m();
    const bool allow_joint_matrix = !(prefer_esimd_small &&
                                     (batch_size < esimd_min_batch || args.M <= prefer_esimd_max_m));
    bool use_xmx_path = false;
#if GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE
    if (allow_joint_matrix && args.use_xmx && can_use_xmx(args.M, args.N, args.K)) {
        // Check if device supports XMX
        sycl::device dev = q.get_device();
        use_xmx_path = dev.has(sycl::aspect::ext_intel_matrix);
    }
#endif

    // ==========================================================================
    // XMX Path: Use joint_matrix for dpas acceleration (optimized)
    // ==========================================================================
#if GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE
    // XMX joint_matrix path should only run if ESIMD_LARGE_TILE was not selected
    // (ESIMD_LARGE_TILE uses its own ESIMD-based large-tile implementation)
    // Set GGML_SYCL_SKIP_JM=1 to skip joint_matrix and use cooperative ESIMD instead
    static const bool skip_joint_matrix = (std::getenv("GGML_SYCL_SKIP_JM") != nullptr);
    if (use_xmx_path && selected_path != KernelPath::ESIMD_LARGE_TILE && !skip_joint_matrix) {
        // XMX path uses fixed tile sizes aligned to XMX dimensions
        constexpr int XMX_TM = 8;   // Must be multiple of XMX_TILE_M=8
        constexpr int XMX_TN = 16;  // Must be multiple of XMX_TILE_N=16
        constexpr int XMX_TK = 32;  // Must be multiple of XMX_TILE_K=16

        const int xmx_grid_m = (static_cast<int>(args.M) + XMX_TM - 1) / XMX_TM;
        const int xmx_grid_n = (static_cast<int>(args.N) + XMX_TN - 1) / XMX_TN;
        if (ggml_sycl_unified_debug_enabled()) {
            fprintf(stderr, "[unified-kernel] XMX path: M=%lld N=%lld K=%lld grid=(%d,%d)\n",
                    static_cast<long long>(args.M), static_cast<long long>(args.N),
                    static_cast<long long>(args.K), xmx_grid_m, xmx_grid_n);
            fflush(stderr);
        }

        // Work-group size: 1 sub-group (16 threads)
        // Each work-group handles one XMX tile (8x16 output)
        constexpr int xmx_wg_m = 1;
        constexpr int xmx_wg_n = XMX_SUBGROUP_SIZE;

        q.submit([&](sycl::handler & cgh) {
            // Allocate SLM for half-precision data
            sycl::local_accessor<sycl::half, 1> slm_w(XMX_TN * XMX_TK, cgh);  // Weights
            sycl::local_accessor<sycl::half, 1> slm_a(XMX_TM * XMX_TK, cgh);  // Activations
            // Float SLM for boundary case accumulator output
            sycl::local_accessor<float, 1> slm_acc_out(XMX_TILE_M * XMX_TILE_N, cgh);

            sycl::nd_range<2> range(
                sycl::range<2>(xmx_grid_m * xmx_wg_m, xmx_grid_n * xmx_wg_n),
                sycl::range<2>(xmx_wg_m, xmx_wg_n)
            );

            cgh.parallel_for<unified_matmul_xmx_kernel_name<XMX_TM, XMX_TN, XMX_TK>>(
                range,
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SUBGROUP_SIZE)]] {
                    unified_matmul_xmx_kernel_impl<XMX_TM, XMX_TN, XMX_TK>(
                        item, args, slm_w, slm_a, slm_acc_out);
                }
            );
        });
        return;
    }
#endif  // GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE

    // ==========================================================================
    // ESIMD dpas Path: Use ESIMD xmx::dpas for explicit SIMD control
    // ==========================================================================
    // Enabled by default while optimizing; set GGML_SYCL_XMX_ESIMD=0 to disable.
    // Each work-item processes one 8x16 output tile using ESIMD dpas instruction.
    //
    // Two variants:
    // 1. INT8 (GGML_SYCL_XMX_INT8=1): Dynamic activation quantization, K=32 per dpas
    // 2. FP16 (default when ESIMD enabled): K=16 per dpas
    //
    // NOTE: INT8 is LOSSY - not bit-exact with FP16 path!
    //
    // Batch-size gating: Only use ESIMD path if selected_path == ESIMD_DPAS
    // Small batches should use scalar DMMV/MMVQ kernels instead.

#if GGML_SYCL_ESIMD_AVAILABLE
    // ==========================================================================
    // ESIMD Path: Hardware Capability Check
    // ==========================================================================
    // Our ESIMD kernels use ExecutionSize=16 for xmx::dpas, which requires:
    //   - PVC (Xe-HPC/Data Center Max) - ExecutionSize=16 supported
    //   - Xe2 (Arc Battlemage B580/B570) - ExecutionSize=16 supported
    //
    // Arc Alchemist (XeHPG) only supports ExecutionSize=8, so our kernels won't work.
    // Use joint_matrix API instead for XeHPG - it handles the difference automatically.
    //
    // NOTE: We use a runtime check here. The ESIMD kernels are still compiled,
    // but they won't be submitted to the queue on unsupported hardware.
    const bool esimd_hw_supported = cfg.supports_esimd_dpas;
    if (!esimd_hw_supported) {
        if (ggml_sycl_unified_debug_enabled()) {
            fprintf(stderr, "[unified-kernel] ESIMD dpas not supported on this GPU, using joint_matrix/scalar path\n");
            fflush(stderr);
        }
        // Fall through to DMMV/scalar paths below
    }

    // Only proceed with ESIMD paths if:
    // 1. Hardware supports ESIMD dpas (PVC only)
    // 2. Batch-size gating selected ESIMD_DPAS or ESIMD_LARGE_TILE
    const bool esimd_enabled_by_gating = esimd_hw_supported && (selected_path == KernelPath::ESIMD_DPAS);
    const bool large_tile_selected     = esimd_hw_supported && (selected_path == KernelPath::ESIMD_LARGE_TILE);

#if GGML_SYCL_LARGE_TILE_KERNEL_ENABLED
    // Large-tile ESIMD path - adaptive based on hardware capabilities
    // Uses cooperative loading with multiple sub-groups for better memory bandwidth
    // Tile configuration selected based on max ESIMD work-group size:
    // - Arc/DG2 (max 64):  32×32 tiles, 64 work-items, 4 sub-groups
    // - PVC (max 256+):    64×64 tiles, 256 work-items, 16 sub-groups
    if (large_tile_selected) {
        // Get hardware-optimal tile configuration
        LargeTileConfig tile_cfg = LargeTileConfig::for_hardware(cfg.max_esimd_workgroup);

        // Check if dimensions are sufficient for this tile size
        if (!tile_cfg.can_use(args.M, args.N, args.K)) {
            // Fall through to cooperative ESIMD path for smaller dimensions
            goto cooperative_path;
        }

        const int large_grid_m = (static_cast<int>(args.M) + tile_cfg.tile_m - 1) / tile_cfg.tile_m;
        const int large_grid_n = (static_cast<int>(args.N) + tile_cfg.tile_n - 1) / tile_cfg.tile_n;

        if (ggml_sycl_unified_debug_enabled()) {
            fprintf(stderr, "[unified-kernel] Large-tile ESIMD path: M=%lld N=%lld K=%lld "
                    "grid=(%d,%d) tile=(%d,%d) wg=%d\n",
                    static_cast<long long>(args.M), static_cast<long long>(args.N),
                    static_cast<long long>(args.K), large_grid_m, large_grid_n,
                    tile_cfg.tile_m, tile_cfg.tile_n, tile_cfg.wg_size);
            fflush(stderr);
        }

        // Dispatch to appropriate kernel instantiation based on work-group size
        // Each WG_SIZE requires a separate template instantiation
        if (tile_cfg.wg_size == 64) {
            q.submit([&](sycl::handler & cgh) {
                constexpr int WG_SIZE = 64;
                sycl::range<2> global(large_grid_m * WG_SIZE, large_grid_n);
                sycl::range<2> local(WG_SIZE, 1);

                cgh.parallel_for<esimd_fp16_large_tile_kernel<WG_SIZE>>(
                    sycl::nd_range<2>(global, local),
                    [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                        const int local_id = item.get_local_id(0);
                        const int sg_id    = local_id / 16;
                        const int lane     = local_id % 16;
                        large_tile_esimd_kernel_impl<WG_SIZE>(item, args, local_id, sg_id, lane);
                    });
            });
            return;
        }
        // Note: WG_SIZE=128 and WG_SIZE=256 instantiations can be added here
        // when validated on hardware that supports larger work-groups
    }
cooperative_path:;  // Fallthrough label for large-tile to cooperative path
#else
    (void)large_tile_selected;  // Suppress unused variable warning when kernels disabled
#endif  // GGML_SYCL_LARGE_TILE_KERNEL_ENABLED

    // Try INT8 path first (requires both ESIMD and INT8 flags AND batch-size gating)
    if (esimd_enabled_by_gating && can_use_esimd_int8_dpas(args.M, args.N, args.K)) {
        // ESIMD INT8 dpas tile sizes (fixed by hardware)
        constexpr int ESIMD_TM = 8;   // RepeatCount = 8
        constexpr int ESIMD_TN = 16;  // ExecutionSize = 16

        const int esimd_grid_m = (static_cast<int>(args.M) + ESIMD_TM - 1) / ESIMD_TM;
        const int esimd_grid_n = (static_cast<int>(args.N) + ESIMD_TN - 1) / ESIMD_TN;

        if (ggml_sycl_unified_debug_enabled()) {
            fprintf(stderr, "[unified-kernel] ESIMD INT8 path: M=%lld N=%lld K=%lld grid=(%d,%d)\n",
                    static_cast<long long>(args.M), static_cast<long long>(args.N),
                    static_cast<long long>(args.K), esimd_grid_m, esimd_grid_n);
            fflush(stderr);
        }

        q.submit([&](sycl::handler & cgh) {
            // ESIMD kernel: one work-item per output tile (no work-group cooperation)
            // Total work items = grid_m * grid_n
            sycl::range<2> global(esimd_grid_m, esimd_grid_n);
            sycl::range<2> local(1, 1);  // Single work-item per work-group for ESIMD

            cgh.parallel_for<esimd_int8_kernel<ESIMD_TM, ESIMD_TN>>(
                sycl::nd_range<2>(global, local),
                [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                    // Calculate tile coordinates
                    const int tile_row = item.get_global_id(0);  // M tile index
                    const int tile_col = item.get_global_id(1);  // N tile index

                    const int64_t m_start = tile_row * ESIMD_TM;
                    const int64_t n_start = tile_col * ESIMD_TN;

                    // Call ESIMD INT8 kernel implementation
                    esimd_matmul_int8_kernel_impl<ESIMD_TM, ESIMD_TN>(args, m_start, n_start);
                }
            );
        });
        return;
    }

#if GGML_SYCL_COOPERATIVE_KERNEL_ENABLED
    // Cooperative ESIMD FP16 path (multi-work-item with work-group barrier)
    // Uses SPIR-V split barriers for synchronization (Arc-compatible).
    // Enabled by default; set GGML_SYCL_XMX_COOPERATIVE=0 to disable
    // Work-group size configurable via GGML_SYCL_ESIMD_WG_SIZE (valid: 32, 64)
    if (esimd_enabled_by_gating && can_use_cooperative_esimd(args.M, args.N, args.K)) {
        // Get configured work-group size (default: 32)
        const int wg_size = get_cooperative_wg_size();

        // Grid dimensions: each work-group handles COOP_WG_M x COOP_WG_N output
        const int coop_grid_m = (static_cast<int>(args.M) + COOP_WG_M - 1) / COOP_WG_M;
        const int coop_grid_n = (static_cast<int>(args.N) + COOP_WG_N - 1) / COOP_WG_N;

        if (ggml_sycl_unified_debug_enabled()) {
            fprintf(stderr, "[unified-kernel] Cooperative ESIMD FP16 path: M=%lld N=%lld K=%lld "
                    "grid=(%d,%d) wg_size=%d\n",
                    static_cast<long long>(args.M), static_cast<long long>(args.N),
                    static_cast<long long>(args.K), coop_grid_m, coop_grid_n, wg_size);
            fflush(stderr);
        }

        // Currently only WG_SIZE=32 is fully implemented
        // WG_SIZE=64 requires larger SLM tiles (TODO: implement in future)
        // The wg_size variable is checked but always returns 32 for now
        (void)wg_size;  // Suppress unused variable warning

        q.submit([&](sycl::handler & cgh) {
            constexpr int WG_SIZE = 32;
            sycl::range<2> global(coop_grid_m * WG_SIZE, coop_grid_n);
            sycl::range<2> local(WG_SIZE, 1);

            cgh.parallel_for<esimd_fp16_cooperative_kernel<WG_SIZE>>(
                sycl::nd_range<2>(global, local),
                [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                    const int local_id = item.get_local_id(0);
                    const int sg_id = local_id / COOP_SUBGROUP_SIZE;
                    const int lane = local_id % COOP_SUBGROUP_SIZE;
                    esimd_matmul_fp16_cooperative_impl<WG_SIZE>(
                        item, args, local_id, sg_id, lane);
                }
            );
        });
        return;
    }
#endif  // GGML_SYCL_COOPERATIVE_KERNEL_ENABLED

    // FP16 path (ESIMD enabled but INT8 not enabled)
    if (esimd_enabled_by_gating && can_use_esimd_dpas(args.M, args.N, args.K)) {
        // ESIMD FP16 dpas tile sizes (fixed by hardware)
        constexpr int ESIMD_TM = 8;   // RepeatCount = 8
        constexpr int ESIMD_TN = 16;  // ExecutionSize = 16

        const int esimd_grid_m = (static_cast<int>(args.M) + ESIMD_TM - 1) / ESIMD_TM;
        const int esimd_grid_n = (static_cast<int>(args.N) + ESIMD_TN - 1) / ESIMD_TN;

        // Query XMX config for double-buffer capability
        // Use device 0 by default (single-GPU typical case)
        XMXConfig cfg = XMXConfig::from_device(0);

        if (ggml_sycl_unified_debug_enabled()) {
            fprintf(stderr, "[unified-kernel] ESIMD FP16 path: M=%lld N=%lld K=%lld grid=(%d,%d) double_buf=%d\n",
                    static_cast<long long>(args.M), static_cast<long long>(args.N),
                    static_cast<long long>(args.K), esimd_grid_m, esimd_grid_n,
                    cfg.use_double_buffer ? 1 : 0);
            fflush(stderr);
        }

        // Use double-buffered kernel if enabled and SLM permits
        if (cfg.use_double_buffer) {
            q.submit([&](sycl::handler & cgh) {
                // ESIMD kernel: one work-item per output tile (no work-group cooperation)
                sycl::range<2> global(esimd_grid_m, esimd_grid_n);
                sycl::range<2> local(1, 1);  // Single work-item per work-group for ESIMD

                // Capture cfg by value for device execution
                XMXConfig cfg_copy = cfg;

                cgh.parallel_for<esimd_fp16_double_buffered_kernel<ESIMD_TM, ESIMD_TN>>(
                    sycl::nd_range<2>(global, local),
                    [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                        // Calculate tile coordinates
                        const int tile_row = item.get_global_id(0);  // M tile index
                        const int tile_col = item.get_global_id(1);  // N tile index

                        const int64_t m_start = tile_row * ESIMD_TM;
                        const int64_t n_start = tile_col * ESIMD_TN;

                        // Call double-buffered ESIMD FP16 kernel implementation
                        esimd_matmul_fp16_double_buffered_impl<ESIMD_TM, ESIMD_TN>(
                            args, m_start, n_start, cfg_copy);
                    }
                );
            });
        } else {
            // Fall back to non-buffered path
            q.submit([&](sycl::handler & cgh) {
                // ESIMD kernel: one work-item per output tile (no work-group cooperation)
                sycl::range<2> global(esimd_grid_m, esimd_grid_n);
                sycl::range<2> local(1, 1);  // Single work-item per work-group for ESIMD

                cgh.parallel_for<esimd_fp16_kernel<ESIMD_TM, ESIMD_TN>>(
                    sycl::nd_range<2>(global, local),
                    [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                        // Calculate tile coordinates
                        const int tile_row = item.get_global_id(0);  // M tile index
                        const int tile_col = item.get_global_id(1);  // N tile index

                        const int64_t m_start = tile_row * ESIMD_TM;
                        const int64_t n_start = tile_col * ESIMD_TN;

                        // Call ESIMD FP16 kernel implementation
                        esimd_matmul_fp16_kernel_impl<ESIMD_TM, ESIMD_TN>(args, m_start, n_start);
                    }
                );
            });
        }
        return;
    }
#endif  // GGML_SYCL_ESIMD_AVAILABLE

    // ==========================================================================
    // DMMV-like Path: Warp-parallel reduction for batch=1
    // ==========================================================================
    // When batch_size==1 and ESIMD is not available/not selected, use a
    // DMMV-like kernel that's ~60x faster than the scalar tiled approach.
    //
    // Key optimizations:
    // - Each warp computes one output element (N column)
    // - Threads in the warp cooperatively process K blocks
    // - Warp-level shuffle reduction (no SLM overhead)
    // - Vectorized Q4_0 dequantization within each thread
    // - SoA layout support for better memory bandwidth on large K
    //
    // Work distribution:
    // - Grid: N work-groups (one per output column)
    // - Work-group: WARP_SIZE threads (32)
    // - Each thread: processes K/WARP_SIZE blocks, reduces partial sums

    constexpr int DMMV_WARP_SIZE = 32;  // Match GGML_SYCL_WARP_SIZE
    constexpr int DMMV_BLOCK_SIZE = 32; // Q4_0 block size (UNIFIED_QK4_0)

    // Use DMMV path for batch=1 when ESIMD path was not selected
    // (selected_path would be DMMV or we fell through ESIMD checks)
    if (batch_size == 1 && selected_path != KernelPath::ESIMD_DPAS) {
        // Grid: one work-group per output column (N dimension)
        // Each work-group computes the dot product for one output element
        const int grid_n = static_cast<int>(args.N);
        const bool use_soa = (args.layout == LayoutMode::SOA);

        if (ggml_sycl_unified_debug_enabled()) {
            fprintf(stderr, "[unified-kernel] DMMV path: M=%lld N=%lld K=%lld grid_n=%d warp_size=%d layout=%s\n",
                    static_cast<long long>(args.M), static_cast<long long>(args.N),
                    static_cast<long long>(args.K), grid_n, DMMV_WARP_SIZE,
                    use_soa ? "SOA" : "AOS");
            fflush(stderr);
        }

        // SoA layout calculations (precomputed on host)
        // SoA layout: [qs: N rows × K/32 blocks × 16 bytes/block] [d: N rows × K/32 blocks × sizeof(half)]
        // Total sizes: qs = N * K/2 bytes, d = N * K/16 bytes
        const int64_t total_blocks = args.N * (args.K / DMMV_BLOCK_SIZE);
        const int64_t d_offset = total_blocks * (DMMV_BLOCK_SIZE / 2);  // Byte offset to scale values

        q.submit([&](sycl::handler & cgh) {
            sycl::nd_range<1> range(
                sycl::range<1>(grid_n * DMMV_WARP_SIZE),  // Global: N * WARP_SIZE threads
                sycl::range<1>(DMMV_WARP_SIZE)           // Local: WARP_SIZE threads per work-group
            );

            cgh.parallel_for<unified_dmmv_kernel_name>(
                range,
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(DMMV_WARP_SIZE)]] {
                    // Work-group handles one output column (n)
                    const int n = item.get_group(0);  // Output column index
                    const int tid = item.get_local_id(0);  // Thread within warp

                    // Bounds check
                    if (n >= args.N) return;

                    const int k_blocks_per_row = args.K / DMMV_BLOCK_SIZE;

                    // Activation pointer (activations are [M=1, K], so just a 1D array)
                    const float * activations = args.activations;

                    // Each thread accumulates partial dot product over its assigned blocks
                    float partial_sum = 0.0f;

                    if (use_soa) {
                        // ==============================================
                        // SoA Layout: Structure of Arrays
                        // ==============================================
                        // Layout: [all qs bytes contiguous][all d values contiguous]
                        // qs: row n starts at qs_base + n * k_blocks_per_row * 16 bytes
                        // d:  row n starts at d_base + n * k_blocks_per_row * sizeof(half)
                        const uint8_t * qs_base = static_cast<const uint8_t *>(args.weights);
                        const sycl::half * d_base = reinterpret_cast<const sycl::half *>(
                            static_cast<const char *>(args.weights) + d_offset);

                        // Calculate base pointers for row n
                        const int row_qs_bytes = k_blocks_per_row * (DMMV_BLOCK_SIZE / 2);  // 16 bytes per block
                        const uint8_t * qs_row = qs_base + n * row_qs_bytes;
                        const sycl::half * d_row = d_base + n * k_blocks_per_row;

                        // Thread tid processes blocks: tid, tid+WARP_SIZE, tid+2*WARP_SIZE, ...
                        for (int block_idx = tid; block_idx < k_blocks_per_row; block_idx += DMMV_WARP_SIZE) {
                            // Get scale factor from contiguous d array
                            const float d = static_cast<float>(d_row[block_idx]);

                            // Get qs pointer for this block (16 bytes per block)
                            const uint8_t * qs = qs_row + block_idx * (DMMV_BLOCK_SIZE / 2);

                            // K offset for activations
                            const int k_offset = block_idx * DMMV_BLOCK_SIZE;

                            // Process all 32 weights in this block
                            float block_sum = 0.0f;
                            #pragma unroll
                            for (int i = 0; i < DMMV_BLOCK_SIZE / 2; i++) {
                                const uint8_t qs_byte = qs[i];
                                const float w0 = static_cast<float>((qs_byte & 0x0F) - 8) * d;
                                const float w1 = static_cast<float>((qs_byte >> 4) - 8) * d;
                                const float a0 = activations[k_offset + i];
                                const float a1 = activations[k_offset + i + 16];
                                block_sum += w0 * a0 + w1 * a1;
                            }
                            partial_sum += block_sum;
                        }
                    } else {
                        // ==============================================
                        // AoS Layout: Array of Structures (original)
                        // ==============================================
                        // Each block is contiguous: [d: fp16][qs: 16 bytes]
                        const block_q4_0_unified * weights =
                            static_cast<const block_q4_0_unified *>(args.weights);

                        // Thread tid processes blocks: tid, tid+WARP_SIZE, tid+2*WARP_SIZE, ...
                        for (int block_idx = tid; block_idx < k_blocks_per_row; block_idx += DMMV_WARP_SIZE) {
                            // Global block index for weight row n
                            const int global_block_idx = n * k_blocks_per_row + block_idx;
                            const block_q4_0_unified * blk = &weights[global_block_idx];

                            // Get scale factor
                            const float d = static_cast<float>(blk->d);

                            // K offset for this block
                            const int k_offset = block_idx * DMMV_BLOCK_SIZE;

                            // Process all 32 weights in this block
                            float block_sum = 0.0f;
                            #pragma unroll
                            for (int i = 0; i < DMMV_BLOCK_SIZE / 2; i++) {
                                // Each byte contains 2 nibbles
                                const uint8_t qs_byte = blk->qs[i];

                                // Low nibble -> position i, High nibble -> position i+16
                                const float w0 = static_cast<float>((qs_byte & 0x0F) - 8) * d;
                                const float w1 = static_cast<float>((qs_byte >> 4) - 8) * d;

                                // Corresponding activation values
                                const float a0 = activations[k_offset + i];
                                const float a1 = activations[k_offset + i + 16];

                                block_sum += w0 * a0 + w1 * a1;
                            }
                            partial_sum += block_sum;
                        }
                    }

                    // Warp-level reduction using shuffle
                    auto sg = item.get_sub_group();
                    #pragma unroll
                    for (int mask = DMMV_WARP_SIZE >> 1; mask > 0; mask >>= 1) {
                        partial_sum += sycl::shift_group_left(sg, partial_sum, mask);
                    }

                    // Thread 0 writes the final result
                    if (tid == 0) {
                        // Output is [M=1, N], so just index by n
                        args.output[n] = partial_sum;
                    }
                }
            );
        });
        return;
    }

    // ==========================================================================
    // Scalar Path: Standard matmul with dequantization
    // ==========================================================================

    // Launch based on tile sizes
    // For simplicity, dispatch to fixed tile sizes initially
    // A more sophisticated version would use template instantiation for common sizes

    if (tile_m == 1) {
        // M=1 fallback: use generic SLM-tiled path (DMMV path above handles batch=1)
        constexpr int TM = 1;
        constexpr int TN = 64;
        constexpr int TK = 32;

        const int tm_grid_m = (static_cast<int>(args.M) + TM - 1) / TM;
        const int tm_grid_n = (static_cast<int>(args.N) + TN - 1) / TN;

        const int wg_m = 1;
        const int wg_n = std::min(TN, 16);

        q.submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> slm_w(TN * TK, cgh);
            sycl::local_accessor<float, 1> slm_a(TM * TK, cgh);

            sycl::nd_range<2> range(
                sycl::range<2>(tm_grid_m * wg_m, tm_grid_n * wg_n),
                sycl::range<2>(wg_m, wg_n)
            );

            cgh.parallel_for<unified_matmul_kernel_name<TM, TN, TK, false>>(
                range,
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                    unified_matmul_kernel_impl<TM, TN, TK, false>(item, args, slm_w, slm_a);
                }
            );
        });
    } else if (tile_m <= 8 && tile_n <= 16 && tile_k <= 32) {
        // Small tiles: 8x16x32
        constexpr int TM = 8;
        constexpr int TN = 16;
        constexpr int TK = 32;

        q.submit([&](sycl::handler & cgh) {
            // Allocate SLM
            // GGML: weights[N,K] -> slm_w[TN * TK]
            // GGML: activations[M,K] -> slm_a[TM * TK]
            sycl::local_accessor<float, 1> slm_w(TN * TK, cgh);  // Weights [TN x TK]
            sycl::local_accessor<float, 1> slm_a(TM * TK, cgh);  // Activations [TM x TK]

            sycl::nd_range<2> range(
                sycl::range<2>(grid_m * wg_size_m, grid_n * wg_size_n),
                sycl::range<2>(wg_size_m, wg_size_n)
            );

            cgh.parallel_for<unified_matmul_kernel_name<TM, TN, TK, false>>(
                range,
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                    unified_matmul_kernel_impl<TM, TN, TK, false>(item, args, slm_w, slm_a);
                }
            );
        });
    } else if (tile_m <= 16 && tile_n <= 32 && tile_k <= 32) {
        // Medium tiles: 16x32x32
        constexpr int TM = 16;
        constexpr int TN = 32;
        constexpr int TK = 32;

        // Recalculate grid dimensions using ACTUAL template tile sizes
        const int tm_grid_m = (static_cast<int>(args.M) + TM - 1) / TM;
        const int tm_grid_n = (static_cast<int>(args.N) + TN - 1) / TN;

        const int wg_m = std::min(TM, 8);
        const int wg_n = std::min(TN, 16);

        if (ggml_sycl_unified_debug_enabled()) {
            fprintf(stderr, "[unified-kernel] MEDIUM path: TM=%d TN=%d TK=%d grid=(%d,%d) wg=(%d,%d)\n",
                    TM, TN, TK, tm_grid_m, tm_grid_n, wg_m, wg_n);
            fflush(stderr);
        }

        q.submit([&](sycl::handler & cgh) {
            // GGML: weights[N,K] -> slm_w[TN * TK]
            // GGML: activations[M,K] -> slm_a[TM * TK]
            sycl::local_accessor<float, 1> slm_w(TN * TK, cgh);  // Weights [TN x TK]
            sycl::local_accessor<float, 1> slm_a(TM * TK, cgh);  // Activations [TM x TK]

            sycl::nd_range<2> range(
                sycl::range<2>(tm_grid_m * wg_m, tm_grid_n * wg_n),
                sycl::range<2>(wg_m, wg_n)
            );

            cgh.parallel_for<unified_matmul_kernel_name<TM, TN, TK, false>>(
                range,
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                    unified_matmul_kernel_impl<TM, TN, TK, false>(item, args, slm_w, slm_a);
                }
            );
        });
        // NOTE: Don't call q.wait_and_throw() here - incompatible with SYCL command graphs.
        // Errors will be caught when the queue is synchronized elsewhere.
    } else {
        // Fallback: use 32x32x32 tiles with dynamic SLM allocation
        // This path handles larger tile sizes
        constexpr int TM = 32;
        constexpr int TN = 32;
        constexpr int TK = 32;

        // IMPORTANT: grid dimensions must match the ACTUAL template tile sizes.
        // Using the caller-provided tile sizes here can under-cover the output.
        const int tm_grid_m = (static_cast<int>(args.M) + TM - 1) / TM;
        const int tm_grid_n = (static_cast<int>(args.N) + TN - 1) / TN;

        const int wg_m = std::min(TM, 8);
        const int wg_n = std::min(TN, 16);

        q.submit([&](sycl::handler & cgh) {
            // GGML: weights[N,K] -> slm_w[TN * TK]
            // GGML: activations[M,K] -> slm_a[TM * TK]
            sycl::local_accessor<float, 1> slm_w(TN * TK, cgh);  // Weights [TN x TK]
            sycl::local_accessor<float, 1> slm_a(TM * TK, cgh);  // Activations [TM x TK]

            sycl::nd_range<2> range(
                sycl::range<2>(tm_grid_m * wg_m, tm_grid_n * wg_n),
                sycl::range<2>(wg_m, wg_n)
            );

            cgh.parallel_for<unified_matmul_kernel_fallback>(
                range,
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                    unified_matmul_kernel_impl<TM, TN, TK, false>(item, args, slm_w, slm_a);
                }
            );
        });
    }
}

}  // namespace ggml_sycl_unified

// =============================================================================
// UnifiedKernel Class Implementation
// =============================================================================

namespace ggml_sycl {

static const char * persistent_op_type_name(OperationType type) {
    switch (type) {
        case OperationType::RMS_NORM:       return "RMS_NORM";
        case OperationType::ADD:            return "ADD";
        case OperationType::MUL:            return "MUL";
        case OperationType::GET_ROWS:       return "GET_ROWS";
        case OperationType::MATMUL_Q_PROJ:  return "MATMUL_Q_PROJ";
        case OperationType::MATMUL_K_PROJ:  return "MATMUL_K_PROJ";
        case OperationType::MATMUL_V_PROJ:  return "MATMUL_V_PROJ";
        case OperationType::MATMUL_OUT_PROJ:return "MATMUL_OUT_PROJ";
        case OperationType::MATMUL_GATE:    return "MATMUL_GATE";
        case OperationType::MATMUL_UP:      return "MATMUL_UP";
        case OperationType::MATMUL_DOWN:    return "MATMUL_DOWN";
        case OperationType::MATMUL_GATE_UP_SILU: return "MATMUL_GATE_UP_SILU";
        case OperationType::ROPE:           return "ROPE";
        case OperationType::ATTENTION_F16:  return "ATTENTION_F16";
        case OperationType::ATTENTION_F32:  return "ATTENTION_F32";
        case OperationType::SILU_MUL:       return "SILU_MUL";
        case OperationType::SET_ROWS:       return "SET_ROWS";
        case OperationType::STRIDED_COPY:   return "STRIDED_COPY";
        case OperationType::SOFTMAX:        return "SOFTMAX";
    }
    return "UNKNOWN";
}

static int persistent_parse_tile_cols_env(const char * env_name, int fallback) {
    if (const char * env = std::getenv(env_name)) {
        char * end = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        if (end && end != env && parsed >= 16 && parsed <= 256 && (parsed % 16) == 0) {
            return static_cast<int>(parsed);
        }
    }
    return fallback;
}

static bool persistent_attention_subgroup_dot_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_SYCL_PERSISTENT_TG_ATTN_SUBGROUP_DOT");
        enabled = (env && std::atoi(env) != 0) ? 1 : 0;
    }
    return enabled != 0;
}

static bool persistent_aggressive_wg_policy_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_SYCL_PERSISTENT_TG_AGGRESSIVE_WG");
        enabled = (env && std::atoi(env) != 0) ? 1 : 0;
    }
    return enabled != 0;
}

// =============================================================================
// Device-Side Persistent Kernel Structures
// =============================================================================

// Packed operation descriptor for device access (aligned for efficient global reads)
struct alignas(64) DeviceOperation {
    int          type;           // OperationType as int
    int          layer;
    const void * weights;
    const void * input;
    void *       output;
    void *       aux;
    const void * mask;
    int64_t      q_nb0;
    int64_t      q_nb1;
    int64_t      q_nb2;
    int64_t      q_nb3;
    int64_t      k_nb0;
    int64_t      k_nb1;
    int64_t      k_nb2;
    int64_t      k_nb3;
    int64_t      v_nb0;
    int64_t      v_nb1;
    int64_t      v_nb2;
    int64_t      v_nb3;
    int          M, N, K;
    int          tile_cols;      // Matmul N columns per tile (0 = default)
    int64_t      output_bytes;
    int          hidden_dim;
    int          intermediate_dim;
    float        eps;
    float        scale;
    int          quant_type;
    int          weight_layout;
    int          n_tiles;        // Number of tiles for this operation
    int          n_kv_heads;     // Number of KV heads for GQA (0 = same as n_heads)
    int          mask_type;      // 0=f32, 1=f16, -1=none
    int64_t      mask_nb0;
    int64_t      mask_nb1;
    int64_t      mask_nb2;
    int64_t      mask_nb3;
    int          mask_ne2;
    int          mask_ne3;
};

// Arguments passed to the persistent kernel
struct PersistentKernelArgs {
    const DeviceOperation * operations;
    int                     n_operations;
    int                     use_split_barrier;
    int                     use_attn_subgroup_dot;
    int *                   tile_counter;
    int *                   barrier_counter;  // Atomic fallback counter (optional)
    int *                   barrier_sense;    // Atomic fallback sense flag (optional)
    void *                  scratch_buffers[4];
    int                     hidden_dim;
    int                     intermediate_dim;
    DeviceDAGState          dag;              // DAG scheduling state
    int                     use_dag;          // 1 = DAG mode, 0 = legacy barriers
};

// =============================================================================
// Persistent Kernel Implementation
// =============================================================================
// This class encapsulates the persistent kernel's work-stealing loop.
// Each work-group processes all operations sequentially, work-stealing tiles
// within each operation. Inter-op synchronization uses device-scope split
// barriers by default, with an atomic sense-reversing fallback for debugging.
//
// SLM layout per operation type:
//   RMS_NORM:     [0..n_warps-1] for cross-warp reduction
//   SILU_MUL:     not used
//   MATMUL:       not used
//   ATTENTION:    [0..head_dim-1] query cache, [head_dim..head_dim+2*N_SGS-1] reduction
// Operations are serialized with device-scope barriers, so SLM is safely reused.

template<int BLOCK_SIZE>
class PersistentTGKernelImpl {
public:
    PersistentTGKernelImpl(const PersistentKernelArgs & args,
                           sycl::local_accessor<float, 1> slm,
                           sycl::nd_item<1> item)
        : args_(args), slm_(slm), item_(item) {}

    void run() {
        const int local_id = item_.get_local_id(0);
        const int wg_id    = item_.get_group_linear_id();
        const int n_wgs    = item_.get_group_range(0);
        const bool use_split_barrier = (args_.use_split_barrier != 0);

        for (int op_idx = 0; op_idx < args_.n_operations; op_idx++) {
            const DeviceOperation & op = args_.operations[op_idx];

            // Work-stealing: each work-group claims tiles atomically
            while (true) {
                int tile_idx = -1;

                // Thread 0 claims the next tile
                if (local_id == 0) {
                    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        counter(*args_.tile_counter);
                    tile_idx = counter.fetch_add(1);
                }

                // Broadcast to all threads in the work-group
                tile_idx = sycl::group_broadcast(item_.get_group(), tile_idx, 0);

                if (tile_idx >= op.n_tiles) break;

                // Dispatch to the appropriate operation handler
                dispatch_operation(op, tile_idx);
            }

            // Synchronize all work-groups between operations.
            // Split barrier mode is the default. Use
            // GGML_SYCL_PERSISTENT_TG_ATOMIC_BARRIER=1 to force atomic fallback.
            if (use_split_barrier) {
                device_split_barrier(local_id, wg_id, /* reset_tile_counter = */ true);
            } else {
                device_barrier_atomic(local_id, n_wgs, /* reset_tile_counter = */ true);
            }
        }
    }

    // DAG-based scheduling: replaces sequential loop + device-scope barriers
    // with per-operation dependency counters and dynamic work scheduling.
    // ZERO device-scope barriers — only intra-WG group_barrier after tile processing.
    void run_dag() {
        const int local_id = item_.get_local_id(0);
        const DeviceDAGState & dag = args_.dag;
        const int n_ops = dag.n_ops;
        int scan_hint = 0;  // start scanning from here (locality optimization)

        while (true) {
            int op_idx = -1;

            // Thread 0 scans for a ready op with unclaimed tiles
            if (local_id == 0) {
                for (int attempt = 0; attempt < n_ops; attempt++) {
                    const int scan = (scan_hint + attempt) % n_ops;
                    sycl::atomic_ref<int, sycl::memory_order::acq_rel,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        rc(dag.ready_counter[scan]);
                    if (rc.load() != 0) continue;  // predecessors pending

                    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        tc(dag.tile_claimed[scan]);
                    if (tc.load() < dag.n_tiles[scan]) {
                        op_idx = scan;
                        break;
                    }
                }
                // Check termination
                if (op_idx < 0) {
                    sycl::atomic_ref<int, sycl::memory_order::acq_rel,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        cc(*dag.completed_count);
                    if (cc.load() >= n_ops) op_idx = -2;  // TERMINATE
                }
            }
            op_idx = sycl::group_broadcast(item_.get_group(), op_idx, 0);

            if (op_idx == -2) break;       // all ops done
            if (op_idx < 0)  continue;     // nothing ready, spin-retry

            // Update scan hint for locality (next scan starts near current op)
            scan_hint = op_idx;

            // Claim and process tiles (same work-stealing pattern as legacy run())
            const DeviceOperation & op = args_.operations[op_idx];
            int my_tiles = 0;
            while (true) {
                int tile_idx = -1;
                if (local_id == 0) {
                    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        tc(dag.tile_claimed[op_idx]);
                    tile_idx = tc.fetch_add(1);
                }
                tile_idx = sycl::group_broadcast(item_.get_group(), tile_idx, 0);
                if (tile_idx >= op.n_tiles) break;
                dispatch_operation(op, tile_idx);
                my_tiles++;
            }

            // Signal completion: last WG to finish this op wakes successors
            if (my_tiles > 0 && local_id == 0) {
                sycl::atomic_ref<int, sycl::memory_order::acq_rel,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    td(dag.tiles_done[op_idx]);
                const int done = td.fetch_add(my_tiles);
                if (done + my_tiles == dag.n_tiles[op_idx]) {
                    // All tiles complete — decrement successors' ready counters
                    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        cc(*dag.completed_count);
                    cc.fetch_add(1);
                    for (int s = dag.successor_offset[op_idx];
                         s < dag.successor_offset[op_idx + 1]; s++) {
                        const int succ = dag.successor_list[s];
                        sycl::atomic_ref<int, sycl::memory_order::acq_rel,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            rc(dag.ready_counter[succ]);
                        rc.fetch_sub(1);
                    }
                }
            }

            // Intra-WG sync only — ensures all threads done before claiming next op
            // NO device-scope barrier!
            sycl::group_barrier(item_.get_group());
        }
    }

private:
    const PersistentKernelArgs &    args_;
    sycl::local_accessor<float, 1> slm_;
    sycl::nd_item<1>               item_;

    // Device-scope split barrier synchronization (default path).
    // Optional tile-counter reset is done by one global thread before barrier.
    void device_split_barrier(int local_id, int wg_id, bool reset_tile_counter = false) {
        sycl::group_barrier(item_.get_group());

        // Device-scope split barrier on Arc is significantly faster when only
        // work-group leaders participate in the global synchronization.
        if (local_id == 0) {
            if (reset_tile_counter && wg_id == 0) {
                sycl::atomic_ref<int, sycl::memory_order::acq_rel,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    tile_counter(*args_.tile_counter);
                tile_counter.store(0);
            }

            constexpr int kDeviceSemantics =
                SemanticsGlobalMem | static_cast<int>(SPIRVMemorySemantics::AcquireRelease);
            split_barrier_arrive(ScopeDevice, kDeviceSemantics);
            split_barrier_wait(ScopeDevice, kDeviceSemantics);
        }

        sycl::group_barrier(item_.get_group());
    }

    // Atomic sense-reversing barrier for device-scope synchronization.
    // All work-groups must call this; it blocks until all n_wgs have arrived.
    // Uses a counter + sense flag to allow reuse across multiple barrier calls.
    // Only thread 0 from each WG participates in the global coordination;
    // all other threads wait via a workgroup barrier.
    // Optional tile-counter reset is done by the last arriving WG before
    // releasing the barrier so next operation can start immediately.
    void device_barrier_atomic(int local_id, int n_wgs, bool reset_tile_counter = false) {
        // First synchronize within the work-group
        sycl::group_barrier(item_.get_group());

        // Only thread 0 per work-group participates in device-scope barrier
        if (local_id == 0) {
            sycl::atomic_ref<int, sycl::memory_order::acq_rel,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                cnt(*args_.barrier_counter);
            sycl::atomic_ref<int, sycl::memory_order::acq_rel,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                sense(*args_.barrier_sense);

            // Read current sense value before arriving
            int cur_sense = sense.load();

            // Last WG to arrive flips the sense and resets the counter
            if (cnt.fetch_add(1) == n_wgs - 1) {
                cnt.store(0);
                if (reset_tile_counter) {
                    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        tile_counter(*args_.tile_counter);
                    tile_counter.store(0);
                }
                sense.store(1 - cur_sense);
            } else {
                // Spin until the sense flips (meaning last WG arrived)
                while (sense.load() == cur_sense) {
                    // Busy-wait
                }
            }
        }

        // Synchronize within WG so all threads see the barrier completion
        sycl::group_barrier(item_.get_group());
    }

    void dispatch_operation(const DeviceOperation & op, int tile_idx) {
        switch (static_cast<OperationType>(op.type)) {
            case OperationType::RMS_NORM:
                compute_rms_norm_tile(op, tile_idx);
                break;
            case OperationType::ADD:
                compute_add_tile(op, tile_idx);
                break;
            case OperationType::MUL:
                compute_mul_tile(op, tile_idx);
                break;
            case OperationType::GET_ROWS:
                compute_get_rows_tile(op, tile_idx);
                break;
            case OperationType::SILU_MUL:
                compute_silu_mul_tile(op, tile_idx);
                break;
            case OperationType::MATMUL_Q_PROJ:
            case OperationType::MATMUL_K_PROJ:
            case OperationType::MATMUL_V_PROJ:
            case OperationType::MATMUL_OUT_PROJ:
            case OperationType::MATMUL_GATE:
            case OperationType::MATMUL_UP:
            case OperationType::MATMUL_DOWN:
                compute_matmul_tile(op, tile_idx);
                break;
            case OperationType::MATMUL_GATE_UP_SILU:
                compute_matmul_gate_up_silu_tile(op, tile_idx);
                break;
            case OperationType::ATTENTION_F16:
            case OperationType::ATTENTION_F32:
                compute_attention_tile(op, tile_idx);
                break;
            case OperationType::ROPE:
                compute_rope_tile(op, tile_idx);
                break;
            case OperationType::SET_ROWS:
                compute_set_rows_tile(op, tile_idx);
                break;
            case OperationType::STRIDED_COPY:
                compute_strided_copy_tile(op, tile_idx);
                break;
            case OperationType::SOFTMAX:
                compute_softmax_tile(op, tile_idx);
                break;
        }
    }

    void compute_rms_norm_tile(const DeviceOperation & op, int tile_idx) {
        // RMS norm is a single-tile cooperative operation (tile_idx ignored)
        (void)tile_idx;

        const int     tid        = item_.get_local_id(0);
        const int     hidden_dim = op.hidden_dim;
        const float   eps        = op.eps;
        const float * input      = static_cast<const float *>(op.input);
        const float * weights    = static_cast<const float *>(op.weights);
        float *       output     = static_cast<float *>(op.output);

        auto      sg      = item_.get_sub_group();
        const int warp_id = sg.get_group_linear_id();
        const int lane_id = sg.get_local_linear_id();
        constexpr int sg_size = 16;
        constexpr int n_warps = BLOCK_SIZE / sg_size;

        // Sum of squares
        float sum_sq = 0.0f;
        for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
            float val = input[i];
            sum_sq += val * val;
        }

        // Subgroup reduction
        sum_sq = sycl::reduce_over_group(sg, sum_sq, sycl::plus<float>());

        // Cross-subgroup reduction via SLM
        if (lane_id == 0) {
            slm_[warp_id] = sum_sq;
        }
        sycl::group_barrier(item_.get_group());

        if (warp_id == 0) {
            sum_sq = (lane_id < n_warps) ? slm_[lane_id] : 0.0f;
            sum_sq = sycl::reduce_over_group(sg, sum_sq, sycl::plus<float>());
            if (lane_id == 0) {
                slm_[0] = sum_sq;
            }
        }
        sycl::group_barrier(item_.get_group());

        // Normalize
        const float rms   = sycl::sqrt(slm_[0] / hidden_dim + eps);
        const float scale = 1.0f / rms;

        for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
            const float w = weights ? weights[i] : 1.0f;
            output[i] = input[i] * scale * w;
        }
    }

    void compute_silu_mul_tile(const DeviceOperation & op, int tile_idx) {
        const int     tid              = item_.get_local_id(0);
        const int     intermediate_dim = op.intermediate_dim;
        const int     tile_size        = BLOCK_SIZE;  // Elements per tile = work-group size
        const int     start            = tile_idx * tile_size;

        const float * gate   = static_cast<const float *>(op.input);
        const float * up     = static_cast<const float *>(op.aux);
        float *       output = static_cast<float *>(op.output);

        const int idx = start + tid;
        if (idx < intermediate_dim) {
            const float g         = gate[idx];
            const float sigmoid_g = 1.0f / (1.0f + sycl::exp(-g));
            output[idx] = g * sigmoid_g * up[idx];
        }
    }

    static inline float load_f32_or_f16(const char * ptr, int type) {
        if (type == 1) {
            return static_cast<float>(*reinterpret_cast<const sycl::half *>(ptr));
        }
        return *reinterpret_cast<const float *>(ptr);
    }

    static inline void store_f32_or_f16(char * ptr, int type, float v) {
        if (type == 1) {
            *reinterpret_cast<sycl::half *>(ptr) = sycl::half(v);
            return;
        }
        *reinterpret_cast<float *>(ptr) = v;
    }

    static inline int64_t load_idx(const char * ptr, int idx_type) {
        if (idx_type == 1) {
            return *reinterpret_cast<const int64_t *>(ptr);
        }
        return static_cast<int64_t>(*reinterpret_cast<const int32_t *>(ptr));
    }

    inline float load_softmax_mask(const DeviceOperation & op, int64_t i01, int64_t i02, int64_t i03, int col) const {
        if (!op.mask || op.mask_type < 0) {
            return 0.0f;
        }
        const int64_t m_ne2 = op.mask_ne2 > 0 ? op.mask_ne2 : 1;
        const int64_t m_ne3 = op.mask_ne3 > 0 ? op.mask_ne3 : 1;
        const int64_t m02   = m_ne2 > 0 ? (i02 % m_ne2) : 0;
        const int64_t m03   = m_ne3 > 0 ? (i03 % m_ne3) : 0;
        const int64_t off   = i01 * op.mask_nb1 + m02 * op.mask_nb2 + m03 * op.mask_nb3 + (int64_t) col * op.mask_nb0;
        const char * mask_b = static_cast<const char *>(op.mask);
        if (op.mask_type == 1) {
            return static_cast<float>(*reinterpret_cast<const sycl::half *>(mask_b + off));
        }
        return *reinterpret_cast<const float *>(mask_b + off);
    }

    void compute_add_tile(const DeviceOperation & op, int tile_idx) {
        const int idx = tile_idx * BLOCK_SIZE + item_.get_local_id(0);
        if (idx >= op.M) {
            return;
        }
        const float * a = static_cast<const float *>(op.input);
        const float * b = static_cast<const float *>(op.aux);
        float *       y = static_cast<float *>(op.output);
        y[idx] = a[idx] + b[idx];
    }

    void compute_mul_tile(const DeviceOperation & op, int tile_idx) {
        const int idx = tile_idx * BLOCK_SIZE + item_.get_local_id(0);
        if (idx >= op.M) {
            return;
        }
        const float * a = static_cast<const float *>(op.input);
        const float * b = static_cast<const float *>(op.aux);
        float *       y = static_cast<float *>(op.output);
        y[idx] = a[idx] * b[idx];
    }

    void compute_get_rows_tile(const DeviceOperation & op, int tile_idx) {
        const int idx = tile_idx * BLOCK_SIZE + item_.get_local_id(0);
        if (idx >= op.M) {
            return;
        }

        const int64_t ne00 = op.q_nb0;
        const int64_t ne10 = op.q_nb1;
        const int64_t ne11 = op.q_nb2;
        const int64_t ne12 = op.q_nb3;
        if (ne00 <= 0 || ne10 <= 0 || ne11 <= 0 || ne12 <= 0) {
            return;
        }

        const int64_t i03 = idx / (ne00 * ne10 * ne11);
        const int64_t r1  = idx - i03 * ne00 * ne10 * ne11;
        const int64_t i02 = r1 / (ne00 * ne10);
        const int64_t r2  = r1 - i02 * ne00 * ne10;
        const int64_t i01 = r2 / ne00;
        const int64_t i00 = r2 - i01 * ne00;

        const int64_t nb01 = op.k_nb0;
        const int64_t nb02 = op.k_nb1;
        const int64_t nb03 = op.k_nb2;
        const int64_t s10  = op.v_nb0;
        const int64_t s11  = op.v_nb1;
        const int64_t s12  = op.v_nb2;
        const int64_t s1   = op.v_nb3;
        const int64_t s2   = op.mask_nb0;
        const int64_t s3   = op.mask_nb1;
        const int     src0_type = op.quant_type;  // 0=f32, 1=f16

        const char * src0 = static_cast<const char *>(op.input);
        const char * src1 = static_cast<const char *>(op.aux);
        float *      dst  = static_cast<float *>(op.output);
        if (!src0 || !src1 || !dst) {
            return;
        }

        const int64_t idx_pos = i01 * s10 + i02 * s11 + i03 * s12;
        const int32_t src_row = *reinterpret_cast<const int32_t *>(src1 + idx_pos * (int64_t) sizeof(int32_t));
        if (src_row < 0) {
            return;
        }

        const int src_elem_size = (src0_type == 1) ? (int) sizeof(sycl::half) : (int) sizeof(float);
        const int64_t src_off = (int64_t) src_row * nb01 + i02 * nb02 + i03 * nb03 + i00 * src_elem_size;
        const float v = load_f32_or_f16(src0 + src_off, src0_type);

        const int64_t dst_off = i00 + i01 * s1 + i02 * s2 + i03 * s3;
        dst[dst_off] = v;
    }

    void compute_set_rows_tile(const DeviceOperation & op, int tile_idx) {
        const int idx = tile_idx * BLOCK_SIZE + item_.get_local_id(0);
        if (idx >= op.M) {
            return;
        }
        const SetRowsMeta * meta = static_cast<const SetRowsMeta *>(op.weights);
        if (!meta || !op.input || !op.aux || !op.output) {
            return;
        }

        const int64_t ne00 = meta->nc;
        const int64_t ne01 = meta->nr;
        const int64_t ne02 = meta->ne02;
        const int64_t ne03 = meta->ne03;
        if (ne00 <= 0 || ne01 <= 0 || ne02 <= 0 || ne03 <= 0) {
            return;
        }

        const int64_t i03 = idx / (ne00 * ne01 * ne02);
        const int64_t r1  = idx - i03 * ne00 * ne01 * ne02;
        const int64_t i02 = r1 / (ne00 * ne01);
        const int64_t r2  = r1 - i02 * ne00 * ne01;
        const int64_t i01 = r2 / ne00;
        const int64_t i00 = r2 - i01 * ne00;

        const int64_t i10 = i01;
        const int64_t i11 = meta->ne11 > 0 ? (i02 % meta->ne11) : 0;
        const int64_t i12 = meta->ne12 > 0 ? (i03 % meta->ne12) : 0;

        const char * src0 = static_cast<const char *>(op.input);
        const char * src1 = static_cast<const char *>(op.aux);
        char *       dst  = static_cast<char *>(op.output);

        const int64_t idx_off = i10 * meta->nb10 + i11 * meta->nb11 + i12 * meta->nb12;
        const int64_t dst_row = load_idx(src1 + idx_off, meta->idx_type);
        if (dst_row < 0 || dst_row >= meta->ne1) {
            return;
        }

        const int src_elem_size = (meta->src_type == 1) ? (int) sizeof(sycl::half) : (int) sizeof(float);
        const int dst_elem_size = (meta->dst_type == 1) ? (int) sizeof(sycl::half) : (int) sizeof(float);
        const int64_t src_off = i01 * meta->nb01 + i02 * meta->nb02 + i03 * meta->nb03 + i00 * src_elem_size;
        const int64_t dst_off = dst_row * meta->nb1 + i02 * meta->nb2 + i03 * meta->nb3 + i00 * dst_elem_size;
        const float v = load_f32_or_f16(src0 + src_off, meta->src_type);
        store_f32_or_f16(dst + dst_off, meta->dst_type, v);
    }

    void compute_strided_copy_tile(const DeviceOperation & op, int tile_idx) {
        const int idx = tile_idx * BLOCK_SIZE + item_.get_local_id(0);
        if (idx >= op.M) {
            return;
        }
        const StridedCopyMeta * meta = static_cast<const StridedCopyMeta *>(op.weights);
        if (!meta || !op.input || !op.output || meta->type_size <= 0) {
            return;
        }

        const int64_t ne0 = meta->ne[0];
        const int64_t ne1 = meta->ne[1] > 0 ? meta->ne[1] : 1;
        const int64_t ne2 = meta->ne[2] > 0 ? meta->ne[2] : 1;
        const int64_t ne3 = meta->ne[3] > 0 ? meta->ne[3] : 1;

        const int64_t i3 = idx / (ne0 * ne1 * ne2);
        const int64_t r1 = idx - i3 * ne0 * ne1 * ne2;
        const int64_t i2 = r1 / (ne0 * ne1);
        const int64_t r2 = r1 - i2 * ne0 * ne1;
        const int64_t i1 = r2 / ne0;
        const int64_t i0 = r2 - i1 * ne0;
        if (i3 >= ne3) {
            return;
        }

        const int64_t src_off = i0 * meta->nb[0] + i1 * meta->nb[1] + i2 * meta->nb[2] + i3 * meta->nb[3];
        const int64_t dst_off = (int64_t) idx * meta->type_size;
        const char * src = static_cast<const char *>(op.input);
        char *       dst = static_cast<char *>(op.output);

        if (meta->type_size == 4) {
            *reinterpret_cast<uint32_t *>(dst + dst_off) = *reinterpret_cast<const uint32_t *>(src + src_off);
        } else if (meta->type_size == 2) {
            *reinterpret_cast<uint16_t *>(dst + dst_off) = *reinterpret_cast<const uint16_t *>(src + src_off);
        } else if (meta->type_size == 1) {
            dst[dst_off] = src[src_off];
        } else {
            for (int b = 0; b < meta->type_size; ++b) {
                dst[dst_off + b] = src[src_off + b];
            }
        }
    }

    void compute_softmax_tile(const DeviceOperation & op, int tile_idx) {
        const int row = tile_idx;
        if (row >= op.M || op.N <= 0 || !op.input || !op.output) {
            return;
        }

        const int tid = item_.get_local_id(0);
        const int n_cols = op.N;
        const float * x = static_cast<const float *>(op.input);
        float *       y = static_cast<float *>(op.output);
        const float   scale = op.scale;

        const int64_t ne01 = op.q_nb0 > 0 ? op.q_nb0 : 1;
        const int64_t ne02 = op.q_nb1 > 0 ? op.q_nb1 : 1;
        const int64_t i03  = row / (ne01 * ne02);
        const int64_t r1   = row - i03 * ne01 * ne02;
        const int64_t i02  = r1 / ne01;
        const int64_t i01  = r1 - i02 * ne01;

        const int64_t row_off = (int64_t) row * n_cols;

        float local_max = -INFINITY;
        for (int col = tid; col < n_cols; col += BLOCK_SIZE) {
            float v = x[row_off + col] * scale + load_softmax_mask(op, i01, i02, i03, col);
            local_max = sycl::fmax(local_max, v);
        }
        const float row_max = sycl::reduce_over_group(item_.get_group(), local_max, sycl::maximum<float>());

        float local_sum = 0.0f;
        for (int col = tid; col < n_cols; col += BLOCK_SIZE) {
            float v = x[row_off + col] * scale + load_softmax_mask(op, i01, i02, i03, col);
            local_sum += sycl::exp(v - row_max);
        }
        const float row_sum = sycl::reduce_over_group(item_.get_group(), local_sum, sycl::plus<float>());
        const float inv_sum = row_sum > 0.0f ? (1.0f / row_sum) : 0.0f;

        for (int col = tid; col < n_cols; col += BLOCK_SIZE) {
            float v = x[row_off + col] * scale + load_softmax_mask(op, i01, i02, i03, col);
            y[row_off + col] = sycl::exp(v - row_max) * inv_sum;
        }
    }

    void compute_matmul_tile(const DeviceOperation & op, int tile_idx) {
        // DMMV (Dequantizing Matrix-Vector Multiply) for M=1 TG workloads.
        // This path is subgroup-local and barrier-free: each lane owns a strided
        // subset of K blocks, keeps activation chunks in registers, and reuses
        // them across multiple output columns in the current tile.

        constexpr int SG_SIZE        = 16;   // Must match reqd_sub_group_size(16)
        constexpr int DMMV_QK4_0     = 32;   // Q4_0 block size
        constexpr int N_SGS          = BLOCK_SIZE / SG_SIZE;  // 16 sub-groups
        constexpr int MAX_ITERS      = 16;   // Supports tile_cols up to 256
        constexpr int QK4_0_PACKED   = DMMV_QK4_0 / 2;        // 16 bytes

        if (op.quant_type != ggml_sycl_unified::QUANT_TYPE_Q4_0) return;

        const int local_id = item_.get_local_id(0);
        const int sg_id    = local_id / SG_SIZE;  // Which sub-group (0-15)
        const int lane_id  = local_id % SG_SIZE;  // Thread within sub-group (0-15)

        const int tile_cols  = op.tile_cols > 0 ? op.tile_cols : 64;
        const int iter_count = (tile_cols + N_SGS - 1) / N_SGS;
        if (iter_count <= 0 || iter_count > MAX_ITERS) return;
        const int tile_start = tile_idx * tile_cols;

        const float * activations = static_cast<const float *>(op.input);
        float *       out         = static_cast<float *>(op.output);
        const int     K           = op.K;
        const int     N           = op.N;
        const int     k_blocks    = K / DMMV_QK4_0;
        if (k_blocks <= 0) return;

        const bool use_soa =
            (static_cast<ggml_sycl_unified::LayoutMode>(op.weight_layout) == ggml_sycl_unified::LayoutMode::SOA);
        const ggml_sycl_unified::block_q4_0_unified * weights =
            static_cast<const ggml_sycl_unified::block_q4_0_unified *>(op.weights);
        const uint8_t * qs_base = static_cast<const uint8_t *>(op.weights);
        const int row_qs_bytes = k_blocks * QK4_0_PACKED;
        const int64_t total_blocks = static_cast<int64_t>(N) * k_blocks;
        const int64_t d_offset = total_blocks * QK4_0_PACKED;  // Byte offset to scale values
        const sycl::half * d_base = reinterpret_cast<const sycl::half *>(
            static_cast<const char *>(op.weights) + d_offset);

        float partial_sums[MAX_ITERS];
        #pragma unroll
        for (int it = 0; it < MAX_ITERS; ++it) {
            partial_sums[it] = 0.0f;
        }

        // Lane-strided K-block loop.
        for (int block_idx = lane_id; block_idx < k_blocks; block_idx += SG_SIZE) {
            const int k_offset = block_idx * DMMV_QK4_0;
            float act_lo[QK4_0_PACKED];
            float act_hi[QK4_0_PACKED];
            #pragma unroll
            for (int i = 0; i < QK4_0_PACKED; ++i) {
                act_lo[i] = activations[k_offset + i];
                act_hi[i] = activations[k_offset + i + QK4_0_PACKED];
            }

            #pragma unroll
            for (int iter = 0; iter < MAX_ITERS; ++iter) {
                if (iter >= iter_count) break;
                const int n = tile_start + iter * N_SGS + sg_id;
                if (n >= N) continue;

                const uint8_t * qs = nullptr;
                float d = 0.0f;
                if (use_soa) {
                    const uint8_t * qs_row = qs_base + static_cast<int64_t>(n) * row_qs_bytes;
                    const sycl::half * d_row = d_base + static_cast<int64_t>(n) * k_blocks;
                    qs = qs_row + block_idx * QK4_0_PACKED;
                    d = static_cast<float>(d_row[block_idx]);
                } else {
                    const int64_t global_block = static_cast<int64_t>(n) * k_blocks + block_idx;
                    const ggml_sycl_unified::block_q4_0_unified * blk = &weights[global_block];
                    qs = blk->qs;
                    d = static_cast<float>(blk->d);
                }

                float block_sum = 0.0f;
                #pragma unroll
                for (int i = 0; i < QK4_0_PACKED; ++i) {
                    const uint8_t qs_byte = qs[i];
                    const float q0 = static_cast<float>((qs_byte & 0x0F) - 8);
                    const float q1 = static_cast<float>((qs_byte >> 4) - 8);
                    block_sum += q0 * act_lo[i] + q1 * act_hi[i];
                }
                partial_sums[iter] += block_sum * d;
            }
        }

        // Final subgroup reduction + output write per N-iteration.
        auto sg = item_.get_sub_group();
        #pragma unroll
        for (int iter = 0; iter < MAX_ITERS; ++iter) {
            if (iter >= iter_count) break;
            const int n = tile_start + iter * N_SGS + sg_id;
            float partial_sum = sycl::reduce_over_group(sg, partial_sums[iter], sycl::plus<float>());
            if (lane_id == 0 && n < N) {
                out[n] = partial_sum;
            }
        }
    }

    void compute_matmul_gate_up_silu_tile(const DeviceOperation & op, int tile_idx) {
        // Fused FFN first stage for TG:
        //   gate = W_gate * x
        //   up   = W_up   * x
        //   y    = silu(gate) * up
        //
        // This reuses the same activation loads for both matmuls and avoids
        // writing/reading intermediate gate/up tensors before SiLU.

        constexpr int SG_SIZE        = 16;
        constexpr int DMMV_QK4_0     = 32;
        constexpr int N_SGS          = BLOCK_SIZE / SG_SIZE;  // 16 sub-groups
        constexpr int MAX_ITERS      = 16;   // Supports tile_cols up to 256
        constexpr int QK4_0_PACKED   = DMMV_QK4_0 / 2;        // 16 bytes

        if (op.quant_type != ggml_sycl_unified::QUANT_TYPE_Q4_0) return;

        const int local_id = item_.get_local_id(0);
        const int sg_id    = local_id / SG_SIZE;
        const int lane_id  = local_id % SG_SIZE;

        const int tile_cols  = op.tile_cols > 0 ? op.tile_cols : 64;
        const int iter_count = (tile_cols + N_SGS - 1) / N_SGS;
        if (iter_count <= 0 || iter_count > MAX_ITERS) return;
        const int tile_start = tile_idx * tile_cols;

        const float * activations = static_cast<const float *>(op.input);
        float *       out         = static_cast<float *>(op.output);
        const int     K           = op.K;
        const int     N           = op.N;
        const int     k_blocks    = K / DMMV_QK4_0;
        if (k_blocks <= 0) return;

        const bool use_soa =
            (static_cast<ggml_sycl_unified::LayoutMode>(op.weight_layout) == ggml_sycl_unified::LayoutMode::SOA);
        const ggml_sycl_unified::block_q4_0_unified * gate_weights =
            static_cast<const ggml_sycl_unified::block_q4_0_unified *>(op.weights);
        const ggml_sycl_unified::block_q4_0_unified * up_weights =
            static_cast<const ggml_sycl_unified::block_q4_0_unified *>(op.aux);
        if (!gate_weights || !up_weights || !activations || !out) return;

        const uint8_t * gate_qs_base = static_cast<const uint8_t *>(op.weights);
        const uint8_t * up_qs_base   = static_cast<const uint8_t *>(op.aux);
        const int row_qs_bytes       = k_blocks * QK4_0_PACKED;
        const int64_t total_blocks   = static_cast<int64_t>(N) * k_blocks;
        const int64_t d_offset       = total_blocks * QK4_0_PACKED;
        const sycl::half * gate_d_base = reinterpret_cast<const sycl::half *>(
            static_cast<const char *>(op.weights) + d_offset);
        const sycl::half * up_d_base = reinterpret_cast<const sycl::half *>(
            static_cast<const char *>(op.aux) + d_offset);

        float partial_gate[MAX_ITERS];
        float partial_up[MAX_ITERS];
        #pragma unroll
        for (int it = 0; it < MAX_ITERS; ++it) {
            partial_gate[it] = 0.0f;
            partial_up[it]   = 0.0f;
        }

        for (int block_idx = lane_id; block_idx < k_blocks; block_idx += SG_SIZE) {
            const int k_offset = block_idx * DMMV_QK4_0;
            float act_lo[QK4_0_PACKED];
            float act_hi[QK4_0_PACKED];
            #pragma unroll
            for (int i = 0; i < QK4_0_PACKED; ++i) {
                act_lo[i] = activations[k_offset + i];
                act_hi[i] = activations[k_offset + i + QK4_0_PACKED];
            }

            #pragma unroll
            for (int iter = 0; iter < MAX_ITERS; ++iter) {
                if (iter >= iter_count) break;
                const int n = tile_start + iter * N_SGS + sg_id;
                if (n >= N) continue;

                const uint8_t * gate_qs = nullptr;
                const uint8_t * up_qs   = nullptr;
                float gate_d            = 0.0f;
                float up_d              = 0.0f;

                if (use_soa) {
                    const uint8_t * gate_qs_row = gate_qs_base + static_cast<int64_t>(n) * row_qs_bytes;
                    const uint8_t * up_qs_row   = up_qs_base + static_cast<int64_t>(n) * row_qs_bytes;
                    const sycl::half * gate_d_row = gate_d_base + static_cast<int64_t>(n) * k_blocks;
                    const sycl::half * up_d_row   = up_d_base + static_cast<int64_t>(n) * k_blocks;
                    gate_qs = gate_qs_row + block_idx * QK4_0_PACKED;
                    up_qs   = up_qs_row + block_idx * QK4_0_PACKED;
                    gate_d  = static_cast<float>(gate_d_row[block_idx]);
                    up_d    = static_cast<float>(up_d_row[block_idx]);
                } else {
                    const int64_t global_block = static_cast<int64_t>(n) * k_blocks + block_idx;
                    const ggml_sycl_unified::block_q4_0_unified * gate_blk = &gate_weights[global_block];
                    const ggml_sycl_unified::block_q4_0_unified * up_blk   = &up_weights[global_block];
                    gate_qs = gate_blk->qs;
                    up_qs   = up_blk->qs;
                    gate_d  = static_cast<float>(gate_blk->d);
                    up_d    = static_cast<float>(up_blk->d);
                }

                float gate_sum = 0.0f;
                float up_sum   = 0.0f;
                #pragma unroll
                for (int i = 0; i < QK4_0_PACKED; ++i) {
                    const float a0 = act_lo[i];
                    const float a1 = act_hi[i];

                    const uint8_t gate_byte = gate_qs[i];
                    const float gate_q0 = static_cast<float>((gate_byte & 0x0F) - 8);
                    const float gate_q1 = static_cast<float>((gate_byte >> 4) - 8);
                    gate_sum += gate_q0 * a0 + gate_q1 * a1;

                    const uint8_t up_byte = up_qs[i];
                    const float up_q0 = static_cast<float>((up_byte & 0x0F) - 8);
                    const float up_q1 = static_cast<float>((up_byte >> 4) - 8);
                    up_sum += up_q0 * a0 + up_q1 * a1;
                }

                partial_gate[iter] += gate_sum * gate_d;
                partial_up[iter]   += up_sum * up_d;
            }
        }

        auto sg = item_.get_sub_group();
        #pragma unroll
        for (int iter = 0; iter < MAX_ITERS; ++iter) {
            if (iter >= iter_count) break;
            const int n = tile_start + iter * N_SGS + sg_id;

            const float gate = sycl::reduce_over_group(sg, partial_gate[iter], sycl::plus<float>());
            const float up   = sycl::reduce_over_group(sg, partial_up[iter], sycl::plus<float>());

            if (lane_id == 0 && n < N) {
                const float sigmoid_gate = 1.0f / (1.0f + sycl::exp(-gate));
                out[n] = gate * sigmoid_gate * up;
            }
        }
    }

    void compute_attention_tile(const DeviceOperation & op, int tile_idx) {
        // Self-attention for M=1 (single query token) in token generation.
        // tile_idx = head index. Each work-group processes one attention head.
        //
        // Fast path: cache attention scores/probabilities in SLM so pass 2 does
        // not recompute Q·K per output dimension.
        constexpr int SG_SIZE = 16;
        constexpr int N_SGS   = BLOCK_SIZE / SG_SIZE;  // 16 sub-groups

        const int tid        = item_.get_local_id(0);
        auto      sg         = item_.get_sub_group();
        const int sg_id      = sg.get_group_linear_id();
        const int lane_id    = sg.get_local_linear_id();
        const int head       = tile_idx;
        const int seq_len    = op.M;
        const int n_heads    = op.N;
        const int head_dim   = op.K;
        const int n_kv_heads = op.n_kv_heads;
        const float scale    = op.scale;
        const bool use_sg_dot = (args_.use_attn_subgroup_dot != 0);

        if (head >= n_heads || seq_len <= 0) return;

        const int kv_head = (n_kv_heads > 0 && n_kv_heads < n_heads)
                            ? head / (n_heads / n_kv_heads)
                            : head;

        const float * q       = static_cast<const float *>(op.input);
        const float * k_cache = static_cast<const float *>(op.weights);
        const float * v_cache = static_cast<const float *>(op.aux);
        float *       output  = static_cast<float *>(op.output);

        const float * q_head = q + head * head_dim;
        const float * k_head = k_cache + kv_head * seq_len * head_dim;
        const float * v_head = v_cache + kv_head * seq_len * head_dim;
        float *       o_head = output + head * head_dim;
        auto wg = item_.get_group();

        // SLM layout:
        //   [0 .. head_dim-1]                  = query vector
        //   [head_dim .. head_dim+2*N_SGS-1]   = reserved reduction scratch
        //   [scores_base ..]                    = score / exp(score-max) cache
        const int slm_reduce_base = head_dim;
        const int slm_scores_base = slm_reduce_base + 2 * N_SGS;
        const int slm_scores_cap  = args_.hidden_dim - slm_scores_base;

        for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
            slm_[d] = q_head[d];
        }
        sycl::group_barrier(wg);

        // Fast path: cache score/probabilities in SLM to avoid pass-2 score recompute.
        if (slm_scores_cap >= seq_len) {
            float local_max = -1e30f;
            if (use_sg_dot) {
                for (int p = sg_id; p < seq_len; p += N_SGS) {
                    const float * k_pos = k_head + p * head_dim;
                    float partial = 0.0f;
                    for (int d = lane_id; d < head_dim; d += SG_SIZE) {
                        partial += slm_[d] * k_pos[d];
                    }
                    float score = sycl::reduce_over_group(sg, partial, sycl::plus<float>());
                    if (lane_id == 0) {
                        score *= scale;
                        slm_[slm_scores_base + p] = score;
                        local_max = sycl::fmax(local_max, score);
                    }
                }
            } else {
                for (int p = tid; p < seq_len; p += BLOCK_SIZE) {
                    float score = 0.0f;
                    const float * k_pos = k_head + p * head_dim;
                    for (int d = 0; d < head_dim; d++) {
                        score += slm_[d] * k_pos[d];
                    }
                    score *= scale;
                    slm_[slm_scores_base + p] = score;
                    local_max = sycl::fmax(local_max, score);
                }
            }

            const float max_contrib = use_sg_dot ? ((lane_id == 0) ? local_max : -1e30f) : local_max;
            const float global_max  = sycl::reduce_over_group(wg, max_contrib, sycl::maximum<float>());

            float local_sum = 0.0f;
            if (use_sg_dot) {
                for (int p = sg_id; p < seq_len; p += N_SGS) {
                    if (lane_id == 0) {
                        const float e = sycl::exp(slm_[slm_scores_base + p] - global_max);
                        slm_[slm_scores_base + p] = e;
                        local_sum += e;
                    }
                }
            } else {
                for (int p = tid; p < seq_len; p += BLOCK_SIZE) {
                    const float e = sycl::exp(slm_[slm_scores_base + p] - global_max);
                    slm_[slm_scores_base + p] = e;
                    local_sum += e;
                }
            }

            const float sum_contrib = use_sg_dot ? ((lane_id == 0) ? local_sum : 0.0f) : local_sum;
            const float global_sum  = sycl::reduce_over_group(wg, sum_contrib, sycl::plus<float>());
            const float inv_sum    = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

            sycl::group_barrier(wg);

            for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
                float acc = 0.0f;
                for (int p = 0; p < seq_len; ++p) {
                    const float prob = slm_[slm_scores_base + p] * inv_sum;
                    acc += prob * v_head[p * head_dim + d];
                }
                o_head[d] = acc;
            }
            return;
        }

        // Fallback when score cache does not fit in SLM.
        float local_max = -1e30f;
        if (use_sg_dot) {
            for (int p = sg_id; p < seq_len; p += N_SGS) {
                const float * k_pos = k_head + p * head_dim;
                float partial = 0.0f;
                for (int d = lane_id; d < head_dim; d += SG_SIZE) {
                    partial += slm_[d] * k_pos[d];
                }
                float score = sycl::reduce_over_group(sg, partial, sycl::plus<float>());
                if (lane_id == 0) {
                    score *= scale;
                    local_max = sycl::fmax(local_max, score);
                }
            }
        } else {
            for (int p = tid; p < seq_len; p += BLOCK_SIZE) {
                float score = 0.0f;
                const float * k_pos = k_head + p * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    score += slm_[d] * k_pos[d];
                }
                score *= scale;
                local_max = sycl::fmax(local_max, score);
            }
        }
        const float max_contrib = use_sg_dot ? ((lane_id == 0) ? local_max : -1e30f) : local_max;
        const float global_max  = sycl::reduce_over_group(wg, max_contrib, sycl::maximum<float>());

        float local_sum = 0.0f;
        if (use_sg_dot) {
            for (int p = sg_id; p < seq_len; p += N_SGS) {
                const float * k_pos = k_head + p * head_dim;
                float partial = 0.0f;
                for (int d = lane_id; d < head_dim; d += SG_SIZE) {
                    partial += slm_[d] * k_pos[d];
                }
                float score = sycl::reduce_over_group(sg, partial, sycl::plus<float>());
                if (lane_id == 0) {
                    score *= scale;
                    local_sum += sycl::exp(score - global_max);
                }
            }
        } else {
            for (int p = tid; p < seq_len; p += BLOCK_SIZE) {
                float score = 0.0f;
                const float * k_pos = k_head + p * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    score += slm_[d] * k_pos[d];
                }
                score *= scale;
                local_sum += sycl::exp(score - global_max);
            }
        }
        const float sum_contrib = use_sg_dot ? ((lane_id == 0) ? local_sum : 0.0f) : local_sum;
        const float global_sum  = sycl::reduce_over_group(wg, sum_contrib, sycl::plus<float>());
        const float inv_sum    = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

        for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
            float acc = 0.0f;
            for (int p = 0; p < seq_len; ++p) {
                float score = 0.0f;
                const float * k_pos = k_head + p * head_dim;
                for (int dd = 0; dd < head_dim; ++dd) {
                    score += slm_[dd] * k_pos[dd];
                }
                score *= scale;
                const float prob = sycl::exp(score - global_max) * inv_sum;
                acc += prob * v_head[p * head_dim + d];
            }
            o_head[d] = acc;
        }
    }

    void compute_rope_tile(const DeviceOperation & op, int tile_idx) {
        // RoPE: Apply rotary position embeddings (NORMAL or NEOX style).
        // cos/sin caches are pre-computed for the current position, size = head_dim/2 each.
        // This is a cooperative operation - all threads in the work-group participate.
        //
        // RoPE mode is encoded in op.scale: 1.0 = NEOX (split pairs), 0.0 = NORMAL (adjacent)
        //   NEOX:   pairs at [i] and [i + half_dim]
        //   NORMAL: pairs at [2*i] and [2*i + 1]
        //
        // Two tensor modes:
        //
        // 1. Dual-tensor mode (n_kv_heads > 0): Q and K are rotated in a single call.
        //    - input  = q_data (in-place read/write)
        //    - aux    = k_data (in-place read/write)
        //    - weights = cos_cache
        //    - output  = sin_cache (field overloaded)
        //
        // 2. Single-tensor mode (n_kv_heads == 0): Only one tensor is rotated.
        //    Used when graph extraction maps each GGML_OP_ROPE node separately.
        //    - input  = source data (read)
        //    - output = destination data (write, may equal input for in-place)
        //    - weights = cos_cache
        //    - aux     = sin_cache (field overloaded)

        (void)tile_idx;  // Single tile, not used

        const int tid        = item_.get_local_id(0);
        const int n_heads    = op.N;
        const int head_dim   = op.K;
        const int n_kv_heads = op.n_kv_heads;
        const int half_dim   = head_dim / 2;
        const bool is_neox   = (op.scale > 0.5f);

        const float * cos_cache = static_cast<const float *>(op.weights);

        if (n_kv_heads > 0) {
            // Dual-tensor mode: rotate both Q and K in-place
            float *       q_data    = const_cast<float *>(static_cast<const float *>(op.input));
            float *       k_data    = static_cast<float *>(op.aux);
            const float * sin_cache_dual = static_cast<const float *>(op.output);

            const int total_heads = n_heads + n_kv_heads;
            const int total_pairs = total_heads * half_dim;

            for (int idx = tid; idx < total_pairs; idx += BLOCK_SIZE) {
                const int head_idx = idx / half_dim;
                const int dim_idx  = idx % half_dim;

                float * data;
                if (head_idx < n_heads) {
                    data = q_data + head_idx * head_dim;
                } else {
                    data = k_data + (head_idx - n_heads) * head_dim;
                }

                const float cos_val = cos_cache[dim_idx];
                const float sin_val = sin_cache_dual[dim_idx];

                if (is_neox) {
                    // NEOX: pairs at [dim_idx] and [dim_idx + half_dim]
                    const float x0 = data[dim_idx];
                    const float x1 = data[dim_idx + half_dim];
                    data[dim_idx]            = x0 * cos_val - x1 * sin_val;
                    data[dim_idx + half_dim] = x0 * sin_val + x1 * cos_val;
                } else {
                    // NORMAL: pairs at [2*dim_idx] and [2*dim_idx + 1]
                    const float x0 = data[2 * dim_idx];
                    const float x1 = data[2 * dim_idx + 1];
                    data[2 * dim_idx]     = x0 * cos_val - x1 * sin_val;
                    data[2 * dim_idx + 1] = x0 * sin_val + x1 * cos_val;
                }
            }
        } else {
            // Single-tensor mode: read from input, write to output
            const float * src_data  = static_cast<const float *>(op.input);
            float *       dst_data  = static_cast<float *>(op.output);
            const float * sin_cache_single = static_cast<const float *>(op.aux);

            const int total_pairs = n_heads * half_dim;

            for (int idx = tid; idx < total_pairs; idx += BLOCK_SIZE) {
                const int head_idx = idx / half_dim;
                const int dim_idx  = idx % half_dim;

                const float * src = src_data + head_idx * head_dim;
                float *       dst = dst_data + head_idx * head_dim;

                const float cos_val = cos_cache[dim_idx];
                const float sin_val = sin_cache_single[dim_idx];

                if (is_neox) {
                    // NEOX: pairs at [dim_idx] and [dim_idx + half_dim]
                    const float x0 = src[dim_idx];
                    const float x1 = src[dim_idx + half_dim];
                    dst[dim_idx]            = x0 * cos_val - x1 * sin_val;
                    dst[dim_idx + half_dim] = x0 * sin_val + x1 * cos_val;
                } else {
                    // NORMAL: pairs at [2*dim_idx] and [2*dim_idx + 1]
                    const float x0 = src[2 * dim_idx];
                    const float x1 = src[2 * dim_idx + 1];
                    dst[2 * dim_idx]     = x0 * cos_val - x1 * sin_val;
                    dst[2 * dim_idx + 1] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }
};

// -----------------------------------------------------------------------------
// Constructor, Destructor, Configuration
// -----------------------------------------------------------------------------

UnifiedKernel::UnifiedKernel(sycl::queue & queue)
    : queue_(queue)
    , device_id_(ggml_sycl_get_device_id_from_queue(queue)) {
    xmx_config_ = {};
    xmx_config_.supported = false;
    last_stats_ = {};
}

UnifiedKernel::~UnifiedKernel() {
    free_persistent_buffers();
    // runtime_tracked_bytes_ is decremented inside free_persistent_buffers()
}

void UnifiedKernel::configure(const ggml_sycl_unified::XMXConfig & xmx_config) {
    xmx_config_ = xmx_config;
    xmx_configured_ = true;
}

bool UnifiedKernel::supports_persistent() const {
    if (!xmx_configured_ || !xmx_config_.supported) {
        return false;
    }
    if (xmx_config_.slm_size < 32 * 1024) {
        return false;
    }
    return true;
}

bool UnifiedKernel::is_building_plan() const {
    return current_plan_ != nullptr;
}

PersistentStats UnifiedKernel::get_last_stats() const {
    return last_stats_;
}

bool UnifiedKernel::persistent_use_split_barrier() const {
    // Default to split barriers. Keep atomic fallback for driver/runtime
    // triage via GGML_SYCL_PERSISTENT_TG_ATOMIC_BARRIER=1.
    if (const char * force_atomic = std::getenv("GGML_SYCL_PERSISTENT_TG_ATOMIC_BARRIER")) {
        if (std::atoi(force_atomic) != 0) {
            return false;
        }
    }
    if (const char * force_split = std::getenv("GGML_SYCL_PERSISTENT_TG_SPLIT_BARRIER")) {
        return std::atoi(force_split) != 0;
    }
    return true;
}

int UnifiedKernel::persistent_matmul_tile_cols(OperationType type, int N, int K) const {
    (void) N;
    (void) K;
    static const int tile_cols_attn =
        persistent_parse_tile_cols_env("GGML_SYCL_PERSISTENT_TG_MATMUL_TILE_N_ATTN", 32);
    static const int tile_cols_ffn =
        persistent_parse_tile_cols_env("GGML_SYCL_PERSISTENT_TG_MATMUL_TILE_N_FFN", 128);

    switch (type) {
        case OperationType::MATMUL_GATE:
        case OperationType::MATMUL_UP:
        case OperationType::MATMUL_DOWN:
        case OperationType::MATMUL_GATE_UP_SILU:
            return tile_cols_ffn;
        case OperationType::MATMUL_Q_PROJ:
        case OperationType::MATMUL_K_PROJ:
        case OperationType::MATMUL_V_PROJ:
        case OperationType::MATMUL_OUT_PROJ:
        default:
            return tile_cols_attn;
    }
}

int UnifiedKernel::persistent_num_workgroups(int total_tiles, bool has_attention, bool has_ffn, bool use_split_barrier) const {
    int n_workgroups = 16;
    if (use_split_barrier) {
        // Split barrier overhead scales poorly with many participating work-groups.
        // Favor a low default here; callers can still override via env vars.
        n_workgroups = 4;
        if (const char * env_split_wgs = std::getenv("GGML_SYCL_PERSISTENT_TG_SPLIT_N_WGS")) {
            char * end = nullptr;
            const long parsed = std::strtol(env_split_wgs, &end, 10);
            if (end && end != env_split_wgs && parsed > 0 && parsed <= 64) {
                n_workgroups = static_cast<int>(parsed);
            }
        }
    } else {
        try {
            const int max_compute_units =
                (int) queue_.get_device().get_info<sycl::info::device::max_compute_units>();
            if (max_compute_units > 0) {
                if (persistent_aggressive_wg_policy_enabled()) {
                    // Aggressive occupancy policy for experimentation/profiling.
                    n_workgroups = max_compute_units * 2;
                    if (has_attention) {
                        n_workgroups += max_compute_units / 2;
                    }
                    if (has_ffn) {
                        n_workgroups += max_compute_units / 2;
                    }
                    n_workgroups = std::clamp(n_workgroups, 8, 128);
                    if (total_tiles > 0) {
                        n_workgroups = std::min(n_workgroups, std::max(1, total_tiles));
                    }
                } else {
                    // Conservative default: ~1 persistent work-group per 4 CUs.
                    n_workgroups = std::clamp(max_compute_units / 4, 8, 32);
                }
            }
        } catch (...) {
            // Keep default when device query is unavailable.
        }
    }

    if (total_tiles > 0) {
        n_workgroups = std::min(n_workgroups, std::max(1, total_tiles));
    }
    if (const char * env_wgs = std::getenv("GGML_SYCL_PERSISTENT_TG_N_WGS")) {
        char * end = nullptr;
        const long parsed = std::strtol(env_wgs, &end, 10);
        if (end && end != env_wgs && parsed > 0 && parsed <= 64) {
            n_workgroups = static_cast<int>(parsed);
        }
    }

    return n_workgroups;
}

// -----------------------------------------------------------------------------
// Buffer Management
// -----------------------------------------------------------------------------

void UnifiedKernel::allocate_persistent_buffers(int hidden_dim, int intermediate_dim) {
    size_t hidden_size = hidden_dim * sizeof(sycl::half);
    size_t ffn_size = intermediate_dim * sizeof(sycl::half);
    size_t required_size = std::max(hidden_size * 4, ffn_size * 2);

    if (persistent_buffer_size_ >= required_size) {
        return;
    }

    free_persistent_buffers();

    for (int i = 0; i < 4; i++) {
        persistent_buffers_[i] = sycl::malloc_device(required_size, queue_);
    }

    if (!sync_block_) {
        sync_block_ = sycl::malloc_device<int>(3, queue_);
    }
    tile_counter_    = sync_block_;
    barrier_counter_ = sync_block_ + 1;
    barrier_sense_   = sync_block_ + 2;
    queue_.memset(sync_block_, 0, 3 * sizeof(int)).wait();

    persistent_buffer_size_ = required_size;

    // Track persistent buffers in cache budget (4 buffers + sync_block)
    const size_t total_bytes = 4 * required_size + 3 * sizeof(int);
    if (device_id_ >= 0) {
        ggml_sycl::unified_cache_add_runtime_bytes(device_id_, total_bytes, ggml_sycl::runtime_category::GRAPH);
        runtime_tracked_bytes_ += total_bytes;
        GGML_SYCL_DEBUG("[UNIFIED-KERNEL] Tracked persistent buffers: %.1f MB on device %d\n",
                        total_bytes / (1024.0f * 1024.0f), device_id_);
    }
}

void UnifiedKernel::free_persistent_buffers() {
    // Untrack from cache budget before freeing
    if (runtime_tracked_bytes_ > 0 && device_id_ >= 0) {
        ggml_sycl::unified_cache_sub_runtime_bytes(device_id_, runtime_tracked_bytes_, ggml_sycl::runtime_category::GRAPH);
        GGML_SYCL_DEBUG("[UNIFIED-KERNEL] Untracked persistent buffers: %.1f MB on device %d\n",
                        runtime_tracked_bytes_ / (1024.0f * 1024.0f), device_id_);
        runtime_tracked_bytes_ = 0;
    }

    for (int i = 0; i < 4; i++) {
        if (persistent_buffers_[i]) {
            sycl::free(persistent_buffers_[i], queue_);
            persistent_buffers_[i] = nullptr;
        }
    }
    if (sync_block_) { sycl::free(sync_block_, queue_); sync_block_ = nullptr; }
    tile_counter_    = nullptr;
    barrier_counter_ = nullptr;
    barrier_sense_   = nullptr;
    if (d_ops_pool_) { sycl::free(d_ops_pool_, queue_); d_ops_pool_ = nullptr; d_ops_pool_size_ = 0; }
    if (get_rows_pool_) {
        if (get_rows_pool_size_ > 0 && device_id_ >= 0) {
            ggml_sycl::unified_cache_sub_runtime_bytes(device_id_, get_rows_pool_size_, ggml_sycl::runtime_category::GRAPH);
        }
        sycl::free(get_rows_pool_, queue_);
        get_rows_pool_ = nullptr;
        get_rows_pool_size_ = 0;
    }
    // Free DAG allocations
    if (dag_allocated_) {
        if (dag_state_.ready_counter)    sycl::free(dag_state_.ready_counter, queue_);
        if (dag_state_.tile_claimed)     sycl::free(dag_state_.tile_claimed, queue_);
        if (dag_state_.tiles_done)       sycl::free(dag_state_.tiles_done, queue_);
        if (dag_state_.successor_offset) sycl::free(dag_state_.successor_offset, queue_);
        if (dag_state_.successor_list)   sycl::free(dag_state_.successor_list, queue_);
        if (dag_state_.n_tiles)          sycl::free(dag_state_.n_tiles, queue_);
        if (dag_state_.completed_count)  sycl::free(dag_state_.completed_count, queue_);
        dag_state_          = {};
        dag_allocated_      = false;
        dag_pool_n_ops_     = 0;
        dag_pool_n_edges_   = 0;
    }
    invalidate_plan_cache();
    persistent_buffer_size_ = 0;
}

// -----------------------------------------------------------------------------
// DAG Scheduling Methods
// -----------------------------------------------------------------------------

void UnifiedKernel::build_dag(const std::vector<std::vector<int>> & successors,
                              const std::vector<int> & in_degree) {
    const int n_ops     = static_cast<int>(in_degree.size());
    int       n_edges   = 0;
    for (const auto & s : successors) {
        n_edges += static_cast<int>(s.size());
    }

    // Reallocate if pool is too small
    if (n_ops > dag_pool_n_ops_ || n_edges > dag_pool_n_edges_) {
        // Free old allocations
        if (dag_allocated_) {
            sycl::free(dag_state_.ready_counter, queue_);
            sycl::free(dag_state_.tile_claimed, queue_);
            sycl::free(dag_state_.tiles_done, queue_);
            sycl::free(dag_state_.successor_offset, queue_);
            sycl::free(dag_state_.successor_list, queue_);
            sycl::free(dag_state_.n_tiles, queue_);
            sycl::free(dag_state_.completed_count, queue_);
        }
        // Allocate new with some headroom
        const int alloc_ops   = n_ops + 64;
        const int alloc_edges = n_edges + 128;
        dag_state_.ready_counter    = sycl::malloc_device<int>(alloc_ops, queue_);
        dag_state_.tile_claimed     = sycl::malloc_device<int>(alloc_ops, queue_);
        dag_state_.tiles_done       = sycl::malloc_device<int>(alloc_ops, queue_);
        dag_state_.successor_offset = sycl::malloc_device<int>(alloc_ops + 1, queue_);
        dag_state_.successor_list   = sycl::malloc_device<int>(std::max(alloc_edges, 1), queue_);
        dag_state_.n_tiles          = sycl::malloc_device<int>(alloc_ops, queue_);
        dag_state_.completed_count  = sycl::malloc_device<int>(1, queue_);
        dag_pool_n_ops_   = alloc_ops;
        dag_pool_n_edges_ = alloc_edges;
    }
    dag_state_.n_ops = n_ops;
    dag_allocated_   = true;

    // Build CSR successor list on host then upload
    std::vector<int> offsets(n_ops + 1);
    std::vector<int> flat_successors;
    flat_successors.reserve(n_edges);
    offsets[0] = 0;
    for (int i = 0; i < n_ops; i++) {
        flat_successors.insert(flat_successors.end(), successors[i].begin(), successors[i].end());
        offsets[i + 1] = static_cast<int>(flat_successors.size());
    }

    // Cache host-side initial state for fast per-token reset
    host_initial_ready_counter_ = in_degree;

    // Upload static topology to device (n_tiles uploaded later in launch_persistent_kernel
    // after tile counts are computed from DeviceOperations)
    queue_.memcpy(dag_state_.successor_offset, offsets.data(), (n_ops + 1) * sizeof(int));
    if (n_edges > 0) {
        queue_.memcpy(dag_state_.successor_list, flat_successors.data(), n_edges * sizeof(int));
    }
    queue_.wait();

    // Log DAG statistics
    int source_count = 0;
    for (int i = 0; i < n_ops; i++) {
        if (in_degree[i] == 0) source_count++;
    }
    GGML_SYCL_DEBUG("[PERSISTENT-TG] DAG built: %d ops, %d edges, %d sources\n",
                    n_ops, n_edges, source_count);
}

void UnifiedKernel::reset_dag_counters() {
    if (!dag_allocated_) return;
    const int n_ops = dag_state_.n_ops;

    // Restore in-degree values (predecessors remaining) from cached initial state
    queue_.memcpy(dag_state_.ready_counter, host_initial_ready_counter_.data(),
                  n_ops * sizeof(int));
    // Reset per-token mutable counters to zero
    queue_.memset(dag_state_.tile_claimed, 0, n_ops * sizeof(int));
    queue_.memset(dag_state_.tiles_done, 0, n_ops * sizeof(int));
    queue_.memset(dag_state_.completed_count, 0, sizeof(int));
    queue_.wait();
}

// -----------------------------------------------------------------------------
// Persistent Plan Building Methods
// -----------------------------------------------------------------------------

void UnifiedKernel::begin_persistent(int n_layers, int batch_size, int hidden_dim,
                                      int intermediate_dim, int n_heads, int n_kv_heads,
                                      int head_dim, int quant_type) {
    cancel_persistent();

    current_plan_ = std::make_unique<PersistentPlan>();
    current_plan_->n_layers = n_layers;
    current_plan_->batch_size = batch_size;
    current_plan_->hidden_dim = hidden_dim;
    current_plan_->intermediate_dim = intermediate_dim;
    current_plan_->n_heads = n_heads;
    current_plan_->n_kv_heads = n_kv_heads;
    current_plan_->head_dim = head_dim;
    current_plan_->quant_type = quant_type;
    current_plan_->operations.reserve(n_layers * 10);

    allocate_persistent_buffers(hidden_dim, intermediate_dim);
}

void UnifiedKernel::add_rms_norm(int layer, const void * weights,
                                 const void * input, void * output,
                                 float eps, int hidden_dim) {
    if (!current_plan_) {
        GGML_LOG_ERROR("UnifiedKernel: add_rms_norm called without begin_persistent\n");
        return;
    }

    OperationDescriptor op = {};
    op.type = OperationType::RMS_NORM;
    op.layer = layer;
    op.weights = weights;
    op.input = input;
    op.output = output;
    op.hidden_dim = hidden_dim > 0 ? hidden_dim : current_plan_->hidden_dim;
    op.eps = eps;

    current_plan_->operations.push_back(op);
}

void UnifiedKernel::add_matmul(int layer, const void * weights, const void * input,
                               void * output, MatmulType type, int M, int N, int K,
                               int quant_type, int weight_layout,
                               const int64_t * weight_nb,
                               const int64_t * input_nb,
                               const int64_t * output_nb,
                               int weight_ne2, int weight_ne3,
                               int input_ne2, int input_ne3) {
    if (!current_plan_) {
        GGML_LOG_ERROR("UnifiedKernel: add_matmul called without begin_persistent\n");
        return;
    }

    OperationDescriptor op = {};

    switch (type) {
        case MatmulType::Q_PROJ:    op.type = OperationType::MATMUL_Q_PROJ; break;
        case MatmulType::K_PROJ:    op.type = OperationType::MATMUL_K_PROJ; break;
        case MatmulType::V_PROJ:    op.type = OperationType::MATMUL_V_PROJ; break;
        case MatmulType::OUT_PROJ:  op.type = OperationType::MATMUL_OUT_PROJ; break;
        case MatmulType::GATE:      op.type = OperationType::MATMUL_GATE; break;
        case MatmulType::UP:        op.type = OperationType::MATMUL_UP; break;
        case MatmulType::DOWN:      op.type = OperationType::MATMUL_DOWN; break;
        default:                    op.type = OperationType::MATMUL_Q_PROJ; break;
    }

    op.layer = layer;
    op.weights = weights;
    op.input = input;
    op.output = output;
    op.M = M;
    op.N = N;
    op.K = K;
    op.quant_type = quant_type;
    op.weight_layout = weight_layout;
    if (weight_nb) {
        op.q_nb0 = weight_nb[0];
        op.q_nb1 = weight_nb[1];
        op.q_nb2 = weight_nb[2];
        op.q_nb3 = weight_nb[3];
    }
    if (input_nb) {
        op.k_nb0 = input_nb[0];
        op.k_nb1 = input_nb[1];
        op.k_nb2 = input_nb[2];
        op.k_nb3 = input_nb[3];
    }
    if (output_nb) {
        op.v_nb0 = output_nb[0];
        op.v_nb1 = output_nb[1];
        op.v_nb2 = output_nb[2];
        op.v_nb3 = output_nb[3];
    }
    // Reuse mask dims to carry batched matmul extent metadata for persistent tiles.
    op.mask_ne2 = input_ne2;
    op.mask_ne3 = input_ne3;
    (void) weight_ne2;
    (void) weight_ne3;

    current_plan_->operations.push_back(op);
}

void UnifiedKernel::add_attention(int layer, const AttentionDescriptor & desc) {
    if (!current_plan_) {
        GGML_LOG_ERROR("UnifiedKernel: add_attention called without begin_persistent\n");
        return;
    }

    OperationDescriptor op = {};
    op.type = (desc.kv_type == KvCacheType::F16)
                  ? OperationType::ATTENTION_F16
                  : OperationType::ATTENTION_F32;
    op.layer = layer;
    op.input = desc.q;
    op.weights = desc.k_cache;
    op.aux = const_cast<void *>(static_cast<const void *>(desc.v_cache));
    op.mask = desc.mask;
    op.output = desc.output;
    op.M = desc.seq_len;
    op.N = desc.n_heads;
    op.K = desc.head_dim;
    op.scale = desc.scale;
    op.n_kv_heads = desc.n_kv_heads;  // GQA: propagate KV head count
    op.q_nb0 = desc.q_nb0;
    op.q_nb1 = desc.q_nb1;
    op.q_nb2 = desc.q_nb2;
    op.q_nb3 = desc.q_nb3;
    op.k_nb0 = desc.k_nb0;
    op.k_nb1 = desc.k_nb1;
    op.k_nb2 = desc.k_nb2;
    op.k_nb3 = desc.k_nb3;
    op.v_nb0 = desc.v_nb0;
    op.v_nb1 = desc.v_nb1;
    op.v_nb2 = desc.v_nb2;
    op.v_nb3 = desc.v_nb3;
    op.mask_type = desc.mask_type;
    op.mask_nb0  = desc.mask_nb0;
    op.mask_nb1  = desc.mask_nb1;
    op.mask_nb2  = desc.mask_nb2;
    op.mask_nb3  = desc.mask_nb3;
    op.mask_ne2  = desc.mask_ne2;
    op.mask_ne3  = desc.mask_ne3;

    current_plan_->operations.push_back(op);
}

void UnifiedKernel::add_silu_mul(int layer, const void * gate, const void * up, void * output) {
    if (!current_plan_) {
        GGML_LOG_ERROR("UnifiedKernel: add_silu_mul called without begin_persistent\n");
        return;
    }

    OperationDescriptor op = {};
    op.type = OperationType::SILU_MUL;
    op.layer = layer;
    op.input = gate;
    op.aux = const_cast<void *>(up);
    op.output = output;
    op.intermediate_dim = current_plan_->intermediate_dim;

    current_plan_->operations.push_back(op);
}

void UnifiedKernel::add_add(int layer, const void * src0, const void * src1, void * output, int n_elements) {
    if (!current_plan_) {
        GGML_LOG_ERROR("UnifiedKernel: add_add called without begin_persistent\n");
        return;
    }

    OperationDescriptor op = {};
    op.type   = OperationType::ADD;
    op.layer  = layer;
    op.input  = src0;
    op.aux    = const_cast<void *>(src1);
    op.output = output;
    op.M      = n_elements;
    current_plan_->operations.push_back(op);
}

void UnifiedKernel::add_mul(int layer, const void * src0, const void * src1, void * output, int n_elements) {
    if (!current_plan_) {
        GGML_LOG_ERROR("UnifiedKernel: add_mul called without begin_persistent\n");
        return;
    }

    OperationDescriptor op = {};
    op.type   = OperationType::MUL;
    op.layer  = layer;
    op.input  = src0;
    op.aux    = const_cast<void *>(src1);
    op.output = output;
    op.M      = n_elements;
    current_plan_->operations.push_back(op);
}

void UnifiedKernel::add_get_rows(int layer, const void * src0, const void * indices, void * output,
                                 int n_elements, int64_t ne00, int64_t ne10, int64_t ne11, int64_t ne12,
                                 int64_t nb01, int64_t nb02, int64_t nb03,
                                 int64_t s10, int64_t s11, int64_t s12,
                                 int64_t s1, int64_t s2, int64_t s3, int src0_type) {
    if (!current_plan_) {
        GGML_LOG_ERROR("UnifiedKernel: add_get_rows called without begin_persistent\n");
        return;
    }

    OperationDescriptor op = {};
    op.type       = OperationType::GET_ROWS;
    op.layer      = layer;
    op.input      = src0;
    op.aux        = const_cast<void *>(indices);
    op.output     = output;
    op.M          = n_elements;
    op.q_nb0      = ne00;
    op.q_nb1      = ne10;
    op.q_nb2      = ne11;
    op.q_nb3      = ne12;
    op.k_nb0      = nb01;
    op.k_nb1      = nb02;
    op.k_nb2      = nb03;
    op.v_nb0      = s10;
    op.v_nb1      = s11;
    op.v_nb2      = s12;
    op.v_nb3      = s1;
    op.mask_nb0   = s2;
    op.mask_nb1   = s3;
    op.quant_type = src0_type;
    current_plan_->operations.push_back(op);
}

void UnifiedKernel::add_set_rows(int layer, const void * src0, const void * indices,
                                 void * dst, const SetRowsMeta * meta, int n_elements,
                                 const void * debug_ptr, int64_t output_bytes) {
    if (!current_plan_) {
        GGML_LOG_ERROR("UnifiedKernel: add_set_rows called without begin_persistent\n");
        return;
    }

    OperationDescriptor op = {};
    op.type         = OperationType::SET_ROWS;
    op.layer        = layer;
    op.input        = src0;
    op.aux          = const_cast<void *>(indices);
    op.output       = dst;
    op.weights      = meta;
    op.mask         = debug_ptr;
    op.M            = n_elements;
    op.output_bytes = output_bytes;
    current_plan_->operations.push_back(op);
}

void UnifiedKernel::add_strided_copy(int layer, const void * src, void * dst,
                                     const StridedCopyMeta * meta, int n_elements,
                                     int64_t output_bytes) {
    if (!current_plan_) {
        GGML_LOG_ERROR("UnifiedKernel: add_strided_copy called without begin_persistent\n");
        return;
    }

    OperationDescriptor op = {};
    op.type         = OperationType::STRIDED_COPY;
    op.layer        = layer;
    op.input        = src;
    op.output       = dst;
    op.weights      = meta;
    op.M            = n_elements;
    op.output_bytes = output_bytes;
    current_plan_->operations.push_back(op);
}

void UnifiedKernel::add_softmax(int layer, const void * input, const void * mask,
                                const void * sinks, void * output, int n_rows, int n_cols,
                                int ne01, int ne02, int ne03, float scale, float max_bias,
                                int mask_type, int64_t mask_nb0, int64_t mask_nb1,
                                int64_t mask_nb2, int64_t mask_nb3, int mask_ne2, int mask_ne3) {
    if (!current_plan_) {
        GGML_LOG_ERROR("UnifiedKernel: add_softmax called without begin_persistent\n");
        return;
    }

    OperationDescriptor op = {};
    op.type      = OperationType::SOFTMAX;
    op.layer     = layer;
    op.input     = input;
    op.mask      = mask;
    op.aux       = const_cast<void *>(sinks);
    op.output    = output;
    op.M         = n_rows;
    op.N         = n_cols;
    op.K         = ne01;
    op.q_nb0     = ne01;
    op.q_nb1     = ne02;
    op.q_nb2     = ne03;
    op.scale     = scale;
    op.eps       = max_bias;
    op.mask_type = mask_type;
    op.mask_nb0  = mask_nb0;
    op.mask_nb1  = mask_nb1;
    op.mask_nb2  = mask_nb2;
    op.mask_nb3  = mask_nb3;
    op.mask_ne2  = mask_ne2;
    op.mask_ne3  = mask_ne3;
    current_plan_->operations.push_back(op);
}

void UnifiedKernel::set_persistent_debug_attn(float * debug_ptr, int layer, int debug_floats) {
    if (!current_plan_) {
        return;
    }
    current_plan_->debug_attn_ptr    = debug_ptr;
    current_plan_->debug_attn_layer  = layer;
    current_plan_->debug_attn_floats = debug_floats;
}

void UnifiedKernel::set_persistent_debug_rms(float * debug_ptr, int layer, int hidden_dim, int * flag) {
    if (!current_plan_) {
        return;
    }
    current_plan_->debug_rms_ptr   = debug_ptr;
    current_plan_->debug_rms_layer = layer;
    current_plan_->debug_rms_dim   = hidden_dim;
    current_plan_->debug_rms_flag  = flag;
}

void UnifiedKernel::set_persistent_debug_matmul(float * debug_ptr, int layer, MatmulType type, int out_dim, int * flag) {
    if (!current_plan_) {
        return;
    }
    current_plan_->debug_matmul_ptr  = debug_ptr;
    current_plan_->debug_matmul_layer = layer;
    current_plan_->debug_matmul_type  = static_cast<int>(type);
    current_plan_->debug_matmul_dim   = out_dim;
    current_plan_->debug_matmul_flag  = flag;
}

void UnifiedKernel::set_persistent_debug_hash(uint64_t * debug_ptr, int debug_bytes) {
    if (!current_plan_) {
        return;
    }
    current_plan_->debug_hash_ptr   = debug_ptr;
    current_plan_->debug_hash_bytes = debug_bytes;
}

void UnifiedKernel::add_rope(int layer, const RopeDescriptor & desc) {
    if (!current_plan_) {
        GGML_LOG_ERROR("UnifiedKernel: add_rope called without begin_persistent\n");
        return;
    }

    OperationDescriptor op = {};
    op.type = OperationType::ROPE;
    op.layer = layer;
    op.weights = desc.cos_cache;
    op.N = desc.n_heads;
    op.K = desc.head_dim;
    op.M = desc.position;
    // Encode RoPE mode in scale: 1.0 = NEOX (split pairs), 0.0 = NORMAL (adjacent pairs)
    op.scale = desc.is_neox ? 1.0f : 0.0f;

    if (desc.k) {
        // Dual-tensor mode: rotate both Q and K in-place
        //   input  = q_data (in-place)
        //   aux    = k_data (in-place)
        //   output = sin_cache (overloaded)
        op.input     = desc.q;
        op.aux       = desc.k;
        op.output    = const_cast<float *>(desc.sin_cache);
        op.n_kv_heads = current_plan_->n_kv_heads;
    } else {
        // Single-tensor mode: read from input, write to output
        //   input  = source data (read)
        //   output = destination data (write)
        //   aux    = sin_cache (overloaded)
        op.input     = desc.q;            // Source pointer (set by caller)
        op.output    = desc.rope_dst;     // Destination pointer
        op.aux       = const_cast<float *>(desc.sin_cache);
        op.n_kv_heads = 0;
    }

    current_plan_->operations.push_back(op);
}

void UnifiedKernel::add_temp_device_alloc(void * ptr, size_t bytes) {
    if (current_plan_ && ptr) {
        current_plan_->temp_device_allocs.push_back({ptr, bytes});
        current_plan_->temp_device_alloc_bytes += bytes;
        if (device_id_ >= 0) {
            ggml_sycl::unified_cache_add_runtime_bytes(device_id_, bytes, ggml_sycl::runtime_category::GRAPH);
        }
    }
}

void UnifiedKernel::cancel_persistent() {
    if (current_plan_) {
        if (current_plan_->temp_device_alloc_bytes > 0 && device_id_ >= 0) {
            ggml_sycl::unified_cache_sub_runtime_bytes(device_id_, current_plan_->temp_device_alloc_bytes, ggml_sycl::runtime_category::GRAPH);
        }
        for (auto & [ptr, sz] : current_plan_->temp_device_allocs) {
            sycl::free(ptr, queue_);
        }
        current_plan_->temp_device_allocs.clear();
        current_plan_->temp_device_alloc_bytes = 0;
    }
    current_plan_.reset();
}

// -----------------------------------------------------------------------------
// Plan Caching Methods
// -----------------------------------------------------------------------------

void UnifiedKernel::copy_plan_shape(const PersistentPlan & src, PersistentPlan & dst) {
    dst.n_layers         = src.n_layers;
    dst.batch_size       = src.batch_size;
    dst.hidden_dim       = src.hidden_dim;
    dst.intermediate_dim = src.intermediate_dim;
    dst.n_heads          = src.n_heads;
    dst.n_kv_heads       = src.n_kv_heads;
    dst.head_dim         = src.head_dim;
    dst.quant_type       = src.quant_type;
}

bool UnifiedKernel::has_cached_plan() const {
    return plan_cache_valid_;
}

int UnifiedKernel::cached_op_count() const {
    return plan_cache_valid_ ? static_cast<int>(cached_ops_.size()) : 0;
}

OperationType UnifiedKernel::plan_op_type(int op_idx) const {
    if (op_idx < 0) {
        return OperationType::RMS_NORM;
    }
    if (current_plan_ && op_idx < (int) current_plan_->operations.size()) {
        return current_plan_->operations[op_idx].type;
    }
    if (plan_cache_valid_ && op_idx < (int) cached_ops_.size()) {
        return cached_ops_[op_idx].type;
    }
    return OperationType::RMS_NORM;
}

void UnifiedKernel::begin_plan_update() {
    // Cancel any in-flight plan but DON'T free cached data
    if (current_plan_) {
        if (current_plan_->temp_device_alloc_bytes > 0 && device_id_ >= 0) {
            ggml_sycl::unified_cache_sub_runtime_bytes(device_id_, current_plan_->temp_device_alloc_bytes, ggml_sycl::runtime_category::GRAPH);
        }
        for (auto & [ptr, sz] : current_plan_->temp_device_allocs) {
            sycl::free(ptr, queue_);
        }
        current_plan_.reset();
    }

    // Clone from cached template
    current_plan_ = std::make_unique<PersistentPlan>();
    copy_plan_shape(cached_plan_template_, *current_plan_);
    current_plan_->operations        = cached_ops_;  // copy the vector
}

bool UnifiedKernel::get_op_descriptor(int op_idx, OperationDescriptor & out) const {
    if (!current_plan_ || op_idx < 0 || op_idx >= (int) current_plan_->operations.size()) {
        return false;
    }
    out = current_plan_->operations[op_idx];
    return true;
}

bool UnifiedKernel::update_op_descriptor(int op_idx, const OperationDescriptor & desc) {
    if (!current_plan_ || op_idx < 0 || op_idx >= (int) current_plan_->operations.size()) {
        return false;
    }
    current_plan_->operations[op_idx] = desc;
    return true;
}

void UnifiedKernel::update_op_pointers(int op_idx, const void * input, void * output,
                                        const void * aux, const void * mask) {
    if (!current_plan_ || op_idx < 0 || op_idx >= (int) current_plan_->operations.size()) {
        return;
    }
    auto & op = current_plan_->operations[op_idx];
    if (input)  op.input  = input;
    if (output) op.output = output;
    if (aux)    op.aux    = const_cast<void *>(aux);
    if (mask)   op.mask   = mask;
}

void UnifiedKernel::update_op_attention(int op_idx, const void * q, const void * k_cache,
                                         const void * v_cache, const void * mask,
                                         void * output,
                                         int64_t q_nb0, int64_t q_nb1, int64_t q_nb2, int64_t q_nb3,
                                         int64_t k_nb0, int64_t k_nb1, int64_t k_nb2, int64_t k_nb3,
                                         int64_t v_nb0, int64_t v_nb1, int64_t v_nb2, int64_t v_nb3,
                                         int seq_len) {
    if (!current_plan_ || op_idx < 0 || op_idx >= (int) current_plan_->operations.size()) {
        return;
    }
    auto & op  = current_plan_->operations[op_idx];
    op.input   = q;
    op.weights = k_cache;
    op.aux     = const_cast<void *>(v_cache);
    op.mask    = mask;
    op.output  = output;
    op.q_nb0   = q_nb0;  op.q_nb1 = q_nb1;  op.q_nb2 = q_nb2;  op.q_nb3 = q_nb3;
    op.k_nb0   = k_nb0;  op.k_nb1 = k_nb1;  op.k_nb2 = k_nb2;  op.k_nb3 = k_nb3;
    op.v_nb0   = v_nb0;  op.v_nb1 = v_nb1;  op.v_nb2 = v_nb2;  op.v_nb3 = v_nb3;
    op.M       = seq_len;
}

void UnifiedKernel::update_op_rope(int op_idx, void * q, void * k, void * rope_dst,
                                    const float * cos_cache, const float * sin_cache,
                                    int position) {
    if (!current_plan_ || op_idx < 0 || op_idx >= (int) current_plan_->operations.size()) {
        return;
    }
    auto & op = current_plan_->operations[op_idx];
    op.input   = q;
    op.weights = cos_cache;
    op.M       = position;

    if (k) {
        // Dual-tensor mode: input=q, aux=k, output=sin_cache.
        op.aux    = k;
        op.output = const_cast<float *>(sin_cache);
    } else {
        // Single-tensor mode: input=q, aux=sin_cache, output=rope_dst.
        op.aux    = const_cast<float *>(sin_cache);
        op.output = rope_dst;
    }
}

void UnifiedKernel::finish_plan_update() {
    // Plan is already populated with updated pointers, nothing else needed
}

void UnifiedKernel::invalidate_plan_cache() {
    plan_cache_valid_ = false;
    cached_ops_.clear();
    cached_plan_template_ = {};
    if (cached_temp_device_alloc_bytes_ > 0 && device_id_ >= 0) {
        ggml_sycl::unified_cache_sub_runtime_bytes(device_id_, cached_temp_device_alloc_bytes_, ggml_sycl::runtime_category::GRAPH);
    }
    for (auto & [ptr, sz] : cached_temp_device_allocs_) {
        if (ptr) {
            sycl::free(ptr, queue_);
        }
    }
    cached_temp_device_allocs_.clear();
    cached_temp_device_alloc_bytes_ = 0;
}

void * UnifiedKernel::get_rows_stable_ptr(size_t bytes) {
    if (bytes <= get_rows_pool_size_ && get_rows_pool_) {
        return get_rows_pool_;
    }
    // Free old pool and untrack
    if (get_rows_pool_) {
        sycl::free(get_rows_pool_, queue_);
        if (get_rows_pool_size_ > 0 && device_id_ >= 0) {
            ggml_sycl::unified_cache_sub_runtime_bytes(device_id_, get_rows_pool_size_, ggml_sycl::runtime_category::GRAPH);
        }
    }
    get_rows_pool_ = sycl::malloc_device(bytes, queue_);
    get_rows_pool_size_ = get_rows_pool_ ? bytes : 0;
    // Track new pool
    if (get_rows_pool_size_ > 0 && device_id_ >= 0) {
        ggml_sycl::unified_cache_add_runtime_bytes(device_id_, get_rows_pool_size_, ggml_sycl::runtime_category::GRAPH);
    }
    return get_rows_pool_;
}

// -----------------------------------------------------------------------------
// Persistent Execution
// -----------------------------------------------------------------------------

void UnifiedKernel::execute_persistent() {
    if (!current_plan_ || !current_plan_->is_valid()) {
        GGML_LOG_ERROR("UnifiedKernel: execute_persistent called with invalid plan\n");
        return;
    }

    // Launch the persistent kernel
    launch_persistent_kernel();

    // Cache plan template after first successful execution
    if (!plan_cache_valid_) {
        copy_plan_shape(*current_plan_, cached_plan_template_);
        cached_ops_ = current_plan_->operations;
        cached_temp_device_allocs_ = current_plan_->temp_device_allocs;
        cached_temp_device_alloc_bytes_ = current_plan_->temp_device_alloc_bytes;
        current_plan_->temp_device_allocs.clear();
        current_plan_->temp_device_alloc_bytes = 0;
        // Budget stays reserved — ownership transfers to cached allocs
        plan_cache_valid_ = true;
        GGML_SYCL_DEBUG("[PERSISTENT-TG] Plan cached: %zu operations\n", cached_ops_.size());
    }

    // Free non-cached temp allocs
    if (current_plan_->temp_device_alloc_bytes > 0 && device_id_ >= 0) {
        ggml_sycl::unified_cache_sub_runtime_bytes(device_id_, current_plan_->temp_device_alloc_bytes, ggml_sycl::runtime_category::GRAPH);
    }
    for (auto & [ptr, sz] : current_plan_->temp_device_allocs) {
        sycl::free(ptr, queue_);
    }
    current_plan_->temp_device_allocs.clear();
    current_plan_->temp_device_alloc_bytes = 0;

    // Clear the plan after execution (cached copy remains)
    current_plan_.reset();
}

void UnifiedKernel::launch_persistent_kernel() {
    if (!current_plan_ || current_plan_->operations.empty()) {
        return;
    }

    // Build device-side operation table
    const size_t n_ops = current_plan_->operations.size();
    std::vector<DeviceOperation> host_ops;
    host_ops.reserve(n_ops);

    int total_tiles = 0;
    bool has_attention = false;
    bool has_ffn_matmul = false;
    for (size_t i = 0; i < n_ops; i++) {
        const auto & src = current_plan_->operations[i];
        DeviceOperation dst = {};

        dst.type             = static_cast<int>(src.type);
        dst.layer            = src.layer;
        dst.weights          = src.weights;
        dst.input            = src.input;
        dst.output           = src.output;
        dst.aux              = src.aux;
        dst.mask             = src.mask;
        dst.q_nb0            = src.q_nb0;
        dst.q_nb1            = src.q_nb1;
        dst.q_nb2            = src.q_nb2;
        dst.q_nb3            = src.q_nb3;
        dst.k_nb0            = src.k_nb0;
        dst.k_nb1            = src.k_nb1;
        dst.k_nb2            = src.k_nb2;
        dst.k_nb3            = src.k_nb3;
        dst.v_nb0            = src.v_nb0;
        dst.v_nb1            = src.v_nb1;
        dst.v_nb2            = src.v_nb2;
        dst.v_nb3            = src.v_nb3;
        dst.M                = src.M;
        dst.N                = src.N;
        dst.K                = src.K;
        dst.tile_cols        = 0;
        dst.output_bytes     = src.output_bytes;
        dst.hidden_dim       = src.hidden_dim;
        dst.intermediate_dim = src.intermediate_dim;
        dst.eps              = src.eps;
        dst.scale            = src.scale;
        dst.quant_type       = src.quant_type;
        dst.weight_layout    = src.weight_layout;
        dst.n_kv_heads       = src.n_kv_heads;
        dst.mask_type        = src.mask_type;
        dst.mask_nb0         = src.mask_nb0;
        dst.mask_nb1         = src.mask_nb1;
        dst.mask_nb2         = src.mask_nb2;
        dst.mask_nb3         = src.mask_nb3;
        dst.mask_ne2         = src.mask_ne2;
        dst.mask_ne3         = src.mask_ne3;

        // Fuse MATMUL_GATE + MATMUL_UP + SILU_MUL into a single op when the
        // dependency chain is explicit and contiguous in the persistent plan.
        // NOTE: Fusion changes the operation count and invalidates DAG indices
        // (built pre-fusion in extract_persistent_plan). Skip fusion when DAG is active.
        if (!dag_allocated_ && src.type == OperationType::MATMUL_GATE && (i + 2) < n_ops) {
            const auto & up   = current_plan_->operations[i + 1];
            const auto & silu = current_plan_->operations[i + 2];
            const bool contiguous_chain =
                (up.type == OperationType::MATMUL_UP) &&
                (silu.type == OperationType::SILU_MUL) &&
                (src.layer == up.layer) &&
                (up.layer == silu.layer) &&
                (src.input == up.input) &&
                (src.M == up.M) &&
                (src.N == up.N) &&
                (src.K == up.K) &&
                (src.quant_type == up.quant_type) &&
                (src.weight_layout == up.weight_layout) &&
                (silu.input == src.output) &&
                (silu.aux == up.output) &&
                (src.weights != nullptr) &&
                (up.weights != nullptr) &&
                (silu.output != nullptr);

            if (contiguous_chain) {
                dst.type   = static_cast<int>(OperationType::MATMUL_GATE_UP_SILU);
                dst.aux    = const_cast<void *>(up.weights);  // second weight tensor
                dst.output = silu.output;                     // fused SiLU output
                i += 2;
            }
        }

        const OperationType op_type = static_cast<OperationType>(dst.type);

        // Calculate tiles for this operation
        switch (op_type) {
            case OperationType::RMS_NORM:
                dst.n_tiles = 1;  // Single cooperative tile -- one work-group processes this
                break;
            case OperationType::ADD:
            case OperationType::MUL:
            case OperationType::GET_ROWS:
            case OperationType::SET_ROWS:
            case OperationType::STRIDED_COPY:
                dst.n_tiles = (dst.M + 255) / 256;
                break;
            case OperationType::SILU_MUL:
                dst.n_tiles = (dst.intermediate_dim + 255) / 256;
                break;
            case OperationType::MATMUL_Q_PROJ:
            case OperationType::MATMUL_K_PROJ:
            case OperationType::MATMUL_V_PROJ:
            case OperationType::MATMUL_OUT_PROJ:
            case OperationType::MATMUL_GATE:
            case OperationType::MATMUL_UP:
            case OperationType::MATMUL_DOWN:
            case OperationType::MATMUL_GATE_UP_SILU: {
                dst.tile_cols = persistent_matmul_tile_cols(op_type, dst.N, dst.K);
                dst.n_tiles = (dst.N + dst.tile_cols - 1) / dst.tile_cols;
                if (op_type == OperationType::MATMUL_GATE ||
                    op_type == OperationType::MATMUL_UP ||
                    op_type == OperationType::MATMUL_DOWN ||
                    op_type == OperationType::MATMUL_GATE_UP_SILU) {
                    has_ffn_matmul = true;
                }
                break;
            }
            case OperationType::ATTENTION_F16:
            case OperationType::ATTENTION_F32:
                dst.n_tiles = dst.N;  // One tile per head
                has_attention = true;
                break;
            case OperationType::ROPE:
                dst.n_tiles = 1;  // Single cooperative tile -- one work-group processes this
                break;
            case OperationType::SOFTMAX:
                dst.n_tiles = std::max(1, dst.M);  // One row per tile
                break;
            default:
                dst.n_tiles = 1;
        }
        total_tiles += dst.n_tiles;
        host_ops.push_back(dst);
    }

    // Cache host-side tile counts for DAG construction (before device upload)
    host_n_tiles_.resize(host_ops.size());
    for (size_t i = 0; i < host_ops.size(); i++) {
        host_n_tiles_[i] = host_ops[i].n_tiles;
    }

    // Upload tile counts to DAG device array (DAG topology was built earlier in
    // extract_persistent_plan, but tile counts weren't available until now)
    if (dag_allocated_ && dag_state_.n_tiles != nullptr) {
        const int n = static_cast<int>(host_n_tiles_.size());
        queue_.memcpy(dag_state_.n_tiles, host_n_tiles_.data(), n * sizeof(int)).wait();
    }

    // Copy operation table to device (reuse pooled allocation when capacity is sufficient)
    const int n_ops_device = static_cast<int>(host_ops.size());
    if (n_ops_device > d_ops_pool_size_) {
        if (d_ops_pool_) sycl::free(d_ops_pool_, queue_);
        d_ops_pool_ = static_cast<void *>(sycl::malloc_device<DeviceOperation>(n_ops_device, queue_));
        d_ops_pool_size_ = d_ops_pool_ ? n_ops_device : 0;
    }
    DeviceOperation * d_ops = static_cast<DeviceOperation *>(d_ops_pool_);
    queue_.memcpy(d_ops, host_ops.data(), host_ops.size() * sizeof(DeviceOperation)).wait();

    // Kernel configuration
    constexpr int BLOCK_SIZE = 256;
    const bool use_split_barrier = persistent_use_split_barrier();
    int n_workgroups;
    // Check if DAG mode is disabled via environment variable
    bool use_dag_mode = dag_allocated_;
    if (use_dag_mode) {
        static int dag_env_checked = -1;
        if (dag_env_checked < 0) {
            const char * env = std::getenv("GGML_SYCL_PERSISTENT_TG_DAG");
            dag_env_checked = (env != nullptr && std::strcmp(env, "0") == 0) ? 0 : 1;
        }
        use_dag_mode = (dag_env_checked != 0);
    }

    if (use_dag_mode) {
        // DAG mode: no device-scope barriers, so WG count is not barrier-constrained.
        // Use max_compute_units / 2 by default for good parallelism across independent ops.
        try {
            const int max_cu = (int)queue_.get_device().get_info<sycl::info::device::max_compute_units>();
            n_workgroups = std::clamp(max_cu / 2, 4, 64);
        } catch (...) {
            n_workgroups = 16;
        }
        if (const char * env_wgs = std::getenv("GGML_SYCL_PERSISTENT_TG_N_WGS")) {
            char * end = nullptr;
            const long parsed = std::strtol(env_wgs, &end, 10);
            if (end && end != env_wgs && parsed > 0 && parsed <= 128) {
                n_workgroups = static_cast<int>(parsed);
            }
        }
    } else {
        n_workgroups = persistent_num_workgroups(total_tiles, has_attention, has_ffn_matmul, use_split_barrier);
    }
    const int attention_slm = current_plan_->head_dim + 2 * (BLOCK_SIZE / 16);
    const int matmul_slm    = (BLOCK_SIZE / 16) * 32;      // SG lanes x Q4_0 block cache
    const int slm_floats   = std::max({BLOCK_SIZE / 16,               // At least n_warps for reduction
                                       current_plan_->hidden_dim,      // For RMS norm
                                       attention_slm,                  // For attention tile
                                       matmul_slm});                   // For matmul activation staging
    const bool use_attn_subgroup_dot = persistent_attention_subgroup_dot_enabled();
    if (const char * log_policy = std::getenv("GGML_SYCL_PERSISTENT_TG_LOG_POLICY")) {
        if (std::atoi(log_policy) != 0) {
            GGML_LOG_INFO("[PERSISTENT-TG] policy: split=%d n_wgs=%d tiles=%d has_attn=%d has_ffn=%d attn_sg_dot=%d wg_aggr=%d\n",
                          use_split_barrier ? 1 : 0, n_workgroups, total_tiles,
                          has_attention ? 1 : 0, has_ffn_matmul ? 1 : 0,
                          use_attn_subgroup_dot ? 1 : 0,
                          persistent_aggressive_wg_policy_enabled() ? 1 : 0);
        }
    }

    auto run_persistent_kernel = [&](const DeviceOperation * operations, int operation_count) -> double {
        const bool use_dag = use_dag_mode;

        if (use_dag) {
            // Reset DAG scheduling counters for this token
            reset_dag_counters();
        } else {
            // Reset tile counter + barrier state (counter=0, sense=0) in single memset
            queue_.memset(sync_block_, 0, 3 * sizeof(int)).wait();
        }

        PersistentKernelArgs args = {};
        args.operations           = operations;
        args.n_operations         = operation_count;
        args.use_split_barrier    = use_split_barrier ? 1 : 0;
        args.use_attn_subgroup_dot = use_attn_subgroup_dot ? 1 : 0;
        args.tile_counter         = tile_counter_;
        args.barrier_counter      = barrier_counter_;
        args.barrier_sense        = barrier_sense_;
        for (int i = 0; i < 4; i++) {
            args.scratch_buffers[i] = persistent_buffers_[i];
        }
        args.hidden_dim       = current_plan_->hidden_dim;
        args.intermediate_dim = current_plan_->intermediate_dim;
        args.dag              = dag_state_;
        args.use_dag          = use_dag ? 1 : 0;

        const auto start = std::chrono::high_resolution_clock::now();
        queue_.submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> slm(slm_floats, cgh);
            const auto args_copy = args;
            cgh.parallel_for(
                sycl::nd_range<1>(n_workgroups * BLOCK_SIZE, BLOCK_SIZE),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                    PersistentTGKernelImpl<BLOCK_SIZE> kernel(args_copy, slm, item);
                    if (args_copy.use_dag) {
                        kernel.run_dag();
                    } else {
                        kernel.run();
                    }
                });
        });
        queue_.wait();
        const auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    };

    const bool profile_exec_by_op =
        (std::getenv("GGML_SYCL_PERSISTENT_TG_PROFILE_EXEC_BY_OP") != nullptr);
    int profile_exec_iters = 1;
    if (const char * env_iters = std::getenv("GGML_SYCL_PERSISTENT_TG_PROFILE_EXEC_ITERS")) {
        char * end = nullptr;
        const long parsed = std::strtol(env_iters, &end, 10);
        if (end && end != env_iters && parsed > 0 && parsed <= 16) {
            profile_exec_iters = static_cast<int>(parsed);
        }
    }
    if (profile_exec_by_op) {
        constexpr int kTypeCount = static_cast<int>(OperationType::SOFTMAX) + 1;
        std::array<std::vector<DeviceOperation>, kTypeCount> ops_by_type;
        std::array<int, kTypeCount> tiles_by_type = {};

        for (const auto & op : host_ops) {
            const int idx = op.type;
            if (idx < 0 || idx >= kTypeCount) {
                continue;
            }
            ops_by_type[idx].push_back(op);
            tiles_by_type[idx] += op.n_tiles;
        }

        GGML_LOG_INFO("[PERSISTENT-TG] execute profile by-op: iters=%d n_wgs=%d\n",
                      profile_exec_iters, n_workgroups);
        for (int idx = 0; idx < kTypeCount; ++idx) {
            if (ops_by_type[idx].empty()) {
                continue;
            }

            DeviceOperation * d_ops_subset =
                sycl::malloc_device<DeviceOperation>(ops_by_type[idx].size(), queue_);
            if (!d_ops_subset) {
                GGML_LOG_WARN("[PERSISTENT-TG] execute profile: alloc failed for op=%s\n",
                              persistent_op_type_name(static_cast<OperationType>(idx)));
                continue;
            }
            queue_.memcpy(d_ops_subset, ops_by_type[idx].data(),
                          ops_by_type[idx].size() * sizeof(DeviceOperation)).wait();

            double total_ms = 0.0;
            for (int it = 0; it < profile_exec_iters; ++it) {
                total_ms += run_persistent_kernel(d_ops_subset, static_cast<int>(ops_by_type[idx].size()));
            }
            const double avg_ms = total_ms / (double) profile_exec_iters;
            GGML_LOG_INFO("[PERSISTENT-TG] execute profile op=%s ops=%zu tiles=%d "
                          "avg_ms=%.3f total_ms=%.3f\n",
                          persistent_op_type_name(static_cast<OperationType>(idx)),
                          ops_by_type[idx].size(),
                          tiles_by_type[idx],
                          avg_ms, total_ms);

            sycl::free(d_ops_subset, queue_);
        }
    }

    // Launch persistent kernel - single kernel for all operations
    double elapsed_ms = run_persistent_kernel(d_ops, n_ops_device);

    // Record stats
    last_stats_.n_operations         = n_ops_device;
    last_stats_.n_layers             = current_plan_->n_layers;
    last_stats_.total_tiles          = total_tiles;
    last_stats_.kernel_time_ms       = elapsed_ms;

    // Device ops table is pooled — no per-call free needed
}

// -----------------------------------------------------------------------------
// Single Operation Wrappers
// -----------------------------------------------------------------------------

void UnifiedKernel::matmul(const ggml_sycl_unified::UnifiedKernelArgs & args) {
    ggml_sycl_unified::launch_unified_matmul(queue_, args);
}

void UnifiedKernel::rms_norm(const RmsNormDescriptor & desc) {
    const int hidden_dim = desc.hidden_dim;
    const float eps = desc.eps;
    const float * input = static_cast<const float *>(desc.input);
    const float * weights = static_cast<const float *>(desc.weights);
    float * output = static_cast<float *>(desc.output);

    const int block_size = 256;
    const int sg_size = 16;  // Intel XMX subgroup size
    const int n_warps = block_size / sg_size;

    queue_.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> slm_reduce(n_warps, cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(block_size, block_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const int tid = item.get_local_id(0);
                auto sg = item.get_sub_group();
                const int warp_id = sg.get_group_linear_id();
                const int lane_id = sg.get_local_linear_id();

                // Phase 1: Compute sum of squares
                float sum_sq = 0.0f;
                for (int i = tid; i < hidden_dim; i += block_size) {
                    float val = input[i];
                    sum_sq += val * val;
                }

                // Phase 2: Subgroup reduction
                sum_sq = sycl::reduce_over_group(sg, sum_sq, sycl::plus<float>());

                // Phase 3: Cross-subgroup reduction via SLM
                if (lane_id == 0) {
                    slm_reduce[warp_id] = sum_sq;
                }
                sycl::group_barrier(item.get_group());

                if (warp_id == 0) {
                    sum_sq = (lane_id < n_warps) ? slm_reduce[lane_id] : 0.0f;
                    sum_sq = sycl::reduce_over_group(sg, sum_sq, sycl::plus<float>());
                    if (lane_id == 0) {
                        slm_reduce[0] = sum_sq;
                    }
                }
                sycl::group_barrier(item.get_group());

                // Phase 4: Normalize
                const float rms = sycl::sqrt(slm_reduce[0] / hidden_dim + eps);
                const float scale = 1.0f / rms;

                for (int i = tid; i < hidden_dim; i += block_size) {
                    output[i] = input[i] * scale * weights[i];
                }
            });
    });
}

void UnifiedKernel::rope(const RopeDescriptor & desc) {
    // Stub - will be implemented later
    (void)desc;
}

void UnifiedKernel::silu_mul(const void * gate, const void * up, void * output, int dim) {
    const float * gate_f = static_cast<const float *>(gate);
    const float * up_f = static_cast<const float *>(up);
    float * output_f = static_cast<float *>(output);

    const int block_size = 256;
    const int n_blocks = (dim + block_size - 1) / block_size;

    queue_.submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(n_blocks * block_size, block_size),
            [=](sycl::nd_item<1> item) {
                const int gid = item.get_global_id(0);

                if (gid < dim) {
                    const float g = gate_f[gid];
                    const float sigmoid_g = 1.0f / (1.0f + sycl::exp(-g));
                    const float silu_g = g * sigmoid_g;
                    output_f[gid] = silu_g * up_f[gid];
                }
            });
    });
}

void UnifiedKernel::softmax(const void * input, void * output, int n, int stride) {
    // Stub - will be implemented later
    (void)input; (void)output; (void)n; (void)stride;
}

}  // namespace ggml_sycl
