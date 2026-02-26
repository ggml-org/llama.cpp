//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Unified Kernel Architecture for SYCL Matmul
//
// This header defines the unified kernel interface that supports:
// - Q4_0 quantization (initially, other types to follow)
// - Scalar and XMX (dpas) compute paths
// - Configurable tile sizes for auto-tuning
// - Boundary handling for non-aligned dimensions
//
// Design principles:
// - Single kernel entry point with runtime dispatch
// - Template parameters for compile-time optimization
// - SLM-based weight staging for memory efficiency
// - Extensible to multiple quantization formats
//
// XMX Path:
// - Uses Intel joint_matrix extensions for dpas acceleration
// - Tile dimensions: 8x16x8 (M x N x K step) for half precision
// - Requires sub-group size 16 for joint_matrix operations
//

#ifndef GGML_SYCL_UNIFIED_KERNEL_HPP
#define GGML_SYCL_UNIFIED_KERNEL_HPP

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include <sycl/sycl.hpp>

#include "unified-cache.hpp"

// Check for joint_matrix support
#if __has_include(<sycl/ext/oneapi/matrix/matrix.hpp>)
#    define GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE 1
#    include <sycl/ext/oneapi/matrix/matrix.hpp>
#else
#    define GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE 0
#endif

// =============================================================================
// ESIMD dpas Support
// =============================================================================
// Check for ESIMD extension availability for low-level dpas access.
// ESIMD provides explicit SIMD control for XMX instructions.
//
// ESIMD dpas Layout Requirements (FP16):
// - A operand: Row-major [M x K] where M=Repeat=8, K=16
//   Layout: a[m * K + k] for m=0..7, k=0..15
// - B operand: VNNI-packed [K x N] where K=16, N=ExecSize=16
//   Layout: b[(k/2) * N * 2 + n * 2 + (k%2)] for k=0..15, n=0..15
//   This groups consecutive K values together for efficient systolic array processing.
// - Accumulator: Row-major [M x N] where M=8, N=16
//   Layout: acc[m * N + n] for m=0..7, n=0..15
//
// NOTE: These includes MUST be before any namespace declaration to avoid
// namespace collision issues with SYCL internal headers.

#if __has_include(<sycl/ext/intel/esimd.hpp>) && __has_include(<sycl/ext/intel/esimd/xmx/dpas.hpp>)
#    define GGML_SYCL_ESIMD_AVAILABLE 1
#    include <sycl/ext/intel/esimd.hpp>
#    include <sycl/ext/intel/esimd/xmx/dpas.hpp>
// Namespace aliases for cleaner ESIMD dpas code (in global namespace)
namespace esimd = sycl::ext::intel::esimd;
namespace xmx   = sycl::ext::intel::esimd::xmx;
#else
#    define GGML_SYCL_ESIMD_AVAILABLE 0
#endif

// Cooperative ESIMD kernel support
// These kernels use split barriers for work-group synchronization.
// Split barriers (SPV_INTEL_split_barrier) allow non-blocking arrival + deferred wait,
// which can overlap computation with synchronization for better performance.
#ifndef GGML_SYCL_COOPERATIVE_KERNEL_ENABLED
#    define GGML_SYCL_COOPERATIVE_KERNEL_ENABLED 1
#endif

// Large-tile ESIMD kernel support (32x32 output tiles with 64 work-items)
// Uses split barriers for work-group synchronization.
// DISABLED BY DEFAULT: The dpas2 intrinsic configuration used by this kernel
// is not supported on XeLPG (Arc B580). Enable only on PVC/future GPUs.
#ifndef GGML_SYCL_LARGE_TILE_KERNEL_ENABLED
#    define GGML_SYCL_LARGE_TILE_KERNEL_ENABLED 0
#endif

// =============================================================================
// SPIR-V Split Barrier Support (from Intel SYCL*TLA)
// =============================================================================
// Split barriers allow non-blocking arrival + deferred wait, enabling overlap
// of computation with synchronization. This is more efficient than monolithic
// esimd::barrier() which blocks immediately.
//
// Usage pattern:
//   split_barrier_arrive(ScopeWorkgroup);  // Non-blocking: signal "I'm done"
//   // ... do other work while waiting for others ...
//   split_barrier_wait(ScopeWorkgroup);    // Block until all have arrived
//
// Scope options:
//   ScopeWorkgroup (2) - Synchronize all work-items in the work-group
//   ScopeSubgroup (3)  - Synchronize work-items in the sub-group
//
// Requires: -Xspirv-translator "--spirv-ext=+SPV_INTEL_split_barrier"

#ifdef __SYCL_DEVICE_ONLY__
SYCL_EXTERNAL __attribute__((convergent)) void __spirv_ControlBarrierArriveINTEL(
    int execution_scope, int memory_scope, int memory_semantics);
SYCL_EXTERNAL __attribute__((convergent)) void __spirv_ControlBarrierWaitINTEL(
    int execution_scope, int memory_scope, int memory_semantics);
#endif

// SPIR-V scope constants (from SPV_INTEL_split_barrier extension)
enum class SPIRVScope : int {
    CrossDevice = 0,
    Device      = 1,
    Workgroup   = 2,
    Subgroup    = 3,
    Invocation  = 4,
};

// SPIR-V memory semantics (for memory ordering)
enum class SPIRVMemorySemantics : int {
    None            = 0,
    Acquire         = 0x2,
    Release         = 0x4,
    AcquireRelease  = 0x8,
    SubgroupMemory  = 0x80,
    WorkgroupMemory = 0x100,
    CrossWorkgroup  = 0x200,
};

// Convenience aliases
constexpr int ScopeWorkgroup    = static_cast<int>(SPIRVScope::Workgroup);
constexpr int ScopeSubgroup     = static_cast<int>(SPIRVScope::Subgroup);
constexpr int ScopeDevice       = static_cast<int>(SPIRVScope::Device);
constexpr int SemanticsNone     = static_cast<int>(SPIRVMemorySemantics::None);
constexpr int SemanticsWGMem    = static_cast<int>(SPIRVMemorySemantics::WorkgroupMemory);
constexpr int SemanticsGlobalMem = static_cast<int>(SPIRVMemorySemantics::CrossWorkgroup);

// Split barrier helper functions for cleaner kernel code
// These wrap the SPIR-V intrinsics with meaningful names

/**
 * Signal arrival at a split barrier (non-blocking).
 * Call this when work-item has finished its contribution.
 * Does NOT wait for other work-items.
 */
inline void split_barrier_arrive(int scope = ScopeWorkgroup,
                                  int memory_semantics = SemanticsWGMem) {
#ifdef __SYCL_DEVICE_ONLY__
    __spirv_ControlBarrierArriveINTEL(scope, scope, memory_semantics);
#else
    (void)scope;
    (void)memory_semantics;
#endif
}

/**
 * Wait at a split barrier for all arrivals.
 * Blocks until all work-items in the scope have called split_barrier_arrive().
 */
inline void split_barrier_wait(int scope = ScopeWorkgroup,
                                int memory_semantics = SemanticsWGMem) {
#ifdef __SYCL_DEVICE_ONLY__
    __spirv_ControlBarrierWaitINTEL(scope, scope, memory_semantics);
#else
    (void)scope;
    (void)memory_semantics;
#endif
}

/**
 * Combined arrive-and-wait (equivalent to esimd::barrier() but using split barrier API).
 * Use when you need immediate synchronization without overlapping work.
 */
inline void split_barrier_sync(int scope = ScopeWorkgroup,
                                int memory_semantics = SemanticsWGMem) {
    split_barrier_arrive(scope, memory_semantics);
    split_barrier_wait(scope, memory_semantics);
}

// =============================================================================
// Forward Declarations
// =============================================================================

class UnifiedCache;

// =============================================================================
// Operation Types for Persistent Kernel
// =============================================================================

// Operation types for persistent kernel
enum class OperationType {
    RMS_NORM,
    ADD,
    MUL,
    GET_ROWS,
    MATMUL_Q_PROJ,
    MATMUL_K_PROJ,
    MATMUL_V_PROJ,
    MATMUL_OUT_PROJ,
    MATMUL_GATE,
    MATMUL_UP,
    MATMUL_DOWN,
    MATMUL_GATE_UP_SILU,
    ROPE,
    ATTENTION_F16,
    ATTENTION_F32,
    SILU_MUL,
    SET_ROWS,
    STRIDED_COPY,
    SOFTMAX
};

// KV cache element type for attention.
enum class KvCacheType {
    F32 = 0,
    F16 = 1,
};

// Matmul type classification
enum class MatmulType {
    Q_PROJ,
    K_PROJ,
    V_PROJ,
    OUT_PROJ,
    GATE,
    UP,
    DOWN,
    GENERIC
};

// Prefetch priority for cache streaming
enum class PrefetchPriority {
    LOW,
    NORMAL,
    HIGH
};

// =============================================================================
// Operation Descriptors
// =============================================================================

// Descriptor for RMS normalization
struct RmsNormDescriptor {
    const void * input;
    const void * weights;
    void *       output;
    int          hidden_dim;
    float        eps;
};

struct AttentionDescriptor {
    const void * q;
    const void * k_cache;
    const void * v_cache;
    const void * mask;
    void *       output;
    KvCacheType  kv_type;
    int          n_heads;
    int          n_kv_heads;
    int          head_dim;
    int          seq_len;
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
    float        scale;
    int          mask_type;   // 0 = F32, 1 = F16, -1 = none
    int64_t      mask_nb0;
    int64_t      mask_nb1;
    int64_t      mask_nb2;
    int64_t      mask_nb3;
    int          mask_ne2;
    int          mask_ne3;
};

struct RopeDescriptor {
    void *        q;          // Q data pointer (or source tensor for single-tensor mode)
    void *        k;          // K data pointer (nullptr for single-tensor mode)
    void *        rope_dst;   // Output pointer for single-tensor mode (ignored in dual mode)
    const float * cos_cache;
    const float * sin_cache;
    int           n_heads;
    int           head_dim;
    int           position;
    bool          is_neox;    // true = NEOX layout (split pairs), false = NORMAL (adjacent pairs)
};

// Metadata for materializing strided/view tensors into contiguous buffers.
struct StridedCopyMeta {
    int64_t ne[4];
    int64_t nb[4];
    int32_t type_size;
    int32_t pad;
};

// Metadata for GGML_OP_SET_ROWS.
struct SetRowsMeta {
    int64_t nc;
    int64_t nr;
    int64_t ne1;
    int64_t ne02;
    int64_t ne03;
    int64_t ne11;
    int64_t ne12;
    int64_t nb00;
    int64_t nb01;
    int64_t nb02;
    int64_t nb03;
    int64_t nb0;
    int64_t nb1;
    int64_t nb2;
    int64_t nb3;
    int64_t nb10;
    int64_t nb11;
    int64_t nb12;
    int32_t src_type;
    int32_t dst_type;
    int32_t idx_type;
    int32_t pad;
};

struct OperationDescriptor {
    OperationType type;
    int           layer;
    const void *  weights;
    const void *  input;
    void *        output;
    void *        aux;
    const void *  mask;
    int64_t       q_nb0;
    int64_t       q_nb1;
    int64_t       q_nb2;
    int64_t       q_nb3;
    int64_t       k_nb0;
    int64_t       k_nb1;
    int64_t       k_nb2;
    int64_t       k_nb3;
    int64_t       v_nb0;
    int64_t       v_nb1;
    int64_t       v_nb2;
    int64_t       v_nb3;
    int           M, N, K;
    int64_t       output_bytes;
    int           hidden_dim;
    int           intermediate_dim;
    float         eps;
    float         scale;
    int           quant_type;
    int           weight_layout;
    int           n_kv_heads;   // For GQA support in attention (0 = same as n_heads)
    int           mask_type;
    int64_t       mask_nb0;
    int64_t       mask_nb1;
    int64_t       mask_nb2;
    int64_t       mask_nb3;
    int           mask_ne2;
    int           mask_ne3;

    // Scratch pool linkage: which plan operation produces this op's input/aux.
    // -1 means the pointer comes from an external source (weights, KV cache,
    // GET_ROWS stable buffer, etc.) and should NOT be remapped by the scratch pool.
    // Non-negative values are plan operation indices whose scratch output should
    // be used as this op's input/aux after scratch pool allocation.
    // This replaces the broken pointer-based remap that fails when ggml's memory
    // allocator recycles buffer addresses across non-overlapping tensor lifetimes.
    int           input_source_op  = -1;
    int           aux_source_op    = -1;

    // Embedded per-op metadata for SET_ROWS and STRIDED_COPY operations.
    // Stored here to avoid separate device allocations and per-token memcpy uploads.
    // The union overlaps with no other fields; only the relevant member is used
    // based on the operation type.
    union {
        SetRowsMeta       set_rows_meta;
        StridedCopyMeta   strided_copy_meta;
    };
    bool has_embedded_meta = false;  // True when set_rows_meta/strided_copy_meta is populated
};

// =============================================================================
// Update Recipe: compact per-token refresh for persistent TG plan
// =============================================================================
// Instead of iterating all 1158 ggml graph nodes per token (which takes ~2-3ms),
// the update recipe captures exactly which plan ops need which fields refreshed.
// Most ops have scratch-pool-linked input/output that is stable across tokens;
// only a small subset needs per-token pointer resolution.

// Classification of how a pointer should be resolved on each token.
enum class PtrSource : uint8_t {
    STABLE,       // Pointer is stable across tokens (scratch pool, weight, etc.)
    SCRATCH_OP,   // Linked to another op's scratch output (input_source_op/aux_source_op)
    GET_ROWS,     // Comes from GET_ROWS re-execution result
    KV_CACHE,     // KV cache base pointer (stable base, but seq_len changes)
    GGML_TENSOR,  // Must resolve from ggml tensor each token
};

// Per-op entry in the update recipe.  Only ops with at least one non-STABLE
// field appear in the "mutable ops" list; fully-stable ops are skipped entirely.
struct UpdateRecipeEntry {
    int           plan_op_idx;       // Index into cached_ops_ / current_plan_->operations
    OperationType op_type;           // Cached for quick dispatch
    PtrSource     input_src;         // How to resolve op.input
    PtrSource     output_src;        // How to resolve op.output
    PtrSource     aux_src;           // How to resolve op.aux
    PtrSource     mask_src;          // How to resolve op.mask
    int           graph_node_idx;    // ggml graph node index (for tensor access)
    int           get_rows_idx;      // GET_ROWS slot index (for GET_ROWS ops)
};

// =============================================================================
// DAG Scheduling State for Persistent Kernel
// =============================================================================

// Device-side arrays for DAG-based event scheduling.
// Replaces device-scope barriers with per-operation atomic dependency counters.
struct DeviceDAGState {
    // Per-op scheduling state (n_ops elements, reset every token)
    int * ready_counter;      // atomic: predecessors remaining (0 = ready to run)
    int * tile_claimed;       // atomic: work-stealing counter per op
    int * tiles_done;         // atomic: completed tile count per op

    // Static DAG topology (set once during plan build, reused across tokens)
    int * successor_offset;   // [n_ops+1] CSR index into successor_list
    int * successor_list;     // [total_edges] successor op indices
    int * n_tiles;            // [n_ops] tile count per op

    // Termination
    int * completed_count;    // [1] atomic: number of fully completed ops
    int   n_ops;              // total operation count
};

// Phase-based scheduling: pre-computed topological levels for O(1) tile claiming.
// Replaces the O(n_ops) DAG scan with a flat per-phase tile counter.
// Each phase contains ops that are fully independent (all predecessors are in
// earlier phases), so within a phase all tiles can run in any order.
struct DevicePhaseEntry {
    int op_idx;        // Index into the DeviceOperation array
    int tile_offset;   // Cumulative tile offset within this phase (for reverse-mapping flat tile → op)
};

struct DevicePhaseSchedule {
    DevicePhaseEntry * entries;       // [total_ops] Flat array grouped by phase
    int *              phase_offset;  // [n_phases+1] CSR: phase i uses entries[phase_offset[i]..phase_offset[i+1])
    int *              phase_tiles;   // [n_phases] Total tile count for each phase
    int *              phase_type;    // [n_phases] 0=HEAVY (device barrier), 1=LIGHT (flag-based)
    int                n_phases;      // Number of execution phases
    int                total_ops;     // Total number of ops across all phases
};

// =============================================================================
// Role-Based WG Specialization Structures
// =============================================================================

// Role categories for persistent TG work-group specialization.
// Elementwise WGs handle small 1-tile ops (RMS_NORM, MUL, ADD, ROPE, etc.)
// Matmul WGs handle compute-heavy ops (all MUL_MAT variants, ATTENTION)
enum class OpRole : int {
    ELEM   = 0,  // Elementwise (small, 1-few tiles)
    MATMUL = 1,  // Matrix multiply / attention (many tiles)
};

// A sync point between roles: one role completes a segment, the other waits.
struct RoleSyncPoint {
    int op_before;   // Last op index in the completing role's segment
    int op_after;    // First op index in the waiting role's segment
    int from_role;   // Role that signals (0=ELEM, 1=MATMUL)
};

// Per-role segment: a contiguous range of ops assigned to one role.
struct RoleSegment {
    int first_op;    // First op index (into device ops array)
    int last_op;     // Last op index (inclusive)
    int role;        // 0=ELEM, 1=MATMUL
    int total_tiles; // Sum of tiles across all ops in this segment
    int sync_before; // Sync point index to wait on before starting (-1 = none)
    int sync_after;  // Sync point index to signal after completing (-1 = none)
};

// Device-side role schedule passed to the kernel.
struct DeviceRoleSchedule {
    const RoleSegment *  elem_segments;     // [n_elem_segments] Segments for elementwise WGs
    const RoleSegment *  matmul_segments;   // [n_matmul_segments] Segments for matmul WGs
    int *                sync_flags;        // [n_sync_points * 2] Pairs: elem_done, matmul_done
    int *                role_tile_counter; // [1] Atomic tile counter for elem role work stealing
    int *                elem_barrier_cnt;  // [1] Atomic counter for elem-role intra-barrier
    int *                elem_barrier_sense;// [1] Sense flag for elem-role intra-barrier
    int *                mm_barrier_cnt;    // [1] Atomic counter for matmul-role intra-barrier
    int *                mm_barrier_sense;  // [1] Sense flag for matmul-role intra-barrier
    int                  n_elem_segments;   // Number of elementwise segments
    int                  n_matmul_segments; // Number of matmul segments
    int                  n_sync_points;     // Number of cross-role sync points
    int                  n_elem_wgs;        // Number of WGs assigned to elementwise role
};

// =============================================================================
// Persistent Plan and Stats
// =============================================================================

struct PersistentStats {
    int    n_operations;
    int    n_layers;
    int    total_tiles;
    double kernel_time_ms;
    double memory_bandwidth_gbps;
};

struct PersistentPlan {
    int n_layers;
    int batch_size;
    int hidden_dim;
    int intermediate_dim;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int quant_type;

    std::vector<OperationDescriptor> operations;

    void * intermediate_buffers[4];
    int *  tile_counter;
    void * weight_table;

    float * debug_attn_ptr = nullptr;
    int     debug_attn_layer = -1;
    int     debug_attn_floats = 0;
    float * debug_rms_ptr = nullptr;
    int     debug_rms_layer = -1;
    int     debug_rms_dim = 0;
    int *   debug_rms_flag = nullptr;
    float * debug_matmul_ptr = nullptr;
    int     debug_matmul_layer = -1;
    int     debug_matmul_type = -1;
    int     debug_matmul_dim = 0;
    int *   debug_matmul_flag = nullptr;
    int     rms_base_wg_size = 0;
    int     rms_match_base = 0;
    uint64_t * debug_hash_ptr = nullptr;
    int        debug_hash_bytes = 0;

    // Device memory allocations made during plan building (e.g. RoPE cos/sin caches).
    // These are freed after execute_persistent() completes.
    std::vector<std::pair<void *, size_t>> temp_device_allocs;
    std::unordered_map<void *, ggml_sycl::alloc_handle> temp_device_alloc_handles;
    size_t                                               temp_device_alloc_bytes = 0;

    bool is_valid() const { return n_layers > 0 && !operations.empty(); }
};

namespace ggml_sycl_unified {

// =============================================================================
// XMX Tile Dimension Constants
// =============================================================================
// Intel XMX (Xe Matrix eXtensions) tile dimensions for dpas instructions.
// These are hardware-defined constraints for joint_matrix operations.
//
// For half precision (fp16) on Intel Arc GPUs:
// - A matrix: 8x16 (M x K) row-major
// - B matrix: 16x16 (K x N) column-major
// - C matrix: 8x16 (M x N) accumulator
//
// Note: INT8 uses different dimensions (8x32 for K).

constexpr int XMX_TILE_M = 8;   // M dimension of XMX output tile
constexpr int XMX_TILE_N = 16;  // N dimension of XMX output tile
constexpr int XMX_TILE_K = 16;  // K step per dpas instruction (for half precision)

// Large-tile ESIMD kernel dimensions - these are the MAXIMUM values
// Actual values used depend on hardware (see LargeTileConfig)
constexpr int LARGE_TILE_M_MAX = 64;   // Max M dimension (limited by registers/SLM)
constexpr int LARGE_TILE_N_MAX = 64;   // Max N dimension
constexpr int LARGE_TILE_K     = 32;   // K step per iteration (fixed for Q4_0 alignment)

// Legacy constants for backward compatibility (Arc B580 configuration)
constexpr int LARGE_TILE_M = 32;  // M dimension for 64 work-item config
constexpr int LARGE_TILE_N = 32;  // N dimension for 64 work-item config

// Sub-group size required for ESIMD/joint_matrix operations (fixed by hardware)
constexpr int XMX_SUBGROUP_SIZE = 16;

// =============================================================================
// Adaptive Large-Tile Configuration
// =============================================================================
// Hardware-aware tile configuration that adapts to device capabilities.
// Different GPUs have different ESIMD work-group size limits:
// - Arc B580/DG2: 64 work-items max -> 32×32 tiles (4 sub-groups)
// - PVC: Up to 256+ work-items -> 64×64 tiles (16 sub-groups)
//
// The tile configuration is selected at runtime based on XMXConfig.

/**
 * Large-tile ESIMD configuration for a specific hardware target.
 *
 * Each sub-group computes 16×16 output (two 8×16 dpas tiles stacked).
 * Sub-groups are arranged in a 2D grid to cover the full tile output.
 */
struct LargeTileConfig {
    int wg_size;      // Total work-items per work-group
    int sg_cols;      // Sub-groups per row (N dimension)
    int sg_rows;      // Rows of sub-groups (M dimension)
    int tile_m;       // Output M dimension (sg_rows × 16)
    int tile_n;       // Output N dimension (sg_cols × 16)
    int tile_k;       // K dimension per iteration (fixed 32)
    int slm_weights;  // SLM size for weights (tile_n × tile_k × sizeof(half))
    int slm_acts;     // SLM size for activations (tile_m × tile_k × sizeof(half))

    /**
     * Get optimal tile configuration for hardware.
     *
     * @param max_esimd_wg  Maximum ESIMD work-group size from XMXConfig
     * @return LargeTileConfig optimized for the hardware
     */
    static LargeTileConfig for_hardware(int max_esimd_wg) {
        LargeTileConfig cfg;
        cfg.tile_k = LARGE_TILE_K;  // Fixed at 32

        if (max_esimd_wg >= 256) {
            // PVC-class: 256 work-items = 16 sub-groups = 4×4 grid = 64×64 output
            cfg.wg_size  = 256;
            cfg.sg_cols  = 4;
            cfg.sg_rows  = 4;
            cfg.tile_m   = 64;
            cfg.tile_n   = 64;
        } else if (max_esimd_wg >= 128) {
            // Mid-range: 128 work-items = 8 sub-groups = 4×2 grid = 32×64 output
            cfg.wg_size  = 128;
            cfg.sg_cols  = 4;
            cfg.sg_rows  = 2;
            cfg.tile_m   = 32;
            cfg.tile_n   = 64;
        } else {
            // Arc/DG2-class: 64 work-items = 4 sub-groups = 2×2 grid = 32×32 output
            cfg.wg_size  = 64;
            cfg.sg_cols  = 2;
            cfg.sg_rows  = 2;
            cfg.tile_m   = 32;
            cfg.tile_n   = 32;
        }

        cfg.slm_weights = cfg.tile_n * cfg.tile_k;  // half count
        cfg.slm_acts    = cfg.tile_m * cfg.tile_k;  // half count

        return cfg;
    }

    /**
     * Check if this configuration can be used for given dimensions.
     */
    bool can_use(int64_t M, int64_t N, int64_t K) const {
        return M >= tile_m && N >= tile_n && K >= tile_k;
    }
};

// =============================================================================
// Environment Variable Checks for XMX Configuration
// =============================================================================
// These functions use static caching to avoid repeated getenv() calls.
// GGML_SYCL_XMX_ESIMD=0: Disable ESIMD dpas path (enabled by default while optimizing)
// GGML_SYCL_XMX_INT8=1: Enable INT8 quantization in dpas (disabled by default)

/**
 * Check if ESIMD dpas path is enabled via environment.
 *
 * The ESIMD dpas path uses low-level ESIMD instructions instead of joint_matrix.
 * This is enabled by default while optimizing. Set GGML_SYCL_XMX_ESIMD=0 to disable.
 *
 * @return true if ESIMD dpas path is enabled
 */
inline bool use_esimd_dpas() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_SYCL_XMX_ESIMD");
        if (!env) {
            enabled = 1;
        } else {
            enabled = (std::string(env) == "0") ? 0 : 1;
        }
    }
    return enabled != 0;
}

/**
 * Check if INT8 dpas quantization is enabled via environment.
 *
 * When enabled, uses INT8 quantization in dpas instructions (K=32 per step).
 * Default is FP16 (K=16 per step). Set GGML_SYCL_XMX_INT8=1 to enable.
 *
 * @return true if INT8 dpas quantization is enabled
 */
inline bool use_int8_dpas() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_SYCL_XMX_INT8");
        enabled          = (env && std::string(env) == "1") ? 1 : 0;
    }
    return enabled != 0;
}

// =============================================================================
// ESIMD Prefetch Configuration (Phase 4 - Task llama.cpp-attk)
// =============================================================================
// Configurable prefetch distance and cache hints for LSC (Load-Store Cache)
// operations in ESIMD dpas kernels. Prefetching improves memory/compute overlap.
//
// Intel LSC Cache Hints:
// - cached/cached: Data is cached at both L1 and L3 (good for reused data)
// - streaming/uncached: Data bypasses cache (good for one-time-use data)
//
// Prefetch pattern:
// - Weights: streaming/uncached (used once per output element)
// - Activations: cached/cached (reused across N columns)
//
// Environment variables:
// - GGML_SYCL_PREFETCH_DISTANCE: Tiles to prefetch ahead (default: 2)

/**
 * Default prefetch distance (tiles ahead to prefetch).
 *
 * Value 2 provides good balance per Intel recommendations:
 * - Distance 1: Minimal hiding, may not cover full load latency
 * - Distance 2: Good default, hides most memory latency
 * - Distance 4: Aggressive, may over-use resources
 *
 * Configurable via GGML_SYCL_PREFETCH_DISTANCE environment variable.
 */
constexpr int DEFAULT_PREFETCH_DISTANCE = 2;

/**
 * Maximum prefetch distance to avoid resource exhaustion.
 *
 * Prefetching too far ahead wastes cache/TLB resources with data
 * that may be evicted before use.
 */
constexpr int MAX_PREFETCH_DISTANCE = 4;

/**
 * Cache hint policy enum for prefetch operations.
 *
 * Used to configure L1/L3 cache behavior for different data patterns.
 */
enum class CacheHintPolicy {
    CACHED,     ///< L1 cached, L3 cached - for reused data (activations)
    STREAMING,  ///< L1 streaming, L3 uncached - for one-time-use data (weights)
    UNCACHED    ///< Bypass cache entirely
};

/**
 * Get prefetch distance from environment (cached).
 *
 * Reads GGML_SYCL_PREFETCH_DISTANCE environment variable once at initialization.
 * Default is 2 (good balance between latency hiding and resource usage).
 * Value is clamped to [0, MAX_PREFETCH_DISTANCE].
 *
 * @return Prefetch distance in K-tiles
 */
inline int get_prefetch_distance() {
    static int distance = -1;
    if (distance < 0) {
        const char * env = std::getenv("GGML_SYCL_PREFETCH_DISTANCE");
        if (env) {
            int val = std::atoi(env);
            // Clamp to valid range [0, MAX_PREFETCH_DISTANCE]
            if (val < 0) {
                val = 0;
            }
            if (val > MAX_PREFETCH_DISTANCE) {
                val = MAX_PREFETCH_DISTANCE;
            }
            distance = val;
        } else {
            distance = DEFAULT_PREFETCH_DISTANCE;
        }
    }
    return distance;
}

/**
 * Get cache hint policy for weights (used once per element).
 *
 * Weights are loaded, dequantized, used in dpas, and then discarded.
 * Streaming hint tells the cache not to retain this data.
 *
 * @return STREAMING cache hint policy
 */
inline CacheHintPolicy get_weights_cache_hint() {
    return CacheHintPolicy::STREAMING;
}

/**
 * Get cache hint policy for activations (reused across N columns).
 *
 * Activations are loaded once per K-tile and reused across multiple
 * output columns. Caching improves hit rate for subsequent accesses.
 *
 * @return CACHED cache hint policy
 */
inline CacheHintPolicy get_activations_cache_hint() {
    return CacheHintPolicy::CACHED;
}

/**
 * Get maximum SLM usage with prefetch enabled.
 *
 * Calculates worst-case SLM usage for cooperative kernel with double-buffering.
 * This is used to verify SLM doesn't exceed 64KB hardware limit.
 *
 * Layout: 2 buffers x (weights_tile + activations_tile)
 * - Weights: TILE_N x K_TILE x sizeof(half) = 16 x 16 x 2 = 512 bytes
 * - Activations: TILE_M x K_TILE x sizeof(half) = 16 x 16 x 2 = 512 bytes
 * - Double-buffer: 2 x (512 + 512) = 2048 bytes
 *
 * @return Maximum SLM bytes used by prefetch-enabled kernel
 */
inline size_t get_max_slm_usage_with_prefetch() {
    // Cooperative kernel tile dimensions
    constexpr int TILE_M      = 16;  // Work-group M dimension
    constexpr int TILE_N      = 16;  // Work-group N dimension
    constexpr int K_TILE      = 16;  // K dimension per dpas instruction
    constexpr int NUM_BUFFERS = 2;   // Double-buffering

    // Per-buffer sizes
    constexpr size_t weights_size     = TILE_N * K_TILE * sizeof(sycl::half);  // 512 bytes
    constexpr size_t activations_size = TILE_M * K_TILE * sizeof(sycl::half);  // 512 bytes
    constexpr size_t buffer_size      = weights_size + activations_size;       // 1024 bytes

    // Total with double-buffering
    return NUM_BUFFERS * buffer_size;  // 2048 bytes - well under 64KB limit
}

// =============================================================================
// Kernel Path Selection for Batch-Size Gating (Phase 3)
// =============================================================================
// ESIMD dpas kernels have overhead that makes them slower than scalar for small
// batches due to register setup cost, SLM initialization, and dpas instruction
// latency vs ALU ops. This section provides intelligent dispatch that routes to
// ESIMD only when beneficial.
//
// Batch regime guidelines:
// - Batch=1: DMMV (memory-bound, optimized for single vector)
// - Batch<threshold: MMVQ (small batch, still memory-bound)
// - Batch>=threshold: ESIMD dpas (compute-bound, XMX beneficial)
//
// The threshold is configurable via GGML_SYCL_ESIMD_MIN_BATCH (default: 8).

/**
 * Kernel path enum for dispatch decisions.
 *
 * Used by select_kernel_path() to determine which kernel implementation
 * to use based on batch size, hardware capabilities, and environment overrides.
 */
enum class KernelPath {
    DMMV,             ///< Dense matrix-vector multiply (batch=1, memory-bound)
    MMVQ,             ///< Matrix-matrix vector quantized (small batch)
    ESIMD_DPAS,       ///< ESIMD dpas path (large batch, compute-bound)
    ESIMD_LARGE_TILE  ///< ESIMD large tile path (very large batch, 32x64 tiles)
};

/**
 * Get minimum batch size for ESIMD dispatch (cached).
 *
 * Reads GGML_SYCL_ESIMD_MIN_BATCH environment variable once at initialization.
 * Default value is 8 (empirically determined crossover point).
 *
 * @return Minimum batch size for routing to ESIMD path
 */
inline int get_esimd_min_batch() {
    static int min_batch = -1;
    if (min_batch < 0) {
        const char * env = std::getenv("GGML_SYCL_ESIMD_MIN_BATCH");
        if (env) {
            int val   = std::atoi(env);
            min_batch = (val > 0) ? val : 8;  // Clamp to positive
        } else {
            min_batch = 8;                    // Default threshold
        }
    }
    return min_batch;
}

/**
 * Prefer ESIMD for small batches (cached).
 *
 * When GGML_SYCL_UNIFIED_PREFER_ESIMD_SMALL=1 (default), the unified kernel
 * will bias toward ESIMD for small M to improve XMX occupancy vs joint_matrix.
 *
 * @return true if ESIMD should be preferred for small batches
 */
inline bool prefer_esimd_small() {
    static int prefer = -1;
    if (prefer < 0) {
        const char * env = std::getenv("GGML_SYCL_UNIFIED_PREFER_ESIMD_SMALL");
        prefer           = env ? ((std::atoi(env) != 0) ? 1 : 0) : 1;
    }
    return prefer != 0;
}

/**
 * Maximum M to prefer ESIMD over joint_matrix (cached).
 *
 * Controlled by GGML_SYCL_UNIFIED_PREFER_ESIMD_MAX_M (default: 32).
 *
 * @return Maximum M for ESIMD preference
 */
inline int prefer_esimd_max_m() {
    static int max_m = -1;
    if (max_m < 0) {
        const char * env = std::getenv("GGML_SYCL_UNIFIED_PREFER_ESIMD_MAX_M");
        if (env) {
            int val = std::atoi(env);
            max_m   = (val > 0) ? val : 32;
        } else {
            max_m = 32;
        }
    }
    return max_m;
}

/**
 * Check if MMVQ path is forced via environment (cached).
 *
 * When GGML_SYCL_FORCE_MMVQ=1 is set, always use MMVQ path regardless of
 * batch size or ESIMD availability. Useful for testing/debugging.
 *
 * @return true if MMVQ is forced
 */
inline bool env_force_mmvq() {
    static int forced = -1;
    if (forced < 0) {
        const char * env = std::getenv("GGML_SYCL_FORCE_MMVQ");
        forced           = (env && std::string(env) == "1") ? 1 : 0;
    }
    return forced != 0;
}

/**
 * Check if ESIMD path is forced via environment (cached).
 *
 * When GGML_SYCL_FORCE_ESIMD=1 is set, always use ESIMD path regardless of
 * batch size (if hardware supports it). Useful for testing/debugging.
 *
 * @return true if ESIMD is forced
 */
inline bool env_force_esimd() {
    static int forced = -1;
    if (forced < 0) {
        const char * env = std::getenv("GGML_SYCL_FORCE_ESIMD");
        forced           = (env && std::string(env) == "1") ? 1 : 0;
    }
    return forced != 0;
}

/**
 * Get minimum batch size for large-tile ESIMD dispatch (cached).
 *
 * Reads GGML_SYCL_LARGE_TILE_MIN_BATCH environment variable once at initialization.
 * Default value is 128 (empirically determined crossover point for 32x64 tiles).
 *
 * @return Minimum batch size for routing to ESIMD large-tile path
 */
inline int get_large_tile_min_batch() {
    static int min_batch = -1;
    if (min_batch < 0) {
        const char * env = std::getenv("GGML_SYCL_LARGE_TILE_MIN_BATCH");
        if (env) {
            int val   = std::atoi(env);
            min_batch = (val > 0) ? val : 128;  // Clamp to positive
        } else {
            min_batch = 128;                    // Default threshold
        }
    }
    return min_batch;
}

/**
 * Check if large-tile ESIMD path is enabled (cached).
 *
 * When GGML_SYCL_UNIFIED_LARGE_TILE=1 is set, the unified kernel will
 * consider the large-tile ESIMD path (32x64 tiles) for very large batches.
 * Default is disabled (0) until the kernel is validated.
 *
 * @return true if large-tile ESIMD path is enabled
 */
inline bool large_tile_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_SYCL_UNIFIED_LARGE_TILE");
        enabled          = env ? ((std::atoi(env) != 0) ? 1 : 0) : 0;
    }
    return enabled != 0;
}

/**
 * Check if large-tile ESIMD kernel can be used for given dimensions.
 *
 * Large-tile ESIMD requires:
 * - ESIMD available (GGML_SYCL_ESIMD_AVAILABLE macro)
 * - Large-tile enabled via large_tile_enabled()
 * - M >= LARGE_TILE_M (32) - enough rows for large tile
 * - N >= LARGE_TILE_N (64) - enough columns for large tile
 * - K >= LARGE_TILE_K (32) - minimum K for efficient dpas
 *
 * @param M  Output rows
 * @param N  Output columns
 * @param K  Reduction dimension
 * @return true if large-tile ESIMD kernel can be used
 */
inline bool can_use_large_tile_esimd(int64_t M, int64_t N, int64_t K) {
#if GGML_SYCL_ESIMD_AVAILABLE
    if (!large_tile_enabled()) {
        return false;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        return false;
    }
    if (M < LARGE_TILE_M || N < LARGE_TILE_N || K < LARGE_TILE_K) {
        return false;
    }
    return true;
#else
    GGML_UNUSED(M);
    GGML_UNUSED(N);
    GGML_UNUSED(K);
    return false;  // ESIMD not available
#endif
}

// Forward declaration for XMXConfig (defined below)
struct XMXConfig;

// Forward declaration for select_kernel_path (defined after XMXConfig)
KernelPath select_kernel_path(int batch_size, int64_t M, int64_t N, int64_t K, int quant_type, const XMXConfig & cfg);

/**
 * Get kernel path name as string for debug logging.
 *
 * @param path Kernel path enum value
 * @return Human-readable name string
 */
inline const char * kernel_path_name(KernelPath path) {
    switch (path) {
        case KernelPath::DMMV:
            return "DMMV";
        case KernelPath::MMVQ:
            return "MMVQ";
        case KernelPath::ESIMD_DPAS:
            return "ESIMD_DPAS";
        case KernelPath::ESIMD_LARGE_TILE:
            return "ESIMD_LARGE_TILE";
        default:
            return "UNKNOWN";
    }
}

// =============================================================================
// XMXConfig: Hardware-queried Configuration for ESIMD dpas
// =============================================================================
// This struct captures hardware-specific XMX dimensions and capabilities.
// Use XMXConfig::from_device(device_id) to query actual hardware values.
// Default constructor provides safe fallback values for Intel Arc GPUs.

// Note: XMXConfig::from_device() is implemented in unified-kernel.cpp
// and requires ggml_sycl_device_info which is defined in common.hpp (global namespace).

/**
 * XMX configuration for ESIMD dpas kernels.
 *
 * Captures hardware-queried tile dimensions and capabilities.
 * Intel ESIMD dpas parameters:
 * - SystolicDepth: Always 8 (fixed in hardware)
 * - RepeatCount: 1-8, determines M dimension (we use 8 for full utilization)
 * - ExecutionSize: 8 for DG2, 16 for PVC/Arc (determines N dimension)
 */
struct XMXConfig {
    // =========================================================================
    // XMX Tile Dimensions
    // =========================================================================
    // These are hardware-defined constraints for dpas instructions.
    // Default values are for Intel Arc B580.

    size_t xmx_m      = 8;   // RepeatCount determines M (1-8, we use 8)
    size_t xmx_n      = 16;  // ExecutionSize: 8 for DG2, 16 for PVC/Arc
    size_t xmx_k_fp16 = 16;  // K for FP16: SystolicDepth(8) x OpsPerChannel(2) = 16
    size_t xmx_k_int8 = 32;  // K for INT8: SystolicDepth(8) x OpsPerChannel(4) = 32

    // =========================================================================
    // Hardware Resources
    // =========================================================================

    size_t slm_size              = 65536;  // SLM bytes per work-group (default 64KB)
    int    nsm                   = 20;     // Compute units (streaming multiprocessors)
    int    max_esimd_workgroup   = 64;     // Max work-group size for ESIMD kernels (Arc: 64, PVC: 1024)

    // =========================================================================
    // Capability Flags (from hardware query)
    // =========================================================================

    bool supported              = false;  // Hardware has XMX support
    bool supports_int8          = false;  // INT8 dpas available
    bool supports_fp16          = false;  // FP16 dpas available
    bool supports_named_barrier = false;  // Named barriers (PVC only)
    bool supports_esimd_dpas    = false;  // ESIMD xmx::dpas with ExecSize=16 (PVC, Xe2/Battlemage)

    // =========================================================================
    // Derived Configuration
    // =========================================================================

    bool use_double_buffer  = false;  // SLM can hold 2x tile buffers
    int  tiles_per_workitem = 1;      // Tiles processed per work-item

    // =========================================================================
    // Factory Method
    // =========================================================================

    /**
     * Query hardware and create configuration for a specific device.
     *
     * @param device_id  Device index (0-based), or -1 for default config
     * @return XMXConfig populated with hardware values, or defaults if unavailable
     *
     * Edge cases handled:
     * - device_id < 0: Returns default config
     * - device_id >= device_count: Returns default config
     * - xmx.M/N/K = 0: Uses fallback defaults
     * - xmx.slm_size = 0: Uses default 65536
     * - xmx.supported = false: Returns config with supported=false
     */
    static XMXConfig from_device(int device_id);
};

// Note: XMXConfig::from_device() is implemented in unified-kernel.cpp
// because it requires access to ggml_sycl_device_info which is defined in common.hpp.

// =============================================================================
// Kernel Path Selection Implementation (defined after XMXConfig)
// =============================================================================

/**
 * Select the optimal kernel path based on batch size and hardware capabilities.
 *
 * This is the primary dispatch function for batch-size gating. It considers:
 * 1. Batch size thresholds (DMMV for 1, MMVQ for small, ESIMD for large)
 * 2. Hardware capability (cfg.supported)
 * 3. Environment overrides (GGML_SYCL_FORCE_MMVQ, GGML_SYCL_FORCE_ESIMD)
 *
 * Decision tree:
 * - FORCE_MMVQ=1 -> MMVQ (environment override)
 * - FORCE_ESIMD=1 -> ESIMD_DPAS (environment override, if hardware supports XMX)
 * - batch_size == 1 -> DMMV unless ESIMD is explicitly enabled (GGML_SYCL_XMX_ESIMD=1)
 * - batch_size < min_batch -> MMVQ unless ESIMD is explicitly enabled
 * - !cfg.supported -> MMVQ (no XMX hardware)
 * - Otherwise -> ESIMD_DPAS (compute-bound, XMX beneficial)
 *
 * @param batch_size   Number of tokens (M dimension)
 * @param M            Output rows (same as batch_size for inference)
 * @param N            Output columns (hidden dim)
 * @param K            Reduction dimension
 * @param quant_type   Quantization type (GGML_TYPE_*)
 * @param cfg          XMX hardware configuration
 * @return Selected kernel path
 *
 * Performance: O(1) time, no memory allocation, simple integer comparisons only.
 */
inline KernelPath select_kernel_path(int               batch_size,
                                     int64_t           M,
                                     int64_t           N,
                                     int64_t           K,
                                     int               quant_type,
                                     const XMXConfig & cfg) {
    // Suppress unused parameter warnings (reserved for future use)
    (void) M;
    (void) N;
    (void) K;
    (void) quant_type;

    // Environment override: Force MMVQ path
    if (env_force_mmvq()) {
        return KernelPath::MMVQ;
    }

    // Environment override: Force ESIMD path (if hardware supports it)
    if (env_force_esimd()) {
        return cfg.supported ? KernelPath::ESIMD_DPAS : KernelPath::MMVQ;
    }

    // Check environment override first (cached at init)
    const int min_batch = get_esimd_min_batch();

    // Explicit ESIMD opt-in via GGML_SYCL_XMX_ESIMD=1 can bypass batch gating.
    const bool esimd_opt_in = use_esimd_dpas();

    // Batch=1: Always use DMMV path for warp-parallel reduction
    // ESIMD DPAS is designed for 8x16 output tiles, not single-row operations.
    // The unified kernel's internal DMMV implementation uses warp-parallel
    // reduction which is ~60x faster than tiled ESIMD for batch=1.
    // Set GGML_SYCL_FORCE_ESIMD_BATCH1=1 to force ESIMD for batch=1 (testing only).
    if (batch_size == 1) {
        static int force_esimd_batch1 = -1;
        if (force_esimd_batch1 < 0) {
            const char * env   = std::getenv("GGML_SYCL_FORCE_ESIMD_BATCH1");
            force_esimd_batch1 = (env && std::atoi(env) != 0) ? 1 : 0;
        }
        if (force_esimd_batch1 && cfg.supported) {
            return KernelPath::ESIMD_DPAS;
        }
        // Default: Always use DMMV for batch=1
        return KernelPath::DMMV;
    }

    // Batch < threshold: MMVQ unless ESIMD explicitly enabled (or prefer ESIMD for small batches)
    if (batch_size < min_batch) {
        if (prefer_esimd_small() && cfg.supported) {
            return KernelPath::ESIMD_DPAS;
        }
        return (esimd_opt_in && cfg.supported) ? KernelPath::ESIMD_DPAS : KernelPath::MMVQ;
    }

    // Check if ESIMD available
    if (!cfg.supported) {
        return KernelPath::MMVQ;
    }

    // Large-tile path for very large batches (32x32 output tiles with 64 work-items)
    // Uses cooperative loading with esimd::barrier() for work-group synchronization.
    // Requires:
    // - GGML_SYCL_LARGE_TILE_KERNEL_ENABLED=1 at compile time (enabled by default)
    // - GGML_SYCL_UNIFIED_LARGE_TILE=1 environment variable
#if GGML_SYCL_LARGE_TILE_KERNEL_ENABLED
    if (batch_size >= get_large_tile_min_batch() && can_use_large_tile_esimd(M, N, K)) {
        return KernelPath::ESIMD_LARGE_TILE;
    }
#endif

    // Default: ESIMD dpas for compute-bound regime
    return KernelPath::ESIMD_DPAS;
}

// =============================================================================
// Batch Strategy for XMX Path
// =============================================================================
// Different batch sizes benefit from different tiling strategies.

enum class BatchStrategy {
    WIDE_N,     // Batch 1-7: Wide N-tiles (single row, process multiple N columns per sub-group)
    STANDARD,   // Batch 8-63: Standard tiling with multiple M and N tiles
    PERSISTENT  // Batch 64+: Multiple tiles per workgroup with persistent threads
};

/**
 * Determine the optimal batch strategy for XMX path.
 *
 * @param batch_size  Number of tokens (M dimension)
 * @return Recommended batch strategy
 */
inline BatchStrategy get_batch_strategy(int batch_size) {
    if (batch_size <= 7) {
        return BatchStrategy::WIDE_N;
    } else if (batch_size <= 63) {
        return BatchStrategy::STANDARD;
    } else {
        return BatchStrategy::PERSISTENT;
    }
}

// =============================================================================
// Layout Mode Constants
// =============================================================================
// These mirror the reorder_mode enum from common.hpp
// 0 = NONE (AoS), 1 = SOA, 2 = COALESCED, 3 = XMX_COALESCED, 4 = XMX_GEMM_TILED

constexpr int LAYOUT_NONE           = 0;
constexpr int LAYOUT_SOA            = 1;
constexpr int LAYOUT_COALESCED      = 2;
constexpr int LAYOUT_XMX_COALESCED  = 3;
constexpr int LAYOUT_XMX_GEMM_TILED = 4;

// =============================================================================
// LayoutMode Enum
// =============================================================================
// Strongly-typed enum for layout selection in unified kernel.
// Values match LAYOUT_* constants above for interoperability.

enum class LayoutMode : int {
    AOS           = LAYOUT_NONE,          // Array of Structures (original contiguous blocks)
    SOA           = LAYOUT_SOA,           // Structure of Arrays (qs bytes first, then scales)
    COALESCED     = LAYOUT_COALESCED,     // Word-major interleaved for sub-group reads
    XMX_COALESCED = LAYOUT_XMX_COALESCED  // K_TILE=32 aligned for dpas instructions
};

// =============================================================================
// Quantization Type Constants
// =============================================================================
// These mirror GGML_TYPE_* enum values
// Used for runtime dispatch to appropriate dequantization code

constexpr int QUANT_TYPE_Q4_0 = 2;   // GGML_TYPE_Q4_0
constexpr int QUANT_TYPE_Q4_1 = 3;   // GGML_TYPE_Q4_1
constexpr int QUANT_TYPE_Q8_0 = 8;   // GGML_TYPE_Q8_0
constexpr int QUANT_TYPE_Q6_K = 14;  // GGML_TYPE_Q6_K
constexpr int QUANT_TYPE_F32  = 0;   // GGML_TYPE_F32
constexpr int QUANT_TYPE_F16  = 1;   // GGML_TYPE_F16

// Weight layout modes (mirror ggml_layout_mode values for AoS/SoA)
constexpr int WEIGHT_LAYOUT_AOS = 0;
constexpr int WEIGHT_LAYOUT_SOA = 1;

// =============================================================================
// Q4_0 Block Structure
// =============================================================================
// Q4_0: 32 weights per block, 4 bits per weight
// Block layout: [d: fp16] [qs: 16 bytes (32 nibbles)]
// Total size: 18 bytes per block

// Note: UNIFIED_QK4_0 may already be defined by ggml-common.h as a macro.
// Use namespaced constant to avoid conflicts.
constexpr int UNIFIED_QK4_0 = 32;  // Weights per Q4_0 block

struct block_q4_0_unified {
    sycl::half d;                      // Scale factor
    uint8_t    qs[UNIFIED_QK4_0 / 2];  // Quantized values: 16 bytes = 32 nibbles
};

static_assert(sizeof(block_q4_0_unified) == sizeof(sycl::half) + UNIFIED_QK4_0 / 2, "wrong q4_0 block size");
static_assert(sizeof(block_q4_0_unified) == 18,
              "block_q4_0_unified must be exactly 18 bytes for correct prefetch behavior");

// =============================================================================
// Q6_K Block Structure
// =============================================================================
// Q6_K: 256 weights per block, 6 bits per weight
// Block layout: [ql: 128 bytes] [qh: 64 bytes] [scales: 16 bytes] [d: fp16]
// Total size: 210 bytes per block

constexpr int UNIFIED_QK6_K = 256;  // Weights per Q6_K block

struct block_q6_K_unified {
    uint8_t    ql[UNIFIED_QK6_K / 2];   // Lower 4 bits
    uint8_t    qh[UNIFIED_QK6_K / 4];   // Upper 2 bits
    int8_t     scales[UNIFIED_QK6_K / 16];
    sycl::half d;                       // Super-block scale
};

static_assert(sizeof(block_q6_K_unified) ==
                  sizeof(sycl::half) + UNIFIED_QK6_K / 16 + 3 * UNIFIED_QK6_K / 4,
              "wrong q6_K block size");

// =============================================================================
// MXFP4 Block Structure
// =============================================================================
// MXFP4: 32 weights per block, 4 bits per weight with E8M0 shared exponent
// Block layout: [e: E8M0 exponent] [qs: 16 bytes (32 nibbles)]
// Total size: 17 bytes per block
//
// E8M0 exponent format: 8-bit unsigned integer representing 2^(e-127)
// E2M1 mantissa values in qs (4 bits each): encoded in kvalues_mxfp4 lookup table
// Values are doubled in lookup table, multiply by 0.5 during dequantization

constexpr int UNIFIED_QK_MXFP4 = 32;  // Weights per MXFP4 block
constexpr int QUANT_TYPE_MXFP4 = 39;  // GGML_TYPE_MXFP4

struct block_mxfp4_unified {
    uint8_t e;                          // E8M0 shared exponent
    uint8_t qs[UNIFIED_QK_MXFP4 / 2];   // Quantized values: 16 bytes = 32 nibbles (E2M1)
};

static_assert(sizeof(block_mxfp4_unified) == sizeof(uint8_t) + UNIFIED_QK_MXFP4 / 2, "wrong mxfp4 block size");
static_assert(sizeof(block_mxfp4_unified) == 17,
              "block_mxfp4_unified must be exactly 17 bytes for correct prefetch behavior");

// E2M1 quantization lookup table (values are doubled, multiply by 0.5 during dequant)
// Positive: 0, 1, 2, 3, 4, 6, 8, 12 (indices 0-7)
// Negative: 0, -1, -2, -3, -4, -6, -8, -12 (indices 8-15)
constexpr int8_t kvalues_mxfp4_unified[16] = {
    0, 1, 2, 3, 4, 6, 8, 12,     // Positive values (doubled)
    0, -1, -2, -3, -4, -6, -8, -12  // Negative values (doubled)
};

// =============================================================================
// UnifiedKernelArgs: Kernel launch parameters
// =============================================================================
// Contains all information needed to launch the unified matmul kernel.
// Designed to be POD for efficient device-side access.
//
// GGML Tensor Layout Convention:
// ==============================
// In GGML, mul_mat computes: dst[m,n] = sum_k(src0[n,k] * src1[m,k])
//
// This is equivalent to: C = B @ A^T where:
// - A = src0 (weights) with shape [N, K]
// - B = src1 (activations) with shape [M, K]
// - C = dst (output) with shape [M, N]
//
// Key insight: weights are indexed by output column (n), NOT output row (m)!
// Each weight row corresponds to an output column.

struct UnifiedKernelArgs {
    // Matrix dimensions (GGML convention)
    int64_t M;  // Output rows (batch * tokens) - from src1->ne[1]
    int64_t N;  // Output columns (hidden dim / output features) - from src0->ne[1]
    int64_t K;  // Reduction dimension (must be multiple of block size) - from src0->ne[0]

    // Tile configuration (from auto-tuning or heuristics)
    int tile_m;  // M dimension tile size (output rows per tile)
    int tile_n;  // N dimension tile size (output columns per tile)
    int tile_k;  // K dimension tile size (typically 32 for Q4_0)

    // Compute path selection
    bool use_xmx;  // true = XMX/dpas path, false = scalar path

    // Memory layout (legacy int field for compatibility)
    int layout_mode;  // 0=NONE(AoS), 1=SOA, 2=COALESCED, etc.

    // Memory layout (strongly-typed enum)
    LayoutMode layout = LayoutMode::AOS;  // Default: Array of Structures

    // Quantization format
    int quant_type;  // GGML_TYPE_* enum value

    // Prefetch configuration
    int prefetch_depth;  // 0 = none, 1-4 typical

    // Data pointers (device memory)
    // GGML Convention: weights indexed by (n, k), activations indexed by (m, k)
    const void *  weights;      // Quantized weight matrix [N, K/block_size blocks] - src0
    const float * activations;  // Activation matrix [M, K] (row-major F32) - src1
    float *       output;       // Output matrix [M, N] (row-major F32) - dst
};

// =============================================================================
// Kernel Launch Function
// =============================================================================

/**
 * Launch the unified matmul kernel.
 *
 * Computes: output[M,N] = activations[M,K] @ weights[N,K]^T
 *
 * GGML Convention: dst[m,n] = sum_k(src0[n,k] * src1[m,k])
 * - weights (src0) has shape [N, K] - indexed by output column n
 * - activations (src1) has shape [M, K] - indexed by output row m
 * - output (dst) has shape [M, N]
 *
 * The kernel automatically handles:
 * - Q4_0 dequantization during computation
 * - Boundary conditions for non-aligned dimensions
 * - SLM staging for weight reuse
 *
 * @param q     SYCL queue for submission
 * @param args  Kernel arguments (dimensions, tiles, data pointers)
 */
void launch_unified_matmul(sycl::queue & q, const UnifiedKernelArgs & args);

// =============================================================================
// Utility Functions
// =============================================================================

// Forward declarations for small-tile XMX gating helpers.
inline bool allow_small_xmx_tiles();

/**
 * Calculate required SLM size for unified kernel.
 *
 * @param tile_m  M tile size (output rows)
 * @param tile_n  N tile size (output columns)
 * @param tile_k  K tile size (reduction dimension)
 * @return Size in bytes needed for SLM
 */
inline size_t calculate_slm_size(int tile_m, int tile_n, int tile_k) {
    // SLM usage (GGML convention):
    // - Weight tile: tile_n * tile_k weights (dequantized to float)
    //   Weights are indexed by output column (n), so we load tile_n rows of weights
    // - Activation tile: tile_m * tile_k floats
    //   Activations are indexed by output row (m), so we load tile_m rows of activations
    // For Q4_0 with scalar path, we dequantize to float in SLM
    size_t weight_slm     = static_cast<size_t>(tile_n) * tile_k * sizeof(float);
    size_t activation_slm = static_cast<size_t>(tile_m) * tile_k * sizeof(float);
    return weight_slm + activation_slm;
}

/**
 * Check if dimensions are valid for unified kernel.
 *
 * @param args Kernel arguments
 * @return true if dimensions are valid
 */
inline bool validate_args(const UnifiedKernelArgs & args) {
    // K must be multiple of block size for Q4_0
    if (args.quant_type == QUANT_TYPE_Q4_0 && (args.K % UNIFIED_QK4_0) != 0) {
        return false;
    }

    // Dimensions must be positive
    if (args.M <= 0 || args.N <= 0 || args.K <= 0) {
        return false;
    }

    // Tile sizes must be positive
    if (args.tile_m <= 0 || args.tile_n <= 0 || args.tile_k <= 0) {
        return false;
    }

    // Pointers must be valid
    if (args.weights == nullptr || args.activations == nullptr || args.output == nullptr) {
        return false;
    }

    return true;
}

// =============================================================================
// Scalar Fallback Path Functions
// =============================================================================
// Used for non-XMX devices and partial tiles with explicit boundary checking

/**
 * Determine if scalar fallback should be used instead of XMX.
 *
 * Use scalar for:
 * - Very small M (< 8, too small for XMX)
 * - K not aligned to 32 (dpas requirement)
 * - Device without XMX support
 * - Partial tiles at boundaries
 *
 * @param args Kernel arguments
 * @return true if scalar fallback should be used
 */
inline bool should_use_scalar_fallback(const UnifiedKernelArgs & args) {
    // Use scalar for very small M unless small-tile XMX is explicitly enabled
    if (args.M < 8 && !allow_small_xmx_tiles()) {
        return true;
    }
    // Use scalar for K not aligned to 32 (dpas requirement)
    if (args.K % 32 != 0) {
        return true;
    }
    // Use scalar when XMX is explicitly disabled
    if (!args.use_xmx) {
        return true;
    }
    return false;
}

/**
 * Check if a specific tile requires scalar fallback due to boundary conditions.
 *
 * @param m_start   Starting M index for this tile
 * @param n_start   Starting N index for this tile
 * @param k_start   Starting K index for this tile
 * @param tile_m    Tile size in M dimension
 * @param tile_n    Tile size in N dimension
 * @param tile_k    Tile size in K dimension
 * @param M         Total M dimension
 * @param N         Total N dimension
 * @param K         Total K dimension
 * @return true if this tile is a partial tile requiring scalar fallback
 */
inline bool is_partial_tile(int64_t m_start,
                            int64_t n_start,
                            int64_t k_start,
                            int     tile_m,
                            int     tile_n,
                            int     tile_k,
                            int64_t M,
                            int64_t N,
                            int64_t K) {
    // Check if tile extends beyond matrix boundaries
    if (m_start + tile_m > M) {
        return true;
    }
    if (n_start + tile_n > N) {
        return true;
    }
    if (k_start + tile_k > K) {
        return true;
    }
    return false;
}

/**
 * Compute scalar tile with explicit boundary checking.
 *
 * This function handles partial tiles at matrix boundaries where
 * dimensions may not align with tile sizes. Each work-item processes
 * one or more output elements using a simple nested loop.
 *
 * GGML Convention: dst[m,n] = sum_k(src0[n,k] * src1[m,k])
 * - weights (src0) indexed by (n, k)
 * - activations (src1) indexed by (m, k)
 *
 * @tparam TILE_M   M tile size (output rows)
 * @tparam TILE_N   N tile size (output columns)
 * @tparam TILE_K   K tile size (reduction)
 * @param activations   Activation matrix (not used - data loaded to SLM)
 * @param slm_weights   Dequantized weights in SLM [TILE_N x TILE_K] indexed as [n * TILE_K + k]
 * @param slm_activations Activations in SLM [TILE_M x TILE_K] indexed as [m * TILE_K + k]
 * @param output        Output matrix [M x N] (row-major)
 * @param M_actual      Actual M elements in this tile (may be < TILE_M)
 * @param N_actual      Actual N elements in this tile (may be < TILE_N)
 * @param K_actual      Actual K elements in this tile (may be < TILE_K)
 * @param m_offset      Starting M index in global matrix
 * @param n_offset      Starting N index in global matrix
 * @param K             Full K dimension (not used - tile size TILE_K used for indexing)
 * @param N             Full N dimension (for output indexing)
 * @param item          ND-item for work distribution
 */
template <int TILE_M, int TILE_N, int TILE_K>
inline void compute_tile_scalar_bounded(const float * /* activations */,  // Not used - data loaded to SLM
                                        sycl::local_accessor<float, 1> & slm_weights,
                                        sycl::local_accessor<float, 1> & slm_activations,
                                        float *                          output,
                                        int                              M_actual,
                                        int                              N_actual,
                                        int                              K_actual,
                                        int64_t                          m_offset,
                                        int64_t                          n_offset,
                                        int64_t /* K */,  // Not used - tile size TILE_K used for indexing
                                        int64_t                  N,
                                        const sycl::nd_item<2> & item) {
    const int local_row      = item.get_local_id(0);
    const int local_col      = item.get_local_id(1);
    const int local_size_row = item.get_local_range(0);
    const int local_size_col = item.get_local_range(1);

    // Each work-item handles a subset of output elements
    // Iterate over output elements assigned to this thread
    for (int m = local_row; m < M_actual; m += local_size_row) {
        for (int n = local_col; n < N_actual; n += local_size_col) {
            float sum = 0.0f;

            // Dot product over K dimension using SLM data
            // GGML: dst[m,n] = sum_k(weights[n,k] * activations[m,k])
            for (int k = 0; k < K_actual; k++) {
                // slm_weights layout: [TILE_N x TILE_K] indexed as [n * TILE_K + k]
                // slm_activations layout: [TILE_M x TILE_K] indexed as [m * TILE_K + k]
                float w = slm_weights[n * TILE_K + k];
                float a = slm_activations[m * TILE_K + k];
                sum += w * a;
            }

            // Accumulate to output (atomically if needed, but with tile-based
            // partitioning each output element is owned by exactly one work-group)
            output[(m_offset + m) * N + (n_offset + n)] += sum;
        }
    }
}

/**
 * Compute scalar tile with sub-group optimization.
 *
 * Uses sub-group collective operations for efficient horizontal reduction.
 * Each sub-group processes rows cooperatively, with lanes distributing
 * the K-dimension work and reducing results.
 *
 * GGML Convention: dst[m,n] = sum_k(src0[n,k] * src1[m,k])
 * - weights (src0) indexed by (n, k)
 * - activations (src1) indexed by (m, k)
 *
 * @tparam TILE_M   M tile size (output rows)
 * @tparam TILE_N   N tile size (output columns)
 * @tparam TILE_K   K tile size (reduction)
 * @param activations   Activation matrix (not used, data in SLM)
 * @param slm_weights   Dequantized weights in SLM [TILE_N x TILE_K] indexed as [n * TILE_K + k]
 * @param slm_activations Activations in SLM [TILE_M x TILE_K] indexed as [m * TILE_K + k]
 * @param output        Output matrix [M x N] (row-major)
 * @param M_actual      Actual M elements in this tile
 * @param N_actual      Actual N elements in this tile
 * @param K_actual      Actual K elements in this tile
 * @param m_offset      Starting M index in global matrix
 * @param n_offset      Starting N index in global matrix
 * @param K             Full K dimension (not used)
 * @param N             Full N dimension (for output indexing)
 * @param sg            Sub-group handle
 * @param item          ND-item for work distribution
 */
template <int TILE_M, int TILE_N, int TILE_K>
inline void compute_tile_scalar_subgroup(const float * /* activations */,  // Not used - data loaded to SLM
                                         sycl::local_accessor<float, 1> & slm_weights,
                                         sycl::local_accessor<float, 1> & slm_activations,
                                         float *                          output,
                                         int                              M_actual,
                                         int                              N_actual,
                                         int                              K_actual,
                                         int64_t                          m_offset,
                                         int64_t                          n_offset,
                                         int64_t /* K */,  // Not used - tile size TILE_K used for indexing
                                         int64_t                  N,
                                         sycl::sub_group          sg,
                                         const sycl::nd_item<2> & item) {
    const int sg_id   = sg.get_local_id()[0];
    const int sg_size = sg.get_local_range()[0];

    // Work-group coordinates
    const int wg_row      = item.get_local_id(0);
    const int wg_size_row = item.get_local_range(0);

    // Each row of work-items handles different M values
    // Within a row, the sub-group cooperates on the K-reduction
    for (int m = wg_row; m < M_actual; m += wg_size_row) {
        for (int n = 0; n < N_actual; n++) {
            float partial = 0.0f;

            // Distribute K across sub-group lanes
            // GGML: dst[m,n] = sum_k(weights[n,k] * activations[m,k])
            for (int k = sg_id; k < K_actual; k += sg_size) {
                float w = slm_weights[n * TILE_K + k];
                float a = slm_activations[m * TILE_K + k];
                partial += w * a;
            }

            // Reduce within sub-group using collective operation
            float sum = sycl::reduce_over_group(sg, partial, sycl::plus<float>());

            // Lane 0 writes result
            if (sg_id == 0) {
                output[(m_offset + m) * N + (n_offset + n)] += sum;
            }
        }
    }
}

/**
 * Vectorized scalar compute for aligned tiles.
 *
 * Uses SYCL vec<float, 4> for better memory throughput when
 * dimensions are aligned to vector width.
 *
 * GGML Convention: dst[m,n] = sum_k(src0[n,k] * src1[m,k])
 * - weights (src0) indexed by (n, k)
 * - activations (src1) indexed by (m, k)
 *
 * @tparam TILE_M   M tile size (output rows)
 * @tparam TILE_N   N tile size (output columns, must be multiple of 4 for vectorization)
 * @tparam TILE_K   K tile size (reduction)
 * @param slm_weights      Dequantized weights in SLM [TILE_N x TILE_K] indexed as [n * TILE_K + k]
 * @param slm_activations  Activations in SLM [TILE_M x TILE_K] indexed as [m * TILE_K + k]
 * @param output           Output matrix [M x N]
 * @param M_actual         Actual M elements
 * @param N_actual         Actual N elements (should be multiple of 4)
 * @param K_actual         Actual K elements
 * @param m_offset         Starting M index
 * @param n_offset         Starting N index
 * @param N                Full N dimension (for output indexing)
 * @param item             ND-item for work distribution
 */
template <int TILE_M, int TILE_N, int TILE_K>
inline void compute_tile_scalar_vectorized(sycl::local_accessor<float, 1> & slm_weights,
                                           sycl::local_accessor<float, 1> & slm_activations,
                                           float *                          output,
                                           int                              M_actual,
                                           int                              N_actual,
                                           int                              K_actual,
                                           int64_t                          m_offset,
                                           int64_t                          n_offset,
                                           int64_t                          N,
                                           const sycl::nd_item<2> &         item) {
    const int local_row      = item.get_local_id(0);
    const int local_col      = item.get_local_id(1);
    const int local_size_row = item.get_local_range(0);
    const int local_size_col = item.get_local_range(1);

    // Process 4 K elements at a time when aligned (vectorize over K for better weight reuse)
    // Note: We iterate over output (m, n) and reduce over k
    for (int m = local_row; m < M_actual; m += local_size_row) {
        for (int n = local_col; n < N_actual; n += local_size_col) {
            float sum = 0.0f;

            // Vectorized K reduction when K_actual is multiple of 4
            const int K_vec = (K_actual / 4) * 4;

            for (int k = 0; k < K_vec; k += 4) {
                // Load 4 weight values for this n
                // GGML: weights[n,k] for consecutive k values
                sycl::vec<float, 4> w;
                w[0] = slm_weights[n * TILE_K + k + 0];
                w[1] = slm_weights[n * TILE_K + k + 1];
                w[2] = slm_weights[n * TILE_K + k + 2];
                w[3] = slm_weights[n * TILE_K + k + 3];

                // Load 4 activation values for this m
                // GGML: activations[m,k] for consecutive k values
                sycl::vec<float, 4> a;
                a[0] = slm_activations[m * TILE_K + k + 0];
                a[1] = slm_activations[m * TILE_K + k + 1];
                a[2] = slm_activations[m * TILE_K + k + 2];
                a[3] = slm_activations[m * TILE_K + k + 3];

                // Dot product contribution
                sum += w[0] * a[0] + w[1] * a[1] + w[2] * a[2] + w[3] * a[3];
            }

            // Scalar cleanup for remaining K elements
            for (int k = K_vec; k < K_actual; k++) {
                float w = slm_weights[n * TILE_K + k];
                float a = slm_activations[m * TILE_K + k];
                sum += w * a;
            }

            output[(m_offset + m) * N + (n_offset + n)] += sum;
        }
    }
}

// =============================================================================
// Layout-Aware Weight Loading Functions
// =============================================================================
// These functions load Q4_0 quantized weights from global memory into SLM,
// handling different memory layouts (AOS, SOA, COALESCED, XMX_COALESCED).
// All functions dequantize to sycl::half in SLM for XMX compatibility.

// COALESCED layout constants (matches dmmv.cpp)
constexpr int MMVQ_COALESCED_TILE_BLOCKS = 32;                                                // Blocks per tile
constexpr int MMVQ_COALESCED_TILE_BYTES  = MMVQ_COALESCED_TILE_BLOCKS * (UNIFIED_QK4_0 / 2);  // 512 bytes

// XMX constants for weight loading
constexpr int XMX_K_TILE_LOADING = 32;  // K dimension alignment for dpas

/**
 * Dequantize a Q4_0 block to half precision.
 *
 * @param block  Pointer to Q4_0 block
 * @param output Output array of UNIFIED_QK4_0 half values
 */
SYCL_EXTERNAL inline void dequant_q4_0_to_half(const block_q4_0_unified * block, sycl::half * output) {
    const sycl::half d = block->d;

#pragma unroll
    for (int i = 0; i < 16; i++) {
        const uint8_t qs = block->qs[i];
        const int     lo = (qs & 0x0F) - 8;
        const int     hi = (qs >> 4) - 8;

        output[i]      = static_cast<sycl::half>(lo) * d;
        output[i + 16] = static_cast<sycl::half>(hi) * d;
    }
}

/**
 * Convert E8M0 exponent to float scale factor.
 * E8M0 is an 8-bit unsigned exponent format: value = 2^(e - 127)
 * For MXFP4, we also multiply by 0.5 since kvalues are doubled.
 *
 * @param e E8M0 exponent byte
 * @return Float scale factor (already halved for MXFP4)
 */
SYCL_EXTERNAL inline float e8m0_to_float_half(uint8_t e) {
    uint32_t bits;
    if (e == 0) {
        // Denormal case: return 2^(-127) * 0.5 = 2^(-128)
        bits = 0x00400000;  // Small positive float
    } else {
        // Normal case: scale by 0.5 by subtracting 1 from exponent
        // 2^(e-127) * 0.5 = 2^(e-128)
        bits = (uint32_t)(e - 1) << 23;
    }
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

/**
 * Dequantize an MXFP4 block to half precision.
 *
 * MXFP4 format:
 * - E8M0 shared exponent (8-bit unsigned): scale = 2^(e - 127)
 * - E2M1 mantissa values (4-bit each): lookup in kvalues_mxfp4_unified
 * - Lookup table values are doubled, so we multiply scale by 0.5
 *
 * @param block  Pointer to MXFP4 block
 * @param output Output array of UNIFIED_QK_MXFP4 half values
 */
SYCL_EXTERNAL inline void dequant_mxfp4_to_half(const block_mxfp4_unified * block, sycl::half * output) {
    // Get scale factor (already halved via e8m0_to_float_half)
    const float scale = e8m0_to_float_half(block->e);

#pragma unroll
    for (int i = 0; i < 16; i++) {
        const uint8_t qs = block->qs[i];
        const int8_t  lo = kvalues_mxfp4_unified[qs & 0x0F];
        const int8_t  hi = kvalues_mxfp4_unified[qs >> 4];

        output[i]      = static_cast<sycl::half>(static_cast<float>(lo) * scale);
        output[i + 16] = static_cast<sycl::half>(static_cast<float>(hi) * scale);
    }
}

/**
 * Load weights from AOS (Array of Structures) layout.
 *
 * AOS layout: Contiguous blocks
 * - Q4_0: 18 bytes [d: fp16][qs: 16 bytes]
 * - MXFP4: 17 bytes [e: E8M0][qs: 16 bytes]
 * Indexed as: blocks[row * blocks_per_row + col]
 *
 * GGML Convention: weights[N, K] - indexed by output column n, then k
 * We load TILE_N rows of weights (one row per output column n)
 *
 * @tparam TILE_M       M dimension tile size (output rows, not used for weight loading)
 * @tparam TILE_N       N dimension tile size (output columns = weight rows to load)
 * @tparam TILE_K       K dimension tile size (should be multiple of 32)
 * @param slm           SLM accessor for dequantized weights [TILE_N * TILE_K]
 * @param weights       Global memory pointer to quantized blocks
 * @param n_start       Starting N (output column) index
 * @param k_start       Starting K index
 * @param N             Total N dimension (weight rows)
 * @param K             Total K dimension
 * @param quant_type    Quantization type (QUANT_TYPE_Q4_0 or QUANT_TYPE_MXFP4)
 * @param item          ND-item for work distribution
 */
template <int TILE_M, int TILE_N, int TILE_K>
inline void load_weights_aos(sycl::local_accessor<sycl::half, 1> & slm,
                             const void *                          weights,
                             int64_t                               n_start,
                             int64_t                               k_start,
                             int64_t                               N,
                             int64_t                               K,
                             int                                   quant_type,
                             const sycl::nd_item<3> &              item) {
    // Both Q4_0 and MXFP4 have 32 weights per block
    constexpr int QK        = 32;
    const int blocks_per_row = static_cast<int>(K / QK);

    const int local_id   = item.get_local_linear_id();
    const int local_size = item.get_local_range().size();

    // Total elements to load: TILE_N rows x TILE_K weights
    // Each thread handles a subset of blocks
    const int tile_k_blocks = TILE_K / QK;
    const int total_blocks  = TILE_N * tile_k_blocks;

    for (int idx = local_id; idx < total_blocks; idx += local_size) {
        const int     n_off          = idx / tile_k_blocks;
        const int     k_block        = idx % tile_k_blocks;
        const int64_t n_global       = n_start + n_off;
        const int64_t k_block_global = k_start / QK + k_block;

        // Bounds check
        if (n_global >= N || k_block_global >= blocks_per_row) {
            // Zero-fill for out-of-bounds
            for (int i = 0; i < QK; i++) {
                slm[n_off * TILE_K + k_block * QK + i] = sycl::half(0.0f);
            }
            continue;
        }

        // Load and dequantize block based on quant type
        sycl::half temp[QK];

        if (quant_type == QUANT_TYPE_MXFP4) {
            const block_mxfp4_unified * blocks = static_cast<const block_mxfp4_unified *>(weights);
            const block_mxfp4_unified * block  = &blocks[n_global * blocks_per_row + k_block_global];
            dequant_mxfp4_to_half(block, temp);
        } else {
            // Default: Q4_0
            const block_q4_0_unified * blocks = static_cast<const block_q4_0_unified *>(weights);
            const block_q4_0_unified * block  = &blocks[n_global * blocks_per_row + k_block_global];
            dequant_q4_0_to_half(block, temp);
        }

        // Store to SLM: [n_off * TILE_K + k]
        for (int i = 0; i < QK; i++) {
            slm[n_off * TILE_K + k_block * QK + i] = temp[i];
        }
    }
}

/**
 * Load weights from SOA (Structure of Arrays) layout.
 *
 * SOA layout: All qs bytes contiguous, then all scales contiguous.
 * qs: [nrows * K/2 bytes]
 * d:  [nrows * K/32 half values]
 *
 * GGML Convention: weights[N, K] - indexed by output column n, then k
 * We load TILE_N rows of weights (one row per output column n)
 *
 * @tparam TILE_M       M dimension tile size (output rows, not used for weight loading)
 * @tparam TILE_N       N dimension tile size (output columns = weight rows to load)
 * @tparam TILE_K       K dimension tile size (should be multiple of UNIFIED_QK4_0)
 * @param slm           SLM accessor for dequantized weights [TILE_N * TILE_K]
 * @param weights       Global memory pointer (SOA layout)
 * @param n_start       Starting N (output column) index
 * @param k_start       Starting K index
 * @param N             Total N dimension (weight rows)
 * @param K             Total K dimension
 * @param nrows_full    Full number of rows in tensor (for scale offset calculation)
 * @param item          ND-item for work distribution
 */
template <int TILE_M, int TILE_N, int TILE_K>
inline void load_weights_soa(sycl::local_accessor<sycl::half, 1> & slm,
                             const void *                          weights,
                             int64_t                               n_start,
                             int64_t                               k_start,
                             int64_t                               N,
                             int64_t                               K,
                             int64_t                               nrows_full,
                             const sycl::nd_item<3> &              item) {
    const uint8_t *    qs_base        = static_cast<const uint8_t *>(weights);
    const int          row_qs_bytes   = static_cast<int>(K / 2);
    const sycl::half * d_base         = reinterpret_cast<const sycl::half *>(qs_base + nrows_full * row_qs_bytes);
    const int          blocks_per_row = static_cast<int>(K / UNIFIED_QK4_0);

    const int local_id   = item.get_local_linear_id();
    const int local_size = item.get_local_range().size();

    // Each thread loads a subset of blocks
    const int tile_k_blocks = TILE_K / UNIFIED_QK4_0;
    const int total_blocks  = TILE_N * tile_k_blocks;

    for (int idx = local_id; idx < total_blocks; idx += local_size) {
        const int     n_off       = idx / tile_k_blocks;
        const int     k_block     = idx % tile_k_blocks;
        const int64_t n_global    = n_start + n_off;
        const int64_t k_start_idx = k_start + k_block * UNIFIED_QK4_0;

        // Bounds check
        if (n_global >= N || k_start_idx >= K) {
            for (int i = 0; i < UNIFIED_QK4_0; i++) {
                slm[n_off * TILE_K + k_block * UNIFIED_QK4_0 + i] = sycl::half(0.0f);
            }
            continue;
        }

        // Get scale for this block
        // GGML: weights[n_global, k] - row n_global
        const int64_t    block_idx = n_global * blocks_per_row + k_start_idx / UNIFIED_QK4_0;
        const sycl::half d         = d_base[block_idx];

        // Get qs pointer for this block
        const uint8_t * qs = qs_base + n_global * row_qs_bytes + k_start_idx / 2;

        // Dequantize and store to SLM: [n_off * TILE_K + k]
        for (int i = 0; i < 16; i++) {
            const uint8_t qs_byte = qs[i];
            const int     lo      = (qs_byte & 0x0F) - 8;
            const int     hi      = (qs_byte >> 4) - 8;

            slm[n_off * TILE_K + k_block * UNIFIED_QK4_0 + i]      = static_cast<sycl::half>(lo) * d;
            slm[n_off * TILE_K + k_block * UNIFIED_QK4_0 + i + 16] = static_cast<sycl::half>(hi) * d;
        }
    }
}

/**
 * Load weights from COALESCED layout.
 *
 * COALESCED layout: Word-major interleaved for efficient sub-group reads.
 * Tiles of MMVQ_COALESCED_TILE_BLOCKS (32) blocks, with 4-byte word interleaving.
 *
 * Layout within a tile:
 * - Word 0: bytes 0-3 from all 32 blocks (128 bytes)
 * - Word 1: bytes 4-7 from all 32 blocks (128 bytes)
 * - ...
 * - Word 3: bytes 12-15 from all 32 blocks (128 bytes)
 *
 * GGML Convention: weights[N, K] - indexed by output column n, then k
 * We load TILE_N rows of weights (one row per output column n)
 *
 * @tparam TILE_M       M dimension tile size (output rows, not used for weight loading)
 * @tparam TILE_N       N dimension tile size (output columns = weight rows to load)
 * @tparam TILE_K       K dimension tile size
 * @param slm           SLM accessor for dequantized weights [TILE_N * TILE_K]
 * @param weights       Global memory pointer (COALESCED layout)
 * @param n_start       Starting N (output column) index
 * @param k_start       Starting K index
 * @param N             Total N dimension (weight rows)
 * @param K             Total K dimension
 * @param nrows_full    Full number of rows in tensor
 * @param item          ND-item for work distribution
 */
template <int TILE_M, int TILE_N, int TILE_K>
inline void load_weights_coalesced(sycl::local_accessor<sycl::half, 1> & slm,
                                   const void *                          weights,
                                   int64_t                               n_start,
                                   int64_t                               k_start,
                                   int64_t                               N,
                                   int64_t                               K,
                                   int64_t                               nrows_full,
                                   const sycl::nd_item<3> &              item) {
    const uint8_t *    qs_base        = static_cast<const uint8_t *>(weights);
    const int          row_qs_bytes   = static_cast<int>(K / 2);
    const sycl::half * d_base         = reinterpret_cast<const sycl::half *>(qs_base + nrows_full * row_qs_bytes);
    const int          blocks_per_row = static_cast<int>(K / UNIFIED_QK4_0);

    const int local_id   = item.get_local_linear_id();
    const int local_size = item.get_local_range().size();

    constexpr int word_stride = MMVQ_COALESCED_TILE_BLOCKS * 4;  // 128 bytes

    // Each thread loads a subset of weights
    const int total_weights = TILE_N * TILE_K;

    for (int idx = local_id; idx < total_weights; idx += local_size) {
        const int     n_off    = idx / TILE_K;
        const int     k_off    = idx % TILE_K;
        const int64_t n_global = n_start + n_off;
        const int64_t k_global = k_start + k_off;

        // Bounds check
        if (n_global >= N || k_global >= K) {
            slm[idx] = sycl::half(0.0f);
            continue;
        }

        // Compute block and position within block
        const int block_idx    = static_cast<int>(k_global / UNIFIED_QK4_0);
        const int pos_in_block = static_cast<int>(k_global % UNIFIED_QK4_0);

        // Compute tile and position within tile
        const int tile_idx      = block_idx / MMVQ_COALESCED_TILE_BLOCKS;
        const int block_in_tile = block_idx % MMVQ_COALESCED_TILE_BLOCKS;

        // Compute byte position within the 16-byte qs region
        const int qs_byte_idx  = (pos_in_block < 16) ? pos_in_block : (pos_in_block - 16);
        const int word_idx     = qs_byte_idx / 4;
        const int byte_in_word = qs_byte_idx % 4;

        // Compute coalesced offset
        // GGML: weights[n_global, k] - row n_global
        const int64_t tile_base   = n_global * row_qs_bytes + tile_idx * MMVQ_COALESCED_TILE_BYTES;
        const int64_t word_offset = tile_base + word_idx * word_stride + block_in_tile * 4 + byte_in_word;

        // Load qs byte
        const uint8_t qs_byte = qs_base[word_offset];

        // Get scale
        const sycl::half d = d_base[n_global * blocks_per_row + block_idx];

        // Dequantize and store to SLM: [n_off * TILE_K + k_off]
        const int nibble = (pos_in_block < 16) ? (qs_byte & 0x0F) : (qs_byte >> 4);
        slm[idx]         = static_cast<sycl::half>(nibble - 8) * d;
    }
}

/**
 * Load weights from XMX_COALESCED layout.
 *
 * XMX_COALESCED layout: Optimized for dpas with K_TILE=32 alignment.
 * Similar to COALESCED but with additional padding/alignment for XMX.
 *
 * GGML Convention: weights[N, K] - indexed by output column n, then k
 * We load TILE_N rows of weights (one row per output column n)
 *
 * @tparam TILE_M       M dimension tile size (output rows, not used for weight loading)
 * @tparam TILE_N       N dimension tile size (output columns = weight rows to load)
 * @tparam TILE_K       K dimension tile size (should be 32 for dpas)
 * @param slm           SLM accessor for dequantized weights [TILE_N * TILE_K]
 * @param weights       Global memory pointer (XMX_COALESCED layout)
 * @param n_start       Starting N (output column) index
 * @param k_start       Starting K index
 * @param N             Total N dimension (weight rows)
 * @param K             Total K dimension
 * @param nrows_full    Full number of rows in tensor
 * @param item          ND-item for work distribution
 */
template <int TILE_M, int TILE_N, int TILE_K>
inline void load_weights_xmx_coalesced(sycl::local_accessor<sycl::half, 1> & slm,
                                       const void *                          weights,
                                       int64_t                               n_start,
                                       int64_t                               k_start,
                                       int64_t                               N,
                                       int64_t                               K,
                                       int64_t                               nrows_full,
                                       const sycl::nd_item<3> &              item) {
    // XMX_COALESCED uses the same basic structure as COALESCED
    // but is aligned to XMX_K_TILE_LOADING boundaries
    static_assert(TILE_K % XMX_K_TILE_LOADING == 0 || TILE_K < XMX_K_TILE_LOADING,
                  "TILE_K must be multiple of XMX_K_TILE_LOADING for XMX_COALESCED");

    const uint8_t *    qs_base        = static_cast<const uint8_t *>(weights);
    const int          row_qs_bytes   = static_cast<int>(K / 2);
    const sycl::half * d_base         = reinterpret_cast<const sycl::half *>(qs_base + nrows_full * row_qs_bytes);
    const int          blocks_per_row = static_cast<int>(K / UNIFIED_QK4_0);

    const int local_id   = item.get_local_linear_id();
    const int local_size = item.get_local_range().size();

    // Each thread handles a subset of N x TILE_K weights
    const int total_weights = TILE_N * TILE_K;

    for (int idx = local_id; idx < total_weights; idx += local_size) {
        const int     n_off    = idx / TILE_K;
        const int     k_off    = idx % TILE_K;
        const int64_t n_global = n_start + n_off;
        const int64_t k_global = k_start + k_off;

        // Bounds check
        if (n_global >= N || k_global >= K) {
            slm[idx] = sycl::half(0.0f);
            continue;
        }

        // Compute block and position
        const int block_idx    = static_cast<int>(k_global / UNIFIED_QK4_0);
        const int pos_in_block = static_cast<int>(k_global % UNIFIED_QK4_0);

        // Compute tile and position within tile (XMX tiles are K_TILE aligned)
        const int tile_idx      = block_idx / MMVQ_COALESCED_TILE_BLOCKS;
        const int block_in_tile = block_idx % MMVQ_COALESCED_TILE_BLOCKS;

        // Compute byte position
        const int qs_byte_idx  = (pos_in_block < 16) ? pos_in_block : (pos_in_block - 16);
        const int word_idx     = qs_byte_idx / 4;
        const int byte_in_word = qs_byte_idx % 4;

        constexpr int word_stride = MMVQ_COALESCED_TILE_BLOCKS * 4;

        // Compute coalesced offset
        // GGML: weights[n_global, k] - row n_global
        const int64_t tile_base   = n_global * row_qs_bytes + tile_idx * MMVQ_COALESCED_TILE_BYTES;
        const int64_t word_offset = tile_base + word_idx * word_stride + block_in_tile * 4 + byte_in_word;

        // Load qs byte
        const uint8_t qs_byte = qs_base[word_offset];

        // Get scale
        const sycl::half d = d_base[n_global * blocks_per_row + block_idx];

        // Dequantize and store to SLM: [n_off * TILE_K + k_off]
        const int nibble = (pos_in_block < 16) ? (qs_byte & 0x0F) : (qs_byte >> 4);
        slm[idx]         = static_cast<sycl::half>(nibble - 8) * d;
    }
}

/**
 * Layout dispatcher for weight loading (3D nd_item version).
 *
 * Dispatches to the appropriate weight loading function based on LayoutMode.
 * Supports Q4_0 and MXFP4 quantization types for AOS layout.
 * SOA/COALESCED layouts currently only support Q4_0.
 *
 * GGML Convention: weights[N, K] - indexed by output column n, then k
 * We load TILE_N rows of weights (one row per output column n)
 *
 * @tparam TILE_M       M dimension tile size (output rows, not used for weight loading)
 * @tparam TILE_N       N dimension tile size (output columns = weight rows to load)
 * @tparam TILE_K       K dimension tile size
 * @param slm           SLM accessor for dequantized weights [TILE_N * TILE_K]
 * @param weights       Global memory pointer to weights
 * @param layout        Memory layout mode
 * @param n_start       Starting N (output column) index
 * @param k_start       Starting K index
 * @param N             Total N dimension (weight rows)
 * @param K             Total K dimension
 * @param nrows_full    Full number of rows (for SOA/COALESCED scale offset)
 * @param quant_type    Quantization type (QUANT_TYPE_Q4_0 or QUANT_TYPE_MXFP4)
 * @param item          ND-item for work distribution
 */
template <int TILE_M, int TILE_N, int TILE_K>
inline void load_weights_to_slm(sycl::local_accessor<sycl::half, 1> & slm,
                                const void *                          weights,
                                LayoutMode                            layout,
                                int64_t                               n_start,
                                int64_t                               k_start,
                                int64_t                               N,
                                int64_t                               K,
                                int64_t                               nrows_full,
                                int                                   quant_type,
                                const sycl::nd_item<3> &              item) {
    switch (layout) {
        case LayoutMode::AOS:
            load_weights_aos<TILE_M, TILE_N, TILE_K>(slm, weights, n_start, k_start, N, K, quant_type, item);
            break;

        case LayoutMode::SOA:
            // SOA layout currently only supports Q4_0
            load_weights_soa<TILE_M, TILE_N, TILE_K>(slm, weights, n_start, k_start, N, K, nrows_full, item);
            break;

        case LayoutMode::COALESCED:
            // COALESCED layout currently only supports Q4_0
            load_weights_coalesced<TILE_M, TILE_N, TILE_K>(slm, weights, n_start, k_start, N, K, nrows_full, item);
            break;

        case LayoutMode::XMX_COALESCED:
            // XMX_COALESCED layout currently only supports Q4_0
            load_weights_xmx_coalesced<TILE_M, TILE_N, TILE_K>(slm, weights, n_start, k_start, N, K, nrows_full, item);
            break;
    }
}

/**
 * Layout dispatcher for weight loading (2D nd_item version).
 *
 * Uses flat linear indexing to distribute work across work-items.
 * This overload is used by the existing kernel implementation.
 * Supports Q4_0 and MXFP4 quantization types for AOS layout.
 *
 * GGML Convention: weights[N, K] - indexed by output column n, then k
 * We load TILE_N rows of weights (one row per output column n)
 *
 * @tparam TILE_M       M dimension tile size (output rows, not used for weight loading)
 * @tparam TILE_N       N dimension tile size (output columns = weight rows to load)
 * @tparam TILE_K       K dimension tile size
 * @param slm           SLM accessor for dequantized weights [TILE_N * TILE_K]
 * @param weights       Global memory pointer to weights
 * @param layout        Memory layout mode
 * @param n_start       Starting N (output column) index
 * @param k_start       Starting K index
 * @param N             Total N dimension (weight rows)
 * @param K             Total K dimension
 * @param nrows_full    Full number of rows (for SOA/COALESCED scale offset)
 * @param quant_type    Quantization type (QUANT_TYPE_Q4_0 or QUANT_TYPE_MXFP4)
 * @param item2d        2D ND-item for work distribution
 */
template <int TILE_M, int TILE_N, int TILE_K>
inline void load_weights_to_slm(sycl::local_accessor<sycl::half, 1> & slm,
                                const void *                          weights,
                                LayoutMode                            layout,
                                int64_t                               n_start,
                                int64_t                               k_start,
                                int64_t                               N,
                                int64_t                               K,
                                int64_t                               nrows_full,
                                int                                   quant_type,
                                const sycl::nd_item<2> &              item2d) {
    // Use flat linear indexing within the work-group
    const int local_id   = static_cast<int>(item2d.get_local_linear_id());
    const int local_size = static_cast<int>(item2d.get_local_range().size());

    // Both Q4_0 and MXFP4 have 32 weights per block
    constexpr int QK             = 32;
    const int     blocks_per_row = static_cast<int>(K / QK);

    // Handle based on layout
    if (layout == LayoutMode::AOS) {
        // AOS: Direct block access - supports both Q4_0 and MXFP4
        const int tile_k_blocks = TILE_K / QK;
        const int total_blocks  = TILE_N * tile_k_blocks;

        for (int idx = local_id; idx < total_blocks; idx += local_size) {
            const int     n_off          = idx / tile_k_blocks;
            const int     k_block        = idx % tile_k_blocks;
            const int64_t n_global       = n_start + n_off;
            const int64_t k_block_global = k_start / QK + k_block;

            if (n_global >= N || k_block_global >= blocks_per_row) {
                for (int i = 0; i < QK; i++) {
                    slm[n_off * TILE_K + k_block * QK + i] = sycl::half(0.0f);
                }
                continue;
            }

            // Load and dequantize block based on quant type
            sycl::half temp[QK];

            if (quant_type == QUANT_TYPE_MXFP4) {
                const block_mxfp4_unified * blocks = static_cast<const block_mxfp4_unified *>(weights);
                const block_mxfp4_unified * block  = &blocks[n_global * blocks_per_row + k_block_global];
                dequant_mxfp4_to_half(block, temp);
            } else {
                // Default: Q4_0
                const block_q4_0_unified * blocks = static_cast<const block_q4_0_unified *>(weights);
                const block_q4_0_unified * block  = &blocks[n_global * blocks_per_row + k_block_global];
                dequant_q4_0_to_half(block, temp);
            }

            // Store to SLM: [n_off * TILE_K + k]
            for (int i = 0; i < QK; i++) {
                slm[n_off * TILE_K + k_block * QK + i] = temp[i];
            }
        }
    } else {
        // SOA/COALESCED/XMX_COALESCED: Per-element loading (Q4_0 only for now)
        const uint8_t *    qs_base      = static_cast<const uint8_t *>(weights);
        const int          row_qs_bytes = static_cast<int>(K / 2);
        const sycl::half * d_base       = reinterpret_cast<const sycl::half *>(qs_base + nrows_full * row_qs_bytes);

        const int total_weights = TILE_N * TILE_K;

        for (int idx = local_id; idx < total_weights; idx += local_size) {
            const int     n_off    = idx / TILE_K;
            const int     k_off    = idx % TILE_K;
            const int64_t n_global = n_start + n_off;
            const int64_t k_global = k_start + k_off;

            if (n_global >= N || k_global >= K) {
                slm[idx] = sycl::half(0.0f);
                continue;
            }

            const int block_idx    = static_cast<int>(k_global / QK);
            const int pos_in_block = static_cast<int>(k_global % QK);

            if (layout == LayoutMode::SOA) {
                // SOA: qs bytes contiguous, then scales
                // GGML: weights[n_global, k] - row n_global
                const uint8_t    qs_byte = qs_base[n_global * row_qs_bytes + k_global / 2];
                const sycl::half d       = d_base[n_global * blocks_per_row + block_idx];
                const int        nibble  = (pos_in_block < 16) ? (qs_byte & 0x0F) : (qs_byte >> 4);
                slm[idx]                 = static_cast<sycl::half>(nibble - 8) * d;
            } else {
                // COALESCED / XMX_COALESCED: Word-interleaved
                // GGML: weights[n_global, k] - row n_global
                const int     tile_idx      = block_idx / MMVQ_COALESCED_TILE_BLOCKS;
                const int     block_in_tile = block_idx % MMVQ_COALESCED_TILE_BLOCKS;
                const int     qs_byte_idx   = (pos_in_block < 16) ? pos_in_block : (pos_in_block - 16);
                const int     word_idx      = qs_byte_idx / 4;
                const int     byte_in_word  = qs_byte_idx % 4;
                constexpr int word_stride   = MMVQ_COALESCED_TILE_BLOCKS * 4;

                const int64_t tile_base   = n_global * row_qs_bytes + tile_idx * MMVQ_COALESCED_TILE_BYTES;
                const int64_t word_offset = tile_base + word_idx * word_stride + block_in_tile * 4 + byte_in_word;

                const uint8_t    qs_byte = qs_base[word_offset];
                const sycl::half d       = d_base[n_global * blocks_per_row + block_idx];
                const int        nibble  = (pos_in_block < 16) ? (qs_byte & 0x0F) : (qs_byte >> 4);
                slm[idx]                 = static_cast<sycl::half>(nibble - 8) * d;
            }
        }
    }
}

// =============================================================================
// XMX Compute Path Functions
// =============================================================================
// Uses Intel joint_matrix API for dpas (Dot Product Accumulate Systolic)
// acceleration on XMX hardware.

#if GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE

// Namespace alias for brevity
namespace sycl_matrix = sycl::ext::oneapi::experimental::matrix;

/**
 * XMX tile compute using joint_matrix.
 *
 * GGML Convention: dst[m,n] = sum_k(src0[n,k] * src1[m,k])
 *
 * For XMX, we compute: C[M,N] = B[M,K] @ A[N,K]^T
 * Where:
 * - A (weights) in SLM: [TILE_N x TILE_K] row-major, indexed as [n * TILE_K + k]
 * - B (activations) in SLM: [TILE_M x TILE_K] row-major, indexed as [m * TILE_K + k]
 * - C (output): [TILE_M x TILE_N]
 *
 * For each output tile (m, n), we need weights[n, :] and activations[m, :]
 *
 * @tparam TILE_M   M tile size (must be multiple of XMX_TILE_M=8)
 * @tparam TILE_N   N tile size (must be multiple of XMX_TILE_N=16)
 * @tparam TILE_K   K tile size (must be multiple of XMX_TILE_K=8)
 * @param sg            Sub-group handle
 * @param weights_slm   Pointer to weights in SLM [TILE_N x TILE_K] half, indexed as [n * TILE_K + k]
 * @param act_slm       Pointer to activations in SLM [TILE_M x TILE_K] half, indexed as [m * TILE_K + k]
 * @param c_regs        Pointer to C accumulator in registers [TILE_M x TILE_N] float
 * @param slm_acc       SLM for accumulator extraction
 * @param item          ND-item for work distribution
 */
template <int TILE_M, int TILE_N, int TILE_K>
[[sycl::reqd_sub_group_size(XMX_SUBGROUP_SIZE)]] inline void compute_tile_xmx(
    sycl::sub_group                  sg,
    const sycl::half *               weights_slm,  // Weights in SLM [TILE_N x TILE_K] row-major
    const sycl::half *               act_slm,      // Activations in SLM [TILE_M x TILE_K] row-major
    float *                          c_regs,       // Accumulator in registers [TILE_M x TILE_N]
    sycl::local_accessor<float, 1> & slm_acc,      // SLM for accumulator extraction
    const sycl::nd_item<2> & /* item */)           // Unused, kept for API consistency
{
    // Number of XMX tiles needed to cover the full tile
    constexpr int NUM_TILES_M = TILE_M / XMX_TILE_M;
    constexpr int NUM_TILES_N = TILE_N / XMX_TILE_N;
    constexpr int NUM_K_STEPS = TILE_K / XMX_TILE_K;

    // Per-tile output size
    constexpr int XMX_OUTPUT_SIZE = XMX_TILE_M * XMX_TILE_N;

    const int lane = sg.get_local_linear_id();

    // Declare joint matrices
    // For C = B @ A^T, we load:
    // - mat_a from activations B[m, k] (row-major)
    // - mat_b from weights A[n, k]^T (col-major view of row-major data)
    sycl_matrix::joint_matrix<sycl::sub_group, sycl::half, sycl_matrix::use::a, XMX_TILE_M, XMX_TILE_K,
                              sycl_matrix::layout::row_major>
        mat_a;  // Activations
    sycl_matrix::joint_matrix<sycl::sub_group, sycl::half, sycl_matrix::use::b, XMX_TILE_K, XMX_TILE_N,
                              sycl_matrix::layout::col_major>
        mat_b;  // Weights transposed
    sycl_matrix::joint_matrix<sycl::sub_group, float, sycl_matrix::use::accumulator, XMX_TILE_M, XMX_TILE_N> acc;

    // Get raw pointer to SLM accumulator region for this sub-group
    float * acc_slm_ptr = const_cast<float *>(&slm_acc[0]);
    auto    acc_slm_cast =
        sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(acc_slm_ptr);

    // Process each XMX tile
    for (int tm = 0; tm < NUM_TILES_M; tm++) {
        const int m_base = tm * XMX_TILE_M;

        for (int tn = 0; tn < NUM_TILES_N; tn++) {
            const int n_base = tn * XMX_TILE_N;

            // Initialize accumulator
            sycl_matrix::joint_matrix_fill(sg, acc, 0.0f);

            // K-dimension loop
            for (int tk = 0; tk < NUM_K_STEPS; tk++) {
                const int k_base = tk * XMX_TILE_K;

                // Load activations tile (row-major: activations[m, k])
                // Index: act_slm[m_base * TILE_K + k_base]
                auto a_ptr =
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        const_cast<sycl::half *>(act_slm + m_base * TILE_K + k_base));
                sycl_matrix::joint_matrix_load(sg, mat_a, a_ptr, TILE_K);

                // Load weights tile (transposed view: weights[n, k] loaded as col-major)
                // Weights are stored row-major as [n * TILE_K + k]
                // For col-major load of transposed matrix, we read from weights[n_base, k_base]
                // The col-major load will read columns of weights (which are rows after transpose)
                auto b_ptr =
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        const_cast<sycl::half *>(weights_slm + n_base * TILE_K + k_base));
                sycl_matrix::joint_matrix_load(sg, mat_b, b_ptr, TILE_K);

                // Compute: acc += A * B (where B is transposed weights)
                sycl_matrix::joint_matrix_mad(sg, acc, mat_a, mat_b, acc);
            }

            // Store accumulator to SLM for extraction
            sycl_matrix::joint_matrix_store(sg, acc, acc_slm_cast, XMX_TILE_N, sycl_matrix::layout::row_major);

            // Sub-group barrier to ensure store is complete
            sycl::group_barrier(sg);

            // Extract and accumulate to output registers
            for (int i = lane; i < XMX_OUTPUT_SIZE; i += XMX_SUBGROUP_SIZE) {
                int row     = i / XMX_TILE_N;
                int col     = i % XMX_TILE_N;
                int out_idx = (m_base + row) * TILE_N + (n_base + col);
                c_regs[out_idx] += acc_slm_ptr[i];
            }

            sycl::group_barrier(sg);
        }
    }
}

/**
 * Check if XMX unified kernel path is enabled via environment.
 *
 * XMX unified path is ENABLED by default while optimizing. Set
 * GGML_SYCL_XMX_UNIFIED=0 to disable if needed.
 *
 * The XMX path correctness issues have been resolved.
 * TODO: Enable by default once kernel is optimized (see llama.cpp-gkvk).
 *
 * @return true if XMX unified path is enabled
 */
inline bool is_xmx_unified_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_SYCL_XMX_UNIFIED");
        if (!env) {
            enabled = 1;
        } else {
            enabled = (std::string(env) == "0") ? 0 : 1;
        }
    }
    return enabled != 0;
}

/**
 * Check if XMX is force-enabled via environment.
 *
 * GGML_SYCL_UNIFIED_FORCE_XMX=1 or GGML_SYCL_FORCE_ESIMD=1 forces XMX usage
 * in dispatch and should also bypass conservative size gating.
 */
inline bool is_xmx_force_enabled() {
    static int forced = -1;
    if (forced < 0) {
        const char * env_force   = std::getenv("GGML_SYCL_UNIFIED_FORCE_XMX");
        const char * env_esimd   = std::getenv("GGML_SYCL_FORCE_ESIMD");
        const bool   force_xmx   = (env_force && std::atoi(env_force) != 0);
        const bool   force_esimd = (env_esimd && std::atoi(env_esimd) != 0);
        forced                   = (force_xmx || force_esimd) ? 1 : 0;
    }
    return forced != 0;
}

/**
 * Allow XMX for small tiles (M < 8 or N < 16).
 *
 * Enabled by default to drive XMX usage during decode. Set
 * GGML_SYCL_XMX_ALLOW_SMALL_TILES=0 to restore strict M/N gating.
 */
inline bool allow_small_xmx_tiles() {
    static int allow = -1;
    if (allow < 0) {
        const char * env = std::getenv("GGML_SYCL_XMX_ALLOW_SMALL_TILES");
        allow            = env ? ((std::atoi(env) != 0) ? 1 : 0) : 1;
        if (!allow && is_xmx_force_enabled()) {
            allow = 1;
        }
    }
    return allow != 0;
}

/**
 * Check if XMX path can be used for given dimensions.
 *
 * XMX requires:
 * - XMX enabled (GGML_SYCL_XMX_UNIFIED=0 disables; default enabled while optimizing)
 * - M >= XMX_TILE_M (8) unless GGML_SYCL_XMX_ALLOW_SMALL_TILES=1
 * - N >= XMX_TILE_N (16) unless GGML_SYCL_XMX_ALLOW_SMALL_TILES=1
 * - K aligned to XMX_TILE_K (16) for dpas
 *
 * @param M  Output rows
 * @param N  Output columns
 * @param K  Reduction dimension
 * @return true if XMX can be used
 */
inline bool can_use_xmx(int64_t M, int64_t N, int64_t K) {
    // XMX path enabled by default; disable with GGML_SYCL_XMX_UNIFIED=0
    if (!is_xmx_unified_enabled()) {
        return false;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        return false;
    }
    if (!allow_small_xmx_tiles()) {
        if (M < XMX_TILE_M || N < XMX_TILE_N) {
            return false;
        }
    }
    return (K % XMX_TILE_K) == 0;
}

#else   // !GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE

/**
 * Fallback when joint_matrix is not available.
 *
 * Note: For production use without joint_matrix, consider implementing
 * an ESIMD-based dpas path using sycl::ext::intel::esimd::xmx::dpas.
 */
inline bool is_xmx_force_enabled() {
    return false;
}

inline bool allow_small_xmx_tiles() {
    return false;
}

inline bool can_use_xmx(int64_t /* M */, int64_t /* N */, int64_t /* K */) {
    return false;  // XMX not available
}

#endif  // GGML_SYCL_XMX_JOINT_MATRIX_AVAILABLE

}  // namespace ggml_sycl_unified

// =============================================================================
// UnifiedKernel Class
// =============================================================================
// High-level kernel orchestration class that wraps individual operation APIs
// and provides persistent execution support for fused multi-layer inference.

namespace ggml_sycl {

// Forward declaration: DeviceOperation is defined in unified-kernel.cpp (device-side struct).
// Needed for build_role_schedule() method signature.
struct DeviceOperation;

// Per-operation multi-device split metadata, passed to UnifiedKernel::set_split_config()
// to populate DeviceOperation cross-device fields during launch_persistent_kernel().
struct SplitOpMeta {
    int     op_idx;          // Matmul index for progress/merge addressing
    int     row_start;       // First output row this device computes
    int     row_count;       // Number of output rows
    int     merge_count;     // Floats to merge from other device (0 = none)
    void *  merge_src;       // Host-pinned buffer for secondary partial output
    void *  merge_dst;       // Device pointer where merged output goes (primary only)
    float * input_staging;   // Host-pinned activation staging buffer (both devices share)
    int     input_K;         // K dimension for activation staging (floats to copy)
};

// Kernel-level split configuration, set once and applied to all ops during launch.
struct KernelSplitConfig {
    int *  progress_counter; // Device-local (malloc_device): kernel writes via atomic_ref, host reads via D2H BCS
    int *  merge_complete;   // Device-local (malloc_device): host writes via H2D, kernel reads via atomic_ref
    int    device_idx;       // 0=primary, 1=secondary
    int    n_devices;        // Total GPU devices in split (0 or 1 = no split)
    std::vector<SplitOpMeta> op_meta;  // Per-matmul split metadata (indexed by plan op_idx)
};

class UnifiedKernel {
public:
    explicit UnifiedKernel(sycl::queue & queue);
    ~UnifiedKernel();

    UnifiedKernel(const UnifiedKernel &)            = delete;
    UnifiedKernel & operator=(const UnifiedKernel &) = delete;

    void configure(const ggml_sycl_unified::XMXConfig & xmx_config);

    // Single Operation API
    void matmul(const ggml_sycl_unified::UnifiedKernelArgs & args);
    void rms_norm(const RmsNormDescriptor & desc);
    void rope(const RopeDescriptor & desc);
    void silu_mul(const void * gate, const void * up, void * output, int dim);
    void softmax(const void * input, void * output, int n, int stride);

    // Persistent Execution API
    void begin_persistent(int n_layers, int batch_size, int hidden_dim,
                          int intermediate_dim, int n_heads, int n_kv_heads,
                          int head_dim, int quant_type);
    void add_rms_norm(int layer, const void * weights, const void * input, void * output,
                      float eps, int hidden_dim, int64_t output_bytes = 0);
    void add_matmul(int layer, const void * weights, const void * input,
                    void * output, MatmulType type, int M, int N, int K,
                    int quant_type, int weight_layout,
                    const int64_t * weight_nb = nullptr,
                    const int64_t * input_nb = nullptr,
                    const int64_t * output_nb = nullptr,
                    int weight_ne2 = 1, int weight_ne3 = 1,
                    int input_ne2 = 1, int input_ne3 = 1,
                    int64_t output_bytes = 0);
    void add_attention(int layer, const AttentionDescriptor & desc, int64_t output_bytes = 0);
    void add_silu_mul(int layer, const void * gate, const void * up, void * output, int64_t output_bytes = 0);
    void add_add(int layer, const void * src0, const void * src1, void * output, int n_elements, int64_t output_bytes = 0);
    void add_mul(int layer, const void * src0, const void * src1, void * output, int n_elements, int64_t output_bytes = 0);
    void add_get_rows(int layer, const void * src0, const void * indices, void * output,
                      int n_elements, int64_t ne00, int64_t ne10, int64_t ne11, int64_t ne12,
                      int64_t nb01, int64_t nb02, int64_t nb03,
                      int64_t s10, int64_t s11, int64_t s12,
                      int64_t s1, int64_t s2, int64_t s3, int src0_type);
    void add_set_rows(int layer, const void * src0, const void * indices,
                      void * dst, const SetRowsMeta & meta, int n_elements,
                      const void * debug_ptr = nullptr, int64_t output_bytes = -1);
    void add_strided_copy(int layer, const void * src, void * dst,
                          const StridedCopyMeta & meta, int n_elements,
                          int64_t output_bytes = -1);
    void add_softmax(int layer, const void * input, const void * mask,
                     const void * sinks, void * output, int n_rows, int n_cols,
                     int ne01, int ne02, int ne03, float scale, float max_bias,
                     int mask_type, int64_t mask_nb0, int64_t mask_nb1,
                     int64_t mask_nb2, int64_t mask_nb3, int mask_ne2, int mask_ne3,
                     int64_t output_bytes = 0);
    void add_rope(int layer, const RopeDescriptor & desc, int64_t output_bytes = 0);
    void set_persistent_debug_attn(float * debug_ptr, int layer, int debug_floats);
    void set_persistent_debug_rms(float * debug_ptr, int layer, int hidden_dim, int * flag);
    void set_persistent_debug_matmul(float * debug_ptr, int layer, MatmulType type, int out_dim, int * flag);
    void set_persistent_debug_hash(uint64_t * debug_ptr, int debug_bytes);
    void add_temp_device_alloc(void * ptr, size_t bytes);
    void add_temp_device_alloc_handle(const ggml_sycl::alloc_handle & handle);
    void execute_persistent();
    // Graph overhead benchmark: measures per-node SYCL graph replay latency.
    // Called once on first persistent token to determine if micro-graph approach
    // is viable (go/no-go decision gate for graph-of-micro-kernels experiment).
    void benchmark_graph_overhead();
    // Phased persistent execution: launches the kernel in segments, calling
    // on_matmul_complete(matmul_index, phase_op_count) after each matmul
    // boundary. This allows host-mediated multi-device sync without
    // requiring mid-kernel GPU-to-host signaling.
    // The callback receives: matmul sequential index (0-based) and the
    // number of operations in the completed phase.
    using phase_callback_t = std::function<void(int matmul_idx)>;
    void execute_persistent_phased(phase_callback_t on_matmul_complete);
    void cancel_persistent();

    // Plan caching: reuse operation sequence between TG tokens
    bool has_cached_plan() const;
    void begin_plan_update();
    void update_op_pointers(int op_idx, const void * input, void * output,
                            const void * aux = nullptr, const void * mask = nullptr);
    void update_op_attention(int op_idx, const void * q, const void * k_cache,
                             const void * v_cache, const void * mask,
                             void * output,
                             int64_t q_nb0, int64_t q_nb1, int64_t q_nb2, int64_t q_nb3,
                             int64_t k_nb0, int64_t k_nb1, int64_t k_nb2, int64_t k_nb3,
                             int64_t v_nb0, int64_t v_nb1, int64_t v_nb2, int64_t v_nb3,
                             int seq_len);
    void update_op_rope(int op_idx, void * q, void * k, void * rope_dst,
                        const float * cos_cache, const float * sin_cache, int position);
    void finish_plan_update();    // API bookend; no-op currently, validates plan readiness in future
    void invalidate_plan_cache();
    void * get_rows_stable_ptr(int get_rows_index, size_t bytes);
    int cached_op_count() const;
    OperationType plan_op_type(int op_idx) const;
    bool get_op_descriptor(int op_idx, OperationDescriptor & out) const;
    bool update_op_descriptor(int op_idx, const OperationDescriptor & desc);
    OperationDescriptor * get_op_descriptor_mut(int op_idx);  // Direct mutable access (no copy)

    // Update recipe: compact per-token refresh that skips the full ggml graph walk
    bool has_update_recipe() const { return update_recipe_valid_; }
    void set_update_recipe(std::vector<UpdateRecipeEntry> && recipe);
    const std::vector<UpdateRecipeEntry> & get_update_recipe() const { return update_recipe_; }
    void invalidate_update_recipe();

    // Multi-device split: set/get per-kernel split config that populates
    // DeviceOperation cross-device fields during launch_persistent_kernel().
    void set_split_config(const KernelSplitConfig & config);
    void get_split_config(KernelSplitConfig & out) const;

    // DAG scheduling for barrier-free persistent kernel
    void build_dag(const std::vector<std::vector<int>> & successors,
                   const std::vector<int> & in_degree);
    void build_phase_schedule(const std::vector<std::vector<int>> & successors,
                              const std::vector<int> & in_degree);
    void reset_dag_counters();
    bool has_dag() const { return dag_allocated_; }
    const DeviceDAGState & dag_state() const { return dag_state_; }
    bool has_phase_schedule() const { return phase_allocated_; }
    const DevicePhaseSchedule & phase_schedule() const { return phase_schedule_; }

    // Scratch pool: single contiguous device allocation for all persistent op outputs.
    // Replaces ggml compute-buffer output pointers with stable kernel-owned addresses.
    // Forward-pass remap rewires the plan's data-flow chain during full_build.
    void   build_scratch_pool();
    void * scratch_output(int op_idx) const;
    void   free_scratch_pool();

    // Debug accessor: get the current plan's operations for diagnostic inspection.
    const std::vector<OperationDescriptor> & get_plan_operations() const {
        static const std::vector<OperationDescriptor> empty;
        if (current_plan_) return current_plan_->operations;
        if (!cached_ops_.empty()) return cached_ops_;
        return empty;
    }

    // Mutable accessor for annotating plan operations during extraction.
    // Used by extract_persistent_plan to set input_source_op/aux_source_op.
    std::vector<OperationDescriptor> & get_plan_operations_mut() {
        static std::vector<OperationDescriptor> empty;
        if (current_plan_) return current_plan_->operations;
        return empty;
    }

    // Deferred copy-back: CPY nodes execute AFTER the persistent kernel, not during
    // plan extraction.  The source is identified by plan op index so that
    // build_scratch_pool() remap is handled transparently via scratch_outputs_.
    struct DeferredCopy {
        int    source_op_idx;  // plan op whose scratch output is the copy source (-1 = use src_ptr)
        void * src_ptr;        // fallback source pointer when source_op_idx < 0
        void * dst;            // destination pointer (ggml output buffer, not remapped)
        size_t bytes;          // number of bytes to copy
    };
    void add_deferred_copy(int source_op_idx, void * src_ptr, void * dst, size_t bytes);
    void clear_deferred_copies() { deferred_copies_.clear(); }
    void execute_deferred_copies();

    // Launch persistent kernel asynchronously (does NOT wait for completion).
    // Returns after submitting the kernel to the queue. The caller is
    // responsible for waiting on queue_ or using the kernel's device-local
    // progress_counter/merge_complete counters for host-mediated coordination.
    // Used by the host coordinator thread approach for multi-device persistent TG.
    void launch_persistent_kernel_async();

    // Finalize after async persistent kernel execution: cache the plan template,
    // free temp device allocations, and reset the current plan. Must be called
    // after the kernel has completed (queue_.wait() has returned).
    void finalize_persistent();

    bool            supports_persistent() const;
    bool            is_building_plan() const;
    PersistentStats get_last_stats() const;

private:
    sycl::queue &                    queue_;
    int                              device_id_ = -1;
    size_t                           runtime_tracked_bytes_ = 0;
    ggml_sycl_unified::XMXConfig     xmx_config_;
    bool                             xmx_configured_ = false;

    std::unique_ptr<PersistentPlan>  current_plan_;
    PersistentStats                  last_stats_;

    // Plan caching
    std::vector<OperationDescriptor> cached_ops_;
    PersistentPlan                   cached_plan_template_;
    std::vector<std::pair<void *, size_t>> cached_temp_device_allocs_;
    std::unordered_map<void *, ggml_sycl::alloc_handle> cached_temp_device_alloc_handles_;
    size_t                                               cached_temp_device_alloc_bytes_ = 0;
    bool                             plan_cache_valid_ = false;

    // Update recipe: compact per-token refresh (see UpdateRecipeEntry)
    std::vector<UpdateRecipeEntry>   update_recipe_;
    bool                             update_recipe_valid_ = false;

    // Device ops table pool (DeviceOperation * — defined in unified-kernel.cpp)
    void * d_ops_pool_      = nullptr;
    int    d_ops_pool_size_ = 0;

    // Batched sync counter (tile_counter + barrier_counter + barrier_sense)
    int *  sync_block_      = nullptr;

    // GET_ROWS stable copy pools (one per GET_ROWS node in graph).
    // Multiple GET_ROWS can appear in the graph (token embedding + intermediate lookups).
    // Each needs its own buffer to avoid data clobbering.
    struct GetRowsSlot {
        void * ptr  = nullptr;
        size_t size = 0;
    };
    std::vector<GetRowsSlot> get_rows_slots_;

    // Scratch pool for persistent kernel — eliminates ggml buffer aliasing
    void *              scratch_pool_      = nullptr;
    size_t              scratch_pool_size_ = 0;
    std::vector<void *> scratch_outputs_;    // per-op scratch pointers (nullptr = use ggml)

    // Final output copy-back: the ggml buffer destination for logits.
    // Set once during the first build_scratch_pool (full_build) when op.output
    // still points to the ggml tensor buffer. On subsequent fast-path tokens,
    // op.output is already remapped to scratch, so we reuse this cached pointer.
    void *              final_output_ggml_dst_ = nullptr;

    // Deferred CPY nodes — executed after persistent kernel, sources remapped by scratch pool
    std::vector<DeferredCopy> deferred_copies_;

    void *  persistent_buffers_[4] = {nullptr, nullptr, nullptr, nullptr};
    int *   tile_counter_          = nullptr;
    int *   barrier_counter_       = nullptr;
    int *   barrier_sense_         = nullptr;
    size_t  persistent_buffer_size_ = 0;

    // Multi-device split configuration (set by set_split_config, consumed by launch)
    KernelSplitConfig      split_config_;
    bool                   split_config_set_   = false;

    // DAG scheduling state
    DeviceDAGState         dag_state_          = {};
    bool                   dag_allocated_      = false;
    int                    dag_pool_n_ops_     = 0;
    int                    dag_pool_n_edges_   = 0;
    std::vector<int>       host_initial_ready_counter_;
    std::vector<int>       host_n_tiles_;

    // Phase-based scheduling state (replaces DAG scan with O(1) tile claiming)
    DevicePhaseSchedule    phase_schedule_     = {};
    bool                   phase_allocated_    = false;
    int                    phase_pool_n_ops_   = 0;
    int                    phase_pool_n_phases_ = 0;
    std::vector<DevicePhaseEntry> host_phase_entries_;
    std::vector<int>       host_phase_offset_;
    std::vector<int>       host_phase_tiles_;
    // Original (pre-fusion-remapping) phase data from build_phase_schedule.
    // launch_persistent_kernel remaps plan indices -> device indices each token,
    // modifying host_phase_entries/offset/tiles. Without these originals,
    // the second token would double-remap (device indices through plan_to_device),
    // producing wrong op indices and stale output.
    std::vector<DevicePhaseEntry> orig_phase_entries_;
    std::vector<int>              orig_phase_offset_;
    std::vector<int>              orig_phase_tiles_;

    // Two-tier light barrier state
    std::vector<int>       host_phase_type_;      // [n_phases] 0=HEAVY, 1=LIGHT
    int *                  light_flags_         = nullptr;  // [max_phases] device-alloc completion flags
    int                    light_flags_size_    = 0;        // Current allocation size in elements

    // Role-based WG specialization state
    DeviceRoleSchedule     role_schedule_       = {};
    bool                   role_allocated_       = false;
    int                    role_pool_n_elem_     = 0;
    int                    role_pool_n_matmul_   = 0;
    int                    role_pool_n_sync_     = 0;
    std::vector<RoleSegment>   host_elem_segments_;
    std::vector<RoleSegment>   host_matmul_segments_;
    std::vector<RoleSyncPoint> host_sync_points_;

    void copy_plan_shape(const PersistentPlan & src, PersistentPlan & dst);
    void allocate_persistent_buffers(int hidden_dim, int intermediate_dim);
    void free_persistent_buffers();
    bool persistent_use_split_barrier() const;
    bool persistent_dispatch_uses_dag() const;
    int persistent_matmul_tile_cols(OperationType type, int N, int K) const;
    int persistent_num_workgroups(int total_tiles, bool has_attention, bool has_ffn, bool use_split_barrier) const;
    void launch_persistent_kernel(bool build_only = false);
    void build_role_schedule(const std::vector<DeviceOperation> & host_ops);

    // Micro-graph: SYCL command graph with one node per phase, replacing software barriers
    void launch_micro_graph_kernel();
    void record_micro_graph();      // Record + finalize command graph from phase schedule
    void invalidate_micro_graph();  // Mark graph as needing re-record

    // Micro-graph state
    // Forward-declared opaque type to avoid exposing SYCL graph extension headers in the .hpp
    struct MicroGraphState;
    std::unique_ptr<MicroGraphState> micro_graph_;
    int *  micro_tile_counters_     = nullptr;  // [n_phases] per-phase tile counters (device alloc)
    int    micro_tile_counters_n_   = 0;        // Allocated count
    bool   micro_graph_valid_       = false;    // True when recorded graph matches current plan
};

}  // namespace ggml_sycl

#endif  // GGML_SYCL_UNIFIED_KERNEL_HPP
