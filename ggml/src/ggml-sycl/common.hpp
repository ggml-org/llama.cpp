//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_COMMON_HPP
#define GGML_SYCL_COMMON_HPP

#include "dpct/helper.hpp"
#include "ggml-sycl.h"
#include "kv-offload.hpp"
#include "layer-streaming.hpp"
#include "presets.hpp"
#include "sycl_hw.hpp"
#include "tensor-types.hpp"
#include "unified-cache.hpp"
#include "orchestrator.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <sycl/sycl.hpp>
#include <thread>
#include <unordered_map>
#include <vector>

struct ggml_backend_sycl_context;

namespace ggml_sycl {
class L2PrefetchManager;  // Forward declaration for l2-prefetch.hpp
class UnifiedKernel;       // Forward declaration for unified-kernel.hpp

// Custom deleters - defined in ggml-sycl.cpp where types are complete
struct L2PrefetchManagerDeleter {
    void operator()(L2PrefetchManager * ptr) const;
};
struct UnifiedKernelDeleter {
    void operator()(UnifiedKernel * ptr) const;
};
}

struct ggml_sycl_fa_graph_snapshot {
    const void * q_ptr                  = nullptr;
    const void * k_ptr                  = nullptr;
    const void * v_ptr                  = nullptr;
    const void * mask_ptr               = nullptr;
    const void * sinks_ptr              = nullptr;
    const void * dst_ptr                = nullptr;
    const void * block_table            = nullptr;
    const void * seq_lens               = nullptr;
    int64_t      q_ne[GGML_MAX_DIMS]    = { 0 };
    int64_t      k_ne[GGML_MAX_DIMS]    = { 0 };
    int64_t      mask_ne[GGML_MAX_DIMS] = { 0 };
    int32_t      use_paged_layout       = 0;
    int32_t      use_paged_attn         = 0;
    int32_t      block_size             = 0;
    int32_t      max_blocks_per_seq     = 0;
};

bool ggml_sycl_cpu_fallback_graph(ggml_backend_sycl_context & ctx, ggml_tensor * dst, const char * reason);
struct ggml_sycl_device_info;
const ggml_sycl_device_info & ggml_sycl_info();

#if GGML_SYCL_DNNL
#    include "dnnl.hpp"
#    include "dnnl_sycl.hpp"
#endif

// Helper macro for deprecated get_pointer() -> get_multi_ptr() migration
// SYCL 2020 deprecates local_accessor::get_pointer() in favor of get_multi_ptr()
#define SYCL_LOCAL_ACC_PTR(acc) ((acc).template get_multi_ptr<sycl::access::decorated::no>().get())

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
/* suppress warning spam */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnested-anon-types"
#include "ggml-common.h"
#pragma clang diagnostic pop
#include "ggml-impl.h"

// MoE token mapping is two int32s (original_idx + expert_idx).
// Keep the size here to avoid a common.hpp <-> moe-sort.hpp include cycle.
constexpr size_t kMoETokenMappingBytes = sizeof(int32_t) * 2;

void * ggml_sycl_host_malloc(size_t size);
void   ggml_sycl_host_free(void * ptr);

// Get shared-context queue for TP mode (returns nullptr if not in TP mode)
sycl::queue * ggml_sycl_get_tp_queue(int device);

// Get shared context for TP mode (returns nullptr if not in TP mode)
sycl::context * ggml_sycl_get_tp_context();

// TP staging cache: stages mmap'd data to USM memory for shared-context access
// Per-device staging: each device gets its own device-local copy
void * ggml_sycl_get_staged_ptr_device(const void * src, size_t size, int device);
void * ggml_sycl_get_staged_ptr(const void * src, size_t size);  // Legacy: returns device 0's pointer
void   ggml_sycl_clear_staging_cache();

// Internal getters for seq_ids host pointers (set by llama layer, used by fattn)
const int32_t * ggml_sycl_get_seq_ids_host_q(size_t * count);
const int32_t * ggml_sycl_get_seq_ids_host_kv(size_t * count);

enum class ggml_sycl_layout_ptr_event : uint8_t {
    HOST_CACHE_TARGET_HIT,
    HOST_CACHE_AOS_HIT,
    HOST_CACHE_LAYOUT_FALLBACK,
    HOST_CACHE_DATA_FALLBACK,
    HOST_CACHE_MISS,
};

void ggml_sycl_layout_ptr_stat(ggml_sycl_layout_ptr_event event);
void ggml_sycl_layout_ptr_stats_dump();

extern int               g_ggml_sycl_debug;
extern int               g_ggml_sycl_debug_sync;
extern int               g_ggml_sycl_tp_debug;  // Tensor Parallelism debug output
extern int               g_ggml_sycl_prioritize_dmmv;
extern std::atomic<bool> g_ggml_sycl_debug_forced_off;

// Track when SYCL graph recording is active
extern thread_local bool g_ggml_sycl_graph_recording;
extern std::atomic<int>  g_ggml_sycl_graph_recording_depth;
extern std::atomic<int>  g_sycl_submit_count_during_recording;  // DIAG: operation dispatches during recording
extern std::atomic<int>  g_sycl_extra_submit_count_during_recording;  // DIAG: extra markers/events during recording
int                      ggml_sycl_graph_inflight_count();

inline bool ggml_sycl_graph_recording_active() {
    return g_ggml_sycl_graph_recording || g_ggml_sycl_graph_recording_depth.load(std::memory_order_acquire) > 0;
}

// Helper to check if we should add an event dependency.
// During graph recording, we MUST always add depends_on() to capture the dependency edge.
// Event status queries are unreliable during recording (events haven't executed yet).
// Outside graph recording, we only add the dependency if the event is not complete.
inline bool ggml_sycl_should_add_dependency(const sycl::event & dep_event) {
    if (ggml_sycl_graph_recording_active()) {
        // During graph recording, avoid depending on already-complete events (e.g., default/ready events),
        // but preserve dependencies for in-flight events to capture correct graph edges.
        try {
            auto status = dep_event.get_info<sycl::info::event::command_execution_status>();
            if (status == sycl::info::event_command_status::complete) {
                return false;
            }
        } catch (...) {
            // If status query fails (e.g., event not yet available), keep the dependency.
        }
        return true;
    }
    // Outside graph recording, check if event is already complete
    return dep_event.get_info<sycl::info::event::command_execution_status>() !=
           sycl::info::event_command_status::complete;
}

// Submit a lightweight marker event without ext_oneapi_submit_barrier on in-order queues.
// This avoids Level Zero barrier event corruption seen on some drivers.
template <typename MarkerKernel>
inline sycl::event ggml_sycl_submit_marker(sycl::queue & q,
                                           const std::vector<sycl::event> & deps = {}) {
    if (g_ggml_sycl_graph_recording) {
        g_sycl_extra_submit_count_during_recording.fetch_add(1, std::memory_order_relaxed);
    }
    if (q.has_property<sycl::property::queue::in_order>()) {
        return q.submit([&](sycl::handler & cgh) {
            if (!deps.empty()) {
                cgh.depends_on(deps);
            }
            cgh.single_task<MarkerKernel>([] {});
        });
    }
    return deps.empty() ? q.ext_oneapi_submit_barrier() : q.ext_oneapi_submit_barrier(deps);
}

// Tiered cache state for memory placement optimization
extern std::atomic<bool> g_tiered_enabled;

// Get cached tensor pointer for tiered dispatch
// Returns nullptr if not in tiered mode or tensor not cached
void * get_cached_tensor_ptr(const char * tensor_name, ggml_sycl::memory_tier * tier_out, bool * found_in_inventory);

// Resolve cached tensor pointer and sync extra state with tiered cache location.
void * ggml_sycl_get_cached_tensor_ptr_for(const ggml_tensor *      tensor,
                                           int                      device,
                                           ggml_sycl::memory_tier * tier_out,
                                           bool *                   found_in_inventory,
                                           sycl::usm::alloc *       alloc_out);

#if defined(__clang__) && __has_builtin(__builtin_expect)
// Hint the optimizer to pipeline the more likely following instruction in branches
#    define LIKELY(expr)   __builtin_expect(expr, true)
#    define UNLIKELY(expr) __builtin_expect(expr, false)
#else
#    define LIKELY(expr)   (expr)
#    define UNLIKELY(expr) (expr)
#endif

#define GGML_SYCL_DEBUG(...)                                                                              \
    do {                                                                                                  \
        if (UNLIKELY(!g_ggml_sycl_debug_forced_off.load(std::memory_order_relaxed) && g_ggml_sycl_debug)) \
            fprintf(stderr, __VA_ARGS__);                                                                 \
    } while (0)

// Tensor Parallelism debug output - controlled by GGML_SYCL_TP_DEBUG env var
#define GGML_SYCL_TP_DEBUG(...)             \
    do {                                    \
        if (UNLIKELY(g_ggml_sycl_tp_debug)) \
            fprintf(stderr, __VA_ARGS__);   \
    } while (0)

// Kernel trace - compile-time toggle for tracing kernel execution flow
// Enable by uncommenting the define below or adding -DGGML_SYCL_KERNEL_TRACE=1
// #define GGML_SYCL_KERNEL_TRACE 1

#ifdef GGML_SYCL_KERNEL_TRACE
#    define GGML_SYCL_KTRACE(kernel_name, ...)           \
        do {                                             \
            fprintf(stderr, "[KTRACE] %s", kernel_name); \
            fprintf(stderr, __VA_ARGS__);                \
            fprintf(stderr, "\n");                       \
            fflush(stderr);                              \
        } while (0)
#else
#    define GGML_SYCL_KTRACE(kernel_name, ...) ((void) 0)
#endif

#define CHECK_TRY_ERROR(expr)                                                                           \
    [&]() {                                                                                             \
        try {                                                                                           \
            expr;                                                                                       \
            return dpct::success;                                                                       \
        } catch (std::exception const & e) {                                                            \
            std::cerr << e.what() << "\nException caught at file:" << __FILE__ << ", line:" << __LINE__ \
                      << ", func:" << __func__ << std::endl;                                            \
            return dpct::default_error;                                                                 \
        }                                                                                               \
    }()

#define __SYCL_ARCH__ DPCT_COMPATIBILITY_TEMP
#define VER_4VEC      610                 // todo for hardward optimize.
#define VER_GEN9      700                 // todo for hardward optimize.
#define VER_GEN12     1000000             // todo for hardward optimize.
#define VER_GEN13     (VER_GEN12 + 1030)  // todo for hardward optimize.
#define VER_XE2       2000

#define GGML_SYCL_MAX_NODES 8192  // TODO: adapt to hardwares

// define for XMX in Intel GPU
// TODO: currently, it's not used for XMX really.
#if !defined(GGML_SYCL_FORCE_MMQ)
#    define SYCL_USE_XMX
#endif

// max batch size to use MMQ kernels when tensor cores are available
// MMQ ESIMD is optimal for small batches. Dequantize path is 2-3x faster for large batches.
#define MMQ_MAX_BATCH_SIZE 2048

// dmmv = dequantize_mul_mat_vec
#ifndef GGML_SYCL_DMMV_X
#    define GGML_SYCL_DMMV_X 32
#endif
#ifndef GGML_SYCL_MMV_Y
#    define GGML_SYCL_MMV_Y 1
#endif

typedef sycl::queue * queue_ptr;

enum ggml_sycl_backend_gpu_mode { SYCL_UNSET_GPU_MODE = -1, SYCL_SINGLE_GPU_MODE = 0, SYCL_MUL_GPU_MODE };

static_assert(sizeof(sycl::half) == sizeof(ggml_fp16_t), "wrong fp16 size");

// SYCL-compatible E8M0 to FP32 conversion (halved for MXFP4)
// E8M0 is an 8-bit exponent-only format used in MX (Microscaling) formats
static __dpct_inline__ float sycl_e8m0_to_fp32_half(uint8_t e) {
    // For e < 2: use precomputed denormal patterns
    // For e >= 2: exponent - 1 gives FP32 exponent (halving = divide by 2)
    uint32_t bits;
    if (e < 2) {
        // Denormal handling: e=0 -> 0.0, e=1 -> very small denormal
        static const uint32_t denorm_table[2] = { 0x00000000, 0x33800000 };
        bits                                  = denorm_table[e];
    } else {
        // Normal case: FP32 exponent = e - 1 (bias 127, so -1 gives halving)
        bits = ((uint32_t) (e - 1)) << 23;
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void crash() {
    int * ptr = NULL;
    *ptr      = 0;
}

[[noreturn]] static void ggml_sycl_error(const char * stmt,
                                         const char * func,
                                         const char * file,
                                         const int    line,
                                         const char * msg) {
    fprintf(stderr, "SYCL error: %s: %s\n", stmt, msg);
    fprintf(stderr, "  in function %s at %s:%d\n", func, file, line);
    GGML_ABORT("SYCL error");
}

#define SYCL_CHECK(err)                                                                                    \
    do {                                                                                                   \
        auto err_ = (err);                                                                                 \
        if (err_ != 0)                                                                                     \
            ggml_sycl_error(#err, __func__, __FILE__, __LINE__, "Exception caught in this line of code."); \
    } while (0)

#if DPCT_COMPAT_RT_VERSION >= 11100
#    define GGML_SYCL_ASSUME(x) __builtin_assume(x)
#else
#    define GGML_SYCL_ASSUME(x)
#endif  // DPCT_COMPAT_RT_VERSION >= 11100

#ifdef GGML_SYCL_F16
typedef sycl::half  dfloat;  // dequantize float
typedef sycl::half2 dfloat2;
#else
typedef float        dfloat;  // dequantize float
typedef sycl::float2 dfloat2;
#endif  // GGML_SYCL_F16

#define MMVQ_MAX_BATCH_SIZE 8

// Multi-row MMVQ kernel configuration
// Processes multiple output rows per work-group, sharing Y-vector in SLM
// This amortizes Y-vector loading across rows, reducing memory bandwidth
#define MMVQ_NROWS_PER_WG 4  // Rows per work-group (tune: 4, 8, or 16)

// SLM sizes for Y-vector caching in multi-row MMVQ
// Q8_1 block: 32 bytes quants (int8[32]) + 4 bytes ds (half2) = 36 bytes
// For Mistral 7B: ncols=4096, blocks_per_row = 4096/32 = 128 blocks
// SLM needed: 128 * 36 = 4.5KB per Y-vector (fits easily in 128KB SLM)
// We store qs as ints for aligned access: 8 ints per block (32 bytes)
// Plus ds as half2: 4 bytes per block
// Add +1 padding to avoid bank conflicts on 32-bank SLM
constexpr int MMVQ_SLM_Y_QS_STRIDE = 9;    // 8 ints + 1 padding to avoid bank conflicts
constexpr int MMVQ_SLM_MAX_BLOCKS  = 256;  // Max blocks per row (ncols=8192, qk=32)
constexpr int MMVQ_SLM_Y_QS_SIZE   = MMVQ_SLM_MAX_BLOCKS * MMVQ_SLM_Y_QS_STRIDE;  // ~9KB ints
constexpr int MMVQ_SLM_Y_DS_SIZE   = MMVQ_SLM_MAX_BLOCKS + 1;                     // half2 array + padding

// Warp-coalesced MMVQ configuration
// Reorganizes weight data so consecutive threads load consecutive bytes
// This achieves 100% cache line utilization (vs 50% with strided access)
constexpr int MMVQ_COALESCED_TILE_BLOCKS      = WARP_SIZE;  // Blocks per warp tile (match WARP_SIZE for 1 thread/block)
constexpr int MMVQ_COALESCED_TILE_BYTES_Q4_0  = MMVQ_COALESCED_TILE_BLOCKS * 16;  // Q4_0: 16 bytes/block
constexpr int MMVQ_COALESCED_TILE_BYTES_Q8_0  = MMVQ_COALESCED_TILE_BLOCKS * 32;  // Q8_0: 32 bytes/block
constexpr int MMVQ_COALESCED_TILE_BYTES_MXFP4 = MMVQ_COALESCED_TILE_BLOCKS * 16;  // MXFP4: 16 bytes/block
// Legacy alias for Q4_0
constexpr int MMVQ_COALESCED_TILE_BYTES       = MMVQ_COALESCED_TILE_BYTES_Q4_0;

// Variable tile decomposition helpers (power-of-2, largest first, max 32)
// Used for Q6_K coalesced layout to support arbitrary block counts
inline int tile_count(int blocks) {
    int count = 0;
    while (blocks > 0) {
        int tile_size = 1;
        while (tile_size * 2 <= blocks && tile_size < 32) {
            tile_size *= 2;
        }
        count++;
        blocks -= tile_size;
    }
    return count;
}

inline int tile_size_at(int blocks, int tile_idx) {
    int idx = 0;
    while (blocks > 0) {
        int tile_size = 1;
        while (tile_size * 2 <= blocks && tile_size < 32) {
            tile_size *= 2;
        }
        if (idx == tile_idx) {
            return tile_size;
        }
        blocks -= tile_size;
        idx++;
    }
    return 0;
}

inline int tile_offset_at(int blocks, int tile_idx) {
    int idx = 0, offset = 0;
    while (blocks > 0 && idx < tile_idx) {
        int tile_size = 1;
        while (tile_size * 2 <= blocks && tile_size < 32) {
            tile_size *= 2;
        }
        offset += tile_size;
        blocks -= tile_size;
        idx++;
    }
    return offset;
}

static int  g_all_sycl_device_count                     = -1;
static bool g_ggml_backend_sycl_buffer_type_initialized = false;

static ggml_sycl_backend_gpu_mode g_ggml_sycl_backend_gpu_mode = SYCL_UNSET_GPU_MODE;

static void * g_scratch_buffer = nullptr;
static size_t g_scratch_size   = 0;  // disabled by default
static size_t g_scratch_offset = 0;

[[noreturn]] static inline void bad_arch(const sycl::stream & stream_ct1) {
    stream_ct1 << "ERROR: ggml-sycl was compiled without support for the "
                  "current GPU architecture.\n";
    // __trap();
    std::exit(1);

    (void) bad_arch;  // suppress unused function warning
}

int get_current_device_id();
int ggml_sycl_map_device_id(int device);
void ggml_sycl_set_device_map(const int * device_ids, int device_count);

inline dpct::device_ext & ggml_sycl_get_device(int device) {
    return dpct::dev_mgr::instance().get_device(ggml_sycl_map_device_id(device));
}

inline int ggml_sycl_get_device_id_from_queue(sycl::queue & queue) {
    try {
        sycl::device dev          = queue.get_device();
        int          device_count = dpct::dev_mgr::instance().device_count();
        for (int i = 0; i < device_count; i++) {
            if (ggml_sycl_get_device(i) == dev) {
                return i;
            }
        }
    } catch (...) {
    }
    return dpct::dev_mgr::instance().current_device_id();
}

inline dpct::err0 ggml_sycl_set_device(const int device) try {
    int current_device_id;
    SYCL_CHECK(CHECK_TRY_ERROR(current_device_id = get_current_device_id()));

    // GGML_SYCL_DEBUG("ggml_sycl_set_device device_id=%d,
    // current_device_id=%d\n", device, current_device);
    const int mapped_device = ggml_sycl_map_device_id(device);
    if (mapped_device == current_device_id) {
        return 0;
    }

    return CHECK_TRY_ERROR(dpct::select_device(mapped_device));
} catch (const sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    crash();
    std::exit(1);
}

//////////////////////
enum class reorder_mode : uint8_t {
    NONE           = 0,  // Original AoS layout (Array of Structures)
    SOA            = 1,  // SoA layout: all qs bytes contiguous, then all d values
    COALESCED      = 2,  // Tile-based layout for MMVQ (word-major interleaved, requires SOA first)
    XMX_COALESCED  = 3,  // XMX-optimized layout for MoE GEMM (K_TILE=32 aligned rows)
    XMX_GEMM_TILED = 4,  // XMX GEMM tiled layout for quant weights
};

// =============================================================================
// Unified Tensor Layout System (replaces scattered layout tracking)
// =============================================================================

using layout_mode = ggml_layout_mode;

enum class tensor_usage : uint8_t {
    UNKNOWN = 0,
    ATTENTION_WEIGHT,   // Q, K, V, O projections
    FFN_WEIGHT,         // feed-forward non-MoE
    MOE_EXPERT_WEIGHT,  // MoE expert gate/up/down
    MOE_GATE,           // MoE routing gate
    MOE_INTERMEDIATE,   // MoE intermediate tensors (probs, indices, etc.)
    EMBEDDING,          // token embeddings
    NORM,               // RMS/LayerNorm weights
};

using tensor_layout_info = ggml_tensor_layout;

static inline void ggml_sycl_release_layout(tensor_layout_info & layout, sycl::queue & q) {
    if (layout.owns_memory && layout.data_ptr) {
        sycl::free(layout.data_ptr, q);
        layout.data_ptr    = nullptr;
        layout.owns_memory = false;
    }
}

// XMX hardware capabilities queried at runtime
// Moved here so layout_policy can reference ggml_sycl_info()
struct XMXCapabilities {
    bool supported = false;

    // Tile dimensions (queried from hardware)
    size_t M = 0;  // Expected: 8
    size_t N = 0;  // Expected: 16
    size_t K = 0;  // Expected: 32

    // Supported types
    bool supports_int8 = false;
    bool supports_fp16 = false;

    // Device memory info
    size_t slm_size = 0;  // Shared local memory per work-group

    // Derived optimal config
    int optimal_tiles_m = 1;
    int optimal_tiles_n = 1;
};

XMXCapabilities query_xmx_capabilities(sycl::device & dev);

struct sycl_device_info {
    int             cc;   // compute capability
    int             nsm;  // number of streaming multiprocessors (CUDA) maps to the maximum
                          // number of compute units on a SYCL device.
    // size_t  smpb;               // max. shared memory per block
    size_t          smpbo;                // max. shared memory per block (with opt-in)
    bool            vmm;                  // virtual memory support
    size_t          total_vram;
    size_t          max_alloc_size;       // device-reported max allocation size
    size_t          safe_max_alloc_size;  // probed safe allocation size
    //sycl_hw_info hw_info;     \\ device id and aarch, currently not used
    bool            supports_soa_reorder = false;  // Device capability: can use SoA weight layout
    XMXCapabilities xmx_caps;                      // XMX matrix engine capabilities (queried at init)
    char            device_name[256] = { 0 };      // Device name for GPU family detection
};

struct ggml_sycl_device_info {
    int device_count    = 0;  // GPUs visible to scheduler (may be reduced to 1)
    int total_gpu_count = 0;  // Total physical GPUs (before scheduler hiding)

    sycl_device_info devices[GGML_SYCL_MAX_DEVICES] = {};

    std::array<float, GGML_SYCL_MAX_DEVICES> default_tensor_split = {};

    int max_work_group_sizes[GGML_SYCL_MAX_DEVICES] = { 0 };

    // Host pinned memory limit (probed at init, driver has per-allocation limit)
    size_t host_max_alloc_size = 0;

    // CPU device for data-local compute (host-tier weight layers)
    bool          has_cpu_device = false;
    sycl::queue * cpu_queue      = nullptr;  // OpenCL CPU queue (owned, allocated with new)
};

const ggml_sycl_device_info & ggml_sycl_info();
size_t                        ggml_sycl_get_safe_max_alloc_size(int device);
size_t                        ggml_sycl_get_host_max_alloc_size();

// CPU offload: route host-resident tensor compute to a CPU SYCL device.
// Off by default; set GGML_SYCL_CPU_OFFLOAD=1 to enable.
inline bool ggml_sycl_cpu_offload_enabled() {
    static bool enabled = [] {
        const char * env = std::getenv("GGML_SYCL_CPU_OFFLOAD");
        return env != nullptr && std::atoi(env) != 0;
    }();
    return enabled;
}

inline bool ggml_sycl_cpu_offload_async_enabled() {
    static bool enabled = [] {
        const char * env = std::getenv("GGML_SYCL_CPU_OFFLOAD_ASYNC");
        if (!env) {
            return true;
        }
        return std::atoi(env) != 0;
    }();
    return enabled;
}

inline bool ggml_sycl_host_task_stable_for_queue(const sycl::queue & q) {
    const auto & info = ggml_sycl_info();
    if (!info.cpu_queue) {
        return true;
    }
    try {
        const sycl::backend queue_backend = q.get_backend();
        const sycl::backend cpu_backend   = info.cpu_queue->get_backend();
        // Mixed L0 (GPU) + OpenCL (CPU) can crash in host_task scheduler cleanup.
        if (queue_backend == sycl::backend::ext_oneapi_level_zero &&
            cpu_backend == sycl::backend::opencl) {
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

// CPU offload: query whether CPU SYCL device is available for data-local compute
bool          ggml_sycl_cpu_offload_available();
sycl::queue * ggml_sycl_get_cpu_queue();

struct layout_policy {
    static layout_mode get_optimal(ggml_type qtype, tensor_usage usage, int device_id = -1) {
        static const bool xmx_moe_enabled   = (std::getenv("GGML_SYCL_XMX_MOE") != nullptr);
        static const bool xmx_tiled_enabled = (xmx_moe_enabled && std::getenv("GGML_SYCL_XMX_MOE_TILED") != nullptr);

        bool xmx_supported = true;
        if (device_id >= 0 && device_id < ggml_sycl_info().device_count) {
            const auto & caps = ggml_sycl_info().devices[device_id].xmx_caps;
            xmx_supported     = caps.supported && caps.supports_int8;
        }

        // MoE experts: pick the layout required by the preferred kernel.
        if (usage == tensor_usage::MOE_EXPERT_WEIGHT) {
            if (xmx_supported) {
                if (xmx_tiled_enabled && qtype == GGML_TYPE_MXFP4) {
                    return GGML_LAYOUT_XMX_TILED;
                }
                if (xmx_moe_enabled && qtype == GGML_TYPE_Q8_0) {
                    return GGML_LAYOUT_SOA;
                }
            }
            if (qtype == GGML_TYPE_MXFP4) {
                return GGML_LAYOUT_COALESCED;
            }
            // MMVQ _id kernels for Q4_0/Q8_0 are AoS-only.
            if (qtype == GGML_TYPE_Q4_0 || qtype == GGML_TYPE_Q8_0) {
                return GGML_LAYOUT_AOS;
            }
        }

        // Q4_K kernels are AoS-only today (MMQ), so keep AoS canonical.
        if (qtype == GGML_TYPE_Q4_K) {
            return GGML_LAYOUT_AOS;
        }

        // Attention/FFN weights: SOA for better TG (batch>1) performance
        if (usage == tensor_usage::ATTENTION_WEIGHT || usage == tensor_usage::FFN_WEIGHT) {
            if (qtype == GGML_TYPE_Q4_0 || qtype == GGML_TYPE_Q8_0 || qtype == GGML_TYPE_Q6_K) {
                return GGML_LAYOUT_SOA;  // SOA for better TG performance
            }
        }

        // Embedding weights: prefer AoS for Q6_K to avoid known SoA kernel hangs on host-backed weights.
        if (usage == tensor_usage::EMBEDDING && qtype == GGML_TYPE_Q6_K) {
            return GGML_LAYOUT_AOS;
        }

        if (!ggml_is_quantized(qtype)) {
            return GGML_LAYOUT_AOS;
        }

        // Default: SOA is safe for all quantized types
        return GGML_LAYOUT_SOA;
    }

    static layout_mode get_with_override(ggml_type qtype, tensor_usage usage, int device_id = -1) {
        // Unified kernel requires AoS layout for supported types (Q4_0 today).
        // It performs reordering internally, so pre-reordering here would
        // double-transform the weights and corrupt results.
        // However, GGML_SYCL_UNIFIED_SOA=1 allows pre-reordering for DMMV SoA path.
        static int unified_dispatch_enabled = -1;
        static int unified_soa_enabled = -1;
        static int persistent_tg_soa_enabled = -1;
        if (unified_dispatch_enabled < 0) {
            const char * env = std::getenv("GGML_SYCL_UNIFIED_DISPATCH");
            // Default unified dispatch to ON unless explicitly disabled.
            unified_dispatch_enabled = (env == nullptr || std::atoi(env) != 0) ? 1 : 0;
        }
        if (unified_soa_enabled < 0) {
            const char * env = std::getenv("GGML_SYCL_UNIFIED_SOA");
            unified_soa_enabled = (env && std::atoi(env) == 0) ? 0 : 1;  // Default ON
        }
        if (persistent_tg_soa_enabled < 0) {
            const char * persistent_tg = std::getenv("GGML_SYCL_PERSISTENT_TG");
            // Default ON — set =0 to disable
            const bool persistent_on = (persistent_tg == nullptr || std::atoi(persistent_tg) != 0);
            const char * prefer_soa = std::getenv("GGML_SYCL_PERSISTENT_TG_PREFER_SOA");
            const bool prefer_on = (prefer_soa == nullptr || std::atoi(prefer_soa) != 0);
            persistent_tg_soa_enabled = (persistent_on && prefer_on) ? 1 : 0;
        }
        // When GGML_SYCL_UNIFIED_SOA=1, allow SoA layout for Q4_0 to enable
        // the DMMV SoA kernel path which has better memory bandwidth.
        // Persistent TG can also opt into SoA without requiring global UNIFIED_SOA.
        if (unified_dispatch_enabled != 0 && qtype == GGML_TYPE_Q4_0 &&
            !unified_soa_enabled && !persistent_tg_soa_enabled) {
            return GGML_LAYOUT_AOS;
        }
        return get_optimal(qtype, usage, device_id);
    }
};

inline tensor_usage infer_tensor_usage(const char * name) {
    if (!name) {
        return tensor_usage::UNKNOWN;
    }

    // MoE expert weights (highest priority - check first)
    if (strstr(name, "ffn_gate_exps") || strstr(name, "ffn_up_exps") || strstr(name, "ffn_down_exps")) {
        return tensor_usage::MOE_EXPERT_WEIGHT;
    }

    // MoE routing gate
    if (strstr(name, "ffn_gate_inp")) {
        return tensor_usage::MOE_GATE;
    }

    // MoE intermediate tensors (probs, indices, expert selection)
    if (strstr(name, "ffn_moe_probs") || strstr(name, "ffn_moe_") || strstr(name, "expert_ids") ||
        strstr(name, "expert_weights")) {
        return tensor_usage::MOE_INTERMEDIATE;
    }

    // Attention weights
    if (strstr(name, "attn_q") || strstr(name, "attn_k") || strstr(name, "attn_v") || strstr(name, "attn_output") ||
        strstr(name, "attn_sinks")) {
        return tensor_usage::ATTENTION_WEIGHT;
    }

    // FFN weights (non-MoE)
    if (strstr(name, "ffn_gate.") || strstr(name, "ffn_up.") || strstr(name, "ffn_down.")) {
        return tensor_usage::FFN_WEIGHT;
    }

        // Embeddings
        if (strstr(name, "token_embd") || strstr(name, "output.weight")) {
            return tensor_usage::EMBEDDING;
        }

    // Norms
    if (strstr(name, "_norm")) {
        return tensor_usage::NORM;
    }

    return tensor_usage::UNKNOWN;
}

// Resolve usage from registered metadata (if available), else fall back to name inference.
tensor_usage ggml_sycl_get_tensor_usage(const ggml_tensor * tensor);
layout_mode  ggml_sycl_adjust_layout_for_tensor(const ggml_tensor * tensor, layout_mode target, int device);
layout_mode  ggml_sycl_select_moe_mmvq_layout(const ggml_tensor * src0, int device, bool host_weights);
bool         ggml_sycl_get_layout_choice_for_tensor(const ggml_tensor * tensor, int device, layout_mode * out);
void *       ggml_sycl_get_weight_layout_ptr(const ggml_tensor * tensor, int device, layout_mode target);
void *       ggml_sycl_get_weight_layout_ptr(const ggml_tensor * tensor,
                                             int                 device,
                                             layout_mode         target,
                                             bool                prefer_host);
void ggml_sycl_enforce_layout_choice(const ggml_tensor * tensor, int device, layout_mode target, const char * context);
bool ggml_sycl_update_moe_ptr_table(ggml_backend_sycl_context &  ctx,
                                    const ggml_tensor *          src0,
                                    const ggml_tensor *          ids,
                                    layout_mode                  layout,
                                    sycl::event *                out_event,
                                    bool                         allow_all_experts = false,
                                    const std::vector<int32_t> * ids_host_override = nullptr,
                                    bool                         skip_device_copy  = false,
                                    bool                         force_cache_aos   = false);
bool ggml_sycl_moe_prepare_compact_list(ggml_backend_sycl_context & ctx,
                                        const ggml_tensor *         src0,
                                        int64_t                     total_batches,
                                        bool                        allow_alloc);
const int32_t * ggml_sycl_get_moe_ids_device_ptr(ggml_backend_sycl_context & ctx,
                                                 const ggml_tensor *         ids,
                                                 sycl::event *               out_event,
                                                 int64_t *                   out_nb0,
                                                 int64_t *                   out_nb1);

// Check if weight reordering is enabled.
bool ggml_sycl_reorder_enabled();

// Check if a tensor type supports coalesced memory layout conversion
// Add new types here as coalesced kernels are implemented
inline bool is_coalesced_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return true;
        case GGML_TYPE_Q6_K:
            return true;
        case GGML_TYPE_Q8_0:
            return true;
        case GGML_TYPE_MXFP4:
            return true;
        default:
            return false;
    }
}

// =============================================================================
// UNIFIED REORDER API - The ONLY way to change tensor reorder state
// =============================================================================
// These functions atomically perform BOTH:
//   1. Data transformation (AoS→SoA, SoA→COALESCED)
//   2. Flag update (reorder_mode state)
//
// CRITICAL: Never expose set_reorder_mode_() or any direct flag setter!
// The flag MUST always reflect actual data layout. Any desync causes garbage output.
//
// reorder_tensor_to_soa:      Defined in ggml-sycl.cpp (SOA transform code there)
// convert_tensor_to_coalesced: Defined in mmvq.cpp (COALESCED transform code there)
//
// For MoE cached experts (data already transformed by unified cache layout fills):
//   Use the explicit constructor: optimize_feature(reorder_mode::SOA)
//   This creates a new instance with the mode set - no mutation after creation.
// =============================================================================
struct ggml_tensor;
struct optimize_feature;
bool reorder_tensor_to_soa(const ggml_tensor * tensor, dpct::queue_ptr stream, const char * caller);
bool convert_tensor_to_coalesced(const ggml_tensor * tensor, dpct::queue_ptr stream, const char * caller);

// Reorder a raw device buffer from AOS to SOA layout for a given quantized type.
// Operates on partial row ranges (ncols x nrows). The device buffer must already
// contain AOS data and will be transformed in-place.
// Used by unified cache for partial-row loading in multi-device tensor split.
bool reorder_rows_to_soa(uint8_t * data_device, ggml_type type, int64_t ncols, int64_t nrows,
                          size_t size, dpct::queue_ptr stream);
bool ggml_sycl_is_optimize_feature_live(const optimize_feature * feature);
void ggml_sycl_register_optimize_feature(optimize_feature * feature);
void ggml_sycl_unregister_optimize_feature(optimize_feature * feature);

struct optimize_feature {
    // ==========================================================================
    // INVARIANT: reorder_ flag MUST always match actual data layout.
    // NEVER add a public setter for reorder_! The friend functions below are
    // the ONLY authorized way to change state because they ensure the data
    // transformation happens atomically with the flag update.
    //
    // For pre-transformed cached data (MoE experts), use the explicit constructor
    // to create a new instance with the correct mode already set.
    // ==========================================================================
    friend bool reorder_tensor_to_soa(const ggml_tensor *, dpct::queue_ptr, const char *);
    friend bool convert_tensor_to_coalesced(const ggml_tensor *, dpct::queue_ptr, const char *);

    // Default: NONE (original AoS layout)
    optimize_feature() = default;

    // Explicit constructor for pre-transformed cached data (MoE experts)
    // Use this ONLY when the data is already in the specified layout.
    explicit optimize_feature(reorder_mode mode) : reorder_(mode) {}

  private:
    reorder_mode reorder_ = reorder_mode::NONE;

    // For view tensors: pointer to the data owner's optimize_feature.
    // Views share data with their parent, so reorder state comes from there.
    // nullptr means this tensor owns its own data (not a view).
    optimize_feature * data_owner_ = nullptr;

    // PRIVATE: Only callable by friend functions!
    // This ONLY sets the flag - does NOT transform data.
    // The friend functions MUST transform data BEFORE calling this.
    //
    // Layout transition state machine:
    //
    // States (reorder_mode):
    //   NONE           - Original AoS tensor layout
    //   SOA            - SoA layout for MMVQ kernels
    //   COALESCED      - Coalesced SoA layout for MMVQ kernels
    //   XMX_COALESCED  - XMX optimized layout for MoE expert weights
    //   XMX_GEMM_TILED - XMX GEMM tiled layout for quant weights
    //   ONEDNN_PACKED  - oneDNN packed layout for matmul weights
    //
    // Valid transitions (upgrade-only):
    //   NONE -> SOA
    //   SOA  -> COALESCED
    //   NONE -> XMX_COALESCED
    //   NONE -> XMX_GEMM_TILED
    //
    // Invalid transitions (warned):
    //   SOA -> NONE, COALESCED -> SOA/NONE, XMX_COALESCED -> *
    //
    // Rationale:
    // - Downgrading requires re-reading original AoS data from host/mmap.
    // - Redundant conversions waste bandwidth and can thrash cache state.
    //
    // Expected callers:
    // - reorder_tensor_to_soa() and convert_tensor_to_coalesced() for MMVQ paths
    // - MoE XMX conversion path for XMX_COALESCED layouts
    void set_reorder_mode_(reorder_mode new_mode, const char * tensor_name, const char * caller) {
        if (new_mode == reorder_) {
            return;  // No change
        }

        bool valid = false;
        if (reorder_ == reorder_mode::NONE && new_mode == reorder_mode::SOA) {
            valid = true;  // NONE → SOA
        } else if (reorder_ == reorder_mode::SOA && new_mode == reorder_mode::COALESCED) {
            valid = true;  // SOA → COALESCED
        } else if (reorder_ == reorder_mode::NONE && new_mode == reorder_mode::XMX_COALESCED) {
            valid = true;  // NONE → XMX_COALESCED (direct conversion for MoE expert weights)
        } else if (reorder_ == reorder_mode::NONE && new_mode == reorder_mode::XMX_GEMM_TILED) {
            valid = true;  // NONE → XMX_GEMM_TILED (direct conversion for XMX GEMM weights)
        }

        if (!valid) {
            fprintf(stderr,
                    "[SYCL WARNING] Invalid reorder transition %d → %d for tensor '%s'. "
                    "Valid: NONE→SOA, SOA→COALESCED, NONE→XMX_COALESCED, NONE→XMX_GEMM_TILED\n",
                    (int) reorder_, (int) new_mode, tensor_name ? tensor_name : "?");
        }

        reorder_ = new_mode;
    }

  public:
    // Reset reorder mode to NONE when new AoS data is written to the tensor
    // This is called by set_tensor to invalidate any prior reordering
    void reset_reorder(const char * tensor_name) {
        if (reorder_ != reorder_mode::NONE) {
            fprintf(stderr, "[REORDER-RESET] %d → 0 for '%s' (data overwritten)\n", (int) reorder_,
                    tensor_name ? tensor_name : "?");
            reorder_ = reorder_mode::NONE;
        }
    }

    // Mark as SoA when data was transformed on CPU before upload (faster than GPU transform)
    // ONLY call this when the data in device memory is already in SoA layout!
    void mark_soa_pretransformed(const char * tensor_name) {
        reorder_ = reorder_mode::SOA;
        GGML_UNUSED(tensor_name);
    }

    // Mark as Coalesced when data was transformed on CPU before upload
    // ONLY call this when the data in device memory is already in Coalesced layout!
    void mark_coalesced_pretransformed(const char * tensor_name) {
        reorder_ = reorder_mode::COALESCED;
        GGML_UNUSED(tensor_name);
    }

    // Mark as XMX Coalesced when MoE expert weights have been converted to XMX-optimized layout
    // ONLY call this when the data has been transformed to XMX coalesced format!
    void mark_xmx_coalesced_pretransformed(const char * tensor_name) {
        reorder_ = reorder_mode::XMX_COALESCED;
        GGML_UNUSED(tensor_name);
    }

    // Mark as XMX GEMM tiled when weights have been converted to XMX GEMM layout
    // ONLY call this when the data has been transformed to XMX GEMM tiled format!
    void mark_xmx_gemm_tiled_pretransformed(const char * tensor_name) {
        reorder_ = reorder_mode::XMX_GEMM_TILED;
        GGML_UNUSED(tensor_name);
    }

    // Set the data owner for view tensors. Call this when creating a view.
    void set_data_owner(optimize_feature * owner) { data_owner_ = owner; }

    // Exact mode checks - use these for kernel dispatch
    bool is_none() const { return get_reorder() == reorder_mode::NONE; }

    bool is_soa() const { return get_reorder() == reorder_mode::SOA; }

    bool is_coalesced() const { return get_reorder() == reorder_mode::COALESCED; }

    bool is_xmx_coalesced() const { return get_reorder() == reorder_mode::XMX_COALESCED; }

    bool is_xmx_gemm_tiled() const { return get_reorder() == reorder_mode::XMX_GEMM_TILED; }

    // Check if ANY reorder was applied - use for "skip if already reordered" logic
    bool is_reordered() const { return get_reorder() != reorder_mode::NONE; }

    // Get current mode - for views, returns the data owner's mode
    reorder_mode get_reorder() const {
        if (data_owner_ != nullptr) {
            if (!ggml_sycl_is_optimize_feature_live(data_owner_)) {
                return reorder_;
            }
            return data_owner_->get_reorder();
        }
        return reorder_;
    }
};

// Tensor Parallelism configuration
// Implements Megatron-LM style column/row parallel for multi-GPU inference
enum class tp_layer_type {
    TP_NONE,             // No tensor parallelism
    TP_COLUMN_PARALLEL,  // Split output features: Q, K, V, gate, up projections
    TP_ROW_PARALLEL,     // Split input features: out_proj, down projections (needs all-reduce)
};

struct ggml_sycl_tp_config {
    bool enabled                        = false;  // Whether tensor parallelism is active
    int  world_size                     = 1;      // Number of GPUs in TP group
    int  rank                           = 0;      // This GPU's rank (0 to world_size-1)
    int  devices[GGML_SYCL_MAX_DEVICES] = { 0 };  // Device IDs in TP group

    // Buffers for all-reduce operations (allocated lazily)
    void * allreduce_buffer[GGML_SYCL_MAX_DEVICES] = { nullptr };
    size_t allreduce_buffer_size                   = 0;

    // Multi-process mode (one GPU per process, coordinated via MPI/CCL)
    bool is_multiprocess = false;  // True if running with mpirun
    int  mpi_rank        = -1;     // MPI rank (process ID)
    int  mpi_world_size  = 0;      // MPI world size (number of processes)
};

// Global TP config (set during init)
extern ggml_sycl_tp_config g_sycl_tp_config;

// Initialize tensor parallelism with specified devices
void ggml_sycl_tp_init(const int * device_ids, int num_devices);

// Clean up tensor parallelism resources
void ggml_sycl_tp_free();

// Perform all-reduce sum across TP group
// buf must be device memory on the calling device
void ggml_sycl_tp_allreduce_sum(float * buf, size_t count, int device, queue_ptr stream);

// Perform all-reduce sum with explicit buffers for each device
void ggml_sycl_tp_allreduce_sum_multi(float ** buf_per_device, size_t count, queue_ptr * streams, int num_devices);

// Get/ensure shared buffer for optimized ALL_REDUCE (malloc_shared for zero-copy)
float * ggml_sycl_tp_ensure_shared_reduce_buffer(size_t bytes);

// Get persistent host buffers for CPU-based ALL_REDUCE (avoids per-call malloc/free)
// Returns two host buffers: one for dev0 data, one for dev1 data
// Grows buffers as needed, reuses across calls
void ggml_sycl_tp_get_host_reduce_buffers(size_t bytes, float ** buf0, float ** buf1);

// Get persistent shared buffer for device-to-device transfers (PP optimization)
// Uses malloc_shared to avoid per-transfer malloc/free overhead
// Auto-grows buffer as needed, reuses across calls
void * ggml_sycl_get_dev2dev_transfer_buffer(size_t bytes);

// Get buffer for double-buffered transfer (returns buffer index via out param)
// Double-buffering allows overlapping src->host copy with host->dst copy
void * ggml_sycl_get_dev2dev_transfer_buffer_double(size_t bytes, int * buf_idx);

// Record that a buffer has a pending transfer (for double-buffering)
void ggml_sycl_set_dev2dev_transfer_event(int buf_idx, sycl::event evt);

// Wait for all pending double-buffered transfers to complete
void ggml_sycl_wait_dev2dev_transfers();

// Free persistent device-to-device transfer buffer (cleanup)
void ggml_sycl_free_dev2dev_transfer_buffer();

// Get the TP rank for a given device
int ggml_sycl_tp_get_rank(int device);

// Check if TP is enabled
bool ggml_sycl_tp_enabled();

// Get TP world size (for graph building)
// In multi-process mode, returns 1 to build full graph
int ggml_sycl_tp_world_size();

// Get actual TP world size (internal use, for ALL_REDUCE)
// Returns true world_size even in multi-process mode
int ggml_sycl_tp_world_size_internal();

// Calculate the slice of a tensor for a given TP rank
void ggml_sycl_tp_get_slice(int64_t total_size, int rank, int world_size, int64_t * offset, int64_t * size);

// Get TP layer type for a tensor (uses cached value if available)
// First call does string matching, subsequent calls just return cached enum
tp_layer_type ggml_sycl_tp_get_layer_type(const ggml_tensor * tensor);

// Check if a tensor requires all-reduce after matmul
bool ggml_sycl_tp_needs_allreduce(const ggml_tensor * tensor);

// Weight sharding functions for tensor parallelism
// Get the sharded dimensions for a TP tensor
void ggml_sycl_tp_get_sharded_dims(const ggml_tensor * tensor,
                                   int                 rank,
                                   int                 world_size,
                                   int64_t *           local_ne0,
                                   int64_t *           local_ne1,
                                   int64_t *           offset_ne0,
                                   int64_t *           offset_ne1);

// Check if a tensor should be sharded for TP
bool ggml_sycl_tp_should_shard(const ggml_tensor * tensor);

// Copy sharded weight data from host to device
void ggml_sycl_tp_copy_weight_shard(void *              dst_device,
                                    const void *        src_host,
                                    const ggml_tensor * tensor,
                                    int                 rank,
                                    int                 world_size,
                                    queue_ptr           stream);

// Get the size in bytes of a sharded tensor for this rank
size_t ggml_sycl_tp_get_shard_size(const ggml_tensor * tensor, int rank, int world_size);

// =============================================================================
// Quantized Communication Buffers (Flash Communication)
// Pre-allocated buffers for INT16 quantized AllReduce - 33% bandwidth reduction
// INT16 has 65536 levels vs INT8's 256 → 0.0015% max error vs 0.4%
// Total bandwidth: 8N bytes (2N×2 INT16 + 4N FP32 result) vs 12N standard
// =============================================================================

struct ggml_sycl_tp_quant_comm_buffers {
    int16_t * dev_q[GGML_SYCL_MAX_DEVICES];       // INT16 device buffers (2 bytes per element)
    float *   dev_minmax[GGML_SYCL_MAX_DEVICES];  // [min, max] per device
    int16_t * host_q0;                            // Host buffer for device 0 INT16
    int16_t * host_q1;                            // Host buffer for device 1 INT16
    float *   host_result;                        // Host buffer for FP32 result
    size_t    capacity;                           // Current allocation size (elements)
    bool      allocated;
};

// Check if quantized AllReduce is enabled via GGML_SYCL_QUANT_ALLREDUCE env var
bool ggml_sycl_quant_allreduce_enabled();

// Check if quantized AllReduce should be used for a given tensor size
// Returns true if enabled AND tensor is large enough to benefit from bandwidth reduction
// Uses GGML_SYCL_QUANT_THRESHOLD env var (default 65536 elements = 256KB FP32)
// Below threshold, FP32 allreduce is faster due to lower kernel overhead
bool ggml_sycl_should_use_quant_allreduce(size_t n_elements);

// Pre-allocate quantized comm buffers (called from ggml_sycl_tp_init)
void ggml_sycl_tp_init_quant_comm_buffers(size_t initial_size);

// Ensure buffers are large enough (resize if needed, called during forward pass)
void ggml_sycl_tp_ensure_quant_comm_buffers(size_t n_elements);

// Get buffer pointers (returns nullptr if not allocated)
ggml_sycl_tp_quant_comm_buffers * ggml_sycl_tp_get_quant_comm_buffers();

// Free quantized comm buffers
void ggml_sycl_tp_free_quant_comm_buffers();

// =============================================================================
// Pipeline Parallelism (PP) configuration
// Implements vLLM-style pipeline parallelism with layer-based device distribution
// =============================================================================

#define GGML_SYCL_PP_MAX_LAYERS 256

struct ggml_sycl_pp_config {
    bool enabled                                  = false;  // Whether pipeline parallelism is active
    int  num_stages                               = 0;      // Number of pipeline stages (typically = num_devices)
    int  layers_per_stage[GGML_SYCL_MAX_DEVICES]  = { 0 };  // Layers per stage (for uneven distribution)
    int  layer_to_device[GGML_SYCL_PP_MAX_LAYERS] = { 0 };  // Quick lookup: layer_id -> device_id
    int  devices[GGML_SYCL_MAX_DEVICES]           = { 0 };  // Device IDs in PP order

    // Inter-stage buffers (malloc_shared for Intel Arc without P2P)
    void * stage_output_buf[GGML_SYCL_MAX_DEVICES] = { nullptr };
    size_t stage_output_size                       = 0;  // Current buffer size per stage

    // Synchronization events for pipelining
    sycl::event stage_complete[GGML_SYCL_MAX_DEVICES];

    // Chunked prefill state
    int32_t chunk_size              = 0;      // Max tokens per prefill chunk (0 = disabled)
    bool    chunked_prefill_enabled = false;  // Whether chunked prefill is active

    // Statistics
    int64_t total_stage_transfers = 0;
    int64_t total_sync_waits      = 0;
};

// Global PP config (set during init)
extern ggml_sycl_pp_config g_sycl_pp_config;

// PP debug output - controlled by GGML_SYCL_PP_DEBUG env var
extern int g_ggml_sycl_pp_debug;

#define GGML_SYCL_PP_DEBUG(...)             \
    do {                                    \
        if (UNLIKELY(g_ggml_sycl_pp_debug)) \
            fprintf(stderr, __VA_ARGS__);   \
    } while (0)

// Initialize pipeline parallelism with specified devices and layer distribution
// If layers_per_stage is nullptr, layers are distributed evenly
void ggml_sycl_pp_init(const int * device_ids,
                       int         num_devices,
                       int         total_layers,
                       const int * layers_per_stage = nullptr);

// Clean up pipeline parallelism resources
void ggml_sycl_pp_free();

// Get the device ID for a given layer
int ggml_sycl_pp_get_device_for_layer(int layer);

// Allocate/ensure inter-stage buffer for given size
// Uses malloc_shared for Intel Arc (no P2P support)
void * ggml_sycl_pp_ensure_stage_buffer(int stage, size_t size);

// Transfer layer output from one stage to the next
// src_device: device that produced the output
// dst_device: device that will consume it
// Returns event that signals transfer completion
sycl::event ggml_sycl_pp_stage_transfer(int          src_device,
                                        int          dst_device,
                                        const void * src,
                                        size_t       size,
                                        queue_ptr    src_queue,
                                        queue_ptr    dst_queue);

// Wait for a stage to complete (blocking)
void ggml_sycl_pp_sync_stage(int stage);

// Wait for all stages to complete
void ggml_sycl_pp_sync_all();

// Check if PP is enabled
bool ggml_sycl_pp_enabled();

// Get number of pipeline stages
int ggml_sycl_pp_num_stages();

// Get layer range for a stage: [start_layer, end_layer)
void ggml_sycl_pp_get_stage_layers(int stage, int * start_layer, int * end_layer);

// Get stage for a given layer
int ggml_sycl_pp_get_stage_for_layer(int layer);

// Set chunked prefill configuration
void ggml_sycl_pp_set_chunked_prefill(int32_t chunk_size, bool enabled);

// Get staging buffer for reading (after stage transfer is complete)
void * ggml_sycl_pp_get_stage_buffer(int stage);

// Get PP statistics (transfers and sync waits)
void ggml_sycl_pp_get_stats(int64_t * transfers, int64_t * syncs);

// Reset PP statistics
void ggml_sycl_pp_reset_stats();

// FFN norm cache for TP: stores FFN norm output immediately after MUL to prevent buffer aliasing
// The GGML scheduler may reuse the FFN norm buffer before TP can use it on device 1
struct ffn_norm_cache_entry {
    void *  data;       // Cached FFN norm output on main device (device 0)
    void *  data_dev1;  // Copy on device 1 for its computation
    int64_t ne0, ne1;   // Dimensions
    size_t  size;       // Buffer size in bytes
    int     pass_id;    // Which compute pass this cache is for (to detect staleness)
};

// Global FFN norm cache indexed by layer number
extern std::unordered_map<int, ffn_norm_cache_entry> g_tp_ffn_norm_cache;
extern std::mutex                                    g_tp_ffn_norm_cache_mutex;
extern int                                           g_tp_current_pass_id;  // Incremented each forward pass
extern bool                                          g_tp_enabled;          // Whether TP mode is enabled

// Store FFN norm output for TP (call after MUL that creates ffn_norm)
void ggml_sycl_tp_cache_ffn_norm(int layer, const void * data, int64_t ne0, int64_t ne1, size_t size, queue_ptr stream);

// Get cached FFN norm for a layer (returns nullptr if not cached or stale)
void * ggml_sycl_tp_get_cached_ffn_norm(int layer, int device);

// Clear FFN norm cache for a layer
void ggml_sycl_tp_clear_ffn_norm_cache(int layer);

// Increment pass ID (call at start of each forward pass)
void ggml_sycl_tp_new_pass();

// FFN input storage: stores the input to FFN column-parallel layers on device 1
// This is needed so that row-parallel (ffn_down) can compute device 1's contribution
struct ffn_input_storage {
    void *  data;      // Buffer on device 1
    int64_t ne0, ne1;  // Dimensions
    size_t  size;      // Buffer size
};

extern std::unordered_map<int, ffn_input_storage> g_tp_ffn_inputs;  // Key: layer number
extern std::mutex                                 g_tp_ffn_input_mutex;

// FFN weight storage: stores references to FFN weight tensors for device 1 computation
struct ffn_weight_refs {
    const ggml_tensor * gate;  // ffn_gate weight tensor
    const ggml_tensor * up;    // ffn_up weight tensor
    const ggml_tensor * down;  // ffn_down weight tensor
};

extern std::unordered_map<int, ffn_weight_refs> g_tp_ffn_weights;  // Key: layer number
extern std::mutex                               g_tp_ffn_weight_mutex;

// Attention input storage: stores the input to attention column-parallel layers on device 1
struct attn_input_storage {
    void *  data;      // Buffer on device 1
    int64_t ne0, ne1;  // Dimensions
    size_t  size;      // Buffer size
};

extern std::unordered_map<int, attn_input_storage> g_tp_attn_inputs;  // Key: layer number
extern std::mutex                                  g_tp_attn_input_mutex;

// Attention weight storage: stores references to attention weight tensors
struct attn_weight_refs {
    const ggml_tensor * q;  // attn_q weight tensor
    const ggml_tensor * k;  // attn_k weight tensor
    const ggml_tensor * v;  // attn_v weight tensor
    const ggml_tensor * o;  // attn_output weight tensor
};

extern std::unordered_map<int, attn_weight_refs> g_tp_attn_weights;  // Key: layer number
extern std::mutex                                g_tp_attn_weight_mutex;

// Async FFN job structure: tracks an in-flight FFN computation on device 1
// This allows device 1 to compute while device 0 continues with other work
struct tp_async_ffn_job {
    int         layer;             // Layer number
    sycl::event completion_event;  // Event signaling computation complete
    float *     result_buf;        // Result buffer (in pinned host memory)
    int64_t     ne0, ne1;          // Output dimensions [N_out, batch]
    size_t      result_size;       // Result buffer size in bytes
    bool        valid;             // Job is valid and pending
};

extern std::unordered_map<int, tp_async_ffn_job> g_tp_async_ffn_jobs;  // Key: layer number
extern std::mutex                                g_tp_async_ffn_mutex;

// Async attention job structure: tracks an in-flight attention computation on device 1
struct tp_async_attn_job {
    int         layer;             // Layer number
    sycl::event completion_event;  // Event signaling computation complete
    float *     result_buf;        // Result buffer (in pinned host memory)
    int64_t     ne0, ne1;          // Output dimensions
    size_t      result_size;       // Result buffer size in bytes
    bool        valid;             // Job is valid and pending
};

extern std::unordered_map<int, tp_async_attn_job> g_tp_async_attn_jobs;  // Key: layer number
extern std::mutex                                 g_tp_async_attn_mutex;

// Extract layer number from tensor name (e.g., "blk.0.ffn_gate" -> 0)
int ggml_sycl_tp_extract_layer_number(const char * name);

// =============================================================================
// Thread-based pipelining for device 1 FFN computation
// Uses a dedicated worker thread instead of SYCL async events (which don't work
// with in-order queues that have multiple wait() calls).
// =============================================================================

// FFN work item: describes an FFN computation to be performed on device 1
struct tp_ffn_work_item {
    int             layer;       // Layer number
    float *         input_dev1;  // Input pointer on device 1 (already copied)
    int64_t         K_full;      // Input dimension
    int64_t         batch;       // Batch size
    ffn_weight_refs weights;     // Weight tensor references

    // Output info (filled in by caller for result allocation)
    int64_t N_out;        // Output dimension
    size_t  result_size;  // Expected result size in bytes
};

// FFN result: result of a completed FFN computation
struct tp_ffn_result {
    int     layer;        // Layer number
    float * result_buf;   // Result buffer (host-pinned memory)
    int64_t ne0, ne1;     // Output dimensions
    size_t  result_size;  // Result size in bytes
    bool    valid;        // Result is valid and ready to consume
};

// Device 1 worker thread: processes FFN jobs independently from main thread
struct tp_device1_worker {
    std::thread worker_thread;

    // Work queue: main thread submits, worker thread processes
    std::queue<tp_ffn_work_item> work_queue;
    std::mutex                   work_mutex;
    std::condition_variable      work_cv;

    // Results: worker thread produces, main thread consumes
    std::unordered_map<int, tp_ffn_result> results;  // Key: layer number
    std::mutex                             result_mutex;
    std::condition_variable                result_cv;

    // Control
    std::atomic<bool> shutdown{ false };
    std::atomic<bool> initialized{ false };

    // Context pointer (set during init)
    void * ctx;  // ggml_backend_sycl_context *
};

// Global worker instance
extern tp_device1_worker g_tp_device1_worker;

// Global flag to enable/disable thread-based pipelining
extern int g_ggml_sycl_tp_threaded_ffn;  // 0 = disabled, 1 = enabled

// Thread-based pipelining functions
void            ggml_sycl_tp_worker_init(void * ctx);                         // Initialize worker thread
void            ggml_sycl_tp_worker_shutdown();                               // Shutdown worker thread
void            ggml_sycl_tp_submit_ffn_work(const tp_ffn_work_item & work);  // Submit work to queue
tp_ffn_result * ggml_sycl_tp_get_ffn_result(int layer, bool wait);            // Get result (optional wait)
void            ggml_sycl_tp_release_ffn_result(int layer);                   // Release result memory

// =============================================================================
// Persistent FFN compute buffers for TP mode
// Pre-allocate all FFN buffers once per layer to eliminate 535K+ malloc/free calls
// =============================================================================

struct tp_ffn_compute_buffers {
    // Input quantization buffer
    char * input_q8_dev;
    size_t input_q8_size;

    // Intermediate float buffers (gate, up, hidden outputs)
    float * gate_out;
    float * up_out;
    float * hidden_out;
    size_t  hidden_size;  // Size of gate_out, up_out, hidden_out

    // Hidden quantization buffer for down matmul
    char * hidden_q8_dev;
    size_t hidden_q8_size;

    // Output buffer for partial result
    float * partial_out;
    size_t  partial_size;

    // Track allocated sizes (for resize detection)
    int64_t K_full_padded;
    int64_t N_hidden_shard_padded;
    int64_t batch_max;
    int64_t N_out;

    // Flag indicating if buffers are allocated
    bool allocated;

    // Device ID these buffers are allocated on
    int device_id;
};

// Global map of persistent FFN buffers indexed by layer
extern std::unordered_map<int, tp_ffn_compute_buffers> g_tp_ffn_buffers;
extern std::mutex                                      g_tp_ffn_buffers_mutex;

// Ensure persistent FFN buffers are allocated for a layer
// Returns pointer to buffers, allocates if needed, resizes if dimensions changed
tp_ffn_compute_buffers * ggml_sycl_tp_ensure_ffn_buffers(int       layer,
                                                         int       device,
                                                         queue_ptr stream,
                                                         int64_t   K_full_padded,
                                                         int64_t   N_hidden_shard_padded,
                                                         int64_t   batch,
                                                         int64_t   N_out);

// Free all persistent FFN buffers (called during cleanup)
void ggml_sycl_tp_free_ffn_buffers();

// =============================================================================
// Persistent host staging buffer for TP input copies
// =============================================================================

struct tp_host_staging_buffer {
    float * buf;
    size_t  size;
    size_t  capacity;
};

extern tp_host_staging_buffer g_tp_host_staging;
extern std::mutex             g_tp_host_staging_mutex;

// Ensure host staging buffer has at least the given capacity
float * ggml_sycl_tp_ensure_host_staging(size_t size, queue_ptr stream);

// Free host staging buffer
void ggml_sycl_tp_free_host_staging();

struct ggml_sycl_pool {
    virtual ~ggml_sycl_pool() = default;

    virtual void * alloc(size_t size, size_t * actual_size) = 0;
    virtual void   free(void * ptr, size_t size)            = 0;
};

// Allocation tracing (optional). Enable with GGML_SYCL_ALLOC_TRACE=1.
bool  ggml_sycl_alloc_trace_enabled();
void  ggml_sycl_alloc_trace_dump(const char * reason);
void  ggml_sycl_alloc_trace_record(const char * kind, size_t size, const char * tag);
void * ggml_sycl_malloc_device(size_t size, const sycl::queue & queue, const char * tag);
void * ggml_sycl_malloc_host(size_t size, const sycl::queue & queue, const char * tag);
void * ggml_sycl_malloc_shared(size_t size, const sycl::queue & queue, const char * tag);

#define GGML_SYCL_STRINGIFY_HELPER(x) #x
#define GGML_SYCL_STRINGIFY(x) GGML_SYCL_STRINGIFY_HELPER(x)
#define GGML_SYCL_ALLOC_TAG (__FILE__ ":" GGML_SYCL_STRINGIFY(__LINE__))

template <typename T>
inline T * ggml_sycl_malloc_device_t(size_t count, const sycl::queue & queue, const char * tag) {
    return static_cast<T *>(ggml_sycl_malloc_device(sizeof(T) * count, queue, tag));
}

template <typename T>
inline T * ggml_sycl_malloc_host_t(size_t count, const sycl::queue & queue, const char * tag) {
    return static_cast<T *>(ggml_sycl_malloc_host(sizeof(T) * count, queue, tag));
}

template <typename T>
inline T * ggml_sycl_malloc_shared_t(size_t count, const sycl::queue & queue, const char * tag) {
    return static_cast<T *>(ggml_sycl_malloc_shared(sizeof(T) * count, queue, tag));
}

#if GGML_SYCL_MAX_DEVICES > 0
inline void * ggml_sycl_malloc_device_tracked_bytes(size_t bytes, sycl::queue & queue, const char * tag) {
    const int device_id = ggml_sycl_get_device_id_from_queue(queue);
    ggml_sycl::unified_cache_add_runtime_bytes(device_id, bytes);
    void * ptr = ggml_sycl_malloc_device(bytes, queue, tag);
    if (!ptr) {
        ggml_sycl::unified_cache_sub_runtime_bytes(device_id, bytes);
    }
    return ptr;
}

inline void ggml_sycl_free_device_tracked_bytes(void * ptr, size_t bytes, sycl::queue & queue) {
    if (!ptr) {
        return;
    }
    sycl::free(ptr, queue);
    ggml_sycl::unified_cache_sub_runtime_bytes(ggml_sycl_get_device_id_from_queue(queue), bytes);
}

template <typename T>
inline T * ggml_sycl_malloc_device_tracked_t(size_t count, sycl::queue & queue, const char * tag) {
    return static_cast<T *>(ggml_sycl_malloc_device_tracked_bytes(sizeof(T) * count, queue, tag));
}

template <typename T>
inline void ggml_sycl_free_device_tracked_t(T * ptr, size_t count, sycl::queue & queue) {
    ggml_sycl_free_device_tracked_bytes(ptr, sizeof(T) * count, queue);
}

inline void * ggml_sycl_malloc_host_tracked_bytes(size_t bytes, sycl::queue & queue, const char * tag) {
    ggml_sycl::unified_cache_add_runtime_host_bytes(bytes);
    void * ptr = ggml_sycl_malloc_host(bytes, queue, tag);
    if (!ptr) {
        ggml_sycl::unified_cache_sub_runtime_host_bytes(bytes);
    }
    return ptr;
}

inline void ggml_sycl_free_host_tracked_bytes(void * ptr, size_t bytes, sycl::queue & queue) {
    if (!ptr) {
        return;
    }
    sycl::free(ptr, queue);
    ggml_sycl::unified_cache_sub_runtime_host_bytes(bytes);
}

template <typename T>
inline T * ggml_sycl_malloc_host_tracked_t(size_t count, sycl::queue & queue, const char * tag) {
    return static_cast<T *>(ggml_sycl_malloc_host_tracked_bytes(sizeof(T) * count, queue, tag));
}

template <typename T>
inline void ggml_sycl_free_host_tracked_t(T * ptr, size_t count, sycl::queue & queue) {
    ggml_sycl_free_host_tracked_bytes(ptr, sizeof(T) * count, queue);
}
#endif

#define GGML_SYCL_MALLOC_DEVICE_BYTES(size, queue) ggml_sycl_malloc_device((size), (queue), GGML_SYCL_ALLOC_TAG)
#define GGML_SYCL_MALLOC_HOST_BYTES(size, queue) ggml_sycl_malloc_host((size), (queue), GGML_SYCL_ALLOC_TAG)
#define GGML_SYCL_MALLOC_SHARED_BYTES(size, queue) ggml_sycl_malloc_shared((size), (queue), GGML_SYCL_ALLOC_TAG)

#define GGML_SYCL_MALLOC_DEVICE_T(T, count, queue) ggml_sycl_malloc_device_t<T>((count), (queue), GGML_SYCL_ALLOC_TAG)
#define GGML_SYCL_MALLOC_HOST_T(T, count, queue) ggml_sycl_malloc_host_t<T>((count), (queue), GGML_SYCL_ALLOC_TAG)
#define GGML_SYCL_MALLOC_SHARED_T(T, count, queue) ggml_sycl_malloc_shared_t<T>((count), (queue), GGML_SYCL_ALLOC_TAG)

template <typename T> struct ggml_sycl_pool_alloc {
    ggml_sycl_pool * pool        = nullptr;
    T *              ptr         = nullptr;
    size_t           actual_size = 0;

    explicit ggml_sycl_pool_alloc(ggml_sycl_pool & pool) : pool(&pool) {}

    ggml_sycl_pool_alloc(ggml_sycl_pool & pool, size_t size) : pool(&pool) { alloc(size); }

    ~ggml_sycl_pool_alloc() {
        if (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }

    T * realloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        if (ptr) {
            pool->free(ptr, actual_size);
        }
        ptr = (T *) pool->alloc(size * sizeof(T), &this->actual_size);
        return ptr;
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * sizeof(T), &this->actual_size);
        return ptr;
    }

    T * alloc(ggml_sycl_pool & pool, size_t size) {
        this->pool = &pool;
        return alloc(size);
    }

    T * get() { return ptr; }

    ggml_sycl_pool_alloc()                                         = default;
    ggml_sycl_pool_alloc(const ggml_sycl_pool_alloc &)             = delete;
    ggml_sycl_pool_alloc(ggml_sycl_pool_alloc &&)                  = delete;
    ggml_sycl_pool_alloc & operator=(const ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc & operator=(ggml_sycl_pool_alloc &&)      = delete;
};

// backend interface

struct ggml_tensor_extra_gpu {
    std::atomic<int> refcount{ 1 };
    uint64_t         cache_uuid = 0;                                        // Monotonic cache identity for weights
    uint64_t         model_id   = 0;                                        // Model identifier for cache keys
    void *           data_device[GGML_SYCL_MAX_DEVICES];                    // 1 pointer for each device for split
                                                                            // tensors
    size_t           data_device_size[GGML_SYCL_MAX_DEVICES] = { 0 };       // Allocation sizes for data_device
    dpct::event_ptr  events[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS];  // events for synchronizing multiple GPUs
    optimize_feature optimized_feature = {};  // Must have = {} to ensure default member initializers apply

    // Unified layout descriptor (new system - coexists with optimize_feature during migration)
    tensor_layout_info layout;
    bool               layout_dirty = false;                // Weight data overwritten; layout must be re-materialized

    tp_layer_type tp_type        = tp_layer_type::TP_NONE;  // Cached TP type (set once, avoids string compare)
    bool          tp_type_cached = false;                   // Whether tp_type has been computed

    // Tensor Parallelism sharding info
    // When TP is enabled, this tensor may hold only a shard of the full weight
    bool    tp_sharded        = false;  // True if this tensor holds a shard
    bool    tp_usm_host       = false;  // True if allocated with malloc_host (cross-device accessible)
    int64_t tp_original_ne[4] = { 0 };  // Original (full) dimensions before sharding
    int64_t tp_local_ne[4]    = { 0 };  // Local dimensions of the shard
    int64_t tp_offset_ne[4]   = { 0 };  // Offset into the original tensor
    int     tp_rank           = 0;      // Which rank this shard belongs to
    int     tp_world_size     = 1;      // Total number of ranks

    // XMX tile-aligned MXFP4 layout (cached at first use)
    void * xmx_mxfp4_tiled[GGML_SYCL_MAX_DEVICES]       = { nullptr };
    size_t xmx_mxfp4_tiled_size                         = 0;
    bool   xmx_mxfp4_tiled_owned[GGML_SYCL_MAX_DEVICES] = { false };

    // Temporary AoS staging for MXFP4 tiled conversion (host -> device)
    void * xmx_mxfp4_tiled_aos_staging[GGML_SYCL_MAX_DEVICES]      = { nullptr };
    size_t xmx_mxfp4_tiled_aos_staging_size[GGML_SYCL_MAX_DEVICES] = { 0 };

    // Track async tile conversion completion for graph compatibility
    sycl::event xmx_mxfp4_tiled_conversion_evt[GGML_SYCL_MAX_DEVICES];
    bool        xmx_mxfp4_tiled_conversion_complete[GGML_SYCL_MAX_DEVICES] = { false };
    std::mutex  xmx_tiled_conversion_mutex[GGML_SYCL_MAX_DEVICES];  // Protect concurrent access

    // MoE expert pointer table (device + host staging) for per-expert layout access
    void *              moe_expert_ptrs_device[GGML_SYCL_MAX_DEVICES] = { nullptr };
    size_t              moe_expert_ptrs_size[GGML_SYCL_MAX_DEVICES]   = { 0 };
    std::vector<void *> moe_expert_ptrs_host[GGML_SYCL_MAX_DEVICES];

    // MoE compact pointer list (row-major by id) and missing flag
    void * moe_expert_ptrs_compact_device[GGML_SYCL_MAX_DEVICES]   = { nullptr };
    size_t moe_expert_ptrs_compact_size[GGML_SYCL_MAX_DEVICES]     = { 0 };
    size_t moe_expert_ptrs_compact_capacity[GGML_SYCL_MAX_DEVICES] = { 0 };
    int *  moe_expert_ptrs_missing_device[GGML_SYCL_MAX_DEVICES]   = { nullptr };

    // MoE expert hotness tracking (per layer)
    std::vector<float> moe_expert_scores;
};

void retain_extra_gpu(ggml_tensor_extra_gpu * extra);
void release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams = {});

// =============================================================================
// Helper: Get effective reorder_mode from unified layout.mode or legacy path
// =============================================================================
static inline layout_mode get_effective_layout_mode(const ggml_tensor_extra_gpu * extra) {
    if (!extra) {
        return GGML_LAYOUT_AOS;
    }

    if (extra->layout.mode != GGML_LAYOUT_AOS) {
        return extra->layout.mode;
    }

    switch (extra->optimized_feature.get_reorder()) {
        case reorder_mode::SOA:
            return GGML_LAYOUT_SOA;
        case reorder_mode::COALESCED:
            return GGML_LAYOUT_COALESCED;
        default:
            return GGML_LAYOUT_AOS;
    }
}

static inline reorder_mode get_effective_reorder_mode(const ggml_tensor_extra_gpu * extra) {
    const layout_mode mode = get_effective_layout_mode(extra);

    switch (mode) {
        case GGML_LAYOUT_SOA:
            return reorder_mode::SOA;
        case GGML_LAYOUT_COALESCED:
            return reorder_mode::COALESCED;
        case GGML_LAYOUT_XMX_TILED:
        case GGML_LAYOUT_XMX_GEMM_TILED:
        case GGML_LAYOUT_ONEDNN_PACKED:
        case GGML_LAYOUT_ONEDNN_WOQ:
            return reorder_mode::NONE;  // XMX uses separate dispatch
        default:
            return reorder_mode::NONE;
    }
}

static inline bool ggml_sycl_layout_is_soa(const ggml_tensor_extra_gpu * extra) {
    return get_effective_layout_mode(extra) == GGML_LAYOUT_SOA;
}

static inline bool ggml_sycl_layout_is_coalesced(const ggml_tensor_extra_gpu * extra) {
    return get_effective_layout_mode(extra) == GGML_LAYOUT_COALESCED;
}

static inline bool ggml_sycl_layout_is_tiled(const ggml_tensor_extra_gpu * extra) {
    return get_effective_layout_mode(extra) == GGML_LAYOUT_XMX_TILED;
}

static inline bool ggml_sycl_layout_is_reordered(const ggml_tensor_extra_gpu * extra) {
    return get_effective_layout_mode(extra) != GGML_LAYOUT_AOS;
}

static inline bool ggml_sycl_layout_is_soa_or_coalesced(const ggml_tensor_extra_gpu * extra) {
    const layout_mode mode = get_effective_layout_mode(extra);
    return mode == GGML_LAYOUT_SOA || mode == GGML_LAYOUT_COALESCED;
}

// Accessors for backend-managed layout metadata
inline const ggml_tensor_layout * ggml_sycl_get_layout_info(const ggml_tensor * tensor) {
    return tensor ? tensor->layout : nullptr;
}

// Get the correct data pointer for a tensor on a specific device
// For TP buffers, returns device-specific pointer; otherwise returns tensor->data
// In TP mode, if returning tensor->data, stages it to USM memory first
inline void ggml_sycl_refresh_cached_input_ptr(void * dst, const void * src, size_t bytes, int device) {
    if (dst == nullptr || src == nullptr || bytes == 0) {
        return;
    }
    sycl::queue & q = ggml_sycl_get_device(device).default_queue();
    sycl::usm::alloc alloc = sycl::get_pointer_type(dst, q.get_context());
    if (alloc == sycl::usm::alloc::device) {
        const bool avoid_wait =
            ggml_sycl_graph_recording_active() || ggml_sycl_graph_inflight_count() > 0;
        if (avoid_wait) {
            q.memcpy(dst, src, bytes);
        } else {
            q.memcpy(dst, src, bytes).wait();
        }
        return;
    }
    // Host/shared/unknown: use CPU memcpy
    std::memcpy(dst, src, bytes);
}

// Cold path: full resolution chain (tiered cache, get_pointer_type, staging).
// Defined in ggml-sycl.cpp to avoid inlining a 100-line function.
void * ggml_sycl_get_data_ptr_slow(const ggml_tensor * tensor, int device);

// Hot path: 2 dereferences + 1 null check for common case (model fits in VRAM)
// Input tensor refresh is handled by set_tensor (scheduler) and graph_refresh_input_tensors (replay),
// NOT here — calling refresh here would add get_pointer_type() driver round-trips to every resolution.
inline void * ggml_sycl_get_data_ptr(const ggml_tensor * tensor, int device) {
    if (tensor == nullptr) {
        return nullptr;
    }
    if (tensor->extra != nullptr) {
        void * ptr = static_cast<ggml_tensor_extra_gpu *>(tensor->extra)->data_device[device];
        if (ptr != nullptr) {
            return ptr;
        }
    }
    return ggml_sycl_get_data_ptr_slow(tensor, device);
}

inline bool ggml_sycl_tensor_is_weight(const ggml_tensor * tensor) {
    if (!tensor || !tensor->buffer) {
        return false;
    }
    if (UNLIKELY(!ggml_backend_buffer_is_valid(tensor->buffer))) {
        if (g_ggml_sycl_debug) {
            GGML_LOG_WARN("[SYCL] tensor=%s has invalid buffer pointer=%p\n", tensor->name, (void *) tensor->buffer);
        }
        return false;
    }
    return ggml_backend_buffer_get_usage(tensor->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

// Get the effective layout data pointer for a tensor on a specific device.
// If a layout is active for the device, return that pointer; otherwise fallback to data pointer.
inline void * ggml_sycl_get_layout_ptr(const ggml_tensor * tensor, int device) {
    if (tensor == nullptr) {
        return nullptr;
    }

    if (const char * dbg = std::getenv("GGML_SYCL_LAYOUT_PTR_DEBUG")) {
        if (std::string(dbg) == "1") {
            static std::atomic<int> dbg_left{ 8 };
            int                     remaining = dbg_left.fetch_sub(1);
            if (remaining > 0) {
                const bool   is_weight = ggml_sycl_tensor_is_weight(tensor);
                const bool   host_buf  = tensor->buffer && ggml_backend_buffer_is_host(tensor->buffer);
                const int    usage     = tensor->buffer ? (int) ggml_backend_buffer_get_usage(tensor->buffer) : -1;
                const char * buft_name =
                    tensor->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(tensor->buffer)) : "null";
                GGML_LOG_INFO("[LAYOUT-PTR-DBG] tensor=%s is_weight=%d weights_evictable=%d host=%d usage=%d buft=%s\n",
                              tensor->name, is_weight ? 1 : 0, ggml_backend_sycl_weights_evictable() ? 1 : 0,
                              host_buf ? 1 : 0, usage, buft_name);
            }
        }
    }

    layout_mode target = GGML_LAYOUT_AOS;
    if (ggml_sycl_tensor_is_weight(tensor)) {
        // Use get_layout_choice_for_tensor which has fallback registration logic.
        // This handles weight tensors that may not have been visited during finalize_layouts,
        // such as norm scale weights accessed via fused operations (RMS_NORM+MUL).
        if (!ggml_sycl_get_layout_choice_for_tensor(tensor, device, &target)) {
            // If no layout choice can be determined, fall back to AOS.
            target = GGML_LAYOUT_AOS;
        }
    }

    const bool host_weights =
        ggml_sycl_tensor_is_weight(tensor) && tensor->buffer && ggml_backend_buffer_is_host(tensor->buffer);
    const bool device_weights =
        ggml_sycl_tensor_is_weight(tensor) && tensor->buffer && ggml_backend_buffer_is_sycl(tensor->buffer);
    const bool cache_weights =
        ggml_sycl::unified_cache_enabled() &&
        (host_weights ||
         (device_weights &&
          (target != GGML_LAYOUT_AOS || ggml_backend_sycl_weights_evictable())));
    if (cache_weights) {
        void * cached = ggml_sycl_get_weight_layout_ptr(tensor, device, target);
        if (cached) {
            if (host_weights) {
                ggml_sycl_layout_ptr_stat(ggml_sycl_layout_ptr_event::HOST_CACHE_TARGET_HIT);
            }
            return cached;
        }
        if (host_weights && target != GGML_LAYOUT_AOS) {
            cached = ggml_sycl_get_weight_layout_ptr(tensor, device, GGML_LAYOUT_AOS);
            if (cached) {
                ggml_sycl_layout_ptr_stat(ggml_sycl_layout_ptr_event::HOST_CACHE_AOS_HIT);
                return cached;
            }
        }
        if (host_weights) {
            // Check layer stream manager first
            if (ggml_sycl::layer_streaming_active(device) && tensor->name) {
                void * streamed = ggml_sycl::layer_streaming_get_weight_ptr(device, tensor->name);
                if (streamed) {
                    ggml_sycl_layout_ptr_stat(ggml_sycl_layout_ptr_event::HOST_CACHE_DATA_FALLBACK);
                    return streamed;
                }
            }
            const ggml_tensor_layout * layout = ggml_sycl_get_layout_info(tensor);
            if (layout != nullptr && layout->data_ptr != nullptr) {
                if (layout->device_id < 0 || layout->device_id == device) {
                    if (auto * cache = ggml_sycl::get_unified_cache_for_device(device)) {
                        ggml_sycl_cache_id cache_key = ggml_backend_sycl_get_weight_cache_key(tensor, device);
                        if (cache_key.valid && cache->is_cached(cache_key, layout->mode)) {
                            ggml_sycl_layout_ptr_stat(ggml_sycl_layout_ptr_event::HOST_CACHE_LAYOUT_FALLBACK);
                            return ggml_tensor_get_layout_ptr(tensor);
                        }
                    }
                }
            }
            ggml_sycl_layout_ptr_stat(ggml_sycl_layout_ptr_event::HOST_CACHE_DATA_FALLBACK);
            return ggml_sycl_get_data_ptr(tensor, device);
        }
    }

    const ggml_tensor_layout * layout        = ggml_sycl_get_layout_info(tensor);
    bool                       layout_cached = true;
    if (layout != nullptr && layout->data_ptr != nullptr && ggml_sycl_tensor_is_weight(tensor) &&
        ggml_sycl::unified_cache_enabled()) {
        if (auto * cache = ggml_sycl::get_unified_cache_for_device(device)) {
            ggml_sycl_cache_id cache_key = ggml_backend_sycl_get_weight_cache_key(tensor, device);
            layout_cached                = cache_key.valid && cache->is_cached(cache_key, layout->mode);
        }
    }
    if (layout != nullptr && layout->data_ptr != nullptr && layout_cached) {
        if ((layout->device_id < 0 || layout->device_id == device) &&
            (!ggml_sycl_tensor_is_weight(tensor) || layout->mode == target)) {
            return ggml_tensor_get_layout_ptr(tensor);
        }
    }

    return ggml_sycl_get_data_ptr(tensor, device);
}

// Resolve a weight layout pointer for a specific target layout.
// Returns nullptr if the requested layout cannot be satisfied.
inline bool ggml_sycl_unified_dispatch_env_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = std::getenv("GGML_SYCL_UNIFIED_DISPATCH");
        // Keep this helper consistent with ggml_sycl_unified_dispatch_enabled().
        enabled          = (env == nullptr || std::atoi(env) != 0) ? 1 : 0;
    }
    return enabled != 0;
}

inline bool ggml_sycl_should_use_unified_type(ggml_type type) {
    // Mirror ggml_sycl::should_use_unified() without pulling in dispatch.hpp
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_MXFP4;
}

inline void * ggml_sycl_get_layout_ptr_for(const ggml_tensor * tensor,
                                           int                 device,
                                           layout_mode         target,
                                           const char **       out_source = nullptr) {
    bool layout_dirty = false;
    if (tensor == nullptr) {
        if (out_source) {
            *out_source = "null_tensor";
        }
        return nullptr;
    }

    if (out_source) {
        *out_source = "unknown";
    }

    if (const char * dbg = std::getenv("GGML_SYCL_LAYOUT_PTR_DEBUG")) {
        if (std::string(dbg) == "1") {
            static std::atomic<int> dbg_left_for{ 8 };
            int                     remaining = dbg_left_for.fetch_sub(1);
            if (remaining > 0) {
                const bool   is_weight = ggml_sycl_tensor_is_weight(tensor);
                const bool   host_buf  = tensor->buffer && ggml_backend_buffer_is_host(tensor->buffer);
                const int    usage     = tensor->buffer ? (int) ggml_backend_buffer_get_usage(tensor->buffer) : -1;
                const char * buft_name =
                    tensor->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(tensor->buffer)) : "null";
                GGML_LOG_INFO(
                    "[LAYOUT-PTR-DBG] tensor=%s is_weight=%d target=%d weights_evictable=%d host=%d usage=%d buft=%s\n",
                    tensor->name, is_weight ? 1 : 0, (int) target, ggml_backend_sycl_weights_evictable() ? 1 : 0,
                    host_buf ? 1 : 0, usage, buft_name);
            }
        }
    }

    if (ggml_sycl_tensor_is_weight(tensor)) {
        const bool unified_aos_request =
            ggml_sycl_unified_dispatch_env_enabled() &&
            ggml_sycl_should_use_unified_type(tensor->type) &&
            target == GGML_LAYOUT_AOS;

        const layout_mode resolved = ggml_sycl_adjust_layout_for_tensor(tensor, target, device);
        if (!unified_aos_request && resolved != target) {
            return nullptr;
        }
        // Ensure layout choice is registered (with fallback for weights not visited during finalization).
        // This handles weight tensors accessed via fused operations.
        layout_mode registered_layout = GGML_LAYOUT_AOS;
        if (!ggml_sycl_get_layout_choice_for_tensor(tensor, device, &registered_layout)) {
            // If no layout choice exists, allow AoS to fall back to raw storage.
            if (target != GGML_LAYOUT_AOS) {
                if (out_source) {
                    *out_source = "no_layout_choice";
                }
                return nullptr;
            }
            if (out_source) {
                *out_source = "no_layout_choice_aos";
            }
        } else {
            // Verify the registered layout matches the requested target
            if (!unified_aos_request && registered_layout != target) {
                if (out_source) {
                    *out_source = "layout_mismatch";
                }
                return nullptr;
            }
            if (unified_aos_request && registered_layout != target && g_ggml_sycl_debug) {
                GGML_SYCL_DEBUG(
                    "[LAYOUT] unified AoS request bypassing registered layout=%d for %s\n",
                    (int) registered_layout, tensor->name ? tensor->name : "(null)");
            }
        }
    }

    if (ggml_sycl_tensor_is_weight(tensor)) {
        const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(tensor->extra);
        layout_dirty = extra && extra->layout_dirty && target != GGML_LAYOUT_AOS;
    }

    const ggml_tensor_layout * layout        = ggml_sycl_get_layout_info(tensor);
    bool                       layout_cached = true;
    if (layout != nullptr && layout->data_ptr != nullptr && ggml_sycl_tensor_is_weight(tensor) &&
        ggml_sycl::unified_cache_enabled()) {
        if (auto * cache = ggml_sycl::get_unified_cache_for_device(device)) {
            ggml_sycl_cache_id cache_key = ggml_backend_sycl_get_weight_cache_key(tensor, device);
            layout_cached                = cache_key.valid && cache->is_cached(cache_key, layout->mode);
        }
    }
    if (layout != nullptr && layout->data_ptr != nullptr && !layout_cached && !layout_dirty &&
        ggml_sycl_tensor_is_weight(tensor)) {
        if ((layout->device_id < 0 || layout->device_id == device) && layout->mode == target &&
            layout->data_ptr == tensor->data) {
            const sycl::usm::alloc alloc =
                sycl::get_pointer_type(layout->data_ptr, ggml_sycl_get_device(device).default_queue().get_context());
            if (alloc == sycl::usm::alloc::device) {
                if (out_source) {
                    *out_source = "layout_ptr_override";
                }
                return ggml_tensor_get_layout_ptr(tensor);
            }
        }
    }
    if (layout != nullptr && layout->data_ptr != nullptr && layout_cached && !layout_dirty) {
        bool layout_ptr_usable = true;
        if (target != GGML_LAYOUT_AOS && ggml_sycl_tensor_is_weight(tensor)) {
            const sycl::usm::alloc alloc =
                sycl::get_pointer_type(layout->data_ptr, ggml_sycl_get_device(device).default_queue().get_context());
            if (alloc == sycl::usm::alloc::unknown) {
                layout_ptr_usable = false;
                if (out_source) {
                    *out_source = "layout_ptr_non_usm";
                }
            }
        }
        if (layout_ptr_usable) {
            if ((layout->device_id < 0 || layout->device_id == device) && layout->mode == target &&
                layout->data_ptr == tensor->data) {
                if (out_source) {
                    *out_source = "layout_ptr";
                }
                return ggml_tensor_get_layout_ptr(tensor);
            }
        }
    }

    if (layout != nullptr && layout->data_ptr != nullptr && layout_cached && !layout_dirty) {
        bool layout_ptr_usable = true;
        if (target != GGML_LAYOUT_AOS && ggml_sycl_tensor_is_weight(tensor)) {
            const sycl::usm::alloc alloc =
                sycl::get_pointer_type(layout->data_ptr, ggml_sycl_get_device(device).default_queue().get_context());
            if (alloc == sycl::usm::alloc::unknown) {
                layout_ptr_usable = false;
                if (out_source) {
                    *out_source = "layout_ptr_non_usm";
                }
            }
        }
        if (layout_ptr_usable) {
            if ((layout->device_id < 0 || layout->device_id == device) && layout->mode == target) {
                if (out_source) {
                    *out_source = "layout_ptr";
                }
                return ggml_tensor_get_layout_ptr(tensor);
            }
        }
    }

    const bool host_weights =
        ggml_sycl_tensor_is_weight(tensor) && tensor->buffer && ggml_backend_buffer_is_host(tensor->buffer);
    const bool device_weights =
        ggml_sycl_tensor_is_weight(tensor) && tensor->buffer && ggml_backend_buffer_is_sycl(tensor->buffer);
    const bool cache_weights =
        ggml_sycl::unified_cache_enabled() &&
        (host_weights ||
         (device_weights &&
          (target != GGML_LAYOUT_AOS || ggml_backend_sycl_weights_evictable())));

    if (cache_weights) {
        void * cached = ggml_sycl_get_weight_layout_ptr(tensor, device, target);
        if (cached) {
            if (target != GGML_LAYOUT_AOS && ggml_sycl_tensor_is_weight(tensor)) {
                const sycl::usm::alloc alloc =
                    sycl::get_pointer_type(cached, ggml_sycl_get_device(device).default_queue().get_context());
                if (alloc == sycl::usm::alloc::unknown) {
                    if (out_source) {
                        *out_source = "cache_ptr_non_usm";
                    }
                    return nullptr;
                }
            }
            if (host_weights) {
                ggml_sycl_layout_ptr_stat(ggml_sycl_layout_ptr_event::HOST_CACHE_TARGET_HIT);
            }
            if (out_source) {
                *out_source = "unified_cache_target";
            }
            return cached;
        }
        if (host_weights) {
            if (target == GGML_LAYOUT_AOS) {
                // Evictable weights must go through unified cache to avoid direct host/mmap access.
                if (ggml_backend_sycl_weights_evictable()) {
                    ggml_sycl_layout_ptr_stat(ggml_sycl_layout_ptr_event::HOST_CACHE_MISS);
                    if (out_source) {
                        *out_source = "host_cache_aos_miss";
                    }
                    return nullptr;
                }
                ggml_sycl_layout_ptr_stat(ggml_sycl_layout_ptr_event::HOST_CACHE_DATA_FALLBACK);
                if (out_source) {
                    *out_source = "host_cache_aos";
                }
                return ggml_sycl_get_data_ptr(tensor, device);
            }
            ggml_sycl_layout_ptr_stat(ggml_sycl_layout_ptr_event::HOST_CACHE_MISS);
            if (out_source) {
                *out_source = "host_cache_miss";
            }
            return nullptr;
        }
    }

    if (!ggml_sycl_tensor_is_weight(tensor)) {
        if (out_source) {
            *out_source = "data_ptr";
        }
        return ggml_sycl_get_data_ptr(tensor, device);
    }
    if (target == GGML_LAYOUT_AOS) {
        if (ggml_sycl::unified_cache_enabled() && ggml_backend_sycl_weights_evictable()) {
            if (out_source) {
                *out_source = "aos_requires_cache";
            }
            return nullptr;
        }
        if (out_source) {
            *out_source = "data_ptr";
        }
        return ggml_sycl_get_data_ptr(tensor, device);
    }

    if (out_source) {
        *out_source = "unavailable";
    }
    return nullptr;
}

namespace sycl_ex = sycl::ext::oneapi::experimental;

struct ggml_backend_sycl_context {
    int         device;
    std::string name;
    // Device capability: does this device support SoA weight layout optimization?
    // This is NOT tensor state - it's a static capability of the GPU.
    // Tensor state is tracked per-tensor in ggml_tensor_extra_gpu::optimized_feature
    bool        supports_soa_reorder;
    ggml_sycl::UnifiedMatmulOrchestrator matmul_orchestrator;
    bool        layouts_finalized       = false;
    uint64_t    layouts_finalized_epoch = 0;
    uint64_t    exec_graph_hash         = 0;
    int         moe_layer_count         = 0;

    struct moe_ids_cache_entry {
        uint64_t             hash = 0;
        std::vector<int32_t> host_ids;
        void *               device_ids    = nullptr;
        size_t               device_bytes  = 0;
        void *               staging_ids   = nullptr;  // Pinned host staging for non-USM sources
        size_t               staging_bytes = 0;
    };

    std::unordered_map<const ggml_tensor *, moe_ids_cache_entry> moe_ids_cache;
    std::mutex                                                   graph_mutex;

    // L2 prefetch manager for TG optimization (owned by this context)
    // Uses custom deleter to allow incomplete type in header
    std::unique_ptr<ggml_sycl::L2PrefetchManager, ggml_sycl::L2PrefetchManagerDeleter> l2_prefetch_manager;

    // Persistent TG kernel instance (cached across graph_compute calls)
    // Lazy-initialized on first persistent dispatch to avoid allocation when unused
    std::unique_ptr<ggml_sycl::UnifiedKernel, ggml_sycl::UnifiedKernelDeleter> unified_kernel;

    queue_ptr qptrs[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS] = { { nullptr } };

    explicit ggml_backend_sycl_context(int device) :
        device(device),
        name(GGML_SYCL_NAME + std::to_string(device)),
        supports_soa_reorder(ggml_sycl_info().devices[device].supports_soa_reorder),
        matmul_orchestrator(*this) {}

    ~ggml_backend_sycl_context();

    // Non-movable: UnifiedMatmulOrchestrator has reference member, context created once per backend
    ggml_backend_sycl_context(ggml_backend_sycl_context&&) = delete;
    ggml_backend_sycl_context& operator=(ggml_backend_sycl_context&&) = delete;

    // Non-copyable (owns resources)
    ggml_backend_sycl_context(const ggml_backend_sycl_context&) = delete;
    ggml_backend_sycl_context& operator=(const ggml_backend_sycl_context&) = delete;

    queue_ptr stream(int device, int stream) {
        // In TP mode, ALWAYS use the shared-context queue so all devices can access
        // memory allocated in the shared context. Check every time since TP may be
        // enabled after queues were first accessed.
        sycl::queue * tp_queue = ggml_sycl_get_tp_queue(device);
        if (tp_queue != nullptr) {
            if (qptrs[device][stream] != tp_queue) {
                qptrs[device][stream] = tp_queue;
                GGML_SYCL_DEBUG("Using shared-context queue for device %d stream %d\n", device, stream);
            }
            return tp_queue;
        }
        // Non-TP mode: use default queue (cached)
        if (qptrs[device][stream] == nullptr) {
            qptrs[device][stream] = &(ggml_sycl_get_device(device).default_queue());
        }
        return qptrs[device][stream];
    }

    queue_ptr stream() { return stream(device, 0); }

#if GGML_SYCL_DNNL
    dnnl::engine make_engine(sycl::queue * q) {
        // Get the device associated with the queue
        sycl::device       dev = q->get_device();
        // Get the context associated with the queue
        sycl::context      ctx = q->get_context();
        const dnnl::engine eng = dnnl::sycl_interop::make_engine(dev, ctx);
        return eng;
    }

    std::unordered_map<sycl::queue *, dnnl::stream> stream_map;
    std::unordered_map<sycl::queue *, dnnl::engine> engine_map;
    std::mutex                                      dnnl_mutex;

    struct dnnl_scratchpad_entry {
        std::vector<std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>>> buffers;
        ggml_sycl_pool_alloc<uint8_t> *                             current = nullptr;
    };

    dnnl::stream stream_dnnl(int device, int _stream) {
        auto q = stream(device, _stream);
        return stream_dnnl(q);
    }

    dnnl::engine engine_dnnl_unlocked(sycl::queue * qptr) {
        auto it = engine_map.find(qptr);
        if (it == engine_map.end()) {
            auto eng         = make_engine(qptr);
            engine_map[qptr] = eng;
            return eng;
        }
        return it->second;
    }

    dnnl::engine engine_dnnl(sycl::queue * qptr) {
        std::lock_guard<std::mutex> lock(dnnl_mutex);
        return engine_dnnl_unlocked(qptr);
    }

    dnnl::stream stream_dnnl(sycl::queue * qptr) {
        std::lock_guard<std::mutex> lock(dnnl_mutex);
        auto                        it = stream_map.find(qptr);
        if (it == stream_map.end()) {
            auto eng         = engine_dnnl_unlocked(qptr);
            auto stream      = dnnl::sycl_interop::make_stream(eng, *qptr);
            stream_map[qptr] = stream;
            return stream;
        }
        return it->second;
    }

    dnnl::stream stream_dnnl() { return stream_dnnl(device, 0); }

    dnnl::memory get_scratchpad_mem(const dnnl::memory::desc & scratchpad_md,
                                    const dnnl::engine &       eng,
                                    const queue_ptr            q) {
        std::lock_guard<std::mutex> lock(dnnl_mutex);

        size_t scratchpad_size = scratchpad_md.get_size();
        if (scratchpad_size == 0) {
            return dnnl::memory();
        }
        auto & entry = scratchpad_map[q];

        if (entry.current == nullptr || scratchpad_size > entry.current->actual_size) {
            auto buffer = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(this->pool());
            buffer->alloc(scratchpad_size);
            entry.current = buffer.get();
            entry.buffers.push_back(std::move(buffer));
        }

        return dnnl::memory(scratchpad_md, eng, entry.current->get());
    }

    // Pre-allocate scratchpad pool to a given size
    // Used before graph recording to avoid realloc during recording
    void pre_allocate_scratchpad(size_t size, const queue_ptr q) {
        if (size == 0) {
            return;
        }

        std::lock_guard<std::mutex> lock(dnnl_mutex);

        auto & entry = scratchpad_map[q];
        if (entry.current == nullptr || size > entry.current->actual_size) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] Pre-allocating scratchpad pool: %zu bytes\n", size);
            auto buffer = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(this->pool());
            buffer->alloc(size);
            entry.current = buffer.get();
            entry.buffers.push_back(std::move(buffer));
        }
    }
#endif

    // pool
    std::unique_ptr<ggml_sycl_pool> pools[GGML_SYCL_MAX_DEVICES];
#if GGML_SYCL_DNNL
    std::unordered_map<sycl::queue *, dnnl_scratchpad_entry> scratchpad_map;
#endif

    std::unique_ptr<ggml_sycl_pool> host_pools[GGML_SYCL_MAX_DEVICES];

    static std::unique_ptr<ggml_sycl_pool> new_pool_for_device(queue_ptr qptr, int device);

    static std::unique_ptr<ggml_sycl_pool> new_pool_for_host(queue_ptr qptr, int device);

    ggml_sycl_pool & pool(int device) {
        if (pools[device] == nullptr) {
            pools[device] = new_pool_for_device(stream(device, 0), device);
        }
        return *pools[device];
    }

    ggml_sycl_pool & pool() { return pool(device); }

#ifdef GGML_SYCL_GRAPH
    std::unique_ptr<sycl_ex::command_graph<sycl_ex::graph_state::executable>> exec_graph = nullptr;
    int  exec_graph_n_nodes       = 0;      // Track graph size for cache invalidation
    bool exec_graph_is_decode     = false;  // Track which phase the cached graph was recorded for
    int  warmup_decode_n_nodes    = 0;      // Track which decode graph has been warmed up
    int  warmup_prompt_n_nodes    = 0;      // Track which prompt graph has been warmed up
    bool graphs_disabled          = false;  // Set when graph recording fails; disables graphs for this context
    bool moe_graphs_disabled      = false;  // Set when MoE preload fails; disables graphs for all splits
    bool moe_graphs_disabled_once = false;  // Set when we skip graphs for a single run

    // === Cached per-graph computations (reset when n_nodes changes) ===
    int  cached_persistent_n_nodes = -1;   // n_nodes when persistent check was cached
    bool cached_persistent_result  = false; // cached should_use_persistent_tg result
    bool cached_is_decode_phase    = false; // cached phase detection result

    uint64_t cached_graph_sig         = 0;  // cached graph signature hash
    int      cached_graph_sig_n_nodes = -1; // n_nodes when hash was cached

    // Pre-cached input tensor set for graph_refresh (populated during recording)
    std::vector<ggml_tensor *> cached_input_tensors;
    // Parallel vector: resolved device pointers for each cached input tensor.
    // When resolved_ptr == tensor->data, set_tensor_async already refreshed the data
    // and no additional copy is needed. When different, a direct async memcpy is done
    // from tensor->data to resolved_ptr, avoiding expensive get_pointer_type() driver calls.
    std::vector<void *> cached_input_dev_ptrs;
    bool input_tensors_cached = false;

    // Pre-allocated buffers for MoE graph recording
    // MUL_MAT_ID needs Q8_1 quantization buffers which cannot be allocated during graph recording
    struct moe_graph_buffers {
        // Q8_1 quantization buffers (one per MUL_MAT_ID in decode phase)
        std::vector<void *> q8_1_buffers;
        std::vector<size_t> q8_1_sizes;

        // Buffer usage tracking
        int  current_buffer_idx = 0;
        bool initialized        = false;

        // Max dimensions seen (for reallocation check)
        int64_t max_ne10      = 0;  // Max input dimension
        int64_t max_src1_rows = 0;  // Max (ne11 × ne12)

        void reset_usage() { current_buffer_idx = 0; }

        void * get_next_buffer(size_t required_size) {
            if (current_buffer_idx >= (int) q8_1_buffers.size()) {
                return nullptr;  // Fall back to pool alloc
            }
            if (required_size > q8_1_sizes[current_buffer_idx]) {
                return nullptr;  // Buffer too small
            }
            return q8_1_buffers[current_buffer_idx++];
        }

        void free_buffers(queue_ptr stream) {
            for (size_t i = 0; i < q8_1_buffers.size(); i++) {
                if (q8_1_buffers[i]) {
                    sycl::free(q8_1_buffers[i], *stream);
                }
            }
            q8_1_buffers.clear();
            q8_1_sizes.clear();
            initialized        = false;
            current_buffer_idx = 0;
            max_ne10           = 0;
            max_src1_rows      = 0;
        }
    } moe_buffers;

    // Pre-allocated buffers for SoA MMVQ graph recording
    // MUL_MAT with SoA reorder flag needs Q8_1 quantization buffers which cannot be
    // allocated from pool during graph recording (pointer would change on replay)
    struct mmvq_soa_buffers_t {
        // Q8_1 quantization buffers (one per SoA MUL_MAT in decode phase)
        std::vector<void *> src1_ddq_buffers;
        std::vector<size_t> src1_ddq_sizes;

        // Bulk allocation (single contiguous block for all sub-buffers)
        void * bulk_ptr  = nullptr;
        size_t bulk_size = 0;

        // Buffer usage tracking
        int  current_buffer_idx = 0;
        bool initialized        = false;

        // Max dimensions seen (for reallocation check)
        int64_t max_ne10  = 0;  // Max input dimension
        int64_t max_nrows = 0;  // Max rows

        void reset_usage() { current_buffer_idx = 0; }

        void * get_next_buffer(size_t required_size) {
            if (current_buffer_idx >= (int) src1_ddq_buffers.size()) {
                return nullptr;  // Fall back to pool alloc
            }
            if (required_size > src1_ddq_sizes[current_buffer_idx]) {
                return nullptr;  // Buffer too small
            }
            return src1_ddq_buffers[current_buffer_idx++];
        }

        void free_buffers(queue_ptr stream) {
            int device_id = ggml_sycl_get_device_id_from_queue(*stream);
            if (bulk_ptr) {
                // Bulk allocation: free the single block
                ggml_sycl::unified_cache_sub_runtime_bytes(device_id, bulk_size);
                sycl::free(bulk_ptr, *stream);
                bulk_ptr  = nullptr;
                bulk_size = 0;
            } else {
                // Legacy per-buffer allocation
                for (size_t i = 0; i < src1_ddq_buffers.size(); i++) {
                    if (src1_ddq_buffers[i]) {
                        ggml_sycl::unified_cache_sub_runtime_bytes(device_id, src1_ddq_sizes[i]);
                        sycl::free(src1_ddq_buffers[i], *stream);
                    }
                }
            }
            src1_ddq_buffers.clear();
            src1_ddq_sizes.clear();
            initialized        = false;
            current_buffer_idx = 0;
            max_ne10           = 0;
            max_nrows          = 0;
        }
    } mmvq_soa_buffers;

    // Pre-allocated buffers for XMX MoE graph recording
    // XMX MoE needs various temporary buffers that can't be allocated during graph recording
    struct xmx_moe_buffers_t {
        // Token sorting buffers (persistent across graph executions)
        sycl::half * tokens_f16_input = nullptr;  // F32->F16 converted tokens [n_input_rows * in_dim]
        sycl::half * tokens_sorted    = nullptr;  // Sorted tokens [total_pairs * in_dim]
        void *       token_map        = nullptr;  // Token mapping for scatter-back [total_pairs] (MoETokenMapping*)
        int32_t *    expert_counts    = nullptr;  // Per-expert token counts [n_experts]
        int32_t *    expert_offsets   = nullptr;  // Prefix sum offsets [n_experts + 1]
        int32_t *    expert_write_pos = nullptr;  // Atomic write positions [n_experts]
        sycl::half * sorted_output    = nullptr;  // XMX output [total_pairs * out_dim]

        // Q8 quantization buffers
        int8_t *     q_tokens     = nullptr;  // Quantized tokens [total_pairs * in_dim]
        sycl::half * token_scales = nullptr;  // Token scales [total_pairs * (in_dim / QK8_0)]

        // Expert scale buffer for AoS Q8_0
        sycl::half * expert_scale_buf = nullptr;  // [out_dim * (in_dim / QK8_0)]

        // Sorted token IDs for fused path
        int32_t * sorted_token_ids = nullptr;  // [total_pairs]

        // Tile mapping buffers for fused XMX MoE kernel
        // Pre-allocated for graph recording (fixed addresses required)
        int32_t * expert_tile_offsets = nullptr;  // [MAX_EXPERTS + 1] prefix sum of tiles per expert
        int32_t * total_tiles         = nullptr;  // [1] scalar: total work tiles across all experts

        // Maximum supported experts for pre-allocation
        static constexpr int MAX_EXPERTS = 64;

        // Buffer dimensions (for reallocation check)
        int64_t max_total_pairs  = 0;
        int64_t max_in_dim       = 0;
        int64_t max_out_dim      = 0;
        int64_t max_n_experts    = 0;
        int64_t max_n_input_rows = 0;

        bool initialized = false;

        void reset_usage() {
            // No per-call reset needed - buffers are persistent
        }

        size_t bytes_tokens_f16_input() const {
            return static_cast<size_t>(max_n_input_rows) * static_cast<size_t>(max_in_dim) * sizeof(sycl::half);
        }
        size_t bytes_tokens_sorted() const {
            return static_cast<size_t>(max_total_pairs) * static_cast<size_t>(max_in_dim) * sizeof(sycl::half);
        }
        size_t bytes_token_map() const { return static_cast<size_t>(max_total_pairs) * kMoETokenMappingBytes; }
        size_t bytes_expert_counts() const { return static_cast<size_t>(max_n_experts) * sizeof(int32_t); }
        size_t bytes_expert_offsets() const { return static_cast<size_t>(max_n_experts + 1) * sizeof(int32_t); }
        size_t bytes_expert_write_pos() const { return static_cast<size_t>(max_n_experts) * sizeof(int32_t); }
        size_t bytes_sorted_output() const {
            return static_cast<size_t>(max_total_pairs) * static_cast<size_t>(max_out_dim) * sizeof(sycl::half);
        }
        size_t bytes_q_tokens() const {
            return static_cast<size_t>(max_total_pairs) * static_cast<size_t>(max_in_dim) * sizeof(int8_t);
        }
        size_t bytes_token_scales() const {
            const size_t blocks = static_cast<size_t>(max_in_dim / QK8_0);
            return static_cast<size_t>(max_total_pairs) * blocks * sizeof(sycl::half);
        }
        size_t bytes_expert_scale_buf() const {
            const size_t blocks = static_cast<size_t>(max_in_dim / QK8_0);
            return static_cast<size_t>(max_out_dim) * blocks * sizeof(sycl::half);
        }
        size_t bytes_sorted_token_ids() const { return static_cast<size_t>(max_total_pairs) * sizeof(int32_t); }

        // Allocate tile mapping buffers for fused XMX MoE kernel
        // Called once during initialization - enables graph recording with fixed addresses
        void allocate_tile_mapping(sycl::queue & q) {
            if (!expert_tile_offsets) {
                ggml_sycl::unified_cache_add_runtime_bytes(ggml_sycl_get_device_id_from_queue(q),
                                                           (MAX_EXPERTS + 1) * sizeof(int32_t));
                expert_tile_offsets = ggml_sycl_malloc_device_t<int32_t>(MAX_EXPERTS + 1, q, "moe_tile_mapping");
                if (!expert_tile_offsets) {
                    ggml_sycl::unified_cache_sub_runtime_bytes(ggml_sycl_get_device_id_from_queue(q),
                                                               (MAX_EXPERTS + 1) * sizeof(int32_t));
                }
                ggml_sycl::unified_cache_add_runtime_bytes(ggml_sycl_get_device_id_from_queue(q), sizeof(int32_t));
                total_tiles         = ggml_sycl_malloc_device_t<int32_t>(1, q, "moe_tile_mapping");
                if (!total_tiles) {
                    ggml_sycl::unified_cache_sub_runtime_bytes(ggml_sycl_get_device_id_from_queue(q), sizeof(int32_t));
                }
            }
        }

        // Free tile mapping buffers
        void free_tile_mapping(sycl::queue & q) {
            if (expert_tile_offsets) {
                ggml_sycl::unified_cache_sub_runtime_bytes(ggml_sycl_get_device_id_from_queue(q),
                                                           (MAX_EXPERTS + 1) * sizeof(int32_t));
                sycl::free(expert_tile_offsets, q);
                expert_tile_offsets = nullptr;
            }
            if (total_tiles) {
                ggml_sycl::unified_cache_sub_runtime_bytes(ggml_sycl_get_device_id_from_queue(q), sizeof(int32_t));
                sycl::free(total_tiles, q);
                total_tiles = nullptr;
            }
        }

        void free_buffers(queue_ptr stream) {
            const int device_id = ggml_sycl_get_device_id_from_queue(*stream);
            if (tokens_f16_input) {
                ggml_sycl::unified_cache_sub_runtime_bytes(device_id, bytes_tokens_f16_input());
                sycl::free(tokens_f16_input, *stream);
            }
            if (tokens_sorted) {
                ggml_sycl::unified_cache_sub_runtime_bytes(device_id, bytes_tokens_sorted());
                sycl::free(tokens_sorted, *stream);
            }
            if (token_map) {
                ggml_sycl::unified_cache_sub_runtime_bytes(device_id, bytes_token_map());
                sycl::free(static_cast<void *>(token_map), *stream);
            }
            if (expert_counts) {
                ggml_sycl::unified_cache_sub_runtime_bytes(device_id, bytes_expert_counts());
                sycl::free(expert_counts, *stream);
            }
            if (expert_offsets) {
                ggml_sycl::unified_cache_sub_runtime_bytes(device_id, bytes_expert_offsets());
                sycl::free(expert_offsets, *stream);
            }
            if (expert_write_pos) {
                ggml_sycl::unified_cache_sub_runtime_bytes(device_id, bytes_expert_write_pos());
                sycl::free(expert_write_pos, *stream);
            }
            if (sorted_output) {
                ggml_sycl::unified_cache_sub_runtime_bytes(device_id, bytes_sorted_output());
                sycl::free(sorted_output, *stream);
            }
            if (q_tokens) {
                ggml_sycl::unified_cache_sub_runtime_bytes(device_id, bytes_q_tokens());
                sycl::free(q_tokens, *stream);
            }
            if (token_scales) {
                ggml_sycl::unified_cache_sub_runtime_bytes(device_id, bytes_token_scales());
                sycl::free(token_scales, *stream);
            }
            if (expert_scale_buf) {
                ggml_sycl::unified_cache_sub_runtime_bytes(device_id, bytes_expert_scale_buf());
                sycl::free(expert_scale_buf, *stream);
            }
            if (sorted_token_ids) {
                ggml_sycl::unified_cache_sub_runtime_bytes(device_id, bytes_sorted_token_ids());
                sycl::free(sorted_token_ids, *stream);
            }
            if (expert_tile_offsets) {
                ggml_sycl::unified_cache_sub_runtime_bytes(device_id, (MAX_EXPERTS + 1) * sizeof(int32_t));
                sycl::free(expert_tile_offsets, *stream);
            }
            if (total_tiles) {
                ggml_sycl::unified_cache_sub_runtime_bytes(device_id, sizeof(int32_t));
                sycl::free(total_tiles, *stream);
            }

            tokens_f16_input    = nullptr;
            tokens_sorted       = nullptr;
            token_map           = nullptr;
            expert_counts       = nullptr;
            expert_offsets      = nullptr;
            expert_write_pos    = nullptr;
            sorted_output       = nullptr;
            q_tokens            = nullptr;
            token_scales        = nullptr;
            expert_scale_buf    = nullptr;
            sorted_token_ids    = nullptr;
            expert_tile_offsets = nullptr;
            total_tiles         = nullptr;

            max_total_pairs  = 0;
            max_in_dim       = 0;
            max_out_dim      = 0;
            max_n_experts    = 0;
            max_n_input_rows = 0;
            initialized      = false;
        }
    } xmx_moe_buffers;

    // Q8_1 quantization cache for MoE: avoids re-quantizing same input across gate/up/down
    // In MoE layers, the same input is used for all projections - caching saves 3x quantization
    struct moe_quant_cache {
        void *       cached_q8_1 = nullptr;  // Cached Q8_1 quantized data
        const void * cached_src  = nullptr;  // Key: source pointer that was quantized
        int64_t      cached_ne10 = 0;        // Input row width
        int64_t      cached_rows = 0;        // Number of rows quantized
        size_t       cached_size = 0;        // Buffer size
        bool         valid       = false;    // Cache entry is valid

        void invalidate() {
            cached_src  = nullptr;
            cached_ne10 = 0;
            cached_rows = 0;
            valid       = false;
            // Note: don't free cached_q8_1 - it's pool memory that gets reused
        }

        // Check if cache matches current request
        bool matches(const void * src, int64_t ne10, int64_t rows) const {
            return valid && cached_src == src && cached_ne10 == ne10 && cached_rows == rows;
        }
    } moe_q8_cache;
#endif

    // Barrier event for cross-ubatch synchronization
    // This provides lighter-weight sync than full queue wait
    std::optional<sycl::event> barrier_event;
    bool                       has_pending_barrier = false;
    std::optional<sycl::event> last_graph_event;

    // Persistent staging buffer for get_tensor_async readback.
    // Avoids per-call USM host alloc/free overhead for logits readback (~128KB/token).
    // Tracked via host memory tracking (ggml_sycl_malloc_host_tracked_bytes).
    void * readback_staging      = nullptr;
    size_t readback_staging_size = 0;

    // Reusable device buffer for BLAS fallback (MXFP4 -> F16 dequantization).
    // Allocated lazily on first BLAS fallback, registered with unified cache budget.
    void * staging_buffer_        = nullptr;
    size_t staging_buffer_size_   = 0;
    int    staging_buffer_device_ = -1;

    // Get or allocate staging buffer for BLAS fallback.
    // Returns {pointer, size} or {nullptr, 0} if allocation fails.
    std::pair<void *, size_t> get_staging_buffer(size_t needed_bytes, sycl::queue & queue);
    // Free staging buffer and release budget reservation.
    void free_staging_buffer();

    ggml_sycl_pool & host_pool(int device) {
        if (host_pools[device] == nullptr) {
            host_pools[device] = new_pool_for_host(stream(device, 0), device);
        }
        return *host_pools[device];
    }

    ggml_sycl_pool & host_pool() { return host_pool(device); }

    // Flag to disable graphs when weight streaming is active
    bool                                                         weight_streaming_graphs_disabled = false;
    // Track graph-pinned cache entries (cache_id + layout) for unpinning.
    std::vector<std::pair<ggml_sycl_cache_id, ggml_layout_mode>> graph_pinned_entries;

    struct fa_graph_ptr_snapshot {
        const void * q           = nullptr;
        const void * k           = nullptr;
        const void * v           = nullptr;
        const void * dst         = nullptr;
        const void * mask        = nullptr;
        const void * sinks       = nullptr;
        const void * block_table = nullptr;
        const void * seq_lens    = nullptr;
    };

    std::vector<fa_graph_ptr_snapshot> fa_graph_ptrs;
    bool                               fa_graph_ptrs_valid     = false;
    bool                               fa_graph_ptrs_recording = false;

    // KV offload manager for long context support (initialized lazily when enabled)
    std::unique_ptr<ggml_sycl::kv_offload_manager> kv_offload_mgr_;

    // Initialize KV offload manager with given configuration
    void init_kv_offload(const ggml_sycl::kv_offload_config & config);

    // Check if KV offload is initialized
    bool has_kv_offload() const { return kv_offload_mgr_ != nullptr; }

    // Get the KV offload manager (must be initialized first)
    ggml_sycl::kv_offload_manager * kv_offload() { return kv_offload_mgr_.get(); }
};

// GGML_OP_ALL_REDUCE_SUM handler for SYCL backend
// For single-device execution, this is a copy operation
// For multi-device TP, this will perform actual all-reduce across devices
void ggml_sycl_all_reduce_sum(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// Async FFN computation for Tensor Parallelism pipelining
// Launches FFN computation on device 1 asynchronously (returns immediately)
void ggml_sycl_tp_launch_async_ffn(ggml_backend_sycl_context & ctx,
                                   int                         layer,
                                   const float *               input_dev1,  // Input on device 1
                                   int64_t                     K_full,      // Full model dimension
                                   int64_t                     batch,       // Batch size
                                   const ffn_weight_refs &     weights      // Weight tensor references
);

// Wait for and retrieve async FFN result (blocks until done)
float * ggml_sycl_tp_wait_async_ffn(int layer, int64_t * out_ne0, int64_t * out_ne1, size_t * out_size);

// Async attention computation for Tensor Parallelism pipelining
void ggml_sycl_tp_launch_async_attn(ggml_backend_sycl_context & ctx,
                                    int                         layer,
                                    const float *               input_dev1,  // Input on device 1
                                    int64_t                     K_full,      // Full model dimension
                                    int64_t                     batch,       // Batch size
                                    const attn_weight_refs &    weights      // Weight tensor references
);

// Wait for and retrieve async attention result
float * ggml_sycl_tp_wait_async_attn(int layer, int64_t * out_ne0, int64_t * out_ne1, size_t * out_size);

// common device functions

static __dpct_inline__ float warp_reduce_sum(float x, const sycl::nd_item<3> & item_ct1) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), x, mask);
    }
    return x;
}

static __dpct_inline__ sycl::float2 warp_reduce_sum(sycl::float2 a, const sycl::nd_item<3> & item_ct1) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        a.x() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.x(), mask);
        a.y() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.y(), mask);
    }
    return a;
}

template <int width = WARP_SIZE> static __dpct_inline__ int warp_reduce_sum(int x) {
    return sycl::reduce_over_group(sycl::ext::oneapi::this_work_item::get_sub_group(), x, sycl::plus<>());
}

template <int width = WARP_SIZE> static __dpct_inline__ float warp_reduce_sum(float x) {
    // Use optimized subgroup reduce for full WARP_SIZE (common case)
    if constexpr (width == WARP_SIZE) {
        return sycl::reduce_over_group(sycl::ext::oneapi::this_work_item::get_sub_group(), x, sycl::plus<float>());
    } else {
        // Fallback for partial subgroup reductions
#pragma unroll
        for (int offset = width / 2; offset > 0; offset >>= 1) {
            x += dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), x, offset, width);
        }
        return x;
    }
}

template <int width = WARP_SIZE> static __dpct_inline__ sycl::float2 warp_reduce_sum(sycl::float2 a) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        a.x() +=
            dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), a.x(), offset, width);
        a.y() +=
            dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), a.y(), offset, width);
    }
    return a;
}

template <int width = WARP_SIZE> static __dpct_inline__ sycl::half2 warp_reduce_sum(sycl::half2 a) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        a = a + dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), a, offset, width);
    }
    return a;
}

static constexpr int ggml_sycl_get_physical_warp_size() {
    // todo: for old iGPU + dGPU case, need to be changed.
    return WARP_SIZE;
}

template <int width = WARP_SIZE> static __dpct_inline__ float warp_reduce_max(float x) {
    // Use optimized subgroup reduce for full WARP_SIZE (common case)
    if constexpr (width == WARP_SIZE) {
        return sycl::reduce_over_group(sycl::ext::oneapi::this_work_item::get_sub_group(), x, sycl::maximum<float>());
    } else {
        // Fallback for partial subgroup reductions
#pragma unroll
        for (int offset = width / 2; offset > 0; offset >>= 1) {
            x = sycl::fmax(x, dpct::permute_sub_group_by_xor(sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                                                             offset, width));
        }
        return x;
    }
}

static __dpct_inline__ float warp_reduce_max(float x, const sycl::nd_item<3> & item_ct1) {
    // Use optimized subgroup reduce
    return sycl::reduce_over_group(item_ct1.get_sub_group(), x, sycl::maximum<float>());
}

/* Helper for Computing the linear offset of a ggml_tensor given
per-dimension sizes, strides, and indices */
template <int N>
__dpct_inline__ size_t calculate_offset(const std::array<int, N> & strides, const std::array<int, N> & indices) {
    size_t offset = 0;
#pragma unroll
    for (int i = 0; i < N; i++) {
        auto index_i = indices[i];
        offset += strides[i] * index_i;
    }
    return offset;
}

// Helper for vec loading aligned data
template <typename Tp, int n> inline sycl::vec<Tp, n> vec_aligned_load(const Tp * aligned_ptr) {
    return *reinterpret_cast<const sycl::vec<Tp, n> *>(aligned_ptr);
}

// Helper for accessing pointers with no warnings
template <typename Tp, int dim> static __dpct_inline__ Tp * get_pointer(sycl::local_accessor<Tp, dim> acc) {
    return acc.template get_multi_ptr<sycl::access::decorated::no>().get();
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_num, int64_t block_size);

constexpr size_t ceil_div(const size_t m, const size_t n) {
    return (m + n - 1) / n;
}

bool gpu_has_xmx(sycl::device & dev);

// XMXCapabilities struct and query_xmx_capabilities() declaration
// moved to line ~487 so sycl_device_info can include xmx_caps as a member

template <int N, class T> std::string debug_get_array_str(const std::string & prefix, const T array[N]) {
    if (LIKELY(!g_ggml_sycl_debug)) {
        return "";
    }
    std::stringstream ss;
    ss << prefix << "=[";
    for (std::size_t i = 0; i < N - 1; ++i) {
        ss << array[i] << ", ";
    }
    if constexpr (N > 0) {
        ss << array[N - 1];
    }
    ss << "]";
    return ss.str();
}

inline std::string debug_get_tensor_str(const std::string & prefix,
                                        const ggml_tensor * tensor,
                                        const std::string & suffix = "") {
    std::stringstream ss;
    if (LIKELY(!g_ggml_sycl_debug)) {
        return ss.str();
    }
    ss << prefix.c_str() << "=";
    if (tensor) {
        ss << "'" << tensor->name << "':type=" << ggml_type_name(tensor->type);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";ne", tensor->ne);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";nb", tensor->nb);

        if (!ggml_is_contiguous(tensor)) {
            ss << ";strided";
        }
        if (ggml_is_permuted(tensor)) {
            ss << ";permuted";
        }
    } else {
        ss << "nullptr";
    }
    ss << suffix;
    return ss.str();
}

inline void debug_check_tensor_ptr(const ggml_tensor * tensor, const char * tag) {
    if (LIKELY(!g_ggml_sycl_debug) || tensor == nullptr || tensor->buffer == nullptr || tensor->data == nullptr) {
        return;
    }

    void * base = ggml_backend_buffer_get_base(tensor->buffer);
    size_t size = ggml_backend_buffer_get_size(tensor->buffer);
    if (base == nullptr || size == 0) {
        return;
    }

    const size_t alloc    = ggml_backend_buffer_get_alloc_size(tensor->buffer, tensor);
    char *       begin    = static_cast<char *>(base);
    char *       end      = begin + size;
    char *       data     = static_cast<char *>(tensor->data);
    const bool   in_range = (data >= begin) && (data + alloc <= end);
    if (!in_range) {
        GGML_LOG_ERROR("[SYCL][PTR] %s tensor=%s data=%p alloc=%zu base=%p size=%zu end=%p\n", tag, tensor->name,
                       tensor->data, alloc, base, size, end);
    } else if (g_ggml_sycl_debug >= 2) {
        GGML_SYCL_DEBUG("[SYCL][PTR] %s tensor=%s data=%p alloc=%zu base=%p size=%zu\n", tag, tensor->name,
                        tensor->data, alloc, base, size);
    }
}

// Use scope_op_debug_print to log operations coming from running a model
struct scope_op_debug_print {
    // Use string_views to avoid the cost of creating a string and concatenating them
    // string_views must be alive for as long as the object is alive
    // scope_op_debug_print are used with string literals in practice which are stored in constant space so always accessible
    scope_op_debug_print(const std::string_view & func,
                         const std::string_view & func_suffix,
                         const ggml_tensor *      dst,
                         std::size_t              num_src,
                         const std::string_view & suffix = "") :
        func(func),
        func_suffix(func_suffix) {
        if (LIKELY(!g_ggml_sycl_debug)) {
            return;
        }
        GGML_SYCL_DEBUG("[SYCL][OP] call %s%s:", func.data(), func_suffix.data());
        GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" dst", dst).c_str());
        debug_check_tensor_ptr(dst, "dst");
        if (dst) {
            for (std::size_t i = 0; i < num_src; ++i) {
                GGML_SYCL_DEBUG("%s", debug_get_tensor_str("\tsrc" + std::to_string(i), dst->src[i]).c_str());
                debug_check_tensor_ptr(dst->src[i], ("src" + std::to_string(i)).c_str());
            }
        }
        GGML_SYCL_DEBUG("%s\n", suffix.data());
    }

    scope_op_debug_print(const std::string_view & func,
                         const ggml_tensor *      dst,
                         std::size_t              num_src,
                         const std::string_view & suffix = "") :
        scope_op_debug_print(func, "", dst, num_src, suffix) {}

    ~scope_op_debug_print() { GGML_SYCL_DEBUG("[SYCL][OP] call %s%s done\n", func.data(), func_suffix.data()); }

  private:
    std::string_view func;
    std::string_view func_suffix;
};

static __dpct_inline__ float get_alibi_slope(const float    max_bias,
                                             const uint32_t h,
                                             const uint32_t n_head_log2,
                                             const float    m0,
                                             const float    m1) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2 * (h - n_head_log2) + 1;

    return dpct::pow(base, exph);
}

static const sycl::uint3 init_fastdiv_values(uint32_t d) {
    GGML_ASSERT(d != 0);

    uint32_t L = 0;
    while (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    return sycl::uint3(mp, L, d);
}

static __dpct_inline__ uint32_t fastdiv(uint32_t n, const sycl::uint3 fastdiv_values) {
    const uint32_t hi = sycl::mul_hi<unsigned>(n, fastdiv_values.x());
    return (hi + n) >> fastdiv_values.y();
}

static __dpct_inline__ sycl::uint2 fast_div_modulo(uint32_t n, const sycl::uint3 fastdiv_values) {
    const uint32_t div_val = fastdiv(n, fastdiv_values);
    const uint32_t mod_val = n - div_val * fastdiv_values.z();
    return sycl::uint2(div_val, mod_val);
}

static __dpct_inline__ int ggml_sycl_dp4a(const int a, const int b, int c) {
    return dpct::dp4a(a, b, c);
}

static __dpct_inline__ float ggml_sycl_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;
    if (x == 0) {
        bits = 0x00400000;
    } else {
        bits = (uint32_t) x << 23;
    }

    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

#endif  // GGML_SYCL_COMMON_HPP
