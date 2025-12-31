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

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "dpct/helper.hpp"
#include "ggml-sycl.h"
#include "presets.hpp"
#include "sycl_hw.hpp"
#include "unified-cache.hpp"
#include "kv-offload.hpp"


#if GGML_SYCL_DNNL
#include "dnnl.hpp"
#include "dnnl_sycl.hpp"
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

void* ggml_sycl_host_malloc(size_t size);
void ggml_sycl_host_free(void* ptr);

// Get shared-context queue for TP mode (returns nullptr if not in TP mode)
sycl::queue * ggml_sycl_get_tp_queue(int device);

// Get shared context for TP mode (returns nullptr if not in TP mode)
sycl::context * ggml_sycl_get_tp_context();

// TP staging cache: stages mmap'd data to USM memory for shared-context access
// Per-device staging: each device gets its own device-local copy
void * ggml_sycl_get_staged_ptr_device(const void * src, size_t size, int device);
void * ggml_sycl_get_staged_ptr(const void * src, size_t size);  // Legacy: returns device 0's pointer
void ggml_sycl_clear_staging_cache();

// Internal getters for seq_ids host pointers (set by llama layer, used by fattn)
const int32_t * ggml_sycl_get_seq_ids_host_q(size_t * count);
const int32_t * ggml_sycl_get_seq_ids_host_kv(size_t * count);


extern int g_ggml_sycl_debug;
extern int g_ggml_sycl_tp_debug;  // Tensor Parallelism debug output
extern int g_ggml_sycl_disable_optimize;
extern int g_ggml_sycl_prioritize_dmmv;

// Track when SYCL graph recording is active
extern thread_local bool g_ggml_sycl_graph_recording;

#if defined(__clang__) && __has_builtin(__builtin_expect)
// Hint the optimizer to pipeline the more likely following instruction in branches
#    define LIKELY(expr)   __builtin_expect(expr, true)
#    define UNLIKELY(expr) __builtin_expect(expr, false)
#else
#    define LIKELY(expr)   (expr)
#    define UNLIKELY(expr) (expr)
#endif

#define GGML_SYCL_DEBUG(...)              \
    do {                                  \
        if (UNLIKELY(g_ggml_sycl_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } while (0)

// Tensor Parallelism debug output - controlled by GGML_SYCL_TP_DEBUG env var
#define GGML_SYCL_TP_DEBUG(...)              \
    do {                                     \
        if (UNLIKELY(g_ggml_sycl_tp_debug))  \
            fprintf(stderr, __VA_ARGS__);    \
    } while (0)

// Kernel trace - compile-time toggle for tracing kernel execution flow
// Enable by uncommenting the define below or adding -DGGML_SYCL_KERNEL_TRACE=1
// #define GGML_SYCL_KERNEL_TRACE 1

#ifdef GGML_SYCL_KERNEL_TRACE
#define GGML_SYCL_KTRACE(kernel_name, ...)                                      \
    do {                                                                        \
        fprintf(stderr, "[KTRACE] %s", kernel_name);                            \
        fprintf(stderr, __VA_ARGS__);                                           \
        fprintf(stderr, "\n");                                                  \
        fflush(stderr);                                                         \
    } while (0)
#else
#define GGML_SYCL_KTRACE(kernel_name, ...) ((void)0)
#endif

#define CHECK_TRY_ERROR(expr)                                            \
  [&]() {                                                                \
    try {                                                                \
      expr;                                                              \
      return dpct::success;                                              \
    } catch (std::exception const& e) {                                  \
      std::cerr << e.what() << "\nException caught at file:" << __FILE__ \
                << ", line:" << __LINE__ << ", func:" << __func__        \
                << std::endl;                                            \
      return dpct::default_error;                                        \
    }                                                                    \
  }()


#define __SYCL_ARCH__ DPCT_COMPATIBILITY_TEMP
#define VER_4VEC 610 // todo for hardward optimize.
#define VER_GEN9 700 // todo for hardward optimize.
#define VER_GEN12 1000000 // todo for hardward optimize.
#define VER_GEN13 (VER_GEN12 + 1030) // todo for hardward optimize.
#define VER_XE2 2000

#define GGML_SYCL_MAX_NODES 8192 // TODO: adapt to hardwares

// define for XMX in Intel GPU
// TODO: currently, it's not used for XMX really.
#if !defined(GGML_SYCL_FORCE_MMQ)
    #define SYCL_USE_XMX
#endif

// max batch size to use MMQ kernels when tensor cores are available
// MMQ ESIMD is optimal for small batches. Dequantize path is 2-3x faster for large batches.
#define MMQ_MAX_BATCH_SIZE 32

// dmmv = dequantize_mul_mat_vec
#ifndef GGML_SYCL_DMMV_X
#define GGML_SYCL_DMMV_X 32
#endif
#ifndef GGML_SYCL_MMV_Y
#define GGML_SYCL_MMV_Y 1
#endif

typedef sycl::queue *queue_ptr;

enum ggml_sycl_backend_gpu_mode {
  SYCL_UNSET_GPU_MODE = -1,
  SYCL_SINGLE_GPU_MODE = 0,
  SYCL_MUL_GPU_MODE
};

static_assert(sizeof(sycl::half) == sizeof(ggml_fp16_t), "wrong fp16 size");

// SYCL-compatible E8M0 to FP32 conversion (halved for MXFP4)
// E8M0 is an 8-bit exponent-only format used in MX (Microscaling) formats
static __dpct_inline__ float sycl_e8m0_to_fp32_half(uint8_t e) {
    // For e < 2: use precomputed denormal patterns
    // For e >= 2: exponent - 1 gives FP32 exponent (halving = divide by 2)
    uint32_t bits;
    if (e < 2) {
        // Denormal handling: e=0 -> 0.0, e=1 -> very small denormal
        static const uint32_t denorm_table[2] = {0x00000000, 0x33800000};
        bits = denorm_table[e];
    } else {
        // Normal case: FP32 exponent = e - 1 (bias 127, so -1 gives halving)
        bits = ((uint32_t)(e - 1)) << 23;
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void crash() {
  int* ptr = NULL;
  *ptr = 0;
}

[[noreturn]] static void ggml_sycl_error(
    const char* stmt,
    const char* func,
    const char* file,
    const int line,
    const char* msg) {
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
#define GGML_SYCL_ASSUME(x) __builtin_assume(x)
#else
#define GGML_SYCL_ASSUME(x)
#endif // DPCT_COMPAT_RT_VERSION >= 11100

#ifdef GGML_SYCL_F16
typedef sycl::half dfloat; // dequantize float
typedef sycl::half2 dfloat2;
#else
typedef float dfloat; // dequantize float
typedef sycl::float2 dfloat2;
#endif // GGML_SYCL_F16

#define MMVQ_MAX_BATCH_SIZE  8

// Multi-row MMVQ kernel configuration
// Processes multiple output rows per work-group, sharing Y-vector in SLM
// This amortizes Y-vector loading across rows, reducing memory bandwidth
#define MMVQ_NROWS_PER_WG 4        // Rows per work-group (tune: 4, 8, or 16)

// SLM sizes for Y-vector caching in multi-row MMVQ
// Q8_1 block: 32 bytes quants (int8[32]) + 4 bytes ds (half2) = 36 bytes
// For Mistral 7B: ncols=4096, blocks_per_row = 4096/32 = 128 blocks
// SLM needed: 128 * 36 = 4.5KB per Y-vector (fits easily in 128KB SLM)
// We store qs as ints for aligned access: 8 ints per block (32 bytes)
// Plus ds as half2: 4 bytes per block
// Add +1 padding to avoid bank conflicts on 32-bank SLM
constexpr int MMVQ_SLM_Y_QS_STRIDE = 9;   // 8 ints + 1 padding to avoid bank conflicts
constexpr int MMVQ_SLM_MAX_BLOCKS = 256;  // Max blocks per row (ncols=8192, qk=32)
constexpr int MMVQ_SLM_Y_QS_SIZE = MMVQ_SLM_MAX_BLOCKS * MMVQ_SLM_Y_QS_STRIDE;  // ~9KB ints
constexpr int MMVQ_SLM_Y_DS_SIZE = MMVQ_SLM_MAX_BLOCKS + 1;  // half2 array + padding

// Warp-coalesced MMVQ configuration
// Reorganizes weight data so consecutive threads load consecutive bytes
// This achieves 100% cache line utilization (vs 50% with strided access)
constexpr int MMVQ_COALESCED_TILE_BLOCKS = 16;  // Blocks per warp tile (must match WARP_SIZE/2)
constexpr int MMVQ_COALESCED_TILE_BYTES_Q4_0 = MMVQ_COALESCED_TILE_BLOCKS * 16;  // 256 bytes quants per tile (Q4_0: 16 bytes/block)
constexpr int MMVQ_COALESCED_TILE_BYTES_Q8_0 = MMVQ_COALESCED_TILE_BLOCKS * 32;  // 512 bytes quants per tile (Q8_0: 32 bytes/block)
constexpr int MMVQ_COALESCED_TILE_BYTES_MXFP4 = MMVQ_COALESCED_TILE_BLOCKS * 16; // 256 bytes quants per tile (MXFP4: 16 bytes/block)
// Legacy alias for Q4_0
constexpr int MMVQ_COALESCED_TILE_BYTES = MMVQ_COALESCED_TILE_BYTES_Q4_0;

static int g_all_sycl_device_count = -1;
static bool g_ggml_backend_sycl_buffer_type_initialized = false;

static ggml_sycl_backend_gpu_mode g_ggml_sycl_backend_gpu_mode =
    SYCL_UNSET_GPU_MODE;

static void* g_scratch_buffer = nullptr;
static size_t g_scratch_size = 0; // disabled by default
static size_t g_scratch_offset = 0;

[[noreturn]] static inline void bad_arch(const sycl::stream& stream_ct1) {
  stream_ct1 << "ERROR: ggml-sycl was compiled without support for the "
                "current GPU architecture.\n";
  // __trap();
  std::exit(1);

  (void)bad_arch; // suppress unused function warning
}

int get_current_device_id();

inline dpct::err0 ggml_sycl_set_device(const int device) try {
  int current_device_id;
  SYCL_CHECK(CHECK_TRY_ERROR(current_device_id = get_current_device_id()));

  // GGML_SYCL_DEBUG("ggml_sycl_set_device device_id=%d,
  // current_device_id=%d\n", device, current_device);
  if (device == current_device_id) {
    return 0;
  }

  return CHECK_TRY_ERROR(dpct::select_device(device));
} catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  crash();
  std::exit(1);
}

//////////////////////
enum class reorder_mode : uint8_t {
    NONE = 0,       // Original AoS layout (Array of Structures)
    SOA = 1,        // SoA layout: all qs bytes contiguous, then all d values
    COALESCED = 2,  // Tile-based layout for better cache line utilization (requires SOA first)
};

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
// For MoE cached experts (data already transformed by cache_moe_expert_with_reorder):
//   Use the explicit constructor: optimize_feature(reorder_mode::SOA)
//   This creates a new instance with the mode set - no mutation after creation.
// =============================================================================
struct ggml_tensor;
bool reorder_tensor_to_soa(const ggml_tensor* tensor, dpct::queue_ptr stream, const char* caller);
bool convert_tensor_to_coalesced(const ggml_tensor* tensor, dpct::queue_ptr stream, const char* caller);

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
    friend bool reorder_tensor_to_soa(const ggml_tensor*, dpct::queue_ptr, const char*);
    friend bool convert_tensor_to_coalesced(const ggml_tensor*, dpct::queue_ptr, const char*);

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
    optimize_feature* data_owner_ = nullptr;

    // PRIVATE: Only callable by friend functions!
    // This ONLY sets the flag - does NOT transform data.
    // The friend functions MUST transform data BEFORE calling this.
    void set_reorder_mode_(reorder_mode new_mode, const char* tensor_name, const char* caller) {
        if (new_mode == reorder_) {
            return;  // No change
        }

        bool valid = false;
        if (reorder_ == reorder_mode::NONE && new_mode == reorder_mode::SOA) {
            valid = true;  // NONE → SOA
        } else if (reorder_ == reorder_mode::SOA && new_mode == reorder_mode::COALESCED) {
            valid = true;  // SOA → COALESCED
        }

        if (!valid) {
            fprintf(stderr, "[SYCL WARNING] Invalid reorder transition %d → %d for tensor '%s'. "
                    "Valid: NONE→SOA, SOA→COALESCED\n",
                    (int)reorder_, (int)new_mode, tensor_name ? tensor_name : "?");
        }

        reorder_ = new_mode;
    }

public:
    // Reset reorder mode to NONE when new AoS data is written to the tensor
    // This is called by set_tensor to invalidate any prior reordering
    void reset_reorder(const char* tensor_name) {
        if (reorder_ != reorder_mode::NONE) {
            fprintf(stderr, "[REORDER-RESET] %d → 0 for '%s' (data overwritten)\n",
                    (int)reorder_, tensor_name ? tensor_name : "?");
            reorder_ = reorder_mode::NONE;
        }
    }

    // Mark as SoA when data was transformed on CPU before upload (faster than GPU transform)
    // ONLY call this when the data in device memory is already in SoA layout!
    void mark_soa_pretransformed(const char* tensor_name) {
        reorder_ = reorder_mode::SOA;
        GGML_UNUSED(tensor_name);
    }

    // Set the data owner for view tensors. Call this when creating a view.
    void set_data_owner(optimize_feature* owner) { data_owner_ = owner; }

    // Exact mode checks - use these for kernel dispatch
    bool is_none() const { return get_reorder() == reorder_mode::NONE; }
    bool is_soa() const { return get_reorder() == reorder_mode::SOA; }
    bool is_coalesced() const { return get_reorder() == reorder_mode::COALESCED; }

    // Check if ANY reorder was applied - use for "skip if already reordered" logic
    bool is_reordered() const { return get_reorder() != reorder_mode::NONE; }

    // Get current mode - for views, returns the data owner's mode
    reorder_mode get_reorder() const {
        if (data_owner_ != nullptr) {
            return data_owner_->get_reorder();
        }
        return reorder_;
    }

};

struct sycl_device_info {
    int     cc;                 // compute capability
    int nsm; // number of streaming multiprocessors (CUDA) maps to the maximum
             // number of compute units on a SYCL device.
    // size_t  smpb;               // max. shared memory per block
    size_t  smpbo;              // max. shared memory per block (with opt-in)
    bool    vmm;                // virtual memory support
    size_t  total_vram;
    //sycl_hw_info hw_info;     \\ device id and aarch, currently not used
    bool    supports_soa_reorder = false;  // Device capability: can use SoA weight layout
};


struct ggml_sycl_device_info {
    int device_count;

    sycl_device_info devices[GGML_SYCL_MAX_DEVICES] = {};

    std::array<float, GGML_SYCL_MAX_DEVICES> default_tensor_split = {};

    int max_work_group_sizes[GGML_SYCL_MAX_DEVICES] = {0};
};

const ggml_sycl_device_info & ggml_sycl_info();

// Tensor Parallelism configuration
// Implements Megatron-LM style column/row parallel for multi-GPU inference
enum class tp_layer_type {
    TP_NONE,           // No tensor parallelism
    TP_COLUMN_PARALLEL, // Split output features: Q, K, V, gate, up projections
    TP_ROW_PARALLEL,    // Split input features: out_proj, down projections (needs all-reduce)
};

struct ggml_sycl_tp_config {
    bool enabled = false;         // Whether tensor parallelism is active
    int world_size = 1;           // Number of GPUs in TP group
    int rank = 0;                 // This GPU's rank (0 to world_size-1)
    int devices[GGML_SYCL_MAX_DEVICES] = {0}; // Device IDs in TP group

    // Buffers for all-reduce operations (allocated lazily)
    void* allreduce_buffer[GGML_SYCL_MAX_DEVICES] = {nullptr};
    size_t allreduce_buffer_size = 0;

    // Multi-process mode (one GPU per process, coordinated via MPI/CCL)
    bool is_multiprocess = false; // True if running with mpirun
    int mpi_rank = -1;            // MPI rank (process ID)
    int mpi_world_size = 0;       // MPI world size (number of processes)
};

// Global TP config (set during init)
extern ggml_sycl_tp_config g_sycl_tp_config;

// Initialize tensor parallelism with specified devices
void ggml_sycl_tp_init(const int* device_ids, int num_devices);

// Clean up tensor parallelism resources
void ggml_sycl_tp_free();

// Perform all-reduce sum across TP group
// buf must be device memory on the calling device
void ggml_sycl_tp_allreduce_sum(float* buf, size_t count, int device, queue_ptr stream);

// Perform all-reduce sum with explicit buffers for each device
void ggml_sycl_tp_allreduce_sum_multi(float** buf_per_device, size_t count,
                                       queue_ptr* streams, int num_devices);

// Get/ensure shared buffer for optimized ALL_REDUCE (malloc_shared for zero-copy)
float* ggml_sycl_tp_ensure_shared_reduce_buffer(size_t bytes);

// Get persistent host buffers for CPU-based ALL_REDUCE (avoids per-call malloc/free)
// Returns two host buffers: one for dev0 data, one for dev1 data
// Grows buffers as needed, reuses across calls
void ggml_sycl_tp_get_host_reduce_buffers(size_t bytes, float** buf0, float** buf1);

// Get persistent shared buffer for device-to-device transfers (PP optimization)
// Uses malloc_shared to avoid per-transfer malloc/free overhead
// Auto-grows buffer as needed, reuses across calls
void* ggml_sycl_get_dev2dev_transfer_buffer(size_t bytes);

// Get buffer for double-buffered transfer (returns buffer index via out param)
// Double-buffering allows overlapping src->host copy with host->dst copy
void* ggml_sycl_get_dev2dev_transfer_buffer_double(size_t bytes, int* buf_idx);

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
void ggml_sycl_tp_get_slice(int64_t total_size, int rank, int world_size,
                             int64_t* offset, int64_t* size);

// Get TP layer type for a tensor (uses cached value if available)
// First call does string matching, subsequent calls just return cached enum
tp_layer_type ggml_sycl_tp_get_layer_type(const ggml_tensor* tensor);

// Check if a tensor requires all-reduce after matmul
bool ggml_sycl_tp_needs_allreduce(const ggml_tensor* tensor);

// Weight sharding functions for tensor parallelism
// Get the sharded dimensions for a TP tensor
void ggml_sycl_tp_get_sharded_dims(const ggml_tensor* tensor, int rank, int world_size,
                                    int64_t* local_ne0, int64_t* local_ne1,
                                    int64_t* offset_ne0, int64_t* offset_ne1);

// Check if a tensor should be sharded for TP
bool ggml_sycl_tp_should_shard(const ggml_tensor* tensor);

// Copy sharded weight data from host to device
void ggml_sycl_tp_copy_weight_shard(void* dst_device, const void* src_host,
                                     const ggml_tensor* tensor, int rank,
                                     int world_size, queue_ptr stream);

// Get the size in bytes of a sharded tensor for this rank
size_t ggml_sycl_tp_get_shard_size(const ggml_tensor* tensor, int rank, int world_size);

// =============================================================================
// Quantized Communication Buffers (Flash Communication)
// Pre-allocated buffers for INT16 quantized AllReduce - 33% bandwidth reduction
// INT16 has 65536 levels vs INT8's 256 → 0.0015% max error vs 0.4%
// Total bandwidth: 8N bytes (2N×2 INT16 + 4N FP32 result) vs 12N standard
// =============================================================================

struct ggml_sycl_tp_quant_comm_buffers {
    int16_t* dev_q[GGML_SYCL_MAX_DEVICES];     // INT16 device buffers (2 bytes per element)
    float* dev_minmax[GGML_SYCL_MAX_DEVICES];  // [min, max] per device
    int16_t* host_q0;                           // Host buffer for device 0 INT16
    int16_t* host_q1;                           // Host buffer for device 1 INT16
    float* host_result;                         // Host buffer for FP32 result
    size_t capacity;                            // Current allocation size (elements)
    bool allocated;
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
ggml_sycl_tp_quant_comm_buffers* ggml_sycl_tp_get_quant_comm_buffers();

// Free quantized comm buffers
void ggml_sycl_tp_free_quant_comm_buffers();

// =============================================================================
// Pipeline Parallelism (PP) configuration
// Implements vLLM-style pipeline parallelism with layer-based device distribution
// =============================================================================

#define GGML_SYCL_PP_MAX_LAYERS 256

struct ggml_sycl_pp_config {
    bool enabled = false;                           // Whether pipeline parallelism is active
    int num_stages = 0;                             // Number of pipeline stages (typically = num_devices)
    int layers_per_stage[GGML_SYCL_MAX_DEVICES] = {0};  // Layers per stage (for uneven distribution)
    int layer_to_device[GGML_SYCL_PP_MAX_LAYERS] = {0}; // Quick lookup: layer_id -> device_id
    int devices[GGML_SYCL_MAX_DEVICES] = {0};       // Device IDs in PP order

    // Inter-stage buffers (malloc_shared for Intel Arc without P2P)
    void* stage_output_buf[GGML_SYCL_MAX_DEVICES] = {nullptr};
    size_t stage_output_size = 0;                   // Current buffer size per stage

    // Synchronization events for pipelining
    sycl::event stage_complete[GGML_SYCL_MAX_DEVICES];

    // Chunked prefill state
    int32_t chunk_size = 0;                         // Max tokens per prefill chunk (0 = disabled)
    bool chunked_prefill_enabled = false;           // Whether chunked prefill is active

    // Statistics
    int64_t total_stage_transfers = 0;
    int64_t total_sync_waits = 0;
};

// Global PP config (set during init)
extern ggml_sycl_pp_config g_sycl_pp_config;

// PP debug output - controlled by GGML_SYCL_PP_DEBUG env var
extern int g_ggml_sycl_pp_debug;

#define GGML_SYCL_PP_DEBUG(...)              \
    do {                                     \
        if (UNLIKELY(g_ggml_sycl_pp_debug))  \
            fprintf(stderr, __VA_ARGS__);    \
    } while (0)

// Initialize pipeline parallelism with specified devices and layer distribution
// If layers_per_stage is nullptr, layers are distributed evenly
void ggml_sycl_pp_init(const int* device_ids, int num_devices, int total_layers,
                        const int* layers_per_stage = nullptr);

// Clean up pipeline parallelism resources
void ggml_sycl_pp_free();

// Get the device ID for a given layer
int ggml_sycl_pp_get_device_for_layer(int layer);

// Allocate/ensure inter-stage buffer for given size
// Uses malloc_shared for Intel Arc (no P2P support)
void* ggml_sycl_pp_ensure_stage_buffer(int stage, size_t size);

// Transfer layer output from one stage to the next
// src_device: device that produced the output
// dst_device: device that will consume it
// Returns event that signals transfer completion
sycl::event ggml_sycl_pp_stage_transfer(int src_device, int dst_device,
                                         const void* src, size_t size,
                                         queue_ptr src_queue, queue_ptr dst_queue);

// Wait for a stage to complete (blocking)
void ggml_sycl_pp_sync_stage(int stage);

// Wait for all stages to complete
void ggml_sycl_pp_sync_all();

// Check if PP is enabled
bool ggml_sycl_pp_enabled();

// Get number of pipeline stages
int ggml_sycl_pp_num_stages();

// Get layer range for a stage: [start_layer, end_layer)
void ggml_sycl_pp_get_stage_layers(int stage, int* start_layer, int* end_layer);

// Get stage for a given layer
int ggml_sycl_pp_get_stage_for_layer(int layer);

// Set chunked prefill configuration
void ggml_sycl_pp_set_chunked_prefill(int32_t chunk_size, bool enabled);

// Get staging buffer for reading (after stage transfer is complete)
void* ggml_sycl_pp_get_stage_buffer(int stage);

// Get PP statistics (transfers and sync waits)
void ggml_sycl_pp_get_stats(int64_t* transfers, int64_t* syncs);

// Reset PP statistics
void ggml_sycl_pp_reset_stats();

// FFN norm cache for TP: stores FFN norm output immediately after MUL to prevent buffer aliasing
// The GGML scheduler may reuse the FFN norm buffer before TP can use it on device 1
struct ffn_norm_cache_entry {
    void* data;           // Cached FFN norm output on main device (device 0)
    void* data_dev1;      // Copy on device 1 for its computation
    int64_t ne0, ne1;     // Dimensions
    size_t size;          // Buffer size in bytes
    int pass_id;          // Which compute pass this cache is for (to detect staleness)
};

// Global FFN norm cache indexed by layer number
extern std::unordered_map<int, ffn_norm_cache_entry> g_tp_ffn_norm_cache;
extern std::mutex g_tp_ffn_norm_cache_mutex;
extern int g_tp_current_pass_id;  // Incremented each forward pass
extern bool g_tp_enabled;  // Whether TP mode is enabled

// Store FFN norm output for TP (call after MUL that creates ffn_norm)
void ggml_sycl_tp_cache_ffn_norm(int layer, const void* data, int64_t ne0, int64_t ne1,
                                  size_t size, queue_ptr stream);

// Get cached FFN norm for a layer (returns nullptr if not cached or stale)
void* ggml_sycl_tp_get_cached_ffn_norm(int layer, int device);

// Clear FFN norm cache for a layer
void ggml_sycl_tp_clear_ffn_norm_cache(int layer);

// Increment pass ID (call at start of each forward pass)
void ggml_sycl_tp_new_pass();

// FFN input storage: stores the input to FFN column-parallel layers on device 1
// This is needed so that row-parallel (ffn_down) can compute device 1's contribution
struct ffn_input_storage {
    void * data;          // Buffer on device 1
    int64_t ne0, ne1;     // Dimensions
    size_t size;          // Buffer size
};
extern std::unordered_map<int, ffn_input_storage> g_tp_ffn_inputs;  // Key: layer number
extern std::mutex g_tp_ffn_input_mutex;

// FFN weight storage: stores references to FFN weight tensors for device 1 computation
struct ffn_weight_refs {
    const ggml_tensor * gate;  // ffn_gate weight tensor
    const ggml_tensor * up;    // ffn_up weight tensor
    const ggml_tensor * down;  // ffn_down weight tensor
};
extern std::unordered_map<int, ffn_weight_refs> g_tp_ffn_weights;  // Key: layer number
extern std::mutex g_tp_ffn_weight_mutex;

// Attention input storage: stores the input to attention column-parallel layers on device 1
struct attn_input_storage {
    void * data;          // Buffer on device 1
    int64_t ne0, ne1;     // Dimensions
    size_t size;          // Buffer size
};
extern std::unordered_map<int, attn_input_storage> g_tp_attn_inputs;  // Key: layer number
extern std::mutex g_tp_attn_input_mutex;

// Attention weight storage: stores references to attention weight tensors
struct attn_weight_refs {
    const ggml_tensor * q;     // attn_q weight tensor
    const ggml_tensor * k;     // attn_k weight tensor
    const ggml_tensor * v;     // attn_v weight tensor
    const ggml_tensor * o;     // attn_output weight tensor
};
extern std::unordered_map<int, attn_weight_refs> g_tp_attn_weights;  // Key: layer number
extern std::mutex g_tp_attn_weight_mutex;

// Async FFN job structure: tracks an in-flight FFN computation on device 1
// This allows device 1 to compute while device 0 continues with other work
struct tp_async_ffn_job {
    int layer;                      // Layer number
    sycl::event completion_event;   // Event signaling computation complete
    float * result_buf;             // Result buffer (in pinned host memory)
    int64_t ne0, ne1;               // Output dimensions [N_out, batch]
    size_t result_size;             // Result buffer size in bytes
    bool valid;                     // Job is valid and pending
};
extern std::unordered_map<int, tp_async_ffn_job> g_tp_async_ffn_jobs;  // Key: layer number
extern std::mutex g_tp_async_ffn_mutex;

// Async attention job structure: tracks an in-flight attention computation on device 1
struct tp_async_attn_job {
    int layer;                      // Layer number
    sycl::event completion_event;   // Event signaling computation complete
    float * result_buf;             // Result buffer (in pinned host memory)
    int64_t ne0, ne1;               // Output dimensions
    size_t result_size;             // Result buffer size in bytes
    bool valid;                     // Job is valid and pending
};
extern std::unordered_map<int, tp_async_attn_job> g_tp_async_attn_jobs;  // Key: layer number
extern std::mutex g_tp_async_attn_mutex;

// Extract layer number from tensor name (e.g., "blk.0.ffn_gate" -> 0)
int ggml_sycl_tp_extract_layer_number(const char * name);

// =============================================================================
// Thread-based pipelining for device 1 FFN computation
// Uses a dedicated worker thread instead of SYCL async events (which don't work
// with in-order queues that have multiple wait() calls).
// =============================================================================

// FFN work item: describes an FFN computation to be performed on device 1
struct tp_ffn_work_item {
    int layer;                          // Layer number
    float * input_dev1;                 // Input pointer on device 1 (already copied)
    int64_t K_full;                     // Input dimension
    int64_t batch;                      // Batch size
    ffn_weight_refs weights;            // Weight tensor references

    // Output info (filled in by caller for result allocation)
    int64_t N_out;                      // Output dimension
    size_t result_size;                 // Expected result size in bytes
};

// FFN result: result of a completed FFN computation
struct tp_ffn_result {
    int layer;                          // Layer number
    float * result_buf;                 // Result buffer (host-pinned memory)
    int64_t ne0, ne1;                   // Output dimensions
    size_t result_size;                 // Result size in bytes
    bool valid;                         // Result is valid and ready to consume
};

// Device 1 worker thread: processes FFN jobs independently from main thread
struct tp_device1_worker {
    std::thread worker_thread;

    // Work queue: main thread submits, worker thread processes
    std::queue<tp_ffn_work_item> work_queue;
    std::mutex work_mutex;
    std::condition_variable work_cv;

    // Results: worker thread produces, main thread consumes
    std::unordered_map<int, tp_ffn_result> results;  // Key: layer number
    std::mutex result_mutex;
    std::condition_variable result_cv;

    // Control
    std::atomic<bool> shutdown{false};
    std::atomic<bool> initialized{false};

    // Context pointer (set during init)
    void * ctx;  // ggml_backend_sycl_context *
};

// Global worker instance
extern tp_device1_worker g_tp_device1_worker;

// Global flag to enable/disable thread-based pipelining
extern int g_ggml_sycl_tp_threaded_ffn;  // 0 = disabled, 1 = enabled

// Thread-based pipelining functions
void ggml_sycl_tp_worker_init(void * ctx);   // Initialize worker thread
void ggml_sycl_tp_worker_shutdown();          // Shutdown worker thread
void ggml_sycl_tp_submit_ffn_work(const tp_ffn_work_item & work);  // Submit work to queue
tp_ffn_result * ggml_sycl_tp_get_ffn_result(int layer, bool wait);  // Get result (optional wait)
void ggml_sycl_tp_release_ffn_result(int layer);  // Release result memory

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
    size_t hidden_size;  // Size of gate_out, up_out, hidden_out

    // Hidden quantization buffer for down matmul
    char * hidden_q8_dev;
    size_t hidden_q8_size;

    // Output buffer for partial result
    float * partial_out;
    size_t partial_size;

    // Track allocated sizes (for resize detection)
    int64_t K_full_padded;
    int64_t N_hidden_shard_padded;
    int64_t batch_max;
    int64_t N_out;

    // Flag indicating if buffers are allocated
    bool allocated;
};

// Global map of persistent FFN buffers indexed by layer
extern std::unordered_map<int, tp_ffn_compute_buffers> g_tp_ffn_buffers;
extern std::mutex g_tp_ffn_buffers_mutex;

// Ensure persistent FFN buffers are allocated for a layer
// Returns pointer to buffers, allocates if needed, resizes if dimensions changed
tp_ffn_compute_buffers * ggml_sycl_tp_ensure_ffn_buffers(
    int layer, int device, queue_ptr stream,
    int64_t K_full_padded, int64_t N_hidden_shard_padded, int64_t batch, int64_t N_out);

// Free all persistent FFN buffers (called during cleanup)
void ggml_sycl_tp_free_ffn_buffers();

// =============================================================================
// Persistent host staging buffer for TP input copies
// =============================================================================

struct tp_host_staging_buffer {
    float * buf;
    size_t size;
    size_t capacity;
};

extern tp_host_staging_buffer g_tp_host_staging;
extern std::mutex g_tp_host_staging_mutex;

// Ensure host staging buffer has at least the given capacity
float * ggml_sycl_tp_ensure_host_staging(size_t size, queue_ptr stream);

// Free host staging buffer
void ggml_sycl_tp_free_host_staging();

struct ggml_sycl_pool {
    virtual ~ggml_sycl_pool() = default;

    virtual void * alloc(size_t size, size_t * actual_size) = 0;
    virtual void free(void * ptr, size_t size) = 0;
};

template<typename T>
struct ggml_sycl_pool_alloc {
    ggml_sycl_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    explicit ggml_sycl_pool_alloc(ggml_sycl_pool & pool) : pool(&pool) {
    }

    ggml_sycl_pool_alloc(ggml_sycl_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_sycl_pool_alloc() {
        if (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }

    T * realloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        if (ptr)
            pool->free(ptr, actual_size);
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

    T * get() {
        return ptr;
    }

    ggml_sycl_pool_alloc() = default;
    ggml_sycl_pool_alloc(const ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc(ggml_sycl_pool_alloc &&) = delete;
    ggml_sycl_pool_alloc& operator=(const ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc& operator=(ggml_sycl_pool_alloc &&) = delete;
};

// backend interface

struct ggml_tensor_extra_gpu {
  void* data_device[GGML_SYCL_MAX_DEVICES]; // 1 pointer for each device for split
                                       // tensors
  dpct::event_ptr events[GGML_SYCL_MAX_DEVICES]
                        [GGML_SYCL_MAX_STREAMS]; // events for synchronizing multiple GPUs
  optimize_feature optimized_feature = {};  // Must have = {} to ensure default member initializers apply
  tp_layer_type tp_type = tp_layer_type::TP_NONE;  // Cached TP type (set once, avoids string compare)
  bool tp_type_cached = false;  // Whether tp_type has been computed

  // Tensor Parallelism sharding info
  // When TP is enabled, this tensor may hold only a shard of the full weight
  bool tp_sharded = false;        // True if this tensor holds a shard
  bool tp_usm_host = false;       // True if allocated with malloc_host (cross-device accessible)
  int64_t tp_original_ne[4] = {0}; // Original (full) dimensions before sharding
  int64_t tp_local_ne[4] = {0};   // Local dimensions of the shard
  int64_t tp_offset_ne[4] = {0};  // Offset into the original tensor
  int tp_rank = 0;                // Which rank this shard belongs to
  int tp_world_size = 1;          // Total number of ranks
};


void release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams={});

// Get the correct data pointer for a tensor on a specific device
// For TP buffers, returns device-specific pointer; otherwise returns tensor->data
// In TP mode, if returning tensor->data, stages it to USM memory first
inline void * ggml_sycl_get_data_ptr(const ggml_tensor * tensor, int device) {
    if (tensor == nullptr) {
        return nullptr;
    }
    if (tensor->extra != nullptr) {
        const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(tensor->extra);
        if (extra->data_device[device] != nullptr) {
            GGML_SYCL_DEBUG("ggml_sycl_get_data_ptr: tensor=%s, device=%d, using extra->data_device[%d]=%p\n",
                            tensor->name, device, device, extra->data_device[device]);
            return extra->data_device[device];
        }
    }

    // In TP mode, tensor->data might be mmap'd memory that can't be accessed by shared-context queues.
    // Stage it to device-local USM memory for each device (since Intel Arc lacks P2P).
    // BUT: If tensor->data is already HOST or SHARED USM memory, use it directly (no staging needed).
    if (g_sycl_tp_config.enabled && g_sycl_tp_config.world_size > 1 && tensor->data != nullptr) {
        // Check if the pointer is already HOST or SHARED USM - these are device-accessible
        sycl::context * tp_ctx = ggml_sycl_get_tp_context();
        if (tp_ctx != nullptr) {
            sycl::usm::alloc ptr_type = sycl::get_pointer_type(tensor->data, *tp_ctx);
            if (ptr_type == sycl::usm::alloc::host || ptr_type == sycl::usm::alloc::shared) {
                // HOST or SHARED USM - directly accessible from device, no staging needed
                GGML_SYCL_DEBUG("ggml_sycl_get_data_ptr: tensor=%s, device=%d, using HOST/SHARED USM tensor->data=%p (type=%d)\n",
                                tensor->name, device, tensor->data, (int)ptr_type);
                return tensor->data;
            }
        }

        // Not HOST/SHARED USM - need to stage to device-local memory
        size_t nbytes = ggml_nbytes(tensor);
        void * staged = ggml_sycl_get_staged_ptr_device(tensor->data, nbytes, device);
        if (staged != nullptr) {
            GGML_SYCL_DEBUG("ggml_sycl_get_data_ptr: tensor=%s, device=%d, staged %p -> %p (%zu bytes)\n",
                            tensor->name, device, tensor->data, staged, nbytes);
            return staged;
        }
        // Staging failed - fall through and return original pointer (will likely fail)
        GGML_SYCL_DEBUG("ggml_sycl_get_data_ptr: tensor=%s, device=%d, staging FAILED, using tensor->data=%p\n",
                        tensor->name, device, tensor->data);
    } else {
        GGML_SYCL_DEBUG("ggml_sycl_get_data_ptr: tensor=%s, device=%d, using tensor->data=%p\n",
                        tensor->name, device, tensor->data);
    }
    return tensor->data;
}

namespace sycl_ex = sycl::ext::oneapi::experimental;

struct ggml_backend_sycl_context {
    int device;
    std::string name;
    // Device capability: does this device support SoA weight layout optimization?
    // This is NOT tensor state - it's a static capability of the GPU.
    // Tensor state is tracked per-tensor in ggml_tensor_extra_gpu::optimized_feature
    bool supports_soa_reorder;

    queue_ptr qptrs[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS] = { { nullptr } };

    explicit ggml_backend_sycl_context(int device) :
        device(device),
        name(GGML_SYCL_NAME + std::to_string(device)),
        supports_soa_reorder(ggml_sycl_info().devices[device].supports_soa_reorder) {
    }

    ~ggml_backend_sycl_context();

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
            qptrs[device][stream] = &(dpct::get_device(device).default_queue());
        }
        return qptrs[device][stream];
    }

    queue_ptr stream() {
        return stream(device, 0);
    }

#if GGML_SYCL_DNNL
    dnnl::engine make_engine(sycl::queue* q) {
        // Get the device associated with the queue
        sycl::device dev = q->get_device();
        // Get the context associated with the queue
        sycl::context ctx = q->get_context();
        const dnnl::engine eng = dnnl::sycl_interop::make_engine(dev, ctx);
        return eng;
    }

    std::unordered_map<sycl::queue*, dnnl::stream> stream_map;
    std::unordered_map<sycl::queue*, dnnl::engine> engine_map;
    dnnl::stream stream_dnnl(int device, int _stream) {
        auto q = stream(device, _stream);
        return stream_dnnl(q);
    }
    dnnl::engine engine_dnnl(sycl::queue* qptr) {
        auto it = engine_map.find(qptr);
        if (it == engine_map.end()) {
            auto eng = make_engine(qptr);
            engine_map[qptr] = eng;
            return eng;
        }
        else
        {
            return it->second;
        }
    }
    dnnl::stream stream_dnnl(sycl::queue* qptr) {
        auto it = stream_map.find(qptr);
        if (it == stream_map.end()) {
            auto eng = engine_dnnl(qptr);
            auto stream = dnnl::sycl_interop::make_stream(eng, *qptr);
            stream_map[qptr] = stream;
            return stream;
        }
        else
        {
            return it->second;
        }
    }
    dnnl::stream stream_dnnl() {
        return stream_dnnl(device, 0);
    }
    dnnl::memory get_scratchpad_mem(const dnnl::memory::desc & scratchpad_md,
                                    const dnnl::engine & eng, const queue_ptr q) {
        ggml_sycl_pool_alloc<uint8_t> * pool;
        auto it = scratchpad_map.find(q);
        if (it == scratchpad_map.end()) {
            scratchpad_map[q] = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(this->pool());
            pool = scratchpad_map[q].get();
        } else {
            pool = it->second.get();
        }

        size_t scratchpad_size = scratchpad_md.get_size();
        if (scratchpad_size > pool->actual_size) {
            pool->realloc(scratchpad_size);
        }
        void * mem_ptr = pool->get();
        return dnnl::memory(scratchpad_md, eng, mem_ptr);
    }

    // Pre-allocate scratchpad pool to a given size
    // Used before graph recording to avoid realloc during recording
    void pre_allocate_scratchpad(size_t size, const queue_ptr q) {
        if (size == 0) return;

        ggml_sycl_pool_alloc<uint8_t> * pool;
        auto it = scratchpad_map.find(q);
        if (it == scratchpad_map.end()) {
            scratchpad_map[q] = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(this->pool());
            pool = scratchpad_map[q].get();
        } else {
            pool = it->second.get();
        }

        if (size > pool->actual_size) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] Pre-allocating scratchpad pool: %zu bytes\n", size);
            pool->realloc(size);
        }
    }
#endif

    // pool
    std::unique_ptr<ggml_sycl_pool> pools[GGML_SYCL_MAX_DEVICES];
    std::unordered_map<sycl::queue *, std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>>> scratchpad_map;

    std::unique_ptr<ggml_sycl_pool> host_pools[GGML_SYCL_MAX_DEVICES];

    static std::unique_ptr<ggml_sycl_pool> new_pool_for_device(queue_ptr qptr, int device);

    static std::unique_ptr<ggml_sycl_pool> new_pool_for_host(queue_ptr qptr, int device);

    ggml_sycl_pool & pool(int device) {
        if (pools[device] == nullptr) {
            pools[device] = new_pool_for_device(stream(device,0), device);
        }
        return *pools[device];
    }

    ggml_sycl_pool & pool() {
        return pool(device);
    }

#ifdef GGML_SYCL_GRAPH
    std::unique_ptr<sycl_ex::command_graph<sycl_ex::graph_state::executable>> exec_graph = nullptr;
    int exec_graph_n_nodes = 0;      // Track graph size for cache invalidation
    bool exec_graph_is_decode = false;  // Track which phase the cached graph was recorded for
    int warmup_decode_n_nodes = 0;   // Track which decode graph has been warmed up
    int warmup_prompt_n_nodes = 0;   // Track which prompt graph has been warmed up
    bool moe_graphs_disabled = false; // Set when MoE preload fails; disables graphs for all splits

    // Pre-allocated buffers for MoE graph recording
    // MUL_MAT_ID needs Q8_1 quantization buffers which cannot be allocated during graph recording
    struct moe_graph_buffers {
        // Q8_1 quantization buffers (one per MUL_MAT_ID in decode phase)
        std::vector<void*> q8_1_buffers;
        std::vector<size_t> q8_1_sizes;

        // Buffer usage tracking
        int current_buffer_idx = 0;
        bool initialized = false;

        // Max dimensions seen (for reallocation check)
        int64_t max_ne10 = 0;      // Max input dimension
        int64_t max_src1_rows = 0; // Max (ne11 × ne12)

        void reset_usage() { current_buffer_idx = 0; }

        void* get_next_buffer(size_t required_size) {
            if (current_buffer_idx >= (int)q8_1_buffers.size()) {
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
            initialized = false;
            current_buffer_idx = 0;
            max_ne10 = 0;
            max_src1_rows = 0;
        }
    } moe_buffers;

    // Pre-allocated buffers for SoA MMVQ graph recording
    // MUL_MAT with SoA reorder flag needs Q8_1 quantization buffers which cannot be
    // allocated from pool during graph recording (pointer would change on replay)
    struct mmvq_soa_buffers_t {
        // Q8_1 quantization buffers (one per SoA MUL_MAT in decode phase)
        std::vector<void*> src1_ddq_buffers;
        std::vector<size_t> src1_ddq_sizes;

        // Buffer usage tracking
        int current_buffer_idx = 0;
        bool initialized = false;

        // Max dimensions seen (for reallocation check)
        int64_t max_ne10 = 0;      // Max input dimension
        int64_t max_nrows = 0;     // Max rows

        void reset_usage() { current_buffer_idx = 0; }

        void* get_next_buffer(size_t required_size) {
            if (current_buffer_idx >= (int)src1_ddq_buffers.size()) {
                return nullptr;  // Fall back to pool alloc
            }
            if (required_size > src1_ddq_sizes[current_buffer_idx]) {
                return nullptr;  // Buffer too small
            }
            return src1_ddq_buffers[current_buffer_idx++];
        }

        void free_buffers(queue_ptr stream) {
            for (size_t i = 0; i < src1_ddq_buffers.size(); i++) {
                if (src1_ddq_buffers[i]) {
                    sycl::free(src1_ddq_buffers[i], *stream);
                }
            }
            src1_ddq_buffers.clear();
            src1_ddq_sizes.clear();
            initialized = false;
            current_buffer_idx = 0;
            max_ne10 = 0;
            max_nrows = 0;
        }
    } mmvq_soa_buffers;

    // Q8_1 quantization cache for MoE: avoids re-quantizing same input across gate/up/down
    // In MoE layers, the same input is used for all projections - caching saves 3x quantization
    struct moe_quant_cache {
        void* cached_q8_1 = nullptr;       // Cached Q8_1 quantized data
        const void* cached_src = nullptr;  // Key: source pointer that was quantized
        int64_t cached_ne10 = 0;           // Input row width
        int64_t cached_rows = 0;           // Number of rows quantized
        size_t cached_size = 0;            // Buffer size
        bool valid = false;                // Cache entry is valid

        void invalidate() {
            cached_src = nullptr;
            cached_ne10 = 0;
            cached_rows = 0;
            valid = false;
            // Note: don't free cached_q8_1 - it's pool memory that gets reused
        }

        // Check if cache matches current request
        bool matches(const void* src, int64_t ne10, int64_t rows) const {
            return valid && cached_src == src && cached_ne10 == ne10 && cached_rows == rows;
        }
    } moe_q8_cache;
#endif

    // Barrier event for cross-ubatch synchronization
    // This provides lighter-weight sync than full queue wait
    std::optional<sycl::event> barrier_event;
    bool has_pending_barrier = false;

    ggml_sycl_pool & host_pool(int device) {
        if (host_pools[device] == nullptr) {
            host_pools[device] = new_pool_for_host(stream(device, 0), device);
        }
        return *host_pools[device];
    }

    ggml_sycl_pool & host_pool() { return host_pool(device); }

    // Flag to disable graphs when weight streaming is active
    bool weight_streaming_graphs_disabled = false;

    // KV offload manager for long context support (initialized lazily when enabled)
    std::unique_ptr<ggml_sycl::kv_offload_manager> kv_offload_mgr_;

    // Initialize KV offload manager with given configuration
    void init_kv_offload(const ggml_sycl::kv_offload_config& config);

    // Check if KV offload is initialized
    bool has_kv_offload() const { return kv_offload_mgr_ != nullptr; }

    // Get the KV offload manager (must be initialized first)
    ggml_sycl::kv_offload_manager* kv_offload() { return kv_offload_mgr_.get(); }
};

// GGML_OP_ALL_REDUCE_SUM handler for SYCL backend
// For single-device execution, this is a copy operation
// For multi-device TP, this will perform actual all-reduce across devices
void ggml_sycl_all_reduce_sum(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// Async FFN computation for Tensor Parallelism pipelining
// Launches FFN computation on device 1 asynchronously (returns immediately)
void ggml_sycl_tp_launch_async_ffn(
    ggml_backend_sycl_context & ctx,
    int layer,
    const float * input_dev1,       // Input on device 1
    int64_t K_full,                 // Full model dimension
    int64_t batch,                  // Batch size
    const ffn_weight_refs & weights // Weight tensor references
);

// Wait for and retrieve async FFN result (blocks until done)
float * ggml_sycl_tp_wait_async_ffn(int layer, int64_t * out_ne0, int64_t * out_ne1, size_t * out_size);

// Async attention computation for Tensor Parallelism pipelining
void ggml_sycl_tp_launch_async_attn(
    ggml_backend_sycl_context & ctx,
    int layer,
    const float * input_dev1,       // Input on device 1
    int64_t K_full,                 // Full model dimension
    int64_t batch,                  // Batch size
    const attn_weight_refs & weights // Weight tensor references
);

// Wait for and retrieve async attention result
float * ggml_sycl_tp_wait_async_attn(int layer, int64_t * out_ne0, int64_t * out_ne1, size_t * out_size);

// common device functions

static __dpct_inline__ float warp_reduce_sum(float x,
    const sycl::nd_item<3>& item_ct1) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), x, mask);
    }
    return x;
}

static __dpct_inline__ sycl::float2
warp_reduce_sum(sycl::float2 a, const sycl::nd_item<3>& item_ct1) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        a.x() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.x(),
            mask);
        a.y() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.y(),
            mask);
    }
    return a;
}

template <int width = WARP_SIZE>
static __dpct_inline__ int warp_reduce_sum(int x) {
  return sycl::reduce_over_group(
      sycl::ext::oneapi::this_work_item::get_sub_group(), x, sycl::plus<>());
}

template <int width = WARP_SIZE>
static __dpct_inline__ float warp_reduce_sum(float x) {
  // Use optimized subgroup reduce for full WARP_SIZE (common case)
  if constexpr (width == WARP_SIZE) {
    return sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), x, sycl::plus<float>());
  } else {
    // Fallback for partial subgroup reductions
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
      x += dpct::permute_sub_group_by_xor(
          sycl::ext::oneapi::this_work_item::get_sub_group(), x, offset, width);
    }
    return x;
  }
}

template <int width = WARP_SIZE>
static __dpct_inline__ sycl::float2 warp_reduce_sum(sycl::float2 a) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    a.x() += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), a.x(), offset,
        width);
    a.y() += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), a.y(), offset,
        width);
  }
  return a;
}

template <int width = WARP_SIZE>
static __dpct_inline__ sycl::half2 warp_reduce_sum(sycl::half2 a) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    a = a + dpct::permute_sub_group_by_xor(
                sycl::ext::oneapi::this_work_item::get_sub_group(), a, offset,
                width);
  }
  return a;
}

static constexpr int ggml_sycl_get_physical_warp_size() {
  // todo: for old iGPU + dGPU case, need to be changed.
  return WARP_SIZE;
}

template <int width = WARP_SIZE>
static __dpct_inline__ float warp_reduce_max(float x) {
  // Use optimized subgroup reduce for full WARP_SIZE (common case)
  if constexpr (width == WARP_SIZE) {
    return sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), x, sycl::maximum<float>());
  } else {
    // Fallback for partial subgroup reductions
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
      x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
                            sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                            offset, width));
    }
    return x;
  }
}

static __dpct_inline__ float warp_reduce_max(float x,
    const sycl::nd_item<3>& item_ct1) {
    // Use optimized subgroup reduce
    return sycl::reduce_over_group(item_ct1.get_sub_group(), x, sycl::maximum<float>());
}

/* Helper for Computing the linear offset of a ggml_tensor given
per-dimension sizes, strides, and indices */
template<int N>
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
template <typename Tp, int n>
inline sycl::vec<Tp, n> vec_aligned_load(const Tp* aligned_ptr) {
    return *reinterpret_cast<const sycl::vec<Tp, n>*>(aligned_ptr);
}

// Helper for accessing pointers with no warnings
template <typename Tp, int dim>
static __dpct_inline__ Tp* get_pointer(sycl::local_accessor<Tp, dim> acc) {
    return acc.template get_multi_ptr<sycl::access::decorated::no>().get();
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_num, int64_t block_size);

constexpr size_t ceil_div(const size_t m, const size_t n) {
    return (m + n - 1) / n;
}

bool gpu_has_xmx(sycl::device &dev);

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

inline std::string debug_get_tensor_str(const std::string &prefix,
        const ggml_tensor *tensor, const std::string &suffix = "") {
    std::stringstream ss;
    if (LIKELY(!g_ggml_sycl_debug)) { return ss.str(); }
    ss << prefix.c_str() << "=";
    if (tensor) {
        ss << "'" << tensor->name << "':type=" << ggml_type_name(tensor->type);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";ne", tensor->ne);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";nb", tensor->nb);

        if (!ggml_is_contiguous(tensor)) { ss << ";strided"; }
        if (ggml_is_permuted(tensor)) { ss << ";permuted"; }
    } else {
        ss << "nullptr";
    }
    ss << suffix;
    return ss.str();
}

// Use scope_op_debug_print to log operations coming from running a model
struct scope_op_debug_print {
    // Use string_views to avoid the cost of creating a string and concatenating them
    // string_views must be alive for as long as the object is alive
    // scope_op_debug_print are used with string literals in practice which are stored in constant space so always accessible
    scope_op_debug_print(const std::string_view & func, const std::string_view & func_suffix, const ggml_tensor * dst,
                         std::size_t num_src, const std::string_view & suffix = "") :
        func(func),
        func_suffix(func_suffix) {
        if (LIKELY(!g_ggml_sycl_debug)) {
            return;
        }
        GGML_SYCL_DEBUG("[SYCL][OP] call %s%s:", func.data(), func_suffix.data());
        GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" dst", dst).c_str());
        if (dst) {
            for (std::size_t i = 0; i < num_src; ++i) {
                GGML_SYCL_DEBUG("%s", debug_get_tensor_str("\tsrc" + std::to_string(i), dst->src[i]).c_str());
            }
        }
        GGML_SYCL_DEBUG("%s\n", suffix.data());
    }

    scope_op_debug_print(const std::string_view & func, const ggml_tensor * dst, std::size_t num_src,
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
    const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

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


#endif // GGML_SYCL_COMMON_HPP
