/*
 * this file has 8 sections:
 * section-1  forward declarations, global vars, macros
 * section-2  data structures
 * section-3  troubleshooting and profiler
 * section-4  configuration class and helper functions
 * section-5  general helper functions
 * section-6  CDSP helper functions
 * section-7  Qualcomm compatibility layer
 * section-8  backend implementation
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stddef.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>

#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <map>
#include <set>
#include <tuple>
#include <queue>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <memory>
#include <regex>
#include <random>
#include <functional>
#include <unordered_map>
#include <condition_variable>
#include <unordered_set>
#include <utility>
#include <future>
#include <algorithm>

#if defined(__ANDROID__) || defined(__linux__)
#include <unistd.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#endif

#if defined(__ANDROID__)
#include "android/log.h"
#endif

#include "rpcmem.h"
#include "remote.h"
#include "AEEStdErr.h"
#include "htp-drv.h"
#include "HAP_power.h"
#include "HAP_farf.h"

#include "ggml-hexagon.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml-quants.h"

#include "kernels/ggml_dsp.h"
#include "kernels/dsp-ctx.h"
#include "kernels/htp-ops.h"
#include "kernels/hex-common.h"
#include "kernels/hex-fastdiv.h"
#include "kernels/matmul-ops.h"
#include "kernels/flash-attn-ops.h"
#include "kernels/unary-ops.h"

// =================================================================================================
//  section-1: forward declarations, global vars, macros
// =================================================================================================
#ifdef NDEBUG
#define GGMLHEXAGON_DEBUG                               0
#else
#define GGMLHEXAGON_DEBUG                               1
#endif

#ifndef PROJECT_NAME
#define PROJECT_NAME                                    "ggml-hexagon"
#endif

#define GGMLHEXAGON_LOGBUF_LEN                          4096
#define GGMLHEXAGON_TMPBUF_LEN                          256

#define GGML_HEXAGON_MAX_DEVICES                        16
#define GGML_HEXAGON_BACKEND_NAME                       "hexagon"

#define GGML_DSP_IDL_VERSION                            "0.0.2"

#define GGMLHEXAGON_LOG_ALWAYS(...)                     ggmlhexagon_log_always_internal(GGML_LOG_LEVEL_NONE , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLHEXAGON_LOG_ERROR(...)                      ggmlhexagon_log_always_internal(GGML_LOG_LEVEL_ERROR, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLHEXAGON_LOG_WARN(...)                       ggmlhexagon_log_internal(GGML_LOG_LEVEL_WARN , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLHEXAGON_LOG_INFO(...)                       ggmlhexagon_log_internal(GGML_LOG_LEVEL_INFO , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLHEXAGON_LOG_VERBOSE(...)                    ggmlhexagon_log_always_internal(GGML_LOG_LEVEL_CONT , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

#if GGMLHEXAGON_DEBUG
#define GGMLHEXAGON_LOG_DEBUG(...)                      ggmlhexagon_log_internal(GGML_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#else
#define GGMLHEXAGON_LOG_DEBUG(...)
#endif

#define SIZE_IN_MB                                      (1 << 20)

#define GGMLHEXAGON_MAX_OPS_PER_TASK                    16

#define GGMLHEXAGON_MAX_TENSORS_PER_TASK                32

#define DMA_BUF_IOCTL_SYNC_IOCTL                        0x40086200u

#if !defined (_WINDOWS)
#pragma weak remote_system_request
#pragma weak remote_session_control
#pragma weak remote_handle_control
#pragma weak remote_handle64_control
#pragma weak fastrpc_mmap
#pragma weak fastrpc_munmap
#endif

// Forward declarations
struct ggml_backend_hexagon_context;
struct ggml_backend_hexagon_reg_context;

static bool                  ggmlhexagon_is_metadata_op(enum ggml_op op);
static int                   ggmlhexagon_probe_dspinfo(ggml_backend_hexagon_context * ctx);
static const char *          ggmlhexagon_get_htparch_desc(size_t htp_arch);
static int                   hexagon_warmup_invoke_timed(ggml_backend_hexagon_context * ctx);
static bool                  ggml_backend_buffer_is_hexagon_repack(const ggml_backend_buffer * b);
static bool                  ggml_backend_hexagon_buffer_is_host(ggml_backend_buffer_type_t buft);
static void                  ggmlhexagon_set_runtime_path(size_t device, const std::string & path);
static const char *          ggml_backend_hexagon_buffer_type_name(ggml_backend_buffer_type_t buft);
static ggml_backend_t        ggml_backend_hexagon_init_ext(size_t device, const char * runtime_libpath);
static size_t                ggml_backend_hexagon_buffer_type_get_max_size(ggml_backend_buffer_type_t buft);
static size_t                ggml_backend_hexagon_buffer_type_get_alignment(ggml_backend_buffer_type_t buft);
static bool                  ggml_backend_hexagon_repack_buffer_is_host(ggml_backend_buffer_type_t buft);
static bool                  ggmlhexagon_op_buffers_belong_to_dev(ggml_backend_dev_t dev, const ggml_tensor * op);
static bool                  ggmlhexagon_tensor_buffer_is_owned_by(ggml_backend_dev_t dev, const ggml_tensor * t);
static ggml_backend_t        ggml_backend_hexagon_device_init_backend(ggml_backend_dev_t dev, const char * params);
static size_t                ggml_backend_hexagon_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor);

static ggml_backend_hexagon_context * ggml_backend_hexagon_ensure_context(ggml_backend_dev_t dev);
static ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);
static bool                  ggml_hexagon_compute_fa_params(const ggml_backend_hexagon_context * ctx, const ggml_tensor * node, htp_fa_kernel_params * kparams);

GGML_BACKEND_API void       ggml_hexagon_set_runtime_libpath(const char * path);

// =================================================================================================
//  section-2: data structures
// =================================================================================================
enum qcom_htp_arch {
    NONE = 0,
    V73 = 73,
    V75 = 75,
    V79 = 79,
    V81 = 81,
};

enum qcom_chipset_soc_model {
    UNKNOWN_SM = 0,
    SM8550 = 43,  // v73, SD 8 Gen 2
    SM8650 = 57,  // v75, SD 8 Gen 3
    SM8750 = 69,  // v79, SD 8 Elite(aka 8 Gen 4)
    SM8850 = 73,  // v81, SD 8 Elite Gen 5(aka 8 Gen 5)
};

struct qcom_socinfo {
    uint32_t soc_model;
    size_t htp_arch;
    size_t vtcm_size_in_mb;
    char soc_desc[GGML_MAX_NAME];
};

// mempool region tracking for free-space management.
// Each region records an allocated area within the mempool.
// When free_buffer is called, the region is marked as not-in-use.
// Free regions can be reused by best-fit allocation.
struct ion_pool_region {
    size_t offset;      // byte offset from mempool base
    size_t size;        // allocation size in bytes
    bool   in_use;      // true if currently allocated
};

struct ggml_backend_hexagon_context {
    int device;
    char name[GGML_MAX_NAME];
    char desc[GGML_MAX_NAME];
    char lib[GGML_MAX_NAME];

    struct ggml_backend * backend;
    struct qcom_socinfo   socinfo;

    int n_threads;
    int dsp_thread_counts = 0;                  // actual worker threads in effect on DSP (max_hw_threads - 2)

    //Hexagon resource management for the general approach through Hexagon CDSP
    int domain_id;
    int session_id;
    remote_handle64 ggmlop_handle;
    size_t rpc_mempool_capacity;
    size_t rpc_mempool_len;
    size_t rpc_mempool_usage;
    int    rpc_mempool_handle;
    void * rpc_mempool;
    void * rpc_mempool_dsp_base;                // DSP-side VA from fastrpc_mmap() (NOT from FastRPC pointer translation)
    bool   weights_dirty;                       // set by set_tensor/memset_tensor, cleared by Phase 9
    std::vector<ion_pool_region> ion_regions;   // region tracking for mempool free-space management

    // FastRPC call statistics
    uint64_t rpc_batch_call_count;              // total ggml_dsp_execute_batch calls
    int64_t  cumulative_graph_us;               // cumulative graph inference duration
    int64_t  last_graph_end_us;                 // wall clock of last graph end (to measure gap)

    // Per-graph node statistics
    uint32_t max_nodes_per_graph;               // max node count in a single graph
    uint32_t min_nodes_per_graph;               // min node count in a single graph
    uint32_t total_nodes_processed;             // cumulative node count across all graphs

    // Per-call execution time range
    int64_t  min_graph_us;                      // shortest single graph execution
    int64_t  max_graph_us;                      // longest single graph execution
    uint32_t max_graph_n_nodes;                 // cgraph node count when max_graph_us recorded
    uint32_t max_graph_n_ops;                   // DSP op count (post-fusion) when max_graph_us recorded
    uint32_t min_n_ops_per_call;                // min hex_ops.size() across all graph_compute calls
    uint32_t max_n_ops_per_call;                // max hex_ops.size() across all graph_compute calls
    int64_t  min_p10_us;                        // shortest single FastRPC call
    int64_t  max_p10_us;                        // longest single FastRPC call

    // Per-call AP-side overhead (graph_dur - p10). Tracks how much time each
    // graph_compute_batch call spends outside of pure DSP execution
    int64_t  min_rpc_overhead_us;
    int64_t  max_rpc_overhead_us;
    int64_t  sum_rpc_overhead_us;

    // AP-side per-phase cumulative time
    int64_t  cum_p1_us;                         // Phase 1: collect unique tensor objects
    int64_t  cum_p2_us;                         // Phase 2: build op descriptors
    int64_t  cum_p3_us;                         // Phase 3: op fusion
    int64_t  cum_p4_us;                         // Phase 4: compute layout sizes
    int64_t  cum_p5_us;                         // Phase 5: tensor mirroring
    int64_t  cum_p6_us;                         // Phase 6: track repacked weight mempool offsets
    int64_t  cum_p7_us;                         // Phase 7: allocate batch descriptor in mempool
    int64_t  cum_p8_us;                         // Phase 8: descriptor construction
    int64_t  cum_p9_us;                         // Phase 9: cache flush

    // Phase 10: FastRPC doorbell call (cumulative) + 3-way breakdown
    int64_t  cum_p10_us;                        // cumulative FastRPC time
    // p10 3-way breakdown: split the FastRPC + DSP exec + cache inval window
    // so we can tell AP-side cache-coherency cost apart from DSP-side work
    int64_t  cum_p10_rpc_setup_us;              // AP setup before ggml_dsp_execute_batch (ioctl / marshalling)
    int64_t  cum_p10_dsp_exec_us;               // pure DSP execution time inside the sync call
    int64_t  cum_p10_civac_us;                  // AP cache invalidate after DSP reply

    int64_t  cum_p11_us;                        // Phase 11: cache inval
    int64_t  cum_p12_us;                        // Phase 12: mempool->heap copy-back
    int64_t  cum_unaccounted_us;                // wall-clock not covered by p1..p12 (gaps, scheduler, etc.)

    // FastRPC transport overhead calibration (measured via 0xFFFB warmup invokes at init)
    // The 0xFFFB warmup mode does no DSP work, so measured time is an upper bound of
    // pure FastRPC transport overhead (invoke round-trip: AP -> DSP -> AP)
    int64_t  rpc_overhead_min_us;               // shortest warmup invoke
    int64_t  rpc_overhead_max_us;               // longest warmup invoke
    int64_t  rpc_overhead_sum_us;               // sum of all warmup invokes (for avg)
    uint32_t rpc_overhead_count;                // number of warmup invokes measured

    // Cumulative MUL_MAT counters (PP optimization diagnostics)
    // Tracked in ctx so we can read rates after a sweep run without
    // changing existing per-call LOG_DEBUG output paths
    uint64_t n_mul_mat_total_cum = 0;           // total MUL_MAT ops in supported_nodes
    uint64_t n_hmx_used_cum      = 0;           // MUL_MAT dispatched to HMX kernels
    uint64_t n_fused_qkv_cum     = 0;           // 3x MUL_MAT -> HTP_OP_MUL_MAT_QKV fusions
    uint64_t n_fused_ffn_cum     = 0;           // 2x MUL_MAT -> HTP_OP_MUL_MAT_FFN fusions
    uint64_t n_fused_mm_add_cum  = 0;           // MUL_MAT + ADD -> HTP_OP_MUL_MAT_ADD fusions

    // HMX eligibility diagnostic counters (why MUL_MATs fall back to HVX)
    uint64_t n_hmx_basic_pass          = 0;     // passed basic HMX eligibility
    uint64_t n_hmx_basic_fail_ne01     = 0;     // ne01_padded %% 32 != 0
    uint64_t n_hmx_basic_fail_ne00     = 0;     // ne00 %% 32 != 0
    uint64_t n_hmx_basic_fail_wtype    = 0;     // weight type not HMX-compatible
    uint64_t n_hmx_basic_fail_batched  = 0;     // batched non-F16
    uint64_t n_hmx_basic_fail_permuted = 0;     // nb[0] > nb[1] (permuted)
    uint64_t n_hmx_basic_fail_small_n  = 0;     // ne11 <= HTP_MM_HMX_MIN_NROWS
    uint64_t n_hmx_vtcm_pass           = 0;     // HMX VTCM precompute succeeded
    uint64_t n_hmx_vtcm_fail           = 0;     // HMX VTCM precompute failed

    // Buffer type owned by this context (each device has its own buft)
    struct ggml_backend_buffer_type buffer_type;
    // Repack buffer type(is_host=false), same mempool as buffer_type
    struct ggml_backend_buffer_type repack_buffer_type;
    char buft_name[GGML_MAX_NAME];              // "hexagon-ion-buffer-<name>", unique per device
    char repack_buft_name[GGML_MAX_NAME];       // "hexagon-ion-buffer-<name>-REPACK"

    // Per-device hardware caps (probed at init, used by supports_op)
    bool has_vtcm;  // domain has VTCM pages available
    bool has_hvx;   // domain has HVX support
    bool has_hmx;   // domain has HMX support

    // Cached htp_mm_kernel_params per (weight_data, ne11). For TG, the
    // precompute math produces identical results for every token, so we
    // cache the params struct to skip the multi-hundred-microsecond
    // thread/chunk search on subsequent calls.
    std::unordered_map<uintptr_t, struct htp_mm_kernel_params> mm_params_cache;

    // cgraph cache: Phase 1 (tensor dedup) + Phase 2 (hex_ops build) +
    // Phase 3 (op fusion) result keyed by content-based cgraph hash.
    // The scheduler rebuilds split->graph every call, so the cgraph pointer
    // is NOT hashed. The underlying node ops/shapes/src/data ptrs are stable
    // for graph-reuse, which is what the hash covers.
    // A FNV-1a hash over {op, ne[4], nb[4], non-null src[0..GGML_MAX_SRC-1] ptr, data ptr}
    // per node gives a 64-bit key that is effectively collision-free.
    struct cgraph_cache_entry {
        uint64_t content_hash = 0;
        int n_nodes = 0;
        int n_tensors = 0;
        int n_ops = 0;
        std::vector<ggml_tensor *> tensor_src;
        std::vector<ggml_tensor *> supported_nodes;
        std::vector<hex_op_desc>   hex_ops;
        std::vector<uint8_t>       is_weight;   // per-tensor boolean
    };
    std::unordered_map<uint64_t, cgraph_cache_entry> cgraph_cache;
    uint64_t cgraph_cache_hits   = 0;
    uint64_t cgraph_cache_misses = 0;

    ggml_backend_hexagon_context(int dev_id, ggml_backend_dev_t dev);
    ~ggml_backend_hexagon_context();
};

struct hexagon_appcfg_t {
    int dump_debug_info;        // enable/disable dump debug info for troubleshooting issues on AP side
    int thread_counts;          // thread_counts on CDSP side
    int dump_diag_info;         // enable/disable dump diag info for troubleshooting issues on CDSP side
    int ndev;                   // number of Hexagon devices (PDs), from GGML_HEXAGON_NDEV env
    int ion_sync_mode;          // 0=both(DC CVAC+ion_sync), 1=ion_sync only(default), 2=DC CVAC only
    int rpc_mmap_mode;          // 0=FASTRPC_MAP_FD_DELAYED (default), 1=FASTRPC_MAP_FD (eager pinning)
    int enable_opfusion;        // 1=enable QKV/FFN op fusion (default), 0=disable (for debugging)
    int fa_select;              // flash attention: 2=HMX->HVX->CPU, 1=HVX->CPU, 0=CPU (default 2)
    int dsp_cache_mode;         // DSP-side entry.c cache optimization bitmask, pushed to DSP at init via
                                //   execute_batch(0xFFFC) special mode (no IDL change). All four bits
                                //   are wired into ggml_dsp_execute_batch(); dsp_cache_mode=0 is
                                //   behaviorally identical to baseline 29c1cf196.
                                //   bit 0 (0x1): first-touch weight bitmap
                                //   bit 1 (0x2): skip dcinva for prior dst
                                //   bit 2 (0x4): bulk dst flush at batch end
                                //   bit 3 (0x8): selective bulk flush - skip batch-end flush for
                                //     dsts still consumed by a later op in the same batch (pure
                                //     intermediates). Requires bit 2. Mirrored dsts (flags&0x1)
                                //     and final outputs always flush.
    int dsp_cache_trace_bit0;   // DSP-side bit 0 (first-touch weight) trace enable. 0=off (production),
                                //   1=emit one [DSP-CACHE-TRACE-BIT0] log line per bit 0 decision
                                //   (SKIP or INVAL) with op/src/ptr/len. Pushed to DSP at init via
                                //   bit 16 of the same execute_batch(0xFFFC) payload as dsp_cache_mode.
                                //   Used for diagnosing the bit 0 stale-L2-read bug observed on llama3
                                //   (33% prompt-repeat rate, 2026-07-10). Once root-caused this can
                                //   be removed.
    int dsp_cache_trace_bit1;   // DSP-side bit 1 (skip dcinva for prior dst) trace enable. 0=off
                                //   (production), 1=emit one [DSP-CACHE-TRACE-BIT1] log line per bit 1
                                //   decision (SKIP if prior_dst_contains_src, INVAL otherwise) with
                                //   op/src/ptr/len. Pushed to DSP at init via bit 17 of the same
                                //   execute_batch(0xFFFC) payload. Used for diagnosing why dsp_cache_mode
                                //   5/6/7 garble on the new matmul pipeline (81ff7abe5). Pair with
                                //   dsp_cache_trace_bit0 to localize the stale-L2-read culprit.
    int enable_graph_optimize;  // enable/disable cgraph reorder pass

    const char * cfgfilename;
    const char * runtime_libpath;
    char version[GGMLHEXAGON_TMPBUF_LEN];
    std::string enabled_ops;    // comma-separated list of ops to offload (empty = all supported ops)
    std::string enabled_types;  // comma-separated list of weight types to offload for MUL_MAT (empty = all supported types)
};

static struct hexagon_appcfg_t g_hexagon_appcfg = {
        .dump_debug_info        = 0,
        .thread_counts          = 6,
        .dump_diag_info         = 0,
        .ndev                   = 1,
        .ion_sync_mode          = 1,
        .rpc_mmap_mode          = 0,
        .enable_opfusion        = 1,
        .fa_select              = 2,
        .dsp_cache_mode         = 5,
        .dsp_cache_trace_bit0   = 0,
        .dsp_cache_trace_bit1   = 0,
        .enable_graph_optimize  = 1,
        .cfgfilename            = "ggml-hexagon.cfg",
#if defined(__ANDROID__)
        .runtime_libpath        = "/data/local/tmp/",
#elif defined(__linux__)
        .runtime_libpath        = "/tmp/",
#elif defined(_WIN32)
        .runtime_libpath        = "C:\\temp\\",
#endif
        .version                = {"0.99.4"},
        .enabled_ops            = "",
        .enabled_types          = "",
};

//supported Snapdragon devices with Hexagon DSP
static struct qcom_socinfo g_hexagon_soc_info_table[] = {
        /* Qualcomm SnapDragon 8 Gen 2 */
        {
                .soc_model         = SM8550,
                .htp_arch          = V73,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 8 Gen 2"},

        /* Qualcomm SnapDragon 8 Gen 3 */
        {
                .soc_model         = SM8650,
                .htp_arch          = V75,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 8 Gen 3 "},

        /* Qualcomm SnapDragon 8 Gen 4 */
        {
                .soc_model         = SM8750,
                .htp_arch          = V79,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 8 Elite"},

        /* Qualcomm SnapDragon 8 Gen 5 */
        {
                .soc_model         = SM8850,
                .htp_arch          = V81,
                .vtcm_size_in_mb   = 8,
                .soc_desc          = "Qualcomm SnapDragon 8 Elite Gen5"},
};

// Owning pointer to the reg context. The framework's ~ggml_backend_registry()
// does not delete reg->context (see FIXME in ggml-backend-reg.cpp), so we rely
// on an atexit handler to release DSP sessions. atexit runs before static
// dtors, so function-local std::mutex objects (e.g. the log mutex) are still
// alive when ~ggml_backend_hexagon_context calls ggmlhexagon_deinit_cdsp.
static ggml_backend_hexagon_reg_context * g_reg_ctx = nullptr;

// Static buffer for runtime libpath (updated via ggml_hexagon_set_runtime_libpath API)
static char g_runtime_libpath_buf[512] =
#if defined(__ANDROID__)
    "/data/local/tmp/"
#elif defined(__linux__)
    "/tmp/"
#elif defined(_WIN32)
    "C:\\temp\\"
#endif
;

// =================================================================================================
//  section-3: troubleshooting and profiler
// =================================================================================================
static void ggmlhexagon_get_timestring(char * p_currenttime) {
    if (nullptr == p_currenttime)
        return;

    auto time_to_string = [](const std::chrono::system_clock::time_point & tp)->std::string {
        auto as_time_t = std::chrono::system_clock::to_time_t(tp);
        struct tm tm;

        localtime_r(&as_time_t, &tm);

        std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch());
        char buf[GGMLHEXAGON_TMPBUF_LEN];
        memset(buf, 0, GGMLHEXAGON_TMPBUF_LEN);
        snprintf(buf, sizeof(buf), "%04d-%02d-%02d,%02d:%02d:%02d",
                 tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
        GGML_UNUSED(ms);
        return buf;
    };

    std::chrono::system_clock::time_point tp = std::chrono::system_clock::now();
    snprintf(p_currenttime, GGMLHEXAGON_TMPBUF_LEN, "%s", time_to_string(tp).c_str());
}

static void ggmlhexagon_log_internal(ggml_log_level level, const char * file, const char * func, int line, const char * format, ...) {
    static std::mutex ggmlhexagon_log_internal_mutex;
    static char s_ggmlhexagon_log_internal_buf[GGMLHEXAGON_LOGBUF_LEN];

    GGML_UNUSED(file);

    if (0 == g_hexagon_appcfg.dump_debug_info) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(ggmlhexagon_log_internal_mutex);
        va_list args;
        va_start(args, format);
        int len_prefix = snprintf(s_ggmlhexagon_log_internal_buf, GGMLHEXAGON_LOGBUF_LEN, "[%s, %d]: ", func, line);
        int len = vsnprintf(s_ggmlhexagon_log_internal_buf + len_prefix, GGMLHEXAGON_LOGBUF_LEN - len_prefix, format, args);
        if (len < (GGMLHEXAGON_LOGBUF_LEN - len_prefix)) {
#if (defined __ANDROID__) || (defined ANDROID)
            __android_log_print(ANDROID_LOG_INFO, PROJECT_NAME, "%s\n", s_ggmlhexagon_log_internal_buf);
            if (GGML_LOG_LEVEL_INFO == level || GGML_LOG_LEVEL_CONT == level) {
                printf("%s\n", s_ggmlhexagon_log_internal_buf);
            }
#else
            //for Snapdragon based WoA(Windows on ARM) device or Linux
            printf("%s\n", s_ggmlhexagon_log_internal_buf);
#endif
        }
        va_end(args);
    }
}

// Always-emit log channel. Bypasses dump_debug_info. The level parameter decides
// whether the message is also printed to stdout on Android:
//   - GGML_LOG_LEVEL_NONE  (ALWAYS)    -> adb logcat only
//   - GGML_LOG_LEVEL_ERROR             -> adb logcat + terminal
//   - GGML_LOG_LEVEL_CONT  (VERBOSE)   -> adb logcat + terminal
static void ggmlhexagon_log_always_internal(ggml_log_level level, const char * file, const char * func, int line, const char * format, ...) {
    static std::mutex s_log_mutex;
    static char s_log_buf[GGMLHEXAGON_LOGBUF_LEN];

    GGML_UNUSED(file);

    {
        std::lock_guard<std::mutex> lock(s_log_mutex);
        va_list args;
        va_start(args, format);
        int len_prefix = snprintf(s_log_buf, GGMLHEXAGON_LOGBUF_LEN, "[%s, %d]: ", func, line);
        int len = vsnprintf(s_log_buf + len_prefix, GGMLHEXAGON_LOGBUF_LEN - len_prefix, format, args);
        if (len < (GGMLHEXAGON_LOGBUF_LEN - len_prefix)) {
#if (defined __ANDROID__) || (defined ANDROID)
            __android_log_print(ANDROID_LOG_INFO, PROJECT_NAME, "%s\n", s_log_buf);
            if (GGML_LOG_LEVEL_ERROR == level || GGML_LOG_LEVEL_CONT == level) {
                printf("%s\n", s_log_buf);
            }
#else
            //for Snapdragon based WoA(Windows on ARM) device or Linux
            printf("%s\n", s_log_buf);
#endif
        }
        va_end(args);
    }
}

// Invoke one no-op warmup call and record FastRPC transport overhead timing.
// The 0xFFFB warmup mode does no DSP work, so measured time is an upper bound
// of pure FastRPC transport overhead. Used at init only, no per-graph overhead.
static int hexagon_warmup_invoke_timed(ggml_backend_hexagon_context * ctx) {
    int64_t t0 = ggml_time_us();
    // Use the existing no-op warmup mode (0xFFFB) instead of the legacy
    // batch_size==0 probe mode so no marker bytes are written to the pool.
    int err = ggml_dsp_execute_batch(ctx->ggmlop_handle, 0, 0xFFFB);
    int64_t dt = ggml_time_us() - t0;
    ctx->rpc_overhead_sum_us += dt;
    ctx->rpc_overhead_count++;
    if (ctx->rpc_overhead_min_us == 0 || dt < ctx->rpc_overhead_min_us) ctx->rpc_overhead_min_us = dt;
    if (dt > ctx->rpc_overhead_max_us)                                  ctx->rpc_overhead_max_us = dt;
    return err;
}

static void ggmlhexagon_print_running_timestamp(ggml_backend_hexagon_context * ctx) {
    char timestamp[GGMLHEXAGON_TMPBUF_LEN];
    memset(timestamp, 0, GGMLHEXAGON_TMPBUF_LEN);

    GGMLHEXAGON_LOG_VERBOSE("ggml_hexagon_version:             %s", g_hexagon_appcfg.version);
    ggmlhexagon_get_timestring(timestamp);
    GGMLHEXAGON_LOG_VERBOSE("offload MUL_MAT types:            %s", g_hexagon_appcfg.enabled_types.empty() ? "ALL" : g_hexagon_appcfg.enabled_types.c_str());
    GGMLHEXAGON_LOG_VERBOSE("thread_counts on CDSP:            %d", g_hexagon_appcfg.thread_counts);
    GGMLHEXAGON_LOG_VERBOSE("ion_sync_mode:                    %d", g_hexagon_appcfg.ion_sync_mode);
    GGMLHEXAGON_LOG_VERBOSE("rpc_mmap_mode:                    %d", g_hexagon_appcfg.rpc_mmap_mode);
    GGMLHEXAGON_LOG_VERBOSE("dsp_cache_mode:                   %d", g_hexagon_appcfg.dsp_cache_mode);
    GGMLHEXAGON_LOG_VERBOSE("dsp_cache_trace_bit0:             %d", g_hexagon_appcfg.dsp_cache_trace_bit0);
    GGMLHEXAGON_LOG_VERBOSE("dsp_cache_trace_bit1:             %d", g_hexagon_appcfg.dsp_cache_trace_bit1);
    GGMLHEXAGON_LOG_VERBOSE("dump diag info(DSP):              %d", g_hexagon_appcfg.dump_diag_info);
    GGMLHEXAGON_LOG_VERBOSE("dump diag info(AP):               %d", g_hexagon_appcfg.dump_debug_info);
    GGMLHEXAGON_LOG_VERBOSE("enable graph_optimize:            %d", g_hexagon_appcfg.enable_graph_optimize);
    if (NULL != ctx) {
        GGMLHEXAGON_LOG_VERBOSE("ggml-dsp use hmx:                 %d", ctx->has_hmx);
    }
    GGMLHEXAGON_LOG_VERBOSE("enabled_ops:                      %s", "ALL");
    GGMLHEXAGON_LOG_VERBOSE("running timestamp:%s", timestamp);
}

// dump accumulated performance statistics collected during graph_compute_batch
static void ggmlhexagon_dump_perf_stats(const ggml_backend_hexagon_context * ctx) {
    if (nullptr == ctx) {
        return;
    }
    // Logging convention in this function (inverted from the usual level names):
    //   VERBOSE -> terminal + adb logcat (key summary for immediate visibility)
    //   ALWAYS  -> adb logcat only (detailed diagnostics for post-analysis)
    GGMLHEXAGON_LOG_VERBOSE("device=%d name=%s arch=%s vtcm=%zuMB hvx=%d hmx=%d",
                             ctx->device, ctx->name,
                             ggmlhexagon_get_htparch_desc(ctx->socinfo.htp_arch),
                             ctx->socinfo.vtcm_size_in_mb,
                             (int)ctx->has_hvx, (int)ctx->has_hmx);
    GGMLHEXAGON_LOG_VERBOSE("rpc stats: batch_calls=%llu cum_p10=%lld us cum_graph=%lld us avg_p10=%lld us avg_graph=%lld us",
                             (unsigned long long)ctx->rpc_batch_call_count,
                             (long long)ctx->cum_p10_us, (long long)ctx->cumulative_graph_us,
                             ctx->rpc_batch_call_count ? (long long)(ctx->cum_p10_us / (int64_t)ctx->rpc_batch_call_count) : 0,
                             ctx->rpc_batch_call_count ? (long long)(ctx->cumulative_graph_us / (int64_t)ctx->rpc_batch_call_count) : 0);
    GGMLHEXAGON_LOG_VERBOSE("graph nodes: min=%u max=%u total=%u",
                             ctx->min_nodes_per_graph, ctx->max_nodes_per_graph, ctx->total_nodes_processed);
    GGMLHEXAGON_LOG_VERBOSE("graph ops (post-fusion): min=%u max=%u",
                             ctx->min_n_ops_per_call, ctx->max_n_ops_per_call);
    GGMLHEXAGON_LOG_VERBOSE("per-call range: graph=[%lld, %lld] us p10=[%lld, %lld] us",
                             (long long)ctx->min_graph_us, (long long)ctx->max_graph_us,
                             (long long)ctx->min_p10_us, (long long)ctx->max_p10_us);
    GGMLHEXAGON_LOG_VERBOSE("per-call overhead: n=%llu min=%lld max=%lld avg=%lld us (graph_dur - p10)",
                             (unsigned long long)ctx->rpc_batch_call_count,
                             (long long)ctx->min_rpc_overhead_us,
                             (long long)ctx->max_rpc_overhead_us,
                             ctx->rpc_batch_call_count ? (long long)(ctx->sum_rpc_overhead_us / (int64_t)ctx->rpc_batch_call_count) : 0);
    GGMLHEXAGON_LOG_VERBOSE("max graph detail: dur=%lld us n_nodes=%u n_ops=%u",
                             (long long)ctx->max_graph_us, ctx->max_graph_n_nodes, ctx->max_graph_n_ops);
    GGMLHEXAGON_LOG_VERBOSE("AP phase cumulative: p1=%lld p2=%lld p3=%lld p4=%lld p5=%lld p6=%lld p7=%lld p8=%lld p9=%lld p11=%lld p12=%lld unaccounted=%lld us",
                             (long long)ctx->cum_p1_us, (long long)ctx->cum_p2_us,
                             (long long)ctx->cum_p3_us, (long long)ctx->cum_p4_us,
                             (long long)ctx->cum_p5_us, (long long)ctx->cum_p6_us,
                             (long long)ctx->cum_p7_us,
                             (long long)ctx->cum_p8_us, (long long)ctx->cum_p9_us,
                             (long long)ctx->cum_p11_us, (long long)ctx->cum_p12_us,
                             (long long)ctx->cum_unaccounted_us);
    // Fine-grained: 3-way p10 split + per-call distribution
    GGMLHEXAGON_LOG_VERBOSE("p10 3-way cumulative: rpc_setup=%lld dsp_exec=%lld civac=%lld us (sum=%lld)",
                             (long long)ctx->cum_p10_rpc_setup_us,
                             (long long)ctx->cum_p10_dsp_exec_us,
                             (long long)ctx->cum_p10_civac_us,
                             (long long)(ctx->cum_p10_rpc_setup_us + ctx->cum_p10_dsp_exec_us + ctx->cum_p10_civac_us));
    GGMLHEXAGON_LOG_VERBOSE("rpc overhead (warmup): n=%u min=%lld max=%lld avg=%lld us (upper bound, pure FastRPC/mempool transport overhead)",
                             ctx->rpc_overhead_count,
                             (long long)ctx->rpc_overhead_min_us, (long long)ctx->rpc_overhead_max_us,
                             ctx->rpc_overhead_count ? (long long)(ctx->rpc_overhead_sum_us / (int64_t)ctx->rpc_overhead_count) : 0);
    const uint64_t total_cache_lookups = ctx->cgraph_cache_hits + ctx->cgraph_cache_misses;
    GGMLHEXAGON_LOG_VERBOSE("cgraph cache: hits=%llu misses=%llu (hit_rate=%.1f%%) entries=%zu",
                             (unsigned long long)ctx->cgraph_cache_hits,
                             (unsigned long long)ctx->cgraph_cache_misses,
                             total_cache_lookups ? (100.0 * ctx->cgraph_cache_hits / total_cache_lookups) : 0.0,
                             ctx->cgraph_cache.size());

    // MUL_MAT optimization diagnostics (PP). Cumulative across the run.
    // - n_mul_mat_total: every MUL_MAT in supported_nodes (cache miss only)
    // - n_hmx_used:      MUL_MAT dispatched to HMX (kparams.n_hmx == 1)
    // - n_fused_qkv:     3x MUL_MAT (Q,K,V) merged into HTP_OP_MUL_MAT_QKV
    // - n_fused_ffn:     2x MUL_MAT (gate,up) merged into HTP_OP_MUL_MAT_FFN
    // - n_fused_mm_add:  MUL_MAT + ADD merged into HTP_OP_MUL_MAT_ADD
    {
        const double total = (double) ctx->n_mul_mat_total_cum;
        const double hmx_pct  = total > 0 ? 100.0 * ctx->n_hmx_used_cum      / total : 0.0;
        const double qkv_pct  = total > 0 ? 100.0 * (3 * ctx->n_fused_qkv_cum) / total : 0.0;
        const double ffn_pct  = total > 0 ? 100.0 * (2 * ctx->n_fused_ffn_cum) / total : 0.0;
        const double add_pct  = total > 0 ? 100.0 * ctx->n_fused_mm_add_cum   / total : 0.0;
        GGMLHEXAGON_LOG_ALWAYS("mul_mat coverage: total=%llu hmx=%llu (%.1f%%) qkv_fused=%llu (saves %.1f%%) "
                               "ffn_fused=%llu (saves %.1f%%) mm_add_fused=%llu (saves %.1f%%)",
                               (unsigned long long)ctx->n_mul_mat_total_cum,
                               (unsigned long long)ctx->n_hmx_used_cum, hmx_pct,
                               (unsigned long long)ctx->n_fused_qkv_cum, qkv_pct,
                               (unsigned long long)ctx->n_fused_ffn_cum, ffn_pct,
                               (unsigned long long)ctx->n_fused_mm_add_cum, add_pct);
    }

    // HMX eligibility diagnostic: why MUL_MATs fall back to HVX
    {
        uint64_t basic_total = ctx->n_hmx_basic_pass
                             + ctx->n_hmx_basic_fail_ne01
                             + ctx->n_hmx_basic_fail_ne00
                             + ctx->n_hmx_basic_fail_wtype
                             + ctx->n_hmx_basic_fail_batched
                             + ctx->n_hmx_basic_fail_permuted
                             + ctx->n_hmx_basic_fail_small_n;
        if (basic_total > 0) {
            GGMLHEXAGON_LOG_ALWAYS("hmx eligibility: total=%llu pass=%llu (%.1f%%)",
                                     (unsigned long long)basic_total,
                                     (unsigned long long)ctx->n_hmx_basic_pass,
                                     basic_total > 0 ? 100.0 * ctx->n_hmx_basic_pass / basic_total : 0.0);
            GGMLHEXAGON_LOG_ALWAYS("hmx basic fail: ne01_align=%llu ne00_align=%llu wtype=%llu batched=%llu permuted=%llu small_n=%llu",
                                     (unsigned long long)ctx->n_hmx_basic_fail_ne01,
                                     (unsigned long long)ctx->n_hmx_basic_fail_ne00,
                                     (unsigned long long)ctx->n_hmx_basic_fail_wtype,
                                     (unsigned long long)ctx->n_hmx_basic_fail_batched,
                                     (unsigned long long)ctx->n_hmx_basic_fail_permuted,
                                     (unsigned long long)ctx->n_hmx_basic_fail_small_n);
            GGMLHEXAGON_LOG_ALWAYS("hmx vtcm: pass=%llu fail=%llu (%.1f%% of basic-pass)",
                                     (unsigned long long)ctx->n_hmx_vtcm_pass,
                                     (unsigned long long)ctx->n_hmx_vtcm_fail,
                                     ctx->n_hmx_basic_pass > 0
                                        ? 100.0 * ctx->n_hmx_vtcm_fail / ctx->n_hmx_basic_pass
                                        : 0.0);
        }
    }
}

// =================================================================================================
//  section-4: configuration class and helper functions
// =================================================================================================
//a simple class to load running configurations in ggml-hexagon.cfg
class hexagon_appcfg {
public:
    hexagon_appcfg() {}

    void dump(std::function<void(const std::string &, const std::string &, const std::string &)> worker) {
        if (!_load_success) {
            GGMLHEXAGON_LOG_WARN("hexagon cfg file %s not loaded", _cfg_filename.c_str());
            return;
        }
        auto iter = _hexagon_appcfg.begin();
        while (iter != _hexagon_appcfg.end()) {
            auto kv_iter = iter->second.begin();
            while (kv_iter != iter->second.end()) {
                worker(iter->first, kv_iter->first, kv_iter->second);
                ++kv_iter;
            }
            ++iter;
        }
    }

    bool load(const std::string & file_name) {
        if (file_name == "") {
            return false;
        }
        _cfg_filename = file_name;
        std::ifstream in;
        std::string line;
        in.open(file_name.c_str());
        if (not in.is_open()) {
            GGMLHEXAGON_LOG_WARN("can't open file %s", file_name.c_str());
            return false;
        }
        while (getline(in, line)) {
            std::string section, key, value;
            if (not parse_line(line, section, key, value)) {
                continue;
            }
            set_section_keyvalue(section, key, value);
        }
        _load_success = true;
        return true;
    }

    void get_stringvalue(const std::string & section, const std::string & key, std::string & value, std::string default_value) {
        value = default_value;
        if (_hexagon_appcfg.find(section) == _hexagon_appcfg.end()) {
            return;
        }
        if (_hexagon_appcfg[section].find(key) == _hexagon_appcfg[section].end()) {
            return;
        }
        value = _hexagon_appcfg[section][key];
    }

    void get_intvalue(const std::string & section, const std::string & key, int & value, int default_value) {
        value = default_value;
        if (_hexagon_appcfg.find(section) == _hexagon_appcfg.end()) {
            return;
        }
        if (_hexagon_appcfg[section].find(key) == _hexagon_appcfg[section].end()) {
            return;
        }
        value = atol(_hexagon_appcfg[section][key].c_str());
    }

private:
    void ltrim(std::string & str) {
        if (str.empty()) return;
        size_t len  = 0;
        const char * temp = str.c_str();
        while (*temp && isblank(*temp)) {
            ++len;
            ++temp;
        }
        if (len > 0) str.erase(0, len);
    }

    void rtrim(std::string & str) {
        if (str.empty()) return;
        size_t len = str.length();
        size_t pos = len;
        while (pos > 0) {
            if (not isblank(str[pos - 1])) {
                break;
            }
            --pos;
        }
        if (pos != len) str.erase(pos);
    }

    void trim(std::string & str) {
        ltrim(str);
        rtrim(str);
    }

    void set_section_keyvalue(std::string & section, std::string & key, std::string & value) {
        if (_hexagon_appcfg.find(section) == _hexagon_appcfg.end()) {
            std::unordered_map<std::string, std::string> kv_map;
            _hexagon_appcfg[section] = kv_map;
        }
        if (key != "" && value != "") _hexagon_appcfg[section][key] = value;
    }

    bool parse_line(std::string & line, std::string & section, std::string & key, std::string & value) {
        static std::string cur_section = "";
        std::string nodes[2] = {"#", ";"};
        for (int i = 0; i < 2; ++i) {
            std::string::size_type pos = line.find(nodes[i]);
            if (pos != std::string::npos) line.erase(pos);
        }
        trim(line);
        if (line == "") return false;
        if (line[0] == '[' && line[line.size() - 1] == ']') {
            section = line.substr(1, line.size() - 2);
            trim(section);
            cur_section = section;
            return false;
        }
        if (cur_section == "") return false;
        bool is_key = true;
        for (size_t i = 0; i < line.size(); ++i) {
            if (line[i] == '=') {
                is_key = false;
                continue;
            }
            if (is_key) {
                key += line[i];
            } else {
                value += line[i];
            }
        }
        section = cur_section;
        trim(key);
        trim(value);

        //"1.00" -> 1.00
        if (value.front() == '"' && value.back() == '"') {
            value.erase(0, 1); // erase the first character "
            value.erase(value.size() - 1); // erase the last character "
        }

        return true;
    }

private:
    hexagon_appcfg(const hexagon_appcfg & ) = delete;
    hexagon_appcfg(const hexagon_appcfg && ) = delete;
    hexagon_appcfg & operator= (const hexagon_appcfg & ) = delete;

private:
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> _hexagon_appcfg;
    bool _load_success = false;
    std::string _cfg_filename;
};

static void ggmlhexagon_load_cfg() {
    //this function can be called in various scenarios
    static bool initialized = false;
    if (initialized) {
        GGMLHEXAGON_LOG_DEBUG("hexagon appcfg file already loaded\n");
        return;
    }
    char time_string[GGMLHEXAGON_TMPBUF_LEN];
    memset(time_string, 0, GGMLHEXAGON_TMPBUF_LEN);
    ggmlhexagon_get_timestring(time_string);
    GGMLHEXAGON_LOG_DEBUG("program running start time:%s", time_string);
    std::string cfg_filename = std::string(g_hexagon_appcfg.runtime_libpath) + std::string(g_hexagon_appcfg.cfgfilename);

    // Prefer user-editable copy in /data/local/tmp/ (accessible via adb shell),
    // fall back to runtime_libpath/ggml-hexagon.cfg (such as app data dir in Android APK).
#if defined(__ANDROID__)
    {
        std::string user_cfg = "/data/local/tmp/" + std::string(g_hexagon_appcfg.cfgfilename);
        if (access(user_cfg.c_str(), F_OK) == 0) {
            cfg_filename = user_cfg;
        }
    }
#endif

    hexagon_appcfg hexagoncfg_instance;
    hexagoncfg_instance.load(cfg_filename);
    hexagoncfg_instance.dump([](const std::string & section, const std::string & key, const std::string value) {
        std::ostringstream  tmposs;
        tmposs << "section[" << std::setw(10) << std::left << section << "],[" << std::setw(25) << std::left << key << "] = [" << value << "]";
        GGMLHEXAGON_LOG_INFO("%s", tmposs.str().c_str());
    });
    std::string version; //version of ggml-hexagon
    hexagoncfg_instance.get_stringvalue("general", "version", version, "0.99");
    hexagoncfg_instance.get_intvalue("general", "dump_debug_info", g_hexagon_appcfg.dump_debug_info, 0);

    hexagoncfg_instance.get_intvalue("cdsp", "thread_counts", g_hexagon_appcfg.thread_counts, 4);
    hexagoncfg_instance.get_intvalue("cdsp", "dump_diag_info", g_hexagon_appcfg.dump_diag_info, 0);
    hexagoncfg_instance.get_intvalue("cdsp", "ndev", g_hexagon_appcfg.ndev, 1);
    hexagoncfg_instance.get_intvalue("cdsp", "ion_sync_mode", g_hexagon_appcfg.ion_sync_mode, 1);
    hexagoncfg_instance.get_intvalue("cdsp", "rpc_mmap_mode", g_hexagon_appcfg.rpc_mmap_mode, 0);
    hexagoncfg_instance.get_intvalue("cdsp", "enable_opfusion", g_hexagon_appcfg.enable_opfusion, 1);
    hexagoncfg_instance.get_intvalue("cdsp", "fa_select", g_hexagon_appcfg.fa_select, 2);
    hexagoncfg_instance.get_intvalue("cdsp", "dsp_cache_mode", g_hexagon_appcfg.dsp_cache_mode, 5);
    hexagoncfg_instance.get_intvalue("cdsp", "dsp_cache_trace_bit0", g_hexagon_appcfg.dsp_cache_trace_bit0, 0);
    hexagoncfg_instance.get_intvalue("cdsp", "dsp_cache_trace_bit1", g_hexagon_appcfg.dsp_cache_trace_bit1, 0);
    hexagoncfg_instance.get_intvalue("cdsp", "enable_graph_optimize", g_hexagon_appcfg.enable_graph_optimize, 1);
    hexagoncfg_instance.get_stringvalue("cdsp", "enabled_ops", g_hexagon_appcfg.enabled_ops, "");
    hexagoncfg_instance.get_stringvalue("cdsp", "enabled_types", g_hexagon_appcfg.enabled_types, "");

    memcpy(g_hexagon_appcfg.version, version.c_str(), strlen(version.c_str()));

    GGMLHEXAGON_LOG_ALWAYS("load hexagon appcfg from %s", cfg_filename.c_str());
    GGMLHEXAGON_LOG_ALWAYS("ggml_hexagon_version=%s", g_hexagon_appcfg.version);
    GGMLHEXAGON_LOG_ALWAYS("runtime libpath=%s", g_hexagon_appcfg.runtime_libpath);

    // env var GGML_HEXAGON_NDEV overrides cfg value (for automation/testing)
    const char * str_ndev = getenv("GGML_HEXAGON_NDEV");
    if (str_ndev) {
        int v = atoi(str_ndev);
        if (v > 0 && v <= GGML_HEXAGON_MAX_DEVICES) {
            g_hexagon_appcfg.ndev = v;
        } else {
            GGMLHEXAGON_LOG_WARN("invalid GGML_HEXAGON_NDEV=%d, must be 1..%d, using cfg value %d",
                                 v, GGML_HEXAGON_MAX_DEVICES, g_hexagon_appcfg.ndev);
        }
    }
    if (g_hexagon_appcfg.ndev < 1 || g_hexagon_appcfg.ndev > GGML_HEXAGON_MAX_DEVICES) {
        GGMLHEXAGON_LOG_WARN("invalid ndev=%d from cfg, must be 1..%d, using default 1",
                             g_hexagon_appcfg.ndev, GGML_HEXAGON_MAX_DEVICES);
        g_hexagon_appcfg.ndev = 1;
    }
    GGMLHEXAGON_LOG_ALWAYS("ndev=%d (from cfg, env GGML_HEXAGON_NDEV overrides if set)", g_hexagon_appcfg.ndev);

    ggmlhexagon_set_runtime_path(0, g_hexagon_appcfg.runtime_libpath);

    initialized = true;
}

static bool ggmlhexagon_check_valid_appcfg() {
    if (g_hexagon_appcfg.thread_counts > 6) {
        GGMLHEXAGON_LOG_WARN("invalid thread_counts %d, reset to 6", g_hexagon_appcfg.thread_counts);
        g_hexagon_appcfg.thread_counts = 6;
    }

    if (g_hexagon_appcfg.dump_diag_info > 1) {
        GGMLHEXAGON_LOG_WARN("invalid dump_diag_info %d, reset to 0", g_hexagon_appcfg.dump_diag_info);
        g_hexagon_appcfg.dump_diag_info = 0;
    }

    return true;
}

// Check if a ggml_type is allowed by the enabled_types config filter
// Returns true if the type is in the enabled list, or if the list is empty (all types allowed)
// Only applies to quantized types and F16/BF16; F32 is always allowed
static bool ggmlhexagon_type_is_enabled(enum ggml_type type) {
    if (g_hexagon_appcfg.enabled_types.empty()) {
        return true;
    }
    // F32 is always allowed
    if (type == GGML_TYPE_F32) {
        return true;
    }
    const char * type_name = ggml_type_name(type);
    if (type_name == NULL) {
        return false;
    }
    // Check if type_name appears as a whole word in the comma-separated list
    const std::string & list = g_hexagon_appcfg.enabled_types;
    size_t pos = 0;
    while (pos < list.size()) {
        size_t end = list.find(',', pos);
        if (end == std::string::npos) end = list.size();
        std::string token = list.substr(pos, end - pos);
        // trim whitespace
        size_t start = token.find_first_not_of(" \t");
        size_t last = token.find_last_not_of(" \t");
        if (start != std::string::npos && last != std::string::npos) {
            token = token.substr(start, last - start + 1);
        }
        // "all" keyword enables all types
        if (token.size() == 3 &&
            tolower((unsigned char)token[0]) == 'a' &&
            tolower((unsigned char)token[1]) == 'l' &&
            tolower((unsigned char)token[2]) == 'l') {
            return true;
        }
        // case-insensitive compare (ggml_type_name returns lowercase, cfg may use Q4_0 or q4_0)
        if (token.size() == strlen(type_name)) {
            bool match = true;
            for (size_t i = 0; i < token.size(); ++i) {
                if (tolower((unsigned char)token[i]) != tolower((unsigned char)type_name[i])) {
                    match = false;
                    break;
                }
            }
            if (match) return true;
        }
        pos = end + 1;
    }
    return false;
}

// Set the runtime library path where DSP skeleton .so files (libggmldsp-skel-v*.so)
// and ggml-hexagon.cfg are located. Must be called before any hexagon backend
// registration to take effect.
GGML_BACKEND_API void ggml_hexagon_set_runtime_libpath(const char * path) {
    if (path == nullptr) {
        return;
    }
    strncpy(g_runtime_libpath_buf, path, sizeof(g_runtime_libpath_buf) - 1);
    g_runtime_libpath_buf[sizeof(g_runtime_libpath_buf) - 1] = '\0';
    g_hexagon_appcfg.runtime_libpath = g_runtime_libpath_buf;
}

// =================================================================================================
//  section-5: general helper functions
// =================================================================================================
static int ion_sync_for_direction(int fd, int direction) {
#if defined(__ANDROID__) || defined(__linux__)
    if (fd <= 0) return -1;
    // Flag definitions from Linux kernel include/uapi/linux/dma-buf.h:
    //   DMA_BUF_SYNC_READ  = (1 << 0) = 1
    //   DMA_BUF_SYNC_WRITE = (2 << 0) = 2
    //   DMA_BUF_SYNC_START = (1 << 2) = 4
    //   DMA_BUF_SYNC_END   = (2 << 2) = 8
    {
        static const uint64_t DMA_BUF_SYNC_READ  = (1u << 0);
        static const uint64_t DMA_BUF_SYNC_WRITE = (2u << 0);
        static const uint64_t DMA_BUF_SYNC_START = (1u << 2);
        static const uint64_t DMA_BUF_SYNC_END   = (2u << 2);
        uint64_t rw = (direction == 1) ? DMA_BUF_SYNC_WRITE : DMA_BUF_SYNC_READ;
        struct { uint64_t flags; } s;
        s.flags = DMA_BUF_SYNC_START | rw;
        int r = ioctl(fd, DMA_BUF_IOCTL_SYNC_IOCTL, &s);
        if (r == 0) {
            s.flags = DMA_BUF_SYNC_END | rw;
            ioctl(fd, DMA_BUF_IOCTL_SYNC_IOCTL, &s);
            static int logged = 0;
            if (!logged) { GGMLHEXAGON_LOG_WARN("DMA_BUF_IOCTL_SYNC(%s) OK fd=%d (kernel cache sync)", direction ? "WRITE" : "READ", fd); logged = 1; }
            return 0;
        }
        static int logged_fail = 0;
        if (!logged_fail) { GGMLHEXAGON_LOG_WARN("DMA_BUF_IOCTL_SYNC(%s) FAILED fd=%d errno=%d (%s)", direction ? "WRITE" : "READ", fd, errno, strerror(errno)); logged_fail = 1; }
    }
    {
        struct ion_sync_data { int fd; unsigned int flags; unsigned int pad; };
        struct ion_sync_data sync = { fd, (unsigned int)direction, 0 };
        int r = ioctl(fd, _IOWR('I', 7, struct ion_sync_data), &sync);
        if (r == 0) {
            static int logged = 0;
            if (!logged) { GGMLHEXAGON_LOG_WARN("ION_IOC_SYNC(%s) fallback OK fd=%d", direction ? "WRITE" : "READ", fd); logged = 1; }
            return 0;
        }
        static int logged_fail2 = 0;
        if (!logged_fail2) { GGMLHEXAGON_LOG_WARN("ION_IOC_SYNC(%s) fallback FAILED fd=%d errno=%d (%s)", direction ? "WRITE" : "READ", fd, errno, strerror(errno)); logged_fail2 = 1; }
    }
#elif defined(_WIN32)
    // WoA: ION allocator and DMA_BUF_IOCTL_SYNC are not available on Windows.
    // Cache sync is handled by dspqueue's platform-specific driver implementation.
    GGML_UNUSED(fd); GGML_UNUSED(direction);
#endif
    return -1;
}

static inline void cpu_dcache_flush_range(ggml_backend_hexagon_context * backend_ctx, int ion_fd, const void * p, size_t size) {
#if defined(__ANDROID__) || defined(__linux__)
    // range-based DC CVAC with 8x loop unrolling
    if (size == 0) return;
    {
        const size_t line_size = 64;
        const char * start = (const char *)((uintptr_t)p & ~(line_size - 1));
        const char * end   = (const char *)p + size;
        // 8x unrolled: 8 cache lines per iteration
        for (; start + line_size * 8 <= end; start += line_size * 8) {
            __asm__ volatile("dc cvac, %0" : : "r"((const void *)(start + line_size * 0)) : "memory");
            __asm__ volatile("dc cvac, %0" : : "r"((const void *)(start + line_size * 1)) : "memory");
            __asm__ volatile("dc cvac, %0" : : "r"((const void *)(start + line_size * 2)) : "memory");
            __asm__ volatile("dc cvac, %0" : : "r"((const void *)(start + line_size * 3)) : "memory");
            __asm__ volatile("dc cvac, %0" : : "r"((const void *)(start + line_size * 4)) : "memory");
            __asm__ volatile("dc cvac, %0" : : "r"((const void *)(start + line_size * 5)) : "memory");
            __asm__ volatile("dc cvac, %0" : : "r"((const void *)(start + line_size * 6)) : "memory");
            __asm__ volatile("dc cvac, %0" : : "r"((const void *)(start + line_size * 7)) : "memory");
        }
        // tail: remaining lines
        for (; start < end; start += line_size) {
            __asm__ volatile("dc cvac, %0" : : "r"((const void *)start) : "memory");
        }
        __asm__ volatile("dsb ish" ::: "memory");
    }
    if (ion_fd > 0) ion_sync_for_direction(ion_fd, 1);
#elif defined(_WIN32)
    // WoA: MSVC does not support GCC inline asm for ARM64.
    // Cache sync is handled by dspqueue's platform-specific driver implementation.
    GGML_UNUSED(backend_ctx); GGML_UNUSED(ion_fd); GGML_UNUSED(p); GGML_UNUSED(size);
#endif
}

static inline void cpu_dcache_inval_range(ggml_backend_hexagon_context * backend_ctx, int ion_fd, const void * p, size_t size) {
#if defined(__ANDROID__) || defined(__linux__)
    // range-based DC CIVAC with 8x loop unrolling
    if (size == 0) return;
    {
        const size_t line_size = 64;
        const char * start = (const char *)((uintptr_t)p & ~(line_size - 1));
        const char * end   = (const char *)p + size;
        // 8x unrolled: 8 cache lines per iteration
        for (; start + line_size * 8 <= end; start += line_size * 8) {
            __asm__ volatile("dc civac, %0" : : "r"((const void *)(start + line_size * 0)) : "memory");
            __asm__ volatile("dc civac, %0" : : "r"((const void *)(start + line_size * 1)) : "memory");
            __asm__ volatile("dc civac, %0" : : "r"((const void *)(start + line_size * 2)) : "memory");
            __asm__ volatile("dc civac, %0" : : "r"((const void *)(start + line_size * 3)) : "memory");
            __asm__ volatile("dc civac, %0" : : "r"((const void *)(start + line_size * 4)) : "memory");
            __asm__ volatile("dc civac, %0" : : "r"((const void *)(start + line_size * 5)) : "memory");
            __asm__ volatile("dc civac, %0" : : "r"((const void *)(start + line_size * 6)) : "memory");
            __asm__ volatile("dc civac, %0" : : "r"((const void *)(start + line_size * 7)) : "memory");
        }
        // tail: remaining lines
        for (; start < end; start += line_size) {
            __asm__ volatile("dc civac, %0" : : "r"((const void *)start) : "memory");
        }
        __asm__ volatile("dsb ish" ::: "memory");
        __asm__ volatile("isb" ::: "memory");
    }
    if (ion_fd > 0) ion_sync_for_direction(ion_fd, 0);
#elif defined(_WIN32)
    // WoA: MSVC does not support GCC inline asm for ARM64.
    // Cache sync is handled by dspqueue's platform-specific driver implementation.
    GGML_UNUSED(backend_ctx); GGML_UNUSED(ion_fd); GGML_UNUSED(p); GGML_UNUSED(size);
#endif
}

// True for metadata-only ops that never execute on CDSP.
// Tests iterate every tensor in the graph and call supports_op on each;
// view/reshape/permute parents must be reported as supported.
static bool ggmlhexagon_is_metadata_op(enum ggml_op op) {
    switch (op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_REPEAT:
            return true;
        default:
            return false;
    }
}

static const char * ggmlhexagon_get_socmodel_desc(uint32_t soc_model) {
    switch (soc_model) {
        case SM8550:
            return "SM8550";
        case SM8650:
            return "SM8650";
        case SM8750:
            return "SM8750";
        case SM8850:
            return "SM8850";
        default:
            return "unknown";
    }
}

//0x73 -> 73, 0x75 -> 75, 0x79 -> 79, 0x81 -> 81
static size_t ggmlhexagon_htparch_hex_to_decimal(size_t htp_arch) {
    //naive algorithm
    int a = htp_arch / 16;
    int b = htp_arch % 16;
    return a * 10 + b;
}

static const char * ggmlhexagon_get_htparch_desc(size_t htp_arch) {
    switch (htp_arch) {
        case V73:
            return "QCOM_HTP_V73";
        case V75:
            return "QCOM_HTP_V75";
        case V79:
            return "QCOM_HTP_V79";
        case V81:
            return "QCOM_HTP_V81";
        default:
            return "unknown";
    }
}

static struct qcom_socinfo * ggmlhexagon_get_socinfo_from_socmodel(uint32_t soc_model) {
    size_t items = sizeof(g_hexagon_soc_info_table) / sizeof(g_hexagon_soc_info_table[0]);
    for (size_t idx = 0; idx < items; idx++) {
        if (soc_model == g_hexagon_soc_info_table[idx].soc_model) {
            return &g_hexagon_soc_info_table[idx];
        }
    }
    return nullptr;
}

static struct qcom_socinfo * ggmlhexagon_get_socinfo_from_htparch(size_t htp_arch) {
    size_t items = sizeof(g_hexagon_soc_info_table) / sizeof(g_hexagon_soc_info_table[0]);
    for (size_t idx = 0; idx < items; idx++) {
        if (htp_arch == g_hexagon_soc_info_table[idx].htp_arch) {
            return &g_hexagon_soc_info_table[idx];
        }
    }
    return nullptr;
}

static size_t ggmlhexagon_get_system_total_memory_in_bytes() {
#if defined(__ANDROID__) || defined(__linux__)
    struct sysinfo info = {};
    if (0 == sysinfo(&info)) {
        return (info.totalram + info.totalswap) * info.mem_unit;
    }
    size_t pages      = (size_t)sysconf(_SC_PHYS_PAGES);
    size_t page_size  = (size_t)sysconf(_SC_PAGE_SIZE);

    return pages * page_size;
#elif defined(_WIN32)
    //TODO
    return 0;
#endif
}

static size_t ggmlhexagon_get_system_free_memory_in_bytes() {
#if defined(__ANDROID__) || defined(__linux__)
    struct sysinfo info = {};
    if (0 == sysinfo(&info)) {
        return (info.freeram + info.freeswap) * info.mem_unit;
    }
    size_t avail_pages = (size_t)sysconf(_SC_AVPHYS_PAGES);
    size_t page_size   = (size_t)sysconf(_SC_PAGE_SIZE);

    return avail_pages * page_size;
#elif defined(_WIN32)
    //TODO
    return 0;
#endif
}

static bool ggmlhexagon_same_types(const ggml_backend_hexagon_context * ctx, const ggml_tensor * op_tensor) {
    GGML_UNUSED(ctx);
    ggml_tensor * src0 = op_tensor->src[0];
    ggml_tensor * src1 = op_tensor->src[1];
    if (nullptr != src1) {
        if (src0->type != op_tensor->type || src1->type != op_tensor->type) {
            return false;
        }
    } else {
        if (src0->type != op_tensor->type) {
            return false;
        }
    }

    if (src0->type != GGML_TYPE_F32)
        return false;

    return true;
}

static void ggmlhexagon_set_runtime_path(size_t device, const std::string & path) {
    GGML_UNUSED(device);
#if defined(__ANDROID__)
    // Android: LD_LIBRARY_PATH uses ':' as separator
    std::string lib_runtime_path = path + ":/vendor/dsp/cdsp:/vendor/lib64:/vendor/dsp/dsp:/vendor/dsp/images";
    if (0 == setenv("LD_LIBRARY_PATH", lib_runtime_path.c_str(), 1)) {
        GGMLHEXAGON_LOG_DEBUG("setenv LD_LIBRARY_PATH %s successfully", lib_runtime_path.c_str());
    } else {
        GGMLHEXAGON_LOG_ERROR("setenv LD_LIBRARY_PATH %s failure", lib_runtime_path.c_str());
    }

    // ADSP_LIBRARY_PATH uses ';' as separator on all platforms
    std::string adsp_runtime_path = path + ";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/vendor/dsp/dsp;/vendor/dsp/images;/dsp";
    if (0 == setenv("ADSP_LIBRARY_PATH", adsp_runtime_path.c_str(), 1)) {
        GGMLHEXAGON_LOG_DEBUG("setenv ADSP_LIBRARY_PATH %s successfully", adsp_runtime_path.c_str());
    } else {
        GGMLHEXAGON_LOG_ERROR("setenv ADSP_LIBRARY_PATH %s failure", adsp_runtime_path.c_str());
    }
#elif defined(__linux__)
    // Linux: LD_LIBRARY_PATH uses ':' as separator
    std::string lib_runtime_path = path + ":/usr/local/lib:/usr/lib";
    if (0 == setenv("LD_LIBRARY_PATH", lib_runtime_path.c_str(), 1)) {
        GGMLHEXAGON_LOG_DEBUG("setenv LD_LIBRARY_PATH %s successfully", lib_runtime_path.c_str());
    } else {
        GGMLHEXAGON_LOG_ERROR("setenv LD_LIBRARY_PATH %s failure", lib_runtime_path.c_str());
    }

    std::string adsp_runtime_path = path + ";/usr/local/lib;/usr/lib";
    if (0 == setenv("ADSP_LIBRARY_PATH", adsp_runtime_path.c_str(), 1)) {
        GGMLHEXAGON_LOG_DEBUG("setenv ADSP_LIBRARY_PATH %s successfully", adsp_runtime_path.c_str());
    } else {
        GGMLHEXAGON_LOG_ERROR("setenv ADSP_LIBRARY_PATH %s failure", adsp_runtime_path.c_str());
    }
#elif defined(_WIN32)
    // WoA: PATH uses ';' as separator
    std::string lib_runtime_path = path + ";C:\\Windows\\System32;C:\\Windows\\SysWOW64";
    if (0 == _putenv_s("PATH", lib_runtime_path.c_str())) {
        GGMLHEXAGON_LOG_DEBUG("setenv PATH %s successfully", lib_runtime_path.c_str());
    } else {
        GGMLHEXAGON_LOG_ERROR("setenv PATH %s failure", lib_runtime_path.c_str());
    }

    std::string adsp_runtime_path = path + ";C:\\Windows\\System32";
    if (0 == _putenv_s("ADSP_LIBRARY_PATH", adsp_runtime_path.c_str())) {
        GGMLHEXAGON_LOG_DEBUG("setenv ADSP_LIBRARY_PATH %s successfully", adsp_runtime_path.c_str());
    } else {
        GGMLHEXAGON_LOG_ERROR("setenv ADSP_LIBRARY_PATH %s failure", adsp_runtime_path.c_str());
    }
#endif
}

static inline bool ggml_hexagon_is_repack_type(enum ggml_type type) {
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 ||
           type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_NL ||
           type == GGML_TYPE_MXFP4;
}

// Some weight types are stored in the repack buffer in a different format
// than their logical ggml type (BF16 as F16, Q4_K/Q6_K as Q4_0); the DSP
// kernels only see the storage type. Q4_K/Q6_K are stored as Q4_0 (not Q8_0)
// so the bandwidth-bound lm-head matvec moves less data per token.
static inline enum ggml_type ggml_hexagon_weight_dsp_type(enum ggml_type type) {
    if (type == GGML_TYPE_BF16) return GGML_TYPE_F16;
    if (type == GGML_TYPE_Q4_K) return GGML_TYPE_Q4_0;
    if (type == GGML_TYPE_Q6_K) return GGML_TYPE_Q4_0;
    return type;
}

static inline bool ggml_hexagon_is_hmx_weight_type(enum ggml_type type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || ggml_hexagon_is_repack_type(type);
}

// =================================================================================================
//  section-6: CDSP helper functions
// =================================================================================================
static const char * ggmlhexagon_get_dsp_name(int domain_id) {
    (void)domain_id;
    return "Hexagon-cDSP";
}

static int ggmlhexagon_get_vtcm_info(int domain, uint32_t attr, uint32_t * capability) {
    int hexagon_error = AEE_SUCCESS;
    *capability = 0;

    if (attr != VTCM_PAGE && attr != VTCM_COUNT) {
        hexagon_error = AEE_EBADPARM;
        GGMLHEXAGON_LOG_DEBUG("unsupported attr, only VTCM_PAGE and VTCM_COUNT supported");
        goto bail;
    }

    if (remote_handle_control) {
        if (domain == CDSP_DOMAIN_ID) {
            struct remote_dsp_capability dsp_capability_vtcm_dsp;
            dsp_capability_vtcm_dsp.domain       = (uint32_t)domain;
            dsp_capability_vtcm_dsp.attribute_ID = attr;
            dsp_capability_vtcm_dsp.capability   = (uint32_t)0;
            hexagon_error = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_vtcm_dsp, sizeof(struct remote_dsp_capability));
            if ((hexagon_error & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
                GGMLHEXAGON_LOG_DEBUG("FastRPC Capability API is not supported on this device");
                GGMLHEXAGON_LOG_DEBUG("running the use case without checking the capability");
                hexagon_error = AEE_SUCCESS;
                goto bail;
            } else if (hexagon_error == AEE_SUCCESS) {
                *capability = dsp_capability_vtcm_dsp.capability;
            } else {
                GGMLHEXAGON_LOG_DEBUG("get_vtcm_info failed with error 0x%x", hexagon_error);
                goto bail;
            }
        } else {
            hexagon_error = AEE_EUNSUPPORTED;
            GGMLHEXAGON_LOG_DEBUG("unsupported domain %d", domain);
            goto bail;
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
        GGMLHEXAGON_LOG_DEBUG("remote_dsp_capability interface is not supported on this device");
    }

bail:
    return hexagon_error;
}

static bool ggmlhexagon_is_unsignedpd_supported(int domain_id) {
    int hexagon_error = AEE_SUCCESS;
    if (remote_handle_control) {
        struct remote_dsp_capability dsp_capability_domain = {static_cast<uint32_t>(domain_id), UNSIGNED_PD_SUPPORT, 0};
        hexagon_error = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_domain, sizeof(struct remote_dsp_capability));
        if ((hexagon_error & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
            GGMLHEXAGON_LOG_WARN("FastRPC Capability API is not supported on this device. Falling back to signed pd");
            return false;
        }

        if (hexagon_error) {
            GGMLHEXAGON_LOG_WARN("error 0x%x: FastRPC Capability API failed. falling back to signed pd", hexagon_error);
            return false;
        }

        if (dsp_capability_domain.capability == 1) {
            return true;
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
        GGMLHEXAGON_LOG_WARN("remote_dsp_capability interface is not supported on this device.falling back to signed pd");
        return false;
    }

    return false;
}

static bool ggmlhexagon_is_async_fastrpc_supported(int domain) {
    int hexagon_error = AEE_SUCCESS;
    if (remote_handle_control) {
        if (domain == CDSP_DOMAIN_ID) {
            struct remote_dsp_capability dsp_capability_async_support;
            dsp_capability_async_support.domain       = (uint32_t)domain;
            dsp_capability_async_support.attribute_ID = ASYNC_FASTRPC_SUPPORT;
            dsp_capability_async_support.capability   = (uint32_t)0;
            hexagon_error = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_async_support, sizeof(struct remote_dsp_capability));
            if ((hexagon_error & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
                GGMLHEXAGON_LOG_WARN("FastRPC Capability API is not supported on this device");
                hexagon_error = AEE_SUCCESS;
                goto bail;
            } else if (dsp_capability_async_support.capability == 1) {
                return true;
            }

            if (hexagon_error != AEE_SUCCESS){
                GGMLHEXAGON_LOG_WARN("failed with error 0x%x", hexagon_error);
                goto bail;
            }
        } else {
            hexagon_error = AEE_EUNSUPPORTED;
            GGMLHEXAGON_LOG_WARN("async FastRPC is not supported on domain %d", domain);
            goto bail;
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
        GGMLHEXAGON_LOG_WARN("remote_dsp_capability interface is not supported on this device");
    }

bail:
    return false;
}

static void ggmlhexagon_set_rpc_latency(remote_handle64 handle, int qos, int latency) {
    int hexagon_error = AEE_SUCCESS;
    (void)latency;

    if (remote_handle_control) {
        // Align with QCOM reference: only enable QoS mode, let DSP decide latency.
        struct remote_rpc_control_latency data;
        memset(&data, 0, sizeof(data));
        data.enable = qos;
        hexagon_error = remote_handle64_control(handle, DSPRPC_CONTROL_LATENCY, (void*)&data, sizeof(data));
        if (hexagon_error != AEE_SUCCESS) {
            GGMLHEXAGON_LOG_WARN("failed with error 0x%x", hexagon_error);
            goto bail;
        } else {
            GGMLHEXAGON_LOG_VERBOSE("set rpc qos %d (DSP default latency)", qos);
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
        GGMLHEXAGON_LOG_WARN("remote_dsp_capability interface is not supported on this device");
    }

bail:
    return;
}

static int ggmlhexagon_get_hmx_support_info(int domain, uint32_t attr, uint32_t * capability) {
    int hexagon_error = AEE_SUCCESS;
    *capability = 0;

    if (attr != HMX_SUPPORT_SPATIAL && attr != HMX_SUPPORT_DEPTH) {
        hexagon_error = AEE_EBADPARM;
        GGMLHEXAGON_LOG_WARN("unsupported attr, only HMX_SUPPORT_SPATIAL and HMX_SUPPORT_DEPTH supported");
        goto bail;
    }

    if (remote_handle_control) {
        if (domain == CDSP_DOMAIN_ID) {
            struct remote_dsp_capability dsp_capability_hmx_dsp;
            dsp_capability_hmx_dsp.domain       = (uint32_t)domain;
            dsp_capability_hmx_dsp.attribute_ID = attr;
            dsp_capability_hmx_dsp.capability   = (uint32_t)0;
            hexagon_error = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_hmx_dsp, sizeof(struct remote_dsp_capability));
            if ((hexagon_error & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
                GGMLHEXAGON_LOG_DEBUG("FastRPC Capability API is not supported on this device");
                hexagon_error = AEE_SUCCESS;
                goto bail;
            }
            else if (hexagon_error == AEE_SUCCESS) {
                *capability = dsp_capability_hmx_dsp.capability;
            } else {
                GGMLHEXAGON_LOG_DEBUG("get_hmx_support_info failed with Error 0x%x", hexagon_error);
                goto bail;
            }
        } else {
            hexagon_error = AEE_EUNSUPPORTED;
            GGMLHEXAGON_LOG_DEBUG("HMX support is not there for domain %d", domain);
            goto bail;
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
        GGMLHEXAGON_LOG_DEBUG("remote_dsp_capability interface is not supported on this device");
    }

bail:
    return hexagon_error;
}

static int ggmlhexagon_get_hvx_arch_ver(int domain, uint32_t * capability) {
    int hexagon_error = AEE_SUCCESS;
    *capability = 0;
    if(remote_handle_control) {
        struct remote_dsp_capability dsp_capability_arch_ver;
        dsp_capability_arch_ver.domain       = (uint32_t)domain;
        dsp_capability_arch_ver.attribute_ID = ARCH_VER;
        dsp_capability_arch_ver.capability   = (uint32_t)0;
        hexagon_error = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_arch_ver, sizeof(struct remote_dsp_capability));
        if ((hexagon_error & 0xFF) == (AEE_EUNSUPPORTEDAPI & 0xFF)) {
            GGMLHEXAGON_LOG_DEBUG("FastRPC Capability API is not supported on this device");
            hexagon_error = AEE_SUCCESS;
            goto bail;
        } else if (hexagon_error == AEE_SUCCESS) {
            *capability = dsp_capability_arch_ver.capability & 0xFF;
        } else {
            GGMLHEXAGON_LOG_DEBUG("get_hex_arch_ver failed with error 0x%x", hexagon_error);
            goto bail;
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
        GGMLHEXAGON_LOG_DEBUG("remote_dsp_capability interface is not supported on this device");
    }

bail:
    return hexagon_error;
}

static int ggmlhexagon_get_hvx_support_info(int domain, uint32_t attr, uint32_t * capability) {
    int hexagon_error = AEE_SUCCESS;
    *capability = 0;
    if (attr == HVX_SUPPORT_64B) {
        hexagon_error = AEE_EBADPARM;
        GGMLHEXAGON_LOG_DEBUG("latest targets have 128 byte HVX register, use HVX_SUPPORT_128B instead of HVX_SUPPORT_64B");
        goto bail;
    }

    if (attr != HVX_SUPPORT_128B) {
        hexagon_error = AEE_EBADPARM;
        GGMLHEXAGON_LOG_DEBUG("unsupported attr. only HVX_SUPPORT_128B supported");
        goto bail;
    }

    if (remote_handle_control) {
        if (domain == CDSP_DOMAIN_ID) {
            struct remote_dsp_capability dsp_capability_hvx_dsp;
            dsp_capability_hvx_dsp.domain       = (uint32_t)domain;
            dsp_capability_hvx_dsp.attribute_ID = attr;
            dsp_capability_hvx_dsp.capability   = (uint32_t)0;
            hexagon_error = remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_capability_hvx_dsp, sizeof(struct remote_dsp_capability));
            if ((hexagon_error & 0xFF)==(AEE_EUNSUPPORTEDAPI & 0xFF)) {
                GGMLHEXAGON_LOG_DEBUG("FastRPC Capability API is not supported on this device");
                hexagon_error = AEE_SUCCESS;
                goto bail;
            } else if (hexagon_error == AEE_SUCCESS) {
                *capability = dsp_capability_hvx_dsp.capability;
            } else {
                GGMLHEXAGON_LOG_DEBUG("failed with error 0x%x", hexagon_error);
                goto bail;
            }
        } else {
            hexagon_error = AEE_EUNSUPPORTED;
            GGMLHEXAGON_LOG_DEBUG("HVX support is not available on domain %d", domain);
            goto bail;
        }
    } else {
        hexagon_error = AEE_EUNSUPPORTEDAPI;
        GGMLHEXAGON_LOG_DEBUG("remote_dsp_capability interface is not supported on this device");
    }

bail:
    return hexagon_error;
}

static int ggmlhexagon_init_rpcmempool(ggml_backend_hexagon_context * ctx) {
    size_t candidate_size   = 0;
    uint8_t * rpc_buffer    = nullptr;
    std::vector<int>        probe_slots;

    int htp_arch = 0;
    htp_arch = ggmlhexagon_probe_dspinfo(ctx);
    if (0 == htp_arch)
        return 1;
    if (nullptr == ctx)
        return 2;

    /* Probe mempool capacity, find a large contiguous block near the 4 GiB boundary on devices. */
    probe_slots.push_back(2048);
    if (htp_arch > 75) {
        probe_slots.push_back(3840);
        probe_slots.push_back(3968);
        probe_slots.push_back(4032);
    } else {
        probe_slots.push_back(3830);
    }

    size_t probe_counts     = probe_slots.size();
    for (size_t idx = 0; idx < probe_counts; idx++) {
        rpc_buffer = static_cast<uint8_t *>(rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (probe_slots[idx] * SIZE_IN_MB)));
        if (nullptr == rpc_buffer) {
            GGMLHEXAGON_LOG_DEBUG("alloc rpcmem %d (MiB) failure during probe rpc memory info, reason: %s\n", probe_slots[idx], strerror(errno));
            break;
        } else {
            candidate_size = probe_slots[idx];
            rpcmem_free(rpc_buffer);
            rpc_buffer = nullptr;
        }
    }
    ctx->rpc_mempool_capacity = candidate_size * SIZE_IN_MB;
    GGMLHEXAGON_LOG_ALWAYS("rpc memory capacity %ld(%d MiB) for device %d",
                          ctx->rpc_mempool_capacity, ctx->rpc_mempool_capacity / SIZE_IN_MB, ctx->device);
    GGML_ASSERT(ctx->rpc_mempool_capacity > (8 * SIZE_IN_MB));
    ctx->rpc_mempool_len = ctx->rpc_mempool_capacity - (8 * SIZE_IN_MB);
    ctx->rpc_mempool = rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, ctx->rpc_mempool_len);
    if (nullptr == ctx->rpc_mempool) {
        GGMLHEXAGON_LOG_ERROR("alloc rpc memorypool %ld(%d MiB) failed", ctx->rpc_mempool_len, ctx->rpc_mempool_capacity / SIZE_IN_MB);
        return 3;
    } else {
        GGMLHEXAGON_LOG_DEBUG("alloc rpc memorypool %p successfully %ld(%d MiB)",
                              ctx->rpc_mempool, ctx->rpc_mempool_len,
                              ctx->rpc_mempool_len / SIZE_IN_MB);
    }
    ctx->rpc_mempool_handle = rpcmem_to_fd(ctx->rpc_mempool);
    GGMLHEXAGON_LOG_INFO("rpc mempool handle %d", ctx->rpc_mempool_handle);
    GGMLHEXAGON_LOG_INFO("rpc mempool addr %p", ctx->rpc_mempool);
    GGMLHEXAGON_LOG_INFO("rpc mempool size %lld(%dMB)", ctx->rpc_mempool_len, ctx->rpc_mempool_len/ SIZE_IN_MB);
    // Register mempool with FastRPC kernel driver.
    // rpc_mmap_mode = 0: FASTRPC_MAP_FD_DELAYED (default), defers DSP-side mapping until HAP_mmap2().
    // rpc_mmap_mode = 1: FASTRPC_MAP_FD (eager), creates immediate kernel-level mapping and pins pages.
    enum fastrpc_map_flags mmap_flags = (g_hexagon_appcfg.rpc_mmap_mode == 1)
                                        ? FASTRPC_MAP_FD
                                        : FASTRPC_MAP_FD_DELAYED;
    const char * mmap_mode_str = (g_hexagon_appcfg.rpc_mmap_mode == 1) ? "EAGER" : "DELAYED";
    int mmap_err = fastrpc_mmap(ctx->domain_id, ctx->rpc_mempool_handle,
                                 ctx->rpc_mempool, 0, ctx->rpc_mempool_len,
                                 mmap_flags);
    if (mmap_err != 0) {
        GGMLHEXAGON_LOG_ERROR("fastrpc_mmap(%s) returned %d (fd=%d), continuing...",
                             mmap_mode_str, mmap_err, ctx->rpc_mempool_handle);
    } else {
        GGMLHEXAGON_LOG_INFO("fastrpc_mmap(%s) OK: fd=%d, size=%dMB",
                             mmap_mode_str, ctx->rpc_mempool_handle, ctx->rpc_mempool_len / SIZE_IN_MB);
    }

    // Register mempool on DSP side via pure-scalar IDL call.
    // This avoids FastRPC's fdlist_fd_from_buf() scan that triggers
    // implicit fd_mmap_create when dsptensor.data pointers are passed.
    // The DSP will call HAP_mmap2(fd) to get a user-space-accessible VA.
    uint32_t ion_fd = (uint32_t)ctx->rpc_mempool_handle;
    uint32_t size_lo = (uint32_t)(ctx->rpc_mempool_len & 0xFFFFFFFF);
    uint32_t size_hi = (uint32_t)((ctx->rpc_mempool_len >> 32) & 0xFFFFFFFF);

    int64_t t0_reg = ggml_time_us();
    int reg_err = ggml_dsp_register_ion(ctx->ggmlop_handle, ion_fd, size_lo, size_hi);
    int64_t dt_reg = ggml_time_us() - t0_reg;
    if (reg_err != AEE_SUCCESS) {
        GGMLHEXAGON_LOG_ERROR("dsp_register_ion failed: 0x%x", reg_err);
    } else {
        GGMLHEXAGON_LOG_ALWAYS("registered mempool base via scalar call: fd=%d, size=%dMB, time=%lld us",
                             ctx->rpc_mempool_handle, ctx->rpc_mempool_len / SIZE_IN_MB, (long long)dt_reg);
    }
    GGMLHEXAGON_LOG_ALWAYS("mempool layout: total=%zuMB", ctx->rpc_mempool_len / SIZE_IN_MB);
    // Prime the RPC overhead profiler with a few no-op warmup calls to get
    // a meaningful min/max/avg distribution without polluting the pool.
    for (int i = 0; i < 6; i++) {
        (void)hexagon_warmup_invoke_timed(ctx);
    }

    return 0;
}

static void ggmlhexagon_deinit_rpcmempool(ggml_backend_hexagon_context * ctx) {
    if (ctx->rpc_mempool) {
        GGMLHEXAGON_LOG_DEBUG("free rpc mempool %p", ctx->rpc_mempool);
        rpcmem_free(ctx->rpc_mempool);
        ctx->rpc_mempool            = nullptr;
        ctx->rpc_mempool_len        = 0;
        ctx->rpc_mempool_capacity   = 0;
    }
}

static int ggmlhexagon_probe_dspinfo(ggml_backend_hexagon_context * ctx) {
    if (ctx == nullptr) {
        return 0;
    }
    uint32_t dsp_version = 0;
    int htp_arch         = 0;
    ggmlhexagon_get_hvx_arch_ver(ctx->domain_id, &dsp_version);

    size_t total_mem = ggmlhexagon_get_system_total_memory_in_bytes();


    if (dsp_version == 0x68 || dsp_version == 0x69 || dsp_version == 0x73
        || dsp_version == 0x75 || dsp_version == 0x79 || dsp_version == 0x81) {
        //0x68 -> 68, 0x69 -> 69, 0x73 -> 73, 0x75 -> 75, 0x79 -> 79, 0x81 -> 81
        htp_arch = ggmlhexagon_htparch_hex_to_decimal(dsp_version);
        struct qcom_socinfo * socinfo = ggmlhexagon_get_socinfo_from_htparch(htp_arch);
        GGML_ASSERT(nullptr != socinfo);
        ctx->socinfo = *socinfo;
        GGMLHEXAGON_LOG_VERBOSE("device info: %s, %s, dsp arch version 0x%x, system mem size %d MiB",
                                socinfo->soc_desc, ggmlhexagon_get_htparch_desc(htp_arch), dsp_version, total_mem / SIZE_IN_MB);
    } else {
        GGMLHEXAGON_LOG_VERBOSE("device info: unknown");
    }

    uint32_t vtcm_count = 0;
    uint32_t vtcm_page  = 0;
    ggmlhexagon_get_vtcm_info(ctx->domain_id, VTCM_COUNT, &vtcm_count);
    ggmlhexagon_get_vtcm_info(ctx->domain_id, VTCM_PAGE, &vtcm_page);
    ctx->has_vtcm = (vtcm_count > 0 && vtcm_page > 0);

    uint32_t hmx_depth   = 0;
    uint32_t hmx_spatial = 0;
    //FIXME: better approach to get correct/accurate info
    ggmlhexagon_get_hmx_support_info(ctx->domain_id, HMX_SUPPORT_DEPTH, &hmx_depth);
    ggmlhexagon_get_hmx_support_info(ctx->domain_id, HMX_SUPPORT_SPATIAL, &hmx_spatial);

    uint32_t hvx_support_128b = 0;
    ggmlhexagon_get_hvx_support_info(ctx->domain_id, HVX_SUPPORT_128B, &hvx_support_128b);
    ctx->has_hvx = (hvx_support_128b > 0);
    ctx->has_hmx = (hmx_depth > 0 || hmx_spatial > 0);
    // Fallback: DSPRPC_GET_DSP_INFO may not report HMX on some devices
    // HMX is present on V73+ (Snapdragon 8 Gen 2 and later); the DSP skel is built with -mhmx.
    if (!ctx->has_hmx && htp_arch >= V73) {
        ctx->has_hmx = true;
    }
    GGMLHEXAGON_LOG_DEBUG("dsp arch version %d, vtcm_count %d, vtcm_page %d", htp_arch, vtcm_count, vtcm_page);
    //FIXME: hmx_depth/hmx_spatial report 0 via DSPRPC_GET_DSP_INFO on some devices
    GGMLHEXAGON_LOG_VERBOSE("device %d caps: has_vtcm=%d,has_hvx=%d,has_hmx=%d,hvx_support_128b %d,"
                            "unsigned pd supported %d, async fastrpc supported %d",
                                ctx->device, (int)ctx->has_vtcm, (int)ctx->has_hvx, (int)ctx->has_hmx, hvx_support_128b,
                                ggmlhexagon_is_unsignedpd_supported(ctx->domain_id),
                                ggmlhexagon_is_async_fastrpc_supported(ctx->domain_id));
    return htp_arch;
}

static void ggmlhexagon_deinit_cdsp(ggml_backend_hexagon_context * ctx) {
    GGMLHEXAGON_LOG_ALWAYS("enter %s", __FUNCTION__);
    int hexagon_error  = AEE_SUCCESS;
    GGML_ASSERT(0 != ctx->ggmlop_handle);
    hexagon_error = ggml_dsp_close(ctx->ggmlop_handle);
    if (AEE_SUCCESS != hexagon_error) {
        GGMLHEXAGON_LOG_ERROR("error 0x%x: failed to close ggmlop dsp handle", hexagon_error);
    }
    ctx->ggmlop_handle = 0;
    ggmlhexagon_deinit_rpcmempool(ctx);
    //probe before domain_id is invalidated so AP-side domain queries still work
    ggmlhexagon_probe_dspinfo(ctx);
    ggmlhexagon_dump_perf_stats(ctx);
    ctx->domain_id             = -1;
    GGMLHEXAGON_LOG_ALWAYS("leave %s", __FUNCTION__);
}

static int ggmlhexagon_init_dsp(ggml_backend_hexagon_context * ctx) {
    GGMLHEXAGON_LOG_ALWAYS("enter %s", __FUNCTION__);
    int htp_arch                = 0;
    int hexagon_error           = AEE_SUCCESS;
    int domain_id               = CDSP_DOMAIN_ID;
    bool got_uri                = false;
    const char * uri            = NULL;
    domain * my_domain          = NULL;
    bool is_unsignedpd_enabled  = false;
    char final_uri[512];
    char ggmldsp_uri[256];

    if (nullptr == ctx)
        return 1;
    if (0 != ctx->ggmlop_handle) {
        GGMLHEXAGON_LOG_DEBUG("already init Hexagon CDSP with backend %d(%s)", ctx->device, ctx->name);
        return 0;
    }
    if (!remote_session_control) {
        GGMLHEXAGON_LOG_ERROR("remote_session_control not available");
        hexagon_error = AEE_EUNSUPPORTED;
        goto bail;
    }

    GGMLHEXAGON_LOG_DEBUG("init Hexagon CDSP with backend %d(%s)", ctx->device, ctx->name);
    ctx->ggmlop_handle = 0;
    my_domain = htpdrv_get_domain(domain_id);
    if (NULL == my_domain) {
        GGMLHEXAGON_LOG_ERROR("unable to get domain struct %d", domain_id);
        goto bail;
    }
    uri = my_domain->uri;
    GGMLHEXAGON_LOG_DEBUG("domain uri=%s", uri);
    // Reserve new FastRPC session (PD) for additional devices (dev_id > 0)
    // dev_id == 0 reuses the default CDSP PD (session_id=0)
    ctx->session_id = 0;
    if (ctx->device > 0) {
        struct remote_rpc_reserve_new_session n;
        n.domain_name_len  = strlen(CDSP_DOMAIN_NAME);
        n.domain_name      = const_cast<char *>(CDSP_DOMAIN_NAME);
        char sess_name[32];
        snprintf(sess_name, sizeof(sess_name), "Hexagon-cDSP%d", ctx->device);
        n.session_name     = sess_name;
        n.session_name_len = strlen(sess_name);

        int err = remote_session_control(FASTRPC_RESERVE_NEW_SESSION, (void *) &n, sizeof(n));
        if (err != AEE_SUCCESS) {
            GGMLHEXAGON_LOG_WARN("FASTRPC_RESERVE_NEW_SESSION failed for device %d: error 0x%x", ctx->device, err);
            hexagon_error = err;
            goto bail;
        }
        ctx->session_id = n.session_id;
        domain_id       = n.effective_domain_id;
        GGMLHEXAGON_LOG_VERBOSE("reserved new session: device=%d session_id=%d effective_domain_id=%d",
                             ctx->device, ctx->session_id, domain_id);
    }

    is_unsignedpd_enabled = ggmlhexagon_is_unsignedpd_supported(domain_id);
    if (!is_unsignedpd_enabled) {
        GGMLHEXAGON_LOG_ERROR("unsigned PD not allowed on domain %d, using signed offload", domain_id);
        goto bail;
    }

    ctx->domain_id = domain_id;
    GGMLHEXAGON_LOG_ALWAYS("using Hexagon domain %d(%s)", domain_id, ggmlhexagon_get_dsp_name(domain_id));
    GGMLHEXAGON_LOG_ALWAYS("unsignedpd_enabled %d", is_unsignedpd_enabled);
    if (is_unsignedpd_enabled) {
        struct remote_rpc_control_unsigned_module data;
        data.enable = 1;
        data.domain = domain_id;
        hexagon_error = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&data, sizeof(data));
        GGMLHEXAGON_LOG_DEBUG("remote_session_control returned %d for configuring unsigned PD", hexagon_error);
        if (AEE_SUCCESS != hexagon_error) {
            GGMLHEXAGON_LOG_ERROR("error 0x%x: remote_session_control failed", hexagon_error);
            goto bail;
        }
    }

    // Probe arch and build the versioned dsp skel URI
    htp_arch = ggmlhexagon_probe_dspinfo(ctx);
    GGML_ASSERT(0 != htp_arch);

    snprintf(ggmldsp_uri, sizeof(ggmldsp_uri),
             "file:///libggmldsp-skel-v%u.so?ggml_dsp_skel_handle_invoke&_modver=1.0&_idlver=" GGML_DSP_IDL_VERSION,
             htp_arch);

    // Build the final URI for ggml_dsp_open.
    // session_id > 0: use FASTRPC_GET_URI to obtain the session-specific URI.
    // session_id == 0 (or FASTRPC_GET_URI failure): concatenate ggmldsp_uri + domain uri.
    if (ctx->session_id > 0) {
        struct remote_rpc_get_uri u = {};
        u.session_id      = ctx->session_id;
        u.domain_name     = const_cast<char *>(CDSP_DOMAIN_NAME);
        u.domain_name_len = strlen(CDSP_DOMAIN_NAME);
        u.module_uri      = const_cast<char *>(ggmldsp_uri);
        u.module_uri_len  = strlen(ggmldsp_uri);
        u.uri             = final_uri;
        u.uri_len         = sizeof(final_uri);
        int err = remote_session_control(FASTRPC_GET_URI, (void *) &u, sizeof(u));
        if (err == AEE_SUCCESS) {
            got_uri = true;
            GGMLHEXAGON_LOG_DEBUG("session URI for session_id=%d: %s", ctx->session_id, final_uri);
        } else {
            GGMLHEXAGON_LOG_WARN("FASTRPC_GET_URI failed for session_id=%d: error 0x%x, fallback to %s%s",
                                 ctx->session_id, err, ggmldsp_uri, uri);
        }
    }
    if (!got_uri) {
        snprintf(final_uri, sizeof(final_uri), "%s%s", ggmldsp_uri, uri);
    }

    GGMLHEXAGON_LOG_DEBUG("ggmlop domain uri: %s", final_uri);
    hexagon_error = ggml_dsp_open(final_uri, &ctx->ggmlop_handle);
    if (AEE_SUCCESS == hexagon_error) {
        GGMLHEXAGON_LOG_VERBOSE("succeed to open domain %d(%s)", domain_id, ggmlhexagon_get_dsp_name(domain_id));
        ggml_dsp_setclocks(ctx->ggmlop_handle, g_hexagon_appcfg.dump_diag_info, g_hexagon_appcfg.thread_counts, &ctx->dsp_thread_counts);
        // Mirror DSP-side clamp into the global cfg so subsequent log sites
        // (including the dtor, where ctx may be unavailable) reflect the
        // real thread count in effect rather than the user's requested hint.
        // Guard: on RPC failure dsp_thread_counts stays 0; keep cfg value.
        if (ctx->dsp_thread_counts > 0) {
            g_hexagon_appcfg.thread_counts = ctx->dsp_thread_counts;
            ctx->n_threads = ctx->dsp_thread_counts;
        }
        ggmlhexagon_set_rpc_latency(ctx->ggmlop_handle, RPC_PM_QOS, 100);
        if (0 != ggmlhexagon_init_rpcmempool(ctx)) {
            GGMLHEXAGON_LOG_ERROR("failed to init rpc mempool");
            goto bail;
        }
        // Push DSP-side cache optimization bitmask via execute_batch(0xFFFC)
        // special mode (no IDL change). batch_offset = packed payload, batch_size = mode tag.
        // Payload bit layout (low bits are dsp_cache_mode):
        //   bits  0..3 : dsp_cache_mode (first-touch weight / prior-dst skip / bulk dst flush / selective bulk flush)
        //   bit  16    : dsp_cache_trace_bit0 (1 = emit [DSP-CACHE-TRACE-BIT0] per bit 0 decision)
        //   bit  17    : dsp_cache_trace_bit1 (1 = emit [DSP-CACHE-TRACE-BIT1] per bit 1 decision)
        // See the dsp_cache_mode and dsp_cache_trace_bit{0,1} comments in hexagon_appcfg_t.
        //
        // Note: must run AFTER ggmlhexagon_init_rpcmempool(). The DSP-side
        // 0xFFFC handler in entry.c asserts ion_dsp_base != NULL; without the
        // mempool registered first it returns AEE_EBADPARM (0x8000040e) and
        // the bitmask is silently dropped.
        {
            const uint32_t mode_bits    = (uint32_t)g_hexagon_appcfg.dsp_cache_mode & 0xFu;
            const uint32_t trace_bit0   = (g_hexagon_appcfg.dsp_cache_trace_bit0 ? 0x10000u : 0u);
            const uint32_t trace_bit1   = (g_hexagon_appcfg.dsp_cache_trace_bit1 ? 0x20000u : 0u);
            const uint32_t payload      = trace_bit1 | trace_bit0 | mode_bits;
            int opts_err = ggml_dsp_execute_batch(ctx->ggmlop_handle, payload, 0xFFFC);
            if (AEE_SUCCESS != opts_err) {
                GGMLHEXAGON_LOG_WARN("set dsp_cache_mode=0x%x + dsp_cache_trace_bit0=%d + dsp_cache_trace_bit1=%d failed: 0x%x (DSP-side optimizations disabled)",
                                     mode_bits, g_hexagon_appcfg.dsp_cache_trace_bit0,
                                     g_hexagon_appcfg.dsp_cache_trace_bit1, opts_err);
                g_hexagon_appcfg.dsp_cache_mode = 0;  // fall back to baseline
                g_hexagon_appcfg.dsp_cache_trace_bit0 = 0;
                g_hexagon_appcfg.dsp_cache_trace_bit1 = 0;
            } else {
                GGMLHEXAGON_LOG_VERBOSE("[AP-CACHE-MODE] dsp_cache_mode=0x%x + dsp_cache_trace_bit0=%d + dsp_cache_trace_bit1=%d pushed to DSP (payload=0x%x)",
                                     mode_bits, g_hexagon_appcfg.dsp_cache_trace_bit0,
                                     g_hexagon_appcfg.dsp_cache_trace_bit1, payload);
            }

            /* Warmup FastRPC/mempool path once before first real inference.
             * This triggers delayed mempool mapping and touches the DSP entry path
             * so the first real batch does not pay cold-state penalty. */
            int warmup_err = ggml_dsp_execute_batch(ctx->ggmlop_handle, 0, 0xFFFB);
            if (AEE_SUCCESS != warmup_err) {
                GGMLHEXAGON_LOG_ERROR("warmup execute_batch failed: 0x%x", warmup_err);
            } else {
                GGMLHEXAGON_LOG_ALWAYS("[AP-WARMUP] FastRPC/mempool warmup done");
            }
        }
    } else {
        GGMLHEXAGON_LOG_ERROR("error 0x%x: failed to open domain %d(%s)", hexagon_error, domain_id, ggmlhexagon_get_dsp_name(domain_id));
        goto bail;
    }

    snprintf(ctx->name, sizeof(ctx->name), "Hexagon-cDSP%d", ctx->device);
    GGMLHEXAGON_LOG_ALWAYS("leave %s", __FUNCTION__);
    return 0;

bail:
    ggmlhexagon_deinit_cdsp(ctx);
    GGMLHEXAGON_LOG_ALWAYS("leave %s", __FUNCTION__);
    return -1;
}

// =================================================================================================
//  section-7: Qualcomm compatibility layer(ported from Qualcomm's ggml-hexagon)
// =================================================================================================

static inline size_t htp_mm_hvx_get_vtcm_sizes(
    int kernel_type, int wtype, uint32_t ne10, uint32_t src1_nrows,
    uint32_t n_threads,
    size_t dst_row_size, size_t src0_row_size, size_t src1_row_size,
    uint32_t n_prefetch,
    size_t * vtcm_src0_size, size_t * vtcm_src1_size, size_t * vtcm_dst_size
) {
    struct htp_mm_hvx_vtcm_layout vtcm_layout;
    htp_mm_hvx_vtcm_layout_build(&vtcm_layout, kernel_type, wtype, ne10, src1_nrows, n_threads,
                                 dst_row_size, src0_row_size, src1_row_size, n_prefetch,
                                 false, false, false);
    *vtcm_src0_size = vtcm_layout.src0_bytes;
    *vtcm_src1_size = vtcm_layout.src1_bytes;
    *vtcm_dst_size  = vtcm_layout.dst_bytes;
    return vtcm_layout.total_bytes;
}

static inline size_t htp_mm_hvx_id_get_vtcm_sizes(
    int wtype, uint32_t ne10, uint32_t src1_nrows, uint32_t n_threads,
    size_t src0_row_size, uint32_t n_prefetch,
    size_t * vtcm_src0_size, size_t * vtcm_src1_size, size_t * vtcm_dst_size
) {
    struct htp_mm_hvx_vtcm_layout vtcm_layout;
    htp_mm_hvx_vtcm_layout_build(&vtcm_layout, 0, wtype, ne10, src1_nrows, n_threads,
                                 0, src0_row_size, 0, n_prefetch,
                                 true, false, false);
    *vtcm_src0_size = vtcm_layout.src0_bytes;
    *vtcm_src1_size = vtcm_layout.src1_bytes;
    *vtcm_dst_size  = vtcm_layout.dst_bytes;
    return vtcm_layout.total_bytes;
}

// FA kernel selection: 2 = HMX -> HVX -> CPU, 1 = HVX -> CPU, 0 = CPU (unsupported)
// Controlled by ggml-hexagon.cfg: [cdsp] fa_select
static int ggml_hexagon_get_fa_select(void) {
    return g_hexagon_appcfg.fa_select;
}

// Precompute htp_fa_kernel_params on AP side for FLASH_ATTN_EXT.
// Writes to kparams; caller casts from op.kernel_params or a stack local.
// Returns true if a valid kernel (HMX or HVX) was selected.
static bool ggml_hexagon_compute_fa_params(
    const ggml_backend_hexagon_context * ctx,
    const ggml_tensor * node,
    struct htp_fa_kernel_params * kparams
) {
    if (ggml_hexagon_get_fa_select() < 1) {
        return false;
    }

    memset(kparams, 0, sizeof(*kparams));

    const ggml_tensor * q    = node->src[0];
    const ggml_tensor * k    = node->src[1];
    const ggml_tensor * v    = node->src[2];
    const ggml_tensor * mask = node->src[3];
    const ggml_tensor * dst  = node;

    const uint32_t DK = (uint32_t) q->ne[0];
    const uint32_t DV = (uint32_t) v->ne[0];
    const uint32_t neq1 = (uint32_t) q->ne[1];
    const uint32_t nek1 = (uint32_t) k->ne[1];
    const uint32_t n_kv_heads = (uint32_t) k->ne[2];
    const uint32_t G = (uint32_t) q->ne[2] / n_kv_heads;

    float scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&scale,         &node->op_params[0], sizeof(float));
    memcpy(&max_bias,      &node->op_params[1], sizeof(float));
    memcpy(&logit_softcap, &node->op_params[2], sizeof(float));
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    kparams->scale         = scale;
    kparams->max_bias      = max_bias;
    kparams->logit_softcap = logit_softcap;
    kparams->is_q_fp32     = (q->type == GGML_TYPE_F32) ? 1 : 0;
    kparams->is_dst_fp32   = (dst->type == GGML_TYPE_F32) ? 1 : 0;
    kparams->G             = G;

    const uint32_t n_head = (uint32_t) q->ne[2];
    // largest power of 2 <= n_head
    uint32_t n_head_log2 = 1;
    while (n_head_log2 * 2u <= n_head) n_head_log2 *= 2;
    kparams->n_head_log2 = n_head_log2;
    // 2^x = exp(x * ln2), avoiding powf dependency
    const float ln2 = 0.6931471805599453f;
    kparams->m0 = expf(-ln2 * max_bias / (float) n_head_log2);
    kparams->m1 = expf(-ln2 * (max_bias * 0.5f) / (float) n_head_log2);

    // HMX eligibility
    bool hmx_eligible = false;
    if (ctx->has_hmx && ggml_hexagon_get_fa_select() >= 2 &&
        k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16) {
        if (DK % 64 == 0 && DV % 64 == 0 && !(DK <= 128 && neq1 < 5)) {
            hmx_eligible = true;
        }
    }

    if (hmx_eligible) {
        size_t Br = 0, Bc = 0;
        const size_t vtcm_budget = ctx->socinfo.vtcm_size_in_mb * 1024 * 1024;
        int ret = hmx_fa_find_chunk_size(&Br, &Bc, G, DK, DV, neq1, nek1,
                                         vtcm_budget, (size_t) ctx->n_threads);
        if (ret == 0) {
            kparams->kernel_type = HTP_FA_KERNEL_HMX;
            kparams->Br          = (uint16_t) Br;
            kparams->Bc          = (uint16_t) Bc;
            kparams->n_kv_blocks = (uint16_t)((nek1 + Bc - 1) / Bc);
            kparams->n_threads   = (kparams->n_kv_blocks >= 3 && ctx->n_threads >= 2)
                                   ? (uint8_t) ctx->n_threads : 1;
            kparams->u.hmx.g_br      = hex_align_up(G * Br, 32);
            kparams->u.hmx.pipeline  = (kparams->n_kv_blocks >= 3 && ctx->n_threads >= 2) ? 1 : 0;
            kparams->vtcm_size       = (uint32_t) hmx_fa_compute_vtcm_usage(
                G, DK, DV, Br, Bc, kparams->n_threads, kparams->u.hmx.pipeline != 0);

            const size_t row_vec_bytes = hex_align_up(Bc * sizeof(uint16_t), 256);
            kparams->u.hmx.row_buf_stride = row_vec_bytes / 128;
            const size_t m_line_bytes = hex_align_up(Bc * sizeof(uint16_t), 128);
            kparams->u.hmx.mask_buf_row_stride = m_line_bytes / sizeof(uint16_t);
            kparams->u.hmx.mask_broadcast = (mask && mask->ne[2] == 1) ? 1 : 0;
            kparams->u.hmx.div_G = init_fastdiv_values(G);
            if (mask) {
                kparams->src3_div2 = init_fastdiv_values((uint32_t) mask->ne[2]);
                kparams->src3_div3 = init_fastdiv_values((uint32_t) mask->ne[3]);
            }
            kparams->qrows = 0;
            kparams->qrows_per_thread = 0;
            return true;
        }
    }

    // Fallback to HVX
    kparams->kernel_type    = HTP_FA_KERNEL_HVX;
    kparams->Br             = 1;
    kparams->Bc             = 64;
    kparams->n_kv_blocks    = (uint16_t)((k->ne[1] + 64 - 1) / 64);
    kparams->n_threads      = (uint8_t) ctx->n_threads;
    kparams->vtcm_size      = (uint32_t) hvx_fa_compute_vtcm_usage(
        DK, DV, kparams->is_q_fp32 != 0, mask != nullptr, (size_t) ctx->n_threads);

    kparams->u.hvx.size_q_row_padded = hex_round_up((uint32_t)(q->ne[0] * (kparams->is_q_fp32 ? 4 : 2)), 128);
    kparams->u.hvx.size_k_row_padded = hex_round_up((uint32_t)(k->ne[0] * 2), 128);
    kparams->u.hvx.size_v_row_padded = hex_round_up((uint32_t)(v->ne[0] * 2), 128);
    kparams->u.hvx.src0_div21     = init_fastdiv_values((uint32_t)(q->ne[2] * q->ne[1]));
    kparams->u.hvx.src0_div1      = init_fastdiv_values((uint32_t) q->ne[1]);
    kparams->broadcast_rk2   = init_fastdiv_values((uint32_t)(q->ne[2] / k->ne[2]));
    kparams->broadcast_rk3   = init_fastdiv_values((uint32_t)(q->ne[3] / k->ne[3]));
    kparams->broadcast_rv2   = init_fastdiv_values((uint32_t)(q->ne[2] / v->ne[2]));
    kparams->broadcast_rv3   = init_fastdiv_values((uint32_t)(q->ne[3] / v->ne[3]));
    if (mask) {
        kparams->src3_div2 = init_fastdiv_values((uint32_t) mask->ne[2]);
        kparams->src3_div3 = init_fastdiv_values((uint32_t) mask->ne[3]);
    }
    kparams->qrows           = (uint32_t)(q->ne[1] * q->ne[2] * q->ne[3]);
    kparams->qrows_per_thread = (kparams->qrows + ctx->n_threads - 1) / ctx->n_threads;
    return true;
}

// Map GGML opcode to HTP opcode for unary-family ops. Mirrors the DSP-side
// ggml_op_to_htp_op() in htp/entry.c, restricted to the subset that
// htp_op_is_unary() in unary-ops.h accepts.
// Returns false if the op is not a precompute-required unary.
static bool ggml_op_to_htp_op_unary(int32_t ggml_op, const int32_t * op_params, uint32_t * htp_op) {
    switch (ggml_op) {
        case GGML_OP_NORM:    *htp_op = HTP_OP_NORM;        return true;
        case GGML_OP_L2_NORM: *htp_op = HTP_OP_L2_NORM;     return true;
        case GGML_OP_RMS_NORM:*htp_op = HTP_OP_RMS_NORM;    return true;
        case GGML_OP_SCALE:   *htp_op = HTP_OP_SCALE;       return true;
        case GGML_OP_SQR:     *htp_op = HTP_OP_SQR;         return true;
        case GGML_OP_SQRT:    *htp_op = HTP_OP_SQRT;        return true;
        case GGML_OP_TRI:     *htp_op = HTP_OP_TRI;         return true;
        case GGML_OP_UNARY:
            if (!op_params) return false;
            switch (op_params[0]) {
                case GGML_UNARY_OP_NEG:        *htp_op = HTP_OP_UNARY_NEG;      return true;
                case GGML_UNARY_OP_TANH:       *htp_op = HTP_OP_UNARY_TANH;     return true;
                case GGML_UNARY_OP_SIGMOID:    *htp_op = HTP_OP_UNARY_SIGMOID;  return true;
                case GGML_UNARY_OP_EXP:        *htp_op = HTP_OP_UNARY_EXP;      return true;
                case GGML_UNARY_OP_SOFTPLUS:   *htp_op = HTP_OP_UNARY_SOFTPLUS; return true;
                case GGML_UNARY_OP_SILU:       *htp_op = HTP_OP_UNARY_SILU;     return true;
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK: *htp_op = HTP_OP_UNARY_GELU;     return true;
                default: return false;
            }
        default:
            return false;
    }
}

// Precompute htp_unary_kernel_params on AP side for unary ops (NORM, RMS_NORM,
// RMS_NORM_MUL, SCALE, SQR, SQRT, UNARY_*, L2_NORM, TRI).
static void ggml_hexagon_precompute_unary_params(
    const ggml_backend_hexagon_context * ctx,
    uint32_t op,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    const ggml_tensor * dst,
    struct htp_unary_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const uint32_t src0_nrows = (uint32_t)(src0->ne[1] * src0->ne[2] * src0->ne[3]);
    const uint32_t n_threads  = (ctx->n_threads < (int) src0_nrows) ? (uint32_t) ctx->n_threads : src0_nrows;

    kparams->n_threads = n_threads;

    const size_t src0_data_row_size = (size_t) src0->ne[0] * sizeof(float);
    const size_t dst_data_row_size  = (size_t) dst->ne[0]  * sizeof(float);

    const size_t src0_row_size_aligned = hex_round_up((uint32_t) src0_data_row_size, 128);
    const size_t dst_row_size_aligned  = hex_round_up((uint32_t) dst_data_row_size,  128);

    kparams->src0_row_size_aligned = (uint32_t) src0_row_size_aligned;
    kparams->dst_row_size_aligned  = (uint32_t) dst_row_size_aligned;

    size_t src1_data_row_size = 0;
    size_t src1_row_size_aligned = 0;
    bool broadcast_weight = false;

    if (op == HTP_OP_RMS_NORM_MUL) {
        GGML_ASSERT(src1 != nullptr);
        src1_data_row_size = (size_t) src1->ne[0] * sizeof(float);
        src1_row_size_aligned = hex_round_up((uint32_t) src1_data_row_size, 128);
        broadcast_weight = (src1->ne[1] * src1->ne[2] * src1->ne[3] == 1);
    }

    kparams->src1_row_size_aligned = (uint32_t) src1_row_size_aligned;
    kparams->broadcast_weight      = broadcast_weight ? 1u : 0u;

    const size_t vtcm_size_budget = ctx->socinfo.vtcm_size_in_mb * 1024ull * 1024ull;

    struct htp_unary_vtcm_layout L;
    uint32_t col_tile = 0;
    uint32_t vtcm_row_per_thread = 0;

    htp_unary_vtcm_layout_build(&L, op, (uint32_t) src0->ne[0], (uint32_t) dst->ne[0],
                                op == HTP_OP_RMS_NORM_MUL ? (uint32_t) src1->ne[0] : 0,
                                broadcast_weight, n_threads, vtcm_size_budget,
                                &col_tile, &vtcm_row_per_thread);

    kparams->col_tile              = col_tile;
    kparams->vtcm_row_per_thread   = vtcm_row_per_thread;
    kparams->vtcm_size             = (uint32_t) L.total_bytes;

    kparams->vtcm_src0_size_per_thread = (uint32_t) L.src0_bytes;
    kparams->vtcm_src1_size_per_thread = (uint32_t) L.src1_bytes;
    kparams->vtcm_dst_size_per_thread  = (uint32_t) L.dst_bytes;

    kparams->vtcm_src0_size = (uint32_t)(L.src0_bytes * n_threads);
    kparams->vtcm_src1_size = (uint32_t)(L.src1_bytes * n_threads);
    kparams->vtcm_dst_size  = (uint32_t)(L.dst_bytes  * n_threads);

    kparams->block = col_tile ? 0u : (uint32_t) ((L.src0_bytes / 2) / src0_row_size_aligned);

    const uint32_t tiles_per_row = col_tile > 0 ? ((uint32_t) src0->ne[0] + col_tile - 1) / col_tile : 1u;
    kparams->div_ne01  = init_fastdiv_values((uint32_t) src0->ne[1]);
    kparams->div_ne02  = init_fastdiv_values((uint32_t) src0->ne[2]);
    kparams->div_ne012 = init_fastdiv_values((uint32_t)(src0->ne[1] * src0->ne[2]));
    kparams->div_tpr   = init_fastdiv_values(tiles_per_row);
}

static bool ggml_hexagon_matmul_is_hmx_eligible(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    int ne01_padded,
    bool is_matmul_id,
    bool is_batched
) {
    const int ne00  = src0->ne[0];
    const int ne11  = src1->ne[1];
    const int ne12  = src1->ne[2];
    // weights may be stored in a different format (see set_tensor)
    const int wtype = ggml_hexagon_weight_dsp_type(src0->type);

    if (ne01_padded % 32 != 0) {
        return false;
    }

    if (!ggml_hexagon_is_hmx_weight_type((ggml_type) wtype)) {
        return false;
    }

    if (ne00 % 32 != 0) {
        return false;
    }

    if (!is_matmul_id && is_batched && wtype != GGML_TYPE_F16) {
        return false;
    }

    if (src0->nb[0] > src0->nb[1] || src1->nb[0] > src1->nb[1]) {
        return false;
    }

    const int m = is_matmul_id ? ne12 : ne11;
    if (m <= HTP_MM_HMX_MIN_NROWS) {
        return false;
    }

    return true;
}

// Shared HMX eligibility check: computes standard params from src0/src1/dst
// and delegates to ggml_hexagon_matmul_is_hmx_eligible. Used by both
// mm_is_hmx_eligible (opfusion gate) and precompute_mm_params (HMX delegation)
// to keep the two decision points consistent.
static bool ggml_hexagon_mm_is_hmx_eligible_shared(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst
) {
    const int wtype = src0->type;
    const bool is_repack    = ggml_hexagon_is_repack_type((ggml_type) wtype);
    const bool is_matmul_id = (dst->op == GGML_OP_MUL_MAT_ID);
    const bool is_batched   = (src0->ne[2] * src0->ne[3] > 1 || src1->ne[2] * src1->ne[3] > 1);
    const int  ne01_padded  = is_repack ? (int) hex_round_up((uint32_t) src0->ne[1], 32) : src0->ne[1];

    return ggml_hexagon_matmul_is_hmx_eligible(src0, src1, dst, ne01_padded, is_matmul_id, is_batched);
}

// Gate for QKV/FFN fusion eligibility: returns true when the MUL_MAT
// is suitable for HMX. Consulted only by is_mergeable_mul_mat to avoid
// merging MUL_MATs that would benefit from HMX.
// HMX dispatch itself is decided independently by ggml_hexagon_precompute_mm_params.
static bool mm_is_hmx_eligible(const ggml_backend_hexagon_context * ctx, const ggml_tensor * t) {
    if (!ctx->has_hmx) return false;
    return ggml_hexagon_mm_is_hmx_eligible_shared(t->src[0], t->src[1], t);
}

// A MUL_MAT is fusion-eligible when:
//   - src0 is quantized (Q4_0/Q8_0/etc.)
//   - src1 is F32 (fusion kernels read F32 activations)
//   - !mm_is_hmx_eligible (avoid merging MUL_MATs that would benefit from HMX)
// NOTE: fusion is only attempted for MUL_MATs that match the QKV/FFN pattern.
static bool is_mergeable_mul_mat(const ggml_backend_hexagon_context * ctx, const ggml_tensor * t) {
    if (!t || t->op != GGML_OP_MUL_MAT)   return false;
    if (t->src[1]->type != GGML_TYPE_F32) return false;
    return ggml_is_quantized(t->src[0]->type) && !mm_is_hmx_eligible(ctx, t);
}

static bool is_mergeable_mul_mat_pair(const ggml_backend_hexagon_context * ctx, const ggml_tensor * n1, const ggml_tensor * n2) {
    if (!is_mergeable_mul_mat(ctx, n1) || !is_mergeable_mul_mat(ctx, n2)) {
        return false;
    }
    if (n1->src[1] != n2->src[1]) {
        return false;
    }
    if (n1->src[0]->ne[0] != n2->src[0]->ne[0] ||
        n1->src[0]->ne[1] != n2->src[0]->ne[1]) {
        return false;
    }
    if (n1->src[0]->type != n2->src[0]->type) {
        return false;
    }
    return true;
}

static bool is_qkv_mergeable(const ggml_backend_hexagon_context * ctx, const ggml_tensor * n_q, const ggml_tensor * n_k, const ggml_tensor * n_v) {
    if (!is_mergeable_mul_mat(ctx, n_q) || !is_mergeable_mul_mat(ctx, n_k) || !is_mergeable_mul_mat(ctx, n_v)) {
        return false;
    }
    if (n_q->src[1] != n_k->src[1] || n_q->src[1] != n_v->src[1]) {
        return false;
    }
    if (n_q->src[0]->type != n_k->src[0]->type || n_q->src[0]->type != n_v->src[0]->type) {
        return false;
    }
    if (n_k->src[0]->ne[0] != n_v->src[0]->ne[0] ||
        n_k->src[0]->ne[1] != n_v->src[0]->ne[1]) {
        return false;
    }
    if (n_q->src[0]->ne[0] != n_k->src[0]->ne[0]) {
        return false;
    }
    return true;
}

// Precompute htp_mm_kernel_params for fused QKV matmul (3 outputs: K, V, Q).
// src0 = Wk (representative of K/V/Q weights), src1 = x (shared activation).
// DSP-side op_matmul_qkv expects src[0]=Wk, src[1]=x, src[2]=Wv, src[3]=Wq.
static void ggml_hexagon_precompute_fused_qkv_params(
    const ggml_backend_hexagon_context * ctx,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const int wtype = src0->type;
    const bool is_repack = ggml_hexagon_is_repack_type((ggml_type) wtype);

    const int ne10 = src1->ne[0];
    const int src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
    const size_t src1_row_size = (wtype == GGML_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
    const size_t src0_row_size = src0->nb[1];
    const size_t src0_row_size_padded = hex_round_up((uint32_t) src0_row_size, 128);

    size_t src0_sz_per_thread = 0;
    size_t src2_sz_per_thread = 0;
    size_t src3_sz_per_thread = 0;
    uint32_t best_n_prefetch = 16;

    const size_t vtcm_budget = ctx->socinfo.vtcm_size_in_mb * 1024 * 1024;
    size_t quant_scratch_size = hex_round_up((uint32_t)(ne10 * sizeof(float)), QK_Q8_0_TILED * sizeof(float)) * (uint32_t)ctx->n_threads;

    if (is_repack) {
        uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(wtype);
        uint32_t n_k_tiles = hex_round_up((uint32_t) ne10, 32) / 32;
        uint32_t tile_row_size = n_k_tiles * aligned_tile_size;
        size_t src1_sz_per_thread = hex_round_up((uint32_t)(src1_row_size * src1_nrows), 128);
        size_t src1_sz = src1_sz_per_thread;

        const uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
        best_n_prefetch = 2;
        for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
            size_t repacked_vtcm_size = hex_round_up(d * tile_row_size, 128);
            size_t src0_sz = repacked_vtcm_size * (uint32_t)ctx->n_threads;
            size_t src2_sz = hex_round_up(d * tile_row_size, 128) * (uint32_t)ctx->n_threads;
            size_t src3_sz = hex_round_up(d * tile_row_size, 128) * (uint32_t)ctx->n_threads;
            size_t tiled_vtcm_size = src0_sz + src1_sz + src2_sz + src3_sz + quant_scratch_size;

            if (tiled_vtcm_size <= vtcm_budget) {
                best_n_prefetch = d;
                src0_sz_per_thread = repacked_vtcm_size;
                src2_sz_per_thread = hex_round_up(d * tile_row_size, 128);
                src3_sz_per_thread = hex_round_up(d * tile_row_size, 128);
                break;
            }
        }
        if (best_n_prefetch == 2 && src0_sz_per_thread == 0) {
            size_t repacked_vtcm_size = hex_round_up(2 * tile_row_size, 128);
            src0_sz_per_thread = repacked_vtcm_size;
            src2_sz_per_thread = hex_round_up(2 * tile_row_size, 128);
            src3_sz_per_thread = hex_round_up(2 * tile_row_size, 128);
        }
    } else {
        best_n_prefetch = 16;
        src0_sz_per_thread = hex_round_up((uint32_t)(best_n_prefetch * src0_row_size_padded), 128);
        src2_sz_per_thread = hex_round_up((uint32_t)(best_n_prefetch * src0_row_size_padded), 128);
        src3_sz_per_thread = hex_round_up((uint32_t)(best_n_prefetch * src0_row_size_padded), 128);
    }

    size_t src1_sz_per_thread = hex_round_up((uint32_t)(src1_row_size * src1_nrows), 128);

    size_t src0_sz = src0_sz_per_thread * (uint32_t)ctx->n_threads;
    size_t src1_sz = src1_sz_per_thread;
    size_t src2_sz = src2_sz_per_thread * (uint32_t)ctx->n_threads;
    size_t src3_sz = src3_sz_per_thread * (uint32_t)ctx->n_threads;

    size_t tiled_vtcm_size = src0_sz + src1_sz + src2_sz + src3_sz + quant_scratch_size;
    bool try_tiled = true;
    if (try_tiled && tiled_vtcm_size <= vtcm_budget) {
        kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW;
        kparams->vtcm_src0_size = (int32_t) src0_sz;
        kparams->vtcm_src1_size = (int32_t) src1_sz;
        kparams->vtcm_src2_size = (int32_t) src2_sz;
        kparams->vtcm_src3_size = (int32_t) src3_sz;
        kparams->vtcm_dst_size  = (int32_t) quant_scratch_size;
        kparams->vtcm_size      = (int32_t) tiled_vtcm_size;
        kparams->n_prefetch     = (int32_t) best_n_prefetch;
    } else {
        kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT;
        size_t flat_src1_row_size = (wtype == GGML_TYPE_Q4_1) ? htp_mm_q8_1_flat_row_size(ne10) : htp_mm_q8_0_flat_row_size(ne10);
        size_t flat_src1_sz = hex_round_up((uint32_t)(flat_src1_row_size * src1_nrows), 128);
        kparams->vtcm_src0_size = (int32_t) src0_sz;
        kparams->vtcm_src1_size = (int32_t) flat_src1_sz;
        kparams->vtcm_src2_size = (int32_t) src2_sz;
        kparams->vtcm_src3_size = (int32_t) src3_sz;
        kparams->vtcm_dst_size  = (int32_t) quant_scratch_size;
        kparams->vtcm_size      = (int32_t)(src0_sz + flat_src1_sz + src2_sz + src3_sz + quant_scratch_size);
        kparams->n_prefetch     = (int32_t) best_n_prefetch;
    }
}

// Precompute htp_mm_kernel_params for fused FFN matmul (2 outputs: gate, up).
// src0 = Wgate, src1 = y (shared activation).
// DSP-side op_matmul_ffn expects src[0]=Wgate, src[1]=y, src[2]=Wup.
static void ggml_hexagon_precompute_fused_ffn_params(
    const ggml_backend_hexagon_context * ctx,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct htp_mm_kernel_params * kparams
) {
    memset(kparams, 0, sizeof(*kparams));

    const int wtype = src0->type;
    const bool is_repack = ggml_hexagon_is_repack_type((ggml_type) wtype);

    const int ne10 = src1->ne[0];
    const int src1_nrows = src1->ne[1] * src1->ne[2] * src1->ne[3];
    const size_t src1_row_size = (wtype == GGML_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
    const size_t src0_row_size = src0->nb[1];
    const size_t src0_row_size_padded = hex_round_up((uint32_t) src0_row_size, 128);

    size_t src0_sz_per_thread = 0;
    size_t src2_sz_per_thread = 0;
    uint32_t best_n_prefetch = 16;

    const size_t vtcm_budget = ctx->socinfo.vtcm_size_in_mb * 1024 * 1024;
    size_t quant_scratch_size = hex_round_up((uint32_t)(ne10 * sizeof(float)), QK_Q8_0_TILED * sizeof(float)) * (uint32_t)ctx->n_threads;

    if (is_repack) {
        uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(wtype);
        uint32_t n_k_tiles = hex_round_up((uint32_t) ne10, 32) / 32;
        uint32_t tile_row_size = n_k_tiles * aligned_tile_size;
        size_t src1_sz_per_thread = hex_round_up((uint32_t)(src1_row_size * src1_nrows), 128);
        size_t src1_sz = src1_sz_per_thread;

        const uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
        best_n_prefetch = 2;
        for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
            size_t repacked_vtcm_size = hex_round_up(d * tile_row_size, 128);
            size_t src0_sz = repacked_vtcm_size * (uint32_t)ctx->n_threads;
            size_t src2_sz = hex_round_up(d * tile_row_size, 128) * (uint32_t)ctx->n_threads;
            size_t tiled_vtcm_size = src0_sz + src1_sz + src2_sz + quant_scratch_size;

            if (tiled_vtcm_size <= vtcm_budget) {
                best_n_prefetch = d;
                src0_sz_per_thread = repacked_vtcm_size;
                src2_sz_per_thread = hex_round_up(d * tile_row_size, 128);
                break;
            }
        }
        if (best_n_prefetch == 2 && src0_sz_per_thread == 0) {
            size_t repacked_vtcm_size = hex_round_up(2 * tile_row_size, 128);
            src0_sz_per_thread = repacked_vtcm_size;
            src2_sz_per_thread = hex_round_up(2 * tile_row_size, 128);
        }
    } else {
        best_n_prefetch = 16;
        src0_sz_per_thread = hex_round_up((uint32_t)(best_n_prefetch * src0_row_size_padded), 128);
        src2_sz_per_thread = hex_round_up((uint32_t)(best_n_prefetch * src0_row_size_padded), 128);
    }

    size_t src1_sz_per_thread = hex_round_up((uint32_t)(src1_row_size * src1_nrows), 128);

    size_t src0_sz = src0_sz_per_thread * (uint32_t)ctx->n_threads;
    size_t src1_sz = src1_sz_per_thread;
    size_t src2_sz = src2_sz_per_thread * (uint32_t)ctx->n_threads;

    size_t tiled_vtcm_size = src0_sz + src1_sz + src2_sz + quant_scratch_size;
    bool try_tiled = true;
    if (try_tiled && tiled_vtcm_size <= vtcm_budget) {
        kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW;
        kparams->vtcm_src0_size = (int32_t) src0_sz;
        kparams->vtcm_src1_size = (int32_t) src1_sz;
        kparams->vtcm_src2_size = (int32_t) src2_sz;
        kparams->vtcm_dst_size  = (int32_t) quant_scratch_size;
        kparams->vtcm_size      = (int32_t) tiled_vtcm_size;
        kparams->n_prefetch     = (int32_t) best_n_prefetch;
    } else {
        kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT;
        size_t flat_src1_row_size = (wtype == GGML_TYPE_Q4_1) ? htp_mm_q8_1_flat_row_size(ne10) : htp_mm_q8_0_flat_row_size(ne10);
        size_t flat_src1_sz = hex_round_up((uint32_t)(flat_src1_row_size * src1_nrows), 128);
        kparams->vtcm_src0_size = (int32_t) src0_sz;
        kparams->vtcm_src1_size = (int32_t) flat_src1_sz;
        kparams->vtcm_src2_size = (int32_t) src2_sz;
        kparams->vtcm_dst_size  = (int32_t) quant_scratch_size;
        kparams->vtcm_size      = (int32_t)(src0_sz + flat_src1_sz + src2_sz + quant_scratch_size);
        kparams->n_prefetch     = (int32_t) best_n_prefetch;
    }
}

// Precompute htp_mm_kernel_params on AP side for MUL_MAT in FastRPC/mempool batch path.
// Mirrors build_mm_kernel_params in kernels/entry.c (F32/F16 HVX paths only).
// Writes directly to op.kernel_params; DSP side consumes via memcpy.
// For unsupported weight types (quant/HMX), leaves kernel_type=0 so DSP falls
// back to build_mm_kernel_params which emits the error.
// When is_matmul_id=false, the node is a plain MUL_MAT (not MUL_MAT_ID).
// =======================================================================
//   ggml_hexagon_precompute_matmul_params
//
// The HMX-first-then-HVX-fallback policy is preserved.
// =======================================================================
static bool ggml_hexagon_precompute_hmx_mm_params(
    const ggml_backend_hexagon_context * ctx,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    int wtype,
    int ne00_padded,
    int ne01_padded,
    int ne02,
    int ne11,
    int ne12,
    int ne11_padded,
    bool is_matmul_id,
    bool is_batched,
    size_t vtcm_budget,
    struct htp_mm_kernel_params * kparams
) {
    const int aligned_tile_size = htp_mm_get_weight_aligned_tile_size(wtype);
    // Force the pipelined path for plain MUL_MAT (MUL_MAT_ID keeps the
    // non-pipelined setting, matching upstream): the synchronous (m<=32)
    // branch yields corrupted, non-deterministic output in this integration
    // (observed with ubatch<=32); upstream's own backend is unaffected.
    const bool pipeline = is_matmul_id ? false : true;
    const int n_threads = (int) ctx->n_threads;
    const int ne10      = src1->ne[0];

    const bool is_batched_val   = is_matmul_id ? false : is_batched;
    const int group_size        = (ne02 > 0 ? ne12 / ne02 : 1);

    size_t m_chunk              = 0;
    size_t n_chunk              = 0;
    size_t vtcm_size            = 0;
    bool use_grouped            = false;
    int act_threads_selected    = 0;

    if (is_batched_val && wtype == GGML_TYPE_F16 && group_size > 1) {
        // Try grouped path first
        const bool use_dma_activation = (src1->nb[1]/sizeof(float) > (size_t)ne00_padded);
        if (htp_mm_hmx_solve_batched_params(wtype, ne00_padded, ne01_padded, ne11,
                    group_size, use_dma_activation, n_threads, pipeline,
                    vtcm_budget, &m_chunk, &n_chunk,
                    &act_threads_selected, &vtcm_size)) {
            use_grouped = true;
        }
    }

    if (!use_grouped) {
        // Fallback to simple 2D path (group_size = 1)
        const int m_id_rows = (int) ((size_t) dst->ne[1] * dst->ne[2]);
        if (!htp_mm_hmx_solve_2d_params(wtype, ne00_padded, m_id_rows,
                    ne01_padded, ne11_padded, ne11, n_threads, pipeline,
                    is_matmul_id, aligned_tile_size, vtcm_budget,
                    &m_chunk, &n_chunk, &act_threads_selected, &vtcm_size)) {
            return false;
        }
    }

    kparams->n_hmx              = 1;
    kparams->pipeline           = pipeline ? 1 : 0;
    kparams->m_chunk            = (int32_t) m_chunk;
    kparams->n_chunk            = (int32_t) n_chunk;
    kparams->n_threads          = n_threads;
    kparams->n_act_threads      = act_threads_selected;
    kparams->tile_size          = (int32_t) htp_mm_get_weight_tile_size(wtype);
    kparams->aligned_tile_size  = (int32_t) aligned_tile_size;
    kparams->src1_row_size      = (int32_t)((wtype == GGML_TYPE_Q4_1)
                                        ? htp_mm_q8_1_tiled_row_size(ne10)
                                        : htp_mm_q8_0_tiled_row_size(ne10));
    kparams->vtcm_size          = (int32_t) vtcm_size;
    kparams->vtcm_src0_size     = 0;
    kparams->div_n_act_threads  = init_fastdiv_values((uint32_t) act_threads_selected);
    kparams->div_ne00_padded    = init_fastdiv_values((uint32_t) ne00_padded);
    kparams->vtcm_src1_size     = 0;
    kparams->vtcm_dst_size      = 0;

    if (is_batched && !is_matmul_id) {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_F16_BATCHED;
    } else {
        kparams->kernel_type = HTP_MM_KERNEL_HMX_2D;
    }
    return true;
}

static void ggml_hexagon_precompute_hvx_mm_params(
    const ggml_backend_hexagon_context * ctx,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * dst,
    int wtype,
    int ne02,
    int ne03,
    int ne10,
    int ne11,
    int ne12,
    int ne13,
    bool is_matmul_id,
    size_t vtcm_budget,
    struct htp_mm_kernel_params * kparams
) {
    kparams->n_hmx = 0;

    const bool is_quant     = (wtype != GGML_TYPE_F16 && wtype != GGML_TYPE_F32);
    const int src1_nrows    = ne11 * ne12 * ne13;

    if (is_quant) {
        // Quantized HVX
        kparams->tile_size = (int32_t) htp_mm_get_weight_tile_size(wtype);
        kparams->aligned_tile_size = (int32_t) htp_mm_get_weight_aligned_tile_size(wtype);

        const bool k_align = (ne10 % 32 == 0);

        if (is_matmul_id) {
            kparams->kernel_type   = (src1_nrows < (int) ctx->n_threads)
                ? HTP_MM_KERNEL_HVX_QUANT_BLOCK : HTP_MM_KERNEL_HVX_QUANT_ROW;
            kparams->src1_row_size = (int32_t)((wtype == GGML_TYPE_Q4_1)
                ? htp_mm_q8_1_tiled_row_size(ne10)
                : htp_mm_q8_0_tiled_row_size(ne10));

            size_t vtcm_src0_size = 0, vtcm_src1_size = 0, vtcm_dst_size = 0;
            uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
            uint32_t best_n_prefetch = 2;
            size_t total_size = 0;
            for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                total_size = htp_mm_hvx_id_get_vtcm_sizes(
                    wtype, ne10, src1_nrows, (uint32_t)ctx->n_threads,
                    src0->nb[1], d,
                    &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size
                );
                if (total_size <= vtcm_budget) {
                    best_n_prefetch = d;
                    break;
                }
            }
            if (best_n_prefetch == 2 && total_size > vtcm_budget) {
                total_size = htp_mm_hvx_id_get_vtcm_sizes(
                    wtype, ne10, src1_nrows, (uint32_t)ctx->n_threads,
                    src0->nb[1], 2,
                    &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size
                );
            }
            kparams->n_prefetch = (int32_t) best_n_prefetch;
            kparams->vtcm_size      = (int32_t) total_size;
            kparams->vtcm_src0_size = (int32_t) vtcm_src0_size;
            kparams->vtcm_src1_size = (int32_t) vtcm_src1_size;
            kparams->vtcm_dst_size  = (int32_t) vtcm_dst_size;
        } else {
            if (k_align) {
                kparams->src1_row_size = (int32_t)((wtype == GGML_TYPE_Q4_1)
                    ? htp_mm_q8_1_tiled_row_size(ne10)
                    : htp_mm_q8_0_tiled_row_size(ne10));
                if (src1_nrows < (int)ctx->n_threads) {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_BLOCK;
                } else {
                    kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW;
                }

                uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
                uint32_t best_n_prefetch = 2;
                size_t vtcm_src0_size = 0, vtcm_src1_size = 0, vtcm_dst_size = 0;
                size_t total_size = 0;
                for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                    total_size = htp_mm_hvx_get_vtcm_sizes(
                        kparams->kernel_type, wtype, ne10, src1_nrows,
                        (uint32_t)ctx->n_threads,
                        dst->nb[1], src0->nb[1], src1->nb[1], d,
                        &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size
                    );
                    if (total_size <= vtcm_budget) {
                        best_n_prefetch = d;
                        break;
                    }
                }
                if (best_n_prefetch == 2 && total_size > vtcm_budget) {
                    total_size = htp_mm_hvx_get_vtcm_sizes(
                        kparams->kernel_type, wtype, ne10, src1_nrows,
                        (uint32_t)ctx->n_threads,
                        dst->nb[1], src0->nb[1], src1->nb[1], 2,
                        &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size
                    );
                }

                kparams->n_prefetch = (int32_t) best_n_prefetch;

                if (total_size <= vtcm_budget) {
                    kparams->vtcm_size = (int32_t) total_size;
                    kparams->vtcm_src0_size = (int32_t) vtcm_src0_size;
                    kparams->vtcm_src1_size = (int32_t) vtcm_src1_size;
                    kparams->vtcm_dst_size = (int32_t) vtcm_dst_size;
                    goto done_quant;
                }
                GGMLHEXAGON_LOG_DEBUG("precompute_hvx: tiled path VTCM too large "
                    "(need=%zu budget=%zu), falling back to flat",
                    total_size, vtcm_budget);
            }

            // Flat HVX fallback
            {
                kparams->src1_row_size = (int32_t)((wtype == GGML_TYPE_Q4_1)
                    ? htp_mm_q8_1_flat_row_size(ne10)
                    : htp_mm_q8_0_flat_row_size(ne10));
                kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT;

                size_t vtcm_src0_size = 0, vtcm_src1_size = 0, vtcm_dst_size = 0;
                size_t total_size = htp_mm_hvx_get_vtcm_sizes(
                    kparams->kernel_type, wtype, ne10, src1_nrows,
                    (uint32_t)ctx->n_threads,
                    dst->nb[1], src0->nb[1], src1->nb[1], 16,
                    &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size
                );

                kparams->n_prefetch = 16;
                kparams->vtcm_size = (int32_t) total_size;
                kparams->vtcm_src0_size = (int32_t) vtcm_src0_size;
                kparams->vtcm_src1_size = (int32_t) vtcm_src1_size;
                kparams->vtcm_dst_size = (int32_t) vtcm_dst_size;
            }
        }

    done_quant:;
    } else if (wtype == GGML_TYPE_F16) {
        // F16 HVX
        const bool is_batched  = (ne02 > 1) || (ne03 > 1);
        const bool is_permuted = ggml_is_permuted(src0) || ggml_is_permuted(src1);

        size_t vtcm_src0_size = 0, vtcm_src1_size = 0, vtcm_dst_size = 0;
        size_t vtcm_size = htp_mm_hvx_get_vtcm_sizes(
            HTP_MM_KERNEL_HVX_F16_F16_VTCM, wtype, ne10, src1_nrows,
            (uint32_t)ctx->n_threads,
            dst->nb[1], src0->nb[1], src1->nb[1], 16,
            &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size
        );

        if (!is_batched && !is_permuted && vtcm_size <= vtcm_budget) {
            kparams->kernel_type    = HTP_MM_KERNEL_HVX_F16_F16_VTCM;
            kparams->src1_row_size  = (int32_t) hex_round_up(ne10 * 2, 128);
            kparams->vtcm_size      = (int32_t) vtcm_size;
            kparams->vtcm_src0_size = (int32_t) vtcm_src0_size;
            kparams->vtcm_src1_size = (int32_t) vtcm_src1_size;
            kparams->vtcm_dst_size  = (int32_t) vtcm_dst_size;
            kparams->n_prefetch     = 16;
        } else {
            if (src1->type == GGML_TYPE_F32) {
                kparams->kernel_type = HTP_MM_KERNEL_HVX_F16_F32_DDR;
            } else {
                kparams->kernel_type = HTP_MM_KERNEL_HVX_F16_F16_DDR;
            }
            kparams->src1_row_size  = (int32_t) src1->nb[1];
            size_t ddr_size         = htp_mm_hvx_get_vtcm_sizes(
                kparams->kernel_type, wtype, ne10, src1_nrows,
                (uint32_t)ctx->n_threads,
                dst->nb[1], src0->nb[1], src1->nb[1], 16,
                &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size
            );
            kparams->vtcm_size      = (int32_t) ddr_size;
            kparams->vtcm_src0_size = (int32_t) vtcm_src0_size;
            kparams->vtcm_src1_size = (int32_t) vtcm_src1_size;
            kparams->vtcm_dst_size  = (int32_t) vtcm_dst_size;
            kparams->n_prefetch     = 16;
        }
    } else {
        // F32 HVX
        const bool is_batched  = (ne02 > 1) || (ne03 > 1);
        const bool is_permuted = ggml_is_permuted(src0) || ggml_is_permuted(src1);

        size_t vtcm_src0_size = 0, vtcm_src1_size = 0, vtcm_dst_size = 0;
        size_t vtcm_size = htp_mm_hvx_get_vtcm_sizes(
            HTP_MM_KERNEL_HVX_F32_F32_VTCM, wtype, ne10, src1_nrows,
            (uint32_t)ctx->n_threads,
            dst->nb[1], src0->nb[1], src1->nb[1], 16,
            &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size
        );

        if (!is_batched && !is_permuted && vtcm_size <= vtcm_budget) {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F32_F32_VTCM;
            kparams->src1_row_size = (int32_t) hex_round_up(ne10 * 4, 128);
            kparams->vtcm_size = (int32_t) vtcm_size;
            kparams->vtcm_src0_size = (int32_t) vtcm_src0_size;
            kparams->vtcm_src1_size = (int32_t) vtcm_src1_size;
            kparams->vtcm_dst_size = (int32_t) vtcm_dst_size;
            kparams->n_prefetch = 16;
        } else {
            kparams->kernel_type = HTP_MM_KERNEL_HVX_F32_F32_DDR;
            kparams->src1_row_size = (int32_t) src1->nb[1];
            size_t ddr_size = htp_mm_hvx_get_vtcm_sizes(
                kparams->kernel_type, wtype, ne10, src1_nrows,
                (uint32_t)ctx->n_threads,
                dst->nb[1], src0->nb[1], src1->nb[1], 16,
                &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size
            );
            kparams->vtcm_size = (int32_t) ddr_size;
            kparams->vtcm_src0_size = (int32_t) vtcm_src0_size;
            kparams->vtcm_src1_size = (int32_t) vtcm_src1_size;
            kparams->vtcm_dst_size = (int32_t) vtcm_dst_size;
            kparams->n_prefetch = 16;
        }
    }
}

static void ggml_hexagon_precompute_mm_params(
    ggml_backend_hexagon_context * ctx,
    const ggml_tensor * node,
    hex_op_desc & op,
    bool is_matmul_id
) {
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    const ggml_tensor * dst  = node;

    struct htp_mm_kernel_params * kparams =
        (struct htp_mm_kernel_params *) op.kernel_params;
    memset(kparams, 0, sizeof(*kparams));

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    // weights may be stored in a different format (see set_tensor);
    // select kernels and VTCM layout by the storage type
    const int wtype         = ggml_hexagon_weight_dsp_type(src0->type);
    const bool is_repack    = ggml_hexagon_is_repack_type((ggml_type) wtype);
    const int ne00_padded   = is_repack ? (int) hex_round_up((uint32_t) ne00, 32) : ne00;
    const int ne01_padded   = is_repack ? (int) hex_round_up((uint32_t) ne01, 32) : ne01;
    const int ne11_padded   = (int) hex_round_up((uint32_t) ne11, 32);

    const bool is_batched   = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    const size_t vtcm_budget  = (size_t)ctx->socinfo.vtcm_size_in_mb * 1024 * 1024;

    // Cache key: tensor ptr (unique per weight object) ^ data ptr (stable
    // mempool offset) ^ ne11 (varies for PP batched matmul, fixed for TG).
    // Including src0 tensor ptr avoids mempool-region-reuse collisions: if the
    // mempool is recycled across model loads, the same data ptr may point
    // to a different weight tensor, but src0 will differ.
    const uintptr_t cache_key = (uintptr_t) src0 ^ (uintptr_t) src0->data ^ ((uintptr_t) ne11 << 32);
    auto it = ctx->mm_params_cache.find(cache_key);
    if (it != ctx->mm_params_cache.end()) {
        *kparams = it->second;
        return;
    }

    // HMX-first policy: try HMX precomputation if eligible, fall back to HVX.
    bool hmx_enabled = ctx->has_hmx;
    bool hmx_basic = hmx_enabled && ggml_hexagon_matmul_is_hmx_eligible(
            src0, src1, dst, ne01_padded, is_matmul_id, is_batched);

    // HMX eligibility diagnostic: categorize why basic check failed
    if (!hmx_enabled) {
        // HMX not available on this SoC; nothing to count
    } else if (hmx_basic) {
        ctx->n_hmx_basic_pass++;
    } else {
        // Diagnose which condition failed (order matches ggml_hexagon_matmul_is_hmx_eligible)
        if (ne01_padded % 32 != 0) {
            ctx->n_hmx_basic_fail_ne01++;
        } else if (!ggml_hexagon_is_hmx_weight_type((ggml_type) wtype)) {
            ctx->n_hmx_basic_fail_wtype++;
        } else if (ne00 % 32 != 0) {
            ctx->n_hmx_basic_fail_ne00++;
        } else if (!is_matmul_id && is_batched && wtype != GGML_TYPE_F16) {
            ctx->n_hmx_basic_fail_batched++;
        } else if (src0->nb[0] > src0->nb[1] || src1->nb[0] > src1->nb[1]) {
            ctx->n_hmx_basic_fail_permuted++;
        } else {
            int m = is_matmul_id ? (int)ne12 : (int)ne11;
            if (m <= HTP_MM_HMX_MIN_NROWS) {
                ctx->n_hmx_basic_fail_small_n++;
            }
        }
    }

    if (hmx_basic) {
        if (ggml_hexagon_precompute_hmx_mm_params(
                ctx, src0, src1, dst, wtype, ne00_padded, ne01_padded,
                ne02, ne11, ne12, ne11_padded, is_matmul_id, is_batched,
                vtcm_budget, kparams)) {
            ctx->n_hmx_vtcm_pass++;
            goto finalize;
        }
        ctx->n_hmx_vtcm_fail++;
    }

    // Fallback to HVX parameter computation
    ggml_hexagon_precompute_hvx_mm_params(
        ctx, src0, src1, dst, wtype,
        ne02, ne03, ne10, ne11, ne12, ne13,
        is_matmul_id, vtcm_budget, kparams);

finalize:
    kparams->div_ne12_ne1 = init_fastdiv_values((uint32_t)(ne12 * ne11));
    kparams->div_ne1      = init_fastdiv_values((uint32_t) ne11);
    kparams->div_r2       = init_fastdiv_values(ne02 > 0 ? (uint32_t)(ne12 / ne02) : 1);
    kparams->div_r3       = init_fastdiv_values(ne03 > 0 ? (uint32_t)(ne13 / ne03) : 1);
    kparams->div_ne11     = init_fastdiv_values((uint32_t) ne11);

    // Cache populated: key includes src0 tensor ptr to avoid mempool-reuse
    // collisions. The cached kparams are valid for the session lifetime
    // because weights are static (never modified after model load).
    ctx->mm_params_cache[cache_key] = *kparams;
}

// =================================================================================================
//  section-8: backend implementation
// =================================================================================================
ggml_backend_hexagon_context::ggml_backend_hexagon_context(int dev_id, ggml_backend_dev_t dev)
    : device(dev_id),
      backend(nullptr),
      socinfo(),
      n_threads(6),
      domain_id(CDSP_DOMAIN_ID),
      session_id(0),
      ggmlop_handle(0),
      rpc_mempool_capacity(0),
      rpc_mempool_len(0),
      rpc_mempool_usage(0),
      rpc_mempool_handle(0),
      rpc_mempool(nullptr),
      rpc_mempool_dsp_base(nullptr),
      weights_dirty(false),
      rpc_batch_call_count(0),
      cumulative_graph_us(0),
      last_graph_end_us(0),
      max_nodes_per_graph(0),
      min_nodes_per_graph(0),
      total_nodes_processed(0),
      min_graph_us(0),
      max_graph_us(0),
      max_graph_n_nodes(0),
      max_graph_n_ops(0),
      min_n_ops_per_call(0),
      max_n_ops_per_call(0),
      min_p10_us(0),
      max_p10_us(0),
      min_rpc_overhead_us(0),
      max_rpc_overhead_us(0),
      sum_rpc_overhead_us(0),
      cum_p5_us(0),
      cum_p6_us(0),
      cum_p8_us(0),
      cum_p9_us(0),
      cum_p10_us(0),
      cum_p10_rpc_setup_us(0),
      cum_p10_dsp_exec_us(0),
      cum_p10_civac_us(0),
      cum_p11_us(0),
      cum_p12_us(0),
      cum_unaccounted_us(0),
      rpc_overhead_min_us(0),
      rpc_overhead_max_us(0),
      rpc_overhead_sum_us(0),
      rpc_overhead_count(0),
      n_mul_mat_total_cum(0),
      n_hmx_used_cum(0),
      n_fused_qkv_cum(0),
      n_fused_ffn_cum(0),
      n_fused_mm_add_cum(0),
      n_hmx_basic_pass(0),
      n_hmx_basic_fail_ne01(0),
      n_hmx_basic_fail_ne00(0),
      n_hmx_basic_fail_wtype(0),
      n_hmx_basic_fail_batched(0),
      n_hmx_basic_fail_permuted(0),
      n_hmx_basic_fail_small_n(0),
      n_hmx_vtcm_pass(0),
      n_hmx_vtcm_fail(0),
      buffer_type{},
      has_vtcm(false),
      has_hvx(false),
      has_hmx(false) {
    snprintf(name, sizeof(name), "Hexagon-cDSP%d", dev_id);
    snprintf(desc, sizeof(desc), "Qualcomm NPU(CDSP%d)", dev_id);
    snprintf(buft_name, sizeof(buft_name), "hexagon-ion-buffer-%s", name);
    snprintf(repack_buft_name, sizeof(repack_buft_name), "hexagon-ion-buffer-%s-REPACK", name);
    lib[0] = '\0';

    buffer_type.iface.get_name         = ggml_backend_hexagon_buffer_type_name;
    buffer_type.iface.alloc_buffer     = ggml_backend_hexagon_buffer_type_alloc_buffer;
    buffer_type.iface.get_alignment    = ggml_backend_hexagon_buffer_type_get_alignment;
    buffer_type.iface.get_max_size     = ggml_backend_hexagon_buffer_type_get_max_size;
    buffer_type.iface.get_alloc_size   = ggml_backend_hexagon_buffer_type_get_alloc_size;
    buffer_type.iface.is_host          = ggml_backend_hexagon_buffer_is_host;
    buffer_type.device  = dev;
    buffer_type.context = this;

    // Repack buffer type: same mempool as buffer_type, but is_host=false
    repack_buffer_type.iface.get_name         = ggml_backend_hexagon_buffer_type_name;
    repack_buffer_type.iface.alloc_buffer     = ggml_backend_hexagon_buffer_type_alloc_buffer;
    repack_buffer_type.iface.get_alignment    = ggml_backend_hexagon_buffer_type_get_alignment;
    repack_buffer_type.iface.get_max_size     = ggml_backend_hexagon_buffer_type_get_max_size;
    repack_buffer_type.iface.get_alloc_size   = ggml_backend_hexagon_buffer_type_get_alloc_size;
    repack_buffer_type.iface.is_host          = ggml_backend_hexagon_repack_buffer_is_host;
    repack_buffer_type.device  = dev;
    repack_buffer_type.context = this;

    // Pre-size the cgraph cache to avoid rehashing during inference. With
    // max_load_factor=0.5 and reserve(1024) the bucket array can hold ~1024
    // entries before rehash; observed peak is ~227 (gemma-4/qwen3), so this keeps the
    // load low and lookup latency stable.
    cgraph_cache.max_load_factor(0.5f);
    cgraph_cache.reserve(1024);

    int result = ggmlhexagon_init_dsp(this);
    if (0 != result) {
        GGMLHEXAGON_LOG_ERROR("init hexagon dsp failure for device %d", dev_id);
    }
}

ggml_backend_hexagon_context::~ggml_backend_hexagon_context() {
    ggmlhexagon_deinit_cdsp(this);
    ggmlhexagon_print_running_timestamp(NULL);
}

static bool ggmlhexagon_supported_mul_mat(const struct ggml_tensor * dst,
                                          ggml_backend_hexagon_context * ctx) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const int64_t m                 = src0->ne[1];
    const int64_t k                 = src0->ne[0];
    const int64_t n                 = src1->ne[1];
    const uint32_t src0_rank        = ggml_n_dims(src0);
    const uint32_t src1_rank        = ggml_n_dims(src1);
    GGML_UNUSED(m);
    GGML_UNUSED(k);
    GGML_UNUSED(n);
    GGML_UNUSED(src0_rank);
    GGML_UNUSED(src1_rank);
    GGMLHEXAGON_LOG_DEBUG("MUL_MAT check: m=%lld, n=%lld, k=%lld, src0_rank=%d, src1_rank=%d", (long long)m, (long long)n, (long long)k, src0_rank, src1_rank);

    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        return false;
    }

    if (!ggmlhexagon_type_is_enabled(src0->type)) {
        // Q4_K/Q6_K are stored as Q4_0 (see set_tensor); inherit Q4_0's enablement
        if (!(src0->type == GGML_TYPE_Q4_K && ggmlhexagon_type_is_enabled(GGML_TYPE_Q4_0)) &&
            !(src0->type == GGML_TYPE_Q6_K && ggmlhexagon_type_is_enabled(GGML_TYPE_Q4_0))) {
            return false;
        }
    }

    switch (src0->type) {
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
        {
            // src0 (weights) must be repacked. In test-backend-ops tensors
            // are allocated in the main buffer, so this filters quantized MUL_MAT test cases
            if (src0->buffer && !ggml_backend_buffer_is_hexagon_repack(src0->buffer)) {
                return false;
            }

            if (src0->ne[0] % 32) {
                return false;
            }

            if (src1->ne[2] != 1 || src1->ne[3] != 1) {
                return false;  // no broadcasting (for now)
            }

            // Quantized HVX kernels assume src0 is laid out contiguously in
            // row-major (ne[0] is innermost). Non-contiguous views (e.g. k_v
            // slices) cause wrong tile offsets -> silent numeric corruption.
            if (!ggml_is_contiguous(src0)) {
                GGMLHEXAGON_LOG_DEBUG("supported_mul_mat FAIL: src0 not contiguous (nb=[%lld,%lld,%lld,%lld] ne=[%lld,%lld,%lld,%lld])",
                                        (long long)src0->nb[0], (long long)src0->nb[1], (long long)src0->nb[2], (long long)src0->nb[3],
                                        (long long)src0->ne[0], (long long)src0->ne[1], (long long)src0->ne[2], (long long)src0->ne[3]);
                return false;
            }

            break;
        }

        case GGML_TYPE_BF16:
            // BF16 weights are converted to F16 bytes in the repack buffer at
            // load time; only offload when the tensor lives there.
            if (!src0->buffer || !ggml_backend_buffer_is_hexagon_repack(src0->buffer)) {
                return false;
            }
            // fall through
        case GGML_TYPE_F16:
            if (src0->nb[1] < src0->nb[0]) {
                GGMLHEXAGON_LOG_WARN("permuted F16 src0 not supported\n");
                return false;
            }
            if (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                GGMLHEXAGON_LOG_WARN("src1 broadcasting not supported\n");
                return false;
            }
            break;

        case GGML_TYPE_F32:
            if (src1->type != GGML_TYPE_F32) {
                return false;
            }
            if (src0->nb[1] < src0->nb[0]) {
                GGMLHEXAGON_LOG_WARN("permuted F32 src0 not supported\n");
                return false;
            }
            if (src1->ne[2] < src0->ne[2] || src1->ne[3] < src0->ne[3]) {
                GGMLHEXAGON_LOG_WARN("src1 broadcasting not supported\n");
                return false;
            }
            break;

        default:
            return false;
    }

    // Precompute kernel params to get the actual VTCM size
    hex_op_desc tmp_op;
    memset(&tmp_op, 0, sizeof(tmp_op));
    tmp_op.opcode = dst->op;
    bool is_matmul_id = (dst->op == GGML_OP_MUL_MAT_ID);
    ggml_hexagon_precompute_mm_params(ctx, dst, tmp_op, is_matmul_id);
    const struct htp_mm_kernel_params * kparams =
        (const struct htp_mm_kernel_params *)tmp_op.kernel_params;

    const size_t vtcm_budget = (size_t)ctx->socinfo.vtcm_size_in_mb * 1024 * 1024;
    if ((size_t)kparams->vtcm_size > vtcm_budget) {
        GGMLHEXAGON_LOG_ALWAYS("MUL_MAT VTCM too small: needed=%d budget=%zu\n",
                               kparams->vtcm_size, vtcm_budget);
        return false;
    }

    return true;
}

// Decide whether a FLASH_ATTN_EXT node can be offloaded to the DSP.
// type/shape checks plus a precompute pass that verifies the selected kernel
// (HMX or HVX) fits the per-domain VTCM budget.
static bool ggmlhexagon_supported_flash_attn(
    const ggml_backend_hexagon_context * ctx, const struct ggml_tensor * dst) {
    const struct ggml_tensor * q     = dst->src[0];
    const struct ggml_tensor * k     = dst->src[1];
    const struct ggml_tensor * v     = dst->src[2];
    const struct ggml_tensor * mask  = dst->src[3];
    const struct ggml_tensor * sinks = dst->src[4];

    if (!q || !k || !v) {
        return false;
    }
    if ((q->type != GGML_TYPE_F16 && q->type != GGML_TYPE_F32) ||
        k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) {
        return false;
    }
    if (mask && mask->type != GGML_TYPE_F16) {
        return false;
    }
    if (sinks && sinks->type != GGML_TYPE_F32) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16) {
        return false;
    }
    if (dst->ne[3] != 1) {
        return false;
    }

    struct htp_fa_kernel_params kparams;
    if (!ggml_hexagon_compute_fa_params(ctx, dst, &kparams)) {
        return false;
    }
    const size_t vtcm_budget = (size_t)ctx->socinfo.vtcm_size_in_mb * 1024 * 1024;
    if ((size_t)kparams.vtcm_size > vtcm_budget) {
        return false;
    }
    return true;
}

// Function pointer table for ggmlhexagon_can_handle_op_through_cdsp.
// Replaces the large switch statement with individual validator functions
// indexed by GGML_OP. Each validator receives the Hexagon context and the
// op tensor; returns true if the DSP can handle the op.
typedef bool (*hexagon_op_validator_t)(ggml_backend_hexagon_context * ctx, const ggml_tensor * op);

// Binary element-wise ops: ADD, SUB, MUL, DIV
static bool hexagon_validate_binary_op(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    GGML_UNUSED(ctx);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * dst  = op;
    if (src0->type == GGML_TYPE_F32) {
        if (!src1 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32)
            return false;
    } else if (src0->type == GGML_TYPE_F16) {
        if (!src1 || src1->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F16)
            return false;
    } else {
        return false;
    }
    if (!ggml_are_same_shape(src0, dst)) return false;
    if (!ggml_can_repeat(src1, src0) || ggml_is_permuted(src1))
        return false;
    return true;
}

static bool hexagon_validate_mul_mat(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    return ggmlhexagon_supported_mul_mat(op, ctx);
}

static bool hexagon_validate_rms_norm(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    GGML_UNUSED(ctx);
    const ggml_tensor * src0 = op->src[0];
    if (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32)
        return false;
    if (!ggml_is_contiguous(src0))
        return false;
    return true;
}

// NORM, L2_NORM: dispatched to op_unary (F32, same shape, contiguous dst)
static bool hexagon_validate_norm_op(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    GGML_UNUSED(ctx);
    const ggml_tensor * src0 = op->src[0];
    if (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32)
        return false;
    if (!ggml_are_same_shape(src0, op))
        return false;
    if (!ggml_is_contiguous(op))
        return false;
    return true;
}

// SQR, SQRT: element-wise unary, same as norm_op
static bool hexagon_validate_sqr_sqrt(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    return hexagon_validate_norm_op(ctx, op);
}

static bool hexagon_validate_rope(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    GGML_UNUSED(ctx);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32)
        return false;
    if (!src1 || src1->type != GGML_TYPE_I32)
        return false;
    const int32_t mode = op->op_params[2];
    if (mode == 24) return false;
    return true;
}

static bool hexagon_validate_soft_max(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    GGML_UNUSED(ctx);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32)
        return false;
    if (src1 != nullptr && src1->type != GGML_TYPE_F16 && src1->type != GGML_TYPE_F32)
        return false;
    if (op->src[2] != nullptr)
        return false;
    return true;
}

static bool hexagon_validate_unary(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    GGML_UNUSED(ctx);
    const ggml_tensor * src0 = op->src[0];
    const int unary_op = (int)op->op_params[0];
    switch (unary_op) {
        case GGML_UNARY_OP_SILU:
        case GGML_UNARY_OP_GELU:
        case GGML_UNARY_OP_GELU_QUICK:
        case GGML_UNARY_OP_NEG:
        case GGML_UNARY_OP_EXP:
        case GGML_UNARY_OP_SIGMOID:
        case GGML_UNARY_OP_SOFTPLUS:
        case GGML_UNARY_OP_TANH:
            if (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32)
                return false;
            if (ggml_is_permuted(src0))
                return false;
            if (!ggml_are_same_shape(src0, op))
                return false;
            if (!ggml_is_contiguous(op))
                return false;
            return true;
        default:
            return false;
    }
}

static bool hexagon_validate_glu(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    GGML_UNUSED(ctx);
    const ggml_tensor * src0 = op->src[0];
    if (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32)
        return false;
    if (!ggml_is_contiguous_1(src0) || !ggml_is_contiguous(op))
        return false;
    const int glu_op = (int)op->op_params[0];
    switch (glu_op) {
        case GGML_GLU_OP_SWIGLU:
        case GGML_GLU_OP_SWIGLU_OAI:
        case GGML_GLU_OP_GEGLU:
            return true;
        default:
            return false;
    }
}

static bool hexagon_validate_scale(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    GGML_UNUSED(ctx);
    const ggml_tensor * src0 = op->src[0];
    if (src0->type != op->type) return false;
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16)
        return false;
    return true;
}

static bool hexagon_validate_cpy(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    GGML_UNUSED(ctx);
    const ggml_tensor * src0 = op->src[0];
    if (src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32)
        return false;
    if (op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_F32)
        return false;
    return true;
}

static bool hexagon_validate_get_rows(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    GGML_UNUSED(ctx);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (!src1 || src1->type != GGML_TYPE_I32)
        return false;
    if (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32)
        return false;
    return true;
}

static bool hexagon_validate_set_rows(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    GGML_UNUSED(ctx);
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (!src1 || (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64))
        return false;
    if (src0->type != GGML_TYPE_F32)
        return false;
    if (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16)
        return false;
    return true;
}

static bool hexagon_validate_sum_rows(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    GGML_UNUSED(ctx);
    if (op->src[0]->type != GGML_TYPE_F32)
        return false;
    return true;
}

static bool hexagon_validate_cont(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    GGML_UNUSED(ctx);
    const ggml_tensor * src0 = op->src[0];
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16)
        return false;
    if (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16)
        return false;
    return true;
}

static bool hexagon_validate_concat(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    GGML_UNUSED(ctx);
    const ggml_tensor * src0 = op->src[0];
    if (src0->type != op->type) return false;
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 &&
        src0->type != GGML_TYPE_I32 && src0->type != GGML_TYPE_I16)
        return false;
    return true;
}

static bool hexagon_validate_repeat(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    GGML_UNUSED(ctx);
    const ggml_tensor * src0 = op->src[0];
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 &&
        src0->type != GGML_TYPE_I32 && src0->type != GGML_TYPE_I16)
        return false;
    return true;
}

static bool hexagon_validate_diag_mask_inf(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    GGML_UNUSED(ctx);
    const ggml_tensor * src0 = op->src[0];
    if (src0->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32)
        return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(op))
        return false;
    return true;
}

static bool hexagon_validate_cumsum(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    GGML_UNUSED(ctx);
    if (op->src[0]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32)
        return false;
    return true;
}

static bool hexagon_validate_diag(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    GGML_UNUSED(ctx);
    if (op->src[0]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32)
        return false;
    return true;
}

static bool hexagon_validate_argsort(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    GGML_UNUSED(ctx);
    if (op->src[0]->type != GGML_TYPE_F32)
        return false;
    return true;
}

static bool hexagon_validate_pad(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    GGML_UNUSED(ctx);
    if (op->src[0]->type != GGML_TYPE_F32)
        return false;
    return true;
}

static bool hexagon_validate_im2col(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    GGML_UNUSED(ctx);
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * dst  = op;
    const bool is_2D = ((const int32_t *) op->op_params)[6] == 1;
    if (!is_2D) {
        return false;
    }
    // F32 image -> F16/F32 columns only
    if (src1->type != GGML_TYPE_F32 || (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_F32)) {
        return false;
    }
    if (!ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    // padded im2col stays on CPU; DSP path only covers patch-embed shape
    const int32_t p0 = ((const int32_t *) op->op_params)[2];
    const int32_t p1 = ((const int32_t *) op->op_params)[3];
    if (p0 != 0 || p1 != 0) {
        return false;
    }
    return true;
}

static bool hexagon_validate_tri(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    GGML_UNUSED(ctx);
    if (op->src[0]->type != GGML_TYPE_F32)
        return false;
    return true;
}

static bool hexagon_validate_fill(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    GGML_UNUSED(ctx);
    if (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16)
        return false;
    return true;
}

static bool hexagon_validate_flash_attn(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    return ggmlhexagon_supported_flash_attn(ctx, op);
}

static bool hexagon_validate_gated_delta_net(ggml_backend_hexagon_context * ctx, const ggml_tensor * op) {
    GGML_UNUSED(ctx);
    const struct ggml_tensor * q     = op->src[0];
    const struct ggml_tensor * k     = op->src[1];
    const struct ggml_tensor * v     = op->src[2];
    const struct ggml_tensor * g     = op->src[3];
    const struct ggml_tensor * beta  = op->src[4];
    const struct ggml_tensor * state = op->src[5];
    const struct ggml_tensor * dst   = op;

    if (!q || !k || !v || !g || !beta || !state) {
        return false;
    }

    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
        g->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        return false;
    }

    if (!ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(k) || !ggml_is_contiguous_rows(v) ||
        !ggml_is_contiguous(g) || !ggml_is_contiguous(beta) || !ggml_is_contiguous(state) ||
        !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t S_v      = v->ne[0];
    const int64_t H        = v->ne[1];
    const int64_t n_tokens = v->ne[2];
    const int64_t n_seqs   = v->ne[3];
    const int64_t K        = ggml_get_op_params_i32(op, 0);

    if (S_v <= 0 || S_v > 128 || H <= 0 || n_tokens <= 0 || n_seqs <= 0) {
        return false;
    }
    if (q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] <= 0 || k->ne[1] <= 0 ||
        q->ne[2] != n_tokens || k->ne[2] != n_tokens || q->ne[3] <= 0 || k->ne[3] <= 0 ||
        (n_seqs % q->ne[3]) != 0 || (n_seqs % k->ne[3]) != 0) {
        return false;
    }
    if ((g->ne[0] != 1 && g->ne[0] != S_v) || beta->ne[0] != 1) {
        return false;
    }
    if (ggml_nelements(state) != S_v * S_v * H * n_seqs) {
        return false;
    }
    if (dst->ne[0] != S_v * H || dst->ne[1] != n_tokens * n_seqs + S_v * n_seqs * K) {
        return false;
    }

    return true;
}

// Static lookup table: one validator per GGML_OP, indexed by op_tensor->op.
// NULL entries mean "not supported on DSP" (returns false).
// Initialized once at first call via init_op_validators().
static hexagon_op_validator_t s_op_validators[GGML_OP_COUNT];

static void init_op_validators(void) {
    static bool s_initialized = false;
    if (s_initialized) return;
    s_initialized = true;

    s_op_validators[GGML_OP_ADD]            = hexagon_validate_binary_op;
    s_op_validators[GGML_OP_SUB]            = hexagon_validate_binary_op;
    s_op_validators[GGML_OP_MUL]            = hexagon_validate_binary_op;
    s_op_validators[GGML_OP_DIV]            = hexagon_validate_binary_op;
    s_op_validators[GGML_OP_MUL_MAT]        = hexagon_validate_mul_mat;
    s_op_validators[GGML_OP_RMS_NORM]       = hexagon_validate_rms_norm;
    s_op_validators[GGML_OP_NORM]           = hexagon_validate_norm_op;
    s_op_validators[GGML_OP_L2_NORM]        = hexagon_validate_norm_op;
    s_op_validators[GGML_OP_SQR]            = hexagon_validate_sqr_sqrt;
    s_op_validators[GGML_OP_SQRT]           = hexagon_validate_sqr_sqrt;
    s_op_validators[GGML_OP_ROPE]           = hexagon_validate_rope;
    s_op_validators[GGML_OP_SOFT_MAX]       = hexagon_validate_soft_max;
    s_op_validators[GGML_OP_UNARY]          = hexagon_validate_unary;
    s_op_validators[GGML_OP_GLU]            = hexagon_validate_glu;
    s_op_validators[GGML_OP_SCALE]          = hexagon_validate_scale;
    s_op_validators[GGML_OP_CPY]            = hexagon_validate_cpy;
    s_op_validators[GGML_OP_GET_ROWS]       = hexagon_validate_get_rows;
    s_op_validators[GGML_OP_SET_ROWS]       = hexagon_validate_set_rows;
    s_op_validators[GGML_OP_SUM_ROWS]       = hexagon_validate_sum_rows;
    s_op_validators[GGML_OP_CONT]           = hexagon_validate_cont;
    s_op_validators[GGML_OP_CONCAT]         = hexagon_validate_concat;
    s_op_validators[GGML_OP_REPEAT]         = hexagon_validate_repeat;
    s_op_validators[GGML_OP_DIAG_MASK_INF]  = hexagon_validate_diag_mask_inf;
    s_op_validators[GGML_OP_CUMSUM]         = hexagon_validate_cumsum;
    s_op_validators[GGML_OP_DIAG]           = hexagon_validate_diag;
    s_op_validators[GGML_OP_ARGSORT]        = hexagon_validate_argsort;
    s_op_validators[GGML_OP_PAD]            = hexagon_validate_pad;
    s_op_validators[GGML_OP_IM2COL]         = hexagon_validate_im2col;
    s_op_validators[GGML_OP_GATED_DELTA_NET]= hexagon_validate_gated_delta_net;
    s_op_validators[GGML_OP_TRI]            = hexagon_validate_tri;
    s_op_validators[GGML_OP_FILL]           = hexagon_validate_fill;
    s_op_validators[GGML_OP_FLASH_ATTN_EXT] = hexagon_validate_flash_attn;
}

static bool ggmlhexagon_can_handle_op_through_cdsp(ggml_backend_dev_t dev, const struct ggml_tensor * op_tensor) {
    if (!ggmlhexagon_is_metadata_op(op_tensor->op)) {
        if (!ggmlhexagon_op_buffers_belong_to_dev(dev, op_tensor)) {
            return false;
        }
    }

    if (ggmlhexagon_is_metadata_op(op_tensor->op)) {
        return true;
    }

    init_op_validators();

    ggml_backend_hexagon_context * ctx = (ggml_backend_hexagon_context *)dev->context;
    hexagon_op_validator_t validator = s_op_validators[op_tensor->op];
    if (validator) {
        return validator(ctx, op_tensor);
    }
    return false;
}

struct ggml_backend_hexagon_buffer_context {
    ~ggml_backend_hexagon_buffer_context() {
        if (buffer) {
            if (is_ion_buffer) {
                if (backend_ctx && backend_ctx->rpc_mempool) {
                    {
                        // Mark the mempool region as free so it can be reused.
                        const char * buf_ptr = (const char *)buffer;
                        const char * pool_base = (const char *)backend_ctx->rpc_mempool;
                        if (buf_ptr >= pool_base && buf_ptr < pool_base + (ptrdiff_t)backend_ctx->rpc_mempool_len) {
                            size_t buf_offset = (size_t)(buf_ptr - pool_base);
                            for (auto & r : backend_ctx->ion_regions) {
                                if (r.in_use && r.offset == buf_offset) {
                                    r.in_use = false;
                                    GGMLHEXAGON_LOG_WARN("[FREE] device=%d region offset=%zu size=%zu", backend_ctx->device, r.offset, r.size);
                                    break;
                                }
                            }
                        }
                    }
                }
            } else {
                ggml_aligned_free(buffer, 0);
            }
        }
    }

    void * buffer       = nullptr;
    size_t buffer_size  = 0;
    bool   is_ion_buffer= false;

    struct ggml_backend_hexagon_context * backend_ctx = nullptr;
};

static void ggml_backend_hexagon_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_hexagon_buffer_context * ctx = (ggml_backend_hexagon_buffer_context *)buffer->context;
    delete ctx;
}

static void * ggml_backend_hexagon_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_hexagon_buffer_context * ctx = (ggml_backend_hexagon_buffer_context *)buffer->context;
    return ctx->buffer;
}

static enum ggml_status ggml_backend_hexagon_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_hexagon_buffer_context * ctx = (ggml_backend_hexagon_buffer_context *)buffer->context;
    GGML_UNUSED(tensor);
    GGML_UNUSED(ctx);
    return GGML_STATUS_SUCCESS;
}

static void repack_q4_0_q4x4x2(ggml_tensor * t, const void * data, size_t size, void * dst_buf = nullptr) {
    const int QK_Q4_0x4x2 = 256;

    int64_t nrows = ggml_nrows(t);

    size_t row_size    = ggml_row_size(t->type, t->ne[0]);
    size_t row_size_pd = ggml_row_size(t->type, hex_round_up((uint32_t)t->ne[0], (uint32_t)QK_Q4_0x4x2));
    size_t row_size_rp = row_size_pd;

    const size_t total_tensor_size = (size_t)nrows * row_size;
    const size_t n_bytes_to_copy = size < total_tensor_size ? size : total_tensor_size;

    const int64_t n_full_rows = n_bytes_to_copy / row_size;
    const size_t  n_rem_bytes = n_bytes_to_copy % row_size;

    uint8_t * out_base = dst_buf ? (uint8_t *)dst_buf : (uint8_t *)t->data;

    void * buf_pd = ggml_aligned_malloc(row_size_pd);
    GGML_ASSERT(buf_pd != NULL);

    void * buf_rp = ggml_aligned_malloc(row_size_rp);
    GGML_ASSERT(buf_rp != NULL);

    // src stride: t->nb[1] handles padded (non-contiguous) tensors (e.g. KV cache views);
    // for contiguous tensors nb[1] == row_size. dst stays contiguous (x4x2 repacked output).
    const size_t src_stride = t->nb[1];

    for (int64_t i = 0; i < n_full_rows; i++) {
        const uint8_t * src = (const uint8_t *) data + (i * src_stride);
        uint8_t *       dst = out_base + (i * row_size);

        memcpy(buf_pd, src, row_size);

        const uint8_t * x = (const uint8_t *) buf_pd;
        uint8_t * y = (uint8_t *) buf_rp;

        const int qk = QK_Q4_0x4x2;
        const int nb = t->ne[0] / qk;

        const int dblk_size = 8 * 2;
        const int qblk_size = qk / 2;
        const int qrow_size = t->ne[0] / 2;
        const int q4_blk_sz = QK4_0 / 2 + 2;

        uint8_t * y_q = y + 0;
        uint8_t * y_d = y + qrow_size;

        for (int ib = 0; ib < nb; ib++) {
            uint8_t qs[QK_Q4_0x4x2];

            for (int j = 0; j < 8; j++) {
                const uint8_t * b = x + (ib * 8 + j) * q4_blk_sz + 2;
                for (int k = 0; k < QK4_0 / 2; k++) {
                    qs[j * QK4_0 + k + 0]         = (b[k] & 0x0F);
                    qs[j * QK4_0 + k + QK4_0 / 2] = (b[k] >> 4);
                }
            }

            uint8_t * q = y_q + (ib * qblk_size);
            for (int j = 0; j < qk / 2; j++) {
                q[j] = (qs[j + 128] << 4) | qs[j];
            }

            uint16_t * d = (uint16_t *) (y_d + ib * dblk_size);
            for (int j = 0; j < 8; j++) {
                const uint16_t * scale = (const uint16_t *)(x + (ib * 8 + j) * q4_blk_sz);
                d[j] = *scale;
            }
        }

        memcpy(dst, buf_rp, row_size);
    }

    if (n_rem_bytes > 0) {
        const uint8_t * src = (const uint8_t *) data + (n_full_rows * src_stride);
        uint8_t *       dst = out_base + (n_full_rows * row_size);

        memset(buf_pd, 0, row_size_pd);
        memcpy(buf_pd, src, n_rem_bytes);

        const uint8_t * x = (const uint8_t *) buf_pd;
        uint8_t * y = (uint8_t *) buf_rp;

        const int qk = QK_Q4_0x4x2;
        const int nb = (t->ne[0] + qk - 1) / qk;

        const int dblk_size = 8 * 2;
        const int qblk_size = qk / 2;
        const int qrow_size = t->ne[0] / 2;
        const int q4_blk_sz = QK4_0 / 2 + 2;

        uint8_t * y_q = y + 0;
        uint8_t * y_d = y + qrow_size;

        for (int ib = 0; ib < nb; ib++) {
            uint8_t qs[QK_Q4_0x4x2] = {0};

            for (int j = 0; j < 8 && (size_t)(ib * 8 + j) * q4_blk_sz < row_size_pd; j++) {
                const uint8_t * b = x + (ib * 8 + j) * q4_blk_sz + 2;
                for (int k = 0; k < QK4_0 / 2; k++) {
                    qs[j * QK4_0 + k + 0]         = (b[k] & 0x0F);
                    qs[j * QK4_0 + k + QK4_0 / 2] = (b[k] >> 4);
                }
            }

            uint8_t * q = y_q + (ib * qblk_size);
            bool partial = (ib == nb - 1);
            for (int j = 0; j < qk / 2; j++) {
                if (partial) {
                    q[j] = (qs[j * 2 + 1] << 4) | qs[j * 2 + 0];
                } else {
                    q[j] = (qs[j + 128] << 4) | qs[j];
                }
            }

            uint16_t * d = (uint16_t *) (y_d + ib * dblk_size);
            for (int j = 0; j < 8 && (size_t)(ib * 8 + j) * q4_blk_sz < row_size_pd; j++) {
                const uint16_t * scale = (const uint16_t *)(x + (ib * 8 + j) * q4_blk_sz);
                d[j] = *scale;
            }
        }

        memcpy(dst, buf_rp, n_rem_bytes);
    }

    ggml_aligned_free(buf_pd, row_size_pd);
    ggml_aligned_free(buf_rp, row_size_rp);
}

// Inverse of repack_q4_0_q4x4x2: convert x4x2 layout back to Q4_0.
// Used by get_tensor so CPU backends receive canonical Q4_0 bytes.
static void repack_q4x4x2_q4_0(const ggml_tensor * t, void * data, size_t size) {
    const int QK_Q4_0x4x2 = 256;

    int64_t nrows = ggml_nrows(t);
    size_t  row_size = ggml_row_size(t->type, t->ne[0]);
    size_t  total = (size_t)nrows * row_size;
    int64_t n_full_rows = (size >= total) ? nrows : (int64_t)(size / row_size);

    const int qk         = QK_Q4_0x4x2;
    const int nb         = t->ne[0] / qk;
    const int qblk_size  = qk / 2;          // 128
    const int dblk_size  = 8 * 2;           // 16
    const int qrow_size  = t->ne[0] / 2;
    const int q4_blk_sz  = QK4_0 / 2 + 2;   // 18

    for (int64_t i = 0; i < n_full_rows; i++) {
        const uint8_t * src = (const uint8_t *) t->data + (i * row_size);
        uint8_t *       dst = (uint8_t *) data + (i * row_size);

        const uint8_t * x_q = src;
        const uint8_t * x_d = src + qrow_size;

        for (int ib = 0; ib < nb; ib++) {
            const uint8_t * q = x_q + (ib * qblk_size);

            uint8_t qs[QK_Q4_0x4x2];
            for (int j = 0; j < qk / 2; j++) {
                qs[j]       = q[j] & 0x0F;
                qs[j + 128] = q[j] >> 4;
            }

            const uint16_t * d_src = (const uint16_t *)(x_d + ib * dblk_size);

            for (int j = 0; j < 8; j++) {
                uint8_t * block = dst + (ib * 8 + j) * q4_blk_sz;
                *(uint16_t *)block = d_src[j];
                uint8_t * b = block + 2;
                for (int k = 0; k < QK4_0 / 2; k++) {
                    b[k] = (qs[j * QK4_0 + k + QK4_0 / 2] << 4) | qs[j * QK4_0 + k];
                }
            }
        }
    }
}

// ---- Tiled repack for HVX-quant MUL_MAT ----
// HVX-quant kernels (hvx_mm_2d_repacked_*_flat etc.) expect tile-based weight
// layout: each 32x32 tile is tile_size bytes, organized as (ct, kt) major with
// (cp, row) minor inside each tile. Standard GGML row-major layout must be
// converted before passing to DSP.

static void unpack_q4_0_quants(uint8_t * qs, const block_q4_0 * x) {
    for (unsigned int i = 0; i < QK4_0 / 2; ++i) {
        const int x0 = (x->qs[i] & 0x0F);
        const int x1 = (x->qs[i] >> 4);
        qs[i + 0]            = x0;
        qs[i + QK4_0 / 2]   = x1;
    }
}

static void unpack_q4_1_quants(uint8_t * qs, const block_q4_1 * x) {
    for (unsigned int i = 0; i < QK4_1 / 2; ++i) {
        const int x0 = (x->qs[i] & 0x0F);
        const int x1 = (x->qs[i] >> 4);
        qs[i + 0]            = x0;
        qs[i + QK4_1 / 2]   = x1;
    }
}

static void unpack_mxfp4_quants(uint8_t * qs, const block_mxfp4 * x) {
    for (unsigned int i = 0; i < QK_MXFP4 / 2; ++i) {
        const int x0 = (x->qs[i] & 0x0F);
        const int x1 = (x->qs[i] >> 4);
        qs[i + 0]            = x0;
        qs[i + QK_MXFP4 / 2] = x1;
    }
}

static void pack_mxfp4_quants(block_mxfp4 * x, const uint8_t * qs) {
    for (unsigned int i = 0; i < QK_MXFP4 / 2; ++i) {
        x->qs[i] = qs[i] | (qs[i + QK_MXFP4 / 2] << 4);
    }
}

static void pack_q4_0_quants(block_q4_0 * x, const uint8_t * qs) {
    for (unsigned int i = 0; i < QK4_0 / 2; ++i) {
        x->qs[i] = qs[i] | (qs[i + QK4_0 / 2] << 4);
    }
}

static void pack_q4_1_quants(block_q4_1 * x, const uint8_t * qs) {
    for (unsigned int i = 0; i < QK4_1 / 2; ++i) {
        x->qs[i] = qs[i] | (qs[i + QK4_1 / 2] << 4);
    }
}

static size_t ggml_hexagon_repacked_size(enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    const uint32_t tile_size = htp_mm_get_weight_tile_size((int)ggml_hexagon_weight_dsp_type(type));
    if (tile_size == 0) return 0;
    const uint32_t ne0_p = hex_round_up((uint32_t)ne0, 32);
    const uint32_t ne1_p = hex_round_up((uint32_t)ne1, 32);
    return (size_t)(ne0_p / 32) * (ne1_p / 32) * tile_size * ne2 * ne3;
}

static void repack_q4_0_tiled_to_buf(const ggml_tensor * t, const void * data, void * dst_buf) {
    const block_q4_0 * src_matrix = (const block_q4_0 *) data;
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    const int n_col_tiles = hex_round_up((uint32_t)ne1, 32) / 32;
    const int n_k_tiles   = hex_round_up((uint32_t)ne0, 32) / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_0;
    const size_t matrix_size = (size_t)n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            const block_q4_0 * src_expert = src_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            uint8_t * matrix_dst = (uint8_t *) dst_buf + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;

                    uint8_t tile_quants[32][32];
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            unpack_q4_0_quants(tile_quants[row], &src_expert[r * (ne0 / 32) + kt]);
                        } else {
                            memset(tile_quants[row], 8, 32);
                        }
                    }

                    for (int cp = 0; cp < 16; cp++) {
                        for (int row = 0; row < 32; row++) {
                            tile_dst[cp * 32 + row] = (tile_quants[row][2 * cp + 1] << 4) | tile_quants[row][2 * cp];
                        }
                    }

                    ggml_half * scale_dst = (ggml_half *)(tile_dst + 512);
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        scale_dst[row] = (r < ne1 && kt < ne0 / 32) ? src_expert[r * (ne0 / 32) + kt].d : 0;
                    }
                }
            }
        }
    }
}

static void repack_q4_1_tiled_to_buf(const ggml_tensor * t, const void * data, void * dst_buf) {
    const block_q4_1 * src_matrix = (const block_q4_1 *) data;
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    const int n_col_tiles = hex_round_up((uint32_t)ne1, 32) / 32;
    const int n_k_tiles   = hex_round_up((uint32_t)ne0, 32) / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_1;
    const size_t matrix_size = (size_t)n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            const block_q4_1 * src_expert = src_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            uint8_t * matrix_dst = (uint8_t *) dst_buf + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;

                    uint8_t tile_quants[32][32];
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            unpack_q4_1_quants(tile_quants[row], &src_expert[r * (ne0 / 32) + kt]);
                        } else {
                            memset(tile_quants[row], 0, 32);
                        }
                    }

                    for (int cp = 0; cp < 16; cp++) {
                        for (int row = 0; row < 32; row++) {
                            tile_dst[cp * 32 + row] = (tile_quants[row][2 * cp + 1] << 4) | tile_quants[row][2 * cp];
                        }
                    }

                    ggml_half * scale_dst = (ggml_half *)(tile_dst + 512);
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            scale_dst[2 * row + 0] = src_expert[r * (ne0 / 32) + kt].d;
                            scale_dst[2 * row + 1] = src_expert[r * (ne0 / 32) + kt].m;
                        } else {
                            scale_dst[2 * row + 0] = 0;
                            scale_dst[2 * row + 1] = 0;
                        }
                    }
                }
            }
        }
    }
}

static void repack_q8_0_tiled_to_buf(const ggml_tensor * t, const void * data, void * dst_buf) {
    const block_q8_0 * src_matrix = (const block_q8_0 *) data;
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    const int n_col_tiles = hex_round_up((uint32_t)ne1, 32) / 32;
    const int n_k_tiles   = hex_round_up((uint32_t)ne0, 32) / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q8_0;
    const size_t matrix_size = (size_t)n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            const block_q8_0 * src_expert = src_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            uint8_t * matrix_dst = (uint8_t *) dst_buf + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;

                    for (int cp = 0; cp < 16; cp++) {
                        int col0 = cp * 2;
                        int col1 = col0 + 1;
                        for (int row = 0; row < 32; row++) {
                            int64_t r = ct * 32 + row;
                            const block_q8_0 * b = (r < ne1 && kt < ne0 / 32) ? &src_expert[r * (ne0 / 32) + kt] : NULL;
                            tile_dst[cp * 64 + 2 * row + 0] = b ? b->qs[col0] : 0;
                            tile_dst[cp * 64 + 2 * row + 1] = b ? b->qs[col1] : 0;
                        }
                    }

                    ggml_half * scale_dst = (ggml_half *)(tile_dst + 1024);
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        scale_dst[row] = (r < ne1 && kt < ne0 / 32) ? src_expert[r * (ne0 / 32) + kt].d : 0;
                    }
                }
            }
        }
    }
}

// Q4_K weights are converted to Q8_0 (dequant Q4_K -> f32 -> requant Q8_0,
// near-lossless) and stored in the Q8_0 tiled layout, so the DSP reuses the
// Q8_0 matmul kernels. Runs once at model load. Per-strip scratch keeps peak
// memory at ~32 rows instead of a full canonical Q8_0 copy.
static void repack_q4k_as_q8_0_tiled_to_buf(const ggml_tensor * t, const void * data, void * dst_buf) {
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    const int n_col_tiles = hex_round_up((uint32_t)ne1, 32) / 32;
    const int n_k_tiles   = hex_round_up((uint32_t)ne0, 32) / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q8_0;
    const size_t matrix_size = (size_t)n_col_tiles * n_k_tiles * tile_size;
    const int64_t nb_q8   = ne0 / QK8_0;   // q8_0 blocks per row
    const int64_t nb_q4k  = ne0 / QK_K;    // q4_K blocks per row

    std::vector<float>      row_f32(ne0);
    std::vector<block_q8_0> strip_q8(32 * nb_q8);  // canonical q8_0 for one 32-row strip

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            const block_q4_K * src_expert = (const block_q4_K *) data + (i3 * ne2 + i2) * (ne1 * nb_q4k);
            uint8_t * matrix_dst = (uint8_t *) dst_buf + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int row = 0; row < 32; row++) {
                    const int64_t r = ct * 32 + row;
                    if (r < ne1) {
                        dequantize_row_q4_K(src_expert + r * nb_q4k, row_f32.data(), ne0);
                        quantize_row_q8_0_ref(row_f32.data(), strip_q8.data() + row * nb_q8, ne0);
                    } else {
                        memset(strip_q8.data() + row * nb_q8, 0, nb_q8 * sizeof(block_q8_0));
                    }
                }

                for (int kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;

                    for (int cp = 0; cp < 16; cp++) {
                        const int col0 = cp * 2;
                        const int col1 = col0 + 1;
                        for (int row = 0; row < 32; row++) {
                            const block_q8_0 * b = (kt < nb_q8) ? &strip_q8[row * nb_q8 + kt] : NULL;
                            tile_dst[cp * 64 + 2 * row + 0] = b ? b->qs[col0] : 0;
                            tile_dst[cp * 64 + 2 * row + 1] = b ? b->qs[col1] : 0;
                        }
                    }

                    ggml_half * scale_dst = (ggml_half *)(tile_dst + 1024);
                    for (int row = 0; row < 32; row++) {
                        scale_dst[row] = (kt < nb_q8) ? strip_q8[row * nb_q8 + kt].d : 0;
                    }
                }
            }
        }
    }
}

static void repack_q4k_as_q4_0_tiled_to_buf(const ggml_tensor * t, const void * data, void * dst_buf) {
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    const int n_col_tiles = hex_round_up((uint32_t)ne1, 32) / 32;
    const int n_k_tiles   = hex_round_up((uint32_t)ne0, 32) / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_0;
    const size_t matrix_size = (size_t)n_col_tiles * n_k_tiles * tile_size;
    const int64_t nb_q4   = ne0 / QK4_0;   // q4_0 blocks per row
    const int64_t nb_q4k  = ne0 / QK_K;    // q4_K blocks per row

    std::vector<float>      row_f32(ne0);
    std::vector<block_q4_0> strip_q4(32 * nb_q4);  // canonical q4_0 for one 32-row strip

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            const block_q4_K * src_expert = (const block_q4_K *) data + (i3 * ne2 + i2) * (ne1 * nb_q4k);
            uint8_t * matrix_dst = (uint8_t *) dst_buf + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int row = 0; row < 32; row++) {
                    const int64_t r = ct * 32 + row;
                    if (r < ne1) {
                        dequantize_row_q4_K(src_expert + r * nb_q4k, row_f32.data(), ne0);
                        quantize_row_q4_0_ref(row_f32.data(), strip_q4.data() + row * nb_q4, ne0);
                    } else {
                        memset(strip_q4.data() + row * nb_q4, 0, nb_q4 * sizeof(block_q4_0));
                    }
                }

                for (int kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;

                    uint8_t tile_quants[32][32];
                    for (int row = 0; row < 32; row++) {
                        if (kt < nb_q4) {
                            unpack_q4_0_quants(tile_quants[row], &strip_q4[row * nb_q4 + kt]);
                        } else {
                            memset(tile_quants[row], 8, 32);
                        }
                    }

                    for (int cp = 0; cp < 16; cp++) {
                        for (int row = 0; row < 32; row++) {
                            tile_dst[cp * 32 + row] = (tile_quants[row][2 * cp + 1] << 4) | tile_quants[row][2 * cp];
                        }
                    }

                    ggml_half * scale_dst = (ggml_half *)(tile_dst + 512);
                    for (int row = 0; row < 32; row++) {
                        scale_dst[row] = (kt < nb_q4) ? strip_q4[row * nb_q4 + kt].d : 0;
                    }
                }
            }
        }
    }
}

// Q6_K weights are converted to Q4_0 (dequant Q6_K -> f32 -> requant Q4_0)
// and stored in the Q4_0 tiled layout, so the DSP reuses the Q4_0 matmul
// kernels. Same approach as Q4_K conversion above.
static void repack_q6k_as_q4_0_tiled_to_buf(const ggml_tensor * t, const void * data, void * dst_buf) {
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    const int n_col_tiles = hex_round_up((uint32_t)ne1, 32) / 32;
    const int n_k_tiles   = hex_round_up((uint32_t)ne0, 32) / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_0;
    const size_t matrix_size = (size_t)n_col_tiles * n_k_tiles * tile_size;
    const int64_t nb_q4   = ne0 / QK4_0;   // q4_0 blocks per row
    const int64_t nb_q6k  = ne0 / QK_K;    // q6_K blocks per row

    std::vector<float>      row_f32(ne0);
    std::vector<block_q4_0> strip_q4(32 * nb_q4);

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            const block_q6_K * src_expert = (const block_q6_K *) data + (i3 * ne2 + i2) * (ne1 * nb_q6k);
            uint8_t * matrix_dst = (uint8_t *) dst_buf + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int row = 0; row < 32; row++) {
                    const int64_t r = ct * 32 + row;
                    if (r < ne1) {
                        dequantize_row_q6_K(src_expert + r * nb_q6k, row_f32.data(), ne0);
                        quantize_row_q4_0_ref(row_f32.data(), strip_q4.data() + row * nb_q4, ne0);
                    } else {
                        memset(strip_q4.data() + row * nb_q4, 0, nb_q4 * sizeof(block_q4_0));
                    }
                }

                for (int kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;

                    uint8_t tile_quants[32][32];
                    for (int row = 0; row < 32; row++) {
                        if (kt < nb_q4) {
                            unpack_q4_0_quants(tile_quants[row], &strip_q4[row * nb_q4 + kt]);
                        } else {
                            memset(tile_quants[row], 8, 32);
                        }
                    }

                    for (int cp = 0; cp < 16; cp++) {
                        for (int row = 0; row < 32; row++) {
                            tile_dst[cp * 32 + row] = (tile_quants[row][2 * cp + 1] << 4) | tile_quants[row][2 * cp];
                        }
                    }

                    ggml_half * scale_dst = (ggml_half *)(tile_dst + 512);
                    for (int row = 0; row < 32; row++) {
                        scale_dst[row] = (kt < nb_q4) ? strip_q4[row * nb_q4 + kt].d : 0;
                    }
                }
            }
        }
    }
}

static void repack_mxfp4_tiled_to_buf(const ggml_tensor * t, const void * data, void * dst_buf) {
    const block_mxfp4 * src_matrix = (const block_mxfp4 *) data;
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    const int n_col_tiles = hex_round_up((uint32_t)ne1, 32) / 32;
    const int n_k_tiles   = hex_round_up((uint32_t)ne0, 32) / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_MXFP4;
    const size_t matrix_size = (size_t)n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            const block_mxfp4 * src_expert = src_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            uint8_t * matrix_dst = (uint8_t *) dst_buf + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    uint8_t * tile_dst = matrix_dst + (ct * n_k_tiles + kt) * tile_size;

                    uint8_t tile_quants[32][32];
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            unpack_mxfp4_quants(tile_quants[row], &src_expert[r * (ne0 / 32) + kt]);
                        } else {
                            memset(tile_quants[row], 0, 32);
                        }
                    }

                    for (int cp = 0; cp < 16; cp++) {
                        for (int row = 0; row < 32; row++) {
                            tile_dst[cp * 32 + row] = (tile_quants[row][2 * cp + 1] << 4) | tile_quants[row][2 * cp];
                        }
                    }

                    uint8_t * scale_dst = tile_dst + 512;
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        scale_dst[row] = (r < ne1 && kt < ne0 / 32) ? src_expert[r * (ne0 / 32) + kt].e : 0;
                    }
                }
            }
        }
    }
}

// Inverse repack: convert tiled layout back to canonical GGML layout.
// Used by get_tensor so CPU backends can read weight data in original format.

static void repack_tiled_q4_0_to_buf(void * dst_data, const ggml_tensor * t, size_t size) {
    block_q4_0 * dst_matrix = (block_q4_0 *) dst_data;
    int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up((uint32_t)ne0, 32);
    int64_t ne1_padded = hex_round_up((uint32_t)ne1, 32);
    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles   = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_0;
    const size_t matrix_size = (size_t)n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            block_q4_0 * dst_expert = dst_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            const uint8_t * matrix_src = (const uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    const uint8_t * tile_src = matrix_src + (ct * n_k_tiles + kt) * tile_size;

                    uint8_t tile_quants[32][32];
                    for (int cp = 0; cp < 16; cp++) {
                        for (int row = 0; row < 32; row++) {
                            uint8_t val = tile_src[cp * 32 + row];
                            tile_quants[row][2 * cp + 0] = val & 0x0F;
                            tile_quants[row][2 * cp + 1] = val >> 4;
                        }
                    }

                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            pack_q4_0_quants(&dst_expert[r * (ne0 / 32) + kt], tile_quants[row]);
                        }
                    }

                    const ggml_half * scale_src = (const ggml_half *)(tile_src + 512);
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            dst_expert[r * (ne0 / 32) + kt].d = scale_src[row];
                        }
                    }
                }
            }
        }
    }
    GGML_UNUSED(size);
}

static void repack_tiled_q4_1_to_buf(void * dst_data, const ggml_tensor * t, size_t size) {
    block_q4_1 * dst_matrix = (block_q4_1 *) dst_data;
    int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up((uint32_t)ne0, 32);
    int64_t ne1_padded = hex_round_up((uint32_t)ne1, 32);
    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles   = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_1;
    const size_t matrix_size = (size_t)n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            block_q4_1 * dst_expert = dst_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            const uint8_t * matrix_src = (const uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    const uint8_t * tile_src = matrix_src + (ct * n_k_tiles + kt) * tile_size;

                    uint8_t tile_quants[32][32];
                    for (int cp = 0; cp < 16; cp++) {
                        for (int row = 0; row < 32; row++) {
                            uint8_t val = tile_src[cp * 32 + row];
                            tile_quants[row][2 * cp + 0] = val & 0x0F;
                            tile_quants[row][2 * cp + 1] = val >> 4;
                        }
                    }

                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            pack_q4_1_quants(&dst_expert[r * (ne0 / 32) + kt], tile_quants[row]);
                        }
                    }

                    const ggml_half * scale_src = (const ggml_half *)(tile_src + 512);
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            dst_expert[r * (ne0 / 32) + kt].d = scale_src[2 * row + 0];
                            dst_expert[r * (ne0 / 32) + kt].m = scale_src[2 * row + 1];
                        }
                    }
                }
            }
        }
    }
    GGML_UNUSED(size);
}

// Inverse of repack_q8_0_tiled_to_buf: convert Q8_0 tiled layout back to
// canonical Q8_0 blocks. Used by get_tensor for CPU reference comparison.
static void repack_tiled_q8_0_to_buf(void * dst_data, const ggml_tensor * t, size_t size) {
    block_q8_0 * dst_matrix = (block_q8_0 *) dst_data;
    int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up((uint32_t)ne0, 32);
    int64_t ne1_padded = hex_round_up((uint32_t)ne1, 32);
    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles   = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q8_0;
    const size_t matrix_size = (size_t)n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            block_q8_0 * dst_expert = dst_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            const uint8_t * matrix_src = (const uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    const uint8_t * tile_src = matrix_src + (ct * n_k_tiles + kt) * tile_size;

                    for (int cp = 0; cp < 16; cp++) {
                        int col0 = cp * 2;
                        int col1 = col0 + 1;
                        for (int row = 0; row < 32; row++) {
                            int64_t r = ct * 32 + row;
                            if (r < ne1 && kt < ne0 / 32) {
                                block_q8_0 & b = dst_expert[r * (ne0 / 32) + kt];
                                b.qs[col0] = tile_src[cp * 64 + 2 * row + 0];
                                b.qs[col1] = tile_src[cp * 64 + 2 * row + 1];
                            }
                        }
                    }

                    const ggml_half * scale_src = (const ggml_half *)(tile_src + 1024);
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            dst_expert[r * (ne0 / 32) + kt].d = scale_src[row];
                        }
                    }
                }
            }
        }
    }
    GGML_UNUSED(size);
}

// Inverse of repack_mxfp4_tiled_to_buf: convert MXFP4 tiled layout back to
// canonical MXFP4 blocks. Used by get_tensor for CPU reference comparison.
static void repack_tiled_mxfp4_to_buf(void * dst_data, const ggml_tensor * t, size_t size) {
    block_mxfp4 * dst_matrix = (block_mxfp4 *) dst_data;
    int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    int64_t ne0_padded = hex_round_up((uint32_t)ne0, 32);
    int64_t ne1_padded = hex_round_up((uint32_t)ne1, 32);
    int n_col_tiles = ne1_padded / 32;
    int n_k_tiles   = ne0_padded / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_MXFP4;
    const size_t matrix_size = (size_t)n_col_tiles * n_k_tiles * tile_size;

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            block_mxfp4 * dst_expert = dst_matrix + (i3 * ne2 + i2) * (ne1 * (ne0 / 32));
            const uint8_t * matrix_src = (const uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;

            for (int ct = 0; ct < n_col_tiles; ct++) {
                for (int kt = 0; kt < n_k_tiles; kt++) {
                    const uint8_t * tile_src = matrix_src + (ct * n_k_tiles + kt) * tile_size;

                    uint8_t tile_quants[32][32];
                    for (int cp = 0; cp < 16; cp++) {
                        for (int row = 0; row < 32; row++) {
                            uint8_t val = tile_src[cp * 32 + row];
                            tile_quants[row][2 * cp + 0] = val & 0x0F;
                            tile_quants[row][2 * cp + 1] = val >> 4;
                        }
                    }

                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            pack_mxfp4_quants(&dst_expert[r * (ne0 / 32) + kt], tile_quants[row]);
                        }
                    }

                    const uint8_t * scale_src = tile_src + 512;
                    for (int row = 0; row < 32; row++) {
                        int64_t r = ct * 32 + row;
                        if (r < ne1 && kt < ne0 / 32) {
                            dst_expert[r * (ne0 / 32) + kt].e = scale_src[row];
                        }
                    }
                }
            }
        }
    }
    GGML_UNUSED(size);
}

// BF16 weights are stored as F16 bytes in the repack buffer so the DSP can
// reuse the F16 matmul kernels. Conversion is exact for values inside the
// F16 exponent range, which always holds for trained projection weights.
static void repack_bf16_to_f16(const ggml_tensor * t, const void * data) {
    const int64_t n = ggml_nelements(t);
    const ggml_bf16_t * src = (const ggml_bf16_t *) data;
    ggml_fp16_t *       dst = (ggml_fp16_t *) t->data;
    for (int64_t i = 0; i < n; i++) {
        dst[i] = ggml_fp32_to_fp16(ggml_bf16_to_fp32(src[i]));
    }
}

// Inverse of repack_bf16_to_f16, used by get_tensor for host read-back.
static void repack_f16_to_bf16(const ggml_tensor * t, void * data, size_t size) {
    const int64_t n = (int64_t)(size / sizeof(ggml_bf16_t));
    const ggml_fp16_t * src = (const ggml_fp16_t *) t->data;
    ggml_bf16_t *       dst = (ggml_bf16_t *) data;
    for (int64_t i = 0; i < n; i++) {
        dst[i] = ggml_fp32_to_bf16(ggml_fp16_to_fp32(src[i]));
    }
}

// Inverse of repack_q4k_as_q4_0_tiled_to_buf, used by get_tensor for host
// read-back: gather each canonical Q4_0 row from the tiled layout, dequant
// to f32 and requant to Q4_K. Row-at-a-time to keep scratch small.
static void repack_tiled_q4_0_to_q4k_buf(void * dst_data, const ggml_tensor * t, size_t size) {
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
    const int n_col_tiles = hex_round_up((uint32_t)ne1, 32) / 32;
    const int n_k_tiles   = hex_round_up((uint32_t)ne0, 32) / 32;
    const size_t tile_size = HTP_MM_WEIGHT_TILE_SIZE_Q4_0;
    const size_t matrix_size = (size_t)n_col_tiles * n_k_tiles * tile_size;
    const int64_t nb_q4  = ne0 / QK4_0;
    const int64_t nb_q4k = ne0 / QK_K;

    std::vector<float>       row_f32(ne0);
    std::vector<block_q4_0>  row_q4(nb_q4);

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            const uint8_t * matrix_src = (const uint8_t *) t->data + (i3 * ne2 + i2) * matrix_size;
            block_q4_K * dst_expert = (block_q4_K *) dst_data + (i3 * ne2 + i2) * (ne1 * nb_q4k);

            for (int64_t r = 0; r < ne1; r++) {
                const int ct  = (int)(r / 32);
                const int row = (int)(r % 32);
                for (int64_t kt = 0; kt < nb_q4; kt++) {
                    const uint8_t * tile_src = matrix_src + (ct * n_k_tiles + kt) * tile_size;
                    uint8_t q[32];
                    for (int cp = 0; cp < 16; cp++) {
                        const uint8_t packed = tile_src[cp * 32 + row];
                        q[2 * cp + 0] = packed & 0x0F;
                        q[2 * cp + 1] = packed >> 4;
                    }
                    for (int i = 0; i < 16; i++) {
                        row_q4[kt].qs[i] = (uint8_t)(q[i + 16] << 4) | q[i];
                    }
                    row_q4[kt].d = ((const ggml_half *)(tile_src + 512))[row];
                }
                dequantize_row_q4_0(row_q4.data(), row_f32.data(), ne0);
                quantize_row_q4_K_ref(row_f32.data(), dst_expert + r * nb_q4k, ne0);
            }
        }
    }
    GGML_UNUSED(size);
}

static void ggml_backend_hexagon_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                               ggml_tensor * tensor, const void * data,
                                               size_t offset, size_t size) {
    // Repack quantized types into tiled (HMX) layout only when the tensor
    // lives in the repack buffer (e.g. weights loaded from GGUF). Tensors
    // in the main buffer (e.g. test-backend-ops allocations) stay in
    // canonical GGML format.
    static int set_tensor_call_count = 0;
    bool is_repack = ggml_backend_buffer_is_hexagon_repack(buffer);
    if (set_tensor_call_count < 10 || is_repack) {
        GGMLHEXAGON_LOG_INFO("[SET_TENSOR] #%d name=%s type=%d ne=[%d,%d,%d,%d] nbytes=%zu is_repack=%d offset=%zu size=%zu\n",
                               set_tensor_call_count, tensor->name, (int)tensor->type,
                               (int)tensor->ne[0], (int)tensor->ne[1], (int)tensor->ne[2], (int)tensor->ne[3],
                               ggml_nbytes(tensor), (int)is_repack, offset, size);
    }
    set_tensor_call_count++;
    if (is_repack) {
        switch (tensor->type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_IQ4_NL:  // identical block layout to Q4_0
                GGML_ASSERT(offset == 0);
                GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
                repack_q4_0_tiled_to_buf(tensor, data, tensor->data);
                break;
            case GGML_TYPE_Q4_1:
                GGML_ASSERT(offset == 0);
                GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
                repack_q4_1_tiled_to_buf(tensor, data, tensor->data);
                break;
            case GGML_TYPE_Q8_0:
                GGML_ASSERT(offset == 0);
                GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
                repack_q8_0_tiled_to_buf(tensor, data, tensor->data);
                break;
            case GGML_TYPE_MXFP4:
                GGML_ASSERT(offset == 0);
                GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
                repack_mxfp4_tiled_to_buf(tensor, data, tensor->data);
                break;
            case GGML_TYPE_BF16:
                GGML_ASSERT(offset == 0);
                GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
                repack_bf16_to_f16(tensor, data);
                break;
            case GGML_TYPE_Q4_K:
                GGML_ASSERT(offset == 0);
                GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
                repack_q4k_as_q4_0_tiled_to_buf(tensor, data, tensor->data);
                break;
            case GGML_TYPE_Q6_K:
                GGML_ASSERT(offset == 0);
                GGML_ASSERT(offset + size <= ggml_nbytes(tensor));
                repack_q6k_as_q4_0_tiled_to_buf(tensor, data, tensor->data);
                break;
            default:
                memcpy((char *)tensor->data + offset, data, size);
                break;
        }
    } else {
        memcpy((char *)tensor->data + offset, data, size);
    }

    // Mark weights dirty so Phase 9 flushes them on the next batch.
    // For repack-buft weights, flush immediately after repack; this moves the
    // cold-state cache-clean cost from the first inference batch to model-load
    // time and lets Phase 9 skip already-coherent repack weights.
    ggml_backend_hexagon_buffer_context * bctx =
        (ggml_backend_hexagon_buffer_context *)buffer->context;
    if (bctx && bctx->is_ion_buffer && bctx->backend_ctx) {
        ggml_backend_hexagon_context * hctx = bctx->backend_ctx;
        const char * dp   = (const char *)tensor->data + offset;
        const char * base = (const char *)hctx->rpc_mempool;
        if (dp >= base && dp < base + (ptrdiff_t)hctx->rpc_mempool_len) {
            if (is_repack) {
                // stored size can differ from ggml_nbytes (Q4_K is kept as
                // Q8_0 tiled, roughly 2x the logical size)
                size_t flush_len = ggml_hexagon_repacked_size(tensor->type, tensor->ne[0], tensor->ne[1],
                                                              tensor->ne[2], tensor->ne[3]);
                if (flush_len == 0) flush_len = ggml_nbytes(tensor);
                cpu_dcache_flush_range(hctx, hctx->rpc_mempool_handle,
                                       tensor->data, flush_len);
                hctx->weights_dirty = false;
            } else {
                hctx->weights_dirty = true;
            }
        }
    }
}

static void ggml_backend_hexagon_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                  struct ggml_tensor * tensor,
                                                  uint8_t value, size_t offset, size_t size) {
    memset((char *)tensor->data + offset, value, size);

    ggml_backend_hexagon_buffer_context * bctx =
        (ggml_backend_hexagon_buffer_context *)buffer->context;
    if (bctx && bctx->is_ion_buffer && bctx->backend_ctx) {
        ggml_backend_hexagon_context * hctx = bctx->backend_ctx;
        const char * dp   = (const char *)tensor->data + offset;
        const char * base = (const char *)hctx->rpc_mempool;
        if (dp >= base && dp < base + (ptrdiff_t)hctx->rpc_mempool_len) {
            hctx->weights_dirty = true;
        }
    }
}

static void ggml_backend_hexagon_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                               const ggml_tensor * tensor,
                                               void * data, size_t offset, size_t size) {
    // Un-repack tiled layout back to canonical GGML format only when
    // the tensor lives in the repack buffer.
    if (ggml_backend_buffer_is_hexagon_repack(buffer)) {
        if (offset == 0 && size == ggml_nbytes(tensor)) {
            switch (tensor->type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_IQ4_NL:
                    repack_tiled_q4_0_to_buf(data, tensor, size);
                    return;
                case GGML_TYPE_Q4_1:
                    repack_tiled_q4_1_to_buf(data, tensor, size);
                    return;
                case GGML_TYPE_Q8_0:
                    repack_tiled_q8_0_to_buf(data, tensor, size);
                    return;
                case GGML_TYPE_MXFP4:
                    repack_tiled_mxfp4_to_buf(data, tensor, size);
                    return;
                case GGML_TYPE_BF16:
                    repack_f16_to_bf16(tensor, data, size);
                    return;
                case GGML_TYPE_Q4_K:
                    repack_tiled_q4_0_to_q4k_buf(data, tensor, size);
                    return;
                // Q6_K get_tensor (tiled Q4_0 -> Q6_K) not implemented;
                // falls through to raw memcpy. Inference is unaffected since
                // the DSP reads the tiled Q4_0 layout directly.
                default:
                    break;
            }
        }
    }
    memcpy(data, (const char *)tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

static bool ggml_backend_hexagon_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                               const struct ggml_tensor * src,
                                               struct ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    // take the slow path via get/set_tensor (which handles repack/un-repack)
    return false;
}

static void ggml_backend_hexagon_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_hexagon_buffer_context * ctx = (ggml_backend_hexagon_buffer_context *)buffer->context;
    memset(ctx->buffer, value, ctx->buffer_size);
}

static void ggml_backend_hexagon_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_UNUSED(buffer);
    for (size_t copy = 0; copy < n_copies; copy++) {
        memcpy((char *)tensor->data + offset + copy * stride_tensor, (const char *)data + copy * stride_data, size);
    }
}

static void ggml_backend_hexagon_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_UNUSED(buffer);
    for (size_t copy = 0; copy < n_copies; copy++) {
        memcpy((char *)data + copy * stride_data, (const char *)tensor->data + offset + copy * stride_tensor, size);
    }
}

static ggml_backend_buffer_i ggml_backend_hexagon_buffer_interface = {
        /* .free_buffer     = */ ggml_backend_hexagon_buffer_free_buffer,
        /* .get_base        = */ ggml_backend_hexagon_buffer_get_base,
        /* .init_tensor     = */ ggml_backend_hexagon_buffer_init_tensor,
        /* .memset_tensor   = */ ggml_backend_hexagon_buffer_memset_tensor,
        /* .set_tensor      = */ ggml_backend_hexagon_buffer_set_tensor,
        /* .get_tensor      = */ ggml_backend_hexagon_buffer_get_tensor,
        /* .set_tensor_2d   = */ ggml_backend_hexagon_buffer_set_tensor_2d,
        /* .get_tensor_2d   = */ ggml_backend_hexagon_buffer_get_tensor_2d,
        /* .cpy_tensor      = */ ggml_backend_hexagon_buffer_cpy_tensor,
        /* .clear           = */ ggml_backend_hexagon_buffer_clear,
        /* .reset           = */ nullptr,
};

static const char * ggml_backend_hexagon_buffer_type_name(ggml_backend_buffer_type_t buft) {
    struct ggml_backend_hexagon_context * ctx = static_cast<ggml_backend_hexagon_context *>(buft->context);
    if (ctx) {
        if (buft == &ctx->repack_buffer_type) {
            return ctx->repack_buft_name;
        }
        return ctx->buft_name;
    }
    return "hexagon-ion-buffer";
}

static ggml_backend_buffer_t ggml_backend_hexagon_buffer_type_alloc_buffer(
           ggml_backend_buffer_type_t buft, size_t size) {
    struct ggml_backend_hexagon_context * ctx = static_cast<ggml_backend_hexagon_context *>(buft->context);
    GGML_ASSERT(nullptr != ctx);
    GGMLHEXAGON_LOG_ALWAYS("[ALLOC] ENTER device=%d size=%zu bytes (%.2f MiB)", ctx->device, size, (double)size / (1024.0 * 1024.0));
    ggml_backend_hexagon_buffer_context * buffer_ctx = new ggml_backend_hexagon_buffer_context;
    buffer_ctx->backend_ctx = ctx;
    buffer_ctx->is_ion_buffer = true;

    size_t size_page = 0;
#if defined(__ANDROID__) || defined(__linux__)
    size_page = sysconf(_SC_PAGESIZE);
#endif
    size_t size_aligned = size;
    if (0 != (size_aligned % size_page)) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    GGMLHEXAGON_LOG_ALWAYS("device %d(%s)", ctx->device, ctx->name);
    GGML_ASSERT(nullptr != ctx->rpc_mempool);
    GGMLHEXAGON_LOG_ALWAYS("device=%d size %ld(%d MiB), rpc_mempool_usage %ld(%d MiB), rpc_mempool_len %ld(%d MiB)",
                          ctx->device, size, size / SIZE_IN_MB, ctx->rpc_mempool_usage, ctx->rpc_mempool_usage / SIZE_IN_MB,
                          ctx->rpc_mempool_len, ctx->rpc_mempool_len / SIZE_IN_MB);

    size_t data_limit = ctx->rpc_mempool_len;

    // Try to reuse a free region (best fit)
    size_t best_idx = (size_t)-1;
    size_t best_waste = (size_t)-1;
    for (size_t ri = 0; ri < ctx->ion_regions.size(); ri++) {
        const auto & r = ctx->ion_regions[ri];
        if (!r.in_use && r.size >= size_aligned) {
            size_t waste = r.size - size_aligned;
            if (waste < best_waste) {
                best_waste = waste;
                best_idx = ri;
            }
        }
    }

    if (best_idx != (size_t)-1) {
        // Reuse free region
        auto & r = ctx->ion_regions[best_idx];
        buffer_ctx->buffer      = (char *)ctx->rpc_mempool + r.offset;
        buffer_ctx->buffer_size = size_aligned;  // actual requested size, not region size
        r.in_use = true;
        if (r.offset + size_aligned > ctx->rpc_mempool_usage) {
            ctx->rpc_mempool_usage = r.offset + size_aligned;
        }
        GGMLHEXAGON_LOG_ALWAYS("[ALLOC] device=%d reuse free region: offset=%zu size=%zu (requested=%zu, waste=%zu)",
                             ctx->device, r.offset, r.size, size_aligned, r.size - size_aligned);
        memset(buffer_ctx->buffer, 0, buffer_ctx->buffer_size);
    } else {
        // Allocate new region from bump allocator tail
        size_t aligned_offset = ((ctx->rpc_mempool_usage + 127) / 128) * 128;
        if (aligned_offset + size_aligned <= data_limit) {
            buffer_ctx->buffer      = (char *)ctx->rpc_mempool + aligned_offset;
            buffer_ctx->buffer_size = size_aligned;
            ctx->rpc_mempool_usage  = aligned_offset + size_aligned;
            // Record new region
            ion_pool_region new_region;
            new_region.offset = aligned_offset;
            new_region.size   = size_aligned;
            new_region.in_use = true;
            ctx->ion_regions.push_back(new_region);
            GGMLHEXAGON_LOG_ALWAYS("[ALLOC] device=%d new region: offset=%zu size=%zu", ctx->device, aligned_offset, size_aligned);
        } else {
            GGMLHEXAGON_LOG_ALWAYS("device=%d ion pool exhausted: needed %zu MiB, remaining %zu MiB -- falling back to system memory",
                                 ctx->device, size_aligned / SIZE_IN_MB,
                                 (data_limit - ctx->rpc_mempool_usage) / SIZE_IN_MB);
            buffer_ctx->buffer = ggml_aligned_malloc(size_aligned);
            buffer_ctx->buffer_size = size_aligned;
            buffer_ctx->is_ion_buffer = false;
        }
    }

    if (nullptr == buffer_ctx->buffer) {
        GGMLHEXAGON_LOG_ERROR("%s: failed to allocate %d MiB\n", __func__, size / SIZE_IN_MB);
        return nullptr;
    } else {
        GGMLHEXAGON_LOG_ALWAYS("%s: succeed to allocate %d MiB\n", __func__, size / SIZE_IN_MB);
    }
    // Report allocation result and current mempool state
    if (buffer_ctx->is_ion_buffer) {
        const char * mem_type = "heap";
        const char * data_ptr = (const char *)buffer_ctx->buffer;
        const char * ion_base = (const char *)ctx->rpc_mempool;
        const char * ion_end  = ion_base + ctx->rpc_mempool_len;
        if (data_ptr >= ion_base && data_ptr < ion_end) {
            mem_type = "mempool";
        }
        GGMLHEXAGON_LOG_ALWAYS("[ALLOC] device=%d LEAVE size=%zu (%.2f MiB) -> %s, pool_used=%zu/%zu (%.2f%%)",
                             ctx->device, size, (double)size / (1024.0 * 1024.0),
                             mem_type,
                             ctx->rpc_mempool_usage, ctx->rpc_mempool_len,
                             ctx->rpc_mempool_len > 0 ? (double)ctx->rpc_mempool_usage * 100.0 / ctx->rpc_mempool_len : 0.0);
    } else {
        GGMLHEXAGON_LOG_ALWAYS("[ALLOC] device=%d LEAVE size=%zu (%.2f MiB) -> heap", ctx->device, size, (double)size / (1024.0 * 1024.0));
    }
    return ggml_backend_buffer_init(buft, ggml_backend_hexagon_buffer_interface, buffer_ctx, size);
}

static size_t ggml_backend_hexagon_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    //Alignment requirement in bytes
    return 128;
}

static size_t ggml_backend_hexagon_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    struct ggml_backend_hexagon_context * ctx = static_cast<ggml_backend_hexagon_context *>(buft->context);
    GGML_ASSERT(nullptr != ctx);
    GGML_ASSERT(ctx->rpc_mempool_len > (8 * SIZE_IN_MB));
    return ctx->rpc_mempool_len - (8 * SIZE_IN_MB);
}

static size_t ggml_backend_hexagon_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    // For quantized weight types that will be repacked to tile-based layout
    // in set_tensor, allocate enough space for the repacked data.
    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
            return ggml_hexagon_repacked_size(tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        default:
            return ggml_nbytes(tensor);
    }
}

static bool ggml_backend_buft_is_hexagon(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_hexagon_buffer_type_name;
}

// Repack buft is identified by comparing the buft pointer against the
// repack_buffer_type member stored in ggml_backend_hexagon_context (both
// main and repack bufts share the same context pointer). Used by
// supports_buft to allow GGML core to route quantized weights through
// set_tensor (which does the in-place tile repack).
static bool ggml_backend_buft_is_hexagon_repack(ggml_backend_buffer_type_t buft) {
    auto * ctx = (ggml_backend_hexagon_context *)buft->context;
    return buft == &ctx->repack_buffer_type;
}

static bool ggml_backend_hexagon_buffer_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

// Repack buffer type: is_host=false forces GGML core to call set_tensor
// (which does the repack) instead of reading model data directly into
// tensor->data. Both main and repack buffer types manage the same mempool
// shared memory pool.
static bool ggml_backend_hexagon_repack_buffer_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

// Returns true if this buffer was allocated from the repack buffer type.
// Repack buffers hold quantized weight data in tiled (HMX) layout and
// require set_tensor/get_tensor to repack/unrepack across the boundary.
static bool ggml_backend_buffer_is_hexagon_repack(const struct ggml_backend_buffer * b) {
    auto * ctx = (ggml_backend_hexagon_context *)b->buft->context;
    return b->buft == &ctx->repack_buffer_type;
}

// Session consistency check:
//   - tensor is null:                  neutral, accept (compute-temporary-like)
//   - tensor has no buffer assigned:   neutral, accept (scheduler will route)
//   - buffer is hexagon (main or repack) on this device: accept
//   - buffer is hexagon on a different device: reject (wrong session)
//   - buffer is non-hexagon (e.g. CPU): reject (scheduler should keep on CPU)
//
// Both the main and repack bufts store the owning ggml_backend_hexagon_context
// pointer at buft->context (set in the constructor as `buffer_type.context = this`).
// The current device's context lives at dev->context.
static bool ggmlhexagon_tensor_buffer_is_owned_by(ggml_backend_dev_t dev, const struct ggml_tensor * t) {
    if (!t || !t->buffer) {
        return true;
    }
    ggml_backend_buffer_type_t buft = t->buffer->buft;
    if (!ggml_backend_buft_is_hexagon(buft) && !ggml_backend_buft_is_hexagon_repack(buft)) {
        return false;
    }
    ggml_backend_hexagon_context * dev_ctx  = (ggml_backend_hexagon_context *)dev->context;
    ggml_backend_hexagon_context * buft_ctx = (ggml_backend_hexagon_context *)buft->context;
    // same logic as ggml_backend_hexagon_device_supports_buft
    return buft_ctx->device == dev_ctx->device;
}

// All srcs and the dst of the op must be mapped to the same hexagon session
// (device). Tensors with no buffer are treated as neutral. Without this,
// the scheduler can incorrectly assign an op to a device whose tensors live
// in another device's mempool region, which would fault on the DSP since
// mempool mappings are not shared across separate FastRPC sessions.
static bool ggmlhexagon_op_buffers_belong_to_dev(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    if (!ggmlhexagon_tensor_buffer_is_owned_by(dev, op)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (!ggmlhexagon_tensor_buffer_is_owned_by(dev, op->src[i])) {
            return false;
        }
    }
    return true;
}

static const char * ggml_backend_hexagon_name(ggml_backend_t backend) {
    ggml_backend_hexagon_context * ctx = (ggml_backend_hexagon_context *) backend->context;
    return ctx->name;
}

static void ggml_backend_hexagon_free(ggml_backend_t backend) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );
    // only delete the backend here; the context (including buffer types)
    // must persist across inferences so that model tensors which reference
    // buffer types remain valid. The context is owned by the device and
    // freed during registry shutdown.
    ggml_backend_hexagon_context * ctx = (ggml_backend_hexagon_context *)backend->context;
    ctx->backend = nullptr;
    delete backend;

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
}

// mempool-based op-batch - packs all ops into mempool shared memory,
// passes only (offset, size) via FastRPC as doorbell.
// avoids FastRPC scatter-gather limits entirely.
static enum ggml_status ggmlhexagon_backend_graph_compute_batch(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status result             = GGML_STATUS_SUCCESS;
    int64_t begin_time                  = ggml_time_us();
    ggml_backend_hexagon_context * ctx  = (ggml_backend_hexagon_context *)backend->context;
    int64_t gap_from_prev               = ctx->last_graph_end_us ? (begin_time - ctx->last_graph_end_us) : 0;
    (void)gap_from_prev; // used only in GGMLHEXAGON_LOG_DEBUG
    uint32_t graph_n_nodes              = (uint32_t)cgraph->n_nodes;
    const char * ion_base               = (const char *)ctx->rpc_mempool;
    const size_t ion_size               = ctx->rpc_mempool_len;
    size_t saved_mempool_usage          = ctx->rpc_mempool_usage;
    uint32_t n_tensors                  = 0;
    uint32_t n_ops                      = 0;
    bool cache_hit                      = false;

    // Snapshot cumulative phase counters at entry so we can compute the
    // current call's accounted time at exit (used for unaccounted-time).
    int64_t snap_p1                     = ctx->cum_p1_us;
    int64_t snap_p2                     = ctx->cum_p2_us;
    int64_t snap_p3                     = ctx->cum_p3_us;
    int64_t snap_p4                     = ctx->cum_p4_us;
    int64_t snap_p7                     = ctx->cum_p7_us;
    int64_t snap_p10                    = ctx->cum_p10_us;

    // Phase timing: declare all timers here, used across the pipeline
    int64_t t_start, t_p1, t_p2, t_p3, t_p4, t_p5, t_p6, t_p7, t_p8, t_p9, t_p10, t_p11, t_p12;

    // Track temporary mempool regions (mirrors, batch descriptors, repacked weights)
    // for cleanup after Phase 12. Mark them as free (no tail compaction).
    std::vector<size_t>         temp_region_indices;

    // Storage for cache-miss path; on cache hit we reference cached vectors
    // directly to avoid copying ~20-110 KB of descriptors per call.
    std::vector<ggml_tensor *>  local_tensor_src;
    std::vector<hex_op_desc>    local_hex_ops;
    std::vector<uint8_t>        local_is_weight;
    std::vector<int32_t>        local_mirror_offset;  // per-tensor heap mirror mempool offset, -1 = none

    // supported_nodes is only required to build descriptors on cache miss.
    // On a cache hit we restore the derived descriptors directly, so avoid
    // scanning the cgraph and allocating this vector in the hot path.
    std::vector<ggml_tensor *>  supported_nodes;

    ggml_backend_hexagon_context::cgraph_cache_entry * cached_entry = nullptr;

    // track per-graph node statistics (what ggml core assigned to this backend)
    ctx->total_nodes_processed += graph_n_nodes;
    if (ctx->min_nodes_per_graph == 0 || graph_n_nodes < ctx->min_nodes_per_graph) {
        ctx->min_nodes_per_graph = graph_n_nodes;
    }
    if (graph_n_nodes > ctx->max_nodes_per_graph) {
        ctx->max_nodes_per_graph = graph_n_nodes;
    }

    t_start = ggml_time_us();

    // ---- Phase 1: collect unique tensor objects + cgraph cache lookup ----
    // Hash over each node's {op, ne[4], nb[4], non-null src[0..GGML_MAX_SRC-1] ptr, data ptr}.
    // The src slot index is folded into the value so that NULL slots do not
    // collide with real tensors at different positions. The node's own data
    // pointer is kept because distinct dst tensors can share the same
    // op/shape/srcs (e.g., buffer reuse), and omitting it caused cache hits
    // to restore descriptors pointing to stale tensors.
    // cgraph pointer is NOT used: the scheduler rebuilds split->graph every
    // call, so the pointer churns. The content is stable.
    auto compute_content_hash = [&]() -> uint64_t {
        uint64_t h = 0xcbf29ce484222325ULL;  // FNV-1a 64-bit offset basis
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];
            if (!node) continue;
            h ^= (uint64_t)node->op; h *= 0x100000001b3ULL;
            for (int j = 0; j < 4; j++) { h ^= (uint64_t)node->ne[j]; h *= 0x100000001b3ULL; }
            for (int j = 0; j < 4; j++) { h ^= (uint64_t)node->nb[j]; h *= 0x100000001b3ULL; }
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor * src = node->src[j];
                if (src) {
                    h ^= (uint64_t)(uintptr_t)src ^ (uint64_t)j;
                    h *= 0x100000001b3ULL;
                }
            }
            h ^= (uint64_t)(uintptr_t)node->data; h *= 0x100000001b3ULL;
        }
        return h;
    };

    const uint64_t content_hash = compute_content_hash();
    {
        auto it = ctx->cgraph_cache.find(content_hash);
        if (it != ctx->cgraph_cache.end() &&
            it->second.n_nodes == cgraph->n_nodes &&
            it->second.hex_ops.size() > 0) {
            cached_entry = &it->second;
            cache_hit = true;
            ctx->cgraph_cache_hits++;
        } else {
            ctx->cgraph_cache_misses++;
        }
    }

    // Bind to cached descriptors on hit, local vectors on miss. This avoids
    // the expensive assign() of hex_ops/tensor_src in the hot path.
    std::vector<ggml_tensor *> & tensor_src = cache_hit ? cached_entry->tensor_src  : local_tensor_src;
    std::vector<hex_op_desc>   & hex_ops    = cache_hit ? cached_entry->hex_ops     : local_hex_ops;
    std::vector<uint8_t>       & is_weight  = cache_hit ? cached_entry->is_weight   : local_is_weight;

    if (cache_hit) {
        n_tensors = (uint32_t)cached_entry->n_tensors;
        n_ops     = (uint32_t)cached_entry->n_ops;
    }

    if (!cache_hit) {
        // ---- collect supported ops (cache miss only) ----
        supported_nodes.reserve(cgraph->n_nodes);
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];

            if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE
                || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW
                || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
                continue;
            }

            supported_nodes.push_back(node);
        }

        if (supported_nodes.empty()) {
            return result;
        }

        tensor_src.reserve(cgraph->n_nodes);
        hex_ops.reserve(supported_nodes.size());
    }

    t_p1 = t_start; t_start = ggml_time_us(); ctx->cum_p1_us += t_start - t_p1;

    // ---- Phase 2: build op descriptors (cache miss only) ----
    if (!cache_hit) {
        std::unordered_map<ggml_tensor *, int32_t> tensor_index_map;
        tensor_index_map.reserve(cgraph->n_nodes * 2);
        auto get_or_add_tensor_idx = [&](ggml_tensor * t) -> int32_t {
            if (!t) return -1;
            auto it = tensor_index_map.find(t);
            if (it != tensor_index_map.end()) return it->second;
            int32_t idx = (int32_t)tensor_src.size();
            tensor_index_map[t] = idx;
            tensor_src.push_back(t);
            return idx;
        };

        for (auto * node : supported_nodes) {
            hex_op_desc op;
            memset(&op, 0, sizeof(op));
            for (int k = 0; k < 4; k++) op.dst_idx[k] = -1;
            for (int k = 0; k < 6; k++) op.src_idx[k] = -1;
            op.opcode   = node->op;
            memcpy(op.params, node->op_params, sizeof(op.params));
            if (node->op == GGML_OP_MUL_MAT) {
                ggml_hexagon_precompute_mm_params(ctx, node, op, false);
                ctx->n_mul_mat_total_cum++;
                if (((const struct htp_mm_kernel_params *) op.kernel_params)->n_hmx) {
                    ctx->n_hmx_used_cum++;
                }
            } else if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                ggml_hexagon_compute_fa_params(ctx, node,
                    (struct htp_fa_kernel_params *) op.kernel_params);
            } else {
                // Unary-family ops (NORM, RMS_NORM, SCALE, SQR, SQRT, UNARY_*,
                // L2_NORM, TRI) require host-precomputed htp_unary_kernel_params
                // since upstream commit fb30ba9a6. Without this the DSP reads
                // zeroed kparams (n_threads=0, etc.) and the output is garbled.
                uint32_t unary_htp_op = 0;
                if (ggml_op_to_htp_op_unary(node->op, node->op_params, &unary_htp_op)) {
                    ggml_hexagon_precompute_unary_params(ctx, unary_htp_op,
                        node->src[0], node->src[1], node,
                        (struct htp_unary_kernel_params *) op.kernel_params);
                    op.htp_opcode = (int32_t) unary_htp_op;
                }
            }
            op.src_idx[0] = get_or_add_tensor_idx(node->src[0]);
            op.src_idx[1] = (node->src[1]) ? get_or_add_tensor_idx(node->src[1]) : -1;
            op.src_idx[2] = (node->src[2]) ? get_or_add_tensor_idx(node->src[2]) : -1;
            op.src_idx[3] = (node->src[3]) ? get_or_add_tensor_idx(node->src[3]) : -1;
            op.src_idx[4] = (node->src[4]) ? get_or_add_tensor_idx(node->src[4]) : -1;
            op.src_idx[5] = (node->src[5]) ? get_or_add_tensor_idx(node->src[5]) : -1;
            op.dst_idx[0]  = get_or_add_tensor_idx(node);
            hex_ops.push_back(op);
        }

        n_tensors = (uint32_t)tensor_src.size();

        GGMLHEXAGON_LOG_DEBUG("mempool-batch %zu ops, %u unique tensors", hex_ops.size(), n_tensors);
        if (1 == g_hexagon_appcfg.dump_debug_info) {
            for (size_t i = 0; i < hex_ops.size(); i++) {
                const hex_op_desc & o = hex_ops[i];
                GGML_UNUSED(o);
                GGMLHEXAGON_LOG_DEBUG("  ion-op[%zu] %s: src0[t%d] src1[t%d] src2[t%d] dst[t%d]",
                                  i, ggml_op_name((ggml_op)o.opcode),
                                  o.src_idx[0], o.src_idx[1], o.src_idx[2], o.dst_idx[0]);
            }
        }

        // Identify weight tensors: src0 of MUL_MAT that is NOT dst of any op.
        // Weights are read-only across batches; AP never modifies them per batch,
        // so cache flush/invalidate can be skipped for them.
        // A tensor that was dst of any op in ANY cgraph (not just this one) is
        // not a read-only weight: check the session-global ever-dst set, else
        // cross-graph staleness occurs with bit 0 (e.g. qwen3-mtp garble).
        {
            static std::unordered_set<const void *> g_ever_dst_ptrs;
            for (const auto & op : hex_ops) {
                uint32_t didx = op.dst_idx[0];
                if (didx < n_tensors) g_ever_dst_ptrs.insert(tensor_src[didx]->data);
            }
            is_weight.assign(n_tensors, 0);
            std::vector<uint8_t> dst_indices(n_tensors, 0);   // indices of tensors that are dst of any op
            for (const auto & op : hex_ops) {
                uint32_t didx = op.dst_idx[0];
                if (didx < n_tensors) dst_indices[didx] = 1;
            }
            for (const auto & op : hex_ops) {
                if (op.opcode == GGML_OP_MUL_MAT) {
                    uint32_t sidx = op.src_idx[0];
                    if (sidx < n_tensors && !dst_indices[sidx] &&
                        !g_ever_dst_ptrs.count(tensor_src[sidx]->data)) {
                        is_weight[sidx] = 1;
                        GGMLHEXAGON_LOG_WARN("weight-cache: tensor[%d] identified as weight (type=%d)",
                                             sidx, (int)tensor_src[sidx]->type);
                    }
                }
            }
        }
    }  // end if (!cache_hit) for Phase 2

    t_p2 = t_start; t_start = ggml_time_us(); ctx->cum_p2_us += t_start - t_p2;

    // ---- Phase 3: op fusion ----
    // Supported fusions:
    //   RMS_NORM + MUL      -> HTP_OP_RMS_NORM_MUL
    //   MUL_MAT + ADD       -> HTP_OP_MUL_MAT_ADD     (bias add inside kernel)
    //   3x MUL_MAT (Q,K,V)  -> HTP_OP_MUL_MAT_QKV     (via DSP execute_op)
    //   2x MUL_MAT (gate,up)-> HTP_OP_MUL_MAT_FFN     (via DSP execute_op)
    //
    // QKV/FFN fusion eligibility:
    //   quantized src0 + F32 src1 + !mm_is_hmx_eligible.
    //   HMX-eligible MUL_MATs are excluded: fusion redirects to HVX fused
    //   kernels, while HMX-eligible ops benefit more from the HMX pipeline.
    if (!cache_hit) {
        // Count src usages of each tensor to ensure fused dst is single-use
        std::vector<int> src_use_count(n_tensors, 0);
        for (const auto & op : hex_ops) {
            for (int k = 0; k < 6; k++) {
                if (op.src_idx[k] >= 0 && op.src_idx[k] < (int)n_tensors) {
                    src_use_count[op.src_idx[k]]++;
                }
            }
        }

        std::vector<hex_op_desc> fused_ops;
        fused_ops.reserve(hex_ops.size());
        size_t n_rms_norm_mul = 0;
        size_t n_mul_mat_add  = 0;
        size_t n_mul_mat_qkv  = 0;
        size_t n_mul_mat_ffn  = 0;
        size_t n_mm_add_skip_use_count    = 0;  // MUL_MAT+ADD candidate but src_use_count > 1
        size_t n_mm_add_skip_not_adjacent = 0;  // MUL_MAT not followed by ADD
        size_t n_mm_add_skip_vtcm         = 0;  // MUL_MAT+ADD candidate but VTCM budget exceeded

        const size_t vtcm_budget = ctx->socinfo.vtcm_size_in_mb * 1024 * 1024;
        // QKV/FFN fusion prerequisites:
        //   - dispatches via DSP execute_op, which provides
        //     op_matmul_qkv / op_matmul_ffn as dedicated fused kernels (in kernels/*.c).
        // htp_arch>=V73 is required because op_matmul_qkv/ffn use HMX instructions.
        bool qkv_ffn_enabled = (ctx->socinfo.htp_arch >= V73 && g_hexagon_appcfg.enable_opfusion);
        for (size_t i = 0; i < hex_ops.size(); i++) {
            hex_op_desc op = hex_ops[i];

            // RMS_NORM + MUL -> RMS_NORM_MUL
            if (op.opcode == GGML_OP_RMS_NORM && i + 1 < hex_ops.size()) {
                const hex_op_desc & next = hex_ops[i + 1];
                if (next.opcode == GGML_OP_MUL &&
                    next.src_idx[0] == op.dst_idx[0] &&
                    src_use_count[op.dst_idx[0]] == 1) {
                    op.htp_opcode = HTP_OP_RMS_NORM_MUL;
                    op.src_idx[1] = next.src_idx[1];
                    op.dst_idx[0] = next.dst_idx[0];
                    // Precompute unary kparams for the fused op (src0 = RMS_NORM's
                    // input, src1 = MUL's other input, dst = MUL's output).
                    if (op.src_idx[0] >= 0 && op.src_idx[1] >= 0 && op.dst_idx[0] >= 0) {
                        ggml_hexagon_precompute_unary_params(ctx, HTP_OP_RMS_NORM_MUL,
                            tensor_src[op.src_idx[0]], tensor_src[op.src_idx[1]], tensor_src[op.dst_idx[0]],
                            (struct htp_unary_kernel_params *) op.kernel_params);
                    }
                    fused_ops.push_back(op);
                    i++;
                    n_rms_norm_mul++;
                    continue;
                }
            }

            // QKV/FFN fusion: requires DSP execute_op.
            if (qkv_ffn_enabled && op.opcode == GGML_OP_MUL_MAT) {
                // QKV fusion: 3 MUL_MAT (Q,K,V) -> HTP_OP_MUL_MAT_QKV.
                // The Q/K/V MUL_MATs may appear in either Q,K,V or Q,V,K order
                // depending on the model (e.g. Gemma4/Llama3 uses Q,K,V, Qwen3 uses Q,V,K).
                // Detect the actual order from tensor names and map src/dst accordingly.
                // DSP-side expects: src[0]=Wk, src[1]=x, src[2]=Wv, src[3]=Wq; dst[0]=K, dst[1]=V, dst[2]=Q.
                if (i + 2 < hex_ops.size()) {
                    const hex_op_desc & next1 = hex_ops[i + 1];
                    const hex_op_desc & next2 = hex_ops[i + 2];
                    if (next1.opcode == GGML_OP_MUL_MAT && next2.opcode == GGML_OP_MUL_MAT) {
                        const ggml_tensor * n_q = tensor_src[op.dst_idx[0]];
                        const ggml_tensor * n1  = tensor_src[next1.dst_idx[0]];
                        const ggml_tensor * n2  = tensor_src[next2.dst_idx[0]];
                        if (is_qkv_mergeable(ctx, n_q, n1, n2)) {
                            // Determine which of n1/n2 is K and which is V by tensor name.
                            // Models name their Q/K/V projection outputs as Qcur-* / Kcur-* /Vcur-*.
                            auto is_k = [](const ggml_tensor * t) { return t && strstr(t->name, "Kcur"); };
                            auto is_v = [](const ggml_tensor * t) { return t && strstr(t->name, "Vcur"); };

                            const ggml_tensor * n_k;
                            const ggml_tensor * n_v;
                            (void)n_v; // used in GGMLHEXAGON_LOG_DEBUG
                            const hex_op_desc * op_k;
                            const hex_op_desc * op_v;
                            if (is_k(n1)) {
                                // Q, K, V order (Gemma4, Llama3)
                                n_k = n1; op_k = &next1;
                                n_v = n2; op_v = &next2;
                            } else if (is_v(n1)) {
                                // Q, V, K order (Qwen3)
                                n_k = n2; op_k = &next2;
                                n_v = n1; op_v = &next1;
                            } else {
                                // Fallback: assume Q, K, V order
                                n_k = n1; op_k = &next1;
                                n_v = n2; op_v = &next2;
                            }

                            struct htp_mm_kernel_params kparams;
                            ggml_hexagon_precompute_fused_qkv_params(ctx, n_k->src[0], n_k->src[1], &kparams);
                            if ((size_t)kparams.vtcm_size <= vtcm_budget) {
                                int32_t wq_idx  = op.src_idx[0];
                                int32_t x_idx   = op.src_idx[1];
                                int32_t q_dst   = op.dst_idx[0];
                                op.htp_opcode   = HTP_OP_MUL_MAT_QKV;
                                op.src_idx[0]   = op_k->src_idx[0];  // Wk
                                op.src_idx[1]   = x_idx;             // x (shared)
                                op.src_idx[2]   = op_v->src_idx[0];  // Wv
                                op.src_idx[3]   = wq_idx;            // Wq
                                op.dst_idx[0]   = op_k->dst_idx[0];  // K
                                op.dst_idx[1]   = op_v->dst_idx[0];  // V
                                op.dst_idx[2]   = q_dst;             // Q
                                op.dst_idx[3]   = -1;
                                memcpy(op.kernel_params, &kparams, sizeof(kparams));
                                fused_ops.push_back(op);
                                i += 2;
                                n_mul_mat_qkv++;
                                ctx->n_fused_qkv_cum++;
                                GGMLHEXAGON_LOG_DEBUG("DBG QKV fusion: q=%s k=%s v=%s | Wq[t%d] Wk[t%d] Wv[t%d] x[t%d] | Q[t%d] K[t%d] V[t%d]",
                                                         n_q->name ? n_q->name : "?",
                                                         n_k->name ? n_k->name : "?",
                                                         n_v->name ? n_v->name : "?",
                                                         wq_idx, op_k->src_idx[0], op_v->src_idx[0], x_idx,
                                                         q_dst, op_k->dst_idx[0], op_v->dst_idx[0]);
                                continue;
                            } else {
                                GGMLHEXAGON_LOG_INFO("skip QKV fusion: VTCM needed (%d) > budget (%zu)", (int)kparams.vtcm_size, vtcm_budget);
                            }
                        }
                    }
                }

                // FFN fusion: 2 MUL_MAT (gate,up) -> HTP_OP_MUL_MAT_FFN.
                // Current op is gate, next is up.
                // src0=Wgate, src1=y, src2=Wup; dst[0]=gate, dst[1]=up.
                // Only triggers when is_mergeable_mul_mat returns true
                // (quantized src0 + F32 src1 + !mm_is_hmx_eligible).
                if (i + 1 < hex_ops.size()) {
                    const hex_op_desc & next = hex_ops[i + 1];
                    if (next.opcode == GGML_OP_MUL_MAT) {
                        const ggml_tensor * n_gate = tensor_src[op.dst_idx[0]];
                        const ggml_tensor * n_up   = tensor_src[next.dst_idx[0]];
                        if (is_mergeable_mul_mat_pair(ctx, n_gate, n_up)) {
                            struct htp_mm_kernel_params kparams;
                            ggml_hexagon_precompute_fused_ffn_params(ctx, n_gate->src[0], n_gate->src[1], &kparams);
                            if ((size_t)kparams.vtcm_size <= vtcm_budget) {
                                op.htp_opcode = HTP_OP_MUL_MAT_FFN;
                                // src0=Wgate (keep), src1=y (keep)
                                op.src_idx[2] = next.src_idx[0];   // Wup
                                op.src_idx[3] = -1;
                                // dst[0]=gate (keep)
                                op.dst_idx[1] = next.dst_idx[0];   // up
                                op.dst_idx[2] = -1;
                                op.dst_idx[3] = -1;
                                memcpy(op.kernel_params, &kparams, sizeof(kparams));
                                fused_ops.push_back(op);
                                i += 1;
                                n_mul_mat_ffn++;
                                ctx->n_fused_ffn_cum++;
                                GGMLHEXAGON_LOG_DEBUG("DBG FFN fusion: gate=%s up=%s | Wgate[t%d] y[t%d] Wup[t%d] | gate[t%d] up[t%d]",
                                                         n_gate->name ? n_gate->name : "?",
                                                         n_up->name ? n_up->name : "?",
                                                         op.src_idx[0], op.src_idx[1], next.src_idx[0],
                                                         op.dst_idx[0], next.dst_idx[0]);
                                continue;
                            } else {
                                GGMLHEXAGON_LOG_DEBUG("skip FFN fusion: VTCM needed (%d) > budget (%zu)",
                                                      (int)kparams.vtcm_size, vtcm_budget);
                            }
                        }
                    }
                }
            }

            // MUL_MAT + ADD -> MUL_MAT_ADD (bias add inside matmul kernel)
            // Only applies to pre-norm models where MUL_MAT (down_proj)
            // is immediately followed by residual ADD. Gemma uses post-norm
            // (MUL_MAT -> RMS_NORM -> MUL -> ADD), so this won't trigger there.
            //
            // Bias (src2) is read from DDR, not VTCM, so the kparams from
            // Phase 2 (MUL_MAT) are reusable. The VTCM budget check is
            // defensive: if the matmul already saturates VTCM as a plain
            // MUL_MAT, fusing into MUL_MAT_ADD cannot save anything; better
            // to keep it as 2 separate ops than risk silent overflow.
            if (op.opcode == GGML_OP_MUL_MAT && i + 1 < hex_ops.size()) {
                const hex_op_desc & next = hex_ops[i + 1];
                if (next.opcode == GGML_OP_ADD &&
                    (next.src_idx[0] == op.dst_idx[0] || next.src_idx[1] == op.dst_idx[0])) {
                    if (src_use_count[op.dst_idx[0]] != 1) {
                        n_mm_add_skip_use_count++;
                    } else {
                        int32_t bias_idx = -1;
                        if (next.src_idx[0] == op.dst_idx[0]) {
                            bias_idx = next.src_idx[1];
                        } else if (next.src_idx[1] == op.dst_idx[0]) {
                            bias_idx = next.src_idx[0];
                        }
                        if (bias_idx >= 0) {
                            const struct htp_mm_kernel_params * kparams_mm =
                                (const struct htp_mm_kernel_params *) op.kernel_params;
                            if ((size_t) kparams_mm->vtcm_size <= vtcm_budget) {
                                op.htp_opcode = HTP_OP_MUL_MAT_ADD;
                                op.src_idx[2]   = bias_idx;
                                op.dst_idx[0]    = next.dst_idx[0];
                                fused_ops.push_back(op);
                                i++;
                                n_mul_mat_add++;
                                ctx->n_fused_mm_add_cum++;
                                continue;
                            } else {
                                GGMLHEXAGON_LOG_INFO("skip MUL_MAT_ADD fusion: VTCM needed (%d) > budget (%zu)",
                                                      (int) kparams_mm->vtcm_size, vtcm_budget);
                                n_mm_add_skip_vtcm++; (void)n_mm_add_skip_vtcm;
                            }
                        }
                    }
                } else {
                    n_mm_add_skip_not_adjacent++;
                }
            }

            fused_ops.push_back(op);
        }

        if (n_rms_norm_mul + n_mul_mat_add + n_mul_mat_qkv + n_mul_mat_ffn > 0) {
            GGMLHEXAGON_LOG_DEBUG("op-fusion: %zu ops -> %zu ops (%zu RMS_NORM_MUL, %zu MUL_MAT_ADD, %zu MUL_MAT_QKV, %zu MUL_MAT_FFN)",
                                    hex_ops.size(), fused_ops.size(),
                                    n_rms_norm_mul, n_mul_mat_add, n_mul_mat_qkv, n_mul_mat_ffn);
            hex_ops = std::move(fused_ops);
        }
        if (n_mm_add_skip_use_count > 0 || n_mm_add_skip_not_adjacent > 0) {
            GGMLHEXAGON_LOG_DEBUG("mm_add fusion diag: skip_use_count=%zu skip_not_adjacent=%zu",
                                    n_mm_add_skip_use_count, n_mm_add_skip_not_adjacent);
        }
    }  // end if (!cache_hit) for Phase 3

    n_ops = (uint32_t)hex_ops.size();

    // ---- Cache save: store Phase 1/2/3 result keyed by content_hash ----
    // Only on miss. operator[] safely creates entry if absent; on hit we
    // already restored from cache, so skip the assign work entirely.
    if (!cache_hit) {
        auto & entry = ctx->cgraph_cache[content_hash];
        entry.content_hash = content_hash;
        entry.n_nodes   = cgraph->n_nodes;
        entry.n_tensors = (int)n_tensors;
        entry.n_ops     = (int)n_ops;
        entry.tensor_src.assign(tensor_src.begin(), tensor_src.end());
        entry.supported_nodes.assign(supported_nodes.begin(), supported_nodes.end());
        entry.hex_ops.assign(hex_ops.begin(), hex_ops.end());
        entry.is_weight.assign(is_weight.begin(), is_weight.end());
    }

    t_p3 = t_start; t_start = ggml_time_us(); ctx->cum_p3_us += t_start - t_p3;

    // ---- Phase 4: compute layout sizes ----
    const uint32_t hdr_size      = (uint32_t)sizeof(hex_batch_hdr);                 // ~24 bytes
    const uint32_t ops_region    = (uint32_t)(n_ops * sizeof(hex_op_desc));         // ~96*N
    const uint32_t tens_region   = (uint32_t)(n_tensors * sizeof(hex_tensor_desc)); // ~104*M
    // align ops/tensors regions
    const uint32_t ops_offset       = hdr_size;
    const uint32_t tensors_offset   = ops_offset + ((ops_region + HEX_OP_ALIGN - 1) & ~(HEX_OP_ALIGN - 1));
    const uint32_t total_desc_size  = tensors_offset + tens_region;

    t_p4 = t_start; t_start = ggml_time_us(); ctx->cum_p4_us += t_start - t_p4;

    // ---- Phase 5: handle heap tensors -> mirror into mempool ----
    int64_t t_prev = ggml_time_us();
    // Two-step approach:
    //   Step 1: Collect unique data pointers and compute max mirror size per buffer
    //   Step 2: Allocate one mirror per unique buffer (not per tensor)
    // This ensures: (a) shared buffers get one mirror with max size,
    //               (b) each tensor descriptor gets correct ne/nb.
    //
    // Cache coherency fix: for in-place ops (src0->data == dst->data), the
    // shared mirror causes Phase 9 DC CVAC to pollute the dst cache lines
    // with stale src0 data. After DSP writes the MUL result to DRAM, the CPU
    // cache still holds the old src0 data, so Phase 12 copy-back reads stale
    // data. Fix: allocate a separate dst mirror for in-place ops so that
    // Phase 9 only flushes the src0 mirror, and the dst mirror is never
    // flushed (CPU cache has no stale data for it).
    struct ion_mirror {
        int32_t  tensor_idx;
        void *   original_data;
        uint32_t mirror_offset;
        uint32_t data_len;
    };
    std::vector<ion_mirror> mirrors;

    // Step 1: Collect unique data pointers and max sizes
    struct buffer_mirror_info {
        uint32_t mirror_offset;
        uint32_t max_data_len;
        bool     allocated;
    };
    std::unordered_map<void *, buffer_mirror_info> buffer_mirrors_map;

    for (int32_t tidx = 0; tidx < (int32_t)n_tensors; tidx++) {
        ggml_tensor * t = tensor_src[tidx];
        if (!t->data) continue;

        const char * data_ptr = (const char *)t->data;
        if (data_ptr >= ion_base && data_ptr < ion_base + (ptrdiff_t)ion_size) {
            continue;  // already in mempool
        }

        uint32_t t_size = (uint32_t)ggml_nbytes(t);
        // For quantized weights repacked in-place by set_tensor,
        // the actual data on heap is the repacked (larger) layout.
        // The mirror must copy the full repacked data for DSP access.
        bool is_quant_weight = t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16;
        if (is_quant_weight) {
            size_t repacked = ggml_hexagon_repacked_size(t->type, t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
            if (repacked > 0) t_size = (uint32_t)repacked;
        }
        auto it = buffer_mirrors_map.find(t->data);
        if (it == buffer_mirrors_map.end()) {
            buffer_mirrors_map[t->data] = {0, t_size, false};
        } else if (t_size > it->second.max_data_len) {
            it->second.max_data_len = t_size;
        }
    }

    // Step 2: Allocate mirrors for each unique data pointer
    size_t data_limit = ion_size;
    for (auto & kv : buffer_mirrors_map) {
        void * data_ptr = kv.first;
        buffer_mirror_info & info = kv.second;
        size_t mirror_size = info.max_data_len;
        size_t aligned_offset = (ctx->rpc_mempool_usage + 127u) & ~127u;

        if (aligned_offset + mirror_size > data_limit) {
            GGMLHEXAGON_LOG_WARN("mempool-batch: mempool full for mirror (%zu bytes)", mirror_size);
            continue;
        }

        uint32_t moff = (uint32_t)aligned_offset;
        void * ion_buf = (char *)ctx->rpc_mempool + moff;
        ctx->rpc_mempool_usage = aligned_offset + mirror_size;

        // Record mirror as a temporary mempool region
        ion_pool_region mirror_region;
        mirror_region.offset = aligned_offset;
        mirror_region.size   = mirror_size;
        mirror_region.in_use = true;
        ctx->ion_regions.push_back(mirror_region);
        temp_region_indices.push_back(ctx->ion_regions.size() - 1);

        memcpy(ion_buf, data_ptr, mirror_size);

        info.mirror_offset = moff;
        info.allocated = true;

        GGMLHEXAGON_LOG_DEBUG("mempool-batch: mirror buffer %p -> mempool offset=0x%x (%u bytes)",
                              data_ptr, moff, info.max_data_len);
    }

    // Step 3: Build per-tensor mirror offset lookup and mirrors list for copy-back.
    local_mirror_offset.assign(n_tensors, -1);
    for (int32_t tidx = 0; tidx < (int32_t)n_tensors; tidx++) {
        ggml_tensor * t = tensor_src[tidx];
        if (!t->data) continue;

        const char * data_ptr = (const char *)t->data;
        if (data_ptr >= ion_base && data_ptr < ion_base + (ptrdiff_t)ion_size) {
            continue;  // already in mempool
        }

        auto it = buffer_mirrors_map.find(t->data);
        if (it == buffer_mirrors_map.end() || !it->second.allocated) continue;

        local_mirror_offset[tidx] = (int32_t)it->second.mirror_offset;

        ion_mirror m;
        m.tensor_idx    = tidx;
        m.original_data = t->data;
        m.mirror_offset = it->second.mirror_offset;
        m.data_len      = (uint32_t)ggml_nbytes(t);
        mirrors.push_back(m);
    }

    // ---- Phase 6: track mempool offsets for repacked quantized weights ----
    t_p5 = ggml_time_us() - t_prev; t_prev = ggml_time_us();

    //   weights are already repacked to tile-based layout
    //   by set_tensor during model loading. Phase 6 only tracks mempool
    //   offsets for DSP descriptor updates in Phase 8.
    std::vector<std::pair<uint32_t, uint32_t>> repacked_ion_weights; // (offset, length)
    static std::unordered_map<const void *, uint32_t> g_tiled_ion_offsets;
    {
        // Quantized weights (Q4_0 / Q4_1 / Q8_0 / IQ4_NL / MXFP4) are repacked
        // to tile-based (HMX) layout in set_tensor during model loading.
        // By the time graph_compute_batch runs, every quantized weight's
        // data at t->data is already in tiled layout, so Phase 6 does
        // NO repack work here.
        //
        // The only thing Phase 6 still needs to do is record the mempool offset
        // of each repacked weight in g_tiled_ion_offsets so Phase 10 can build
        // the DSP descriptor with the correct data_offset. Any quantized weight
        // that somehow lives outside the repack buft is logged as a one-shot
        // warning (should not happen with the current model loader) but its
        // mempool offset is still recorded if we can find one, so the DSP
        // descriptor remains well-formed.
        static std::unordered_set<const void *> s_warned_non_repack;
        for (uint32_t i = 0; i < n_tensors; i++) {
            ggml_tensor * t = tensor_src[i];
            if (!t || !t->data) continue;
            bool is_quant_weight = is_weight[i] && t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16;
            if (!is_quant_weight) continue;
            if (t->type != GGML_TYPE_Q4_0 && t->type != GGML_TYPE_Q4_1 &&
                t->type != GGML_TYPE_Q8_0 && t->type != GGML_TYPE_IQ4_NL &&
                t->type != GGML_TYPE_Q4_K && t->type != GGML_TYPE_Q6_K &&
                t->type != GGML_TYPE_MXFP4) continue;
            const int32_t K = t->ne[0];
            if (K % 32 != 0 || K <= 0) continue;

            if (!t->buffer || !ggml_backend_buffer_is_hexagon_repack(t->buffer)) {
                if (s_warned_non_repack.insert(t->data).second) {
                    GGMLHEXAGON_LOG_WARN("tiled: weight %s (data=%p) not in repack buft; "
                                         "assuming set_tensor already repacked it",
                                         t->name, t->data);
                }
            }

            if (g_tiled_ion_offsets.find(t->data) != g_tiled_ion_offsets.end()) {
                continue;  // already recorded on a prior graph_compute call
            }

            // Record the mempool offset for Phase 10. The data is in mempool either
            // directly (repack buft) or via the Phase 5 heap->mempool mirror.
            const char * dp = (const char *)t->data;
            if (dp >= ion_base && dp < ion_base + (ptrdiff_t)ion_size) {
                g_tiled_ion_offsets[t->data] = (uint32_t)(dp - ion_base);
            } else if (local_mirror_offset[i] >= 0) {
                g_tiled_ion_offsets[t->data] = (uint32_t)local_mirror_offset[i];
            }
        }
    }

    // ---- Phase 7: allocate batch descriptor region in mempool ----
    t_p6 = ggml_time_us() - t_prev; t_prev = ggml_time_us();
    size_t batch_align = HEX_BATCH_ALIGN;
    size_t batch_offset_raw = ctx->rpc_mempool_usage;
    size_t batch_offset_aligned = (batch_offset_raw + batch_align - 1) & ~(batch_align - 1);

    if (batch_offset_aligned + total_desc_size > data_limit) {
        GGMLHEXAGON_LOG_ERROR("mempool-batch: mempool full for batch desc (%zu bytes at offset %zu)",
                              total_desc_size, batch_offset_aligned);
        // Free temporary mirror regions before returning
        for (size_t ri : temp_region_indices) {
            ctx->ion_regions[ri].in_use = false;
        }
        return result;
    }

    uint32_t batch_offset = (uint32_t)batch_offset_aligned;
    ctx->rpc_mempool_usage = batch_offset_aligned + total_desc_size;
    // Record batch descriptor as a temporary mempool region
    ion_pool_region batch_region;
    batch_region.offset = batch_offset_aligned;
    batch_region.size   = total_desc_size;
    batch_region.in_use = true;
    ctx->ion_regions.push_back(batch_region);
    temp_region_indices.push_back(ctx->ion_regions.size() - 1);

    t_p7 = t_prev; t_prev = ggml_time_us(); ctx->cum_p7_us += t_prev - t_p7;

    // ---- Phase 8: build descriptors directly in mempool ----
    t_prev = ggml_time_us();
    uint8_t *ion_batch = (uint8_t *)ctx->rpc_mempool + batch_offset;
    hex_batch_hdr * hdr = (hex_batch_hdr *)ion_batch;
    memset(hdr, 0, sizeof(*hdr));
    hdr->n_ops         = n_ops;
    hdr->n_tensors    = n_tensors;
    hdr->ops_offset   = ops_offset;
    hdr->tensors_offset = tensors_offset;
    hdr->total_size   = total_desc_size;

    // write op descriptors
    hex_op_desc * ops_out = (hex_op_desc *)(ion_batch + ops_offset);
    memcpy(ops_out, hex_ops.data(), ops_region);

    // write tensor descriptors with computed offsets
    hex_tensor_desc * tens_out = (hex_tensor_desc *)(ion_batch + tensors_offset);
    for (uint32_t i = 0; i < n_tensors; i++) {
        ggml_tensor * t = tensor_src[i];
        hex_tensor_desc * td = &tens_out[i];

        // weights may be stored in a different format (see set_tensor);
        // the DSP only knows the storage-type kernels
        td->type = (int32_t)ggml_hexagon_weight_dsp_type(t->type);
        td->ne[0] = (int32_t)t->ne[0]; td->ne[1] = (int32_t)t->ne[1];
        td->ne[2] = (int32_t)t->ne[2]; td->ne[3] = (int32_t)t->ne[3];
        td->nb[0] = (int32_t)t->nb[0]; td->nb[1] = (int32_t)t->nb[1];
        td->nb[2] = (int32_t)t->nb[2]; td->nb[3] = (int32_t)t->nb[3];
        memcpy(td->op_params, t->op_params, sizeof(td->op_params));
        td->data_len = (uint32_t)ggml_nbytes(t);

        const char * data_ptr = (const char *)t->data;
        if (data_ptr >= ion_base && data_ptr < ion_base + (ptrdiff_t)ion_size) {
            // mempool tensor: direct offset
            td->data_offset = (uint32_t)(data_ptr - ion_base);
            td->flags = is_weight[i] ? 2 : 0;  // 2=weight (skip cache flush)
        } else {
            // heap tensor: look up mempool offset
            int32_t moff = local_mirror_offset[i];
            if (moff >= 0) {
                td->data_offset = (uint32_t)moff;
                td->flags = 1;  // writable (mirrored)
            } else {
                td->data_offset = 0;
                td->flags = 0;
                GGMLHEXAGON_LOG_WARN("mempool-batch: tensor[%d] is non-mempool heap without mirror!", i);
            }
        }

        // tiled: update descriptor to match tile-based repacked layout
        // (repack done in set_tensor during model loading via repack buffer type)
        bool is_quant_weight = is_weight[i] && t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16;
        if (is_quant_weight) {
            auto it = g_tiled_ion_offsets.find(t->data);
            if (it != g_tiled_ion_offsets.end()) {
                const int32_t ne0_p = (int32_t)hex_round_up((uint32_t)t->ne[0], 32);
                const int32_t ne1_p = (int32_t)hex_round_up((uint32_t)t->ne[1], 32);
                td->ne[0] = ne0_p;
                td->ne[1] = ne1_p;
                // nb[1] is used by DSP DMA as weight_stride = weight + nc * nb[1].
                // In the tiled layout, tiles are stored as (ne1_p/32) x (ne0_p/32)
                // tiles of tile_size bytes each. The byte offset to column nc is
                // (nc/32) * (ne0_p/32) * tile_size, so nb[1] = (ne0_p/32) * tile_size / 32.
                // ggml_row_size gives the original (non-tiled) stride, which is wrong here.
                td->nb[1] = (int32_t)((ne0_p / 32) * htp_mm_get_weight_tile_size((int)ggml_hexagon_weight_dsp_type(t->type)) / 32);
                td->nb[2] = td->nb[1] * ne1_p;
                td->nb[3] = td->nb[2] * (int32_t)t->ne[2];
                td->data_len = (uint32_t)ggml_hexagon_repacked_size(t->type, t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
                td->data_offset = it->second;
                td->flags = 2;  // weight (skip cache flush after first-touch)
            }
        }
    }

    GGMLHEXAGON_LOG_DEBUG("mempool-batch: submitted offset=0x%x size=%u (%u ops, %u tensors)", batch_offset, total_desc_size, n_ops, n_tensors);

    // ion_sync_mode controls which cache coherency mechanism to use:
    //   0 = both DC CVAC/CIVAC + DMA_BUF_IOCTL_SYNC (default, safest)
    //   1 = ion_sync only (skip manual DC CVAC/CIVAC, rely on kernel DMA_BUF_IOCTL_SYNC)
    //   2 = DC CVAC/CIVAC only (skip ion_sync, manual cache maintenance only)
    const bool do_dc_cvac  = (g_hexagon_appcfg.ion_sync_mode != 1);
    const bool do_ion_sync = (g_hexagon_appcfg.ion_sync_mode != 2);

    // ---- Phase 9: AP -> DSP cache coherency ----
    t_p8 = ggml_time_us() - t_prev; t_prev = ggml_time_us();

    // Flush CPU cache to DRAM so DSP can read AP-written data.
    {
        // Collect per-tensor dirty ranges and flush them individually (merged).
        // A single continuous [min, max] range would also flush the holes
        // between low-offset activations and high-offset weights (~940MB of
        // wasted cache-line writes when only a few MB are actually dirty).
        std::vector<std::pair<uint32_t, uint32_t>> ranges;
        ranges.reserve(n_tensors + mirrors.size() + repacked_ion_weights.size() + (size_t)cgraph->n_nodes + 4);

        // Diagnostic counters (per-call, reset every batch) so we can see which
        // source is dominating the flush cost. Logged once per call.
        uint64_t dbg_bytes_tensor = 0, dbg_bytes_mirror = 0;
        uint64_t dbg_bytes_repack_ion = 0, dbg_bytes_batch = 0, dbg_bytes_cgraph = 0;
        uint32_t dbg_ranges_tensor = 0, dbg_ranges_cgraph = 0;
        (void)dbg_bytes_tensor; (void)dbg_bytes_mirror;
        (void)dbg_bytes_repack_ion; (void)dbg_bytes_batch; (void)dbg_bytes_cgraph;
        (void)dbg_ranges_tensor; (void)dbg_ranges_cgraph;

        auto add_range = [&](uint32_t off, uint32_t len) {
            if (len > 0) ranges.push_back({off, off + len});
        };

        // ion_sync_mode=1 path: skip the per-tensor/cgraph range scans
        // entirely; the DMA_BUF_IOCTL_SYNC below handles cache coherency
        // for the whole mempool. The scan work is pure overhead in this
        // mode (the collected ranges are never used to drive a DC CVAC).
        if (!do_dc_cvac) {
            if (do_ion_sync) {
                int ion_fd = ctx->rpc_mempool_handle;
                if (ion_fd > 0) ion_sync_for_direction(ion_fd, 1);
            }
            int was_weights_dirty = ctx->weights_dirty ? 1 : 0;
            ctx->weights_dirty = false;
            GGMLHEXAGON_LOG_WARN("mempool-batch: phase9 skipped (ion_sync_mode=%d) dirty=%d",
                                  g_hexagon_appcfg.ion_sync_mode, was_weights_dirty);
        } else {

            for (uint32_t i = 0; i < n_tensors; i++) {
                ggml_tensor * t = tensor_src[i];
                if (!t || !t->data) continue;
                if (is_weight[i] && !ctx->weights_dirty) continue;
                const char * dp = (const char *)t->data;
                if (dp >= ion_base && dp < ion_base + (ptrdiff_t)ion_size) {
                    // For quantized weights repacked in-place by set_tensor,
                    // use the repacked size (larger than ggml_nbytes).
                    bool is_quant_weight = is_weight[i] && t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16;
                    size_t flush_size = is_quant_weight ? ggml_hexagon_repacked_size(t->type, t->ne[0], t->ne[1], t->ne[2], t->ne[3]) : ggml_nbytes(t);
                    if (flush_size == 0) flush_size = ggml_nbytes(t);
                    add_range((uint32_t)(dp - ion_base), (uint32_t)flush_size);
                    dbg_bytes_tensor += flush_size;
                    dbg_ranges_tensor++;
                }
            }
            for (const auto & m : mirrors) {
                add_range(m.mirror_offset, m.data_len);
                dbg_bytes_mirror += m.data_len;
            }
            for (const auto & rw : repacked_ion_weights) {
                add_range(rw.first, rw.second);
                dbg_bytes_repack_ion += rw.second;
            }
            add_range(batch_offset, total_desc_size);
            dbg_bytes_batch += total_desc_size;

            // Also flush non-op tensors in cgraph not in tensor_src (e.g., test sentinels).
            // Without this, Phase 11 DC CIVAC can invalidate cache lines containing
            // unflushed sentinel data, causing sentinel mismatch.
            //
            // Skip repack-buft weights when weights_dirty is false.
            // The per-tensor loop above already guards against re-flushing clean
            // weights via weight_indices; this loop did not, which caused every
            // graph_compute call to re-flush the entire repack weight region
            // even though no repack had happened.
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * t = cgraph->nodes[i];
                if (!t || !t->data) continue;
                if (!ctx->weights_dirty &&
                    t->buffer && ggml_backend_buffer_is_hexagon_repack(t->buffer)) {
                    continue;  // repack-buft weight, cache already coherent
                }
                const char * dp = (const char *)t->data;
                if (dp >= ion_base && dp < ion_base + (ptrdiff_t)ion_size) {
                    size_t sz = ggml_nbytes(t);
                    add_range((uint32_t)(dp - ion_base), (uint32_t)sz);
                    dbg_bytes_cgraph += sz;
                    dbg_ranges_cgraph++;
                }
            }

            uint32_t flush_bytes = 0;
            uint32_t n_flush     = 0;
            (void)flush_bytes; (void)n_flush;
            if (do_dc_cvac && !ranges.empty()) {
                std::sort(ranges.begin(), ranges.end());
                // Merge overlapping/adjacent ranges. Merge gap = 1 cache line (64B):
                // flushing a tiny gap is cheaper than issuing a second flush call.
                const uint32_t merge_gap = 64;
                uint32_t cur_start = ranges[0].first;
                uint32_t cur_end   = ranges[0].second;
                for (size_t i = 1; i < ranges.size(); i++) {
                    if (ranges[i].first <= cur_end + merge_gap) {
                        if (ranges[i].second > cur_end) cur_end = ranges[i].second;
                    } else {
                        cpu_dcache_flush_range(ctx, 0,
                            (char *)ctx->rpc_mempool + cur_start, cur_end - cur_start);
                        flush_bytes += cur_end - cur_start;
                        n_flush++;
                        cur_start = ranges[i].first;
                        cur_end   = ranges[i].second;
                    }
                }
                cpu_dcache_flush_range(ctx, 0,
                    (char *)ctx->rpc_mempool + cur_start, cur_end - cur_start);
                flush_bytes += cur_end - cur_start;
                n_flush++;
            }
            (void)flush_bytes; (void)n_flush;
            (void)dbg_bytes_tensor; (void)dbg_bytes_mirror;
            (void)dbg_bytes_repack_ion; (void)dbg_bytes_batch; (void)dbg_bytes_cgraph;
            (void)dbg_ranges_tensor; (void)dbg_ranges_cgraph;
            // Also try DMA_BUF_IOCTL_SYNC as extra safeguard
            if (do_ion_sync) {
                int ion_fd = ctx->rpc_mempool_handle;
                if (ion_fd > 0) ion_sync_for_direction(ion_fd, 1);
            }

            int was_weights_dirty = ctx->weights_dirty ? 1 : 0;
            ctx->weights_dirty = false;
            (void)was_weights_dirty;
        }  // end else (do_dc_cvac)
    }

    // ---- Phase 10: FastRPC doorbell call (only 2 scalars!) ----
    // 3-way split for fine-grained perf:
    //   rpc_setup: AP-side work between Phase 9 end and invoke() entry
    //   dsp_exec:  the synchronous invoke() call itself (RPC round-trip
    //              + DSP-side work + DSP->AP reply)
    //   civac:     AP-side cache invalidate after invoke() returns
    //               (measured in Phase 11 below, accumulated into cum_p10_civac_us)
    t_p9 = ggml_time_us() - t_prev; t_prev = ggml_time_us();
    ctx->rpc_batch_call_count++;

    int64_t t_p10_pre = ggml_time_us();
    int hexagon_error = ggml_dsp_execute_batch(ctx->ggmlop_handle, batch_offset, total_desc_size);
    int64_t t_p10_post = ggml_time_us();

    if (AEE_SUCCESS != hexagon_error) {
        GGMLHEXAGON_LOG_WARN("ggml_dsp_execute_batch failed: 0x%x", hexagon_error);
    }

    // t_p10 captures the entire synchronous invoke (== old p10 minus civac)
    t_p10 = t_p10_post - t_p10_pre;
    ctx->cum_p10_us += t_p10;
    ctx->cum_p10_dsp_exec_us  += t_p10;
    // rpc_setup = AP-side cost between Phase 9 end and the invoke entry
    int64_t p10_rpc_setup = t_p10_pre - t_prev;
    ctx->cum_p10_rpc_setup_us  += p10_rpc_setup;
    t_prev = ggml_time_us();

    // ---- Phase 11: invalidate CPU cache for DSP-written mempool regions ----
    // civac is now tracked separately via t_civac so the AP-side cache-coherency
    // cost is broken out from p10 (sync invoke) and from p11 (verify+copy-back).
    int64_t t_civac = ggml_time_us();  // civac start
    // DSP writes results to DRAM via mempool buffer, but CPU cache may still hold
    // stale data.  DC CIVAC + ion_sync controlled by ion_sync_mode (see Phase 9).
    if (hexagon_error == AEE_SUCCESS) {
        if (!do_dc_cvac) {
            // ion_sync_mode=1: rely solely on DMA_BUF_IOCTL_SYNC.
            if (do_ion_sync) {
                int ion_fd = ctx->rpc_mempool_handle;
                if (ion_fd > 0) ion_sync_for_direction(ion_fd, 0);
            }
        } else {
            uint32_t inval_min = ~0u, inval_max = 0;
            for (uint32_t oi = 0; oi < n_ops; oi++) {
                const hex_op_desc & cur_op = hex_ops[oi];
                uint32_t dst_idx = cur_op.dst_idx[0];
                if (dst_idx >= n_tensors) continue;
                ggml_tensor * dst_t = tensor_src[dst_idx];
                if (!dst_t || !dst_t->data) continue;

                uint32_t dst_off = 0xFFFFFFFFu;
                const char * dp = (const char *)dst_t->data;
                if (dp >= ion_base && dp < ion_base + (ptrdiff_t)ion_size) {
                    dst_off = (uint32_t)(dp - ion_base);
                } else {
                    int32_t moff = local_mirror_offset[dst_idx];
                    if (moff >= 0) dst_off = (uint32_t)moff;
                }
                if (dst_off == 0xFFFFFFFFu) continue;

                uint32_t dst_len = (uint32_t)ggml_nbytes(dst_t);
                uint32_t start = dst_off & ~63u;
                uint32_t end   = (dst_off + dst_len + 63u) & ~63u;
                if (start < inval_min) inval_min = start;
                if (end > inval_max) inval_max = end;
            }
            if (inval_max > inval_min) {
                cpu_dcache_inval_range(ctx, 0, (const char *)ctx->rpc_mempool + inval_min, inval_max - inval_min);
                GGMLHEXAGON_LOG_DEBUG("mempool-batch: phase11 DC CIVAC [0x%x, 0x%x] (%u bytes)",
                                      inval_min, inval_max, inval_max - inval_min);
            }
            // Also try DMA_BUF_IOCTL_SYNC as extra safeguard
            if (do_ion_sync) {
                int ion_fd = ctx->rpc_mempool_handle;
                if (ion_fd > 0) ion_sync_for_direction(ion_fd, 0);
            }
        }  // end else (do_dc_cvac)
    }

    // record civac time (Phase 11 only). Accumulated into cum_p10_civac_us.
    {
        int64_t civac_us = ggml_time_us() - t_civac;
        ctx->cum_p10_civac_us  += civac_us;
    }

    // Reset bump pointer so next graph_compute reuses the same mempool region.
    // Without this, rpc_mempool_usage only grows and eventually exhausts the pool,
    // causing mirror alloc failure (data_offset=0 -> DSP corrupts model weights).
    ctx->rpc_mempool_usage = saved_mempool_usage;

    // ---- Phase 12: copy-back mirrored results to heap ----
    t_p11 = ggml_time_us() - t_prev; t_prev = ggml_time_us();
    ctx->cum_p11_us += t_p11;
    if (hexagon_error == AEE_SUCCESS && !mirrors.empty()) {
        std::unordered_map<void *, std::pair<uint32_t, uint32_t>> copyback_map;
        for (const auto & m : mirrors) {
            auto it = copyback_map.find(m.original_data);
            if (it == copyback_map.end()) {
                copyback_map[m.original_data] = {m.mirror_offset, m.data_len};
            } else {
                if (m.data_len > it->second.second) {
                    it->second.second = m.data_len;
                }
            }
        }
        for (const auto & kv : copyback_map) {
            void * orig_data = kv.first;
            uint32_t moff = kv.second.first;
            uint32_t max_len = kv.second.second;
            memcpy(orig_data, (const char *)ctx->rpc_mempool + moff, max_len);
        }
    }

    // Free temporary mempool regions (mirrors, batch descriptors, repacked weights).
    // These are only needed during this graph_compute call and can be reused
    // in the next call. Mark them as free (no tail compaction).
    for (size_t ri : temp_region_indices) {
        ctx->ion_regions[ri].in_use = false;
    }

    t_p12 = ggml_time_us() - t_prev;

    int64_t end_time = ggml_time_us();
    int64_t graph_dur = end_time - begin_time;
    // (cum_p10_us is already updated at the end of the Phase 10 invoke()
    //  block; do not add t_p10 a second time here.)

    // compute unaccounted time (wall-clock not covered by phase timers above)
    // Use entry snapshots so the subtraction yields this call's contribution only.
    {
        int64_t accounted_this_call = (ctx->cum_p1_us  - snap_p1)
                                    + (ctx->cum_p2_us  - snap_p2)
                                    + (ctx->cum_p3_us - snap_p3)
                                    + (ctx->cum_p4_us  - snap_p4)
                                    + (ctx->cum_p7_us  - snap_p7)
                                    + (ctx->cum_p10_us - snap_p10)
                                    + t_p5 + t_p6 + t_p8 + t_p9 + t_p11 + t_p12;
        int64_t unaccounted = graph_dur - accounted_this_call;
        if (unaccounted < 0) unaccounted = 0;  // guard against measurement noise
        ctx->cum_unaccounted_us += unaccounted;
    }

    // update cumulative stats
    ctx->cumulative_graph_us += graph_dur;
    ctx->last_graph_end_us   = end_time;

    // per-phase cumulative time (cum_p11_us / cum_p10_civac_us already
    //  accumulated at the end of their respective phase; p5..p12 + p9 still
    //  use the trailing accumulator pattern.)
    ctx->cum_p5_us  += t_p5;
    ctx->cum_p6_us += t_p6;
    ctx->cum_p8_us  += t_p8;
    ctx->cum_p9_us += t_p9;
    ctx->cum_p12_us  += t_p12;
    // per-call min/max
    if (ctx->min_p10_us == 0 || t_p10 < ctx->min_p10_us)    ctx->min_p10_us = t_p10;
    if (t_p10 > ctx->max_p10_us)                           ctx->max_p10_us = t_p10;
    if (ctx->min_n_ops_per_call == 0 || n_ops < ctx->min_n_ops_per_call) {
        ctx->min_n_ops_per_call = n_ops;
    }
    if (n_ops > ctx->max_n_ops_per_call) {
        ctx->max_n_ops_per_call = n_ops;
    }

    {
        int64_t rpc_overhead = graph_dur - t_p10;
        if (rpc_overhead < 0) rpc_overhead = 0;
        if (ctx->min_rpc_overhead_us == 0 || rpc_overhead < ctx->min_rpc_overhead_us) {
            ctx->min_rpc_overhead_us = rpc_overhead;
        }
        if (rpc_overhead > ctx->max_rpc_overhead_us) {
            ctx->max_rpc_overhead_us = rpc_overhead;
        }
        ctx->sum_rpc_overhead_us += rpc_overhead;
    }

    if (ctx->min_graph_us == 0 || graph_dur < ctx->min_graph_us) ctx->min_graph_us = graph_dur;

    if (graph_dur > ctx->max_graph_us) {
        ctx->max_graph_us     = graph_dur;
        ctx->max_graph_n_nodes = graph_n_nodes;
        ctx->max_graph_n_ops   = n_ops;
        GGMLHEXAGON_LOG_DEBUG("new max graph_dur=%lld us (n_nodes=%u n_ops=%u p10=%lld p9=%lld p11=%lld)",
                              (long long)graph_dur, graph_n_nodes, n_ops,
                              (long long)t_p10, (long long)t_p9, (long long)t_p11);
    }
    GGMLHEXAGON_LOG_DEBUG("mempool-batch timing: p5=%lld p6=%lld p8=%lld p9=%lld p10=%lld p11=%lld p12=%lld (us) ops=%u",
                          (long long)t_p5, (long long)t_p6, (long long)t_p8, (long long)t_p9,
                          (long long)t_p10, (long long)t_p11, (long long)t_p12, n_ops);
    GGMLHEXAGON_LOG_DEBUG("graph n_ops   %u", n_ops);
    GGMLHEXAGON_LOG_DEBUG("graph inference duration %lld microseconds (gap_from_prev=%lld us)", (long long)graph_dur, (long long)gap_from_prev);
    GGMLHEXAGON_LOG_DEBUG("rpc stats: batch_calls=%llu cum_p10=%lld us cum_graph=%lld us avg_p10=%lld us avg_graph=%lld us",
                          (unsigned long long)ctx->rpc_batch_call_count,
                          (long long)ctx->cum_p10_us, (long long)ctx->cumulative_graph_us,
                          ctx->rpc_batch_call_count ? (long long)(ctx->cum_p10_us / (int64_t)ctx->rpc_batch_call_count) : 0,
                          ctx->rpc_batch_call_count ? (long long)(ctx->cumulative_graph_us / (int64_t)ctx->rpc_batch_call_count) : 0);

    return result;
}

// Reorder cgraph nodes to improve DSP VTCM cache locality.
// Stack MUL_MAT ops sharing the same src1 (input activation) so the DSP can
// reuse VTCM-resident dynamically quantized src1 across consecutive matmuls.
//
// Fusion pairs recognized by Phase 3 inline fusion in graph_compute_batch
// (RMS_NORM+MUL, MUL_MAT+ADD) are kept adjacent so the inline fusion still
// triggers. Only independent MUL_MAT groups (single node, quantized src0) are
// eligible for reordering.
static void ggml_backend_hexagon_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * gf) {
    GGML_ASSERT(backend);
    GGML_ASSERT(gf);

    const int n = gf->n_nodes;
    if (n < 2) {
        return;
    }

    // Step 1: mark fusion pairs (Phase 3 patterns). Nodes sharing group_id
    // must stay adjacent and in order so Phase 3 can still detect (i, i+1).
    std::vector<int> group_id(n, -1);
    int next_group = 0;
    int n_mm_add_groups = 0;  // count of MUL_MAT+ADD fusion pairs found
    int n_rms_mul_groups = 0; // count of RMS_NORM+MUL fusion pairs found
    (void)n_mm_add_groups; (void)n_rms_mul_groups; // used in GGMLHEXAGON_LOG_DEBUG
    for (int i = 0; i < n; i++) {
        if (group_id[i] != -1) {
            continue;
        }
        struct ggml_tensor * node = gf->nodes[i];

        if (node->op == GGML_OP_RMS_NORM && i + 1 < n) {
            struct ggml_tensor * next = gf->nodes[i + 1];
            if (next->op == GGML_OP_MUL && next->src[0] == node) {
                group_id[i]     = next_group;
                group_id[i + 1] = next_group;
                next_group++;
                n_rms_mul_groups++;
                i++;
                continue;
            }
        }

        if (node->op == GGML_OP_MUL_MAT && i + 1 < n) {
            struct ggml_tensor * next = gf->nodes[i + 1];
            if (next->op == GGML_OP_ADD &&
                (next->src[0] == node || next->src[1] == node)) {
                group_id[i]     = next_group;
                group_id[i + 1] = next_group;
                next_group++;
                n_mm_add_groups++;
                i++;
                continue;
            }
        }
    }

    GGMLHEXAGON_LOG_DEBUG("graph_optimize: n_nodes=%d n_mm_add_groups=%d n_rms_mul_groups=%d",
                           n, n_mm_add_groups, n_rms_mul_groups);

    // Step 2: build group list (each group is 1 or 2 contiguous node indices).
    std::vector<std::vector<int>> groups;
    {
        std::vector<bool> visited(n, false);
        for (int i = 0; i < n; i++) {
            if (visited[i]) {
                continue;
            }
            std::vector<int> g;
            if (group_id[i] != -1) {
                for (int j = i; j < n; j++) {
                    if (group_id[j] == group_id[i]) {
                        g.push_back(j);
                        visited[j] = true;
                    }
                }
            } else {
                g.push_back(i);
                visited[i] = true;
            }
            groups.push_back(std::move(g));
        }
    }

    // Step 3: reorder. Move stackable MUL_MAT groups with the same src1 close
    // together via a forward 16-group window. Non-stackable groups stay put.
    auto is_stackable_mul_mat = [](const struct ggml_tensor * node) -> bool {
        if (node == nullptr) {
            return false;
        }
        if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) {
            return false;
        }
        return node->src[0] && ggml_is_quantized(node->src[0]->type);
    };

    auto same_src1 = [](const struct ggml_tensor * a, const struct ggml_tensor * b) -> bool {
        return a->src[1] != nullptr && a->src[1] == b->src[1];
    };

    std::vector<int> new_node_order;
    new_node_order.reserve(n);
    std::vector<bool> group_used(groups.size(), false);
    constexpr int N_FORWARD = 16;

    for (size_t g0 = 0; g0 < groups.size(); g0++) {
        if (group_used[g0]) {
            continue;
        }
        group_used[g0] = true;
        for (int idx : groups[g0]) {
            new_node_order.push_back(idx);
        }

        if (groups[g0].size() != 1) {
            continue;
        }
        const struct ggml_tensor * node0 = gf->nodes[groups[g0][0]];
        if (!is_stackable_mul_mat(node0)) {
            continue;
        }

        for (size_t g1 = g0 + 1; g1 < groups.size() && g1 <= g0 + N_FORWARD; g1++) {
            if (group_used[g1] || groups[g1].size() != 1) {
                continue;
            }
            const struct ggml_tensor * node1 = gf->nodes[groups[g1][0]];
            if (!is_stackable_mul_mat(node1) || !same_src1(node0, node1)) {
                continue;
            }
            group_used[g1] = true;
            for (int idx : groups[g1]) {
                new_node_order.push_back(idx);
            }
        }
    }

    // Step 4: write back reordered nodes. Only order changes; tensor pointers
    // remain valid, so all src/dst links stay intact.
    std::vector<struct ggml_tensor *> new_nodes(n);
    for (int i = 0; i < n; i++) {
        new_nodes[i] = gf->nodes[new_node_order[i]];
    }
    for (int i = 0; i < n; i++) {
        gf->nodes[i] = new_nodes[i];
    }
}

static const char * ggml_backend_hexagon_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "Hexagon-cDSP";
}

static const char * ggml_backend_hexagon_device_get_name(ggml_backend_dev_t dev) {
    struct ggml_backend_hexagon_context * ctx = static_cast<ggml_backend_hexagon_context *>(dev->context);
    if (nullptr == ctx) {
        GGMLHEXAGON_LOG_ALWAYS("pls check why ctx is null");
        return "unknown";
    }
    return ctx->name;
}

static void ggml_backend_hexagon_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    struct ggml_backend_hexagon_context * ctx = static_cast<ggml_backend_hexagon_context *>(dev->context);
    if ((nullptr == ctx) || (ctx->device >= GGML_HEXAGON_MAX_DEVICES)) {
        GGMLHEXAGON_LOG_ALWAYS("pls check params");
        *free = 0;
        *total = 0;
        return;
    }

    GGMLHEXAGON_LOG_WARN("get_memory: enter device=%d domain_id=%d", ctx->device, ctx->domain_id);

    // ggml backend has domain_id == -1 (not a real CDSP PD)
    if (-1 == ctx->domain_id) {
        *total = ggmlhexagon_get_system_total_memory_in_bytes();
        *free = ggmlhexagon_get_system_free_memory_in_bytes();
    } else {
        size_t rpc_ion_memsize = 0;
        size_t rpc_ion_usage   = 0;
        rpc_ion_memsize = ctx->rpc_mempool_capacity;
        rpc_ion_usage   = ctx->rpc_mempool_usage;
        *total = rpc_ion_memsize;
        *free = (rpc_ion_memsize - rpc_ion_usage);
        GGMLHEXAGON_LOG_WARN("get_memory: device %d, rpc memsize %d MiB, usage %d MiB, free %d MiB",
                             ctx->device, rpc_ion_memsize / SIZE_IN_MB,
                             rpc_ion_usage / SIZE_IN_MB, (rpc_ion_memsize - rpc_ion_usage) / SIZE_IN_MB);
    }
}

static enum ggml_backend_dev_type ggml_backend_hexagon_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_hexagon_device_get_props(ggml_backend_dev_t dev,
                                              struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_hexagon_device_get_name(dev);
    props->description = ggml_backend_hexagon_device_get_description(dev);
    props->type        = ggml_backend_hexagon_device_get_type(dev);
    props->device_id   = nullptr;  // no PCI bus id for Hexagon CDSP devices
    ggml_backend_hexagon_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
            /* .async                 = */ false,
            /* .host_buffer           = */ false,
            /* .buffer_from_host_ptr  = */ false,
            /* .events                = */ false,
    };
}

static ggml_backend_i ggml_backend_hexagon_interface = {
        /* .get_name                = */ ggml_backend_hexagon_name,
        /* .free                    = */ ggml_backend_hexagon_free,
        /* .set_tensor_async        = */ nullptr,
        /* .get_tensor_async        = */ nullptr,
        /* .set_tensor_2d_async     = */ nullptr,
        /* .get_tensor_2d_async     = */ nullptr,
        /* .cpy_tensor_async        = */ nullptr,
        /* .synchronize             = */ nullptr,
        /* .graph_plan_create       = */ nullptr,
        /* .graph_plan_free         = */ nullptr,
        /* .graph_plan_update       = */ nullptr,
        /* .graph_plan_compute      = */ nullptr,
        /* .graph_compute           = */ ggmlhexagon_backend_graph_compute_batch,
        /* .event_record            = */ nullptr,
        /* .event_wait              = */ nullptr,
        /* .graph_optimize          = */ nullptr,
};

//FIXME: this guid does not make sense
static ggml_guid_t ggml_backend_hexagon_guid() {
    static ggml_guid guid = {
            0x1a, 0x2b, 0x3c, 0x4d, 0x5e, 0x6f, 0x70, 0x81,
            0x92, 0xa3, 0xb4, 0xc5, 0xd6, 0xe7, 0xf8, 0x09
    };
    return &guid;
}

struct ggml_backend_hexagon_reg_context {
    std::vector<ggml_backend_dev_t> devices;
    ~ggml_backend_hexagon_reg_context() {
        for (auto * dev : devices) {
            // context persists across inferences (not freed in ggml_backend_hexagon_free),
            // so it is always present and must be cleaned up here.
            if (dev->context) {
                auto * hctx = static_cast<ggml_backend_hexagon_context *>(dev->context);
                delete hctx;
            }
            delete dev;
        }
    }
};

// Lazily create the context (DSP session + mempool) if it doesn't exist yet.
// The ggml framework calls get_buffer_type / supports_buft BEFORE init_backend
// during model loading, so the context must exist by then.
// Called from get_buffer_type, get_repack_buffer_type, supports_buft, and
// device_init_backend. Context persists across inferences (ggml_backend_hexagon_free
// only deletes the backend, not the context) and is freed during registry shutdown.
static ggml_backend_hexagon_context * ggml_backend_hexagon_ensure_context(ggml_backend_dev_t dev) {
    if (nullptr != dev && nullptr != dev->context) {
        return (ggml_backend_hexagon_context *)dev->context;
    }

    ggmlhexagon_load_cfg();
    if (!ggmlhexagon_check_valid_appcfg()) {
        return nullptr;
    }

    // Find dev_index by matching dev in the registry
    int dev_index = 0;
    if (nullptr != dev) {
        auto * reg_ctx = (ggml_backend_hexagon_reg_context *)g_reg_ctx;
        if (reg_ctx) {
            for (size_t i = 0; i < reg_ctx->devices.size(); i++) {
                if (reg_ctx->devices[i] == dev) {
                    dev_index = (int)i;
                    break;
                }
            }
        }
    }
    if (dev_index >= GGML_HEXAGON_MAX_DEVICES) {
        GGMLHEXAGON_LOG_ERROR("invalid dev_index %d", dev_index);
        return nullptr;
    }

    GGMLHEXAGON_LOG_ALWAYS("creating context for dev_index=%d", dev_index);
    auto * ctx = new ggml_backend_hexagon_context(dev_index, dev);
    GGML_ASSERT(0 != ctx->ggmlop_handle);
    if (nullptr != dev) {
        dev->context = ctx;
    }
    return ctx;
}

static ggml_backend_t ggml_backend_hexagon_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggmlhexagon_load_cfg();
    if (!ggmlhexagon_check_valid_appcfg()) {
        return nullptr;
    }

    // Get the device from registry if not provided
    if (nullptr == dev) {
        int dev_index = 0;
        if (nullptr != params) {
            dev_index = (int)(intptr_t)params;
            if (dev_index < 0) dev_index = 0;
        }
        auto * reg_ctx = (ggml_backend_hexagon_reg_context *)g_reg_ctx;
        if (reg_ctx && dev_index < (int)reg_ctx->devices.size()) {
            dev = reg_ctx->devices[dev_index];
        }
    }

    // Ensure context exists (may have been created by get_buffer_type)
    auto * ctx = ggml_backend_hexagon_ensure_context(dev);
    if (nullptr == ctx) {
        GGMLHEXAGON_LOG_ERROR("failed to create context");
        return nullptr;
    }

    // If backend already exists for this context, return it
    if (nullptr != ctx->backend) {
        GGMLHEXAGON_LOG_ALWAYS("backend already exists for device %d, reusing", ctx->device);
        return ctx->backend;
    }

    ggml_backend_hexagon_interface.graph_optimize =
        g_hexagon_appcfg.enable_graph_optimize ? ggml_backend_hexagon_graph_optimize : nullptr;
    GGMLHEXAGON_LOG_ALWAYS("graph_optimize: %s", g_hexagon_appcfg.enable_graph_optimize ? "enabled" : "disabled");

    ggml_backend_t hexagon_backend = new ggml_backend{
            /* .guid      = */ ggml_backend_hexagon_guid(),
            /* .iface     = */ ggml_backend_hexagon_interface,
            /* .device    = */ dev,
            /* .context   = */ ctx
    };

    ctx->backend = hexagon_backend;
    return hexagon_backend;
}

static ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_hexagon_context * ctx = ggml_backend_hexagon_ensure_context(dev);
    GGMLHEXAGON_LOG_WARN("get_buffer_type: device=%d domain_id=%d buft=%p", ctx->device, ctx->domain_id, (void*)&ctx->buffer_type);
    return &ctx->buffer_type;
}

static ggml_backend_buffer_type_t ggml_backend_hexagon_device_get_repack_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_hexagon_context * ctx = ggml_backend_hexagon_ensure_context(dev);
    return &ctx->repack_buffer_type;
}

static ggml_backend_buffer_type_t * ggml_backend_hexagon_device_get_extra_buffers_type(ggml_backend_dev_t dev) {
    static ggml_backend_buffer_type_t bufts[2];
    bufts[0] = ggml_backend_hexagon_device_get_repack_buffer_type(dev);
    bufts[1] = NULL;
    return bufts;
}

static bool ggml_backend_hexagon_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (ggml_backend_buft_is_hexagon(buft) || ggml_backend_buft_is_hexagon_repack(buft)) {
        ggml_backend_hexagon_context * dev_ctx  = ggml_backend_hexagon_ensure_context(dev);
        ggml_backend_hexagon_context * buft_ctx = (ggml_backend_hexagon_context *)buft->context;
        return buft_ctx->device == dev_ctx->device;
    }
    return false;
}

static struct ggml_backend_device_i ggml_backend_hexagon_device_interface = {
        /* .get_name             = */ ggml_backend_hexagon_device_get_name,
        /* .get_description      = */ ggml_backend_hexagon_device_get_description,
        /* .get_memory           = */ ggml_backend_hexagon_device_get_memory,
        /* .get_type             = */ ggml_backend_hexagon_device_get_type,
        /* .get_props            = */ ggml_backend_hexagon_device_get_props,
        /* .init_backend         = */ ggml_backend_hexagon_device_init_backend,
        /* .get_buffer_type      = */ ggml_backend_hexagon_device_get_buffer_type,
        /* .get_host_buffer_type = */ nullptr,
        /* .buffer_from_host_ptr = */ nullptr,
        /* .supports_op          = */ ggmlhexagon_can_handle_op_through_cdsp,
        /* .supports_buft        = */ ggml_backend_hexagon_device_supports_buft,
        /* .offload_op           = */ nullptr,
        /* .event_new            = */ nullptr,
        /* .event_free           = */ nullptr,
        /* .event_synchronize    = */ nullptr,
};

bool ggml_backend_is_hexagon(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_hexagon_guid());
}

static void ggml_backend_hexagon_set_n_threads(ggml_backend_t backend, int n_threads) {
    GGML_ASSERT(ggml_backend_is_hexagon(backend));

    struct ggml_backend_hexagon_context * ctx = (struct ggml_backend_hexagon_context *)backend->context;
    // Prefer DSP-reported cap (set in ggmlhexagon_init_dsp via setclocks
    // out param, equals max_hw_threads - 2). Fall back to cfg value if
    // set_n_threads is called before DSP init completes.
    const int cap = (ctx->dsp_thread_counts > 0) ? ctx->dsp_thread_counts : g_hexagon_appcfg.thread_counts;
    ctx->n_threads = (n_threads < cap) ? n_threads : cap;
}

static int ggml_backend_hexagon_get_device_count() {
    return g_hexagon_appcfg.ndev;
}

static void ggml_backend_hexagon_atexit_cleanup() {
    if (g_reg_ctx) {
        delete g_reg_ctx;
        g_reg_ctx = nullptr;
    }
}

static const char * ggml_backend_hexagon_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "Hexagon-cDSP";
}

static size_t ggml_backend_hexagon_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_hexagon_reg_context * ctx = (ggml_backend_hexagon_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_hexagon_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_hexagon_reg_context * ctx = (ggml_backend_hexagon_reg_context *)reg->context;
    GGMLHEXAGON_LOG_WARN("reg_get_device: index=%zu count=%zu", index, ctx->devices.size());
    if (index >= ctx->devices.size()) {
        GGMLHEXAGON_LOG_ERROR("invalid device index %d (count=%zu)", index, ctx->devices.size());
        return nullptr;
    }
    return ctx->devices[index];
}

static void * ggml_backend_hexagon_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);

    if (nullptr == name)
        return nullptr;

    if (0 == strcmp(name, "ggml_backend_set_n_threads")) {
        return (void *)ggml_backend_hexagon_set_n_threads;
    }
    if (0 == strcmp(name, "ggml_backend_dev_get_extra_bufts")) {
        return (void *)ggml_backend_hexagon_device_get_extra_buffers_type;
    }

    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_hexagon_reg_interface = {
        /* .get_name          = */ ggml_backend_hexagon_reg_get_name,
        /* .get_device_count  = */ ggml_backend_hexagon_reg_get_device_count,
        /* .get_device        = */ ggml_backend_hexagon_reg_get_device,
        /* .get_proc_address  = */ ggml_backend_hexagon_reg_get_proc_address,
};

ggml_backend_reg_t ggml_backend_hexagon_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    ggmlhexagon_load_cfg();
    if (!ggmlhexagon_check_valid_appcfg()) {
        return nullptr;
    }

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            int ret = htpdrv_init();
            if (AEE_SUCCESS != ret) {
                GGMLHEXAGON_LOG_ERROR("htpdrv_init failed with error %d", ret);
                return nullptr;
            }

            int ndev = g_hexagon_appcfg.ndev;
            ggml_backend_hexagon_reg_context * ctx = new ggml_backend_hexagon_reg_context;
            GGMLHEXAGON_LOG_VERBOSE("registering %d Hexagon device(s), ndev=%d", ndev, g_hexagon_appcfg.ndev);

            for (int i = 0; i < ndev; i++) {
                if (i >= GGML_HEXAGON_MAX_DEVICES) {
                    GGMLHEXAGON_LOG_WARN("ndev=%d exceeds GGML_HEXAGON_MAX_DEVICES=%d, only %d devices registered",
                                         ndev, GGML_HEXAGON_MAX_DEVICES, i);
                    break;
                }

                GGMLHEXAGON_LOG_VERBOSE("register backend device %d (context created lazily)", i);
                // Only register the device struct here. Context (DSP session,
                // mempool) is created lazily by ggml_backend_hexagon_ensure_context
                // (called from get_buffer_type or init_backend) and persists across
                // inferences. It is freed during registry shutdown.
                ggml_backend_dev_t dev = new ggml_backend_device{
                        /* .iface       = */ ggml_backend_hexagon_device_interface,
                        /* .reg         = */ &reg,
                        /* .context     = */ nullptr  // set in device_init_backend
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                    /* .api_version = */ GGML_BACKEND_API_VERSION,
                    /* .iface       = */ ggml_backend_hexagon_reg_interface,
                    /* .context     = */ ctx
            };

            g_reg_ctx = ctx;
            std::atexit(ggml_backend_hexagon_atexit_cleanup);
        }

        initialized = true;
    }
    return &reg;
}

static const char * ggml_backend_hexagon_get_devname(size_t dev_num) {
    // CDSP devices: Hexagon-cDSP0, Hexagon-cDSP1, ...
    static char dev_names[GGML_HEXAGON_MAX_DEVICES][32];
    if (dev_num < GGML_HEXAGON_MAX_DEVICES) {
        snprintf(dev_names[dev_num], sizeof(dev_names[dev_num]), "Hexagon-cDSP%zu", dev_num);
        return dev_names[dev_num];
    }
    return "unknown";
}

static ggml_backend_t ggml_backend_hexagon_init_ext(size_t device, const char * runtime_libpath) {
    ggmlhexagon_load_cfg();
    if (!ggmlhexagon_check_valid_appcfg()) {
        return nullptr;
    }

    if (nullptr == runtime_libpath) {
        runtime_libpath = g_hexagon_appcfg.runtime_libpath;
    }

    GGMLHEXAGON_LOG_ALWAYS("device %d", device);
    GGMLHEXAGON_LOG_ALWAYS("runtime libpath %s", runtime_libpath);
    if (device >= GGML_HEXAGON_MAX_DEVICES) {
        GGMLHEXAGON_LOG_ERROR("invalid device %d", device);
        return nullptr;
    }

    if (0 != memcmp(runtime_libpath, g_hexagon_appcfg.runtime_libpath, strlen(g_hexagon_appcfg.runtime_libpath))) {
        //re-setting runtime libpath
        ggmlhexagon_set_runtime_path(device, runtime_libpath);
    }

    // Get the device from registry
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_hexagon_reg(), device);

    // Ensure context exists (lazy creation, same as device_init_backend)
    auto * ctx = ggml_backend_hexagon_ensure_context(dev);
    if (nullptr == ctx) {
        GGMLHEXAGON_LOG_ERROR("failed to create context");
        return nullptr;
    }

    // If backend already exists, return it
    if (nullptr != ctx->backend) {
        GGMLHEXAGON_LOG_ALWAYS("backend already exists for device %d, reusing", ctx->device);
        return ctx->backend;
    }

    ggml_backend_hexagon_interface.graph_optimize =
        g_hexagon_appcfg.enable_graph_optimize ? ggml_backend_hexagon_graph_optimize : nullptr;
    GGMLHEXAGON_LOG_ALWAYS("graph_optimize: %s", g_hexagon_appcfg.enable_graph_optimize ? "enabled" : "disabled");

    ggml_backend_t hexagon_backend = new ggml_backend{
            /* .guid      = */ ggml_backend_hexagon_guid(),
            /* .iface     = */ ggml_backend_hexagon_interface,
            /* .device    = */ dev,
            /* .context   = */ ctx
    };

    ctx->backend = hexagon_backend;
    return hexagon_backend;
}

ggml_backend_t ggml_backend_hexagon_init(void) {
    return ggml_backend_hexagon_init_ext(0, nullptr);
}

GGML_BACKEND_DL_IMPL(ggml_backend_hexagon_reg)
