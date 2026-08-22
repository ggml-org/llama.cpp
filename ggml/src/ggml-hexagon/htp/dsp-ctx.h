#ifndef GGMLDSP_CTX_H
#define GGMLDSP_CTX_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stddef.h>

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

/* Op-level profiling is disabled by default
 * Compile with -DHEX_OP_PROF=1 to enable */
#ifndef HEX_OP_PROF
#define HEX_OP_PROF                     0
#endif

/* Alignment requirements */
#define HEX_BATCH_ALIGN                 128
#define HEX_TENSOR_ALIGN                128
#define HEX_OP_ALIGN                    128

// HTP_TENSOR_FLUSHED was removed in upstream b2dd28a3b: per-tensor flush
// flags were replaced by htp_context.dirty_map, maintained by the dspqueue
// scheduler in main.c (kernels never flush L2 themselves). The mempool
// entry.c replaces that scheduler and does its own cache management, so
// htp_tensor.flags is never read on this path. Defined as 0 to keep the
// legacy assignments in entry.c compiling.
#ifndef HTP_TENSOR_FLUSHED
#define HTP_TENSOR_FLUSHED              0
#endif

/* Array size limits for per-batch tracking arrays. */
#ifndef WEIGHT_INVAL_MAX_PTRS
#define WEIGHT_INVAL_MAX_PTRS           4096
#endif

#ifndef DSP_OPT_MAX_TENSORS
#define DSP_OPT_MAX_TENSORS             4096
#endif

/* max n_ops * HTP_OP_MAX_OUTPUTS; n_ops upper bound = DSP_OPT_MAX_TENSORS * 4 */
#ifndef DSP_OPT_MAX_BATCH_DSTS
#define DSP_OPT_MAX_BATCH_DSTS          (DSP_OPT_MAX_TENSORS * 4 * 4)
#endif

#ifndef HTP_OP_MAX_OUTPUTS
#define HTP_OP_MAX_OUTPUTS              4
#endif

#ifndef HEX_OP_PROF_BUCKETS
#define HEX_OP_PROF_BUCKETS             64
#endif

#define GGMLHEXAGON_LOGBUF_LEN          4096
#define GGMLHEXAGON_TMPBUF_LEN          256

#define GGMLHEXAGON_LOG_ALWAYS(...)     ggmlhexagon_log_always_internal(GGML_LOG_LEVEL_NONE , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLHEXAGON_LOG_ERROR(...)      ggmlhexagon_log_always_internal(GGML_LOG_LEVEL_ERROR, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLHEXAGON_LOG_VERBOSE(...)    ggmlhexagon_log_always_internal(GGML_LOG_LEVEL_CONT , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLHEXAGON_LOG_WARN(...)       ggmlhexagon_log_internal(GGML_LOG_LEVEL_WARN , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLHEXAGON_LOG_INFO(...)       ggmlhexagon_log_internal(GGML_LOG_LEVEL_INFO , __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

#ifndef NDEBUG
#define GGMLHEXAGON_LOG_DEBUG(...)      ggmlhexagon_log_internal(GGML_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#else
#define GGMLHEXAGON_LOG_DEBUG(...)
#endif

// Forward declarations for types used in dsp_context.
// Note: hmx_queue_t is `struct hmx_queue_s *` (typedef in hmx-queue.h), so the
// forward declaration must use the _s suffix to match the new Qualcomm API.
struct hmx_queue_s;
struct htp_context;
struct htp_tensor;

typedef struct dsptensor dsptensor;

struct dsptensor {
   int32_t type;
   int32_t ne[4];
   int32_t nb[4];
   int32_t op;
   int32_t op_params[16];
   int32_t flags;
   void * data;
   int data_len;
};

typedef struct {
    void * base;
    size_t len;
    uint32_t tensor_idx;
} dsp_dst_range_t;

typedef struct dsp_op_desc dsp_op_desc;
struct dsp_op_desc {
   int32_t opcode;
   int32_t params[16];
   int32_t src0_idx;
   int32_t src1_idx;
   int32_t src2_idx;
   int32_t src3_idx;
   int32_t dst_idx;
};

/*
 * Shared memory batch descriptor for ION-based multi-op offload.
 *
 * Layout in mempool:
 *   [hex_batch_hdr]
 *   [hex_op_desc[0..n_ops-1]]
 *   [hex_tensor_desc[0..n_tensors-1]]
 *
 * All data_offset fields are byte offsets from the mempool base.
 * DSP side accesses data as: mempool_dsp_base + tensor->data_offset
 */

/* Tensor descriptor - uses offset instead of pointer */
typedef struct hex_tensor_desc {
    int32_t  type;            /* ggml_type */
    int32_t  ne[4];           /* element counts per dimension */
    int32_t  nb[4];           /* strides (bytes) per dimension */
    int32_t  op_params[16];   /* operation-specific parameters */
    uint32_t flags;           /* 0=ION tensor, 1=mirrored (heap), 2=weight (skip cache flush) */
    uint32_t data_offset;     /* byte offset of data in mempool */
    uint32_t data_len;        /* data length in bytes */
} hex_tensor_desc;

/* Op descriptor - references tensors by index.
 * Mirrors Qualcomm htp_op_desc layout: src_idx[HTP_OP_MAX_INPUTS=6] + dst_idx[HTP_OP_MAX_OUTPUTS=4]. */
typedef struct hex_op_desc {
    int32_t opcode;          /* GGML_OP_XXX */
    int32_t params[16];      /* operation parameters */
    int32_t kernel_params[32]; /* precomputed kernel params (e.g. htp_mm_kernel_params for MUL_MAT) */
    int32_t src_idx[6];      /* indices into tensor table (-1 = none); mirrors htp_op_desc.src[6] */
    int32_t dst_idx[4];      /* multi-output support (e.g. QKV fusion), -1 = unused */
    int32_t htp_opcode;      /* Direct HTP opcode for fused ops (0 = use ggml_op_to_htp_op) */
} hex_op_desc;

/* Batch header - entry point for DSP to find everything */
typedef struct hex_batch_hdr {
    uint32_t n_ops;             /* number of ops */
    uint32_t n_tensors;         /* number of tensors */
    uint32_t ops_offset;        /* offset from hdr start -> hex_op_desc[] */
    uint32_t tensors_offset;    /* offset from hdr start -> hex_tensor_desc[] */
    uint32_t total_size;        /* total size of this batch region (hdr + ops + tensors) */
    uint32_t reserved;          /* padding / future use */
} hex_batch_hdr;

// DSP session context: bundles all per-session state.
// Allocated in ggml_dsp_open, freed in ggml_dsp_close.
struct dsp_context {
    // Configuration
    int thread_counts;
    int max_hw_threads;     /* qurt_sysenv_get_max_hw_threads, set once at open */
    int dump_diag_info;

    // VTCM
    void * vtcm_base;
    size_t vtcm_size;
    unsigned int compute_res_ctx_id;
    volatile int vtcm_needs_release;
    volatile int vtcm_valid;
    int thread_prio;

    // Power
    int power_ctx;
    void * hexagon_power_ctx;

    // HMX
    int hmx_available;
    struct hmx_queue_s * hmx_queue;
    // Backing buffer for hmx_queue (NULL if hmx_queue is owned externally).
    // Allocated via memalign in ggml_dsp_setclocks and freed in ggml_dsp_close.
    void * hmx_queue_buf;

    // mempool
    void * mempool_dsp_base;
    size_t mempool_dsp_size;

    // DSP-side entry.c cache optimization bitmask. Pushed by AP at init via
    // execute_batch(0xFFFC) special mode (no IDL change). All three bits are
    //   are wired into ggml_dsp_execute_batch(); dsp_cache_mode=0 is behaviorally
    // identical to baseline 29c1cf196.
    //   bit 0 (0x1): first-touch weight bitmap    - skip dcinva for repack weights (flags==2) after first access
    //   bit 1 (0x2): skip dcinva for prior dst     - DSP's own dst writes stay in L2; next op's src read skips dcinva
    //   bit 2 (0x4): bulk dst flush at batch end   - collect/sort/merge dst ranges, flush once per region
    //   bit 3..31  : reserved for future use
    uint32_t dsp_cache_mode;

    // DSP-side bit 0 (first-touch weight bitmap) trace enable. Pushed by AP at
    // init via the same execute_batch(0xFFFC) special mode as dsp_cache_mode
    // (bit 16 of the same payload word, so the special-mode encoding is
    //   payload = (dsp_cache_trace_bit0 << 16) | (dsp_cache_mode & 0x7u)
    // ). When non-zero, INVAL_SRC_IF_NEEDED emits one [DSP-CACHE-TRACE-BIT0]
    // log line per bit 0 decision (SKIP or INVAL), with op index, src index,
    // weight address, weight length, current ctx id, and qurt_timer tick count.
    // Default 0 (off) so production perf is unaffected. Set to 1 only when
    // diagnosing the bit 0 stale L2 read bug (llama3 33% prompt-repeat rate
    // observed 2026-07-10). Once the bug is root-caused this can be removed.
    uint32_t dsp_cache_trace_bit0;

    // DSP-side bit 1 (skip dcinva for prior dst) trace enable. Pushed by AP at
    // init via bit 17 of the same execute_batch(0xFFFC) payload word, so the
    // special-mode encoding is
    //   payload = (dsp_cache_trace_bit1 << 17) | (dsp_cache_trace_bit0 << 16)
    //           | (dsp_cache_mode & 0x7u)
    // When non-zero, INVAL_SRC_IF_NEEDED emits one [DSP-CACHE-TRACE-BIT1] log
    // line per bit 1 decision (SKIP if prior_dst_contains_src, INVAL otherwise)
    // with the same op/src/ptr/len fields as the bit 0 trace. Default 0 (off)
    // so production perf is unaffected. Set to 1 to measure bit 1 SKIP rate
    // (how often INVAL_SRC_IF_NEEDED takes the prior-dst skip path). Currently
    // a no-op: PRIOR_DST_MAX_LEN=64 (entry.c) excludes all real-world cgraph
    // intermediate tensors (>= 256B). Pair with dsp_cache_trace_bit0 to
    // cross-check L2 staleness between weight and activation domains.
    uint32_t dsp_cache_trace_bit1;

    // htp_context for calling the shared execute_op.
    struct htp_context * htp_ctx;

    // Backing buffers for queues owned by this dsp_context (allocated via
    // memalign in ggml_dsp_setclocks, freed in ggml_dsp_close). The new
    // Qualcomm API (b2dd28a3b) requires callers to provide pre-allocated
    // memory to *_queue_init and does not free it in *_queue_free, so we
    // must track these buffers separately to avoid leaking them.
    //   work_queue_buf         : backing for htp_ctx->work_queue
    //   dma_queue_bufs[i]      : backing for htp_ctx->dma_cached[i] (NULL when slot unused)
    //   dma_alias_bufs[i]      : backing for htp_ctx->dma[i] alias (NULL when slot unused)
    void * work_queue_buf;
    void * dma_queue_bufs[16];  // HTP_MAX_NTHREADS == 10, but use 16 for safety
    void * dma_alias_bufs[16];

    // Per-session state (moved from file-static globals for multi-session isolation).
    // Small arrays are embedded; large arrays are allocated from arrays_pool.

    // Weight first-touch invalidate tracking (bit 0)
    const void * weight_inval_ptrs[WEIGHT_INVAL_MAX_PTRS];
    uint32_t weight_inval_count;

    // Per-batch src invalidation tracking
    uint8_t batch_tensor_needs_inval[DSP_OPT_MAX_TENSORS];

    // bit 3 last consumer op index per tensor
    uint32_t tensor_last_use_op[DSP_OPT_MAX_TENSORS];

    // Per-op dst staging buffers
    dsptensor        dst_dt_buf [HTP_OP_MAX_OUTPUTS];
    const dsptensor * dst_dt_ptrs[HTP_OP_MAX_OUTPUTS];

    // Large arrays: single mempool allocation for cache locality
    void * arrays_pool;
    dsptensor * pre_dt;                     // [DSP_OPT_MAX_TENSORS]
    struct htp_tensor * pre_ht;             // [DSP_OPT_MAX_TENSORS]
    dsp_dst_range_t * prior_dst_ranges;     // [DSP_OPT_MAX_BATCH_DSTS]
    dsp_dst_range_t * bulk_flush_ranges;    // [DSP_OPT_MAX_BATCH_DSTS]
    int prior_dst_count;
    int bulk_flush_count;

#if HEX_OP_PROF
    // Per-op profiling (compiled in when HEX_OP_PROF is non-zero)
    uint64_t op_prof_dur_us[HEX_OP_PROF_BUCKETS];
    uint64_t op_prof_count  [HEX_OP_PROF_BUCKETS];
    uint64_t op_prof_min_us[HEX_OP_PROF_BUCKETS];
    uint64_t op_prof_max_us[HEX_OP_PROF_BUCKETS];
    uint32_t op_prof_batch_count;
    uint64_t op_prof_batch_wall_us;
    uint64_t nonop_hdr_inval_us;
    uint64_t nonop_preconvert_us;
    uint64_t nonop_w_inval_us;
    uint64_t nonop_w_inval_bytes;
    uint64_t nonop_a_inval_us;
    uint64_t nonop_a_inval_bytes;
    uint64_t nonop_dst_track_us;
    uint64_t nonop_bulk_flush_us;
    uint64_t nonop_queue_us;
#endif
};

void ggmlhexagon_log_internal(int log_level, const char * file, const char * func, int line, const char * format, ...);
void ggmlhexagon_log_always_internal(int log_level, const char * file, const char * func, int line, const char * format, ...);

#ifdef  __cplusplus
}
#endif

#endif /* GGMLDSP_CTX_H */
