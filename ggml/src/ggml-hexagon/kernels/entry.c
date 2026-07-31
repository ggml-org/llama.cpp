#include <stdio.h>
#include <stdarg.h>

#include <HAP_power.h>
#include <HAP_farf.h>
#include <HAP_dcvs.h>
#include <HAP_mem.h>
#include <HAP_compute_res.h>
#include <math.h>
#include <qurt.h>
#include <qurt_mutex.h>
#include <remote.h>
#include <AEEStdDef.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

#include "ggml.h"
#include "dsp-ctx.h"
#include "hmx-queue.h"
#include "htp-ctx.h"
#include "matmul-ops.h"
#include "flash-attn-ops.h"

// =================================================================================================
// forward declarations, global vars, macros
// =================================================================================================
#define DSP_CACHE_LINE_SIZE             64

#define GGMLHEXAGON_LOGBUF_LEN          4096

#define DEFAULT_VTCM_SIZE               (8 * 1024 * 1024)

#define DSP_OPT_MAX_BATCH_DSTS          (256 * 4)  /* 256 ops * HTP_OP_MAX_OUTPUTS */

#define HEX_OP_PROF_BUCKETS             64

#define HEX_OP_PROF_DUMP_INTERVAL       25

#define WEIGHT_INVAL_MAX_PTRS           4096

#define DSP_OPT_MAX_TENSORS             2048

// Queue capacity/stack sizes for JZ-owned work/hmx queues. Mirror the defaults
// in htp/main.c (HMX_QUEUE_CAPACITY=16, HMX_QUEUE_STACK_SIZE=16384,
// WORK_QUEUE_CAPACITY=16, WORK_QUEUE_STACK_SIZE=16384). main.c keeps these as
// file-local #defines; we replicate them here with a JZ_ prefix so the JZ
// entry.c path can construct queues with the same geometry.
#define JZ_HMX_QUEUE_CAPACITY        16
#define JZ_HMX_QUEUE_STACK_SIZE      16384
#define JZ_WORK_QUEUE_CAPACITY       16
#define JZ_WORK_QUEUE_STACK_SIZE     16384

/* Maximum dst length eligible for prior-dst skip (bit 1). Strategy 2: only
 * allow skipping invalidation when the prior dst fits within a single L2
 * cacheline. Anything larger may have been produced through async DMA/HMX
 * paths and risks stale scalar L2 reads. The op-type whitelist provides an
 * extra safety net.
 *
 * Note: experiments with 8KB and 64KB limits (2026-07-18) caused garbled
 * output and immediate [end of text] emission. The deferred-flush pattern
 * (bit 2) is unsafe to combine with bit 1 on any meaningful range, because
 * L2 can evict dirty dst data before the deferred flush, causing stale
 * DRAM reads. The single-cacheline limit is the only size that survives
 * the L2 churn from concurrent weight reads. */
#define PRIOR_DST_MAX_LEN               DSP_CACHE_LINE_SIZE

#ifndef NDEBUG
#define GGMLHEXAGON_DEBUG               1
#else
#define GGMLHEXAGON_DEBUG               0
#endif

// Per-op timing profiler: cumulative us per HTP op kind, indexed by octx->op.
// Bumped by execute_op() in entry.c, dumped via FARF every N batches inside
// ggml_dsp_execute_batch(). Tied to GGMLHEXAGON_DEBUG: release builds (NDEBUG)
// compile it out entirely.
#if GGMLHEXAGON_DEBUG
#define HEX_OP_PROF                     1
#else
#define HEX_OP_PROF                     0
#endif

#if GGMLHEXAGON_DEBUG
#define GGMLHEXAGON_LOG_DEBUG(...)      ggml_log_internal(GGMLHEXAGON_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#define GGMLHEXAGON_LOG_WARN(...)       ggml_log_internal(GGMLHEXAGON_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#else
#define GGMLHEXAGON_LOG_DEBUG(...)
#define GGMLHEXAGON_LOG_WARN(...)
#endif

#define GGMLHEXAGON_LOG_INFO(...)       ggml_log_always(GGMLHEXAGON_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

#define GGMLHEXAGON_LOG_ERROR(...)      ggml_log_always(GGMLHEXAGON_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

#define INVAL_SRC_IF_NEEDED(op_i, src_idx, dt_ptr, tensor_idx) do {                     \
    if (dt_ptr) {                                                                      \
        /* Per-batch dedup: skip if already invalidated and not dirtied since. */      \
        if (!g_batch_tensor_needs_inval[tensor_idx]) {                                 \
            /* already invalidated this batch, L2 line is fresh */                     \
        } else if ((dt_ptr)->flags & 0x2) {                                             \
            /* Weight tensor: bit 0 check */                                           \
            bool _already_inval = false;                                               \
            if ((g_dsp_ctx->dsp_cache_mode & 0x1) &&                                   \
                (_already_inval = weight_inval_check_and_mark((dt_ptr)->data))) {       \
                if (g_dsp_ctx->dsp_cache_trace_bit0) {                                \
                    GGMLHEXAGON_LOG_INFO("[DSP-CACHE-TRACE-BIT0] op=%u src=%d SKIP ptr=%p len=0x%x (cache_mode=0x%x)", \
                                         (op_i), (int)(src_idx), (dt_ptr)->data, (dt_ptr)->data_len, g_dsp_ctx->dsp_cache_mode); \
                }                                                                       \
            } else {                                                                   \
                prof_cache_inval_range((dt_ptr)->data, (dt_ptr)->data_len, 1);         \
                if (g_dsp_ctx->dsp_cache_trace_bit0) {                                \
                    GGMLHEXAGON_LOG_INFO("[DSP-CACHE-TRACE-BIT0] op=%u src=%d INVAL ptr=%p len=0x%x (cache_mode=0x%x)", \
                                         (op_i), (int)(src_idx), (dt_ptr)->data, (dt_ptr)->data_len, g_dsp_ctx->dsp_cache_mode); \
                }                                                                       \
            }                                                                          \
        } else {                                                                       \
            /* Activation tensor: bit 1 check */                                       \
            if ((g_dsp_ctx->dsp_cache_mode & 0x2) &&                                   \
                prior_dst_contains_src((tensor_idx), (dt_ptr)->data, (dt_ptr)->data_len)) { \
                if (g_dsp_ctx->dsp_cache_trace_bit1) {                                \
                    GGMLHEXAGON_LOG_INFO("[DSP-CACHE-TRACE-BIT1] op=%u src=%d SKIP ptr=%p len=0x%x (cache_mode=0x%x)", \
                                         (op_i), (int)(src_idx), (dt_ptr)->data, (dt_ptr)->data_len, g_dsp_ctx->dsp_cache_mode); \
                }                                                                       \
            } else {                                                                   \
                prof_cache_inval_range((dt_ptr)->data, (dt_ptr)->data_len, 0);         \
                if (g_dsp_ctx->dsp_cache_trace_bit1) {                                \
                    GGMLHEXAGON_LOG_INFO("[DSP-CACHE-TRACE-BIT1] op=%u src=%d INVAL ptr=%p len=0x%x (cache_mode=0x%x)", \
                                         (op_i), (int)(src_idx), (dt_ptr)->data, (dt_ptr)->data_len, g_dsp_ctx->dsp_cache_mode); \
                }                                                                       \
            }                                                                          \
        }                                                                              \
        g_batch_tensor_needs_inval[tensor_idx] = 0;                                    \
    }                                                                                  \
} while (0)

enum ggmlhexagon_log_level {
    GGMLHEXAGON_LOG_LEVEL_NONE  = 0,
    GGMLHEXAGON_LOG_LEVEL_DEBUG = 1,
    GGMLHEXAGON_LOG_LEVEL_WARN  = 2,
    GGMLHEXAGON_LOG_LEVEL_ERROR = 3,
    GGMLHEXAGON_LOG_LEVEL_INFO  = 4,
};

typedef struct {
    void * base;
    size_t len;
    uint32_t tensor_idx;  /* only used by prior_dst (bit 1) */
} dsp_dst_range_t;

int64_t ggml_time_us(void); /* defined below */
static void ggml_dsp_cache_inval_range(void * addr, size_t size); /* defined below */

#if HEX_OP_PROF
static uint64_t g_op_prof_dur_us[HEX_OP_PROF_BUCKETS];
static uint64_t g_op_prof_count  [HEX_OP_PROF_BUCKETS];

// Per-call min/max: min is init to UINT64_MAX so the first real call always
// sets it; max is init to 0 for the symmetric reason. init_op_prof_min()
// below applies the min init lazily (called from dump_op_prof) so the
// arrays stay in BSS as plain zero-init globals.
static uint64_t g_op_prof_min_us[HEX_OP_PROF_BUCKETS];
static uint64_t g_op_prof_max_us[HEX_OP_PROF_BUCKETS];
static uint32_t g_op_prof_batch_count;
static uint64_t g_op_prof_batch_wall_us; /* cumulative whole-batch wall time, vs per-op sum */

/* Non-op section timers: decompose batch-wall - op-sum. All cumulative. */
static uint64_t g_nonop_hdr_inval_us;    /* batch descriptor dcinva */
static uint64_t g_nonop_preconvert_us;   /* hex_tensor_to_dsptensor/htp_tensor loop */
static uint64_t g_nonop_w_inval_us;      /* src dcinva, flags&0x2 (weight) path */
static uint64_t g_nonop_w_inval_bytes;
static uint64_t g_nonop_a_inval_us;      /* src dcinva, activation path */
static uint64_t g_nonop_a_inval_bytes;
static uint64_t g_nonop_dst_track_us;    /* per-op dst tracker (bulk add / direct flush) */
static uint64_t g_nonop_bulk_flush_us;   /* bulk_flush_all at batch end */
static uint64_t g_nonop_queue_us;        /* dsp_queues_wakeup + suspend */

static inline void prof_cache_inval_range(void * p, size_t len, int is_weight) {
    const int64_t t0 = ggml_time_us();
    ggml_dsp_cache_inval_range(p, len);
    const uint64_t dt = (uint64_t)(ggml_time_us() - t0);
    if (is_weight) {
        g_nonop_w_inval_us += dt;
        g_nonop_w_inval_bytes += len;
    } else {
        g_nonop_a_inval_us += dt;
        g_nonop_a_inval_bytes += len;
    }
}
#else
#define prof_cache_inval_range(p, len, is_weight) ggml_dsp_cache_inval_range((p), (len))
#endif // HEX_OP_PROF

// Per-weight-region first-touch invalidate tracking.
// Repack weights (flags==2) live in stable ION regions that AP writes
// once at model load and never touches again, so DSP can cache them in
// L2 for the entire session after a single first-touch invalidate.
// We use an exact sorted pointer array (not a hash bitmap) to avoid the
// address-collision bug that caused garbled output with bit 0.
static const void * g_weight_inval_ptrs[WEIGHT_INVAL_MAX_PTRS];
static uint32_t g_weight_inval_count = 0;

// Per-batch src invalidation tracking: avoids redundant dcinva calls
// when the same tensor is used as src by multiple ops within the same batch.
// Keyed by tensor index (0..n_tensors-1). Each byte is 1 if the tensor
// needs invalidation (dirty or never invalidated), 0 if already invalidated
// and clean. Reset at batch start; cleared when a tensor is invalidated;
// set when a tensor is written as dst (dirtied).
static uint8_t g_batch_tensor_needs_inval[DSP_OPT_MAX_TENSORS];

/* bit 3 (0x8): last consumer op index per tensor (0 = never used as src).
 * Built once at batch start; consulted by the dst tracker to skip the
 * batch-end flush for intermediates that a later op still consumes. */
static uint32_t g_tensor_last_use_op[DSP_OPT_MAX_TENSORS];

// Pre-converted tensor descriptors: converted once at batch start instead
// of per-op in the loop. Saves hex_tensor_to_dsptensor() calls for srcs.
static dsptensor g_pre_dt[DSP_OPT_MAX_TENSORS];

// Pre-converted htp_tensor descriptors for op dispatch. Eliminates per-op
// dsptensor_to_htp_tensor() calls and stack-allocated src_ht/dst_ht arrays.
// Mirrors htp/main.c prep_tensors pattern: tensors are directly usable by
// execute_op without any per-op conversion.
static struct htp_tensor g_pre_ht[DSP_OPT_MAX_TENSORS];

static int             g_prior_dst_count;
static dsp_dst_range_t g_prior_dst_ranges[DSP_OPT_MAX_BATCH_DSTS];

static int             g_bulk_flush_count;
static dsp_dst_range_t g_bulk_flush_ranges[DSP_OPT_MAX_BATCH_DSTS];

// Per-op dst staging buffers (moved out of the per-op stack frame).
// Hexagon hardware stack is shallow; ~500 bytes of stack alloc per op across
// 30+ ops/batch * 256 tokens adds up. Static storage removes the per-op
// frame setup/teardown. The DSP batch path is single-threaded, so static
// state is safe.
static dsptensor        g_dst_dt_buf [HTP_OP_MAX_OUTPUTS];
static const dsptensor * g_dst_dt_ptrs[HTP_OP_MAX_OUTPUTS];

static struct dsp_context * g_dsp_ctx = NULL;

// =================================================================================================
// troubleshooting and profiler
// =================================================================================================
// ggml_abort() is declared in ggml.h; GGML_ASSERT/GGML_ABORT macros call it.
void ggml_abort(const char * file, int line, const char * fmt, ...) {
    abort();
}

int64_t ggml_time_us(void) {
    unsigned long long count;
    asm volatile(" %0 = c31:30 " : "=r"(count));
    return (uint64_t)(count) * 10ull / 192ull;
}

#if GGMLHEXAGON_DEBUG
static void ggml_log_internal(int level, const char *file, const char *func, int line, const char *format, ...) {
    static char s_ggmlhexagon_log_internal_buf[GGMLHEXAGON_LOGBUF_LEN];
    va_list args;
    va_start(args, format);
    int len_prefix = snprintf(s_ggmlhexagon_log_internal_buf, GGMLHEXAGON_LOGBUF_LEN, "[%s, %d]: ",
                              func, line);
    int len = vsnprintf(s_ggmlhexagon_log_internal_buf + len_prefix,
                        GGMLHEXAGON_LOGBUF_LEN - len_prefix, format, args);
    if (len < (GGMLHEXAGON_LOGBUF_LEN - len_prefix)) {
        FARF(ALWAYS, "%s\n", s_ggmlhexagon_log_internal_buf);
    }
    va_end(args);
}
#endif // GGMLHEXAGON_DEBUG

static void ggml_log_always(int level, const char *file, const char *func, int line, const char *format, ...) {
    if (!g_dsp_ctx->dump_diag_info) {
        return;
    }
    static char s_ggmlhexagon_log_internal_buf[GGMLHEXAGON_LOGBUF_LEN];
    va_list args;
    va_start(args, format);
    int len_prefix = snprintf(s_ggmlhexagon_log_internal_buf, GGMLHEXAGON_LOGBUF_LEN, "[%s, %d]: ",
                              func, line);
    int len = vsnprintf(s_ggmlhexagon_log_internal_buf + len_prefix,
                        GGMLHEXAGON_LOGBUF_LEN - len_prefix, format, args);
    if (len < (GGMLHEXAGON_LOGBUF_LEN - len_prefix)) {
        FARF(ALWAYS, "%s\n", s_ggmlhexagon_log_internal_buf);
    }
    va_end(args);
}

// Dump per-op timing accumulators via FARF. Best-effort: maps known HTP op
// codes to short names so the log is readable; unknown indices are emitted
// as plain numeric IDs. Only buckets that have at least one call are printed,
// so a single 1-line entry per op kind keeps log volume manageable.
#if HEX_OP_PROF
static const char * htp_op_short_name(unsigned int op) {
    switch (op) {
        case HTP_OP_MUL_MAT:         return "MUL_MAT";
        case HTP_OP_MUL_MAT_ADD:     return "MUL_MAT_ADD";
        case HTP_OP_MUL_MAT_ID:      return "MUL_MAT_ID";
        case HTP_OP_MUL_MAT_QKV:     return "MUL_MAT_QKV";
        case HTP_OP_MUL_MAT_FFN:     return "MUL_MAT_FFN";
        case HTP_OP_MUL:             return "MUL";
        case HTP_OP_ADD:             return "ADD";
        case HTP_OP_SUB:             return "SUB";
        case HTP_OP_DIV:             return "DIV";
        case HTP_OP_ADD_ID:          return "ADD_ID";
        case HTP_OP_NORM:            return "NORM";
        case HTP_OP_RMS_NORM:        return "RMS_NORM";
        case HTP_OP_RMS_NORM_MUL:    return "RMS_NORM_MUL";
        case HTP_OP_SCALE:           return "SCALE";
        case HTP_OP_SQR:             return "SQR";
        case HTP_OP_SQRT:            return "SQRT";
        case HTP_OP_L2_NORM:         return "L2_NORM";
        case HTP_OP_UNARY_SOFTPLUS:  return "UNARY_SOFTPLUS";
        case HTP_OP_UNARY_SIGMOID:   return "UNARY_SIGMOID";
        case HTP_OP_UNARY_NEG:       return "UNARY_NEG";
        case HTP_OP_UNARY_EXP:       return "UNARY_EXP";
        case HTP_OP_UNARY_TANH:      return "UNARY_TANH";
        case HTP_OP_UNARY_SILU:      return "UNARY_SILU";
        case HTP_OP_UNARY_GELU:      return "UNARY_GELU";
        case HTP_OP_GLU_SWIGLU:      return "GLU_SWIGLU";
        case HTP_OP_GLU_SWIGLU_OAI:  return "GLU_SWIGLU_OAI";
        case HTP_OP_GLU_GEGLU:       return "GLU_GEGLU";
        case HTP_OP_SOFTMAX:         return "SOFTMAX";
        case HTP_OP_ROPE:            return "ROPE";
        case HTP_OP_FLASH_ATTN_EXT:  return "FLASH_ATTN_EXT";
        case HTP_OP_SET_ROWS:        return "SET_ROWS";
        case HTP_OP_GET_ROWS:        return "GET_ROWS";
        case HTP_OP_SUM_ROWS:        return "SUM_ROWS";
        case HTP_OP_CPY:             return "CPY";
        case HTP_OP_REPEAT:          return "REPEAT";
        case HTP_OP_ARGSORT:         return "ARGSORT";
        case HTP_OP_SSM_CONV:        return "SSM_CONV";
        case HTP_OP_CUMSUM:          return "CUMSUM";
        case HTP_OP_FILL:            return "FILL";
        case HTP_OP_DIAG:            return "DIAG";
        case HTP_OP_SOLVE_TRI:       return "SOLVE_TRI";
        case HTP_OP_PAD:             return "PAD";
        case HTP_OP_CONCAT:          return "CONCAT";
        case HTP_OP_CLAMP:           return "CLAMP";
        case HTP_OP_IM2COL:          return "IM2COL";
        case HTP_OP_GATED_DELTA_NET: return "GATED_DELTA_NET";
        case HTP_OP_TRI:             return "TRI";
        case HTP_OP_INVALID:         return "INVALID";
        default:                     return NULL;
    }
}

// One-shot init for the per-op profiler: stamp min to UINT64_MAX so the
// first real call always sets it. Called lazily from dump_op_prof so we
// don't need a separate init hook in ggml_dsp_open. Idempotent.
static void init_op_prof_min(void) {
    static int done = 0;
    if (done) return;
    for (unsigned int i = 0; i < HEX_OP_PROF_BUCKETS; i++) {
        g_op_prof_min_us[i] = UINT64_MAX;
    }
    done = 1;
}

static void dump_op_prof(const char * tag) {
    init_op_prof_min();
    for (unsigned int i = 0; i < HEX_OP_PROF_BUCKETS; i++) {
        if (g_op_prof_count[i] == 0) continue;
        const char * name = htp_op_short_name(i);
        const uint64_t avg = g_op_prof_dur_us[i] / g_op_prof_count[i];
        // Pre-format numeric fields via snprintf so the field width is honored
        // (Hexagon FARF does not implement the width modifier in %9llu, so the
        // values would print left-justified otherwise). Widths leave headroom:
        //   cum   -> 10 chars (up to 9_999_999_999 us ~ 2.7h of DSP time)
        //   count ->  7 chars (up to       9_999_999 calls)
        //   avg   ->  5 chars (per-op cost is bounded by graph structure)
        //   min   ->  5 chars
        //   max   ->  6 chars (handles up to 999_999 us, well above any
        //                     realistic per-op stall in this profiler)
        char cum_s[16], cnt_s[16], avg_s[16], min_s[16], max_s[16];
        snprintf(cum_s, sizeof(cum_s), "%10llu", (unsigned long long)g_op_prof_dur_us[i]);
        snprintf(cnt_s, sizeof(cnt_s), "%7llu",  (unsigned long long)g_op_prof_count[i]);
        snprintf(avg_s, sizeof(avg_s), "%5llu",  (unsigned long long)avg);
        snprintf(min_s, sizeof(min_s), "%5llu",  (unsigned long long)g_op_prof_min_us[i]);
        snprintf(max_s, sizeof(max_s), "%6llu",  (unsigned long long)g_op_prof_max_us[i]);
        if (name) {
            FARF(ERROR, "[OP-PROF] %s op=%s cum=%s us count=%s avg=%s min=%s max=%s us",
                 tag, name, cum_s, cnt_s, avg_s, min_s, max_s);
        } else {
            FARF(ERROR, "[OP-PROF] %s op=%u cum=%s us count=%s avg=%s min=%s max=%s us",
                 tag, i, cum_s, cnt_s, avg_s, min_s, max_s);
        }
    }
}
#endif // HEX_OP_PROF

// =================================================================================================
// cache infrastructure
// =================================================================================================
static void ggml_dsp_cache_flush_range(void * addr, size_t size) {
    if (!addr || size == 0) return;
    char * p = (char *)addr;
    char * end = p + size;
    const size_t line_size = DSP_CACHE_LINE_SIZE;
    p = (char *)((uintptr_t)p & ~(line_size - 1));
    for (; p + line_size * 8 <= end; p += line_size * 8) {
        Q6_dccleaninva_A(p + line_size * 0);
        Q6_dccleaninva_A(p + line_size * 1);
        Q6_dccleaninva_A(p + line_size * 2);
        Q6_dccleaninva_A(p + line_size * 3);
        Q6_dccleaninva_A(p + line_size * 4);
        Q6_dccleaninva_A(p + line_size * 5);
        Q6_dccleaninva_A(p + line_size * 6);
        Q6_dccleaninva_A(p + line_size * 7);
    }
    for (; p < end; p += line_size) {
        Q6_dccleaninva_A(p);
    }
    __asm__ __volatile__("syncht\n");
}

// Flush range WITHOUT the trailing syncht. Used by bulk_flush_all to issue a
// single syncht after processing all merged regions, instead of one per
// region. Caller must issue syncht before any read of the flushed data.
static inline void ggml_dsp_cache_flush_range_nosync(void * addr, size_t size) {
    if (!addr || size == 0) return;
    char * p = (char *)addr;
    char * end = p + size;
    const size_t line_size = DSP_CACHE_LINE_SIZE;
    p = (char *)((uintptr_t)p & ~(line_size - 1));
    for (; p + line_size * 8 <= end; p += line_size * 8) {
        Q6_dccleaninva_A(p + line_size * 0);
        Q6_dccleaninva_A(p + line_size * 1);
        Q6_dccleaninva_A(p + line_size * 2);
        Q6_dccleaninva_A(p + line_size * 3);
        Q6_dccleaninva_A(p + line_size * 4);
        Q6_dccleaninva_A(p + line_size * 5);
        Q6_dccleaninva_A(p + line_size * 6);
        Q6_dccleaninva_A(p + line_size * 7);
    }
    for (; p < end; p += line_size) {
        Q6_dccleaninva_A(p);
    }
}

static void ggml_dsp_cache_inval_range(void * addr, size_t size) {
    if (!addr || size == 0) return;
    char * p = (char *)addr;
    char * end = p + size;
    const size_t line_size = DSP_CACHE_LINE_SIZE;
    p = (char *)((uintptr_t)p & ~(line_size - 1));
    for (; p + line_size * 8 <= end; p += line_size * 8) {
        Q6_dcinva_A(p + line_size * 0);
        Q6_dcinva_A(p + line_size * 1);
        Q6_dcinva_A(p + line_size * 2);
        Q6_dcinva_A(p + line_size * 3);
        Q6_dcinva_A(p + line_size * 4);
        Q6_dcinva_A(p + line_size * 5);
        Q6_dcinva_A(p + line_size * 6);
        Q6_dcinva_A(p + line_size * 7);
    }
    for (; p < end; p += line_size) {
        Q6_dcinva_A(p);
    }
    __asm__ __volatile__("syncht\n");
}

static inline bool weight_inval_check_and_mark(const void * ptr) {
    int lo = 0;
    int hi = (int)g_weight_inval_count - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        const void * mid_ptr = g_weight_inval_ptrs[mid];
        if (mid_ptr == ptr) {
            return true;  // already invalidated, can skip
        }
        if (mid_ptr < ptr) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    // Not found: insert at lo and keep array sorted.
    if (g_weight_inval_count >= WEIGHT_INVAL_MAX_PTRS) {
        // Table full: fall back to always invalidate. This is safe but
        // loses the optimization. With 4096 slots and read-only weights,
        // overflow should not happen in practice.
        return false;
    }
    for (int i = (int)g_weight_inval_count; i > lo; i--) {
        g_weight_inval_ptrs[i] = g_weight_inval_ptrs[i - 1];
    }
    g_weight_inval_ptrs[lo] = ptr;
    g_weight_inval_count++;
    return false;
}

static inline void weight_inval_reset_all(void) {
    g_weight_inval_count = 0;
}

/* A tensor written as dst is no longer read-only: drop it from the
 * first-touch weight list so its next src use re-invalidates. Fixes
 * cross-graph staleness where a tensor is dst in one (sub)graph and
 * misclassified as weight (flags=2) in another, e.g. qwen3-mtp. */
static inline void weight_inval_unmark(const void * ptr) {
    int lo = 0;
    int hi = (int)g_weight_inval_count - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        const void * mid_ptr = g_weight_inval_ptrs[mid];
        if (mid_ptr == ptr) {
            for (int i = mid; i < (int)g_weight_inval_count - 1; i++) {
                g_weight_inval_ptrs[i] = g_weight_inval_ptrs[i + 1];
            }
            g_weight_inval_count--;
            return;
        }
        if (mid_ptr < ptr) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
}

/* DSP-side cache optimization: prior_dst tracking (bit 1) + bulk flush (bit 2).
 *
 * Both are gated by g_dsp_ctx->dsp_cache_mode and operate on per-batch state.
 * They use simple range lists (no hash bitmap) to avoid the ptr-based hash
 * collision bug from the earlier commit that caused qwen3 garbled output.
 *
 * - bit 1 (skip dcinva for prior dst): when the next op's src is the *same*
 *   tensor as a small dst of an earlier op in the same batch, and its range is
 *   fully contained, skip dcinva because the L2 line is fresh (DSP's own write).
 *   We require tensor_idx equality and len <= PRIOR_DST_MAX_LEN to avoid stale
 *   reads across DMA/HMX cache domains or aliased views. The per-op dst tracker
 *   populates the prior_dst list only when bit 2 is on, so bit 1 alone is a
 *   no-op (a cfg comment in ggml-hexagon.cfg covers this).
 * - bit 2 (bulk dst flush at batch end): collect all dst ranges during the
 *   op loop, sort + merge adjacent/overlapping ranges at batch end, then
 *   call ggml_dsp_cache_flush_range() once per merged region. Replaces
 *   per-op flush with fewer but larger flushes. prior_dst is also updated
 *   when bit 2 is on, regardless of bit 1.
 *
 * State is reset at the start of every batch (next batch sees empty lists).
 * Both lists are sized for the worst case: 256 ops/batch * 4 dst/op.
 * Hexagon batch path is single-threaded (one FastRPC call drives one batch),
 * so static globals are safe.
 *
 * bit 0 (first-touch weight tracking) is a separate mechanism; see
 * weight_inval_check_and_mark() and INVAL_SRC_IF_NEEDED(). It is session-
 * scoped (array never reset) because repack weights are stable ION regions
 * written once at model load.
 */
/* True iff [q_base, q_base+q_len) is fully contained in [r_base, r_base+r_len). */
static inline bool dsp_range_contains(const void * r_base, size_t r_len,
                                      const void * q_base, size_t q_len) {
    if (!r_base || !q_base || q_len == 0) return false;
    uintptr_t rb = (uintptr_t)r_base;
    uintptr_t re = rb + r_len;
    uintptr_t qb = (uintptr_t)q_base;
    uintptr_t qe = qb + q_len;
    return (qb >= rb) && (qe <= re);
}

static inline bool prior_dst_contains_src(uint32_t src_idx,
                                          const void * base, size_t len) {
    if (!base) return false;
    for (int i = 0; i < g_prior_dst_count; i++) {
        if (g_prior_dst_ranges[i].tensor_idx == src_idx &&
            dsp_range_contains(g_prior_dst_ranges[i].base, g_prior_dst_ranges[i].len,
                               base, len)) {
            return true;
        }
    }
    return false;
}

static inline void prior_dst_add(void * base, size_t len, uint32_t tensor_idx,
                                 enum htp_op_code op) {
    (void)op;
    if (!base || len == 0) return;
    /* With PRIOR_DST_MAX_LEN == cacheline size, the prior-dst is small enough
     * that it should still reside in scalar L2. */
    if (len > PRIOR_DST_MAX_LEN) return;
    if (g_prior_dst_count >= DSP_OPT_MAX_BATCH_DSTS) return;  /* overflow guard */
    g_prior_dst_ranges[g_prior_dst_count].base = base;
    g_prior_dst_ranges[g_prior_dst_count].len  = len;
    g_prior_dst_ranges[g_prior_dst_count].tensor_idx = tensor_idx;
    g_prior_dst_count++;
}

static inline void bulk_flush_add(void * base, size_t len) {
    if (!base || len == 0) return;
    if (g_bulk_flush_count >= DSP_OPT_MAX_BATCH_DSTS) {
        /* Overflow: fall back to immediate per-range flush for THIS dst only
         * (degraded perf, but correctness preserved). */
        ggml_dsp_cache_flush_range(base, len);
        return;
    }
    g_bulk_flush_ranges[g_bulk_flush_count].base = base;
    g_bulk_flush_ranges[g_bulk_flush_count].len  = len;
    g_bulk_flush_count++;
}

static inline void prior_dst_reset_all(void) {
    g_prior_dst_count = 0;
}

static inline void bulk_flush_reset_all(void) {
    g_bulk_flush_count = 0;
}

/* Insertion sort (small N typical for batch dst list: 30-50 dsts).
 * Avoids libc qsort dependency and works on 32-bit pointers. */
static inline void bulk_flush_sort(void) {
    for (int i = 1; i < g_bulk_flush_count; i++) {
        dsp_dst_range_t cur = g_bulk_flush_ranges[i];
        int j = i - 1;
        while (j >= 0 &&
               (uintptr_t)g_bulk_flush_ranges[j].base > (uintptr_t)cur.base) {
            g_bulk_flush_ranges[j + 1] = g_bulk_flush_ranges[j];
            j--;
        }
        g_bulk_flush_ranges[j + 1] = cur;
    }
}

/* Walk sorted list, merge adjacent/overlapping ranges, flush each merged
 * region once. Overlap defined as next_start <= cur_end (1B threshold:
 * touching ranges merge). Flushes in sorted order to preserve ION ordering.
 *
 * Uses ggml_dsp_cache_flush_range_nosync() to avoid one syncht per region;
 * a single syncht is issued at the end so AP reads of the flushed tensors
 * see fresh DRAM. This batch-end consolidation is the only safe place to
 * merge syncht barriers; per-op synchts are still required for the
 * bit-2-disabled fallback path (see ggml_dsp_execute_batch). */
static inline void bulk_flush_all(void) {
    if (g_bulk_flush_count == 0) return;
    bulk_flush_sort();
    void * cur_base = g_bulk_flush_ranges[0].base;
    uintptr_t cur_end = (uintptr_t)cur_base + g_bulk_flush_ranges[0].len;
    int i = 1;
    while (i < g_bulk_flush_count) {
        void * next_base = g_bulk_flush_ranges[i].base;
        uintptr_t next_end = (uintptr_t)next_base + g_bulk_flush_ranges[i].len;
        if ((uintptr_t)next_base <= cur_end) {
            /* Overlap or adjacent: extend current region */
            if (next_end > cur_end) cur_end = next_end;
        } else {
            /* Gap: flush [cur_base, cur_end) and start new region */
            ggml_dsp_cache_flush_range_nosync(cur_base, (size_t)(cur_end - (uintptr_t)cur_base));
            cur_base = next_base;
            cur_end  = next_end;
        }
        i++;
    }
    ggml_dsp_cache_flush_range_nosync(cur_base, (size_t)(cur_end - (uintptr_t)cur_base));
    /* Single syncht covers all merged-region flushes above. AP reads
     * following this point see consistent DRAM. */
    __asm__ __volatile__("syncht\n");
}


// =================================================================================================
// VTCM
// =================================================================================================
static void dsp_vtcm_release(void) {
    if (g_dsp_ctx->vtcm_valid) {
        g_dsp_ctx->vtcm_valid = 0;
        g_dsp_ctx->vtcm_needs_release = 0;
        HAP_compute_res_release_cached(g_dsp_ctx->compute_res_ctx_id);
    }
}

static void dsp_vtcm_acquire(void) {
    if (!g_dsp_ctx->vtcm_valid) {
        int err = HAP_compute_res_acquire_cached(g_dsp_ctx->compute_res_ctx_id, 1000000u);
        if (err != 0) {
            FARF(ERROR, "failed to acquire VTCM: 0x%08x", (unsigned)err);
            abort();
        }
        g_dsp_ctx->vtcm_needs_release = 0;
        g_dsp_ctx->vtcm_valid = 1;
        HAP_compute_res_update_priority(g_dsp_ctx->compute_res_ctx_id,
                                        g_dsp_ctx->thread_prio + 10);
    }
}

static int vtcm_release_callback(unsigned int rctx, void * state) {
    // Async notification only: flag that another session wants VTCM.
    // Do NOT clear g_dsp_ctx->vtcm_valid here - the current batch keeps running
    // and releases VTCM at the batch boundary.
    g_dsp_ctx->vtcm_needs_release = 1;
    return 0;
}

// =================================================================================================
// IDL helper functions
// =================================================================================================
static int power_on_hvx_hmx(void) {
    HAP_power_request_t req;

    /* Set client class */
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_apptype;
    req.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    if (HAP_power_set((void *)&g_dsp_ctx->power_ctx, &req) != 0) {
        GGMLHEXAGON_LOG_ERROR("HAP_power_set apptype failed");
        return -1;
    }

    /* DCVS performance mode */
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_DCVS_v3;
    req.dcvs_v3.set_dcvs_enable = 1;
    req.dcvs_v3.dcvs_enable = 0;  // disable DVFS, pin to fixed frequency for stable performance
    req.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
    req.dcvs_v3.set_bus_params = 1;
    req.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.set_core_params = 1;
    req.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.set_sleep_disable = 1;
    req.dcvs_v3.sleep_disable = 1;

    GGMLHEXAGON_LOG_INFO("__HVX_ARCH__ = %d\n", __HVX_ARCH__);

    // v79 architecture requires protected bus corners setting
#if __HEXAGON_ARCH__ >= 79
    HAP_set_dcvs_v3_protected_bus_corners(&req, 1);
#endif

    if (HAP_power_set((void *)&g_dsp_ctx->power_ctx, &req) != 0) {
        GGMLHEXAGON_LOG_ERROR("HAP_power_set DCVS failed");
        return -2;
    }

    /* Power up HVX */
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_HVX;
    req.hvx.power_up = 1;
    if (HAP_power_set((void *)&g_dsp_ctx->power_ctx, &req) != 0) {
        GGMLHEXAGON_LOG_ERROR("HAP_power_set HVX failed");
        return -3;
    }

    /* Power up HMX with v2 settings for v75+ architecture */
#if __HVX_ARCH__ >= 75
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_HMX_v2;
    req.hmx_v2.set_power = 1;
    req.hmx_v2.power_up = 1;
    req.hmx_v2.set_clock = 1;
    req.hmx_v2.target_corner = HAP_DCVS_EXP_VCORNER_MAX;
    req.hmx_v2.min_corner = HAP_DCVS_EXP_VCORNER_MAX;
    req.hmx_v2.max_corner = HAP_DCVS_EXP_VCORNER_MAX;
    req.hmx_v2.perf_mode = HAP_CLK_PERF_HIGH;
    GGMLHEXAGON_LOG_INFO("Setting HMX clock with HMX_v2 for v75+ architecture");
    if (HAP_power_set((void *)&g_dsp_ctx->power_ctx, &req) != 0) {
        GGMLHEXAGON_LOG_ERROR("HAP_power_set HMX_v2 failed, continuing without HMX");
        return -4;
    }
#else
    /* Power up HMX (legacy for older architectures) */
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_HMX;
    req.hmx.power_up = 1;
    if (HAP_power_set((void *)&g_dsp_ctx->power_ctx, &req) != 0) {
        GGMLHEXAGON_LOG_ERROR("HAP_power_set HMX failed, continuing without HMX");
        return -4;
    }
#endif

    GGMLHEXAGON_LOG_INFO("HAP_power_set for HVX and HMX succeeded");
    return 0;
}

static AEEResult hap_probe_dsp(remote_handle64 h) {
    int retVal = 0;

    boolean dcvs_enabled;
    unsigned int max_mips       = 0;
    unsigned int max_bus_bw     = 0;
    int client_class            = 0;
    unsigned int clk_freq_hz    = 0;
    void * context_ptr          = NULL;

    HAP_power_response_t response;
    context_ptr = g_dsp_ctx->hexagon_power_ctx;

    memset(&response, 0, sizeof(HAP_power_response_t));
    response.type = HAP_power_get_max_mips;
    retVal = HAP_power_get(context_ptr, &response);
    if (retVal!=AEE_SUCCESS) {
        FARF(ERROR, "Unable to get the maximum MIPS supported");
        return AEE_EFAILED;
    }

    max_mips = response.max_mips;
    memset(&response, 0, sizeof(HAP_power_response_t));
    response.type = HAP_power_get_max_bus_bw;
    retVal = HAP_power_get(context_ptr, &response);
    if (retVal!=AEE_SUCCESS) {
        FARF(ERROR, "Unable to get the maximum bus bandwidth supported");
        return AEE_EFAILED;
    }

    max_bus_bw = response.max_bus_bw;
    memset(&response, 0, sizeof(HAP_power_response_t));
    response.type = HAP_power_get_client_class;
    retVal = HAP_power_get(context_ptr, &response);
    if (retVal!=AEE_SUCCESS) {
        FARF(ERROR, "Unable to get the client class");
        return AEE_EFAILED;
    }

    client_class = response.client_class;
    memset(&response, 0, sizeof(HAP_power_response_t));
    response.type = HAP_power_get_clk_Freq;
    retVal = HAP_power_get(context_ptr, &response);
    if (retVal!=AEE_SUCCESS) {
        FARF(ERROR, "Unable to get the DSP core clock frequency");
        return AEE_EFAILED;
    }

    clk_freq_hz = response.clkFreqHz;
    memset(&response, 0, sizeof(HAP_power_response_t));
    response.type = HAP_power_get_dcvsEnabled;
    retVal = HAP_power_get(context_ptr, &response);
    if (retVal!=AEE_SUCCESS) {
        FARF(ERROR, "Unable to get the DCVS status");
        return AEE_EFAILED;
    }

    dcvs_enabled = response.dcvsEnabled;
    printf("\nMaximum MIPS of DSP:             %u"
                 "\nMaximum Bus Bandwidth supported: %u Bytes/second(%u MiB/s)"
                 "\nClient Class:                    %x"
                 "\nCore clock frequency of the DSP: %u"
                 "\nDCVS status:                     %d\n\n",
                  max_mips, max_bus_bw, max_bus_bw >> 20, client_class, clk_freq_hz, dcvs_enabled);

    return AEE_SUCCESS;
}

// =================================================================================================
// Qualcomm compatibility layer (ported from Qualcomm's ggml-hexagon)
// =================================================================================================
// begin translation layer {

// Adapters for the old htp_mm_hvx vtcm_sizes API that was replaced by
// htp_mm_hvx_vtcm_layout_build in upstream htp/matmul-ops.h.
static inline size_t htp_mm_hvx_get_vtcm_sizes(
    int kernel_type, int wtype, uint32_t ne10, uint32_t src1_nrows,
    uint32_t n_threads,
    size_t dst_row_size, size_t src0_row_size, size_t src1_row_size,
    uint32_t n_prefetch,
    size_t * vtcm_src0_size, size_t * vtcm_src1_size, size_t * vtcm_dst_size
) {
    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kernel_type, wtype, ne10, src1_nrows, n_threads,
                                 dst_row_size, src0_row_size, src1_row_size, n_prefetch,
                                 false, false, false);
    *vtcm_src0_size = L.src0_bytes;
    *vtcm_src1_size = L.src1_bytes;
    *vtcm_dst_size  = L.dst_bytes;
    return L.total_bytes;
}

// ===========================================================================
// execute_op dispatch wrapper
// All op_xxx functions are exported from htp/*.c (non-static, declared in
// htp-ctx.h). We only need this dispatch wrapper + a translation layer.
//
// Function pointer table replaces the switch statement: eliminates branch
// prediction overhead (single indirect jump vs. chained conditional branches).
// Table is ~180 bytes, fits in L1I cache. Indexed directly by htp_op_code.
// ===========================================================================
typedef int (*htp_op_func_t)(struct htp_ops_context *);

static const htp_op_func_t g_op_dispatch[HTP_OP_INVALID] = {
    [HTP_OP_MUL]             = op_binary,
    [HTP_OP_ADD]             = op_binary,
    [HTP_OP_SUB]             = op_binary,
    [HTP_OP_DIV]             = op_binary,
    [HTP_OP_MUL_MAT]         = op_matmul,
    [HTP_OP_MUL_MAT_ID]      = op_matmul_id,
    [HTP_OP_MUL_MAT_QKV]     = op_matmul_qkv,
    [HTP_OP_MUL_MAT_FFN]     = op_matmul_ffn,
    [HTP_OP_MUL_MAT_ADD]     = op_matmul,
    [HTP_OP_NORM]            = op_unary,
    [HTP_OP_RMS_NORM]        = op_unary,
    [HTP_OP_RMS_NORM_MUL]    = op_unary,
    [HTP_OP_SCALE]           = op_unary,
    [HTP_OP_SQR]             = op_unary,
    [HTP_OP_SQRT]            = op_unary,
    [HTP_OP_UNARY_SOFTPLUS]  = op_unary,
    [HTP_OP_UNARY_SIGMOID]   = op_unary,
    [HTP_OP_UNARY_NEG]       = op_unary,
    [HTP_OP_UNARY_EXP]       = op_unary,
    [HTP_OP_UNARY_TANH]      = op_unary,
    [HTP_OP_L2_NORM]         = op_unary,
    [HTP_OP_UNARY_SILU]      = op_unary,
    [HTP_OP_UNARY_GELU]      = op_unary,
    [HTP_OP_GLU_SWIGLU]      = op_activations,
    [HTP_OP_GLU_SWIGLU_OAI]  = op_activations,
    [HTP_OP_GLU_GEGLU]       = op_activations,
    [HTP_OP_SOFTMAX]         = op_softmax,
    [HTP_OP_ADD_ID]          = op_binary,
    [HTP_OP_ROPE]            = op_rope,
    [HTP_OP_FLASH_ATTN_EXT]  = op_flash_attn_ext,
    [HTP_OP_SET_ROWS]        = op_set_rows,
    [HTP_OP_GET_ROWS]        = op_get_rows,
    [HTP_OP_SUM_ROWS]        = op_sum_rows,
    [HTP_OP_CPY]             = op_cpy,
    [HTP_OP_REPEAT]          = op_repeat,
    [HTP_OP_ARGSORT]         = op_argsort,
    [HTP_OP_SSM_CONV]        = op_ssm_conv,
    [HTP_OP_CUMSUM]          = op_cumsum,
    [HTP_OP_FILL]            = op_fill,
    [HTP_OP_DIAG]            = op_diag,
    [HTP_OP_SOLVE_TRI]       = op_solve_tri,
    [HTP_OP_PAD]             = op_pad,
    [HTP_OP_CONCAT]          = op_concat,
    [HTP_OP_IM2COL]          = op_im2col,
    [HTP_OP_GATED_DELTA_NET] = op_gated_delta_net,
    [HTP_OP_TRI]             = op_unary,
};

static int execute_op(struct htp_ops_context * octx) {
#if HEX_OP_PROF
    const uint64_t t0 = ggml_time_us();
#endif
    const unsigned int op = (unsigned int) octx->op;
    int ret;
    if (op < HTP_OP_INVALID) {
        htp_op_func_t fn = g_op_dispatch[op];
        if (fn) {
            ret = fn(octx);
        } else {
            FARF(ERROR, "Unknown Op %u", op);
            ret = -1;
        }
    } else {
        FARF(ERROR, "Unknown Op %u", op);
        ret = -1;
    }
#if HEX_OP_PROF
    {
        const uint64_t dt = ggml_time_us() - t0;
        if (op < HEX_OP_PROF_BUCKETS) {
            g_op_prof_dur_us[op] += dt;
            g_op_prof_count  [op] += 1;
            if (dt > g_op_prof_max_us[op]) g_op_prof_max_us[op] = dt;
            if (dt < g_op_prof_min_us[op]) g_op_prof_min_us[op] = dt;
        }
    }
#endif
    return ret;
}

// Convert hex_tensor_desc to dsptensor in-place. Avoids the memset+memcpy
// pattern that was ~200 ns per tensor. The hex_tensor_desc layout is:
//   {type, ne[4], nb[4], op_params[16], flags, data_offset, data_len}
// The dsptensor layout is:
//   {type, ne[4], nb[4], op, op_params[16], flags, data, data_len}
// The only difference is dsptensor has an extra 'op' field between nb and
// op_params, and data is a pointer instead of an offset.
static inline void hex_tensor_to_dsptensor(const hex_tensor_desc * ht,
                                            const char * ion_base,
                                            dsptensor * dt) {
    dt->type = ht->type;
    dt->ne[0] = ht->ne[0]; dt->ne[1] = ht->ne[1];
    dt->ne[2] = ht->ne[2]; dt->ne[3] = ht->ne[3];
    dt->nb[0] = ht->nb[0]; dt->nb[1] = ht->nb[1];
    dt->nb[2] = ht->nb[2]; dt->nb[3] = ht->nb[3];
    dt->op = 0;  // not used by execute_op path
    memcpy(dt->op_params, ht->op_params, sizeof(dt->op_params));
    dt->flags    = ht->flags;
    dt->data     = (void *)(ion_base + ht->data_offset);
    dt->data_len = ht->data_len;
}

// Convert hex_tensor_desc directly to htp_tensor. Eliminates the intermediate
// dsptensor step for op dispatch. Mirrors htp/main.c prep_tensor: data pointer
// is computed from mempool base + offset. flags is never read on the JZ path;
// cache coherency is handled by entry.c itself.
static inline void hex_tensor_to_htp_tensor(const hex_tensor_desc * ht,
                                             const char * ion_base,
                                             struct htp_tensor * htp) {
    htp->data  = (uint32_t)(uintptr_t)(ion_base + ht->data_offset);
    htp->size  = (uint32_t)ht->data_len;
    htp->flags = HTP_TENSOR_FLUSHED;
    htp->type  = (uint16_t)ht->type;
    htp->bi    = 0;
    htp->ne[0] = (uint32_t)ht->ne[0];
    htp->ne[1] = (uint32_t)ht->ne[1];
    htp->ne[2] = (uint32_t)ht->ne[2];
    htp->ne[3] = (uint32_t)ht->ne[3];
    htp->nb[0] = (uint32_t)ht->nb[0];
    htp->nb[1] = (uint32_t)ht->nb[1];
    htp->nb[2] = (uint32_t)ht->nb[2];
    htp->nb[3] = (uint32_t)ht->nb[3];
}

// Map GGML opcode to HTP opcode. Returns 0 on success, -1 if unsupported.
// For GGML_OP_UNARY, op_params[0] selects the unary sub-op.
static int ggml_op_to_htp_op(int32_t ggml_op, const int32_t * op_params,
                             enum htp_op_code * htp_op) {
    switch (ggml_op) {
        case GGML_OP_ADD:      *htp_op = HTP_OP_ADD;         return 0;
        case GGML_OP_SUB:      *htp_op = HTP_OP_SUB;         return 0;
        case GGML_OP_MUL:      *htp_op = HTP_OP_MUL;         return 0;
        case GGML_OP_DIV:      *htp_op = HTP_OP_DIV;         return 0;
        case GGML_OP_MUL_MAT:  *htp_op = HTP_OP_MUL_MAT;     return 0;
        case GGML_OP_RMS_NORM: *htp_op = HTP_OP_RMS_NORM;    return 0;
        case GGML_OP_ROPE:     *htp_op = HTP_OP_ROPE;        return 0;
        case GGML_OP_FLASH_ATTN_EXT: *htp_op = HTP_OP_FLASH_ATTN_EXT; return 0;
        case GGML_OP_SOFT_MAX: *htp_op = HTP_OP_SOFTMAX;     return 0;
        case GGML_OP_SCALE:   *htp_op = HTP_OP_SCALE;       return 0;
        case GGML_OP_CONCAT:  *htp_op = HTP_OP_CONCAT;      return 0;
        case GGML_OP_CPY:     *htp_op = HTP_OP_CPY;         return 0;
        case GGML_OP_GET_ROWS: *htp_op = HTP_OP_GET_ROWS;   return 0;
        case GGML_OP_SET_ROWS: *htp_op = HTP_OP_SET_ROWS;   return 0;
        case GGML_OP_SUM_ROWS: *htp_op = HTP_OP_SUM_ROWS;   return 0;
        case GGML_OP_CONT:    *htp_op = HTP_OP_CPY;         return 0;
        case GGML_OP_REPEAT:  *htp_op = HTP_OP_REPEAT;       return 0;
        case GGML_OP_NORM:    *htp_op = HTP_OP_NORM;        return 0;
        case GGML_OP_L2_NORM: *htp_op = HTP_OP_L2_NORM;     return 0;
        case GGML_OP_SQR:     *htp_op = HTP_OP_SQR;         return 0;
        case GGML_OP_SQRT:    *htp_op = HTP_OP_SQRT;        return 0;
        case GGML_OP_ARGSORT: *htp_op = HTP_OP_ARGSORT;     return 0;
        case GGML_OP_PAD:     *htp_op = HTP_OP_PAD;         return 0;
        case GGML_OP_IM2COL:  *htp_op = HTP_OP_IM2COL;      return 0;
        case GGML_OP_GATED_DELTA_NET: *htp_op = HTP_OP_GATED_DELTA_NET; return 0;
        case GGML_OP_CUMSUM:  *htp_op = HTP_OP_CUMSUM;      return 0;
        case GGML_OP_FILL:    *htp_op = HTP_OP_FILL;        return 0;
        case GGML_OP_DIAG:    *htp_op = HTP_OP_DIAG;        return 0;
        case GGML_OP_TRI:     *htp_op = HTP_OP_TRI;         return 0;
        case GGML_OP_UNARY: {
            if (!op_params) {
                FARF(ERROR, "ggml_op_to_htp_op: UNARY missing op_params");
                return -1;
            }
            switch (op_params[0]) {
                case GGML_UNARY_OP_NEG:      *htp_op = HTP_OP_UNARY_NEG;      return 0;
                case GGML_UNARY_OP_TANH:     *htp_op = HTP_OP_UNARY_TANH;     return 0;
                case GGML_UNARY_OP_SIGMOID:  *htp_op = HTP_OP_UNARY_SIGMOID;  return 0;
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK: *htp_op = HTP_OP_UNARY_GELU;     return 0;
                case GGML_UNARY_OP_SILU:      *htp_op = HTP_OP_UNARY_SILU;     return 0;
                case GGML_UNARY_OP_EXP:       *htp_op = HTP_OP_UNARY_EXP;      return 0;
                case GGML_UNARY_OP_SOFTPLUS: *htp_op = HTP_OP_UNARY_SOFTPLUS; return 0;
                default:
                    FARF(ERROR, "ggml_op_to_htp_op: unsupported unary_op %d", op_params[0]);
                    return -1;
            }
        }
        case GGML_OP_GLU: {
            if (!op_params) {
                FARF(ERROR, "ggml_op_to_htp_op: GLU missing op_params");
                return -1;
            }
            switch (op_params[0]) {
                case GGML_GLU_OP_SWIGLU:     *htp_op = HTP_OP_GLU_SWIGLU;     return 0;
                case GGML_GLU_OP_SWIGLU_OAI: *htp_op = HTP_OP_GLU_SWIGLU_OAI; return 0;
                case GGML_GLU_OP_GEGLU:      *htp_op = HTP_OP_GLU_GEGLU;      return 0;
                default:
                    FARF(ERROR, "ggml_op_to_htp_op: unsupported glu_op %d", op_params[0]);
                    return -1;
            }
        }
        default:
            FARF(ERROR, "ggml_op_to_htp_op: unsupported ggml_op %d", ggml_op);
            return -1;
    }
}

// Build htp_ops_context directly from pre-converted g_pre_ht tensors.
// Eliminates per-op dsptensor_to_htp_tensor() calls and stack-allocated
// src_ht/dst_ht arrays. Mirrors htp/main.c proc_op_req: direct tensor
// table indexing (tens + op->src[i]).
static void build_htp_octx(
    struct htp_ops_context * octx,
    enum htp_op_code htp_op,
    const int32_t * op_params,
    const int32_t * kernel_params,
    const int32_t src_idx[HTP_OP_MAX_INPUTS],
    const int32_t dst_idx[HTP_OP_MAX_OUTPUTS]) {

    // Zero only spad pointers and flags; execute_op reads these
    // unconditionally on some paths. Avoid full-struct memset which
    // is wasted on fields we immediately overwrite.
    octx->flags = 0;
    octx->src0_spad.src = NULL;
    octx->src1_spad.src = NULL;
    octx->src2_spad.src = NULL;
    octx->src3_spad.src = NULL;
    octx->dst_spad.src  = NULL;

    octx->ctx = g_dsp_ctx->htp_ctx;
    octx->op  = htp_op;
    memcpy(octx->op_params, op_params, sizeof(octx->op_params));
    if (kernel_params) {
        memcpy(octx->kernel_params, kernel_params, sizeof(octx->kernel_params));
    } else {
        memset(octx->kernel_params, 0, sizeof(octx->kernel_params));
    }

    for (int i = 0; i < HTP_OP_MAX_INPUTS; i++) {
        octx->src[i] = (src_idx[i] >= 0) ? &g_pre_ht[src_idx[i]] : NULL;
    }

    for (int i = 0; i < HTP_OP_MAX_OUTPUTS; i++) {
        octx->dsts[i] = (dst_idx[i] >= 0) ? &g_pre_ht[dst_idx[i]] : NULL;
    }

    octx->n_threads = (uint32_t)g_dsp_ctx->thread_counts;
}

// Try HMX precompute (simple 2D path). Mirrors ggml_hexagon_precompute_hmx_mm_params
// without the grouped batched path (we don't use MUL_MAT_ID).
// Returns true on success, false to fall back to HVX.
static bool build_mm_hmx_params(struct htp_ops_context * octx,
                                struct htp_mm_kernel_params * kparams) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];

    const int      wtype = src0->type;
    const uint32_t ne00  = src0->ne[0];
    const uint32_t ne01  = src0->ne[1];
    const uint32_t ne02  = src0->ne[2];
    const uint32_t ne03  = src0->ne[3];
    const uint32_t ne10  = src1->ne[0];
    const uint32_t ne11  = src1->ne[1];
    const uint32_t ne12  = src1->ne[2];
    const uint32_t ne13  = src1->ne[3];

    const bool is_repack = (wtype == HTP_TYPE_Q4_0 || wtype == HTP_TYPE_Q4_1 ||
                            wtype == HTP_TYPE_Q8_0 || wtype == HTP_TYPE_IQ4_NL ||
                            wtype == HTP_TYPE_MXFP4);
    const bool is_hmx_wtype = (wtype == HTP_TYPE_F16 || wtype == HTP_TYPE_F32 || is_repack);
    if (!is_hmx_wtype) return false;

    const bool is_batched = (ne02 * ne03 > 1 || ne12 * ne13 > 1);

    const int ne00_padded = is_repack ? hex_round_up(ne00, 32) : (int) ne00;
    const int ne01_padded = is_repack ? hex_round_up(ne01, 32) : (int) ne01;
    const int ne11_padded = hex_round_up(ne11, 32);

    // Eligibility (mirrors ggml_hexagon_matmul_is_hmx_eligible)
    if (ne01_padded % 32 != 0) return false;
    if (ne00 % 32 != 0) return false;
    if (is_batched && wtype != HTP_TYPE_F16) return false;
    if (src0->nb[0] > src0->nb[1] || src1->nb[0] > src1->nb[1]) return false;
    if (ne11 <= HTP_MM_HMX_MIN_NROWS) return false;

    const uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(wtype);
    const bool     pipeline          = htp_mm_hmx_pipeline(ne11);
    const int      n_threads         = (int) octx->n_threads;
    const size_t   vtcm_budget       = g_dsp_ctx->vtcm_size;

    size_t best_mblocks       = SIZE_MAX;
    int    best_act_threads   = 0;
    size_t best_m_chunk       = 0;
    size_t best_n_chunk       = 0;
    size_t best_vtcm_size     = 0;

    int act_threads = n_threads;
    while (act_threads >= 1) {
        const size_t act_f32_size = hex_align_up(
            (size_t) act_threads * HTP_MM_DMA_ACT_MULTIPLIER * ne00_padded * sizeof(float),
            HTP_MM_HMX_TILE_SIZE);
        const size_t overhead = 256 + act_f32_size;

        size_t cost_n = 0, cost_m = 0, cost_mn = 0;
        htp_mm_hmx_get_2d_chunk_costs(wtype, ne00_padded, pipeline, aligned_tile_size,
                                      &cost_n, &cost_m, &cost_mn);

        size_t m_chunk_cand = 0, n_chunk_cand = 0, vtcm_size_cand = 0;
        if (htp_mm_hmx_compute_chunks(vtcm_budget, overhead, cost_n, cost_m, cost_mn,
                                      (size_t) ne11_padded, (size_t) ne01_padded,
                                      (size_t) ne01_padded * HTP_MM_HMX_COST_W_DEQUANT,
                                      (size_t) ne11 * HTP_MM_HMX_COST_A_CONVERT,
                                      &m_chunk_cand, &n_chunk_cand, &vtcm_size_cand) == 0) {
            size_t exact_size = htp_mm_hmx_get_2d_vtcm_size(
                wtype, ne00_padded, m_chunk_cand, n_chunk_cand, pipeline,
                act_threads, aligned_tile_size);
            if (exact_size <= vtcm_budget) {
                size_t mblocks = ((size_t) ne11 + m_chunk_cand - 1) / m_chunk_cand;
                if (mblocks < best_mblocks ||
                    (mblocks == best_mblocks && act_threads > best_act_threads)) {
                    best_mblocks     = mblocks;
                    best_act_threads = act_threads;
                    best_m_chunk     = m_chunk_cand;
                    best_n_chunk     = n_chunk_cand;
                    best_vtcm_size   = exact_size;
                }
            }
        }
        if (act_threads == 1) break;
        act_threads /= 2;
    }

    if (best_act_threads == 0) return false;

    kparams->n_hmx             = 1;
    kparams->pipeline           = pipeline ? 1 : 0;
    kparams->m_chunk            = (int32_t) best_m_chunk;
    kparams->n_chunk            = (int32_t) best_n_chunk;
    kparams->n_threads          = n_threads;
    kparams->n_act_threads      = best_act_threads;
    kparams->tile_size          = (int32_t) htp_mm_get_weight_tile_size(wtype);
    kparams->aligned_tile_size  = (int32_t) aligned_tile_size;
    kparams->src1_row_size      = (int32_t)((wtype == HTP_TYPE_Q4_1)
                                            ? htp_mm_q8_1_tiled_row_size(ne10)
                                            : htp_mm_q8_0_tiled_row_size(ne10));
    kparams->vtcm_size          = (int32_t) best_vtcm_size;
    kparams->vtcm_src0_size     = 0;
    kparams->vtcm_src1_size     = 0;
    kparams->vtcm_dst_size      = 0;
    kparams->n_prefetch         = 16;
    kparams->kernel_type        = is_batched ? HTP_MM_KERNEL_HMX_F16_BATCHED
                                             : HTP_MM_KERNEL_HMX_2D;
    // Used by HMX kernels for activation DMA/work split (matmul-ops.c).
    kparams->div_n_act_threads  = init_fastdiv_values((uint32_t) best_act_threads);
    kparams->div_ne00_padded    = init_fastdiv_values((uint32_t) ne00_padded);
    return true;
}

// Compute htp_mm_kernel_params on DSP side for MUL_MAT.
// Tries HMX first (if available), falls back to HVX F32/F16/quantized paths.
static int build_mm_kernel_params(struct htp_ops_context * octx) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;
    if (!src0 || !src1 || !dst) return -1;

    struct htp_mm_kernel_params * kparams =
        (struct htp_mm_kernel_params *) octx->kernel_params;

    // If AP side already precomputed kernel params (kernel_type != 0),
    // skip DSP-side recomputation. The AP side uses the same HMX-first
    // then HVX-fallback policy (ggml_hexagon_precompute_matmul_params)
    // as this function, and may have precomputed HMX chunk sizes that
    // the DSP-side build_mm_hmx_params would otherwise recompute.
    if (kparams->kernel_type != 0) {
        return 0;
    }

    memset(kparams, 0, sizeof(*kparams));

    const int wtype = src0->type;
    const uint32_t ne02 = src0->ne[2];
    const uint32_t ne03 = src0->ne[3];
    const uint32_t ne10 = src1->ne[0];
    const uint32_t ne11 = src1->ne[1];
    const uint32_t ne12 = src1->ne[2];
    const uint32_t ne13 = src1->ne[3];
    const uint32_t src1_nrows = ne11 * ne12 * ne13;

    kparams->n_hmx       = 0;
    kparams->n_threads   = octx->n_threads;
    kparams->n_prefetch  = 16;

    // Try HMX first (mirrors ggml_hexagon_precompute_matmul_params: HMX-first, HVX-fallback)
    if (g_dsp_ctx->hmx_available && build_mm_hmx_params(octx, kparams)) {
        goto mm_finalize;
    }

    const bool is_batched  = (ne02 > 1) || (ne03 > 1);
    const bool is_permuted = (src0->nb[0] > src0->nb[1] || src0->nb[1] > src0->nb[2] || src0->nb[2] > src0->nb[3]) ||
                             (src1->nb[0] > src1->nb[1] || src1->nb[1] > src1->nb[2] || src1->nb[2] > src1->nb[3]);

    size_t vtcm_src0_size = 0, vtcm_src1_size = 0, vtcm_dst_size = 0;

    if (wtype == HTP_TYPE_F32) {
        size_t vtcm_size = htp_mm_hvx_get_vtcm_sizes(
            HTP_MM_KERNEL_HVX_F32_F32_VTCM, wtype, ne10, src1_nrows, octx->n_threads,
            dst->nb[1], src0->nb[1], src1->nb[1], 16,
            &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size);

        if (!is_batched && !is_permuted && vtcm_size <= g_dsp_ctx->vtcm_size) {
            kparams->kernel_type    = HTP_MM_KERNEL_HVX_F32_F32_VTCM;
            kparams->src1_row_size  = hex_round_up(ne10 * 4, 128);
        } else {
            kparams->kernel_type    = HTP_MM_KERNEL_HVX_F32_F32_DDR;
            kparams->src1_row_size  = src1->nb[1];
            vtcm_size = htp_mm_hvx_get_vtcm_sizes(
                kparams->kernel_type, wtype, ne10, src1_nrows, octx->n_threads,
                dst->nb[1], src0->nb[1], src1->nb[1], 16,
                &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size);
        }
        kparams->vtcm_size      = (int32_t) vtcm_size;
        kparams->vtcm_src0_size = (int32_t) vtcm_src0_size;
        kparams->vtcm_src1_size = (int32_t) vtcm_src1_size;
        kparams->vtcm_dst_size  = (int32_t) vtcm_dst_size;
    } else if (wtype == HTP_TYPE_F16) {
        size_t vtcm_size = htp_mm_hvx_get_vtcm_sizes(
            HTP_MM_KERNEL_HVX_F16_F16_VTCM, wtype, ne10, src1_nrows, octx->n_threads,
            dst->nb[1], src0->nb[1], src1->nb[1], 16,
            &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size);

        if (!is_batched && !is_permuted && vtcm_size <= g_dsp_ctx->vtcm_size) {
            kparams->kernel_type    = HTP_MM_KERNEL_HVX_F16_F16_VTCM;
            kparams->src1_row_size  = hex_round_up(ne10 * 2, 128);
        } else {
            if (src1->type == HTP_TYPE_F32) {
                kparams->kernel_type = HTP_MM_KERNEL_HVX_F16_F32_DDR;
            } else {
                kparams->kernel_type = HTP_MM_KERNEL_HVX_F16_F16_DDR;
            }
            kparams->src1_row_size  = src1->nb[1];
            vtcm_size = htp_mm_hvx_get_vtcm_sizes(
                kparams->kernel_type, wtype, ne10, src1_nrows, octx->n_threads,
                dst->nb[1], src0->nb[1], src1->nb[1], 16,
                &vtcm_src0_size, &vtcm_src1_size, &vtcm_dst_size);
        }
        kparams->vtcm_size      = (int32_t) vtcm_size;
        kparams->vtcm_src0_size = (int32_t) vtcm_src0_size;
        kparams->vtcm_src1_size = (int32_t) vtcm_src1_size;
        kparams->vtcm_dst_size  = (int32_t) vtcm_dst_size;
    } else {
        // Quantized HVX path (Q4_0, Q4_1, Q5_0, Q8_0, IQ4_NL, MXFP4)
        kparams->tile_size         = (int32_t) htp_mm_get_weight_tile_size(wtype);
        kparams->aligned_tile_size = (int32_t) htp_mm_get_weight_aligned_tile_size(wtype);

        const bool k_align   = (ne10 % 32 == 0);
        const bool try_tiled = k_align && kparams->tile_size > 0;
        bool tiled_ok = false;

        if (try_tiled) {
            kparams->src1_row_size = (int32_t)((wtype == HTP_TYPE_Q4_1)
                ? htp_mm_q8_1_tiled_row_size(ne10)
                : htp_mm_q8_0_tiled_row_size(ne10));
            kparams->kernel_type = (src1_nrows < octx->n_threads)
                ? HTP_MM_KERNEL_HVX_QUANT_BLOCK
                : HTP_MM_KERNEL_HVX_QUANT_ROW;

            const uint32_t max_prefetch = (src1_nrows > HTP_MM_HMX_MIN_NROWS) ? 2 : 16;
            uint32_t best_n_prefetch = 2;
            size_t vs0 = 0, vs1 = 0, vd = 0;
            size_t total_size = 0;
            for (uint32_t d = max_prefetch; d >= 2; d /= 2) {
                total_size = htp_mm_hvx_get_vtcm_sizes(
                    kparams->kernel_type, wtype, ne10, src1_nrows, octx->n_threads,
                    dst->nb[1], src0->nb[1], src1->nb[1], d,
                    &vs0, &vs1, &vd);
                if (total_size <= g_dsp_ctx->vtcm_size) {
                    best_n_prefetch = d;
                    break;
                }
            }
            if (best_n_prefetch == 2 && total_size > g_dsp_ctx->vtcm_size) {
                total_size = htp_mm_hvx_get_vtcm_sizes(
                    kparams->kernel_type, wtype, ne10, src1_nrows, octx->n_threads,
                    dst->nb[1], src0->nb[1], src1->nb[1], 2,
                    &vs0, &vs1, &vd);
            }
            kparams->n_prefetch = (int32_t) best_n_prefetch;

            if (total_size <= g_dsp_ctx->vtcm_size) {
                kparams->vtcm_size      = (int32_t) total_size;
                kparams->vtcm_src0_size = (int32_t) vs0;
                kparams->vtcm_src1_size = (int32_t) vs1;
                kparams->vtcm_dst_size  = (int32_t) vd;
                tiled_ok = true;
            }
        }

        if (!tiled_ok) {
            kparams->src1_row_size = (int32_t)((wtype == HTP_TYPE_Q4_1)
                ? htp_mm_q8_1_flat_row_size(ne10)
                : htp_mm_q8_0_flat_row_size(ne10));
            kparams->kernel_type = HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT;

            size_t vs0 = 0, vs1 = 0, vd = 0;
            const size_t total_size = htp_mm_hvx_get_vtcm_sizes(
                kparams->kernel_type, wtype, ne10, src1_nrows, octx->n_threads,
                dst->nb[1], src0->nb[1], src1->nb[1], 16,
                &vs0, &vs1, &vd);

            kparams->n_prefetch     = 16;
            kparams->vtcm_size      = (int32_t) total_size;
            kparams->vtcm_src0_size = (int32_t) vs0;
            kparams->vtcm_src1_size = (int32_t) vs1;
            kparams->vtcm_dst_size  = (int32_t) vd;
        }
    }

mm_finalize:
    kparams->div_ne12_ne1 = init_fastdiv_values(ne12 * ne11);
    kparams->div_ne1      = init_fastdiv_values(ne11);
    kparams->div_r2       = init_fastdiv_values(ne02 > 0 ? ne12 / ne02 : 1);
    kparams->div_r3       = init_fastdiv_values(ne03 > 0 ? ne13 / ne03 : 1);
    kparams->div_ne11     = init_fastdiv_values(ne11);

    return 0;
}

// Build htp_fa_kernel_params on DSP side for FLASH_ATTN_EXT.
// Mirrors ggml_hexagon_precompute_flash_attn_params on AP side, using
// DSP-side globals (g_dsp_ctx->vtcm_size, g_dsp_ctx->thread_counts, g_dsp_ctx->hmx_available).
static int build_fa_kernel_params(struct htp_ops_context * octx) {
    const struct htp_tensor * q  = octx->src[0];
    const struct htp_tensor * k  = octx->src[1];
    const struct htp_tensor * v  = octx->src[2];
    const struct htp_tensor * mask = octx->src[3];
    const struct htp_tensor * dst = octx->dst;
    if (!q || !k || !v || !dst) return -1;
    FARF(ALWAYS, "build_fa: DK=%u DV=%u neq1=%u nek1=%u G=%u ktype=%d vtype=%d hmx=%d",
         q->ne[0], v->ne[0], q->ne[1], k->ne[1], q->ne[2]/k->ne[2],
         k->type, v->type, g_dsp_ctx->hmx_available);

    struct htp_fa_kernel_params * kparams =
        (struct htp_fa_kernel_params *) octx->kernel_params;
    memset(kparams, 0, sizeof(*kparams));

    const uint32_t DK = q->ne[0];
    const uint32_t DV = v->ne[0];
    const uint32_t neq1 = q->ne[1];
    const uint32_t nek1 = k->ne[1];
    const uint32_t n_kv_heads = k->ne[2];
    const uint32_t G = q->ne[2] / n_kv_heads;

    float scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&scale,         &octx->op_params[0], sizeof(float));
    memcpy(&max_bias,      &octx->op_params[1], sizeof(float));
    memcpy(&logit_softcap, &octx->op_params[2], sizeof(float));
    if (logit_softcap != 0.0f) scale /= logit_softcap;

    kparams->scale         = scale;
    kparams->max_bias      = max_bias;
    kparams->logit_softcap = logit_softcap;
    kparams->is_q_fp32     = (q->type == HTP_TYPE_F32) ? 1 : 0;
    kparams->is_dst_fp32   = (dst->type == HTP_TYPE_F32) ? 1 : 0;
    kparams->G             = G;

    // ALiBi: find largest power of 2 <= n_head, then compute slope bases.
    // AP uses std::pow(2, -x); here we use 2^x = exp(x * ln2) to avoid powf.
    // Always computed (matches AP): when max_bias = 0, m0 = m1 = 1.0.
    const float ln2 = 0.6931471805599453f;
    uint32_t n_head_log2 = 1;
    while (n_head_log2 * 2u <= q->ne[2]) n_head_log2 *= 2;
    kparams->n_head_log2 = n_head_log2;
    kparams->m0 = expf(-ln2 * max_bias / (float)n_head_log2);
    kparams->m1 = expf(-ln2 * (max_bias * 0.5f) / (float)n_head_log2);

    // HMX eligibility: k/v F16, DK/DV divisible by 64, enough tokens.
    bool hmx_eligible = false;
    if (g_dsp_ctx->hmx_available && k->type == HTP_TYPE_F16 && v->type == HTP_TYPE_F16) {
        if (DK % 64 == 0 && DV % 64 == 0 && !(DK <= 128 && neq1 < 5)) {
            hmx_eligible = true;
        }
    }

    if (hmx_eligible) {
        size_t Br = 0, Bc = 0;
        int ret = hmx_fa_find_chunk_size(&Br, &Bc, G, DK, DV, neq1, nek1,
                                         g_dsp_ctx->vtcm_size, g_dsp_ctx->thread_counts);
        if (ret == 0) {
            kparams->kernel_type = HTP_FA_KERNEL_HMX;
            kparams->Br          = (uint16_t)Br;
            kparams->Bc          = (uint16_t)Bc;
            kparams->n_kv_blocks = (uint16_t)((nek1 + Bc - 1) / Bc);
            kparams->n_threads   = (kparams->n_kv_blocks >= 3 && g_dsp_ctx->thread_counts >= 2)
                                    ? (uint8_t)g_dsp_ctx->thread_counts : 1;
            kparams->u.hmx.g_br      = hex_align_up(G * Br, 32);
            kparams->u.hmx.pipeline  = (kparams->n_kv_blocks >= 3 && g_dsp_ctx->thread_counts >= 2) ? 1 : 0;
            kparams->vtcm_size       = (uint32_t)hmx_fa_compute_vtcm_usage(
                G, DK, DV, Br, Bc, kparams->n_threads, kparams->u.hmx.pipeline != 0);

            const size_t row_vec_bytes = hex_align_up(Bc * sizeof(uint16_t), 256);
            kparams->u.hmx.row_buf_stride = row_vec_bytes / 128;
            const size_t m_line_bytes = hex_align_up(Bc * sizeof(uint16_t), 128);
            kparams->u.hmx.mask_buf_row_stride = m_line_bytes / sizeof(uint16_t);
            kparams->u.hmx.mask_broadcast = (mask && mask->ne[2] == 1) ? 1 : 0;
            kparams->u.hmx.div_G = init_fastdiv_values(G);
            if (mask) {
                kparams->src3_div2 = init_fastdiv_values(mask->ne[2]);
                kparams->src3_div3 = init_fastdiv_values(mask->ne[3]);
            }
            kparams->qrows = 0;
            kparams->qrows_per_thread = 0;
            return 0;
        }
    }

    // Fallback to HVX
    kparams->kernel_type    = HTP_FA_KERNEL_HVX;
    kparams->Br             = 1;
    kparams->Bc             = 64;
    kparams->n_kv_blocks    = (uint16_t)((k->ne[1] + 64 - 1) / 64);
    kparams->n_threads      = (uint8_t)g_dsp_ctx->thread_counts;
    kparams->vtcm_size      = (uint32_t)hvx_fa_compute_vtcm_usage(
        DK, DV, kparams->is_q_fp32 != 0, mask != NULL, g_dsp_ctx->thread_counts);

    kparams->u.hvx.size_q_row_padded = hex_round_up(q->ne[0] * (kparams->is_q_fp32 ? 4 : 2), 128);
    kparams->u.hvx.size_k_row_padded = hex_round_up(k->ne[0] * 2, 128);
    kparams->u.hvx.size_v_row_padded = hex_round_up(v->ne[0] * 2, 128);
    kparams->u.hvx.src0_div21     = init_fastdiv_values(q->ne[2] * q->ne[1]);
    kparams->u.hvx.src0_div1      = init_fastdiv_values(q->ne[1]);
    kparams->broadcast_rk2   = init_fastdiv_values(q->ne[2] / k->ne[2]);
    kparams->broadcast_rk3   = init_fastdiv_values(q->ne[3] / k->ne[3]);
    kparams->broadcast_rv2   = init_fastdiv_values(q->ne[2] / v->ne[2]);
    kparams->broadcast_rv3   = init_fastdiv_values(q->ne[3] / v->ne[3]);
    if (mask) {
        kparams->src3_div2 = init_fastdiv_values(mask->ne[2]);
        kparams->src3_div3 = init_fastdiv_values(mask->ne[3]);
    }
    kparams->qrows           = q->ne[1] * q->ne[2] * q->ne[3];
    kparams->qrows_per_thread = (kparams->qrows + g_dsp_ctx->thread_counts - 1) / g_dsp_ctx->thread_counts;
    return 0;
}
// end translation layer }


// =================================================================================================
// IDL implementation
// =================================================================================================
int ggml_dsp_open(const char * uri, remote_handle64 * handle) {
    struct dsp_context * ctx = NULL;

    // Guard against double initialization
    if (g_dsp_ctx != NULL) {
        GGMLHEXAGON_LOG_ERROR("ggml_dsp_open: g_dsp_ctx already initialized");
        return AEE_EITEMBUSY;
    }

    ctx = (struct dsp_context *)calloc(1, sizeof(struct dsp_context));
    GGML_ASSERT(NULL != ctx);
    ctx->thread_counts      = 4;
    ctx->htp_ctx = (struct htp_context *)calloc(1, sizeof(struct htp_context));
    GGML_ASSERT(NULL != ctx->htp_ctx);
    g_dsp_ctx = ctx;
    g_dsp_ctx->thread_prio = qurt_thread_get_priority(qurt_thread_get_id());
    *handle = (remote_handle64)ctx;

    // Reset first-touch invalidate bitmap so a fresh session starts with
    // no weight regions marked. The bitmap is session-scoped: AP will
    // re-mirror weights at model load and their ION offsets are stable
    // for the lifetime of this session.
    weight_inval_reset_all();
    // Default to 0: no DSP-side cache optimizations beyond the baseline
    // first-touch weight bitmap. AP will push the configured bitmask via
    // execute_batch(0xFFFC) right after ggml_dsp_open returns. Until then,
    // every code path that consults dsp_cache_mode sees 0 and behaves like
    // baseline 29c1cf196.
    g_dsp_ctx->dsp_cache_mode = 0;
    g_dsp_ctx->dsp_cache_trace_bit0 = 0;  // default off; AP pushes via 0xFFFC bit 16
    g_dsp_ctx->dsp_cache_trace_bit1 = 0;  // default off; AP pushes via 0xFFFC bit 17

    printf("uri %s\n", uri);

    unsigned int api_version = qurt_api_version();
    printf("qurt_api_version            = 0x%x\n", api_version);
    printf("qurt_hvx_units              = 0x%d\n", qurt_hvx_get_units());
    qurt_arch_version_t  vers;
    qurt_sysenv_get_arch_version(&vers);
    printf("qurt_arch_version           = 0x%x\n", vers.arch_version);
    qurt_sysenv_app_heap_t aheap;
    qurt_sysenv_get_app_heap(&aheap);
    printf("aheap.heap_base=0x%x, aheap.heap_limit=0x%x\n", aheap.heap_base, aheap.heap_limit);
    qurt_sysenv_max_hthreads_t mhwt;
    qurt_sysenv_get_max_hw_threads(&mhwt);
    printf("qurt_hardware_thread_counts = %d\n", mhwt.max_hthreads);
    g_dsp_ctx->max_hw_threads = mhwt.max_hthreads;
    g_dsp_ctx->thread_counts = mhwt.max_hthreads;

    /* Step 1: Power up HVX and HMX */
    int power_result = power_on_hvx_hmx();
    if (power_result != 0) {
        printf("power_on_hvx_hmx failed (%d), continuing without HMX\n", power_result);
        g_dsp_ctx->hmx_available = 0;
    } else {
        g_dsp_ctx->hmx_available = 1;
    }

    /* Step 2: Query VTCM size and allocate resources */
    unsigned int vtcm_size_query = 0;
    unsigned int availBlockSize;
    compute_res_vtcm_page_t availBlock;
    compute_res_vtcm_page_t totalBlock;
    int result = 0;
    result = HAP_compute_res_query_VTCM(0, &vtcm_size_query, &totalBlock, &availBlockSize, &availBlock);
    GGMLHEXAGON_LOG_INFO("VTCM total = %u bytes\n", vtcm_size_query);
    printf("Querying VTCM before acquiring resources:\n");
    printf("Compute resource query return %d, vtcm_size_query %d, availBlockSize %d\n",
                                 result, vtcm_size_query, availBlockSize);
    printf("Compute resource query ctd, valid page sizes in total table: %d, valid page sizes in avail table: %d\n",
                                 totalBlock.page_list_len, availBlock.page_list_len);
    if (totalBlock.page_list_len >= 1 && availBlock.page_list_len >= 2) {
        printf("Compute resource query ctd, (Size, num pages); total (0x%x, %d) Avail (0x%x, %d, 0x%x, %d)\n",
                                    totalBlock.page_list[0].page_size,
                                    totalBlock.page_list[0].num_pages,
                                    availBlock.page_list[0].page_size,
                                    availBlock.page_list[0].num_pages,
                                    availBlock.page_list[1].page_size,
                                    availBlock.page_list[1].num_pages);
    } else if (totalBlock.page_list_len >= 1 && availBlock.page_list_len >= 1) {
        printf("Compute resource query ctd, (Size, num pages); total (0x%x, %d) Avail (0x%x, %d)\n",
                                    totalBlock.page_list[0].page_size,
                                    totalBlock.page_list[0].num_pages,
                                    availBlock.page_list[0].page_size,
                                    availBlock.page_list[0].num_pages);
    } else {
        printf("Compute resource query ctd, no page list data available\n");
    }

    /* Step 3: Acquire compute resources (including VTCM and HMX) */
    compute_res_attr_t attr;
    unsigned int vtcm_size_to_use = (DEFAULT_VTCM_SIZE < vtcm_size_query) ? DEFAULT_VTCM_SIZE : vtcm_size_query;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_serialize(&attr, 0);
    HAP_compute_res_attr_set_cache_mode(&attr, 1);  // Enable cache mode (matching official implementation)
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, vtcm_size_to_use, vtcm_size_to_use, vtcm_size_to_use); // single page (matching official implementation)
    HAP_compute_res_attr_set_release_callback(&attr, vtcm_release_callback, NULL);  // Enable release callback for cache mode
    HAP_compute_res_attr_set_hmx_param(&attr, 1);
    // Allocate VTCM for scratch pads
    g_dsp_ctx->compute_res_ctx_id = HAP_compute_res_acquire(&attr, 1000000);
    if (g_dsp_ctx->compute_res_ctx_id == 0) {
        GGMLHEXAGON_LOG_ERROR("HAP_compute_res_acquire failed, no VTCM available\n");
    } else {
        /* Using VTCM acquired via HAP_compute_res */
        void * vtcm_ptr = NULL;
        unsigned int vtcm_ptr_size = 0;
        if (HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &vtcm_ptr, &vtcm_ptr_size) != 0) {
            GGMLHEXAGON_LOG_INFO("HAP_compute_res_attr_get_vtcm_ptr_v2 failed\n");
            HAP_compute_res_release(g_dsp_ctx->compute_res_ctx_id);
            g_dsp_ctx->compute_res_ctx_id = 0;
        } else {
            g_dsp_ctx->vtcm_base = vtcm_ptr;
            g_dsp_ctx->vtcm_size = vtcm_ptr_size;
            GGMLHEXAGON_LOG_INFO("allocated VTCM pool via compute_res: %zu bytes at %p\n", g_dsp_ctx->vtcm_size, g_dsp_ctx->vtcm_base);

            /* VTCM: acquire once for the whole session (matches
             * htp/main.c htp_packet_callback pattern: acquire once at
             * session start, release once at session end).
             * Avoids per-batch HAP_compute_res_acquire/release_cached churn
             * (~8700 calls/session -> 2 calls/session) and the SDK's
             * adsprpc FARF log noise. */
            dsp_vtcm_acquire();
        }
    }

    /* Step 4: Create async HMX queue for pipeline overlap (DMA/HVX/HMX).
     * New Qualcomm API (b2dd28a3b) requires a pre-allocated backing buffer;
     * we memalign it here and track it in dsp_context.hmx_queue_buf for
     * cleanup in ggml_dsp_close. Capacity/stack_size match main.c defaults. */
    if (g_dsp_ctx->hmx_available && g_dsp_ctx->compute_res_ctx_id != 0) {
        if (g_dsp_ctx->hmx_queue != NULL) {
            GGMLHEXAGON_LOG_INFO("hmx_queue already exists, deleting old one\n");
            hmx_queue_free(g_dsp_ctx->hmx_queue);
            free(g_dsp_ctx->hmx_queue_buf);
            g_dsp_ctx->hmx_queue     = NULL;
            g_dsp_ctx->hmx_queue_buf = NULL;
        }
        size_t hmx_size  = hmx_queue_sizeof(JZ_HMX_QUEUE_CAPACITY, JZ_HMX_QUEUE_STACK_SIZE);
        size_t hmx_align = hmx_queue_alignof();
        void * hmx_buf   = memalign(hmx_align, hmx_size);
        if (hmx_buf) {
            // Trace slot mirrors htp/main.c (&ctx->trace[HTP_MAX_NTHREADS]): must be a
            // valid (zeroed) htp_thread_trace, htp_trace_event_start/stop dereference
            // it unconditionally on every HMX descriptor completion.
            g_dsp_ctx->hmx_queue = hmx_queue_init(hmx_buf, JZ_HMX_QUEUE_CAPACITY,
                                                  JZ_HMX_QUEUE_STACK_SIZE,
                                                  g_dsp_ctx->compute_res_ctx_id,
                                                  &g_dsp_ctx->htp_ctx->trace[HTP_MAX_NTHREADS]);
            if (g_dsp_ctx->hmx_queue) {
                g_dsp_ctx->hmx_queue_buf = hmx_buf;
                GGMLHEXAGON_LOG_INFO("async HMX queue created (capacity %u, rctx %u)\n",
                                     hmx_queue_capacity(g_dsp_ctx->hmx_queue), g_dsp_ctx->compute_res_ctx_id);
            } else {
                free(hmx_buf);
                g_dsp_ctx->hmx_queue_buf = NULL;
                GGMLHEXAGON_LOG_INFO("hmx_queue_init failed, HMX path will run synchronously\n");
            }
        } else {
            GGMLHEXAGON_LOG_INFO("memalign for hmx_queue failed, HMX path will run synchronously\n");
        }
    } else {
        GGMLHEXAGON_LOG_INFO("HMX not available (hmx=%d, rctx=%u), skipping hmx_queue creation\n",
                             g_dsp_ctx->hmx_available, g_dsp_ctx->compute_res_ctx_id);
    }

    /* Step 5: probe DSP memory for information only (no allocation) */
    {
        struct HAP_mem_stats mem_stats;
        memset(&mem_stats, 0, sizeof(mem_stats));
        int ret = HAP_mem_get_stats(&mem_stats);
        if (ret == 0) {
            printf("DSP HAP_mem_stats: bytes_free=%llu, bytes_used=%llu, seg_free=%llu, seg_used=%llu\n",
                 (unsigned long long)mem_stats.bytes_free, (unsigned long long)mem_stats.bytes_used,
                 (unsigned long long)mem_stats.seg_free, (unsigned long long)mem_stats.seg_used);
        } else {
            printf("HAP_mem_get_stats failed: %d\n", ret);
        }

        // Probe available DSP heap (information only, no allocation)
        size_t max_avail_mb = 0;
        for (int mb = 2048; mb >= 16; mb -= 16) {
            void * ptr = malloc((size_t)mb * 1024 * 1024);
            if (ptr) {
                printf("DSP malloc probe: %d MB succeeded at %p\n", mb, ptr);
                free(ptr);
                max_avail_mb = mb;
                break;
            }
        }
        if (max_avail_mb == 0) {
            printf("DSP malloc probe: even 16 MB failed!\n");
        } else {
            printf("DSP malloc probe: max available = %zu MB (for work data only, cache uses ION)\n", max_avail_mb);
        }
    }

    return 0;
}

int ggml_dsp_close(remote_handle64 handle) {
    struct dsp_context * ctx = (struct dsp_context *)handle;
    if (!ctx) return 0;

    // Cleanup htp_context resources (work_queue + dma queues).
    // New Qualcomm API (b2dd28a3b): *_queue_free() does not free the backing
    // buffer, so we track them in dsp_context and free them explicitly here.
    if (ctx->htp_ctx) {
        if (ctx->htp_ctx->work_queue) {
            work_queue_free(ctx->htp_ctx->work_queue);
            ctx->htp_ctx->work_queue = NULL;
        }
        if (ctx->work_queue_buf) {
            free(ctx->work_queue_buf);
            ctx->work_queue_buf = NULL;
        }
        for (int i = 0; i < HTP_MAX_NTHREADS; i++) {
            if (ctx->htp_ctx->dma[i]) {
                dma_queue_alias_free(ctx->htp_ctx->dma[i]);
                ctx->htp_ctx->dma[i] = NULL;
            }
            if (ctx->dma_alias_bufs[i]) {
                free(ctx->dma_alias_bufs[i]);
                ctx->dma_alias_bufs[i] = NULL;
            }
            if (ctx->htp_ctx->dma_cached[i]) {
                dma_queue_free(ctx->htp_ctx->dma_cached[i]);
                ctx->htp_ctx->dma_cached[i] = NULL;
            }
            if (ctx->dma_queue_bufs[i]) {
                free(ctx->dma_queue_bufs[i]);
                ctx->dma_queue_bufs[i] = NULL;
            }
        }
        free(ctx->htp_ctx);
        ctx->htp_ctx = NULL;
    }

    if (ctx->hmx_queue != NULL) {
        hmx_queue_free(ctx->hmx_queue);
        ctx->hmx_queue = NULL;
        if (ctx->hmx_queue_buf) {
            free(ctx->hmx_queue_buf);
            ctx->hmx_queue_buf = NULL;
        }
        GGMLHEXAGON_LOG_INFO("released async HMX queue");
    }

    /* VTCM: release once at session end (matches the pattern, see
     * ggml_dsp_open). The release callback is still registered and will
     * set vtcm_needs_release=1 if another session preempts during the
     * session, but we intentionally ignore it here so that VTCM stays
     * cached for the full session. */
    dsp_vtcm_release();

    if (ctx->compute_res_ctx_id != 0) {
        // HAP_compute_res_release_cached is already called inside dsp_vtcm_release()
        // NOTE: HMX lock is managed per-operation in mulmat.c, not here
        // HAP_compute_res_hmx_unlock(ctx->compute_res_ctx_id);

        HAP_compute_res_release(ctx->compute_res_ctx_id);
        ctx->compute_res_ctx_id = 0;
        ctx->vtcm_base = NULL;
        ctx->vtcm_size = 0;
        GGMLHEXAGON_LOG_INFO("released compute resources");
    }

    g_dsp_ctx = NULL;
    free(ctx);
    return 0;
}

AEEResult ggml_dsp_setclocks(remote_handle64 handle, int32 diag_info, int32 requested_thread_counts, int32 * actual_thread_counts) {
    /* Reserve 2 hw thread slots: one for the hmx_queue thread, one for
     * FastRPC listener/system activity. An op needs requested_thread_counts+1
     * co-resident threads; oversubscribing deadlocks because QuRT does
     * not preempt equal-priority workers spinning in hex_pause, so the
     * unscheduled worker never decrements the task barrier (observed on
     * v75/8Gen3: max_hthreads=6, requested_thread_counts=6 hangs; 5 is flaky). */
    int max_usable = g_dsp_ctx->max_hw_threads - 2;
    if (requested_thread_counts > max_usable) {
        printf("setclocks: requested_thread_counts %ld exceeds safe limit %d (max_hthreads %d - 2), clamped\n",
               requested_thread_counts, max_usable, g_dsp_ctx->max_hw_threads);
        requested_thread_counts = max_usable;
    }
    if (requested_thread_counts <= g_dsp_ctx->thread_counts) {
        g_dsp_ctx->thread_counts = requested_thread_counts;
    }

    g_dsp_ctx->dump_diag_info      = diag_info;

    // Expose the actual thread count in effect on DSP side so AP can mirror it
    // (avoids jobs > work-queue threads when n_act_threads is precomputed on AP).
    if (actual_thread_counts) {
        *actual_thread_counts = (int32_t)g_dsp_ctx->thread_counts;
    }

    printf("\n");
    printf("actual thread_counts:           %d\n", g_dsp_ctx->thread_counts);
    printf("dump_diag_info:                 %d\n\n", g_dsp_ctx->dump_diag_info);

    // Initialize htp_context for calling Qualcomm's execute_op.
    // Shares our already-acquired VTCM and HMX queue.
    // New Qualcomm API (b2dd28a3b) requires pre-allocated backing buffers for
    // work_queue and dma queues; we memalign them here and track the pointers
    // in dsp_context for cleanup in ggml_dsp_close.
    if (g_dsp_ctx->thread_counts >= 1) {
        memset(g_dsp_ctx->htp_ctx, 0, sizeof(*g_dsp_ctx->htp_ctx));
        g_dsp_ctx->htp_ctx->vtcm_base      = (uint8_t *)g_dsp_ctx->vtcm_base;
        g_dsp_ctx->htp_ctx->vtcm_size      = g_dsp_ctx->vtcm_size;
        g_dsp_ctx->htp_ctx->vtcm_rctx      = g_dsp_ctx->compute_res_ctx_id;
        g_dsp_ctx->htp_ctx->hmx_queue      = g_dsp_ctx->hmx_queue;
        g_dsp_ctx->htp_ctx->n_threads      = (uint32_t)g_dsp_ctx->thread_counts;
        g_dsp_ctx->htp_ctx->n_threads_div  = init_fastdiv_values((uint32_t)g_dsp_ctx->thread_counts);
        g_dsp_ctx->htp_ctx->hmx_enabled    = g_dsp_ctx->hmx_available ? true : false;

        // work_queue: backing buffer holds worker stacks + queue struct.
        size_t wq_size  = work_queue_sizeof((uint32_t)g_dsp_ctx->thread_counts,
                                            JZ_WORK_QUEUE_CAPACITY, JZ_WORK_QUEUE_STACK_SIZE);
        size_t wq_align = work_queue_alignof();
        void * wq_buf   = memalign(wq_align, wq_size);
        AEEResult wp    = AEE_SUCCESS;
        if (wq_buf) {
            g_dsp_ctx->htp_ctx->work_queue = work_queue_init(wq_buf,
                                                              (uint32_t)g_dsp_ctx->thread_counts,
                                                              JZ_WORK_QUEUE_CAPACITY,
                                                              JZ_WORK_QUEUE_STACK_SIZE);
            if (g_dsp_ctx->htp_ctx->work_queue) {
                g_dsp_ctx->work_queue_buf = wq_buf;
            } else {
                free(wq_buf);
                g_dsp_ctx->work_queue_buf = NULL;
                wp = AEE_EFAILED;
            }
        } else {
            g_dsp_ctx->htp_ctx->work_queue = NULL;
            g_dsp_ctx->work_queue_buf      = NULL;
            wp = AEE_ENOMEMORY;
        }
        printf("htp_ctx work_queue_init returned %d (n_threads=%d)\n", wp, g_dsp_ctx->thread_counts);

        // dma queues: one main queue (dma_cached) + one nocache alias (dma) per
        // thread, mirroring main.c. Ops use the alias with nocache=1 so DMA DDR
        // accesses bypass L2 (same semantics as the pre-b2dd28a3b hex-dma, which
        // hardcoded bypass=1) and stay coherent with our manual dcinva/dccleaninva
        // cache management. dma_queue_init must get a valid (zeroed) trace:
        // htp_trace_event_start/stop dereference it unconditionally on every
        // push/pop.
        size_t dma_size       = dma_queue_sizeof(256);
        size_t dma_alias_size = dma_queue_alias_sizeof();
        size_t dma_align      = dma_queue_alignof();
        for (int i = 0; i < g_dsp_ctx->thread_counts; i++) {
            void * dma_buf = memalign(dma_align, dma_size);
            if (dma_buf) {
                g_dsp_ctx->htp_ctx->dma_cached[i] = dma_queue_init(dma_buf, 256,
                                                                   (uintptr_t)g_dsp_ctx->vtcm_base,
                                                                   g_dsp_ctx->vtcm_size,
                                                                   &g_dsp_ctx->htp_ctx->trace[i]);
                if (g_dsp_ctx->htp_ctx->dma_cached[i]) {
                    g_dsp_ctx->dma_queue_bufs[i] = dma_buf;
                } else {
                    free(dma_buf);
                    dma_buf = NULL;
                    g_dsp_ctx->dma_queue_bufs[i] = NULL;
                }
            } else {
                g_dsp_ctx->htp_ctx->dma_cached[i] = NULL;
                g_dsp_ctx->dma_queue_bufs[i]      = NULL;
            }

            void * alias_buf = memalign(dma_align, dma_alias_size);
            if (alias_buf && g_dsp_ctx->htp_ctx->dma_cached[i]) {
                g_dsp_ctx->htp_ctx->dma[i] = dma_queue_alias_init(alias_buf,
                                                                  g_dsp_ctx->htp_ctx->dma_cached[i], 1);
                if (g_dsp_ctx->htp_ctx->dma[i]) {
                    g_dsp_ctx->dma_alias_bufs[i] = alias_buf;
                } else {
                    free(alias_buf);
                    g_dsp_ctx->dma_alias_bufs[i] = NULL;
                }
            } else {
                if (alias_buf) free(alias_buf);
                g_dsp_ctx->htp_ctx->dma[i]   = NULL;
                g_dsp_ctx->dma_alias_bufs[i] = NULL;
            }
        }
        printf("htp_ctx dma_queue created x%d (main+alias)\n", g_dsp_ctx->thread_counts);
    }

    g_dsp_ctx->hexagon_power_ctx = (void *)(handle);

    // Test VTCM memory read/write (must ensure VTCM is available in cache mode)
    if (g_dsp_ctx->vtcm_base != NULL) {
        dsp_vtcm_acquire();
        uint8_t *weight = (uint8_t *)g_dsp_ctx->vtcm_base;
        uint8_t *active = (uint8_t *)g_dsp_ctx->vtcm_base + 256;
        // Write test patterns
        memset(weight, 0xaa, 128);
        memset(active, 0xbb, 128);
        // Verify write
        if (weight[0] == 0xaa && active[0] == 0xbb) {
            GGMLHEXAGON_LOG_INFO("VTCM read/write test PASSED: weight[0]=0x%02x, active[0]=0x%02x", weight[0], active[0]);
        } else {
            GGMLHEXAGON_LOG_ERROR("VTCM read/write test FAILED: weight[0]=0x%02x, active[0]=0x%02x", weight[0], active[0]);
        }
    } else {
        GGMLHEXAGON_LOG_WARN("VTCM not available, skipping VTCM test");
    }

    hap_probe_dsp(handle);

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return AEE_SUCCESS;
}

AEEResult ggml_dsp_register_ion(remote_handle64 h, uint32_t ion_fd, uint32_t size_lo, uint32_t size_hi) {
    (void)h;
    int32_t fd = (int32_t)ion_fd;
    uint64_t size = ((uint64_t)size_hi << 32) | (uint64_t)size_lo;

    GGMLHEXAGON_LOG_INFO("[ION-REG] fd=%d, size=%llu bytes (%dMB)",
                         fd, (unsigned long long)size, (int32_t)(size >> 20));

    int64_t t0_mmap = ggml_time_us();
#if __HVX_ARCH__ > 73
    void * va = HAP_mmap2(NULL, (size_t)size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#else
    void * va = HAP_mmap(NULL, (size_t)size, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
#endif
    int64_t dt_mmap = ggml_time_us() - t0_mmap;

    if (va == (void *)-1) {
        g_dsp_ctx->ion_dsp_base = NULL;
        GGMLHEXAGON_LOG_ERROR("[ION-REG] HAP_mmap2 FAILED: returned -1 (fd=%d, size=%llu)", fd, (unsigned long long)size);
        return AEE_EFAILED;
    }

    g_dsp_ctx->ion_dsp_base = va;
    g_dsp_ctx->ion_dsp_size = (size_t)size;
    // Use FARF(ALWAYS) so the timing log is visible via adb logcat on all builds
    FARF(ALWAYS, "[ION-REG] HAP_mmap2 OK: va=%p (fd=%d, size=%zuMB, time=%lld us)", va, fd, g_dsp_ctx->ion_dsp_size / (1024*1024), (long long)dt_mmap);

    return AEE_SUCCESS;
}

// Wake/suspend the work_queue + hmx_queue worker threads around a batch,
// mirroring htp/main.c htp_packet_callback. Without wakeup the workers stay in
// qurt_futex_wait and (depending on QuRT futex semantics) can miss seqn
// bumps; without hmx_queue_flush the batch can return while HMX descriptors
// are still in flight, so AP reads back incomplete dst tensors.
static void dsp_queues_wakeup(void) {
    struct htp_context * htp = g_dsp_ctx->htp_ctx;
    if (htp->work_queue) {
        work_queue_wakeup(htp->work_queue);
    }
    if (htp->hmx_queue) {
        hmx_queue_wakeup(htp->hmx_queue);
    }
}

static void dsp_queues_suspend(void) {
    struct htp_context * htp = g_dsp_ctx->htp_ctx;
    if (htp->hmx_queue) {
        hmx_queue_suspend(htp->hmx_queue);
        hmx_queue_flush(htp->hmx_queue);
    }
    if (htp->work_queue) {
        work_queue_suspend(htp->work_queue);
    }
}

/*
 * ION-based op-batch execution: FastRPC only passes 2 scalars (offset, size) - all data is in the mempool.
 */
AEEResult ggml_dsp_execute_batch(remote_handle64 h, uint32_t batch_offset, uint32_t batch_size) {
    if (g_dsp_ctx->ion_dsp_base == NULL) {
        GGMLHEXAGON_LOG_ERROR("ION base not registered");
        return AEE_EBADPARM;
    }

    const char * base = (const char *)g_dsp_ctx->ion_dsp_base;

    /* dsp_cache_mode config mode: batch_size == 0xFFFC. batch_offset encodes
     *   bits  0..3 : dsp_cache_mode (first-touch weight / prior-dst skip / bulk dst flush / selective bulk flush)
     *   bit  16    : dsp_cache_trace_bit0 (1 = emit [DSP-CACHE-TRACE-BIT0] per bit 0 decision)
     *   bit  17    : dsp_cache_trace_bit1 (1 = emit [DSP-CACHE-TRACE-BIT1] per bit 1 decision)
     * Pushed by AP at ggmlhexagon_init_cdsp() time. Bit definitions:
     *   bit 0 (0x1): first-touch weight bitmap    - INVAL_SRC_IF_NEEDED skips
     *               dcinva for repack weights (flags==2) once invalidated.
     *   bit 1 (0x2): skip dcinva for prior dst     - INVAL_SRC_IF_NEEDED skips
     *               dcinva for activations (flags!=2) when [base,base+len) is
     *               fully contained in a dst range that DSP wrote earlier in
     *               this batch. Only effective when bit 2 is also on (the
     *               per-op dst tracker populates the prior_dst list).
     *   bit 2 (0x4): bulk dst flush at batch end  - per-op flush is suppressed;
     *               dst ranges are collected/sort/merged/flushed once after
     *               the per-op loop.
     *   bit 3 (0x8): selective bulk flush         - skip batch-end flush for
     *               dsts that a later op in this batch still consumes (pure
     *               intermediates). Only effective when bit 2 is also on.
     *               Mirrored dsts (flags&0x1) and dsts with no later consumer
     *               (final outputs, KV write views) are always flushed. */
    if (batch_size == 0xFFFC) {
        g_dsp_ctx->dsp_cache_mode = batch_offset & 0xFu;
        g_dsp_ctx->dsp_cache_trace_bit0 = (batch_offset >> 16) & 0x1u;
        g_dsp_ctx->dsp_cache_trace_bit1 = (batch_offset >> 17) & 0x1u;
        GGMLHEXAGON_LOG_INFO("[DSP-CACHE-MODE] dsp_cache_mode=0x%x (bit0=first-touch-weight=%d bit1=skip-prior-dst=%d bit2=bulk-dst-flush=%d bit3=selective-bulk-flush=%d) dsp_cache_trace_bit0=%d dsp_cache_trace_bit1=%d",
                             g_dsp_ctx->dsp_cache_mode,
                             (g_dsp_ctx->dsp_cache_mode & 0x1) ? 1 : 0,
                             (g_dsp_ctx->dsp_cache_mode & 0x2) ? 1 : 0,
                             (g_dsp_ctx->dsp_cache_mode & 0x4) ? 1 : 0,
                             (g_dsp_ctx->dsp_cache_mode & 0x8) ? 1 : 0,
                             g_dsp_ctx->dsp_cache_trace_bit0,
                             g_dsp_ctx->dsp_cache_trace_bit1);
        return AEE_SUCCESS;
    }

    /* Warmup mode: batch_size == 0xFFFB.
     * AP calls this once after session init to warm up the FastRPC/ION path
     * without doing any real compute. Just logs and returns. */
    if (batch_size == 0xFFFB) {
        GGMLHEXAGON_LOG_INFO("[DSP-WARMUP] no-op warmup done");
        return AEE_SUCCESS;
    }

    /* Normal batch execution */
#if HEX_OP_PROF
    const int64_t prof_batch_t0 = ggml_time_us();
#endif
    /* Invalidate DSP cache for the batch descriptor before reading.
     * ION is non-coherent: AP reuses the mempool and writes a new batch
     * at the same offset, so DSP must invalidate to fetch fresh data.
     * Use dcinva (invalidate only) instead of dccleaninva (clean+invalidate):
     * dccleaninva would write back stale DSP cache lines to DRAM, overwriting
     * the fresh data AP just flushed via DC CVAC. */
#if HEX_OP_PROF
    const int64_t prof_hdr_t0 = ggml_time_us();
#endif
    ggml_dsp_cache_inval_range((void *)(base + batch_offset), batch_size);
#if HEX_OP_PROF
    g_nonop_hdr_inval_us += (uint64_t)(ggml_time_us() - prof_hdr_t0);
#endif
    const hex_batch_hdr * hdr = (const hex_batch_hdr *)(base + batch_offset);

    if (hdr->n_ops == 0 || hdr->n_tensors == 0) {
        GGMLHEXAGON_LOG_ERROR("empty ion-batch: n_ops=%u n_tensors=%u", hdr->n_ops, hdr->n_tensors);
        return AEE_EBADPARM;
    }

    const hex_op_desc * ops = (const hex_op_desc *)((const char *)hdr + hdr->ops_offset);
    const hex_tensor_desc * tens = (const hex_tensor_desc *)((const char *)hdr + hdr->tensors_offset);

    /* VTCM is acquired once in ggml_dsp_open and held for the whole
     * session (matches htp/main.c htp_packet_callback pattern). Per-batch
     * acquire/release removed: see dsp_vtcm_acquire in ggml_dsp_open. */

    GGMLHEXAGON_LOG_DEBUG("ion-batch: start n_ops=%u n_tensors=%u", hdr->n_ops, hdr->n_tensors);

    /* Reset per-batch dst trackers.
     *  - prior_dst_ranges is consulted by bit 1; the per-op dst tracker
     *    populates it when bit 2 is on. Resetting on bit 1 OR bit 2 keeps
     *    the list clean even if bit 1 is enabled without bit 2.
     *  - bulk_flush_ranges is populated only when bit 2 is on, but the
     *    list itself is harmless when empty (bulk_flush_all() early-returns
     *    if count==0). */
    if (g_dsp_ctx->dsp_cache_mode & (0x2 | 0x4)) {
        prior_dst_reset_all();
    }
    if (g_dsp_ctx->dsp_cache_mode & 0x4) {
        bulk_flush_reset_all();
    }

    /* Pre-convert all tensors once: saves one hex_tensor_to_dsptensor()
     * call per tensor per op. The per-op loop below references g_pre_dt
     * by pointer (srcs) or copies + overrides op_params (dsts). */
    if (hdr->n_tensors > DSP_OPT_MAX_TENSORS) {
        GGMLHEXAGON_LOG_ERROR("n_tensors %u exceeds DSP_OPT_MAX_TENSORS %u",
                              hdr->n_tensors, DSP_OPT_MAX_TENSORS);
        return AEE_EFAILED;
    }

#if HEX_OP_PROF
    const int64_t prof_pre_t0 = ggml_time_us();
#endif
    for (uint32_t ti = 0; ti < hdr->n_tensors; ti++) {
        hex_tensor_to_dsptensor(&tens[ti], base, &g_pre_dt[ti]);
        hex_tensor_to_htp_tensor(&tens[ti], base, &g_pre_ht[ti]);
    }
#if HEX_OP_PROF
    g_nonop_preconvert_us += (uint64_t)(ggml_time_us() - prof_pre_t0);
#endif

    /* Reset per-batch invalidation tracking: all tensors start as
     * "needs invalidation" (1). Set to 0 when invalidated; set back
     * to 1 when written as dst (dirtied). */
    memset(g_batch_tensor_needs_inval, 1, hdr->n_tensors);

    /* bit 3: record each tensor's last consumer op index so the dst
     * tracker can tell pure intermediates (last_use > producer op)
     * from final outputs (never consumed later). */
    if (g_dsp_ctx->dsp_cache_mode & 0x8) {
        memset(g_tensor_last_use_op, 0, hdr->n_tensors * sizeof(g_tensor_last_use_op[0]));
        for (uint32_t oi = 0; oi < hdr->n_ops; oi++) {
            for (int s = 0; s < HTP_OP_MAX_INPUTS; s++) {
                const int32_t si = ops[oi].src_idx[s];
                if (si >= 0 && si < (int32_t)hdr->n_tensors) {
                    g_tensor_last_use_op[si] = oi;
                }
            }
        }
    }

#if HEX_OP_PROF
    const int64_t prof_q_t0 = ggml_time_us();
#endif
    dsp_queues_wakeup();
#if HEX_OP_PROF
    g_nonop_queue_us += (uint64_t)(ggml_time_us() - prof_q_t0);
#endif

    for (uint32_t i = 0; i < hdr->n_ops; i++) {
        const hex_op_desc * op = &ops[i];

        /* srcs: reference pre-converted tensors directly */
        const dsptensor *src0_dt = &g_pre_dt[op->src_idx[0]];
        const dsptensor *src1_dt = (op->src_idx[1] >= 0) ? &g_pre_dt[op->src_idx[1]] : NULL;
        const dsptensor *src2_dt = (op->src_idx[2] >= 0) ? &g_pre_dt[op->src_idx[2]] : NULL;
        const dsptensor *src3_dt = (op->src_idx[3] >= 0) ? &g_pre_dt[op->src_idx[3]] : NULL;

        if (1 == g_dsp_ctx->dump_diag_info) {
            if (src0_dt->data && src0_dt->data_len >= 16) {
                const float * fv = (const float *)src0_dt->data;
                GGMLHEXAGON_LOG_INFO("[DSP-DIAG] op%u src0 PRE-INVAL off=0x%x ptr=%p f32=[%.4f, %.4f, %.4f, %.4f]",
                                 i, tens[op->src_idx[0]].data_offset, src0_dt->data, fv[0], fv[1], fv[2], fv[3]);
            }
            if (src1_dt && src1_dt->data && src1_dt->data_len >= 16 && src1_dt->type == 0) {
                const float * fv = (const float *)src1_dt->data;
                GGMLHEXAGON_LOG_INFO("[DSP-DIAG] op%u src1 off=0x%x ptr=%p f32=[%.4f, %.4f, %.4f, %.4f] ne=[%d,%d,%d,%d]",
                                 i, tens[op->src_idx[1]].data_offset, src1_dt->data, fv[0], fv[1], fv[2], fv[3],
                                 (int)src1_dt->ne[0], (int)src1_dt->ne[1], (int)src1_dt->ne[2], (int)src1_dt->ne[3]);
            }
        }

        /* dsts: copy from pre-converted, then override op_params with
         * per-op params (node->op_params is correct; dst tensor's op_params
         * can be zero or stale for in-place reuse ops like SCALE).
         * Uses file-scope static buffers (see g_dst_dt_buf/g_dst_dt_ptrs)
         * to avoid per-op stack frame pressure. */
        for (int k = 0; k < HTP_OP_MAX_OUTPUTS; k++) {
            g_dst_dt_ptrs[k] = NULL;
            if (op->dst_idx[k] < 0) continue;
            g_dst_dt_buf[k] = g_pre_dt[op->dst_idx[k]];
            memcpy(g_dst_dt_buf[k].op_params, op->params, sizeof(g_dst_dt_buf[k].op_params));
            g_dst_dt_ptrs[k] = &g_dst_dt_buf[k];
        }

        /* Cache maintenance for non-coherent mempool memory.
         * Per-batch tracking skips redundant dcinva when the same tensor is
         * used as src by multiple ops. See dsp_cache_mode bit definitions
         * at the batch_size==0xFFFC gate above. */
        INVAL_SRC_IF_NEEDED(i, 0, src0_dt, op->src_idx[0]);
        if (src1_dt) INVAL_SRC_IF_NEEDED(i, 1, src1_dt, op->src_idx[1]);
        if (src2_dt) INVAL_SRC_IF_NEEDED(i, 2, src2_dt, op->src_idx[2]);
        if (src3_dt) INVAL_SRC_IF_NEEDED(i, 3, src3_dt, op->src_idx[3]);

        if (1 == g_dsp_ctx->dump_diag_info) {
            if (src0_dt->data && src0_dt->data_len >= 16) {
                const float * fv = (const float *)src0_dt->data;
                float eps_f;
                memcpy(&eps_f, g_dst_dt_buf[0].op_params, sizeof(float));
                GGMLHEXAGON_LOG_INFO("[DSP-DIAG] op%u src0 POST-INVAL off=0x%x ptr=%p f32=[%.4f, %.4f, %.4f, %.4f] eps=%f ne=[%d,%d,%d,%d]",
                                 i, tens[op->src_idx[0]].data_offset, src0_dt->data, fv[0], fv[1], fv[2], fv[3], eps_f,
                                 (int)src0_dt->ne[0], (int)src0_dt->ne[1], (int)src0_dt->ne[2], (int)src0_dt->ne[3]);
            }
        }

        GGMLHEXAGON_LOG_DEBUG("ion-batch: op %u/%u opc=%d", i, hdr->n_ops, op->opcode);

        // Translation layer: map GGML op to HTP op, build octx, call execute_op.
        // For fused ops, AP sets htp_opcode directly (skip ggml_op_to_htp_op).
        enum htp_op_code htp_op;
        if (op->htp_opcode != 0) {
            htp_op = (enum htp_op_code) op->htp_opcode;
        } else if (ggml_op_to_htp_op(op->opcode, op->params, &htp_op) != 0) {
            GGMLHEXAGON_LOG_ERROR("ion-op %u: unsupported opcode %d", i, op->opcode);
            dsp_queues_suspend();
            return AEE_EUNSUPPORTED;
        }

        struct htp_ops_context octx;

        build_htp_octx(&octx, htp_op, op->params, op->kernel_params,
                       op->src_idx, op->dst_idx);

        if (htp_op == HTP_OP_MUL_MAT) {
            const int32_t kp_kernel_type = octx.kernel_params[0];
            if (kp_kernel_type == 0) {
                if (build_mm_kernel_params(&octx) != 0) {
                    dsp_queues_suspend();
                    return AEE_EFAILED;
                }
            }
        }

        if (htp_op == HTP_OP_FLASH_ATTN_EXT) {
            const int32_t kp_kernel_type = octx.kernel_params[0];
            if (kp_kernel_type == HTP_FA_KERNEL_UNSUPPORTED) {
                if (build_fa_kernel_params(&octx) != 0) {
                    dsp_queues_suspend();
                    return AEE_EFAILED;
                }
            }
        }

#if GGMLHEXAGON_DEBUG
        /* F32 MUL_MAT diagnostic: dump src0 row 0/16, src1 row 0, dst[16] BEFORE execute_op. */
        if (htp_op == HTP_OP_MUL_MAT && src0_dt->type == 0 /*F32*/ &&
            src0_dt->data && src0_dt->data_len >= (size_t)(17 * 256) &&
            src1_dt && src1_dt->data && src1_dt->data_len >= 16 &&
            g_dst_dt_ptrs[0] && g_dst_dt_buf[0].data && g_dst_dt_buf[0].data_len >= (size_t)(17 * 4)) {
            const float * s0  = (const float *) src0_dt->data;
            const float * s1  = (const float *) src1_dt->data;
            const float * dp  = (const float *) g_dst_dt_buf[0].data;
            const uint32_t s0_row16_off = src0_dt->nb[1] * 16 / 4;
            GGMLHEXAGON_LOG_ERROR("[DSP-MM-PRE] op%u kp_type=%d s0r0=[%.4f,%.4f,%.4f,%.4f] s0r16=[%.4f,%.4f,%.4f,%.4f] s1r0=[%.4f,%.4f,%.4f,%.4f] dst16=[%.4f,%.4f,%.4f,%.4f] nb=[%u,%u,%u,%u] ne=[%u,%u,%u,%u]",
                i, octx.kernel_params[0],
                s0[0], s0[1], s0[2], s0[3],
                s0[s0_row16_off+0], s0[s0_row16_off+1], s0[s0_row16_off+2], s0[s0_row16_off+3],
                s1[0], s1[1], s1[2], s1[3],
                dp[16], dp[17], dp[18], dp[19],
                src0_dt->nb[0], src0_dt->nb[1], src0_dt->nb[2], src0_dt->nb[3],
                src0_dt->ne[0], src0_dt->ne[1], src0_dt->ne[2], src0_dt->ne[3]);
            /* ktype, pipe, mch, nch, nthr, nact, nhmx, npf, src1rs, vtcm_sz,
             * vtcm_src0, vtcm_src1, vtcm_dst */
            GGMLHEXAGON_LOG_ERROR("[DSP-MM-KP]  op%u ktype=%d pipe=%d mch=%d nch=%d nthr=%d nact=%d nhmx=%d npf=%d src1rs=%d vtcm_sz=%d src0_sz=%d src1_sz=%d dst_sz=%d",
                i, octx.kernel_params[0], octx.kernel_params[1], octx.kernel_params[2],
                octx.kernel_params[3], octx.kernel_params[4], octx.kernel_params[5],
                octx.kernel_params[6], octx.kernel_params[7], octx.kernel_params[10],
                octx.kernel_params[11], octx.kernel_params[12], octx.kernel_params[13],
                octx.kernel_params[16]);
        }
#endif

        int op_ret = execute_op(&octx);

#if GGMLHEXAGON_DEBUG
        /* F32 MUL_MAT diagnostic: dump dst[0..3] and dst[16..19] AFTER execute_op. */
        if (htp_op == HTP_OP_MUL_MAT && src0_dt->type == 0 /*F32*/ &&
            g_dst_dt_ptrs[0] && g_dst_dt_buf[0].data && g_dst_dt_buf[0].data_len >= (size_t)(20 * 4)) {
            const float * dp = (const float *) g_dst_dt_buf[0].data;
            GGMLHEXAGON_LOG_ERROR("[DSP-MM-POST] op%u d[0..3]=[%.4f,%.4f,%.4f,%.4f] d[16..19]=[%.4f,%.4f,%.4f,%.4f]",
                i, dp[0], dp[1], dp[2], dp[3], dp[16], dp[17], dp[18], dp[19]);
        }
#endif

        // Clear spad refs (matches proc_op_req post-execute cleanup)
        octx.src0_spad.src = NULL;
        octx.src1_spad.src = NULL;
        octx.src2_spad.src = NULL;
        octx.src3_spad.src = NULL;
        octx.dst_spad.src  = NULL;

        if (op_ret != HTP_STATUS_OK) {
            const char * st_name =
                (op_ret == HTP_STATUS_INTERNAL_ERR)   ? "INTERNAL_ERR"   :
                (op_ret == HTP_STATUS_NO_SUPPORT)    ? "NO_SUPPORT"     :
                (op_ret == HTP_STATUS_INVAL_PARAMS)  ? "INVAL_PARAMS"   :
                (op_ret == HTP_STATUS_VTCM_TOO_SMALL) ? "VTCM_TOO_SMALL" : "UNKNOWN";
            GGMLHEXAGON_LOG_ERROR("ion-op %u: execute_op returned %d/%s (htp_op=%d)",
                                  i, op_ret, st_name, htp_op);
            dsp_queues_suspend();
            return AEE_EFAILED;
        }

        GGMLHEXAGON_LOG_DEBUG("ion-batch: op %u done", i);

        /* bit 2: bulk dst flush at batch end.
         * Also mark dst tensors as dirty so they get re-invalidated
         * when used as src later in the same batch. */
#if HEX_OP_PROF
        const int64_t prof_dst_t0 = ggml_time_us();
#endif
        for (int k = 0; k < HTP_OP_MAX_OUTPUTS; k++) {
            if (!g_dst_dt_ptrs[k]) continue;
            if (g_dsp_ctx->dsp_cache_mode & 0x4) {
                /* prior_dst ranges are only consumed by bit 1; avoid dead
                 * work when bit 2 is enabled without bit 1. */
                if (g_dsp_ctx->dsp_cache_mode & 0x2) {
                    prior_dst_add(g_dst_dt_buf[k].data, g_dst_dt_buf[k].data_len,
                                  op->dst_idx[k], htp_op);
                }
                /* bit 3: pure intermediates are read back from DSP L2 by
                 * their in-batch consumers, so their batch-end flush is
                 * dead traffic. Mirrored dsts (flags&0x1) are read by AP
                 * after the batch and must always be flushed. */
                const int32_t di = op->dst_idx[k];
                const bool skip_flush =
                    (g_dsp_ctx->dsp_cache_mode & 0x8) &&
                    di >= 0 && di < (int32_t)hdr->n_tensors &&
                    g_tensor_last_use_op[di] > i &&
                    !(g_dst_dt_buf[k].flags & 0x1);
                if (!skip_flush) {
                    bulk_flush_add(g_dst_dt_buf[k].data, g_dst_dt_buf[k].data_len);
                }
            } else {
                ggml_dsp_cache_flush_range(g_dst_dt_buf[k].data, g_dst_dt_buf[k].data_len);
            }
            /* Mark dst tensor as dirty: next time it's used as src,
             * it must be re-invalidated. */
            g_batch_tensor_needs_inval[op->dst_idx[k]] = 1;
            if (g_dsp_ctx->dsp_cache_mode & 0x1) {
                weight_inval_unmark(g_dst_dt_buf[k].data);
            }
        }
#if HEX_OP_PROF
        g_nonop_dst_track_us += (uint64_t)(ggml_time_us() - prof_dst_t0);
#endif

        if (1 == g_dsp_ctx->dump_diag_info) {
            /* DSP-side DIAG: dump first 4 f32 values from each dst output */
            for (int k = 0; k < HTP_OP_MAX_OUTPUTS; k++) {
                if (g_dst_dt_ptrs[k] && g_dst_dt_buf[k].data && g_dst_dt_buf[k].data_len >= 16) {
                    const float * fv = (const float *)g_dst_dt_buf[k].data;
                    GGMLHEXAGON_LOG_INFO("[DSP-DIAG] op%u dst[%d] off=0x%x ptr=%p f32=[%.4f, %.4f, %.4f, %.4f]",
                                     i, k, tens[op->dst_idx[k]].data_offset, g_dst_dt_buf[k].data, fv[0], fv[1], fv[2], fv[3]);
                }
            }
        }
    }

    GGMLHEXAGON_LOG_DEBUG("ion-batch: all %u ops done", hdr->n_ops);

#if HEX_OP_PROF
    const int64_t prof_qs_t0 = ggml_time_us();
#endif
    dsp_queues_suspend();
#if HEX_OP_PROF
    g_nonop_queue_us += (uint64_t)(ggml_time_us() - prof_qs_t0);
#endif

    /* bit 2: bulk dst flush. Sort + merge collected ranges, then flush once
     * per merged region. Flushes happen here (after all ops in the batch
     * finished) so AP reads see fresh DRAM. */
#if HEX_OP_PROF
    const int64_t prof_bf_t0 = ggml_time_us();
#endif
    if (g_dsp_ctx->dsp_cache_mode & 0x4) {
        bulk_flush_all();
    }
#if HEX_OP_PROF
    g_nonop_bulk_flush_us += (uint64_t)(ggml_time_us() - prof_bf_t0);
#endif

    __asm__ __volatile__("" ::: "memory");
    if (hdr->n_ops > 0 && ops[hdr->n_ops - 1].dst_idx[0] < hdr->n_tensors) {
        uint32_t last_off = tens[ops[hdr->n_ops - 1].dst_idx[0]].data_offset;
        if (batch_size > last_off + 4)
            (void) *(volatile const int *)(base + last_off);
    }
    __asm__ __volatile__("" ::: "memory");

    /* VTCM is held for the whole session; release is deferred to
     * ggml_dsp_close. Matches htp/main.c htp_packet_callback pattern. */

#if HEX_OP_PROF
    /* Per-op profiler: print accumulated cum/count/avg every N batches so
     * log volume stays bounded (one multi-line dump per interval, not per op). */
    g_op_prof_batch_wall_us += (uint64_t)(ggml_time_us() - prof_batch_t0);
    g_op_prof_batch_count++;
    if ((g_op_prof_batch_count % HEX_OP_PROF_DUMP_INTERVAL) == 0) {
        char tag[32];
        snprintf(tag, sizeof(tag), "batch#%u", g_op_prof_batch_count);
        dump_op_prof(tag);
        /* whole-batch wall vs sum of per-op buckets: the delta is the DSP-side
         * non-op overhead (descriptor inval, tensor pre-convert, per-op src
         * dcinva, bulk dst flush, queue wakeup/suspend). */
        uint64_t op_sum = 0;
        for (unsigned int b = 0; b < HEX_OP_PROF_BUCKETS; b++) op_sum += g_op_prof_dur_us[b];
        FARF(ERROR, "[OP-PROF] %s batch-wall cum=%llu us (avg=%llu) op-sum cum=%llu us (avg=%llu) non-op avg=%lld us/batch",
             tag,
             (unsigned long long) g_op_prof_batch_wall_us,
             (unsigned long long) (g_op_prof_batch_wall_us / g_op_prof_batch_count),
             (unsigned long long) op_sum,
             (unsigned long long) (op_sum / g_op_prof_batch_count),
             (long long) ((int64_t) (g_op_prof_batch_wall_us - op_sum) / (int64_t) g_op_prof_batch_count));
        const uint64_t nbc = g_op_prof_batch_count;
        FARF(ERROR, "[OP-PROF-NONOP] %s hdr=%llu pre=%llu w-inv=%llu(%lluMB) a-inv=%llu(%lluMB) dst=%llu bulk=%llu queue=%llu us/batch",
             tag,
             (unsigned long long) (g_nonop_hdr_inval_us / nbc),
             (unsigned long long) (g_nonop_preconvert_us / nbc),
             (unsigned long long) (g_nonop_w_inval_us / nbc),
             (unsigned long long) (g_nonop_w_inval_bytes / nbc / (1024 * 1024)),
             (unsigned long long) (g_nonop_a_inval_us / nbc),
             (unsigned long long) (g_nonop_a_inval_bytes / nbc / (1024 * 1024)),
             (unsigned long long) (g_nonop_dst_track_us / nbc),
             (unsigned long long) (g_nonop_bulk_flush_us / nbc),
             (unsigned long long) (g_nonop_queue_us / nbc));
    }
#endif

    return AEE_SUCCESS;
}
