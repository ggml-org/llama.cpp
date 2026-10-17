#pragma once

// per-action trace of decode steps: fixed-size records written by the acting thread into per-thread chunks, a writer
// thread serializes them as Chrome Trace Event JSON (ui.perfetto.dev); see ggml-trace.cpp for the file layout
//
// state machine: OFF -(start)-> IDLE -(batch_begin)-> TRACED | SUSPENDED -(batch_end)-> IDLE
// ggml_trace_active() is true only while a traced batch is open, ggml_trace_enabled() also in IDLE / SUSPENDED,
// ggml_trace_io_enabled() in IDLE / TRACED (hints are recorded between traced steps, not during prompt batches)

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

    enum ggml_trace_type {
        GGML_TRACE_TOKEN = 0,        // one traced decode step, batch_begin .. batch_end; aux = sampled token (-1 unknown)
        GGML_TRACE_SYNC,             // the host waits for a backend; detail GGML_TRACE_OUTPUT for the logits wait
        GGML_TRACE_SAMPLE,           // llama_sampler_sample; aux = sampled token
        GGML_TRACE_XFER,             // ggml_backend_tensor_set / get; detail GGML_TRACE_INPUT / OUTPUT, bytes
        GGML_TRACE_SPLIT,            // one scheduler split; aux = n_nodes
        GGML_TRACE_COPY,             // copy of a split input; detail GGML_TRACE_COPY_SYNC / ASYNC (instant), bytes
        GGML_TRACE_OP_CPU,           // one computed node or fused group on the CPU, incl. the barrier; detail = nodes in the group
        GGML_TRACE_OP_GPU,           // same on a device: t0 / t1 device time, aux / aux2 host launch t0 / t1, piece = stream
        GGML_TRACE_EXPERT_THREAD,    // one thread's compute of one expert slice; aux = stall ns, aux2 = minflt | majflt << 32, n_parts = nth
        GGML_TRACE_EXPERT,           // written by the writer only: EXPERT_THREAD of all threads merged
        GGML_TRACE_EXPERT_RESIDENCY, // instant: mincore of the slice at compute start; bytes = resident pages, aux = pages, aux2 = sample delay ns
        GGML_TRACE_IO_ISSUE,         // one posix_madvise piece; detail = source | flags, aux = cookie, aux2 = read_bytes delta
        GGML_TRACE_IO_RESIDENT,      // instant: the piece is in the page cache; detail = GGML_TRACE_IO_CACHED / RESOLVED / UNRESOLVED, aux = cookie
        GGML_TRACE_PREDICT,          // the predictor's on_layer; bytes = guesses
        GGML_TRACE_META,             // bookkeeping (poll interval adaptation); aux = new interval us
        GGML_TRACE_TYPE_COUNT,
    };

    // detail values
    enum ggml_trace_detail {
        GGML_TRACE_INPUT      = 1, // XFER
        GGML_TRACE_OUTPUT     = 2, // XFER, SYNC
        GGML_TRACE_COPY_SYNC  = 1,
        GGML_TRACE_COPY_ASYNC = 2,

        // IO_ISSUE: source in the low byte, flags above
        GGML_TRACE_IO_PREFETCH     = 0,
        GGML_TRACE_IO_PREDICT_HIGH = 1,
        GGML_TRACE_IO_PREDICT_LOW  = 2,
        GGML_TRACE_IO_SOURCE_MASK  = 0xff,
        GGML_TRACE_IO_DUPLICATE    = 0x100, // the range was already pending, aux = the original cookie
        GGML_TRACE_IO_CACHED_FLAG  = 0x200, // no read was caused by the hint

        // IO_RESIDENT
        GGML_TRACE_IO_CACHED     = 1, // at issue, by the issuer
        GGML_TRACE_IO_RESOLVED   = 2, // by the poller; aux2 = t_prev_poll, t0 = t_this_poll
        GGML_TRACE_IO_UNRESOLVED = 3, // still pending at stop; bytes = pages, aux2 = resident pages
    };

    enum ggml_trace_flag {
        GGML_TRACE_FLAG_LAYER_INFERRED = 1, // layer inherited from the previous event of the emitting thread
        GGML_TRACE_FLAG_WEIGHT_SRC     = 2, // OP_GPU: src[0] is a weight (a weight buffer of the device)
    };

    // tid of the named host tracks; compute threads use their worker index (>= 0)
    enum ggml_trace_track {
        GGML_TRACE_TID_SCHED     = -1,
        GGML_TRACE_TID_PREDICTOR = -2,
        GGML_TRACE_TID_POLLER    = -3,
        GGML_TRACE_TID_SAMPLER   = -4,
        GGML_TRACE_TID_PREDICTOR1 = -5, // second prediction worker
    };

    struct ggml_trace_event {
        int64_t t0_ns;
        int64_t t1_ns;   // 0 for instants
        int32_t type;    // enum ggml_trace_type
        int32_t detail;
        int32_t backend; // 0 = CPU, 1.. = ggml_trace_backend_id()
        int32_t tid;
        int32_t step;    // traced decode counter, -1 outside a traced batch
        int32_t token;
        int32_t pos;
        int32_t layer;   // -1: the writer parses it from name
        int32_t expert;  // -1: n/a
        int32_t op;      // enum ggml_op
        int32_t piece;   // IO: piece index of the slice, OP_GPU: stream
        int32_t n_parts; // IO: pieces of the slice, EXPERT_THREAD: threads
        int32_t rank;    // IO: guess rank of the predictor (-1 n/a)
        int32_t flags;   // GGML_TRACE_FLAG_*
        int64_t bytes;
        int64_t aux;
        int64_t aux2;
        char    name[32];
    };

    struct ggml_trace_params {
        const char * path;
        int    n_tokens;        // traced tokens before the file is closed, 0 = unlimited
        int    n_skip;          // decode-size batches that are not traced first
        int    max_batch;       // batches with more tokens are SUSPENDED
        int    detail;          // 1, or 2 for per-thread expert records and per-piece I/O spans
        int    poll_us;         // residency poll interval
        int    buffer_mb;       // chunk pool cap; full pool drops records (counted)
        bool   residency_valid; // mincore reports page cache residency for the weight files (owner or writable)
        const char * meta;      // extra members of the metadata record, e.g. "\"model\":\"x\",\"n_threads\":14", or NULL
    };

    // refcounted; the first start opens the file and starts the writer and poller threads
    GGML_API bool ggml_trace_start(const struct ggml_trace_params * params);
    GGML_API void ggml_trace_stop(void);

    GGML_API bool ggml_trace_active(void);
    GGML_API bool ggml_trace_enabled(void);
    GGML_API bool ggml_trace_io_enabled(void);
    GGML_API int  ggml_trace_detail(void);
    GGML_API int  ggml_trace_step(void);      // current traced step, -1 otherwise
    GGML_API int  ggml_trace_last_step(void); // last traced step, -1 if none

    GGML_API void ggml_trace_batch_begin(int n_tokens, int32_t token, int32_t pos);
    GGML_API void ggml_trace_batch_end(void);
    // the sampled token, into the most recent traced token unless a batch began since; returns its step, else -1
    GGML_API int  ggml_trace_stamp_token_out(int32_t token);

    GGML_API int64_t ggml_trace_now_ns(void);
    GGML_API int64_t ggml_trace_thread_cpu_ns(void); // CLOCK_THREAD_CPUTIME_ID, 0 if unavailable
    GGML_API bool    ggml_trace_thread_faults(int64_t * minflt, int64_t * majflt);
    GGML_API int64_t ggml_trace_read_bytes(void);    // read_bytes of the calling thread (/proc), -1 if unavailable

    // fills the defaults: step from the state, layer / expert / rank -1
    GGML_API void ggml_trace_event_init(struct ggml_trace_event * ev, enum ggml_trace_type type, int tid);
    GGML_API void ggml_trace_emit(const struct ggml_trace_event * ev);
    // a name longer than the record keeps its head and tail ("blk.12." and "-12" stay parsable)
    GGML_API void ggml_trace_event_set_name(struct ggml_trace_event * ev, const char * name);

    // guess rank of a predictor hint: router position r, or GGML_TRACE_RANK_POPULARITY + position p; -1 no guess
    #define GGML_TRACE_RANK_POPULARITY 1000

    // one posix_madvise piece, after the call; returns the cookie (the original one for a duplicate), 0 when off
    // layer -1: parsed from name
    // step: ggml_trace_step() taken once per slice by the issuer, so that the pieces of one slice stay together;
    // ignored (0 returned) unless ggml_trace_io_enabled()
    GGML_API int64_t ggml_trace_io_issue(int step, const void * addr, size_t len, int source, int rank, int layer, int expert,
            const char * name, int piece, int n_pieces, int64_t t0, int64_t t1, int64_t read_bytes_delta, int tid);
    // the first thread that touches an expert slice asks the poller to sample its residency; never blocks
    GGML_API void ggml_trace_residency_request(int layer, int expert, const char * name, const void * addr, size_t len, int64_t t_first_touch);

    // resolves the pending device timings now (after the output of a step has been waited for)
    GGML_API void ggml_trace_flush(void);

    // called on the batch_begin / stop thread so that backends resolve their pending device timings;
    // the callback runs under the registry lock: it must not register, unregister or call ggml_trace_backend_id
    GGML_API void ggml_trace_register_flush  (void (*fn)(void * ud), void * ud);
    GGML_API void ggml_trace_unregister_flush(void (*fn)(void * ud), void * ud);
    GGML_API int  ggml_trace_backend_id(const char * name); // interned, "CPU" = 0

#ifdef __cplusplus
}
#endif
