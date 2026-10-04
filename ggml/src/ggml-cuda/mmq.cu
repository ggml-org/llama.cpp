#include "common.cuh"
#include "mmq.cuh"
#include "mmq-profile.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// MMQ profiler: host implementation (see mmq-profile.cuh).
// Inert unless GGML_MQ_PROFILE is set.
//
// The profiler targets the Blackwell CUDA FP4 (NVFP4 / MXFP4) MMQ path and is
// compiled out to no-op stubs under HIP/MUSA: those backends never run that
// path, and the ROCm/MUSA runtime does not expose the cuda* event API the host
// code below uses. The device-side phase counters in mmq.cuh stay portable
// (they are a no-op whenever prof_buffer() returns nullptr, which the stubs do).
// ---------------------------------------------------------------------------
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// Per-graph CUDA event ring. Events recorded during a stream capture become
// nodes of THAT graph and may not be reused in another capture, so each
// distinct graph (shape) owns its own ring.
struct event_ring {
    cudaEvent_t ev[2 * mmq_profile::RING_SLOTS]; // 2 events per slot (start, end)
    int8_t      kind[mmq_profile::RING_SLOTS];
    int8_t      bucket[mmq_profile::RING_SLOTS];
    int         count = 0; // kernels recorded (set on capture/direct; retained for replays)
};

struct mq_host_state {
    bool initialized = false;
    bool enabled     = false;

    // graph boundary events (recorded OUTSIDE capture; safe to share across graphs)
    cudaEvent_t graph_start = nullptr;
    cudaEvent_t graph_end   = nullptr;

    // Per-kernel event ring. An event recorded during a stream capture becomes a
    // node of THAT graph, and cudaEventElapsedTime on a captured event fails
    // ("invalid argument") — so per-kernel events are only valid for DIRECT
    // (non-capture) execution. direct_ring is a single reusable ring for direct
    // graphs (its events are never captured, so they are freely reusable).
    event_ring * direct_ring  = nullptr;
    event_ring * current_ring = nullptr; // ring for the graph being recorded now (direct only)
    event_ring * prev_ring    = nullptr; // ring of the previous graph (to flush)
    bool        prev_pending  = false;   // a graph finished and needs flushing
    bool        ring_overflow = false;

    // in-kernel phase counters (device mem)
    mmq_profile::device_counters * d_counters = nullptr;

    // accumulators (host)
    double  ms[mmq_profile::NUM_KINDS][mmq_profile::NUM_BUCKETS] = {};
    double  graph_ms_total = 0.0;
    double  dev_cycles[mmq_profile::NUM_PHASES][mmq_profile::NUM_BUCKETS] = {};
    unsigned long long dev_tiles[mmq_profile::NUM_BUCKETS] = {};
    unsigned long long n_graphs = 0;
    unsigned long long n_direct_graphs = 0; // graphs with per-kernel event data
    unsigned long long n_kernels[mmq_profile::NUM_KINDS] = {};
    int     sm_clock_khz = 0;

    ~mq_host_state() {
        if (direct_ring) {
            for (int i = 0; i < 2 * mmq_profile::RING_SLOTS; i++)
                if (direct_ring->ev[i]) { cudaEventDestroy(direct_ring->ev[i]); }
            delete direct_ring;
        }
    }
};

// File-local helpers (static; forward-declared to satisfy -Wmissing-declarations).
// A named/anonymous namespace is avoided: GCC emits a spurious -Wextra-semi at a
// bogus column for anonymous namespaces.
static mq_host_state & mq_state();
static const char * mq_bucket_label(int b);
static void mq_print_summary(const char * tag);
static void mq_flush_last();
static void mq_atexit();

static mq_host_state & mq_state() {
    static mq_host_state s;
    return s;
}

static const char * mq_bucket_label(int b) {
    switch (b) {
        case 0:  return "ne11=1";
        case 1:  return "ne11=2..64";
        case 2:  return "ne11=65..256";
        case 3:  return "ne11=257..1024";
        default: return "ne11>1024";
    }
}

static void mq_print_summary(const char * tag) {
    mq_host_state & s = mq_state();
    // Per-kernel event data is only recorded for DIRECT graphs (a captured event
    // cannot be timed). When CUDA graphs are enabled, scale the direct-graph
    // in-MMQ-path total up to all graphs (assumes each graph's MMQ path ~ direct).
    const double in_mmq_direct = [&] {
        double t = 0.0;
        for (int k = 0; k < mmq_profile::NUM_KINDS; k++)
            for (int b = 0; b < mmq_profile::NUM_BUCKETS; b++) t += s.ms[k][b];
        return t;
    }();
    const double scale = (s.n_direct_graphs > 0) ? (double) s.n_graphs / (double) s.n_direct_graphs : 0.0;
    const double in_mmq = in_mmq_direct * scale;
    const double outside = s.graph_ms_total - in_mmq;

    fprintf(stderr, "\n=================== MMQ PROFILE [%s] ===================\n", tag);
    fprintf(stderr, "graphs: %llu   sm_clock: %d MHz   (device phases are FP4-only)\n",
            (unsigned long long) s.n_graphs, s.sm_clock_khz / 1000);
    fprintf(stderr, "graph wall (total)            : %10.2f ms\n", s.graph_ms_total);
    fprintf(stderr, "  in MMQ path (before+during+after): %10.2f ms  (%.1f%%)\n",
            in_mmq, s.graph_ms_total > 0.0 ? 100.0 * in_mmq / s.graph_ms_total : 0.0);
    fprintf(stderr, "  OUTSIDE MMQ (attention/GDN/norms/...): %10.2f ms  (%.1f%%)\n",
            outside, s.graph_ms_total > 0.0 ? 100.0 * (outside > 0.0 ? outside : 0.0) / s.graph_ms_total : 0.0);
    if (s.n_direct_graphs < s.n_graphs) {
        fprintf(stderr, "  NOTE: per-kernel (in-MMQ-path) data from %llu/%llu graphs (direct only), scaled x%.2f.\n",
                (unsigned long long) s.n_direct_graphs, (unsigned long long) s.n_graphs, scale);
        fprintf(stderr, "        Set GGML_CUDA_DISABLE_GRAPHS=1 for exact per-kernel timing on every graph.\n");
    }
    if (s.ring_overflow) {
        fprintf(stderr, "  WARNING: event ring overflowed; some kernels were not timed.\n");
    }
    fprintf(stderr, "\n  %-14s | %10s %10s %10s | %9s %9s %9s | %7s %7s %7s %7s\n",
            "bucket", "BQUANT", "MMQ", "FIXUP", "before%", "during%", "after%", "LOAD%", "MMA%", "EPI%", "BOOK%");
    fprintf(stderr, "  %-14s-+-%10s-%10s-%10s-+-%9s-%9s-%9s-+-%7s-%7s-%7s-%7s\n",
            "--------------", "----------", "----------", "----------", "---------", "---------", "---------",
            "-------", "-------", "-------", "-------");
    for (int b = 0; b < mmq_profile::NUM_BUCKETS; b++) {
        const double bq = s.ms[mmq_profile::KIND_BQUANT][b] * scale;
        const double mm = s.ms[mmq_profile::KIND_MMQ][b] * scale;
        const double fx = s.ms[mmq_profile::KIND_FIXUP][b] * scale;
        const double tot = bq + mm + fx;
        auto pct = [](double x, double d) { return d > 0.0 ? 100.0 * x / d : 0.0; };
        // device phase % within the MMQ kernel for this bucket
        double dl = s.dev_cycles[mmq_profile::PHASE_LOAD][b];
        double dm = s.dev_cycles[mmq_profile::PHASE_MMA ][b];
        double de = s.dev_cycles[mmq_profile::PHASE_EPI ][b];
        double db = s.dev_cycles[mmq_profile::PHASE_BOOK][b];
        const double dtot = dl + dm + de + db;
        fprintf(stderr, "  %-14s | %10.2f %10.2f %10.2f | %8.1f%% %8.1f%% %8.1f%% | %6.1f%% %6.1f%% %6.1f%% %6.1f%%\n",
                mq_bucket_label(b), bq, mm, fx,
                pct(bq, tot), pct(mm, tot), pct(fx, tot),
                pct(dl, dtot), pct(dm, dtot), pct(de, dtot), pct(db, dtot));
    }
    fprintf(stderr, "  (BQUANT/MMQ/FIXUP = host wall ms; before/during/after = share of that bucket's MMQ path;\n"
                    "   LOAD/MMA/EPI/BOOK = share of in-kernel SM time, FP4 only)\n");
    fprintf(stderr, "=================================================================\n\n");
    fflush(stderr);
}

// Sync the last in-flight graph and fold its timings into the host
// accumulators. Makes CUDA calls; must run while the context is alive.
static void mq_flush_last() {
    mq_host_state & s = mq_state();
    if (!s.enabled || !s.prev_pending) { return; }
    CUDA_CHECK(cudaEventSynchronize(s.graph_end));
    float gms = 0.0f;
    if (cudaEventElapsedTime(&gms, s.graph_start, s.graph_end) == cudaSuccess) {
        s.graph_ms_total += gms;
    }
    if (s.prev_ring) {
        for (int i = 0; i < s.prev_ring->count; i++) {
            float kms = 0.0f;
            if (cudaEventElapsedTime(&kms, s.prev_ring->ev[2 * i], s.prev_ring->ev[2 * i + 1]) == cudaSuccess) {
                s.ms[s.prev_ring->kind[i]][s.prev_ring->bucket[i]] += kms;
                s.n_kernels[s.prev_ring->kind[i]]++;
            }
        }
    }
    mmq_profile::device_counters dc;
    CUDA_CHECK(cudaMemcpy(&dc, s.d_counters, sizeof(mmq_profile::device_counters), cudaMemcpyDeviceToHost));
    for (int p = 0; p < mmq_profile::NUM_PHASES; p++) {
        for (int b = 0; b < mmq_profile::NUM_BUCKETS; b++) {
            s.dev_cycles[p][b] += (double) dc.cycles[p][b];
        }
    }
    for (int b = 0; b < mmq_profile::NUM_BUCKETS; b++) {
        s.dev_tiles[b] += dc.tiles[b];
    }
    CUDA_CHECK(cudaMemset(s.d_counters, 0, sizeof(mmq_profile::device_counters)));
    s.prev_pending = false;
}

// Zero the host accumulators so each backend lifetime (context) gets its own
// self-contained summary (llama-bench frees the backend between tests). The
// device counters are already zeroed per-flush.
static void mq_reset() {
    mq_host_state & s = mq_state();
    for (int k = 0; k < mmq_profile::NUM_KINDS; k++)
        for (int b = 0; b < mmq_profile::NUM_BUCKETS; b++) s.ms[k][b] = 0.0;
    s.graph_ms_total = 0.0;
    for (int p = 0; p < mmq_profile::NUM_PHASES; p++)
        for (int b = 0; b < mmq_profile::NUM_BUCKETS; b++) s.dev_cycles[p][b] = 0.0;
    for (int b = 0; b < mmq_profile::NUM_BUCKETS; b++) s.dev_tiles[b] = 0;
    s.n_graphs = 0;
    s.n_direct_graphs = 0;
    for (int k = 0; k < mmq_profile::NUM_KINDS; k++) s.n_kernels[k] = 0;
    s.ring_overflow = false;
}

// atexit fallback: print any data not already flushed+printed by a backend free
// (e.g. a run that exits without freeing the backend). This handler was
// registered before the CUDA runtime's own cleanup, so it runs first and the
// context is still alive.
static void mq_atexit() {
    mq_host_state & s = mq_state();
    if (s.enabled && s.n_graphs > 0) {
        mq_flush_last();
        mq_print_summary("final (atexit)");
        mq_reset();
    }
}

bool mmq_profile::enabled() {
    mq_host_state & s = mq_state();
    if (!s.initialized) {
        const char * env = std::getenv("GGML_MQ_PROFILE");
        s.enabled = env != nullptr && std::atoi(env) != 0;
        if (s.enabled) {
            CUDA_CHECK(cudaEventCreate(&s.graph_start));
            CUDA_CHECK(cudaEventCreate(&s.graph_end));
            event_ring * dr = new event_ring();
            for (int i = 0; i < 2 * mmq_profile::RING_SLOTS; i++) {
                dr->ev[i] = nullptr;
                CUDA_CHECK(cudaEventCreate(&dr->ev[i]));
            }
            s.direct_ring = dr;
            CUDA_CHECK(cudaMalloc(&s.d_counters, sizeof(mmq_profile::device_counters)));
            CUDA_CHECK(cudaMemset(s.d_counters, 0, sizeof(mmq_profile::device_counters)));
            int khz = 0;
            cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0);
            s.sm_clock_khz = khz;
            std::atexit(mq_atexit);
            fprintf(stderr, "[mmq-profile] enabled (ring %d slots, sm_clock=%d kHz)\n",
                    mmq_profile::RING_SLOTS, khz);
        }
        s.initialized = true;
    }
    return s.enabled;
}

mmq_profile::device_counters * mmq_profile::prof_buffer() {
    return mq_state().d_counters;
}

void mmq_profile::on_kernel_begin(int kind_idx, int bucket, cudaStream_t stream) {
    if (!enabled()) { return; }
    mq_host_state & s = mq_state();
    event_ring * r = s.current_ring;
    if (r == nullptr) { return; } // replay: no per-kernel event recording
    if (r->count >= mmq_profile::RING_SLOTS) { s.ring_overflow = true; return; }
    const int slot = r->count;
    r->kind[slot]   = (int8_t) kind_idx;
    r->bucket[slot] = (int8_t) bucket;
    CUDA_CHECK(cudaEventRecord(r->ev[2 * slot], stream));
}

void mmq_profile::on_kernel_end(cudaStream_t stream) {
    if (!enabled()) { return; }
    mq_host_state & s = mq_state();
    event_ring * r = s.current_ring;
    if (r == nullptr) { return; }
    if (r->count >= mmq_profile::RING_SLOTS) { s.ring_overflow = true; return; }
    const int slot = r->count;
    CUDA_CHECK(cudaEventRecord(r->ev[2 * slot + 1], stream));
    r->count = slot + 1;
}

void mmq_profile::graph_begin(cudaStream_t stream, mq_graph_kind kind) {
    if (!enabled()) { return; }
    mq_host_state & s = mq_state();
    // Flush the previous graph's timings (syncs its completion event).
    mq_flush_last();
    // Per-kernel events are only valid for DIRECT execution (a captured event
    // becomes a graph node that cannot be timed). For capture/replay we record no
    // per-kernel events; the graph wall and device counters still apply.
    if (kind == MQ_DIRECT) {
        s.current_ring = s.direct_ring;
        s.current_ring->count = 0;
        s.n_direct_graphs++;
    } else {
        s.current_ring = nullptr;
    }
    CUDA_CHECK(cudaEventRecord(s.graph_start, stream));
}

void mmq_profile::graph_end(cudaStream_t stream) {
    if (!enabled()) { return; }
    mq_host_state & s = mq_state();
    s.prev_ring = s.current_ring;
    s.current_ring = nullptr;
    CUDA_CHECK(cudaEventRecord(s.graph_end, stream));
    s.prev_pending = true;
    s.n_graphs++;
    if (s.n_graphs % mmq_profile::PRINT_EVERY_GRAPHS == 0) {
        mq_print_summary("periodic");
    }
}

void mmq_profile::shutdown() {
    mq_host_state & s = mq_state();
    if (!s.enabled) { return; }
    mq_flush_last();
    if (s.n_graphs > 0) {
        mq_print_summary("final");
        mq_reset(); // per-context summary; llama-bench frees the backend per test
    }
}

#else  // GGML_USE_HIP || GGML_USE_MUSA : no-op profiler (never runs the CUDA path)

bool mmq_profile::enabled() { return false; }
mmq_profile::device_counters * mmq_profile::prof_buffer() { return nullptr; }
void mmq_profile::on_kernel_begin(int, int, cudaStream_t) {}
void mmq_profile::on_kernel_end(cudaStream_t) {}
void mmq_profile::graph_begin(cudaStream_t, mq_graph_kind) {}
void mmq_profile::graph_end(cudaStream_t) {}
void mmq_profile::shutdown() {}

#endif  // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream, const ggml_prec prec_src1) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_MXFP4:
            // src1 at Q4 uses the native FP4 instructions, which are Blackwell-only
            if (prec_src1 == GGML_PREC_Q4) {
                mul_mat_q_case<GGML_TYPE_MXFP4, GGML_PREC_Q4>(ctx, args, stream);
                break;
            }
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            if (prec_src1 == GGML_PREC_Q4) {
                mul_mat_q_case<GGML_TYPE_NVFP4, GGML_PREC_Q4>(ctx, args, stream);
                break;
            }
            mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// overrides the src1 precision requested by the graph, "auto" keeps the requested one
static ggml_prec ggml_cuda_mmq_get_prec_env() {
    const char * env_c = getenv("GGML_CUDA_MMQ_PREC");
    if (env_c == nullptr) {
        return GGML_PREC_UNDEFINED;
    }
    std::string env_cpp = env_c;
    for (char & c : env_cpp) {
        c = std::tolower(c);
    }
    if (env_cpp == "q4") {
        return GGML_PREC_Q4;
    }
    if (env_cpp == "q8") {
        return GGML_PREC_Q8;
    }
    if (env_cpp != "auto") {
        GGML_LOG_WARN("%s: Unknown value for GGML_CUDA_MMQ_PREC: '%s'. Available: 'q4', 'q8', 'auto'.\n", __func__, env_cpp.c_str());
    }
    return GGML_PREC_UNDEFINED;
}

// src1 is quantized to Q8_1 unless the FP4 types can use 4-bit activations, in which case they
// default to the native W4A4 instructions on Blackwell.
static ggml_prec ggml_cuda_mmq_get_prec_src1(const ggml_tensor * src0, const ggml_tensor * dst, const int cc) {
    static const ggml_prec prec_env = ggml_cuda_mmq_get_prec_env();

    ggml_prec prec = prec_env;
    if (prec == GGML_PREC_UNDEFINED) {
        prec = (ggml_prec) ggml_get_op_params_i32(dst, 3);
    }

    // Q4 only for the FP4 types on Blackwell
    GGML_ASSERT(prec == GGML_PREC_UNDEFINED || prec == GGML_PREC_Q8 || prec == GGML_PREC_Q4);
    const bool can_use_q4 = (src0->type == GGML_TYPE_NVFP4 || src0->type == GGML_TYPE_MXFP4) && blackwell_mma_available(cc);
    if (prec == GGML_PREC_Q8 || !can_use_q4) {
        return GGML_PREC_Q8;
    }
    return GGML_PREC_Q4;
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool fallback = ne01 % 128 != 0;

    const ggml_prec prec_src1 = ggml_cuda_mmq_get_prec_src1(src0, dst, cc);

    const bool use_native_fp4 = prec_src1 == GGML_PREC_Q4;
    const bool use_native_fp  = use_native_fp4;
    const size_t y_block_size       = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ            : QK8_1_MMQ;

    if (!ids) {
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * y_block_size/y_values_per_block +
            ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
        ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
        if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
            src1_scale.alloc(ne13*ne12*ne11);
        }

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            const int prof_bucket = mmq_profile::bucket_of((int) ne11);
            mmq_profile::on_kernel_begin(mmq_profile::KIND_BQUANT, prof_bucket, stream);
            if (use_native_fp) {
                static constexpr size_t align_float8 = 32;
                const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());
            mmq_profile::on_kernel_end(stream);
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_fp ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) / ((use_native_fp4 ? QK_FP4_MMQ : QK8_1_MMQ) * sizeof(int)) :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
            src0->type == GGML_TYPE_NVFP4 && use_native_fp4 ? src1_scale.ptr : nullptr,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            ne1, ne1};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

    // gate/up activations are broadcast across experts (ne11 == 1): quantize each token once and
    // scatter to its slots. ids_src1 then holds the inverse map (token slot -> compact row).
    const bool dedup_bcast = ne11 == 1 && n_expert_used > 1;

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, /*write_inverse =*/ dedup_bcast, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block +
        ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne12) * sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
    if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
        src1_scale.alloc(ne12*n_expert_used);
    }

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        const int prof_bucket = mmq_profile::bucket_of((int) ne_get_rows);
        mmq_profile::on_kernel_begin(mmq_profile::KIND_BQUANT, prof_bucket, stream);
        if (use_native_fp) {
            static constexpr size_t align_float8 = 32;
            const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
            if (dedup_bcast) {
                quantize_scatter_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10,
                                        /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
            } else {
                quantize_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13,
                                        ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            }
        } else if (dedup_bcast) {
            quantize_scatter_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10,
                                    /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        mmq_profile::on_kernel_end(stream);
    }

    static_assert(QK_FP4_MMQ == 8 * QK_MXFP4, "QK_FP4_MMQ needs to be 8 * QK_MXFP4");
    const int64_t s12 = use_native_fp ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / ((use_native_fp4 ? QK_FP4_MMQ : QK8_1_MMQ) * sizeof(int)) :
                                        ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // Each expert only sees ne12*n_expert_used/ne02 tokens on average.
    // On RDNA3 and RDNA4 it is faster to pick the tile size against this value instead of ne12.
    int64_t ncols_opt = ne12;
    if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
        ncols_opt = (ne12*n_expert_used + ne02 - 1) / ne02;
    }

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
        src1_scale.ptr,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        ne12, ncols_opt};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
// -------------------------------------------------
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
// -------------------------------------------------
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
// -------------------------------------------------
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    // MMQ tiles require at least 48 KiB per-block shared memory; fall back to BLAS otherwise.
    {
        const int    id    = ggml_cuda_get_device();
        const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
        if (smpbo < 48 * 1024) {
            return false;
        }
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        // for MoE, mmq is faster even without native dp4a
        // TODO: check if cards older than pascal might benefit from this as well
        return cc >= GGML_CUDA_CC_PASCAL && n_experts > 0;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    // gfx900 (Vega 10), gfx909, and gfx90c lack native dp4a, losing to dequant + hipBLAS
    // for dense matrices; keep MMQ only for MoE, where the
    // hipBLAS path is much slower.
    if (cc == GGML_CUDA_CC_VEGA || GGML_CUDA_CC_IS_GCN_APU(cc)) {
        return n_experts > 0;
    }

    // MUSA: the MMQ kernels compute wrong values on PH1 (MTT S5000).
    if (cc == GGML_CUDA_CC_PH1) {
        return false;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
