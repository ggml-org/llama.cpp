#include "fp16-mkl.h"

#include <mkl.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <tuple>

// Use the converter selected by this CPU backend (including its SIMD support).
extern "C" void ggml_cpu_fp16_to_fp32(const ggml_fp16_t *, float *, int64_t);

namespace {
struct Config {
    bool enabled = false;
    bool threaded = false;
    bool stats = false;
    Config() {
        const char * mode = std::getenv("GGML_FP16_MKL_MODE");
        if (mode && std::strcmp(mode, "auto") == 0) enabled = true;
        else if (mode && std::strcmp(mode, "off") != 0)
            std::fprintf(stderr, "GGML FP16 MKL: unsupported mode '%s'; using off (choose auto/off)\n", mode);
        const char * parallel = std::getenv("GGML_FP16_MKL_PARALLEL");
        if (parallel && std::strcmp(parallel, "threaded") == 0) threaded = true;
        else if (parallel && std::strcmp(parallel, "split") != 0)
            std::fprintf(stderr, "GGML FP16 MKL: unsupported parallel mode '%s'; using split\n", parallel);
        const char * value = std::getenv("GGML_FP16_MKL_STATS");
        stats = value && value[0] && std::strcmp(value, "0") != 0;
    }
};
const Config & config() { static const Config value; return value; }

bool candidate(const ggml_tensor * op) {
    return op && op->op == GGML_OP_MUL_MAT && op->src[0] && op->src[1] &&
        op->src[0]->type == GGML_TYPE_F16 && op->type == GGML_TYPE_F32 &&
        (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32);
}

// Restrict the first implementation to packed 2D inputs/output and LP64-sized
// GEMMs. Unsupported layouts return to the existing ggml implementation.
size_t workspace(const ggml_tensor * op) {
    if (!candidate(op)) return 0;
    const auto * a = op->src[0];
    const auto * b = op->src[1];
    const int64_t m = a->ne[1], n = b->ne[1], k = a->ne[0];
    const auto imax = std::numeric_limits<MKL_INT>::max();
    if (m <= 0 || n <= 0 || k <= 0 || m > imax || n > imax || k > imax ||
        b->ne[0] != k || op->ne[0] != m || op->ne[1] != n ||
        a->ne[2] != 1 || a->ne[3] != 1 || b->ne[2] != 1 || b->ne[3] != 1 ||
        op->ne[2] != 1 || op->ne[3] != 1 ||
        !ggml_is_contiguous(a) || !ggml_is_contiguous(b) || !ggml_is_contiguous(op)) return 0;
    const size_t rows = size_t(m) + (b->type == GGML_TYPE_F16 ? size_t(n) : 0);
    const size_t limit = (std::numeric_limits<size_t>::max() - 63) / sizeof(float);
    if (rows > limit / size_t(k)) return 0;
    return 63 + rows * size_t(k) * sizeof(float);
}

using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct ShapeStats {
    uint64_t dispatch = 0, fallback = 0, worker_calls = 0;
    size_t workspace_bytes = 0;
    double conversion_worker_ms = 0, sgemm_worker_ms = 0;
};
struct Stats {
    std::mutex mutex;
    std::map<std::tuple<int64_t, int64_t, int64_t>, ShapeStats> shapes;
    ~Stats() {
        std::fprintf(stderr, "GGML_FP16_MKL_STATS mode=%s parallel=%s timings=sum_of_worker_durations\n",
            config().enabled ? "auto" : "off", config().threaded ? "threaded" : "split");
        for (const auto & item : shapes) {
            const auto & s = item.second;
            std::fprintf(stderr, "GGML_FP16_MKL_SHAPE M=%lld N=%lld K=%lld dispatch=%llu fallback=%llu"
                " conversion_worker_ms=%.6f sgemm_worker_ms=%.6f worker_calls=%llu workspace_bytes=%zu\n",
                (long long) std::get<0>(item.first), (long long) std::get<1>(item.first),
                (long long) std::get<2>(item.first), (unsigned long long) s.dispatch,
                (unsigned long long) s.fallback, s.conversion_worker_ms, s.sgemm_worker_ms,
                (unsigned long long) s.worker_calls, s.workspace_bytes);
        }
    }
};
void record(const ggml_tensor * op, bool dispatch, bool fallback, bool worker,
            size_t bytes, double conversion_ms, double gemm_ms) {
    if (!config().stats) return;
    static Stats stats;
    std::lock_guard<std::mutex> lock(stats.mutex);
    auto & s = stats.shapes[{op->src[0]->ne[1], op->src[1]->ne[1], op->src[0]->ne[0]}];
    s.dispatch += dispatch;
    s.fallback += fallback;
    s.worker_calls += worker;
    s.workspace_bytes = std::max(s.workspace_bytes, bytes);
    s.conversion_worker_ms += conversion_ms;
    s.sgemm_worker_ms += gemm_ms;
}

// Balanced partition without count * ith overflowing.
size_t boundary(size_t count, int ith, int nth) {
    return count / size_t(nth) * size_t(ith) + std::min(size_t(ith), count % size_t(nth));
}
void convert_part(const ggml_fp16_t * src, float * dst, size_t count, int ith, int nth) {
    const size_t begin = boundary(count, ith, nth);
    const size_t end = boundary(count, ith + 1, nth);
    if (begin != end) ggml_cpu_fp16_to_fp32(src + begin, dst + begin, int64_t(end - begin));
}
struct LocalThreads {
    int previous;
    explicit LocalThreads(int count) : previous(mkl_set_num_threads_local(count)) {}
    ~LocalThreads() { mkl_set_num_threads_local(previous); }
};
} // namespace

extern "C" size_t ggml_fp16_mkl_workspace_size(const ggml_tensor * op) {
    return config().enabled ? workspace(op) : 0;
}

extern "C" bool ggml_fp16_mkl_compute(const ggml_compute_params * params, ggml_tensor * op) {
    if (!candidate(op)) return false;
    const auto & cfg = config();
    const size_t required = cfg.enabled && !params->use_ref ? workspace(op) : 0;
    if (!required || !params->wdata || params->wsize < required || !params->threadpool ||
        params->nth < 1 || params->ith < 0 || params->ith >= params->nth) {
        if (params->ith == 0) record(op, false, true, false, 0, 0, 0);
        return false;
    }

    const auto * a = op->src[0];
    const auto * b = op->src[1];
    const size_t m = size_t(a->ne[1]), n = size_t(b->ne[1]), k = size_t(a->ne[0]);
    const uintptr_t address = reinterpret_cast<uintptr_t>(params->wdata);
    float * af = reinterpret_cast<float *>((address + 63) & ~uintptr_t(63));
    float * bf_scratch = af + m * k;
    // Preserve F32 activations: do not round them through FP16.
    const float * bf = b->type == GGML_TYPE_F32 ? static_cast<const float *>(b->data) : bf_scratch;

    const auto c0 = cfg.stats ? Clock::now() : Clock::time_point{};
    convert_part(static_cast<const ggml_fp16_t *>(a->data), af, m * k, params->ith, params->nth);
    if (b->type == GGML_TYPE_F16)
        convert_part(static_cast<const ggml_fp16_t *>(b->data), bf_scratch, n * k, params->ith, params->nth);
    const double conversion_ms = cfg.stats ? elapsed_ms(c0) : 0;
    ggml_barrier(params->threadpool);

    double gemm_ms = 0;
    bool worker = false;
    if (!cfg.threaded || params->ith == 0) {
        const size_t first = cfg.threaded ? 0 : boundary(m, params->ith, params->nth);
        const size_t end = cfg.threaded ? m : boundary(m, params->ith + 1, params->nth);
        if (first != end) {
            LocalThreads threads(cfg.threaded ? params->nth : 1);
            const auto g0 = cfg.stats ? Clock::now() : Clock::time_point{};
            cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                MKL_INT(end - first), MKL_INT(n), MKL_INT(k), 1.0f,
                af + first * k, MKL_INT(k), bf, MKL_INT(k), 0.0f,
                static_cast<float *>(op->data) + first, MKL_INT(m));
            gemm_ms = cfg.stats ? elapsed_ms(g0) : 0;
            worker = true;
        }
    }
    ggml_barrier(params->threadpool);
    record(op, params->ith == 0, false, worker, required, conversion_ms, gemm_ms);
    // Keep all workers in this op until statistics updates have completed.
    if (cfg.stats) ggml_barrier(params->threadpool);
    return true;
}
