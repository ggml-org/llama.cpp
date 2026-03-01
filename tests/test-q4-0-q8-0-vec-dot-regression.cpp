#include "../ggml/src/ggml-cpu/quants.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static int parse_env_int(const char * name, int fallback, int min_value) {
    const char * env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return fallback;
    }
    const int v = std::atoi(env);
    return std::max(min_value, v);
}

static double parse_env_double(const char * name, double fallback, double min_value) {
    const char * env = std::getenv(name);
    if (!env || env[0] == '\0') {
        return fallback;
    }
    const double v = std::atof(env);
    return std::max(min_value, v);
}

using vec_dot_fn_t = void (*)(int, float *, size_t, const void *, size_t, const void *, size_t, int);

static int64_t bench_vec_dot_us(vec_dot_fn_t     fn,
                                int              n,
                                int              rows,
                                int              iters,
                                size_t           q4_row_size,
                                size_t           q8_row_size,
                                const uint8_t *  q4_data,
                                const uint8_t *  q8_data,
                                volatile float * sink) {
    const auto t0    = std::chrono::high_resolution_clock::now();
    float      accum = 0.0f;
    for (int it = 0; it < iters; ++it) {
        for (int r = 0; r < rows; ++r) {
            float        out = 0.0f;
            const void * vx  = q4_data + static_cast<size_t>(r) * q4_row_size;
            const void * vy  = q8_data + static_cast<size_t>(r) * q8_row_size;
            fn(n, &out, 0, vx, 0, vy, 0, 1);
            accum += out;
        }
    }
    *sink += accum;
    const auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

int main() {
    ggml_cpu_init();

    int          n               = parse_env_int("GGML_TEST_Q4Q8_VECDOT_N", 4096, QK8_0);
    int          rows            = parse_env_int("GGML_TEST_Q4Q8_VECDOT_ROWS", 96, 1);
    int          iters           = parse_env_int("GGML_TEST_Q4Q8_VECDOT_ITERS", 256, 1);
    const double min_speedup_x86 = parse_env_double("GGML_TEST_Q4Q8_VECDOT_MIN_SPEEDUP_X86", 1.05, 0.1);
    (void) min_speedup_x86;

    n = (n / QK8_0) * QK8_0;
    if (n <= 0) {
        n = QK8_0;
    }

    const size_t q4_row_size = ggml_row_size(GGML_TYPE_Q4_0, n);
    const size_t q8_row_size = ggml_row_size(GGML_TYPE_Q8_0, n);

    std::vector<float>   src0(static_cast<size_t>(rows) * n);
    std::vector<float>   src1(static_cast<size_t>(rows) * n);
    std::vector<uint8_t> q4_data(static_cast<size_t>(rows) * q4_row_size);
    std::vector<uint8_t> q8_data(static_cast<size_t>(rows) * q8_row_size);

    std::mt19937                          rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float & v : src0) {
        v = dist(rng);
    }
    for (float & v : src1) {
        v = dist(rng);
    }

    for (int r = 0; r < rows; ++r) {
        const float * row0 = src0.data() + static_cast<size_t>(r) * n;
        const float * row1 = src1.data() + static_cast<size_t>(r) * n;
        quantize_row_q4_0(row0, q4_data.data() + static_cast<size_t>(r) * q4_row_size, n);
        quantize_row_q8_0(row1, q8_data.data() + static_cast<size_t>(r) * q8_row_size, n);
    }

    float max_abs_diff = 0.0f;
    for (int r = 0; r < rows; ++r) {
        const void * vx  = q4_data.data() + static_cast<size_t>(r) * q4_row_size;
        const void * vy  = q8_data.data() + static_cast<size_t>(r) * q8_row_size;
        float        opt = 0.0f;
        float        ref = 0.0f;
        ggml_vec_dot_q4_0_q8_0(n, &opt, 0, vx, 0, vy, 0, 1);
        ggml_vec_dot_q4_0_q8_0_generic(n, &ref, 0, vx, 0, vy, 0, 1);
        max_abs_diff = std::max(max_abs_diff, std::fabs(opt - ref));
    }

    if (max_abs_diff > 1e-3f) {
        std::fprintf(stderr, "FAIL: max abs diff too large: %.6f\n", max_abs_diff);
        return 1;
    }

    volatile float sink = 0.0f;
    for (int i = 0; i < 3; ++i) {
        (void) bench_vec_dot_us(ggml_vec_dot_q4_0_q8_0, n, rows, 8, q4_row_size, q8_row_size, q4_data.data(),
                                q8_data.data(), &sink);
        (void) bench_vec_dot_us(ggml_vec_dot_q4_0_q8_0_generic, n, rows, 8, q4_row_size, q8_row_size, q4_data.data(),
                                q8_data.data(), &sink);
    }

    std::vector<int64_t> opt_times;
    std::vector<int64_t> ref_times;
    opt_times.reserve(5);
    ref_times.reserve(5);
    for (int r = 0; r < 5; ++r) {
        opt_times.push_back(bench_vec_dot_us(ggml_vec_dot_q4_0_q8_0, n, rows, iters, q4_row_size, q8_row_size,
                                             q4_data.data(), q8_data.data(), &sink));
        ref_times.push_back(bench_vec_dot_us(ggml_vec_dot_q4_0_q8_0_generic, n, rows, iters, q4_row_size, q8_row_size,
                                             q4_data.data(), q8_data.data(), &sink));
    }

    std::sort(opt_times.begin(), opt_times.end());
    std::sort(ref_times.begin(), ref_times.end());
    const double opt_us  = static_cast<double>(opt_times[opt_times.size() / 2]);
    const double ref_us  = static_cast<double>(ref_times[ref_times.size() / 2]);
    const double speedup = ref_us / std::max(1.0, opt_us);

    std::fprintf(stderr,
                 "[q4_0_q8_0_vec_dot] n=%d rows=%d iters=%d opt_us=%.0f ref_us=%.0f speedup=%.3fx max_diff=%.6f\n", n,
                 rows, iters, opt_us, ref_us, speedup, max_abs_diff);

#if defined(__x86_64__) || defined(__i386__)
#    if defined(__AVX2__) || defined(__AVX512F__)
    if (speedup < min_speedup_x86) {
        std::fprintf(stderr,
                     "FAIL: speedup %.3fx below threshold %.3fx (set GGML_TEST_Q4Q8_VECDOT_MIN_SPEEDUP_X86 to tune)\n",
                     speedup, min_speedup_x86);
        return 1;
    }
#    endif
#endif

    if (std::isnan(sink)) {
        std::fprintf(stderr, "FAIL: invalid sink value\n");
        return 1;
    }

    std::fprintf(stderr, "PASS\n");
    return 0;
}
