// test-log-soft-max.cpp -- verify ggml_log_soft_max / ggml_log_soft_max_inplace
// match PyTorch float32 log_softmax (reference golden in test-log-soft-max-golden.h).
//
// Coverage:
//   - Precision baseline = PyTorch 2.9.1 float32 log_softmax, tolerance TOL = 5e-5
//     (measured FP32 sequential accumulation vs torch-fp32 max diff ~1.9e-6, leaving
//      ~25x headroom for cross-platform libm differences).
//   - In-place vs out-of-place equivalence.
//   - Multi-row batching: each row is independent.
//   - Normalization property: sum(exp(out)) == 1.

#include "ggml.h"
#include "ggml-cpu.h"

#include "test-log-soft-max-golden.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static const float TOL = 5e-5f;

// Run log_soft_max on a [ne0, ne1, ne2, ne3] input on the CPU and return the flattened output.
static std::vector<float> run_lsm_4d(const std::vector<float> & in, int64_t ne0, int64_t ne1,
                                     int64_t ne2, int64_t ne3, bool inplace, int n_threads) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ 64ull * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
    memcpy(x->data, in.data(), in.size() * sizeof(float));

    struct ggml_tensor * y = inplace ? ggml_log_soft_max_inplace(ctx, x)
                                     : ggml_log_soft_max(ctx, x);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    std::vector<float> out(in.size());
    memcpy(out.data(), y->data, out.size() * sizeof(float));

    ggml_free(ctx);
    return out;
}

// 2D convenience wrapper, single-threaded for deterministic output.
static std::vector<float> run_lsm(const std::vector<float> & in, int64_t ne0, int64_t ne1, bool inplace) {
    return run_lsm_4d(in, ne0, ne1, 1, 1, inplace, 1);
}

// Double-precision reference log_softmax over each contiguous ne0-length row.
static std::vector<float> ref_log_softmax_rows(const std::vector<float> & in, int64_t ne0) {
    std::vector<float> out(in.size());
    const int64_t nrows = (int64_t) in.size() / ne0;
    for (int64_t r = 0; r < nrows; ++r) {
        const float * x = in.data() + r * ne0;
        double mx = -INFINITY;
        for (int64_t i = 0; i < ne0; ++i) mx = std::max(mx, (double) x[i]);
        double sum = 0.0;
        for (int64_t i = 0; i < ne0; ++i) sum += std::exp((double) x[i] - mx);
        const double logsum = std::log(sum);
        for (int64_t i = 0; i < ne0; ++i) out[r * ne0 + i] = (float) (((double) x[i] - mx) - logsum);
    }
    return out;
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

int main() {
    bool all_ok = true;
    float worst = 0.0f;

    // 1) Per-case alignment with the PyTorch reference (out-of-place and in-place).
    for (size_t c = 0; c < k_lsm_golden.size(); ++c) {
        const lsm_golden_case & gc = k_lsm_golden[c];
        for (int ip = 0; ip < 2; ++ip) {
            const bool inplace = (ip == 1);
            std::vector<float> out = run_lsm(gc.in, gc.n, 1, inplace);
            const float d = max_abs_diff(out, gc.expected);
            worst = std::max(worst, d);
            const bool ok = d <= TOL;
            all_ok = all_ok && ok;
            printf("[golden] case n=%-4d %-11s max|diff|=%.3e  %s\n",
                   gc.n, inplace ? "(inplace)" : "(out)", d, ok ? "OK" : "FAIL");
        }
    }

    // 2) In-place and out-of-place must match element by element.
    for (size_t c = 0; c < k_lsm_golden.size(); ++c) {
        const lsm_golden_case & gc = k_lsm_golden[c];
        std::vector<float> a = run_lsm(gc.in, gc.n, 1, false);
        std::vector<float> b = run_lsm(gc.in, gc.n, 1, true);
        const float d = max_abs_diff(a, b);
        const bool ok = d == 0.0f;
        all_ok = all_ok && ok;
        printf("[inplace==out] case n=%-4d max|diff|=%.3e  %s\n", gc.n, d, ok ? "OK" : "FAIL");
    }

    // 3) Multi-row batching: take the n=64 case, build 3 distinct rows, and require each
    //    row of the batched result to match the single-row result.
    {
        const lsm_golden_case * p64 = nullptr;
        for (const auto & gc : k_lsm_golden) {
            if (gc.n == 64) { p64 = &gc; break; }
        }
        if (p64) {
            const int64_t n = p64->n;
            std::vector<float> row0 = p64->in;                          // original
            std::vector<float> row1(p64->in.rbegin(), p64->in.rend());  // reversed
            std::vector<float> row2 = p64->in;                          // scaled
            for (auto & v : row2) v *= 0.5f;

            std::vector<float> batch;
            batch.insert(batch.end(), row0.begin(), row0.end());
            batch.insert(batch.end(), row1.begin(), row1.end());
            batch.insert(batch.end(), row2.begin(), row2.end());

            std::vector<float> bout = run_lsm(batch, n, 3, false);

            std::vector<float> r0(bout.begin() + 0 * n, bout.begin() + 1 * n);
            std::vector<float> r1(bout.begin() + 1 * n, bout.begin() + 2 * n);
            std::vector<float> r2(bout.begin() + 2 * n, bout.begin() + 3 * n);

            const float d0 = max_abs_diff(r0, run_lsm(row0, n, 1, false));
            const float d1 = max_abs_diff(r1, run_lsm(row1, n, 1, false));
            const float d2 = max_abs_diff(r2, run_lsm(row2, n, 1, false));
            const bool ok = d0 == 0.0f && d1 == 0.0f && d2 == 0.0f;
            all_ok = all_ok && ok;
            printf("[batch rows] n=64x3 diffs={%.3e,%.3e,%.3e}  %s\n", d0, d1, d2, ok ? "OK" : "FAIL");
        }
    }

    // 4) Normalization property: sum(exp(log_softmax)) == 1.
    for (size_t c = 0; c < k_lsm_golden.size(); ++c) {
        const lsm_golden_case & gc = k_lsm_golden[c];
        std::vector<float> out = run_lsm(gc.in, gc.n, 1, false);
        double s = 0.0;
        for (float v : out) s += std::exp((double) v);
        const double d = std::fabs(s - 1.0);
        const bool ok = d <= 1e-4;
        all_ok = all_ok && ok;
        printf("[sum exp] case n=%-4d sum=%.7f |sum-1|=%.3e  %s\n", gc.n, s, d, ok ? "OK" : "FAIL");
    }

    // 5) 4D striding: exercise the i02/i03 loops and nb strides, comparing against a
    //    double-precision per-row reference (out-of-place and in-place).
    {
        const int64_t ne0 = 11, ne1 = 3, ne2 = 2, ne3 = 2;
        std::vector<float> in((size_t) (ne0 * ne1 * ne2 * ne3));
        for (size_t i = 0; i < in.size(); ++i) {
            in[i] = std::sin((float) i * 0.7f) * 5.0f + std::cos((float) i * 0.13f) * 3.0f;
        }
        std::vector<float> ref = ref_log_softmax_rows(in, ne0);

        std::vector<float> o1 = run_lsm_4d(in, ne0, ne1, ne2, ne3, false, 1);
        std::vector<float> o2 = run_lsm_4d(in, ne0, ne1, ne2, ne3, true, 1);
        const float d1 = max_abs_diff(o1, ref);
        const float d2 = max_abs_diff(o2, ref);
        const bool ok = d1 <= TOL && d2 <= TOL;
        all_ok = all_ok && ok;
        printf("[4d strides] ne=[%lld,%lld,%lld,%lld] out=%.3e inplace=%.3e  %s\n",
               (long long) ne0, (long long) ne1, (long long) ne2, (long long) ne3, d1, d2, ok ? "OK" : "FAIL");

        // 6) threading determinism: 1 vs 4 threads must be bit-identical (rows are independent).
        std::vector<float> ot = run_lsm_4d(in, ne0, ne1, ne2, ne3, false, 4);
        const float dt = max_abs_diff(ot, o1);
        const bool tok = dt == 0.0f;
        all_ok = all_ok && tok;
        printf("[threads 1v4] max|diff|=%.3e  %s\n", dt, tok ? "OK" : "FAIL");
    }

    printf("\nworst golden max|diff| = %.3e (TOL=%.1e)\n", worst, TOL);
    printf("%s\n", all_ok ? "ALL OK" : "FAILED");
    return all_ok ? 0 : 1;
}
