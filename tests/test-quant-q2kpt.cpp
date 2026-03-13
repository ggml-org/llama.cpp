// test-quant-q2kpt.cpp
// Correctness and accuracy test for Q2_KPT (Q2_K with learned per-tensor float levels).
// Tests:
//   1. Level training sanity (levels in [0,1], strictly increasing)
//   2. Round-trip RMSE vs Q2_K and Q2_KPT-uniform-levels across distributions
//   3. Manual vec-dot consistency: for a single QK_K row, verify that
//      dequantize_row + manual dot == hand-rolled accumulation matches.

#include "ggml-backend.h"
#include "ggml.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// Q2_K/Q2_KPT block size (same as QK_K in ggml-common.h)
#define MY_QK_K 256

// ---------------------------------------------------------------------------
// Declarations for Q2_KPT internals (all in libggml-base.so)
// ---------------------------------------------------------------------------
#define Q2KPT_N_LEVELS 4

extern "C" {
    void         q2kpt_train_levels(const float * data, int64_t nrow, int64_t n_per_row,
                                     const float * imatrix, float levels_out[4]);
    void         q2kpt_set_levels(const float * levels);
    void         q2kpt_prepare_levels(int64_t nrows, int64_t n_per_row);
    const float *q2kpt_get_levels(void);
    size_t       quantize_q2_kpt(const float * src, void * dst,
                                  int64_t start_row, int64_t nrows, int64_t n_per_row,
                                  const float * imatrix);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static float rmse(const float * a, const float * b, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = (double)a[i] - (double)b[i];
        s += d * d;
    }
    return (float)std::sqrt(s / (double)n);
}

// Quantize float data to `type`, dequantize back, return RMSE.
static float std_quant_rmse(ggml_type type, const float * data,
                              size_t nrow, size_t n_per_row) {
    const size_t         rs = ggml_row_size(type, n_per_row);
    std::vector<uint8_t> qb(nrow * rs);
    std::vector<float>   dq(nrow * n_per_row);
    ggml_quantize_chunk(type, data, qb.data(), 0, nrow, n_per_row, nullptr);
    const ggml_type_traits * tr = ggml_get_type_traits(type);
    for (size_t r = 0; r < nrow; ++r) {
        tr->to_float(qb.data() + r * rs, dq.data() + r * n_per_row,
                     (int64_t)n_per_row, nullptr);
    }
    return rmse(data, dq.data(), nrow * n_per_row);
}

// Train Q2_KPT levels on data, quantize (with imatrix=1), dequantize, return RMSE.
// Q2_KPT has per-block levels (4 floats per 256-elem block), so we need to handle that.
static float q2kpt_rmse(const float * data, size_t nrow, size_t n_per_row,
                          float out_levels[4]) {
    std::vector<float> imatrix(n_per_row, 1.0f);

    // Train initial levels (used as starting point, but quantize will train per-row)
    q2kpt_train_levels(data, (int64_t)nrow, (int64_t)n_per_row,
                        imatrix.data(), out_levels);
    
    const int nb = (int)(n_per_row / MY_QK_K);  // blocks per row
    const size_t total_levels = nrow * nb * Q2KPT_N_LEVELS;
    std::vector<float> all_levels(total_levels);

    // Prepare level storage for per-block levels
    q2kpt_prepare_levels((int64_t)nrow, (int64_t)n_per_row);
    
    const size_t         rs = ggml_row_size(GGML_TYPE_Q2_KPT, n_per_row);
    std::vector<uint8_t> qb(nrow * rs);
    std::vector<float>   dq(nrow * n_per_row);

    for (size_t r = 0; r < nrow; ++r) {
        quantize_q2_kpt(data + r * n_per_row,
                         qb.data() + r * rs,
                         r, 1, (int64_t)n_per_row,
                         imatrix.data());
    }
    
    // Get the trained per-block levels
    const float * trained_levels = q2kpt_get_levels();
    memcpy(all_levels.data(), trained_levels, total_levels * sizeof(float));
    
    // Dequant each row with its own per-block levels
    const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_Q2_KPT);
    for (size_t r = 0; r < nrow; ++r) {
        tr->to_float(qb.data() + r * rs,
                     dq.data() + r * n_per_row,
                     (int64_t)n_per_row, 
                     all_levels.data() + r * nb * Q2KPT_N_LEVELS);
    }
    return rmse(data, dq.data(), nrow * n_per_row);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(void) {
    ggml_backend_load_all();

    std::mt19937 gen(42);
    bool all_ok = true;

    // -----------------------------------------------------------------------
    // Section 1: Level training sanity
    // -----------------------------------------------------------------------
    printf("=== Section 1: Trained level values ===\n");
    {
        const size_t nrow = 32, n_per_row = MY_QK_K;
        std::vector<float> data(nrow * n_per_row);
        std::normal_distribution<float> dist(0.0f, 0.02f);
        for (auto & v : data) v = dist(gen);

        std::vector<float> imatrix(n_per_row, 1.0f);
        float levels[4];
        q2kpt_train_levels(data.data(), (int64_t)nrow, (int64_t)n_per_row,
                            imatrix.data(), levels);
        printf("  Trained levels: [%.6f, %.6f, %.6f, %.6f]\n",
               levels[0], levels[1], levels[2], levels[3]);
        bool ordered = (levels[0] < levels[1]) && (levels[1] < levels[2]) &&
                       (levels[2] < levels[3]);
        bool in_range = (levels[0] >= 0.0f) && (levels[3] <= 1.0f);
        printf("  Strictly increasing: %s  In [0,1]: %s\n",
               ordered ? "YES" : "NO", in_range ? "YES" : "NO");
        if (!ordered || !in_range) {
            printf("  FAIL: levels are malformed!\n");
            all_ok = false;
        } else {
            printf("  PASS\n");
        }
    }
    printf("\n");

    // -----------------------------------------------------------------------
    // Section 2: Round-trip RMSE vs Q2_K
    // -----------------------------------------------------------------------
    printf("=== Section 2: Round-trip RMSE (ratio vs Q2_K) ===\n");
    printf("%-30s  %10s  %10s  %8s  %s\n", "Distribution", "Q2_K", "Q2_KPT", "ratio", "");
    printf("%-30s  %10s  %10s  %8s\n",
           "------------------------------", "----------", "----------", "--------");

    const size_t nrow = 64, n_per_row = 4 * MY_QK_K;

    auto run_dist_test = [&](const char * name, std::vector<float> & data) {
        float r_q2k = std_quant_rmse(GGML_TYPE_Q2_K, data.data(), nrow, n_per_row);
        float levels[4];
        float r_kpt = q2kpt_rmse(data.data(), nrow, n_per_row, levels);
        float ratio = r_kpt / (r_q2k + 1e-10f);
        // Sanity: Q2_KPT at same BPW should be within 3x of Q2_K
        bool ok = ratio < 3.0f;
        printf("%-30s  %10.6f  %10.6f  %8.4f  %s  levels=[%.3f,%.3f,%.3f,%.3f]\n",
               name, r_q2k, r_kpt, ratio, ok ? "PASS" : "FAIL",
               levels[0], levels[1], levels[2], levels[3]);
        if (!ok) all_ok = false;
    };

    {
        std::vector<float> data(nrow * n_per_row);
        std::normal_distribution<float> dist(0.0f, 0.02f);
        for (auto & v : data) v = dist(gen);
        run_dist_test("Gaussian(0, 0.02)", data);
    }
    {
        std::vector<float> data(nrow * n_per_row);
        std::exponential_distribution<float> edist(100.0f);
        std::uniform_int_distribution<int> sign_d(0, 1);
        for (auto & v : data) v = edist(gen) * (sign_d(gen) ? 1.0f : -1.0f);
        run_dist_test("Laplace(0, 0.01)", data);
    }
    {
        std::vector<float> data(nrow * n_per_row);
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        for (auto & v : data) v = dist(gen);
        run_dist_test("Uniform(-0.1, 0.1)", data);
    }
    printf("\n");

    // -----------------------------------------------------------------------
    // Section 3: Uniform-levels baseline: Q2_KPT with {0, 1/3, 2/3, 1}
    //   should behave similarly to Q2_K (both 2-bit, same BPW)
    // -----------------------------------------------------------------------
    printf("=== Section 3: Uniform-level baseline ===\n");
    {
        float uniform_levels_4[4] = {0.0f, 1.0f/3.0f, 2.0f/3.0f, 1.0f};
        q2kpt_set_levels(uniform_levels_4);

        std::vector<float> data(nrow * n_per_row);
        std::normal_distribution<float> dist(0.0f, 0.02f);
        for (auto & v : data) v = dist(gen);

        // Q2_K baseline
        float r_q2k = std_quant_rmse(GGML_TYPE_Q2_K, data.data(), nrow, n_per_row);

        // Q2_KPT with uniform levels (no re-training)
        // Need per-block levels: repeat uniform_levels for each block
        const int nb = n_per_row / MY_QK_K;  // blocks per row
        std::vector<float> uniform_levels(nb * 4);
        for (int b = 0; b < nb; ++b) {
            for (int k = 0; k < 4; ++k) {
                uniform_levels[b * 4 + k] = uniform_levels_4[k];
            }
        }

        const size_t rs = ggml_row_size(GGML_TYPE_Q2_KPT, n_per_row);
        std::vector<uint8_t> qb(nrow * rs);
        std::vector<float>   dq(nrow * n_per_row);
        for (size_t r = 0; r < nrow; ++r) {
            quantize_q2_kpt(data.data() + r * n_per_row,
                             qb.data() + r * rs, r, 1, (int64_t)n_per_row, nullptr);
        }
        const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_Q2_KPT);
        for (size_t r = 0; r < nrow; ++r) {
            tr->to_float(qb.data() + r * rs, dq.data() + r * n_per_row, (int64_t)n_per_row, uniform_levels.data());
        }
        float r_kpt_unif = rmse(data.data(), dq.data(), nrow * n_per_row);
        float ratio = r_kpt_unif / (r_q2k + 1e-10f);
        bool ok = ratio < 5.0f;  // with uniform levels Q2_KPT uses different quantizer
        printf("  Q2_K:          %.6f\n", r_q2k);
        printf("  Q2_KPT(unif):  %.6f  (ratio %.4f vs Q2_K)  %s\n",
               r_kpt_unif, ratio, ok ? "PASS" : "FAIL (too bad)");
        if (!ok) all_ok = false;
    }
    printf("\n");

    // -----------------------------------------------------------------------
    // Section 4: Dequant-only consistency check
    //   Quantize with known levels, dequantize, verify specific values.
    // -----------------------------------------------------------------------
    printf("=== Section 4: Dequant value spot-check ===\n");
    {
        // Use uniform levels so we know exactly what the mapping should be
        float ulev[4] = {0.0f, 1.0f/3.0f, 2.0f/3.0f, 1.0f};
        q2kpt_set_levels(ulev);

        // Create data spanning [0, 3] (to fill levels[k]*3 = {0,1,2,3})
        // A single QK_K block with values {0, 1, 2, 3, 0, 1, 2, 3, ...}
        std::vector<float> data(MY_QK_K);
        for (int i = 0; i < MY_QK_K; ++i) {
            // Values 0,1,2,3 repeating - with uniform levels these should quantize
            // to indices 0,1,2,3 and dequant as 0, d*1, d*2, d*3 for some d
            data[i] = (float)(i % 4);
        }

        const size_t rs = ggml_row_size(GGML_TYPE_Q2_KPT, MY_QK_K);
        std::vector<uint8_t> qb(rs);
        std::vector<float>   dq(MY_QK_K);

        quantize_q2_kpt(data.data(), qb.data(), 0, 1, MY_QK_K, nullptr);
        const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_Q2_KPT);
        tr->to_float(qb.data(), dq.data(), MY_QK_K, ulev);

        float err = rmse(data.data(), dq.data(), MY_QK_K);
        printf("  Input pattern {0,1,2,3,...} with uniform levels:\n");
        printf("  First 12 input:  %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n",
               data[0], data[1], data[2], data[3], data[4], data[5],
               data[6], data[7], data[8], data[9], data[10], data[11]);
        printf("  First 12 dequant: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
               dq[0], dq[1], dq[2], dq[3], dq[4], dq[5],
               dq[6], dq[7], dq[8], dq[9], dq[10], dq[11]);
        printf("  RMSE: %.6f\n", err);
        // Expect dequant ≈ input * scale (some small error due to scale quantization)
        bool ok = err < 0.5f;  // very generous: at most 0.5 absolute error on 0-3 range
        printf("  %s\n", ok ? "PASS" : "FAIL (reconstruction wildly wrong)");
        if (!ok) all_ok = false;
    }
    printf("\n");

    // -----------------------------------------------------------------------
    // Section 5: Vec-dot via ggml_mul_mat vs dequant-based reference
    // This exercises the full inference path: level dispatch + vec_dot kernel.
    // -----------------------------------------------------------------------
    printf("=== Section 5: Vec-dot via ggml_mul_mat ===\n");

    // Get CPU backend after load_all
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        printf("  CPU backend device not found, skipping\n");
    } else {
        ggml_backend_t backend = ggml_backend_dev_init(cpu_dev, nullptr);

        // Set trained levels
        float ml_levels[4];
        {
            const int nrow2 = 16, nprow2 = MY_QK_K;
            std::vector<float> td(nrow2 * nprow2);
            std::normal_distribution<float> dist(0.0f, 0.02f);
            for (auto & v : td) v = dist(gen);
            std::vector<float> im(nprow2, 1.0f);
            q2kpt_train_levels(td.data(), nrow2, nprow2, im.data(), ml_levels);
            q2kpt_set_levels(ml_levels);
        }
        printf("  Levels: [%.4f, %.4f, %.4f, %.4f]\n",
               ml_levels[0], ml_levels[1], ml_levels[2], ml_levels[3]);

        const int ne0 = 4 * MY_QK_K;  // columns (embedding)
        const int ne1 = 4;             // rows (output features)

        // Generate weight and activation data
        std::vector<float> weights(ne1 * ne0), acts(ne0);
        {
            std::normal_distribution<float> dw(0.0f, 0.02f);
            std::normal_distribution<float> da(0.0f, 1.0f);
            for (auto & v : weights) v = dw(gen);
            for (auto & v : acts)    v = da(gen);
        }

        // Quantize weights to Q2_KPT
        const size_t rs = ggml_row_size(GGML_TYPE_Q2_KPT, ne0);
        std::vector<uint8_t> qw(ne1 * rs);
        for (int r = 0; r < ne1; ++r)
            quantize_q2_kpt(weights.data() + r * ne0, qw.data() + r * rs, r, 1, ne0, nullptr);

        // Reference: dequant weights x float acts
        const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_Q2_KPT);
        std::vector<float> dw(ne1 * ne0);
        for (int r = 0; r < ne1; ++r)
            tr->to_float(qw.data() + r * rs, dw.data() + r * ne0, ne0, ml_levels);

        std::vector<float> ref_out(ne1, 0.0f);
        for (int r = 0; r < ne1; ++r)
            for (int c = 0; c < ne0; ++c)
                ref_out[r] += dw[r * ne0 + c] * acts[c];

        // ggml_mul_mat - use no_alloc so ggml_backend_alloc_ctx_tensors works
        const size_t ctx_mem = 1 * 1024 * 1024;  // just for tensor metadata
        ggml_init_params params = { ctx_mem, nullptr, true };
        ggml_context * ctx = ggml_init(params);

        ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_Q2_KPT, ne0, ne1);
        ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
        ggml_tensor * y = ggml_mul_mat(ctx, W, x);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, y);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        W->quant_levels = ml_levels;  // set per-tensor levels for inference
        ggml_backend_tensor_set(W, qw.data(), 0, (int64_t)(ne1 * rs));
        ggml_backend_tensor_set(x, acts.data(), 0, ne0 * sizeof(float));

        ggml_backend_graph_compute(backend, graph);

        std::vector<float> got_out(ne1);
        ggml_backend_tensor_get(y, got_out.data(), 0, ne1 * sizeof(float));

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);

        float max_rel = 0;
        for (int r = 0; r < ne1; ++r) {
            float rel = std::abs(got_out[r] - ref_out[r]) /
                        (std::abs(ref_out[r]) + 1e-9f);
            max_rel = std::max(max_rel, rel);
            printf("  row %d: ref=%.6f  got=%.6f  rel_err=%.3e\n",
                   r, ref_out[r], got_out[r], rel);
        }
        bool vd_ok = max_rel < 0.01f;
        printf("  max_rel_err=%.3e  %s\n", max_rel, vd_ok ? "PASS" : "FAIL");
        if (!vd_ok) all_ok = false;
    }
    printf("\n");

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    printf("=== Summary: %s ===\n", all_ok ? "ALL PASS" : "SOME FAILURES");

    return all_ok ? 0 : 1;
}
