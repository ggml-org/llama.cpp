//
// test_e2e_pipeline.cpp
//
// End-to-end integration test for the Tessera pipeline: corpus ->
// imatrix -> regime routing -> quantize -> fitness -> A/B comparison.
// Proves all modules compose correctly in a single flow.
//

#include "tessera-corpus.h"
#include "tessera-regime.h"
#include "tessera-quant.h"
#include "tessera-ab-harness.h"
#include "tessera-higgs.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static int g_fail = 0;

static void check(const char * name, bool ok) {
    if (!ok) {
        std::printf("FAIL %s\n", name);
        g_fail++;
    } else {
        std::printf("ok   %s\n", name);
    }
}

// test-local xorshift PRNG
static uint32_t e2e_rng = 42;

static float e2e_randf() {
    e2e_rng ^= e2e_rng << 13;
    e2e_rng ^= e2e_rng >> 17;
    e2e_rng ^= e2e_rng << 5;
    return (float)(e2e_rng & 0xFFFFFF) / (float)0x1000000;
}

static float e2e_randn() {
    float u1 = e2e_randf() + 1e-7f;
    float u2 = e2e_randf();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

int main() {
    const int64_t n_tokens  = 32;
    const int64_t in_dim    = 64;
    const int64_t out_dim   = 16;
    const int     n_tensors = 4;

    const char * names[4] = {
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_down.weight",
    };

    // ----------------------------------------------------------------
    // Step 1: generate synthetic calibration corpus
    // ----------------------------------------------------------------
    std::printf("[1/8] generating calibration corpus (%lld x %lld)\n",
                (long long)n_tokens, (long long)in_dim);

    ts_corpus_params cp = ts_corpus_default_params();
    cp.n_tokens = n_tokens;
    cp.in_dim   = in_dim;
    cp.seed     = 777;

    std::vector<float> corpus = ts_corpus_generate(&cp);
    check("corpus size", (int64_t)corpus.size() == n_tokens * in_dim);

    // derive per-channel imatrix from corpus column mean magnitudes
    std::vector<float> imatrix((size_t)in_dim, 0.0f);
    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t j = 0; j < in_dim; j++) {
            imatrix[(size_t)j] += fabsf(corpus[(size_t)(t * in_dim + j)]);
        }
    }
    for (int64_t j = 0; j < in_dim; j++) {
        imatrix[(size_t)j] /= (float)n_tokens;
    }
    check("imatrix derived", imatrix[0] > 0.0f);

    // ----------------------------------------------------------------
    // Step 2: create synthetic weights (4 tensors, 16x64 each)
    // ----------------------------------------------------------------
    std::printf("[2/8] creating synthetic weights (%lld x %lld x %d)\n",
                (long long)out_dim, (long long)in_dim, n_tensors);

    std::vector<std::vector<float>> weights(n_tensors);
    for (int t = 0; t < n_tensors; t++) {
        weights[t].resize((size_t)(out_dim * in_dim));
        float spread = 0.5f + 0.3f * (float)t;
        for (int64_t i = 0; i < out_dim * in_dim; i++) {
            weights[t][(size_t)i] = e2e_randn() * spread;
        }
    }
    check("weights allocated", weights[0].size() == (size_t)(out_dim * in_dim));

    // ----------------------------------------------------------------
    // Step 3: compute regime descriptors
    // ----------------------------------------------------------------
    std::printf("[3/8] computing regime descriptors\n");

    // vary imatrix tail behavior per tensor to exercise different routes:
    //   0,1: corpus-derived (well-behaved)
    //   2:   moderate outliers (kurtosis > 5)
    //   3:   extreme outliers  (kurtosis > 10)
    std::vector<std::vector<float>> imat(n_tensors);
    for (int t = 0; t < n_tensors; t++) {
        imat[t] = imatrix;
        if (t == 2) {
            for (int64_t j = 0; j < in_dim; j += 8) {
                imat[t][(size_t)j] *= 8.0f;
            }
        } else if (t == 3) {
            for (int64_t j = 0; j < in_dim; j += 4) {
                imat[t][(size_t)j] *= 20.0f;
            }
        }
    }

    std::vector<ts_regime_descriptor> descs(n_tensors);
    for (int t = 0; t < n_tensors; t++) {
        descs[t] = ts_regime_compute_descriptor(
            names[t], weights[t].data(), out_dim, in_dim,
            imat[t].data(), in_dim);
        std::printf("  %-24s family=%-10s kurt=%7.2f eff_rank=%.3f\n",
                    descs[t].tensor_name.c_str(), descs[t].family.c_str(),
                    (double)descs[t].kurtosis, (double)descs[t].eff_rank);
    }
    check("family attn_q", descs[0].family == "attn_q");
    check("family ffn_down", descs[3].family == "ffn_down");
    check("kurtosis finite", std::isfinite(descs[0].kurtosis));

    // ----------------------------------------------------------------
    // Step 4: route each tensor to its expert
    // ----------------------------------------------------------------
    std::printf("[4/8] routing tensors to experts\n");

    static const char * expert_names[TS_EXPERT_COUNT] = {
        "AWQ", "LRQ", "DartQuant", "FLRQ", "CHAMP-Q", "SEPTQ",
    };

    std::vector<ts_regime_routing> routings(n_tensors);
    for (int t = 0; t < n_tensors; t++) {
        routings[t] = ts_regime_classify(&descs[t]);
        std::printf("  %-24s -> %-10s conf=%.2f  %s\n",
                    routings[t].tensor_name.c_str(),
                    expert_names[routings[t].expert],
                    (double)routings[t].confidence,
                    routings[t].reason.c_str());
    }

    bool all_valid = true;
    for (int t = 0; t < n_tensors; t++) {
        if (routings[t].expert < 0 || routings[t].expert >= TS_EXPERT_COUNT) {
            all_valid = false;
        }
    }
    check("all experts valid", all_valid);

    ts_regime_summary summary = ts_regime_summarize(&routings, descs.data(), n_tensors);
    std::printf("  summary: mean_kurt=%.2f mean_eff_rank=%.3f\n",
                (double)summary.mean_kurtosis, (double)summary.mean_eff_rank);

    // ----------------------------------------------------------------
    // Step 5: quantize each tensor with routed expert params
    // ----------------------------------------------------------------
    std::printf("[5/8] quantizing tensors\n");

    std::vector<ts_quant_result_2d> qres(n_tensors);
    for (int t = 0; t < n_tensors; t++) {
        ts_quant_params_2d qp = {};
        qp.alpha          = 0.0f;
        qp.clip           = 1.0f;
        qp.max_outliers   = 4;
        qp.outlier_thresh = 2.0f;
        qp.awq_grid       = 5;

        int rc = ts_quantize_2d(
            weights[t].data(),
            imat[t].data(),     // act_scales
            nullptr, nullptr,   // no calib -> skip AWQ search
            imat[t].data(),     // imatrix
            out_dim, in_dim, 0,
            &qp, &qres[t]);

        check("quantize rc == 0", rc == 0);
        check("packed non-empty", !qres[t].packed.empty());
        std::printf("  %-24s mse=%.6g outliers=%zu\n",
                    names[t], (double)qres[t].mse, qres[t].outlier_cols.size());
    }

    // ----------------------------------------------------------------
    // Step 6: compute t_l^2 (relative Frobenius) per tensor
    // ----------------------------------------------------------------
    std::printf("[6/8] computing t_l^2 (relative Frobenius)\n");

    std::vector<float> t2(n_tensors);
    for (int t = 0; t < n_tensors; t++) {
        float w_norm_sq = 0.0f;
        for (int64_t i = 0; i < out_dim * in_dim; i++) {
            w_norm_sq += weights[t][(size_t)i] * weights[t][(size_t)i];
        }
        // mse = mean((W - W_hat)^2); t_l^2 = mse * n / ||W||_F^2
        float n = (float)(out_dim * in_dim);
        t2[t] = (w_norm_sq > 1e-12f) ? (qres[t].mse * n / w_norm_sq) : 0.0f;
        std::printf("  %-24s t_l^2=%.6g  ||W||_F^2=%.4g\n",
                    names[t], (double)t2[t], (double)w_norm_sq);
    }

    bool t2_ok = true;
    for (int t = 0; t < n_tensors; t++) {
        if (!std::isfinite(t2[t]) || t2[t] < 0.0f) {
            t2_ok = false;
        }
    }
    check("t2 finite and non-negative", t2_ok);

    // ----------------------------------------------------------------
    // Step 7: composite fitness (uniform alpha, Sum t_l^2)
    // ----------------------------------------------------------------
    std::printf("[7/8] computing composite fitness\n");

    float composite = 0.0f;
    for (int t = 0; t < n_tensors; t++) {
        composite += t2[t];
    }
    std::printf("  composite = %.6g\n", (double)composite);
    check("fitness > 0", composite > 0.0f);
    check("fitness finite", std::isfinite(composite));

    // serialize uniform alphas via HIGGS JSON (proves linkage)
    {
        ts_higgs_result hr;
        hr.n_valid = n_tensors;
        hr.n_fallback_uniform = n_tensors;
        hr.mean_alpha = 1.0f;
        for (int t = 0; t < n_tensors; t++) {
            ts_higgs_layer_result lr;
            lr.name      = names[t];
            lr.alpha_l   = 1.0f;
            lr.r_squared = 1.0f;
            lr.valid     = true;
            hr.layers.push_back(lr);
        }
        std::string hj = ts_higgs_to_json(&hr);
        check("higgs json non-empty", !hj.empty());
        std::printf("  higgs: %s\n", hj.c_str());
    }

    // ----------------------------------------------------------------
    // Step 8: A/B harness (t_l^2 as both proxy and kernel-direct)
    // ----------------------------------------------------------------
    std::printf("[8/8] running A/B harness\n");

    std::vector<ts_ab_tensor_scores> scores(n_tensors);
    for (int t = 0; t < n_tensors; t++) {
        scores[t].name              = names[t];
        scores[t].offline_proxy_mse = t2[t];
        scores[t].kernel_direct_t2  = t2[t];
        scores[t].alpha_l           = 1.0f;
    }

    ts_ab_harness_params abp;
    abp.n_heldout       = n_tensors;
    abp.measure_ranking = true;
    abp.verbose         = false;

    ts_ab_harness_result abr;
    int rc = ts_ab_run(&scores, &abp, &abr);
    check("ab_run rc == 0", rc == 0);
    check("ab scores count", (int64_t)abr.scores.size() == n_tensors);
    check("ab composite_offline > 0", abr.composite_offline > 0.0f);
    check("ab composite_kernel > 0", abr.composite_kernel > 0.0f);

    // identical proxy and kernel signals -> perfect rank agreement
    check("kendall_tau == 1", std::fabs(abr.kendall_tau - 1.0f) < 1e-6f);
    check("disagreement == 0", std::fabs(abr.ranking_disagreement) < 1e-6f);

    std::string receipt = ts_ab_receipt_json(&abr);
    check("receipt non-empty", !receipt.empty());
    check("receipt has kendall_tau", receipt.find("kendall_tau") != std::string::npos);
    std::printf("  receipt: %s\n", receipt.c_str());

    // ----------------------------------------------------------------
    // verdict
    // ----------------------------------------------------------------
    std::printf("\n========================================\n");
    std::printf("E2E: %d tensors quantized, fitness=%.6g\n",
                n_tensors, (double)composite);
    std::printf("========================================\n");
    std::printf("\n%s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
