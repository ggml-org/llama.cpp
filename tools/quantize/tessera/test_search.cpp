#include "tessera-search.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

static int test_lrq() {
    // 8x16 matrix, rank 2, verify MSE decreases from baseline
    const int64_t out_dim = 8, in_dim = 16, rank = 2;
    std::vector<float> W(out_dim * in_dim);
    uint32_t rng = 123;
    for (auto & v : W) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        v = ((float)(rng & 0xFFFF) / 65536.0f - 0.5f) * 2.0f;
    }

    // baseline MSE: ternarize W directly
    float thresh = 0.0f;
    for (auto v : W) thresh += fabsf(v);
    thresh /= (float)W.size();
    float baseline_mse = 0.0f;
    for (auto v : W) {
        float q = (fabsf(v) >= thresh) ? (v > 0 ? thresh : -thresh) : 0.0f;
        float d = v - q;
        baseline_mse += d * d;
    }
    baseline_mse /= (float)W.size();

    ts_lrq_params params = {};
    params.rank = rank;
    params.max_iters = 100;
    params.lr = 1e-2f;
    params.tol = 1e-8f;
    params.seed = 42;

    ts_lrq_result result;
    int rc = ts_train_lrq(W.data(), out_dim, in_dim, &params, &result);
    if (rc != 0) {
        printf("FAIL lrq: ts_train_lrq returned %d\n", rc);
        return 1;
    }
    if ((int64_t)result.U.size() != out_dim * rank) {
        printf("FAIL lrq: U size %zu != %lld\n", result.U.size(), (long long)(out_dim * rank));
        return 1;
    }
    if ((int64_t)result.V.size() != rank * in_dim) {
        printf("FAIL lrq: V size %zu != %lld\n", result.V.size(), (long long)(rank * in_dim));
        return 1;
    }
    if (result.mse >= baseline_mse) {
        printf("FAIL lrq: mse %.6e >= baseline %.6e\n", result.mse, baseline_mse);
        return 1;
    }
    printf("PASS lrq: mse=%.6e baseline=%.6e\n", result.mse, baseline_mse);
    return 0;
}

static int test_dartquant() {
    // 4x64 matrix, block_size=64, verify R^T R = I
    const int64_t out_dim = 4, in_dim = 64;
    std::vector<float> W(out_dim * in_dim);
    uint32_t rng = 456;
    for (auto & v : W) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        v = ((float)(rng & 0xFFFF) / 65536.0f - 0.5f) * 2.0f;
    }

    ts_dartquant_params params = {};
    params.block_size = 64;
    params.max_iters = 20;
    params.lr = 0.05f;
    params.seed = 7;

    ts_dartquant_result result;
    int rc = ts_dartquant_qr_orth(W.data(), out_dim, in_dim, &params, &result);
    if (rc != 0) {
        printf("FAIL dartquant: ts_dartquant_qr_orth returned %d\n", rc);
        return 1;
    }

    int64_t K = 64;
    if ((int64_t)result.R.size() != K * K) {
        printf("FAIL dartquant: R size %zu != %lld\n", result.R.size(), (long long)(K * K));
        return 1;
    }

    // check orthogonality: R^T R should be close to I
    float max_err = 0.0f;
    for (int64_t i = 0; i < K; i++) {
        for (int64_t j = 0; j < K; j++) {
            float dot = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                dot += result.R[k * K + i] * result.R[k * K + j];
            }
            float expected = (i == j) ? 1.0f : 0.0f;
            float err = fabsf(dot - expected);
            if (err > max_err) max_err = err;
        }
    }
    if (max_err > 1e-3f) {
        printf("FAIL dartquant: R^T R max deviation from I = %.6e\n", max_err);
        return 1;
    }
    printf("PASS dartquant: orthogonality err=%.6e whip=%.4f\n", max_err, result.whip_loss);
    return 0;
}

static int test_champq() {
    // 4x8 matrix, verify perm is a valid permutation
    const int64_t out_dim = 4, in_dim = 8;
    std::vector<float> W(out_dim * in_dim);
    uint32_t rng = 789;
    for (auto & v : W) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        v = ((float)(rng & 0xFFFF) / 65536.0f - 0.5f) * 2.0f;
    }

    ts_champq_params params = {};
    params.max_iters = 30;
    params.sinkhorn_iters = 15;
    params.use_lbfgs = true;
    params.seed = 42;

    ts_champq_result result;
    int rc = ts_champq_compute(W.data(), out_dim, in_dim, &params, &result);
    if (rc != 0) {
        printf("FAIL champq: ts_champq_compute returned %d\n", rc);
        return 1;
    }
    if ((int64_t)result.perm.size() != in_dim) {
        printf("FAIL champq: perm size %zu != %lld\n", result.perm.size(), (long long)in_dim);
        return 1;
    }

    // check valid permutation: each index in [0, in_dim) appears exactly once
    std::vector<int> seen(in_dim, 0);
    for (int64_t i = 0; i < in_dim; i++) {
        int32_t v = result.perm[i];
        if (v < 0 || v >= in_dim) {
            printf("FAIL champq: perm[%lld] = %d out of range\n", (long long)i, v);
            return 1;
        }
        seen[v]++;
    }
    for (int64_t i = 0; i < in_dim; i++) {
        if (seen[i] != 1) {
            printf("FAIL champq: index %lld appears %d times\n", (long long)i, seen[i]);
            return 1;
        }
    }
    printf("PASS champq: valid permutation, smoothness=%.4f improvement=%.4f\n",
           result.smoothness, result.mse_improvement);
    return 0;
}

static int test_router() {
    // kurtosis > 10 -> DartQuant
    if (ts_route_expert(15.0f, 0.5f, "ffn_gate") != TS_EXPERT_DARTQUANT) {
        printf("FAIL router: kurtosis=15 should route to DARTQUANT\n");
        return 1;
    }
    // eff_rank < 0.3 -> FLRQ
    if (ts_route_expert(3.0f, 0.1f, "attn_q") != TS_EXPERT_FLRQ) {
        printf("FAIL router: eff_rank=0.1 should route to FLRQ\n");
        return 1;
    }
    // else -> AWQ
    if (ts_route_expert(3.0f, 0.7f, "attn_k") != TS_EXPERT_AWQ) {
        printf("FAIL router: normal regime should route to AWQ\n");
        return 1;
    }
    // kurtosis takes priority over eff_rank
    if (ts_route_expert(20.0f, 0.1f, "ffn_up") != TS_EXPERT_DARTQUANT) {
        printf("FAIL router: kurtosis should take priority\n");
        return 1;
    }
    printf("PASS router: all routing cases correct\n");
    return 0;
}

int main() {
    int failures = 0;
    failures += test_lrq();
    failures += test_dartquant();
    failures += test_champq();
    failures += test_router();

    if (failures == 0) {
        printf("\nAll tests passed.\n");
    } else {
        printf("\n%d test(s) FAILED.\n", failures);
    }
    return failures;
}
