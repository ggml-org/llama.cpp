#include "tessera-peqat.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static uint32_t lcg_state = 12345;

static float lcg_uniform() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return (float)(lcg_state >> 8) * (1.0f / 16777216.0f);
}

int main() {
    const int64_t out_dim  = 8;
    const int64_t in_dim   = 16;
    const int64_t n_tokens = 32;
    const int64_t rank     = 2;

    std::vector<float> W((size_t)(out_dim * in_dim));
    std::vector<float> X((size_t)(n_tokens * in_dim));
    for (auto & v : W) v = 2.0f * lcg_uniform() - 1.0f;
    for (auto & v : X) v = 2.0f * lcg_uniform() - 1.0f;

    // ref_output = X @ W^T
    std::vector<float> ref((size_t)(n_tokens * out_dim));
    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t o = 0; o < out_dim; o++) {
            float acc = 0.0f;
            for (int64_t j = 0; j < in_dim; j++) {
                acc += X[(size_t)(t * in_dim + j)] * W[(size_t)(o * in_dim + j)];
            }
            ref[(size_t)(t * out_dim + o)] = acc;
        }
    }

    ts_peqat_params params;
    params.lora_rank    = rank;
    params.smooth_alpha = 0.5f;
    params.lr           = 1e-2f;
    params.weight_decay = 0.01f;
    params.beta1        = 0.9f;
    params.beta2        = 0.999f;
    params.eps          = 1e-8f;
    params.tol          = 1e-9f;
    params.seed         = 42;
    params.verbose      = false;

    // Initial loss: single forward at the (B = 0) initialization.
    params.max_epochs = 1;
    ts_peqat_result r0;
    int rc = ts_peqat_train(W.data(), X.data(), ref.data(),
                            out_dim, in_dim, n_tokens, &params, &r0);
    assert(rc == 0);
    float initial_loss = r0.final_loss;

    // Trained loss.
    params.max_epochs = 50;
    ts_peqat_result r1;
    rc = ts_peqat_train(W.data(), X.data(), ref.data(),
                        out_dim, in_dim, n_tokens, &params, &r1);
    assert(rc == 0);

    printf("initial_loss=%.6e final_loss=%.6e epochs=%lld\n",
           initial_loss, r1.final_loss, (long long)r1.epochs_run);

    // Training reduces error.
    assert(r1.final_loss < initial_loss);

    // Dimensions.
    assert(r1.lora_A.size() == (size_t)(in_dim * rank));
    assert(r1.lora_B.size() == (size_t)(rank * out_dim));
    assert(r1.smooth_s.size() == (size_t)in_dim);

    // Determinism: same seed reproduces the trained loss exactly.
    ts_peqat_result r2;
    rc = ts_peqat_train(W.data(), X.data(), ref.data(),
                        out_dim, in_dim, n_tokens, &params, &r2);
    assert(rc == 0);
    assert(r1.final_loss == r2.final_loss);

    printf("PASS\n");
    return 0;
}
