#include "../src/llama-graph.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

static int failures = 0;

static void expect_true(bool value, const char * what) {
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

static void expect_close(float actual, float expected, const char * what) {
    if (std::fabs(actual - expected) > 1e-6f) {
        std::fprintf(stderr, "FAIL: %s (actual %g, expected %g)\n", what, (double) actual, (double) expected);
        ++failures;
    }
}

template <typename F> static void expect_throw(F && fn, const char * what) {
    try {
        fn();
    } catch (const std::runtime_error &) {
        return;
    }
    std::fprintf(stderr, "FAIL: expected exception: %s\n", what);
    ++failures;
}

int main() {
    const int64_t n_tokens = 8;
    const int64_t n_blocks = 2;
    const auto    feat     = llm_dspark_log_snr_feat(n_tokens, n_blocks, -9.0f, 9.0f);

    expect_true(feat.size() == (size_t) (LLM_DSPARK_LOG_SNR_N_FREQ * n_tokens), "feature size");

    for (int64_t i = 0; i < LLM_DSPARK_LOG_SNR_N_FREQ; ++i) {
        expect_close(feat[(size_t) (0 * LLM_DSPARK_LOG_SNR_N_FREQ + i)],
                     feat[(size_t) (4 * LLM_DSPARK_LOG_SNR_N_FREQ + i)], "anchor rows differ");
        expect_close(feat[(size_t) (1 * LLM_DSPARK_LOG_SNR_N_FREQ + i)],
                     feat[(size_t) (5 * LLM_DSPARK_LOG_SNR_N_FREQ + i)], "draft rows differ");
        expect_close(feat[(size_t) (1 * LLM_DSPARK_LOG_SNR_N_FREQ + i)],
                     feat[(size_t) (2 * LLM_DSPARK_LOG_SNR_N_FREQ + i)], "draft row 1 differs from 2");
        expect_close(feat[(size_t) (2 * LLM_DSPARK_LOG_SNR_N_FREQ + i)],
                     feat[(size_t) (3 * LLM_DSPARK_LOG_SNR_N_FREQ + i)], "draft row 2 differs from 3");
    }

    expect_close(feat[0], std::sin(1000.0f), "anchor sin");
    expect_close(feat[LLM_DSPARK_LOG_SNR_N_FREQ / 2], std::cos(1000.0f), "anchor cos");
    expect_close(feat[LLM_DSPARK_LOG_SNR_N_FREQ], 0.0f, "draft sin");
    expect_close(feat[LLM_DSPARK_LOG_SNR_N_FREQ + LLM_DSPARK_LOG_SNR_N_FREQ / 2], 1.0f, "draft cos");
    expect_true(std::isfinite(feat[0]), "first feature is finite");
    expect_true(std::fabs(feat[0] - feat[LLM_DSPARK_LOG_SNR_N_FREQ]) > 1e-6f, "anchor differs from draft");

    expect_throw([] { llm_dspark_log_snr_feat(0, 1, -9.0f, 9.0f); }, "empty tokens");
    expect_throw([] { llm_dspark_log_snr_feat(8, 0, -9.0f, 9.0f); }, "empty blocks");
    expect_throw([] { llm_dspark_log_snr_feat(7, 2, -9.0f, 9.0f); }, "non-divisible blocks");
    expect_throw([] { llm_dspark_log_snr_feat(2, 4, -9.0f, 9.0f); }, "more blocks than tokens");
    expect_throw([] { llm_dspark_log_snr_feat(8, 2, 9.0f, 9.0f); }, "equal bounds");
    expect_throw([] { llm_dspark_log_snr_feat(8, 2, 9.0f, -9.0f); }, "reversed bounds");
    expect_throw([] { llm_dspark_log_snr_feat(8, 2, NAN, 9.0f); }, "non-finite min");
    expect_throw([] { llm_dspark_log_snr_feat(8, 2, -9.0f, INFINITY); }, "non-finite max");

    llm_dspark_log_snr_validate(false, 4, NAN, INFINITY);
    expect_throw([] { llm_dspark_log_snr_validate(true, 1, -9.0f, 9.0f); }, "DSV4 conflict");
    expect_throw([] { llm_dspark_log_snr_validate(true, 0, INFINITY, 9.0f); }, "invalid metadata min");
    expect_throw([] { llm_dspark_log_snr_validate(true, 0, 9.0f, 9.0f); }, "invalid metadata order");

    llm_graph_params params  = {};
    params.ubatch.n_tokens   = n_tokens;
    params.ubatch.n_seqs_unq = n_blocks;

    ggml_tensor tensor = {};
    tensor.ne[1]       = n_tokens;

    llm_graph_input_dspark_logsnr input(std::vector<float>(feat.size(), 0.0f), n_blocks);
    expect_true(!input.can_reuse(params), "reuse without tensor");

    input.feat = &tensor;
    expect_true(input.can_reuse(params), "reuse with matching geometry");

    params.ubatch.n_seqs_unq = 4;
    expect_true(!input.can_reuse(params), "reuse with different block count");

    params.ubatch.n_seqs_unq = n_blocks;
    params.ubatch.n_tokens   = n_tokens - 1;
    expect_true(!input.can_reuse(params), "reuse with different token count");

    if (failures != 0) {
        std::fprintf(stderr, "%d test(s) failed\n", failures);
        return 1;
    }

    std::printf("OK\n");
    return 0;
}
