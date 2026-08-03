// test_lk_train_data.cpp - offline tests for tessera-lk-train-data (pure, no model)
#include "tessera-lk-train-data.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; fprintf(stderr, "FAIL: %s\n", msg); } \
} while(0)

#define APPROX(a, b, eps) (std::fabs((a) - (b)) < (eps))
#define EPS 1e-5

// One llama.tessera.spec.v1 step: block_size = 2 (n_dft = 2), so n_ctx = 3.
// prime = 5, drafted = [7, 8]. Verifier top-k given for all 3 positions.
static const char * kLine =
    "{\"schema\":\"llama.tessera.spec.v1\",\"seq_id\":0,\"step_idx\":0,"
    "\"prime_token\":5,\"drafted\":2,\"accepted\":2,\"topk\":3,"
    "\"drafted_tokens\":[7,8],\"accepted_tokens\":[7,8,9],"
    "\"verifier_argmax\":[1,3,0],"
    "\"verifier_topk_tokens\":[[1,2],[3],[0,1,2]],"
    "\"verifier_topk_probs\":[[0.6,0.3],[0.5],[0.4,0.4,0.2]],"
    "\"drafter_topk_tokens\":[[7],[8],[9]],"
    "\"drafter_topk_probs\":[[0.5],[0.4],[0.3]],"
    "\"confidence\":[0.5,0.4]}";

static const int NV = 10; // small vocab for hand-checkable dense columns

static void test_usability() {
    CHECK(ts_lk_train_line_usable(kLine, 2) == 1, "usable at matching block_size");
    CHECK(ts_lk_train_line_usable(kLine, 3) == 0, "not usable at wrong block_size");
    CHECK(ts_lk_train_line_usable(kLine, 1) == 0, "not usable at smaller block_size");
    CHECK(ts_lk_train_line_usable("{\"schema\":\"some.other.schema\",\"drafted\":2}", 2) == 0,
          "wrong schema rejected");
    CHECK(ts_lk_train_line_usable("not json at all", 2) == 0, "garbage line rejected");
    CHECK(ts_lk_train_line_usable(kLine, 2) == 1, "usable re-check (no state)");
}

static void test_tokens_layout() {
    std::vector<int32_t> tokens(3, -1);
    std::vector<float>   labels((size_t) 3 * NV, -1.0f);
    const int rc = ts_lk_train_example_from_line(kLine, 2, NV, tokens.data(), labels.data());
    CHECK(rc == 1, "example produced");
    // [prime, draft[0], draft[1]] = [5, 7, 8]
    CHECK(tokens[0] == 5, "tokens[0] = prime");
    CHECK(tokens[1] == 7, "tokens[1] = draft[0]");
    CHECK(tokens[2] == 8, "tokens[2] = draft[1]");
}

static void test_dense_labels() {
    std::vector<int32_t> tokens(3, -1);
    std::vector<float>   labels((size_t) 3 * NV, -1.0f);
    CHECK(ts_lk_train_example_from_line(kLine, 2, NV, tokens.data(), labels.data()) == 1,
          "example produced for label checks");

    // Position 0: top-k [1:0.6, 2:0.3], mass 0.9, residual 0.1 over 8 slots = 0.0125.
    const float * p0 = labels.data() + 0 * NV;
    CHECK(APPROX(p0[1], 0.6, EPS),    "pos0 slot1 = 0.6");
    CHECK(APPROX(p0[2], 0.3, EPS),    "pos0 slot2 = 0.3");
    CHECK(APPROX(p0[0], 0.0125, EPS), "pos0 residual slot = 0.0125");
    double sum0 = 0.0; for (int i = 0; i < NV; ++i) sum0 += p0[i];
    CHECK(APPROX(sum0, 1.0, EPS), "pos0 sums to 1");

    // Position 1: top-k [3:0.5], mass 0.5, residual 0.5 over 9 slots.
    const float * p1 = labels.data() + 1 * NV;
    CHECK(APPROX(p1[3], 0.5, EPS),            "pos1 slot3 = 0.5");
    CHECK(APPROX(p1[0], 0.5 / 9.0, EPS),      "pos1 residual slot = 0.5/9");

    // Position 2: top-k [0:0.4, 1:0.4, 2:0.2], mass 1.0, zero residual.
    const float * p2 = labels.data() + 2 * NV;
    CHECK(APPROX(p2[0], 0.4, EPS), "pos2 slot0 = 0.4");
    CHECK(APPROX(p2[1], 0.4, EPS), "pos2 slot1 = 0.4");
    CHECK(APPROX(p2[2], 0.2, EPS), "pos2 slot2 = 0.2");
    CHECK(APPROX(p2[3], 0.0, EPS), "pos2 non-topk slot = 0 (zero residual)");
    double sum2 = 0.0; for (int i = 0; i < NV; ++i) sum2 += p2[i];
    CHECK(APPROX(sum2, 1.0, EPS), "pos2 sums to 1");
}

static void test_skip_leaves_buffers_untouched() {
    std::vector<int32_t> tokens(3, 42);
    std::vector<float>   labels((size_t) 3 * NV, 7.0f);
    const int rc = ts_lk_train_example_from_line(
        "{\"schema\":\"other\",\"drafted\":2}", 2, NV, tokens.data(), labels.data());
    CHECK(rc == 0, "non-usable line returns 0");
    bool untouched = (tokens[0] == 42 && tokens[1] == 42 && tokens[2] == 42 &&
                      labels[0] == 7.0f && labels.back() == 7.0f);
    CHECK(untouched, "buffers untouched on skip");
}

static void test_densify_error_is_fatal() {
    // verifier_topk token 99 is out of range for n_vocab = 10 -> densify fails.
    const char * bad =
        "{\"schema\":\"llama.tessera.spec.v1\",\"prime_token\":5,\"drafted\":1,"
        "\"drafted_tokens\":[7],"
        "\"verifier_topk_tokens\":[[99],[0]],"
        "\"verifier_topk_probs\":[[0.5],[0.5]]}";
    std::vector<int32_t> tokens(2, -1);
    std::vector<float>   labels((size_t) 2 * NV, 0.0f);
    CHECK(ts_lk_train_example_from_line(bad, 1, NV, tokens.data(), labels.data()) == -1,
          "out-of-range token -> -1");
}

static void test_detect_block_size() {
    const char * path = "/tmp/tessera_lk_train_detect.jsonl";
    {
        std::ofstream f(path);
        // three records with drafted=2, one with drafted=4 -> modal = 2
        for (int i = 0; i < 3; ++i) f << kLine << "\n";
        f << "{\"schema\":\"llama.tessera.spec.v1\",\"drafted\":4}\n";
        f << "garbage\n";
    }
    CHECK(ts_lk_train_detect_block_size(path) == 2, "modal block_size = 2");
    CHECK(ts_lk_train_detect_block_size("/tmp/does_not_exist_lk.jsonl") == -1,
          "missing file -> -1");
}

int main() {
    test_usability();
    test_tokens_layout();
    test_dense_labels();
    test_skip_leaves_buffers_untouched();
    test_densify_error_is_fatal();
    test_detect_block_size();

    fprintf(stderr, "test_lk_train_data: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
