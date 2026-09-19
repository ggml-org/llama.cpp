// Test: is_n_embd_{k,v}_gqa_variable() correctly ignores n_head_kv=0 layers
//
// Hybrid recurrent/attention architectures (Jamba, Granite-hybrid, etc.)
// represent recurrent layers with n_head_kv=0. The variable-GQA check must
// skip these layers; otherwise any mix of zero and non-zero n_head_kv falsely
// reports variable KV dimensions.

#include "llama-hparams.h"

#include <cassert>
#include <cstdio>
#include <vector>

struct test_case {
    const char * name;
    std::vector<uint32_t> n_head_kv;
    bool expected;
};

static void run(const test_case & tc) {
    llama_hparams hp = {};

    const uint32_t n = tc.n_head_kv.size();
    hp.n_layer_all   = n;
    hp.n_layer_nextn = 0;

    hp.n_embd_head_k_full = 128;
    hp.n_embd_head_v_full = 128;
    hp.n_embd_head_k_swa  = 128;
    hp.n_embd_head_v_swa  = 128;

    std::fill(hp.n_head_arr.begin(),    hp.n_head_arr.end(),    0u);
    std::fill(hp.n_head_kv_arr.begin(), hp.n_head_kv_arr.end(), 0u);
    std::fill(hp.is_swa_impl.begin(),   hp.is_swa_impl.end(),   false);

    for (uint32_t i = 0; i < n; ++i) {
        hp.n_head_kv_arr[i] = tc.n_head_kv[i];
        hp.n_head_arr[i]    = tc.n_head_kv[i] > 0 ? tc.n_head_kv[i] * 4 : 0;
    }

    const bool got_k = hp.is_n_embd_k_gqa_variable();
    const bool got_v = hp.is_n_embd_v_gqa_variable();

    printf("  %-25s  expect=%-5s  K=%-5s  V=%-5s  %s\n",
        tc.name,
        tc.expected ? "true" : "false",
        got_k ? "true" : "false",
        got_v ? "true" : "false",
        (got_k == tc.expected && got_v == tc.expected) ? "OK" : "FAIL");

    assert(got_k == tc.expected);
    assert(got_v == tc.expected);
}

int main() {
    const test_case cases[] = {
        { "uniform_attn",       {8, 8, 8, 8},       false },
        { "variable_attn",      {8, 4, 8, 4},       true  },
        { "hybrid_uniform",     {0, 0, 8, 0, 8, 0}, false },
        { "hybrid_variable",    {0, 8, 0, 4, 0},    true  },
        { "pure_recurrent",     {0, 0, 0, 0},        false },
    };

    printf("test-hparams-variable-gqa:\n");
    for (const auto & tc : cases) {
        run(tc);
    }
    printf("  all passed\n");

    return 0;
}
