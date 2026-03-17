// Test: verify tensor name pattern matching identifies expert vs non-expert tensors
// This mirrors the logic in llama-model.cpp load_tensors for MoE expert host-pinned routing

#include <cstdio>
#include <cstring>
#include <cstdlib>

// Simplified tensor type enum matching LLM_TENSOR values used in llama-model.cpp
enum test_tensor_type {
    TEST_TENSOR_ATTN_Q,
    TEST_TENSOR_ATTN_K,
    TEST_TENSOR_ATTN_V,
    TEST_TENSOR_ATTN_OUTPUT,
    TEST_TENSOR_FFN_NORM,
    TEST_TENSOR_OUTPUT_NORM,
    TEST_TENSOR_TOKEN_EMBD,
    TEST_TENSOR_FFN_GATE_INP,    // router gate — NOT expert
    TEST_TENSOR_FFN_GATE_EXPS,   // expert
    TEST_TENSOR_FFN_DOWN_EXPS,   // expert
    TEST_TENSOR_FFN_UP_EXPS,     // expert
    TEST_TENSOR_FFN_NORM_EXPS,   // expert
    TEST_TENSOR_FFN_GATE_CHEXPS, // expert (chunked)
    TEST_TENSOR_FFN_DOWN_CHEXPS, // expert (chunked)
    TEST_TENSOR_FFN_UP_CHEXPS,   // expert (chunked)
};

// This matches the is_expert_tensor logic in llama-model.cpp:load_tensors
static bool is_expert_tensor(test_tensor_type t) {
    return t == TEST_TENSOR_FFN_GATE_EXPS || t == TEST_TENSOR_FFN_DOWN_EXPS ||
           t == TEST_TENSOR_FFN_UP_EXPS   || t == TEST_TENSOR_FFN_NORM_EXPS ||
           t == TEST_TENSOR_FFN_GATE_CHEXPS ||
           t == TEST_TENSOR_FFN_DOWN_CHEXPS || t == TEST_TENSOR_FFN_UP_CHEXPS;
}

struct test_case {
    const char *     name;
    test_tensor_type type;
    bool             expected_expert;
};

int main() {
    int n_pass = 0;
    int n_fail = 0;

    test_case cases[] = {
        // Non-expert tensors — must stay on device
        {"attn_q",       TEST_TENSOR_ATTN_Q,         false},
        {"attn_k",       TEST_TENSOR_ATTN_K,         false},
        {"attn_v",       TEST_TENSOR_ATTN_V,         false},
        {"attn_output",  TEST_TENSOR_ATTN_OUTPUT,     false},
        {"ffn_norm",     TEST_TENSOR_FFN_NORM,        false},
        {"output_norm",  TEST_TENSOR_OUTPUT_NORM,      false},
        {"token_embd",   TEST_TENSOR_TOKEN_EMBD,       false},
        {"ffn_gate_inp", TEST_TENSOR_FFN_GATE_INP,     false},  // router gate, NOT expert

        // Expert tensors — should go to host-pinned
        {"ffn_gate_exps",   TEST_TENSOR_FFN_GATE_EXPS,   true},
        {"ffn_down_exps",   TEST_TENSOR_FFN_DOWN_EXPS,   true},
        {"ffn_up_exps",     TEST_TENSOR_FFN_UP_EXPS,     true},
        {"ffn_norm_exps",   TEST_TENSOR_FFN_NORM_EXPS,   true},
        {"ffn_gate_chexps", TEST_TENSOR_FFN_GATE_CHEXPS, true},
        {"ffn_down_chexps", TEST_TENSOR_FFN_DOWN_CHEXPS, true},
        {"ffn_up_chexps",   TEST_TENSOR_FFN_UP_CHEXPS,   true},
    };

    for (const auto & tc : cases) {
        bool result = is_expert_tensor(tc.type);
        if (result == tc.expected_expert) {
            n_pass++;
        } else {
            printf("FAIL: %s — expected expert=%d, got %d\n", tc.name, tc.expected_expert, result);
            n_fail++;
        }
    }

    printf("\n%d/%d tests passed\n", n_pass, n_pass + n_fail);
    if (n_fail > 0) {
        printf("FAILED\n");
        return 1;
    }
    printf("OK\n");
    return 0;
}
