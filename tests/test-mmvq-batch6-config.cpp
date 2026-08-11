#include "mmvq-batch6-config.h"

#include <cstdio>
#include <cstdlib>

namespace {

[[noreturn]] void fail(const char * message) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
}

void check(bool condition, const char * message) {
    if (!condition) {
        fail(message);
    }
}

ggml_cuda_mmvq_batch6_input generic_input() {
    return {
        true,
        true,
        ggml_cuda_mmvq_batch6_type::q4_k,
        false,
        4,
    };
}

void test_bounded_generic_policy() {
    auto input = generic_input();
    check(ggml_cuda_mmvq_mmid_batch6(input), "native RDNA2 Q4_K top-k 4 rejected");

    input.type = ggml_cuda_mmvq_batch6_type::q6_k;
    check(ggml_cuda_mmvq_mmid_batch6(input), "native RDNA2 Q6_K top-k 4 rejected");

    input.n_expert_used = 1;
    check(ggml_cuda_mmvq_mmid_batch6(input), "top-k 1 rejected");

    input.n_expert_used = 5;
    check(!ggml_cuda_mmvq_mmid_batch6(input), "generic top-k 5 accepted");

    input.n_expert_used = 0;
    check(!ggml_cuda_mmvq_mmid_batch6(input), "missing routed experts accepted");
}

void test_qwen_hint_policy() {
    auto input = generic_input();
    input.model_hint = true;
    input.n_expert_used = 8;
    check(ggml_cuda_mmvq_mmid_batch6(input), "native Qwen top-k 8 hint rejected");

    input.n_expert_used = 256;
    check(ggml_cuda_mmvq_mmid_batch6(input), "model hint unexpectedly limited by top-k");
}

void test_stock_and_architecture_guards() {
    auto input = generic_input();
    input.gfx1030_native = false;
    check(!ggml_cuda_mmvq_mmid_batch6(input), "native-off generic path accepted");

    input.model_hint = true;
    input.n_expert_used = 8;
    check(!ggml_cuda_mmvq_mmid_batch6(input), "native-off model hint accepted");

    input.gfx1030_native = true;
    input.rdna2 = false;
    check(!ggml_cuda_mmvq_mmid_batch6(input), "non-RDNA2 path accepted");
}

void test_type_guard() {
    auto input = generic_input();
    input.type = ggml_cuda_mmvq_batch6_type::other;
    check(!ggml_cuda_mmvq_mmid_batch6(input), "unsupported generic type accepted");

    input.model_hint = true;
    input.n_expert_used = 8;
    check(!ggml_cuda_mmvq_mmid_batch6(input), "unsupported hinted type accepted");
}

} // namespace

int main() {
    test_bounded_generic_policy();
    test_qwen_hint_policy();
    test_stock_and_architecture_guards();
    test_type_guard();
    std::puts("gfx1030 six-row routed MMVQ policy tests: PASS");
    return 0;
}