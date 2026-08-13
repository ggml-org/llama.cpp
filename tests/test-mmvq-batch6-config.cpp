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

void test_validated_hint_policy() {
    auto input = generic_input();
    input.validated_hint = true;
    input.n_expert_used = 8;
    check(ggml_cuda_mmvq_mmid_batch6(input), "validated top-k 8 hint rejected");

    input.n_expert_used = 256;
    check(ggml_cuda_mmvq_mmid_batch6(input), "validated hint unexpectedly limited by top-k");
}

void test_native_hint_policy() {
    // Hardware/opt-in gating is handled by mmvq.cu; this pure policy is
    // intentionally generic for every RDNA2 model loader.
    auto input = generic_input();
    input.validated_hint = true;
    input.n_expert_used = 8;
    check(ggml_cuda_mmvq_mmid_batch6(input), "validated high-top-k hint rejected");

    input.n_expert_used = 256;
    check(ggml_cuda_mmvq_mmid_batch6(input), "validated hint unexpectedly limited by top-k");
}

void test_type_guard() {
    auto input = generic_input();
    input.type = ggml_cuda_mmvq_batch6_type::other;
    check(!ggml_cuda_mmvq_mmid_batch6(input), "unsupported generic type accepted");

    input.validated_hint = true;
    input.n_expert_used = 8;
    check(!ggml_cuda_mmvq_mmid_batch6(input), "unsupported hinted type accepted");
}

} // namespace

int main() {
    test_bounded_generic_policy();
    test_validated_hint_policy();
    test_native_hint_policy();
    test_type_guard();
    std::puts("gfx1030 six-row routed MMVQ policy tests: PASS");
    return 0;
}