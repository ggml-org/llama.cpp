#include "mmvq-rdna2-config.h"

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

ggml_cuda_mmvq_rdna2_q8_w8_input muse_kv_input() {
    return {
        ggml_cuda_mmvq_rdna2_type::q8_0,
        false,
        true,
        6656,
        128,
        1,
    };
}

void test_exact_muse_kv_shape() {
    check(ggml_cuda_mmvq_use_rdna2_q8_w8(muse_kv_input()), "validated Muse Q8_0 K/V shape rejected");
}

void test_shape_guards() {
    auto input = muse_kv_input();
    input.nrows_x = 2048;
    check(!ggml_cuda_mmvq_use_rdna2_q8_w8(input), "regressing Muse Q/gate shape accepted");

    input = muse_kv_input();
    input.nrows_x = 64;
    check(!ggml_cuda_mmvq_use_rdna2_q8_w8(input), "unvalidated output width accepted");

    input = muse_kv_input();
    input.ncols_x = 4096;
    check(!ggml_cuda_mmvq_use_rdna2_q8_w8(input), "unvalidated input width accepted");

    input = muse_kv_input();
    input.ncols_dst = 2;
    check(!ggml_cuda_mmvq_use_rdna2_q8_w8(input), "multi-column MMVQ accepted");
}

void test_layout_and_routing_guards() {
    auto input = muse_kv_input();
    input.has_ids = true;
    check(!ggml_cuda_mmvq_use_rdna2_q8_w8(input), "routed MMVQ accepted");

    input = muse_kv_input();
    input.standard_q8_1_layout = false;
    check(!ggml_cuda_mmvq_use_rdna2_q8_w8(input), "packed Q8_1 layout accepted");
}

void test_type_guard() {
    auto input = muse_kv_input();
    input.type = ggml_cuda_mmvq_rdna2_type::other;
    check(!ggml_cuda_mmvq_use_rdna2_q8_w8(input), "non-Q8_0 type accepted");
}

} // namespace

int main() {
    test_exact_muse_kv_shape();
    test_shape_guards();
    test_layout_and_routing_guards();
    test_type_guard();
    std::puts("RDNA2 Q8_0 MMVQ warp policy tests: PASS");
    return 0;
}
