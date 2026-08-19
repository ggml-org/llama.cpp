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

struct rows2_shape {
    int64_t k;
    int64_t n;
};

ggml_cuda_mmvq_rdna2_w8_rows2_input rows2_input(
        ggml_cuda_mmvq_rdna2_type type, int64_t k, int64_t n) {
    return {
        type,
        true,
        false,
        true,
        k,
        n,
        8,
    };
}

void test_w8_rows2_validated_shapes() {
    const rows2_shape q4_0_shapes[] = {
        { 1536, 5120 }, { 4352, 5120 }, { 5120, 12 },   { 5120, 256 },
        { 5120, 1536 }, { 5120, 2560 }, { 5120, 3072 }, { 5120, 4352 },
    };
    for (const auto & shape : q4_0_shapes) {
        check(ggml_cuda_mmvq_use_rdna2_w8_rows2(
                rows2_input(ggml_cuda_mmvq_rdna2_type::q4_0, shape.k, shape.n)),
                "validated Q4_0 width-eight shape rejected");
    }

    const rows2_shape q4_k_shapes[] = {
        { 4096, 5120 },  { 5120, 256 },   { 5120, 1024 }, { 5120, 1280 },
        { 5120, 4096 },  { 5120, 17408 }, { 17408, 5120 }, { 25600, 5120 },
    };
    for (const auto & shape : q4_k_shapes) {
        check(ggml_cuda_mmvq_use_rdna2_w8_rows2(
                rows2_input(ggml_cuda_mmvq_rdna2_type::q4_k, shape.k, shape.n)),
                "validated Q4_K width-eight shape rejected");
    }

    const rows2_shape q6_k_shapes[] = {
        { 5120, 1024 }, { 5120, 248320 }, { 17408, 5120 },
    };
    for (const auto & shape : q6_k_shapes) {
        check(ggml_cuda_mmvq_use_rdna2_w8_rows2(
                rows2_input(ggml_cuda_mmvq_rdna2_type::q6_k, shape.k, shape.n)),
                "validated Q6_K width-eight shape rejected");
    }
}

void test_w8_rows2_fallback_guards() {
    auto input = rows2_input(ggml_cuda_mmvq_rdna2_type::q4_0, 5120, 4352);
    input.enabled = false;
    check(!ggml_cuda_mmvq_use_rdna2_w8_rows2(input), "disabled rows2 policy accepted");

    input = rows2_input(ggml_cuda_mmvq_rdna2_type::q4_0, 5120, 4352);
    input.has_ids = true;
    check(!ggml_cuda_mmvq_use_rdna2_w8_rows2(input), "routed rows2 MMVQ accepted");

    input = rows2_input(ggml_cuda_mmvq_rdna2_type::q4_0, 5120, 4352);
    input.standard_q8_1_layout = false;
    check(!ggml_cuda_mmvq_use_rdna2_w8_rows2(input), "packed rows2 MMVQ accepted");

    input = rows2_input(ggml_cuda_mmvq_rdna2_type::q4_0, 5120, 4352);
    input.ncols_dst = 7;
    check(!ggml_cuda_mmvq_use_rdna2_w8_rows2(input), "non-width-eight MMVQ accepted");

    input = rows2_input(ggml_cuda_mmvq_rdna2_type::q4_0, 5119, 4352);
    check(!ggml_cuda_mmvq_use_rdna2_w8_rows2(input), "unaligned K accepted");

    input = rows2_input(ggml_cuda_mmvq_rdna2_type::q4_0, 5088, 4352);
    check(!ggml_cuda_mmvq_use_rdna2_w8_rows2(input), "unknown aligned K accepted");

    input = rows2_input(ggml_cuda_mmvq_rdna2_type::q4_0, 1536, 4352);
    check(!ggml_cuda_mmvq_use_rdna2_w8_rows2(input), "unvalidated Q4_0 K/N pairing accepted");

    input = rows2_input(ggml_cuda_mmvq_rdna2_type::q4_0, 5120, 4096);
    check(!ggml_cuda_mmvq_use_rdna2_w8_rows2(input), "unknown Q4_0 output shape accepted");

    input = rows2_input(ggml_cuda_mmvq_rdna2_type::q4_k, 4096, 4096);
    check(!ggml_cuda_mmvq_use_rdna2_w8_rows2(input), "unvalidated Q4_K K/N pairing accepted");

    input = rows2_input(ggml_cuda_mmvq_rdna2_type::q6_k, 17408, 1024);
    check(!ggml_cuda_mmvq_use_rdna2_w8_rows2(input), "unvalidated Q6_K K/N pairing accepted");

    input = rows2_input(ggml_cuda_mmvq_rdna2_type::q8_0, 5120, 5120);
    check(!ggml_cuda_mmvq_use_rdna2_w8_rows2(input), "unvalidated quantization type accepted");
}

} // namespace

int main() {
    test_exact_muse_kv_shape();
    test_shape_guards();
    test_layout_and_routing_guards();
    test_type_guard();
    test_w8_rows2_validated_shapes();
    test_w8_rows2_fallback_guards();
    std::puts("RDNA2 MMVQ policy tests: PASS");
    return 0;
}
