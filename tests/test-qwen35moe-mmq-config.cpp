#include "qwen35moe-mmq-config.h"
#include "mmq-auto-config.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>

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

void test_model_selector() {
    float split[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    qwen35moe_mmq_model_config config = {
        true, 3072, 1024, 256, 8, true, 4, split,
    };
    check(qwen35moe_use_auto_rdna2_q4_k_j16(config), "exact model signature rejected");

    auto negative = [&](qwen35moe_mmq_model_config value, const char * message) {
        check(!qwen35moe_use_auto_rdna2_q4_k_j16(value), message);
    };
    auto changed = config; changed.is_122b_a10b = false; negative(changed, "wrong model accepted");
    changed = config; changed.n_embd = 4096; negative(changed, "wrong embedding accepted");
    changed = config; changed.n_ff_exp = 2048; negative(changed, "wrong expert width accepted");
    changed = config; changed.n_expert = 128; negative(changed, "wrong expert count accepted");
    changed = config; changed.n_expert_used = 4; negative(changed, "wrong top-k accepted");
    changed = config; changed.row_split = false; negative(changed, "non-row split accepted");
    changed = config; changed.n_devices = 2; negative(changed, "non-four-way split accepted");
    changed = config; changed.tensor_split = nullptr; negative(changed, "implicit split accepted");

    split[3] = 2.0f; negative(config, "unequal split accepted");
    split[3] = 0.0f; negative(config, "zero split accepted");
    split[3] = std::numeric_limits<float>::infinity(); negative(config, "infinite split accepted");
    split[3] = std::numeric_limits<float>::quiet_NaN(); negative(config, "NaN split accepted");
}

void test_environment_parser() {
    using mode = ggml_cuda_mmq_J_setting::mode;
    check(ggml_cuda_mmq_parse_J_setting(nullptr).state == mode::absent, "unset setting not absent");
    check(ggml_cuda_mmq_parse_J_setting("0").state == mode::heuristic, "0 not heuristic");
    check(ggml_cuda_mmq_parse_J_setting("default").state == mode::heuristic, "default not heuristic");
    for (const char * value : {"8", "16", "64", "128"}) {
        const auto parsed = ggml_cuda_mmq_parse_J_setting(value);
        check(parsed.state == mode::forced && parsed.value == std::atoi(value), "valid forced J rejected");
    }
    for (const char * value : {"", "7", "17", "129", "invalid", "16x"}) {
        check(ggml_cuda_mmq_parse_J_setting(value).state == mode::invalid, "invalid setting accepted");
    }
}

ggml_cuda_mmq_auto_J_input positive_backend_input() {
    return {
        true, true, true, true, true,
        3072, 256, 2048, 256, 256, 1, 1, 256,
    };
}

void test_backend_selector() {
    auto input = positive_backend_input();
    check(ggml_cuda_mmq_auto_J(input) == 16, "exact backend signature rejected");

    auto negative = [](ggml_cuda_mmq_auto_J_input value, const char * message) {
        check(ggml_cuda_mmq_auto_J(value) == 0, message);
    };
    auto changed = input; changed.hint_j16 = false; negative(changed, "missing hint accepted");
    changed = input; changed.rdna2 = false; negative(changed, "non-RDNA2 accepted");
    changed = input; changed.q4_k = false; negative(changed, "non-Q4_K accepted");
    changed = input; changed.routed_ids = false; negative(changed, "missing routed ids accepted");
    changed = input; changed.routed_bounds = false; negative(changed, "missing expert bounds accepted");
    changed = input; changed.ncols_x = 1024; negative(changed, "wrong K accepted");
    changed = input; changed.nrows_x = 512; negative(changed, "wrong N accepted");
    changed = input; changed.nchannels_x = 128; negative(changed, "wrong source experts accepted");
    changed = input; changed.nchannels_y = 128; negative(changed, "wrong destination experts accepted");
    changed = input; changed.nsamples_x = 2; negative(changed, "wrong source samples accepted");
    changed = input; changed.nsamples_y = 2; negative(changed, "wrong destination samples accepted");
    changed = input; changed.ncols_max = 255; negative(changed, "unvalidated ubatch accepted");
    changed = input; changed.ncols_dst = 1792; negative(changed, "top-7 accepted");
    changed = input; changed.ncols_dst = 2049; negative(changed, "non-integral top-k accepted");
}

} // namespace

int main() {
    test_model_selector();
    test_environment_parser();
    test_backend_selector();
    std::puts("Qwen3.5 automatic RDNA2 MMQ config tests: PASS");
    return 0;
}