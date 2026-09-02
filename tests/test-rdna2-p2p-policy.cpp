#include "rdna2-p2p-allreduce-policy.h"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>

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

void test_boolean_flag_parser() {
    check(ggml_env_parse_flag(nullptr) == ggml_env_flag_value::unset, "unset flag not recognized");
    for (const char * value : { "1", "on", "true", "yes" }) {
        check(ggml_env_parse_flag(value) == ggml_env_flag_value::enabled,
                "enabled flag spelling not recognized");
    }
    for (const char * value : { "0", "off", "false", "no" }) {
        check(ggml_env_parse_flag(value) == ggml_env_flag_value::disabled,
                "disabled flag spelling not recognized");
    }
    for (const char * value : { "", "TRUE", "garbage" }) {
        check(ggml_env_parse_flag(value) == ggml_env_flag_value::invalid,
                "invalid flag spelling accepted");
    }
}

void test_p2p_mode_parser() {
    const char * automatic_values[] = { nullptr, "auto", "auto-expanded", "1", "on", "true", "yes" };
    for (const char * value : automatic_values) {
        const auto result = ggml_cuda_rdna2_p2p_host_parse_mode(true, value);
        check(result.recognized && result.mode == GGML_CUDA_RDNA2_P2P_HOST_AUTO_EXPANDED,
                "automatic P2P mode spelling rejected");
    }
    for (const char * value : { "0", "off", "false", "no" }) {
        const auto result = ggml_cuda_rdna2_p2p_host_parse_mode(true, value);
        check(result.recognized && result.mode == GGML_CUDA_RDNA2_P2P_HOST_OFF,
                "disabled P2P mode spelling rejected");
    }

    check(ggml_cuda_rdna2_p2p_host_parse_mode(true, "auto-basic").mode ==
            GGML_CUDA_RDNA2_P2P_HOST_AUTO, "auto-basic mode rejected");
    check(ggml_cuda_rdna2_p2p_host_parse_mode(true, "host").mode ==
            GGML_CUDA_RDNA2_P2P_HOST_SIMPLE, "host mode rejected");
    check(ggml_cuda_rdna2_p2p_host_parse_mode(true, "host-fused").mode ==
            GGML_CUDA_RDNA2_P2P_HOST_FUSED, "host-fused mode rejected");
    check(ggml_cuda_rdna2_p2p_host_parse_mode(true, "host-mtp").mode ==
            GGML_CUDA_RDNA2_P2P_HOST_MTP, "host-mtp mode rejected");

    const auto invalid = ggml_cuda_rdna2_p2p_host_parse_mode(true, "invalid");
    check(!invalid.recognized && invalid.mode == GGML_CUDA_RDNA2_P2P_HOST_OFF,
            "unknown P2P mode did not fail closed");

    const auto global_off = ggml_cuda_rdna2_p2p_host_parse_mode(false, "host-mtp");
    check(global_off.recognized && global_off.mode == GGML_CUDA_RDNA2_P2P_HOST_OFF,
            "global RDNA2 kill switch did not dominate feature mode");
}

void test_p2p_route_policy() {
    using route = ggml_cuda_rdna2_p2p_host_route;
    using reason = ggml_cuda_rdna2_p2p_host_fallback_reason;

    auto result = ggml_cuda_rdna2_p2p_host_select_route(2560, 1, 1, 1, true, true, true, true, true);
    check(result.route == route::qwen4exp_width1 && result.fallback_reason == reason::none,
            "exact Qwen4Exp width-one route rejected");

    result = ggml_cuda_rdna2_p2p_host_select_route(2560, 1, 1, 1, true, false, true, true, true);
    check(result.route == route::fallback && result.fallback_reason == reason::self_test_failed,
            "Qwen4Exp width-one self-test failure misclassified");

    result = ggml_cuda_rdna2_p2p_host_select_route(5120, 1, 1, 1, true, true, true, true, true);
    check(result.route == route::ordinary_width1 && result.fallback_reason == reason::none,
            "exact width-one route rejected");

    result = ggml_cuda_rdna2_p2p_host_select_route(5120, 5, 1, 1, true, true, true, true, true);
    check(result.route == route::speculative_width5 && result.fallback_reason == reason::none,
            "exact width-five route rejected");

    result = ggml_cuda_rdna2_p2p_host_select_route(5120, 6, 1, 1, true, true, true, true, true);
    check(result.route == route::speculative_width6 && result.fallback_reason == reason::none,
            "exact width-six route rejected");

    result = ggml_cuda_rdna2_p2p_host_select_route(5120, 7, 1, 1, true, true, true, true, true);
    check(result.route == route::fallback && result.fallback_reason == reason::unsupported_width,
            "unsupported width did not report the width reason");

    result = ggml_cuda_rdna2_p2p_host_select_route(5120, 1, 1, 1, true, true, false, true, true);
    check(result.route == route::fallback && result.fallback_reason == reason::self_test_failed,
            "width-one self-test failure misclassified");

    result = ggml_cuda_rdna2_p2p_host_select_route(5120, 5, 1, 1, true, true, true, false, true);
    check(result.route == route::fallback && result.fallback_reason == reason::self_test_failed,
            "width-five self-test failure misclassified");

    result = ggml_cuda_rdna2_p2p_host_select_route(5120, 5, 1, 1, false, true, true, true, true);
    check(result.route == route::fallback && result.fallback_reason == reason::policy_disabled,
            "disabled speculative-width policy misclassified");

    result = ggml_cuda_rdna2_p2p_host_select_route(5120, 6, 1, 1, true, true, true, true, false);
    check(result.route == route::fallback && result.fallback_reason == reason::self_test_failed,
            "width-six self-test failure misclassified");

    result = ggml_cuda_rdna2_p2p_host_select_route(4096, 6, 1, 1, true, true, true, true, true);
    check(result.route == route::fallback && result.fallback_reason == reason::unrelated_shape,
            "unrelated tensor shape misclassified");

    check(ggml_cuda_p2p_two_rank_width_allowed(false, 1),
            "automatic two-rank width one was rejected");
    check(!ggml_cuda_p2p_two_rank_width_allowed(false, 2),
            "automatic two-rank speculative width was accepted");
    check(ggml_cuda_p2p_two_rank_width_allowed(true, 1) &&
          ggml_cuda_p2p_two_rank_width_allowed(true, 6),
            "explicit two-rank supported widths were rejected");
    check(!ggml_cuda_p2p_two_rank_width_allowed(true, 0) &&
          !ggml_cuda_p2p_two_rank_width_allowed(true, 7),
            "explicit two-rank unsupported width was accepted");

    check(ggml_cuda_rdna2_p2p_host_width_bit(1) == 2u, "width-one log bit is wrong");
    check(ggml_cuda_rdna2_p2p_host_width_bit(5) == 32u, "width-five log bit is wrong");
    check(ggml_cuda_rdna2_p2p_host_width_bit(6) == 64u, "width-six log bit is wrong");
    check(ggml_cuda_rdna2_p2p_host_width_bit(0) == 0u, "zero width produced a log bit");
    check(ggml_cuda_rdna2_p2p_host_width_bit(32) == 0u, "out-of-range width produced a log bit");
}

} // namespace

int main() {
    test_boolean_flag_parser();
    test_p2p_mode_parser();
    test_p2p_route_policy();
    std::puts("RDNA2 P2P policy tests: PASS");
    return 0;
}
