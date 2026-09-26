#include "../src/llama-kv-cache.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

static int failures = 0;

static void check(int got, int want, const char * msg) {
    if (got != want) {
        std::printf("FAIL: %s (want %d, got %d)\n", msg, want, got);
        ++failures;
    }
}

static void check_bytes(const uint8_t * got, const uint8_t * want, size_t size, const char * msg) {
    if (std::memcmp(got, want, size) != 0) {
        std::printf("FAIL: %s\n", msg);
        ++failures;
    }
}

static void test_turbo_type_predicate() {
    check(ggml_type_is_turbo(GGML_TYPE_TURBO2_0), 1, "turbo2 is a turbo KV type");
    check(ggml_type_is_turbo(GGML_TYPE_TURBO3_0), 1, "turbo3 is a turbo KV type");
    check(ggml_type_is_turbo(GGML_TYPE_TURBO4_0), 1, "turbo4 is a turbo KV type");
    check(ggml_type_is_turbo(GGML_TYPE_F16),      0, "f16 is not a turbo KV type");
    check(ggml_type_is_turbo(GGML_TYPE_F32),      0, "f32 is not a turbo KV type");
    check(ggml_type_is_turbo(GGML_TYPE_BF16),     0, "bf16 is not a turbo KV type");
    check(ggml_type_is_turbo(GGML_TYPE_Q8_0),     0, "q8_0 is not a turbo KV type");
    check(ggml_type_is_turbo(GGML_TYPE_TQ3_1S),   0, "turbo weight types are not turbo KV types");
}

static void test_adaptive_mode_matrix() {
    static const bool supported[] = { false, true, true, false, false, true, true, true, false, false };
    static const bool changes_k[] = { false, true, true, false, false, false, false, false, false, false };
    static const bool changes_v[] = { false, true, true, false, false, true, true, true, false, false };

    for (int mode = 0; mode < 10; ++mode) {
        check(llama_kv_cache_adaptive_mode_is_supported(mode), supported[mode], "adaptive mode support matrix");
        check(llama_kv_cache_adaptive_mode_changes_k(mode), changes_k[mode], "adaptive mode K-change matrix");
        check(llama_kv_cache_adaptive_mode_changes_v(mode), changes_v[mode], "adaptive mode V-change matrix");
    }
    check(llama_kv_cache_adaptive_mode_is_supported(-1), 0, "negative adaptive mode is unsupported");
    check(llama_kv_cache_adaptive_mode_changes_k(-1), 0, "negative adaptive mode does not change K");
    check(llama_kv_cache_adaptive_mode_changes_v(-1), 0, "negative adaptive mode does not change V");
}

static void test_q8_kv_repack_round_trip() {
    constexpr size_t blocks_per_group = 4;
    constexpr size_t quants_per_block = 32;
    constexpr size_t scale_bytes = sizeof(ggml_fp16_t);
    constexpr size_t block_bytes = scale_bytes + quants_per_block;
    constexpr size_t group_bytes = blocks_per_group * block_bytes;

    std::vector<uint8_t> data(2 * group_bytes);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>((37 * i + 11) & 0xff);
    }
    const std::vector<uint8_t> canonical = data;

    llama_kv_cache_q8_repack_groups(data.data(), data.size(), true);
    for (size_t group = 0; group < 2; ++group) {
        const size_t group_offset = group * group_bytes;
        for (size_t block = 0; block < blocks_per_group; ++block) {
            const size_t canonical_offset = group_offset + block * block_bytes;
            const size_t quants_offset = group_offset + block * quants_per_block;
            const size_t scale_offset = group_offset + blocks_per_group * quants_per_block + block * scale_bytes;
            check_bytes(
                data.data() + quants_offset,
                canonical.data() + canonical_offset + scale_bytes,
                quants_per_block,
                "quants-first q8_0 quant placement");
            check_bytes(
                data.data() + scale_offset,
                canonical.data() + canonical_offset,
                scale_bytes,
                "quants-first q8_0 scale placement");
        }
    }

    llama_kv_cache_q8_repack_groups(data.data(), data.size(), false);
    check_bytes(data.data(), canonical.data(), data.size(), "quants-first q8_0 canonical round trip");
}

static void test_auto_asymmetric_turbo_k() {
    // Two independent triggers (GQA ratio, Qwen-family), an opt-out that beats
    // both, and the symmetric-type precondition that also beats both. See
    // llama_kv_cache_auto_asymmetric_turbo_k() in llama-kv-cache.cpp.
    check(
        llama_kv_cache_auto_asymmetric_turbo_k(false, 2, true, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0),
        true, "low-GQA Qwen fires on the family trigger alone");
    check(
        llama_kv_cache_auto_asymmetric_turbo_k(false, 2, false, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0),
        false, "low-GQA non-Qwen does not fire");
    check(
        llama_kv_cache_auto_asymmetric_turbo_k(false, 6, false, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0),
        true, "high-GQA non-Qwen fires on the ratio trigger alone");
    check(
        llama_kv_cache_auto_asymmetric_turbo_k(false, 6, true, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0),
        true, "high-GQA Qwen fires (both triggers hold)");
    check(
        llama_kv_cache_auto_asymmetric_turbo_k(false, 1, false, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0),
        false, "low-GQA non-Qwen baseline (neither trigger) does not fire");
    check(
        llama_kv_cache_auto_asymmetric_turbo_k(true, 2, true, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0),
        false, "opt-out beats the Qwen-family trigger");
    check(
        llama_kv_cache_auto_asymmetric_turbo_k(true, 6, false, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0),
        false, "opt-out beats the GQA-ratio trigger");
    check(
        llama_kv_cache_auto_asymmetric_turbo_k(false, 6, true, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0),
        false, "asymmetric K/V type already beats both triggers");
}

static void test_cpu_rejects_quants_first_q8() {
    ggml_init_params params = { 16 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * src = ggml_new_tensor_1d(ctx, GGML_TYPE_Q8_0, 32);
    ggml_tensor * op = ggml_dup(ctx, src);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_dev_t cpu_dev = ggml_backend_get_device(cpu);

    check(ggml_backend_dev_supports_op(cpu_dev, op), 1, "CPU accepts canonical q8_0");
    src->flags |= GGML_TENSOR_FLAG_KV_Q8_QUANTS_FIRST;
    check(ggml_backend_dev_supports_op(cpu_dev, op), 0, "CPU rejects quants-first q8_0");

    ggml_backend_free(cpu);
    ggml_free(ctx);
}

int main() {
    test_turbo_type_predicate();
    test_adaptive_mode_matrix();
    // auto-enable path (env unset)
    check(llama_kv_cache_adaptive_mode(nullptr, GGML_TYPE_TURBO2_0, 32), 7, "turbo2 V, 32 layers auto-enables Boundary V");
    check(llama_kv_cache_adaptive_mode(nullptr, GGML_TYPE_TURBO2_0, 7),  0, "turbo2 V, 7 layers stays uniform");
    check(llama_kv_cache_adaptive_mode(nullptr, GGML_TYPE_TURBO3_0, 32), 0, "turbo3 V does not auto-enable");
    check(llama_kv_cache_adaptive_mode(nullptr, GGML_TYPE_F16, 32),      0, "f16 V stays uniform");
    // env override path (env always wins, including opt-out)
    check(llama_kv_cache_adaptive_mode("0", GGML_TYPE_TURBO2_0, 32), 0, "explicit opt-out beats auto-enable");
    check(llama_kv_cache_adaptive_mode("7", GGML_TYPE_TURBO4_0, 32), 7, "env 7 applies regardless of V type");
    check(llama_kv_cache_adaptive_mode("5", GGML_TYPE_F16, 32),      5, "env 5 applies regardless of V type");
    check(llama_kv_cache_adaptive_mode("2", GGML_TYPE_TURBO3_0, 32), 2, "env 2 applies");
    check(llama_kv_cache_adaptive_mode("", GGML_TYPE_TURBO2_0, 32),  0, "empty env suppresses auto-enable");
    // unsupported env values must NOT silently fall through; mode 0 = uniform
    check(llama_kv_cache_adaptive_mode("3", GGML_TYPE_F16, 32), 0, "unsupported mode 3 falls back to uniform");
    check(llama_kv_cache_adaptive_mode("4", GGML_TYPE_F16, 32), 0, "unsupported mode 4 falls back to uniform");
    check(llama_kv_cache_adaptive_mode("8", GGML_TYPE_F16, 32), 0, "out-of-range mode 8 falls back to uniform");
    check(llama_kv_cache_adaptive_mode("99", GGML_TYPE_F16, 32), 0, "two-digit mode 99 falls back to uniform");
    check(llama_kv_cache_adaptive_mode("1junk", GGML_TYPE_F16, 32), 0, "trailing junk rejected (not 1)");
    check(llama_kv_cache_adaptive_mode("-1", GGML_TYPE_F16, 32), 0, "negative sign rejected");
    // per-call independence: same process, interleaved differing inputs
    check(llama_kv_cache_adaptive_mode(nullptr, GGML_TYPE_TURBO2_0, 32), 7, "call A unaffected by prior calls");
    check(llama_kv_cache_adaptive_mode(nullptr, GGML_TYPE_F16, 32),      0, "call B after A does not inherit A's mode");
    check(llama_kv_cache_adaptive_mode(nullptr, GGML_TYPE_TURBO2_0, 32), 7, "call A repeated after B still selects its own mode");

    test_q8_kv_repack_round_trip();
    test_auto_asymmetric_turbo_k();
    test_cpu_rejects_quants_first_q8();

    if (failures == 0) {
        std::printf("test-kv-cache-adaptive-mode: all cases PASS\n");
    }
    return failures == 0 ? 0 : 1;
}
