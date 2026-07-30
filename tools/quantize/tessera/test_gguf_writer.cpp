//
// test_gguf_writer.cpp
//
// Smoke test for tessera-gguf-writer metadata path. Verifies that
// ts_gguf_write_metadata sets the expected KV pairs on a gguf_context.
//

#include "tessera-gguf-writer.h"

#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>

static int g_fail = 0;

static void check_u32(struct gguf_context * ctx, const char * key, uint32_t want) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        std::printf("FAIL %-40s key not found\n", key);
        g_fail++;
        return;
    }
    const uint32_t got = gguf_get_val_u32(ctx, id);
    if (got != want) {
        std::printf("FAIL %-40s got %u want %u\n", key, got, want);
        g_fail++;
    } else {
        std::printf("ok   %-40s %u\n", key, got);
    }
}

static void check_u64(struct gguf_context * ctx, const char * key, uint64_t want) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        std::printf("FAIL %-40s key not found\n", key);
        g_fail++;
        return;
    }
    const uint64_t got = gguf_get_val_u64(ctx, id);
    if (got != want) {
        std::printf("FAIL %-40s got %llu want %llu\n", key,
                    (unsigned long long)got, (unsigned long long)want);
        g_fail++;
    } else {
        std::printf("ok   %-40s %llu\n", key, (unsigned long long)got);
    }
}

static void check_f32(struct gguf_context * ctx, const char * key, float want) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        std::printf("FAIL %-40s key not found\n", key);
        g_fail++;
        return;
    }
    const float got = gguf_get_val_f32(ctx, id);
    if (std::fabs(got - want) > 1e-6f) {
        std::printf("FAIL %-40s got %f want %f\n", key, (double)got, (double)want);
        g_fail++;
    } else {
        std::printf("ok   %-40s %f\n", key, (double)got);
    }
}

static void check_str(struct gguf_context * ctx, const char * key, const char * want) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        std::printf("FAIL %-40s key not found\n", key);
        g_fail++;
        return;
    }
    const char * got = gguf_get_val_str(ctx, id);
    if (std::strcmp(got, want) != 0) {
        std::printf("FAIL %-40s got \"%s\" want \"%s\"\n", key, got, want);
        g_fail++;
    } else {
        std::printf("ok   %-40s \"%s\"\n", key, got);
    }
}

int main() {
    struct gguf_context * ctx = gguf_init_empty();
    if (!ctx) {
        std::printf("FAIL gguf_init_empty returned null\n");
        return 1;
    }

    ts_gguf_writer_params params;
    params.seed          = 42;
    params.alpha         = 0.5f;
    params.clip          = 0.9f;
    params.outlier_frac  = 0.005f;
    params.policy_summary = "attn_q:awq;attn_k:awq;ffn_gate:septq";
    params.policy_sha256  = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    params.build_info     = "tessera-cpp-g3-test";
    params.main_tip       = "abc1234";

    ts_gguf_write_metadata(ctx, &params);

    check_u32(ctx, "tessera.version", 1);
    check_u64(ctx, "tessera.quantize.seed", 42);
    check_f32(ctx, "tessera.quantize.alpha", 0.5f);
    check_f32(ctx, "tessera.quantize.clip", 0.9f);
    check_f32(ctx, "tessera.quantize.outlier_frac", 0.005f);
    check_str(ctx, "tessera.calibration.policy", "attn_q:awq;attn_k:awq;ffn_gate:septq");
    check_str(ctx, "tessera.calibration.sha256",
              "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789");
    check_str(ctx, "tessera.provenance.build_info", "tessera-cpp-g3-test");
    check_str(ctx, "tessera.provenance.main_tip", "abc1234");

    // test alpha = 0 writes "auto" string
    {
        struct gguf_context * ctx2 = gguf_init_empty();
        ts_gguf_writer_params p2 = params;
        p2.alpha = 0.0f;
        ts_gguf_write_metadata(ctx2, &p2);
        check_str(ctx2, "tessera.quantize.alpha", "auto");
        gguf_free(ctx2);
    }

    gguf_free(ctx);

    if (g_fail > 0) {
        std::printf("\n%d FAILURES\n", g_fail);
        return 1;
    }
    std::printf("\nall passed\n");
    return 0;
}
