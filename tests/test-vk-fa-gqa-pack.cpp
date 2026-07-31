// Regression for Vulkan FA GQA packing gate (PR #26358 / issue #25618).
//
// The old host condition packed whenever N<=8 (query tokens), which is wrong for
// multi-token speculative verify. Op-level NMSE vs CPU does not catch that path
// (error stays ~1e-5 either way). This test asserts the packing predicate itself:
// pack only for single-token decode (neq1 == 1).
//
// Fails before the fix (old predicate returns true for neq1 in 2..8).
// Passes after (neq1 == 1 required).

#include "ggml-vulkan-fa-gqa.h"

#include <cstdio>

static int fails = 0;

static void expect(bool cond, const char * msg) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        ++fails;
    }
}

int main() {
    const uint32_t max_gqa = 16;
    // Qwen3-like: 32 Q heads, 8 KV heads -> qk_ratio 4
    const uint32_t qk = 4, nek2 = 8, neq2 = 32, nev2 = 8, nem2 = 1;

    expect(ggml_vk_fa_should_pack_gqa(1, 1, qk, max_gqa, nek2, neq2, nev2, nem2),
           "single-token GQA decode should pack");

    const uint32_t multi[] = { 2, 3, 4, 5, 7, 8 };
    for (uint32_t i = 0; i < sizeof(multi) / sizeof(multi[0]); ++i) {
        const uint32_t neq1 = multi[i];
        char buf[128];
        snprintf(buf, sizeof(buf), "multi-token neq1=%u must not pack", neq1);
        expect(!ggml_vk_fa_should_pack_gqa(neq1, neq1, qk, max_gqa, nek2, neq2, nev2, nem2), buf);
    }

    // Old broken predicate (N<=8 only) would pack these - document the delta.
    auto old_pack = [](uint32_t N, uint32_t qk_ratio, uint32_t max_gqa,
                       uint32_t nek2, uint32_t neq2, uint32_t nev2, uint32_t nem2) {
        return N <= 8 && qk_ratio > 1 && qk_ratio <= max_gqa &&
               qk_ratio * nek2 == neq2 && nek2 == nev2 && nem2 <= 1;
    };
    expect(old_pack(8, qk, max_gqa, nek2, neq2, nev2, nem2),
           "sanity: old predicate would pack neq1=8");
    expect(!ggml_vk_fa_should_pack_gqa(8, 8, qk, max_gqa, nek2, neq2, nev2, nem2),
           "new predicate must disagree with old for neq1=8");

    if (fails) {
        fprintf(stderr, "%d check(s) failed\n", fails);
        return 1;
    }
    printf("test-vk-fa-gqa-pack: OK\n");
    return 0;
}
