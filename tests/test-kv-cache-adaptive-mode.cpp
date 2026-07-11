#include "../src/llama-kv-cache.h"

#include <cstdio>

static int failures = 0;

static void check(int got, int want, const char * msg) {
    if (got != want) {
        std::printf("FAIL: %s (want %d, got %d)\n", msg, want, got);
        ++failures;
    }
}

int main() {
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
    check(llama_kv_cache_adaptive_mode("", GGML_TYPE_TURBO2_0, 32),  0, "empty env suppresses auto-enable (atoi semantics)");
    // per-call independence: same process, interleaved differing inputs
    check(llama_kv_cache_adaptive_mode(nullptr, GGML_TYPE_TURBO2_0, 32), 7, "call A unaffected by prior calls");
    check(llama_kv_cache_adaptive_mode(nullptr, GGML_TYPE_F16, 32),      0, "call B after A does not inherit A's mode");
    check(llama_kv_cache_adaptive_mode(nullptr, GGML_TYPE_TURBO2_0, 32), 7, "call A repeated after B still selects its own mode");

    if (failures == 0) {
        std::printf("test-kv-cache-adaptive-mode: all cases PASS\n");
    }
    return failures == 0 ? 0 : 1;
}
