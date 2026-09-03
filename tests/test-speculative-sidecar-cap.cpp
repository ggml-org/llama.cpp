#include "speculative-sidecar-cap.h"
#include "ngram-map.h"

#include <cstdio>
#include <cstdlib>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "sidecar cap test failure: %s\n", message);
        std::abort();
    }
}

static void test_cap_and_explicit_override() {
    common_speculative_sidecar_cap_config cap { 3 };
    common_speculative_draft_params dp;
    dp.n_max = 8;
    require(common_speculative_sidecar_cap_request_enabled(cap, dp), "default request uses cap");
    require(common_speculative_sidecar_cap_limit(cap, dp) == 3, "cap follows configured MTP width");

    dp.n_max_user_override = true;
    require(!common_speculative_sidecar_cap_request_enabled(cap, dp), "explicit request bypasses cap");
}

static void test_ngram_map_fixed_width() {
    const llama_tokens prompt = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 1,
    };

    common_ngram_map normal(2, 6, true, 1);
    common_ngram_map_begin(normal, prompt);
    llama_tokens normal_draft;
    common_ngram_map_draft(normal, prompt, 2, normal_draft);
    require(normal_draft.size() == 6, "normal map uses configured width");
    common_ngram_map_accept(normal, 1);
    normal_draft.clear();
    common_ngram_map_draft(normal, prompt, 2, normal_draft);
    require(normal_draft.size() == 1, "normal map adapts to last accepted width");

    common_ngram_map capped(2, 6, true, 1);
    common_ngram_map_begin(capped, prompt);
    capped.draft_limit = 3;
    llama_tokens capped_draft;
    common_ngram_map_draft(capped, prompt, 2, capped_draft);
    require(capped_draft.size() == 3, "map honors fixed sidecar cap");
    common_ngram_map_accept(capped, 1);
    capped_draft.clear();
    common_ngram_map_draft(capped, prompt, 2, capped_draft);
    require(capped_draft.size() == 3, "fixed sidecar cap owns width after partial acceptance");

    common_ngram_map complex(2, 6, false, 1);
    common_ngram_map_begin(complex, prompt);
    complex.draft_limit = 3;
    llama_tokens complex_draft;
    common_ngram_map_draft(complex, prompt, 2, complex_draft);
    require(complex_draft.size() == 3, "complex map honors fixed sidecar cap");

    common_ngram_map short_value(2, 1, true, 1);
    common_ngram_map_begin(short_value, prompt);
    short_value.draft_limit = 6;
    llama_tokens short_draft;
    common_ngram_map_draft(short_value, prompt, 2, short_draft);
    require(short_draft.size() == 1, "cap preserves available value bound");
}

int main() {
    test_cap_and_explicit_override();
    test_ngram_map_fixed_width();
    std::puts("test-speculative-sidecar-cap: PASS");
    return 0;
}
