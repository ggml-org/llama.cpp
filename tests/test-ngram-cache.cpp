#include "ngram-cache.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cstdio>
#include <vector>

// drafting with fewer tokens than LLAMA_NGRAM_MAX must not read before the start of inp
static void test_short_input_empty_caches() {
    for (int n_inp = 1; n_inp <= LLAMA_NGRAM_MAX; ++n_inp) {
        std::vector<llama_token> inp;
        for (int i = 0; i < n_inp; ++i) {
            inp.push_back(100 + i);
        }
        std::vector<llama_token> draft = { inp.back() };

        common_ngram_cache nc_context;
        common_ngram_cache nc_dynamic;
        common_ngram_cache nc_static;

        common_ngram_cache_draft(inp, draft, 8, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, nc_context, nc_dynamic, nc_static);

        assert(draft.size() == 1);
    }
}

// a short input still drafts from the ngram sizes that fit, and larger sizes are used once the draft grows
static void test_short_input_drafts() {
    std::vector<llama_token> seq = { 1, 2, 3, 1, 2, 3, 1, 2, 3 };

    common_ngram_cache nc_context;
    common_ngram_cache nc_dynamic;
    common_ngram_cache nc_static;
    common_ngram_cache_update(nc_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, seq, seq.size(), false);

    std::vector<llama_token> inp   = { 1, 2 };
    std::vector<llama_token> draft = { 2 };

    common_ngram_cache_draft(inp, draft, 4, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, nc_context, nc_dynamic, nc_static);

    const std::vector<llama_token> expected = { 2, 3, 1, 2, 3 };
    assert(draft == expected);
}

int main() {
    test_short_input_empty_caches();
    test_short_input_drafts();

    printf("OK\n");

    return 0;
}
