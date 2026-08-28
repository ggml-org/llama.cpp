#undef NDEBUG
#include "speculative.h"

#include <cassert>
#include <cstdio>
#include <vector>

// ngram-cache keeps request-local state (context n-gram cache and its size) per sequence.
// A new request on the same seq_id must not reuse the n-grams ingested from the previous
// request, otherwise the draft is computed from stale context (see issue #27852).
static void test_ngram_cache_begin_resets_request_state() {
    common_params_speculative params;
    params.types = { COMMON_SPECULATIVE_TYPE_NGRAM_CACHE };

    const uint32_t n_seq = 1;
    common_speculative_ptr spec(common_speculative_init(params, n_seq));
    assert(spec != nullptr);

    // request 1: ingest a prompt so that the n-gram (104, 105, 106) -> 107
    // ends up in the per-sequence context cache
    llama_tokens prompt1 = { 101, 102, 103, 104, 105, 106 };
    common_speculative_begin(spec.get(), 0, prompt1);

    llama_tokens result1;
    auto & dp = common_speculative_get_draft_params(spec.get(), 0);
    dp.drafting = true;
    dp.prompt   = &prompt1;
    dp.id_last  = 107;
    dp.result   = &result1;
    common_speculative_draft(spec.get());

    // request 2: new, shorter prompt on the same seq_id
    // its ending n-gram (104, 105, 106) matches the stale cache from request 1,
    // which would draft token 107 if the per-sequence state was not reset
    llama_tokens prompt2 = { 100, 104, 105 };
    common_speculative_begin(spec.get(), 0, prompt2);

    llama_tokens result2;
    dp.drafting = true;
    dp.prompt   = &prompt2;
    dp.id_last  = 106;
    dp.result   = &result2;
    common_speculative_draft(spec.get());

    // no draft must be produced for request 2 - the reset context cache
    // contains no n-gram that matches the new prompt
    assert(result2.empty());
}

int main() {
    test_ngram_cache_begin_resets_request_state();

    printf("test-ngram-cache: all tests passed\n");

    return 0;
}
