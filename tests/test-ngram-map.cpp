#include "ngram-map.h"
#include "testing.h"

int main() {
    testing t;

    t.test("common_ngram_map_begin rebases idx_last_check after prompt shrink", [](testing & t) {
        common_ngram_map map(/* sz_key = */ 4, /* sz_value = */ 4, /* only_keys = */ false, /* min_hits = */ 1);

        llama_tokens prompt_long(20);
        for (size_t i = 0; i < prompt_long.size(); ++i) {
            prompt_long[i] = static_cast<llama_token>(i + 1);
        }

        common_ngram_map_begin(map, prompt_long);

        llama_tokens draft;
        common_ngram_map_draft(map, prompt_long, /* sampled = */ 21, draft);

        const size_t idx_after_draft = map.idx_last_check;
        t.assert_equal("draft updates idx_last_check to the prompt length", prompt_long.size(), idx_after_draft);

        llama_tokens prompt_short(prompt_long.begin(), prompt_long.begin() + 14);
        common_ngram_map_begin(map, prompt_short);

        t.assert_equal("begin rebases idx_last_check to the new prompt length",
                prompt_short.size() - 1, map.idx_last_check);
        t.assert_equal("begin tracks the new prompt length", prompt_short.size(), map.size_last_begin);
    });

    return t.failures ? 1 : 0;
}
