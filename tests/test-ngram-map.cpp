#include "common.h"
#include "ngram-map.h"

#include <cstdio>
#include <random>

static llama_tokens rand_tokens(std::mt19937 & rng, size_t n, int lo, int hi) {
    std::uniform_int_distribution<int> dist(lo, hi - 1);
    llama_tokens out(n);
    for (auto & t : out) {
        t = dist(rng);
    }
    return out;
}

// expect a draft of prompt[pos+n, pos+n+m) after re-typing prompt[pos, pos+n)
static bool check_copy_draft(common_ngram_map & map, const llama_tokens & prompt, size_t pos, const char * what) {
    const size_t n = map.size_key;
    const size_t m = map.size_value;

    llama_tokens inp = prompt;
    inp.insert(inp.end(), prompt.begin() + pos, prompt.begin() + pos + n - 1);
    const llama_token sampled = prompt[pos + n - 1];

    llama_tokens draft;
    common_ngram_map_draft(map, inp, sampled, draft);

    bool ok = draft.size() == m;
    for (size_t i = 0; ok && i < m; ++i) {
        ok = draft[i] == prompt[pos + n + i];
    }
    printf("%s: draft size %zu, %s\n", what, draft.size(), ok ? "ok" : "FAILED");
    return ok;
}

int main() {
    std::mt19937 rng(42);
    const uint16_t n = 4;
    const uint16_t m = 16;

    common_ngram_map map(n, m, /*key_only*/ true, /*min_hits*/ 1);

    const llama_tokens a = rand_tokens(rng, 3000, 1, 1000);
    common_ngram_map_begin(map, a);
    bool ok = check_copy_draft(map, a, 500, "first prompt");

    // different, shorter prompt in the same map
    const llama_tokens b = rand_tokens(rng, 800, 1000, 2000);
    common_ngram_map_begin(map, b);
    ok = check_copy_draft(map, b, 200, "different shorter prompt") && ok;

    const llama_tokens c = rand_tokens(rng, 5000, 2000, 3000);
    common_ngram_map_begin(map, c);
    ok = check_copy_draft(map, c, 1000, "different longer prompt") && ok;

    // continuation of the previous prompt
    llama_tokens d = c;
    const llama_tokens tail = rand_tokens(rng, 300, 3000, 4000);
    d.insert(d.end(), tail.begin(), tail.end());
    common_ngram_map_begin(map, d);
    ok = check_copy_draft(map, d, 1000, "continued prompt, old part") && ok;
    ok = check_copy_draft(map, d, 5100, "continued prompt, new part") && ok;

    return ok ? 0 : 1;
}
