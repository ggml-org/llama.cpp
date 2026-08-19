#include "server-task.h"

#include <cstdio>
#include <initializer_list>

static server_tokens tokens(std::initializer_list<llama_token> values, bool has_mtmd = false) {
    return server_tokens(llama_tokens(values), has_mtmd);
}

static server_prompt prompt(std::initializer_list<llama_token> values) {
    return server_prompt {
        tokens(values),
        {},
    };
}

static void cache_prompt(server_prompt_cache & cache, std::initializer_list<llama_token> values) {
    cache.states.push_back({
        prompt(values),
        {},
    });
}

static bool check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        return false;
    }
    return true;
}

int main() {
    bool ok = true;

    {
        auto resident = tokens({ 1, 2, 3 });
        auto incoming = tokens({ 1, 2, 3, 4, 5 });
        ok &= check(server_prompt_cache_can_reuse_in_place(resident, incoming, true),
                    "an exact text prefix should reuse the resident state");
    }
    {
        auto resident = tokens({ 1, 2, 3 });
        auto incoming = tokens({ 1, 2, 3 });
        ok &= check(server_prompt_cache_can_reuse_in_place(resident, incoming, true),
                    "an identical text prompt should reuse the resident state");
    }
    {
        auto resident = tokens({});
        auto incoming = tokens({ 1 });
        ok &= check(!server_prompt_cache_can_reuse_in_place(resident, incoming, true),
                    "an empty resident prompt must use the established path");
    }
    {
        auto resident = tokens({ 1, 2, 3 });
        auto incoming = tokens({ 1, 2 });
        ok &= check(!server_prompt_cache_can_reuse_in_place(resident, incoming, true),
                    "a rewind must use the established path");
    }
    {
        auto resident = tokens({ 1, 2, 3 });
        auto incoming = tokens({ 1, 9, 3, 4 });
        ok &= check(!server_prompt_cache_can_reuse_in_place(resident, incoming, true),
                    "a divergent prompt must use the established path");
    }
    {
        auto resident = tokens({ 1, 2, 3 });
        auto incoming = tokens({ 1, 2, 3, 4 });
        ok &= check(!server_prompt_cache_can_reuse_in_place(resident, incoming, false),
                    "cache_prompt=false must use the established path");
    }
    {
        auto resident = tokens({ 1, 2, 3 }, true);
        auto incoming = tokens({ 1, 2, 3, 4 });
        ok &= check(!server_prompt_cache_can_reuse_in_place(resident, incoming, true),
                    "a multimedia resident prompt must use the established path");
    }
    {
        auto resident = tokens({ 1, 2, 3 });
        auto incoming = tokens({ 1, 2, 3, 4 }, true);
        ok &= check(!server_prompt_cache_can_reuse_in_place(resident, incoming, true),
                    "a multimedia incoming prompt must use the established path");
    }

    // The update planner must keep every ineligible or useful fallback route.
    ok &= check(!server_prompt_cache_can_skip_recurrent_update(false, false, false, false),
                "an ineligible route must keep the established update policy");
    ok &= check(!server_prompt_cache_can_skip_recurrent_update(true, true, false, false),
                "a resident branch selected for preservation must still be saved");
    ok &= check(!server_prompt_cache_can_skip_recurrent_update(true, false, false, true),
                "a better cached state must still be loaded");
    ok &= check(server_prompt_cache_can_skip_recurrent_update(true, false, false, false),
                "a no-save, no-load recurrent update should be skipped");
    ok &= check(server_prompt_cache_can_skip_recurrent_update(true, true, true, false),
                "an exact resident prefix should override a redundant save request");

    // Avoid even sizing a state when an existing cache entry already contains
    // the resident prompt. Preserve negative paths for extensions and divergence.
    {
        server_prompt_cache cache(0, 0);
        cache_prompt(cache, { 1, 2, 3, 4 });
        auto resident = prompt({ 1, 2, 3 });
        ok &= check(cache.contains(resident),
                    "a longer cached state should contain its exact prompt prefix");
    }
    {
        server_prompt_cache cache(0, 0);
        cache_prompt(cache, { 1, 2, 3 });
        auto resident = prompt({ 1, 2, 3, 4 });
        ok &= check(!cache.contains(resident),
                    "a shorter cached state must not suppress saving an extension");
    }
    {
        server_prompt_cache cache(0, 0);
        cache_prompt(cache, { 1, 2, 3, 4 });
        auto resident = prompt({ 1, 2, 9 });
        ok &= check(!cache.contains(resident),
                    "a divergent cached state must not suppress saving the resident prompt");
    }
    {
        server_prompt_cache cache(0, 0);
        cache_prompt(cache, { 1, 2, 3, 4 });
        auto resident = prompt({});
        ok &= check(!cache.contains(resident),
                    "an empty resident prompt must not count as cached");
    }

    // Incomplete serializations must be removed transactionally and must not
    // poison contains() or future match probes.
    {
        server_prompt_cache cache(0, 0);
        cache_prompt(cache, { 1, 2, 3 });
        const auto * state = &cache.states.front();
        ok &= check(cache.discard(state),
                    "an allocated incomplete state should be discardable");
        ok &= check(cache.states.empty(),
                    "discarding an incomplete state should remove it from the cache");
        ok &= check(!cache.discard(nullptr),
                    "discarding a null state should fail safely");
    }
    {
        server_prompt_cache cache(0, 0);
        cache_prompt(cache, { 1, 2, 3 });
        server_prompt_cache_state unrelated {
            prompt({ 9, 8, 7 }),
            {},
        };
        ok &= check(!cache.discard(&unrelated),
                    "discarding a state owned by another cache must fail safely");
        ok &= check(cache.states.size() == 1,
                    "a failed discard must leave valid cache entries intact");
    }

    // The non-destructive probe must use exactly the same two-score policy as
    // load(): a candidate has to improve both retained fraction and similarity.
    {
        server_prompt_cache cache(0, 0);
        auto resident = prompt({ 1, 2, 90, 91, 92, 93, 94, 95, 96, 97 });
        auto incoming = tokens({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
        ok &= check(!cache.has_better_match(resident, incoming),
                    "an empty host cache must not request an update");
    }
    {
        server_prompt_cache cache(0, 0);
        cache_prompt(cache, { 1, 2, 3, 99 });
        auto resident = prompt({ 1, 2, 90, 91, 92, 93, 94, 95, 96, 97 });
        auto incoming = tokens({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
        ok &= check(cache.has_better_match(resident, incoming),
                    "a candidate improving both scores must request a cache load");
    }
    {
        server_prompt_cache cache(0, 0);
        cache_prompt(cache, { 1, 2, 3, 99 });
        auto resident = prompt({ 1, 2, 3, 90, 91, 92, 93, 94, 95, 96 });
        auto incoming = tokens({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
        ok &= check(!cache.has_better_match(resident, incoming),
                    "a candidate improving only retained fraction must not replace the resident state");
    }
    {
        server_prompt_cache cache(0, 0);
        cache_prompt(cache, { 1, 2, 3, 90, 91, 92, 93, 94, 95, 96 });
        auto resident = prompt({ 1, 2, 90, 91 });
        auto incoming = tokens({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
        ok &= check(!cache.has_better_match(resident, incoming),
                    "a candidate improving only similarity must not replace the resident state");
    }
    {
        server_prompt_cache cache(0, 0);
        cache_prompt(cache, { 1, 2, 3, 4, 90, 91, 92, 93, 94, 95,
                              96, 97, 98, 99, 100, 101, 102, 103, 104, 105 });
        auto resident = prompt({ 90, 91, 92, 93, 94, 95, 96, 97, 98, 99 });
        auto incoming = tokens({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
        ok &= check(!cache.has_better_match(resident, incoming),
                    "a candidate retaining less than 25 percent must remain rejected");
    }
    {
        server_prompt_cache cache(0, 0);
        cache_prompt(cache, { 1, 2, 3 });
        auto resident = prompt({ 1, 9, 8 });
        auto incoming = tokens({});
        ok &= check(!cache.has_better_match(resident, incoming),
                    "an empty incoming prompt must safely fall back");
    }
    {
        server_prompt_cache cache(0, 0);
        cache_prompt(cache, {});
        auto resident = prompt({ 1, 9, 8 });
        auto incoming = tokens({ 1, 2, 3 });
        ok &= check(!cache.has_better_match(resident, incoming),
                    "an empty cached prompt must be ignored");
    }

    if (!ok) {
        return 1;
    }

    std::puts("prompt cache policy: PASS");
    return 0;
}
