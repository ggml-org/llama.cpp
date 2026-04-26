#include "server-task.h"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <utility>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

static server_prompt make_prompt(std::initializer_list<llama_token> tokens) {
    server_prompt prompt;
    prompt.tokens = server_tokens(llama_tokens(tokens), false);
    return prompt;
}

static void add_checkpoint(server_prompt & prompt, int64_t n_tokens) {
    server_prompt_checkpoint checkpoint = {};
    checkpoint.pos_min  = 0;
    checkpoint.pos_max  = n_tokens;
    checkpoint.n_tokens = n_tokens;
    checkpoint.data     = { 1, 2, 3, 4 };
    prompt.checkpoints.push_back(std::move(checkpoint));
}

static void test_full_removal_keeps_exact_shorter_without_checkpoint() {
    server_prompt_cache cache(0, 0);

    server_prompt long_prompt  = make_prompt({ 1, 2, 3, 4, 5, 6 });
    server_prompt short_prompt = make_prompt({ 1, 2, 3, 4 });

    require(cache.alloc(long_prompt, 8, COMMON_CONTEXT_SEQ_RM_TYPE_FULL) != nullptr,
            "failed to allocate initial long prompt");
    require(cache.alloc(short_prompt, 8, COMMON_CONTEXT_SEQ_RM_TYPE_FULL) != nullptr,
            "short exact prompt should be cached when the longer state has no restorable prefix checkpoint");
    require(cache.states.size() == 2,
            "cache should retain both long and short prompts without a restorable prefix checkpoint");
}

static void test_full_removal_reuses_longer_checkpoint_for_shorter_prompt() {
    server_prompt_cache cache(0, 0);

    server_prompt long_prompt  = make_prompt({ 1, 2, 3, 4, 5, 6 });
    server_prompt short_prompt = make_prompt({ 1, 2, 3, 4 });
    add_checkpoint(long_prompt, short_prompt.n_tokens());

    require(cache.alloc(long_prompt, 8, COMMON_CONTEXT_SEQ_RM_TYPE_FULL) != nullptr,
            "failed to allocate checkpointed long prompt");
    require(cache.alloc(short_prompt, 8, COMMON_CONTEXT_SEQ_RM_TYPE_FULL) == nullptr,
            "short exact prompt should be skipped when a longer checkpoint can restore it");
    require(cache.states.size() == 1,
            "checkpointed long prompt should make the exact shorter prompt redundant");
}

static void test_full_removal_only_removes_obsolete_shorter_with_checkpoint() {
    {
        server_prompt_cache cache(0, 0);

        server_prompt short_prompt = make_prompt({ 1, 2, 3, 4 });
        server_prompt long_prompt  = make_prompt({ 1, 2, 3, 4, 5, 6 });

        require(cache.alloc(short_prompt, 8, COMMON_CONTEXT_SEQ_RM_TYPE_FULL) != nullptr,
                "failed to allocate initial short prompt");
        require(cache.alloc(long_prompt, 8, COMMON_CONTEXT_SEQ_RM_TYPE_FULL) != nullptr,
                "long prompt without checkpoint should still be cached");
        require(cache.states.size() == 2,
                "short prompt must not be removed when long prompt cannot restore that prefix");
    }

    {
        server_prompt_cache cache(0, 0);

        server_prompt short_prompt = make_prompt({ 1, 2, 3, 4 });
        server_prompt long_prompt  = make_prompt({ 1, 2, 3, 4, 5, 6 });
        add_checkpoint(long_prompt, short_prompt.n_tokens());

        require(cache.alloc(short_prompt, 8, COMMON_CONTEXT_SEQ_RM_TYPE_FULL) != nullptr,
                "failed to allocate initial short prompt");
        require(cache.alloc(long_prompt, 8, COMMON_CONTEXT_SEQ_RM_TYPE_FULL) != nullptr,
                "failed to allocate checkpointed long prompt");
        require(cache.states.size() == 1,
                "short prompt should be removed once long prompt has a restorable prefix checkpoint");
        require(cache.states.front().n_tokens() == long_prompt.n_tokens(),
                "remaining cache entry should be the checkpointed long prompt");
    }
}

int main() {
    test_full_removal_keeps_exact_shorter_without_checkpoint();
    test_full_removal_reuses_longer_checkpoint_for_shorter_prompt();
    test_full_removal_only_removes_obsolete_shorter_with_checkpoint();

    return 0;
}
