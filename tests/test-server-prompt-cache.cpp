#include "server-task.h"

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <vector>

static server_prompt make_prompt(
        int64_t n_tokens,
        std::initializer_list<int64_t> checkpoints,
        size_t checkpoint_size = 0,
        llama_token token = 1) {
    server_prompt prompt {
        server_tokens(llama_tokens(n_tokens, token), false),
        {},
    };

    for (const int64_t n_tokens_checkpoint : checkpoints) {
        auto & checkpoint = prompt.checkpoints.emplace_back();
        checkpoint.n_tokens = n_tokens_checkpoint;
        checkpoint.data_tgt.resize(checkpoint_size);
    }

    return prompt;
}

static bool cache_contains(
        const server_prompt_cache & cache,
        const server_prompt_cache_state * state) {
    for (const auto & cur : cache.states) {
        if (&cur == state) {
            return true;
        }
    }

    return false;
}

static bool cache_contains_token(
        const server_prompt_cache & cache,
        llama_token token) {
    for (const auto & cur : cache.states) {
        if (!cur.prompt.tokens.empty() && cur.prompt.tokens[0] == token) {
            return true;
        }
    }

    return false;
}

int main() {
    {
        auto prompt = make_prompt(20000, {4096, 12000, 18000});

        const auto checkpoint = prompt.find_reusable_checkpoint(15000, 21000);
        assert(checkpoint != prompt.checkpoints.end());
        assert(checkpoint->n_tokens == 12000);
        assert(prompt.reusable_prefix_tokens(15000, 21000, true) == 12000);
    }

    {
        auto prompt = make_prompt(20000, {4096, 12000, 18000});

        // An exact full state is directly reusable when the new prompt appends tokens.
        assert(prompt.reusable_prefix_tokens(20000, 21000, true) == 20000);

        // An identical prompt must still evaluate one token for logits.
        assert(prompt.reusable_prefix_tokens(20000, 20000, true) == 18000);
    }

    {
        auto prompt = make_prompt(20000, {});

        assert(prompt.reusable_prefix_tokens(15000, 21000, true) == 0);
        assert(prompt.reusable_prefix_tokens(15000, 21000, false) == 15000);
    }

    {
        auto prompt = make_prompt(20000, {12000, 20000});

        const auto checkpoint = prompt.find_reusable_checkpoint(20000, 20000);
        assert(checkpoint != prompt.checkpoints.end());
        assert(checkpoint->n_tokens == 12000);
    }

    {
        constexpr size_t mib = 1024*1024;

        server_prompt_cache cache(1, 10000);
        auto prompt = make_prompt(1000, {100, 300, 600, 900}, 128*1024);

        auto * state = cache.alloc(prompt, 768*1024, 0);

        assert(state != nullptr);
        assert(state->size() == mib);
        assert(state->prompt.checkpoints.size() == 2);

        std::vector<int64_t> selected;
        for (const auto & checkpoint : state->prompt.checkpoints) {
            selected.push_back(checkpoint.n_tokens);
        }

        // The newest checkpoint is preferred, then the selection is spread
        // across the remaining prompt instead of retaining an adjacent pair.
        assert((selected == std::vector<int64_t> {100, 900}));
    }

    {
        constexpr size_t mib = 1024*1024;

        server_prompt_cache cache(1, 10000);
        auto prompt = make_prompt(1000, {100, 900}, 128*1024);

        auto * state = cache.alloc(prompt, mib + 1, 0);

        assert(state == nullptr);
        assert(cache.states.empty());
    }

    {
        server_prompt_cache cache(-1, 10000);
        auto prompt = make_prompt(1000, {100, 300, 600, 900}, 128*1024);

        auto * state = cache.alloc(prompt, 768*1024, 0);

        assert(state != nullptr);
        assert(state->prompt.checkpoints.size() == prompt.checkpoints.size());
    }

    {
        server_prompt_cache cache(4, 10000);
        auto live   = make_prompt(1000, {100});
        auto cached = make_prompt(900, {});
        server_tokens target(llama_tokens(950, 1), false);

        auto * cached_state = cache.alloc(cached, 512*1024, 0);
        auto * selected = cache.find_better(live, target, true, 1);

        assert(selected == cached_state);
    }

    {
        constexpr size_t mib = 1024*1024;

        server_prompt_cache cache(1, 10000);
        auto cached = make_prompt(900, {}, 0, 1);
        auto live   = make_prompt(1000, {}, 0, 2);
        server_tokens target(llama_tokens(950, 1), false);

        auto * cached_state = cache.alloc(cached, 512*1024, 0);
        auto * selected = cache.find_better(live, target, true, 1);
        assert(selected == cached_state);

        auto * saved_state = cache.alloc(live, 768*1024, 0);
        assert(cache.finalize(saved_state, &target, true, selected));

        // The selected entry is pinned during the swap. The cache may exceed
        // its steady-state budget until that entry is restored and removed.
        assert(saved_state != nullptr);
        assert(cache_contains(cache, cached_state));
        assert(cache_contains(cache, saved_state));
        assert(cache.size() == 1280*1024);
        assert(cache.size() > mib);
    }

    {
        server_prompt_cache cache(2, 10000);
        auto useful = make_prompt(600, {}, 0, 1);
        auto useless = make_prompt(600, {}, 0, 2);
        auto current = make_prompt(600, {}, 0, 3);
        server_tokens target(llama_tokens(700, 1), false);

        auto * useful_state = cache.alloc(useful, 512*1024, 0);
        auto * useless_state = cache.alloc(useless, 512*1024, 0);
        assert(useful_state != nullptr);
        assert(useless_state != nullptr);

        auto * current_state = cache.alloc(current, 1280*1024, 0);
        assert(cache.finalize(current_state, &target, false));

        assert(current_state != nullptr);
        assert(cache_contains(cache, useful_state));
        assert(!cache_contains_token(cache, 2));
        assert(cache_contains(cache, current_state));
    }

    {
        server_prompt_cache cache(4, 10000);
        auto live = make_prompt(1000, {322});
        auto cached = make_prompt(900, {322});

        llama_tokens target_tokens(950, 1);
        target_tokens[3] = 2;
        server_tokens target(target_tokens, false);

        cache.alloc(cached, 512*1024, 0);

        // No exact recurrent state exists before the divergence at token 3.
        assert(live.reusable_prefix_tokens(3, target.size(), true) == 0);
        assert(cache.find_better(live, target, true, 1) == nullptr);
    }

    return 0;
}
