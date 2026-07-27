#include "server-task.h"

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <vector>

static server_prompt make_prompt(
        int64_t n_tokens,
        std::initializer_list<int64_t> checkpoints,
        size_t checkpoint_size = 0) {
    server_prompt prompt {
        server_tokens(llama_tokens(n_tokens, 1), false),
        {},
    };

    for (const int64_t n_tokens_checkpoint : checkpoints) {
        auto & checkpoint = prompt.checkpoints.emplace_back();
        checkpoint.n_tokens = n_tokens_checkpoint;
        checkpoint.data_tgt.resize(checkpoint_size);
    }

    return prompt;
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

    return 0;
}
