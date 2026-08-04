#include "server-task.h"

#include <cassert>
#include <cstdint>
#include <initializer_list>

static server_prompt make_prompt(int64_t n_tokens, std::initializer_list<int64_t> checkpoints) {
    server_prompt prompt {
        server_tokens(llama_tokens(n_tokens, 1), false),
        {},
    };

    for (const int64_t n_tokens_checkpoint : checkpoints) {
        auto & checkpoint = prompt.checkpoints.emplace_back();
        checkpoint.n_tokens = n_tokens_checkpoint;
    }

    return prompt;
}

int main() {
    auto prompt = make_prompt(20000, {4096, 12000, 18000});

    // A recurrent checkpoint is valid only when its saved token boundary is
    // within the shared prefix and at least one token remains for logits.
    auto checkpoint = prompt.find_reusable_checkpoint(15000, 21000);
    assert(checkpoint != prompt.checkpoints.cend());
    assert(checkpoint->n_tokens == 12000);

    checkpoint = prompt.find_reusable_checkpoint(3, 21000);
    assert(checkpoint == prompt.checkpoints.cend());

    checkpoint = prompt.find_reusable_checkpoint(20000, 20000);
    assert(checkpoint != prompt.checkpoints.cend());
    assert(checkpoint->n_tokens == 18000);

    return 0;
}
