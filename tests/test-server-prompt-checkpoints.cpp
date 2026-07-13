#include "server-task.h"

#include <cassert>
#include <cstdio>

int main() {
    server_prompt prompt;

    // populate tokens, mirroring a slot that has processed a prompt
    for (llama_token t = 0; t < 32; t++) {
        prompt.tokens.push_back(t);
    }

    // populate checkpoints, mirroring the context-checkpoint mechanism
    // (server-context.cpp creates these via slot.prompt.checkpoints.emplace_back())
    for (int i = 0; i < 4; i++) {
        common_prompt_checkpoint ckpt;
        ckpt.n_tokens = 16;
        ckpt.pos_min  = 0;
        ckpt.pos_max  = 15;
        ckpt.data_tgt = std::vector<uint8_t>(1024, 0xAB); // stand-in for a real KV-cache state blob

        prompt.checkpoints.push_back(std::move(ckpt));
    }

    // sanity check on the fixture itself
    assert(prompt.tokens.size() == 32);
    assert(prompt.checkpoints.size() == 4);
    assert(prompt.size() > 0); // dominated by the checkpoint buffers

    const size_t size_before = prompt.size();

    prompt.clear();

    if (!prompt.tokens.empty()) {
        fprintf(stderr, "FAIL: prompt.tokens not cleared (size = %zu)\n", prompt.tokens.size());
        return 1;
    }

    if (!prompt.checkpoints.empty()) {
        fprintf(stderr, "FAIL: prompt.checkpoints not cleared (size = %zu) - checkpoint buffers leaked "
                         "(this reproduces https://github.com/ggml-org/llama.cpp/issues/25437)\n",
                prompt.checkpoints.size());
        return 1;
    }

    if (prompt.size() != 0) {
        fprintf(stderr, "FAIL: prompt.size() == %zu after clear (was %zu before) - memory still held\n",
                prompt.size(), size_before);
        return 1;
    }

    printf("[test-server-prompt-checkpoints] OK: %zu bytes across %d checkpoints freed by prompt.clear()\n",
           size_before, 4);

    return 0;
}
