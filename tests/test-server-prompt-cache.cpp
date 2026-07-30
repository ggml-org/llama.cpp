#include "server-task.h"

#include <cassert>
#include <string>
#include <vector>

static server_prompt make_prompt(std::initializer_list<llama_token> tokens) {
    return server_prompt {
        server_tokens(llama_tokens(tokens), false),
        {},
    };
}

int main() {
    // A radix-style prompt cache retains a reusable short prefix alongside a
    // longer descendant. The former serves sibling requests; the latter
    // serves exact or extended requests. Eviction is controlled separately by
    // the existing byte/token limits.
    server_prompt_cache cache(/* limit_size_mib = */ 32, /* limit_tokens = */ 0);
    const server_prompt prefix = make_prompt({ 1, 2, 3 });
    const server_prompt branch = make_prompt({ 1, 2, 3, 4, 5 });

    assert(cache.alloc(prefix, 64, 0) != nullptr);
    assert(cache.alloc(branch, 96, 0) != nullptr);
    assert(cache.states.size() == 2);
    assert(cache.n_tokens() == 8);

    // The radix index returns the deepest state for each request without a
    // linear cache scan; sibling prompts still attach to the shared prefix.
    const server_prompt sibling = make_prompt({ 1, 2, 3, 9 });
    assert(cache.find_longest_prefix(branch.tokens) != nullptr);
    assert(cache.find_longest_prefix(branch.tokens)->prompt.tokens.size() == 5);
    assert(cache.find_longest_prefix(sibling.tokens) != nullptr);
    assert(cache.find_longest_prefix(sibling.tokens)->prompt.tokens.size() == 3);

    // Re-adding an exact state remains a no-op.
    assert(cache.alloc(prefix, 64, 0) == nullptr);
    assert(cache.states.size() == 2);

    // The native-KV radix tracks block ownership separately from serialized
    // prompt snapshots. A descendant attaches to the same source sequence,
    // releases independently, and only becomes an LRU eviction candidate
    // after the final sequence reference is gone.
    server_kv_block_radix blocks(/* block_tokens = */ 32);
    std::vector<std::string> keys;
    for (int i = 0; i < 64; ++i) {
        keys.emplace_back("t:" + std::to_string(i));
    }
    blocks.publish(keys, { 32, 64 }, /* seq = */ 1, /* now = */ 10);
    assert(blocks.n_blocks() == 2);
    const auto attached = blocks.attach(keys, /* seq = */ 2, /* now = */ 20);
    assert(attached.source == 1);
    assert(attached.positions == 64);
    assert(attached.block_ids.size() == 2);
    assert(blocks.n_owners(attached.block_ids[0]) == 2);
    blocks.release(1);
    assert(blocks.evict(0) == 0);
    blocks.release(2);
    assert(blocks.evict(0) == 2);
    assert(blocks.n_blocks() == 0);

    // Eviction is block-granular LRU, not a whole-prefix purge. Touching B
    // protects it while the older independent A block is removed first.
    server_kv_block_radix lru(/* block_tokens = */ 1);
    const std::vector<std::string> key_a = { "t:a" };
    const std::vector<std::string> key_b = { "t:b" };
    const std::vector<std::string> key_c = { "t:c" };
    lru.publish(key_a, { 1 }, /* seq = */ 10, /* now = */ 10);
    lru.publish(key_b, { 1 }, /* seq = */ 11, /* now = */ 20);
    lru.publish(key_c, { 1 }, /* seq = */ 12, /* now = */ 30);
    const auto a = lru.attach(key_a, /* seq = */ 20, /* now = */ 10);
    const auto b = lru.attach(key_b, /* seq = */ 21, /* now = */ 40);
    assert(a.block_ids.size() == 1 && b.block_ids.size() == 1);
    lru.release(10);
    lru.release(11);
    lru.release(12);
    lru.release(20);
    lru.release(21);
    assert(lru.evict(2) == 1);
    assert(!lru.contains(a.block_ids[0]));
    assert(lru.contains(b.block_ids[0]));
    assert(lru.n_blocks() == 2);

    return 0;
}
