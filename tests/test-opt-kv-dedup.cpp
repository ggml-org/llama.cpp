// CPUOPTI: Tests for KV cache context block deduplication

#include "llama-kv-cache-dedup.h"

#include <cassert>
#include <cstdio>
#include <vector>

static void test_pool_basic() {
    printf("test_pool_basic... ");

    llama_opt_block_pool pool(100);
    assert(pool.size() == 0);
    assert(pool.capacity() == 100);

    llama_token tokens[] = {1, 2, 3, 4, 5};
    uint64_t hash = llama_opt_hash_block(tokens, 5);

    // Insert a block
    auto * block = pool.insert(hash, tokens, 5);
    assert(block != nullptr);
    assert(pool.size() == 1);
    assert(block->hash == hash);
    assert(block->n_tokens == 5);
    assert(block->ref_count == 0);

    // Lookup should find it
    auto * found = pool.lookup(hash, tokens, 5);
    assert(found != nullptr);
    assert(found->hash == hash);

    printf("OK\n");
}

static void test_pool_collision_handling() {
    printf("test_pool_collision_handling... ");

    llama_opt_block_pool pool(100);

    // Two different token blocks that happen to share the same hash bucket
    // (in practice, hash collisions are rare, but we test the verification)
    llama_token tokens_a[] = {10, 20, 30};
    llama_token tokens_b[] = {40, 50, 60};

    uint64_t hash_a = llama_opt_hash_block(tokens_a, 3);
    uint64_t hash_b = llama_opt_hash_block(tokens_b, 3);

    pool.insert(hash_a, tokens_a, 3);
    pool.insert(hash_b, tokens_b, 3);

    // Each should find its own block
    auto * found_a = pool.lookup(hash_a, tokens_a, 3);
    auto * found_b = pool.lookup(hash_b, tokens_b, 3);

    assert(found_a != nullptr);
    assert(found_b != nullptr);
    assert(found_a->tokens[0] == 10);
    assert(found_b->tokens[0] == 40);

    // Looking up with wrong tokens (same hash but different content) should fail
    auto * not_found = pool.lookup(hash_a, tokens_b, 3);
    assert(not_found == nullptr);

    printf("OK\n");
}

static void test_pool_ref_counting() {
    printf("test_pool_ref_counting... ");

    llama_opt_block_pool pool(100);

    llama_token tokens[] = {1, 2, 3};
    uint64_t hash = llama_opt_hash_block(tokens, 3);

    auto * block = pool.insert(hash, tokens, 3);
    assert(block->ref_count == 0);

    pool.ref(block);
    assert(block->ref_count == 1);

    pool.ref(block);
    assert(block->ref_count == 2);

    pool.unref(block);
    assert(block->ref_count == 1);

    pool.unref(block);
    assert(block->ref_count == 0);

    printf("OK\n");
}

static void test_pool_eviction_lru() {
    printf("test_pool_eviction_lru... ");

    llama_opt_block_pool pool(3); // Max 3 blocks

    // Insert 3 blocks
    for (int i = 0; i < 3; i++) {
        llama_token tokens[] = {(llama_token)(i * 10)};
        uint64_t hash = llama_opt_hash_block(tokens, 1);
        pool.insert(hash, tokens, 1);
        pool.advance_turn();
    }

    assert(pool.size() == 3);

    // Insert a 4th — should evict the oldest (first inserted)
    llama_token tokens_new[] = {99};
    uint64_t hash_new = llama_opt_hash_block(tokens_new, 1);
    auto * new_block = pool.insert(hash_new, tokens_new, 1);

    assert(new_block != nullptr);
    assert(pool.size() == 3); // Still 3 (one evicted)

    // The first block should be evicted
    llama_token tokens_first[] = {0};
    uint64_t hash_first = llama_opt_hash_block(tokens_first, 1);
    auto * first = pool.lookup(hash_first, tokens_first, 1);
    assert(first == nullptr); // Should be evicted

    // The new block should be present
    auto * found = pool.lookup(hash_new, tokens_new, 1);
    assert(found != nullptr);

    printf("OK\n");
}

static void test_pool_eviction_respects_refs() {
    printf("test_pool_eviction_respects_refs... ");

    llama_opt_block_pool pool(2); // Max 2 blocks

    // Insert block 1 and ref it
    llama_token tokens1[] = {1};
    uint64_t hash1 = llama_opt_hash_block(tokens1, 1);
    auto * block1 = pool.insert(hash1, tokens1, 1);
    pool.ref(block1);
    pool.advance_turn();

    // Insert block 2 (unreferenced)
    llama_token tokens2[] = {2};
    uint64_t hash2 = llama_opt_hash_block(tokens2, 1);
    pool.insert(hash2, tokens2, 1);
    pool.advance_turn();

    // Insert block 3 — should evict block 2 (unreferenced), not block 1 (referenced)
    llama_token tokens3[] = {3};
    uint64_t hash3 = llama_opt_hash_block(tokens3, 1);
    pool.insert(hash3, tokens3, 1);

    // Block 1 should still be present (referenced)
    assert(pool.lookup(hash1, tokens1, 1) != nullptr);

    // Block 2 should be evicted (unreferenced and older)
    assert(pool.lookup(hash2, tokens2, 1) == nullptr);

    // Block 3 should be present
    assert(pool.lookup(hash3, tokens3, 1) != nullptr);

    printf("OK\n");
}

static void test_pool_evict_stale() {
    printf("test_pool_evict_stale... ");

    llama_opt_block_pool pool(100);

    // Insert blocks at different turns
    for (int i = 0; i < 5; i++) {
        llama_token tokens[] = {(llama_token)(i)};
        uint64_t hash = llama_opt_hash_block(tokens, 1);
        pool.insert(hash, tokens, 1);
        pool.advance_turn();
    }

    assert(pool.size() == 5);

    // Evict blocks older than turn 3
    uint32_t evicted = pool.evict_stale(3);
    assert(evicted == 3); // Turns 0, 1, 2 should be evicted
    assert(pool.size() == 2);

    printf("OK\n");
}

static void test_dedup_analyze() {
    printf("test_dedup_analyze... ");

    llama_opt_block_pool pool(100);
    const uint32_t block_size = 4;

    // Create a context with repeated blocks
    llama_token context[] = {
        1, 2, 3, 4,   // block 0
        5, 6, 7, 8,   // block 1
        1, 2, 3, 4,   // block 2 (same as block 0)
        9, 10, 11, 12  // block 3
    };

    // First pass — no hits (pool is empty)
    auto plan1 = llama_opt_dedup_analyze(pool, context, 16, block_size);
    assert(plan1.blocks_total == 4);
    assert(plan1.blocks_hit == 0);
    assert(plan1.tokens_saved == 0);

    // Insert blocks from first pass
    auto blocks = llama_opt_segment_and_hash(context, 16, block_size);
    for (const auto & bh : blocks) {
        pool.insert(bh.hash, context + bh.offset, bh.n_tokens);
    }

    // Second pass — should find hits for duplicate blocks
    auto plan2 = llama_opt_dedup_analyze(pool, context, 16, block_size);
    assert(plan2.blocks_total == 4);
    assert(plan2.blocks_hit == 4); // All blocks now in pool
    assert(plan2.tokens_saved == 16);

    printf("OK\n");
}

int main() {
    printf("=== CPUOPTI: KV dedup tests ===\n");

    test_pool_basic();
    test_pool_collision_handling();
    test_pool_ref_counting();
    test_pool_eviction_lru();
    test_pool_eviction_respects_refs();
    test_pool_evict_stale();
    test_dedup_analyze();

    printf("=== all KV dedup tests passed ===\n");
    return 0;
}
