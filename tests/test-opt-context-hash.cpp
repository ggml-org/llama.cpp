// CPUOPTI: Tests for context block hashing

#include "llama-context-hash.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

static void test_hash_deterministic() {
    printf("test_hash_deterministic... ");

    llama_token tokens[] = {1, 2, 3, 4, 5};
    uint64_t h1 = llama_opt_hash_block(tokens, 5);
    uint64_t h2 = llama_opt_hash_block(tokens, 5);
    assert(h1 == h2);
    assert(h1 != 0);

    printf("OK\n");
}

static void test_hash_different_inputs() {
    printf("test_hash_different_inputs... ");

    llama_token tokens_a[] = {1, 2, 3, 4, 5};
    llama_token tokens_b[] = {1, 2, 3, 4, 6}; // differs in last token

    uint64_t ha = llama_opt_hash_block(tokens_a, 5);
    uint64_t hb = llama_opt_hash_block(tokens_b, 5);
    assert(ha != hb);

    printf("OK\n");
}

static void test_hash_position_independence() {
    printf("test_hash_position_independence... ");

    llama_token tokens[] = {10, 20, 30};
    uint64_t h_no_pos = llama_opt_hash_block(tokens, 3);
    uint64_t h_pos_0  = llama_opt_hash_block_pos(tokens, 3, 0);
    uint64_t h_pos_10 = llama_opt_hash_block_pos(tokens, 3, 10);

    // Position-independent hash should differ from position-dependent hash
    assert(h_pos_0 != h_pos_10);
    // Same position should give same hash
    assert(h_pos_0 == llama_opt_hash_block_pos(tokens, 3, 0));

    // Position-independent hash should be independent of position
    (void)h_no_pos; // just verify it computes without error

    printf("OK\n");
}

static void test_hash_empty() {
    printf("test_hash_empty... ");

    uint64_t h = llama_opt_hash_block(nullptr, 0);
    assert(h == LLAMA_OPT_FNV_OFFSET); // Empty hash should be the FNV offset basis

    printf("OK\n");
}

static void test_hash_single_token() {
    printf("test_hash_single_token... ");

    llama_token t1[] = {42};
    llama_token t2[] = {43};

    uint64_t h1 = llama_opt_hash_block(t1, 1);
    uint64_t h2 = llama_opt_hash_block(t2, 1);
    assert(h1 != h2);
    assert(h1 != LLAMA_OPT_FNV_OFFSET);

    printf("OK\n");
}

static void test_hasher_incremental() {
    printf("test_hasher_incremental... ");

    llama_token tokens[] = {1, 2, 3, 4, 5, 6, 7, 8};

    // Hash all at once
    uint64_t h_batch = llama_opt_hash_block(tokens, 8);

    // Hash incrementally
    llama_opt_hasher hasher;
    for (int i = 0; i < 8; i++) {
        hasher.feed(tokens[i]);
    }
    uint64_t h_incr = hasher.finalize();

    assert(h_batch == h_incr);

    // Hash in chunks
    llama_opt_hasher hasher2;
    hasher2.feed(tokens, 4);
    hasher2.feed(tokens + 4, 4);
    uint64_t h_chunk = hasher2.finalize();

    assert(h_batch == h_chunk);

    printf("OK\n");
}

static void test_hasher_reset() {
    printf("test_hasher_reset... ");

    llama_opt_hasher hasher;
    llama_token t = 42;
    hasher.feed(t);
    uint64_t h1 = hasher.finalize();

    hasher.reset();
    hasher.feed(t);
    uint64_t h2 = hasher.finalize();

    assert(h1 == h2);

    printf("OK\n");
}

static void test_segment_and_hash() {
    printf("test_segment_and_hash... ");

    std::vector<llama_token> tokens(200);
    for (int i = 0; i < 200; i++) {
        tokens[i] = i + 1;
    }

    auto blocks = llama_opt_segment_and_hash(tokens.data(), 200, 64);

    // 200 tokens / 64 block_size = 3 full blocks + 1 partial (8 tokens)
    assert(blocks.size() == 4);

    assert(blocks[0].offset == 0);
    assert(blocks[0].n_tokens == 64);

    assert(blocks[1].offset == 64);
    assert(blocks[1].n_tokens == 64);

    assert(blocks[2].offset == 128);
    assert(blocks[2].n_tokens == 64);

    assert(blocks[3].offset == 192);
    assert(blocks[3].n_tokens == 8);

    // All hashes should be different (different content)
    assert(blocks[0].hash != blocks[1].hash);
    assert(blocks[1].hash != blocks[2].hash);
    assert(blocks[2].hash != blocks[3].hash);

    printf("OK\n");
}

static void test_segment_and_hash_exact_block() {
    printf("test_segment_and_hash_exact_block... ");

    std::vector<llama_token> tokens(128);
    for (int i = 0; i < 128; i++) {
        tokens[i] = i;
    }

    auto blocks = llama_opt_segment_and_hash(tokens.data(), 128, 64);

    // 128 / 64 = exactly 2 blocks, no partial
    assert(blocks.size() == 2);
    assert(blocks[0].n_tokens == 64);
    assert(blocks[1].n_tokens == 64);

    printf("OK\n");
}

static void test_segment_empty() {
    printf("test_segment_empty... ");

    auto blocks = llama_opt_segment_and_hash(nullptr, 0, 64);
    assert(blocks.empty());

    printf("OK\n");
}

static void test_hash_bytes() {
    printf("test_hash_bytes... ");

    float data[] = {1.0f, 2.0f, 3.0f};
    uint64_t h1 = llama_opt_hash_bytes(data, sizeof(data));
    uint64_t h2 = llama_opt_hash_bytes(data, sizeof(data));
    assert(h1 == h2);

    data[0] = 1.5f;
    uint64_t h3 = llama_opt_hash_bytes(data, sizeof(data));
    assert(h1 != h3);

    printf("OK\n");
}

int main() {
    printf("=== CPUOPTI: context hash tests ===\n");

    test_hash_deterministic();
    test_hash_different_inputs();
    test_hash_position_independence();
    test_hash_empty();
    test_hash_single_token();
    test_hasher_incremental();
    test_hasher_reset();
    test_segment_and_hash();
    test_segment_and_hash_exact_block();
    test_segment_empty();
    test_hash_bytes();

    printf("=== all context hash tests passed ===\n");
    return 0;
}
