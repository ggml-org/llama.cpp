#include "llama-kv-block.h"

#include <cstdio>
#include <cassert>
#include <vector>

// Simple test framework
#define TEST(name) static void test_##name()
#define RUN_TEST(name) do { \
    printf("Running test: %s... ", #name); \
    test_##name(); \
    printf("PASSED\n"); \
} while(0)

// =============================================================================
// Block Manager Tests
// =============================================================================

TEST(block_manager_init) {
    llama_kv_block_manager mgr(100, 16);

    assert(mgr.get_total_blocks() == 100);
    assert(mgr.get_free_blocks() == 100);
    assert(mgr.get_used_blocks() == 0);
    assert(mgr.get_block_size() == 16);
}

TEST(block_manager_allocate) {
    llama_kv_block_manager mgr(10, 16);

    // Allocate single block
    int32_t block_id = mgr.allocate();
    assert(block_id >= 0);
    assert(mgr.get_free_blocks() == 9);
    assert(mgr.get_used_blocks() == 1);

    // Verify block metadata
    const auto & block = mgr.get_block(block_id);
    assert(!block.is_free);
    assert(block.ref_count == 1);
}

TEST(block_manager_allocate_all) {
    llama_kv_block_manager mgr(5, 16);

    // Allocate all blocks
    for (int i = 0; i < 5; ++i) {
        int32_t block_id = mgr.allocate();
        assert(block_id >= 0);
    }

    assert(mgr.get_free_blocks() == 0);
    assert(mgr.get_used_blocks() == 5);

    // Next allocation should fail
    int32_t block_id = mgr.allocate();
    assert(block_id == -1);
}

TEST(block_manager_free) {
    llama_kv_block_manager mgr(10, 16);

    // Allocate and free
    int32_t block_id = mgr.allocate();
    assert(block_id >= 0);
    assert(mgr.get_free_blocks() == 9);

    mgr.free(block_id);
    assert(mgr.get_free_blocks() == 10);

    const auto & block = mgr.get_block(block_id);
    assert(block.is_free);
    assert(block.ref_count == 0);
}

TEST(block_manager_ref_counting) {
    llama_kv_block_manager mgr(10, 16);

    int32_t block_id = mgr.allocate();
    assert(mgr.get_ref_count(block_id) == 1);

    // Increment ref count
    mgr.ref_inc(block_id);
    assert(mgr.get_ref_count(block_id) == 2);

    // Decrement ref count (should not free)
    bool freed = mgr.ref_dec(block_id);
    assert(!freed);
    assert(mgr.get_ref_count(block_id) == 1);
    assert(mgr.get_free_blocks() == 9);

    // Decrement again (should free)
    freed = mgr.ref_dec(block_id);
    assert(freed);
    assert(mgr.get_free_blocks() == 10);
}

TEST(block_manager_allocate_n) {
    llama_kv_block_manager mgr(20, 16);

    // Allocate multiple blocks
    std::vector<int32_t> blocks = mgr.allocate_n(5);
    assert(blocks.size() == 5);
    assert(mgr.get_free_blocks() == 15);

    // Try to allocate more than available
    std::vector<int32_t> blocks2 = mgr.allocate_n(20);
    assert(blocks2.empty());  // Should fail
    assert(mgr.get_free_blocks() == 15);  // Nothing changed
}

TEST(block_manager_free_n) {
    llama_kv_block_manager mgr(20, 16);

    std::vector<int32_t> blocks = mgr.allocate_n(10);
    assert(mgr.get_free_blocks() == 10);

    mgr.free_n(blocks);
    assert(mgr.get_free_blocks() == 20);
}

TEST(block_manager_reset) {
    llama_kv_block_manager mgr(10, 16);

    mgr.allocate_n(5);
    assert(mgr.get_used_blocks() == 5);

    mgr.reset();
    assert(mgr.get_free_blocks() == 10);
    assert(mgr.get_used_blocks() == 0);
}

// =============================================================================
// Block Table Tests
// =============================================================================

TEST(block_table_basic) {
    llama_kv_block_table table(0);  // seq_id = 0

    assert(table.get_seq_id() == 0);
    assert(table.num_blocks() == 0);
    assert(table.get_num_tokens() == 0);

    // Add some blocks
    table.append_block(5);
    table.append_block(10);
    table.append_block(3);

    assert(table.num_blocks() == 3);

    // Test position lookups (block_size = 16)
    assert(table.get_physical_block(0) == 5);   // Position 0-15 -> block 5
    assert(table.get_physical_block(15) == 5);
    assert(table.get_physical_block(16) == 10); // Position 16-31 -> block 10
    assert(table.get_physical_block(32) == 3);  // Position 32-47 -> block 3
    assert(table.get_physical_block(48) == -1); // Out of range

    // Test offsets
    assert(llama_kv_block_table::get_block_offset(0) == 0);
    assert(llama_kv_block_table::get_block_offset(5) == 5);
    assert(llama_kv_block_table::get_block_offset(16) == 0);
    assert(llama_kv_block_table::get_block_offset(17) == 1);
}

TEST(block_table_pop) {
    llama_kv_block_table table(0);

    table.append_block(1);
    table.append_block(2);
    table.append_block(3);

    assert(table.num_blocks() == 3);

    int32_t popped = table.pop_block();
    assert(popped == 3);
    assert(table.num_blocks() == 2);

    popped = table.pop_block();
    assert(popped == 2);
    assert(table.num_blocks() == 1);
}

TEST(block_table_tokens_to_blocks) {
    // block_size = 16
    assert(llama_kv_block_table::tokens_to_blocks(0) == 0);
    assert(llama_kv_block_table::tokens_to_blocks(1) == 1);
    assert(llama_kv_block_table::tokens_to_blocks(16) == 1);
    assert(llama_kv_block_table::tokens_to_blocks(17) == 2);
    assert(llama_kv_block_table::tokens_to_blocks(32) == 2);
    assert(llama_kv_block_table::tokens_to_blocks(100) == 7);

    // Custom block size
    assert(llama_kv_block_table::tokens_to_blocks(100, 32) == 4);
    assert(llama_kv_block_table::tokens_to_blocks(128, 32) == 4);
    assert(llama_kv_block_table::tokens_to_blocks(129, 32) == 5);
}

// =============================================================================
// Paged KV Cache Tests
// =============================================================================

TEST(paged_cache_basic) {
    // 1024 context, 4 sequences max
    llama_kv_cache_paged cache(1024, 4, 16);

    // Initially all blocks free
    uint32_t total_blocks = llama_kv_block_table::tokens_to_blocks(1024, 16);
    assert(cache.get_total_blocks() == total_blocks);
    assert(cache.get_free_blocks() == total_blocks);
    assert(cache.get_num_sequences() == 0);
}

TEST(paged_cache_allocate_sequence) {
    llama_kv_cache_paged cache(1024, 4, 16);

    // Allocate for 100 tokens (needs 7 blocks)
    bool success = cache.allocate_for_sequence(0, 100);
    assert(success);
    assert(cache.has_sequence(0));
    assert(cache.get_sequence_length(0) == 100);
    assert(cache.get_used_blocks() == 7);

    // Allocate for another sequence
    success = cache.allocate_for_sequence(1, 50);  // Needs 4 blocks
    assert(success);
    assert(cache.has_sequence(1));
    assert(cache.get_used_blocks() == 11);
}

TEST(paged_cache_extend_sequence) {
    llama_kv_cache_paged cache(1024, 4, 16);

    cache.allocate_for_sequence(0, 16);  // Exactly 1 block
    assert(cache.get_used_blocks() == 1);

    // Extend by 1 token (should need new block)
    bool success = cache.extend_sequence(0, 1);
    assert(success);
    assert(cache.get_sequence_length(0) == 17);
    assert(cache.get_used_blocks() == 2);

    // Extend within same block
    success = cache.extend_sequence(0, 15);  // Total 32 tokens
    assert(success);
    assert(cache.get_sequence_length(0) == 32);
    assert(cache.get_used_blocks() == 2);  // Still 2 blocks
}

TEST(paged_cache_free_sequence) {
    llama_kv_cache_paged cache(1024, 4, 16);

    cache.allocate_for_sequence(0, 100);  // 7 blocks
    cache.allocate_for_sequence(1, 50);   // 4 blocks
    assert(cache.get_used_blocks() == 11);

    cache.free_sequence(0);
    assert(!cache.has_sequence(0));
    assert(cache.get_used_blocks() == 4);

    cache.free_sequence(1);
    assert(cache.get_used_blocks() == 0);
}

TEST(paged_cache_copy_sequence) {
    llama_kv_cache_paged cache(1024, 4, 16);

    cache.allocate_for_sequence(0, 100);
    uint32_t used_before = cache.get_used_blocks();

    // Copy sequence (uses copy-on-write, increments ref counts)
    bool success = cache.copy_sequence(0, 1);
    assert(success);
    assert(cache.has_sequence(1));
    assert(cache.get_sequence_length(1) == 100);

    // With CoW, used blocks should stay same (shared blocks)
    assert(cache.get_used_blocks() == used_before);

    // Free original - blocks should still be used by copy
    cache.free_sequence(0);
    assert(cache.get_used_blocks() == used_before);

    // Free copy - now blocks are freed
    cache.free_sequence(1);
    assert(cache.get_used_blocks() == 0);
}

TEST(paged_cache_truncate_sequence) {
    llama_kv_cache_paged cache(1024, 4, 16);

    cache.allocate_for_sequence(0, 100);  // 7 blocks
    assert(cache.get_used_blocks() == 7);

    // Truncate to 32 tokens (2 blocks)
    cache.truncate_sequence(0, 32);
    assert(cache.get_sequence_length(0) == 32);
    assert(cache.get_used_blocks() == 2);
}

TEST(paged_cache_block_table_tensor) {
    llama_kv_cache_paged cache(1024, 4, 16);

    cache.allocate_for_sequence(0, 48);  // 3 blocks
    cache.allocate_for_sequence(1, 32);  // 2 blocks

    std::vector<llama_seq_id> seq_ids = {0, 1};
    std::vector<int32_t> tensor = cache.build_block_table_tensor(seq_ids, 4);

    // Shape should be [2, 4] = 8 elements
    assert(tensor.size() == 8);

    // Seq 0: 3 blocks + 1 padding
    assert(tensor[0] >= 0);  // Block 0
    assert(tensor[1] >= 0);  // Block 1
    assert(tensor[2] >= 0);  // Block 2
    assert(tensor[3] == -1); // Padding

    // Seq 1: 2 blocks + 2 padding
    assert(tensor[4] >= 0);
    assert(tensor[5] >= 0);
    assert(tensor[6] == -1);
    assert(tensor[7] == -1);
}

TEST(paged_cache_memory_efficiency) {
    llama_kv_cache_paged cache(1024, 4, 16);

    // Perfect efficiency: all slots used
    cache.allocate_for_sequence(0, 16);  // 1 block, 16 slots, 16 tokens = 100%
    float eff = cache.get_memory_efficiency();
    assert(eff == 1.0f);

    // Reset
    cache.reset();

    // Partial efficiency: 17 tokens needs 2 blocks (32 slots)
    cache.allocate_for_sequence(0, 17);
    eff = cache.get_memory_efficiency();
    assert(eff > 0.5f && eff < 0.6f);  // 17/32 = 53.125%
}

TEST(paged_cache_out_of_memory) {
    // Very small cache: only 32 tokens worth of blocks
    llama_kv_cache_paged cache(32, 4, 16);

    // Should work
    bool success = cache.allocate_for_sequence(0, 32);
    assert(success);

    // Should fail (no more blocks)
    success = cache.allocate_for_sequence(1, 16);
    assert(!success);

    // Extend should also fail
    success = cache.extend_sequence(0, 16);
    assert(!success);
}

// =============================================================================
// Defragmentation Tests
// =============================================================================

TEST(defrag_fragmentation_ratio_no_fragmentation) {
    llama_kv_block_manager mgr(10, 16);

    // Empty cache: no fragmentation
    assert(mgr.get_fragmentation_ratio() == 0.0f);

    // Allocate blocks 0, 1, 2 contiguously
    mgr.allocate();  // Block 0
    mgr.allocate();  // Block 1
    mgr.allocate();  // Block 2

    // All used blocks are contiguous at start: no fragmentation
    assert(mgr.get_fragmentation_ratio() == 0.0f);
}

TEST(defrag_fragmentation_ratio_with_gaps) {
    llama_kv_block_manager mgr(10, 16);

    // Allocate blocks 0, 1, 2
    mgr.allocate();  // Block 0 (unused)
    int32_t b1 = mgr.allocate();
    mgr.allocate();  // Block 2 (unused)

    // Free middle block to create a gap
    mgr.free(b1);

    // Now we have blocks 0 and 2 used (gap at 1)
    // Used = 2, max_used_idx = 2, occupied_range = 3
    // Fragmentation = 1 - (2/3) = 0.333...
    float frag = mgr.get_fragmentation_ratio();
    assert(frag > 0.32f && frag < 0.34f);
}

TEST(defrag_needs_defrag_threshold) {
    llama_kv_block_manager mgr(10, 16);

    // No fragmentation: doesn't need defrag
    assert(!mgr.needs_defrag(0.2f));

    // Create fragmentation
    mgr.allocate();  // Block 0
    int32_t b1 = mgr.allocate();
    mgr.allocate();  // Block 2
    mgr.free(b1);  // Creates gap

    // 33% fragmentation > 20% threshold
    assert(mgr.needs_defrag(0.2f));

    // 33% fragmentation < 50% threshold
    assert(!mgr.needs_defrag(0.5f));
}

TEST(defrag_compute_plan_empty) {
    llama_kv_block_manager mgr(10, 16);

    // Empty: no moves needed
    auto moves = mgr.compute_defrag_plan();
    assert(moves.empty());
}

TEST(defrag_compute_plan_no_fragmentation) {
    llama_kv_block_manager mgr(10, 16);

    // Contiguous allocation: no moves needed
    mgr.allocate();
    mgr.allocate();
    mgr.allocate();

    auto moves = mgr.compute_defrag_plan();
    assert(moves.empty());
}

TEST(defrag_compute_plan_with_gap) {
    llama_kv_block_manager mgr(10, 16);

    // Allocate 0, 1, 2, then free 1
    mgr.allocate();               // Block 0
    int32_t b1 = mgr.allocate();  // Block 1
    mgr.allocate();               // Block 2
    mgr.free(b1);                 // Free block 1

    // Now: used = [0, 2], gap at 1
    // Plan should move block 2 to slot 1
    auto moves = mgr.compute_defrag_plan();
    assert(moves.size() == 1);
    assert(moves[0].src_block == 2);
    assert(moves[0].dst_block == 1);
}

TEST(defrag_compute_plan_multiple_gaps) {
    llama_kv_block_manager mgr(10, 16);

    // Allocate 0, 1, 2, 3, 4, then free 1 and 3
    mgr.allocate();  // Block 0
    int32_t b1 = mgr.allocate();  // Block 1
    mgr.allocate();  // Block 2
    int32_t b3 = mgr.allocate();  // Block 3
    mgr.allocate();  // Block 4
    mgr.free(b1);    // Free 1
    mgr.free(b3);    // Free 3

    // Now: used = [0, 2, 4], gaps at [1, 3]
    // Plan should move 4 to 1 (only one move since used=3, goal is [0,1,2])
    auto moves = mgr.compute_defrag_plan();
    assert(moves.size() == 1);
    assert(moves[0].src_block == 4);
    assert(moves[0].dst_block == 1);  // First free slot < used
}

TEST(defrag_apply_updates_metadata) {
    llama_kv_block_manager mgr(10, 16);

    // Create fragmented state
    mgr.allocate();  // Block 0
    int32_t b1 = mgr.allocate();  // Block 1
    mgr.allocate();  // Block 2
    mgr.free(b1);

    auto moves = mgr.compute_defrag_plan();
    assert(!moves.empty());

    // Apply defrag
    mgr.apply_defrag(moves);

    // After defrag: blocks 0, 1 should be used, block 2 should be free
    assert(!mgr.get_block(0).is_free);
    assert(!mgr.get_block(1).is_free);
    assert(mgr.get_block(2).is_free);

    // Fragmentation should be 0
    assert(mgr.get_fragmentation_ratio() == 0.0f);
}

TEST(paged_cache_defrag_updates_block_tables) {
    llama_kv_cache_paged cache(256, 4, 16);

    // Allocate sequence 0 with 32 tokens (2 blocks: 0, 1)
    cache.allocate_for_sequence(0, 32);

    // Allocate sequence 1 with 16 tokens (1 block: 2)
    cache.allocate_for_sequence(1, 16);

    // Allocate sequence 2 with 16 tokens (1 block: 3)
    cache.allocate_for_sequence(2, 16);

    // Free sequence 1 (frees block 2)
    cache.free_sequence(1);

    // Now blocks used: [0, 1, 3], gap at 2
    // Fragmentation exists
    assert(cache.get_fragmentation_ratio() > 0.0f);

    // Compute and apply defrag
    auto moves = cache.compute_defrag_plan();
    assert(moves.size() == 1);
    assert(moves[0].src_block == 3);
    assert(moves[0].dst_block == 2);

    // Apply defrag
    cache.apply_defrag(moves);

    // After defrag: sequence 2 should now use block 2 instead of 3
    const auto * table = cache.get_block_table(2);
    assert(table != nullptr);
    const auto & block_ids = table->get_block_ids();
    assert(block_ids.size() == 1);
    assert(block_ids[0] == 2);  // Block remapped from 3 to 2

    // Fragmentation should be 0
    assert(cache.get_fragmentation_ratio() == 0.0f);
}

TEST(paged_cache_defrag_no_change_when_not_needed) {
    llama_kv_cache_paged cache(256, 4, 16);

    // Contiguous allocation
    cache.allocate_for_sequence(0, 32);
    cache.allocate_for_sequence(1, 32);

    // No fragmentation
    assert(cache.get_fragmentation_ratio() == 0.0f);

    // No moves needed
    auto moves = cache.compute_defrag_plan();
    assert(moves.empty());
}

// =============================================================================
// Prefix Caching Tests
// =============================================================================

TEST(prefix_hash_function) {
    // Test that FNV-1a hash produces consistent results
    int32_t tokens1[] = {1, 2, 3, 4};
    int32_t tokens2[] = {1, 2, 3, 4};
    int32_t tokens3[] = {1, 2, 3, 5};

    uint64_t hash1 = llama_hash_tokens(tokens1, 4, 0);
    uint64_t hash2 = llama_hash_tokens(tokens2, 4, 0);
    uint64_t hash3 = llama_hash_tokens(tokens3, 4, 0);

    // Same tokens should produce same hash
    assert(hash1 == hash2);

    // Different tokens should produce different hash
    assert(hash1 != hash3);

    // Different layer_id should produce different hash for same tokens
    uint64_t hash1_layer0 = llama_hash_tokens(tokens1, 4, 0);
    uint64_t hash1_layer1 = llama_hash_tokens(tokens1, 4, 1);
    assert(hash1_layer0 != hash1_layer1);
}

TEST(prefix_cache_disabled_by_default) {
    llama_kv_block_manager mgr(10, 16);

    // Prefix caching should be disabled by default
    assert(!mgr.is_prefix_caching_enabled());

    // find_block_by_hash should return -1 when disabled
    assert(mgr.find_block_by_hash(12345) == -1);
}

TEST(prefix_cache_enable_disable) {
    llama_kv_block_manager mgr(10, 16);

    mgr.set_prefix_caching(true);
    assert(mgr.is_prefix_caching_enabled());

    mgr.set_prefix_caching(false);
    assert(!mgr.is_prefix_caching_enabled());
}

TEST(prefix_cache_set_and_find_hash) {
    llama_kv_block_manager mgr(10, 16);
    mgr.set_prefix_caching(true);

    // Allocate a block and set its hash
    int32_t block_id = mgr.allocate();
    assert(block_id >= 0);

    uint64_t test_hash = 0x123456789ABCDEF0ULL;
    mgr.set_content_hash(static_cast<uint32_t>(block_id), test_hash);

    // Should be able to find block by hash
    int32_t found = mgr.find_block_by_hash(test_hash);
    assert(found == block_id);

    // Non-existent hash should return -1
    assert(mgr.find_block_by_hash(0xDEADBEEFULL) == -1);
}

TEST(prefix_cache_allocate_or_share) {
    llama_kv_block_manager mgr(10, 16);
    mgr.set_prefix_caching(true);

    uint64_t test_hash = 0xABCDEF0123456789ULL;

    // First allocation should create a new block
    auto [block1, shared1] = mgr.allocate_or_share(test_hash);
    assert(block1 >= 0);
    assert(!shared1);  // New block, not shared
    assert(mgr.get_ref_count(static_cast<uint32_t>(block1)) == 1);

    // Second allocation with same hash should share the block
    auto [block2, shared2] = mgr.allocate_or_share(test_hash);
    assert(block2 == block1);  // Same block
    assert(shared2);  // Was shared
    assert(mgr.get_ref_count(static_cast<uint32_t>(block1)) == 2);

    // Different hash should allocate new block
    auto [block3, shared3] = mgr.allocate_or_share(0xDEADBEEF);
    assert(block3 != block1);
    assert(!shared3);
}

TEST(prefix_cache_statistics) {
    llama_kv_block_manager mgr(10, 16);
    mgr.set_prefix_caching(true);

    // Initially no hits or misses
    assert(mgr.get_prefix_cache_hits() == 0);
    assert(mgr.get_prefix_cache_misses() == 0);

    // First allocation is a miss
    mgr.allocate_or_share(0x111);
    assert(mgr.get_prefix_cache_hits() == 0);
    assert(mgr.get_prefix_cache_misses() == 1);

    // Second allocation with same hash is a hit
    mgr.allocate_or_share(0x111);
    assert(mgr.get_prefix_cache_hits() == 1);
    assert(mgr.get_prefix_cache_misses() == 1);

    // Reset stats
    mgr.reset_prefix_cache_stats();
    assert(mgr.get_prefix_cache_hits() == 0);
    assert(mgr.get_prefix_cache_misses() == 0);
}

TEST(prefix_cache_hash_cleared_on_free) {
    llama_kv_block_manager mgr(10, 16);
    mgr.set_prefix_caching(true);

    uint64_t test_hash = 0x999;

    // Allocate and set hash
    int32_t block_id = mgr.allocate();
    mgr.set_content_hash(static_cast<uint32_t>(block_id), test_hash);
    assert(mgr.find_block_by_hash(test_hash) == block_id);

    // Free the block
    mgr.free(static_cast<uint32_t>(block_id));

    // Hash should no longer be findable
    assert(mgr.find_block_by_hash(test_hash) == -1);
}

TEST(paged_cache_prefix_caching) {
    llama_kv_cache_paged cache(256, 4, 16);
    cache.set_prefix_caching(true);

    assert(cache.is_prefix_caching_enabled());

    // Create some token hashes
    int32_t tokens[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    auto hashes = cache.compute_block_hashes(tokens, 16);

    // Should have 1 hash for 16 tokens with block_size=16
    assert(hashes.size() == 1);

    // Allocate first sequence with these hashes
    bool success = cache.allocate_with_prefix(0, 16, hashes);
    assert(success);

    // Allocate second sequence with same prefix
    success = cache.allocate_with_prefix(1, 16, hashes);
    assert(success);

    // Should have prefix cache hit
    assert(cache.get_prefix_cache_hits() == 1);
    assert(cache.get_prefix_cache_misses() == 1);  // First allocation was a miss

    // Both sequences should use the same block
    const auto * table0 = cache.get_block_table(0);
    const auto * table1 = cache.get_block_table(1);
    assert(table0 && table1);
    assert(table0->get_block_ids()[0] == table1->get_block_ids()[0]);
}

TEST(paged_cache_prefix_sharing_ratio) {
    llama_kv_cache_paged cache(256, 4, 16);
    cache.set_prefix_caching(true);

    // Create hashes
    int32_t tokens[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    auto hashes = cache.compute_block_hashes(tokens, 16);

    // No sequences yet
    assert(cache.get_prefix_sharing_ratio() == 0.0f);

    // Allocate first sequence
    cache.allocate_with_prefix(0, 16, hashes);
    assert(cache.get_prefix_sharing_ratio() == 0.0f);  // No sharing yet

    // Allocate second sequence with same prefix
    cache.allocate_with_prefix(1, 16, hashes);

    // Now both sequences share the same block
    // Both refs point to shared block, so sharing ratio = 1.0
    float ratio = cache.get_prefix_sharing_ratio();
    assert(ratio > 0.99f);  // Should be 1.0 (100% sharing)
}

TEST(paged_cache_compute_block_hashes) {
    llama_kv_cache_paged cache(256, 4, 16);

    // 32 tokens = 2 blocks
    int32_t tokens[32];
    for (int i = 0; i < 32; i++) tokens[i] = i;

    auto hashes = cache.compute_block_hashes(tokens, 32);
    assert(hashes.size() == 2);

    // First hash is for tokens 0-15, second is for tokens 0-31 (cumulative)
    // They should be different
    assert(hashes[0] != hashes[1]);

    // Verify consistency
    auto hashes2 = cache.compute_block_hashes(tokens, 32);
    assert(hashes[0] == hashes2[0]);
    assert(hashes[1] == hashes2[1]);
}

TEST(paged_cache_set_block_hash) {
    llama_kv_cache_paged cache(256, 4, 16);
    cache.set_prefix_caching(true);

    // Allocate without prefix (old method)
    cache.allocate_for_sequence(0, 16);

    // Set hash after the fact
    cache.set_block_hash(0, 0, 0xCAFEBABE);

    // Create another sequence with same hash should share
    std::vector<uint64_t> hashes = {0xCAFEBABE};
    cache.allocate_with_prefix(1, 16, hashes);

    // Should have found the existing block
    assert(cache.get_prefix_cache_hits() == 1);
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("=== KV Block Manager Tests ===\n\n");

    // Block Manager Tests
    RUN_TEST(block_manager_init);
    RUN_TEST(block_manager_allocate);
    RUN_TEST(block_manager_allocate_all);
    RUN_TEST(block_manager_free);
    RUN_TEST(block_manager_ref_counting);
    RUN_TEST(block_manager_allocate_n);
    RUN_TEST(block_manager_free_n);
    RUN_TEST(block_manager_reset);

    printf("\n");

    // Block Table Tests
    RUN_TEST(block_table_basic);
    RUN_TEST(block_table_pop);
    RUN_TEST(block_table_tokens_to_blocks);

    printf("\n");

    // Paged KV Cache Tests
    RUN_TEST(paged_cache_basic);
    RUN_TEST(paged_cache_allocate_sequence);
    RUN_TEST(paged_cache_extend_sequence);
    RUN_TEST(paged_cache_free_sequence);
    RUN_TEST(paged_cache_copy_sequence);
    RUN_TEST(paged_cache_truncate_sequence);
    RUN_TEST(paged_cache_block_table_tensor);
    RUN_TEST(paged_cache_memory_efficiency);
    RUN_TEST(paged_cache_out_of_memory);

    printf("\n");

    // Defragmentation Tests
    RUN_TEST(defrag_fragmentation_ratio_no_fragmentation);
    RUN_TEST(defrag_fragmentation_ratio_with_gaps);
    RUN_TEST(defrag_needs_defrag_threshold);
    RUN_TEST(defrag_compute_plan_empty);
    RUN_TEST(defrag_compute_plan_no_fragmentation);
    RUN_TEST(defrag_compute_plan_with_gap);
    RUN_TEST(defrag_compute_plan_multiple_gaps);
    RUN_TEST(defrag_apply_updates_metadata);
    RUN_TEST(paged_cache_defrag_updates_block_tables);
    RUN_TEST(paged_cache_defrag_no_change_when_not_needed);

    printf("\n");

    // Prefix Caching Tests
    RUN_TEST(prefix_hash_function);
    RUN_TEST(prefix_cache_disabled_by_default);
    RUN_TEST(prefix_cache_enable_disable);
    RUN_TEST(prefix_cache_set_and_find_hash);
    RUN_TEST(prefix_cache_allocate_or_share);
    RUN_TEST(prefix_cache_statistics);
    RUN_TEST(prefix_cache_hash_cleared_on_free);
    RUN_TEST(paged_cache_prefix_caching);
    RUN_TEST(paged_cache_prefix_sharing_ratio);
    RUN_TEST(paged_cache_compute_block_hashes);
    RUN_TEST(paged_cache_set_block_hash);

    printf("\n=== All tests passed! ===\n");

    return 0;
}
