// Unit tests for llama_kv_block_pool and llama_kv_block_table
// Tests: allocation, deallocation, reference counting, sequence management

// Include from src directory
#include "../src/llama-kv-block.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#include <set>

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
        return false; \
    } \
} while(0)

//
// Block Pool Tests
//

static bool test_pool_init() {
    printf("  test_pool_init... ");

    llama_kv_block_pool pool;
    pool.init(100);

    TEST_ASSERT(pool.n_total() == 100);
    TEST_ASSERT(pool.n_free() == 100);
    TEST_ASSERT(pool.n_used() == 0);

    printf("OK\n");
    return true;
}

static bool test_pool_allocate_single() {
    printf("  test_pool_allocate_single... ");

    llama_kv_block_pool pool;
    pool.init(10);

    int32_t idx = pool.allocate();
    TEST_ASSERT(idx >= 0 && idx < 10);
    TEST_ASSERT(pool.n_free() == 9);
    TEST_ASSERT(pool.n_used() == 1);
    TEST_ASSERT(!pool.get(idx).is_free());
    TEST_ASSERT(pool.get(idx).ref_count == 1);

    printf("OK\n");
    return true;
}

static bool test_pool_allocate_all() {
    printf("  test_pool_allocate_all... ");

    llama_kv_block_pool pool;
    pool.init(5);

    std::vector<int32_t> allocated;
    for (int i = 0; i < 5; i++) {
        int32_t idx = pool.allocate();
        TEST_ASSERT(idx >= 0);
        allocated.push_back(idx);
    }

    TEST_ASSERT(pool.n_free() == 0);
    TEST_ASSERT(pool.n_used() == 5);

    // Next allocation should fail
    int32_t idx = pool.allocate();
    TEST_ASSERT(idx == -1);

    // Verify all indices are unique
    std::set<int32_t> unique(allocated.begin(), allocated.end());
    TEST_ASSERT(unique.size() == 5);

    printf("OK\n");
    return true;
}

static bool test_pool_deallocate() {
    printf("  test_pool_deallocate... ");

    llama_kv_block_pool pool;
    pool.init(10);

    int32_t idx1 = pool.allocate();
    int32_t idx2 = pool.allocate();
    TEST_ASSERT(pool.n_used() == 2);

    pool.deallocate(idx1);
    TEST_ASSERT(pool.n_used() == 1);
    TEST_ASSERT(pool.get(idx1).is_free());
    TEST_ASSERT(!pool.get(idx2).is_free());

    // Reallocate - should get same block back (LIFO)
    int32_t idx3 = pool.allocate();
    TEST_ASSERT(idx3 == idx1);

    printf("OK\n");
    return true;
}

static bool test_pool_batch_allocate() {
    printf("  test_pool_batch_allocate... ");

    llama_kv_block_pool pool;
    pool.init(10);

    auto blocks = pool.allocate_batch(5);
    TEST_ASSERT(blocks.size() == 5);
    TEST_ASSERT(pool.n_used() == 5);

    // Allocate more than available should fail
    auto blocks2 = pool.allocate_batch(6);
    TEST_ASSERT(blocks2.empty());
    TEST_ASSERT(pool.n_used() == 5);  // No change

    printf("OK\n");
    return true;
}

static bool test_pool_reference_counting() {
    printf("  test_pool_reference_counting... ");

    llama_kv_block_pool pool;
    pool.init(10);

    int32_t idx = pool.allocate();
    TEST_ASSERT(pool.get(idx).ref_count == 1);

    pool.add_ref(idx);
    TEST_ASSERT(pool.get(idx).ref_count == 2);
    TEST_ASSERT(pool.get(idx).is_shared());

    pool.deallocate(idx);
    TEST_ASSERT(pool.get(idx).ref_count == 1);
    TEST_ASSERT(!pool.get(idx).is_shared());
    TEST_ASSERT(!pool.get(idx).is_free());

    pool.deallocate(idx);
    TEST_ASSERT(pool.get(idx).is_free());
    TEST_ASSERT(pool.n_free() == 10);

    printf("OK\n");
    return true;
}

static bool test_pool_stats() {
    printf("  test_pool_stats... ");

    llama_kv_block_pool pool;
    pool.init(100);

    const uint32_t tokens_per_block = 64;

    // Allocate some blocks and set token counts
    for (int i = 0; i < 10; i++) {
        int32_t idx = pool.allocate();
        pool.get(idx).n_tokens = (i == 9) ? 32 : 64;  // Last block half full
    }

    auto stats = pool.compute_stats(tokens_per_block);

    TEST_ASSERT(stats.n_blocks_total == 100);
    TEST_ASSERT(stats.n_blocks_used == 10);
    TEST_ASSERT(stats.n_blocks_free == 90);
    TEST_ASSERT(stats.n_tokens_total == 6400);  // 100 * 64
    TEST_ASSERT(stats.n_tokens_used == 9 * 64 + 32);  // 608
    TEST_ASSERT(stats.n_tokens_wasted == 32);  // 10 * 64 - 608

    printf("OK\n");
    return true;
}

static bool test_pool_clear() {
    printf("  test_pool_clear... ");

    llama_kv_block_pool pool;
    pool.init(10);

    pool.allocate();
    pool.allocate();
    pool.allocate();
    TEST_ASSERT(pool.n_used() == 3);

    pool.clear();
    TEST_ASSERT(pool.n_used() == 0);
    TEST_ASSERT(pool.n_free() == 10);

    printf("OK\n");
    return true;
}

//
// Block Table Tests
//

static bool test_table_mapping() {
    printf("  test_table_mapping... ");

    llama_kv_block_table table;

    // Set mapping for seq 0
    table.set_mapping(0, 0, 5);
    table.set_mapping(0, 1, 10);
    table.set_mapping(0, 2, 15);

    TEST_ASSERT(table.get_physical(0, 0) == 5);
    TEST_ASSERT(table.get_physical(0, 1) == 10);
    TEST_ASSERT(table.get_physical(0, 2) == 15);
    TEST_ASSERT(table.get_physical(0, 3) == -1);  // Not set

    // Non-existent sequence
    TEST_ASSERT(table.get_physical(1, 0) == -1);

    printf("OK\n");
    return true;
}

static bool test_table_append() {
    printf("  test_table_append... ");

    llama_kv_block_table table;

    uint32_t idx0 = table.append_block(0, 100);
    uint32_t idx1 = table.append_block(0, 200);
    uint32_t idx2 = table.append_block(0, 300);

    TEST_ASSERT(idx0 == 0);
    TEST_ASSERT(idx1 == 1);
    TEST_ASSERT(idx2 == 2);

    TEST_ASSERT(table.get_physical(0, 0) == 100);
    TEST_ASSERT(table.get_physical(0, 1) == 200);
    TEST_ASSERT(table.get_physical(0, 2) == 300);

    printf("OK\n");
    return true;
}

static bool test_table_sequence_blocks() {
    printf("  test_table_sequence_blocks... ");

    llama_kv_block_table table;

    table.append_block(0, 10);
    table.append_block(0, 20);
    table.append_block(1, 30);

    TEST_ASSERT(table.get_sequence_n_blocks(0) == 2);
    TEST_ASSERT(table.get_sequence_n_blocks(1) == 1);
    TEST_ASSERT(table.get_sequence_n_blocks(2) == 0);

    const auto * blocks = table.get_sequence_blocks(0);
    TEST_ASSERT(blocks != nullptr);
    TEST_ASSERT(blocks->size() == 2);
    TEST_ASSERT((*blocks)[0] == 10);
    TEST_ASSERT((*blocks)[1] == 20);

    printf("OK\n");
    return true;
}

static bool test_table_clear_sequence() {
    printf("  test_table_clear_sequence... ");

    llama_kv_block_table table;

    table.append_block(0, 10);
    table.append_block(0, 20);
    table.append_block(1, 30);

    TEST_ASSERT(table.has_sequence(0));
    TEST_ASSERT(table.has_sequence(1));

    table.clear_sequence(0);

    TEST_ASSERT(!table.has_sequence(0));
    TEST_ASSERT(table.has_sequence(1));
    TEST_ASSERT(table.get_physical(0, 0) == -1);

    printf("OK\n");
    return true;
}

static bool test_table_truncate() {
    printf("  test_table_truncate... ");

    llama_kv_block_table table;

    table.append_block(0, 10);
    table.append_block(0, 20);
    table.append_block(0, 30);
    table.append_block(0, 40);

    auto removed = table.truncate_sequence(0, 2);

    TEST_ASSERT(removed.size() == 2);
    TEST_ASSERT(removed[0] == 40);  // LIFO order
    TEST_ASSERT(removed[1] == 30);

    TEST_ASSERT(table.get_sequence_n_blocks(0) == 2);
    TEST_ASSERT(table.get_physical(0, 0) == 10);
    TEST_ASSERT(table.get_physical(0, 1) == 20);

    printf("OK\n");
    return true;
}

static bool test_table_copy_sequence() {
    printf("  test_table_copy_sequence... ");

    llama_kv_block_table table;

    table.append_block(0, 10);
    table.append_block(0, 20);
    table.append_block(0, 30);

    table.copy_sequence(0, 1);

    TEST_ASSERT(table.has_sequence(1));
    TEST_ASSERT(table.get_sequence_n_blocks(1) == 3);
    TEST_ASSERT(table.get_physical(1, 0) == 10);
    TEST_ASSERT(table.get_physical(1, 1) == 20);
    TEST_ASSERT(table.get_physical(1, 2) == 30);

    printf("OK\n");
    return true;
}

static bool test_table_get_sequences() {
    printf("  test_table_get_sequences... ");

    llama_kv_block_table table;

    table.append_block(0, 10);
    table.append_block(5, 20);
    table.append_block(10, 30);

    auto seqs = table.get_sequences();
    TEST_ASSERT(seqs.size() == 3);

    std::set<llama_seq_id> seq_set(seqs.begin(), seqs.end());
    TEST_ASSERT(seq_set.count(0) == 1);
    TEST_ASSERT(seq_set.count(5) == 1);
    TEST_ASSERT(seq_set.count(10) == 1);

    printf("OK\n");
    return true;
}

static bool test_table_total_mappings() {
    printf("  test_table_total_mappings... ");

    llama_kv_block_table table;

    table.append_block(0, 10);
    table.append_block(0, 20);
    table.append_block(1, 30);
    table.append_block(1, 40);
    table.append_block(1, 50);

    TEST_ASSERT(table.total_mappings() == 5);
    TEST_ASSERT(table.n_sequences() == 2);

    printf("OK\n");
    return true;
}

//
// Helper Function Tests
//

static bool test_helper_functions() {
    printf("  test_helper_functions... ");

    const uint32_t tpb = 64;  // tokens per block

    TEST_ASSERT(cell_to_block(0, tpb) == 0);
    TEST_ASSERT(cell_to_block(63, tpb) == 0);
    TEST_ASSERT(cell_to_block(64, tpb) == 1);
    TEST_ASSERT(cell_to_block(128, tpb) == 2);

    TEST_ASSERT(block_to_cell(0, tpb) == 0);
    TEST_ASSERT(block_to_cell(1, tpb) == 64);
    TEST_ASSERT(block_to_cell(2, tpb) == 128);

    TEST_ASSERT(cell_offset_in_block(0, tpb) == 0);
    TEST_ASSERT(cell_offset_in_block(32, tpb) == 32);
    TEST_ASSERT(cell_offset_in_block(64, tpb) == 0);
    TEST_ASSERT(cell_offset_in_block(65, tpb) == 1);

    TEST_ASSERT(tokens_to_blocks(0, tpb) == 0);
    TEST_ASSERT(tokens_to_blocks(1, tpb) == 1);
    TEST_ASSERT(tokens_to_blocks(64, tpb) == 1);
    TEST_ASSERT(tokens_to_blocks(65, tpb) == 2);
    TEST_ASSERT(tokens_to_blocks(128, tpb) == 2);
    TEST_ASSERT(tokens_to_blocks(129, tpb) == 3);

    printf("OK\n");
    return true;
}

//
// Integration Tests
//

static bool test_pool_table_integration() {
    printf("  test_pool_table_integration... ");

    llama_kv_block_pool pool;
    llama_kv_block_table table;

    pool.init(100);

    // Simulate allocating blocks for a sequence
    const llama_seq_id seq_id = 0;
    const uint32_t tokens_per_block = 64;

    // Allocate 3 blocks for sequence 0
    for (int i = 0; i < 3; i++) {
        int32_t phys = pool.allocate();
        TEST_ASSERT(phys >= 0);

        uint32_t logical = table.append_block(seq_id, phys);

        // Update block metadata
        pool.get(phys).seq_id = seq_id;
        pool.get(phys).logical_idx = logical;
        pool.get(phys).n_tokens = tokens_per_block;
    }

    TEST_ASSERT(pool.n_used() == 3);
    TEST_ASSERT(table.get_sequence_n_blocks(seq_id) == 3);

    // Deallocate sequence
    const auto * blocks = table.get_sequence_blocks(seq_id);
    for (int32_t phys : *blocks) {
        pool.deallocate(phys);
    }
    table.clear_sequence(seq_id);

    TEST_ASSERT(pool.n_used() == 0);
    TEST_ASSERT(!table.has_sequence(seq_id));

    printf("OK\n");
    return true;
}

static bool test_cow_simulation() {
    printf("  test_cow_simulation... ");

    llama_kv_block_pool pool;
    llama_kv_block_table table;

    pool.init(100);

    // Simulate copy-on-write for prefix sharing
    // 1. Seq 0 allocates blocks
    int32_t phys0 = pool.allocate();
    int32_t phys1 = pool.allocate();
    table.append_block(0, phys0);
    table.append_block(0, phys1);

    // 2. Seq 1 copies from seq 0 (shares blocks)
    table.copy_sequence(0, 1);
    pool.add_ref(phys0);
    pool.add_ref(phys1);

    TEST_ASSERT(pool.get(phys0).ref_count == 2);
    TEST_ASSERT(pool.get(phys1).ref_count == 2);
    TEST_ASSERT(pool.get(phys0).is_shared());

    // 3. Seq 1 removes its reference
    for (int32_t phys : {phys0, phys1}) {
        pool.deallocate(phys);
    }
    table.clear_sequence(1);

    // Blocks should still exist with ref_count = 1
    TEST_ASSERT(pool.get(phys0).ref_count == 1);
    TEST_ASSERT(!pool.get(phys0).is_shared());
    TEST_ASSERT(!pool.get(phys0).is_free());

    printf("OK\n");
    return true;
}

//
// Main
//

int main() {
    printf("Running KV block tests...\n\n");

    int n_pass = 0;
    int n_fail = 0;

    printf("Block Pool Tests:\n");
    if (test_pool_init()) n_pass++; else n_fail++;
    if (test_pool_allocate_single()) n_pass++; else n_fail++;
    if (test_pool_allocate_all()) n_pass++; else n_fail++;
    if (test_pool_deallocate()) n_pass++; else n_fail++;
    if (test_pool_batch_allocate()) n_pass++; else n_fail++;
    if (test_pool_reference_counting()) n_pass++; else n_fail++;
    if (test_pool_stats()) n_pass++; else n_fail++;
    if (test_pool_clear()) n_pass++; else n_fail++;

    printf("\nBlock Table Tests:\n");
    if (test_table_mapping()) n_pass++; else n_fail++;
    if (test_table_append()) n_pass++; else n_fail++;
    if (test_table_sequence_blocks()) n_pass++; else n_fail++;
    if (test_table_clear_sequence()) n_pass++; else n_fail++;
    if (test_table_truncate()) n_pass++; else n_fail++;
    if (test_table_copy_sequence()) n_pass++; else n_fail++;
    if (test_table_get_sequences()) n_pass++; else n_fail++;
    if (test_table_total_mappings()) n_pass++; else n_fail++;

    printf("\nHelper Function Tests:\n");
    if (test_helper_functions()) n_pass++; else n_fail++;

    printf("\nIntegration Tests:\n");
    if (test_pool_table_integration()) n_pass++; else n_fail++;
    if (test_cow_simulation()) n_pass++; else n_fail++;

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", n_pass, n_fail);

    return n_fail > 0 ? 1 : 0;
}
