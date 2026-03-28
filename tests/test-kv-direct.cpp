#include <cassert>
#include <cstdio>
#include <vector>

// We need the kv_direct_state type. It's nested inside llama_kv_cache.
// Since we can't easily instantiate llama_kv_cache without a model,
// we test the kv_direct_state struct in isolation.
#include "llama-kv-cache.h"

static void test_disabled_state() {
    printf("test_disabled_state...\n");
    llama_kv_cache::kv_direct_state state(0);
    assert(!state.enabled);
    assert(state.budget_tokens == 0);
    printf("  PASS\n");
}

static void test_enabled_state() {
    printf("test_enabled_state...\n");
    llama_kv_cache::kv_direct_state state(100);
    assert(state.enabled);
    assert(state.budget_tokens == 100);
    printf("  PASS\n");
}

static void test_save_and_find() {
    printf("test_save_and_find...\n");
    llama_kv_cache::kv_direct_state state(100);

    const uint32_t n_floats = 512;  // e.g., n_embd * n_layer_kv
    std::vector<float> data(n_floats, 1.0f);

    state.save_residual(0, 0, data.data(), n_floats);
    state.save_residual(5, 0, data.data(), n_floats);
    state.save_residual(10, 1, data.data(), n_floats);

    // Find existing
    auto * cp = state.find(5);
    assert(cp != nullptr);
    assert(cp->pos == 5);
    assert(cp->seq_id == 0);
    assert(cp->data.size() == n_floats);

    // Find non-existing
    assert(state.find(3) == nullptr);
    assert(state.find(99) == nullptr);

    printf("  PASS\n");
}

static void test_remove_range() {
    printf("test_remove_range...\n");
    llama_kv_cache::kv_direct_state state(100);

    std::vector<float> data(128, 0.0f);
    state.save_residual(1, 0, data.data(), 128);
    state.save_residual(5, 0, data.data(), 128);
    state.save_residual(10, 0, data.data(), 128);
    state.save_residual(15, 0, data.data(), 128);

    // Remove [1, 6) — removes pos 1 and 5
    state.remove(1, 6);
    assert(state.find(1) == nullptr);
    assert(state.find(5) == nullptr);
    assert(state.find(10) != nullptr);
    assert(state.find(15) != nullptr);

    printf("  PASS\n");
}

static void test_sorted_insert() {
    printf("test_sorted_insert...\n");
    llama_kv_cache::kv_direct_state state(1000);

    std::vector<float> data(64, 0.0f);

    // Insert out of order
    state.save_residual(10, 0, data.data(), 64);
    state.save_residual(5, 0, data.data(), 64);
    state.save_residual(15, 0, data.data(), 64);
    state.save_residual(1, 0, data.data(), 64);

    // Verify sorted
    assert(state.checkpoints.size() == 4);
    assert(state.checkpoints[0].pos == 1);
    assert(state.checkpoints[1].pos == 5);
    assert(state.checkpoints[2].pos == 10);
    assert(state.checkpoints[3].pos == 15);

    printf("  PASS\n");
}

static void test_data_integrity() {
    printf("test_data_integrity...\n");
    llama_kv_cache::kv_direct_state state(100);

    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    state.save_residual(42, 7, data.data(), 4);

    auto * cp = state.find(42);
    assert(cp != nullptr);
    assert(cp->pos == 42);
    assert(cp->seq_id == 7);
    assert(cp->data.size() == 4);
    assert(cp->data[0] == 1.0f);
    assert(cp->data[1] == 2.0f);
    assert(cp->data[2] == 3.0f);
    assert(cp->data[3] == 4.0f);

    printf("  PASS\n");
}

int main() {
    test_disabled_state();
    test_enabled_state();
    test_save_and_find();
    test_remove_range();
    test_sorted_insert();
    test_data_integrity();

    printf("\nAll KV Direct tests passed!\n");
    return 0;
}
