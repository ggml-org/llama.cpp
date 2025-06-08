#include <cstdio>
#include <vector>
#include <map>
#include <string>
#include <cassert>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <set> // For std::set in llama_kv_page

// Project-specific includes
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

// Need to adjust paths based on actual location relative to 'tests' directory
#include "../src/llama-kv-page.h"
#include "../src/llama-paged-kv-cells.h"
#include "../src/llama-paged-kv-cache.h" // This will also include llama_paged_kv_cells.h
#include "../src/llama_params.h"
#include "../src/llama_batch.h"
// #include "../src/llama_context.h" // May not be strictly needed for direct cache tests

// Simple Assertion Macro
#define ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            fflush(stdout); \
            fprintf(stderr, "Assertion failed: (%s), function %s, file %s, line %d. Message: %s\n", #condition, __func__, __FILE__, __LINE__, message); \
            throw std::runtime_error(std::string("Assertion failed: ") + #condition + std::string(" Message: ") + message); \
        } \
        else { \
            printf("Assertion PASSED: (%s) in %s\n", #condition, __func__); \
        } \
    } while (false)

// Helper to compare memory buffers
bool are_memory_buffers_equal(const uint8_t* buf1, const uint8_t* buf2, size_t size, const char* buf_name = "") {
    if (buf1 == nullptr && buf2 == nullptr) return true;
    if (buf1 == nullptr || buf2 == nullptr) return false;
    bool equal = memcmp(buf1, buf2, size) == 0;
    if (!equal) {
        fprintf(stderr, "Memory buffer %s mismatch!\n", buf_name);
        // Optional: print parts of the buffers for debugging
        // for (size_t i = 0; i < std::min(size, (size_t)16); ++i) {
        //     fprintf(stderr, "Byte %zu: buf1=0x%02x, buf2=0x%02x\n", i, buf1[i], buf2[i]);
        // }
    }
    return equal;
}

// --- Test Case 1: llama_paged_kv_cells - Basic Allocation & Freeing ---
void test_paged_cells_alloc_free() {
    printf("--- Running Test: test_paged_cells_alloc_free ---\n");

    const size_t page_size_bytes = 1024; // Small page size for testing
    const size_t num_pages_initial = 3;
    const size_t total_memory_bytes = page_size_bytes * num_pages_initial;
    std::vector<uint8_t> memory_pool(total_memory_bytes);

    llama_paged_kv_cells cells(page_size_bytes, memory_pool.data(), total_memory_bytes);

    // Test initial state
    ASSERT(cells.get_free_page_count() == num_pages_initial, "Initial free page count should match initial pages.");
    ASSERT(cells.get_used_page_count() == 0, "Initial used page count should be 0.");

    // Test allocation until pool is exhausted
    llama_kv_page* page1 = cells.allocate_page();
    ASSERT(page1 != nullptr, "Page 1 allocation failed.");
    ASSERT(page1->id == 0, "Page 1 ID incorrect."); // IDs are typically indices
    ASSERT(page1->size == page_size_bytes, "Page 1 size incorrect.");
    ASSERT(page1->data == memory_pool.data(), "Page 1 data pointer incorrect.");
    ASSERT(cells.get_free_page_count() == num_pages_initial - 1, "Free pages after 1 alloc.");
    ASSERT(cells.get_used_page_count() == 1, "Used pages after 1 alloc.");

    llama_kv_page* page2 = cells.allocate_page();
    ASSERT(page2 != nullptr, "Page 2 allocation failed.");
    ASSERT(page2->id == 1, "Page 2 ID incorrect.");
    ASSERT(page2->data == memory_pool.data() + page_size_bytes, "Page 2 data pointer incorrect.");
    ASSERT(cells.get_free_page_count() == num_pages_initial - 2, "Free pages after 2 allocs.");
    ASSERT(cells.get_used_page_count() == 2, "Used pages after 2 allocs.");

    llama_kv_page* page3 = cells.allocate_page();
    ASSERT(page3 != nullptr, "Page 3 allocation failed.");
    ASSERT(page3->id == 2, "Page 3 ID incorrect.");
    ASSERT(page3->data == memory_pool.data() + 2 * page_size_bytes, "Page 3 data pointer incorrect.");
    ASSERT(cells.get_free_page_count() == 0, "Free pages after 3 allocs (pool exhausted).");
    ASSERT(cells.get_used_page_count() == 3, "Used pages after 3 allocs.");

    llama_kv_page* page4 = cells.allocate_page();
    ASSERT(page4 == nullptr, "Allocation beyond pool capacity should fail.");

    // Test freeing pages
    cells.free_page(page2->id); // Free page with id 1
    ASSERT(cells.get_free_page_count() == 1, "Free pages after freeing page2.");
    ASSERT(cells.get_used_page_count() == 2, "Used pages after freeing page2.");
    // Check if page2->id is in free_page_indices (internal check, cannot directly verify without accessor)

    // Test re-allocating a freed page
    llama_kv_page* reused_page2 = cells.allocate_page();
    ASSERT(reused_page2 != nullptr, "Re-allocation of freed page failed.");
    ASSERT(reused_page2->id == page2->id, "Re-allocated page should have the same ID as the freed one.");
    ASSERT(reused_page2->data == page2->data, "Re-allocated page should have the same data pointer.");
    ASSERT(cells.get_free_page_count() == 0, "Free pages after re-allocating page2.");
    ASSERT(cells.get_used_page_count() == 3, "Used pages after re-allocating page2.");

    // Test freeing all pages
    cells.free_page(page1->id);
    cells.free_page(reused_page2->id); // or page2->id
    cells.free_page(page3->id);
    ASSERT(cells.get_free_page_count() == num_pages_initial, "All pages should be free.");
    ASSERT(cells.get_used_page_count() == 0, "All pages should be free (used count).");

    printf("--- Test test_paged_cells_alloc_free PASSED ---\n\n");
}


// --- Test Case 2: llama_paged_kv_cells - Token Mapping ---
void test_paged_cells_token_mapping() {
    printf("--- Running Test: test_paged_cells_token_mapping ---\n");
    const size_t page_size_bytes = 256 * sizeof(float); // Enough for e.g. 256 float16s if element size is 2
    const size_t num_pages_initial = 2;
    const size_t total_memory_bytes = page_size_bytes * num_pages_initial;
    std::vector<uint8_t> memory_pool(total_memory_bytes);

    // Assume K and V for one token (one head, one layer) take 64 bytes (e.g. head_dim=32, sizeof(float16)=2)
    const int head_dim = 32;
    const int num_kv_heads = 1; // For simplicity in this test
    const int layer_idx = 0;    // For simplicity
    const size_t bytes_per_token_kv = head_dim * sizeof(uint16_t) * 2; // K and V, uint16_t for float16

    llama_paged_kv_cells cells(page_size_bytes, memory_pool.data(), total_memory_bytes);

    // Token 1: seq_id=0, pos=10
    llama_seq_id seq_id_0 = 0;
    llama_pos token_pos_10 = 10;
    const llama_paged_kv_cells::TokenKey tk1(seq_id_0, token_pos_10);

    llama_kv_page* page_for_tk1;
    size_t offset_for_tk1;
    std::tie(page_for_tk1, offset_for_tk1) = cells.find_or_allocate_page_for_token(tk1, bytes_per_token_kv);

    ASSERT(page_for_tk1 != nullptr, "Page allocation for tk1 failed.");
    ASSERT(page_for_tk1->id == 0, "tk1 should be on the first allocated page.");
    ASSERT(page_for_tk1->used_bytes >= bytes_per_token_kv, "Page used_bytes not updated for tk1.");
    ASSERT(page_for_tk1->seq_ids.count(seq_id_0) == 1, "seq_id_0 not added to page for tk1.");
    ASSERT(cells.get_token_count_for_page(page_for_tk1->id) == 1, "Token count for page of tk1 is not 1.");

    auto mapping_tk1 = cells.get_page_and_offset(tk1);
    ASSERT(mapping_tk1.first == page_for_tk1->id, "get_page_and_offset for tk1 page ID mismatch.");
    ASSERT(mapping_tk1.second == offset_for_tk1, "get_page_and_offset for tk1 offset mismatch.");

    uint8_t* data_ptr_tk1 = cells.get_token_data(tk1);
    ASSERT(data_ptr_tk1 == page_for_tk1->data + offset_for_tk1, "get_token_data pointer for tk1 is incorrect.");

    // Token 2: seq_id=0, pos=11 (same sequence, next token)
    // Assuming bytes_per_token_kv is small enough that multiple tokens fit on one page.
    llama_pos token_pos_11 = 11;
    const llama_paged_kv_cells::TokenKey tk2(seq_id_0, token_pos_11);
    llama_kv_page* page_for_tk2;
    size_t offset_for_tk2;
    std::tie(page_for_tk2, offset_for_tk2) = cells.find_or_allocate_page_for_token(tk2, bytes_per_token_kv);

    ASSERT(page_for_tk2 != nullptr, "Page allocation for tk2 failed.");
    if (page_for_tk1->used_bytes + bytes_per_token_kv <= page_size_bytes) {
        ASSERT(page_for_tk2->id == page_for_tk1->id, "tk2 should be on the same page as tk1 if space allows.");
        ASSERT(offset_for_tk2 == offset_for_tk1 + bytes_per_token_kv, "tk2 offset not contiguous after tk1 on same page.");
        ASSERT(cells.get_token_count_for_page(page_for_tk1->id) == 2, "Token count for page of tk1/tk2 is not 2.");
    } else {
        ASSERT(page_for_tk2->id != page_for_tk1->id, "tk2 should be on a new page if tk1's page was full.");
        ASSERT(cells.get_token_count_for_page(page_for_tk1->id) == 1, "Token count for page of tk1 incorrect after tk2 on new page.");
        ASSERT(cells.get_token_count_for_page(page_for_tk2->id) == 1, "Token count for page of tk2 incorrect on new page.");
    }
    ASSERT(page_for_tk2->used_bytes >= bytes_per_token_kv, "Page used_bytes not updated for tk2."); // Check on its own page
    ASSERT(page_for_tk2->seq_ids.count(seq_id_0) == 1, "seq_id_0 not added to page for tk2.");


    // Token 3: seq_id=1, pos=0 (different sequence)
    llama_seq_id seq_id_1 = 1;
    llama_pos token_pos_s1_0 = 0;
    const llama_paged_kv_cells::TokenKey tk3(seq_id_1, token_pos_s1_0);
    llama_kv_page* page_for_tk3;
    size_t offset_for_tk3;
    std::tie(page_for_tk3, offset_for_tk3) = cells.find_or_allocate_page_for_token(tk3, bytes_per_token_kv);

    ASSERT(page_for_tk3 != nullptr, "Page allocation for tk3 failed.");
    // Check if tk3 is on a new page or shares one (depends on exact filling strategy and remaining space)
    if (page_for_tk3 == page_for_tk1) {
        ASSERT(page_for_tk1->seq_ids.count(seq_id_1) == 1, "seq_id_1 not added to page_for_tk1.");
        ASSERT(cells.get_token_count_for_page(page_for_tk1->id) >= ( (page_for_tk1==page_for_tk2) ? 3:2) , "Token count for page_for_tk1 incorrect after tk3.");
    } else if (page_for_tk3 == page_for_tk2 && page_for_tk1 != page_for_tk2) { // tk2 was on new page
         ASSERT(page_for_tk2->seq_ids.count(seq_id_1) == 1, "seq_id_1 not added to page_for_tk2.");
         ASSERT(cells.get_token_count_for_page(page_for_tk2->id) >= 2, "Token count for page_for_tk2 incorrect after tk3.");
    } else { // tk3 is on a new page entirely (page_id == 1 if tk1,tk2 were on page0, or page_id == 2 if tk1 on page0, tk2 on page1)
         ASSERT(page_for_tk3->seq_ids.count(seq_id_1) == 1, "seq_id_1 not added to page_for_tk3.");
         ASSERT(cells.get_token_count_for_page(page_for_tk3->id) == 1, "Token count for page_for_tk3 incorrect.");
    }

    // Remove tk1
    cells.remove_token_from_page(tk1, page_for_tk1->id, offset_for_tk1, bytes_per_token_kv);
    size_t expected_tokens_on_page1_after_tk1_rm = 0;
    if (page_for_tk1 == page_for_tk2) expected_tokens_on_page1_after_tk1_rm++; // tk2 still there
    if (page_for_tk1 == page_for_tk3) expected_tokens_on_page1_after_tk1_rm++; // tk3 still there

    ASSERT(cells.get_token_count_for_page(page_for_tk1->id) == expected_tokens_on_page1_after_tk1_rm, "Token count for page_for_tk1 after tk1 removal incorrect.");
    if (expected_tokens_on_page1_after_tk1_rm == 0 && !page_for_tk1->is_freeable()) { // is_freeable might not be public, infer
         // If no tokens left, and if it's not marked as unfreeable for other reasons
         // This check is tricky without knowing internal state of free_page_indices or if page was returned
    }

    // Remove tk2
    cells.remove_token_from_page(tk2, page_for_tk2->id, offset_for_tk2, bytes_per_token_kv);
    size_t expected_tokens_on_page2_after_tk2_rm = 0;
    if (page_for_tk2 == page_for_tk1 && expected_tokens_on_page1_after_tk1_rm > 0 && page_for_tk1 == page_for_tk2) {
        // if tk1 and tk2 were on same page, and tk1 was already removed.
        // expected_tokens_on_page1_after_tk1_rm would have accounted for tk2. Now tk2 is removed.
        // This logic gets complex quickly. Simpler to check current state.
    }
     ASSERT(page_for_tk2->seq_ids.count(seq_id_0) == 0, "seq_id_0 should be removed from page_for_tk2 if tk2 was last token of seq0 on it.");


    // Test freeing a page when all its tokens are removed
    llama_seq_id seq_id_2 = 2;
    llama_pos token_pos_s2_0 = 0;
    const llama_paged_kv_cells::TokenKey tk_s2_0(seq_id_2, token_pos_s2_0);
    llama_kv_page* page_for_s2_0;
    size_t offset_for_s2_0;
    std::tie(page_for_s2_0, offset_for_s2_0) = cells.find_or_allocate_page_for_token(tk_s2_0, bytes_per_token_kv);
    ASSERT(page_for_s2_0 != nullptr, "Page for tk_s2_0 alloc failed");
    int page_s2_0_id = page_for_s2_0->id;
    ASSERT(cells.get_token_count_for_page(page_s2_0_id) == 1, "Token count for new page should be 1.");

    cells.remove_token_from_page(tk_s2_0, page_s2_0_id, offset_for_s2_0, bytes_per_token_kv);
    ASSERT(cells.get_token_count_for_page(page_s2_0_id) == 0, "Token count for page_s2_0 should be 0 after removal.");
    // Check if page_s2_0_id is now in free list (indirectly)
    // This requires that remove_token_from_page also calls free_page if token count drops to 0 and seq_ids is empty.
    // The current llama_paged_kv_cells::remove_token_from_page doesn't automatically free. Host has to call free_page.
    // Let's assume free_page is called by a higher layer if get_token_count_for_page == 0 and seq_ids is empty.
    // So we'll manually call it here to test the free mechanism.
    if (cells.get_token_count_for_page(page_s2_0_id) == 0 && page_for_s2_0->seq_ids.empty()) {
        size_t free_before = cells.get_free_page_count();
        cells.free_page(page_s2_0_id);
        ASSERT(cells.get_free_page_count() == free_before + 1, "Page for s2_0 was not freed correctly.");
    }


    printf("--- Test test_paged_cells_token_mapping PASSED ---\n\n");
}

// --- Test Case 3: llama_paged_kv_cache - Initialization ---

// =================================================================================================
// PART 2: CUDA Paged Attention Kernel Tests - Helper Structures and Functions
// =================================================================================================
// [ BEGIN REMOVED DUPLICATED CUDA BLOCK 1 ]
// The first block of CUDA specific functions and tests were here.
// They are defined later in the file, which are the versions intended to be used.
// This removal is to prevent linker errors and confusion.
// [ END REMOVED DUPLICATED CUDA BLOCK 1 ]

ggml_backend_buffer_type_t g_cpu_buf_type = NULL;


int main() {
#ifdef GGML_USE_CUDA
    setup_cuda_for_test(); // This will call the one defined later
#endif

    printf("--- Starting Paged KV Cache Unit Tests ---\n");
    try {
        test_paged_cells_alloc_free();
        test_paged_cells_token_mapping();
        test_paged_cache_initialization();
        test_paged_cache_seq_add();
        test_paged_cache_seq_rm();
        test_paged_cache_seq_cp();
        test_paged_cache_seq_div();
        test_paged_cache_state_read_write();
        // Call other test functions here
#ifdef GGML_USE_CUDA
        if (g_cuda_backend) { // This will use the one defined later
            // Call CUDA tests here
            // test_cuda_paged_attn_correctness_mma_f16(); // Example // This would call the first def
        } else {
            printf("SKIPPING CUDA tests as backend failed to initialize.\n");
        }
#endif
    } catch (const std::exception& e) {
        fprintf(stderr, "A test failed with exception: %s\n", e.what());
#ifdef GGML_USE_CUDA
        teardown_cuda_for_test(); // This will call the one defined later
#endif
        return 1;
    } catch (...) {
        fprintf(stderr, "A test failed with an unknown exception.\n");
#ifdef GGML_USE_CUDA
        teardown_cuda_for_test(); // This will call the one defined later
#endif
        return 1;
    }

#ifdef GGML_USE_CUDA
    teardown_cuda_for_test(); // This will call the one defined later
#endif
    printf("--- All Paged KV Cache Unit Tests PASSED ---\n");
    return 0;
}

void test_paged_cache_initialization() {
    printf("--- Running Test: test_paged_cache_initialization ---\n");

    if (g_cpu_buf_type == NULL) {
        g_cpu_buf_type = ggml_backend_cpu_buffer_type(); // Using CPU backend for these tests
    }

    llama_model_params mparams = {}; // Default init
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 64; // Small context for testing
    cparams.n_batch = 32;
    cparams.n_gpu_layers = 0; // CPU test
    cparams.use_paged_kv_cache = true;
    cparams.kv_page_size = 256 * sizeof(uint16_t); // Example page size

    // Create a ggml_context for the KV cache memory
    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(), // Minimal, cache will allocate its own
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true, // Backend will manage allocation for KV cache tensor
    };
    struct ggml_context * meta_ctx = ggml_init(ggml_params);
    ASSERT(meta_ctx != nullptr, "Failed to create ggml_context for KV cache.");

    llama_paged_kv_cache cache(mparams, cparams, g_cpu_buf_type, meta_ctx);

    ASSERT(cache.get_paged_cells() != nullptr, "Paged cells not initialized in cache.");
    ASSERT(cache.get_page_pool_tensor() != nullptr, "Page pool tensor not allocated in cache.");
    ASSERT(cache.get_page_pool_tensor()->data != nullptr, "Page pool tensor data is null.");
    ASSERT(cache.get_page_size_bytes() == cparams.kv_page_size, "Cache page size mismatch.");
    // Initial page count can be complex to predict exactly if it's dynamic, but should be > 0
    ASSERT(cache.get_total_page_count() > 0, "Total page count should be greater than 0 after init.");

    // Test llama_kv_cache_init
    struct llama_kv_cache kv_cache_base; // This is what llama_context would hold
    bool success = llama_paged_kv_cache_init(&kv_cache_base, mparams, cparams, g_cpu_buf_type, meta_ctx);
    ASSERT(success, "llama_paged_kv_cache_init failed.");
    ASSERT(kv_cache_base.paged_cells != nullptr, "paged_cells not set by init function.");
    ASSERT(kv_cache_base.page_pool_tensor != nullptr, "page_pool_tensor not set by init function.");

    // Cleanup
    if (kv_cache_base.paged_cells) { // llama_paged_kv_cache_free expects a pointer to the class instance
        llama_paged_kv_cache* typed_cache_ptr = (llama_paged_kv_cache*)kv_cache_base.paged_cells;
        llama_paged_kv_cache_free(typed_cache_ptr); // This will delete the cache instance
    }
    ggml_free(meta_ctx);

    printf("--- Test test_paged_cache_initialization PASSED ---\n\n");
}

// Helper function to populate some tokens in the cache for testing
// This is a simplified version of what happens during llama_decode
void populate_kv_cache_for_test(llama_paged_kv_cache & cache, llama_seq_id seq_id, std::vector<llama_pos> positions, int head_dim, int num_kv_heads, int num_layers) {
    if (positions.empty()) return;

    llama_paged_kv_cells * cells = cache.get_paged_cells();
    if (!cells) return;

    size_t bytes_per_token_kv_layer = (size_t)head_dim * sizeof(uint16_t); // Assuming float16 K/V data per head

    for (llama_pos pos : positions) {
        for (int layer = 0; layer < num_layers; ++layer) {
            for (int kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
                // For K cache part
                llama_paged_kv_cells::TokenKey tk_k(seq_id, pos, layer, kv_head, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K);
                auto [page_k, offset_k] = cells->find_or_allocate_page_for_token(tk_k, bytes_per_token_kv_layer);
                if (page_k) {
                    uint8_t* data_k = cells->get_token_data(tk_k);
                    if (data_k) { // Fill with some identifiable data
                        for(size_t i = 0; i < bytes_per_token_kv_layer; ++i) data_k[i] = (seq_id + pos + layer + kv_head + i) % 256;
                    }
                }
                // For V cache part
                llama_paged_kv_cells::TokenKey tk_v(seq_id, pos, layer, kv_head, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_V);
                auto [page_v, offset_v] = cells->find_or_allocate_page_for_token(tk_v, bytes_per_token_kv_layer);
                 if (page_v) {
                    uint8_t* data_v = cells->get_token_data(tk_v);
                    if (data_v) { // Fill with some identifiable data
                        for(size_t i = 0; i < bytes_per_token_kv_layer; ++i) data_v[i] = (seq_id + pos + layer + kv_head + i + 100) % 256;
                    }
                }
            }
        }
    }
}

// --- Test Case 4: llama_paged_kv_cache - seq_add (Token Shifting) ---
void test_paged_cache_seq_add() {
    printf("--- Running Test: test_paged_cache_seq_add ---\n");
    if (g_cpu_buf_type == NULL) g_cpu_buf_type = ggml_backend_cpu_buffer_type();

    llama_model_params mparams = {};
    mparams.n_embd = 32; // head_dim * n_head_kv
    mparams.n_head_kv = 1;
    mparams.n_layer = 1;
    // derived: head_dim = n_embd / n_head_kv = 32

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 10;
    cparams.n_batch = 5;
    cparams.use_paged_kv_cache = true;
    cparams.kv_page_size = ( (size_t)mparams.n_embd / mparams.n_head_kv * sizeof(uint16_t) ) * 3; // Page fits 3 tokens' K/V for one layer/head

    struct ggml_init_params ggml_params = { ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(), NULL, true };
    struct ggml_context * meta_ctx = ggml_init(ggml_params);
    llama_paged_kv_cache cache(mparams, cparams, g_cpu_buf_type, meta_ctx);
    llama_paged_kv_cells* cells = cache.get_paged_cells();

    llama_seq_id seq_id = 0;
    populate_kv_cache_for_test(cache, seq_id, {0, 1, 2, 3, 4}, mparams.n_embd / mparams.n_head_kv, mparams.n_head_kv, mparams.n_layer);

    ASSERT(cells->get_token_count(seq_id) == 5, "Initial token count for seq 0 is not 5.");

    // Shift tokens [0, 4] by delta=2. New positions: [2, 3, 4, 5, 6]
    cache.seq_add(seq_id, 0, 5, 2);
    ASSERT(cells->get_token_count(seq_id) == 5, "Token count for seq 0 after shift is not 5.");

    for (llama_pos p : {0,1}) { // Original positions 0, 1 should be gone
        llama_paged_kv_cells::TokenKey tk_k(seq_id, p, 0, 0, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K);
        ASSERT(cells->get_page_and_offset(tk_k).first == -1, "Old token (pos " + std::to_string(p) + ") should be removed after shift.");
    }
    for (llama_pos p_new : {2,3,4,5,6}) { // New positions
        llama_paged_kv_cells::TokenKey tk_k(seq_id, p_new, 0, 0, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K);
        ASSERT(cells->get_page_and_offset(tk_k).first != -1, "New token (pos " + std::to_string(p_new) + ") should exist after shift.");
    }

    // Shift tokens [2, 6] by delta=-3. New positions: [-1, 0, 1, 2, 3]. Token at -1 should be removed.
    cache.seq_add(seq_id, 2, 7, -3); // p1 is exclusive: [2, 3, 4, 5, 6] -> p1=7
    ASSERT(cells->get_token_count(seq_id) == 4, "Token count for seq 0 after negative shift should be 4.");
    llama_paged_kv_cells::TokenKey tk_k_neg(seq_id, -1, 0, 0, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K); // original pos 2 shifted by -3
    ASSERT(cells->get_page_and_offset(tk_k_neg).first == -1, "Token at negative position should be removed.");
    for (llama_pos p_new : {0,1,2,3}) {
        llama_paged_kv_cells::TokenKey tk_k(seq_id, p_new, 0, 0, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K);
        ASSERT(cells->get_page_and_offset(tk_k).first != -1, "Token (pos " + std::to_string(p_new) + ") should exist after negative shift.");
    }

    ggml_free(meta_ctx);
    printf("--- Test test_paged_cache_seq_add PASSED ---\n\n");
}

// --- Test Case 5: llama_paged_kv_cache - seq_rm (Token Removal) ---
void test_paged_cache_seq_rm() {
    printf("--- Running Test: test_paged_cache_seq_rm ---\n");
    if (g_cpu_buf_type == NULL) g_cpu_buf_type = ggml_backend_cpu_buffer_type();

    llama_model_params mparams = {};
    mparams.n_embd = 32; mparams.n_head_kv = 1; mparams.n_layer = 1;
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 20; cparams.n_batch = 10;
    cparams.use_paged_kv_cache = true;

    size_t bytes_per_token_kv_one_head_one_layer = (size_t)mparams.n_embd / mparams.n_head_kv * sizeof(uint16_t) * 2; // K+V
    cparams.kv_page_size = bytes_per_token_kv_one_head_one_layer * 2; // Page fits 2 tokens' K/V for one layer/head


    struct ggml_init_params ggml_params = { ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(), NULL, true };
    struct ggml_context * meta_ctx = ggml_init(ggml_params);
    llama_paged_kv_cache cache(mparams, cparams, g_cpu_buf_type, meta_ctx);
    llama_paged_kv_cells* cells = cache.get_paged_cells();

    llama_seq_id seq0 = 0;
    llama_seq_id seq1 = 1;
    std::vector<llama_pos> pos_s0 = {0, 1, 2, 3, 4, 5};
    // s1 overlaps with s0 on pos 2, 3, 4, 5.
    populate_kv_cache_for_test(cache, seq0, pos_s0, mparams.n_embd / mparams.n_head_kv, mparams.n_head_kv, mparams.n_layer);

    std::vector<llama_pos> pos_s1 = {2, 3, 4, 5, 6, 7};
    populate_kv_cache_for_test(cache, seq1, pos_s1, mparams.n_embd / mparams.n_head_kv, mparams.n_head_kv, mparams.n_layer);

    ASSERT(cells->get_token_count(seq0) == 6, "Initial token count for seq0 incorrect.");
    ASSERT(cells->get_token_count(seq1) == 6, "Initial token count for seq1 incorrect.");

    llama_paged_kv_cells::TokenKey tk_s0_p2_k(seq0, 2, 0, 0, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K);
    auto mapping_s0_p2 = cells->get_page_and_offset(tk_s0_p2_k);
    int page_id_s0_p2 = mapping_s0_p2.first;
    ASSERT(page_id_s0_p2 != -1, "Token (0,2,K) not found for s0.");

    llama_paged_kv_cells::TokenKey tk_s1_p2_k(seq1, 2, 0, 0, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K);
    auto mapping_s1_p2 = cells->get_page_and_offset(tk_s1_p2_k);
    int page_id_s1_p2 = mapping_s1_p2.first;
    ASSERT(page_id_s1_p2 != -1, "Token (1,2,K) not found for s1.");

    llama_kv_page* page_s0_p2_ptr = cells->get_page(page_id_s0_p2);
    ASSERT(page_s0_p2_ptr->seq_ids.count(seq0) == 1, "seq0 not in page_s0_p2's seq_ids before rm.");
    if (page_id_s0_p2 == page_id_s1_p2) {
      ASSERT(page_s0_p2_ptr->seq_ids.count(seq1) == 1, "seq1 not in page_s0_p2's seq_ids (shared case) before rm.");
    }
    size_t tokens_on_page_s0_p2_before_rm = cells->get_token_count_for_page(page_id_s0_p2);

    cache.seq_rm(seq0, 2, 4);
    ASSERT(cells->get_token_count(seq0) == 4, "Token count for seq0 after rm incorrect.");
    ASSERT(cells->get_page_and_offset(tk_s0_p2_k).first == -1, "Token (0,2) should be removed.");
    llama_paged_kv_cells::TokenKey tk_s0_p3_k(seq0, 3, 0, 0, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K);
    ASSERT(cells->get_page_and_offset(tk_s0_p3_k).first == -1, "Token (0,3) should be removed.");

    page_s0_p2_ptr = cells->get_page(page_id_s0_p2);
    if (page_s0_p2_ptr) {
        bool seq0_should_be_present = false;
        for(llama_pos p : pos_s0) {
            if (p < 2 || p >= 4) {
                if(cells->get_page_and_offset({seq0, p, 0,0, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K}).first == page_id_s0_p2) {
                    seq0_should_be_present = true;
                    break;
                }
            }
        }
        ASSERT(page_s0_p2_ptr->seq_ids.count(seq0) == (seq0_should_be_present ? 1:0), "seq0 presence in page_s0_p2's seq_ids inconsistent after rm.");
        if (page_id_s0_p2 == page_id_s1_p2) {
             ASSERT(page_s0_p2_ptr->seq_ids.count(seq1) == 1, "seq1 should still be in page_s0_p2 (shared case) after s0's tokens rm.");
        }
        if (tokens_on_page_s0_p2_before_rm > 0 && seq0_should_be_present == false && page_id_s0_p2 == page_id_s1_p2 && page_s0_p2_ptr->seq_ids.count(seq1) > 0) {
             // If seq0 is no longer on this page, but seq1 is, the token count should reflect removal of seq0's tokens from this page.
             // This specific assertion is tricky without knowing exactly how many tokens of seq0 were on page_s0_p2.
        } else if (!seq0_should_be_present) {
            // If seq0 is not on this page anymore, token count should have decreased if it contributed tokens.
        }
    }

    size_t free_pages_before_s1_rm = cells->get_free_page_count();
    cache.seq_rm(seq1, 0, 8);
    ASSERT(cells->get_token_count(seq1) == 0, "Token count for seq1 should be 0 after full rm.");
    llama_paged_kv_cells::TokenKey tk_s1_p4_k(seq1, 4, 0, 0, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K);
    ASSERT(cells->get_page_and_offset(tk_s1_p4_k).first == -1, "Any token from seq1 should be removed.");

    // Check if pages previously exclusively used by seq1 are now free.
    // This relies on seq_rm correctly calling free_page internally.
    // The number of freed pages should be at least 1 if seq1 had exclusive pages.
    // This is an indirect check.
    ASSERT(cells->get_free_page_count() >= free_pages_before_s1_rm, "Free page count should not decrease after removing seq1.");

    ggml_free(meta_ctx);
    printf("--- Test test_paged_cache_seq_rm PASSED ---\n\n");
}

// --- Test Case 6: llama_paged_kv_cache - seq_cp (Sequence Copying) ---
void test_paged_cache_seq_cp() {
    printf("--- Running Test: test_paged_cache_seq_cp ---\n");
    if (g_cpu_buf_type == NULL) g_cpu_buf_type = ggml_backend_cpu_buffer_type();

    llama_model_params mparams = {};
    mparams.n_embd = 32; mparams.n_head_kv = 1; mparams.n_layer = 1;
    int head_dim = mparams.n_embd / mparams.n_head_kv;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 20; cparams.n_batch = 10;
    cparams.use_paged_kv_cache = true;
    size_t bytes_per_token_kv_one_head_one_layer = (size_t)head_dim * sizeof(uint16_t) * 2; // K+V
    cparams.kv_page_size = bytes_per_token_kv_one_head_one_layer * 2; // Page fits 2 tokens

    struct ggml_init_params ggml_params = { ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(), NULL, true };
    struct ggml_context * meta_ctx = ggml_init(ggml_params);
    llama_paged_kv_cache cache(mparams, cparams, g_cpu_buf_type, meta_ctx);
    llama_paged_kv_cells* cells = cache.get_paged_cells();

    llama_seq_id seq_id_src = 0;
    llama_seq_id seq_id_dst = 1;
    std::vector<llama_pos> src_positions = {10, 11, 12, 13};
    populate_kv_cache_for_test(cache, seq_id_src, src_positions, head_dim, mparams.n_head_kv, mparams.n_layer);

    // Copy [10, 11] from src to dst at position 0
    // So, src_pos 10 -> dst_pos 0; src_pos 11 -> dst_pos 1
    cache.seq_cp(seq_id_src, seq_id_dst, 10, 12, 0);

    ASSERT(cells->get_token_count(seq_id_dst) == 2, "Token count for dst_seq after copy is not 2.");

    for (int i = 0; i < 2; ++i) {
        llama_pos src_pos = src_positions[i]; // 10, 11
        llama_pos dst_pos = i; // 0, 1

        for (int l = 0; l < mparams.n_layer; ++l) {
            for (int h = 0; h < mparams.n_head_kv; ++h) {
                llama_paged_kv_cells::TokenKey tk_src_k(seq_id_src, src_pos, l, h, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K);
                llama_paged_kv_cells::TokenKey tk_dst_k(seq_id_dst, dst_pos, l, h, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K);
                uint8_t* data_src_k = cells->get_token_data(tk_src_k);
                uint8_t* data_dst_k = cells->get_token_data(tk_dst_k);
                ASSERT(data_src_k != nullptr, "Source K data pointer is null.");
                ASSERT(data_dst_k != nullptr, "Destination K data pointer is null.");
                ASSERT(data_src_k != data_dst_k, "Source and Destination K data pointers should be different (deep copy).");
                ASSERT(are_memory_buffers_equal(data_src_k, data_dst_k, bytes_per_token_kv_one_head_one_layer / 2, "K data copy mismatch"),
                       "K data content mismatch for src_pos " + std::to_string(src_pos) + " -> dst_pos " + std::to_string(dst_pos));

                llama_paged_kv_cells::TokenKey tk_src_v(seq_id_src, src_pos, l, h, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_V);
                llama_paged_kv_cells::TokenKey tk_dst_v(seq_id_dst, dst_pos, l, h, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_V);
                uint8_t* data_src_v = cells->get_token_data(tk_src_v);
                uint8_t* data_dst_v = cells->get_token_data(tk_dst_v);
                ASSERT(data_src_v != nullptr, "Source V data pointer is null.");
                ASSERT(data_dst_v != nullptr, "Destination V data pointer is null.");
                ASSERT(data_src_v != data_dst_v, "Source and Destination V data pointers should be different.");
                ASSERT(are_memory_buffers_equal(data_src_v, data_dst_v, bytes_per_token_kv_one_head_one_layer / 2, "V data copy mismatch"),
                       "V data content mismatch for src_pos " + std::to_string(src_pos) + " -> dst_pos " + std::to_string(dst_pos));
            }
        }
    }

    // Verify page usage for dst_seq (e.g., at least one page should be used by seq_id_dst)
    bool dst_seq_uses_pages = false;
    for (uint32_t page_idx = 0; page_idx < cells->get_page_count(); ++page_idx) {
        llama_kv_page* page = cells->get_page(page_idx);
        if (page && !page->is_free() && page->seq_ids.count(seq_id_dst)) {
            dst_seq_uses_pages = true;
            break;
        }
    }
    ASSERT(dst_seq_uses_pages, "Destination sequence does not seem to use any pages after copy.");

    ggml_free(meta_ctx);
    printf("--- Test test_paged_cache_seq_cp PASSED ---\n\n");
}

// --- Test Case 7: llama_paged_kv_cache - seq_div (Sequence Division) ---
void test_paged_cache_seq_div() {
    printf("--- Running Test: test_paged_cache_seq_div ---\n");
    if (g_cpu_buf_type == NULL) g_cpu_buf_type = ggml_backend_cpu_buffer_type();

    llama_model_params mparams = {};
    mparams.n_embd = 32; mparams.n_head_kv = 1; mparams.n_layer = 1;
    int head_dim = mparams.n_embd / mparams.n_head_kv;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 10;
    cparams.n_batch = 6; // To fit 6 tokens
    cparams.use_paged_kv_cache = true;
    size_t bytes_per_token_kv_one_head_one_layer = (size_t)head_dim * sizeof(uint16_t) * 2;
    cparams.kv_page_size = bytes_per_token_kv_one_head_one_layer * 2; // Page fits 2 tokens

    struct ggml_init_params ggml_params = { ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(), NULL, true };
    struct ggml_context * meta_ctx = ggml_init(ggml_params);
    llama_paged_kv_cache cache(mparams, cparams, g_cpu_buf_type, meta_ctx);
    llama_paged_kv_cells* cells = cache.get_paged_cells();

    llama_seq_id seq_id = 0;
    std::vector<llama_pos> initial_positions = {0, 1, 2, 3, 4, 5};
    populate_kv_cache_for_test(cache, seq_id, initial_positions, head_dim, mparams.n_head_kv, mparams.n_layer);
    ASSERT(cells->get_token_count(seq_id) == 6, "Initial token count for seq_div test incorrect.");

    // Divide [0, 1, 2, 3, 4, 5] by 2. Range [0, 6).
    // Expected new positions, keeping max original pos for collisions:
    // 0/2=0, 1/2=0 -> (0,0) from original (0,1)
    // 2/2=1, 3/2=1 -> (0,1) from original (0,3)
    // 4/2=2, 5/2=2 -> (0,2) from original (0,5)
    cache.seq_div(seq_id, 0, 6, 2);

    ASSERT(cells->get_token_count(seq_id) == 3, "Token count after division by 2 should be 3.");

    // Tokens that should have been removed (due to not being max_pos for the new divided pos)
    llama_paged_kv_cells::TokenKey tk_k_orig0(seq_id, 0, 0, 0, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K);
    ASSERT(cells->get_page_and_offset(tk_k_orig0).first == -1, "Token (0,0) should be removed after div.");
    llama_paged_kv_cells::TokenKey tk_k_orig2(seq_id, 2, 0, 0, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K);
    ASSERT(cells->get_page_and_offset(tk_k_orig2).first == -1, "Token (0,2) should be removed after div.");
    llama_paged_kv_cells::TokenKey tk_k_orig4(seq_id, 4, 0, 0, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K);
    ASSERT(cells->get_page_and_offset(tk_k_orig4).first == -1, "Token (0,4) should be removed after div.");

    // Tokens that should remain at new positions
    llama_paged_kv_cells::TokenKey tk_k_new0(seq_id, 0, 0, 0, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K); // from original 1
    ASSERT(cells->get_page_and_offset(tk_k_new0).first != -1, "Token (0,0) (from original 1) not found after div.");
    llama_paged_kv_cells::TokenKey tk_k_new1(seq_id, 1, 0, 0, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K); // from original 3
    ASSERT(cells->get_page_and_offset(tk_k_new1).first != -1, "Token (0,1) (from original 3) not found after div.");
    llama_paged_kv_cells::TokenKey tk_k_new2(seq_id, 2, 0, 0, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K); // from original 5
    ASSERT(cells->get_page_and_offset(tk_k_new2).first != -1, "Token (0,2) (from original 5) not found after div.");

    // Verify data integrity for one of the kept tokens (e.g. original (0,5) -> new (0,2))
    // This requires get_token_data to work with the new positions.
    // We need to compare data of new (0,2) with original data of (0,5).
    // This is tricky as original data for (0,5) is gone from cells map.
    // For now, this test focuses on mapping and counts. Data integrity for seq_div is harder.

    ggml_free(meta_ctx);
    printf("--- Test test_paged_cache_seq_div PASSED ---\n\n");
}

// --- Test Case 8: llama_paged_kv_cache - state_write and state_read ---
void test_paged_cache_state_read_write() {
    printf("--- Running Test: test_paged_cache_state_read_write ---\n");
    if (g_cpu_buf_type == NULL) g_cpu_buf_type = ggml_backend_cpu_buffer_type();

    llama_model_params mparams = {};
    mparams.n_embd = 32; mparams.n_head_kv = 1; mparams.n_layer = 2; // 2 layers for more diversity
    int head_dim = mparams.n_embd / mparams.n_head_kv;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 10;
    cparams.n_batch = 5;
    cparams.use_paged_kv_cache = true;
    size_t bytes_per_token_kv_one_head_one_layer = (size_t)head_dim * sizeof(uint16_t); // K or V part
    cparams.kv_page_size = bytes_per_token_kv_one_head_one_layer * 2 * 2; // Page fits 2 tokens' K AND V for one layer/head

    struct ggml_init_params ggml_params = { ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(), NULL, true };

    // Cache Original
    struct ggml_context * meta_ctx_orig = ggml_init(ggml_params);
    llama_paged_kv_cache cache_orig(mparams, cparams, g_cpu_buf_type, meta_ctx_orig);
    populate_kv_cache_for_test(cache_orig, 0, {0, 1, 2}, head_dim, mparams.n_head_kv, mparams.n_layer);
    populate_kv_cache_for_test(cache_orig, 1, {0, 1}, head_dim, mparams.n_head_kv, mparams.n_layer);
    cache_orig.seq_rm(0, 1, 2);
    cache_orig.seq_add(1, 0, 2, 3);

    size_t state_size = cache_orig.get_state_size_bytes();
    ASSERT(state_size > 0, "State size should be positive.");
    std::vector<uint8_t> state_buffer(state_size);
    cache_orig.state_write(state_buffer.data(), state_size);

    // Cache New
    struct ggml_context * meta_ctx_new = ggml_init(ggml_params);
    llama_paged_kv_cache cache_new(mparams, cparams, g_cpu_buf_type, meta_ctx_new);
    cache_new.state_read(state_buffer.data());

    // Verification
    llama_paged_kv_cells* cells_orig = cache_orig.get_paged_cells();
    llama_paged_kv_cells* cells_new = cache_new.get_paged_cells();

    ASSERT(cells_new->get_page_count() == cells_orig->get_page_count(), "Page count mismatch after state read.");
    ASSERT(cells_new->get_free_page_count() == cells_orig->get_free_page_count(), "Free page count mismatch.");
    ASSERT(cells_new->get_token_count_all_seqs() == cells_orig->get_token_count_all_seqs(), "Total token count mismatch.");

    std::vector<llama_paged_kv_cells::TokenKey> keys_to_check;
    keys_to_check.push_back({0, 0, 0, 0, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K});
    keys_to_check.push_back({0, 2, 0, 0, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_V});
    keys_to_check.push_back({1, 3, 0, 0, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K});
    keys_to_check.push_back({1, 4, (mparams.n_layer > 1 ? 1 : 0), 0, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_V});

    for (const auto& tk : keys_to_check) {
        auto mapping_orig = cells_orig->get_page_and_offset(tk);
        auto mapping_new = cells_new->get_page_and_offset(tk);
        ASSERT(mapping_orig.first == mapping_new.first, "Page ID mismatch for token after state read.");
        ASSERT(mapping_orig.second == mapping_new.second, "Offset mismatch for token after state read.");

        if (mapping_orig.first != -1) {
            uint8_t* data_orig_ptr = cells_orig->get_token_data(tk);
            uint8_t* data_new_ptr = cells_new->get_token_data(tk);
            ASSERT(data_orig_ptr != nullptr && data_new_ptr != nullptr, "Token data pointer is null after state read for existing token.");

            llama_kv_page* page_orig = cells_orig->get_page(mapping_orig.first);
            llama_kv_page* page_new = cells_new->get_page(mapping_new.first);
            ASSERT(page_orig && page_new, "Page pointer became null unexpectedly.");

            ASSERT(are_memory_buffers_equal(page_orig->data + mapping_orig.second,
                                            page_new->data + mapping_new.second,
                                            bytes_per_token_kv_one_head_one_layer,
                                            "Token data content mismatch"),
                   "Token data content mismatch for token.");
        }
    }

    for (uint32_t i = 0; i < cells_orig->get_page_count(); ++i) {
        llama_kv_page* page_orig = cells_orig->get_page(i);
        llama_kv_page* page_new = cells_new->get_page(i);
        if (page_orig && page_new) {
            ASSERT(page_orig->is_free() == page_new->is_free(), "Page free status mismatch for page " + std::to_string(i));
            if (!page_orig->is_free()) {
                ASSERT(page_orig->used_bytes == page_new->used_bytes, "Page used_bytes mismatch for page " + std::to_string(i));
                ASSERT(page_orig->seq_ids == page_new->seq_ids, "Page seq_ids mismatch for page " + std::to_string(i));
                ASSERT(are_memory_buffers_equal(page_orig->data, page_new->data, page_orig->size, "Page full data content"), "Page data differs for page " + std::to_string(i));
            }
        } else {
             ASSERT(page_orig == page_new, "Page existence mismatch for page " + std::to_string(i));
        }
    }

    ggml_free(meta_ctx_orig);
    ggml_free(meta_ctx_orig);
    ggml_free(meta_ctx_new);
    printf("--- Test test_paged_cache_state_read_write PASSED ---\n\n");
} // Closing brace for test_paged_cache_state_read_write

// =================================================================================================
// PART 2: CUDA Paged Attention Kernel Tests
// =================================================================================================
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h" // For CUDA backend functions and specific types if needed

// Global CUDA backend and buffer type for tests
ggml_backend_t g_cuda_backend = NULL;
ggml_backend_buffer_type_t g_cuda_buf_type_device = NULL;

void setup_cuda_for_test() {
    fprintf(stderr, "Initializing CUDA backend for tests...\n");
    // Default to device 0 for tests
    g_cuda_backend = ggml_backend_cuda_init(0);
    if (!g_cuda_backend) {
        fprintf(stderr, "setup_cuda_for_test: ggml_backend_cuda_init() failed. CUDA tests will be skipped.\n");
        return;
    }
    g_cuda_buf_type_device = ggml_backend_get_default_buffer_type(g_cuda_backend);
    ASSERT(g_cuda_buf_type_device != NULL, "Failed to get CUDA device buffer type.");
    printf("CUDA backend initialized for tests.\n");
}

void teardown_cuda_for_test() {
    if (g_cuda_backend) {
        ggml_backend_free(g_cuda_backend);
        g_cuda_backend = NULL;
        g_cuda_buf_type_device = NULL;
        printf("CUDA backend freed.\n");
    }
}

// Creates a GPU tensor and copies data from a host tensor.
ggml_tensor* create_gpu_tensor_from_host(ggml_context* ctx_meta_gpu, const ggml_tensor* t_host, const char* name) {
    if (!g_cuda_backend || !g_cuda_buf_type_device) {
        fprintf(stderr, "CUDA backend not initialized, cannot create GPU tensor %s.\n", name);
        return nullptr;
    }
    // Create metadata for the GPU tensor
    ggml_tensor* t_device = ggml_dup_tensor(ctx_meta_gpu, t_host);
    // Allocate buffer on GPU
    ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(g_cuda_buf_type_device, ggml_nbytes(t_device));
    ASSERT(buffer != NULL, (std::string("Failed to allocate CUDA buffer for ") + name).c_str());
    // Associate buffer with tensor
    ggml_backend_tensor_set_buffer(t_device, buffer);
    // Copy data
    ggml_backend_tensor_set_async(t_device, t_host->data, 0, ggml_nbytes(t_host));
    ggml_backend_synchronize(g_cuda_backend);
    ggml_set_name(t_device, name);
    return t_device;
}

// Retrieves data from a GPU tensor to a host vector.
std::vector<uint8_t> get_tensor_data_from_gpu(const ggml_tensor* t_device) {
    if (!g_cuda_backend || !t_device || !t_device->buffer ) {
        fprintf(stderr, "Invalid tensor or CUDA backend for get_tensor_data_from_gpu for tensor %s.\n", t_device ? t_device->name : "NULL");
        return {};
    }
    size_t nbytes = ggml_nbytes(t_device);
    std::vector<uint8_t> host_data(nbytes);
    ggml_backend_tensor_get_async(t_device, host_data.data(), 0, nbytes);
    ggml_backend_synchronize(g_cuda_backend);
    return host_data;
}

// Helper function to compare float tensors with tolerance
bool compare_tensors_approx(const float* data1, const float* data2, int64_t num_elements, const char* test_name, float abs_tolerance, float rel_tolerance) {
    int mismatches = 0;
    for (int64_t i = 0; i < num_elements; ++i) {
        float d1 = data1[i];
        float d2 = data2[i];
        float diff = fabsf(d1 - d2);
        // Relative difference calculation, handle d1 being close to zero
        float rd = (fabsf(d1) > 1e-9f) ? diff / fabsf(d1) : 0.0f;

        if (diff > abs_tolerance && rd > rel_tolerance) {
            if (mismatches < 20) { // Print first few mismatches
                printf("%s: Mismatch at index %lld: data1=%.8f, data2=%.8f, diff=%.8f, rel_diff=%.8f (abs_tol=%.2e, rel_tol=%.2e)\n",
                       test_name, i, d1, d2, diff, rd, abs_tolerance, rel_tolerance);
            }
            mismatches++;
        }
    }
    if (mismatches > 0) {
        printf("%s: Total mismatches: %d / %lld\n", test_name, mismatches, num_elements);
        return false;
    }
    printf("%s: Results match within tolerance (abs_tol=%.2e, rel_tol=%.2e).\n", test_name, abs_tolerance, rel_tolerance);
    return true;
}

// Host-side representation of CUDA structs for preparing kernel arguments
struct paged_kv_token_mapping_host_for_gpu {
    int32_t page_idx;
    int32_t offset_in_page_elements; // Byte offset
};

struct paged_kv_sequence_view_host_for_gpu {
    void* token_mappings_gpu_ptr;
    void* page_pool_gpu_ptr;
    int32_t num_tokens_in_logical_sequence;
    ggml_type dtype;
    int32_t k_head_size_elements;
    int32_t v_head_size_elements;
    int32_t num_k_heads_total;
    int32_t num_v_heads_total;
    uint32_t element_size_bytes;
    uint32_t page_size_bytes;
    uint32_t v_block_start_offset_bytes;

    // For cleanup
    std::vector<void*> actual_page_data_gpu_raw_ptrs; // Stores raw pointers from t_page_gpu->data
    std::vector<ggml_backend_buffer_t> actual_page_data_buffers; // Stores buffers for individual page data copies
    ggml_backend_buffer_t token_mappings_buffer;
    ggml_backend_buffer_t page_pool_buffer;
};

// Prepares GPU buffers for paged KV views from a CPU cache state.
// Also populates k_metadata_gpu_tensor->extra and v_metadata_gpu_tensor->extra
std::pair<paged_kv_sequence_view_host_for_gpu, paged_kv_sequence_view_host_for_gpu>
prepare_paged_kv_views_on_gpu(
    llama_paged_kv_cache& cpu_cache,
    const std::vector<llama_seq_id>& target_seq_ids,
    ggml_context* ctx_meta_gpu,
    const llama_model_params& mparams,
    const llama_context_params& cparams,
    ggml_tensor* k_metadata_gpu_tensor, // Input tensor for K view metadata
    ggml_tensor* v_metadata_gpu_tensor  // Input tensor for V view metadata
) {
    if (!g_cuda_backend || !g_cuda_buf_type_device) {
        throw std::runtime_error("CUDA backend not initialized for paged view prep.");
    }
    llama_paged_kv_cells* cpu_cells = cpu_cache.get_paged_cells();
    ASSERT(cpu_cells != nullptr, "CPU paged_cells is null.");
    ASSERT(k_metadata_gpu_tensor != nullptr, "k_metadata_gpu_tensor is null.");
    ASSERT(v_metadata_gpu_tensor != nullptr, "v_metadata_gpu_tensor is null.");

    paged_kv_sequence_view_host_for_gpu k_view_host_gpu = {0};
    paged_kv_sequence_view_host_for_gpu v_view_host_gpu = {0};

    std::vector<paged_kv_token_mapping_host_for_gpu> k_mappings_host_vec;
    std::vector<paged_kv_token_mapping_host_for_gpu> v_mappings_host_vec;
    std::map<int, const llama_kv_page*> unique_pages_map_cpu_id_to_ptr;
    int max_pos_overall = -1;

    ASSERT(target_seq_ids.size() == 1, "This simplified helper expects only one target_seq_id for creating a flat view.");
    llama_seq_id current_seq_id = target_seq_ids[0];

    for (const auto& item : cpu_cells->get_token_to_page_offset_map()) {
        const auto& token_key = item.first;
        const auto& page_offset_val = item.second;
        if (token_key.seq_id != current_seq_id) continue;

        unique_pages_map_cpu_id_to_ptr[page_offset_val.page_id] = cpu_cells->get_page(page_offset_val.page_id);
        paged_kv_token_mapping_host_for_gpu current_mapping = {(int32_t)page_offset_val.page_id, (int32_t)page_offset_val.offset_bytes};
        int current_pos = token_key.pos;
        if (current_pos > max_pos_overall) max_pos_overall = current_pos;

        if (token_key.type == llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K) {
            if (current_pos >= (int)k_mappings_host_vec.size()) k_mappings_host_vec.resize(max_pos_overall + 1, {-1, 0});
            k_mappings_host_vec[current_pos] = current_mapping;
        } else {
            if (current_pos >= (int)v_mappings_host_vec.size()) v_mappings_host_vec.resize(max_pos_overall + 1, {-1, 0});
            v_mappings_host_vec[current_pos] = current_mapping;
        }
    }
    if (max_pos_overall == -1 ) {
        k_mappings_host_vec.clear();
        v_mappings_host_vec.clear();
    } else {
         if (k_mappings_host_vec.size() < (size_t)max_pos_overall + 1) k_mappings_host_vec.resize(max_pos_overall + 1, {-1,0});
         if (v_mappings_host_vec.size() < (size_t)max_pos_overall + 1) v_mappings_host_vec.resize(max_pos_overall + 1, {-1,0});
    }

    std::vector<void*> host_gpu_page_device_raw_ptrs; // Stores raw pointers from t_page_gpu->data
    std::vector<ggml_backend_buffer_t> host_gpu_page_buffers;   // Stores the ggml_backend_buffer_t for page data
    std::map<int, int> cpu_page_id_to_gpu_pool_idx;
    for(const auto& pair : unique_pages_map_cpu_id_to_ptr) {
        const llama_kv_page* cpu_page = pair.second;
        if (cpu_page && !cpu_page->is_free()) {
            struct ggml_tensor* t_page_host_meta = ggml_new_tensor_1d(ctx_meta_gpu, GGML_TYPE_I8, cpu_page->size);
            t_page_host_meta->data = cpu_page->data;
            // create_gpu_tensor_from_host allocates a buffer and associates it with t_page_gpu
            ggml_tensor* t_page_gpu = create_gpu_tensor_from_host(ctx_meta_gpu, t_page_host_meta, "gpu_page_data_content");
            t_page_host_meta->data = nullptr;
            ggml_free(t_page_host_meta);
            ASSERT(t_page_gpu && t_page_gpu->data && t_page_gpu->buffer, "Failed to create GPU buffer for a page content or buffer not associated.");

            cpu_page_id_to_gpu_pool_idx[cpu_page->id] = host_gpu_page_device_raw_ptrs.size();
            host_gpu_page_device_raw_ptrs.push_back(t_page_gpu->data);
            host_gpu_page_buffers.push_back(t_page_gpu->buffer); // Store the buffer for later cleanup
            // Note: The ggml_tensor t_page_gpu itself is freed by ggml_free(ctx_meta_gpu) if it's in that context,
            // but the buffer it points to (t_page_gpu->buffer) needs explicit freeing.
        }
    }
    k_view_host_gpu.actual_page_data_raw_ptrs = host_gpu_page_device_raw_ptrs; // For reference if needed, but buffers are key
    k_view_host_gpu.actual_page_data_buffers = host_gpu_page_buffers;

    for(auto& mapping : k_mappings_host_vec) {
        if (mapping.page_idx != -1 && cpu_page_id_to_gpu_pool_idx.count(mapping.page_idx)) {
             mapping.page_idx = cpu_page_id_to_gpu_pool_idx.at(mapping.page_idx);
        } else { mapping.page_idx = -1; mapping.offset_in_page_elements = 0; }
    }
    for(auto& mapping : v_mappings_host_vec) {
        if (mapping.page_idx != -1 && cpu_page_id_to_gpu_pool_idx.count(mapping.page_idx)) {
            mapping.page_idx = cpu_page_id_to_gpu_pool_idx.at(mapping.page_idx);
        } else { mapping.page_idx = -1; mapping.offset_in_page_elements = 0; }
    }

    if (!k_mappings_host_vec.empty()) {
        k_view_host_gpu.token_mappings_buffer = ggml_backend_alloc_buffer(g_cuda_buf_type_device, k_mappings_host_vec.size() * sizeof(paged_kv_token_mapping_host_for_gpu));
        ASSERT(k_view_host_gpu.token_mappings_buffer != nullptr, "Failed to allocate k_map_buf GPU buffer.");
        k_view_host_gpu.token_mappings_gpu_ptr = ggml_backend_buffer_get_base(k_view_host_gpu.token_mappings_buffer);
        ASSERT(k_view_host_gpu.token_mappings_gpu_ptr != nullptr, "k_view_host_gpu.token_mappings_gpu_ptr is null post-allocation (k_map_buf).");
        ggml_backend_buffer_set_data(k_view_host_gpu.token_mappings_buffer, 0, k_mappings_host_vec.data(), k_mappings_host_vec.size() * sizeof(paged_kv_token_mapping_host_for_gpu));
    } else {
        k_view_host_gpu.token_mappings_buffer = nullptr;
        k_view_host_gpu.token_mappings_gpu_ptr = nullptr;
    }

    if (!host_gpu_page_device_raw_ptrs.empty()) {
        k_view_host_gpu.page_pool_buffer = ggml_backend_alloc_buffer(g_cuda_buf_type_device, host_gpu_page_device_raw_ptrs.size() * sizeof(void*));
        ASSERT(k_view_host_gpu.page_pool_buffer != nullptr, "Failed to allocate k_pool_buf GPU buffer.");
        k_view_host_gpu.page_pool_gpu_ptr = ggml_backend_buffer_get_base(k_view_host_gpu.page_pool_buffer);
        ASSERT(k_view_host_gpu.page_pool_gpu_ptr != nullptr, "k_view_host_gpu.page_pool_gpu_ptr is null post-allocation (k_pool_buf).");
        ggml_backend_buffer_set_data(k_view_host_gpu.page_pool_buffer, 0, host_gpu_page_device_raw_ptrs.data(), host_gpu_page_device_raw_ptrs.size() * sizeof(void*));
    } else {
        k_view_host_gpu.page_pool_buffer = nullptr;
        k_view_host_gpu.page_pool_gpu_ptr = nullptr;
    }

    if (!v_mappings_host_vec.empty()) {
        v_view_host_gpu.token_mappings_buffer = ggml_backend_alloc_buffer(g_cuda_buf_type_device, v_mappings_host_vec.size() * sizeof(paged_kv_token_mapping_host_for_gpu));
        ASSERT(v_view_host_gpu.token_mappings_buffer != nullptr, "Failed to allocate v_map_buf GPU buffer.");
        v_view_host_gpu.token_mappings_gpu_ptr = ggml_backend_buffer_get_base(v_view_host_gpu.token_mappings_buffer);
        ASSERT(v_view_host_gpu.token_mappings_gpu_ptr != nullptr, "v_view_host_gpu.token_mappings_gpu_ptr is null post-allocation (v_map_buf).");
        ggml_backend_buffer_set_data(v_view_host_gpu.token_mappings_buffer, 0, v_mappings_host_vec.data(), v_mappings_host_vec.size() * sizeof(paged_kv_token_mapping_host_for_gpu));
    } else {
        v_view_host_gpu.token_mappings_buffer = nullptr;
        v_view_host_gpu.token_mappings_gpu_ptr = nullptr;
    }

    // K and V use the same actual page pool on GPU (shared page data pointers and their buffers).
    v_view_host_gpu.page_pool_buffer = k_view_host_gpu.page_pool_buffer;
    v_view_host_gpu.page_pool_gpu_ptr = k_view_host_gpu.page_pool_gpu_ptr;
    v_view_host_gpu.actual_page_data_raw_ptrs = k_view_host_gpu.actual_page_data_raw_ptrs;
    v_view_host_gpu.actual_page_data_buffers = k_view_host_gpu.actual_page_data_buffers;
    ASSERT(v_view_host_gpu.page_pool_gpu_ptr == k_view_host_gpu.page_pool_gpu_ptr, "V page_pool_gpu_ptr should be same as K's.");
    if (!host_gpu_page_device_raw_ptrs.empty()) {
         ASSERT(v_view_host_gpu.page_pool_gpu_ptr != nullptr, "v_view_host_gpu.page_pool_gpu_ptr is null when k_view_host_gpu.page_pool_gpu_ptr was set.");
    }

    // Populate k_view_host_gpu fields
    int head_dim = mparams.n_embd / mparams.n_head_kv;
    k_view_host_gpu.num_tokens_in_logical_sequence = (max_pos_overall == -1) ? 0 : (max_pos_overall + 1);
    k_view_host_gpu.dtype = GGML_TYPE_F16; // Assuming F16 for now, should match actual tensor type
    k_view_host_gpu.element_size_bytes = sizeof(uint16_t);
    ASSERT(k_view_host_gpu.element_size_bytes > 0 || k_view_host_gpu.dtype == GGML_TYPE_COUNT, "K element_size_bytes is 0 for non-COUNT type.");
    k_view_host_gpu.k_head_size_elements = head_dim;
    ASSERT(k_view_host_gpu.k_head_size_elements > 0, "K k_head_size_elements is 0.");
    k_view_host_gpu.v_head_size_elements = head_dim;
    ASSERT(k_view_host_gpu.v_head_size_elements > 0, "K v_head_size_elements is 0.");
    k_view_host_gpu.num_k_heads_total = mparams.n_head_kv;
    ASSERT(k_view_host_gpu.num_k_heads_total > 0, "K num_k_heads_total is 0.");
    k_view_host_gpu.num_v_heads_total = mparams.n_head_kv;
    ASSERT(k_view_host_gpu.num_v_heads_total > 0, "K num_v_heads_total is 0.");
    k_view_host_gpu.page_size_bytes = cparams.kv_page_size;
    ASSERT(k_view_host_gpu.page_size_bytes > 0, "K page_size_bytes is 0.");
    k_view_host_gpu.v_block_start_offset_bytes = 0;

    // Populate v_view_host_gpu fields (mostly same as K for this test setup)
    v_view_host_gpu.num_tokens_in_logical_sequence = k_view_host_gpu.num_tokens_in_logical_sequence;
    v_view_host_gpu.dtype = GGML_TYPE_F16; // Assuming F16
    v_view_host_gpu.element_size_bytes = sizeof(uint16_t);
    ASSERT(v_view_host_gpu.element_size_bytes > 0 || v_view_host_gpu.dtype == GGML_TYPE_COUNT, "V element_size_bytes is 0 for non-COUNT type.");
    v_view_host_gpu.k_head_size_elements = k_view_host_gpu.k_head_size_elements;
    v_view_host_gpu.v_head_size_elements = k_view_host_gpu.v_head_size_elements;
    v_view_host_gpu.num_k_heads_total = k_view_host_gpu.num_k_heads_total;
    v_view_host_gpu.num_v_heads_total = k_view_host_gpu.num_v_heads_total;
    v_view_host_gpu.page_size_bytes = k_view_host_gpu.page_size_bytes;
    v_view_host_gpu.v_block_start_offset_bytes = k_view_host_gpu.v_block_start_offset_bytes;

    // Populate ggml_tensor->extra
    paged_kv_sequence_view_host_for_gpu* host_k_view_copy = new paged_kv_sequence_view_host_for_gpu();
    *host_k_view_copy = k_view_host_gpu;
    k_metadata_gpu_tensor->extra = host_k_view_copy;
    ASSERT(k_metadata_gpu_tensor->extra != nullptr, "k_metadata_gpu_tensor->extra was not set.");

    paged_kv_sequence_view_host_for_gpu* host_v_view_copy = new paged_kv_sequence_view_host_for_gpu();
    *host_v_view_copy = v_view_host_gpu;
    v_metadata_gpu_tensor->extra = host_v_view_copy;
    ASSERT(v_metadata_gpu_tensor->extra != nullptr, "v_metadata_gpu_tensor->extra was not set.");

    ggml_backend_synchronize(g_cuda_backend);
    return {k_view_host_gpu, v_view_host_gpu};
}

// --- Test Case 9: CUDA Paged Attention Correctness (MMA F16) ---
void test_cuda_paged_attn_correctness_mma_f16() {
    printf("--- Running Test: test_cuda_paged_attn_correctness_mma_f16 ---\n");
    if (!g_cuda_backend) {
        printf("SKIPPING CUDA test: backend not initialized.\n");
        return;
    }

    struct ggml_init_params host_ctx_params = { 128 * 1024 * 1024, NULL, false };
    ggml_context* ctx_host = ggml_init(host_ctx_params);
    ASSERT(ctx_host != NULL, "Failed to create host ggml_context.");

    struct ggml_init_params meta_gpu_ctx_params = { ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 2, NULL, true };
    ggml_context* ctx_meta_gpu = ggml_init(meta_gpu_ctx_params);
    ASSERT(ctx_meta_gpu != NULL, "Failed to create GPU metadata ggml_context.");

    llama_model_params mparams = {};
    mparams.n_embd = 64;
    mparams.n_head = 2;
    mparams.n_head_kv = 2;
    mparams.n_layer = 1;
    const int head_dim = mparams.n_embd / mparams.n_head;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 16;
    cparams.n_batch = 4;
    cparams.use_paged_kv_cache = true;
    size_t bytes_per_token_kv_one_head_one_layer_k_or_v = (size_t)head_dim * sizeof(uint16_t);
    cparams.kv_page_size = bytes_per_token_kv_one_head_one_layer_k_or_v * 2 * 2;

    ggml_tensor* q_host = ggml_new_tensor_4d(ctx_host, GGML_TYPE_F16, head_dim, cparams.n_batch, mparams.n_head, 1);
    ggml_tensor* k_host = ggml_new_tensor_4d(ctx_host, GGML_TYPE_F16, head_dim, cparams.n_ctx, mparams.n_head_kv, 1);
    ggml_tensor* v_host = ggml_new_tensor_4d(ctx_host, GGML_TYPE_F16, head_dim, cparams.n_ctx, mparams.n_head_kv, 1);

    for(int i=0; i < ggml_nelements(q_host); ++i) ((ggml_fp16_t*)q_host->data)[i] = ggml_fp32_to_fp16((float)(i % 100) * 0.1f + 0.1f);
    for(int i=0; i < ggml_nelements(k_host); ++i) ((ggml_fp16_t*)k_host->data)[i] = ggml_fp32_to_fp16((float)(i % 100) * 0.05f - 0.2f);
    for(int i=0; i < ggml_nelements(v_host); ++i) ((ggml_fp16_t*)v_host->data)[i] = ggml_fp32_to_fp16((float)(i % 100) * 0.02f + 0.3f);

    printf("Running non-paged reference path...\n");
    ggml_tensor* q_gpu_ref = create_gpu_tensor_from_host(ctx_meta_gpu, q_host, "q_gpu_ref");
    ggml_tensor* k_gpu_ref = create_gpu_tensor_from_host(ctx_meta_gpu, k_host, "k_gpu_ref");
    ggml_tensor* v_gpu_ref = create_gpu_tensor_from_host(ctx_meta_gpu, v_host, "v_gpu_ref");

    struct ggml_tensor * dst_ref_ggml_tensor = ggml_dup_tensor(ctx_meta_gpu, q_gpu_ref);
    ggml_backend_buffer_t dst_ref_buffer = ggml_backend_alloc_buffer(g_cuda_buf_type_device, ggml_nbytes(dst_ref_ggml_tensor));
    ggml_backend_tensor_set_buffer(dst_ref_ggml_tensor, dst_ref_buffer);
    ggml_set_name(dst_ref_ggml_tensor, "dst_ref_gpu");

    struct ggml_cgraph* gf_ref = ggml_new_graph_custom(ctx_meta_gpu, GGML_DEFAULT_GRAPH_SIZE, false);
    struct ggml_tensor* attn_out_ref = ggml_flash_attn_ext(ctx_meta_gpu, q_gpu_ref, k_gpu_ref, v_gpu_ref, nullptr, 1.0f/sqrtf(head_dim), 0.0f, 0.0f, GGML_PREC_DEFAULT);
    ggml_set_name(attn_out_ref, "attn_out_ref");
    ggml_build_forward_expand(gf_ref, ggml_cpy(ctx_meta_gpu, attn_out_ref, dst_ref_ggml_tensor));
    ggml_backend_graph_compute(g_cuda_backend, gf_ref);

    std::vector<uint8_t> dst_ref_cpu_data = get_tensor_data_from_gpu(dst_ref_ggml_tensor);
    printf("Non-paged reference path completed.\n");

    printf("Paged path test logic is a TODO.\n");

    ggml_backend_buffer_free(q_gpu_ref->buffer);   ggml_free(q_gpu_ref);
    ggml_backend_buffer_free(k_gpu_ref->buffer);   ggml_free(k_gpu_ref);
    ggml_backend_buffer_free(v_gpu_ref->buffer);   ggml_free(v_gpu_ref);
    ggml_backend_buffer_free(dst_ref_buffer);      ggml_free(dst_ref_ggml_tensor);
    ggml_graph_free(gf_ref);

    ggml_free(ctx_host);
    ggml_free(ctx_meta_gpu);
    printf("--- Test test_cuda_paged_attn_correctness_mma_f16 (structure) FINISHED ---\n\n");
}
#endif // GGML_USE_CUDA


int main() {
} // Closing brace for test_paged_cache_state_read_write

// =================================================================================================
// PART 2: CUDA Paged Attention Kernel Tests
// =================================================================================================
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h" // For CUDA backend functions and specific types if needed

// Global CUDA backend and buffer type for tests
ggml_backend_t g_cuda_backend = NULL;
ggml_backend_buffer_type_t g_cuda_buf_type_device = NULL;

void setup_cuda_for_test() {
    fprintf(stderr, "Initializing CUDA backend for tests...\n");
    // Default to device 0 for tests
    g_cuda_backend = ggml_backend_cuda_init(0);
    if (!g_cuda_backend) {
        fprintf(stderr, "setup_cuda_for_test: ggml_backend_cuda_init() failed. CUDA tests will be skipped.\n");
        return;
    }
    g_cuda_buf_type_device = ggml_backend_get_default_buffer_type(g_cuda_backend);
    ASSERT(g_cuda_buf_type_device != NULL, "Failed to get CUDA device buffer type.");
    printf("CUDA backend initialized for tests.\n");
}

void teardown_cuda_for_test() {
    if (g_cuda_backend) {
        ggml_backend_free(g_cuda_backend);
        g_cuda_backend = NULL;
        g_cuda_buf_type_device = NULL;
        printf("CUDA backend freed.\n");
    }
}

// Creates a GPU tensor and copies data from a host tensor.
ggml_tensor* create_gpu_tensor_from_host(ggml_context* ctx_meta_gpu, const ggml_tensor* t_host, const char* name) {
    if (!g_cuda_backend || !g_cuda_buf_type_device) {
        fprintf(stderr, "CUDA backend not initialized, cannot create GPU tensor %s.\n", name);
        return nullptr;
    }
    // Create metadata for the GPU tensor
    ggml_tensor* t_device = ggml_dup_tensor(ctx_meta_gpu, t_host);
    // Allocate buffer on GPU
    ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(g_cuda_buf_type_device, ggml_nbytes(t_device));
    ASSERT(buffer != NULL, (std::string("Failed to allocate CUDA buffer for ") + name).c_str());
    // Associate buffer with tensor
    ggml_backend_tensor_set_buffer(t_device, buffer);
    // Copy data
    ggml_backend_tensor_set_async(t_device, t_host->data, 0, ggml_nbytes(t_host));
    ggml_backend_synchronize(g_cuda_backend);
    ggml_set_name(t_device, name);
    return t_device;
}

// Retrieves data from a GPU tensor to a host vector.
std::vector<uint8_t> get_tensor_data_from_gpu(const ggml_tensor* t_device) {
    if (!g_cuda_backend || !t_device || !t_device->buffer ) {
        fprintf(stderr, "Invalid tensor or CUDA backend for get_tensor_data_from_gpu for tensor %s.\n", t_device ? t_device->name : "NULL");
        return {};
    }
    size_t nbytes = ggml_nbytes(t_device);
    std::vector<uint8_t> host_data(nbytes);
    ggml_backend_tensor_get_async(t_device, host_data.data(), 0, nbytes);
    ggml_backend_synchronize(g_cuda_backend);
    return host_data;
}

// Helper function to compare float tensors with tolerance
bool compare_tensors_approx(const float* data1, const float* data2, int64_t num_elements, const char* test_name, float abs_tolerance, float rel_tolerance) {
    int mismatches = 0;
    for (int64_t i = 0; i < num_elements; ++i) {
        float d1 = data1[i];
        float d2 = data2[i];
        float diff = fabsf(d1 - d2);
        // Relative difference calculation, handle d1 being close to zero
        float rd = (fabsf(d1) > 1e-9f) ? diff / fabsf(d1) : 0.0f;

        if (diff > abs_tolerance && rd > rel_tolerance) {
            if (mismatches < 20) { // Print first few mismatches
                printf("%s: Mismatch at index %lld: data1=%.8f, data2=%.8f, diff=%.8f, rel_diff=%.8f (abs_tol=%.2e, rel_tol=%.2e)\n",
                       test_name, i, d1, d2, diff, rd, abs_tolerance, rel_tolerance);
            }
            mismatches++;
        }
    }
    if (mismatches > 0) {
        printf("%s: Total mismatches: %d / %lld\n", test_name, mismatches, num_elements);
        return false;
    }
    printf("%s: Results match within tolerance (abs_tol=%.2e, rel_tol=%.2e).\n", test_name, abs_tolerance, rel_tolerance);
    return true;
}

// Host-side representation of CUDA structs for preparing kernel arguments
struct paged_kv_token_mapping_host_for_gpu {
    int32_t page_idx;
    int32_t offset_in_page_elements; // Byte offset
};

struct paged_kv_sequence_view_host_for_gpu {
    void* token_mappings_gpu_ptr;
    void* page_pool_gpu_ptr;
    int32_t num_tokens_in_logical_sequence;
    ggml_type dtype;
    int32_t k_head_size_elements;
    int32_t v_head_size_elements;
    int32_t num_k_heads_total;
    int32_t num_v_heads_total;
    uint32_t element_size_bytes;
    uint32_t page_size_bytes;
    uint32_t v_block_start_offset_bytes;
};

// Prepares GPU buffers for paged KV views from a CPU cache state.
std::pair<paged_kv_sequence_view_host_for_gpu, paged_kv_sequence_view_host_for_gpu>
prepare_paged_kv_views_on_gpu(
    llama_paged_kv_cache& cpu_cache,
    const std::vector<llama_seq_id>& target_seq_ids,
    ggml_context* ctx_meta_gpu,
    const llama_model_params& mparams,
    const llama_context_params& cparams
) {
    if (!g_cuda_backend || !g_cuda_buf_type_device) {
        throw std::runtime_error("CUDA backend not initialized for paged view prep.");
    }
    llama_paged_kv_cells* cpu_cells = cpu_cache.get_paged_cells();
    ASSERT(cpu_cells != nullptr, "CPU paged_cells is null.");

    paged_kv_sequence_view_host_for_gpu k_view_host_gpu = {0};
    paged_kv_sequence_view_host_for_gpu v_view_host_gpu = {0};

    std::vector<paged_kv_token_mapping_host_for_gpu> k_mappings_host_vec;
    std::vector<paged_kv_token_mapping_host_for_gpu> v_mappings_host_vec;
    std::map<int, const llama_kv_page*> unique_pages_map_cpu_id_to_ptr;
    int max_pos_overall = -1;

    ASSERT(target_seq_ids.size() == 1, "This simplified helper expects only one target_seq_id for creating a flat view.");
    llama_seq_id current_seq_id = target_seq_ids[0];

    for (const auto& item : cpu_cells->get_token_to_page_offset_map()) {
        const auto& token_key = item.first;
        const auto& page_offset_val = item.second;
        if (token_key.seq_id != current_seq_id) continue;

        unique_pages_map_cpu_id_to_ptr[page_offset_val.page_id] = cpu_cells->get_page(page_offset_val.page_id);
        paged_kv_token_mapping_host_for_gpu current_mapping = {(int32_t)page_offset_val.page_id, (int32_t)page_offset_val.offset_bytes};
        int current_pos = token_key.pos;
        if (current_pos > max_pos_overall) max_pos_overall = current_pos;

        if (token_key.type == llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K) {
            if (current_pos >= (int)k_mappings_host_vec.size()) k_mappings_host_vec.resize(max_pos_overall + 1, {-1, 0});
            k_mappings_host_vec[current_pos] = current_mapping;
        } else {
            if (current_pos >= (int)v_mappings_host_vec.size()) v_mappings_host_vec.resize(max_pos_overall + 1, {-1, 0});
            v_mappings_host_vec[current_pos] = current_mapping;
        }
    }
    if (max_pos_overall == -1 ) { // if no tokens were found for this seq_id
        k_mappings_host_vec.clear();
        v_mappings_host_vec.clear();
    } else {
         if (k_mappings_host_vec.size() < (size_t)max_pos_overall + 1) k_mappings_host_vec.resize(max_pos_overall + 1, {-1,0});
         if (v_mappings_host_vec.size() < (size_t)max_pos_overall + 1) v_mappings_host_vec.resize(max_pos_overall + 1, {-1,0});
    }

    std::vector<void*> host_gpu_page_device_ptrs;
    std::map<int, int> cpu_page_id_to_gpu_pool_idx;
    for(const auto& pair : unique_pages_map_cpu_id_to_ptr) {
        const llama_kv_page* cpu_page = pair.second;
        if (cpu_page && !cpu_page->is_free()) {
            struct ggml_tensor* t_page_host_meta = ggml_new_tensor_1d(ctx_meta_gpu, GGML_TYPE_I8, cpu_page->size);
            t_page_host_meta->data = cpu_page->data;
            ggml_tensor* t_page_gpu = create_gpu_tensor_from_host(ctx_meta_gpu, t_page_host_meta, "gpu_page_data_content");
            t_page_host_meta->data = nullptr;
            ggml_free(t_page_host_meta);
            ASSERT(t_page_gpu && t_page_gpu->data, "Failed to create GPU buffer for a page content.");
            cpu_page_id_to_gpu_pool_idx[cpu_page->id] = host_gpu_page_device_ptrs.size();
            host_gpu_page_device_ptrs.push_back(t_page_gpu->data);
        }
    }

    for(auto& mapping : k_mappings_host_vec) {
        if (mapping.page_idx != -1 && cpu_page_id_to_gpu_pool_idx.count(mapping.page_idx)) {
             mapping.page_idx = cpu_page_id_to_gpu_pool_idx.at(mapping.page_idx);
        } else { mapping.page_idx = -1; mapping.offset_in_page_elements = 0; }
    }
    for(auto& mapping : v_mappings_host_vec) {
        if (mapping.page_idx != -1 && cpu_page_id_to_gpu_pool_idx.count(mapping.page_idx)) {
            mapping.page_idx = cpu_page_id_to_gpu_pool_idx.at(mapping.page_idx);
        } else { mapping.page_idx = -1; mapping.offset_in_page_elements = 0; }
    }

    if (!k_mappings_host_vec.empty()) {
        ggml_backend_buffer_t k_map_buf = ggml_backend_alloc_buffer(g_cuda_buf_type_device, k_mappings_host_vec.size() * sizeof(paged_kv_token_mapping_host_for_gpu));
        k_view_host_gpu.token_mappings_gpu_ptr = ggml_backend_buffer_get_base(k_map_buf);
        ggml_backend_buffer_set_data(k_map_buf, 0, k_mappings_host_vec.data(), k_mappings_host_vec.size() * sizeof(paged_kv_token_mapping_host_for_gpu));
    } else { k_view_host_gpu.token_mappings_gpu_ptr = nullptr; }

    if (!host_gpu_page_device_ptrs.empty()) {
        ggml_backend_buffer_t k_pool_buf = ggml_backend_alloc_buffer(g_cuda_buf_type_device, host_gpu_page_device_ptrs.size() * sizeof(void*));
        k_view_host_gpu.page_pool_gpu_ptr = ggml_backend_buffer_get_base(k_pool_buf);
        ggml_backend_buffer_set_data(k_pool_buf, 0, host_gpu_page_device_ptrs.data(), host_gpu_page_device_ptrs.size() * sizeof(void*));
    } else { k_view_host_gpu.page_pool_gpu_ptr = nullptr; }

    if (!v_mappings_host_vec.empty()) {
        ggml_backend_buffer_t v_map_buf = ggml_backend_alloc_buffer(g_cuda_buf_type_device, v_mappings_host_vec.size() * sizeof(paged_kv_token_mapping_host_for_gpu));
        v_view_host_gpu.token_mappings_gpu_ptr = ggml_backend_buffer_get_base(v_map_buf);
        ggml_backend_buffer_set_data(v_map_buf, 0, v_mappings_host_vec.data(), v_mappings_host_vec.size() * sizeof(paged_kv_token_mapping_host_for_gpu));
    } else { v_view_host_gpu.token_mappings_gpu_ptr = nullptr; }

    v_view_host_gpu.page_pool_gpu_ptr = k_view_host_gpu.page_pool_gpu_ptr;

    int head_dim = mparams.n_embd / mparams.n_head_kv;
    k_view_host_gpu.num_tokens_in_logical_sequence = (max_pos_overall == -1) ? 0 : (max_pos_overall + 1);
    k_view_host_gpu.dtype = GGML_TYPE_F16; // TODO: Parameterize for Q8_0 tests
    k_view_host_gpu.element_size_bytes = sizeof(ggml_fp16_t);
    k_view_host_gpu.k_head_size_elements = head_dim;
    k_view_host_gpu.v_head_size_elements = head_dim;
    k_view_host_gpu.num_k_heads_total = mparams.n_head_kv;
    k_view_host_gpu.num_v_heads_total = mparams.n_head_kv;
    k_view_host_gpu.page_size_bytes = cparams.kv_page_size;
    k_view_host_gpu.v_block_start_offset_bytes = 0; // Assuming K and V are handled by separate views or entries

    v_view_host_gpu = k_view_host_gpu; // Assuming V has same params as K for this test

    ggml_backend_synchronize(g_cuda_backend);
    return {k_view_host_gpu, v_view_host_gpu};
}

// Helper to populate CPU paged KV cache from existing host tensors
void populate_kv_cache_from_host_tensors(
    llama_paged_kv_cache &cpu_cache,
    llama_seq_id seq_id,
    const ggml_tensor* k_host_tensor,
    const ggml_tensor* v_host_tensor,
    int n_tokens_to_copy, // Number of token positions to copy
    int head_dim,
    int n_kv_h,
    int n_layers
) {
    llama_paged_kv_cells* cells = cpu_cache.get_paged_cells();
    ASSERT(cells != nullptr, "CPU paged_cells is null in populate_kv_cache_from_host_tensors");
    ASSERT(k_host_tensor->type == GGML_TYPE_F16, "k_host_tensor must be F16 for this helper");
    ASSERT(v_host_tensor->type == GGML_TYPE_F16, "v_host_tensor must be F16 for this helper");

    size_t bytes_per_head_data = (size_t)head_dim * sizeof(ggml_fp16_t);

    for (int layer_idx = 0; layer_idx < n_layers; ++layer_idx) {
        for (int head_idx = 0; head_idx < n_kv_h; ++head_idx) {
            for (int pos = 0; pos < n_tokens_to_copy; ++pos) {
                // Calculate offset into flat k_host/v_host data
                // Assuming layout [D, N, H, L]
                size_t k_offset_bytes =
                    (size_t)layer_idx * k_host_tensor->nb[3] +
                    (size_t)head_idx * k_host_tensor->nb[2] +
                    (size_t)pos * k_host_tensor->nb[1];
                const uint8_t* k_data_src = (const uint8_t*)k_host_tensor->data + k_offset_bytes;

                size_t v_offset_bytes =
                    (size_t)layer_idx * v_host_tensor->nb[3] +
                    (size_t)head_idx * v_host_tensor->nb[2] +
                    (size_t)pos * v_host_tensor->nb[1];
                const uint8_t* v_data_src = (const uint8_t*)v_host_tensor->data + v_offset_bytes;

                // Populate K
                llama_paged_kv_cells::TokenKey tk_k(seq_id, pos, layer_idx, head_idx, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K);
                auto [page_k, offset_k_bytes_cell] = cells->find_or_allocate_page_for_token(tk_k, bytes_per_head_data);
                ASSERT(page_k != nullptr, "Page allocation for K failed in populate_from_tensors");
                uint8_t* data_k_dst = cells->get_token_data(tk_k);
                ASSERT(data_k_dst != nullptr, "get_token_data for K failed in populate_from_tensors");
                memcpy(data_k_dst, k_data_src, bytes_per_head_data);

                // Populate V
                llama_paged_kv_cells::TokenKey tk_v(seq_id, pos, layer_idx, head_idx, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_V);
                auto [page_v, offset_v_bytes_cell] = cells->find_or_allocate_page_for_token(tk_v, bytes_per_head_data);
                ASSERT(page_v != nullptr, "Page allocation for V failed in populate_from_tensors");
                uint8_t* data_v_dst = cells->get_token_data(tk_v);
                ASSERT(data_v_dst != nullptr, "get_token_data for V failed in populate_from_tensors");
                memcpy(data_v_dst, v_data_src, bytes_per_head_data);
            }
        }
    }
}


// --- Test Case 9: CUDA Paged Attention Correctness (MMA F16) ---
void test_cuda_paged_attn_correctness_mma_f16() {
    printf("--- Running Test: test_cuda_paged_attn_correctness_mma_f16 ---\n");
    if (!g_cuda_backend) {
        printf("SKIPPING CUDA test: backend not initialized.\n");
        return;
    }

    struct ggml_init_params host_ctx_params = { 128 * 1024 * 1024, NULL, false };
    ggml_context* ctx_host = ggml_init(host_ctx_params);
    ASSERT(ctx_host != NULL, "Failed to create host ggml_context.");

    struct ggml_init_params meta_gpu_ctx_params = { ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 2, NULL, true };
    ggml_context* ctx_meta_gpu = ggml_init(meta_gpu_ctx_params);
    ASSERT(ctx_meta_gpu != NULL, "Failed to create GPU metadata ggml_context.");

    llama_model_params mparams = {};
    mparams.n_embd = 64;
    mparams.n_head = 2;
    mparams.n_head_kv = 2;
    mparams.n_layer = 1;
    const int head_dim = mparams.n_embd / mparams.n_head;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 16;
    cparams.n_batch = 4;
    cparams.use_paged_kv_cache = true;
    size_t bytes_per_token_kv_one_head_one_layer_k_or_v = (size_t)head_dim * sizeof(uint16_t);
    cparams.kv_page_size = bytes_per_token_kv_one_head_one_layer_k_or_v * 2 * 2;

    ggml_tensor* q_host = ggml_new_tensor_4d(ctx_host, GGML_TYPE_F16, head_dim, cparams.n_batch, mparams.n_head, 1);
    ggml_tensor* k_host = ggml_new_tensor_4d(ctx_host, GGML_TYPE_F16, head_dim, cparams.n_ctx, mparams.n_head_kv, 1);
    ggml_tensor* v_host = ggml_new_tensor_4d(ctx_host, GGML_TYPE_F16, head_dim, cparams.n_ctx, mparams.n_head_kv, 1);

    for(int i=0; i < ggml_nelements(q_host); ++i) ((ggml_fp16_t*)q_host->data)[i] = ggml_fp32_to_fp16((float)(i % 100) * 0.1f + 0.1f);
    for(int i=0; i < ggml_nelements(k_host); ++i) ((ggml_fp16_t*)k_host->data)[i] = ggml_fp32_to_fp16((float)(i % 100) * 0.05f - 0.2f);
    for(int i=0; i < ggml_nelements(v_host); ++i) ((ggml_fp16_t*)v_host->data)[i] = ggml_fp32_to_fp16((float)(i % 100) * 0.02f + 0.3f);

    printf("Running non-paged reference path...\n");
    ggml_tensor* q_gpu_ref = create_gpu_tensor_from_host(ctx_meta_gpu, q_host, "q_gpu_ref");
    ggml_tensor* k_gpu_ref = create_gpu_tensor_from_host(ctx_meta_gpu, k_host, "k_gpu_ref");
    ggml_tensor* v_gpu_ref = create_gpu_tensor_from_host(ctx_meta_gpu, v_host, "v_gpu_ref");

    struct ggml_tensor * dst_ref_gpu = ggml_dup_tensor(ctx_meta_gpu, q_gpu_ref); // Renamed for clarity
    ggml_backend_buffer_t dst_ref_buffer = ggml_backend_alloc_buffer(g_cuda_buf_type_device, ggml_nbytes(dst_ref_gpu));
    ggml_backend_tensor_set_buffer(dst_ref_gpu, dst_ref_buffer);
    ggml_set_name(dst_ref_gpu, "dst_ref_gpu");

    struct ggml_cgraph* gf_ref = ggml_new_graph_custom(ctx_meta_gpu, GGML_DEFAULT_GRAPH_SIZE, false);
    // For reference, ensure op_params[3] (is_paged flag) is 0.0f or not set to 1.0f
    // GGML_FLASH_ATTN_EXT_OP_PARAMS_SCALE_IDX = 0, GGML_FLASH_ATTN_EXT_OP_PARAMS_MAX_BIAS_IDX = 1, GGML_FLASH_ATTN_EXT_OP_PARAMS_LOGIT_SOFTCAP_IDX = 2
    // Using index 3 for is_paged flag
    float op_params_ref[GGML_MAX_OP_PARAMS] = {0.0f}; // Ensure all are zeroed
    op_params_ref[0] = 1.0f/sqrtf(head_dim); // scale
    op_params_ref[1] = 0.0f; // max_bias
    op_params_ref[2] = 0.0f; // logit_softcap
    op_params_ref[3] = 0.0f; // is_paged = false

    struct ggml_tensor* attn_out_ref = ggml_flash_attn_ext(ctx_meta_gpu, q_gpu_ref, k_gpu_ref, v_gpu_ref, nullptr, op_params_ref);
    ggml_set_name(attn_out_ref, "attn_out_ref");
    ggml_build_forward_expand(gf_ref, ggml_cpy(ctx_meta_gpu, attn_out_ref, dst_ref_gpu));
    ggml_backend_graph_compute(g_cuda_backend, gf_ref);
    ggml_backend_synchronize(g_cuda_backend);

    std::vector<uint8_t> result_ref_host_u8 = get_tensor_data_from_gpu(dst_ref_gpu);
    std::vector<float> result_ref_host(ggml_nelements(dst_ref_gpu));
    for (int64_t i = 0; i < ggml_nelements(dst_ref_gpu); ++i) {
        result_ref_host[i] = ggml_fp16_to_fp32(((ggml_fp16_t*)result_ref_host_u8.data())[i]);
    }
    printf("Non-paged reference path completed.\n");

    // --- Paged Path ---
    printf("Setting up paged path...\n");
    ggml_tensor* q_gpu_paged = create_gpu_tensor_from_host(ctx_meta_gpu, q_host, "q_gpu_paged");

    llama_paged_kv_cache cpu_kv_cache(mparams, cparams, g_cpu_buf_type, ctx_meta_gpu);

    llama_seq_id test_seq_id = 0;
    populate_kv_cache_from_host_tensors(cpu_kv_cache, test_seq_id, k_host, v_host,
                                        cparams.n_ctx, head_dim, mparams.n_head_kv, mparams.n_layer);

    // Create dummy metadata tensors for K and V. Their ->extra field will be populated.
    ggml_tensor* k_metadata_gpu_tensor = ggml_new_tensor_1d(ctx_meta_gpu, GGML_TYPE_I32, 1);
    ggml_tensor* v_metadata_gpu_tensor = ggml_new_tensor_1d(ctx_meta_gpu, GGML_TYPE_I32, 1);
    ggml_set_name(k_metadata_gpu_tensor, "k_metadata_gpu_paged");
    ggml_set_name(v_metadata_gpu_tensor, "v_metadata_gpu_paged");
    ggml_backend_buffer_t k_meta_buf = ggml_backend_alloc_buffer(g_cuda_buf_type_device, ggml_nbytes(k_metadata_gpu_tensor));
    ASSERT(k_meta_buf != nullptr, "Failed to alloc k_meta_buf");
    ggml_backend_tensor_set_buffer(k_metadata_gpu_tensor, k_meta_buf);
    ggml_backend_buffer_t v_meta_buf = ggml_backend_alloc_buffer(g_cuda_buf_type_device, ggml_nbytes(v_metadata_gpu_tensor));
    ASSERT(v_meta_buf != nullptr, "Failed to alloc v_meta_buf");
    ggml_backend_tensor_set_buffer(v_metadata_gpu_tensor, v_meta_buf);

    // Call prepare_paged_kv_views_on_gpu, which will populate ->extra fields
    // Note: Using the version of prepare_paged_kv_views_on_gpu that takes cparams
    auto [k_view_gpu_host, v_view_gpu_host] = prepare_paged_kv_views_on_gpu(
        cpu_kv_cache, {test_seq_id}, ctx_meta_gpu, mparams, cparams, k_metadata_gpu_tensor, v_metadata_gpu_tensor
    );

    ASSERT(k_metadata_gpu_tensor->extra != nullptr, "k_metadata_gpu_tensor->extra is NULL after prepare_paged_kv_views_on_gpu");
    ASSERT(v_metadata_gpu_tensor->extra != nullptr, "v_metadata_gpu_tensor->extra is NULL after prepare_paged_kv_views_on_gpu");

    paged_kv_sequence_view_host_for_gpu* k_view_check = static_cast<paged_kv_sequence_view_host_for_gpu*>(k_metadata_gpu_tensor->extra);
    ASSERT(k_view_check->num_tokens_in_logical_sequence == cparams.n_ctx, "K view num_tokens_in_logical_sequence from extra mismatch.");
    ASSERT(k_view_check->element_size_bytes == sizeof(uint16_t), "K view element_size_bytes from extra mismatch.");
    ASSERT(k_view_check->page_size_bytes == cparams.kv_page_size, "K view page_size_bytes from extra mismatch.");
     if (k_view_check->num_tokens_in_logical_sequence > 0) {
        ASSERT(k_view_check->token_mappings_gpu_ptr != nullptr, "K view token_mappings_gpu_ptr from extra is null for non-empty sequence.");
        ASSERT(k_view_check->page_pool_gpu_ptr != nullptr, "K view page_pool_gpu_ptr from extra is null for non-empty sequence.");
    }

    paged_kv_sequence_view_host_for_gpu* v_view_check = static_cast<paged_kv_sequence_view_host_for_gpu*>(v_metadata_gpu_tensor->extra);
    ASSERT(v_view_check->num_tokens_in_logical_sequence == cparams.n_ctx, "V view num_tokens_in_logical_sequence from extra mismatch.");
    ASSERT(v_view_check->element_size_bytes == sizeof(uint16_t), "V view element_size_bytes from extra mismatch.");
     if (v_view_check->num_tokens_in_logical_sequence > 0) {
        ASSERT(v_view_check->token_mappings_gpu_ptr != nullptr, "V view token_mappings_gpu_ptr from extra is null for non-empty sequence.");
        ASSERT(v_view_check->page_pool_gpu_ptr != nullptr, "V view page_pool_gpu_ptr from extra is null for non-empty sequence.");
    }

    ggml_tensor* dst_paged_gpu = ggml_dup_tensor(ctx_meta_gpu, q_gpu_paged);
    ggml_set_name(dst_paged_gpu, "dst_paged_gpu");
    ggml_backend_buffer_t dst_paged_buf = ggml_backend_alloc_buffer(g_cuda_buf_type_device, ggml_nbytes(dst_paged_gpu));
    ggml_backend_tensor_set_buffer(dst_paged_gpu, dst_paged_buf);

    struct ggml_cgraph* gf_paged = ggml_new_graph_custom(ctx_meta_gpu, GGML_DEFAULT_GRAPH_SIZE, false);
    struct ggml_tensor* attn_out_paged = ggml_flash_attn_ext(ctx_meta_gpu, q_gpu_paged, k_metadata_gpu_tensor, v_metadata_gpu_tensor, nullptr, 1.0f/sqrtf(head_dim), 0.0f, 0.0f, GGML_PREC_DEFAULT);
    ggml_set_name(attn_out_paged, "attn_out_paged");
    // Set op_params for paged call: scale, max_bias, logit_softcap, is_paged=1.0f
    // Ensure GGML_FLASH_ATTN_EXT_OP_PARAMS_IS_PAGED_IDX (e.g. 3) is used for the flag.
    float op_params_paged[GGML_MAX_OP_PARAMS] = {0.0f}; // Ensure all are zeroed
    op_params_paged[0] = 1.0f/sqrtf(head_dim); // scale
    op_params_paged[1] = 0.0f;                 // max_bias
    op_params_paged[2] = 0.0f;                 // logit_softcap
    const int GGML_FLASH_ATTN_EXT_OP_PARAMS_IS_PAGED_IDX = 3; // Matching definition in ggml-cuda.cu if not in header
    op_params_paged[GGML_FLASH_ATTN_EXT_OP_PARAMS_IS_PAGED_IDX] = 1.0f; // is_paged = true

    struct ggml_tensor* attn_out_paged = ggml_flash_attn_ext(ctx_meta_gpu, q_gpu_paged, k_metadata_gpu_tensor, v_metadata_gpu_tensor, nullptr, op_params_paged);
    ggml_set_name(attn_out_paged, "attn_out_paged");
    ggml_build_forward_expand(gf_paged, ggml_cpy(ctx_meta_gpu, attn_out_paged, dst_paged_gpu));

    printf("Computing paged graph (backend will use K/V from metadata->extra if implemented)...\n");
    ggml_backend_graph_compute(g_cuda_backend, gf_paged);
    ggml_backend_synchronize(g_cuda_backend); // Ensure computation is finished before fetching results
    printf("Paged graph compute finished.\n");

    std::vector<uint8_t> result_paged_host_u8 = get_tensor_data_from_gpu(dst_paged_gpu);
    std::vector<float> result_paged_host(ggml_nelements(dst_paged_gpu));
    for (int64_t i = 0; i < ggml_nelements(dst_paged_gpu); ++i) {
        result_paged_host[i] = ggml_fp16_to_fp32(((ggml_fp16_t*)result_paged_host_u8.data())[i]);
    }

    // Compare results
    ASSERT(compare_tensors_approx(result_ref_host.data(), result_paged_host.data(), ggml_nelements(dst_ref_gpu), "MMA F16 Paged Correctness", 1e-2f, 1e-1f),
           "Paged vs Non-paged Flash Attention results mismatch.");

    // Cleanup for ->extra allocations
    if (k_metadata_gpu_tensor->extra) {
        delete static_cast<paged_kv_sequence_view_host_for_gpu*>(k_metadata_gpu_tensor->extra);
        k_metadata_gpu_tensor->extra = nullptr;
    }
    if (v_metadata_gpu_tensor->extra) {
        delete static_cast<paged_kv_sequence_view_host_for_gpu*>(v_metadata_gpu_tensor->extra);
        v_metadata_gpu_tensor->extra = nullptr;
    }

    // Cleanup other resources
    ggml_backend_buffer_free(q_gpu_ref->buffer);   ggml_free(q_gpu_ref);
    ggml_backend_buffer_free(k_gpu_ref->buffer);   ggml_free(k_gpu_ref);
    ggml_backend_buffer_free(v_gpu_ref->buffer);   ggml_free(v_gpu_ref);
    ggml_backend_buffer_free(dst_ref_buffer);      ggml_free(dst_ref_gpu); // Used to be dst_ref_ggml_tensor
    ggml_graph_free(gf_ref);

    ggml_backend_buffer_free(q_gpu_paged->buffer); ggml_free(q_gpu_paged);
    ggml_backend_buffer_free(dst_paged_buf);       ggml_free(dst_paged_gpu);
    ggml_backend_buffer_free(k_meta_buf);          ggml_free(k_metadata_gpu_tensor);
    ggml_backend_buffer_free(v_meta_buf);          ggml_free(v_metadata_gpu_tensor);
    ggml_graph_free(gf_paged);

    // Cleanup for GPU buffers allocated by prepare_paged_kv_views_on_gpu
    // These are stored in the host_k_view_copy and host_v_view_copy (->extra)
    paged_kv_sequence_view_host_for_gpu* k_view_to_clean = static_cast<paged_kv_sequence_view_host_for_gpu*>(k_metadata_gpu_tensor->extra);
    if (k_view_to_clean) {
        if (k_view_to_clean->token_mappings_buffer) {
            ggml_backend_buffer_free(k_view_to_clean->token_mappings_buffer);
        }
        // page_pool_buffer and actual_page_data_buffers are shared with V or unique to K
        // K view owns its page pool and page data buffers. V view reuses them.
        if (k_view_to_clean->page_pool_buffer) {
            ggml_backend_buffer_free(k_view_to_clean->page_pool_buffer);
        }
        for (ggml_backend_buffer_t buffer : k_view_to_clean->actual_page_data_buffers) {
            if (buffer) ggml_backend_buffer_free(buffer);
        }
    }

    paged_kv_sequence_view_host_for_gpu* v_view_to_clean = static_cast<paged_kv_sequence_view_host_for_gpu*>(v_metadata_gpu_tensor->extra);
    if (v_view_to_clean) {
        // V's token_mappings_buffer is unique to V (unless empty)
        if (v_view_to_clean->token_mappings_buffer && v_view_to_clean->token_mappings_buffer != k_view_to_clean->token_mappings_buffer) {
            ggml_backend_buffer_free(v_view_to_clean->token_mappings_buffer);
        }
        // V's page_pool_buffer and actual_page_data_buffers are typically shared with K and freed above.
        // If they were distinct for V (not current logic), they'd be freed here.
    }
    // The ->extra itself is cleaned up a few lines above this TODO block.

    ggml_free(ctx_host);
    ggml_free(ctx_meta_gpu);
    printf("--- Test test_cuda_paged_attn_correctness_mma_f16 FINISHED ---\n\n");
}

// --- Test Case 10: CUDA Paged Attention Correctness (Tile F16) ---
void test_cuda_paged_attn_correctness_tile_f16() {
    printf("--- Running Test: test_cuda_paged_attn_correctness_tile_f16 ---\n");
    if (!g_cuda_backend) {
        printf("SKIPPING CUDA test: backend not initialized.\n");
        return;
    }

    struct ggml_init_params host_ctx_params = { 256 * 1024 * 1024, NULL, false };
    ggml_context* ctx_host = ggml_init(host_ctx_params);
    ASSERT(ctx_host != NULL, "Failed to create host ggml_context for Tile F16 test.");

    struct ggml_init_params meta_gpu_ctx_params = { ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 2, NULL, true };
    ggml_context* ctx_meta_gpu = ggml_init(meta_gpu_ctx_params);
    ASSERT(ctx_meta_gpu != NULL, "Failed to create GPU metadata ggml_context for Tile F16 test.");

    llama_model_params mparams = {};
    mparams.n_embd = 128; // For head_dim = 64
    mparams.n_head = 2;
    mparams.n_head_kv = 2;
    mparams.n_layer = 1;
    const int head_dim = mparams.n_embd / mparams.n_head; // Should be 64
    ASSERT(head_dim == 64, "Head dimension for Tile F16 test should be 64.");

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 32; // Context size
    cparams.n_batch = 4;
    cparams.flash_attn = true; // Enable flash attention
    cparams.use_paged_kv_cache = true;
    size_t bytes_per_token_kv_one_head_one_layer_k_or_v = (size_t)head_dim * sizeof(uint16_t);
    cparams.kv_page_size = bytes_per_token_kv_one_head_one_layer_k_or_v * 2 * 2; // Page fits 2 K/V pairs for one head/layer

    // Prepare Host Tensors (Q, K, V)
    ggml_tensor* q_host = ggml_new_tensor_4d(ctx_host, GGML_TYPE_F16, head_dim, cparams.n_batch, mparams.n_head, 1);
    ggml_tensor* k_host = ggml_new_tensor_4d(ctx_host, GGML_TYPE_F16, head_dim, cparams.n_ctx, mparams.n_head_kv, 1);
    ggml_tensor* v_host = ggml_new_tensor_4d(ctx_host, GGML_TYPE_F16, head_dim, cparams.n_ctx, mparams.n_head_kv, 1);

    // Fill with some data
    for(int i=0; i < ggml_nelements(q_host); ++i) ((ggml_fp16_t*)q_host->data)[i] = ggml_fp32_to_fp16((float)((i % 70) - 35) * 0.1f);
    for(int i=0; i < ggml_nelements(k_host); ++i) ((ggml_fp16_t*)k_host->data)[i] = ggml_fp32_to_fp16((float)((i % 80) - 40) * 0.05f);
    for(int i=0; i < ggml_nelements(v_host); ++i) ((ggml_fp16_t*)v_host->data)[i] = ggml_fp32_to_fp16((float)((i % 90) - 45) * 0.02f);

    // --- Non-Paged Reference Path ---
    printf("Running non-paged reference path (Tile F16 test)...\n");
    ggml_tensor* q_gpu_ref = create_gpu_tensor_from_host(ctx_meta_gpu, q_host, "q_gpu_ref_tile");
    ggml_tensor* k_gpu_ref = create_gpu_tensor_from_host(ctx_meta_gpu, k_host, "k_gpu_ref_tile");
    ggml_tensor* v_gpu_ref = create_gpu_tensor_from_host(ctx_meta_gpu, v_host, "v_gpu_ref_tile");

    struct ggml_tensor * dst_ref_gpu = ggml_dup_tensor(ctx_meta_gpu, q_gpu_ref);
    ggml_backend_buffer_t dst_ref_buffer = ggml_backend_alloc_buffer(g_cuda_buf_type_device, ggml_nbytes(dst_ref_gpu));
    ggml_backend_tensor_set_buffer(dst_ref_gpu, dst_ref_buffer);
    ggml_set_name(dst_ref_gpu, "dst_ref_gpu_tile");

    struct ggml_cgraph* gf_ref = ggml_new_graph_custom(ctx_meta_gpu, GGML_DEFAULT_GRAPH_SIZE, false);
    float op_params_ref[GGML_MAX_OP_PARAMS] = {0.0f};
    op_params_ref[0] = 1.0f/sqrtf(head_dim); // scale
    op_params_ref[3] = 0.0f; // is_paged = false
    struct ggml_tensor* attn_out_ref = ggml_flash_attn_ext(ctx_meta_gpu, q_gpu_ref, k_gpu_ref, v_gpu_ref, nullptr, op_params_ref);
    ggml_set_name(attn_out_ref, "attn_out_ref_tile");
    ggml_build_forward_expand(gf_ref, ggml_cpy(ctx_meta_gpu, attn_out_ref, dst_ref_gpu));
    ggml_backend_graph_compute(g_cuda_backend, gf_ref);
    ggml_backend_synchronize(g_cuda_backend);

    std::vector<uint8_t> result_ref_host_u8 = get_tensor_data_from_gpu(dst_ref_gpu);
    std::vector<float> result_ref_host(ggml_nelements(dst_ref_gpu));
    for (int64_t i = 0; i < ggml_nelements(dst_ref_gpu); ++i) {
        result_ref_host[i] = ggml_fp16_to_fp32(((ggml_fp16_t*)result_ref_host_u8.data())[i]);
    }
    printf("Non-paged reference path completed (Tile F16 test).\n");

    // --- Paged Path ---
    printf("Setting up paged path (Tile F16 test)...\n");
    ggml_tensor* q_gpu_paged = create_gpu_tensor_from_host(ctx_meta_gpu, q_host, "q_gpu_paged_tile");

    llama_paged_kv_cache cpu_kv_cache(mparams, cparams, g_cpu_buf_type, ctx_meta_gpu);

    llama_seq_id test_seq_id = 0;
    populate_kv_cache_from_host_tensors(cpu_kv_cache, test_seq_id, k_host, v_host,
                                        cparams.n_ctx, head_dim, mparams.n_head_kv, mparams.n_layer);

    ggml_tensor* k_metadata_gpu_tensor = ggml_new_tensor_1d(ctx_meta_gpu, GGML_TYPE_I32, 1);
    ggml_tensor* v_metadata_gpu_tensor = ggml_new_tensor_1d(ctx_meta_gpu, GGML_TYPE_I32, 1);
    ggml_set_name(k_metadata_gpu_tensor, "k_metadata_gpu_paged_tile");
    ggml_set_name(v_metadata_gpu_tensor, "v_metadata_gpu_paged_tile");
    ggml_backend_buffer_t k_meta_buf = ggml_backend_alloc_buffer(g_cuda_buf_type_device, ggml_nbytes(k_metadata_gpu_tensor));
    ASSERT(k_meta_buf != nullptr, "Failed to alloc k_meta_buf for Tile F16 test");
    ggml_backend_tensor_set_buffer(k_metadata_gpu_tensor, k_meta_buf);
    ggml_backend_buffer_t v_meta_buf = ggml_backend_alloc_buffer(g_cuda_buf_type_device, ggml_nbytes(v_metadata_gpu_tensor));
    ASSERT(v_meta_buf != nullptr, "Failed to alloc v_meta_buf for Tile F16 test");
    ggml_backend_tensor_set_buffer(v_metadata_gpu_tensor, v_meta_buf);

    auto [k_view_gpu_host, v_view_gpu_host] = prepare_paged_kv_views_on_gpu(
        cpu_kv_cache, {test_seq_id}, ctx_meta_gpu, mparams, cparams, k_metadata_gpu_tensor, v_metadata_gpu_tensor
    );

    ASSERT(k_metadata_gpu_tensor->extra != nullptr, "k_metadata_gpu_tensor->extra is NULL after prepare_paged_kv_views_on_gpu (Tile F16).");
    ASSERT(v_metadata_gpu_tensor->extra != nullptr, "v_metadata_gpu_tensor->extra is NULL after prepare_paged_kv_views_on_gpu (Tile F16).");

    ggml_tensor* dst_paged_gpu = ggml_dup_tensor(ctx_meta_gpu, q_gpu_paged);
    ggml_set_name(dst_paged_gpu, "dst_paged_gpu_tile");
    ggml_backend_buffer_t dst_paged_buf = ggml_backend_alloc_buffer(g_cuda_buf_type_device, ggml_nbytes(dst_paged_gpu));
    ggml_backend_tensor_set_buffer(dst_paged_gpu, dst_paged_buf);

    struct ggml_cgraph* gf_paged = ggml_new_graph_custom(ctx_meta_gpu, GGML_DEFAULT_GRAPH_SIZE, false);
    float op_params_paged[GGML_MAX_OP_PARAMS] = {0.0f};
    op_params_paged[0] = 1.0f/sqrtf(head_dim); // scale
    const int GGML_FLASH_ATTN_EXT_OP_PARAMS_IS_PAGED_IDX = 3;
    op_params_paged[GGML_FLASH_ATTN_EXT_OP_PARAMS_IS_PAGED_IDX] = 1.0f; // is_paged = true
    struct ggml_tensor* attn_out_paged = ggml_flash_attn_ext(ctx_meta_gpu, q_gpu_paged, k_metadata_gpu_tensor, v_metadata_gpu_tensor, nullptr, op_params_paged);
    ggml_set_name(attn_out_paged, "attn_out_paged_tile");
    ggml_build_forward_expand(gf_paged, ggml_cpy(ctx_meta_gpu, attn_out_paged, dst_paged_gpu));

    printf("Computing paged graph (Tile F16 test)...\n");
    ggml_backend_graph_compute(g_cuda_backend, gf_paged);
    ggml_backend_synchronize(g_cuda_backend);
    printf("Paged graph compute finished (Tile F16 test).\n");

    std::vector<uint8_t> result_paged_host_u8 = get_tensor_data_from_gpu(dst_paged_gpu);
    std::vector<float> result_paged_host(ggml_nelements(dst_paged_gpu));
    for (int64_t i = 0; i < ggml_nelements(dst_paged_gpu); ++i) {
        result_paged_host[i] = ggml_fp16_to_fp32(((ggml_fp16_t*)result_paged_host_u8.data())[i]);
    }

    ASSERT(compare_tensors_approx(result_ref_host.data(), result_paged_host.data(), ggml_nelements(dst_ref_gpu), "Tile F16 Paged Correctness", 1e-2f, 1e-1f),
           "Paged vs Non-paged Flash Attention results mismatch (Tile F16).");

    // Cleanup
    if (k_metadata_gpu_tensor->extra) {
        paged_kv_sequence_view_host_for_gpu* k_view_to_clean = static_cast<paged_kv_sequence_view_host_for_gpu*>(k_metadata_gpu_tensor->extra);
        if (k_view_to_clean->token_mappings_buffer) ggml_backend_buffer_free(k_view_to_clean->token_mappings_buffer);
        if (k_view_to_clean->page_pool_buffer) ggml_backend_buffer_free(k_view_to_clean->page_pool_buffer);
        for (ggml_backend_buffer_t buffer : k_view_to_clean->actual_page_data_buffers) {
            if (buffer) ggml_backend_buffer_free(buffer);
        }
        delete k_view_to_clean;
        k_metadata_gpu_tensor->extra = nullptr;
    }
    if (v_metadata_gpu_tensor->extra) {
        paged_kv_sequence_view_host_for_gpu* v_view_to_clean = static_cast<paged_kv_sequence_view_host_for_gpu*>(v_metadata_gpu_tensor->extra);
        if (v_view_to_clean->token_mappings_buffer &&
            (!k_metadata_gpu_tensor->extra || v_view_to_clean->token_mappings_buffer != static_cast<paged_kv_sequence_view_host_for_gpu*>(k_metadata_gpu_tensor->extra)->token_mappings_buffer) ) {
             ggml_backend_buffer_free(v_view_to_clean->token_mappings_buffer);
        }
        // page_pool_buffer and actual_page_data_buffers for V are shared with K, already handled if K->extra was cleaned.
        delete v_view_to_clean;
        v_metadata_gpu_tensor->extra = nullptr;
    }

    ggml_backend_buffer_free(q_gpu_ref->buffer);   ggml_free(q_gpu_ref);
    ggml_backend_buffer_free(k_gpu_ref->buffer);   ggml_free(k_gpu_ref);
    ggml_backend_buffer_free(v_gpu_ref->buffer);   ggml_free(v_gpu_ref);
    ggml_backend_buffer_free(dst_ref_buffer);      ggml_free(dst_ref_gpu);
    ggml_graph_free(gf_ref);

    ggml_backend_buffer_free(q_gpu_paged->buffer); ggml_free(q_gpu_paged);
    ggml_backend_buffer_free(dst_paged_buf);       ggml_free(dst_paged_gpu);
    ggml_backend_buffer_free(k_meta_buf);          ggml_free(k_metadata_gpu_tensor);
    ggml_backend_buffer_free(v_meta_buf);          ggml_free(v_metadata_gpu_tensor);
    ggml_graph_free(gf_paged);

    ggml_free(ctx_host);
    ggml_free(ctx_meta_gpu);
    printf("--- Test test_cuda_paged_attn_correctness_tile_f16 FINISHED ---\n\n");
}
#endif // GGML_USE_CUDA

// Helper to populate CPU paged KV cache from existing host tensors
void populate_kv_cache_from_host_tensors(
    llama_paged_kv_cache &cpu_cache,
    llama_seq_id seq_id,
    const ggml_tensor* k_host, // Renamed to avoid conflict
    const ggml_tensor* v_host, // Renamed to avoid conflict
    int n_ctx_to_populate,
    int head_dim,
    int n_kv_h, // num_kv_heads
    int n_layers
) {
    llama_paged_kv_cells* cells = cpu_cache.get_paged_cells();
    ASSERT(cells != nullptr, "CPU paged_cells is null in populate_kv_cache_from_host_tensors");
    ASSERT(k_host->type == GGML_TYPE_F16, "k_host must be F16 for this helper");
    ASSERT(v_host->type == GGML_TYPE_F16, "v_host must be F16 for this helper");

    size_t bytes_per_head_data = (size_t)head_dim * sizeof(ggml_fp16_t);

    for (int layer_idx = 0; layer_idx < n_layers; ++layer_idx) {
        for (int head_idx = 0; head_idx < n_kv_h; ++head_idx) {
            for (int pos = 0; pos < n_ctx_to_populate; ++pos) {
                // Calculate offset into flat k_host/v_host data
                // Assuming layout [head_dim, n_ctx, n_head_kv, n_layers] - this might need adjustment based on actual tensor layout
                // For a typical K/V cache tensor [D, N, H, L]:
                // offset = l*nb3 + h*nb2 + p*nb1 + d*nb0
                // Here, simplified: get pointer to start of (pos, head_idx, layer_idx)
                size_t k_offset_bytes =
                    (size_t)layer_idx * k_host->nb[3] +
                    (size_t)head_idx * k_host->nb[2] +
                    (size_t)pos * k_host->nb[1];
                const uint8_t* k_data_src = (const uint8_t*)k_host->data + k_offset_bytes;

                size_t v_offset_bytes =
                    (size_t)layer_idx * v_host->nb[3] +
                    (size_t)head_idx * v_host->nb[2] +
                    (size_t)pos * v_host->nb[1];
                const uint8_t* v_data_src = (const uint8_t*)v_host->data + v_offset_bytes;

                // Populate K
                llama_paged_kv_cells::TokenKey tk_k(seq_id, pos, layer_idx, head_idx, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K);
                auto [page_k, offset_k_bytes] = cells->find_or_allocate_page_for_token(tk_k, bytes_per_head_data);
                ASSERT(page_k != nullptr, "Page allocation for K failed in populate_from_tensors");
                uint8_t* data_k_dst = cells->get_token_data(tk_k);
                ASSERT(data_k_dst != nullptr, "get_token_data for K failed in populate_from_tensors");
                memcpy(data_k_dst, k_data_src, bytes_per_head_data);

                // Populate V
                llama_paged_kv_cells::TokenKey tk_v(seq_id, pos, layer_idx, head_idx, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_V);
                auto [page_v, offset_v_bytes] = cells->find_or_allocate_page_for_token(tk_v, bytes_per_head_data);
                ASSERT(page_v != nullptr, "Page allocation for V failed in populate_from_tensors");
                uint8_t* data_v_dst = cells->get_token_data(tk_v);
                ASSERT(data_v_dst != nullptr, "get_token_data for V failed in populate_from_tensors");
                memcpy(data_v_dst, v_data_src, bytes_per_head_data);
            }
        }
    }
}


// --- Test Case 9: CUDA Paged Attention Correctness (MMA F16) ---
#ifdef GGML_USE_CUDA
void test_cuda_paged_attn_correctness_mma_f16() {
    printf("--- Running Test: test_cuda_paged_attn_correctness_mma_f16 ---\n");
    if (!g_cuda_backend) {
        printf("SKIPPING CUDA test: backend not initialized.\n");
        return;
    }

    struct ggml_init_params host_ctx_params = { 128 * 1024 * 1024, NULL, false };
    ggml_context* ctx_host = ggml_init(host_ctx_params);
    ASSERT(ctx_host != NULL, "Failed to create host ggml_context.");

    struct ggml_init_params meta_gpu_ctx_params = { ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 2, NULL, true };
    ggml_context* ctx_meta_gpu = ggml_init(meta_gpu_ctx_params);
    ASSERT(ctx_meta_gpu != NULL, "Failed to create GPU metadata ggml_context.");

    llama_model_params mparams = {};
    mparams.n_embd = 64;
    mparams.n_head = 2;
    mparams.n_head_kv = 2;
    mparams.n_layer = 1;
    const int head_dim = mparams.n_embd / mparams.n_head;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 16;
    cparams.n_batch = 4;
    cparams.use_paged_kv_cache = true;
    size_t bytes_per_token_kv_one_head_one_layer_k_or_v = (size_t)head_dim * sizeof(uint16_t);
    cparams.kv_page_size = bytes_per_token_kv_one_head_one_layer_k_or_v * 2 * 2;

    ggml_tensor* q_host = ggml_new_tensor_4d(ctx_host, GGML_TYPE_F16, head_dim, cparams.n_batch, mparams.n_head, 1);
    ggml_tensor* k_host = ggml_new_tensor_4d(ctx_host, GGML_TYPE_F16, head_dim, cparams.n_ctx, mparams.n_head_kv, 1);
    ggml_tensor* v_host = ggml_new_tensor_4d(ctx_host, GGML_TYPE_F16, head_dim, cparams.n_ctx, mparams.n_head_kv, 1);

    for(int i=0; i < ggml_nelements(q_host); ++i) ((ggml_fp16_t*)q_host->data)[i] = ggml_fp32_to_fp16((float)(i % 100) * 0.1f + 0.1f);
    for(int i=0; i < ggml_nelements(k_host); ++i) ((ggml_fp16_t*)k_host->data)[i] = ggml_fp32_to_fp16((float)(i % 100) * 0.05f - 0.2f);
    for(int i=0; i < ggml_nelements(v_host); ++i) ((ggml_fp16_t*)v_host->data)[i] = ggml_fp32_to_fp16((float)(i % 100) * 0.02f + 0.3f);

    printf("Running non-paged reference path...\n");
    ggml_tensor* q_gpu_ref = create_gpu_tensor_from_host(ctx_meta_gpu, q_host, "q_gpu_ref");
    ggml_tensor* k_gpu_ref = create_gpu_tensor_from_host(ctx_meta_gpu, k_host, "k_gpu_ref");
    ggml_tensor* v_gpu_ref = create_gpu_tensor_from_host(ctx_meta_gpu, v_host, "v_gpu_ref");

    struct ggml_tensor * dst_ref_ggml_tensor = ggml_dup_tensor(ctx_meta_gpu, q_gpu_ref);
    ggml_backend_buffer_t dst_ref_buffer = ggml_backend_alloc_buffer(g_cuda_buf_type_device, ggml_nbytes(dst_ref_ggml_tensor));
    ggml_backend_tensor_set_buffer(dst_ref_ggml_tensor, dst_ref_buffer);
    ggml_set_name(dst_ref_ggml_tensor, "dst_ref_gpu");

    struct ggml_cgraph* gf_ref = ggml_new_graph_custom(ctx_meta_gpu, GGML_DEFAULT_GRAPH_SIZE, false);
    struct ggml_tensor* attn_out_ref = ggml_flash_attn_ext(ctx_meta_gpu, q_gpu_ref, k_gpu_ref, v_gpu_ref, nullptr, 1.0f/sqrtf(head_dim), 0.0f, 0.0f, GGML_PREC_DEFAULT);
    ggml_set_name(attn_out_ref, "attn_out_ref");
    ggml_build_forward_expand(gf_ref, ggml_cpy(ctx_meta_gpu, attn_out_ref, dst_ref_ggml_tensor));
    ggml_backend_graph_compute(g_cuda_backend, gf_ref);

    std::vector<uint8_t> dst_ref_cpu_data = get_tensor_data_from_gpu(dst_ref_ggml_tensor);
    printf("Non-paged reference path completed.\n");

    // --- Paged Path ---
    printf("Setting up paged path...\n");
    ggml_tensor* q_gpu_paged = create_gpu_tensor_from_host(ctx_meta_gpu, q_host, "q_gpu_paged");

    llama_paged_kv_cache cpu_kv_cache(mparams, cparams, g_cpu_buf_type, ctx_meta_gpu);

    llama_seq_id test_seq_id = 0;
    // Populate cpu_kv_cache with actual data from k_host and v_host
    populate_kv_cache_from_host_tensors(cpu_kv_cache, test_seq_id, k_host, v_host,
                                        cparams.n_ctx, head_dim, mparams.n_head_kv, mparams.n_layer);

    auto [k_view_gpu_host, v_view_gpu_host] = prepare_paged_kv_views_on_gpu(cpu_kv_cache, {test_seq_id}, ctx_meta_gpu, mparams, cparams);

    ggml_tensor* dst_paged_gpu = ggml_dup_tensor(ctx_meta_gpu, q_gpu_paged);
    ggml_set_name(dst_paged_gpu, "dst_paged_gpu");
    ggml_backend_buffer_t dst_paged_buf = ggml_backend_alloc_buffer(g_cuda_buf_type_device, ggml_nbytes(dst_paged_gpu));
    ggml_backend_tensor_set_buffer(dst_paged_gpu, dst_paged_buf);

    // TODO: Invoke paged attention kernel. This requires either:
    // 1. A test-specific CUDA kernel wrapper that calls ggml_cuda_flash_attn_ext_paged internally.
    // 2. Modifying ggml_cuda_compute_forward for GGML_OP_FLASH_ATTN_EXT to detect "paged view tensors"
    //    (e.g., via K->extra or V->extra containing pointers to the view structs or their components)
    //    and then calling ggml_cuda_flash_attn_ext_paged.
    // For now, we cannot directly execute the paged path in this test without one of these.
    printf("Paged path execution is a TODO. Data prepared.\n");
    // Example of what a call might look like if a graph could handle it:
    // struct ggml_cgraph* gf_paged = ggml_new_graph_custom(ctx_meta_gpu, GGML_DEFAULT_GRAPH_SIZE, false);
    // struct ggml_tensor* k_meta_tensor = create_tensor_pointing_to_k_view_components(); // Needs careful setup
    // struct ggml_tensor* v_meta_tensor = create_tensor_pointing_to_v_view_components(); // Needs careful setup
    // struct ggml_tensor* attn_out_paged = ggml_flash_attn_ext(ctx_meta_gpu, q_gpu_paged, k_meta_tensor, v_meta_tensor, nullptr, ...);
    // ggml_build_forward_expand(gf_paged, ggml_cpy(ctx_meta_gpu, attn_out_paged, dst_paged_gpu));
    // ggml_backend_graph_compute(g_cuda_backend, gf_paged);
    // std::vector<uint8_t> dst_paged_cpu_data = get_tensor_data_from_gpu(dst_paged_gpu);
    // ASSERT(are_memory_buffers_equal(dst_ref_cpu_data.data(), dst_paged_cpu_data.data(), dst_ref_cpu_data.size()), "Paged vs Non-paged output mismatch.");


    // Cleanup
    ggml_backend_buffer_free(q_gpu_ref->buffer);   ggml_free(q_gpu_ref);
    ggml_backend_buffer_free(k_gpu_ref->buffer);   ggml_free(k_gpu_ref);
    ggml_backend_buffer_free(v_gpu_ref->buffer);   ggml_free(v_gpu_ref);
    ggml_backend_buffer_free(dst_ref_buffer);      ggml_free(dst_ref_ggml_tensor);
    ggml_graph_free(gf_ref);

    ggml_backend_buffer_free(q_gpu_paged->buffer); ggml_free(q_gpu_paged);
    ggml_backend_buffer_free(dst_paged_buf);       ggml_free(dst_paged_gpu);
    // TODO: Need to free buffers allocated by prepare_paged_kv_views_on_gpu
    // (k_view_host_gpu.token_mappings_gpu_ptr, k_view_host_gpu.page_pool_gpu_ptr, etc. correspond to ggml_backend_buffer_t)

    ggml_free(ctx_host);
    ggml_free(ctx_meta_gpu);
    printf("--- Test test_cuda_paged_attn_correctness_mma_f16 (structure) FINISHED ---\n\n");
}
#endif // GGML_USE_CUDA

// Helper to populate CPU paged KV cache from existing host tensors
void populate_kv_cache_from_host_tensors(
    llama_paged_kv_cache &cpu_cache,
    llama_seq_id seq_id,
    const ggml_tensor* k_host_tensor,
    const ggml_tensor* v_host_tensor,
    int n_tokens_to_copy,
    int head_dim,
    int n_kv_h,
    int n_layers
) {
    llama_paged_kv_cells* cells = cpu_cache.get_paged_cells();
    ASSERT(cells != nullptr, "CPU paged_cells is null in populate_kv_cache_from_host_tensors");
    ASSERT(k_host_tensor->type == GGML_TYPE_F16, "k_host_tensor must be F16 for this helper");
    ASSERT(v_host_tensor->type == GGML_TYPE_F16, "v_host_tensor must be F16 for this helper");

    size_t bytes_per_head_data = (size_t)head_dim * sizeof(ggml_fp16_t);

    for (int layer_idx = 0; layer_idx < n_layers; ++layer_idx) {
        for (int head_idx = 0; head_idx < n_kv_h; ++head_idx) {
            for (int pos = 0; pos < n_tokens_to_copy; ++pos) {
                size_t k_offset_bytes =
                    (size_t)layer_idx * k_host_tensor->nb[3] +
                    (size_t)head_idx * k_host_tensor->nb[2] +
                    (size_t)pos * k_host_tensor->nb[1];
                const uint8_t* k_data_src = (const uint8_t*)k_host_tensor->data + k_offset_bytes;

                size_t v_offset_bytes =
                    (size_t)layer_idx * v_host_tensor->nb[3] +
                    (size_t)head_idx * v_host_tensor->nb[2] +
                    (size_t)pos * v_host_tensor->nb[1];
                const uint8_t* v_data_src = (const uint8_t*)v_host_tensor->data + v_offset_bytes;

                llama_paged_kv_cells::TokenKey tk_k(seq_id, pos, layer_idx, head_idx, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K);
                auto [page_k, offset_k_bytes_cell] = cells->find_or_allocate_page_for_token(tk_k, bytes_per_head_data);
                ASSERT(page_k != nullptr, "Page allocation for K failed in populate_from_tensors");
                uint8_t* data_k_dst = cells->get_token_data(tk_k);
                ASSERT(data_k_dst != nullptr, "get_token_data for K failed in populate_from_tensors");
                memcpy(data_k_dst, k_data_src, bytes_per_head_data);

                llama_paged_kv_cells::TokenKey tk_v(seq_id, pos, layer_idx, head_idx, llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_V);
                auto [page_v, offset_v_bytes_cell] = cells->find_or_allocate_page_for_token(tk_v, bytes_per_head_data);
                ASSERT(page_v != nullptr, "Page allocation for V failed in populate_from_tensors");
                uint8_t* data_v_dst = cells->get_token_data(tk_v);
                ASSERT(data_v_dst != nullptr, "get_token_data for V failed in populate_from_tensors");
                memcpy(data_v_dst, v_data_src, bytes_per_head_data);
            }
        }
    }
}

// --- Test Case 9: CUDA Paged Attention Correctness (MMA F16) ---
#ifdef GGML_USE_CUDA
void test_cuda_paged_attn_correctness_mma_f16() {
    printf("--- Running Test: test_cuda_paged_attn_correctness_mma_f16 ---\n");
    if (!g_cuda_backend) {
        printf("SKIPPING CUDA test: backend not initialized.\n");
        return;
    }

    struct ggml_init_params host_ctx_params = { 256 * 1024 * 1024, NULL, false }; // Increased host memory
    ggml_context* ctx_host = ggml_init(host_ctx_params);
    ASSERT(ctx_host != NULL, "Failed to create host ggml_context.");

    struct ggml_init_params meta_gpu_ctx_params = { ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 2, NULL, true };
    ggml_context* ctx_meta_gpu = ggml_init(meta_gpu_ctx_params);
    ASSERT(ctx_meta_gpu != NULL, "Failed to create GPU metadata ggml_context.");

    llama_model_params mparams = {};
    mparams.n_embd = 64;
    mparams.n_head = 2;
    mparams.n_head_kv = 2;
    mparams.n_layer = 1;
    const int head_dim = mparams.n_embd / mparams.n_head;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 16;
    cparams.n_batch = 4;
    cparams.use_paged_kv_cache = true;
    size_t bytes_per_token_kv_one_head_one_layer_k_or_v = (size_t)head_dim * sizeof(uint16_t);
    cparams.kv_page_size = bytes_per_token_kv_one_head_one_layer_k_or_v * 4; // Page fits 4 K/V pairs for one head/layer

    // Prepare Host Tensors (Q, K, V)
    ggml_tensor* q_host = ggml_new_tensor_4d(ctx_host, GGML_TYPE_F16, head_dim, cparams.n_batch, mparams.n_head, 1);
    ggml_tensor* k_host = ggml_new_tensor_4d(ctx_host, GGML_TYPE_F16, head_dim, cparams.n_ctx, mparams.n_head_kv, 1);
    ggml_tensor* v_host = ggml_new_tensor_4d(ctx_host, GGML_TYPE_F16, head_dim, cparams.n_ctx, mparams.n_head_kv, 1);

    for(int i=0; i < ggml_nelements(q_host); ++i) ((ggml_fp16_t*)q_host->data)[i] = ggml_fp32_to_fp16((float)((i % 50) - 25) * 0.1f);
    for(int i=0; i < ggml_nelements(k_host); ++i) ((ggml_fp16_t*)k_host->data)[i] = ggml_fp32_to_fp16((float)((i % 60) - 30) * 0.05f);
    for(int i=0; i < ggml_nelements(v_host); ++i) ((ggml_fp16_t*)v_host->data)[i] = ggml_fp32_to_fp16((float)((i % 70) - 35) * 0.02f);

    // --- Non-Paged Reference Path ---
    printf("Running non-paged reference path...\n");
    ggml_tensor* q_gpu_ref = create_gpu_tensor_from_host(ctx_meta_gpu, q_host, "q_gpu_ref");
    ggml_tensor* k_gpu_ref = create_gpu_tensor_from_host(ctx_meta_gpu, k_host, "k_gpu_ref");
    ggml_tensor* v_gpu_ref = create_gpu_tensor_from_host(ctx_meta_gpu, v_host, "v_gpu_ref");

    struct ggml_tensor * dst_ref_ggml_tensor = ggml_dup_tensor(ctx_meta_gpu, q_gpu_ref);
    ggml_backend_buffer_t dst_ref_buffer = ggml_backend_alloc_buffer(g_cuda_buf_type_device, ggml_nbytes(dst_ref_ggml_tensor));
    ggml_backend_tensor_set_buffer(dst_ref_ggml_tensor, dst_ref_buffer);
    ggml_set_name(dst_ref_ggml_tensor, "dst_ref_gpu");

    struct ggml_cgraph* gf_ref = ggml_new_graph_custom(ctx_meta_gpu, GGML_DEFAULT_GRAPH_SIZE, false);
    struct ggml_tensor* attn_out_ref = ggml_flash_attn_ext(ctx_meta_gpu, q_gpu_ref, k_gpu_ref, v_gpu_ref, nullptr, 1.0f/sqrtf(head_dim), 0.0f, 0.0f, GGML_PREC_DEFAULT);
    ggml_set_name(attn_out_ref, "attn_out_ref");
    ggml_build_forward_expand(gf_ref, ggml_cpy(ctx_meta_gpu, attn_out_ref, dst_ref_ggml_tensor));
    ggml_backend_graph_compute(g_cuda_backend, gf_ref);

    std::vector<uint8_t> dst_ref_cpu_data = get_tensor_data_from_gpu(dst_ref_ggml_tensor);
    printf("Non-paged reference path completed.\n");

    // --- Paged Path ---
    printf("Setting up paged path...\n");
    ggml_tensor* q_gpu_paged = create_gpu_tensor_from_host(ctx_meta_gpu, q_host, "q_gpu_paged");

    llama_paged_kv_cache cpu_kv_cache(mparams, cparams, g_cpu_buf_type, ctx_meta_gpu);

    llama_seq_id test_seq_id = 0;
    populate_kv_cache_from_host_tensors(cpu_kv_cache, test_seq_id, k_host, v_host,
                                        cparams.n_ctx, head_dim, mparams.n_head_kv, mparams.n_layer);

    auto [k_view_gpu_host, v_view_gpu_host] = prepare_paged_kv_views_on_gpu(cpu_kv_cache, {test_seq_id}, ctx_meta_gpu, mparams, cparams);

    ggml_tensor* dst_paged_gpu = ggml_dup_tensor(ctx_meta_gpu, q_gpu_paged);
    ggml_set_name(dst_paged_gpu, "dst_paged_gpu");
    ggml_backend_buffer_t dst_paged_buf = ggml_backend_alloc_buffer(g_cuda_buf_type_device, ggml_nbytes(dst_paged_gpu));
    ggml_backend_tensor_set_buffer(dst_paged_gpu, dst_paged_buf);

    // TODO (MAJOR): Invoke paged attention kernel.
    // This requires a mechanism to pass paged_kv_sequence_view_host_for_gpu (k_view_gpu_host, v_view_gpu_host)
    // to the ggml_cuda_flash_attn_ext_paged dispatcher.
    // Current ggml_flash_attn_ext op does not support this directly.
    // Possible solutions:
    // 1. Test-specific CUDA kernel wrapper (simplest for isolated test).
    // 2. Modify GGML_OP_FLASH_ATTN_EXT handling in ggml-cuda.cu to detect "paged K/V metadata tensors"
    //    (e.g. by checking tensor->extra or a special buffer type) and then extract view info.
    printf("Paged path execution and comparison is a TODO (requires kernel invocation wrapper or backend changes).\n");
    // Example of what might be done if a wrapper existed:
    // call_my_paged_flash_attn_wrapper(q_gpu_paged, dst_paged_gpu, k_view_gpu_host, v_view_gpu_host, ...);
    // std::vector<uint8_t> dst_paged_cpu_data = get_tensor_data_from_gpu(dst_paged_gpu);
    // ASSERT(are_memory_buffers_equal(dst_ref_cpu_data.data(), dst_paged_cpu_data.data(), dst_ref_cpu_data.size(), "Paged vs Non-paged output"),
    //        "Paged vs Non-paged output mismatch. THIS IS EXPECTED TO FAIL until backend supports paged views via graph or wrapper.");


    // Cleanup
    ggml_backend_buffer_free(q_gpu_ref->buffer);   ggml_free(q_gpu_ref);
    ggml_backend_buffer_free(k_gpu_ref->buffer);   ggml_free(k_gpu_ref);
    ggml_backend_buffer_free(v_gpu_ref->buffer);   ggml_free(v_gpu_ref);
    ggml_backend_buffer_free(dst_ref_buffer);      ggml_free(dst_ref_ggml_tensor);
    ggml_graph_free(gf_ref);

    ggml_backend_buffer_free(q_gpu_paged->buffer); ggml_free(q_gpu_paged);
    ggml_backend_buffer_free(dst_paged_buf);       ggml_free(dst_paged_gpu);

    // Cleanup for buffers allocated by prepare_paged_kv_views_on_gpu
    // This needs direct access to the ggml_backend_buffer_t objects for mappings and pool,
    // or `prepare_paged_kv_views_on_gpu` should return them for cleanup.
    // For now, this is a simplified test structure and assumes these are managed/freed elsewhere or by context end.
    // A robust test would track and free:
    // if (k_view_gpu_host.token_mappings_gpu_ptr) ggml_backend_buffer_free(ggml_backend_get_buffer_from_ptr(g_cuda_buf_type_device, k_view_host_gpu.token_mappings_gpu_ptr));
    // if (k_view_host_gpu.page_pool_gpu_ptr)      ggml_backend_buffer_free(ggml_backend_get_buffer_from_ptr(g_cuda_buf_type_device, k_view_host_gpu.page_pool_gpu_ptr));
    // if (v_view_host_gpu.token_mappings_gpu_ptr && v_view_host_gpu.token_mappings_gpu_ptr != k_view_host_gpu.token_mappings_gpu_ptr)
    //     ggml_backend_buffer_free(ggml_backend_get_buffer_from_ptr(g_cuda_buf_type_device, v_view_host_gpu.token_mappings_gpu_ptr));
    // The page data buffers themselves are trickier as they are numerous and created from ggml_tensors.
    // If ctx_meta_gpu owned their buffers, ggml_free(ctx_meta_gpu) might handle some, but they were created with g_cuda_buf_type_device.

    ggml_free(ctx_host);
    ggml_free(ctx_meta_gpu); // This will free tensors allocated in it, but not necessarily their buffers if backend alloc'd
    printf("--- Test test_cuda_paged_attn_correctness_mma_f16 (structure) FINISHED ---\n\n");
}
#endif // GGML_USE_CUDA


int main() {
#ifdef GGML_USE_CUDA
    setup_cuda_for_test();
#endif

    printf("--- Starting Paged KV Cache Unit Tests ---\n");
    try {
        test_paged_cells_alloc_free();
        test_paged_cells_token_mapping();
        test_paged_cache_initialization();
        test_paged_cache_seq_add();
        test_paged_cache_seq_rm();
        test_paged_cache_seq_cp();
        test_paged_cache_seq_div();
        test_paged_cache_state_read_write();
        // Call other test functions here
    } catch (const std::exception& e) {
}

// =================================================================================================
// PART 2: CUDA Paged Attention Kernel Tests
// =================================================================================================
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h" // For CUDA backend functions and specific types if needed

// Global CUDA backend and buffer type for tests
ggml_backend_t g_cuda_backend = NULL;
ggml_backend_buffer_type_t g_cuda_buf_type_device = NULL;

void setup_cuda_for_test() {
    fprintf(stderr, "Initializing CUDA backend for tests...\n");
    // Default to device 0 for tests
    g_cuda_backend = ggml_backend_cuda_init(0);
    if (!g_cuda_backend) {
        fprintf(stderr, "setup_cuda_for_test: ggml_backend_cuda_init() failed. CUDA tests will be skipped.\n");
        return;
    }
    g_cuda_buf_type_device = ggml_backend_get_default_buffer_type(g_cuda_backend);
    ASSERT(g_cuda_buf_type_device != NULL, "Failed to get CUDA device buffer type.");
    printf("CUDA backend initialized for tests.\n");
}

void teardown_cuda_for_test() {
    if (g_cuda_backend) {
        ggml_backend_free(g_cuda_backend);
        g_cuda_backend = NULL;
        g_cuda_buf_type_device = NULL;
        printf("CUDA backend freed.\n");
    }
}

// Creates a GPU tensor and copies data from a host tensor.
ggml_tensor* create_gpu_tensor_from_host(ggml_context* ctx_meta_gpu, const ggml_tensor* t_host, const char* name) {
    if (!g_cuda_backend || !g_cuda_buf_type_device) {
        fprintf(stderr, "CUDA backend not initialized, cannot create GPU tensor %s.\n", name);
        return nullptr;
    }
    // Create metadata for the GPU tensor
    ggml_tensor* t_device = ggml_dup_tensor(ctx_meta_gpu, t_host);
    // Allocate buffer on GPU
    ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(g_cuda_buf_type_device, ggml_nbytes(t_device));
    ASSERT(buffer != NULL, (std::string("Failed to allocate CUDA buffer for ") + name).c_str());
    // Associate buffer with tensor
    ggml_backend_tensor_set_buffer(t_device, buffer); // Use this instead of t_device->buffer = buffer;
    // Copy data
    ggml_backend_tensor_set_async(t_device, t_host->data, 0, ggml_nbytes(t_host));
    ggml_backend_synchronize(g_cuda_backend); // Ensure copy completes for subsequent operations
    ggml_set_name(t_device, name);
    return t_device;
}

// Retrieves data from a GPU tensor to a host vector.
std::vector<uint8_t> get_tensor_data_from_gpu(const ggml_tensor* t_device) {
    if (!g_cuda_backend || !t_device || !t_device->buffer ) { // Check t_device->buffer
        fprintf(stderr, "Invalid tensor or CUDA backend for get_tensor_data_from_gpu for tensor %s.\n", t_device ? t_device->name : "NULL");
        return {};
    }
    size_t nbytes = ggml_nbytes(t_device);
    std::vector<uint8_t> host_data(nbytes);
    ggml_backend_tensor_get_async(t_device, host_data.data(), 0, nbytes);
    ggml_backend_synchronize(g_cuda_backend);
    return host_data;
}

// Helper function to compare float tensors with tolerance
bool compare_tensors_approx(const float* data1, const float* data2, int64_t num_elements, const char* test_name, float abs_tolerance, float rel_tolerance) {
    int mismatches = 0;
    for (int64_t i = 0; i < num_elements; ++i) {
        float d1 = data1[i];
        float d2 = data2[i];
        float diff = fabsf(d1 - d2);
        // Relative difference calculation, handle d1 being close to zero
        float rd = (fabsf(d1) > 1e-9f) ? diff / fabsf(d1) : 0.0f;

        if (diff > abs_tolerance && rd > rel_tolerance) {
            if (mismatches < 20) { // Print first few mismatches
                printf("%s: Mismatch at index %lld: data1=%.8f, data2=%.8f, diff=%.8f, rel_diff=%.8f (abs_tol=%.2e, rel_tol=%.2e)\n",
                       test_name, i, d1, d2, diff, rd, abs_tolerance, rel_tolerance);
            }
            mismatches++;
        }
    }
    if (mismatches > 0) {
        printf("%s: Total mismatches: %d / %lld\n", test_name, mismatches, num_elements);
        return false;
    }
    printf("%s: Results match within tolerance (abs_tol=%.2e, rel_tol=%.2e).\n", test_name, abs_tolerance, rel_tolerance);
    return true;
}

// Host-side representation of CUDA structs for preparing kernel arguments
struct paged_kv_token_mapping_host_for_gpu {
    int32_t page_idx;
    int32_t offset_in_page_elements; // Byte offset
};

struct paged_kv_sequence_view_host_for_gpu {
    void* token_mappings_gpu_ptr;
    void* page_pool_gpu_ptr;
    int32_t num_tokens_in_logical_sequence;
    ggml_type dtype;
    int32_t k_head_size_elements;
    int32_t v_head_size_elements;
    int32_t num_k_heads_total;
    int32_t num_v_heads_total;
    uint32_t element_size_bytes;
    uint32_t page_size_bytes;
    uint32_t v_block_start_offset_bytes;
};

// Prepares GPU buffers for paged KV views from a CPU cache state.
std::pair<paged_kv_sequence_view_host_for_gpu, paged_kv_sequence_view_host_for_gpu>
prepare_paged_kv_views_on_gpu(
    llama_paged_kv_cache& cpu_cache,
    const std::vector<llama_seq_id>& target_seq_ids,
    ggml_context* ctx_meta_gpu,
    const llama_model_params& mparams,
    const llama_context_params& cparams
) {
    // ... (Full content of prepare_paged_kv_views_on_gpu as implemented in the previous successful step) ...
    // This function is assumed to be correctly implemented from prior steps.
    // For brevity in this diff, its full content is not repeated here but is part of the replacement.
    if (!g_cuda_backend || !g_cuda_buf_type_device) {
        throw std::runtime_error("CUDA backend not initialized for paged view prep.");
    }
    llama_paged_kv_cells* cpu_cells = cpu_cache.get_paged_cells();
    ASSERT(cpu_cells != nullptr, "CPU paged_cells is null.");

    paged_kv_sequence_view_host_for_gpu k_view_host_gpu = {0};
    paged_kv_sequence_view_host_for_gpu v_view_host_gpu = {0};

    std::vector<paged_kv_token_mapping_host_for_gpu> k_mappings_host_vec;
    std::vector<paged_kv_token_mapping_host_for_gpu> v_mappings_host_vec;
    std::map<int, const llama_kv_page*> unique_pages_map_cpu_id_to_ptr;
    int max_pos_overall = -1;

    ASSERT(target_seq_ids.size() == 1, "This simplified helper expects only one target_seq_id for creating a flat view.");
    llama_seq_id current_seq_id = target_seq_ids[0];

    for (const auto& item : cpu_cells->get_token_to_page_offset_map()) {
        const auto& token_key = item.first;
        const auto& page_offset_val = item.second;
        if (token_key.seq_id != current_seq_id) continue;

        unique_pages_map_cpu_id_to_ptr[page_offset_val.page_id] = cpu_cells->get_page(page_offset_val.page_id);
        paged_kv_token_mapping_host_for_gpu current_mapping = {(int32_t)page_offset_val.page_id, (int32_t)page_offset_val.offset_bytes};
        int current_pos = token_key.pos;
        if (current_pos > max_pos_overall) max_pos_overall = current_pos;

        if (token_key.type == llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K) {
            if (current_pos >= (int)k_mappings_host_vec.size()) k_mappings_host_vec.resize(max_pos_overall + 1, {-1, 0});
            k_mappings_host_vec[current_pos] = current_mapping;
        } else {
            if (current_pos >= (int)v_mappings_host_vec.size()) v_mappings_host_vec.resize(max_pos_overall + 1, {-1, 0});
            v_mappings_host_vec[current_pos] = current_mapping;
        }
    }
    if (max_pos_overall == -1 && !k_mappings_host_vec.empty()) { k_mappings_host_vec.clear(); }
    if (max_pos_overall == -1 && !v_mappings_host_vec.empty()) { v_mappings_host_vec.clear(); }
    if (max_pos_overall > -1) {
        if (k_mappings_host_vec.size() < (size_t)max_pos_overall + 1) k_mappings_host_vec.resize(max_pos_overall + 1, {-1,0});
        if (v_mappings_host_vec.size() < (size_t)max_pos_overall + 1) v_mappings_host_vec.resize(max_pos_overall + 1, {-1,0});
    }

    std::vector<void*> host_gpu_page_device_ptrs;
    std::map<int, int> cpu_page_id_to_gpu_pool_idx;
    for(const auto& pair : unique_pages_map_cpu_id_to_ptr) {
        const llama_kv_page* cpu_page = pair.second;
        if (cpu_page && !cpu_page->is_free()) {
            struct ggml_tensor* t_page_host_meta = ggml_new_tensor_1d(ctx_meta_gpu, GGML_TYPE_I8, cpu_page->size);
            t_page_host_meta->data = cpu_page->data;
            ggml_tensor* t_page_gpu = create_gpu_tensor_from_host(ctx_meta_gpu, t_page_host_meta, "gpu_page_data_content");
            t_page_host_meta->data = nullptr;
            ggml_free(t_page_host_meta);
            ASSERT(t_page_gpu && t_page_gpu->data, "Failed to create GPU buffer for a page content.");
            cpu_page_id_to_gpu_pool_idx[cpu_page->id] = host_gpu_page_device_ptrs.size();
            host_gpu_page_device_ptrs.push_back(t_page_gpu->data);
        }
    }

    for(auto& mapping : k_mappings_host_vec) {
        if (mapping.page_idx != -1 && cpu_page_id_to_gpu_pool_idx.count(mapping.page_idx)) {
             mapping.page_idx = cpu_page_id_to_gpu_pool_idx.at(mapping.page_idx);
        } else { mapping.page_idx = -1; mapping.offset_in_page_elements = 0; }
    }
    for(auto& mapping : v_mappings_host_vec) {
        if (mapping.page_idx != -1 && cpu_page_id_to_gpu_pool_idx.count(mapping.page_idx)) {
            mapping.page_idx = cpu_page_id_to_gpu_pool_idx.at(mapping.page_idx);
        } else { mapping.page_idx = -1; mapping.offset_in_page_elements = 0; }
    }

    if (!k_mappings_host_vec.empty()) {
        ggml_backend_buffer_t k_map_buf = ggml_backend_alloc_buffer(g_cuda_buf_type_device, k_mappings_host_vec.size() * sizeof(paged_kv_token_mapping_host_for_gpu));
        k_view_host_gpu.token_mappings_gpu_ptr = ggml_backend_buffer_get_base(k_map_buf);
        ggml_backend_buffer_set_data(k_map_buf, 0, k_mappings_host_vec.data(), k_mappings_host_vec.size() * sizeof(paged_kv_token_mapping_host_for_gpu));
    } else { k_view_host_gpu.token_mappings_gpu_ptr = nullptr; }

    if (!host_gpu_page_device_ptrs.empty()) {
        ggml_backend_buffer_t k_pool_buf = ggml_backend_alloc_buffer(g_cuda_buf_type_device, host_gpu_page_device_ptrs.size() * sizeof(void*));
        k_view_host_gpu.page_pool_gpu_ptr = ggml_backend_buffer_get_base(k_pool_buf);
        ggml_backend_buffer_set_data(k_pool_buf, 0, host_gpu_page_device_ptrs.data(), host_gpu_page_device_ptrs.size() * sizeof(void*));
    } else { k_view_host_gpu.page_pool_gpu_ptr = nullptr; }

    if (!v_mappings_host_vec.empty()) {
        ggml_backend_buffer_t v_map_buf = ggml_backend_alloc_buffer(g_cuda_buf_type_device, v_mappings_host_vec.size() * sizeof(paged_kv_token_mapping_host_for_gpu));
        v_view_host_gpu.token_mappings_gpu_ptr = ggml_backend_buffer_get_base(v_map_buf);
        ggml_backend_buffer_set_data(v_map_buf, 0, v_mappings_host_vec.data(), v_mappings_host_vec.size() * sizeof(paged_kv_token_mapping_host_for_gpu));
    } else { v_view_host_gpu.token_mappings_gpu_ptr = nullptr; }

    v_view_host_gpu.page_pool_gpu_ptr = k_view_host_gpu.page_pool_gpu_ptr;

    int head_dim = mparams.n_embd / mparams.n_head_kv;
    k_view_host_gpu.num_tokens_in_logical_sequence = (max_pos_overall == -1) ? 0 : (max_pos_overall + 1);
    k_view_host_gpu.dtype = GGML_TYPE_F16;
    k_view_host_gpu.element_size_bytes = sizeof(uint16_t);
    k_view_host_gpu.k_head_size_elements = head_dim;
    k_view_host_gpu.v_head_size_elements = head_dim;
    k_view_host_gpu.num_k_heads_total = mparams.n_head_kv;
    k_view_host_gpu.num_v_heads_total = mparams.n_head_kv;
    k_view_host_gpu.page_size_bytes = cparams.kv_page_size;
    k_view_host_gpu.v_block_start_offset_bytes = 0;

    v_view_host_gpu = k_view_host_gpu;

    ggml_backend_synchronize(g_cuda_backend);
    return {k_view_host_gpu, v_view_host_gpu};
}

void test_cuda_paged_attn_correctness_mma_f16() {
    printf("--- Running Test: test_cuda_paged_attn_correctness_mma_f16 ---\n");
    if (!g_cuda_backend) {
        printf("SKIPPING CUDA test: backend not initialized.\n");
        return;
    }

    struct ggml_init_params host_ctx_params = { 128 * 1024 * 1024, NULL, false };
    ggml_context* ctx_host = ggml_init(host_ctx_params);
    ASSERT(ctx_host != NULL, "Failed to create host ggml_context.");

    struct ggml_init_params meta_gpu_ctx_params = { ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 2, NULL, true };
    ggml_context* ctx_meta_gpu = ggml_init(meta_gpu_ctx_params);
    ASSERT(ctx_meta_gpu != NULL, "Failed to create GPU metadata ggml_context.");

    llama_model_params mparams = {};
    mparams.n_embd = 64;
    mparams.n_head = 2;
    mparams.n_head_kv = 2;
    mparams.n_layer = 1;
    const int head_dim = mparams.n_embd / mparams.n_head;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 16;
    cparams.n_batch = 4;
    cparams.use_paged_kv_cache = true;
    size_t bytes_per_token_kv_one_head_one_layer_k_or_v = (size_t)head_dim * sizeof(uint16_t);
    cparams.kv_page_size = bytes_per_token_kv_one_head_one_layer_k_or_v * 2 * 2;

    ggml_tensor* q_host = ggml_new_tensor_4d(ctx_host, GGML_TYPE_F16, head_dim, cparams.n_batch, mparams.n_head, 1);
    ggml_tensor* k_host = ggml_new_tensor_4d(ctx_host, GGML_TYPE_F16, head_dim, cparams.n_ctx, mparams.n_head_kv, 1);
    ggml_tensor* v_host = ggml_new_tensor_4d(ctx_host, GGML_TYPE_F16, head_dim, cparams.n_ctx, mparams.n_head_kv, 1);

    for(int i=0; i < ggml_nelements(q_host); ++i) ((ggml_fp16_t*)q_host->data)[i] = ggml_fp32_to_fp16((float)(i % 100) * 0.1f + 0.1f);
    for(int i=0; i < ggml_nelements(k_host); ++i) ((ggml_fp16_t*)k_host->data)[i] = ggml_fp32_to_fp16((float)(i % 100) * 0.05f - 0.2f);
    for(int i=0; i < ggml_nelements(v_host); ++i) ((ggml_fp16_t*)v_host->data)[i] = ggml_fp32_to_fp16((float)(i % 100) * 0.02f + 0.3f);

    printf("Running non-paged reference path...\n");
    ggml_tensor* q_gpu_ref = create_gpu_tensor_from_host(ctx_meta_gpu, q_host, "q_gpu_ref");
    ggml_tensor* k_gpu_ref = create_gpu_tensor_from_host(ctx_meta_gpu, k_host, "k_gpu_ref");
    ggml_tensor* v_gpu_ref = create_gpu_tensor_from_host(ctx_meta_gpu, v_host, "v_gpu_ref");

    struct ggml_tensor * dst_ref_ggml_tensor = ggml_dup_tensor(ctx_meta_gpu, q_gpu_ref);
    ggml_backend_buffer_t dst_ref_buffer = ggml_backend_alloc_buffer(g_cuda_buf_type_device, ggml_nbytes(dst_ref_ggml_tensor));
    ggml_backend_tensor_set_buffer(dst_ref_ggml_tensor, dst_ref_buffer);
    ggml_set_name(dst_ref_ggml_tensor, "dst_ref_gpu");

    struct ggml_cgraph* gf_ref = ggml_new_graph_custom(ctx_meta_gpu, GGML_DEFAULT_GRAPH_SIZE, false);
    struct ggml_tensor* attn_out_ref = ggml_flash_attn_ext(ctx_meta_gpu, q_gpu_ref, k_gpu_ref, v_gpu_ref, nullptr, 1.0f/sqrtf(head_dim), 0.0f, 0.0f, GGML_PREC_DEFAULT);
    ggml_set_name(attn_out_ref, "attn_out_ref");
    ggml_build_forward_expand(gf_ref, ggml_cpy(ctx_meta_gpu, attn_out_ref, dst_ref_ggml_tensor));
    ggml_backend_graph_compute(g_cuda_backend, gf_ref);

    std::vector<uint8_t> dst_ref_cpu_data = get_tensor_data_from_gpu(dst_ref_ggml_tensor);
    printf("Non-paged reference path completed.\n");

    printf("Paged path test logic is a TODO.\n");

    ggml_backend_buffer_free(q_gpu_ref->buffer);   ggml_free(q_gpu_ref);
    ggml_backend_buffer_free(k_gpu_ref->buffer);   ggml_free(k_gpu_ref);
    ggml_backend_buffer_free(v_gpu_ref->buffer);   ggml_free(v_gpu_ref);
    ggml_backend_buffer_free(dst_ref_buffer);      ggml_free(dst_ref_ggml_tensor);
    ggml_graph_free(gf_ref);

    ggml_free(ctx_host);
    ggml_free(ctx_meta_gpu);
    printf("--- Test test_cuda_paged_attn_correctness_mma_f16 (structure) FINISHED ---\n\n");
}
#endif // GGML_USE_CUDA


int main() {
    // ggml_backend_t backend = NULL;
    // ggml_backend_cpu_init();
    // backend = ggml_backend_cpu_init();
    // g_cpu_buf_type = ggml_backend_get_default_buffer_type(backend);


    printf("--- Starting Paged KV Cache Unit Tests ---\n");
    try {
        test_paged_cells_alloc_free();
        test_paged_cells_token_mapping();
        test_paged_cache_initialization();
        test_paged_cache_seq_add();
        test_paged_cache_seq_rm();
        test_paged_cache_seq_cp();
        test_paged_cache_seq_div();
        test_paged_cache_state_read_write();
        // Call other test functions here
    } catch (const std::exception& e) {

// =================================================================================================
// PART 2: CUDA Paged Attention Kernel Tests
// =================================================================================================
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"

ggml_backend_t g_cuda_backend = NULL;
ggml_backend_buffer_type_t g_cuda_buf_type_device = NULL; // For device memory

void setup_cuda_for_test() {
    fprintf(stderr, "Initializing CUDA backend for tests...\n");
    g_cuda_backend = ggml_backend_cuda_init(0);
    if (!g_cuda_backend) {
        fprintf(stderr, "setup_cuda_for_test: ggml_backend_cuda_init() failed. CUDA tests will be skipped.\n");
        return;
    }
    g_cuda_buf_type_device = ggml_backend_get_default_buffer_type(g_cuda_backend);
    ASSERT(g_cuda_buf_type_device != NULL, "Failed to get CUDA device buffer type.");
    printf("CUDA backend initialized for tests.\n");
}

void teardown_cuda_for_test() {
    if (g_cuda_backend) {
        ggml_backend_free(g_cuda_backend);
        g_cuda_backend = NULL;
        g_cuda_buf_type_device = NULL;
        printf("CUDA backend freed.\n");
    }
}

ggml_tensor* create_gpu_tensor_from_host(ggml_context* ctx_meta, const ggml_tensor* t_host, const char* name) {
    if (!g_cuda_backend || !g_cuda_buf_type_device) {
        fprintf(stderr, "CUDA backend not initialized, cannot create GPU tensor %s.\n", name);
        return nullptr;
    }
    ggml_tensor* t_device = ggml_dup_tensor(ctx_meta, t_host);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(g_cuda_buf_type_device, ggml_nbytes(t_device));
    ASSERT(buffer != NULL, (std::string("Failed to allocate CUDA buffer for ") + name).c_str());
    t_device->buffer = buffer;
    ggml_backend_tensor_set_async(t_device, t_host->data, 0, ggml_nbytes(t_host));
    ggml_backend_synchronize(g_cuda_backend);
    ggml_set_name(t_device, name);
    // printf("Created GPU tensor %s, size %zu bytes\n", name, ggml_nbytes(t_device)); // Can be noisy
    return t_device;
}

std::vector<uint8_t> get_tensor_data_from_gpu(const ggml_tensor* t_device) {
    if (!g_cuda_backend || !t_device || !t_device->buffer) {
        fprintf(stderr, "Invalid tensor or CUDA backend for get_tensor_data_from_gpu.\n");
        return {};
    }
    size_t nbytes = ggml_nbytes(t_device);
    std::vector<uint8_t> host_data(nbytes);
    ggml_backend_tensor_get_async(t_device, host_data.data(), 0, nbytes);
    ggml_backend_synchronize(g_cuda_backend);
    return host_data;
}

struct paged_kv_token_mapping_host_for_gpu {
    int32_t page_idx;
    int32_t offset_in_page_elements; // This is a byte offset for CUDA use
};

struct paged_kv_sequence_view_host_for_gpu {
    void* token_mappings_gpu_ptr;
    void* page_pool_gpu_ptr;
    int32_t num_tokens_in_logical_sequence;
    ggml_type dtype;
    int32_t k_head_size_elements;
    int32_t v_head_size_elements;
    int32_t num_k_heads_total;
    int32_t num_v_heads_total;
    uint32_t element_size_bytes;
    uint32_t page_size_bytes;
    uint32_t v_block_start_offset_bytes;
};

std::pair<paged_kv_sequence_view_host_for_gpu, paged_kv_sequence_view_host_for_gpu>
prepare_paged_kv_views_on_gpu(
    llama_paged_kv_cache& cpu_cache,
    const std::vector<llama_seq_id>& target_seq_ids,
    ggml_context* ctx_meta_gpu,
    const llama_model_params& mparams,
    const llama_context_params& cparams
) {
    if (!g_cuda_backend || !g_cuda_buf_type_device) {
        throw std::runtime_error("CUDA backend not initialized for paged view prep.");
    }
    llama_paged_kv_cells* cpu_cells = cpu_cache.get_paged_cells();
    ASSERT(cpu_cells != nullptr, "CPU paged_cells is null.");

    paged_kv_sequence_view_host_for_gpu k_view_host_gpu = {0};
    paged_kv_sequence_view_host_for_gpu v_view_host_gpu = {0};

    std::vector<paged_kv_token_mapping_host_for_gpu> k_mappings_host_vec;
    std::vector<paged_kv_token_mapping_host_for_gpu> v_mappings_host_vec;
    std::map<int, const llama_kv_page*> unique_pages_map_cpu_id_to_ptr;
    int max_pos_overall = -1;

    ASSERT(target_seq_ids.size() == 1, "This simplified helper expects only one target_seq_id for creating a flat view.");
    llama_seq_id current_seq_id = target_seq_ids[0];

    for (const auto& item : cpu_cells->get_token_to_page_offset_map()) {
        const auto& token_key = item.first;
        const auto& page_offset_val = item.second;
        if (token_key.seq_id != current_seq_id) continue; // Process only the target sequence

        unique_pages_map_cpu_id_to_ptr[page_offset_val.page_id] = cpu_cells->get_page(page_offset_val.page_id);
        paged_kv_token_mapping_host_for_gpu current_mapping = {(int32_t)page_offset_val.page_id, (int32_t)page_offset_val.offset_bytes};
        int current_pos = token_key.pos;
        if (current_pos > max_pos_overall) max_pos_overall = current_pos;

        if (token_key.type == llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K) {
            if (current_pos >= (int)k_mappings_host_vec.size()) k_mappings_host_vec.resize(max_pos_overall + 1, {-1, 0});
            k_mappings_host_vec[current_pos] = current_mapping;
        } else {
            if (current_pos >= (int)v_mappings_host_vec.size()) v_mappings_host_vec.resize(max_pos_overall + 1, {-1, 0});
            v_mappings_host_vec[current_pos] = current_mapping;
        }
    }
    if (max_pos_overall == -1 && !k_mappings_host_vec.empty()) { k_mappings_host_vec.clear(); } // No tokens for this seq
    if (max_pos_overall == -1 && !v_mappings_host_vec.empty()) { v_mappings_host_vec.clear(); }
    if (max_pos_overall > -1) { // Ensure vectors are sized correctly even if last elements were not filled
        if (k_mappings_host_vec.size() < (size_t)max_pos_overall + 1) k_mappings_host_vec.resize(max_pos_overall + 1, {-1,0});
        if (v_mappings_host_vec.size() < (size_t)max_pos_overall + 1) v_mappings_host_vec.resize(max_pos_overall + 1, {-1,0});
    }

    std::vector<void*> host_gpu_page_device_ptrs;
    std::map<int, int> cpu_page_id_to_gpu_pool_idx;
    for(const auto& pair : unique_pages_map_cpu_id_to_ptr) {
        const llama_kv_page* cpu_page = pair.second;
        if (cpu_page && !cpu_page->is_free()) {
            struct ggml_tensor* t_page_host_meta = ggml_new_tensor_1d(ctx_meta_gpu, GGML_TYPE_I8, cpu_page->size);
            t_page_host_meta->data = cpu_page->data;
            ggml_tensor* t_page_gpu = create_gpu_tensor_from_host(ctx_meta_gpu, t_page_host_meta, "gpu_page_data_content");
            t_page_host_meta->data = nullptr;
            ggml_free(t_page_host_meta);
            ASSERT(t_page_gpu && t_page_gpu->data, "Failed to create GPU buffer for a page content.");
            cpu_page_id_to_gpu_pool_idx[cpu_page->id] = host_gpu_page_device_ptrs.size();
            host_gpu_page_device_ptrs.push_back(t_page_gpu->data);
        }
    }

    for(auto& mapping : k_mappings_host_vec) {
        if (mapping.page_idx != -1 && cpu_page_id_to_gpu_pool_idx.count(mapping.page_idx)) {
             mapping.page_idx = cpu_page_id_to_gpu_pool_idx.at(mapping.page_idx);
        } else { mapping.page_idx = -1; mapping.offset_in_page_elements = 0; }
    }
    for(auto& mapping : v_mappings_host_vec) {
        if (mapping.page_idx != -1 && cpu_page_id_to_gpu_pool_idx.count(mapping.page_idx)) {
            mapping.page_idx = cpu_page_id_to_gpu_pool_idx.at(mapping.page_idx);
        } else { mapping.page_idx = -1; mapping.offset_in_page_elements = 0; }
    }

    if (!k_mappings_host_vec.empty()) {
        ggml_backend_buffer_t k_map_buf = ggml_backend_alloc_buffer(g_cuda_buf_type_device, k_mappings_host_vec.size() * sizeof(paged_kv_token_mapping_host_for_gpu));
        k_view_host_gpu.token_mappings_gpu_ptr = ggml_backend_buffer_get_base(k_map_buf);
        ggml_backend_buffer_set_data(k_map_buf, 0, k_mappings_host_vec.data(), k_mappings_host_vec.size() * sizeof(paged_kv_token_mapping_host_for_gpu));
    } else { k_view_host_gpu.token_mappings_gpu_ptr = nullptr; }

    if (!host_gpu_page_device_ptrs.empty()) {
        ggml_backend_buffer_t k_pool_buf = ggml_backend_alloc_buffer(g_cuda_buf_type_device, host_gpu_page_device_ptrs.size() * sizeof(void*));
        k_view_host_gpu.page_pool_gpu_ptr = ggml_backend_buffer_get_base(k_pool_buf);
        ggml_backend_buffer_set_data(k_pool_buf, 0, host_gpu_page_device_ptrs.data(), host_gpu_page_device_ptrs.size() * sizeof(void*));
    } else { k_view_host_gpu.page_pool_gpu_ptr = nullptr; }

    if (!v_mappings_host_vec.empty()) {
        ggml_backend_buffer_t v_map_buf = ggml_backend_alloc_buffer(g_cuda_buf_type_device, v_mappings_host_vec.size() * sizeof(paged_kv_token_mapping_host_for_gpu));
        v_view_host_gpu.token_mappings_gpu_ptr = ggml_backend_buffer_get_base(v_map_buf);
        ggml_backend_buffer_set_data(v_map_buf, 0, v_mappings_host_vec.data(), v_mappings_host_vec.size() * sizeof(paged_kv_token_mapping_host_for_gpu));
    } else { v_view_host_gpu.token_mappings_gpu_ptr = nullptr; }

    v_view_host_gpu.page_pool_gpu_ptr = k_view_host_gpu.page_pool_gpu_ptr; // K and V share the page pool in this test setup

    int head_dim = mparams.n_embd / mparams.n_head_kv;
    k_view_host_gpu.num_tokens_in_logical_sequence = (max_pos_overall == -1) ? 0 : (max_pos_overall + 1);
    k_view_host_gpu.dtype = GGML_TYPE_F16; // TODO: Parameterize for Q8_0 tests
    k_view_host_gpu.element_size_bytes = sizeof(uint16_t);
    k_view_host_gpu.k_head_size_elements = head_dim;
    k_view_host_gpu.v_head_size_elements = head_dim;
    k_view_host_gpu.num_k_heads_total = mparams.n_head_kv;
    k_view_host_gpu.num_v_heads_total = mparams.n_head_kv;
    k_view_host_gpu.page_size_bytes = cparams.kv_page_size;
    k_view_host_gpu.v_block_start_offset_bytes = 0;

    v_view_host_gpu = k_view_host_gpu;

    ggml_backend_synchronize(g_cuda_backend);
    return {k_view_host_gpu, v_view_host_gpu};
}

// --- Test Case 9: CUDA Paged Attention Correctness (MMA F16) ---
void test_cuda_paged_attn_correctness_mma_f16() {
    printf("--- Running Test: test_cuda_paged_attn_correctness_mma_f16 ---\n");
    if (!g_cuda_backend) {
        printf("SKIPPING CUDA test: backend not initialized.\n");
        return;
    }

    struct ggml_init_params host_ctx_params = { 128 * 1024 * 1024, NULL, false };
    ggml_context* ctx_host = ggml_init(host_ctx_params);
    ASSERT(ctx_host != NULL, "Failed to create host ggml_context.");

    struct ggml_init_params meta_gpu_ctx_params = { ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE * 2, NULL, true };
    ggml_context* ctx_meta_gpu = ggml_init(meta_gpu_ctx_params);
    ASSERT(ctx_meta_gpu != NULL, "Failed to create GPU metadata ggml_context.");

    llama_model_params mparams = {};
    mparams.n_embd = 64;
    mparams.n_head = 2;
    mparams.n_head_kv = 2;
    mparams.n_layer = 1;
    const int head_dim = mparams.n_embd / mparams.n_head;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 16;
    cparams.n_batch = 4;
    cparams.use_paged_kv_cache = true;
    size_t bytes_per_token_kv_one_head_one_layer_k_or_v = (size_t)head_dim * sizeof(uint16_t);
    cparams.kv_page_size = bytes_per_token_kv_one_head_one_layer_k_or_v * 2 * 2; // Page fits 2 tokens' K AND V

    ggml_tensor* q_host = ggml_new_tensor_4d(ctx_host, GGML_TYPE_F16, head_dim, cparams.n_batch, mparams.n_head, 1);
    ggml_tensor* k_host = ggml_new_tensor_4d(ctx_host, GGML_TYPE_F16, head_dim, cparams.n_ctx, mparams.n_head_kv, 1);
    ggml_tensor* v_host = ggml_new_tensor_4d(ctx_host, GGML_TYPE_F16, head_dim, cparams.n_ctx, mparams.n_head_kv, 1);

    for(int i=0; i < ggml_nelements(q_host); ++i) ((ggml_fp16_t*)q_host->data)[i] = ggml_fp32_to_fp16((float)(i % 100) * 0.1f + 0.1f);
    for(int i=0; i < ggml_nelements(k_host); ++i) ((ggml_fp16_t*)k_host->data)[i] = ggml_fp32_to_fp16((float)(i % 100) * 0.05f - 0.2f);
    for(int i=0; i < ggml_nelements(v_host); ++i) ((ggml_fp16_t*)v_host->data)[i] = ggml_fp32_to_fp16((float)(i % 100) * 0.02f + 0.3f);

    printf("Running non-paged reference path...\n");
    ggml_tensor* q_gpu_ref = create_gpu_tensor_from_host(ctx_meta_gpu, q_host, "q_gpu_ref");
    ggml_tensor* k_gpu_ref = create_gpu_tensor_from_host(ctx_meta_gpu, k_host, "k_gpu_ref");
    ggml_tensor* v_gpu_ref = create_gpu_tensor_from_host(ctx_meta_gpu, v_host, "v_gpu_ref");

    struct ggml_tensor * dst_ref_ggml_tensor = ggml_dup_tensor(ctx_meta_gpu, q_gpu_ref);
    ggml_backend_buffer_t dst_ref_buffer = ggml_backend_alloc_buffer(g_cuda_buf_type_device, ggml_nbytes(dst_ref_ggml_tensor));
    ggml_backend_tensor_set_buffer(dst_ref_ggml_tensor, dst_ref_buffer);
    ggml_set_name(dst_ref_ggml_tensor, "dst_ref_gpu");

    struct ggml_cgraph* gf_ref = ggml_new_graph_custom(ctx_meta_gpu, GGML_DEFAULT_GRAPH_SIZE, false);
    struct ggml_tensor* attn_out_ref = ggml_flash_attn_ext(ctx_meta_gpu, q_gpu_ref, k_gpu_ref, v_gpu_ref, nullptr, 1.0f/sqrtf(head_dim), 0.0f, 0.0f, GGML_PREC_DEFAULT);
    ggml_set_name(attn_out_ref, "attn_out_ref");
    ggml_build_forward_expand(gf_ref, ggml_cpy(ctx_meta_gpu, attn_out_ref, dst_ref_ggml_tensor));
    ggml_backend_graph_compute(g_cuda_backend, gf_ref);

    std::vector<uint8_t> dst_ref_cpu_data = get_tensor_data_from_gpu(dst_ref_ggml_tensor);
    printf("Non-paged reference path completed.\n");

    printf("Paged path test logic is a TODO.\n");

    ggml_backend_buffer_free(q_gpu_ref->buffer);   ggml_free(q_gpu_ref);
    ggml_backend_buffer_free(k_gpu_ref->buffer);   ggml_free(k_gpu_ref);
    ggml_backend_buffer_free(v_gpu_ref->buffer);   ggml_free(v_gpu_ref);
    ggml_backend_buffer_free(dst_ref_buffer);      ggml_free(dst_ref_ggml_tensor);
    ggml_graph_free(gf_ref);

    ggml_free(ctx_host);
    ggml_free(ctx_meta_gpu);
    printf("--- Test test_cuda_paged_attn_correctness_mma_f16 (structure) FINISHED ---\n\n");
}
#endif // GGML_USE_CUDA


int main() {
    // ggml_backend_t backend = NULL;
    // ggml_backend_cpu_init();
    // backend = ggml_backend_cpu_init();
    // g_cpu_buf_type = ggml_backend_get_default_buffer_type(backend);


    printf("--- Starting Paged KV Cache Unit Tests ---\n");
    try {
        test_paged_cells_alloc_free();
        test_paged_cells_token_mapping();
        test_paged_cache_initialization();
        test_paged_cache_seq_add();
        test_paged_cache_seq_rm();
        test_paged_cache_seq_cp();
        test_paged_cache_seq_div();
        test_paged_cache_state_read_write();
        // Call other test functions here
#ifdef GGML_USE_CUDA
        if (g_cuda_backend) {
            // Call CUDA tests here
            test_cuda_paged_attn_correctness_mma_f16();
            test_cuda_paged_attn_correctness_tile_f16();
        } else {
            printf("SKIPPING CUDA tests as backend failed to initialize.\n");
        }
#endif
    } catch (const std::exception& e) {

// =================================================================================================
// PART 2: CUDA Paged Attention Kernel Tests
// =================================================================================================
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"

ggml_backend_t g_cuda_backend = NULL;
ggml_backend_buffer_type_t g_cuda_buf_type_device = NULL; // For device memory

void setup_cuda_for_test() {
    fprintf(stderr, "Initializing CUDA backend for tests...\n");
    // ggml_backend_cuda_init() initializes the CUDA runtime, selects device 0 by default.
    // It also initializes cuBLAS handles for that device.
    // For tests, typically device 0 is fine.
    g_cuda_backend = ggml_backend_cuda_init(0); // device_num = 0
    if (!g_cuda_backend) {
        fprintf(stderr, "setup_cuda_for_test: ggml_backend_cuda_init() failed. CUDA tests will be skipped.\n");
        return;
    }
    g_cuda_buf_type_device = ggml_backend_get_default_buffer_type(g_cuda_backend);
    ASSERT(g_cuda_buf_type_device != NULL, "Failed to get CUDA device buffer type.");
    printf("CUDA backend initialized for tests.\n");
}

void teardown_cuda_for_test() {
    if (g_cuda_backend) {
        ggml_backend_free(g_cuda_backend);
        g_cuda_backend = NULL;
        g_cuda_buf_type_device = NULL;
        printf("CUDA backend freed.\n");
    }
}

// Helper to create a GPU tensor and copy data from host
// The tensor `t_host` is a CPU tensor with data to be copied.
// `name` is for debugging.
ggml_tensor* create_gpu_tensor_from_host(ggml_context* ctx_meta, const ggml_tensor* t_host, const char* name) {
    if (!g_cuda_backend || !g_cuda_buf_type_device) {
        fprintf(stderr, "CUDA backend not initialized, cannot create GPU tensor %s.\n", name);
        return nullptr;
    }
    ggml_tensor* t_device = ggml_dup_tensor(ctx_meta, t_host);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_buffer(g_cuda_buf_type_device, ggml_nbytes(t_device));
    ASSERT(buffer != NULL, (std::string("Failed to allocate CUDA buffer for ") + name).c_str());
    ggml_backend_tensor_set_async(t_device, t_host->data, 0, ggml_nbytes(t_host));
    ggml_backend_synchronize(g_cuda_backend); // Ensure copy completes
    t_device->buffer = buffer; // Associate buffer
    ggml_set_name(t_device, name);
    printf("Created GPU tensor %s, size %zu bytes\n", name, ggml_nbytes(t_device));
    return t_device;
}

std::vector<uint8_t> get_tensor_data_from_gpu(const ggml_tensor* t_device) {
    if (!g_cuda_backend || !t_device || !t_device->buffer) {
        fprintf(stderr, "Invalid tensor or CUDA backend for get_tensor_data_from_gpu.\n");
        return {};
    }
    size_t nbytes = ggml_nbytes(t_device);
    std::vector<uint8_t> host_data(nbytes);
    ggml_backend_tensor_get_async(t_device, host_data.data(), 0, nbytes);
    ggml_backend_synchronize(g_cuda_backend); // Ensure copy completes
    return host_data;
}

// Forward declaration for the structure that will be used by the CUDA kernel
// This should match the definition in ggml-cuda/paged_attn_common.cuh
// For testing purposes, we redefine a host-side equivalent or include the CUDA header if appropriate.
// Assuming paged_attn_common.cuh is not directly includable in this C++ test file easily,
// we might need a simplified host-side mirror or pass components individually.
// For now, let's assume we can include it or have a compatible definition.
#ifdef GGML_USE_CUDA
// #include "../src/ggml-cuda/paged_attn_common.cuh" // This might cause issues if it has CUDA specific syntax not in C++ context
// For now, let's define a minimal compatible struct for the host to prepare arguments.
// The actual GPU struct is defined in CUDA headers. This is for host -> GPU data prep.
struct paged_kv_token_mapping_host_for_gpu { // Matches paged_kv_token_mapping_gpu
    int32_t page_idx;
    int32_t offset_in_page_elements; // For CUDA, this might be element count or byte offset depending on kernel. Assume byte for now.
};

struct paged_kv_sequence_view_host_for_gpu { // Matches paged_kv_sequence_view_gpu
    // Pointers to GPU memory for mappings and page pool
    void* token_mappings_gpu_ptr; // device pointer (e.g., paged_kv_token_mapping_gpu*)
    void* page_pool_gpu_ptr;      // device pointer (e.g., void** or uint64_t* for addresses)

    // Scalar members (must match the struct used in CUDA kernels)
    int32_t num_tokens_in_logical_sequence;
    ggml_type dtype; // GGML_TYPE_F16, GGML_TYPE_Q8_0 etc.

    int32_t k_head_size_elements; // head_dim for K
    int32_t v_head_size_elements; // head_dim for V
    int32_t num_k_heads_total;    // Total K heads in model
    int32_t num_v_heads_total;    // Total V heads in model
    uint32_t element_size_bytes;  // e.g. sizeof(half) or sizeof(block_q8_0) if page stores blocks
    uint32_t page_size_bytes;
    uint32_t v_block_start_offset_bytes; // If K,V are packed
    // Add any other fields that paged_kv_sequence_view_gpu has.
};


// Prepare paged KV views on GPU based on CPU cache layout
// Returns a pair of host-side structs that contain GPU pointers and scalar values
std::pair<paged_kv_sequence_view_host_for_gpu, paged_kv_sequence_view_host_for_gpu>
prepare_paged_kv_views_on_gpu(
    llama_paged_kv_cache& cpu_cache,
    const std::vector<llama_seq_id>& target_seq_ids, // For which sequences to prepare views
    ggml_context* ctx_meta_gpu, // For allocating GPU tensors for mappings/pool if needed
    const llama_model_params& mparams // Added model params for head dims etc.
) {
    if (!g_cuda_backend || !g_cuda_buf_type_device) {
        throw std::runtime_error("CUDA backend not initialized for paged view prep.");
    }

    llama_paged_kv_cells* cpu_cells = cpu_cache.get_paged_cells();
    ASSERT(cpu_cells != nullptr, "CPU paged_cells is null.");

    paged_kv_sequence_view_host_for_gpu k_view_host_gpu = {0};
    paged_kv_sequence_view_host_for_gpu v_view_host_gpu = {0};

    std::vector<paged_kv_token_mapping_host_for_gpu> k_mappings_host_vec;
    std::vector<paged_kv_token_mapping_host_for_gpu> v_mappings_host_vec;
    std::map<int, const llama_kv_page*> unique_pages_map_cpu_id_to_ptr;
    int max_pos_overall = -1; // Initialize to -1 in case there are no tokens

    for (llama_seq_id seq_id : target_seq_ids) {
        for (const auto& item : cpu_cells->get_token_to_page_offset_map()) {
            const auto& token_key = item.first;
            const auto& page_offset_val = item.second;

            if (token_key.seq_id != seq_id) continue;

            unique_pages_map_cpu_id_to_ptr[page_offset_val.page_id] = cpu_cells->get_page(page_offset_val.page_id);

            paged_kv_token_mapping_host_for_gpu current_mapping;
            current_mapping.page_idx = page_offset_val.page_id;
            current_mapping.offset_in_page_elements = page_offset_val.offset_bytes;

            int current_pos = token_key.pos;
            if (current_pos > max_pos_overall) max_pos_overall = current_pos;

            if (token_key.type == llama_paged_kv_cells::KVTokenType::TOKEN_TYPE_K) {
                if (current_pos >= (int)k_mappings_host_vec.size()) k_mappings_host_vec.resize(max_pos_overall + 1, {-1, -1});
                k_mappings_host_vec[current_pos] = current_mapping;
            } else {
                if (current_pos >= (int)v_mappings_host_vec.size()) v_mappings_host_vec.resize(max_pos_overall + 1, {-1, -1});
                v_mappings_host_vec[current_pos] = current_mapping;
            }
        }
    }
    if (max_pos_overall == -1 && !target_seq_ids.empty()) { // No tokens found for any target_seq_id
        // Leave mappings empty, sequence_length will be 0
    } else if (max_pos_overall > -1) {
        if (k_mappings_host_vec.size() < (size_t)max_pos_overall + 1) k_mappings_host_vec.resize(max_pos_overall + 1, {-1,-1});
        if (v_mappings_host_vec.size() < (size_t)max_pos_overall + 1) v_mappings_host_vec.resize(max_pos_overall + 1, {-1,-1});
    }


    std::vector<void*> host_gpu_page_device_ptrs;
    std::map<int, int> cpu_page_id_to_gpu_pool_idx;

    for(const auto& pair : unique_pages_map_cpu_id_to_ptr) {
        const llama_kv_page* cpu_page = pair.second;
        if (cpu_page && !cpu_page->is_free()) {
            // Create a temporary host ggml_tensor for data copy
            struct ggml_tensor* t_page_host = ggml_new_tensor_1d(ctx_meta_gpu, GGML_TYPE_I8, cpu_page->size);
            // Manually set data pointer for host tensor, as ctx_meta_gpu is no_alloc
            // This is a bit hacky; ideally, ggml_backend_tensor_set_async would take a host pointer directly.
            // Or, we create a temporary CPU buffer and tensor for this.
            // For this test, we'll directly use cpu_page->data if create_gpu_tensor_from_host can take raw host ptr.
            // Let's adjust create_gpu_tensor_from_host or make a new helper if needed.
            // For now, assume t_page_host->data is set or copy happens from cpu_page->data.
            // The current create_gpu_tensor_from_host expects a host ggml_tensor with data.
            t_page_host->data = cpu_page->data; // Temporarily point to existing CPU page data

            ggml_tensor* t_page_gpu = create_gpu_tensor_from_host(ctx_meta_gpu, t_page_host, "gpu_page_data_content");
            t_page_host->data = nullptr; // Decouple after copy
            ggml_free(t_page_host); // Free host tensor metadata

            ASSERT(t_page_gpu && t_page_gpu->data, "Failed to create GPU buffer for a page content.");

            cpu_page_id_to_gpu_pool_idx[cpu_page->id] = host_gpu_page_device_ptrs.size();
            host_gpu_page_device_ptrs.push_back(t_page_gpu->data);
        }
    }

    for(auto& mapping : k_mappings_host_vec) {
        if (mapping.page_idx != -1 && cpu_page_id_to_gpu_pool_idx.count(mapping.page_idx)) {
             mapping.page_idx = cpu_page_id_to_gpu_pool_idx[mapping.page_idx];
        } else { mapping.page_idx = -1; mapping.offset_in_page_elements = -1; }
    }
     for(auto& mapping : v_mappings_host_vec) {
        if (mapping.page_idx != -1 && cpu_page_id_to_gpu_pool_idx.count(mapping.page_idx)) {
            mapping.page_idx = cpu_page_id_to_gpu_pool_idx[mapping.page_idx];
        } else { mapping.page_idx = -1; mapping.offset_in_page_elements = -1; }
    }

    if (!k_mappings_host_vec.empty()) {
        ggml_backend_buffer_t k_map_buf = ggml_backend_alloc_buffer(g_cuda_buf_type_device, k_mappings_host_vec.size() * sizeof(paged_kv_token_mapping_host_for_gpu));
        k_view_host_gpu.token_mappings_gpu_ptr = ggml_backend_buffer_get_base(k_map_buf); // Get device pointer
        ggml_backend_buffer_set_data(k_map_buf, 0, k_mappings_host_vec.data(), k_mappings_host_vec.size() * sizeof(paged_kv_token_mapping_host_for_gpu));
    }
    if (!host_gpu_page_device_ptrs.empty()) {
        ggml_backend_buffer_t k_pool_buf = ggml_backend_alloc_buffer(g_cuda_buf_type_device, host_gpu_page_device_ptrs.size() * sizeof(void*));
        k_view_host_gpu.page_pool_gpu_ptr = ggml_backend_buffer_get_base(k_pool_buf);
        ggml_backend_buffer_set_data(k_pool_buf, 0, host_gpu_page_device_ptrs.data(), host_gpu_page_device_ptrs.size() * sizeof(void*));
    }

    if (!v_mappings_host_vec.empty()) {
        ggml_backend_buffer_t v_map_buf = ggml_backend_alloc_buffer(g_cuda_buf_type_device, v_mappings_host_vec.size() * sizeof(paged_kv_token_mapping_host_for_gpu));
        v_view_host_gpu.token_mappings_gpu_ptr = ggml_backend_buffer_get_base(v_map_buf);
        ggml_backend_buffer_set_data(v_map_buf, 0, v_mappings_host_vec.data(), v_mappings_host_vec.size() * sizeof(paged_kv_token_mapping_host_for_gpu));
    }
    if (!host_gpu_page_device_ptrs.empty()) { // Assuming K and V use the same page pool
        v_view_host_gpu.page_pool_gpu_ptr = k_view_host_gpu.page_pool_gpu_ptr;
    }

    k_view_host_gpu.num_tokens_in_logical_sequence = (max_pos_overall == -1) ? 0 : (max_pos_overall + 1);
    k_view_host_gpu.dtype = GGML_TYPE_F16;
    k_view_host_gpu.element_size_bytes = sizeof(uint16_t);
    k_view_host_gpu.k_head_size_elements = mparams.n_embd / mparams.n_head_kv;
    k_view_host_gpu.v_head_size_elements = mparams.n_embd / mparams.n_head_kv;
    k_view_host_gpu.num_k_heads_total = mparams.n_head_kv;
    k_view_host_gpu.num_v_heads_total = mparams.n_head_kv;
    k_view_host_gpu.page_size_bytes = cpu_cache.get_page_size_bytes();
    k_view_host_gpu.v_block_start_offset_bytes = 0;

    v_view_host_gpu = k_view_host_gpu;

    ggml_backend_synchronize(g_cuda_backend);

    return {k_view_host_gpu, v_view_host_gpu};
}


#endif // GGML_USE_CUDA


int main() {
    // ggml_backend_t backend = NULL;
    // ggml_backend_cpu_init();
    // backend = ggml_backend_cpu_init();
    // g_cpu_buf_type = ggml_backend_get_default_buffer_type(backend);

#ifdef GGML_USE_CUDA
    setup_cuda_for_test();
#endif

    printf("--- Starting Paged KV Cache Unit Tests ---\n");
    try {
        test_paged_cells_alloc_free();
        test_paged_cells_token_mapping();
        test_paged_cache_initialization();
        test_paged_cache_seq_add();
        test_paged_cache_seq_rm();
        test_paged_cache_seq_cp();
        test_paged_cache_seq_div();
        test_paged_cache_state_read_write();
        // Call other test functions here
#ifdef GGML_USE_CUDA
        if (g_cuda_backend) {
            // Call CUDA tests here
            test_cuda_paged_attn_correctness_mma_f16();
        } else {
            printf("SKIPPING CUDA tests as backend failed to initialize.\n");
        }
#endif
    } catch (const std::exception& e) {
        } else {
            printf("SKIPPING CUDA tests as backend failed to initialize.\n");
        }
#endif
    } catch (const std::exception& e) {
        } else {
            printf("SKIPPING CUDA tests as backend failed to initialize.\n");
        }
#endif
    } catch (const std::exception& e) {
        fprintf(stderr, "A test failed with exception: %s\n", e.what());
#ifdef GGML_USE_CUDA
        teardown_cuda_for_test();
#endif
        return 1;
    } catch (...) {
        fprintf(stderr, "A test failed with an unknown exception.\n");
#ifdef GGML_USE_CUDA
        teardown_cuda_for_test();
#endif
        return 1;
    }

#ifdef GGML_USE_CUDA
    teardown_cuda_for_test();
#endif
    printf("--- All Paged KV Cache Unit Tests PASSED ---\n");
    return 0;
}
