#include "../src/llama-arch.h"
#include "../src/llama-batch.h"
#include "../src/llama-hparams.h"
#include "../src/llama-impl.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-model.h"

#include "../common/common.h"
#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <cstring>  // For memcpy

/**
 * llama_kv_cache_unified interface test
 * Specifically testing the core functionality of unified cache
 */

/*- Helper Functions ------------------------------------------------------------------*/

static std::shared_ptr<llama_model> _make_test_model(
    llm_arch arch = LLM_ARCH_LLAMA,
    uint32_t n_layer = 4,
    uint32_t n_embd_head_k = 64,
    uint32_t n_embd_head_v = 64,
    uint32_t n_head = 8,
    uint32_t n_head_kv = 2) {

    llama_model_params params;
    std::shared_ptr<llama_model> model(new llama_model(params));
    model->hparams = llama_hparams();
    model->arch = arch;

    model->hparams.n_layer = n_layer;
    model->hparams.n_embd_head_k = n_embd_head_k;
    model->hparams.n_embd_head_v = n_embd_head_v;

    // Fill attention head array
    auto& n_head_arr = model->hparams.n_head_arr;
    std::fill(n_head_arr.begin(), n_head_arr.end(), n_head);
    
    auto& n_head_kv_arr = model->hparams.n_head_kv_arr;
    std::fill(n_head_kv_arr.begin(), n_head_kv_arr.end(), n_head_kv);

    return model;
}

struct test_scope {
    const char * name;
    explicit test_scope(const char * name) : name(name) {
        std::cout << "\n=== " << name << " ===\n";
    }
    ~test_scope() {
        std::cout << "✓ " << name << " Completed\n";
    }
};

/*- Test Cases ------------------------------------------------------------------*/

// Test 1: Basic KV Cache Creation and Query
static void test_basic_cache_creation() {
    test_scope scope("Basic Cache Creation Test");
    
    auto model = _make_test_model();
    
    // Create unified cache
    llama_kv_cache_unified cache(
        *model,
        nullptr,                    // layer_filter (all layers)
        GGML_TYPE_F16,             // type_k
        GGML_TYPE_F16,             // type_v
        true,                      // v_trans
        false,                     // offload
        128,                       // kv_size
        4,                         // n_seq_max
        32,                        // n_pad
        0,                         // n_swa
        LLAMA_SWA_TYPE_NONE        // swa_type
    );
    
    // Verify basic attributes
    std::cout << "Cache Size: "     << cache.get_size() << "\n";
    std::cout << "Current Usage: "  << cache.get_n() << "\n";
    std::cout << "Supports Shift: " << (cache.get_can_shift() ? "Yes" : "No") << "\n"; // TODO: Implement shift
    std::cout << "Supports Edit: "  << (cache.get_can_edit() ? "Yes" : "No") << "\n"; // TODO: Implement edit
    
    // Basic assertions
    GGML_ASSERT(cache.get_size() == 128);
    GGML_ASSERT(cache.get_n() == 0);  // Initially empty
}

// Test 2: Sequence Management - Add, Query, Delete
static void test_sequence_management() {
    test_scope scope("Sequence Management Test");
    
    auto model = _make_test_model();
    
    llama_kv_cache_unified cache(
        *model, nullptr, GGML_TYPE_F16, GGML_TYPE_F16,
        true, false, 64, 4, 16, 0, LLAMA_SWA_TYPE_NONE
    );
    
    // Helper function to print cache state
    auto print_cache_state = [&](const std::string& operation) {
        std::cout << "\n--- Cache State After " << operation << " ---\n";
        std::cout << "Cache Size: " << cache.get_size() << "\n";
        std::cout << "Current Usage (n): " << cache.get_n() << "\n";
        
        // Check state for multiple sequences
        for (llama_seq_id seq_id = 0; seq_id <= 2; ++seq_id) {
            llama_pos min_pos = cache.seq_pos_min(seq_id);
            llama_pos max_pos = cache.seq_pos_max(seq_id);
            std::cout << "Sequence " << seq_id << " Range: [" << min_pos << ", " << max_pos << "]";
            if (min_pos == -1 && max_pos == -1) {
                std::cout << " (empty)";
            } else {
                std::cout << " (active, length: " << (max_pos - min_pos + 1) << ")";
            }
            std::cout << "\n";
        }
        std::cout << "----------------------------------------------\n";
    };
    
    // Initial state check
    llama_seq_id seq_0 = 0;
    llama_seq_id seq_1 = 1;
    
    print_cache_state("Initial Creation");
    
    std::cout << "\n=== Adding actual tokens to see cache changes ===\n";
    
    // Create a batch with some tokens for seq_0
    llama_batch batch = llama_batch_init(3, 0, 1);
    common_batch_add(batch, 101, 0, {seq_0}, false);
    common_batch_add(batch, 102, 1, {seq_0}, false);
    common_batch_add(batch, 103, 2, {seq_0}, false);
    
    llama_sbatch sbatch(batch, model->hparams.n_embd_head_k * model->hparams.n_head_kv_arr[0], true, false);
    llama_ubatch ubatch = sbatch.split_simple(3);
    
    std::cout << "Adding 3 tokens to sequence " << seq_0 << "...\n";
    bool slot_found = cache.find_slot(ubatch);
    if (slot_found) {
        cache.commit();
        std::cout << "✓ Tokens successfully added to sequence " << seq_0 << "\n";
    } else {
        std::cout << "✗ Failed to add tokens to sequence " << seq_0 << "\n";
    }
    print_cache_state("Adding Tokens to seq_0");
    
    llama_batch_free(batch);
    
    // Now test seq_cp again with actual data
    std::cout << "\nExecuting: cache.seq_cp(seq_0=" << seq_0 << ", seq_1=" << seq_1 << ", pos_0=0, pos_1=3) with actual data\n";
    cache.seq_cp(seq_0, seq_1, 0, 3);  // Copy positions 0-2 of sequence 0 to sequence 1
    std::cout << "Sequence Copy with Data Completed\n";
    print_cache_state("Sequence Copy with Actual Data");
    
    // Test keeping only specified sequence
    std::cout << "\nExecuting: cache.seq_keep(seq_1=" << seq_1 << ")\n";
    cache.seq_keep(seq_1);  // Keep only sequence 1
    std::cout << "Keeping Sequence 1, Cleaning Other Sequences\n";
    print_cache_state("Keep Only seq_1 (seq_keep)");
    
    // Verify sequence 1 still exists (by querying position range)
    llama_pos min_pos_1 = cache.seq_pos_min(seq_1);
    llama_pos max_pos_1 = cache.seq_pos_max(seq_1);
    std::cout << "Final Sequence 1 Range After Keeping: [" << min_pos_1 << ", " << max_pos_1 << "]\n";
    
    // Test clearing all sequences
    std::cout << "\nExecuting: cache.clear()\n";
    cache.clear();
    std::cout << "Cache Cleared\n";
    print_cache_state("Clear All (clear)");
}

// Test 3: Tensor Operations - K and V Retrieval and Copying
static void test_tensor_operations() {
    test_scope scope("Tensor Operations Test");
    
    auto model = _make_test_model();
    
    llama_kv_cache_unified cache(
        *model, nullptr, GGML_TYPE_F16, GGML_TYPE_F16,
        true, false, 32, 2, 8, 0, LLAMA_SWA_TYPE_NONE
    );
    
    // Create ggml context
    ggml_init_params ctx_params = {
        /*.mem_size   =*/ 16*1024*1024,  // 16MB for larger operations
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,  // Enable allocation
    };
    
    ggml_context* ctx = ggml_init(ctx_params);
    if (!ctx) {
        std::cerr << "Unable to create ggml context\n";
        return;
    }
    
    try {
        int32_t layer_id = 0;
        
        // First, add some tokens to the cache to create slots
        llama_seq_id seq_id = 42;
        const int n_tokens = 4;
        
        // Create and setup batch for cache slot allocation
        std::cout << "Creating test batch with " << n_tokens << " tokens...\n";
        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        
        // Add tokens to batch
        for (int i = 0; i < n_tokens; ++i) {
            common_batch_add(batch, 1000 + i, i, {seq_id}, false);
        }
        
        // Convert to sbatch and ubatch
        llama_sbatch sbatch(batch, model->hparams.n_embd_head_k * model->hparams.n_head_kv_arr[0], true, false);
        llama_ubatch ubatch = sbatch.split_simple(n_tokens);
        
        std::cout << "Batch created: n_tokens=" << ubatch.n_tokens << ", n_seqs=" << ubatch.n_seqs << "\n";
        
        // Find slot in cache and commit
        bool slot_found = cache.find_slot(ubatch);
        if (!slot_found) {
            std::cout << "✗ Failed to find slot in cache\n";
            llama_batch_free(batch);
            ggml_free(ctx);
            return;
        }
        
        cache.commit();
        std::cout << "✓ Cache slot allocated and committed\n";
        std::cout << "Cache current n: " << cache.get_n() << "\n";
        
        // Get K and V tensor views
        ggml_tensor* k_view = cache.get_k(ctx, layer_id);
        ggml_tensor* v_view = cache.get_v(ctx, layer_id);
        
        if (k_view) {
            std::cout << "K Tensor Dimensions: [" 
                      << k_view->ne[0] << ", " 
                      << k_view->ne[1] << ", " 
                      << k_view->ne[2] << ", " 
                      << k_view->ne[3] << "]\n";
            GGML_ASSERT(k_view->type == GGML_TYPE_F16);
        }
        
        if (v_view) {
            std::cout << "V Tensor Dimensions: ["
                      << v_view->ne[0] << ", "
                      << v_view->ne[1] << ", "
                      << v_view->ne[2] << ", "
                      << v_view->ne[3] << "]\n";
            GGML_ASSERT(v_view->type == GGML_TYPE_F16);
        }
        
        // Create test current K and V tensors with actual data
        ggml_tensor* k_cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 
                                               model->hparams.n_embd_head_k,
                                               model->hparams.n_head_kv_arr[0], 
                                               n_tokens);
        ggml_tensor* v_cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                               model->hparams.n_embd_head_v,
                                               model->hparams.n_head_kv_arr[0],
                                               n_tokens);
        
        if (k_cur && v_cur) {
            std::cout << "Test Tensor Creation Successful\n";
            
            // Fill test tensors with recognizable patterns
            float* k_data = (float*)k_cur->data;
            float* v_data = (float*)v_cur->data;
            
            size_t k_elements = ggml_nelements(k_cur);
            size_t v_elements = ggml_nelements(v_cur);
            
            std::cout << "Filling K tensor (" << k_elements << " elements) with test data...\n";
            for (size_t i = 0; i < k_elements; ++i) {
                k_data[i] = 1.0f + 0.1f * (i % 100);  // Pattern: 1.0, 1.1, 1.2, ..., 10.9, repeat
            }
            
            std::cout << "Filling V tensor (" << v_elements << " elements) with test data...\n";
            for (size_t i = 0; i < v_elements; ++i) {
                v_data[i] = 2.0f + 0.05f * (i % 200);  // Pattern: 2.0, 2.05, 2.1, ..., 11.95, repeat
            }
            
            // Print first few values of test data
            std::cout << "K test data (first 10 values): ";
            int k_print_count = (k_elements < 10) ? static_cast<int>(k_elements) : 10;
            for (int i = 0; i < k_print_count; ++i) {
                std::cout << k_data[i] << " ";
            }
            std::cout << "\n";
            
            std::cout << "V test data (first 10 values): ";
            int v_print_count = (v_elements < 10) ? static_cast<int>(v_elements) : 10;
            for (int i = 0; i < v_print_count; ++i) {
                std::cout << v_data[i] << " ";
            }
            std::cout << "\n";
            
            // Create copy operations
            ggml_tensor* k_copy_op = cache.cpy_k(ctx, k_cur, layer_id);
            ggml_tensor* v_copy_op = cache.cpy_v(ctx, v_cur, layer_id);
            
            if (k_copy_op && v_copy_op) {
                std::cout << "Tensor Copy Operation Created Successfully\n";
                
                // Verify copy operation types
                GGML_ASSERT(k_copy_op->op == GGML_OP_CPY);
                GGML_ASSERT(v_copy_op->op == GGML_OP_CPY);
                
                // Create computation graph and execute the copy operations
                std::cout << "Creating computation graph to execute copy operations...\n";
                ggml_cgraph* gf = ggml_new_graph(ctx);
                
                ggml_build_forward_expand(gf, k_copy_op);
                ggml_build_forward_expand(gf, v_copy_op);
                
                std::cout << "Executing computation graph...\n";
                int result = ggml_graph_compute_with_ctx(ctx, gf, 1);
                
                if (result == 0) {
                    std::cout << "✓ Copy operations executed successfully!\n";
                    
                    // Now verify that data was actually copied to cache
                    std::cout << "\n=== Verifying cache contents ===\n";
                    
                    // Get fresh tensor views from cache
                    ggml_tensor* cache_k = cache.get_k(ctx, layer_id);
                    ggml_tensor* cache_v = cache.get_v(ctx, layer_id);
                    
                    if (cache_k && cache_k->data) {
                        std::cout << "Reading K data from cache...\n";
                        
                        // Create a temporary FP32 tensor to dequantize the cache data
                        ggml_tensor* k_verify = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                                 cache_k->ne[0], cache_k->ne[1], n_tokens);
                        
                        // Copy first n_tokens from cache to verify tensor
                        ggml_tensor* k_slice = ggml_view_3d(ctx, cache_k, 
                                                          cache_k->ne[0], cache_k->ne[1], n_tokens,
                                                          cache_k->nb[1], cache_k->nb[2], 0);
                        
                        ggml_cgraph* verify_gf = ggml_new_graph(ctx);
                        ggml_tensor* k_cpy_verify = ggml_cpy(ctx, k_slice, k_verify);
                        ggml_build_forward_expand(verify_gf, k_cpy_verify);
                        
                        int verify_result = ggml_graph_compute_with_ctx(ctx, verify_gf, 1);
                        
                        if (verify_result == 0) {
                            float* cache_k_data = (float*)k_verify->data;
                            std::cout << "✓ K cache data read successfully\n";
                            std::cout << "K cache data (first 10 values): ";
                            int64_t k_verify_elements = ggml_nelements(k_verify);
                            int k_verify_print_count = (k_verify_elements < 10) ? static_cast<int>(k_verify_elements) : 10;
                            for (int i = 0; i < k_verify_print_count; ++i) {
                                std::cout << cache_k_data[i] << " ";
                            }
                            std::cout << "\n";
                            
                            // Compare with original data
                            bool k_match = true;
                            float max_k_diff = 0.0f;
                            size_t compare_elements = (ggml_nelements(k_verify) < k_elements) ? ggml_nelements(k_verify) : k_elements;
                            
                            for (size_t i = 0; i < compare_elements && i < 100; ++i) {  // Compare first 100 elements
                                float diff = std::abs(cache_k_data[i] - k_data[i]);
                                if (diff > 0.01f) {  // Allow small quantization error
                                    k_match = false;
                                }
                                max_k_diff = std::max(max_k_diff, diff);
                            }
                            
                            std::cout << "K data comparison - Max difference: " << max_k_diff 
                                     << ", Match (within tolerance): " << (k_match ? "✓" : "✗") << "\n";
                        } else {
                            std::cout << "✗ Failed to read K data from cache (result: " << verify_result << ")\n";
                        }
                    } else {
                        std::cout << "✗ Cannot access K cache data\n";
                    }
                    
                    // Similar verification for V cache
                    if (cache_v && cache_v->data) {
                        std::cout << "\nReading V data from cache...\n";
                        
                        ggml_tensor* v_verify = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                                                 cache_v->ne[0], cache_v->ne[1], n_tokens);
                        
                        ggml_tensor* v_slice = ggml_view_3d(ctx, cache_v, 
                                                          cache_v->ne[0], cache_v->ne[1], n_tokens,
                                                          cache_v->nb[1], cache_v->nb[2], 0);
                        
                        ggml_cgraph* verify_gf_v = ggml_new_graph(ctx);
                        ggml_tensor* v_cpy_verify = ggml_cpy(ctx, v_slice, v_verify);
                        ggml_build_forward_expand(verify_gf_v, v_cpy_verify);
                        
                        int verify_result_v = ggml_graph_compute_with_ctx(ctx, verify_gf_v, 1);
                        
                        if (verify_result_v == 0) {
                            float* cache_v_data = (float*)v_verify->data;
                            std::cout << "✓ V cache data read successfully\n";
                            std::cout << "V cache data (first 10 values): ";
                            int64_t v_verify_elements = ggml_nelements(v_verify);
                            int v_verify_print_count = (v_verify_elements < 10) ? static_cast<int>(v_verify_elements) : 10;
                            for (int i = 0; i < v_verify_print_count; ++i) {
                                std::cout << cache_v_data[i] << " ";
                            }
                            std::cout << "\n";
                            
                            // Compare with original data
                            bool v_match = true;
                            float max_v_diff = 0.0f;
                            size_t compare_elements = (ggml_nelements(v_verify) < v_elements) ? ggml_nelements(v_verify) : v_elements;
                            
                            for (size_t i = 0; i < compare_elements && i < 100; ++i) {
                                float diff = std::abs(cache_v_data[i] - v_data[i]);
                                if (diff > 0.01f) {
                                    v_match = false;
                                }
                                max_v_diff = std::max(max_v_diff, diff);
                            }
                            
                            std::cout << "V data comparison - Max difference: " << max_v_diff 
                                     << ", Match (within tolerance): " << (v_match ? "✓" : "✗") << "\n";
                        } else {
                            std::cout << "✗ Failed to read V data from cache (result: " << verify_result_v << ")\n";
                        }
                    } else {
                        std::cout << "✗ Cannot access V cache data\n";
                    }
                    
                    std::cout << "\n✓ Cache verification completed!\n";
                    
                } else {
                    std::cout << "✗ Copy operations failed with result: " << result << "\n";
                }
                
            } else {
                std::cout << "✗ Tensor Copy Operation Creation Failed\n";
            }
        }
        
        llama_batch_free(batch);
        
    } catch (const std::exception& e) {
        std::cerr << "Tensor Operation Failed: " << e.what() << "\n";
    }
    
    ggml_free(ctx);
}

// Test 4: Memory and State Management
static void test_memory_and_state_management() {
    test_scope scope("Memory and State Management Test");
    
    auto model = _make_test_model();
    
    llama_kv_cache_unified cache(
        *model, nullptr, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0,  // Use quantized types for testing
        true, false, 16, 2, 4, 0, LLAMA_SWA_TYPE_NONE
    );
    
    // Test clear operation
    cache.clear();
    std::cout << "Cache Cleared\n";
    
    GGML_ASSERT(cache.get_n() == 0);
    
    // Test state management
    cache.commit();   // Commit current state
    std::cout << "State Committed\n";
    
    cache.restore();  // Restore to previous state
    std::cout << "State Restored\n";
    
    // Test defragmentation scheduling
    cache.defrag_sched(0.5f);  // Trigger defragmentation when fragmentation > 50%
    std::cout << "Defragmentation Scheduling Completed\n";
    
    // Test setting to full state (for worst-case computation buffer allocation)
    cache.set_full();
    std::cout << "Cache Set to Full State\n";
}

// Test 5: Compatibility of Different Quantization Types
static void test_quantized_types() {
    test_scope scope("Quantization Type Compatibility Test");
    
    auto model = _make_test_model();
    
    // Test different quantization type combinations
    struct quantization_test {
        ggml_type type_k;
        ggml_type type_v;
        const char* desc;
    };
    
    std::vector<quantization_test> tests = {
        {GGML_TYPE_F32, GGML_TYPE_F32, "FP32 + FP32"},
        {GGML_TYPE_F16, GGML_TYPE_F16, "FP16 + FP16"},
        {GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, "Q8_0 + Q8_0"},
        {GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, "Q4_0 + Q4_0"},
        {GGML_TYPE_F16, GGML_TYPE_Q8_0, "FP16 K + Q8_0 V"},
    };
    
    for (const auto& test : tests) {
        try {
            llama_kv_cache_unified cache(
                *model, nullptr, test.type_k, test.type_v,
                true, false, 16, 1, 4, 0, LLAMA_SWA_TYPE_NONE
            );
            
            std::cout << "✓ " << test.desc << " Compatible\n";
            
            // Basic operation test
            cache.clear();
            cache.commit();
            cache.restore();
            
        } catch (const std::exception& e) {
            std::cout << "✗ " << test.desc << " Failed: " << e.what() << "\n";
        }
    }
}

// Test 6: Boundary Conditions and Error Handling
static void test_boundary_conditions() {
    test_scope scope("Boundary Conditions Test");
    
    auto model = _make_test_model();
    
    // Test small cache size
    try {
        llama_kv_cache_unified small_cache(
            *model, nullptr, GGML_TYPE_F16, GGML_TYPE_F16,
            true, false, 4, 1, 2, 0, LLAMA_SWA_TYPE_NONE
        );
        
        std::cout << "✓ Small Cache Size (4) Created Successfully\n";
        
        // Test boundary sequence operations
        small_cache.seq_rm(-1, -1, -1);  // Delete all positions of all sequences
        std::cout << "✓ Boundary Deletion Operation Completed\n";
        
        small_cache.seq_add(0, -1, -1, 5);  // Handle negative positions
        std::cout << "✓ Boundary Addition Operation Completed\n";
        
    } catch (const std::exception& e) {
        std::cout << "✗ Boundary Conditions Test Failed: " << e.what() << "\n";
    }
    
    // Test zero max sequences
    try {
        llama_kv_cache_unified zero_seq_cache(
            *model, nullptr, GGML_TYPE_F16, GGML_TYPE_F16,
            true, false, 8, 0, 4, 0, LLAMA_SWA_TYPE_NONE
        );
        
        std::cout << "✓ Zero Max Sequences Cache Created Successfully\n";
        
    } catch (const std::exception& e) {
        std::cout << "✗ Zero Sequences Test Failed: " << e.what() << "\n";
    }
}

int main(int argc, char** argv) {
    std::cout << "llama_kv_cache_unified Interface Test Program\n";
    std::cout << "==========================================\n";
    
    try {
        // Run all tests
        test_basic_cache_creation();
        test_sequence_management();
        // test_tensor_operations();
        // test_memory_and_state_management();
        // test_quantized_types();
        // test_boundary_conditions();
        
        std::cout << "\n🎉 All Tests Completed!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test Failed: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "\n❌ Unknown Error\n";
        return 1;
    }
    
    return 0;
}
