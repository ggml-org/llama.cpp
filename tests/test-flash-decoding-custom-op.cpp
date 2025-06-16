#include "../src/llama-kv-cache-mixed.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-impl.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <cstdlib>
#include <algorithm>

#include <iostream>

#ifdef LLAMA_TORCH_AVAILABLE
#include <torch/torch.h>

void test_torch_integration() {
    std::cout << "Testing PyTorch C++ integration..." << std::endl;
    
    // Create a simple tensor
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << "Created tensor with shape: " << tensor.sizes() << std::endl;
    std::cout << "Tensor data:\n" << tensor << std::endl;
    
    // Test basic operations
    torch::Tensor result = tensor * 2.0;
    std::cout << "After multiplication by 2:\n" << result << std::endl;
    
    // Check CUDA availability
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available!" << std::endl;
        std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
    } else {
        std::cout << "CUDA is not available, using CPU" << std::endl;
    }
    
    std::cout << "PyTorch integration test completed successfully!" << std::endl;
}
#endif // LLAMA_TORCH_AVAILABLE

// Forward declaration of the flash decoding function
void ggml_custom_flash_attn_mixed_simple(
    ggml_tensor * dst,
    int ith,
    int nth,
    void* wdata,
    size_t wsize,
    void * userdata
);

// Parameters for flash attention are defined in llama-kv-cache-mixed.h

static void fill_random_f32(float* data, size_t n, float min_val = -1.0f, float max_val = 1.0f) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);

    for (size_t i = 0; i < n; i++) {
        data[i] = dis(gen);
    }
}

static void fill_random_f16(ggml_fp16_t* data, size_t n, float min_val = -1.0f, float max_val = 1.0f) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);

    for (size_t i = 0; i < n; i++) {
        data[i] = ggml_fp32_to_fp16(dis(gen));
    }
}

static void fill_causal_mask(float* mask_data, int64_t n_tokens, int64_t kv_len) {
    for (int64_t i = 0; i < n_tokens; i++) {
        for (int64_t j = 0; j < kv_len; j++) {
            if (j <= i + (kv_len - n_tokens)) {
                mask_data[i * kv_len + j] = 0.0f;
            } else {
                mask_data[i * kv_len + j] = -INFINITY;
            }
        }
    }
}

static void print_tensor_info(const char* name, ggml_tensor* tensor) {
    printf("%s: [%ld, %ld, %ld, %ld] type=%s\n",
           name, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
           ggml_type_name(tensor->type));
}

int main() {
    printf("Testing Flash-Decoding Custom Operation vs Standard Flash Attention\n");

    // Test parameters - reduce KV length to minimize F16 accumulation errors
    const int head_dim  = 4;
    const int n_heads   = 4;
    const int n_kv_heads = 1;
    const int seq_len   = 1;     // Q length
    const int kv_len    = 48;    // K/V length - reduced for better F16 precision
    const int n_threads = 12;

    printf("Test Parameters:\n");
    printf("  head_dim=%d, n_heads=%d, n_kv_heads=%d, seq_len=%d, kv_len=%d\n",
           head_dim, n_heads, n_kv_heads, seq_len, kv_len);
    printf("  GQA ratio: %d query heads per KV head\n", n_heads / n_kv_heads);

    // Initialize ggml context
    const size_t ctx_size = 256*1024*1024; // 256MB for context
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize ggml context\n");
        return 1;
    }

    // Create tensors for custom flash attention (our format)
    // Format: [head_dim, seq_len, n_heads, 1] for Q, K, V
    // Based on mixed implementation: Q=F32, K=F16, V=F32, mask=F32
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, seq_len, n_heads,    1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, kv_len,  n_kv_heads, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, kv_len,  n_kv_heads, 1);

    // Create mask tensor for custom flash attention
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv_len, GGML_PAD(seq_len, 256));

    // Fill tensors with random data
    fill_random_f32((float*)q->data, ggml_nelements(q));

    if (k->type == GGML_TYPE_F32) {
        fill_random_f32((float*)k->data, ggml_nelements(k));
    } else {
        fill_random_f16((ggml_fp16_t*)k->data, ggml_nelements(k));  // K is F16
    }

    if (v->type == GGML_TYPE_F32) {
        fill_random_f32((float*)v->data, ggml_nelements(v));
    } else {
        fill_random_f16((ggml_fp16_t*)v->data, ggml_nelements(v));
    }

    // Fill mask - use identity mask (all positions visible)
    float* mask_data = (float*)mask->data;
    fill_causal_mask(mask_data, seq_len, kv_len);

    for (int i = seq_len; i < GGML_PAD(seq_len, 256); i++) {
        for (int j = 0; j < kv_len; j++) {
            mask_data[i * kv_len + j] = -INFINITY;
        }
    }

    //> Use random data for realistic testing
    // ggml_set_f32(q, 1.0f);    // Q = [1, 1]
    // ggml_set_f32(k, 2.0f);    // K = [2, 2] for all tokens
    // ggml_set_f32(v, 3.0f);    // V = [3, 3] for all tokens

    ggml_set_f32(mask, 0.0f); // No masking

    // Adjust fp16_window to fit within kv_len for this test
    size_t fp16_window  = std::min((size_t)kv_len, (size_t)32);
    size_t quant_len = kv_len - fp16_window > 0 ? kv_len - fp16_window : 0;
    size_t fp16_nb1          = head_dim * ggml_type_size(k->type);
    size_t fp16_nb2          = fp16_window * fp16_nb1;
    size_t fp16_nb3          = fp16_nb2 * n_kv_heads;

    size_t quant_nb1          = head_dim * ggml_type_size(k->type);
    size_t quant_nb2          = quant_len * quant_nb1;
    size_t quant_nb3          = quant_nb2 * n_kv_heads;
    
    size_t kv_quant_offset = n_kv_heads * fp16_window * fp16_nb1;

    ggml_tensor * k_fp16    = ggml_view_4d(ctx, k, head_dim, fp16_window, n_kv_heads, 1, fp16_nb1,  fp16_nb2, fp16_nb3, 0);
    ggml_tensor * v_fp16    = ggml_view_4d(ctx, v, head_dim, fp16_window, n_kv_heads, 1, fp16_nb1,  fp16_nb2, fp16_nb3, 0);
    
    // Only create quantized views if we have quantized tokens
    // NOTICE: This quant_len can be 0;
    ggml_tensor * k_quant = ggml_view_4d(ctx, k, head_dim, quant_len, n_kv_heads, 1, quant_nb1, quant_nb2, quant_nb3, kv_quant_offset);
    ggml_tensor * v_quant = ggml_view_4d(ctx, v, head_dim, quant_len, n_kv_heads, 1, quant_nb1, quant_nb2, quant_nb3, kv_quant_offset);

    // ============================================================================
    // Test 1: Custom F32 Flash-attention Implementation
    // ============================================================================
    printf("\n--- Testing Custom Flash-Decoding Implementation ---\n");

    ggml_tensor * custom_result = ggml_flash_attn_mixed(
        ctx, q, k_fp16, v_fp16, 
        k_quant, v_quant, mask,  // Use NULL mask for comparison
        1 / std::sqrt(head_dim),
        0.0f,  // max_bias
        0.0f   // logit_softcap
    );
    ggml_flash_attn_ext_set_prec(custom_result, GGML_PREC_MIXED);

    // ggml_set_f32(custom_result, 1.2f);

    if (!custom_result) {
        printf("ERROR: Failed to create custom flash attention operation\n");
        ggml_free(ctx);
        return 1;
    }

    // Build and execute computation graph for custom implementation
    struct ggml_cgraph * graph_custom = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph_custom, custom_result);

    // Calculate workspace size for custom operation
    const size_t output_size = seq_len * n_heads * head_dim;
    const size_t local_max_size = seq_len * n_heads;  // Updated to match LOCAL_MAX_SIZE
    const size_t local_sum_size = seq_len * n_heads;  // Add sum tracking
    const size_t temp_buffer_size = head_dim;
    const size_t q_quantized_float_elements = (head_dim * sizeof(ggml_fp16_t) + sizeof(float) - 1) / sizeof(float);
    const size_t elements_per_thread = output_size + local_max_size + local_sum_size + 2 * temp_buffer_size + q_quantized_float_elements + 1 + 16; // +1 for sync_buffer, +16 for CACHE_LINE_SIZE_F32

    struct ggml_threadpool_params * tp_params = (struct ggml_threadpool_params *)malloc(sizeof(struct ggml_threadpool_params));
    for (int i = 0; i < GGML_MAX_N_THREADS; i++) {
        tp_params->cpumask[i] = false;
    }
    tp_params->n_threads = n_threads;
    tp_params->prio = GGML_SCHED_PRIO_HIGH;
    tp_params->poll = 0;
    tp_params->strict_cpu = false;
    tp_params->paused = false;

    struct ggml_threadpool * tp = ggml_threadpool_new(tp_params);

    struct ggml_cplan cplan_custom = ggml_graph_plan(graph_custom, n_threads, tp);

    // Build and execute computation graph for custom implementation
    // ggml_build_forward_expand(graph_custom, custom_result);

    // Allocate workspace
    size_t workspace_size = n_threads * elements_per_thread * sizeof(float);
    workspace_size = std::max(workspace_size, cplan_custom.work_size);
    uint8_t* workspace = (uint8_t*)malloc(workspace_size);
    cplan_custom.work_data = workspace;
    cplan_custom.work_size = workspace_size;

    // printf("Computing custom flash-decoding...\n");
    enum ggml_status status_custom = ggml_graph_compute(graph_custom, &cplan_custom);

    printf("Computing standard flash attention...\n");
    // enum ggml_status status_custom = ggml_graph_compute_with_ctx(ctx, graph_custom, n_threads);

    if (status_custom != GGML_STATUS_SUCCESS) {
        printf("ERROR: Custom flash attention computation failed with status: %d\n", status_custom);
        // free(workspace);
        ggml_free(ctx);
        return 1;
    }

    printf("Custom flash-decoding computation successful\n");

    // ============================================================================
    // Test 2: Standard Flash Attention Implementation (for comparison)
    // ============================================================================
    printf("\n--- Testing Standard Flash Attention ---\n");

    // Create tensors for standard flash attention
    // Standard format: [head_dim, seq_len, n_heads, batch_size] for Q, K, V
    ggml_tensor * q_std = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim,  seq_len,    n_heads,       1);
    ggml_tensor * k_std = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim,  kv_len,     n_kv_heads,    1);
    ggml_tensor * v_std = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim,  kv_len,     n_kv_heads,    1);

    // Convert data types and rearrange dimensions for GQA
    float* q_f32_src = (float*)q->data;
    ggml_fp16_t* k_f16_src = (ggml_fp16_t*)k->data;  // K is already F16
    float* k_f32_src = (float*)v->data;

    // NOTE: v is F16 in the custom implementation, but F32 in the standard implementation
    ggml_fp16_t* v_f16_src = (ggml_fp16_t*)v->data;
    float* v_f32_src = (float*)v->data;
    
    float* q_f32_std = (float*)q_std->data;  // Q_std is now F32
    ggml_fp16_t* k_f16 = (ggml_fp16_t*)k_std->data;
    ggml_fp16_t* v_f16 = (ggml_fp16_t*)v_std->data;

    // Copy Q: [head_dim, seq_len, n_heads] -> [head_dim, seq_len, n_heads] (F32 -> F32, no conversion needed)
    for (int h = 0; h < n_heads; h++) {
        for (int t = 0; t < seq_len; t++) {
            for (int d = 0; d < head_dim; d++) {
                // Source: [d + t*head_dim + h*head_dim*seq_len]
                // Dest:   [d + t*head_dim + h*head_dim*seq_len] (same layout for now)
                int src_idx = d + t * head_dim + h * head_dim * seq_len;
                int dst_idx = d + t * head_dim + h * head_dim * seq_len;
                q_f32_std[dst_idx] = q_f32_src[src_idx];
            }
        }
    }

    // Copy and convert K,V: [head_dim, kv_len, n_kv_heads] -> [head_dim, kv_len, n_kv_heads]
    // For K and V in GQA, we need to use n_kv_heads (not n_heads)
    for (int h = 0; h < n_kv_heads; h++) {  // Use n_kv_heads for GQA
        for (int t = 0; t < kv_len; t++) {
            for (int d = 0; d < head_dim; d++) {
                // Source: [d + t*head_dim + h*head_dim*kv_len]
                // Dest:   [d + t*head_dim + h*head_dim*kv_len]  (same layout)
                int src_idx = d + t * head_dim + h * head_dim * kv_len;
                int dst_idx = d + t * head_dim + h * head_dim * kv_len;

                if (k_std->type == GGML_TYPE_F32) {
                    k_f16[dst_idx] = ggml_fp32_to_fp16(k_f32_src[src_idx]);
                } else {
                    k_f16[dst_idx] = k_f16_src[src_idx];  // K is already F16, just copy
                }
                
                if (v_std->type == GGML_TYPE_F32) {
                    v_f16[dst_idx] = ggml_fp32_to_fp16(v_f32_src[src_idx]);
                } else {
                    v_f16[dst_idx] = v_f16_src[src_idx];
                }
            }
        }
    }

    const float scale = 1.0f / sqrtf((float)head_dim);

    ggml_tensor * standard_result = ggml_flash_attn_ext(
        ctx, q_std, k_std, v_std, NULL,  // Use NULL mask for comparison
        scale,
        0.0f,  // max_bias
        0.0f   // logit_softcap
    );
    ggml_flash_attn_ext_set_prec(standard_result, GGML_PREC_F32);

    if (!standard_result) {
        printf("ERROR: Failed to create standard flash attention operation\n");
        // free(workspace);
        ggml_free(ctx);
        return 1;
    }

    printf("Standard flash attention tensor created successfully\n");
    printf("Standard result shape: [%ld, %ld, %ld, %ld]\n",
           standard_result->ne[0], standard_result->ne[1], standard_result->ne[2], standard_result->ne[3]);

    // Build and execute computation graph for standard implementation
    struct ggml_cgraph * graph_standard = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph_standard, standard_result);

    printf("Computing standard flash attention...\n");
    enum ggml_status status_standard = ggml_graph_compute_with_ctx(ctx, graph_standard, n_threads);

    if (status_standard != GGML_STATUS_SUCCESS) {
        printf("ERROR: Standard flash attention computation failed with status: %d\n", status_standard);
        // free(workspace);
        ggml_free(ctx);
        return 1;
    }

    printf("Standard flash attention computation successful\n");

    // ============================================================================
    // Test 3: PyTorch Verification with scaled_dot_product_attention
    // ============================================================================
    printf("\n--- PyTorch Verification ---\n");
    
    // Variables to store PyTorch results for later comparison
    std::vector<float> torch_result_data;
    bool torch_success = false;
    
#ifdef LLAMA_TORCH_AVAILABLE
    try {
        // Convert data to torch tensors
        // PyTorch expects [batch_size, num_heads, seq_len, head_dim] format
        
        // Create torch tensors from existing data
        auto torch_options = torch::TensorOptions().dtype(torch::kFloat32);
        
        // Query: [1, n_heads, seq_len, head_dim]
        auto q_torch = torch::zeros({1, n_heads, seq_len, head_dim}, torch_options);
        float* q_torch_data = q_torch.data_ptr<float>();
        
        // Convert from ggml format [head_dim, seq_len, n_heads, 1] to torch format [1, n_heads, seq_len, head_dim]
        for (int h = 0; h < n_heads; h++) {
            for (int s = 0; s < seq_len; s++) {
                for (int d = 0; d < head_dim; d++) {
                    int ggml_idx = d + s * head_dim + h * head_dim * seq_len;
                    int torch_idx = h * seq_len * head_dim + s * head_dim + d;
                    q_torch_data[torch_idx] = ((float*)q->data)[ggml_idx];
                }
            }
        }
        
        // Key: [1, n_kv_heads, kv_len, head_dim]
        auto k_torch = torch::zeros({1, n_kv_heads, kv_len, head_dim}, torch_options);
        float* k_torch_data = k_torch.data_ptr<float>();
        
        // Convert from ggml format [head_dim, kv_len, n_kv_heads, 1] to torch format [1, n_kv_heads, kv_len, head_dim]
        for (int h = 0; h < n_kv_heads; h++) {
            for (int s = 0; s < kv_len; s++) {
                for (int d = 0; d < head_dim; d++) {
                    int ggml_idx = d + s * head_dim + h * head_dim * kv_len;
                    int torch_idx = h * kv_len * head_dim + s * head_dim + d;
                    // Convert F16 to F32
                    k_torch_data[torch_idx] = ggml_fp16_to_fp32(((ggml_fp16_t*)k->data)[ggml_idx]);
                }
            }
        }
        
        // Value: [1, n_kv_heads, kv_len, head_dim]  
        auto v_torch = torch::zeros({1, n_kv_heads, kv_len, head_dim}, torch_options);
        float* v_torch_data = v_torch.data_ptr<float>();
        
        // Convert from ggml format [head_dim, kv_len, n_kv_heads, 1] to torch format [1, n_kv_heads, kv_len, head_dim]
        for (int h = 0; h < n_kv_heads; h++) {
            for (int s = 0; s < kv_len; s++) {
                for (int d = 0; d < head_dim; d++) {
                    int ggml_idx = d + s * head_dim + h * head_dim * kv_len;
                    int torch_idx = h * kv_len * head_dim + s * head_dim + d;
                    // Convert F16 to F32
                    v_torch_data[torch_idx] = ggml_fp16_to_fp32(((ggml_fp16_t*)v->data)[ggml_idx]);
                }
            }
        }

        auto mask_torch = torch::zeros({1, n_heads, seq_len, kv_len}, torch_options);
        float* mask_torch_data = mask_torch.data_ptr<float>();

        for (int h = 0; h < n_heads; h++) {
            for (int s = 0; s < seq_len; s++) {
                for (int d = 0; d < kv_len; d++) {
                    int ggml_idx = d + s * kv_len + h * kv_len * seq_len;
                    int torch_idx = h * seq_len * kv_len + s * kv_len + d;
                    mask_torch_data[torch_idx] = 1.0f;
                }
            }
        }
        
        // For GQA (Grouped Query Attention), we need to repeat KV heads to match Q heads
        if (n_heads > n_kv_heads) {
            // Repeat KV heads
            k_torch = k_torch.repeat_interleave(n_heads / n_kv_heads, /*dim=*/1);
            v_torch = v_torch.repeat_interleave(n_heads / n_kv_heads, /*dim=*/1);
        }
        
        printf("PyTorch tensor shapes:\n");
        printf("  Q: [%ld, %ld, %ld, %ld]\n", q_torch.size(0), q_torch.size(1), q_torch.size(2), q_torch.size(3));
        printf("  K: [%ld, %ld, %ld, %ld]\n", k_torch.size(0), k_torch.size(1), k_torch.size(2), k_torch.size(3));
        printf("  V: [%ld, %ld, %ld, %ld]\n", v_torch.size(0), v_torch.size(1), v_torch.size(2), v_torch.size(3));

        // Compute scaled dot product attention
        float scale_factor = 1.0f / sqrtf((float)head_dim);
        auto torch_result = torch::scaled_dot_product_attention(
            q_torch, k_torch, v_torch, mask_torch, 
            /*dropout_p=*/0.0,
            /*is_causal=*/false,
            /*scale=*/scale_factor
        );
        
        printf("PyTorch result shape: [%ld, %ld, %ld, %ld]\n", 
               torch_result.size(0), torch_result.size(1), torch_result.size(2), torch_result.size(3));
        
        // Store PyTorch result data for later comparison
        float* torch_data_ptr = torch_result.data_ptr<float>();
        size_t torch_elements = torch_result.numel();
        torch_result_data.resize(torch_elements);
        
        // Convert torch result from [1, n_heads, seq_len, head_dim] to [head_dim, seq_len, n_heads, 1] format
        for (int h = 0; h < n_heads; h++) {
            for (int s = 0; s < seq_len; s++) {
                for (int d = 0; d < head_dim; d++) {
                    // PyTorch result format: [1, n_heads, seq_len, head_dim]
                    int torch_idx = h * seq_len * head_dim + s * head_dim + d;
                    // Custom result format: [head_dim, seq_len, n_heads, 1]
                    int custom_idx = d + s * head_dim + h * head_dim * seq_len;
                    torch_result_data[custom_idx] = torch_data_ptr[torch_idx];
                }
            }
        }
        
        torch_success = true;
        printf("PyTorch computation successful\n");
        
    } catch (const std::exception& e) {
        printf("PyTorch verification failed with exception: %s\n", e.what());
        printf("This might be due to PyTorch not being properly installed or linked.\n");
        torch_success = false;
    }
#else
    printf("PyTorch verification skipped (PyTorch not available)\n");
    torch_success = false;
#endif // LLAMA_TORCH_AVAILABLE

    // ============================================================================
    // Unified Comparison of Custom, PyTorch, and Standard Results
    // ============================================================================
    printf("\n--- Unified Results Comparison ---\n");

    float* custom_data = (float*)custom_result->data;
    float* standard_data = nullptr;

    // Handle different output types from standard flash attention
    std::vector<float> standard_f32_data;
    if (standard_result->type == GGML_TYPE_F16) {
        ggml_fp16_t* standard_f16 = (ggml_fp16_t*)standard_result->data;
        size_t n_elements = ggml_nelements(standard_result);
        standard_f32_data.resize(n_elements);
        for (size_t i = 0; i < n_elements; i++) {
            standard_f32_data[i] = ggml_fp16_to_fp32(standard_f16[i]);
        }
        standard_data = standard_f32_data.data();
    } else {
        standard_data = (float*)standard_result->data;
    }

    // Compare element by element
    size_t custom_elements = ggml_nelements(custom_result);
    size_t standard_elements = ggml_nelements(standard_result);

    printf("Result tensor information:\n");
    printf("  Custom result elements: %zu\n", custom_elements);
    printf("  Standard result elements: %zu\n", standard_elements);
    if (torch_success) {
        printf("  PyTorch result elements: %zu\n", torch_result_data.size());
    } else {
        printf("  PyTorch result: FAILED\n");
    }

    // Calculate comparison statistics
    float max_custom_standard = 0.0f, sum_custom_standard = 0.0f;
    float max_custom_torch = 0.0f, sum_custom_torch = 0.0f;
    float max_standard_torch = 0.0f, sum_standard_torch = 0.0f;
    size_t compared_elements = 0;

    // Compare the first min(custom_elements, standard_elements) elements
    size_t min_elements = std::min(custom_elements, standard_elements);
    if (torch_success) {
        min_elements = std::min(min_elements, torch_result_data.size());
    }

    for (size_t i = 0; i < min_elements; i++) {
        float custom_val = custom_data[i];
        float standard_val = standard_data[i];
        float torch_val = torch_success ? torch_result_data[i] : NAN;

        if (std::isfinite(custom_val) && std::isfinite(standard_val)) {
            float abs_diff_cs = std::abs(custom_val - standard_val);
            max_custom_standard = std::max(max_custom_standard, abs_diff_cs);
            sum_custom_standard += abs_diff_cs;

            if (torch_success && std::isfinite(torch_val)) {
                float abs_diff_ct = std::abs(custom_val - torch_val);
                float abs_diff_st = std::abs(standard_val - torch_val);
                max_custom_torch = std::max(max_custom_torch, abs_diff_ct);
                max_standard_torch = std::max(max_standard_torch, abs_diff_st);
                sum_custom_torch += abs_diff_ct;
                sum_standard_torch += abs_diff_st;
            }
            compared_elements++;
        }
    }

    // Print detailed comparison table
    printf("\nDetailed Comparison Table (first 128 elements):\n");
    if (torch_success) {
        printf("Index | Custom      | Standard    | PyTorch     | C-S Diff    | C-P Diff    | S-P Diff\n");
        printf("------|-------------|-------------|-------------|-------------|-------------|----------\n");
    } else {
        printf("Index | Custom      | Standard    | C-S Diff\n");
        printf("------|-------------|-------------|-----------\n");
    }

    size_t show_elements = std::min(size_t(128), min_elements);
    for (size_t i = 0; i < show_elements; i++) {
        float custom_val = custom_data[i];
        float standard_val = standard_data[i];

        if (torch_success) {
            float torch_val = torch_result_data[i];
            
            if (std::isfinite(custom_val) && std::isfinite(standard_val) && std::isfinite(torch_val)) {
                float abs_diff_cs = std::abs(custom_val - standard_val);
                float abs_diff_ct = std::abs(custom_val - torch_val);
                float abs_diff_st = std::abs(standard_val - torch_val);
                printf("%5zu | %11.6f | %11.6f | %11.6f | %.6e | %.6e | %.6e\n", 
                       i, custom_val, standard_val, torch_val, abs_diff_cs, abs_diff_ct, abs_diff_st);
            } else {
                printf("%5zu | %11.6f | %11.6f | %11.6f |     N/A     |     N/A     |     N/A\n", 
                       i, custom_val, standard_val, torch_val);
            }
        } else {
            if (std::isfinite(custom_val) && std::isfinite(standard_val)) {
                float abs_diff_cs = std::abs(custom_val - standard_val);
                printf("%5zu | %11.6f | %11.6f | %.6e\n", i, custom_val, standard_val, abs_diff_cs);
            } else {
                printf("%5zu | %11.6f | %11.6f |     N/A\n", i, custom_val, standard_val);
            }
        }
    }

    // Print comparison statistics
    printf("\nComparison Statistics:\n");
    printf("  Total compared elements: %zu\n", compared_elements);
    
    if (compared_elements > 0) {
        float avg_custom_standard = sum_custom_standard / compared_elements;
        printf("  Custom vs Standard:\n");
        printf("    Max absolute difference: %.6e\n", max_custom_standard);
        printf("    Average absolute difference: %.6e\n", avg_custom_standard);
        
        if (torch_success) {
            float avg_custom_torch = sum_custom_torch / compared_elements;
            float avg_standard_torch = sum_standard_torch / compared_elements;
            printf("  Custom vs PyTorch:\n");
            printf("    Max absolute difference: %.6e\n", max_custom_torch);
            printf("    Average absolute difference: %.6e\n", avg_custom_torch);
            printf("  Standard vs PyTorch:\n");
            printf("    Max absolute difference: %.6e\n", max_standard_torch);
            printf("    Average absolute difference: %.6e\n", avg_standard_torch);
        }
    } else {
        printf("  No finite elements to compare\n");
    }

    // Determine test result - adjust tolerance for F16 precision
    const float tolerance = 1e-3f;  // Tolerance for F16 numerical differences
    bool test_passed = (compared_elements > 0) && (max_custom_standard < tolerance);
    
    if (torch_success) {
        bool torch_test_passed = (compared_elements > 0) && (max_custom_torch < tolerance);
        test_passed = test_passed && torch_test_passed;
    }

    printf("\nOverall Test Result: %s\n", test_passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m");
    printf("  Custom vs Standard: %s (max diff: %.6e)\n", 
           (compared_elements > 0 && max_custom_standard < tolerance) ? "PASS" : "FAIL", 
           max_custom_standard);
    
    if (torch_success) {
        printf("  Custom vs PyTorch: %s (max diff: %.6e)\n", 
               (compared_elements > 0 && max_custom_torch < tolerance) ? "PASS" : "FAIL", 
               max_custom_torch);
        printf("  Standard vs PyTorch: %s (max diff: %.6e)\n", 
               (compared_elements > 0 && max_standard_torch < tolerance) ? "PASS" : "FAIL", 
               max_standard_torch);
    } else {
        printf("  PyTorch comparison: SKIPPED (PyTorch failed)\n");
    }

    // Cleanup
    // free(workspace);
    ggml_free(ctx);

    return test_passed ? 0 : 1;
}
