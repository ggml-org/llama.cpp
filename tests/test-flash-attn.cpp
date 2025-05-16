#include "log.h"
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <string>
#include <vector>
#include <numeric> // For std::iota if needed, or manual loops

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#endif

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

//
// logging
//

#if (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG(...)
#endif

#if (GGML_DEBUG >= 5)
#define GGML_PRINT_DEBUG_5(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_5(...)
#endif

#if (GGML_DEBUG >= 10)
#define GGML_PRINT_DEBUG_10(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_10(...)
#endif

#define GGML_PRINT(...) printf(__VA_ARGS__)

static float frand(void) {
    return (float)rand()/(float)RAND_MAX;
}

static struct ggml_tensor * get_ones_tensor_f32(
        struct ggml_context * ctx0,
        int ndims,
        const int64_t ne[]) {
    struct ggml_tensor * result = ggml_new_tensor(ctx0, GGML_TYPE_F32, ndims, ne);
    ggml_set_f32(result, 1.0f);
    return result;
}

static struct ggml_tensor * get_random_tensor_f32(
        struct ggml_context * ctx0,
        int ndims,
        const int64_t ne[],
        float fmin,
        float fmax) {
    struct ggml_tensor * result = ggml_new_tensor(ctx0, GGML_TYPE_F32, ndims, ne);

    // Initialize with random data
    float *data = (float *)result->data;
    for (int i = 0; i < ggml_nelements(result); ++i) {
        data[i] = i % static_cast<int>(fmax - fmin) + fmin;
    }
    return result;
}

static struct ggml_tensor * get_ones_tensor_f16(
        struct ggml_context * ctx0,
        int ndims,
        const int64_t ne[]) {
    struct ggml_tensor * result = ggml_new_tensor(ctx0, GGML_TYPE_F16, ndims, ne);
    ggml_set_f32(result, 1.0f); // ggml_set_f32 handles conversion to f16 internally
    return result;
}

static struct ggml_tensor * get_random_tensor_f16(
        struct ggml_context * ctx0,
        int ndims,
        const int64_t ne[],
        float fmin,
        float fmax) {
    struct ggml_tensor * result = ggml_new_tensor(ctx0, GGML_TYPE_F16, ndims, ne);

    // Initialize with random data
    ggml_fp16_t *data = (ggml_fp16_t *)result->data;
    for (int i = 0; i < ggml_nelements(result); ++i) {
        float val = i % static_cast<int>(fmax - fmin) + fmin;
        data[i] = ggml_fp32_to_fp16(val);
    }
    return result;
}

static std::string ggml_tensor_shape_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS && t->ne[i+1] > 0) { // Print comma only if next dim exists
             if (i < GGML_MAX_DIMS -1 && t->ne[i+1] != 0 ) { // check if there is a next dimension
                bool has_more_dims = false;
                for(int j=i+1; j < GGML_MAX_DIMS; ++j) {
                    if (t->ne[j] != 0 && t->ne[j] != 1) { // only count meaningful dims
                        has_more_dims = true;
                        break;
                    }
                }
                if(has_more_dims || (i<2 && t->ne[i+1] > 1)) str += ", "; // Heuristic for 1D/2D vs higher D
             }
        }
    }
    // Remove trailing comma and space if any for tensors with fewer than MAX_DIMS
    if (str.length() > 2 && str.substr(str.length() - 2) == ", ") {
        str = str.substr(0, str.length() - 2);
    }
    return str;
}

static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads, nullptr);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    } else {
        plan.work_data = nullptr; // Ensure work_data is null if work_size is 0
    }

    ggml_graph_compute(graph, &plan);
}

static void ggml_print_tensor_summary(const char* title, const ggml_tensor *t) {
    if (!t) return;
    LOG("%s: %s, Type: %s, Shape: [%s]\n",
        title,
        (t->name[0] != '\0' ? t->name : "(unnamed)"),
        ggml_type_name(t->type),
        ggml_tensor_shape_string(t).c_str());
}

static void ggml_print_tensor_data(const ggml_tensor * t, uint8_t * data_ptr_override, int64_t n_to_print) {
    ggml_print_tensor_summary("Tensor Data Dump", t);

    uint8_t * data_to_print = data_ptr_override;
    if (!data_to_print) {
        LOG(" (Data not available or not on host for direct printing)\n");
        return;
    }
    if (ggml_is_quantized(t->type)) {
        LOG(" (Quantized tensor - data printing not implemented for this example)\n");
        return;
    }

    GGML_ASSERT(n_to_print > 0);
    float sum = 0;
    const int64_t* ne = t->ne;
    const size_t* nb = t->nb;
    ggml_type type = t->type;

    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        LOG("                                     [\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n_to_print && ne[2] > 2*n_to_print) {
                LOG("                                      ..., \n");
                i2 = ne[2] - n_to_print;
            }
            LOG("                                      [\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n_to_print && ne[1] > 2*n_to_print) {
                    LOG("                                       ..., \n");
                    i1 = ne[1] - n_to_print;
                }
                LOG("                                       [");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n_to_print && ne[0] > 2*n_to_print) {
                        LOG("..., ");
                        i0 = ne[0] - n_to_print;
                    }
                    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    float v;
                    if (type == GGML_TYPE_F16) {
                        v = ggml_fp16_to_fp32(*(ggml_fp16_t *) &data_to_print[i]);
                    } else if (type == GGML_TYPE_F32) {
                        v = *(float *) &data_to_print[i];
                    } else if (type == GGML_TYPE_I32) {
                        v = (float) *(int32_t *) &data_to_print[i];
                    } else if (type == GGML_TYPE_I16) {
                        v = (float) *(int16_t *) &data_to_print[i];
                    } else if (type == GGML_TYPE_I8) {
                        v = (float) *(int8_t *) &data_to_print[i];
                    } else {
                        LOG("Unsupported type for printing: %s\n", ggml_type_name(type));
                        GGML_ABORT("fatal error: unsupported tensor type in ggml_print_tensor_data");
                    }
                    LOG("%12.4f", v);
                    sum += v;
                    if (i0 < ne[0] - 1) LOG(", ");
                }
                LOG("],\n");
            }
            LOG("                                      ],\n");
        }
        LOG("                                     ]\n");
        LOG("                                     sum = %f\n", sum);
    }
}

static void get_tensor_data_if_needed(struct ggml_tensor * t, std::vector<uint8_t>& buffer, uint8_t** data_ptr) {
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    if (is_host) {
        *data_ptr = (uint8_t *)t->data;
    } else {
        if (t->data == nullptr && ggml_nbytes(t) > 0) {     // Tensor might have data on device but t->data is null if not mapped
            LOG("Tensor %s data is on device and not mapped to host, attempting to fetch.\n", (t->name[0] != '\0' ? t->name : "(unnamed)"));
        } else if (t->data == nullptr && ggml_nbytes(t) == 0) {
             LOG("Tensor %s has no data (0 bytes).\n", (t->name[0] != '\0' ? t->name : "(unnamed)"));
            *data_ptr = nullptr;
            return;
        }
        auto n_bytes = ggml_nbytes(t);
        buffer.resize(n_bytes);
        ggml_backend_tensor_get(t, buffer.data(), 0, n_bytes);
        *data_ptr = buffer.data();
    }
}

// helper to print a tensor (first few elements)
static void print_tensor_brief(const struct ggml_tensor * tensor, const char * name) {
    printf("%s: shape(%ld, %ld, %ld, %ld), type %s, backend %d\n",
        name, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
        ggml_type_name(tensor->type), 0);
    if (tensor->data == nullptr) {
        printf("  (data is null - graph not computed or offloaded?)\n");
        return;
    }
    const float * data = (const float *)tensor->data;
    int n_to_print = (int)MIN(10, ggml_nelements(tensor));
    printf("  Data: ");
    for (int i = 0; i < n_to_print; ++i) {
        printf("%.4f ", data[i]);
    }
    if (ggml_nelements(tensor) > n_to_print) {
        printf("...");
    }
    printf("\n\n");
}

int main(int /*argc*/, const char ** /*argv*/) {
    srand(2024); // for reproducibility

    struct ggml_init_params params = {
        /* .mem_size   = */ 256 * 1024 * 1024, // 256 MB, Flash Attention can be memory intensive
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    std::vector<uint8_t> work_buffer;

    struct ggml_context * ctx0 = ggml_init(params);

    // Define tensor dimensions for Flash Attention
    // Q: (head_dim, seq_len_q, n_head, batch_size)
    // K: (head_dim, seq_len_kv, n_head_kv, batch_size)
    // V: (head_dim, seq_len_kv, n_head_kv, batch_size)
    // Result: (head_dim, seq_len_q, n_head, batch_size) - Note: ggml_flash_attn_ext output has permuted shape

    const int64_t batch_size  = 1;
    const int64_t n_head      = 1;      // Query heads
    const int64_t n_head_kv   = 1;      // KV heads (n_head if not GQA/MQA)
    const int64_t seq_len_q   = 1;      // Query sequence length
    const int64_t seq_len_kv  = 1;      // Key/Value sequence length
    const int64_t head_dim    = 128;    // Dimension of each attention head

    const int64_t ne_q[4] = {head_dim, seq_len_q,  n_head,    batch_size};
    const int64_t ne_k[4] = {head_dim, seq_len_kv, n_head_kv, batch_size};
    const int64_t ne_v[4] = {head_dim, seq_len_kv, n_head_kv, batch_size}; // Assuming head_dim_v = head_dim

    struct ggml_tensor * q = get_random_tensor_f32(ctx0, 4, ne_q, -128.0f, 128.0f);
    struct ggml_tensor * k = get_random_tensor_f32(ctx0, 4, ne_k, -128.0f, 128.0f);
    struct ggml_tensor * v = get_random_tensor_f32(ctx0, 4, ne_v, -128.0f, 128.0f);
    
    //> ===================================================================================================
    //> Print the shapes of Q, K, V tensors
    //> ===================================================================================================
    struct ggml_tensor * mask = NULL; // No mask for this basic example

    // Convert to float16
    q = ggml_cast(ctx0, q, GGML_TYPE_F16);
    k = ggml_cast(ctx0, k, GGML_TYPE_F16);
    v = ggml_cast(ctx0, v, GGML_TYPE_F16);

    const float scale = 1.0f / sqrtf((float)head_dim);
    const float max_bias = 0.0f; // No ALIBI
    const float logit_softcap = 0.0f; // No logit softcapping

    printf("Constructing ggml_flash_attn_ext...\n");
    struct ggml_tensor * flash_attn_output = ggml_flash_attn_ext(ctx0, q, k, v, mask, scale, max_bias, logit_softcap);
    ggml_set_name(flash_attn_output, "flash_attn_output");

    //> ===================================================================================================
    //> Standard Attention Calculation for comparison
    //> ===================================================================================================
    printf("\nConstructing Standard Attention path...\n");
    struct ggml_tensor * q_std = ggml_cast(ctx0, ggml_dup(ctx0, q), GGML_TYPE_F32);
    struct ggml_tensor * k_std = ggml_cast(ctx0, ggml_dup(ctx0, k), GGML_TYPE_F32);
    struct ggml_tensor * v_std = ggml_cast(ctx0, ggml_dup(ctx0, v), GGML_TYPE_F32);

    ggml_set_name(q_std, "q_std");
    ggml_set_name(k_std, "k_std");
    ggml_set_name(v_std, "v_std");

    struct ggml_tensor * output_std = ggml_mul_mat(ctx0, k_std, q_std);
    ggml_set_name(output_std, "output_std");

    struct ggml_tensor * output_std_softmax = ggml_soft_max_ext(ctx0, output_std, mask, scale, max_bias);
    ggml_set_name(output_std_softmax, "output_std_softmax");

    struct ggml_tensor * v_std_permuted = ggml_view_3d(
        ctx0, 
        v_std,
        v_std->ne[1], 
        v_std->ne[0],
        v_std->ne[2],
        ggml_type_size(v_std->type) * v_std->ne[1],
        ggml_type_size(v_std->type) * v_std->ne[1] * v_std->ne[0],
        0
    );
    ggml_set_name(v_std_permuted, "v_std_permuted");

    struct ggml_tensor * output_std_mul_v = ggml_mul_mat(ctx0, v_std_permuted, output_std_softmax);
    ggml_set_name(output_std_mul_v, "output_std_mul_v");

    //> ===================================================================================================
    //> Build and compute graph
    //> ===================================================================================================
    // Build and compute graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    ggml_build_forward_expand(gf, flash_attn_output);
    ggml_build_forward_expand(gf, output_std_mul_v); // Add standard attention output to graph

    printf("Computing graph...\n");
    ggml_graph_compute_helper(work_buffer, gf, 1);  // Using 1 thread for simplicity
    
    //> Print the data of the flash_attn_output tensor
    printf("\n--- Flash Attention Output ---\n");
    uint8_t* q_data = (uint8_t*)malloc(ggml_nbytes(q));
    std::vector<uint8_t> buffer;
    get_tensor_data_if_needed(q, buffer, &q_data);
    ggml_print_tensor_data(flash_attn_output, q_data, 128);


    printf("\n--- Output Tensor ---\n");
    print_tensor_brief(flash_attn_output, "Flash Attention Output");

    printf("\n--- Standard Attention Output ---\n");
    print_tensor_brief(output_std_mul_v, "Standard Attention Output");

    // Expected output shape from ggml.c: { v->ne[0], q->ne[2], q->ne[1], q->ne[3] }
    // Which is (head_dim, n_head, seq_len_q, batch_size)
    printf("\nExpected output shape: (%lld, %lld, %lld, %lld)\n", head_dim, n_head, seq_len_q, batch_size);

    ggml_free(ctx0);

    return 0;
} 