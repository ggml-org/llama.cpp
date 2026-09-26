#include "ggml.h"
#include "ggml-sycl.h"
#include "ggml-backend.h"
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstring>

extern "C" {
    void quantize_row_turbo2_0_ref(const float * x, void * y, int64_t k);
    void dequantize_row_turbo2_0(const void * x, float * y, int64_t k);
    void quantize_row_turbo3_0_ref(const float * x, void * y, int64_t k);
    void dequantize_row_turbo3_0(const void * x, float * y, int64_t k);
    void quantize_row_turbo4_0_ref(const float * x, void * y, int64_t k);
    void dequantize_row_turbo4_0(const void * x, float * y, int64_t k);
    
    void quantize_row_tq3_1s_ref(const float * x, void * y, int64_t k);
    void dequantize_row_tq3_1s(const void * x, float * y, int64_t k);
    void quantize_row_tq4_1s_ref(const float * x, void * y, int64_t k);
    void dequantize_row_tq4_1s(const void * x, float * y, int64_t k);

    void turbo_cpu_fwht(float * x, int group_size);
}

static void run_test(ggml_backend_t backend, ggml_type type, const char * name, 
                    void (*quant_ref)(const float *, void *, int64_t),
                    void (*dequant_ref)(const void *, float *, int64_t)) {
    const int d = 128;
    printf("Testing %s quantization on SYCL...\n", name);

    struct ggml_init_params params_sycl = { 2 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx_sycl = ggml_init(params_sycl);

    // 1. Create tensors in context
    struct ggml_tensor * input   = ggml_new_tensor_1d(ctx_sycl, GGML_TYPE_F32, d);
    struct ggml_tensor * output  = ggml_new_tensor_1d(ctx_sycl, type, d);
    struct ggml_tensor * indices = ggml_new_tensor_1d(ctx_sycl, GGML_TYPE_I32, 1);
    
    // 2. Build graph (creates views)
    struct ggml_tensor * view = ggml_set_rows(ctx_sycl, output, input, indices);

    // 3. Allocate tensors on backend
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_sycl, backend);

    // 4. Set input data
    std::vector<float> host_input(d);
    for (int i = 0; i < d; i++) {
        host_input[i] = sinf(i * 0.1f + 0.5f) * 10.0f;
    }
    ggml_backend_tensor_set(input, host_input.data(), 0, d * sizeof(float));

    int32_t host_indices[1] = { 0 };
    ggml_backend_tensor_set(indices, host_indices, 0, sizeof(int32_t));

    struct ggml_cgraph * gf = ggml_new_graph(ctx_sycl);
    ggml_build_forward_expand(gf, view);

    // 5. Compute
    ggml_backend_graph_compute(backend, gf);

    // 6. Copy back
    std::vector<char> sycl_data(ggml_nbytes(output));
    ggml_backend_tensor_get(output, sycl_data.data(), 0, sycl_data.size());

    // 7. Ref
    std::vector<char> ref_data(ggml_nbytes(output));
    quant_ref(host_input.data(), ref_data.data(), d);

    // 8. Compare
    std::vector<float> sycl_float(d);
    std::vector<float> ref_float(d);

    dequant_ref(sycl_data.data(), sycl_float.data(), d);
    dequant_ref(ref_data.data(), ref_float.data(), d);

    float mse = 0;
    float cosv = 0;
    float ni = 0;
    float no = 0;
    for (int i = 0; i < d; i++) {
        float s = sycl_float[i];
        float r = ref_float[i];
        mse += (s - r) * (s - r);
        cosv += s * r;
        ni += r * r;
        no += s * s;
    }

    float cosine = cosv / (sqrtf(ni) * sqrtf(no));
    printf("  MSE: %.8f, Cosine: %.6f\n", mse / d, cosine);
    if (mse / d > 1e-5 || cosine < 0.99) {
        printf("  FAILED\n");
    } else {
        printf("  PASSED\n");
    }

    ggml_free(ctx_sycl);
    ggml_backend_buffer_free(buffer);
}

static void run_weight_test(ggml_backend_t backend, ggml_type type, const char * name,
                           void (*quant_ref)(const float *, void *, int64_t),
                           void (*dequant_ref)(const void *, float *, int64_t)) {
    const int M = 32; // rows
    const int K = 128; // cols
    printf("Testing %s weight multiplication on SYCL...\n", name);

    struct ggml_init_params params_sycl = { 2 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx_sycl = ggml_init(params_sycl);

    // 1. Create tensors
    struct ggml_tensor * weights = ggml_new_tensor_2d(ctx_sycl, type, K, M);
    struct ggml_tensor * input   = ggml_new_tensor_1d(ctx_sycl, GGML_TYPE_F32, K);
    struct ggml_tensor * result  = ggml_mul_mat(ctx_sycl, weights, input);

    // 2. Allocate
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_sycl, backend);

    // 3. Set data
    std::vector<float> host_weights(M * K);
    for (int i = 0; i < M * K; i++) host_weights[i] = (float)((i % 7 - 3) * 0.1);
    
    std::vector<char> quantized_weights(ggml_nbytes(weights));
    quant_ref(host_weights.data(), quantized_weights.data(), M * K);
    ggml_backend_tensor_set(weights, quantized_weights.data(), 0, quantized_weights.size());

    std::vector<float> host_input(K);
    for (int i = 0; i < K; i++) host_input[i] = (float)((i % 5 - 2) * 0.2);
    ggml_backend_tensor_set(input, host_input.data(), 0, K * sizeof(float));

    // 4. Compute
    struct ggml_cgraph * gf = ggml_new_graph(ctx_sycl);
    ggml_build_forward_expand(gf, result);
    ggml_backend_graph_compute(backend, gf);

    // 5. Copy back
    std::vector<float> sycl_result(M);
    ggml_backend_tensor_get(result, sycl_result.data(), 0, M * sizeof(float));

    // 6. Compute CPU reference
    // W_orig @ input
    std::vector<float> ref_result(M);
    size_t row_size = ggml_row_size(type, K);
    for (int i = 0; i < M; i++) {
        float sum = 0;
        // TQ weights are already dequantized to W_orig by dequant_ref
        std::vector<float> dequant_weights(K);
        dequant_ref(quantized_weights.data() + i * row_size, dequant_weights.data(), K);
        
        for (int j = 0; j < K; j++) {
            sum += dequant_weights[j] * host_input[j];
        }
        ref_result[i] = sum;
    }

    // 7. Compare
    float mse = 0;
    float cosv = 0, ni = 0, no = 0;
    for (int i = 0; i < M; i++) {
        float s = sycl_result[i], r = ref_result[i];
        mse += (s-r)*(s-r); cosv += s*r; ni += r*r; no += s*s;
    }
    float cosine = cosv / (sqrtf(ni) * sqrtf(no));
    printf("  MSE: %.8f, Cosine: %.6f\n", mse / M, cosine);
    if (mse / M > 1e-4 || cosine < 0.99) {
        printf("  FAILED\n");
    } else {
        printf("  PASSED\n");
    }

    ggml_free(ctx_sycl);
    ggml_backend_buffer_free(buffer);
}

int main() {
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        fprintf(stderr, "Failed to initialize SYCL backend\n");
        return 1;
    }

    run_test(backend, GGML_TYPE_TURBO3_0, "TURBO3_0", quantize_row_turbo3_0_ref, dequantize_row_turbo3_0);
    run_test(backend, GGML_TYPE_TURBO2_0, "TURBO2_0", quantize_row_turbo2_0_ref, dequantize_row_turbo2_0);
    run_test(backend, GGML_TYPE_TURBO4_0, "TURBO4_0", quantize_row_turbo4_0_ref, dequantize_row_turbo4_0);

    run_weight_test(backend, GGML_TYPE_TQ3_1S, "TQ3_1S", quantize_row_tq3_1s_ref, dequantize_row_tq3_1s);
    run_weight_test(backend, GGML_TYPE_TQ4_1S, "TQ4_1S", quantize_row_tq4_1s_ref, dequantize_row_tq4_1s);

    ggml_backend_free(backend);
    return 0;
}
