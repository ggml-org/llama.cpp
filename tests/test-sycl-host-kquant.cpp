// SYCL Host-Weight K-Quant Test: Validate unified-cache layouts for Q4_K/Q5_K
// Builds a graph with both small (MMVQ) and large (MMQ) batch sizes using host weights
// and compares SYCL outputs against CPU reference.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

static float max_diff(const float * a, const float * b, size_t n) {
    float max_d = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float d = fabsf(a[i] - b[i]);
        if (d > max_d) {
            max_d = d;
        }
    }
    return max_d;
}

static void get_tolerances(ggml_type type, float & abs_tol, float & rel_tol) {
    abs_tol = 1.0f;
    rel_tol = 0.35f;
    if (type == GGML_TYPE_Q5_K) {
        abs_tol = 0.8f;
        rel_tol = 0.30f;
    }
}

static void dequantize_row_q8_1_ref(const block_q8_1 * x, float * y, int64_t k) {
    const int nb = k / QK8_1;
    for (int i = 0; i < nb; ++i) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK8_1; ++j) {
            y[i * QK8_1 + j] = d * x[i].qs[j];
        }
    }
}

static void build_input_reference_q8_1(const float * input_data, int n_embd, int n_tokens,
                                       std::vector<float> & input_ref) {
    input_ref.resize(n_embd * n_tokens);
    const int blocks = n_embd / QK8_1;
    std::vector<block_q8_1> q8(n_tokens * blocks);
    for (int t = 0; t < n_tokens; ++t) {
        quantize_row_q8_1_ref(input_data + t * n_embd, q8.data() + t * blocks, n_embd);
        dequantize_row_q8_1_ref(q8.data() + t * blocks, input_ref.data() + t * n_embd, n_embd);
    }
}

static bool run_reference_mul_mat(ggml_type weight_type, const void * weight_data,
                                  const float * input_data, int n_embd, int n_rows, int n_tokens,
                                  std::vector<float> & output) {
    std::vector<float> input_ref;
    build_input_reference_q8_1(input_data, n_embd, n_tokens, input_ref);
    output.assign(n_rows * n_tokens, 0.0f);

    const int blocks_per_row = n_embd / QK_K;
    std::vector<float> weight_f32(n_embd);

    for (int row = 0; row < n_rows; ++row) {
        const int row_offset = row * blocks_per_row;
        if (weight_type == GGML_TYPE_Q4_K) {
            dequantize_row_q4_K(reinterpret_cast<const block_q4_K *>(weight_data) + row_offset,
                                weight_f32.data(), n_embd);
        } else if (weight_type == GGML_TYPE_Q5_K) {
            dequantize_row_q5_K(reinterpret_cast<const block_q5_K *>(weight_data) + row_offset,
                                weight_f32.data(), n_embd);
        } else {
            return false;
        }

        for (int t = 0; t < n_tokens; ++t) {
            float sum = 0.0f;
            const float * x = input_ref.data() + t * n_embd;
            for (int k = 0; k < n_embd; ++k) {
                sum += weight_f32[k] * x[k];
            }
            output[t * n_rows + row] = sum;
        }
    }

    return true;
}

static ggml_backend_buffer_t alloc_tensor_buffer(ggml_backend_buffer_type_t buft, ggml_tensor * tensor,
                                                 ggml_backend_buffer_usage usage) {
    const size_t size = ggml_backend_buft_get_alloc_size(buft, tensor);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
    if (!buffer) {
        return nullptr;
    }
    ggml_backend_buffer_set_usage(buffer, usage);
    ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
    return buffer;
}

static bool run_mul_mat_dual(ggml_backend_t backend, ggml_type weight_type,
                             const void * weight_data, size_t weight_bytes,
                             const float * input_small, int n_tokens_small,
                             const float * input_large, int n_tokens_large,
                             int n_embd, int n_rows,
                             std::vector<float> & out_small,
                             std::vector<float> & out_large) {
    ggml_backend_buffer_type_t weight_buft = ggml_backend_sycl_host_buffer_type();
    ggml_backend_buffer_type_t dev_buft = ggml_backend_get_default_buffer_type(backend);

    const ggml_init_params params = {
        64 * 1024 * 1024,
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, weight_type, n_embd, n_rows);
    ggml_set_name(weight, "host_weight");
    ggml_tensor * input_s = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens_small);
    ggml_set_name(input_s, "input_small");
    ggml_tensor * input_l = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens_large);
    ggml_set_name(input_l, "input_large");

    ggml_tensor * out_s = ggml_mul_mat(ctx, weight, input_s);
    ggml_set_name(out_s, "out_small");
    ggml_tensor * out_l = ggml_mul_mat(ctx, weight, input_l);
    ggml_set_name(out_l, "out_large");

    ggml_backend_buffer_t weight_buf = alloc_tensor_buffer(weight_buft, weight, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (!weight_buf) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev) {
        ggml_backend_sycl_register_host_weight_tensor(dev, weight);
    }

    ggml_backend_buffer_t input_s_buf = alloc_tensor_buffer(dev_buft, input_s, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_t input_l_buf = alloc_tensor_buffer(dev_buft, input_l, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_t out_s_buf   = alloc_tensor_buffer(dev_buft, out_s, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_t out_l_buf   = alloc_tensor_buffer(dev_buft, out_l, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    if (!input_s_buf || !input_l_buf || !out_s_buf || !out_l_buf) {
        ggml_backend_buffer_free(weight_buf);
        if (input_s_buf) ggml_backend_buffer_free(input_s_buf);
        if (input_l_buf) ggml_backend_buffer_free(input_l_buf);
        if (out_s_buf) ggml_backend_buffer_free(out_s_buf);
        if (out_l_buf) ggml_backend_buffer_free(out_l_buf);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(weight, weight_data, 0, weight_bytes);
    ggml_backend_tensor_set(input_s, input_small, 0, n_embd * n_tokens_small * sizeof(float));
    ggml_backend_tensor_set(input_l, input_large, 0, n_embd * n_tokens_large * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out_s);
    ggml_build_forward_expand(graph, out_l);

    enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(weight_buf);
        ggml_backend_buffer_free(input_s_buf);
        ggml_backend_buffer_free(input_l_buf);
        ggml_backend_buffer_free(out_s_buf);
        ggml_backend_buffer_free(out_l_buf);
        ggml_free(ctx);
        return false;
    }

    out_small.resize(n_rows * n_tokens_small);
    out_large.resize(n_rows * n_tokens_large);
    ggml_backend_tensor_get(out_s, out_small.data(), 0, out_small.size() * sizeof(float));
    ggml_backend_tensor_get(out_l, out_large.data(), 0, out_large.size() * sizeof(float));

    ggml_backend_buffer_free(weight_buf);
    ggml_backend_buffer_free(input_s_buf);
    ggml_backend_buffer_free(input_l_buf);
    ggml_backend_buffer_free(out_s_buf);
    ggml_backend_buffer_free(out_l_buf);
    ggml_free(ctx);
    return true;
}

static bool compare_outputs(const char * label, ggml_type type,
                            const std::vector<float> & sycl_out,
                            const std::vector<float> & cpu_out) {
    if (sycl_out.size() != cpu_out.size()) {
        printf("  FAIL: %s size mismatch sycl=%zu cpu=%zu\n", label, sycl_out.size(), cpu_out.size());
        return false;
    }

    float abs_tol = 0.0f;
    float rel_tol = 0.0f;
    get_tolerances(type, abs_tol, rel_tol);

    float max_d = max_diff(sycl_out.data(), cpu_out.data(), sycl_out.size());
    float max_rel = 0.0f;
    bool has_nan = false;
    for (size_t i = 0; i < sycl_out.size(); ++i) {
        const float s = sycl_out[i];
        if (std::isnan(s) || std::isinf(s)) {
            has_nan = true;
        }
        const float diff = fabsf(sycl_out[i] - cpu_out[i]);
        const float denom = fmaxf(1.0f, fabsf(cpu_out[i]));
        max_rel = fmaxf(max_rel, diff / denom);
    }

    if (has_nan) {
        printf("  FAIL: %s contains NaN/Inf\n", label);
        return false;
    }
    if (max_d > abs_tol && max_rel > rel_tol) {
        printf("  FAIL: %s max_diff=%.6f (abs_tol=%.6f, rel_tol=%.6f)\n", label, max_d, abs_tol, rel_tol);
        return false;
    }

    printf("  PASS: %s max_diff=%.6f\n", label, max_d);
    return true;
}

static bool test_kquant_type(const char * name, ggml_type type, int n_embd, int n_rows,
                             int n_tokens_small, int n_tokens_large) {
    printf("\n=== %s host-weight unified-cache test ===\n", name);

    ggml_backend_t sycl_backend = ggml_backend_sycl_init(0);
    if (!sycl_backend) {
        printf("  SKIP: Could not initialize SYCL backend\n");
        return true;
    }

    const auto * qfns = ggml_get_type_traits(type);
    const auto * qfns_cpu = ggml_get_type_traits_cpu(type);
    if (!qfns || !qfns_cpu || !qfns_cpu->from_float || qfns->blck_size == 0) {
        printf("  SKIP: Quantization functions not available for %s\n", name);
        ggml_backend_free(sycl_backend);
        return true;
    }

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> weight_float(n_rows * n_embd);
    for (auto & v : weight_float) v = dist(rng);

    const size_t block_size = qfns->blck_size;
    const size_t type_size = qfns->type_size;
    const size_t nblocks = (n_rows * n_embd) / block_size;
    const size_t weight_bytes = nblocks * type_size;

    std::vector<uint8_t> weight_quant(weight_bytes);
    qfns_cpu->from_float(weight_float.data(), weight_quant.data(), n_rows * n_embd);

    std::vector<float> input_small(n_embd * n_tokens_small);
    std::vector<float> input_large(n_embd * n_tokens_large);
    for (auto & v : input_small) v = dist(rng);
    for (auto & v : input_large) v = dist(rng);

    std::vector<float> sycl_small;
    std::vector<float> sycl_large;
    if (!run_mul_mat_dual(sycl_backend, type, weight_quant.data(), weight_bytes,
                          input_small.data(), n_tokens_small,
                          input_large.data(), n_tokens_large,
                          n_embd, n_rows, sycl_small, sycl_large)) {
        printf("  FAIL: SYCL compute failed for %s\n", name);
        ggml_backend_free(sycl_backend);
        return false;
    }

    std::vector<float> cpu_small;
    std::vector<float> cpu_large;
    if (!run_reference_mul_mat(type, weight_quant.data(),
                               input_small.data(), n_embd, n_rows, n_tokens_small,
                               cpu_small)) {
        printf("  FAIL: CPU reference failed for %s small batch\n", name);
        ggml_backend_free(sycl_backend);
        return false;
    }
    if (!run_reference_mul_mat(type, weight_quant.data(),
                               input_large.data(), n_embd, n_rows, n_tokens_large,
                               cpu_large)) {
        printf("  FAIL: CPU reference failed for %s large batch\n", name);
        ggml_backend_free(sycl_backend);
        return false;
    }

    bool ok = true;
    ok = compare_outputs("small-batch", type, sycl_small, cpu_small) && ok;
    ok = compare_outputs("large-batch", type, sycl_large, cpu_large) && ok;

    ggml_backend_free(sycl_backend);
    return ok;
}

int main() {
    printf("SYCL Host-Weight K-Quant Test\n");
    printf("=========================================================\n");
    printf("This test exercises unified-cache layout resolution for host weights\n");
    printf("by running both MMVQ (batch=1) and MMQ (batch>8) in the same graph.\n");

    ggml_cpu_init();

    const int n_embd = 512;
    const int n_rows = 1024;
    const int n_tokens_small = 1;
    const int n_tokens_large = 9;  // forces MMQ (MMVQ max batch is 8)

    int failed = 0;
    if (!test_kquant_type("Q4_K", GGML_TYPE_Q4_K, n_embd, n_rows, n_tokens_small, n_tokens_large)) failed++;
    if (!test_kquant_type("Q5_K", GGML_TYPE_Q5_K, n_embd, n_rows, n_tokens_small, n_tokens_large)) failed++;

    printf("\n=========================================================\n");
    if (failed == 0) {
        printf("All tests PASSED\n");
        return 0;
    }
    printf("%d tests FAILED\n", failed);
    return 1;
}
