// MoE MUL_MAT_ID SYCL test: compare SYCL vs CPU for MXFP4 expert weights.
// Targets GPT-OSS style MoE path (quantized expert weights + ids routing).

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

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

static double nmse(const float * a, const float * b, size_t n) {
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double da = a[i];
        const double db = b[i];
        const double diff = da - db;
        mse_a_b += diff * diff;
        mse_a_0 += da * da;
    }
    return mse_a_b / (mse_a_0 > 0.0 ? mse_a_0 : 1.0);
}

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

static void fill_random(std::vector<float> & data, float scale, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (float & v : data) {
        v = dist(rng);
    }
}

static void quantize_weights_mxfp4(const std::vector<float> & src, int in_dim, int out_dim, int n_experts,
                                   std::vector<block_mxfp4> & dst) {
    const int blocks_per_row = in_dim / QK_MXFP4;
    dst.resize(static_cast<size_t>(n_experts) * out_dim * blocks_per_row);
    for (int e = 0; e < n_experts; ++e) {
        for (int r = 0; r < out_dim; ++r) {
            const size_t row_idx = static_cast<size_t>(e) * out_dim + r;
            const float * row = src.data() + row_idx * in_dim;
            block_mxfp4 * out = dst.data() + row_idx * blocks_per_row;
            quantize_row_mxfp4_ref(row, out, in_dim);
        }
    }
}

static bool run_mul_mat_id_backend(ggml_backend_t backend,
                                   ggml_backend_buffer_type_t weight_buft,
                                   const std::vector<block_mxfp4> & weight_data,
                                   const std::vector<float> & input_data,
                                   const std::vector<int32_t> & ids_data,
                                   int in_dim, int out_dim, int n_experts,
                                   int n_used, int n_tokens,
                                   std::vector<float> & output) {
    const ggml_init_params params = {
        64 * 1024 * 1024,
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, in_dim, out_dim, n_experts);
    ggml_set_name(weights, "moe_weights");
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, in_dim, n_used, n_tokens);
    ggml_set_name(input, "moe_input");
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_set_name(ids, "moe_ids");
    ggml_tensor * out = ggml_mul_mat_id(ctx, weights, input, ids);
    ggml_set_name(out, "moe_out");

    ggml_backend_buffer_type_t dev_buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_t weight_buf = alloc_tensor_buffer(weight_buft, weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (!weight_buf) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev && weight_buft == ggml_backend_sycl_host_buffer_type()) {
        ggml_backend_sycl_register_host_weight_tensor(dev, weights);
    }

    ggml_backend_buffer_t input_buf = alloc_tensor_buffer(dev_buft, input, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_t ids_buf   = alloc_tensor_buffer(dev_buft, ids, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_t out_buf   = alloc_tensor_buffer(dev_buft, out, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    if (!input_buf || !ids_buf || !out_buf) {
        ggml_backend_buffer_free(weight_buf);
        if (input_buf) ggml_backend_buffer_free(input_buf);
        if (ids_buf) ggml_backend_buffer_free(ids_buf);
        if (out_buf) ggml_backend_buffer_free(out_buf);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(weights, weight_data.data(), 0,
                            weight_data.size() * sizeof(block_mxfp4));
    ggml_backend_tensor_set(input, input_data.data(), 0,
                            input_data.size() * sizeof(float));
    ggml_backend_tensor_set(ids, ids_data.data(), 0,
                            ids_data.size() * sizeof(int32_t));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(weight_buf);
        ggml_backend_buffer_free(input_buf);
        ggml_backend_buffer_free(ids_buf);
        ggml_backend_buffer_free(out_buf);
        ggml_free(ctx);
        return false;
    }

    output.resize(static_cast<size_t>(out_dim) * n_used * n_tokens);
    ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(float));

    ggml_backend_buffer_free(weight_buf);
    ggml_backend_buffer_free(input_buf);
    ggml_backend_buffer_free(ids_buf);
    ggml_backend_buffer_free(out_buf);
    ggml_free(ctx);
    return true;
}

struct moe_case {
    const char * label;
    int in_dim;
    int out_dim;
    int n_experts;
    int n_used;
    int n_tokens;
    float weight_scale;
    float input_scale;
};

static bool run_case(const moe_case & tc, ggml_backend_t cpu_backend, ggml_backend_t sycl_backend, std::mt19937 & rng) {
    if (tc.in_dim % QK_MXFP4 != 0) {
        fprintf(stderr, "FAIL: %s in_dim must be divisible by QK_MXFP4\n", tc.label);
        return false;
    }

    std::vector<float> weight_f32(static_cast<size_t>(tc.n_experts) * tc.out_dim * tc.in_dim);
    std::vector<float> input_f32(static_cast<size_t>(tc.in_dim) * tc.n_used * tc.n_tokens);
    fill_random(weight_f32, tc.weight_scale, rng);
    fill_random(input_f32, tc.input_scale, rng);

    std::vector<block_mxfp4> weight_q;
    quantize_weights_mxfp4(weight_f32, tc.in_dim, tc.out_dim, tc.n_experts, weight_q);

    std::vector<int32_t> ids(static_cast<size_t>(tc.n_used) * tc.n_tokens);
    for (int t = 0; t < tc.n_tokens; ++t) {
        for (int i = 0; i < tc.n_used; ++i) {
            ids[t * tc.n_used + i] = (t + i) % tc.n_experts;
        }
    }

    std::vector<float> cpu_out;
    std::vector<float> sycl_out;

    const bool cpu_ok = run_mul_mat_id_backend(
        cpu_backend,
        ggml_backend_get_default_buffer_type(cpu_backend),
        weight_q, input_f32, ids,
        tc.in_dim, tc.out_dim, tc.n_experts, tc.n_used, tc.n_tokens,
        cpu_out);

    const bool sycl_ok = run_mul_mat_id_backend(
        sycl_backend,
        ggml_backend_sycl_host_buffer_type(),
        weight_q, input_f32, ids,
        tc.in_dim, tc.out_dim, tc.n_experts, tc.n_used, tc.n_tokens,
        sycl_out);

    if (!cpu_ok || !sycl_ok) {
        fprintf(stderr, "FAIL: %s backend compute failed (cpu_ok=%d sycl_ok=%d)\n",
                tc.label, cpu_ok ? 1 : 0, sycl_ok ? 1 : 0);
        return false;
    }

    if (cpu_out.size() != sycl_out.size()) {
        fprintf(stderr, "FAIL: %s output size mismatch cpu=%zu sycl=%zu\n",
                tc.label, cpu_out.size(), sycl_out.size());
        return false;
    }

    const double err = nmse(cpu_out.data(), sycl_out.data(), cpu_out.size());
    const float max_d = max_diff(cpu_out.data(), sycl_out.data(), cpu_out.size());

    fprintf(stderr, "%s: nmse=%.6e max_diff=%.6f\n", tc.label, err, max_d);

    const double nmse_tol = 5e-4;
    const float max_tol = 1.0f;
    if (!std::isfinite(err) || err > nmse_tol || !std::isfinite(max_d) || max_d > max_tol) {
        fprintf(stderr, "FAIL: %s mismatch beyond tolerance (nmse>%.1e or max_diff>%.2f)\n",
                tc.label, nmse_tol, max_tol);
        return false;
    }

    return true;
}

int main() {
    setenv("GGML_SYCL_XMX_MOE", "1", 1);
    setenv("GGML_SYCL_XMX_MOE_TILED", "1", 1);

    ggml_backend_t sycl_backend = ggml_backend_sycl_init(0);
    if (!sycl_backend) {
        fprintf(stderr, "SKIP: Could not initialize SYCL backend\n");
        return 0;
    }

    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu_backend) {
        fprintf(stderr, "FAIL: Could not initialize CPU backend\n");
        ggml_backend_free(sycl_backend);
        return 1;
    }

    const moe_case cases[] = {
        {"MoE MUL_MAT_ID MXFP4 (base)", 256, 256, 8, 4, 4, 0.5f, 0.5f},
        {"MoE MUL_MAT_ID MXFP4 (wide)", 512, 128, 16, 4, 8, 0.4f, 0.4f},
        {"MoE MUL_MAT_ID MXFP4 (full)", 256, 192, 8, 8, 2, 0.6f, 0.6f},
    };

    std::mt19937 rng(1234);
    bool ok = true;
    for (const auto & tc : cases) {
        if (!run_case(tc, cpu_backend, sycl_backend, rng)) {
            ok = false;
            break;
        }
    }

    ggml_backend_free(cpu_backend);
    ggml_backend_free(sycl_backend);

    if (!ok) {
        return 1;
    }

    fprintf(stderr, "PASS\n");
    return 0;
}
#endif
