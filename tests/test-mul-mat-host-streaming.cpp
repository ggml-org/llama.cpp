// Mul-mat host weight streaming test: compare CPU vs SYCL with host-resident weights.
// Ensures unified cache DMA streaming path works for non-quantized MUL_MAT.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-cpu.h"
#include "ggml-sycl/unified-cache.hpp"

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

static bool run_mul_mat_backend(ggml_backend_t backend,
                                ggml_backend_buffer_type_t weight_buft,
                                const std::vector<ggml_fp16_t> & weight_data,
                                const std::vector<float> & input_data,
                                int in_dim, int out_dim, int n_tokens,
                                std::vector<float> & output) {
    const ggml_init_params params = {
        32 * 1024 * 1024,
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, in_dim, out_dim);
    ggml_set_name(weights, "dense_weights");
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, n_tokens);
    ggml_set_name(input, "dense_input");
    ggml_tensor * out = ggml_mul_mat(ctx, weights, input);
    ggml_set_name(out, "dense_out");

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
    ggml_backend_buffer_t out_buf   = alloc_tensor_buffer(dev_buft, out, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    if (!input_buf || !out_buf) {
        ggml_backend_buffer_free(weight_buf);
        if (input_buf) ggml_backend_buffer_free(input_buf);
        if (out_buf) ggml_backend_buffer_free(out_buf);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(weights, weight_data.data(), 0,
                            weight_data.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(input, input_data.data(), 0,
                            input_data.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(weight_buf);
        ggml_backend_buffer_free(input_buf);
        ggml_backend_buffer_free(out_buf);
        ggml_free(ctx);
        return false;
    }

    output.resize(static_cast<size_t>(out_dim) * n_tokens);
    ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(float));

    ggml_backend_buffer_free(weight_buf);
    ggml_backend_buffer_free(input_buf);
    ggml_backend_buffer_free(out_buf);
    ggml_free(ctx);
    return true;
}

int main() {
    setenv("GGML_SYCL_UNIFIED_CACHE", "1", 1);
    setenv("GGML_SYCL_DMA_SLICE_MB", "1", 1);
    setenv("GGML_SYCL_DMA_BUFFERS", "2", 1);

    ggml_sycl::set_unified_cache_budget(3 * 1024ULL * 1024ULL);

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

    const int in_dim   = 1024;
    const int out_dim  = 1024;
    const int n_tokens = 4;

    std::mt19937 rng(42);
    std::vector<float> weight_f32(static_cast<size_t>(in_dim) * out_dim);
    std::vector<float> input_f32(static_cast<size_t>(in_dim) * n_tokens);
    fill_random(weight_f32, 0.5f, rng);
    fill_random(input_f32, 0.5f, rng);

    std::vector<ggml_fp16_t> weight_f16(weight_f32.size());
    for (size_t i = 0; i < weight_f32.size(); ++i) {
        weight_f16[i] = ggml_fp32_to_fp16(weight_f32[i]);
    }

    std::vector<float> cpu_out;
    std::vector<float> sycl_out;

    const bool cpu_ok = run_mul_mat_backend(
        cpu_backend,
        ggml_backend_get_default_buffer_type(cpu_backend),
        weight_f16, input_f32,
        in_dim, out_dim, n_tokens,
        cpu_out);

    const bool sycl_ok = run_mul_mat_backend(
        sycl_backend,
        ggml_backend_sycl_host_buffer_type(),
        weight_f16, input_f32,
        in_dim, out_dim, n_tokens,
        sycl_out);

    if (!cpu_ok || !sycl_ok) {
        fprintf(stderr, "FAIL: mul_mat backend compute failed (cpu_ok=%d sycl_ok=%d)\n",
                cpu_ok ? 1 : 0, sycl_ok ? 1 : 0);
        ggml_backend_free(cpu_backend);
        ggml_backend_free(sycl_backend);
        return 1;
    }

    if (cpu_out.size() != sycl_out.size()) {
        fprintf(stderr, "FAIL: output size mismatch cpu=%zu sycl=%zu\n",
                cpu_out.size(), sycl_out.size());
        ggml_backend_free(cpu_backend);
        ggml_backend_free(sycl_backend);
        return 1;
    }

    const double err   = nmse(cpu_out.data(), sycl_out.data(), cpu_out.size());
    const float  max_d = max_diff(cpu_out.data(), sycl_out.data(), cpu_out.size());

    fprintf(stderr, "mul_mat host streaming: nmse=%.6e max_diff=%.6f\n", err, max_d);

    const double nmse_tol = 5e-5;
    const float  max_tol  = 1e-2f;
    if (!std::isfinite(err) || err > nmse_tol || !std::isfinite(max_d) || max_d > max_tol) {
        fprintf(stderr, "FAIL: mismatch beyond tolerance (nmse>%.1e or max_diff>%.2f)\n", nmse_tol, max_tol);
        ggml_backend_free(cpu_backend);
        ggml_backend_free(sycl_backend);
        return 1;
    }

    ggml_backend_free(cpu_backend);
    ggml_backend_free(sycl_backend);
    return 0;
}
#endif
