// XMX GEMM host weight streaming test: compare CPU vs SYCL for Q4_0 weights.
// Ensures unified cache DMA streaming path works for XMX GEMM with host-resident weights.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-cpu.h"
#include "ggml-sycl/ggml-sycl-test.hpp"
#include "ggml-sycl/unified-cache.hpp"
#include "ggml-sycl/mmq_xmx.hpp"

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

static bool quantize_q4_0_rows(const std::vector<float> & src, int64_t ncols, int64_t nrows,
                               std::vector<uint8_t> & dst) {
    ggml_quantize_init(GGML_TYPE_Q4_0);
    const size_t row_size = ggml_row_size(GGML_TYPE_Q4_0, ncols);
    dst.resize(row_size * static_cast<size_t>(nrows));
    for (int64_t row = 0; row < nrows; ++row) {
        const float * row_src = src.data() + row * ncols;
        uint8_t * row_dst = dst.data() + row * row_size;
        ggml_quantize_chunk(GGML_TYPE_Q4_0, row_src, row_dst, 0, 1, ncols, nullptr);
    }
    return true;
}

static bool run_mul_mat_backend(ggml_backend_t backend,
                                ggml_backend_buffer_type_t weight_buft,
                                const std::vector<uint8_t> & weight_q4,
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

    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, in_dim, out_dim);
    ggml_set_name(weights, "xmx_q4_weights");
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, n_tokens);
    ggml_set_name(input, "xmx_input");
    ggml_tensor * out = ggml_mul_mat(ctx, weights, input);
    ggml_set_name(out, "xmx_out");

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

    ggml_backend_tensor_set(weights, weight_q4.data(), 0, weight_q4.size());
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

static bool run_sycl_case(ggml_backend_t sycl_backend,
                          const std::vector<uint8_t> & weight_q4,
                          const std::vector<float> & input_f32,
                          const std::vector<float> & cpu_out,
                          int in_dim, int out_dim, int n_tokens,
                          const char * layout_override) {
    ggml_layout_mode layout = GGML_LAYOUT_AOS;
    if (strcmp(layout_override, "aos") == 0) {
        layout = GGML_LAYOUT_AOS;
    } else if (strcmp(layout_override, "soa") == 0) {
        layout = GGML_LAYOUT_SOA;
    } else if (strcmp(layout_override, "coalesced") == 0) {
        layout = GGML_LAYOUT_COALESCED;
    } else if (strcmp(layout_override, "xmx_tiled") == 0) {
        layout = GGML_LAYOUT_XMX_TILED;
    } else if (strcmp(layout_override, "xmx_gemm_tiled") == 0) {
        layout = GGML_LAYOUT_XMX_GEMM_TILED;
    } else {
        fprintf(stderr, "FAIL: unknown layout override '%s'\n", layout_override);
        return false;
    }
    ggml_sycl::test_layout_override_guard guard(layout);

    std::vector<float> sycl_out;
    const bool sycl_ok = run_mul_mat_backend(
        sycl_backend,
        ggml_backend_sycl_host_buffer_type(),
        weight_q4, input_f32,
        in_dim, out_dim, n_tokens,
        sycl_out);

    if (!sycl_ok) {
        fprintf(stderr, "FAIL: mul_mat backend compute failed for layout=%s\n", layout_override);
        return false;
    }

    if (cpu_out.size() != sycl_out.size()) {
        fprintf(stderr, "FAIL: output size mismatch layout=%s cpu=%zu sycl=%zu\n",
                layout_override, cpu_out.size(), sycl_out.size());
        return false;
    }

    const double err_nmse = nmse(cpu_out.data(), sycl_out.data(), cpu_out.size());
    const float err_max   = max_diff(cpu_out.data(), sycl_out.data(), cpu_out.size());

    fprintf(stderr, "XMX host streaming (%s) nmse=%g max_diff=%g\n", layout_override, err_nmse, err_max);
    if (err_nmse > 1e-4 || err_max > 5e-2f) {
        fprintf(stderr, "FAIL: output mismatch (%s) nmse=%g max_diff=%g\n", layout_override, err_nmse, err_max);
        return false;
    }

    return true;
}

int main() {
#if !defined(GGML_SYCL_XMX_GEMM) || !defined(GGML_SYCL_MMQ_XMX)
    fprintf(stderr, "XMX GEMM not enabled at build time; skipping test.\n");
    return 0;
#else
    setenv("GGML_SYCL_UNIFIED_CACHE", "1", 1);
    setenv("GGML_SYCL_DMA_SLICE_MB", "1", 1);
    setenv("GGML_SYCL_DMA_BUFFERS", "2", 1);
    setenv("GGML_SYCL_USE_XMX_GEMM", "1", 1);
    setenv("GGML_SYCL_XMX_THRESHOLD", "64", 1);

    ggml_sycl::set_unified_cache_budget(4 * 1024ULL * 1024ULL);

    ggml_backend_t sycl_backend = ggml_backend_sycl_init(0);
    if (!sycl_backend) {
        fprintf(stderr, "SKIP: Could not initialize SYCL backend\n");
        return 0;
    }

    int xmx_m = 0, xmx_n = 0, xmx_k = 0;
    ggml_sycl_xmx_get_tile_dims(&xmx_m, &xmx_n, &xmx_k);
    if (xmx_m == 0 || xmx_n == 0 || xmx_k == 0 || !ggml_sycl_xmx_supports_type(GGML_TYPE_Q4_0)) {
        fprintf(stderr, "SKIP: Device does not support XMX INT8/Q4_0\n");
        ggml_backend_free(sycl_backend);
        return 0;
    }

    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu_backend) {
        fprintf(stderr, "FAIL: Could not initialize CPU backend\n");
        ggml_backend_free(sycl_backend);
        return 1;
    }

    const int in_dim   = 256;  // Must be divisible by XMX_K (32)
    const int out_dim  = 128;
    const int n_tokens = 16;   // > MMVQ_MAX_BATCH_SIZE to force XMX over MMVQ

    std::mt19937 rng(123);
    std::vector<float> weight_f32(static_cast<size_t>(in_dim) * out_dim);
    std::vector<float> input_f32(static_cast<size_t>(in_dim) * n_tokens);
    fill_random(weight_f32, 0.5f, rng);
    fill_random(input_f32, 0.5f, rng);

    std::vector<uint8_t> weight_q4;
    if (!quantize_q4_0_rows(weight_f32, in_dim, out_dim, weight_q4)) {
        fprintf(stderr, "FAIL: quantize_q4_0_rows failed\n");
        ggml_backend_free(cpu_backend);
        ggml_backend_free(sycl_backend);
        return 1;
    }

    std::vector<float> cpu_out;
    const bool cpu_ok = run_mul_mat_backend(
        cpu_backend,
        ggml_backend_get_default_buffer_type(cpu_backend),
        weight_q4, input_f32,
        in_dim, out_dim, n_tokens,
        cpu_out);

    if (!cpu_ok) {
        fprintf(stderr, "FAIL: mul_mat backend compute failed on CPU\n");
        ggml_backend_free(cpu_backend);
        ggml_backend_free(sycl_backend);
        return 1;
    }

    const bool aos_ok = run_sycl_case(sycl_backend, weight_q4, input_f32, cpu_out,
                                      in_dim, out_dim, n_tokens, "aos");
    const bool tiled_ok = run_sycl_case(sycl_backend, weight_q4, input_f32, cpu_out,
                                        in_dim, out_dim, n_tokens, "xmx_gemm_tiled");

    if (!aos_ok || !tiled_ok) {
        ggml_backend_free(cpu_backend);
        ggml_backend_free(sycl_backend);
        return 1;
    }

    ggml_backend_free(cpu_backend);
    ggml_backend_free(sycl_backend);
    return 0;
#endif
}

#endif
