// Test for SYCL fused RMS_NORM + MUL kernel.
//
// Build: cmake --build build --target test-rms-norm-fused-sycl
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-rms-norm-fused-sycl

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"

static void fill_random(float * data, int64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int64_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }
}

static void rms_norm_mul_reference(const float * x,
                                   const float * gamma,
                                   float * out,
                                   int64_t ncols,
                                   int64_t nrows,
                                   float eps) {
    for (int64_t row = 0; row < nrows; ++row) {
        const float * x_row = x + row * ncols;
        const float * g_row = gamma + row * ncols;
        float *       o_row = out + row * ncols;

        double sum_sq = 0.0;
        for (int64_t col = 0; col < ncols; ++col) {
            const double v = x_row[col];
            sum_sq += v * v;
        }
        const float inv_rms = 1.0f / std::sqrt(static_cast<float>(sum_sq / ncols) + eps);
        for (int64_t col = 0; col < ncols; ++col) {
            o_row[col] = x_row[col] * inv_rms * g_row[col];
        }
    }
}

static void rms_norm_mul_add_reference(const float * x,
                                       const float * gamma,
                                       const float * add,
                                       float * out,
                                       int64_t ncols,
                                       int64_t nrows,
                                       float eps) {
    for (int64_t row = 0; row < nrows; ++row) {
        const float * x_row   = x + row * ncols;
        const float * g_row   = gamma + row * ncols;
        const float * add_row = add + row * ncols;
        float *       o_row   = out + row * ncols;

        double sum_sq = 0.0;
        for (int64_t col = 0; col < ncols; ++col) {
            const double v = x_row[col];
            sum_sq += v * v;
        }
        const float inv_rms = 1.0f / std::sqrt(static_cast<float>(sum_sq / ncols) + eps);
        for (int64_t col = 0; col < ncols; ++col) {
            o_row[col] = x_row[col] * inv_rms * g_row[col] + add_row[col];
        }
    }
}

static void add_rms_norm_reference(const float * x,
                                   const float * add,
                                   float * add_out,
                                   float * rms_out,
                                   int64_t ncols,
                                   int64_t nrows,
                                   float eps) {
    for (int64_t row = 0; row < nrows; ++row) {
        const float * x_row   = x + row * ncols;
        const float * add_row = add + row * ncols;
        float *       add_dst = add_out + row * ncols;
        float *       rms_dst = rms_out + row * ncols;

        double sum_sq = 0.0;
        for (int64_t col = 0; col < ncols; ++col) {
            const float v = x_row[col] + add_row[col];
            add_dst[col] = v;
            sum_sq += static_cast<double>(v) * v;
        }
        const float inv_rms = 1.0f / std::sqrt(static_cast<float>(sum_sq / ncols) + eps);
        for (int64_t col = 0; col < ncols; ++col) {
            rms_dst[col] = add_dst[col] * inv_rms;
        }
    }
}

static float max_diff(const float * a, const float * b, int64_t n) {
    float max_d = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        const float d = std::fabs(a[i] - b[i]);
        if (d > max_d) {
            max_d = d;
        }
    }
    return max_d;
}

static bool alloc_tensor_on_buft(ggml_backend_buffer_type_t buft,
                                 ggml_tensor *             tensor,
                                 ggml_backend_buffer_t *   out_buf) {
    if (!tensor || !out_buf) {
        return false;
    }

    size_t align = ggml_backend_buft_get_alignment(buft);
    size_t size  = ggml_nbytes(tensor) + align;
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, size);
    if (!buf) {
        printf("FAIL: alloc buffer for %s\n", tensor->name);
        return false;
    }

    size_t alloc_size = ggml_backend_buffer_get_alloc_size(buf, tensor);
    if (alloc_size > size) {
        ggml_backend_buffer_free(buf);
        buf = ggml_backend_buft_alloc_buffer(buft, alloc_size);
        if (!buf) {
            printf("FAIL: alloc resized buffer for %s\n", tensor->name);
            return false;
        }
    }

    void * base = ggml_backend_buffer_get_base(buf);
    if (ggml_backend_tensor_alloc(buf, tensor, base) != GGML_STATUS_SUCCESS) {
        printf("FAIL: tensor alloc for %s\n", tensor->name);
        ggml_backend_buffer_free(buf);
        return false;
    }

    *out_buf = buf;
    return true;
}

static int run_rms_norm_mul_case(ggml_backend_t backend, bool gamma_on_host, float * diff_out) {
    if (gamma_on_host) {
        setenv("GGML_SYCL_NO_PINNED", "1", 1);
    }

    const int64_t ncols = 4096;  // Uses SLM-cached path (<= 8192) and matches model-sized rows.
    const int64_t nrows = 2;
    const float   eps   = 1e-5f;

    struct ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("FAIL: ggml_init failed\n");
        return 1;
    }

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
    ggml_set_name(x, "rms_x");
    ggml_tensor * gamma = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
    ggml_set_name(gamma, "rms_gamma");

    ggml_tensor * rms = ggml_rms_norm(ctx, x, eps);
    ggml_set_name(rms, "rms_out");
    ggml_tensor * out = ggml_mul(ctx, rms, gamma);
    ggml_set_name(out, "rms_mul_out");

    std::vector<ggml_backend_buffer_t> buffers;
    if (!gamma_on_host) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            printf("FAIL: ggml_backend_alloc_ctx_tensors failed\n");
            ggml_free(ctx);
            return 1;
        }
        buffers.push_back(buf);
    } else {
        ggml_backend_buffer_type_t dev_buft  = ggml_backend_get_default_buffer_type(backend);
        ggml_backend_buffer_type_t host_buft = ggml_backend_sycl_host_buffer_type();

        ggml_backend_buffer_t x_buf    = nullptr;
        ggml_backend_buffer_t gamma_buf = nullptr;
        ggml_backend_buffer_t rms_buf  = nullptr;
        ggml_backend_buffer_t out_buf  = nullptr;

        if (!alloc_tensor_on_buft(dev_buft, x, &x_buf) ||
            !alloc_tensor_on_buft(host_buft, gamma, &gamma_buf) ||
            !alloc_tensor_on_buft(dev_buft, rms, &rms_buf) ||
            !alloc_tensor_on_buft(dev_buft, out, &out_buf)) {
            if (x_buf) {
                ggml_backend_buffer_free(x_buf);
            }
            if (gamma_buf) {
                ggml_backend_buffer_free(gamma_buf);
            }
            if (rms_buf) {
                ggml_backend_buffer_free(rms_buf);
            }
            if (out_buf) {
                ggml_backend_buffer_free(out_buf);
            }
            ggml_free(ctx);
            return 1;
        }

        ggml_backend_buffer_set_usage(gamma_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        buffers.push_back(x_buf);
        buffers.push_back(gamma_buf);
        buffers.push_back(rms_buf);
        buffers.push_back(out_buf);
    }

    const int64_t n_elems = ncols * nrows;
    std::vector<float> x_data(n_elems);
    std::vector<float> gamma_data(n_elems);
    fill_random(x_data.data(), n_elems, 123);
    fill_random(gamma_data.data(), n_elems, 321);

    ggml_backend_tensor_set(x, x_data.data(), 0, n_elems * sizeof(float));
    ggml_backend_tensor_set(gamma, gamma_data.data(), 0, n_elems * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        printf("FAIL: ggml_backend_graph_compute returned %d\n", status);
        for (ggml_backend_buffer_t buf : buffers) {
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
        return 1;
    }

    std::vector<float> out_data(n_elems);
    ggml_backend_tensor_get(out, out_data.data(), 0, n_elems * sizeof(float));

    std::vector<float> ref_data(n_elems);
    rms_norm_mul_reference(x_data.data(), gamma_data.data(), ref_data.data(), ncols, nrows, eps);

    const float diff = max_diff(out_data.data(), ref_data.data(), n_elems);
    const float tol = 1e-3f;
    printf("max_diff=%.6f (tol=%.6f)\n", diff, tol);

    if (diff_out != nullptr) {
        *diff_out = diff;
    }

    for (ggml_backend_buffer_t buf : buffers) {
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);

    if (diff > tol || std::isnan(diff)) {
        printf("FAIL: output mismatch\n");
        return 1;
    }

    return 0;
}

static int run_rms_norm_mul_add_case(ggml_backend_t backend, bool weights_on_host, float * diff_out) {
    if (weights_on_host) {
        setenv("GGML_SYCL_NO_PINNED", "1", 1);
    }

    const int64_t ncols = 4096;  // Uses SLM-cached path (<= 8192) and matches model-sized rows.
    const int64_t nrows = 2;
    const float   eps   = 1e-5f;

    struct ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("FAIL: ggml_init failed\n");
        return 1;
    }

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
    ggml_set_name(x, "rms_x");
    ggml_tensor * gamma = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
    ggml_set_name(gamma, "rms_gamma");
    ggml_tensor * add = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
    ggml_set_name(add, "rms_add");

    ggml_tensor * rms = ggml_rms_norm(ctx, x, eps);
    ggml_set_name(rms, "rms_out");
    ggml_tensor * mul = ggml_mul(ctx, rms, gamma);
    ggml_set_name(mul, "rms_mul_out");
    ggml_tensor * out = ggml_add(ctx, mul, add);
    ggml_set_name(out, "rms_mul_add_out");

    std::vector<ggml_backend_buffer_t> buffers;
    if (!weights_on_host) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            printf("FAIL: ggml_backend_alloc_ctx_tensors failed\n");
            ggml_free(ctx);
            return 1;
        }
        buffers.push_back(buf);
    } else {
        ggml_backend_buffer_type_t dev_buft  = ggml_backend_get_default_buffer_type(backend);
        ggml_backend_buffer_type_t host_buft = ggml_backend_sycl_host_buffer_type();

        ggml_backend_buffer_t x_buf     = nullptr;
        ggml_backend_buffer_t gamma_buf = nullptr;
        ggml_backend_buffer_t add_buf   = nullptr;
        ggml_backend_buffer_t rms_buf   = nullptr;
        ggml_backend_buffer_t mul_buf   = nullptr;
        ggml_backend_buffer_t out_buf   = nullptr;

        if (!alloc_tensor_on_buft(dev_buft, x, &x_buf) ||
            !alloc_tensor_on_buft(host_buft, gamma, &gamma_buf) ||
            !alloc_tensor_on_buft(host_buft, add, &add_buf) ||
            !alloc_tensor_on_buft(dev_buft, rms, &rms_buf) ||
            !alloc_tensor_on_buft(dev_buft, mul, &mul_buf) ||
            !alloc_tensor_on_buft(dev_buft, out, &out_buf)) {
            if (x_buf) {
                ggml_backend_buffer_free(x_buf);
            }
            if (gamma_buf) {
                ggml_backend_buffer_free(gamma_buf);
            }
            if (add_buf) {
                ggml_backend_buffer_free(add_buf);
            }
            if (rms_buf) {
                ggml_backend_buffer_free(rms_buf);
            }
            if (mul_buf) {
                ggml_backend_buffer_free(mul_buf);
            }
            if (out_buf) {
                ggml_backend_buffer_free(out_buf);
            }
            ggml_free(ctx);
            return 1;
        }

        ggml_backend_buffer_set_usage(gamma_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_set_usage(add_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        buffers.push_back(x_buf);
        buffers.push_back(gamma_buf);
        buffers.push_back(add_buf);
        buffers.push_back(rms_buf);
        buffers.push_back(mul_buf);
        buffers.push_back(out_buf);
    }

    const int64_t n_elems = ncols * nrows;
    std::vector<float> x_data(n_elems);
    std::vector<float> gamma_data(n_elems);
    std::vector<float> add_data(n_elems);
    fill_random(x_data.data(), n_elems, 123);
    fill_random(gamma_data.data(), n_elems, 321);
    fill_random(add_data.data(), n_elems, 777);

    ggml_backend_tensor_set(x, x_data.data(), 0, n_elems * sizeof(float));
    ggml_backend_tensor_set(gamma, gamma_data.data(), 0, n_elems * sizeof(float));
    ggml_backend_tensor_set(add, add_data.data(), 0, n_elems * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        printf("FAIL: ggml_backend_graph_compute returned %d\n", status);
        for (ggml_backend_buffer_t buf : buffers) {
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
        return 1;
    }

    std::vector<float> out_data(n_elems);
    ggml_backend_tensor_get(out, out_data.data(), 0, n_elems * sizeof(float));

    std::vector<float> ref_data(n_elems);
    rms_norm_mul_add_reference(x_data.data(), gamma_data.data(), add_data.data(),
                               ref_data.data(), ncols, nrows, eps);

    const float diff = max_diff(out_data.data(), ref_data.data(), n_elems);
    const float tol = 1e-3f;
    printf("max_diff=%.6f (tol=%.6f)\n", diff, tol);

    if (diff_out != nullptr) {
        *diff_out = diff;
    }

    for (ggml_backend_buffer_t buf : buffers) {
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);

    if (diff > tol || std::isnan(diff)) {
        printf("FAIL: output mismatch\n");
        return 1;
    }

    return 0;
}

static int run_add_rms_norm_case(ggml_backend_t backend, float * diff_out) {
    const int64_t ncols = 4096;  // Uses SLM-cached path (<= 8192) and matches model-sized rows.
    const int64_t nrows = 2;
    const float   eps   = 1e-5f;

    struct ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("FAIL: ggml_init failed\n");
        return 1;
    }

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
    ggml_set_name(x, "add_x");
    ggml_tensor * add = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
    ggml_set_name(add, "add_src");

    ggml_tensor * add_out = ggml_add(ctx, x, add);
    ggml_set_name(add_out, "add_out");
    ggml_tensor * rms_out = ggml_rms_norm(ctx, add_out, eps);
    ggml_set_name(rms_out, "add_rms_out");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        printf("FAIL: ggml_backend_alloc_ctx_tensors failed\n");
        ggml_free(ctx);
        return 1;
    }

    const int64_t n_elems = ncols * nrows;
    std::vector<float> x_data(n_elems);
    std::vector<float> add_data(n_elems);
    fill_random(x_data.data(), n_elems, 555);
    fill_random(add_data.data(), n_elems, 999);

    ggml_backend_tensor_set(x, x_data.data(), 0, n_elems * sizeof(float));
    ggml_backend_tensor_set(add, add_data.data(), 0, n_elems * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, rms_out);
    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        printf("FAIL: ggml_backend_graph_compute returned %d\n", status);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 1;
    }

    std::vector<float> add_out_data(n_elems);
    std::vector<float> rms_out_data(n_elems);
    ggml_backend_tensor_get(add_out, add_out_data.data(), 0, n_elems * sizeof(float));
    ggml_backend_tensor_get(rms_out, rms_out_data.data(), 0, n_elems * sizeof(float));

    std::vector<float> add_ref_data(n_elems);
    std::vector<float> rms_ref_data(n_elems);
    add_rms_norm_reference(x_data.data(), add_data.data(),
                           add_ref_data.data(), rms_ref_data.data(),
                           ncols, nrows, eps);

    const float diff_add = max_diff(add_out_data.data(), add_ref_data.data(), n_elems);
    const float diff_rms = max_diff(rms_out_data.data(), rms_ref_data.data(), n_elems);
    const float tol = 1e-3f;
    printf("max_diff_add=%.6f max_diff_rms=%.6f (tol=%.6f)\n", diff_add, diff_rms, tol);

    if (diff_out != nullptr) {
        *diff_out = diff_add > diff_rms ? diff_add : diff_rms;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    if (diff_add > tol || diff_rms > tol || std::isnan(diff_add) || std::isnan(diff_rms)) {
        printf("FAIL: output mismatch\n");
        return 1;
    }

    return 0;
}

int main() {
    printf("=== SYCL RMS_NORM+MUL Fused Kernel Test ===\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        printf("SKIP: SYCL backend not available\n");
        return 0;
    }

    // Force immediate error propagation for the fused kernel.
    setenv("GGML_SYCL_WAIT_AFTER_RMS_NORM_MUL", "1", 1);
    float diff = 0.0f;
    printf("--- case: device gamma ---\n");
    if (run_rms_norm_mul_case(backend, /*gamma_on_host=*/ false, &diff) != 0) {
        ggml_backend_free(backend);
        return 1;
    }

    printf("--- case: device gamma + add ---\n");
    if (run_rms_norm_mul_add_case(backend, /*weights_on_host=*/ false, &diff) != 0) {
        ggml_backend_free(backend);
        return 1;
    }

    printf("--- case: add + rms_norm ---\n");
    if (run_add_rms_norm_case(backend, &diff) != 0) {
        ggml_backend_free(backend);
        return 1;
    }

    if (getenv("GGML_SYCL_TEST_HOST_WEIGHTS") != nullptr) {
        printf("--- case: host gamma (GGML_SYCL_NO_PINNED=1) ---\n");
        if (run_rms_norm_mul_case(backend, /*gamma_on_host=*/ true, &diff) != 0) {
            ggml_backend_free(backend);
            return 1;
        }

        printf("--- case: host gamma + add (GGML_SYCL_NO_PINNED=1) ---\n");
        if (run_rms_norm_mul_add_case(backend, /*weights_on_host=*/ true, &diff) != 0) {
            ggml_backend_free(backend);
            return 1;
        }
    }

    ggml_backend_free(backend);
    printf("PASS\n");
    return 0;
}
