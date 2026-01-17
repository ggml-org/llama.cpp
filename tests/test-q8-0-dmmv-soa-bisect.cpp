/*
 * Q8_0 SoA DMMV Bisect Test
 *
 * Runs a tiny DMMV case through the production path while switching
 * between SoA kernel variants (reorder/simple/direct).
 *
 * Build: cmake --build build --target test-q8-0-dmmv-soa-bisect
 * Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-q8-0-dmmv-soa-bisect
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cstring>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-cpu.h"
#include "ggml-sycl/ggml-sycl-test.hpp"

#define QK8_0 32

typedef struct {
    uint16_t d;
    int8_t qs[QK8_0];
} block_q8_0_test;

static inline uint16_t fp32_to_fp16(float f) {
    union {
        float f;
        uint32_t u;
    } fu = {f};

    uint32_t sign = (fu.u >> 16) & 0x8000;
    int32_t exponent = ((fu.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = (fu.u >> 13) & 0x3FF;

    if (exponent <= 0) {
        return (uint16_t)sign;
    } else if (exponent >= 31) {
        return (uint16_t)(sign | 0x7C00);
    }

    return (uint16_t)(sign | (exponent << 10) | mantissa);
}

static void quantize_to_q8_0(const float* src, block_q8_0_test* dst, int n) {
    const int nblocks = n / QK8_0;
    for (int ib = 0; ib < nblocks; ib++) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            amax = std::max(amax, std::fabs(src[ib * QK8_0 + j]));
        }
        const float d = amax / 127.0f;
        const float id = (d != 0.0f) ? 127.0f / amax : 0.0f;

        dst[ib].d = fp32_to_fp16(d);
        for (int j = 0; j < QK8_0; j++) {
            dst[ib].qs[j] = (int8_t)std::round(src[ib * QK8_0 + j] * id);
        }
    }
}

static bool run_cpu_reference(ggml_backend_t cpu_backend,
                              const block_q8_0_test* weight_data, int nblocks,
                              const float* input_data,
                              float* output_data,
                              int ncols, int nrows, int batch) {
    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) return false;

    struct ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, ncols, nrows);
    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, batch);
    struct ggml_tensor* output = ggml_mul_mat(ctx, weight, input);

    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    const size_t total_size = ggml_nbytes(weight) + ggml_nbytes(input) + ggml_nbytes(output) + 4096;
    ggml_backend_buffer_t cpu_buffer = ggml_backend_buft_alloc_buffer(cpu_buft, total_size);
    if (!cpu_buffer) {
        ggml_free(ctx);
        return false;
    }

    uint8_t* base = (uint8_t*)ggml_backend_buffer_get_base(cpu_buffer);
    ggml_backend_tensor_alloc(cpu_buffer, weight, base);
    ggml_backend_tensor_alloc(cpu_buffer, input, base + ggml_nbytes(weight) + 256);
    ggml_backend_tensor_alloc(cpu_buffer, output, base + ggml_nbytes(weight) + ggml_nbytes(input) + 512);

    ggml_backend_tensor_set(weight, weight_data, 0, nblocks * sizeof(block_q8_0_test));
    ggml_backend_tensor_set(input, input_data, 0, ncols * batch * sizeof(float));

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    const enum ggml_status status = ggml_backend_graph_compute(cpu_backend, graph);

    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(output, output_data, 0, nrows * batch * sizeof(float));
    }

    ggml_backend_buffer_free(cpu_buffer);
    ggml_free(ctx);

    return status == GGML_STATUS_SUCCESS;
}

static bool run_gpu_compute(ggml_backend_t gpu_backend,
                            const block_q8_0_test* weight_data, int nblocks,
                            const float* input_data,
                            float* output_data,
                            int ncols, int nrows, int batch,
                            const char * layout_override) {
    ggml_layout_mode prev_layout = GGML_LAYOUT_AOS;
    const bool       had_prev    = ggml_sycl::test_get_layout_override(&prev_layout);
    struct OverrideGuard {
        bool            restore = false;
        ggml_layout_mode prev   = GGML_LAYOUT_AOS;
        ~OverrideGuard() {
            if (restore) {
                ggml_sycl::test_set_layout_override(prev);
            } else {
                ggml_sycl::test_clear_layout_override();
            }
        }
    } override_guard{ had_prev, prev_layout };

    if (layout_override) {
        if (strcmp(layout_override, "aos") == 0) {
            ggml_sycl::test_set_layout_override(GGML_LAYOUT_AOS);
        } else if (strcmp(layout_override, "soa") == 0) {
            ggml_sycl::test_set_layout_override(GGML_LAYOUT_SOA);
        } else if (strcmp(layout_override, "coalesced") == 0) {
            ggml_sycl::test_set_layout_override(GGML_LAYOUT_COALESCED);
        } else {
            fprintf(stderr, "FAIL: unknown layout override '%s'\n", layout_override);
            return false;
        }
    } else {
        ggml_sycl::test_clear_layout_override();
    }

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) return false;

    struct ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, ncols, nrows);
    struct ggml_tensor* input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, batch);
    struct ggml_tensor* output = ggml_mul_mat(ctx, weight, input);

    ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(gpu_backend);

    const size_t weight_size = ggml_nbytes(weight) * 2 + 4096;
    ggml_backend_buffer_t gpu_weight_buffer = ggml_backend_buft_alloc_buffer(gpu_buft, weight_size);
    if (!gpu_weight_buffer) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_set_usage(gpu_weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(gpu_weight_buffer, weight,
                               (void*)ggml_backend_buffer_get_base(gpu_weight_buffer));

    const size_t compute_size = ggml_nbytes(input) + ggml_nbytes(output) + 4096;
    ggml_backend_buffer_t gpu_compute_buffer = ggml_backend_buft_alloc_buffer(gpu_buft, compute_size);
    if (!gpu_compute_buffer) {
        ggml_backend_buffer_free(gpu_weight_buffer);
        ggml_free(ctx);
        return false;
    }

    uint8_t* compute_base = (uint8_t*)ggml_backend_buffer_get_base(gpu_compute_buffer);
    ggml_backend_tensor_alloc(gpu_compute_buffer, input, compute_base);
    const size_t input_alloc = ggml_backend_buft_get_alloc_size(gpu_buft, input);
    ggml_backend_tensor_alloc(gpu_compute_buffer, output, compute_base + input_alloc + 512);

    ggml_backend_tensor_set(weight, weight_data, 0, nblocks * sizeof(block_q8_0_test));
    ggml_backend_tensor_set(input, input_data, 0, ncols * batch * sizeof(float));

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    const enum ggml_status status = ggml_backend_graph_compute(gpu_backend, graph);

    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(output, output_data, 0, nrows * batch * sizeof(float));
    }

    ggml_backend_buffer_free(gpu_compute_buffer);
    ggml_backend_buffer_free(gpu_weight_buffer);
    ggml_free(ctx);

    return status == GGML_STATUS_SUCCESS;
}

struct Variant {
    const char* name;
    bool simple;
    bool direct;
};

static void set_variant_env(const Variant& v) {
    unsetenv("GGML_SYCL_DMMV_SIMPLE_SOA");
    unsetenv("GGML_SYCL_DMMV_DIRECT_SOA");
    if (v.simple) {
        setenv("GGML_SYCL_DMMV_SIMPLE_SOA", "1", 1);
    }
    if (v.direct) {
        setenv("GGML_SYCL_DMMV_DIRECT_SOA", "1", 1);
    }
}

int main() {
    printf("=== Q8_0 DMMV SoA Bisect (Production Path) ===\n");

    setenv("GGML_SYCL_FORCE_DMMV", "1", 1);

    const int ncols = 32;
    const int nrows = 4;
    const int batch = 1;
    const int nblocks = nrows * (ncols / QK8_0);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> weight_float(ncols * nrows);
    for (float &v : weight_float) {
        v = dist(rng);
    }

    std::vector<block_q8_0_test> weight_q8(nblocks);
    quantize_to_q8_0(weight_float.data(), weight_q8.data(), (int)weight_float.size());

    std::vector<float> input_data(ncols * batch);
    for (float &v : input_data) {
        v = dist(rng);
    }

    std::vector<float> cpu_results(nrows * batch);

    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        printf("FAIL: Could not initialize CPU backend\n");
        return 1;
    }

    if (!run_cpu_reference(cpu_backend, weight_q8.data(), nblocks,
                           input_data.data(), cpu_results.data(),
                           ncols, nrows, batch)) {
        printf("FAIL: CPU reference compute failed\n");
        ggml_backend_free(cpu_backend);
        return 1;
    }

    ggml_backend_t gpu_backend = ggml_backend_sycl_init(0);
    if (!gpu_backend) {
        printf("FAIL: Could not initialize SYCL backend\n");
        ggml_backend_free(cpu_backend);
        return 1;
    }

    int failures = 0;
    std::vector<float> gpu_results(nrows * batch);
    unsetenv("GGML_SYCL_DMMV_SIMPLE_SOA");
    unsetenv("GGML_SYCL_DMMV_DIRECT_SOA");
    if (!run_gpu_compute(gpu_backend, weight_q8.data(), nblocks,
                         input_data.data(), gpu_results.data(),
                         ncols, nrows, batch, "aos")) {
        printf("  [aos] FAIL: GPU compute failed\n");
        failures++;
    } else {
        float max_err = 0.0f;
        int bad = 0;
        for (size_t i = 0; i < cpu_results.size(); ++i) {
            const float err = std::fabs(gpu_results[i] - cpu_results[i]);
            max_err = std::max(max_err, err);
            const float rel = (std::fabs(cpu_results[i]) > 1e-6f) ? err / std::fabs(cpu_results[i]) : err;
            if (rel > 0.01f && err > 0.001f) {
                bad++;
            }
        }
        printf("  [aos] errors=%d/%zu max_err=%.6f %s\n",
               bad, cpu_results.size(), max_err,
               bad == 0 ? "PASS" : "FAIL");
        if (bad > 0) {
            printf("    Idx | CPU | GPU_AoS | diff\n");
            for (size_t i = 0; i < cpu_results.size(); ++i) {
                const float diff = gpu_results[i] - cpu_results[i];
                printf("    %3zu | % .6f | % .6f | % .6f\n", i, cpu_results[i], gpu_results[i], diff);
            }
            failures++;
        }
    }

    static const Variant variants[] = {
        { "reorder", false, false },
        { "simple",  true,  false },
        { "direct",  false, true  },
    };

    for (const auto & v : variants) {
        set_variant_env(v);
        std::fill(gpu_results.begin(), gpu_results.end(), 0.0f);
        const bool ok = run_gpu_compute(gpu_backend, weight_q8.data(), nblocks,
                                        input_data.data(), gpu_results.data(),
                                        ncols, nrows, batch, "soa");
        if (!ok) {
            printf("  [%s] FAIL: GPU compute failed\n", v.name);
            failures++;
            continue;
        }

        float max_err = 0.0f;
        int bad = 0;
        for (size_t i = 0; i < cpu_results.size(); ++i) {
            const float err = std::fabs(gpu_results[i] - cpu_results[i]);
            max_err = std::max(max_err, err);
            const float rel = (std::fabs(cpu_results[i]) > 1e-6f) ? err / std::fabs(cpu_results[i]) : err;
            if (rel > 0.01f && err > 0.001f) {
                bad++;
            }
        }

        printf("  [%s] errors=%d/%zu max_err=%.6f %s\n",
               v.name, bad, cpu_results.size(), max_err,
               bad == 0 ? "PASS" : "FAIL");
        if (bad > 0) {
            printf("    Idx | CPU | GPU_SoA | diff\n");
            for (size_t i = 0; i < cpu_results.size(); ++i) {
                const float diff = gpu_results[i] - cpu_results[i];
                printf("    %3zu | % .6f | % .6f | % .6f\n", i, cpu_results[i], gpu_results[i], diff);
            }
            failures++;
        }
    }

    ggml_backend_free(gpu_backend);
    ggml_backend_free(cpu_backend);

    printf("=== Summary ===\n");
    printf("Failed variants: %d\n", failures);
    return failures ? 1 : 0;
}
