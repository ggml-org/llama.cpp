// Graph regression: ensure SYCL graph replay refreshes input tensors.
// Runs the same graph twice with different input data and checks that
// outputs change and match a CPU reference.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml-sycl/ggml-sycl-test.hpp"
#include "ggml-quants.h"

#if !defined(GGML_USE_SYCL)
int main() {
    std::fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

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

static bool max_abs_diff(const std::vector<float> & a, const std::vector<float> & b, float & out_diff) {
    if (a.size() != b.size()) {
        return false;
    }
    float max_d = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float da = a[i];
        const float db = b[i];
        if (!std::isfinite(da) || !std::isfinite(db)) {
            out_diff = std::numeric_limits<float>::infinity();
            return true;
        }
        const float d = std::fabs(da - db);
        if (d > max_d) {
            max_d = d;
        }
    }
    out_diff = max_d;
    return true;
}

static bool run_cpu_reference(const std::vector<block_q4_0> & weight_data, const std::vector<float> & input_data,
                              int k, int n, std::vector<float> & out) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        std::fprintf(stderr, "SKIP: CPU backend unavailable\n");
        return false;
    }

    const ggml_init_params params = {
        8 * 1024 * 1024,
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        return false;
    }

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, k, n);
    ggml_set_name(weight, "blk.0.attn_q.weight.cpu");
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 1);
    ggml_set_name(input, "graph_refresh_input_cpu");
    ggml_set_input(input);
    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(output, "graph_refresh_output_cpu");
    ggml_tensor * final_out = output;
    std::vector<ggml_tensor *> extra_ops;
    // Ensure the graph is large enough to trigger SYCL graph capture.
    for (int i = 0; i < 12; ++i) {
        final_out = ggml_scale(ctx, final_out, 1.0f);
        extra_ops.push_back(final_out);
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    if (!buft) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    std::vector<ggml_backend_buffer_t> buffers;
    auto alloc_or_fail = [&](ggml_tensor * t, ggml_backend_buffer_usage usage) {
        ggml_backend_buffer_t buf = alloc_tensor_buffer(buft, t, usage);
        if (!buf) {
            return false;
        }
        buffers.push_back(buf);
        return true;
    };

    if (!alloc_or_fail(weight, GGML_BACKEND_BUFFER_USAGE_WEIGHTS) ||
        !alloc_or_fail(input, GGML_BACKEND_BUFFER_USAGE_COMPUTE) ||
        !alloc_or_fail(output, GGML_BACKEND_BUFFER_USAGE_COMPUTE)) {
        for (ggml_backend_buffer_t buf : buffers) {
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    for (ggml_tensor * extra : extra_ops) {
        if (!alloc_or_fail(extra, GGML_BACKEND_BUFFER_USAGE_COMPUTE)) {
            for (ggml_backend_buffer_t buf : buffers) {
                ggml_backend_buffer_free(buf);
            }
            ggml_free(ctx);
            ggml_backend_free(backend);
            return false;
        }
    }

    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(block_q4_0));
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, final_out);
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        for (ggml_backend_buffer_t buf : buffers) {
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    out.resize(static_cast<size_t>(n));
    ggml_backend_tensor_get(final_out, out.data(), 0, out.size() * sizeof(float));

    for (ggml_backend_buffer_t buf : buffers) {
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

static bool run_sycl_graph_refresh_test() {
    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        std::printf("SKIP: SYCL backend unavailable\n");
        return true;
    }

    if (!ggml_sycl::test_backend_supports_graphs(backend)) {
        std::printf("SKIP: SYCL backend does not support graphs\n");
        ggml_backend_free(backend);
        return true;
    }
    if (ggml_sycl::test_backend_graphs_disabled(backend)) {
        std::printf("SKIP: SYCL graphs disabled by configuration\n");
        ggml_backend_free(backend);
        return true;
    }

    const int k = 128;
    const int n = 64;

    if (k % QK4_0 != 0) {
        std::printf("SKIP: k must be divisible by QK4_0 (k=%d QK4_0=%d)\n", k, QK4_0);
        ggml_backend_free(backend);
        return true;
    }

    const int blocks_per_row = k / QK4_0;
    std::vector<float> weight_f32(static_cast<size_t>(k) * n);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < k; ++col) {
            weight_f32[static_cast<size_t>(row) * k + col] = 0.001f * float((row + 1) + (col % 13));
        }
    }
    std::vector<block_q4_0> weight_q4(static_cast<size_t>(n) * blocks_per_row);
    for (int row = 0; row < n; ++row) {
        const float * row_ptr = weight_f32.data() + static_cast<size_t>(row) * k;
        block_q4_0 * out_ptr = weight_q4.data() + static_cast<size_t>(row) * blocks_per_row;
        quantize_row_q4_0_ref(row_ptr, out_ptr, k);
    }

    std::vector<float> input_a(static_cast<size_t>(k));
    std::vector<float> input_b(static_cast<size_t>(k));
    for (int i = 0; i < k; ++i) {
        input_a[static_cast<size_t>(i)] = 0.01f * float(i + 1);
        input_b[static_cast<size_t>(i)] = 0.02f * float(i + 3);
    }

    std::vector<float> cpu_out_a;
    std::vector<float> cpu_out_b;
    if (!run_cpu_reference(weight_q4, input_a, k, n, cpu_out_a) ||
        !run_cpu_reference(weight_q4, input_b, k, n, cpu_out_b)) {
        std::fprintf(stderr, "FAIL: CPU reference computation failed\n");
        ggml_backend_free(backend);
        return false;
    }

    const ggml_init_params params = {
        16 * 1024 * 1024,
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        return false;
    }

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, k, n);
    ggml_set_name(weight, "blk.0.attn_q.weight");
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 1);
    ggml_set_name(input, "graph_refresh_input");
    ggml_set_input(input);
    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
    ggml_set_name(output, "graph_refresh_output");
    ggml_tensor * final_out = output;
    std::vector<ggml_tensor *> extra_ops;
    for (int i = 0; i < 12; ++i) {
        final_out = ggml_scale(ctx, final_out, 1.0f);
        extra_ops.push_back(final_out);
    }

    ggml_backend_buffer_type_t host_buft = ggml_backend_sycl_host_buffer_type();
    ggml_backend_buffer_type_t dev_buft  = ggml_backend_get_default_buffer_type(backend);
    if (!host_buft || !dev_buft) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    std::vector<ggml_backend_buffer_t> buffers;
    auto alloc_or_fail = [&](ggml_backend_buffer_type_t buft, ggml_tensor * t, ggml_backend_buffer_usage usage) {
        ggml_backend_buffer_t buf = alloc_tensor_buffer(buft, t, usage);
        if (!buf) {
            return false;
        }
        buffers.push_back(buf);
        return true;
    };

    if (!alloc_or_fail(host_buft, weight, GGML_BACKEND_BUFFER_USAGE_WEIGHTS) ||
        !alloc_or_fail(dev_buft, input, GGML_BACKEND_BUFFER_USAGE_COMPUTE) ||
        !alloc_or_fail(dev_buft, output, GGML_BACKEND_BUFFER_USAGE_COMPUTE)) {
        for (ggml_backend_buffer_t buf : buffers) {
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    for (ggml_tensor * extra : extra_ops) {
        if (!alloc_or_fail(dev_buft, extra, GGML_BACKEND_BUFFER_USAGE_COMPUTE)) {
            for (ggml_backend_buffer_t buf : buffers) {
                ggml_backend_buffer_free(buf);
            }
            ggml_free(ctx);
            ggml_backend_free(backend);
            return false;
        }
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev) {
        ggml_backend_sycl_register_host_weight_tensor(dev, weight);
    }

    ggml_backend_tensor_set(weight, weight_q4.data(), 0, weight_q4.size() * sizeof(block_q4_0));
    ggml_backend_tensor_set(input, input_a.data(), 0, input_a.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, final_out);

    ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: SYCL graph compute (first run) failed\n");
        for (ggml_backend_buffer_t buf : buffers) {
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    const size_t pinned_after_first = ggml_sycl::test_graph_pinned_entry_count(backend);
    if (pinned_after_first == 0) {
        std::printf("SKIP: SYCL graph did not pin cache entries; cannot validate replay\n");
        for (ggml_backend_buffer_t buf : buffers) {
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
        ggml_backend_free(backend);
        return true;
    }

    std::vector<float> sycl_out_a(static_cast<size_t>(n));
    ggml_backend_tensor_get(final_out, sycl_out_a.data(), 0, sycl_out_a.size() * sizeof(float));

    // Update input data and replay the same graph.
    ggml_backend_tensor_set(input, input_b.data(), 0, input_b.size() * sizeof(float));
    status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: SYCL graph compute (replay) failed\n");
        for (ggml_backend_buffer_t buf : buffers) {
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    std::vector<float> sycl_out_b(static_cast<size_t>(n));
    ggml_backend_tensor_get(final_out, sycl_out_b.data(), 0, sycl_out_b.size() * sizeof(float));

    float refresh_delta = 0.0f;
    if (!max_abs_diff(sycl_out_a, sycl_out_b, refresh_delta)) {
        std::fprintf(stderr, "FAIL: output size mismatch between runs\n");
        for (ggml_backend_buffer_t buf : buffers) {
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }
    if (refresh_delta < 1e-4f) {
        std::fprintf(stderr, "FAIL: graph replay output did not change after input update (delta=%g)\n", refresh_delta);
        for (ggml_backend_buffer_t buf : buffers) {
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    float diff_a = 0.0f;
    float diff_b = 0.0f;
    if (!max_abs_diff(sycl_out_a, cpu_out_a, diff_a) || !max_abs_diff(sycl_out_b, cpu_out_b, diff_b)) {
        std::fprintf(stderr, "FAIL: SYCL/CPU output size mismatch\n");
        for (ggml_backend_buffer_t buf : buffers) {
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    const float tol = 2e-2f;
    if (diff_a > tol || diff_b > tol) {
        std::fprintf(stderr, "FAIL: SYCL/CPU mismatch (diff_a=%g diff_b=%g tol=%g)\n", diff_a, diff_b, tol);
        for (ggml_backend_buffer_t buf : buffers) {
            ggml_backend_buffer_free(buf);
        }
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    std::printf("PASS: SYCL graph replay refreshes inputs (delta=%g diff_a=%g diff_b=%g)\n",
                refresh_delta, diff_a, diff_b);

    for (ggml_backend_buffer_t buf : buffers) {
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

int main() {
    const bool ok = run_sycl_graph_refresh_test();
    return ok ? 0 : 1;
}

#endif  // GGML_USE_SYCL
