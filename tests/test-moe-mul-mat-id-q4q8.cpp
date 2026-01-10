// MoE MUL_MAT_ID SYCL test: compare SYCL vs CPU for Q4_0/Q8_0 expert weights.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
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

template <typename BlockT>
using quantize_fn = void (*)(const float * GGML_RESTRICT, BlockT * GGML_RESTRICT, int64_t);

template <typename BlockT>
static void quantize_weights(const std::vector<float> & src, int in_dim, int out_dim, int n_experts, int qk,
                             quantize_fn<BlockT> qfn, std::vector<BlockT> & dst) {
    const int blocks_per_row = in_dim / qk;
    dst.resize(static_cast<size_t>(n_experts) * out_dim * blocks_per_row);
    for (int e = 0; e < n_experts; ++e) {
        for (int r = 0; r < out_dim; ++r) {
            const size_t row_idx = static_cast<size_t>(e) * out_dim + r;
            const float * row = src.data() + row_idx * in_dim;
            BlockT * out = dst.data() + row_idx * blocks_per_row;
            qfn(row, out, in_dim);
        }
    }
}

template <typename BlockT>
static bool run_mul_mat_id_backend(ggml_backend_t backend,
                                   ggml_backend_buffer_type_t weight_buft,
                                   ggml_type weight_type,
                                   const std::vector<BlockT> & weight_data,
                                   const std::vector<float> & input_data,
                                   const std::vector<int32_t> & ids_data,
                                   bool ids_view,
                                   const char * weight_name,
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

    ggml_tensor * weights = ggml_new_tensor_3d(ctx, weight_type, in_dim, out_dim, n_experts);
    ggml_set_name(weights, weight_name);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, in_dim, n_used, n_tokens);
    ggml_set_name(input, "moe_input");
    ggml_tensor * ids_base = ggml_new_tensor_2d(ctx, GGML_TYPE_I32,
                                                (ids_view && n_used < n_experts) ? n_experts : n_used,
                                                n_tokens);
    ggml_set_name(ids_base, "moe_ids_base");
    ggml_tensor * ids = ids_base;
    if (ids_view && n_used < n_experts) {
        ids = ggml_view_2d(ctx, ids_base, n_used, n_tokens, ids_base->nb[1], 0);
        ggml_set_name(ids, "moe_ids_view");
    }
    ggml_tensor * out = ggml_mul_mat_id(ctx, weights, input, ids);
    ggml_set_name(out, "moe_out");

    ggml_backend_sycl_register_weight_usage(weight_name, GGML_SYCL_TENSOR_USAGE_MOE_EXPERT_WEIGHT);

    ggml_backend_buffer_type_t dev_buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_t weight_buf = alloc_tensor_buffer(weight_buft, weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (!weight_buf) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_t input_buf = alloc_tensor_buffer(dev_buft, input, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_t ids_buf   = alloc_tensor_buffer(dev_buft, ids_base, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_buffer_t out_buf   = alloc_tensor_buffer(dev_buft, out, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    if (!input_buf || !ids_buf || !out_buf) {
        ggml_backend_buffer_free(weight_buf);
        if (input_buf) ggml_backend_buffer_free(input_buf);
        if (ids_buf) ggml_backend_buffer_free(ids_buf);
        if (out_buf) ggml_backend_buffer_free(out_buf);
        ggml_free(ctx);
        return false;
    }
    if (ids_view && n_used < n_experts) {
        if (ggml_backend_view_init(ids) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(weight_buf);
            ggml_backend_buffer_free(input_buf);
            ggml_backend_buffer_free(ids_buf);
            ggml_backend_buffer_free(out_buf);
            ggml_free(ctx);
            return false;
        }
    }

    ggml_backend_tensor_set(weights, weight_data.data(), 0,
                            weight_data.size() * sizeof(BlockT));
    ggml_backend_tensor_set(input, input_data.data(), 0,
                            input_data.size() * sizeof(float));
    if (ids_view && n_used < n_experts) {
        std::vector<int32_t> ids_full(static_cast<size_t>(n_experts) * n_tokens, 0);
        for (int t = 0; t < n_tokens; ++t) {
            std::memcpy(ids_full.data() + static_cast<size_t>(t) * n_experts,
                        ids_data.data() + static_cast<size_t>(t) * n_used,
                        static_cast<size_t>(n_used) * sizeof(int32_t));
        }
        ggml_backend_tensor_set(ids_base, ids_full.data(), 0,
                                ids_full.size() * sizeof(int32_t));
    } else {
        ggml_backend_tensor_set(ids_base, ids_data.data(), 0,
                                ids_data.size() * sizeof(int32_t));
    }

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
    const char * weight_name;
    int in_dim;
    int out_dim;
    int n_experts;
    int n_used;
    int n_tokens;
    bool ids_view;
    float weight_scale;
    float input_scale;
    double nmse_tol;
    float max_tol;
};

template <typename BlockT>
static bool run_case_quant(const moe_case & tc, ggml_type weight_type, int qk, quantize_fn<BlockT> qfn,
                           ggml_backend_t cpu_backend, ggml_backend_t sycl_backend, std::mt19937 & rng) {
    if (tc.in_dim % qk != 0) {
        fprintf(stderr, "FAIL: %s in_dim must be divisible by %d\n", tc.label, qk);
        return false;
    }

    std::vector<float> weight_f32(static_cast<size_t>(tc.n_experts) * tc.out_dim * tc.in_dim);
    std::vector<float> input_f32(static_cast<size_t>(tc.in_dim) * tc.n_used * tc.n_tokens);
    fill_random(weight_f32, tc.weight_scale, rng);
    fill_random(input_f32, tc.input_scale, rng);

    std::vector<BlockT> weight_q;
    quantize_weights(weight_f32, tc.in_dim, tc.out_dim, tc.n_experts, qk, qfn, weight_q);

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
        weight_type,
        weight_q, input_f32, ids, tc.ids_view,
        tc.weight_name,
        tc.in_dim, tc.out_dim, tc.n_experts, tc.n_used, tc.n_tokens,
        cpu_out);

    const bool sycl_ok = run_mul_mat_id_backend(
        sycl_backend,
        ggml_backend_get_default_buffer_type(sycl_backend),
        weight_type,
        weight_q, input_f32, ids, tc.ids_view,
        tc.weight_name,
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

    if (!std::isfinite(err) || err > tc.nmse_tol || !std::isfinite(max_d) || max_d > tc.max_tol) {
        fprintf(stderr, "FAIL: %s mismatch beyond tolerance (nmse>%.1e or max_diff>%.2f)\n",
                tc.label, tc.nmse_tol, tc.max_tol);
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

    const moe_case q4_cases[] = {
        {"MoE MUL_MAT_ID Q4_0 (base)", "ffn_gate_exps_q4_0_test", 256, 128, 8, 4, 4, true, 0.6f, 0.6f, 1e-3, 2.0f},
        {"MoE MUL_MAT_ID Q4_0 (wide)", "ffn_gate_exps_q4_0_test", 384, 96, 8, 4, 2, false, 0.5f, 0.5f, 1e-3, 2.0f},
    };
    const moe_case q8_cases[] = {
        {"MoE MUL_MAT_ID Q8_0 (base)", "ffn_gate_exps_q8_0_test", 256, 128, 8, 4, 4, true, 0.6f, 0.6f, 5e-4, 1.0f},
        {"MoE MUL_MAT_ID Q8_0 (wide)", "ffn_gate_exps_q8_0_test", 384, 96, 8, 4, 2, false, 0.5f, 0.5f, 5e-4, 1.0f},
    };

    std::mt19937 rng(1234);
    bool ok = true;
    for (const auto & tc : q4_cases) {
        if (!run_case_quant<block_q4_0>(tc, GGML_TYPE_Q4_0, QK4_0, quantize_row_q4_0_ref,
                                        cpu_backend, sycl_backend, rng)) {
            ok = false;
            break;
        }
    }
    if (ok) {
        for (const auto & tc : q8_cases) {
            if (!run_case_quant<block_q8_0>(tc, GGML_TYPE_Q8_0, QK8_0, quantize_row_q8_0_ref,
                                            cpu_backend, sycl_backend, rng)) {
                ok = false;
                break;
            }
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
