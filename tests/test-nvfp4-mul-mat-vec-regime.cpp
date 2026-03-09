#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#define GGML_COMMON_DECL_CPP
#include "../ggml/src/ggml-common.h"
#include "../ggml/src/ggml-nvfp4-helpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static constexpr int32_t GGML_CUDA_STREAM_K_CONTROL_TAG = 0x534b4d51; // "SKMQ"

struct diff_stats {
    float  max_abs = 0.0f;
    double rms = 0.0;
    size_t idx_max = 0;
};

static diff_stats compare(const std::vector<float> & a, const std::vector<float> & b) {
    diff_stats s;
    double sum_sq = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = double(a[i]) - double(b[i]);
        const float ad = (float) std::fabs(d);
        if (ad > s.max_abs) {
            s.max_abs = ad;
            s.idx_max = i;
        }
        sum_sq += d*d;
    }
    s.rms = std::sqrt(sum_sq / std::max<size_t>(size_t(1), a.size()));
    return s;
}

static double fit_alpha(const std::vector<float> & ref, const std::vector<float> & test) {
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        num += (double) ref[i] * (double) test[i];
        den += (double) ref[i] * (double) ref[i];
    }
    return den > 0.0 ? num / den : 0.0;
}

static void set_weight_row(
        std::vector<uint8_t> & out,
        int64_t k,
        int row,
        int block,
        const std::array<uint8_t, 16> & scales,
        const std::array<uint8_t, 256> & codes) {
    const size_t row_size = ggml_row_size(GGML_TYPE_NVFP4, k);
    block_nvfp4 * row_packs = reinterpret_cast<block_nvfp4 *>(out.data() + row * row_size);
    const int pack = block >> 2;
    const int lane = block & 3;
    std::memcpy(row_packs[pack].scales[lane], scales.data(), scales.size());
    ggml_nvfp4_pack_codes_256(codes.data(), row_packs[pack].qs[lane]);
}

static std::vector<float> run_mul_mat_q(
        ggml_backend_t backend,
        const std::vector<uint8_t> & a_q,
        const std::vector<float> & b_f32,
        int64_t k,
        int64_t m,
        int64_t n,
        float activation_scale,
        float tensor_scale) {
    struct ggml_init_params params = {
        16 * 1024 * 1024,
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed\n");
        std::abort();
    }

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_NVFP4, k, m);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    out->op_params[0] = GGML_CUDA_STREAM_K_CONTROL_TAG;
    out->op_params[1] = 0;
    out->op_params[2] = 0;

    const int op_param_idx_nvfp4_tensor_scale = GGML_MAX_OP_PARAMS / (int) sizeof(int32_t) - 2;
    const int op_param_idx_nvfp4_input_scale  = GGML_MAX_OP_PARAMS / (int) sizeof(int32_t) - 1;
    int32_t tensor_scale_bits = 0;
    const float runtime_input_scale = activation_scale > 0.0f ? 1.0f / activation_scale : 1.0f;
    int32_t runtime_input_scale_bits = 0;
    std::memcpy(&tensor_scale_bits, &tensor_scale, sizeof(tensor_scale_bits));
    std::memcpy(&runtime_input_scale_bits, &runtime_input_scale, sizeof(runtime_input_scale_bits));
    a->op_params[op_param_idx_nvfp4_tensor_scale] = tensor_scale_bits;
    a->op_params[op_param_idx_nvfp4_input_scale] = runtime_input_scale_bits;

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        std::fprintf(stderr, "ggml_backend_alloc_ctx_tensors failed\n");
        std::abort();
    }

    ggml_backend_tensor_set(a, a_q.data(), 0, a_q.size());
    ggml_backend_tensor_set(b, b_f32.data(), 0, b_f32.size() * sizeof(float));

    const ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "ggml_backend_graph_compute failed: %s\n", ggml_status_to_string(status));
        std::abort();
    }

    std::vector<float> out_f32(ggml_nelements(out));
    ggml_backend_tensor_get(out, out_f32.data(), 0, out_f32.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out_f32;
}

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t dev_gpu = nullptr;
    ggml_backend_dev_t dev_cpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t cur = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(cur) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            dev_gpu = cur;
        } else if (ggml_backend_dev_type(cur) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            dev_cpu = cur;
        }
    }
    if (!dev_gpu) {
        std::fprintf(stderr, "missing GPU backend\n");
        return 3;
    }
    if (!dev_cpu) {
        std::fprintf(stderr, "missing CPU backend\n");
        return 5;
    }

    ggml_backend_t backend_gpu = ggml_backend_dev_init(dev_gpu, nullptr);
    ggml_backend_t backend_cpu = ggml_backend_dev_init(dev_cpu, nullptr);
    if (!backend_gpu) {
        std::fprintf(stderr, "backend init failed\n");
        return 4;
    }
    if (!backend_cpu) {
        std::fprintf(stderr, "CPU backend init failed\n");
        ggml_backend_free(backend_gpu);
        return 6;
    }

    constexpr int64_t k = 1024;
    constexpr int64_t m = 32;
    constexpr int64_t n_full = 129;
    constexpr int64_t n_split = 128;
    constexpr int64_t n_one = 1;

    const float activation_scale = 0.3125f;
    const float tensor_scale = 1.75f;
    const int n_blocks = (int) (k / 256);

    std::vector<uint8_t> a_q((size_t) m * ggml_row_size(GGML_TYPE_NVFP4, k));
    std::memset(a_q.data(), 0, a_q.size());
    for (int row = 0; row < m; ++row) {
        for (int block = 0; block < n_blocks; ++block) {
            std::array<uint8_t, 16> scales = {};
            std::array<uint8_t, 256> codes = {};
            for (int sb = 0; sb < 16; ++sb) {
                const float scale = 0.0625f * float(1 + ((row * 7 + block * 3 + sb * 5) % 8));
                scales[sb] = ggml_fp8_ue4m3_from_fp32(scale);
            }
            for (int i = 0; i < 256; ++i) {
                codes[i] = (uint8_t) ((row * 3 + block * 5 + i * 5 + (i / 8)) & 0xF);
            }
            set_weight_row(a_q, k, row, block, scales, codes);
        }
    }

    std::vector<float> b_full(k * n_full);
    for (int64_t col = 0; col < n_full; ++col) {
        for (int64_t i = 0; i < k; ++i) {
            const float base = 0.25f * std::sin(0.13f * float(i) + 0.01f * float(col));
            const float ramp = 0.02f * float((i + col) % 17 - 8);
            b_full[col * k + i] = activation_scale * (base + ramp);
        }
    }

    std::vector<float> b_one(k);
    for (int64_t i = 0; i < k; ++i) {
        b_one[i] = b_full[(n_full - 1) * k + i];
    }

    const std::vector<float> out_full = run_mul_mat_q(backend_gpu, a_q, b_full, k, m, n_full, activation_scale, tensor_scale);
    const std::vector<float> out_split = run_mul_mat_q(backend_gpu, a_q, std::vector<float>(b_full.begin(), b_full.begin() + k * n_split), k, m, n_split, activation_scale, tensor_scale);
    const std::vector<float> out_one_all = run_mul_mat_q(backend_gpu, a_q, b_one, k, m, n_one, activation_scale, tensor_scale);
    const std::vector<float> out_full_cpu = run_mul_mat_q(backend_cpu, a_q, b_full, k, m, n_full, activation_scale, tensor_scale);
    const std::vector<float> out_one_cpu = run_mul_mat_q(backend_cpu, a_q, b_one, k, m, n_one, activation_scale, tensor_scale);
    std::vector<float> out_full_mid(m);
    std::vector<float> out_split_last(m);
    std::vector<float> out_full_last(m);
    std::vector<float> out_full_last_cpu(m);
    for (int64_t row = 0; row < m; ++row) {
        out_full_mid[row] = out_full[(n_split - 1) * m + row];
        out_split_last[row] = out_split[(n_split - 1) * m + row];
    }
    for (int64_t row = 0; row < m; ++row) {
        out_full_last[row] = out_full[(n_full - 1) * m + row];
        out_full_last_cpu[row] = out_full_cpu[(n_full - 1) * m + row];
    }

    const diff_stats d_prompt = compare(out_full_mid, out_split_last);
    const double alpha_prompt = fit_alpha(out_full_mid, out_split_last);
    const diff_stats d = compare(out_full_last, out_one_all);
    const double alpha = fit_alpha(out_full_last, out_one_all);
    const diff_stats d_cpu_full_vs_one = compare(out_full_last_cpu, out_one_cpu);
    
    std::printf("mul_mat NVFP4 prompt diff: max_abs=%.8g rms=%.8g idx=%zu fit_alpha=%.8g activation_scale=%.8g tensor_scale=%.8g\n",
        d_prompt.max_abs, d_prompt.rms, d_prompt.idx_max, alpha_prompt, activation_scale, tensor_scale);
    std::printf("mul_mat NVFP4 tail diff: max_abs=%.8g rms=%.8g idx=%zu fit_alpha=%.8g activation_scale=%.8g tensor_scale=%.8g\n",
        d.max_abs, d.rms, d.idx_max, alpha, activation_scale, tensor_scale);
    std::printf("mul_mat NVFP4 cpu-full-vs-cpu-one tail: max_abs=%.8g rms=%.8g idx=%zu\n",
        d_cpu_full_vs_one.max_abs, d_cpu_full_vs_one.rms, d_cpu_full_vs_one.idx_max);
    // Note: GPU is NVFPxNVFP4; CPU is NVFP4xQ8.  Q8 does not use activation_scaling to make apples/apples comparison.
    

    ggml_backend_free(backend_cpu);
    ggml_backend_free(backend_gpu);

    if (d_prompt.max_abs > 1e-3f || d_prompt.rms > 1e-4f || d.max_abs > 1e-3f || d.rms > 1e-4f) {
        return 1;
    }

    return 0;
}
