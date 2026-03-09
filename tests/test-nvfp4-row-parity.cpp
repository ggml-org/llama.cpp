#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

static constexpr int32_t GGML_CUDA_STREAM_K_CONTROL_TAG = 0x534b4d51; // "SKMQ"

struct diff_stats {
    float  max_abs = 0.0f;
    double rms = 0.0;
    size_t idx_max = 0;
};

static std::vector<float> make_pattern(int64_t k) {
    static const float pat[16] = {
         0.25f, -0.25f,  0.50f, -0.50f,
         1.00f, -1.00f,  1.50f, -1.50f,
         2.00f, -2.00f,  3.00f, -3.00f,
         4.00f, -4.00f,  5.00f, -5.00f,
    };

    std::vector<float> out(k);
    for (int64_t i = 0; i < k; ++i) {
        const int64_t sb = i / 16;
        out[i] = pat[i % 16] * (0.08f * float(sb + 1));
    }
    return out;
}

static std::vector<uint8_t> quantize_2d(enum ggml_type qtype, const std::vector<float> & src, int64_t k, int64_t m) {
    const size_t row_size = ggml_row_size(qtype, k);
    std::vector<uint8_t> out(row_size * m);
    for (int64_t row = 0; row < m; ++row) {
        ggml_quantize_chunk(qtype, src.data() + row*k, out.data() + row*row_size, 0, 1, k, nullptr);
    }
    return out;
}

static std::vector<float> make_identity_b(int64_t k, float diag_value) {
    std::vector<float> b((size_t) k * (size_t) k, 0.0f);
    for (int64_t col = 0; col < k; ++col) {
        b[(size_t) col * (size_t) k + (size_t) col] = diag_value;
    }
    return b;
}

static std::vector<float> run_mul_mat_q(
        ggml_backend_t backend,
        enum ggml_type qtype,
        const std::vector<uint8_t> & a_q,
        const std::vector<float> & b_f32,
        int64_t m, int64_t n, int64_t k) {

    struct ggml_init_params params = {
        16*1024*1024,
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed\n");
        std::abort();
    }

    ggml_tensor * a = ggml_new_tensor_2d(ctx, qtype, k, m);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    out->op_params[0] = GGML_CUDA_STREAM_K_CONTROL_TAG;
    out->op_params[1] = 0;
    out->op_params[2] = 0;

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
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        num += (double) ref[i] * (double) test[i];
        den += (double) ref[i] * (double) ref[i];
    }
    return den > 0.0 ? num / den : 0.0;
}

static bool parse_qtype(const char * s, enum ggml_type & qtype) {
    if (std::strcmp(s, "nvfp4") == 0) { qtype = GGML_TYPE_NVFP4; return true; }
    if (std::strcmp(s, "q4_0")  == 0) { qtype = GGML_TYPE_Q4_0;  return true; }
    return false;
}

int main(int argc, char ** argv) {
    const char * qname = argc > 1 ? argv[1] : "nvfp4";
    enum ggml_type qtype = GGML_TYPE_NVFP4;
    if (!parse_qtype(qname, qtype)) {
        std::fprintf(stderr, "unknown qtype: %s\n", qname);
        return 2;
    }

    const int64_t k = 256;
    const int64_t m = 1;
    const int64_t n = 256;
    const float diag_value = argc > 2 ? std::strtof(argv[2], nullptr) : 1.0f;

    ggml_backend_load_all();

    ggml_backend_dev_t dev_gpu = nullptr;
    ggml_backend_dev_t dev_cpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t cur = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(cur) == GGML_BACKEND_DEVICE_TYPE_GPU && !dev_gpu) dev_gpu = cur;
        if (ggml_backend_dev_type(cur) == GGML_BACKEND_DEVICE_TYPE_CPU && !dev_cpu) dev_cpu = cur;
    }
    if (!dev_gpu || !dev_cpu) {
        std::fprintf(stderr, "missing backend(s)\n");
        return 3;
    }

    ggml_backend_t backend_gpu = ggml_backend_dev_init(dev_gpu, nullptr);
    ggml_backend_t backend_cpu = ggml_backend_dev_init(dev_cpu, nullptr);
    if (!backend_gpu || !backend_cpu) {
        std::fprintf(stderr, "backend init failed\n");
        return 4;
    }

    const std::vector<float> a_f32 = make_pattern(k);
    const std::vector<uint8_t> a_q = quantize_2d(qtype, a_f32, k, m);
    const std::vector<float> b_f32 = make_identity_b(k, diag_value);

    const std::vector<float> out_cpu = run_mul_mat_q(backend_cpu, qtype, a_q, b_f32, m, n, k);
    const std::vector<float> out_gpu = run_mul_mat_q(backend_gpu, qtype, a_q, b_f32, m, n, k);

    ggml_backend_free(backend_cpu);
    ggml_backend_free(backend_gpu);
    ggml_quantize_free();

    const diff_stats d = compare(out_cpu, out_gpu);
    const double alpha = fit_alpha(out_cpu, out_gpu);

    std::printf("qtype=%s k=%lld diag=%g\n", qname, (long long) k, diag_value);
    std::printf("global: max_abs=%.8g rms=%.8g idx=%zu alpha=%.8g\n", d.max_abs, d.rms, d.idx_max, alpha);

    for (int sb = 0; sb < 16; ++sb) {
        const int lo = sb * 16;
        const int hi = lo + 16;
        double num = 0.0, den = 0.0, sum_sq = 0.0;
        float max_abs = 0.0f;
        int idx_max = lo;
        for (int i = lo; i < hi; ++i) {
            const double dd = double(out_cpu[i]) - double(out_gpu[i]);
            const float ad = (float) std::fabs(dd);
            if (ad > max_abs) {
                max_abs = ad;
                idx_max = i;
            }
            sum_sq += dd * dd;
            num += (double) out_cpu[i] * (double) out_gpu[i];
            den += (double) out_cpu[i] * (double) out_cpu[i];
        }
        const double alpha_sb = den > 0.0 ? num / den : 0.0;
        const double rms = std::sqrt(sum_sq / 16.0);
        std::printf("sb=%2d max_abs=% .7g rms=% .7g alpha=% .7g idx=%d\n", sb, max_abs, rms, alpha_sb, idx_max);
    }

    for (int i = 0; i < 64; ++i) {
        const float c = out_cpu[i];
        const float g = out_gpu[i];
        const float diff = g - c;
        const float ratio = std::fabs(c) > 1e-12f ? g / c : 0.0f;
        std::printf("%3d cpu=% .7g gpu=% .7g diff=% .7g ratio=% .7g\n", i, c, g, diff, ratio);
    }

    return 0;
}
