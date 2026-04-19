#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

enum route_kind {
    ROUTE_XMEM,
    ROUTE_FA,
    ROUTE_NOFUSE,
};

struct attn_case {
    int dq = 128;
    int dv = 128;
    int nq = 512;
    int nkv = 512;
    int n_head = 8;
    int n_head_kv = 8;
    int n_batch = 1;
    bool use_mask = false;
    bool causal_mask = false;
    int warmup = 1;
    int iters = 5;
    bool skip_ref = false;
    route_kind route = ROUTE_XMEM;
};

static void usage(const char * argv0) {
    std::fprintf(stderr,
            "Usage: %s [--route xmem|fa|nofuse] [--dq N] [--dv N] [--nq N] [--nkv N] "
            "[--n-head N] [--n-head-kv N] [--n-batch N] [--mask 0|1] [--causal 0|1] "
            "[--warmup N] [--iters N] [--skip-ref 0|1]\n",
            argv0);
}

static bool parse_int_arg(const char * value, int * out) {
    char * end = nullptr;
    long v = std::strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        return false;
    }
    *out = (int) v;
    return true;
}

static bool parse_args(int argc, char ** argv, attn_case * cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_val = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(2);
            }
            return argv[++i];
        };

        if (arg == "--route") {
            const std::string v = need_val("--route");
            if (v == "xmem") cfg->route = ROUTE_XMEM;
            else if (v == "fa") cfg->route = ROUTE_FA;
            else if (v == "nofuse") cfg->route = ROUTE_NOFUSE;
            else return false;
        } else if (arg == "--dq") {
            if (!parse_int_arg(need_val("--dq"), &cfg->dq)) return false;
        } else if (arg == "--dv") {
            if (!parse_int_arg(need_val("--dv"), &cfg->dv)) return false;
        } else if (arg == "--nq") {
            if (!parse_int_arg(need_val("--nq"), &cfg->nq)) return false;
        } else if (arg == "--nkv") {
            if (!parse_int_arg(need_val("--nkv"), &cfg->nkv)) return false;
        } else if (arg == "--n-head") {
            if (!parse_int_arg(need_val("--n-head"), &cfg->n_head)) return false;
        } else if (arg == "--n-head-kv") {
            if (!parse_int_arg(need_val("--n-head-kv"), &cfg->n_head_kv)) return false;
        } else if (arg == "--n-batch") {
            if (!parse_int_arg(need_val("--n-batch"), &cfg->n_batch)) return false;
        } else if (arg == "--mask") {
            int v = 0;
            if (!parse_int_arg(need_val("--mask"), &v)) return false;
            cfg->use_mask = v != 0;
        } else if (arg == "--causal") {
            int v = 0;
            if (!parse_int_arg(need_val("--causal"), &v)) return false;
            cfg->causal_mask = v != 0;
        } else if (arg == "--warmup") {
            if (!parse_int_arg(need_val("--warmup"), &cfg->warmup)) return false;
        } else if (arg == "--iters") {
            if (!parse_int_arg(need_val("--iters"), &cfg->iters)) return false;
        } else if (arg == "--skip-ref") {
            int v = 0;
            if (!parse_int_arg(need_val("--skip-ref"), &v)) return false;
            cfg->skip_ref = v != 0;
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else {
            return false;
        }
    }

    return true;
}

static void fill_random_f32(std::vector<float> & data, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float & v : data) {
        v = dist(rng);
    }
}

static void fill_mask(std::vector<ggml_fp16_t> & data, const attn_case & cfg) {
    std::vector<float> tmp(data.size(), 0.0f);
    for (int b = 0; b < cfg.n_batch; ++b) {
        for (int q = 0; q < cfg.nq; ++q) {
            for (int k = 0; k < cfg.nkv; ++k) {
                const size_t idx = ((size_t) b * cfg.nq + q) * cfg.nkv + k;
                if (cfg.causal_mask && k > (cfg.nkv - cfg.nq + q)) {
                    tmp[idx] = -INFINITY;
                }
            }
        }
    }
    ggml_fp32_to_fp16_row(tmp.data(), data.data(), (int64_t) tmp.size());
}

static ggml_tensor * build_graph(ggml_context * ctx, const attn_case & cfg, ggml_tensor ** q_out, ggml_tensor ** k_out,
        ggml_tensor ** v_out, ggml_tensor ** mask_out) {
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cfg.dq, cfg.nq, cfg.n_head, cfg.n_batch);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cfg.dq, cfg.nkv, cfg.n_head_kv, cfg.n_batch);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cfg.dv, cfg.nkv, cfg.n_head_kv, cfg.n_batch);
    ggml_tensor * mask = nullptr;
    const bool need_explicit_mask = cfg.use_mask || (cfg.causal_mask && cfg.route == ROUTE_NOFUSE);
    if (need_explicit_mask) {
        mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cfg.nkv, cfg.nq, 1, cfg.n_batch);
    }

    ggml_tensor * out = nullptr;
    const float scale = 1.0f / std::sqrt((float) cfg.dq);

    if (cfg.route == ROUTE_NOFUSE) {
        ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);
        ggml_tensor * vv = ggml_cont(ctx, ggml_transpose(ctx, v));
        ggml_tensor * kqv = ggml_mul_mat(ctx, vv, kq);
        out = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    } else {
        out = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    }

    *q_out = q;
    *k_out = k;
    *v_out = v;
    *mask_out = mask;
    return out;
}

static void set_tensor_f32(ggml_tensor * t, const std::vector<float> & data) {
    ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
}

static void set_tensor_f16(ggml_tensor * t, const std::vector<float> & data_f32, std::vector<ggml_fp16_t> & tmp_f16) {
    tmp_f16.resize(data_f32.size());
    ggml_fp32_to_fp16_row(data_f32.data(), tmp_f16.data(), (int64_t) data_f32.size());
    ggml_backend_tensor_set(t, tmp_f16.data(), 0, tmp_f16.size() * sizeof(ggml_fp16_t));
}

static bool run_backend(
        ggml_backend_t backend,
        const attn_case & cfg,
        const std::vector<float> & q_data,
        const std::vector<float> & k_data,
        const std::vector<float> & v_data,
        const std::vector<ggml_fp16_t> & mask_data,
        std::vector<float> * out,
        double * avg_ms) {
    const size_t ctx_size = 16 * 1024 * 1024;
    ggml_init_params params = {};
    params.mem_size = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc = true;

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "ggml_init failed\n");
        return false;
    }

    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * mask = nullptr;
    ggml_tensor * result = build_graph(ctx, cfg, &q, &k, &v, &mask);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(gf, result);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        std::fprintf(stderr, "ggml_backend_alloc_ctx_tensors failed\n");
        ggml_free(ctx);
        return false;
    }

    std::vector<ggml_fp16_t> tmp_k_f16;
    std::vector<ggml_fp16_t> tmp_v_f16;
    set_tensor_f32(q, q_data);
    set_tensor_f16(k, k_data, tmp_k_f16);
    set_tensor_f16(v, v_data, tmp_v_f16);
    if (mask) {
        ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
    }

    for (int i = 0; i < cfg.warmup; ++i) {
        ggml_status status = ggml_backend_graph_compute(backend, gf);
        if (status != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "warmup compute failed: %s\n", ggml_status_to_string(status));
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return false;
        }
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < cfg.iters; ++i) {
        ggml_status status = ggml_backend_graph_compute(backend, gf);
        if (status != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "compute failed: %s\n", ggml_status_to_string(status));
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return false;
        }
    }
    auto t1 = std::chrono::steady_clock::now();

    out->resize(ggml_nelements(result));
    ggml_backend_tensor_get(result, out->data(), 0, out->size() * sizeof(float));
    *avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / std::max(cfg.iters, 1);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

static void compare_outputs(const std::vector<float> & ref, const std::vector<float> & got, double * mae, double * max_abs, double * cos_sim) {
    double sum_abs = 0.0;
    double max_err = 0.0;
    double dot = 0.0;
    double ref_norm = 0.0;
    double got_norm = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double a = ref[i];
        const double b = got[i];
        const double diff = std::abs(a - b);
        sum_abs += diff;
        max_err = std::max(max_err, diff);
        dot += a * b;
        ref_norm += a * a;
        got_norm += b * b;
    }
    *mae = sum_abs / std::max<size_t>(1, ref.size());
    *max_abs = max_err;
    *cos_sim = dot / std::sqrt(std::max(1e-30, ref_norm * got_norm));
}

static size_t count_nans(const std::vector<float> & data) {
    size_t n = 0;
    for (float v : data) {
        if (std::isnan(v)) {
            ++n;
        }
    }
    return n;
}

int main(int argc, char ** argv) {
    attn_case cfg;
    if (!parse_args(argc, argv, &cfg)) {
        usage(argv[0]);
        return 2;
    }

    if (cfg.route == ROUTE_XMEM) {
        setenv("GGML_OPENCL_ADRENO_XMEM_ATTN", "1", 1);
    } else {
        unsetenv("GGML_OPENCL_ADRENO_XMEM_ATTN");
    }

    ggml_backend_t backend_gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend_gpu || !backend_cpu) {
        std::fprintf(stderr, "backend init failed\n");
        return 1;
    }

    const size_t q_elems = (size_t) cfg.dq * cfg.nq * cfg.n_head * cfg.n_batch;
    const size_t k_elems = (size_t) cfg.dq * cfg.nkv * cfg.n_head_kv * cfg.n_batch;
    const size_t v_elems = (size_t) cfg.dv * cfg.nkv * cfg.n_head_kv * cfg.n_batch;
    const size_t m_elems = (size_t) cfg.nkv * cfg.nq * cfg.n_batch;

    std::vector<float> q_data(q_elems);
    std::vector<float> k_data(k_elems);
    std::vector<float> v_data(v_elems);
    std::vector<ggml_fp16_t> mask_data(m_elems);
    fill_random_f32(q_data, 1);
    fill_random_f32(k_data, 2);
    fill_random_f32(v_data, 3);
    if (cfg.use_mask || cfg.causal_mask) {
        fill_mask(mask_data, cfg);
    }

    std::vector<float> ref_out;
    std::vector<float> gpu_out;
    double cpu_ms = 0.0;
    double gpu_ms = 0.0;

    if (!run_backend(backend_gpu, cfg, q_data, k_data, v_data, mask_data, &gpu_out, &gpu_ms)) {
        std::fprintf(stderr, "GPU run failed\n");
        return 1;
    }

    double mae = NAN, max_abs = NAN, cos_sim = NAN;
    size_t cpu_nan = 0;
    if (!cfg.skip_ref) {
        if (!run_backend(backend_cpu, cfg, q_data, k_data, v_data, mask_data, &ref_out, &cpu_ms)) {
            std::fprintf(stderr, "CPU run failed\n");
            return 1;
        }
        compare_outputs(ref_out, gpu_out, &mae, &max_abs, &cos_sim);
        cpu_nan = count_nans(ref_out);
    }

    const double flops = 2.0 * (double) cfg.n_head * cfg.n_batch * cfg.nq * cfg.nkv * (cfg.dq + cfg.dv);
    const double tops = flops / (gpu_ms * 1.0e-3) / 1.0e12;
    const char * route_name = cfg.route == ROUTE_XMEM ? "xmem" : cfg.route == ROUTE_FA ? "fa" : "nofuse";
    const size_t gpu_nan = count_nans(gpu_out);

    std::printf("route=%s dq=%d dv=%d nq=%d nkv=%d n_head=%d n_head_kv=%d n_batch=%d mask=%d causal=%d\n",
            route_name, cfg.dq, cfg.dv, cfg.nq, cfg.nkv, cfg.n_head, cfg.n_head_kv, cfg.n_batch,
            (int) cfg.use_mask, (int) cfg.causal_mask);
    std::printf("cpu_ms=%.6f gpu_ms=%.6f tops=%.6f mae=%.9f max_abs=%.9f cos=%.9f cpu_nan=%zu gpu_nan=%zu skip_ref=%d\n",
            cpu_ms, gpu_ms, tops, mae, max_abs, cos_sim, cpu_nan, gpu_nan, (int) cfg.skip_ref);

    ggml_backend_free(backend_gpu);
    ggml_backend_free(backend_cpu);
    return 0;
}
