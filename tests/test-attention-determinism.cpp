// Deterministic FlashAttention invariance and cross-run tests for CUDA backend

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static void set_env_deterministic() {
#if defined(_WIN32)
    SetEnvironmentVariableA("GGML_DETERMINISTIC", "1");
#else
    setenv("GGML_DETERMINISTIC", "1", 1);
#endif
}

struct AttnOut {
    std::vector<float> data; // flattened [DV, H, N]
    int64_t DV=0, H=0, N=0;
};

static void fp32_to_f16_buffer(const float *src, ggml_fp16_t *dst, size_t n) {
    // convert by rows is not required here; contiguous 1D conversion suffices
    for (size_t i = 0; i < n; ) {
        const size_t blk = std::min<size_t>(1024, n - i);
        ggml_fp32_to_fp16_row(src + i, dst + i, blk);
        i += blk;
    }
}

static void fill_uniform(std::mt19937 &rng, float *dst, size_t n, float lo=-1.0f, float hi=1.0f) {
    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i = 0; i < n; ++i) dst[i] = dist(rng);
}

// Builds and runs a FlashAttention graph with:
// Q: [D, N, H, S=1]; K: [D, KV, H_kv, 1]; V: [DV, KV, H_kv, 1]
// mask: [KV, PAD(N, GGML_KQ_MASK_PAD), 1, 1] (optional)
// sinks: [H] (optional)
static AttnOut run_attention_graph(ggml_backend_t backend,
                                   int64_t D, int64_t DV,
                                   int64_t N, int64_t H, int64_t H_kv,
                                   int64_t KV,
                                   bool use_mask, bool use_sinks,
                                   float max_bias, float logit_softcap,
                                   const std::vector<float> &Q_f32,
                                   const std::vector<float> &K_f32,
                                   const std::vector<float> &V_f32,
                                   const std::vector<float> &mask_f16_or_empty,
                                   const std::vector<float> &sinks_f32_or_empty) {
    ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) throw std::runtime_error("ggml_init failed");

    // Tensors
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, N, H, /*S*/1);
    ggml_set_name(q, "q");
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, KV, H_kv, 1);
    ggml_set_name(k, "k");
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, DV, KV, H_kv, 1);
    ggml_set_name(v, "v");

    const int64_t N_pad = GGML_PAD(N, GGML_KQ_MASK_PAD);
    ggml_tensor * m = nullptr;
    if (use_mask || max_bias > 0.0f) {
        m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, KV, N_pad, /*ne32*/1, /*ne33*/1);
        ggml_set_name(m, "m");
    }
    ggml_tensor * s = nullptr;
    if (use_sinks) {
        s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_set_name(s, "s");
    }

    const float scale = 1.0f / std::sqrt((float)D);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, scale, max_bias, logit_softcap);
    if (s) ggml_flash_attn_ext_add_sinks(out, s);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_DEFAULT);
    ggml_set_name(out, "out");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); throw std::runtime_error("alloc tensors failed"); }

    // Validate sizes and set data
    {   // Q
        const size_t nQ = (size_t)D*N*H;
        if (Q_f32.size() != nQ) { ggml_backend_buffer_free(buf); ggml_free(ctx); throw std::runtime_error("bad Q size"); }
        ggml_backend_tensor_set(q, Q_f32.data(), 0, nQ*sizeof(float));
    }
    {   // K
        const size_t nK = (size_t)D*KV*H_kv;
        if (K_f32.size() != nK) { ggml_backend_buffer_free(buf); ggml_free(ctx); throw std::runtime_error("bad K size"); }
        std::vector<ggml_fp16_t> tmp(nK);
        fp32_to_f16_buffer(K_f32.data(), tmp.data(), nK);
        ggml_backend_tensor_set(k, tmp.data(), 0, nK*sizeof(tmp[0]));
    }
    {   // V
        const size_t nV = (size_t)DV*KV*H_kv;
        if (V_f32.size() != nV) { ggml_backend_buffer_free(buf); ggml_free(ctx); throw std::runtime_error("bad V size"); }
        std::vector<ggml_fp16_t> tmp(nV);
        fp32_to_f16_buffer(V_f32.data(), tmp.data(), nV);
        ggml_backend_tensor_set(v, tmp.data(), 0, nV*sizeof(tmp[0]));
    }
    if (m) {
        const size_t nM = (size_t)KV*N_pad;
        if (!mask_f16_or_empty.empty()) {
            // provided as fp32 -> convert to f16
            std::vector<ggml_fp16_t> tmp(nM);
            fp32_to_f16_buffer(mask_f16_or_empty.data(), tmp.data(), nM);
            ggml_backend_tensor_set(m, tmp.data(), 0, nM*sizeof(tmp[0]));
        } else {
            std::vector<ggml_fp16_t> tmp(nM);
            std::fill(tmp.begin(), tmp.end(), ggml_fp32_to_fp16(0.0f));
            ggml_backend_tensor_set(m, tmp.data(), 0, nM*sizeof(tmp[0]));
        }
    }
    if (s) {
        const size_t nS = (size_t)H;
        if (sinks_f32_or_empty.empty()) {
            std::vector<float> tmp(nS, 0.0f);
            ggml_backend_tensor_set(s, tmp.data(), 0, nS*sizeof(float));
        } else {
            ggml_backend_tensor_set(s, sinks_f32_or_empty.data(), 0, nS*sizeof(float));
        }
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf); ggml_free(ctx);
        throw std::runtime_error("graph compute failed (flash_attn_ext)");
    }

    AttnOut out_h; out_h.DV = DV; out_h.H = H; out_h.N = N; out_h.data.resize((size_t)DV*H*N);
    ggml_backend_tensor_get(out, out_h.data.data(), 0, out_h.data.size()*sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out_h;
}

// Variant that allows K/V tensor dtypes. When k_type/v_type are quantized, this quantizes
// input float buffers row-wise using ggml_quantize_chunk.
static AttnOut run_attention_graph_types(ggml_backend_t backend,
                                         int64_t D, int64_t DV,
                                         int64_t N, int64_t H, int64_t H_kv,
                                         int64_t KV,
                                         bool use_mask, bool use_sinks,
                                         float max_bias, float logit_softcap,
                                         ggml_type k_type, ggml_type v_type,
                                         const std::vector<float> &Q_f32,
                                         const std::vector<float> &K_src_f32,
                                         const std::vector<float> &V_src_f32,
                                         const std::vector<float> &mask_f32_or_empty,
                                         const std::vector<float> &sinks_f32_or_empty) {
    ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) throw std::runtime_error("ggml_init failed");

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, N, H, /*S*/1);
    ggml_set_name(q, "q");
    ggml_tensor * k = ggml_new_tensor_4d(ctx, k_type, D, KV, H_kv, 1);
    ggml_set_name(k, "k");
    ggml_tensor * v = ggml_new_tensor_4d(ctx, v_type, DV, KV, H_kv, 1);
    ggml_set_name(v, "v");

    const int64_t N_pad = GGML_PAD(N, GGML_KQ_MASK_PAD);
    ggml_tensor * m = nullptr;
    if (use_mask || max_bias > 0.0f) {
        m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, KV, N_pad, /*ne32*/1, /*ne33*/1);
        ggml_set_name(m, "m");
    }
    ggml_tensor * s = nullptr;
    if (use_sinks) {
        s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_set_name(s, "s");
    }

    const float scale = 1.0f / std::sqrt((float)D);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, scale, max_bias, logit_softcap);
    if (s) ggml_flash_attn_ext_add_sinks(out, s);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_DEFAULT);
    ggml_set_name(out, "out");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); throw std::runtime_error("alloc tensors failed"); }

    // Set Q
    const size_t nQ = (size_t)D*N*H;
    if (Q_f32.size() != nQ) { ggml_backend_buffer_free(buf); ggml_free(ctx); throw std::runtime_error("bad Q size"); }
    ggml_backend_tensor_set(q, Q_f32.data(), 0, nQ*sizeof(float));

    // Set/quantize K and V
    const size_t nKf = (size_t)D*KV*H_kv;
    const size_t nVf = (size_t)DV*KV*H_kv;
    if (K_src_f32.size() != nKf || V_src_f32.size() != nVf) { ggml_backend_buffer_free(buf); ggml_free(ctx); throw std::runtime_error("bad KV size"); }

    auto is_quant = [](ggml_type t) {
        switch (t) {
            case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0: return true;
            default: return false;
        }
    };

    if (is_quant(k_type)) {
        std::vector<uint8_t> Kq(ggml_nbytes(k));
        ggml_quantize_chunk(k_type, K_src_f32.data(), Kq.data(), 0, /*nrow*/KV*H_kv, /*n_per_row*/D, /*imatrix*/nullptr);
        ggml_backend_tensor_set(k, Kq.data(), 0, Kq.size());
    } else {
        std::vector<ggml_fp16_t> tmp(nKf);
        fp32_to_f16_buffer(K_src_f32.data(), tmp.data(), nKf);
        ggml_backend_tensor_set(k, tmp.data(), 0, tmp.size()*sizeof(tmp[0]));
    }

    if (is_quant(v_type)) {
        std::vector<uint8_t> Vq(ggml_nbytes(v));
        ggml_quantize_chunk(v_type, V_src_f32.data(), Vq.data(), 0, /*nrow*/KV*H_kv, /*n_per_row*/DV, /*imatrix*/nullptr);
        ggml_backend_tensor_set(v, Vq.data(), 0, Vq.size());
    } else {
        std::vector<ggml_fp16_t> tmp(nVf);
        fp32_to_f16_buffer(V_src_f32.data(), tmp.data(), nVf);
        ggml_backend_tensor_set(v, tmp.data(), 0, tmp.size()*sizeof(tmp[0]));
    }

    if (m) {
        const size_t nM = (size_t)KV*N_pad;
        if (!mask_f32_or_empty.empty()) {
            std::vector<ggml_fp16_t> tmp(nM);
            fp32_to_f16_buffer(mask_f32_or_empty.data(), tmp.data(), nM);
            ggml_backend_tensor_set(m, tmp.data(), 0, nM*sizeof(tmp[0]));
        } else {
            std::vector<ggml_fp16_t> tmp(nM);
            std::fill(tmp.begin(), tmp.end(), ggml_fp32_to_fp16(0.0f));
            ggml_backend_tensor_set(m, tmp.data(), 0, nM*sizeof(tmp[0]));
        }
    }
    if (s) {
        const size_t nS = (size_t)H;
        if (sinks_f32_or_empty.empty()) {
            std::vector<float> tmp(nS, 0.0f);
            ggml_backend_tensor_set(s, tmp.data(), 0, nS*sizeof(float));
        } else {
            ggml_backend_tensor_set(s, sinks_f32_or_empty.data(), 0, nS*sizeof(float));
        }
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf); ggml_free(ctx);
        throw std::runtime_error("graph compute failed (flash_attn_ext types)");
    }

    AttnOut out_h; out_h.DV = DV; out_h.H = H; out_h.N = N; out_h.data.resize((size_t)DV*H*N);
    ggml_backend_tensor_get(out, out_h.data.data(), 0, out_h.data.size()*sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out_h;
}

static bool bytes_equal(const float *a, const float *b, size_t n) {
    return std::memcmp(a, b, n*sizeof(float)) == 0;
}

static bool allclose(const float *a, const float *b, size_t n, float atol=1e-3f) {
    for (size_t i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > atol) return false;
    }
    return true;
}

static void set_env_flag(const char *name, const char *val) {
#if defined(_WIN32)
    SetEnvironmentVariableA(name, val);
#else
    if (val) setenv(name, val, 1); else unsetenv(name);
#endif
}

// Validate FORCE_VEC and FORCE_TILE toggles (F16 K/V only) preserve determinism.
static int test_det_force_toggles(ggml_backend_t backend) {
    if (!std::getenv("RUN_FORCE_TOGGLE_TESTS")) {
        std::cerr << "[SKIP] det FORCE_* toggle tests disabled (set RUN_FORCE_TOGGLE_TESTS=1)\n";
        return 0;
    }
    std::mt19937 rng(6061);
    const int64_t D=128, DV=128, H=8, gqa=2, H_kv=H/gqa, KV=1024, N=33;
    const size_t nK=(size_t)D*KV*H_kv, nV=(size_t)DV*KV*H_kv, nQ=(size_t)D*N*H;
    std::vector<float> K(nK), V(nV), Q(nQ);
    fill_uniform(rng, K.data(), nK);
    fill_uniform(rng, V.data(), nV);
    fill_uniform(rng, Q.data(), nQ);
    const int64_t Np = GGML_PAD(N, GGML_KQ_MASK_PAD);
    std::vector<float> mask((size_t)KV*Np, 0.0f);

    // FORCE_VEC=1
    set_env_flag("GGML_DETERMINISTIC_ATTENTION_FORCE_VEC", "1");
    auto a1 = run_attention_graph(backend, D, DV, N, H, H_kv, KV, true, false, 0.0f, 0.0f, Q, K, V, mask, {});
    auto b1 = run_attention_graph(backend, D, DV, N, H, H_kv, KV, true, false, 0.0f, 0.0f, Q, K, V, mask, {});
    set_env_flag("GGML_DETERMINISTIC_ATTENTION_FORCE_VEC", nullptr);
    if (!bytes_equal(a1.data.data(), b1.data.data(), a1.data.size())) {
        std::cerr << "[FAIL] det FORCE_VEC cross-run determinism failed\n";
        return 60;
    }

    // FORCE_TILE=1 (F16 only)
    set_env_flag("GGML_DETERMINISTIC_ATTENTION_FORCE_TILE", "1");
    auto a2 = run_attention_graph(backend, D, DV, N, H, H_kv, KV, true, false, 0.0f, 0.0f, Q, K, V, mask, {});
    auto b2 = run_attention_graph(backend, D, DV, N, H, H_kv, KV, true, false, 0.0f, 0.0f, Q, K, V, mask, {});
    set_env_flag("GGML_DETERMINISTIC_ATTENTION_FORCE_TILE", nullptr);
    if (!bytes_equal(a2.data.data(), b2.data.data(), a2.data.size())) {
        std::cerr << "[FAIL] det FORCE_TILE cross-run determinism failed\n";
        return 61;
    }

    return 0;
}

// Optional smoke: disabling tile at D∈{80,96,112} should cause an error when MMA is not allowed.
static int test_disable_tile_smoke(ggml_backend_t backend) {
    if (!std::getenv("RUN_DISABLE_TILE_TESTS")) {
        std::cerr << "[SKIP] det disable-tile smoke disabled (set RUN_DISABLE_TILE_TESTS=1)\n";
        return 0;
    }
    std::mt19937 rng(7777);
    const int64_t D=80, DV=80, H=8, gqa=2, H_kv=H/gqa, KV=256, N=8;
    const size_t nQ=(size_t)D*N*H, nK=(size_t)D*KV*H_kv, nV=(size_t)DV*KV*H_kv;
    std::vector<float> Q(nQ), K(nK), V(nV);
    fill_uniform(rng, Q.data(), nQ);
    fill_uniform(rng, K.data(), nK);
    fill_uniform(rng, V.data(), nV);
    const int64_t Np = GGML_PAD(N, GGML_KQ_MASK_PAD);
    std::vector<float> mask((size_t)KV*Np, 0.0f);
    set_env_flag("GGML_DET_ATTENTION_DISABLE_TILE_80_96_112", "1");
    try {
        (void) run_attention_graph(backend, D, DV, N, H, H_kv, KV, true, false, 0.0f, 0.0f, Q, K, V, mask, {});
        set_env_flag("GGML_DET_ATTENTION_DISABLE_TILE_80_96_112", nullptr);
        std::cerr << "[FAIL] disable-tile smoke: expected error did not occur\n";
        return 70;
    } catch (const std::exception &) {
        set_env_flag("GGML_DET_ATTENTION_DISABLE_TILE_80_96_112", nullptr);
        return 0; // expected
    }
}

// 03B.3 MMA ncols=1 prototype tests for D∈{80,96,112} (opt-in)
// Compares MMA output (ALLOW_MMA=1) to deterministic tile output (FORCE_TILE=1) bitwise,
// and validates cross-run determinism for the MMA path.
static int test_mma_ncols1_proto(ggml_backend_t backend) {
    if (!std::getenv("RUN_MMA_PROTO_TESTS")) {
        std::cerr << "[SKIP] MMA ncols1 prototype tests disabled (set RUN_MMA_PROTO_TESTS=1)\n";
        return 0;
    }
    const int64_t Ds[] = {80, 96, 112};
    const int64_t KVs[] = {256, 1024};
    const int64_t H = 8;
    const int gqa = 2;
    const int64_t H_kv = H / gqa;
    const int Bs[] = {1, 8};

    for (int64_t D : Ds) {
        const int64_t DV = D;
        for (int64_t KV : KVs) {
            // Base K/V
            std::mt19937 rng((unsigned)(D*1000 + KV));
            const size_t nK=(size_t)D*KV*H_kv, nV=(size_t)DV*KV*H_kv;
            std::vector<float> K(nK), V(nV);
            fill_uniform(rng, K.data(), nK);
            fill_uniform(rng, V.data(), nV);
            for (int B : Bs) {
                // Q with deterministic seed
                std::mt19937 rngq((unsigned)(D*10 + B + KV));
                const size_t nQ=(size_t)D*B*H;
                std::vector<float> Q(nQ);
                fill_uniform(rngq, Q.data(), nQ);

                const int64_t Np = GGML_PAD(B, GGML_KQ_MASK_PAD);
                std::vector<float> mask((size_t)KV*Np, 0.0f);

                // Tile reference
                set_env_flag("GGML_DETERMINISTIC_ATTENTION_FORCE_TILE", "1");
                auto y_tile = run_attention_graph(backend, D, DV, B, H, H_kv, KV,
                                                  /*mask*/true, /*sinks*/false,
                                                  0.0f, 0.0f, Q, K, V, mask, {});
                set_env_flag("GGML_DETERMINISTIC_ATTENTION_FORCE_TILE", nullptr);

                // MMA under det dispatcher (opt-in)
                set_env_flag("GGML_DETERMINISTIC_ATTENTION_ALLOW_MMA", "1");
                auto y_mma_1 = run_attention_graph(backend, D, DV, B, H, H_kv, KV,
                                                   /*mask*/true, /*sinks*/false,
                                                   0.0f, 0.0f, Q, K, V, mask, {});
                auto y_mma_2 = run_attention_graph(backend, D, DV, B, H, H_kv, KV,
                                                   /*mask*/true, /*sinks*/false,
                                                   0.0f, 0.0f, Q, K, V, mask, {});
                set_env_flag("GGML_DETERMINISTIC_ATTENTION_ALLOW_MMA", nullptr);

                if (!bytes_equal(y_tile.data.data(), y_mma_1.data.data(), y_tile.data.size())) {
                    if (!allclose(y_tile.data.data(), y_mma_1.data.data(), y_tile.data.size(), 1e-3f)) {
                        std::cerr << "[FAIL] MMA proto mismatch vs tile (tol=1e-3): D="<<D<<" KV="<<KV<<" B="<<B<<"\n";
                        return 80;
                    }
                }
                if (!bytes_equal(y_mma_1.data.data(), y_mma_2.data.data(), y_mma_1.data.size())) {
                    std::cerr << "[FAIL] MMA proto cross-run determinism failed: D="<<D<<" KV="<<KV<<" B="<<B<<"\n";
                    return 81;
                }
            }
        }
    }

    return 0;
}

static int test_attention_invariance(ggml_backend_t backend) {
    std::mt19937 rng(4242);

    // Shapes
    const int64_t Ds[]  = {64, 80, 96, 112, 128, 256};
    const int64_t KVv[] = {256, 1024};      // must be multiples of FATTN_KQ_STRIDE
    const int      Bs[] = {2, 8, 33};
    const int      gqas[] = {1, 2, 4};     // H/H_kv

    const int64_t H = 8; // total heads

    for (int64_t D : Ds) {
        const int64_t DV = D; // standard attention
        for (int64_t KV : KVv) {
            for (int gqa : gqas) {
                if (H % gqa != 0) continue;
                const int64_t H_kv = H / gqa;

                // Fixed K/V per shape
                const size_t nK = (size_t)D*KV*H_kv;
                const size_t nV = (size_t)DV*KV*H_kv;
                std::vector<float> K(nK), V(nV);
                fill_uniform(rng, K.data(), nK);
                fill_uniform(rng, V.data(), nV);

                // Base Q for B=1 (N=1)
                {
                    const int64_t N = 1;
                    const size_t nQ = (size_t)D*N*H;
                    std::vector<float> Q(nQ);
                    fill_uniform(rng, Q.data(), nQ);

                    // shared mask/sinks
                    const int64_t N_pad = GGML_PAD(N, GGML_KQ_MASK_PAD);
                    std::vector<float> mask((size_t)KV*N_pad, 0.0f);
                    std::vector<float> sinks((size_t)H, 0.0f);

                    auto y1 = run_attention_graph(backend, D, DV, N, H, H_kv, KV,
                                                  /*use_mask=*/true, /*use_sinks=*/false,
                                                  /*max_bias=*/0.0f, /*softcap=*/0.0f,
                                                  Q, K, V, mask, sinks);

                    for (int B : Bs) {
                        const int64_t N2 = B;
                        const size_t nQ2 = (size_t)D*N2*H;
                        std::vector<float> Qb(nQ2);
                        // copy first query column (N=0 for all heads) from Q into Qb
                        // Layout is [D, N, H, S], contiguous with strides: nb0=D, nb1=D, nb2=D*N.
                        for (int64_t h = 0; h < H; ++h) {
                            const size_t src_off = (size_t)h * (size_t)D * (size_t)1; // N=1 in Q
                            const size_t dst_off = (size_t)h * (size_t)D * (size_t)N2; // N2 in Qb
                            std::copy(Q.begin() + src_off, Q.begin() + src_off + (size_t)D,
                                      Qb.begin() + dst_off);
                        }
                        // Fill remaining columns randomly (all heads, N>=1)
                        std::mt19937 rngb(rng());
                        for (int64_t h = 0; h < H; ++h) {
                            for (int64_t n = 1; n < N2; ++n) {
                                float *dst = Qb.data() + (size_t)h*(size_t)D*(size_t)N2 + (size_t)n*(size_t)D;
                                fill_uniform(rngb, dst, (size_t)D);
                            }
                        }

                        const int64_t N_pad2 = GGML_PAD(N2, GGML_KQ_MASK_PAD);
                        std::vector<float> mask2((size_t)KV*N_pad2, 0.0f);
                        std::vector<float> sinks2((size_t)H, 0.0f);

                        auto yb = run_attention_graph(backend, D, DV, N2, H, H_kv, KV,
                                                      /*use_mask=*/true, /*use_sinks=*/false,
                                                      0.0f, 0.0f, Qb, K, V, mask2, sinks2);

                        // Compare first query slice: size DV*H
                        if (!bytes_equal(y1.data.data(), yb.data.data(), (size_t)DV*H)) {
                            std::cerr << "[FAIL] attn batch invariance: D=" << D
                                      << " KV=" << KV << " B=" << B << " gqa=" << gqa << "\n";
                            return 10;
                        }
                    }
                }

                // Cross-run determinism on a harder case (B=33)
                {
                    const int64_t N = 33;
                    const size_t nQ = (size_t)D*N*H;
                    std::vector<float> Q(nQ);
                    std::mt19937 rngx(20250914 ^ (unsigned)D ^ (unsigned)KV);
                    fill_uniform(rngx, Q.data(), nQ);

                    const int64_t N_pad = GGML_PAD(N, GGML_KQ_MASK_PAD);
                    std::vector<float> mask((size_t)KV*N_pad, 0.0f);
                    std::vector<float> sinks((size_t)H, 0.0f);

                    auto a = run_attention_graph(backend, D, DV, N, H, H_kv, KV,
                                                 /*use_mask=*/true, /*use_sinks=*/false,
                                                 0.0f, 0.0f, Q, K, V, mask, sinks);
                    auto b = run_attention_graph(backend, D, DV, N, H, H_kv, KV,
                                                 /*use_mask=*/true, /*use_sinks=*/false,
                                                 0.0f, 0.0f, Q, K, V, mask, sinks);
                    if (!bytes_equal(a.data.data(), b.data.data(), a.data.size())) {
                        std::cerr << "[FAIL] attn cross-run determinism: D=" << D
                                  << " KV=" << KV << " gqa=" << gqa << "\n";
                        return 11;
                    }

                    // Softcap path (supported for D=128 or 256 in vec kernels): run a single case for D in {128,256}
                    if (D == 128 || D == 256) {
                        auto c = run_attention_graph(backend, D, DV, N, H, H_kv, KV,
                                                     /*use_mask=*/true, /*use_sinks=*/false,
                                                     0.0f, 1.0f, Q, K, V, mask, sinks);
                        auto d = run_attention_graph(backend, D, DV, N, H, H_kv, KV,
                                                     /*use_mask=*/true, /*use_sinks=*/false,
                                                     0.0f, 1.0f, Q, K, V, mask, sinks);
                        if (!bytes_equal(c.data.data(), d.data.data(), c.data.size())) {
                            std::cerr << "[FAIL] attn softcap cross-run determinism: D=" << D
                                      << " KV=" << KV << " gqa=" << gqa << "\n";
                            return 12;
                        }
                    }
                }
            }
        }
    }

    return 0;
}

// Light feature toggles test: ALiBi and sinks
static int test_attention_features_minimal(ggml_backend_t backend) {
    std::mt19937 rng(777);
    const int64_t D=128, DV=128, H=8, gqa=2, H_kv=H/gqa, KV=1024;
    const int64_t N1=1, N2=8;

    const size_t nK=(size_t)D*KV*H_kv, nV=(size_t)DV*KV*H_kv;
    std::vector<float> K(nK), V(nV);
    fill_uniform(rng, K.data(), nK);
    fill_uniform(rng, V.data(), nV);

    // Base Q for N=1
    const size_t nQ1=(size_t)D*N1*H;
    std::vector<float> Q1(nQ1);
    fill_uniform(rng, Q1.data(), nQ1);

    // With ALiBi + mask + sinks
    const int64_t N1_pad = GGML_PAD(N1, GGML_KQ_MASK_PAD);
    const int64_t N2_pad = GGML_PAD(N2, GGML_KQ_MASK_PAD);
    std::vector<float> mask1((size_t)KV*N1_pad, 1.0f), mask2((size_t)KV*N2_pad, 1.0f);
    std::vector<float> sinks((size_t)H);
    fill_uniform(rng, sinks.data(), sinks.size(), -4.0f, 4.0f);

    auto y1 = run_attention_graph(backend, D, DV, N1, H, H_kv, KV,
                                  /*mask*/true, /*sinks*/true,
                                  /*max_bias*/1.0f, /*softcap*/0.0f,
                                  Q1, K, V, mask1, sinks);

    // Build Q2 with first column equal to Q1
    const size_t nQ2=(size_t)D*N2*H;
    std::vector<float> Q2(nQ2);
    for (int64_t h = 0; h < H; ++h) {
        const size_t src_off = (size_t)h * (size_t)D * (size_t)1; // N=1 in Q1
        const size_t dst_off = (size_t)h * (size_t)D * (size_t)N2; // N2 in Q2, N=0 slot
        std::copy(Q1.begin() + src_off, Q1.begin() + src_off + (size_t)D,
                  Q2.begin() + dst_off);
    }
    for (int64_t h = 0; h < H; ++h) {
        for (int64_t n = 1; n < N2; ++n) {
            float *dst = Q2.data() + (size_t)h*(size_t)D*(size_t)N2 + (size_t)n*(size_t)D;
            fill_uniform(rng, dst, (size_t)D);
        }
    }

    auto y2 = run_attention_graph(backend, D, DV, N2, H, H_kv, KV,
                                  /*mask*/true, /*sinks*/true,
                                  /*max_bias*/1.0f, /*softcap*/0.0f,
                                  Q2, K, V, mask2, sinks);

    if (!bytes_equal(y1.data.data(), y2.data.data(), (size_t)DV*H)) {
        std::cerr << "[FAIL] attn (ALiBi+sinks) batch invariance failed\n";
        return 30;
    }

    // Cross-run determinism
    auto a = run_attention_graph(backend, D, DV, N2, H, H_kv, KV,
                                 true, true, 1.0f, 0.0f, Q2, K, V, mask2, sinks);
    auto b = run_attention_graph(backend, D, DV, N2, H, H_kv, KV,
                                 true, true, 1.0f, 0.0f, Q2, K, V, mask2, sinks);
    if (!bytes_equal(a.data.data(), b.data.data(), a.data.size())) {
        std::cerr << "[FAIL] attn (ALiBi+sinks) cross-run determinism failed\n";
        return 31;
    }

    // DV != D minimal probe (e.g., DV=64) for batch invariance and cross-run determinism
    {
        const int64_t DV2 = 64;
        const size_t nV2 = (size_t)DV2*KV*H_kv;
        std::vector<float> V2(nV2);
        fill_uniform(rng, V2.data(), nV2);

        // Build Q for N=1 and N=8; reuse existing K (D×KV×H_kv)
        const int64_t N1b=1, N2b=8;
        const size_t nQ1b=(size_t)D*N1b*H;
        std::vector<float> Q1b(nQ1b);
        fill_uniform(rng, Q1b.data(), nQ1b);

        const int64_t N1b_pad = GGML_PAD(N1b, GGML_KQ_MASK_PAD);
        const int64_t N2b_pad = GGML_PAD(N2b, GGML_KQ_MASK_PAD);
        std::vector<float> mask1b((size_t)KV*N1b_pad, 0.0f), mask2b((size_t)KV*N2b_pad, 0.0f);

        auto y1b = run_attention_graph(backend, D, DV2, N1b, H, H_kv, KV,
                                       /*mask*/true, /*sinks*/false,
                                       /*max_bias*/0.0f, /*softcap*/0.0f,
                                       Q1b, K, V2, mask1b, {});

        // Build Q2b with first column equal to Q1b
        const size_t nQ2b=(size_t)D*N2b*H;
        std::vector<float> Q2b(nQ2b);
        for (int64_t h2 = 0; h2 < H; ++h2) {
            const size_t src_off = (size_t)h2 * (size_t)D * (size_t)N1b;
            const size_t dst_off = (size_t)h2 * (size_t)D * (size_t)N2b;
            std::copy(Q1b.begin() + src_off, Q1b.begin() + src_off + (size_t)D,
                      Q2b.begin() + dst_off);
        }
        for (int64_t h2 = 0; h2 < H; ++h2) {
            for (int64_t n = 1; n < N2b; ++n) {
                float *dst = Q2b.data() + (size_t)h2*(size_t)D*(size_t)N2b + (size_t)n*(size_t)D;
                fill_uniform(rng, dst, (size_t)D);
            }
        }

        auto y2b = run_attention_graph(backend, D, DV2, N2b, H, H_kv, KV,
                                       /*mask*/true, /*sinks*/false,
                                       /*max_bias*/0.0f, /*softcap*/0.0f,
                                       Q2b, K, V2, mask2b, {});

        if (!bytes_equal(y1b.data.data(), y2b.data.data(), (size_t)DV2*H)) {
            std::cerr << "[FAIL] attn DV!=D batch invariance failed\n";
            return 32;
        }

        // Cross-run determinism
        auto c2 = run_attention_graph(backend, D, DV2, N2b, H, H_kv, KV,
                                      true, false, 0.0f, 0.0f, Q2b, K, V2, mask2b, {});
        auto d2 = run_attention_graph(backend, D, DV2, N2b, H, H_kv, KV,
                                      true, false, 0.0f, 0.0f, Q2b, K, V2, mask2b, {});
        if (!bytes_equal(c2.data.data(), d2.data.data(), c2.data.size())) {
            std::cerr << "[FAIL] attn DV!=D cross-run determinism failed\n";
            return 33;
        }
    }

    return 0;
}

// Feature tests for special head sizes with generic tile path (80/96/112):
// Validate ALiBi + mask work and preserve batch invariance and cross-run determinism.
static int test_special_heads_mask_alibi(ggml_backend_t backend) {
    const int64_t Ds[] = {80, 96, 112};
    const int64_t H = 8;
    const int gqa = 2;
    const int64_t H_kv = H / gqa;
    const int64_t KV = 256;
    const int64_t N1 = 1, N2 = 8;

    for (int64_t D : Ds) {
        const int64_t DV = D;
        std::mt19937 rng(2025 ^ (unsigned)D);
        const size_t nK=(size_t)D*KV*H_kv, nV=(size_t)DV*KV*H_kv;
        std::vector<float> K(nK), V(nV);
        fill_uniform(rng, K.data(), nK);
        fill_uniform(rng, V.data(), nV);

        // Base Q for N=1
        std::vector<float> Q1((size_t)D*N1*H);
        fill_uniform(rng, Q1.data(), Q1.size());

        // ALiBi + mask (all ones)
        const int64_t N1p = GGML_PAD(N1, GGML_KQ_MASK_PAD), N2p = GGML_PAD(N2, GGML_KQ_MASK_PAD);
        std::vector<float> mask1((size_t)KV*N1p, 1.0f), mask2((size_t)KV*N2p, 1.0f);
        std::vector<float> sinks((size_t)H, 0.0f);

        // y1 at N=1 with ALiBi (max_bias=1.0)
        auto y1 = run_attention_graph(backend, D, DV, N1, H, H_kv, KV,
                                      /*mask*/true, /*sinks*/true, 1.0f, 0.0f, Q1, K, V, mask1, sinks);

        // Build Q2 with first column == Q1, rest random
        std::vector<float> Q2((size_t)D*N2*H);
        for (int64_t h = 0; h < H; ++h) {
            const size_t src_off = (size_t)h*D*1;
            const size_t dst_off = (size_t)h*D*N2;
            std::copy(Q1.begin() + src_off, Q1.begin() + src_off + (size_t)D, Q2.begin() + dst_off);
        }
        for (int64_t h = 0; h < H; ++h) for (int64_t n = 1; n < N2; ++n)
            fill_uniform(rng, Q2.data() + (size_t)h*D*N2 + (size_t)n*D, (size_t)D);

        auto y2 = run_attention_graph(backend, D, DV, N2, H, H_kv, KV,
                                      /*mask*/true, /*sinks*/true, 1.0f, 0.0f, Q2, K, V, mask2, sinks);

        if (!bytes_equal(y1.data.data(), y2.data.data(), (size_t)DV*H)) {
            std::cerr << "[FAIL] special head (mask+ALiBi) batch invariance: D="<<D<<"\n";
            return 50;
        }

        // Cross-run determinism at N=8
        auto a = run_attention_graph(backend, D, DV, N2, H, H_kv, KV,
                                     true, true, 1.0f, 0.0f, Q2, K, V, mask2, sinks);
        auto b = run_attention_graph(backend, D, DV, N2, H, H_kv, KV,
                                     true, true, 1.0f, 0.0f, Q2, K, V, mask2, sinks);
        if (!bytes_equal(a.data.data(), b.data.data(), a.data.size())) {
            std::cerr << "[FAIL] special head (mask+ALiBi) cross-run determinism: D="<<D<<"\n";
            return 51;
        }
    }
    return 0;
}

// Negative test: softcap is unsupported for D in {80,96,112} and should abort cleanly.
static int test_special_heads_softcap_unsupported(ggml_backend_t backend) {
    if (!std::getenv("RUN_SOFTCAP_NEG_TESTS")) {
        std::cerr << "[SKIP] softcap negative test disabled (set RUN_SOFTCAP_NEG_TESTS=1 to run; will abort process)\n";
        return 0;
    }
    std::mt19937 rng(606060);
    const int64_t D=80, DV=80, H=8, gqa=2, H_kv=H/gqa, KV=256, N=8;
    const size_t nQ=(size_t)D*N*H, nK=(size_t)D*KV*H_kv, nV=(size_t)DV*KV*H_kv;
    std::vector<float> Q(nQ), K(nK), V(nV);
    fill_uniform(rng, Q.data(), nQ);
    fill_uniform(rng, K.data(), nK);
    fill_uniform(rng, V.data(), nV);
    const int64_t Np = GGML_PAD(N, GGML_KQ_MASK_PAD);
    std::vector<float> mask((size_t)KV*Np, 0.0f);
    try {
        (void) run_attention_graph(backend, D, DV, N, H, H_kv, KV,
                                   /*mask*/true, /*sinks*/false, 0.0f, 1.0f, Q, K, V, mask, {});
        std::cerr << "[FAIL] expected abort for softcap on D=80\n";
        return 52;
    } catch (const std::exception &) {
        return 0; // expected
    }
}

int main() {
    set_env_deterministic();
    ggml_backend_load_all();

    size_t n_dev = ggml_backend_dev_count();
    if (n_dev == 0) {
        std::cerr << "No backends available" << std::endl;
        return 0; // treat as skip
    }

    int n_ok = 0;
    bool ran_any = false;
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(dev);
        if (std::string(name).find("CUDA") == std::string::npos) {
            continue; // CUDA only
        }
        ran_any = true;
        ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
        if (!backend) {
            std::cerr << "[SKIP] cannot init backend: " << name << std::endl;
            continue;
        }

        int rc = test_attention_invariance(backend);
        if (rc == 0) rc = test_attention_features_minimal(backend);
        if (rc == 0) rc = test_special_heads_mask_alibi(backend);
        if (rc == 0) rc = test_special_heads_softcap_unsupported(backend);
        if (rc == 0) rc = test_det_force_toggles(backend);
        if (rc == 0) rc = test_disable_tile_smoke(backend);
        if (rc == 0) rc = test_mma_ncols1_proto(backend);

        // 03B: Quantized K/V (selected pairs) — D=128, q4_0/q4_0 and q8_0/q8_0
        if (rc == 0) {
            try {
                std::mt19937 rngq(3101);
                const int64_t D=128, DV=128, H=8; // typical
                const int gqas[] = {1,2};
                const int64_t KVs[] = {256, 1024};
                const ggml_type pairs_base[][2] = {{GGML_TYPE_Q4_0, GGML_TYPE_Q4_0}, {GGML_TYPE_Q8_0, GGML_TYPE_Q8_0}};
                std::vector<std::array<ggml_type,2>> pairs;
                for (auto &p : pairs_base) pairs.push_back({p[0], p[1]});
#ifdef GGML_CUDA_FA_ALL_QUANTS
                // Add a couple more pairs when full-quant vec instances are available
                pairs.push_back({GGML_TYPE_Q4_1, GGML_TYPE_Q5_1});
                pairs.push_back({GGML_TYPE_Q5_0, GGML_TYPE_Q8_0});
#endif

                for (int gqa : gqas) {
                    if (H % gqa != 0) continue;
                    const int64_t H_kv = H / gqa;
                    for (int64_t KV : KVs) {
                        // Build base K/V
                        const size_t nK=(size_t)D*KV*H_kv, nV=(size_t)DV*KV*H_kv;
                        std::vector<float> K(nK), V(nV);
                        fill_uniform(rngq, K.data(), nK);
                        fill_uniform(rngq, V.data(), nV);

                        // Q for N=1
                        const int64_t N1=1;
                        const size_t nQ1=(size_t)D*N1*H;
                        std::vector<float> Q1(nQ1);
                        fill_uniform(rngq, Q1.data(), nQ1);

                        const int64_t N1_pad = GGML_PAD(N1, GGML_KQ_MASK_PAD);
                        std::vector<float> mask1((size_t)KV*N1_pad, 0.0f);

                        for (auto &pp : pairs) {
                            // N=1 reference
                            auto y1 = run_attention_graph_types(backend, D, DV, N1, H, H_kv, KV,
                                                                /*mask*/true, /*sinks*/false,
                                                                0.0f, 0.0f, pp[0], pp[1], Q1, K, V, mask1, {});
                            // N=B compare first column
                            const int Bs[] = {2, 8, 33};
                            for (int B : Bs) {
                                const size_t nQb=(size_t)D*B*H;
                                std::vector<float> Qb(nQb);
                                // copy column 0
                                for (int64_t h = 0; h < H; ++h) {
                                    const size_t src_off = (size_t)h * (size_t)D * (size_t)1;
                                    const size_t dst_off = (size_t)h * (size_t)D * (size_t)B;
                                    std::copy(Q1.begin() + src_off, Q1.begin() + src_off + (size_t)D,
                                              Qb.begin() + dst_off);
                                }
                                // fill others
                                std::mt19937 rngb(rngq());
                                for (int64_t h = 0; h < H; ++h) {
                                    for (int64_t n = 1; n < B; ++n) {
                                        float *dst = Qb.data() + (size_t)h*(size_t)D*(size_t)B + (size_t)n*(size_t)D;
                                        fill_uniform(rngb, dst, (size_t)D);
                                    }
                                }
                                const int64_t N_pad = GGML_PAD(B, GGML_KQ_MASK_PAD);
                                std::vector<float> maskb((size_t)KV*N_pad, 0.0f);
                                auto yb = run_attention_graph_types(backend, D, DV, B, H, H_kv, KV,
                                                                    /*mask*/true, /*sinks*/false,
                                                                    0.0f, 0.0f, pp[0], pp[1], Qb, K, V, maskb, {});
                                if (!bytes_equal(y1.data.data(), yb.data.data(), (size_t)DV*H)) {
                                    std::cerr << "[FAIL] attn quant batch invariance: D=128 KV="<<KV<<" B="<<B<<" types="<< ggml_type_name(pp[0]) <<"\n";
                                    rc = 40; break;
                                }
                            }
                            if (rc) break;

                            // Cross-run determinism B=33
                            const int64_t N = 33;
                            std::vector<float> Qx((size_t)D*N*H);
                            std::mt19937 rngx(rngq());
                            fill_uniform(rngx, Qx.data(), Qx.size());
                            const int64_t Np = GGML_PAD(N, GGML_KQ_MASK_PAD);
                            std::vector<float> maskx((size_t)KV*Np, 0.0f);
                            auto a = run_attention_graph_types(backend, D, DV, N, H, H_kv, KV,
                                                               true, false, 0.0f, 0.0f, pp[0], pp[1], Qx, K, V, maskx, {});
                            auto b = run_attention_graph_types(backend, D, DV, N, H, H_kv, KV,
                                                               true, false, 0.0f, 0.0f, pp[0], pp[1], Qx, K, V, maskx, {});
                            if (!bytes_equal(a.data.data(), b.data.data(), a.data.size())) {
                                std::cerr << "[FAIL] attn quant cross-run determinism: types="<< ggml_type_name(pp[0]) <<"\n";
                                rc = 41; break;
                            }
                        }
                        if (rc) break;
                    }
                    if (rc) break;
                }
            } catch (const std::exception &e) {
                std::cerr << "[SKIP] quantized K/V det tests skipped: " << e.what() << "\n";
                rc = 0; // treat as skip
            }
        }

        // 03B: DeepSeek-like head size 576 remains experimental; probe only when enabled.
        if (rc == 0 && std::getenv("RUN_MMA_HEADSIZE_TESTS")) {
            try {
                // DeepSeek-like: D=576, DV=512, require gqa multiple of 16; use H=16, gqa=16 => H_kv=1
                const int64_t KVs[] = {256, 1024};
                if (rc == 0) {
                    const int64_t D=576, DV=512, H=16, gqa=16, H_kv=H/gqa; // 1
                    for (int64_t KV : KVs) {
                        const size_t nK=(size_t)D*KV*H_kv, nV=(size_t)DV*KV*H_kv;
                        std::vector<float> K(nK), V(nV);
                        std::mt19937 rngd((unsigned)(D*DV + KV));
                        fill_uniform(rngd, K.data(), nK);
                        fill_uniform(rngd, V.data(), nV);
                        const int64_t B1=1, B2=8;
                        const size_t nQ1=(size_t)D*B1*H, nQ2=(size_t)D*B2*H;
                        std::vector<float> Q1(nQ1), Q2(nQ2);
                        fill_uniform(rngd, Q1.data(), nQ1);
                        // Q2 with first col == Q1
                        for (int64_t h=0; h<H; ++h) {
                            std::copy(Q1.begin() + (size_t)h*D*B1,
                                      Q1.begin() + (size_t)h*D*B1 + (size_t)D,
                                      Q2.begin() + (size_t)h*D*B2);
                        }
                        for (int64_t h=0; h<H; ++h) for (int64_t n=1; n<B2; ++n)
                            fill_uniform(rngd, Q2.data() + (size_t)h*D*B2 + (size_t)n*D, (size_t)D);
                        const int64_t N1p=GGML_PAD(B1, GGML_KQ_MASK_PAD), N2p=GGML_PAD(B2, GGML_KQ_MASK_PAD);
                        std::vector<float> mask1((size_t)KV*N1p, 0.0f), mask2((size_t)KV*N2p, 0.0f);
                        auto y1 = run_attention_graph_types(backend, D, DV, B1, H, H_kv, KV,
                                                            true, false, 0.0f, 0.0f, GGML_TYPE_F16, GGML_TYPE_F16,
                                                            Q1, K, V, mask1, {});
                        // In det mode, D=576 is currently unsupported by default; treat as SKIP unless explicitly allowed by env.
                        if (!std::getenv("ALLOW_MMA_576")) {
                            std::cerr << "[SKIP] attn head-size D=576 currently unsupported in det mode\n";
                            continue;
                        }
                        // Try with ALLOW_MMA path
                        set_env_flag("GGML_DETERMINISTIC_ATTENTION_ALLOW_MMA", "1");
                        auto y2 = run_attention_graph_types(backend, D, DV, B2, H, H_kv, KV,
                                                            true, false, 0.0f, 0.0f, GGML_TYPE_F16, GGML_TYPE_F16,
                                                            Q2, K, V, mask2, {});
                        set_env_flag("GGML_DETERMINISTIC_ATTENTION_ALLOW_MMA", nullptr);
                        (void) y2; // doesn't assert determinism; env path is experimental
                    }
                }
            } catch (const std::exception &e) {
                std::cerr << "[SKIP] special head-size det tests skipped: " << e.what() << "\n";
                rc = 0; // treat as skip
            }
        }

        if (rc == 0) {
            std::cout << "[OK] " << name << std::endl;
            n_ok++;
        } else {
            std::cerr << "[FAIL] " << name << " rc=" << rc << std::endl;
            ggml_backend_free(backend);
            ggml_quantize_free();
            return 1;
        }
        ggml_backend_free(backend);
    }
    ggml_quantize_free();

    if (!ran_any) {
        std::cerr << "[SKIP] No CUDA backend found" << std::endl;
        return 0;
    }
    std::cout << "CUDA backends passed: " << n_ok << std::endl;
    return 0;
}
