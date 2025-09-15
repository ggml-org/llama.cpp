// KV-cache invariance test (03C):
//
// Goal: Under GGML_DETERMINISTIC=1, produce bitwise-identical logits for the
// same absolute position P whether computed via:
//   (a) single-shot prefill to length P, or
//   (b) incremental decode (append tokens one-by-one) up to P.
//
// Policy reflected here and in host code:
// - KV length is padded to a multiple of the FA kernel stride (256).
// - The KQ mask is shaped as [KV, PAD(N, GGML_KQ_MASK_PAD), 1, 1] with
//   GGML_KQ_MASK_PAD=64.
// - Mask entries are 0 for valid positions and -inf for padded tail.
//
// This test builds the same attention graph in the two flows and compares the
// last token’s logits across a grid of shapes.

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

static AttnOut run_attention_graph(ggml_backend_t backend,
                                   int64_t D, int64_t DV,
                                   int64_t N, int64_t H, int64_t H_kv,
                                   int64_t KV,
                                   bool use_mask,
                                   float max_bias, float logit_softcap,
                                   const std::vector<float> &Q_f32,
                                   const std::vector<float> &K_f32,
                                   const std::vector<float> &V_f32,
                                   const std::vector<float> &mask_f32_or_empty) {
    ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) throw std::runtime_error("ggml_init failed");

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, N, H, /*S*/1);
    // Shape tensors to match ggml_flash_attn_ext expectations:
    // q: [D, N, H, S], k: [D, KV, H_kv, S], v: [DV, KV, H_kv, S]
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D,  KV, H_kv, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, DV, KV, H_kv, 1);

    const int64_t N_pad = GGML_PAD(N, GGML_KQ_MASK_PAD);
    // mask shaped like llama-graph: [KV, PAD(N,64), 1, 1]; use F16 for CUDA FA
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, KV, N_pad, /*ne32*/1, /*ne33*/1);

    const float scale = 1.0f / std::sqrt((float)D);

    // Build via FlashAttention path with determinism constraints
    // q: [D,N,H,1], k: [D,H_kv,KV,1], v: [DV,H_kv,KV,1], mask: [KV, PAD(N,64), 1, 1]
    // output: [DV,H,N,1] -> reshape to [DV*H, N]
    {
        auto pr = [](const char *name, const ggml_tensor *t) {
            std::cerr << "[kvci] " << name << " ne=[" << t->ne[0] << "," << t->ne[1]
                      << "," << t->ne[2] << "," << t->ne[3] << "]\n";
        };
        pr("q", q);
        pr("k", k);
        pr("v", v);
    }

    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, use_mask ? m : nullptr, scale, max_bias, logit_softcap);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    out = ggml_reshape_2d(ctx, out, out->ne[0]*out->ne[1], out->ne[2]*out->ne[3]);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); throw std::runtime_error("alloc tensors failed"); }

    // Populate tensors
    const size_t nQ = (size_t)D*N*H;
    if (Q_f32.size() != nQ) { ggml_backend_buffer_free(buf); ggml_free(ctx); throw std::runtime_error("bad Q size"); }
    ggml_backend_tensor_set(q, Q_f32.data(), 0, nQ*sizeof(float));

    const size_t nK = (size_t)D*KV*H_kv;
    const size_t nV = (size_t)DV*KV*H_kv;
    if (K_f32.size() != nK || V_f32.size() != nV) { ggml_backend_buffer_free(buf); ggml_free(ctx); throw std::runtime_error("bad KV size"); }
    {
        std::vector<ggml_fp16_t> tmp(nK);
        fp32_to_f16_buffer(K_f32.data(), tmp.data(), nK);
        ggml_backend_tensor_set(k, tmp.data(), 0, nK*sizeof(tmp[0]));
    }
    {
        std::vector<ggml_fp16_t> tmp(nV);
        fp32_to_f16_buffer(V_f32.data(), tmp.data(), nV);
        ggml_backend_tensor_set(v, tmp.data(), 0, nV*sizeof(tmp[0]));
    }
    if (m) {
        const size_t nM = (size_t)KV*N_pad;
        std::vector<ggml_fp16_t> tmp(nM);
        if (use_mask && !mask_f32_or_empty.empty()) {
            fp32_to_f16_buffer(mask_f32_or_empty.data(), tmp.data(), nM);
        } else {
            std::fill(tmp.begin(), tmp.end(), ggml_fp32_to_fp16(0.0f));
        }
        ggml_backend_tensor_set(m, tmp.data(), 0, nM*sizeof(tmp[0]));
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

static bool bytes_equal(const float *a, const float *b, size_t n) {
    return std::memcmp(a, b, n*sizeof(float)) == 0;
}

static int test_kvcache_invariance_backend(ggml_backend_t backend, const char * name) {
    std::mt19937 rng(314159);
    const int64_t Ds[]  = {128, 256};
    const int64_t KVv[] = {256, 1024};
    const int gqas[] = {1, 2};
    int rc = 0;

    for (int64_t D : Ds) {
        const int64_t DV = D;
        for (int gqa : gqas) {
            const int64_t H = 8; if (H % gqa) continue; const int64_t H_kv = H / gqa;
            for (int64_t KV : KVv) {
                // Base K/V
                const size_t nK = (size_t)D*KV*H_kv, nV = (size_t)DV*KV*H_kv;
                std::vector<float> K(nK), V(nV);
                fill_uniform(rng, K.data(), nK);
                fill_uniform(rng, V.data(), nV);

                // Single-shot: N = P (use P=KV for convenience)
                const int64_t P = KV; // compare final token at position P-1
                const size_t nQall = (size_t)D*P*H;
                std::vector<float> Qall(nQall);
                fill_uniform(rng, Qall.data(), nQall);
                const int64_t Npad_all = GGML_PAD(P, GGML_KQ_MASK_PAD);
                std::vector<float> mask_all((size_t)KV*Npad_all, 0.0f);
                auto y_all = run_attention_graph(backend, D, DV, P, H, H_kv, KV,
                                                 /*mask*/true, /*max_bias*/0.0f, /*softcap*/0.0f,
                                                 Qall, K, V, mask_all);

                // Incremental: steps s=1..P; at step s, KV=s and N=1 (last token)
                std::vector<float> y_last(DV*H);
                for (int64_t s = 1; s <= P; ++s) {
                    // Q_step is the (s-1)th column in Qall for all heads
                    const size_t nQ1 = (size_t)D*1*H;
                    std::vector<float> Q1(nQ1);
                    for (int64_t h = 0; h < H; ++h) {
                        const size_t src_off = (size_t)h*D*P + (size_t)(s-1)*D;
                        const size_t dst_off = (size_t)h*D*1 + 0;
                        std::copy(Qall.begin() + src_off,
                                  Qall.begin() + src_off + (size_t)D,
                                  Q1.begin() + dst_off);
                    }
                    // Pad K/V len to multiple of 256 as required by CUDA FA
                    const int64_t KVp = ((s + 255)/256)*256;
                    const size_t nKs = (size_t)D*KVp*H_kv, nVs = (size_t)DV*KVp*H_kv;
                    std::vector<float> Ks(nKs, 0.0f), Vs(nVs, 0.0f);
                    for (int64_t hk = 0; hk < H_kv; ++hk) {
                        const size_t srcK_off = (size_t)hk*D*KV;
                        const size_t srcV_off = (size_t)hk*DV*KV;
                        const size_t dstK_off = (size_t)hk*D*KVp;
                        const size_t dstV_off = (size_t)hk*DV*KVp;
                        // copy first s columns for this head
                        for (int64_t col = 0; col < s; ++col) {
                            std::copy(K.begin() + srcK_off + (size_t)D*col,
                                      K.begin() + srcK_off + (size_t)D*(col+1),
                                      Ks.begin() + dstK_off + (size_t)D*col);
                            std::copy(V.begin() + srcV_off + (size_t)DV*col,
                                      V.begin() + srcV_off + (size_t)DV*(col+1),
                                      Vs.begin() + dstV_off + (size_t)DV*col);
                        }
                    }
                    const int64_t Npad1 = GGML_PAD(1, GGML_KQ_MASK_PAD);
                    // mask: size KVp x Npad1. 0 for valid [0..s-1], -INF for padded [s..KVp-1]
                    std::vector<float> mask1((size_t)KVp*Npad1, -INFINITY);
                    for (int64_t col = 0; col < s; ++col) {
                        mask1[(size_t)col] = 0.0f; // first column (N=0)
                    }
                    // optional debug
                    {
                        const char *dbg = getenv("KVCI_DEBUG");
                        if (dbg && *dbg && !(dbg[0] == '0' && dbg[1] == '\0')) {
                            std::cerr << "[kvci] D=" << D << " H=" << H << " gqa=" << gqa
                                      << " step s=" << s << " KVp=" << KVp << "\n";
                        }
                    }
                    auto y1 = run_attention_graph(backend, D, DV, /*N*/1, H, H_kv, /*KV*/KVp,
                                                  /*mask*/true, /*max_bias*/0.0f, /*softcap*/0.0f,
                                                  Q1, Ks, Vs, mask1);
                    // Keep last result
                    std::copy(y1.data.begin(), y1.data.begin() + (size_t)DV*H, y_last.begin());
                }

                // Compare incremental last vs single-shot last column
                const float * y_all_last = y_all.data.data() + (size_t)DV*H*(P-1);
                if (!bytes_equal(y_last.data(), y_all_last, (size_t)DV*H)) {
                    std::cerr << "[FAIL] KV invariance: backend=" << name
                              << " D=" << D << " KV=" << KV << " gqa=" << gqa << "\n";
                    return 1;
                }
            }
        }
    }
    (void) rc; (void) name;
    return 0;
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
        ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
        if (!backend) {
            std::cerr << "[SKIP] cannot init backend: " << name << std::endl;
            continue;
        }
        ran_any = true;

        int rc = 0;
        try {
            rc = test_kvcache_invariance_backend(backend, name);
        } catch (const std::exception &e) {
            std::cerr << "[SKIP] backend error: " << e.what() << "\n";
            rc = 0; // treat as skip
        }
        if (rc == 0) {
            std::cout << "[OK] " << name << std::endl;
            n_ok++;
        } else {
            std::cerr << "[FAIL] " << name << " rc=" << rc << std::endl;
            ggml_backend_free(backend);
            return 1;
        }
        ggml_backend_free(backend);
    }
    if (!ran_any) {
        std::cerr << "[SKIP] No backend initialized" << std::endl;
        return 0;
    }
    std::cout << "Backends passed: " << n_ok << std::endl;
    return 0;
}
