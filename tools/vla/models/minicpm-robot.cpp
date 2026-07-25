// GR00T DiT action head for MiniCPM-Robot - self-contained ggml module.
// Supported config: prediction_type=clean_action, proprio_inject=concat,
// interleave_self_attention=true, num_inference_timesteps=4, F32 weights.

#include "models.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define AH_LOG_ERR(...) std::fprintf(stderr, __VA_ARGS__)
#define AH_LOG_INF(...) do { \
    if (std::getenv("VLA_VERBOSE")) std::fprintf(stdout, __VA_ARGS__); \
} while (0)
namespace {

struct dit_layer_w {
    ggml_tensor * adaln_w = nullptr;
    ggml_tensor * adaln_b = nullptr;
    ggml_tensor * Wq = nullptr;
    ggml_tensor * bq = nullptr;
    ggml_tensor * Wk = nullptr;
    ggml_tensor * bk = nullptr;
    ggml_tensor * Wv = nullptr;
    ggml_tensor * bv = nullptr;
    ggml_tensor * Wo = nullptr;
    ggml_tensor * bo = nullptr;
    ggml_tensor * Wff0 = nullptr;
    ggml_tensor * bff0 = nullptr;
    ggml_tensor * Wff2 = nullptr;
    ggml_tensor * bff2 = nullptr;
};

} // namespace

struct action_head_model {
    ggml_backend_t        backend     = nullptr;
    bool                  is_cpu      = false;
    int                   n_threads   = 8;
    ggml_context *        ctx_weights = nullptr;
    ggml_backend_buffer_t weight_buf  = nullptr;

    // hparams (overridden by GGUF metadata)
    int64_t action_dim = 80;
    int64_t state_dim  = 80;
    int64_t horizon    = 30;
    int64_t num_steps  = 4;
    int64_t num_buckets = 1000;
    int64_t n_layers = 16;
    int64_t n_heads  = 12;
    int64_t head_dim = 64;
    int64_t hidden   = 768;
    int64_t ffn      = 3072;
    int64_t cross_dim  = 1024;
    int64_t output_dim = 1024;
    int64_t dec_hidden = 1024;
    int64_t n_future = 32;
    int64_t max_pos  = 1024;
    int64_t n_embodiments = 32;
    float   ln_eps       = 1e-5f;
    float   norm_out_eps = 1e-6f;

    ggml_tensor * te_l1W = nullptr;
    ggml_tensor * te_l1b = nullptr;
    ggml_tensor * te_l2W = nullptr;
    ggml_tensor * te_l2b = nullptr;
    // CategorySpecific banks: ggml ne = [out, in, n_emb] after GGUF reverse
    ggml_tensor * enc_w1W = nullptr;
    ggml_tensor * enc_w1b = nullptr;
    ggml_tensor * enc_w2W = nullptr;
    ggml_tensor * enc_w2b = nullptr;
    ggml_tensor * enc_w3W = nullptr;
    ggml_tensor * enc_w3b = nullptr;
    ggml_tensor * dec_l1W = nullptr;
    ggml_tensor * dec_l1b = nullptr;
    ggml_tensor * dec_l2W = nullptr;
    ggml_tensor * dec_l2b = nullptr;
    ggml_tensor * future_tokens = nullptr;
    ggml_tensor * pos_embd = nullptr;
    ggml_tensor * po1W = nullptr;
    ggml_tensor * po1b = nullptr;
    ggml_tensor * po2W = nullptr;
    ggml_tensor * po2b = nullptr;
    std::vector<dit_layer_w> blk;

    // per-step sinusoid constants (num_steps == 4)
    ggml_tensor * tproj_c[4] = {};
    ggml_tensor * tau_c[4]   = {};
};

static void action_head_free(action_head_model * m);

namespace {

// -------------------------------------------------------------------------
// host-side sinusoid tables
// -------------------------------------------------------------------------

// diffusers Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1):
//   half=128; emb_i = t * exp(-log(10000) * i / (half-1)); out = [cos|sin]
void calc_tproj(int64_t bucket, std::vector<float> & out) {
    const int64_t half = 128;
    const float   lm   = std::log(10000.0f);
    const float   t    = (float) bucket;
    out.assign(256, 0.0f);
    for (int64_t i = 0; i < half; ++i) {
        const float e = t * std::exp(-lm * (float) i / (float) (half - 1));
        out[i]        = std::cos(e);
        out[half + i] = std::sin(e);
    }
}

// gr00t_action_head.py SinusoidalPositionalEncoding(768):
//   half=384; freqs_i = t * exp(-i * log(10000) / half); out = [sin|cos]
// identical for every action token -> materialize [T, dim]
void calc_tau(int64_t bucket, int64_t dim, int64_t T, std::vector<float> & out) {
    const int64_t half = dim / 2;
    const float   step = std::log(10000.0f) / (float) half;
    const float   t    = (float) bucket;
    out.assign((size_t) T * dim, 0.0f);
    for (int64_t tk = 0; tk < T; ++tk) {
        for (int64_t i = 0; i < half; ++i) {
            const float e = t * std::exp(-(float) i * step);
            out[tk * dim + i]        = std::sin(e);
            out[tk * dim + half + i] = std::cos(e);
        }
    }
}

// -------------------------------------------------------------------------
// gguf metadata helpers
// -------------------------------------------------------------------------

uint32_t kv_u32(const gguf_context * g, const char * key, uint32_t def) {
    const int64_t i = gguf_find_key(g, key);
    return i < 0 ? def : gguf_get_val_u32(g, i);
}

float kv_f32(const gguf_context * g, const char * key, float def) {
    const int64_t i = gguf_find_key(g, key);
    return i < 0 ? def : gguf_get_val_f32(g, i);
}

std::string kv_str(const gguf_context * g, const char * key, const char * def) {
    const int64_t i = gguf_find_key(g, key);
    return i < 0 ? std::string(def) : std::string(gguf_get_val_str(g, i));
}

bool kv_has(const gguf_context * g, const char * key) {
    return gguf_find_key(g, key) >= 0;
}

bool kv_bool(const gguf_context * g, const char * key, bool def) {
    const int64_t i = gguf_find_key(g, key);
    return i < 0 ? def : gguf_get_val_bool(g, i);
}

bool require_str_eq(const gguf_context * g, const char * key, const char * expect) {
    if (!kv_has(g, key)) {
        AH_LOG_ERR("vla: missing required metadata '%s' (reconvert with tools/vla/convert_hf_to_vla_gguf.py)\n", key);
        return false;
    }
    const std::string v = kv_str(g, key, "");
    if (v != expect) {
        AH_LOG_ERR("action-head: unsupported %s='%s' (expected '%s')\n", key, v.c_str(), expect);
        return false;
    }
    return true;
}

bool require_bool_eq(const gguf_context * g, const char * key, bool expect) {
    if (!kv_has(g, key)) {
        AH_LOG_ERR("vla: missing required metadata '%s' (reconvert with tools/vla/convert_hf_to_vla_gguf.py)\n", key);
        return false;
    }
    const bool v = kv_bool(g, key, !expect);
    if (v != expect) {
        AH_LOG_ERR("action-head: unsupported %s=%s (expected %s)\n",
                   key, v ? "true" : "false", expect ? "true" : "false");
        return false;
    }
    return true;
}

// CategorySpecificLinear: y = x @ W[emb] + b[emb]
// bank_W ggml ne = [in, out, n_emb], bank_b ne = [out, n_emb]
ggml_tensor * cat_linear(ggml_context * C, ggml_tensor * bank_W, ggml_tensor * bank_b,
                         ggml_tensor * x, int emb_id) {
    const int64_t in_dim  = bank_W->ne[0];
    const int64_t out_dim = bank_W->ne[1];
    ggml_tensor * W = ggml_view_2d(C, bank_W, in_dim, out_dim, bank_W->nb[1],
                                   (size_t) emb_id * bank_W->nb[2]);
    ggml_tensor * b = ggml_view_1d(C, bank_b, out_dim, (size_t) emb_id * bank_b->nb[1]);
    return ggml_add(C, ggml_mul_mat(C, W, x), b);
}
// -------------------------------------------------------------------------
// graph builders
// -------------------------------------------------------------------------

// AdaLayerNorm: cond = adaln_w @ silu(temb) + adaln_b; [scale|shift] halves
// (scale first inside the blocks); y = LN_noaffine(x) * (1 + scale) + shift
ggml_tensor * adaln(ggml_context * C, ggml_tensor * x, ggml_tensor * temb,
                    ggml_tensor * lw, ggml_tensor * lb, int64_t dim, float eps) {
    ggml_tensor * cond = ggml_add(C, ggml_mul_mat(C, lw, ggml_silu(C, temb)), lb);
    ggml_tensor * sc = ggml_view_1d(C, cond, dim, 0);
    ggml_tensor * sh = ggml_view_1d(C, cond, dim, (size_t) dim * sizeof(float));
    ggml_tensor * xn = ggml_norm(C, x, eps);
    return ggml_add(C, ggml_add(C, xn, ggml_mul(C, xn, sc)), sh);
}

// K/V projection for one DiT layer given kv source (feature dim = kv->ne[0]).
void dit_kv(ggml_context * C, const action_head_model & m, const dit_layer_w & w,
            ggml_tensor * kv, ggml_tensor ** K_out, ggml_tensor ** V_out) {
    const int64_t hd = m.head_dim, heads = m.n_heads, Tkv = kv->ne[1];
    ggml_tensor * k = ggml_add(C, ggml_mul_mat(C, w.Wk, kv), w.bk);
    ggml_tensor * v = ggml_add(C, ggml_mul_mat(C, w.Wv, kv), w.bv);
    *K_out = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, k, hd, heads, Tkv), 0, 2, 1, 3));
    *V_out = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, v, hd, heads, Tkv), 1, 2, 0, 3));
}

// One BasicTransformerBlock: adaLN -> attn (cross when K_pre/V_pre given,
// self otherwise) -> residual -> LN_noaffine -> GELU-tanh FFN -> residual.
ggml_tensor * build_dit_block(ggml_context * C, const action_head_model & m, const dit_layer_w & w,
                              ggml_tensor * h, ggml_tensor * temb,
                              ggml_tensor * K_pre, ggml_tensor * V_pre) {
    const int64_t hd = m.head_dim, heads = m.n_heads, dim = m.hidden, Tq = h->ne[1];
    const float scale = 1.0f / std::sqrt((float) hd);

    ggml_tensor * n = adaln(C, h, temb, w.adaln_w, w.adaln_b, dim, m.ln_eps);

    ggml_tensor * q = ggml_add(C, ggml_mul_mat(C, w.Wq, n), w.bq);
    ggml_tensor * Q = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, q, hd, heads, Tq), 0, 2, 1, 3));
    ggml_tensor * K, * V;
    if (K_pre) { K = K_pre; V = V_pre; }
    else       { dit_kv(C, m, w, n, &K, &V); }

    ggml_tensor * kq = ggml_mul_mat(C, K, Q);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    ggml_tensor * aw  = ggml_soft_max_ext(C, kq, nullptr, scale, 0.0f);
    ggml_tensor * att = ggml_reshape_2d(C, ggml_cont(C, ggml_permute(C, ggml_mul_mat(C, V, aw), 0, 2, 1, 3)), dim, Tq);

    ggml_tensor * h1 = ggml_add(C, h, ggml_add(C, ggml_mul_mat(C, w.Wo, att), w.bo));
    ggml_tensor * n3 = ggml_norm(C, h1, m.ln_eps);
    ggml_tensor * ff = ggml_add(C, ggml_mul_mat(C, w.Wff2, ggml_gelu(C, ggml_add(C, ggml_mul_mat(C, w.Wff0, n3), w.bff0))), w.bff2);
    return ggml_add(C, h1, ff);
}

struct dbg_item {
    std::string   name;
    ggml_tensor * t;
};

// Build one predict_clean(noisy, t_disc) pass; returns actions [action_dim, horizon].
ggml_tensor * build_predict_clean(ggml_context * C, const action_head_model & m,
                                  ggml_tensor * noisy, ggml_tensor * t_state, int s,
                                  int emb_id,
                                  const std::vector<ggml_tensor *> & Kc,
                                  const std::vector<ggml_tensor *> & Vc,
                                  std::vector<dbg_item> * dbg) {
    auto dbg_add = [&](const char * name, ggml_tensor * t) {
        if (dbg) {
            ggml_set_output(t);
            dbg->push_back({ std::string("step") + std::to_string(s) + "_" + name, t });
        }
    };

    // (a) time embedding: temb = te_l2(silu(te_l1 @ tproj + b1)) + b2   [hidden]
    ggml_tensor * temb = ggml_add(C,
        ggml_mul_mat(C, m.te_l2W, ggml_silu(C, ggml_add(C, ggml_mul_mat(C, m.te_l1W, m.tproj_c[s]), m.te_l1b))),
        m.te_l2b);
    dbg_add("temb", temb);

    // (b) MultiEmbodimentActionEncoder (proprio_inject=concat):
    //   x = [noisy | state]; a_emb = W1(x); h = W3(silu(W2([a_emb|tau])))
    ggml_tensor * x     = ggml_concat(C, noisy, t_state, 0);
    ggml_tensor * a_emb = cat_linear(C, m.enc_w1W, m.enc_w1b, x, emb_id);
    dbg_add("a_emb", a_emb);
    ggml_tensor * cat   = ggml_concat(C, a_emb, m.tau_c[s], 0);
    ggml_tensor * h2    = cat_linear(C, m.enc_w2W, m.enc_w2b, cat, emb_id);
    ggml_tensor * h     = cat_linear(C, m.enc_w3W, m.enc_w3b, ggml_silu(C, h2), emb_id);
    h = ggml_add(C, h, ggml_view_2d(C, m.pos_embd, m.hidden, m.horizon, m.pos_embd->nb[1], 0));
    dbg_add("af", h);

    // (c) sequence: [future_tokens(32) | action(T)]  -> [hidden, 32+T]
    ggml_tensor * seq = ggml_concat(C, m.future_tokens, h, 1);
    dbg_add("seq_in", seq);

    // (d) 16 DiT blocks: even layers cross-attend to vl_embs, odd layers self-attend
    for (int64_t il = 0; il < m.n_layers; ++il) {
        const bool is_cross = (il % 2 == 0);
        seq = build_dit_block(C, m, m.blk[il], seq, temb,
                              is_cross ? Kc[il] : nullptr,
                              is_cross ? Vc[il] : nullptr);
        if (dbg) {
            char nm[32];
            std::snprintf(nm, sizeof(nm), "blk%02lld", (long long) il);
            dbg_add(nm, seq);
        }
    }

    // (e) output modulation + CategorySpecificMLP decode
    ggml_tensor * po = ggml_add(C, ggml_mul_mat(C, m.po1W, ggml_silu(C, temb)), m.po1b);
    ggml_tensor * sh = ggml_view_1d(C, po, m.hidden, 0);
    ggml_tensor * sc = ggml_view_1d(C, po, m.hidden, (size_t) m.hidden * sizeof(float));
    ggml_tensor * hn = ggml_norm(C, seq, m.norm_out_eps);
    ggml_tensor * h_mod = ggml_add(C, ggml_add(C, hn, ggml_mul(C, hn, sc)), sh);
    ggml_tensor * model_output = ggml_add(C, ggml_mul_mat(C, m.po2W, h_mod), m.po2b);
    dbg_add("model_output", model_output);

    ggml_tensor * pred = cat_linear(C, m.dec_l2W, m.dec_l2b,
        ggml_relu(C, cat_linear(C, m.dec_l1W, m.dec_l1b, model_output, emb_id)), emb_id);
    dbg_add("pred_full", pred);

    ggml_tensor * actions = ggml_cont(C, ggml_view_2d(C, pred, m.action_dim, m.horizon,
        pred->nb[1], (size_t) m.n_future * pred->nb[1]));
    dbg_add("actions", actions);
    return actions;
}

} // namespace

// ===========================================================================
// public API
// ===========================================================================

static action_head_model * action_head_load(const char * gguf_path, const vla_context_params & params) {
    auto * m = new action_head_model();

    // ---- backend: GPU preferred, CPU fallback ----
    if (params.use_gpu) {
        m->backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }
    if (params.use_gpu && !m->backend) {
        m->backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    }
    if (!m->backend) {
        m->backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        m->is_cpu  = true;
        if (m->backend) {
            m->n_threads = params.n_threads > 0
                ? params.n_threads
                : std::max(1u, std::thread::hardware_concurrency() / 2);
            ggml_backend_cpu_set_n_threads(m->backend, m->n_threads);
        }
    }
    if (!m->backend) {
        AH_LOG_ERR("action-head: failed to initialize any ggml backend\n");
        delete m;
        return nullptr;
    }
    AH_LOG_INF("action-head: backend = %s\n", ggml_backend_name(m->backend));

    // ---- open GGUF with host data ----
    ggml_context * data_ctx = nullptr;
    gguf_init_params ip = { /*no_alloc =*/ false, /*ctx =*/ &data_ctx };
    gguf_context * g = gguf_init_from_file(gguf_path, ip);
    if (!g) {
        AH_LOG_ERR("action-head: failed to open GGUF '%s'\n", gguf_path);
        delete m;
        return nullptr;
    }

    const std::string arch = kv_str(g, "general.architecture", "");
    if (arch != "vla" || kv_str(g, "vla.model_type", "") != "minicpm_robot") {
        AH_LOG_ERR("vla: '%s' is not a MiniCPM-Robot VLA GGUF (arch='%s')\n", gguf_path, arch.c_str());
        gguf_free(g); ggml_free(data_ctx); delete m;
        return nullptr;
    }

    // Fail fast on configs the graph does not implement.
    if (!require_str_eq(g, "mra.prediction_type", "clean_action") ||
        !require_str_eq(g, "mra.proprio_inject", "concat") ||
        !require_bool_eq(g, "mra.interleave_self_attention", true) ||
        !require_bool_eq(g, "mra.multi_embodiment", true)) {
        gguf_free(g); ggml_free(data_ctx); delete m;
        return nullptr;
    }

    // ---- hparams (dims come from GGUF metadata written by the convert script) ----
    m->action_dim   = kv_u32(g, "vla.action_dim", 80);
    m->state_dim    = kv_u32(g, "vla.state_dim", 80);
    m->horizon      = kv_u32(g, "vla.action_horizon", 30);
    m->num_steps    = kv_u32(g, "mra.num_inference_timesteps", 4);
    m->num_buckets  = kv_u32(g, "mra.num_timestep_buckets", 1000);
    m->n_layers     = kv_u32(g, "mra.dit_layers", 16);
    m->n_heads      = kv_u32(g, "mra.dit_heads", 12);
    m->head_dim     = kv_u32(g, "mra.dit_head_dim", 64);
    m->hidden       = kv_u32(g, "mra.dit_hidden", 768);
    m->ffn          = kv_u32(g, "mra.dit_ffn", 3072);
    m->cross_dim    = kv_u32(g, "vla.conditioning_dim", 1024);
    m->output_dim   = kv_u32(g, "mra.output_dim", 1024);
    m->dec_hidden   = kv_u32(g, "mra.dec_hidden", 1024);
    m->n_future     = kv_u32(g, "mra.n_future_tokens", 32);
    m->max_pos      = kv_u32(g, "mra.max_seq_len", 1024);
    m->n_embodiments = kv_u32(g, "vla.n_embodiments", 32);
    m->ln_eps       = kv_f32(g, "mra.ln_eps", 1e-5f);
    m->norm_out_eps = kv_f32(g, "mra.norm_out_eps", 1e-6f);

    if (m->num_steps != 4) {
        AH_LOG_ERR("action-head: num_inference_timesteps=%lld unsupported (expected 4)\n", (long long) m->num_steps);
        gguf_free(g); ggml_free(data_ctx); delete m;
        return nullptr;
    }
    if (m->n_layers <= 0 || m->hidden <= 0 || m->action_dim <= 0 || m->horizon <= 0 || m->n_embodiments <= 0) {
        AH_LOG_ERR("action-head: invalid hparams (layers/hidden/action_dim/horizon/n_embodiments)\n");
        gguf_free(g); ggml_free(data_ctx); delete m;
        return nullptr;
    }
    if (m->n_heads * m->head_dim != m->hidden) {
        AH_LOG_ERR("action-head: dit_heads*dit_head_dim (%lld*%lld) != dit_hidden (%lld)\n",
                   (long long) m->n_heads, (long long) m->head_dim, (long long) m->hidden);
        gguf_free(g); ggml_free(data_ctx); delete m;
        return nullptr;
    }
    AH_LOG_INF("action-head: dit=%lldL x %lldh x %lld (inner %lld, ffn %lld)  cross_dim=%lld  "
               "horizon=%lld action_dim=%lld state_dim=%lld  steps=%lld  future=%lld\n",
               (long long) m->n_layers, (long long) m->n_heads, (long long) m->head_dim,
               (long long) m->hidden, (long long) m->ffn, (long long) m->cross_dim,
               (long long) m->horizon, (long long) m->action_dim, (long long) m->state_dim,
               (long long) m->num_steps, (long long) m->n_future);

    // ---- weight context (meta only) ----
    const int64_t n_gguf_tensors = gguf_get_n_tensors(g);
    const size_t  n_extra        = 8;  // 4x tproj + 4x tau
    ggml_init_params wp = {
        /*mem_size   =*/ (size_t) (n_gguf_tensors + n_extra + 8) * ggml_tensor_overhead(),
        /*mem_buffer =*/ nullptr,
        /*no_alloc   =*/ true,
    };
    m->ctx_weights = ggml_init(wp);
    if (!m->ctx_weights) {
        AH_LOG_ERR("action-head: ggml_init(ctx_weights) failed\n");
        gguf_free(g); ggml_free(data_ctx); delete m;
        return nullptr;
    }
    ggml_context * W = m->ctx_weights;

    bool ok = true;
    auto mk = [&](const std::string & name) -> ggml_tensor * {
        ggml_tensor * src = ggml_get_tensor(data_ctx, name.c_str());
        if (!src) {
            AH_LOG_ERR("action-head: missing tensor %s\n", name.c_str());
            ok = false;
            return nullptr;
        }
        if (src->type != GGML_TYPE_F32) {
            AH_LOG_ERR("action-head: tensor %s is not F32\n", name.c_str());
            ok = false;
            return nullptr;
        }
        ggml_tensor * t = ggml_new_tensor(W, GGML_TYPE_F32, ggml_n_dims(src), src->ne);
        ggml_set_name(t, name.c_str());
        return t;
    };

    m->te_l1W = mk("act.time_emb.l1.weight"); m->te_l1b = mk("act.time_emb.l1.bias");
    m->te_l2W = mk("act.time_emb.l2.weight"); m->te_l2b = mk("act.time_emb.l2.bias");
    m->enc_w1W = mk("act.enc.w1.weight"); m->enc_w1b = mk("act.enc.w1.bias");
    m->enc_w2W = mk("act.enc.w2.weight"); m->enc_w2b = mk("act.enc.w2.bias");
    m->enc_w3W = mk("act.enc.w3.weight"); m->enc_w3b = mk("act.enc.w3.bias");
    m->dec_l1W = mk("act.dec.l1.weight"); m->dec_l1b = mk("act.dec.l1.bias");
    m->dec_l2W = mk("act.dec.l2.weight"); m->dec_l2b = mk("act.dec.l2.bias");
    m->future_tokens = mk("act.future_tokens");
    m->pos_embd      = mk("act.pos_embd");
    m->po1W = mk("act.proj_out1.weight"); m->po1b = mk("act.proj_out1.bias");
    m->po2W = mk("act.proj_out2.weight"); m->po2b = mk("act.proj_out2.bias");

    // Validate CategorySpecific banks: ne = [in, out, n_emb]
    auto check_cat_W = [&](ggml_tensor * t, int64_t in_d, int64_t out_d, const char * name) {
        if (!ok || !t) return;
        if (ggml_n_dims(t) < 3 || t->ne[0] != in_d || t->ne[1] != out_d || t->ne[2] != m->n_embodiments) {
            AH_LOG_ERR("action-head: %s shape [%lld,%lld,%lld], expected [%lld,%lld,%lld]\n",
                       name, (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
                       (long long) in_d, (long long) out_d, (long long) m->n_embodiments);
            ok = false;
        }
    };
    auto check_cat_b = [&](ggml_tensor * t, int64_t out_d, const char * name) {
        if (!ok || !t) return;
        if (t->ne[0] != out_d || t->ne[1] != m->n_embodiments) {
            AH_LOG_ERR("action-head: %s shape [%lld,%lld], expected [%lld,%lld]\n",
                       name, (long long) t->ne[0], (long long) t->ne[1],
                       (long long) out_d, (long long) m->n_embodiments);
            ok = false;
        }
    };
    const int64_t enc_in = m->action_dim + m->state_dim;
    check_cat_W(m->enc_w1W, enc_in, m->hidden, "enc.w1.weight");
    check_cat_b(m->enc_w1b, m->hidden, "enc.w1.bias");
    check_cat_W(m->enc_w2W, 2 * m->hidden, m->hidden, "enc.w2.weight");
    check_cat_b(m->enc_w2b, m->hidden, "enc.w2.bias");
    check_cat_W(m->enc_w3W, m->hidden, m->hidden, "enc.w3.weight");
    check_cat_b(m->enc_w3b, m->hidden, "enc.w3.bias");
    check_cat_W(m->dec_l1W, m->output_dim, m->dec_hidden, "dec.l1.weight");
    check_cat_b(m->dec_l1b, m->dec_hidden, "dec.l1.bias");
    check_cat_W(m->dec_l2W, m->dec_hidden, m->action_dim, "dec.l2.weight");
    check_cat_b(m->dec_l2b, m->action_dim, "dec.l2.bias");

    m->blk.resize(m->n_layers);
    for (int64_t i = 0; i < m->n_layers; ++i) {
        char p[64];
        auto N = [&](const char * s) { std::snprintf(p, sizeof(p), "act.blk.%lld.%s", (long long) i, s); return std::string(p); };
        auto & w = m->blk[i];
        w.adaln_w = mk(N("adaln.weight")); w.adaln_b = mk(N("adaln.bias"));
        w.Wq = mk(N("attn_q.weight")); w.bq = mk(N("attn_q.bias"));
        w.Wk = mk(N("attn_k.weight")); w.bk = mk(N("attn_k.bias"));
        w.Wv = mk(N("attn_v.weight")); w.bv = mk(N("attn_v.bias"));
        w.Wo = mk(N("attn_o.weight")); w.bo = mk(N("attn_o.bias"));
        w.Wff0 = mk(N("ff0.weight")); w.bff0 = mk(N("ff0.bias"));
        w.Wff2 = mk(N("ff2.weight")); w.bff2 = mk(N("ff2.bias"));

        // sanity: even layers cross-attend (K width = cross_dim), odd self (K width = hidden)
        if (ok) {
            const int64_t expect = (i % 2 == 0) ? m->cross_dim : m->hidden;
            if (w.Wk->ne[0] != expect) {
                AH_LOG_ERR("action-head: blk %lld attn_k in-dim %lld, expected %lld (%s layer)\n",
                           (long long) i, (long long) w.Wk->ne[0], (long long) expect,
                           (i % 2 == 0) ? "cross" : "self");
                ok = false;
            }
        }
    }

    // per-step sinusoid constants
    for (int s = 0; s < 4 && ok; ++s) {
        char nm[32];
        std::snprintf(nm, sizeof(nm), "const.tproj.%d", s);
        m->tproj_c[s] = ggml_new_tensor_1d(W, GGML_TYPE_F32, 256);
        ggml_set_name(m->tproj_c[s], nm);
        std::snprintf(nm, sizeof(nm), "const.tau.%d", s);
        m->tau_c[s] = ggml_new_tensor_2d(W, GGML_TYPE_F32, m->hidden, m->horizon);
        ggml_set_name(m->tau_c[s], nm);
    }

    if (!ok) {
        gguf_free(g); ggml_free(data_ctx);
        action_head_free(m);
        return nullptr;
    }

    // ---- allocate on backend and upload ----
    m->weight_buf = ggml_backend_alloc_ctx_tensors(m->ctx_weights, m->backend);
    if (!m->weight_buf) {
        AH_LOG_ERR("action-head: ggml_backend_alloc_ctx_tensors failed (OOM?)\n");
        gguf_free(g); ggml_free(data_ctx);
        action_head_free(m);
        return nullptr;
    }

    int64_t n_loaded = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(W); t; t = ggml_get_next_tensor(W, t)) {
        const char * name = ggml_get_name(t);
        if (std::strncmp(name, "const.", 6) == 0) {
            continue;  // filled below
        }
        ggml_tensor * src = ggml_get_tensor(data_ctx, name);
        if (!src || ggml_nbytes(src) != ggml_nbytes(t)) {
            AH_LOG_ERR("action-head: failed to load %s\n", name);
            gguf_free(g); ggml_free(data_ctx);
            action_head_free(m);
            return nullptr;
        }
        ggml_backend_tensor_set(t, src->data, 0, ggml_nbytes(t));
        n_loaded++;
    }

    // fill sinusoid constants: step s (0..3) -> t_disc for step = num_steps - s
    for (int s = 0; s < 4; ++s) {
        const int64_t step   = m->num_steps - s;                       // 4,3,2,1
        const double  t_cont = (double) step / (double) m->num_steps;  // 1.0,.75,.5,.25
        int64_t t_disc = (int64_t) (t_cont * (double) m->num_buckets);
        if (t_disc > m->num_buckets - 1) t_disc = m->num_buckets - 1;  // 999,750,500,250

        std::vector<float> tproj, tau;
        calc_tproj(t_disc, tproj);
        calc_tau(t_disc, m->hidden, m->horizon, tau);
        ggml_backend_tensor_set(m->tproj_c[s], tproj.data(), 0, ggml_nbytes(m->tproj_c[s]));
        ggml_backend_tensor_set(m->tau_c[s],   tau.data(),   0, ggml_nbytes(m->tau_c[s]));
    }

    // Shared weights: 20 globals + 14 per DiT layer (see convert name map).
    const int64_t n_expected = 20 + 14 * m->n_layers;
    if (n_loaded != n_expected) {
        AH_LOG_ERR("action-head: loaded %lld weight tensors, expected %lld\n",
                   (long long) n_loaded, (long long) n_expected);
        gguf_free(g); ggml_free(data_ctx);
        action_head_free(m);
        return nullptr;
    }

    AH_LOG_INF("action-head: %lld weight tensors resident in %.1f MiB (F32)\n",
               (long long) n_loaded, ggml_backend_buffer_get_size(m->weight_buf) / (1024.0 * 1024.0));

    gguf_free(g);
    ggml_free(data_ctx);
    return m;
}

static void action_head_free(action_head_model * m) {
    if (!m) return;
    if (m->weight_buf)  ggml_backend_buffer_free(m->weight_buf);
    if (m->ctx_weights) ggml_free(m->ctx_weights);
    if (m->backend)     ggml_backend_free(m->backend);
    delete m;
}

static bool action_head_predict(action_head_model * m,
                         const float * vl_embs, int64_t n_tokens,
                         const float * state,
                         const float * noise,
                         int embodiment_id,
                         float * out) {
    if (!m || !vl_embs || n_tokens <= 0 || !state || !out) {
        AH_LOG_ERR("action-head: invalid predict arguments\n");
        return false;
    }
    if (embodiment_id < 0 || (int64_t) embodiment_id >= m->n_embodiments) {
        AH_LOG_ERR("action-head: embodiment_id=%d out of range [0,%lld)\n",
                   embodiment_id, (long long) m->n_embodiments);
        return false;
    }

    const auto t0 = std::chrono::steady_clock::now();

    const int64_t AD = m->action_dim, T = m->horizon, S = n_tokens;

    // noise: use caller's buffer or sample N(0,1)
    std::vector<float> noise_host((size_t) T * AD);
    if (noise) {
        std::memcpy(noise_host.data(), noise, noise_host.size() * sizeof(float));
    } else {
        std::mt19937 rng((uint32_t) std::chrono::steady_clock::now().time_since_epoch().count());
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto & v : noise_host) v = nd(rng);
    }

    // debug dump dir (set MRA_DEBUG_DUMP=<dir> to dump intermediates)
    const char * dump_dir = std::getenv("MRA_DEBUG_DUMP");
    std::vector<dbg_item> dbg_items;
    std::vector<dbg_item> * dbg = dump_dir && dump_dir[0] ? &dbg_items : nullptr;

    ggml_init_params cp = { (size_t) 64 * 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(cp);
    if (!C) {
        AH_LOG_ERR("action-head: ggml_init(compute ctx) failed\n");
        return false;
    }

    // ---- graph inputs ----
    ggml_tensor * t_vl    = ggml_new_tensor_2d(C, GGML_TYPE_F32, m->cross_dim, S); ggml_set_input(t_vl);
    ggml_tensor * t_state = ggml_new_tensor_2d(C, GGML_TYPE_F32, m->state_dim, T); ggml_set_input(t_state);
    ggml_tensor * t_noise = ggml_new_tensor_2d(C, GGML_TYPE_F32, AD, T);           ggml_set_input(t_noise);

    // ---- precompute cross-attention K/V for even layers (vl_embs is step-invariant) ----
    std::vector<ggml_tensor *> Kc(m->n_layers, nullptr), Vc(m->n_layers, nullptr);
    for (int64_t i = 0; i < m->n_layers; ++i) {
        if (i % 2 != 0) continue;
        dit_kv(C, *m, m->blk[i], t_vl, &Kc[i], &Vc[i]);
    }

    // ---- clean_action sampler, all steps unrolled into ONE graph ----
    ggml_tensor * actions = nullptr;
    for (int s = 0; s < (int) m->num_steps; ++s) {
        const float t_cont = (float) (m->num_steps - s) / (float) m->num_steps;
        ggml_tensor * noisy = (s == 0)
            ? t_noise
            : ggml_add(C, ggml_scale(C, t_noise, t_cont), ggml_scale(C, actions, 1.0f - t_cont));
        actions = build_predict_clean(C, *m, noisy, t_state, s, embodiment_id, Kc, Vc, dbg);
    }
    ggml_set_name(actions, "action_pred");
    ggml_set_output(actions);

    ggml_cgraph * gf = ggml_new_graph_custom(C, 8192, false);
    ggml_build_forward_expand(gf, actions);
    if (dbg) {
        for (auto & it : dbg_items) ggml_build_forward_expand(gf, it.t);
    }

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
        AH_LOG_ERR("action-head: gallocr alloc failed\n");
        if (galloc) ggml_gallocr_free(galloc);
        ggml_free(C);
        return false;
    }

    // ---- set inputs ----
    ggml_backend_tensor_set(t_vl, vl_embs, 0, ggml_nbytes(t_vl));
    {
        std::vector<float> st((size_t) m->state_dim * T);
        for (int64_t tk = 0; tk < T; ++tk) {
            std::memcpy(st.data() + tk * m->state_dim, state, m->state_dim * sizeof(float));
        }
        ggml_backend_tensor_set(t_state, st.data(), 0, ggml_nbytes(t_state));
    }
    ggml_backend_tensor_set(t_noise, noise_host.data(), 0, ggml_nbytes(t_noise));

    // ---- compute ----
    const ggml_status st = ggml_backend_graph_compute(m->backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        AH_LOG_ERR("action-head: graph compute failed (%d)\n", (int) st);
        ggml_gallocr_free(galloc);
        ggml_free(C);
        return false;
    }

    ggml_backend_tensor_get(actions, out, 0, (size_t) T * AD * sizeof(float));

    // ---- debug dump ----
    if (dbg) {
        for (auto & it : dbg_items) {
            std::string path = std::string(dump_dir) + "/" + it.name + ".bin";
            FILE * f = std::fopen(path.c_str(), "wb");
            if (!f) { AH_LOG_ERR("action-head: cannot open %s\n", path.c_str()); continue; }
            std::vector<uint8_t> buf(ggml_nbytes(it.t));
            ggml_backend_tensor_get(it.t, buf.data(), 0, buf.size());
            std::fwrite(buf.data(), 1, buf.size(), f);
            std::fclose(f);
        }
        AH_LOG_INF("action-head: dumped %zu debug tensors to %s\n", dbg_items.size(), dump_dir);
    }

    ggml_gallocr_free(galloc);
    ggml_free(C);

    const float ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    AH_LOG_INF("action-head: predict S=%lld emb=%d done in %.1f ms\n",
               (long long) S, embodiment_id, ms);
    return true;
}

namespace {

class minicpm_robot_vla_model final : public vla_model {
public:
    explicit minicpm_robot_vla_model(action_head_model * model) : model_(model) {}

    ~minicpm_robot_vla_model() override {
        action_head_free(model_);
    }

    const char * model_type() const override {
        return "minicpm_robot";
    }

    int64_t state_dim() const override {
        return model_->state_dim;
    }

    int64_t action_dim() const override {
        return model_->action_dim;
    }

    int64_t action_horizon() const override {
        return model_->horizon;
    }

    int64_t conditioning_dim() const override {
        return model_->cross_dim;
    }

    int64_t n_embodiments() const override {
        return model_->n_embodiments;
    }

    bool predict(const vla_input & input, vla_output & output) override {
        return action_head_predict(
                model_,
                input.embeddings,
                input.n_tokens,
                input.state,
                input.noise,
                input.embodiment_id,
                output.actions);
    }

private:
    action_head_model * model_;
};

} // namespace

std::unique_ptr<vla_model> vla_model_minicpm_robot_create(
        const char * path,
        const vla_context_params & params) {
    action_head_model * model = action_head_load(path, params);
    if (!model) {
        return nullptr;
    }
    return std::make_unique<minicpm_robot_vla_model>(model);
}
