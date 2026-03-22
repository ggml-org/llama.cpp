#include "models.h"
#include <cmath>

// Phi-4-mini-flash-reasoning (SambaY-DA architecture)
//
// Layer map:
//   even 0,2,...,16  : Mamba-1
//   odd  1,3,...,15  : SWA differential attention  (512-token window)
//   17               : Full differential attention  (YOCO pivot: saves K,V)
//   even 18,20,...,30: GMU  (Gated Memory Unit)
//   odd  19,21,...,31: Cross differential attention (YOCO K,V from layer 17)
//
// === EXACT DIFFERENTIAL ATTENTION (fa_og=False path) ===
//
//   QKV from Wqkv (after GGUF head-reorder, block layout):
//     Q rows 0..19   = signal Q1  (even heads 0,2,...,38)
//     Q rows 20..39  = noise  Q2  (odd  heads 1,3,...,39)
//     K rows 0..9    = signal K1  (even heads 0,2,...,18)
//     K rows 10..19  = noise  K2  (odd  heads 1,3,...,19)
//     V [64,20,T]    = reshaped to V_meta [128,10,T] (same bytes, no copy)
//
//   attn1 = softmax(Q1 @ K1^T / sqrt(64)) @ V_meta   [128, 20, T]
//   attn2 = softmax(Q2 @ K2^T / sqrt(64)) @ V_meta   [128, 20, T]
//   lambda = exp(dot(lq1,lk1)) - exp(dot(lq2,lk2)) + lambda_init
//   diff   = attn1 - lambda * attn2                   [128, 20, T]
//   diff   = subln(diff)   (RMSNorm + learned scale [128])
//   diff   = diff * (1 - lambda_init)
//   out    = rearrange(diff, "H (two D) -> (H two) D")  [64, 40, T]
//   out    = wo(flatten(out))
//
// === KV CACHE ===
//   We use four helper methods added to llm_graph_context:
//     write_kv_iswa(inp, K, V, il)     -- writes K[64,20,T] and V[64,20,T] to cache
//     read_k_iswa(inp, il)             -- reads K[64,20,n_kv] from cache
//     read_v_iswa(inp, il)             -- reads V[64,20,n_kv] from cache
//     get_kq_mask_iswa(inp, il)        -- gets causal / SWA mask tensor
//   After reading V we reshape [64,20,n_kv] -> [128,10,n_kv] (same bytes, no copy).
//   Then build_attn_mha is called twice: Q1/K1/V_meta and Q2/K2/V_meta.
//
// === OPTIMIZATIONS ===
//
//   OPT 1 - PRE-COMPUTED LAMBDA SCALARS:
//     exp(dot(lq1,lk1)) - exp(dot(lq2,lk2)) is constant per attention layer.
//     We read the lambda weight tensors safely via ggml_backend_tensor_get()
//     (which works for tensors on any backend: CPU, CUDA, Metal, etc.) and
//     compute the scalar once at model load time.
//     In the graph: ggml_scale(attn2, lambda_full) replaces the original chain of
//     ggml_mul + ggml_sum + ggml_exp*2 + ggml_sub + ggml_repeat ops.
//
//   OPT 2 - PRE-BAKED (1-lambda_init) INTO SUBLN WEIGHT SCALE:
//     ggml_scale(subln_w [128], 1-lf) costs 128 muls, but saves ggml_scale on
//     normed [128,20,T] which costs 128*20*T muls.  Net win for all T >= 1.
//
//   OPT 3 - K STRIDE CORRECTNESS (bug fix):
//     K_cached [64, 20, n_kv]: nb[2] is the token stride, not nb[1].

// ---------------------------------------------------------------------------
// Per-layer pre-computed constants (computed once from model weights).
// ggml_backend_tensor_get() is used to safely read tensors from any backend.
// ---------------------------------------------------------------------------
struct phi4flash_layer_consts {
    float lambda_full;   // exp(dot(lq1,lk1)) - exp(dot(lq2,lk2)) + lambda_init
    float scale_subln;   // (1 - lambda_init)
    bool  ready;         // false => fall back to in-graph computation
};

static phi4flash_layer_consts g_layer_consts[32] = {};
static bool g_consts_initialized = false;
static bool g_consts_all_ready   = false;  // true once every attn layer succeeded

// Safe helper: copy at most `max_n` floats from a tensor (any backend, any type)
// into `out`.  Handles F32, F16, and BF16.  For quantized types it falls back to
// returning false so the caller can skip the optimisation for that layer.
static bool tensor_to_float_cpu(const ggml_tensor * t, float * out, int64_t max_n) {
    // Guard: during sched_reserve, build_graph is called before backend buffers
    // are fully assigned.  A tensor may have data != NULL but buffer == NULL,
    // which causes ggml_backend_tensor_get to assert internally.
    // We must check both data and buffer before touching the tensor.
    if (t == nullptr || t->data == nullptr || t->buffer == nullptr) return false;

    const int64_t n = ggml_nelements(t);
    if (n > max_n) return false;

    // Allocate a raw byte buffer and pull from whatever backend holds the tensor.
    const size_t nbytes = ggml_nbytes(t);
    std::vector<char> raw(nbytes);
    ggml_backend_tensor_get(t, raw.data(), 0, nbytes);

    // Convert to float based on stored type.
    switch (t->type) {
        case GGML_TYPE_F32: {
            memcpy(out, raw.data(), n * sizeof(float));
            return true;
        }
        case GGML_TYPE_F16: {
            const ggml_fp16_t * src = reinterpret_cast<const ggml_fp16_t *>(raw.data());
            for (int64_t i = 0; i < n; ++i) {
                out[i] = ggml_fp16_to_fp32(src[i]);
            }
            return true;
        }
        case GGML_TYPE_BF16: {
            // BF16: upper 16 bits of a float32.  Shift left to recover float.
            const uint16_t * src = reinterpret_cast<const uint16_t *>(raw.data());
            for (int64_t i = 0; i < n; ++i) {
                uint32_t v = (uint32_t)src[i] << 16;
                memcpy(&out[i], &v, sizeof(float));
            }
            return true;
        }
        default:
            // Quantized types (Q8_0, etc.) -- skip optimisation for this layer.
            return false;
    }
}

static bool compute_lambda_const(const llama_layer & layer, float & out) {
    float lq1[64], lk1[64], lq2[64], lk2[64];
    if (!tensor_to_float_cpu(layer.attn_lambda_q1, lq1, 64)) return false;
    if (!tensor_to_float_cpu(layer.attn_lambda_k1, lk1, 64)) return false;
    if (!tensor_to_float_cpu(layer.attn_lambda_q2, lq2, 64)) return false;
    if (!tensor_to_float_cpu(layer.attn_lambda_k2, lk2, 64)) return false;

    const int64_t n = ggml_nelements(layer.attn_lambda_q1);
    double dot1 = 0.0, dot2 = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        dot1 += (double)lq1[i] * (double)lk1[i];
        dot2 += (double)lq2[i] * (double)lk2[i];
    }
    out = (float)(std::exp(dot1) - std::exp(dot2));
    return true;
}

// ---------------------------------------------------------------------------
llm_build_phi4flash::llm_build_phi4flash(const llama_model & model, const llm_graph_params & params)
    : llm_build_mamba_base(params) {

    const int64_t n_embd        = hparams.n_embd;
    const int64_t n_head        = hparams.n_head(1);
    const int64_t n_head_kv     = hparams.n_head_kv(1);
    const int64_t n_embd_head_k = hparams.n_embd_head_k();
    const int64_t half_heads    = n_head    / 2;
    const int64_t half_kv_heads = n_head_kv / 2;
    const int64_t meta_dim      = 2 * n_embd_head_k;
    const float   attn_scale    = 1.0f / sqrtf((float)n_embd_head_k);
    const float   norm_eps      = hparams.f_norm_rms_eps;

    // -------------------------------------------------------------------------
    // OPT 1+2: Pre-compute per-layer constants from model weights.
    // Uses ggml_backend_tensor_get() -- safe for any backend (CPU, CUDA, Metal).
    // The constructor is called multiple times by sched_reserve before all backend
    // buffers are assigned (tensor->data may be NULL on early calls).  We retry
    // on every call until all attention layers have been successfully initialised.
    // -------------------------------------------------------------------------
    if (!g_consts_all_ready) {
        bool all_ready = true;
        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];
            auto & lc = g_layer_consts[il];
            if (lc.ready) continue;              // already done in a previous call
            if (layer.attn_lambda_q1 == nullptr) continue;  // not an attn layer

            const float lambda_init = 0.8f - 0.6f * expf(-0.3f * (float)il);
            float lambda_const = 0.0f;
            if (!compute_lambda_const(layer, lambda_const)) {
                all_ready = false;  // tensor->data not ready yet -- try again later
                continue;
            }

            lc.lambda_full = lambda_const + lambda_init;
            lc.scale_subln = 1.0f - lambda_init;
            lc.ready       = true;
        }
        if (all_ready) g_consts_all_ready = true;
        g_consts_initialized = true;
    }

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    auto * inp      = build_inp_mem_hybrid_iswa();
    auto * inp_attn = inp->get_attn();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * ssm_y_cache = nullptr;
    ggml_tensor * yoco_K      = nullptr;
    ggml_tensor * yoco_V_meta = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const bool is_mamba     = (il % 2 == 0);
        const bool is_gmu       = (il >= 18 && is_mamba);
        const bool is_full_attn = (il == 17);
        const bool is_cross     = (!is_mamba && il > 17);
        const bool is_swa_attn  = (!is_mamba && il < 17);
        const bool is_attn      = is_swa_attn || is_full_attn || is_cross;

        ggml_tensor * residual = inpL;

        cur = build_norm(inpL,
                         model.layers[il].attn_norm,
                         model.layers[il].attn_norm_b,
                         LLM_NORM, il);
        cb(cur, "attn_norm", il);

        // =====================================================================
        // A. MAMBA LAYERS (even 0,2,...,16)
        // =====================================================================
        if (is_mamba && !is_gmu) {
            if (il == 16) {
                ggml_tensor * y_pre = nullptr;
                cur = build_mamba_layer(inp->get_recr(), cur, model, ubatch, il, &y_pre);
                ssm_y_cache = ggml_dup(ctx0, y_pre ? y_pre : cur);
                cb(ssm_y_cache, "ssm_y_cache", il);
            } else {
                cur = build_mamba_layer(inp->get_recr(), cur, model, ubatch, il);
            }
        }

        // =====================================================================
        // B. GMU LAYERS (even 18,20,...,30)
        // =====================================================================
        else if (is_gmu) {
            if (ssm_y_cache != nullptr) {
                ggml_tensor * gate = build_lora_mm(model.layers[il].gmu_in, cur);
                gate = ggml_silu(ctx0, gate);
                cur = build_lora_mm(model.layers[il].gmu_out,
                                    ggml_mul(ctx0, ssm_y_cache, gate));
                cb(cur, "gmu_out", il);
            }
        }

        // =====================================================================
        // C. ATTENTION LAYERS (SWA: 1-15 | Full: 17 | Cross: 19,21,...,31)
        // =====================================================================
        else if (is_attn) {

            const float lf = 0.8f - 0.6f * expf(-0.3f * (float)il);

            ggml_tensor * attn1 = nullptr;
            ggml_tensor * attn2 = nullptr;

            if (is_cross) {
                GGML_ASSERT(yoco_K != nullptr && yoco_V_meta != nullptr);

                ggml_tensor * q_proj = build_lora_mm(model.layers[il].wqkv, cur);
                if (model.layers[il].bqkv) {
                    q_proj = ggml_add(ctx0, q_proj, model.layers[il].bqkv);
                }
                const size_t q_esz = ggml_element_size(q_proj);

                ggml_tensor * Q1 = ggml_view_3d(ctx0, q_proj,
                    n_embd_head_k, half_heads, n_tokens,
                    n_embd_head_k * q_esz, q_proj->nb[1], (size_t)0);
                ggml_tensor * Q2 = ggml_view_3d(ctx0, q_proj,
                    n_embd_head_k, half_heads, n_tokens,
                    n_embd_head_k * q_esz, q_proj->nb[1],
                    half_heads * n_embd_head_k * q_esz);

                // yoco_K: [64, 20, n_kv] contiguous.
                // nb[1]=head stride, nb[2]=token stride -- use nb[2] for K1/K2.
                const int64_t n_kv_yoco = yoco_K->ne[2];
                const size_t  k_esz     = ggml_element_size(yoco_K);

                ggml_tensor * K1 = ggml_view_3d(ctx0, yoco_K,
                    n_embd_head_k, half_kv_heads, n_kv_yoco,
                    n_embd_head_k * k_esz, yoco_K->nb[2], (size_t)0);
                ggml_tensor * K2 = ggml_view_3d(ctx0, yoco_K,
                    n_embd_head_k, half_kv_heads, n_kv_yoco,
                    n_embd_head_k * k_esz, yoco_K->nb[2],
                    half_kv_heads * n_embd_head_k * k_esz);

                ggml_tensor * cross_mask = get_kq_mask_iswa(inp_attn, il);

                attn1 = build_attn_mha(Q1, K1, yoco_V_meta,
                                       nullptr, cross_mask, nullptr, nullptr,
                                       attn_scale, il);
                attn2 = build_attn_mha(Q2, K2, yoco_V_meta,
                                       nullptr, cross_mask, nullptr, nullptr,
                                       attn_scale, il);

            } else {
                ggml_tensor * qkv = build_lora_mm(model.layers[il].wqkv, cur);
                if (model.layers[il].bqkv) {
                    qkv = ggml_add(ctx0, qkv, model.layers[il].bqkv);
                }

                const int64_t q_total  = n_head    * n_embd_head_k;
                const int64_t kv_total = n_head_kv * n_embd_head_k;
                const size_t  esz      = ggml_element_size(qkv);

                // qkv->nb[1] = token stride (5120*esz) -- correct for Q/K/V views.
                ggml_tensor * Q1 = ggml_view_3d(ctx0, qkv,
                    n_embd_head_k, half_heads, n_tokens,
                    n_embd_head_k * esz, qkv->nb[1], (size_t)0);
                ggml_tensor * Q2 = ggml_view_3d(ctx0, qkv,
                    n_embd_head_k, half_heads, n_tokens,
                    n_embd_head_k * esz, qkv->nb[1],
                    half_heads * n_embd_head_k * esz);

                ggml_tensor * K_full = ggml_view_3d(ctx0, qkv,
                    n_embd_head_k, n_head_kv, n_tokens,
                    n_embd_head_k * esz, qkv->nb[1], q_total * esz);
                ggml_tensor * V_raw = ggml_view_3d(ctx0, qkv,
                    n_embd_head_k, n_head_kv, n_tokens,
                    n_embd_head_k * esz, qkv->nb[1], (q_total + kv_total) * esz);

                ggml_build_forward_expand(gf, Q1);
                ggml_build_forward_expand(gf, Q2);
                ggml_build_forward_expand(gf, K_full);
                ggml_build_forward_expand(gf, V_raw);

                write_kv_iswa(inp_attn, K_full, V_raw, il);

                ggml_tensor * K_cached = read_k_iswa(inp_attn, il);
                ggml_tensor * V_cached = read_v_iswa(inp_attn, il);

                const int64_t n_kv = K_cached->ne[2];

                ggml_tensor * V_meta = ggml_reshape_3d(ctx0,
                    ggml_cont(ctx0, V_cached),
                    meta_dim, half_kv_heads, n_kv);

                // K_cached [64, 20, n_kv] contiguous.
                // nb[1]=head stride, nb[2]=token stride -- use nb[2] for K1/K2.
                const size_t k_esz = ggml_element_size(K_cached);
                ggml_tensor * K1 = ggml_view_3d(ctx0, K_cached,
                    n_embd_head_k, half_kv_heads, n_kv,
                    n_embd_head_k * k_esz, K_cached->nb[2], (size_t)0);
                ggml_tensor * K2 = ggml_view_3d(ctx0, K_cached,
                    n_embd_head_k, half_kv_heads, n_kv,
                    n_embd_head_k * k_esz, K_cached->nb[2],
                    half_kv_heads * n_embd_head_k * k_esz);

                ggml_tensor * kq_mask_nc = const_cast<ggml_tensor *>(
                    get_kq_mask_iswa(inp_attn, il));

                attn1 = build_attn_mha(Q1, K1, V_meta,
                                       nullptr, kq_mask_nc, nullptr, nullptr,
                                       attn_scale, il);
                attn2 = build_attn_mha(Q2, K2, V_meta,
                                       nullptr, kq_mask_nc, nullptr, nullptr,
                                       attn_scale, il);

                if (is_full_attn) {
                    yoco_K      = ggml_dup(ctx0, K_cached);
                    yoco_V_meta = ggml_dup(ctx0, V_meta);
                    cb(yoco_K,      "yoco_K",      il);
                    cb(yoco_V_meta, "yoco_V_meta", il);
                }
            }

            // ---- Lambda + differential subtraction ----
            // OPT 1: use pre-computed scalar lambda_full (avoids 6 tensor ops per layer).
            // Falls back to in-graph computation if the lambda tensors were quantized
            // or could not be read (g_layer_consts[il].ready == false).
            ggml_tensor * diff;
            if (g_layer_consts[il].ready) {
                // Fast path: scalar scale -- no graph ops for lambda computation.
                diff = ggml_sub(ctx0, attn1,
                    ggml_scale(ctx0, attn2, g_layer_consts[il].lambda_full));
            } else {
                // Fallback: full in-graph lambda computation (original approach).
                const float lf_fb = lf;
                if (model.layers[il].attn_lambda_q1 != nullptr) {
                    ggml_tensor * e1 = ggml_exp(ctx0, ggml_sum(ctx0, ggml_mul(ctx0,
                        model.layers[il].attn_lambda_q1,
                        model.layers[il].attn_lambda_k1)));
                    ggml_tensor * e2 = ggml_exp(ctx0, ggml_sum(ctx0, ggml_mul(ctx0,
                        model.layers[il].attn_lambda_q2,
                        model.layers[il].attn_lambda_k2)));
                    ggml_tensor * e_diff    = ggml_sub(ctx0, e1, e2);
                    ggml_tensor * e_diff_3d = ggml_reshape_3d(ctx0, e_diff, 1, 1, 1);
                    ggml_tensor * noise_learned = ggml_mul(ctx0, attn2,
                        ggml_repeat(ctx0, e_diff_3d, attn2));
                    ggml_tensor * noise_init = ggml_scale(ctx0, attn2, lf_fb);
                    diff = ggml_sub(ctx0, attn1,
                        ggml_add(ctx0, noise_learned, noise_init));
                } else {
                    diff = ggml_sub(ctx0, attn1, ggml_scale(ctx0, attn2, lf_fb));
                }
            }
            cb(diff, "diff_attn", il);

            // ---- SubLN: RMSNorm then scale by (subln_w * (1 - lambda_init)) ----
            // OPT 2: scale the tiny subln_w [128] rather than the large normed [128,20,T].
            ggml_tensor * normed = ggml_rms_norm(ctx0, diff, norm_eps);
            if (model.layers[il].attn_subln != nullptr) {
                ggml_tensor * sw = ggml_reshape_3d(ctx0,
                    model.layers[il].attn_subln, meta_dim, 1, 1);
                // Apply (1-lambda_init) to the 128-element weight, not to normed.
                float s = g_layer_consts[il].ready ? g_layer_consts[il].scale_subln
                                                   : (1.0f - lf);
                ggml_tensor * sw_scaled = ggml_scale(ctx0, sw, s);
                normed = ggml_mul(ctx0, normed, ggml_repeat(ctx0, sw_scaled, normed));
            }
            cb(normed, "subln_out", il);

            // ---- Re-interleave: [128, 20, T] -> [2560, T] ----
            // View [128,20,T] as [64,2,20,T] with natural contiguous strides,
            // flatten to [2560,T] -- correct interleaved layout, no data copy.
            ggml_tensor * nc    = ggml_cont(ctx0, normed);
            const size_t nc_esz = ggml_element_size(nc);

            ggml_tensor * v4d = ggml_view_4d(ctx0, nc,
                n_embd_head_k,
                2,
                half_heads,
                n_tokens,
                n_embd_head_k * nc_esz,
                meta_dim      * nc_esz,
                meta_dim * half_heads * nc_esz,
                0);

            ggml_tensor * kqv_flat = ggml_reshape_2d(ctx0, v4d, n_embd, n_tokens);
            cb(kqv_flat, "kqv_flat", il);

            cur = build_lora_mm(model.layers[il].wo, kqv_flat);
            if (model.layers[il].bo) {
                cur = ggml_add(ctx0, cur, model.layers[il].bo);
            }
            cb(cur, "attn_out", il);
        }

        // =====================================================================
        // D. RESIDUAL + POST-ATTN NORM + SwiGLU FFN
        // =====================================================================
        if (il == n_layer - 1 && inp_out_ids) {
            cur      = ggml_get_rows(ctx0, cur, inp_out_ids);
            residual = ggml_get_rows(ctx0, residual, inp_out_ids);
        }

        cur = ggml_add(ctx0, cur, residual);
        residual = cur;

        if (model.layers[il].ffn_up && model.layers[il].ffn_down) {
            ggml_tensor * ffn_inp = build_norm(cur,
                                               model.layers[il].ffn_norm,
                                               model.layers[il].ffn_norm_b,
                                               LLM_NORM, il);
            cb(ffn_inp, "ffn_norm", il);

            cur = build_ffn(ffn_inp,
                            model.layers[il].ffn_up,    nullptr, nullptr,
                            nullptr,                    nullptr, nullptr,
                            model.layers[il].ffn_down,  nullptr, nullptr,
                            nullptr,
                            LLM_FFN_SWIGLU, LLM_FFN_SEQ, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, residual, cur);
        }

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
        inpL = cur;
    }

    // Final LayerNorm (has bias -- LLM_NORM not LLM_NORM_RMS)
    cur = build_norm(inpL,
                     model.output_norm, model.output_norm_b,
                     LLM_NORM, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}