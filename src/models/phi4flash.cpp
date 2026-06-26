#include "models.h"
#include "../llama-kv-cache-iswa.h"
#include <cmath>

// Phi-4-mini-flash-reasoning (SambaY-DA architecture)
//
// Layer map:
//  even 0,2,...,16  : Mamba-1
//  odd  1,3,...,15  : SWA differential attention  (512-token window)
//  17               : Full differential attention  (YOCO pivot: saves K,V)
//  even 18,20,...,30: GMU  (Gated Memory Unit)
//  odd  19,21,...,31: Cross differential attention (YOCO K,V from layer 17)
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
//   diff   = attn1 - lambda * attn2                  [128, 20, T]
//   diff   = subln(diff)   (RMSNorm + learned scale [128])
//   diff   = diff * (1 - lambda_init)
//   out    = rearrange(diff, "H (two D) -> (H two) D")  [64, 40, T]
//   out    = wo(flatten(out))
//
// === KV CACHE ===
//   We use four helper methods added to llm_graph_context:
//     write_kv_iswa(inp, K, V, il)      -- writes K[64,20,T] and V[64,20,T] to cache
//     read_k_iswa(inp, il)              -- reads K[64,20,n_kv] from cache
//     read_v_iswa(inp, il)              -- reads V[64,20,n_kv] from cache
//     get_kq_mask_iswa(inp, il)         -- gets causal / SWA mask tensor
//   After reading V we reshape [64,20,n_kv] -> [128,10,n_kv] (same bytes, no copy).
//   Then build_attn_mha is called twice: Q1/K1/V_meta and Q2/K2/V_meta.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// llama_model_phi4flash  --  model-level methods
// ---------------------------------------------------------------------------

void llama_model_phi4flash::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // Phi-4-mini-flash-reasoning is a fixed 32-layer hybrid architecture.
    switch (hparams.n_layer()) {
        case 32: type = LLM_TYPE_3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }

    // Load Mamba SSM parameters from GGUF metadata, providing standard fallbacks
    if (!ml.get_key(LLM_KV_SSM_CONV_KERNEL, hparams.ssm_d_conv, false)) {
        hparams.ssm_d_conv = 4;
    }
    if (!ml.get_key(LLM_KV_SSM_STATE_SIZE, hparams.ssm_d_state, false)) {
        hparams.ssm_d_state = 16; // Standard Mamba state size is 16
    }
    if (!ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank, false)) {
        hparams.ssm_dt_rank = 160; // Standard dt_rank for Phi-4-mini-flash-reasoning
    }

    // Load d_inner explicitly - Resolves Problem 4 (ssm_d_inner = 0)
    if (!ml.get_key(LLM_KV_SSM_INNER_SIZE, hparams.ssm_d_inner, false)) {
        hparams.ssm_d_inner = 5120; // 2 * n_embd for this architecture
    }

    // Also load the standard LayerNorm epsilon (used by the final output norm)
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps, false);

    // SWA: odd layers 1,3,5,...,15 use a 512-token sliding window.
    // Even layers are Mamba/GMU (no KV) and must NOT be marked as SWA.
    uint32_t swa_window = 512;
    if (!ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, swa_window, false)) {
        swa_window = 512;
    }
    hparams.n_swa    = swa_window;
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;

    // SWA pattern: set_swa_pattern(2, dense_first=true) gives:
    //   il%2==0 -> dense, il%2==1 -> SWA
    // Then override layers 17+ to dense (pivot, GMU, cross-attn are not SWA).
    hparams.set_swa_pattern(2, /*dense_first=*/true);
    for (uint32_t il = 17; il < hparams.n_layer(); ++il) {
        hparams.is_swa_impl[il] = 0;
    }

    // Recurrent pattern: even layers 0,2,...,16 are Mamba (recurrent).
    // set_recr_pattern(2, dense_first=false) gives il%2==0 -> recurrent,
    // il%2==1 -> non-recurrent, which is exactly right for layers 0-16.
    // Then override layers 17+ to non-recurrent (pivot/GMU/cross have no SSM state).
    hparams.set_recr_pattern(2, /*dense_first=*/false);
    for (uint32_t il = 17; il < hparams.n_layer(); ++il) {
        hparams.is_recr_impl[il] = 0;
    }

    // Per-layer head counts — drives KV cache slot allocation.
    //   even layers 0-16 (Mamba):  n_head=0, n_head_kv=0  -> recurrent memory
    //   odd layers 1-15 (SWA attn):n_head=40, n_head_kv=20 -> SWA KV cache
    //   layer 17 (full attn):      n_head=40, n_head_kv=20 -> base KV cache
    //   even layers 18-30 (GMU):   n_head=0, n_head_kv=0  -> neither cache
    //   odd layers 19-31 (cross):  n_head=40, n_head_kv=20 -> base KV cache
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        const bool is_mamba     = (il % 2 == 0) && (il <= 16);
        const bool is_gmu       = (il % 2 == 0) && (il >= 18);
        const bool is_full_attn = (il == 17);
        const bool is_cross     = (il % 2 == 1) && (il > 17);
        const bool is_swa_attn  = (il % 2 == 1) && (il < 17);
        const bool is_attn      = is_swa_attn || is_full_attn || is_cross;

        (void)is_mamba; (void)is_gmu; // used implicitly via is_recr_impl

        if (is_attn) {
            hparams.n_head_arr[il]    = 40;
            hparams.n_head_kv_arr[il] = 20;
        } else {
            hparams.n_head_arr[il]    = 0;
            hparams.n_head_kv_arr[il] = 0;
        }
    }
}

void llama_model_phi4flash::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    // Phi-4-mini-flash uses a LayerNorm bias on the output norm.
    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), { n_embd }, TENSOR_NOT_REQUIRED);
    output        = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const bool is_mamba     = (i % 2 == 0);
        const bool is_gmu       = (i >= 18 && is_mamba);
        const bool is_full_attn = (i == 17);
        const bool is_cross     = (!is_mamba && i > 17);
        const bool is_swa_attn  = (!is_mamba && i < 17);
        const bool is_attn      = is_swa_attn || is_full_attn || is_cross;

        // Pre-norm is shared by all layer types.
        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM,  "weight", i), { n_embd }, 0);
        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM,  "bias",   i), { n_embd }, TENSOR_NOT_REQUIRED);

        if (is_mamba && !is_gmu) {
            // Use actual loaded hparam size instead of hardcoded multiplier
            const int64_t d_inner = hparams.ssm_d_inner;

            // Mamba-1 tensors aligned with correct d_inner dimensions
            layer.ssm_in      = create_tensor(tn(LLM_TENSOR_SSM_IN,     "weight", i), { n_embd, 2 * d_inner }, 0);
            layer.ssm_conv1d  = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), { hparams.ssm_d_conv, d_inner }, 0);
            
            // Explicitly load 1D Convolution Bias (This fills the missing 9 tensors!)
            layer.ssm_conv1d_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "bias",   i), { d_inner }, TENSOR_NOT_REQUIRED);

            layer.ssm_x       = create_tensor(tn(LLM_TENSOR_SSM_X,      "weight", i), { d_inner, hparams.ssm_dt_rank + 2 * hparams.ssm_d_state }, 0);
            layer.ssm_dt      = create_tensor(tn(LLM_TENSOR_SSM_DT,     "weight", i), { hparams.ssm_dt_rank, d_inner }, 0);
            layer.ssm_dt_b    = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   i), { d_inner }, TENSOR_NOT_REQUIRED);

            // Attempt to load ssm_a with suffix "weight", fallback to no suffix (using nullptr to avoid appending a dot)
            layer.ssm_a       = create_tensor(tn(LLM_TENSOR_SSM_A,      "weight", i), { hparams.ssm_d_state, d_inner }, TENSOR_NOT_REQUIRED);
            if (layer.ssm_a == NULL) {
                layer.ssm_a   = create_tensor(tn(LLM_TENSOR_SSM_A,      nullptr, i),  { hparams.ssm_d_state, d_inner }, 0);
            }

            // Attempt to load ssm_d with suffix "weight", fallback to no suffix (using nullptr to avoid appending a dot)
            layer.ssm_d       = create_tensor(tn(LLM_TENSOR_SSM_D,      "weight", i), { d_inner }, TENSOR_NOT_REQUIRED);
            if (layer.ssm_d == NULL) {
                layer.ssm_d   = create_tensor(tn(LLM_TENSOR_SSM_D,      nullptr, i),  { d_inner }, 0);
            }

            layer.ssm_out     = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", i), { d_inner, n_embd }, 0);
        } else if (is_gmu) {
            // gmu_in  [n_embd, 2*n_embd]: projects to val+gate (matches GGUF [2560,5120])
            // gmu_out [n_embd, n_embd]:   projects gated value back (matches GGUF [5120,2560])
            layer.gmu_in  = create_tensor(tn(LLM_TENSOR_GMU_IN,  "weight", i), { n_embd, 2 * n_embd }, 0);
            layer.gmu_out = create_tensor(tn(LLM_TENSOR_GMU_OUT, "weight", i), { 2 * n_embd, n_embd }, 0);
        } else if (is_attn) {
            // Differential attention (SWA, full, or cross).
            if (is_cross) {
                // Cross attention layers only project Q. K and V are reused from YOCO pivot Layer 17.
                layer.wqkv   = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), { n_embd, hparams.n_head(i) * hparams.n_embd_head_k() }, 0);
                layer.wqkv_b = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias",   i), { hparams.n_head(i) * hparams.n_embd_head_k() }, TENSOR_NOT_REQUIRED);
            } else {
                // Standard and SWA attention layers project Q, K, and V.
                layer.wqkv   = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), { n_embd, (hparams.n_head(i) + 2 * hparams.n_head_kv(i)) * hparams.n_embd_head_k() }, 0);
                layer.wqkv_b = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias",   i), { (hparams.n_head(i) + 2 * hparams.n_head_kv(i)) * hparams.n_embd_head_k() }, TENSOR_NOT_REQUIRED);
            }
            layer.wo     = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", i), { n_embd, n_embd }, 0);
            layer.wo_b   = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "bias",   i), { n_embd }, TENSOR_NOT_REQUIRED);

            // Differential attention lambda weights.
            layer.attn_lambda_q1 = create_tensor(tn(LLM_TENSOR_ATTN_LAMBDA_Q1, "weight", i), { hparams.n_embd_head_k() }, 0);
            layer.attn_lambda_k1 = create_tensor(tn(LLM_TENSOR_ATTN_LAMBDA_K1, "weight", i), { hparams.n_embd_head_k() }, 0);
            layer.attn_lambda_q2 = create_tensor(tn(LLM_TENSOR_ATTN_LAMBDA_Q2, "weight", i), { hparams.n_embd_head_k() }, 0);
            layer.attn_lambda_k2 = create_tensor(tn(LLM_TENSOR_ATTN_LAMBDA_K2, "weight", i), { hparams.n_embd_head_k() }, 0);

            // SubLN scale (per-channel, dim = 2 * n_embd_head_k).
            layer.attn_subln = create_tensor(tn(LLM_TENSOR_ATTN_SUBLN, "weight", i), { 2 * hparams.n_embd_head_k() }, 0);
        }

        // FFN is present for all layer types that have an ffn_up tensor.
        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, TENSOR_NOT_REQUIRED);
        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias",   i), { n_embd }, TENSOR_NOT_REQUIRED);
        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, 2 * n_ff }, TENSOR_NOT_REQUIRED);
        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd    }, TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_phi4flash::build_arch_graph(const llm_graph_params & params) const {
    // This model always uses the iSWA memory context (hybrid Mamba + SWA attn).
    // The <true> path in the graph constructor selects iSWA KV helpers.
    return std::make_unique<graph>(*this, params);
}

// ---------------------------------------------------------------------------
// graph  --  per-token inference
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// KV-cache helpers (member functions of llama_model_phi4flash::graph)
// ---------------------------------------------------------------------------

// is_swa_layer: true for odd attention layers 1,3,...,15 (sliding window attn).
// We cannot use hparams.is_swa(il) because set_swa_pattern(2) marks EVEN
// layers as SWA which is inverted for this architecture.
static inline bool is_swa_layer(int il) {
    return (il % 2 == 1) && (il < 17);
}

void llama_model_phi4flash::graph::write_kv_iswa(
        llm_graph_input_attn_kv_iswa * inp,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        int il) {
    const auto * mctx_iswa = inp->mctx;
    const bool   is_swa    = is_swa_layer(il);
    const auto * mctx_cur  = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();
    const auto & k_idxs    = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();
    const auto & v_idxs    = is_swa ? inp->get_v_idxs_swa() : inp->get_v_idxs();
    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
}

ggml_tensor * llama_model_phi4flash::graph::read_k_iswa(
        llm_graph_input_attn_kv_iswa * inp,
        int il) {
    const auto * mctx_iswa = inp->mctx;
    const bool   is_swa    = is_swa_layer(il);
    const auto * mctx_cur  = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();
    return mctx_cur->get_k(ctx0, il);
}

ggml_tensor * llama_model_phi4flash::graph::read_v_iswa(
        llm_graph_input_attn_kv_iswa * inp,
        int il) {
    const auto * mctx_iswa = inp->mctx;
    const bool   is_swa    = is_swa_layer(il);
    const auto * mctx_cur  = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();
    return mctx_cur->get_v(ctx0, il);
}

ggml_tensor * llama_model_phi4flash::graph::get_kq_mask_iswa(
        llm_graph_input_attn_kv_iswa * inp,
        int il) {
    return is_swa_layer(il) ? inp->get_kq_mask_swa() : inp->get_kq_mask();
}

// ---------------------------------------------------------------------------
// graph constructor  --  replaces old llm_build_phi4flash constructor
// ---------------------------------------------------------------------------

llama_model_phi4flash::graph::graph(const llama_model & model, const llm_graph_params & params)
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

        // FIXED: Reverted strictly to LLM_NORM as specified by working old codebase
        cur = build_norm(inpL,
                         model.layers[il].attn_norm,
                         model.layers[il].attn_norm_b,
                         LLM_NORM, il);
        cb(cur, "attn_norm", il);

        // =====================================================================
        // A. MAMBA LAYERS
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
        // B. GMU LAYERS
        // =====================================================================
        else if (is_gmu) {
            if (ssm_y_cache != nullptr) {
                // FIXED: Robust, shape-aligned double gating logic mapped cleanly 
                // matching the exact GGUF [5120, 2560] gmu_out bounds
                ggml_tensor * gate = build_lora_mm(model.layers[il].gmu_in, cur);
                gate = ggml_silu(ctx0, gate); // [5120, T]
                
                ggml_tensor * ssm_y_repeated = ggml_repeat(ctx0, ssm_y_cache, gate); // [5120, T]
                ggml_tensor * gated = ggml_mul(ctx0, ssm_y_repeated, gate); // [5120, T]
                
                cur = build_lora_mm(model.layers[il].gmu_out, gated);
                cb(cur, "gmu_out", il);
            }
        }

        // =====================================================================
        // C. ATTENTION LAYERS
        // =====================================================================
        else if (is_attn) {
            ggml_tensor * attn1 = nullptr;
            ggml_tensor * attn2 = nullptr;

            if (is_cross) {
                ggml_tensor * q_proj = build_lora_mm(model.layers[il].wqkv, cur);
                if (model.layers[il].wqkv_b) {
                    q_proj = ggml_add(ctx0, q_proj, model.layers[il].wqkv_b);
                }
                const size_t q_esz = ggml_element_size(q_proj);

                ggml_tensor * Q1 = ggml_view_3d(ctx0, q_proj,
                    n_embd_head_k, half_heads, n_tokens,
                    n_embd_head_k * q_esz, q_proj->nb[1], (size_t)0);
                ggml_tensor * Q2 = ggml_view_3d(ctx0, q_proj,
                    n_embd_head_k, half_heads, n_tokens,
                    n_embd_head_k * q_esz, q_proj->nb[1],
                    half_heads * n_embd_head_k * q_esz);

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
                if (model.layers[il].wqkv_b) {
                    qkv = ggml_add(ctx0, qkv, model.layers[il].wqkv_b);
                }

                const int64_t q_total  = n_head    * n_embd_head_k;
                const int64_t kv_total = n_head_kv * n_embd_head_k;
                const size_t  esz      = ggml_element_size(qkv);

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
            const float lf = 0.8f - 0.6f * expf(-0.3f * (float)il);
            ggml_tensor * diff;

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
                ggml_tensor * noise_init = ggml_scale(ctx0, attn2, lf);
                
                diff = ggml_sub(ctx0, attn1,
                    ggml_add(ctx0, noise_learned, noise_init));
            } else {
                diff = ggml_sub(ctx0, attn1, ggml_scale(ctx0, attn2, lf));
            }
            cb(diff, "diff_attn", il);

            // ---- SubLN: RMSNorm then scale by (subln_w * (1 - lambda_init)) ----
            ggml_tensor * normed = ggml_rms_norm(ctx0, diff, norm_eps);
            if (model.layers[il].attn_subln != nullptr) {
                ggml_tensor * sw = ggml_reshape_3d(ctx0,
                    model.layers[il].attn_subln, meta_dim, 1, 1);
                
                float s = (1.0f - lf);
                ggml_tensor * sw_scaled = ggml_scale(ctx0, sw, s);
                normed = ggml_mul(ctx0, normed, ggml_repeat(ctx0, sw_scaled, normed));
            }
            cb(normed, "subln_out", il);

            // ---- Re-interleave: [128, 20, T] -> [2560, T] ----
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
            if (model.layers[il].wo_b) {
                cur = ggml_add(ctx0, cur, model.layers[il].wo_b);
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
            // FIXED: Reverted strictly back to LLM_NORM as specified by working old codebase
            ggml_tensor * ffn_inp = build_norm(cur,
                                               model.layers[il].ffn_norm,
                                               model.layers[il].ffn_norm_b,
                                               LLM_NORM, il);
            cb(ffn_inp, "ffn_norm", il);

            cur = build_ffn(ffn_inp,
                            model.layers[il].ffn_up,   nullptr, nullptr,
                            nullptr,                   nullptr, nullptr,
                            model.layers[il].ffn_down, nullptr, nullptr,
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