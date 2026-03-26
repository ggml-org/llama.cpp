#include "models.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"

#include <algorithm>
#include <set>
#include <string>

static ggml_tensor * build_kq_mask(
        ggml_context * ctx,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const auto n_kv     = mctx->get_n_kv();
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    return ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, n_tokens/n_stream, 1, n_stream);
}

// Custom input that extends ISWA input with extra SWA masks for different window sizes
class llm_graph_input_puzzle_iswa : public llm_graph_input_attn_kv_iswa {
public:
    using llm_graph_input_attn_kv_iswa::llm_graph_input_attn_kv_iswa;

    struct extra_mask {
        uint32_t       n_swa;
        ggml_tensor * mask     = nullptr;
        ggml_tensor * mask_cnv = nullptr;
    };

    std::vector<extra_mask> extra_swa_masks;

    void set_input(const llama_ubatch * ubatch) override {
        // Fill base + default SWA masks
        llm_graph_input_attn_kv_iswa::set_input(ubatch);

        // Fill extra masks with their respective window sizes
        for (auto & m : extra_swa_masks) {
            if (m.mask) {
                mctx->get_swa()->set_input_kq_mask(m.mask, ubatch, cparams.causal_attn, m.n_swa);
            }
        }
    }

    ggml_tensor * get_kq_mask_for_swa(uint32_t ws) const {
        for (const auto & m : extra_swa_masks) {
            if (m.n_swa == ws) return m.mask_cnv;
        }
        // Fallback to default SWA mask
        return self_kq_mask_swa_cnv;
    }
};

llm_build_openai_moe_puzzle_iswa::llm_build_openai_moe_puzzle_iswa(
        const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    // Build ISWA input with extra SWA masks for per-layer windows
    const auto * mctx_cur = static_cast<const llama_kv_cache_iswa_context *>(mctx);

    auto inp_ptr = std::make_unique<llm_graph_input_puzzle_iswa>(hparams, cparams, mctx_cur);

    // Build base (full attention) KV input
    {
        inp_ptr->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp_ptr->self_v_idxs = mctx_cur->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp_ptr->self_kq_mask = build_kq_mask(ctx0, mctx_cur->get_base(), ubatch, cparams);
        ggml_set_input(inp_ptr->self_kq_mask);
        ggml_set_name(inp_ptr->self_kq_mask, "self_kq_mask");

        inp_ptr->self_kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp_ptr->self_kq_mask, GGML_TYPE_F16) : inp_ptr->self_kq_mask;
        ggml_set_name(inp_ptr->self_kq_mask_cnv, "self_kq_mask_cnv");
    }

    // Build SWA KV input with default (max) window mask
    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Puzzle model requires SWA");

        inp_ptr->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp_ptr->self_v_idxs_swa = mctx_cur->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp_ptr->self_kq_mask_swa = build_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
        ggml_set_input(inp_ptr->self_kq_mask_swa);
        ggml_set_name(inp_ptr->self_kq_mask_swa, "self_kq_mask_swa");

        inp_ptr->self_kq_mask_swa_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp_ptr->self_kq_mask_swa, GGML_TYPE_F16) : inp_ptr->self_kq_mask_swa;
        ggml_set_name(inp_ptr->self_kq_mask_swa_cnv, "self_kq_mask_swa_cnv");
    }

    // Create extra SWA mask tensors for unique smaller window sizes
    {
        std::set<uint32_t> unique_swa;
        for (int il = 0; il < n_layer; ++il) {
            uint32_t ws = hparams.n_swa_layer(il);
            if (ws > 0 && ws != hparams.n_swa) {
                unique_swa.insert(ws);
            }
        }

        for (uint32_t ws : unique_swa) {
            llm_graph_input_puzzle_iswa::extra_mask m;
            m.n_swa = ws;
            m.mask = ggml_dup_tensor(ctx0, inp_ptr->self_kq_mask_swa);
            ggml_set_input(m.mask);
            std::string name = "swa_mask_" + std::to_string(ws);
            ggml_set_name(m.mask, name.c_str());
            m.mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, m.mask, GGML_TYPE_F16) : m.mask;
            inp_ptr->extra_swa_masks.push_back(m);
        }
    }

    auto * inp_attn = (llm_graph_input_puzzle_iswa *) res->add_input(std::move(inp_ptr));

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);
            if (model.layers[il].bq) {
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                cb(Qcur, "Qcur", il);
            }
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);
            if (model.layers[il].bk) {
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                cb(Kcur, "Kcur", il);
            }
            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);
            if (model.layers[il].bv) {
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                cb(Vcur, "Vcur", il);
            }
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_rot, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_rot, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_rot, n_head_kv, n_tokens);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // Manual ISWA attention with per-layer mask selection
            ggml_build_forward_expand(gf, Qcur);
            if (Kcur) ggml_build_forward_expand(gf, Kcur);
            if (Vcur) ggml_build_forward_expand(gf, Vcur);

            const uint32_t ws = hparams.n_swa_layer(il);
            const bool is_swa = ws > 0;

            const auto * kv_ctx = is_swa ? mctx_cur->get_swa() : mctx_cur->get_base();

            // Store K/V to cache
            {
                const auto & k_idxs = is_swa ? inp_attn->get_k_idxs_swa() : inp_attn->get_k_idxs();
                const auto & v_idxs = is_swa ? inp_attn->get_v_idxs_swa() : inp_attn->get_v_idxs();
                ggml_build_forward_expand(gf, kv_ctx->cpy_k(ctx0, Kcur, k_idxs, il));
                ggml_build_forward_expand(gf, kv_ctx->cpy_v(ctx0, Vcur, v_idxs, il));
            }

            // Select the correct mask for this layer's window size
            ggml_tensor * kq_mask;
            if (!is_swa) {
                kq_mask = inp_attn->get_kq_mask();
            } else if (ws == hparams.n_swa) {
                kq_mask = inp_attn->get_kq_mask_swa();
            } else {
                kq_mask = inp_attn->get_kq_mask_for_swa(ws);
            }

            ggml_tensor * k = kv_ctx->get_k(ctx0, il);
            ggml_tensor * v = kv_ctx->get_v(ctx0, il);

            cur = build_attn_mha(Qcur, k, v, nullptr, kq_mask, model.layers[il].attn_sinks, nullptr, 1.0f/sqrtf(float(n_rot)), il);
            cb(cur, "kqv_out", il);

            if (model.layers[il].wo) {
                cur = build_lora_mm(model.layers[il].wo, cur);
            }
            if (model.layers[il].bo) {
                cur = ggml_add(ctx0, cur, model.layers[il].bo);
            }
        }

        cb(cur, "attn_out", il);

        if (il == n_layer - 1) {
            // skip computing output for unused tokens
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = ffn_inp;
        cur = build_norm(cur,
                model.layers[il].attn_post_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        // MoE branch - use PER-LAYER expert count
        // Cap n_expert_used to per-layer count (during warmup, n_expert_used == global max which may exceed smaller layers)
        const int64_t n_expert_il = hparams.n_expert_layer(il);
        const int64_t n_expert_used_il = std::min(n_expert_used, n_expert_il);
        cur = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,  model.layers[il].ffn_gate_inp_b,
                model.layers[il].ffn_up_exps,   model.layers[il].ffn_up_exps_b,
                model.layers[il].ffn_gate_exps, model.layers[il].ffn_gate_exps_b,
                model.layers[il].ffn_down_exps, model.layers[il].ffn_down_exps_b,
                nullptr,
                n_expert_il, n_expert_used_il,
                LLM_FFN_SWIGLU_OAI_MOE, false,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT,
                il);
        cb(cur, "ffn_moe_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
