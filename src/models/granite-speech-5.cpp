#include "models.h"

#include <algorithm>
#include <map>

// granite_speech_5: a CTC (Connectionist Temporal Classification) acoustic model. Non-causal,
// no KV-cache (see create_memory() in llama-model.cpp), single forward pass - llama_decode()
// auto-redirects to the non-causal encode() path since this arch has no memory. Input is raw
// per-frame features injected via the standard .embd path (fed by a paired, front-end-only
// mmproj, see conversion/granite.py's GraniteSpeech5FrontendMmprojModel) rather than token
// ids. Output (res->t_logits) is per-frame CTC vocab logits, not next-token logits.
//
// The conformer body (Shaw relative-position block attention, GLU conv module via
// ggml_ssm_conv, half-step FFNs, mid-stack CTC self-conditioning) mirrors
// tools/mtmd/models/granite-speech.cpp's existing ggml graph almost verbatim. The one
// genuinely new piece is time subsampling (hparams.ctc.subsample_layers): a stride-2
// depthwise conv plus a mean-pooled residual halve the sequence length at each configured
// layer, so num_blocks/padded_len/remainder must be recomputed per-layer instead of once.

// constant (not per-request) graph inputs: computed once at build time from hparams alone,
// written into their backend tensor by set_input() regardless of the ubatch passed in.
struct llm_graph_input_ctc_const_i32 : public llm_graph_input_i {
    llm_graph_input_ctc_const_i32(ggml_tensor * t, std::vector<int32_t> data) : t(t), data(std::move(data)) {}
    void set_input(const llama_ubatch *) override {
        ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(int32_t));
    }
    ggml_tensor * t;
    std::vector<int32_t> data;
};

struct llm_graph_input_ctc_const_f32 : public llm_graph_input_i {
    llm_graph_input_ctc_const_f32(ggml_tensor * t, std::vector<float> data) : t(t), data(std::move(data)) {}
    void set_input(const llama_ubatch *) override {
        ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
    }
    ggml_tensor * t;
    std::vector<float> data;
};

void llama_model_granite_speech_5::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

    ml.get_key(LLM_KV_CTC_CONTEXT_SIZE,          hparams.ctc.context_size);
    ml.get_key(LLM_KV_CTC_MAX_POS_EMB,           hparams.ctc.max_pos_emb);
    ml.get_key(LLM_KV_CTC_CONV_KERNEL,           hparams.ctc.conv_kernel);
    ml.get_key(LLM_KV_CTC_CONV_EXPANSION_FACTOR, hparams.ctc.conv_expansion_factor);
    ml.get_arr(LLM_KV_CTC_SUBSAMPLE_LAYERS,      hparams.subsample_factor_impl, false);
}

void llama_model_granite_speech_5::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    // this arch is only ever driven via raw feature (.embd) input, never token ids, but
    // build_inp_embd() requires a valid tok_embd tensor to exist (ggml_get_rows(tok_embd, ...)
    // is built into the graph even though the token-lookup branch is never selected at
    // runtime) - the converter writes a single dummy row, just enough to satisfy the
    // tensor-shape assert without wasting space on an unused vocab table
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, 1}, 0);

    const int64_t inner_dim = n_embd * hparams.ctc.conv_expansion_factor;
    const int64_t max_dist  = 2 * (int64_t) hparams.ctc.max_pos_emb + 1;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i), {n_embd}, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd}, 0);

        layer.attn_rel_pos = create_tensor(tn(LLM_TENSOR_ATTN_REL_POS, "weight", i), {n_embd_head_k, max_dist}, 0);

        layer.conv_norm   = create_tensor(tn(LLM_TENSOR_CONV_NORM, "weight", i), {n_embd}, 0);
        layer.conv_norm_b = create_tensor(tn(LLM_TENSOR_CONV_NORM, "bias",   i), {n_embd}, 0);

        layer.conv_pw1   = create_tensor(tn(LLM_TENSOR_CONV_PW1, "weight", i), {n_embd, 2 * inner_dim}, 0);
        layer.conv_pw1_b = create_tensor(tn(LLM_TENSOR_CONV_PW1, "bias",   i), {2 * inner_dim}, 0);

        layer.conv_dw = create_tensor(tn(LLM_TENSOR_CONV_DW, "weight", i), {(int64_t) hparams.ctc.conv_kernel, inner_dim}, 0);

        layer.conv_dw_norm   = create_tensor(tn(LLM_TENSOR_CONV_DW_NORM, "weight", i), {inner_dim}, 0);
        layer.conv_dw_norm_b = create_tensor(tn(LLM_TENSOR_CONV_DW_NORM, "bias",   i), {inner_dim}, 0);

        layer.conv_pw2   = create_tensor(tn(LLM_TENSOR_CONV_PW2, "weight", i), {inner_dim, n_embd}, 0);
        layer.conv_pw2_b = create_tensor(tn(LLM_TENSOR_CONV_PW2, "bias",   i), {n_embd}, 0);

        layer.ffn_norm   = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias",   i), {n_embd}, 0);
        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias",   i), {n_ff}, 0);
        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias",   i), {n_embd}, 0);

        layer.ffn_norm_1   = create_tensor(tn(LLM_TENSOR_FFN_NORM_1, "weight", i), {n_embd}, 0);
        layer.ffn_norm_1_b = create_tensor(tn(LLM_TENSOR_FFN_NORM_1, "bias",   i), {n_embd}, 0);
        layer.ffn_up_1     = create_tensor(tn(LLM_TENSOR_FFN_UP_1,   "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up_1_b   = create_tensor(tn(LLM_TENSOR_FFN_UP_1,   "bias",   i), {n_ff}, 0);
        layer.ffn_down_1   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_1, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_down_1_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN_1, "bias",   i), {n_embd}, 0);

        layer.ffn_post_norm   = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_post_norm_b = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "bias",   i), {n_embd}, 0);
    }

    // terminal CTC head; the same weight is also applied mid-stack for self-conditioning
    output   = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, 0);
    output_b = create_tensor(tn(LLM_TENSOR_OUTPUT, "bias"),   {n_vocab}, 0);

    // mid-stack self-conditioning back-projection (softmax(mid ctc logits) -> hidden_dim)
    ctc_out_mid   = create_tensor(tn(LLM_TENSOR_CTC_OUT_MID, "weight"), {n_vocab, n_embd}, 0);
    ctc_out_mid_b = create_tensor(tn(LLM_TENSOR_CTC_OUT_MID, "bias"),   {n_embd}, 0);
}

std::unique_ptr<llm_graph_context> llama_model_granite_speech_5::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_granite_speech_5::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t context_size = hparams.ctc.context_size;
    const int64_t conv_kernel  = hparams.ctc.conv_kernel;
    const int64_t conv_pad     = conv_kernel / 2;
    const int64_t ctc_layer    = n_layer / 2;

    // Shaw relative-position distance table: depends only on context_size/max_pos_emb, shared
    // by every layer regardless of the sequence length in effect at that layer
    ggml_tensor * attn_dists = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, context_size * context_size);
    ggml_set_input(attn_dists);
    {
        const int64_t max_pos_emb = hparams.ctc.max_pos_emb;
        std::vector<int32_t> dists((size_t) context_size * (size_t) context_size);
        for (int64_t i = 0; i < context_size; i++) {
            for (int64_t j = 0; j < context_size; j++) {
                int64_t d = i - j;
                if (d < -context_size) d = -context_size;
                if (d >  context_size) d =  context_size;
                dists[(size_t) i * (size_t) context_size + (size_t) j] = (int32_t) (d + max_pos_emb);
            }
        }
        res->add_input(std::make_unique<llm_graph_input_ctc_const_i32>(attn_dists, std::move(dists)));
    }

    // the sequence length changes at each subsampling layer; simulate the same halving
    // sequence here (host-side, at graph-build time) that the loop below applies to the
    // actual tensors, so attn_mask can be precomputed once per distinct length
    std::vector<int64_t> len_by_layer(n_layer);
    {
        int64_t len = n_tokens;
        for (int il = 0; il < n_layer; il++) {
            len_by_layer[il] = len;
            const auto subsample_factor = hparams.subsample_factor(il);
            if (subsample_factor > 1) {
                // clamp to >= 1 so the conv module's causal-padding trick (which needs
                // ne[0] > conv_pad strictly) stays valid even for pathologically short
                // inputs (e.g. the n_tokens=1 buffer-size reserve pass at context init)
                len = std::max<int64_t>(len / subsample_factor, 1);
            }
        }
    }

    // one attn_mask per distinct length that occurs (skip lengths with no ragged remainder)
    std::map<int64_t, ggml_tensor *> attn_mask_by_len;
    for (int64_t len : len_by_layer) {
        if (attn_mask_by_len.count(len)) {
            continue;
        }
        const int64_t remainder = len % context_size;
        if (remainder == 0) {
            continue;
        }
        const int64_t num_blocks = (len + context_size - 1) / context_size;
        ggml_tensor * attn_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, context_size, context_size, 1, num_blocks);
        ggml_set_input(attn_mask);

        std::vector<float> mask((size_t) context_size * (size_t) context_size * (size_t) num_blocks, 0.0f);
        const size_t last_block_offset = (size_t) (num_blocks - 1) * (size_t) context_size * (size_t) context_size;
        for (int64_t q = 0; q < context_size; q++) {
            for (int64_t k = 0; k < context_size; k++) {
                if (q >= remainder || k >= remainder) {
                    mask[last_block_offset + (size_t) q * (size_t) context_size + (size_t) k] = -INFINITY;
                }
            }
        }
        res->add_input(std::make_unique<llm_graph_input_ctc_const_f32>(attn_mask, std::move(mask)));

        attn_mask_by_len[len] = attn_mask;
    }

    ggml_tensor * cur = build_inp_embd(model.tok_embd);
    cb(cur, "inp_embd", -1);

    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];

        const int64_t cur_len     = len_by_layer[il];
        const int64_t num_blocks  = (cur_len + context_size - 1) / context_size;
        const int64_t padded_len  = num_blocks * context_size;
        ggml_tensor * attn_mask   = attn_mask_by_len.count(cur_len) ? attn_mask_by_len[cur_len] : nullptr;

        ggml_tensor * residual = cur;

        // ffn1 (half-step)
        {
            ggml_tensor * ffn1 = build_norm(cur, layer.ffn_norm, layer.ffn_norm_b, LLM_NORM, il);
            ffn1 = build_ffn(ffn1,
                layer.ffn_up, layer.ffn_up_b, nullptr,
                nullptr, nullptr, nullptr,
                layer.ffn_down, layer.ffn_down_b, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_SEQ, il);
            residual = ggml_add(ctx0, residual, ggml_scale(ctx0, ffn1, 0.5f));
        }

        // block-local self-attention with Shaw relative position bias
        {
            ggml_tensor * normed = build_norm(residual, layer.attn_norm, layer.attn_norm_b, LLM_NORM, il);

            if (cur_len < padded_len) {
                normed = ggml_pad(ctx0, normed, 0, padded_len - cur_len, 0, 0);
            }

            ggml_tensor * Q = ggml_mul_mat(ctx0, layer.wq, normed);
            ggml_tensor * K = ggml_mul_mat(ctx0, layer.wk, normed);
            ggml_tensor * V = ggml_mul_mat(ctx0, layer.wv, normed);

            Q = ggml_reshape_4d(ctx0, Q, n_embd_head_k, n_head, context_size, num_blocks);
            K = ggml_reshape_4d(ctx0, K, n_embd_head_k, n_head, context_size, num_blocks);
            V = ggml_reshape_4d(ctx0, V, n_embd_head_k, n_head, context_size, num_blocks);

            ggml_tensor * Q_perm = ggml_permute(ctx0, Q, 0, 2, 1, 3);
            ggml_tensor * K_perm = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));

            ggml_tensor * kq = ggml_mul_mat(ctx0, K_perm, Q_perm);

            // Shaw RPE: pos_emb ne[2]=1 broadcasts against Q ne[2]=num_blocks in mul_mat
            ggml_tensor * pos_emb = ggml_get_rows(ctx0, layer.attn_rel_pos, attn_dists);
            pos_emb = ggml_reshape_3d(ctx0, pos_emb, n_embd_head_k, context_size, context_size);
            pos_emb = ggml_reshape_4d(ctx0, pos_emb, n_embd_head_k, context_size, 1, context_size);

            ggml_tensor * Q_shaw = ggml_permute(ctx0, Q, 0, 1, 3, 2);
            ggml_tensor * pos_attn = ggml_mul_mat(ctx0, pos_emb, Q_shaw);
            pos_attn = ggml_cont(ctx0, ggml_permute(ctx0, pos_attn, 0, 2, 3, 1));

            ggml_tensor * scores = ggml_add(ctx0, kq, pos_attn);
            ggml_tensor * attn_weights = ggml_soft_max_ext(ctx0, scores, attn_mask, 1.0f / sqrtf((float) n_embd_head_k), 0.0f);

            ggml_tensor * V_perm = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3));
            ggml_tensor * attn_out = ggml_mul_mat(ctx0, V_perm, attn_weights);

            attn_out = ggml_permute(ctx0, attn_out, 0, 2, 1, 3);
            attn_out = ggml_cont_2d(ctx0, attn_out, n_embd, padded_len);

            if (cur_len < padded_len) {
                attn_out = ggml_view_2d(ctx0, attn_out, n_embd, cur_len, attn_out->nb[1], 0);
            }

            cur = ggml_mul_mat(ctx0, layer.wo, attn_out);
            cur = ggml_add(ctx0, cur, layer.wo_b);
        }

        residual = ggml_add(ctx0, residual, cur);

        // conv module
        {
            cur = build_norm(residual, layer.conv_norm, layer.conv_norm_b, LLM_NORM, il);

            ggml_tensor * x = ggml_mul_mat(ctx0, layer.conv_pw1, cur);
            x = ggml_add(ctx0, x, layer.conv_pw1_b);

            // GLU: ggml has no fused op, manual split + sigmoid gate
            {
                const int64_t d = x->ne[0] / 2;
                ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], d * x->nb[0]));
                x = ggml_mul(ctx0, ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], 0), gate);
                x = ggml_cont(ctx0, ggml_transpose(ctx0, x));
            }

            x = ggml_pad(ctx0, x, conv_pad, 0, 0, 0);
            x = ggml_roll(ctx0, x, conv_pad, 0, 0, 0);
            x = ggml_pad(ctx0, x, conv_pad, 0, 0, 0);
            x = ggml_ssm_conv(ctx0, x, layer.conv_dw); // -> [inner_dim, cur_len, 1]

            const auto subsample_factor = hparams.subsample_factor(il);
            // must match the clamp used when precomputing len_by_layer above, so this
            // layer's actual output length matches what the next layer expects
            const int64_t half_len = std::max<int64_t>(cur_len / 2, 1);

            if (subsample_factor > 1) {
                // exact stride-2 subsampling: output[i] of a true stride-2 conv equals
                // output[2*i] of the stride-1 conv computed above (same input window, same
                // kernel) - take every other frame, then trim to half_len like the reference
                x = ggml_view_3d(ctx0, x, x->ne[0], half_len, x->ne[2], x->nb[1] * subsample_factor, x->nb[2], 0);
                x = ggml_cont(ctx0, x);
            }

            x = ggml_add(ctx0, ggml_mul(ctx0, x, layer.conv_dw_norm), layer.conv_dw_norm_b);
            x = ggml_silu(ctx0, x);

            x = ggml_mul_mat(ctx0, layer.conv_pw2, x);
            x = ggml_add(ctx0, x, layer.conv_pw2_b);

            if (subsample_factor > 1) {
                // pool the pre-conv residual to the same half length instead of a plain add
                ggml_tensor * pooled;
                if (cur_len >= subsample_factor) {
                    GGML_ASSERT(subsample_factor == 2 && "Only subsample factor 2 supported");
                    ggml_tensor * even = ggml_view_2d(ctx0, residual, n_embd, half_len, residual->nb[1] * 2, 0);
                    ggml_tensor * odd  = ggml_view_2d(ctx0, residual, n_embd, half_len, residual->nb[1] * 2, residual->nb[1]);
                    pooled = ggml_scale(ctx0, ggml_add(ctx0, ggml_cont(ctx0, even), ggml_cont(ctx0, odd)), 0.5f);
                } else {
                    // cur_len == 1: a pathologically short input (e.g. the n_tokens=1
                    // buffer-size reserve pass at context init) with no pair to average -
                    // use the single available frame as-is
                    pooled = ggml_cont(ctx0, ggml_view_2d(ctx0, residual, n_embd, half_len, residual->nb[1], 0));
                }

                residual = ggml_add(ctx0, x, pooled);
            } else {
                residual = ggml_add(ctx0, residual, x);
            }
        }

        // ffn2 (half-step)
        {
            ggml_tensor * ffn2 = build_norm(residual, layer.ffn_norm_1, layer.ffn_norm_1_b, LLM_NORM, il);
            ffn2 = build_ffn(ffn2,
                layer.ffn_up_1, layer.ffn_up_1_b, nullptr,
                nullptr, nullptr, nullptr,
                layer.ffn_down_1, layer.ffn_down_1_b, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_SEQ, il);
            residual = ggml_add(ctx0, residual, ggml_scale(ctx0, ffn2, 0.5f));
        }

        cur = build_norm(residual, layer.ffn_post_norm, layer.ffn_post_norm_b, LLM_NORM, il);
        cb(cur, "layer_out", il);

        // mid-stack self-conditioning: predict CTC logits early and feed the softmax back in
        if (il + 1 == ctc_layer) {
            ggml_tensor * mid = ggml_mul_mat(ctx0, model.output, cur);
            mid = ggml_add(ctx0, mid, model.output_b);
            mid = ggml_soft_max(ctx0, mid);
            mid = ggml_mul_mat(ctx0, model.ctc_out_mid, mid);
            mid = ggml_add(ctx0, mid, model.ctc_out_mid_b);
            cur = ggml_add(ctx0, cur, mid);
            cb(cur, "ctc_branch", il);
        }
    }

    // terminal CTC head: plain linear projection to vocab logits, no softmax (only argmax is
    // taken downstream, which is invariant under it)
    cur = ggml_mul_mat(ctx0, model.output, cur);
    cur = ggml_add(ctx0, cur, model.output_b);

    cb(cur, "result_logits", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
