#include "models.h"

#include "llama-memory-recurrent.h"

#include <algorithm>

llm_build_lfm_base::llm_build_lfm_base(const llm_graph_params & params) : llm_graph_context(params) {}

ggml_tensor * llm_build_lfm_base::build_lfm_shortconv(const llama_model & model,
                                                      const llama_memory_recurrent_context * mctx_recr,
                                                      llm_graph_input_rs * inp_recr,
                                                      ggml_tensor * cur,
                                                      int il) {
    const uint32_t kv_head      = mctx_recr->get_head();
    const int64_t  n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t  n_seqs       = ubatch.n_seqs;
    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    GGML_ASSERT(hparams.n_shortconv_l_cache > 1);
    const uint32_t d_conv = hparams.n_shortconv_l_cache - 1;
    const int64_t  n_embd = hparams.n_embd;

    // {n_embd, n_tokens} => {n_embd, n_seq_tokens, n_seqs}
    cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

    auto * bcx = build_lora_mm(model.layers[il].shortconv.in_proj, cur);
    cb(bcx, "model.layers.{}.conv.in_proj", il);

    constexpr auto n_chunks = 3;
    GGML_ASSERT(bcx->ne[0] % n_chunks == 0);
    const auto chunk_size = bcx->ne[0] / n_chunks;
    auto *     b          = ggml_view_3d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->ne[2], bcx->nb[1], bcx->nb[2],
                                         0 * chunk_size * ggml_element_size(bcx));
    auto *     c          = ggml_view_3d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->ne[2], bcx->nb[1], bcx->nb[2],
                                         1 * chunk_size * ggml_element_size(bcx));
    auto *     x          = ggml_view_3d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->ne[2], bcx->nb[1], bcx->nb[2],
                                         2 * chunk_size * ggml_element_size(bcx));

    auto * bx = ggml_transpose(ctx0, ggml_mul(ctx0, b, x));

    // read conv state
    auto * conv_state = mctx_recr->get_r_l(il);
    auto * conv_rs    = build_rs(inp_recr, conv_state, hparams.n_embd_r(), n_seqs);
    auto * conv       = ggml_reshape_3d(ctx0, conv_rs, d_conv, n_embd, n_seqs);

    // causal prepends the state, non-causal pads symmetrically for a centered window
    if (hparams.causal_attn) {
        bx = ggml_concat(ctx0, conv, bx, 0);
    } else {
        const int64_t pad = (hparams.n_shortconv_l_cache - 1) / 2;
        auto * left = ggml_cont(ctx0,
            ggml_view_3d(ctx0, conv, pad, n_embd, n_seqs, conv->nb[1], conv->nb[2], (d_conv - pad) * conv->nb[0]));
        bx = ggml_pad_ext(ctx0, ggml_concat(ctx0, left, bx, 0), 0, pad, 0, 0, 0, 0, 0, 0);
    }
    GGML_ASSERT(bx->ne[0] > conv->ne[0]);

    // write conv states: slot 0 = the final state, slot s = the state s tokens back (partial rollback)
    const int64_t K         = hparams.causal_attn && cparams.n_rs_seq > 0 ? (int64_t) cparams.n_rs_seq + 1 : 1;
    const int64_t n_written = std::min<int64_t>(n_seq_tokens, K);
    const auto    mem_size  = mctx_recr->get_size();
    const size_t  row_size  = ggml_row_size(conv_state->type, (int64_t) d_conv * n_embd);

    for (int64_t slot = 0; slot < n_written; ++slot) {
        auto * conv_snap = ggml_view_3d(ctx0, bx, d_conv, bx->ne[1], bx->ne[2], bx->nb[1], bx->nb[2],
                                        (bx->ne[0] - d_conv - slot) * ggml_element_size(bx));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, conv_snap,
                                               ggml_view_2d(ctx0, conv_state, (int64_t) d_conv * n_embd, n_seqs,
                                                            conv_state->nb[1],
                                                            ((size_t) slot * mem_size + kv_head) * row_size)));
    }

    auto * conv_kernel = model.layers[il].shortconv.conv;
    auto * conv_out    = ggml_ssm_conv(ctx0, bx, conv_kernel);
    cb(conv_out, "model.layers.{}.conv.conv", il);

    auto * y = ggml_mul(ctx0, c, conv_out);
    y        = build_lora_mm(model.layers[il].shortconv.out_proj, y);
    cb(y, "model.layers.{}.conv.out_proj", il);
    // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
    y = ggml_reshape_2d(ctx0, y, y->ne[0], n_seq_tokens * n_seqs);

    return y;
}

ggml_tensor * llm_build_lfm_base::build_lfm_shortconv(const llama_model & model,
                                                      ggml_tensor * cur,
                                                      int il) {
    const int64_t d_conv = hparams.n_shortconv_l_cache - 1;

    auto * bcx = build_lora_mm(model.layers[il].shortconv.in_proj, cur);
    cb(bcx, "model.layers.{}.conv.in_proj", il);

    const int64_t chunk_size = bcx->ne[0] / 3;
    auto * b = ggml_view_2d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->nb[1], 0 * chunk_size * ggml_element_size(bcx));
    auto * c = ggml_view_2d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->nb[1], 1 * chunk_size * ggml_element_size(bcx));
    auto * x = ggml_view_2d(ctx0, bcx, chunk_size, bcx->ne[1], bcx->nb[1], 2 * chunk_size * ggml_element_size(bcx));

    // causal from a zero state
    auto * bx = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_mul(ctx0, b, x)));
    bx = ggml_pad_ext(ctx0, bx, d_conv, 0, 0, 0, 0, 0, 0, 0);

    auto * conv_out = ggml_ssm_conv(ctx0, bx, model.layers[il].shortconv.conv);
    cb(conv_out, "model.layers.{}.conv.conv", il);

    auto * y = ggml_mul(ctx0, c, ggml_reshape_2d(ctx0, conv_out, chunk_size, n_tokens));
    y = build_lora_mm(model.layers[il].shortconv.out_proj, y);
    cb(y, "model.layers.{}.conv.out_proj", il);

    return y;
}

ggml_tensor * llm_build_lfm_base::build_lfm_dense_ffn(const llama_model & model,
                                                      ggml_tensor * cur,
                                                      int il) {
    GGML_ASSERT(!model.layers[il].ffn_up_b);
    GGML_ASSERT(!model.layers[il].ffn_gate_b);
    GGML_ASSERT(!model.layers[il].ffn_down_b);
    return build_ffn(cur,
        model.layers[il].ffn_up,   NULL,
        model.layers[il].ffn_gate, NULL,
        model.layers[il].ffn_down, NULL,
        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
}

ggml_tensor * llm_build_lfm_base::build_lfm_moe_ffn(const llama_model & model,
                                                    ggml_tensor * cur,
                                                    int il) {
    return build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            model.layers[il].ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            static_cast<llama_expert_gating_func_type>(hparams.expert_gating_func),
            il);
}
