#include "models.h"

#include "llama-impl.h"
#include "llama-memory-recurrent.h"

llm_build_delta_net_base::llm_build_delta_net_base(const llm_graph_params & params) : llm_graph_context(params) {}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b,
        ggml_tensor * s,
        int           il) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(H_v % H_k == 0);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

    // K=1: output carries the final state only. state s is 4D [S_v, S_v, H_v, n_seqs].
    ggml_tensor * result = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, nullptr, nullptr, nullptr, nullptr, /*K=*/1, false);
    if (n_tokens == 1) {
        res->add_fused_node({LLM_FUSED_OP_GDN_AR, result, il});
    } else {
        res->add_fused_node({LLM_FUSED_OP_GDN_CH, result, il});
    }

    ggml_tensor * output = ggml_view_4d(ctx0, result,
            S_v, H_v, n_tokens, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v * H_v),
            ggml_row_size(result->type, S_v * H_v * n_tokens), 0);

    ggml_tensor * new_state = ggml_view_4d(ctx0, result,
            S_v, S_v, H_v, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v * S_v),
            ggml_row_size(result->type, S_v * S_v * H_v),
            ggml_row_size(result->type, S_v * H_v * n_tokens * n_seqs));

    return {output, new_state};
}

ggml_tensor * llm_build_delta_net_base::build_conv_state(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        qkv_mixed,
        int64_t              conv_kernel_size,
        int64_t              conv_channels,
        int                  il) {
    const int64_t n_seqs = ubatch.n_seqs;

    if (cparams.rs_replay) {
        return build_conv_state_replay(inp, conv_states_all, qkv_mixed, conv_kernel_size - 1, conv_channels, il);
    }

    ggml_tensor * conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    cb(conv_states, "conv_states", il);

    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);

    qkv_mixed = ggml_transpose(ctx0, qkv_mixed);
    cb(qkv_mixed, "qkv_mixed_transposed", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);
    cb(conv_input, "conv_input", il);

    build_conv_state_store(inp, conv_states_all, conv_input, conv_kernel_size - 1, 0);

    return conv_input;
}

// [TAG_RECURRENT_REPLAY]
// rs_replay keeps one committed state per seq plus the inputs of the last ubatch when it is short enough to be rolled
// back (2 <= n_seq_tokens <= n_rs_seq + 1). The next ubatch first replays the pending tokens that seq_rm kept, commits
// the result and then runs its own tokens. Must match the rule in llama_memory_recurrent::find_slot.
// Every ubatch builds the same nodes, with or without pending tokens, so the allocation planned at reserve time
// stays valid and the scheduler never re-plans.
static bool rs_replay_keep(const llama_cparams & cparams, int64_t n_seq_tokens) {
    return n_seq_tokens >= 2 && n_seq_tokens <= (int64_t) cparams.n_rs_seq + 1;
}

ggml_tensor * llm_build_delta_net_base::build_conv_state_replay(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        x,
        int64_t              n_conv,
        int64_t              channels,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head = mctx_cur->get_head();

    const int64_t n_hist       = n_conv + cparams.n_rs_seq + 1;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(conv_states_all->ne[0] == channels*n_hist);
    GGML_ASSERT(x->ne[0] == channels && x->ne[1] == n_seq_tokens && x->ne[2] == n_seqs);
    GGML_ASSERT(n_conv <= inp->n_conv);

    // channels first: [window | pending]
    ggml_tensor * hist = build_rs(inp, conv_states_all, conv_states_all->ne[0], n_seqs);
    hist = ggml_reshape_3d(ctx0, hist, channels, n_hist, n_seqs);
    cb(hist, "conv_hist", il);

    // the column table is sized for the widest window of the model
    ggml_tensor * win_cols = ggml_view_2d(ctx0, inp->rep_cols, n_conv, n_seqs, inp->rep_cols->nb[1], 0);

    // [channels, n_conv + n_seq_tokens]: the window after the kept pending tokens, then the new tokens
    // (built channels first, a transposed concat is much slower than a concat and a transpose)
    ggml_tensor * cols = ggml_concat(ctx0, ggml_get_rows(ctx0, hist, win_cols), x, 1);

    ggml_tensor * conv_input = ggml_cont(ctx0, ggml_transpose(ctx0, cols));
    cb(conv_input, "conv_input", il);

    // keep: the new tokens stay pending behind the committed window, else commit the last n_conv columns
    const int64_t n_dst = rs_replay_keep(cparams, n_seq_tokens) ? n_conv + n_seq_tokens : n_conv;

    ggml_tensor * tail = ggml_view_3d(ctx0, cols, channels, n_dst, n_seqs, cols->nb[1], cols->nb[2],
            (n_conv + n_seq_tokens - n_dst)*cols->nb[1]);
    // the rows as [channels, n_hist, cells], so that the view of the ubatch cells keeps their strides (-sm tensor)
    ggml_tensor * hist_all = ggml_reshape_3d(ctx0, conv_states_all, channels, n_hist, conv_states_all->ne[1]);
    ggml_tensor * hist_dst = ggml_view_3d(ctx0, hist_all, channels, n_dst, n_seqs, hist_all->nb[1], hist_all->nb[2],
            kv_head*hist_all->nb[2]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, tail, hist_dst));

    return conv_input;
}

ggml_tensor * llm_build_delta_net_base::build_recurrent_attn_replay(
        llm_graph_input_rs * inp,
        ggml_tensor *        ssm_states_all,
        ggml_tensor *        q,
        ggml_tensor *        k,
        ggml_tensor *        v,
        ggml_tensor *        g,
        ggml_tensor *        b,
        ggml_tensor *        s,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head = mctx_cur->get_head();

    const int64_t n_pend       = cparams.n_rs_seq + 1;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    const int64_t S_k = k->ne[0];
    const int64_t H_k = k->ne[1];
    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];
    const int64_t g0  = g->ne[0];

    const size_t row_size = ggml_row_size(ssm_states_all->type, hparams.n_embd_s());

    // the pending inputs of the cells: k [S_k, H_k, n_pend], v [S_v, H_v, n_pend], gb [g0, H_v, 2*n_pend] (the gates,
    // then the betas). The views of the ubatch cells keep the strides of these 4D views, so that -sm tensor can split
    // them by heads
    ggml_tensor * spk  = mctx_cur->get_spk_l(il);
    ggml_tensor * spv  = mctx_cur->get_spv_l(il);
    ggml_tensor * spgb = mctx_cur->get_spgb_l(il);

    ggml_tensor * ck  = ggml_reshape_4d(ctx0, spk,  S_k, H_k,   n_pend, spk->ne[1]);
    ggml_tensor * cv  = ggml_reshape_4d(ctx0, spv,  S_v, H_v,   n_pend, spv->ne[1]);
    ggml_tensor * cgb = ggml_reshape_4d(ctx0, spgb, g0,  H_v, 2*n_pend, spgb->ne[1]);

    const auto gather = [&](ggml_tensor * rows, ggml_tensor * c) {
        return ggml_reshape_4d(ctx0, build_rs(inp, rows, rows->ne[0], n_seqs), c->ne[0], c->ne[1], c->ne[2], n_seqs);
    };

    ggml_tensor * pk  = gather(spk,  ck);
    ggml_tensor * pv  = gather(spv,  cv);
    ggml_tensor * pgb = gather(spgb, cgb);

    const bool keep = rs_replay_keep(cparams, n_seq_tokens);

    ggml_tensor * state_dst = ggml_view_2d(ctx0, ssm_states_all, hparams.n_embd_s(), n_seqs, ssm_states_all->nb[1], kv_head*row_size);

    // store the inputs of the new tokens as pending; only used when keep, a longer ubatch stores its first n_pend tokens
    const auto store_inputs = [&]() {
        const int64_t n_st = std::min(n_seq_tokens, n_pend);

        const auto store = [&](ggml_tensor * x, ggml_tensor * c, int64_t t0) {
            ggml_tensor * x_st = ggml_view_4d(ctx0, x, x->ne[0], x->ne[1], n_st, n_seqs, x->nb[1], x->nb[2], x->nb[3], 0);
            ggml_tensor * dst  = ggml_view_4d(ctx0, c, x->ne[0], x->ne[1], n_st, n_seqs, c->nb[1], c->nb[2], c->nb[3],
                    kv_head*c->nb[3] + t0*c->nb[2]);
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, x_st, dst));
        };

        store(k, ck,  0);
        store(v, cv,  0);
        store(g, cgb, 0);
        store(b, cgb, n_pend);
    };

    // the gdn op replays the kept pending tokens, then runs the new tokens
    ggml_tensor * result = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, pk, pv, pgb, inp->rep_n, 1, keep);

    ggml_tensor * output = ggml_view_4d(ctx0, result, S_v, H_v, n_seq_tokens, n_seqs,
            ggml_row_size(result->type, S_v),
            ggml_row_size(result->type, S_v*H_v),
            ggml_row_size(result->type, S_v*H_v*n_seq_tokens), 0);
    cb(output, "attn_output", il);

    // keep: the state after the replay, else the final state
    ggml_tensor * new_state = ggml_view_2d(ctx0, result, hparams.n_embd_s(), n_seqs,
            ggml_row_size(result->type, hparams.n_embd_s()),
            ggml_row_size(result->type, S_v*H_v*n_seq_tokens*n_seqs));
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, new_state, state_dst));

    store_inputs();

    return output;
}

ggml_tensor * llm_build_delta_net_base::build_recurrent_attn(
        llm_graph_input_rs * inp,
        ggml_tensor *        ssm_states_all,
        ggml_tensor *        q,
        ggml_tensor *        k,
        ggml_tensor *        v,
        ggml_tensor *        g,
        ggml_tensor *        b,
        ggml_tensor *        s,
        int                  il) {
    const auto * mctx_cur   = inp->mctx;
    const auto   kv_head    = mctx_cur->get_head();
    const uint32_t mem_size = mctx_cur->get_size();

    if (cparams.rs_replay) {
        return build_recurrent_attn_replay(inp, ssm_states_all, q, k, v, g, b, s, il);
    }

    const int64_t S_v          = s->ne[0];
    const int64_t H_v          = s->ne[2];
    const int64_t n_seqs       = s->ne[3];
    const int64_t n_seq_tokens = q->ne[2];

    const bool keep = cparams.n_rs_seq > 0;

    if (!keep) {
        auto attn_out = build_delta_net(q, k, v, g, b, s, il);
        ggml_tensor * output    = attn_out.first;
        ggml_tensor * new_state = attn_out.second;
        cb(output, "attn_output", il);
        cb(new_state, "new_state", il);

        ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, new_state,
                    ggml_view_2d(ctx0, ssm_states_all, hparams.n_embd_s(), n_seqs, ssm_states_all->nb[1],
                        kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

        return output;
    }

    const int64_t D = S_v * S_v * H_v;
    const int64_t K = cparams.n_rs_seq + 1;

    // state s is 4D [S_v, S_v, H_v, n_seqs]; K snapshot slots are written into the output.
    ggml_tensor * gdn_out = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, nullptr, nullptr, nullptr, nullptr, K, false);
    if (n_seq_tokens > 1) {
        res->add_fused_node({LLM_FUSED_OP_GDN_CH, gdn_out, il});
    } else {
        res->add_fused_node({LLM_FUSED_OP_GDN_AR, gdn_out, il});
    }

    const int64_t attn_score_elems    = S_v * H_v * n_seq_tokens * n_seqs;
    const int64_t state_size_per_snap = S_v * S_v * H_v * n_seqs;

    ggml_tensor * output = ggml_view_4d(ctx0, gdn_out,
        S_v, H_v, n_seq_tokens, n_seqs,
        ggml_row_size(gdn_out->type, S_v),
        ggml_row_size(gdn_out->type, S_v * H_v),
        ggml_row_size(gdn_out->type, S_v * H_v * n_seq_tokens),
        0);
    cb(output, "attn_output", il);

    const size_t row_size = hparams.n_embd_s() * ggml_element_size(ssm_states_all);

    // op writes the last min(n_seq_tokens, K) snapshots; trailing slots are left unwritten
    const int64_t n_written = std::min<int64_t>(n_seq_tokens, K);

    // write the produced snapshots into the recurrent cache (snapshot slot i -> rollback group i)
    ggml_tensor * src = ggml_view_3d(ctx0, gdn_out,
        D, n_seqs, n_written,
        ggml_row_size(gdn_out->type, D),
        ggml_row_size(gdn_out->type, state_size_per_snap),
        ggml_row_size(gdn_out->type, attn_score_elems));

    ggml_tensor * dst = ggml_view_3d(ctx0, ssm_states_all,
        D, n_seqs, n_written,
        ssm_states_all->nb[1],
        (size_t) mem_size * row_size,
        (size_t) kv_head * row_size);

    ggml_build_forward_expand(gf, ggml_cpy(ctx0, src, dst));

    return output;
}
