#include "models.h"

#define CHUNK_SIZE 64

// utility to get one slice from the third dimension
// input dim:  [x, y, c, b]
// output dim: [x, y, 1, b]
static ggml_tensor * get_slice_2d(ggml_context * ctx0, ggml_tensor * t, int64_t c) {
    return ggml_view_4d(ctx0, t, t->ne[0], t->ne[1], 1, t->ne[3],
        t->nb[1], t->nb[2], t->nb[3], t->nb[2] * c);
}

llm_build_delta_net_base::llm_build_delta_net_base(const llm_graph_params & params) : llm_graph_context(params) {}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net_chunking(
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
    const bool kda = (g->ne[0] == S_k && g->ne[1] == H_k);

    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(H_v % H_k == 0);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

    const float scale = 1.0f / sqrtf(S_k);

    q = ggml_scale(ctx0, q, scale);

    cb(q, "q_in", il);
    cb(k, "k_in", il);
    cb(v, "v_in", il);
    cb(b, "b_in", il);
    cb(g, "g_in", il);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    k = ggml_permute(ctx0, k, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    v = ggml_permute(ctx0, v, 0, 2, 1, 3); // [S_v, n_tokens, H_v, n_seqs]
    g = ggml_permute(ctx0, g, 0, 2, 1, 3); // [g_0, n_tokens, H_v, n_seqs]
    b = ggml_permute(ctx0, b, 0, 2, 1, 3); // [  1, n_tokens, H_v, n_seqs]

    const int CS = CHUNK_SIZE;

    const int pad = (CS - n_tokens % CS) % CS;
    const int n_chunks = (n_tokens + pad) / CS;

    q = ggml_pad(ctx0, q, 0, pad, 0, 0);
    k = ggml_pad(ctx0, k, 0, pad, 0, 0);
    v = ggml_pad(ctx0, v, 0, pad, 0, 0);
    g = ggml_pad(ctx0, g, 0, pad, 0, 0);
    b = ggml_pad(ctx0, b, 0, pad, 0, 0);

    ggml_tensor * v_b = ggml_mul(ctx0, v, b);
    ggml_tensor * k_b = ggml_mul(ctx0, k, b);

    cb(v_b, "v_b", il);
    cb(k_b, "k_b", il);

    q   = ggml_reshape_4d(ctx0, q,   S_k, CS, n_chunks, H_k * n_seqs);
    k   = ggml_reshape_4d(ctx0, k,   S_k, CS, n_chunks, H_k * n_seqs);
    k_b = ggml_reshape_4d(ctx0, k_b, S_k, CS, n_chunks, H_v * n_seqs);
    v   = ggml_reshape_4d(ctx0, v,   S_v, CS, n_chunks, H_v * n_seqs);
    v_b = ggml_reshape_4d(ctx0, v_b, S_v, CS, n_chunks, H_v * n_seqs);

    g = ggml_reshape_4d(ctx0, g, g->ne[0], CS, n_chunks, H_v * n_seqs);
    b = ggml_reshape_4d(ctx0, b, 1,        CS, n_chunks, H_v * n_seqs);

    // [CS, g_0, n_chunks, H_v * n_seqs]
    // TODO: extend ggml_cumsum with axis parameter to avoid transpose
    ggml_tensor * g_cs = ggml_cumsum(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, g)));
    cb(g_cs, "g_cs", il);

    ggml_tensor * kb = nullptr;
    ggml_tensor * kq = nullptr;
    if (kda) {
        const int64_t HB = H_k * n_seqs;
        const int64_t CHB = n_chunks * H_k * n_seqs;
        const int64_t block_size = 16;
        const int64_t n_blocks = CHUNK_SIZE / block_size;

        ggml_tensor * k_block[n_blocks];
        ggml_tensor * q_block[n_blocks];
        ggml_tensor * gk_block[n_blocks];
        ggml_tensor * gk_block_bc[n_blocks];
        for (int64_t j = 0; j < n_blocks; ++j) {
            int64_t j_start = j * block_size;

            // k_i_block: [S, block_size, C, HB]
            k_block[j] = ggml_cont(ctx0, ggml_view_4d(ctx0, k,
                S_k, block_size, n_chunks, HB,
                k->nb[1], k->nb[2], k->nb[3],
                j_start * k->nb[1]));

            k_block[j] = ggml_reshape_4d(ctx0, k_block[j], S_k, block_size, 1, CHB);
            // q_i_block: [S, block_size, C, HB]
            q_block[j] = ggml_cont(ctx0, ggml_view_4d(ctx0, q,
                    S_k, block_size, n_chunks, HB,
                    q->nb[1], q->nb[2], q->nb[3],
                    j_start * q->nb[1]));

            q_block[j] = ggml_reshape_4d(ctx0, q_block[j], S_k, block_size, 1, CHB);
            // gk_j_block: [S, block_size, C, HB]
            gk_block[j] = ggml_cont(ctx0, ggml_view_4d(ctx0, g_cs,
                block_size, S_k, n_chunks, HB,
                g_cs->nb[1], g_cs->nb[2], g_cs->nb[3],
                j_start * g_cs->nb[0]));
            gk_block[j] = ggml_reshape_4d(ctx0, gk_block[j], 1, block_size, S_k, CHB);

            gk_block_bc[j] = ggml_repeat_4d(ctx0, gk_block[j], block_size, block_size, S_k, CHB);

            gk_block[j] = ggml_reshape_4d(ctx0, gk_block[j], block_size, 1, S_k, CHB);
        }

        ggml_tensor * kb_rows[n_blocks];
        ggml_tensor * kq_rows[n_blocks];
        for (int64_t j = 0; j < n_blocks; ++j) {
            ggml_tensor * kb_row = nullptr;
            ggml_tensor * kq_row = nullptr;
            ggml_tensor * k_j_block = ggml_reshape_4d(ctx0, k_block[j], S_k, 1, block_size, CHB);
            for (int64_t i = 0; i <= j; ++i) {
                ggml_tensor * decay_mask = ggml_sub(ctx0, gk_block_bc[j], gk_block[i]);
                cb(decay_mask, "decay_mask", il);

                // Apply diag_mask only at diagnoal blocks
                if (i == j) {
                    decay_mask = ggml_tri(ctx0, decay_mask, GGML_TRI_TYPE_LOWER_DIAG);
                }
                decay_mask = ggml_exp(ctx0, decay_mask);

                // decay_mask [S_k,BT_j,BT_i,ShHB] *Note* second and third chunk_sizes are switched
                decay_mask = ggml_cont_4d(ctx0, ggml_permute(ctx0, decay_mask, 2, 1, 0, 3), S_k, block_size, block_size, CHB);

                ggml_tensor * decay_k_i = ggml_mul(ctx0, decay_mask, k_block[i]);
                ggml_tensor * decay_q_i = ggml_mul(ctx0, decay_mask, q_block[i]);

                ggml_tensor * Akk_block = ggml_mul_mat(ctx0, decay_k_i, k_j_block);
                ggml_tensor * Aqk_block = ggml_mul_mat(ctx0, decay_q_i, k_j_block);

                Akk_block = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_4d(ctx0, Akk_block, block_size, block_size, n_chunks, HB)));
                Aqk_block = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_4d(ctx0, Aqk_block, block_size, block_size, n_chunks, HB)));

                if (i == j) {
                    Aqk_block = ggml_tri(ctx0, Aqk_block, GGML_TRI_TYPE_LOWER_DIAG);
                    Akk_block = ggml_tri(ctx0, Akk_block, GGML_TRI_TYPE_LOWER);
                }


                kb_row = (kb_row == nullptr) ? Akk_block : ggml_concat(ctx0, kb_row, Akk_block, 0);
                kq_row = (kq_row == nullptr) ? Aqk_block : ggml_concat(ctx0, kq_row, Aqk_block, 0);
            }

            int64_t pad_cols = (n_blocks - j - 1) * block_size;
            if (pad_cols > 0) {
                kb_row = ggml_pad(ctx0, kb_row, pad_cols, 0, 0, 0);
                kq_row = ggml_pad(ctx0, kq_row, pad_cols, 0, 0, 0);
            }

            kb_rows[j] = kb_row;
            kq_rows[j] = kq_row;
        }

        // O(n*log(n)) for binary concat vs O(n^2) for linear concat
        // number of blocks must be power of 2
        // GGML_ASSERT(n_blocks > 0 && (n_blocks & (n_blocks - 1)) == 0);
        // Binary concat rows along dim 1
        {
            int64_t count = n_blocks;
            while (count > 1) {
                int64_t next = 0;
                for (int64_t i = 0; i + 1 < count; i += 2) {
                    kb_rows[next] = ggml_concat(ctx0, kb_rows[i], kb_rows[i + 1], 1);
                    kq_rows[next] = ggml_concat(ctx0, kq_rows[i], kq_rows[i + 1], 1);
                    next++;
                }
                if (count % 2 == 1) {
                    kb_rows[next] = kb_rows[count - 1];
                    kq_rows[next] = kq_rows[count - 1];
                    next++;
                }
                count = next;
            }
        }

        kb = kb_rows[0];
        kq = kq_rows[0];
        kb = ggml_mul(ctx0, kb, b);
        cb(kq, "kq", il);
    } else {
        ggml_tensor * g_cs_i = g_cs;
        ggml_tensor * g_cs_j = ggml_reshape_4d(ctx0, g_cs, 1, CS, n_chunks, H_v * n_seqs);

        g_cs_j = ggml_repeat_4d(ctx0, g_cs_j, CS, CS, n_chunks, H_v * n_seqs);

        // [CS, CS, n_chunks, H_v * n_seqs]
        ggml_tensor * decay_mask;
        decay_mask = ggml_sub(ctx0, g_cs_j, g_cs_i);
        decay_mask = ggml_tri(ctx0, decay_mask, GGML_TRI_TYPE_LOWER_DIAG);
        decay_mask = ggml_exp(ctx0, decay_mask);
        cb(decay_mask, "decay_mask", il);

        // [CS, CS, n_chunks, H_k * n_seqs]
        kb = ggml_mul_mat(ctx0, k,  k_b);
        kb = ggml_mul    (ctx0, kb, decay_mask);
        kb = ggml_tri(ctx0, kb, GGML_TRI_TYPE_LOWER);

        // [CS, CS, n_chunks, H_k * n_seqs]
        kq = ggml_mul_mat(ctx0, k, q);
        kq = ggml_mul(ctx0, kq, decay_mask);
        kq = ggml_tri(ctx0, kq, GGML_TRI_TYPE_LOWER_DIAG);
        cb(kq, "kq", il);
    }

    // [CS, CS, n_chunks, H_k * n_seqs]
    ggml_tensor * attn = kb;
    cb(attn, "attn", il);

    ggml_tensor * identity;
    identity = ggml_view_1d(ctx0, attn, CS, 0);
    identity = ggml_fill   (ctx0, identity, 1.0f);
    identity = ggml_diag   (ctx0, identity);

    ggml_tensor * lhs = ggml_add(ctx0, attn, identity);
    cb(lhs, "dnet_add_ch_lhs", il);

    attn = ggml_neg(ctx0, attn);
    cb(attn, "attn_pre_solve", il);

    ggml_tensor * lin_solve = ggml_solve_tri(ctx0, lhs, attn, true, true, false);
    attn = ggml_add(ctx0, lin_solve, identity);
    cb(attn, "dnet_add_ch_attn_solved", il); // [CS, CS, n_chunks, H_k * n_seqs]

    // [S_v, CS, n_chunks, H_v * n_seqs]
    v = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, v_b)), attn);

    // [CS, 1, n_chunks, H_v * n_seqs] KDA: [CS, S_k, n_chunks, H_v * n_seqs]
    ggml_tensor * g_exp = ggml_exp(ctx0, g_cs);

    k_b = ggml_cont(ctx0, ggml_transpose(ctx0, k_b));

    // [CS, S_k, n_chunks, H_k * n_seqs]
    ggml_tensor * kbg = ggml_mul(ctx0, k_b, g_exp);
    cb(kbg, "k_beta_g_exp", il);

    // [S_k, CS, n_chunks, H_k * n_seqs]
    ggml_tensor * k_cd = ggml_mul_mat(ctx0, kbg, attn);
    cb(k_cd, "k_cumdecay", il);

    // [1, CS, n_chunks, H_k * n_seqs] KDA: [S_k, CS, n_chunks, H_k * n_seqs]
    ggml_tensor * g_exp_t = ggml_cont(ctx0, ggml_transpose(ctx0, g_exp));
    ggml_tensor * q_g_exp = ggml_mul(ctx0, q, g_exp_t);

    // vectorized calculation of key_gdiff
    // improved from the chunked version:
    //   g_last = torch.clamp(g_cum[:, :, -1], max=50.0).exp().unsqueeze(-1).unsqueeze(-1)
    //   g_diff = torch.clamp(g_cum[:, :, -1:] - g_cum, max=50.0).exp()
    //   key_gdiff = key * g_diff.unsqueeze(-1)
    //   kgdmulvnew = (key_gdiff).transpose(-1, -2) @ v_new
    //   last_recurrent_state = last_recurrent_state * g_last + kgdmulvnew

    // get last element in g_cumsum along CS dimension (ne0)
    // example: [[x, y, z, ..., last], ...] -> [[last], ...]
    // [1, 1, n_chunks, H_v * n_seqs] KDA: [1, S_k, n_chunks, H_v * n_seqs]
    ggml_tensor * g_last = ggml_view_4d(ctx0, g_cs, 1, g_cs->ne[1], g_cs->ne[2], g_cs->ne[3],
            g_cs->nb[1],
            g_cs->nb[2],
            g_cs->nb[3],
            ggml_row_size(g_cs->type, g_cs->ne[0] - 1));
    cb(g_last, "g_last", il);

    // TODO: remove this cont when CUDA supports non-cont unary ops
    g_last = ggml_cont(ctx0, g_last);

    // [1, 1, n_chunks, H_v * n_seqs] KDA: [S_k, 1, n_chunks, H_v * n_seqs]
    ggml_tensor * g_last_exp_t = ggml_transpose(ctx0, ggml_exp(ctx0, g_last));
    cb(g_last_exp_t, "g_last_exp_t", il);

    // [CS, 1, n_chunks, H_v * n_seqs] KDA: [CS, S_k, n_chunks, H_v * n_seqs]
    ggml_tensor * g_diff = ggml_neg(ctx0, ggml_sub(ctx0, g_cs, g_last));
    cb(g_diff, "g_diff", il);

    ggml_tensor * g_diff_exp_t = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_exp(ctx0, g_diff)));

    // [S_k, CS, n_chunks, H_v * n_seqs]
    ggml_tensor * kg = ggml_mul(ctx0, k, g_diff_exp_t);
    cb(kg, "key_gdiff", il);

    // [CS, S_k, n_chunks, H_v * n_seqs]
    ggml_tensor * kg_t = ggml_cont(ctx0, ggml_transpose(ctx0, kg));
    cb(kg_t, "key_gdiff_t", il);

    ggml_tensor * s_t = ggml_transpose(ctx0, s);
    s_t = ggml_cont_4d(ctx0, s_t, S_v, S_v, 1, H_v * n_seqs);
    cb(s_t, "dnet_add_ch_state", il);

    // [CS, S_v, n_chunks, H_v * n_seqs]
    ggml_tensor * v_t = ggml_cont(ctx0, ggml_transpose(ctx0, v));

    for (int64_t chunk = 0; chunk < n_chunks; chunk++) {
        ggml_tensor * ch_k_cd    = get_slice_2d(ctx0, k_cd,    chunk); // [S_k,  CS, 1, H_k * n_seqs]
        ggml_tensor * ch_v_t     = get_slice_2d(ctx0, v_t,     chunk); // [ CS, S_v, 1, H_v * n_seqs]
        ggml_tensor * ch_kq      = get_slice_2d(ctx0, kq,      chunk); // [ CS,  CS, 1, H_k * n_seqs]
        ggml_tensor * ch_q_g_exp = get_slice_2d(ctx0, q_g_exp, chunk); // [S_k,  CS, 1, H_k * n_seqs]
        ggml_tensor * ch_kg_t    = get_slice_2d(ctx0, kg_t,    chunk); // [ CS, S_k, 1, H_v * n_seqs]

        // [CS, S_v, 1, H_v * n_seqs]
        ggml_tensor * v_t_p = ggml_mul_mat(ctx0, ch_k_cd, s_t);
        cb(v_t_p, "v_prime", il);

        // [CS, S_v, 1, H_v * n_seqs]
        ggml_tensor * v_t_new = ggml_sub(ctx0, ch_v_t, v_t_p);
        cb(v_t_new, "v_t_new", il);

        // [S_v, CS, 1, H_v * n_seqs]
        ggml_tensor * v_attn = ggml_mul_mat(ctx0, v_t_new, ch_kq);
        cb(v_attn, "v_attn", il);

        // [S_v, CS, 1, H_v * n_seqs]
        ggml_tensor * attn_inter = ggml_mul_mat(ctx0, s_t, ch_q_g_exp);
        cb(attn_inter, "attn_inter", il);

        // [S_v, CS, 1, H_v * n_seqs]
        ggml_tensor * o_ch = ggml_add(ctx0, attn_inter, v_attn);
        cb(o_ch, "dnet_add_ch_attn_out", il);

        v = ggml_set_inplace(ctx0, v, o_ch, v->nb[1], v->nb[2], v->nb[3], chunk * v->nb[2]);

        // kgdmulvnew = (key_gdiff).transpose(-1, -2) @ v_new
        // TODO: head broadcast might not work here - probably will need a transpose
        ggml_tensor * kgv = ggml_mul_mat(ctx0, ch_kg_t, v_t_new); // [S_k, S_v, 1, H_k * n_seqs]

        // last_recurrent_state = last_recurrent_state * g_last + kgdmulvnew
        ggml_tensor * ch_g_last_exp_t = get_slice_2d(ctx0, g_last_exp_t, chunk);

        s_t = ggml_mul(ctx0, s_t, ch_g_last_exp_t);
        s_t = ggml_add(ctx0, s_t, kgv);
        cb(s_t, "dnet_add_ch_state", il);
    }

    s_t = ggml_reshape_4d(ctx0, s_t, S_v, S_v, H_v, n_seqs);

    // truncate padded tokens
    ggml_tensor * o = ggml_view_4d(ctx0, v,
            S_v, n_tokens, H_v, n_seqs,
            ggml_row_size(v->type, S_v),
            ggml_row_size(v->type, S_v * CS * n_chunks),
            ggml_row_size(v->type, S_v * CS * n_chunks * H_v), 0);
    o = ggml_permute  (ctx0, o, 0, 2, 1, 3); // [S_v, H_v, n_tokens, n_seqs]
    s = ggml_transpose(ctx0, s_t);
    cb(s, "output_state", il);

    return {o, s};
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::build_delta_net_autoregressive(
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * b, // beta
        ggml_tensor * s, // state
        int           il) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(n_tokens == 1);

    GGML_ASSERT(S_k == S_v);
    GGML_ASSERT(H_v % H_k == 0);

    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

    const float scale = 1.0f / sqrtf(S_k);

    q = ggml_scale(ctx0, q, scale);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    k = ggml_permute(ctx0, k, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    v = ggml_permute(ctx0, v, 0, 2, 1, 3); // [S_v, n_tokens, H_v, n_seqs]

    cb(q, "q_in", il);
    cb(k, "k_in", il);
    cb(v, "v_in", il);
    cb(b, "b_in", il);
    cb(g, "g_in", il);

    // GDA: [1,  1,  H_v, n_seqs]
    // KDA: [1, S_k, H_v, n_seqs]
    g = ggml_reshape_4d(ctx0, g, 1, g->ne[0], H_v, n_seqs);
    b = ggml_reshape_4d(ctx0, b, 1,        1, H_v, n_seqs);

    // [S_v, S_v, H_v, n_seqs]
    g = ggml_exp(ctx0, g);
    s = ggml_mul(ctx0, s, g);

    ggml_tensor * s_t = ggml_cont(ctx0, ggml_transpose(ctx0, s));

    // [1, S_v, H_v, n_seqs]
    ggml_tensor * sk;
    sk = ggml_mul     (ctx0, s_t, k);
    sk = ggml_sum_rows(ctx0, sk);

    // [S_v, 1, H_v, n_seqs]
    ggml_tensor * d;
    d = ggml_sub(ctx0, v, ggml_transpose(ctx0, sk));
    d = ggml_mul(ctx0, d, b);

    // [1, S_v, H_v, n_seqs]
    ggml_tensor * d_t;
    d_t = ggml_transpose(ctx0, d);

    // [S_v, S_v, H_v, n_seqs]
    ggml_tensor * kd;
    k  = ggml_repeat(ctx0, k, s);
    kd = ggml_mul   (ctx0, k, d_t);

    s_t = ggml_add(ctx0, s_t, kd);

    cb(s_t, "dnet_add_ar_state", il);

    ggml_tensor * s_q = ggml_mul     (ctx0, s_t, q);
    ggml_tensor * o   = ggml_sum_rows(ctx0, s_q);

    o = ggml_permute  (ctx0, o, 2, 0, 1, 3); // [S_v, H_v, n_tokens, n_seqs]
    s = ggml_transpose(ctx0, s_t);           // [S_v, S_v, H_v, n_seqs]

    return {o, s};
}
