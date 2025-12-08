#include "models.h"

#include "llama-impl.h"

llm_graph_context_mamba::llm_graph_context_mamba(const llm_graph_params & params) : llm_graph_context(params) {}

ggml_tensor * llm_graph_context_mamba::build_mamba_layer(llm_graph_input_rs * inp,
                                                         ggml_tensor *        cur,
                                                         const llama_model &  model,
                                                         const llama_ubatch & ubatch,
                                                         int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head = mctx_cur->get_head();

    const auto & layer = model.layers[il];

    const int64_t d_conv         = hparams.ssm_d_conv;
    const int64_t d_inner        = hparams.ssm_d_inner;
    const int64_t d_state        = hparams.ssm_d_state;
    const int64_t dt_rank        = hparams.ssm_dt_rank;
    const int64_t n_head         = d_inner;
    const int64_t head_dim       = 1;
    const int64_t n_seqs         = ubatch.n_seqs;
    // Some variants of Mamba arch (e.g. FalconMamba do apply layer norm on B and Dt layers)
    const bool    ssm_dt_b_c_rms = hparams.ssm_dt_b_c_rms;

    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    conv               = ggml_reshape_3d(ctx0, conv, d_conv - 1, d_inner, n_seqs);

    // {n_embd, n_tokens} => {n_embd, n_seq_tokens, n_seqs}
    cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

    // {n_embd, 2*d_inner} @ {n_embd, n_seq_tokens, n_seqs} => {2*d_inner, n_seq_tokens, n_seqs}
    ggml_tensor * xz = build_lora_mm(layer.ssm_in, cur);
    // split the above in two
    // => {d_inner, n_seq_tokens, n_seqs}
    ggml_tensor * x  = ggml_view_3d(ctx0, xz, d_inner, xz->ne[1], xz->ne[2], xz->nb[1], xz->nb[2], 0);
    ggml_tensor * z =
        ggml_view_3d(ctx0, xz, d_inner, xz->ne[1], xz->ne[2], xz->nb[1], xz->nb[2], d_inner * ggml_element_size(xz));

    // conv
    {
        // => {d_conv - 1 + n_seq_tokens, d_inner, n_seqs}
        ggml_tensor * conv_x = ggml_concat(ctx0, conv, ggml_transpose(ctx0, x), 0);

        // copy last (d_conv - 1) columns back into the state cache
        ggml_tensor * last_conv = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs, conv_x->nb[1], conv_x->nb[2],
                                               n_seq_tokens * (conv_x->nb[0]));

        ggml_build_forward_expand(
            gf, ggml_cpy(ctx0, last_conv,
                         ggml_view_1d(ctx0, conv_states_all, (d_conv - 1) * (d_inner) * (n_seqs),
                                      kv_head * (d_conv - 1) * (d_inner) *ggml_element_size(conv_states_all))));

        // 1D convolution
        // The equivalent is to make a self-overlapping view of conv_x
        // over d_conv columns at each stride in the 3rd dimension,
        // then element-wise multiply that with the conv1d weight,
        // then sum the elements of each row,
        // (the last two steps are a dot product over rows (also doable with mul_mat))
        // then permute away the ne[0] dimension,
        // and then you're left with the resulting x tensor.
        // For simultaneous sequences, all sequences need to have the same length.
        x = ggml_ssm_conv(ctx0, conv_x, layer.ssm_conv1d);

        // bias
        x = ggml_add(ctx0, x, layer.ssm_conv1d_b);

        x = ggml_silu(ctx0, x);
    }

    // ssm
    {
        // {d_inner, dt_rank + 2*d_state} @ {d_inner, n_seq_tokens, n_seqs} => {dt_rank + 2*d_state, n_seq_tokens, n_seqs}
        ggml_tensor * x_db = build_lora_mm(layer.ssm_x, x);
        // split
        ggml_tensor * dt   = ggml_view_3d(ctx0, x_db, dt_rank, n_seq_tokens, n_seqs, x_db->nb[1], x_db->nb[2], 0);
        ggml_tensor * B =
            ggml_view_4d(ctx0, x_db, d_state, /* n_group */ 1, n_seq_tokens, n_seqs, d_state * x_db->nb[0], x_db->nb[1],
                         x_db->nb[2], ggml_element_size(x_db) * dt_rank);
        ggml_tensor * C =
            ggml_view_4d(ctx0, x_db, d_state, /* n_group */ 1, n_seq_tokens, n_seqs, d_state * x_db->nb[0], x_db->nb[1],
                         x_db->nb[2], ggml_element_size(x_db) * (dt_rank + d_state));

        // Some Mamba variants (e.g. FalconMamba, Jamba) apply RMS norm in B, C & Dt layers
        if (ssm_dt_b_c_rms || (layer.ssm_dt_norm && layer.ssm_b_norm && layer.ssm_c_norm)) {
            dt = build_norm(dt, layer.ssm_dt_norm, NULL, LLM_NORM_RMS, il);
            B  = build_norm(B, layer.ssm_b_norm, NULL, LLM_NORM_RMS, il);
            C  = build_norm(C, layer.ssm_c_norm, NULL, LLM_NORM_RMS, il);
        }

        // {dt_rank, d_inner} @ {dt_rank, n_seq_tokens, n_seqs} => {d_inner, n_seq_tokens, n_seqs}
        dt = build_lora_mm(layer.ssm_dt, dt);
        dt = ggml_add(ctx0, dt, layer.ssm_dt_b);

        cur = x;
        x   = ggml_reshape_4d(ctx0, x, head_dim, n_head, n_seq_tokens, n_seqs);

        ggml_tensor * A = layer.ssm_a;

        // use the states and the indices provided by build_recurrent_state
        // (this is necessary in order to properly use the states before they are overwritten,
        //  while avoiding to make unnecessary copies of the states)
        auto get_ssm_rows = [&](ggml_context * ctx, ggml_tensor * states, ggml_tensor * ids) {
            ggml_tensor * ssm = ggml_reshape_4d(ctx, states, d_state, head_dim, n_head, mctx_cur->get_size());

            // Custom operator to optimize the parallel associative scan
            // as described in the Annex D of the Mamba paper.
            // => {d_inner, n_seq_tokens, n_seqs} and {d_state, d_inner, n_seqs}
            return ggml_ssm_scan(ctx, ssm, x, dt, A, B, C, ids);
        };

        ggml_tensor * y_ssm = build_rs(inp, ssm_states_all, hparams.n_embd_s(), ubatch.n_seqs, get_ssm_rows);

        // store last states
        ggml_build_forward_expand(
            gf, ggml_cpy(ctx0, ggml_view_1d(ctx0, y_ssm, d_state * d_inner * n_seqs, x->nb[3] * x->ne[3]),
                         ggml_view_1d(ctx0, ssm_states_all, d_state * d_inner * n_seqs,
                                      kv_head * d_state * d_inner * ggml_element_size(ssm_states_all))));

        ggml_tensor * y = ggml_view_3d(ctx0, y_ssm, d_inner, n_seq_tokens, n_seqs, x->nb[2], x->nb[3], 0);

        // TODO: skip computing output earlier for unused tokens

        y = ggml_add(ctx0, y, ggml_mul(ctx0, cur, layer.ssm_d));
        y = ggml_swiglu_split(ctx0, ggml_cont(ctx0, z), y);

        // {d_inner, n_embd} @ {d_inner, n_seq_tokens, n_seqs} => {n_embd, n_seq_tokens, n_seqs}
        cur = build_lora_mm(layer.ssm_out, y);
    }

    // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llm_graph_context_mamba::build_mamba2_layer(llm_graph_input_rs * inp,
                                                          ggml_tensor *        cur,
                                                          const llama_model &  model,
                                                          const llama_ubatch & ubatch,
                                                          int                  il) const {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head = mctx_cur->get_head();

    const int64_t d_conv   = hparams.ssm_d_conv;
    const int64_t d_inner  = hparams.ssm_d_inner;
    const int64_t d_state  = hparams.ssm_d_state;
    const int64_t n_head   = hparams.ssm_dt_rank;
    const int64_t head_dim = d_inner / n_head;
    const int64_t n_group  = hparams.ssm_n_group;
    const int64_t n_seqs   = ubatch.n_seqs;

    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    conv               = ggml_reshape_3d(ctx0, conv, d_conv - 1, d_inner + 2 * n_group * d_state, n_seqs);

    // {n_embd, n_tokens} => {n_embd, n_seq_tokens, n_seqs}
    cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

    // d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads

    // {n_embd, d_in_proj} @ {n_embd, n_seq_tokens, n_seqs} => {d_in_proj, n_seq_tokens, n_seqs}
    ggml_tensor * zxBCdt = build_lora_mm(model.layers[il].ssm_in, cur);

    // split the above in three
    ggml_tensor * z   = ggml_view_4d(ctx0, zxBCdt, head_dim, n_head, n_seq_tokens, n_seqs, head_dim * zxBCdt->nb[0],
                                     zxBCdt->nb[1], zxBCdt->nb[2], 0);
    ggml_tensor * xBC = ggml_view_3d(ctx0, zxBCdt, d_inner + 2 * n_group * d_state, n_seq_tokens, n_seqs, zxBCdt->nb[1],
                                     zxBCdt->nb[2], d_inner * ggml_element_size(zxBCdt));
    ggml_tensor * dt  = ggml_view_3d(ctx0, zxBCdt, n_head, n_seq_tokens, n_seqs, zxBCdt->nb[1], zxBCdt->nb[2],
                                     (2 * d_inner + 2 * n_group * d_state) * ggml_element_size(zxBCdt));

    // conv
    {
        // => {d_conv - 1 + n_seq_tokens, d_inner + 2*n_group*d_state, n_seqs}
        ggml_tensor * conv_x = ggml_concat(ctx0, conv, ggml_transpose(ctx0, xBC), 0);

        // copy last (d_conv - 1) columns back into the state cache
        ggml_tensor * last_conv = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner + 2 * n_group * d_state, n_seqs,
                                               conv_x->nb[1], conv_x->nb[2], n_seq_tokens * (conv_x->nb[0]));

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_conv,
                                               ggml_view_1d(ctx0, conv_states_all,
                                                            (d_conv - 1) * (d_inner + 2 * n_group * d_state) * (n_seqs),
                                                            kv_head * (d_conv - 1) * (d_inner + 2 * n_group * d_state) *
                                                                ggml_element_size(conv_states_all))));

        // 1D convolution
        // The equivalent is to make a self-overlapping view of conv_x
        // over d_conv columns at each stride in the 3rd dimension,
        // then element-wise multiply that with the conv1d weight,
        // then sum the elements of each row,
        // (the last two steps are a dot product over rows (also doable with mul_mat))
        // then permute away the ne[0] dimension,
        // and then you're left with the resulting x tensor.
        // For simultaneous sequences, all sequences need to have the same length.
        xBC = ggml_ssm_conv(ctx0, conv_x, model.layers[il].ssm_conv1d);

        // bias
        xBC = ggml_add(ctx0, xBC, model.layers[il].ssm_conv1d_b);

        xBC = ggml_silu(ctx0, xBC);
    }

    //DEBUG
    // cur = ggml_view_4d(ctx0, xBC, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3],
    //     cur->nb[1], cur->nb[2], cur->nb[3], 0);

    // ssm
    {
        // These correspond to V K Q in SSM/attention duality
        ggml_tensor * x = ggml_view_4d(ctx0, xBC, head_dim, n_head, n_seq_tokens, n_seqs, head_dim * xBC->nb[0],
                                       xBC->nb[1], xBC->nb[2], 0);
        ggml_tensor * B = ggml_view_4d(ctx0, xBC, d_state, n_group, n_seq_tokens, n_seqs, d_state * xBC->nb[0],
                                       xBC->nb[1], xBC->nb[2], d_inner * ggml_element_size(xBC));
        ggml_tensor * C = ggml_view_4d(ctx0, xBC, d_state, n_group, n_seq_tokens, n_seqs, d_state * xBC->nb[0],
                                       xBC->nb[1], xBC->nb[2], (d_inner + n_group * d_state) * ggml_element_size(xBC));

        // {n_head, n_seq_tokens, n_seqs}
        dt = ggml_add(ctx0, ggml_cont(ctx0, dt), model.layers[il].ssm_dt_b);

        ggml_tensor * A = model.layers[il].ssm_a;

        // use the states and the indices provided by build_recurrent_state
        // (this is necessary in order to properly use the states before they are overwritten,
        //  while avoiding to make unnecessary copies of the states)
        auto get_ssm_rows = [&](ggml_context * ctx, ggml_tensor * states, ggml_tensor * ids) {
            ggml_tensor * ssm = ggml_reshape_4d(ctx, states, d_state, head_dim, n_head, mctx_cur->get_size());

            // Use SSM_SCAN op for all cases - the Metal kernel handles both
            // single-token (sequential scan) and multi-token (SSD formulation) internally
            if (true) {
            // if (n_seq_tokens == 1) {
                //DEBUG
                LLAMA_LOG_DEBUG("build_mamba2_layer(layer %d): single-token update\n", il);
                // If single-token, use ssm_scan op
                ssm = ggml_cast(ctx, ssm, GGML_TYPE_F32);
                return ggml_ssm_scan(ctx, ssm, x, dt, A, B, C, ids);
            } else {
                //DEBUG
                LLAMA_LOG_DEBUG("build_mamba2_layer(layer %d): multi-token chunk scan\n", il);

                // ============================================================
                // OPTIMIZED SSD (State Space Duality) Implementation
                // Key optimizations:
                // 1. Pre-permute B and C once at the start to avoid repeated permutes
                // 2. Minimize ggml_cont() calls by choosing layouts carefully
                // 3. Reuse intermediate tensors where possible
                // ============================================================

                // extract the state(s) for the sequences identified by ids
                if (ssm->ne[3] != ids->ne[0]) {
                    ggml_tensor * ssm_perm = ggml_permute(ctx, ssm, 0, 2, 3, 1);
                    ggml_tensor * ids_perm_rep = ggml_repeat_4d(ctx, ids,
                        ids->ne[0], ssm->ne[1], ssm->ne[2], 1);
                    ggml_tensor * ssm_ids = ggml_get_rows(ctx, ssm_perm, ids_perm_rep);
                    ssm = ggml_cont(ctx, ggml_permute(ctx, ssm_ids, 0, 3, 1, 2));
                    GGML_ASSERT(ssm->ne[3] == ids->ne[0]);
                }
                // ssm -> {d_state, head_dim, n_head, n_seqs}

                // step 1: compute dt softplus
                ggml_tensor * dt_softplus = ggml_softplus(ctx, dt); // {n_head, n_seq_tokens, n_seqs}
                dt_softplus = ggml_clamp(ctx, dt_softplus, 0.001, 100.0);
                cb(dt_softplus, "dt_softplus", il);

                // step 2: compute dtA and dtX
                // dtA: {n_head, n_seq_tokens, n_seqs}
                ggml_tensor * dtA = ggml_mul(ctx, dt_softplus, ggml_reshape_1d(ctx, A, A->ne[1]));
                cb(dtA, "dtA", il);

                // dtX: {head_dim, n_head, n_seq_tokens, n_seqs}
                ggml_tensor * dtX = ggml_mul(ctx, x, ggml_reshape_4d(ctx, dt_softplus,
                    1, dt_softplus->ne[0], dt_softplus->ne[1], dt_softplus->ne[2]));
                cb(dtX, "dtX", il);

                // Pre-permute dtX once for the attention-like matmul: {head_dim, n_head, n_seq_tokens, n_seqs} -> {n_head, n_seq_tokens, head_dim, n_seqs}
                // This layout is what we need for: y = SAM @ dtX^T
                ggml_tensor * dtX_perm = ggml_cont(ctx, ggml_permute(ctx, dtX, 1, 2, 0, 3)); // {n_head, n_seq_tokens, head_dim, n_seqs}
                cb(dtX_perm, "dtX_perm", il);

                // Pre-permute B and C for mul_mat: {d_state, n_group, n_seq_tokens, n_seqs} -> {d_state, n_seq_tokens, n_group, n_seqs}
                // These permuted versions will be used throughout
                ggml_tensor * B_perm_full = ggml_permute(ctx, B, 0, 2, 1, 3); // {d_state, n_seq_tokens, n_group, n_seqs}
                ggml_tensor * C_perm_full = ggml_permute(ctx, C, 0, 2, 1, 3); // {d_state, n_seq_tokens, n_group, n_seqs}

                uint32_t repeats = n_head / n_group;

                // For the state update, we need B in a different layout
                // Pre-compute the expanded B for state updates: {d_state, n_seq_tokens, n_head, n_seqs}
                ggml_tensor * B_for_state = ggml_cont(ctx, B_perm_full);
                B_for_state = ggml_repeat_4d(ctx, B_for_state,
                    B_for_state->ne[0], B_for_state->ne[1], B_for_state->ne[2] * repeats, B_for_state->ne[3]);
                // Permute for mul_mat: {n_seq_tokens, d_state, n_head, n_seqs}
                ggml_tensor * B_for_state_perm = ggml_cont(ctx, ggml_permute(ctx, B_for_state, 1, 0, 2, 3));
                cb(B_for_state_perm, "B_for_state_perm", il);

                // Empty y that will be extended with each chunk of tokens
                ggml_tensor * y = ggml_new_tensor_4d(ctx, x->type, x->ne[0], x->ne[1], 0, x->ne[3]);

                const uint32_t chunk_size = 512;
                for (auto chunk_i = 0; chunk_i < n_seq_tokens; chunk_i += chunk_size) {
                    const auto chunk_size_i = std::min(chunk_size, uint32_t(n_seq_tokens - chunk_i));

                    // Create chunk views
                    ggml_tensor * dtA_chunk = (chunk_size_i == n_seq_tokens) ? dtA :
                        ggml_view_3d(ctx, dtA, dtA->ne[0], chunk_size_i, dtA->ne[2],
                            dtA->nb[1], dtA->nb[2], chunk_i * dtA->nb[1]);

                    ggml_tensor * dtX_chunk_perm = (chunk_size_i == n_seq_tokens) ? dtX_perm :
                        ggml_view_4d(ctx, dtX_perm,
                            dtX_perm->ne[0], chunk_size_i, dtX_perm->ne[2], dtX_perm->ne[3],
                            dtX_perm->nb[1], dtX_perm->nb[2], dtX_perm->nb[3],
                            chunk_i * dtX_perm->nb[1]);

                    ggml_tensor * dtX_chunk = (chunk_size_i == n_seq_tokens) ? dtX :
                        ggml_view_4d(ctx, dtX, dtX->ne[0], dtX->ne[1], chunk_size_i, dtX->ne[3],
                            dtX->nb[1], dtX->nb[2], dtX->nb[3], chunk_i * dtX->nb[2]);

                    // Use pre-permuted B and C chunks
                    ggml_tensor * B_perm_chunk = (chunk_size_i == n_seq_tokens) ? B_perm_full :
                        ggml_view_4d(ctx, B_perm_full,
                            B_perm_full->ne[0], chunk_size_i, B_perm_full->ne[2], B_perm_full->ne[3],
                            B_perm_full->nb[1], B_perm_full->nb[2], B_perm_full->nb[3],
                            chunk_i * B_perm_full->nb[1]);

                    ggml_tensor * C_perm_chunk = (chunk_size_i == n_seq_tokens) ? C_perm_full :
                        ggml_view_4d(ctx, C_perm_full,
                            C_perm_full->ne[0], chunk_size_i, C_perm_full->ne[2], C_perm_full->ne[3],
                            C_perm_full->nb[1], C_perm_full->nb[2], C_perm_full->nb[3],
                            chunk_i * C_perm_full->nb[1]);

                    ggml_tensor * B_state_chunk = (chunk_size_i == n_seq_tokens) ? B_for_state_perm :
                        ggml_view_4d(ctx, B_for_state_perm,
                            chunk_size_i, B_for_state_perm->ne[1], B_for_state_perm->ne[2], B_for_state_perm->ne[3],
                            B_for_state_perm->nb[1], B_for_state_perm->nb[2], B_for_state_perm->nb[3],
                            chunk_i * B_for_state_perm->nb[0]);

                    cb(dtA_chunk, "dtA_chunk", il);
                    cb(dtX_chunk_perm, "dtX_chunk_perm", il);

                    // step 3: compute CB = C @ B^T
                    // B_perm_chunk, C_perm_chunk: {d_state, chunk_size_i, n_group, n_seqs}
                    ggml_tensor * CB = ggml_mul_mat(ctx, B_perm_chunk, C_perm_chunk); // {chunk_size_i, chunk_size_i, n_group, n_seqs}
                    CB = ggml_repeat_4d(ctx, CB, CB->ne[0], CB->ne[1], CB->ne[2] * repeats, CB->ne[3]);
                    cb(CB, "CB", il);

                    // step 4: compute decay matrix
                    // dtA_chunk: {n_head, chunk_size_i, n_seqs}
                    // We need to build the lower-triangular cumsum matrix
                    ggml_tensor * dtA_for_decay = ggml_permute(ctx, dtA_chunk, 2, 1, 3, 0); // {1, chunk_size_i, n_head, n_seqs}
                    ggml_tensor * dtA_expanded = ggml_repeat_4d(ctx, dtA_for_decay,
                        dtA_for_decay->ne[0] * chunk_size_i, dtA_for_decay->ne[1],
                        dtA_for_decay->ne[2], dtA_for_decay->ne[3]);
                    ggml_tensor * dtA_tri = ggml_tri(ctx, dtA_expanded, GGML_TRI_TYPE_LOWER);
                    ggml_tensor * segsum = ggml_cumsum(ctx, dtA_tri);
                    segsum = ggml_cont(ctx, ggml_transpose(ctx, segsum)); // Need cont for transpose
                    ggml_tensor * decay = ggml_exp(ctx, segsum);
                    cb(decay, "decay", il);

                    // step 5: compute surrogate attention matrix
                    ggml_tensor * CBdecay = ggml_mul(ctx, CB, decay);
                    ggml_tensor * SAM = ggml_tri(ctx, CBdecay, GGML_TRI_TYPE_LOWER_DIAG);
                    cb(SAM, "SAM", il);

                    // step 6: compute y = SAM @ dtX^T
                    // SAM: {chunk_size_i, chunk_size_i, n_head, n_seqs}
                    // dtX_chunk_perm: {n_head, chunk_size_i, head_dim, n_seqs}
                    ggml_tensor * y_chunk = ggml_mul_mat(ctx, dtX_chunk_perm, SAM);
                    // Result: {head_dim, chunk_size_i, n_head, n_seqs}
                    // We need: {head_dim, n_head, chunk_size_i, n_seqs}
                    y_chunk = ggml_cont(ctx, ggml_permute(ctx, y_chunk, 0, 2, 1, 3));
                    cb(y_chunk, "y_chunk", il);

                    // step 7: compute state update contribution
                    // decay_last: last row of decay matrix
                    ggml_tensor * decay_last = ggml_view_4d(ctx, decay,
                        decay->ne[0], 1, decay->ne[2], decay->ne[3],
                        decay->nb[1], decay->nb[2], decay->nb[3],
                        (decay->ne[1] - 1) * decay->nb[1]);
                    // decay_last: {chunk_size_i, 1, n_head, n_seqs} -> need {1, n_head, chunk_size_i, n_seqs} for broadcast
                    ggml_tensor * decay_last_bc = ggml_cont(ctx, ggml_permute(ctx, decay_last, 2, 0, 1, 3));

                    // dtxdecay = dtX * decay_last (broadcast)
                    // dtX_chunk: {head_dim, n_head, chunk_size_i, n_seqs}
                    ggml_tensor * dtxdecay = ggml_mul(ctx, dtX_chunk, decay_last_bc);
                    // Permute for mul_mat: {n_head, chunk_size_i, head_dim, n_seqs}
                    ggml_tensor * dtxdecay_perm = ggml_cont(ctx, ggml_permute(ctx, dtxdecay, 1, 2, 0, 3));

                    // step 8: compute next_state = B^T @ dtxdecay
                    // B_state_chunk: {chunk_size_i, d_state, n_head, n_seqs}
                    // dtxdecay_perm: {n_head, chunk_size_i, head_dim, n_seqs}
                    ggml_tensor * next_state = ggml_mul_mat(ctx, B_state_chunk, dtxdecay_perm);
                    // Result: {d_state, head_dim, n_head, n_seqs}
                    if (next_state->type != ssm->type) {
                        next_state = ggml_cast(ctx, next_state, ssm->type);
                    }
                    cb(next_state, "next_state", il);

                    // step 9: update state from previous state
                    // Compute exp(cumsum(dtA)) for state decay
                    ggml_tensor * dtA_for_state = ggml_cont(ctx, dtA_for_decay);
                    ggml_tensor * dtA_flat = ggml_view_3d(ctx, dtA_for_state,
                        dtA_for_state->ne[1], dtA_for_state->ne[2], dtA_for_state->ne[3],
                        dtA_for_state->nb[2], dtA_for_state->nb[3], 0);
                    ggml_tensor * exp_dtA_cumsum = ggml_exp(ctx, ggml_cumsum(ctx, dtA_flat));
                    exp_dtA_cumsum = ggml_view_4d(ctx, exp_dtA_cumsum,
                        1, dtA_for_state->ne[1], dtA_for_state->ne[2], dtA_for_state->ne[3],
                        dtA_for_state->nb[1], dtA_for_state->nb[2], dtA_for_state->nb[3], 0);

                    // Get last value for state update
                    ggml_tensor * state_decay = ggml_view_4d(ctx, exp_dtA_cumsum,
                        exp_dtA_cumsum->ne[0], 1, exp_dtA_cumsum->ne[2], exp_dtA_cumsum->ne[3],
                        exp_dtA_cumsum->nb[1], exp_dtA_cumsum->nb[2], exp_dtA_cumsum->nb[3],
                        (exp_dtA_cumsum->ne[1] - 1) * exp_dtA_cumsum->nb[1]);

                    next_state = ggml_add(ctx, next_state, ggml_mul(ctx, ssm, ggml_cont(ctx, state_decay)));
                    cb(next_state, "next_state_updated", il);

                    // step 10: update y from previous state
                    // y_prev = C @ ssm (project state through C)
                    // C_perm_chunk: {d_state, chunk_size_i, n_group, n_seqs}
                    // ssm: {d_state, head_dim, n_head, n_seqs}
                    ggml_tensor * y_prev = ggml_mul_mat(ctx, C_perm_chunk, ssm);
                    // Result: {chunk_size_i, head_dim, n_head, n_seqs}
                    // Need: {head_dim, n_head, chunk_size_i, n_seqs}
                    y_prev = ggml_cont(ctx, ggml_permute(ctx, y_prev, 2, 0, 1, 3));

                    // Scale by cumulative decay
                    // exp_dtA_cumsum: {1, chunk_size_i, n_head, n_seqs}
                    // Need: {1, n_head, chunk_size_i, n_seqs} for broadcast
                    ggml_tensor * y_decay = ggml_cont(ctx, ggml_permute(ctx, exp_dtA_cumsum, 0, 2, 1, 3));
                    y_prev = ggml_mul(ctx, y_prev, y_decay);

                    y_chunk = ggml_add(ctx, y_chunk, y_prev);
                    cb(y_chunk, "y_chunk_final", il);

                    // step 11: accumulate results
                    if (chunk_size_i == n_seq_tokens) {
                        y = y_chunk;
                    } else {
                        y = ggml_concat(ctx, y, y_chunk, 2);
                    }
                    ssm = next_state;
                }

                // Concat the output y and state
                if (ssm->type != y->type) {
                    ssm = ggml_cast(ctx, ssm, y->type);
                }
                ggml_tensor * out = ggml_concat(ctx,
                    ggml_view_1d(ctx, y, ggml_nelements(y), 0),
                    ggml_view_1d(ctx, ssm, ggml_nelements(ssm), 0),
                    0);
                return out;
            }
        };

        ggml_tensor * y_ssm = build_rs(inp, ssm_states_all, hparams.n_embd_s(), ubatch.n_seqs, get_ssm_rows);

        // store last states
        ggml_build_forward_expand(
            gf, ggml_cpy(ctx0, ggml_view_1d(ctx0, y_ssm, d_state * d_inner * n_seqs, ggml_nelements(x) * x->nb[0]),
                         ggml_view_1d(ctx0, ssm_states_all, d_state * d_inner * n_seqs,
                                      kv_head * d_state * d_inner * ggml_element_size(ssm_states_all))));

        ggml_tensor * y = ggml_view_4d(ctx0, y_ssm, head_dim, n_head, n_seq_tokens, n_seqs, x->nb[1], n_head * x->nb[1],
                                       n_seq_tokens * n_head * x->nb[1], 0);

        // TODO: skip computing output earlier for unused tokens

        y = ggml_add(ctx0, y, ggml_mul(ctx0, x, model.layers[il].ssm_d));
        cb(y, "mamba2_y_add_d", il);
        y = ggml_swiglu_split(ctx0, ggml_cont(ctx0, z), y);

        // grouped RMS norm
        if (model.layers[il].ssm_norm) {
            y = ggml_reshape_4d(ctx0, y, d_inner / n_group, n_group, n_seq_tokens, n_seqs);
            y = build_norm(y, model.layers[il].ssm_norm, NULL, LLM_NORM_RMS, il);
        }

        y = ggml_reshape_3d(ctx0, y, d_inner, n_seq_tokens, n_seqs);

        // {d_inner, n_embd} @ {d_inner, n_seq_tokens, n_seqs} => {n_embd, n_seq_tokens, n_seqs}
        cur = build_lora_mm(model.layers[il].ssm_out, y);
    }

    // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens * n_seqs);
    cb(cur, "mamba_out", il);

    return cur;
}
