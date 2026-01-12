#include "models.h"
#include "ggml.h"
#include <cmath>
#include <utility>
#include <cassert>

llm_graph_context_delta::llm_graph_context_delta(const llm_graph_params & params) : llm_graph_context_mamba(params) {}

/**
 * Unified Delta Net implementation supporting both GDA and KDA modes.
 *
 * GDA (Gated Delta Attention): g has shape [H, T, B] in GGML (PyTorch: [B, T, H])
 *   - Per-head gating, broadcasts over K dimension
 *
 * KDA (Key-wise Delta Attention): g has shape [K, H, T, B] in GGML (PyTorch: [B, T, H, K])
 *   - Per-key gating
 *
 * The mode is auto-detected based on g's dimensionality.
 *
 * Tensor dimension convention:
 *   GGML: ne[0] is innermost (fastest varying), ne[3] is outermost
 *   PyTorch: dim 0 is outermost, dim -1 is innermost
 *   So GGML [A, B, C, D] corresponds to PyTorch [D, C, B, A]
 */

// Helper to get a slice along dimension 2 (n_chunks dimension)
static ggml_tensor * get_slice_2d(ggml_context * ctx, ggml_tensor * t, int64_t chunk) {
    return ggml_view_4d(ctx, t,
        t->ne[0], t->ne[1], 1, t->ne[3],
        t->nb[1], t->nb[2], t->nb[3],
        chunk * t->nb[2]);
}

/**
 * Unified chunked Delta Net implementation.
 *
 * Input tensor format matches qwen3next conventions:
 * @param q         Query tensor [S_k, H_k, n_tokens, n_seqs]
 * @param k         Key tensor [S_k, H_k, n_tokens, n_seqs]
 * @param v         Value tensor [S_v, H_v, n_tokens, n_seqs]
 * @param g         Gate tensor:
 *                    GDA: [H_v, n_tokens, n_seqs]
 *                    KDA: [S_k, H_v, n_tokens, n_seqs]
 * @param beta      Beta tensor [H_v, 1, n_tokens, n_seqs]
 * @param state     State tensor [S_v, S_v * H_v, 1, n_seqs]
 * @param causal_mask   Lower triangular mask [chunk_size, chunk_size]
 * @param identity      Identity matrix [chunk_size, chunk_size]
 * @param diag_mask     Diagonal mask [chunk_size, chunk_size]
 * @param il            Layer index (for debugging callbacks)
 * @param chunk_size    Chunk size for chunked processing
 * @param eps_norm      Epsilon for L2 normalization
 *
 * @return Pair of (output_tokens, new_state)
 */
std::pair<ggml_tensor *, ggml_tensor *> llm_graph_context_delta::build_delta_net_unified_chunking(
        ggml_context * ctx0,
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * beta,
        ggml_tensor * state_reshaped,
        ggml_tensor * causal_mask,
        ggml_tensor * identity,
        ggml_tensor * diag_mask,
        int           il,
        int64_t       chunk_size,
        float         eps_norm) {

    // Input format: [S, H, n_tokens, n_seqs] (matching qwen3next convention)
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    // Detect KDA vs GDA based on g's shape
    // GDA: g has shape [H_v, n_tokens, n_seqs]
    // KDA: g has shape [S_k, H_v, n_tokens, n_seqs] (4D with ne[0]=S_k)
    const bool is_kda = (g->ne[0] == S_k && g->ne[1] == H_v);

    // Validate tensor shapes
    GGML_ASSERT(v->ne[2] == n_tokens);
    GGML_ASSERT(k->ne[2] == n_tokens);
    GGML_ASSERT(state_reshaped->ne[0] == S_v && state_reshaped->ne[1] == S_v && state_reshaped->ne[2] == H_v && state_reshaped->ne[3] == n_seqs);
    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(beta->ne[0] == H_v && beta->ne[2] == n_tokens && beta->ne[3] == n_seqs);
    GGML_ASSERT(H_k == H_v);

    if (is_kda) {
        // KDA: g shape [S_k, H_v, n_tokens, n_seqs]
        GGML_ASSERT(g->ne[0] == S_k && g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    } else {
        // GDA: g shape [H_v, n_tokens, n_seqs]
        GGML_ASSERT(g->ne[0] == H_v && g->ne[1] == n_tokens && g->ne[2] == n_seqs);
    }

    // L2 normalize q and k
    q = ggml_l2_norm(ctx0, q, eps_norm);
    k = ggml_l2_norm(ctx0, k, eps_norm);

    const float scale = 1.0f / sqrtf((float)S_v);
    q = ggml_scale(ctx0, q, scale);

    beta = ggml_sigmoid(ctx0, beta);

    cb(q, "q_in", il);
    cb(k, "k_in", il);
    cb(v, "v_in", il);
    cb(beta, "beta_in", il);
    cb(g, "g_in", il);

    // Permute tensors to working format [S, n_tokens, H, n_seqs]
    // Input: [S, H, n_tokens, n_seqs] -> permute(0, 2, 1, 3) -> [S, n_tokens, H, n_seqs]
    q = ggml_cont_4d(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3), S_k, n_tokens, H_k, n_seqs);
    k = ggml_cont_4d(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3), S_k, n_tokens, H_k, n_seqs);
    v = ggml_cont_4d(ctx0, ggml_permute(ctx0, v, 0, 2, 1, 3), S_v, n_tokens, H_v, n_seqs);

    // Permute g based on mode
    if (is_kda) {
        // KDA: g [S_k, H_v, n_tokens, n_seqs] -> [S_k, n_tokens, H_k, n_seqs]
        g = ggml_cont_4d(ctx0, ggml_permute(ctx0, g, 0, 2, 1, 3), S_k, n_tokens, H_k, n_seqs);
    } else {
        // GDA: g [H_v, n_tokens, n_seqs] -> [n_tokens, 1, H_k, n_seqs]
        g = ggml_cont_4d(ctx0, ggml_permute(ctx0, g, 2, 0, 3, 1), n_tokens, 1, H_k, n_seqs);
    }

    beta = ggml_cont(ctx0, ggml_permute(ctx0, beta, 2, 0, 1, 3));

    cb(q, "q_perm", il);
    cb(k, "k_perm", il);
    cb(v, "v_perm", il);
    cb(beta, "beta_perm", il);
    cb(g, "g_perm", il);
    cb(state_reshaped, "state_in", il);

    // Padding for chunk processing
    const int64_t pad = (chunk_size - n_tokens % chunk_size) % chunk_size;
    const int64_t n_chunks = (n_tokens + pad) / chunk_size;

    q = ggml_pad(ctx0, q, 0, pad, 0, 0);
    k = ggml_pad(ctx0, k, 0, pad, 0, 0);
    v = ggml_pad(ctx0, v, 0, pad, 0, 0);
    beta = ggml_pad(ctx0, beta, 0, pad, 0, 0);

    if (is_kda) {
        // KDA: g shape [S_k, n_tokens, H_k, n_seqs] -> pad along dim 1
        g = ggml_pad(ctx0, g, 0, pad, 0, 0);
    } else {
        // GDA: g shape [n_tokens, 1, H_k, n_seqs] -> pad along dim 0
        g = ggml_pad(ctx0, g, pad, 0, 0, 0);
    }

    cb(q, "q_pad", il);
    cb(k, "k_pad", il);
    cb(v, "v_pad", il);
    cb(beta, "beta_pad", il);
    cb(g, "g_pad", il);

    ggml_tensor * v_beta = ggml_mul(ctx0, v, beta);
    ggml_tensor * k_beta = ggml_mul(ctx0, k, beta);

    cb(v_beta, "v_beta", il);
    cb(k_beta, "k_beta", il);

    // Reshape to chunks
    q      = ggml_reshape_4d(ctx0, q,      S_k, chunk_size, n_chunks, H_k * n_seqs);
    k      = ggml_reshape_4d(ctx0, k,      S_k, chunk_size, n_chunks, H_k * n_seqs);
    k_beta = ggml_reshape_4d(ctx0, k_beta, S_k, chunk_size, n_chunks, H_k * n_seqs);
    v      = ggml_reshape_4d(ctx0, v,      S_v, chunk_size, n_chunks, H_v * n_seqs);
    v_beta = ggml_reshape_4d(ctx0, v_beta, S_v, chunk_size, n_chunks, H_v * n_seqs);
    beta   = ggml_reshape_4d(ctx0, beta, 1, chunk_size, n_chunks, H_k * n_seqs);

    // Reshape g for chunks
    ggml_tensor * g_cumsum;
    if (is_kda) {
        // KDA: g [S_k, n_tokens+pad, H_k, n_seqs] -> [S_k, chunk_size, n_chunks, H_k * n_seqs]
        g = ggml_reshape_4d(ctx0, g, S_k, chunk_size, n_chunks, H_k * n_seqs);
        // Cumsum along chunk_size dimension (ne[1])
        // GGML cumsum operates on ne[0], so we need to transpose, cumsum, transpose back
        g = ggml_cont(ctx0, ggml_transpose(ctx0, g));  // [chunk_size, S_k, n_chunks, H_k * n_seqs]
        g_cumsum = ggml_cumsum(ctx0, g);
        g_cumsum = ggml_cont(ctx0, ggml_transpose(ctx0, g_cumsum));  // [S_k, chunk_size, n_chunks, H_k * n_seqs]
    } else {
        // GDA: g [n_tokens+pad, 1, H_k, n_seqs] -> [chunk_size, 1, n_chunks, H_k * n_seqs]
        g = ggml_reshape_4d(ctx0, g, chunk_size, 1, n_chunks, H_k * n_seqs);
        g_cumsum = ggml_cumsum(ctx0, g);
    }

    cb(g_cumsum, "g_cumsum", il);

    // Build attention matrix A for the WY representation solve
    // For GDA: A[j,i] = sum_k(k[j,k] * exp(g[j] - g[i]) * k[i,k]) = (k @ k^T) * exp(g[j] - g[i])
    // For KDA: A[j,i] = sum_k(k[j,k] * exp(g[j,k] - g[i,k]) * k[i,k])
    //                 = sum_k(k[j,k] * exp(g[j,k]) * k[i,k] * exp(-g[i,k]))
    //                 = (k * exp(g))^T @ (k * exp(-g))
    // The KDA formulation factorizes into a matmul, avoiding the need for a 5D decay mask!

    ggml_tensor * k_decay;
    ggml_tensor * decay_mask = nullptr;
    ggml_tensor * g_exp_pos = nullptr;  // For KDA: exp(g_cumsum)
    ggml_tensor * g_exp_neg = nullptr;  // For KDA: exp(-g_cumsum)

    if (is_kda) {
        // KDA: Exact computation using factorization
        // k_pos_beta = k_beta * exp(g_cumsum), k_neg = k * exp(-g_cumsum)
        // A = k_pos_beta^T @ k_neg (via mul_mat)
        g_exp_pos = ggml_exp(ctx0, g_cumsum);
        g_exp_neg = ggml_exp(ctx0, ggml_neg(ctx0, g_cumsum));
        ggml_tensor * k_pos_beta = ggml_mul(ctx0, k_beta, g_exp_pos);
        ggml_tensor * k_neg = ggml_mul(ctx0, k, g_exp_neg);
        k_decay = ggml_mul_mat(ctx0, k_pos_beta, k_neg);
    } else {
        // GDA: Use decay mask approach (g broadcasts over K dimension)
        // g_cumsum [chunk_size, 1, n_chunks, H_v * n_seqs]
        ggml_tensor * gcs_i = g_cumsum;
        ggml_tensor * gcs_j = ggml_reshape_4d(ctx0, g_cumsum, 1, chunk_size, n_chunks, H_v * n_seqs);
        ggml_tensor * gcs_j_broadcast = ggml_repeat_4d(ctx0, gcs_j, chunk_size, chunk_size, n_chunks, H_v * n_seqs);
        decay_mask = ggml_sub(ctx0, gcs_j_broadcast, gcs_i);

        cb(decay_mask, "decay_mask", il);

        decay_mask = ggml_mul(ctx0, decay_mask, diag_mask);
        decay_mask = ggml_exp(ctx0, decay_mask);
        decay_mask = ggml_mul(ctx0, decay_mask, diag_mask);

        ggml_tensor * kmulkbeta = ggml_mul_mat(ctx0, k, k_beta);
        k_decay = ggml_mul(ctx0, kmulkbeta, decay_mask);
    }

    ggml_tensor * attn = ggml_neg(ctx0, ggml_mul(ctx0, k_decay, causal_mask));

    cb(attn, "attn_pre_solve", il);

    // Solve triangular system: (I + L) @ X = I, where L is strictly lower triangular
    ggml_tensor * attn_lower = ggml_mul(ctx0, attn, causal_mask);
    ggml_tensor * lhs = ggml_sub(ctx0, ggml_repeat(ctx0, identity, attn_lower), attn_lower);
    ggml_tensor * lin_solve = ggml_solve_tri(ctx0, lhs, attn, true, true, false);
    attn = ggml_mul(ctx0, lin_solve, causal_mask);
    attn = ggml_add(ctx0, attn, identity);

    cb(attn, "attn_solved", il);

    // Compute u = A @ v and w = A @ (g.exp() * k)
    v = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, v_beta)), attn);

    ggml_tensor * gexp;
    if (is_kda) {
        // KDA: Reuse g_exp_pos computed earlier
        gexp = g_exp_pos;
    } else {
        // GDA: g_cumsum [chunk_size, 1, n_chunks, H_k * n_seqs]
        ggml_tensor * g_cumsum_t = ggml_cont(ctx0, ggml_transpose(ctx0, g_cumsum));
        gexp = ggml_exp(ctx0, g_cumsum_t);
    }

    ggml_tensor * kbeta_gexp = ggml_mul(ctx0, k_beta, gexp);
    cb(kbeta_gexp, "kbeta_gexp", il);

    ggml_tensor * k_cumdecay = ggml_cont(ctx0, ggml_transpose(ctx0,
        ggml_mul_mat(ctx0, attn, ggml_cont(ctx0, ggml_transpose(ctx0, kbeta_gexp)))));
    cb(k_cumdecay, "k_cumdecay", il);

    // Attention scores q @ k^T with decay
    // For GDA: attn_kq[j,i] = sum_k(q[j,k] * exp(g[j] - g[i]) * k[i,k])
    // For KDA: attn_kq[j,i] = sum_k(q[j,k] * exp(g[j,k] - g[i,k]) * k[i,k])
    //                       = (q * exp(g))^T @ (k * exp(-g))
    ggml_tensor * attn_kq;
    if (is_kda) {
        // KDA: Use factorization
        ggml_tensor * q_exp_pos = ggml_mul(ctx0, q, g_exp_pos);
        ggml_tensor * k_exp_neg = ggml_mul(ctx0, k, g_exp_neg);
        attn_kq = ggml_mul_mat(ctx0, q_exp_pos, k_exp_neg);
    } else {
        // GDA: Use decay mask
        attn_kq = ggml_mul_mat(ctx0, k, q);
        attn_kq = ggml_mul(ctx0, attn_kq, decay_mask);
    }
    attn_kq = ggml_mul(ctx0, attn_kq, diag_mask);
    cb(attn_kq, "attn_kq", il);

    // Compute g_last and g_diff for state updates
    ggml_tensor * g_last;
    ggml_tensor * g_diff_exp;
    ggml_tensor * g_last_exp;

    if (is_kda) {
        // KDA: g_cumsum [S_k, chunk_size, n_chunks, H_k * n_seqs]
        // Get last element along chunk_size dimension (ne[1])
        g_last = ggml_view_4d(ctx0, g_cumsum,
            g_cumsum->ne[0], 1, g_cumsum->ne[2], g_cumsum->ne[3],
            g_cumsum->nb[1], g_cumsum->nb[2], g_cumsum->nb[3],
            (g_cumsum->ne[1] - 1) * g_cumsum->nb[1]);
        g_last = ggml_cont(ctx0, g_last);
        g_last_exp = ggml_exp(ctx0, g_last);

        // g_diff = g_last - g_cumsum
        ggml_tensor * g_last_broadcast = ggml_repeat_4d(ctx0, g_last,
            g_cumsum->ne[0], g_cumsum->ne[1], g_cumsum->ne[2], g_cumsum->ne[3]);
        ggml_tensor * g_diff = ggml_sub(ctx0, g_last_broadcast, g_cumsum);
        g_diff_exp = ggml_exp(ctx0, g_diff);
    } else {
        // GDA: g_cumsum [chunk_size, 1, n_chunks, H_k * n_seqs]
        g_last = ggml_view_4d(ctx0, g_cumsum,
            1, 1, g_cumsum->ne[2], g_cumsum->ne[3],
            g_cumsum->nb[1], g_cumsum->nb[2], g_cumsum->nb[3],
            (g_cumsum->ne[0] - 1) * ggml_element_size(g_cumsum));
        g_last = ggml_cont(ctx0, g_last);
        g_last_exp = ggml_exp(ctx0, g_last);

        ggml_tensor * g_diff = ggml_neg(ctx0, ggml_sub(ctx0, g_cumsum, g_last));
        g_diff_exp = ggml_exp(ctx0, g_diff);
    }

    cb(g_last, "g_last", il);
    cb(g_last_exp, "g_last_exp", il);

    ggml_tensor * key_gdiff = ggml_mul(ctx0, k, g_diff_exp);
    cb(key_gdiff, "key_gdiff", il);

    // Process chunks
    ggml_tensor * new_state = state_reshaped;
    ggml_tensor * core_attn_out = nullptr;

    for (int64_t chunk = 0; chunk < n_chunks; chunk++) {
        ggml_tensor * q_chunk = get_slice_2d(ctx0, q, chunk);
        ggml_tensor * v_chunk = get_slice_2d(ctx0, v, chunk);
        ggml_tensor * k_cumdecay_chunk = get_slice_2d(ctx0, k_cumdecay, chunk);
        ggml_tensor * attn_chunk = get_slice_2d(ctx0, attn_kq, chunk);
        ggml_tensor * gexp_chunk = get_slice_2d(ctx0, gexp, chunk);

        cb(attn_chunk, "attn_chunk", il);

        ggml_tensor * state_t = ggml_cont_4d(ctx0, ggml_permute(ctx0, new_state, 1, 0, 2, 3),
            S_v, S_v, 1, H_v * n_seqs);

        // v_prime = k_cumdecay @ state
        ggml_tensor * v_prime = ggml_mul_mat(ctx0, state_t, k_cumdecay_chunk);
        cb(v_prime, "v_prime_chunk", il);

        // v_new = v - v_prime
        ggml_tensor * v_new = ggml_sub(ctx0, ggml_repeat(ctx0, v_chunk, v_prime), v_prime);
        ggml_tensor * v_new_t = ggml_cont(ctx0, ggml_transpose(ctx0, v_new));
        cb(v_new, "v_new_chunk", il);

        // attn_inter = (q * g.exp()) @ state
        ggml_tensor * q_g_exp = ggml_mul(ctx0, q_chunk, gexp_chunk);
        ggml_tensor * attn_inter = ggml_mul_mat(ctx0, state_t, q_g_exp);
        cb(attn_inter, "attn_inter_chunk", il);

        // output = attn_inter + attn @ v_new
        ggml_tensor * v_attn = ggml_mul_mat(ctx0, v_new_t, attn_chunk);
        cb(v_attn, "v_attn_chunk", il);

        ggml_tensor * core_attn_out_chunk = ggml_add(ctx0, attn_inter, v_attn);
        cb(core_attn_out_chunk, "core_attn_out_chunk", il);

        core_attn_out = core_attn_out == nullptr
            ? core_attn_out_chunk
            : ggml_concat(ctx0, core_attn_out, core_attn_out_chunk, 2);

        // State update: state = state * g_last_exp + key_gdiff^T @ v_new
        ggml_tensor * k_gdiff = ggml_cont(ctx0, get_slice_2d(ctx0, key_gdiff, chunk));
        ggml_tensor * kgdmulvnew = ggml_mul_mat(ctx0, v_new_t, ggml_cont(ctx0, ggml_transpose(ctx0, k_gdiff)));

        ggml_tensor * gexp_last_chunk = ggml_cont(ctx0, get_slice_2d(ctx0, g_last_exp, chunk));

        if (is_kda) {
            // KDA: g_last_exp [S_k, 1, n_chunks, H_k * n_seqs]
            // State: [S_v, S_v, H_v, n_seqs]
            // Need to reshape g_last_exp to broadcast correctly over V dimension only
            gexp_last_chunk = ggml_reshape_4d(ctx0, gexp_last_chunk,
                1, gexp_last_chunk->ne[0], H_v, n_seqs);  // [1, S_k, H_v, n_seqs]
            // Transpose to [S_k, 1, H_v, n_seqs] then broadcast
            gexp_last_chunk = ggml_cont(ctx0, ggml_permute(ctx0, gexp_last_chunk, 1, 0, 2, 3));
        } else {
            // GDA: g_last_exp [1, 1, n_chunks, H_k * n_seqs]
            // Broadcasts over both K and V dimensions
            gexp_last_chunk = ggml_reshape_4d(ctx0, gexp_last_chunk,
                gexp_last_chunk->ne[0], gexp_last_chunk->ne[1], H_v, n_seqs);
        }

        new_state = ggml_add(ctx0,
            ggml_mul(ctx0, new_state, gexp_last_chunk),
            ggml_reshape_4d(ctx0, kgdmulvnew, kgdmulvnew->ne[0], kgdmulvnew->ne[1], H_v, n_seqs));
    }

    // Truncate padding and permute back
    ggml_tensor * output_tokens = ggml_view_4d(ctx0, core_attn_out,
        S_v, n_tokens, H_v, n_seqs,
        ggml_row_size(core_attn_out->type, S_v),
        ggml_row_size(core_attn_out->type, S_v * chunk_size * n_chunks),
        ggml_row_size(core_attn_out->type, S_v * chunk_size * n_chunks * H_v), 0);
    output_tokens = ggml_cont(ctx0, output_tokens);

    cb(output_tokens, "output_tokens", il);

    output_tokens = ggml_permute(ctx0, output_tokens, 0, 2, 1, 3);
    output_tokens = ggml_cont(ctx0, output_tokens);

    return {output_tokens, new_state};
}


/**
 * Unified autoregressive Delta Net implementation (single token processing).
 *
 * Input tensor format matches qwen3next conventions:
 * @param q         Query tensor [S_k, H_k, 1, n_seqs]
 * @param k         Key tensor [S_k, H_k, 1, n_seqs]
 * @param v         Value tensor [S_v, H_v, 1, n_seqs]
 * @param g         Gate tensor:
 *                    GDA: [H_v, 1, n_seqs]
 *                    KDA: [S_k, H_v, 1, n_seqs]
 * @param beta      Beta tensor [H_v, 1, 1, n_seqs]
 * @param state     State tensor [S_v, S_v * H_v, 1, n_seqs]
 * @param il        Layer index (for debugging callbacks)
 * @param eps_norm  Epsilon for L2 normalization
 *
 * @return Pair of (output_tokens, new_state)
 */
std::pair<ggml_tensor *, ggml_tensor *> llm_graph_context_delta::build_delta_net_unified_autoregressive(
        ggml_context * ctx0,
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * beta,
        ggml_tensor * state,
        int           il,
        float         eps_norm) {

    // Input format: [S, H, n_tokens, n_seqs] (matching qwen3next convention)
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    GGML_ASSERT(n_tokens == 1);  // Autoregressive mode is for single token

    // Detect KDA vs GDA based on g's shape
    // GDA: g has shape [H_v, 1, n_seqs] or [H_v, n_tokens, n_seqs]
    // KDA: g has shape [S_k, H_v, 1, n_seqs] or [S_k, H_v, n_tokens, n_seqs]
    const bool is_kda = (g->ne[0] == S_k && g->ne[1] == H_v);

    // Validate shapes
    GGML_ASSERT(v->ne[2] == n_tokens);
    GGML_ASSERT(k->ne[2] == n_tokens);
    GGML_ASSERT(state->ne[0] == S_v && state->ne[1] == S_v && state->ne[2] == H_v && state->ne[3] == n_seqs);
    GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    GGML_ASSERT(beta->ne[0] == H_v && beta->ne[2] == n_tokens && beta->ne[3] == n_seqs);
    GGML_ASSERT(H_k == H_v);

    if (is_kda) {
        GGML_ASSERT(g->ne[0] == S_k && g->ne[1] == H_v);
    } else {
        GGML_ASSERT(g->ne[0] == H_v);
    }

    // L2 normalize q and k
    q = ggml_l2_norm(ctx0, q, eps_norm);
    k = ggml_l2_norm(ctx0, k, eps_norm);

    const float scale = 1.0f / sqrtf((float)S_v);
    q = ggml_scale(ctx0, q, scale);
    beta = ggml_sigmoid(ctx0, beta);

    cb(q, "q_in", il);
    cb(k, "k_in", il);
    cb(v, "v_in", il);
    cb(beta, "beta_in", il);
    cb(g, "g_in", il);

    // Reshape g and beta for broadcasting
    ggml_tensor * g_t;
    ggml_tensor * beta_t;

    if (is_kda) {
        // KDA: g [S_k, H_v, 1, n_seqs] -> [S_k, 1, H_k, n_seqs]
        // For state multiplication, need [1, S_k, H_v, n_seqs] to broadcast over V only
        g_t = ggml_reshape_4d(ctx0, g, S_k, 1, H_k, n_seqs);
    } else {
        // GDA: g [H_v, 1, n_seqs] -> [1, 1, H_k, n_seqs]
        // For state multiplication, broadcasts over both K and V
        g_t = ggml_reshape_4d(ctx0, ggml_transpose(ctx0, g), 1, 1, H_k, n_seqs);
    }

    beta_t = ggml_reshape_4d(ctx0, ggml_transpose(ctx0, beta), 1, 1, H_k, n_seqs);

    // Apply exponential to g_t
    g_t = ggml_exp(ctx0, g_t);

    // State decay: state = state * exp(g)
    if (is_kda) {
        // KDA: g_t [S_k, 1, H_k, n_seqs], state [S_v, S_v, H_v, n_seqs]
        // Need to broadcast g_t over V dimension (ne[0] of state)
        // Permute g_t to [1, S_k, H_k, n_seqs] for correct broadcasting
        ggml_tensor * g_broadcast = ggml_cont(ctx0, ggml_permute(ctx0, g_t, 1, 0, 2, 3));
        state = ggml_mul(ctx0, state, g_broadcast);
    } else {
        // GDA: g_t [1, 1, H_k, n_seqs] broadcasts over both dimensions
        state = ggml_mul(ctx0, state, g_t);
    }

    // kv_mem = sum_k(state * k) = (state * k.unsqueeze(-1)).sum(dim=-2)
    // k shape after unsqueeze: [1, S_k, H_v, n_seqs]
    ggml_tensor * k_t_unsqueezed = ggml_reshape_4d(ctx0, k, 1, S_k, H_v, n_seqs);
    ggml_tensor * kv_mem = ggml_mul(ctx0, state, k_t_unsqueezed);
    // Sum over K dimension (ne[1]): transpose, sum_rows, transpose back
    kv_mem = ggml_transpose(ctx0, ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, kv_mem))));

    // v_t with singleton dimension: [S_v, 1, H_v, n_seqs]
    ggml_tensor * v_t = ggml_reshape_4d(ctx0, v, S_v, 1, H_v, n_seqs);

    // delta = (v - kv_mem) * beta
    ggml_tensor * v_diff = ggml_sub(ctx0, v_t, kv_mem);
    ggml_tensor * delta = ggml_mul(ctx0, v_diff, beta_t);

    // State update: state = state + k.unsqueeze(-1) * delta
    // k_t_unsqueezed: [1, S_k, H_v, n_seqs], delta: [S_v, 1, H_v, n_seqs]
    // Broadcasting: [S_v, S_k, H_v, n_seqs]
    ggml_tensor * k_t_delta = ggml_mul(ctx0,
        ggml_repeat_4d(ctx0, k_t_unsqueezed, S_v, S_k, H_v, n_seqs),
        delta);
    state = ggml_add(ctx0, state, k_t_delta);

    // Output: sum_k(state * q) = (state * q.unsqueeze(-1)).sum(dim=-2)
    ggml_tensor * q_t_unsqueezed = ggml_reshape_4d(ctx0, q, 1, S_k, H_v, n_seqs);
    ggml_tensor * state_q = ggml_mul(ctx0, state, q_t_unsqueezed);
    // Sum over K dimension
    ggml_tensor * core_attn_out = ggml_transpose(ctx0,
        ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, state_q))));

    cb(core_attn_out, "output_tokens", il);
    cb(state, "new_state", il);

    return {core_attn_out, state};
}


/**
 * Main entry point that dispatches to chunked or autoregressive based on n_tokens.
 *
 * Input tensor format matches qwen3next conventions:
 * @param q         Query tensor [S_k, H_k, n_tokens, n_seqs]
 * @param k         Key tensor [S_k, H_k, n_tokens, n_seqs]
 * @param v         Value tensor [S_v, H_v, n_tokens, n_seqs]
 * @param g         Gate tensor (GDA: [H_v, n_tokens, n_seqs], KDA: [S_k, H_v, n_tokens, n_seqs])
 * @param beta      Beta tensor [H_v, 1, n_tokens, n_seqs]
 * @param state     State tensor [S_v, S_v * H_v, 1, n_seqs]
 */
std::pair<ggml_tensor *, ggml_tensor *> llm_graph_context_delta::build_delta_net_unified(
        ggml_context * ctx0,
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * g,
        ggml_tensor * beta,
        ggml_tensor * state,
        ggml_tensor * causal_mask,
        ggml_tensor * identity,
        ggml_tensor * diag_mask,
        int           il,
        int64_t       chunk_size,
        float         eps_norm) {

    // Input format: [S, H, n_tokens, n_seqs] (matching qwen3next convention)
    const int64_t n_tokens = q->ne[2];

    if (n_tokens == 1) {
        return build_delta_net_unified_autoregressive(
            ctx0, q, k, v, g, beta, state, il, eps_norm);
    }
    return build_delta_net_unified_chunking(
        ctx0, q, k, v, g, beta, state, causal_mask, identity, diag_mask,
        il, chunk_size, eps_norm);
}
