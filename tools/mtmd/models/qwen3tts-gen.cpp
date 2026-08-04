#include "models.h"

#include <string>

// on-device sampling: top-k, top-p, then a random draw
ggml_tensor * clip_graph_qwen3tts_gen::code_gen::do_sampling(ggml_tensor * logits, ggml_tensor * inp_rand) const {
    logits = ggml_reshape_1d(ctx0, logits, ggml_nelements(logits));
    const int64_t n_vocab = logits->ne[0];

    // sort a's rows by idx
    auto sort_by = [this](ggml_tensor * a, ggml_tensor * idx) {
        ggml_tensor * a2d = ggml_reshape_2d(ctx0, a, 1, a->ne[0]);
        return ggml_reshape_1d(ctx0, ggml_get_rows(ctx0, a2d, idx), idx->ne[0]);
    };

    ggml_tensor * cur        = logits;
    ggml_tensor * candidates = nullptr; // maps row index back to vocab id

    if (top_k > 0 && top_k < n_vocab) {
        ggml_tensor * idx = ggml_top_k(ctx0, cur, top_k);
        candidates = idx;
        cur        = sort_by(cur, idx);
        cb(cur, "sample_top_k_logits", -1);
    }

    if (top_p < 1.0f) {
        ggml_tensor * sorted_idx    = ggml_argsort(ctx0, cur, GGML_SORT_ORDER_DESC);
        ggml_tensor * sorted_logits = sort_by(cur, sorted_idx);
        candidates = candidates ? sort_by(candidates, sorted_idx) : sorted_idx;

        ggml_tensor * probs = ggml_soft_max(ctx0, sorted_logits);
        ggml_tensor * cdf   = ggml_cumsum(ctx0, probs);

        // keep_mask[i] = 1 once cdf[i] crosses top_p
        ggml_tensor * cdf_scaled = ggml_scale_bias(ctx0, cdf, -1.0f, top_p);
        ggml_tensor * keep_mask  = ggml_step(ctx0, cdf_scaled);
        ggml_tensor * idxf       = ggml_sum(ctx0, keep_mask);
        idxf = ggml_clamp(ctx0, idxf, 0.0f, (float) keep_mask->ne[0] - 1);
        ggml_tensor * ones = ggml_scale_bias(ctx0, idxf, 0.0f, 1.0f);

        // top-p must include the crossing element, so force it to 1
        ggml_tensor * keep_mask_2d = ggml_reshape_2d(ctx0, keep_mask, 1, keep_mask->ne[0]);
        keep_mask_2d = ggml_set_rows(ctx0, keep_mask_2d, ones, ggml_cast(ctx0, idxf, GGML_TYPE_I32));
        keep_mask    = ggml_reshape_1d(ctx0, keep_mask_2d, keep_mask->ne[0]);

        // log(1) = 0 (keep), log(0) = -inf (drop)
        ggml_tensor * bias = ggml_log(ctx0, keep_mask);
        cur = ggml_add(ctx0, sorted_logits, bias);
        cb(cur, "sample_top_p_logits", -1);
    }

    // draw one token: find where the cdf crosses inp_rand
    ggml_tensor * probs  = ggml_soft_max(ctx0, cur);
    ggml_tensor * cumsum = ggml_cumsum(ctx0, probs);

    ggml_tensor * diff       = ggml_sub(ctx0, cumsum, inp_rand);
    ggml_tensor * cross_mask = ggml_step(ctx0, diff);
    ggml_tensor * idxf       = ggml_sum(ctx0, cross_mask);
    ggml_tensor * idx        = ggml_cast(ctx0, ggml_scale_bias(ctx0, idxf, -1.0f, (float) cross_mask->ne[0]), GGML_TYPE_I32);

    if (candidates) {
        ggml_tensor * cand_2d = ggml_reshape_2d(ctx0, candidates, 1, candidates->ne[0]);
        idx = ggml_get_rows(ctx0, cand_2d, idx);
    }
    cb(idx, "sample_token_id", -1);

    return idx;
}

// returns a new cache with row row_idx set to value
ggml_tensor * clip_graph_qwen3tts_gen::code_gen::cache_set(ggml_tensor * cache, int row_idx, ggml_tensor * value) const {
    const int64_t n_embd  = cache->ne[0];
    const int64_t n_cache = cache->ne[1];
    GGML_ASSERT(row_idx >= 0 && row_idx < n_cache);

    // append value as the last row, then gather it back into place
    ggml_tensor * value_2d  = ggml_reshape_2d(ctx0, value, n_embd, 1);
    ggml_tensor * cache_ext = ggml_concat(ctx0, cache, value_2d, 1); // [n_embd, n_cache + 1]

    // gather indices [0..row_idx-1, n_cache, row_idx+1..n_cache-1]
    // built via concat, since ggml_set_rows needs F32/F16 values, not an I32 index array
    ggml_tensor * idx = const_i32(cache, (float) n_cache);
    if (row_idx > 0) {
        ggml_tensor * prefix = ggml_cast(ctx0, ggml_arange(ctx0, 0.0f, (float) row_idx, 1.0f), GGML_TYPE_I32);
        idx = ggml_concat(ctx0, prefix, idx, 0);
    }
    if (row_idx < n_cache - 1) {
        ggml_tensor * suffix = ggml_cast(ctx0, ggml_arange(ctx0, (float) (row_idx + 1), (float) n_cache, 1.0f), GGML_TYPE_I32);
        idx = ggml_concat(ctx0, idx, suffix, 0);
    }

    ggml_tensor * result = ggml_get_rows(ctx0, cache_ext, idx);
    cb(result, "cache_set_out", -1);
    return result;
}

// builds a const i32 with no host upload: view a tensor, zero it via scale, add value, cast to i32
ggml_tensor * clip_graph_qwen3tts_gen::code_gen::const_i32(ggml_tensor * anchor, float value) const {
    ggml_tensor * v = ggml_view_1d(ctx0, anchor, 1, 0);
    if (v->type != GGML_TYPE_F32) {
        v = ggml_cast(ctx0, v, GGML_TYPE_F32);
    }
    return ggml_cast(ctx0, ggml_scale_bias(ctx0, v, 0.0f, value), GGML_TYPE_I32);
}

// causal keep-mask row for a query at position pos, window size n_kv_pad
ggml_tensor * clip_graph_qwen3tts_gen::code_gen::causal_mask_row(int64_t n_kv_pad, int pos) const {
    ggml_tensor * ones = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv_pad, n_kv_pad), 1.0f);
    ggml_tensor * keep = ggml_tri(ctx0, ones, GGML_TRI_TYPE_LOWER_DIAG);
    ggml_tensor * row  = ggml_view_1d(ctx0, keep, n_kv_pad, (size_t) pos * keep->nb[1]);
    ggml_tensor * mask = ggml_log(ctx0, row); // 0 = keep, -inf = masked
    return ggml_reshape_4d(ctx0, mask, n_kv_pad, 1, 1, 1);
}

// talker hidden size -> predictor hidden size (small_to_mtp_projection)
ggml_tensor * clip_graph_qwen3tts_gen::code_gen::project_in(ggml_tensor * cur) const {
    if (!model.gen_code_proj_in_w) {
        return cur;
    }
    cur = ggml_mul_mat(ctx0, model.gen_code_proj_in_w, cur);
    if (model.gen_code_proj_in_b) {
        cur = ggml_add(ctx0, cur, model.gen_code_proj_in_b);
    }
    return cur;
}

// one transformer layer at position pos; writes k/v into k_cache_layer/v_cache_layer at row pos
ggml_tensor * clip_graph_qwen3tts_gen::code_gen::layer_forward(
        ggml_tensor * cur,
        const clip_layer & layer,
        ggml_tensor * inp_pos,
        ggml_tensor * kq_mask,
        ggml_tensor *& k_cache_layer,
        ggml_tensor *& v_cache_layer,
        int64_t n_kv_pad,
        int pos,
        int il) const {
    const int     n_head    = hparams.n_head;
    const int     n_head_kv = hparams.n_head_kv;
    const int64_t d_head    = layer.q_w->ne[1] / n_head; // real head_dim, not n_embd / n_head
    const float   kq_scale  = 1.0f / sqrtf((float) d_head);

    ggml_tensor * residual = cur;

    ggml_tensor * h = ggml_rms_norm(ctx0, cur, hparams.eps);
    h = ggml_mul(ctx0, h, layer.ln_1_w);

    ggml_tensor * q = ggml_mul_mat(ctx0, layer.q_w, h);
    ggml_tensor * k = ggml_mul_mat(ctx0, layer.k_w, h);
    ggml_tensor * v = ggml_mul_mat(ctx0, layer.v_w, h);

    q = ggml_reshape_3d(ctx0, q, d_head, n_head, 1);
    k = ggml_reshape_3d(ctx0, k, d_head, n_head_kv, 1);

    q = ggml_rms_norm(ctx0, q, hparams.eps);
    q = ggml_mul(ctx0, q, layer.q_norm);
    k = ggml_rms_norm(ctx0, k, hparams.eps);
    k = ggml_mul(ctx0, k, layer.k_norm);

    q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, (int) d_head, GGML_ROPE_TYPE_NEOX, 0,
                      hparams.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx0, k, inp_pos, nullptr, (int) d_head, GGML_ROPE_TYPE_NEOX, 0,
                      hparams.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // write k/v into the cache at row pos, flat layout
    ggml_tensor * k_flat = ggml_reshape_1d(ctx0, k, d_head * n_head_kv);
    k_cache_layer = cache_set(k_cache_layer, pos, k_flat);
    v_cache_layer = cache_set(v_cache_layer, pos, v);

    ggml_tensor * q_cur = ggml_reshape_4d(ctx0, q, d_head, n_head, 1, 1);
    ggml_tensor * k_cur = ggml_reshape_4d(ctx0, k_cache_layer, d_head, n_head_kv, n_kv_pad, 1);
    ggml_tensor * v_cur = ggml_reshape_4d(ctx0, v_cache_layer, d_head, n_head_kv, n_kv_pad, 1);

    ggml_tensor * attn_out = build_attn(layer.o_w, layer.o_b, q_cur, k_cur, v_cur, kq_mask, kq_scale, il);

    cur = ggml_add(ctx0, residual, attn_out);

    ggml_tensor * h2 = ggml_rms_norm(ctx0, cur, hparams.eps);
    h2 = ggml_mul(ctx0, h2, layer.ln_2_w);

    ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ff_gate_w, h2);
    ggml_tensor * up   = ggml_mul_mat(ctx0, layer.ff_up_w, h2);
    ggml_tensor * gu   = ggml_swiglu_split(ctx0, gate, up);
    ggml_tensor * down = ggml_mul_mat(ctx0, layer.ff_down_w, gu);

    return ggml_add(ctx0, cur, down);
}

// position 0: hidden bridge, seeds the k/v cache, no sampling
// position 1: embed(code0), sample with lm_head[0], write out_code_cache[1]
void clip_graph_qwen3tts_gen::code_gen::prefill(
        std::vector<ggml_tensor *> & k_cache,
        std::vector<ggml_tensor *> & v_cache,
        ggml_tensor *& out_code_cache,
        ggml_tensor * h_state,
        ggml_tensor * code0_embd,
        ggml_tensor * inp_rand) const {
    const int64_t n_kv_pad = k_cache[0]->ne[1];

    {
        ggml_tensor * cur     = project_in(h_state);
        ggml_tensor * kq_mask = causal_mask_row(n_kv_pad, 0);
        ggml_tensor * inp_pos = const_i32(k_cache[0], 0.0f);
        for (size_t il = 0; il < model.layers.size(); il++) {
            cur = layer_forward(cur, model.layers[il], inp_pos, kq_mask, k_cache[il], v_cache[il], n_kv_pad, 0, (int) il);
        }
        // position 0's output is unused, it only seeded the cache
    }

    {
        ggml_tensor * cur     = project_in(code0_embd);
        ggml_tensor * kq_mask = causal_mask_row(n_kv_pad, 1);
        ggml_tensor * inp_pos = const_i32(k_cache[0], 1.0f);
        for (size_t il = 0; il < model.layers.size(); il++) {
            cur = layer_forward(cur, model.layers[il], inp_pos, kq_mask, k_cache[il], v_cache[il], n_kv_pad, 1, (int) il);
        }

        cur = ggml_rms_norm(ctx0, cur, hparams.eps);
        cur = ggml_mul(ctx0, cur, model.gen_code_norm_w);

        ggml_tensor * head_w = model.gen_code_head_w;
        ggml_tensor * head_g = ggml_view_2d(ctx0, head_w, head_w->ne[0], head_w->ne[1], head_w->nb[1], 0); // lm_head[0]
        ggml_tensor * logits = ggml_mul_mat(ctx0, head_g, cur);

        ggml_tensor * sampled = do_sampling(logits, inp_rand);
        out_code_cache = cache_set(out_code_cache, 1, sampled);
    }
}

// one decode step of code_predictor
// at step_idx g:
// - read code from out_code_cache[g], then embed it with codebook table g-1
// - write new kv at cache row g+1, sample with lm_head[g]
// - write result to out_code_cache[g+1]
// step_idx must be in [1, n_acoustic - 1]
ggml_tensor * clip_graph_qwen3tts_gen::code_gen::step(
        std::vector<ggml_tensor *> & k_cache,
        std::vector<ggml_tensor *> & v_cache,
        ggml_tensor * out_code_cache,
        ggml_tensor * inp_rand,
        int step_idx) const {
    const int64_t n_acoustic = model.gen_code_head_w->ne[2];
    GGML_ASSERT(step_idx >= 1 && step_idx < n_acoustic);
    GGML_ASSERT(k_cache.size() == model.layers.size());
    GGML_ASSERT(v_cache.size() == model.layers.size());

    const int64_t n_kv_pad = k_cache[0]->ne[1];
    const int     pos      = step_idx + 1; // new cache row and RoPE position

    // embed the previous code via this step's codebook table (rows are already scalars)
    ggml_tensor * code_in = ggml_view_1d(ctx0, out_code_cache, 1, (size_t) step_idx * out_code_cache->nb[1]);

    ggml_tensor * embd_w = model.gen_code_embd_w; // [n_embd_talker, vocab, n_acoustic]
    ggml_tensor * embd_g = ggml_view_2d(ctx0, embd_w, embd_w->ne[0], embd_w->ne[1], embd_w->nb[1],
                                        (size_t) (step_idx - 1) * embd_w->nb[2]);
    ggml_tensor * cur = ggml_get_rows(ctx0, embd_g, code_in);
    cur = ggml_reshape_1d(ctx0, cur, cur->ne[0]);
    cb(cur, "step_embd_in", step_idx);

    cur = project_in(cur);
    cb(cur, "step_proj_in", step_idx);

    ggml_tensor * kq_mask = causal_mask_row(n_kv_pad, pos);
    ggml_tensor * inp_pos = const_i32(k_cache[0], (float) pos);

    for (size_t il = 0; il < model.layers.size(); il++) {
        cur = layer_forward(cur, model.layers[il], inp_pos, kq_mask, k_cache[il], v_cache[il], n_kv_pad, pos, (int) il);
        cb(cur, "step_layer_out", (int) il);
    }

    // final norm, this step's lm_head, sample, write the result
    cur = ggml_rms_norm(ctx0, cur, hparams.eps);
    cur = ggml_mul(ctx0, cur, model.gen_code_norm_w);

    ggml_tensor * head_w = model.gen_code_head_w; // [n_embd_pred, vocab, n_acoustic]
    ggml_tensor * head_g = ggml_view_2d(ctx0, head_w, head_w->ne[0], head_w->ne[1], head_w->nb[1],
                                        (size_t) step_idx * head_w->nb[2]);
    ggml_tensor * logits = ggml_mul_mat(ctx0, head_g, cur);
    cb(logits, "step_logits", step_idx);

    ggml_tensor * sampled = do_sampling(logits, inp_rand);
    cb(sampled, "step_sampled", step_idx);

    return cache_set(out_code_cache, pos, sampled);
}

// causal conv1d, stride 1: prepend persisted left-context instead of zero-padding, then a plain conv
// x: [T, IC] (T-first). w: [K, IC, OC]. state_name empty means K == 1 (no left-context). returns [T, OC]
ggml_tensor * clip_graph_qwen3tts_gen::code2wav::causal_conv1d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int dilation, const std::string & state_name) const {
    const int K   = (int) w->ne[0];
    const int pad = (K - 1) * dilation;

    ggml_tensor * x_full = x;
    if (pad > 0) {
        ggml_tensor * left = state_in.at(state_name); // [pad, IC]
        x_full = ggml_concat(ctx0, left, x, 0);
    }
    ggml_tensor * y = ggml_conv_1d(ctx0, w, x_full, 1, 0, dilation); // [T, OC, 1]
    y = ggml_reshape_2d(ctx0, y, y->ne[0], y->ne[1]);
    if (b) {
        y = ggml_add(ctx0, y, ggml_reshape_2d(ctx0, b, 1, b->ne[0]));
    }
    if (pad > 0) {
        ggml_tensor * new_left = ggml_cont(ctx0, ggml_view_2d(ctx0, x_full, pad, x_full->ne[1], x_full->nb[1],
                                                              (size_t) (x_full->ne[0] - pad) * x_full->nb[0]));
        state_out.push_back({state_name, new_left});
    }
    return y;
}

// causal depthwise conv1d, stride 1, dilation 1, kernel from w's shape.
// x: [T, C]. w: [K, 1, C]. returns [T, C]. see causal_conv1d for the state contract.
ggml_tensor * clip_graph_qwen3tts_gen::code2wav::causal_conv1d_dw(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, const std::string & state_name) const {
    const int K   = (int) w->ne[0];
    const int pad = K - 1;

    ggml_tensor * x_full = x;
    if (pad > 0) {
        ggml_tensor * left = state_in.at(state_name); // [pad, C]
        x_full = ggml_concat(ctx0, left, x, 0);
    }
    ggml_tensor * y = ggml_conv_1d_dw(ctx0, w, x_full, 1, 0, 1); // [T, C, 1]
    y = ggml_reshape_2d(ctx0, y, y->ne[0], y->ne[1]);
    if (b) {
        y = ggml_add(ctx0, y, ggml_reshape_2d(ctx0, b, 1, b->ne[0]));
    }
    if (pad > 0) {
        ggml_tensor * new_left = ggml_cont(ctx0, ggml_view_2d(ctx0, x_full, pad, x_full->ne[1], x_full->nb[1],
                                                              (size_t) (x_full->ne[0] - pad) * x_full->nb[0]));
        state_out.push_back({state_name, new_left});
    }
    return y;
}

// causal ConvTranspose1d, the (kernel - stride) overlap tail is kept as state for the next call
// x: [T, IC], w: [K, OC, IC]. state_name empty means K == stride (no overlap). returns [T * stride, OC]
ggml_tensor * clip_graph_qwen3tts_gen::code2wav::causal_conv_transpose1d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int stride, const std::string & state_name) const {
    const int     K        = (int) w->ne[0];
    const int     OC       = (int) w->ne[1];
    const int     trim     = K - stride;
    const int64_t emit_len = x->ne[0] * stride;

    // transposed conv as GEMM + col2im scatter-add, y: [emit_len + trim, OC]
    ggml_tensor * w2  = ggml_reshape_2d(ctx0, w, (int64_t) K * OC, w->ne[2]);
    w2                = ggml_cont(ctx0, ggml_transpose(ctx0, w2));
    ggml_tensor * xt  = ggml_cont(ctx0, ggml_transpose(ctx0, x));
    ggml_tensor * col = ggml_mul_mat(ctx0, w2, xt);
    ggml_tensor * y   = ggml_col2im_1d(ctx0, col, stride, OC, 0);

    ggml_tensor * out = y;
    if (trim > 0) {
        ggml_tensor * tail = state_in.at(state_name); // [trim, OC]
        ggml_tensor * head = ggml_add(ctx0, ggml_view_2d(ctx0, y, trim, y->ne[1], y->nb[1], 0), tail);
        if (emit_len > trim) {
            ggml_tensor * middle = ggml_view_2d(ctx0, y, emit_len - trim, y->ne[1], y->nb[1], (size_t) trim * y->nb[0]);
            out = ggml_concat(ctx0, head, middle, 0);
        } else {
            out = head;
        }
        ggml_tensor * new_tail = ggml_cont(ctx0, ggml_view_2d(ctx0, y, trim, y->ne[1], y->nb[1], (size_t) emit_len * y->nb[0]));
        state_out.push_back({state_name, new_tail});
    }
    if (b) {
        out = ggml_add(ctx0, out, ggml_reshape_2d(ctx0, b, 1, b->ne[0]));
    }
    return out;
}

// SnakeBeta activation: y = x + sin(alpha*x)^2 * inv_beta (alpha/inv_beta folded via exp/reciprocal at conversion time)
// x: [T, C]. alpha/beta: [C], broadcasts over T
ggml_tensor * clip_graph_qwen3tts_gen::code2wav::snake(ggml_tensor * x, ggml_tensor * alpha, ggml_tensor * beta) const {
    ggml_tensor * a = ggml_reshape_2d(ctx0, alpha, 1, alpha->ne[0]);
    ggml_tensor * b = ggml_reshape_2d(ctx0, beta,  1, beta->ne[0]);

    // expand reshapes first so mul/sin/sqr/mul/add lands as consecutive nodes, letting backends fuse them
    ggml_build_forward_expand(gf, a);
    ggml_build_forward_expand(gf, b);

    ggml_tensor * s = ggml_sin(ctx0, ggml_mul(ctx0, x, a));
    s = ggml_sqr(ctx0, s);
    s = ggml_mul(ctx0, s, b);
    return ggml_add(ctx0, x, s);
}

// RVQ codebook decode: T frames of 16 codes -> 512-dim hidden (C-first, [512, T])
// codebook 0 (semantic) and 1..15 (acoustic) sum within their group, project separately, then add
ggml_tensor * clip_graph_qwen3tts_gen::code2wav::quant_decode(ggml_tensor * inp_codes) const {
    const auto & c2w = model.c2w;
    const int64_t T = inp_codes->ne[0];

    // ids for codebook group g over all T frames, [T] I32
    auto group_ids = [&](int g) {
        return ggml_view_1d(ctx0, inp_codes, T, (size_t) g * inp_codes->nb[1]);
    };

    ggml_tensor * sem     = ggml_get_rows(ctx0, c2w.quant_first_cb_w, group_ids(0)); // [256, T]
    ggml_tensor * sem_out = ggml_mul_mat(ctx0, c2w.quant_first_out_w, sem);          // [512, T]

    ggml_tensor * acc = nullptr;
    const int64_t n_acoustic = c2w.quant_rest_cb_w->ne[2];
    for (int g = 1; g <= n_acoustic; g++) {
        ggml_tensor * cb_g  = ggml_view_2d(ctx0, c2w.quant_rest_cb_w, c2w.quant_rest_cb_w->ne[0], c2w.quant_rest_cb_w->ne[1],
                                           c2w.quant_rest_cb_w->nb[1], (size_t) (g - 1) * c2w.quant_rest_cb_w->nb[2]);
        ggml_tensor * embd = ggml_get_rows(ctx0, cb_g, group_ids(g)); // [256, T]
        acc = acc ? ggml_add(ctx0, acc, embd) : embd;
    }
    ggml_tensor * ac_out = ggml_mul_mat(ctx0, c2w.quant_rest_out_w, acc); // [512, T]

    ggml_tensor * hidden = ggml_add(ctx0, sem_out, ac_out);
    cb(hidden, "wav_quant_hidden", -1);
    return hidden;
}

// one pre_transformer layer over a batch of N = sliding_window new frames
// attention runs over [(W-1)-frame prefix from the last batch] + [N new frames]
// RoPE positions come from a persisted counter, so phases line up across batches
ggml_tensor * clip_graph_qwen3tts_gen::code2wav::tfm_layer_forward(ggml_tensor * cur, const clip_layer & layer, int il) const {
    const int     n_head    = hparams.wav_tfm_n_head;
    const int     n_head_kv = hparams.wav_tfm_n_head_kv;
    const int64_t d_head    = layer.q_w->ne[1] / n_head;
    const float   kq_scale  = 1.0f / sqrtf((float) d_head);
    const int64_t W         = hparams.wav_tfm_swa; // == N, frames per batch
    const int64_t N         = cur->ne[1];
    const int64_t prefix    = W - 1;
    const int64_t total_kv  = prefix + N;

    ggml_tensor * residual = cur;
    ggml_tensor * h = ggml_rms_norm(ctx0, cur, hparams.wav_tfm_eps);
    h = ggml_mul(ctx0, h, layer.ln_1_w);

    ggml_tensor * q = ggml_mul_mat(ctx0, layer.q_w, h); // [n_head*d_head, N]
    ggml_tensor * k = ggml_mul_mat(ctx0, layer.k_w, h); // [n_head_kv*d_head, N]
    ggml_tensor * v = ggml_mul_mat(ctx0, layer.v_w, h); // [n_head_kv*d_head, N]

    q = ggml_reshape_3d(ctx0, q, d_head, n_head, N);
    k = ggml_reshape_3d(ctx0, k, d_head, n_head_kv, N);

    // real, ever-increasing positions: base (persisted) .. base+N-1
    ggml_tensor * base   = ggml_reshape_1d(ctx0, state_in.at("tfm_pos"), 1);
    ggml_tensor * offset = ggml_arange(ctx0, 0.0f, (float) N, 1.0f);
    ggml_tensor * pos    = ggml_cast(ctx0, ggml_add(ctx0, offset, base), GGML_TYPE_I32);

    q = ggml_rope_ext(ctx0, q, pos, nullptr, (int) d_head, GGML_ROPE_TYPE_NEOX, 0,
                      hparams.wav_tfm_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx0, k, pos, nullptr, (int) d_head, GGML_ROPE_TYPE_NEOX, 0,
                      hparams.wav_tfm_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // the position counter is the same for all layers, push it once from layer 0
    if (il == 0) {
        state_out.push_back({"tfm_pos", ggml_scale_bias(ctx0, state_in.at("tfm_pos"), 1.0f, (float) N)});
    }

    ggml_tensor * k_new = ggml_reshape_2d(ctx0, k, d_head * n_head_kv, N);
    ggml_tensor * v_new = ggml_reshape_2d(ctx0, v, d_head * n_head_kv, N);

    ggml_tensor * old_k = state_in.at("tfm_k_" + std::to_string(il)); // [d_head*n_head_kv, W-1]
    ggml_tensor * old_v = state_in.at("tfm_v_" + std::to_string(il));

    ggml_tensor * k_full = ggml_concat(ctx0, old_k, k_new, 1); // [.., prefix+N]
    ggml_tensor * v_full = ggml_concat(ctx0, old_v, v_new, 1);

    // next batch's prefix: the last (W-1) frames of this batch
    state_out.push_back({"tfm_k_" + std::to_string(il),
        ggml_cont(ctx0, ggml_view_2d(ctx0, k_full, k_full->ne[0], prefix, k_full->nb[1], (size_t) N * k_full->nb[1]))});
    state_out.push_back({"tfm_v_" + std::to_string(il),
        ggml_cont(ctx0, ggml_view_2d(ctx0, v_full, v_full->ne[0], prefix, v_full->nb[1], (size_t) N * v_full->nb[1]))});

    // banded causal mask: key j is visible to query i iff 0 <= (prefix+i) - j < W
    ggml_tensor * pos_k = ggml_reshape_2d(ctx0, ggml_arange(ctx0, 0.0f, (float) total_kv, 1.0f), total_kv, 1);
    ggml_tensor * pos_q = ggml_reshape_2d(ctx0, ggml_arange(ctx0, (float) prefix, (float) (prefix + N), 1.0f), 1, N);
    ggml_tensor * pos_q_grid = ggml_repeat_4d(ctx0, pos_q, total_kv, N, 1, 1);
    ggml_tensor * diff = ggml_sub(ctx0, pos_q_grid, pos_k); // [total_kv, N]

    ggml_tensor * causal_keep = ggml_step(ctx0, ggml_scale_bias(ctx0, diff, 1.0f, 0.5f));            // diff >= 0
    ggml_tensor * in_window   = ggml_step(ctx0, ggml_scale_bias(ctx0, diff, -1.0f, (float) W - 0.5f)); // diff < W
    ggml_tensor * keep = ggml_mul(ctx0, causal_keep, in_window);

    // on a cold start, key j is real state only when j >= prefix - tfm_pos, mask out the rest
    ggml_tensor * warm = ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_add(ctx0, pos_k, base),
                                                         1.0f, 0.5f - (float) prefix)); // j + pos > prefix - 0.5
    keep = ggml_mul(ctx0, keep, warm);

    ggml_tensor * mask = ggml_reshape_4d(ctx0, ggml_log(ctx0, keep), total_kv, N, 1, 1); // 0 = keep, -inf = masked

    ggml_tensor * q_cur = ggml_reshape_4d(ctx0, q, d_head, n_head, N, 1);
    ggml_tensor * k_cur = ggml_reshape_4d(ctx0, k_full, d_head, n_head_kv, total_kv, 1);
    ggml_tensor * v_cur = ggml_reshape_4d(ctx0, v_full, d_head, n_head_kv, total_kv, 1);

    ggml_tensor * attn_out = build_attn(layer.o_w, layer.o_b, q_cur, k_cur, v_cur, mask, kq_scale, il);
    if (layer.ls_1_w) {
        attn_out = ggml_mul(ctx0, attn_out, layer.ls_1_w);
    }
    cur = ggml_add(ctx0, residual, attn_out);

    ggml_tensor * residual2 = cur;
    ggml_tensor * h2 = ggml_rms_norm(ctx0, cur, hparams.wav_tfm_eps);
    h2 = ggml_mul(ctx0, h2, layer.ln_2_w);

    ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ff_gate_w, h2);
    ggml_tensor * up   = ggml_mul_mat(ctx0, layer.ff_up_w, h2);
    ggml_tensor * gu   = ggml_swiglu_split(ctx0, gate, up);
    ggml_tensor * down = ggml_mul_mat(ctx0, layer.ff_down_w, gu);
    if (layer.ls_2_w) {
        down = ggml_mul(ctx0, down, layer.ls_2_w);
    }
    return ggml_add(ctx0, residual2, down);
}

// dwconv -> LayerNorm -> pwconv1 -> GELU -> pwconv2 -> layer scale -> residual
// x: [T, C] T-first; LayerNorm/pwconv need C on ne0, so this transposes in and back out
ggml_tensor * clip_graph_qwen3tts_gen::code2wav::convnext_block(ggml_tensor * x, const clip_code2wav::upsample_block & blk, const std::string & state_prefix) const {
    ggml_tensor * residual = x;

    ggml_tensor * h = causal_conv1d_dw(x, blk.dwconv_w, blk.dwconv_b, state_prefix + "_dwconv"); // [T, C]
    ggml_tensor * hc = ggml_cont(ctx0, ggml_transpose(ctx0, h)); // [C, T]

    hc = ggml_norm(ctx0, hc, 1e-6f);
    hc = ggml_mul(ctx0, hc, blk.norm_w);
    hc = ggml_add(ctx0, hc, blk.norm_b);

    ggml_tensor * g = ggml_mul_mat(ctx0, blk.pw1_w, hc);
    g = ggml_add(ctx0, g, blk.pw1_b);
    g = ggml_gelu(ctx0, g);
    g = ggml_mul_mat(ctx0, blk.pw2_w, g);
    g = ggml_add(ctx0, g, blk.pw2_b);
    g = ggml_mul(ctx0, g, blk.gamma);

    ggml_tensor * g_t = ggml_cont(ctx0, ggml_transpose(ctx0, g)); // back to [T, C]
    return ggml_add(ctx0, residual, g_t);
}

// SnakeBeta -> dilated causal conv (k=7) -> SnakeBeta -> pointwise causal conv (k=1) -> residual.
// x: [T, C]. returns [T, C].
ggml_tensor * clip_graph_qwen3tts_gen::code2wav::dac_res_unit(ggml_tensor * x, const clip_code2wav::dac_res & res, int dilation, const std::string & state_name) const {
    ggml_tensor * residual = x;
    ggml_tensor * h = snake(x, res.act1_alpha, res.act1_beta);
    h = causal_conv1d(h, res.conv1_w, res.conv1_b, dilation, state_name);
    h = snake(h, res.act2_alpha, res.act2_beta);
    h = causal_conv1d(h, res.conv2_w, res.conv2_b, 1, ""); // k=1, no left-context needed
    return ggml_add(ctx0, residual, h);
}

// RVQ codes -> raw PCM for a batch of N = sliding_window frames
ggml_tensor * clip_graph_qwen3tts_gen::code2wav::decode(ggml_tensor * inp_codes) const {
    const auto & c2w = model.c2w;

    // 1. quantizer decode: N frames of 16 codes -> [512, N] (C-first)
    ggml_tensor * hidden = quant_decode(inp_codes);

    // 2. pre_conv: [512, N] -> T-first [N, 512] -> causal conv k=3 -> [N, 1024]
    ggml_tensor * x = ggml_cont(ctx0, ggml_transpose(ctx0, hidden)); // [N, 512]
    x = causal_conv1d(x, c2w.pre_conv_w, c2w.pre_conv_b, 1, "pre_conv"); // [N, 1024]
    cb(x, "wav_pre_conv_out", -1);

    // 3. pre_transformer: back to C-first [1024, N], project down, run the layers, project back up
    ggml_tensor * cur = ggml_cont(ctx0, ggml_transpose(ctx0, x)); // [1024, N]
    cur = ggml_mul_mat(ctx0, c2w.tfm_in_proj_w, cur);
    cur = ggml_add(ctx0, cur, c2w.tfm_in_proj_b); // [512 (tfm hidden), N]

    for (int il = 0; il < hparams.wav_tfm_n_layer; il++) {
        cur = tfm_layer_forward(cur, c2w.tfm_layers[il], il);
    }

    cur = ggml_rms_norm(ctx0, cur, hparams.wav_tfm_eps);
    cur = ggml_mul(ctx0, cur, c2w.tfm_output_norm_w);
    cur = ggml_mul_mat(ctx0, c2w.tfm_out_proj_w, cur);
    cur = ggml_add(ctx0, cur, c2w.tfm_out_proj_b); // [1024, N]
    cb(cur, "wav_tfm_out", -1);

    // 4. upsample: 2x (causal ConvTranspose1d, stride 2 + ConvNeXt block), back to T-first
    // kernel == stride here, so there is no overlap tail to persist
    x = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // [N, 1024]
    for (size_t il = 0; il < c2w.upsample.size(); il++) {
        const auto & up = c2w.upsample[il];
        x = causal_conv_transpose1d(x, up.conv_w, up.conv_b, 2, "");
        x = convnext_block(x, up, "up" + std::to_string(il));
        cb(x, "wav_upsample_out", (int) il);
    }

    // 5. DAC decoder: conv_pre -> n blocks (SnakeBeta -> ConvTranspose1d -> 3 res units) -> conv_post
    static constexpr int DAC_DILATIONS[3] = { 1, 3, 9 };

    x = causal_conv1d(x, c2w.dac_entry_w, c2w.dac_entry_b, 1, "dac_entry");
    cb(x, "wav_dac_entry_out", -1);

    for (size_t il = 0; il < c2w.dac.size(); il++) {
        const auto & blk = c2w.dac[il];
        const int stride = (int) (blk.conv_w->ne[0] / 2); // kernel == 2*stride for all 4 blocks
        const std::string blk_name = "dac" + std::to_string(il);
        x = snake(x, blk.snake_alpha, blk.snake_beta);
        x = causal_conv_transpose1d(x, blk.conv_w, blk.conv_b, stride, blk_name + "_tail");
        for (size_t ir = 0; ir < blk.res.size(); ir++) {
            x = dac_res_unit(x, blk.res[ir], DAC_DILATIONS[ir], blk_name + "_res" + std::to_string(ir));
        }
        cb(x, "wav_dac_block_out", (int) il);
    }

    x = snake(x, c2w.dac_post_snake_alpha, c2w.dac_post_snake_beta);
    x = causal_conv1d(x, c2w.dac_post_conv_w, c2w.dac_post_conv_b, 1, "dac_post_conv"); // [n_samples, 1]

    x = ggml_clamp(ctx0, x, -1.0f, 1.0f);
    x = ggml_reshape_1d(ctx0, x, x->ne[0]);
    cb(x, "wav_audio_out", -1);
    return x;
}

// code2wav's persisted state buffers: RoPE position counter, K/V per pre_transformer layer,
// left-context/tail per stateful conv. shape lookup only, no graph needed
std::vector<c2w_state_slot> list_c2w_state_slots(const clip_hparams & hparams, const clip_model & model) {
    const auto & c2w = model.c2w;
    std::vector<c2w_state_slot> slots;

    slots.push_back({"tfm_pos", 1, 1});

    // prefix is (W-1) frames, the batch itself gives the other N=W frames (see tfm_layer_forward)
    const int64_t d_head = c2w.tfm_layers[0].q_w->ne[1] / hparams.wav_tfm_n_head;
    const int64_t kv_ch  = d_head * hparams.wav_tfm_n_head_kv;
    const int64_t prefix = hparams.wav_tfm_swa - 1;
    for (int il = 0; il < hparams.wav_tfm_n_layer; il++) {
        slots.push_back({"tfm_k_" + std::to_string(il), kv_ch, prefix});
        slots.push_back({"tfm_v_" + std::to_string(il), kv_ch, prefix});
    }

    slots.push_back({"pre_conv", c2w.pre_conv_w->ne[0] - 1, c2w.pre_conv_w->ne[1]});

    for (size_t il = 0; il < c2w.upsample.size(); il++) {
        const auto & up = c2w.upsample[il];
        slots.push_back({"up" + std::to_string(il) + "_dwconv", up.dwconv_w->ne[0] - 1, up.dwconv_w->ne[2]});
    }

    slots.push_back({"dac_entry", c2w.dac_entry_w->ne[0] - 1, c2w.dac_entry_w->ne[1]});

    static constexpr int DAC_DILATIONS[3] = { 1, 3, 9 };
    for (size_t il = 0; il < c2w.dac.size(); il++) {
        const auto & blk = c2w.dac[il];
        const int64_t stride = blk.conv_w->ne[0] / 2; // kernel == 2*stride for all 4 blocks
        const std::string blk_name = "dac" + std::to_string(il);
        slots.push_back({blk_name + "_tail", stride, blk.conv_w->ne[1]});
        for (size_t ir = 0; ir < blk.res.size(); ir++) {
            const auto & res = blk.res[ir];
            slots.push_back({blk_name + "_res" + std::to_string(ir),
                              (res.conv1_w->ne[0] - 1) * DAC_DILATIONS[ir], res.conv1_w->ne[1]});
        }
    }

    slots.push_back({"dac_post_conv", c2w.dac_post_conv_w->ne[0] - 1, c2w.dac_post_conv_w->ne[1]});

    return slots;
}

// both sub-graphs are always built, so the topology stays constant
// ggml_build_forward_select() then picks the one that actually runs
ggml_cgraph * clip_graph_qwen3tts_gen::build() {
    GGML_ASSERT(n_batch == 1); // this module only ever processes one frame at a time

    int idx;
    switch (gen_process) {
        case CLIP_GEN_PROCESS_GEN_CODE: idx = 0; break;
        case CLIP_GEN_PROCESS_GEN_WAV:  idx = 1; break;
        default: GGML_ABORT("unknown gen_process");
    }

    // ---- CLIP_GEN_PROCESS_GEN_CODE: backbone hidden state -> 16 RVQ codes + next-step embd ----
    // not build_inp_raw(), a GEN_WAV call's `img` has no hidden-state data
    ggml_tensor * h_state = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_mmproj_embd);
    ggml_set_name(h_state, "inp_raw"); // must keep this exact name, clip_encode() sets it by name
    ggml_set_input(h_state);

    ggml_tensor * code0 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(code0, "inp_code0");
    ggml_set_input(code0);

    ggml_tensor * code0_embd = ggml_get_rows(ctx0, model.gen_code_out_embd_w, code0);
    code0_embd = ggml_reshape_1d(ctx0, code0_embd, code0_embd->ne[0]);
    cb(code0_embd, "code0_embd", -1);

    const int64_t n_acoustic = model.gen_code_head_w->ne[2]; // 15
    const int     n_codes    = (int) n_acoustic + 1;         // 16
    const int64_t n_kv_pad   = n_codes;
    const int     n_layer    = (int) model.layers.size();
    const int     n_head     = hparams.n_head;
    const int     n_head_kv  = hparams.n_head_kv;
    const int64_t d_head     = model.layers[0].q_w->ne[1] / n_head;

    // zero-filled per layer k/v caches, so masked-out rows can't hold garbage
    std::vector<ggml_tensor *> k_cache(n_layer), v_cache(n_layer);
    for (int il = 0; il < n_layer; il++) {
        k_cache[il] = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_head * n_head_kv, n_kv_pad), 0.0f);
        v_cache[il] = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_head * n_head_kv, n_kv_pad), 0.0f);
    }

    code_gen cg(*this, top_k, top_p);

    ggml_tensor * out_code_cache = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_codes);
    out_code_cache = cg.cache_set(out_code_cache, 0, code0);

    ggml_tensor * inp_rand0 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_set_name(inp_rand0, "inp_rand_0");
    ggml_set_input(inp_rand0);

    cg.prefill(k_cache, v_cache, out_code_cache, h_state, code0_embd, inp_rand0);

    for (int g = 1; g < n_acoustic; g++) {
        ggml_tensor * inp_rand = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
        ggml_set_name(inp_rand, ("inp_rand_" + std::to_string(g)).c_str());
        ggml_set_input(inp_rand);
        out_code_cache = cg.step(k_cache, v_cache, out_code_cache, inp_rand, g);
    }

    // output 1: this frame's 16 sampled codes, for the caller's code2wav window
    ggml_tensor * out_codes = ggml_cont(ctx0, out_code_cache);
    ggml_set_name(out_codes, "out_codes");
    ggml_set_output(out_codes);

    // output 2: sum of all 16 codebook embeddings, fed back to the talker for the next frame
    ggml_tensor * out_embd = code0_embd;
    for (int g = 1; g <= n_acoustic; g++) {
        ggml_tensor * code_g = ggml_view_1d(ctx0, out_code_cache, 1, (size_t) g * out_code_cache->nb[1]);

        ggml_tensor * embd_g = ggml_view_2d(ctx0, model.gen_code_embd_w, model.gen_code_embd_w->ne[0], model.gen_code_embd_w->ne[1],
                                            model.gen_code_embd_w->nb[1], (size_t) (g - 1) * model.gen_code_embd_w->nb[2]);
        ggml_tensor * e = ggml_get_rows(ctx0, embd_g, code_g);
        e = ggml_reshape_1d(ctx0, e, e->ne[0]);

        out_embd = ggml_add(ctx0, out_embd, e);
    }
    out_embd = ggml_reshape_2d(ctx0, out_embd, out_embd->ne[0], 1);
    cb(out_embd, "gen_audio_out", -1);

    // ---- CLIP_GEN_PROCESS_GEN_WAV: 16 RVQ codes -> raw PCM ----
    const int n_frames = hparams.wav_tfm_swa; // frames per batch, == the attention window

    ggml_tensor * inp_codes = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_frames, n_codes);
    ggml_set_name(inp_codes, "inp_codes");
    ggml_set_input(inp_codes);

    code2wav c2w(*this);
    for (const auto & slot : list_c2w_state_slots(hparams, model)) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, slot.ne0, slot.ne1);
        ggml_set_name(t, ("state_in_" + slot.name).c_str());
        ggml_set_input(t);
        c2w.state_in[slot.name] = t;
    }

    ggml_tensor * out_audio = c2w.decode(inp_codes);
    ggml_set_name(out_audio, "out_audio");
    ggml_set_output(out_audio);

    for (auto & slot : c2w.state_out) {
        ggml_set_name(slot.second, ("state_out_" + slot.first).c_str());
        ggml_set_output(slot.second);
    }

    // out_embd goes last, clip_encode() reads it back via ggml_graph_node(gf, -1)
    ggml_tensor * outs[2];
    outs[0] = out_codes; outs[1] = out_audio;
    ggml_build_forward_select(gf, outs, 2, idx);
    for (auto & slot : c2w.state_out) {
        outs[0] = out_codes; outs[1] = slot.second;
        ggml_build_forward_select(gf, outs, 2, idx);
    }
    outs[0] = out_embd; outs[1] = out_audio;
    ggml_build_forward_select(gf, outs, 2, idx);

    return gf;
}
