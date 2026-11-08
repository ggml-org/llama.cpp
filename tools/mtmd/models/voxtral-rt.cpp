#include "models.h"

// the input holds 2 context mel frames followed by the new frames, so the causal conv stem
// sees the same left context as on the full spectrogram
ggml_cgraph * clip_graph_voxtral_rt::build() {
    const int n_frames_in = img.nx();
    const int n_new       = (n_frames_in - 2) / 2; // encoder frames, the conv stem has stride 2
    const int n_window    = hparams.attn_window_size;
    const int stack       = hparams.proj_stack_factor;
    const int n_embd_kv   = n_head_kv * d_head;
    const int n_ctx       = state ? state->n_ctx : n_window + n_new;

    GGML_ASSERT(n_window > 0 && stack > 0);
    GGML_ASSERT(n_new > 0 && n_new % stack == 0);
    GGML_ASSERT(n_new <= n_ctx - n_window);

    ggml_tensor * inp = build_inp_raw(1); // [n_frames_in, n_mel]

    ggml_tensor * stem_tail = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_embd);
    ggml_set_name(stem_tail, "inp_stem_tail");
    ggml_set_input(stem_tail);

    ggml_tensor * pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_new);
    ggml_set_name(pos, "inp_pos");
    ggml_set_input(pos);

    ggml_tensor * kv_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_new);
    ggml_set_name(kv_idx, "inp_kv_idx");
    ggml_set_input(kv_idx);

    ggml_tensor * kq_mask = build_inp_attn_mask(n_ctx, n_new);

    // causal conv stem
    ggml_tensor * cur = ggml_conv_1d(ctx0, model.conv1d_1_w, inp, 1, 0, 1);
    cur = ggml_add(ctx0, cur, model.conv1d_1_b);
    cur = ggml_gelu_erf(ctx0, cur); // [2*n_new, n_embd]

    // the last frame is the left context of the second conv on the next call
    ggml_tensor * tail = ggml_view_2d(ctx0, cur, 1, n_embd, cur->nb[1], (2*n_new - 1) * ggml_element_size(cur));
    tail = ggml_cont(ctx0, tail);
    ggml_set_name(tail, "out_stem_tail");
    ggml_set_output(tail);
    ggml_build_forward_expand(gf, tail);

    cur = ggml_concat(ctx0, stem_tail, cur, 0); // [2*n_new + 1, n_embd]
    cur = ggml_conv_1d(ctx0, model.conv1d_2_w, cur, 2, 0, 1);
    cur = ggml_add(ctx0, cur, model.conv1d_2_b);
    cur = ggml_gelu_erf(ctx0, cur); // [n_new, n_embd]
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // [n_embd, n_new]
    cb(cur, "after_conv1d", -1);

    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];

        ggml_tensor * x = build_norm(cur, layer.ln_1_w, nullptr, NORM_TYPE_RMS, eps, il);

        ggml_tensor * q = ggml_add(ctx0, build_mm(layer.q_w, x), layer.q_b);
        ggml_tensor * k = build_mm(layer.k_w, x);
        ggml_tensor * v = ggml_add(ctx0, build_mm(layer.v_w, x), layer.v_b);

        q = ggml_reshape_3d(ctx0, q, d_head, n_head,    n_new);
        k = ggml_reshape_3d(ctx0, k, d_head, n_head_kv, n_new);

        // interleaved rope on absolute frame positions
        q = ggml_rope_ext(ctx0, q, pos, nullptr, d_head, GGML_ROPE_TYPE_NORMAL, 0, hparams.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx0, k, pos, nullptr, d_head, GGML_ROPE_TYPE_NORMAL, 0, hparams.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        cb(q, "q", il);
        cb(k, "k", il);
        cb(v, "v", il);

        ggml_tensor * k_cache;
        ggml_tensor * v_cache;
        if (state) {
            k_cache = state->k[il];
            v_cache = state->v[il];
        } else {
            k_cache = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_embd_kv, n_ctx);
            v_cache = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_embd_kv, n_ctx);
            ggml_set_input(k_cache);
            ggml_set_input(v_cache);
        }

        // write the new frames into the ring, the mask selects the window out of the whole ring
        ggml_build_forward_expand(gf, ggml_set_rows(ctx0, k_cache, ggml_reshape_2d(ctx0, k, n_embd_kv, n_new), kv_idx));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx0, v_cache, v, kv_idx));

        ggml_tensor * k_all = ggml_reshape_3d(ctx0, k_cache, d_head, n_head_kv, n_ctx);
        ggml_tensor * v_all = ggml_reshape_3d(ctx0, v_cache, d_head, n_head_kv, n_ctx);

        x = build_attn(layer.o_w, layer.o_b, q, k_all, v_all, kq_mask, kq_scale, il);
        cur = ggml_add(ctx0, cur, x);

        x = build_norm(cur, layer.ln_2_w, nullptr, NORM_TYPE_RMS, eps, il);
        x = build_ffn(x, layer.ff_up_w, nullptr, layer.ff_gate_w, nullptr, layer.ff_down_w, layer.ff_down_b, hparams.ffn_op, il);
        cur = ggml_add(ctx0, cur, x);
        cb(cur, "layer_out", il);
    }

    cur = build_norm(cur, model.post_ln_w, nullptr, NORM_TYPE_RMS, eps, -1);
    cb(cur, "after_transformer", -1);

    cur = build_stack(cur, stack, n_embd);
    cur = build_mm(model.mm_1_w, cur);
    cur = ggml_gelu_erf(ctx0, cur);
    cur = build_mm(model.mm_2_w, cur);
    cb(cur, "projected", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
