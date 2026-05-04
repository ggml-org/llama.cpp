#include "models.h"

// MiMoVL vision tower for MiMo-V2.5 (non-Pro). Qwen2.5-VL-shaped ViT, except:
//   1. GQA in attention (32 Q / 8 KV heads, head_dim 64). The converter
//      splits the fused qkv weight; ggml_mul_mat broadcasts n_head_kv to
//      n_head via interleave.
//   2. Per-head attention sinks on every windowed layer. These follow the
//      GPT-OSS/FA3 semantic that sglang's reference impl and b12x's SM120
//      kernel use: the sinks adjust the softmax denominator via row_scale
//      (equivalently, a virtual extra K column with V=0), so they decay
//      attention weight without contributing to the output.
//   3. Per-layer window-attention mode in hparams.wa_pattern_mode:
//        -1 -> full,  0 -> row-window+sinks,  1 -> col-window+sinks.
//      Col mode transposes the merge-unit grid on entry and restores
//      it on exit. Both patch and rotary orderings are pre-computed
//      host-side.
//   4. 1D banded sliding window (|q-k| > window_size -> -inf) as a
//      single 2D mask broadcast across heads.
//   5. Per-block MLP biases.
// Everything else (patch embed Conv3D split, rotary, RMSNorm, merger)
// is identical to Qwen2.5-VL.
ggml_cgraph * clip_graph_mimovl::build() {
    GGML_ASSERT(model.patch_embeddings_0 != nullptr);
    GGML_ASSERT(model.patch_embeddings_1 != nullptr);
    GGML_ASSERT(model.class_embedding == nullptr);
    GGML_ASSERT(hparams.n_head_kv > 0);
    GGML_ASSERT(n_head % hparams.n_head_kv == 0);
    GGML_ASSERT((int) hparams.wa_pattern_mode.size() == n_layer);

    const int batch_size = 1;
    const int n_pos      = n_patches;
    const int n_head_kv  = hparams.n_head_kv;
    const int merge      = hparams.n_merge > 0 ? hparams.n_merge : 2;
    const int merge_unit = merge * merge;
    const int n_units    = n_pos / merge_unit;
    GGML_ASSERT(n_units * merge_unit == n_pos);

    // MiMoVL has head_dim=64 with n_embd=1280, so n_embd is NOT n_head*head_dim
    // (the base class's d_head = n_embd/n_head = 40 is wrong here). Derive
    // head_dim from the Q projection's output width.
    GGML_ASSERT(model.layers[0].q_w != nullptr);
    const int head_dim     = model.layers[0].q_w->ne[1] / n_head;
    const float attn_scale = 1.0f / std::sqrt((float) head_dim);

    // ggml_rope_multi VISION layout: the non-CPU kernels only read positions from
    // slot 0 (sections[0] pairs) and slot 1 (sections[1] pairs); slots 2,3 are
    // ignored even though the CPU kernel honors them. h_pos is packed in slot 0 and w_pos
    // in slot 1, and split the rotary dims half/half between them.
    const int rope_n_dims = head_dim / 2;
    int mrope_sections[4] = {rope_n_dims/2, rope_n_dims/2, 0, 0};

    // Patch embed: Conv3D(kt=2) split into two Conv2D, then interleave-merge
    // along the height axis to match the merge-tile token order.
    ggml_tensor * inp_raw = build_inp_raw();
    ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw,
                                     patch_size, patch_size, 0, 0, 1, 1);
    {
        ggml_tensor * inp_1 = ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw,
                                           patch_size, patch_size, 0, 0, 1, 1);
        inp = ggml_add(ctx0, inp, inp_1);

        GGML_ASSERT(img.nx % (patch_size * 2) == 0);
        GGML_ASSERT(img.ny % (patch_size * 2) == 0);

        inp = ggml_permute(ctx0, inp, 1, 2, 0, 3);  // [w,h,c,b] -> [c,w,h,b]
        inp = ggml_cont_4d(ctx0, inp, n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
        inp = ggml_reshape_4d(ctx0, inp, n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
        inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = ggml_cont_3d(ctx0, inp, n_embd, n_patches_x * n_patches_y, batch_size);
    }
    cb(inp, "patch_embed", -1);

    ggml_tensor * positions_row = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos * 4);
    ggml_set_name(positions_row, "mimovl_positions_row");
    ggml_set_input(positions_row);

    ggml_tensor * positions_col = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos * 4);
    ggml_set_name(positions_col, "mimovl_positions_col");
    ggml_set_input(positions_col);

    // idx_col is the col-major merge-unit permutation. Take it as F32 so we can
    // derive the inverse permutation in-graph via ggml_argsort (returning I32);
    // ggml_get_rows requires its index tensor to be I32, so cast back as well.
    ggml_tensor * idx_col_f = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_units);
    ggml_set_name(idx_col_f, "mimovl_idx_col");
    ggml_set_input(idx_col_f);
    ggml_tensor * idx_col     = ggml_cast(ctx0, idx_col_f, GGML_TYPE_I32);
    ggml_tensor * idx_col_inv = ggml_argsort(ctx0, idx_col_f, GGML_SORT_ORDER_ASC);

    ggml_tensor * window_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_pos, n_pos);
    ggml_set_name(window_mask, "mimovl_window_mask");
    ggml_set_input(window_mask);

    // Reorder helper: permute patches at merge-unit granularity. The patch
    // sequence is laid out as n_units groups of merge_unit (=4) consecutive
    // patches; the row<->col transpose only permutes whole groups. We keep
    // the per-group (h,w) ordering intact by reshaping to
    // [n_embd*merge_unit, n_units] before ggml_get_rows.
    auto reorder = [&](ggml_tensor * x, ggml_tensor * idx) {
        ggml_tensor * y = ggml_reshape_2d(ctx0, x, n_embd * merge_unit, n_units);
        y = ggml_get_rows(ctx0, y, idx);
        return ggml_reshape_3d(ctx0, y, n_embd, n_pos, batch_size);
    };

    ggml_tensor * inpL = inp;
    int prev_mode = -1;

    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];
        const int  mode    = hparams.wa_pattern_mode[il];
        const bool is_full = (mode == -1);
        const bool is_col  = (mode == 1);

        // Reorder transitions on entry/exit of a col-mode run.
        if (is_col && prev_mode != 1) {
            inpL = reorder(inpL, idx_col);
            cb(inpL, "reorder_to_col", il);
        } else if (!is_col && prev_mode == 1) {
            inpL = reorder(inpL, idx_col_inv);
            cb(inpL, "reorder_to_row", il);
        }

        ggml_tensor * cur = inpL;

        // Pre-attention RMSNorm.
        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_RMS, eps, il);
        cb(cur, "ln1", il);

        // Q/K/V projections + biases (GQA: Q has n_head, K/V have n_head_kv).
        ggml_tensor * Qcur = ggml_add(ctx0, build_mm(layer.q_w, cur), layer.q_b);
        ggml_tensor * Kcur = ggml_add(ctx0, build_mm(layer.k_w, cur), layer.k_b);
        ggml_tensor * Vcur = ggml_add(ctx0, build_mm(layer.v_w, cur), layer.v_b);

        Qcur = ggml_reshape_3d(ctx0, Qcur, head_dim, n_head,    n_pos);
        Kcur = ggml_reshape_3d(ctx0, Kcur, head_dim, n_head_kv, n_pos);
        Vcur = ggml_reshape_3d(ctx0, Vcur, head_dim, n_head_kv, n_pos);

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        // 2D RoPE
        ggml_tensor * pos = is_col ? positions_col : positions_row;
        Qcur = ggml_rope_multi(ctx0, Qcur, pos, nullptr, rope_n_dims, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        Kcur = ggml_rope_multi(ctx0, Kcur, pos, nullptr, rope_n_dims, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        cb(Qcur, "Qcur_rope", il);
        cb(Kcur, "Kcur_rope", il);

        // Attention.
        //
        // Full layers use plain build_attn (no mask, no sinks). Windowed
        // layers go through ggml_flash_attn_ext with the row/col window
        // mask and ggml_flash_attn_ext_add_sinks for the per-head sink.
        // Non-FA fallback uses the equivalent manual concat+pad trick:
        // append a virtual sink K column to kq and zero-pad V's K dim,
        // so the sink shows up in the softmax denominator but contributes
        // 0 to the weighted V sum.
        ggml_tensor * attn_out;
        if (is_full) {
            attn_out = build_attn(layer.o_w, layer.o_b,
                                  Qcur, Kcur, Vcur, nullptr, attn_scale, il);
        } else if (flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) {
            ggml_build_forward_expand(gf, Qcur);
            ggml_build_forward_expand(gf, Kcur);
            ggml_build_forward_expand(gf, Vcur);

            ggml_tensor * q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
            ggml_tensor * k = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);
            ggml_tensor * v = ggml_permute(ctx0, Vcur, 0, 2, 1, 3);
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
            ggml_tensor * mask_fa = ggml_cast(ctx0, window_mask, GGML_TYPE_F16);

            ggml_tensor * cur_attn = ggml_flash_attn_ext(ctx0, q, k, v, mask_fa, attn_scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(cur_attn, GGML_PREC_F32);
            GGML_ASSERT(layer.attn_sinks != nullptr);
            ggml_flash_attn_ext_add_sinks(cur_attn, layer.attn_sinks);
            cur_attn = ggml_reshape_2d(ctx0, cur_attn,
                                       cur_attn->ne[0] * cur_attn->ne[1],
                                       cur_attn->ne[2] * cur_attn->ne[3]);
            cb(cur_attn, "kqv_out", il);

            attn_out = build_mm(layer.o_w, cur_attn);
            attn_out = ggml_add(ctx0, attn_out, layer.o_b);
        } else {
            ggml_build_forward_expand(gf, Qcur);
            ggml_build_forward_expand(gf, Kcur);
            ggml_build_forward_expand(gf, Vcur);

            ggml_tensor * q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
            ggml_tensor * k = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);
            ggml_tensor * v = ggml_permute(ctx0, Vcur, 1, 2, 0, 3);
            v = ggml_cont(ctx0, v);

            ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
            kq = ggml_scale(ctx0, kq, attn_scale);
            kq = ggml_add(ctx0, kq, window_mask);
            cb(kq, "kq_masked", il);

            GGML_ASSERT(layer.attn_sinks != nullptr);
            {
                ggml_tensor * s = ggml_reshape_3d(ctx0, layer.attn_sinks, 1, 1, n_head);
                ggml_tensor * sink_target = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_pos, n_head);
                ggml_tensor * sink_col = ggml_repeat(ctx0, s, sink_target);
                kq = ggml_concat(ctx0, kq, sink_col, 0);   // (n_pos+1, n_pos, n_head)
                v  = ggml_pad(ctx0, v, 1, 0, 0, 0);        // (n_pos+1, head_dim, n_head_kv)
                cb(kq, "kq_sink_ext", il);
            }

            kq = ggml_soft_max(ctx0, kq);

            ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
            ggml_tensor * cur_attn = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
            cur_attn = ggml_cont_2d(ctx0, cur_attn, cur_attn->ne[0] * cur_attn->ne[1], cur_attn->ne[2] * cur_attn->ne[3]);
            cb(cur_attn, "kqv_out", il);

            attn_out = build_mm(layer.o_w, cur_attn);
            attn_out = ggml_add(ctx0, attn_out, layer.o_b);
        }
        cb(attn_out, "attn_out", il);

        // Residual 1.
        cur = ggml_add(ctx0, attn_out, inpL);
        inpL = cur;
        cb(cur, "ffn_inp", il);

        // Pre-FFN RMSNorm.
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_RMS, eps, il);
        cb(cur, "ffn_inp_normed", il);

        // SwiGLU MLP with biases
        cur = build_ffn(cur, 
            layer.ff_up_w,   layer.ff_up_b, 
            layer.ff_gate_w, layer.ff_gate_b, 
            layer.ff_down_w, layer.ff_down_b, 
            hparams.ffn_op, il);
        cb(cur, "ffn_out", il);

        // Residual 2.
        cur = ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);

        inpL = cur;
        prev_mode = mode;
    }

    // If the last block was col-mode, undo the transpose so the merger sees patches in row order.
    if (prev_mode == 1) {
        inpL = reorder(inpL, idx_col_inv);
        cb(inpL, "reorder_to_row_final", -1);
    }

    // Merger: post-LayerNorm
    inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, NORM_TYPE_NORMAL, 1e-6f, n_layer);
    cb(inpL, "post_ln", -1);

    // Spatial merge: pack each merge_unit (=4) of patches into a single
    // (n_embd*merge_unit)-wide row, then run the 2-layer MLP (no biases).
    ggml_tensor * embeddings = ggml_reshape_3d(ctx0, inpL, n_embd * merge_unit, n_units, batch_size);
    embeddings = build_ffn(embeddings,
        model.mm_0_w, nullptr,
        nullptr,      nullptr,
        model.mm_1_w, nullptr,
        FFN_GELU, -1);
    cb(embeddings, "vit_out", -1);

    ggml_build_forward_expand(gf, embeddings);
    return gf;
}
