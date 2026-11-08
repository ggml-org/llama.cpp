#include "mtmd-graph.h"
#include "models/models.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

//
// mtmd_graph
//

mtmd_graph::mtmd_graph(mtmd_subcontext * ctx, const mtmd_image_f32 & img) :
        model(ctx->model),
        hparams(model.hparams),
        proj_type(ctx->model.proj_type),
        img(img),
        patch_size(hparams.patch_size),
        n_patches_x(img.nx() / patch_size),
        n_patches_y(img.ny() / patch_size),
        n_patches(n_patches_x * n_patches_y),
        n_embd(hparams.n_embd),
        n_head(hparams.n_head),
        n_head_kv(hparams.n_head_kv),
        d_head(hparams.n_embd_head > 0 ? hparams.n_embd_head : (n_head > 0 ? n_embd / n_head : 0)),
        n_layer(hparams.n_layer),
        n_mmproj_embd(ctx->model.n_mmproj_embd()),
        eps(hparams.eps),
        kq_scale(d_head > 0 ? 1.0f / sqrtf((float)d_head) : 0.0f),
        flash_attn_type(ctx->flash_attn_type) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx->buf_compute_meta.size(),
        /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    ctx0_ptr.reset(ggml_init(params));
    ctx0 = ctx0_ptr.get();
    gf = ggml_new_graph_custom(ctx0, ctx->max_nodes, false);
}

mtmd_graph::mtmd_graph(const mtmd_graph & parent) :
        model(parent.model),
        hparams(parent.hparams),
        proj_type(parent.proj_type),
        img(parent.img),
        patch_size(parent.patch_size),
        n_patches_x(parent.n_patches_x),
        n_patches_y(parent.n_patches_y),
        n_patches(parent.n_patches),
        n_embd(parent.n_embd),
        n_head(parent.n_head),
        n_head_kv(parent.n_head_kv),
        d_head(parent.d_head),
        n_layer(parent.n_layer),
        n_mmproj_embd(parent.n_mmproj_embd),
        eps(parent.eps),
        kq_scale(parent.kq_scale),
        flash_attn_type(parent.flash_attn_type) {
    // reuse from parent
    ctx0 = parent.ctx0;
    gf   = parent.gf;
}

ggml_tensor * mtmd_graph::build_mm(ggml_tensor * w, ggml_tensor * x) const {
    return ggml_mul_mat(ctx0, w, x);
}

void mtmd_graph::cb(ggml_tensor * cur, const char * name, int il) const {
    if (il >= 0) {
        ggml_format_name(cur, "%s-%d", name, il);
    } else {
        ggml_set_name(cur, name);
    }
}

// siglip2 naflex
ggml_tensor * mtmd_graph::resize_position_embeddings(uint32_t interpolation_mode) {
    ggml_tensor * pos_embd = model.position_embeddings;
    const int height       = img.ny() / patch_size;
    const int width        = img.nx() / patch_size;
    const uint32_t mode    = interpolation_mode;
    const int n_per_side   = (int)std::sqrt(pos_embd->ne[1]);

    GGML_ASSERT(pos_embd);

    if (height == n_per_side && width == n_per_side) {
        return pos_embd;
    }

    pos_embd = ggml_reshape_3d(ctx0, pos_embd, n_embd, n_per_side, n_per_side);  // -> (n_embd, n_per_side, n_per_side)
    pos_embd = ggml_permute(ctx0, pos_embd, 2, 0, 1, 3);                         // -> (n_per_side, n_per_side, n_embd)
    pos_embd = ggml_interpolate(ctx0, pos_embd, width, height, n_embd, 1, mode); // -> (width, height, n_embd)
    pos_embd = ggml_permute(ctx0, pos_embd, 1, 2, 0, 3);                         // -> (n_embd, width, height)
    pos_embd = ggml_cont_2d(ctx0, pos_embd, n_embd, width * height);             // -> (n_embd, width * height)

    return pos_embd;
}

// build vision transformer (ViT) cgraph
// this function should cover most of the models
// if your model has specific features, you should probably duplicate this function
ggml_tensor * mtmd_graph::build_vit(
            ggml_tensor * inp,
            int64_t n_pos,
            norm_type norm_t,
            ffn_op_type ffn_t,
            ggml_tensor * learned_pos_embd,
            std::function<ggml_tensor *(ggml_tensor *, const mtmd_layer &)> add_pos,
            const build_vit_opts & opts
        ) {
    // batch dim: inp is [n_embd, n_pos, B]
    const int64_t B = inp->ne[2];

    if (learned_pos_embd) {
        inp = ggml_add(ctx0, inp, learned_pos_embd);
        cb(inp, "pos_embed", -1);
    }

    // flatten batch; unflatten again in attention
    inp = ggml_reshape_2d(ctx0, inp, n_embd, n_pos * B);

    ggml_tensor * inpL = inp;

    // pre-layernorm
    if (model.pre_ln_w) {
        inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, norm_t, eps, -1);
        cb(inpL, "pre_ln", -1);
    }

    // loop over layers
    for (int il = 0; il < n_layer; il++) {
        auto & layer = model.layers[il];
        ggml_tensor * cur = inpL; // inpL = residual, cur = hidden_states

        ggml_tensor * attn_mask = opts.attn_mask;
        if (opts.attn_mask_layers.size() > (size_t) il) {
            attn_mask = opts.attn_mask_layers[il];
        }

        // layernorm1
        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
        cb(cur, "layer_inp_normed", il);

        // self-attention
        {
            ggml_tensor * Qcur = nullptr;
            ggml_tensor * Kcur = nullptr;
            ggml_tensor * Vcur = nullptr;
            if (layer.qkv_w != nullptr) {
                // fused qkv
                cur = build_mm(layer.qkv_w, cur);
                if (layer.qkv_b != nullptr) {
                    cur = ggml_add(ctx0, cur, layer.qkv_b);
                }

                // Q/K/V as [d_head, n_head, n_pos, B], the batch stride is cur->nb[1]*n_pos.
                Qcur = ggml_view_4d(ctx0, cur, d_head, n_head, n_pos, B,
                /* nb1    */ ggml_row_size(cur->type, d_head),
                /* nb2    */ cur->nb[1],
                /* nb3    */ cur->nb[1] * n_pos,
                /* offset */ 0);

                Kcur = ggml_view_4d(ctx0, cur, d_head, n_head, n_pos, B,
                /* nb1    */ ggml_row_size(cur->type, d_head),
                /* nb2    */ cur->nb[1],
                /* nb3    */ cur->nb[1] * n_pos,
                /* offset */ ggml_row_size(cur->type, n_head * d_head));

                Vcur = ggml_view_4d(ctx0, cur, d_head, n_head, n_pos, B,
                /* nb1    */ ggml_row_size(cur->type, d_head),
                /* nb2    */ cur->nb[1],
                /* nb3    */ cur->nb[1] * n_pos,
                /* offset */ ggml_row_size(cur->type, 2 * n_head * d_head));

                if (layer.q_norm) {
                    GGML_ASSERT(layer.q_norm->ne[0] == Qcur->ne[0]);
                    Qcur = build_norm(Qcur, layer.q_norm, layer.q_norm_b, norm_t, eps, il);
                    cb(Qcur, "Qcur_norm", il);
                }

                if (layer.k_norm) {
                    GGML_ASSERT(layer.k_norm->ne[0] == Kcur->ne[0]);
                    Kcur = build_norm(Kcur, layer.k_norm, layer.k_norm_b, norm_t, eps, il);
                    cb(Kcur, "Kcur_norm", il);
                }

            } else {
                // separate q, k, v
                Qcur = build_mm(layer.q_w, cur);
                if (layer.q_b) {
                    Qcur = ggml_add(ctx0, Qcur, layer.q_b);
                }

                Kcur = build_mm(layer.k_w, cur);
                if (layer.k_b) {
                    Kcur = ggml_add(ctx0, Kcur, layer.k_b);
                }

                Vcur = build_mm(layer.v_w, cur);
                if (layer.v_b) {
                    Vcur = ggml_add(ctx0, Vcur, layer.v_b);
                }

                // if true, norm must be applied after reshaping to (d_head, n_head, n_pos)
                bool norm_per_head = layer.q_norm && layer.q_norm->ne[0] == d_head;

                if (!norm_per_head) {
                    if (layer.q_norm) {
                        Qcur = build_norm(Qcur, layer.q_norm, layer.q_norm_b, norm_t, eps, il);
                        cb(Qcur, "Qcur_norm", il);
                    }
                    if (layer.k_norm) {
                        Kcur = build_norm(Kcur, layer.k_norm, layer.k_norm_b, norm_t, eps, il);
                        cb(Kcur, "Kcur_norm", il);
                    }
                }

                Qcur = ggml_reshape_4d(ctx0, Qcur, d_head, n_head,    n_pos, B);
                Kcur = ggml_reshape_4d(ctx0, Kcur, d_head, n_head_kv, n_pos, B);
                Vcur = ggml_reshape_4d(ctx0, Vcur, d_head, n_head_kv, n_pos, B);

                if (norm_per_head) {
                    if (layer.q_norm) {
                        Qcur = build_norm(Qcur, layer.q_norm, layer.q_norm_b, norm_t, eps, il);
                        cb(Qcur, "Qcur_norm_per_head", il);
                    }
                    if (layer.k_norm) {
                        Kcur = build_norm(Kcur, layer.k_norm, layer.k_norm_b, norm_t, eps, il);
                        cb(Kcur, "Kcur_norm_per_head", il);
                    }
                }
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            if (add_pos) {
                Qcur = add_pos(Qcur, layer);
                Kcur = add_pos(Kcur, layer);
                cb(Qcur, "Qcur_pos", il);
                cb(Kcur, "Kcur_pos", il);
            }

            if (proj_type == PROJECTOR_TYPE_GEMMA4V) {
                Vcur = ggml_rms_norm(ctx0, Vcur, eps);
                cb(Vcur, "Vcur_normed", il);
            }

            // build_attn returns a flat 2D [n_embd, n_pos*B]
            cur = build_attn(layer.o_w, layer.o_b,
                Qcur, Kcur, Vcur, attn_mask, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        if (layer.ls_1_w) {
            cur = ggml_mul(ctx0, cur, layer.ls_1_w);
            cb(cur, "attn_out_scaled", il);
        }

        if (layer.attn_post_norm_w) {
            cur = build_norm(cur, layer.attn_post_norm_w, nullptr, norm_t, eps, il);
            cb(cur, "attn_post_normed", il);
        }

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, inpL);

        inpL = cur; // inpL = residual, cur = hidden_states

        if (opts.callback_layer_out) {
            opts.callback_layer_out(cur, il);
        }

        cb(cur, "ffn_inp", il);

        // layernorm2 (pre-ffn norm)
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);
        cb(cur, "ffn_inp_normed", il);

        // ffn
        cur = layer.ff_gate_exps_w
            ? build_moe_ffn(cur, layer, ffn_t, il)
            : build_ffn(cur,
                layer.ff_up_w, layer.ff_up_b,
                layer.ff_gate_w, layer.ff_gate_b,
                layer.ff_down_w, layer.ff_down_b,
                ffn_t, il);

        cb(cur, "ffn_out", il);

        if (layer.ff_post_norm_w) {
            cur = build_norm(cur, layer.ff_post_norm_w, nullptr, norm_t, eps, il);
            cb(cur, "ffn_post_normed", il);
        }

        if (layer.ls_2_w) {
            cur = ggml_mul(ctx0, cur, layer.ls_2_w);
            cb(cur, "ffn_out_scaled", il);
        }

        // residual 2
        cur = ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);

        if (layer.ls_out_w) {
            cur = ggml_mul(ctx0, cur, layer.ls_out_w);
            cb(cur, "layer_out_scaled", il);
        }

        inpL = cur;
    }

    if (model.audio_has_avgpool()) {
        ggml_tensor * cur = inpL;
        cur = ggml_transpose(ctx0, cur);
        cur = ggml_cont(ctx0, cur);
        cur = ggml_pool_1d(ctx0, cur, GGML_OP_POOL_AVG, 2, 2, 0);
        cur = ggml_transpose(ctx0, cur);
        cur = ggml_cont(ctx0, cur);
        inpL = cur;
    }

    // post-layernorm
    if (model.post_ln_w && !opts.skip_post_ln) {
        inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, norm_t, eps, -1);
    }

    // restore the batch dim
    GGML_ASSERT(inpL->ne[1] % B == 0);
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, inpL->ne[1] / B, B);
    return inpL;
}

// build the input after conv2d (inp_raw --> patches)
// returns tensor with shape [n_embd, n_patches]
ggml_tensor * mtmd_graph::build_inp() {
    ggml_tensor * inp_raw = build_inp_raw();
    ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
    inp = ggml_reshape_3d(ctx0, inp, n_patches, n_embd, n_batch);
    inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
    if (model.patch_bias) {
        inp = ggml_add(ctx0, inp, model.patch_bias);
        cb(inp, "patch_bias", -1);
    }
    return inp;
}

ggml_tensor * mtmd_graph::build_inp_raw(int channels) {
    ggml_tensor * inp_raw = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, img.nx(), img.ny(), channels, n_batch);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);
    return inp_raw;
}

// qwen-vl style temporal merge: Conv3D(kt=2) split into 2 Conv2D
ggml_tensor * mtmd_graph::build_inp_temporal_merge() {
    ggml_tensor * inp_raw = build_inp_raw();

    GGML_ASSERT(img.nx() % (patch_size * 2) == 0);
    GGML_ASSERT(img.ny() % (patch_size * 2) == 0);

    const size_t nb1 = ggml_row_size(inp_raw->type, img.nx());
    const size_t nb2 = ggml_row_size(inp_raw->type, img.nx() * img.ny());

    if (n_batch == 1) {
        // still image input
        return ggml_add(ctx0,
            ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1),
            ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, patch_size, patch_size, 0, 0, 1, 1));
    } else if (n_batch == 2) {
        // 2 frames input (video input)
        ggml_tensor * inp_0 = ggml_view_3d(ctx0, inp_raw,
                                    img.nx(), img.ny(), 3, nb1, nb2, 0);
        ggml_tensor * inp_1 = ggml_view_3d(ctx0, inp_raw,
                                    img.nx(), img.ny(), 3, nb1, nb2,
                                    nb2 * 3); // move to the second frame
        return ggml_add(ctx0,
            ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_0, patch_size, patch_size, 0, 0, 1, 1),
            ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_1, patch_size, patch_size, 0, 0, 1, 1));
    } else {
        GGML_ASSERT(false && "n_batch > 2 is not supported");
    }
}

ggml_tensor * mtmd_graph::build_inp_attn_mask(int64_t n_kv, int64_t n_q, int idx) {
    const ggml_type type = flash_attn_type == MTMD_FLASH_ATTN_TYPE_ENABLED ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ggml_tensor * mask = ggml_new_tensor_2d(ctx0, type, n_kv, n_q);
    ggml_set_name(mask, get_attn_mask_name(idx).c_str());
    ggml_set_input(mask);
    return mask;
}

ggml_tensor * mtmd_graph::build_norm(
        ggml_tensor * cur,
        ggml_tensor * mw,
        ggml_tensor * mb,
        norm_type type,
        float norm_eps,
        int il) const {

    cur = type == NORM_TYPE_RMS
        ? ggml_rms_norm(ctx0, cur, norm_eps)
        : ggml_norm(ctx0, cur, norm_eps);

    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
        cb(cur, "norm_w", il);
    }

    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
        cb(cur, "norm_b", il);
    }

    return cur;
}

ggml_tensor * mtmd_graph::build_ffn(
        ggml_tensor * cur,
        ggml_tensor * up,
        ggml_tensor * up_b,
        ggml_tensor * gate,
        ggml_tensor * gate_b,
        ggml_tensor * down,
        ggml_tensor * down_b,
        ffn_op_type type_op,
        int il) const {

    ggml_tensor * tmp = up ? build_mm(up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (gate) {
        cur = build_mm(gate, cur);
        cb(cur, "ffn_gate", il);

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }
    } else {
        cur = tmp;
    }

    // we only support parallel ffn for now
    switch (type_op) {
        case FFN_SILU:
            if (gate) {
                cur = hparams.swiglu_clamp > 0.0f
                    ? ggml_swiglu_clamp(ctx0, cur, tmp, hparams.swiglu_clamp)
                    : ggml_swiglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_swiglu", il);
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case FFN_GELU:
            if (gate) {
                cur = ggml_geglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu", il);
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_gelu", il);
            } break;
        case FFN_GELU_ERF:
            if (gate) {
                cur = ggml_geglu_erf_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu_erf", il);
            } else {
                cur = ggml_gelu_erf(ctx0, cur);
                cb(cur, "ffn_gelu_erf", il);
            } break;
        case FFN_GELU_QUICK:
            if (gate) {
                cur = ggml_geglu_quick_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu_quick", il);
            } else {
                cur = ggml_gelu_quick(ctx0, cur);
                cb(cur, "ffn_gelu_quick", il);
            } break;
        case FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx0, cur);
                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_relu_sqr", il);
            } break;
    }

    if (down) {
        cur = build_mm(down, cur);
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }

    return cur;
}

// MoE FFN with sigmoid router and normalized top-k weights (dots3note vision)
// the router runs in fp32; exp_probs_b only affects expert selection, not the weights
ggml_tensor * mtmd_graph::build_moe_ffn(ggml_tensor * cur, const mtmd_layer & layer, ffn_op_type type_op, int il) const {
    const int64_t n_tokens      = cur->ne[1];
    const int64_t n_expert      = layer.ff_gate_exps_w->ne[2];
    const int64_t n_expert_used = std::min((int64_t) hparams.n_expert_used, n_expert);
    GGML_ASSERT(n_expert_used > 0);
    GGML_ASSERT(type_op == FFN_SILU);

    ggml_tensor * probs = ggml_sigmoid(ctx0, build_mm(layer.ff_gate_inp_w, cur)); // [n_expert, n_tokens]
    cb(probs, "ffn_moe_probs", il);

    ggml_tensor * sel = layer.ff_exp_probs_b
        ? ggml_add(ctx0, probs, layer.ff_exp_probs_b)
        : probs;
    ggml_tensor * selected = ggml_top_k(ctx0, sel, n_expert_used); // [n_expert_used, n_tokens]

    ggml_tensor * weights = ggml_get_rows(ctx0,
        ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens), selected);
    weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
    weights = ggml_div(ctx0, weights, ggml_sum_rows(ctx0, weights));
    weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
    cb(weights, "ffn_moe_weights", il);

    cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, n_tokens);
    ggml_tensor * gate = ggml_mul_mat_id(ctx0, layer.ff_gate_exps_w, cur, selected); // [n_ff, n_expert_used, n_tokens]
    ggml_tensor * up   = ggml_mul_mat_id(ctx0, layer.ff_up_exps_w,   cur, selected);
    cur = ggml_mul(ctx0, ggml_silu(ctx0, gate), up);
    cur = ggml_mul_mat_id(ctx0, layer.ff_down_exps_w, cur, selected); // [n_embd, n_expert_used, n_tokens]
    cur = ggml_mul(ctx0, cur, weights);

    // sum over the selected experts
    ggml_tensor * out = nullptr;
    for (int64_t i = 0; i < n_expert_used; i++) {
        ggml_tensor * v = ggml_view_2d(ctx0, cur, cur->ne[0], n_tokens, cur->nb[2], i * cur->nb[1]);
        out = out ? ggml_add(ctx0, out, v) : v;
    }
    if (n_expert_used == 1) {
        out = ggml_cont(ctx0, out);
    }
    cb(out, "ffn_moe_out", il);
    return out;
}

ggml_tensor * mtmd_graph::build_attn(
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_mask,
        float kq_scale,
        int il,
        ggml_tensor * sinks) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // the order is fixed without the compute flag, so an unselected branch stays out of the compute set
    ggml_build_forward_order(gf, q_cur);
    ggml_build_forward_order(gf, k_cur);
    ggml_build_forward_order(gf, v_cur);

    ggml_tensor * q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
    //cb(q, "q", il);

    ggml_tensor * k = ggml_permute(ctx0, k_cur, 0, 2, 1, 3);
    //cb(k, "k", il);

    ggml_tensor * cur;

    if (flash_attn_type == MTMD_FLASH_ATTN_TYPE_ENABLED) {
        ggml_tensor * v = ggml_permute(ctx0, v_cur, 0, 2, 1, 3);

        k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        // mask must be f16 here, use build_inp_attn_mask()
        GGML_ASSERT(!kq_mask || kq_mask->type == GGML_TYPE_F16);

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, 0.0f, 0.0f);
        ggml_prec_set_acc(cur, GGML_PREC_F32);
        if (sinks != nullptr) {
            ggml_flash_attn_ext_add_sinks(cur, sinks);
        }

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

    } else {
        ggml_tensor * v = ggml_permute(ctx0, v_cur, 1, 2, 0, 3);
        v = ggml_cont(ctx0, v);

        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        // TODO: this most likely requires GGML_PREC_BF16, but will enable it later to avoid scope-creep
        // ggml_prec_set_acc(kq, GGML_PREC_BF16);

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, 0.0f);
        if (sinks != nullptr) {
            ggml_soft_max_add_sinks(kq, sinks);
        }

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
    }

    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_mm(wo, cur);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

// implementation of the 2D RoPE without adding a new op in ggml
ggml_tensor * mtmd_graph::build_rope_2d(
    ggml_context * ctx0,
    ggml_tensor * cur,
    ggml_tensor * pos_a, // first half
    ggml_tensor * pos_b, // second half
    const float freq_base,
    const bool interleave_freq
) {
    const int64_t n_dim = cur->ne[0];

    // for example, if we have cur tensor of shape (n_dim=8, n_head, n_pos)
    // we will have a list of 4 inv_freq: 1e-0, 1e-1, 1e-2, 1e-3
    // first half of cur will use 1e-0, 1e-2 (even)
    // second half of cur will use 1e-1, 1e-3 (odd)
    // the trick here is to rotate just half of n_dim, so inv_freq will automatically be even
    //  ^ don't ask me why, it's math! -2(2i) / n_dim == -2i / (n_dim/2)
    // then for the second half, we use freq_scale to shift the inv_freq
    //  ^ why? replace (2i) with (2i+1) in the above equation
    const float freq_scale_odd = interleave_freq
                                ? std::pow(freq_base, (float)-2/n_dim)
                                : 1.0;

    // first half, dims [0, n_dim/2)
    cur = ggml_rope_ext(
        ctx0,
        cur,
        pos_a,      // positions
        nullptr,    // freq factors
        n_dim/2,    // n_dims
        0, 0, freq_base,
        1.0f, 0.0f, 1.0f, 0.0f, 0.0f
    );

    // second half, dims [n_dim/2, n_dim)
    cur = ggml_rope_ext(
        ctx0,
        cur,
        pos_b,      // positions
        nullptr,    // freq factors
        n_dim/2,    // n_dims
        0, 0, freq_base,
        freq_scale_odd,
        0.0f, 1.0f, 0.0f, 0.0f
    );
    cur = ggml_rope_set_offset(cur, n_dim/2);

    return cur;
}

// Generic function to stack frames for audio processing
// Abstracts out the StackAudioFrames logic used by ultravox
ggml_tensor * mtmd_graph::build_stack(ggml_tensor * cur, int32_t stack_factor, int32_t n_embed) {
    if (stack_factor <= 1) {
        return cur;
    }

    int64_t total_elements = ggml_nelements(cur);
    int64_t stride = n_embed * stack_factor;

    // Calculate padded length
    int64_t padded_len = GGML_PAD(total_elements, stride);
    int64_t pad = padded_len - total_elements;

    if (pad > 0) {
        // Pad the tensor to make it divisible by stride
        cur = ggml_view_1d(ctx0, cur, total_elements, 0);
        cur = ggml_pad(ctx0, cur, pad, 0, 0, 0);
    }

    // Reshape to [stride, padded_len / stride]
    cur = ggml_view_2d(ctx0, cur, stride, padded_len / stride,
                        ggml_row_size(cur->type, stride), 0);
    return cur;
}

// aka pixel_shuffle / pixel_unshuffle / patch_merger (Kimi-VL)
// support dynamic resolution
ggml_tensor * mtmd_graph::build_suffix(ggml_tensor * cur) {
    for (int idx : mtmd_suffix_rows(img.suffix_type)) {
        GGML_ASSERT(model.tok_embd_sep && idx < model.tok_embd_sep->ne[1]);
        ggml_tensor * row = ggml_view_2d(ctx0, model.tok_embd_sep, model.tok_embd_sep->ne[0], 1,
                                         model.tok_embd_sep->nb[1], idx * model.tok_embd_sep->nb[1]);
        cur = ggml_concat(ctx0, cur, ggml_cast(ctx0, row, cur->type), 1);
    }
    return cur;
}

ggml_tensor * mtmd_graph::build_patch_merge_permute(ggml_tensor * cur, int scale_factor) {
    GGML_ASSERT(scale_factor > 1);

    const int n_embd = cur->ne[0];
    int width  = img.nx() / patch_size;
    int height = img.ny() / patch_size;

    // pad width and height to factor
    const int64_t pad_width  = MTMD_ALIGN(width,  scale_factor) - width;
    const int64_t pad_height = MTMD_ALIGN(height, scale_factor) - height;
    cur = ggml_reshape_3d(ctx0, cur, n_embd, width, height);
    if (pad_width || pad_height) {
        cur     = ggml_pad(ctx0, cur, 0, pad_width, pad_height, 0);
        width  += pad_width;
        height += pad_height;
    }

    // unshuffle h
    cur = ggml_reshape_3d(ctx0, cur, n_embd * scale_factor, width / scale_factor, height);
    cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);

    // unshuffle w
    cur = ggml_cont_3d(ctx0, cur, n_embd * scale_factor * scale_factor, height / scale_factor, width / scale_factor);
    cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);

    cur = ggml_cont_2d(ctx0, cur, cur->ne[0], cur->ne[1] * cur->ne[2]);
    cb(cur, "pixel_shuffle", -1);

    return cur;
}

std::unique_ptr<mtmd_graph> mtmd_get_graph_builder(mtmd_subcontext * ctx, const mtmd_image_f32_batch & imgs,
                                                   const mtmd_subctx_encode_params * params) {
    const mtmd_image_f32 & img = imgs.entries[0];
    std::unique_ptr<mtmd_graph> builder;

    switch (ctx->model.proj_type) {
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_IDEFICS3:
        case PROJECTOR_TYPE_COHERE2V:
        case PROJECTOR_TYPE_LFM2:
        case PROJECTOR_TYPE_D1OMNI_V:
        case PROJECTOR_TYPE_JANUS_PRO:
        case PROJECTOR_TYPE_PHI4:
            {
                builder = std::make_unique<mtmd_graph_siglip>(ctx, img);
            } break;
        case PROJECTOR_TYPE_GEMMA3NV:
            {
                builder = std::make_unique<mtmd_graph_mobilenetv5>(ctx, img);
            } break;
        case PROJECTOR_TYPE_GEMMA4V:
            {
                builder = std::make_unique<mtmd_graph_gemma4v>(ctx, img);
            } break;
        case PROJECTOR_TYPE_GEMMA4UV:
            {
                builder = std::make_unique<mtmd_graph_gemma4uv>(ctx, img);
            } break;
        case PROJECTOR_TYPE_PIXTRAL:
        case PROJECTOR_TYPE_LIGHTONOCR:
            {
                builder = std::make_unique<mtmd_graph_pixtral>(ctx, img);
            } break;
        case PROJECTOR_TYPE_DOTS_OCR:
        case PROJECTOR_TYPE_DOTS3NOTE_V: // same ViT + merger; pyramid MoE is handled by build_vit
            {
                builder = std::make_unique<mtmd_graph_dotsocr>(ctx, img);
            } break;
        case PROJECTOR_TYPE_DOTS3NOTE_A:
            {
                builder = std::make_unique<mtmd_graph_dots3note_a>(ctx, img);
            } break;
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
            {
                builder = std::make_unique<mtmd_graph_qwen2vl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_QWEN3VL:
            {
                builder = std::make_unique<mtmd_graph_qwen3vl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_MAGEVL:
            {
                builder = std::make_unique<mtmd_graph_magevl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_SENSENOVA_U1:
            {
                builder = std::make_unique<mtmd_graph_sensenova_u1>(ctx, img);
            } break;
        case PROJECTOR_TYPE_LFM3VL:
            {
                builder = std::make_unique<mtmd_graph_lfm3vl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_LING3VL:
            {
                builder = std::make_unique<mtmd_graph_ling3vl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_EXAONE4_5:
            {
                builder = std::make_unique<mtmd_graph_exaone4_5>(ctx, img);
            } break;
        case PROJECTOR_TYPE_MIMOVL:
            {
                builder = std::make_unique<mtmd_graph_mimovl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_MINIMAX_M3:
            {
                builder = std::make_unique<mtmd_graph_minimax_m3>(ctx, img);
            } break;
        case PROJECTOR_TYPE_MUSE_GLIMMER:
            {
                builder = std::make_unique<mtmd_graph_muse_glimmer>(ctx, img);
            } break;
        case PROJECTOR_TYPE_STEP3VL:
            {
                builder = std::make_unique<mtmd_graph_step3vl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_MINICPMV:
            {
                builder = std::make_unique<mtmd_graph_minicpmv>(ctx, img);
            } break;
        case PROJECTOR_TYPE_MINICPMV4_6:
        case PROJECTOR_TYPE_MINICPMV4_7:
        case PROJECTOR_TYPE_MINICPM_OCR_1_0:
            {
                builder = std::make_unique<mtmd_graph_minicpmv4_6>(ctx, img);
            } break;
        case PROJECTOR_TYPE_INTERNVL:
            {
                builder = std::make_unique<mtmd_graph_internvl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_NEMOTRON_V2_VL:
            {
                builder = std::make_unique<mtmd_graph_nemotron_v2_vl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_NEMOTRON_OMNI:
            {
                builder = std::make_unique<mtmd_graph_nemotron_omni>(ctx, img);
            } break;
        case PROJECTOR_TYPE_LLAMA4:
            {
                builder = std::make_unique<mtmd_graph_llama4>(ctx, img);
            } break;
        case PROJECTOR_TYPE_VOXTRAL_RT:
            {
                builder = std::make_unique<mtmd_graph_voxtral_rt>(ctx, img, params ? params->enc_state : nullptr);
            } break;
        case PROJECTOR_TYPE_ULTRAVOX:
        case PROJECTOR_TYPE_VOXTRAL:
        case PROJECTOR_TYPE_QWEN2A:
        case PROJECTOR_TYPE_GLMA:
        case PROJECTOR_TYPE_MERALION:
        case PROJECTOR_TYPE_MUSIC_FLAMINGO:
            {
                builder = std::make_unique<mtmd_graph_whisper_enc>(ctx, img);
            } break;
        case PROJECTOR_TYPE_MUSE_SPARK_AUDIO:
            {
                builder = std::make_unique<mtmd_graph_muse_spark_audio>(ctx, img);
            } break;
        case PROJECTOR_TYPE_KIMIVL:
            {
                builder = std::make_unique<mtmd_graph_kimivl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_PADDLEOCR:
            {
                builder = std::make_unique<mtmd_graph_paddleocr>(ctx, img);
            } break;
        case PROJECTOR_TYPE_KIMIK25:
            {
                builder = std::make_unique<mtmd_graph_kimik25>(ctx, img);
            } break;
        case PROJECTOR_TYPE_DEEPSEEK4V:
        case PROJECTOR_TYPE_DEEPSEEK41V:
            {
                builder = std::make_unique<mtmd_graph_deepseek4v>(ctx, img);
            } break;
        case PROJECTOR_TYPE_COGVLM:
            {
                builder = std::make_unique<mtmd_graph_cogvlm>(ctx, img);
            } break;
        case PROJECTOR_TYPE_HUNYUANVL:
            {
                builder = std::make_unique<mtmd_graph_hunyuanvl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_MLP:
        case PROJECTOR_TYPE_MLP_NORM:
        case PROJECTOR_TYPE_LDP:
        case PROJECTOR_TYPE_LDPV2:
        case PROJECTOR_TYPE_GLM_EDGE:
            {
                builder = std::make_unique<mtmd_graph_llava>(ctx, img);
            } break;
        case PROJECTOR_TYPE_DEEPSEEKOCR:
            {
                builder = std::make_unique<mtmd_graph_deepseekocr>(ctx, img);
            } break;
        case PROJECTOR_TYPE_DEEPSEEKOCR2:
             {
                builder = std::make_unique<mtmd_graph_deepseekocr2>(ctx, img);
            } break;
        case PROJECTOR_TYPE_LFM2A:
        case PROJECTOR_TYPE_D1OMNI_A:
            {
                builder = std::make_unique<mtmd_graph_conformer>(ctx, img);
            } break;
        case PROJECTOR_TYPE_LFM3A:
            {
                builder = std::make_unique<mtmd_graph_lfm3a>(ctx, img);
            } break;
        case PROJECTOR_TYPE_GEMMA4A:
            {
                builder = std::make_unique<mtmd_graph_gemma4a>(ctx, img);
            } break;
        case PROJECTOR_TYPE_GEMMA4UA:
            {
                builder = std::make_unique<mtmd_graph_gemma4ua>(ctx, img);
            } break;
        case PROJECTOR_TYPE_GRANITE_SPEECH:
            {
                builder = std::make_unique<mtmd_graph_granite_speech>(ctx, img);
            } break;
        case PROJECTOR_TYPE_GLM4V:
        case PROJECTOR_TYPE_GLM5V:
            {
                builder = std::make_unique<mtmd_graph_glm4v>(ctx, img);
            } break;
        case PROJECTOR_TYPE_QWEN3A:
            {
                builder = std::make_unique<mtmd_graph_qwen3a>(ctx, img);
            } break;
        case PROJECTOR_TYPE_MIMO_AUDIO:
            {
                builder = std::make_unique<mtmd_graph_mimo_audio>(ctx, img);
            } break;
        case PROJECTOR_TYPE_QWEN3TTS_SPKENC:
            {
                builder = std::make_unique<mtmd_graph_qwen3tts_spkenc>(ctx, img);
            } break;
        case PROJECTOR_TYPE_POCKETTTS_SPKENC:
            {
                builder = std::make_unique<mtmd_graph_pockettts_spkenc>(ctx, img);
            } break;
        case PROJECTOR_TYPE_POCKETTTS_GEN:
            {
                const auto gen_process = params ? params->gen_process : MTMD_SUBCTX_GEN_PROCESS_GEN_CODE;
                const int  n_step = ctx->model.hparams.flow_n_step;
                const int64_t n_latent = ctx->model.gen_input_lin_w->ne[0];
                GGML_ASSERT(n_step > 0);
                GGML_ASSERT(n_latent > 0);
                // "inp_feats" takes the caller's buffer as-is, the graph must consume all of it
                if (params && params->feats) {
                    GGML_ASSERT(params->feats->size() % (size_t) n_latent == 0);
                    GGML_ASSERT(params->feats->size() >= (size_t) n_latent);
                }
                const int  n_frames = params && params->feats ? (int) (params->feats->size() / n_latent) : 1;
                builder = std::make_unique<mtmd_graph_pockettts_gen>(ctx, img, gen_process, n_step, n_frames);
            } break;
        case PROJECTOR_TYPE_CHATTERBOX_SPKENC:
            {
                builder = std::make_unique<mtmd_graph_chatterbox_spkenc>(ctx, img);
            } break;
        case PROJECTOR_TYPE_CHATTERBOX:
            {
                const auto gen_process = params ? params->gen_process : MTMD_SUBCTX_GEN_PROCESS_GEN_CODE;
                int n_tokens = params && params->codes ? (int) params->codes->size() : 0;
                int n_prompt_mel = 0;
                if (n_tokens > 0) {
                    GGML_ASSERT(params->ref);
                    n_tokens += (int) params->ref->tokens.size();
                    n_prompt_mel = (int) (params->ref->feat.size() / 80);
                }
                const int vnm = params ? params->vocode_n_mel  : 0;
                const int vns = params ? params->vocode_n_stft : 0;
                builder = std::make_unique<mtmd_graph_chatterbox>(ctx, img, gen_process, n_tokens, n_prompt_mel, vnm, vns);
            } break;
        case PROJECTOR_TYPE_LFM3A_GEN:
            {
                const auto    gen_process = params ? params->gen_process : MTMD_SUBCTX_GEN_PROCESS_GEN_CODE;
                const int64_t n_cb        = ctx->model.gen_code_embd_w->ne[2];
                bool all_known = false;
                int  n_frames  = 1;
                if (params && params->codes) {
                    GGML_ASSERT(params->codes->size() % (size_t) n_cb == 0);
                    if (gen_process == MTMD_SUBCTX_GEN_PROCESS_GEN_WAV) {
                        n_frames = (int) (params->codes->size() / n_cb);
                        GGML_ASSERT(n_frames > 0);
                    } else {
                        // -1 marks a masked codebook, none masked means only the feedback embedding is needed
                        all_known = std::all_of(params->codes->begin(), params->codes->end(), [](int32_t c) { return c >= 0; });
                    }
                }
                builder = std::make_unique<mtmd_graph_lfm3a_gen>(ctx, img, gen_process, all_known, n_frames);
            } break;
        case PROJECTOR_TYPE_DRAMA3_GEN:
            {
                const auto gen_process = params ? params->gen_process : MTMD_SUBCTX_GEN_PROCESS_GEN_CODE;
                const int64_t n_cb = ctx->model.gen_code_embd_w->ne[2];
                const int n_frames = params && params->codes ? (int) (params->codes->size() / n_cb) : 1;
                builder = std::make_unique<mtmd_graph_drama3_gen>(ctx, img, gen_process, params ? params->gen_state : nullptr, n_frames,
                                                                  params ? params->top_k : 30, params ? params->top_p : 0.9f,
                                                                  params ? params->temp : 1.0f);
            } break;
        case PROJECTOR_TYPE_QWEN3TTS_GEN:
            {
                const auto  gen_process = params ? params->gen_process : MTMD_SUBCTX_GEN_PROCESS_GEN_CODE;
                const int   top_k = params ? params->top_k : 50;
                const float top_p = params ? params->top_p : 1.0f;
                builder = std::make_unique<mtmd_graph_qwen3tts_gen>(ctx, img, gen_process, top_k, top_p);
            } break;
        case PROJECTOR_TYPE_SOPRANO:
            {
                const int n_frames = params && params->feats ? (int) (params->feats->size() / ctx->model.hparams.projection_dim) : 2;
                builder = std::make_unique<mtmd_graph_soprano>(ctx, img, n_frames);
            } break;
        case PROJECTOR_TYPE_YOUTUVL:
            {
                builder = std::make_unique<mtmd_graph_youtuvl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_YASA2:
            {
                builder = std::make_unique<mtmd_graph_yasa2>(ctx, img);
            } break;
        case PROJECTOR_TYPE_PARAKEET:
            {
                builder = std::make_unique<mtmd_graph_parakeet>(ctx, img);
            } break;
        case PROJECTOR_TYPE_NEMOTRON_TRANSCRIBE:
            {
                builder = std::make_unique<mtmd_graph_nemotron_transcribe>(ctx, img, params ? params->enc_state : nullptr);
            } break;
        case PROJECTOR_TYPE_GRANITE4_VISION:
            {
                builder = std::make_unique<mtmd_graph_granite4_vision>(ctx, img);
            } break;
        default:
            GGML_ABORT("missing cgraph builder");
    }

    builder->img_batch = &imgs;

    // TODO [QWEN_VIDEO]: improve this in the future
    builder->n_batch = imgs.entries.size();

    return builder;
}
