#include "models.h"

ggml_cgraph * clip_graph_nemotron_v2_vl::build() {
    GGML_ASSERT(model.class_embedding != nullptr);
    GGML_ASSERT(model.position_embeddings != nullptr);

    const int n_registers = model.class_embedding->ne[1];
    const int n_pos = n_patches + n_registers;

    ggml_tensor * inp_raw = build_inp_raw();
    ggml_tensor * inp;
    {
        ggml_tensor * kernel = ggml_reshape_4d(ctx0, model.patch_embeddings_0,
                                                patch_size, patch_size, 3, n_embd);
        inp = ggml_im2col(ctx0, kernel, inp_raw, patch_size, patch_size, 0, 0, 1, 1, true, inp_raw->type);
        inp = ggml_mul_mat(ctx0, model.patch_embeddings_0, inp);
        inp = ggml_reshape_2d(ctx0, inp, n_embd, n_patches);
        cb(inp, "patch_embd", -1);
    }

    {
        const int max_patches_per_side = (int)std::sqrt((float)model.position_embeddings->ne[1]);
        
        ggml_tensor * pos_embd = ggml_reshape_3d(ctx0, model.position_embeddings, n_embd, max_patches_per_side, max_patches_per_side);
        
        const int pool_h = max_patches_per_side / n_patches_y;
        const int pool_w = max_patches_per_side / n_patches_x;
        
        if (pool_h > 1 || pool_w > 1) {
            pos_embd = ggml_cont(ctx0, ggml_permute(ctx0, pos_embd, 1, 2, 0, 3));
            pos_embd = ggml_pool_2d(ctx0, pos_embd, GGML_OP_POOL_AVG, pool_w, pool_h, pool_w, pool_h, 0, 0);
            pos_embd = ggml_cont(ctx0, ggml_permute(ctx0, pos_embd, 2, 0, 1, 3));
        }
        
        pos_embd = ggml_reshape_2d(ctx0, pos_embd, n_embd, n_patches);
        
        inp = ggml_add(ctx0, inp, pos_embd);
        cb(inp, "inp_pos", -1);
    }

    inp = ggml_concat(ctx0, model.class_embedding, inp, 1);

    ggml_tensor * cur = build_vit(inp, n_pos, NORM_TYPE_NORMAL, hparams.ffn_op, nullptr, nullptr);

    cur = ggml_view_2d(ctx0, cur,
        n_embd, n_patches,
        ggml_row_size(cur->type, n_embd),
        n_registers * ggml_row_size(cur->type, n_embd));

    {
        const int scale_factor = model.hparams.n_merge;
        const int bsz    = 1;
        const int height = n_patches_y;
        const int width  = n_patches_x;
        GGML_ASSERT(scale_factor > 0);
        cur = ggml_reshape_4d(ctx0, cur, n_embd * scale_factor, height / scale_factor, width, bsz);
        cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
        cur = ggml_cont_4d(ctx0, cur,
            n_embd * scale_factor * scale_factor,
            height / scale_factor,
            width / scale_factor,
            bsz);
        cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, cur,
            n_embd * scale_factor * scale_factor,
            cur->ne[1] * cur->ne[2]);
    }

    {
        cur = build_norm(cur, model.mm_0_w, nullptr, NORM_TYPE_RMS, 1e-6, -1);
        cur = build_ffn(cur, model.mm_1_w, nullptr, nullptr, nullptr, model.mm_3_w, nullptr, FFN_RELU_SQR, -1);
    }

    ggml_build_forward_expand(gf, cur);

    return gf;
}

