#include "models.h"

ggml_cgraph * clip_graph_cradio::build() {
    ggml_tensor * inp_raw = build_inp_raw();
        ggml_tensor * cur     = inp_raw;

        {
            const auto py         = inp_raw->ne[0] / hparams.patch_size;
            const auto px         = inp_raw->ne[1] / hparams.patch_size;
            const auto n_channels = inp_raw->ne[2];
            const auto batch_size = inp_raw->ne[3];
            
            cur = ggml_reshape_4d(ctx0, cur, hparams.patch_size, px, hparams.patch_size, py * n_channels * batch_size);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont(ctx0, cur);
            cur = ggml_reshape_4d(ctx0, cur, hparams.patch_size * hparams.patch_size, px * py, n_channels, batch_size);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont(ctx0, cur);
            cur = ggml_reshape_3d(ctx0, cur, n_channels * hparams.patch_size * hparams.patch_size, px * py, batch_size);
        }

        cur                     = ggml_mul_mat(ctx0, model.patch_embeddings_0, cur);
        ggml_tensor * pos_embed = ggml_reshape_4d(ctx0, model.position_embeddings, model.position_embeddings->ne[0], hparams.max_resolution/hparams.patch_size, hparams.max_resolution/hparams.patch_size, 1);
        pos_embed               = ggml_permute(ctx0, pos_embed, 2, 0, 1, 3);
        pos_embed               = ggml_cont(ctx0, pos_embed);

        int dim = hparams.image_size/hparams.patch_size;
        pos_embed               = ggml_interpolate(ctx0, pos_embed, dim, dim, pos_embed->ne[2], pos_embed->ne[3], GGML_SCALE_MODE_BILINEAR);
        pos_embed               = ggml_reshape_3d(ctx0, pos_embed, dim*dim, pos_embed->ne[2], 1);
        pos_embed               = ggml_permute(ctx0, pos_embed, 1, 0, 2, 3);
        pos_embed               = ggml_cont(ctx0, pos_embed);
        cur                     = ggml_add_inplace(ctx0, cur, pos_embed);


        ggml_tensor * cls_token = model.class_embedding;
        {
            ggml_tensor * shape_tensor = ggml_new_tensor_3d(ctx0, cls_token->type, cls_token->ne[0], cls_token->ne[1], cur->ne[2]);
            cls_token                  = ggml_repeat(ctx0, cls_token, shape_tensor);
        }

        cur = ggml_concat(ctx0, cls_token, cur, 1);
        
        const int ftype = (model.layers[0].qkv_w->type == GGML_TYPE_F16) ? 1 : 0;
        enum ggml_type type     = ftype ? GGML_TYPE_F16 : GGML_TYPE_F32;
        ggml_tensor * intermediate_features_shape = ggml_new_tensor_4d(ctx0, type, cur->ne[0], cur->ne[1] - 16, cur->ne[2], cur->ne[3]);
        ggml_tensor * intermediate_features = ggml_repeat(ctx0, ggml_arange(ctx0, 0.0f, 1.0f, 1.0f), intermediate_features_shape);
        ggml_tensor * ifdiv = ggml_arange(ctx0, 1.0f / 16.0f, 1.0f / 16.0f + 1.0f, 1.0f);
        ggml_tensor * final_features        = nullptr;

        ggml_tensor * x = nullptr;

        for (int i = 0; i < n_layer; i++) {
            // BLock forward 
            {
                if (ftype) cur      = ggml_cast(ctx0, cur, GGML_TYPE_F32);
                x                   = build_norm(cur, model.layers[i].ln_1_w, model.layers[i].ln_1_b, NORM_TYPE_NORMAL, 1e-6, i);
                if (ftype) {
                    x               = ggml_cast(ctx0, x, GGML_TYPE_F16);
                    cur             = ggml_cast(ctx0, cur, GGML_TYPE_F16);
                }

                // Attention
                {
                    ggml_tensor * qkv     = ggml_mul_mat(ctx0, model.layers[i].qkv_w, x);
                    if (ftype) qkv        = ggml_cast(ctx0, qkv, GGML_TYPE_F16);
                    qkv                   = ggml_add_inplace(ctx0, qkv, model.layers[i].qkv_b);
                    
                    const int qkv_dim     = hparams.n_embd;
                    const int head_dim    = qkv_dim / n_head;
                    const int num_patches = x->ne[1];
                    const int batch_size  = x->ne[2];

                    qkv                   = ggml_reshape_4d(ctx0, qkv, qkv_dim, 3, num_patches, batch_size);
                    qkv                   = ggml_permute(ctx0, qkv, 1, 3, 0, 2);
                    qkv                   = ggml_cont(ctx0, qkv);

                    ggml_tensor * q       = ggml_view_4d(ctx0, qkv, num_patches, qkv_dim, batch_size, 1, qkv->nb[1], qkv->nb[2], qkv->nb[3], 0);
                    q                     = ggml_reshape_4d(ctx0, q, num_patches, head_dim, n_head, batch_size);
                    q                     = ggml_permute(ctx0, q, 1, 0, 2, 3);
                    q                     = ggml_cont(ctx0, q);

                    ggml_tensor * k       = ggml_view_4d(ctx0, qkv, num_patches, qkv_dim, batch_size, 1, qkv->nb[1], qkv->nb[2], qkv->nb[3], qkv->nb[3]);
                    k                     = ggml_reshape_4d(ctx0, k, num_patches, head_dim, n_head, batch_size);
                    k                     = ggml_permute(ctx0, k, 1, 0, 2, 3);
                    k                     = ggml_cont(ctx0, k);

                    ggml_tensor * v       = ggml_view_4d(ctx0, qkv, num_patches, qkv_dim, batch_size, 1, qkv->nb[1], qkv->nb[2], qkv->nb[3], 2 * qkv->nb[3]);
                    v                     = ggml_reshape_4d(ctx0, v, num_patches, head_dim, n_head, batch_size);
                    
                    if (ftype) {
                        q                 = ggml_cast(ctx0, q, GGML_TYPE_F16);
                        k                 = ggml_cast(ctx0, k, GGML_TYPE_F16);
                        v                 = ggml_cast(ctx0, v, GGML_TYPE_F16);
                    }

                    qkv                   = ggml_mul_mat(ctx0, k, q);
                    qkv                   = ggml_soft_max_ext(ctx0, qkv, NULL, 1.0f / sqrt(k->ne[0]), 0.0f);
                    qkv                   = ggml_mul_mat(ctx0, v, qkv);

                    if (ftype) qkv        = ggml_cast(ctx0, qkv, GGML_TYPE_F16);
                    qkv                   = ggml_permute(ctx0, qkv, 0, 2, 1, 3);
                    qkv                   = ggml_cont(ctx0, qkv);
                    qkv                   = ggml_reshape_4d(ctx0, qkv, qkv_dim, num_patches, batch_size, 1);
                    qkv                   = ggml_mul_mat(ctx0, model.layers[i].o_w, qkv);
                    if (ftype) qkv        = ggml_cast(ctx0, qkv, GGML_TYPE_F16);
                    qkv                   = ggml_add_inplace(ctx0, qkv, model.layers[i].o_b);

                    cur                   = ggml_add_inplace(ctx0, cur, qkv);
                }

                if (ftype) cur      = ggml_cast(ctx0, cur, GGML_TYPE_F32);
                x                   = build_norm(cur, model.layers[i].ln_2_w, model.layers[i].ln_2_b, NORM_TYPE_NORMAL, 1e-6, i);
                if (ftype) {
                    x               = ggml_cast(ctx0, x, GGML_TYPE_F16);
                    cur             = ggml_cast(ctx0, cur, GGML_TYPE_F16);
                }

                // mlp
                {
                    x               = ggml_mul_mat(ctx0, model.layers[i].ff_down_w, x);
                    if (ftype) x    = ggml_cast(ctx0, x, GGML_TYPE_F16);
                    x               = ggml_add_inplace(ctx0, x, model.layers[i].ff_down_b);
                    x               = ggml_gelu_inplace(ctx0, x);

                    x               = ggml_mul_mat(ctx0, model.layers[i].ff_up_w, x);
                    if (ftype) x    = ggml_cast(ctx0, x, GGML_TYPE_F16);
                    x               = ggml_add_inplace(ctx0, x, model.layers[i].ff_up_b);
                }

                cur                 = ggml_add_inplace(ctx0, cur, x);
            }

        }

        {
            x   = ggml_dup(ctx0, cur);
            x   = ggml_permute(ctx0, x, 0, 2, 1, 3);
            x   = ggml_cont(ctx0, x);
            x   = ggml_view_3d(ctx0, x, x->ne[0], x->ne[1], x->ne[2] - hparams.num_skip, x->nb[1], x->nb[2], hparams.num_skip * x->nb[2]);
            x   = ggml_permute(ctx0, x, 0, 2, 1, 3);
            cur = ggml_cont(ctx0, x);
        }

        int h = sqrt(cur->ne[1]);
        int w = h;
        // pixel shuffle
        {
            cur = ggml_reshape_4d(ctx0, cur, cur->ne[0] * 2, h/2, w, cur->ne[2]);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont(ctx0, cur);
            cur = ggml_reshape_4d(ctx0, cur, cur->ne[0] * 2, w/2, h/2, cur->ne[3]);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont(ctx0, cur);
        }

        cur            = ggml_reshape_4d(ctx0, cur, cur->ne[0], (h*w)/4, cur->ne[3], 1);
        if (ftype) cur = ggml_cast(ctx0, cur, GGML_TYPE_F32);
        cur            = build_norm(cur, model.mm_0_w, model.mm_0_b, NORM_TYPE_NORMAL, 1e-6, 0);
        cur            = ggml_mul_mat(ctx0, model.mm_1_w, cur);
        cur            = ggml_add_inplace(ctx0, cur, model.mm_1_b);
        cur            = ggml_gelu_inplace(ctx0, cur);
        cur            = ggml_mul_mat(ctx0, model.mm_3_w, cur);
        cur            = ggml_add_inplace(ctx0, cur, model.mm_3_b);

        ggml_build_forward_expand(gf, cur);

        return gf;
}
