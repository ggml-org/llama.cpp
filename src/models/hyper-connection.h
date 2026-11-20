#pragma once

// manifold-constrained hyper-connections (mHC)
//
// shared by the graphs that use the dsv4 HC tensors and the ggml_dsv4_hc_* fused ops.
// the graph type T supplies ctx0, hparams, cparams, res, n_embd, norm_rms_eps, cb and the
// hc_pre / hc_mixes result structs

template <class T>
struct hc_graph {
    // all outputs of the hc mix builders: cur is the collapsed input, post / comb feed build_hc_post
    struct hc_pre {
        ggml_tensor * cur;
        ggml_tensor * post;
        ggml_tensor * comb;
    };

    struct hc_mixes {
        ggml_tensor * pre;
        ggml_tensor * post;
        ggml_tensor * comb;
    };

    const T & self() const { return *static_cast<const T *>(this); }

    static ggml_tensor * hc_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
        return ggml_view_1d(ctx, t, ne0, ggml_row_size(t->type, i0));
    }

    static ggml_tensor * hc_view_2d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t i0) {
        return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], ggml_row_size(t->type, i0));
    }

    static ggml_tensor * hc_affine(ggml_context * ctx, ggml_tensor * x, ggml_tensor * scale, ggml_tensor * base) {
        x = ggml_mul(ctx, x, scale);
        x = ggml_add(ctx, x, base);
        return x;
    }

    // mean over the hyper-connection streams: [n_embd, hc, n_tokens] -> [n_embd, n_tokens]
    ggml_tensor * build_hc_mean(ggml_tensor * x) const {
        ggml_context * ctx = self().ctx0;
        const int64_t hc = x->ne[1];

        ggml_tensor * acc = ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], 0);
        for (int64_t s = 1; s < hc; ++s) {
            acc = ggml_add(ctx, acc, ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], s*x->nb[1]));
        }
        return ggml_scale(ctx, acc, 1.0f/hc);
    }

    // collapse the hc streams with per-stream weights
    ggml_tensor * build_hc_pre(
            ggml_tensor * x,
            ggml_tensor * weights,
            int           il) const {
        ggml_context * ctx = self().ctx0;
        const auto & hparams = self().hparams;
        const auto & cparams = self().cparams;

        const int64_t hc = hparams.dsv4_hc_mult;
        const int64_t nt = x->ne[2];

        if (cparams.fused_dsv4_hc_pre && il >= 0) {
            ggml_tensor * result = ggml_dsv4_hc_pre(ctx, x, weights);
            self().res->add_fused_node({LLM_FUSED_OP_DSV4_HC_PRE, result, il});
            return result;
        }

        ggml_tensor * result = nullptr;
        for (int64_t ih = 0; ih < hc; ++ih) {
            ggml_tensor * xh = ggml_view_2d(ctx, x, self().n_embd, nt, x->nb[2], ih*x->nb[1]);
            ggml_tensor * wh = ggml_view_2d(ctx, weights, 1, nt, weights->nb[1], ih*weights->nb[0]);
            ggml_tensor * cur = ggml_mul(ctx, xh, wh);
            result = result ? ggml_add(ctx, result, cur) : cur;
        }

        return result;
    }

    ggml_tensor * build_hc_sinkhorn(ggml_tensor * comb, int il) const {
        ggml_context * ctx = self().ctx0;
        const auto & hparams = self().hparams;
        GGML_UNUSED(il);

        // row softmax over dst, one column normalization, then repeated row/column normalization
        comb = ggml_soft_max(ctx, comb);

        ggml_tensor * eps = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        eps = ggml_fill(ctx, eps, hparams.dsv4_hc_eps);
        comb = ggml_add(ctx, comb, eps);

        auto norm_cols = [&]() {
            ggml_tensor * comb_src_dst = ggml_cont(ctx, ggml_permute(ctx, comb, 1, 0, 2, 3));
            ggml_tensor * col_sum = ggml_sum_rows(ctx, comb_src_dst);
            col_sum = ggml_add(ctx, col_sum, eps);
            col_sum = ggml_permute(ctx, col_sum, 1, 0, 2, 3);
            comb = ggml_div(ctx, comb, col_sum);
        };
        auto norm_rows = [&]() {
            ggml_tensor * row_sum = ggml_sum_rows(ctx, comb);
            row_sum = ggml_add(ctx, row_sum, eps);
            comb = ggml_div(ctx, comb, row_sum);
        };

        norm_cols();
        for (uint32_t i = 1; i < hparams.dsv4_hc_sinkhorn_iters; ++i) {
            norm_rows();
            norm_cols();
        }

        return comb;
    }

    hc_mixes build_hc_mixes(
            ggml_tensor * x,
            ggml_tensor * hc_fn,
            ggml_tensor * hc_scale,
            ggml_tensor * hc_base,
            int il) const {
        ggml_context * ctx = self().ctx0;
        const auto & hparams = self().hparams;
        const auto & cparams = self().cparams;
        const auto & self_    = self();

        const int64_t hc         = hparams.dsv4_hc_mult;
        const int64_t hc_dim     = hc*self_.n_embd;
        const int64_t hc_mix_dim = (2 + hc)*hc;
        const int64_t nt         = x->ne[2];

        GGML_ASSERT(hc_fn->ne[1] == hc_mix_dim);

        ggml_tensor * flat      = ggml_reshape_2d(ctx, x, hc_dim, nt);
        ggml_tensor * flat_norm = ggml_rms_norm(ctx, flat, self_.norm_rms_eps);
        ggml_tensor * mixes     = ggml_mul_mat(ctx, hc_fn, flat_norm);
        self_.cb(mixes, "hc_mixes", il);

        ggml_tensor * scale_pre  = hc_view_1d(ctx, hc_scale, 1, 0);
        ggml_tensor * scale_post = hc_view_1d(ctx, hc_scale, 1, 1);
        ggml_tensor * scale_comb = hc_view_1d(ctx, hc_scale, 1, 2);

        ggml_tensor * base_pre  = hc_view_1d(ctx, hc_base, hc, 0);
        ggml_tensor * base_post = hc_view_1d(ctx, hc_base, hc, hc);
        ggml_tensor * base_comb = hc_view_1d(ctx, hc_base, hc*hc, 2*hc);

        ggml_tensor * pre = hc_view_2d(ctx, mixes, hc, nt, 0);
        pre = hc_affine(ctx, pre, scale_pre, base_pre);
        pre = ggml_sigmoid(ctx, pre);
        pre = ggml_scale_bias(ctx, pre, 1.0f, hparams.dsv4_hc_eps);
        self_.cb(pre, "hc_pre", il);

        ggml_tensor * post = hc_view_2d(ctx, mixes, hc, nt, hc);
        post = hc_affine(ctx, post, scale_post, base_post);
        post = ggml_sigmoid(ctx, post);
        post = ggml_scale(ctx, post, 2.0f);
        self_.cb(post, "hc_post", il);

        ggml_tensor * comb = nullptr;
        if (cparams.fused_dsv4_hc_comb) {
            comb = ggml_dsv4_hc_comb(ctx, mixes, hc_scale, hc_base, hparams.dsv4_hc_eps,
                    (int32_t) hparams.dsv4_hc_sinkhorn_iters);
            self_.res->add_fused_node({LLM_FUSED_OP_DSV4_HC_COMB, comb, il});
        } else {
            comb = hc_view_2d(ctx, mixes, hc*hc, nt, 2*hc);
            comb = hc_affine(ctx, comb, scale_comb, base_comb);
            comb = ggml_reshape_3d(ctx, comb, hc, hc, nt);
            comb = build_hc_sinkhorn(comb, il);
        }
        self_.cb(comb, "hc_comb", il);

        return { pre, post, comb };
    }

    ggml_tensor * build_hc_post(
            ggml_tensor * x,
            ggml_tensor * residual,
            ggml_tensor * post,
            ggml_tensor * comb,
            int il) const {
        ggml_context * ctx = self().ctx0;
        const auto & hparams = self().hparams;
        const auto & cparams = self().cparams;

        const int64_t hc = hparams.dsv4_hc_mult;
        const int64_t nt = x->ne[1];

        if (cparams.fused_dsv4_hc_post) {
            ggml_tensor * result = ggml_dsv4_hc_post(ctx, x, residual, post, comb);
            self().res->add_fused_node({LLM_FUSED_OP_DSV4_HC_POST, result, il});
            return result;
        }

        ggml_tensor * out = nullptr;
        for (int64_t dst = 0; dst < hc; ++dst) {
            ggml_tensor * post_dst = ggml_view_2d(ctx, post, 1, nt, post->nb[1], dst*post->nb[0]);
            ggml_tensor * cur = ggml_mul(ctx, x, post_dst);

            for (int64_t src = 0; src < hc; ++src) {
                ggml_tensor * res_src = ggml_view_2d(ctx, residual, self().n_embd, nt, residual->nb[2], src*residual->nb[1]);
                ggml_tensor * comb_src_dst = ggml_view_2d(ctx, comb, 1, nt, comb->nb[2],
                        dst*comb->nb[0] + src*comb->nb[1]);
                cur = ggml_add(ctx, cur, ggml_mul(ctx, res_src, comb_src_dst));
            }

            cur = ggml_reshape_3d(ctx, cur, self().n_embd, 1, nt);
            out = out ? ggml_concat(ctx, out, cur, 1) : cur;
        }

        GGML_UNUSED(il);
        return out;
    }

    // all outputs of the hc mix builders: cur is the collapsed input, post / comb feed build_hc_post
    hc_pre build_hc_pre(
            ggml_tensor * x,
            ggml_tensor * hc_fn,
            ggml_tensor * hc_scale,
            ggml_tensor * hc_base,
            int il) const {
        hc_mixes mixes = build_hc_mixes(x, hc_fn, hc_scale, hc_base, il);
        return { build_hc_pre(x, mixes.pre, il), mixes.post, mixes.comb };
    }

    ggml_tensor * build_hc_head(
            ggml_tensor * x,
            ggml_tensor * hc_fn,
            ggml_tensor * hc_scale,
            ggml_tensor * hc_base) const {
        ggml_context * ctx = self().ctx0;
        const auto & hparams = self().hparams;
        const auto & self_    = self();

        const int64_t hc     = hparams.dsv4_hc_mult;
        const int64_t hc_dim = hc*self_.n_embd;
        const int64_t nt     = x->ne[2];

        ggml_tensor * flat      = ggml_reshape_2d(ctx, x, hc_dim, nt);
        ggml_tensor * flat_norm = ggml_rms_norm(ctx, flat, self_.norm_rms_eps);
        ggml_tensor * mixes     = ggml_mul_mat(ctx, hc_fn, flat_norm);
        self_.cb(mixes, "hc_head_mixes", -1);

        ggml_tensor * pre = hc_affine(ctx, mixes, hc_scale, hc_base);
        pre = ggml_sigmoid(ctx, pre);
        pre = ggml_scale_bias(ctx, pre, 1.0f, hparams.dsv4_hc_eps);
        self_.cb(pre, "hc_head_pre", -1);

        return build_hc_pre(x, pre, -1);
    }
};
