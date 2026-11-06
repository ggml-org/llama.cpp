#include "models.h"

// Chatterbox speaker encoder: CAMPPlus x-vector on kaldi fbank features, see s3gen/xvector.py

static constexpr float bn_eps = 1e-5f;

// per-channel batchnorm on x [C, T], w/b are null on the affine=False variant
ggml_tensor * clip_graph_chatterbox_spkenc::bn1d(const clip_chatterbox::bn & n, ggml_tensor * x) {
    ggml_tensor * sd = ggml_sqrt(ctx0, ggml_scale_bias(ctx0, n.var, 1.0f, bn_eps));
    if (!n.w) {
        return ggml_div(ctx0, ggml_sub(ctx0, x, n.mean), sd);
    }
    ggml_tensor * a     = ggml_div(ctx0, n.w, sd);
    ggml_tensor * shift = ggml_sub(ctx0, n.b, ggml_mul(ctx0, n.mean, a));
    return ggml_add(ctx0, ggml_mul(ctx0, x, a), shift);
}

// batchnorm on a conv2d activation [W=T, H=F, C, 1], stats on ne2
static ggml_tensor * cbx_bn2d(ggml_context * ctx0, const clip_chatterbox::bn & n, ggml_tensor * x) {
    const int C = (int) x->ne[2];
    ggml_tensor * mean = ggml_reshape_4d(ctx0, n.mean, 1, 1, C, 1);
    ggml_tensor * var  = ggml_reshape_4d(ctx0, n.var,  1, 1, C, 1);
    ggml_tensor * w    = ggml_reshape_4d(ctx0, n.w, 1, 1, C, 1);
    ggml_tensor * b    = ggml_reshape_4d(ctx0, n.b, 1, 1, C, 1);
    ggml_tensor * a     = ggml_div(ctx0, w, ggml_sqrt(ctx0, ggml_scale_bias(ctx0, var, 1.0f, bn_eps)));
    ggml_tensor * shift = ggml_sub(ctx0, b, ggml_mul(ctx0, mean, a));
    return ggml_add(ctx0, ggml_mul(ctx0, x, a), shift);
}

ggml_tensor * clip_graph_chatterbox_spkenc::bn2d_relu(const clip_chatterbox::bn & n, ggml_tensor * x) {
    return ggml_relu(ctx0, cbx_bn2d(ctx0, n, x));
}

// fcm residual 2d block, stride on the frequency axis only
ggml_tensor * clip_graph_chatterbox_spkenc::res2d(const clip_chatterbox::spk_res2d & r, ggml_tensor * x,
                                                  int stride) {
    ggml_tensor * cur = ggml_conv_2d(ctx0, r.conv1_w, x, 1, stride, 1, 1, 1, 1);
    cur = bn2d_relu(r.bn1, cur);
    cur = ggml_conv_2d(ctx0, r.conv2_w, cur, 1, 1, 1, 1, 1, 1);
    // bn2 without the relu, applied before the residual add
    cur = cbx_bn2d(ctx0, r.bn2, cur);
    ggml_tensor * res = x;
    if (r.shortcut_w) {
        res = ggml_conv_2d(ctx0, r.shortcut_w, x, 1, stride, 0, 0, 1, 1);
        res = cbx_bn2d(ctx0, r.shortcut_bn, res);
    }
    return ggml_relu(ctx0, ggml_add(ctx0, cur, res));
}

// cam dense tdnn layer: bottleneck then context-gated conv; x [C_in, T] -> [growth, T]
ggml_tensor * clip_graph_chatterbox_spkenc::cam_layer(const clip_chatterbox::spk_cam_layer & l, ggml_tensor * x,
                                                      int dil, ggml_tensor * segfix) {
    ggml_tensor * h = ggml_relu(ctx0, bn1d(l.nl1_bn, x));
    h = cbx_conv1d(l.linear1_w, nullptr, h, 1, 0, 0);
    h = ggml_relu(ctx0, bn1d(l.nl2_bn, h));

    ggml_tensor * k = l.local_w;
    const int pad = ((int) k->ne[0] - 1) / 2 * dil;
    ggml_tensor * y = cbx_conv1d_dil(k, nullptr, h, pad, dil);

    // context: global mean plus ceil-mode segment means of length 100
    const int T = (int) h->ne[1];
    const int C = (int) h->ne[0];
    const int S = (T + 99) / 100;
    ggml_tensor * ht = ggml_cont(ctx0, ggml_transpose(ctx0, h)); // [T, C]
    ggml_tensor * gmean = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_mean(ctx0, ht))); // [C, 1]
    ggml_tensor * seg;
    {
        ggml_tensor * padded = ht;
        if (S * 100 != T) {
            ggml_tensor * z = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, S * 100 - T, C);
            z = ggml_scale(ctx0, z, 0.0f);
            padded = ggml_concat(ctx0, ht, z, 0);
        }
        ggml_tensor * pooled = ggml_pool_1d(ctx0, padded, GGML_OP_POOL_AVG, 100, 100, 0); // [S, C]
        pooled = ggml_mul(ctx0, pooled, segfix);
        ggml_tensor * exp = ggml_interpolate(ctx0, pooled, S * 100, C, 1, 1, GGML_SCALE_MODE_NEAREST);
        exp = ggml_cont(ctx0, ggml_view_2d(ctx0, exp, T, C, exp->nb[1], 0));
        seg = ggml_cont(ctx0, ggml_transpose(ctx0, exp));        // [C, T]
    }
    ggml_tensor * context = ggml_add(ctx0, seg, gmean);
    context = cbx_conv1d(l.ctx1_w, l.ctx1_b, context, 1, 0, 0);
    context = ggml_relu(ctx0, context);
    context = cbx_conv1d(l.ctx2_w, l.ctx2_b, context, 1, 0, 0);
    ggml_tensor * m = ggml_sigmoid(ctx0, context);
    return ggml_mul(ctx0, y, m);
}

ggml_cgraph * clip_graph_chatterbox_spkenc::build() {
    const auto & c = model.cbx;

    const int T  = img.nx();
    const int T1 = (T - 1) / 2 + 1;
    const int S  = (T1 + 99) / 100;
    ggml_tensor * segfix = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, S);
    ggml_set_name(segfix, "inp_segfix");
    ggml_set_input(segfix);

    // fbank features [T, 80] from the preprocessor
    ggml_tensor * inp = build_inp_raw(1);

    // fcm 2d front: [W=T, H=F=80, C=1] -> [T, 10, 32] -> [320, T]
    ggml_tensor * x = ggml_reshape_4d(ctx0, inp, T, 80, 1, 1);
    x = ggml_conv_2d(ctx0, c.spk_conv1_w, x, 1, 1, 1, 1, 1, 1);
    x = bn2d_relu(c.spk_bn1, x);
    x = res2d(c.spk_layer1_0, x, 2);
    x = res2d(c.spk_layer1_1, x, 1);
    x = res2d(c.spk_layer2_0, x, 2);
    x = res2d(c.spk_layer2_1, x, 1);
    x = ggml_conv_2d(ctx0, c.spk_conv2_w, x, 1, 2, 1, 1, 1, 1);
    x = bn2d_relu(c.spk_bn2, x);
    x = ggml_reshape_2d(ctx0, x, T, 320);
    x = ggml_cont(ctx0, ggml_transpose(ctx0, x));               // [320, T]
    cb(x, "spk_fcm", -1);

    // tdnn k5 stride 2 over time, then the three cam dense blocks
    x = cbx_conv1d(c.spk_tdnn_w, nullptr, x, 2, 2, 2); // [128, T1]
    x = ggml_relu(ctx0, bn1d(c.spk_tdnn_bn, x));
    cb(x, "spk_tdnn", -1);

    static const int block_dil[3] = {1, 2, 2};
    const clip_chatterbox::spk_cam_block * blocks[3] = { &c.spk_block1, &c.spk_block2, &c.spk_block3 };
    for (int bi = 1; bi <= 3; bi++) {
        const auto & blk = *blocks[bi - 1];
        for (const auto & l : blk.layers) {
            ggml_tensor * out = cam_layer(l, x, block_dil[bi - 1], segfix);
            x = ggml_concat(ctx0, x, out, 0);
        }
        x = ggml_relu(ctx0, bn1d(blk.transit_bn, x));
        x = cbx_conv1d(blk.transit_w, nullptr, x, 1, 0, 0);
        cb(x, "spk_block", bi);
    }
    x = ggml_relu(ctx0, bn1d(c.spk_out_bn, x)); // [512, T1]

    // statistics pooling: mean and unbiased std over time -> [1024, 1]
    ggml_tensor * xt = ggml_cont(ctx0, ggml_transpose(ctx0, x));                 // [T1, 512]
    ggml_tensor * mean = ggml_mean(ctx0, xt);                                    // [1, 512]
    ggml_tensor * m2   = ggml_mean(ctx0, ggml_mul(ctx0, xt, xt));
    ggml_tensor * var  = ggml_sub(ctx0, m2, ggml_mul(ctx0, mean, mean));
    var = ggml_scale(ctx0, var, (float) T1 / (float) (T1 - 1));
    ggml_tensor * sd = ggml_sqrt(ctx0, ggml_relu(ctx0, var));
    ggml_tensor * stats = ggml_concat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, mean)),
                                            ggml_cont(ctx0, ggml_transpose(ctx0, sd)), 0); // [1024, 1]
    cb(stats, "spk_stats_pool", -1);

    // the x-vector stays raw, the gen graph normalizes it
    ggml_tensor * dw = ggml_reshape_2d(ctx0, c.spk_dense_w, 1024, 192);
    ggml_tensor * emb = ggml_mul_mat(ctx0, dw, stats);                           // [192, 1]
    emb = bn1d(c.spk_dense_bn, emb);
    emb = ggml_cont(ctx0, ggml_reshape_1d(ctx0, emb, 192));
    cb(emb, "spk_embd", -1);
    ggml_set_name(emb, "out_xvec");
    ggml_set_output(emb);
    ggml_build_forward_expand(gf, emb);
    return gf;
}
