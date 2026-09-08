#include "mediagen-ltx-graph.h"

#include <cmath>

// activations are [W, H, C, T]

// 3x3x3 conv, replicate padded in time, as the sum of three 2-D convs over the temporal taps
static ggml_tensor * vae_conv3d(ggml_context * ctx, const mg_weights & w, const std::string & prefix, ggml_tensor * x, bool causal) {
    const int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2], T = x->ne[3];
    ggml_tensor * k0 = w.get(prefix + ".weight.t0");
    ggml_tensor * k1 = w.get(prefix + ".weight.t1");
    ggml_tensor * k2 = w.get(prefix + ".weight.t2");
    ggml_tensor * b  = w.get(prefix + ".bias", false);

    ggml_tensor * y;
    if (T == 1) {
        // every tap sees the same frame
        ggml_tensor * y0 = ggml_conv_2d(ctx, k0, x, 1, 1, 1, 1, 1, 1);
        ggml_tensor * y1 = ggml_conv_2d(ctx, k1, x, 1, 1, 1, 1, 1, 1);
        ggml_tensor * y2 = ggml_conv_2d(ctx, k2, x, 1, 1, 1, 1, 1, 1);
        y = ggml_add(ctx, ggml_add(ctx, y0, y1), y2);
    } else {
        ggml_tensor * first = ggml_view_4d(ctx, x, W, H, C, 1, x->nb[1], x->nb[2], x->nb[3], 0);
        ggml_tensor * last  = ggml_view_4d(ctx, x, W, H, C, 1, x->nb[1], x->nb[2], x->nb[3], (T - 1) * x->nb[3]);
        ggml_tensor * xp;
        if (causal) {
            xp = ggml_concat(ctx, ggml_concat(ctx, first, first, 3), x, 3);
        } else {
            xp = ggml_concat(ctx, ggml_concat(ctx, first, x, 3), last, 3);
        }
        y = nullptr;
        ggml_tensor * ks[3] = { k0, k1, k2 };
        for (int k = 0; k < 3; k++) {
            ggml_tensor * xv = ggml_view_4d(ctx, xp, W, H, C, T, xp->nb[1], xp->nb[2], xp->nb[3], k * xp->nb[3]);
            ggml_tensor * yk = ggml_conv_2d(ctx, ks[k], xv, 1, 1, 1, 1, 1, 1);
            y = y ? ggml_add(ctx, y, yk) : yk;
        }
    }
    if (b) {
        y = ggml_add(ctx, y, ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1));
    }
    return y;
}

static ggml_tensor * vae_pixel_norm(ggml_context * ctx, ggml_tensor * x) {
    return ltx_pixel_norm(ctx, x, 1e-8f);
}

static ggml_tensor * vae_resblock(ggml_context * ctx, const mg_weights & w, const std::string & prefix, ggml_tensor * x, bool causal) {
    ggml_tensor * h = ggml_silu(ctx, vae_pixel_norm(ctx, x));
    h = vae_conv3d(ctx, w, prefix + ".conv1.conv", h, causal);
    h = ggml_silu(ctx, vae_pixel_norm(ctx, h));
    h = vae_conv3d(ctx, w, prefix + ".conv2.conv", h, causal);
    return ggml_add(ctx, x, h);
}

// move the innermost factor `p` of the channel dim into the spatial width: [W, H, p*R, T] -> [p*W, H, R, T]
static ggml_tensor * vae_shuffle_w(ggml_context * ctx, ggml_tensor * y, int p) {
    const int64_t W = y->ne[0], H = y->ne[1], C = y->ne[2], T = y->ne[3];
    ggml_tensor * v = ggml_reshape_4d(ctx, y, W * H, p, C / p, T);
    v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3)); // [p, W*H, R, T]
    return ggml_reshape_4d(ctx, v, p * W, H, C / p, T);
}

// move the innermost factor `p` of the channel dim into the spatial height: [W, H, p*R, T] -> [W, p*H, R, T]
static ggml_tensor * vae_shuffle_h(ggml_context * ctx, ggml_tensor * y, int p) {
    const int64_t W = y->ne[0], H = y->ne[1], C = y->ne[2], T = y->ne[3];
    ggml_tensor * v = ggml_reshape_4d(ctx, y, W, H, p, (C / p) * T);
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3)); // [W, p, H, R*T]
    return ggml_reshape_4d(ctx, v, W, p * H, C / p, T);
}

// move the innermost factor `p` of the channel dim into time: [W, H, p*R, T] -> [W, H, R, p*T]
static ggml_tensor * vae_shuffle_t(ggml_context * ctx, ggml_tensor * y, int p) {
    const int64_t W = y->ne[0], H = y->ne[1], C = y->ne[2], T = y->ne[3];
    ggml_tensor * v = ggml_reshape_4d(ctx, y, W * H, p, C / p, T);
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3)); // [W*H, R, p, T]
    return ggml_reshape_4d(ctx, v, W, H, C / p, p * T);
}

// DepthToSpaceUpsample: conv, pixel shuffle (pt, ps, ps), drop the first frame when pt == 2
static ggml_tensor * vae_upsample(ggml_context * ctx, const mg_weights & w, const std::string & prefix, ggml_tensor * x, int pt, int ps, bool causal) {
    ggml_tensor * y = vae_conv3d(ctx, w, prefix + ".conv.conv", x, causal);
    // channel index = c*(pt*ps*ps) + t*(ps*ps) + h*ps + w
    if (ps > 1) {
        y = vae_shuffle_w(ctx, y, ps);
        y = vae_shuffle_h(ctx, y, ps);
    }
    if (pt > 1) {
        y = vae_shuffle_t(ctx, y, pt);
        const int64_t T = y->ne[3];
        y = ggml_view_4d(ctx, y, y->ne[0], y->ne[1], y->ne[2], T - 1, y->nb[1], y->nb[2], y->nb[3], y->nb[3]);
        y = ggml_cont(ctx, y);
    }
    return y;
}

struct vae_block_desc {
    std::string kind;
    int n_layers  = 0;
    int multiplier = 1;
};

// LTX-2.x decoder blocks (the config lists them in encoder order)
static std::vector<vae_block_desc> ltx_decoder_blocks() {
    return {
        { "res_x", 2, 1 },
        { "compress_all", 0, 2 },
        { "res_x", 2, 1 },
        { "compress_all", 0, 1 },
        { "res_x", 4, 1 },
        { "compress_time", 0, 2 },
        { "res_x", 6, 1 },
        { "compress_space", 0, 2 },
        { "res_x", 4, 1 },
    };
}

bool ltx_vae_decode(const ltx_model & model, mg_backend & be, const ltx_latent_video & lat,
                    int & out_frames, int & out_height, int & out_width, std::vector<float> & rgb) {
    const auto & hp = model.hp;
    const auto & w  = model.vae;
    const int F = lat.n_frames, Hl = lat.height, Wl = lat.width, C = lat.channels;
    const bool causal = false; // causal_decoder = false

    // denormalize, lay out as [W, H, C, T]
    std::vector<float> mean, stdv;
    if (!mg_tensor_to_f32(w.get("per_channel_statistics.mean-of-means"), C, mean) ||
        !mg_tensor_to_f32(w.get("per_channel_statistics.std-of-means"),  C, stdv)) {
        return false;
    }
    std::vector<float> x((size_t) Wl * Hl * C * F);
    for (int f = 0; f < F; f++) {
        for (int h = 0; h < Hl; h++) {
            for (int wi = 0; wi < Wl; wi++) {
                const float * src = lat.x.data() + ((size_t) f * Hl * Wl + (size_t) h * Wl + wi) * C;
                for (int c = 0; c < C; c++) {
                    x[wi + (size_t) Wl * (h + (size_t) Hl * (c + (size_t) C * f))] = src[c] * stdv[c] + mean[c];
                }
            }
        }
    }

    mg_graph g(be);
    ggml_context * ctx = g.get();
    ggml_tensor * cur = g.new_input_f32({ Wl, Hl, C, F }, x, "latent");

    cur = vae_conv3d(ctx, w, "decoder.conv_in.conv", cur, causal);

    const auto blocks = ltx_decoder_blocks();
    for (size_t i = 0; i < blocks.size(); i++) {
        const auto & bd = blocks[i];
        const std::string bp = mg_format("decoder.up_blocks.%zu", i);
        if (bd.kind == "res_x") {
            for (int l = 0; l < bd.n_layers; l++) {
                cur = vae_resblock(ctx, w, mg_format("%s.res_blocks.%d", bp.c_str(), l), cur, causal);
            }
        } else if (bd.kind == "compress_all") {
            cur = vae_upsample(ctx, w, bp, cur, 2, 2, causal);
        } else if (bd.kind == "compress_time") {
            cur = vae_upsample(ctx, w, bp, cur, 2, 1, causal);
        } else if (bd.kind == "compress_space") {
            cur = vae_upsample(ctx, w, bp, cur, 1, 2, causal);
        } else {
            MG_ERR("%s: unknown block %s\n", __func__, bd.kind.c_str());
            return false;
        }
    }

    cur = ggml_silu(ctx, vae_pixel_norm(ctx, cur));
    cur = vae_conv3d(ctx, w, "decoder.conv_out.conv", cur, causal);

    // unpatchify: channel = c*16 + r*4 + q with q -> height, r -> width
    const int p = hp.vae_patch_size;
    cur = vae_shuffle_h(ctx, cur, p);
    cur = vae_shuffle_w(ctx, cur, p);

    g.mark_output(cur);
    if (!g.compute(be)) {
        return false;
    }

    out_width  = (int) cur->ne[0];
    out_height = (int) cur->ne[1];
    out_frames = (int) cur->ne[3];
    if (cur->ne[2] != 3) {
        MG_ERR("%s: decoder produced %lld channels\n", __func__, (long long) cur->ne[2]);
        return false;
    }

    std::vector<float> y;
    g.get_output(cur, y);
    // [W, H, 3, T] -> [T][H][W][3]
    rgb.resize(y.size());
    const size_t W = out_width, H = out_height;
    for (int f = 0; f < out_frames; f++) {
        for (size_t h = 0; h < H; h++) {
            for (size_t wi = 0; wi < W; wi++) {
                for (int c = 0; c < 3; c++) {
                    rgb[((f * H + h) * W + wi) * 3 + c] = y[wi + W * (h + H * (c + 3 * (size_t) f))];
                }
            }
        }
    }
    return true;
}
