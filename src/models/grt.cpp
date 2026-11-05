#include "models.h"

// epsilons of the reference implementation, they are not part of the model config
static constexpr float GRT_EPS_ATTN_FEAT  = 1e-6f;
static constexpr float GRT_EPS_MIXER_FEAT = 1e-2f;
static constexpr float GRT_EPS_GELU       = 1e-8f;
static constexpr float GRT_EPS_HEAD       = 1e-6f;

// stage k of a butterfly chain over n values pairs x[j] with x[j + stride] inside blocks of 2*stride
// the stride cycles through 1, 2, ..., 2^(n_levels - 1) with n_levels = max(1, floor(log2(n/2)))
static int64_t grt_n_levels(int64_t n) {
    int64_t n_levels = 1;
    while ((int64_t(1) << (n_levels + 1)) <= n/2) {
        n_levels++;
    }
    return n_levels;
}

static int64_t grt_stride(int64_t n, int64_t k) {
    return int64_t(1) << (k % grt_n_levels(n));
}

static bool grt_chain_fits(int64_t n, uint32_t n_stage) {
    if (n < 2 || n % 2 != 0) {
        return false;
    }
    // the strides repeat after n_levels stages
    for (int64_t k = 0; k < n_stage && k < grt_n_levels(n); ++k) {
        if (n % (2*grt_stride(n, k)) != 0) {
            return false;
        }
    }
    return true;
}

// pairs per mixer group: the largest of 16, 8, 4, 2 that divides the pair count
static int64_t grt_mixer_group(int64_t n_embd) {
    for (int64_t g = 16; g > 1; g /= 2) {
        if ((n_embd/2) % g == 0) {
            return g;
        }
    }
    return 1;
}

void llama_model_grt::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ROPE_FREQ_BASE,              hparams.rope_freq_base_train);
    ml.get_key(LLM_KV_ATTENTION_ROTATION_COUNT,    hparams.grt_n_rot_attn);
    ml.get_key(LLM_KV_MIXER_ROTATION_COUNT,        hparams.grt_n_rot_mixer);
    ml.get_key(LLM_KV_FEED_FORWARD_ROTATION_COUNT, hparams.grt_n_rot_ffn);

    const int64_t n_embd      = hparams.n_embd;
    const int64_t n_head      = hparams.n_head();
    const int64_t n_embd_head = hparams.n_embd_head_k();

    if (n_head == 0 || hparams.n_head_kv() != n_head || n_embd_head*n_head != n_embd || hparams.n_embd_head_v() != n_embd_head) {
        throw std::runtime_error("grt: the attention needs n_head_kv == n_head and n_embd_head*n_head == n_embd");
    }
    // the position is a rotation of the whole residual stream
    if (hparams.n_rot() != n_embd) {
        throw std::runtime_error("grt: rope dimension count must equal n_embd");
    }
    if (hparams.n_ff() < n_embd) {
        throw std::runtime_error("grt: feed_forward_length must be at least n_embd");
    }
    if (hparams.grt_n_rot_attn == 0 || hparams.grt_n_rot_ffn == 0) {
        throw std::runtime_error("grt: the attention and feed-forward rotation counts must be > 0");
    }
    if (!grt_chain_fits(n_embd_head, hparams.grt_n_rot_attn) ||
        !grt_chain_fits(n_embd,      hparams.grt_n_rot_mixer) ||
        !grt_chain_fits(hparams.n_ff(), hparams.grt_n_rot_ffn)) {
        throw std::runtime_error("grt: a rotation chain does not split into whole butterfly blocks");
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_grt::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_rot_attn  = hparams.grt_n_rot_attn;
    const int64_t n_rot_mixer = hparams.grt_n_rot_mixer;
    const int64_t n_rot_ffn   = hparams.grt_n_rot_ffn;
    const int64_t n_pair_head = n_embd_head_k/2;
    const int64_t n_pair_mix  = grt_mixer_group(n_embd);
    const int64_t n_group_mix = n_embd/2/n_pair_mix;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    // the position rotation runs once, before the first block
    layers[0].rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", 0), {n_rot/2}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        // weight: coupling of the pair features, bias: base angles, scale: per element scales, stacked over the stages
        layer.wqkv_rot   = create_tensor(tn(LLM_TENSOR_ATTN_QKV_ROT, "weight", i), {4*n_pair_head, n_pair_head, 3*n_head*n_rot_attn}, 0);
        layer.wqkv_rot_b = create_tensor(tn(LLM_TENSOR_ATTN_QKV_ROT, "bias",   i), {n_pair_head, 3*n_head, n_rot_attn}, 0);
        layer.wqkv_rot_s = create_tensor(tn(LLM_TENSOR_ATTN_QKV_ROT, "scale",  i), {n_embd_head_k, 3*n_head, n_rot_attn}, 0);
        layer.wo_rot     = create_tensor(tn(LLM_TENSOR_ATTN_OUT_ROT, "weight", i), {4*n_pair_head, n_pair_head, n_head*n_rot_attn}, 0);
        layer.wo_rot_b   = create_tensor(tn(LLM_TENSOR_ATTN_OUT_ROT, "bias",   i), {n_pair_head, n_head, n_rot_attn}, 0);
        layer.wo_rot_s   = create_tensor(tn(LLM_TENSOR_ATTN_OUT_ROT, "scale",  i), {n_embd_head_k, n_head, n_rot_attn}, 0);

        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);

        if (n_rot_mixer > 0) {
            layer.mixer_norm  = create_tensor(tn(LLM_TENSOR_MIXER_NORM, "weight", i), {n_embd}, 0);
            layer.mixer_rot   = create_tensor(tn(LLM_TENSOR_MIXER_ROT,  "weight", i), {4*n_pair_mix, n_pair_mix, n_group_mix*n_rot_mixer}, 0);
            layer.mixer_rot_b = create_tensor(tn(LLM_TENSOR_MIXER_ROT,  "bias",   i), {n_pair_mix, n_group_mix, n_rot_mixer}, 0);
            layer.mixer_rot_s = create_tensor(tn(LLM_TENSOR_MIXER_ROT,  "scale",  i), {n_embd, n_rot_mixer}, 0);
        }

        layer.ffn_norm       = create_tensor(tn(LLM_TENSOR_FFN_NORM,     "weight", i), {n_embd}, 0);
        layer.ffn_up_rot_s   = create_tensor(tn(LLM_TENSOR_FFN_UP_ROT,   "scale",  i), {n_ff, 2, n_rot_ffn}, 0);
        layer.ffn_down_rot_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_ROT, "scale",  i), {n_ff, 2, n_rot_ffn}, 0);

        layer.ffn_post_norm  = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM,  "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.layer_out_norm = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_grt::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// x / (|x| + eps) over the rows, unlike ggml_l2_norm the eps is outside the square root
static ggml_tensor * grt_normalize(ggml_context * ctx0, ggml_tensor * x, float eps) {
    ggml_tensor * norm = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, x)));
    return ggml_div(ctx0, x, ggml_scale_bias(ctx0, norm, 1.0f, eps));
}

// x [n, m] -> [n/2, m, 2] with the pairs of a butterfly stage: x[j] in the first half, x[j + stride] in the second
static ggml_tensor * grt_pair_split(ggml_context * ctx0, ggml_tensor * x, int64_t stride) {
    const int64_t n = x->ne[0];
    const int64_t m = ggml_nelements(x)/n;

    x = ggml_reshape_4d(ctx0, x, stride, 2, n/(2*stride), m);
    x = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 3, 1, 2));

    return ggml_reshape_3d(ctx0, x, n/2, m, 2);
}

// inverse of grt_pair_split, r0 and r1 are [n/2, m]
static ggml_tensor * grt_pair_join(ggml_context * ctx0, ggml_tensor * r0, ggml_tensor * r1, int64_t stride) {
    const int64_t n_pair = r0->ne[0];
    const int64_t m      = ggml_nelements(r0)/n_pair;

    r0 = ggml_reshape_4d(ctx0, r0, stride, 1, n_pair/stride, m);
    r1 = ggml_reshape_4d(ctx0, r1, stride, 1, n_pair/stride, m);

    return ggml_reshape_2d(ctx0, ggml_concat(ctx0, r0, r1, 1), 2*n_pair, m);
}

// first (h = 0) or second (h = 1) element of the pairs in x [n_pair, n_tokens, n_batch, 2]
static ggml_tensor * grt_half(ggml_context * ctx0, ggml_tensor * x, int h) {
    return ggml_view_3d(ctx0, x, x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], h*x->nb[3]);
}

// content-modulated rotation chain on x [n, n_tokens, n_head], used by the attention projections and the mixer
// a stage rotates its pairs by the base angle plus tanh of a coupling of the pair features, then scales the result
// the pairs of a head form groups of n_pair pairs with one coupling each, more than one group only happens in the mixer
static ggml_tensor * grt_coupled_chain(
        ggml_context * ctx0,
        ggml_tensor  * x,
        ggml_tensor  * coupling,
        ggml_tensor  * angle,
        ggml_tensor  * scale,
        int64_t        n_stage,
        int64_t        n_pair,
        float          eps,
        bool           mixer) {
    const int64_t n       = x->ne[0];
    const int64_t n_tok   = x->ne[1];
    const int64_t n_head  = x->ne[2];
    const int64_t n_group = n/2/n_pair;
    const int64_t n_batch = n_head*n_group;

    GGML_ASSERT(n_head == 1 || n_group == 1);

    for (int64_t k = 0; k < n_stage; ++k) {
        const int64_t stride = grt_stride(n, k);

        ggml_tensor * xp = grt_pair_split(ctx0, x, stride);
        if (n_group > 1) {
            // the coupling is batched over the groups, so they go after the tokens
            xp = ggml_reshape_4d(ctx0, xp, n_pair, n_group, n_tok, 2);
            xp = ggml_cont(ctx0, ggml_permute(ctx0, xp, 0, 2, 1, 3));
        } else {
            xp = ggml_reshape_4d(ctx0, xp, n_pair, n_tok, n_head, 2);
        }

        // features: pair energy, both components and the bivector with the next pair of the group
        // the mixer was trained with the other bivector contraction
        ggml_tensor * sq   = ggml_sqr(ctx0, xp);
        ggml_tensor * prod = ggml_mul(ctx0, xp, ggml_roll(ctx0, xp, -1, 0, 0, mixer ? 0 : 1));

        ggml_tensor * feat = ggml_add(ctx0, grt_half(ctx0, sq, 0), grt_half(ctx0, sq, 1));
        feat = ggml_concat(ctx0, feat, grt_half(ctx0, xp, 0), 0);
        feat = ggml_concat(ctx0, feat, grt_half(ctx0, xp, 1), 0);
        feat = ggml_concat(ctx0, feat, ggml_sub(ctx0, grt_half(ctx0, prod, 0), grt_half(ctx0, prod, 1)), 0);
        feat = grt_normalize(ctx0, feat, eps);

        ggml_tensor * coupling_k = ggml_view_3d(ctx0, coupling, coupling->ne[0], coupling->ne[1], n_batch,
                coupling->nb[1], coupling->nb[2], k*n_batch*coupling->nb[2]);
        ggml_tensor * angle_k = ggml_view_3d(ctx0, angle, n_pair, 1, n_batch, angle->nb[1], angle->nb[1], k*angle->nb[2]);
        ggml_tensor * scale_k = ggml_view_3d(ctx0, scale, n, 1, n_head, scale->nb[1], scale->nb[1], k*n*n_head*ggml_element_size(scale));

        ggml_tensor * theta = ggml_mul_mat(ctx0, coupling_k, feat);
        theta = ggml_add(ctx0, ggml_tanh(ctx0, theta), angle_k);

        ggml_tensor * xc = ggml_mul(ctx0, xp, ggml_cos(ctx0, theta));
        ggml_tensor * xs = ggml_mul(ctx0, xp, ggml_sin(ctx0, theta));

        ggml_tensor * r0 = ggml_sub(ctx0, grt_half(ctx0, xc, 0), grt_half(ctx0, xs, 1));
        ggml_tensor * r1 = ggml_add(ctx0, grt_half(ctx0, xs, 0), grt_half(ctx0, xc, 1));
        if (n_group > 1) {
            r0 = ggml_cont(ctx0, ggml_permute(ctx0, r0, 0, 2, 1, 3));
            r1 = ggml_cont(ctx0, ggml_permute(ctx0, r1, 0, 2, 1, 3));
        }

        // the converter stores the scales in the unpermuted order, so they apply after the join
        x = grt_pair_join(ctx0, r0, r1, stride);
        x = ggml_mul(ctx0, ggml_reshape_3d(ctx0, x, n, n_tok, n_head), scale_k);
    }

    return x;
}

// fixed rotation chain of the feed-forward on x [n, n_tokens]
// the converter folds the rotation and the scale of a stage into coefficients a and b: x*a + partner(x)*b
static ggml_tensor * grt_plain_chain(ggml_context * ctx0, ggml_tensor * x, ggml_tensor * coef, int64_t n_stage) {
    const int64_t n     = x->ne[0];
    const int64_t n_tok = x->ne[1];

    for (int64_t k = 0; k < n_stage; ++k) {
        const int64_t stride = grt_stride(n, k);

        ggml_tensor * partner = ggml_reshape_4d(ctx0, x, stride, 2, n/(2*stride), n_tok);
        partner = ggml_reshape_2d(ctx0, ggml_roll(ctx0, partner, 0, 1, 0, 0), n, n_tok);

        ggml_tensor * a = ggml_view_1d(ctx0, coef, n, (2*k + 0)*coef->nb[1]);
        ggml_tensor * b = ggml_view_1d(ctx0, coef, n, (2*k + 1)*coef->nb[1]);

        x = ggml_add(ctx0, ggml_mul(ctx0, x, a), ggml_mul(ctx0, partner, b));
    }

    return x;
}

llama_model_grt::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t n_ff        = hparams.n_ff();
    const int64_t n_pair_head = n_embd_head/2;
    const int64_t n_pair_mix  = grt_mixer_group(n_embd);

    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    // the position is a rotation of the whole residual stream before the first block, the projections then carry it into q and k
    // this is also why the model reports no rope type: a cached k cannot be rotated to a new position
    inpL = ggml_rope_ext(ctx0, ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens), inp_pos, model.layers[0].rope_freqs,
            n_rot, LLAMA_ROPE_TYPE_NORM, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    inpL = ggml_reshape_2d(ctx0, inpL, n_embd, n_tokens);
    cb(inpL, "inp_rope", -1);

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const llama_layer & layer = model.layers[il];

        cur = build_norm(inpL, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention, q, k and v read the same input and run as one chain over 3*n_head heads
        {
            cur = ggml_reshape_3d(ctx0, cur, n_embd_head, n_head, n_tokens);
            cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
            cur = ggml_repeat_4d(ctx0, cur, n_embd_head, n_tokens, 3*n_head, 1);

            cur = grt_coupled_chain(ctx0, cur, layer.wqkv_rot, layer.wqkv_rot_b, layer.wqkv_rot_s,
                    hparams.grt_n_rot_attn, n_pair_head, GRT_EPS_ATTN_FEAT, false);
            cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));

            ggml_tensor * Qcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head, n_tokens, cur->nb[1], cur->nb[2], 0*n_head*cur->nb[1]);
            ggml_tensor * Kcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head, n_tokens, cur->nb[1], cur->nb[2], 1*n_head*cur->nb[1]);
            ggml_tensor * Vcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head, n_tokens, cur->nb[1], cur->nb[2], 2*n_head*cur->nb[1]);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);

            cur = ggml_reshape_3d(ctx0, cur, n_embd_head, n_head, n_tokens);
            cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));

            cur = grt_coupled_chain(ctx0, cur, layer.wo_rot, layer.wo_rot_b, layer.wo_rot_s,
                    hparams.grt_n_rot_attn, n_pair_head, GRT_EPS_ATTN_FEAT, false);
            cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
            cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
            cb(cur, "attn_out", il);
        }

        if (layer.attn_post_norm) {
            cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "attn_post_norm", il);
        }

        // the residual gates are folded into the last scales of each sublayer at conversion
        inpL = ggml_add(ctx0, inpL, cur);
        cb(inpL, "attn_residual", il);

        // cross-head mixer
        if (layer.mixer_rot) {
            cur = build_norm(inpL, layer.mixer_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "mixer_norm", il);

            cur = grt_coupled_chain(ctx0, ggml_reshape_3d(ctx0, cur, n_embd, n_tokens, 1),
                    layer.mixer_rot, layer.mixer_rot_b, layer.mixer_rot_s,
                    hparams.grt_n_rot_mixer, n_pair_mix, GRT_EPS_MIXER_FEAT, true);
            cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
            cb(cur, "mixer_out", il);

            inpL = ggml_add(ctx0, inpL, cur);
            cb(inpL, "mixer_residual", il);
        }

        cur = build_norm(inpL, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // feed-forward: zero pad to n_ff, rotate, gelu on the radius of each pair, rotate, truncate
        {
            cur = ggml_pad(ctx0, cur, (int) (n_ff - n_embd), 0, 0, 0);
            cur = grt_plain_chain(ctx0, cur, layer.ffn_up_rot_s, hparams.grt_n_rot_ffn);

            ggml_tensor * pairs  = ggml_reshape_3d(ctx0, cur, 2, n_ff/2, n_tokens);
            ggml_tensor * radius = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, pairs)));
            ggml_tensor * gate   = ggml_div(ctx0, ggml_gelu_erf(ctx0, radius), ggml_scale_bias(ctx0, radius, 1.0f, GRT_EPS_GELU));
            cur = ggml_reshape_2d(ctx0, ggml_mul(ctx0, pairs, gate), n_ff, n_tokens);
            cb(cur, "ffn_act", il);

            cur = grt_plain_chain(ctx0, cur, layer.ffn_down_rot_s, hparams.grt_n_rot_ffn);
            cur = ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 0);
            cb(cur, "ffn_out", il);
        }

        if (layer.ffn_post_norm) {
            cur = build_norm(cur, layer.ffn_post_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "ffn_post_norm", il);
        }

        inpL = ggml_add(ctx0, inpL, cur);

        if (layer.layer_out_norm) {
            inpL = build_norm(inpL, layer.layer_out_norm, nullptr, LLM_NORM_RMS, il);
        }

        inpL = build_cvec(inpL, il);
        cb(inpL, "l_out", il);
    }

    // prune after the last block: a ubatch can have no outputs and ggml_roll does not take empty tensors
    if (inp_out_ids) {
        inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
    }

    cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // cosine head, output holds the normalized embedding table times the logit scale
    cur = build_lora_mm(model.output, grt_normalize(ctx0, cur, GRT_EPS_HEAD));

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
