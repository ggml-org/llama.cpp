#include "models.h"

// Shared omni encoder for images and audio.
// RoPE is chunked MTHW with m_dim = 0: sections [T, H, W] over head_dim/2,
// each section using its own theta base. ggml_rope_multi(VISION) resets
// theta per section but scales it by the full n_dims, so a constant
// freq_factors tensor restores the per-section bases.

ggml_tensor * clip_graph_youtu_omni::build_mm(ggml_tensor * w, ggml_tensor * x) const {
    ggml_tensor * cur = ggml_mul_mat(ctx0, w, x);
    // SwiGLU outputs can exceed the f16 range, so the down projection must accumulate in f32
    for (const auto & layer : model.layers) {
        if (w == layer.ff_down_w) {
            ggml_prec_set_acc(cur, GGML_PREC_F32);
            break;
        }
    }
    return cur;
}

// raw: [nx, ny, 3] planar pixels. Returns [n_embd, pw*ph] in 2x2 merge order (same as qwen2vl).
ggml_tensor * clip_graph_youtu_omni::build_image_tokens(ggml_tensor * raw, int pw, int ph) {
    GGML_ASSERT(hparams.n_merge == 2);
    GGML_ASSERT(pw % 2 == 0 && ph % 2 == 0);
    ggml_tensor * cur = ggml_conv_2d(ctx0, model.patch_embeddings_0, raw, patch_size, patch_size, 0, 0, 1, 1);
    cur = ggml_permute(ctx0, cur, 1, 2, 0, 3); // [w, h, c, b] -> [c, w, h, b]
    cur = ggml_cont_4d(ctx0, cur, n_embd * 2, pw / 2, ph, 1);
    cur = ggml_reshape_4d(ctx0, cur, n_embd * 2, pw / 2, 2, ph / 2);
    cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
    cur = ggml_cont_2d(ctx0, cur, n_embd, pw * ph);
    if (model.patch_bias) {
        cur = ggml_add(ctx0, cur, model.patch_bias);
    }
    return cur;
}

// mel: [n_frames, n_mel], padded to whole 100-frame chunks. Returns [n_embd, n_keep].
ggml_tensor * clip_graph_youtu_omni::build_audio_tokens(ggml_tensor * mel, int n_keep) {
    ggml_tensor * cur = build_audio_conv2d_stem(mel);
    const int64_t n_full = cur->ne[1];
    // padded frames only shorten the last chunk (padded_mask_after_cnn), so the kept tokens are a prefix
    GGML_ASSERT(n_keep > n_full - 13 && n_keep <= n_full);
    if (n_keep < n_full) {
        cur = ggml_view_2d(ctx0, cur, cur->ne[0], n_keep, cur->nb[1], 0);
    }
    return cur;
}

// cur: [n_embd, n_tok] in 2x2 merge order
ggml_tensor * clip_graph_youtu_omni::build_vision_merger(ggml_tensor * cur) {
    const int merge2 = hparams.n_merge * hparams.n_merge;
    cur = ggml_reshape_2d(ctx0, cur, n_embd * merge2, cur->ne[1] / merge2);
    cur = build_norm(cur, model.mm_input_norm_w, nullptr, NORM_TYPE_RMS, eps, -1);
    return build_ffn(cur,
        model.mm_merger_fc1_w, nullptr,
        nullptr, nullptr,
        model.mm_merger_fc2_w, nullptr,
        FFN_GELU_ERF, -1);
}

// temporal_merge_size is 1: RMSNorm over hidden, then a bias-free GELU MLP
ggml_tensor * clip_graph_youtu_omni::build_audio_merger(ggml_tensor * cur) {
    cur = build_norm(cur, model.mm_norm_pre_w, nullptr, NORM_TYPE_RMS, eps, -1);
    return build_ffn(cur,
        model.mm_1_w, nullptr,
        nullptr, nullptr,
        model.mm_2_w, nullptr,
        FFN_GELU_ERF, -1);
}

static ggml_tensor * youtu_omni_rope(clip_graph & g, ggml_tensor * cur, ggml_tensor * pos, ggml_tensor * freq) {
    const int n_rot = g.d_head / 2;
    const int per   = n_rot / 3;
    int sections[4] = {per, per, n_rot - per * 2, 0};
    return ggml_rope_multi(
        g.ctx0, cur, pos, freq,
        n_rot, sections, GGML_ROPE_TYPE_VISION,
        32768, g.hparams.rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
}

static ggml_tensor * youtu_omni_inp_i32(ggml_context * ctx0, const char * name, int64_t n) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n);
    ggml_set_name(t, name);
    ggml_set_input(t);
    return t;
}

ggml_cgraph * clip_graph_youtu_omni::build() {
    ggml_tensor * freq = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, d_head / 2);
    ggml_set_name(freq, "youtu_omni_rope_factors");
    ggml_set_input(freq);

    ggml_tensor * inp = nullptr;
    int n_pos = 0;
    if (model.modality == CLIP_MODALITY_AUDIO) {
        // the whole mel chunk is one bidirectional segment, t = 0..n-1, h = w = 0
        GGML_ASSERT(model.mm_norm_pre_w && model.mm_1_w && model.mm_2_w);
        n_pos = clip_youtu_omni_audio_tokens(img.audio_n_frames > 0 ? img.audio_n_frames : img.nx());
        inp = build_audio_tokens(build_inp_raw(1), n_pos);
    } else {
        GGML_ASSERT(model.mm_input_norm_w && model.mm_merger_fc1_w && model.mm_merger_fc2_w);
        n_pos = n_patches;
        inp = build_image_tokens(build_inp_raw(), n_patches_x, n_patches_y);
    }

    ggml_tensor * positions = youtu_omni_inp_i32(ctx0, "positions", n_pos * 4);
    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return youtu_omni_rope(*this, cur, positions, freq);
    };

    ggml_tensor * cur = build_vit(inp, n_pos, NORM_TYPE_RMS, FFN_SILU, nullptr, add_pos);

    cur = model.modality == CLIP_MODALITY_AUDIO ? build_audio_merger(cur) : build_vision_merger(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}
