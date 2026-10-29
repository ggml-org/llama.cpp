#include "models.h"

#include <cstdio>
#include <vector>

// Shared omni encoder for images, audio, and grouped video.
// RoPE is chunked MTHW with m_dim = 0: sections [T, H, W] over head_dim/2,
// each section using its own theta base. ggml_rope_multi(VISION) resets
// theta per section but scales it by the full n_dims, so a constant
// freq_factors tensor restores the per-section bases.

static bool youtu_omni_fusion_layer(const clip_hparams & hparams, int il) {
    return il >= 0 && il < (int) hparams.fusion_layer.size() && hparams.fusion_layer[il] != 0;
}

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

static ggml_tensor * youtu_omni_rope(clip_graph & g, ggml_tensor * cur, ggml_tensor * pos, ggml_tensor * freq, float freq_scale) {
    const int n_rot = g.d_head / 2;
    const int per   = n_rot / 3;
    int sections[4] = {per, per, n_rot - per * 2, 0};
    return ggml_rope_multi(
        g.ctx0, cur, pos, freq,
        n_rot, sections, GGML_ROPE_TYPE_VISION,
        32768, g.hparams.rope_theta, freq_scale, 0.0f, 1.0f, 32.0f, 1.0f);
}

static ggml_tensor * youtu_omni_inp_i32(ggml_context * ctx0, const char * name, int64_t n) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n);
    ggml_set_name(t, name);
    ggml_set_input(t);
    return t;
}

// One attention group: frames and audio chunks in event order.
// Fusion layers see the whole group. Other layers keep each event to itself.
// Time is stored in half-steps (freq_scale 0.5) so a later frame can sit at 12.5.
ggml_cgraph * clip_graph_youtu_omni::build_group(ggml_tensor * freq) {
    const int n_events = img.youtu_n_events;
    const clip_youtu_event * events = img.youtu_events;
    GGML_ASSERT(n_events > 0 && events != nullptr);
    GGML_ASSERT(model.mm_norm_pre_w && model.mm_1_w && model.mm_2_w);

    std::vector<clip_youtu_span> spans;
    spans.reserve(n_events);
    int n_seq = 0;
    ggml_tensor * seq = nullptr;
    for (int i = 0; i < n_events; i++) {
        const clip_youtu_event & ev = events[i];
        const clip_youtu_span span = clip_youtu_span_of(ev, patch_size, hparams.n_merge);
        GGML_ASSERT(span.n_tok > 0 && span.n_out > 0);

        char name[32];
        snprintf(name, sizeof(name), "youtu_ev_%d", i);
        ggml_tensor * tok = nullptr;
        if (ev.kind == 0) {
            GGML_ASSERT(span.n_tok / (hparams.n_merge * hparams.n_merge) == span.n_out);
            ggml_tensor * raw = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, ev.nx, ev.ny, 3);
            ggml_set_name(raw, name);
            ggml_set_input(raw);
            tok = build_image_tokens(raw, span.pw, span.ph);
        } else {
            ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, ev.nx, ev.ny);
            ggml_set_name(mel, name);
            ggml_set_input(mel);
            tok = build_audio_tokens(mel, span.n_tok);
        }
        seq = seq ? ggml_concat(ctx0, seq, tok, 1) : tok;
        n_seq += span.n_tok;
        spans.push_back(span);
    }

    ggml_tensor * positions       = youtu_omni_inp_i32(ctx0, "positions",       n_seq * 4);
    ggml_tensor * positions_local = youtu_omni_inp_i32(ctx0, "positions_local", n_seq * 4);

    auto add_pos = [&](ggml_tensor * cur, const clip_layer & layer) {
        const int il = (int) (&layer - model.layers.data());
        ggml_tensor * pos = youtu_omni_fusion_layer(hparams, il) ? positions : positions_local;
        return youtu_omni_rope(*this, cur, pos, freq, 0.5f);
    };

    build_vit_opts opts;
    ggml_tensor * mask = build_inp_attn_mask(n_seq, n_seq);
    opts.attn_mask_layers.resize(n_layer);
    for (int il = 0; il < n_layer; il++) {
        opts.attn_mask_layers[il] = youtu_omni_fusion_layer(hparams, il) ? nullptr : mask;
    }

    ggml_tensor * cur = build_vit(seq, n_seq, NORM_TYPE_RMS, FFN_SILU, nullptr, add_pos, opts);

    // vision embeddings first, then audio, each in event order
    ggml_tensor * vis = nullptr;
    ggml_tensor * aud = nullptr;
    int offset = 0;
    for (const clip_youtu_span & span : spans) {
        ggml_tensor * part = ggml_cont(ctx0, ggml_view_2d(
            ctx0, cur, n_embd, span.n_tok, cur->nb[1], (int64_t) offset * cur->nb[1]));
        offset += span.n_tok;
        ggml_tensor *& dst = span.kind == 0 ? vis : aud;
        dst = dst ? ggml_concat(ctx0, dst, part, 1) : part;
    }
    GGML_ASSERT(vis);
    cur = build_vision_merger(vis);
    if (aud) {
        cur = ggml_concat(ctx0, cur, build_audio_merger(aud), 1);
    }
    ggml_build_forward_expand(gf, cur);
    return gf;
}

ggml_cgraph * clip_graph_youtu_omni::build() {
    ggml_tensor * freq = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, d_head / 2);
    ggml_set_name(freq, "youtu_omni_rope_factors");
    ggml_set_input(freq);

    if (model.modality != CLIP_MODALITY_AUDIO && img.youtu_n_events > 0) {
        return build_group(freq);
    }

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
        return youtu_omni_rope(*this, cur, positions, freq, 1.0f);
    };

    ggml_tensor * cur = build_vit(inp, n_pos, NORM_TYPE_RMS, FFN_SILU, nullptr, add_pos);

    cur = model.modality == CLIP_MODALITY_AUDIO ? build_audio_merger(cur) : build_vision_merger(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}
