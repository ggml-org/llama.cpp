#include "models.h"

// VibeVoice-ASR audio tower: two causal ConvNeXt pyramids (acoustic and semantic) over the raw 24 kHz waveform
// each one is a VAE tokenizer encoder, its latents go through their own connector and the two are summed
// see modeling_vibevoice_acoustic_tokenizer.py and modeling_vibevoice_asr.py in transformers
//
// tensors are T-first [T, C] around the convs and C-first [C, T] in the blocks

static std::string cnx_slot(int e, const char * kind, int i) {
    return "e" + std::to_string(e) + "_" + kind + "_" + std::to_string(i);
}

std::vector<c2w_state_slot> list_vibevoice_asr_state_slots(const mtmd_hparams & hparams, const mtmd_model & model) {
    std::vector<c2w_state_slot> slots;
    slots.push_back({"pos", 1, 1}); // unused, set_enc_state_in() expects it
    const int n_stage = (int) hparams.audio_patch_sizes.size();
    for (int e = 0; e < 2; ++e) {
        const auto & enc = model.cnx_enc[e];
        for (int i = 0; i <= n_stage; ++i) {
            const int64_t stride = i < n_stage ? hparams.audio_patch_sizes[i] : 1;
            slots.push_back({cnx_slot(e, "conv", i), enc.conv_w[i]->ne[0] - stride, enc.conv_w[i]->ne[1]});
        }
        for (int il = 0; il < (int) enc.blocks.size(); ++il) {
            const auto * w = enc.blocks[il].conv_w;
            slots.push_back({cnx_slot(e, "dw", il), w->ne[0] - 1, w->ne[1]});
        }
    }
    return slots;
}

// x: [T, C] -> [left + T, C], the left context comes from the state, its new value is the tail
ggml_tensor * mtmd_graph_vibevoice_asr::with_left_ctx(ggml_tensor * x, const std::string & name) const {
    ggml_tensor * left = state_in.at(name);
    x = ggml_concat(ctx0, left, x, 0);
    state_out.push_back({name, ggml_cont(ctx0, ggml_view_2d(ctx0, x, left->ne[0], x->ne[1], x->nb[1], (x->ne[0] - left->ne[0]) * x->nb[0]))});
    return x;
}

// x: [T, IC], w: [K, IC, OC] -> [T / stride, OC]
ggml_tensor * mtmd_graph_vibevoice_asr::causal_conv(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int stride, const std::string & name) const {
    x = with_left_ctx(x, name);
    ggml_tensor * y = ggml_conv_1d(ctx0, w, x, stride, 0, 1);
    y = ggml_reshape_2d(ctx0, y, y->ne[0], y->ne[1]);
    return ggml_add(ctx0, y, ggml_reshape_2d(ctx0, b, 1, b->ne[0]));
}

// x: [T, 1] -> [C_latent, T / hop]
ggml_tensor * mtmd_graph_vibevoice_asr::encode(int e, ggml_tensor * x) const {
    const auto & enc     = model.cnx_enc[e];
    const int    n_stage = (int) hparams.audio_patch_sizes.size();

    ggml_tensor * cur = x;
    int bid = 0;
    for (int stage = 0; stage < n_stage; ++stage) {
        cur = causal_conv(cur, enc.conv_w[stage], enc.conv_b[stage], hparams.audio_patch_sizes[stage], cnx_slot(e, "conv", stage));
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        cb(cur, "stage_in", stage);

        const int64_t n_pos = cur->ne[1];
        for (int i = 0; i < hparams.audio_depths[stage]; ++i, ++bid) {
            const auto & layer = enc.blocks[bid];

            // causal depthwise conv mixer
            ggml_tensor * h = build_norm(cur, layer.ln_1_w, nullptr, NORM_TYPE_RMS, hparams.eps, bid);
            h = ggml_cont(ctx0, ggml_transpose(ctx0, h));
            h = with_left_ctx(h, cnx_slot(e, "dw", bid));
            h = ggml_reshape_3d(ctx0, h, h->ne[0], h->ne[1], 1);
            h = ggml_ssm_conv(ctx0, h, layer.conv_w);
            h = ggml_reshape_2d(ctx0, h, h->ne[0], n_pos);
            h = ggml_add(ctx0, h, layer.conv_b);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, h, layer.ls_1_w));

            h = build_norm(cur, layer.ln_2_w, nullptr, NORM_TYPE_RMS, hparams.eps, bid);
            h = build_ffn(h,
                layer.ff_up_w, layer.ff_up_b,
                nullptr, nullptr,
                layer.ff_down_w, layer.ff_down_b,
                FFN_GELU_ERF, bid);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, h, layer.ls_2_w));
            cb(cur, "block_out", bid);
        }

        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    }

    cur = causal_conv(cur, enc.conv_w[n_stage], enc.conv_b[n_stage], 1, cnx_slot(e, "conv", n_stage));
    return ggml_cont(ctx0, ggml_transpose(ctx0, cur));
}

ggml_cgraph * mtmd_graph_vibevoice_asr::build() {
    const int64_t hop      = hparams.audio_samples_per_token();
    const int64_t n_frames = img.nx() / hop;
    const int64_t n_ctx    = img.audio_n_ctx / hop;
    GGML_ASSERT(img.nx() % hop == 0 && img.audio_n_ctx % hop == 0 && n_ctx < n_frames);

    for (const auto & slot : list_vibevoice_asr_state_slots(hparams, model)) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, slot.ne0, slot.ne1);
        ggml_set_name(t, ("state_in_" + slot.name).c_str());
        ggml_set_input(t);
        state_in[slot.name] = t;
    }
    state_out.push_back({"pos", ggml_scale_bias(ctx0, state_in.at("pos"), 1.0f, (float) n_frames)});

    ggml_tensor * inp = build_inp_raw(1);
    inp = ggml_reshape_2d(ctx0, inp, img.nx(), 1);

    // the leading context frames only warm up the causal convs
    auto drop_ctx = [&](ggml_tensor * t) {
        return ggml_view_2d(ctx0, t, t->ne[0], n_frames - n_ctx, t->nb[1], n_ctx * t->nb[1]);
    };

    // the reference adds VAE noise to the acoustic latents, skipped here to keep the output deterministic
    ggml_tensor * acoustic = drop_ctx(encode(0, inp));
    ggml_tensor * semantic = drop_ctx(encode(1, inp));
    cb(acoustic, "acoustic_latents", -1);
    cb(semantic, "semantic_latents", -1);

    acoustic = ggml_add(ctx0, build_mm(model.mm_0_w, acoustic), model.mm_0_b);
    acoustic = build_norm(acoustic, model.mm_norm_pre_w, nullptr, NORM_TYPE_RMS, 1e-6f, -1);
    acoustic = ggml_add(ctx0, build_mm(model.mm_1_w, acoustic), model.mm_1_b);

    semantic = ggml_add(ctx0, build_mm(model.mm_2_w, semantic), model.mm_2_b);
    semantic = build_norm(semantic, model.mm_norm_mid_w, nullptr, NORM_TYPE_RMS, 1e-6f, -1);
    semantic = ggml_add(ctx0, build_mm(model.mm_3_w, semantic), model.mm_3_b);

    ggml_tensor * cur = ggml_add(ctx0, acoustic, semantic);
    cb(cur, "projected", -1);

    // the embeddings must stay the last node
    for (const auto & s : state_out) {
        ggml_set_name(s.second, ("state_out_" + s.first).c_str());
        ggml_set_output(s.second);
        ggml_build_forward_expand(gf, s.second);
    }

    ggml_build_forward_expand(gf, cur);
    return gf;
}
