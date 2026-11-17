#include "models.h"

// Moshi family (e.g. Hibiki), see conversion/moshi.py
//
// encoder (audio stream): streaming mimi encoder -> RVQ -> backbone embeddings of the input codes
// GEN_CODE: depformer, the codes of one frame from the backbone hidden state and the text token, then their embedding
// GEN_WAV : RVQ decode -> streaming mimi decoder

// 2D slice i of a tensor stacked on dim 2
static ggml_tensor * moshi_slice(ggml_context * ctx0, ggml_tensor * w, int64_t i) {
    return ggml_view_2d(ctx0, w, w->ne[0], w->ne[1], w->nb[1], (size_t) i * w->nb[2]);
}

std::vector<c2w_state_slot> list_moshi_state_slots(const mtmd_hparams & hparams, const mtmd_model & model) {
    std::vector<c2w_state_slot> slots;
    const auto & seanet = model.seanet;

    slots.push_back({"pos", 1, 1});

    const int64_t n_embd_a = model.layers[0].q_w->ne[1];
    const int64_t prefix   = hparams.mimi_tfm_context - 1;
    for (size_t il = 0; il < model.layers.size(); il++) {
        slots.push_back({"tfm_k_" + std::to_string(il), n_embd_a, prefix});
        slots.push_back({"tfm_v_" + std::to_string(il), n_embd_a, prefix});
    }

    // left context of the causal convs, [kernel - stride, in channels]
    slots.push_back({"enc_in", seanet.conv_in_w->ne[0] - 1, seanet.conv_in_w->ne[1]});
    for (int i = 0; i < hparams.seanet_n_stage; i++) {
        const auto & stage = seanet.stages[i];
        slots.push_back({"enc_res_" + std::to_string(i), stage.res_conv1_w->ne[0] - 1, stage.res_conv1_w->ne[1]});
        slots.push_back({"enc_down_" + std::to_string(i), stage.scale_conv_w->ne[0] - hparams.seanet_ratios[i], stage.scale_conv_w->ne[1]});
    }
    slots.push_back({"enc_out", seanet.conv_out_w->ne[0] - 1, seanet.conv_out_w->ne[1]});
    slots.push_back({"down", model.downsample_w->ne[0] - hparams.mimi_downsample, model.downsample_w->ne[1]});

    return slots;
}

// waveform of whole frames -> [n_embd, n_q, n_frames + 1]: the embedding of each input code, then of the initial codes
ggml_cgraph * mtmd_graph_moshi::build() {
    // the waveform comes in as a single-row "mel", [n_samples, 1]
    ggml_tensor * inp_raw = build_inp_raw(1);
    ggml_tensor * cur = ggml_reshape_2d(ctx0, inp_raw, inp_raw->ne[0], inp_raw->ne[1]);
    GGML_ASSERT(cur->ne[0] % n_samples_per_frame == 0);

    // the graph always streams, a zero state encodes a clip from its start
    mtmd_graph_pockettts_seanet seanet(*this);
    seanet.state_inputs(list_moshi_state_slots(hparams, model));
    ggml_tensor * pos_in = seanet.state_in.at("pos");

    cur = seanet.encode(cur);
    cb(cur, "mimi_enc", -1);

    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    cur = seanet.transformer(cur, model.layers, hparams.n_head, hparams.eps);
    cb(cur, "mimi_enc_tfm", -1);

    // the downsample conv repeats the first frame on its left at the stream start, the state is zero then
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    {
        const int64_t n_left = seanet.state_in.at("down")->ne[0];
        ggml_tensor * first    = ggml_repeat_4d(ctx0, ggml_view_2d(ctx0, cur, 1, cur->ne[1], cur->nb[1], 0), n_left, cur->ne[1], 1, 1);
        ggml_tensor * is_first = ggml_step(ctx0, ggml_scale_bias(ctx0, pos_in, -1.0f, 0.5f));
        seanet.state_in["down"] = ggml_add(ctx0, seanet.state_in.at("down"), ggml_mul(ctx0, first, is_first));
    }
    cur = seanet.conv1d(cur, model.downsample_w, nullptr, hparams.mimi_downsample, 1, false, "down");
    cb(cur, "mimi_downsample", -1);

    seanet.state_outputs(gf);

    // split RVQ: the semantic codebook, then the residual acoustic ones, all from the same latent
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // [512, n_frames]
    const auto &  c2w      = model.c2w;
    const int64_t n_frames = cur->ne[1];
    const int64_t n_q      = model.mm_a_code_embd->ne[2];
    const int64_t n_embd   = model.mm_a_code_embd->ne[0];

    const std::vector<ggml_tensor *> codes = mtmd_graph_qwen3tts_gen::rvq_encode(ctx0,
        build_mm(c2w.quant_first_in_w, cur), c2w.quant_first_cb_w, build_mm(c2w.quant_rest_in_w, cur), c2w.quant_rest_cb_w);
    GGML_ASSERT((int64_t) codes.size() == n_q);

    // [n_embd, n_frames, n_q] -> [n_embd, n_q, n_frames]
    ggml_tensor * embd = nullptr;
    for (int64_t k = 0; k < n_q; k++) {
        cb(codes[k], "in_code", (int) k);
        ggml_tensor * e = ggml_get_rows(ctx0, moshi_slice(ctx0, model.mm_a_code_embd, k), codes[k]);
        e = ggml_reshape_3d(ctx0, e, n_embd, n_frames, 1);
        embd = embd ? ggml_concat(ctx0, embd, e, 2) : e;
    }
    embd = ggml_cont(ctx0, ggml_permute(ctx0, embd, 0, 2, 1, 3));

    // the initial code is the last entry of every table
    const float   code_initial = (float) (model.mm_a_code_embd->ne[1] - 1);
    ggml_tensor * idx_initial  = ggml_cast(ctx0, ggml_arange(ctx0, code_initial, code_initial + 1.0f, 1.0f), GGML_TYPE_I32);
    ggml_tensor * init = nullptr;
    for (int64_t k = 0; k < n_q; k++) {
        ggml_tensor * e = ggml_get_rows(ctx0, moshi_slice(ctx0, model.mm_a_code_embd, k), idx_initial);
        init = init ? ggml_concat(ctx0, init, e, 1) : e;
    }
    embd = ggml_concat(ctx0, embd, ggml_reshape_3d(ctx0, init, n_embd, n_q, 1), 2);

    embd = ggml_reshape_2d(ctx0, embd, n_embd, n_q * (n_frames + 1));
    cb(embd, "moshi_in_embd", -1);
    ggml_build_forward_expand(gf, embd);

    return gf;
}

// one depformer step: input row x [n_dep, 1], the K/V of the steps so far grow by one row per layer
ggml_tensor * mtmd_graph_moshi_gen::dep_step(ggml_tensor * x, int64_t iw, std::vector<ggml_tensor *> & k_all, std::vector<ggml_tensor *> & v_all) {
    const int64_t n_dep  = x->ne[0];
    const int64_t d_head = n_dep / n_head;

    for (size_t il = 0; il < model.layers.size(); il++) {
        const auto & layer = model.layers[il];
        ggml_tensor * inp = x;

        ggml_tensor * cur = ggml_mul(ctx0, ggml_rms_norm(ctx0, x, eps), layer.ln_1_w);
        ggml_tensor * q = ggml_mul_mat(ctx0, moshi_slice(ctx0, layer.q_w, iw), cur);
        ggml_tensor * k = ggml_mul_mat(ctx0, moshi_slice(ctx0, layer.k_w, iw), cur);
        ggml_tensor * v = ggml_mul_mat(ctx0, moshi_slice(ctx0, layer.v_w, iw), cur);

        // no positional embedding, causal over the steps of this frame
        k_all[il] = k_all[il] ? ggml_concat(ctx0, k_all[il], k, 1) : k;
        v_all[il] = v_all[il] ? ggml_concat(ctx0, v_all[il], v, 1) : v;
        const int64_t n_kv = k_all[il]->ne[1];

        cur = build_attn(nullptr, nullptr,
                         ggml_reshape_4d(ctx0, q,         d_head, n_head, 1,    1),
                         ggml_reshape_4d(ctx0, k_all[il], d_head, n_head, n_kv, 1),
                         ggml_reshape_4d(ctx0, v_all[il], d_head, n_head, n_kv, 1),
                         nullptr, 1.0f / sqrtf((float) d_head), (int) il);
        x = ggml_add(ctx0, inp, ggml_mul_mat(ctx0, moshi_slice(ctx0, layer.o_w, iw), cur));

        cur = ggml_mul(ctx0, ggml_rms_norm(ctx0, x, eps), layer.ln_2_w);
        cur = ggml_swiglu_split(ctx0, ggml_mul_mat(ctx0, moshi_slice(ctx0, layer.ff_gate_w, iw), cur),
                                      ggml_mul_mat(ctx0, moshi_slice(ctx0, layer.ff_up_w,   iw), cur));
        x = ggml_add(ctx0, x, ggml_mul_mat(ctx0, moshi_slice(ctx0, layer.ff_down_w, iw), cur));
    }
    return x;
}

// backbone input of the next frame: sum of the embeddings of the generated codes
ggml_tensor * mtmd_graph_moshi_gen::feedback(const std::vector<ggml_tensor *> & codes) {
    ggml_tensor * fb = nullptr;
    for (size_t k = 0; k < codes.size(); k++) {
        ggml_tensor * e = ggml_get_rows(ctx0, moshi_slice(ctx0, model.gen_code_out_embd_w, (int64_t) k), codes[k]);
        fb = fb ? ggml_add(ctx0, fb, e) : e;
    }
    return fb;
}

ggml_cgraph * mtmd_graph_moshi_gen::build() {
    const int64_t n_q = model.gen_code_head_w->ne[2];

    if (gen_process == MTMD_SUBCTX_GEN_PROCESS_GEN_WAV) {
        // codebook-major: row k holds codebook k of every frame
        ggml_tensor * inp_codes = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_frames, n_q);
        ggml_set_name(inp_codes, "inp_codes");
        ggml_set_input(inp_codes);

        auto codes_of = [&](int64_t k) {
            return ggml_view_1d(ctx0, inp_codes, n_frames, (size_t) k * inp_codes->nb[1]);
        };

        // RVQ decode: output_proj(sum of the codebook entries) of the semantic and acoustic parts
        const auto & c2w = model.c2w;
        ggml_tensor * rest = nullptr;
        for (int64_t k = 1; k < n_q; k++) {
            ggml_tensor * e = ggml_get_rows(ctx0, moshi_slice(ctx0, c2w.quant_rest_cb_w, k - 1), codes_of(k));
            rest = rest ? ggml_add(ctx0, rest, e) : e;
        }
        ggml_tensor * cur = build_mm(c2w.quant_first_out_w, ggml_get_rows(ctx0, c2w.quant_first_cb_w, codes_of(0)));
        cur = ggml_add(ctx0, cur, build_mm(c2w.quant_rest_out_w, rest));
        cb(cur, "rvq_decode", -1);

        mtmd_graph_pockettts_seanet seanet(*this);
        seanet.state_inputs(list_pockettts_state_slots(hparams, model));

        cur = seanet.mimi_decode(cur, mimi_n_head, mimi_eps);

        seanet.state_outputs(gf);

        cur = ggml_reshape_1d(ctx0, cur, cur->ne[0]);
        ggml_set_name(cur, "out_audio");
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);
        return gf;
    }

    // GEN_CODE with all codes known: only their embedding
    if (fb_only) {
        ggml_tensor * inp_codes = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_q);
        ggml_set_name(inp_codes, "inp_codes");
        ggml_set_input(inp_codes);

        std::vector<ggml_tensor *> codes;
        for (int64_t k = 0; k < n_q; k++) {
            codes.push_back(ggml_view_1d(ctx0, inp_codes, 1, (size_t) k * sizeof(int32_t)));
        }
        ggml_tensor * fb = feedback(codes);
        cb(fb, "gen_audio_out", -1);
        ggml_build_forward_expand(gf, fb);
        return gf;
    }

    // the backbone hidden state arrives as the single batch entry
    ggml_tensor * h = ggml_reshape_2d(ctx0, build_inp_raw(1), n_mmproj_embd, 1);

    ggml_tensor * inp_code0 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1); // the sampled text token
    ggml_set_name(inp_code0, "inp_code0");
    ggml_set_input(inp_code0);

    ggml_tensor * inp_rand = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_q);
    ggml_set_name(inp_rand, "inp_rand");
    ggml_set_input(inp_rand);

    mtmd_graph_qwen3tts_gen::code_gen cg(*this, top_k, 1.0f);

    // steps past the last weight set reuse it
    const int64_t n_w = model.gen_code_proj_in_w->ne[2];

    std::vector<ggml_tensor *> k_all(model.layers.size(), nullptr);
    std::vector<ggml_tensor *> v_all(model.layers.size(), nullptr);
    std::vector<ggml_tensor *> codes;
    ggml_tensor * out_codes = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_q);

    for (int64_t k = 0; k < n_q; k++) {
        const int64_t iw = std::min(k, n_w - 1);

        // the previous token: the text token, then the codes of this frame
        ggml_tensor * prev = k == 0 ? ggml_get_rows(ctx0, model.gen_code_tok_embd_w, inp_code0)
                                    : ggml_get_rows(ctx0, moshi_slice(ctx0, model.gen_code_embd_w, k - 1), codes.back());
        ggml_tensor * x = ggml_add(ctx0, ggml_mul_mat(ctx0, moshi_slice(ctx0, model.gen_code_proj_in_w, iw), h), prev);

        x = dep_step(x, iw, k_all, v_all);

        ggml_tensor * logits = ggml_mul_mat(ctx0, moshi_slice(ctx0, model.gen_code_head_w, k), x);
        cb(logits, "dep_logits", (int) k);

        ggml_tensor * code;
        if (temp <= 0.0f) {
            code = ggml_argmax(ctx0, logits);
        } else {
            if (temp != 1.0f) {
                logits = ggml_scale(ctx0, logits, 1.0f / temp);
            }
            code = cg.do_sampling(logits, ggml_view_1d(ctx0, inp_rand, 1, (size_t) k * sizeof(float)));
        }
        codes.push_back(code);
        out_codes = cg.cache_set(out_codes, (int) k, code);
    }

    out_codes = ggml_cont(ctx0, out_codes);
    ggml_set_name(out_codes, "out_codes");
    ggml_set_output(out_codes);
    ggml_build_forward_expand(gf, out_codes);

    ggml_tensor * fb = feedback(codes);
    cb(fb, "gen_audio_out", -1);
    ggml_build_forward_expand(gf, fb);

    return gf;
}
