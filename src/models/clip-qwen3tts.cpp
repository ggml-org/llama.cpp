// Qwen3-TTS code predictor and codec: the native graphs of a gen step, on the blocks and the weights of the engine
// the gen runtime of the engine sets their inputs and reads their outputs by name (clip_subcontext::set_inputs(),
// read_outputs()), as for the other gen towers: the RNG draws, the codec state slots and the windows come from there

#include "models.h"

#include "../clip/clip-model.h"
#include "../clip/clip-subcontext.h"
#include "../clip/models/models.h"

#include <string>

ggml_cgraph * llama_model_clip_build_qwen3tts_gen(ggml_context * ctx0, ggml_cgraph * gf, const clip_model & model, bool flash_attn,
        const clip_subctx_encode_params & req) {
    const auto & hparams = model.hparams;

    clip_tts_blocks b(ctx0, gf, model, flash_attn);

    const int64_t n_acoustic = model.gen_code_head_w->ne[2]; // 15
    const int     n_codes    = (int) n_acoustic + 1;         // 16

    if (req.gen_process == CLIP_SUBCTX_GEN_PROCESS_GEN_WAV) {
        // one window of codes, group-major and rear-padded -> PCM; the codec state of the sequence in and out
        ggml_tensor * inp_codes = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, hparams.wav_tfm_swa, n_codes);
        ggml_set_name(inp_codes, "inp_codes");
        ggml_set_input(inp_codes);

        for (const auto & slot : list_c2w_state_slots(hparams, model)) {
            ggml_tensor * t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, slot.ne0, slot.ne1);
            ggml_set_name(t, ("state_in_" + slot.name).c_str());
            ggml_set_input(t);
            b.state_in[slot.name] = t;
        }

        ggml_tensor * out_audio = b.decode(inp_codes);
        ggml_set_name(out_audio, "out_audio");
        ggml_set_output(out_audio);
        ggml_build_forward_expand(gf, out_audio);

        for (auto & slot : b.state_out) {
            ggml_set_name(slot.second, ("state_out_" + slot.first).c_str());
            ggml_set_output(slot.second);
            ggml_build_forward_expand(gf, slot.second);
        }
        return gf;
    }

    GGML_ASSERT(req.gen_process == CLIP_SUBCTX_GEN_PROCESS_GEN_CODE);

    // backbone hidden state + code0 -> 16 codes and the feedback embedding
    ggml_tensor * h_state = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, model.n_mmproj_embd());
    ggml_set_name(h_state, "inp_raw");
    ggml_set_input(h_state);

    ggml_tensor * code0 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(code0, "inp_code0");
    ggml_set_input(code0);

    ggml_tensor * code0_embd = ggml_get_rows(ctx0, model.gen_code_out_embd_w, code0);
    code0_embd = ggml_reshape_1d(ctx0, code0_embd, code0_embd->ne[0]);
    b.cb(code0_embd, "code0_embd", -1);

    const int64_t n_kv_pad  = n_codes;
    const int     n_layer   = (int) model.layers.size();
    const int64_t d_head    = model.layers[0].q_w->ne[1] / hparams.n_head;
    const int     n_head_kv = hparams.n_head_kv;

    // zero-filled per layer k/v caches, so masked-out rows can't hold garbage
    std::vector<ggml_tensor *> k_cache(n_layer), v_cache(n_layer);
    for (int il = 0; il < n_layer; il++) {
        k_cache[il] = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_head * n_head_kv, n_kv_pad), 0.0f);
        v_cache[il] = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_head * n_head_kv, n_kv_pad), 0.0f);
    }

    b.top_k = req.top_k;
    b.top_p = req.top_p;

    ggml_tensor * out_code_cache = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_codes);
    out_code_cache = b.cache_set(out_code_cache, 0, code0);

    // one uniform draw per codebook, from the RNG of the sequence
    auto inp_rand = [&](int g) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
        ggml_set_name(t, ("inp_rand_" + std::to_string(g)).c_str());
        ggml_set_input(t);
        return t;
    };

    b.prefill(k_cache, v_cache, out_code_cache, h_state, code0_embd, inp_rand(0));
    for (int g = 1; g < n_acoustic; g++) {
        out_code_cache = b.step(k_cache, v_cache, out_code_cache, inp_rand(g), g);
    }

    ggml_tensor * out_codes = ggml_cont(ctx0, out_code_cache);
    ggml_set_name(out_codes, "out_codes");
    ggml_set_output(out_codes);
    ggml_build_forward_expand(gf, out_codes);

    // sum of the 16 codebook rows, fed back to the backbone for the next frame
    ggml_tensor * out_embd = code0_embd;
    for (int g = 1; g <= n_acoustic; g++) {
        ggml_tensor * code_g = ggml_view_1d(ctx0, out_code_cache, 1, (size_t) g * out_code_cache->nb[1]);
        ggml_tensor * embd_g = ggml_view_2d(ctx0, model.gen_code_embd_w, model.gen_code_embd_w->ne[0], model.gen_code_embd_w->ne[1],
                                            model.gen_code_embd_w->nb[1], (size_t) (g - 1) * model.gen_code_embd_w->nb[2]);
        ggml_tensor * e = ggml_get_rows(ctx0, embd_g, code_g);
        out_embd = ggml_add(ctx0, out_embd, ggml_reshape_1d(ctx0, e, e->ne[0]));
    }
    out_embd = ggml_reshape_2d(ctx0, out_embd, out_embd->ne[0], 1);
    b.cb(out_embd, "gen_audio_out", -1);

    // the last node, the embedding output of the graph
    ggml_build_forward_expand(gf, out_embd);
    return gf;
}
