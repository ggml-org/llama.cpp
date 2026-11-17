#include "models.h"

#include <cmath>
#include <vector>

// one frame: backbone hidden + semantic code -> the codes of this frame + next backbone input
// fast AR row 0 is the backbone hidden, row p > 0 is the embedding of code p - 1 and picks code p (greedy, as the reference)
ggml_cgraph * mtmd_graph_fish_s2_gen::build_fast_ar() {
    const int64_t n_cb      = model.gen_code_embd_w->ne[2];
    const int64_t n_embd_cb = model.gen_code_embd_w->ne[0];
    const int64_t n_embd    = model.gen_code_tok_embd_w->ne[0];
    const int64_t n_kv      = model.layers[0].k_w->ne[1];
    const int     n_layer   = (int) model.layers.size();

    ggml_tensor * h = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_mmproj_embd);
    ggml_set_name(h, "inp_raw");
    ggml_set_input(h);

    ggml_tensor * inp_code0 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(inp_code0, "inp_code0");
    ggml_set_input(inp_code0);

    mtmd_graph_qwen3tts_gen::code_gen cg(*this, top_k, top_p);

    std::vector<ggml_tensor *> k_cache(n_layer), v_cache(n_layer);
    for (int il = 0; il < n_layer; il++) {
        k_cache[il] = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv, n_cb), 0.0f);
        v_cache[il] = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv, n_cb), 0.0f);
    }

    ggml_tensor * out_codes = cg.cache_set(ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_cb), 0, inp_code0);

    for (int p = 0; p < n_cb; p++) {
        ggml_tensor * cur = h;
        if (p > 0) {
            ggml_tensor * code = ggml_view_1d(ctx0, out_codes, 1, (size_t) (p - 1) * out_codes->nb[1]);
            cur = ggml_reshape_1d(ctx0, ggml_get_rows(ctx0, model.gen_code_tok_embd_w, code), n_embd);
        }
        ggml_tensor * kq_mask = cg.causal_mask_row(n_cb, p);
        ggml_tensor * inp_pos = cg.const_i32(k_cache[0], (float) p);
        for (int il = 0; il < n_layer; il++) {
            cur = cg.layer_forward(cur, model.layers[il], inp_pos, kq_mask, k_cache[il], v_cache[il], n_cb, p, il);
        }
        if (p == 0) {
            continue; // only fills the KV
        }
        cur = ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, hparams.eps), model.gen_code_norm_w);
        ggml_tensor * logits = ggml_mul_mat(ctx0, model.gen_code_head_w, cur);
        out_codes = cg.cache_set(out_codes, p, ggml_argmax(ctx0, logits));
    }

    out_codes = ggml_cont(ctx0, out_codes);
    ggml_set_name(out_codes, "out_codes");
    ggml_set_output(out_codes);
    ggml_build_forward_expand(gf, out_codes);

    // backbone input of the next row: (E_lm[semantic] + sum of E_cb[g][code g]) / sqrt(n_cb + 1)
    ggml_tensor * out_embd = ggml_get_rows(ctx0, model.gen_code_out_embd_w, inp_code0);
    for (int64_t g = 0; g < n_cb; g++) {
        ggml_tensor * table = ggml_view_2d(ctx0, model.gen_code_embd_w, n_embd_cb, model.gen_code_embd_w->ne[1],
                                           model.gen_code_embd_w->nb[1], (size_t) g * model.gen_code_embd_w->nb[2]);
        ggml_tensor * code  = ggml_view_1d(ctx0, out_codes, 1, (size_t) g * out_codes->nb[1]);
        out_embd = ggml_add(ctx0, out_embd, ggml_get_rows(ctx0, table, code));
    }
    out_embd = ggml_scale(ctx0, out_embd, 1.0f / sqrtf((float) (n_cb + 1)));
    cb(out_embd, "gen_audio_out", -1);
    ggml_build_forward_expand(gf, out_embd);

    return gf;
}

ggml_cgraph * mtmd_graph_fish_s2_gen::build() {
    return gen_process == MTMD_SUBCTX_GEN_PROCESS_GEN_WAV ? build_codec() : build_fast_ar();
}
