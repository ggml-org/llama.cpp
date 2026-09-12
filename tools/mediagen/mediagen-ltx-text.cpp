#include "mediagen-ltx-graph.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

//
// text encoder: capture all hidden states of the llama model
// depends on the llama graph tensor names "inp_scaled" and "l_out-N", encode() fails if they change
//

std::vector<llama_token> ltx_tokenize(const llama_vocab * vocab, const std::string & text, bool parse_special) {
    std::vector<llama_token> tokens(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), tokens.data(), (int) tokens.size(), true, parse_special);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), (int) text.size(), tokens.data(), (int) tokens.size(), true, parse_special);
    }
    tokens.resize(std::max(n, 0));
    return tokens;
}

struct ltx_capture {
    int n_layers  = 0;
    int n_tokens  = 0;
    int n_hidden  = 0;
    std::vector<std::vector<float>> states; // [n_layers + 1] x [n_tokens * n_hidden]
    std::vector<bool> got;

    int index_of(const char * name) const {
        if (strcmp(name, "inp_scaled") == 0) {
            return 0;
        }
        if (strncmp(name, "l_out-", 6) == 0) {
            const int il = atoi(name + 6);
            // the last state is the final norm output, read from the embeddings
            return il + 1 < n_layers ? il + 1 : -1;
        }
        return -1;
    }
};

static bool ltx_cb_eval(ggml_tensor * t, bool ask, void * user_data) {
    ltx_capture * cap = (ltx_capture *) ((ltx_text_encoder *) user_data)->cur_capture;
    if (!cap) {
        return false;
    }
    const int idx = cap->index_of(t->name);
    if (idx < 0) {
        return false;
    }
    if (ask) {
        return true;
    }
    if (t->type != GGML_TYPE_F32 || t->ne[1] != cap->n_tokens) {
        MG_WRN("%s: unexpected tensor %s type=%s ne=[%lld, %lld]\n", __func__, t->name, ggml_type_name(t->type), (long long) t->ne[0], (long long) t->ne[1]);
        return true;
    }
    if (cap->n_hidden == 0) {
        cap->n_hidden = (int) t->ne[0];
    }
    auto & dst = cap->states[idx];
    dst.resize((size_t) cap->n_tokens * cap->n_hidden);
    if (ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, dst.data(), 0, ggml_nbytes(t));
    } else {
        for (int64_t i = 0; i < t->ne[1]; i++) {
            ggml_backend_tensor_get(t, dst.data() + i * t->ne[0], i * t->nb[1], t->ne[0] * sizeof(float));
        }
    }
    cap->got[idx] = true;
    return true;
}

bool ltx_text_encoder::init(llama_model * model_, int n_ctx_, int n_threads, bool flash_attn) {
    model = model_;
    n_ctx = n_ctx_;
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = n_ctx;
    cparams.n_batch    = n_ctx;
    cparams.n_ubatch   = n_ctx;
    cparams.n_seq_max  = 1;
    cparams.n_threads  = n_threads;
    cparams.n_threads_batch = n_threads;
    // embeddings mode, every token an output: final norm on all tokens, no lm_head
    cparams.embeddings   = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cparams.no_perf      = true;
    cparams.flash_attn_type = flash_attn ? LLAMA_FLASH_ATTN_TYPE_AUTO : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.cb_eval = ltx_cb_eval;
    cparams.cb_eval_user_data = this;
    lctx = llama_init_from_model(model, cparams);
    if (!lctx) {
        MG_ERR("%s: failed to create text encoder context\n", __func__);
        return false;
    }
    return true;
}

void ltx_text_encoder::free() {
    if (lctx) {
        llama_free(lctx);
        lctx = nullptr;
    }
}

bool ltx_text_encoder::encode(const std::string & prompt, int max_tokens, int & n_tokens, std::vector<float> & packed, int & n_hidden, int & n_states) {
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // strip like the reference pipeline
    size_t b = prompt.find_first_not_of(" \t\r\n");
    size_t e = prompt.find_last_not_of(" \t\r\n");
    std::string text = b == std::string::npos ? "" : prompt.substr(b, e - b + 1);

    std::vector<llama_token> tokens = ltx_tokenize(vocab, text, false);
    int n = (int) tokens.size();
    if (n == 0) {
        MG_ERR("%s: failed to tokenize prompt\n", __func__);
        return false;
    }
    if (n > max_tokens) {
        MG_WRN("%s: prompt has %d tokens, truncating to %d\n", __func__, n, max_tokens);
        n = max_tokens;
        tokens.resize(n);
    }
    n_tokens = n;

    ltx_capture cap;
    cap.n_layers = llama_model_n_layer(model);
    cap.n_tokens = n;
    cap.states.resize(cap.n_layers + 1);
    cap.got.assign(cap.n_layers + 1, false);

    llama_memory_clear(llama_get_memory(lctx), true);
    cur_capture = &cap;

    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i]    = tokens[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = true;
    }
    batch.n_tokens = n;

    const int ret = llama_decode(lctx, batch);
    llama_batch_free(batch);
    cur_capture = nullptr;
    if (ret != 0) {
        MG_ERR("%s: llama_decode failed with %d\n", __func__, ret);
        return false;
    }
    // last hidden state: the final norm output, i.e. the per-token embeddings
    {
        const int H = cap.n_hidden > 0 ? cap.n_hidden : llama_model_n_embd(model);
        auto & dst = cap.states[cap.n_layers];
        dst.resize((size_t) n * H);
        for (int i = 0; i < n; i++) {
            const float * e = llama_get_embeddings_ith(lctx, i);
            if (!e) {
                MG_ERR("%s: no embeddings for token %d\n", __func__, i);
                return false;
            }
            memcpy(dst.data() + (size_t) i * H, e, H * sizeof(float));
        }
        cap.got[cap.n_layers] = true;
        cap.n_hidden = H;
    }
    for (int i = 0; i <= cap.n_layers; i++) {
        if (!cap.got[i]) {
            MG_ERR("%s: hidden state %d was not captured\n", __func__, i);
            return false;
        }
    }

    n_hidden = cap.n_hidden;
    n_states = cap.n_layers + 1;

    // per-token, per-state rms norm, packed as [t][h * L + l]
    const int H = n_hidden;
    const int L = n_states;
    packed.resize((size_t) n * H * L);
    for (int t = 0; t < n; t++) {
        float * dst = packed.data() + (size_t) t * H * L;
        for (int l = 0; l < L; l++) {
            const float * src = cap.states[l].data() + (size_t) t * H;
            double ss = 0.0;
            for (int h = 0; h < H; h++) {
                ss += (double) src[h] * src[h];
            }
            const float r = (float) (1.0 / std::sqrt(ss / H + 1e-6));
            for (int h = 0; h < H; h++) {
                dst[(size_t) h * L + l] = src[h] * r;
            }
        }
    }
    return true;
}

//
// text projection + connectors
//

static ggml_tensor * ltx_connector(ggml_context * ctx, const ltx_model & model, const std::string & prefix, ggml_tensor * x,
                                   ggml_tensor * reg_idx, const ltx_rope_in & pe, int n_heads, bool flash_attn) {
    const auto & hp = model.hp;
    const auto & w  = model.dit;
    const int64_t n_real = x->ne[1];
    const int64_t n_tot  = hp.text_max_tokens;

    if (n_real < n_tot) {
        ggml_tensor * regs = ggml_get_rows(ctx, w.get(prefix + ".learnable_registers"), reg_idx); // [D, n_tot - n_real]
        x = ggml_concat(ctx, x, regs, 1);
    }

    for (int il = 0; il < hp.connector_layers; il++) {
        const std::string bp = mg_format("%s.transformer_1d_blocks.%d", prefix.c_str(), il);
        ltx_attn_weights aw = ltx_attn_weights::load(w, bp + ".attn1");
        ggml_tensor * h = ltx_rms(ctx, x, 1e-6f);
        ggml_tensor * a = ltx_attention(ctx, aw, h, nullptr, pe.cos_t, pe.sin_t, pe.cos_t, pe.sin_t, n_heads, flash_attn);
        x = ggml_add(ctx, x, a);
        h = ltx_rms(ctx, x, 1e-6f);
        x = ggml_add(ctx, x, ltx_ff(ctx, w, bp + ".ff", h));
    }
    return ltx_rms(ctx, x, 1e-6f);
}

bool ltx_build_text_cond(const ltx_model & model, mg_backend & be, const std::vector<float> & packed, int n_tokens, int n_hidden, int n_states,
                         bool flash_attn, ltx_text_cond & out) {
    const auto & hp = model.hp;
    const int n_tot = hp.text_max_tokens;
    const int n_in  = n_hidden * n_states;
    if (n_tokens > n_tot) {
        MG_ERR("%s: %d tokens exceed the maximum of %d\n", __func__, n_tokens, n_tot);
        return false;
    }

    mg_graph g(be);
    ggml_context * ctx = g.get();

    ggml_tensor * inp = g.new_input_f32({ n_in, n_tokens }, packed, "text_hidden");

    // learnable registers fill the padded tail
    std::vector<int32_t> reg_idx;
    for (int p = n_tokens; p < n_tot; p++) {
        reg_idx.push_back(p % hp.connector_regs);
    }
    ggml_tensor * t_reg = nullptr;
    if (!reg_idx.empty()) {
        t_reg = g.new_input(GGML_TYPE_I32, { (int64_t) reg_idx.size() }, reg_idx.data(), "reg_idx");
    }

    // 1-D rope
    std::vector<double> pos(n_tot);
    for (int p = 0; p < n_tot; p++) {
        pos[p] = (double) p / hp.connector_max_pos;
    }
    const ltx_rope_in v_pe = ltx_make_rope(g, pos, n_tot, 1, hp.v_dim, hp.rope_theta, "v_pe");
    const ltx_rope_in a_pe = ltx_make_rope(g, pos, n_tot, 1, hp.a_dim, hp.rope_theta, "a_pe");

    // per-modality projections
    ggml_tensor * v_in = ggml_scale(ctx, inp, std::sqrt((float) hp.v_dim / hp.text_hidden_size));
    ggml_tensor * a_in = ggml_scale(ctx, inp, std::sqrt((float) hp.a_dim / hp.text_hidden_size));
    ggml_tensor * v = mg_linear(ctx, v_in, model.tp("text_embedding_projection.video_aggregate_embed.weight"),
                                          model.tp("text_embedding_projection.video_aggregate_embed.bias", false));
    ggml_tensor * a = mg_linear(ctx, a_in, model.tp("text_embedding_projection.audio_aggregate_embed.weight"),
                                          model.tp("text_embedding_projection.audio_aggregate_embed.bias", false));

    v = ltx_connector(ctx, model, "video_embeddings_connector", v, t_reg, v_pe, hp.v_heads, flash_attn);
    a = ltx_connector(ctx, model, "audio_embeddings_connector", a, t_reg, a_pe, hp.a_heads, flash_attn);

    g.mark_output(v);
    g.mark_output(a);
    if (!g.compute(be)) {
        return false;
    }
    out.n_tokens = n_tot;
    g.get_output(v, out.video);
    g.get_output(a, out.audio);
    return true;
}
