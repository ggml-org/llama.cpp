#include "models.h"

#include "../clip/clip-model.h"
#include "../clip/clip-gen-memory.h"

#include "llama-impl.h"
#include "llama-memory-slots.h"

#include <functional>
#include <string>
#include <utility>

// Qwen3-TTS code2wav: RVQ codes -> PCM, streaming
// the decoder keeps, per sequence: a position counter, a (window-1)-frame k/v prefix per transformer layer,
// the left-context of every causal conv and the overlap tail of every transposed conv (see c2w_slots())

static constexpr int C2W_DAC_DILATIONS[3] = { 1, 3, 9 };

void llama_model_clip_load_c2w(llama_model_clip & m, const std::function<ggml_tensor *(const LLM_TN_IMPL &, int)> & create, int not_required) {
    const LLM_TN tn(LLM_ARCH_CLIP);
    auto & c = m.c2w;

    c.quant_first_out = create(tn(LLM_TENSOR_A_GEN_WAV_QUANT_FIRST_OUT, "weight"), not_required);
    if (c.quant_first_out == nullptr) {
        return; // no codec in this file
    }
    c.quant_first_cb  = create(tn(LLM_TENSOR_A_GEN_WAV_QUANT_FIRST_CB,  "weight"), 0);
    c.quant_rest_out  = create(tn(LLM_TENSOR_A_GEN_WAV_QUANT_REST_OUT,  "weight"), 0);
    c.quant_rest_cb   = create(tn(LLM_TENSOR_A_GEN_WAV_QUANT_REST_CB,   "weight"), 0);
    c.pre_conv_w      = create(tn(LLM_TENSOR_A_GEN_WAV_PRE_CONV, "weight"), 0);
    c.pre_conv_b      = create(tn(LLM_TENSOR_A_GEN_WAV_PRE_CONV, "bias"),   0);
    c.tfm_in_w        = create(tn(LLM_TENSOR_A_GEN_WAV_TFM_IN,   "weight"), 0);
    c.tfm_in_b        = create(tn(LLM_TENSOR_A_GEN_WAV_TFM_IN,   "bias"),   0);
    c.tfm_out_w       = create(tn(LLM_TENSOR_A_GEN_WAV_TFM_OUT,  "weight"), 0);
    c.tfm_out_b       = create(tn(LLM_TENSOR_A_GEN_WAV_TFM_OUT,  "bias"),   0);
    c.tfm_norm        = create(tn(LLM_TENSOR_A_GEN_WAV_TFM_NORM, "weight"), 0);

    for (int il = 0; ; ++il) {
        llama_model_clip::c2w_tfm_layer l;
        l.ln1 = create(tn(LLM_TENSOR_A_GEN_WAV_TFM_LN1, "weight", il), not_required);
        if (l.ln1 == nullptr) {
            break;
        }
        l.q    = create(tn(LLM_TENSOR_A_GEN_WAV_TFM_Q,    "weight", il), 0);
        l.k    = create(tn(LLM_TENSOR_A_GEN_WAV_TFM_K,    "weight", il), 0);
        l.v    = create(tn(LLM_TENSOR_A_GEN_WAV_TFM_V,    "weight", il), 0);
        l.o    = create(tn(LLM_TENSOR_A_GEN_WAV_TFM_O,    "weight", il), 0);
        l.ls1  = create(tn(LLM_TENSOR_A_GEN_WAV_TFM_LS1,  "weight", il), not_required);
        l.ln2  = create(tn(LLM_TENSOR_A_GEN_WAV_TFM_LN2,  "weight", il), 0);
        l.gate = create(tn(LLM_TENSOR_A_GEN_WAV_TFM_GATE, "weight", il), 0);
        l.up   = create(tn(LLM_TENSOR_A_GEN_WAV_TFM_UP,   "weight", il), 0);
        l.down = create(tn(LLM_TENSOR_A_GEN_WAV_TFM_DOWN, "weight", il), 0);
        l.ls2  = create(tn(LLM_TENSOR_A_GEN_WAV_TFM_LS2,  "weight", il), not_required);
        c.tfm.push_back(l);
    }

    for (int il = 0; ; ++il) {
        llama_model_clip::c2w_up_block b;
        b.conv_w = create(tn(LLM_TENSOR_A_GEN_WAV_UP_CONV, "weight", il), not_required);
        if (b.conv_w == nullptr) {
            break;
        }
        b.conv_b = create(tn(LLM_TENSOR_A_GEN_WAV_UP_CONV,   "bias",   il), 0);
        b.dw_w   = create(tn(LLM_TENSOR_A_GEN_WAV_UP_DWCONV, "weight", il), 0);
        b.dw_b   = create(tn(LLM_TENSOR_A_GEN_WAV_UP_DWCONV, "bias",   il), 0);
        b.norm_w = create(tn(LLM_TENSOR_A_GEN_WAV_UP_NORM,   "weight", il), 0);
        b.norm_b = create(tn(LLM_TENSOR_A_GEN_WAV_UP_NORM,   "bias",   il), 0);
        b.pw1_w  = create(tn(LLM_TENSOR_A_GEN_WAV_UP_PW1,    "weight", il), 0);
        b.pw1_b  = create(tn(LLM_TENSOR_A_GEN_WAV_UP_PW1,    "bias",   il), 0);
        b.pw2_w  = create(tn(LLM_TENSOR_A_GEN_WAV_UP_PW2,    "weight", il), 0);
        b.pw2_b  = create(tn(LLM_TENSOR_A_GEN_WAV_UP_PW2,    "bias",   il), 0);
        b.gamma  = create(tn(LLM_TENSOR_A_GEN_WAV_UP_GAMMA,  nullptr,  il), 0);
        c.up.push_back(b);
    }

    c.dac_entry_w = create(tn(LLM_TENSOR_A_GEN_WAV_DAC_ENTRY, "weight"), 0);
    c.dac_entry_b = create(tn(LLM_TENSOR_A_GEN_WAV_DAC_ENTRY, "bias"),   0);

    for (int ib = 0; ; ++ib) {
        llama_model_clip::c2w_dac_block b;
        b.snake_a = create(tn(LLM_TENSOR_A_GEN_WAV_DAC_SNAKE, "alpha", ib), not_required);
        if (b.snake_a == nullptr) {
            break;
        }
        b.snake_b = create(tn(LLM_TENSOR_A_GEN_WAV_DAC_SNAKE, "beta",   ib), 0);
        b.conv_w  = create(tn(LLM_TENSOR_A_GEN_WAV_DAC_CONV,  "weight", ib), 0);
        b.conv_b  = create(tn(LLM_TENSOR_A_GEN_WAV_DAC_CONV,  "bias",   ib), 0);
        for (int ir = 0; ir < 3; ++ir) {
            llama_model_clip::c2w_dac_res r;
            r.act1_a  = create(tn(LLM_TENSOR_A_GEN_WAV_DAC_RES_ACT1,  "alpha",  ib, ir), 0);
            r.act1_b  = create(tn(LLM_TENSOR_A_GEN_WAV_DAC_RES_ACT1,  "beta",   ib, ir), 0);
            r.conv1_w = create(tn(LLM_TENSOR_A_GEN_WAV_DAC_RES_CONV1, "weight", ib, ir), 0);
            r.conv1_b = create(tn(LLM_TENSOR_A_GEN_WAV_DAC_RES_CONV1, "bias",   ib, ir), 0);
            r.act2_a  = create(tn(LLM_TENSOR_A_GEN_WAV_DAC_RES_ACT2,  "alpha",  ib, ir), 0);
            r.act2_b  = create(tn(LLM_TENSOR_A_GEN_WAV_DAC_RES_ACT2,  "beta",   ib, ir), 0);
            r.conv2_w = create(tn(LLM_TENSOR_A_GEN_WAV_DAC_RES_CONV2, "weight", ib, ir), 0);
            r.conv2_b = create(tn(LLM_TENSOR_A_GEN_WAV_DAC_RES_CONV2, "bias",   ib, ir), 0);
            b.res.push_back(r);
        }
        c.dac.push_back(b);
    }

    c.post_snake_a = create(tn(LLM_TENSOR_A_GEN_WAV_DAC_POST_SNAKE, "alpha"),  0);
    c.post_snake_b = create(tn(LLM_TENSOR_A_GEN_WAV_DAC_POST_SNAKE, "beta"),   0);
    c.post_conv_w  = create(tn(LLM_TENSOR_A_GEN_WAV_DAC_POST_CONV,  "weight"), 0);
    c.post_conv_b  = create(tn(LLM_TENSOR_A_GEN_WAV_DAC_POST_CONV,  "bias"),   0);
}

std::vector<llama_model_clip::c2w_slot> llama_model_clip::c2w_slots() const {
    std::vector<c2w_slot> slots;
    if (c2w.dac_entry_w == nullptr) {
        return slots;
    }

    slots.push_back({ "tfm_pos", 1, 1 });

    const int64_t d_head = c2w.tfm[0].q->ne[1] / c2w.n_head;
    const int64_t kv_ch  = d_head * c2w.n_head_kv;
    const int64_t prefix = c2w.window - 1;
    for (size_t il = 0; il < c2w.tfm.size(); il++) {
        slots.push_back({ "tfm_k_" + std::to_string(il), kv_ch, prefix });
        slots.push_back({ "tfm_v_" + std::to_string(il), kv_ch, prefix });
    }

    slots.push_back({ "pre_conv", c2w.pre_conv_w->ne[0] - 1, c2w.pre_conv_w->ne[1] });

    for (size_t il = 0; il < c2w.up.size(); il++) {
        slots.push_back({ "up" + std::to_string(il) + "_dwconv", c2w.up[il].dw_w->ne[0] - 1, c2w.up[il].dw_w->ne[2] });
    }

    slots.push_back({ "dac_entry", c2w.dac_entry_w->ne[0] - 1, c2w.dac_entry_w->ne[1] });

    for (size_t il = 0; il < c2w.dac.size(); il++) {
        const auto & blk = c2w.dac[il];
        const int64_t stride = blk.conv_w->ne[0] / 2; // kernel == 2*stride
        const std::string name = "dac" + std::to_string(il);
        slots.push_back({ name + "_tail", stride, blk.conv_w->ne[1] });
        for (size_t ir = 0; ir < blk.res.size(); ir++) {
            slots.push_back({ name + "_res" + std::to_string(ir), (blk.res[ir].conv1_w->ne[0] - 1) * C2W_DAC_DILATIONS[ir], blk.res[ir].conv1_w->ne[1] });
        }
    }

    slots.push_back({ "dac_post_conv", c2w.post_conv_w->ne[0] - 1, c2w.post_conv_w->ne[1] });

    return slots;
}

bool llama_model_clip::c2w_codes_valid(const int32_t * codes, int32_t n_codes) const {
    if (!c2w.quant_first_cb || !c2w.quant_rest_cb || n_codes != c2w.quant_rest_cb->ne[2] + 1) {
        return false;
    }
    for (int32_t g = 0; g < n_codes; ++g) {
        const int64_t size = g == 0 ? c2w.quant_first_cb->ne[1] : c2w.quant_rest_cb->ne[1];
        if (codes[g] < 0 || codes[g] >= size) {
            return false;
        }
    }
    return true;
}

int64_t llama_model_clip::c2w_samples_per_frame() const {
    int64_t n = 1;
    for (const auto & up : c2w.up) {
        n *= up.conv_w->ne[0]; // kernel == stride
    }
    for (const auto & blk : c2w.dac) {
        n *= blk.conv_w->ne[0] / 2;
    }
    return n;
}

llama_memory_i * llama_model_clip::create_clip_memory(const llama_memory_params & params, const llama_cparams & cparams) const {
    GGML_UNUSED(params);
    const bool gen = cparams.ctx_type == LLAMA_CONTEXT_TYPE_GEN_CODE || cparams.ctx_type == LLAMA_CONTEXT_TYPE_GEN_WAV;
    if (gen && !native_gen && engine_gen() != nullptr) {
        // the gen pipelines of the engine keep their per-sequence state on the host side of the context
        return new llama_memory_clip_gen();
    }
    if (cparams.ctx_type != LLAMA_CONTEXT_TYPE_GEN_WAV) {
        return nullptr;
    }
    std::vector<llama_memory_slots::slot_desc> descs;
    for (const auto & s : c2w_slots()) {
        descs.push_back({ s.name, s.ne0, s.ne1 });
    }
    return new llama_memory_slots(*this, descs, cparams.n_seq_max, cparams.offload_kqv);
}

uint32_t llama_model_clip::ctx_n_embd_out(enum llama_context_type type) const {
    if (type == LLAMA_CONTEXT_TYPE_GEN_WAV && native_gen) {
        return (uint32_t) c2w_samples_per_frame();
    }
    if ((type == LLAMA_CONTEXT_TYPE_GEN_CODE || type == LLAMA_CONTEXT_TYPE_GEN_WAV) && !native_gen && engine_gen()) {
        return (uint32_t) engine_gen()->n_mmproj_embd(); // the feedback row of a gen step
    }
    if (type == LLAMA_CONTEXT_TYPE_ENCODER && !native_v) {
        if (const clip_model * m = engine_for_ctx(type)) {
            return (uint32_t) m->n_mmproj_embd();
        }
    }
    return hparams.n_embd_out();
}

//
// graph
//

class llm_graph_input_codes : public llm_graph_input_i {
public:
    void set_input(const llama_ubatch * ubatch) override {
        GGML_ASSERT(ubatch->codes && ubatch->n_codes == (uint32_t) codes->ne[1]);
        const int64_t n_frames = codes->ne[0];
        const int64_t n_codes  = codes->ne[1];
        std::vector<int32_t> data(n_frames * n_codes);
        for (int64_t f = 0; f < n_frames; f++) {
            for (int64_t g = 0; g < n_codes; g++) {
                data[g * n_frames + f] = ubatch->codes[f * n_codes + g];
            }
        }
        ggml_backend_tensor_set(codes, data.data(), 0, ggml_nbytes(codes));
    }

    ggml_tensor * codes = nullptr; // I32 [n_frames, n_codes]
};

struct llama_model_clip::graph_qwen3tts_wav : public llm_graph_context {
    const llama_model_clip & clip;
    const llama_memory_slots * mem = nullptr;
    llama_seq_id seq = 0;

    // new slot values, written after every read of the old ones (see the end of the constructor)
    std::vector<std::pair<std::string, ggml_tensor *>> pending;

    graph_qwen3tts_wav(const llama_model_clip & clip, const llm_graph_params & params) : llm_graph_context(params), clip(clip) {
        mem = static_cast<const llama_memory_slots_context *>(mctx)->get_mem();
        seq = ubatch.n_tokens > 0 && ubatch.seq_id ? ubatch.seq_id[0][0] : 0;

        // reservation: a full window
        const int64_t n_frames = ubatch.codes ? (int64_t) ubatch.n_tokens : clip.c2w.window;
        const int64_t n_codes  = clip.c2w.quant_rest_cb->ne[2] + 1;

        auto inp = std::make_unique<llm_graph_input_codes>();
        inp->codes = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_frames, n_codes);
        ggml_set_input(inp->codes);
        ggml_tensor * codes = inp->codes;
        res->add_input(std::move(inp));

        ggml_tensor * audio = decode(codes);                                   // [n_frames * spf]
        audio = ggml_reshape_2d(ctx0, audio, audio->ne[0] / n_frames, n_frames); // one row per frame

        cb(audio, "result_embd", -1);
        ggml_set_output(audio);
        res->t_embd = audio;
        ggml_build_forward_expand(gf, audio);

        // every slot read is an ancestor of the output, so the writes go last
        for (const auto & [name, value] : pending) {
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, value, mem->view(ctx0, name, seq)));
        }
    }

    ggml_tensor * state_in(const std::string & name) const {
        return mem->view(ctx0, name, seq);
    }

    void state_out(const std::string & name, ggml_tensor * value) {
        pending.emplace_back(name, value);
    }

    // causal conv1d with persisted left-context, x: [T, IC], w: [K, IC, OC] -> [T, OC]
    ggml_tensor * causal_conv1d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int dilation, const std::string & slot) {
        const int pad = ((int) w->ne[0] - 1) * dilation;
        ggml_tensor * x_full = x;
        if (pad > 0) {
            x_full = ggml_concat(ctx0, state_in(slot), x, 0);
        }
        ggml_tensor * y = ggml_conv_1d(ctx0, w, x_full, 1, 0, dilation);
        y = ggml_reshape_2d(ctx0, y, y->ne[0], y->ne[1]);
        if (b) {
            y = ggml_add(ctx0, y, ggml_reshape_2d(ctx0, b, 1, b->ne[0]));
        }
        if (pad > 0) {
            state_out(slot, ggml_cont(ctx0, ggml_view_2d(ctx0, x_full, pad, x_full->ne[1], x_full->nb[1], (size_t) (x_full->ne[0] - pad) * x_full->nb[0])));
        }
        return y;
    }

    // causal depthwise conv1d, x: [T, C], w: [K, 1, C] -> [T, C]
    ggml_tensor * causal_conv1d_dw(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, const std::string & slot) {
        const int pad = (int) w->ne[0] - 1;
        ggml_tensor * x_full = x;
        if (pad > 0) {
            x_full = ggml_concat(ctx0, state_in(slot), x, 0);
        }
        ggml_tensor * y = ggml_conv_1d_dw(ctx0, w, x_full, 1, 0, 1);
        y = ggml_reshape_2d(ctx0, y, y->ne[0], y->ne[1]);
        if (b) {
            y = ggml_add(ctx0, y, ggml_reshape_2d(ctx0, b, 1, b->ne[0]));
        }
        if (pad > 0) {
            state_out(slot, ggml_cont(ctx0, ggml_view_2d(ctx0, x_full, pad, x_full->ne[1], x_full->nb[1], (size_t) (x_full->ne[0] - pad) * x_full->nb[0])));
        }
        return y;
    }

    // causal ConvTranspose1d with the overlap tail persisted, x: [T, IC], w: [K, OC, IC] -> [T * stride, OC]
    ggml_tensor * causal_conv_transpose1d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int stride, const std::string & slot) {
        const int     K        = (int) w->ne[0];
        const int     OC       = (int) w->ne[1];
        const int     trim     = K - stride;
        const int64_t emit_len = x->ne[0] * stride;

        ggml_tensor * w2  = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, w, (int64_t) K * OC, w->ne[2])));
        ggml_tensor * xt  = ggml_cont(ctx0, ggml_transpose(ctx0, x));
        ggml_tensor * col = ggml_mul_mat(ctx0, w2, xt);
        ggml_tensor * y   = ggml_col2im_1d(ctx0, col, stride, OC, 0);

        ggml_tensor * out = y;
        if (trim > 0) {
            ggml_tensor * head = ggml_add(ctx0, ggml_view_2d(ctx0, y, trim, y->ne[1], y->nb[1], 0), state_in(slot));
            if (emit_len > trim) {
                ggml_tensor * middle = ggml_view_2d(ctx0, y, emit_len - trim, y->ne[1], y->nb[1], (size_t) trim * y->nb[0]);
                out = ggml_concat(ctx0, head, middle, 0);
            } else {
                out = head;
            }
            state_out(slot, ggml_cont(ctx0, ggml_view_2d(ctx0, y, trim, y->ne[1], y->nb[1], (size_t) emit_len * y->nb[0])));
        }
        if (b) {
            out = ggml_add(ctx0, out, ggml_reshape_2d(ctx0, b, 1, b->ne[0]));
        }
        return out;
    }

    // SnakeBeta: x + sin(alpha*x)^2 * inv_beta, x: [T, C]
    ggml_tensor * snake(ggml_tensor * x, ggml_tensor * alpha, ggml_tensor * beta) {
        ggml_tensor * a = ggml_reshape_2d(ctx0, alpha, 1, alpha->ne[0]);
        ggml_tensor * b = ggml_reshape_2d(ctx0, beta,  1, beta->ne[0]);
        ggml_build_forward_expand(gf, a);
        ggml_build_forward_expand(gf, b);
        ggml_tensor * s = ggml_sqr(ctx0, ggml_sin(ctx0, ggml_mul(ctx0, x, a)));
        return ggml_add(ctx0, x, ggml_mul(ctx0, s, b));
    }

    // RVQ decode: [T, n_codes] codes -> [512, T]
    ggml_tensor * quant_decode(ggml_tensor * codes) {
        const auto & c = clip.c2w;
        const int64_t T = codes->ne[0];
        auto group = [&](int g) { return ggml_view_1d(ctx0, codes, T, (size_t) g * codes->nb[1]); };

        ggml_tensor * sem_out = ggml_mul_mat(ctx0, c.quant_first_out, ggml_get_rows(ctx0, c.quant_first_cb, group(0)));

        ggml_tensor * acc = nullptr;
        for (int g = 1; g <= c.quant_rest_cb->ne[2]; g++) {
            ggml_tensor * cb_g = ggml_view_2d(ctx0, c.quant_rest_cb, c.quant_rest_cb->ne[0], c.quant_rest_cb->ne[1],
                                              c.quant_rest_cb->nb[1], (size_t) (g - 1) * c.quant_rest_cb->nb[2]);
            ggml_tensor * e = ggml_get_rows(ctx0, cb_g, group(g));
            acc = acc ? ggml_add(ctx0, acc, e) : e;
        }
        return ggml_add(ctx0, sem_out, ggml_mul_mat(ctx0, c.quant_rest_out, acc));
    }

    // one transformer layer over N new frames, attending to the (W-1)-frame prefix of the previous call
    ggml_tensor * tfm_layer(ggml_tensor * cur, int il) {
        const auto & l = clip.c2w.tfm[il];
        const int     nh       = clip.c2w.n_head;
        const int     nh_kv    = clip.c2w.n_head_kv;
        const int64_t d_head   = l.q->ne[1] / nh;
        const float   kq_scale = 1.0f / sqrtf((float) d_head);
        const int64_t W        = clip.c2w.window;
        const int64_t N        = cur->ne[1];
        const int64_t prefix   = W - 1;
        const int64_t total_kv = prefix + N;

        ggml_tensor * residual = cur;
        ggml_tensor * h = ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, clip.c2w.eps), l.ln1);

        ggml_tensor * q = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, l.q, h), d_head, nh,    N);
        ggml_tensor * k = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, l.k, h), d_head, nh_kv, N);
        ggml_tensor * v = ggml_mul_mat(ctx0, l.v, h);

        ggml_tensor * pos_in = state_in("tfm_pos");
        ggml_tensor * base   = ggml_reshape_1d(ctx0, pos_in, 1);
        ggml_tensor * pos    = ggml_cast(ctx0, ggml_add(ctx0, ggml_arange(ctx0, 0.0f, (float) N, 1.0f), base), GGML_TYPE_I32);

        q = ggml_rope_ext(ctx0, q, pos, nullptr, (int) d_head, GGML_ROPE_TYPE_NEOX, 0, clip.c2w.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx0, k, pos, nullptr, (int) d_head, GGML_ROPE_TYPE_NEOX, 0, clip.c2w.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        ggml_tensor * k_full = ggml_concat(ctx0, state_in("tfm_k_" + std::to_string(il)), ggml_reshape_2d(ctx0, k, d_head * nh_kv, N), 1);
        ggml_tensor * v_full = ggml_concat(ctx0, state_in("tfm_v_" + std::to_string(il)), ggml_reshape_2d(ctx0, v, d_head * nh_kv, N), 1);

        // banded causal mask: key j is visible to query i iff 0 <= (prefix+i) - j < W; on a cold start only real keys
        ggml_tensor * pos_k = ggml_reshape_2d(ctx0, ggml_arange(ctx0, 0.0f, (float) total_kv, 1.0f), total_kv, 1);
        ggml_tensor * pos_q = ggml_reshape_2d(ctx0, ggml_arange(ctx0, (float) prefix, (float) (prefix + N), 1.0f), 1, N);
        ggml_tensor * diff  = ggml_sub(ctx0, ggml_repeat_4d(ctx0, pos_q, total_kv, N, 1, 1), pos_k);
        ggml_tensor * keep  = ggml_mul(ctx0,
                ggml_step(ctx0, ggml_scale_bias(ctx0, diff,  1.0f, 0.5f)),
                ggml_step(ctx0, ggml_scale_bias(ctx0, diff, -1.0f, (float) W - 0.5f)));
        keep = ggml_mul(ctx0, keep, ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_add(ctx0, pos_k, base), 1.0f, 0.5f - (float) prefix)));
        ggml_tensor * mask = ggml_log(ctx0, keep);
        if (cparams.flash_attn) {
            mask = ggml_cast(ctx0, mask, GGML_TYPE_F16);
        }

        ggml_tensor * kc = ggml_reshape_3d(ctx0, k_full, d_head, nh_kv, total_kv);
        ggml_tensor * vc = ggml_reshape_3d(ctx0, v_full, d_head, nh_kv, total_kv);
        // the codec layers are not layers of the main tower, the fused-op probe needs a layer it knows
        ggml_tensor * attn = build_attn_mha(q, kc, vc, nullptr, mask, nullptr, nullptr, nullptr, 0, kq_scale, (int) n_layer - 1);
        attn = ggml_mul_mat(ctx0, l.o, attn);
        if (l.ls1) {
            attn = ggml_mul(ctx0, attn, l.ls1);
        }

        // the next call's prefix: the last W-1 frames, and the position counter (once)
        state_out("tfm_k_" + std::to_string(il), ggml_cont(ctx0, ggml_view_2d(ctx0, k_full, k_full->ne[0], prefix, k_full->nb[1], (size_t) N * k_full->nb[1])));
        state_out("tfm_v_" + std::to_string(il), ggml_cont(ctx0, ggml_view_2d(ctx0, v_full, v_full->ne[0], prefix, v_full->nb[1], (size_t) N * v_full->nb[1])));
        if (il == (int) clip.c2w.tfm.size() - 1) {
            state_out("tfm_pos", ggml_scale_bias(ctx0, pos_in, 1.0f, (float) N));
        }

        cur = ggml_add(ctx0, residual, attn);

        ggml_tensor * h2   = ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, clip.c2w.eps), l.ln2);
        ggml_tensor * down = ggml_mul_mat(ctx0, l.down, ggml_swiglu_split(ctx0, ggml_mul_mat(ctx0, l.gate, h2), ggml_mul_mat(ctx0, l.up, h2)));
        if (l.ls2) {
            down = ggml_mul(ctx0, down, l.ls2);
        }
        return ggml_add(ctx0, cur, down);
    }

    // dwconv -> LayerNorm -> pw1 -> GELU -> pw2 -> scale -> residual, x: [T, C]
    ggml_tensor * convnext(ggml_tensor * x, const c2w_up_block & b, const std::string & slot) {
        ggml_tensor * h  = causal_conv1d_dw(x, b.dw_w, b.dw_b, slot);
        ggml_tensor * hc = ggml_cont(ctx0, ggml_transpose(ctx0, h));
        hc = ggml_add(ctx0, ggml_mul(ctx0, ggml_norm(ctx0, hc, 1e-6f), b.norm_w), b.norm_b);
        ggml_tensor * g = ggml_add(ctx0, ggml_mul_mat(ctx0, b.pw1_w, hc), b.pw1_b);
        g = ggml_gelu(ctx0, g);
        g = ggml_add(ctx0, ggml_mul_mat(ctx0, b.pw2_w, g), b.pw2_b);
        g = ggml_mul(ctx0, g, b.gamma);
        return ggml_add(ctx0, x, ggml_cont(ctx0, ggml_transpose(ctx0, g)));
    }

    ggml_tensor * decode(ggml_tensor * codes) {
        const auto & c = clip.c2w;

        ggml_tensor * x = ggml_cont(ctx0, ggml_transpose(ctx0, quant_decode(codes)));     // [N, 512]
        x = causal_conv1d(x, c.pre_conv_w, c.pre_conv_b, 1, "pre_conv");                 // [N, 1024]

        ggml_tensor * cur = ggml_cont(ctx0, ggml_transpose(ctx0, x));
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, c.tfm_in_w, cur), c.tfm_in_b);
        for (size_t il = 0; il < c.tfm.size(); il++) {
            cur = tfm_layer(cur, (int) il);
        }
        cur = ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, c.eps), c.tfm_norm);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, c.tfm_out_w, cur), c.tfm_out_b);           // [1024, N]

        x = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        for (size_t il = 0; il < c.up.size(); il++) {
            x = causal_conv_transpose1d(x, c.up[il].conv_w, c.up[il].conv_b, 2, "");
            x = convnext(x, c.up[il], "up" + std::to_string(il) + "_dwconv");
        }

        x = causal_conv1d(x, c.dac_entry_w, c.dac_entry_b, 1, "dac_entry");
        for (size_t il = 0; il < c.dac.size(); il++) {
            const auto & blk = c.dac[il];
            const int stride = (int) (blk.conv_w->ne[0] / 2);
            const std::string name = "dac" + std::to_string(il);
            x = snake(x, blk.snake_a, blk.snake_b);
            x = causal_conv_transpose1d(x, blk.conv_w, blk.conv_b, stride, name + "_tail");
            for (size_t ir = 0; ir < blk.res.size(); ir++) {
                const auto & r = blk.res[ir];
                ggml_tensor * h = snake(x, r.act1_a, r.act1_b);
                h = causal_conv1d(h, r.conv1_w, r.conv1_b, C2W_DAC_DILATIONS[ir], name + "_res" + std::to_string(ir));
                h = snake(h, r.act2_a, r.act2_b);
                h = causal_conv1d(h, r.conv2_w, r.conv2_b, 1, "");
                x = ggml_add(ctx0, x, h);
            }
        }

        x = snake(x, c.post_snake_a, c.post_snake_b);
        x = causal_conv1d(x, c.post_conv_w, c.post_conv_b, 1, "dac_post_conv");
        x = ggml_clamp(ctx0, x, -1.0f, 1.0f);
        return ggml_reshape_1d(ctx0, x, x->ne[0]);
    }
};

std::unique_ptr<llm_graph_context> llama_model_clip_build_wav_graph(const llama_model_clip & clip, const llm_graph_params & params) {
    return std::make_unique<llama_model_clip::graph_qwen3tts_wav>(clip, params);
}
