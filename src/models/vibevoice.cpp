#include "models.h"

#include "ggml-alloc.h"

#include <mutex>
#include <random>

struct llama_model_vibevoice::tok_tensors {
    llm_tensor conv;
    llm_tensor head;
    llm_tensor norm;
    llm_tensor dwconv;
    llm_tensor gamma;
    llm_tensor ffn_norm;
    llm_tensor ffn_up;
    llm_tensor ffn_down;
    llm_tensor ffn_gamma;
};

// there is one stage per depth, and one ratio per resample conv (all convs but the stem)
static uint32_t vv_get_stages(llama_model_loader & ml, llm_kv kid_ratios, llm_kv kid_depths,
        std::array<uint32_t, LLAMA_MAX_VV_STAGES> & ratios, std::array<uint32_t, LLAMA_MAX_VV_STAGES> & depths) {
    std::vector<uint32_t> r;
    std::vector<uint32_t> d;
    ml.get_arr(kid_ratios, r);
    ml.get_arr(kid_depths, d);

    if (d.empty() || d.size() > LLAMA_MAX_VV_STAGES || r.size() + 1 != d.size()) {
        throw std::runtime_error(format("vibevoice: invalid tokenizer layout, %zu ratios and %zu depths", r.size(), d.size()));
    }
    for (uint32_t depth : d) {
        if (depth == 0 || depth > LLAMA_MAX_LAYERS) {
            throw std::runtime_error(format("vibevoice: invalid tokenizer stage depth %u", depth));
        }
    }

    int64_t hop = 1;
    for (uint32_t ratio : r) {
        if (ratio == 0 || hop > INT64_MAX / ratio) {
            throw std::runtime_error(format("vibevoice: invalid tokenizer ratio %u, hop must fit in int64_t", ratio));
        }
        hop *= ratio;
    }

    ratios.fill(0);
    depths.fill(0);
    std::copy(r.begin(), r.end(), ratios.begin());
    std::copy(d.begin(), d.end(), depths.begin());

    return d.size();
}

void llama_model_vibevoice::load_arch_hparams(llama_model_loader & ml) {
    llama_model_qwen2::load_arch_hparams(ml);

    auto & vv = hparams.vibevoice;

    ml.get_key(LLM_KV_VIBEVOICE_SPEECH_SCALING_FACTOR, vv.speech_scaling_factor);
    ml.get_key(LLM_KV_VIBEVOICE_SPEECH_BIAS_FACTOR,    vv.speech_bias_factor);

    ml.get_key(LLM_KV_VIBEVOICE_ACOUSTIC_VAE_DIM,       vv.acoustic.vae_dim);
    ml.get_key(LLM_KV_VIBEVOICE_ACOUSTIC_N_FILTERS,     vv.acoustic.n_filters);
    ml.get_key(LLM_KV_VIBEVOICE_ACOUSTIC_FIX_STD,       vv.acoustic.fix_std);
    ml.get_key(LLM_KV_VIBEVOICE_ACOUSTIC_LAYERNORM_EPS, vv.acoustic.f_norm_rms_eps);
    vv.acoustic.n_enc_stages = vv_get_stages(ml, LLM_KV_VIBEVOICE_ACOUSTIC_ENCODER_RATIOS, LLM_KV_VIBEVOICE_ACOUSTIC_ENCODER_DEPTHS, vv.acoustic.enc_ratios, vv.acoustic.enc_depths);
    vv.acoustic.n_dec_stages = vv_get_stages(ml, LLM_KV_VIBEVOICE_ACOUSTIC_DECODER_RATIOS, LLM_KV_VIBEVOICE_ACOUSTIC_DECODER_DEPTHS, vv.acoustic.dec_ratios, vv.acoustic.dec_depths);

    ml.get_key(LLM_KV_VIBEVOICE_SEMANTIC_VAE_DIM,       vv.semantic.vae_dim);
    ml.get_key(LLM_KV_VIBEVOICE_SEMANTIC_N_FILTERS,     vv.semantic.n_filters);
    ml.get_key(LLM_KV_VIBEVOICE_SEMANTIC_LAYERNORM_EPS, vv.semantic.f_norm_rms_eps);
    vv.semantic.fix_std      = 0.0f;
    vv.semantic.n_enc_stages = vv_get_stages(ml, LLM_KV_VIBEVOICE_SEMANTIC_ENCODER_RATIOS, LLM_KV_VIBEVOICE_SEMANTIC_ENCODER_DEPTHS, vv.semantic.enc_ratios, vv.semantic.enc_depths);
    vv.semantic.n_dec_stages = 0;

    ml.get_key(LLM_KV_VIBEVOICE_DIFFUSION_BLOCK_COUNT,      vv.diff_n_layer);
    if (vv.diff_n_layer == 0 || vv.diff_n_layer > LLAMA_MAX_LAYERS) {
        throw std::runtime_error(format("vibevoice: invalid diffusion block count %u", vv.diff_n_layer));
    }
    ml.get_key(LLM_KV_VIBEVOICE_DIFFUSION_FEED_FORWARD_LEN, vv.diff_n_ff);
    ml.get_key(LLM_KV_VIBEVOICE_DIFFUSION_LATENT_SIZE,      vv.diff_latent_size);
    ml.get_key(LLM_KV_VIBEVOICE_DIFFUSION_LAYERNORM_EPS,    vv.diff_f_norm_rms_eps);
    ml.get_key(LLM_KV_VIBEVOICE_DIFFUSION_TRAIN_STEPS,      vv.diff_train_steps);
    ml.get_key(LLM_KV_VIBEVOICE_DIFFUSION_INFERENCE_STEPS,  vv.diff_inference_steps);
    if (vv.diff_inference_steps == 0 || vv.diff_inference_steps > vv.diff_train_steps) {
        throw std::runtime_error(format("vibevoice: invalid diffusion step counts: inference %u, training %u (require 0 < inference <= training)", vv.diff_inference_steps, vv.diff_train_steps));
    }

    // all released checkpoints use the same scheduler, it is the only one we will implement
    std::string diff_type;
    std::string diff_pred_type;
    std::string diff_beta_schedule;
    ml.get_key(LLM_KV_VIBEVOICE_DIFFUSION_TYPE,            diff_type);
    ml.get_key(LLM_KV_VIBEVOICE_DIFFUSION_PREDICTION_TYPE, diff_pred_type);
    ml.get_key(LLM_KV_VIBEVOICE_DIFFUSION_BETA_SCHEDULE,   diff_beta_schedule);
    if (diff_type != "ddpm" || diff_pred_type != "v_prediction" || diff_beta_schedule != "cosine") {
        throw std::runtime_error(format("vibevoice: unsupported diffusion head %s/%s/%s",
                diff_type.c_str(), diff_pred_type.c_str(), diff_beta_schedule.c_str()));
    }
}

void llama_model_vibevoice::load_connector(llama_model_loader & ml, connector & conn, llm_tensor fc1, llm_tensor norm, llm_tensor fc2, int64_t n_in) {
    const int64_t n_embd = hparams.n_embd;
    conn.fc1   = create_tensor(ml, tn(fc1,  "weight"), {n_in, n_embd}, 0);
    conn.fc1_b = create_tensor(ml, tn(fc1,  "bias"),   {n_embd}, 0);
    conn.norm  = create_tensor(ml, tn(norm, "weight"), {n_embd}, 0);
    conn.fc2   = create_tensor(ml, tn(fc2,  "weight"), {n_embd, n_embd}, 0);
    conn.fc2_b = create_tensor(ml, tn(fc2,  "bias"),   {n_embd}, 0);
}

void llama_model_vibevoice::load_tok_part(llama_model_loader & ml, tok_part & part, const tok_tensors & ids, const llama_hparams_vv_tokenizer & hp, bool decoder) {
    const uint32_t n_stages = decoder ? hp.n_dec_stages : hp.n_enc_stages;
    const auto & depths = decoder ? hp.dec_depths : hp.enc_depths;
    part.stages.resize(n_stages);

    for (uint32_t s = 0; s < n_stages; ++s) {
        auto & stage = part.stages[s];
        const int64_t c = (int64_t) hp.n_filters << (decoder ? n_stages - 1 - s : s);

        if (s == 0) {
            stage.conv = create_tensor(ml, tn(ids.conv, "weight", s), {7, decoder ? (int64_t) hp.vae_dim : 1, c}, 0);
        } else if (decoder) {
            stage.conv = create_tensor(ml, tn(ids.conv, "weight", s), {2LL * hp.dec_ratios[s - 1], c, 2*c}, 0);
        } else {
            stage.conv = create_tensor(ml, tn(ids.conv, "weight", s), {2LL * hp.enc_ratios[n_stages - 1 - s], c/2, c}, 0);
        }
        stage.conv_b = create_tensor(ml, tn(ids.conv, "bias", s), {c}, 0);

        stage.blocks.resize(depths[s]);
        for (uint32_t b = 0; b < depths[s]; ++b) {
            auto & blk = stage.blocks[b];

            blk.norm       = create_tensor(ml, tn(ids.norm,      "weight", s, b), {c}, 0);
            blk.dwconv     = create_tensor(ml, tn(ids.dwconv,    "weight", s, b), {7, 1, c}, 0);
            blk.dwconv_b   = create_tensor(ml, tn(ids.dwconv,    "bias",   s, b), {c}, 0);
            blk.gamma      = create_tensor(ml, tn(ids.gamma,     "weight", s, b), {c}, 0);
            blk.ffn_norm   = create_tensor(ml, tn(ids.ffn_norm,  "weight", s, b), {c}, 0);
            blk.ffn_up     = create_tensor(ml, tn(ids.ffn_up,    "weight", s, b), {c, 4*c}, 0);
            blk.ffn_up_b   = create_tensor(ml, tn(ids.ffn_up,    "bias",   s, b), {4*c}, 0);
            blk.ffn_down   = create_tensor(ml, tn(ids.ffn_down,  "weight", s, b), {4*c, c}, 0);
            blk.ffn_down_b = create_tensor(ml, tn(ids.ffn_down,  "bias",   s, b), {c}, 0);
            blk.ffn_gamma  = create_tensor(ml, tn(ids.ffn_gamma, "weight", s, b), {c}, 0);
        }
    }

    const int64_t n_in  = decoder ? hp.n_filters : (int64_t) hp.n_filters << (n_stages - 1);
    const int64_t n_out = decoder ? 1 : hp.vae_dim;
    part.head   = create_tensor(ml, tn(ids.head, "weight"), {7, n_in, n_out}, 0);
    part.head_b = create_tensor(ml, tn(ids.head, "bias"),   {n_out}, 0);
}

void llama_model_vibevoice::load_arch_tensors(llama_model_loader & ml) {
    llama_model_qwen2::load_arch_tensors(ml);

    const auto & vv = hparams.vibevoice;
    const int64_t n_embd = hparams.n_embd;
    const int64_t n_ff   = vv.diff_n_ff;
    const int64_t n_lat  = vv.diff_latent_size;

    load_connector(ml, acoustic_conn, LLM_TENSOR_VV_ACOUSTIC_CONN_FC1, LLM_TENSOR_VV_ACOUSTIC_CONN_NORM, LLM_TENSOR_VV_ACOUSTIC_CONN_FC2, vv.diff_latent_size);
    load_connector(ml, semantic_conn, LLM_TENSOR_VV_SEMANTIC_CONN_FC1, LLM_TENSOR_VV_SEMANTIC_CONN_NORM, LLM_TENSOR_VV_SEMANTIC_CONN_FC2, vv.semantic.vae_dim);

    // diffusion head
    {
        diff.noisy_proj = create_tensor(ml, tn(LLM_TENSOR_VV_DIFF_NOISY_PROJ, "weight"), {n_lat, n_embd}, 0);
        diff.cond_proj  = create_tensor(ml, tn(LLM_TENSOR_VV_DIFF_COND_PROJ,  "weight"), {n_embd, n_embd}, 0);
        diff.t_embd_fc1 = create_tensor(ml, tn(LLM_TENSOR_VV_DIFF_T_EMBD_FC1, "weight"), {256, n_embd}, 0);
        diff.t_embd_fc2 = create_tensor(ml, tn(LLM_TENSOR_VV_DIFF_T_EMBD_FC2, "weight"), {n_embd, n_embd}, 0);

        diff.layers.resize(vv.diff_n_layer);

        for (uint32_t i = 0; i < vv.diff_n_layer; ++i) {
            auto & layer = diff.layers[i];

            layer.norm     = create_tensor(ml, tn(LLM_TENSOR_VV_DIFF_NORM,     "weight", i), {n_embd}, 0);
            layer.adaln    = create_tensor(ml, tn(LLM_TENSOR_VV_DIFF_ADALN,    "weight", i), {n_embd, 3*n_embd}, 0);
            layer.ffn_gate = create_tensor(ml, tn(LLM_TENSOR_VV_DIFF_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_up   = create_tensor(ml, tn(LLM_TENSOR_VV_DIFF_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(ml, tn(LLM_TENSOR_VV_DIFF_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        }

        diff.final_adaln = create_tensor(ml, tn(LLM_TENSOR_VV_DIFF_FINAL_ADALN, "weight"), {n_embd, 2*n_embd}, 0);
        diff.final_proj  = create_tensor(ml, tn(LLM_TENSOR_VV_DIFF_FINAL_PROJ,  "weight"), {n_embd, n_lat}, 0);
    }

    // tokenizers
    {
        const tok_tensors acoustic_enc_ids = {
            LLM_TENSOR_VV_ACOUSTIC_ENC_CONV, LLM_TENSOR_VV_ACOUSTIC_ENC_HEAD,
            LLM_TENSOR_VV_ACOUSTIC_ENC_NORM, LLM_TENSOR_VV_ACOUSTIC_ENC_DWCONV, LLM_TENSOR_VV_ACOUSTIC_ENC_GAMMA,
            LLM_TENSOR_VV_ACOUSTIC_ENC_FFN_NORM, LLM_TENSOR_VV_ACOUSTIC_ENC_FFN_UP, LLM_TENSOR_VV_ACOUSTIC_ENC_FFN_DOWN, LLM_TENSOR_VV_ACOUSTIC_ENC_FFN_GAMMA,
        };
        const tok_tensors acoustic_dec_ids = {
            LLM_TENSOR_VV_ACOUSTIC_DEC_CONV, LLM_TENSOR_VV_ACOUSTIC_DEC_HEAD,
            LLM_TENSOR_VV_ACOUSTIC_DEC_NORM, LLM_TENSOR_VV_ACOUSTIC_DEC_DWCONV, LLM_TENSOR_VV_ACOUSTIC_DEC_GAMMA,
            LLM_TENSOR_VV_ACOUSTIC_DEC_FFN_NORM, LLM_TENSOR_VV_ACOUSTIC_DEC_FFN_UP, LLM_TENSOR_VV_ACOUSTIC_DEC_FFN_DOWN, LLM_TENSOR_VV_ACOUSTIC_DEC_FFN_GAMMA,
        };
        const tok_tensors semantic_enc_ids = {
            LLM_TENSOR_VV_SEMANTIC_ENC_CONV, LLM_TENSOR_VV_SEMANTIC_ENC_HEAD,
            LLM_TENSOR_VV_SEMANTIC_ENC_NORM, LLM_TENSOR_VV_SEMANTIC_ENC_DWCONV, LLM_TENSOR_VV_SEMANTIC_ENC_GAMMA,
            LLM_TENSOR_VV_SEMANTIC_ENC_FFN_NORM, LLM_TENSOR_VV_SEMANTIC_ENC_FFN_UP, LLM_TENSOR_VV_SEMANTIC_ENC_FFN_DOWN, LLM_TENSOR_VV_SEMANTIC_ENC_FFN_GAMMA,
        };

        load_tok_part(ml, acoustic_enc, acoustic_enc_ids, vv.acoustic, false);
        load_tok_part(ml, acoustic_dec, acoustic_dec_ids, vv.acoustic, true);
        load_tok_part(ml, semantic_enc, semantic_enc_ids, vv.semantic, false);
    }
}

//
// acoustic decoder
//

// left context of the causal convs when a sequence runs chunk by chunk (upstream VibeVoiceTokenizerStreamingCache)
// every conv takes its cache as a graph input and outputs the updated one, in graph order
struct llama_model_vibevoice::stream_io {
    std::vector<ggml_tensor *> inp;
    std::vector<ggml_tensor *> out;
};

struct llama_model_vibevoice::graph_cache {
    std::mutex mutex;
    ggml_context_ptr ctx;
    ggml_gallocr_ptr galloc;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * out = nullptr;
    std::vector<ggml_tensor *> inputs;
    stream_io st;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    int64_t n = 0;
    int64_t hop = 0;
    size_t n_nodes = 0;
    bool allocated = false;

    bool matches(ggml_backend_t backend_new, int64_t n_new, int64_t hop_new = 0) const {
        return allocated && backend == backend_new && buft == ggml_backend_get_default_buffer_type(backend_new) && n == n_new && hop == hop_new;
    }

    void init(ggml_backend_t backend_new, int64_t n_new, int64_t hop_new, size_t nodes) {
        allocated = false;
        galloc.reset();
        ctx.reset();
        gf = nullptr;
        out = nullptr;
        inputs.clear();
        st = {};
        backend = backend_new;
        buft = ggml_backend_get_default_buffer_type(backend);
        n = n_new;
        hop = hop_new;
        n_nodes = nodes;

        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * n_nodes + ggml_graph_overhead_custom(n_nodes, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx.reset(ggml_init(params));
        if (!ctx) {
            throw std::runtime_error("vibevoice: failed to create the speech graph context");
        }
    }

    void alloc(ggml_tensor * output) {
        out = output;
        ggml_set_output(out);
        gf = ggml_new_graph_custom(ctx.get(), n_nodes, false);
        ggml_build_forward_expand(gf, out);
        for (ggml_tensor * t : st.out) {
            ggml_build_forward_expand(gf, t);
        }

        galloc.reset(ggml_gallocr_new(buft));
        if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
            throw std::runtime_error("vibevoice: failed to allocate the speech graph");
        }
        allocated = true;
    }

    std::vector<float> run_stream(tok_stream_cache & cache, const std::vector<const float *> & data) const {
        if (cache.convs.empty()) {
            cache.convs.resize(st.inp.size());
            for (size_t i = 0; i < st.inp.size(); ++i) {
                cache.convs[i].assign(ggml_nelements(st.inp[i]), 0.0f);
            }
        }
        if (cache.convs.size() != st.inp.size()) {
            throw std::runtime_error("vibevoice: stream cache does not match the graph");
        }
        for (size_t i = 0; i < st.inp.size(); ++i) {
            if (cache.convs[i].size() != (size_t) ggml_nelements(st.inp[i])) {
                throw std::runtime_error("vibevoice: stream cache does not match the graph");
            }
        }

        // The allocator can reuse input buffers. Set all inputs before each compute.
        for (size_t i = 0; i < st.inp.size(); ++i) {
            ggml_backend_tensor_set(st.inp[i], cache.convs[i].data(), 0, ggml_nbytes(st.inp[i]));
        }
        GGML_ASSERT(data.size() == inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            ggml_backend_tensor_set(inputs[i], data[i], 0, ggml_nbytes(inputs[i]));
        }
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("vibevoice: failed to compute the streaming graph");
        }
        for (size_t i = 0; i < st.out.size(); ++i) {
            ggml_backend_tensor_get(st.out[i], cache.convs[i].data(), 0, ggml_nbytes(st.out[i]));
        }

        std::vector<float> res(ggml_nelements(out));
        ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
        return res;
    }
};

llama_model_vibevoice::llama_model_vibevoice(const llama_model_params & params) :
    llama_model_qwen2(params),
    diffusion_graph(new graph_cache),
    acoustic_dec_graph(new graph_cache),
    semantic_enc_graph(new graph_cache),
    feedback_graph(new graph_cache) {}

llama_model_vibevoice::~llama_model_vibevoice() = default;

// x: [T, C] -> [n_ctx + T, C], the left context is the stream cache, or zeros without a stream
// the updated cache is the last n_ctx rows of the result
static ggml_tensor * vv_left_context(ggml_context * ctx, ggml_tensor * x, int64_t n_ctx, llama_model_vibevoice::stream_io * st) {
    if (n_ctx == 0) {
        return x;
    }
    if (st == nullptr) {
        return ggml_pad_ext(ctx, x, n_ctx, 0, 0, 0, 0, 0, 0, 0);
    }

    ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ctx, x->ne[1]);
    ggml_set_input(cache);

    x = ggml_concat(ctx, cache, x, 0);

    ggml_tensor * next = ggml_cont(ctx, ggml_view_2d(ctx, x, n_ctx, x->ne[1], x->nb[1], (x->ne[0] - n_ctx) * x->nb[0]));
    ggml_set_output(next);

    st->inp.push_back(cache);
    st->out.push_back(next);

    return x;
}

// causal conv1d with optional stride, zero left padding. x: [T, IC], w: [K, IC, OC]
// padding_total = K - stride (upstream SConv1d); caller should pass a T that needs no extra right pad
// returns [ceil-ish T/stride, OC]; with T % stride == 0 and K == 2*stride this is exactly T/stride
// streaming: the padding is replaced by the last K - stride inputs, chunks must be multiples of stride
static ggml_tensor * vv_causal_conv1d(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, int stride = 1, llama_model_vibevoice::stream_io * st = nullptr) {
    GGML_ASSERT(stride >= 1 && w->ne[0] >= stride);
    x = vv_left_context(ctx, x, w->ne[0] - stride, st);
    ggml_tensor * y = ggml_conv_1d(ctx, w, x, stride, 0, 1);
    return ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
}

// depthwise variant of vv_causal_conv1d. x: [T, C], w: [K, 1, C], returns [T, C]
static ggml_tensor * vv_causal_conv1d_dw(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, llama_model_vibevoice::stream_io * st) {
    x = vv_left_context(ctx, x, w->ne[0] - 1, st);
    ggml_tensor * y = ggml_conv_1d_dw(ctx, w, x, 1, 0, 1);
    return ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
}

// causal ConvTranspose1d, the (K - stride) overlap is trimmed on the right. x: [T, IC], w: [K, OC, IC], returns [T * stride, OC]
// streaming: the last K - 1 inputs are prepended like upstream, only the outputs of the new inputs are returned
static ggml_tensor * vv_causal_conv_transpose1d(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, int stride, llama_model_vibevoice::stream_io * st) {
    GGML_ASSERT(w->ne[0] >= stride);
    const int64_t n_new = x->ne[0];
    const int64_t n_ctx = st ? w->ne[0] - 1 : 0;
    x = vv_left_context(ctx, x, n_ctx, st);
    ggml_tensor * y = ggml_conv_transpose_1d(ctx, w, x, stride, 0, 1);
    return ggml_view_2d(ctx, y, n_new * stride, y->ne[1], y->nb[1], n_ctx * stride * y->nb[0]);
}

// the decoder stream is [C, T], the convs run on the transposed [T, C]
static ggml_tensor * vv_transpose(ggml_context * ctx, ggml_tensor * x) {
    return ggml_cont(ctx, ggml_transpose(ctx, x));
}

// ConvNeXt-style Block1D: RMSNorm -> depthwise conv -> gamma -> residual, RMSNorm -> FFN -> ffn_gamma -> residual
static ggml_tensor * vv_build_tok_block(ggml_context * ctx, ggml_tensor * x, const llama_model_vibevoice::tok_block & blk, float eps,
        llama_model_vibevoice::stream_io * st) {
    ggml_tensor * cur = ggml_rms_norm(ctx, x, eps);
    cur = ggml_mul(ctx, cur, blk.norm);
    cur = vv_transpose(ctx, vv_causal_conv1d_dw(ctx, vv_transpose(ctx, cur), blk.dwconv, st));
    cur = ggml_add(ctx, cur, blk.dwconv_b);
    cur = ggml_mul(ctx, cur, blk.gamma);
    x = ggml_add(ctx, x, cur);

    cur = ggml_rms_norm(ctx, x, eps);
    cur = ggml_mul(ctx, cur, blk.ffn_norm);
    cur = ggml_mul_mat(ctx, blk.ffn_up, cur);
    cur = ggml_add(ctx, cur, blk.ffn_up_b);
    cur = ggml_gelu_erf(ctx, cur);
    cur = ggml_mul_mat(ctx, blk.ffn_down, cur);
    cur = ggml_add(ctx, cur, blk.ffn_down_b);
    cur = ggml_mul(ctx, cur, blk.ffn_gamma);

    return ggml_add(ctx, x, cur);
}

ggml_tensor * llama_model_vibevoice::build_acoustic_dec(ggml_context * ctx, ggml_tensor * latents) const {
    return build_acoustic_dec(ctx, latents, nullptr);
}

ggml_tensor * llama_model_vibevoice::build_acoustic_dec(ggml_context * ctx, ggml_tensor * latents, stream_io * st) const {
    const auto & hp = hparams.vibevoice.acoustic;

    GGML_ASSERT(ggml_is_matrix(latents) && latents->ne[0] == hp.vae_dim);

    ggml_tensor * cur = latents;

    for (uint32_t s = 0; s < hp.n_dec_stages; ++s) {
        const auto & stage = acoustic_dec.stages[s];

        // the first conv is the stem, the others upsample by the decoder ratios in config order
        ggml_tensor * x = vv_transpose(ctx, cur);
        if (s == 0) {
            x = vv_causal_conv1d(ctx, x, stage.conv, 1, st);
        } else {
            x = vv_causal_conv_transpose1d(ctx, x, stage.conv, hp.dec_ratios[s - 1], st);
        }
        cur = ggml_add(ctx, vv_transpose(ctx, x), stage.conv_b);

        for (const auto & blk : stage.blocks) {
            cur = vv_build_tok_block(ctx, cur, blk, hp.f_norm_rms_eps, st);
        }
    }

    // disable_last_norm: there is no norm before the head
    cur = vv_causal_conv1d(ctx, vv_transpose(ctx, cur), acoustic_dec.head, 1, st);
    cur = ggml_add(ctx, cur, ggml_reshape_2d(ctx, acoustic_dec.head_b, 1, acoustic_dec.head_b->ne[0]));

    return cur;
}

//
// tokenizer encoder (semantic / acoustic) + speech connector
//

// mono waveform [T, 1] -> latents [vae_dim, T / hop]
// encoder ratios are stored in HF config order; upstream TokenizerEncoder reverses them at build time
static ggml_tensor * vv_build_tok_enc(
        ggml_context * ctx,
        ggml_tensor * wav,
        const llama_model_vibevoice::tok_part & enc,
        const llama_hparams_vv_tokenizer & hp,
        llama_model_vibevoice::stream_io * st) {
    GGML_ASSERT(ggml_is_matrix(wav) && wav->ne[1] == 1);
    GGML_ASSERT(hp.n_enc_stages > 0);

    // channel-first stream like the decoder: [C, T]
    ggml_tensor * cur = vv_transpose(ctx, wav);

    const uint32_t n_ratios = hp.n_enc_stages - 1;

    for (uint32_t s = 0; s < hp.n_enc_stages; ++s) {
        const auto & stage = enc.stages[s];

        ggml_tensor * x = vv_transpose(ctx, cur);
        if (s == 0) {
            x = vv_causal_conv1d(ctx, x, stage.conv, 1, st);
        } else {
            // reversed config ratios: first downsample uses the last config ratio
            const uint32_t stride = hp.enc_ratios[n_ratios - s];
            x = vv_causal_conv1d(ctx, x, stage.conv, (int) stride, st);
        }
        cur = ggml_add(ctx, vv_transpose(ctx, x), stage.conv_b);

        for (const auto & blk : stage.blocks) {
            cur = vv_build_tok_block(ctx, cur, blk, hp.f_norm_rms_eps, st);
        }
    }

    // disable_last_norm: there is no norm before the head
    cur = vv_causal_conv1d(ctx, vv_transpose(ctx, cur), enc.head, 1, st);
    cur = ggml_add(ctx, cur, ggml_reshape_2d(ctx, enc.head_b, 1, enc.head_b->ne[0]));

    // [T', vae_dim] -> [vae_dim, T']
    return vv_transpose(ctx, cur);
}

ggml_tensor * llama_model_vibevoice::build_semantic_enc(ggml_context * ctx, ggml_tensor * wav) const {
    return build_semantic_enc(ctx, wav, nullptr);
}

ggml_tensor * llama_model_vibevoice::build_semantic_enc(ggml_context * ctx, ggml_tensor * wav, stream_io * st) const {
    return vv_build_tok_enc(ctx, wav, semantic_enc, hparams.vibevoice.semantic, st);
}

ggml_tensor * llama_model_vibevoice::build_acoustic_enc(ggml_context * ctx, ggml_tensor * wav) const {
    return vv_build_tok_enc(ctx, wav, acoustic_enc, hparams.vibevoice.acoustic, nullptr);
}

// speech connector: Linear -> RMSNorm -> Linear (no activation), matches upstream SpeechConnector
// features [in_dim, N] -> [n_embd, N]
ggml_tensor * llama_model_vibevoice::build_connector(ggml_context * ctx, ggml_tensor * features, const connector & conn) const {
    GGML_ASSERT(ggml_is_matrix(features));
    GGML_ASSERT(features->ne[0] == conn.fc1->ne[0]);

    ggml_tensor * cur = ggml_mul_mat(ctx, conn.fc1, features);
    cur = ggml_add(ctx, cur, conn.fc1_b);

    cur = ggml_rms_norm(ctx, cur, hparams.f_norm_rms_eps);
    cur = ggml_mul(ctx, cur, conn.norm);

    cur = ggml_mul_mat(ctx, conn.fc2, cur);
    cur = ggml_add(ctx, cur, conn.fc2_b);

    return cur;
}

//
// diffusion head
//

// x * (1 + scale) + shift
static ggml_tensor * vv_modulate(ggml_context * ctx, ggml_tensor * x, ggml_tensor * shift, ggml_tensor * scale) {
    return ggml_add(ctx, ggml_add(ctx, x, ggml_mul(ctx, x, scale)), shift);
}

// i-th of n equal chunks along the features, x: [n * C, N], returns [C, N]
static ggml_tensor * vv_chunk(ggml_context * ctx, ggml_tensor * x, int n, int i) {
    const int64_t c = x->ne[0] / n;
    return ggml_view_2d(ctx, x, c, x->ne[1], x->nb[1], i * c * x->nb[0]);
}

ggml_tensor * llama_model_vibevoice::build_diffusion_head(ggml_context * ctx, ggml_tensor * noisy, ggml_tensor * cond, ggml_tensor * t) const {
    const auto & vv = hparams.vibevoice;

    GGML_ASSERT(ggml_is_matrix(noisy) && noisy->ne[0] == vv.diff_latent_size);
    GGML_ASSERT(ggml_is_matrix(cond)  && cond->ne[1] == noisy->ne[1]);
    GGML_ASSERT(ggml_is_vector(t)     && (t->ne[0] == noisy->ne[1] || t->ne[0] == 1));

    // the frequency embedding size comes from fc1, it is 256 for all released checkpoints
    ggml_tensor * t_emb = ggml_timestep_embedding(ctx, t, diff.t_embd_fc1->ne[0], 10000);
    t_emb = ggml_mul_mat(ctx, diff.t_embd_fc1, t_emb);
    t_emb = ggml_silu(ctx, t_emb);
    t_emb = ggml_mul_mat(ctx, diff.t_embd_fc2, t_emb);

    // every adaLN modulation starts with a SiLU of the condition
    ggml_tensor * c = ggml_add(ctx, ggml_mul_mat(ctx, diff.cond_proj, cond), t_emb);
    c = ggml_silu(ctx, c);

    ggml_tensor * x = ggml_mul_mat(ctx, diff.noisy_proj, noisy);

    for (const auto & layer : diff.layers) {
        // shift, scale, gate
        ggml_tensor * mod = ggml_mul_mat(ctx, layer.adaln, c);

        ggml_tensor * cur = ggml_rms_norm(ctx, x, vv.diff_f_norm_rms_eps);
        cur = ggml_mul(ctx, cur, layer.norm);
        cur = vv_modulate(ctx, cur, vv_chunk(ctx, mod, 3, 0), vv_chunk(ctx, mod, 3, 1));

        cur = ggml_swiglu_split(ctx, ggml_mul_mat(ctx, layer.ffn_gate, cur), ggml_mul_mat(ctx, layer.ffn_up, cur));
        cur = ggml_mul_mat(ctx, layer.ffn_down, cur);

        x = ggml_add(ctx, x, ggml_mul(ctx, cur, vv_chunk(ctx, mod, 3, 2)));
    }

    // shift, scale, the final norm has no weight
    ggml_tensor * mod = ggml_mul_mat(ctx, diff.final_adaln, c);

    x = ggml_rms_norm(ctx, x, vv.diff_f_norm_rms_eps);
    x = vv_modulate(ctx, x, vv_chunk(ctx, mod, 2, 0), vv_chunk(ctx, mod, 2, 1));

    return ggml_mul_mat(ctx, diff.final_proj, x);
}

// the schedule follows the float32 math of the reference scheduler, its cumprod accumulates in double
llama_model_vibevoice::diff_sampler::diff_sampler(uint32_t n_train_steps, uint32_t n_steps) {
    GGML_ASSERT(n_steps > 0 && n_steps <= n_train_steps);

    // squaredcos_cap_v2 betas
    const auto alpha_bar = [](double t) {
        const double c = std::cos((t + 0.008) / 1.008 * std::acos(-1.0) / 2);
        return c * c;
    };

    std::vector<float> sigmas_train(n_train_steps);
    double alphas_cumprod = 1.0;
    for (uint32_t t = 0; t < n_train_steps; ++t) {
        const float beta = std::min(1.0 - alpha_bar((t + 1.0) / n_train_steps) / alpha_bar((double) t / n_train_steps), 0.999);
        alphas_cumprod *= 1.0f - beta;
        const float ac = alphas_cumprod;
        sigmas_train[t] = std::sqrt((1.0f - ac) / ac);
    }

    // linspace(0, n_train_steps - 1, n_steps + 1) rounded half to even like numpy, descending, without the 0
    const double dt = (double) (n_train_steps - 1) / n_steps;
    for (uint32_t k = n_steps; k > 0; --k) {
        const uint32_t t = k == n_steps ? n_train_steps - 1 : (uint32_t) std::nearbyint(k * dt);
        timesteps.push_back(t);
        sigmas.push_back(sigmas_train[t]);
    }
    sigmas.push_back(0.0f);
}

void llama_model_vibevoice::diff_sampler::step(const float * v, float * x, size_t n) {
    GGML_ASSERT(i < timesteps.size());
    GGML_ASSERT(x0_prev.empty() || x0_prev.size() == n);

    const auto alpha_sigma = [](double sigma) {
        const double alpha = 1.0 / std::sqrt(sigma * sigma + 1.0);
        return std::make_pair(alpha, sigma * alpha);
    };

    const auto [alpha_s0, sigma_s0] = alpha_sigma(sigmas[i]);
    const auto [alpha_t,  sigma_t]  = alpha_sigma(sigmas[i + 1]);

    // v_prediction -> data prediction
    std::vector<float> x0(n);
    for (size_t j = 0; j < n; ++j) {
        x0[j] = alpha_s0 * x[j] - sigma_s0 * v[j];
    }

    if (sigma_t == 0.0) {
        // the last step is first order, with a final sigma of 0 it lands on the data prediction
        std::copy(x0.begin(), x0.end(), x);
    } else {
        const double lambda_t  = std::log(alpha_t)  - std::log(sigma_t);
        const double lambda_s0 = std::log(alpha_s0) - std::log(sigma_s0);
        const double h = lambda_t - lambda_s0;

        const double c_x  = sigma_t / sigma_s0;
        const double c_x0 = alpha_t * (std::exp(-h) - 1.0);

        if (x0_prev.empty()) {
            // first order (DDIM) for the first step
            for (size_t j = 0; j < n; ++j) {
                x[j] = c_x * x[j] - c_x0 * x0[j];
            }
        } else {
            const auto [alpha_s1, sigma_s1] = alpha_sigma(sigmas[i - 1]);
            const double lambda_s1 = std::log(alpha_s1) - std::log(sigma_s1);
            const double r0 = (lambda_s0 - lambda_s1) / h;

            for (size_t j = 0; j < n; ++j) {
                const double d1 = (x0[j] - x0_prev[j]) / r0;
                x[j] = c_x * x[j] - c_x0 * x0[j] - 0.5 * c_x0 * d1;
            }
        }
    }

    x0_prev = std::move(x0);
    i++;
}

//
// generation wiring: CFG + speech feedback embeds
//

void llama_model_vibevoice::cfg_guide(const float * v_cond, const float * v_uncond, float * guided, size_t n, float scale) {
    for (size_t i = 0; i < n; ++i) {
        guided[i] = v_uncond[i] + scale * (v_cond[i] - v_uncond[i]);
    }
}

// upstream sample_speech_tokens: the cond and uncond halves get the same sample, the guided v drives both
std::vector<float> llama_model_vibevoice::sample_speech_latents(ggml_backend_t backend, const float * noise, const float * cond, const float * uncond,
        int64_t n_frames, float cfg_scale) const {
    const auto & vv = hparams.vibevoice;

    const int64_t n_cond = uncond ? 2 * n_frames : n_frames;
    const size_t  n      = (size_t) vv.diff_latent_size * n_frames;
    const size_t  n_embd = (size_t) hparams.n_embd * n_frames;

    auto & graph = *diffusion_graph;
    std::lock_guard<std::mutex> lock(graph.mutex);
    if (!graph.matches(backend, n_cond)) {
        graph.init(backend, n_cond, 0, GGML_DEFAULT_GRAPH_SIZE);
        ggml_context * ctx = graph.ctx.get();

        ggml_tensor * noisy = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vv.diff_latent_size, n_cond);
        ggml_set_input(noisy);
        ggml_tensor * c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hparams.n_embd, n_cond);
        ggml_set_input(c);
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_set_input(t);
        graph.inputs = { noisy, c, t };
        graph.alloc(build_diffusion_head(ctx, noisy, c, t));
    }
    ggml_tensor * noisy = graph.inputs[0];
    ggml_tensor * c     = graph.inputs[1];
    ggml_tensor * t     = graph.inputs[2];
    ggml_tensor * out   = graph.out;

    std::vector<float> x(noise, noise + n);
    std::vector<float> noisy_data(n * (uncond ? 2 : 1));
    std::vector<float> v(n * (uncond ? 2 : 1));
    std::vector<float> guided(uncond ? n : 0);

    if (ggml_nbytes(out) != v.size() * sizeof(float)) {
        throw std::runtime_error(format(
                "vibevoice: diffusion output size %zu does not match latent buffer %zu (diff_latent_size %u)",
                (size_t) ggml_nelements(out), v.size(), vv.diff_latent_size));
    }

    diff_sampler sampler(vv.diff_train_steps, vv.diff_inference_steps);

    for (float timestep : sampler.timesteps) {
        std::copy(x.begin(), x.end(), noisy_data.begin());
        if (uncond) {
            std::copy(x.begin(), x.end(), noisy_data.begin() + n);
        }

        // the allocator may reuse the input buffers, all inputs are set before each compute
        ggml_backend_tensor_set(noisy, noisy_data.data(), 0, ggml_nbytes(noisy));
        ggml_backend_tensor_set(c, cond, 0, n_embd * sizeof(float));
        if (uncond) {
            ggml_backend_tensor_set(c, uncond, n_embd * sizeof(float), n_embd * sizeof(float));
        }
        ggml_backend_tensor_set(t, &timestep, 0, sizeof(float));

        if (ggml_backend_graph_compute(backend, graph.gf) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("vibevoice: failed to compute the diffusion head graph");
        }
        ggml_backend_tensor_get(out, v.data(), 0, ggml_nbytes(out));

        if (uncond) {
            cfg_guide(v.data(), v.data() + n, guided.data(), n, cfg_scale);
            sampler.step(guided.data(), x.data(), n);
        } else {
            sampler.step(v.data(), x.data(), n);
        }
    }

    return x;
}

std::vector<float> llama_model_vibevoice::sample_speech_latents(ggml_backend_t backend, const float * cond, const float * uncond,
        int64_t n_frames, float cfg_scale, uint32_t seed) const {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> noise((size_t) hparams.vibevoice.diff_latent_size * n_frames);
    for (float & x : noise) {
        x = dist(rng);
    }

    return sample_speech_latents(backend, noise.data(), cond, uncond, n_frames, cfg_scale);
}

void llama_model_vibevoice::unscale_speech_latents(float * latents, size_t n) const {
    const float scale = hparams.vibevoice.speech_scaling_factor;
    const float bias  = hparams.vibevoice.speech_bias_factor;
    for (size_t i = 0; i < n; ++i) {
        latents[i] = latents[i] / scale - bias;
    }
}

// acoustic_conn(scaled latents) + semantic_conn(semantic_enc(wav)), matches upstream feedback embeds
ggml_tensor * llama_model_vibevoice::build_speech_feedback(ggml_context * ctx, ggml_tensor * scaled_latents, ggml_tensor * wav) const {
    return build_speech_feedback(ctx, scaled_latents, wav, nullptr);
}

ggml_tensor * llama_model_vibevoice::build_speech_feedback(ggml_context * ctx, ggml_tensor * scaled_latents, ggml_tensor * wav, stream_io * st) const {
    const auto & vv = hparams.vibevoice;

    GGML_ASSERT(ggml_is_matrix(scaled_latents) && scaled_latents->ne[0] == vv.diff_latent_size);
    GGML_ASSERT(ggml_is_matrix(wav) && wav->ne[1] == 1);

    ggml_tensor * acoustic_emb = build_connector(ctx, scaled_latents, acoustic_conn);

    ggml_tensor * semantic_latents = build_semantic_enc(ctx, wav, st);
    GGML_ASSERT(semantic_latents->ne[1] == scaled_latents->ne[1]);

    ggml_tensor * semantic_emb = build_connector(ctx, semantic_latents, semantic_conn);

    return ggml_add(ctx, acoustic_emb, semantic_emb);
}

//
// streaming tokenizer: chunk by chunk decode/encode with causal conv caches
//

std::vector<float> llama_model_vibevoice::acoustic_dec_stream(ggml_backend_t backend, tok_stream_cache & cache, const float * latents, int64_t n_new) const {
    GGML_ASSERT(n_new > 0);

    auto & graph = *acoustic_dec_graph;
    std::lock_guard<std::mutex> lock(graph.mutex);
    if (!graph.matches(backend, n_new)) {
        graph.init(backend, n_new, 0, 4 * GGML_DEFAULT_GRAPH_SIZE);
        ggml_context * ctx = graph.ctx.get();
        ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hparams.vibevoice.acoustic.vae_dim, n_new);
        ggml_set_input(inp);
        graph.inputs = { inp };
        graph.alloc(build_acoustic_dec(ctx, inp, &graph.st));
    }
    return graph.run_stream(cache, { latents });
}

// the strided convs only see whole strides when the chunk is a multiple of the encoder hop
static int64_t vv_enc_hop(const llama_hparams_vv_tokenizer & hp) {
    int64_t hop = 1;
    for (uint32_t s = 0; s + 1 < hp.n_enc_stages; ++s) {
        const uint32_t ratio = hp.enc_ratios[s];
        if (ratio == 0 || hop > INT64_MAX / ratio) {
            throw std::runtime_error(format("vibevoice: invalid tokenizer ratio %u, hop must fit in int64_t", ratio));
        }
        hop *= ratio;
    }
    return hop;
}

// waveform samples per latent frame from the acoustic decoder upsample ratios
static int64_t vv_dec_hop(const llama_hparams_vv_tokenizer & hp) {
    int64_t hop = 1;
    for (uint32_t s = 0; s + 1 < hp.n_dec_stages; ++s) {
        const uint32_t ratio = hp.dec_ratios[s];
        if (ratio == 0 || hop > INT64_MAX / ratio) {
            throw std::runtime_error(format("vibevoice: invalid tokenizer ratio %u, hop must fit in int64_t", ratio));
        }
        hop *= ratio;
    }
    return hop;
}

std::vector<float> llama_model_vibevoice::semantic_enc_stream(ggml_backend_t backend, tok_stream_cache & cache, const float * wav, int64_t n_samples) const {
    GGML_ASSERT(n_samples > 0 && n_samples % vv_enc_hop(hparams.vibevoice.semantic) == 0);

    const int64_t hop = vv_enc_hop(hparams.vibevoice.semantic);
    auto & graph = *semantic_enc_graph;
    std::lock_guard<std::mutex> lock(graph.mutex);
    if (!graph.matches(backend, n_samples, hop)) {
        graph.init(backend, n_samples, hop, 4 * GGML_DEFAULT_GRAPH_SIZE);
        ggml_context * ctx = graph.ctx.get();
        ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_samples, 1);
        ggml_set_input(inp);
        graph.inputs = { inp };
        graph.alloc(build_semantic_enc(ctx, inp, &graph.st));
    }
    return graph.run_stream(cache, { wav });
}

std::vector<float> llama_model_vibevoice::speech_feedback_stream(ggml_backend_t backend, tok_stream_cache & semantic_cache,
        const float * scaled_latents, const float * wav, int64_t n_new) const {
    const int64_t hop    = vv_enc_hop(hparams.vibevoice.semantic);
    const int64_t ac_hop = vv_dec_hop(hparams.vibevoice.acoustic);
    GGML_ASSERT(n_new > 0);

    // feedback needs one semantic frame per acoustic frame
    if (hop <= 0 || ac_hop <= 0 || hop != ac_hop) {
        throw std::runtime_error(format(
                "vibevoice: semantic hop %zu is incompatible with acoustic hop %zu",
                (size_t) hop, (size_t) ac_hop));
    }

    auto & graph = *feedback_graph;
    std::lock_guard<std::mutex> lock(graph.mutex);
    if (!graph.matches(backend, n_new, hop)) {
        graph.init(backend, n_new, hop, 4 * GGML_DEFAULT_GRAPH_SIZE);
        ggml_context * ctx = graph.ctx.get();
        ggml_tensor * lat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hparams.vibevoice.diff_latent_size, n_new);
        ggml_set_input(lat);
        ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_new * hop, 1);
        ggml_set_input(inp);
        graph.inputs = { lat, inp };
        graph.alloc(build_speech_feedback(ctx, lat, inp, &graph.st));
    }
    return graph.run_stream(semantic_cache, { scaled_latents, wav });
}
