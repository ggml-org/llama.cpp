#include "models.h"

#include <cmath>

// Qwen3-TTS-12Hz Code2Wav vocoder (Qwen3TTSTokenizerV2Decoder).
//
// One frame = n_codebooks interleaved code ids (codebook j of frame f is
// token f*n_codebooks + j). The graph turns F frames into F*1920 samples
// of 24 kHz PCM (1920 = prod(upsampling_ratios) * prod(upsample_rates)).
//
// Pipeline (mirrors the HF decoder):
//   codes -> per-codebook lookup, codebook 0 projected by vq_first_proj
//           and codebooks 1..K-1 summed then projected by vq_rest_proj
//         -> causal pre_conv (k = 3)
//         -> causal pre-transformer: sliding-window RoPE attention, layer
//            scale, SiLU gated MLP
//         -> ConvNeXt upsample stages (transposed conv + dwconv + LN + MLP)
//         -> stem conv (k = 7)
//         -> per rate: snake + transposed conv (k = 2*rate) + residual
//            dilated snake conv units
//         -> snake + causal output conv (k = 7) -> clamp [-1, 1]
//
// snake_beta(x) = x + b*sin^2(a*x); the converter bakes a = exp(alpha) and
// b = 1/(exp(beta) + 1e-9), so the graph only needs mul/sin/sqr/add.
//
// Causality (streaming, no lookahead):
//   - plain convs run with symmetric pad p0 = (k-1)*d and keep the first L
//     outputs, so every output sample depends only on current and past
//     input samples;
//   - conv_transpose runs with p0 = 0 (required by the op); output m
//     depends only on frames <= m/rate, so keeping the first L*rate
//     outputs preserves causality (this is the upstream trim of the
//     trailing `rate` samples).
//
// Consequence: the waveform prefix of frames [0, F) equals the prefix of
// any longer stream, so PCM can be emitted at every frame boundary.
//
// Attention window: the additive mask applies the sliding window (n_swa)
// exactly, so a chunk may contain any number of frames.
//
// Output contract (encode() path, all tokens are outputs):
//   t_embd is [n_embd_out, n_tokens] with n_embd_out = samples_per_frame /
//   n_codebooks. Token f*n_codebooks + j carries samples [j*n_embd_out,
//   (j+1)*n_embd_out) of frame f, so reading the embedding buffer
//   sequentially yields the PCM stream.

namespace {

// frame positions for RoPE: every frame takes the position of its first
// code token (callers give all tokens of a frame the same position)
class llm_graph_input_c2w_frame_pos : public llm_graph_input_i {
public:
    llm_graph_input_c2w_frame_pos(int64_t n_codebooks) : n_codebooks(n_codebooks) {}

    void set_input(const llama_ubatch * ubatch) override {
        if (ubatch->pos && pos) {
            const int64_t n_frames = pos->ne[0];

            GGML_ASSERT(ubatch->n_tokens == n_frames*n_codebooks);

            std::vector<llama_pos> fp(n_frames);
            for (int64_t f = 0; f < n_frames; ++f) {
                fp[f] = ubatch->pos[f*n_codebooks];
            }

            ggml_backend_tensor_set(pos, fp.data(), 0, n_frames*ggml_element_size(pos));
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        return pos->ne[0]*n_codebooks == params.ubatch.n_tokens;
    }

    ggml_tensor * pos = nullptr; // I32 [n_frames]

    const int64_t n_codebooks;
};

// additive sliding-window causal mask: 0 where key <= query and the key is
// inside the window, -inf otherwise. Layout [n_frames, n_frames] with
// ne[0] = key, ne[1] = query, as consumed by soft_max_ext / flash_attn_ext
class llm_graph_input_c2w_mask : public llm_graph_input_i {
public:
    llm_graph_input_c2w_mask(int64_t n_frames, uint32_t n_swa) {
        arr.resize(n_frames*n_frames);
        for (int64_t i = 0; i < n_frames; ++i) {
            for (int64_t j = 0; j < n_frames; ++j) {
                const bool visible = j <= i && (n_swa == 0 || i - j < (int64_t) n_swa);
                arr[i*n_frames + j] = visible ? 0.0f : -INFINITY;
            }
        }
    }

    void set_input(const llama_ubatch * /*ubatch*/) override {
        ggml_backend_tensor_set(mask, arr.data(), 0, arr.size()*ggml_element_size(mask));
    }

    bool can_reuse(const llm_graph_params & /*params*/) override {
        return true;
    }

    ggml_tensor * mask = nullptr; // F32 [n_frames, n_frames]

    std::vector<float> arr;
};

} // anonymous namespace

void llama_model_qwen3_tts_code2wav::load_arch_hparams(llama_model_loader & ml) {
    type = LLM_TYPE_UNKNOWN;

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);

    // raw arch-prefixed keys; no LLM_KV entries needed
    const std::string kv = std::string(llm_arch_name(arch)) + ".";

    ml.get_key(kv + "codebook_count",    n_codebooks);
    ml.get_key(kv + "residual_units",    n_res_units);
    ml.get_key(kv + "sample_rate",       sample_rate, false);
    ml.get_key(kv + "convnext_norm_eps", f_convnext_eps, false);
    ml.get_arr(kv + "upsample_rates",    upsample_rates);
    ml.get_arr(kv + "upsample_ratios",   upsample_ratios);
    ml.get_arr(kv + "residual_dilations", residual_dilations);

    GGML_ASSERT(n_codebooks > 0);
    GGML_ASSERT(n_res_units > 0);
    GGML_ASSERT(!upsample_rates.empty());
    GGML_ASSERT(!upsample_ratios.empty());
    GGML_ASSERT(residual_dilations.size() == n_res_units);

    int64_t up = 1;
    for (uint32_t r : upsample_rates)  up *= r;
    for (uint32_t r : upsample_ratios) up *= r;

    // each code token carries an equal slice of the frame's PCM
    GGML_ASSERT(up == (int64_t) hparams.n_embd_out()*n_codebooks);
}

void llama_model_qwen3_tts_code2wav::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    LLAMA_LOG_WARN("c2w: load_arch_tensors enter (n_codebooks=%u, n_vocab=%lld, n_embd=%lld)\n", n_codebooks, (long long) n_vocab, (long long) n_embd);

    GGML_UNUSED(n_embd_head_v);
    GGML_UNUSED(n_embd_k_gqa);
    GGML_UNUSED(n_embd_v_gqa);
    GGML_UNUSED(n_embd_gqa);

    const int64_t n_embd_head = n_embd_head_k;
    const int64_t n_embd_attn = n_embd_head*n_head;      // q/k/v/o width (16*64)
    const int64_t n_up        = upsample_ratios.size();  // convnext stages
    const int64_t n_blk       = upsample_rates.size();   // decoder blocks

    // codebook tables: width comes from the checkpoint itself
    c2w_codebook_embd.resize(n_codebooks, nullptr);
    {
        const std::string name0 = tn(LLM_TENSOR_C2W_CODEBOOK_EMBD, "weight", 0);
        const struct ggml_tensor * meta = ml.get_tensor_meta(name0.c_str());
        GGML_ASSERT(meta != nullptr && "missing c2w codebook table");
        vq_dim = meta->ne[0];
        GGML_ASSERT(meta->ne[1] == n_vocab);
    }
    for (uint32_t j = 0; j < n_codebooks; ++j) {
        const std::string name = tn(LLM_TENSOR_C2W_CODEBOOK_EMBD, "weight", j);
        LLAMA_LOG_WARN("c2w: creating codebook %u, name=%s\n", j, name.c_str());
        try {
            LLAMA_LOG_WARN("c2w: pre lookup name\n");
            const struct ggml_tensor * meta = ml.get_tensor_meta(name.c_str());
            LLAMA_LOG_WARN("c2w: meta ptr %p\n", (const void *) meta);
            if (meta) {
                LLAMA_LOG_WARN("c2w: meta ne=[%lld,%lld,%lld,%lld]\n",
                    (long long) meta->ne[0], (long long) meta->ne[1],
                    (long long) meta->ne[2], (long long) meta->ne[3]);
            }
            c2w_codebook_embd[j] = create_tensor(tn(LLM_TENSOR_C2W_CODEBOOK_EMBD, "weight", j), { vq_dim, n_vocab }, 0);
        } catch (const std::exception & e) {
            LLAMA_LOG_WARN("c2w: codebook %u FAILED: %s\n", j, e.what());
            throw;
        }
        LLAMA_LOG_WARN("c2w: codebook %u done (ne = [%lld, %lld])\n", j, (long long) c2w_codebook_embd[j]->ne[0], (long long) c2w_codebook_embd[j]->ne[1]);
    }

    LLAMA_LOG_WARN("c2w: codebook loop done\n");
    c2w_vq_first_proj = create_tensor(tn(LLM_TENSOR_C2W_VQ_FIRST_PROJ, "weight"), { vq_dim, n_embd }, 0);
    c2w_vq_rest_proj  = create_tensor(tn(LLM_TENSOR_C2W_VQ_REST_PROJ,  "weight"), { vq_dim, n_embd }, 0);

    LLAMA_LOG_WARN("c2w: vq tables + projections done\n");
    // pre_conv: [k, latent, codebook_dim] -> latent comes from the checkpoint
    {
        const std::string name = tn(LLM_TENSOR_C2W_PRE_CONV, "weight");
        const struct ggml_tensor * meta = ml.get_tensor_meta(name.c_str());
        GGML_ASSERT(meta != nullptr && "missing c2w pre_conv");
        GGML_ASSERT(meta->ne[0] == 3);
        GGML_ASSERT(meta->ne[1] == n_embd); // codebook_dim == 2 * vq_dim == 512
        latent_dim = meta->ne[2];
    }

    c2w_pre_conv   = create_tensor(tn(LLM_TENSOR_C2W_PRE_CONV, "weight"), { 3, n_embd, latent_dim }, 0);
    c2w_pre_conv_b = create_tensor(tn(LLM_TENSOR_C2W_PRE_CONV, "bias"),   { 1, latent_dim }, 0);

    // transformer
    c2w_tf_in_proj   = create_tensor(tn(LLM_TENSOR_C2W_TF_IN_PROJ, "weight"), { latent_dim, n_embd }, 0);
    c2w_tf_in_proj_b = create_tensor(tn(LLM_TENSOR_C2W_TF_IN_PROJ, "bias"),   { 1, n_embd }, 0);

    tf_attn_norm.resize(n_layer);
    tf_wq.resize(n_layer); tf_wk.resize(n_layer); tf_wv.resize(n_layer); tf_wo.resize(n_layer);
    tf_attn_scale.resize(n_layer);
    tf_ffn_norm.resize(n_layer);
    tf_ffn_gate.resize(n_layer); tf_ffn_up.resize(n_layer); tf_ffn_down.resize(n_layer);
    tf_ffn_scale.resize(n_layer);

    for (int i = 0; i < n_layer; ++i) {
        tf_attn_norm[i]  = create_tensor(tn(LLM_TENSOR_C2W_TF_ATTN_NORM, "weight", i), { n_embd }, 0);
        tf_wq[i]         = create_tensor(tn(LLM_TENSOR_C2W_TF_ATTN_Q,    "weight", i), { n_embd, n_embd_attn }, 0);
        tf_wk[i]         = create_tensor(tn(LLM_TENSOR_C2W_TF_ATTN_K,    "weight", i), { n_embd, n_embd_attn }, 0);
        tf_wv[i]         = create_tensor(tn(LLM_TENSOR_C2W_TF_ATTN_V,    "weight", i), { n_embd, n_embd_attn }, 0);
        tf_wo[i]         = create_tensor(tn(LLM_TENSOR_C2W_TF_ATTN_O,    "weight", i), { n_embd_attn, n_embd }, 0);
        tf_attn_scale[i] = create_tensor(tn(LLM_TENSOR_C2W_TF_ATTN_SCALE, nullptr, i), { n_embd }, 0);

        tf_ffn_norm[i]  = create_tensor(tn(LLM_TENSOR_C2W_TF_FFN_NORM, "weight", i), { n_embd }, 0);
        tf_ffn_gate[i]  = create_tensor(tn(LLM_TENSOR_C2W_TF_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
        tf_ffn_up[i]    = create_tensor(tn(LLM_TENSOR_C2W_TF_FFN_UP,   "weight", i), { n_embd, n_ff }, 0);
        tf_ffn_down[i]  = create_tensor(tn(LLM_TENSOR_C2W_TF_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
        tf_ffn_scale[i] = create_tensor(tn(LLM_TENSOR_C2W_TF_FFN_SCALE, nullptr, i), { n_embd }, 0);
    }

    c2w_tf_norm      = create_tensor(tn(LLM_TENSOR_C2W_TF_NORM,     "weight"), { n_embd }, 0);
    c2w_tf_out_proj  = create_tensor(tn(LLM_TENSOR_C2W_TF_OUT_PROJ, "weight"), { n_embd, latent_dim }, 0);
    c2w_tf_out_proj_b = create_tensor(tn(LLM_TENSOR_C2W_TF_OUT_PROJ, "bias"),  { 1, latent_dim }, 0);

    // convnext upsample stages; widths come from the transposed conv meta
    up_transconv.resize(n_up); up_transconv_b.resize(n_up);
    up_dwconv.resize(n_up);    up_dwconv_b.resize(n_up);
    up_norm.resize(n_up);      up_norm_b.resize(n_up);
    up_pw1.resize(n_up);       up_pw1_b.resize(n_up);
    up_pw2.resize(n_up);       up_pw2_b.resize(n_up);
    up_gamma.resize(n_up);

    for (int64_t s = 0; s < n_up; ++s) {
        const std::string name = tn(LLM_TENSOR_C2W_UP_TRANSCONV, "weight", s);
        const struct ggml_tensor * meta = ml.get_tensor_meta(name.c_str());
        GGML_ASSERT(meta != nullptr && "missing c2w upsample transconv");
        GGML_ASSERT(meta->ne[0] == (int64_t) upsample_ratios[s]);
        GGML_ASSERT(meta->ne[2] == (s == 0 ? latent_dim : up_transconv[s - 1]->ne[1]));
        const int64_t c_out = meta->ne[1];
        const int64_t c_in  = meta->ne[2];

        up_transconv[s]   = create_tensor(tn(LLM_TENSOR_C2W_UP_TRANSCONV, "weight", s), { meta->ne[0], c_out, c_in }, 0);
        up_transconv_b[s] = create_tensor(tn(LLM_TENSOR_C2W_UP_TRANSCONV, "bias",   s), { 1, c_out }, 0);

        up_dwconv[s]   = create_tensor(tn(LLM_TENSOR_C2W_UP_DWCONV, "weight", s), { 7, 1, c_out }, 0);
        up_dwconv_b[s] = create_tensor(tn(LLM_TENSOR_C2W_UP_DWCONV, "bias",   s), { 1, c_out }, 0);

        up_norm[s]   = create_tensor(tn(LLM_TENSOR_C2W_UP_NORM, "weight", s), { c_out }, 0);
        up_norm_b[s] = create_tensor(tn(LLM_TENSOR_C2W_UP_NORM, "bias",   s), { c_out }, 0);

        // pwconv1 width read from meta to stay checkpoint-driven
        const std::string name_pw1 = tn(LLM_TENSOR_C2W_UP_PW1, "weight", s);
        const struct ggml_tensor * meta_pw1 = ml.get_tensor_meta(name_pw1.c_str());
        GGML_ASSERT(meta_pw1 != nullptr && "missing c2w upsample pwconv1");
        const int64_t c_mid = meta_pw1->ne[1];

        up_pw1[s]   = create_tensor(tn(LLM_TENSOR_C2W_UP_PW1, "weight", s), { c_out, c_mid }, 0);
        up_pw1_b[s] = create_tensor(tn(LLM_TENSOR_C2W_UP_PW1, "bias",   s), { 1, c_mid }, 0);
        up_pw2[s]   = create_tensor(tn(LLM_TENSOR_C2W_UP_PW2, "weight", s), { c_mid, c_out }, 0);
        up_pw2_b[s] = create_tensor(tn(LLM_TENSOR_C2W_UP_PW2, "bias",   s), { 1, c_out }, 0);

        up_gamma[s] = create_tensor(tn(LLM_TENSOR_C2W_UP_GAMMA, nullptr, s), { c_out }, 0);
    }

    // stem conv: decoder_dim comes from the checkpoint
    {
        const std::string name = tn(LLM_TENSOR_C2W_STEM, "weight");
        const struct ggml_tensor * meta = ml.get_tensor_meta(name.c_str());
        GGML_ASSERT(meta != nullptr && "missing c2w stem");
        GGML_ASSERT(meta->ne[0] == 7);
        GGML_ASSERT(meta->ne[1] == latent_dim);
        decoder_dim = meta->ne[2];
    }

    c2w_stem   = create_tensor(tn(LLM_TENSOR_C2W_STEM, "weight"), { 7, latent_dim, decoder_dim }, 0);
    c2w_stem_b = create_tensor(tn(LLM_TENSOR_C2W_STEM, "bias"),   { 1, decoder_dim }, 0);

    // decoder blocks: width halves per block
    blk_snake_a.resize(n_blk); blk_snake_b.resize(n_blk);
    blk_transconv.resize(n_blk); blk_transconv_b.resize(n_blk);

    res_a1.resize(n_blk*n_res_units); res_b1.resize(n_blk*n_res_units);
    res_conv1.resize(n_blk*n_res_units); res_conv1_b.resize(n_blk*n_res_units);
    res_a2.resize(n_blk*n_res_units); res_b2.resize(n_blk*n_res_units);
    res_conv2.resize(n_blk*n_res_units); res_conv2_b.resize(n_blk*n_res_units);

    for (int64_t b = 0; b < n_blk; ++b) {
        const int64_t c_in  = decoder_dim >> b;
        const int64_t c_out = decoder_dim >> (b + 1);

        blk_snake_a[b] = create_tensor(tn(LLM_TENSOR_C2W_BLOCK_SNAKE_A, nullptr, b), { c_in }, 0);
        blk_snake_b[b] = create_tensor(tn(LLM_TENSOR_C2W_BLOCK_SNAKE_B, nullptr, b), { c_in }, 0);

        blk_transconv[b]   = create_tensor(tn(LLM_TENSOR_C2W_BLOCK_TRANSCONV, "weight", b), { 2*upsample_rates[b], c_out, c_in }, 0);
        blk_transconv_b[b] = create_tensor(tn(LLM_TENSOR_C2W_BLOCK_TRANSCONV, "bias",   b), { 1, c_out }, 0);

        for (uint32_t u = 0; u < n_res_units; ++u) {
            const int64_t uid = b*n_res_units + u;

            res_a1[uid]    = create_tensor(tn(LLM_TENSOR_C2W_RES_SNAKE1_A, nullptr, b, u), { c_out }, 0);
            res_b1[uid]    = create_tensor(tn(LLM_TENSOR_C2W_RES_SNAKE1_B, nullptr, b, u), { c_out }, 0);
            res_conv1[uid] = create_tensor(tn(LLM_TENSOR_C2W_RES_CONV1, "weight", b, u), { 7, c_out, c_out }, 0);
            res_conv1_b[uid] = create_tensor(tn(LLM_TENSOR_C2W_RES_CONV1, "bias", b, u), { 1, c_out }, 0);

            res_a2[uid]    = create_tensor(tn(LLM_TENSOR_C2W_RES_SNAKE2_A, nullptr, b, u), { c_out }, 0);
            res_b2[uid]    = create_tensor(tn(LLM_TENSOR_C2W_RES_SNAKE2_B, nullptr, b, u), { c_out }, 0);
            res_conv2[uid] = create_tensor(tn(LLM_TENSOR_C2W_RES_CONV2, "weight", b, u), { 1, c_out, c_out }, 0);
            res_conv2_b[uid] = create_tensor(tn(LLM_TENSOR_C2W_RES_CONV2, "bias", b, u), { 1, c_out }, 0);
        }
    }

    const int64_t c_last = decoder_dim >> n_blk;

    c2w_out_snake_a = create_tensor(tn(LLM_TENSOR_C2W_OUT_SNAKE_A, nullptr), { c_last }, 0);
    c2w_out_snake_b = create_tensor(tn(LLM_TENSOR_C2W_OUT_SNAKE_B, nullptr), { c_last }, 0);
    c2w_output      = create_tensor(tn(LLM_TENSOR_C2W_OUTPUT, "weight"), { 7, 1, c_last }, 0);
    c2w_output_b    = create_tensor(tn(LLM_TENSOR_C2W_OUTPUT, "bias"),   { 1 }, 0);
}

std::unique_ptr<llm_graph_context> llama_model_qwen3_tts_code2wav::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen3_tts_code2wav::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const auto & m = static_cast<const llama_model_qwen3_tts_code2wav &>(model);

    if (n_tokens % m.n_codebooks != 0) {
        GGML_ABORT("code2wav: n_tokens (%d) must be a multiple of n_codebooks (%u)",
                   (int) n_tokens, m.n_codebooks);
    }
    const int64_t n_frames = n_tokens / m.n_codebooks;

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t n_head      = hparams.n_head();

    // ---- inputs ----

    // codes arrive as a token batch: codes[f*n_codebooks + j] = codebook j id
    auto inp_embd = std::make_unique<llm_graph_input_embd>(hparams.n_embd);
    inp_embd->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_embd->tokens);
    ggml_tensor * codes = inp_embd->tokens;
    cb(codes, "inp_codes", -1);
    res->add_input(std::move(inp_embd));

    auto inp_pos = std::make_unique<llm_graph_input_c2w_frame_pos>(m.n_codebooks);
    inp_pos->pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_frames);
    ggml_set_input(inp_pos->pos);
    ggml_tensor * frame_pos = inp_pos->pos;
    cb(frame_pos, "inp_frame_pos", -1);
    res->add_input(std::move(inp_pos));

    auto inp_mask = std::make_unique<llm_graph_input_c2w_mask>(n_frames, hparams.n_swa);
    inp_mask->mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_frames, n_frames);
    ggml_set_input(inp_mask->mask);
    ggml_tensor * kq_mask = inp_mask->mask;
    cb(kq_mask, "inp_kq_mask", -1);
    res->add_input(std::move(inp_mask));

    // ---- RVQ decode: [vq_dim, F] -> [n_embd, F] ----

    ggml_tensor * ids = ggml_reshape_2d(ctx0, codes, m.n_codebooks, n_frames);

    ggml_tensor * e_first = nullptr;
    ggml_tensor * e_rest  = nullptr;
    for (uint32_t j = 0; j < m.n_codebooks; ++j) {
        ggml_tensor * ids_j = ggml_cont_1d(ctx0,
                ggml_view_2d(ctx0, ids, 1, n_frames, ids->nb[1], j*ids->nb[0]), n_frames);

        ggml_tensor * e = ggml_get_rows(ctx0, m.c2w_codebook_embd[j], ids_j);

        if (j == 0) {
            e_first = e;
        } else {
            e_rest = e_rest ? ggml_add(ctx0, e_rest, e) : e;
        }
    }

    ggml_tensor * cur = ggml_add(ctx0,
            ggml_mul_mat(ctx0, m.c2w_vq_first_proj, e_first),
            ggml_mul_mat(ctx0, m.c2w_vq_rest_proj,  e_rest));
    cb(cur, "rvq_sum", -1);

    if (cur->type != GGML_TYPE_F32) {
        cur = ggml_cast(ctx0, cur, GGML_TYPE_F32);
    }

    // ---- pre_conv (time-major) ----

    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // [F, latent]

    cur = build_causal_conv(m.c2w_pre_conv, m.c2w_pre_conv_b, cur, 1);
    cb(cur, "pre_conv", -1);

    // ---- pre-transformer (channel-major) ----

    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // [latent, F]

    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, m.c2w_tf_in_proj, cur), m.c2w_tf_in_proj_b);

    for (uint32_t il = 0; il < n_layer; ++il) {
        ggml_tensor * x = cur;

        // sliding-window causal self-attention
        {
            ggml_tensor * h = build_norm(cur, m.tf_attn_norm[il], nullptr, LLM_NORM_RMS, il);

            ggml_tensor * q = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, m.tf_wq[il], h), n_embd_head, n_head, n_frames);
            ggml_tensor * k = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, m.tf_wk[il], h), n_embd_head, n_head, n_frames);
            ggml_tensor * v = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, m.tf_wv[il], h), n_embd_head, n_head, n_frames);

            q = ggml_rope_ext(ctx0, q, frame_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            k = ggml_rope_ext(ctx0, k, frame_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            cb(q, "tf_q", il);
            cb(k, "tf_k", il);
            cb(v, "tf_v", il);

            ggml_tensor * attn = build_attn_mha(q, k, v,
                    nullptr, kq_mask, nullptr, nullptr,
                    1.0f/sqrtf((float) n_embd_head), il);

            cur = ggml_add(ctx0, x,
                    ggml_mul(ctx0, ggml_mul_mat(ctx0, m.tf_wo[il], attn), m.tf_attn_scale[il]));
        }

        // SiLU gated MLP with layer scale
        {
            x = cur;

            ggml_tensor * h = build_norm(cur, m.tf_ffn_norm[il], nullptr, LLM_NORM_RMS, il);

            ggml_tensor * ffn = ggml_mul(ctx0,
                    ggml_silu(ctx0, ggml_mul_mat(ctx0, m.tf_ffn_gate[il], h)),
                    ggml_mul_mat(ctx0, m.tf_ffn_up[il], h));
            ffn = ggml_mul_mat(ctx0, m.tf_ffn_down[il], ffn);

            cur = ggml_add(ctx0, x, ggml_mul(ctx0, ffn, m.tf_ffn_scale[il]));
        }

        cb(cur, "tf_layer", il);
    }

    cur = build_norm(cur, m.c2w_tf_norm, nullptr, LLM_NORM_RMS, -1);
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, m.c2w_tf_out_proj, cur), m.c2w_tf_out_proj_b);
    cb(cur, "tf_out", -1);

    // ---- convnext upsample stages (time-major) ----

    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // [F, latent]

    for (size_t s = 0; s < m.upsample_ratios.size(); ++s) {
        cur = build_upsample(m.up_transconv[s], m.up_transconv_b[s], cur, m.upsample_ratios[s]);
        cb(cur, "up_transconv", s);

        ggml_tensor * x = cur;

        cur = build_causal_conv_dw(m.up_dwconv[s], m.up_dwconv_b[s], cur);

        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // [C, L]
        cur = ggml_norm(ctx0, cur, m.f_convnext_eps);
        cur = ggml_mul(ctx0, cur, m.up_norm[s]);
        cur = ggml_add(ctx0, cur, m.up_norm_b[s]);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, m.up_pw1[s], cur), m.up_pw1_b[s]);
        cur = ggml_gelu(ctx0, cur);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, m.up_pw2[s], cur), m.up_pw2_b[s]);
        cur = ggml_mul(ctx0, cur, m.up_gamma[s]);
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // [L, C]

        cur = ggml_add(ctx0, cur, x);
        cb(cur, "up_convnext", s);
    }

    // ---- stem + decoder blocks ----

    cur = build_causal_conv(m.c2w_stem, m.c2w_stem_b, cur, 1);
    cb(cur, "stem", -1);

    for (size_t b = 0; b < m.upsample_rates.size(); ++b) {
        cur = build_snake(cur, m.blk_snake_a[b], m.blk_snake_b[b]);
        cur = build_upsample(m.blk_transconv[b], m.blk_transconv_b[b], cur, m.upsample_rates[b]);
        cb(cur, "blk_up", b);

        for (uint32_t u = 0; u < m.n_res_units; ++u) {
            const int64_t uid = (int64_t) b*m.n_res_units + u;

            ggml_tensor * x = cur;

            cur = build_snake(cur, m.res_a1[uid], m.res_b1[uid]);
            cur = build_causal_conv(m.res_conv1[uid], m.res_conv1_b[uid], cur, m.residual_dilations[u]);
            cur = build_snake(cur, m.res_a2[uid], m.res_b2[uid]);
            cur = build_causal_conv(m.res_conv2[uid], m.res_conv2_b[uid], cur, 1);

            cur = ggml_add(ctx0, cur, x);
            cb(cur, "res_unit", uid);
        }
    }

    // ---- head ----

    cur = build_snake(cur, m.c2w_out_snake_a, m.c2w_out_snake_b);
    cur = build_causal_conv(m.c2w_output, m.c2w_output_b, cur, 1); // [F*samples_per_frame, 1]

    const int64_t samples_per_frame = (int64_t) hparams.n_embd_out()*m.n_codebooks;

    GGML_ASSERT(cur->ne[0] == n_frames*samples_per_frame);

    cur = ggml_clamp(ctx0, cur, -1.0f, 1.0f);

    // [n_embd_out, n_tokens]: token f*n_codebooks + j carries slice j of
    // frame f, so the embedding buffer reads back as the PCM stream
    cur = ggml_reshape_2d(ctx0, cur, hparams.n_embd_out(), n_tokens);
    cb(cur, "result_embd", -1);

    res->t_embd = cur;
    ggml_set_output(cur);

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llama_model_qwen3_tts_code2wav::graph::build_snake(ggml_tensor * x, ggml_tensor * a, ggml_tensor * b) const {
    // x is time-major [L, C]; broadcast the per-channel vectors over time
    ggml_tensor * a2 = ggml_reshape_2d(ctx0, a, 1, a->ne[0]);
    ggml_tensor * b2 = ggml_reshape_2d(ctx0, b, 1, b->ne[0]);

    ggml_tensor * cur = ggml_mul(ctx0, x, a2);  // a*x
    cur = ggml_sin(ctx0, cur);
    cur = ggml_sqr(ctx0, cur);                  // sin^2(a*x)
    cur = ggml_mul(ctx0, cur, b2);              // b*sin^2(a*x)

    return ggml_add(ctx0, x, cur);
}

ggml_tensor * llama_model_qwen3_tts_code2wav::graph::build_causal_conv(ggml_tensor * w, ggml_tensor * b, ggml_tensor * x, int64_t dilation) const {
    const int64_t n_in = x->ne[0];
    const int64_t pad  = (w->ne[0] - 1)*dilation;

    // symmetric pad; the trailing pad outputs are non-causal, drop them
    ggml_tensor * cur = ggml_conv_1d(ctx0, w, x, 1, (int) pad, (int) dilation);
    cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_in, cur->ne[1], cur->nb[1], 0));

    if (b != nullptr) {
        cur = ggml_add(ctx0, cur, b);
    }

    return cur;
}

ggml_tensor * llama_model_qwen3_tts_code2wav::graph::build_causal_conv_dw(ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) const {
    const int64_t n_in = x->ne[0];
    const int64_t pad  = w->ne[0] - 1;

    ggml_tensor * cur = ggml_conv_1d_dw(ctx0, w, x, 1, (int) pad, 1);
    cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_in, cur->ne[1], cur->nb[1], 0));

    if (b != nullptr) {
        cur = ggml_add(ctx0, cur, b);
    }

    return cur;
}

ggml_tensor * llama_model_qwen3_tts_code2wav::graph::build_upsample(ggml_tensor * w, ggml_tensor * b, ggml_tensor * x, int64_t rate) const {
    const int64_t n_in = x->ne[0];

    // emits (L-1)*rate + k samples; the trailing k - rate only depend on
    // the last frame's "future" boundary, drop them (upstream trim)
    ggml_tensor * cur = ggml_conv_transpose_1d(ctx0, w, x, (int) rate, 0, 1);
    cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_in*rate, cur->ne[1], cur->nb[1], 0));

    if (b != nullptr) {
        cur = ggml_add(ctx0, cur, b);
    }

    return cur;
}


