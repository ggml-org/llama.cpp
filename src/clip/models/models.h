#pragma once

#include "../clip-graph.h"

#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

/*
 * IMPORTANT: The mtmd module does NOT accept pull requests that are fully or predominantly AI-generated.
 * We encourage human contributors to ensure the quality and reliability of the codebase.
 */

struct clip_graph_siglip : clip_graph {
    clip_graph_siglip(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_gemma4v : clip_graph {
    clip_graph_gemma4v(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
    ggml_tensor * build_mm(ggml_tensor * w, ggml_tensor * x) const override;
    bool support_batch() const override { return true; }
};

struct clip_graph_gemma4uv : clip_graph {
    clip_graph_gemma4uv(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_nemotron_transcribe : clip_graph {
    const clip_subctx_enc_state * state; // nullptr during warmup
    clip_graph_nemotron_transcribe(clip_subcontext * ctx, const clip_image_f32 & img, const clip_subctx_enc_state * state)
        : clip_graph(ctx, img), state(state) {}
    ggml_cgraph * build() override;
    ggml_tensor * build_spk_encoder(ggml_tensor * cur, ggml_tensor * pos);

    // one recording: encodes its windows in turn, carrying the speaker state between them
    static bool encode_windows(clip_subcontext * ctx, const clip_model & model, clip_subctx_encode_params * params);

    // NeMo ParallelExpertEncoder online defaults, in output frames: 40 s windows with 4 s of context on each side
    static constexpr int window_core  = 500;
    static constexpr int window_left  = 50;
    static constexpr int window_right = 50;
};

struct clip_graph_pixtral : clip_graph {
    clip_graph_pixtral(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_deepseek4v : clip_graph {
    clip_graph_deepseek4v(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_magevl : clip_graph {
    clip_graph_magevl(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_sensenova_u1 : clip_graph {
    clip_graph_sensenova_u1(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_lfm3vl : clip_graph {
    clip_graph_lfm3vl(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_ling3vl : clip_graph {
    clip_graph_ling3vl(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_minimax_m3 : clip_graph {
    clip_graph_minimax_m3(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
    ggml_tensor * apply_rope(ggml_tensor * x, ggml_tensor * pos_h, ggml_tensor * pos_w);
};

struct clip_graph_mimovl : clip_graph {
    clip_graph_mimovl(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
    // Force F32 mat-mul accumulation to avoid F16 overflow in the FFN down-proj
    // when the mmproj is stored in F16 (the source weights are BF16; downcasting
    // to F16 reduces dynamic range below the SwiGLU output magnitude on the last few layers).
    ggml_tensor * build_mm(ggml_tensor * w, ggml_tensor * x) const override;
};

struct clip_graph_step3vl : clip_graph {
    clip_graph_step3vl(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_youtuvl : clip_graph {
    clip_graph_youtuvl(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_yasa2 : clip_graph {
    clip_graph_yasa2(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;

    ggml_tensor * layer_norm_channels(ggml_tensor * inp, ggml_tensor * w, ggml_tensor * b, float eps = 1e-6f);
    ggml_tensor * convnext_grn(ggml_tensor * inp, ggml_tensor * w, ggml_tensor * b);
};

struct clip_graph_minicpmv : clip_graph {
    clip_graph_minicpmv(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_minicpmv4_6 : clip_graph {
    clip_graph_minicpmv4_6(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_internvl : clip_graph {
    clip_graph_internvl(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
    bool support_batch() const override { return true; }
};

struct clip_graph_nemotron_v2_vl : clip_graph {
    clip_graph_nemotron_v2_vl(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_nemotron_omni : clip_graph {
    clip_graph_nemotron_omni(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
    ggml_tensor * build_inp_with_temporal_merge();
};

struct clip_graph_llama4 : clip_graph {
    clip_graph_llama4(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_kimivl : clip_graph {
    clip_graph_kimivl(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_paddleocr : clip_graph {
    clip_graph_paddleocr(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_dotsocr : clip_graph {
    clip_graph_dotsocr(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_dots3note_a : clip_graph {
    clip_graph_dots3note_a(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_cogvlm : clip_graph {
    clip_graph_cogvlm(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_llava : clip_graph {
    clip_graph_llava(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_whisper_enc : clip_graph {
    clip_graph_whisper_enc(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

// causal whisper-like encoder with a sliding window KV cache (voxtral realtime)
struct clip_graph_voxtral_rt : clip_graph {
    clip_subctx_enc_state * state; // nullptr during warmup, the cache is then a graph input
    clip_graph_voxtral_rt(clip_subcontext * ctx, const clip_image_f32 & img, clip_subctx_enc_state * state) : clip_graph(ctx, img), state(state) {}
    ggml_cgraph * build() override;
};

struct clip_graph_deepseekocr : clip_graph {
    clip_graph_deepseekocr(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
    ggml_tensor * build_sam(ggml_tensor * inp); // build the SAM model
    bool support_batch() const override { return true; }
};

struct clip_graph_deepseekocr2 : clip_graph_deepseekocr {
    clip_graph_deepseekocr2(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph_deepseekocr(ctx, img) {}
    ggml_cgraph * build() override; // reuses build_sam() from base
    bool support_batch() const override { return true; }
};

struct clip_graph_conformer : clip_graph {
    clip_graph_conformer(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

// builders shared by the chatterbox gen and spkenc graphs, defined in chatterbox-gen.cpp
struct clip_graph_chatterbox_base : clip_graph {
    using clip_graph::clip_graph;

  protected:
    ggml_tensor * cbx_linear(ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) const;
    ggml_tensor * cbx_conv1d(ggml_tensor * k, ggml_tensor * b, ggml_tensor * x,
                             int stride, int pad_l, int pad_r) const;
    ggml_tensor * cbx_conv1d_dil(ggml_tensor * k, ggml_tensor * b, ggml_tensor * x,
                                 int pad, int dil) const;
};

struct clip_graph_chatterbox_spkenc : clip_graph_chatterbox_base {
    clip_graph_chatterbox_spkenc(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph_chatterbox_base(ctx, img) {}
    ggml_cgraph * build() override;

  private:
    ggml_tensor * bn1d(const clip_chatterbox::bn & n, ggml_tensor * x);
    ggml_tensor * bn2d_relu(const clip_chatterbox::bn & n, ggml_tensor * x);
    ggml_tensor * res2d(const clip_chatterbox::spk_res2d & r, ggml_tensor * x, int stride);
    ggml_tensor * cam_layer(const clip_chatterbox::spk_cam_layer & l, ggml_tensor * x, int dil, ggml_tensor * segfix);
};

struct clip_graph_chatterbox : clip_graph_chatterbox_base {
    clip_subctx_gen_process gen_process = CLIP_SUBCTX_GEN_PROCESS_GEN_CODE;
    int n_tokens = 0;
    int n_prompt_mel  = 0;
    int vocode_n_mel  = 0;
    int vocode_n_stft = 0;
    clip_graph_chatterbox(clip_subcontext * ctx, const clip_image_f32 & img, clip_subctx_gen_process gen_process, int n_tokens,
                          int n_prompt_mel, int vocode_n_mel, int vocode_n_stft)
        : clip_graph_chatterbox_base(ctx, img), gen_process(gen_process), n_tokens(n_tokens), n_prompt_mel(n_prompt_mel),
          vocode_n_mel(vocode_n_mel), vocode_n_stft(vocode_n_stft) {}
    ggml_cgraph * build() override;

  private:
    ggml_tensor * enc_layer(const clip_chatterbox::enc_layer & l, ggml_tensor * x, ggml_tensor * pos, int T);
    ggml_tensor * causal_block(const clip_chatterbox::causal_block & b, ggml_tensor * x);
    ggml_tensor * resnet(const clip_chatterbox::resnet & r, ggml_tensor * x, ggml_tensor * temb);
    ggml_tensor * tfm_block(const clip_chatterbox::tfm_block & b, ggml_tensor * x);
    ggml_tensor * estimator(ggml_tensor * x_noise, ggml_tensor * mu, ggml_tensor * spks,
                            ggml_tensor * cond, ggml_tensor * temb, int T);
    ggml_tensor * hift_resblock(const clip_chatterbox::hift_res & r, ggml_tensor * x);
    ggml_cgraph * build_s3tok(int T);
    ggml_cgraph * build_vocoder(int n_mel, int n_stft);
};

struct clip_graph_kokoro : clip_graph {
    enum stage_type {
        STAGE_TEXT, // tokens -> text encoder CNN
        STAGE_F0N,  // shared LSTM output -> F0 and energy curves
        STAGE_DEC,  // aligned features + source STFT -> STFT coefficients
    };
    static constexpr int n_style = 128; // a voice row is the decoder style, then the predictor style

    stage_type stage;
    int n_tok    = 0;
    int n_frames = 0;
    int n_har    = 0;
    clip_graph_kokoro(clip_subcontext * ctx, const clip_image_f32 & img, stage_type stage, int n_tok, int n_frames, int n_har)
        : clip_graph(ctx, img), stage(stage), n_tok(n_tok), n_frames(n_frames), n_har(n_har) {}
    ggml_cgraph * build() override;

    // host side, between the graphs
    // bert [n_tok, n_embd], text_cnn [n_tok, 512], style is the predictor style
    // fills shared and asr as [n_frames, 512], returns n_frames
    static int prosody(const clip_model & model, const float * bert, int n_embd, const float * text_cnn, int n_tok,
                       const float * style, float speed, std::vector<float> & shared, std::vector<float> & asr);
    // fills har as [n_har, 22], returns n_har
    static int har_source(const clip_model & model, const std::vector<float> & f0, std::mt19937 & rng, std::vector<float> & har);
    static void istft(const std::vector<float> & spec, std::vector<float> & audio);

  private:
    ggml_tensor * tensor(const std::string & name) const;
    bool          has_tensor(const std::string & name) const;
    ggml_tensor * conv(const std::string & name, ggml_tensor * x, int stride, int pad, int dil) const;
    ggml_tensor * adain(const std::string & name, ggml_tensor * x, ggml_tensor * style) const;
    ggml_tensor * snake(const std::string & name, ggml_tensor * x) const;
    ggml_tensor * interleave(ggml_tensor * a, ggml_tensor * b) const;
    ggml_tensor * adain_resblk(const std::string & name, ggml_tensor * x, ggml_tensor * style, bool upsample) const;
    ggml_tensor * gen_resblock(const std::string & name, ggml_tensor * x, ggml_tensor * style) const;
    ggml_tensor * input(const char * name, ggml_type type, int64_t ne0, int64_t ne1) const;
    ggml_cgraph * output(const char * name, ggml_tensor * t) const;
    ggml_cgraph * build_text();
    ggml_cgraph * build_f0n();
    ggml_cgraph * build_dec();
};

struct clip_graph_lfm3a : clip_graph {
    clip_graph_lfm3a(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_granite_speech : clip_graph {
    clip_graph_granite_speech(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_vibevoice_asr : clip_graph {
    clip_graph_vibevoice_asr(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;

    std::map<std::string, ggml_tensor *> state_in;
    mutable std::vector<std::pair<std::string, ggml_tensor *>> state_out;

    ggml_tensor * with_left_ctx(ggml_tensor * x, const std::string & name) const;
    ggml_tensor * causal_conv(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int stride, const std::string & name) const;
    ggml_tensor * encode(int e, ggml_tensor * x) const;
};

struct clip_graph_muse_spark_audio : clip_graph {
    clip_graph_muse_spark_audio(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_gemma4a : clip_graph {
    // attention_chunk_size, attention_context_left - 1
    static constexpr int64_t chunk_size = 12;
    static constexpr int64_t max_past   = 12;

    clip_graph_gemma4a(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
    ggml_tensor * build_mm(ggml_tensor * w, ggml_tensor * x) const override;
};

struct clip_graph_gemma4ua : clip_graph {
    clip_graph_gemma4ua(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_glm4v : clip_graph {
    clip_graph_glm4v(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_hunyuanvl : clip_graph {
    clip_graph_hunyuanvl(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_mobilenetv5 : clip_graph {
    clip_graph_mobilenetv5(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;

    ggml_tensor * rms_norm_2d(
        ggml_tensor * inp,
        ggml_tensor * weight,
        float eps = 1e-6f);

    ggml_tensor* pad_same_2d(
        ggml_tensor* inp,
        int kernel_h,
        int kernel_w,
        int stride_h,
        int stride_w,
        int dilation_h = 1,
        int dilation_w = 1);

    ggml_tensor * build_edge_residual(
        ggml_tensor * inp,
        const mobilenetv5_block & block,
        int stride);

    ggml_tensor * build_inverted_residual(
        ggml_tensor * inp,
        const mobilenetv5_block & block,
        int stride);

    ggml_tensor * build_mobilenet_attn(
        ggml_tensor * inp,
        const mobilenetv5_block & block);
};

struct clip_graph_qwen3a : clip_graph {
    clip_graph_qwen3a(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_mimo_audio : clip_graph {
    clip_graph_mimo_audio(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

// zonos3 speaker encoder: MECT-B2 + ASTP, then the speaker projection of the backbone
struct clip_graph_zonos3_spkenc : clip_graph {
    clip_graph_zonos3_spkenc(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;

    ggml_tensor * W(const std::string & name) const;
    bool          has(const std::string & name) const;
    ggml_tensor * chan(ggml_tensor * v) const;
    ggml_tensor * bn(ggml_tensor * x, const std::string & p) const;
    ggml_tensor * conv(ggml_tensor * x, const std::string & p, int s_t, int s_f, bool bias, int pad) const;
    ggml_tensor * grouped3x3(ggml_tensor * x, const std::string & p, int s_t, int s_f) const;
    ggml_tensor * res_block(ggml_tensor * x, const std::string & q, int s_f, int s_t) const;
    ggml_tensor * linear(ggml_tensor * x, const std::string & p) const;
    ggml_tensor * layer_norm(ggml_tensor * x, const std::string & p) const;
    ggml_tensor * flatten(ggml_tensor * x) const;
    ggml_tensor * unflatten(ggml_tensor * x, int64_t F, int64_t C) const;
    ggml_tensor * attention(ggml_tensor * h, const std::string & p) const;
    ggml_tensor * moe_ffn(ggml_tensor * h, const std::string & p) const;
    ggml_tensor * pos_conv(ggml_tensor * h, const std::string & p) const;
};

struct clip_graph_qwen3tts_spkenc : clip_graph {
    clip_graph_qwen3tts_spkenc(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;

    ggml_tensor * conv1d_same(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int dilation) const;
    ggml_tensor * res2net(ggml_tensor * x, const clip_layer & layer, int dilation, int scale) const;
    ggml_tensor * se_block(ggml_tensor * x, const clip_layer & layer) const;
    ggml_tensor * se_res2net_block(ggml_tensor * x, const clip_layer & layer, int dilation, int scale) const;
    ggml_tensor * attentive_stats_pool(ggml_tensor * x) const;
};

// the code predictor and the codec of Qwen3-TTS, the building blocks of the Fish S2 and Drama 3 graphs; as a graph
// of the engine, the reference encoder only (Qwen3-TTS generates on native graphs, see llama_model_clip)
struct clip_graph_qwen3tts_gen : clip_graph {
    clip_graph_qwen3tts_gen(clip_subcontext * ctx, const clip_image_f32 & img, clip_subctx_gen_process gen_process)
        : clip_graph(ctx, img), gen_process(gen_process) {}
    ggml_cgraph * build() override;

    // CLIP_SUBCTX_GEN_PROCESS_TOKENIZE: reference clip -> codes and their talker input rows
    ggml_cgraph * build_ref_encoder();
    ggml_tensor * nearest_code(ggml_tensor * x, ggml_tensor * codebook) const;

    // reference encoder transformer, see encoder_config in speech_tokenizer/config.json
    static constexpr int   ref_enc_n_head     = 8;
    static constexpr float ref_enc_eps        = 1e-5f;
    static constexpr float ref_enc_rope_theta = 10000.0f;

    clip_subctx_gen_process gen_process;

};

// the blocks of the Qwen3-TTS code predictor and codec, on the weights of a clip_model: the native Qwen3-TTS
// graphs of libllama and the Fish S2, Drama 3 and Minimax-Music3 graphs of the engine build on them
struct clip_tts_blocks {
    ggml_context * ctx0;
    ggml_cgraph  * gf;
    const clip_model   & model;
    const clip_hparams & hparams;
    const clip_flash_attn_type flash_attn_type;

    clip_tts_blocks(ggml_context * ctx0, ggml_cgraph * gf, const clip_model & model, bool flash_attn)
        : ctx0(ctx0), gf(gf), model(model), hparams(model.hparams),
          flash_attn_type(flash_attn ? CLIP_FLASH_ATTN_TYPE_ENABLED : CLIP_FLASH_ATTN_TYPE_DISABLED) {}
    explicit clip_tts_blocks(const clip_graph & g)
        : clip_tts_blocks(g.ctx0, g.gf, g.model, g.flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) {}

    void          cb(ggml_tensor * cur, const char * name, int il) const;
    // attention, then the output projection, as clip_graph::build_attn()
    ggml_tensor * build_attn(ggml_tensor * wo, ggml_tensor * wo_b, ggml_tensor * q_cur, ggml_tensor * k_cur,
                             ggml_tensor * v_cur, ggml_tensor * kq_mask, float kq_scale, int il) const;

    //
    // code predictor: backbone hidden state + sampled code0 -> 16 RVQ codes, one token per codebook
    //
    int   top_k = 50;
    float top_p = 1.0f;

    ggml_tensor * cache_set(ggml_tensor * cache, int row_idx, ggml_tensor * value) const;
    ggml_tensor * do_sampling(ggml_tensor * logits, ggml_tensor * inp_rand) const;

    ggml_tensor * const_i32(ggml_tensor * anchor, float value) const;
    ggml_tensor * causal_mask_row(int64_t n_kv_pad, int pos) const;
    ggml_tensor * project_in(ggml_tensor * cur) const;

    ggml_tensor * layer_forward(
            ggml_tensor * cur,
            const clip_layer & layer,
            ggml_tensor * inp_pos,
            ggml_tensor * kq_mask,
            ggml_tensor *& k_cache_layer,
            ggml_tensor *& v_cache_layer,
            int64_t n_kv_pad,
            int pos,
            int il) const;

    void prefill(
            std::vector<ggml_tensor *> & k_cache,
            std::vector<ggml_tensor *> & v_cache,
            ggml_tensor *& out_code_cache,
            ggml_tensor * h_state,
            ggml_tensor * code0_embd,
            ggml_tensor * inp_rand) const;

    ggml_tensor * step(
            std::vector<ggml_tensor *> & k_cache,
            std::vector<ggml_tensor *> & v_cache,
            ggml_tensor * out_code_cache,
            ggml_tensor * inp_rand,
            int step_idx) const;

    //
    // codec: RVQ codes -> raw PCM (quantizer + pre_conv + pre_transformer + upsample + DAC)
    //

    // state_in: previous call's persisted state, by slot name (see list_c2w_state_slots())
    std::map<std::string, ggml_tensor *> state_in;
    // state_out: this call's state to persist, by slot name; the caller makes them graph outputs
    mutable std::vector<std::pair<std::string, ggml_tensor *>> state_out;

    // stateful conv ops: read/update their state via state_in/state_out[state_name]
    ggml_tensor * causal_conv1d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int dilation, const std::string & state_name) const;
    ggml_tensor * causal_conv1d_dw(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, const std::string & state_name) const;
    ggml_tensor * causal_conv_transpose1d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int stride, const std::string & state_name) const;
    ggml_tensor * snake(ggml_tensor * x, ggml_tensor * alpha, ggml_tensor * beta) const;

    ggml_tensor * quant_decode(ggml_tensor * inp_codes) const;
    // window <= 0 uses hparams.wav_tfm_swa
    ggml_tensor * tfm_layer_forward(ggml_tensor * cur, const clip_layer & layer, int il, int64_t window = 0) const;
    ggml_tensor * convnext_block(ggml_tensor * x, const clip_code2wav::upsample_block & blk, const std::string & state_prefix) const;
    ggml_tensor * dac_res_unit(ggml_tensor * x, const clip_code2wav::dac_res & res, int dilation, const std::string & state_name) const;

    // drama3: erf GELU in ConvNeXt, not the tanh approximation
    bool gelu_erf = false;

    // inp_codes [1, n_codes] I32 -> this frame's audio samples [n_samples] F32, clamped to [-1, 1]
    ggml_tensor * decode(ggml_tensor * inp_codes) const;
};

//
// pocket-tts: SEANet convolution stack, shared by the voice encoder and the mimi decoder.
// stateless unless state_in is populated: convs then pad instead of carrying left-context.
//
struct clip_graph_pockettts_seanet : clip_graph {
    clip_graph_pockettts_seanet(const clip_graph & parent) : clip_graph(parent) {}
    ggml_cgraph * build() override { GGML_ABORT("call encode()/decode() instead"); }

    // qwen3tts reference codes: the output is quantized, so the im2col is F32 (the kernels must be F32 too)
    bool f32_im2col = false;
    // qwen3tts reference codes: pad_replicate also repeats the last frame on the right, like MimiConv1d
    bool pad_replicate_end = false;

    // per-call streaming state, keyed by slot name (see list_pockettts_state_slots)
    std::map<std::string, ggml_tensor *> state_in;
    mutable std::vector<std::pair<std::string, ggml_tensor *>> state_out;

    ggml_tensor * conv1d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int stride, int dilation,
                         bool pad_replicate = false, const std::string & state_name = "") const;
    ggml_tensor * conv_transpose1d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int stride,
                                   const std::string & state_name = "") const;
    ggml_tensor * res_unit(ggml_tensor * x, const clip_seanet::stage & stage, int dilation,
                           const std::string & state_prefix = "") const;

    // x: [T, C] -> [T / hop, dim]
    ggml_tensor * encode(ggml_tensor * x) const;
    // x: [T, dim] -> [T * hop, 1], streams when state_in is populated
    ggml_tensor * decode(ggml_tensor * x) const;
};

// mimi encoder + speaker_proj: reference waveform -> voice conditioning rows
// the qwen3tts reference encoder reuses encode() with its own transformer options
struct clip_graph_pockettts_spkenc : clip_graph {
    clip_graph_pockettts_spkenc(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    clip_graph_pockettts_spkenc(const clip_graph & parent) : clip_graph(parent) {}
    ggml_cgraph * build() override;

    int         tfm_n_head     = n_head;
    float       tfm_eps        = eps;
    float       tfm_rope_theta = hparams.rope_theta;
    int         tfm_rope_type  = GGML_ROPE_TYPE_NORMAL;
    ffn_op_type tfm_ffn_type   = FFN_GELU;

    ggml_tensor * tfm_layer_forward(ggml_tensor * cur, const clip_layer & layer, ggml_tensor * inp_pos, ggml_tensor * kq_mask, int il) const;

    // waveform [n_samples, 1] -> [n_frames, C], at the frame rate after model.downsample_w
    ggml_tensor * encode(const clip_graph_pockettts_seanet & seanet, ggml_tensor * x, const std::vector<clip_layer> & layers);
};

//
// pocket-tts generation:
// GEN_CODE = flow-matching decoder + end-of-speech head, one latent per call
// GEN_WAV  = mimi decoder, a window of latents -> PCM
//
struct clip_graph_pockettts_gen : clip_graph {
    clip_graph_pockettts_gen(clip_subcontext * ctx, const clip_image_f32 & img, clip_subctx_gen_process gen_process, int n_step, int n_frames)
        : clip_graph(ctx, img), gen_process(gen_process), n_step(n_step), n_frames(n_frames) {}
    ggml_cgraph * build() override;

    clip_subctx_gen_process gen_process;
    int n_step;   // lsd_decode steps, fixed at graph-build time
    int n_frames; // GEN_WAV only: number of latents to decode

    // AdaLN modulation: x * (1 + scale) + shift
    ggml_tensor * modulate(ggml_tensor * x, ggml_tensor * shift, ggml_tensor * scale) const;
    ggml_tensor * time_embed(const clip_flow_net::time_embd & te, float t) const;
    ggml_tensor * flow_forward(ggml_tensor * cond, ggml_tensor * x, float s, float t) const;
};

// one persisted state buffer used by code2wav, see qwen3tts-gen.cpp
struct c2w_state_slot {
    std::string name;
    int64_t     ne0;
    int64_t     ne1;
    bool        in_place = false; // graph writes the slot directly, no state_in/state_out copy
};
std::vector<c2w_state_slot> list_c2w_state_slots(const clip_hparams & hparams, const clip_model & model);

// drama3: tower KV slots, then list_c2w_state_slots()
std::vector<c2w_state_slot> list_drama3_state_slots(const clip_hparams & hparams, const clip_model & model);

// same, for the streaming mimi decoder (pocket-tts GEN_WAV)
std::vector<c2w_state_slot> list_pockettts_state_slots(const clip_hparams & hparams, const clip_model & model);

//
// drama3 generation:
// GEN_CODE = one codebook tower row, each layer attends over the whole sequence (KV in gen state)
// GEN_WAV  = Fish DAC decoder, frames -> PCM
//
struct clip_graph_drama3_gen : clip_graph {
    clip_graph_drama3_gen(clip_subcontext * ctx, const clip_image_f32 & img, clip_subctx_gen_process gen_process, clip_subctx_gen_state * gen_state,
                          int n_frames, int top_k, float top_p, float temp)
        : clip_graph(ctx, img), gen_process(gen_process), gen_state(gen_state), n_frames(n_frames), top_k(top_k), top_p(top_p), temp(temp) {}
    ggml_cgraph * build() override;

    // tower KV rows, shared by the prompt and the generated frames
    static constexpr int n_pos_max = 4096;

    clip_subctx_gen_process gen_process;
    clip_subctx_gen_state * gen_state; // nullptr on buffer reserve
    int   n_frames;             // GEN_WAV only
    int   top_k;
    float top_p;
    float temp;

    ggml_cgraph * build_tower();
    ggml_cgraph * build_codec();
    ggml_tensor * tower_layer(ggml_tensor * x, int il, ggml_tensor * code, ggml_tensor * mask, ggml_tensor * extra,
                              ggml_tensor * inp_pos, ggml_tensor * k_slot, ggml_tensor * v_slot, int pos);
};

//
// fish_s2 generation:
// GEN_CODE = fast AR, backbone hidden + semantic code -> other codes of the frame + next backbone input
// GEN_WAV  = Fish DAC decoder, same as drama3
//
struct clip_graph_fish_s2_gen : clip_graph_drama3_gen {
    using clip_graph_drama3_gen::clip_graph_drama3_gen;
    ggml_cgraph * build() override;

    ggml_cgraph * build_fast_ar();
};

// fish_s2 voice cloning: raw 44.1 kHz clip -> codec encoder + RVQ -> one backbone input row per frame
struct clip_graph_fish_s2_spkenc : clip_graph {
    clip_graph_fish_s2_spkenc(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;

    // codec encoder transformer layers, the rest of model.layers is the quantizer pre_module
    static constexpr int n_enc_tfm_layer = 4;
    static constexpr int n_tfm_layer     = 12;

    ggml_tensor * conv(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int stride, int dilation) const;
    ggml_tensor * snake(ggml_tensor * x, ggml_tensor * alpha, ggml_tensor * beta) const;
    ggml_tensor * convnext(ggml_tensor * x, const clip_code2wav::upsample_block & blk) const;
};

//
// zonos3 generation:
// GEN_CODE = 10 codebook heads, or the backbone row of a sampled frame
// GEN_WAV  = ZAC codec decoder up to the iSTFT spectra
//
struct clip_graph_zonos3_gen : clip_graph {
    clip_graph_zonos3_gen(clip_subcontext * ctx, const clip_image_f32 & img, clip_subctx_gen_process gen_process, bool all_known, int n_frames)
        : clip_graph(ctx, img), gen_process(gen_process), all_known(all_known), n_frames(n_frames) {}
    ggml_cgraph * build() override;

    static constexpr int   n_codebooks = 10;
    static constexpr int   n_cb_vocab  = 4096;
    static constexpr float softcap     = 15.0f;

    // ZAC decoder, fixed by the codec config
    static constexpr int   n_stages              = 5;
    static constexpr int   stage_ctx[n_stages]   = { 150, 300, 480, 480, 480 }; // attention window
    static constexpr int   head_dim              = 64;
    static constexpr float ln_eps                = 1e-5f;
    static constexpr float convnext_eps          = 1e-6f;
    static constexpr int   hop                   = 184;  // at 44160 Hz
    static constexpr int   n_fft[2]              = { 736, 2208 };
    static constexpr int   rs_orig               = 736;  // 44160 -> 44100 Hz
    static constexpr int   rs_new                = 735;
    static constexpr int   samples_per_frame     = 2940;

    clip_subctx_gen_process gen_process;
    bool all_known; // GEN_CODE: only the backbone row
    int  n_frames;  // GEN_WAV

    std::map<std::string, ggml_tensor *> state_in;
    std::vector<std::pair<std::string, ggml_tensor *>> state_out;

    ggml_cgraph * build_gen_code();
    ggml_cgraph * build_gen_wav();
    ggml_tensor * convnext(ggml_tensor * x, const clip_zonos3_wav::convnext & b, const std::string & state_name);
    ggml_tensor * moss_layer(ggml_tensor * x, const clip_zonos3_wav::layer & l, int s, int il, ggml_tensor * base);
};

std::vector<c2w_state_slot> list_zonos3_gen_state_slots(const clip_hparams & hparams, const clip_model & model);

// graphs that decode with a clip_snake_vae (yue2, minimax_music3), see snake-vae.cpp
struct clip_graph_snake_vae : clip_graph {
    using clip_graph::clip_graph;

    // x: [T, C]
    ggml_tensor * snake(ggml_tensor * x, const clip_snake_vae::snake & s) const;
    ggml_tensor * conv1d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int dilation) const;
    // x: [T, n_latent] -> [T * ratio, n_out]
    ggml_tensor * build_snake_vae(ggml_tensor * x, const clip_snake_vae & vae) const;
};

//
// YuE2 generation, the NAR transformer itself runs on a llama_context, see clip_gen_flow_yue2():
// LATENT_IN  = latent rows -> backbone embd rows
// LATENT_OUT = backbone hidden rows -> velocity rows
// GEN_WAV    = Oobleck VAE decoder, latent rows -> interleaved stereo PCM
//
struct clip_graph_yue2_gen : clip_graph_snake_vae {
    clip_graph_yue2_gen(clip_subcontext * ctx, const clip_image_f32 & img, clip_subctx_gen_process gen_process, int n_rows)
        : clip_graph_snake_vae(ctx, img), gen_process(gen_process), n_rows(n_rows) {}
    ggml_cgraph * build() override;

    static constexpr int   n_latent   = 64;
    static constexpr int   n_freq     = 256;   // time embedder input
    static constexpr int   n_channels = 2;
    static constexpr int   vae_ratio  = 1920;  // output samples per latent frame
    static constexpr int   vae_trim   = 64;    // a decode of n frames has vae_ratio * n - vae_trim samples
    static constexpr float rope_theta = 1000000.0f;
    static constexpr int   n_ctx_max  = 24576; // the AR prefix and the NAR rows of one chunk
    static constexpr int   ode_steps  = 32;    // midpoint solver

    clip_subctx_gen_process gen_process;
    int n_rows;
};

//
// MiniMax Music 3 generation, see clip_gen_wav_music3() for the flow-matching loop:
// GEN_CODE  = depth decoder of one frame: backbone hidden rows (cond, then uncond with guidance) + semantic code -> the 8 codes, the next backbone input and the condition row (mix of the 8 hidden states of the frame)
// FLOW_COND = condition rows of a window -> DiT condition, resampled to the latent rate
// FLOW_STEP = one DiT pass, cond and uncond rows as a batch of 2
// GEN_WAV   = DAC decoder, latent rows -> interleaved stereo PCM
//
struct clip_graph_music3_gen : clip_graph_snake_vae {
    clip_graph_music3_gen(clip_subcontext * ctx, const clip_image_f32 & img, clip_subctx_gen_process gen_process,
                          int n_rows, int n_out, float cfg_scale, float temp)
        : clip_graph_snake_vae(ctx, img), gen_process(gen_process), n_rows(n_rows), n_out(n_out), cfg_scale(cfg_scale), temp(temp) {}
    ggml_cgraph * build() override;

    static constexpr int   n_latent   = 128;  // two channels of 64
    static constexpr int   n_channels = 2;
    static constexpr int   vae_ratio  = 512;  // output samples per latent frame
    static constexpr int   dit_d_head = 64;
    static constexpr int   dit_n_rot  = 32;   // partial rotary
    static constexpr float dit_theta  = 10000.0f;
    static constexpr float dit_eps    = 1e-5f;
    static constexpr int   top_k      = 50;

    // sampling and flow matching of the reference (encoders.py, denoise.py)
    static constexpr float depth_cfg  = 1.5f;
    static constexpr float dit_cfg    = 1.7f;
    static constexpr int   flow_steps = 30;
    static constexpr int   n_overlap  = 172; // latent frames shared by neighboring windows

    clip_subctx_gen_process gen_process;
    int   n_rows; // GEN_CODE: backbone rows, FLOW_COND: frames, FLOW_STEP and GEN_WAV: latent frames
    int   n_out;  // FLOW_COND: latent frames
    float cfg_scale;
    float temp;

    static int64_t latent_frames(int64_t n_frames);

    ggml_cgraph * build_depth();
    ggml_cgraph * build_cond();
    ggml_cgraph * build_dit();
    ggml_cgraph * build_vocoder();
};

// tiny-voca: small graphs around the stacks, the stacks run on libllama (see clip_gen_wav_tiny_voca())
// LATENT_IN:  token states [n_embd, n_rows] -> duration outputs [2, n_rows]
// LATENT_OUT: acoustic states [n_embd, n_rows] -> source-filter params [n_voc, n_rows]
struct clip_graph_tiny_voca : clip_graph {
    clip_graph_tiny_voca(clip_subcontext * ctx, const clip_image_f32 & img, clip_subctx_gen_process gen_process, int n_rows)
        : clip_graph(ctx, img), gen_process(gen_process), n_rows(n_rows) {}
    ggml_cgraph * build() override;

    static constexpr int   n_prosody_layer  = 2;
    static constexpr int   n_acoustic_layer = 5;
    static constexpr int   n_max_frames     = 3000;
    static constexpr int   hop              = 256;  // samples per frame
    static constexpr int   n_cep_causal     = 80;
    static constexpr int   n_cep_anti       = 16;
    static constexpr int   n_noise_cep      = 32;
    static constexpr int   harm_fft         = 512;
    static constexpr int   harm_hop         = 128;
    static constexpr int   harm_steps       = hop / harm_hop;
    static constexpr int   noise_fft        = 256;
    static constexpr int   noise_hop        = 64;
    static constexpr int   noise_steps      = hop / noise_hop;
    static constexpr int   n_voc            = n_cep_causal + n_cep_anti + harm_steps + n_noise_cep + noise_steps;
    static constexpr float rope_theta       = 10000.0f;
    static constexpr float sample_rate      = 24000.0f;
    static constexpr float harmonic_limit   = 0.47f;  // highest harmonic, relative to the sample rate
    static constexpr float f0_ref           = 200.0f; // Hz, log F0 is predicted relative to it
    static constexpr float f0_min           = 50.0f;
    static constexpr float f0_max           = 600.0f;
    static constexpr float max_dur          = 400.0f; // frames per token before the speed

    clip_subctx_gen_process gen_process;
    int n_rows;

    // host: frame params [n_frames, n_voc] and F0 [n_frames] in Hz -> waveform [n_frames * hop]
    static void synth(const std::vector<float> & voc, const std::vector<float> & f0, std::mt19937 & rng, std::vector<float> & audio);

    // unknown-word model: letters (a-z and ') -> CMUdict phonemes separated by spaces, other characters are skipped
    static constexpr int n_g2p_slot   = 2;  // positions per letter
    static constexpr int n_g2p_letter = 27;
    static constexpr int n_g2p_symbol = 69;
    static std::string g2p(const clip_tiny_voca & tv, const std::string & word);

  private:
    ggml_tensor * conv3(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) const;
};

struct clip_graph_soprano : clip_graph {
    clip_graph_soprano(clip_subcontext * ctx, const clip_image_f32 & img, int n_frames) : clip_graph(ctx, img), n_frames(n_frames) {}
    ggml_cgraph * build() override;
    static void decode_spectrum(const clip_model & model, const std::vector<float> & spectrum, std::vector<float> & pcm);
    // SopranoDecoder default
    static constexpr int upscale = 4;
    int n_frames;
};

//
// LFM3-Audio generation:
// GEN_CODE = one MaskGIT pass of the depthformer, or only the feedback embedding once all codes are known
// GEN_WAV  = liquid-codec decoder, a window of frames -> polar STFT frames
//
struct clip_graph_lfm3a_gen : clip_graph {
    clip_graph_lfm3a_gen(clip_subcontext * ctx, const clip_image_f32 & img, clip_subctx_gen_process gen_process, bool all_known, int n_frames)
        : clip_graph(ctx, img), gen_process(gen_process), all_known(all_known), n_frames(n_frames) {}
    ggml_cgraph * build() override;

    clip_subctx_gen_process gen_process;
    bool all_known; // GEN_CODE only
    int  n_frames;  // GEN_WAV only

    ggml_tensor * build_gen_code();
    ggml_tensor * build_gen_wav();

    ggml_tensor * build_lfm_layer(ggml_tensor * inpL, const clip_layer & layer, float eps, int il,
                                  const std::function<ggml_tensor *(ggml_tensor *)> & op);
    ggml_tensor * build_lfm_attn(ggml_tensor * cur, const clip_layer & layer, int n_head, int n_head_kv,
                                 float eps, float rope_theta, int il, ggml_tensor * inp_pos,
                                 ggml_tensor * kq_mask, ggml_tensor * k_cache, ggml_tensor * v_cache,
                                 ggml_tensor ** k_all, ggml_tensor ** v_all);
    ggml_tensor * build_lfm_conv(ggml_tensor * cur, const clip_layer & layer, int il,
                                 ggml_tensor * conv_state, ggml_tensor ** conv_state_out);
};

// state slots of the streaming codec decoder (lfm3a_gen GEN_WAV)
std::vector<c2w_state_slot> list_lfm3a_gen_state_slots(const clip_hparams & hparams, const clip_model & model);

// state slots of the streaming LFM3-Audio encoder (lfm3a)
std::vector<c2w_state_slot> list_lfm3a_state_slots(const clip_hparams & hparams, const clip_model & model);

// state slots of the streaming Gemma 4 audio encoder (gemma4a)
std::vector<c2w_state_slot> list_gemma4a_state_slots(const clip_hparams & hparams, const clip_model & model);

// vibevoice-asr: the left context of every causal conv, a zero state is the causal zero padding
std::vector<c2w_state_slot> list_vibevoice_asr_state_slots(const clip_hparams & hparams, const clip_model & model);

struct clip_graph_kimik25 : clip_graph {
    clip_graph_kimik25(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;

    ggml_tensor * resize_position_embeddings_3d(uint32_t interpolation_mode);
};

struct clip_graph_parakeet : clip_graph {
    clip_graph_parakeet(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_exaone4_5 : clip_graph {
    clip_graph_exaone4_5(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_granite4_vision : clip_graph {
    clip_graph_granite4_vision(clip_subcontext * ctx, const clip_image_f32 & img)
        : clip_graph(ctx, img),
          anyres(img.anyres),
          n_tiles(img.ny() / img.nx()),
          tile_side(img.nx() / patch_size) {}

    ggml_cgraph * build() override;

private:
    // the input image is a stack of tiles on the Y axis: [overview, tile(0,0), tile(0,1), ...]
    const clip_image_f32::anyres_info anyres;
    const int n_tiles;
    const int tile_side; // patches per tile side

    ggml_tensor * build_tile_inp();
    ggml_tensor * gather(ggml_tensor * src, const std::string & name, int idx_len);
    ggml_tensor * interp_down(ggml_tensor * src, int side, int new_side);
    ggml_tensor * build_block(const qf_block & blk, ggml_tensor * h, int bid,
                              int spatial_offset, int image_side, int window_side,
                              int query_side, float qformer_eps);

    ggml_tensor * build_newline_row(ggml_context * ctx0);
    ggml_tensor * build_anyres_assembly(ggml_tensor * cur, int out_side);
};

struct clip_graph_muse_glimmer : clip_graph {
    clip_graph_muse_glimmer(clip_subcontext * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_inkling2v : clip_graph {
    using clip_graph::clip_graph;
    ggml_cgraph * build() override;
};
