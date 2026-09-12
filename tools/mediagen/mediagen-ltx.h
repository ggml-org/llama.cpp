#pragma once

// LTX-2.x audio-video diffusion transformer, video VAE and audio VAE

#include "mediagen-impl.h"

struct ltx_hparams {
    // transformer
    int   n_layers          = 48;
    int   v_dim             = 4096;
    int   v_heads           = 32;
    int   v_head_dim        = 128;
    int   a_dim             = 2048;
    int   a_heads           = 32;
    int   a_head_dim        = 64;
    int   in_channels       = 128;
    int   audio_in_channels = 128;
    int   caption_channels  = 3840;
    int   connector_layers  = 8;
    int   connector_regs    = 128;
    int   connector_max_pos = 4096;
    int   text_max_tokens   = 1024;
    float rope_theta        = 10000.0f;
    float max_pos[3]        = { 20.0f, 2048.0f, 2048.0f };
    float audio_max_pos     = 20.0f;
    float timestep_scale    = 1000.0f;
    float av_ca_timestep_scale = 1000.0f;
    bool  causal_temporal_positioning = true;
    bool  cross_attention_adaln = true;
    bool  gated_attention   = true;
    bool  has_audio         = true;
    // audio latent geometry
    int   audio_channels    = 8;
    int   audio_freq_bins   = 16;
    int   audio_sample_rate = 16000;
    int   audio_hop_length  = 160;
    int   audio_latent_downsample = 4;
    // vae
    int   latent_channels   = 128;
    int   vae_scale_t       = 8;
    int   vae_scale_s       = 32;
    int   vae_patch_size    = 4;
    // text encoder
    int   text_hidden_size  = 3840;
    int   text_n_states     = 49;
};

// text conditioning: per-modality caption embeddings after the connectors
struct ltx_text_cond {
    int n_tokens = 0;           // == hparams.text_max_tokens
    std::vector<float> video;   // [n_tokens][v_dim]
    std::vector<float> audio;   // [n_tokens][a_dim]
};

struct ltx_model {
    ltx_hparams hp;
    mg_weights  dit;        // transformer (+ connectors, + text projection if bundled)
    mg_weights  text_proj;  // text_embedding_projection.* (may be empty if bundled in dit)
    mg_weights  vae;        // video vae decoder
    mg_weights  audio_vae;  // audio vae decoder + vocoder
    bool has_video_vae = false;
    bool has_audio_vae = false;
    bool has_text_proj = false;

    ggml_tensor * tp(const std::string & name, bool required = true) const {
        if (text_proj.has(name)) return text_proj.get(name);
        return dit.get(name, required);
    }
};

// parse the `config` json of the gguf into hparams
bool ltx_load_hparams(const std::string & config_json, ltx_hparams & hp);

//
// text encoder
//

struct ltx_text_encoder {
    llama_model *   model = nullptr;
    llama_context * lctx  = nullptr;
    void *          cur_capture = nullptr; // capture state for the eval callback during encode()
    int n_ctx = 1024;
    bool init(llama_model * model, int n_ctx, int n_threads, bool flash_attn);
    void free();
    // run the text model and return the packed, per-token RMS-normalized hidden states:
    // out[t * (H * L) + h * L + l] with H = hidden size, L = number of hidden states
    bool encode(const std::string & prompt, int max_tokens, int & n_tokens, std::vector<float> & packed, int & n_hidden, int & n_states);
    // expand a short prompt into a detailed caption with the text model itself
    bool enhance(const std::string & prompt, uint32_t seed, int max_new_tokens, std::string & out);
};

// tokenize with the special tokens added, empty on failure
std::vector<llama_token> ltx_tokenize(const llama_vocab * vocab, const std::string & text, bool parse_special);

// text projection + connectors -> conditioning for both modalities
bool ltx_build_text_cond(const ltx_model & model, mg_backend & be, const std::vector<float> & packed, int n_tokens, int n_hidden, int n_states,
                         bool flash_attn, ltx_text_cond & out);

//
// transformer
//

struct ltx_latent_video {
    int n_frames = 0;  // latent frames
    int height   = 0;  // latent height
    int width    = 0;  // latent width
    int channels = 128;
    // token-major: x[t * channels + c], t = f*H*W + h*W + w
    std::vector<float> x;
    int n_tokens() const { return n_frames * height * width; }
};

struct ltx_latent_audio {
    int n_frames = 0;  // latent time steps
    int channels = 8;
    int freq     = 16;
    // token-major: x[t * (channels * freq) + c * freq + f]
    std::vector<float> x;
    int n_tokens() const { return n_frames; }
};

struct ltx_dit_inputs {
    const ltx_latent_video * video = nullptr;
    const ltx_latent_audio * audio = nullptr;   // nullptr = video only
    const ltx_text_cond *    cond  = nullptr;
    float sigma = 1.0f;        // shared timestep for all tokens
    float fps   = 24.0f;
    bool  flash_attn = true;
};

// one denoising forward pass: returns predicted velocity for video (and audio) tokens
bool ltx_dit_forward(const ltx_model & model, mg_backend & be, const ltx_dit_inputs & in,
                     std::vector<float> & v_out, std::vector<float> & a_out);

//
// video vae
//

// decode latents to RGB float frames in [-1, 1]; out layout: frame-major [f][h][w][3]
bool ltx_vae_decode(const ltx_model & model, mg_backend & be, const ltx_latent_video & lat,
                    int & out_frames, int & out_height, int & out_width, std::vector<float> & rgb);

//
// audio vae + vocoder
//

// decode audio latents to stereo float PCM (interleaved) at the vocoder output rate
bool ltx_audio_decode(const ltx_model & model, mg_backend & be, const ltx_latent_audio & lat,
                      int & sample_rate, int & n_channels, std::vector<float> & pcm);

// number of audio latent frames covering a video of n_frames at fps
int ltx_audio_latent_frames(const ltx_hparams & hp, int n_video_frames, float fps);

// mg_load_opts::transform for the audio vae weights
void ltx_audio_transform(const std::string & name, float * data, int64_t n);

//
// scheduler
//

// distilled models use a fixed schedule; otherwise the LTX2 shifted linear schedule for n steps
std::vector<float> ltx_get_sigmas(int n_steps, int n_tokens, bool distilled);
