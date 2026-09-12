#ifndef MEDIAGEN_H
#define MEDIAGEN_H

#include "ggml.h"
#include "llama.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// libmediagen: image, video and audio generation with latent diffusion models
// experimental API, see mediagen-cli.cpp for usage

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define MEDIAGEN_API __declspec(dllexport)
#        else
#            define MEDIAGEN_API __declspec(dllimport)
#        endif
#    else
#        define MEDIAGEN_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define MEDIAGEN_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct mediagen_context;

enum mediagen_arch {
    MEDIAGEN_ARCH_UNKNOWN = 0,
    MEDIAGEN_ARCH_LTXV    = 1, // Lightricks LTX-2.x audio-video DiT
};

struct mediagen_context_params {
    const char * model_path;          // diffusion transformer GGUF (required)
    const char * vae_path;            // video VAE (safetensors or GGUF)
    const char * audio_vae_path;      // audio VAE + vocoder (safetensors), optional
    const char * text_proj_path;      // text embedding projection ("embeddings connectors"), optional if bundled
    struct llama_model * text_model;  // text encoder loaded via libllama (required)
    bool         use_gpu;
    int          n_threads;
    bool         flash_attn;
    ggml_log_level verbosity;
};

MEDIAGEN_API struct mediagen_context_params mediagen_context_params_default(void);

// returns nullptr on failure, error is logged
MEDIAGEN_API struct mediagen_context * mediagen_init(struct mediagen_context_params params);
MEDIAGEN_API void                      mediagen_free(struct mediagen_context * ctx);

MEDIAGEN_API enum mediagen_arch mediagen_get_arch(const struct mediagen_context * ctx);
MEDIAGEN_API bool mediagen_supports_image(const struct mediagen_context * ctx);
MEDIAGEN_API bool mediagen_supports_video(const struct mediagen_context * ctx);
MEDIAGEN_API bool mediagen_supports_audio(const struct mediagen_context * ctx);

// returns true if the GGUF at path is a diffusion model handled by this library
MEDIAGEN_API bool mediagen_is_diffusion_model(const char * path);

// progress callback: called after each denoising step, return false to cancel
typedef bool (*mediagen_progress_cb)(int step, int n_steps, void * user_data);

struct mediagen_gen_params {
    const char * prompt;
    const char * negative_prompt;   // used when cfg_scale > 1
    int          width;             // pixels, multiple of 32
    int          height;            // pixels, multiple of 32
    int          n_frames;          // 1 for an image, 8k+1 for video
    float        fps;               // video frame rate
    int          n_steps;           // 0 = model default
    float        cfg_scale;         // 1.0 = no guidance
    uint32_t     seed;              // UINT32_MAX = random
    bool         gen_audio;         // also generate the audio track (video only)
    bool         enhance_prompt;    // expand the prompt into a detailed caption with the text model first
    mediagen_progress_cb progress;
    void *       progress_user_data;
};

MEDIAGEN_API struct mediagen_gen_params mediagen_gen_params_default(void);

// a decoded media output
struct mediagen_result {
    // video: n_frames frames of width*height RGB8 pixels, frame-major
    int       width;
    int       height;
    int       n_frames;
    float     fps;
    uint8_t * rgb;          // width*height*3*n_frames bytes
    // audio: interleaved float PCM
    int       sample_rate;
    int       n_channels;
    int64_t   n_samples;    // per channel
    float *   pcm;
    // the prompt actually used (the enhanced one when enhance_prompt was set)
    char *    revised_prompt;
};

// returns nullptr on failure or cancellation, error is logged
MEDIAGEN_API struct mediagen_result * mediagen_generate(struct mediagen_context * ctx, const struct mediagen_gen_params * params);
MEDIAGEN_API void                     mediagen_result_free(struct mediagen_result * res);

// encode one RGB8 frame as PNG, returns malloc'd buffer (caller frees with free())
MEDIAGEN_API uint8_t * mediagen_encode_png(const uint8_t * rgb, int width, int height, size_t * n_bytes);
// encode interleaved float PCM as 16-bit WAV, returns malloc'd buffer
MEDIAGEN_API uint8_t * mediagen_encode_wav(const float * pcm, int64_t n_samples, int n_channels, int sample_rate, size_t * n_bytes);
// mux the frames (and audio) of a result into an mp4 (h264 + aac) with the ffmpeg binary in PATH
// returns nullptr when ffmpeg is unavailable; caller frees with free()
MEDIAGEN_API uint8_t * mediagen_encode_mp4(const struct mediagen_result * res, size_t * n_bytes);
MEDIAGEN_API bool      mediagen_has_ffmpeg(void);

#ifdef __cplusplus
}
#endif

#endif // MEDIAGEN_H
