#pragma once

// media generation (image / video / audio) endpoints backed by libmediagen

#include "server-http.h"
#include "common.h"

#include <mutex>
#include <string>
#include <vector>

struct mediagen_context;
struct mediagen_gen_params;
struct mediagen_result;
struct llama_model;

struct server_mediagen {
    server_mediagen();
    ~server_mediagen();

    // true if the model in params is a latent diffusion model handled by this module
    static bool is_diffusion_model(const common_params & params);

    bool load(const common_params & params);
    void unload();
    bool is_loaded() const { return ctx != nullptr; }

    // OpenAI compatible endpoints
    server_http_context::handler_t post_images_generations; // POST /v1/images/generations
    server_http_context::handler_t post_videos;             // POST /v1/videos
    server_http_context::handler_t get_video;               // GET  /v1/videos/:id
    server_http_context::handler_t get_video_content;       // GET  /v1/videos/:id/content
    server_http_context::handler_t post_audio_speech;       // POST /v1/audio/speech
    server_http_context::handler_t get_models;              // GET  /v1/models
    server_http_context::handler_t get_props;               // GET  /props
    server_http_context::handler_t not_supported;           // any text endpoint

private:
    // completed videos, most recent last
    struct video_job {
        std::string id;
        std::string meta;   // job object json
        std::string mp4;    // empty when ffmpeg is unavailable
    };
    std::vector<video_job> videos;
    std::mutex videos_mutex;

    mediagen_result *   generate(const mediagen_gen_params & gp);
    server_http_res_ptr get_video_res(const std::string & id, bool content);

    mediagen_context * ctx = nullptr;
    llama_model * text_model = nullptr;
    std::mutex mutex; // one generation at a time
    common_params params;
    std::string model_name;
};
