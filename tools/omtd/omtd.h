#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define OMTD_API __declspec(dllexport)
#        else
#            define OMTD_API __declspec(dllimport)
#        endif
#    else
#        define OMTD_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define OMTD_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum omtd_modality {
    OMTD_MODALITY_UNKNOWN = 0,
    OMTD_MODALITY_AUDIO   = 1,
    OMTD_MODALITY_IMAGE   = 2,
    OMTD_MODALITY_VIDEO   = 3,
} omtd_modality;

typedef enum omtd_model_type {
    OMTD_MODEL_TYPE_UNKNOWN = 0,
} omtd_model_type;

typedef enum omtd_status {
    OMTD_STATUS_SUCCESS       = 0,
    OMTD_STATUS_INVALID_PARAM = 1,
    OMTD_STATUS_UNSUPPORTED   = 2,
    OMTD_STATUS_RUNTIME_ERROR = 3,
} omtd_status;

typedef struct omtd_audio_generation_params {
    const char * model_path;
    const char * companion_path;
    const char * prompt;
    const char * output_path;

    const char * device;
    const char * omtd_backend;
    const char * rvq_backend;
    const char * vocoder_backend;

    const char * ref_voice_path;
    const char * ref_text_path;

    int32_t n_gpu_layers;
    int32_t n_ctx;

    float duration_seconds;
    float max_duration_seconds;
    float temperature;
    int32_t top_k;
    int32_t seed;
    int32_t stream_stride;
    int32_t stream_holdback;

    bool seed_is_set;
    bool stream_wav;
    bool raw_prompt;
    bool verbose;
    bool flash_attn;
} omtd_audio_generation_params;

// Detects output-modality companion GGUF files. The OMTD framework patch only
// defines the boundary; model-specific recognizers are added by backend patches.
OMTD_API bool omtd_is_output_companion_gguf(const char * path);

OMTD_API omtd_model_type omtd_get_model_type(const char * path);
OMTD_API omtd_modality   omtd_get_model_modality(const char * path);

// Generate an audio file through an output-modality companion. The framework
// returns OMTD_STATUS_UNSUPPORTED until a concrete audio backend is registered.
OMTD_API omtd_status omtd_audio_generate_file(
        const omtd_audio_generation_params * params,
        char * error,
        size_t error_size);

#ifdef __cplusplus
}
#endif
