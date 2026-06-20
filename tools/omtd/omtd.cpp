#include "omtd.h"

#include <cstring>

static void omtd_set_error(char * error, const size_t error_size, const char * msg) {
    if (!error || error_size == 0) {
        return;
    }
    const char * text = msg ? msg : "";
    std::strncpy(error, text, error_size - 1);
    error[error_size - 1] = '\0';
}

static bool omtd_has_text(const char * value) {
    return value && value[0] != '\0';
}

extern "C" bool omtd_is_output_companion_gguf(const char * path) {
    return omtd_get_model_type(path) != OMTD_MODEL_TYPE_UNKNOWN;
}

extern "C" omtd_model_type omtd_get_model_type(const char * path) {
    if (!omtd_has_text(path)) {
        return OMTD_MODEL_TYPE_UNKNOWN;
    }

    // Model-specific OMTD patches add companion GGUF metadata recognizers here.
    return OMTD_MODEL_TYPE_UNKNOWN;
}

extern "C" omtd_modality omtd_get_model_modality(const char * path) {
    switch (omtd_get_model_type(path)) {
        case OMTD_MODEL_TYPE_UNKNOWN:
        default:
            return OMTD_MODALITY_UNKNOWN;
    }
}

extern "C" omtd_status omtd_audio_generate_file(
        const omtd_audio_generation_params * params,
        char * error,
        const size_t error_size) {
    if (!params) {
        omtd_set_error(error, error_size, "missing OMTD audio generation params");
        return OMTD_STATUS_INVALID_PARAM;
    }
    if (!omtd_has_text(params->model_path)) {
        omtd_set_error(error, error_size, "missing backbone model path");
        return OMTD_STATUS_INVALID_PARAM;
    }
    if (!omtd_has_text(params->companion_path)) {
        omtd_set_error(error, error_size, "missing OMTD companion path");
        return OMTD_STATUS_INVALID_PARAM;
    }
    if (!omtd_has_text(params->prompt)) {
        omtd_set_error(error, error_size, "missing audio generation prompt");
        return OMTD_STATUS_INVALID_PARAM;
    }
    if (!omtd_has_text(params->output_path)) {
        omtd_set_error(error, error_size, "missing output path");
        return OMTD_STATUS_INVALID_PARAM;
    }

    omtd_set_error(error, error_size, "no OMTD audio backend is available");
    return OMTD_STATUS_UNSUPPORTED;
}
