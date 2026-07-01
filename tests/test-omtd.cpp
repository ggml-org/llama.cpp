#include "omtd.h"

#include <cassert>
#include <cstring>

int main() {
    char error[128] = {};

    assert(!omtd_is_output_companion_gguf(nullptr));
    assert(omtd_get_model_type(nullptr) == OMTD_MODEL_TYPE_UNKNOWN);
    assert(omtd_get_model_modality(nullptr) == OMTD_MODALITY_UNKNOWN);

    assert(omtd_audio_generate_file(nullptr, error, sizeof(error)) == OMTD_STATUS_INVALID_PARAM);
    assert(std::strlen(error) > 0);

    omtd_audio_generation_params params = {};
    assert(omtd_audio_generate_file(&params, error, sizeof(error)) == OMTD_STATUS_INVALID_PARAM);

    params.model_path = "model.gguf";
    params.companion_path = "companion.gguf";
    params.prompt = "hello";
    params.output_path = "out.wav";
    assert(omtd_audio_generate_file(&params, error, sizeof(error)) == OMTD_STATUS_UNSUPPORTED);

    return 0;
}
