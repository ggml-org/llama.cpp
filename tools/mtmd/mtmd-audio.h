#pragma once

#include "ggml.h"

#include <cstdint>
#include <vector>
#include <string>

#define WHISPER_ASSERT GGML_ASSERT

#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT       400
#define WHISPER_HOP_LENGTH  160
#define WHISPER_CHUNK_SIZE  30

#define COMMON_SAMPLE_RATE 16000

namespace whisper_preprocessor {

struct whisper_mel {
    int n_len;
    int n_len_org;
    int n_mel;

    std::vector<float> data;
};

struct whisper_filters {
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};

extern bool preprocess_audio(
        const float * samples,
        size_t n_samples,
        const whisper_filters & filters,
        std::vector<whisper_mel> & output);

} // namespace whisper_preprocessor


// TODO @ngxson : move this helper to mtmd-helpers.cpp
namespace audio_helpers {

extern bool is_audio_file(const char * buf, size_t len);

extern bool decode_audio_from_buf(
        const unsigned char * buf_in,
        size_t len,
        int target_sampler_rate,
        std::vector<float> & pcmf32_mono);

} // namespace audio_helpers


namespace whisper_precalc_filters {

extern whisper_preprocessor::whisper_filters get_128_bins();

} // namespace whisper_precalc_filters
