#pragma once

#include "ggml.h"

#include <cstdint>
#include <vector>

#define WHISPER_ASSERT GGML_ASSERT

namespace whisper_preprocessor {

struct whisper_mel {
    int n_len;
    int n_len_org;
    int n_mel;

    std::vector<float> data;
};

struct whisper_filter_params {
    int32_t n_mel;
    int32_t n_fft_bins;
    int32_t hann_window_size;
    int32_t hop_length;
    int32_t sample_rate;
    bool    center_padding = false;
    float   preemph = 0.f;
    bool    use_natural_log = false;
    bool    normalize_per_feature = false;
    bool    need_chunking = true;

    std::vector<float> mel_filters;
    std::vector<float> hann_window;
    std::vector<float> sin_vals;
    std::vector<float> cos_vals;
};

bool preprocess_audio(
        const float * samples,
        size_t n_samples,
        const whisper_filter_params & filters,
        std::vector<whisper_mel> & output);

} // namespace whisper_preprocessor

namespace whisper_precalc_filters {

whisper_preprocessor::whisper_filter_params get_whisper_params(
        int32_t n_mel,
        int32_t n_fft,
        int32_t window_size,
        int32_t hop_length,
        int32_t sample_rate);

} // namespace whisper_precalc_filters
