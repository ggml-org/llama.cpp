#pragma once

#include "mtmd.h"

#include <cstdint>
#include <memory>
#include <vector>

enum class mtmd_audio_decoder_type {
    LFM25,
    OTHER,
};

struct mtmd_audio_decode_result {
    std::vector<int16_t> pcm16;
    std::vector<float>   embedding;
    bool                 is_final = true;
};

struct mtmd_audio_decoder {
    virtual ~mtmd_audio_decoder() = default;

    virtual void reset() = 0;

    virtual mtmd_audio_decoder_type get_type() = 0;

    virtual int get_sample_rate() const = 0;

    virtual int decode(mtmd_audio_decode_result & result,
                       const float *              embd,
                       size_t                     n_embd,
                       float                      temperature,
                       int                        top_k) = 0;

    // returns next modality after text token
    virtual mtmd_output_modality accept_text_token(llama_token token) = 0;

    virtual void set_modalities(const std::vector<mtmd_output_modality> & modalities) = 0;
};

using mtmd_audio_decoder_ptr = std::unique_ptr<mtmd_audio_decoder>;

struct llama_model;
mtmd_audio_decoder_ptr mtmd_audio_decoder_create(const llama_model * text_model,
                                                 const std::string & vocoder_path,
                                                 const std::string & tokenizer_path,
                                                 int                 n_threads,
                                                 bool                use_gpu);
