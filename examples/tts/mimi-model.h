#pragma once

#include "ggml.h"
#include <memory>
#include <vector>

struct mimi_ggml_ctx;
struct mimi_encoder_decoder;
struct mimi_transformer;
struct mimi_residual_vector_quantizer;

struct mimi_model {
    bool verbose = false;
    std::unique_ptr<mimi_ggml_ctx> ctx;

    std::unique_ptr<mimi_encoder_decoder>           seanet_dec;
    std::unique_ptr<mimi_transformer>               transformer_dec;
    std::unique_ptr<mimi_residual_vector_quantizer> quantizer;

    mimi_model(const char * fname, bool verbose = false);
    ~mimi_model();

    int get_sample_rate() const;

    // layout of codes: (1 semantic code followed by 31 acoustic codes) repeast N times
    std::vector<float> decode(const std::vector<int> & codes);

    // TODO: implement encoding pass
    // std::vector<int> encode(const std::vector<float> & wav_data);

private:
    std::vector<float> decode_frame(const std::vector<int> & codes, int & n_past);

    // transpose layout (from streaming layout to non-streaming):
    // - from: (1 semantic code followed by 31 acoustic codes) repeast N times
    // - to:   N semantic codes followed by (N*31) acoustic codes
    // streaming layout is 1-31, 1-31, 1-31, ..., used for real-time processing
    static std::vector<int> transpose_input(const std::vector<int> & codes);
};
