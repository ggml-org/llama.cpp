#ifndef COGAGENT_H
#define COGAGENT_H

#include "vision_encoder.h"
#include "cross_vision.h"
#include "cogagent_util.h"
#include "image_util.h"
#include "ggml.h"
#include "gguf.h"

struct cogagent_ctx {
    // Vision encoder and cross vision encoder models
    vision_encoder_ctx vision_encoder;
    cross_vision_ctx cross_vision;

    struct llama_context * ctx_llama;
    struct llama_model * cogvlm_model;

    // Context for storing vision tokens and cross vision
    // embedded picture tensor
    ggml_context * token_ctx;

    std::string user_prompt;
    std::vector<float> vision_encoder_image;  // Image encoded by the vision encoder
    struct ggml_tensor * cross_vision_image_tensor;  // Image encoded by the cross vision encoder

    int vision_encoder_img_size = 224;
    int cross_vision_img_size = 1120;

    float norm_mean[3] = {0.48145466, 0.4578275, 0.40821073};
    float norm_deviation[3] = {0.26862954, 0.26130258, 0.27577711};
};

extern struct cogagent_ctx cogagent_global;

#endif