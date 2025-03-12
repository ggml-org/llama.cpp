#ifndef CROSS_VISION_H
#define CROSS_VISION_H

#include "ggml-backend.h"
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cassert>

#include "cogagent.h"

struct cross_vision_layer {
    struct ggml_tensor * norm1_w;
    struct ggml_tensor * norm1_b;

    // No bias for K projection
    struct ggml_tensor * q_w;
    struct ggml_tensor * q_b;
    struct ggml_tensor * k_w;
    struct ggml_tensor * v_w;
    struct ggml_tensor * v_b;

    struct ggml_tensor * attn_ln_w;
    struct ggml_tensor * attn_ln_b;
    struct ggml_tensor * attn_linear_w;
    struct ggml_tensor * attn_linear_b;

    struct ggml_tensor * norm2_w;
    struct ggml_tensor * norm2_b;

    struct ggml_tensor * mlp_linear1_w;
    struct ggml_tensor * mlp_linear1_b;
    struct ggml_tensor * mlp_linear2_w;
    struct ggml_tensor * mlp_linear2_b;
    struct ggml_tensor * mlp_ln_w;
    struct ggml_tensor * mlp_ln_b;
    struct ggml_tensor * mlp_linear3_w;
    struct ggml_tensor * mlp_linear3_b;
};

struct cross_vision {
    struct ggml_tensor * patch_conv_w;
    struct ggml_tensor * patch_conv_b;

    struct ggml_tensor * cls_embed;
    struct ggml_tensor * pos_embed_1;

    struct ggml_tensor * rope_freqs_cos;  // 6400 x 64. In other words, l x k
    struct ggml_tensor * rope_freqs_sin;

    std::vector<cross_vision_layer> transformer_layers;

    struct ggml_tensor * pos_embed_2;

    struct ggml_tensor * input_image;
    struct ggml_tensor * output_tensor;

    float layernorm_eps = 0.000001;  // Need to check all of these to confirm
    int num_heads = 16;  // For sure
    int hidden_size = 1024;
    int head_hidden_size = hidden_size / num_heads;
    int num_layers = 24;
    float attn_scale = 1.0 / std::sqrt(head_hidden_size);
};

struct cross_vision_ctx {
    struct ggml_context * ctx_weight;
    struct ggml_context * ctx_compute;
    ggml_backend_buffer_t weight_data;
    ggml_backend_t backend;
    ggml_gallocr_t allocr;
    cross_vision model;
};

bool cross_vision_init_load(const char * filename);

void run_cross_vision(std::vector<float> img_data);

void free_cross_vision_ctx();

#endif