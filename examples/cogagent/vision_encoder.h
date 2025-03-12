#ifndef VISION_ENCODER_H
#define VISION_ENCODER_H

#include "ggml-backend.h"
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cassert>

#include "cogagent.h"

struct vision_encoder_layer {
    struct ggml_tensor * qkv_w;
    struct ggml_tensor * qkv_b;

    struct ggml_tensor * attn_dense_w;
    struct ggml_tensor * attn_dense_b;

    struct ggml_tensor * input_norm_w;
    struct ggml_tensor * input_norm_b;

    struct ggml_tensor * fc1_w;
    struct ggml_tensor * fc1_b;

    struct ggml_tensor * fc2_w;
    struct ggml_tensor * fc2_b;

    struct ggml_tensor * post_attention_norm_w;
    struct ggml_tensor * post_attention_norm_b;
};

struct vision_encoder {
    struct ggml_tensor * cls_embed;
    struct ggml_tensor * patch_conv_w;
    struct ggml_tensor * patch_conv_b;
    struct ggml_tensor * position_embed_1;

    std::vector<vision_encoder_layer> transformer_layers;

    struct ggml_tensor * position_embed_2;

    struct ggml_tensor * linear_proj_w;

    struct ggml_tensor * linear_proj_norm_w;
    struct ggml_tensor * linear_proj_norm_b;

    struct ggml_tensor * gate_proj_w;
    struct ggml_tensor * dense_h_to_4h_w;
    struct ggml_tensor * dense_4h_to_h_w;

    struct ggml_tensor * boi;
    struct ggml_tensor * eoi;

    int hidden_size = 1792;
    int num_heads = 16;
    int head_hidden_size = hidden_size / num_heads;
    int num_layers = 63;
    float layernorm_eps = 0.000001;
    float attn_scale = 1.0 / std::sqrt(head_hidden_size);
    struct ggml_tensor * input_image;
    struct ggml_tensor * output_tensor;
};

struct vision_encoder_ctx {
    struct ggml_context * ctx_weight;
    struct ggml_context * ctx_compute;
    ggml_backend_buffer_t weight_data;
    ggml_backend_t backend;
    ggml_gallocr_t allocr;
    vision_encoder model;
};

// Assume that overall context is accessible as a static variable
bool vision_encoder_init_load(const char * filename);

// Defines a graph and runs the vision encoder
// Assumes that the picture is of the correct size
void run_vision_encoder(std::vector<float> img_data);

// Free the weights stored for the vision encoder
void free_vision_encoder_ctx();

#endif