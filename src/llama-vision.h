#pragma once

#include "ggml.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-arch.h"

#include <vector>
#include <array>

#define VISION_GRAPH_MAX_NODE 2048

enum vision_projector_type {
    VISION_PROJECTOR_TYPE_UNKNOWN,
    VISION_PROJECTOR_TYPE_MLP,
    VISION_PROJECTOR_TYPE_LDPV2,
    VISION_PROJECTOR_TYPE_MINICPMV_2_5,
    VISION_PROJECTOR_TYPE_MINICPMV_2_6,
};

enum mm_patch_merge {
    MM_PATCH_MERGE_UNKNOWN,
    MM_PATCH_MERGE_FLAT,
    MM_PATCH_MERGE_SPATIAL_UNPAD,
};

struct llama_vision_model {
    struct vision_hparams {
        llm_arch arch = LLM_ARCH_UNKNOWN;

        uint32_t image_size;
        uint32_t patch_size;
        uint32_t hidden_size;
        uint32_t n_intermediate;
        uint32_t projection_dim;
        uint32_t n_head;
        uint32_t n_layer;
        uint32_t max_pos_embd;
        int32_t select_layer = 0;
        bool use_gelu = false;

        float eps;

        vision_projector_type proj_type = VISION_PROJECTOR_TYPE_UNKNOWN;
        mm_patch_merge mm_patch_merge_type = MM_PATCH_MERGE_UNKNOWN;

        std::array<float, 3> image_mean;
        std::array<float, 3> image_std;

        std::array<int32_t, 32> image_grid_pinpoints; // TODO: should this be array of (x, y) pairs?
        int32_t image_crop_resolution;

        // idefics3
        int scale_factor = 0;
    };
    struct vision_hparams hparams;
    ggml_backend_buffer_type_t buft;

    // embeddings
    struct ggml_tensor * class_embedding     = nullptr;
    struct ggml_tensor * patch_embeddings    = nullptr;
    struct ggml_tensor * patch_bias          = nullptr;
    struct ggml_tensor * position_embeddings = nullptr;

    struct ggml_tensor * pre_norm_w = nullptr;
    struct ggml_tensor * pre_norm_b = nullptr;

    struct vision_layer {
        // attention
        struct ggml_tensor * k_w = nullptr;
        struct ggml_tensor * k_b = nullptr;
        struct ggml_tensor * q_w = nullptr;
        struct ggml_tensor * q_b = nullptr;
        struct ggml_tensor * v_w = nullptr;
        struct ggml_tensor * v_b = nullptr;

        struct ggml_tensor * output_w = nullptr;
        struct ggml_tensor * output_b = nullptr;

        // layernorm 1
        struct ggml_tensor * norm_in_w = nullptr;
        struct ggml_tensor * norm_in_b = nullptr;

        // ff
        struct ggml_tensor * ffn_up_w = nullptr;
        struct ggml_tensor * ffn_up_b = nullptr;

        struct ggml_tensor * ffn_down_w = nullptr;
        struct ggml_tensor * ffn_down_b = nullptr;

        // layernorm 2
        struct ggml_tensor * norm_out_w = nullptr;
        struct ggml_tensor * norm_out_b = nullptr;
    };
    std::vector<vision_layer> layers;

    struct ggml_tensor * post_norm_w = nullptr;
    struct ggml_tensor * post_norm_b = nullptr;

    struct ggml_tensor * projection = nullptr;

    // LLaVA projection
    struct ggml_tensor * mm_1_w = nullptr;
    struct ggml_tensor * mm_1_b = nullptr;
    struct ggml_tensor * mm_2_w = nullptr;
    struct ggml_tensor * mm_2_b = nullptr;

    // MobileVLM_V2 projection
    struct ggml_tensor * mm_model_mlp_0_w = nullptr;
    struct ggml_tensor * mm_model_mlp_0_b = nullptr;
    struct ggml_tensor * mm_model_mlp_2_w = nullptr;
    struct ggml_tensor * mm_model_mlp_2_b = nullptr;
    struct ggml_tensor * mm_model_peg_0_w = nullptr;
    struct ggml_tensor * mm_model_peg_0_b = nullptr;

    // MINICPMV projection
    struct ggml_tensor * mm_model_pos_embed_k = nullptr;
    struct ggml_tensor * mm_model_query       = nullptr;
    struct ggml_tensor * mm_model_proj        = nullptr;
    struct ggml_tensor * mm_model_kv_proj     = nullptr;
    struct ggml_tensor * mm_model_attn_q_w    = nullptr;
    struct ggml_tensor * mm_model_attn_q_b    = nullptr;
    struct ggml_tensor * mm_model_attn_k_w    = nullptr;
    struct ggml_tensor * mm_model_attn_k_b    = nullptr;
    struct ggml_tensor * mm_model_attn_v_w    = nullptr;
    struct ggml_tensor * mm_model_attn_v_b    = nullptr;
    struct ggml_tensor * mm_model_attn_o_w    = nullptr;
    struct ggml_tensor * mm_model_attn_o_b    = nullptr;
    struct ggml_tensor * mm_model_ln_q_w      = nullptr;
    struct ggml_tensor * mm_model_ln_q_b      = nullptr;
    struct ggml_tensor * mm_model_ln_kv_w     = nullptr;
    struct ggml_tensor * mm_model_ln_kv_b     = nullptr;
    struct ggml_tensor * mm_model_ln_post_w   = nullptr;
    struct ggml_tensor * mm_model_ln_post_b   = nullptr;

    // special tokens
    struct ggml_tensor * mm_tok_embd_image     = nullptr;
    struct ggml_tensor * mm_tok_embd_end_image = nullptr;
    struct ggml_tensor * mm_tok_embd_slice     = nullptr;
    struct ggml_tensor * mm_tok_embd_end_slice = nullptr;
};

struct llama_vision_context {
    // memory buffers used to evaluate the model
    std::vector<uint8_t> buf_compute_meta;
    ggml_backend_sched_ptr sched;
    std::vector<ggml_backend_ptr> backends;
    ggml_backend_t backend_cpu;

    const llama_vision_model * model;

    // temporary output data, to be picked up by llama_decode()
    struct ggml_context * ctx_ggml = nullptr;
    struct ggml_tensor * output;
};

// for now, this only contains:
// - the instruction for ggml_conv_2d to break the image into patches
// - the pre-processed image data in f32
struct llama_vision_tokens {
    uint32_t px; // size of patch
    uint32_t py; // size of patch
    size_t n_px; // number of patches in x direction
    size_t n_py; // number of patches in y direction
    // RGB float32 image (NHWC)
    // Memory layout: RGBRGBRGB...
    std::vector<std::vector<float>> buf; // preprocessed image data
};

inline mm_patch_merge mm_patch_merge_from_name(std::string & name) {
    if (name == "flat") {
        return MM_PATCH_MERGE_FLAT;
    } else if (name == "spatial_unpad") {
        return MM_PATCH_MERGE_SPATIAL_UNPAD;
    }
    return MM_PATCH_MERGE_UNKNOWN;
}

inline vision_projector_type vision_projector_type_from_name(std::string & name) {
    if (name == "mlp") {
        return VISION_PROJECTOR_TYPE_MLP;
    } else if (name == "ldpv2") {
        return VISION_PROJECTOR_TYPE_LDPV2;
    } else if (name == "minicpmv-2.5") {
        return VISION_PROJECTOR_TYPE_MINICPMV_2_5;
    } else if (name == "minicpmv-2.6") {
        return VISION_PROJECTOR_TYPE_MINICPMV_2_6;
    }
    return VISION_PROJECTOR_TYPE_UNKNOWN;
}

// only for sanity check: must be equal to n_embd of language model
uint32_t llama_vision_n_mmproj_embd(const llama_vision_model & vmodel);

struct ggml_tensor * llama_vision_get_output_tensor(llama_context * ctx);
