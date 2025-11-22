#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "clip-impl.h"
#include <vector>
#include <string>

// Swin Transformer constants
#define KEY_SWIN_WINDOW_SIZE      "swin.window_size"
#define KEY_SWIN_PATCH_SIZE       "swin.patch_size"
#define KEY_SWIN_IMAGE_SIZE       "swin.image_size"
#define KEY_SWIN_DEPTHS           "swin.depths"
#define KEY_SWIN_NUM_HEADS        "swin.num_heads"
#define KEY_SWIN_HIDDEN_DIM       "swin.hidden_dim"
#define KEY_SWIN_NUM_CHANNELS     "swin.num_channels"
#define KEY_SWIN_MLP_RATIO        "swin.mlp_ratio"
#define KEY_SWIN_DROP_PATH_RATE   "swin.drop_path_rate"
#define KEY_SWIN_NORM_EPS         "swin.norm_eps"

// Tensor names for Swin Transformer
#define TN_SWIN_PATCH_EMBED       "swin.patch_embed.weight"
#define TN_SWIN_PATCH_NORM        "swin.patch_embed.norm.%s"
#define TN_SWIN_POS_EMBED         "swin.pos_embed"
#define TN_SWIN_DOWNSAMPLE_NORM   "swin.stage.%d.downsample.norm.%s"
#define TN_SWIN_DOWNSAMPLE_PROJ   "swin.stage.%d.downsample.reduction.weight"
#define TN_SWIN_ATTN_NORM         "swin.stage.%d.layer.%d.norm1.%s"
#define TN_SWIN_ATTN_QKV          "swin.stage.%d.layer.%d.attn.qkv.%s"
#define TN_SWIN_ATTN_PROJ         "swin.stage.%d.layer.%d.attn.proj.%s"
#define TN_SWIN_ATTN_REL_POS      "swin.stage.%d.layer.%d.attn.relative_position_bias_table"
#define TN_SWIN_FFN_NORM          "swin.stage.%d.layer.%d.norm2.%s"
#define TN_SWIN_FFN_FC1           "swin.stage.%d.layer.%d.mlp.fc1.%s"
#define TN_SWIN_FFN_FC2           "swin.stage.%d.layer.%d.mlp.fc2.%s"
#define TN_SWIN_OUTPUT_NORM       "swin.norm.%s"

// Forward declarations
struct swin_ctx;

// Swin Transformer hyperparameters
struct swin_hparams {
    int32_t image_size = 384;
    int32_t patch_size = 4;
    int32_t num_channels = 3;
    int32_t window_size = 7;
    int32_t hidden_dim = 96;
    std::vector<int32_t> depths = {2, 2, 6, 2};       // depths for each stage
    std::vector<int32_t> num_heads = {3, 6, 12, 24};  // number of heads for each stage
    float mlp_ratio = 4.0f;
    float drop_path_rate = 0.1f;
    float norm_eps = 1e-5f;
    bool use_checkpoint = false;

    // Computed values
    int32_t num_stages() const { return depths.size(); }
    int32_t num_patches() const { return (image_size / patch_size) * (image_size / patch_size); }
};

// Swin Transformer layer
struct swin_layer {
    // Window attention
    struct ggml_tensor * ln1_w;
    struct ggml_tensor * ln1_b;
    struct ggml_tensor * qkv_w;
    struct ggml_tensor * qkv_b;
    struct ggml_tensor * proj_w;
    struct ggml_tensor * proj_b;
    struct ggml_tensor * relative_position_bias_table;

    // FFN
    struct ggml_tensor * ln2_w;
    struct ggml_tensor * ln2_b;
    struct ggml_tensor * fc1_w;
    struct ggml_tensor * fc1_b;
    struct ggml_tensor * fc2_w;
    struct ggml_tensor * fc2_b;
};

// Swin Transformer stage
struct swin_stage {
    std::vector<swin_layer> layers;

    // Patch merging (downsample) layer
    struct ggml_tensor * downsample_norm_w = nullptr;
    struct ggml_tensor * downsample_norm_b = nullptr;
    struct ggml_tensor * downsample_reduction = nullptr;
};

// Swin Transformer vision model
struct swin_vision_model {
    swin_hparams hparams;

    // Patch embedding
    struct ggml_tensor * patch_embed;
    struct ggml_tensor * patch_norm_w;
    struct ggml_tensor * patch_norm_b;
    struct ggml_tensor * pos_embed;

    // Stages
    std::vector<swin_stage> stages;

    // Output norm
    struct ggml_tensor * output_norm_w;
    struct ggml_tensor * output_norm_b;
};

// Main Swin context
struct swin_ctx {
    bool has_vision_encoder = false;
    bool has_projector = false;

    swin_vision_model vision_model;

    // Backend and compute
    struct ggml_backend * backend = nullptr;
    ggml_backend_buffer_t params_buffer = nullptr;

    struct ggml_context * ctx = nullptr;
    std::vector<uint8_t> buf_compute_meta;

    // GGML compute resources
    struct ggml_backend_buffer * compute_buffer = nullptr;
    struct ggml_context * ctx_compute = nullptr;
    struct ggml_alloc * compute_alloc = nullptr;
};

// Public API functions
struct swin_ctx * swin_model_load(const std::string & fname, int verbosity = 1);
void swin_free(struct swin_ctx * ctx);

// Build Swin Transformer graph for inference
struct ggml_cgraph * swin_build_graph(
    struct swin_ctx * ctx,
    const swin_image_batch * imgs,
    std::pair<int, int> load_image_size = {0, 0},
    bool is_inf = false);

// Encode image batch
bool swin_image_batch_encode(
    struct swin_ctx * ctx,
    int n_threads,
    const swin_image_batch * imgs,
    float * vec);

// Utility functions
int swin_patch_size(const struct swin_ctx * ctx);
bool swin_image_preprocess(struct swin_ctx * ctx, const swin_image_u8 * img, swin_image_f32 * res);
bool swin_image_batch_preprocess(struct swin_ctx * ctx, int n_threads, const swin_image_batch * imgs, swin_image_f32_batch * res_batch);

// Window operations for Swin Transformer
struct ggml_tensor * swin_window_partition(struct ggml_context * ctx, struct ggml_tensor * x, int window_size);
struct ggml_tensor * swin_window_reverse(struct ggml_context * ctx, struct ggml_tensor * windows, int window_size, int H, int W);
struct ggml_tensor * swin_create_window_mask(struct ggml_context * ctx, int window_size, int shift_size, int H, int W);
struct ggml_tensor * swin_compute_mask(struct ggml_context * ctx, int window_size, int shift_size, int H, int W);