#include "swin.h"
#include "clip.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <array>

// Window partition operation - splits input into non-overlapping windows
struct ggml_tensor * swin_window_partition(struct ggml_context * ctx, struct ggml_tensor * x, int window_size) {
    // x shape: [batch_size, height, width, channels]
    // output shape: [batch_size * num_windows, window_size, window_size, channels]

    int batch_size = x->ne[3];
    int H = x->ne[2];
    int W = x->ne[1];
    int C = x->ne[0];

    int nH = H / window_size;
    int nW = W / window_size;

    // Reshape to [batch_size, nH, window_size, nW, window_size, C]
    struct ggml_tensor * reshaped = ggml_reshape_4d(ctx, x,
        C * window_size,
        window_size * nW,
        nH,
        batch_size);

    // Permute to [batch_size, nH, nW, window_size, window_size, C]
    struct ggml_tensor * permuted = ggml_permute(ctx, reshaped, 0, 2, 1, 3);

    // Reshape to [batch_size * nH * nW, window_size, window_size, C]
    struct ggml_tensor * output = ggml_reshape_4d(ctx, permuted,
        C,
        window_size,
        window_size,
        batch_size * nH * nW);

    return output;
}

// Window reverse operation - merges windows back to original spatial dimensions
struct ggml_tensor * swin_window_reverse(struct ggml_context * ctx, struct ggml_tensor * windows, int window_size, int H, int W) {
    // windows shape: [batch_size * num_windows, window_size, window_size, channels]
    // output shape: [batch_size, height, width, channels]

    int C = windows->ne[0];
    int nH = H / window_size;
    int nW = W / window_size;
    int batch_size = windows->ne[3] / (nH * nW);

    // Reshape to [batch_size, nH, nW, window_size, window_size, C]
    struct ggml_tensor * reshaped = ggml_reshape_4d(ctx, windows,
        C * window_size * window_size,
        nW,
        nH,
        batch_size);

    // Permute to [batch_size, nH, window_size, nW, window_size, C]
    struct ggml_tensor * permuted = ggml_permute(ctx, reshaped, 0, 2, 1, 3);

    // Reshape to [batch_size, H, W, C]
    struct ggml_tensor * output = ggml_reshape_4d(ctx, permuted, C, W, H, batch_size);

    return output;
}

// Create attention mask for shifted window attention
struct ggml_tensor * swin_create_window_mask(struct ggml_context * ctx, int window_size, int shift_size, int H, int W) {
    if (shift_size == 0) {
        return nullptr; // No mask needed for non-shifted windows
    }

    // Create a mask tensor
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, W);

    // Initialize mask with region indices
    float * mask_data = (float *)mask->data;
    int h_slices[] = {0, H - window_size, H - shift_size, H};
    int w_slices[] = {0, W - window_size, W - shift_size, W};

    int cnt = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int h = h_slices[i]; h < h_slices[i + 1]; h++) {
                for (int w = w_slices[j]; w < w_slices[j + 1]; w++) {
                    mask_data[h * W + w] = cnt;
                }
            }
            cnt++;
        }
    }

    return mask;
}

// Build window attention layer
static struct ggml_tensor * swin_window_attention(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    const swin_layer & layer,
    int num_heads,
    int window_size,
    bool shifted) {

    int batch_size = x->ne[3];
    int seq_len = x->ne[2] * x->ne[1]; // window_size * window_size
    int hidden_dim = x->ne[0];
    int head_dim = hidden_dim / num_heads;

    // Reshape input for attention: [batch_size, seq_len, hidden_dim]
    x = ggml_reshape_3d(ctx, x, hidden_dim, seq_len, batch_size);

    // Layer norm
    x = ggml_norm(ctx, x, layer.ln1_w->ne[0]);
    x = ggml_add(ctx, ggml_mul(ctx, x, layer.ln1_w), layer.ln1_b);

    // QKV projection
    struct ggml_tensor * qkv = ggml_mul_mat(ctx, layer.qkv_w, x);
    qkv = ggml_add(ctx, qkv, layer.qkv_b);

    // Split into Q, K, V
    int qkv_dim = qkv->ne[0] / 3;
    struct ggml_tensor * q = ggml_view_3d(ctx, qkv, qkv_dim, seq_len, batch_size, qkv->nb[1], qkv->nb[2], 0);
    struct ggml_tensor * k = ggml_view_3d(ctx, qkv, qkv_dim, seq_len, batch_size, qkv->nb[1], qkv->nb[2], qkv_dim * ggml_element_size(qkv));
    struct ggml_tensor * v = ggml_view_3d(ctx, qkv, qkv_dim, seq_len, batch_size, qkv->nb[1], qkv->nb[2], 2 * qkv_dim * ggml_element_size(qkv));

    // Reshape for multi-head attention
    q = ggml_reshape_4d(ctx, q, head_dim, num_heads, seq_len, batch_size);
    k = ggml_reshape_4d(ctx, k, head_dim, num_heads, seq_len, batch_size);
    v = ggml_reshape_4d(ctx, v, head_dim, num_heads, seq_len, batch_size);

    // Transpose for attention: [batch_size, num_heads, seq_len, head_dim]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    // Scaled dot-product attention
    float scale = 1.0f / sqrtf(head_dim);
    struct ggml_tensor * attn = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, k)), q);
    attn = ggml_scale(ctx, attn, scale);

    // Add relative position bias if available
    if (layer.relative_position_bias_table != nullptr) {
        // This would need proper indexing based on relative positions
        // For now, simplified version
        attn = ggml_add(ctx, attn, layer.relative_position_bias_table);
    }

    // Apply mask for shifted window attention
    if (shifted) {
        // Create and apply attention mask
        struct ggml_tensor * mask = swin_create_window_mask(ctx, window_size, window_size / 2,
                                                           window_size, window_size);
        if (mask != nullptr) {
            // Convert mask to attention mask
            attn = ggml_add(ctx, attn, mask);
        }
    }

    // Softmax
    attn = ggml_soft_max(ctx, attn);

    // Apply attention to values
    struct ggml_tensor * out = ggml_mul_mat(ctx, v, attn);

    // Transpose back: [batch_size, seq_len, num_heads, head_dim]
    out = ggml_permute(ctx, out, 0, 2, 1, 3);

    // Reshape to merge heads: [batch_size, seq_len, hidden_dim]
    out = ggml_reshape_3d(ctx, out, hidden_dim, seq_len, batch_size);

    // Output projection
    out = ggml_mul_mat(ctx, layer.proj_w, out);
    out = ggml_add(ctx, out, layer.proj_b);

    return out;
}

// Build FFN layer
static struct ggml_tensor * swin_ffn(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    const swin_layer & layer,
    float mlp_ratio) {

    // Layer norm
    x = ggml_norm(ctx, x, layer.ln2_w->ne[0]);
    x = ggml_add(ctx, ggml_mul(ctx, x, layer.ln2_w), layer.ln2_b);

    // FFN: Linear -> GELU -> Linear
    x = ggml_mul_mat(ctx, layer.fc1_w, x);
    x = ggml_add(ctx, x, layer.fc1_b);
    x = ggml_gelu(ctx, x);

    x = ggml_mul_mat(ctx, layer.fc2_w, x);
    x = ggml_add(ctx, x, layer.fc2_b);

    return x;
}

// Build Swin Transformer block
static struct ggml_tensor * swin_block(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    const swin_layer & layer,
    int num_heads,
    int window_size,
    bool shifted,
    float mlp_ratio) {

    int H = x->ne[2];
    int W = x->ne[1];

    struct ggml_tensor * shortcut = x;

    // Shifted window partitioning if needed
    if (shifted && (H > window_size || W > window_size)) {
        // Cyclic shift
        int shift_size = window_size / 2;
        x = ggml_roll(ctx, x, -shift_size, 2); // Roll along H dimension
        x = ggml_roll(ctx, x, -shift_size, 1); // Roll along W dimension
    }

    // Partition into windows
    if (H > window_size || W > window_size) {
        x = swin_window_partition(ctx, x, window_size);
    }

    // Window attention
    x = swin_window_attention(ctx, x, layer, num_heads, window_size, shifted);

    // Reverse window partition
    if (H > window_size || W > window_size) {
        x = swin_window_reverse(ctx, x, window_size, H, W);
    }

    // Reverse cyclic shift if needed
    if (shifted && (H > window_size || W > window_size)) {
        int shift_size = window_size / 2;
        x = ggml_roll(ctx, x, shift_size, 2); // Roll back along H dimension
        x = ggml_roll(ctx, x, shift_size, 1); // Roll back along W dimension
    }

    // Residual connection
    x = ggml_add(ctx, x, shortcut);

    // FFN with residual
    shortcut = x;
    x = swin_ffn(ctx, x, layer, mlp_ratio);
    x = ggml_add(ctx, x, shortcut);

    return x;
}

// Patch merging layer (downsampling)
static struct ggml_tensor * swin_patch_merging(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    struct ggml_tensor * norm_w,
    struct ggml_tensor * norm_b,
    struct ggml_tensor * reduction) {

    int batch_size = x->ne[3];
    int H = x->ne[2];
    int W = x->ne[1];
    int C = x->ne[0];

    // Reshape to merge 2x2 patches
    x = ggml_reshape_4d(ctx, x, C, W/2, 2, H/2 * 2 * batch_size);
    x = ggml_permute(ctx, x, 0, 2, 1, 3);
    x = ggml_reshape_4d(ctx, x, C * 4, W/2, H/2, batch_size);

    // Layer norm
    x = ggml_norm(ctx, x, norm_w->ne[0]);
    x = ggml_add(ctx, ggml_mul(ctx, x, norm_w), norm_b);

    // Linear reduction
    x = ggml_mul_mat(ctx, reduction, x);

    return x;
}

// Build complete Swin Transformer graph
struct ggml_cgraph * swin_build_graph(
    struct swin_ctx * ctx,
    const swin_image_batch * imgs,
    std::pair<int, int> load_image_size,
    bool is_inf) {

    if (!ctx->has_vision_encoder) {
        return nullptr;
    }

    const auto & model = ctx->vision_model;
    const auto & hparams = model.hparams;

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx->buf_compute_meta.size(),
        /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * cgraph = ggml_new_graph(ctx0);

    const int batch_size = imgs->size;
    const int image_size = hparams.image_size;
    const int patch_size = hparams.patch_size;
    const int num_patches_side = image_size / patch_size;
    const int num_patches = num_patches_side * num_patches_side;
    const int hidden_dim = hparams.hidden_dim;

    // Input image tensor
    struct ggml_tensor * inp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32,
                                                  3, image_size, image_size, batch_size);
    ggml_set_name(inp, "inp");

    // Patch embedding: Conv2D with stride=patch_size
    struct ggml_tensor * x = ggml_conv_2d(ctx0, model.patch_embed, inp, patch_size, patch_size, 0, 0, 1, 1);

    // Reshape to [batch_size, num_patches, hidden_dim]
    x = ggml_reshape_3d(ctx0, x, hidden_dim, num_patches, batch_size);

    // Add positional embeddings if available
    if (model.pos_embed != nullptr) {
        x = ggml_add(ctx0, x, model.pos_embed);
    }

    // Layer norm after patch embedding
    if (model.patch_norm_w != nullptr) {
        x = ggml_norm(ctx0, x, model.patch_norm_w->ne[0]);
        x = ggml_add(ctx0, ggml_mul(ctx0, x, model.patch_norm_w), model.patch_norm_b);
    }

    // Reshape for spatial processing
    x = ggml_reshape_4d(ctx0, x, hidden_dim, num_patches_side, num_patches_side, batch_size);

    // Process through Swin stages
    int H = num_patches_side;
    int W = num_patches_side;
    int C = hidden_dim;

    for (size_t stage_idx = 0; stage_idx < model.stages.size(); stage_idx++) {
        const auto & stage = model.stages[stage_idx];

        // Process layers in this stage
        for (size_t layer_idx = 0; layer_idx < stage.layers.size(); layer_idx++) {
            const auto & layer = stage.layers[layer_idx];
            bool shifted = (layer_idx % 2 == 1); // Alternate between regular and shifted windows

            x = swin_block(ctx0, x, layer,
                         hparams.num_heads[stage_idx],
                         hparams.window_size,
                         shifted,
                         hparams.mlp_ratio);
        }

        // Patch merging (downsampling) between stages, except for the last stage
        if (stage_idx < model.stages.size() - 1 && stage.downsample_reduction != nullptr) {
            x = swin_patch_merging(ctx0, x,
                                 stage.downsample_norm_w,
                                 stage.downsample_norm_b,
                                 stage.downsample_reduction);
            H /= 2;
            W /= 2;
            C *= 2; // Channel dimension doubles after patch merging
        }
    }

    // Global average pooling
    x = ggml_reshape_3d(ctx0, x, C, H * W, batch_size);
    x = ggml_mean(ctx0, x); // Average over spatial dimensions

    // Final layer norm
    if (model.output_norm_w != nullptr) {
        x = ggml_norm(ctx0, x, model.output_norm_w->ne[0]);
        x = ggml_add(ctx0, ggml_mul(ctx0, x, model.output_norm_w), model.output_norm_b);
    }

    ggml_set_name(x, "output");
    ggml_build_forward_expand(cgraph, x);

    return cgraph;
}

// Model loading function
struct swin_ctx * swin_model_load(const std::string & fname, int verbosity) {
    struct swin_ctx * ctx = new swin_ctx();

    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx->ctx,
    };

    struct gguf_context * gguf_ctx = gguf_init_from_file(fname.c_str(), params);
    if (!gguf_ctx) {
        fprintf(stderr, "%s: failed to load model from %s\n", __func__, fname.c_str());
        swin_free(ctx);
        return nullptr;
    }

    // Load hyperparameters
    auto & hparams = ctx->vision_model.hparams;

    // Read Swin-specific parameters from GGUF
    const int n_kv = gguf_get_n_kv(gguf_ctx);
    for (int i = 0; i < n_kv; ++i) {
        const char * key = gguf_get_key(gguf_ctx, i);

        if (strcmp(key, KEY_SWIN_WINDOW_SIZE) == 0) {
            hparams.window_size = gguf_get_val_i32(gguf_ctx, i);
        } else if (strcmp(key, KEY_SWIN_PATCH_SIZE) == 0) {
            hparams.patch_size = gguf_get_val_i32(gguf_ctx, i);
        } else if (strcmp(key, KEY_SWIN_IMAGE_SIZE) == 0) {
            hparams.image_size = gguf_get_val_i32(gguf_ctx, i);
        } else if (strcmp(key, KEY_SWIN_HIDDEN_DIM) == 0) {
            hparams.hidden_dim = gguf_get_val_i32(gguf_ctx, i);
        } else if (strcmp(key, KEY_SWIN_MLP_RATIO) == 0) {
            hparams.mlp_ratio = gguf_get_val_f32(gguf_ctx, i);
        } else if (strcmp(key, KEY_SWIN_NORM_EPS) == 0) {
            hparams.norm_eps = gguf_get_val_f32(gguf_ctx, i);
        }
        // TODO: Load depths and num_heads arrays
    }

    ctx->has_vision_encoder = true;

    if (verbosity >= 1) {
        printf("Swin Transformer model loaded:\n");
        printf("  image_size:  %d\n", hparams.image_size);
        printf("  patch_size:  %d\n", hparams.patch_size);
        printf("  window_size: %d\n", hparams.window_size);
        printf("  hidden_dim:  %d\n", hparams.hidden_dim);
        printf("  num_stages:  %d\n", hparams.num_stages());
    }

    // TODO: Load actual tensor weights from GGUF file

    gguf_free(gguf_ctx);

    return ctx;
}

// Free context
void swin_free(struct swin_ctx * ctx) {
    if (ctx == nullptr) {
        return;
    }

    if (ctx->backend) {
        ggml_backend_free(ctx->backend);
    }

    if (ctx->params_buffer) {
        ggml_backend_buffer_free(ctx->params_buffer);
    }

    if (ctx->compute_buffer) {
        ggml_backend_buffer_free(ctx->compute_buffer);
    }

    if (ctx->ctx) {
        ggml_free(ctx->ctx);
    }

    delete ctx;
}