#pragma once

#include "ggml.h"
#include "mtmd.h"

#include <stddef.h>
#include <stdint.h>

#include <map>

// !!! Internal header, to be used by mtmd only !!!

#define MTMD_INTERNAL_HEADER

struct clip_ctx;

struct clip_image_size {
    int width;
    int height;
    bool operator==(const clip_image_size & other) const {
        return width == other.width && height == other.height;
    }
    bool operator!=(const clip_image_size & other) const {
        return !(*this == other);
    }
    int area() const {
        // avoid overflow when computing area
        GGML_ASSERT(width  >= 0 && width  <= 46000);
        GGML_ASSERT(height >= 0 && height <= 46000);
        return width * height;
    }
};

struct clip_image_f32;
struct clip_image_f32_batch;

enum clip_modality {
    CLIP_MODALITY_VISION,
    CLIP_MODALITY_AUDIO,
};

enum clip_flash_attn_type {
    CLIP_FLASH_ATTN_TYPE_AUTO     = -1,
    CLIP_FLASH_ATTN_TYPE_DISABLED = 0,
    CLIP_FLASH_ATTN_TYPE_ENABLED  = 1,
};

struct clip_context_params {
    bool use_gpu;
    enum clip_flash_attn_type flash_attn_type;
    int image_min_tokens;
    int image_max_tokens;
    bool warmup;
    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
    bool no_alloc;
    mtmd_progress_callback progress_callback;
    void * progress_callback_user_data;
};

struct clip_init_result {
    struct clip_ctx * ctx_v; // vision context
    struct clip_ctx * ctx_a; // audio context
};

struct clip_init_result clip_init(const char * fname, struct clip_context_params ctx_params);

void clip_free(struct clip_ctx * ctx);

// TODO: should be enum, not string
const char * clip_patch_merge_type(const struct clip_ctx * ctx);

int clip_n_output_tokens(const clip_ctx * ctx, const clip_image_f32 * img);

// for M-RoPE, this will be the number of token positions in X and Y directions
// for other models, X will be the total number of tokens and Y will be 1
int clip_n_output_tokens_x(const clip_ctx * ctx, const clip_image_f32 * img);
int clip_n_output_tokens_y(const clip_ctx * ctx, const clip_image_f32 * img);

// this should be equal to the embedding dimension of the text model
int clip_n_mmproj_embd(const struct clip_ctx * ctx);

// TODO: remove clip_image_encode() and always use batched version
bool clip_image_encode      (struct clip_ctx * ctx, int n_threads, const clip_image_f32 * img, std::vector<float> & out_vec);
bool clip_image_batch_encode(struct clip_ctx * ctx, int n_threads, const struct clip_image_f32_batch * imgs, std::vector<float> & out_batch_embd);

bool clip_is_llava(const struct clip_ctx * ctx);
// note for contributor: this clip_is_(model) pattern is deprecated
//                       do NOT add new functions like this

bool clip_has_vision_encoder(const struct clip_ctx * ctx);
bool clip_has_audio_encoder(const struct clip_ctx * ctx);

bool clip_support_batch(const struct clip_ctx * ctx);

int clip_model_n_temporal_merge(const struct clip_ctx * ctx); // TODO @ngxson : remove, refactor this

std::map<ggml_backend_dev_t, size_t> clip_get_mem_usage(const struct clip_ctx * ctx);

struct clip_cap {
    bool has_vision;
    bool has_audio;
};
struct clip_cap clip_get_cap(const char * fname);
