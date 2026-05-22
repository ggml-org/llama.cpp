#pragma once

#include "../clip-graph.h"

/*
 * IMPORTANT: The mtmd module does NOT accept pull requests that are fully or predominantly AI-generated.
 * We encourage human contributors to ensure the quality and reliability of the codebase.
 */

struct clip_graph_siglip : clip_graph {
    clip_graph_siglip(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_gemma4v : clip_graph {
    clip_graph_gemma4v(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
    ggml_tensor * build_mm(ggml_tensor * w, ggml_tensor * x) const override;
};

struct clip_graph_pixtral : clip_graph {
    clip_graph_pixtral(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_qwen2vl : clip_graph {
    clip_graph_qwen2vl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_qwen3vl : clip_graph {
    clip_graph_qwen3vl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_mimovl : clip_graph {
    clip_graph_mimovl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
    // Force F32 mat-mul accumulation to avoid F16 overflow in the FFN down-proj
    // when the mmproj is stored in F16 (the source weights are BF16; downcasting
    // to F16 reduces dynamic range below the SwiGLU output magnitude on the last few layers).
    ggml_tensor * build_mm(ggml_tensor * w, ggml_tensor * x) const override;
};

struct clip_graph_step3vl : clip_graph {
    clip_graph_step3vl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_youtuvl : clip_graph {
    clip_graph_youtuvl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_yasa2 : clip_graph {
    clip_graph_yasa2(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;

    ggml_tensor * layer_norm_channels(ggml_tensor * inp, ggml_tensor * w, ggml_tensor * b, float eps = 1e-6f);
    ggml_tensor * convnext_grn(ggml_tensor * inp, ggml_tensor * w, ggml_tensor * b);
};

struct clip_graph_minicpmv : clip_graph {
    clip_graph_minicpmv(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_minicpmv4_6 : clip_graph {
    clip_graph_minicpmv4_6(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_internvl : clip_graph {
    clip_graph_internvl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_nemotron_v2_vl : clip_graph {
    clip_graph_nemotron_v2_vl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_llama4 : clip_graph {
    clip_graph_llama4(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_kimivl : clip_graph {
    clip_graph_kimivl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_paddleocr : clip_graph {
    clip_graph_paddleocr(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_dotsocr : clip_graph {
    clip_graph_dotsocr(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_cogvlm : clip_graph {
    clip_graph_cogvlm(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_llava : clip_graph {
    clip_graph_llava(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_whisper_enc : clip_graph {
    clip_graph_whisper_enc(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_deepseekocr : clip_graph {
    clip_graph_deepseekocr(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
    ggml_tensor * build_sam(ggml_tensor * inp); // build the SAM model
};

struct clip_graph_conformer : clip_graph {
    clip_graph_conformer(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_granite_speech : clip_graph {
    clip_graph_granite_speech(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_gemma4a : clip_graph {
    clip_graph_gemma4a(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
    ggml_tensor * build_mm(ggml_tensor * w, ggml_tensor * x) const override;
};

struct clip_graph_glm4v : clip_graph {
    clip_graph_glm4v(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_hunyuanvl : clip_graph {
    clip_graph_hunyuanvl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_mobilenetv5 : clip_graph {
    clip_graph_mobilenetv5(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;

    ggml_tensor * rms_norm_2d(
        ggml_tensor * inp,
        ggml_tensor * weight,
        float eps = 1e-6f);

    ggml_tensor* pad_same_2d(
        ggml_tensor* inp,
        int kernel_h,
        int kernel_w,
        int stride_h,
        int stride_w,
        int dilation_h = 1,
        int dilation_w = 1);

    ggml_tensor * build_edge_residual(
        ggml_tensor * inp,
        const mobilenetv5_block & block,
        int stride);

    ggml_tensor * build_inverted_residual(
        ggml_tensor * inp,
        const mobilenetv5_block & block,
        int stride);

    ggml_tensor * build_mobilenet_attn(
        ggml_tensor * inp,
        const mobilenetv5_block & block);
};

struct clip_graph_qwen3a : clip_graph {
    clip_graph_qwen3a(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_kimik25 : clip_graph {
    clip_graph_kimik25(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;

    ggml_tensor * resize_position_embeddings_3d(uint32_t interpolation_mode);
};

struct clip_graph_granite4_vision : clip_graph {
    clip_graph_granite4_vision(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

// --- Granite Vision 4 assembler ---
//
// Per-tile encoder outputs are reshaped (pack-and-unpad) and interleaved with
// a learned, base-scaled newline embedding to produce the final image-token
// sequence. The cgraph is a single ggml_get_rows over a virtual buffer formed
// by concatenating the per-tile encoder outputs with the K-tiled, base-scaled
// newline row. All model-specific knowledge (n_per_tile inference, gather
// index construction, input tensor names) lives on the class below.

struct clip_assembler_granite4_vision : clip_assembler {
    const clip_model & model;
    const float *      per_tile_embd; // borrowed, caller-owned lifetime
    int                n_tiles;
    int                grid_x;
    int                grid_y;
    int                n_per_tile;
    int                n_mmproj_embd;
    std::vector<int32_t> gather_idx;  // computed in constructor

    clip_assembler_granite4_vision(const clip_ctx * ctx,
                                   const float * per_tile_embd,
                                   int n_tiles, int grid_x, int grid_y);

    ggml_tensor * build(ggml_context * ctx0, ggml_cgraph * gf) override;
    void set_inputs(ggml_cgraph * gf) override;
};

// Final assembled token count for a preprocessed tile batch. Called before
// per-tile encoding (from add_media), so kept as a free function rather than
// a method on the assembler class.
size_t granite4_vision_n_assembled_output_tokens(
        const clip_ctx * ctx,
        const clip_image_f32_batch * batch);
