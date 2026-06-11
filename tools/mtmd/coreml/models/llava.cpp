// CoreML adapter for Llava 1.5.
//
// The exported CoreML bundle covers the full vision pipeline:
//   CLIP ViT-L/14@336 → MLP projector → LLM embedding space
//
// .mlmodelc schema:
//   inputs:
//     pixel_values : F32 [1, 3, 336, 336]   raw image tensor
//   outputs:
//     output       : F32 [1, 576, 4096]     576 patches mapped to LLM hidden dim

#include "models.h"

#include "../backend.h"

#include <cstdint>
#include <cstring>

namespace mtmd_coreml::models::llava {

static bool detect(const mlmodelc_meta & meta, const struct llama_model *) {
    // Llava: single pixel_values input, single output
    const bool inputs_match  = meta.input_names ==
                               std::set<std::string>{"pixel_values"};
    const bool outputs_match = meta.output_names == std::set<std::string>{"output"};
    if (!inputs_match || !outputs_match) return false;

    // Output must be a 3D tensor [1, n_tokens, embed_dim]
    auto it = meta.output_shapes.find("output");
    if (it == meta.output_shapes.end() || it->second.size() != 3) return false;
    return it->second[2] > 0;
}

static void setup(context &                  ctx,
                  const mlmodelc_meta &      meta,
                  const struct llama_model * /*text_model*/) {
    const auto & out_shape = meta.output_shapes.at("output");
    ctx.n_tokens_out = static_cast<int>(out_shape[1]);
    ctx.n_embd_out   = static_cast<int>(out_shape[2]);

    auto & hp = ctx.hparams;
    hp.image_size         = 336;
    hp.patch_size         = 14;

    ctx.image_preproc = std::make_unique<mtmd_image_preprocessor_fixed_size>(hp);
}

static bool encode_slice(const context &               ctx,
                         const struct clip_image_f32 & img,
                         float *                       out) {
    // clip_image_f32 stores planar RGB data matching NCHW layout
    const int n = img.nx * img.ny;
    return backend::predict_single_output(ctx.model_handle, {
        { "pixel_values", img.buf, { 1, 3, img.ny, img.nx }, backend::DTYPE_F32 },
    }, "output", out);
}

const model_adapter g_adapter = {
    /* name         */ "llava",
    /* detect       */ detect,
    /* setup        */ setup,
    /* encode_slice */ encode_slice,
};

} // namespace mtmd_coreml::models::llava
