// CoreML adapter for the MiniCPM-V family with the unified "vit_all" ANE
// pipeline (used by v4.0 / v4.5 / v4.6 and any future variant that follows
// the same export contract).
//
// .mlmodelc schema (single adapter handles all current variants):
//   inputs:
//     pixel_values : F32 [1, 3, 14, 14336]   packed row-of-patches (max_patches=1024)
//     patch_w      : I32 [1]                 patch grid width as a single scalar
//   outputs:
//     output       : F32 [1, n_tokens, embed_dim]   resampler/merger output
//                                                    (embed_dim is LLM hidden)
//
// position_ids / sincos pos_embed / window indices are all computed inside
// the ANE graph from `patch_w`; the host only packs pixels and passes one
// integer. n_embd_out and n_tokens_out are read directly from the
// .mlmodelc metadata.json so the same adapter works across all variants
// that differ only in LLM hidden dim (2560 for v4.0, 4096 for v4.5,
// 1024 for v4.6, ...).

#include "models.h"

#include "../backend.h"

#include <cstdint>

namespace mtmd_coreml::models::minicpmv {

// Static shape contract shared by all variants of this family. Only the
// output embed_dim and n_tokens are variant-specific and read from metadata.
static constexpr int IMAGE_SIZE  = 980;   // siglip-so400m-14-980 ViT input
static constexpr int PATCH_SIZE  = 14;
static constexpr int MAX_PATCHES = 1024;

static bool detect(const mlmodelc_meta & meta, const struct llama_model *) {
    const bool inputs_match  = meta.input_names ==
                               std::set<std::string>{"pixel_values", "patch_w"};
    const bool outputs_match = meta.output_names == std::set<std::string>{"output"};
    if (!inputs_match || !outputs_match) return false;

    // Reject bundles whose output isn't a [1, n_tokens, embed_dim] tensor;
    // this guards against accidentally matching a totally different family
    // that happens to share the (pixel_values, patch_w) input contract.
    auto it = meta.output_shapes.find("output");
    if (it == meta.output_shapes.end() || it->second.size() != 3) return false;
    return it->second[2] > 0;
}

static void setup(context &                  ctx,
                  const mlmodelc_meta &      meta,
                  const struct llama_model * /*text_model*/) {
    const auto & out_shape = meta.output_shapes.at("output");  // [1, n_tokens, embed_dim]
    ctx.n_tokens_out = static_cast<int>(out_shape[1]);
    ctx.n_embd_out   = static_cast<int>(out_shape[2]);

    auto & hp = ctx.hparams;
    hp.image_size         = IMAGE_SIZE;
    hp.patch_size         = PATCH_SIZE;
    // minicpmv_version >= 3 makes mtmd select the MINICPMV_2_6 slice
    // template (overview <image> ... </image> <slice> ... </slice> ...).
    // The actual numeric variant (4 / 5 / 6) doesn't change slice layout
    // in mtmd, so we use a single value for all bundles.
    hp.minicpmv_version   = 4;
    hp.minicpmv_query_num = ctx.n_tokens_out;
    hp.image_mean[0] = hp.image_mean[1] = hp.image_mean[2] = 0.5f;
    hp.image_std [0] = hp.image_std [1] = hp.image_std [2] = 0.5f;
    // Other llava-uhd knobs are left at clip_hparams defaults; this matches
    // clip.cpp's PROJECTOR_TYPE_MINICPMV / MINICPMV4_6 branches which both
    // comment "use default llava-uhd preprocessing params".

    ctx.image_preproc = std::make_unique<mtmd_image_preprocessor_llava_uhd>(hp);
}

static bool encode_slice(const context &               ctx,
                         const struct clip_image_f32 & img,
                         float *                       out) {
    const int patch  = ctx.hparams.patch_size;
    const int32_t pw = img.nx / patch;

    std::vector<float> pixels;
    util::pack_pixels_row_of_patches(img, patch, MAX_PATCHES, pixels);

    return backend::predict_single_output(ctx.model_handle, {
        { "pixel_values", pixels.data(), { 1, 3, PATCH_SIZE, PATCH_SIZE * MAX_PATCHES },
          backend::DTYPE_F32 },
        { "patch_w",      &pw,           { 1 },
          backend::DTYPE_I32 },
    }, "output", out);
}

const model_adapter g_adapter = {
    /* name         */ "minicpmv",
    /* detect       */ detect,
    /* setup        */ setup,
    /* encode_slice */ encode_slice,
};

} // namespace mtmd_coreml::models::minicpmv
