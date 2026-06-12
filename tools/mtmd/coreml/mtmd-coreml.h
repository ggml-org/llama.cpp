#pragma once

// !!! Internal header for the mtmd library !!!
// It is not installed as a public header.

#define MTMD_INTERNAL_HEADER

#include "clip-model.h"
#include "mtmd-image.h"

#include "llama.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace mtmd_coreml {

struct context;

// Lightweight view of the relevant fields from a .mlmodelc/metadata.json.
// Parsed once at init_from_file() time and passed to each adapter so
// detection + shape introspection are 100% input-driven (no filename
// heuristics, no MLModel load required for dispatch).
struct mlmodelc_meta {
    std::set<std::string>                            input_names;   // e.g. {"pixel_values", "patch_w"}
    std::set<std::string>                            output_names;  // e.g. {"output"}
    std::map<std::string, std::vector<int64_t>>      output_shapes; // name -> shape (parsed from "[1, 64, 2560]")
    std::string                                      base_model;    // "MiniCPM-V 4.6", may be empty
};

// One CoreML adapter == one model family supported on Apple hardware.
// Each adapter is fully self-contained: it knows how to detect its own
// .mlmodelc, fill the context (hparams + preprocessor + output shape) from
// the bundle's metadata, and run a single preprocessed slice through CoreML.
//
// Adding a new model family = adding a new file under coreml/models/, plus
// one line in coreml/models/models.h and one line in the registry in
// mtmd-coreml.cpp. No changes to mtmd-coreml.cpp / mtmd.cpp dispatch logic.
struct model_adapter {
    const char * name; // stable identifier, e.g. "minicpmv"

    // Return true if THIS adapter handles the given .mlmodelc + text model.
    // Detection is driven by `meta` (input/output schema from metadata.json)
    // for stability across renamed bundles.
    bool (*detect)(const mlmodelc_meta &      meta,
                   const struct llama_model * text_model);

    // Populate `ctx` from `meta` + `text_model`: fill hparams, build the
    // host-side preprocessor, and set the output-shape contract
    // (`n_embd_out`, `n_tokens_out`). The output dims are typically read
    // directly from `meta.output_shapes`, which lets one adapter handle
    // multiple variants that differ only in LLM hidden size (e.g. all
    // MiniCPM-V perceiver-resampler bundles).
    void (*setup)(context &                  ctx,
                  const mlmodelc_meta &      meta,
                  const struct llama_model * text_model);

    // Encode a single preprocessed slice. Writes
    // `ctx.n_tokens_out * ctx.n_embd_out` floats to `out`. The adapter owns
    // input-tensor packing (pack pixels, derive patch_w, etc.) and calls
    // backend::predict_*.
    bool (*encode_slice)(const context &               ctx,
                         const struct clip_image_f32 & img,
                         float *                       out);
};

// All registered adapters (defined in mtmd-coreml.cpp).
const std::vector<const model_adapter *> & all_adapters();

// Pick the unique adapter that matches (meta, text_model). Returns nullptr
// if no adapter claims to handle the bundle. Throws std::runtime_error if
// more than one adapter claims it (which would indicate overlapping detect()
// rules and is treated as a bug, not a user error).
const model_adapter * find_adapter(const mlmodelc_meta &      meta,
                                   const struct llama_model * text_model);

// Owned by mtmd_context (one per CoreML-enabled session).
struct context {
    const model_adapter * adapter = nullptr;

    // Owns hparams driving the borrowed preprocessor. Lifetime contract:
    // image_preproc holds a reference to this hparams, so context must
    // outlive image_preproc when it is moved into mtmd_context.
    clip_hparams hparams;

    // Built against `hparams`. After init_from_file() returns successfully,
    // ownership of this preprocessor is moved into mtmd_context.
    std::unique_ptr<mtmd_image_preprocessor> image_preproc;

    // Backend handle (opaque MLModel pointer); freed by backend::unload().
    void *      model_handle = nullptr;
    std::string model_path;

    // Output-shape contract, filled by the adapter's setup() from
    // mlmodelc metadata (or hard-coded constants for adapters that don't
    // vary across bundles).
    int n_embd_out   = 0; // must match text model n_embd
    int n_tokens_out = 0; // tokens produced per single slice

    context();
    ~context();

    context(const context &)             = delete;
    context & operator=(const context &) = delete;
};

// Load a CoreML .mlmodelc and select the right adapter for the given text
// model. Returns nullptr on any failure (bundle missing, unsupported model,
// adapter rejected the schema, backend load failed).
std::unique_ptr<context> init_from_file(const char *               mlmodelc_path,
                                        const struct llama_model * text_model);

// Convenience accessors used by mtmd_context.
inline int n_embd          (const context & ctx) { return ctx.n_embd_out;   }
inline int n_output_tokens (const context & ctx) { return ctx.n_tokens_out; }

// Encode one preprocessed image slice through the bound adapter.
bool encode(context & ctx, const struct clip_image_f32 & img, float * out);

// Shared utilities reused by multiple adapters. Keep these strictly
// generic; family-specific math lives inside its adapter file.
namespace util {

// Pack a preprocessed image into the canonical ViT input layout
// `[3, patch_size, patch_size * max_patches]`, zero-padded when the image has
// fewer than max_patches patches. This is the input shape consumed by every
// MiniCPM-V CoreML bundle shipped so far; new families that share the same
// row-of-patches layout can reuse it.
void pack_pixels_row_of_patches(const struct clip_image_f32 & img,
                                int                          patch_size,
                                int                          max_patches,
                                std::vector<float> &         out);

} // namespace util

} // namespace mtmd_coreml
