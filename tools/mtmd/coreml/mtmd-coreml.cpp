#include "mtmd-coreml.h"

#include "../clip-impl.h"
#include "../clip.h"
#include "backend.h"
#include "models/models.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace mtmd_coreml {

context::context() = default;

context::~context() {
    if (model_handle) {
        backend::unload(model_handle);
        model_handle = nullptr;
    }
}

//
// Registry
//

const std::vector<const model_adapter *> & all_adapters() {
    static const std::vector<const model_adapter *> regs = {
        &models::minicpmv::g_adapter,
        &models::llava::g_adapter,
    };
    return regs;
}

const model_adapter * find_adapter(const mlmodelc_meta &      meta,
                                   const struct llama_model * text_model) {
    const model_adapter * match = nullptr;
    for (auto * a : all_adapters()) {
        if (a->detect(meta, text_model)) {
            if (match) {
                throw std::runtime_error(
                    std::string("ambiguous CoreML adapter selection: both '") + match->name
                    + "' and '" + a->name + "' claim this .mlmodelc; this is a registry bug");
            }
            match = a;
        }
    }
    return match;
}

//
// metadata.json parsing
//

namespace {

// Read input/output schema names from a .mlmodelc/metadata.json bundle.
// metadata.json is a small (< 4 KB) JSON array with a single object that
// always contains "inputSchema", "outputSchema", optionally "userDefinedMetadata".
mlmodelc_meta parse_metadata(const std::string & mlmodelc_path) {
    mlmodelc_meta meta;
    const std::string fn = mlmodelc_path + "/metadata.json";

    std::ifstream f(fn);
    if (!f.is_open()) {
        throw std::runtime_error("failed to open metadata.json at " + fn);
    }

    nlohmann::json j;
    try {
        f >> j;
    } catch (const nlohmann::json::exception & e) {
        throw std::runtime_error(std::string("invalid metadata.json: ") + e.what());
    }

    // metadata.json is wrapped in a one-element array
    if (!j.is_array() || j.empty() || !j[0].is_object()) {
        throw std::runtime_error("unexpected metadata.json layout (not a non-empty array)");
    }
    const auto & root = j[0];

    auto collect_names = [](const nlohmann::json & arr, std::set<std::string> & out) {
        if (!arr.is_array()) return;
        for (const auto & item : arr) {
            if (item.is_object() && item.contains("name") && item["name"].is_string()) {
                out.insert(item["name"].get<std::string>());
            }
        }
    };

    // mlmodelc/metadata.json stores the shape as a JSON string like
    // "[1, 64, 2560]" (NOT as a JSON array). We re-parse the inner string
    // back through nlohmann so we don't roll our own integer-list parser.
    auto parse_shape_string = [](const std::string & s) -> std::vector<int64_t> {
        std::vector<int64_t> out;
        try {
            auto inner = nlohmann::json::parse(s);
            if (inner.is_array()) {
                for (const auto & v : inner) {
                    if (v.is_number_integer()) out.push_back(v.get<int64_t>());
                }
            }
        } catch (...) {
            // leave `out` empty; adapter should fail detect() on missing shape
        }
        return out;
    };

    auto collect_shapes = [&](const nlohmann::json & arr,
                              std::map<std::string, std::vector<int64_t>> & out) {
        if (!arr.is_array()) return;
        for (const auto & item : arr) {
            if (!item.is_object()) continue;
            if (!item.contains("name") || !item["name"].is_string()) continue;
            const std::string nm = item["name"].get<std::string>();
            if (item.contains("shape") && item["shape"].is_string()) {
                out[nm] = parse_shape_string(item["shape"].get<std::string>());
            }
        }
    };

    if (root.contains("inputSchema"))  collect_names (root["inputSchema"],  meta.input_names);
    if (root.contains("outputSchema")) collect_names (root["outputSchema"], meta.output_names);
    if (root.contains("outputSchema")) collect_shapes(root["outputSchema"], meta.output_shapes);

    if (root.contains("userDefinedMetadata") && root["userDefinedMetadata"].is_object()) {
        const auto & udm = root["userDefinedMetadata"];
        if (udm.contains("base_model") && udm["base_model"].is_string()) {
            meta.base_model = udm["base_model"].get<std::string>();
        }
    }

    return meta;
}

} // namespace

//
// init / encode
//

std::unique_ptr<context> init_from_file(const char *               mlmodelc_path,
                                        const struct llama_model * text_model) {
    if (!mlmodelc_path || !mlmodelc_path[0]) {
        return nullptr;
    }

    mlmodelc_meta meta;
    try {
        meta = parse_metadata(mlmodelc_path);
    } catch (const std::exception & e) {
        LOG_ERR("%s: %s\n", __func__, e.what());
        return nullptr;
    }

    const model_adapter * adapter = find_adapter(meta, text_model);
    if (!adapter) {
        std::string have = "{";
        for (auto it = meta.input_names.begin(); it != meta.input_names.end(); ++it) {
            if (it != meta.input_names.begin()) have += ", ";
            have += *it;
        }
        have += "}";
        std::string known;
        for (auto * a : all_adapters()) {
            if (!known.empty()) known += ", ";
            known += a->name;
        }
        LOG_ERR("%s: no CoreML adapter matches this .mlmodelc\n"
                "  detected inputs: %s\n"
                "  available adapters: %s\n"
                "  hint: bundle may be from an unsupported model family\n",
                __func__, have.c_str(), known.c_str());
        return nullptr;
    }

    auto ctx = std::unique_ptr<context>(new context());
    ctx->adapter    = adapter;
    ctx->model_path = mlmodelc_path;

    // Adapter fills hparams, builds preprocessor, and sets
    // n_embd_out / n_tokens_out from meta.output_shapes.
    adapter->setup(*ctx, meta, text_model);

    if (ctx->n_embd_out <= 0 || ctx->n_tokens_out <= 0 || !ctx->image_preproc) {
        LOG_ERR("%s: adapter '%s' setup() left context in an incomplete state "
                "(n_embd_out=%d, n_tokens_out=%d, preproc=%s)\n",
                __func__, adapter->name,
                ctx->n_embd_out, ctx->n_tokens_out,
                ctx->image_preproc ? "ok" : "null");
        return nullptr;
    }

    ctx->model_handle = backend::load(mlmodelc_path);
    if (!ctx->model_handle) {
        return nullptr;
    }

    LOG_INF("%s: CoreML backend ready: adapter=%s, n_embd=%d, n_tokens/slice=%d\n",
            __func__, adapter->name, ctx->n_embd_out, ctx->n_tokens_out);

    return ctx;
}

bool encode(context & ctx, const struct clip_image_f32 & img, float * out) {
    if (!ctx.adapter || !ctx.model_handle || !out) {
        return false;
    }
    return ctx.adapter->encode_slice(ctx, img, out);
}

//
// util
//

namespace util {

void pack_pixels_row_of_patches(const struct clip_image_f32 & img,
                                int                          patch_size,
                                int                          max_patches,
                                std::vector<float> &         out) {
    const int nx = max_patches * patch_size;
    const int ny = patch_size;
    const int n  = nx * ny;
    out.assign((size_t)3 * n, 0.0f);

    const int image_w = img.nx;
    const int image_h = img.ny;

    int patch_index = 0;
    for (int i = 0; i < image_h && patch_index < max_patches; i += patch_size) {
        for (int j = 0; j < image_w && patch_index < max_patches; j += patch_size) {
            for (int pi = 0; pi < patch_size; ++pi) {
                for (int pj = 0; pj < patch_size; ++pj) {
                    const int src = ((i + pi) * image_w + (j + pj)) * 3;
                    const int dst = nx * pi + patch_index * patch_size + pj;
                    out[dst]         = img.buf[src];
                    out[n     + dst] = img.buf[src + 1];
                    out[2 * n + dst] = img.buf[src + 2];
                }
            }
            ++patch_index;
        }
    }
}

} // namespace util

} // namespace mtmd_coreml
