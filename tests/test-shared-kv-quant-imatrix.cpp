#include "../src/llama-ext.h"
#include "../src/llama-model.h"
#include "ggml-cpp.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Regression test for shared-KV models + imatrix-dependent quantization.
// Models with shared KV layers (where layers >= n_layer_kv_from_start reuse
// KV cache from earlier layers) have attn_k/attn_v tensors that are unused
// at inference. The imatrix collector never records data for them, so
// quantization types that require an imatrix must fall back gracefully.

int main() {
    printf("test-shared-kv-quant-imatrix: Starting\n");

    // Create a mock model with shared KV layers
    llama_quant_model_desc desc = {};
    desc.architecture  = "llama";
    desc.n_embd        = 2048;
    desc.n_embd_head_k = 128;
    desc.n_embd_head_v = 128;
    desc.n_layer       = 24;
    desc.n_head        = 16;
    desc.n_head_kv     = 2;
    desc.n_ff          = 8192;
    desc.n_expert      = 0;

    llama_model * model = llama_quant_model_from_metadata(&desc);
    if (!model) {
        fprintf(stderr, "FAIL: llama_quant_model_from_metadata returned nullptr\n");
        return 1;
    }

    // Simulate shared KV: first 12 layers have KV, layers 12-23 share
    model->hparams.n_layer_kv_from_start = 12;

    printf("  Model: n_layer=%u, n_layer_kv_from_start=12\n", desc.n_layer);

    // Use an imatrix-dependent quantization type, with no imatrix provided
    llama_model_quantize_params qparams = llama_model_quantize_default_params();
    qparams.ftype                  = LLAMA_FTYPE_MOSTLY_IQ2_M;
    qparams.allow_requantize       = false;
    qparams.quantize_output_tensor = true;
    qparams.only_copy              = false;
    qparams.pure                   = false;
    qparams.imatrix                = nullptr;  // no imatrix — this is the key scenario

    quantize_state_impl * qs = llama_quant_init(model, &qparams);
    if (!qs) {
        fprintf(stderr, "FAIL: llama_quant_init returned nullptr\n");
        llama_model_free(model);
        return 1;
    }

    struct mock_tensor_data {
        std::string name;
        int64_t     ne[4];
    };

    std::vector<mock_tensor_data> tensor_specs = {
        { "blk.5.attn_k.weight",  { 2048, 256, 1, 1 } },  // non-shared layer
        { "blk.5.attn_v.weight",  { 2048, 256, 1, 1 } },  // non-shared layer
        { "blk.15.attn_k.weight", { 2048, 256, 1, 1 } },  // shared-KV layer
        { "blk.15.attn_v.weight", { 2048, 256, 1, 1 } },  // shared-KV layer
        { "blk.22.attn_k.weight", { 2048, 256, 1, 1 } },  // shared-KV layer
    };

    std::vector<ggml_tensor*> tensors;
    std::vector<ggml_type>    result_types;
    tensors.reserve(tensor_specs.size());
    result_types.resize(tensor_specs.size());

    ggml_init_params params = {
        .mem_size   = 1024 * 1024 * 10,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "FAIL: ggml_init failed\n");
        llama_quant_free(qs);
        llama_model_free(model);
        return 1;
    }

    for (const auto & spec : tensor_specs) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, spec.ne[0], spec.ne[1], spec.ne[2], spec.ne[3]);
        ggml_set_name(t, spec.name.c_str());
        tensors.push_back(t);
    }

    // This should not crash — shared-KV tensors should get Q8_0 fallback
    llama_quant_compute_types(qs, qparams.ftype, tensors.data(), result_types.data(), tensors.size());

    printf("  Results:\n");
    bool all_ok = true;

    for (size_t i = 0; i < tensors.size(); i++) {
        const char * name = ggml_get_name(tensors[i]);
        printf("    %-30s -> %s\n", name, ggml_type_name(result_types[i]));

        int layer = -1;
        sscanf(name, "blk.%d.", &layer);

        if (layer >= 12) {
            // shared-KV layer: must get Q8_0 fallback (no imatrix needed)
            if (result_types[i] != GGML_TYPE_Q8_0) {
                fprintf(stderr, "FAIL: Expected Q8_0 for %s (shared-KV, unused at inference), got %s\n",
                        name, ggml_type_name(result_types[i]));
                all_ok = false;
            }
        } else {
            // non-shared layer: should get a real quantization type
            if (result_types[i] == GGML_TYPE_F16 || result_types[i] == GGML_TYPE_COUNT) {
                fprintf(stderr, "FAIL: Unexpected type %s for %s (non-shared layer)\n",
                        ggml_type_name(result_types[i]), name);
                all_ok = false;
            }
        }
    }

    ggml_free(ctx);
    llama_quant_free(qs);
    llama_model_free(model);

    if (all_ok) {
        printf("PASS: test-shared-kv-quant-imatrix\n");
        return 0;
    } else {
        fprintf(stderr, "FAIL: test-shared-kv-quant-imatrix\n");
        return 1;
    }
}
