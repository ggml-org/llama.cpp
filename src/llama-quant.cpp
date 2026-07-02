#include "llama-ext.h"
#include "llama-model-loader.h"
#include "llama-model.h"
#include "llama-quant.h"

#include <atomic>
#include <cinttypes>
#include <csignal>
#include <fstream>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <regex>
#include <thread>

// result of parsing --tensor-type option
// (changes to this struct must be reflected in tools/quantize/quantize.cpp)
struct tensor_type_option {
    std::string name;
    ggml_type   type = GGML_TYPE_COUNT;
};

// tensor categorization - used to avoid repeated string matching in quantization logic.
// this is different from LLM_TN - we want broad categories, not specific tensor names per arch.
enum tensor_category {
    TENSOR_CATEGORY_TOKEN_EMBD,
    TENSOR_CATEGORY_ATTENTION_Q,
    TENSOR_CATEGORY_ATTENTION_V,
    TENSOR_CATEGORY_ATTENTION_K,
    TENSOR_CATEGORY_ATTENTION_QKV,
    TENSOR_CATEGORY_ATTENTION_KV_B,
    TENSOR_CATEGORY_ATTENTION_OUTPUT,
    TENSOR_CATEGORY_FFN_UP,
    TENSOR_CATEGORY_FFN_GATE,
    TENSOR_CATEGORY_FFN_DOWN,
    TENSOR_CATEGORY_OUTPUT,
    TENSOR_CATEGORY_OTHER
};

static void zeros(std::ofstream & file, const size_t n) {
    constexpr char zero = 0;
    for (size_t i = 0; i < n; ++i) { file.write(& zero, 1); }
}

static std::string remap_layer(const std::string & orig_name, const std::vector<int> & prune, std::map<int, std::string> & mapped, int & next_id) {
    if (prune.empty()) { return orig_name; }

    static const std::regex pattern(R"(blk\.(\d+)\.)");
    if (std::smatch match; std::regex_search(orig_name, match, pattern)) {
        const int blk = std::stoi(match[1]);
        std::string new_name = orig_name;

        if (mapped.count(blk)) {
            // Already mapped, do nothing
        } else if (std::find(prune.begin(), prune.end(), blk) != prune.end()) {
            mapped[blk] = "";
        } else if (blk < prune.front()) {
            mapped[blk] = std::to_string(blk);
            next_id = blk + 1;
        } else {
            mapped[blk] = std::to_string(next_id);
            ++next_id;
        }

        return mapped[blk].empty() ? mapped[blk] : new_name.replace(match.position(1), match.length(1), mapped[blk]);
    }

    return orig_name;
}

static std::string remap_imatrix(const std::string & orig_name, const std::map<int, std::string> & mapped) {
    if (mapped.empty()) { return orig_name; }

    static const std::regex pattern(R"(blk\.(\d+)\.)");
    if (std::smatch match; std::regex_search(orig_name, match, pattern)) {
        const std::string blk(match[1]);
        std::string new_name = orig_name;

        for (const auto & p : mapped) {
            if (p.second == blk) { return new_name.replace(match.position(1), match.length(1), std::to_string(p.first)); }
        }

        GGML_ABORT("\n%s: imatrix mapping error for %s\n", __func__, orig_name.c_str());
    }

    return orig_name;
}

// helper functions for tensor name matching
static bool tensor_name_match_token_embd(const char * tensor_name) {
    return std::strcmp(tensor_name, "token_embd.weight") == 0 ||
           std::strcmp(tensor_name, "per_layer_token_embd.weight") == 0;
}

static bool tensor_name_match_output_weight(const char * tensor_name) {
    return std::strcmp(tensor_name, "output.weight") == 0;
}

// tensor categorization for quantization
static tensor_category tensor_get_category(const std::string & tensor_name) {
    if (tensor_name_match_output_weight(tensor_name.c_str()))           { return TENSOR_CATEGORY_OUTPUT; }
    if (tensor_name_match_token_embd(tensor_name.c_str()))              { return TENSOR_CATEGORY_TOKEN_EMBD; }
    if (tensor_name.find("attn_qkv.weight") != std::string::npos)    { return TENSOR_CATEGORY_ATTENTION_QKV; }
    if (tensor_name.find("attn_kv_b.weight") != std::string::npos)   { return TENSOR_CATEGORY_ATTENTION_KV_B; }
    if (tensor_name.find("attn_v.weight") != std::string::npos)      { return TENSOR_CATEGORY_ATTENTION_V; }
    if (tensor_name.find("attn_k.weight") != std::string::npos)      { return TENSOR_CATEGORY_ATTENTION_K; }
    if (tensor_name.find("attn_q.weight") != std::string::npos)      { return TENSOR_CATEGORY_ATTENTION_Q; }
    if (tensor_name.find("attn_output.weight") != std::string::npos) { return TENSOR_CATEGORY_ATTENTION_OUTPUT; }
    if (tensor_name.find("ffn_up") != std::string::npos)             { return TENSOR_CATEGORY_FFN_UP; }
    if (tensor_name.find("ffn_gate") != std::string::npos)           { return TENSOR_CATEGORY_FFN_GATE; }
    if (tensor_name.find("ffn_down") != std::string::npos)           { return TENSOR_CATEGORY_FFN_DOWN; }

    return TENSOR_CATEGORY_OTHER;
}

// check if category is for attention-v-like tensors (more sensitive to quantization)
static bool category_is_attn_v(const tensor_category cat) {
    return cat == TENSOR_CATEGORY_ATTENTION_V || cat == TENSOR_CATEGORY_ATTENTION_QKV || cat == TENSOR_CATEGORY_ATTENTION_KV_B;
}

// quantization state
struct quantize_state_impl {
    const llama_model                 & model;
    const llama_model_quantize_params * params;

    int n_attention_wv = 0;
    int n_ffn_down     = 0;
    int n_ffn_gate     = 0;
    int n_ffn_up       = 0;
    int i_attention_wv = 0;
    int i_ffn_down     = 0;
    int i_ffn_gate     = 0;
    int i_ffn_up       = 0;
    int n_fallback    = 0;
    bool has_imatrix = false;

    // used to figure out if a model has tied embeddings (tok_embd shares weights with output)
    bool has_tied_embeddings = true; // assume tied until we see output.weight
    bool has_activations = false;

    // tensor type override patterns (compiled once, used twice)
    std::vector<std::pair<std::regex, ggml_type>> tensor_type_patterns;

    quantize_state_impl(const llama_model & model, const llama_model_quantize_params * params) : model(model), params(params) {
        // compile regex patterns once - they are expensive
        if (params->tt_overrides) {
            for (const auto * o = params->tt_overrides; o->pattern != nullptr; o++) {
                tensor_type_patterns.emplace_back(std::regex(o->pattern), o->type);
            }
        }
    }
};

// per-tensor metadata, computed in the preliminary loop and used in the main loop
struct tensor_metadata {
    std::string     name;
    ggml_type       target_type;
    tensor_category category;
    std::string     remapped_imatrix_name;
    bool            allows_quantization;
    bool            requires_imatrix;
};

// dequantization
static void llama_tensor_dequantize_impl(ggml_tensor * tensor,
    std::vector<no_init<float>> & output,
    std::vector<std::thread> & workers,
    const size_t nelements,
    const int nthread
) {
    if (output.size() < nelements) { output.resize(nelements); }
    auto f32_output = (float *)output.data();

    const ggml_type_traits * qtype = ggml_get_type_traits(tensor->type);
    if (ggml_is_quantized(tensor->type)) {
        if (qtype->to_float == nullptr) {
            throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available", ggml_type_name(tensor->type)));
        }
    } else if (tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_BF16) {
        throw std::runtime_error(format("cannot dequantize/convert tensor type %s", ggml_type_name(tensor->type)));
    }

    if (nthread < 2) {
        if (tensor->type == GGML_TYPE_F16) { ggml_fp16_to_fp32_row((ggml_fp16_t *)tensor->data, f32_output, nelements); }
        else if (tensor->type == GGML_TYPE_BF16) { ggml_bf16_to_fp32_row((ggml_bf16_t *)tensor->data, f32_output, nelements); }
        else if (ggml_is_quantized(tensor->type)) { qtype->to_float(tensor->data, f32_output, nelements); }
        else { GGML_ABORT("fatal error"); }

        return;
    }

    size_t block_size;
    if (tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) { block_size = 1; }
    else { block_size = (size_t)ggml_blck_size(tensor->type); }

    const size_t block_size_bytes = ggml_type_size(tensor->type);

    GGML_ASSERT(nelements % block_size == 0);
    const size_t nblocks = nelements / block_size;
    const size_t blocks_per_thread = nblocks / nthread;
    const size_t spare_blocks = nblocks - (blocks_per_thread * nthread); // if blocks aren't divisible by thread count

    size_t in_buff_offs = 0;
    size_t out_buff_offs = 0;

    for (int tnum = 0; tnum < nthread; tnum++) {
        const size_t thr_blocks = blocks_per_thread + (tnum == nthread - 1 ? spare_blocks : 0); // num blocks for this thread
        size_t thr_elems = thr_blocks * block_size; // number of elements for this thread
        const size_t thr_block_bytes = thr_blocks * block_size_bytes; // number of input bytes for this thread

        auto compute = [qtype] (ggml_type typ, uint8_t * inbuf, float * outbuf, int nels) {
            if (typ == GGML_TYPE_F16) { ggml_fp16_to_fp32_row((ggml_fp16_t *)inbuf, outbuf, nels); }
            else if (typ == GGML_TYPE_BF16) { ggml_bf16_to_fp32_row((ggml_bf16_t *)inbuf, outbuf, nels); }
            else { qtype->to_float(inbuf, outbuf, nels); }
        };
        workers.emplace_back(compute, tensor->type, (uint8_t *) tensor->data + in_buff_offs, f32_output + out_buff_offs, thr_elems);
        in_buff_offs += thr_block_bytes;
        out_buff_offs += thr_elems;
    }
    for (auto & w : workers) { w.join(); }
    workers.clear();
}

// do we allow this tensor to be quantized?
static bool tensor_allows_quantization(const llama_model_quantize_params * params, llm_arch arch, const ggml_tensor * tensor) {
    if (params->only_copy) { return false; }

    // quantize only 2D and 3D tensors (experts)
    if (ggml_n_dims(tensor) < 2) { return false; }

    const std::string name = ggml_get_name(tensor);

    // ends with 'weight'?
    bool quantize = name.rfind("weight") == name.size() - 6;

    // do not quantize norm tensors
    quantize &= name.find("_norm.weight") == std::string::npos;

    quantize &= params->quantize_output_tensor || name != "output.weight";

    // do not quantize expert gating tensors
    quantize &= name.find("ffn_gate_inp.weight") == std::string::npos;

    // these are very small (e.g. 4x4)
    quantize &= name.find("altup")  == std::string::npos;
    quantize &= name.find("laurel") == std::string::npos;

    // these are not too big so keep them as it is
    quantize &= name.find("per_layer_model_proj") == std::string::npos;

    // do not quantize positional embeddings and token types (BERT)
    quantize &= name != LLM_TN(arch)(LLM_TENSOR_POS_EMBD,    "weight");
    quantize &= name != LLM_TN(arch)(LLM_TENSOR_TOKEN_TYPES, "weight");

    // do not quantize Mamba/Kimi's small conv1d weights
    quantize &= name.find("ssm_conv1d") == std::string::npos;
    quantize &= name.find("shortconv.conv.weight") == std::string::npos;

    // do not quantize RWKV's small yet 2D weights
    quantize &= name.find("time_mix_first.weight") == std::string::npos;
    quantize &= name.find("time_mix_w0.weight") == std::string::npos;
    quantize &= name.find("time_mix_w1.weight") == std::string::npos;
    quantize &= name.find("time_mix_w2.weight") == std::string::npos;
    quantize &= name.find("time_mix_v0.weight") == std::string::npos;
    quantize &= name.find("time_mix_v1.weight") == std::string::npos;
    quantize &= name.find("time_mix_v2.weight") == std::string::npos;
    quantize &= name.find("time_mix_a0.weight") == std::string::npos;
    quantize &= name.find("time_mix_a1.weight") == std::string::npos;
    quantize &= name.find("time_mix_a2.weight") == std::string::npos;
    quantize &= name.find("time_mix_g1.weight") == std::string::npos;
    quantize &= name.find("time_mix_g2.weight") == std::string::npos;
    quantize &= name.find("time_mix_decay_w1.weight") == std::string::npos;
    quantize &= name.find("time_mix_decay_w2.weight") == std::string::npos;
    quantize &= name.find("time_mix_lerp_fused.weight") == std::string::npos;

    // do not quantize relative position bias (T5)
    quantize &= name.find("attn_rel_b.weight") == std::string::npos;

    // do not quantize specific multimodal tensors
    quantize &= name.find(".position_embd") == std::string::npos;
    quantize &= name.find("sam.pos_embd")   == std::string::npos;
    quantize &= name.find("sam.neck.")      == std::string::npos;
    quantize &= name.find("sam.net_")       == std::string::npos;
    quantize &= name.find(".rel_pos")       == std::string::npos;
    quantize &= name.find(".patch_embd")    == std::string::npos;
    quantize &= name.find(".patch_merger")  == std::string::npos;

    return quantize;
}

// mapping from an incompatible target type to its block-size-32 substitute
static ggml_type fallback_type_for(const ggml_type target_type) {
    switch (target_type) {
        // types on the left, block size 256; types on the right, block size 32
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS: return GGML_TYPE_IQ4_NL;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:  return GGML_TYPE_Q4_0;
        case GGML_TYPE_Q4_K:   return GGML_TYPE_Q5_0;
        case GGML_TYPE_Q5_K:   return GGML_TYPE_Q5_1;
        case GGML_TYPE_Q6_K:   return GGML_TYPE_Q8_0;
        default:
            throw std::runtime_error(format("no tensor type fallback is defined for type %s", ggml_type_name(target_type)));
    }
}

// incompatible tensor shapes are handled here - fallback to a compatible type
static ggml_type tensor_type_fallback(quantize_state_impl & qs, const ggml_tensor * t, const ggml_type target_type) {
    const int64_t ncols = t->ne[0];
    const int64_t qk_k = ggml_blck_size(target_type);
    if (ncols % qk_k == 0) { return target_type; } // this tensor's shape is compatible with this quant
    ++qs.n_fallback;
    LLAMA_LOG_WARN("warning: %-36s - ncols %6" PRId64 " not divisible by %3" PRId64 " (required for type %7s) ",
        t->name, ncols, qk_k, ggml_type_name(target_type));

    ggml_type fallback_type = fallback_type_for(target_type);
    if (ncols % ggml_blck_size(fallback_type) != 0) {
        //
        // the fallback return type is still not compatible for this tensor!
        //
        // most likely, this tensor's first dimension is not divisible by 32.
        // this is very rare. we can either abort the quantization, or
        // fallback to F16 / F32.
        //
        LLAMA_LOG_WARN("(WARNING: must use F16 due to unusual shape) ");
        fallback_type = GGML_TYPE_F16;
    }
    LLAMA_LOG_WARN("-> falling back to %7s\n", ggml_type_name(fallback_type));

    return fallback_type;
}

// select the target tensor type based on tensor category, ftype, and model arch
static ggml_type llama_tensor_get_type_impl(quantize_state_impl & qs, ggml_type new_type, const ggml_tensor * tensor, llama_ftype ftype, tensor_category category) {
    const std::string name = ggml_get_name(tensor);

    // TODO: avoid hardcoded tensor names - use the TN_* constants
    const llm_arch arch = qs.model.arch;
    auto use_more_bits = [](const int i_layer, const int n_layers) -> bool {
        return i_layer < n_layers / 8 || i_layer >= 7 * n_layers / 8 || (i_layer - n_layers / 8) % 3 == 2;
    };

    const int n_expert = std::max(1, (int)qs.model.hparams.n_expert);
    auto layer_info = [n_expert] (int i_layer, int n_layer, const char * tname) {
        if (n_expert > 1) {
            // Experts may not be consecutive and simply dividing i_ffn_down by n_expert does not work,
            // so we need to parse the tensor name
            if (sscanf(tname, "blk.%d.", &i_layer) != 1) { throw std::runtime_error(format("Failed to determine layer for tensor %s", tname)); }
            if (i_layer < 0 || i_layer >= n_layer) { throw std::runtime_error(format("Bad layer %d for tensor %s. Must be in [0, %d)", i_layer, tname, n_layer)); }
        }

        return std::make_pair(i_layer, n_layer);
    };

    // for arches that share the same tensor between the token embeddings and the output, we quantize the token embeddings
    // with the quantization of the output tensor
    if (category == TENSOR_CATEGORY_OUTPUT || (qs.has_tied_embeddings && category == TENSOR_CATEGORY_TOKEN_EMBD)) {
        if (qs.params->output_tensor_type < GGML_TYPE_COUNT) { new_type = qs.params->output_tensor_type; }
        else {
            const int64_t nx = tensor->ne[0];
            const int64_t qk_k = ggml_blck_size(new_type);

            if (ftype == LLAMA_FTYPE_MOSTLY_MXFP4_MOE) {
                new_type = GGML_TYPE_Q8_0;
            }
            else if (arch == LLM_ARCH_FALCON || nx % qk_k != 0) {
                new_type = GGML_TYPE_Q8_0;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ1_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ2_S  || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M   ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
                new_type = GGML_TYPE_Q5_K;
            }
            else if (new_type != GGML_TYPE_Q8_0) {
                new_type = GGML_TYPE_Q6_K;
            }
        }
    } else if (ftype == LLAMA_FTYPE_MOSTLY_MXFP4_MOE) {
        if (tensor->ne[2] > 1) {
            // MoE   tensors -> MXFP4
            new_type = GGML_TYPE_MXFP4;
        } else {
            // other tensors -> Q8_0
            new_type = GGML_TYPE_Q8_0;
        }
    } else if (category == TENSOR_CATEGORY_TOKEN_EMBD) {
        if (qs.params->token_embedding_type < GGML_TYPE_COUNT) {
            new_type = qs.params->token_embedding_type;
        } else {
            if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS ||
                ftype == LLAMA_FTYPE_MOSTLY_IQ1_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
                new_type = GGML_TYPE_Q2_K;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M) {
                new_type = GGML_TYPE_IQ3_S;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
                new_type = GGML_TYPE_IQ3_S;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_TQ1_0 || ftype == LLAMA_FTYPE_MOSTLY_TQ2_0) {
                new_type = GGML_TYPE_Q4_K;
            }
        }
    } else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ1_S ||
               ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M    || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
        if (category_is_attn_v(category)) {
            if (qs.model.hparams.n_gqa() >= 4 || qs.model.hparams.n_expert >= 4) { new_type = GGML_TYPE_Q4_K; }
            else { new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K; }

            ++qs.i_attention_wv;
        }
        else if (qs.model.hparams.n_expert == 8 && category == TENSOR_CATEGORY_ATTENTION_K) { new_type = GGML_TYPE_Q4_K; }
        else if (category == TENSOR_CATEGORY_FFN_DOWN) {
            if (qs.i_ffn_down < qs.n_ffn_down/8) {
                new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            }

            ++qs.i_ffn_down;
        }
        else if (category == TENSOR_CATEGORY_ATTENTION_OUTPUT) {
            if (qs.model.hparams.n_expert == 8) { new_type = GGML_TYPE_Q5_K; }
            else {
                if (ftype == LLAMA_FTYPE_MOSTLY_IQ1_S || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) { new_type = GGML_TYPE_IQ2_XXS; }
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M) { new_type = GGML_TYPE_IQ3_S; }
            }
        }
    } else if (category_is_attn_v(category)) {
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) { new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K; }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S && qs.model.hparams.n_gqa() >= 4) { new_type = GGML_TYPE_Q4_K; }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : !qs.has_imatrix ? GGML_TYPE_IQ3_S : GGML_TYPE_IQ3_XXS;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S) && qs.model.hparams.n_gqa() >= 4) { new_type = GGML_TYPE_Q4_K; }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) { new_type = GGML_TYPE_Q4_K; }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) { new_type = qs.i_attention_wv < 2 ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K; }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) { new_type = GGML_TYPE_Q5_K; }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && qs.model.hparams.n_gqa() >= 4) { new_type = GGML_TYPE_Q5_K; }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) &&
                use_more_bits(qs.i_attention_wv, qs.n_attention_wv)) { new_type = GGML_TYPE_Q6_K; }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && qs.i_attention_wv < 4) { new_type = GGML_TYPE_Q5_K; }
        if (qs.model.type == LLM_TYPE_70B) {
            // In the 70B model we have 8 heads sharing the same attn_v weights. As a result, the attn_v.weight tensor is
            // 8x smaller compared to attn_q.weight. Hence, we can get a nice boost in quantization accuracy with
            // nearly negligible increase in model size by quantizing this tensor with more bits:
            if (new_type == GGML_TYPE_Q3_K || new_type == GGML_TYPE_Q4_K) { new_type = GGML_TYPE_Q5_K; }
        }
        if (qs.model.hparams.n_expert == 8) {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }

        ++qs.i_attention_wv;
    } else if (category == TENSOR_CATEGORY_ATTENTION_K) {
        if (qs.model.hparams.n_expert == 8) {
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) { new_type = GGML_TYPE_IQ3_XXS; }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) { new_type = GGML_TYPE_IQ2_S; }
    } else if (category == TENSOR_CATEGORY_ATTENTION_Q) {
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) { new_type = GGML_TYPE_IQ3_XXS; }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) { new_type = GGML_TYPE_IQ2_S; }
    } else if (category == TENSOR_CATEGORY_FFN_DOWN) {
        auto info = layer_info(qs.i_ffn_down, qs.n_ffn_down, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) { new_type = GGML_TYPE_Q3_K; }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S) { if (i_layer < n_layer/8) new_type = GGML_TYPE_Q4_K; }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS && !qs.has_imatrix) { new_type = i_layer < n_layer / 8 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K; }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = i_layer < n_layer/16 ? GGML_TYPE_Q5_K
                     : arch != LLM_ARCH_FALCON || use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q4_K
                     : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M && (i_layer < n_layer / 8 || (qs.model.hparams.n_expert == 8 && use_more_bits(i_layer, n_layer)))) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) { new_type = arch == LLM_ARCH_FALCON ? GGML_TYPE_Q4_K : GGML_TYPE_Q5_K; }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) {
            if (arch == LLM_ARCH_FALCON) {
                new_type = i_layer < n_layer / 16 ? GGML_TYPE_Q6_K : use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
            } else if (use_more_bits(i_layer, n_layer)) { new_type = GGML_TYPE_Q6_K; }
        }
        else if (i_layer < n_layer / 8 && (ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && !qs.has_imatrix) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M && use_more_bits(i_layer, n_layer)) { new_type = GGML_TYPE_Q6_K; }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && arch != LLM_ARCH_FALCON && i_layer < n_layer/8) { new_type = GGML_TYPE_Q5_K; }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_0 || ftype == LLAMA_FTYPE_MOSTLY_Q5_0) && qs.has_imatrix && i_layer < n_layer / 8) {
            // Safeguard against instability in the initial ffn_down layers, which can occur with Q4_0/Q5_0 quantization even when an imatrix is used.
            // The rationale is two-fold: first, to guarantee that the resulting quantization remains consistent with the pre-imatrix state,
            // and second, because Q4_1/Q5_1 tend to exhibit instability in ffn_down layers without an imatrix.
            new_type = ftype == LLAMA_FTYPE_MOSTLY_Q4_0 ? GGML_TYPE_Q4_1 : GGML_TYPE_Q5_1;
        }

        ++qs.i_ffn_down;
    } else if (category == TENSOR_CATEGORY_ATTENTION_OUTPUT) {
        if (arch != LLM_ARCH_FALCON) {
            if (qs.model.hparams.n_expert == 8) {
                if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q3_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL  ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S  ||
                    ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) { new_type = GGML_TYPE_Q5_K; }
            } else {
                if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   ) { new_type = GGML_TYPE_Q3_K; }
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) { new_type = GGML_TYPE_IQ3_S; }
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M ) { new_type = GGML_TYPE_Q4_K; }
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L ) { new_type = GGML_TYPE_Q5_K; }
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  ) { new_type = GGML_TYPE_Q4_K; }
            }
        } else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) { new_type = GGML_TYPE_Q4_K; }
    }
    else if (category == TENSOR_CATEGORY_ATTENTION_QKV) {
        if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L || ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) { new_type = GGML_TYPE_Q4_K; }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) { new_type = GGML_TYPE_Q5_K; }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) { new_type = GGML_TYPE_Q6_K; }
    }
    else if (category == TENSOR_CATEGORY_FFN_GATE) {
        auto info = layer_info(qs.i_ffn_gate, qs.n_ffn_gate, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && i_layer >= n_layer / 8 && i_layer < 7 * n_layer / 8) { new_type = GGML_TYPE_IQ3_XXS; }

        ++qs.i_ffn_gate;
    }
    else if (category == TENSOR_CATEGORY_FFN_UP) {
        auto info = layer_info(qs.i_ffn_up, qs.n_ffn_up, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && i_layer >= n_layer / 8 && i_layer < 7 * n_layer / 8) { new_type = GGML_TYPE_IQ3_XXS; }

        ++qs.i_ffn_up;
    }

    return new_type;
}

// determine the ggml_type that this tensor should be quantized to
static ggml_type llama_tensor_get_type(quantize_state_impl & qs,
    const llama_model_quantize_params * params,
    const ggml_tensor * tensor,
    const ggml_type default_type,
    const tensor_metadata & tm
) {
    if (params->target_bpw != -1.0f || params->target_size != -1) { return tensor->type; }  // defer tensor type selection to target_bpw_type()
    if (!tensor_allows_quantization(params, qs.model.arch, tensor)) { return tensor->type; }
    if (params->token_embedding_type < GGML_TYPE_COUNT && tm.category == TENSOR_CATEGORY_TOKEN_EMBD) { return params->token_embedding_type; }
    if (params->output_tensor_type < GGML_TYPE_COUNT && tm.category == TENSOR_CATEGORY_OUTPUT) { return params->output_tensor_type; }

    ggml_type new_type = default_type;

    // get more optimal quantization type based on the tensor shape, layer, etc.
    if (!params->pure && ggml_is_quantized(default_type)) {
        // if the user provided tensor types, use those
        bool manual = false;
        if (!qs.tensor_type_patterns.empty()) {
            const std::string tensor_name(tensor->name);
            for (const auto & [pattern, qtype] : qs.tensor_type_patterns) {
                if (std::regex_search(tensor_name, pattern)) {
                    if (qtype != new_type) {
                        LLAMA_LOG_WARN("%s: %-36s - applying manual override: %s -> %s\n",
                            __func__, tensor_name.c_str(), ggml_type_name(new_type), ggml_type_name(qtype));
                        new_type = qtype;
                    }
                    manual = true;
                    break;
                }
            }
        }

        // otherwise, use the standard logic
        if (!manual) { new_type = llama_tensor_get_type_impl(qs, new_type, tensor, params->ftype, tm.category); }

        // if incompatible tensor shape, fallback to a compatible type
        new_type = tensor_type_fallback(qs, tensor, new_type);
    }

    return new_type;
}

static size_t llama_tensor_quantize_impl(ggml_type new_type,
    const float * f32_data,
    void * new_data,
    const int64_t chunk_size,
    int64_t nrows,
    int64_t n_per_row,
    const float * imatrix,
    std::vector<std::thread> & workers,
    const int nthread
) {
    if (nthread < 2) {
        // single-thread
        size_t new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, nrows, n_per_row, imatrix);
        if (!ggml_validate_row_data(new_type, new_data, new_size)) { throw std::runtime_error("quantized data validation failed"); }

        return new_size;
    }

    std::mutex mutex;
    int64_t counter = 0;
    size_t new_size = 0;
    bool valid = true;
    auto compute = [&mutex, &counter, &new_size, &valid, new_type, f32_data, new_data, chunk_size, nrows, n_per_row, imatrix] {
        const int64_t nrows_per_chunk = chunk_size / n_per_row;
        size_t local_size = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int64_t first_row = counter; counter += nrows_per_chunk;
            if (first_row >= nrows) {
                if (local_size > 0) { new_size += local_size; }
                break;
            }
            lock.unlock();
            const int64_t this_nrow = std::min(nrows - first_row, nrows_per_chunk);
            size_t this_size = ggml_quantize_chunk(new_type, f32_data, new_data, first_row * n_per_row, this_nrow, n_per_row, imatrix);
            local_size += this_size;

            // validate the quantized data
            const size_t row_size  = ggml_row_size(new_type, n_per_row);
            void * this_data = (char *) new_data + first_row * row_size;
            if (!ggml_validate_row_data(new_type, this_data, this_size)) {
                std::unique_lock<std::mutex> lock(mutex);
                valid = false;
                break;
            }
        }
    };

    for (int it = 0; it < nthread - 1; ++it) { workers.emplace_back(compute); }
    compute();
    for (auto & w : workers) { w.join(); }
    workers.clear();
    if (!valid) { throw std::runtime_error("quantized data validation failed"); }

    return new_size;
}

static std::atomic<bool> bpw_stop{ false };

static void signal_handler(int) { bpw_stop.store(true, std::memory_order_relaxed); }

// Returns tensor type overrides that meet a global file size or bpw target
static std::unordered_map<std::string, ggml_type> target_bpw_type(
    llama_model_loader & ml,
    quantize_state_impl & qs,
    const std::vector<const llama_model_loader::llama_tensor_weight *> & tensors,
    const std::map<int, std::string> & mapped,
    const std::unordered_map<std::string, std::vector<float>> * values_data,
    const std::unordered_map<std::string, std::vector<float>> * activations_data,
    const std::unordered_map<std::string, std::vector<float>> * statistics_data,
    const std::unordered_map<std::string, ggml_type> * locked_tensors,
    int nthread
) {
    bpw_stop.store(false, std::memory_order_relaxed);

    // Vector indices for statistics_data's metrics
    enum { ENERGY = 0, MEAN = 1, ELEMENTS = 2, STDDEV = 3, SKEWNESS = 4, KURTOSIS = 5, GAIN = 6,
           H_NORM = 7, L2_DIST = 8, COSSIM = 9, PCC = 10, COVAR = 11 };

    // SIGINT/SIGTERM signal handlers
    struct signal_scope_guard {
        using handler_t = void (*)(int);
        handler_t prev_int = SIG_DFL;
        handler_t prev_term = SIG_DFL;
        signal_scope_guard() {
            prev_int = std::signal(SIGINT, signal_handler);
            prev_term = std::signal(SIGTERM, signal_handler);
        }
        ~signal_scope_guard() {
            std::signal(SIGINT, prev_int);
            std::signal(SIGTERM, prev_term);
        }
    } signal_guard;

    // GGML_TYPE scores
    struct type_scores {
        ggml_type type = GGML_TYPE_COUNT;
        float bpw = 0.0f;
        size_t bytes = 0;
        double error = 0.0;
        double mse = 0.0;
        double wce = 0.0;
    };

    // Tensor quantization type choice
    struct type_choice {
        const llama_model_loader::llama_tensor_weight * w = nullptr;
        std::vector<type_scores> candidates;
        int choice = -1;
        float min_bpw = 0.0;
        float max_bpw = 0.0;
        size_t n_elements = 0;
        bool important = false;
    };

    // Quantization types
    constexpr ggml_type quant_types[] = {
        GGML_TYPE_IQ1_S,
        GGML_TYPE_IQ1_M,
        GGML_TYPE_IQ2_XXS,
        GGML_TYPE_IQ2_XS,
        GGML_TYPE_IQ2_S,
        GGML_TYPE_IQ3_XXS,
        GGML_TYPE_IQ3_S,
        GGML_TYPE_IQ4_XS,
        GGML_TYPE_IQ4_NL,
        GGML_TYPE_Q2_K,
        GGML_TYPE_Q3_K,
        GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_1,
        GGML_TYPE_Q4_K,
        GGML_TYPE_Q5_0,
        GGML_TYPE_Q5_1,
        GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K,
        GGML_TYPE_Q8_0,
#ifdef GGML_USE_METAL
        GGML_TYPE_F16
#else
        GGML_TYPE_BF16
#endif
    };

    constexpr double EPSILON = 1e-12;
    constexpr double INFINITE = std::numeric_limits<double>::infinity();
    constexpr uint64_t STATE_MAGIC = 0x4250572d5631; // "BPW-V1"
    constexpr uint64_t HASH_MAGIC = 0xeabada55cafed00d;
    constexpr float boost = 2.5f;
    const char * func = __func__;

    // Tensor size in bytes for a given type
    auto tensor_bytes = [](const ggml_tensor * gt, const ggml_type gq) -> size_t {
        return (size_t)ggml_nrows(gt) * ggml_row_size(gq, gt->ne[0]);
    };

    // Tensor bpw for a given type
    auto tensor_bpw = [&](const ggml_tensor * gt, const ggml_type gq) -> double {
        return (double)tensor_bytes(gt, gq) * 8.0 / (double)ggml_nelements(gt);
    };

    // Check if tensor is compatible with quantization type
    auto is_compatible = [](const ggml_tensor * gt, const ggml_type gq) -> bool {
        const int64_t blck = ggml_blck_size(gq);
        return blck <= 1 || gt->ne[0] % blck == 0;
    };

    // Get suitable fallback for type
    auto make_compatible = [&](const ggml_tensor * gt, const ggml_type gq) -> ggml_type {
        if (is_compatible(gt, gq)) { return gq; }
        const ggml_type fb = fallback_type_for(gq);
        return is_compatible(gt, fb) ? fb : GGML_TYPE_F16;
    };

    // Check if tensor is an IQ type
    auto is_iq = [](const enum ggml_type gt) {
        switch (gt) {
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ1_M:
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ2_S:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_IQ3_S:
            case GGML_TYPE_IQ4_NL:
            case GGML_TYPE_IQ4_XS:
                return true;
            default:
                return false;
        }
    };

    // Check if tensor can be quantized
    auto can_quantize = [&](const ggml_tensor * gt) -> bool {
        if (ggml_n_dims(gt) < 2 || ggml_n_dims(gt) > 3) { return false; } // skip 1D & 4D+ tensors
        return tensor_allows_quantization(qs.params, qs.model.arch, gt);
    };

    // DJB2 hashing algorithm
    auto djb2_hash = [&](const uint8_t * data, const size_t n) -> uint64_t {
        uint64_t h = 5381;
        for (size_t i = 0; i < n; ++i) { h = (h << 5) + h + data[i]; }
        return h ? h : HASH_MAGIC;
    };

    // Model ID from metadata hash
    const uint64_t model_id = [&] {
        const size_t sz = gguf_get_meta_size(ml.metadata);
        std::vector<uint8_t> buf(sz);
        gguf_get_meta_data(ml.metadata, buf.data());
        return djb2_hash(buf.data(), buf.size());
    }();

    std::string checkpoint_file;

    {
        char hex[17];
        std::string name;
        std::snprintf(hex, sizeof(hex), "%016" PRIx64, (uint64_t)model_id);
        ml.get_key(LLM_KV_GENERAL_NAME, name, false);
        name.erase(0, name.find_last_of('/') + 1);
        std::replace(name.begin(), name.end(), ' ', '_');
        name.empty() ? checkpoint_file = ml.arch_name : checkpoint_file = name;
        checkpoint_file += "-" + std::string(hex) + ".bpw_state";

        if (qs.params->state_file) {
            const char * filename = qs.params->state_file;
            bool is_valid = false;

            if (filename[0] == '\0') { is_valid = true; }
            else if (std::ifstream(filename, std::ios::binary).good()) { is_valid = true; }
            else {
                std::ofstream ofs(filename, std::ios::binary | std::ios::app);
                if (ofs.is_open()) {
                    is_valid = true;
                    ofs.close();
                    std::remove(filename);
                }
            }

            if (is_valid) {
                if (filename[0] != '\0') { checkpoint_file = filename; }
            } else {
                LLAMA_LOG_WARN("%s: '%s' is not a valid state file, ignoring\n", func, filename);
                checkpoint_file = filename;
            }
        }
    }

    // Save vector<type_choice> state to disk
    auto save_state = [&](const std::vector<type_choice> & all_tensors) {
        const std::string tmp = checkpoint_file + ".tmp";
        std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
        if (!ofs) { return; }
        ofs.write((const char *)& STATE_MAGIC, sizeof(STATE_MAGIC));
        ofs.write((const char *)& model_id, sizeof(model_id));
        const uint64_t n = all_tensors.size();
        ofs.write((const char *)& n, sizeof(n));
        for (const auto & tn : all_tensors) {
            const std::string name = ggml_get_name(tn.w->tensor);
            const auto len = (uint32_t)name.size();
            ofs.write((const char *)& len, sizeof(len));
            ofs.write(name.data(), len);

            const uint64_t sz = tn.candidates.size();
            ofs.write((const char *)& sz, sizeof(sz));
            ofs.write((const char *)& tn.choice, sizeof(tn.choice));
            ofs.write((const char *)& tn.min_bpw, sizeof(tn.min_bpw));
            ofs.write((const char *)& tn.max_bpw, sizeof(tn.max_bpw));
            const uint64_t ne = tn.n_elements;
            ofs.write((const char *)& ne, sizeof(ne));

            for (const auto & c : tn.candidates) {
                const int32_t tp = c.type;
                const uint64_t bt = c.bytes;
                ofs.write((const char *)& tp, sizeof(tp));
                ofs.write((const char *)& c.bpw, sizeof(c.bpw));
                ofs.write((const char *)& bt, sizeof(bt));
                ofs.write((const char *)& c.error, sizeof(c.error));
            }
        }

        const bool werr = ofs.fail() || ofs.bad();
        ofs.close();
        if (werr || ofs.fail() || ofs.bad()) {
            std::remove(tmp.c_str());
            LLAMA_LOG_WARN("\n\t%s: failed to save progress to %s (write error)\n", func, tmp.c_str());
            return;
        }

        std::remove(checkpoint_file.c_str());
        std::rename(tmp.c_str(), checkpoint_file.c_str());
        LLAMA_LOG_INFO("\t%s: saved target progress for %lu tensors to %s\n", func, all_tensors.size(), checkpoint_file.c_str());
    };

    // Load vector<type_choice> state from disk
    auto load_state = [&]() -> std::unordered_map<std::string, type_choice> {
        std::ifstream ifs(checkpoint_file, std::ios::binary);
        if (!ifs) { return {}; }

        uint64_t magic = 0;
        uint64_t id = 0;
        ifs.read((char *)& magic, sizeof(magic));
        ifs.read((char *)& id, sizeof(id));
        if (id != model_id) {
            LLAMA_LOG_WARN("%s: invalid target state file, ignoring\n", func);
            return {};
        }

        if (magic != STATE_MAGIC) {
            LLAMA_LOG_WARN("%s: invalid state file, ignoring\n", func);
            return {};
        }

        std::unordered_map<std::string, type_choice> out;
        uint64_t n = 0;
        ifs.read((char *)& n, sizeof(n));
        for (uint64_t i = 0; i < n; ++i) {
            if (!ifs.good()) { break; }
            uint32_t len = 0;
            ifs.read((char *)& len, sizeof(len));
            if (len >= GGML_MAX_NAME) {
                LLAMA_LOG_WARN("%s: state file name length exceeds limits, ignoring\n", func);
                return {};
            }
            std::string name(len, '\0');
            ifs.read(name.data(), len);

            type_choice si;
            uint64_t sz = 0;
            ifs.read((char *)& sz, sizeof(sz));
            if (sz > std::size(quant_types)) {
                LLAMA_LOG_WARN("%s: state file candidate size exceeds limits, ignoring\n", func);
                return {};
            }
            ifs.read((char *)& si.choice, sizeof(si.choice));
            ifs.read((char *)& si.min_bpw, sizeof(si.min_bpw));
            ifs.read((char *)& si.max_bpw, sizeof(si.max_bpw));
            uint64_t ne = 0;
            ifs.read((char *)& ne, sizeof(ne));
            si.n_elements = (size_t)ne;

            si.candidates.resize(sz);
            for (auto & cd : si.candidates) {
                if (!ifs.good()) { break; }
                int32_t t = 0;
                uint64_t b = 0;
                ifs.read((char *)& t, sizeof(t));
                cd.type = (ggml_type)t;
                ifs.read((char *)& cd.bpw, sizeof(cd.bpw));
                ifs.read((char *)& b, sizeof(b));
                cd.bytes = (size_t)b;
                ifs.read((char *)& cd.error, sizeof(cd.error));
            }

            out.emplace(std::move(name), std::move(si));
        }

        if (!ifs.good()) {
            LLAMA_LOG_WARN("%s: state file truncated or read error, ignoring\n", func);
            return {};
        }

        LLAMA_LOG_INFO("\t%s: using %s (data for %lu tensors loaded)\n", func, checkpoint_file.c_str(), out.size());
        return out;
    };

    // Check for user interrupt and save progress
    auto check_signal_handler = [&](const std::vector<type_choice> & all_tensors) {
        if (bpw_stop.load(std::memory_order_relaxed)) {
            if (qs.params->state_file) {
                LLAMA_LOG_INFO("\n\t%s: interrupted, saving progress for %lu tensors to %s\n", func, all_tensors.size(), checkpoint_file.c_str());
                save_state(all_tensors);
            } else {
                LLAMA_LOG_INFO("\n\t%s: interrupted\n", func);
            }

            throw std::runtime_error("user terminated the process");
        }
    };

    // Quality metrics
    struct quant_error {
        double error = std::numeric_limits<double>::infinity();
        double mse = 0.0;
        double wce = 0.0;
    };

    // Pre-calculated stats for MSE
    struct mse_cache {
        std::vector<double> row_sq_norm;
    };

    // Pre-calculated stats for WCE
    struct wce_cache {
        std::vector<double> row_sq_norm;
    };

    // Determine optimization strategy
    auto is_angle_sensitive = [](const std::string & name) -> bool {
        return name.find("attn_q.weight") != std::string::npos ||
               name.find("attn_k.weight") != std::string::npos ||
               name.find("attn_qkv.weight") != std::string::npos ||
               name.find("attn_q_a.weight") != std::string::npos ||
               name.find("attn_q_b.weight") != std::string::npos;
    };

    // Tensors sensitive to distortion in dot-product/inner-product calculations (TurboQuant)
    auto is_inner_product_sensitive = [](const std::string & name) -> bool {
        // Query/Key and variants (fused, latent, and enc/dec)
        if (name.find("attn_q") != std::string::npos || name.find("attn_k") != std::string::npos) { return true; }

        // Attention Output and variants (standard, enc/dec, cross-attn)
        if (name.find("attn_o") != std::string::npos) { return true; }

        if (name.find("ffn_down.weight") != std::string::npos) { return true; }

        return false;
    };

    // Estimate error for a given type using a sampled subset of rows
    auto compute_quant_error = [&](
        const ggml_tensor * t,
        const ggml_type quant_type,
        const std::vector<float> & f32_sample,
        const std::vector<int64_t> & rows_sample,
        const float * values_sample,
        const float * activations_sample,
        std::vector<uint8_t> & quantized_buffer,
        std::vector<float> & dequantized_buffer,
        const wce_cache * ref_wce = nullptr,
        const mse_cache * ref_mse = nullptr
    ) -> quant_error
    {
        const int64_t n_per_row = t->ne[0];
        const int64_t ne2 = t->ne[2] > 0 ? t->ne[2] : 1;
        const size_t sample_elems = f32_sample.size();
        const size_t sample_rows = n_per_row > 0 ? sample_elems / (size_t)n_per_row : 0;

        quant_error qe;
        if (sample_rows == 0) {
            qe.error = 0.0;
            return qe;
        }

        const size_t row_sz = ggml_row_size(quant_type, n_per_row);
        constexpr size_t SAFETY_PADDING = 256;
        if (quantized_buffer.size() < row_sz + SAFETY_PADDING) { quantized_buffer.resize(row_sz + SAFETY_PADDING); }
        if (dequantized_buffer.size() < (size_t)n_per_row) { dequantized_buffer.resize((size_t)n_per_row); }

        const ggml_type_traits * traits = ggml_get_type_traits(quant_type);
        if (!traits || !traits->to_float) { return qe; }

        auto quant_dequant_row = [&](const float * src, float * dst, const float * v) {
            ggml_quantize_chunk(quant_type, src, quantized_buffer.data(), 0, 1, n_per_row, v);
            if (quant_type == GGML_TYPE_F16) { ggml_fp16_to_fp32_row((const ggml_fp16_t *)quantized_buffer.data(), dst, (int)n_per_row); }
            else if (quant_type == GGML_TYPE_BF16) { ggml_bf16_to_fp32_row((const ggml_bf16_t *)quantized_buffer.data(), dst, (int)n_per_row); }
            else { traits->to_float(quantized_buffer.data(), dst, (int)n_per_row); }
        };

        const bool has_vals = values_sample != nullptr;
        const bool has_acts = activations_sample != nullptr;
        const bool use_wce = has_acts && has_vals && is_angle_sensitive(t->name);

        // Sampled stats for MSE
        std::vector<double> local_row_sq_norm;
        const std::vector<double> * ptr_row_sq_norm = nullptr;

        // Setup reference stats pointers for MSE
        if (!use_wce) {
            if (ref_mse) {
                ptr_row_sq_norm = & ref_mse->row_sq_norm;
            } else {
                local_row_sq_norm.reserve(sample_rows);
                size_t off = 0;
                for (int64_t s = 0; s < ne2; ++s) {
                    const int64_t rs = rows_sample[s];
                    const float * val = has_vals ? values_sample + s * n_per_row : nullptr;
                    const float * act = has_acts ? activations_sample + s * n_per_row : nullptr;
                    for (int64_t r = 0; r < rs; ++r) {
                        const float * x = f32_sample.data() + off;
                        double sum = 0.0;
                        double bias_sum = 0.0;
                        if (val && act) {
                            for (int64_t j = 0; j < n_per_row; ++j) {
                                const double act_j = act[j];
                                const double variance = std::max(0.0, (double)val[j] - act_j * act_j);
                                sum += variance * x[j] * x[j];
                                bias_sum += act_j * x[j];
                            }
                            sum += bias_sum * bias_sum;
                        } else if (val) {
                            for (int64_t j = 0; j < n_per_row; ++j) { sum += std::max(0.0f, val[j]) * x[j] * x[j]; }
                        } else {
                            for (int64_t j = 0; j < n_per_row; ++j) { sum += x[j] * x[j]; }
                        }

                        local_row_sq_norm.push_back(sum);
                        off += (size_t)n_per_row;
                    }
                }

                ptr_row_sq_norm = & local_row_sq_norm;
            }
        }

        // Helper for trimmed mean
        auto trimmed_mean = [](std::vector<double> & v) -> double {
            const auto n = v.size();
            if (n == 0) { return 0.0; }
            if (n < 100) { return std::accumulate(v.begin(), v.end(), 0.0) / (double)n; }
            const auto k = (size_t)((double)n * 0.01); // trim 1% from each end
            std::nth_element(v.begin(), v.begin() + k, v.end());
            std::nth_element(v.begin() + k, v.end() - k, v.end());
            return std::accumulate(v.begin() + k, v.end() - k, 0.0) / std::max(1.0, (double)(n - 2 * k));
        };

        // Weighted Cosine Error (WCE)
        if (use_wce) {
            double total_cos_error = 0.0;
            size_t off = 0;
            size_t sample_idx = 0;

            const std::vector<double> * cached_norm_x = ref_wce && !ref_wce->row_sq_norm.empty() ? & ref_wce->row_sq_norm : nullptr;

            for (int64_t s = 0; s < ne2; ++s) {
                const int64_t rs = rows_sample[s];
                if (rs == 0) { continue; }

                const float * v = values_sample + s * n_per_row;
                double slice_sum = 0.0;

                for (int64_t r = 0; r < rs; ++r, ++sample_idx) {
                    const float * wx = f32_sample.data() + off;
                    float * wy = dequantized_buffer.data();
                    quant_dequant_row(wx, wy, v);

                    double dot = 0.0;
                    double ny = 0.0;
                    double nx = 0.0;
                    const bool calc_nx = !cached_norm_x;

                    if (calc_nx) {
                        for (int64_t j = 0; j < n_per_row; ++j) {
                            const double w = std::max(0.0f, v[j]);
                            const double xj = wx[j];
                            const double yj = wy[j];
                            const double yw = yj * w;
                            dot += xj * yw;
                            ny += yj * yw;
                            nx += xj * xj * w;
                        }
                    } else {
                        nx = (* cached_norm_x)[sample_idx];
                        for (int64_t j = 0; j < n_per_row; ++j) {
                            const double w = std::max(0.0f, v[j]);
                            const double yj = wy[j];
                            const double yw = yj * w;
                            dot += (double) wx[j] * yw;
                            ny += yj * yw;
                        }
                    }

                    // Cosine Distance
                    double cos_sim;
                    const double norm_prod = nx * ny;

                    if (norm_prod <= EPSILON) { cos_sim = nx <= EPSILON && ny <= EPSILON ? 1.0 : 0.0; }
                    else { cos_sim = dot / std::sqrt(norm_prod); }

                    if (cos_sim > 1.0) { cos_sim = 1.0; }
                    else if (cos_sim < -1.0) { cos_sim = -1.0; }

                    slice_sum += 1.0 - cos_sim;
                    off += (size_t) n_per_row;
                }

                const double nrows = t->ne[1];
                total_cos_error += slice_sum / (double)rs * (double)nrows;
            }

            qe.wce = total_cos_error;
            qe.error = qe.wce;
            return qe;
        }

        // Expected Output Error MSE (EOE-MSE)
        size_t off = 0;
        size_t row_idx = 0;
        double total_wmse = 0.0;

        for (int64_t s = 0; s < ne2; ++s) {
            const int64_t rs = rows_sample[s];
            if (rs == 0) { continue; }

            const float * val = has_vals ? values_sample + s * n_per_row : nullptr;
            const float * act = has_acts ? activations_sample + s * n_per_row : nullptr;

            std::vector<double> slice_mse_norm;
            slice_mse_norm.reserve(rs);

            for (int64_t r = 0; r < rs; ++r, ++row_idx) {
                const float * x = f32_sample.data() + off;
                float * y = dequantized_buffer.data();
                quant_dequant_row(x, y, val);

                double w_err = 0.0;
                double bias_num = 0.0;

                if (val && act) {
                    for (int64_t j = 0; j < n_per_row; ++j) {
                        const double w = std::max(0.0f, val[j]);
                        const double a = act[j];
                        const double e = (double)y[j] - (double)x[j];
                        w_err += w * e * e;
                        bias_num += a * e;
                    }
                    w_err += bias_num * bias_num;
                } else if (val) {
                    for (int64_t j = 0; j < n_per_row; ++j) {
                        const double w = std::max(0.0f, val[j]);
                        const double e = (double)y[j] - (double)x[j];
                        w_err += w * e * e;
                    }
                } else {
                    for (int64_t j = 0; j < n_per_row; ++j) {
                        const double e = (double)y[j] - (double)x[j];
                        w_err += e * e;
                    }
                }

                const double rsn = (* ptr_row_sq_norm)[row_idx];
                const double m_norm = rsn > EPSILON ? w_err / rsn : 0.0;
                slice_mse_norm.push_back(std::isfinite(m_norm) ? m_norm : INFINITE);

                off += (size_t)n_per_row;
            }

            const int64_t nrows = t->ne[1];
            const double slice_mean_mse = trimmed_mean(slice_mse_norm) * (double)nrows;

            total_wmse += slice_mean_mse;
        }

        qe.mse = total_wmse;
        qe.error = total_wmse;
        return qe;
    };

    std::unordered_map<std::string, type_choice> bpw_data;
    if (qs.params->state_file && !checkpoint_file.empty()) { bpw_data = load_state(); }

    auto has_side_data = [&](const auto * m, const std::string & name, const int64_t n2, const int64_t n_per_row) {
        if (!m) { return false; }
        auto it = m->find(name);
        if (it == m->end()) { return false; }
        const size_t req = (size_t) n2 * (size_t) n_per_row;
        const size_t sz = it->second.size();
        return sz == req || sz == (size_t) n_per_row;
    };

    // Precompute the Centered‑Normalized Importance Factor (CNIF)
    std::unordered_map<std::string, float> cnif_scores;
    if (statistics_data && !statistics_data->empty()) {
        struct tensor_score {
            std::string name;
            float score = 0.0f;
        };

        auto corr_error = [](float x) {
            if (!std::isfinite(x)) { return 0.0f; }
            return 1.0f - std::abs(std::clamp(x, -1.0f, 1.0f));
        };

        auto sort_scores = [](std::vector<tensor_score> & scores) {
            std::sort(scores.begin(), scores.end(), [](const tensor_score & a, const tensor_score & b) {
                return a.score < b.score || (a.score == b.score && a.name < b.name);
            });
        };

        auto median = [](const auto & values, auto get_value) {
            const size_t mid = values.size() / 2;
            if (values.size() % 2 != 0) { return get_value(values[mid]); }
            return 0.5f * (get_value(values[mid - 1]) + get_value(values[mid]));
        };

        auto rank = [&](std::vector<tensor_score> & scores, const float beta) {
            if (scores.empty()) { return; }

            sort_scores(scores);

            if (scores.size() == 1) {
                cnif_scores[scores[0].name] = 1.0f;
                return;
            }

            for (size_t i = 0; i < scores.size();) {
                size_t j = i + 1;
                while (j < scores.size() && std::abs(scores[j].score - scores[i].score) <= 1e-6f) { ++j; }

                const float p = 0.5f * (float)(i + j - 1) / (float)(scores.size() - 1);
                const float scale = std::exp(beta * (2.0f * p - 1.0f));
                for (size_t k = i; k < j; ++k) { cnif_scores[scores[k].name] = scale; }
                i = j;
            }
        };

        auto cnif = [&](std::vector<tensor_score> & scores, const float beta) {
            constexpr size_t min_scores = 8;
            constexpr float eps = 1e-6f;
            constexpr float mad_factor = 1.4826f; // Median Absolute Deviation estimator for normal distributions

            if (scores.size() < min_scores) {
                rank(scores, beta);
                return;
            }

            sort_scores(scores);

            const float med = median(scores, [](const tensor_score & s) { return s.score; });
            std::vector<float> deviations;
            deviations.reserve(scores.size());
            for (const auto & score : scores) { deviations.push_back(std::abs(score.score - med)); }

            std::sort(deviations.begin(), deviations.end());
            const float mad = median(deviations, [](float x) { return x; });
            if (mad <= eps) {
                rank(scores, beta);
                return;
            }

            const float inv_scale = 1.0f / (mad_factor * mad + eps);
            for (const auto & score : scores) {
                const float z = std::clamp((score.score - med) * inv_scale, -4.0f, 4.0f);
                cnif_scores[score.name] = std::exp(beta * std::tanh(0.8f * z));
            }
        };

        std::vector<tensor_score> mse_scores;
        std::vector<tensor_score> wce_scores;
        mse_scores.reserve(tensors.size());
        wce_scores.reserve(tensors.size());
        cnif_scores.reserve(tensors.size());

        for (const auto * tw : tensors) {
            const ggml_tensor * tensor = tw->tensor;
            if (!can_quantize(tensor)) { continue; }

            const std::string name = remap_imatrix(ggml_get_name(tensor), mapped);
            auto it = statistics_data->find(name);
            if (it == statistics_data->end() || it->second.size() <= (size_t) L2_DIST) { continue; }

            const auto & ts = it->second;
            const float h_norm = std::isfinite(ts[H_NORM]) ? std::clamp(ts[H_NORM], 0.0f, 100.0f) : 100.0f;
            const float concentration = 1.0f - h_norm / 100.0f;
            const float energy = std::isfinite(ts[ENERGY]) ? std::max(0.0f, ts[ENERGY]) : 0.0f;
            const float l2_dist = std::isfinite(ts[L2_DIST]) ? std::max(0.0f, ts[L2_DIST]) : 0.0f;
            const float rel_l2 = l2_dist / (std::sqrt(energy) + EPSILON);
            const float density = rel_l2 / (1.0f + rel_l2);
            const float fragility = 0.5f * (corr_error(ts[COSSIM]) + corr_error(ts[PCC]));

            const int64_t n_per_row = tensor->ne[0];
            const int64_t ne2 = tensor->ne[2] > 0 ? tensor->ne[2] : 1;
            const bool has_vals = has_side_data(values_data, name, ne2, n_per_row);
            const bool has_acts = has_side_data(activations_data, name, ne2, n_per_row);
            const bool use_wce = has_vals && has_acts && is_angle_sensitive(name);

            const float score = use_wce
                ? 0.75f * concentration + 0.15f * density + 0.10f * fragility
                : 0.40f * concentration + 0.35f * density + 0.25f * fragility;

            auto & scores = use_wce ? wce_scores : mse_scores;
            scores.push_back({ name, score });
        }

        cnif(mse_scores, 0.40f);
        cnif(wce_scores, 0.25f);
    }

    // Outlier mitigation (simulate TurboQuant rotation)
    auto outlier_smoothing = [&](const std::vector<float>& src, std::vector<float>& dst, const int64_t n_rows, const int64_t len) {
        dst.resize(src.size());
        for (int64_t r = 0; r < n_rows; ++r) {
            const float* src_row = src.data() + r * len;
            float* dst_row = dst.data() + r * len;

            // Fast variance estimation
            double mean_d = 0.0;
            double M2 = 0.0;
            for (int64_t i = 0; i < len; ++i) {
                double delta = src_row[i] - mean_d;
                mean_d += delta / (i + 1);
                M2 += delta * (src_row[i] - mean_d);
            }

            const float mean = mean_d;
            float variance = M2 / len;
            const float std_dev = std::sqrt(std::max(0.0f, variance));

            // Clip at 4 standard deviations to simulate the effect of rotation spreading the outlier energy
            float clip_val = 4.0f * std_dev;

            if (clip_val > 1e-4f) {
                for (int64_t i = 0; i < len; ++i) { dst_row[i] = std::clamp(src_row[i], mean - clip_val, mean + clip_val); }
            } else {
                std::memcpy(dst_row, src_row, len * sizeof(float));
            }
        }
    };

    // HMT Randomized SVD condition number estimator: kappa = sigma 1 / sigma 20. See https://arxiv.org/abs/0909.4061
    auto randomized_svd_condition_number = [&](const std::vector<float> & f32_sample, const int64_t n_per_row, const int64_t total_rows_sampled) -> float {
        constexpr int64_t target_rank = 25;
        constexpr int64_t k = 20;   // 20th singular value for condition number estimation
        constexpr float eps_norm = 1e-8f;
        constexpr float eps_eig = EPSILON;
        constexpr int jacobi_sweeps = 20;
        if (total_rows_sampled < 1 || n_per_row < 1) { return INFINITE; }

        const int64_t m = total_rows_sampled;
        const int64_t n = n_per_row;

        // Jacobi eigenvalue decomposition for symmetric matrix
        auto jacobi_diag = [&](std::vector<float> & A, const int64_t dim, const int max_sweeps) {
            for (int sweep = 0; sweep < max_sweeps; ++sweep) {
                bool converged = true;
                for (int64_t p = 0; p < dim - 1; ++p) {
                    for (int64_t q = p + 1; q < dim; ++q) {
                        const float app = A[p * dim + p];
                        const float aqq = A[q * dim + q];
                        const float apq = A[p * dim + q];
                        if (std::abs(apq) <= eps_eig) { continue; }
                        converged = false;

                        const float tau = (aqq - app) / (2.0f * apq);
                        const float t   = tau >= 0.0f
                            ? 1.0f / (tau + std::sqrt(1.0f + tau * tau))
                            : -1.0f / (-tau + std::sqrt(1.0f + tau * tau));
                        const float c   = 1.0f / std::sqrt(1.0f + t * t);
                        const float s   = c * t;

                        A[p * dim + p] = app - t * apq;
                        A[q * dim + q] = aqq + t * apq;
                        A[p * dim + q] = 0.0f;
                        A[q * dim + p] = 0.0f;

                        for (int64_t r = 0; r < dim; ++r) {
                            if (r == p || r == q) { continue; }
                            const float arp = A[r * dim + p];
                            const float arq = A[r * dim + q];
                            A[r * dim + p] = c * arp - s * arq;
                            A[p * dim + r] = A[r * dim + p];
                            A[r * dim + q] = s * arp + c * arq;
                            A[q * dim + r] = A[r * dim + q];
                        }
                    }
                }

                if (converged) { break; }
            }
        };

        auto matmul_a = [&](
            const std::vector<float> & A, const int64_t rows_a,
            const int64_t cols_a, const std::vector<float> & B,
            const int64_t cols_b, std::vector<float> & C
        )
        {
            C.assign((size_t)rows_a * (size_t)cols_b, 0.0f);
            for (int64_t i = 0; i < rows_a; ++i) {
                const size_t i_a = (size_t)i * (size_t)cols_a;
                const size_t i_c = (size_t)i * (size_t)cols_b;
                for (int64_t l = 0; l < cols_a; ++l) {
                    const float aik = A[i_a + (size_t)l];
                    if (aik == 0.0f) { continue; }
                    const size_t k_b = (size_t)l * (size_t)cols_b;
                    for (int64_t j = 0; j < cols_b; ++j) {
                        C[i_c + (size_t)j] += aik * B[k_b + (size_t)j];
                    }
                }
            }
        };

        // C = A^T * B  (A is rows_a x cols_a, B is rows_a x cols_b, C is cols_a x cols_b)
        auto matmul_b = [&](
            const std::vector<float> & A,
            const int64_t rows_a,
            const int64_t cols_a,
            const std::vector<float> & B, const int64_t cols_b,
            std::vector<float> & C
        )
        {
            C.assign((size_t)cols_a * (size_t)cols_b, 0.0f);
            for (int64_t i = 0; i < cols_a; ++i) {
                const size_t i_c = (size_t)i * (size_t)cols_b;
                for (int64_t l = 0; l < rows_a; ++l) {
                    const float aki = A[(size_t)l * (size_t)cols_a + (size_t)i];
                    if (aki == 0.0f) { continue; }
                    const size_t k_b = (size_t)l * (size_t)cols_b;
                    for (int64_t j = 0; j < cols_b; ++j) {
                        C[i_c + (size_t)j] += aki * B[k_b + (size_t)j];
                    }
                }
            }
        };

        // Modified Gram-Schmidt QR Decomposition. Q gets orthonormal columns from Y (Y is rows x cols)
        auto mgs_qr = [&](
            const std::vector<float> & Y,
            const int64_t rows, const int64_t cols,
            std::vector<float> & Q
        )
        {
            Q.assign((size_t)rows * (size_t)cols, 0.0f);
            std::vector<float> v(rows);
            for (int64_t j = 0; j < cols; ++j) {
                for (int64_t i = 0; i < rows; ++i) {
                    v[i] = Y[(size_t)i * (size_t)cols + (size_t)j];
                }

                for (int64_t b = 0; b < j; ++b) {
                    double dot = 0.0;
                    for (int64_t i = 0; i < rows; ++i) {
                        dot += (double)v[i] * (double)Q[(size_t)i * (size_t)cols + (size_t)b];
                    }
                    for (int64_t i = 0; i < rows; ++i) {
                        v[i] -= (float)dot * Q[(size_t)i * (size_t)cols + (size_t)b];
                    }
                }

                double norm_sq = 0.0;
                for (int64_t i = 0; i < rows; ++i) {
                    norm_sq += (double)v[i] * (double)v[i];
                }

                const float norm = std::sqrt((float)norm_sq);
                if (norm > eps_norm) {
                    const float inv = 1.0f / norm;
                    for (int64_t i = 0; i < rows; ++i) {
                        Q[(size_t)i * (size_t)cols + (size_t)j] = v[i] * inv;
                    }
                }
            }
        };

        // Filter and row-normalize valid rows (exclude NaN/Inf and near zero norm rows)
        std::vector<float> norm_matrix;
        norm_matrix.reserve((size_t)m * (size_t)n);
        for (int64_t r = 0; r < m; ++r) {
            const float * row = f32_sample.data() + r * n;
            double l2_sq = 0.0;
            for (int64_t j = 0; j < n; ++j) {
                const float val = row[j];
                if (!std::isfinite(val)) { l2_sq = -1.0; break; }
                l2_sq += (double)val * (double)val;
            }

            if (l2_sq < 0.0) { continue; }
            const float l2 = std::sqrt((float)l2_sq);
            if (l2 <= eps_norm) { continue; }
            const float inv_l2 = 1.0f / l2;
            const size_t cur = norm_matrix.size();
            norm_matrix.resize(cur + (size_t)n);
            float * dst = norm_matrix.data() + cur;
            for (int64_t j = 0; j < n; ++j) { dst[j] = row[j] * inv_l2; }
        }

        const int64_t valid_m = (int64_t)norm_matrix.size() / n;
        if (valid_m < 2) { return INFINITE; }
        const bool hmt_direct = valid_m >= target_rank;
        const bool hmt_transpose = !hmt_direct && n >= target_rank;

        // Deterministic random Gaussian projection using Golden Ratio * Random Scrambling Hashing
        std::mt19937 rng((uint64_t)(n_per_row * 0x9E3779B97F4A7C15ULL + valid_m * 0xBF58476D1CE4E5B9ULL));
        std::normal_distribution<float> normal(0.0f, 1.0f);

        // Estimate the condition number (kappa)
        auto condition_number = [&](const std::vector<float> & eigenvalues) -> float {
            if ((int64_t)eigenvalues.size() < k) { return INFINITE; }
            const float sigma_max = std::sqrt(std::max(0.0f, eigenvalues[0]));
            const float sigma_min = std::sqrt(std::max(0.0f, eigenvalues[k - 1]));
            if (sigma_min <= eps_eig) { return INFINITE; }

            return sigma_max / sigma_min;
        };

        if (hmt_direct) {
            // HMT on A (valid_m x n) with q=1 power iteration
            std::vector<float> Omega((size_t)n * (size_t)target_rank);
            for (auto & v : Omega) { v = normal(rng); }

            std::vector<float> Y0;
            std::vector<float> Z;
            std::vector<float> Y1;
            std::vector<float> Q;
            std::vector<float> B;
            std::vector<float> C;
            matmul_a(norm_matrix, valid_m, n, Omega, target_rank, Y0);  // Y0 = A * Omega
            matmul_b(norm_matrix, valid_m, n, Y0, target_rank, Z);      // Z = A^T * Y0
            matmul_a(norm_matrix, valid_m, n, Z, target_rank, Y1);      // Y1 = A * Z
            mgs_qr(Y1, valid_m, target_rank, Q);
            matmul_b(Q, valid_m, target_rank, norm_matrix, n, B);       // B = Q^T * A

            // C = B * B^T (target_rank x target_rank)
            C.assign((size_t)target_rank * (size_t)target_rank, 0.0f);
            for (int64_t i = 0; i < target_rank; ++i) {
                for (int64_t j = 0; j < target_rank; ++j) {
                    double sum = 0.0;
                    for (int64_t l = 0; l < n; ++l) {
                        sum += (double)B[(size_t)i * (size_t)n + (size_t)l] * (double)B[(size_t)j * (size_t)n + (size_t)l];
                    }

                    C[(size_t)i * (size_t)target_rank + (size_t)j] = (float)sum;
                }
            }

            jacobi_diag(C, target_rank, jacobi_sweeps);
            std::vector<float> eigenvalues;
            eigenvalues.reserve(target_rank);
            for (int64_t i = 0; i < target_rank; ++i) { eigenvalues.push_back(C[i * target_rank + i]); }
            std::sort(eigenvalues.begin(), eigenvalues.end(), std::greater<float>());

            return condition_number(eigenvalues);
        }

        if (hmt_transpose) {
            // HMT on A^T (n x valid_m) with q=1 power iteration
            std::vector<float> Omega((size_t)valid_m * (size_t)target_rank);
            for (auto & v : Omega) { v = normal(rng); }

            std::vector<float> Y0;
            std::vector<float> Z;
            std::vector<float> Y1;
            std::vector<float> Q;
            std::vector<float> B;
            std::vector<float> C;
            matmul_b(norm_matrix, valid_m, n, Omega, target_rank, Y0);   // Y0 = A^T * Omega
            matmul_a(norm_matrix, valid_m, n, Y0, target_rank, Z);       // Z = A * Y0
            matmul_b(norm_matrix, valid_m, n, Z, target_rank, Y1);       // Y1 = A^T * Z
            mgs_qr(Y1, n, target_rank, Q);

            // B = Q^T * A^T  (target_rank x valid_m)
            B.assign((size_t)target_rank * (size_t)valid_m, 0.0f);
            for (int64_t j = 0; j < target_rank; ++j) {
                for (int64_t k = 0; k < valid_m; ++k) {
                    double sum = 0.0;
                    for (int64_t i = 0; i < n; ++i) {
                        sum += (double)Q[(size_t)i * (size_t)target_rank + (size_t)j] * (double)norm_matrix[(size_t)k * (size_t)n + (size_t)i];
                    }

                    B[(size_t)j * (size_t)valid_m + (size_t)k] = (float)sum;
                }
            }

            // C = B * B^T (target_rank x target_rank)
            C.assign((size_t)target_rank * (size_t)target_rank, 0.0f);
            for (int64_t i = 0; i < target_rank; ++i) {
                for (int64_t j = 0; j < target_rank; ++j) {
                    double sum = 0.0;
                    for (int64_t l = 0; l < valid_m; ++l) {
                        sum += (double)B[(size_t)i * (size_t)valid_m + (size_t)l] * (double)B[(size_t)j * (size_t)valid_m + (size_t)l];
                    }

                    C[(size_t)i * (size_t)target_rank + (size_t)j] = (float)sum;
                }
            }

            jacobi_diag(C, target_rank, jacobi_sweeps);
            std::vector<float> eigenvalues;
            eigenvalues.reserve(target_rank);
            for (int64_t i = 0; i < target_rank; ++i) { eigenvalues.push_back(C[i * target_rank + i]); }
            std::sort(eigenvalues.begin(), eigenvalues.end(), std::greater<float>());

            return condition_number(eigenvalues);
        }

        // Fallback: full thin SVD via Gram matrix of smaller dimension + Jacobi
        const bool gram_rows = valid_m <= n;
        if (gram_rows) {
            const int64_t gram_dim = valid_m;
            std::vector<float> G((size_t)gram_dim * (size_t)gram_dim, 0.0f);
            for (int64_t i = 0; i < gram_dim; ++i) {
                for (int64_t j = i; j < gram_dim; ++j) {
                    double sum = 0.0;
                    for (int64_t l = 0; l < n; ++l) {
                        sum += (double)norm_matrix[(size_t)i * (size_t)n + (size_t)l] * (double)norm_matrix[(size_t)j * (size_t)n + (size_t)l];
                    }

                    G[(size_t)i * (size_t)gram_dim + (size_t)j] = (float)sum;
                    G[(size_t)j * (size_t)gram_dim + (size_t)i] = (float)sum;
                }
            }

            jacobi_diag(G, gram_dim, jacobi_sweeps);
            std::vector<float> eigenvalues;
            eigenvalues.reserve(gram_dim);
            for (int64_t i = 0; i < gram_dim; ++i) { eigenvalues.push_back(G[i * gram_dim + i]); }
            std::sort(eigenvalues.begin(), eigenvalues.end(), std::greater<float>());

            return condition_number(eigenvalues);
        }

        // G = A^T * A (n x n)
        const int64_t gram_dim = n;
        std::vector<float> G((size_t)gram_dim * (size_t)gram_dim, 0.0f);
        for (int64_t i = 0; i < gram_dim; ++i) {
            for (int64_t j = i; j < gram_dim; ++j) {
                double sum = 0.0;
                for (int64_t l = 0; l < valid_m; ++l) {
                    sum += (double)norm_matrix[(size_t)l * (size_t)n + (size_t)i] * (double)norm_matrix[(size_t)l * (size_t)n + (size_t)j];
                }

                G[(size_t)i * (size_t)gram_dim + (size_t)j] = (float)sum;
                G[(size_t)j * (size_t)gram_dim + (size_t)i] = (float)sum;
            }
        }

        jacobi_diag(G, gram_dim, jacobi_sweeps);
        std::vector<float> eigenvalues;
        eigenvalues.reserve(gram_dim);
        for (int64_t i = 0; i < gram_dim; ++i) { eigenvalues.push_back(G[i * gram_dim + i]); }
        std::sort(eigenvalues.begin(), eigenvalues.end(), std::greater<float>());

        return condition_number(eigenvalues);
    };

    // Map [1, ∞) to [1, M]; alpha controls mapping aggressiveness towards M
    auto squash_kappa = [](float kappa) -> float {
        constexpr float M = 5.0f;
        constexpr float alpha = 0.15f;
        if (!std::isfinite(kappa)) { return M; }

        return 1.0f + (M - 1.0f) * std::tanh(alpha * std::log(kappa));
    };

    // Parallelize tensor processing (courtesy of https://github.com/ddh0)
    auto process_tensor = [&](
        const llama_model_loader::llama_tensor_weight * tw,
        std::vector<no_init<uint8_t>> & thread_local_buffer,
        std::vector<float> & f32_sample,
        std::vector<float> & smoothed_sample,
        std::vector<float> & val_cache,
        std::vector<float> & act_cache,
        std::mutex & loader_mutex,
        std::mutex & log_mutex
    ) -> std::optional<type_choice>
    {
        ggml_tensor * tensor = tw->tensor;
        struct tensor_guard {
            ggml_tensor * t;
            void * orig;
            ~tensor_guard() { t->data = orig; }
        } guard{tensor, tensor->data};

        const std::string name = ggml_get_name(tensor);
        if (bpw_stop.load(std::memory_order_relaxed)) { return std::nullopt; }

        const std::string remapped_name = remap_imatrix(name, mapped);

        // Check cache
        if (auto tn = bpw_data.find(name); tn != bpw_data.end()) {
            type_choice tc;
            tc.w = tw;
            tc.candidates = tn->second.candidates;
            tc.choice = tn->second.choice;
            tc.min_bpw = tn->second.min_bpw;
            tc.max_bpw = tn->second.max_bpw;
            tc.n_elements = tn->second.n_elements ? tn->second.n_elements : (size_t)ggml_nelements(tensor);
            return tc;
        }
        {
            std::lock_guard<std::mutex> lock(log_mutex);
            LLAMA_LOG_INFO("\t%s: - processing tensor %45s \t(%12" PRId64 " elements)\n", func, name.c_str(), ggml_nelements(tensor));
        }

        if (!ml.use_mmap) {
            if (thread_local_buffer.size() < ggml_nbytes(tensor)) { thread_local_buffer.resize(ggml_nbytes(tensor)); }
            tensor->data = thread_local_buffer.data();
        }
        {
            std::lock_guard<std::mutex> lock(loader_mutex);
            // Temporarily revert to the original name to locate the weight
            if (qs.params->prune_layers) { ggml_set_name(tensor, remap_imatrix(tensor->name, mapped).c_str()); }
            ml.load_data_for(tensor);
            // Restore the remapped name for the final quantized output
            if (qs.params->prune_layers) { ggml_set_name(tensor, name.c_str()); }
        }

        // Sampling
        const int64_t n_per_row = tensor->ne[0];
        const int64_t nrows_total = tensor->ne[1];
        const int64_t ne2 = tensor->ne[2] > 0 ? tensor->ne[2] : 1;

        // Compute rows based on tensor shape and slice count
        auto sample_count = [&](const int64_t n, const int64_t rows, const int64_t n2, const bool has_acts) {
            constexpr double k_scale = 1.0;
            const double tensor_budget = (has_acts ? 1.0 : 0.5) * k_scale * 1024.0 * 1024.0;
            const double scale = std::clamp(std::sqrt(std::max(1.0, (double)rows) / 4096.0), 0.5, 2.0); // more rows for large tensors
            const double slice_budget = tensor_budget * scale / std::max<int64_t>(1, n2);
            const int64_t min_r = (has_acts ? 512 : 256) * (int64_t)k_scale;
            const int64_t max_r = 4096 * (int64_t)k_scale;
            int64_t tr = std::llround(slice_budget / std::max<int64_t>(1, n));
            tr = std::max<int64_t>(min_r, std::min<int64_t>(tr, std::min<int64_t>(rows, max_r)));
            if (rows <= min_r * 2) { tr = rows; }
            return tr;
        };

        const int64_t rows_to_sample = sample_count(n_per_row, nrows_total, ne2, activations_data != nullptr);
        f32_sample.clear();
        f32_sample.reserve((size_t)ne2 * (size_t)std::min(nrows_total, rows_to_sample) * (size_t)n_per_row);
        std::vector<int64_t> rows_sample(ne2, 0);

        // Populate f32_sample
        {
            const ggml_type src_type = tensor->type;
            const size_t src_row_sz = ggml_row_size(src_type, n_per_row);
            const ggml_type_traits * traits = ggml_get_type_traits(src_type);

            for (int64_t slice = 0; slice < ne2; ++slice) {
                std::mt19937 rng(djb2_hash((const uint8_t*)name.data(), name.size()) ^ HASH_MAGIC ^ slice);
                const int64_t limit = std::max<int64_t>(1, std::min<int64_t>(nrows_total, rows_to_sample));
                const double stride = (double)nrows_total / limit;
                int64_t offset = stride > 1.0 ? std::uniform_int_distribution<int64_t>(0, (int64_t)stride - 1)(rng) : 0;

                int64_t count = 0;
                for (int64_t i = 0; i < limit; ++i) {
                    int64_t r = offset + std::llround(i * stride);
                    if (r >= nrows_total) { break; }

                    const uint8_t * src = (const uint8_t *)tensor->data + slice * (src_row_sz * nrows_total) + r * src_row_sz;
                    size_t cur_sz = f32_sample.size();
                    f32_sample.resize(cur_sz + n_per_row);
                    float * dst = f32_sample.data() + cur_sz;

                    if (src_type == GGML_TYPE_F32) { std::memcpy(dst, src, n_per_row * sizeof(float)); }
                    else if (src_type == GGML_TYPE_F16) { ggml_fp16_to_fp32_row((const ggml_fp16_t*)src, dst, (int)n_per_row); }
                    else if (src_type == GGML_TYPE_BF16) { ggml_bf16_to_fp32_row((const ggml_bf16_t*)src, dst, (int)n_per_row); }
                    else if (traits && traits->to_float) { traits->to_float(src, dst, (int)n_per_row); }
                    else { throw std::runtime_error(format("unsupported source type %s for sampling", ggml_type_name(src_type))); }

                    ++count;
                }

                rows_sample[slice] = count;
            }
        }

        smoothed_sample.clear();
        bool is_outlier_heavy = remapped_name.find("ffn_down") != std::string::npos;
        if (is_outlier_heavy) { outlier_smoothing(f32_sample, smoothed_sample, rows_to_sample * ne2, n_per_row); }

        // Prepare side data
        auto get_side_data = [&](const auto * m) {
            if (!m) { return std::pair<const float *, size_t>{nullptr, 0}; }
            auto it = m->find(remapped_name);
            return it != m->end() ? std::pair{it->second.data(), it->second.size()} : std::pair<const float*, size_t>{nullptr, 0};
        };

        auto [val_ptr, val_sz] = get_side_data(values_data);
        auto [act_ptr, act_sz] = get_side_data(activations_data);

        // Cache WCE stats per tensor
        const float * val_vec_ptr = nullptr;
        const float * act_vec_ptr = nullptr;

        auto prepare_broadcast = [&](const float* src, size_t sz, std::vector<float>& storage, const float*& out_ptr) {
            if (!src) {
                out_ptr = nullptr;
                return;
            }
            size_t req = (size_t)ne2 * n_per_row;
            if (sz == req) { out_ptr = src; }
            else if (sz == (size_t)n_per_row) {
                storage.resize(req);
                for (int s = 0; s < ne2; ++s) { std::memcpy(storage.data() + s * n_per_row, src, n_per_row * sizeof(float)); }
                out_ptr = storage.data();
            } else {
                std::lock_guard<std::mutex> lock(log_mutex);
                out_ptr = nullptr;
                LLAMA_LOG_WARN("%s: side data mismatch for %s\n", func, name.c_str());
            }
        };

        prepare_broadcast(val_ptr, val_sz, val_cache, val_vec_ptr);
        prepare_broadcast(act_ptr, act_sz, act_cache, act_vec_ptr);

        const bool use_wce = val_vec_ptr && act_vec_ptr && is_angle_sensitive(remapped_name);

        // Precompute WCE reference stats
        wce_cache ref_wce;
        mse_cache ref_mse;
        size_t total_rows_sampled = 0;
        for (int64_t r : rows_sample) { total_rows_sampled += r; }

        if (use_wce) {
            ref_wce.row_sq_norm.reserve(total_rows_sampled);
            size_t off = 0;
            for (int64_t s = 0; s < ne2; ++s) {
                const int64_t rs = rows_sample[s];
                if (rs == 0) { continue; }
                const float * v = val_vec_ptr + s * n_per_row;
                for (int64_t r = 0; r < rs; ++r) {
                    const float * wx = f32_sample.data() + off;
                    double norm_x = 0.0;
                    for (int64_t j = 0; j < n_per_row; ++j) {
                        const double w = v ? std::max(0.0f, v[j]) : 1.0;
                        norm_x += (double)wx[j] * wx[j] * w;
                    }
                    ref_wce.row_sq_norm.push_back(norm_x);
                    off += n_per_row;
                }
            }
        } else {
            // Precompute MSE reference stats
            ref_mse.row_sq_norm.reserve(total_rows_sampled);
            const bool has_acts = act_vec_ptr != nullptr;
            const bool has_vals = val_vec_ptr != nullptr;

            size_t off = 0;
            for (int64_t s = 0; s < ne2; ++s) {
                const int64_t rs = rows_sample[s];
                const float * val = has_vals ? val_vec_ptr + s * n_per_row : nullptr;
                const float * act = has_acts ? act_vec_ptr + s * n_per_row : nullptr;
                for (int64_t r = 0; r < rs; ++r) {
                    const float * x = f32_sample.data() + off;
                    double sum = 0.0;
                    double bias_sum = 0.0;
                    if (val && act) {
                        for (int64_t j = 0; j < n_per_row; ++j) {
                            const double act_j = act[j];
                            const double variance = std::max(0.0, (double)val[j] - act_j * act_j);
                            sum += variance * x[j] * x[j];
                            bias_sum += act_j * x[j];
                        }
                        sum += bias_sum * bias_sum;
                    } else if (val) {
                        for (int64_t j = 0; j < n_per_row; ++j) {
                            sum += std::max(0.0f, val[j]) * x[j] * x[j];
                        }
                    } else {
                        for (int64_t j = 0; j < n_per_row; ++j) { sum += x[j] * x[j]; }
                    }

                    ref_mse.row_sq_norm.push_back(sum);
                    off += (size_t)n_per_row;
                }
            }
        }

        float kappa_factor = 1.0f;
        if (statistics_data && !statistics_data->empty()) {
            const float kappa = randomized_svd_condition_number(f32_sample, n_per_row, (int64_t)total_rows_sampled);
            kappa_factor = squash_kappa(kappa);
        }

        // Build candidates
        std::vector<ggml_type> valid_types;
        valid_types.reserve(std::size(quant_types));
        size_t max_row_sz = 0;
        const bool valid_matrix = val_vec_ptr != nullptr;

        for (auto t : quant_types) {
            if (is_iq(t) && !valid_matrix) { continue; }
            ggml_type compat = make_compatible(tensor, t);
            if (!is_compatible(tensor, compat)) { continue; }
            valid_types.push_back(compat);
            max_row_sz = std::max(max_row_sz, ggml_row_size(compat, n_per_row));
        }

        std::sort(valid_types.begin(), valid_types.end());
        valid_types.erase(std::unique(valid_types.begin(), valid_types.end()), valid_types.end());

        // Evaluate candidates
        std::vector<type_scores> evaluations;
        evaluations.reserve(valid_types.size());
        std::vector<uint8_t> q_buf;
        std::vector<float> dq_buf;
        if (total_rows_sampled > 0 && max_row_sz > 0) {
            q_buf.reserve(max_row_sz + 256); // safety padding
            dq_buf.reserve(n_per_row);
        }

        float scaling_factor = 1.0f;
        if (auto it = cnif_scores.find(remapped_name); it != cnif_scores.end()) { scaling_factor = it->second; }
        scaling_factor *= kappa_factor;

        for (ggml_type vt : valid_types) {
            if (bpw_stop.load(std::memory_order_relaxed)) { return std::nullopt; }
            const wce_cache * ptr_ref_wce = use_wce && !ref_wce.row_sq_norm.empty() ? & ref_wce : nullptr;
            const mse_cache * ptr_ref_mse = !use_wce && !ref_mse.row_sq_norm.empty() ? & ref_mse : nullptr;
            const float bpw = (float)ggml_type_size(vt) * 8.0f / (float)ggml_blck_size(vt);
            const std::vector<float>& eval_sample = is_outlier_heavy && bpw < 3.0f ? smoothed_sample : f32_sample;

            quant_error qe = compute_quant_error(
                tensor,
                vt,
                eval_sample,
                rows_sample,
                val_vec_ptr,
                act_vec_ptr,
                q_buf,
                dq_buf,
                ptr_ref_wce,
                ptr_ref_mse
            );

            double error = qe.error;
            // Error adjustment for inner-product sensitive tensors at low bpw
            if (is_inner_product_sensitive(remapped_name)) { error *= 1.0f + std::pow(2.0f, 3.0f - bpw); }

            type_scores candidate;
            candidate.type = vt;
            candidate.bpw = (float)tensor_bpw(tensor, vt);
            candidate.bytes = tensor_bytes(tensor, vt);
            candidate.error = error * scaling_factor;
            candidate.mse = qe.mse;
            candidate.wce = qe.wce;
            evaluations.push_back(candidate);
        }

        type_choice ch;
        ch.w = tw;
        ch.n_elements = ggml_nelements(tensor);

        for (const auto & ev : evaluations) {
            if (ev.bytes == 0) { continue; }
            ch.candidates.push_back(ev);
        }

        if (ch.candidates.empty()) {
            type_scores fb;
            fb.type = tensor->type;
            fb.bytes = ggml_nbytes(tensor);
            fb.bpw = fb.bytes * 8.0f / ch.n_elements;
            ch.candidates.push_back(fb);
        }

        auto simplify_pareto = [&](std::vector<type_scores> & candidates) {
            std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                return a.bytes < b.bytes || (a.bytes == b.bytes && a.error < b.error);
            });
            candidates.erase(std::unique(candidates.begin(), candidates.end(),
                [](const auto & a, const auto &b) { return a.bytes == b.bytes; }), candidates.end());

            std::vector<type_scores> hull;
            double min_err = INFINITE;
            for(const auto & c : candidates) {
                if (c.error < min_err) {
                    min_err = c.error;
                    hull.push_back(c);
                }
            }
            candidates = std::move(hull);

            if (candidates.size() < 3) { return; }
            std::vector<type_scores> convex;
            auto cross = [](const auto& a, const auto& b, const auto& c) {
                return ((double)b.bytes - (double)a.bytes) * (c.error - a.error) - ((double)c.bytes - (double)a.bytes) * (b.error - a.error);
            };

            for (const auto & c : candidates) {
                while (convex.size() >= 2 && cross(convex[convex.size()-2], convex.back(), c) <= EPSILON) { convex.pop_back(); }
                convex.push_back(c);
            }

            candidates = std::move(convex);
        };

        simplify_pareto(ch.candidates);
        ch.choice = 0;
        ch.min_bpw = ch.candidates.front().bpw;
        ch.max_bpw = ch.candidates.back().bpw;
        return ch;
    };

    std::vector<type_choice> all_tensors; // this vector will be populated by the parallel workers
    {
        std::atomic<size_t> idx{0};
        std::mutex m_load;
        std::mutex m_log;
        std::mutex m_res;
        std::exception_ptr w_exception;
        std::atomic<bool> w_failed{false};
        std::vector<std::thread> threads;
        int n_workers = std::max(1, std::min(nthread, (int)tensors.size()));
        threads.reserve(n_workers);

        for (int i = 0; i < n_workers; ++i) {
            threads.emplace_back([&](){
                try {
                    std::vector<no_init<uint8_t>> buf;
                    std::vector<float> f32_sample;
                    std::vector<float> smoothed_sample;
                    std::vector<float> val_cache;
                    std::vector<float> act_cache;
                    while(true) {
                        if (w_failed.load(std::memory_order_relaxed) || bpw_stop.load(std::memory_order_relaxed)) { break; }
                        const size_t cur = idx.fetch_add(1);
                        if (cur >= tensors.size()) { break; }
                        if (!can_quantize(tensors[cur]->tensor)) { continue; }

                        auto res = process_tensor(tensors[cur], buf, f32_sample, smoothed_sample, val_cache, act_cache, m_load, m_log);
                        if (res) {
                            std::lock_guard<std::mutex> lock(m_res);
                            all_tensors.push_back(std::move(*res));
                        } else {
                            break;
                        }
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> lock(m_res);
                    if (!w_failed.exchange(true)) {
                        w_exception = std::current_exception();
                    }
                }
            });
        }

        for(auto& t : threads) { t.join(); }

        if (w_exception) {
            std::rethrow_exception(w_exception);
        }
    }

    check_signal_handler(all_tensors);
    if (qs.params->state_file) { save_state(all_tensors); }
    if (all_tensors.empty()) { return {}; }

    // Compute total elements across all tensors and bytes for non-quantizable tensors
    size_t n_elements = 0;
    size_t nq_bytes = 0;
    for (const auto * it : tensors) {
        const ggml_tensor * tensor = it->tensor;
        n_elements += (size_t)ggml_nelements(tensor);
        if (!can_quantize(tensor)) { nq_bytes += ggml_nbytes(tensor); }
    }

    // Exclude user defined tensors (--tensor-type) from the Lagrangian optimization (courtesy of https://github.com/AesSedai)
    std::vector<type_choice> pinned_tensors;
    if (locked_tensors && !locked_tensors->empty()) {
        std::vector<type_choice> variable_tensors;
        variable_tensors.reserve(all_tensors.size());
        for (auto & tn : all_tensors) {
            const std::string name = ggml_get_name(tn.w->tensor);
            auto lt = locked_tensors->find(name);
            if (lt != locked_tensors->end()) {
                const ggml_type mc = tensor_type_fallback(qs, tn.w->tensor, lt->second);
                int idx = -1;
                for (int j = 0; j < (int)tn.candidates.size(); ++j) {
                    if (tn.candidates[j].type == mc) {
                        idx = j;
                        break;
                    }
                }

                if (idx == -1) {
                    // Pinned type was pareto-pruned or not evaluated; add with unknown error
                    type_scores ts;
                    ts.type = mc;
                    ts.bpw = (float)tensor_bpw(tn.w->tensor, mc);
                    ts.bytes = tensor_bytes(tn.w->tensor, mc);
                    ts.error = std::numeric_limits<float>::quiet_NaN();
                    tn.candidates.push_back(ts);
                    idx = (int)tn.candidates.size() - 1;
                }

                tn.choice = idx;
                pinned_tensors.push_back(std::move(tn));
            } else {
                variable_tensors.push_back(std::move(tn));
            }
        }

        all_tensors = std::move(variable_tensors);
    }

    size_t pinned_bytes = 0;
    for (const auto & tc : pinned_tensors) { pinned_bytes += tc.candidates[tc.choice].bytes; }

    size_t min_total_bytes = 0;
    size_t max_total_bytes = 0;
    for (const auto & tn : all_tensors) {
        min_total_bytes += tn.candidates.front().bytes;
        max_total_bytes += tn.candidates.back().bytes;
    }

    size_t total_budget_bytes = 0;
    size_t budget_bytes = 0;
    if (qs.params->target_size != -1) {
        const auto metadata_size = gguf_get_meta_size(ml.metadata);
        const size_t total_fixed_bytes = metadata_size + nq_bytes + pinned_bytes;
        total_budget_bytes = (size_t)qs.params->target_size;
        if (total_budget_bytes < total_fixed_bytes + min_total_bytes) {
            LLAMA_LOG_WARN("\t%s: requested file size %zu is smaller than minimum possible model size (~%zu bytes); clamping to minimum\n",
            func, (size_t)qs.params->target_size, total_fixed_bytes + min_total_bytes);
            budget_bytes = min_total_bytes;
        } else {
            budget_bytes = total_budget_bytes - total_fixed_bytes;
        }
    } else {
        total_budget_bytes = std::llround(qs.params->target_bpw * (double)n_elements / 8.0);
        const size_t total_fixed_bytes = nq_bytes + pinned_bytes;
        if (total_budget_bytes < total_fixed_bytes + min_total_bytes) {
            LLAMA_LOG_WARN("\t%s: requested BPW %.4f is smaller than minimum possible model size (~%.4f BPW); clamping to minimum\n",
                func, qs.params->target_bpw, (double)(total_fixed_bytes + min_total_bytes) * 8.0 / n_elements);
            budget_bytes = min_total_bytes;
        } else {
            budget_bytes = total_budget_bytes - total_fixed_bytes;
        }
    }

    if (locked_tensors && pinned_bytes >= total_budget_bytes) {
        LLAMA_LOG_WARN("%s: pinned tensors alone exceed the target budget; ignoring budget and using pinned tensors\n", func);
        budget_bytes = min_total_bytes;
    } else if (locked_tensors && pinned_bytes >= total_budget_bytes * 0.33) {
        LLAMA_LOG_WARN("%s: pinned tensors will consume over 1/3 of the target budget; optimization potential may be limited\n", func);
    }

    // Get the types' override
    auto build_mix = [&]() -> std::unordered_map<std::string, ggml_type> {
        std::unordered_map<std::string, ggml_type> quant_mix;
        std::unordered_map<std::string, const type_choice *> optimized;
        std::unordered_map<std::string, const type_choice *> pinned;
        optimized.reserve(all_tensors.size());
        for (const auto & tn : all_tensors) { optimized[ggml_get_name(tn.w->tensor)] = & tn; }
        pinned.reserve(pinned_tensors.size());
        for (const auto & tn : pinned_tensors) { pinned[ggml_get_name(tn.w->tensor)] = & tn; }

        LLAMA_LOG_INFO("\t%s: tensor quantization mix:\n", func);
        for (const auto * tn : tensors) {
            const std::string name = ggml_get_name(tn->tensor);

            auto pd = pinned.find(name);
            bool is_pinned = pd != pinned.end();
            const auto * tn_ptr = is_pinned ? pd->second : nullptr;

            if (!tn_ptr) {
                auto opt = optimized.find(name);
                if (opt != optimized.end()) { tn_ptr = opt->second; }
            }

            if (tn_ptr) {
                const auto & tp = * tn_ptr;
                const auto & choice = tp.candidates[tp.choice];

                LLAMA_LOG_INFO("\t%s: %45s %s\t%8s, \t%1.4f bpw,\terror: %.4f%s\n",
                    func,
                    name.c_str(),
                    tp.important ? "⬆︎" : "-",
                    ggml_type_name(choice.type),
                    choice.bpw,
                    choice.error,
                    is_pinned ? " (pinned)" : "");

                quant_mix[name] = choice.type;
            }
        }

        return quant_mix;
    };

    if (budget_bytes <= min_total_bytes) {
        for(auto & tn : all_tensors) { tn.choice = 0; }
        return build_mix();
    }
    if (budget_bytes >= max_total_bytes) {
        for(auto & tn : all_tensors) { tn.choice = (int)tn.candidates.size() - 1; }
        return build_mix();
    }

    // Certain tensors have a higher impact on model quality, so we apply a lower penalty to them
    auto is_important = [&](const std::string & tensor_name) -> bool {
        if (tensor_name == "output.weight") { return true; }

        return false;
    };

    // Determine tensor importance
    for (auto & tn : all_tensors) { tn.important = is_important(ggml_get_name(tn.w->tensor)); }

    // Minimize error subject to a size target constraint
    auto lagrangian_relaxation = [&](const double mu, std::vector<int> & choices, size_t & bytes, double & cost) {
        choices.resize(all_tensors.size());
        bytes = 0;
        cost = 0.0;
        for (size_t i = 0; i < all_tensors.size(); ++i) {
            const auto & tn = all_tensors[i];
            const double eff_mu = tn.important ? mu / boost : mu; // important tensors get a lower penalty

            int best = 0;
            for(int j = 1; j < (int)tn.candidates.size(); ++j) {
                double lr = (tn.candidates[j].error - tn.candidates[best].error) + eff_mu * ((double)tn.candidates[j].bytes - (double)tn.candidates[best].bytes) * 8.0;
                if (lr < -EPSILON || (std::abs(lr) <= EPSILON && tn.candidates[j].bytes < tn.candidates[best].bytes)) {
                    best = j;
                }
            }

            choices[i] = best;
            bytes += tn.candidates[best].bytes;
            cost += tn.candidates[best].error;
        }
    };

    // Binary search for mu
    double mu_lo = 0.0;
    double mu_hi = 1.0;
    std::vector<int> ch_lo;
    std::vector<int> ch_hi;
    std::vector<int> ch_under;
    std::vector<int> ch_over;
    size_t bt_lo;
    size_t bt_hi;
    size_t bt_mid;
    double dummy;

    lagrangian_relaxation(mu_lo, ch_lo, bt_lo, dummy);
    int safety = 0;

    do {
        lagrangian_relaxation(mu_hi, ch_hi, bt_hi, dummy);
        if (bt_hi <= budget_bytes || bt_hi == std::numeric_limits<size_t>::max()) { break; }
        mu_hi *= 2.0;
    } while(++safety < 60);

    double gap_under = INFINITE;
    double gap_over = INFINITE;

    for(int i = 0; i < 40; ++i) {
        double mu = 0.5 * (mu_lo + mu_hi);
        std::vector<int> ch_mid;
        double cost_mid = 0.0;
        lagrangian_relaxation(mu, ch_mid, bt_mid, cost_mid);

        double gap = std::abs((double)bt_mid - (double)budget_bytes);
        if (bt_mid > budget_bytes) {
            mu_lo = mu;
            if (gap < gap_over) {
                gap_over = gap;
                ch_over = ch_mid;
            }
        } else {
            mu_hi = mu;
            if (gap < gap_under) {
                gap_under = gap;
                ch_under = ch_mid;
            }
        }
    }

    if (!ch_under.empty()) {
        for(size_t i = 0; i < all_tensors.size(); ++i) { all_tensors[i].choice = ch_under[i]; }
    }
    else if (!ch_over.empty()) {
        for(size_t i = 0; i < all_tensors.size(); ++i) { all_tensors[i].choice = ch_over[i]; }
    }
    else if (bt_hi <= budget_bytes && !ch_hi.empty()) {
        for(size_t i = 0; i < all_tensors.size(); ++i) { all_tensors[i].choice = ch_hi[i]; }
    }
    else {
        for(auto& tn : all_tensors) { tn.choice = 0; }
    }

    // Single pass greedy upgrade in case there is budget left
    auto current_bytes = [&] {
        size_t cb = 0;
        for(const auto & tn : all_tensors) { cb += tn.candidates[tn.choice].bytes; }
        return cb;
    };
    size_t cb = current_bytes();

    struct tensor_upgrade {
        int index;
        int next_choice;
        double score;
        bool operator<(const tensor_upgrade & other) const {
            return score < other.score;
        }
    };

    std::priority_queue<tensor_upgrade> queue;

    auto push_next = [&](const int i) {
        const auto & tn = all_tensors[i];
        int next = tn.choice + 1;
        if (next < (int)tn.candidates.size()) {
            const double err = std::max(0.0, tn.candidates[tn.choice].error - tn.candidates[next].error);
            auto bytes = (double)(tn.candidates[next].bytes - tn.candidates[tn.choice].bytes);
            if (bytes > EPSILON) {
                double ratio = err / bytes;
                if (tn.important) { ratio *= boost; } // important tensors get a higher priority
                queue.push({i, next, ratio});
            }
        }
    };

    for (size_t i = 0; i < all_tensors.size(); ++i) { push_next((int)i); }

    while (!queue.empty()) {
        auto top = queue.top();
        queue.pop();

        int i = top.index;
        int next = top.next_choice;
        if (all_tensors[i].choice >= next) { continue; }

        size_t delta_bt = all_tensors[i].candidates[next].bytes - all_tensors[i].candidates[all_tensors[i].choice].bytes;
        if (cb + delta_bt <= budget_bytes) {
            cb += delta_bt;
            all_tensors[i].choice = next;
            push_next(i);
        }
    }

    return build_mix();
}

// imatrix requirement check
static bool tensor_requires_imatrix(const char * tensor_name, const ggml_type dst_type, const llama_ftype ftype) {
    if (tensor_name_match_token_embd(tensor_name) || tensor_name_match_output_weight(tensor_name)) { return false; }
    switch (dst_type) {
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ1_S:
            return true;
        case GGML_TYPE_Q2_K:
            // as a general rule, the k-type quantizations don't require imatrix data.
            // the only exception is Q2_K tensors that are part of a Q2_K_S file.
            return ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S;
        default:
            return false;
    }
}

// given a file type, get the default tensor type
ggml_type llama_ftype_get_default_type(const llama_ftype ftype) {
    switch (ftype) {
        case LLAMA_FTYPE_MOSTLY_Q4_0: return GGML_TYPE_Q4_0;
        case LLAMA_FTYPE_MOSTLY_Q4_1: return GGML_TYPE_Q4_1;
        case LLAMA_FTYPE_MOSTLY_Q5_0: return GGML_TYPE_Q5_0;
        case LLAMA_FTYPE_MOSTLY_Q5_1: return GGML_TYPE_Q5_1;
        case LLAMA_FTYPE_MOSTLY_Q8_0: return GGML_TYPE_Q8_0;
        case LLAMA_FTYPE_MOSTLY_F16:  return GGML_TYPE_F16;
        case LLAMA_FTYPE_MOSTLY_BF16: return GGML_TYPE_BF16;
        case LLAMA_FTYPE_ALL_F32:     return GGML_TYPE_F32;
        case LLAMA_FTYPE_MOSTLY_Q1_0: return GGML_TYPE_Q1_0;

        case LLAMA_FTYPE_MOSTLY_MXFP4_MOE: return GGML_TYPE_MXFP4;

        // K-quants
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:
        case LLAMA_FTYPE_MOSTLY_Q2_K:    return GGML_TYPE_Q2_K;
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:  return GGML_TYPE_IQ3_S;
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:  return GGML_TYPE_Q3_K;
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:  return GGML_TYPE_Q4_K;
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:  return GGML_TYPE_Q5_K;
        case LLAMA_FTYPE_MOSTLY_Q6_K:    return GGML_TYPE_Q6_K;
        case LLAMA_FTYPE_MOSTLY_TQ1_0:   return GGML_TYPE_TQ1_0;
        case LLAMA_FTYPE_MOSTLY_TQ2_0:   return GGML_TYPE_TQ2_0;
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS: return GGML_TYPE_IQ2_XXS;
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:  return GGML_TYPE_IQ2_XS;
        case LLAMA_FTYPE_MOSTLY_IQ2_S:   return GGML_TYPE_IQ2_XS;
        case LLAMA_FTYPE_MOSTLY_IQ2_M:   return GGML_TYPE_IQ2_S;
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS: return GGML_TYPE_IQ3_XXS;
        case LLAMA_FTYPE_MOSTLY_IQ1_S:   return GGML_TYPE_IQ1_S;
        case LLAMA_FTYPE_MOSTLY_IQ1_M:   return GGML_TYPE_IQ1_M;
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:  return GGML_TYPE_IQ4_NL;
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:  return GGML_TYPE_IQ4_XS;
        case LLAMA_FTYPE_MOSTLY_IQ3_S:
        case LLAMA_FTYPE_MOSTLY_IQ3_M:   return GGML_TYPE_IQ3_S;

        default: return GGML_TYPE_COUNT;
    }
}

static void init_quantize_state_counters(quantize_state_impl & qs, std::vector<tensor_metadata> & metadata) {
    for (auto & tm : metadata) {
        tm.category = tensor_get_category(tm.name);
        if (category_is_attn_v(tm.category)) { ++qs.n_attention_wv; }
        if (tm.category == TENSOR_CATEGORY_OUTPUT) { qs.has_tied_embeddings = false; }
    }

    qs.n_ffn_down = qs.n_ffn_gate = qs.n_ffn_up = (int)qs.model.hparams.n_layer_all;
}

// main quantization driver
static void llama_model_quantize_impl(const std::string & fname_inp, const std::string & fname_out, const llama_model_quantize_params * params) {
    llama_ftype ftype = params->ftype;
    int nthread = params->nthread;
    if (nthread <= 0) { nthread = (int)std::thread::hardware_concurrency(); }

    const ggml_type default_type = llama_ftype_get_default_type(ftype);
    if (default_type == GGML_TYPE_COUNT) { throw std::runtime_error(format("invalid output file type %d\n", ftype)); }

    // mmap consistently increases speed on Linux, and also increases speed on Windows with
    // hot cache. It may cause a slowdown on macOS, possibly related to free memory.
#if defined(__linux__) || defined(_WIN32)
    constexpr bool use_mmap = true;
#else
    constexpr bool use_mmap = false;
#endif

    const llama_model_kv_override * kv_overrides = params->kv_overrides;
    std::vector<std::string> splits = {};
    llama_model_loader ml(/*metadata*/ nullptr, /*set_tensor_data*/ nullptr, /*set_tensor_data_ud*/ nullptr,
        fname_inp, splits, /*file*/ nullptr, use_mmap, /*use_direct_io*/ false, /*check_tensors*/ true, /*no_alloc*/ false, kv_overrides, nullptr);
    ml.init_mappings(false); // no prefetching

    auto mparams = llama_model_default_params();
    std::unique_ptr<llama_model> model_ptr(llama_model_create(ml, mparams));

    auto * model = dynamic_cast<llama_model_base *>(model_ptr.get());
    if (model == nullptr) {
        GGML_ABORT("fatal error: model does not implement llama_model_base");
    }

    model->load_hparams(ml);
    model->load_stats  (ml);

    quantize_state_impl qs(*model, params);

    if (params->only_copy) { ftype = ml.ftype; }
    const std::unordered_map<std::string, std::vector<float>> * values_data = nullptr;
    const std::unordered_map<std::string, std::vector<float>> * activations_data = nullptr;
    const std::unordered_map<std::string, std::vector<float>> * statistics_data = nullptr;
    std::unordered_map<std::string, std::vector<float>> vdata;
    std::unordered_map<std::string, std::vector<float>> adata;
    std::unordered_map<std::string, std::vector<float>> sdata;
    if (params->imatrix) {
        for (const llama_model_imatrix_data * i = params->imatrix; i->name != nullptr; i++) {
            vdata.emplace(i->name, std::vector<float>(i->data, i->data + i->size));
        }

        values_data = & vdata;
        if (values_data) {
            LLAMA_LOG_INFO("%s: have importance matrix data with %d entries", __func__, (int)values_data->size());
            qs.has_imatrix = true;
            // check imatrix for nans or infs
            for (const auto & kv : * values_data) {
                for (float f : kv.second) {
                    if (!std::isfinite(f)) { throw std::runtime_error(format("imatrix contains non-finite value %f\n", f)); }
                }
            }
        }
    }
    if (params->activations) {
        for (const llama_model_imatrix_data * a = params->activations; a->name != nullptr; a++) {
            adata.emplace(a->name, std::vector<float>(a->data, a->data + a->size));
        }

        activations_data = & adata;
        if (activations_data) {
            LLAMA_LOG_INFO(" - %d activations", (int)activations_data->size());
            qs.has_activations = true;
            // check activations for nans or infs
            for (const auto & kv : * activations_data) {
                for (float f : kv.second) {
                    if (!std::isfinite(f)) { throw std::runtime_error(format("activations contain non-finite value %f\n", f)); }
                }
            }
        }
    }
    if (params->statistics) {
        for (const llama_model_imatrix_data * s = params->statistics; s->name != nullptr; s++) {
            sdata.emplace(s->name, std::vector<float>(s->data, s->data + s->size));
        }

        statistics_data = & sdata;
        if (statistics_data) { LLAMA_LOG_INFO(" and %d statistics", (int)statistics_data->size());
        }
    }
    LLAMA_LOG_INFO("\n");

    const size_t align = GGUF_DEFAULT_ALIGNMENT;
    gguf_context_ptr ctx_out { gguf_init_empty() };

    std::vector<int> prune_list = {};
    if (params->prune_layers) {
        for (const int32_t * p = params->prune_layers; * p != -1; p++) { prune_list.push_back(* p); }
    }

    // copy the KV pairs from the input file
    gguf_set_kv     (ctx_out.get(), ml.metadata);
    gguf_set_val_u32(ctx_out.get(), ml.llm_kv(LLM_KV_GENERAL_QUANTIZATION_VERSION).c_str(), GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out.get(), ml.llm_kv(LLM_KV_GENERAL_FILE_TYPE).c_str(), ftype);

    // Remove split metadata
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_NO).c_str());
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str());
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str());

    if (params->kv_overrides) {
        for (const llama_model_kv_override * kvo = params->kv_overrides; kvo->key[0] != 0; ++kvo) {
            if (kvo->tag == LLAMA_KV_OVERRIDE_TYPE_FLOAT) {
                gguf_set_val_f32(ctx_out.get(), kvo->key, kvo->val_f64);
            } else if (kvo->tag == LLAMA_KV_OVERRIDE_TYPE_INT) {
                // Setting type to UINT32. See https://github.com/ggml-org/llama.cpp/pull/14182 for context
                gguf_set_val_u32(ctx_out.get(), kvo->key, (uint32_t)std::abs(kvo->val_i64));
            } else if (kvo->tag == LLAMA_KV_OVERRIDE_TYPE_BOOL) {
                gguf_set_val_bool(ctx_out.get(), kvo->key, kvo->val_bool);
            } else if (kvo->tag == LLAMA_KV_OVERRIDE_TYPE_STR) {
                gguf_set_val_str(ctx_out.get(), kvo->key, kvo->val_str);
            } else {
                LLAMA_LOG_WARN("%s: unknown KV override type for key %s\n", __func__, kvo->key);
            }
        }
    }

    std::map<int, std::string> mapped;
    int blk_id = 0;

    // make a list of weights
    std::vector<const llama_model_loader::llama_tensor_weight *> tensors;
    tensors.reserve(ml.weights_map.size());
    for (const auto & it : ml.weights_map) {
        const std::string remapped_name(remap_layer(it.first, prune_list, mapped, blk_id));
        if (remapped_name.empty()) {
            LLAMA_LOG_DEBUG("%s: pruning tensor %s\n", __func__, it.first.c_str());
            continue;
        }
        if (remapped_name != it.first) {
            ggml_set_name(it.second.tensor, remapped_name.c_str());
            LLAMA_LOG_DEBUG("%s: tensor %s remapped to %s\n", __func__, it.first.c_str(), ggml_get_name(it.second.tensor));
        }

        tensors.push_back(&it.second);
    }
    if (!prune_list.empty()) { gguf_set_val_u32(ctx_out.get(), ml.llm_kv(LLM_KV_BLOCK_COUNT).c_str(), blk_id); }

    // keep_split requires that the weights are sorted by split index
    if (params->keep_split) {
        std::sort(tensors.begin(), tensors.end(), [](const llama_model_loader::llama_tensor_weight * a, const llama_model_loader::llama_tensor_weight * b) {
            if (a->idx == b->idx) { return a->offs < b->offs; }
            return a->idx < b->idx;
        });
    }

    // compute tensor metadata once and cache it
    std::vector<tensor_metadata> metadata(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) { metadata[i].name = ggml_get_name(tensors[i]->tensor); }

    // initialize quantization state counters and metadata categories
    init_quantize_state_counters(qs, metadata);

    int idx = 0;
    uint16_t n_split = 1;

    // Assume split index is continuous
    if (params->keep_split) {
        for (const auto * it : tensors) { n_split = std::max((uint16_t)(it->idx + 1), n_split); }
    }
    std::vector<gguf_context_ptr> ctx_outs(n_split);
    ctx_outs[0] = std::move(ctx_out);

    // flag for --dry-run
    bool will_require_imatrix = false;

    // preliminary iteration over all weights
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto * it = tensors[i];
        const struct ggml_tensor * tensor = it->tensor;

        uint16_t i_split = params->keep_split ? it->idx : 0;
        if (!ctx_outs[i_split]) { ctx_outs[i_split].reset(gguf_init_empty()); }
        gguf_add_tensor(ctx_outs[i_split].get(), tensor);

        metadata[i].allows_quantization = tensor_allows_quantization(params, model->arch, tensor);

        if (metadata[i].allows_quantization) { metadata[i].target_type = llama_tensor_get_type(qs, params, tensor, default_type, metadata[i]); }
        else { metadata[i].target_type = tensor->type; }

        metadata[i].requires_imatrix = tensor_requires_imatrix(tensor->name, metadata[i].target_type, ftype);

        if (params->imatrix) { metadata[i].remapped_imatrix_name = remap_imatrix(tensor->name, mapped); }
        else if (metadata[i].allows_quantization && metadata[i].requires_imatrix) {
            if (params->dry_run) { will_require_imatrix = true; }
            else {
                LLAMA_LOG_ERROR("\n============================================================================\n"
                                " ERROR: this quantization requires an importance matrix!\n"
                                "        - offending tensor: %s\n"
                                "        - target type: %s\n"
                                "============================================================================\n\n",
                                metadata[i].name.c_str(), ggml_type_name(metadata[i].target_type));
                throw std::runtime_error("this quantization requires an imatrix!");
            }
        }
    }

    // Set split info if needed
    if (n_split > 1) {
        for (size_t i = 0; i < ctx_outs.size(); ++i) {
            gguf_set_val_u16(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_NO).c_str(), i);
            gguf_set_val_u16(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str(), n_split);
            gguf_set_val_i32(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str(), (int32_t)tensors.size());
        }
    }

    size_t total_size_org = 0;
    size_t total_size_new = 0;

    std::vector<std::thread> workers;
    workers.reserve(nthread);

    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<uint8_t>> work;
    std::vector<no_init<float>> f32_conv_buf;

    std::unordered_map<std::string, ggml_type> bpw_overrides = {};
    if ((params->target_bpw != -1.0f || params->target_size != -1) && !params->only_copy) {
        if (params->imatrix) {
            if (params->pure) {
                LLAMA_LOG_WARN("%s: --target-bpw/--target-size specified with --pure, ignoring --pure\n", __func__);
            }
            if (params->target_size >= 0) {
                LLAMA_LOG_INFO("%s: computing tensor quantization mix to achieve file size %.2f MiB\n", __func__, (double)params->target_size / 1024.0 / 1024.0);
            } else {
                LLAMA_LOG_INFO("%s: computing tensor quantization mix to achieve %.4f bpw\n", __func__, params->target_bpw);
            }

            // Build locked tensor type map from --tensor-type patterns
            std::unordered_map<std::string, ggml_type> locked_tensors;
            if (!qs.tensor_type_patterns.empty()) {
                for (size_t i = 0; i < tensors.size(); ++i) {
                    if (!metadata[i].allows_quantization) { continue; }
                    const std::string name = ggml_get_name(tensors[i]->tensor);
                    for (const auto & [pattern, qtype] : qs.tensor_type_patterns) {
                        if (std::regex_search(name, pattern)) {
                            locked_tensors[name] = qtype;
                            break;
                        }
                    }
                }

                if (!locked_tensors.empty()) {
                    LLAMA_LOG_INFO("%s: locking %zu tensors to user-specified types, allocating remaining budget to the rest\n",
                        __func__, locked_tensors.size());
                }
            }

            // get quantization type overrides targeting a given bits per weight budget
            const auto * lt = locked_tensors.empty() ? nullptr : & locked_tensors;
            bpw_overrides = target_bpw_type(ml, qs, tensors, mapped, values_data, activations_data, statistics_data, lt, nthread);
            for (size_t i = 0; i < tensors.size(); ++i) {
                const std::string name = ggml_get_name(tensors[i]->tensor);
                auto it = bpw_overrides.find(name);
                if (it != bpw_overrides.end()) {
                    metadata[i].target_type = it->second;
                    metadata[i].requires_imatrix = tensor_requires_imatrix(name.c_str(), metadata[i].target_type, ftype);
                }
            }
        } else {
            throw std::runtime_error(format("--target-bpw/--target-size require an imatrix but none was provided\n"));
        }
    }

    int cur_split = -1;
    std::ofstream fout;
    auto close_ofstream = [&]() {
        // Write metadata and close file handler
        if (fout.is_open()) {
            fout.seekp(0);
            std::vector<uint8_t> data(gguf_get_meta_size(ctx_outs[cur_split].get()));
            gguf_get_meta_data(ctx_outs[cur_split].get(), data.data());
            fout.write((const char *) data.data(), data.size());
            fout.close();
        }
    };
    auto new_ofstream = [&](int index) {
        cur_split = index;
        GGML_ASSERT(ctx_outs[cur_split] && "Find uninitialized gguf_context");
        std::string fname = fname_out;
        if (params->keep_split) {
            std::vector<char> split_path(llama_path_max(), 0);
            llama_split_path(split_path.data(), split_path.size(), fname_out.c_str(), cur_split, n_split);
            fname = std::string(split_path.data());
        }

        fout = std::ofstream(fname, std::ios::binary);
        fout.exceptions(std::ofstream::failbit); // fail fast on write errors
        const size_t meta_size = gguf_get_meta_size(ctx_outs[cur_split].get());
        // placeholder for the meta data
        ::zeros(fout, meta_size);
    };

    // no output file for --dry-run
    if (!params->dry_run) { new_ofstream(0); }

    // main loop: iterate over all weights
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto & weight = *tensors[i];
        const auto & tm = metadata[i];
        ggml_tensor * tensor = weight.tensor;

        if (!params->dry_run && (weight.idx != cur_split && params->keep_split)) {
            close_ofstream();
            new_ofstream(weight.idx);
        }

        const size_t tensor_size = ggml_nbytes(tensor);
        if (!params->dry_run) {
            if (!ml.use_mmap) {
                if (read_data.size() < tensor_size) { read_data.resize(tensor_size); }
                tensor->data = read_data.data();
            }

            // Temporarily revert to the original name to locate the weight
            if (params->prune_layers) { ggml_set_name(tensor, remap_imatrix(tensor->name, mapped).c_str()); }

            ml.load_data_for(tensor);

            // Restore the remapped name for the final quantized output
            if (params->prune_layers) { ggml_set_name(tensor, tm.name.c_str()); }
        }

        LLAMA_LOG_INFO("[%4d/%4d] %-36s - [%s], type = %6s, ",
            ++idx, ml.n_tensors, ggml_get_name(tensor), llama_format_tensor_shape(tensor).c_str(), ggml_type_name(tensor->type));

        const ggml_type cur_type = tensor->type;
        const ggml_type new_type = tm.target_type;

        // If we've decided to quantize to the same type the tensor is already in then there's nothing to do.
        bool quantize = cur_type != new_type;

        void * new_data;
        size_t new_size;

        if (params->dry_run) {
            // the --dry-run option calculates the final quantization size without quantizing
            if (quantize) {
                new_size = ggml_nrows(tensor) * ggml_row_size(new_type, tensor->ne[0]);
                LLAMA_LOG_INFO("size = %.2f MiB -> %.2f MiB (%s)\n", tensor_size/1024.0/1024.0, new_size/1024.0/1024.0, ggml_type_name(new_type));
                if (!will_require_imatrix && tm.requires_imatrix) { will_require_imatrix = true; }
            } else {
                new_size = tensor_size;
                LLAMA_LOG_INFO("size = %.2f MiB\n", new_size/1024.0/1024.0);
            }
            total_size_org += tensor_size;
            total_size_new += new_size;
        } else {
            // no --dry-run, perform quantization
            if (!quantize) {
                new_data = tensor->data;
                new_size = tensor_size;
                LLAMA_LOG_INFO("size = %.2f MiB\n", tensor_size/1024.0/1024.0);
            } else {
                const int64_t nelements = ggml_nelements(tensor);

                const float * imatrix = nullptr;
                if (values_data) {
                    auto it = values_data->find(tm.remapped_imatrix_name);
                    if (it == values_data->end()) {
                        LLAMA_LOG_INFO("\n====== %s: did not find imatrix data for %s\n", __func__, tensor->name);
                    } else {
                        if (it->second.size() == (size_t)tensor->ne[0]*tensor->ne[2]) {
                            imatrix = it->second.data();
                        } else {
                            LLAMA_LOG_INFO("\n====== %s: imatrix size %d is different from tensor size %d for %s\n", __func__,
                                    (int)it->second.size(), (int)tensor->ne[0] * (int)tensor->ne[2], tensor->name);

                            // this can happen when quantizing an old mixtral model with split tensors with a new incompatible imatrix
                            // this is a significant error and it may be good idea to abort the process if this happens,
                            // since many people will miss the error and not realize that most of the model is being quantized without an imatrix
                            // tok_embd should be ignored in this case, since it always causes this warning
                            if (!tensor_name_match_token_embd(tensor->name)) {
                                throw std::runtime_error(format("imatrix size %d is different from tensor size %d for %s",
                                    (int)it->second.size(), (int)tensor->ne[0] * (int)tensor->ne[2], tensor->name));
                            }
                        }
                    }
                }
                if (!imatrix && tm.requires_imatrix) {
                    LLAMA_LOG_ERROR("\n\n============================================================\n");
                    LLAMA_LOG_ERROR("Missing importance matrix for tensor %s in a very low-bit quantization\n", tensor->name);
                    LLAMA_LOG_ERROR("The result will be garbage, so bailing out\n");
                    LLAMA_LOG_ERROR("============================================================\n\n");
                    throw std::runtime_error(format("Missing importance matrix for tensor %s in a very low-bit quantization", tensor->name));
                }

                float * f32_data;

                if (tensor->type == GGML_TYPE_F32) {
                    f32_data = (float *) tensor->data;
                } else if (ggml_is_quantized(tensor->type) && !params->allow_requantize) {
                    throw std::runtime_error(format("requantizing from type %s is disabled", ggml_type_name(tensor->type)));
                } else {
                    llama_tensor_dequantize_impl(tensor, f32_conv_buf, workers, nelements, nthread);
                    f32_data = (float *) f32_conv_buf.data();
                }

                LLAMA_LOG_INFO("converting to %7s ", ggml_type_name(new_type));
                fflush(stdout);

                if (work.size() < (size_t)nelements * 4) { work.resize(nelements * 4); } // upper bound on size
                new_data = work.data();

                const int64_t n_per_row = tensor->ne[0];
                const int64_t nrows = tensor->ne[1];

                static constexpr int64_t min_chunk_size = 32 * 512;
                const int64_t chunk_size = (n_per_row >= min_chunk_size ? n_per_row : n_per_row * ((min_chunk_size + n_per_row - 1)/n_per_row));

                const int64_t nelements_matrix = tensor->ne[0] * tensor->ne[1];
                const int64_t nchunk = (nelements_matrix + chunk_size - 1)/chunk_size;
                const int64_t nthread_use = nthread > 1 ? std::max((int64_t)1, std::min((int64_t)nthread, nchunk)) : 1;

                // quantize each expert separately since they have different importance matrices
                new_size = 0;
                for (int64_t i03 = 0; i03 < tensor->ne[2]; ++i03) {
                    const float * f32_data_03 = f32_data + i03 * nelements_matrix;
                    void * new_data_03 = (char *)new_data + ggml_row_size(new_type, n_per_row) * i03 * nrows;
                    const float * imatrix_03 = imatrix ? imatrix + i03 * n_per_row : nullptr;

                    new_size += llama_tensor_quantize_impl(new_type, f32_data_03, new_data_03, chunk_size, nrows, n_per_row, imatrix_03, workers, nthread_use);
                }
                LLAMA_LOG_INFO("size = %.2f MiB -> %.2f MiB\n", tensor_size/1024.0/1024.0, new_size/1024.0/1024.0);
            }
            total_size_org += tensor_size;
            total_size_new += new_size;

            // update the gguf meta data as we go
            gguf_set_tensor_type(ctx_outs[cur_split].get(), metadata[i].name.c_str(), new_type);
            GGML_ASSERT(gguf_get_tensor_size(ctx_outs[cur_split].get(), gguf_find_tensor(ctx_outs[cur_split].get(), metadata[i].name.c_str())) == new_size);
            gguf_set_tensor_data(ctx_outs[cur_split].get(), metadata[i].name.c_str(), new_data);

            // write tensor data + padding
            fout.write((const char *) new_data, new_size);
            zeros(fout, GGML_PAD(new_size, align) - new_size);

            if (ml.use_mmap) {
                ml.mappings.at(weight.idx)->unmap_fragment(weight.offs, weight.offs + tensor_size);
            }
        }
    }

    if (!params->dry_run) { close_ofstream(); }

    LLAMA_LOG_INFO("%s: model size  = %8.2f MiB (%7.4f BPW)\n", __func__, total_size_org/1024.0/1024.0, total_size_org*8.0/ml.n_elements);
    LLAMA_LOG_INFO("%s: quant size  = %8.2f MiB (%7.4f BPW)\n", __func__, total_size_new/1024.0/1024.0, total_size_new*8.0/ml.n_elements);

    if (!params->imatrix && params->dry_run && will_require_imatrix) {
        LLAMA_LOG_WARN("%s: WARNING: dry run completed successfully, but actually completing this quantization will require an imatrix!\n", __func__);
    }

    if (qs.n_fallback > 0) {
        LLAMA_LOG_WARN("%s: WARNING: %d of %d tensor(s) required fallback quantization\n", __func__, qs.n_fallback, ml.n_tensors);
    }
}

// interface implementation
llama_model_quantize_params llama_model_quantize_default_params() {
    llama_model_quantize_params result = {
        /*.nthread                     =*/ 0,
        /*.ftype                       =*/ LLAMA_FTYPE_MOSTLY_Q8_0,
        /*.output_tensor_type          =*/ GGML_TYPE_COUNT,
        /*.token_embedding_type        =*/ GGML_TYPE_COUNT,
        /*.allow_requantize            =*/ false,
        /*.quantize_output_tensor      =*/ true,
        /*.only_copy                   =*/ false,
        /*.pure                        =*/ false,
        /*.keep_split                  =*/ false,
        /*.dry_run                     =*/ false,
        /*.imatrix                     =*/ nullptr,
        /*.activations                 =*/ nullptr,
        /*.statistics                  =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.tensor_type                 =*/ nullptr,
        /*.prune_layers                =*/ nullptr,
        /*.target_bpw                  =*/ -1.0f,
        /*.target_size                 =*/ -1,
        /*.state_file                  =*/ nullptr
    };

    return result;
}

uint32_t llama_model_quantize(const char * fname_inp, const char * fname_out, const llama_model_quantize_params * params) {
    try { llama_model_quantize_impl(fname_inp, fname_out, params); }
    catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to quantize: %s\n", __func__, err.what());
        return 1;
    }

    return 0;
}

// Helper functions for external tools exposed in llama-ext.h
quantize_state_impl * llama_quant_init(const llama_model * model, const llama_model_quantize_params * params) {
    return new quantize_state_impl(*model, params);
}

void llama_quant_free(const quantize_state_impl * qs) {
    delete qs;
}

llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc) {
    struct llama_model_params mparams = llama_model_default_params();
    const auto arch = llm_arch_from_string(desc->architecture);
    auto * model = llama_model_create(arch, mparams);
    model->arch = arch;

    // infer llm_type: only LLM_TYPE_70B matters for quantization logic
    if (model->arch == LLM_ARCH_LLAMA && desc->n_layer == 80 && desc->n_head != desc->n_head_kv) { model->type = LLM_TYPE_70B; }

    model->hparams.n_embd = desc->n_embd;
    model->hparams.n_embd_head_k_full = desc->n_embd_head_k;
    model->hparams.n_embd_head_v_full = desc->n_embd_head_v;
    model->hparams.n_layer_all        = desc->n_layer;
    model->hparams.n_expert           = desc->n_expert;

    for (uint32_t i = 0; i < desc->n_layer; i++) {
        model->hparams.n_head_arr[i] = desc->n_head;
        model->hparams.n_head_kv_arr[i] = desc->n_head_kv;
        model->hparams.n_ff_arr[i] = desc->n_ff;
    }

    return model;
}

bool llama_quant_tensor_allows_quantization(const quantize_state_impl * qs, const ggml_tensor * tensor) {
    return tensor_allows_quantization(qs->params, qs->model.arch, tensor);
}

void llama_quant_compute_types(
    quantize_state_impl * qs,
    const llama_ftype ftype,
    ggml_tensor ** tensors,
    ggml_type * result_types,
    const size_t n_tensors
) {
    // reset per-computation state
    qs->n_attention_wv = 0;
    qs->n_ffn_down = 0;
    qs->n_ffn_gate = 0;
    qs->n_ffn_up = 0;
    qs->i_attention_wv = 0;
    qs->i_ffn_down = 0;
    qs->i_ffn_gate = 0;
    qs->i_ffn_up = 0;
    qs->n_fallback = 0;
    qs->has_imatrix = false;
    qs->has_tied_embeddings = true;

    // build metadata from tensor names
    std::vector<tensor_metadata> metadata(n_tensors);
    for (size_t i = 0; i < n_tensors; i++) { metadata[i].name = ggml_get_name(tensors[i]); }

    // initialize counters and categories
    init_quantize_state_counters(*qs, metadata);

    // use a local copy of params with the requested ftype
    llama_model_quantize_params local_params = *qs->params;
    local_params.ftype = ftype;

    const ggml_type default_type = llama_ftype_get_default_type(ftype);

    // compute types
    for (size_t i = 0; i < n_tensors; i++) { result_types[i] = llama_tensor_get_type(*qs, &local_params, tensors[i], default_type, metadata[i]); }
}
