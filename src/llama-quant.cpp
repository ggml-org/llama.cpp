#include "llama-quant.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-model-loader.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <csignal>
#include <cstring>
#include <fstream>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <regex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

// result of parsing --tensor-type option
// (changes to this struct must be reflected in tools/quantize/quantize.cpp)
struct tensor_type_option {
    std::string name;
    ggml_type type = GGML_TYPE_COUNT;
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
        if (params->tensor_types) {
            const auto & tensor_types = *static_cast<const std::vector<tensor_type_option> *>(params->tensor_types);
            for (const auto & [tname, qtype] : tensor_types) { tensor_type_patterns.emplace_back(std::regex(tname), qtype); }
        }
    }
};

// per-tensor metadata, computed in the preliminary loop and used in the main loop
struct tensor_metadata {
    ggml_type target_type;
    tensor_category category;
    std::string remapped_imatrix_name;
    bool allows_quantization;
    bool requires_imatrix;
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
    quantize &= name.find(".position_embd.") == std::string::npos;

    return quantize;
}

// incompatible tensor shapes are handled here - fallback to a compatible type
static ggml_type tensor_type_fallback(quantize_state_impl & qs, const ggml_tensor * t, const ggml_type target_type) {
    ggml_type fallback_type = target_type;
    const int64_t ncols = t->ne[0];
    const int64_t qk_k = ggml_blck_size(target_type);

    if (ncols % qk_k != 0) { // this tensor's shape is incompatible with this quant
        LLAMA_LOG_WARN("warning: %-36s - ncols %6" PRId64 " not divisible by %3" PRId64 " (required for type %7s) ",
            t->name, ncols, qk_k, ggml_type_name(target_type));

        ++qs.n_fallback;

        switch (target_type) {
            // types on the left, block size 256; types on the right, block size 32
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ1_M:
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ2_S:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_IQ3_S:
            case GGML_TYPE_IQ4_XS: fallback_type = GGML_TYPE_IQ4_NL; break;
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_TQ1_0:
            case GGML_TYPE_TQ2_0:  fallback_type = GGML_TYPE_Q4_0; break;
            case GGML_TYPE_Q4_K:   fallback_type = GGML_TYPE_Q5_0; break;
            case GGML_TYPE_Q5_K:   fallback_type = GGML_TYPE_Q5_1; break;
            case GGML_TYPE_Q6_K:   fallback_type = GGML_TYPE_Q8_0; break;
            default:
                throw std::runtime_error(format("no tensor type fallback is defined for type %s", ggml_type_name(target_type)));
        }
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
    }

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
                        manual = true;
                        break;
                    }
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
        GGML_TYPE_Q2_K,
        GGML_TYPE_IQ3_XXS,
        GGML_TYPE_Q3_K,
        GGML_TYPE_IQ4_XS,
        GGML_TYPE_IQ4_NL,
        GGML_TYPE_Q4_K,
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
    constexpr float penalty = 5.0f;
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
        const ggml_type fb = tensor_type_fallback(qs, gt, gq);
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
            const auto * filename = static_cast<const char*>(qs.params->state_file);
            bool is_valid = false;

            if (std::ifstream(filename, std::ios::binary).good()) {
                is_valid = true;
            } else if (qs.params->save_state) {
                std::ofstream ofs(filename, std::ios::binary | std::ios::app);
                if (ofs.is_open()) {
                    is_valid = true;
                    ofs.close();
                    std::remove(filename);
                }
            }

            if (is_valid) {
                checkpoint_file = filename;
            } else {
                LLAMA_LOG_WARN("%s: '%s' is not a valid state file\n", func, filename);
                checkpoint_file.clear();
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

        ofs.close();
        std::remove(checkpoint_file.c_str());
        std::rename(tmp.c_str(), checkpoint_file.c_str());
        LLAMA_LOG_INFO("%s: saved target progress for %lu tensors to %s\n", func, all_tensors.size(), checkpoint_file.c_str());
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

        LLAMA_LOG_INFO("%s: state file found, resuming tensor quantization\n", func);

        std::unordered_map<std::string, type_choice> out;
        uint64_t n = 0;
        ifs.read((char *)& n, sizeof(n));
        for (uint64_t i = 0; i < n; ++i) {
            uint32_t len = 0;
            ifs.read((char *)& len, sizeof(len));
            std::string name(len, '\0');
            ifs.read(name.data(), len);

            type_choice si;
            uint64_t sz = 0;
            ifs.read((char *)& sz, sizeof(sz));
            ifs.read((char *)& si.choice, sizeof(si.choice));
            ifs.read((char *)& si.min_bpw, sizeof(si.min_bpw));
            ifs.read((char *)& si.max_bpw, sizeof(si.max_bpw));
            uint64_t ne = 0;
            ifs.read((char *)& ne, sizeof(ne));
            si.n_elements = (size_t)ne;

            si.candidates.resize(sz);
            for (auto & cd : si.candidates) {
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

        LLAMA_LOG_INFO("%s: resuming from %s (data for %lu tensors loaded)\n", func, checkpoint_file.c_str(), out.size());
        return out;
    };

    // Check for user interrupt and save progress
    auto check_signal_handler = [&](const std::vector<type_choice> & all_tensors) {
        if (bpw_stop.load(std::memory_order_relaxed)) {
            LLAMA_LOG_INFO("\n%s: interrupted, saving progress for %lu tensors to %s\n", func, all_tensors.size(), checkpoint_file.c_str());
            save_state(all_tensors);
            throw std::runtime_error("user terminated the process");
        }
    };

    // Quality metrics
    struct quant_error {
        double error = INFINITE;
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
        if (quantized_buffer.size() < row_sz * sample_rows + SAFETY_PADDING) { quantized_buffer.resize(row_sz * sample_rows + SAFETY_PADDING); }
        if (dequantized_buffer.size() < sample_elems) { dequantized_buffer.resize(sample_elems); }

        const bool has_vals = values_sample != nullptr;
        const bool has_acts = activations_sample != nullptr;
        const bool use_wce_for_tensor = has_acts && has_vals && is_angle_sensitive(t->name);

        // Sampled stats for MSE
        std::vector<double> local_row_sq_norm;
        const std::vector<double> * ptr_row_sq_norm = nullptr;

        // Setup reference stats pointers for MSE
        if (!use_wce_for_tensor) {
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
                                const double w = std::max(0.0f, val[j]);
                                const double act_j = act[j];
                                sum += w * x[j] * x[j];
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

        // Quantize & dequantize row samples
        {
            size_t qoff = 0;
            size_t foff = 0;
            for (int64_t s = 0; s < ne2; ++s) {
                const int64_t rs = rows_sample[s];
                if (rs == 0) { continue; }

                const float * v = has_vals ? values_sample + s * n_per_row : nullptr;
                ggml_quantize_chunk(quant_type, f32_sample.data() + foff, quantized_buffer.data() + qoff, 0, rs, n_per_row, v);
                qoff += row_sz * (size_t)rs;
                foff += (size_t)rs * (size_t)n_per_row;
            }

            const ggml_type_traits * traits = ggml_get_type_traits(quant_type);
            if (!traits || !traits->to_float) { return qe; }
            for (size_t r = 0; r < sample_rows; ++r) {
                const void * src = quantized_buffer.data() + r * row_sz;
                float * dst = dequantized_buffer.data() + r * (size_t)n_per_row;
                if (quant_type == GGML_TYPE_F16) { ggml_fp16_to_fp32_row((const ggml_fp16_t *)src, dst, (int)n_per_row); }
                else if (quant_type == GGML_TYPE_BF16) { ggml_bf16_to_fp32_row((const ggml_bf16_t *)src, dst, (int)n_per_row); }
                else { traits->to_float(src, dst, (int)n_per_row); }
            }
        }

        // Helper for trimmed mean
        auto trimmed_mean = [](std::vector<double> & v) -> double {
            const auto n = v.size();
            if (n == 0) { return 0.0; }
            if (n < 50) { return std::accumulate(v.begin(), v.end(), 0.0) / (double)n; }
            const auto k = (size_t)((double)n * 0.01); // trim 1% from each end
            std::nth_element(v.begin(), v.begin() + k, v.end());
            std::nth_element(v.begin() + k, v.end() - k, v.end());
            return std::accumulate(v.begin() + k, v.end() - k, 0.0) / std::max(1.0, (double)(n - 2 * k));
        };

        // Weighted Cosine Error (WCE)
        if (use_wce_for_tensor) {
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
                    const float * wy = dequantized_buffer.data() + off;

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
                const float * y = dequantized_buffer.data() + off;
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
    if (qs.params->state_file && !checkpoint_file.empty()) { bpw_data = load_state(); } // ToDo: rethink this condition

    // Parallelize tensor processing (courtesy of https://github.com/ddh0)
    auto process_tensor = [&](
        const llama_model_loader::llama_tensor_weight * tw,
        std::vector<no_init<uint8_t>> & thread_local_buffer,
        std::mutex & loader_mutex,
        std::mutex & log_mutex
    ) -> std::optional<type_choice>
    {
        ggml_tensor * tensor = tw->tensor;
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
            ml.load_data_for(tensor);
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
        std::vector<float> f32_sample;
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
                const int64_t stride = std::max<int64_t>(1, nrows_total / limit);
                int64_t offset = stride > 1 ? std::uniform_int_distribution<int64_t>(0, stride - 1)(rng) : 0;

                int64_t count = 0;
                for (int64_t r = offset; r < nrows_total && count < limit; r += stride) {
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

        // Prepare side data
        auto get_side_data = [&](const auto * m) {
            if (!m) { return std::pair<const float *, size_t>{nullptr, 0}; }
            auto it = m->find(remapped_name);
            return it != m->end() ? std::pair{it->second.data(), it->second.size()} : std::pair<const float*, size_t>{nullptr, 0};
        };

        auto [val_ptr, val_sz] = get_side_data(values_data);
        auto [act_ptr, act_sz] = get_side_data(activations_data);

        // Cache WCE stats per tensor
        std::vector<float> val_storage;
        std::vector<float> act_storage;
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

        prepare_broadcast(val_ptr, val_sz, val_storage, val_vec_ptr);
        prepare_broadcast(act_ptr, act_sz, act_storage, act_vec_ptr);

        const bool use_wce_for_tensor = val_vec_ptr && act_vec_ptr && is_angle_sensitive(remapped_name);

        // Precompute WCE reference stats
        wce_cache ref_wce;
        mse_cache ref_mse;
        size_t total_rows_sampled = 0;
        for (int64_t r : rows_sample) { total_rows_sampled += r; }

        if (use_wce_for_tensor) {
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
                            const double w = std::max(0.0f, val[j]);
                            const double act_j = act[j];
                            sum += w * x[j] * x[j];
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
            q_buf.reserve(total_rows_sampled * max_row_sz + 256); // safety padding
            dq_buf.reserve(total_rows_sampled * n_per_row);
        }

        // Kurtosis-Gain error scaling factor
        float scaling_factor = 1.0f;
        if (statistics_data) {
            if (auto it = statistics_data->find(remapped_name); it != statistics_data->end() && !it->second.empty()) {
                const auto & ts = it->second;
                const float gain = std::isnan(ts[GAIN]) ? 1.0f : ts[GAIN];
                const float alignment = std::isfinite(ts[COSSIM]) ? std::max(0.0f, 1.0f - ts[COSSIM]) : 0.0f;
                const float concentration = 1.0f - std::clamp(ts[H_NORM], 0.0f, 100.0f) / 100.0f;
                scaling_factor = 1.0f + std::log1p(std::max(0.0f, ts[KURTOSIS])) * std::max(1.0f, gain) * (1.0f + alignment) * (1.0f + concentration);
            }
        }

        for (ggml_type vt : valid_types) {
            if (bpw_stop.load(std::memory_order_relaxed)) { return std::nullopt; }
            const wce_cache * ptr_ref_wce = use_wce_for_tensor && !ref_wce.row_sq_norm.empty() ? & ref_wce : nullptr;
            const mse_cache * ptr_ref_mse = !use_wce_for_tensor && !ref_mse.row_sq_norm.empty() ? & ref_mse : nullptr;

            quant_error qe = compute_quant_error(
                tensor,
                vt,
                f32_sample,
                rows_sample,
                val_vec_ptr,
                act_vec_ptr,
                q_buf,
                dq_buf,
                ptr_ref_wce,
                ptr_ref_mse
            );

            type_scores candidate;
            candidate.type = vt;
            candidate.bpw = (float)tensor_bpw(tensor, vt);
            candidate.bytes = tensor_bytes(tensor, vt);
            candidate.error = qe.error * scaling_factor;
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
        std::vector<std::thread> threads;
        int n_workers = std::max(1, std::min(nthread, (int)tensors.size()));
        threads.reserve(n_workers);

        for (int i = 0; i < n_workers; ++i) {
            threads.emplace_back([&](){
                std::vector<no_init<uint8_t>> buf;
                while(true) {
                    const size_t cur = idx.fetch_add(1);
                    if (cur >= tensors.size()) { break; }
                    if (!can_quantize(tensors[cur]->tensor)) { continue; }

                    auto res = process_tensor(tensors[cur], buf, m_load, m_log);
                    if (res) {
                        std::lock_guard<std::mutex> lock(m_res);
                        all_tensors.push_back(std::move(*res));
                    }
                }
            });
        }

        for(auto& t : threads) { t.join(); }
    }

    check_signal_handler(all_tensors);
    if (qs.params->save_state) { save_state(all_tensors); }
    if (all_tensors.empty()) { return {}; }

    // Compute total elements across all tensors and bytes for non-quantizable tensors
    size_t nq_elements = 0;
    size_t nq_bytes = 0;
    for (const auto * it : tensors) {
        const ggml_tensor * tensor = it->tensor;
        nq_elements += (size_t)ggml_nelements(tensor);
        if (!can_quantize(tensor)) { nq_bytes += ggml_nbytes(tensor); }
    }

    size_t min_total_bytes = 0;
    size_t max_total_bytes = 0;
    for (const auto & tn : all_tensors) {
        min_total_bytes += tn.candidates.front().bytes;
        max_total_bytes += tn.candidates.back().bytes;
    }

    size_t budget_bytes = 0;

    if (qs.params->target_size != -1) {
        const auto metadata_size = gguf_get_meta_size(ml.metadata);
        // Budget for quantizable weights = target - metadata - Non-Quantizable Weights
        int64_t available = (int64_t)qs.params->target_size - (int64_t)metadata_size - (int64_t)nq_bytes;

        // Clamp to the absolute minimum possible size for the variable tensors
        if (available < (int64_t)min_total_bytes) {
            LLAMA_LOG_WARN("%s: requested file size %zu is smaller than minimum possible model size (~%zu), clamping to minimum.\n",
                func, (size_t)qs.params->target_size, min_total_bytes + nq_bytes + metadata_size);
            budget_bytes = min_total_bytes;
        } else {
            budget_bytes = (size_t)available;
        }
    } else {
        const double target_bpw = qs.params->target_bpw;
        size_t target_total_bytes = std::llround(target_bpw * (double)nq_elements / 8.0);
        budget_bytes = target_total_bytes >= nq_bytes ? target_total_bytes - nq_bytes : min_total_bytes;
    }

    // Get the types' override
    auto build_mix = [&]() -> std::unordered_map<std::string, ggml_type> {
        std::unordered_map<std::string, ggml_type> mix;
        LLAMA_LOG_INFO("\t%s: tensor quantization mix:\n", func);
        for (const auto & tn : all_tensors) {
            LLAMA_LOG_INFO("\t%s: %45s %s\t%8s, \t%1.4f bpw,\terror: %.4f\n",
                func, ggml_get_name(tn.w->tensor), tn.important ? "⬆︎" : "-", ggml_type_name(tn.candidates[tn.choice].type), tn.candidates[tn.choice].bpw,
                tn.candidates[tn.choice].error);
            mix[ggml_get_name(tn.w->tensor)] = tn.candidates[tn.choice].type;
        }

        return mix;
    };

    if (budget_bytes <= min_total_bytes) {
        for(auto & tn : all_tensors) { tn.choice = 0; }
        return build_mix();
    }
    if (budget_bytes >= max_total_bytes) {
        for(auto & tn : all_tensors) { tn.choice = (int)tn.candidates.size() - 1; }
        return build_mix();
    }

    auto importance_score = [](const std::vector<float> & tstats) -> float {
        if (tstats.size() < 12) { return 0.0f; }

        const float energy = std::log1pf(std::max(0.0f, (float)tstats[ENERGY]));
        const float range = 1.0f + std::max(0.0f, tstats[STDDEV]);
        const float magnitude = std::isfinite(tstats[L2_DIST]) ? 1.0f + tstats[L2_DIST] : 1.0f;
        const float alignment = std::isfinite(tstats[COSSIM]) ? 1.0f - tstats[COSSIM] : 1.0f;
        const float concentration = 1.0f - std::clamp(tstats[H_NORM], 0.0f, 100.0f) / 100.0f + EPSILON;

        return energy * range * magnitude * alignment * concentration;
    };

    // Threshold at which pct of tensors will be marked as important
    auto threshold_score = [&](const std::unordered_map<std::string, std::vector<float>> & stats, const float pct) -> float {
        if (stats.empty() || pct < 0.0f || pct > 100.0f) { return std::numeric_limits<float>::quiet_NaN(); }

        std::vector<float> val;
        val.reserve(stats.size());
        for (const auto & ts : stats) { val.push_back(importance_score(ts.second)); }
        if (val.empty()) { return std::numeric_limits<float>::quiet_NaN(); }

        size_t idx = std::round((1.0f - pct / 100.0f) * (val.size() - 1));
        if (idx >= val.size()) { idx = val.size() - 1; }
        std::nth_element(val.begin(), val.begin() + idx, val.end());

        return val[idx];
    };

    float cutoff = std::numeric_limits<float>::quiet_NaN();
    if (statistics_data && !statistics_data->empty()) { cutoff = threshold_score(* statistics_data, qs.params->importance_pct); }

    // Certain tensors have a higher impact on model quality, so we apply a lower penalty to them
    auto is_important = [&](const std::string & tensor_name) -> bool {
        if (tensor_name == "output.weight") { return true; }
        if (qs.params->importance_pct == 0.0f) { return false; }
        if (std::isfinite(cutoff)) {
            if (auto it = statistics_data->find(remap_imatrix(tensor_name, mapped)); it != statistics_data->end() && !it->second.empty()) {
                return importance_score(it->second) >= cutoff;
            }
        } else {
            return tensor_name.find(".attn_output.weight") != std::string::npos ||
                tensor_name.find(".attn_o.weight") != std::string::npos ||
                tensor_name.find(".attn_v.weight") != std::string::npos ||
                tensor_name.find(".ffn_down.weight") != std::string::npos ||
                tensor_name.find(".ffn_down_exps.weight") != std::string::npos ||
                tensor_name.find(".time_mix_output.weight") != std::string::npos ||
                tensor_name.find(".time_mix_value.weight") != std::string::npos;
        }

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
            const double eff_mu = tn.important ? mu / penalty : mu; // important tensors get a lower penalty

            int best = 0;
            double min = INFINITE;

            for(int j = 0; j < (int)tn.candidates.size(); ++j) {
                double lr = tn.candidates[j].error + eff_mu * (double)tn.candidates[j].bytes * 8.0;
                if (lr < min - EPSILON || (std::abs(lr - min) <= EPSILON && tn.candidates[j].bytes < tn.candidates[best].bytes)) {
                    min = lr;
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
                if (tn.important) { ratio *= penalty; } // important tensors get a higher priority
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
static ggml_type llama_ftype_get_default_type(llama_ftype ftype) {
    switch (ftype) {
        case LLAMA_FTYPE_MOSTLY_Q4_0: return GGML_TYPE_Q4_0;
        case LLAMA_FTYPE_MOSTLY_Q4_1: return GGML_TYPE_Q4_1;
        case LLAMA_FTYPE_MOSTLY_Q5_0: return GGML_TYPE_Q5_0;
        case LLAMA_FTYPE_MOSTLY_Q5_1: return GGML_TYPE_Q5_1;
        case LLAMA_FTYPE_MOSTLY_Q8_0: return GGML_TYPE_Q8_0;
        case LLAMA_FTYPE_MOSTLY_F16:  return GGML_TYPE_F16;
        case LLAMA_FTYPE_MOSTLY_BF16: return GGML_TYPE_BF16;
        case LLAMA_FTYPE_ALL_F32:     return GGML_TYPE_F32;

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

        default: throw std::runtime_error(format("invalid output file type %d\n", ftype));
    }
}

// main quantization driver
static void llama_model_quantize_impl(const std::string & fname_inp, const std::string & fname_out, const llama_model_quantize_params * params) {
    ggml_type default_type;
    llama_ftype ftype = params->ftype;

    int nthread = params->nthread;

    if (nthread <= 0) { nthread = std::thread::hardware_concurrency(); }
    default_type = llama_ftype_get_default_type(ftype);

    // mmap consistently increases speed on Linux, and also increases speed on Windows with
    // hot cache. It may cause a slowdown on macOS, possibly related to free memory.
#if defined(__linux__) || defined(_WIN32)
    constexpr bool use_mmap = true;
#else
    constexpr bool use_mmap = false;
#endif

    llama_model_kv_override * kv_overrides = nullptr;
    if (params->kv_overrides) {
        auto * v = (std::vector<llama_model_kv_override>*)params->kv_overrides;
        kv_overrides = v->data();
    }

    std::vector<std::string> splits = {};
    llama_model_loader ml(/*metadata*/ nullptr, /*set_tensor_data*/ nullptr, /*set_tensor_data_ud*/ nullptr,
        fname_inp, splits, use_mmap, /*use_direct_io*/ false, /*check_tensors*/ true, /*no_alloc*/ false, kv_overrides, nullptr);
    ml.init_mappings(false); // no prefetching

    llama_model model(llama_model_default_params());

    model.load_arch   (ml);
    model.load_hparams(ml);
    model.load_stats  (ml);

    quantize_state_impl qs(model, params);

    if (params->only_copy) { ftype = ml.ftype; }
    const std::unordered_map<std::string, std::vector<float>> * values_data = nullptr;
    const std::unordered_map<std::string, std::vector<float>> * activations_data = nullptr;
    const std::unordered_map<std::string, std::vector<float>> * statistics_data = nullptr;
    if (params->imatrix) {
        values_data = static_cast<const std::unordered_map<std::string, std::vector<float>>*>(params->imatrix);
        if (values_data) {
            LLAMA_LOG_INFO("%s: have importance matrix data with %d entries", __func__, (int)values_data->size());
            qs.has_imatrix = true;
            // check imatrix for nans or infs
            for (const auto & kv : *values_data) {
                for (float f : kv.second) {
                    if (!std::isfinite(f)) { throw std::runtime_error(format("imatrix contains non-finite value %f\n", f)); }
                }
            }
        }
    }
    if (params->activations) {
        activations_data = static_cast<const std::unordered_map<std::string, std::vector<float>>*>(params->activations);
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
        statistics_data = static_cast<const std::unordered_map<std::string, std::vector<float>>*>(params->statistics);
        if (statistics_data) { LLAMA_LOG_INFO(" and %d statistics", (int)statistics_data->size());
        }
    }
    LLAMA_LOG_INFO("\n");

    const size_t align = GGUF_DEFAULT_ALIGNMENT;
    gguf_context_ptr ctx_out { gguf_init_empty() };

    std::vector<int> prune_list = {};
    if (params->prune_layers) { prune_list = * static_cast<const std::vector<int> *>(params->prune_layers); }

    // copy the KV pairs from the input file
    gguf_set_kv     (ctx_out.get(), ml.metadata);
    gguf_set_val_u32(ctx_out.get(), "general.quantization_version", GGML_QNT_VERSION); // TODO: use LLM_KV
    gguf_set_val_u32(ctx_out.get(), "general.file_type", ftype); // TODO: use LLM_KV

    // Remove split metadata
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_NO).c_str());
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str());
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str());

    if (params->kv_overrides) {
        const std::vector<llama_model_kv_override> & overrides = *(const std::vector<llama_model_kv_override> *)params->kv_overrides;
        for (const auto & o : overrides) {
            if (o.key[0] == 0) break;
            if (o.tag == LLAMA_KV_OVERRIDE_TYPE_FLOAT) { gguf_set_val_f32(ctx_out.get(), o.key, o.val_f64); }
            else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_INT) { gguf_set_val_u32(ctx_out.get(), o.key, (uint32_t)std::abs(o.val_i64)); }
            else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_BOOL) { gguf_set_val_bool(ctx_out.get(), o.key, o.val_bool); }
            else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_STR) { gguf_set_val_str(ctx_out.get(), o.key, o.val_str); }
            else { LLAMA_LOG_WARN("%s: unknown KV override type for key %s\n", __func__, o.key); }
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

    int idx = 0;
    uint16_t n_split = 1;

    // Assume split index is continuous
    if (params->keep_split) {
        for (const auto * it : tensors) { n_split = std::max((uint16_t)(it->idx + 1), n_split); }
    }
    std::vector<gguf_context_ptr> ctx_outs(n_split);
    ctx_outs[0] = std::move(ctx_out);

    // compute tensor metadata once and cache it
    std::vector<tensor_metadata> metadata(tensors.size());

    // initialize quantization state before preliminary loop (counters for use_more_bits)
    {
        for (size_t i = 0; i < tensors.size(); ++i) {
            const auto cat = tensor_get_category(tensors[i]->tensor->name);
            if (category_is_attn_v(cat)) { ++qs.n_attention_wv; }
            if (cat == TENSOR_CATEGORY_OUTPUT) { qs.has_tied_embeddings = false; }
            metadata[i].category = cat; // save and re-use the category while we're at it
        }
        // these also need to be set to n_layer by default
        qs.n_ffn_down = qs.n_ffn_gate = qs.n_ffn_up = (int)qs.model.hparams.n_layer;
    }

    // flag for --dry-run
    bool will_require_imatrix = false;

    // preliminary iteration over all weights
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto * it = tensors[i];
        const ggml_tensor * tensor = it->tensor;
        const std::string name = ggml_get_name(tensor);

        uint16_t i_split = params->keep_split ? it->idx : 0;
        if (!ctx_outs[i_split]) { ctx_outs[i_split].reset(gguf_init_empty()); }
        gguf_add_tensor(ctx_outs[i_split].get(), tensor);

        metadata[i].allows_quantization = tensor_allows_quantization(params, model.arch, tensor);

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
                                name.c_str(), ggml_type_name(metadata[i].target_type));
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
            if (params->tensor_types || params->pure) {
                LLAMA_LOG_WARN("%s: --target-bpw/--target-size specified, ignoring all other type overrides\n", __func__);
            }
            if (params->activations) {
                LLAMA_LOG_INFO("%s: imatrix has activations, process will be more accurate\n", __func__);
            } else {
                LLAMA_LOG_INFO("%s: imatrix does not have activations, process may be less accurate\n", __func__);
            }
            if (params->statistics) {
                LLAMA_LOG_INFO("%s: imatrix has statistics\n", __func__);
            }
            if (params->importance_pct != 0.0f) {
                LLAMA_LOG_INFO("%s: marking up to %.2f%% of tensors as important\n", __func__, params->importance_pct);
            }
            if (params->target_size >= 0) {
                LLAMA_LOG_INFO("%s: computing tensor quantization mix to achieve file size %.2f MiB\n", __func__, (double)params->target_size / 1024.0 / 1024.0);
            } else {
                LLAMA_LOG_INFO("%s: computing tensor quantization mix to achieve %.4f bpw\n", __func__, params->target_bpw);
            }

            // get quantization type overrides targeting a given bits per weight budget
            bpw_overrides = target_bpw_type(ml, qs, tensors, mapped, values_data, activations_data, statistics_data, nthread);
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
    if (!params->dry_run) {
        new_ofstream(0);
    }

    // main loop: iterate over all weights
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto & weight = *tensors[i];
        const auto & tm = metadata[i];
        ggml_tensor * tensor = weight.tensor;

        if (!params->dry_run && (weight.idx != cur_split && params->keep_split)) {
            close_ofstream();
            new_ofstream(weight.idx);
        }

        const std::string name = ggml_get_name(tensor);
        const size_t tensor_size = ggml_nbytes(tensor);

        if (!params->dry_run) {
            if (!ml.use_mmap) {
                if (read_data.size() < tensor_size) {
                    read_data.resize(tensor_size);
                }
                tensor->data = read_data.data();
            }
            ml.load_data_for(tensor);
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
                        LLAMA_LOG_INFO("\n====== %s: did not find weights for %s\n", __func__, tensor->name);
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
            gguf_set_tensor_type(ctx_outs[cur_split].get(), name.c_str(), new_type);
            GGML_ASSERT(gguf_get_tensor_size(ctx_outs[cur_split].get(), gguf_find_tensor(ctx_outs[cur_split].get(), name.c_str())) == new_size);
            gguf_set_tensor_data(ctx_outs[cur_split].get(), name.c_str(), new_data);

            // write tensor data + padding
            fout.write((const char *) new_data, new_size);
            zeros(fout, GGML_PAD(new_size, align) - new_size);
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
        /*.ftype                       =*/ LLAMA_FTYPE_MOSTLY_Q5_1,
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
        /*.save_state                  =*/ false,
        /*.state_file                  =*/ nullptr,
        /*.importance_pct              =*/ 0.0f
    };

    return result;
}

uint32_t llama_model_quantize(
        const char * fname_inp,
        const char * fname_out,
        const llama_model_quantize_params * params) {
    try { llama_model_quantize_impl(fname_inp, fname_out, params); }
    catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to quantize: %s\n", __func__, err.what());
        return 1;
    }

    return 0;
}
