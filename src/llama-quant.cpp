#include "llama-quant.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-model-loader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cinttypes>
#include <fstream>
#include <mutex>
#include <regex>
#include <thread>
#include <memory>
#include <atomic>
#include <unordered_map>

// Quantization types. Changes to this struct must be replicated in quantize.cpp
struct tensor_quantization {
    std::string name;
    ggml_type quant = GGML_TYPE_COUNT;
};

struct quantization_state_impl {
    const llama_model                 & model;
    const llama_model_quantize_params * params;

    // per-model
    int32_t n_attention_wv = 0;
    int32_t n_ffn_down     = 0;
    int32_t n_ffn_gate     = 0;
    int32_t n_ffn_up       = 0;

    // per-layer
    int32_t i_attention_wv = 0;
    int32_t i_ffn_down     = 0;
    int32_t i_ffn_gate     = 0;
    int32_t i_ffn_up       = 0;

    // per-quantization
    int32_t n_fallback     = 0;

    // flags
    bool has_imatrix = false; // do we have imatrix data?
    bool has_output  = false; // used to figure out if a model shares tok_embd with the output weight

    // tensor type override patterns
    std::vector<std::pair<std::regex, ggml_type>> tensor_type_patterns;

    quantization_state_impl(const llama_model & model, const llama_model_quantize_params * params)
        : model(model)
        , params(params)
    {
        // compile regex patterns once - they are expensive, and used twice
        if (params->tensor_types) {
            const auto & tensor_types =
                *static_cast<const std::vector<tensor_quantization> *>(params->tensor_types);
            for (const auto & [tname, qtype] : tensor_types) {
                tensor_type_patterns.emplace_back(std::regex(tname), qtype);
            }
        }
    }
};

static void zeros(std::ofstream & file, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
}

static std::string remap_layer(const std::string & orig_name, const std::vector<int> & prune, std::map<int, std::string> & mapped, int & next_id) {
    if (prune.empty()) {
        return orig_name;
    }

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
    if (mapped.empty()) {
        return orig_name;
    }

    static const std::regex pattern(R"(blk\.(\d+)\.)");
    if (std::smatch match; std::regex_search(orig_name, match, pattern)) {
        const std::string blk(match[1]);
        std::string new_name = orig_name;

        for (const auto & p : mapped) {
            if (p.second == blk) {
                LLAMA_LOG_DEBUG("(blk.%d imatrix) ", p.first);
                return new_name.replace(match.position(1), match.length(1), std::to_string(p.first));
            }
        }
        GGML_ABORT("\n%s: imatrix mapping error for %s\n", __func__, orig_name.c_str());
    }

    return orig_name;
}

static void llama_tensor_dequantize_impl(
    ggml_tensor * tensor, std::vector<no_init<float>> & output, std::vector<std::thread> & workers,
    const size_t nelements, const int nthread
) {
    if (output.size() < nelements) {
        output.resize(nelements);
    }
    float * f32_output = (float *) output.data();

    const ggml_type_traits * qtype = ggml_get_type_traits(tensor->type);
    if (ggml_is_quantized(tensor->type)) {
        if (qtype->to_float == NULL) {
            throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available", ggml_type_name(tensor->type)));
        }
    } else if (tensor->type != GGML_TYPE_F16 &&
               tensor->type != GGML_TYPE_BF16) {
        throw std::runtime_error(format("cannot dequantize/convert tensor type %s", ggml_type_name(tensor->type)));
    }

    if (nthread < 2) {
        if (tensor->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((ggml_fp16_t *)tensor->data, f32_output, nelements);
        } else if (tensor->type == GGML_TYPE_BF16) {
            ggml_bf16_to_fp32_row((ggml_bf16_t *)tensor->data, f32_output, nelements);
        } else if (ggml_is_quantized(tensor->type)) {
            qtype->to_float(tensor->data, f32_output, nelements);
        } else {
            GGML_ABORT("fatal error"); // unreachable
        }
        return;
    }

    size_t block_size;
    if (tensor->type == GGML_TYPE_F16 ||
        tensor->type == GGML_TYPE_BF16) {
        block_size = 1;
    } else {
        block_size = (size_t)ggml_blck_size(tensor->type);
    }

    size_t block_size_bytes = ggml_type_size(tensor->type);

    GGML_ASSERT(nelements % block_size == 0);
    size_t nblocks = nelements / block_size;
    size_t blocks_per_thread = nblocks / nthread;
    size_t spare_blocks = nblocks - (blocks_per_thread * nthread); // if blocks aren't divisible by thread count

    size_t in_buff_offs = 0;
    size_t out_buff_offs = 0;

    for (int tnum = 0; tnum < nthread; tnum++) {
        size_t thr_blocks = blocks_per_thread + (tnum == nthread - 1 ? spare_blocks : 0); // num blocks for this thread
        size_t thr_elems = thr_blocks * block_size; // number of elements for this thread
        size_t thr_block_bytes = thr_blocks * block_size_bytes; // number of input bytes for this thread

        auto compute = [qtype] (ggml_type typ, uint8_t * inbuf, float * outbuf, int nels) {
            if (typ == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((ggml_fp16_t *)inbuf, outbuf, nels);
            } else if (typ == GGML_TYPE_BF16) {
                ggml_bf16_to_fp32_row((ggml_bf16_t *)inbuf, outbuf, nels);
            } else {
                qtype->to_float(inbuf, outbuf, nels);
            }
        };
        workers.emplace_back(compute, tensor->type, (uint8_t *) tensor->data + in_buff_offs, f32_output + out_buff_offs, thr_elems);
        in_buff_offs += thr_block_bytes;
        out_buff_offs += thr_elems;
    }
    for (auto & w : workers) { w.join(); }
    workers.clear();
}

static bool tensor_name_match_token_embd(const char * tensor_name) {
    return (
        std::strcmp(tensor_name, "token_embd.weight") == 0 ||
        std::strcmp(tensor_name, "per_layer_token_embd.weight") == 0
    );
}

// do we allow this tensor to be quantized?
static bool tensor_allows_quantization(const llama_model_quantize_params * params, llm_arch arch, const ggml_tensor * tensor) {
    const std::string name = tensor->name;

    // This used to be a regex, but <regex> has an extreme cost to compile times.
    bool allowed = name.rfind("weight") == name.size() - 6; // ends with 'weight'?

    // quantize only 2D and 3D tensors (experts)
    allowed &= (ggml_n_dims(tensor) >= 2);

    // do not quantize norm tensors
    allowed &= name.find("_norm.weight") == std::string::npos;

    allowed &= params->quantize_output_tensor || name != "output.weight";
    allowed &= !params->only_copy;

    // do not quantize expert gating tensors
    // NOTE: can't use LLM_TN here because the layer number is not known
    allowed &= name.find("ffn_gate_inp.weight") == std::string::npos;

    // these are very small (e.g. 4x4)
    allowed &= name.find("altup")  == std::string::npos;
    allowed &= name.find("laurel") == std::string::npos;

    // these are not too big so keep them as it is
    allowed &= name.find("per_layer_model_proj") == std::string::npos;

    // do not quantize positional embeddings and token types (BERT)
    allowed &= name != LLM_TN(arch)(LLM_TENSOR_POS_EMBD,    "weight");
    allowed &= name != LLM_TN(arch)(LLM_TENSOR_TOKEN_TYPES, "weight");

    // do not quantize Mamba /Kimi's small conv1d weights
    // NOTE: can't use LLM_TN here because the layer number is not known
    allowed &= name.find("ssm_conv1d") == std::string::npos;
    allowed &= name.find("shortconv.conv.weight") == std::string::npos;

    // do not quantize RWKV's small yet 2D weights
    allowed &= name.find("time_mix_first.weight") == std::string::npos;
    allowed &= name.find("time_mix_w0.weight") == std::string::npos;
    allowed &= name.find("time_mix_w1.weight") == std::string::npos;
    allowed &= name.find("time_mix_w2.weight") == std::string::npos;
    allowed &= name.find("time_mix_v0.weight") == std::string::npos;
    allowed &= name.find("time_mix_v1.weight") == std::string::npos;
    allowed &= name.find("time_mix_v2.weight") == std::string::npos;
    allowed &= name.find("time_mix_a0.weight") == std::string::npos;
    allowed &= name.find("time_mix_a1.weight") == std::string::npos;
    allowed &= name.find("time_mix_a2.weight") == std::string::npos;
    allowed &= name.find("time_mix_g1.weight") == std::string::npos;
    allowed &= name.find("time_mix_g2.weight") == std::string::npos;
    allowed &= name.find("time_mix_decay_w1.weight") == std::string::npos;
    allowed &= name.find("time_mix_decay_w2.weight") == std::string::npos;
    allowed &= name.find("time_mix_lerp_fused.weight") == std::string::npos;

    // do not quantize relative position bias (T5)
    allowed &= name.find("attn_rel_b.weight") == std::string::npos;

    // do not quantize specific multimodal tensors
    allowed &= name.find(".position_embd.") == std::string::npos;

    return allowed;
}

// incompatible tensor shapes are handled here - fallback to a compatible type
static ggml_type tensor_type_fallback(quantization_state_impl * qs, const ggml_tensor * t, const ggml_type target_type) {

    ggml_type return_type = target_type;

    const int64_t ncols = t->ne[0];
    const int64_t qk_k = ggml_blck_size(target_type);

    if (ncols % qk_k != 0) { // this tensor's shape is incompatible with this quant
        LLAMA_LOG_WARN("warning: %36s: ncols %6" PRId64 " not divisible by %3" PRId64 " (required for type %7s) ",
                        t->name, ncols, qk_k, ggml_type_name(target_type));
        ++qs->n_fallback;

        switch (target_type) {
            // types on the left: block size 256
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ1_M:
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ2_S:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_IQ3_S:   // types on the right: block size 32
            case GGML_TYPE_IQ4_XS:  return_type = GGML_TYPE_IQ4_NL; break;
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_TQ1_0:
            case GGML_TYPE_TQ2_0:   return_type = GGML_TYPE_Q4_0;   break;
            case GGML_TYPE_Q4_K:    return_type = GGML_TYPE_Q5_0;   break;
            case GGML_TYPE_Q5_K:    return_type = GGML_TYPE_Q5_1;   break;
            case GGML_TYPE_Q6_K:    return_type = GGML_TYPE_Q8_0;   break;
            default:
                throw std::runtime_error(format(
                    "no tensor type fallback is defined for type %s",
                    ggml_type_name(target_type)));
        }
        if (ncols % ggml_blck_size(return_type) != 0) {
            //
            // the fallback return type is still not compatible for this tensor!
            //
            // most likely, this tensor's first dimension is not divisible by 32.
            // this is very rare. we can either abort the quantization, or
            // fallback to F16 / F32.
            //
            LLAMA_LOG_WARN("(WARNING: falling back to F16 due to unusual shape) ");
            return_type = GGML_TYPE_F16;
        }
        LLAMA_LOG_WARN("-> falling back to %7s\n", ggml_type_name(return_type));
    }
    return return_type;
}

// internal standard logic for selecting the target tensor type for a given tensor, ftype, and model arch
static ggml_type llama_tensor_get_type_impl(
    quantization_state_impl * qs,
                  ggml_type   new_type,
          const ggml_tensor * tensor,
                llama_ftype   ftype
) {
    const std::string name = ggml_get_name(tensor);

    // TODO: avoid hardcoded tensor names - use the TN_* constants
    const llm_arch arch = qs->model.arch;
    const auto       tn = LLM_TN(arch);

    auto use_more_bits = [](int i_layer, int n_layers) -> bool {
        return i_layer < n_layers/8 || i_layer >= 7*n_layers/8 || (i_layer - n_layers/8)%3 == 2;
    };
    const int n_expert = std::max(1, (int)qs->model.hparams.n_expert);
    auto layer_info = [n_expert] (int i_layer, int n_layer, const char * name) {
        if (n_expert > 1) {
            // Believe it or not, "experts" in the FFN of Mixtral-8x7B are not consecutive, but occasionally randomly
            // sprinkled in the model. Hence, simply dividing i_ffn_down by n_expert does not work
            // for getting the current layer as I initially thought, and we need to resort to parsing the
            // tensor name.
            if (sscanf(name, "blk.%d.", &i_layer) != 1) {
                throw std::runtime_error(format("Failed to determine layer for tensor %s", name));
            }
            if (i_layer < 0 || i_layer >= n_layer) {
                throw std::runtime_error(format("Bad layer %d for tensor %s. Must be in [0, %d)", i_layer, name, n_layer));
            }
        }
        return std::make_pair(i_layer, n_layer);
    };

    // for arches that share the same tensor between the token embeddings and the output, we quantize the token embeddings
    // with the quantization of the output tensor
    if (name == tn(LLM_TENSOR_OUTPUT, "weight") || (!qs->has_output && name == tn(LLM_TENSOR_TOKEN_EMBD, "weight"))) {
        if (qs->params->output_tensor_type < GGML_TYPE_COUNT) {
            new_type = qs->params->output_tensor_type;
        } else {
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
        // MoE   tensors -> MXFP4
        // other tensors -> Q8_0
        if (tensor->ne[2] > 1) {
            new_type = GGML_TYPE_MXFP4;
        } else {
            new_type = GGML_TYPE_Q8_0;
        }
    } else if (name == "token_embd.weight" || name == "per_layer_token_embd.weight") {
        if (qs->params->token_embedding_type < GGML_TYPE_COUNT) {
            new_type = qs->params->token_embedding_type;
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
        if (name.find("attn_v.weight") != std::string::npos) {
            if (qs->model.hparams.n_gqa() >= 4 || qs->model.hparams.n_expert >= 4) new_type = GGML_TYPE_Q4_K;
            else new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            ++qs->i_attention_wv;
        }
        else if (qs->model.hparams.n_expert == 8 && name.find("attn_k.weight") != std::string::npos) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (name.find("ffn_down") != std::string::npos) {
            if (qs->i_ffn_down < qs->n_ffn_down/8) {
                new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            }
            ++qs->i_ffn_down;
        }
        else if (name.find("attn_output.weight") != std::string::npos) {
            if (qs->model.hparams.n_expert == 8) {
                new_type = GGML_TYPE_Q5_K;
            } else {
                if (ftype == LLAMA_FTYPE_MOSTLY_IQ1_S || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) new_type = GGML_TYPE_IQ2_XXS;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M) new_type = GGML_TYPE_IQ3_S;
            }
        }
    } else if (name.find("attn_v.weight") != std::string::npos) {
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) {
            new_type = qs->model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S && qs->model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = qs->model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : !qs->has_imatrix ? GGML_TYPE_IQ3_S : GGML_TYPE_IQ3_XXS;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S) && qs->model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = qs->i_attention_wv < 2 ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q5_K;
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && qs->model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) &&
                use_more_bits(qs->i_attention_wv, qs->n_attention_wv)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && qs->i_attention_wv < 4) new_type = GGML_TYPE_Q5_K;
        if (qs->model.type == LLM_TYPE_70B) {
            // In the 70B model we have 8 heads sharing the same attn_v weights. As a result, the attn_v.weight tensor is
            // 8x smaller compared to attn_q.weight. Hence, we can get a nice boost in quantization accuracy with
            // nearly negligible increase in model size by quantizing this tensor with more bits:
            if (new_type == GGML_TYPE_Q3_K || new_type == GGML_TYPE_Q4_K) new_type = GGML_TYPE_Q5_K;
        }
        if (qs->model.hparams.n_expert == 8) {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        ++qs->i_attention_wv;
    } else if (name.find("attn_k.weight") != std::string::npos) {
        if (qs->model.hparams.n_expert == 8) {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = GGML_TYPE_IQ2_S;
        }
    } else if (name.find("attn_q.weight") != std::string::npos) {
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = GGML_TYPE_IQ2_S;
        }
    } else if (name.find("ffn_down") != std::string::npos) {
        auto info = layer_info(qs->i_ffn_down, qs->n_ffn_down, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S) {
            if (i_layer < n_layer/8) new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS && !qs->has_imatrix) {
            new_type = i_layer < n_layer/8 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = i_layer < n_layer/16 ? GGML_TYPE_Q5_K
                     : arch != LLM_ARCH_FALCON || use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q4_K
                     : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M && (i_layer < n_layer/8 ||
                    (qs->model.hparams.n_expert == 8 && use_more_bits(i_layer, n_layer)))) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) {
            new_type = arch == LLM_ARCH_FALCON ? GGML_TYPE_Q4_K : GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) {
            if (arch == LLM_ARCH_FALCON) {
                new_type = i_layer < n_layer/16 ? GGML_TYPE_Q6_K :
                           use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
            } else {
                if (use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
            }
        }
        else if (i_layer < n_layer/8 && (ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && !qs->has_imatrix) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M && use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && arch != LLM_ARCH_FALCON && i_layer < n_layer/8) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_0 || ftype == LLAMA_FTYPE_MOSTLY_Q5_0)
                && qs->has_imatrix && i_layer < n_layer/8) {
            // Guard against craziness in the first few ffn_down layers that can happen even with imatrix for Q4_0/Q5_0.
            // We only do it when an imatrix is provided because a) we want to make sure that one can always get the
            // same quantization as before imatrix stuff, and b) Q4_1/Q5_1 do go crazy on ffn_down without an imatrix.
            new_type = ftype == LLAMA_FTYPE_MOSTLY_Q4_0 ? GGML_TYPE_Q4_1 : GGML_TYPE_Q5_1;
        }
        ++qs->i_ffn_down;
    } else if (name.find("attn_output.weight") != std::string::npos) {
        if (arch != LLM_ARCH_FALCON) {
            if (qs->model.hparams.n_expert == 8) {
                if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q3_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL  ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S  ||
                    ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) {
                    new_type = GGML_TYPE_Q5_K;
                }
            } else {
                if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   ) new_type = GGML_TYPE_Q3_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) new_type = GGML_TYPE_IQ3_S;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M ) new_type = GGML_TYPE_Q4_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L ) new_type = GGML_TYPE_Q5_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  ) new_type = GGML_TYPE_Q4_K;
            }
        } else {
            if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q4_K;
        }
    }
    else if (name.find("attn_qkv.weight") != std::string::npos) {
        if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L || ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) new_type = GGML_TYPE_Q5_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) new_type = GGML_TYPE_Q6_K;
    }
    else if (name.find("ffn_gate") != std::string::npos) {
        auto info = layer_info(qs->i_ffn_gate, qs->n_ffn_gate, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        ++qs->i_ffn_gate;
    }
    else if (name.find("ffn_up") != std::string::npos) {
        auto info = layer_info(qs->i_ffn_up, qs->n_ffn_up, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        ++qs->i_ffn_up;
    }

    return new_type;
}

// determine the ggml_type that this tensor should be quantized to
static ggml_type llama_tensor_get_type(
              quantization_state_impl * qs,
    const llama_model_quantize_params * params,
                    const ggml_tensor * tensor,
                      const ggml_type   default_type
) {
    if (!tensor_allows_quantization(params, qs->model.arch, tensor)) {
        return tensor->type; // if quantization not allowed, just return the current type
    }

    if (params->token_embedding_type < GGML_TYPE_COUNT && tensor_name_match_token_embd(tensor->name)) {
        return params->token_embedding_type;
    }
    if (params->output_tensor_type < GGML_TYPE_COUNT && strcmp(tensor->name, "output.weight") == 0) {
        return params->output_tensor_type;
    }

    ggml_type new_type = default_type;

    // get more optimal quantization type based on the tensor shape, layer, etc.
    if (!params->pure && ggml_is_quantized(default_type)) {

        // if the user provided tensor types - use those
        bool manual = false;
        if (!qs->tensor_type_patterns.empty()) {
            const std::string tensor_name(tensor->name);
            for (const auto & [pattern, qtype] : qs->tensor_type_patterns) {
                if (std::regex_search(tensor_name, pattern)) {
                    if (qtype != new_type) {
                        new_type = qtype;
                        manual = true;
                        break;
                    }
                }
            }
        }

        // if not manual - use the internal logic for choosing the quantization type
        if (!manual) {
            new_type = llama_tensor_get_type_impl(qs, new_type, tensor, params->ftype);
        }

        // fallback to a compatible type if necessary
        new_type = tensor_type_fallback(qs, tensor, new_type);
    }
    return new_type;
}

static size_t llama_tensor_quantize_impl(enum ggml_type new_type, const float * f32_data, void * new_data, const int64_t chunk_size, int64_t nrows, int64_t n_per_row, const float * imatrix, std::vector<std::thread> & workers, const int nthread) {
    if (nthread < 2) {
        // single-thread
        size_t new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, nrows, n_per_row, imatrix);
        if (!ggml_validate_row_data(new_type, new_data, new_size)) {
            throw std::runtime_error("quantized data validation failed");
        }
        return new_size;
    }

    std::mutex mutex;
    int64_t counter = 0;
    size_t new_size = 0;
    bool valid = true;
    auto compute = [&mutex, &counter, &new_size, &valid, new_type, f32_data, new_data, chunk_size,
            nrows, n_per_row, imatrix]() {
        const int64_t nrows_per_chunk = chunk_size / n_per_row;
        size_t local_size = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int64_t first_row = counter; counter += nrows_per_chunk;
            if (first_row >= nrows) {
                if (local_size > 0) {
                    new_size += local_size;
                }
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
    for (int it = 0; it < nthread - 1; ++it) {
        workers.emplace_back(compute);
    }
    compute();
    for (auto & w : workers) { w.join(); }
    workers.clear();
    if (!valid) {
        throw std::runtime_error("quantized data validation failed");
    }
    return new_size;
}

// quantize a single tensor, handling expert parallelism and work buffer management
static size_t llama_tensor_quantize(
                const ggml_tensor * tensor,
                        ggml_type   new_type,
                      const float * f32_data,
                      const float * imatrix,
    std::vector<no_init<uint8_t>> & work,
         std::vector<std::thread> & workers,
                              int   nthread
) {
    const int64_t nelements = ggml_nelements(tensor);

    if (work.size() < (size_t)nelements * 4) {
        work.resize(nelements * 4);
    }

    void * new_data = work.data();

    const int64_t n_per_row = tensor->ne[0];
    const int64_t nrows     = tensor->ne[1];

    static const int64_t min_chunk_size = 32 * 512;
    const int64_t chunk_size = (n_per_row >= min_chunk_size
        ? n_per_row
        : n_per_row * ((min_chunk_size + n_per_row - 1)/n_per_row));

    const int64_t ne_0_x_1  = tensor->ne[0] * tensor->ne[1];
    const int64_t n_experts = tensor->ne[2];
    const int64_t nchunk = (ne_0_x_1 + chunk_size - 1) / chunk_size;

    // enough chunks to feed all threads?
    const bool expert_parallel = n_experts >= nthread && nthread > 1 && nchunk < nthread;

    // parallelize across experts instead of within them, if it would be more efficient
    //
    // this launches threads once rather than n_experts times,
    // which avoids massive thread creation overhead for
    // MoE models (which can have 256+ experts)
    if (expert_parallel) {
        std::mutex mutex;
        std::atomic<int64_t> next_expert{0};
        size_t new_size = 0;
        bool valid = true;

        auto compute = [&]() {
            size_t local_size = 0;
            while (true) {
                // grab the next available expert
                const int64_t expert = next_expert.fetch_add(1);
                if (expert >= n_experts) {
                    break;
                }

                const float * f32_data_expert = f32_data + expert * ne_0_x_1;
                void        * new_data_expert = (char *)new_data + ggml_row_size(new_type, n_per_row)
                                                                 * expert * nrows;
                const float * imatrix_expert  = imatrix ? imatrix + expert * n_per_row : nullptr;

                // quantize this entire expert single-threaded
                size_t expert_size = ggml_quantize_chunk(
                    new_type, f32_data_expert, new_data_expert,
                    0, nrows, n_per_row, imatrix_expert);

                // validate
                if (!ggml_validate_row_data(new_type, new_data_expert, expert_size)) {
                    std::lock_guard<std::mutex> lock(mutex);
                    valid = false;
                    break;
                }

                local_size += expert_size;
            }

            // accumulate local result
            std::lock_guard<std::mutex> lock(mutex);
            new_size += local_size;
        };

        for (int i = 0; i < nthread - 1; ++i) {
            workers.emplace_back(compute);
        }
        compute(); // main thread participates too // TODO: is this optimal?
        for (auto & w : workers) {
            w.join();
        }
        workers.clear();
        if (!valid) {
            throw std::runtime_error("quantized data validation failed");
        }

        return new_size;
    } else {
        // parallelize within each expert (original behaviour)
        const int64_t nthread_use = nthread > 1
            ? std::max((int64_t)1, std::min((int64_t)nthread, nchunk))
            : 1;

        size_t new_size = 0;
        for (int64_t i03 = 0; i03 < n_experts; ++i03) {
            const float * imatrix_03  = imatrix ? imatrix + i03 * n_per_row : nullptr;
            const float * f32_data_03 = f32_data + i03 * ne_0_x_1;
            void        * new_data_03 = (char *)new_data + ggml_row_size(new_type, n_per_row)
                                                         * i03 * nrows;

            new_size += llama_tensor_quantize_impl(
                new_type, f32_data_03, new_data_03, chunk_size,
                nrows, n_per_row, imatrix_03, workers, nthread_use);
        }

        return new_size;
    }
}

// does this tensor require importance matrix data?
static bool tensor_requires_imatrix(const char * tensor_name, const ggml_type dst_type, const llama_ftype ftype) {
    // token embedding tensors should never require imatrix data
    if (tensor_name_match_token_embd(tensor_name)) {
        return false;
    }
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
    ggml_type return_type;
    switch (ftype) {
        // floating-point types
        case LLAMA_FTYPE_ALL_F32:     return_type = GGML_TYPE_F32;  break;
        case LLAMA_FTYPE_MOSTLY_F16:  return_type = GGML_TYPE_F16;  break;
        case LLAMA_FTYPE_MOSTLY_BF16: return_type = GGML_TYPE_BF16; break;

        // static quants
        case LLAMA_FTYPE_MOSTLY_Q4_0: return_type = GGML_TYPE_Q4_0; break;
        case LLAMA_FTYPE_MOSTLY_Q4_1: return_type = GGML_TYPE_Q4_1; break;
        case LLAMA_FTYPE_MOSTLY_Q5_0: return_type = GGML_TYPE_Q5_0; break;
        case LLAMA_FTYPE_MOSTLY_Q5_1: return_type = GGML_TYPE_Q5_1; break;
        case LLAMA_FTYPE_MOSTLY_Q8_0: return_type = GGML_TYPE_Q8_0; break;

        // k-quants
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:
        case LLAMA_FTYPE_MOSTLY_Q2_K:   return_type = GGML_TYPE_Q2_K; break;
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:
        case LLAMA_FTYPE_MOSTLY_Q3_K_L: return_type = GGML_TYPE_Q3_K; break;
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:
        case LLAMA_FTYPE_MOSTLY_Q4_K_M: return_type = GGML_TYPE_Q4_K; break;
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:
        case LLAMA_FTYPE_MOSTLY_Q5_K_M: return_type = GGML_TYPE_Q5_K; break;
        case LLAMA_FTYPE_MOSTLY_Q6_K:   return_type = GGML_TYPE_Q6_K; break;

        // i-quants
        case LLAMA_FTYPE_MOSTLY_IQ1_S:   return_type = GGML_TYPE_IQ1_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ1_M:   return_type = GGML_TYPE_IQ1_M;   break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS: return_type = GGML_TYPE_IQ2_XXS; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:  return_type = GGML_TYPE_IQ2_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ2_S:   return_type = GGML_TYPE_IQ2_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ2_M:   return_type = GGML_TYPE_IQ2_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS: return_type = GGML_TYPE_IQ3_XXS; break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:
        case LLAMA_FTYPE_MOSTLY_IQ3_S:
        case LLAMA_FTYPE_MOSTLY_IQ3_M:   return_type = GGML_TYPE_IQ3_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:  return_type = GGML_TYPE_IQ4_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:  return_type = GGML_TYPE_IQ4_NL;  break;

        // MXFP4
        case LLAMA_FTYPE_MOSTLY_MXFP4_MOE: return_type = GGML_TYPE_MXFP4; break;

        // ternary
        case LLAMA_FTYPE_MOSTLY_TQ1_0: return_type = GGML_TYPE_TQ1_0; break;
        case LLAMA_FTYPE_MOSTLY_TQ2_0: return_type = GGML_TYPE_TQ2_0; break;

        // otherwise, invalid
        default: throw std::runtime_error(format("invalid output file type %d\n", ftype));
    }
    return return_type;
}

static void llama_model_quantize_impl(const std::string & fname_inp, const std::string & fname_out, const llama_model_quantize_params * params) {
    ggml_type default_type;
    llama_ftype ftype = params->ftype;

    int nthread = params->nthread <= 0 ? std::thread::hardware_concurrency() : params->nthread;

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
    llama_model_loader ml(fname_inp, splits, use_mmap, /*use_direct_io*/ false, /*check_tensors*/ true, /*no_alloc*/ false, kv_overrides, nullptr);
    ml.init_mappings(false); // no prefetching

    llama_model model(llama_model_default_params());

    model.load_arch   (ml);
    model.load_hparams(ml);
    model.load_stats  (ml);

    // quantization state
    auto qs = std::make_unique<quantization_state_impl>(model, params);

    // these need to be set to n_layer by default
    qs->n_ffn_down = qs->n_ffn_gate = qs->n_ffn_up = (int)model.hparams.n_layer;

    if (params->only_copy) {
        ftype = ml.ftype;
    }

    const std::unordered_map<std::string, std::vector<float>> * imatrix_data = nullptr;
    if (params->imatrix) {
        imatrix_data = static_cast<const std::unordered_map<std::string, std::vector<float>>*>(params->imatrix);
        if (imatrix_data) {
            LLAMA_LOG_INFO("\n%s: have importance matrix data with %d entries\n",
                           __func__, (int)imatrix_data->size());
            // check imatrix for nans or infs
            for (const auto & kv : *imatrix_data) {
                for (float f : kv.second) {
                    if (!std::isfinite(f)) {
                        throw std::runtime_error(format("imatrix contains non-finite value %f\n", f));
                    }
                }
            }
            qs->has_imatrix = true;
        }
    }

    const size_t align = GGUF_DEFAULT_ALIGNMENT;
    gguf_context_ptr ctx_out { gguf_init_empty() };

    std::vector<int> prune_list = {};
    if (params->prune_layers) {
        prune_list = *static_cast<const std::vector<int> *>(params->prune_layers);
    }

    // copy the KV pairs from the input file
    gguf_set_kv     (ctx_out.get(), ml.meta.get());
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
            if (o.tag == LLAMA_KV_OVERRIDE_TYPE_FLOAT) {
                gguf_set_val_f32(ctx_out.get(), o.key, o.val_f64);
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_INT) {
                // Setting type to UINT32. See https://github.com/ggml-org/llama.cpp/pull/14182 for context
                gguf_set_val_u32(ctx_out.get(), o.key, (uint32_t)std::abs(o.val_i64));
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_BOOL) {
                gguf_set_val_bool(ctx_out.get(), o.key, o.val_bool);
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_STR) {
                gguf_set_val_str(ctx_out.get(), o.key, o.val_str);
            } else {
                LLAMA_LOG_WARN("%s: unknown KV override type for key '%s' (not applied)\n",
                               __func__, o.key);
            }
        }
    }

    std::map<int, std::string> mapped;
    int blk_id = 0;

    // make a list of weights
    std::vector<const llama_model_loader::llama_tensor_weight *> weights;
    weights.reserve(ml.weights_map.size());
    for (const auto & it : ml.weights_map) {
        const std::string remapped_name(remap_layer(it.first, prune_list, mapped, blk_id));
        if (remapped_name.empty()) {
            LLAMA_LOG_DEBUG("%s: pruning tensor %s\n", __func__, it.first.c_str());
            continue;
        }

        if (remapped_name != it.first) {
            ggml_set_name(it.second.tensor, remapped_name.c_str());
            LLAMA_LOG_DEBUG("%s: tensor %s remapped to %s\n",
                            __func__, it.first.c_str(), ggml_get_name(it.second.tensor));
        }
        weights.push_back(&it.second);
    }
    if (!prune_list.empty()) {
        gguf_set_val_u32(ctx_out.get(), ml.llm_kv(LLM_KV_BLOCK_COUNT).c_str(), blk_id);
    }

    // keep_split requires that the weights are sorted by split index
    if (params->keep_split) {
        std::sort(weights.begin(), weights.end(), [](
            const llama_model_loader::llama_tensor_weight * a,
            const llama_model_loader::llama_tensor_weight * b
        ) {
            if (a->idx == b->idx) {
                return a->offs < b->offs;
            }
            return a->idx < b->idx;
        });
    }

    std::vector<std::thread> workers;
    workers.reserve(nthread);

    int idx = 0;

    uint16_t n_split = 1;

    // Assume split index is continuous
    if (params->keep_split) {
        for (const auto * it : weights) {
            n_split = std::max(uint16_t(it->idx + 1), n_split);
        }
    }
    std::vector<gguf_context_ptr> ctx_outs(n_split);
    ctx_outs[0] = std::move(ctx_out);

    std::vector<ggml_type> target_types(weights.size());

    // flag for `--dry-run`, to let the user know if imatrix will be required
    // for a real quantization, as a courtesy
    bool will_require_imatrix = false;

    //
    // preliminary iteration over all weights (not the main loop)
    //

    for (size_t i = 0; i < weights.size(); ++i) {
        const auto * it = weights[i];
        const ggml_tensor * tensor = it->tensor;
        const std::string name = tensor->name;

        // TODO: avoid hardcoded tensor names - use the TN_* constants
        if (name.find("attn_v.weight")   != std::string::npos ||
            name.find("attn_qkv.weight") != std::string::npos ||
            name.find("attn_kv_b.weight")!= std::string::npos) {
            ++qs->n_attention_wv;
        }

        if (name == LLM_TN(model.arch)(LLM_TENSOR_OUTPUT, "weight")) {
            qs->has_output = true;
        }

        // populate the original weights so we get an initial meta data
        uint16_t i_split = params->keep_split ? it->idx : 0;
        if (!ctx_outs[i_split]) {
            ctx_outs[i_split].reset(gguf_init_empty());
        }
        gguf_add_tensor(ctx_outs[i_split].get(), tensor);

        target_types[i] = llama_tensor_get_type(qs.get(), params, tensor, default_type);

        if (!params->imatrix &&
            tensor_allows_quantization(params, model.arch, tensor) &&
            tensor_requires_imatrix(tensor->name, target_types[i], ftype)
        ) { // this tensor requires an imatrix but we don't have one!
            if (params->dry_run) {
                // set flag for warning later, but continue with dry run
                will_require_imatrix = true;
            } else {
                LLAMA_LOG_ERROR("\n============================================================================\n"
                                  " ERROR: this quantization requires an importance matrix!\n"
                                  "        - offending tensor: %s\n"
                                  "        - target type: %s\n"
                                  "============================================================================\n\n",
                                  name.c_str(), ggml_type_name(target_types[i]));
                throw std::runtime_error("this quantization requires an imatrix!");
            }
        }
    }

    // Set split info if needed
    if (n_split > 1) {
        for (size_t i = 0; i < ctx_outs.size(); ++i) {
            gguf_set_val_u16(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_NO).c_str(), i);
            gguf_set_val_u16(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str(), n_split);
            gguf_set_val_i32(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str(), (int32_t)weights.size());
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

    const auto tn = LLM_TN(model.arch);

    // no output file for --dry-run
    if (!params->dry_run) {
        new_ofstream(0);
    }

    // work buffers

    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<uint8_t>> work;
    std::vector<no_init<float>> f32_conv_buf;

    // pre-allocate work buffers to avoid repeated resizing
    {
        size_t max_tensor_bytes    = 0;
        size_t max_nelements       = 0;
        size_t max_nelements_dequant = 0;
        for (const auto * w : weights) {
            max_tensor_bytes = std::max(max_tensor_bytes, ggml_nbytes(w->tensor));
            max_nelements    = std::max(max_nelements,    (size_t)ggml_nelements(w->tensor));
            if (w->tensor->type != GGML_TYPE_F32) {
                max_nelements_dequant = std::max(max_nelements_dequant, (size_t)ggml_nelements(w->tensor));
            }
        }
        if (!ml.use_mmap) {
            read_data.resize(max_tensor_bytes);
        }
        work.resize(max_nelements * 4);
        if (max_nelements_dequant > 0) {
            f32_conv_buf.resize(max_nelements_dequant);
        }
    }

    size_t total_size_org = 0;
    size_t total_size_new = 0;

    //
    // iterate over all weights (main loop)
    //

    for (size_t i = 0; i < weights.size(); ++i) {
        const auto * it = weights[i];
        const auto & weight = *it;
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

        LLAMA_LOG_INFO("[%4d/%4d] %36s: [%s], type: %7s, ",
                       ++idx, ml.n_tensors,
                       ggml_get_name(tensor),
                       llama_format_tensor_shape(tensor).c_str(),
                       ggml_type_name(tensor->type));

        ggml_type const new_type = target_types[i];
        bool do_quantize = (new_type != tensor->type);

        void * new_data;
        size_t new_size;

        //
        // perform quantization (or dry run)
        //

        if (params->dry_run) {
            // the --dry-run option calculates the final quantization size without quantizing
            if (do_quantize) {
                new_size = ggml_nrows(tensor) * ggml_row_size(new_type, tensor->ne[0]);
                LLAMA_LOG_INFO("size: %8.2f MiB -> type: %7s, size: %8.2f MiB\n",
                               tensor_size/1024.0/1024.0,
                               ggml_type_name(new_type),
                               new_size/1024.0/1024.0);
                if (!will_require_imatrix && tensor_requires_imatrix(tensor->name, new_type, ftype)) {
                    will_require_imatrix = true;
                }
            } else {
                new_size = tensor_size;
                LLAMA_LOG_INFO("size: %8.2f MiB\n", new_size/1024.0/1024.0);
            }
            total_size_org += tensor_size;
            total_size_new += new_size;
            continue;
        } else {
            // no --dry-run, perform quantization
            if (!do_quantize) {
                GGML_ASSERT(tensor->type == new_type);
                new_data = tensor->data;
                new_size = tensor_size;
                LLAMA_LOG_INFO("size: %8.2f MiB\n", tensor_size/1024.0/1024.0);
            } else {
                const int64_t nelements = ggml_nelements(tensor);

                const float * imatrix = nullptr;
                if (imatrix_data) {
                    auto it = imatrix_data->find(remap_imatrix(tensor->name, mapped));
                    if (it == imatrix_data->end()) {
                        LLAMA_LOG_INFO("\n%s: did not find imatrix data for %s\n", __func__, tensor->name);
                    } else {
                        if (it->second.size() == (size_t)tensor->ne[0]*tensor->ne[2]) {
                            imatrix = it->second.data();
                        } else {
                            LLAMA_LOG_INFO("\n%s: imatrix size %d is different from tensor size %d for %s\n",
                                           __func__, int(it->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name);

                            // this can happen when quantizing an old mixtral model with split tensors with a new incompatible imatrix
                            // this is a significant error and it may be good idea to abort the process if this happens,
                            // since many people will miss the error and not realize that most of the model is being quantized without an imatrix
                            // tok_embd should be ignored in this case, since it always causes this warning
                            if (name != tn(LLM_TENSOR_TOKEN_EMBD, "weight")) {
                                throw std::runtime_error(format("imatrix size %d is different from tensor size %d for %s",
                                        int(it->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name));
                            }
                        }
                    }
                }
                if (!imatrix && tensor_requires_imatrix(tensor->name, new_type, ftype)) {
                    // we should never reach this anymore, since we now check for imatrix
                    // requirements in the preliminary loop over model weights, but this check
                    // is left here as a guard, just in case.
                    LLAMA_LOG_ERROR("\n============================================================\n");
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

                LLAMA_LOG_INFO("quantizing to %7s ... ", ggml_type_name(new_type));
                fflush(stdout);

                new_size = llama_tensor_quantize(tensor, new_type, f32_data, imatrix, work, workers, nthread);
                new_data = work.data();

                LLAMA_LOG_INFO("%8.2f MiB -> %8.2f MiB\n", tensor_size/1024.0/1024.0, new_size/1024.0/1024.0);
            } // do_quantize

            total_size_org += tensor_size;
            total_size_new += new_size;

            // update the gguf meta data as we go
            gguf_set_tensor_type(ctx_outs[cur_split].get(), name.c_str(), new_type);
            GGML_ASSERT(gguf_get_tensor_size(ctx_outs[cur_split].get(), gguf_find_tensor(ctx_outs[cur_split].get(), name.c_str())) == new_size);
            gguf_set_tensor_data(ctx_outs[cur_split].get(), name.c_str(), new_data);

            // write tensor data + padding
            fout.write((const char *) new_data, new_size);
            zeros(fout, GGML_PAD(new_size, align) - new_size);
        } // no --dry-run
    } // iterate over tensors

    if (!params->dry_run) {
        close_ofstream();
    }

    LLAMA_LOG_INFO("%s: model size  = %8.2f MiB (%.2f BPW)\n", __func__, total_size_org/1024.0/1024.0, total_size_org*8.0/ml.n_elements);
    LLAMA_LOG_INFO("%s: quant size  = %8.2f MiB (%.2f BPW)\n", __func__, total_size_new/1024.0/1024.0, total_size_new*8.0/ml.n_elements);

    if (!params->imatrix && params->dry_run && will_require_imatrix) {
        LLAMA_LOG_WARN("%s: WARNING: dry run completed successfully, but actually completing this quantization will require an imatrix!\n",
                       __func__
        );
    }

    if (qs->n_fallback > 0) {
        LLAMA_LOG_WARN("%s: WARNING: %d of %d tensor(s) required fallback quantization\n",
                       __func__, qs->n_fallback, ml.n_tensors);
    }
}

//
// interface implementation
//

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
        /*.kv_overrides                =*/ nullptr,
        /*.tensor_type                 =*/ nullptr,
        /*.prune_layers                =*/ nullptr
    };

    return result;
}

uint32_t llama_model_quantize(
        const char * fname_inp,
        const char * fname_out,
        const llama_model_quantize_params * params) {
    try {
        llama_model_quantize_impl(fname_inp, fname_out, params);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to quantize: %s\n", __func__, err.what());
        return 1;
    }

    return 0;
}
