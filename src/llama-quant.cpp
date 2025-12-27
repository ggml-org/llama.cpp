#include "llama-quant.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-model-loader.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <cinttypes>
#include <csignal>
#include <fstream>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <regex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

// Quantization types. Changes to this struct must be replicated in quantize.cpp
struct tensor_quantization {
    std::string name;
    ggml_type quant = GGML_TYPE_COUNT;
};

static bool is_quantizable(const std::string & name, const llm_arch arch, const llama_model_quantize_params * params) {
    if (params->only_copy) { return false; }

    const auto tn = LLM_TN(arch);

    // This used to be a regex, but <regex> has an extreme cost to compile times.
    bool q = name.size() >= 6 && name.rfind("weight") == name.size() - 6; // ends with 'weight'?

    // Do not quantize norm tensors
    q &= name.find("_norm.weight") == std::string::npos;

    // Do not quantize expert gating tensors
    // NOTE: can't use LLM_TN here because the layer number is not known
    q &= name.find("ffn_gate_inp.weight") == std::string::npos;

    // These are very small (e.g. 4x4)
    q &= name.find("altup") == std::string::npos;
    q &= name.find("laurel") == std::string::npos;

    // These are not too big so keep them as it is
    q &= name.find("per_layer_model_proj") == std::string::npos;

    // Do not quantize positional embeddings and token types (BERT)
    q &= name != tn(LLM_TENSOR_POS_EMBD, "weight");
    q &= name != tn(LLM_TENSOR_TOKEN_TYPES, "weight");

    // Do not quantize Jamba, Mamba, LFM2's small yet 2D weights
    // NOTE: can't use LLM_TN here because the layer number is not known
    q &= name.find("ssm_conv1d.weight") == std::string::npos;
    q &= name.find("shortconv.conv.weight") == std::string::npos;

    // Do not quantize ARWKV, RWKV's small yet 2D weights
    q &= name.find("time_mix_first.weight") == std::string::npos;
    q &= name.find("time_mix_w0.weight") == std::string::npos;
    q &= name.find("time_mix_w1.weight") == std::string::npos;
    q &= name.find("time_mix_w2.weight") == std::string::npos;
    q &= name.find("time_mix_v0.weight") == std::string::npos;
    q &= name.find("time_mix_v1.weight") == std::string::npos;
    q &= name.find("time_mix_v2.weight") == std::string::npos;
    q &= name.find("time_mix_a0.weight") == std::string::npos;
    q &= name.find("time_mix_a1.weight") == std::string::npos;
    q &= name.find("time_mix_a2.weight") == std::string::npos;
    q &= name.find("time_mix_g1.weight") == std::string::npos;
    q &= name.find("time_mix_g2.weight") == std::string::npos;
    q &= name.find("time_mix_decay_w1.weight") == std::string::npos;
    q &= name.find("time_mix_decay_w2.weight") == std::string::npos;
    q &= name.find("time_mix_lerp_fused.weight") == std::string::npos;

    // Do not quantize relative position bias (T5)
    q &= name.find("attn_rel_b.weight") == std::string::npos;

    return q;
}

static enum ggml_type fallback_type(const enum ggml_type new_type) {
    switch (new_type) {
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
            return GGML_TYPE_Q4_0; // symmetric-ish fallback
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_IQ4_XS:
            return GGML_TYPE_IQ4_NL;
        case GGML_TYPE_Q4_K:
            return GGML_TYPE_Q5_0;
        case GGML_TYPE_Q5_K:
            return GGML_TYPE_Q5_1;
        case GGML_TYPE_Q6_K:
            return GGML_TYPE_Q8_0;
        default:
            return new_type;
    }
}

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

static std::string remap_imatrix (const std::string & orig_name, const std::map<int, std::string> & mapped) {
    if (mapped.empty()) {
        return orig_name;
    }

    static const std::regex pattern(R"(blk\.(\d+)\.)");
    if (std::smatch match; std::regex_search(orig_name, match, pattern)) {
        const std::string blk(match[1]);
        std::string new_name = orig_name;

        for (const auto & p : mapped) {
            if (p.second == blk) {
                return new_name.replace(match.position(1), match.length(1), std::to_string(p.first));
            }
        }
        GGML_ABORT("\n%s: imatrix mapping error for %s\n", __func__, orig_name.c_str());
    }

    return orig_name;
}

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

    int n_k_quantized  = 0;
    int n_fallback     = 0;

    bool has_imatrix     = false;
    bool has_activations = false;

    // used to figure out if a model shares tok_embd with the output weight
    bool has_output = false;

    quantize_state_impl(const llama_model & model, const llama_model_quantize_params * params)
        : model(model)
        , params(params)
        {}
};

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

static ggml_type llama_tensor_get_type(quantize_state_impl & qs, ggml_type new_type, const ggml_tensor * tensor, llama_ftype ftype) {
    const std::string name = ggml_get_name(tensor);

    // TODO: avoid hardcoded tensor names - use the TN_* constants
    const llm_arch arch = qs.model.arch;
    const auto       tn = LLM_TN(arch);

    auto use_more_bits = [](int i_layer, int n_layers) -> bool {
        return i_layer < n_layers/8 || i_layer >= 7*n_layers/8 || (i_layer - n_layers/8)%3 == 2;
    };
    const int n_expert = std::max(1, (int)qs.model.hparams.n_expert);
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
    if (name == tn(LLM_TENSOR_OUTPUT, "weight") || (!qs.has_output && name == tn(LLM_TENSOR_TOKEN_EMBD, "weight"))) {
        if (qs.params->output_tensor_type < GGML_TYPE_COUNT) {
            new_type = qs.params->output_tensor_type;
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
        if (name.find("attn_v.weight") != std::string::npos) {
            if (qs.model.hparams.n_gqa() >= 4 || qs.model.hparams.n_expert >= 4) new_type = GGML_TYPE_Q4_K;
            else new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            ++qs.i_attention_wv;
        }
        else if (qs.model.hparams.n_expert == 8 && name.find("attn_k.weight") != std::string::npos) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (name.find("ffn_down") != std::string::npos) {
            if (qs.i_ffn_down < qs.n_ffn_down/8) {
                new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            }
            ++qs.i_ffn_down;
        }
        else if (name.find("attn_output.weight") != std::string::npos) {
            if (qs.model.hparams.n_expert == 8) {
                new_type = GGML_TYPE_Q5_K;
            } else {
                if (ftype == LLAMA_FTYPE_MOSTLY_IQ1_S || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) new_type = GGML_TYPE_IQ2_XXS;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M) new_type = GGML_TYPE_IQ3_S;
            }
        }
    } else if (name.find("attn_v.weight") != std::string::npos) {
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : !qs.has_imatrix ? GGML_TYPE_IQ3_S : GGML_TYPE_IQ3_XXS;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S) && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = qs.i_attention_wv < 2 ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q5_K;
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) &&
                use_more_bits(qs.i_attention_wv, qs.n_attention_wv)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && qs.i_attention_wv < 4) new_type = GGML_TYPE_Q5_K;
        if (qs.model.type == LLM_TYPE_70B) {
            // In the 70B model we have 8 heads sharing the same attn_v weights. As a result, the attn_v.weight tensor is
            // 8x smaller compared to attn_q.weight. Hence, we can get a nice boost in quantization accuracy with
            // nearly negligible increase in model size by quantizing this tensor with more bits:
            if (new_type == GGML_TYPE_Q3_K || new_type == GGML_TYPE_Q4_K) new_type = GGML_TYPE_Q5_K;
        }
        if (qs.model.hparams.n_expert == 8) {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        ++qs.i_attention_wv;
    } else if (name.find("attn_k.weight") != std::string::npos) {
        if (qs.model.hparams.n_expert == 8) {
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
        auto info = layer_info(qs.i_ffn_down, qs.n_ffn_down, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S) {
            if (i_layer < n_layer/8) new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS && !qs.has_imatrix) {
            new_type = i_layer < n_layer/8 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = i_layer < n_layer/16 ? GGML_TYPE_Q5_K
                     : arch != LLM_ARCH_FALCON || use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q4_K
                     : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M && (i_layer < n_layer/8 ||
                    (qs.model.hparams.n_expert == 8 && use_more_bits(i_layer, n_layer)))) {
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
        else if (i_layer < n_layer/8 && (ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && !qs.has_imatrix) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M && use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && arch != LLM_ARCH_FALCON && i_layer < n_layer/8) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_0 || ftype == LLAMA_FTYPE_MOSTLY_Q5_0)
                && qs.has_imatrix && i_layer < n_layer/8) {
            // Guard against craziness in the first few ffn_down layers that can happen even with imatrix for Q4_0/Q5_0.
            // We only do it when an imatrix is provided because a) we want to make sure that one can always get the
            // same quantization as before imatrix stuff, and b) Q4_1/Q5_1 do go crazy on ffn_down without an imatrix.
            new_type = ftype == LLAMA_FTYPE_MOSTLY_Q4_0 ? GGML_TYPE_Q4_1 : GGML_TYPE_Q5_1;
        }
        ++qs.i_ffn_down;
    } else if (name.find("attn_output.weight") != std::string::npos) {
        if (arch != LLM_ARCH_FALCON) {
            if (qs.model.hparams.n_expert == 8) {
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
        auto info = layer_info(qs.i_ffn_gate, qs.n_ffn_gate, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        ++qs.i_ffn_gate;
    }
    else if (name.find("ffn_up") != std::string::npos) {
        auto info = layer_info(qs.i_ffn_up, qs.n_ffn_up, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        ++qs.i_ffn_up;
    }

    //    if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
    //}
    // IK: let's remove this, else Q2_K is almost the same as Q3_K_S
    //else if (name.find("ffn_gate") != std::string::npos || name.find("ffn_up") != std::string::npos) {
    //    if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
    //}
    // This can be used to reduce the size of the Q5_K_S model.
    // The associated PPL increase is fully in line with the size reduction
    //else {
    //    if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_S) new_type = GGML_TYPE_Q4_K;
    //}
    bool convert_incompatible_tensor = false;
    {
        const int64_t nx = tensor->ne[0];
        const int64_t ny = tensor->ne[1];
        const int64_t qk_k = ggml_blck_size(new_type);

        if (nx % qk_k != 0) {
            LLAMA_LOG_WARN("\n\n%s : tensor cols %" PRId64 " x %" PRId64 " are not divisible by %" PRId64 ", required for %s", __func__, nx, ny, qk_k, ggml_type_name(new_type));
            convert_incompatible_tensor = true;
        } else {
            ++qs.n_k_quantized;
        }
    }

    if (convert_incompatible_tensor) {
        switch (new_type) {
            case GGML_TYPE_TQ1_0:
            case GGML_TYPE_TQ2_0:  new_type = GGML_TYPE_Q4_0; break;  // TODO: use a symmetric type instead
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ2_S:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_IQ3_S:
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ1_M:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_IQ4_XS: new_type = GGML_TYPE_IQ4_NL; break;
            case GGML_TYPE_Q4_K:   new_type = GGML_TYPE_Q5_0;   break;
            case GGML_TYPE_Q5_K:   new_type = GGML_TYPE_Q5_1;   break;
            case GGML_TYPE_Q6_K:   new_type = GGML_TYPE_Q8_0;   break;
            default: throw std::runtime_error("\nUnsupported tensor size encountered\n");
        }
        if (tensor->ne[0] % ggml_blck_size(new_type) != 0) {
            new_type = GGML_TYPE_F16;
        }
        LLAMA_LOG_WARN(" - using fallback quantization %s\n", ggml_type_name(new_type));
        ++qs.n_fallback;
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

static std::atomic<bool> bpw_stop{ false };

static void signal_handler(int) {
    bpw_stop.store(true, std::memory_order_relaxed);
}

// Returns tensor type overrides that meet a global bpw target
static std::unordered_map<std::string, ggml_type> target_bpw_type(
    llama_model_loader & ml,
    const llama_model & model,
    const std::vector<const llama_model_loader::llama_tensor_weight *> & tensors,
    const std::map<int, std::string> & mapped,
    const std::unordered_map<std::string, std::vector<float>> * values_data,
    const std::unordered_map<std::string, std::vector<float>> * activations_data,
    const llama_model_quantize_params * params,
    int nthread
) {
    bpw_stop.store(false, std::memory_order_relaxed);
    // SIGINT/SIGTERM signal handlers
    struct signal_scope_guard {
        using handler_t = void (*)(int);
        handler_t prev_int = SIG_DFL;
        handler_t prev_term = SIG_DFL;
        signal_scope_guard() {
            prev_int  = std::signal(SIGINT,  signal_handler);
            prev_term = std::signal(SIGTERM, signal_handler);
        }
        ~signal_scope_guard() {
            std::signal(SIGINT,  prev_int);
            std::signal(SIGTERM, prev_term);
        }
    } signal_guard;

    // Error and bias projection per GGML_TYPE per tensor
    struct candidate_types {
        ggml_type type = GGML_TYPE_COUNT;
        float bpw = 0.0f;
        size_t bytes = 0;
        double error = 0.0;
        double mse = 0.0;
        double proj = 0.0;
    };

    // Per‑tensor quantization mix that satisfies a global bpw target
    struct tensor_info {
        const llama_model_loader::llama_tensor_weight * w = nullptr;
        std::vector<candidate_types> candidate;
        int choice = -1;
        float min_bpw = 0.0;
        float max_bpw = 0.0;
        size_t n_elements = 0;
    };

    // subset of quantization types with the best accuracy/size tradeoff
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

    constexpr double epsilon = 1e-12;
    constexpr double infinity = std::numeric_limits<double>::infinity();
    constexpr uint32_t file_magic = 0x42505731;  // BPW1
    constexpr uint64_t arbitrary_magic = 0xeabada55cafed00d;
    const char * func = __func__;

    // Tensor size in bytes for a given type
    auto tensor_bytes = [](const ggml_tensor * t, const ggml_type typ) -> size_t {
        const int64_t n_per_row = t->ne[0];
        const size_t row_sz = ggml_row_size(typ, n_per_row);
        return (size_t)ggml_nrows(t) * row_sz;
    };

    // Tensor bpw for a given type
    auto tensor_bpw = [&](const ggml_tensor * t, const ggml_type typ) -> double {
        const size_t bytes = tensor_bytes(t, typ);
        return (double)bytes * 8.0 / (double)ggml_nelements(t);
    };

    // Check if tensor is compatible with quantization type
    auto is_compatible = [](const ggml_tensor * t, const ggml_type typ) -> bool {
        const int64_t blck = ggml_blck_size(typ);
        return blck <= 1 || (t->ne[0] % blck) == 0;
    };

    // Get suitable fallback for type
    auto make_compatible = [&](const ggml_tensor * t, const ggml_type typ) -> ggml_type {
        if (is_compatible(t, typ)) { return typ; }
        const ggml_type fb = fallback_type(typ);
        return is_compatible(t, fb) ? fb : GGML_TYPE_F16;
    };

    // Check if tensor is an IQ type
    auto is_iq = [](const enum ggml_type t) {
        switch (t) {
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
    auto can_quantize = [&](const ggml_tensor * t) -> bool {
        if (ggml_n_dims(t) < 2) { return false; } // skip 1D tensors
        return is_quantizable(ggml_get_name(t), model.arch, params);
    };

    // Saved state per tensor
    struct saved_info {
        std::vector<candidate_types> candidate;
        int choice = -1;
        float min_bpw = 0.0f;
        float max_bpw = 0.0f;
        size_t n_elements = 0;
    };

    // DJB2 hashing algorithm
    auto djb2_hash = [&](const uint8_t * data, const size_t n) -> uint64_t {
        uint64_t h = 5381;
        for (size_t i = 0; i < n; ++i) {
            h = (h << 5) + h + data[i];
        }
        return h ? h : arbitrary_magic;
    };

    // Get model ID from metadata hash
    auto metadata_id = [&](const gguf_context * ctx) -> uint64_t {
        const size_t sz = gguf_get_meta_size(ctx);
        std::vector<uint8_t> buf(sz);
        gguf_get_meta_data(ctx, buf.data());
        return djb2_hash(buf.data(), buf.size());
    };

    std::string gen_name;
    std::string checkpoint_file;
    char hex[17];
    const uint64_t model_id = metadata_id(ml.meta.get());

    std::snprintf(hex, sizeof(hex), "%016" PRIx64, (uint64_t)model_id);
    ml.get_key(LLM_KV_GENERAL_NAME, gen_name, false);
    std::replace(gen_name.begin(), gen_name.end(), ' ', '_');

    gen_name.empty() ? checkpoint_file = ml.arch_name : checkpoint_file = gen_name;
    checkpoint_file += "-" + std::string(hex) + ".bpw_state";

    if (params->keep_bpw_state && params->bpw_state) {
        const auto * filename = static_cast<const char*>(params->bpw_state);
        std::ifstream ifs(filename, std::ios::binary);
        if (ifs.good()) {
            checkpoint_file = std::string(filename);
        } else {
            std::ofstream ofs(filename, std::ios::binary | std::ios::app);
            if (ofs.is_open()) {
                checkpoint_file = std::string(filename);
                ofs.close();
                std::remove(checkpoint_file.c_str());
            } else {
                LLAMA_LOG_WARN("%s: %s is not a valid file name. Using %s instead\n", func, filename, checkpoint_file.c_str());
            }
        }
    }

    // Serializes vector<tensor_info> to disk
    auto save_bpw_state = [&](const std::vector<tensor_info> & all_vec) {
        const std::string tmp = checkpoint_file + ".tmp";
        std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
        if (!ofs) { return; }
        ofs.write((const char *)&file_magic, sizeof(file_magic));
        ofs.write((const char *)&model_id, sizeof(model_id));
        const uint64_t n = all_vec.size();
        ofs.write((const char *)&n, sizeof(n));
        for (const auto & ti : all_vec) {
            const std::string name = ggml_get_name(ti.w->tensor);
            const auto len = (uint32_t)name.size();
            ofs.write((const char *)&len, sizeof(len));
            ofs.write(name.data(), len);

            const uint64_t cn = ti.candidate.size();
            ofs.write((const char *)&cn, sizeof(cn));
            ofs.write((const char *)&ti.choice, sizeof(ti.choice));
            ofs.write((const char *)&ti.min_bpw, sizeof(ti.min_bpw));
            ofs.write((const char *)&ti.max_bpw, sizeof(ti.max_bpw));
            const uint64_t ne = ti.n_elements;
            ofs.write((const char *)&ne, sizeof(ne));

            for (const auto & c : ti.candidate) {
                const int32_t  t = c.type;
                const uint64_t b = c.bytes;
                ofs.write((const char *)&t, sizeof(t));
                ofs.write((const char *)&c.bpw, sizeof(c.bpw));
                ofs.write((const char *)&b, sizeof(b));
                ofs.write((const char *)&c.error, sizeof(c.error));
            }
        }

        ofs.close();
        std::remove(checkpoint_file.c_str());
        std::rename(tmp.c_str(), checkpoint_file.c_str());
        LLAMA_LOG_INFO("%s: saved progress for %lu tensors to %s\n", func, all_vec.size(), checkpoint_file.c_str());
    };

    // Deserializes vector<tensor_info> from disk
    auto load_bpw_state = [&]() -> std::unordered_map<std::string, saved_info> {
        std::unordered_map<std::string, saved_info> out;
        std::ifstream ifs(checkpoint_file, std::ios::binary);
        if (!ifs) { return out; }

        uint32_t magic = 0;
        uint64_t id = 0;
        ifs.read((char *)&magic, sizeof(magic));
        ifs.read((char *)&id, sizeof(id));
        if (magic != file_magic) {
            LLAMA_LOG_WARN("%s: invalid resume file, ignoring: %s\n", func, checkpoint_file.c_str());
            return out;
        }
        if (id != model_id) {
            LLAMA_LOG_WARN("%s: model ID mismatch, ignoring: %s\n", func, checkpoint_file.c_str());
            return out;
        }

        LLAMA_LOG_INFO("%s: state file found, resuming tensor quantization\n", func);

        uint64_t n = 0;
        ifs.read((char *)&n, sizeof(n));
        for (uint64_t i = 0; i < n; ++i) {
            uint32_t len = 0;
            ifs.read((char *)&len, sizeof(len));
            std::string name(len, '\0');
            ifs.read(name.data(), len);

            uint64_t cn = 0;
            ifs.read((char *)&cn, sizeof(cn));

            saved_info si;
            ifs.read((char *)&si.choice, sizeof(si.choice));
            ifs.read((char *)&si.min_bpw, sizeof(si.min_bpw));
            ifs.read((char *)&si.max_bpw, sizeof(si.max_bpw));
            uint64_t ne = 0;
            ifs.read((char *)&ne, sizeof(ne));
            si.n_elements = (size_t)ne;

            si.candidate.resize(cn);
            for (auto & s : si.candidate) {
                int32_t t = 0;
                uint64_t b = 0;
                ifs.read((char *)&t, sizeof(t));
                s.type = (ggml_type)t;
                ifs.read((char *)&s.bpw, sizeof(s.bpw));
                ifs.read((char *)&b, sizeof(b));
                s.bytes = (size_t)b;
                ifs.read((char *)&s.error, sizeof(s.error));
            }

            out.emplace(std::move(name), std::move(si));
        }

        LLAMA_LOG_INFO("%s: loaded bpw state for %lu tensors from %s\n", func, out.size(), checkpoint_file.c_str());
        return out;
    };

    // Deletes checkpoint file unless --keep-bpw-state is set
    auto delete_bpw_state = [&] {
        std::ifstream ifs(checkpoint_file);
        if (ifs.good() && !params->keep_bpw_state) {
            LLAMA_LOG_INFO("%s: deleting %s\n", func, checkpoint_file.c_str());
            std::remove(checkpoint_file.c_str());
        }
    };

    // Check for user interrupt and save progress
    auto check_signal_handler = [&](const std::vector<tensor_info> & all_vec) {
        if (bpw_stop.load(std::memory_order_relaxed)) {
            LLAMA_LOG_INFO("\n%s: saving progress for %lu tensors to %s\n", func, all_vec.size(), checkpoint_file.c_str());
            save_bpw_state(all_vec);
            throw std::runtime_error("user interrupted the process");
        }
    };

    // Estimate error for a given type using a sampled subset of rows
    auto estimate_error = [&](const ggml_tensor * t,
        const ggml_type quant_type,
        const std::vector<float> & f32_sample,
        const std::vector<int64_t> & rows_sample,
        const float * values_sample,
        const float * activations_sample,
        std::vector<uint8_t> & quantized_buffer,
        std::vector<float> & dequantized_buffer,
        float tensor_bias_lambda,
        const float * slice_bias_lambda,
        double * out_mse = nullptr,
        double * out_proj = nullptr) -> double
    {
        const int64_t n_per_row = t->ne[0];
        const int64_t nrows = t->ne[1];
        const int64_t ne2 = t->ne[2] > 0 ? t->ne[2] : 1;
        const size_t sample_elems = f32_sample.size();
        const size_t sample_rows  = n_per_row > 0 ? sample_elems / (size_t)n_per_row : 0;

        if (sample_rows == 0) {
            if (out_mse) { *out_mse = 0.0; }
            if (out_proj) { *out_proj = 0.0; }
            return 0.0;
        }

        size_t expected_rows = 0;
        for (int64_t s = 0; s < ne2; ++s) {
            expected_rows += (size_t)rows_sample[s];
        }

        if (expected_rows != sample_rows) {
            if (out_mse) { *out_mse = infinity; }
            if (out_proj) { *out_proj = 0.0; }
            return infinity;
        }

        const size_t row_sz = ggml_row_size(quant_type, n_per_row);
        const size_t buf_sz = row_sz * sample_rows;

        if (quantized_buffer.size() < buf_sz) { quantized_buffer.resize(buf_sz); }
        if (dequantized_buffer.size() < sample_elems) { dequantized_buffer.resize(sample_elems); }

        const bool has_values = values_sample != nullptr;
        const bool has_activations = activations_sample != nullptr;

        // Bias denominators per slice
        std::vector<double> bias_denom(ne2, 0.0);
        if (has_activations) {
            for (int64_t s = 0; s < ne2; ++s) {
                const float * v = has_values ? values_sample + s * n_per_row : nullptr;
                const float * a = activations_sample + s * n_per_row;
                double denom = 0.0;
                for (int64_t j = 0; j < n_per_row; ++j) {
                    const double w  = v ? std::max(0.0f, v[j]) : 1.0;
                    const double aj = a[j];
                    denom += w * aj * aj;
                }

                bias_denom[s] = denom;
            }
        }

        // Row squared norms (weighted if values present)
        std::vector<double> row_sq_norm(sample_rows, 0.0);
        {
            size_t off = 0;
            size_t ridx = 0;
            for (int64_t s = 0; s < ne2; ++s) {
                const int64_t rs = rows_sample[s];
                if (rs == 0) { continue; }

                const float * v = has_values ? values_sample + s * n_per_row : nullptr;
                for (int64_t r = 0; r < rs; ++r, ++ridx) {
                    const float * x = f32_sample.data() + off;
                    double sum = 0.0;
                    if (v) {
                        for (int64_t j = 0; j < n_per_row; ++j) {
                            const double w = std::max(0.0f, v[j]);
                            const double xx = x[j];
                            sum += w * xx * xx;
                        }
                    } else {
                        for (int64_t j = 0; j < n_per_row; ++j) {
                            const double xx = x[j];
                            sum += xx * xx;
                        }
                    }

                    row_sq_norm[ridx] = sum;
                    off += (size_t)n_per_row;
                }
            }
        }

        // Quantize per slice into quantized_buffer
        {
            size_t qoff = 0;
            size_t foff = 0;
            for (int64_t s = 0; s < ne2; ++s) {
                const int64_t rs = rows_sample[s];
                if (rs == 0) { continue; }

                const float * v = has_values ? values_sample + s * n_per_row : nullptr;
                (void)ggml_quantize_chunk(quant_type, f32_sample.data() + foff, quantized_buffer.data() + qoff, 0, rs, n_per_row, v);
                qoff += row_sz * (size_t)rs;
                foff += (size_t)rs * (size_t)n_per_row;
            }
        }

        // Dequantize into dequantized_buffer
        {
            if (quant_type == GGML_TYPE_F16) {
                for (size_t r = 0; r < sample_rows; ++r) {
                    auto src = (const ggml_fp16_t *)(quantized_buffer.data() + r * row_sz);
                    float * dst = dequantized_buffer.data() + r * (size_t)n_per_row;
                    ggml_fp16_to_fp32_row(src, dst, (int)n_per_row);
                }
            } else if (quant_type == GGML_TYPE_BF16) {
                for (size_t r = 0; r < sample_rows; ++r) {
                    auto src = (const ggml_bf16_t *)(quantized_buffer.data() + r * row_sz);
                    float * dst = dequantized_buffer.data() + r * (size_t)n_per_row;
                    ggml_bf16_to_fp32_row(src, dst, (int)n_per_row);
                }
            } else {
                const ggml_type_traits * traits = ggml_get_type_traits(quant_type);
                if (!traits || !traits->to_float) {
                    if (out_mse) { *out_mse = infinity; }
                    if (out_proj) { *out_proj = 0.0; }
                    return infinity;
                }
                for (size_t r = 0; r < sample_rows; ++r) {
                    const uint8_t * src = quantized_buffer.data() + r * row_sz;
                    float * dst = dequantized_buffer.data() + r * (size_t)n_per_row;
                    traits->to_float(src, dst, (int)n_per_row);
                }
            }
        }

        // Compute error per slice with trimmed aggregation
        auto trimmed_mean = [](std::vector<double> & v) -> double {
            const int64_t n = (int64_t)v.size();
            if (n == 0) { return 0.0; }
            double sum = std::accumulate(v.begin(), v.end(), 0.0);
            if (n < 50) { return sum / (double)n; } // too few elements to trim
            int64_t k = (int64_t) std::floor(0.025 * (double)n); // trim 5% (2.5% each side)
            std::sort(v.begin(), v.end());
            const auto num = (double)(n - 2 * k);
            sum = std::accumulate(v.begin() + k, v.begin() + (n - k), 0.0);
            return sum / std::max(1.0, num);
        };

        size_t off = 0;
        size_t ridx = 0;
        double total_mse = 0.0;
        double total_proj = 0.0;
        double total_bias = 0.0;
        for (int64_t s = 0; s < ne2; ++s) {
            const int64_t rs = rows_sample[s];
            if (rs == 0) { continue; }

            const float * v = has_values ? values_sample + s * n_per_row : nullptr;
            const float * a = has_activations ? activations_sample + s * n_per_row : nullptr;
            const double denom_bias = has_activations ? bias_denom[s] : 0.0;
            std::vector<double> row_mse_norm;
            row_mse_norm.reserve(rs);
            std::vector<double> row_proj_norm;
            if (a) { row_proj_norm.reserve(rs); }

            for (int64_t r = 0; r < rs; ++r, ++ridx) {
                const float * x = f32_sample.data() + off;
                const float * y = dequantized_buffer.data() + off;
                double w_mse = 0.0;
                double bias_num = 0.0;
                for (int64_t j = 0; j < n_per_row; ++j) {
                    const double wj = v ? std::max(0.0f, v[j]) : 1.0;
                    const double e = y[j] - x[j];
                    w_mse += wj * e * e;
                    if (a) { bias_num += wj * e * a[j]; }
                }

                const double denom_x = row_sq_norm[ridx];
                const double m_norm = w_mse / (denom_x + epsilon);
                row_mse_norm.push_back(std::isfinite(m_norm) ? m_norm : infinity);

                if (a) {
                    double p_norm = 0.0;
                    if (denom_bias > 0.0) {
                        const double proj = bias_num * bias_num / (denom_bias + epsilon);
                        p_norm = std::isfinite(proj) ? proj : 0.0;
                    }

                    row_proj_norm.push_back(p_norm);
                }

                off += (size_t)n_per_row;
            }

            const double slice_mse = trimmed_mean(row_mse_norm) * (double)nrows;
            const double slice_proj = a ? trimmed_mean(row_proj_norm) * (double)nrows : 0.0;

            total_mse += slice_mse;
            total_proj += slice_proj;

            const double bl = slice_bias_lambda ? (double)std::max(0.0f, slice_bias_lambda[s]) : (double)tensor_bias_lambda;
            total_bias += bl * slice_proj;

            if (!std::isfinite(total_mse) || !std::isfinite(total_proj) || !std::isfinite(total_bias)) {
                if (out_mse) { *out_mse = infinity; }
                if (out_proj) { *out_proj = 0.0; }
                return infinity;
            }
        }

        if (out_mse) { *out_mse = total_mse; }
        if (out_proj) { *out_proj = total_proj; }

        const double total_err = total_mse + total_bias;
        return std::isfinite(total_err) ? total_err : infinity;
    };

    // Returns lambda per slice or 0.0 if no activations
    auto estimate_lambda = [&](const float * values, const float * activations, const int64_t n_per_row, const int64_t ne2) -> std::vector<float> {
        const int64_t ns = std::max<int64_t>(1, ne2);
        std::vector<float> lambdas(ns, 0.0f);
        if (!activations) { return lambdas; }

        for (int64_t s = 0; s < ns; ++s) {
            const float * v = values ? values + s * n_per_row : nullptr;
            const float * a = activations + s * n_per_row;
            double s1 = 0.0;
            double s2 = 0.0;
            for (int64_t j = 0; j < n_per_row; ++j) {
                const double w = v ? std::max(0.0f, v[j]) : 1.0;
                const double aw = std::sqrt(w) * a[j];
                const double z  = aw * aw;
                s1 += z;
                s2 += z * z;
            }

            float l = 0.0f;
            if (s1 > 0.0) {
                const auto n = (double)n_per_row;
                const double c = std::max(0.0, s2 / (s1 * s1 + epsilon) - 1.0 / n);
                l = (float)std::clamp(12.0 * (c / (c + 1.0)), 0.0, 16.0);
            }

            lambdas[(size_t)s] = l;
        }

        return lambdas;
    };

    const auto bpw_data = load_bpw_state();

    // Parallelize tensor processing (courtesy of https://github.com/ddh0)
    auto process_tensor = [&](const llama_model_loader::llama_tensor_weight * tw,
        std::vector<no_init<uint8_t>> & thread_local_buffer,
        std::mutex & loader_mutex,
        std::mutex & log_mutex) -> std::optional<tensor_info>
    {
        ggml_tensor * tensor = tw->tensor;
        const std::string name = ggml_get_name(tensor);
        if (bpw_stop.load(std::memory_order_relaxed)) {
            return std::nullopt;
        }

        // check for pre-computed results from a checkpoint file.
        auto it_saved = bpw_data.find(name);
        if (it_saved != bpw_data.end()) {
            tensor_info info;
            info.w = tw;
            info.candidate = it_saved->second.candidate;
            info.choice = it_saved->second.choice;
            info.min_bpw = it_saved->second.min_bpw;
            info.max_bpw = it_saved->second.max_bpw;
            info.n_elements = it_saved->second.n_elements ? it_saved->second.n_elements : (size_t)ggml_nelements(tensor);
            return info;
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

        // Dequantize sampled rows into f32_sample
        const int64_t n_per_row = tensor->ne[0];
        const int64_t nrows_total = tensor->ne[1];
        const int64_t ne2 = tensor->ne[2] > 0 ? tensor->ne[2] : 1;

        // Compute rows based on tensor shape and slice count
        auto sample_rows = [](const int64_t n, const int64_t rows, const int64_t n2, const bool has_acts) -> int64_t {
            const double tensor_budget = has_acts ? 1 * 1024 * 1024 : 0.5 * 1024 * 1024;
            const double scale_rows = std::clamp(std::sqrt(std::max(1.0, (double)rows) / 4096.0), 0.5, 2.0); // favour more rows for large tensors
            const double slice_budget = tensor_budget * scale_rows / std::max<int64_t>(1, n2);
            const int64_t min_rows = has_acts ? 128 : 64;
            constexpr int64_t max_rows = 4096; // row limit to avoid excessive memory use
            int64_t total_rows = std::llround(slice_budget / std::max<int64_t>(1, n));
            total_rows = std::max<int64_t>(min_rows, std::min<int64_t>(total_rows, std::min<int64_t>(rows, max_rows)));
            if (rows <= min_rows * 2) { total_rows = rows; }
            return total_rows;
        };

        const int64_t rows_sample_per_expert = sample_rows(n_per_row, nrows_total, ne2, activations_data != nullptr);
        std::vector<float> f32_sample;
        f32_sample.reserve((size_t)ne2 * (size_t)std::min<int64_t>(nrows_total, rows_sample_per_expert) * (size_t)n_per_row);
        std::vector<int64_t> rows_sample(ne2, 0);
        const ggml_type src_type = tensor->type;
        const ggml_type_traits * src_traits = ggml_get_type_traits(src_type);
        const bool src_is_quant = ggml_is_quantized(src_type);
        const size_t src_row_sz = ggml_row_size(src_type, n_per_row);

        // Convert a single row to fp32
        auto row_to_fp32 = [&](const uint8_t * src, float * dst) {
            const ggml_type t = src_type;
            if (t == GGML_TYPE_F32) {
                std::memcpy(dst, src, sizeof(float) * (size_t)n_per_row);
                return;
            }
            if (t == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *)src, dst, (int)n_per_row);
                return;
            }
            if (t == GGML_TYPE_BF16) {
                ggml_bf16_to_fp32_row((const ggml_bf16_t *)src, dst, (int)n_per_row);
                return;
            }
            if (src_is_quant) {
                GGML_ASSERT(src_traits && src_traits->to_float);
                src_traits->to_float(src, dst, (int)n_per_row);
                return;
            }

            throw std::runtime_error(format("unsupported src type %s for sampling", ggml_type_name(t)));
        };

        // Sample rows randomly per slice
        {
            f32_sample.clear();
            std::vector<float> row_buffer(n_per_row);
            for (int64_t slice = 0; slice < ne2; ++slice) {
                std::mt19937 rng(std::hash<std::string>{}(name) ^ arbitrary_magic ^ slice);
                const int64_t rows_sample_max = std::max<int64_t>(1, std::min<int64_t>(nrows_total, rows_sample_per_expert));
                const int64_t stride = std::max<int64_t>(1, nrows_total / rows_sample_max);
                int64_t offset = 0;
                if (stride > 1) {
                    std::uniform_int_distribution<int64_t> dist(0, stride - 1);
                    offset = dist(rng);
                }

                int64_t current = 0;
                for (int64_t r = offset; r < nrows_total && current < rows_sample_max; r += stride) {
                    const uint8_t * src_row = (const uint8_t *)tensor->data + slice * (src_row_sz * nrows_total) + r * src_row_sz;
                    if (src_type == GGML_TYPE_F32) {
                        const auto *src_f32 = (const float *)src_row;
                        f32_sample.insert(f32_sample.end(), src_f32, src_f32 + n_per_row);
                    } else {
                        row_to_fp32(src_row, row_buffer.data());
                        f32_sample.insert(f32_sample.end(), row_buffer.begin(), row_buffer.end());
                    }

                    ++current;
                }

                rows_sample[slice] = current;
            }
        }

        auto side_data = [&](const std::unordered_map<std::string, std::vector<float>> * m, const std::string & tensor_name) {
            if (!m) { return std::pair<const float*, size_t>{nullptr, 0}; }

            const std::string key = remap_imatrix(tensor_name, mapped);
            const auto it = m->find(key);
            return it == m->end() ? std::pair<const float*, size_t>{nullptr, 0} : std::pair<const float*, size_t>{ it->second.data(), it->second.size() };
        };

        // Copy this row's side data (values and activations), or broadcasts to all slices
        auto copy_or_broadcast = [&](const float * src, size_t src_sz, std::vector<float> & dst) {
            dst.clear();
            if (!src || src_sz == 0) { return; }

            const size_t want = (size_t)ne2 * (size_t)n_per_row;
            if (src_sz == want) {
                dst.assign(src, src + want);
                return;
            }
            if (src_sz == (size_t)n_per_row) {
                dst.resize(want);
                for (int64_t s = 0; s < ne2; ++s) {
                    std::memcpy(dst.data() + s * n_per_row, src, n_per_row * sizeof(float));
                }
                return;
            }

            std::lock_guard<std::mutex> lock(log_mutex);
            LLAMA_LOG_WARN("%s: side data size mismatch for %s: got %zu, expected %zu or %zu; ignoring\n", func, name.c_str(), src_sz, (size_t)n_per_row, want);
        };

        const auto [values_all, values_sz] = side_data(values_data, name);
        const auto [activations_all, activations_sz] = side_data(activations_data, name);
        std::vector<float> values_sample;
        std::vector<float> activations_sample;
        if (values_all) { copy_or_broadcast(values_all, values_sz, values_sample); }
        if (activations_all) { copy_or_broadcast(activations_all, activations_sz, activations_sample); }

        tensor_info info;
        info.w = tw;
        info.n_elements = ggml_nelements(tensor);
        size_t total_sampled_rows = f32_sample.size() / n_per_row;

        // Build list of candidate types first (compatible ones)
        const bool has_valid_imatrix = !values_sample.empty() && values_sample.size() == (size_t)ne2 * (size_t)n_per_row;
        size_t max_row_sz = 0;
        const ggml_type * base_arr = quant_types;
        const size_t base_sz = std::size(quant_types);
        std::vector<ggml_type> compatible_candidates;
        compatible_candidates.reserve(base_sz);

        for (size_t i = 0; i < base_sz; ++i) {
            ggml_type ts_type = base_arr[i];
            if (is_iq(ts_type) && !has_valid_imatrix) {
                std::lock_guard<std::mutex> lock(log_mutex);
                LLAMA_LOG_WARN("\t%s: skipping %s for %s, no or mismatched imatrix\n", func, ggml_type_name(ts_type), name.c_str());
                continue;
            }

            ggml_type tt = make_compatible(tensor, ts_type);
            if (!is_compatible(tensor, tt)) { continue; }
            compatible_candidates.push_back(tt);
            max_row_sz = std::max(max_row_sz, ggml_row_size(tt, n_per_row));
        }

        std::sort(compatible_candidates.begin(), compatible_candidates.end());
        compatible_candidates.erase(std::unique(compatible_candidates.begin(), compatible_candidates.end()), compatible_candidates.end());

        // Adjusts the trade-off between systematic bias (introduced by block‑wise scaling) and MSE.
        // Larger values favours quantisation types that produce smaller bias even if the MSE is slightly bigger
        float tensor_lambda = 0.0f;
        std::vector<float> lambdas;
        const float * values = values_sample.empty() ? nullptr : values_sample.data();
        const float * activations = activations_sample.empty() ? nullptr : activations_sample.data();
        double acc = 0.0;
        int ns = 0;
        lambdas = estimate_lambda(values, activations, n_per_row, ne2);
        for (float l : lambdas) { acc += l; ++ns; }
        tensor_lambda = ns ? (float)(acc / ns) : 0.0f;

        // Evaluate candidates
        std::vector<candidate_types> eval_candidates(compatible_candidates.size());
        std::vector<uint8_t> quantized_buffer(max_row_sz * total_sampled_rows);
        std::vector<float> dequantized_buffer(f32_sample.size());
        const float * slice_lambda = lambdas.empty() ? nullptr : lambdas.data();
        for (size_t i = 0; i < compatible_candidates.size(); ++i) {
            if (bpw_stop.load(std::memory_order_relaxed)) { return std::nullopt; }

            const ggml_type tensor_type = compatible_candidates[i];
            const auto bpw = (float)tensor_bpw(tensor, tensor_type);
            const size_t bytes = tensor_bytes(tensor, tensor_type);
            double mse = 0.0;
            double proj = 0.0;
            const auto err = estimate_error(tensor, tensor_type, f32_sample, rows_sample, values, activations,
                quantized_buffer, dequantized_buffer, tensor_lambda, slice_lambda, &mse, &proj);
            eval_candidates[i] = candidate_types{ tensor_type, bpw, bytes, err, mse, proj };
        }

        if (bpw_stop.load(std::memory_order_relaxed)) { return std::nullopt; }

        // Check if biasing is needed
        bool bias_needed = false;
        if (!lambdas.empty()) {
            int min_mse  = -1;
            int min_bias = -1;
            double best_mse = std::numeric_limits<double>::infinity();
            double best_err = std::numeric_limits<double>::infinity();
            for (int i = 0; i < (int)eval_candidates.size(); ++i) {
                const auto & c = eval_candidates[i];
                if (c.bytes == 0) { continue; }
                if (c.mse  < best_mse) {
                    best_mse = c.mse;
                    min_mse  = i;
                }
                if (c.error < best_err) {
                    best_err = c.error;
                    min_bias = i;
                }
            }

            if (min_mse != min_bias) {
                bias_needed = true;
            } else {
                double max_rel_bias = 0.0;
                for (const auto & c : eval_candidates) {
                    if (c.bytes == 0) { continue; }
                    const double mse = std::max(c.mse, epsilon);
                    const double bias_term = std::max(0.0, c.error - c.mse);
                    max_rel_bias = std::max(bias_term / mse, max_rel_bias);
                }

                bias_needed = max_rel_bias >= 0.5; // >= 50% of MSE?
            }
        }

        for (auto & c : eval_candidates) {
            if (c.bytes == 0) { continue; }
            const double final_err = bias_needed ? c.error : c.mse;
            info.candidate.push_back(candidate_types{ c.type, c.bpw, c.bytes, final_err, c.mse, c.proj });
        }

        if (info.candidate.empty()) {
            // As a last resort, keep original type
            float bpw = ggml_nbytes(tensor) * 8.0f / info.n_elements;
            info.candidate.push_back(candidate_types{ tensor->type, bpw, ggml_nbytes(tensor), 0.0 });
        }

        // Keep only the pareto‑optimal candidates and enforce convexity in (bytes, error) curve
        auto pareto_convex = [&](std::vector<candidate_types> & candidates) {
            if (candidates.empty()) { return; }

            std::sort(candidates.begin(), candidates.end(), [](const candidate_types & a, const candidate_types & b) {
                if (a.bytes != b.bytes) { return a.bytes < b.bytes; }
                return a.error < b.error;
            });
            candidates.erase(std::unique(candidates.begin(), candidates.end(), [](const candidate_types & a, const candidate_types & b) {
                return a.bytes == b.bytes;
            }), candidates.end());
            std::vector<candidate_types> pareto;
            pareto.reserve(candidates.size());
            double best_err = infinity;
            for (const auto & c : candidates) {
                if (c.error < best_err) {
                    best_err = c.error;
                    pareto.push_back(c);
                }
            }
            candidates.swap(pareto);
            if (candidates.size() < 3) { return; } // need at least 3 points to do convex hull

            // Convex hull (lower envelope)
            auto cross_product = [](const candidate_types & h0, const candidate_types & h1, const candidate_types & p) -> double {
                const double dx1 = (double)h1.bytes - (double)h0.bytes;
                const double dy1 = h1.error - h0.error;
                const double dx2 = (double)p.bytes - (double)h0.bytes;
                const double dy2 = p.error - h0.error;
                return dx1 * dy2 - dx2 * dy1;
            };
            std::vector<candidate_types> hull; hull.reserve(candidates.size());
            for (const auto & c : candidates) {
                while (hull.size() >= 2) {
                    if (cross_product(hull[hull.size() - 2], hull[hull.size() - 1], c) <= epsilon) {
                        hull.pop_back();
                    } else {
                        break;
                    }
                }

                hull.push_back(c);
            }

            candidates.swap(hull);
        };

        pareto_convex(info.candidate);

        // Initialize choice at the smallest bpw candidate
        info.choice = 0;
        info.min_bpw = info.candidate.front().bpw;
        info.max_bpw = info.candidate.back().bpw;

        return info;
    };

    std::vector<tensor_info> all; // this vector will be populated by the parallel workers
    {
        std::atomic<size_t> tensor_idx{0}; // shared work queue index for all threads
        const size_t tensors_to_process = tensors.size();
        std::mutex loader_mutex;
        std::mutex log_mutex;
        std::mutex results_mutex;
        std::vector<std::thread> workers;
        int threads_to_spawn = std::max(1, std::min<int>(nthread, (int)tensors_to_process));

        for (int i = 0; i < threads_to_spawn; ++i) {
            workers.emplace_back([&]() {
                std::vector<no_init<uint8_t>> thread_local_buffer;
                while (true) {
                    const size_t current_idx = tensor_idx.fetch_add(1);
                    if (current_idx >= tensors_to_process) { break; }
                    const auto * tw = tensors[current_idx];
                    if (!can_quantize(tw->tensor)) { continue; }
                    // Execute the main processing logic for this tensor
                    std::optional<tensor_info> result_info = process_tensor(tw, thread_local_buffer, loader_mutex, log_mutex);
                    if (result_info) {
                        std::lock_guard<std::mutex> lock(results_mutex);
                        all.push_back(std::move(*result_info));
                    }
                }
            });
        }

        for (auto & w : workers) { w.join(); }
    }

    check_signal_handler(all);
    if (params->keep_bpw_state) { save_bpw_state(all); }

    if (all.empty()) { return {}; }

    // Compute total elements across all tensors and bytes for non-quantizable tensors
    size_t nq_elements = 0;
    size_t nq_bytes = 0;
    for (const auto * it : tensors) {
        const ggml_tensor * tensor = it->tensor;
        const std::string name = ggml_get_name(tensor);
        nq_elements += (size_t)ggml_nelements(tensor);
        if (!can_quantize(tensor)) { nq_bytes += ggml_nbytes(tensor); }
    }

    auto total_bytes = [&]() -> size_t {
        size_t tb = 0;
        for (const auto & ti : all) {
            tb += ti.candidate[ti.choice].bytes;
        }

        return tb;
    };

    size_t q_elements = 0;
    size_t min_bytes = 0;
    size_t max_bytes = 0;
    for (const auto & ti : all) {
        q_elements += (size_t)ti.n_elements;
        min_bytes += ti.candidate.front().bytes;  // smallest candidate per tensor
        max_bytes += ti.candidate.back().bytes;   // largest candidate per tensor
    }

    if (q_elements == 0) { return {}; }

    const double target_bpw = params->target_bpw;
    size_t target_total_bytes = std::llround(target_bpw * (double)nq_elements / 8.0);
    size_t budget_bytes = target_total_bytes >= nq_bytes ? target_total_bytes - nq_bytes : min_bytes;

    // Get the types' override
    auto emit_overrides = [&]() -> std::unordered_map<std::string, ggml_type> {
        std::unordered_map<std::string, ggml_type> overrides;
        LLAMA_LOG_INFO("%s: - estimated tensor quantization mix:\n", func);
        for (const auto & ti : all) {
            LLAMA_LOG_INFO("\t%s: %45s - \t%8s, \t%1.4f bpw,\terror: %.4f\n",
                func, ggml_get_name(ti.w->tensor), ggml_type_name(ti.candidate[ti.choice].type), ti.candidate[ti.choice].bpw, ti.candidate[ti.choice].error);
            overrides[ggml_get_name(ti.w->tensor)] = ti.candidate[ti.choice].type;
        }

        return overrides;
    };

    if (budget_bytes <= min_bytes) {
        for (auto & ti : all) { ti.choice = 0; }
        return emit_overrides();
    }
    if (budget_bytes >= max_bytes) {
        for (auto & ti : all) { ti.choice = (int)ti.candidate.size() - 1; }
        return emit_overrides();
    }

    // Certain tensors have a higher impact on model quality, so we apply a lower penalty to them
    auto is_important = [&](const std::string & tensor_name) -> bool {
        bool important = tensor_name == "output.weight";
        if (!important && !params->no_importance) {
            important = tensor_name.find(".attn_v.weight") != std::string::npos ||
                        tensor_name.find(".time_mix_value.weight") != std::string::npos ||
                        tensor_name.find(".ffn_down.weight") != std::string::npos ||
                        tensor_name.find(".ffn_down_exps.weight") != std::string::npos ||
                        tensor_name.find(".attn_output.weight") != std::string::npos ||
                        tensor_name.find(".time_mix_output.weight") != std::string::npos ||
                        tensor_name.find(".attn_o.weight") != std::string::npos;
        }

        return important;
    };

    // Lagrangian relaxation to minimize error subject to a bpw target constraint
    auto lagrange_penalty = [&](const double mu, std::vector<int> & choice, size_t & bytes, double & err) {
        choice.resize(all.size());
        bytes = 0;
        err = 0.0;
        for (size_t i = 0; i < all.size(); ++i) {
            const auto & candidate = all[i].candidate;
            const std::string tensor_name = ggml_get_name(all[i].w->tensor);
            double effective_mu = mu;
            if (is_important(tensor_name)) { effective_mu *= 0.1; } // important tensors get 10x lower penalty

            int best_j = 0;
            double best_val = infinity;
            for (int j = 0; j < (int)candidate.size(); ++j) {
                const double bits = (double)candidate[j].bytes * 8.0;
                const double val = candidate[j].error + effective_mu * bits;
                if (val < best_val - epsilon || (std::abs(val - best_val) <= epsilon && candidate[j].bytes < candidate[best_j].bytes)) {
                    best_val = val;
                    best_j = j;
                }
            }

            choice[i] = best_j;
            bytes += candidate[best_j].bytes;
            err += candidate[best_j].error;
        }
    };

    size_t bytes_lo = 0;
    size_t bytes_hi = 0;
    size_t bytes_mid = 0;
    double mu_lo = 0.0;
    double mu_hi = 1.0;
    double err_lo = 0.0;
    double err_hi = 0.0;
    double err_mid = 0.0;
    std::vector<int> choice_lo;
    std::vector<int> choice_hi;
    std::vector<int> choice_mid;
    std::vector<int> best_under_choice;
    std::vector<int> best_over_choice;

    lagrange_penalty(mu_lo, choice_lo, bytes_lo, err_lo);

    // Increase mu until we get under budget or hit a safety cap
    {
        int expand = 0;
        size_t prev_bytes_hi = std::numeric_limits<size_t>::max();
        while (true) {
            lagrange_penalty(mu_hi, choice_hi, bytes_hi, err_hi);
            if (bytes_hi <= budget_bytes) { break; }
            if (bytes_hi >= prev_bytes_hi) { break; }
            prev_bytes_hi = bytes_hi;

            mu_hi *= 2.0; // double the penalty multiplier to reduce tensor sizes
            if (++expand > 60) { break; } // safety cap to prevent an infinite loop
        }
    }

    double best_under_gap = infinity;
    double best_over_gap = infinity;
    double best_under_err = infinity;
    double best_over_err = infinity;
    for (int it = 0; it < 40; ++it) { // binary search iterations for optimal Lagrange multiplier (40 ≈ 1e-12 precision)
        double mu = 0.5 * (mu_lo + mu_hi); // midpoint of current bounds
        lagrange_penalty(mu, choice_mid, bytes_mid, err_mid);

        const double gap = std::abs((double)bytes_mid - (double)budget_bytes);
        if (bytes_mid > budget_bytes) {
            // Too big, need stronger penalty
            mu_lo = mu;
            if (gap < best_over_gap - epsilon || (std::abs(gap - best_over_gap) <= epsilon && err_mid < best_over_err)) {
                best_over_gap = gap;
                best_over_err = err_mid;
                best_over_choice = choice_mid;
            }
        } else {
            // Under budget, good candidate
            mu_hi = mu;
            if (gap < best_under_gap - epsilon || (std::abs(gap - best_under_gap) <= epsilon && err_mid < best_under_err)) {
                best_under_gap = gap;
                best_under_err = err_mid;
                best_under_choice = choice_mid;
            }
        }
    }

    if (!best_under_choice.empty()) {
        for (size_t i = 0; i < all.size(); ++i) {
            all[i].choice = best_under_choice[i];
        }
    } else if (!best_over_choice.empty()) {
        for (size_t i = 0; i < all.size(); ++i) {
            all[i].choice = best_over_choice[i];
        }
    } else {
        // Pick whichever side we already have, or keep minimal
        if (bytes_hi <= budget_bytes && !choice_hi.empty()) {
            for (size_t i = 0; i < all.size(); ++i) {
                all[i].choice = choice_hi[i];
            }
        } else {
            for (auto & ti : all) {
                ti.choice = 0;
            }
        }
    }

    // Spend any remaining budget with best upgrades that still fit (one pass)
    {
        auto cur_bytes = total_bytes();
        while (true) {
            int best_i = -1;
            int best_j = -1;
            double best_ratio = -1.0;
            double best_gain = -1.0;

            for (int i = 0; i < (int)all.size(); ++i) {
                const auto & ti = all[i];
                const std::string tensor_name  = ggml_get_name(ti.w->tensor);
                int j = ti.choice + 1;
                if (j >= (int)ti.candidate.size()) { continue; } // no upgrade available

                size_t delta_bytes = ti.candidate[j].bytes - ti.candidate[ti.choice].bytes;
                if (cur_bytes + delta_bytes > budget_bytes) { continue; } // won't fit in budget

                double err_gain = std::max(0.0, ti.candidate[ti.choice].error - ti.candidate[j].error);
                if (err_gain < epsilon) { continue; } // no error improvement

                double ratio = err_gain / (double)delta_bytes; // error reduction per byte
                if (is_important(tensor_name)) { ratio *= 5.0; } // important tensors get 5x boost

                // For tie-breaking, prioritize the largest absolute error improvement.
                if (ratio > best_ratio + epsilon || (std::abs(ratio - best_ratio) <= epsilon && err_gain > best_gain)) {
                    best_ratio = ratio;
                    best_gain = err_gain;
                    best_i = i;
                    best_j = j;
                }
            }

            if (best_i < 0) { break; } // no more upgrades within budget found

            size_t upgrade_cost = all[best_i].candidate[best_j].bytes - all[best_i].candidate[all[best_i].choice].bytes;
            all[best_i].choice = best_j;
            cur_bytes += upgrade_cost;
        }
    }

    delete_bpw_state();

    return emit_overrides();
}

static void llama_model_quantize_impl(const std::string & fname_inp, const std::string & fname_out, const llama_model_quantize_params * params) {
    ggml_type default_type;
    llama_ftype ftype = params->ftype;

    switch (params->ftype) {
        case LLAMA_FTYPE_MOSTLY_Q4_0: default_type = GGML_TYPE_Q4_0; break;
        case LLAMA_FTYPE_MOSTLY_Q4_1: default_type = GGML_TYPE_Q4_1; break;
        case LLAMA_FTYPE_MOSTLY_Q5_0: default_type = GGML_TYPE_Q5_0; break;
        case LLAMA_FTYPE_MOSTLY_Q5_1: default_type = GGML_TYPE_Q5_1; break;
        case LLAMA_FTYPE_MOSTLY_Q8_0: default_type = GGML_TYPE_Q8_0; break;
        case LLAMA_FTYPE_MOSTLY_F16:  default_type = GGML_TYPE_F16;  break;
        case LLAMA_FTYPE_MOSTLY_BF16: default_type = GGML_TYPE_BF16; break;
        case LLAMA_FTYPE_ALL_F32:     default_type = GGML_TYPE_F32;  break;

        case LLAMA_FTYPE_MOSTLY_MXFP4_MOE: default_type = GGML_TYPE_MXFP4; break;

        // K-quants
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:
        case LLAMA_FTYPE_MOSTLY_Q2_K:    default_type = GGML_TYPE_Q2_K;    break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:  default_type = GGML_TYPE_IQ3_S;   break;
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:  default_type = GGML_TYPE_Q3_K;    break;
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:  default_type = GGML_TYPE_Q4_K;    break;
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:  default_type = GGML_TYPE_Q5_K;    break;
        case LLAMA_FTYPE_MOSTLY_Q6_K:    default_type = GGML_TYPE_Q6_K;    break;
        case LLAMA_FTYPE_MOSTLY_TQ1_0:   default_type = GGML_TYPE_TQ1_0;   break;
        case LLAMA_FTYPE_MOSTLY_TQ2_0:   default_type = GGML_TYPE_TQ2_0;   break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS: default_type = GGML_TYPE_IQ2_XXS; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:  default_type = GGML_TYPE_IQ2_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ2_S:   default_type = GGML_TYPE_IQ2_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ2_M:   default_type = GGML_TYPE_IQ2_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS: default_type = GGML_TYPE_IQ3_XXS; break;
        case LLAMA_FTYPE_MOSTLY_IQ1_S:   default_type = GGML_TYPE_IQ1_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ1_M:   default_type = GGML_TYPE_IQ1_M;   break;
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:  default_type = GGML_TYPE_IQ4_NL;  break;
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:  default_type = GGML_TYPE_IQ4_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ3_S:   default_type = GGML_TYPE_IQ3_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ3_M:   default_type = GGML_TYPE_IQ3_S;   break;

        default: throw std::runtime_error(format("invalid output file type %d\n", ftype));
    }

    int nthread = params->nthread;

    if (nthread <= 0) {
        nthread = std::thread::hardware_concurrency();
    }

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
    llama_model_loader ml(fname_inp, splits, use_mmap, /*check_tensors*/ true, /*no_alloc*/ false, kv_overrides, nullptr);
    ml.init_mappings(false); // no prefetching

    llama_model model(llama_model_default_params());

    model.load_arch   (ml);
    model.load_hparams(ml);
    model.load_stats  (ml);

    quantize_state_impl qs(model, params);

    if (params->only_copy) {
        ftype = ml.ftype;
    }
    const std::unordered_map<std::string, std::vector<float>> * values_data = nullptr;
    const std::unordered_map<std::string, std::vector<float>> * activations_data = nullptr;
    if (params->imatrix) {
        values_data = static_cast<const std::unordered_map<std::string, std::vector<float>>*>(params->imatrix);
        if (values_data) {
            LLAMA_LOG_INFO("================================ Have weights data with %d entries",int(values_data->size()));
            qs.has_imatrix = true;
            // check imatrix for nans or infs
            for (const auto & kv : *values_data) {
                for (float f : kv.second) {
                    if (!std::isfinite(f)) {
                        throw std::runtime_error(format("imatrix contains non-finite value %f\n", f));
                    }
                }
            }
        }
    }
    if (params->activations) {
        activations_data = static_cast<const std::unordered_map<std::string, std::vector<float>>*>(params->activations);
        if (activations_data) {
            LLAMA_LOG_INFO(" and %d activations",int(activations_data->size()));
            qs.has_activations = true;
            // check activations for nans or infs
            for (const auto & kv : *activations_data) {
                for (float f : kv.second) {
                    if (!std::isfinite(f)) {
                        throw std::runtime_error(format("activations contain non-finite value %f\n", f));
                    }
                }
            }
        }
    }
    LLAMA_LOG_INFO("\n");

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
                LLAMA_LOG_WARN("%s: unknown KV override type for key %s\n", __func__, o.key);
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
    if (!prune_list.empty()) {
        gguf_set_val_u32(ctx_out.get(), ml.llm_kv(LLM_KV_BLOCK_COUNT).c_str(), blk_id);
    }

    // keep_split requires that the weights are sorted by split index
    if (params->keep_split) {
        std::sort(tensors.begin(), tensors.end(), [](const llama_model_loader::llama_tensor_weight * a, const llama_model_loader::llama_tensor_weight * b) {
            if (a->idx == b->idx) {
                return a->offs < b->offs;
            }
            return a->idx < b->idx;
        });
    }

    for (const auto * it : tensors) {
        const struct ggml_tensor * tensor = it->tensor;

        const std::string name = ggml_get_name(tensor);

        // TODO: avoid hardcoded tensor names - use the TN_* constants
        if (name.find("attn_v.weight")   != std::string::npos ||
            name.find("attn_qkv.weight") != std::string::npos ||
            name.find("attn_kv_b.weight")!= std::string::npos) {
            ++qs.n_attention_wv;
        } else if (name == LLM_TN(model.arch)(LLM_TENSOR_OUTPUT, "weight")) {
            qs.has_output = true;
        }
    }

    qs.n_ffn_down = qs.n_ffn_gate = qs.n_ffn_up = (int)model.hparams.n_layer;

    size_t total_size_org = 0;
    size_t total_size_new = 0;

    std::vector<std::thread> workers;
    workers.reserve(nthread);

    int idx = 0;

    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<uint8_t>> work;
    std::vector<no_init<float>> f32_conv_buf;

    uint16_t n_split = 1;

    // Assume split index is continuous
    if (params->keep_split) {
        for (const auto * it : tensors) {
            n_split = std::max(uint16_t(it->idx + 1), n_split);
        }
    }
    std::vector<gguf_context_ptr> ctx_outs(n_split);
    ctx_outs[0] = std::move(ctx_out);

    // populate the original tensors so we get an initial meta data
    for (const auto * it : tensors) {
        uint16_t i_split = params->keep_split ? it->idx : 0;
        ggml_tensor * tensor = it->tensor;
        if (!ctx_outs[i_split]) {
            ctx_outs[i_split].reset(gguf_init_empty());
        }
        gguf_add_tensor(ctx_outs[i_split].get(), tensor);
    }

    // Set split info if needed
    if (n_split > 1) {
        for (size_t i = 0; i < ctx_outs.size(); ++i) {
            gguf_set_val_u16(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_NO).c_str(), i);
            gguf_set_val_u16(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str(), n_split);
            gguf_set_val_i32(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str(), (int32_t)tensors.size());
        }
    }

    std::unordered_map<std::string, ggml_type> bpw_overrides = {};
    if (params->target_bpw != -1.0f && !params->only_copy) {
        if (params->imatrix) {
            if (params->activations) {
                LLAMA_LOG_INFO("%s: imatrix has activations, process will be more accurate\n", __func__);
            } else {
                LLAMA_LOG_INFO("%s: imatrix does not have activations, process may be less accurate\n", __func__);
            }
            if (params->no_importance) {
                LLAMA_LOG_INFO("%s: distributing bpw budget equitably across all tensors\n", __func__);
            } else {
                LLAMA_LOG_INFO("%s: assigning more bpw budget to important tensors\n", __func__);
            }
            LLAMA_LOG_INFO("%s: computing tensor quantization mix to achieve %.4f bpw\n", __func__, params->target_bpw);

            bpw_overrides = target_bpw_type(ml, model, tensors, mapped, values_data, activations_data, params, nthread);
        } else {
            LLAMA_LOG_WARN("%s: --target-bpw requires an imatrix but none was provided, option will be ignored\n", __func__);
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
    new_ofstream(0);
    for (const auto * it : tensors) {
        const size_t  align  = GGUF_DEFAULT_ALIGNMENT;
        const auto & weight = *it;
        ggml_tensor * tensor = weight.tensor;
        if (weight.idx != cur_split && params->keep_split) {
            close_ofstream();
            new_ofstream(weight.idx);
        }

        const std::string name = ggml_get_name(tensor);

        if (!ml.use_mmap) {
            if (read_data.size() < ggml_nbytes(tensor)) {
                read_data.resize(ggml_nbytes(tensor));
            }
            tensor->data = read_data.data();
        }
        ml.load_data_for(tensor);

        LLAMA_LOG_INFO("[%4d/%4d] %36s - [%s], type = %6s, ",
            ++idx, ml.n_tensors, ggml_get_name(tensor), llama_format_tensor_shape(tensor).c_str(), ggml_type_name(tensor->type));

        bool quantize = ggml_n_dims(tensor) >= 2 && is_quantizable(name, model.arch, params);
        quantize &= params->quantize_output_tensor || name != "output.weight";

        // do not quantize specific multimodal tensors
        quantize &= name.find(".position_embd.") == std::string::npos;

        ggml_type new_type;
        void * new_data;
        size_t new_size;

        if (quantize) {
            new_type = default_type;

            // get more optimal quantization type based on the tensor shape, layer, etc.
            if (!params->pure && (ggml_is_quantized(default_type) || params->target_bpw != -1.0f)) {
                int fallback = qs.n_fallback;
                new_type = llama_tensor_get_type(qs, new_type, tensor, ftype);

                // get quantization type overrides targeting a given bits per weight budget
                if (params->target_bpw != -1.0f && !bpw_overrides.empty()) {
                    const auto override = bpw_overrides.find(name);
                    if (override != bpw_overrides.end() && override->second != new_type) {
                        LLAMA_LOG_DEBUG("(bpw override %s) ", ggml_type_name(new_type));
                        new_type = override->second;
                    }
                }

                // unless the user specifies a type, and the tensor shape will not require fallback quantisation
                if (params->tensor_types && qs.n_fallback - fallback == 0) {
                    const std::vector<tensor_quantization> & tensor_types = *static_cast<const std::vector<tensor_quantization> *>(params->tensor_types);
                    const std::string tensor_name(tensor->name);
                    for (const auto & [tname, qtype] : tensor_types) {
                        if (std::regex pattern(tname); std::regex_search(tensor_name, pattern)) {
                            if  (qtype != new_type) {
                                LLAMA_LOG_DEBUG("(type override %s) ", ggml_type_name(new_type));
                                new_type = qtype; // if two or more types are specified for the same tensor, the last match wins
                            }
                        }
                    }
                }
            }
            if (params->token_embedding_type < GGML_TYPE_COUNT && strcmp(tensor->name, "token_embd.weight") == 0) {
                new_type = params->token_embedding_type;
            }
            if (params->output_tensor_type < GGML_TYPE_COUNT && strcmp(tensor->name, "output.weight") == 0) {
                new_type = params->output_tensor_type;
            }

            // If we've decided to quantize to the same type the tensor is already
            // in then there's nothing to do.
            quantize = tensor->type != new_type;
        }

        if (!quantize) {
            new_type = tensor->type;
            new_data = tensor->data;
            new_size = ggml_nbytes(tensor);
            LLAMA_LOG_INFO("size = %8.3f MiB\n", ggml_nbytes(tensor)/1024.0/1024.0);
        } else {
            const int64_t nelements = ggml_nelements(tensor);

            const float * imatrix = nullptr;
            if (values_data) {
                auto it = values_data->find(remap_imatrix(tensor->name, mapped));
                if (it == values_data->end()) {
                    LLAMA_LOG_INFO("\n====== %s: did not find weights for %s, ", __func__, tensor->name);
                } else {
                    if (it->second.size() == (size_t)tensor->ne[0]*tensor->ne[2]) {
                        imatrix = it->second.data();
                    } else {
                        LLAMA_LOG_INFO("\n====== %s: imatrix size %d is different from tensor size %d for %s\n", __func__,
                                int(it->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name);

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
            if ((new_type == GGML_TYPE_IQ2_XXS ||
                 new_type == GGML_TYPE_IQ2_XS  ||
                 new_type == GGML_TYPE_IQ2_S   ||
                 new_type == GGML_TYPE_IQ1_S   ||
                (new_type == GGML_TYPE_IQ1_M && strcmp(tensor->name, "token_embd.weight") && strcmp(tensor->name, "output.weight"))  ||
                (new_type == GGML_TYPE_Q2_K && params->ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S && strcmp(tensor->name, "token_embd.weight") != 0)) && !imatrix) {
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

            LLAMA_LOG_INFO("converting to %s .. ", ggml_type_name(new_type));
            fflush(stdout);

            if (work.size() < (size_t)nelements * 4) {
                work.resize(nelements * 4); // upper bound on size
            }
            new_data = work.data();

            const int64_t n_per_row = tensor->ne[0];
            const int64_t nrows = tensor->ne[1];

            static const int64_t min_chunk_size = 32 * 512;
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

                // TODO: temporary sanity check that the F16 -> MXFP4 is lossless
#if 0
                if (new_type == GGML_TYPE_MXFP4) {
                    auto * x = f32_data_03;

                    //LLAMA_LOG_INFO("nrows = %d, n_per_row = %d\n", nrows, n_per_row);
                    std::vector<float> deq(nrows*n_per_row);
                    const ggml_type_traits * qtype = ggml_get_type_traits(new_type);
                    qtype->to_float(new_data_03, deq.data(), deq.size());

                    double err = 0.0f;
                    for (int i = 0; i < (int) deq.size(); ++i) {
                        err += fabsf(deq[i] - x[i]);
                        //if (fabsf(deq[i] - x[i]) > 0.00001 && i < 256) {
                        if (deq[i] != x[i]) {
                            LLAMA_LOG_INFO("deq[%d] = %f, x[%d] = %f\n", i, deq[i], i, x[i]);
                        }
                    }
                    //LLAMA_LOG_INFO("err = %f\n", err);
                    GGML_ASSERT(err == 0.00000);
                }
#endif
            }
            LLAMA_LOG_INFO("size = %8.2f MiB -> %8.2f MiB\n", ggml_nbytes(tensor)/1024.0/1024.0, new_size/1024.0/1024.0);
        }
        total_size_org += ggml_nbytes(tensor);
        total_size_new += new_size;

        // update the gguf meta data as we go
        gguf_set_tensor_type(ctx_outs[cur_split].get(), name.c_str(), new_type);
        GGML_ASSERT(gguf_get_tensor_size(ctx_outs[cur_split].get(), gguf_find_tensor(ctx_outs[cur_split].get(), name.c_str())) == new_size);
        gguf_set_tensor_data(ctx_outs[cur_split].get(), name.c_str(), new_data);

        // write tensor data + padding
        fout.write((const char *) new_data, new_size);
        zeros(fout, GGML_PAD(new_size, align) - new_size);
    }
    close_ofstream();

    LLAMA_LOG_INFO("%s: model size  = %8.2f MiB\n", __func__, total_size_org/1024.0/1024.0);
    LLAMA_LOG_INFO("%s: quant size  = %8.2f MiB\n", __func__, total_size_new/1024.0/1024.0);

    if (qs.n_fallback > 0) {
        LLAMA_LOG_WARN("%s: WARNING: %d of %d tensor(s) required fallback quantization\n",
                __func__, qs.n_fallback, qs.n_k_quantized + qs.n_fallback);
    }
}

//
// interface implementation
//

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
        /*.imatrix                     =*/ nullptr,
        /*.activations                 =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.tensor_type                 =*/ nullptr,
        /*.prune_layers                =*/ nullptr,
        /*.target_bpw                  =*/ -1.0f,
        /*.keep_bpw_state              =*/ false,
        /*.bpw_state                   =*/ nullptr,
        /*.no_importance               =*/ false
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
