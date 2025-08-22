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
#include <random>
#include <regex>
#include <thread>
#include <unordered_map>

// Quantization types. Changes to this struct must be replicated in quantize.cpp
struct tensor_quantization {
    std::string name;
    ggml_type quant = GGML_TYPE_COUNT;
};

static bool is_iq(const enum ggml_type t) {
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
}

static bool is_iq(const enum llama_ftype t) {
    switch (t) {
        case LLAMA_FTYPE_MOSTLY_IQ1_S:
        case LLAMA_FTYPE_MOSTLY_IQ1_M:
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS:
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:
        case LLAMA_FTYPE_MOSTLY_IQ2_S:
        case LLAMA_FTYPE_MOSTLY_IQ2_M:
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS:
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:
        case LLAMA_FTYPE_MOSTLY_IQ3_S:
        case LLAMA_FTYPE_MOSTLY_IQ3_M:
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:
            return true;
        default:
            return false;
    }
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
                LLAMA_LOG_DEBUG("(blk.%d imatrix) ", p.first);
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

// Returns per-tensor type overrides to meet target BPW at lowest ppl
static std::unordered_map<std::string, ggml_type> target_bpw_type(
    llama_model_loader & ml,
    std::vector<no_init<uint8_t>> & buffer,
    const llama_model & model,
    const std::vector<const llama_model_loader::llama_tensor_weight *> & tensors,
    const std::map<int, std::string> & mapped,
    const std::unordered_map<std::string, std::vector<float>> * values_data,
    const std::unordered_map<std::string, std::vector<float>> * activations_data,
    const llama_model_quantize_params * params,
    int nthread
) {
    struct candidate_types {
        ggml_type type;
        float bpw;
        size_t bytes;
        float error;
    };

    struct tensor_info {
        const llama_model_loader::llama_tensor_weight * w = nullptr;
        std::vector<candidate_types> candidate = {};
        int choice = -1;
        float min_bpw = 0.0;
        float max_bpw = 0.0;
        size_t n_elements = 0;
    };

    constexpr ggml_type k_quants[] = {
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
// TODO: find better way to handle F16/BF16
#ifdef GGML_USE_METAL
        GGML_TYPE_F16
#else
        GGML_TYPE_BF16
#endif
    };

    constexpr ggml_type iq_quants[] = {
        GGML_TYPE_IQ1_S,
        GGML_TYPE_IQ1_M,
        GGML_TYPE_IQ2_XXS,
        GGML_TYPE_IQ2_XS,
        GGML_TYPE_IQ2_S,
        GGML_TYPE_IQ3_XXS,
        GGML_TYPE_IQ3_S,
        GGML_TYPE_IQ4_XS,
        GGML_TYPE_IQ4_NL,
        // TODO: add higher-precision fallbacks for IQ mixes to improve ppl if bpw budget allows it?
        GGML_TYPE_Q5_0,
        GGML_TYPE_Q5_1,
        GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K
    };

    auto get_values = [&](const std::string & tensor_name) -> const float * {
        if (!values_data) { return nullptr; }
        const auto it = values_data->find(remap_imatrix(tensor_name, mapped));
        if (it == values_data->end()) { return nullptr; }
        return it->second.data();
    };

    auto get_activations = [&](const std::string & tensor_name) -> const float * {
        if (!activations_data) { return nullptr; }
        const auto it = activations_data->find(remap_imatrix(tensor_name, mapped));
        if (it == activations_data->end()) { return nullptr; }
        return it->second.data();
    };

    auto tensor_bytes = [](const ggml_tensor * t, const ggml_type typ) -> size_t {
        const int64_t n_per_row = t->ne[0];
        const size_t row_sz = ggml_row_size(typ, n_per_row);
        const int64_t nrows = ggml_nrows(t);
        return (size_t)nrows * row_sz;
    };

    auto tensor_bpw = [&](const ggml_tensor * t, const ggml_type typ) -> double {
        const int64_t nelem = ggml_nelements(t);
        const size_t bytes = tensor_bytes(t, typ);
        return (double)bytes * 8.0 / (double)nelem;
    };

    auto is_compatible = [&](const ggml_tensor * t, const ggml_type typ) -> bool {
        const int64_t n_per_row = t->ne[0];
        const int64_t blck = ggml_blck_size(typ);
        if (blck <= 1) { return true; }
        return n_per_row % blck == 0;
    };

    auto make_compatible = [&](const ggml_tensor * t, const ggml_type typ) -> ggml_type {
        if (is_compatible(t, typ)) { return typ; }
        ggml_type fb = fallback_type(typ);
        if (is_compatible(t, fb)) { return fb; }
        return GGML_TYPE_F16;
    };

    auto name_tn = LLM_TN(model.arch);
    auto can_quantize = [&](const ggml_tensor * t) -> bool {
        // This list should be kept in sync with llama_tensor_quantize_impl()
        const std::string name = ggml_get_name(t);
        bool q = name.rfind("weight") == name.size() - 6;
        q &= ggml_n_dims(t) >= 2;
        q &= name.find("_norm.weight") == std::string::npos;
        q &= name.find("ffn_gate_inp.weight") == std::string::npos;
        q &= name.find("altup") == std::string::npos;
        q &= name.find("laurel") == std::string::npos;
        q &= name.find("per_layer_model_proj") == std::string::npos;
        q &= name != name_tn(LLM_TENSOR_POS_EMBD, "weight");
        q &= name != name_tn(LLM_TENSOR_TOKEN_TYPES, "weight");
        q &= name.find("ssm_conv1d.weight") == std::string::npos;
        q &= name.find("shortconv.conv.weight") == std::string::npos;
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
        q &= name.find("attn_rel_b.weight") == std::string::npos;
        q &= !params->only_copy;

        return q;
    };

    // Estimate error for a given type using a sampled subset of rows
    auto estimate_error = [&](const ggml_tensor * t,
        const ggml_type quant_type,
        const std::vector<float> & f32_sample,
        const std::vector<int64_t> & sample_rows_per_slice,
        const float * values_sample,
        const float * activations_sample,
        std::vector<uint8_t> & quantized_buffer,
        std::vector<float> & dequantized_buffer) -> double
    {
        const int64_t n_per_row = t->ne[0];
        const int64_t nrows = t->ne[1];
        const int64_t ne2 = t->ne[2] > 0 ? t->ne[2] : 1;

        const size_t sample_element_count = f32_sample.size();
        const size_t sample_row_count = sample_element_count / (size_t)n_per_row;
        if (sample_row_count == 0) { return 0.0; }

        const size_t row_size = ggml_row_size(quant_type, n_per_row);
        const size_t buffer_size = row_size * sample_row_count;
        if (quantized_buffer.size() < buffer_size) { quantized_buffer.resize(buffer_size); }
        if (dequantized_buffer.size() < sample_element_count) { dequantized_buffer.resize(sample_element_count); }

        std::vector row_sq_norm(sample_row_count, 0.0);
        std::vector bias_denominator_per_slice(ne2, 0.0);

        // Precompute bias denominator per slice
        const bool has_values = (values_sample != nullptr);
        const bool has_activations = (activations_sample != nullptr);
        if (has_activations) {
            for (int64_t s = 0; s < ne2; ++s) {
                const float * values = has_values ? values_sample + s * n_per_row : nullptr;
                const float * activations = activations_sample + s * n_per_row;
                double bias_denominator = 0.0;
                if (has_values) {
                    for (int64_t j = 0; j < n_per_row; ++j) {
                        const double a = activations[j];
                        bias_denominator += values[j] * a * a;
                    }
                } else {
                    for (int64_t j = 0; j < n_per_row; ++j) {
                        const double a = activations[j];
                        bias_denominator += a * a;
                    }
                }
                bias_denominator_per_slice[s] = bias_denominator;
            }
        }

        // Compute squared norms of sampled rows
        {
            size_t offset = 0;
            size_t row_idx = 0;
            for (int64_t s = 0; s < ne2; ++s) {
                const int64_t rs = sample_rows_per_slice[s];
                if (rs == 0) { continue; }

                const float * values = has_values ? values_sample + s * n_per_row : nullptr;

                for (int64_t r = 0; r < rs; ++r, ++row_idx) {
                    const float * row  = f32_sample.data() + offset;
                    double rsn = 0.0;
                    if (has_values) {
                        for (int64_t j = 0; j < n_per_row; ++j) {
                            const double v = values[j];
                            const double x = row[j];
                            rsn += v * x * x;
                        }
                    } else {
                        for (int64_t j = 0; j < n_per_row; ++j) {
                            const double x = row[j];
                            rsn += x * x;
                        }
                    }
                    row_sq_norm[row_idx] = rsn;
                    offset += (size_t)n_per_row;
                }
            }
        }

        // Quantize sampled rows slice-by-slice into quantized_buffer
        size_t quantised_offset = 0;
        size_t floats_offset = 0;
        for (int64_t slice = 0; slice < ne2; ++slice) {
            const int64_t rs = sample_rows_per_slice[slice];
            if (rs == 0) { continue; }

            const float * value = values_sample ? values_sample + slice * n_per_row : nullptr;
            (void)ggml_quantize_chunk(quant_type, f32_sample.data() + floats_offset, quantized_buffer.data() + quantised_offset, 0, rs, n_per_row, value);

            quantised_offset += row_size * (size_t)rs;
            floats_offset += (size_t)rs * (size_t)n_per_row;
        }

        // Dequantize into dequantized_buffer
        {
            const ggml_type_traits * traits = ggml_get_type_traits(quant_type);
            if (quant_type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *)quantized_buffer.data(), dequantized_buffer.data(), (int)sample_element_count);
            } else if (quant_type == GGML_TYPE_BF16) {
                ggml_bf16_to_fp32_row((const ggml_bf16_t *)quantized_buffer.data(), dequantized_buffer.data(), (int)sample_element_count);
            } else {
                if (!traits || !traits->to_float) {
                    LLAMA_LOG_WARN("%s: unsupported quantization type %s\n", __func__, ggml_type_name(quant_type));
                    return 1e35;
                }

                size_t done = 0;
                while (done < sample_element_count) {
                    const size_t chunk = std::min((size_t)n_per_row, sample_element_count - done);
                    traits->to_float(quantized_buffer.data() + done / n_per_row * row_size, dequantized_buffer.data() + done, (int)chunk);
                    done += chunk;
                }
            }
        }

        // Compute error
        size_t offset = 0;
        size_t row_idx = 0;
        double total_err = 0.0;
        for (int64_t slice = 0; slice < ne2; ++slice) {
            const int64_t rs = sample_rows_per_slice[slice];
            if (rs == 0) { continue; }

            const float * values = has_values ? values_sample + slice * n_per_row : nullptr;
            const float * activations = has_activations ? activations_sample + slice * n_per_row : nullptr;
            const double bias_denominator = has_activations ? bias_denominator_per_slice[slice] : 0.0;
            double slice_err = 0.0;
            for (int64_t r = 0; r < rs; ++r, ++row_idx) {
                const float * x = f32_sample.data() + offset;
                const float * y = dequantized_buffer.data() + offset;
                double weighted_mse = 0.0;
                double bias_numerator  = 0.0;
                if (values && activations) {
                    for (int64_t j = 0; j < n_per_row; ++j) {
                        const double v = values[j];
                        const double e = y[j] - x[j];
                        const double a = activations[j];
                        weighted_mse += v * e * e;
                        bias_numerator += v * e * a;
                    }
                } else if (values) {
                    for (int64_t j = 0; j < n_per_row; ++j) {
                        const double v = values[j];
                        const double e = y[j] - x[j];
                        weighted_mse += v * e * e;
                    }
                } else if (activations) {
                    for (int64_t j = 0; j < n_per_row; ++j) {
                        const double e = y[j] - x[j];
                        const double a = activations[j];
                        weighted_mse += e * e;
                        bias_numerator += e * a;
                    }
                } else {
                    for (int64_t j = 0; j < n_per_row; ++j) {
                        const double e = y[j] - x[j];
                        weighted_mse += e * e;
                    }
                }

                double err_numerator = weighted_mse;
                constexpr double epsilon = 1e-12;
                constexpr float bias_lambda = 1.0;
                //bias_lambda defines the weight of the bias term in the weigthed MSE error function
                // 0.0 means no bias (standard MSE) 1.0 means equal weight for bias and error,
                // 2.0 means twice as much weight for bias, etc. Default is 1.0.
                if (activations && bias_lambda != 0.0) {
                    const double proj = bias_numerator * bias_numerator / (bias_denominator + epsilon);
                    err_numerator += bias_lambda * proj;
                }

                const double err_denominator = row_sq_norm[row_idx] + epsilon;
                const double row_err = err_numerator / err_denominator;
                slice_err += row_err;
                offset += (size_t)n_per_row;
            }

            // scale to full rows (nrows)
            const double scale_rows = (double)nrows / std::max(1.0, (double)rs);
            total_err += slice_err * scale_rows;
        }

        return std::isfinite(total_err) ? total_err : 1e35;
    };

    std::vector<tensor_info> all;
    all.reserve(tensors.size());
    for (const auto * tw : tensors) {
        std::vector<std::thread> workers;
        workers.reserve(std::max(1, nthread));
        ggml_tensor * t = tw->tensor;
        const std::string name = ggml_get_name(t);
        if (!can_quantize(t)) { continue; }

        LLAMA_LOG_INFO("\t%s: - processing tensor %45s \t(%12d elements)\n", __func__, name.c_str(), (int)ggml_nelements(t));
        if (!ml.use_mmap) {
            if (buffer.size() < ggml_nbytes(t)) { buffer.resize(ggml_nbytes(t)); }
            t->data = buffer.data();
        }
        ml.load_data_for(t);

        // Dequantize only sampled rows into f32_sample
        const int64_t n_per_row = t->ne[0];
        const int64_t nrows_total = t->ne[1];
        const int64_t ne2 = t->ne[2] > 0 ? t->ne[2] : 1;

        // Larger sample_rows_per_expert values may result in more accurate error estimates, but will take longer to compute
        int sample_rows_per_expert = 512;
        std::vector<float> f32_sample;
        f32_sample.reserve((size_t)ne2 * (size_t)std::min<int64_t>(nrows_total, sample_rows_per_expert) * (size_t)n_per_row);

        // deterministic sampling seed based on tensor name + fixed constant
        std::mt19937 rng(std::hash<std::string>{}(name) ^0xeabada55cafed00d);
        std::vector<int64_t> sample_rows_per_slice(ne2, 0);
        const int64_t sample_rows_max = std::max<int64_t>(1, std::min<int64_t>(nrows_total, sample_rows_per_expert));
        const int64_t stride = std::max<int64_t>(1, nrows_total / sample_rows_max);
        std::vector<float> row_buffer(n_per_row);
        const ggml_type src_type = t->type;
        const ggml_type_traits *src_traits = ggml_get_type_traits(src_type);
        const bool src_is_quant = ggml_is_quantized(src_type);
        const size_t src_row_sz = ggml_row_size(src_type, n_per_row);
        for (int64_t slice = 0; slice < ne2; ++slice) {
            int64_t current_sampled_rows = 0;
            int64_t offset = 0;
            if (stride > 1) {
                std::uniform_int_distribution<int64_t> dist(0, stride - 1);
                offset = dist(rng);
            }

            for (int64_t r = offset; r < nrows_total && current_sampled_rows < sample_rows_max; r += stride) {
                if (src_type == GGML_TYPE_F32) {
                    const float * src_row = (const float *)t->data + slice * (n_per_row * nrows_total) + r * n_per_row;
                    f32_sample.insert(f32_sample.end(), src_row, src_row + n_per_row);
                } else if (src_type == GGML_TYPE_F16) {
                    const ggml_fp16_t * src_row = (const ggml_fp16_t *)((const uint8_t *)t->data + slice * (src_row_sz * nrows_total) + r * src_row_sz);
                    ggml_fp16_to_fp32_row(src_row, row_buffer.data(), (int)n_per_row);
                    f32_sample.insert(f32_sample.end(), row_buffer.begin(), row_buffer.end());
                } else if (src_type == GGML_TYPE_BF16) {
                    const ggml_bf16_t * src_row = (const ggml_bf16_t *)((const uint8_t *)t->data + slice * (src_row_sz * nrows_total) + r * src_row_sz);
                    ggml_bf16_to_fp32_row(src_row, row_buffer.data(), (int)n_per_row);
                    f32_sample.insert(f32_sample.end(), row_buffer.begin(), row_buffer.end());
                } else if (src_is_quant) {
                    const uint8_t * qrow = (const uint8_t *)t->data + slice * (src_row_sz * nrows_total) + r * src_row_sz;
                    if (!src_traits || !src_traits->to_float) {
                        throw std::runtime_error(format("cannot dequantize type %s for sampling", ggml_type_name(src_type)));
                    }
                    src_traits->to_float(qrow, row_buffer.data(), (int)n_per_row);
                    f32_sample.insert(f32_sample.end(), row_buffer.begin(), row_buffer.end());
                } else {
                    throw std::runtime_error(format("unsupported src type %s for sampling", ggml_type_name(src_type)));
                }
                ++current_sampled_rows;
            }
            sample_rows_per_slice[slice] = current_sampled_rows;
        }

        auto copy_or_broadcast = [&](const float *src, size_t src_sz, std::vector<float> &dst) {
            const size_t want = (size_t)ne2 * (size_t)n_per_row;
            dst.clear();
            if (!src || src_sz == 0) { return; }

            if (src_sz == want) {
                dst.resize(want);
                std::memcpy(dst.data(), src, want * sizeof(float));
            } else if (src_sz == (size_t)n_per_row) {
                dst.resize(want);
                for (int64_t s = 0; s < ne2; ++s) {
                    std::memcpy(dst.data() + s * n_per_row, src, n_per_row * sizeof(float));
                }
            } else {
                // Mismatch – safer to skip using it for this tensor
                LLAMA_LOG_WARN("%s: side data size mismatch for %s: got %zu, expected %zu or %zu; ignoring\n",
                    __func__, name.c_str(), src_sz, (size_t)n_per_row, want);
            }
        };

        const float * values_all = get_values(name);
        const float * activations_all = get_activations(name);
        std::vector<float> values_sample;
        std::vector<float> activations_sample;
        if (values_all) {
            // get size from the map (not just the raw pointer)
            auto itv = values_data->find(remap_imatrix(name, mapped));
            const size_t sz = itv == values_data->end() ? 0 : itv->second.size();
            copy_or_broadcast(values_all, sz, values_sample);
        }
        if (activations_all) {
            auto ita = activations_data->find(remap_imatrix(name, mapped));
            const size_t sz = ita == activations_data->end() ? 0 : ita->second.size();
            copy_or_broadcast(activations_all, sz, activations_sample);
        }

        const int64_t nelem = ggml_nelements(t);
        tensor_info info;
        info.w = tw;
        info.n_elements = nelem;

        // Prepare scratch buffers sized for the largest candidate row size
        size_t total_sampled_rows = f32_sample.size() / n_per_row;

        // Build list of candidate types first (compatible ones)
        std::vector<ggml_type> quant_candidates;
        if (is_iq(params->ftype)) {
            quant_candidates.assign(std::begin(iq_quants), std::end(iq_quants));
        } else {
            quant_candidates.assign(std::begin(k_quants), std::end(k_quants));
        }

        // Compute maximum row size among compatible candidates (to size quantized_buffer once)
        size_t max_row_sz = 0;
        const bool has_valid_imatrix = !values_sample.empty() && values_sample.size() == (size_t)ne2 * (size_t)n_per_row;
        std::vector<ggml_type> compatible_candidates;
        compatible_candidates.reserve(quant_candidates.size());
        for (ggml_type ts_type : quant_candidates) {
            if (is_iq(ts_type) && !has_valid_imatrix) {
                LLAMA_LOG_WARN("%s: skipping IQ quantization for %s, no or mismatched imatrix provided\n", __func__, name.c_str());
                continue;
            }
            ggml_type tt = make_compatible(t, ts_type);
            if (!is_compatible(t, tt)) { continue; }
            compatible_candidates.push_back(tt);
            max_row_sz = std::max(max_row_sz, ggml_row_size(tt, n_per_row));
        }

        std::sort(compatible_candidates.begin(), compatible_candidates.end());
        compatible_candidates.erase(std::unique(compatible_candidates.begin(), compatible_candidates.end()), compatible_candidates.end());

        // Now evaluate candidates
        std::vector<candidate_types> eval_candidates(compatible_candidates.size());
        const float *values = values_sample.empty() ? nullptr : values_sample.data();
        const float *activations = activations_sample.empty() ? nullptr : activations_sample.data();
        std::vector<uint8_t> quantized_buffer(max_row_sz * total_sampled_rows);
        std::vector<float> dequantised_buffer(f32_sample.size());
        int n_eval_threads = std::max(1, std::min<int>(nthread, (int)compatible_candidates.size()));
        std::atomic<size_t> cidx{0};
        std::vector<std::thread> eval_workers;
        eval_workers.reserve(n_eval_threads);
        for (int ti = 0; ti < n_eval_threads; ++ti) {
            eval_workers.emplace_back([&] {
                // thread-local scratch
                std::vector<uint8_t> tl_quantized_buffer(quantized_buffer.size());
                std::vector<float>   tl_dequantised_buffer(dequantised_buffer.size());

                for (;;) {
                    const size_t i = cidx.fetch_add(1, std::memory_order_relaxed);
                    if (i >= compatible_candidates.size()) { break; }

                    const ggml_type tt = compatible_candidates[i];
                    const auto bpw = (float)tensor_bpw(t, tt);
                    const size_t bytes = tensor_bytes(t, tt);
                    const auto err = (float)estimate_error(t, tt, f32_sample, sample_rows_per_slice, values, activations, tl_quantized_buffer, tl_dequantised_buffer);
                    eval_candidates[i] = candidate_types{ tt, bpw, bytes, err };
                }
            });
        }

        for (auto &th : eval_workers) { th.join(); }

        for (auto &c : eval_candidates) {
            if (c.bytes > 0) { info.candidate.push_back(c); }
        }

        if (info.candidate.empty()) {
            // As a last resort, keep original type
            float bpw = ggml_nbytes(t) * 8.0f / nelem;
            info.candidate.push_back(candidate_types{ t->type, bpw, ggml_nbytes(t), 0.0 });
        }

        // Keep only the Pareto‑optimal candidates: if A has >= bytes and >= error than B, drop A.
        {
            std::vector<candidate_types> pruned;
            pruned.reserve(info.candidate.size());
            // Sort by bytes asc, error asc
            std::sort(info.candidate.begin(), info.candidate.end(), [](const candidate_types &a, const candidate_types &b) {
                if (a.bytes != b.bytes) { return a.bytes < b.bytes; }
                return a.error < b.error;
            });

            double best_err = std::numeric_limits<double>::infinity();
            size_t last_bytes = std::numeric_limits<size_t>::max();

            for (const auto &c : info.candidate) {
                if (c.error < best_err || c.bytes > last_bytes) {
                    pruned.push_back(c);
                    best_err = std::min(best_err, (double)c.error);
                    last_bytes = c.bytes;
                }
            }
            info.candidate.swap(pruned);
        }

        // Collapse candidates with identical storage size (bytes)
        {
            std::vector<candidate_types> unique;
            unique.reserve(info.candidate.size());
            // Sort by bpw asc, error asc, bytes asc
            std::sort(info.candidate.begin(), info.candidate.end(), [](const candidate_types & a, const candidate_types & b) {
                if (a.bpw != b.bpw) { return a.bpw < b.bpw; }
                if (a.error != b.error) { return a.error < b.error; }
                return a.bytes < b.bytes;
            });

            for (size_t i = 0; i < info.candidate.size();) {
                size_t          j    = i + 1;
                candidate_types best = info.candidate[i];
                // group same-byte entries, keep the one with the lowest error
                while (j < info.candidate.size() && info.candidate[j].bytes == info.candidate[i].bytes) {
                    if (info.candidate[j].error < best.error) {
                        best = info.candidate[j];
                    }
                    ++j;
                }
                unique.push_back(best);
                i = j;
            }
            info.candidate.swap(unique);
        }

        // Initialize choice at the smallest bpw candidate
        info.choice = 0;
        info.min_bpw = info.candidate.front().bpw;
        info.max_bpw = info.candidate.back().bpw;
        all.push_back(std::move(info));
    }

    if (all.empty()) { return {}; }

    // Greedy allocation from minimum bpw upward to reach target_bpw
    auto current_total_bytes = [&]() -> size_t {
        size_t b = 0;
        for (const auto & ti : all) {
            b += ti.candidate[ti.choice].bytes;
        }

        return b;
    };

    auto total_weights = [&]() -> size_t {
        size_t w = 0;
        for (const auto & ti : all) {
            w += ti.n_elements;
        }

        return w;
    };

    const size_t tw = total_weights();
    auto current_bpw = [&]() -> double {
        return (double)current_total_bytes() * 8.0f / (double)tw;
    };

    // Precompute current bpw
    double bpw_now = current_bpw();

    float target_bpw = params->target_bpw;
    // If minimal bpw is already above the target, we're constrained by the tensor's shape; return closest (min bpw)
    if (bpw_now >= target_bpw) {
        std::unordered_map<std::string, ggml_type> overrides;
        for (const auto & ti : all) {
            overrides[ggml_get_name(ti.w->tensor)] = ti.candidate[ti.choice].type;
        }

        return overrides;
    }

    struct upgrade {
        int idx;
        int next;
        double err;
        size_t delta_bytes;
        double ratio;
    };

    // Find next strictly-larger candidate index for a tensor
    auto next_distinct_idx = [&](const tensor_info & ti) -> int {
        const auto & cand = ti.candidate;
        const auto & cur  = cand[ti.choice];
        int j = ti.choice + 1;
        while (j < (int)cand.size() && cand[j].bytes == cur.bytes) {
            ++j;
        }

        return j < (int)cand.size() ? j : -1;
    };

    auto recompute_best_upgrade = [&]() -> upgrade {
        const double eps = 1e-12;
        upgrade best{ -1, -1, 0.0, 0, -1.0 };
        for (int i = 0; i < (int) all.size(); ++i) {
            const auto & ti = all[i];
            if (ti.choice >= (int)ti.candidate.size() - 1) { continue; }

            const int j = next_distinct_idx(ti);
            if (j < 0) { continue; }

            const auto & cur = ti.candidate[ti.choice];
            const auto & nxt = ti.candidate[j];

            const size_t delta_bytes = nxt.bytes - cur.bytes;
            if (delta_bytes == 0) { continue; }

            double err = cur.error - nxt.error;
            err = std::max(err, 0.0);

            double ratio = err / (double)(delta_bytes * 8ull);
            if (ratio > best.ratio + eps || (std::abs(ratio - best.ratio) <= eps && delta_bytes < best.delta_bytes)) {
                best = upgrade{ i, j, err, delta_bytes, ratio };
            }
        }

        return best;
    };

    while (true) {
        upgrade up = recompute_best_upgrade();
        if (up.idx < 0) { break; }

        size_t now_bytes = current_total_bytes();
        size_t next_bytes = now_bytes + up.delta_bytes;
        double bpw_next = (double)next_bytes * 8.0 / (double)tw;
        if (bpw_next <= target_bpw + 1e-12) {
            all[up.idx].choice = up.next;
            bpw_now = bpw_next;
        } else {
            break;
        }
    }

    // We might still be below target so we try to find the best upgrade one last time
    {
        upgrade best_over{ -1, -1, 0.0, 0, -1.0 };
        double  best_over_gap = 1e300;
        double  under_gap = target_bpw - bpw_now;
        size_t now_bytes = current_total_bytes();
        for (int i = 0; i < (int) all.size(); ++i) {
            const auto & ti = all[i];
            if (ti.choice >= (int)ti.candidate.size() - 1) { continue; }

            int j = next_distinct_idx(ti);
            if (j < 0) { continue; }

            const auto & cur = ti.candidate[ti.choice];
            const auto & nxt = ti.candidate[j];
            size_t delta_bytes = nxt.bytes - cur.bytes;
            if (delta_bytes == 0) { continue; }

            size_t over_bytes = now_bytes + delta_bytes;
            double bpw_over = (double)over_bytes * 8.0 / (double)tw;
            double err = cur.error - nxt.error;
            if (err < 0.0) { err = 0.0; }
            double ratio = err / (double)(delta_bytes * 8ull);

            double over_gap = std::abs(bpw_over - (double)target_bpw);
            if (over_gap < best_over_gap - 1e-12 || (std::abs(over_gap - best_over_gap) <= 1e-12 && ratio > best_over.ratio)) {
                best_over_gap = over_gap;
                best_over = upgrade{ i, j, err, delta_bytes, ratio };
            }
        }

        if (best_over.idx >= 0) {
            if (best_over_gap < under_gap) {
                all[best_over.idx].choice = best_over.next;
            }
        }
    }

    // Build the override map
    std::unordered_map<std::string, ggml_type> overrides;
    LLAMA_LOG_INFO("%s: - estimated tensor quantization mix to achieve %.4f bpw at lowest ppl\n", __func__, target_bpw);
    for (const auto & ti : all) {
        LLAMA_LOG_INFO("\t%s: %45s - \t%8s, \t%1.4f bpw,\terror: %.4f\n",
            __func__, ggml_get_name(ti.w->tensor), ggml_type_name(ti.candidate[ti.choice].type), ti.candidate[ti.choice].bpw, ti.candidate[ti.choice].error);
        overrides[ggml_get_name(ti.w->tensor)] = ti.candidate[ti.choice].type;
    }

    return overrides;
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
    llama_model_loader ml(fname_inp, splits, use_mmap, /*check_tensors*/ true, kv_overrides, nullptr);
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
            LLAMA_LOG_INFO("================================ Have weights data with %d entries\n",int(values_data->size()));
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
            LLAMA_LOG_INFO("================================ Have activations data with %d entries\n",int(activations_data->size()));
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
                gguf_set_val_u32(ctx_out.get(), o.key, (uint32_t)abs(o.val_i64));
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
    int pruned_attention_w = 0;

    // make a list of weights
    std::vector<const llama_model_loader::llama_tensor_weight *> tensors;
    tensors.reserve(ml.weights_map.size());
    for (const auto & it : ml.weights_map) {
        const std::string remapped_name(remap_layer(it.first, prune_list, mapped, blk_id));
        if (remapped_name.empty()) {
            if (it.first.find("attn_v.weight") != std::string::npos ||
                it.first.find("attn_qkv.weight") != std::string::npos ||
                it.first.find("attn_kv_b.weight") != std::string::npos) {
                    pruned_attention_w++;
            }
            LLAMA_LOG_DEBUG("%s: pruning tensor %s\n", __func__, it.first.c_str());
            continue;
        } else if (remapped_name != it.first) {
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

    // sanity checks for models that have attention layers
    if (qs.n_attention_wv != 0)
    {
        const auto & n_head_kv_iter = model.hparams.n_head_kv_arr.begin();
        // attention layers have a non-zero number of kv heads
        int32_t n_attn_layer = model.hparams.n_layer - std::count(n_head_kv_iter, n_head_kv_iter + model.hparams.n_layer, 0);
        if (llama_model_has_encoder(&model)) {
            n_attn_layer *= 3;
        }
        GGML_ASSERT((qs.n_attention_wv == n_attn_layer - pruned_attention_w) && "n_attention_wv is unexpected");
    }

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
    if (params->target_bpw != -1.0f) {
        LLAMA_LOG_INFO("%s: computing tensor quantization mix to achieve %.3f bpw at lowest ppl - this opearation may take some time\n", __func__, params->target_bpw);
        bpw_overrides = target_bpw_type(ml, read_data, model, tensors, mapped, values_data, activations_data, params, nthread);
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

        // This used to be a regex, but <regex> has an extreme cost to compile times.
        bool quantize = name.rfind("weight") == name.size() - 6; // ends with 'weight'?

        // quantize only 2D and 3D tensors (experts)
        quantize &= (ggml_n_dims(tensor) >= 2);

        // do not quantize norm tensors
        quantize &= name.find("_norm.weight") == std::string::npos;

        quantize &= params->quantize_output_tensor || name != "output.weight";
        quantize &= !params->only_copy;

        // do not quantize expert gating tensors
        // NOTE: can't use LLM_TN here because the layer number is not known
        quantize &= name.find("ffn_gate_inp.weight") == std::string::npos;

        // these are very small (e.g. 4x4)
        quantize &= name.find("altup")  == std::string::npos;
        quantize &= name.find("laurel") == std::string::npos;

        // these are not too big so keep them as it is
        quantize &= name.find("per_layer_model_proj") == std::string::npos;

        // do not quantize positional embeddings and token types (BERT)
        quantize &= name != LLM_TN(model.arch)(LLM_TENSOR_POS_EMBD,    "weight");
        quantize &= name != LLM_TN(model.arch)(LLM_TENSOR_TOKEN_TYPES, "weight");

        // do not quantize Mamba's small yet 2D weights
        // NOTE: can't use LLM_TN here because the layer number is not known
        quantize &= name.find("ssm_conv1d.weight") == std::string::npos;
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

        ggml_type new_type;
        void * new_data;
        size_t new_size;

        if (quantize) {
            new_type = default_type;

            // get more optimal quantization type based on the tensor shape, layer, etc.
            if (!params->pure && ggml_is_quantized(default_type)) {
                int fallback = qs.n_fallback;
                new_type = llama_tensor_get_type(qs, new_type, tensor, ftype);
                // get bpw override
                const auto override = bpw_overrides.find(name);
                if (override != bpw_overrides.end()) { new_type = override->second; }
                // unless the user specifies a type, and the tensor geometry will not require fallback quantisation
                if (params->tensor_types && qs.n_fallback - fallback == 0) {
                    const std::vector<tensor_quantization> & tensor_types = *static_cast<const std::vector<tensor_quantization> *>(params->tensor_types);
                    const std::string tensor_name(tensor->name);
                    for (const auto & [tname, qtype] : tensor_types) {
                        if (std::regex pattern(tname); std::regex_search(tensor_name, pattern)) {
                            if  (qtype != new_type) {
                                LLAMA_LOG_DEBUG("(overriding %s) ", ggml_type_name(new_type));
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
            LLAMA_LOG_INFO("size = %8.3f MB\n", ggml_nbytes(tensor)/1024.0/1024.0);
        } else {
            const int64_t nelements = ggml_nelements(tensor);

            const float * imatrix = nullptr;
            if (values_data) {
                auto it = values_data->find(remap_imatrix(tensor->name, mapped));
                if (it == values_data->end()) {
                    LLAMA_LOG_INFO("\n====== %s: did not find weights for %s\n", __func__, tensor->name);
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

    LLAMA_LOG_INFO("%s: model size  = %8.2f MB\n", __func__, total_size_org/1024.0/1024.0);
    LLAMA_LOG_INFO("%s: quant size  = %8.2f MB\n", __func__, total_size_new/1024.0/1024.0);

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
        /*.target_bpw                  =*/ -1.0f
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
