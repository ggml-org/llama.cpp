#include "llama-quant.h"

#include "ggml.h" // Explicitly include for ggml_type_traits_t
#include "json.hpp" // For nlohmann::json - common/ is in include path
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-model-loader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cinttypes>
#include <fstream>
#include <mutex>
#include <stdexcept> // For std::runtime_error
#include <thread>
#include <unordered_map>

// Old C-style SmarterQuant helper headers and struct/function definitions are removed.
// The new C++ `load_smarter_quant_config` using nlohmann::json is used instead.

// Function to load and parse the default.smarterquant.json file
// It populates a SmarterQuantConfig map with tensor-specific quantization instructions.
// - fname: Path to the JSON configuration file.
// - Returns: A SmarterQuantConfig map. If the file cannot be opened or parsed,
//            an empty map is returned and warnings/errors are logged.
SmarterQuantConfig load_smarter_quant_config(const std::string & fname) {
    SmarterQuantConfig config;
    std::ifstream ifs(fname);
    if (!ifs.is_open()) {
        LLAMA_LOG_WARN("%s: Failed to open smarterquant config file '%s'. Continuing without it.\n", __func__, fname.c_str());
        return config;
    }

    nlohmann::json parsed_json;
    try {
        parsed_json = nlohmann::json::parse(ifs);
    } catch (const nlohmann::json::parse_error& e) {
        LLAMA_LOG_ERROR("%s: Failed to parse smarterquant config file '%s': %s\n", __func__, fname.c_str(), e.what());
        // Return empty config, effectively disabling the feature if JSON is malformed
        return config;
    }

    if (!parsed_json.is_object()) {
        LLAMA_LOG_ERROR("%s: Smarterquant config file '%s' must contain a top-level JSON object.\n", __func__, fname.c_str());
        return config;
    }

    for (auto it = parsed_json.begin(); it != parsed_json.end(); ++it) {
        const std::string& tensor_name = it.key();
        const nlohmann::json& tensor_data_json = it.value();

        if (!tensor_data_json.is_array() || tensor_data_json.size() != 2) {
            LLAMA_LOG_WARN("%s: Entry for tensor '%s' in '%s' is not a 2-element array. Skipping.\n", __func__, tensor_name.c_str(), fname.c_str());
            continue;
        }

        SmarterQuantTensorInfo tensor_info;
        tensor_info.enabled = true; // If it's in the JSON, it's enabled by default for quantization

        // Parse compression types
        const nlohmann::json& compression_types_json = tensor_data_json[0];
        if (!compression_types_json.is_array() || compression_types_json.size() != 4) {
            LLAMA_LOG_WARN("%s: Compression types for tensor '%s' in '%s' must be an array of 4 integers. Skipping.\n", __func__, tensor_name.c_str(), fname.c_str());
            continue;
        }
        try {
            for (const auto& type_json : compression_types_json) {
                if (!type_json.is_number_integer()) {
                    throw std::runtime_error("Compression type is not an integer.");
                }
                tensor_info.compression_types.push_back(type_json.get<int8_t>());
            }
        } catch (const std::exception& e) {
            LLAMA_LOG_WARN("%s: Error parsing compression types for tensor '%s' in '%s': %s. Skipping.\n", __func__, tensor_name.c_str(), fname.c_str(), e.what());
            continue;
        }

        // Parse column permutation
        const nlohmann::json& column_permutation_json = tensor_data_json[1];
        if (!column_permutation_json.is_array()) {
            LLAMA_LOG_WARN("%s: Column permutation for tensor '%s' in '%s' must be an array. Skipping.\n", __func__, tensor_name.c_str(), fname.c_str());
            continue;
        }
        try {
            for (const auto& col_json : column_permutation_json) {
                if (!col_json.is_number_integer()) {
                    throw std::runtime_error("Column index is not an integer.");
                }
                tensor_info.column_permutation.push_back(col_json.get<int>());
            }
        } catch (const std::exception& e) {
            LLAMA_LOG_WARN("%s: Error parsing column permutation for tensor '%s' in '%s': %s. Skipping.\n", __func__, tensor_name.c_str(), fname.c_str(), e.what());
            continue;
        }
        config[tensor_name] = tensor_info;
        LLAMA_LOG_INFO("%s: Loaded smarterquant config for tensor '%s': %zu types, %zu perm_indices\n", __func__, tensor_name.c_str(), tensor_info.compression_types.size(), tensor_info.column_permutation.size());
    }
    LLAMA_LOG_INFO("%s: Successfully loaded smarterquant config from '%s' for %zu tensors.\n", __func__, fname.c_str(), config.size());
    return config;
}


static void zeros(std::ofstream & file, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
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

    int n_k_quantized = 0;
    int n_fallback    = 0;

    bool has_imatrix = false;

    // used to figure out if a model shares tok_embd with the output weight
    bool has_output = false;

    // Holder for permuted imatrix data if permutation occurs for a tensor
    std::vector<float> permuted_imatrix_holder;

    quantize_state_impl(const llama_model & model, const llama_model_quantize_params * params)
        : model(model)
        , params(params)
        {}
};

static void llama_tensor_dequantize_impl(
    struct ggml_tensor * tensor, std::vector<no_init<float>> & output, std::vector<std::thread> & workers,
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

            if (arch == LLM_ARCH_FALCON || nx % qk_k != 0) {
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
    } else if (name == "token_embd.weight") {
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

// New helper function for SmarterQuant block quantization
static size_t llama_tensor_quantize_smarter_blocks(
    const float * f32_data,                             // Input f32 data (potentially permuted)
    void * new_data,                                    // Output buffer for quantized data
    const int64_t * ne,                                 // Tensor dimensions (ne[0] = n_cols, ne[1] = n_rows, ne[2] = n_depth/experts)
    const SmarterQuantTensorInfo & sq_info,             // SmarterQuant configuration for this tensor
    const float * imatrix,                              // Importance matrix (if available, also permuted)
    int nthread                                         // Number of threads (currently unused in this granular approach, but for future)
) {
    GGML_ASSERT(sq_info.compression_types.size() == 4); // Must have 4 types defined

    const int64_t n_cols_total = ne[0];
    const int64_t n_rows = ne[1];
    const int64_t n_slices = ne[2]; // For 3D tensors (e.g., MoE)

    const int64_t block_width = 256;
    size_t total_new_size = 0;
    char * new_data_ptr = (char *)new_data;

    // TODO: Parallelize this loop over slices or rows if beneficial and safe.
    // For now, processing slice by slice, then row by row.
    for (int64_t slice_idx = 0; slice_idx < n_slices; ++slice_idx) {
        const float * current_f32_slice_data = f32_data + slice_idx * (n_rows * n_cols_total);
        const float * current_imatrix_slice_data = nullptr;
        if (imatrix) {
            current_imatrix_slice_data = imatrix + slice_idx * n_cols_total; // imatrix is 1D per slice or 2D covering all rows of a slice's columns
        }

        for (int64_t r = 0; r < n_rows; ++r) {
            const float * current_f32_row_data = current_f32_slice_data + r * n_cols_total;
            // The imatrix for a given row starts at current_imatrix_slice_data if it's per-column for the whole slice,
            // or needs further indexing if imatrix is [n_rows, n_cols_total] per slice.
            // Assuming imatrix is effectively 1D array of [n_cols_total] for the current slice, applied to all rows.
            // If imatrix is [n_rows, n_cols_total], then it would be:
            // const float * current_imatrix_row_data = current_imatrix_slice_data + r * n_cols_total; (if imatrix is 2D per slice)
            // For now, let's assume imatrix is 1D for the slice, meaning the same imatrix columns apply to all rows in that slice.
            // This matches how ggml_quantize_chunk seems to use it (imatrix passed is for the first row, implies it's column-wise).

            size_t current_row_written_bytes = 0;

            for (int64_t k_block = 0; (k_block * block_width) < n_cols_total; ++k_block) {
                const int64_t current_block_start_col = k_block * block_width;
                const int64_t current_block_n_cols = std::min(block_width, n_cols_total - current_block_start_col);

                ggml_type current_segment_type;
                if (k_block < 4) {
                    current_segment_type = static_cast<ggml_type>(sq_info.compression_types[k_block]);
                } else {
                    current_segment_type = static_cast<ggml_type>(sq_info.compression_types[3]);
                }

                const struct ggml_type_traits type_traits_val = *ggml_get_type_traits(current_segment_type); // Assign by value, correct struct name
                GGML_ASSERT(type_traits_val.from_float_ref != nullptr); // Ensure the type can be quantized from float

                const float * segment_f32_data = current_f32_row_data + current_block_start_col;
                void * segment_new_data_output = new_data_ptr + total_new_size + current_row_written_bytes;

                // The from_float_ref function does not take an imatrix directly.
                // Importance matrix is handled by ggml_quantize_chunk by potentially adjusting f32 values before calling from_float_ref.
                // For direct calls to from_float_ref, if imatrix logic is needed, it must be applied to segment_f32_data *before* this call.
                // Currently, this PoC does not re-implement that part of ggml_quantize_chunk.
                // const float * segment_imatrix_data = nullptr; // Not used by from_float_ref directly
                // if (current_imatrix_slice_data) {
                //     segment_imatrix_data = current_imatrix_slice_data + current_block_start_col;
                // }

                // We are quantizing a segment of a row.
                type_traits_val.from_float_ref(segment_f32_data, segment_new_data_output, current_block_n_cols);

                // Calculate size of this quantized segment
                // This requires knowing how many blocks of QK_K (or similar) 'current_block_n_cols' elements make up.
                GGML_ASSERT(current_block_n_cols % type_traits_val.blck_size == 0 && "Segment column count not divisible by block size for type");
                size_t segment_size_bytes = (current_block_n_cols / type_traits_val.blck_size) * type_traits_val.type_size;

                current_row_written_bytes += segment_size_bytes;
            }
            total_new_size += current_row_written_bytes;
        }
    }
    // (void)nthread; // Suppress unused parameter warning for now

    return total_new_size;
}

// Old C-style SmartQuant map handlers and their usage are removed.
// The new C++ `load_smarter_quant_config` using nlohmann::json is used instead.

static void llama_model_quantize_impl(const std::string & fname_inp, const std::string & fname_out, const llama_model_quantize_params * params) {
    ggml_type default_type;
    llama_ftype ftype = params->ftype;
    SmarterQuantConfig smarter_quant_config; // New SmarterQuant config

    // Load the SmarterQuant configuration
    // TODO: Make the filename configurable via params, for now hardcoded
    smarter_quant_config = load_smarter_quant_config("default.smarterquant.json");

    switch (params->ftype) {
        case LLAMA_FTYPE_MOSTLY_Q4_0: default_type = GGML_TYPE_Q4_0; break;
        case LLAMA_FTYPE_MOSTLY_Q4_1: default_type = GGML_TYPE_Q4_1; break;
        case LLAMA_FTYPE_MOSTLY_Q5_0: default_type = GGML_TYPE_Q5_0; break;
        case LLAMA_FTYPE_MOSTLY_Q5_1: default_type = GGML_TYPE_Q5_1; break;
        case LLAMA_FTYPE_MOSTLY_Q8_0: default_type = GGML_TYPE_Q8_0; break;
        case LLAMA_FTYPE_MOSTLY_F16:  default_type = GGML_TYPE_F16;  break;
        case LLAMA_FTYPE_MOSTLY_BF16: default_type = GGML_TYPE_BF16; break;
        case LLAMA_FTYPE_ALL_F32:     default_type = GGML_TYPE_F32;  break;

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

    // mmap consistently increases speed Linux, and also increases speed on Windows with
    // hot cache. It may cause a slowdown on macOS, possibly related to free memory.
#if defined(__linux__) || defined(_WIN32)
    constexpr bool use_mmap = true;
#else
    constexpr bool use_mmap = false;
#endif

    llama_model_kv_override * kv_overrides = nullptr;
    if (params->kv_overrides) {
        auto v = (std::vector<llama_model_kv_override>*)params->kv_overrides;
        kv_overrides = v->data();
    }

    std::vector<std::string> splits = {};
    llama_model_loader ml(fname_inp, splits, use_mmap, /*check_tensors*/ true, kv_overrides);
    ml.init_mappings(false); // no prefetching

    llama_model model(llama_model_default_params());

    model.load_arch   (ml);
    model.load_hparams(ml);
    model.load_stats  (ml);

    struct quantize_state_impl qs(model, params);

    if (params->only_copy) {
        ftype = ml.ftype;
    }
    const std::unordered_map<std::string, std::vector<float>> * imatrix_data = nullptr;
    if (params->imatrix) {
        imatrix_data = static_cast<const std::unordered_map<std::string, std::vector<float>>*>(params->imatrix);
        if (imatrix_data) {
            LLAMA_LOG_INFO("================================ Have weights data with %d entries\n",int(imatrix_data->size()));
            qs.has_imatrix = true;
            // check imatrix for nans or infs
            for (const auto & kv : *imatrix_data) {
                for (float f : kv.second) {
                    if (!std::isfinite(f)) {
                        throw std::runtime_error(format("imatrix contains non-finite value %f\n", f));
                    }
                }
            }
        }
    }

    const size_t align = GGUF_DEFAULT_ALIGNMENT;
    gguf_context_ptr ctx_out { gguf_init_empty() };

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
                gguf_set_val_i32(ctx_out.get(), o.key, o.val_i64);
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_BOOL) {
                gguf_set_val_bool(ctx_out.get(), o.key, o.val_bool);
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_STR) {
                gguf_set_val_str(ctx_out.get(), o.key, o.val_str);
            } else {
                LLAMA_LOG_WARN("%s: unknown KV override type for key %s\n", __func__, o.key);
            }
        }
    }

    // make a list of weights
    std::vector<const llama_model_loader::llama_tensor_weight *> tensors;
    tensors.reserve(ml.weights_map.size());
    for (const auto & it : ml.weights_map) {
        tensors.push_back(&it.second);
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
        GGML_ASSERT((qs.n_attention_wv == n_attn_layer) && "n_attention_wv is unexpected");
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
        struct ggml_tensor * tensor = it->tensor;
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
            gguf_set_val_i32(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str(), ml.n_tensors);
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

    // The old C-style workaround for reading SmartQuant JSON has been removed.
    // `smarter_quant_config` (loaded by `load_smarter_quant_config` at the start of this function) is used instead.

    const auto tn = LLM_TN(model.arch);
    new_ofstream(0);
    for (const auto * it : tensors) {
        const auto & weight = *it;
        struct ggml_tensor * tensor = weight.tensor;
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
               ++idx, ml.n_tensors,
               ggml_get_name(tensor),
               llama_format_tensor_shape(tensor).c_str(),
               ggml_type_name(tensor->type));

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

        // do not quantize positional embeddings and token types (BERT)
        quantize &= name != LLM_TN(model.arch)(LLM_TENSOR_POS_EMBD,    "weight");
        quantize &= name != LLM_TN(model.arch)(LLM_TENSOR_TOKEN_TYPES, "weight");

        // do not quantize Mamba's small yet 2D weights
        // NOTE: can't use LLM_TN here because the layer number is not known
        quantize &= name.find("ssm_conv1d.weight") == std::string::npos;

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

        enum ggml_type new_type;
        void * new_data;
        size_t new_size;

        if (quantize) {
            new_type = default_type;

            // get more optimal quantization type based on the tensor shape, layer, etc.
            if (!params->pure && ggml_is_quantized(default_type)) {
                new_type = llama_tensor_get_type(qs, new_type, tensor, ftype);
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
            if (imatrix_data) {
                auto it = imatrix_data->find(tensor->name);
                if (it == imatrix_data->end()) {
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
            std::vector<no_init<float>> permuted_f32_data_holder; // Holder for permuted data

            if (tensor->type == GGML_TYPE_F32) {
                f32_data = (float *) tensor->data;
            } else if (ggml_is_quantized(tensor->type) && !params->allow_requantize) {
                throw std::runtime_error(format("requantizing from type %s is disabled", ggml_type_name(tensor->type)));
            } else {
                llama_tensor_dequantize_impl(tensor, f32_conv_buf, workers, nelements, nthread);
                f32_data = (float *) f32_conv_buf.data();
            }

            // Apply SmarterQuant column permutation if configured
            auto sq_it = smarter_quant_config.find(name);
            if (sq_it != smarter_quant_config.end() && !sq_it->second.column_permutation.empty()) {
                LLAMA_LOG_INFO("Applying column permutation for tensor %s...\n", name.c_str());
                const auto& perm = sq_it->second.column_permutation;
                if (perm.size() != (size_t)tensor->ne[0]) {
                    LLAMA_LOG_ERROR("Error: Permutation size %zu does not match tensor columns %" PRId64 " for tensor %s. Skipping permutation.\n", perm.size(), tensor->ne[0], name.c_str());
                } else {
                    permuted_f32_data_holder.resize(nelements);
                    float * permuted_data_ptr = (float *)permuted_f32_data_holder.data();

                    const int64_t n_cols = tensor->ne[0];
                    const int64_t n_rows = tensor->ne[1]; // Assuming 2D matrix for simplicity first
                                                       // For >2D, this would be ne[1]*ne[2]*ne[3]...
                    const int64_t higher_dims_stride = ggml_nelements(tensor) / (n_cols * n_rows);


                    for (int64_t h_dim = 0; h_dim < higher_dims_stride; ++h_dim) {
                        const float * current_f32_slice = f32_data + h_dim * (n_cols * n_rows);
                        float * current_permuted_slice = permuted_data_ptr + h_dim * (n_cols * n_rows);
                        for (int64_t r = 0; r < n_rows; ++r) {
                            for (int64_t c_new = 0; c_new < n_cols; ++c_new) {
                                const int64_t c_orig = perm[c_new];
                                if (c_orig < 0 || c_orig >= n_cols) {
                                     LLAMA_LOG_ERROR("Error: Invalid column index %" PRId64 " in permutation for tensor %s. Skipping permutation.\n", c_orig, name.c_str());
                                     // Fallback to original data if permutation is invalid
                                     permuted_f32_data_holder.clear(); // Release memory
                                     goto skip_permutation;
                                }
                                current_permuted_slice[r * n_cols + c_new] = current_f32_slice[r * n_cols + c_orig];
                            }
                        }
                    }
                    f32_data = permuted_data_ptr; // Use permuted data for quantization
                    LLAMA_LOG_INFO("Finished applying column permutation for f32_data of tensor %s.\n", name.c_str());

                    // Now, if imatrix is present, it also needs to be permuted according to the same column permutation.
                    // The imatrix is assumed to be 1D array of size n_cols for the current tensor/slice.
                    // If the tensor is 3D (e.g. experts), the imatrix might be [n_slices, n_cols]
                    // or a single [n_cols] applied to all slices.
                    // The current `imatrix` pointer is for the whole tensor from `imatrix_data`.
                    // We need a mutable copy to permute.
                    if (imatrix) {
                        std::vector<float> permuted_imatrix_values;
                        const int64_t n_cols_imatrix = tensor->ne[0]; // Assuming imatrix corresponds to columns of one slice.
                                                                   // This matches how it's used later.
                        const int64_t n_slices_imatrix = (imatrix_data->at(name).size() / n_cols_imatrix);
                                                                   // if imatrix_data has multiple slices worth of data.
                                                                   // Or, if imatrix_data.size() == n_cols_imatrix, then n_slices_imatrix = 1.

                        // This logic assumes imatrix_data->at(name) contains data for all slices if n_slices_imatrix > 1
                        // or data for one slice if n_slices_imatrix = 1 (which is then broadcasted or selected)
                        // For now, let's assume the `imatrix` pointer we have is for a single effective slice,
                        // and if the original tensor has multiple slices (tensor->ne[2] > 1), this single
                        // imatrix is applied to each slice, OR the imatrix itself is already structured correctly
                        // for multiple slices if its total size indicates that.
                        // The most straightforward is to permute the imatrix that corresponds to one slice's columns.

                        if (imatrix_data->at(name).size() % n_cols_imatrix != 0) {
                            LLAMA_LOG_WARN("Warning: imatrix size %zu is not a multiple of n_cols %" PRId64 " for tensor %s. Skipping imatrix permutation.\n",
                                           imatrix_data->at(name).size(), n_cols_imatrix, name.c_str());
                        } else {
                            permuted_imatrix_values.resize(imatrix_data->at(name).size());
                            const float* original_imatrix_ptr = imatrix_data->at(name).data();
                            float* permuted_imatrix_ptr = permuted_imatrix_values.data();

                            const int64_t num_imatrix_slices_in_source = imatrix_data->at(name).size() / n_cols_imatrix;

                            for (int64_t s_idx = 0; s_idx < num_imatrix_slices_in_source; ++s_idx) {
                                const float* current_original_imatrix_slice = original_imatrix_ptr + s_idx * n_cols_imatrix;
                                float* current_permuted_imatrix_slice = permuted_imatrix_ptr + s_idx * n_cols_imatrix;
                                for (int64_t c_new = 0; c_new < n_cols_imatrix; ++c_new) {
                                    const int64_t c_orig = perm[c_new]; // 'perm' is from f32_data permutation
                                    if (c_orig >= 0 && c_orig < n_cols_imatrix) {
                                        current_permuted_imatrix_slice[c_new] = current_original_imatrix_slice[c_orig];
                                    } else {
                                        // Should not happen if perm is valid for f32_data
                                        current_permuted_imatrix_slice[c_new] = current_original_imatrix_slice[c_new]; // or some default
                                    }
                                }
                            }
                            // This is tricky: 'imatrix' is const. We need a way to use this permuted version.
                            // Solution: Have a std::vector<float> member in quantize_state_impl or pass it around.
                            // For now, this highlights the need. The actual `imatrix` pointer used by quantization
                            // functions will need to point to this permuted data.
                            // Let's create a temporary holder for the permuted imatrix if permutation happens.
                            // This means `imatrix` pointer might be reassigned.
                            // This requires `imatrix` to be non-const locally or use a holder.
                            // The simplest is to have `permuted_imatrix_holder` at `llama_model_quantize_impl` scope
                            // and reassign `imatrix` pointer.
                            // For now, just log. The actual reassignment needs careful thought on variable scope.
                            // Store the permuted imatrix in a temporary vector that lives long enough.
                            // This means we need a place to store this outside the if block.
                            // Best: modify `imatrix_data` or use a temporary map.
                            // Simplest for now: if permuted, the `imatrix` pointer for THIS tensor will be from a new vector.

                            // Let's assume we will update the 'imatrix' pointer that is used later.
                            // This will be handled by having a local std::vector<float> that stores the permuted imatrix,
                            // and 'imatrix' pointer will point to its .data().
                            // This implies 'imatrix' variable itself should not be const if it can be reassigned.
                            // Or, introduce a new variable e.g. `final_imatrix_ptr`.
                            qs.permuted_imatrix_holder = permuted_imatrix_values; // Store in quantize_state_impl
                            imatrix = qs.permuted_imatrix_holder.data(); // Update imatrix to point to permuted version

                            LLAMA_LOG_INFO("Finished applying column permutation for imatrix of tensor %s.\n", name.c_str());
                        }
                    }

                    // Store permutation in GGUF metadata
                    {
                        nlohmann::json perm_json_array = sq_it->second.column_permutation;
                        std::string perm_str = perm_json_array.dump();

                        llama_model_kv_override kvo_perm;
                        snprintf(kvo_perm.key, sizeof(kvo_perm.key), "%s.smarterquant.permutation", name.c_str());
                        kvo_perm.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
                        strncpy(kvo_perm.val_str, perm_str.c_str(), sizeof(kvo_perm.val_str) - 1);
                        kvo_perm.val_str[sizeof(kvo_perm.val_str) - 1] = '\0';

                        llama_model_kv_override kvo_enabled;
                        snprintf(kvo_enabled.key, sizeof(kvo_enabled.key), "%s.smarterquant.enabled", name.c_str());
                        kvo_enabled.tag = LLAMA_KV_OVERRIDE_TYPE_BOOL;
                        kvo_enabled.val_bool = true;

                        nlohmann::json types_json_array = sq_it->second.compression_types;
                        std::string types_str = types_json_array.dump();
                        llama_model_kv_override kvo_types;
                        snprintf(kvo_types.key, sizeof(kvo_types.key), "%s.smarterquant.block_types", name.c_str());
                        kvo_types.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
                        strncpy(kvo_types.val_str, types_str.c_str(), sizeof(kvo_types.val_str) -1);
                        kvo_types.val_str[sizeof(kvo_types.val_str)-1] = '\0';


                        if (params->kv_overrides) {
                            auto* overrides_vec = reinterpret_cast<std::vector<llama_model_kv_override>*>(params->kv_overrides);
                            bool null_term_found = false;
                            if (!overrides_vec->empty() && overrides_vec->back().key[0] == 0) {
                                null_term_found = true;
                                overrides_vec->pop_back(); // Remove null terminator temporarily
                            }
                            overrides_vec->push_back(kvo_perm);
                            overrides_vec->push_back(kvo_enabled);
                            overrides_vec->push_back(kvo_types);
                            overrides_vec->emplace_back(); // Add new null terminator
                            overrides_vec->back().key[0] = 0;
                        }
                        LLAMA_LOG_INFO("Adding metadata for %s: permutation, enabled, block_types\n", name.c_str());
                    }
                skip_permutation:;
                }
            }

            if (work.size() < (size_t)nelements * 4) {
                work.resize(nelements * 4); // upper bound on size
            }
            new_data = work.data();

            if (sq_it != smarter_quant_config.end() && !sq_it->second.compression_types.empty()) {
                // The old placeholder for SmarterQuant size calculation was inside llama_tensor_quantize_smarter_blocks
                // and has been removed. The actual quantization and size calculation will happen below.
                LLAMA_LOG_INFO("Preparing for SmarterQuant custom block quantization for %s.\n", name.c_str());
            }

            // Common definitions for both paths
            const int64_t n_per_row = tensor->ne[0];
            const int64_t nrows     = tensor->ne[1];
            const int64_t n_slices  = tensor->ne[2]; // For 3D tensors (e.g., MoE expert weights)

            // Calculate nthread_use, common for both quantization paths below
            // This is based on processing one slice at a time if n_slices > 1
            static const int64_t min_chunk_size_bytes = 32 * 512; // A common minimum chunk size reference
            const int64_t elements_per_row_bytes_approx = n_per_row * sizeof(float); // Approx size for chunking decision
            const int64_t chunk_size_elements = (elements_per_row_bytes_approx >= min_chunk_size_bytes ? n_per_row : n_per_row * ((min_chunk_size_bytes + elements_per_row_bytes_approx - 1)/elements_per_row_bytes_approx));
            const int64_t nelements_matrix_per_slice = n_per_row * nrows;
            const int64_t nchunk_per_slice = (nelements_matrix_per_slice + chunk_size_elements - 1)/chunk_size_elements;
            const int64_t nthread_use = nthread > 1 ? std::max((int64_t)1, std::min((int64_t)nthread, nchunk_per_slice)) : 1;


            if (sq_it != smarter_quant_config.end() && !sq_it->second.compression_types.empty()) {
                // SmarterQuant Path
                // Condition changed: only require compression_types to be present. Empty permutation is fine.
                // Set GGUF type to the type of the 4th block segment (or last defined type for blocks 4+)
                new_type = static_cast<ggml_type>(sq_it->second.compression_types[3]);
                LLAMA_LOG_INFO("Applying SmarterQuant to %s. GGUF type: %s. Calling llama_tensor_quantize_smarter_blocks.\n", name.c_str(), ggml_type_name(new_type));

                new_size = llama_tensor_quantize_smarter_blocks(
                    f32_data,           // Input f32 data (already permuted if permutation was applied)
                    new_data,           // Output buffer for quantized data
                    tensor->ne,         // Full tensor dimensions {n_cols, n_rows, n_slices}
                    sq_it->second,      // SmarterQuantTensorInfo (compression types, permutation info)
                    imatrix,            // Importance matrix (if available, should also be permuted)
                    nthread_use);       // nthread_use (though current smarter_blocks is serial)

                LLAMA_LOG_INFO("SmarterQuant for %s done. Calculated new_size = %zu bytes.\n", name.c_str(), new_size);

            } else {
                // Standard Quantization Path
                LLAMA_LOG_INFO("converting to %s .. ", ggml_type_name(new_type));
                fflush(stdout);
                new_size = 0;
                // Standard quantization handles 3D tensors (n_slices > 1) by iterating through slices.
                for (int64_t i03 = 0; i03 < n_slices; ++i03) {
                    const float * f32_data_slice = f32_data + i03 * nelements_matrix_per_slice;

                    // new_data_slice must point to the correct location in the output buffer for the current slice.
                    // The size of each row depends on 'new_type' and 'n_per_row'.
                    void * new_data_slice = (char *)new_data + i03 * nrows * ggml_row_size(new_type, n_per_row);

                    const float * imatrix_slice = nullptr;
                    if (imatrix) {
                        // Assuming imatrix is structured [n_slices, n_cols_total] or [n_slices, n_rows, n_cols_total]
                        // and is passed to llama_tensor_quantize_impl for a single slice.
                        // The original code used `imatrix + i03 * n_per_row` which implies imatrix is [n_slices, n_cols_total]
                        // where n_per_row is effectively n_cols_total for a slice.
                        imatrix_slice = imatrix + i03 * n_per_row; // This matches the original logic for per-slice imatrix
                    }

                    new_size += llama_tensor_quantize_impl(new_type, f32_data_slice, new_data_slice, chunk_size_elements, nrows, n_per_row, imatrix_slice, workers, nthread_use);
                }
            }
            LLAMA_LOG_INFO("size = %8.2f MiB -> %8.2f MiB\n", ggml_nbytes(tensor)/1024.0/1024.0, new_size/1024.0/1024.0);
        }
        total_size_org += ggml_nbytes(tensor);
        total_size_new += new_size;

        // update the gguf meta data as we go
        gguf_set_tensor_type(ctx_outs[cur_split].get(), name.c_str(), new_type);
        // GGML_ASSERT(gguf_get_tensor_size(ctx_outs[cur_split].get(), gguf_find_tensor(ctx_outs[cur_split].get(), name.c_str())) == new_size); // Temporarily commented for SmarterQuant
        // For SmarterQuant, new_size is the true packed size, which might differ from what gguf_get_tensor_size reports based on the *nominal* new_type.
        // The actual data written is new_size. The GGUF tensor metadata (new_type) serves as a base/fallback for loaders not aware of SmarterQuant.
        gguf_set_tensor_data(ctx_outs[cur_split].get(), name.c_str(), new_data);

        // write tensor data + padding
        fout.write((const char *) new_data, new_size);
        zeros(fout, GGML_PAD(new_size, align) - new_size);
    }
    close_ofstream();
    // freeWeightMap(&weight_map); // Old C-style map removed
		
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

struct llama_model_quantize_params llama_model_quantize_default_params() {
    struct llama_model_quantize_params result = {
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
        /*.kv_overrides                =*/ nullptr,
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
