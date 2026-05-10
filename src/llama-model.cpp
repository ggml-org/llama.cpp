#include "llama-model.h"

#include "llama-arch.h"
#include "llama-ext.h"
#include "llama-hparams.h"
#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-cparams.h"
#include "llama-model-loader.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

#include "models/models.h"

#include "ggml.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static llama_model * llama_model_mapping(llm_arch arch, const llama_model_params & params) {
    switch (arch) {
        case LLM_ARCH_LLAMA:
            return new llama_model_llama(params);
        case LLM_ARCH_LLAMA4:
            return new llama_model_llama4(params);
        case LLM_ARCH_LLAMA_EMBED:
            return new llama_model_llama_embed(params);
        case LLM_ARCH_MAINCODER:
            return new llama_model_maincoder(params);
        case LLM_ARCH_DECI:
            return new llama_model_deci(params);
        case LLM_ARCH_BAICHUAN:
            return new llama_model_baichuan(params);
        case LLM_ARCH_FALCON:
            return new llama_model_falcon(params);
        case LLM_ARCH_GROK:
            return new llama_model_grok(params);
        case LLM_ARCH_STARCODER:
            return new llama_model_starcoder(params);
        case LLM_ARCH_REFACT:
            return new llama_model_refact(params);
        case LLM_ARCH_BERT:
            return new llama_model_bert(params);
        case LLM_ARCH_JINA_BERT_V2:
            return new llama_model_jina_bert_v2(params);
        case LLM_ARCH_JINA_BERT_V3:
            return new llama_model_jina_bert_v3(params);
        case LLM_ARCH_NOMIC_BERT:
            return new llama_model_nomic_bert(params);
        case LLM_ARCH_NOMIC_BERT_MOE:
            return new llama_model_nomic_bert_moe(params);
        case LLM_ARCH_MODERN_BERT:
            return new llama_model_modern_bert(params);
        case LLM_ARCH_NEO_BERT:
            return new llama_model_neo_bert(params);
        case LLM_ARCH_EUROBERT:
            return new llama_model_eurobert(params);
        case LLM_ARCH_BLOOM:
            return new llama_model_bloom(params);
        case LLM_ARCH_MPT:
            return new llama_model_mpt(params);
        case LLM_ARCH_STABLELM:
            return new llama_model_stablelm(params);
        case LLM_ARCH_QWEN:
            return new llama_model_qwen(params);
        case LLM_ARCH_QWEN2:
            return new llama_model_qwen2(params);
        case LLM_ARCH_DREAM:
            return new llama_model_dream(params);
        case LLM_ARCH_LLADA:
            return new llama_model_llada(params);
        case LLM_ARCH_LLADA_MOE:
            return new llama_model_llada_moe(params);
        case LLM_ARCH_RND1:
            return new llama_model_rnd1(params);
        case LLM_ARCH_QWEN2VL:
            return new llama_model_qwen2vl(params);
        case LLM_ARCH_QWEN2MOE:
            return new llama_model_qwen2moe(params);
        case LLM_ARCH_QWEN3:
            return new llama_model_qwen3(params);
        case LLM_ARCH_QWEN3MOE:
            return new llama_model_qwen3moe(params);
        case LLM_ARCH_QWEN3VL:
            return new llama_model_qwen3vl(params);
        case LLM_ARCH_QWEN3VLMOE:
            return new llama_model_qwen3vlmoe(params);
        case LLM_ARCH_PHI2:
            return new llama_model_phi2(params);
        case LLM_ARCH_PHI3:
            return new llama_model_phi3(params);
        case LLM_ARCH_PHIMOE:
            return new llama_model_phimoe(params);
        case LLM_ARCH_PLAMO:
            return new llama_model_plamo(params);
        case LLM_ARCH_PLAMO2:
            return new llama_model_plamo2(params);
        case LLM_ARCH_PLAMO3:
            return new llama_model_plamo3(params);
        case LLM_ARCH_GPT2:
            return new llama_model_gpt2(params);
        case LLM_ARCH_CODESHELL:
            return new llama_model_codeshell(params);
        case LLM_ARCH_ORION:
            return new llama_model_orion(params);
        case LLM_ARCH_INTERNLM2:
            return new llama_model_internlm2(params);
        case LLM_ARCH_MINICPM3:
            return new llama_model_minicpm3(params);
        case LLM_ARCH_GEMMA:
            return new llama_model_gemma(params);
        case LLM_ARCH_GEMMA2:
            return new llama_model_gemma2(params);
        case LLM_ARCH_GEMMA3:
            return new llama_model_gemma3(params);
        case LLM_ARCH_GEMMA3N:
            return new llama_model_gemma3n(params);
        case LLM_ARCH_GEMMA4:
            return new llama_model_gemma4(params);
        case LLM_ARCH_GEMMA_EMBEDDING:
            return new llama_model_gemma_embedding(params);
        case LLM_ARCH_STARCODER2:
            return new llama_model_starcoder2(params);
        case LLM_ARCH_MAMBA:
            return new llama_model_mamba(params);
        case LLM_ARCH_MAMBA2:
            return new llama_model_mamba2(params);
        case LLM_ARCH_JAMBA:
            return new llama_model_jamba(params);
        case LLM_ARCH_XVERSE:
            return new llama_model_xverse(params);
        case LLM_ARCH_COMMAND_R:
            return new llama_model_command_r(params);
        case LLM_ARCH_COHERE2:
            return new llama_model_cohere2(params);
        case LLM_ARCH_DBRX:
            return new llama_model_dbrx(params);
        case LLM_ARCH_OLMO:
            return new llama_model_olmo(params);
        case LLM_ARCH_OLMO2:
            return new llama_model_olmo2(params);
        case LLM_ARCH_OLMOE:
            return new llama_model_olmoe(params);
        case LLM_ARCH_OPENELM:
            return new llama_model_openelm(params);
        case LLM_ARCH_GPTNEOX:
            return new llama_model_gptneox(params);
        case LLM_ARCH_ARCTIC:
            return new llama_model_arctic(params);
        case LLM_ARCH_DEEPSEEK:
            return new llama_model_deepseek(params);
        case LLM_ARCH_DEEPSEEK2:
            return new llama_model_deepseek2(params);
        case LLM_ARCH_DEEPSEEK2OCR:
            return new llama_model_deepseek2ocr(params);
        case LLM_ARCH_GLM_DSA:
            return new llama_model_glm_dsa(params);
        case LLM_ARCH_MISTRAL4:
            return new llama_model_mistral4(params);
        case LLM_ARCH_CHATGLM:
            return new llama_model_chatglm(params);
        case LLM_ARCH_GLM4:
            return new llama_model_glm4(params);
        case LLM_ARCH_GLM4_MOE:
            return new llama_model_glm4_moe(params);
        case LLM_ARCH_BITNET:
            return new llama_model_bitnet(params);
        case LLM_ARCH_T5:
            return new llama_model_t5(params);
        case LLM_ARCH_T5ENCODER:
            return new llama_model_t5encoder(params);
        case LLM_ARCH_JAIS:
            return new llama_model_jais(params);
        case LLM_ARCH_JAIS2:
            return new llama_model_jais2(params);
        case LLM_ARCH_NEMOTRON:
            return new llama_model_nemotron(params);
        case LLM_ARCH_NEMOTRON_H:
            return new llama_model_nemotron_h(params);
        case LLM_ARCH_NEMOTRON_H_MOE:
            return new llama_model_nemotron_h_moe(params);
        case LLM_ARCH_EXAONE:
            return new llama_model_exaone(params);
        case LLM_ARCH_EXAONE4:
            return new llama_model_exaone4(params);
        case LLM_ARCH_EXAONE_MOE:
            return new llama_model_exaone_moe(params);
        case LLM_ARCH_RWKV6:
            return new llama_model_rwkv6(params);
        case LLM_ARCH_RWKV6QWEN2:
            return new llama_model_rwkv6qwen2(params);
        case LLM_ARCH_RWKV7:
            return new llama_model_rwkv7(params);
        case LLM_ARCH_ARWKV7:
            return new llama_model_arwkv7(params);
        case LLM_ARCH_GRANITE:
            return new llama_model_granite(params);
        case LLM_ARCH_GRANITE_MOE:
            return new llama_model_granite_moe(params);
        case LLM_ARCH_MINICPM:
            return new llama_model_minicpm(params);
        case LLM_ARCH_GRANITE_HYBRID:
            return new llama_model_granite_hybrid(params);
        case LLM_ARCH_CHAMELEON:
            return new llama_model_chameleon(params);
        case LLM_ARCH_WAVTOKENIZER_DEC:
            return new llama_model_wavtokenizer_dec(params);
        case LLM_ARCH_PLM:
            return new llama_model_plm(params);
        case LLM_ARCH_BAILINGMOE:
            return new llama_model_bailingmoe(params);
        case LLM_ARCH_BAILINGMOE2:
            return new llama_model_bailingmoe2(params);
        case LLM_ARCH_SEED_OSS:
            return new llama_model_seed_oss(params);
        case LLM_ARCH_DOTS1:
            return new llama_model_dots1(params);
        case LLM_ARCH_ARCEE:
            return new llama_model_arcee(params);
        case LLM_ARCH_AFMOE:
            return new llama_model_afmoe(params);
        case LLM_ARCH_ERNIE4_5:
            return new llama_model_ernie4_5(params);
        case LLM_ARCH_ERNIE4_5_MOE:
            return new llama_model_ernie4_5_moe(params);
        case LLM_ARCH_PADDLEOCR:
            return new llama_model_paddleocr(params);
        case LLM_ARCH_HUNYUAN_MOE:
            return new llama_model_hunyuan_moe(params);
        case LLM_ARCH_HUNYUAN_VL:
            return new llama_model_hunyuan_vl(params);
        case LLM_ARCH_HUNYUAN_DENSE:
            return new llama_model_hunyuan_dense(params);
        case LLM_ARCH_SMOLLM3:
            return new llama_model_smollm3(params);
        case LLM_ARCH_OPENAI_MOE:
            return new llama_model_openai_moe(params);
        case LLM_ARCH_FALCON_H1:
            return new llama_model_falcon_h1(params);
        case LLM_ARCH_LFM2:
            return new llama_model_lfm2(params);
        case LLM_ARCH_LFM2MOE:
            return new llama_model_lfm2moe(params);
        case LLM_ARCH_SMALLTHINKER:
            return new llama_model_smallthinker(params);
        case LLM_ARCH_GROVEMOE:
            return new llama_model_grovemoe(params);
        case LLM_ARCH_APERTUS:
            return new llama_model_apertus(params);
        case LLM_ARCH_MINIMAX_M2:
            return new llama_model_minimax_m2(params);
        case LLM_ARCH_COGVLM:
            return new llama_model_cogvlm(params);
        case LLM_ARCH_PANGU_EMBED:
            return new llama_model_pangu_embed(params);
        case LLM_ARCH_QWEN3NEXT:
            return new llama_model_qwen3next(params);
        case LLM_ARCH_QWEN35:
            return new llama_model_qwen35(params);
        case LLM_ARCH_QWEN35MOE:
            return new llama_model_qwen35moe(params);
        case LLM_ARCH_MISTRAL3:
            return new llama_model_mistral3(params);
        case LLM_ARCH_MIMO2:
            return new llama_model_mimo2(params);
        case LLM_ARCH_KIMI_LINEAR:
            return new llama_model_kimi_linear(params);
        case LLM_ARCH_STEP35:
            return new llama_model_step35(params);
        default:
            throw std::runtime_error(std::string("unsupported model architecture: '") + llm_arch_name(arch) + "'");
    }

}

llama_model * llama_model_create(llm_arch arch, const llama_model_params & params) {
    llama_model * model = llama_model_mapping(arch, params);

    if (model != nullptr) {
        model->arch = arch;
        auto & devices = model->devices;
        if (!devices.empty() && devices[0].is_meta && !llm_arch_supports_sm_tensor(arch)) {
            throw std::runtime_error(std::string("LLAMA_SPLIT_MODE_TENSOR not implemented for architecture '") + llm_arch_name(arch) + "'");
        }
    }

    return model;
}

llama_model * llama_model_create(llama_model_loader & ml, const llama_model_params & params) {
    llm_arch arch = ml.get_arch();
    if (arch == LLM_ARCH_UNKNOWN) {
        throw std::runtime_error("unknown model architecture: '" + ml.get_arch_name() + "'");
    }

    return llama_model_create(arch, params);
}

struct ggml_backend_meta_split_state llama_meta_device_get_split_state(const struct ggml_tensor * tensor, void * userdata) {
    const llama_meta_device_get_split_state_userdata * ud = (const llama_meta_device_get_split_state_userdata *) userdata;
    const llama_hparams & hparams = ud->model->hparams;
    const std::string tensor_name = tensor->name;

    const std::regex pattern_q_weight        ("blk\\.\\d*\\.attn_q.weight");
    const std::regex pattern_kv_weight       ("blk\\.\\d*\\.attn_(k|v).weight");
    const std::regex pattern_qkv_weight      ("blk\\.\\d*\\.attn_qkv.weight");
    const std::regex pattern_q_bias          ("blk\\.\\d*\\.attn_q\\.bias");
    const std::regex pattern_kv_bias         ("blk\\.\\d*\\.attn_(k|v)\\.bias");
    const std::regex pattern_qkv_bias        ("blk\\.\\d*\\.attn_qkv.bias");
    const std::regex pattern_qk_norm         ("blk\\.\\d*\\.attn_(q|k)_norm\\.weight");
    const std::regex pattern_kv_cache        ("cache_(k|v)_l\\d*");
    const std::regex pattern_attn_sinks      ("blk\\.\\d*\\.attn_sinks.weight");
    const std::regex pattern_attn_out_weight ("blk\\.\\d*\\.attn_output.weight");
    const std::regex pattern_attn_out_bias   ("blk\\.\\d*\\.attn_output.bias");
    const std::regex pattern_attn_gate_weight("blk\\.\\d*\\.attn_gate.weight");

    const std::regex pattern_ssm_dt          ("blk\\.\\d*\\.ssm_dt.bias");
    const std::regex pattern_ssm_a           ("blk\\.\\d*\\.ssm_a");
    const std::regex pattern_ssm_alpha       ("blk\\.\\d*\\.ssm_alpha.weight");
    const std::regex pattern_ssm_beta        ("blk\\.\\d*\\.ssm_beta.weight");
    const std::regex pattern_ssm_beta_alpha  ("blk\\.\\d*\\.ssm_ba.weight");
    const std::regex pattern_r_cache         ("cache_r_l\\d*");
    const std::regex pattern_s_cache         ("cache_s_l\\d*");
    const std::regex pattern_ssm_conv1d      ("blk\\.\\d*\\.ssm_conv1d.weight");
    const std::regex pattern_ssm_out_weight  ("blk\\.\\d*\\.ssm_out.weight");

    const std::regex pattern_ffn_up_gate_weight("blk\\.\\d*\\.ffn_(up|gate)(_exps)?.weight");
    const std::regex pattern_ffn_up_gate_bias  ("blk\\.\\d*\\.ffn_(up|gate)(_exps)?.bias");
    const std::regex pattern_ffn_gate_up_weight("blk\\.\\d*\\.ffn_gate_up(_exps)?.weight");
    const std::regex pattern_ffn_down_weight   ("blk\\.\\d*\\.ffn_down(_exps)?.weight");
    const std::regex pattern_ffn_down_bias     ("blk\\.\\d*\\.ffn_down.bias");
    const std::regex pattern_ffn_down_exps_bias("blk\\.\\d*\\.ffn_down_exps.bias");

    const std::regex pattern_output_weight("output\\.weight");
    const std::regex pattern_output_bias  ("output\\.bias");

    struct tensor_config {
        ggml_backend_meta_split_axis axis;

        const ggml_tensor * tensor_axis_0;

        uint32_t il;
        size_t   rotation; // when assigning tensor slices, rotate how the rounding is done for more even allocation
    };

    auto get_tensor_config_impl = [&](
                const ggml_backend_meta_split_axis axis, const std::string & suffix = "", const std::string & suffix_fallback = "") -> tensor_config {
        // the layers in a tensor can be inhomogeneous, if the pattern is cleanly divided by the number of GPUs there can be aliasing effects,
        //     count only the same type of previous layers to avoid this
        auto get_il_eff = [&](const size_t il){
            size_t ret = 0;
            const bool il_is_recurrent = hparams.is_recurrent(il);
            const bool il_is_swa       = hparams.is_swa(il);
            for (size_t il_prev = 0; il_prev < il; il_prev++) {
                ret += hparams.is_recurrent(il_prev) == il_is_recurrent && hparams.is_swa(il_prev) == il_is_swa;
            }
            return ret;
        };

        uint32_t il;
        std::string prefix;
        size_t rotation;
        if (tensor_name.substr(0, 4) == "blk.") {
            const size_t length_prefix = tensor_name.find('.', 4);
            GGML_ASSERT(length_prefix != std::string::npos);
            prefix = tensor_name.substr(0, length_prefix + 1);
            il = std::stoull(tensor_name.substr(4, length_prefix));
            rotation = get_il_eff(il) % ud->n_devices;
        } else if (tensor_name.substr(0, 6) == "cache_") {
            const size_t layer_index_start = tensor_name.find("_l", 6);
            GGML_ASSERT(layer_index_start != std::string::npos);
            il = std::stoull(tensor_name.substr(layer_index_start + 2));
            prefix = "blk." + std::to_string(il) + ".";
            rotation = get_il_eff(il) % ud->n_devices;
        } else {
            il = 0;
            rotation = hparams.n_layer % ud->n_devices;
        }
        const ggml_tensor * tensor_axis_0 = suffix.empty() ? tensor : ud->model->get_tensor((prefix + suffix).c_str());
        if (tensor_axis_0 == nullptr) {
            GGML_ASSERT(!suffix_fallback.empty());
            tensor_axis_0 = ud->model->get_tensor((prefix + suffix_fallback).c_str());
        }
        GGML_ASSERT(tensor_axis_0 != nullptr);
        return {axis, tensor_axis_0, il, rotation};
    };

    auto get_tensor_config = [&]() -> tensor_config {
        // standard attention
        if (std::regex_match(tensor_name, pattern_q_weight) || std::regex_match(tensor_name, pattern_kv_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight");
        }
        if (std::regex_match(tensor_name, pattern_q_bias) || std::regex_match(tensor_name, pattern_kv_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight");
        }
        if (std::regex_match(tensor_name, pattern_qkv_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1);
        }
        if ( std::regex_match(tensor_name, pattern_qkv_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        }
        if (std::regex_match(tensor_name, pattern_qk_norm)) {
            return get_tensor_config_impl(tensor->ne[1] == 1 ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight");
        }
        if (std::regex_match(tensor_name, pattern_kv_cache) || std::regex_match(tensor_name, pattern_attn_sinks)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight");
        }
        if (std::regex_match(tensor_name, pattern_attn_out_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        }
        if (std::regex_match(tensor_name, pattern_attn_out_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }

        if (std::regex_match(tensor_name, pattern_attn_gate_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1);
        }
        if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta) ||
                std::regex_match(tensor_name, pattern_ssm_beta_alpha)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_r_cache) || std::regex_match(tensor_name, pattern_s_cache)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_conv1d)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_out_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        }

        // FFN
        if (std::regex_match(tensor_name, pattern_ffn_up_gate_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_up_gate_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_down_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_down_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        if (std::regex_match(tensor_name, pattern_ffn_down_exps_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_PARTIAL);
        }

        // output
        if (std::regex_match(tensor_name, pattern_output_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1);
        }
        if (std::regex_match(tensor_name, pattern_output_bias)) {
            const ggml_tensor * output_weight = ud->model->get_tensor("output.weight");
            GGML_ASSERT(output_weight != nullptr);
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        }

        // everything else
        return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    };

    auto get_split_segments = [&](int axis, uint32_t il) -> std::vector<int64_t> {
        if (ud->model->arch == LLM_ARCH_QWEN3NEXT || ud->model->arch == LLM_ARCH_QWEN35 || ud->model->arch == LLM_ARCH_QWEN35MOE) {
            const int64_t head_k_dim = hparams.ssm_d_state;
            const int64_t head_v_dim = hparams.ssm_d_state;
            const int64_t n_k_heads  = hparams.ssm_n_group;
            const int64_t n_v_heads  = hparams.ssm_dt_rank;
            const int64_t key_dim    = head_k_dim * n_k_heads;
            const int64_t value_dim  = head_v_dim * n_v_heads;

            // both Qwen 3 Next and Qwen 3.5 support n_v_heads > n_k_heads but the broadcasting pattern is different:
            //   - Qwen 3 Next: [k0_v0, k0_v1, k1_v2, k1_v3] (this is the default split pattern)
            //   - Qwen 3.5:    [k0_v0, k1_v1, k0_v2, k1_v3] (needs segmenting of V on the scale of K to get the correct pattern)
            if (ud->model->arch == LLM_ARCH_QWEN3NEXT) {
                if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_ssm_conv1d)) {
                    GGML_ASSERT(tensor->ne[axis] == 2*key_dim + value_dim);
                    return {key_dim, key_dim, value_dim};
                }
            } else {
                const int64_t head_ratio = n_v_heads / n_k_heads;
                if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_ssm_conv1d)) {
                    GGML_ASSERT(tensor->ne[axis] == 2*key_dim + value_dim);
                    return std::vector<int64_t>(2 + head_ratio, key_dim);
                }
                if (std::regex_match(tensor_name, pattern_attn_gate_weight) || std::regex_match(tensor_name, pattern_ssm_out_weight)) {
                    return std::vector<int64_t>(head_ratio, key_dim);
                }
                if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a) ||
                        std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta)) {
                    return std::vector<int64_t>(head_ratio, n_k_heads);
                }
                if (std::regex_match(tensor_name, pattern_r_cache)) {
                    return std::vector<int64_t>(2 + head_ratio, key_dim * (hparams.ssm_d_conv - 1));
                }
                if (std::regex_match(tensor_name, pattern_s_cache)) {
                    return std::vector<int64_t>(head_ratio, n_k_heads * head_v_dim * head_v_dim);
                }
            }

            // the FFN is the same for Qwen 3 Next and Qwen 3.5:
            if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
                const int64_t n_ff_exp = hparams.n_ff_exp;
                GGML_ASSERT(tensor->ne[axis] == 2*n_ff_exp);
                return {n_ff_exp, n_ff_exp};
            }
            return {tensor->ne[axis]};
        }

        if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_qkv_bias)) {
            const int64_t n_embd      = hparams.n_embd;
            const int64_t n_embd_gqa  = hparams.n_embd_v_gqa(il);
            GGML_ASSERT(hparams.n_embd_k_gqa() == n_embd_gqa);
            GGML_ASSERT(tensor->ne[axis] == n_embd + 2*n_embd_gqa);
            return {n_embd, n_embd_gqa, n_embd_gqa};
        }
        if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
            const int64_t n_ff_exp = hparams.n_ff_exp;
            GGML_ASSERT(tensor->ne[axis] == 2*n_ff_exp);
            return {n_ff_exp, n_ff_exp};
        }
        return {tensor->ne[axis]};
    };

    auto get_split_granularity = [&](int64_t blck_size, uint32_t il, const std::vector<int64_t> & segments) -> std::vector<int64_t> {
        if (hparams.is_recurrent(il)) {
            // linear attention
            const int64_t head_dim  = hparams.ssm_d_state;
            const int64_t granularity_qkv = std::lcm(blck_size, head_dim);
            if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_attn_gate_weight) ||
                    std::regex_match(tensor_name, pattern_ssm_conv1d) || std::regex_match(tensor_name, pattern_ssm_out_weight)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv);
            }
            if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a) ||
                    std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv / head_dim);
            }
            if (std::regex_match(tensor_name, pattern_ssm_beta_alpha)) {
                return std::vector<int64_t>(segments.size(), 2 * (granularity_qkv / head_dim));
            }
            if (std::regex_match(tensor_name, pattern_r_cache)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv * (hparams.ssm_d_conv - 1));
            }
            if (std::regex_match(tensor_name, pattern_s_cache)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv * head_dim);
            }
        } else {
            // regular attention
            const uint32_t n_gqa    = hparams.n_gqa(il);
            const uint32_t n_embd_q = n_gqa * hparams.n_embd_head_k(il);
            if (std::regex_match(tensor_name, pattern_attn_sinks)) {
                GGML_ASSERT(segments.size() == 1);
                return {std::lcm(n_embd_q, blck_size)/n_embd_q * n_gqa};
            }

            const int64_t granularity_q = std::lcm(n_embd_q, blck_size);
            if (std::regex_match(tensor_name, pattern_q_weight) || std::regex_match(tensor_name, pattern_q_bias)) {
                GGML_ASSERT(segments.size() == 1);
                // some models have Q gate tensors, for those cases the granularity needs to be doubled:
                if (ud->model->arch == LLM_ARCH_QWEN3NEXT || ud->model->arch == LLM_ARCH_QWEN35 || ud->model->arch == LLM_ARCH_QWEN35MOE) {
                    return {std::lcm(2*n_embd_q, blck_size)};
                }
                return {granularity_q};
            }
            if (std::regex_match(tensor_name, pattern_attn_out_weight)) {
                GGML_ASSERT(segments.size() == 1);
                return {granularity_q};
            }

            const int64_t granularity_kv = granularity_q / n_gqa;
            if (std::regex_match(tensor_name, pattern_kv_weight) ||
                std::regex_match(tensor_name, pattern_kv_bias) ||
                std::regex_match(tensor_name, pattern_kv_cache)) {
                GGML_ASSERT(segments.size() == 1);
                return {granularity_kv};
            }
            if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_qkv_bias)) {
                GGML_ASSERT(segments.size() == 3);
                return {granularity_q, granularity_kv, granularity_kv};
            }
        }

        // FFN
        if (std::regex_match(tensor_name, pattern_ffn_up_gate_weight) || std::regex_match(tensor_name, pattern_ffn_up_gate_bias) ||
                std::regex_match(tensor_name, pattern_ffn_gate_up_weight) || std::regex_match(tensor_name, pattern_ffn_down_weight)) {
            GGML_ASSERT(segments.size() <= 2);
            return std::vector<int64_t>(segments.size(), blck_size);
        }

        // everything else
        GGML_ASSERT(segments.size() == 1);
        return {1};
    };

    ggml_backend_meta_split_state split_state;
    memset(&split_state, 0, sizeof(split_state));
    tensor_config tc = get_tensor_config();
    split_state.axis = tc.axis;
    if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
        const int64_t ne_full = tensor->ne[split_state.axis];
        const int64_t blck_size = ggml_blck_size(tc.tensor_axis_0->type);
        const float * tensor_split = ud->model->tensor_split();
        std::vector<float> tensor_split_scan;
        tensor_split_scan.reserve(ud->n_devices);
        for (size_t j = 0; j < ud->n_devices; j++) {
            tensor_split_scan.push_back(tensor_split == nullptr ? 0.0f : tensor_split[(j + tc.rotation) % ud->n_devices]);
            if (j > 0) {
                tensor_split_scan[j] += tensor_split_scan[j - 1];
            }
        }
        const std::vector<int64_t> segments = get_split_segments(split_state.axis, tc.il);
        const std::vector<int64_t> granularity = get_split_granularity(blck_size, tc.il, segments);
        for (size_t is = 0; is < segments.size(); is++) {
            const int64_t ne_s = segments[is];
            const int64_t g_s = granularity[is];
            GGML_ASSERT(ne_full % g_s == 0);
            int64_t low = 0;
            size_t j = 0;
            for (; j < ud->n_devices - 1; j++) {
                int64_t high = tensor_split_scan.back() == 0.0f ?
                    ne_s * (j+1)/ud->n_devices : ne_s * tensor_split_scan[j]/tensor_split_scan.back();
                if (high % g_s != 0) {
                    high -= high % g_s;
                }
                split_state.ne[is*ud->n_devices + (j + tc.rotation) % ud->n_devices] = high - low;
                low = high;
            }
            split_state.ne[is*ud->n_devices + (j + tc.rotation) % ud->n_devices] = ne_s - low;
        }
        split_state.n_segments = segments.size();
    } else {
        memset(split_state.ne, 0, sizeof(split_state.ne));
        split_state.n_segments = 1;
    }
    return split_state;
    GGML_UNUSED(userdata);
}

const char * llm_type_name(llm_type type) {
    switch (type) {
        case LLM_TYPE_14M:           return "14M";
        case LLM_TYPE_17M:           return "17M";
        case LLM_TYPE_22M:           return "22M";
        case LLM_TYPE_33M:           return "33M";
        case LLM_TYPE_47M:           return "47M";
        case LLM_TYPE_60M:           return "60M";
        case LLM_TYPE_70M:           return "70M";
        case LLM_TYPE_80M:           return "80M";
        case LLM_TYPE_109M:          return "109M";
        case LLM_TYPE_137M:          return "137M";
        case LLM_TYPE_140M:          return "140M";
        case LLM_TYPE_149M:          return "149M";
        case LLM_TYPE_160M:          return "160M";
        case LLM_TYPE_190M:          return "190M";
        case LLM_TYPE_220M:          return "220M";
        case LLM_TYPE_250M:          return "250M";
        case LLM_TYPE_256M:          return "256M";
        case LLM_TYPE_270M:          return "270M";
        case LLM_TYPE_335M:          return "335M";
        case LLM_TYPE_350M:          return "350M";
        case LLM_TYPE_360M:          return "360M";
        case LLM_TYPE_395M:          return "395M";
        case LLM_TYPE_410M:          return "410M";
        case LLM_TYPE_450M:          return "450M";
        case LLM_TYPE_475M:          return "475M";
        case LLM_TYPE_558M:          return "558M";
        case LLM_TYPE_700M:          return "700M";
        case LLM_TYPE_770M:          return "770M";
        case LLM_TYPE_780M:          return "780M";
        case LLM_TYPE_950M:          return "950M";
        case LLM_TYPE_0_3B:          return "0.3B";
        case LLM_TYPE_0_5B:          return "0.5B";
        case LLM_TYPE_0_6B:          return "0.6B";
        case LLM_TYPE_0_8B:          return "0.8B";
        case LLM_TYPE_1B:            return "1B";
        case LLM_TYPE_1_2B:          return "1.2B";
        case LLM_TYPE_1_3B:          return "1.3B";
        case LLM_TYPE_1_4B:          return "1.4B";
        case LLM_TYPE_1_5B:          return "1.5B";
        case LLM_TYPE_1_6B:          return "1.6B";
        case LLM_TYPE_1_7B:          return "1.7B";
        case LLM_TYPE_1_8B:          return "1.8B";
        case LLM_TYPE_2B:            return "2B";
        case LLM_TYPE_2_6B:          return "2.6B";
        case LLM_TYPE_2_8B:          return "2.8B";
        case LLM_TYPE_2_9B:          return "2.9B";
        case LLM_TYPE_3B:            return "3B";
        case LLM_TYPE_4B:            return "4B";
        case LLM_TYPE_6B:            return "6B";
        case LLM_TYPE_6_9B:          return "6.9B";
        case LLM_TYPE_7B:            return "7B";
        case LLM_TYPE_8B:            return "8B";
        case LLM_TYPE_9B:            return "9B";
        case LLM_TYPE_11B:           return "11B";
        case LLM_TYPE_12B:           return "12B";
        case LLM_TYPE_13B:           return "13B";
        case LLM_TYPE_14B:           return "14B";
        case LLM_TYPE_15B:           return "15B";
        case LLM_TYPE_16B:           return "16B";
        case LLM_TYPE_20B:           return "20B";
        case LLM_TYPE_26B:           return "26B";
        case LLM_TYPE_27B:           return "27B";
        case LLM_TYPE_30B:           return "30B";
        case LLM_TYPE_31B:           return "31B";
        case LLM_TYPE_32B:           return "32B";
        case LLM_TYPE_34B:           return "34B";
        case LLM_TYPE_35B:           return "35B";
        case LLM_TYPE_36B:           return "36B";
        case LLM_TYPE_40B:           return "40B";
        case LLM_TYPE_65B:           return "65B";
        case LLM_TYPE_70B:           return "70B";
        case LLM_TYPE_120B:          return "120B";
        case LLM_TYPE_142B:          return "142B";
        case LLM_TYPE_236B:          return "236B";
        case LLM_TYPE_290B:          return "290B";
        case LLM_TYPE_314B:          return "314B";
        case LLM_TYPE_405B:          return "405B";
        case LLM_TYPE_671B:          return "671B";
        case LLM_TYPE_SMALL:         return "0.1B";
        case LLM_TYPE_MEDIUM:        return "0.4B";
        case LLM_TYPE_LARGE:         return "0.8B";
        case LLM_TYPE_XL:            return "1.5B";
        case LLM_TYPE_A1_7B:         return "A1.7B";
        case LLM_TYPE_A2_7B:         return "A2.7B";
        case LLM_TYPE_8x7B:          return "8x7B";
        case LLM_TYPE_8x22B:         return "8x22B";
        case LLM_TYPE_16x12B:        return "16x12B";
        case LLM_TYPE_16x3_8B:       return "16x3.8B";
        case LLM_TYPE_10B_128x3_66B: return "10B+128x3.66B";
        case LLM_TYPE_57B_A14B:      return "57B.A14B";
        case LLM_TYPE_17B_16E:       return "17Bx16E (Scout)";
        case LLM_TYPE_17B_128E:      return "17Bx128E (Maverick)";
        case LLM_TYPE_A13B:          return "A13B";
        case LLM_TYPE_7B_A1B:        return "7B.A1B";
        case LLM_TYPE_8B_A1B:        return "8B.A1B";
        case LLM_TYPE_16B_A1B:       return "16B.A1B";
        case LLM_TYPE_21B_A3B:       return "21B.A3B";
        case LLM_TYPE_24B_A2B:       return "24B.A2B";
        case LLM_TYPE_26B_A4B:       return "26B.A4B";
        case LLM_TYPE_30B_A3B:       return "30B.A3B";
        case LLM_TYPE_31B_A3_5B:     return "31B.A3.5B";
        case LLM_TYPE_35B_A3B:       return "35B.A3B";
        case LLM_TYPE_48B_A3B:       return "48B.A3B";
        case LLM_TYPE_80B_A3B:       return "80B.A3B";
        case LLM_TYPE_100B_A6B:      return "100B.A6B";
        case LLM_TYPE_102B_A12B:     return "102B.A12B";
        case LLM_TYPE_106B_A12B:     return "106B.A12B";
        case LLM_TYPE_120B_A12B:     return "120B.A12B";
        case LLM_TYPE_122B_A10B:     return "122B.A10B";
        case LLM_TYPE_196B_A11B:     return "196B.A11B";
        case LLM_TYPE_230B_A10B:     return "230B.A10B";
        case LLM_TYPE_235B_A22B:     return "235B.A22B";
        case LLM_TYPE_300B_A47B:     return "300B.A47B";
        case LLM_TYPE_310B_A15B:     return "310B.A15B";
        case LLM_TYPE_355B_A32B:     return "355B.A32B";
        case LLM_TYPE_397B_A17B:     return "397B.A17B";
        case LLM_TYPE_744B_A40B:     return "744B.A40B";
        case LLM_TYPE_E2B:           return "E2B";
        case LLM_TYPE_E4B:           return "E4B";
        default:                     return "?B";
    }
}

static const char * llama_expert_gating_func_name(llama_expert_gating_func_type type) {
    switch (type) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX: return "softmax";
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID: return "sigmoid";
        default:                                    return "unknown";
    }
}

static const std::map<llama_rope_scaling_type, const char *> LLAMA_ROPE_SCALING_TYPES = {
    { LLAMA_ROPE_SCALING_TYPE_NONE,       "none"       },
    { LLAMA_ROPE_SCALING_TYPE_LINEAR,     "linear"     },
    { LLAMA_ROPE_SCALING_TYPE_YARN,       "yarn"       },
    { LLAMA_ROPE_SCALING_TYPE_LONGROPE,   "longrope"   },
};

std::string llama_rope_scaling_type_name(llama_rope_scaling_type rope_scaling_type) {
    return LLAMA_ROPE_SCALING_TYPES.at(rope_scaling_type);
}

static llama_rope_scaling_type llama_rope_scaling_type_from_string(const std::string & name) {
    for (const auto & kv : LLAMA_ROPE_SCALING_TYPES) {
        if (kv.second == name) {
            return (llama_rope_scaling_type) kv.first;
        }
    }

    return LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
}

// CPU: ACCEL -> GPU host -> CPU extra -> CPU
static buft_list_t make_cpu_buft_list(const std::vector<llama_device> & devices, bool use_extra_bufts, bool no_host) {
    buft_list_t buft_list;

    // add ACCEL buffer types
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            auto * buft = ggml_backend_dev_buffer_type(dev);
            // skip
            if (buft != ggml_backend_cpu_buffer_type()) {
                buft_list.emplace_back(dev, buft);
            }
        }
    }

    // add a host buffer type
    // storing the tensors in a host buffer is useful when the processing of large batches
    // is offloaded to a GPU device, since it reduces the time spent on data transfers
    // generally, this will be done using the first device in the list
    // a better approach would be to handle this on a weight-by-weight basis using the offload_op
    // function of the device to determine if it would benefit from being stored in a host buffer
    if (!no_host) {
        for (const auto & dev : devices) {
            ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev.dev);
            if (buft) {
                buft_list.emplace_back(dev.dev, buft);
                break;
            }
        }
    }

    // add extra buffer types
    if (use_extra_bufts) {
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            throw std::runtime_error(format("%s: no CPU backend found", __func__));
        }

        auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
        if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(cpu_dev);
            while (extra_bufts && *extra_bufts) {
                buft_list.emplace_back(cpu_dev, *extra_bufts);
                ++extra_bufts;
            }
        }
    }

    // add the CPU buffer type
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));
        }
    }

    return buft_list;
}

// GPU: split if LLAMA_SPLIT_MODE_ROW -> GPU
static buft_list_t make_gpu_buft_list(ggml_backend_dev_t dev, llama_split_mode split_mode, const float * tensor_split) {
    buft_list_t buft_list;

    // add the device split buffer type if requested and available
    if (split_mode == LLAMA_SPLIT_MODE_ROW) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto ggml_backend_split_buffer_type_fn = (ggml_backend_split_buffer_type_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_split_buffer_type");
        if (ggml_backend_split_buffer_type_fn) {
            size_t dev_index = [&]() {
                auto * reg = ggml_backend_dev_backend_reg(dev);
                for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
                    if (ggml_backend_reg_dev_get(reg, i) == dev) {
                        return i;
                    }
                }
                throw std::runtime_error(format("device %s not found in its backend reg", ggml_backend_dev_name(dev)));
            }();
            auto * buft = ggml_backend_split_buffer_type_fn(dev_index, tensor_split);
            if (buft != nullptr) {
                buft_list.emplace_back(dev, buft);
            }
        }
    }

    // add the device default buffer type
    buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));

    // add the device extra buffer type (if any)
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg) {
        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");

        if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(dev);
            while (extra_bufts && *extra_bufts) {
                buft_list.emplace_back(dev, *extra_bufts);
                ++extra_bufts;
            }
        }
    }

    return buft_list;
}

struct llama_model::impl {
    impl() = default;
    ~impl() = default;

    uint64_t n_elements = 0;

    size_t n_bytes = 0;

    std::string desc_str;

    // model memory mapped files
    llama_mmaps mappings;

    // objects representing data potentially being locked in memory
    llama_mlocks mlock_bufs;
    llama_mlocks mlock_mmaps;

    // contexts where the model tensors metadata is stored as well as the corresponding buffers:
    std::vector<std::pair<ggml_context_ptr, std::vector<ggml_backend_buffer_ptr>>> ctxs_bufs;

    buft_list_t cpu_buft_list;
    std::map<ggml_backend_dev_t, buft_list_t> gpu_buft_list;

    struct layer_dev {
        ggml_backend_dev_t dev;
        buft_list_t * buft_list;
    };

    layer_dev dev_input = {};
    layer_dev dev_output = {};
    std::vector<layer_dev> dev_layer;

    bool has_tensor_overrides;
};

llama_model::llama_model(const llama_model_params & params) : params(params), pimpl(std::make_unique<impl>()) {
    pimpl->has_tensor_overrides = params.tensor_buft_overrides && params.tensor_buft_overrides[0].pattern;
}

llama_model::~llama_model() {
    for (auto * lora : loras) {
        delete lora;
    }
}

void llama_model_base::load_stats(llama_model_loader & ml) {
    pimpl->n_elements = ml.n_elements;
    pimpl->n_bytes = ml.n_bytes;
}

void llama_model_base::load_hparams(llama_model_loader & ml) {
    const gguf_context * ctx = ml.metadata;

    // get metadata as string
    for (int i = 0; i < gguf_get_n_kv(ctx); i++) {
        gguf_type type = gguf_get_kv_type(ctx, i);
        if (type == GGUF_TYPE_ARRAY) {
            continue;
        }
        const char * name = gguf_get_key(ctx, i);
        const std::string value = gguf_kv_to_str(ctx, i);
        gguf_kv.emplace(name, value);
    }

    // get general kv
    ml.get_key(LLM_KV_GENERAL_NAME, name, false);

    // everything past this point is not vocab-related
    // for CLIP models, we only need to load tensors, no hparams
    if (hparams.vocab_only || ml.get_arch() == LLM_ARCH_CLIP) {
        return;
    }

    ml.get_key(LLM_KV_CONTEXT_LENGTH,          hparams.n_ctx_train);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH,        hparams.n_embd);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH_OUT,    hparams.n_embd_out_impl, false);
    ml.get_key(LLM_KV_ATTENTION_CAUSAL,        hparams.causal_attn,     false);
    ml.get_key(LLM_KV_POOLING_TYPE,            hparams.pooling_type,    false);
    ml.get_key(LLM_KV_BLOCK_COUNT,             hparams.n_layer);
    ml.get_key(LLM_KV_EXPERT_COUNT,            hparams.n_expert,        false);
    ml.get_key(LLM_KV_EXPERT_USED_COUNT,       hparams.n_expert_used,   false);
    ml.get_key(LLM_KV_EXPERT_GROUP_COUNT,      hparams.n_expert_groups, false);
    ml.get_key(LLM_KV_EXPERT_GROUP_USED_COUNT, hparams.n_group_used,    false);

    if (arch == LLM_ARCH_HUNYUAN_VL || arch == LLM_ARCH_HUNYUAN_DENSE) {
        if (hparams.n_expert <= 1) {
            hparams.n_expert      = 0;
            hparams.n_expert_used = 0;
        }
    }

    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        ml.get_key(LLM_KV_FEATURES_LENGTH,  hparams.n_embd);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH, hparams.n_embd_out_impl);

        ml.get_key(LLM_KV_POSNET_EMBEDDING_LENGTH, hparams.posnet.n_embd);
        ml.get_key(LLM_KV_POSNET_BLOCK_COUNT,      hparams.posnet.n_layer);

        ml.get_key(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, hparams.convnext.n_embd);
        ml.get_key(LLM_KV_CONVNEXT_BLOCK_COUNT,      hparams.convnext.n_layer);
    }

    GGML_ASSERT(hparams.n_expert <= LLAMA_MAX_EXPERTS);
    GGML_ASSERT(hparams.n_expert_used <= hparams.n_expert);
    if (hparams.n_expert > 0) {
        GGML_ASSERT(hparams.n_expert_used > 0);
        GGML_ASSERT(hparams.n_expert_groups < hparams.n_expert);
        if (hparams.n_expert_groups > 1) {
            GGML_ASSERT(hparams.n_expert % hparams.n_expert_groups == 0);
            GGML_ASSERT(hparams.n_group_used > 0);
            GGML_ASSERT(hparams.n_group_used < hparams.n_expert_groups);
        }
    } else {
        GGML_ASSERT(hparams.n_expert_used == 0);
        GGML_ASSERT(hparams.n_expert_groups == 0);
    }

    std::fill(hparams.n_head_arr.begin(),    hparams.n_head_arr.end(),    0);
    std::fill(hparams.n_head_kv_arr.begin(), hparams.n_head_kv_arr.end(), 0);
    std::fill(hparams.n_ff_arr.begin(),      hparams.n_ff_arr.end(),      0);
    std::fill(
        hparams.recurrent_layer_arr.begin(),
        hparams.recurrent_layer_arr.end(),
        llm_arch_is_recurrent(ml.get_arch()));

    std::fill(hparams.rope_sections.begin(), hparams.rope_sections.end(), 0);
    std::fill(hparams.swa_layers.begin(), hparams.swa_layers.end(), 0);

    std::fill(hparams.xielu_alpha_n.begin(), hparams.xielu_alpha_n.end(), 0.0f);
    std::fill(hparams.xielu_alpha_p.begin(), hparams.xielu_alpha_p.end(), 0.0f);
    std::fill(hparams.xielu_beta.begin(), hparams.xielu_beta.end(), 0.0f);
    std::fill(hparams.xielu_eps.begin(), hparams.xielu_eps.end(), 0.0f);
    std::fill(hparams.swiglu_clamp_exp.begin(),   hparams.swiglu_clamp_exp.end(),   0.0f);
    std::fill(hparams.swiglu_clamp_shexp.begin(), hparams.swiglu_clamp_shexp.end(), 0.0f);

    ml.get_key_or_arr(LLM_KV_FEED_FORWARD_LENGTH,  hparams.n_ff_arr,   hparams.n_layer, false);
    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT, hparams.n_head_arr, hparams.n_layer, false);

    // n_head_kv is optional, default to n_head
    hparams.n_head_kv_arr = hparams.n_head_arr;

    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT_KV, hparams.n_head_kv_arr, hparams.n_layer, false);

    bool rope_finetuned = false;
    ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
    hparams.rope_finetuned = rope_finetuned;

    hparams.n_ctx_orig_yarn = hparams.n_ctx_train;
    ml.get_key(LLM_KV_ROPE_SCALING_ORIG_CTX_LEN, hparams.n_ctx_orig_yarn, false);

    // rope_freq_base (optional)
    hparams.rope_freq_base_train = 10000.0f;
    ml.get_key(LLM_KV_ROPE_FREQ_BASE, hparams.rope_freq_base_train, false);

    std::string rope_scaling("linear");
    ml.get_key(LLM_KV_ROPE_SCALING_TYPE, rope_scaling, false);
    hparams.rope_scaling_type_train = llama_rope_scaling_type_from_string(rope_scaling);
    GGML_ASSERT(hparams.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED);

    // TODO: Handle SWA metadata similarly when models start implementing it
    // rope_freq_scale (inverse of the kv) is optional
    float ropescale = 0.0f;
    if (!ml.get_key(LLM_KV_ROPE_SCALING_FACTOR, ropescale, false)) {
        // try the old key name
        ml.get_key(LLM_KV_ROPE_SCALE_LINEAR, ropescale, false);
    }
    hparams.rope_freq_scale_train = ropescale == 0.0f ? 1.0f : 1.0f/ropescale;

    ml.get_key(LLM_KV_ROPE_SCALING_ATTN_FACTOR, hparams.rope_attn_factor, false);
    ml.get_key(LLM_KV_ROPE_SCALING_ALPHA,       hparams.rope_scaling_alpha, false);

    // non-transformer models do not have attention heads
    if (hparams.n_head() > 0) {
        // gpt-neox n_rot = rotary_pct * (n_embd / n_head)
        // gpt-j n_rot = rotary_dim

        hparams.n_embd_head_k_full = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH, hparams.n_embd_head_k_full, false);

        hparams.n_embd_head_v_full = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH, hparams.n_embd_head_v_full, false);

        // sanity check for n_rot (optional)
        hparams.n_rot_full = hparams.n_embd_head_k_full;

        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot_full, false);

        if (arch == LLM_ARCH_LLAMA || arch == LLM_ARCH_DECI || arch == LLM_ARCH_FALCON || arch == LLM_ARCH_LLAMA_EMBED) {
            if (hparams.n_rot_full != hparams.n_embd_head_k_full) {
                throw std::runtime_error(format("invalid n_rot: %u, expected %u", hparams.n_rot_full, hparams.n_embd_head_k_full));
            }
        }
    } else {
        hparams.n_rot_full = 0;
        hparams.n_embd_head_k_full = 0;
        hparams.n_embd_head_v_full = 0;
    }

    // head size and n_rot for SWA layers
    {
        hparams.n_embd_head_k_swa = hparams.n_embd_head_k_full;
        hparams.n_embd_head_v_swa = hparams.n_embd_head_v_full;
        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA, hparams.n_embd_head_k_swa, false);
        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA, hparams.n_embd_head_v_swa, false);

        hparams.n_rot_swa = hparams.n_rot_full;
        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT_SWA, hparams.n_rot_swa, false);
    }

    // for classifier models
    ml.get_arr(LLM_KV_CLASSIFIER_OUTPUT_LABELS, classifier_labels, false);
    if (!classifier_labels.empty()) {
        hparams.n_cls_out = classifier_labels.size();
    }

    // per-arch hparams
    load_arch_hparams(ml);

    pimpl->n_bytes = ml.n_bytes;

    pimpl->desc_str = arch_name() + " " + type_name() + " " + ml.ftype_name();

    if (hparams.f_max_alibi_bias > 0.0f) {
        hparams.use_alibi = true;
    }

    hparams.rope_type = llama_model_rope_type(this);
}

void llama_model_base::load_vocab(llama_model_loader & ml) {
    const auto kv = LLM_KV(arch);

    vocab.load(ml, kv);
}

bool llama_model_base::load_tensors(llama_model_loader & ml) {
    const auto & split_mode   = params.split_mode;
    const auto & use_mlock    = params.use_mlock;
    const auto & tensor_split = params.tensor_split;

    const int n_layer      = hparams.n_layer;
    const int n_gpu_layers = this->n_gpu_layers();

    const bool use_mmap_buffer = true;

    this->ml = &ml; // to be used by create_tensor() and load_arch_tensors()

    LLAMA_LOG_INFO("%s: loading model tensors, this can take a while... (mmap = %s, direct_io = %s)\n",
        __func__, ml.use_mmap ? "true" : "false", ml.use_direct_io ? "true" : "false");

    // build a list of buffer types for the CPU and GPU devices
    pimpl->cpu_buft_list = make_cpu_buft_list(devices, params.use_extra_bufts, params.no_host);
    for (const auto & dev : devices) {
        buft_list_t buft_list = make_gpu_buft_list(dev.dev, split_mode, tensor_split);
        // add CPU buffer types as a fallback
        buft_list.insert(buft_list.end(), pimpl->cpu_buft_list.begin(), pimpl->cpu_buft_list.end());
        pimpl->gpu_buft_list.emplace(dev.dev, std::move(buft_list));
    }

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev == nullptr) {
        throw std::runtime_error(format("%s: no CPU backend found", __func__));
    }

    // calculate the split points
    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + n_devices(), [](float x) { return x == 0.0f; });
    std::vector<float> splits(n_devices());
    if (all_zero) {
        // default split, by free memory
        for (size_t i = 0; i < n_devices(); ++i) {
            ggml_backend_dev_t dev = devices[i].dev;
            size_t total;
            size_t free;
            ggml_backend_dev_memory(dev, &free, &total);

            // devices can return 0 bytes for free and total memory if they do not
            // have any to report. in this case, we will use the host memory as a fallback
            // fixes: https://github.com/ggml-org/llama.cpp/issues/18577
            if (free == 0 && total == 0) {
                ggml_backend_dev_memory(cpu_dev, &free, &total);
            }
            splits[i] = free;
        }
    } else {
        std::copy(tensor_split, tensor_split + n_devices(), splits.begin());
    }

    // sum and normalize the splits to get the split points
    float split_sum = 0.0f;
    for (size_t i = 0; i < n_devices(); ++i) {
        split_sum += splits[i];
        splits[i] = split_sum;
    }
    for (size_t i = 0; i < n_devices(); ++i) {
        splits[i] /= split_sum;
    }

    const int i_gpu_start = std::max(int(hparams.n_layer) + 1 - n_gpu_layers, 0);
    const int act_gpu_layers = devices.empty() ? 0 : std::min(n_gpu_layers, int(n_layer) + 1);
    auto get_layer_buft_list = [&](int il) -> llama_model::impl::layer_dev {
        const bool is_swa = il < int(hparams.n_layer) && hparams.is_swa(il);
        if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
            LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(cpu_dev), is_swa);
            return {cpu_dev, &pimpl->cpu_buft_list};
        }
        const int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + n_devices(), float(il - i_gpu_start)/act_gpu_layers) - splits.begin();
        auto * dev = devices.at(layer_gpu).dev;
        LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(dev), is_swa);
        return {dev, &pimpl->gpu_buft_list.at(dev)};
    };

    // assign the input layer
    // there is very little benefit to offloading the input layer, so always keep it on the CPU
    pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };

    // assign the repeating layers to the devices according to the splits
    pimpl->dev_layer.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        pimpl->dev_layer[il] = get_layer_buft_list(il);
    }

    // assign the output layer
    pimpl->dev_output = get_layer_buft_list(n_layer);

    const auto TENSOR_NOT_REQUIRED = llama_model_loader::TENSOR_NOT_REQUIRED;

    // create tensors for the weights
    {
        // TODO: move to a separate function
        const auto tn = LLM_TN(arch);

        const int64_t n_expert      = hparams.n_expert;
        const int64_t n_expert_used = hparams.n_expert_used;

        if (n_expert > 0 && n_expert_used == 0) {
            throw std::runtime_error("model has expert layers but no expert layers are used");
        }

        layers.resize(n_layer);

        // call the per-model loading function
        load_arch_tensors(ml);

        // generic pass: load optional per-tensor/per-expert ".scale" tensors (e.g. NVFP4 scale2)
        // this avoids having to add scale loading to every architecture
        for (int i = 0; i < n_layer; ++i) {
            auto & layer = layers[i];

            // attention weight scales (per-tensor, shape {1})
            if (!layer.wq_s && layer.wq) {
                layer.wq_s = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wk_s && layer.wk) {
                layer.wk_s = create_tensor(tn(LLM_TENSOR_ATTN_K,   "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wv_s && layer.wv) {
                layer.wv_s = create_tensor(tn(LLM_TENSOR_ATTN_V,   "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wo_s && layer.wo) {
                layer.wo_s = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_s && layer.wqkv) {
                layer.wqkv_s = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_gate_s && layer.wqkv_gate) {
                layer.wqkv_gate_s = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }

            // dense FFN weight scales (per-tensor, shape {1})
            if (!layer.ffn_gate_s && layer.ffn_gate) {
                layer.ffn_gate_s = create_tensor(tn(LLM_TENSOR_FFN_GATE, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_s && layer.ffn_down) {
                layer.ffn_down_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_s && layer.ffn_up) {
                layer.ffn_up_s = create_tensor(tn(LLM_TENSOR_FFN_UP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_shexp_s && layer.ffn_gate_shexp) {
                layer.ffn_gate_shexp_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_shexp_s && layer.ffn_down_shexp) {
                layer.ffn_down_shexp_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_shexp_s && layer.ffn_up_shexp) {
                layer.ffn_up_shexp_s = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }

            // MoE expert weight scales (per-expert, shape {n_expert})
            if (!layer.ffn_gate_exps_s && layer.ffn_gate_exps) {
                layer.ffn_gate_exps_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_exps_s && layer.ffn_down_exps) {
                layer.ffn_down_exps_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_exps_s && layer.ffn_up_exps) {
                layer.ffn_up_exps_s = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }

            // recurrent / linear-attention weight scales (per-tensor, shape {1})
            if (!layer.ssm_in_s && layer.ssm_in) {
                layer.ssm_in_s = create_tensor(tn(LLM_TENSOR_SSM_IN, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_out_s && layer.ssm_out) {
                layer.ssm_out_s = create_tensor(tn(LLM_TENSOR_SSM_OUT, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_alpha_s && layer.ssm_alpha) {
                layer.ssm_alpha_s = create_tensor(tn(LLM_TENSOR_SSM_ALPHA, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_beta_s && layer.ssm_beta) {
                layer.ssm_beta_s = create_tensor(tn(LLM_TENSOR_SSM_BETA, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }

            // input scales
            if (!layer.wq_in_s && layer.wq) {
                layer.wq_in_s = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wk_in_s && layer.wk) {
                layer.wk_in_s = create_tensor(tn(LLM_TENSOR_ATTN_K,   "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wv_in_s && layer.wv) {
                layer.wv_in_s = create_tensor(tn(LLM_TENSOR_ATTN_V,   "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wo_in_s && layer.wo) {
                layer.wo_in_s = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_in_s && layer.wqkv) {
                layer.wqkv_in_s = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.wqkv_gate_in_s && layer.wqkv_gate) {
                layer.wqkv_gate_in_s = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_in_s && layer.ffn_gate) {
                layer.ffn_gate_in_s = create_tensor(tn(LLM_TENSOR_FFN_GATE, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_in_s && layer.ffn_down) {
                layer.ffn_down_in_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_in_s && layer.ffn_up) {
                layer.ffn_up_in_s = create_tensor(tn(LLM_TENSOR_FFN_UP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_exps_in_s && layer.ffn_gate_exps) {
                layer.ffn_gate_exps_in_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "input_scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_exps_in_s && layer.ffn_down_exps) {
                layer.ffn_down_exps_in_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "input_scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_exps_in_s && layer.ffn_up_exps) {
                layer.ffn_up_exps_in_s = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "input_scale", i), {n_expert}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_gate_shexp_in_s && layer.ffn_gate_shexp) {
                layer.ffn_gate_shexp_in_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_down_shexp_in_s && layer.ffn_down_shexp) {
                layer.ffn_down_shexp_in_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ffn_up_shexp_in_s && layer.ffn_up_shexp) {
                layer.ffn_up_shexp_in_s = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_in_in_s && layer.ssm_in) {
                layer.ssm_in_in_s = create_tensor(tn(LLM_TENSOR_SSM_IN, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_out_in_s && layer.ssm_out) {
                layer.ssm_out_in_s = create_tensor(tn(LLM_TENSOR_SSM_OUT, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_alpha_in_s && layer.ssm_alpha) {
                layer.ssm_alpha_in_s = create_tensor(tn(LLM_TENSOR_SSM_ALPHA, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
            if (!layer.ssm_beta_in_s && layer.ssm_beta) {
                layer.ssm_beta_in_s = create_tensor(tn(LLM_TENSOR_SSM_BETA, "input_scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
        }
    }

    ml.done_getting_tensors();

    // populate tensors_by_name
    for (auto & [_, ctx_ptr] : ml.ctx_map) {
        for (auto * cur = ggml_get_first_tensor(ctx_ptr.get()); cur != NULL; cur = ggml_get_next_tensor(ctx_ptr.get(), cur)) {
            tensors_by_name.emplace_back(ggml_get_name(cur), cur);
        }
    }

    ml.init_mappings(true, use_mlock ? &pimpl->mlock_mmaps : nullptr);
    pimpl->mappings.reserve(ml.mappings.size());

    // create the backend buffers
    std::vector<std::pair<ggml_context *, llama_buf_map>> ctx_buf_maps;
    ctx_buf_maps.reserve(ml.ctx_map.size());

    // Ensure we have enough capacity for the maximum backend buffer we will potentially create
    const size_t n_max_backend_buffer = ml.ctx_map.size() * ml.files.size();
    pimpl->ctxs_bufs.reserve(n_max_backend_buffer);

    for (auto & [buft, ctx_ptr] : ml.ctx_map) {
        ggml_context * ctx = ctx_ptr.get();

        // skip contexts without tensors
        if (ggml_get_first_tensor(ctx) == nullptr) {
            continue;
        }

        llama_buf_map buf_map;
        buf_map.reserve(n_max_backend_buffer);

        // check if it is possible to use buffer_from_host_ptr with this buffer type
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            // FIXME: workaround for CPU backend buft having a NULL device
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (!dev) {
                throw std::runtime_error(format("%s: no CPU backend found", __func__));
            }
        }
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        bool buffer_from_host_ptr_supported = props.caps.buffer_from_host_ptr;
        bool is_default_buft = buft == ggml_backend_dev_buffer_type(dev);

        std::vector<ggml_backend_buffer_ptr> bufs;
        if (ml.use_mmap && use_mmap_buffer && buffer_from_host_ptr_supported && is_default_buft) {
            GGML_ASSERT(!ml.no_alloc);
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                // only the mmap region containing the tensors in the model is mapped to the backend buffer
                // this is important for metal with apple silicon: if the entire model could be mapped to a metal buffer,
                //     then we could just use metal for all layers
                // this allows using partial offloading when the model size exceeds the metal buffer size, but not the RAM size
                void * addr = nullptr;
                size_t first, last; // NOLINT
                ml.get_mapping_range(&first, &last, &addr, idx, ctx);
                if (first >= last) {
                    continue;
                }
                const size_t max_size = ggml_get_max_tensor_size(ctx);
                ggml_backend_buffer_t buf = ggml_backend_dev_buffer_from_host_ptr(dev, (char *) addr + first, last - first, max_size);
                if (buf == nullptr) {
                    throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
                }
                bufs.emplace_back(buf);
                buf_map.emplace(idx, buf);
            }
        } else {
            ggml_backend_buffer_t buf;
            if (ml.no_alloc) {
                buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
                for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                    t->buffer = buf; // set dummy buffer for weights so that the backend scheduler won't try to allocate them
                }
            } else {
                buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft); // real buffer
            }
            if (buf == nullptr) {
                throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(buft)));
            }
            if (use_mlock && ggml_backend_buffer_is_host(buf)) {
                pimpl->mlock_bufs.emplace_back(new llama_mlock);
                auto & mlock_buf = pimpl->mlock_bufs.back();
                mlock_buf->init   (ggml_backend_buffer_get_base(buf));
                mlock_buf->grow_to(ggml_backend_buffer_get_size(buf));
            }
            bufs.emplace_back(buf);
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                buf_map.emplace(idx, buf);
            }
        }

        for (auto & buf : bufs) {
            // indicate that this buffer contains weights
            // this is used by ggml_backend_sched to improve op scheduling: ops that use a weight are preferably scheduled to the backend that contains the weight
            ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        pimpl->ctxs_bufs.emplace_back(std::move(ctx_ptr), std::move(bufs));

        ctx_buf_maps.emplace_back(ctx, buf_map);
    }

    if (llama_supports_gpu_offload()) {
        const int n_gpu = std::min(n_gpu_layers, int(hparams.n_layer));

        int n_repeating = n_gpu;
        if (n_repeating > 0) {
            LLAMA_LOG_INFO("%s: offloading output layer to GPU\n", __func__);
            n_repeating--;
        }
        LLAMA_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, n_repeating);

        const int max_backend_supported_layers = hparams.n_layer + 1;
        const int max_offloadable_layers       = hparams.n_layer + 1;

        LLAMA_LOG_INFO("%s: offloaded %d/%d layers to GPU\n", __func__, std::min(n_gpu_layers, max_offloadable_layers), max_backend_supported_layers);
    }

    // print memory requirements per buffer type
    for (auto & [_, bufs] : pimpl->ctxs_bufs) {
        for (auto & buf: bufs) {
            LLAMA_LOG_INFO("%s: %12s model buffer size = %8.2f MiB\n",
                __func__, ggml_backend_buffer_name(buf.get()), ggml_backend_buffer_get_size(buf.get()) / 1024.0 / 1024.0);
        }
    }

    if (ml.no_alloc) {
        return true;
    }

    // load tensor data
    for (auto & [ctx, buf_map] : ctx_buf_maps) {
        if (!ml.load_all_data(ctx, buf_map, use_mlock ? &pimpl->mlock_mmaps : NULL, params.progress_callback, params.progress_callback_user_data)) {
            return false;
        }
    }

    if (use_mmap_buffer) {
        for (auto & mapping : ml.mappings) {
            pimpl->mappings.emplace_back(std::move(mapping));
        }
    }

    return true;
}

ggml_tensor * llama_model_base::create_tensor(llama_model_loader & ml, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) {
    const buft_list_t * buft_list_layer = tn.bid == -1 ? nullptr : pimpl->dev_layer.at(tn.bid).buft_list;
    return ml.create_tensor(
        hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, buft_list_layer,
        tn, ne, flags);
}

std::string llama_model::arch_name() const {
    return llm_arch_name(arch);
}

std::string llama_model::type_name() const {
    return llm_type_name(type);
}

std::string llama_model::desc() const {
    return pimpl->desc_str;
}

size_t llama_model::size() const {
    return pimpl->n_bytes;
}

size_t llama_model::n_tensors() const {
    return tensors_by_name.size();
}

size_t llama_model::n_devices() const {
    return devices.size();
}

const float * llama_model::tensor_split() const {
    return params.tensor_split;
}

uint32_t llama_model::n_gpu_layers() const {
    return params.n_gpu_layers >= 0 ? params.n_gpu_layers : hparams.n_layer + 1;
}

llama_split_mode llama_model::split_mode() const {
    return params.split_mode;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_model::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, bufs] : pimpl->ctxs_bufs) {
        if (hparams.no_alloc) {
            GGML_ASSERT(bufs.size() == 1);
            ggml_backend_buffer_t buf = bufs[0].get();
            GGML_ASSERT(ggml_backend_buffer_get_base(buf) == nullptr);
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            for (const auto & buf : bufs) {
                // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
                ret[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
            }
        }
    }
    return ret;
}

uint64_t llama_model::n_elements() const {
    return pimpl->n_elements;
}

void llama_model::print_info() const {
    const std::string rope_scaling_type = llama_rope_scaling_type_name(hparams.rope_scaling_type_train);

    auto print_f = [](const std::function<uint32_t(uint32_t)> & f, uint32_t n) {
        bool is_var = false;

        std::vector<uint32_t> v;
        for (uint32_t i = 0; i < n; ++i) {
            v.push_back(f(i));
            if (v[i] != v[0]) {
                is_var = true;
            }
        }

        std::stringstream ss;

        if (is_var) {
            ss << "[";
            for (uint32_t i = 0; i < n; ++i) {
                ss << v[i];
                if (i < n - 1) {
                    ss << ", ";
                }
            }
            ss << "]";
        } else {
            ss << v[0];
        }

        return ss.str();
    };

    // hparams
    LLAMA_LOG_INFO("%s: arch                  = %s\n",     __func__, arch_name().c_str());
    LLAMA_LOG_INFO("%s: vocab_only            = %d\n",     __func__, hparams.vocab_only);
    LLAMA_LOG_INFO("%s: no_alloc              = %d\n",     __func__, hparams.no_alloc);

    if (!hparams.vocab_only) {
        LLAMA_LOG_INFO("%s: n_ctx_train           = %u\n",     __func__, hparams.n_ctx_train);
        LLAMA_LOG_INFO("%s: n_embd                = %u\n",     __func__, hparams.n_embd);
        LLAMA_LOG_INFO("%s: n_embd_inp            = %u\n",     __func__, hparams.n_embd_inp());
        LLAMA_LOG_INFO("%s: n_layer               = %u\n",     __func__, hparams.n_layer);
        LLAMA_LOG_INFO("%s: n_head                = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_head(il);    }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_head_kv             = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_head_kv(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_rot                 = %u\n",     __func__, hparams.n_rot_full);
        LLAMA_LOG_INFO("%s: n_swa                 = %u\n",     __func__, hparams.n_swa);
        LLAMA_LOG_INFO("%s: is_swa_any            = %u\n",     __func__, hparams.is_swa_any());
        LLAMA_LOG_INFO("%s: n_embd_head_k         = %u\n",     __func__, hparams.n_embd_head_k_full);
        LLAMA_LOG_INFO("%s: n_embd_head_v         = %u\n",     __func__, hparams.n_embd_head_v_full);
        LLAMA_LOG_INFO("%s: n_gqa                 = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_gqa(il);        }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_embd_k_gqa          = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_embd_k_gqa(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_embd_v_gqa          = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_embd_v_gqa(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: f_norm_eps            = %.1e\n",   __func__, hparams.f_norm_eps);
        LLAMA_LOG_INFO("%s: f_norm_rms_eps        = %.1e\n",   __func__, hparams.f_norm_rms_eps);
        LLAMA_LOG_INFO("%s: f_clamp_kqv           = %.1e\n",   __func__, hparams.f_clamp_kqv);
        LLAMA_LOG_INFO("%s: f_max_alibi_bias      = %.1e\n",   __func__, hparams.f_max_alibi_bias);
        LLAMA_LOG_INFO("%s: f_logit_scale         = %.1e\n",   __func__, hparams.f_logit_scale);
        LLAMA_LOG_INFO("%s: f_attn_scale          = %.1e\n",   __func__, hparams.f_attention_scale);
        LLAMA_LOG_INFO("%s: f_attn_value_scale    = %.4f\n",   __func__, hparams.f_attn_value_scale);
        LLAMA_LOG_INFO("%s: n_ff                  = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_ff(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_expert              = %u\n",     __func__, hparams.n_expert);
        LLAMA_LOG_INFO("%s: n_expert_used         = %u\n",     __func__, hparams.n_expert_used);
        LLAMA_LOG_INFO("%s: n_expert_groups       = %d\n",     __func__, hparams.n_expert_groups);
        LLAMA_LOG_INFO("%s: n_group_used          = %d\n",     __func__, hparams.n_group_used);
        LLAMA_LOG_INFO("%s: causal attn           = %d\n",     __func__, hparams.causal_attn);
        LLAMA_LOG_INFO("%s: pooling type          = %d\n",     __func__, hparams.pooling_type);
        LLAMA_LOG_INFO("%s: rope type             = %d\n",     __func__, hparams.rope_type);
        LLAMA_LOG_INFO("%s: rope scaling          = %s\n",     __func__, rope_scaling_type.c_str());
        LLAMA_LOG_INFO("%s: freq_base_train       = %.1f\n",   __func__, hparams.rope_freq_base_train);
        LLAMA_LOG_INFO("%s: freq_scale_train      = %g\n",     __func__, hparams.rope_freq_scale_train);
        if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
            LLAMA_LOG_INFO("%s: freq_base_swa         = %.1f\n",   __func__, hparams.rope_freq_base_train_swa);
            LLAMA_LOG_INFO("%s: freq_scale_swa        = %g\n",     __func__, hparams.rope_freq_scale_train_swa);
            LLAMA_LOG_INFO("%s: n_embd_head_k_swa     = %u\n",     __func__, hparams.n_embd_head_k_swa);
            LLAMA_LOG_INFO("%s: n_embd_head_v_swa     = %u\n",     __func__, hparams.n_embd_head_v_swa);
            LLAMA_LOG_INFO("%s: n_rot_swa             = %u\n",     __func__, hparams.n_rot_swa);
        }
        LLAMA_LOG_INFO("%s: n_ctx_orig_yarn       = %u\n",     __func__, hparams.n_ctx_orig_yarn);
        LLAMA_LOG_INFO("%s: rope_yarn_log_mul     = %.4f\n",   __func__, hparams.rope_yarn_log_mul);
        LLAMA_LOG_INFO("%s: rope_finetuned        = %s\n",     __func__, hparams.rope_finetuned ? "yes" : "unknown");
        // MRoPE (Multi-axis Rotary Position Embedding) sections
        if (const auto & s = hparams.rope_sections; s[0] || s[1] || s[2] || s[3]) {
            LLAMA_LOG_INFO("%s: mrope sections        = [%d, %d, %d, %d]\n", __func__, s[0], s[1], s[2], s[3]);
        }
        if (!classifier_labels.empty()) {
            LLAMA_LOG_INFO("%s: n_cls_out             = %u\n", __func__, hparams.n_cls_out);

            size_t i = 0;
            for (const auto & label : classifier_labels) {
                LLAMA_LOG_INFO("%s: cls_label[%2zu]         = %s\n", __func__, i++, label.c_str());
            }
        }

        if (arch == LLM_ARCH_MAMBA ||
                arch == LLM_ARCH_MAMBA2 ||
                arch == LLM_ARCH_JAMBA ||
                arch == LLM_ARCH_FALCON_H1 ||
                arch == LLM_ARCH_PLAMO2 ||
                arch == LLM_ARCH_GRANITE_HYBRID ||
                arch == LLM_ARCH_QWEN3NEXT ||
                arch == LLM_ARCH_QWEN35 ||
                arch == LLM_ARCH_QWEN35MOE ||
                arch == LLM_ARCH_NEMOTRON_H ||
                arch == LLM_ARCH_NEMOTRON_H_MOE) {
            LLAMA_LOG_INFO("%s: ssm_d_conv            = %u\n",     __func__, hparams.ssm_d_conv);
            LLAMA_LOG_INFO("%s: ssm_d_inner           = %u\n",     __func__, hparams.ssm_d_inner);
            LLAMA_LOG_INFO("%s: ssm_d_state           = %u\n",     __func__, hparams.ssm_d_state);
            LLAMA_LOG_INFO("%s: ssm_dt_rank           = %u\n",     __func__, hparams.ssm_dt_rank);
            LLAMA_LOG_INFO("%s: ssm_n_group           = %u\n",     __func__, hparams.ssm_n_group);
            LLAMA_LOG_INFO("%s: ssm_dt_b_c_rms        = %d\n",     __func__, hparams.ssm_dt_b_c_rms);
        }

        LLAMA_LOG_INFO("%s: model type            = %s\n",     __func__, type_name().c_str());
        if (pimpl->n_elements >= 1e12) {
            LLAMA_LOG_INFO("%s: model params          = %.2f T\n", __func__, pimpl->n_elements*1e-12);
        } else if (pimpl->n_elements >= 1e9) {
            LLAMA_LOG_INFO("%s: model params          = %.2f B\n", __func__, pimpl->n_elements*1e-9);
        } else if (pimpl->n_elements >= 1e6) {
            LLAMA_LOG_INFO("%s: model params          = %.2f M\n", __func__, pimpl->n_elements*1e-6);
        } else {
            LLAMA_LOG_INFO("%s: model params          = %.2f K\n", __func__, pimpl->n_elements*1e-3);
        }

        // general kv
        LLAMA_LOG_INFO("%s: general.name          = %s\n",    __func__, name.c_str());

        if (arch == LLM_ARCH_DEEPSEEK) {
            LLAMA_LOG_INFO("%s: n_layer_dense_lead    = %d\n",     __func__, hparams.n_layer_dense_lead);
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp);
            LLAMA_LOG_INFO("%s: n_expert_shared       = %d\n",     __func__, hparams.n_expert_shared);
            LLAMA_LOG_INFO("%s: expert_weights_scale  = %.1f\n",   __func__, hparams.expert_weights_scale);
        }

        if (arch == LLM_ARCH_DEEPSEEK2 || arch == LLM_ARCH_DEEPSEEK2OCR || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_MISTRAL4) {
            LLAMA_LOG_INFO("%s: n_layer_dense_lead    = %d\n",     __func__, hparams.n_layer_dense_lead);
            LLAMA_LOG_INFO("%s: n_lora_q              = %d\n",     __func__, hparams.n_lora_q);
            LLAMA_LOG_INFO("%s: n_lora_kv             = %d\n",     __func__, hparams.n_lora_kv);
            LLAMA_LOG_INFO("%s: n_embd_head_k_mla     = %d\n",     __func__, hparams.n_embd_head_k_mla());
            LLAMA_LOG_INFO("%s: n_embd_head_v_mla     = %d\n",     __func__, hparams.n_embd_head_v_mla());
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp);
            LLAMA_LOG_INFO("%s: n_expert_shared       = %d\n",     __func__, hparams.n_expert_shared);
            LLAMA_LOG_INFO("%s: expert_weights_scale  = %.1f\n",   __func__, hparams.expert_weights_scale);
            LLAMA_LOG_INFO("%s: expert_weights_norm   = %d\n",     __func__, hparams.expert_weights_norm);
            LLAMA_LOG_INFO("%s: expert_gating_func    = %s\n",     __func__, llama_expert_gating_func_name((llama_expert_gating_func_type) hparams.expert_gating_func));
        }

        if (arch == LLM_ARCH_QWEN2MOE) {
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp);
            LLAMA_LOG_INFO("%s: n_ff_shexp            = %d\n",     __func__, hparams.n_ff_shexp);
        }

        if (arch == LLM_ARCH_QWEN3MOE || arch == LLM_ARCH_OPENAI_MOE || arch == LLM_ARCH_QWEN3VLMOE || arch == LLM_ARCH_RND1) {
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp);
        }

        if (arch == LLM_ARCH_MINICPM ||
                arch == LLM_ARCH_GRANITE ||
                arch == LLM_ARCH_GRANITE_MOE ||
                arch == LLM_ARCH_GRANITE_HYBRID ||
                arch == LLM_ARCH_NEMOTRON_H_MOE) {
            LLAMA_LOG_INFO("%s: f_embedding_scale     = %f\n", __func__, hparams.f_embedding_scale);
            LLAMA_LOG_INFO("%s: f_residual_scale      = %f\n", __func__, hparams.f_residual_scale);
            LLAMA_LOG_INFO("%s: f_attention_scale     = %f\n", __func__, hparams.f_attention_scale);
            LLAMA_LOG_INFO("%s: n_ff_shexp            = %d\n", __func__, hparams.n_ff_shexp);
        }

        if (arch == LLM_ARCH_BAILINGMOE) {
            LLAMA_LOG_INFO("%s: n_layer_dense_lead    = %d\n",     __func__, hparams.n_layer_dense_lead);
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp);
            LLAMA_LOG_INFO("%s: n_expert_shared       = %d\n",     __func__, hparams.n_expert_shared);
            LLAMA_LOG_INFO("%s: expert_weights_scale  = %.1f\n",   __func__, hparams.expert_weights_scale);
            LLAMA_LOG_INFO("%s: expert_weights_norm   = %d\n",     __func__, hparams.expert_weights_norm);
        }

        if (arch == LLM_ARCH_BAILINGMOE2) {
            LLAMA_LOG_INFO("%s: n_layer_dense_lead    = %d\n",     __func__, hparams.n_layer_dense_lead);
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp);
            LLAMA_LOG_INFO("%s: n_ff_shexp            = %d\n",     __func__, hparams.n_ff_shexp);
            LLAMA_LOG_INFO("%s: n_expert_shared       = %d\n",     __func__, hparams.n_expert_shared);
            LLAMA_LOG_INFO("%s: expert_weights_scale  = %.1f\n",   __func__, hparams.expert_weights_scale);
            LLAMA_LOG_INFO("%s: expert_weights_norm   = %d\n",     __func__, hparams.expert_weights_norm);
            LLAMA_LOG_INFO("%s: expert_gating_func    = %s\n",     __func__, llama_expert_gating_func_name((llama_expert_gating_func_type) hparams.expert_gating_func));
            LLAMA_LOG_INFO("%s: nextn_predict_layers  = %d\n",     __func__, hparams.nextn_predict_layers);
        }

        if (arch == LLM_ARCH_SMALLTHINKER || arch == LLM_ARCH_LFM2MOE) {
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp);
            LLAMA_LOG_INFO("%s: expert_gating_func    = %s\n",     __func__, llama_expert_gating_func_name((llama_expert_gating_func_type) hparams.expert_gating_func));
        }

        if (arch == LLM_ARCH_GROVEMOE) {
            LLAMA_LOG_INFO("%s: n_ff_exp              = %d\n",     __func__, hparams.n_ff_exp);
            LLAMA_LOG_INFO("%s: n_ff_chexp            = %d\n",     __func__, hparams.n_ff_chexp);
            LLAMA_LOG_INFO("%s: n_group_experts       = %d\n",     __func__, hparams.n_group_experts);
            LLAMA_LOG_INFO("%s: expert_group_scale    = %.2f\n",   __func__, hparams.expert_group_scale);
        }
    }

    vocab.print_info();
}

ggml_backend_dev_t llama_model::dev_layer(int il) const {
    return pimpl->dev_layer.at(il).dev;
}

ggml_backend_dev_t llama_model::dev_output() const {
    return pimpl->dev_output.dev;
}

template<typename F>
static bool buft_supported(ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev, F & fn) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx { ggml_init(params) };
    if (!ctx) {
        throw std::runtime_error(format("failed to create ggml context"));
    }

    ggml_backend_buffer_ptr buf { ggml_backend_buft_alloc_buffer(buft, 0) };
    ggml_tensor * op_tensor = fn(ctx.get());
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op_tensor->src[i] != nullptr) {
            assert(op_tensor->src[i]->buffer == nullptr);
            op_tensor->src[i]->buffer = buf.get();
        }
    }

    bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);

    return op_supported;
}

template<typename F>
static ggml_backend_buffer_type_t select_buft(const buft_list_t & buft_list, const F & fn) {
    for (const auto & cur : buft_list) {
        ggml_backend_dev_t cur_dev = cur.first;
        ggml_backend_buffer_type_t cur_buft = cur.second;
        if (buft_supported(cur_buft, cur_dev, fn)) {
            return cur_buft;
        }
    }

    throw std::runtime_error(format("no suitable buffer type found"));
}

ggml_backend_buffer_type_t llama_model::select_buft(int il) const {
    return ::select_buft(
            *pimpl->dev_layer.at(il).buft_list,
            [&](ggml_context * ctx) {
                ggml_tensor * cur = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
                ggml_tensor * layer_dir = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
                return ggml_add(ctx, cur, layer_dir);
            });
}

bool llama_model::has_tensor_overrides() const {
    return pimpl->has_tensor_overrides;
}

const ggml_tensor * llama_model::get_tensor(const char * name) const {
    auto it = std::find_if(tensors_by_name.begin(), tensors_by_name.end(),
            [name](const std::pair<std::string, ggml_tensor *> & it) {
                return it.first == name;
            });
    if (it == tensors_by_name.end()) {
        return nullptr;
    }

    return it->second;
}

float llama_model::get_rope_freq_base (const llama_cparams & cparams, int il) const {
    return hparams.is_swa(il) ? hparams.rope_freq_base_train_swa : cparams.rope_freq_base;
}

float llama_model::get_rope_freq_scale(const llama_cparams & cparams, int il) const {
    return hparams.is_swa(il) ? hparams.rope_freq_scale_train_swa : cparams.rope_freq_scale;
}

ggml_tensor * llama_model::get_rope_factors(const llama_cparams & cparams, int il) const {
    const uint32_t n_ctx_seq = cparams.n_ctx_seq;

    // choose long/short freq factors based on the context size
    if (layers[il].rope_freqs != nullptr) {
        return layers[il].rope_freqs;
    }

    if (n_ctx_seq > hparams.n_ctx_orig_yarn) {
        return layers[il].rope_long;
    }

    return layers[il].rope_short;
}

llama_memory_i * llama_model::create_memory(const llama_memory_params & params, const llama_cparams & cparams) const {
    llama_memory_i * res;

    switch (arch) {
        // Models that need specific instantiation should be handled in the
        // switch statement
        case LLM_ARCH_BERT:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_JINA_BERT_V3:
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_NOMIC_BERT_MOE:
        case LLM_ARCH_NEO_BERT:
        case LLM_ARCH_EUROBERT:
        case LLM_ARCH_WAVTOKENIZER_DEC:
        case LLM_ARCH_MODERN_BERT:
        case LLM_ARCH_GEMMA_EMBEDDING:
        case LLM_ARCH_DREAM:
        case LLM_ARCH_LLADA:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_RND1:
            {
                res = nullptr;
            } break;
        // Models that need standard caching should rely on recurrent/hybrid
        // checks
        default:
            {
                if (llm_arch_is_recurrent(arch)) {
                    res = new llama_memory_recurrent(
                            *this,
                            GGML_TYPE_F32,
                            GGML_TYPE_F32,
                            cparams.offload_kqv,
                            std::max((uint32_t) 1, cparams.n_seq_max),
                            cparams.n_seq_max,
                            nullptr);
                } else if (llm_arch_is_hybrid(arch)) {
                    // The main difference between hybrid architectures is the
                    // layer filters, so pick the right one here
                    llama_memory_hybrid::layer_filter_cb filter_attn = nullptr;
                    llama_memory_hybrid::layer_filter_cb filter_recr = nullptr;
                    if (arch == LLM_ARCH_FALCON_H1) {
                        filter_attn = [&](int32_t) { return true; };
                        filter_recr = [&](int32_t) { return true; };
                    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
                        filter_attn = [&](int32_t il) {
                            return !hparams.is_recurrent(il) && hparams.n_ff(il) == 0;
                        };
                        filter_recr = [&](int32_t il) {
                            return hparams.is_recurrent(il) && hparams.n_ff(il) == 0;
                        };
                    }

                    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
                        // Use hybrid-iswa for hybrid models with SWA
                        res = new llama_memory_hybrid_iswa(
                            /* model             */ *this,
                            /* attn_type_k       */ params.type_k,
                            /* attn_type_v       */ params.type_v,
                            /* attn_v_trans      */ !cparams.flash_attn,
                            /* attn_swa_full     */ params.swa_full,
                            /* attn_kv_size      */ cparams.n_ctx_seq,
                            /* attn_n_ubatch     */ cparams.n_ubatch,
                            /* attn_n_pad        */ 1,
                            /* recurrent_type_r  */ GGML_TYPE_F32,
                            /* recurrent_type_s  */ GGML_TYPE_F32,
                            /* recurrent_rs_size */ std::max((uint32_t) 1, cparams.n_seq_max),
                            /* n_seq_max         */ cparams.n_seq_max,
                            /* offload           */ cparams.offload_kqv,
                            /* unified           */ cparams.kv_unified,
                            /* filter_attn       */ std::move(filter_attn),
                            /* filter_recr       */ std::move(filter_recr));
                    } else {
                        res = new llama_memory_hybrid(
                            /* model             */ *this,
                            /* attn_type_k       */ params.type_k,
                            /* attn_type_v       */ params.type_v,
                            /* attn_v_trans      */ !cparams.flash_attn,
                            /* attn_kv_size      */ cparams.n_ctx_seq,
                            /* attn_n_pad        */ 1,
                            /* attn_n_swa        */ hparams.n_swa,
                            /* attn_swa_type     */ hparams.swa_type,
                            /* recurrent_type_k  */ GGML_TYPE_F32,
                            /* recurrent_type_v  */ GGML_TYPE_F32,
                            /* recurrent_kv_size */ std::max((uint32_t) 1, cparams.n_seq_max),
                            /* n_seq_max         */ cparams.n_seq_max,
                            /* offload           */ cparams.offload_kqv,
                            /* unified           */ cparams.kv_unified,
                            /* filter_attn       */ std::move(filter_attn),
                            /* filter_recr       */ std::move(filter_recr));
                    }
                } else {
                    llama_memory_i::layer_reuse_cb reuse = nullptr;

                    if (arch == LLM_ARCH_GEMMA3N || arch == LLM_ARCH_GEMMA4) {
                        reuse = [&](int32_t il) {
                            if (il >= (int32_t) hparams.n_layer_kv_from_start) {
                                return (int32_t) hparams.n_layer_kv_from_start - (hparams.is_swa(il) ? 2 : 1);
                            }

                            return -1;
                        };
                    }

                    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
                        GGML_ASSERT(hparams.is_swa_any());

                        res = new llama_kv_cache_iswa(
                                *this,
                                params.type_k,
                                params.type_v,
                                !cparams.flash_attn,
                                cparams.offload_kqv,
                                params.swa_full,
                                cparams.kv_unified,
                                cparams.n_ctx_seq,
                                cparams.n_seq_max,
                                cparams.n_ubatch,
                                1,
                                nullptr,
                                reuse);
                    } else {
                        GGML_ASSERT(!hparams.is_swa_any());

                        res = new llama_kv_cache(
                                *this,
                                params.type_k,
                                params.type_v,
                                !cparams.flash_attn,
                                cparams.offload_kqv,
                                cparams.kv_unified,
                                cparams.n_ctx_seq,
                                cparams.n_seq_max,
                                1,
                                hparams.n_swa,
                                hparams.swa_type,
                                nullptr,
                                nullptr);
                    }
                }
            }
    }

    return res;
}

ggml_cgraph * llama_model::build_graph(const llm_graph_params & params) const {
    std::unique_ptr<llm_graph_context> llm = build_arch_graph(params);

    // add on pooling layer
    llm->build_pooling(cls, cls_b, cls_out, cls_out_b, cls_norm);

    // add backend sampling layers (if any)
    llm->build_sampling();

    // if the gguf model was converted with --sentence-transformers-dense-modules
    // there will be two additional dense projection layers
    // dense linear projections are applied after pooling
    // TODO: move reranking logic here and generalize
    llm->build_dense_out(dense_2_out_layers, dense_2_out_layers_b, dense_3_out_layers);

    llm->res->set_outputs();

    return llm->res->get_gf();
}


//
// interface implementation
//

llama_model_params llama_model_default_params() {
    llama_model_params result = {
        /*.devices                     =*/ nullptr,
        /*.tensor_buft_overrides       =*/ nullptr,
        /*.n_gpu_layers                =*/ -1,
        /*.split_mode                  =*/ LLAMA_SPLIT_MODE_LAYER,
        /*.main_gpu                    =*/ 0,
        /*.tensor_split                =*/ nullptr,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.vocab_only                  =*/ false,
        /*.use_mmap                    =*/ true,
        /*.use_direct_io               =*/ false,
        /*.use_mlock                   =*/ false,
        /*.check_tensors               =*/ false,
        /*.use_extra_bufts             =*/ true,
        /*.no_host                     =*/ false,
        /*.no_alloc                    =*/ false,
    };

    return result;
}

const llama_vocab * llama_model_get_vocab(const llama_model * model) {
    return &model->vocab;
}

void llama_free_model(llama_model * model) {
    llama_model_free(model);
}

void llama_model_free(llama_model * model) {
    delete model;
}

int32_t llama_model_n_ctx_train(const llama_model * model) {
    return model->hparams.n_ctx_train;
}

int32_t llama_model_n_embd(const llama_model * model) {
    return model->hparams.n_embd;
}

int32_t llama_model_n_embd_inp(const llama_model * model) {
    return model->hparams.n_embd_inp();
}

int32_t llama_model_n_embd_out(const llama_model * model) {
    return model->hparams.n_embd_out();
}

int32_t llama_model_n_layer(const llama_model * model) {
    return model->hparams.n_layer;
}

int32_t llama_model_n_head(const llama_model * model) {
    return model->hparams.n_head();
}

int32_t llama_model_n_head_kv(const llama_model * model) {
    return model->hparams.n_head_kv();
}

int32_t llama_model_n_swa(const llama_model * model) {
    return model->hparams.n_swa;
}

uint32_t llama_model_n_cls_out(const struct llama_model * model) {
    return model->hparams.n_cls_out;
}

const char * llama_model_cls_label(const struct llama_model * model, uint32_t i) {
    if (i < model->classifier_labels.size()) {
        return model->classifier_labels[i].c_str();
    }

    return nullptr;
}

// deprecated
int32_t llama_n_ctx_train(const llama_model * model) {
    return llama_model_n_ctx_train(model);
}

// deprecated
int32_t llama_n_embd(const llama_model * model) {
    return llama_model_n_embd(model);
}

// deprecated
int32_t llama_n_layer(const llama_model * model) {
    return llama_model_n_layer(model);
}

// deprecated
int32_t llama_n_head(const llama_model * model) {
    return llama_model_n_head(model);
}

llama_rope_type llama_model_rope_type(const llama_model * model) {
    switch (model->arch) {
        // these models do not use RoPE
        case LLM_ARCH_CLIP:
        case LLM_ARCH_GPT2:
        case LLM_ARCH_GPTJ:
        case LLM_ARCH_MPT:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_BLOOM:
        case LLM_ARCH_MAMBA:
        case LLM_ARCH_MAMBA2:
        case LLM_ARCH_JAMBA:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_T5:
        case LLM_ARCH_T5ENCODER:
        case LLM_ARCH_JAIS:
        case LLM_ARCH_RWKV6:
        case LLM_ARCH_RWKV6QWEN2:
        case LLM_ARCH_RWKV7:
        case LLM_ARCH_ARWKV7:
        case LLM_ARCH_WAVTOKENIZER_DEC:
        case LLM_ARCH_NEMOTRON_H:
        case LLM_ARCH_NEMOTRON_H_MOE:
        case LLM_ARCH_KIMI_LINEAR:
            return LLAMA_ROPE_TYPE_NONE;

        // use what we call a normal RoPE, operating on pairs of consecutive head values
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_LLADA:
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_DECI:
        case LLM_ARCH_BAICHUAN:
        case LLM_ARCH_STARCODER:
        case LLM_ARCH_INTERNLM2:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_XVERSE:
        case LLM_ARCH_COMMAND_R:
        case LLM_ARCH_COHERE2:
        case LLM_ARCH_OLMO:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK2OCR:
        case LLM_ARCH_PLM:
        case LLM_ARCH_CHATGLM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_GRANITE_HYBRID:
        case LLM_ARCH_CHAMELEON:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_NEO_BERT:
        case LLM_ARCH_SMOLLM3:
        case LLM_ARCH_ARCEE:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_LLAMA_EMBED:
        case LLM_ARCH_MAINCODER:
        case LLM_ARCH_GLM_DSA:
            return LLAMA_ROPE_TYPE_NORM;

        // the pairs of head values are offset by n_rot/2
        case LLM_ARCH_FALCON:
        case LLM_ARCH_FALCON_H1:
        case LLM_ARCH_GROK:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_BERT:
        case LLM_ARCH_JINA_BERT_V3:
        case LLM_ARCH_MODERN_BERT:
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_NOMIC_BERT_MOE:
        case LLM_ARCH_EUROBERT:
        case LLM_ARCH_STABLELM:
        case LLM_ARCH_BITNET:
        case LLM_ARCH_QWEN:
        case LLM_ARCH_QWEN2:
        case LLM_ARCH_DREAM:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_RND1:
        case LLM_ARCH_OLMO2:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_PHI2:
        case LLM_ARCH_PHI3:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_PLAMO:
        case LLM_ARCH_PLAMO2:
        case LLM_ARCH_PLAMO3:
        case LLM_ARCH_GEMMA:
        case LLM_ARCH_GEMMA2:
        case LLM_ARCH_GEMMA3:
        case LLM_ARCH_GEMMA3N:
        case LLM_ARCH_GEMMA4:
        case LLM_ARCH_GEMMA_EMBEDDING:
        case LLM_ARCH_STARCODER2:
        case LLM_ARCH_OPENELM:
        case LLM_ARCH_GPTNEOX:
        case LLM_ARCH_CODESHELL:
        case LLM_ARCH_ORION:
        case LLM_ARCH_NEMOTRON:
        case LLM_ARCH_EXAONE:
        case LLM_ARCH_EXAONE4:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_MINICPM3:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_JAIS2:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_HUNYUAN_DENSE:
        case LLM_ARCH_LFM2:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_SEED_OSS:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_APERTUS:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_COGVLM:
        case LLM_ARCH_PANGU_EMBED:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_STEP35:
            return LLAMA_ROPE_TYPE_NEOX;

        case LLM_ARCH_QWEN2VL:
        case LLM_ARCH_PADDLEOCR:
            return LLAMA_ROPE_TYPE_MROPE;
        case LLM_ARCH_QWEN3VL:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35:
        case LLM_ARCH_QWEN35MOE:
            return LLAMA_ROPE_TYPE_IMROPE;

        case LLM_ARCH_GLM4:
            return model->hparams.use_mrope() ? LLAMA_ROPE_TYPE_MROPE : LLAMA_ROPE_TYPE_NORM;
        case LLM_ARCH_GLM4_MOE:
            return model->hparams.use_mrope() ? LLAMA_ROPE_TYPE_MROPE : LLAMA_ROPE_TYPE_NEOX;

        case LLM_ARCH_HUNYUAN_VL:
            return model->hparams.use_mrope() ? LLAMA_ROPE_TYPE_MROPE : LLAMA_ROPE_TYPE_NEOX;

        // all model arches should be listed explicitly here
        case LLM_ARCH_UNKNOWN:
            GGML_ABORT("unknown architecture");
    }

    return LLAMA_ROPE_TYPE_NONE;
}

float llama_model_rope_freq_scale_train(const llama_model * model) {
    return model->hparams.rope_freq_scale_train;
}

int32_t llama_model_meta_val_str(const llama_model * model, const char * key, char * buf, size_t buf_size) {
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_model_meta_count(const llama_model * model) {
    return (int)model->gguf_kv.size();
}

const char * llama_model_meta_key_str(llama_model_meta_key key) {
    switch (key) {
        case LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE:        return "general.sampling.sequence";
        case LLAMA_MODEL_META_KEY_SAMPLING_TOP_K:           return "general.sampling.top_k";
        case LLAMA_MODEL_META_KEY_SAMPLING_TOP_P:           return "general.sampling.top_p";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIN_P:           return "general.sampling.min_p";
        case LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY: return "general.sampling.xtc_probability";
        case LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD:   return "general.sampling.xtc_threshold";
        case LLAMA_MODEL_META_KEY_SAMPLING_TEMP:            return "general.sampling.temp";
        case LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N:  return "general.sampling.penalty_last_n";
        case LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT:  return "general.sampling.penalty_repeat";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT:        return "general.sampling.mirostat";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU:    return "general.sampling.mirostat_tau";
        case LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA:    return "general.sampling.mirostat_eta";
        default:                                            return nullptr;
    }
}

int32_t llama_model_meta_key_by_index(const llama_model * model, int i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->first.c_str());
}

int32_t llama_model_meta_val_str_by_index(const llama_model * model, int32_t i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_model_desc(const llama_model * model, char * buf, size_t buf_size) {
    return snprintf(buf, buf_size, "%s", model->desc().c_str());
}

uint64_t llama_model_size(const llama_model * model) {
    return model->size();
}

const char * llama_model_chat_template(const llama_model * model, const char * name) {
    const auto key = name ? LLM_KV(model->arch, name)(LLM_KV_TOKENIZER_CHAT_TEMPLATE)
        : LLM_KV(model->arch)(LLM_KV_TOKENIZER_CHAT_TEMPLATE);
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        // one-off fix for very popular models (so we are not flooded with issues)
        // do not extend this list unless absolutely necessary
        // Mistral-Small-2503 does not have built-in chat template
        llama_vocab_pre_type pre_type = model->vocab.get_pre_type();
        if (!name && pre_type == LLAMA_VOCAB_PRE_TYPE_TEKKEN && model->layers.size() == 40) {
            return "mistral-v7-tekken";
        }

        return nullptr;
    }

    return it->second.c_str();
}

uint64_t llama_model_n_params(const llama_model * model) {
    return model->n_elements();
}

bool llama_model_has_encoder(const llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_T5:        return true;
        case LLM_ARCH_T5ENCODER: return true;
        default:                 return false;
    }
}

bool llama_model_has_decoder(const llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_T5ENCODER: return false;
        default:                 return true;
    }
}

llama_token llama_model_decoder_start_token(const llama_model * model) {
    return model->hparams.dec_start_token_id;
}

bool llama_model_is_recurrent(const llama_model * model) {
    return llm_arch_is_recurrent(model->arch);
}

bool llama_model_is_hybrid(const llama_model * model) {
    return llm_arch_is_hybrid(model->arch);
}

bool llama_model_is_diffusion(const llama_model * model) {
    return llm_arch_is_diffusion(model->arch);
}

const std::vector<std::pair<std::string, ggml_tensor *>> & llama_internal_get_tensor_map(const llama_model * model) {
    return model->tensors_by_name;
}

int32_t llama_model_n_expert(const struct llama_model * model) {
    return model->hparams.n_expert;
}

int32_t llama_model_n_devices(const struct llama_model * model) {
    return (int32_t)model->devices.size();
}

ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i) {
    if (i < 0 || i >= (int)model->devices.size()) {
        return nullptr;
    }
    return model->devices[i].dev;
}

//
// llama_model_base
//

llama_model_base::llama_model_base(const struct llama_model_params & params) : llama_model(params), model(this), tn(model->arch),
    TENSOR_DUPLICATED     (llama_model_loader::TENSOR_DUPLICATED),
    TENSOR_NOT_REQUIRED   (llama_model_loader::TENSOR_NOT_REQUIRED),
    TENSOR_SKIP           (llama_model_loader::TENSOR_SKIP),
    TENSOR_SKIP_IF_VIRTUAL(llama_model_loader::TENSOR_SKIP_IF_VIRTUAL) {}

ggml_tensor * llama_model_base::create_tensor(const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) {
    GGML_ASSERT(ml != nullptr);
    return create_tensor(*ml, tn, ne, flags);
}

void llama_model_base::create_tensor_gate_up_exps(llama_layer & layer, int bid, int64_t n_embd_, int64_t n_ff_, int64_t n_expert_, int flags) {
    layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", bid), {n_embd_, n_ff_ * 2, n_expert_}, TENSOR_NOT_REQUIRED);
    if (layer.ffn_gate_up_exps == nullptr) {
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", bid), {n_embd_, n_ff_, n_expert_}, flags);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", bid), {n_embd_, n_ff_, n_expert_}, flags);
    }
}

void llama_model_base::create_tensor_qkv(llama_layer & layer, int bid,
        int64_t n_embd_, int64_t n_embd_q_, int64_t n_embd_k_, int64_t n_embd_v_,
        int flags) {
    const int64_t n_embd_qkv = n_embd_q_ + n_embd_k_ + n_embd_v_;
    layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", bid), {n_embd_, n_embd_qkv}, TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
    if (layer.wqkv) {
        layer.wqkv_b = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", bid), {n_embd_qkv}, TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
    } else {
        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", bid), {n_embd_, n_embd_q_}, flags);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", bid), {n_embd_, n_embd_k_}, flags);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", bid), {n_embd_, n_embd_v_}, flags);
        layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", bid), {n_embd_q_}, TENSOR_NOT_REQUIRED);
        layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias", bid), {n_embd_k_}, TENSOR_NOT_REQUIRED);
        layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias", bid), {n_embd_v_}, TENSOR_NOT_REQUIRED);
    }
}
