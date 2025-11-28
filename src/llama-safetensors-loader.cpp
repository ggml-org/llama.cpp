#include "llama-safetensors-loader.h"

#include "llama.h"
#include "llama-impl.h"

#include <algorithm>
#include <fstream>
#include <regex>

// Map HuggingFace architecture names to llama.cpp architectures
llm_arch hf_arch_to_llm_arch(const std::string & hf_arch) {
    // Llama family
    if (hf_arch == "LlamaForCausalLM" ||
        hf_arch == "LLaMAForCausalLM") {
        return LLM_ARCH_LLAMA;
    }
    // Mistral (uses Llama architecture)
    if (hf_arch == "MistralForCausalLM" ||
        hf_arch == "MixtralForCausalLM") {
        return LLM_ARCH_LLAMA;
    }
    // Phi
    if (hf_arch == "PhiForCausalLM" ||
        hf_arch == "Phi3ForCausalLM") {
        return LLM_ARCH_PHI3;
    }
    // Qwen2
    if (hf_arch == "Qwen2ForCausalLM") {
        return LLM_ARCH_QWEN2;
    }
    // Gemma
    if (hf_arch == "GemmaForCausalLM" ||
        hf_arch == "Gemma2ForCausalLM") {
        return LLM_ARCH_GEMMA;
    }

    return LLM_ARCH_UNKNOWN;
}

// Llama/Mistral tensor name mapper
std::string llama_tensor_mapper::map_tensor_name(const std::string & hf_name) const {
    // HuggingFace Llama/Mistral tensor naming:
    // model.embed_tokens.weight -> token_embd.weight
    // model.layers.{N}.self_attn.q_proj.weight -> blk.{N}.attn_q.weight
    // model.layers.{N}.self_attn.k_proj.weight -> blk.{N}.attn_k.weight
    // model.layers.{N}.self_attn.v_proj.weight -> blk.{N}.attn_v.weight
    // model.layers.{N}.self_attn.o_proj.weight -> blk.{N}.attn_output.weight
    // model.layers.{N}.mlp.gate_proj.weight -> blk.{N}.ffn_gate.weight
    // model.layers.{N}.mlp.up_proj.weight -> blk.{N}.ffn_up.weight
    // model.layers.{N}.mlp.down_proj.weight -> blk.{N}.ffn_down.weight
    // model.layers.{N}.input_layernorm.weight -> blk.{N}.attn_norm.weight
    // model.layers.{N}.post_attention_layernorm.weight -> blk.{N}.ffn_norm.weight
    // model.norm.weight -> output_norm.weight
    // lm_head.weight -> output.weight

    if (hf_name == "model.embed_tokens.weight") {
        return "token_embd.weight";
    }

    if (hf_name == "lm_head.weight") {
        return "output.weight";
    }

    if (hf_name == "model.norm.weight") {
        return "output_norm.weight";
    }

    // Handle layer-specific tensors
    std::regex layer_regex(R"(model\.layers\.(\d+)\.(.+))");
    std::smatch match;
    if (std::regex_match(hf_name, match, layer_regex)) {
        std::string layer_idx = match[1].str();
        std::string rest = match[2].str();

        std::string mapped_name = "blk." + layer_idx + ".";

        if (rest == "self_attn.q_proj.weight") {
            mapped_name += "attn_q.weight";
        } else if (rest == "self_attn.k_proj.weight") {
            mapped_name += "attn_k.weight";
        } else if (rest == "self_attn.v_proj.weight") {
            mapped_name += "attn_v.weight";
        } else if (rest == "self_attn.o_proj.weight") {
            mapped_name += "attn_output.weight";
        } else if (rest == "mlp.gate_proj.weight") {
            mapped_name += "ffn_gate.weight";
        } else if (rest == "mlp.up_proj.weight") {
            mapped_name += "ffn_up.weight";
        } else if (rest == "mlp.down_proj.weight") {
            mapped_name += "ffn_down.weight";
        } else if (rest == "input_layernorm.weight") {
            mapped_name += "attn_norm.weight";
        } else if (rest == "post_attention_layernorm.weight") {
            mapped_name += "ffn_norm.weight";
        } else {
            // Unknown tensor
            return "";
        }

        return mapped_name;
    }

    // Unknown tensor - skip it
    return "";
}

std::vector<std::string> llama_tensor_mapper::get_required_tensors(int n_layers) const {
    std::vector<std::string> required;

    required.push_back("model.embed_tokens.weight");
    required.push_back("model.norm.weight");
    required.push_back("lm_head.weight");

    for (int i = 0; i < n_layers; i++) {
        std::string prefix = "model.layers." + std::to_string(i) + ".";
        required.push_back(prefix + "self_attn.q_proj.weight");
        required.push_back(prefix + "self_attn.k_proj.weight");
        required.push_back(prefix + "self_attn.v_proj.weight");
        required.push_back(prefix + "self_attn.o_proj.weight");
        required.push_back(prefix + "mlp.gate_proj.weight");
        required.push_back(prefix + "mlp.up_proj.weight");
        required.push_back(prefix + "mlp.down_proj.weight");
        required.push_back(prefix + "input_layernorm.weight");
        required.push_back(prefix + "post_attention_layernorm.weight");
    }

    return required;
}

// Phi tensor name mapper
std::string phi_tensor_mapper::map_tensor_name(const std::string & hf_name) const {
    // Phi-3 uses similar structure to Llama with some differences
    // TODO: Implement Phi-specific mappings
    // For now, use Llama mappings as they're similar
    llama_tensor_mapper llama_mapper;
    return llama_mapper.map_tensor_name(hf_name);
}

std::vector<std::string> phi_tensor_mapper::get_required_tensors(int n_layers) const {
    // TODO: Implement Phi-specific required tensors
    llama_tensor_mapper llama_mapper;
    return llama_mapper.get_required_tensors(n_layers);
}

// Qwen2 tensor name mapper
std::string qwen2_tensor_mapper::map_tensor_name(const std::string & hf_name) const {
    // Qwen2 uses similar structure to Llama
    // TODO: Implement Qwen2-specific mappings
    llama_tensor_mapper llama_mapper;
    return llama_mapper.map_tensor_name(hf_name);
}

std::vector<std::string> qwen2_tensor_mapper::get_required_tensors(int n_layers) const {
    // TODO: Implement Qwen2-specific required tensors
    llama_tensor_mapper llama_mapper;
    return llama_mapper.get_required_tensors(n_layers);
}

// Gemma tensor name mapper
std::string gemma_tensor_mapper::map_tensor_name(const std::string & hf_name) const {
    // Gemma uses similar structure to Llama with some differences
    // TODO: Implement Gemma-specific mappings
    llama_tensor_mapper llama_mapper;
    return llama_mapper.map_tensor_name(hf_name);
}

std::vector<std::string> gemma_tensor_mapper::get_required_tensors(int n_layers) const {
    // TODO: Implement Gemma-specific required tensors
    llama_tensor_mapper llama_mapper;
    return llama_mapper.get_required_tensors(n_layers);
}

// Factory function
std::unique_ptr<safetensors_tensor_mapper> create_tensor_mapper(const std::string & hf_arch) {
    llm_arch arch = hf_arch_to_llm_arch(hf_arch);

    switch (arch) {
        case LLM_ARCH_LLAMA:
            return std::make_unique<llama_tensor_mapper>();
        case LLM_ARCH_PHI3:
            return std::make_unique<phi_tensor_mapper>();
        case LLM_ARCH_QWEN2:
            return std::make_unique<qwen2_tensor_mapper>();
        case LLM_ARCH_GEMMA:
            return std::make_unique<gemma_tensor_mapper>();
        default:
            return nullptr;
    }
}

// Main loader implementation
bool safetensors_model_loader::load_config(const std::string & model_dir) {
    std::string config_path = model_dir + "/config.json";
    if (!config.load_from_file(config_path)) {
        error_msg = "Failed to load config.json: " + config.get_error();
        return false;
    }
    return true;
}

bool safetensors_model_loader::load_safetensors_files(const std::string & model_dir) {
    st_loader = std::make_unique<safetensors_loader>();

    // Try loading single file first
    std::string single_file = model_dir + "/model.safetensors";
    std::ifstream test_single(single_file);
    if (test_single.good()) {
        test_single.close();
        if (st_loader->load_single(single_file)) {
            return true;
        }
    }

    // Try loading sharded model
    std::string index_file = model_dir + "/model.safetensors.index.json";
    std::ifstream test_index(index_file);
    if (test_index.good()) {
        test_index.close();
        if (st_loader->load_sharded(index_file, model_dir)) {
            return true;
        }
    }

    error_msg = "No safetensors files found in directory: " + model_dir;
    return false;
}

bool safetensors_model_loader::create_mapper() {
    std::string arch = config.get_architecture();
    if (arch.empty()) {
        error_msg = "Failed to detect architecture from config.json";
        return false;
    }

    mapper = create_tensor_mapper(arch);
    if (!mapper) {
        error_msg = "Unsupported architecture: " + arch;
        return false;
    }

    return true;
}

llama_model * safetensors_model_loader::load(
    const std::string & model_dir,
    const llama_model_params & params
) {
    (void)params; // Unused for now - reserved for future use

    // Load config.json
    if (!load_config(model_dir)) {
        return nullptr;
    }

    // Load safetensors files
    if (!load_safetensors_files(model_dir)) {
        return nullptr;
    }

    // Create tensor mapper
    if (!create_mapper()) {
        return nullptr;
    }

    // TODO: Actually load the model
    // This requires:
    // 1. Create llama_model structure
    // 2. Allocate memory for tensors
    // 3. Load tensor data from safetensors
    // 4. Map tensor names and populate model structure
    // 5. Initialize model parameters

    error_msg = "Safetensors loading not yet fully implemented - this is a work in progress";
    return nullptr;
}
