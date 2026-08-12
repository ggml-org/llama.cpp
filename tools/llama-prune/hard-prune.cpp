#include "hard-prune.h"

#include "ggml.h"
#include "gguf.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

struct gguf_deleter {
    void operator()(gguf_context * ctx) const { gguf_free(ctx); }
};

struct ggml_deleter {
    void operator()(ggml_context * ctx) const { ggml_free(ctx); }
};

using gguf_ptr = std::unique_ptr<gguf_context, gguf_deleter>;
using ggml_ptr = std::unique_ptr<ggml_context, ggml_deleter>;

bool ends_with(const std::string & value, const std::string & suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int32_t tensor_layer(const std::string & name) {
    int32_t layer = -1;
    return sscanf(name.c_str(), "blk.%d.", &layer) == 1 ? layer : -1;
}

bool is_routed_expert_weight(const std::string & name) {
    return ends_with(name, ".ffn_gate_up_exps.weight") || ends_with(name, ".ffn_gate_exps.weight") ||
           ends_with(name, ".ffn_up_exps.weight") || ends_with(name, ".ffn_down_exps.weight");
}

bool is_routed_expert_scale(const std::string & name) {
    return ends_with(name, ".ffn_gate_exps.scale") || ends_with(name, ".ffn_up_exps.scale") ||
           ends_with(name, ".ffn_down_exps.scale");
}

std::vector<int32_t> surviving_experts(int32_t n_expert, const std::vector<int32_t> & disabled) {
    std::vector<bool> removed(n_expert, false);
    for (int32_t expert : disabled) removed[expert] = true;
    std::vector<int32_t> result;
    for (int32_t expert = 0; expert < n_expert; ++expert) {
        if (!removed[expert]) result.push_back(expert);
    }
    return result;
}

void copy_range(std::ifstream & input, std::ofstream & output, uint64_t offset, uint64_t size) {
    input.clear();
    input.seekg((std::streamoff) offset);
    if (!input) throw std::runtime_error("failed to seek source GGUF");
    std::vector<char> buffer(4 * 1024 * 1024);
    while (size > 0) {
        const size_t chunk = (size_t) std::min<uint64_t>(size, buffer.size());
        input.read(buffer.data(), chunk);
        if ((size_t) input.gcount() != chunk) throw std::runtime_error("failed to read source GGUF tensor data");
        output.write(buffer.data(), chunk);
        if (!output) throw std::runtime_error("failed to write pruned GGUF tensor data");
        size -= chunk;
    }
}

uint64_t file_size(const std::string & path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("failed to stat file: " + path);
    return (uint64_t) in.tellg();
}

void write_report(const llama_prune_hard_prune_report & report, const std::string & output_path) {
    json mappings = json::array();
    for (const auto & layer : report.original_to_new) {
        json mapping = json::object();
        for (const auto & item : layer.second) mapping[std::to_string(item.first)] = item.second;
        mappings.push_back({ { "layer", layer.first }, { "original_to_new", mapping } });
    }
    json root = {
        { "format", "llama-moe-hard-prune-report" },
        { "version", 1 },
        { "source_bytes", report.source_bytes },
        { "output_bytes", report.output_bytes },
        { "expert_bytes_removed", report.expert_bytes_removed },
        { "layers", mappings },
    };
    const std::string path = output_path + ".report.json";
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("failed to write hard-pruning report");
        out << root.dump(2) << '\n';
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        throw std::runtime_error("failed to replace hard-pruning report");
    }
}

}

llama_prune_hard_prune_report llama_prune_hard_prune_gemma4_q4_0(
        const std::string & model_path,
        const common_moe_prune_profile & profile,
        const common_moe_prune_model_info & model_info,
        const std::string & output_path) {
    if (model_path == output_path) throw std::runtime_error("hard pruning never modifies the source GGUF in place");
    common_moe_prune_profile_validate(profile, model_info);
    if (model_info.architecture != "gemma4" || model_info.layer_count != 30) {
        throw std::runtime_error("unsupported architecture: hard pruning supports Gemma 4 26B A4B only");
    }

    ggml_context * source_tensor_ctx_raw = nullptr;
    gguf_ptr source(gguf_init_from_file(model_path.c_str(), { true, &source_tensor_ctx_raw }));
    if (!source) throw std::runtime_error("failed to read source GGUF");
    ggml_ptr source_tensor_ctx(source_tensor_ctx_raw);
    gguf_ptr output(gguf_init_empty());
    gguf_set_kv(output.get(), source.get());
    gguf_set_val_u32(output.get(), GGUF_KEY_GENERAL_ALIGNMENT, (uint32_t) gguf_get_alignment(output.get()));
    gguf_set_val_u32(output.get(), "gemma4.expert_count", model_info.expert_count - (uint32_t) profile.layers.begin()->second.disabled_experts.size());

    const size_t tensor_count = (size_t) gguf_get_n_tensors(source.get());
    const size_t metadata_memory = std::max<size_t>(16 * 1024 * 1024, tensor_count * sizeof(ggml_tensor) * 4);
    ggml_ptr tensor_ctx(ggml_init({ metadata_memory, nullptr, true }));
    if (!tensor_ctx) throw std::runtime_error("failed to allocate GGUF tensor metadata context");

    struct tensor_plan {
        enum kind { COPY, ROUTER_ROWS, EXPERT_SLICES, EXPERT_SCALE } action = COPY;
        int64_t source_id = -1;
        int32_t layer = -1;
        std::vector<int32_t> survivors;
        size_t old_size = 0;
        size_t new_size = 0;
    };
    std::vector<tensor_plan> plans;
    plans.reserve(tensor_count);
    bool saw_fused_or_separate_experts = false;
    llama_prune_hard_prune_report report;

    for (int64_t i = 0; i < (int64_t) tensor_count; ++i) {
        const std::string name = gguf_get_tensor_name(source.get(), i);
        const ggml_tensor * old_tensor = ggml_get_tensor(source_tensor_ctx.get(), name.c_str());
        if (old_tensor == nullptr) throw std::runtime_error(name + ": missing source tensor metadata");
        const int64_t * old_ne = gguf_get_tensor_ne(source.get(), i);
        int64_t new_ne[GGML_MAX_DIMS] = { old_ne[0], old_ne[1], old_ne[2], old_ne[3] };
        const ggml_type type = gguf_get_tensor_type(source.get(), i);
        tensor_plan plan;
        plan.source_id = i;
        plan.old_size = gguf_get_tensor_size(source.get(), i);
        plan.layer = tensor_layer(name);
        auto profile_layer = profile.layers.find(plan.layer);
        const bool pruned_layer = profile_layer != profile.layers.end();

        if (pruned_layer && ends_with(name, ".ffn_gate_inp.weight")) {
            if (old_ne[1] != model_info.expert_count || old_ne[2] != 1) throw std::runtime_error(name + ": unexpected Gemma 4 router layout");
            if (old_ne[0] % ggml_blck_size(type) != 0) throw std::runtime_error(name + ": router rows are not block aligned");
            plan.action = tensor_plan::ROUTER_ROWS;
            plan.survivors = surviving_experts(model_info.expert_count, profile_layer->second.disabled_experts);
            new_ne[1] = plan.survivors.size();
        } else if (pruned_layer && is_routed_expert_weight(name)) {
            saw_fused_or_separate_experts = true;
            if (type != GGML_TYPE_Q4_0) throw std::runtime_error(name + ": hard pruning requires Q4_0 routed expert weights");
            if (old_ne[2] != model_info.expert_count || old_ne[3] != 1) throw std::runtime_error(name + ": expert axis is not dimension 2");
            if (old_ne[0] % 32 != 0) throw std::runtime_error(name + ": Q4_0 rows are not aligned to 32 values");
            const size_t slice_size = ggml_row_size(type, old_ne[0]) * old_ne[1];
            if (slice_size * old_ne[2] != plan.old_size) throw std::runtime_error(name + ": experts are interleaved or packed and cannot be copied safely");
            plan.action = tensor_plan::EXPERT_SLICES;
            plan.survivors = surviving_experts(model_info.expert_count, profile_layer->second.disabled_experts);
            new_ne[2] = plan.survivors.size();
        } else if (pruned_layer && is_routed_expert_scale(name)) {
            if (old_ne[0] != model_info.expert_count || old_ne[1] != 1 || ggml_is_quantized(type)) throw std::runtime_error(name + ": unexpected per-expert scale layout");
            plan.action = tensor_plan::EXPERT_SCALE;
            plan.survivors = surviving_experts(model_info.expert_count, profile_layer->second.disabled_experts);
            new_ne[0] = plan.survivors.size();
        } else if (pruned_layer && name.find("_exps.") != std::string::npos) {
            throw std::runtime_error(name + ": unsupported Gemma 4 expert tensor");
        }

        const int n_dims = ggml_n_dims(old_tensor);
        ggml_tensor * tensor = ggml_new_tensor(tensor_ctx.get(), type, n_dims, new_ne);
        ggml_set_name(tensor, name.c_str());
        gguf_add_tensor(output.get(), tensor);
        plan.new_size = ggml_nbytes(tensor);
        plans.push_back(std::move(plan));
    }
    if (!saw_fused_or_separate_experts) throw std::runtime_error("no Gemma 4 routed expert tensors were found");

    for (const auto & layer : profile.layers) {
        int32_t next = 0;
        const std::vector<int32_t> survivors = surviving_experts(model_info.expert_count, layer.second.disabled_experts);
        for (int32_t original : survivors) {
            report.original_to_new[layer.first][original] = next++;
        }
    }

    const std::string tmp_path = output_path + ".tmp";
    if (!gguf_write_to_file(output.get(), tmp_path.c_str(), true)) {
        throw std::runtime_error("failed to write pruned GGUF metadata");
    }
    std::ifstream input(model_path, std::ios::binary);
    std::ofstream out(tmp_path, std::ios::binary | std::ios::app);
    if (!input || !out) {
        std::remove(tmp_path.c_str());
        throw std::runtime_error("failed to open GGUF tensor streams");
    }
    const uint64_t source_data = gguf_get_data_offset(source.get());
    const size_t alignment = gguf_get_alignment(output.get());
    uint64_t output_cursor = 0;
    for (size_t i = 0; i < plans.size(); ++i) {
        const tensor_plan & plan = plans[i];
        const uint64_t target_offset = gguf_get_tensor_offset(output.get(), i);
        if (target_offset < output_cursor) throw std::runtime_error("invalid output GGUF tensor offsets");
        std::vector<char> padding((size_t) (target_offset - output_cursor), 0);
        out.write(padding.data(), padding.size());
        const uint64_t source_offset = source_data + gguf_get_tensor_offset(source.get(), plan.source_id);
        const int64_t * ne = gguf_get_tensor_ne(source.get(), plan.source_id);
        const ggml_type type = gguf_get_tensor_type(source.get(), plan.source_id);
        if (plan.action == tensor_plan::COPY) {
            copy_range(input, out, source_offset, plan.old_size);
        } else if (plan.action == tensor_plan::ROUTER_ROWS) {
            const size_t row = ggml_row_size(type, ne[0]);
            for (int32_t expert : plan.survivors) copy_range(input, out, source_offset + expert * row, row);
        } else if (plan.action == tensor_plan::EXPERT_SLICES) {
            const size_t slice = ggml_row_size(type, ne[0]) * ne[1];
            for (int32_t expert : plan.survivors) copy_range(input, out, source_offset + expert * slice, slice);
        } else {
            const size_t element = ggml_type_size(type);
            for (int32_t expert : plan.survivors) copy_range(input, out, source_offset + expert * element, element);
        }
        output_cursor = target_offset + plan.new_size;
        if (plan.old_size > plan.new_size) report.expert_bytes_removed += plan.old_size - plan.new_size;
    }
    const size_t end_padding = (alignment - output_cursor % alignment) % alignment;
    std::vector<char> padding(end_padding, 0);
    out.write(padding.data(), padding.size());
    out.close();
    if (!out) {
        std::remove(tmp_path.c_str());
        throw std::runtime_error("failed to finish pruned GGUF");
    }
    input.close();

    report.source_bytes = file_size(model_path);
    report.output_bytes = file_size(tmp_path);
    if (std::rename(tmp_path.c_str(), output_path.c_str()) != 0) {
        std::remove(tmp_path.c_str());
        throw std::runtime_error("failed to atomically replace hard-pruned GGUF output");
    }
    write_report(report, output_path);
    return report;
}
