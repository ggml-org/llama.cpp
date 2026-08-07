#include "moe-prune.h"

#include "ggml.h"
#include "gguf.h"
extern "C" {
#include "../examples/gguf-hash/deps/sha256/sha256.h"
}

#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>

using json = nlohmann::ordered_json;

namespace {

struct gguf_deleter {
    void operator()(gguf_context * ctx) const { gguf_free(ctx); }
};

using gguf_ptr = std::unique_ptr<gguf_context, gguf_deleter>;

std::string digest_hex(const unsigned char digest[SHA256_DIGEST_SIZE]) {
    static const char hex[] = "0123456789abcdef";
    std::string result(SHA256_DIGEST_SIZE * 2, '0');
    for (size_t i = 0; i < SHA256_DIGEST_SIZE; ++i) {
        result[2 * i] = hex[digest[i] >> 4];
        result[2 * i + 1] = hex[digest[i] & 0x0f];
    }
    return "sha256:" + result;
}

void hash_bytes(sha256_t & hash, const void * data, size_t size) {
    sha256_update(&hash, static_cast<const unsigned char *>(data), size);
}

bool is_expert_identity_tensor(const std::string & name) {
    if (name.find(".ffn_gate_inp.weight") != std::string::npos || name.find(".ffn_gate_inp.scale") != std::string::npos) {
        return true;
    }
    return name.find(".ffn_gate_up_exps.") != std::string::npos ||
           name.find(".ffn_gate_exps.") != std::string::npos ||
           name.find(".ffn_up_exps.") != std::string::npos ||
           name.find(".ffn_down_exps.") != std::string::npos;
}

int32_t metadata_i32(const gguf_context * ctx, const std::string & key) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) {
        throw std::runtime_error("missing GGUF metadata: " + key);
    }
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT32: return (int32_t) gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(ctx, id);
        case GGUF_TYPE_UINT64: return (int32_t) gguf_get_val_u64(ctx, id);
        case GGUF_TYPE_INT64:  return (int32_t) gguf_get_val_i64(ctx, id);
        default: throw std::runtime_error("GGUF metadata is not an integer: " + key);
    }
}

void write_json_atomic(const json & value, const std::string & path) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("failed to open output: " + tmp);
        }
        out << value.dump(2) << '\n';
        if (!out) {
            throw std::runtime_error("failed to write output: " + tmp);
        }
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        throw std::runtime_error("failed to replace output: " + path);
    }
}

}

double common_moe_prune_expert_stats::mean_probability() const {
    return selection_count == 0 ? 0.0 : probability_sum / selection_count;
}

double common_moe_prune_expert_stats::mean_output_norm() const {
    return selection_count == 0 ? 0.0 : output_norm_sum / selection_count;
}

double common_moe_prune_expert_stats::importance() const {
    return selection_count == 0 ? 0.0 : weighted_output_sum / selection_count;
}

std::string common_moe_prune_sha256_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open file for hashing: " + path);
    }
    sha256_t hash;
    sha256_init(&hash);
    std::vector<char> buffer(4 * 1024 * 1024);
    while (in) {
        in.read(buffer.data(), buffer.size());
        const std::streamsize n = in.gcount();
        if (n > 0) {
            hash_bytes(hash, buffer.data(), n);
        }
    }
    if (!in.eof()) {
        throw std::runtime_error("failed while hashing file: " + path);
    }
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&hash, digest);
    return digest_hex(digest);
}

common_moe_prune_model_info common_moe_prune_inspect_model(const std::string & path) {
    gguf_ptr ctx(gguf_init_from_file(path.c_str(), { true, nullptr }));
    if (!ctx) {
        throw std::runtime_error("failed to read GGUF model: " + path);
    }

    const int64_t arch_id = gguf_find_key(ctx.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(ctx.get(), arch_id) != GGUF_TYPE_STRING) {
        throw std::runtime_error("missing GGUF metadata: general.architecture");
    }

    common_moe_prune_model_info result;
    result.architecture = gguf_get_val_str(ctx.get(), arch_id);
    if (result.architecture != "gemma4") {
        throw std::runtime_error("unsupported architecture: MoE pruning supports Gemma 4 26B A4B only");
    }
    result.layer_count = metadata_i32(ctx.get(), "gemma4.block_count");
    if (result.layer_count != 30) {
        throw std::runtime_error("unsupported Gemma 4 variant: expected 30 layers for 26B A4B");
    }
    result.expert_count = metadata_i32(ctx.get(), "gemma4.expert_count");
    result.experts_used = metadata_i32(ctx.get(), "gemma4.expert_used_count");
    if (result.expert_count <= 0 || result.experts_used <= 0 || result.experts_used > result.expert_count) {
        throw std::runtime_error("invalid Gemma 4 expert metadata");
    }

    struct expert_tensor_span {
        std::string name;
        std::array<int64_t, GGML_MAX_DIMS> ne;
        int32_t type;
        uint64_t offset;
        size_t size;
    };
    std::vector<expert_tensor_span> expert_tensors;
    sha256_t expert_hash;
    sha256_init(&expert_hash);
    const size_t data_offset = gguf_get_data_offset(ctx.get());
    std::set<int32_t> moe_layers;
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx.get()); ++i) {
        const std::string name = gguf_get_tensor_name(ctx.get(), i);
        if (!is_expert_identity_tensor(name)) {
            continue;
        }
        int32_t layer = -1;
        if (sscanf(name.c_str(), "blk.%d.", &layer) == 1) {
            moe_layers.insert(layer);
        }
        const int64_t * ne = gguf_get_tensor_ne(ctx.get(), i);
        const int32_t type = (int32_t) gguf_get_tensor_type(ctx.get(), i);
        const size_t size = gguf_get_tensor_size(ctx.get(), i);
        expert_tensor_span span { name, {}, type, data_offset + gguf_get_tensor_offset(ctx.get(), i), size };
        std::copy(ne, ne + GGML_MAX_DIMS, span.ne.begin());
        expert_tensors.push_back(std::move(span));
        result.expert_bytes += size;
    }
    if (moe_layers.empty()) {
        throw std::runtime_error("Gemma 4 model has no routed expert tensors");
    }
    result.moe_layers.assign(moe_layers.begin(), moe_layers.end());

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open GGUF tensor data: " + path);
    }
    sha256_t model_hash;
    sha256_init(&model_hash);
    std::vector<unsigned char> buffer(4 * 1024 * 1024);
    uint64_t cursor = 0;
    auto read_range = [&](uint64_t size, bool is_expert) {
        while (size > 0) {
            const size_t chunk = (size_t) std::min<uint64_t>(size, buffer.size());
            file.read(reinterpret_cast<char *>(buffer.data()), chunk);
            if ((size_t) file.gcount() != chunk) {
                throw std::runtime_error("failed while hashing model: " + path);
            }
            hash_bytes(model_hash, buffer.data(), chunk);
            if (is_expert) hash_bytes(expert_hash, buffer.data(), chunk);
            cursor += chunk;
            size -= chunk;
        }
    };
    for (const expert_tensor_span & tensor : expert_tensors) {
        if (tensor.offset < cursor) {
            throw std::runtime_error("GGUF expert tensors are not stored in tensor order");
        }
        read_range(tensor.offset - cursor, false);
        hash_bytes(expert_hash, tensor.name.data(), tensor.name.size());
        hash_bytes(expert_hash, tensor.ne.data(), sizeof(int64_t) * tensor.ne.size());
        hash_bytes(expert_hash, &tensor.type, sizeof(tensor.type));
        read_range(tensor.size, true);
    }
    while (file) {
        file.read(reinterpret_cast<char *>(buffer.data()), buffer.size());
        const std::streamsize n = file.gcount();
        if (n > 0) hash_bytes(model_hash, buffer.data(), n);
    }
    if (!file.eof()) {
        throw std::runtime_error("failed while hashing model: " + path);
    }

    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&expert_hash, digest);
    result.expert_tensor_hash = digest_hex(digest);
    sha256_final(&model_hash, digest);
    result.model_hash = digest_hex(digest);
    return result;
}

common_moe_prune_profile common_moe_prune_profile_load(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open pruning profile: " + path);
    }
    json root;
    try {
        in >> root;
    } catch (const std::exception & e) {
        throw std::runtime_error("invalid pruning profile JSON: " + std::string(e.what()));
    }
    if (!root.is_object() || root.value("format", "") != "aikar-moe-prune") {
        throw std::runtime_error("invalid pruning profile format");
    }
    common_moe_prune_profile profile;
    profile.version = root.at("version").get<int32_t>();
    if (profile.version != 1) {
        throw std::runtime_error("unsupported pruning profile version: " + std::to_string(profile.version));
    }
    profile.mode = root.at("mode").get<std::string>();
    if (profile.mode != "soft") {
        throw std::runtime_error("unsupported pruning profile mode: " + profile.mode);
    }
    const json & model = root.at("model");
    profile.architecture = model.at("architecture").get<std::string>();
    profile.model_hash = model.at("model_hash").get<std::string>();
    profile.expert_tensor_hash = model.at("expert_tensor_hash").get<std::string>();
    profile.expert_count = model.at("expert_count").get<int32_t>();
    profile.experts_used = model.at("experts_used").get<int32_t>();
    const json & calibration = root.at("calibration");
    profile.dataset_hash = calibration.at("dataset_hash").get<std::string>();
    profile.ppl_mask = calibration.at("ppl_mask").get<std::string>();
    profile.metric = calibration.at("metric").get<std::string>();
    profile.evaluated_tokens = calibration.at("evaluated_tokens").get<int64_t>();
    const json & pruning = root.at("pruning");
    profile.requested_ratio = pruning.at("requested_ratio").get<double>();
    profile.actual_ratio = pruning.at("actual_ratio").get<double>();
    for (auto it = pruning.at("layers").begin(); it != pruning.at("layers").end(); ++it) {
        size_t used = 0;
        int32_t layer = std::stoi(it.key(), &used);
        if (used != it.key().size()) {
            throw std::runtime_error("invalid layer key in pruning profile: " + it.key());
        }
        profile.layers[layer].disabled_experts = it.value().at("disabled_experts").get<std::vector<int32_t>>();
    }
    return profile;
}

void common_moe_prune_profile_write(const common_moe_prune_profile & profile, const std::string & path) {
    json layers = json::object();
    for (const auto & item : profile.layers) {
        layers[std::to_string(item.first)] = { { "disabled_experts", item.second.disabled_experts } };
    }
    json root = {
        { "format", "aikar-moe-prune" },
        { "version", profile.version },
        { "mode", profile.mode },
        { "model", {
            { "architecture", profile.architecture },
            { "model_hash", profile.model_hash },
            { "expert_tensor_hash", profile.expert_tensor_hash },
            { "expert_count", profile.expert_count },
            { "experts_used", profile.experts_used },
        } },
        { "calibration", {
            { "dataset_hash", profile.dataset_hash },
            { "ppl_mask", profile.ppl_mask },
            { "metric", profile.metric },
            { "evaluated_tokens", profile.evaluated_tokens },
        } },
        { "pruning", {
            { "requested_ratio", profile.requested_ratio },
            { "actual_ratio", profile.actual_ratio },
            { "layers", layers },
        } },
    };
    write_json_atomic(root, path);
}

void common_moe_prune_profile_validate(const common_moe_prune_profile & profile, const common_moe_prune_model_info & model) {
    if (profile.architecture != model.architecture) throw std::runtime_error("pruning profile architecture mismatch");
    if (profile.model_hash != model.model_hash) throw std::runtime_error("pruning profile model hash mismatch");
    if (profile.expert_tensor_hash != model.expert_tensor_hash) throw std::runtime_error("pruning profile expert tensor hash mismatch");
    if (profile.expert_count != model.expert_count) throw std::runtime_error("pruning profile expert count mismatch");
    if (profile.experts_used != model.experts_used) throw std::runtime_error("pruning profile router Top-K mismatch");
    if (profile.layers.size() != model.moe_layers.size()) throw std::runtime_error("pruning profile MoE layer count mismatch");
    size_t expected_disabled = SIZE_MAX;
    for (int32_t layer : model.moe_layers) {
        auto it = profile.layers.find(layer);
        if (it == profile.layers.end()) throw std::runtime_error("pruning profile is missing MoE layer " + std::to_string(layer));
        const auto & disabled = it->second.disabled_experts;
        if (disabled.empty()) throw std::runtime_error("pruning profile disables no experts in layer " + std::to_string(layer));
        if (expected_disabled == SIZE_MAX) expected_disabled = disabled.size();
        if (disabled.size() != expected_disabled) throw std::runtime_error("heterogeneous surviving expert counts are unsupported");
        std::set<int32_t> unique;
        for (int32_t expert : disabled) {
            if (expert < 0 || expert >= model.expert_count) throw std::runtime_error("invalid expert ID in layer " + std::to_string(layer));
            if (!unique.insert(expert).second) throw std::runtime_error("duplicate expert ID in layer " + std::to_string(layer));
        }
        if (model.expert_count - (int32_t) disabled.size() < model.experts_used) throw std::runtime_error("pruning profile violates router Top-K safety");
    }
}

void common_moe_prune_profile_apply(llama_model * model, const common_moe_prune_profile & profile) {
    std::vector<std::vector<int32_t>> storage;
    std::vector<llama_moe_prune_layer> layers;
    storage.reserve(profile.layers.size());
    layers.reserve(profile.layers.size());
    for (const auto & item : profile.layers) {
        storage.push_back(item.second.disabled_experts);
        layers.push_back({ item.first, storage.back().data(), storage.back().size() });
    }
    char error[512];
    if (!llama_model_set_moe_prune(model, layers.data(), layers.size(), error, sizeof(error))) {
        throw std::runtime_error(error);
    }
}

std::vector<common_moe_prune_profile> common_moe_prune_make_profiles(
        const common_moe_prune_model_info & model,
        const common_moe_prune_stats & stats,
        const std::vector<double> & ratios,
        double max_layer_ratio,
        const std::string & dataset_hash,
        const std::string & ppl_mask,
        const std::string & metric,
        int64_t evaluated_tokens) {
    if (ratios.empty()) throw std::runtime_error("no pruning ratios were requested");
    if (max_layer_ratio < 0.0 || max_layer_ratio >= 1.0) throw std::runtime_error("max layer ratio must be in [0, 1)");
    std::vector<double> sorted_ratios = ratios;
    std::sort(sorted_ratios.begin(), sorted_ratios.end());
    if (std::adjacent_find(sorted_ratios.begin(), sorted_ratios.end()) != sorted_ratios.end()) throw std::runtime_error("duplicate pruning ratio");
    for (double ratio : sorted_ratios) {
        if (ratio <= 0.0 || ratio > max_layer_ratio) throw std::runtime_error("pruning ratio must be positive and no greater than max layer ratio");
    }

    std::map<int32_t, std::vector<int32_t>> ranking;
    for (int32_t layer : model.moe_layers) {
        auto found = stats.find(layer);
        if (found == stats.end() || found->second.size() != (size_t) model.expert_count) throw std::runtime_error("missing expert statistics for layer " + std::to_string(layer));
        auto & ids = ranking[layer];
        ids.resize(model.expert_count);
        for (int32_t i = 0; i < model.expert_count; ++i) ids[i] = i;
        std::stable_sort(ids.begin(), ids.end(), [&](int32_t a, int32_t b) {
            const double ia = found->second[a].importance();
            const double ib = found->second[b].importance();
            return ia == ib ? a < b : ia < ib;
        });
    }

    std::vector<common_moe_prune_profile> result;
    for (double ratio : sorted_ratios) {
        int32_t count = (int32_t) std::floor(model.expert_count * ratio + 1e-12);
        count = std::min(count, model.expert_count - model.experts_used);
        if (count == 0) throw std::runtime_error("pruning ratio is too small to remove an expert");
        common_moe_prune_profile profile;
        profile.architecture = model.architecture;
        profile.model_hash = model.model_hash;
        profile.expert_tensor_hash = model.expert_tensor_hash;
        profile.expert_count = model.expert_count;
        profile.experts_used = model.experts_used;
        profile.dataset_hash = dataset_hash;
        profile.ppl_mask = ppl_mask;
        profile.metric = metric;
        profile.evaluated_tokens = evaluated_tokens;
        profile.requested_ratio = ratio;
        profile.actual_ratio = model.expert_count == 0 ? 0.0 : (double) count / model.expert_count;
        for (int32_t layer : model.moe_layers) {
            auto disabled = std::vector<int32_t>(ranking[layer].begin(), ranking[layer].begin() + count);
            std::sort(disabled.begin(), disabled.end());
            profile.layers[layer].disabled_experts = std::move(disabled);
        }
        result.push_back(std::move(profile));
    }
    return result;
}
