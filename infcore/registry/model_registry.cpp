// infcore — корпоративная лицензия.
#include "model_registry.h"

namespace infcore {

const char* modality_to_string(Modality m) {
    switch (m) {
        case Modality::Text:      return "text";
        case Modality::Embedding: return "embedding";
        case Modality::Vision:    return "vision";
        case Modality::Audio:     return "audio";
    }
    return "text";
}

void ModelRegistry::add(const ModelEntry& e) {
    std::lock_guard<std::mutex> lock(mu_);
    models_[e.logical_name] = e;
}

bool ModelRegistry::get(const std::string& logical_name, ModelEntry& out) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = models_.find(logical_name);
    if (it == models_.end()) return false;
    out = it->second;
    return true;
}

bool ModelRegistry::set_enabled(const std::string& logical_name, bool enabled) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = models_.find(logical_name);
    if (it == models_.end()) return false;
    it->second.enabled = enabled;
    return true;
}

std::vector<ModelEntry> ModelRegistry::list() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<ModelEntry> out;
    out.reserve(models_.size());
    for (const auto& kv : models_) out.push_back(kv.second);
    return out;
}

}  // namespace infcore
