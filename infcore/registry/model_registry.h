// infcore — корпоративная лицензия.
// Реестр локальных моделей: logical_name -> {путь GGUF, модальность, провайдер, параметры}.
// Идея заимствована из llm_gateway (Go): метаданные моделей + /admin/models.
// Источник истины — config (offline, без скачивания весов).
#pragma once

#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace infcore {

enum class Modality { Text, Embedding, Vision, Audio };

struct ModelEntry {
    std::string logical_name;   // напр. "qwen3-moe-a3b"
    std::string gguf_path;      // локальный путь к весам (для справки/будущего in-process)
    std::string arch;           // напр. "qwen3moe" (из метаданных GGUF)
    std::string backend_url;    // базовый URL бэкенда llama-server, напр. http://127.0.0.1:8081
    std::string upstream_model; // имя модели на бэкенде (если отличается); по умолч. = logical_name
    Modality    modality = Modality::Text;
    bool        enabled  = true;
    int32_t     n_ctx        = 8192;
    int32_t     n_gpu_layers = 0;
};

const char* modality_to_string(Modality m);

// Потокобезопасный реестр метаданных моделей. Загрузка/выгрузка контекстов —
// на уровне gateway; здесь только каталог и валидация.
class ModelRegistry {
public:
    void add(const ModelEntry& e);
    bool get(const std::string& logical_name, ModelEntry& out) const;
    std::vector<ModelEntry> list() const;

private:
    mutable std::mutex      mu_;
    std::map<std::string, ModelEntry> models_;
};

}  // namespace infcore
