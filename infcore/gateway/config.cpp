// infcore gateway — корпоративная лицензия.
#include "config.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace infcore {

static Modality parse_modality(const std::string& s) {
    if (s == "embedding") return Modality::Embedding;
    if (s == "vision")    return Modality::Vision;
    if (s == "audio")     return Modality::Audio;
    return Modality::Text;
}

GatewayConfig load_config(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("infcore: не удалось открыть конфиг: " + path);

    std::stringstream ss;
    ss << f.rdbuf();

    json j;
    try {
        j = json::parse(ss.str(), /*cb*/ nullptr, /*allow_exceptions*/ true,
                        /*ignore_comments*/ true);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("infcore: ошибка разбора конфига: ") + e.what());
    }

    GatewayConfig cfg;

    if (j.contains("server")) {
        const auto& s = j.at("server");
        cfg.host = s.value("host", cfg.host);
        cfg.port = s.value("port", cfg.port);
        cfg.metrics_port = s.value("metrics_port", cfg.metrics_port);
        cfg.request_timeout_ms = s.value("request_timeout_ms", cfg.request_timeout_ms);
    }
    if (j.contains("security")) {
        const auto& s = j.at("security");
        cfg.rbac_enabled = s.value("rbac_enabled", cfg.rbac_enabled);
        if (s.contains("api_keys"))
            for (const auto& k : s.at("api_keys")) cfg.api_keys.push_back(k.get<std::string>());
    }
    if (j.contains("offline"))
        cfg.enforce_no_egress = j.at("offline").value("enforce_no_egress", true);

    if (j.contains("models")) {
        for (const auto& m : j.at("models")) {
            ModelEntry e;
            e.logical_name   = m.value("logical_name", std::string());
            e.gguf_path      = m.value("gguf_path", std::string());
            e.arch           = m.value("arch", std::string());
            e.backend_url    = m.value("backend_url", std::string());
            e.upstream_model = m.value("upstream_model", e.logical_name);
            e.modality       = parse_modality(m.value("modality", std::string("text")));
            e.enabled        = m.value("enabled", true);
            e.n_ctx          = m.value("n_ctx", 8192);
            e.n_gpu_layers   = m.value("n_gpu_layers", 0);
            if (e.logical_name.empty())
                throw std::runtime_error("infcore: model без logical_name");
            cfg.models.push_back(std::move(e));
        }
    }

    if (cfg.api_keys.empty())
        throw std::runtime_error("infcore: security.api_keys пуст — нужен хотя бы один ключ");
    if (cfg.models.empty())
        throw std::runtime_error("infcore: models пуст");

    return cfg;
}

}  // namespace infcore
