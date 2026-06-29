// infcore gateway — корпоративная лицензия.
#include "config.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

#include "json_schema.hpp"
#include "schema_embedded.h"   // сгенерировано CMake: kGatewaySchemaJson

using json = nlohmann::json;

namespace infcore {

static Modality parse_modality(const std::string& s) {
    if (s == "embedding") return Modality::Embedding;
    if (s == "vision")    return Modality::Vision;
    if (s == "audio")     return Modality::Audio;
    return Modality::Text;
}

// Разрешает секрет, не зашивая его в конфиг/образ:
//   "env:VAR"   -> значение переменной окружения VAR
//   "file:/p"   -> содержимое файла /p (обрезаются хвостовые переводы строк)
//   иначе       -> строка как есть (literal)
static std::string resolve_secret(const std::string& v) {
    if (v.rfind("env:", 0) == 0) {
        const char* e = std::getenv(v.c_str() + 4);
        if (!e || !*e)
            throw std::runtime_error("infcore: переменная окружения не задана: " + v.substr(4));
        return e;
    }
    if (v.rfind("file:", 0) == 0) {
        std::ifstream f(v.substr(5));
        if (!f) throw std::runtime_error("infcore: не удалось прочитать файл секрета: " + v.substr(5));
        std::stringstream ss; ss << f.rdbuf();
        std::string s = ss.str();
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        return s;
    }
    return v;
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

    // Формальная валидация по встроенной JSON-Schema (fail-fast при старте).
    {
        json schema = json::parse(kGatewaySchemaJson, nullptr, true, true);
        auto errs = json_schema_validate(j, schema);
        if (!errs.empty()) {
            std::string msg = "infcore: конфиг не соответствует JSON-Schema:";
            for (const auto& e : errs) msg += "\n  - " + e;
            throw std::runtime_error(msg);
        }
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
            for (const auto& k : s.at("api_keys")) cfg.api_keys.push_back(resolve_secret(k.get<std::string>()));
        if (s.contains("principals")) {
            for (const auto& p : s.at("principals")) {
                ApiKeyPrincipal ap;
                ap.api_key           = resolve_secret(p.value("api_key", std::string()));
                ap.principal.subject = p.value("subject", std::string());
                ap.principal.role    = p.value("role", std::string());
                if (ap.api_key.empty())
                    throw std::runtime_error("infcore: principal без api_key");
                cfg.principals.push_back(std::move(ap));
            }
        }
        if (s.contains("roles")) {
            for (const auto& r : s.at("roles")) {
                Role role;
                role.name = r.value("name", std::string());
                if (role.name.empty())
                    throw std::runtime_error("infcore: role без name");
                if (r.contains("allow_models"))
                    for (const auto& m : r.at("allow_models")) role.allow_models.push_back(m.get<std::string>());
                if (r.contains("allow_endpoints"))
                    for (const auto& ep : r.at("allow_endpoints")) role.allow_endpoints.push_back(ep.get<std::string>());
                cfg.roles.push_back(std::move(role));
            }
        }
        if (s.contains("audit")) {
            const auto& a = s.at("audit");
            cfg.audit_sink = a.value("sink", cfg.audit_sink);
            cfg.audit_path = a.value("path", cfg.audit_path);
        }
    }
    if (j.contains("offline"))
        cfg.enforce_no_egress = j.at("offline").value("enforce_no_egress", true);

    if (j.contains("runtime")) {
        const auto& r = j.at("runtime");
        cfg.llama_server_bin   = r.value("llama_server_bin", cfg.llama_server_bin);
        cfg.port_range_start   = r.value("port_range_start", cfg.port_range_start);
        cfg.idle_timeout_ms    = r.value("idle_timeout_ms", cfg.idle_timeout_ms);
        cfg.startup_timeout_ms = r.value("startup_timeout_ms", cfg.startup_timeout_ms);
    }

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

    if (cfg.api_keys.empty() && cfg.principals.empty())
        throw std::runtime_error("infcore: нет ни security.api_keys, ни security.principals — нужен хотя бы один ключ");
    if (cfg.models.empty())
        throw std::runtime_error("infcore: models пуст");

    // При включённом RBAC роль каждого principal должна быть объявлена в security.roles.
    if (cfg.rbac_enabled) {
        for (const auto& ap : cfg.principals) {
            bool found = false;
            for (const auto& r : cfg.roles) if (r.name == ap.principal.role) { found = true; break; }
            if (!found)
                throw std::runtime_error("infcore: роль '" + ap.principal.role +
                    "' principal'а '" + ap.principal.subject + "' не объявлена в security.roles");
        }
    }

    // Управляемая модель (без backend_url) требует runtime.llama_server_bin и gguf_path.
    for (const auto& m : cfg.models) {
        if (!m.backend_url.empty()) continue;
        if (cfg.llama_server_bin.empty())
            throw std::runtime_error("infcore: модель '" + m.logical_name +
                "' без backend_url требует runtime.llama_server_bin");
        if (m.gguf_path.empty())
            throw std::runtime_error("infcore: управляемая модель '" + m.logical_name +
                "' требует gguf_path");
    }

    return cfg;
}

}  // namespace infcore
