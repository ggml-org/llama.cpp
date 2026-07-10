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

// Хост URL локальный (loopback/RFC1918/localhost)? Для offline-инварианта:
// внешние backend_url обязаны указывать внутрь контура, не в интернет.
static bool is_local_host(const std::string& url) {
    std::string h = url;
    auto p = h.find("://");
    if (p != std::string::npos) h = h.substr(p + 3);
    h = h.substr(0, h.find_first_of("/?"));            // отбрасываем путь
    if (!h.empty() && h.front() == '[') {              // IPv6 в скобках
        auto e = h.find(']');
        std::string v6 = h.substr(1, e == std::string::npos ? std::string::npos : e - 1);
        return v6 == "::1" || v6.rfind("fd", 0) == 0 || v6.rfind("fc", 0) == 0 || v6.rfind("fe80", 0) == 0;
    }
    h = h.substr(0, h.rfind(':'));                     // отбрасываем порт
    if (h == "localhost") return true;
    if (h.rfind("127.", 0) == 0) return true;
    if (h.rfind("10.", 0) == 0) return true;
    if (h.rfind("192.168.", 0) == 0) return true;
    if (h.rfind("172.", 0) == 0) {                     // 172.16.0.0 - 172.31.255.255
        int oct = std::atoi(h.c_str() + 4);
        if (oct >= 16 && oct <= 31) return true;
    }
    return false;
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
        cfg.max_concurrent_requests = s.value("max_concurrent_requests", cfg.max_concurrent_requests);
        cfg.request_timeout_ms = s.value("request_timeout_ms", cfg.request_timeout_ms);
    }
    if (j.contains("security")) {
        const auto& s = j.at("security");
        cfg.rbac_enabled = s.value("rbac_enabled", cfg.rbac_enabled);
        if (s.contains("api_keys"))
            for (const auto& k : s.at("api_keys")) {
                std::string key = resolve_secret(k.get<std::string>());
                if (key.rfind("change-me", 0) == 0)
                    throw std::runtime_error("infcore: заглушечный ключ 'change-me...' в security.api_keys - задайте реальный ключ (env:/file:)");
                cfg.api_keys.push_back(std::move(key));
            }
        if (s.contains("principals")) {
            for (const auto& p : s.at("principals")) {
                ApiKeyPrincipal ap;
                ap.api_key           = resolve_secret(p.value("api_key", std::string()));
                ap.principal.subject = p.value("subject", std::string());
                ap.principal.role    = p.value("role", std::string());
                if (ap.api_key.empty())
                    throw std::runtime_error("infcore: principal без api_key");
                if (ap.api_key.rfind("change-me", 0) == 0)
                    throw std::runtime_error("infcore: заглушечный ключ 'change-me...' у principal '" +
                        ap.principal.subject + "' - задайте реальный ключ (env:/file:)");
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
            e.mmproj_path    = m.value("mmproj_path", std::string());
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

    for (const auto& m : cfg.models) {
        const bool vlm = (m.modality == Modality::Vision || m.modality == Modality::Audio);
        if (!m.backend_url.empty()) {
            // Внешний бэкенд: при жёстком offline обязан быть локальным (не в интернет).
            if (cfg.enforce_no_egress && !is_local_host(m.backend_url))
                throw std::runtime_error("infcore: enforce_no_egress: backend_url модели '" +
                    m.logical_name + "' не локальный: " + m.backend_url);
            continue;
        }
        // Управляемая модель (без backend_url) требует llama_server_bin и gguf_path.
        if (cfg.llama_server_bin.empty())
            throw std::runtime_error("infcore: модель '" + m.logical_name +
                "' без backend_url требует runtime.llama_server_bin");
        if (m.gguf_path.empty())
            throw std::runtime_error("infcore: управляемая модель '" + m.logical_name +
                "' требует gguf_path");
        // Vision/audio без проектора запустились бы битыми - ловим на старте.
        if (vlm && m.mmproj_path.empty())
            throw std::runtime_error("infcore: модель '" + m.logical_name +
                "' модальности vision/audio требует mmproj_path");
    }

    return cfg;
}

}  // namespace infcore
