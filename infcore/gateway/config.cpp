// infcore gateway — корпоративная лицензия.
#include "config.hpp"

#include <cctype>
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

// Строгий разбор dotted-quad IPv4: ровно 4 числовых октета 0..255, без лишних
// символов. Хостнейм вида "127.0.0.1.evil.com" или "10.example.org" НЕ является
// IPv4 -> не пройдёт как локальный (закрывает обход по префиксу).
static bool parse_ipv4(const std::string& h, unsigned o[4]) {
    unsigned vals[4];
    size_t i = 0;
    const size_t len = h.size();
    for (int part = 0; part < 4; ++part) {
        if (i >= len || !std::isdigit((unsigned char)h[i])) return false;
        unsigned v = 0;
        int digits = 0;
        while (i < len && std::isdigit((unsigned char)h[i])) {
            v = v * 10 + (unsigned)(h[i] - '0');
            if (++digits > 3) return false;
            ++i;
        }
        if (v > 255) return false;
        vals[part] = v;
        if (part < 3) { if (i >= len || h[i] != '.') return false; ++i; }
    }
    if (i != len) return false;   // хвост после 4-го октета -> это не IPv4
    for (int k = 0; k < 4; ++k) o[k] = vals[k];
    return true;
}

// Хост URL локальный (loopback/RFC1918/localhost)? Для offline-инварианта:
// внешние backend_url обязаны указывать внутрь контура, не в интернет.
// Строго отсекаем userinfo ("http://127.0.0.1@evil.com" -> host = evil.com) и
// проверяем именно IP/точное localhost, а не совпадение префикса строки.
static bool is_local_host(const std::string& url) {
    std::string h = url;
    auto p = h.find("://");
    if (p != std::string::npos) h = h.substr(p + 3);
    h = h.substr(0, h.find_first_of("/?#"));           // только authority
    auto at = h.rfind('@');                            // userinfo -> берём хост после '@'
    if (at != std::string::npos) h = h.substr(at + 1);
    if (h.empty()) return false;
    if (h.front() == '[') {                            // IPv6 в скобках [..]:port
        auto e = h.find(']');
        if (e == std::string::npos) return false;
        std::string v6 = h.substr(1, e - 1);
        for (auto& c : v6) c = (char)std::tolower((unsigned char)c);
        return v6 == "::1" || v6.rfind("fd", 0) == 0 || v6.rfind("fc", 0) == 0 ||
               v6.rfind("fe80", 0) == 0;
    }
    auto colon = h.rfind(':');                         // отбрасываем порт
    if (colon != std::string::npos) h = h.substr(0, colon);
    if (h.empty()) return false;
    {
        std::string lower = h;
        for (auto& c : lower) c = (char)std::tolower((unsigned char)c);
        if (lower == "localhost") return true;
    }
    unsigned o[4];
    if (!parse_ipv4(h, o)) return false;               // не IPv4 и не localhost -> внешний
    if (o[0] == 127) return true;                              // 127.0.0.0/8 loopback
    if (o[0] == 10)  return true;                              // 10.0.0.0/8
    if (o[0] == 192 && o[1] == 168) return true;               // 192.168.0.0/16
    if (o[0] == 172 && o[1] >= 16 && o[1] <= 31) return true;  // 172.16.0.0/12
    if (o[0] == 169 && o[1] == 254) return true;               // 169.254.0.0/16 link-local (не в интернет)
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
        cfg.read_timeout_ms  = s.value("read_timeout_ms", cfg.read_timeout_ms);
        cfg.write_timeout_ms = s.value("write_timeout_ms", cfg.write_timeout_ms);
        cfg.max_body_bytes   = s.value("max_body_bytes", cfg.max_body_bytes);
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
            cfg.audit_sink    = a.value("sink", cfg.audit_sink);
            cfg.audit_path    = a.value("path", cfg.audit_path);
            cfg.audit_require = a.value("require", cfg.audit_require);
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

    // Оркестрация/Docker: host и port можно переопределить окружением, НЕ редактируя
    // смонтированный read-only конфиг. В контейнере INFCORE_HOST=0.0.0.0 (наружу
    // публикуется только loopback хоста), на bare-metal обычно не задаётся.
    if (const char* h = std::getenv("INFCORE_HOST"); h && *h) cfg.host = h;
    if (const char* p = std::getenv("INFCORE_PORT"); p && *p) {
        int v = std::atoi(p);
        if (v < 1 || v > 65535)
            throw std::runtime_error("infcore: INFCORE_PORT вне диапазона 1..65535: " + std::string(p));
        cfg.port = v;
    }

    for (const auto& m : cfg.models) {
        const bool vlm = (m.modality == Modality::Vision);
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
        // Vision без проектора запустилась бы битой - ловим на старте.
        if (vlm && m.mmproj_path.empty())
            throw std::runtime_error("infcore: модель '" + m.logical_name +
                "' модальности vision требует mmproj_path");
    }

    return cfg;
}

}  // namespace infcore
