// infcore gateway — корпоративная лицензия.
// Загрузка конфигурации gateway из JSON (валиден и как YAML-подмножество).
#pragma once

#include <string>
#include <vector>

#include "registry/model_registry.h"
#include "security/authn/authn.h"
#include "security/rbac/rbac.h"

namespace infcore {

// API-ключ + кому он принадлежит (subject/role). Источник identity, offline.
struct ApiKeyPrincipal {
    std::string api_key;
    Principal   principal;
};

struct GatewayConfig {
    std::string host = "127.0.0.1";
    int         port = 8080;
    int         max_concurrent_requests = 8;
    int         request_timeout_ms = 120000;
    bool        rbac_enabled = true;
    bool        enforce_no_egress = true;

    // runtime: lazy-подъём управляемых бэкендов (модели с пустым backend_url)
    std::string llama_server_bin;            // путь к нашему llama-server (из сборки)
    int         port_range_start   = 8100;
    int         idle_timeout_ms    = 300000;
    int         startup_timeout_ms = 120000;

    std::vector<std::string> api_keys;   // legacy: плоский список ключей (роль admin)
    std::vector<ModelEntry>  models;

    // RBAC: principals (ключ -> subject/role) и роли (allowlists моделей/эндпоинтов).
    std::vector<ApiKeyPrincipal> principals;
    std::vector<Role>            roles;

    // audit: локальный append-only журнал.
    std::string audit_sink = "file";     // "file" | "none"
    std::string audit_path = "infcore-audit.log";
};

// Загружает конфиг из файла. Бросает std::runtime_error при ошибке/невалидности.
GatewayConfig load_config(const std::string& path);

}  // namespace infcore
