// infcore gateway — корпоративная лицензия.
// Загрузка конфигурации gateway из JSON (валиден и как YAML-подмножество).
#pragma once

#include <string>
#include <vector>

#include "registry/model_registry.h"

namespace infcore {

struct GatewayConfig {
    std::string host = "127.0.0.1";
    int         port = 8080;
    int         metrics_port = 9090;
    int         request_timeout_ms = 120000;
    bool        rbac_enabled = true;
    bool        enforce_no_egress = true;

    std::vector<std::string> api_keys;
    std::vector<ModelEntry>  models;
};

// Загружает конфиг из файла. Бросает std::runtime_error при ошибке/невалидности.
GatewayConfig load_config(const std::string& path);

}  // namespace infcore
