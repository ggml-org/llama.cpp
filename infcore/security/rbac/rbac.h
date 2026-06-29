// infcore — корп. лицензия. RBAC: роль -> разрешённые модели/эндпоинты.
// Default-deny: при включённом RBAC доступ есть только по явному правилу.
#pragma once

#include <map>
#include <string>
#include <vector>

namespace infcore {

struct Role {
    std::string name;
    std::vector<std::string> allow_models;     // "*" = любые
    std::vector<std::string> allow_endpoints;  // "*" = любые
};

class Authorizer {
public:
    void add_role(const Role& r);
    void set_enabled(bool e) { enabled_ = e; }

    // Проверяет, что роль допускает endpoint и (если непуста) модель.
    // reason заполняется причиной отказа (для аудита). При enabled_=false всегда true.
    bool allow(const std::string& role, const std::string& endpoint,
               const std::string& model, std::string& reason) const;

    // Разрешает ли роль доступ к модели (для фильтрации /v1/models).
    bool model_allowed(const std::string& role, const std::string& model) const;

private:
    bool enabled_ = true;
    std::map<std::string, Role> roles_;
};

}  // namespace infcore
