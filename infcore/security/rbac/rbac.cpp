// infcore — корп. лицензия.
#include "rbac/rbac.h"

namespace infcore {

namespace {
bool listed(const std::vector<std::string>& allow, const std::string& item) {
    for (const auto& a : allow) {
        if (a == "*") return true;
        if (a == item) return true;
    }
    return false;
}
}  // namespace

void Authorizer::add_role(const Role& r) { roles_[r.name] = r; }

bool Authorizer::allow(const std::string& role, const std::string& endpoint,
                       const std::string& model, std::string& reason) const {
    if (!enabled_) return true;

    auto it = roles_.find(role);
    if (it == roles_.end()) { reason = "unknown role: " + role; return false; }
    const Role& r = it->second;

    if (!listed(r.allow_endpoints, endpoint)) {
        reason = "endpoint not allowed for role " + role + ": " + endpoint;
        return false;
    }
    if (!model.empty() && !listed(r.allow_models, model)) {
        reason = "model not allowed for role " + role + ": " + model;
        return false;
    }
    return true;
}

bool Authorizer::model_allowed(const std::string& role, const std::string& model) const {
    if (!enabled_) return true;
    auto it = roles_.find(role);
    if (it == roles_.end()) return false;
    return listed(it->second.allow_models, model);
}

}  // namespace infcore
