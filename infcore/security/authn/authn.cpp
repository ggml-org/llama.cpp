// infcore — корп. лицензия.
#include "authn/authn.h"

namespace infcore {

void Authenticator::add_key(const std::string& api_key, const Principal& p) {
    if (!api_key.empty()) by_key_[api_key] = p;
}

bool Authenticator::verify(const std::string& token, Principal& out) const {
    auto it = by_key_.find(token);
    if (it == by_key_.end()) return false;
    out = it->second;
    return true;
}

std::string parse_bearer(const std::string& header) {
    const std::string pfx = "Bearer ";
    if (header.size() <= pfx.size() || header.compare(0, pfx.size(), pfx) != 0)
        return std::string();
    return header.substr(pfx.size());
}

}  // namespace infcore
