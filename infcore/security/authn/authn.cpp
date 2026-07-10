// infcore — корп. лицензия.
#include "authn/authn.h"

#include <cctype>

namespace infcore {

namespace {
// Сравнение без раннего выхода: время зависит только от длины token, не от
// содержимого/длины сохранённого ключа и не от позиции расхождения.
bool ct_eq(const std::string& token, const std::string& key) {
    unsigned diff = (unsigned)(token.size() ^ key.size());
    for (size_t i = 0; i < token.size(); ++i)
        diff |= (unsigned char)token[i] ^ (unsigned char)(i < key.size() ? key[i] : 0);
    return diff == 0;
}
}  // namespace

void Authenticator::add_key(const std::string& api_key, const Principal& p) {
    if (!api_key.empty()) keys_.push_back({api_key, p});
}

bool Authenticator::verify(const std::string& token, Principal& out) const {
    if (token.empty()) return false;
    bool found = false;
    Principal p;
    for (const auto& e : keys_) {
        if (ct_eq(token, e.key)) { found = true; p = e.principal; }  // без break: постоянное число сравнений
    }
    if (found) out = p;
    return found;
}

std::string parse_bearer(const std::string& header) {
    const std::string pfx = "bearer ";           // схема auth регистронезависима (RFC 7235)
    if (header.size() <= pfx.size()) return std::string();
    for (size_t i = 0; i < pfx.size(); ++i)
        if (std::tolower((unsigned char)header[i]) != pfx[i]) return std::string();
    return header.substr(pfx.size());
}

}  // namespace infcore
