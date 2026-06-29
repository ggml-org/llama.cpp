// infcore — корп. лицензия. Аутентификация: статические API-ключи внутреннего
// контура (mTLS/OIDC — на будущее). Без обращений к внешним провайдерам.
#pragma once

#include <map>
#include <string>

namespace infcore {

struct Principal {
    std::string subject;   // кто (для аудита)
    std::string role;      // роль (для RBAC)
};

// Сопоставляет bearer-токен с принципалом. Источник истины - конфиг (offline).
class Authenticator {
public:
    void add_key(const std::string& api_key, const Principal& p);
    bool verify(const std::string& token, Principal& out) const;  // token без префикса "Bearer "
    bool empty() const { return by_key_.empty(); }

private:
    std::map<std::string, Principal> by_key_;
};

// Извлекает токен из заголовка Authorization ("Bearer <token>"); пусто при ошибке.
std::string parse_bearer(const std::string& header);

}  // namespace infcore
