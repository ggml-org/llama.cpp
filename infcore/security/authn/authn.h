// infcore — корп. лицензия. Аутентификация: статические API-ключи внутреннего
// контура (mTLS/OIDC — на будущее). Без обращений к внешним провайдерам.
#pragma once

#include <string>
#include <vector>

namespace infcore {

struct Principal {
    std::string subject;   // кто (для аудита)
    std::string role;      // роль (для RBAC)
};

// Сопоставляет bearer-токен с принципалом. Источник истины - конфиг (offline).
// Сравнение ключей constant-time и без раннего выхода по списку, чтобы не
// давать timing-side-channel (ни длина совпадения, ни номер ключа).
class Authenticator {
public:
    void add_key(const std::string& api_key, const Principal& p);
    bool verify(const std::string& token, Principal& out) const;  // token без префикса "Bearer "
    bool empty() const { return keys_.empty(); }

private:
    struct Entry { std::string key; Principal principal; };
    std::vector<Entry> keys_;
};

// Извлекает токен из заголовка Authorization ("Bearer <token>"); пусто при ошибке.
std::string parse_bearer(const std::string& header);

}  // namespace infcore
