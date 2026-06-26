// infcore — корп. лицензия. Аутентификация: статические API-ключи / mTLS / OIDC
// внутреннего IdP (offline). Без обращений к внешним провайдерам.
namespace infcore { bool authn_verify(const char* /*bearer*/) { return false; } }
