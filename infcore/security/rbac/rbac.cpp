// infcore — корп. лицензия. RBAC: роль -> разрешённые модели/эндпоинты/лимиты.
namespace infcore { bool rbac_allow(const char* /*role*/, const char* /*resource*/) { return false; } }
