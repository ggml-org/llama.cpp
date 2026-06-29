# infcore gateway — корпоративная лицензия.
# Строится РЯДОМ с tools/server (его файлы не редактируем). OpenAI-совместимый
# control-plane перед бэкендами llama-server. Использует vendored cpp-httplib +
# nlohmann/json из дерева движка; с ядром общается через C-API llama.h.
# Встраиваем JSON-Schema в бинарь: единый источник истины - файл схемы,
# заголовок генерируется на этапе конфигурации (config валидируется по нему при старте).
# Правка схемы должна пере-встраивать заголовок (иначе file(READ) кэшируется).
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/../config/schema/gateway.schema.json)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/../config/schema/gateway.schema.json INFCORE_SCHEMA_JSON)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/schema_embedded.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/schema_embedded.h
    @ONLY)

add_executable(infcore_gateway
    main.cpp
    config.cpp
    json_schema.cpp
    server.cpp
)

# Складываем бинарь в общий build/bin (туда же движок кладёт llama-server) -
# чтобы Dockerfile/деплой копировали всё из одного каталога.
set_target_properties(infcore_gateway PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

target_include_directories(infcore_gateway PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..          # registry/model_registry.h, gateway/*.hpp
    ${CMAKE_CURRENT_BINARY_DIR}             # schema_embedded.h (сгенерировано)
    ${INFCORE_LLAMA_ROOT}/vendor            # nlohmann/json.hpp
    ${INFCORE_LLAMA_ROOT}/vendor/cpp-httplib
)

# Прототип — proxy-front: gateway не линкует libllama (бэкенд = отдельный
# llama-server). Когда понадобится in-process режим, добавить llama-common +
# include движка (см. AUDIT.md §5, граница по C-API llama.h).
target_link_libraries(infcore_gateway PRIVATE
    infcore_security
    infcore_observability
    infcore_registry
    infcore_runtime       # lazy-supervisor дочерних llama-server
    cpp-httplib           # vendored HTTP сервер/клиент (REST + прокси)
)

install(TARGETS infcore_gateway RUNTIME)
