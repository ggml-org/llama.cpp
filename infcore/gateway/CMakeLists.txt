# infcore gateway — корпоративная лицензия.
# Строится РЯДОМ с tools/server (его файлы не редактируем). OpenAI-совместимый
# control-plane перед бэкендами llama-server. Использует vendored cpp-httplib +
# nlohmann/json из дерева движка; с ядром общается через C-API llama.h.
add_executable(infcore_gateway
    main.cpp
    config.cpp
    server.cpp
)

target_include_directories(infcore_gateway PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..          # registry/model_registry.h, gateway/*.hpp
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
