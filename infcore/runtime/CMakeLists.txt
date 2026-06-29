# infcore runtime — корпоративная лицензия.
# Lazy-supervisor дочерних процессов llama-server. Использует vendored cpp-httplib
# для health-поллинга; с движком не линкуется (управляет им как отдельным процессом).
add_library(infcore_runtime STATIC
    backend_supervisor.cpp
)

target_include_directories(infcore_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..          # registry/model_registry.h
)
target_include_directories(infcore_runtime PRIVATE
    ${INFCORE_LLAMA_ROOT}/vendor/cpp-httplib
)

target_link_libraries(infcore_runtime PUBLIC
    infcore_registry
)
target_link_libraries(infcore_runtime PRIVATE
    cpp-httplib
)
