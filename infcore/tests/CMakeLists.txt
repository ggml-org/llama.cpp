# infcore tests — корпоративная лицензия. Юнит-тесты слоя gateway (без движка).
# Встраиваем ту же JSON-Schema, что и gateway (config.cpp её требует).
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/../config/schema/gateway.schema.json)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/../config/schema/gateway.schema.json INFCORE_SCHEMA_JSON)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/../gateway/schema_embedded.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/schema_embedded.h
    @ONLY)

add_executable(infcore_unit_tests
    unit/test_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../gateway/config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../gateway/json_schema.cpp
)
target_include_directories(infcore_unit_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..          # gateway/*.hpp, security/*, registry/*
    ${CMAKE_CURRENT_SOURCE_DIR}/../gateway  # config.hpp, json_schema.hpp
    ${CMAKE_CURRENT_BINARY_DIR}             # schema_embedded.h
    ${INFCORE_LLAMA_ROOT}/vendor            # nlohmann/json.hpp
)
target_link_libraries(infcore_unit_tests PRIVATE infcore_security infcore_registry)

add_test(NAME infcore_unit COMMAND infcore_unit_tests)
