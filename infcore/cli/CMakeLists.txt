# infcore CLI — корпоративная лицензия.
# Терминальный клиент к gateway (HTTP). С ядром не линкуется: только vendored
# cpp-httplib + nlohmann/json.
add_executable(infcore_cli main.cpp)
set_target_properties(infcore_cli PROPERTIES OUTPUT_NAME infcore-cli)

target_include_directories(infcore_cli PRIVATE
    ${INFCORE_LLAMA_ROOT}/vendor            # nlohmann/json.hpp
    ${INFCORE_LLAMA_ROOT}/vendor/cpp-httplib
)
target_link_libraries(infcore_cli PRIVATE cpp-httplib)

install(TARGETS infcore_cli RUNTIME)
