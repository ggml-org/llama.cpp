# llama-cli-impl: CLI logic, reusable by app

set(TARGET llama-cli-impl)

add_library(${TARGET} cli.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../server)
target_link_libraries(${TARGET} PUBLIC server-context llama-common ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-cli executable

set(TARGET llama-cli)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-cli-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
