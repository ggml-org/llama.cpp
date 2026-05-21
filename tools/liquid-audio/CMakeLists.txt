# lib
set(TARGET_LIB liquid-audio)
add_library(${TARGET_LIB} runner.cpp)
target_include_directories(${TARGET_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET_LIB} PUBLIC llama common mtmd ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_LIB} PRIVATE cxx_std_17)

# cli
set(TARGET_CLI llama-liquid-audio-cli)
add_executable(${TARGET_CLI} cli.cpp miniaudio_impl.c)
target_include_directories(${TARGET_CLI} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)
target_link_libraries(${TARGET_CLI} PRIVATE ${TARGET_LIB} ${CMAKE_THREAD_LIBS_INIT} dl m pthread)
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET_CLI} RUNTIME)
endif()

# server
set(TARGET_SERVER llama-liquid-audio-server)
add_executable(${TARGET_SERVER} server.cpp)
target_link_libraries(${TARGET_SERVER} PRIVATE ${TARGET_LIB} cpp-httplib)
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET_SERVER} RUNTIME)
endif()
