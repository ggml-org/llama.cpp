set(TARGET llama-semantic-server)

add_executable(${TARGET} semantic-server.cpp)

target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE llama common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (WIN32)
    # No additional libraries needed for Windows named pipes
endif()

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

# Unit test for validator (doesn't require a model)
set(TARGET_TEST llama-semantic-server-test)
add_executable(${TARGET_TEST} test-validator.cpp)
target_include_directories(${TARGET_TEST} PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(${TARGET_TEST} PRIVATE ${CMAKE_SOURCE_DIR}/ggml/include)
target_include_directories(${TARGET_TEST} PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(${TARGET_TEST} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)
target_link_libraries(${TARGET_TEST} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_TEST} PRIVATE cxx_std_17)
