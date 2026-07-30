set(TARGET llama-tweak)

add_executable(${TARGET} main.cpp record.cpp)
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../llama-bench)
target_link_libraries(${TARGET} PRIVATE llama-bench-impl llama-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
