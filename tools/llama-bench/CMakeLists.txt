# bench-lib: benchmark logic, reusable by app

set(TARGET bench-lib)

add_library(${TARGET} llama-bench.cpp)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common llama ${CMAKE_THREAD_LIBS_INIT})

# llama-bench executable

set(TARGET llama-bench)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE bench-lib)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
