set(TARGET llama-bench)
add_executable(${TARGET} llama-bench.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET_QUANT quant-bench)
add_executable(${TARGET_QUANT} quant-bench.cpp)
target_link_libraries(${TARGET_QUANT} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_QUANT} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} ${TARGET_QUANT} RUNTIME)
endif()
