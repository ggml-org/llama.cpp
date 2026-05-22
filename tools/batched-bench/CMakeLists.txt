# llama-batched-bench-impl: batched-bench logic, reusable by app

set(TARGET llama-batched-bench-impl)

add_library(${TARGET} batched-bench.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common llama ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-batched-bench executable

set(TARGET llama-batched-bench)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-batched-bench-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
