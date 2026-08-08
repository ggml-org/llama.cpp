# benchmark-deterministic-draft: Benchmark for deterministic draft with MTP

set(TARGET benchmark-deterministic-draft)

add_executable(${TARGET} bench-deterministic-draft.cpp)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE llama-common deterministic_draft_spec llama ${CMAKE_THREAD_LIBS_INIT})

target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
