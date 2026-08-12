set(TARGET llama-prune)

add_executable(${TARGET}
    dataset.cpp
    dataset.h
    hard-prune.cpp
    hard-prune.h
    main.cpp
)

target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/ggml/include)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
