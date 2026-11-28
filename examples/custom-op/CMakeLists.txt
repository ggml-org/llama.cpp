set(TARGET custom-op)
add_executable(${TARGET} custom-op.cpp)
target_link_libraries(${TARGET} PRIVATE ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
