set(TARGET llama-diffusion-gemma-server)
add_executable(${TARGET} diffusion-gemma-server.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Visual server: runs the optimized entropy-bound decoder and streams per-step canvas frames.
set(TARGET llama-diffusion-gemma-visual-server)
add_executable(${TARGET} diffusion-gemma-visual-server.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../diffusion)
target_link_libraries(${TARGET} PRIVATE llama-diffusion llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
