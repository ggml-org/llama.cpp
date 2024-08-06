find_package (Threads REQUIRED)

set(TARGET vulkan-shaders-gen)
add_executable(${TARGET} vulkan-shaders-gen.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_compile_features(${TARGET} PRIVATE cxx_std_11)
target_link_libraries(vulkan-shaders-gen PUBLIC Threads::Threads)
