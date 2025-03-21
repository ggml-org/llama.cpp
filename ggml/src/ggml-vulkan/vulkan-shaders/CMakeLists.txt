cmake_minimum_required(VERSION 3.14) # Match top-level project
project(vulkan-shaders-gen)
find_package(Threads REQUIRED)
set(TARGET vulkan-shaders-gen)
add_executable(${TARGET} vulkan-shaders-gen.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_link_libraries(vulkan-shaders-gen PUBLIC Threads::Threads)
