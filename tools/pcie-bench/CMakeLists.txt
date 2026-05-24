cmake_minimum_required(VERSION 3.12)
set(PROJECT_NAME pcie-bench)

find_package(CUDAToolkit QUIET)

if(NOT CUDAToolkit_FOUND)
    message(STATUS "pcie-bench: CUDA not found, skipping")
    return()
endif()

add_executable(${PROJECT_NAME} pcie-bench.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ggml-cpu ggml-base CUDA::cudart)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Tools)
