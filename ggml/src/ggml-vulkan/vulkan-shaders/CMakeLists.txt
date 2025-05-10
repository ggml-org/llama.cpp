cmake_minimum_required(VERSION 3.19)
project("vulkan-shaders-gen" C CXX)

find_package(Threads REQUIRED)

# Include the shader extensions definitions file
if(NOT DEFINED VULKAN_SHADER_EXTENSIONS_FILE OR NOT EXISTS "${VULKAN_SHADER_EXTENSIONS_FILE}")
    message(FATAL_ERROR "VULKAN_SHADER_EXTENSIONS_FILE must be defined and must exist")
endif()

message(STATUS "Loading shader extension definitions from: ${VULKAN_SHADER_EXTENSIONS_FILE}")
include("${VULKAN_SHADER_EXTENSIONS_FILE}")

set(TARGET vulkan-shaders-gen)
add_executable(${TARGET} vulkan-shaders-gen.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_link_libraries(vulkan-shaders-gen PUBLIC Threads::Threads)

# Configure output directories for MSVC builds
if(MSVC)
    # Get the main project's runtime output directory if possible
    if(DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
        foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
            string(TOUPPER ${CONFIG} CONFIG)
            set_target_properties(${TARGET} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY_${CONFIG} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
        endforeach()
    endif()
endif()
