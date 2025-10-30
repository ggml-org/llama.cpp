# Axelera Backend Tests

cmake_minimum_required(VERSION 3.14)

# Test executable for Axelera backend
add_executable(test-backend-axelera test-backend-axelera.cpp)

# Link against GGML libraries
target_link_libraries(test-backend-axelera PRIVATE ggml ggml-axelera)

# Include directories
target_include_directories(test-backend-axelera PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
    ${CMAKE_SOURCE_DIR}/include
)

# Set output directory
set_target_properties(test-backend-axelera PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# On macOS, set rpath for finding shared libraries
if(APPLE)
    set_target_properties(test-backend-axelera PROPERTIES
        BUILD_RPATH "@executable_path"
        INSTALL_RPATH "@executable_path"
    )
endif()

message(STATUS "Axelera backend tests configured")
