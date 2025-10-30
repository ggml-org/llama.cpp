# Axelera AI Accelerator Backend
#
# This CMakeLists.txt builds the Axelera backend for GGML.
# To use this backend, you need the Axelera SDK installed.

# Check if Axelera SDK path is provided
if (NOT AXELERA_SDK_PATH AND DEFINED ENV{AXELERA_SDK_PATH})
    set(AXELERA_SDK_PATH $ENV{AXELERA_SDK_PATH})
    message(STATUS "Axelera: Using AXELERA_SDK_PATH from environment: ${AXELERA_SDK_PATH}")
endif()

set(TARGET_NAME ggml-axelera)

# Build the Axelera backend library
ggml_add_backend_library(${TARGET_NAME}
    ggml-axelera.cpp
    ggml-axelera-compiler.cpp
    ggml-axelera-compiler.h
    ../../include/ggml-axelera.h
)

# If Axelera SDK is available, link against it
if (AXELERA_SDK_PATH)
    if (NOT EXISTS "${AXELERA_SDK_PATH}")
        message(WARNING "Axelera: AXELERA_SDK_PATH set but directory does not exist: ${AXELERA_SDK_PATH}")
    else()
        message(STATUS "Axelera: Found Axelera SDK at ${AXELERA_SDK_PATH}")

        # Set include directories for Axelera SDK
        if (EXISTS "${AXELERA_SDK_PATH}/include")
            target_include_directories(${TARGET_NAME} PRIVATE ${AXELERA_SDK_PATH}/include)
            message(STATUS "Axelera: Including SDK headers from ${AXELERA_SDK_PATH}/include")
        endif()

        # Set library directories for Axelera SDK
        if (EXISTS "${AXELERA_SDK_PATH}/lib")
            target_link_directories(${TARGET_NAME} PRIVATE ${AXELERA_SDK_PATH}/lib)
            message(STATUS "Axelera: Linking SDK libraries from ${AXELERA_SDK_PATH}/lib")

            # TODO: Link against actual Axelera SDK libraries when available
            # Example: target_link_libraries(${TARGET_NAME} PRIVATE axelera_runtime axelera_device)
        endif()

        # Define AXELERA_SDK_AVAILABLE to enable SDK-specific code
        target_compile_definitions(${TARGET_NAME} PRIVATE AXELERA_SDK_AVAILABLE)
    endif()
else()
    message(STATUS "Axelera: Building without SDK (stub implementation only)")
    message(STATUS "Axelera: Set AXELERA_SDK_PATH to enable full hardware support")
endif()

# Platform-specific options
if (UNIX AND NOT APPLE)
    # Linux-specific options
    target_link_libraries(${TARGET_NAME} PRIVATE pthread dl)
elseif (APPLE)
    # macOS-specific options
    target_link_libraries(${TARGET_NAME} PRIVATE pthread)
elseif (WIN32)
    # Windows-specific options
endif()

# Add any additional Axelera-specific compile definitions
# target_compile_definitions(${TARGET_NAME} PRIVATE AXELERA_OPTION=value)

message(STATUS "Axelera: Backend configured successfully")
