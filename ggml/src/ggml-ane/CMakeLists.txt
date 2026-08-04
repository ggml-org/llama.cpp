# The ANE backend targets Apple Silicon only (IOSurface + Core ML). Fail fast
# if someone enables GGML_ANE on another platform instead of silently dropping it.
if (NOT APPLE)
    message(FATAL_ERROR "GGML_ANE requires Apple Silicon (IOSurface / Core ML).")
endif()

find_library(FOUNDATION_LIBRARY Foundation  REQUIRED)
find_library(IO_SURFACE_FRAMEWORK IOSurface  REQUIRED)
find_library(COREML_FRAMEWORK    CoreML     REQUIRED)
find_library(METAL_FRAMEWORK     Metal      REQUIRED)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)

message(STATUS "ANE (Core ML) backend enabled")

ggml_add_backend_library(ggml-ane
                         ggml-ane.mm
                        )

target_link_libraries(ggml-ane PRIVATE
                      ${FOUNDATION_LIBRARY}
                      ${IO_SURFACE_FRAMEWORK}
                      ${COREML_FRAMEWORK}
                      ${METAL_FRAMEWORK}
                      ${ACCELERATE_FRAMEWORK}
                      )

# Smoke test: builds a tiny graph of supported ops and checks the output
# against a CPU reference. Also exercises the Core ML program loader when
# --bundle <mlmodelc dir> is passed.
if (GGML_ANE_BUILD_TEST)
    add_executable(test-ggml-ane test-ggml-ane.cpp)
    # ggml (the loader/registration shim) provides the backend API and the
    # dynamic loader; ggml-base provides the allocator and tensor helpers.
    target_link_libraries(test-ggml-ane PRIVATE ggml ggml-base)
    set_target_properties(test-ggml-ane PROPERTIES
                          RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()
