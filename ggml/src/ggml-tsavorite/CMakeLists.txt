message(STATUS "Tsavorite framework is found")
#
# tsavorite Kernel Library
add_compile_options(--std=c++20)
ggml_add_backend_library(ggml-tsavorite
                         ggml-tsavorite.cpp
                        )

# Add compiler lib directory to link path for FFM libraries
if(DEFINED MLIR_COMPILER_DIR)
    target_link_directories(ggml-tsavorite PRIVATE "${MLIR_COMPILER_DIR}/lib")
endif()

target_link_libraries(ggml-tsavorite PRIVATE ${TLIBS} dl rt)
