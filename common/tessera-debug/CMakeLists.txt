llama_add_compile_flags()

#
# llama-tessera-debug
#
# Tessera Layer 1 sidecar writer. Lives below the backends
# (ggml-cpu / ggml-cuda / ggml-metal) so the backends can link against
# it directly without dragging in the whole common/ library. Common
# itself links against this target so arg.cpp's --tessera-dequant-dir
# CLI flag still resolves the same symbols.
#

set(TARGET llama-tessera-debug)

add_library(${TARGET} STATIC
    tessera-debug.cpp
    tessera-debug.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}       # tessera-debug.h via "tessera-debug.h"
)

target_compile_features(${TARGET} PUBLIC cxx_std_17)
