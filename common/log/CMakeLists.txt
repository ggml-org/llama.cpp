llama_add_compile_flags()

#
# llama-log
#
# Low-level logging primitives. The LOG_* macros, the common_log_*
# worker-thread queue, and the tty_can_use_colors() helper all live
# here so the backends (ggml-cpu, ggml-cuda, ggml-metal) and the
# tessera-debug sidecar writer can use them without dragging in the
# whole common/ library. Both llama-common and llama-tessera-debug
# link against this target.
#
# Header layout:
#   common/log.h     back-compat shim, just `#include "log/log.h"`
#   common/log/log.h real header, public API
#   common/log/log.cpp implementation
#

set(TARGET llama-log)

add_library(${TARGET} STATIC
    log.cpp
    log.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# Expose the new subdir for `#include "log/log.h"` and the parent
# common/ dir for the back-compat `#include "log.h"`. The latter is
# what existing code (and the rest of llama.cpp) uses.
target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}        # for "log/log.h"
    ${CMAKE_CURRENT_SOURCE_DIR}/..     # for "log.h" (the shim)
)

target_compile_features(${TARGET} PUBLIC cxx_std_17)

# The ggml_log_level enum (used by LOG_* macros) lives in ggml.h. We
# do NOT link against the ggml target here -- the backends are
# shared libraries that link back to llama-tessera-debug (the
# Layer 1 dependency) which would form a cycle. The header is
# enough; llama-log only needs the enum and the GGML_LOG_LEVEL_*
# integer constants, all of which are pure declarations.
target_include_directories(${TARGET} PUBLIC
    ${CMAKE_SOURCE_DIR}/ggml/include
)

# log.cpp uses std::thread for the worker thread.
target_link_libraries(${TARGET} PUBLIC Threads::Threads)
