# check-cuda-rsqrt-noexcept.cmake
#
# Detect and optionally fix the CUDA / glibc rsqrt noexcept mismatch.
#
# CUDA 12.8 – 13.x ship math_functions.h with rsqrt/rsqrtf declared
# without noexcept, but glibc >= 2.40 declares them with noexcept(true).
# This causes enable_language(CUDA) and any CUDA compilation to fail:
#
#   error: exception specification is incompatible with that of previous
#   function "rsqrt"
#
# When GGML_CUDA_PATCH_RSQRT_NOEXCEPT is ON (default) this module patches
# the CUDA Toolkit header in-place so compilation succeeds.  The original
# file is backed up with a .orig suffix.  If the header is not writable
# (common when the toolkit was installed system-wide via a package manager)
# the module prints exact sudo commands to apply the fix manually.
#
# Set GGML_CUDA_PATCH_RSQRT_NOEXCEPT=OFF to skip auto-patching and get a
# clear diagnostic instead.
#
# References:
#   https://github.com/ggml-org/llama.cpp/issues/16685
#   https://github.com/ggml-org/llama.cpp/issues/17041
#   https://github.com/ggml-org/llama.cpp/issues/19100

option(GGML_CUDA_PATCH_RSQRT_NOEXCEPT
    "Automatically patch CUDA math_functions.h to fix rsqrt noexcept mismatch with glibc >= 2.40"
    ON)

function(ggml_check_cuda_rsqrt_noexcept)
    if (NOT CUDAToolkit_FOUND)
        return()
    endif()

    set(_math_h "${CUDAToolkit_TARGET_DIR}/include/crt/math_functions.h")
    if (NOT EXISTS "${_math_h}")
        return()
    endif()

    # ---- 1. Check whether the header needs patching ----
    file(READ "${_math_h}" _content)

    # Match the un-patched pattern: "double<whitespace>rsqrt(double x);"
    # without any "noexcept" on the same line.
    string(REGEX MATCH "double[ \t]+rsqrt\\(double [a-z]\\)[ \t]*;" _match "${_content}")
    if (NOT _match)
        # Already patched or different layout — nothing to do.
        return()
    endif()

    # ---- 2. Check whether the running glibc causes the conflict ----
    if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        return()
    endif()

    execute_process(
        COMMAND ldd --version
        OUTPUT_VARIABLE _ldd_out
        ERROR_VARIABLE  _ldd_out
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    string(REGEX MATCH "([0-9]+)\\.([0-9]+)" _glibc_ver "${_ldd_out}")
    if (NOT _glibc_ver)
        return()
    endif()

    string(REGEX REPLACE "^([0-9]+)\\.[0-9]+$" "\\1" _major "${_glibc_ver}")
    string(REGEX REPLACE "^[0-9]+\\.([0-9]+)$" "\\1" _minor "${_glibc_ver}")
    math(EXPR _combined "${_major} * 100 + ${_minor}")
    if (_combined LESS 240)
        return()
    endif()

    # ---- 3. Mismatch confirmed ----
    message(STATUS "Detected CUDA/glibc rsqrt noexcept mismatch "
                   "(glibc ${_glibc_ver}, CUDA ${CUDAToolkit_VERSION})")

    set(_sed_cmds
        "  sudo sed -i 's/double *rsqrt(double x);/double rsqrt(double x) noexcept(true);/' '${_math_h}'\n"
        "  sudo sed -i 's/float *rsqrtf(float x);/float  rsqrtf(float x) noexcept(true);/' '${_math_h}'")

    if (NOT GGML_CUDA_PATCH_RSQRT_NOEXCEPT)
        message(FATAL_ERROR
            "CUDA math_functions.h declares rsqrt/rsqrtf without noexcept, "
            "but glibc ${_glibc_ver} requires noexcept(true).\n"
            "This is a known CUDA Toolkit bug.\n"
            "\n"
            "Fix options:\n"
            "  1. Re-run cmake with -DGGML_CUDA_PATCH_RSQRT_NOEXCEPT=ON (default)\n"
            "  2. Manually patch:\n"
            "${_sed_cmds}\n"
            "  3. Wait for NVIDIA to ship a fixed CUDA Toolkit\n")
    endif()

    # ---- 4. Try to auto-patch ----
    # Check writability before attempting file(WRITE) which would abort.
    execute_process(
        COMMAND test -w "${_math_h}"
        RESULT_VARIABLE _not_writable)

    if (_not_writable)
        message(FATAL_ERROR
            "CUDA/glibc rsqrt noexcept mismatch detected but ${_math_h}\n"
            "is not writable by the current user.\n"
            "\n"
            "Please run:\n"
            "${_sed_cmds}\n"
            "\n"
            "Then re-run cmake.  (Or use -DGGML_CUDA_PATCH_RSQRT_NOEXCEPT=OFF to skip.)\n")
    endif()

    # Back up the original header into the build directory (the CUDA
    # include directory itself is typically not writable).
    set(_backup "${CMAKE_CURRENT_BINARY_DIR}/math_functions.h.orig")
    if (NOT EXISTS "${_backup}")
        file(WRITE "${_backup}" "${_content}")
        message(STATUS "  Backed up original to ${_backup}")
    endif()

    # Apply the fix.
    string(REGEX REPLACE
        "(double[ \t]+)rsqrt\\(double ([a-z])\\)[ \t]*;"
        "\\1rsqrt(double \\2) noexcept(true);"
        _patched "${_content}")
    string(REGEX REPLACE
        "(float[ \t]+)rsqrtf\\(float ([a-z])\\)[ \t]*;"
        "\\1rsqrtf(float \\2) noexcept(true);"
        _patched "${_patched}")

    file(WRITE "${_math_h}" "${_patched}")
    message(STATUS "  Patched ${_math_h} — added noexcept(true) to rsqrt/rsqrtf")
endfunction()
