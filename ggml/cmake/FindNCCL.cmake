# cmake/FindNCCL.cmake

# NVIDIA does not distribute CMake files with NCCL, therefore use this file to find it instead.
#
# Inputs:
#   NCCL_ROOT        — root of an NCCL installation or source build tree
#   NCCL_STATIC      — if ON, prefer the static library and search cmake/lib/Release (or Debug);
#                      if OFF (default), prefer the shared/import library and search cmake/src/Release

if(NCCL_STATIC)
    # cmake source-build layout: cmake/lib/<Config>/nccl_static.lib (or nccl.lib)
    set(_nccl_lib_names nccl_static nccl)
    set(_nccl_extra_lib_hints
        "${NCCL_ROOT}/cmake/lib/Release"
        "${NCCL_ROOT}/cmake/lib/Debug"
        "${NCCL_ROOT}/cmake/lib"
    )
else()
    # cmake source-build layout: cmake/src/<Config>/nccl.lib (import lib for nccl.dll)
    set(_nccl_lib_names nccl)
    set(_nccl_extra_lib_hints
        "${NCCL_ROOT}/cmake/src/Release"
        "${NCCL_ROOT}/cmake/src/Debug"
        "${NCCL_ROOT}/cmake/src"
    )
endif()

find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    HINTS
        ${NCCL_ROOT}
        "${NCCL_ROOT}/cmake/src/Release"
        "${NCCL_ROOT}/cmake/src/Debug"
        "${NCCL_ROOT}/cmake/src"
        "${NCCL_ROOT}/cmake"
        $ENV{NCCL_ROOT}
        $ENV{CUDA_HOME}
        /usr/local/cuda
    PATH_SUFFIXES include src/include
)

find_library(NCCL_LIBRARY
    NAMES ${_nccl_lib_names}
    HINTS
        ${_nccl_extra_lib_hints}
        ${NCCL_ROOT}
        $ENV{NCCL_ROOT}
        $ENV{CUDA_HOME}
        /usr/local/cuda
    PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL
    DEFAULT_MSG
    NCCL_LIBRARY NCCL_INCLUDE_DIR
)

if(NCCL_FOUND)
    set(NCCL_LIBRARIES ${NCCL_LIBRARY})
    set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})

    if(NOT TARGET NCCL::NCCL)
        if(NCCL_STATIC)
            add_library(NCCL::NCCL STATIC IMPORTED)
        else()
            add_library(NCCL::NCCL UNKNOWN IMPORTED)
        endif()
        set_target_properties(NCCL::NCCL PROPERTIES
            IMPORTED_LOCATION "${NCCL_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY)
