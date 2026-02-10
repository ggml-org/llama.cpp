# cmake/FindNCCL.cmake

find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    HINTS ${NCCL_ROOT} $ENV{NCCL_ROOT} $ENV{CUDA_HOME} /usr/local/cuda
    PATH_SUFFIXES include
)

find_library(NCCL_LIBRARY
    NAMES nccl
    HINTS ${NCCL_ROOT} $ENV{NCCL_ROOT} $ENV{CUDA_HOME} /usr/local/cuda
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
        add_library(NCCL::NCCL UNKNOWN IMPORTED)
        set_target_properties(NCCL::NCCL PROPERTIES
            IMPORTED_LOCATION "${NCCL_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY)
