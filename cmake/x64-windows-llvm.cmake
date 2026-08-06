set( CMAKE_SYSTEM_NAME Windows )
set( CMAKE_SYSTEM_PROCESSOR x86_64 )

set( CMAKE_C_COMPILER    clang )
set( CMAKE_CXX_COMPILER  clang++ )

set( GGML_OPENMP_LLVM_ROOT "" CACHE PATH "LLVM root containing the target OpenMP runtime" )
if ((NOT DEFINED GGML_OPENMP OR GGML_OPENMP) AND GGML_OPENMP_LLVM_ROOT)
    file( TO_CMAKE_PATH "${GGML_OPENMP_LLVM_ROOT}" ggml_openmp_llvm_root )
    set( ggml_openmp_library "${ggml_openmp_llvm_root}/lib/libomp.lib" )
    set( ggml_openmp_runtime "${ggml_openmp_llvm_root}/bin/libomp.dll" )

    if (NOT EXISTS "${ggml_openmp_library}" OR NOT EXISTS "${ggml_openmp_runtime}")
        message( FATAL_ERROR "GGML_OPENMP_LLVM_ROOT must contain lib/libomp.lib and bin/libomp.dll" )
    endif()

    set( OpenMP_libomp_LIBRARY "${ggml_openmp_library}" CACHE FILEPATH "LLVM OpenMP import library" FORCE )
    message( STATUS "Using LLVM OpenMP runtime from ${ggml_openmp_llvm_root}" )
endif()
