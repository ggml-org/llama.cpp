# Used to cross-compile ggml-cuda for Windows ARM64 on an x64 Windows host.
set( CMAKE_SYSTEM_NAME Windows )
set( CMAKE_SYSTEM_PROCESSOR arm64 )

if ( DEFINED CUDAToolkit_ROOT )
    file( TO_CMAKE_PATH "${CUDAToolkit_ROOT}" CUDA_ROOT )
elseif ( DEFINED ENV{CUDA_PATH} )
    file( TO_CMAKE_PATH "$ENV{CUDA_PATH}" CUDA_ROOT )
else()
    message( FATAL_ERROR "Set CUDAToolkit_ROOT or CUDA_PATH to a Windows CUDA Toolkit with ARM64 target libraries" )
endif()

if ( DEFINED ENV{VCToolsInstallDir} )
    file( TO_CMAKE_PATH "$ENV{VCToolsInstallDir}" MSVC_TOOLS_ROOT )
    set( CMAKE_CUDA_HOST_COMPILER "${MSVC_TOOLS_ROOT}/bin/Hostx64/arm64/cl.exe" CACHE FILEPATH "" )
endif()

set( CMAKE_CUDA_COMPILER "${CUDA_ROOT}/bin/nvcc.exe" CACHE FILEPATH "" )
set( CMAKE_CUDA_FLAGS_INIT "-target-dir=arm64" )

# FindCUDAToolkit selects lib/x64 from the host architecture on Windows.
set( CUDA_CUDART              "${CUDA_ROOT}/lib/arm64/cudart.lib"   CACHE FILEPATH "" )
set( CUDA_cudart_LIBRARY      "${CUDA_ROOT}/lib/arm64/cudart.lib"   CACHE FILEPATH "" )
set( CUDA_cublas_LIBRARY      "${CUDA_ROOT}/lib/arm64/cublas.lib"   CACHE FILEPATH "" )
set( CUDA_cublasLt_LIBRARY    "${CUDA_ROOT}/lib/arm64/cublasLt.lib" CACHE FILEPATH "" )
set( CUDA_cuda_driver_LIBRARY "${CUDA_ROOT}/lib/arm64/cuda.lib"     CACHE FILEPATH "" )
