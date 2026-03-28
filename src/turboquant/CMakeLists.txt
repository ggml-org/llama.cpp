# TurboQuant KV Cache Quantization - Core Implementation

add_library(turboquant
  turboquant_impl.h
  turboquant_impl.cpp
  turboquant_simd.h
  turboquant_simd.cpp
  turboquant_op.h
  turboquant_op.cpp
  turboquant_op_data.h
  turboquant_constants.h
  turboquant_cuda_streams.h
  turboquant_cuda_streams.cpp
)

target_include_directories(turboquant PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src
  ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/ggml-cpu
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

if (GGML_CUDA)
  # Add CUDA implementation
  enable_language(CUDA)
  
  # Add CUDA files to the library
  target_sources(turboquant PRIVATE turboquant_cuda.cu)
  
  # Set CUDA properties and architecture
  set_target_properties(turboquant PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "89"
  )
  
  # Add CUDA include directories
  target_include_directories(turboquant PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )
  
  message(STATUS "TurboQuant: Building with CUDA support (architecture 89)")
elseif (GGML_HIP)
  # Add AMD HIP implementation
  enable_language(HIP)
  
  # Add HIP files to the library
  target_sources(turboquant PRIVATE turboquant_hip.cpp)
  
  # Set HIP properties
  set_target_properties(turboquant PROPERTIES
    HIP_SEPARABLE_COMPILATION ON
    HIP_RESOLVE_DEVICE_SYMBOLS ON
  )
  
  # Add HIP include directories
  target_include_directories(turboquant PRIVATE
    ${CMAKE_HIP_INCLUDE_DIRS}
  )
  
  message(STATUS "TurboQuant: Building with HIP/AMD support")
else()
  message(STATUS "TurboQuant: Building CPU-only version (CUDA/HIP not available)")
endif()

# SIMD optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-mavx2 -mfma)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(NOT MSVC)
      add_compile_options(-mavx2 -mfma)
    endif()
  endif()
endif()
