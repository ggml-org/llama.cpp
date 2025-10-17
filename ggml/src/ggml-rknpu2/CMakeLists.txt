# Finding OpenMP
find_package(OpenMP REQUIRED)

# Adding backend files
ggml_add_backend_library(ggml-rknpu2
    ggml-rknpu2.cpp
    dma-alloc.cpp
)

# Linking rknpu2 header files
target_include_directories(ggml-rknpu2
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/libs/include
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Linking rknpu2 library
target_link_libraries(ggml-rknpu2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/librknnrt.so
)

# Adding OpenMP in build
if(OpenMP_FOUND)
    add_definitions(-DHAVE_OPENMP=1)
    list(APPEND CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS}")
endif()