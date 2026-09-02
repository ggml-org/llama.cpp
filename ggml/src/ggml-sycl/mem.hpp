#ifndef GGML_SYCL_MEM_HPP
#define GGML_SYCL_MEM_HPP

#include <sycl/sycl.hpp>

enum MemoryAPIType {
    MEMORY_API_TYPE_LEVEL_ZERO = 0,
    MEMORY_API_TYPE_SYCL = 1,
};

const char* mem_api_int2str(int mem_api);

bool get_memory_size(sycl::device dev, size_t & free_bytes, size_t & total_bytes,
    MemoryAPIType api_type);

#endif  // GGML_SYCL_MEM_HPP
