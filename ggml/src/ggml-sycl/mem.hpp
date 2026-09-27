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

// Non-aborting device memory query (defined in ggml-sycl.cpp). Returns false
// when neither Level Zero Sysman nor the SYCL free-memory aspect could answer;
// ggml_backend_sycl_get_device_memory() wraps it and GGML_ABORTs instead.
bool sycl_get_mem_info(int device, size_t * free, size_t * total);

#endif  // GGML_SYCL_MEM_HPP
