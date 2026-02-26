//******************************************************************************
// Memory Operations Kernel
// Handles memset operations on device memory
//******************************************************************************

#include <stdint.h>
#include <stddef.h>
#include "platform.h"

// Operation identifiers for memops kernel
enum ggml_et_memop_type {
    GGML_ET_MEMOP_MEMSET = 0,
};

// Memset operation parameters
struct memset_params {
    uint32_t op_type;      // GGML_ET_MEMOP_MEMSET
    uint32_t value;        // Value to set (extended to uint32_t for alignment)
    void* dst_ptr;         // Destination device pointer
    size_t size;           // Number of bytes to set
};

// Main entry point for memory operations kernel
int entry_point(struct memset_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;

    if (!kernel_env) {
        return -1;
    }

    // Get thread info - we only use thread 0 for memops
    int thread_id = get_relative_thread_id(kernel_env->shire_mask);

    if (thread_id != 0) {
        return 0;
    }

    if (params == 0 || ((uint64_t)params & 0x7) != 0) {
        return -1; // Invalid pointer
    }

    if (params->op_type != GGML_ET_MEMOP_MEMSET) {
        return -1; // Unsupported operation
    }

    uint8_t* dst = (uint8_t*)params->dst_ptr;
    uint8_t value = (uint8_t)params->value;
    size_t size = params->size;

    if (!dst || size == 0) {
        return -1; // Invalid parameters
    }

    // Perform memset operation
    // Use optimized 8-byte writes when possible for better performance

    // Handle unaligned start
    while (size > 0 && ((uint64_t)dst & 0x7) != 0) {
        *dst++ = value;
        size--;
    }

    // Create 8-byte pattern from the byte value
    uint64_t pattern = value;
    pattern |= pattern << 8;
    pattern |= pattern << 16;
    pattern |= pattern << 32;

    // Write 8 bytes at a time
    uint64_t* dst64 = (uint64_t*)dst;
    while (size >= 8) {
        *dst64++ = pattern;
        size -= 8;
    }

    // Handle remaining bytes
    dst = (uint8_t*)dst64;
    while (size > 0) {
        *dst++ = value;
        size--;
    }

    return 0;
}
