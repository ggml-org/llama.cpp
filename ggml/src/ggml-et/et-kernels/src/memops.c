//******************************************************************************
// Memory Operations Kernel
// Handles memset operations on device memory
//******************************************************************************

#include <stdint.h>
#include <stddef.h>
#include "platform.h"

#define CACHE_LINE_SIZE_BYTES 64

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

static void* vectorized_memset(void* dest, int val, size_t n) {
    char* d = (char*)dest;
    size_t i = 0;
    uint64_t temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask


    // Replicate the 8-bit value across a 32-bit word
    uint32_t v = (uint8_t)val;
    uint32_t fill_word = v | (v << 8) | (v << 16) | (v << 24);

    if (n >= 32) {
        // Set the mask register m0 to 0xFF (all 8 lanes active)
        __asm__ volatile ("mov.m.x m0, zero, 0xFF");

        // Broadcast the 32-bit fill_word to all 8 lanes of the 256-bit register f0
        __asm__ volatile (
            "fbcx.ps f0, %0\n\t"
            :
            : "r"(fill_word)
            : "f0"
        );

        // unrolled by 4 to write 128 bytes per iteration.
        // FSQ2 ignores m0 and writes the full 256 bits unconditionally.
        if (n >= 128) {
            for (; i <= n - 128; i += 128) {
                __asm__ volatile (
                    "fsq2    f0, 0(%0)\n\t"
                    "fsq2    f0, 32(%0)\n\t"
                    "fsq2    f0, 64(%0)\n\t"
                    "fsq2    f0, 96(%0)\n\t"
                    :
                    : "r"(d + i)
                    : "memory"
                );
            }
        }

        // Process any remaining full 32-byte chunks
        for (; i <= n - 32; i += 32) {
            __asm__ volatile (
                "fsq2    f0, 0(%0)\n\t"
                :
                : "r"(d + i)
                : "memory"
            );
        }
    }
    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));

    // Handle the unaligned tail (0 to 31 bytes).
    // As with memcpy, a scalar loop is used to avoid overwriting or accessing
    // out-of-bounds bytes that aren't perfectly aligned to 32-bit boundaries.
    for (; i < n; ++i) {
        d[i] = (char)val;
    }

    return dest;
}


// Main entry point for memory operations kernel
int entry_point(struct memset_params* params, void* env) {
    kernel_environment_t* kernel_env = (kernel_environment_t*)env;

    if (!kernel_env) {
        return -1;
    }

    // Get thread info - we only use thread 0 for memops
    int thread_id = get_relative_thread_id(kernel_env->shire_mask);
    int num_threads = get_num_threads(kernel_env->shire_mask);

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
    // XXX: Does this cause UB or us writing to someone else's address?
    while (size > 0 && ((uint64_t)dst & 0x7) != 0) {
        *dst++ = value;
        size--;
    }

    // Split work across threads. Except for the last thread, each thread's
    // write range starts and ends on a CACHE_LINE_SIZE_BYTES boundary.
    uint64_t thread_work_size = 0;
    uint64_t start_offset = 0;
    uint64_t end_offset = 0;

    uint64_t total_size = (uint64_t)size;
    uint64_t aligned_total = total_size & ~((uint64_t)(CACHE_LINE_SIZE_BYTES - 1));

    uint64_t chunks = aligned_total / CACHE_LINE_SIZE_BYTES;
    uint64_t worker_count = (uint64_t)(num_threads - 1);
    uint64_t chunks_per_thread = chunks / worker_count;
    uint64_t extra_chunks = chunks % worker_count;

    uint64_t tid = (uint64_t)thread_id;
    uint64_t my_chunks = chunks_per_thread + (unsigned)(tid < extra_chunks);
    uint64_t prior_chunks =
        chunks_per_thread * tid +
        (tid < extra_chunks ? tid : extra_chunks);

    start_offset = prior_chunks * CACHE_LINE_SIZE_BYTES;
    end_offset = start_offset + my_chunks * CACHE_LINE_SIZE_BYTES;

    // Last thread handles any remaining aligned chunks and all tail bytes.
    if (thread_id == num_threads - 1) {
        end_offset = total_size;
    }

    thread_work_size = end_offset - start_offset;

    if (thread_work_size > 0) {
        vectorized_memset(dst + start_offset, value, (size_t)thread_work_size);
    }

    return 0;
}
