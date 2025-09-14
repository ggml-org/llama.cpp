/**
 * @file ggml-numa-allocator.c
 * @brief Minimal NUMA-Aware Memory Allocator for Mirror Mode
 * 
 * Provides basic NUMA allocation functions for intermediate tensors
 * in NUMA mirror mode only.
 */

#include "ggml-numa-allocator.h"
#include "ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <numa.h>
#include <numaif.h>

// Simple NUMA allocation for intermediate tensors
void* ggml_numa_alloc(size_t size) {
    if (numa_available() < 0) {
        return malloc(size);
    }
    
    // Allocate on current NUMA node
    extern __thread int ggml_current_numa_node;
    int node = ggml_current_numa_node;
    if (node == -1 || node >= numa_num_configured_nodes()) {
        node = 0;
    }
    
    void* ptr = numa_alloc_onnode(size, node);
    return ptr ? ptr : malloc(size);
}

void ggml_numa_free(void* ptr, size_t size) {
    if (ptr) {
        numa_free(ptr, size);
    }
}

// First-touch allocation with SIMD alignment for model weights
void* numa_alloc_mmap_first_touch(size_t size, int node) {
    // Define SIMD alignment
#if defined(__s390x__)
    const size_t alignment = 256;
#else
    const size_t alignment = 64;  // 64-byte alignment for AVX-512
#endif
    
    // Bind current thread to the target NUMA node for first-touch
    struct bitmask* old_mask = numa_get_run_node_mask();
    if (numa_run_on_node(node) != 0) {
        // Continue anyway - might still work
    }
    
    // Use posix_memalign for SIMD alignment
    void* ptr = NULL;
    int ret = posix_memalign(&ptr, alignment, size);
    if (ret != 0) {
        // Restore original thread binding
        if (old_mask) {
            numa_run_on_node_mask(old_mask);
            numa_free_nodemask(old_mask);
        }
        return NULL;
    }
    
    // First-touch: touch every page to ensure physical allocation on current node
    volatile char* mem = (volatile char*)ptr;
    const size_t page_size = sysconf(_SC_PAGESIZE);
    for (size_t i = 0; i < size; i += page_size) {
        mem[i] = 0; // First touch allocates the page on current NUMA node
    }
    
    // Restore original thread binding
    if (old_mask) {
        numa_run_on_node_mask(old_mask);
        numa_free_nodemask(old_mask);
    }
    
    return ptr;
}

void numa_free_mmap_first_touch(void* ptr, size_t size) {
    if (ptr) {
        free(ptr);  // Use free() for posix_memalign() allocated memory
    }
}