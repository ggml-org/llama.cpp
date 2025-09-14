/**
 * @file ggml-numa-allocator.h
 * @brief Minimal NUMA-Aware Memory Allocator Header for Mirror Mode
 */

#pragma once

#include <stddef.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif

// Basic NUMA allocation functions
void* ggml_numa_alloc(size_t size);
void ggml_numa_free(void* ptr, size_t size);

// First-touch allocation for model weights
void* numa_alloc_mmap_first_touch(size_t size, int node);
void numa_free_mmap_first_touch(void* ptr, size_t size);

#ifdef __cplusplus
}
#endif