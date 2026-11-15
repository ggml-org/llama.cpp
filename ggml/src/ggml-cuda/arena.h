#pragma once

#include <stddef.h>
#include <stdint.h>

struct ggml_cuda_arena;

struct ggml_cuda_arena_copy {
    size_t src, dst, src_pitch, dst_pitch, width, rows, scratch;
};

struct ggml_cuda_arena_span {
    size_t offset, size;
};

// Private backend extension. Fallible operations return a static error string or nullptr.
struct ggml_cuda_arena_api {
    const char * (*create)(int device, size_t reserved, ggml_cuda_arena ** arena, uintptr_t * base);
    void (*destroy)(ggml_cuda_arena * arena);
    const char * (*grow)(ggml_cuda_arena * arena, size_t end, size_t * mapped);
    const char * (*host_alloc)(void ** ptr, size_t size);
    void (*host_free)(void * ptr);
    const char * (*snapshot)(ggml_cuda_arena * arena, void * dst, size_t size);
    const char * (*restore)(ggml_cuda_arena * arena, size_t scratch_offset,
            const ggml_cuda_arena_copy * local, size_t n_local, const void * uploads,
            const ggml_cuda_arena_copy * upload, size_t n_upload, const ggml_cuda_arena_span * gaps, size_t n_gaps);
};

const ggml_cuda_arena_api * ggml_backend_cuda_get_arena_api();
