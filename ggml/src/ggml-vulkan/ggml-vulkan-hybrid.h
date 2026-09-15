#pragma once

#include <cstddef>
#include <cstdint>

struct ggml_tensor;

bool ggml_vk_hybrid_supported(const ggml_tensor * node);
bool ggml_vk_hybrid_try(void * backend_ctx, ggml_tensor * node);
bool ggml_vk_hybrid_read_tensors(void * backend_ctx, const ggml_tensor * const * tensors, void * const * data, const size_t * sizes, int count);

struct ggml_vk_hybrid_external_span {
    uint64_t allocation_id;
    void * handle;
    size_t allocation_size;
    size_t offset;
    size_t size;
};

bool ggml_vk_hybrid_get_external_span(const ggml_tensor * tensor, size_t offset, size_t size, ggml_vk_hybrid_external_span * span);
