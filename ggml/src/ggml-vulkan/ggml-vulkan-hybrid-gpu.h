#pragma once

#include <cstddef>
#include <cstdint>

struct ggml_tensor;

struct ggml_vk_hybrid_gpu_workspace;

struct ggml_vk_hybrid_gpu_planes {
    uint64_t allocation_id = 0;
    void * handle = nullptr;
    size_t allocation_size = 0;

    size_t q_off = 0, k_off = 0, ks_off = 0;
    size_t v_off = 0, vs_off = 0, mask_off = 0, divisor_off = 0, out_off = 0;
    size_t q_size = 0, k_size = 0, ks_size = 0;
    size_t v_size = 0, vs_size = 0, mask_size = 0, divisor_size = 0, out_size = 0;

    void * workspace = nullptr;
};

bool ggml_vk_hybrid_gpu_pack(void * backend_ctx,
                             const ggml_tensor * q, const ggml_tensor * k,
                             const ggml_tensor * v, const ggml_tensor * mask,
                             float divisor,
                             ggml_vk_hybrid_gpu_planes * out);

bool ggml_vk_hybrid_gpu_unpack(void * backend_ctx, ggml_tensor * dst, void * workspace);
void ggml_vk_hybrid_gpu_release(void * workspace);
void ggml_vk_hybrid_gpu_mark_imported(void * backend_ctx);
void ggml_vk_hybrid_gpu_release_arena(void * backend_ctx);
void ggml_vk_hybrid_gpu_release_external_allocation(uint64_t allocation_id);
