#pragma once

#include <cstddef>
#include <cstdint>

namespace ggml::cpu::riscv64_spacemit {

enum class spine_mem_pool_backend : uint8_t {
    none,
    posix_memalign,
    transparent_hugepage,
    hugetlb_1g,
};

struct spine_mem_pool_tcm_info {
    bool   available{ false };
    size_t blk_size{ 0 };
    size_t blk_num{ 0 };
    bool   is_fake_tcm{ false };
};

bool   spine_mem_pool_tcm_init(spine_mem_pool_tcm_info * info) noexcept;
void * spine_mem_pool_tcm_mem_get(int cpu_id) noexcept;
void * spine_mem_pool_tcm_mem_wait(int cpu_id) noexcept;
int    spine_mem_pool_tcm_mem_release(int cpu_id) noexcept;

void * spine_mem_pool_alloc(size_t size, size_t alignment) noexcept;
void * spine_mem_pool_shared_mem_alloc(size_t size, size_t alignment) noexcept;
void   spine_mem_pool_free(void * base) noexcept;
void   spine_mem_pool_shared_mem_free(void * base) noexcept;

}  // namespace ggml::cpu::riscv64_spacemit
