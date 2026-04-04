//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_VMEM_KV_HPP
#define GGML_SYCL_VMEM_KV_HPP

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>
#include <sycl/sycl.hpp>

namespace ggml_sycl {

// Level Zero virtual memory manager for KV cache.
//
// Provides a contiguous virtual address range for the entire KV cache,
// with physical pages mapped on demand as context fills.  This eliminates:
//   1. Per-layer KV allocation complexity (one virtual range vs 32+ mallocs)
//   2. Synthetic alloc_base hack (vmem IS the contiguous address space)
//   3. Pointer invalidation on migration (virtual addresses are stable)
//
// Uses dlopen/dlsym for L0 APIs — no compile-time ze_api.h dependency.
// Graceful fallback: if any L0 VM API is unavailable or fails, all methods
// return nullptr/false and the caller falls back to P5 arena allocation.
//
// Page size: 2 MB on BMG (queried at init via zeVirtualMemQueryPageSize).
// Physical pages are backed by device VRAM or host memory (Phase 3).
class vmem_kv_pool {
  public:
    vmem_kv_pool() = default;
    ~vmem_kv_pool();

    // Non-copyable, non-movable (owns L0 resources)
    vmem_kv_pool(const vmem_kv_pool &)             = delete;
    vmem_kv_pool & operator=(const vmem_kv_pool &) = delete;
    vmem_kv_pool(vmem_kv_pool &&)                  = delete;
    vmem_kv_pool & operator=(vmem_kv_pool &&)      = delete;

    // Initialize the pool: resolve L0 APIs, query page size, reserve virtual range.
    // max_kv_bytes: maximum total KV cache size (n_layers * n_ctx * kv_per_token).
    // Returns true if vmem is available and reservation succeeded.
    // Returns false if L0 VM APIs are unavailable or reservation fails.
    bool init(sycl::queue & queue, size_t max_kv_bytes);

    // Map physical pages for a layer's KV region.
    // layer_offset: byte offset from the start of the virtual range.
    // size: bytes to map (will be rounded up to page_size).
    // on_device: must be true — L0 zePhysicalMemCreate requires a device handle
    //            and does not support host-only physical memory.  CPU-layer KV
    //            falls back to P5 arena (sycl::malloc_host).
    // Returns pointer within the virtual range, or nullptr on failure.
    void * map_layer(size_t layer_offset, size_t size, bool on_device);

    // Unmap physical pages for a layer's KV region.
    // Frees the physical memory objects and unmaps the virtual range.
    void unmap_layer(size_t layer_offset, size_t size);

    // Get the base virtual address of the entire KV range.
    // Returns nullptr if not initialized.
    void * base() const { return vaddr_; }

    // Get pointer for a specific layer offset.
    void * get_ptr(size_t offset) const {
        return vaddr_ ? static_cast<char *>(vaddr_) + offset : nullptr;
    }

    // Total reserved virtual range size.
    size_t reserved_size() const { return reserved_size_; }

    // Page size (2 MB on BMG).
    size_t page_size() const { return page_size_; }

    // Number of physical pages currently mapped.
    size_t mapped_page_count() const;

    // Total physical memory currently mapped (mapped pages * page_size).
    size_t mapped_bytes() const;

    // Is the pool initialized and ready?
    bool active() const { return vaddr_ != nullptr; }

    // Destroy the pool: unmap all pages, free physical memory, release virtual range.
    void destroy();

  private:
    // L0 function pointers stored as void * — cast in the .cpp which
    // includes <level_zero/ze_api.h> for actual type definitions.
    void * fn_query_page_size_ = nullptr;
    void * fn_reserve_         = nullptr;
    void * fn_free_            = nullptr;
    void * fn_map_             = nullptr;
    void * fn_unmap_           = nullptr;
    void * fn_phys_create_     = nullptr;
    void * fn_phys_destroy_    = nullptr;

    // L0 handles (extracted from SYCL queue via get_native)
    void * ze_ctx_ = nullptr;
    void * ze_dev_ = nullptr;

    // Virtual memory state
    void * vaddr_         = nullptr;  // Base of reserved virtual range
    size_t reserved_size_ = 0;        // Total reserved bytes
    size_t page_size_     = 0;        // L0 page size (2 MB on BMG)

    // Physical page tracking (handle stored as void *)
    struct physical_page {
        void * handle    = nullptr;  // ze_physical_mem_handle_t
        size_t offset    = 0;        // Offset from vaddr_ where mapped
        bool   on_device = true;
    };
    std::vector<physical_page> pages_;
    mutable std::mutex         mutex_;

    void * dl_handle_ = nullptr;  // dlopen handle for libze_loader.so

    // Resolve all L0 VM function pointers. Returns true if all resolved.
    bool resolve_l0_functions();
};

// Check if L0 virtual memory is available for KV cache on this system.
// Performs a lightweight probe (dlopen + function resolution + page size query).
// Result is cached after first call.
bool vmem_kv_available(sycl::queue & queue);

}  // namespace ggml_sycl

#endif  // GGML_SYCL_VMEM_KV_HPP
