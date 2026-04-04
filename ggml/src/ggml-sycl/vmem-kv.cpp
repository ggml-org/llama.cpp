//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "vmem-kv.hpp"

#include <cstdlib>
#include <dlfcn.h>

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>

#include "ggml-impl.h"

namespace ggml_sycl {
namespace {

// L0 function pointer typedefs for casting from void *
using fn_QueryPageSize = ze_result_t (*)(ze_context_handle_t, ze_device_handle_t, size_t, size_t *);
using fn_Reserve       = ze_result_t (*)(ze_context_handle_t, const void *, size_t, void **);
using fn_Free          = ze_result_t (*)(ze_context_handle_t, const void *, size_t);
using fn_Map           = ze_result_t (*)(ze_context_handle_t, const void *, size_t,
                                         ze_physical_mem_handle_t, size_t, ze_memory_access_attribute_t);
using fn_Unmap         = ze_result_t (*)(ze_context_handle_t, const void *, size_t);
using fn_PhysCreate    = ze_result_t (*)(ze_context_handle_t, ze_device_handle_t,
                                         ze_physical_mem_desc_t *, ze_physical_mem_handle_t *);
using fn_PhysDestroy   = ze_result_t (*)(ze_context_handle_t, ze_physical_mem_handle_t);

size_t align_to_page(size_t bytes, size_t page_size) {
    return ((bytes + page_size - 1) / page_size) * page_size;
}

}  // namespace

vmem_kv_pool::~vmem_kv_pool() {
    destroy();
}

bool vmem_kv_pool::resolve_l0_functions() {
    // dlopen libze_loader.so (already loaded by SYCL runtime, should be fast)
    dl_handle_ = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (!dl_handle_) {
        dl_handle_ = dlopen("libze_loader.so", RTLD_NOW | RTLD_NOLOAD);
    }
    if (!dl_handle_) {
        // Not already loaded — try loading fresh
        dl_handle_ = dlopen("libze_loader.so.1", RTLD_NOW);
    }
    if (!dl_handle_) {
        GGML_LOG_DEBUG("[VMEM-KV] dlopen(libze_loader) failed: %s\n", dlerror());
        return false;
    }

    fn_query_page_size_ = dlsym(dl_handle_, "zeVirtualMemQueryPageSize");
    fn_reserve_         = dlsym(dl_handle_, "zeVirtualMemReserve");
    fn_free_            = dlsym(dl_handle_, "zeVirtualMemFree");
    fn_map_             = dlsym(dl_handle_, "zeVirtualMemMap");
    fn_unmap_           = dlsym(dl_handle_, "zeVirtualMemUnmap");
    fn_phys_create_     = dlsym(dl_handle_, "zePhysicalMemCreate");
    fn_phys_destroy_    = dlsym(dl_handle_, "zePhysicalMemDestroy");

    if (!fn_query_page_size_ || !fn_reserve_ || !fn_free_ ||
        !fn_map_ || !fn_unmap_ || !fn_phys_create_ || !fn_phys_destroy_) {
        GGML_LOG_DEBUG("[VMEM-KV] L0 virtual memory APIs not fully available\n");
        return false;
    }

    return true;
}

bool vmem_kv_pool::init(sycl::queue & queue, size_t max_kv_bytes) {
    if (vaddr_) {
        return true;  // Already initialized
    }

    if (!resolve_l0_functions()) {
        return false;
    }

    // Extract L0 handles from SYCL queue
    auto sycl_ctx = queue.get_context();
    auto sycl_dev = queue.get_device();

    // Check we're on Level Zero backend
    if (sycl_ctx.get_backend() != sycl::backend::ext_oneapi_level_zero) {
        GGML_LOG_DEBUG("[VMEM-KV] Not on Level Zero backend, vmem unavailable\n");
        return false;
    }

    ze_ctx_ = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_ctx);
    ze_dev_ = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_dev);

    if (!ze_ctx_ || !ze_dev_) {
        GGML_LOG_WARN("[VMEM-KV] Failed to extract L0 handles from SYCL queue\n");
        return false;
    }

    auto qps = reinterpret_cast<fn_QueryPageSize>(fn_query_page_size_);
    auto res = reinterpret_cast<fn_Reserve>(fn_reserve_);

    // Query page size
    auto ze_c = static_cast<ze_context_handle_t>(ze_ctx_);
    auto ze_d = static_cast<ze_device_handle_t>(ze_dev_);

    ze_result_t r = qps(ze_c, ze_d, max_kv_bytes, &page_size_);
    if (r != ZE_RESULT_SUCCESS || page_size_ == 0) {
        GGML_LOG_WARN("[VMEM-KV] zeVirtualMemQueryPageSize failed (result=%u)\n", r);
        return false;
    }

    // Round up reservation to page boundary
    reserved_size_ = align_to_page(max_kv_bytes, page_size_);

    // Reserve contiguous virtual address range (costs zero physical memory)
    r = res(ze_c, nullptr, reserved_size_, &vaddr_);
    if (r != ZE_RESULT_SUCCESS || !vaddr_) {
        GGML_LOG_WARN("[VMEM-KV] zeVirtualMemReserve(%.1f MB) failed (result=%u)\n",
                      reserved_size_ / (1024.0 * 1024.0), r);
        vaddr_         = nullptr;
        reserved_size_ = 0;
        return false;
    }

    GGML_LOG_INFO("[VMEM-KV] Reserved %.1f MB virtual range at %p "
                  "(page_size=%zu KB, max_pages=%zu)\n",
                  reserved_size_ / (1024.0 * 1024.0), vaddr_,
                  page_size_ / 1024, reserved_size_ / page_size_);

    return true;
}

void * vmem_kv_pool::map_layer(size_t layer_offset, size_t size, bool on_device) {
    if (!vaddr_ || !fn_phys_create_ || !fn_map_) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto phys_create = reinterpret_cast<fn_PhysCreate>(fn_phys_create_);
    auto vmap        = reinterpret_cast<fn_Map>(fn_map_);
    auto phys_destroy = reinterpret_cast<fn_PhysDestroy>(fn_phys_destroy_);
    auto ze_c = static_cast<ze_context_handle_t>(ze_ctx_);
    auto ze_d = static_cast<ze_device_handle_t>(ze_dev_);

    // Round offset down and size up to page boundaries
    const size_t page_start  = (layer_offset / page_size_) * page_size_;
    const size_t page_end    = align_to_page(layer_offset + size, page_size_);
    const size_t n_pages     = (page_end - page_start) / page_size_;

    if (page_end > reserved_size_) {
        GGML_LOG_ERROR("[VMEM-KV] map_layer: offset+size (%.1f MB) exceeds reservation (%.1f MB)\n",
                       page_end / (1024.0 * 1024.0), reserved_size_ / (1024.0 * 1024.0));
        return nullptr;
    }

    // Create physical pages and map them
    for (size_t i = 0; i < n_pages; i++) {
        size_t page_offset = page_start + i * page_size_;

        // Check if this page is already mapped
        bool already_mapped = false;
        for (const auto & p : pages_) {
            if (p.offset == page_offset) {
                already_mapped = true;
                break;
            }
        }
        if (already_mapped) {
            continue;
        }

        // Create physical memory
        ze_physical_mem_desc_t pd = {};
        pd.stype = ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC;
        pd.pNext = nullptr;
        pd.flags = 0;
        pd.size  = page_size_;

        ze_physical_mem_handle_t phys = nullptr;
        ze_result_t r = phys_create(ze_c, ze_d, &pd, &phys);
        if (r != ZE_RESULT_SUCCESS || !phys) {
            GGML_LOG_ERROR("[VMEM-KV] zePhysicalMemCreate failed for page at offset %zu (result=%u)\n",
                           page_offset, r);
            return nullptr;
        }

        // Map virtual -> physical
        void * page_addr = static_cast<char *>(vaddr_) + page_offset;
        r = vmap(ze_c, page_addr, page_size_, phys, 0, ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE);
        if (r != ZE_RESULT_SUCCESS) {
            GGML_LOG_ERROR("[VMEM-KV] zeVirtualMemMap failed at offset %zu (result=%u)\n",
                           page_offset, r);
            phys_destroy(ze_c, phys);
            return nullptr;
        }

        pages_.push_back({ phys, page_offset, on_device });
    }

    GGML_LOG_DEBUG("[VMEM-KV] Mapped %zu pages for layer at offset=%zu size=%zu "
                   "(total mapped=%zu pages, %.1f MB)\n",
                   n_pages, layer_offset, size,
                   pages_.size(), pages_.size() * page_size_ / (1024.0 * 1024.0));

    return static_cast<char *>(vaddr_) + layer_offset;
}

void vmem_kv_pool::unmap_layer(size_t layer_offset, size_t size) {
    if (!vaddr_ || !fn_unmap_ || !fn_phys_destroy_) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto vunmap       = reinterpret_cast<fn_Unmap>(fn_unmap_);
    auto phys_destroy = reinterpret_cast<fn_PhysDestroy>(fn_phys_destroy_);
    auto ze_c = static_cast<ze_context_handle_t>(ze_ctx_);

    const size_t page_start = (layer_offset / page_size_) * page_size_;
    const size_t page_end   = align_to_page(layer_offset + size, page_size_);

    // Find and unmap pages in this range
    auto it = pages_.begin();
    while (it != pages_.end()) {
        if (it->offset >= page_start && it->offset < page_end) {
            void * page_addr = static_cast<char *>(vaddr_) + it->offset;
            vunmap(ze_c, page_addr, page_size_);
            phys_destroy(ze_c, static_cast<ze_physical_mem_handle_t>(it->handle));
            it = pages_.erase(it);
        } else {
            ++it;
        }
    }
}

size_t vmem_kv_pool::mapped_page_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pages_.size();
}

size_t vmem_kv_pool::mapped_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pages_.size() * page_size_;
}

void vmem_kv_pool::destroy() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!vaddr_) {
        return;
    }

    auto ze_c = static_cast<ze_context_handle_t>(ze_ctx_);

    // Unmap and destroy all physical pages
    for (auto & p : pages_) {
        void * page_addr = static_cast<char *>(vaddr_) + p.offset;
        if (fn_unmap_) {
            reinterpret_cast<fn_Unmap>(fn_unmap_)(ze_c, page_addr, page_size_);
        }
        if (fn_phys_destroy_) {
            reinterpret_cast<fn_PhysDestroy>(fn_phys_destroy_)(
                ze_c, static_cast<ze_physical_mem_handle_t>(p.handle));
        }
    }
    size_t n_pages = pages_.size();
    pages_.clear();

    // Free virtual reservation
    if (fn_free_) {
        reinterpret_cast<fn_Free>(fn_free_)(ze_c, vaddr_, reserved_size_);
    }

    GGML_LOG_INFO("[VMEM-KV] Destroyed: freed %zu physical pages, released %.1f MB virtual range\n",
                  n_pages, reserved_size_ / (1024.0 * 1024.0));

    vaddr_         = nullptr;
    reserved_size_ = 0;
    page_size_     = 0;
    ze_ctx_        = nullptr;
    ze_dev_        = nullptr;

    if (dl_handle_) {
        // Don't dlclose — the L0 loader is shared with SYCL runtime
        dl_handle_ = nullptr;
    }
}

bool vmem_kv_available(sycl::queue & queue) {
    static std::atomic<int> cached{ -1 };
    int val = cached.load(std::memory_order_acquire);
    if (val >= 0) {
        return val != 0;
    }

    // Check env var gate: GGML_SYCL_VMEM_KV=1 required (opt-in)
    const char * env = std::getenv("GGML_SYCL_VMEM_KV");
    if (!env || std::atoi(env) != 1) {
        GGML_LOG_DEBUG("[VMEM-KV] Not enabled (set GGML_SYCL_VMEM_KV=1 to enable)\n");
        cached.store(0, std::memory_order_release);
        return false;
    }

    // Probe: create a temporary pool, try to reserve 1 page
    vmem_kv_pool probe;
    bool ok = probe.init(queue, 2 * 1024 * 1024);  // 2 MB probe
    probe.destroy();

    cached.store(ok ? 1 : 0, std::memory_order_release);
    if (ok) {
        GGML_LOG_INFO("[VMEM-KV] L0 virtual memory available for KV cache\n");
    } else {
        GGML_LOG_INFO("[VMEM-KV] L0 virtual memory NOT available, using arena fallback\n");
    }
    return ok;
}

}  // namespace ggml_sycl
