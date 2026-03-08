//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_ALLOC_REGISTRY_HPP
#define GGML_SYCL_ALLOC_REGISTRY_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <vector>

namespace ggml_sycl {

// Type of USM allocation, mirrors sycl::usm::alloc but decoupled from runtime
enum class alloc_type : uint8_t {
    DEVICE,       // sycl::malloc_device — GPU-only
    HOST_PINNED,  // sycl::malloc_host   — CPU-accessible, GPU zero-copy via PCIe
    SHARED,       // sycl::malloc_shared  — migrates between host/device
    MMAP,         // mmap'd from GGUF    — CPU-only, no USM
    UNKNOWN,      // Not registered — pointer type is unknown
};

// Metadata for a single allocation
struct alloc_info {
    uintptr_t  base;       // Allocation base address (as integer for fast comparison)
    size_t     size;       // Allocation size in bytes
    int        device_id;  // -1 for host/mmap, 0..N for device allocations
    alloc_type type;       // Allocation type
};

// Central registry mapping pointer ranges to allocation metadata.
//
// Design:
//   - Sorted vector of alloc_info, binary search on base address
//   - Supports interior pointer lookup (pointer within an allocation range)
//   - Thread-safe: readers use shared_lock, writers use unique_lock
//   - Singleton via instance()
//
// Usage pattern:
//   - Registration during model load / buffer init (write-locked)
//   - Lookups during inference (read-locked, concurrent)
//
class alloc_registry {
  public:
    static alloc_registry & instance() {
        static alloc_registry reg;
        return reg;
    }

    // Register a new allocation. Thread-safe.
    void register_alloc(const void * ptr, size_t size, int device_id, alloc_type type) {
        if (ptr == nullptr || size == 0) {
            return;
        }
        uintptr_t                           addr = reinterpret_cast<uintptr_t>(ptr);
        std::unique_lock<std::shared_mutex> lock(mu_);
        // Find insertion point (keep sorted by base address)
        alloc_info                          info{ addr, size, device_id, type };
        auto                                it = std::lower_bound(entries_.begin(), entries_.end(), info,
                                                                  [](const alloc_info & a, const alloc_info & b) { return a.base < b.base; });
        // Check for duplicate base address (re-registration after realloc)
        if (it != entries_.end() && it->base == addr) {
            *it = info;  // Update in place
        } else {
            entries_.insert(it, info);
        }
    }

    // Unregister an allocation by its base pointer. Thread-safe.
    void unregister_alloc(const void * ptr) {
        if (ptr == nullptr) {
            return;
        }
        uintptr_t                           addr = reinterpret_cast<uintptr_t>(ptr);
        std::unique_lock<std::shared_mutex> lock(mu_);
        auto                                it = std::lower_bound(entries_.begin(), entries_.end(), addr,
                                                                  [](const alloc_info & a, uintptr_t val) { return a.base < val; });
        if (it != entries_.end() && it->base == addr) {
            entries_.erase(it);
        }
    }

    // Look up a pointer (may be interior to an allocation). Thread-safe.
    // Returns pointer to alloc_info if found, nullptr otherwise.
    const alloc_info * lookup(const void * ptr) const {
        if (ptr == nullptr) {
            return nullptr;
        }
        uintptr_t                           addr = reinterpret_cast<uintptr_t>(ptr);
        std::shared_lock<std::shared_mutex> lock(mu_);
        // Find the first entry with base > addr, then step back one
        auto                                it = std::upper_bound(entries_.begin(), entries_.end(), addr,
                                                                  [](uintptr_t val, const alloc_info & a) { return val < a.base; });
        if (it == entries_.begin()) {
            return nullptr;  // addr is before all allocations
        }
        --it;
        // Check if addr falls within [base, base + size)
        if (addr >= it->base && addr < it->base + it->size) {
            return &(*it);
        }
        return nullptr;
    }

    // Convenience: is this pointer host-accessible?
    // Returns true for HOST_PINNED, SHARED, MMAP.
    // Returns false for DEVICE.
    // Returns false if pointer is not in the registry (unknown = not host-accessible).
    bool is_host_accessible(const void * ptr) const {
        const alloc_info * info = lookup(ptr);
        if (info == nullptr) {
            return false;  // Unknown pointer — conservative: treat as not host-accessible
        }
        return info->type != alloc_type::DEVICE;
    }

    // Convenience: is this pointer on a device?
    bool is_device(const void * ptr) const {
        const alloc_info * info = lookup(ptr);
        return info != nullptr && info->type == alloc_type::DEVICE;
    }

    // Convenience: get the sycl::usm::alloc equivalent for compatibility with existing code.
    // Note: this header does NOT include sycl.hpp — callers must convert.
    alloc_type get_alloc_type(const void * ptr) const {
        const alloc_info * info = lookup(ptr);
        if (info == nullptr) {
            return alloc_type::UNKNOWN;  // Not registered — caller must handle
        }
        return info->type;
    }

    // Get owning device ID. Returns -1 for host/mmap or unknown pointers.
    int owning_device(const void * ptr) const {
        const alloc_info * info = lookup(ptr);
        return info ? info->device_id : -1;
    }

    // Number of registered allocations (for debugging/testing).
    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mu_);
        return entries_.size();
    }

    // Clear all entries (for testing only).
    void clear() {
        std::unique_lock<std::shared_mutex> lock(mu_);
        entries_.clear();
    }

  private:
    alloc_registry() {
        entries_.reserve(4096);  // Pre-allocate for typical model weight count
    }

    mutable std::shared_mutex mu_;
    std::vector<alloc_info>   entries_;
};

// Convert alloc_type to sycl::usm::alloc (for interop with existing code).
// Must be called from files that include sycl/sycl.hpp.
// Not defined here to keep this header sycl-free.
//
// Usage in .cpp files:
//   sycl::usm::alloc sycl_type = to_sycl_alloc(registry_type);
//
// Inline in caller:
//   switch (type) {
//       case alloc_type::DEVICE:      return sycl::usm::alloc::device;
//       case alloc_type::HOST_PINNED: return sycl::usm::alloc::host;
//       case alloc_type::SHARED:      return sycl::usm::alloc::shared;
//       case alloc_type::MMAP:        return sycl::usm::alloc::host;  // treat as host-accessible
//       case alloc_type::UNKNOWN:     return sycl::usm::alloc::unknown;
//   }

}  // namespace ggml_sycl

#endif  // GGML_SYCL_ALLOC_REGISTRY_HPP
