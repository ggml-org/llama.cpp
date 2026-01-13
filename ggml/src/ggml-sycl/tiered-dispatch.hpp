//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_TIERED_DISPATCH_HPP
#define GGML_SYCL_TIERED_DISPATCH_HPP

// This header provides a helper macro for tiered dispatch checks.
// It requires common.hpp to be included first (for GGML_SYCL_DEBUG,
// g_tiered_enabled, get_cached_tensor_ptr, and memory_tier).
//
// Usage in kernel entry points:
//   SYCL_TIERED_DISPATCH_CHECK(src0, "mmq");
//
// This logs cache hits and pending states when tiered mode is active.

#include "tensor-types.hpp"

#include <atomic>

// Forward declarations - these are defined in common.hpp/ggml-sycl.cpp
// but we declare them here to make this header self-contained for IDEs.
extern std::atomic<bool> g_tiered_enabled;
extern void *            get_cached_tensor_ptr(const char *             tensor_name,
                                               ggml_sycl::memory_tier * tier_out,
                                               bool *                   found_in_inventory);

// Macro for tiered dispatch check - use at kernel entry points.
// Includes BOTH hit AND pending logging to match existing kernel patterns.
// The do-while(0) idiom ensures the macro behaves like a single statement.
//
// Parameters:
//   tensor      - pointer to ggml_tensor (typically src0 for weight tensors)
//   kernel_name - string literal for logging (e.g., "mmq", "mmvq", "dmmv")
//
// Note: This macro requires GGML_SYCL_DEBUG to be defined (from common.hpp).
// If GGML_SYCL_DEBUG is not available, include common.hpp before this header.
#define SYCL_TIERED_DISPATCH_CHECK(tensor, kernel_name)                                                           \
    do {                                                                                                          \
        if ((tensor)->name && g_tiered_enabled.load(std::memory_order_acquire)) {                                 \
            ggml_sycl::memory_tier _tier;                                                                         \
            bool                   _in_inventory = false;                                                         \
            void *                 _cached_ptr   = get_cached_tensor_ptr((tensor)->name, &_tier, &_in_inventory); \
            if (_cached_ptr != nullptr) {                                                                         \
                GGML_SYCL_DEBUG("[SYCL] %s tiered hit: %s (tier=%d)\n", kernel_name, (tensor)->name,              \
                                static_cast<int>(_tier));                                                         \
            } else if (_in_inventory) {                                                                           \
                GGML_SYCL_DEBUG("[SYCL] %s tiered pending: %s\n", kernel_name, (tensor)->name);                   \
            }                                                                                                     \
        }                                                                                                         \
    } while (0)

#endif  // GGML_SYCL_TIERED_DISPATCH_HPP
