#pragma once

#include "ggml.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_backend_memory_planner * ggml_backend_memory_planner_t;

enum ggml_backend_memory_region_flags {
    GGML_BACKEND_MEMORY_REGION_NONE       = 0,
    GGML_BACKEND_MEMORY_REGION_PERSISTENT = 1 << 0,
};

enum ggml_backend_memory_plan_flags {
    GGML_BACKEND_MEMORY_PLAN_NONE                = 0,
    GGML_BACKEND_MEMORY_PLAN_PRESERVE_PERSISTENT = 1 << 0,
};

struct ggml_backend_memory_region {
    uint64_t id;
    size_t offset;
    size_t size;
    size_t alignment;
    uint32_t flags;
};

// Create a transactional layout planner for one aligned address range.
GGML_API ggml_backend_memory_planner_t ggml_backend_memory_planner_new(
        size_t capacity, size_t alignment);

// Destroy the planner without affecting any backend allocation.
GGML_API void ggml_backend_memory_planner_free(
        ggml_backend_memory_planner_t planner);

// Start a staged layout, optionally carrying committed persistent regions forward.
GGML_API bool ggml_backend_memory_planner_begin(
        ggml_backend_memory_planner_t planner, uint32_t flags);

// Reserve the first aligned free interval in the staged layout.
GGML_API bool ggml_backend_memory_planner_reserve(
        ggml_backend_memory_planner_t planner,
        uint64_t id,
        size_t size,
        size_t alignment,
        uint32_t flags,
        struct ggml_backend_memory_region * region);

// Reserve an exact aligned interval in the staged layout.
GGML_API bool ggml_backend_memory_planner_reserve_at(
        ggml_backend_memory_planner_t planner,
        uint64_t id,
        size_t offset,
        size_t size,
        size_t alignment,
        uint32_t flags,
        struct ggml_backend_memory_region * region);

// Atomically replace the committed layout with the staged layout.
GGML_API bool ggml_backend_memory_planner_commit(
        ggml_backend_memory_planner_t planner);

// Discard the staged layout without changing the committed layout.
GGML_API void ggml_backend_memory_planner_rollback(
        ggml_backend_memory_planner_t planner);

// Find a committed region by its caller-defined identifier.
GGML_API bool ggml_backend_memory_planner_get_region(
        ggml_backend_memory_planner_t planner,
        uint64_t id,
        struct ggml_backend_memory_region * region);

// Return a committed region in increasing address order.
GGML_API bool ggml_backend_memory_planner_get_region_at(
        ggml_backend_memory_planner_t planner,
        size_t index,
        struct ggml_backend_memory_region * region);

GGML_API size_t ggml_backend_memory_planner_capacity(
        ggml_backend_memory_planner_t planner);

GGML_API size_t ggml_backend_memory_planner_alignment(
        ggml_backend_memory_planner_t planner);

GGML_API size_t ggml_backend_memory_planner_region_count(
        ggml_backend_memory_planner_t planner);

GGML_API size_t ggml_backend_memory_planner_used(
        ggml_backend_memory_planner_t planner);

GGML_API size_t ggml_backend_memory_planner_high_water(
        ggml_backend_memory_planner_t planner);

GGML_API uint64_t ggml_backend_memory_planner_generation(
        ggml_backend_memory_planner_t planner);

#ifdef __cplusplus
}
#endif
