#pragma once

#include "ggml-backend.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_backend_memory_planner * ggml_backend_memory_planner_t;
typedef struct ggml_backend_memory_arena * ggml_backend_memory_arena_t;
typedef struct ggml_backend_memory_lease * ggml_backend_memory_lease_t;

enum ggml_backend_memory_region_flags {
    GGML_BACKEND_MEMORY_REGION_NONE       = 0,
    GGML_BACKEND_MEMORY_REGION_PERSISTENT = 1 << 0,
};

enum ggml_backend_memory_plan_flags {
    GGML_BACKEND_MEMORY_PLAN_NONE                = 0,
    GGML_BACKEND_MEMORY_PLAN_PRESERVE_PERSISTENT = 1 << 0,
};

// Lease admission states. A lease must cover all backend work that uses its region.
enum ggml_backend_memory_arena_state {
    GGML_BACKEND_MEMORY_ARENA_STATE_INVALID   = 0,
    GGML_BACKEND_MEMORY_ARENA_STATE_OPEN      = 1,
    GGML_BACKEND_MEMORY_ARENA_STATE_DRAINING  = 2,
    GGML_BACKEND_MEMORY_ARENA_STATE_QUIESCENT = 3,
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

// Allocate an arena parent buffer from a backend buffer type.
GGML_API ggml_backend_memory_arena_t ggml_backend_memory_arena_new(
        ggml_backend_buffer_type_t buft, size_t capacity);

// Create an arena that retains an existing parent buffer.
GGML_API ggml_backend_memory_arena_t ggml_backend_memory_arena_new_from_buffer(
        ggml_backend_buffer_t buffer);

// Retain an arena while a lease or another owner uses it.
GGML_API ggml_backend_memory_arena_t ggml_backend_memory_arena_retain(
        ggml_backend_memory_arena_t arena);

// Release an arena reference and destroy its storage after the final release.
GGML_API void ggml_backend_memory_arena_free(
        ggml_backend_memory_arena_t arena);

GGML_API bool ggml_backend_memory_arena_begin(
        ggml_backend_memory_arena_t arena, uint32_t flags);

GGML_API bool ggml_backend_memory_arena_reserve(
        ggml_backend_memory_arena_t arena,
        uint64_t id,
        size_t size,
        size_t alignment,
        uint32_t flags,
        struct ggml_backend_memory_region * region);

GGML_API bool ggml_backend_memory_arena_reserve_at(
        ggml_backend_memory_arena_t arena,
        uint64_t id,
        size_t offset,
        size_t size,
        size_t alignment,
        uint32_t flags,
        struct ggml_backend_memory_region * region);

// Materialize all staged regions as views before committing the layout.
GGML_API bool ggml_backend_memory_arena_commit(
        ggml_backend_memory_arena_t arena);

GGML_API void ggml_backend_memory_arena_rollback(
        ggml_backend_memory_arena_t arena);

// Return a borrowed parent buffer valid for the arena lifetime.
GGML_API ggml_backend_buffer_t ggml_backend_memory_arena_parent(
        ggml_backend_memory_arena_t arena);

// Return a borrowed region view valid until the next successful commit.
GGML_API ggml_backend_buffer_t ggml_backend_memory_arena_get_buffer(
        ggml_backend_memory_arena_t arena, uint64_t id);

GGML_API bool ggml_backend_memory_arena_get_region(
        ggml_backend_memory_arena_t arena,
        uint64_t id,
        struct ggml_backend_memory_region * region);

GGML_API size_t ggml_backend_memory_arena_capacity(
        ggml_backend_memory_arena_t arena);

GGML_API size_t ggml_backend_memory_arena_region_count(
        ggml_backend_memory_arena_t arena);

GGML_API size_t ggml_backend_memory_arena_used(
        ggml_backend_memory_arena_t arena);

GGML_API size_t ggml_backend_memory_arena_high_water(
        ggml_backend_memory_arena_t arena);

GGML_API uint64_t ggml_backend_memory_arena_generation(
        ggml_backend_memory_arena_t arena);

// Stop new leases while existing leases drain.
GGML_API bool ggml_backend_memory_arena_quiesce(
        ggml_backend_memory_arena_t arena);

// Accept new leases after the staged layout is committed or rolled back.
GGML_API bool ggml_backend_memory_arena_resume(
        ggml_backend_memory_arena_t arena);

// Return the current lease-gate state.
GGML_API enum ggml_backend_memory_arena_state ggml_backend_memory_arena_get_state(
        ggml_backend_memory_arena_t arena);

// Return the number of live lease objects across all committed regions.
GGML_API size_t ggml_backend_memory_arena_lease_count(
        ggml_backend_memory_arena_t arena);

// Acquire a retained lease for one committed arena region while the gate is open.
GGML_API ggml_backend_memory_lease_t ggml_backend_memory_arena_acquire(
        ggml_backend_memory_arena_t arena, uint64_t id);

// Attach and retain a lease as the complete workspace for one scheduler backend.
// The caller may release its lease handle after this succeeds.
GGML_API bool ggml_backend_sched_attach_memory_lease(
        ggml_backend_sched_t sched,
        ggml_backend_t backend,
        ggml_backend_memory_lease_t lease);

// Synchronize the scheduler, detach its workspace, and release its lease.
// Existing graph tensor addresses become invalid.
GGML_API bool ggml_backend_sched_detach_memory_lease(
        ggml_backend_sched_t sched, ggml_backend_t backend);

// Retain a lease handle without acquiring another region lease.
GGML_API ggml_backend_memory_lease_t ggml_backend_memory_lease_retain(
        ggml_backend_memory_lease_t lease);

// Release a lease handle and end the region lease after the final release.
GGML_API void ggml_backend_memory_lease_free(
        ggml_backend_memory_lease_t lease);

// Return a borrowed buffer valid for the lease lifetime.
GGML_API ggml_backend_buffer_t ggml_backend_memory_lease_buffer(
        ggml_backend_memory_lease_t lease);

// Return the region metadata captured when the lease was acquired.
GGML_API bool ggml_backend_memory_lease_get_region(
        ggml_backend_memory_lease_t lease,
        struct ggml_backend_memory_region * region);

// Return the committed arena generation in which the lease was acquired.
GGML_API uint64_t ggml_backend_memory_lease_generation(
        ggml_backend_memory_lease_t lease);

#ifdef __cplusplus
}
#endif
