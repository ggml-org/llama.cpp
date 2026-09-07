#include "ggml-backend-memory.h"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <limits>
#include <new>
#include <vector>

struct ggml_backend_memory_planner {
    size_t capacity;
    size_t alignment;
    uint64_t generation;
    bool building;
    std::vector<ggml_backend_memory_region> committed;
    std::vector<ggml_backend_memory_region> staged;
};

struct ggml_backend_memory_arena_view {
    ggml_backend_memory_region region;
    ggml_backend_buffer_t buffer;
};

struct ggml_backend_memory_arena {
    ggml_backend_buffer_t parent;
    ggml_backend_memory_planner_t planner;
    std::vector<ggml_backend_memory_arena_view> views;
};

static bool is_power_of_two(size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

static bool valid_region_flags(uint32_t flags) {
    return (flags & ~GGML_BACKEND_MEMORY_REGION_PERSISTENT) == 0;
}

static bool valid_plan_flags(uint32_t flags) {
    return (flags & ~GGML_BACKEND_MEMORY_PLAN_PRESERVE_PERSISTENT) == 0;
}

static bool effective_alignment(
        const ggml_backend_memory_planner * planner, size_t requested, size_t * alignment) {
    if (requested != 0 && !is_power_of_two(requested)) {
        return false;
    }
    *alignment = std::max(planner->alignment, requested);
    return true;
}

static bool align_up(size_t value, size_t alignment, size_t * result) {
    const size_t mask = alignment - 1;
    if (value > std::numeric_limits<size_t>::max() - mask) {
        return false;
    }
    *result = (value + mask) & ~mask;
    return true;
}

static bool has_region_id(
        const std::vector<ggml_backend_memory_region> & regions, uint64_t id) {
    return std::any_of(regions.begin(), regions.end(), [id](const ggml_backend_memory_region & region) {
        return region.id == id;
    });
}

static bool same_region(
        const ggml_backend_memory_region & lhs, const ggml_backend_memory_region & rhs) {
    return lhs.id == rhs.id && lhs.offset == rhs.offset && lhs.size == rhs.size &&
           lhs.alignment == rhs.alignment && lhs.flags == rhs.flags;
}

static ggml_backend_memory_arena_view * find_arena_view(
        ggml_backend_memory_arena * arena, uint64_t id) {
    const auto found = std::find_if(
        arena->views.begin(), arena->views.end(),
        [id](const ggml_backend_memory_arena_view & view) {
            return view.region.id == id;
        });
    return found != arena->views.end() ? &*found : nullptr;
}

static void free_arena_views(std::vector<ggml_backend_memory_arena_view> & views) {
    for (const auto & view : views) {
        ggml_backend_buffer_free(view.buffer);
    }
    views.clear();
}

static bool insert_region(
        ggml_backend_memory_planner * planner,
        uint64_t id,
        size_t offset,
        size_t size,
        size_t alignment,
        uint32_t flags,
        ggml_backend_memory_region * result) {
    if (!planner->building || size == 0 || !valid_region_flags(flags) ||
            has_region_id(planner->staged, id)) {
        return false;
    }

    size_t required_alignment = 0;
    if (!effective_alignment(planner, alignment, &required_alignment) ||
            offset % required_alignment != 0 ||
            offset > planner->capacity ||
            size > planner->capacity - offset) {
        return false;
    }

    const size_t end = offset + size;
    const auto position = std::lower_bound(
        planner->staged.begin(), planner->staged.end(), offset,
        [](const ggml_backend_memory_region & region, size_t value) {
            return region.offset < value;
        });

    if (position != planner->staged.end() && end > position->offset) {
        return false;
    }
    if (position != planner->staged.begin()) {
        const auto & previous = *(position - 1);
        if (previous.offset + previous.size > offset) {
            return false;
        }
    }

    const ggml_backend_memory_region region = {
        id,
        offset,
        size,
        required_alignment,
        flags,
    };
    try {
        planner->staged.insert(position, region);
    } catch (const std::bad_alloc &) {
        return false;
    }

    if (result != nullptr) {
        *result = region;
    }
    return true;
}

ggml_backend_memory_planner_t ggml_backend_memory_planner_new(
        size_t capacity, size_t alignment) {
    if (capacity == 0 || !is_power_of_two(alignment)) {
        return nullptr;
    }

    return new (std::nothrow) ggml_backend_memory_planner {
        capacity,
        alignment,
        0,
        false,
        {},
        {},
    };
}

void ggml_backend_memory_planner_free(ggml_backend_memory_planner_t planner) {
    delete planner;
}

bool ggml_backend_memory_planner_begin(
        ggml_backend_memory_planner_t planner, uint32_t flags) {
    if (planner == nullptr || planner->building || !valid_plan_flags(flags)) {
        return false;
    }

    std::vector<ggml_backend_memory_region> staged;
    if (flags & GGML_BACKEND_MEMORY_PLAN_PRESERVE_PERSISTENT) {
        try {
            staged.reserve(planner->committed.size());
            for (const auto & region : planner->committed) {
                if (region.flags & GGML_BACKEND_MEMORY_REGION_PERSISTENT) {
                    staged.push_back(region);
                }
            }
        } catch (const std::bad_alloc &) {
            return false;
        }
    }

    planner->staged.swap(staged);
    planner->building = true;
    return true;
}

bool ggml_backend_memory_planner_reserve(
        ggml_backend_memory_planner_t planner,
        uint64_t id,
        size_t size,
        size_t alignment,
        uint32_t flags,
        struct ggml_backend_memory_region * region) {
    if (planner == nullptr || !planner->building || size == 0 ||
            !valid_region_flags(flags) || has_region_id(planner->staged, id)) {
        return false;
    }

    size_t required_alignment = 0;
    if (!effective_alignment(planner, alignment, &required_alignment)) {
        return false;
    }

    size_t cursor = 0;
    for (const auto & existing : planner->staged) {
        size_t candidate = 0;
        if (align_up(cursor, required_alignment, &candidate) &&
                candidate <= existing.offset &&
                size <= existing.offset - candidate) {
            return insert_region(planner, id, candidate, size, required_alignment, flags, region);
        }
        cursor = existing.offset + existing.size;
    }

    size_t candidate = 0;
    if (!align_up(cursor, required_alignment, &candidate) ||
            candidate > planner->capacity ||
            size > planner->capacity - candidate) {
        return false;
    }
    return insert_region(planner, id, candidate, size, required_alignment, flags, region);
}

bool ggml_backend_memory_planner_reserve_at(
        ggml_backend_memory_planner_t planner,
        uint64_t id,
        size_t offset,
        size_t size,
        size_t alignment,
        uint32_t flags,
        struct ggml_backend_memory_region * region) {
    if (planner == nullptr) {
        return false;
    }
    return insert_region(planner, id, offset, size, alignment, flags, region);
}

bool ggml_backend_memory_planner_commit(ggml_backend_memory_planner_t planner) {
    if (planner == nullptr || !planner->building ||
            planner->generation == std::numeric_limits<uint64_t>::max()) {
        return false;
    }

    planner->committed.swap(planner->staged);
    planner->staged.clear();
    planner->building = false;
    planner->generation++;
    return true;
}

void ggml_backend_memory_planner_rollback(ggml_backend_memory_planner_t planner) {
    if (planner == nullptr) {
        return;
    }
    planner->staged.clear();
    planner->building = false;
}

bool ggml_backend_memory_planner_get_region(
        ggml_backend_memory_planner_t planner,
        uint64_t id,
        struct ggml_backend_memory_region * region) {
    if (planner == nullptr || region == nullptr) {
        return false;
    }

    const auto found = std::find_if(
        planner->committed.begin(), planner->committed.end(),
        [id](const ggml_backend_memory_region & candidate) {
            return candidate.id == id;
        });
    if (found == planner->committed.end()) {
        return false;
    }
    *region = *found;
    return true;
}

bool ggml_backend_memory_planner_get_region_at(
        ggml_backend_memory_planner_t planner,
        size_t index,
        struct ggml_backend_memory_region * region) {
    if (planner == nullptr || region == nullptr || index >= planner->committed.size()) {
        return false;
    }
    *region = planner->committed[index];
    return true;
}

size_t ggml_backend_memory_planner_capacity(ggml_backend_memory_planner_t planner) {
    return planner != nullptr ? planner->capacity : 0;
}

size_t ggml_backend_memory_planner_alignment(ggml_backend_memory_planner_t planner) {
    return planner != nullptr ? planner->alignment : 0;
}

size_t ggml_backend_memory_planner_region_count(ggml_backend_memory_planner_t planner) {
    return planner != nullptr ? planner->committed.size() : 0;
}

size_t ggml_backend_memory_planner_used(ggml_backend_memory_planner_t planner) {
    if (planner == nullptr) {
        return 0;
    }

    size_t used = 0;
    for (const auto & region : planner->committed) {
        used += region.size;
    }
    return used;
}

size_t ggml_backend_memory_planner_high_water(ggml_backend_memory_planner_t planner) {
    if (planner == nullptr || planner->committed.empty()) {
        return 0;
    }

    const auto & region = planner->committed.back();
    return region.offset + region.size;
}

uint64_t ggml_backend_memory_planner_generation(ggml_backend_memory_planner_t planner) {
    return planner != nullptr ? planner->generation : 0;
}

ggml_backend_memory_arena_t ggml_backend_memory_arena_new(
        ggml_backend_buffer_type_t buft, size_t capacity) {
    if (buft == nullptr || capacity == 0) {
        return nullptr;
    }
    ggml_backend_buffer_t parent = ggml_backend_buft_alloc_buffer(buft, capacity);
    if (parent == nullptr) {
        return nullptr;
    }
    ggml_backend_memory_arena_t arena = ggml_backend_memory_arena_new_from_buffer(parent);
    ggml_backend_buffer_free(parent);
    return arena;
}

ggml_backend_memory_arena_t ggml_backend_memory_arena_new_from_buffer(
        ggml_backend_buffer_t buffer) {
    if (buffer == nullptr || buffer->view_buffer == nullptr ||
            ggml_backend_buffer_get_size(buffer) == 0) {
        return nullptr;
    }

    ggml_backend_memory_planner_t planner = ggml_backend_memory_planner_new(
        ggml_backend_buffer_get_size(buffer), ggml_backend_buffer_get_alignment(buffer));
    if (planner == nullptr) {
        return nullptr;
    }

    auto * arena = new (std::nothrow) ggml_backend_memory_arena {
        ggml_backend_buffer_retain(buffer),
        planner,
        {},
    };
    if (arena == nullptr) {
        ggml_backend_memory_planner_free(planner);
        return nullptr;
    }
    return arena;
}

void ggml_backend_memory_arena_free(ggml_backend_memory_arena_t arena) {
    if (arena == nullptr) {
        return;
    }
    free_arena_views(arena->views);
    ggml_backend_memory_planner_free(arena->planner);
    ggml_backend_buffer_free(arena->parent);
    delete arena;
}

bool ggml_backend_memory_arena_begin(
        ggml_backend_memory_arena_t arena, uint32_t flags) {
    return arena != nullptr && ggml_backend_memory_planner_begin(arena->planner, flags);
}

bool ggml_backend_memory_arena_reserve(
        ggml_backend_memory_arena_t arena,
        uint64_t id,
        size_t size,
        size_t alignment,
        uint32_t flags,
        struct ggml_backend_memory_region * region) {
    return arena != nullptr && ggml_backend_memory_planner_reserve(
        arena->planner, id, size, alignment, flags, region);
}

bool ggml_backend_memory_arena_reserve_at(
        ggml_backend_memory_arena_t arena,
        uint64_t id,
        size_t offset,
        size_t size,
        size_t alignment,
        uint32_t flags,
        struct ggml_backend_memory_region * region) {
    return arena != nullptr && ggml_backend_memory_planner_reserve_at(
        arena->planner, id, offset, size, alignment, flags, region);
}

bool ggml_backend_memory_arena_commit(ggml_backend_memory_arena_t arena) {
    if (arena == nullptr || !arena->planner->building) {
        return false;
    }

    std::vector<ggml_backend_memory_arena_view> next_views;
    try {
        next_views.reserve(arena->planner->staged.size());
    } catch (const std::bad_alloc &) {
        ggml_backend_memory_planner_rollback(arena->planner);
        return false;
    }

    for (const auto & region : arena->planner->staged) {
        ggml_backend_buffer_t view = nullptr;
        if (region.flags & GGML_BACKEND_MEMORY_REGION_PERSISTENT) {
            ggml_backend_memory_arena_view * existing = find_arena_view(arena, region.id);
            if (existing != nullptr && same_region(existing->region, region)) {
                view = ggml_backend_buffer_retain(existing->buffer);
            }
        }
        if (view == nullptr) {
            view = ggml_backend_buffer_view(arena->parent, region.offset, region.size);
        }
        if (view == nullptr) {
            free_arena_views(next_views);
            ggml_backend_memory_planner_rollback(arena->planner);
            return false;
        }

        try {
            next_views.push_back({region, view});
        } catch (const std::bad_alloc &) {
            ggml_backend_buffer_free(view);
            free_arena_views(next_views);
            ggml_backend_memory_planner_rollback(arena->planner);
            return false;
        }
    }

    if (!ggml_backend_memory_planner_commit(arena->planner)) {
        free_arena_views(next_views);
        ggml_backend_memory_planner_rollback(arena->planner);
        return false;
    }

    arena->views.swap(next_views);
    free_arena_views(next_views);
    return true;
}

void ggml_backend_memory_arena_rollback(ggml_backend_memory_arena_t arena) {
    if (arena != nullptr) {
        ggml_backend_memory_planner_rollback(arena->planner);
    }
}

ggml_backend_buffer_t ggml_backend_memory_arena_parent(ggml_backend_memory_arena_t arena) {
    return arena != nullptr ? arena->parent : nullptr;
}

ggml_backend_buffer_t ggml_backend_memory_arena_get_buffer(
        ggml_backend_memory_arena_t arena, uint64_t id) {
    if (arena == nullptr) {
        return nullptr;
    }
    ggml_backend_memory_arena_view * view = find_arena_view(arena, id);
    return view != nullptr ? view->buffer : nullptr;
}

bool ggml_backend_memory_arena_get_region(
        ggml_backend_memory_arena_t arena,
        uint64_t id,
        struct ggml_backend_memory_region * region) {
    return arena != nullptr && ggml_backend_memory_planner_get_region(arena->planner, id, region);
}

size_t ggml_backend_memory_arena_capacity(ggml_backend_memory_arena_t arena) {
    return arena != nullptr ? ggml_backend_memory_planner_capacity(arena->planner) : 0;
}

size_t ggml_backend_memory_arena_region_count(ggml_backend_memory_arena_t arena) {
    return arena != nullptr ? ggml_backend_memory_planner_region_count(arena->planner) : 0;
}

size_t ggml_backend_memory_arena_used(ggml_backend_memory_arena_t arena) {
    return arena != nullptr ? ggml_backend_memory_planner_used(arena->planner) : 0;
}

size_t ggml_backend_memory_arena_high_water(ggml_backend_memory_arena_t arena) {
    return arena != nullptr ? ggml_backend_memory_planner_high_water(arena->planner) : 0;
}

uint64_t ggml_backend_memory_arena_generation(ggml_backend_memory_arena_t arena) {
    return arena != nullptr ? ggml_backend_memory_planner_generation(arena->planner) : 0;
}
