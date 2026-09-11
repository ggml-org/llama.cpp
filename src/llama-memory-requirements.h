#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// IDs are session-local; zero is reserved for an unspecified resource or domain.
using llama_memory_resource_id = uint64_t;
using llama_memory_domain_id = uint64_t;

enum llama_memory_allocation_class : uint32_t {
    LLAMA_MEMORY_ALLOCATION_HOST         = 1u << 0,
    LLAMA_MEMORY_ALLOCATION_HOST_PINNED  = 1u << 1,
    LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL = 1u << 2,
    LLAMA_MEMORY_ALLOCATION_MANAGED      = 1u << 3,
};

enum llama_memory_capability : uint32_t {
    LLAMA_MEMORY_CAPABILITY_NONE                    = 0,
    LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS            = 1u << 0,
    LLAMA_MEMORY_CAPABILITY_EXECUTABLE_INVALIDATION = 1u << 1,
};

enum llama_memory_access : uint32_t {
    LLAMA_MEMORY_ACCESS_NONE       = 0,
    LLAMA_MEMORY_ACCESS_READ       = 1u << 0,
    LLAMA_MEMORY_ACCESS_WRITE      = 1u << 1,
    LLAMA_MEMORY_ACCESS_READ_WRITE = LLAMA_MEMORY_ACCESS_READ | LLAMA_MEMORY_ACCESS_WRITE,
};

enum class llama_memory_content : uint32_t {
    discardable,
    preserve,
    reconstructible,
};

// A domain identifies one placement target, not a backend name or a GPU ordinal.
// Adapters report verified allocation/execution capabilities; this contract does not probe hardware.
struct llama_memory_domain {
    llama_memory_domain_id id = 0;
    uint32_t allocation_classes = 0;
    uint32_t capabilities = LLAMA_MEMORY_CAPABILITY_NONE;
};

struct llama_memory_resource {
    llama_memory_resource_id id = 0;
    llama_memory_domain_id domain = 0;
    llama_memory_allocation_class allocation_class = LLAMA_MEMORY_ALLOCATION_HOST;
    // Preserve means contents must survive, not that an address or lease must stay fixed.
    // Reconstructible requires an independent source and a consumer that can restore it.
    llama_memory_content content = llama_memory_content::discardable;
};

// Presence means the resource is live in this stage, including when access is NONE.
// Omission alone does not authorize losing preserved contents; whole-plan lifetime checks follow separately.
struct llama_memory_requirement {
    llama_memory_resource_id resource = 0;
    size_t size_min = 0;
    size_t size_preferred = 0;
    size_t alignment = 1;
    llama_memory_access access = LLAMA_MEMORY_ACCESS_NONE;
    uint32_t capabilities = LLAMA_MEMORY_CAPABILITY_NONE;
};

enum class llama_memory_requirements_status {
    success,
    invalid_domain,
    duplicate_domain,
    invalid_resource,
    duplicate_resource,
    missing_domain,
    missing_resource,
    duplicate_requirement,
    invalid_requirement,
    unsupported_allocation,
    unsupported_capability,
};

struct llama_memory_requirements_result {
    llama_memory_requirements_status status = llama_memory_requirements_status::success;
    llama_memory_domain_id domain = 0;
    llama_memory_resource_id resource = 0;
};

// Validate declarations and one stage's requirements without allocation or backend calls.
// Structural errors precede capability checks; capabilities are checked only for requested resources.
// Zero-byte requirements are valid declarations, not zero-byte arena regions. Alignment is a nonzero power of two.
// This does not check dependency ordering, byte preservation, available capacity, or backend capability truthfulness.
llama_memory_requirements_result llama_memory_requirements_validate(
        const std::vector<llama_memory_domain> & domains,
        const std::vector<llama_memory_resource> & resources,
        const std::vector<llama_memory_requirement> & requirements);
