#include "llama-memory-requirements.h"

#include <limits>

static constexpr uint32_t allocation_classes =
    LLAMA_MEMORY_ALLOCATION_HOST | LLAMA_MEMORY_ALLOCATION_HOST_PINNED |
    LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL | LLAMA_MEMORY_ALLOCATION_MANAGED;
static constexpr uint32_t capabilities =
    LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS | LLAMA_MEMORY_CAPABILITY_EXECUTABLE_INVALIDATION;

// Resolve a session-local domain without backend discovery or allocation.
static const llama_memory_domain * find_domain(
        const std::vector<llama_memory_domain> & domains, llama_memory_domain_id id) {
    for (const auto & domain : domains) {
        if (domain.id == id) {
            return &domain;
        }
    }
    return nullptr;
}

// Resolve resource identity independently of its use in the current stage.
static const llama_memory_resource * find_resource(
        const std::vector<llama_memory_resource> & resources, llama_memory_resource_id id) {
    for (const auto & resource : resources) {
        if (resource.id == id) {
            return &resource;
        }
    }
    return nullptr;
}

// Reject unknown policies rather than treating them as permission to discard data.
static bool valid_content(llama_memory_content content) {
    switch (content) {
        case llama_memory_content::discardable:
        case llama_memory_content::preserve:
        case llama_memory_content::reconstructible:
            return true;
    }
    return false;
}

// Check declarations first so malformed input cannot select an unsupported-path fallback.
llama_memory_requirements_result llama_memory_requirements_validate(
        const std::vector<llama_memory_domain> & domains,
        const std::vector<llama_memory_resource> & resources,
        const std::vector<llama_memory_requirement> & requirements) {
    using status = llama_memory_requirements_status;

    for (size_t i = 0; i < domains.size(); ++i) {
        const auto & domain = domains[i];
        if (domain.id == 0 || (domain.allocation_classes & ~allocation_classes) != 0 ||
                (domain.capabilities & ~capabilities) != 0) {
            return {status::invalid_domain, domain.id, 0};
        }
        for (size_t j = 0; j < i; ++j) {
            if (domains[j].id == domain.id) {
                return {status::duplicate_domain, domain.id, 0};
            }
        }
    }

    for (size_t i = 0; i < resources.size(); ++i) {
        const auto & resource = resources[i];
        const uint32_t allocation = resource.allocation_class;
        if (resource.id == 0 || allocation == 0 || (allocation & (allocation - 1)) != 0 ||
                (allocation & ~allocation_classes) != 0 || !valid_content(resource.content)) {
            return {status::invalid_resource, resource.domain, resource.id};
        }
        for (size_t j = 0; j < i; ++j) {
            if (resources[j].id == resource.id) {
                return {status::duplicate_resource, resource.domain, resource.id};
            }
        }
        if (find_domain(domains, resource.domain) == nullptr) {
            return {status::missing_domain, resource.domain, resource.id};
        }
    }

    for (size_t i = 0; i < requirements.size(); ++i) {
        const auto & requirement = requirements[i];
        const auto * resource = find_resource(resources, requirement.resource);
        if (resource == nullptr) {
            return {status::missing_resource, 0, requirement.resource};
        }
        for (size_t j = 0; j < i; ++j) {
            if (requirements[j].resource == requirement.resource) {
                return {status::duplicate_requirement, resource->domain, resource->id};
            }
        }
        const size_t alignment = requirement.alignment;
        if (requirement.size_min > requirement.size_preferred ||
                alignment == 0 || (alignment & (alignment - 1)) != 0 ||
                requirement.size_preferred > std::numeric_limits<size_t>::max() - (alignment - 1) ||
                (requirement.access & ~uint32_t(LLAMA_MEMORY_ACCESS_READ_WRITE)) != 0 ||
                (requirement.capabilities & ~capabilities) != 0) {
            return {status::invalid_requirement, resource->domain, resource->id};
        }
    }

    for (const auto & requirement : requirements) {
        const auto & resource = *find_resource(resources, requirement.resource);
        const auto & domain = *find_domain(domains, resource.domain);
        if ((domain.allocation_classes & resource.allocation_class) == 0) {
            return {status::unsupported_allocation, domain.id, resource.id};
        }
        if ((requirement.capabilities & ~domain.capabilities) != 0) {
            return {status::unsupported_capability, domain.id, resource.id};
        }
    }

    return {};
}
