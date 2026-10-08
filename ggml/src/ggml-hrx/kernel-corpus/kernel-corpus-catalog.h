#pragma once

#include <cstddef>
#include <cstdint>

namespace ggml::hrx {

static constexpr uint64_t kUncatalogedKernelId = 0;

constexpr bool kernel_catalog_name_equal(const char * lhs, const char * rhs) {
    while (*lhs != 0 && *rhs != 0) {
        if (*lhs != *rhs) {
            return false;
        }
        ++lhs;
        ++rhs;
    }
    return *lhs == *rhs;
}

constexpr uint64_t kernel_catalog_id(const char * family, const char * name) {
    uint64_t hash = UINT64_C(1469598103934665603);
    while (*family != 0) {
        hash ^= static_cast<unsigned char>(*family);
        hash *= UINT64_C(1099511628211);
        ++family;
    }
    hash ^= 0;
    hash *= UINT64_C(1099511628211);
    while (*name != 0) {
        hash ^= static_cast<unsigned char>(*name);
        hash *= UINT64_C(1099511628211);
        ++name;
    }
    return hash;
}

struct KernelCatalogRef {
    const char * family = "";
    const char * name   = "";
    uint64_t     id     = kUncatalogedKernelId;

    constexpr bool valid() const {
        return id != kUncatalogedKernelId && family != nullptr && family[0] != 0 && name != nullptr && name[0] != 0;
    }
};

constexpr KernelCatalogRef kernel_catalog_ref(const char * family, const char * name) {
    return {
        family,
        name,
        family != nullptr && name != nullptr ? kernel_catalog_id(family, name) : kUncatalogedKernelId,
    };
}

}  // namespace ggml::hrx
