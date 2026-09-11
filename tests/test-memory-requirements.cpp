#include "../src/llama-memory-requirements.h"
#include "testing.h"

#include <limits>

using status = llama_memory_requirements_status;

// Keep the test contract independent of device discovery and backend allocation.
struct fixture {
    std::vector<llama_memory_domain> domains = {
        {1, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL | LLAMA_MEMORY_ALLOCATION_MANAGED,
            LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS | LLAMA_MEMORY_CAPABILITY_EXECUTABLE_INVALIDATION},
        {2, LLAMA_MEMORY_ALLOCATION_HOST | LLAMA_MEMORY_ALLOCATION_HOST_PINNED, LLAMA_MEMORY_CAPABILITY_NONE},
    };
    std::vector<llama_memory_resource> resources = {
        {11, 1, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, llama_memory_content::preserve},
        {12, 1, LLAMA_MEMORY_ALLOCATION_MANAGED, llama_memory_content::reconstructible},
        {13, 2, LLAMA_MEMORY_ALLOCATION_HOST, llama_memory_content::discardable},
    };
    std::vector<llama_memory_requirement> requirements = {
        {11, 128, 256, 64, LLAMA_MEMORY_ACCESS_READ_WRITE, LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS},
    };

    // Check structured diagnostics rather than relying on log text.
    void check(testing & t, status expected, uint64_t domain = 0, uint64_t resource = 0) const {
        const auto result = llama_memory_requirements_validate(domains, resources, requirements);
        t.assert_equal(static_cast<int>(expected), static_cast<int>(result.status));
        t.assert_equal(domain, result.domain);
        t.assert_equal(resource, result.resource);
    }
};

int main() {
    testing t;

    t.test("empty_and_valid_contracts", [](testing & t) {
        t.assert_true(llama_memory_requirements_validate({}, {}, {}).status == status::success);
        fixture f;
        f.check(t, status::success);
        f.requirements.clear();
        f.check(t, status::success);
    });

    t.test("missing_and_duplicate_domains", [](testing & t) {
        fixture f;
        f.domains[0].id = 0;
        f.check(t, status::invalid_domain);
        f = {};
        f.domains.push_back(f.domains[0]);
        f.check(t, status::duplicate_domain, 1);
        f = {};
        f.resources[0].domain = 99;
        f.check(t, status::missing_domain, 99, 11);
        f.resources[0].domain = 0;
        f.check(t, status::missing_domain, 0, 11);
    });

    t.test("unknown_domain_flags", [](testing & t) {
        fixture f;
        f.domains[0].allocation_classes |= 1u << 31;
        f.check(t, status::invalid_domain, 1);
        f = {};
        f.domains[0].capabilities |= 1u << 31;
        f.check(t, status::invalid_domain, 1);
    });

    t.test("missing_and_duplicate_resources", [](testing & t) {
        fixture f;
        f.resources[0].id = 0;
        f.check(t, status::invalid_resource, 1);
        f = {};
        f.resources.push_back(f.resources[0]);
        f.check(t, status::duplicate_resource, 1, 11);
        f = {};
        f.requirements[0].resource = 99;
        f.check(t, status::missing_resource, 0, 99);
        f.requirements[0].resource = 0;
        f.check(t, status::missing_resource);
        f = {};
        f.requirements.push_back(f.requirements[0]);
        f.check(t, status::duplicate_requirement, 1, 11);
    });

    t.test("invalid_resource_properties", [](testing & t) {
        for (uint32_t value : {0u, 3u, 1u << 31}) {
            fixture f;
            f.resources[0].allocation_class = static_cast<llama_memory_allocation_class>(value);
            f.check(t, status::invalid_resource, 1, 11);
        }
        fixture f;
        f.resources[0].content = static_cast<llama_memory_content>(99);
        f.check(t, status::invalid_resource, 1, 11);
    });

    t.test("required_and_preferred_sizes", [](testing & t) {
        fixture f;
        f.requirements[0].size_min = 257;
        f.check(t, status::invalid_requirement, 1, 11);
        f.requirements[0].size_min = 256;
        f.check(t, status::success);
        f.requirements[0].size_min = 0;
        f.check(t, status::success);
        f.requirements[0].size_preferred = 0;
        f.check(t, status::success);
    });

    t.test("alignment_and_size_limits", [](testing & t) {
        for (size_t value : {size_t(0), size_t(3), size_t(63)}) {
            fixture f;
            f.requirements[0].alignment = value;
            f.check(t, status::invalid_requirement, 1, 11);
        }
        fixture f;
        auto & req = f.requirements[0];
        req.size_min = req.size_preferred = std::numeric_limits<size_t>::max();
        req.alignment = 1;
        f.check(t, status::success);
        req.alignment = 2;
        f.check(t, status::invalid_requirement, 1, 11);
        req.size_min = req.size_preferred = std::numeric_limits<size_t>::max() - 1;
        f.check(t, status::success);
        req.alignment = size_t(1) << (std::numeric_limits<size_t>::digits - 1);
        req.size_min = req.size_preferred = req.alignment;
        f.check(t, status::success);
        ++req.size_preferred;
        f.check(t, status::invalid_requirement, 1, 11);
    });

    t.test("access_flags", [](testing & t) {
        fixture f;
        for (auto access : {LLAMA_MEMORY_ACCESS_NONE, LLAMA_MEMORY_ACCESS_READ,
                            LLAMA_MEMORY_ACCESS_WRITE, LLAMA_MEMORY_ACCESS_READ_WRITE}) {
            f.requirements[0].access = access;
            f.check(t, status::success);
        }
        f.requirements[0].access = static_cast<llama_memory_access>(4);
        f.check(t, status::invalid_requirement, 1, 11);
        f.requirements[0].access = static_cast<llama_memory_access>(1u << 31);
        f.check(t, status::invalid_requirement, 1, 11);
    });

    t.test("preservation_is_not_stage_access", [](testing & t) {
        fixture f;
        const std::vector<llama_memory_requirement> prefill = {
            {11, 128, 256, 64, LLAMA_MEMORY_ACCESS_READ_WRITE, LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS},
        };
        auto vision = prefill;
        vision[0].access = LLAMA_MEMORY_ACCESS_NONE;
        auto decode = prefill;
        decode[0].access = LLAMA_MEMORY_ACCESS_READ;
        for (auto content : {llama_memory_content::preserve, llama_memory_content::reconstructible}) {
            f.resources[0].content = content;
            for (const auto & requirements : {prefill, vision, decode}) {
                t.assert_true(llama_memory_requirements_validate(f.domains, f.resources, requirements).status == status::success);
            }
            t.assert_true(f.resources[0].content == content);
        }
        t.assert_true(vision[0].access == LLAMA_MEMORY_ACCESS_NONE);
        t.assert_equal(size_t(128), vision[0].size_min);
    });

    t.test("allocation_class_is_exact_not_a_preference", [](testing & t) {
        fixture f;
        f.domains[0].allocation_classes = LLAMA_MEMORY_ALLOCATION_MANAGED;
        f.check(t, status::unsupported_allocation, 1, 11);
        f.resources[0].allocation_class = LLAMA_MEMORY_ALLOCATION_MANAGED;
        f.check(t, status::success);
        f.domains[0].allocation_classes = 0;
        f.check(t, status::unsupported_allocation, 1, 11);
    });

    t.test("capability_diagnostics", [](testing & t) {
        fixture f;
        f.domains[0].capabilities = LLAMA_MEMORY_CAPABILITY_NONE;
        f.check(t, status::unsupported_capability, 1, 11);
        f.requirements[0].capabilities = LLAMA_MEMORY_CAPABILITY_NONE;
        f.check(t, status::success);
        f.requirements[0].capabilities = LLAMA_MEMORY_CAPABILITY_EXECUTABLE_INVALIDATION;
        f.check(t, status::unsupported_capability, 1, 11);
        f.domains[0].capabilities = LLAMA_MEMORY_CAPABILITY_EXECUTABLE_INVALIDATION;
        f.check(t, status::success);
        f.requirements[0].capabilities |= LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS;
        f.check(t, status::unsupported_capability, 1, 11);
        f.requirements[0].capabilities = 1u << 31;
        f.check(t, status::invalid_requirement, 1, 11);
    });

    t.test("capabilities_checked_only_for_live_resources", [](testing & t) {
        fixture f;
        f.resources[2].allocation_class = LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL;
        f.check(t, status::success);
        f.requirements.push_back({13, 0, 0, 1, LLAMA_MEMORY_ACCESS_NONE, LLAMA_MEMORY_CAPABILITY_NONE});
        f.check(t, status::unsupported_allocation, 2, 13);
        f.requirements.back().size_min = 2;
        f.requirements.back().size_preferred = 1;
        f.check(t, status::invalid_requirement, 2, 13);
    });

    t.test("malformed_requests_precede_unsupported_capabilities", [](testing & t) {
        fixture f;
        f.domains[0].capabilities = 0;
        f.requirements.push_back({99, 0, 0, 1, LLAMA_MEMORY_ACCESS_NONE, LLAMA_MEMORY_CAPABILITY_NONE});
        f.check(t, status::missing_resource, 0, 99);
        f.requirements.back().resource = 13;
        f.requirements.back().alignment = 3;
        f.check(t, status::invalid_requirement, 2, 13);
    });

    t.test("placement_domain_identity", [](testing & t) {
        fixture f;
        f.domains.push_back({3, LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, LLAMA_MEMORY_CAPABILITY_NONE});
        f.resources[0].domain = 3;
        f.check(t, status::unsupported_capability, 3, 11);
        f.domains.back().capabilities = LLAMA_MEMORY_CAPABILITY_BUFFER_VIEWS;
        f.check(t, status::success);
    });

    t.test("all_storage_and_content_classes", [](testing & t) {
        fixture f;
        for (auto allocation : {LLAMA_MEMORY_ALLOCATION_HOST, LLAMA_MEMORY_ALLOCATION_HOST_PINNED,
                                LLAMA_MEMORY_ALLOCATION_DEVICE_LOCAL, LLAMA_MEMORY_ALLOCATION_MANAGED}) {
            f.domains[0].allocation_classes = allocation;
            f.resources[0].allocation_class = allocation;
            for (auto content : {llama_memory_content::discardable, llama_memory_content::preserve,
                                 llama_memory_content::reconstructible}) {
                f.resources[0].content = content;
                f.check(t, status::success);
            }
        }
    });

    t.test("validation_is_repeatable_and_does_not_rewrite_inputs", [](testing & t) {
        fixture f;
        for (int i = 0; i < 8; ++i) {
            f.check(t, status::success);
        }
        t.assert_equal(size_t(2), f.domains.size());
        t.assert_equal(size_t(3), f.resources.size());
        t.assert_equal(size_t(1), f.requirements.size());
        t.assert_equal(size_t(128), f.requirements[0].size_min);
        t.assert_equal(size_t(256), f.requirements[0].size_preferred);
        t.assert_equal(size_t(64), f.requirements[0].alignment);
        t.assert_true(f.resources[0].content == llama_memory_content::preserve);
    });

    return t.summary();
}
