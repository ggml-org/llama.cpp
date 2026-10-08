#include "immutable-program-resource.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace ggml::hrx {
namespace {

struct ResourceCandidate {
    const CommandPlanTransient * transient      = nullptr;
    size_t                       matching_count = 0;
};

ResourceCandidate find_resource_candidate(const CommandPlan &   plan,
                                          const DispatchMatch & match,
                                          const std::string &   key) {
    ResourceCandidate result;
    auto              consider = [&](const CommandPlanTransient & transient) {
        if (transient.name == key) {
            result.transient = &transient;
            ++result.matching_count;
        }
    };
    for (const CommandPlanTransient & transient : plan.transients) {
        consider(transient);
    }
    for (const CommandPlanTransient & transient : match.transients) {
        consider(transient);
    }
    return result;
}

const CommandPlanConstantInitialization * find_initialization(const CommandPlan &   plan,
                                                              const DispatchMatch & match,
                                                              ValueId               value,
                                                              size_t &              matching_count) {
    const CommandPlanConstantInitialization * result   = nullptr;
    auto                                      consider = [&](const CommandPlanConstantInitialization & initialization) {
        if (initialization.value == value) {
            result = &initialization;
            ++matching_count;
        }
    };
    for (const CommandPlanConstantInitialization & initialization : plan.constant_initializations) {
        consider(initialization);
    }
    for (const CommandPlanConstantInitialization & initialization : match.constant_initializations) {
        consider(initialization);
    }
    return result;
}

size_t count_initializations_by_name(const CommandPlan & plan, const DispatchMatch & match, const std::string & key) {
    const auto named = [&](const CommandPlanConstantInitialization & initialization) {
        return initialization.name == key;
    };
    return static_cast<size_t>(
               std::count_if(plan.constant_initializations.begin(), plan.constant_initializations.end(), named)) +
           static_cast<size_t>(
               std::count_if(match.constant_initializations.begin(), match.constant_initializations.end(), named));
}

bool valid_alignment(size_t alignment) {
    return alignment != 0 && (alignment & (alignment - 1)) == 0;
}

}  // namespace

ValueId require_immutable_program_resource(const DispatchMatchContext &         context,
                                           DispatchMatch &                      match,
                                           const ImmutableProgramResourceSpec & spec) {
    if (spec.key.empty()) {
        match.status.log("immutable program resource key must not be empty");
        return {};
    }
    if (spec.data.empty()) {
        match.status.log("immutable program resource '%s' data must not be empty", spec.key.c_str());
        return {};
    }
    if (!valid_alignment(spec.alignment)) {
        match.status.log("immutable program resource '%s' alignment %zu must be a non-zero power of two",
                         spec.key.c_str(), spec.alignment);
        return {};
    }

    const ResourceCandidate candidate = find_resource_candidate(context.plan, match, spec.key);
    if (candidate.matching_count > 1) {
        match.status.log("immutable program resource key '%s' is not unique", spec.key.c_str());
        return {};
    }
    if (candidate.transient != nullptr) {
        size_t                                    matching_initializations = 0;
        const CommandPlanConstantInitialization * initialization =
            find_initialization(context.plan, match, candidate.transient->value, matching_initializations);
        if (matching_initializations != 1 || count_initializations_by_name(context.plan, match, spec.key) != 1 ||
            initialization == nullptr || initialization->name != spec.key || initialization->offset != 0 ||
            candidate.transient->size != spec.data.size() || candidate.transient->alignment != spec.alignment ||
            initialization->data != spec.data) {
            match.status.log("immutable program resource key '%s' conflicts with an existing resource",
                             spec.key.c_str());
            return {};
        }
        return candidate.transient->value;
    }

    const bool orphaned_initialization =
        std::any_of(context.plan.constant_initializations.begin(), context.plan.constant_initializations.end(),
                    [&](const CommandPlanConstantInitialization & initialization) {
                        return initialization.name == spec.key;
                    }) ||
        std::any_of(
            match.constant_initializations.begin(), match.constant_initializations.end(),
            [&](const CommandPlanConstantInitialization & initialization) { return initialization.name == spec.key; });
    if (orphaned_initialization) {
        match.status.log("immutable program resource key '%s' has an initialization without a transient",
                         spec.key.c_str());
        return {};
    }

    if (match.transients.size() > std::numeric_limits<size_t>::max() - match.completion_counter_requests.size()) {
        match.status.log("immutable program resource '%s' cannot allocate a plan value", spec.key.c_str());
        return {};
    }
    const size_t value_offset = match.transients.size() + match.completion_counter_requests.size();
    if (context.next_plan_value.value < 0 || value_offset > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        context.next_plan_value.value > std::numeric_limits<int32_t>::max() - static_cast<int32_t>(value_offset)) {
        match.status.log("immutable program resource '%s' cannot allocate a plan value", spec.key.c_str());
        return {};
    }

    const ValueId value(context.next_plan_value.value + static_cast<int32_t>(value_offset));
    match.transients.push_back({ value, spec.key, spec.data.size(), spec.alignment });
    match.constant_initializations.push_back({ value, spec.key, 0, spec.data });
    return value;
}

}  // namespace ggml::hrx
