#pragma once

#include "command-plan-metadata.h"
#include "dispatch.h"
#include "graph/graph.h"
#include "status.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ggml::hrx {

struct CommandPlanTransient {
    ValueId     value;
    std::string name;
    size_t      size      = 0;
    size_t      alignment = 256;
};

struct CommandPlanConstantInitialization {
    ValueId              value;
    std::string          name;
    size_t               offset = 0;
    std::vector<uint8_t> data;
};

struct CommandPlanCompletionCounterRequest {
    ValueId     value;
    std::string name;
    uint32_t    count = 0;
};

struct CommandPlan {
    std::vector<Dispatch>                            initialization_dispatches;
    std::vector<Dispatch>                            dispatches;
    std::vector<CommandPlanTransient>                transients;
    std::vector<CommandPlanConstantInitialization>   constant_initializations;
    std::vector<CommandPlanCompletionCounterRequest> completion_counter_requests;
    CommandPlanMetadata                              metadata;
    Status                                           status;

    bool valid() const { return status.success(); }
};

inline const CommandPlanAlternateValue * find_alternate_value(const CommandPlan & plan, ValueId graph_value) {
    return plan.metadata.find_alternate_value(graph_value);
}

inline const CommandPlanAlternateValue * find_alternate_value(const CommandPlan & plan,
                                                              ValueId             graph_value,
                                                              ggml_type           type,
                                                              size_t              byte_count) {
    return plan.metadata.find_alternate_value(graph_value, type, byte_count);
}

inline bool same_full_value_range(const Value & lhs, const Value & rhs) {
    return lhs.storage == rhs.storage && lhs.storage_offset == rhs.storage_offset && lhs.byte_count == rhs.byte_count;
}

inline const CommandPlanAlternateValue * find_alternate_value(const Graph &       graph,
                                                              const CommandPlan & plan,
                                                              ValueId             graph_value,
                                                              ggml_type           type,
                                                              size_t              byte_count) {
    const CommandPlanAlternateValue * exact = find_alternate_value(plan, graph_value, type, byte_count);
    if (exact != nullptr) {
        return exact;
    }

    const Value * value = graph.values().find(graph_value);
    if (value == nullptr) {
        return nullptr;
    }

    auto find_if_same_range = [&](ValueId candidate_id) -> const CommandPlanAlternateValue * {
        const Value * candidate = graph.values().find(candidate_id);
        if (candidate == nullptr || !same_full_value_range(*value, *candidate)) {
            return nullptr;
        }
        return find_alternate_value(plan, candidate_id, type, byte_count);
    };

    ValueId alias = value->alias_source;
    for (size_t i = 0; alias.value >= 0 && i < graph.values().size(); ++i) {
        const CommandPlanAlternateValue * alternate = find_if_same_range(alias);
        if (alternate != nullptr) {
            return alternate;
        }
        const Value * alias_value = graph.values().find(alias);
        if (alias_value == nullptr) {
            break;
        }
        alias = alias_value->alias_source;
    }

    const CommandPlanAlternateValue * root_alternate = find_if_same_range(value->storage_root);
    if (root_alternate != nullptr) {
        return root_alternate;
    }

    for (const CommandPlanAlternateValue & alternate : plan.metadata.alternate_values()) {
        if (alternate.type != type || alternate.byte_count != byte_count) {
            continue;
        }
        const Value * alternate_value = graph.values().find(alternate.graph_value);
        if (alternate_value != nullptr && same_full_value_range(*value, *alternate_value)) {
            return &alternate;
        }
    }
    return nullptr;
}

}  // namespace ggml::hrx
