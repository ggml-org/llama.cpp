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

inline bool same_full_ordered_value_range(const Value & lhs, const Value & rhs) {
    return same_full_value_range(lhs, rhs) && lhs.type == rhs.type && lhs.element_count == rhs.element_count &&
           lhs.contiguous && rhs.contiguous;
}

inline bool same_packed_row_geometry(const Value & lhs, const Value & rhs) {
    return same_full_ordered_value_range(lhs, rhs) && lhs.ne[0] > 0 && rhs.ne[0] > 0 && lhs.ne[0] == rhs.ne[0] &&
           lhs.element_count / lhs.ne[0] == rhs.element_count / rhs.ne[0];
}

inline bool same_q8_1_x4_geometry(const Value & lhs, const Value & rhs) {
    return same_full_ordered_value_range(lhs, rhs) && lhs.ne[0] > 0 && rhs.ne[0] > 0 && lhs.ne[0] % 128 == 0 &&
           rhs.ne[0] % 128 == 0;
}

namespace detail {

template <typename Resource, typename Resources, typename SourceValue, typename Matches, typename Compatible>
inline const Resource * find_full_ordered_value_resource(const Graph &     graph,
                                                         ValueId           graph_value,
                                                         const Resources & resources,
                                                         SourceValue       source_value,
                                                         Matches           matches,
                                                         Compatible        compatible) {
    auto find_exact = [&](ValueId candidate_id) -> const Resource * {
        for (const Resource & resource : resources) {
            if (source_value(resource) == candidate_id && matches(resource)) {
                return &resource;
            }
        }
        return nullptr;
    };

    const Resource * exact = find_exact(graph_value);
    if (exact != nullptr) {
        return exact;
    }

    const Value * value = graph.values().find(graph_value);
    if (value == nullptr) {
        return nullptr;
    }

    auto find_if_compatible = [&](ValueId candidate_id) -> const Resource * {
        const Value * candidate = graph.values().find(candidate_id);
        if (candidate == nullptr) {
            return nullptr;
        }
        const Resource * resource = find_exact(candidate_id);
        return resource != nullptr && compatible(*resource, *value, *candidate) ? resource : nullptr;
    };

    if (!graph.has_index()) {
        return nullptr;
    }

    const Value * current = value;
    for (size_t i = 0; i < graph.values().size(); ++i) {
        const GraphNode * alias = graph.index().producer(current->id);
        if (alias == nullptr || (alias->op != GGML_OP_VIEW && alias->op != GGML_OP_RESHAPE) ||
            alias->output != current->id || alias->inputs.size() != 1 || current->alias_source != alias->inputs[0] ||
            !is_layout_alias_node(graph, *alias)) {
            break;
        }
        const Value * source = graph.values().find(alias->inputs[0]);
        if (source == nullptr || !same_full_ordered_value_range(*current, *source)) {
            break;
        }
        const Resource * resource = find_if_compatible(source->id);
        if (resource != nullptr) {
            return resource;
        }
        current = source;
    }
    return nullptr;
}

}  // namespace detail

inline const CommandPlanAlternateValue * find_alternate_value(const Graph &       graph,
                                                              const CommandPlan & plan,
                                                              ValueId             graph_value,
                                                              ggml_type           type,
                                                              size_t              byte_count) {
    return detail::find_full_ordered_value_resource<CommandPlanAlternateValue>(
        graph, graph_value, plan.metadata.alternate_values(),
        [](const CommandPlanAlternateValue & alternate) { return alternate.graph_value; },
        [=](const CommandPlanAlternateValue & alternate) {
            return alternate.type == type && alternate.byte_count == byte_count;
        },
        [](const CommandPlanAlternateValue & alternate, const Value & value, const Value & source) {
            return alternate.type == GGML_TYPE_F16 ? same_full_ordered_value_range(value, source) :
                   alternate.type == GGML_TYPE_Q8_1 ? same_q8_1_x4_geometry(value, source) :
                                                      same_packed_row_geometry(value, source);
        });
}

inline const CommandPlanGeneratedResource * find_generated_resource(const Graph &         graph,
                                                                    const CommandPlan &   plan,
                                                                    ValueId               graph_value,
                                                                    GeneratedResourceRole role,
                                                                    size_t                byte_count) {
    return detail::find_full_ordered_value_resource<CommandPlanGeneratedResource>(
        graph, graph_value, plan.metadata.generated_resources(),
        [](const CommandPlanGeneratedResource & resource) { return resource.source_value; },
        [=](const CommandPlanGeneratedResource & resource) {
            return resource.role == role && resource.byte_count == byte_count;
        },
        [](const CommandPlanGeneratedResource & resource, const Value & value, const Value & source) {
            return resource.role == GeneratedResourceRole::F16K16Major ? same_packed_row_geometry(value, source) :
                                                                         value.ne == source.ne;
        });
}

}  // namespace ggml::hrx
