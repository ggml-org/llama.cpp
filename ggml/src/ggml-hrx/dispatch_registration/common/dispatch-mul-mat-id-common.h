#pragma once

#include "../dispatch-registry.h"
#include "dispatch-mul-mat-weight-format.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace ggml::hrx {

struct CommonMulMatIdMatch {
    const Value *               input     = nullptr;
    const Value *               weight    = nullptr;
    const Value *               output    = nullptr;
    const Value *               route_ids = nullptr;
    CommandPlanMoeRoutingBundle routing_bundle;
    bool                        has_routing_bundle = false;
    int64_t                     input_size         = 0;
    int64_t                     output_size        = 0;
    int64_t                     token_count        = 0;
    int64_t                     route_count        = 0;
    int64_t                     input_route_count  = 0;
    int64_t                     expert_count       = 0;
    CommonMulMatWeightFormat    weight_format      = CommonMulMatWeightFormat::Q4K;

    bool matched() const { return input != nullptr && weight != nullptr && output != nullptr && route_ids != nullptr; }
};

inline const Value * common_mul_mat_id_graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

inline bool common_mul_mat_id_is_shape(const Value & value, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    return value.ne[0] == ne0 && value.ne[1] == ne1 && value.ne[2] == ne2 && value.ne[3] == ne3;
}

inline bool common_mul_mat_id_same_shape(const Value & lhs, const Value & rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i]) {
            return false;
        }
    }
    return true;
}

inline bool common_mul_mat_id_supported_dense_input_size(CommonMulMatWeightFormat format, int64_t input_size) {
    return common_mul_mat_supported_dense_input_size(format, input_size);
}

inline bool common_mul_mat_id_supported_dense_output_size(int64_t output_size) {
    return output_size >= 1 && output_size <= 262144;
}

inline bool common_mul_mat_id_supported_rmsnorm_hidden_size(int64_t hidden_size) {
    return hidden_size >= 128 && hidden_size <= 32768 && hidden_size % 128 == 0;
}

inline bool common_mul_mat_id_supported_token_count(int64_t token_count) {
    return token_count >= 1 && token_count <= 2048;
}

inline bool common_mul_mat_id_supported_route_count(int64_t route_count) {
    return route_count >= 1 && route_count <= 32;
}

inline bool common_mul_mat_id_supported_expert_count(int64_t expert_count) {
    return expert_count >= 1 && expert_count <= 512;
}

inline std::string common_mul_mat_id_to_config_value(int64_t value) {
    return std::to_string(value);
}

inline std::string common_mul_mat_id_to_config_value(float value) {
    std::ostringstream out;
    out.precision(9);
    out << value;
    return out.str();
}

inline int64_t common_mul_mat_id_ceil_div(int64_t value, int64_t divisor) {
    return (value + divisor - 1) / divisor;
}

inline int64_t common_mul_mat_id_partition_descriptor_capacity(int64_t token_count,
                                                               int64_t route_count,
                                                               int64_t expert_count) {
    return common_mul_mat_id_ceil_div(token_count * route_count, 32) + expert_count;
}

inline const GraphNode * common_mul_mat_id_find_single_consumer_with_op(const Graph & graph,
                                                                        ValueId       value,
                                                                        ggml_op       op) {
    if (!graph.has_index()) {
        return nullptr;
    }
    const GraphNode * match = nullptr;
    for (const GraphNode * consumer : graph.index().consumers(value)) {
        if (consumer == nullptr || consumer->op != op) {
            continue;
        }
        if (match != nullptr) {
            return nullptr;
        }
        match = consumer;
    }
    return match;
}

inline const GraphNode * common_mul_mat_id_find_only_consumer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    if (!graph.has_index()) {
        return nullptr;
    }
    const std::vector<const GraphNode *> & consumers = graph.index().consumers(value);
    if (consumers.size() != 1 || consumers.front() == nullptr || consumers.front()->op != op) {
        return nullptr;
    }
    return consumers.front();
}

inline bool common_mul_mat_id_binary_node_is_mul(const GraphNode & node) {
    if (node.op != GGML_OP_MUL || node.inputs.size() != 2) {
        return false;
    }
    const BinaryParams * binary_params = op_params_as<BinaryParams>(node.params);
    return binary_params == nullptr || binary_params->op == BinaryKind::Mul;
}

inline bool common_mul_mat_id_binary_node_is_add(const GraphNode & node) {
    if (node.op != GGML_OP_ADD || node.inputs.size() != 2) {
        return false;
    }
    const BinaryParams * binary_params = op_params_as<BinaryParams>(node.params);
    return binary_params == nullptr || binary_params->op == BinaryKind::Add;
}

inline bool common_mul_mat_id_is_swiglu_params(const OpParams & params) {
    const BinaryParams * binary_params = op_params_as<BinaryParams>(params);
    if (binary_params != nullptr) {
        return binary_params->op == BinaryKind::SwiGLU;
    }

    const GluParams * glu_params = op_params_as<GluParams>(params);
    return glu_params != nullptr && glu_params->op == GGML_GLU_OP_SWIGLU;
}

inline bool common_mul_mat_id_is_weight_shape(const Value & weight, int64_t hidden_size) {
    if (weight.ne[0] != hidden_size) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (weight.ne[i] != 1) {
            return false;
        }
    }
    return true;
}

inline size_t common_mul_mat_id_expert_table_size(int64_t token_count, int64_t expert_count) {
    return static_cast<size_t>(expert_count + expert_count * token_count) * sizeof(int32_t);
}

inline size_t common_mul_mat_id_partition_table_size(int64_t token_count, int64_t route_count, int64_t expert_count) {
    const int64_t assignment_count           = token_count * route_count;
    const int64_t assignment_partition_count = (assignment_count + 31) / 32;
    return static_cast<size_t>(1 + assignment_partition_count + expert_count) * sizeof(int32_t);
}

inline bool common_mul_mat_id_bundle_matches(const CommandPlanMoeRoutingBundle & bundle,
                                             ValueId                             route_ids,
                                             int64_t                             token_count,
                                             int64_t                             route_count,
                                             int64_t                             expert_count) {
    return bundle.route_ids == route_ids && bundle.expert_table.value >= 0 && bundle.partition_table.value >= 0 &&
           bundle.expert_table_byte_count == common_mul_mat_id_expert_table_size(token_count, expert_count) &&
           bundle.partition_table_byte_count ==
               common_mul_mat_id_partition_table_size(token_count, route_count, expert_count) &&
           bundle.token_count == token_count && bundle.route_count == route_count &&
           bundle.expert_count == expert_count && bundle.route_stride >= route_count;
}

inline CommonMulMatIdMatch common_match_mul_mat_id_any_format(const Graph &       graph,
                                                              const GraphNode *   node,
                                                              const CommandPlan & plan) {
    CommonMulMatIdMatch match;
    if (node == nullptr || node->op != GGML_OP_MUL_MAT_ID || node->inputs.size() != 3 || !graph.has_index()) {
        return match;
    }

    const Value * weight    = common_mul_mat_id_graph_value(graph, node->inputs[0]);
    const Value * input     = common_mul_mat_id_graph_value(graph, node->inputs[1]);
    const Value * route_ids = common_mul_mat_id_graph_value(graph, node->inputs[2]);
    const Value * output    = common_mul_mat_id_graph_value(graph, node->output);
    if (weight == nullptr || input == nullptr || route_ids == nullptr || output == nullptr || !weight->contiguous ||
        !input->contiguous || !output->contiguous || input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
        route_ids->type != GGML_TYPE_I32) {
        return {};
    }

    CommonMulMatWeightFormat format = CommonMulMatWeightFormat::Q4K;
    if (!common_mul_mat_format_for_type(weight->type, format)) {
        return {};
    }

    const int64_t input_size        = weight->ne[0];
    const int64_t output_size       = weight->ne[1];
    const int64_t expert_count      = weight->ne[2];
    const int64_t input_route_count = input->ne[1];
    const int64_t token_count       = input->ne[2];
    const int64_t route_count       = route_ids->ne[0];
    if (!common_mul_mat_id_is_shape(*weight, input_size, output_size, expert_count, 1) ||
        !common_mul_mat_id_is_shape(*input, input_size, input_route_count, token_count, 1) ||
        !common_mul_mat_id_is_shape(*route_ids, route_count, token_count, 1, 1) ||
        !common_mul_mat_id_is_shape(*output, output_size, route_count, token_count, 1) ||
        !common_mul_mat_id_supported_dense_input_size(format, input_size) ||
        !common_mul_mat_id_supported_dense_output_size(output_size) ||
        !common_mul_mat_id_supported_token_count(token_count) ||
        !common_mul_mat_id_supported_route_count(route_count) ||
        !common_mul_mat_id_supported_expert_count(expert_count) || input_route_count <= 0 ||
        input_route_count > route_count || route_count % input_route_count != 0) {
        return {};
    }

    match.input             = input;
    match.weight            = weight;
    match.output            = output;
    match.route_ids         = route_ids;
    match.input_size        = input_size;
    match.output_size       = output_size;
    match.token_count       = token_count;
    match.route_count       = route_count;
    match.input_route_count = input_route_count;
    match.expert_count      = expert_count;
    match.weight_format     = format;

    const CommandPlanMoeRoutingBundle * routing_bundle = plan.metadata.find_moe_routing_bundle(route_ids->id);
    if (routing_bundle != nullptr &&
        common_mul_mat_id_bundle_matches(*routing_bundle, route_ids->id, token_count, route_count, expert_count)) {
        match.routing_bundle     = *routing_bundle;
        match.has_routing_bundle = true;
    }
    return match;
}

inline void common_mul_mat_id_add_moe_routing_compile_parameters(Dispatch &                  dispatch,
                                                                 const CommonMulMatIdMatch & match) {
    dispatch.kernel.compile_parameters.emplace("ggml.workload.token_capacity",
                                               common_mul_mat_id_to_config_value(match.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.moe_routing.route_count",
                                               common_mul_mat_id_to_config_value(match.route_count));
    dispatch.kernel.compile_parameters.emplace("ggml.moe_routing.expert_count",
                                               common_mul_mat_id_to_config_value(match.expert_count));
    dispatch.kernel.compile_parameters.emplace("ggml.moe_routing.descriptor_expert_mask", "511");
    dispatch.kernel.compile_parameters.emplace("ggml.moe_routing.descriptor_partition_shift", "9");
    dispatch.kernel.compile_parameters.emplace("ggml.moe_routing.descriptor_row_count_shift", "15");
    dispatch.kernel.compile_parameters.emplace("ggml.moe_routing.partition_workgroup_size", "512");
}

inline bool common_mul_mat_id_ensure_moe_routing_bundle(const DispatchMatchContext &  context,
                                                        const CommonMulMatIdMatch &   match,
                                                        DispatchMatch &               dispatch_match,
                                                        CommandPlanMoeRoutingBundle & routing_bundle) {
    static constexpr KernelCatalogRef kMoeBuildExpertTableKernel =
        GGML_HRX_KERNEL_REF("loom_libs", "ggml_moe_build_expert_table");
    static constexpr KernelCatalogRef kMoeBuildExpertPartitionTableKernel =
        GGML_HRX_KERNEL_REF("loom_libs", "ggml_moe_build_expert_partition_table");

    if (match.has_routing_bundle) {
        routing_bundle = match.routing_bundle;
        return true;
    }

    const ValueId expert_table_value(context.next_plan_value.value +
                                     static_cast<int32_t>(dispatch_match.transients.size()));
    const ValueId partition_table_value(expert_table_value.value + 1);
    const size_t  expert_table_bytes = common_mul_mat_id_expert_table_size(match.token_count, match.expert_count);
    const size_t  partition_table_bytes =
        common_mul_mat_id_partition_table_size(match.token_count, match.route_count, match.expert_count);
    const int64_t route_stride     = match.route_count;
    const size_t  route_ids_length = static_cast<size_t>(match.token_count * route_stride) * sizeof(int32_t);

    dispatch_match.transients.push_back(
        { expert_table_value, "common.moe_routing.expert_table", expert_table_bytes, 256 });
    dispatch_match.transients.push_back(
        { partition_table_value, "common.moe_routing.partition_table", partition_table_bytes, 256 });

    const CommandPlanResourceMetadata routing_metadata = make_command_plan_resource_metadata(MoeRoutingResourceMetadata{
        match.token_count,
        match.route_count,
        route_stride,
        match.expert_count,
    });
    routing_bundle                                     = {
        match.route_ids->id,   ValueId(),         expert_table_value, partition_table_value, expert_table_bytes,
        partition_table_bytes, match.token_count, match.route_count,  route_stride,          match.expert_count,
    };

    Status metadata_status;
    if (!dispatch_match.metadata.append_generated_resource(
            {
                match.route_ids->id,
                GeneratedResourceRole::MoeExpertTable,
                expert_table_value,
                expert_table_bytes,
                routing_metadata,
            },
            metadata_status) ||
        !dispatch_match.metadata.append_generated_resource(
            {
                match.route_ids->id,
                GeneratedResourceRole::MoePartitionTable,
                partition_table_value,
                partition_table_bytes,
                routing_metadata,
            },
            metadata_status) ||
        !dispatch_match.metadata.append_moe_routing_bundle(routing_bundle, metadata_status)) {
        dispatch_match.status.append(metadata_status);
        return false;
    }

    Dispatch expert_table_dispatch;
    expert_table_dispatch.kernel = make_kernel_specialization(kMoeBuildExpertTableKernel);
    expert_table_dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    expert_table_dispatch.kernel.integer_parameters.emplace("route_count", match.route_count);
    expert_table_dispatch.kernel.integer_parameters.emplace("route_stride", route_stride);
    expert_table_dispatch.kernel.integer_parameters.emplace("expert_count", match.expert_count);
    common_mul_mat_id_add_moe_routing_compile_parameters(expert_table_dispatch, match);
    expert_table_dispatch.bindings.push_back({ match.route_ids->id, 0, route_ids_length });
    expert_table_dispatch.bindings.push_back({ expert_table_value, 0, expert_table_bytes });
    dispatch_match.dispatches.push_back(std::move(expert_table_dispatch));

    Dispatch partition_table_dispatch;
    partition_table_dispatch.kernel = make_kernel_specialization(kMoeBuildExpertPartitionTableKernel);
    partition_table_dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    partition_table_dispatch.kernel.integer_parameters.emplace("route_count", match.route_count);
    partition_table_dispatch.kernel.integer_parameters.emplace("expert_count", match.expert_count);
    common_mul_mat_id_add_moe_routing_compile_parameters(partition_table_dispatch, match);
    partition_table_dispatch.bindings.push_back({ expert_table_value, 0, expert_table_bytes });
    partition_table_dispatch.bindings.push_back({ partition_table_value, 0, partition_table_bytes });
    dispatch_match.dispatches.push_back(std::move(partition_table_dispatch));
    return true;
}

}  // namespace ggml::hrx
