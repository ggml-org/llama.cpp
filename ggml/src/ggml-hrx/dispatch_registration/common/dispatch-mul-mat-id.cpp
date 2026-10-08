#include "dispatch-mul-mat-id.h"

#include "dispatch-mul-mat-id-common.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kMulMatIdTiledF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_id_tiled_input_f32_publish_f32");
static constexpr KernelCatalogRef kMulMatIdSkinnyF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_id_skinny_input_f32_publish_f32");
static constexpr KernelCatalogRef kMulMatIdTiledPostOpsF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_id_tiled_input_f32_postops_publish_f32");
static constexpr KernelCatalogRef kMulMatIdSkinnyPostOpsF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_id_skinny_input_f32_postops_publish_f32");
static constexpr KernelCatalogRef kMulMatIdTiledPostOpsNextRmsNormF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_id_tiled_input_f32_postops_next_rmsnorm_publish_f32");
static constexpr KernelCatalogRef kMulMatIdSkinnyPostOpsNextRmsNormF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_id_skinny_input_f32_postops_next_rmsnorm_publish_f32");

struct MulMatIdPostOpsMatch {
    CommonMulMatIdMatch            root;
    const Value *                  bias              = nullptr;
    const Value *                  residual_input    = nullptr;
    const Value *                  residual_output   = nullptr;
    const Value *                  norm_weight       = nullptr;
    const Value *                  normalized_output = nullptr;
    std::vector<const GraphNode *> add_nodes;
    const GraphNode *              rms_node     = nullptr;
    const GraphNode *              mul_node     = nullptr;
    KernelCatalogRef               kernel       = {};
    float                          epsilon      = 0.0f;
    bool                           has_bias     = false;
    bool                           has_residual = false;
    bool                           has_rmsnorm  = false;

    bool matched() const {
        return root.matched() && residual_output != nullptr && kernel.id != kUncatalogedKernelId &&
               (has_bias || has_residual);
    }
};

static bool is_bias_shape(const Value & value, int64_t output_size) {
    if (value.type != GGML_TYPE_F32 || !value.contiguous || value.ne[0] != output_size) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (value.ne[i] != 1) {
            return false;
        }
    }
    return true;
}

static bool common_mul_mat_id_is_skinny_route(const CommonMulMatIdMatch & match) {
    return match.matched() && match.token_count <= 5;
}

static KernelCatalogRef common_mul_mat_id_f32_kernel(const CommonMulMatIdMatch & match) {
    return common_mul_mat_id_is_skinny_route(match) ? kMulMatIdSkinnyF32F32Kernel : kMulMatIdTiledF32F32Kernel;
}

static KernelCatalogRef common_mul_mat_id_postops_f32_kernel(const CommonMulMatIdMatch & match, bool has_rmsnorm) {
    if (has_rmsnorm) {
        return common_mul_mat_id_is_skinny_route(match) ? kMulMatIdSkinnyPostOpsNextRmsNormF32F32Kernel :
                                                          kMulMatIdTiledPostOpsNextRmsNormF32F32Kernel;
    }
    return common_mul_mat_id_is_skinny_route(match) ? kMulMatIdSkinnyPostOpsF32F32Kernel :
                                                      kMulMatIdTiledPostOpsF32F32Kernel;
}

static MulMatIdPostOpsMatch match_mul_mat_id_postops(const DispatchMatchContext & context) {
    MulMatIdPostOpsMatch      match;
    const CommonMulMatIdMatch root = common_match_mul_mat_id_any_format(context.graph, context.root_node, context.plan);
    if (!root.matched() || !context.graph.has_index()) {
        return match;
    }

    const Value * current = root.output;
    for (int add_index = 0; add_index < 2; ++add_index) {
        const GraphNode * add_node =
            common_mul_mat_id_find_only_consumer_with_op(context.graph, current->id, GGML_OP_ADD);
        if (add_node == nullptr || !common_mul_mat_id_binary_node_is_add(*add_node)) {
            break;
        }

        const bool current_is_lhs = add_node->inputs[0] == current->id;
        const bool current_is_rhs = add_node->inputs[1] == current->id;
        if (!current_is_lhs && !current_is_rhs) {
            return {};
        }

        const ValueId other_id = current_is_lhs ? add_node->inputs[1] : add_node->inputs[0];
        const Value * other    = common_mul_mat_id_graph_value(context.graph, other_id);
        const Value * output   = common_mul_mat_id_graph_value(context.graph, add_node->output);
        if (other == nullptr || output == nullptr || output->type != GGML_TYPE_F32 || !output->contiguous ||
            !common_mul_mat_id_same_shape(*root.output, *output)) {
            return {};
        }

        if (!match.has_bias && is_bias_shape(*other, root.output_size)) {
            match.bias = other;
            match.add_nodes.push_back(add_node);
            match.has_bias = true;
            current        = output;
            continue;
        }

        if (!match.has_residual && other->type == GGML_TYPE_F32 && other->contiguous &&
            common_mul_mat_id_same_shape(*root.output, *other)) {
            match.residual_input = other;
            match.add_nodes.push_back(add_node);
            match.has_residual = true;
            current            = output;
            continue;
        }

        break;
    }

    if (!match.has_bias && !match.has_residual) {
        return {};
    }

    match.residual_output = current;

    const bool may_match_next_rmsnorm =
        match.has_residual && common_mul_mat_id_supported_rmsnorm_hidden_size(root.output_size);
    if (may_match_next_rmsnorm) {
        const GraphNode * rms_node =
            common_mul_mat_id_find_single_consumer_with_op(context.graph, current->id, GGML_OP_RMS_NORM);
        if (rms_node != nullptr && rms_node->inputs.size() == 1) {
            const RmsNormParams * rms_params = op_params_as<RmsNormParams>(rms_node->params);
            const Value *         rms_output = common_mul_mat_id_graph_value(context.graph, rms_node->output);
            if (rms_params != nullptr && std::isfinite(rms_params->eps) && rms_params->eps > 0.0f &&
                rms_output != nullptr && rms_output->type == GGML_TYPE_F32 && rms_output->contiguous &&
                common_mul_mat_id_same_shape(*current, *rms_output)) {
                const GraphNode * mul_node =
                    common_mul_mat_id_find_single_consumer_with_op(context.graph, rms_node->output, GGML_OP_MUL);
                if (mul_node != nullptr && common_mul_mat_id_binary_node_is_mul(*mul_node)) {
                    const bool rms_is_lhs = mul_node->inputs[0] == rms_node->output;
                    const bool rms_is_rhs = mul_node->inputs[1] == rms_node->output;
                    if (rms_is_lhs || rms_is_rhs) {
                        const ValueId norm_weight_id = rms_is_lhs ? mul_node->inputs[1] : mul_node->inputs[0];
                        const Value * norm_weight    = common_mul_mat_id_graph_value(context.graph, norm_weight_id);
                        const Value * normalized_output =
                            common_mul_mat_id_graph_value(context.graph, mul_node->output);
                        if (norm_weight != nullptr && normalized_output != nullptr &&
                            norm_weight->type == GGML_TYPE_F32 && normalized_output->type == GGML_TYPE_F32 &&
                            norm_weight->contiguous && normalized_output->contiguous &&
                            common_mul_mat_id_is_weight_shape(*norm_weight, root.output_size) &&
                            common_mul_mat_id_same_shape(*current, *normalized_output)) {
                            match.norm_weight       = norm_weight;
                            match.normalized_output = normalized_output;
                            match.rms_node          = rms_node;
                            match.mul_node          = mul_node;
                            match.epsilon           = rms_params->eps;
                            match.has_rmsnorm       = true;
                        }
                    }
                }
            }
        }
    }

    match.root   = root;
    match.kernel = common_mul_mat_id_postops_f32_kernel(root, match.has_rmsnorm);
    return match;
}

}  // namespace

static bool match_mul_mat_id_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const CommonMulMatIdMatch match =
        common_match_mul_mat_id_any_format(context.graph, context.root_node, context.plan);
    if (!match.matched()) {
        return false;
    }

    CommandPlanMoeRoutingBundle routing_bundle;
    if (!common_mul_mat_id_ensure_moe_routing_bundle(context, match, dispatch_match, routing_bundle)) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(common_mul_mat_id_f32_kernel(match));
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.workload.token_capacity",
                                               common_mul_mat_id_to_config_value(match.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_id.input_size",
                                               common_mul_mat_id_to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_id.output_size",
                                               common_mul_mat_id_to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_id.expert_count",
                                               common_mul_mat_id_to_config_value(match.expert_count));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_id.route_count",
                                               common_mul_mat_id_to_config_value(match.route_count));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_id.input_route_count",
                                               common_mul_mat_id_to_config_value(match.input_route_count));
    dispatch.kernel.compile_parameters.emplace(
        "ggml.mul_mat_id.weight_format",
        common_mul_mat_id_to_config_value(common_mul_mat_format_config_value(match.weight_format)));
    dispatch.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    dispatch.bindings.push_back({ routing_bundle.expert_table, 0, routing_bundle.expert_table_byte_count });
    dispatch.bindings.push_back({ routing_bundle.partition_table, 0, routing_bundle.partition_table_byte_count });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_mul_mat_id_postops_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const MulMatIdPostOpsMatch match = match_mul_mat_id_postops(context);
    if (!match.matched()) {
        return false;
    }

    CommandPlanMoeRoutingBundle routing_bundle;
    if (!common_mul_mat_id_ensure_moe_routing_bundle(context, match.root, dispatch_match, routing_bundle)) {
        return false;
    }

    const int64_t completion_counter_count =
        match.has_rmsnorm ? common_mul_mat_id_partition_descriptor_capacity(
                                match.root.token_count, match.root.route_count, match.root.expert_count) :
                            0;
    const ValueId completion_counters(context.next_plan_value.value +
                                      static_cast<int32_t>(dispatch_match.transients.size()));

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(match.kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.root.token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.workload.token_capacity",
                                               common_mul_mat_id_to_config_value(match.root.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_id_postops.input_size",
                                               common_mul_mat_id_to_config_value(match.root.input_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_id_postops.output_size",
                                               common_mul_mat_id_to_config_value(match.root.output_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_id_postops.expert_count",
                                               common_mul_mat_id_to_config_value(match.root.expert_count));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_id_postops.route_count",
                                               common_mul_mat_id_to_config_value(match.root.route_count));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_id_postops.input_route_count",
                                               common_mul_mat_id_to_config_value(match.root.input_route_count));
    dispatch.kernel.compile_parameters.emplace(
        "ggml.mul_mat_id_postops.weight_format",
        common_mul_mat_id_to_config_value(common_mul_mat_format_config_value(match.root.weight_format)));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_id_postops.has_bias", match.has_bias ? "1" : "0");
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_id_postops.has_residual", match.has_residual ? "1" : "0");
    if (match.has_rmsnorm) {
        dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_id_postops.rms_epsilon",
                                                   common_mul_mat_id_to_config_value(match.epsilon));
    }

    dispatch.bindings.push_back({ match.root.input->id, 0, match.root.input->byte_count });
    dispatch.bindings.push_back({ routing_bundle.expert_table, 0, routing_bundle.expert_table_byte_count });
    dispatch.bindings.push_back({ routing_bundle.partition_table, 0, routing_bundle.partition_table_byte_count });
    dispatch.bindings.push_back({ match.root.weight->id, 0, match.root.weight->byte_count });
    if (match.has_bias) {
        dispatch.bindings.push_back({ match.bias->id, 0, match.bias->byte_count });
    } else {
        dispatch.bindings.push_back({ match.residual_output->id, 0, match.residual_output->byte_count });
    }
    if (match.has_residual) {
        dispatch.bindings.push_back({ match.residual_input->id, 0, match.residual_input->byte_count });
    } else {
        dispatch.bindings.push_back({ match.residual_output->id, 0, match.residual_output->byte_count });
    }
    dispatch.bindings.push_back({ match.residual_output->id, 0, match.residual_output->byte_count });
    if (match.has_rmsnorm) {
        dispatch.bindings.push_back({ match.norm_weight->id, 0, match.norm_weight->byte_count });
        dispatch.bindings.push_back({ match.normalized_output->id, 0, match.normalized_output->byte_count });
        dispatch.bindings.push_back(
            { completion_counters, 0, static_cast<size_t>(completion_counter_count) * sizeof(int32_t) });
    }

    if (!append_covered_node_index_once(context.graph, context.covered_nodes, context.root_node,
                                        dispatch_match.covered_nodes)) {
        return false;
    }
    for (const GraphNode * add_node : match.add_nodes) {
        if (!append_covered_node_index_once(context.graph, context.covered_nodes, add_node,
                                            dispatch_match.covered_nodes)) {
            return false;
        }
    }
    if (match.has_rmsnorm && (!append_covered_node_index_once(context.graph, context.covered_nodes, match.rms_node,
                                                              dispatch_match.covered_nodes) ||
                              !append_covered_node_index_once(context.graph, context.covered_nodes, match.mul_node,
                                                              dispatch_match.covered_nodes))) {
        return false;
    }

    if (match.has_rmsnorm) {
        dispatch_match.completion_counter_requests.push_back({ completion_counters,
                                                               "common.mul_mat_id_postops.completion_counters",
                                                               static_cast<uint32_t>(completion_counter_count) });
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

void register_mul_mat_id_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.mul_mat_id_postops.f32_f32_wmma",
        GGML_OP_MUL_MAT_ID,
        DispatchMatchKind::Fused,
        180,
        DispatchSource::Common,
        match_mul_mat_id_postops_dispatch,
    });
    registry.add({
        "common.mul_mat_id.f32_f32_wmma",
        GGML_OP_MUL_MAT_ID,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Common,
        match_mul_mat_id_dispatch,
    });
}

}  // namespace ggml::hrx
