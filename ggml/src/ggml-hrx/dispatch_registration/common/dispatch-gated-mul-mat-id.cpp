#include "dispatch-gated-mul-mat-id.h"

#include "dispatch-mul-mat-id-common.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kMulMatIdTiledSwiGLUF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_id_tiled_pair_input_f32_swiglu_publish_f32");
static constexpr KernelCatalogRef kMulMatIdSkinnySwiGLUF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_mul_mat_id_skinny_pair_input_f32_swiglu_publish_f32");

struct MulMatIdSwiGLUMatch {
    const Value *               input       = nullptr;
    const Value *               route_ids   = nullptr;
    const Value *               gate_weight = nullptr;
    const Value *               up_weight   = nullptr;
    const Value *               gate_output = nullptr;
    const Value *               up_output   = nullptr;
    const Value *               output      = nullptr;
    const GraphNode *           gate_node   = nullptr;
    const GraphNode *           up_node     = nullptr;
    const GraphNode *           glu_node    = nullptr;
    CommandPlanMoeRoutingBundle routing_bundle;
    bool                        has_routing_bundle = false;
    int64_t                     input_size         = 0;
    int64_t                     output_size        = 0;
    int64_t                     token_count        = 0;
    int64_t                     route_count        = 0;
    int64_t                     input_route_count  = 0;
    int64_t                     expert_count       = 0;
    CommonMulMatWeightFormat    gate_format        = CommonMulMatWeightFormat::Q4K;
    CommonMulMatWeightFormat    up_format          = CommonMulMatWeightFormat::Q4K;

    bool matched() const {
        return input != nullptr && route_ids != nullptr && gate_weight != nullptr && up_weight != nullptr &&
               gate_output != nullptr && up_output != nullptr && output != nullptr && gate_node != nullptr &&
               up_node != nullptr && glu_node != nullptr;
    }
};

static MulMatIdSwiGLUMatch match_mul_mat_id_swiglu(const DispatchMatchContext & context) {
    MulMatIdSwiGLUMatch       match;
    const CommonMulMatIdMatch root = common_match_mul_mat_id_any_format(context.graph, context.root_node, context.plan);
    if (!root.matched() || !context.graph.has_index()) {
        return match;
    }

    const std::vector<const GraphNode *> & root_consumers = context.graph.index().consumers(context.root_node->output);
    if (root_consumers.size() != 1 || root_consumers.front() == nullptr || root_consumers.front()->op != GGML_OP_GLU) {
        return {};
    }

    const GraphNode * glu_node = root_consumers.front();
    if (glu_node->inputs.size() != 2 || !common_mul_mat_id_is_swiglu_params(glu_node->params)) {
        return {};
    }

    size_t glu_index = 0;
    if (!context.graph.index().node_index(glu_node, glu_index) || glu_index >= context.covered_nodes.size() ||
        context.covered_nodes[glu_index]) {
        return {};
    }

    const bool root_is_gate = glu_node->inputs[0] == context.root_node->output;
    const bool root_is_up   = glu_node->inputs[1] == context.root_node->output;
    if (!root_is_gate && !root_is_up) {
        return {};
    }

    const ValueId     peer_output_id = root_is_gate ? glu_node->inputs[1] : glu_node->inputs[0];
    const GraphNode * peer_node      = context.graph.index().producer(peer_output_id);
    if (peer_node == nullptr || peer_node == context.root_node || peer_node->op != GGML_OP_MUL_MAT_ID) {
        return {};
    }

    size_t peer_index = 0;
    if (!context.graph.index().node_index(peer_node, peer_index) || peer_index >= context.covered_nodes.size() ||
        context.covered_nodes[peer_index]) {
        return {};
    }

    const std::vector<const GraphNode *> & peer_consumers = context.graph.index().consumers(peer_output_id);
    if (peer_consumers.size() != 1 || peer_consumers.front() != glu_node) {
        return {};
    }

    const CommonMulMatIdMatch peer = common_match_mul_mat_id_any_format(context.graph, peer_node, context.plan);
    if (!peer.matched() || peer.input->id != root.input->id || peer.route_ids->id != root.route_ids->id ||
        peer.input_size != root.input_size || peer.output_size != root.output_size ||
        peer.token_count != root.token_count || peer.route_count != root.route_count ||
        peer.input_route_count != root.input_route_count || peer.expert_count != root.expert_count ||
        !common_mul_mat_id_same_shape(*root.output, *peer.output)) {
        return {};
    }

    const Value * output = common_mul_mat_id_graph_value(context.graph, glu_node->output);
    if (output == nullptr || output->type != GGML_TYPE_F32 || !output->contiguous ||
        !common_mul_mat_id_same_shape(*output, *root.output)) {
        return {};
    }

    match.input              = root.input;
    match.route_ids          = root.route_ids;
    match.gate_weight        = root_is_gate ? root.weight : peer.weight;
    match.up_weight          = root_is_gate ? peer.weight : root.weight;
    match.gate_output        = root_is_gate ? root.output : peer.output;
    match.up_output          = root_is_gate ? peer.output : root.output;
    match.output             = output;
    match.gate_node          = root_is_gate ? context.root_node : peer_node;
    match.up_node            = root_is_gate ? peer_node : context.root_node;
    match.glu_node           = glu_node;
    match.routing_bundle     = root.routing_bundle;
    match.has_routing_bundle = root.has_routing_bundle;
    match.input_size         = root.input_size;
    match.output_size        = root.output_size;
    match.token_count        = root.token_count;
    match.route_count        = root.route_count;
    match.input_route_count  = root.input_route_count;
    match.expert_count       = root.expert_count;
    match.gate_format        = root_is_gate ? root.weight_format : peer.weight_format;
    match.up_format          = root_is_gate ? peer.weight_format : root.weight_format;
    return match;
}

static CommonMulMatIdMatch routing_match_for_swiglu(const MulMatIdSwiGLUMatch & match) {
    CommonMulMatIdMatch routed;
    routed.input              = match.input;
    routed.weight             = match.gate_weight;
    routed.output             = match.gate_output;
    routed.route_ids          = match.route_ids;
    routed.routing_bundle     = match.routing_bundle;
    routed.has_routing_bundle = match.has_routing_bundle;
    routed.input_size         = match.input_size;
    routed.output_size        = match.output_size;
    routed.token_count        = match.token_count;
    routed.route_count        = match.route_count;
    routed.input_route_count  = match.input_route_count;
    routed.expert_count       = match.expert_count;
    routed.weight_format      = match.gate_format;
    return routed;
}

static KernelCatalogRef mul_mat_id_swiglu_f32_kernel(const MulMatIdSwiGLUMatch & match) {
    return match.token_count <= 5 ? kMulMatIdSkinnySwiGLUF32F32Kernel : kMulMatIdTiledSwiGLUF32F32Kernel;
}

static bool match_mul_mat_id_swiglu_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const MulMatIdSwiGLUMatch match = match_mul_mat_id_swiglu(context);
    if (!match.matched()) {
        return false;
    }

    const CommonMulMatIdMatch   routed = routing_match_for_swiglu(match);
    CommandPlanMoeRoutingBundle routing_bundle;
    if (!common_mul_mat_id_ensure_moe_routing_bundle(context, routed, dispatch_match, routing_bundle)) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(mul_mat_id_swiglu_f32_kernel(match));
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.workload.token_capacity",
                                               common_mul_mat_id_to_config_value(match.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_id_swiglu.input_size",
                                               common_mul_mat_id_to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_id_swiglu.output_size",
                                               common_mul_mat_id_to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_id_swiglu.expert_count",
                                               common_mul_mat_id_to_config_value(match.expert_count));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_id_swiglu.route_count",
                                               common_mul_mat_id_to_config_value(match.route_count));
    dispatch.kernel.compile_parameters.emplace("ggml.mul_mat_id_swiglu.input_route_count",
                                               common_mul_mat_id_to_config_value(match.input_route_count));
    dispatch.kernel.compile_parameters.emplace(
        "ggml.mul_mat_id_swiglu.gate_weight_format",
        common_mul_mat_id_to_config_value(common_mul_mat_format_config_value(match.gate_format)));
    dispatch.kernel.compile_parameters.emplace(
        "ggml.mul_mat_id_swiglu.up_weight_format",
        common_mul_mat_id_to_config_value(common_mul_mat_format_config_value(match.up_format)));
    dispatch.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    dispatch.bindings.push_back({ routing_bundle.expert_table, 0, routing_bundle.expert_table_byte_count });
    dispatch.bindings.push_back({ routing_bundle.partition_table, 0, routing_bundle.partition_table_byte_count });
    dispatch.bindings.push_back({ match.gate_weight->id, 0, match.gate_weight->byte_count });
    dispatch.bindings.push_back({ match.up_weight->id, 0, match.up_weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    if (!append_covered_node_index_once(context.graph, context.covered_nodes, match.gate_node,
                                        dispatch_match.covered_nodes) ||
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.up_node,
                                        dispatch_match.covered_nodes) ||
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.glu_node,
                                        dispatch_match.covered_nodes)) {
        return false;
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_gated_mul_mat_id_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.mul_mat_id_swiglu.f32_f32_wmma",
        GGML_OP_MUL_MAT_ID,
        DispatchMatchKind::Fused,
        200,
        DispatchSource::Common,
        match_mul_mat_id_swiglu_dispatch,
    });
}

}  // namespace ggml::hrx
