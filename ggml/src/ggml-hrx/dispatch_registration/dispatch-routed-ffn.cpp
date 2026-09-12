#include "dispatch-routed-ffn.h"

#include "dispatch-llm-shapes.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kQwenRoutedGateUpSwiGLUQ4KF16WmmaKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_gate_up_swiglu_q4k_f16_wmma");
static constexpr KernelCatalogRef kQwenRoutedGateUpSwiGLUQ4KQ8Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_gate_up_swiglu_q4k_q8");
static constexpr KernelCatalogRef kQwenRoutedGateUpSwiGLUQ4KQ8NextQ8Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_gate_up_swiglu_q4k_q8_1_x4_next_q8");
static constexpr KernelCatalogRef kQwenRoutedDownQ4KF16WmmaGroupedKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_down_q4k_f16_wmma_grouped");
static constexpr KernelCatalogRef kQwenRoutedDownQ6KF16WmmaGroupedKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_down_q6k_f16_wmma_grouped");
static constexpr KernelCatalogRef kQwenRoutedDownQ4KQ8NextQ8Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_down_q4k_q8_1_x4_next_q8");
static constexpr KernelCatalogRef kQwenRoutedDownQ6KF32Wave64NextQ8Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_down_q6k_f32_wave64_next_q8");
static constexpr KernelCatalogRef kQwenRoutedDownWeightedReduceF16F32Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_down_weighted_reduce_f16_f32");
static constexpr KernelCatalogRef kQwenRoutedDownWeightedReduceNextRmsNormF32Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_routed_down_weighted_reduce_next_rmsnorm_f32");

static constexpr const LlmMoeDispatchProfile & kRoutedFfnProfile                 = kActiveLlmMoeDispatchProfile;
static constexpr int64_t                       kRoutedFfnInputSize               = kRoutedFfnProfile.hidden_size;
static constexpr int64_t                       kRoutedFfnExpertHiddenSize        = kRoutedFfnProfile.expert_hidden_size;
static constexpr int64_t                       kRoutedFfnExpertCount             = kRoutedFfnProfile.expert_count;
static constexpr int64_t                       kRoutedFfnRouteCount              = kRoutedFfnProfile.route_count;
static constexpr size_t                        kRoutedFfnPlanTransientAlignment  = 256;
static constexpr const char *                  kRoutedFfnF16GateUpOutputName     = "qwen.moe.gate_up_swiglu_f16";
static constexpr const char *                  kRoutedFfnF16RoutedDownOutputName = "qwen.moe.routed_down_f16";
static constexpr const char *                  kRoutedFfnQ8GateUpOutputName      = "qwen.decode.moe.gate_up_swiglu_q8";
static constexpr const char *                  kRoutedFfnQ8HiddenOutputName      = "qwen.decode.moe.hidden_q8";

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool is_shape(const Value & value, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    return value.ne[0] == ne0 && value.ne[1] == ne1 && value.ne[2] == ne2 && value.ne[3] == ne3;
}

static bool same_shape(const Value & lhs, const Value & rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i]) {
            return false;
        }
    }
    return true;
}

static bool is_profile_rms_norm_epsilon(float eps) {
    const float expected = kRoutedFfnProfile.rms_norm_epsilon;
    return eps >= expected * 0.9f && eps <= expected * 1.1f;
}

static bool is_routed_ffn_gate_up_weight(const Value & value) {
    return value.type == GGML_TYPE_Q4_K && value.contiguous &&
           is_shape(value, kRoutedFfnInputSize, kRoutedFfnExpertHiddenSize, kRoutedFfnExpertCount, 1);
}

static bool is_routed_ffn_down_weight(const Value & value) {
    return (value.type == GGML_TYPE_Q4_K || value.type == GGML_TYPE_Q6_K) && value.contiguous &&
           is_shape(value, kRoutedFfnExpertHiddenSize, kRoutedFfnInputSize, kRoutedFfnExpertCount, 1);
}

static bool is_routed_ffn_projection_output(const Value & value, int64_t token_count) {
    return value.type == GGML_TYPE_F32 && value.contiguous &&
           is_shape(value, kRoutedFfnExpertHiddenSize, kRoutedFfnRouteCount, token_count, 1);
}

static bool is_routed_ffn_down_output(const Value & value, int64_t token_count) {
    return value.type == GGML_TYPE_F32 && value.contiguous &&
           is_shape(value, kRoutedFfnInputSize, kRoutedFfnRouteCount, token_count, 1);
}

static const GraphNode * find_consumer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    for (const GraphNode * consumer : graph.index().consumers(value)) {
        if (consumer != nullptr && consumer->op == op) {
            return consumer;
        }
    }
    return nullptr;
}

static std::vector<const GraphNode *> find_consumers_with_op(const Graph & graph, ValueId value, ggml_op op) {
    std::vector<const GraphNode *> matches;
    for (const GraphNode * consumer : graph.index().consumers(value)) {
        if (consumer != nullptr && consumer->op == op) {
            matches.push_back(consumer);
        }
    }
    return matches;
}

static const GraphNode * find_single_consumer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    const std::vector<const GraphNode *> consumers = find_consumers_with_op(graph, value, op);
    return consumers.size() == 1 ? consumers.front() : nullptr;
}

static const GraphNode * producer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    const GraphNode * producer = graph.index().producer(value);
    return producer != nullptr && producer->op == op ? producer : nullptr;
}

static bool append_covered_node(const DispatchMatchContext & context, const GraphNode * node, DispatchMatch & match) {
    return append_covered_node_index_once(context.graph, context.covered_nodes, node, match.covered_nodes);
}

static std::string to_config_value(int64_t value) {
    return std::to_string(value);
}

static size_t expert_table_size(int64_t token_count) {
    return static_cast<size_t>(kRoutedFfnExpertCount + kRoutedFfnExpertCount * token_count) * sizeof(int32_t);
}

static size_t partition_table_size(int64_t token_count) {
    const int64_t assignment_count           = token_count * kRoutedFfnRouteCount;
    const int64_t assignment_partition_count = (assignment_count + 31) / 32;
    return static_cast<size_t>(1 + assignment_partition_count + kRoutedFfnExpertCount) * sizeof(int32_t);
}

static size_t f16_gate_up_output_size(int64_t token_count) {
    return static_cast<size_t>(token_count * kRoutedFfnRouteCount * kRoutedFfnExpertHiddenSize) * sizeof(ggml_fp16_t);
}

static size_t f16_routed_down_output_size(int64_t token_count) {
    return static_cast<size_t>(token_count * kRoutedFfnRouteCount * kRoutedFfnInputSize) * sizeof(ggml_fp16_t);
}

static size_t q8_1_x4_byte_count(int64_t row_count, int64_t input_size) {
    if (row_count <= 0 || input_size <= 0) {
        return 0;
    }
    return static_cast<size_t>(row_count) * ggml_row_size(GGML_TYPE_Q8_1, input_size);
}

static uint32_t gate_up_completion_counter_count(int64_t token_count) {
    const int64_t physical_group_count = (kRoutedFfnExpertHiddenSize + 127) / 128;
    return static_cast<uint32_t>(token_count * kRoutedFfnRouteCount * physical_group_count);
}

static void add_routed_down_compile_parameters(Dispatch & dispatch, int64_t token_count) {
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_down.input_size",
                                               to_config_value(kRoutedFfnExpertHiddenSize));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_down.route_count",
                                               to_config_value(kRoutedFfnRouteCount));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_down.expert_count",
                                               to_config_value(kRoutedFfnExpertCount));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_down.output_size",
                                               to_config_value(kRoutedFfnInputSize));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity", to_config_value(token_count));
}

struct RoutedGateUpMatch {
    const Value *                       gate_weight    = nullptr;
    const Value *                       up_weight      = nullptr;
    const Value *                       input          = nullptr;
    const Value *                       route_ids      = nullptr;
    const Value *                       gate_output    = nullptr;
    const Value *                       up_output      = nullptr;
    const Value *                       glu_output     = nullptr;
    const CommandPlanMoeRoutingBundle * routing_bundle = nullptr;
    const GraphNode *                   gate_node      = nullptr;
    const GraphNode *                   up_node        = nullptr;
    const GraphNode *                   glu_node       = nullptr;
    int64_t                             token_count    = 0;

    bool matched() const {
        return gate_weight != nullptr && up_weight != nullptr && input != nullptr && route_ids != nullptr &&
               gate_output != nullptr && up_output != nullptr && glu_output != nullptr && routing_bundle != nullptr &&
               gate_node != nullptr && up_node != nullptr && glu_node != nullptr && token_count > 0;
    }
};

struct DecodeRoutedGateUpMatch {
    const Value *                     gate_weight     = nullptr;
    const Value *                     up_weight       = nullptr;
    const Value *                     input           = nullptr;
    const Value *                     route_ids       = nullptr;
    const Value *                     glu_output      = nullptr;
    const CommandPlanAlternateValue * input_alternate = nullptr;
    const GraphNode *                 down_node       = nullptr;
    const Value *                     down_weight     = nullptr;
    const GraphNode *                 gate_node       = nullptr;
    const GraphNode *                 up_node         = nullptr;
    const GraphNode *                 glu_node        = nullptr;
    int64_t                           token_count     = 0;
    int64_t                           route_stride    = 0;
    bool                              publish_q8      = false;

    bool matched() const {
        return gate_weight != nullptr && up_weight != nullptr && input != nullptr && route_ids != nullptr &&
               glu_output != nullptr && input_alternate != nullptr && down_node != nullptr && down_weight != nullptr &&
               gate_node != nullptr && up_node != nullptr && glu_node != nullptr && token_count == 1 &&
               route_stride >= kRoutedFfnRouteCount;
    }
};

struct RoutedDownMatch {
    const Value *                       input_graph_value = nullptr;
    const CommandPlanAlternateValue *   input_alternate   = nullptr;
    const Value *                       weight            = nullptr;
    const Value *                       output            = nullptr;
    const Value *                       route_ids         = nullptr;
    const CommandPlanMoeRoutingBundle * routing_bundle    = nullptr;
    KernelCatalogRef                    kernel            = {};
    int64_t                             token_count       = 0;

    bool matched() const {
        return input_graph_value != nullptr && input_alternate != nullptr && weight != nullptr && output != nullptr &&
               route_ids != nullptr && routing_bundle != nullptr && kernel.id != kUncatalogedKernelId &&
               token_count > 0;
    }
};

struct WeightedReduceNextRmsNormMatch {
    const GraphNode * rms_node    = nullptr;
    const GraphNode * mul_node    = nullptr;
    const Value *     norm_weight = nullptr;
    const Value *     output      = nullptr;

    bool matched() const {
        return rms_node != nullptr && mul_node != nullptr && norm_weight != nullptr && output != nullptr;
    }
};

struct WeightedReduceMatch {
    const Value *                     route_weights    = nullptr;
    const Value *                     routed_output    = nullptr;
    const CommandPlanAlternateValue * routed_alternate = nullptr;
    const Value *                     residual_input   = nullptr;
    const Value *                     output           = nullptr;
    const GraphNode *                 weighted_node    = nullptr;
    std::vector<const GraphNode *>    views;
    std::vector<const GraphNode *>    reductions;
    const GraphNode *                 residual = nullptr;
    WeightedReduceNextRmsNormMatch    next_rmsnorm;
    int64_t                           token_count = 0;

    bool topology_matched() const {
        return route_weights != nullptr && routed_output != nullptr && residual_input != nullptr && output != nullptr &&
               weighted_node != nullptr && !views.empty() && residual != nullptr && token_count > 0;
    }

    bool matched() const { return topology_matched() && routed_alternate != nullptr; }
};

struct DecodeRoutedDownMatch {
    const Value *                     input_graph_value = nullptr;
    const CommandPlanAlternateValue * input_alternate   = nullptr;
    const Value *                     weight            = nullptr;
    const Value *                     output            = nullptr;
    const Value *                     route_ids         = nullptr;
    WeightedReduceMatch               reduce;
    KernelCatalogRef                  kernel       = {};
    int64_t                           token_count  = 0;
    int64_t                           route_stride = 0;
    bool                              input_is_q8  = false;

    bool matched() const {
        return input_graph_value != nullptr && weight != nullptr && output != nullptr && route_ids != nullptr &&
               reduce.topology_matched() && reduce.next_rmsnorm.matched() && kernel.id != kUncatalogedKernelId &&
               token_count == 1 && route_stride >= kRoutedFfnRouteCount && (!input_is_q8 || input_alternate != nullptr);
    }
};

static bool bundle_matches_moe_routing(const CommandPlanMoeRoutingBundle & bundle,
                                       ValueId                             route_ids,
                                       int64_t                             token_count) {
    return bundle.route_ids == route_ids && bundle.route_weights.value >= 0 && bundle.expert_table.value >= 0 &&
           bundle.partition_table.value >= 0 && bundle.expert_table_byte_count == expert_table_size(token_count) &&
           bundle.partition_table_byte_count == partition_table_size(token_count) &&
           bundle.token_count == token_count && bundle.route_count == kRoutedFfnRouteCount &&
           bundle.expert_count == kRoutedFfnExpertCount && bundle.route_stride >= kRoutedFfnRouteCount;
}

static bool match_same_route_projection(const Graph &     graph,
                                        const GraphNode & node,
                                        ValueId           expected_input,
                                        ValueId           expected_route_ids,
                                        int64_t           token_count,
                                        const Value *&    weight,
                                        const Value *&    output) {
    if (node.op != GGML_OP_MUL_MAT_ID || node.inputs.size() != 3 || node.inputs[1] != expected_input ||
        node.inputs[2] != expected_route_ids) {
        return false;
    }
    weight = graph_value(graph, node.inputs[0]);
    output = graph_value(graph, node.output);
    return weight != nullptr && output != nullptr && is_routed_ffn_gate_up_weight(*weight) &&
           is_routed_ffn_projection_output(*output, token_count);
}

static RoutedDownMatch match_routed_ffn_down_grouped(const DispatchMatchContext & context) {
    RoutedDownMatch   match;
    const GraphNode * root = context.root_node;
    if (root == nullptr || root->op != GGML_OP_MUL_MAT_ID || root->inputs.size() != 3 || !context.graph.has_index()) {
        return match;
    }

    const Value * weight      = graph_value(context.graph, root->inputs[0]);
    const Value * input       = graph_value(context.graph, root->inputs[1]);
    const Value * route_ids   = graph_value(context.graph, root->inputs[2]);
    const Value * root_output = graph_value(context.graph, root->output);
    if (weight == nullptr || input == nullptr || route_ids == nullptr || root_output == nullptr ||
        !is_routed_ffn_down_weight(*weight) || !is_routed_ffn_projection_output(*input, input->ne[2]) ||
        route_ids->type != GGML_TYPE_I32 || !is_shape(*route_ids, kRoutedFfnRouteCount, input->ne[2], 1, 1)) {
        return {};
    }

    const int64_t token_count = input->ne[2];
    if (!is_llm_supported_query_length(kRoutedFfnProfile, token_count) ||
        !is_routed_ffn_down_output(*root_output, token_count)) {
        return {};
    }

    const CommandPlanMoeRoutingBundle * routing_bundle = context.plan.metadata.find_moe_routing_bundle(route_ids->id);
    if (routing_bundle == nullptr || !bundle_matches_moe_routing(*routing_bundle, route_ids->id, token_count)) {
        return {};
    }

    const CommandPlanAlternateValue * input_alternate =
        find_alternate_value(context.plan, input->id, GGML_TYPE_F16, f16_gate_up_output_size(token_count));
    if (input_alternate == nullptr) {
        return {};
    }

    match.input_graph_value = input;
    match.input_alternate   = input_alternate;
    match.weight            = weight;
    match.output            = root_output;
    match.route_ids         = route_ids;
    match.routing_bundle    = routing_bundle;
    match.kernel            = weight->type == GGML_TYPE_Q4_K ? kQwenRoutedDownQ4KF16WmmaGroupedKernel :
                                                               kQwenRoutedDownQ6KF16WmmaGroupedKernel;
    match.token_count       = token_count;
    return match;
}

static RoutedGateUpMatch match_routed_ffn_gate_up_swiglu(const DispatchMatchContext & context) {
    RoutedGateUpMatch match;
    const GraphNode * root = context.root_node;
    if (root == nullptr || root->op != GGML_OP_MUL_MAT_ID || root->inputs.size() != 3 || !context.graph.has_index()) {
        return match;
    }

    const Value * root_weight = graph_value(context.graph, root->inputs[0]);
    const Value * input       = graph_value(context.graph, root->inputs[1]);
    const Value * route_ids   = graph_value(context.graph, root->inputs[2]);
    const Value * root_output = graph_value(context.graph, root->output);
    if (root_weight == nullptr || input == nullptr || route_ids == nullptr || root_output == nullptr ||
        !is_routed_ffn_gate_up_weight(*root_weight) || input->type != GGML_TYPE_F32 || !input->contiguous ||
        !is_shape(*input, kRoutedFfnInputSize, 1, input->ne[2], 1) || route_ids->type != GGML_TYPE_I32 ||
        !is_shape(*route_ids, kRoutedFfnRouteCount, input->ne[2], 1, 1)) {
        return {};
    }

    const int64_t token_count = input->ne[2];
    if (!is_llm_supported_query_length(kRoutedFfnProfile, token_count) ||
        !is_routed_ffn_projection_output(*root_output, token_count)) {
        return {};
    }

    const CommandPlanMoeRoutingBundle * routing_bundle = context.plan.metadata.find_moe_routing_bundle(route_ids->id);
    if (routing_bundle == nullptr || !bundle_matches_moe_routing(*routing_bundle, route_ids->id, token_count)) {
        return {};
    }

    const GraphNode * glu_node = find_consumer_with_op(context.graph, root->output, GGML_OP_GLU);
    if (glu_node == nullptr || glu_node->inputs.size() != 2) {
        return {};
    }
    const GluParams * glu_params = op_params_as<GluParams>(glu_node->params);
    if (glu_params == nullptr || glu_params->op != GGML_GLU_OP_SWIGLU) {
        return {};
    }

    const GraphNode * gate_node = producer_with_op(context.graph, glu_node->inputs[0], GGML_OP_MUL_MAT_ID);
    const GraphNode * up_node   = producer_with_op(context.graph, glu_node->inputs[1], GGML_OP_MUL_MAT_ID);
    if (gate_node == nullptr || up_node == nullptr || gate_node == up_node || (gate_node != root && up_node != root)) {
        return {};
    }

    const Value * gate_weight = nullptr;
    const Value * gate_output = nullptr;
    const Value * up_weight   = nullptr;
    const Value * up_output   = nullptr;
    if (!match_same_route_projection(context.graph, *gate_node, input->id, route_ids->id, token_count, gate_weight,
                                     gate_output) ||
        !match_same_route_projection(context.graph, *up_node, input->id, route_ids->id, token_count, up_weight,
                                     up_output)) {
        return {};
    }
    if (!same_shape(*gate_output, *up_output)) {
        return {};
    }

    const Value * glu_output = graph_value(context.graph, glu_node->output);
    if (glu_output == nullptr || glu_output->kind != ValueKind::Transient ||
        !is_routed_ffn_projection_output(*glu_output, token_count)) {
        return {};
    }

    match.gate_weight    = gate_weight;
    match.up_weight      = up_weight;
    match.input          = input;
    match.route_ids      = route_ids;
    match.gate_output    = gate_output;
    match.up_output      = up_output;
    match.glu_output     = glu_output;
    match.routing_bundle = routing_bundle;
    match.gate_node      = gate_node;
    match.up_node        = up_node;
    match.glu_node       = glu_node;
    match.token_count    = token_count;
    return match;
}

static DecodeRoutedGateUpMatch match_decode_routed_ffn_gate_up_swiglu(const DispatchMatchContext & context) {
    DecodeRoutedGateUpMatch match;
    const GraphNode *       root = context.root_node;
    if (root == nullptr || root->op != GGML_OP_MUL_MAT_ID || root->inputs.size() != 3 || !context.graph.has_index()) {
        return match;
    }

    const Value * root_weight = graph_value(context.graph, root->inputs[0]);
    const Value * input       = graph_value(context.graph, root->inputs[1]);
    const Value * route_ids   = graph_value(context.graph, root->inputs[2]);
    const Value * root_output = graph_value(context.graph, root->output);
    if (root_weight == nullptr || input == nullptr || route_ids == nullptr || root_output == nullptr ||
        !is_routed_ffn_gate_up_weight(*root_weight) || input->type != GGML_TYPE_F32 || !input->contiguous ||
        !is_shape(*input, kRoutedFfnInputSize, 1, 1, 1) || route_ids->type != GGML_TYPE_I32 ||
        !is_shape(*route_ids, kRoutedFfnRouteCount, 1, 1, 1) || !is_routed_ffn_projection_output(*root_output, 1)) {
        return {};
    }

    const int64_t                     route_stride    = static_cast<int64_t>(route_ids->nb[1] / sizeof(int32_t));
    const CommandPlanAlternateValue * input_alternate = find_alternate_value(
        context.graph, context.plan, input->id, GGML_TYPE_Q8_1, q8_1_x4_byte_count(1, kRoutedFfnInputSize));
    if (input_alternate == nullptr) {
        return {};
    }

    const GraphNode * glu_node = find_consumer_with_op(context.graph, root->output, GGML_OP_GLU);
    if (glu_node == nullptr || glu_node->inputs.size() != 2) {
        return {};
    }
    const GluParams * glu_params = op_params_as<GluParams>(glu_node->params);
    if (glu_params == nullptr || glu_params->op != GGML_GLU_OP_SWIGLU) {
        return {};
    }

    const GraphNode * gate_node = producer_with_op(context.graph, glu_node->inputs[0], GGML_OP_MUL_MAT_ID);
    const GraphNode * up_node   = producer_with_op(context.graph, glu_node->inputs[1], GGML_OP_MUL_MAT_ID);
    if (gate_node == nullptr || up_node == nullptr || gate_node == up_node || (gate_node != root && up_node != root)) {
        return {};
    }

    const Value * gate_weight = nullptr;
    const Value * gate_output = nullptr;
    const Value * up_weight   = nullptr;
    const Value * up_output   = nullptr;
    if (!match_same_route_projection(context.graph, *gate_node, input->id, route_ids->id, 1, gate_weight,
                                     gate_output) ||
        !match_same_route_projection(context.graph, *up_node, input->id, route_ids->id, 1, up_weight, up_output) ||
        !same_shape(*gate_output, *up_output)) {
        return {};
    }

    const Value * glu_output = graph_value(context.graph, glu_node->output);
    if (glu_output == nullptr || glu_output->kind != ValueKind::Transient ||
        !is_routed_ffn_projection_output(*glu_output, 1)) {
        return {};
    }

    const GraphNode * down_node   = find_single_consumer_with_op(context.graph, glu_output->id, GGML_OP_MUL_MAT_ID);
    const Value *     down_weight = down_node == nullptr || down_node->inputs.size() != 3 ?
                                        nullptr :
                                        graph_value(context.graph, down_node->inputs[0]);
    if (down_node == nullptr || down_weight == nullptr || !is_routed_ffn_down_weight(*down_weight)) {
        return {};
    }

    match.gate_weight     = gate_weight;
    match.up_weight       = up_weight;
    match.input           = input;
    match.route_ids       = route_ids;
    match.glu_output      = glu_output;
    match.input_alternate = input_alternate;
    match.down_node       = down_node;
    match.down_weight     = down_weight;
    match.gate_node       = gate_node;
    match.up_node         = up_node;
    match.glu_node        = glu_node;
    match.token_count     = 1;
    match.route_stride    = route_stride;
    match.publish_q8      = down_weight->type == GGML_TYPE_Q4_K;
    return match;
}

static WeightedReduceNextRmsNormMatch match_qwen_weighted_reduce_next_rmsnorm(const DispatchMatchContext & context,
                                                                              const Value &                residual) {
    WeightedReduceNextRmsNormMatch match;
    const GraphNode * rms_node = find_single_consumer_with_op(context.graph, residual.id, GGML_OP_RMS_NORM);
    if (rms_node == nullptr || rms_node->inputs.size() != 1) {
        return match;
    }
    const RmsNormParams * rms_params = op_params_as<RmsNormParams>(rms_node->params);
    if (rms_params == nullptr || !is_profile_rms_norm_epsilon(rms_params->eps)) {
        return {};
    }

    const Value * rms = graph_value(context.graph, rms_node->output);
    if (rms == nullptr || rms->type != GGML_TYPE_F32 || !same_shape(*rms, residual)) {
        return {};
    }

    const GraphNode * mul_node = find_single_consumer_with_op(context.graph, rms_node->output, GGML_OP_MUL);
    if (mul_node == nullptr || mul_node->inputs.size() != 2) {
        return {};
    }

    const Value * norm_weight = nullptr;
    for (ValueId input : mul_node->inputs) {
        if (input != rms_node->output) {
            norm_weight = graph_value(context.graph, input);
        }
    }
    const Value * output = graph_value(context.graph, mul_node->output);
    if (norm_weight == nullptr || output == nullptr || norm_weight->type != GGML_TYPE_F32 ||
        output->type != GGML_TYPE_F32 || !norm_weight->contiguous || !output->contiguous ||
        !is_shape(*norm_weight, kRoutedFfnInputSize, 1, 1, 1) || !same_shape(*output, residual)) {
        return {};
    }

    match.rms_node    = rms_node;
    match.mul_node    = mul_node;
    match.norm_weight = norm_weight;
    match.output      = output;
    return match;
}

static bool append_node_if_uncovered(const DispatchMatchContext &     context,
                                     const GraphNode *                node,
                                     std::vector<const GraphNode *> & nodes) {
    size_t index = 0;
    if (node == nullptr || !context.graph.index().node_index(node, index) || index >= context.covered_nodes.size() ||
        context.covered_nodes[index]) {
        return false;
    }
    for (const GraphNode * existing : nodes) {
        if (existing == node) {
            return true;
        }
    }
    nodes.push_back(node);
    return true;
}

static bool node_is_covered(const DispatchMatchContext & context, const GraphNode * node) {
    size_t index = 0;
    return node != nullptr && context.graph.index().node_index(node, index) && index < context.covered_nodes.size() &&
           context.covered_nodes[index];
}

static bool residual_input_is_safe_for_in_place(const DispatchMatchContext & context,
                                                const WeightedReduceMatch &  match) {
    if (match.residual_input == nullptr || match.residual == nullptr) {
        return false;
    }
    for (const GraphNode * consumer : context.graph.index().consumers(match.residual_input->id)) {
        if (consumer == match.residual || node_is_covered(context, consumer)) {
            continue;
        }
        return false;
    }
    return true;
}

static const Value * find_qwen_route_weights_for_route_ids(const Graph & graph,
                                                           ValueId       route_ids,
                                                           int64_t       token_count) {
    const GraphNode * get_rows = find_single_consumer_with_op(graph, route_ids, GGML_OP_GET_ROWS);
    if (get_rows == nullptr || get_rows->inputs.size() != 2) {
        return nullptr;
    }
    const Value * selected = graph_value(graph, get_rows->output);
    if (selected == nullptr || selected->type != GGML_TYPE_F32 ||
        !is_shape(*selected, 1, kRoutedFfnRouteCount, token_count, 1)) {
        return nullptr;
    }
    const GraphNode * reshape      = find_single_consumer_with_op(graph, get_rows->output, GGML_OP_RESHAPE);
    const Value *     flat_weights = reshape == nullptr ? nullptr : graph_value(graph, reshape->output);
    if (flat_weights == nullptr || flat_weights->type != GGML_TYPE_F32 ||
        !is_shape(*flat_weights, kRoutedFfnRouteCount, token_count, 1, 1)) {
        return nullptr;
    }
    const GraphNode * sum_rows = find_single_consumer_with_op(graph, reshape->output, GGML_OP_SUM_ROWS);
    const GraphNode * clamp =
        sum_rows == nullptr ? nullptr : find_single_consumer_with_op(graph, sum_rows->output, GGML_OP_CLAMP);
    const GraphNode * div =
        clamp == nullptr ? nullptr : find_single_consumer_with_op(graph, clamp->output, GGML_OP_DIV);
    const GraphNode * output_reshape =
        div == nullptr ? nullptr : find_single_consumer_with_op(graph, div->output, GGML_OP_RESHAPE);
    const Value * weights = output_reshape == nullptr ? nullptr : graph_value(graph, output_reshape->output);
    if (weights == nullptr || weights->type != GGML_TYPE_F32 ||
        !is_shape(*weights, 1, kRoutedFfnRouteCount, token_count, 1)) {
        return nullptr;
    }
    return weights;
}

static WeightedReduceMatch match_routed_ffn_down_weighted_reduce_topology(const DispatchMatchContext & context,
                                                                          const GraphNode *            weighted,
                                                                          const Value *                routed_output,
                                                                          const Value *                route_weights) {
    WeightedReduceMatch match;
    if (weighted == nullptr || weighted->op != GGML_OP_MUL || weighted->inputs.size() != 2 ||
        routed_output == nullptr || route_weights == nullptr || !context.graph.has_index()) {
        return match;
    }
    if (!node_has_input_or_alias(context.graph, *weighted, routed_output->id) ||
        !node_has_input_or_alias(context.graph, *weighted, route_weights->id)) {
        return {};
    }
    const Value * weighted_output = graph_value(context.graph, weighted->output);
    if (!is_routed_ffn_down_output(*routed_output, routed_output->ne[2]) || route_weights->type != GGML_TYPE_F32 ||
        !route_weights->contiguous || !is_shape(*route_weights, 1, kRoutedFfnRouteCount, routed_output->ne[2], 1) ||
        weighted_output == nullptr || !same_shape(*weighted_output, *routed_output)) {
        return {};
    }

    const int64_t token_count = routed_output->ne[2];
    if (!is_llm_supported_query_length(kRoutedFfnProfile, token_count)) {
        return {};
    }

    std::vector<const GraphNode *> views =
        layout_alias_consumers_with_op(context.graph, weighted->output, GGML_OP_VIEW);
    if (views.size() != kRoutedFfnRouteCount) {
        return {};
    }

    std::set<int32_t>              routed_values;
    std::vector<const GraphNode *> owned_views;
    for (const GraphNode * view : views) {
        const Value * value = view == nullptr ? nullptr : graph_value(context.graph, view->output);
        if (value == nullptr || value->type != GGML_TYPE_F32 ||
            !is_shape(*value, kRoutedFfnInputSize, token_count, 1, 1) ||
            !append_node_if_uncovered(context, view, owned_views)) {
            return {};
        }
        routed_values.insert(view->output.value);
    }

    std::vector<const GraphNode *> reductions;
    bool                           changed = true;
    while (changed) {
        changed                        = false;
        const std::set<int32_t> values = routed_values;
        for (int32_t value : values) {
            for (const GraphNode * add : find_consumers_with_op(context.graph, ValueId(value), GGML_OP_ADD)) {
                if (add == nullptr || add->inputs.size() != 2) {
                    continue;
                }
                bool already_owned = false;
                for (const GraphNode * reduction : reductions) {
                    if (reduction == add) {
                        already_owned = true;
                        break;
                    }
                }
                if (already_owned) {
                    continue;
                }
                bool all_routed = true;
                for (ValueId input : add->inputs) {
                    all_routed = all_routed && routed_values.count(input.value) != 0;
                }
                if (!all_routed) {
                    continue;
                }
                const Value * output = graph_value(context.graph, add->output);
                if (output == nullptr || output->type != GGML_TYPE_F32 ||
                    !is_shape(*output, kRoutedFfnInputSize, token_count, 1, 1) ||
                    !append_node_if_uncovered(context, add, reductions)) {
                    return {};
                }
                routed_values.insert(add->output.value);
                changed = true;
            }
        }
    }

    const GraphNode * residual       = nullptr;
    const Value *     residual_input = nullptr;
    for (int32_t value : routed_values) {
        for (const GraphNode * add : find_consumers_with_op(context.graph, ValueId(value), GGML_OP_ADD)) {
            if (add == nullptr || add->inputs.size() != 2) {
                continue;
            }
            bool is_reduction = false;
            for (const GraphNode * reduction : reductions) {
                if (reduction == add) {
                    is_reduction = true;
                    break;
                }
            }
            if (is_reduction) {
                continue;
            }
            int           routed_input_count = 0;
            const Value * non_routed_input   = nullptr;
            for (ValueId input : add->inputs) {
                if (routed_values.count(input.value) != 0) {
                    ++routed_input_count;
                } else {
                    non_routed_input = graph_value(context.graph, input);
                }
            }
            if (routed_input_count != 1 || non_routed_input == nullptr || residual != nullptr) {
                return {};
            }
            residual       = add;
            residual_input = non_routed_input;
        }
    }
    if (residual == nullptr || reductions.size() + 1 != views.size()) {
        return {};
    }
    const Value * output = graph_value(context.graph, residual->output);
    if (output == nullptr || residual_input == nullptr || output->type != GGML_TYPE_F32 ||
        residual_input->type != GGML_TYPE_F32 || !is_shape(*output, kRoutedFfnInputSize, token_count, 1, 1) ||
        !same_shape(*output, *residual_input) || output->byte_count != residual_input->byte_count ||
        !output->contiguous || !residual_input->contiguous) {
        return {};
    }

    match.route_weights  = route_weights;
    match.routed_output  = routed_output;
    match.residual_input = residual_input;
    match.output         = output;
    match.weighted_node  = weighted;
    match.views          = std::move(owned_views);
    match.reductions     = std::move(reductions);
    match.residual       = residual;
    match.next_rmsnorm   = match_qwen_weighted_reduce_next_rmsnorm(context, *output);
    match.token_count    = token_count;
    return match;
}

static WeightedReduceMatch match_routed_ffn_down_weighted_reduce(const DispatchMatchContext & context) {
    WeightedReduceMatch match;
    const GraphNode *   weighted = context.root_node;
    if (weighted == nullptr || weighted->op != GGML_OP_MUL || weighted->inputs.size() != 2 ||
        !context.graph.has_index()) {
        return match;
    }

    const Value * routed_output = nullptr;
    const Value * route_weights = nullptr;
    for (ValueId input : weighted->inputs) {
        const Value * value = graph_value(context.graph, input);
        if (value == nullptr) {
            return {};
        }
        if (is_routed_ffn_down_output(*value, value->ne[2])) {
            routed_output = value;
        } else if (value->type == GGML_TYPE_F32 && value->contiguous &&
                   is_shape(*value, 1, kRoutedFfnRouteCount, value->ne[2], 1)) {
            route_weights = value;
        }
    }
    match = match_routed_ffn_down_weighted_reduce_topology(context, weighted, routed_output, route_weights);
    if (!match.topology_matched()) {
        return {};
    }
    const int64_t                     token_count = match.token_count;
    const CommandPlanAlternateValue * routed_alternate =
        find_alternate_value(context.plan, routed_output->id, GGML_TYPE_F16, f16_routed_down_output_size(token_count));
    if (routed_alternate == nullptr) {
        return {};
    }

    bool known_route_weights = false;
    for (const CommandPlanMoeRoutingBundle & bundle : context.plan.metadata.moe_routing_bundles()) {
        if (bundle.route_weights == route_weights->id &&
            bundle_matches_moe_routing(bundle, bundle.route_ids, token_count)) {
            known_route_weights = true;
            break;
        }
    }
    if (!known_route_weights) {
        return {};
    }

    match.routed_alternate = routed_alternate;
    return match;
}

static bool match_routed_ffn_gate_up_swiglu_q4k_f16_wmma_dispatch(const DispatchMatchContext & context,
                                                                  DispatchMatch &              dispatch_match) {
    const RoutedGateUpMatch match = match_routed_ffn_gate_up_swiglu(context);
    if (!match.matched()) {
        return false;
    }

    const ValueId f16_output(context.next_plan_value.value);
    const size_t  f16_output_bytes = f16_gate_up_output_size(match.token_count);
    dispatch_match.transients.push_back(
        { f16_output, kRoutedFfnF16GateUpOutputName, f16_output_bytes, kRoutedFfnPlanTransientAlignment });
    Status metadata_status;
    if (!dispatch_match.metadata.append_alternate_value(
            { match.glu_output->id, f16_output, GGML_TYPE_F16, f16_output_bytes, kRoutedFfnF16GateUpOutputName },
            metadata_status)) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kQwenRoutedGateUpSwiGLUQ4KF16WmmaKernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.input_size",
                                               to_config_value(kRoutedFfnInputSize));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.expert_count",
                                               to_config_value(kRoutedFfnExpertCount));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.route_count",
                                               to_config_value(kRoutedFfnRouteCount));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.output_size",
                                               to_config_value(kRoutedFfnExpertHiddenSize));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity", to_config_value(match.token_count));

    dispatch.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    dispatch.bindings.push_back(
        { match.routing_bundle->expert_table, 0, match.routing_bundle->expert_table_byte_count });
    dispatch.bindings.push_back(
        { match.routing_bundle->partition_table, 0, match.routing_bundle->partition_table_byte_count });
    dispatch.bindings.push_back({ match.gate_weight->id, 0, match.gate_weight->byte_count });
    dispatch.bindings.push_back({ match.up_weight->id, 0, match.up_weight->byte_count });
    dispatch.bindings.push_back({ f16_output, 0, f16_output_bytes });

    if (!append_covered_node(context, match.gate_node, dispatch_match) ||
        !append_covered_node(context, match.up_node, dispatch_match) ||
        !append_covered_node(context, match.glu_node, dispatch_match)) {
        return false;
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_decode_routed_ffn_gate_up_swiglu_q4k_q8_dispatch(const DispatchMatchContext & context,
                                                                   DispatchMatch &              dispatch_match) {
    const DecodeRoutedGateUpMatch match = match_decode_routed_ffn_gate_up_swiglu(context);
    if (!match.matched()) {
        return false;
    }

    const size_t q8_output_bytes =
        q8_1_x4_byte_count(match.token_count * kRoutedFfnRouteCount, kRoutedFfnExpertHiddenSize);
    const ValueId q8_output           = match.publish_q8 ? ValueId(context.next_plan_value.value) : ValueId();
    const ValueId completion_counters = match.publish_q8 ? ValueId(context.next_plan_value.value + 1) : ValueId();

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(match.publish_q8 ? kQwenRoutedGateUpSwiGLUQ4KQ8NextQ8Kernel :
                                                                    kQwenRoutedGateUpSwiGLUQ4KQ8Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.integer_parameters.emplace("route_count", kRoutedFfnRouteCount);
    dispatch.kernel.integer_parameters.emplace("route_stride", match.route_stride);
    dispatch.kernel.integer_parameters.emplace("expert_count", kRoutedFfnExpertCount);
    dispatch.kernel.integer_parameters.emplace("output_size", kRoutedFfnExpertHiddenSize);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.input_size",
                                               to_config_value(kRoutedFfnInputSize));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.expert_count",
                                               to_config_value(kRoutedFfnExpertCount));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.route_count",
                                               to_config_value(kRoutedFfnRouteCount));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.output_size",
                                               to_config_value(kRoutedFfnExpertHiddenSize));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity", to_config_value(match.token_count));

    const size_t route_id_length = static_cast<size_t>(match.token_count * match.route_stride) * sizeof(int32_t);
    dispatch.bindings.push_back({ match.input_alternate->alternate_value, 0, match.input_alternate->byte_count });
    dispatch.bindings.push_back({ match.route_ids->id, 0, route_id_length });
    dispatch.bindings.push_back({ match.gate_weight->id, 0, match.gate_weight->byte_count });
    dispatch.bindings.push_back({ match.up_weight->id, 0, match.up_weight->byte_count });
    dispatch.bindings.push_back({ match.glu_output->id, 0, match.glu_output->byte_count });
    if (match.publish_q8) {
        dispatch.bindings.push_back(
            { completion_counters, 0, gate_up_completion_counter_count(match.token_count) * sizeof(int32_t) });
        dispatch.bindings.push_back({ q8_output, 0, q8_output_bytes });
        dispatch_match.completion_counter_requests.push_back({
            completion_counters,
            "qwen.decode.moe.gate_up_completion_counters",
            gate_up_completion_counter_count(match.token_count),
        });
        dispatch_match.transients.push_back(
            { q8_output, kRoutedFfnQ8GateUpOutputName, q8_output_bytes, kRoutedFfnPlanTransientAlignment });
        Status metadata_status;
        if (!dispatch_match.metadata.append_alternate_value(
                { match.glu_output->id, q8_output, GGML_TYPE_Q8_1, q8_output_bytes, kRoutedFfnQ8GateUpOutputName },
                metadata_status)) {
            dispatch_match.status.append(metadata_status);
            return false;
        }
    }

    if (!append_covered_node(context, match.gate_node, dispatch_match) ||
        !append_covered_node(context, match.up_node, dispatch_match) ||
        !append_covered_node(context, match.glu_node, dispatch_match)) {
        return false;
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static DecodeRoutedDownMatch match_decode_routed_ffn_down_next_q8(const DispatchMatchContext & context) {
    DecodeRoutedDownMatch match;
    const GraphNode *     root = context.root_node;
    if (root == nullptr || root->op != GGML_OP_MUL_MAT_ID || root->inputs.size() != 3 || !context.graph.has_index()) {
        return match;
    }

    const Value * weight      = graph_value(context.graph, root->inputs[0]);
    const Value * input       = graph_value(context.graph, root->inputs[1]);
    const Value * route_ids   = graph_value(context.graph, root->inputs[2]);
    const Value * root_output = graph_value(context.graph, root->output);
    if (weight == nullptr || input == nullptr || route_ids == nullptr || root_output == nullptr ||
        !is_routed_ffn_down_weight(*weight) || !is_routed_ffn_projection_output(*input, 1) ||
        !is_routed_ffn_down_output(*root_output, 1) || route_ids->type != GGML_TYPE_I32 ||
        route_ids->nb[0] != sizeof(int32_t) || route_ids->nb[1] % sizeof(int32_t) != 0 ||
        !is_shape(*route_ids, kRoutedFfnRouteCount, 1, 1, 1)) {
        return {};
    }

    const int64_t                     route_stride = static_cast<int64_t>(route_ids->nb[1] / sizeof(int32_t));
    const GraphNode *                 glu_node     = producer_with_op(context.graph, input->id, GGML_OP_GLU);
    const GraphNode *                 gate_node    = glu_node == nullptr || glu_node->inputs.size() != 2 ?
                                                         nullptr :
                                                         producer_with_op(context.graph, glu_node->inputs[0], GGML_OP_MUL_MAT_ID);
    const Value *                     gate_input   = gate_node == nullptr || gate_node->inputs.size() != 3 ?
                                                         nullptr :
                                                         graph_value(context.graph, gate_node->inputs[1]);
    const CommandPlanAlternateValue * gate_input_q8 =
        gate_input == nullptr ? nullptr :
                                find_alternate_value(context.graph, context.plan, gate_input->id, GGML_TYPE_Q8_1,
                                                     q8_1_x4_byte_count(1, kRoutedFfnInputSize));
    if (gate_input_q8 == nullptr) {
        return {};
    }

    const Value *     route_weights = find_qwen_route_weights_for_route_ids(context.graph, route_ids->id, 1);
    const GraphNode * weighted =
        find_single_consumer_with_op_through_layout_aliases(context.graph, root_output->id, GGML_OP_MUL);
    WeightedReduceMatch reduce =
        match_routed_ffn_down_weighted_reduce_topology(context, weighted, root_output, route_weights);
    if (!reduce.topology_matched() || !reduce.next_rmsnorm.matched() ||
        !residual_input_is_safe_for_in_place(context, reduce)) {
        return {};
    }

    const bool                        input_is_q8 = weight->type == GGML_TYPE_Q4_K;
    const CommandPlanAlternateValue * input_alternate =
        input_is_q8 ? find_alternate_value(context.graph, context.plan, input->id, GGML_TYPE_Q8_1,
                                           q8_1_x4_byte_count(kRoutedFfnRouteCount, kRoutedFfnExpertHiddenSize)) :
                      nullptr;
    if (input_is_q8 && input_alternate == nullptr) {
        return {};
    }

    match.input_graph_value = input;
    match.input_alternate   = input_alternate;
    match.weight            = weight;
    match.output            = reduce.output;
    match.route_ids         = route_ids;
    match.reduce            = std::move(reduce);
    match.kernel =
        weight->type == GGML_TYPE_Q4_K ? kQwenRoutedDownQ4KQ8NextQ8Kernel : kQwenRoutedDownQ6KF32Wave64NextQ8Kernel;
    match.token_count  = 1;
    match.route_stride = route_stride;
    match.input_is_q8  = input_is_q8;
    return match;
}

static bool match_decode_routed_ffn_down_next_q8_dispatch(const DispatchMatchContext & context,
                                                          DispatchMatch &              dispatch_match) {
    const DecodeRoutedDownMatch match = match_decode_routed_ffn_down_next_q8(context);
    if (!match.matched()) {
        return false;
    }

    const ValueId completion_counter_value(context.next_plan_value.value);
    const ValueId q8_output(context.next_plan_value.value + 1);
    const size_t  q8_output_bytes = q8_1_x4_byte_count(match.token_count, kRoutedFfnInputSize);

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(match.kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.integer_parameters.emplace("input_size", kRoutedFfnExpertHiddenSize);
    dispatch.kernel.integer_parameters.emplace("route_count", kRoutedFfnRouteCount);
    dispatch.kernel.integer_parameters.emplace("route_id_stride", match.route_stride);
    dispatch.kernel.integer_parameters.emplace("expert_count", kRoutedFfnExpertCount);
    dispatch.kernel.integer_parameters.emplace("output_size", kRoutedFfnInputSize);
    add_routed_down_compile_parameters(dispatch, match.token_count);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.hidden_size", to_config_value(kRoutedFfnInputSize));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.rms_epsilon", "0.000001");

    const size_t route_id_length = static_cast<size_t>(match.token_count * match.route_stride) * sizeof(int32_t);
    if (match.input_is_q8) {
        dispatch.bindings.push_back({ match.input_alternate->alternate_value, 0, match.input_alternate->byte_count });
    } else {
        dispatch.bindings.push_back({ match.input_graph_value->id, 0, match.input_graph_value->byte_count });
    }
    dispatch.bindings.push_back({ match.route_ids->id, 0, route_id_length });
    dispatch.bindings.push_back({ match.reduce.route_weights->id, 0, match.reduce.route_weights->byte_count });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    dispatch.bindings.push_back(
        { match.reduce.next_rmsnorm.norm_weight->id, 0, match.reduce.next_rmsnorm.norm_weight->byte_count });
    dispatch.bindings.push_back({ completion_counter_value, 0, sizeof(int32_t) });
    dispatch.bindings.push_back({ q8_output, 0, q8_output_bytes });

    dispatch_match.value_aliases.push_back({ match.reduce.residual_input->id, match.output->id });
    dispatch_match.completion_counter_requests.push_back({
        completion_counter_value,
        "qwen.decode.moe.routed_down_completion_counter",
        1,
    });
    dispatch_match.transients.push_back(
        { q8_output, kRoutedFfnQ8HiddenOutputName, q8_output_bytes, kRoutedFfnPlanTransientAlignment });
    Status metadata_status;
    if (!dispatch_match.metadata.append_alternate_value(
            { match.reduce.next_rmsnorm.output->id, q8_output, GGML_TYPE_Q8_1, q8_output_bytes,
              kRoutedFfnQ8HiddenOutputName },
            metadata_status)) {
        dispatch_match.status.append(metadata_status);
        return false;
    }

    if (!append_covered_node(context, context.root_node, dispatch_match) ||
        !append_covered_node(context, match.reduce.weighted_node, dispatch_match)) {
        return false;
    }
    for (const GraphNode * view : match.reduce.views) {
        if (!append_covered_node(context, view, dispatch_match)) {
            return false;
        }
    }
    for (const GraphNode * reduction : match.reduce.reductions) {
        if (!append_covered_node(context, reduction, dispatch_match)) {
            return false;
        }
    }
    if (!append_covered_node(context, match.reduce.residual, dispatch_match) ||
        !append_covered_node(context, match.reduce.next_rmsnorm.rms_node, dispatch_match) ||
        !append_covered_node(context, match.reduce.next_rmsnorm.mul_node, dispatch_match)) {
        return false;
    }

    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool build_routed_ffn_down_grouped_dispatch(const DispatchMatchContext & context,
                                                   DispatchMatch &              dispatch_match,
                                                   KernelCatalogRef             expected_kernel) {
    const RoutedDownMatch match = match_routed_ffn_down_grouped(context);
    if (!match.matched() || match.kernel.id != expected_kernel.id) {
        return false;
    }

    const ValueId f16_output(context.next_plan_value.value);
    const size_t  f16_output_bytes = f16_routed_down_output_size(match.token_count);
    dispatch_match.transients.push_back(
        { f16_output, kRoutedFfnF16RoutedDownOutputName, f16_output_bytes, kRoutedFfnPlanTransientAlignment });
    Status metadata_status;
    if (!dispatch_match.metadata.append_alternate_value(
            { match.output->id, f16_output, GGML_TYPE_F16, f16_output_bytes, kRoutedFfnF16RoutedDownOutputName },
            metadata_status)) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(match.kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    add_routed_down_compile_parameters(dispatch, match.token_count);
    dispatch.bindings.push_back({ match.input_alternate->alternate_value, 0, match.input_alternate->byte_count });
    dispatch.bindings.push_back(
        { match.routing_bundle->expert_table, 0, match.routing_bundle->expert_table_byte_count });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ f16_output, 0, f16_output_bytes });

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_routed_ffn_down_weighted_reduce_dispatch(const DispatchMatchContext & context,
                                                           DispatchMatch &              dispatch_match) {
    const WeightedReduceMatch match = match_routed_ffn_down_weighted_reduce(context);
    if (!match.matched()) {
        return false;
    }
    const bool use_next_rmsnorm = match.next_rmsnorm.matched() && match.output->kind == ValueKind::Transient &&
                                  residual_input_is_safe_for_in_place(context, match);

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(use_next_rmsnorm ? kQwenRoutedDownWeightedReduceNextRmsNormF32Kernel :
                                                                    kQwenRoutedDownWeightedReduceF16F32Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    add_routed_down_compile_parameters(dispatch, match.token_count);
    if (use_next_rmsnorm) {
        dispatch_match.value_aliases.push_back({ match.residual_input->id, match.output->id });
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.hidden_size", to_config_value(kRoutedFfnInputSize));
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.rms_epsilon", "0.000001");
        dispatch.bindings.push_back({ match.route_weights->id, 0, match.route_weights->byte_count });
        dispatch.bindings.push_back({ match.routed_alternate->alternate_value, 0, match.routed_alternate->byte_count });
        dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
        dispatch.bindings.push_back(
            { match.next_rmsnorm.norm_weight->id, 0, match.next_rmsnorm.norm_weight->byte_count });
        dispatch.bindings.push_back({ match.next_rmsnorm.output->id, 0, match.next_rmsnorm.output->byte_count });
    } else {
        dispatch.bindings.push_back({ match.route_weights->id, 0, match.route_weights->byte_count });
        dispatch.bindings.push_back({ match.routed_alternate->alternate_value, 0, match.routed_alternate->byte_count });
        dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    }

    if (!append_covered_node(context, match.weighted_node, dispatch_match)) {
        return false;
    }
    for (const GraphNode * view : match.views) {
        if (!append_covered_node(context, view, dispatch_match)) {
            return false;
        }
    }
    for (const GraphNode * reduction : match.reductions) {
        if (!append_covered_node(context, reduction, dispatch_match)) {
            return false;
        }
    }
    if (!append_covered_node(context, match.residual, dispatch_match)) {
        return false;
    }
    if (use_next_rmsnorm && (!append_covered_node(context, match.next_rmsnorm.rms_node, dispatch_match) ||
                             !append_covered_node(context, match.next_rmsnorm.mul_node, dispatch_match))) {
        return false;
    }

    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_routed_ffn_down_q4k_f16_wmma_grouped_dispatch(const DispatchMatchContext & context,
                                                                DispatchMatch &              dispatch_match) {
    return build_routed_ffn_down_grouped_dispatch(context, dispatch_match, kQwenRoutedDownQ4KF16WmmaGroupedKernel);
}

static bool match_routed_ffn_down_q6k_f16_wmma_grouped_dispatch(const DispatchMatchContext & context,
                                                                DispatchMatch &              dispatch_match) {
    return build_routed_ffn_down_grouped_dispatch(context, dispatch_match, kQwenRoutedDownQ6KF16WmmaGroupedKernel);
}

}  // namespace

void register_routed_ffn_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "llm.routed_ffn.decode_gate_up_swiglu_q4k_q8",
        GGML_OP_MUL_MAT_ID,
        DispatchMatchKind::Fused,
        1200,
        DispatchSource::Llm,
        match_decode_routed_ffn_gate_up_swiglu_q4k_q8_dispatch,
    });
    registry.add({
        "llm.routed_ffn.decode_down_next_q8",
        GGML_OP_MUL_MAT_ID,
        DispatchMatchKind::Fused,
        1150,
        DispatchSource::Llm,
        match_decode_routed_ffn_down_next_q8_dispatch,
    });
    registry.add({
        "llm.routed_ffn.gate_up_swiglu_q4k_f16_wmma",
        GGML_OP_MUL_MAT_ID,
        DispatchMatchKind::Fused,
        1000,
        DispatchSource::Llm,
        match_routed_ffn_gate_up_swiglu_q4k_f16_wmma_dispatch,
    });
    registry.add({
        "llm.routed_ffn.down_q4k_f16_wmma_grouped",
        GGML_OP_MUL_MAT_ID,
        DispatchMatchKind::Fused,
        900,
        DispatchSource::Llm,
        match_routed_ffn_down_q4k_f16_wmma_grouped_dispatch,
    });
    registry.add({
        "llm.routed_ffn.down_q6k_f16_wmma_grouped",
        GGML_OP_MUL_MAT_ID,
        DispatchMatchKind::Fused,
        900,
        DispatchSource::Llm,
        match_routed_ffn_down_q6k_f16_wmma_grouped_dispatch,
    });
    registry.add({
        "llm.routed_ffn.down_weighted_reduce",
        GGML_OP_MUL,
        DispatchMatchKind::Fused,
        800,
        DispatchSource::Llm,
        match_routed_ffn_down_weighted_reduce_dispatch,
    });
}

}  // namespace ggml::hrx
