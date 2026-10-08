#include "dispatch-moe-router.h"

#include "dispatch-llm-shapes.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kQwenRouterTop8F32Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_router_top8_f32");
static constexpr KernelCatalogRef kQwenRouterProjectionTop8FusedDecodeF32Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_router_projection_top8_fused_decode_f32");
static constexpr KernelCatalogRef kQwenRouterProjectionF32FourRowWave32Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_router_projection_f32_four_row_wave32");
static constexpr KernelCatalogRef kMoeBuildExpertTableKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_moe_build_expert_table");
static constexpr KernelCatalogRef kMoeBuildExpertPartitionTableKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_moe_build_expert_partition_table");
static constexpr KernelCatalogRef kQwenBuildExpertTablePartitionPrefill512Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_build_expert_table_partition_prefill_512");

static constexpr const LlmMoeDispatchProfile & kMoeRouterProfile                = kActiveLlmMoeDispatchProfile;
static constexpr size_t                        kMoeRouterPlanTransientAlignment = 256;

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool nearly_equal(float lhs, float rhs) {
    return std::fabs(lhs - rhs) <= 1.0e-12f;
}

static const GraphNode * find_consumer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    const std::vector<const GraphNode *> consumers = consumers_with_op_through_layout_aliases(graph, value, op);
    return consumers.empty() ? nullptr : consumers.front();
}

static const GraphNode * find_single_consumer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    const std::vector<const GraphNode *> consumers = consumers_with_op_through_layout_aliases(graph, value, op);
    return consumers.size() == 1 ? consumers.front() : nullptr;
}

static const GraphNode * find_consumer_with_op_and_input(const Graph & graph,
                                                         ValueId       value,
                                                         ggml_op       op,
                                                         ValueId       input) {
    for (const GraphNode * consumer : consumers_with_op_through_layout_aliases(graph, value, op)) {
        if (consumer != nullptr && node_has_input_or_alias(graph, *consumer, input)) {
            return consumer;
        }
    }
    return nullptr;
}

static bool is_shape(const Value & value, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    return value.ne[0] == ne0 && value.ne[1] == ne1 && value.ne[2] == ne2 && value.ne[3] == ne3;
}

static bool is_2d(const Value & value) {
    return value.ne[0] > 0 && value.ne[1] > 0 && value.ne[2] == 1 && value.ne[3] == 1;
}

static bool same_shape(const Value & lhs, const Value & rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i]) {
            return false;
        }
    }
    return true;
}

static bool is_supported_expert_count(int64_t expert_count) {
    return expert_count >= 32 && expert_count <= 512 && expert_count % 32 == 0;
}

static bool is_supported_route_count(int64_t route_count, int64_t expert_count) {
    return route_count >= 1 && route_count <= 32 && route_count <= expert_count;
}

static bool is_supported_route_stride(int64_t route_stride, int64_t route_count, int64_t expert_count) {
    return route_stride >= route_count && route_stride <= expert_count;
}

static bool is_default_scale_softmax(const GraphNode & node) {
    const SoftMaxParams * params = op_params_as<SoftMaxParams>(node.params);
    return params != nullptr && nearly_equal(params->scale, 1.0f) && nearly_equal(params->max_bias, 0.0f);
}

static bool is_descending_argsort(const GraphNode & node) {
    const ArgsortParams * params = op_params_as<ArgsortParams>(node.params);
    return params != nullptr && params->order == GGML_SORT_ORDER_DESC;
}

static bool is_topk_normalization_clamp(const GraphNode & node) {
    const ClampParams * params = op_params_as<ClampParams>(node.params);
    return params != nullptr && params->min >= 0.0f && params->min <= 1.0e-4f && std::isinf(params->max) &&
           params->max > 0.0f;
}

static std::string to_config_value(int64_t value) {
    return std::to_string(value);
}

static size_t expert_table_size(int64_t token_count, int64_t expert_count) {
    return static_cast<size_t>(expert_count + expert_count * token_count) * sizeof(int32_t);
}

static size_t partition_table_size(int64_t token_count, int64_t route_count, int64_t expert_count) {
    const int64_t assignment_count           = token_count * route_count;
    const int64_t assignment_partition_count = (assignment_count + 31) / 32;
    return static_cast<size_t>(1 + assignment_partition_count + expert_count) * sizeof(int32_t);
}

static std::string value_summary(const Graph & graph, const Value * value) {
    if (value == nullptr) {
        return "missing";
    }
    std::ostringstream stream;
    stream << value->id.value << ":" << ggml_type_name(value->type) << "[" << value->ne[0] << "," << value->ne[1] << ","
           << value->ne[2] << "," << value->ne[3] << "] nb=[" << value->nb[0] << "," << value->nb[1] << ","
           << value->nb[2] << "," << value->nb[3] << "]";
    const GraphNode * producer = graph.index().producer(value->id);
    if (producer != nullptr) {
        stream << "<-" << ggml_op_name(producer->op);
    }
    return stream.str();
}

static bool is_moe_router_candidate_root(const Graph & graph, const GraphNode * softmax_node) {
    if (softmax_node == nullptr || softmax_node->op != GGML_OP_SOFT_MAX || softmax_node->inputs.size() != 1 ||
        !graph.has_index() || !is_default_scale_softmax(*softmax_node)) {
        return false;
    }
    const Value * logits = graph_value(graph, softmax_node->inputs[0]);
    const Value * probs  = graph_value(graph, softmax_node->output);
    if (logits == nullptr || probs == nullptr || logits->type != GGML_TYPE_F32 || probs->type != GGML_TYPE_F32) {
        return false;
    }
    return find_consumer_with_op(graph, softmax_node->output, GGML_OP_RESHAPE) != nullptr &&
           find_consumer_with_op(graph, softmax_node->output, GGML_OP_ARGSORT) != nullptr;
}

static void log_router_reject(Status *            status,
                              const Graph &       graph,
                              const GraphNode *   node,
                              const std::string & reason) {
    if (status == nullptr || !is_moe_router_candidate_root(graph, node)) {
        return;
    }
    const Value * logits = node == nullptr || node->inputs.empty() ? nullptr : graph_value(graph, node->inputs[0]);
    const Value * probs  = node == nullptr ? nullptr : graph_value(graph, node->output);
    status->log("MoE router top-k matcher rejected node: %s logits=%s probs=%s", reason.c_str(),
                value_summary(graph, logits).c_str(), value_summary(graph, probs).c_str());
}

static bool append_covered_node(const DispatchMatchContext & context, const GraphNode * node, DispatchMatch & match) {
    return append_covered_node_index_once(context.graph, context.covered_nodes, node, match.covered_nodes);
}

struct RouterTop8Match {
    const Value * logits        = nullptr;
    const Value * route_ids     = nullptr;
    const Value * route_weights = nullptr;
    int64_t       token_count   = 0;
    int64_t       expert_count  = 0;
    int64_t       route_count   = 0;
    int64_t       route_stride  = 0;

    bool matched() const {
        return logits != nullptr && route_ids != nullptr && route_weights != nullptr && token_count > 0 &&
               is_supported_expert_count(expert_count) && is_supported_route_count(route_count, expert_count) &&
               is_supported_route_stride(route_stride, route_count, expert_count);
    }
};

static bool supports_fused_prefill_expert_table_partition(const RouterTop8Match & router_match) {
    // Matches the reference prefill recipe gate; q=1 uses decode routing paths.
    return is_llm_prefill_512_query_length(kMoeRouterProfile, router_match.token_count) &&
           router_match.route_count == kMoeRouterProfile.route_count &&
           router_match.route_stride == router_match.route_count &&
           router_match.expert_count == kMoeRouterProfile.expert_count;
}

static RouterTop8Match match_moe_router_top8(const Graph & graph, const GraphNode * softmax_node, Status * status) {
    RouterTop8Match match;
    if (softmax_node == nullptr || softmax_node->op != GGML_OP_SOFT_MAX || softmax_node->inputs.size() != 1 ||
        !graph.has_index() || !is_default_scale_softmax(*softmax_node)) {
        return match;
    }

    const Value * logits = graph_value(graph, softmax_node->inputs[0]);
    const Value * probs  = graph_value(graph, softmax_node->output);
    if (logits == nullptr || probs == nullptr || logits->type != GGML_TYPE_F32 || probs->type != GGML_TYPE_F32 ||
        !same_shape(*logits, *probs) || !logits->contiguous || !probs->contiguous) {
        log_router_reject(status, graph, softmax_node, "logits/probs must be same contiguous F32 shape");
        return {};
    }
    const int64_t expert_count = logits->ne[0];
    const int64_t token_count  = logits->ne[1];
    if (!is_shape(*logits, expert_count, token_count, 1, 1) || !is_supported_expert_count(expert_count) ||
        !is_llm_supported_query_length(kMoeRouterProfile, token_count)) {
        log_router_reject(status, graph, softmax_node, "unsupported logits expert/token shape");
        return {};
    }

    const GraphNode * probs_reshape = find_consumer_with_op(graph, softmax_node->output, GGML_OP_RESHAPE);
    const GraphNode * argsort       = find_consumer_with_op(graph, softmax_node->output, GGML_OP_ARGSORT);
    if (probs_reshape == nullptr || argsort == nullptr || argsort->inputs.size() != 1 ||
        !is_descending_argsort(*argsort)) {
        log_router_reject(status, graph, softmax_node, "missing probability reshape or descending argsort");
        return {};
    }

    const Value * probs_reshaped = graph_value(graph, probs_reshape->output);
    const Value * argsort_output = graph_value(graph, argsort->output);
    if (probs_reshaped == nullptr || argsort_output == nullptr || probs_reshaped->type != GGML_TYPE_F32 ||
        argsort_output->type != GGML_TYPE_I32 || !is_shape(*probs_reshaped, 1, expert_count, token_count, 1) ||
        !is_shape(*argsort_output, expert_count, token_count, 1, 1)) {
        log_router_reject(status, graph, softmax_node, "probability reshape or argsort output shape is incompatible");
        return {};
    }

    const GraphNode * topk_view = find_consumer_with_op(graph, argsort->output, GGML_OP_VIEW);
    if (topk_view == nullptr || topk_view->inputs.size() != 1) {
        return {};
    }
    const Value * route_ids   = graph_value(graph, topk_view->output);
    const int64_t route_count = route_ids == nullptr ? 0 : route_ids->ne[0];
    if (route_ids == nullptr || route_ids->type != GGML_TYPE_I32 ||
        !is_shape(*route_ids, route_count, token_count, 1, 1) || !is_supported_route_count(route_count, expert_count) ||
        route_ids->nb[0] != sizeof(int32_t) || route_ids->nb[1] % sizeof(int32_t) != 0) {
        log_router_reject(status, graph, softmax_node, "top-k route id view shape or stride is incompatible");
        return {};
    }
    const int64_t route_stride = static_cast<int64_t>(route_ids->nb[1] / sizeof(int32_t));
    if (!is_supported_route_stride(route_stride, route_count, expert_count)) {
        log_router_reject(status, graph, softmax_node, "top-k route id stride is outside supported bounds");
        return {};
    }

    const GraphNode * get_rows =
        find_consumer_with_op_and_input(graph, probs_reshape->output, GGML_OP_GET_ROWS, topk_view->output);
    if (get_rows == nullptr || get_rows->inputs.size() != 2) {
        log_router_reject(status, graph, softmax_node, "missing GET_ROWS from reshaped probabilities and top-k ids");
        return {};
    }
    const Value * selected_weights = graph_value(graph, get_rows->output);
    if (selected_weights == nullptr || selected_weights->type != GGML_TYPE_F32 ||
        !is_shape(*selected_weights, 1, route_count, token_count, 1)) {
        log_router_reject(status, graph, softmax_node, "selected route weight shape is incompatible");
        return {};
    }

    const GraphNode * weights_reshape = find_consumer_with_op(graph, get_rows->output, GGML_OP_RESHAPE);
    const Value *     weights_flat = weights_reshape == nullptr ? nullptr : graph_value(graph, weights_reshape->output);
    if (weights_flat == nullptr || weights_flat->type != GGML_TYPE_F32 ||
        !is_shape(*weights_flat, route_count, token_count, 1, 1)) {
        log_router_reject(status, graph, softmax_node, "flattened route weight shape is incompatible");
        return {};
    }

    const GraphNode * sum_rows = find_consumer_with_op(graph, weights_reshape->output, GGML_OP_SUM_ROWS);
    const Value *     sum      = sum_rows == nullptr ? nullptr : graph_value(graph, sum_rows->output);
    if (sum == nullptr || sum->type != GGML_TYPE_F32 || !is_shape(*sum, 1, token_count, 1, 1)) {
        log_router_reject(status, graph, softmax_node, "missing SUM_ROWS over selected route weights");
        return {};
    }

    const GraphNode * clamp       = find_consumer_with_op(graph, sum_rows->output, GGML_OP_CLAMP);
    const Value *     clamped_sum = clamp == nullptr ? nullptr : graph_value(graph, clamp->output);
    if (clamped_sum == nullptr || clamped_sum->type != GGML_TYPE_F32 || !is_shape(*clamped_sum, 1, token_count, 1, 1) ||
        !is_topk_normalization_clamp(*clamp)) {
        log_router_reject(status, graph, softmax_node, "missing supported CLAMP on selected route weight sum");
        return {};
    }

    const GraphNode * div = find_consumer_with_op_and_input(graph, weights_reshape->output, GGML_OP_DIV, clamp->output);
    const Value *     normalized = div == nullptr ? nullptr : graph_value(graph, div->output);
    if (normalized == nullptr || normalized->type != GGML_TYPE_F32 ||
        !is_shape(*normalized, route_count, token_count, 1, 1)) {
        log_router_reject(status, graph, softmax_node, "missing DIV normalization for selected route weights");
        return {};
    }

    const GraphNode * output_reshape = find_consumer_with_op(graph, div->output, GGML_OP_RESHAPE);
    const Value *     route_weights  = output_reshape == nullptr ? nullptr : graph_value(graph, output_reshape->output);
    if (route_weights == nullptr || route_weights->type != GGML_TYPE_F32 ||
        !is_shape(*route_weights, 1, route_count, token_count, 1) || !route_weights->contiguous) {
        log_router_reject(status, graph, softmax_node, "route weight output shape is incompatible");
        return {};
    }

    match.logits        = logits;
    match.route_ids     = route_ids;
    match.route_weights = route_weights;
    match.token_count   = token_count;
    match.expert_count  = expert_count;
    match.route_count   = route_count;
    match.route_stride  = route_stride;
    return match;
}

static bool append_moe_router_top8_coverage(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * softmax       = context.root_node;
    const GraphNode * probs_reshape = find_consumer_with_op(context.graph, softmax->output, GGML_OP_RESHAPE);
    const GraphNode * argsort       = find_consumer_with_op(context.graph, softmax->output, GGML_OP_ARGSORT);
    const GraphNode * topk_view =
        argsort == nullptr ? nullptr : find_consumer_with_op(context.graph, argsort->output, GGML_OP_VIEW);
    const GraphNode * get_rows =
        probs_reshape == nullptr || topk_view == nullptr ?
            nullptr :
            find_consumer_with_op_and_input(context.graph, probs_reshape->output, GGML_OP_GET_ROWS, topk_view->output);
    const GraphNode * weights_reshape =
        get_rows == nullptr ? nullptr : find_consumer_with_op(context.graph, get_rows->output, GGML_OP_RESHAPE);
    const GraphNode * sum_rows = weights_reshape == nullptr ?
                                     nullptr :
                                     find_consumer_with_op(context.graph, weights_reshape->output, GGML_OP_SUM_ROWS);
    const GraphNode * clamp =
        sum_rows == nullptr ? nullptr : find_consumer_with_op(context.graph, sum_rows->output, GGML_OP_CLAMP);
    const GraphNode * div =
        weights_reshape == nullptr || clamp == nullptr ?
            nullptr :
            find_consumer_with_op_and_input(context.graph, weights_reshape->output, GGML_OP_DIV, clamp->output);
    const GraphNode * output_reshape =
        div == nullptr ? nullptr : find_consumer_with_op(context.graph, div->output, GGML_OP_RESHAPE);

    return append_covered_node(context, softmax, match) && append_covered_node(context, probs_reshape, match) &&
           append_covered_node(context, argsort, match) && append_covered_node(context, topk_view, match) &&
           append_covered_node(context, get_rows, match) && append_covered_node(context, weights_reshape, match) &&
           append_covered_node(context, sum_rows, match) && append_covered_node(context, clamp, match) &&
           append_covered_node(context, div, match) && append_covered_node(context, output_reshape, match);
}

static bool append_moe_router_top8_coverage_from_softmax(const DispatchMatchContext & context,
                                                         const GraphNode *            softmax,
                                                         DispatchMatch &              match) {
    if (softmax == nullptr) {
        return false;
    }
    DispatchMatchContext softmax_context = context;
    softmax_context.root_node            = softmax;
    if (!context.graph.index().node_index(softmax, softmax_context.root_index)) {
        return false;
    }
    return append_moe_router_top8_coverage(softmax_context, match);
}

static void add_routed_gate_up_compile_parameters(Dispatch & dispatch, const RouterTop8Match & router_match) {
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.expert_count",
                                               to_config_value(router_match.expert_count));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.routed_gate_up.route_count",
                                               to_config_value(router_match.route_count));
}

static void add_moe_routing_compile_parameters(Dispatch & dispatch, const RouterTop8Match & router_match) {
    dispatch.kernel.compile_parameters.emplace("ggml.workload.token_capacity",
                                               to_config_value(router_match.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.moe_routing.route_count",
                                               to_config_value(router_match.route_count));
    dispatch.kernel.compile_parameters.emplace("ggml.moe_routing.expert_count",
                                               to_config_value(router_match.expert_count));
    dispatch.kernel.compile_parameters.emplace("ggml.moe_routing.descriptor_expert_mask", "127");
    dispatch.kernel.compile_parameters.emplace("ggml.moe_routing.descriptor_partition_shift", "7");
    dispatch.kernel.compile_parameters.emplace("ggml.moe_routing.descriptor_row_count_shift", "13");
    dispatch.kernel.compile_parameters.emplace("ggml.moe_routing.partition_workgroup_size", "128");
}

struct RouterProjectionTop8Match {
    const GraphNode * projection = nullptr;
    const GraphNode * softmax    = nullptr;
    const Value *     input      = nullptr;
    const Value *     weight     = nullptr;
    const Value *     logits     = nullptr;
    RouterTop8Match   top8;

    bool matched() const {
        return projection != nullptr && softmax != nullptr && input != nullptr && weight != nullptr &&
               logits != nullptr && top8.matched();
    }
};

struct RouterProjectionMatch {
    const Value * input       = nullptr;
    const Value * weight      = nullptr;
    const Value * output      = nullptr;
    int64_t       token_count = 0;

    bool matched() const { return input != nullptr && weight != nullptr && output != nullptr && token_count > 0; }
};

static RouterProjectionTop8Match match_moe_router_projection_top8_decode(const DispatchMatchContext & context,
                                                                         Status *                     status) {
    RouterProjectionTop8Match match;
    const GraphNode *         projection = context.root_node;
    if (projection == nullptr || projection->op != GGML_OP_MUL_MAT || projection->inputs.size() != 2 ||
        !context.graph.has_index()) {
        return match;
    }

    const Value * weight = graph_value(context.graph, projection->inputs[0]);
    const Value * input  = graph_value(context.graph, projection->inputs[1]);
    const Value * logits = graph_value(context.graph, projection->output);
    if (weight == nullptr || input == nullptr || logits == nullptr || weight->type != GGML_TYPE_F32 ||
        input->type != GGML_TYPE_F32 || logits->type != GGML_TYPE_F32 || !weight->contiguous || !input->contiguous ||
        !logits->contiguous || !is_shape(*input, kMoeRouterProfile.hidden_size, 1, 1, 1) ||
        !is_shape(*weight, kMoeRouterProfile.hidden_size, kMoeRouterProfile.expert_count, 1, 1) ||
        !is_shape(*logits, kMoeRouterProfile.expert_count, 1, 1, 1)) {
        return {};
    }

    const GraphNode * softmax = find_single_consumer_with_op(context.graph, projection->output, GGML_OP_SOFT_MAX);
    RouterTop8Match   top8    = match_moe_router_top8(context.graph, softmax, status);
    if (!top8.matched() || top8.token_count != 1) {
        return {};
    }

    match.projection = projection;
    match.softmax    = softmax;
    match.input      = input;
    match.weight     = weight;
    match.logits     = logits;
    match.top8       = top8;
    return match;
}

static RouterProjectionMatch match_moe_router_projection_f32(const DispatchMatchContext & context) {
    RouterProjectionMatch match;
    const GraphNode *     projection = context.root_node;
    if (projection == nullptr || projection->op != GGML_OP_MUL_MAT || projection->inputs.size() != 2) {
        return match;
    }

    const Value * weight = graph_value(context.graph, projection->inputs[0]);
    const Value * input  = graph_value(context.graph, projection->inputs[1]);
    const Value * output = graph_value(context.graph, projection->output);
    if (weight == nullptr || input == nullptr || output == nullptr || !is_2d(*weight) || !is_2d(*input) ||
        !is_2d(*output) || !weight->contiguous || !input->contiguous || !output->contiguous ||
        weight->type != GGML_TYPE_F32 || input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32) {
        return {};
    }

    const int64_t input_size  = weight->ne[0];
    const int64_t output_size = weight->ne[1];
    const int64_t token_count = input->ne[1];
    if (input_size != kMoeRouterProfile.hidden_size || output_size != kMoeRouterProfile.expert_count ||
        input->ne[0] != input_size || output->ne[0] != output_size || output->ne[1] != token_count ||
        !is_llm_supported_query_length(kMoeRouterProfile, token_count)) {
        return {};
    }

    match.input       = input;
    match.weight      = weight;
    match.output      = output;
    match.token_count = token_count;
    return match;
}

static bool has_common_matmul_fusible_unary_consumer(const DispatchMatchContext & context, const Value & output) {
    const std::vector<const GraphNode *> & consumers = context.graph.index().consumers(output.id);
    if (consumers.size() != 1 || consumers.front() == nullptr) {
        return false;
    }

    const GraphNode * unary       = consumers.front();
    size_t            unary_index = 0;
    if (!context.graph.index().node_index(unary, unary_index) || unary_index >= context.covered_nodes.size() ||
        context.covered_nodes[unary_index] || unary->inputs.size() != 1 || unary->inputs[0] != output.id) {
        return false;
    }

    const UnaryParams * params = op_params_as<UnaryParams>(unary->params);
    const Value *       result = graph_value(context.graph, unary->output);
    return params != nullptr && unary_kind_supported(params->op) && result != nullptr &&
           result->type == GGML_TYPE_F32 && result->contiguous && same_shape(output, *result);
}

}  // namespace

static bool match_moe_router_projection_f32_dispatch(const DispatchMatchContext & context,
                                                     DispatchMatch &              dispatch_match) {
    const RouterProjectionMatch match = match_moe_router_projection_f32(context);
    if (!match.matched()) {
        return false;
    }
    if (has_common_matmul_fusible_unary_consumer(context, *match.output)) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kQwenRouterProjectionF32FourRowWave32Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity", to_config_value(match.token_count));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.hidden_size",
                                               to_config_value(kMoeRouterProfile.hidden_size));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.router.expert_count",
                                               to_config_value(kMoeRouterProfile.expert_count));
    dispatch.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_moe_router_projection_top8_fused_decode_dispatch(const DispatchMatchContext & context,
                                                                   DispatchMatch &              dispatch_match) {
    const RouterProjectionTop8Match match = match_moe_router_projection_top8_decode(context, &dispatch_match.status);
    if (!match.matched()) {
        return false;
    }

    const ValueId completion_counter_value = context.next_plan_value;
    Dispatch      dispatch;
    dispatch.kernel = make_kernel_specialization(kQwenRouterProjectionTop8FusedDecodeF32Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.top8.token_count);
    dispatch.kernel.integer_parameters.emplace("route_id_stride", match.top8.route_stride);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.hidden_size",
                                               to_config_value(kMoeRouterProfile.hidden_size));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.router.expert_count",
                                               to_config_value(match.top8.expert_count));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.router.route_count", to_config_value(match.top8.route_count));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity",
                                               to_config_value(match.top8.token_count));

    const size_t route_id_length =
        static_cast<size_t>(match.top8.token_count * match.top8.route_stride) * sizeof(int32_t);
    dispatch.bindings.push_back({ match.input->id, 0, match.input->byte_count });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ match.logits->id, 0, match.logits->byte_count });
    dispatch.bindings.push_back({ completion_counter_value, 0, sizeof(int32_t) });
    dispatch.bindings.push_back({ match.top8.route_ids->id, 0, route_id_length });
    dispatch.bindings.push_back({ match.top8.route_weights->id, 0, match.top8.route_weights->byte_count });

    dispatch_match.completion_counter_requests.push_back({
        completion_counter_value,
        "qwen.router.decode_projection_top8_completion_counter",
        1,
    });
    if (!append_covered_node(context, match.projection, dispatch_match) ||
        !append_moe_router_top8_coverage_from_softmax(context, match.softmax, dispatch_match)) {
        return false;
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_moe_router_top8_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const RouterTop8Match router_match =
        match_moe_router_top8(context.graph, context.root_node, &dispatch_match.status);
    if (!router_match.matched()) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kQwenRouterTop8F32Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", router_match.token_count);
    dispatch.kernel.integer_parameters.emplace("route_id_stride", router_match.route_stride);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.router.expert_count",
                                               to_config_value(router_match.expert_count));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.router.route_count",
                                               to_config_value(router_match.route_count));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity",
                                               to_config_value(router_match.token_count));

    const size_t route_id_length =
        static_cast<size_t>(router_match.token_count * router_match.route_stride) * sizeof(int32_t);
    dispatch.bindings.push_back({ router_match.logits->id, 0, router_match.logits->byte_count });
    dispatch.bindings.push_back({ router_match.route_ids->id, 0, route_id_length });
    dispatch.bindings.push_back({ router_match.route_weights->id, 0, router_match.route_weights->byte_count });

    if (!append_moe_router_top8_coverage(context, dispatch_match)) {
        return false;
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));

    const ValueId expert_table_value(context.next_plan_value.value);
    const ValueId partition_table_value(context.next_plan_value.value + 1);
    const ValueId completion_counter_value(context.next_plan_value.value + 2);
    const size_t  expert_table_bytes = expert_table_size(router_match.token_count, router_match.expert_count);
    const size_t  partition_table_bytes =
        partition_table_size(router_match.token_count, router_match.route_count, router_match.expert_count);
    const bool use_fused_prefill_expert_table_partition = supports_fused_prefill_expert_table_partition(router_match);
    dispatch_match.transients.push_back(
        { expert_table_value, "qwen.router.expert_table", expert_table_bytes, kMoeRouterPlanTransientAlignment });
    dispatch_match.transients.push_back({ partition_table_value, "qwen.router.partition_table", partition_table_bytes,
                                          kMoeRouterPlanTransientAlignment });
    if (use_fused_prefill_expert_table_partition) {
        dispatch_match.completion_counter_requests.push_back({
            completion_counter_value,
            "qwen.router.prefill_expert_table_partition_completion_counter",
            1,
        });
    }
    const CommandPlanResourceMetadata routing_metadata = make_command_plan_resource_metadata(MoeRoutingResourceMetadata{
        router_match.token_count,
        router_match.route_count,
        router_match.route_stride,
        router_match.expert_count,
    });
    Status                            metadata_status;
    if (!dispatch_match.metadata.append_generated_resource(
            {
                router_match.route_ids->id,
                GeneratedResourceRole::MoeExpertTable,
                expert_table_value,
                expert_table_bytes,
                routing_metadata,
            },
            metadata_status) ||
        !dispatch_match.metadata.append_generated_resource(
            {
                router_match.route_ids->id,
                GeneratedResourceRole::MoePartitionTable,
                partition_table_value,
                partition_table_bytes,
                routing_metadata,
            },
            metadata_status) ||
        !dispatch_match.metadata.append_moe_routing_bundle(
            {
                router_match.route_ids->id,
                router_match.route_weights->id,
                expert_table_value,
                partition_table_value,
                expert_table_bytes,
                partition_table_bytes,
                router_match.token_count,
                router_match.route_count,
                router_match.route_stride,
                router_match.expert_count,
            },
            metadata_status)) {
        return false;
    }

    if (use_fused_prefill_expert_table_partition) {
        Dispatch expert_table_partition_dispatch;
        expert_table_partition_dispatch.kernel =
            make_kernel_specialization(kQwenBuildExpertTablePartitionPrefill512Kernel);
        expert_table_partition_dispatch.kernel.integer_parameters.emplace("token_count", router_match.token_count);
        expert_table_partition_dispatch.kernel.integer_parameters.emplace("route_count", router_match.route_count);
        expert_table_partition_dispatch.kernel.integer_parameters.emplace("route_stride", router_match.route_stride);
        expert_table_partition_dispatch.kernel.integer_parameters.emplace("expert_count", router_match.expert_count);
        expert_table_partition_dispatch.bindings.push_back({ router_match.route_ids->id, 0, route_id_length });
        expert_table_partition_dispatch.bindings.push_back({ expert_table_value, 0, expert_table_bytes });
        expert_table_partition_dispatch.bindings.push_back({ partition_table_value, 0, partition_table_bytes });
        expert_table_partition_dispatch.bindings.push_back({ completion_counter_value, 0, sizeof(int32_t) });
        dispatch_match.dispatches.push_back(std::move(expert_table_partition_dispatch));
    } else {
        Dispatch expert_table_dispatch;
        expert_table_dispatch.kernel = make_kernel_specialization(kMoeBuildExpertTableKernel);
        expert_table_dispatch.kernel.integer_parameters.emplace("token_count", router_match.token_count);
        expert_table_dispatch.kernel.integer_parameters.emplace("route_count", router_match.route_count);
        expert_table_dispatch.kernel.integer_parameters.emplace("route_stride", router_match.route_stride);
        expert_table_dispatch.kernel.integer_parameters.emplace("expert_count", router_match.expert_count);
        add_moe_routing_compile_parameters(expert_table_dispatch, router_match);
        expert_table_dispatch.bindings.push_back({ router_match.route_ids->id, 0, route_id_length });
        expert_table_dispatch.bindings.push_back({ expert_table_value, 0, expert_table_bytes });
        dispatch_match.dispatches.push_back(std::move(expert_table_dispatch));

        Dispatch partition_table_dispatch;
        partition_table_dispatch.kernel = make_kernel_specialization(kMoeBuildExpertPartitionTableKernel);
        partition_table_dispatch.kernel.integer_parameters.emplace("token_count", router_match.token_count);
        partition_table_dispatch.kernel.integer_parameters.emplace("route_count", router_match.route_count);
        partition_table_dispatch.kernel.integer_parameters.emplace("expert_count", router_match.expert_count);
        add_moe_routing_compile_parameters(partition_table_dispatch, router_match);
        partition_table_dispatch.bindings.push_back({ expert_table_value, 0, expert_table_bytes });
        partition_table_dispatch.bindings.push_back({ partition_table_value, 0, partition_table_bytes });
        dispatch_match.dispatches.push_back(std::move(partition_table_dispatch));
    }
    return true;
}

void register_moe_router_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "llm.moe_router.projection_top8_fused_decode",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        1200,
        DispatchSource::Llm,
        match_moe_router_projection_top8_fused_decode_dispatch,
    });
    registry.add({
        "llm.moe_router.top8_f32",
        GGML_OP_SOFT_MAX,
        DispatchMatchKind::Fused,
        1000,
        DispatchSource::Llm,
        match_moe_router_top8_dispatch,
    });
    registry.add({
        "llm.moe_router.projection_f32_four_row_wave32",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        90,
        DispatchSource::Llm,
        match_moe_router_projection_f32_dispatch,
    });
}

}  // namespace ggml::hrx
