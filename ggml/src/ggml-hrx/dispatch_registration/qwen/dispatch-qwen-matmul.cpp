#include "dispatch-qwen-matmul.h"

#include "dispatch-llm-shapes.h"
#include "dispatch_registration/common/dispatch-activation-publication.h"
#include "dispatch_registration/common/dispatch-mul-mat-common.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kQwenDenseLinearQ4KQ8NextQ8Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_dense_linear_q4k_q8_1_x4_next_q8");
static constexpr KernelCatalogRef kGgmlLinearQ6KQ8_1X4Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "ggml_linear_q6k_q8_1_x4");

static constexpr int64_t kQwenHiddenSize      = kQwen30BMoeDispatchProfile.hidden_size;
static constexpr int64_t kQwenVocabularyCount = 151936;

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool is_2d(const Value & value) {
    return value.ne[0] > 0 && value.ne[1] > 0 && value.ne[2] == 1 && value.ne[3] == 1;
}

static bool is_supported_dense_input_size(int64_t input_size) {
    return input_size >= 256 && input_size <= 32768 && input_size % 256 == 0;
}

static bool is_supported_dense_output_size(int64_t output_size) {
    return output_size >= 1 && output_size <= 262144;
}

static std::string to_config_value(int64_t value) {
    return std::to_string(value);
}

struct QwenMatmulMatch {
    const Value *    input       = nullptr;
    const Value *    weight      = nullptr;
    const Value *    output      = nullptr;
    KernelCatalogRef kernel      = {};
    ValueId          input_value = {};
    size_t           input_bytes = 0;
    int64_t          input_size  = 0;
    int64_t          output_size = 0;
    int64_t          token_count = 0;

    bool matched() const {
        return input != nullptr && weight != nullptr && output != nullptr && kernel.id != kUncatalogedKernelId;
    }
};

struct QwenAttentionOutputNextQ8Match {
    const Value *                     input               = nullptr;
    const CommandPlanAlternateValue * input_alternate     = nullptr;
    const Value *                     weight              = nullptr;
    const Value *                     projection_output   = nullptr;
    const Value *                     residual_input      = nullptr;
    const Value *                     residual_output     = nullptr;
    const Value *                     norm_weight         = nullptr;
    const Value *                     normalized_output   = nullptr;
    const GraphNode *                 projection_get_rows = nullptr;
    const GraphNode *                 residual_get_rows   = nullptr;
    const GraphNode *                 add_node            = nullptr;
    const GraphNode *                 rms_node            = nullptr;
    const GraphNode *                 mul_node            = nullptr;
    int64_t                           input_size          = 0;
    int64_t                           output_size         = 0;
    int64_t                           token_count         = 0;

    bool matched() const {
        return input != nullptr && input_alternate != nullptr && weight != nullptr && projection_output != nullptr &&
               residual_input != nullptr && residual_output != nullptr && norm_weight != nullptr &&
               normalized_output != nullptr && add_node != nullptr && rms_node != nullptr && mul_node != nullptr;
    }
};

static size_t q8_1_x4_byte_count(int64_t token_count, int64_t input_size) {
    if (token_count <= 0 || input_size <= 0) {
        return 0;
    }
    return static_cast<size_t>(token_count) * ggml_row_size(GGML_TYPE_Q8_1, input_size);
}

static const GraphNode * find_single_consumer_with_op(const Graph & graph, ValueId value, ggml_op op) {
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

static bool value_has_no_uncovered_consumers_except(const DispatchMatchContext & context,
                                                    ValueId                      value,
                                                    const GraphNode *            expected_consumer) {
    for (const GraphNode * consumer : context.graph.index().consumers(value)) {
        if (consumer == expected_consumer) {
            continue;
        }
        size_t consumer_index = 0;
        if (!context.graph.index().node_index(consumer, consumer_index) ||
            consumer_index >= context.covered_nodes.size() || !context.covered_nodes[consumer_index]) {
            return false;
        }
    }
    return true;
}

static bool same_value_layout(const Value & lhs, const Value & rhs) {
    if (lhs.type != rhs.type || lhs.byte_count != rhs.byte_count || lhs.element_count != rhs.element_count ||
        lhs.contiguous != rhs.contiguous) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i] || lhs.nb[i] != rhs.nb[i]) {
            return false;
        }
    }
    return true;
}

static const Value * get_rows_source_with_same_layout(const Graph & graph, const GraphNode * node) {
    if (node == nullptr || node->op != GGML_OP_GET_ROWS || node->inputs.size() != 2) {
        return nullptr;
    }
    const Value * source = graph_value(graph, node->inputs[0]);
    const Value * output = graph_value(graph, node->output);
    if (source == nullptr || output == nullptr || !same_value_layout(*source, *output)) {
        return nullptr;
    }
    return source;
}

static QwenMatmulMatch match_qwen_q6k_q8_matmul(const Graph & graph, const GraphNode * node, const CommandPlan & plan) {
    QwenMatmulMatch match;
    if (node == nullptr || node->op != GGML_OP_MUL_MAT || node->inputs.size() != 2) {
        return match;
    }

    const Value * weight = graph_value(graph, node->inputs[0]);
    const Value * input  = graph_value(graph, node->inputs[1]);
    const Value * output = graph_value(graph, node->output);
    if (weight == nullptr || input == nullptr || output == nullptr) {
        return {};
    }
    if (!is_2d(*weight) || !is_2d(*input) || !is_2d(*output)) {
        return {};
    }
    if (!weight->contiguous || !input->contiguous || !output->contiguous) {
        return {};
    }
    if (weight->type != GGML_TYPE_Q6_K || input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32) {
        return {};
    }

    const int64_t input_size  = weight->ne[0];
    const int64_t output_size = weight->ne[1];
    const int64_t token_count = input->ne[1];
    if (input->ne[0] != input_size || output->ne[0] != output_size || output->ne[1] != token_count) {
        return {};
    }
    if (!is_qwen_decode_query_length(token_count) || input_size != kQwenHiddenSize ||
        output_size != kQwenVocabularyCount || !is_supported_dense_input_size(input_size) ||
        !is_supported_dense_output_size(output_size)) {
        return {};
    }

    const size_t                      q8_byte_count = q8_1_x4_byte_count(token_count, input_size);
    const CommandPlanAlternateValue * alternate =
        find_alternate_value(graph, plan, input->id, GGML_TYPE_Q8_1, q8_byte_count);
    if (alternate == nullptr) {
        return {};
    }

    match.input       = input;
    match.weight      = weight;
    match.output      = output;
    match.input_value = alternate->alternate_value;
    match.input_bytes = alternate->byte_count;
    match.kernel      = kGgmlLinearQ6KQ8_1X4Kernel;
    match.input_size  = input_size;
    match.output_size = output_size;
    match.token_count = token_count;
    return match;
}

static QwenAttentionOutputNextQ8Match match_qwen_attention_output_next_q8(const DispatchMatchContext & context) {
    QwenAttentionOutputNextQ8Match match;
    const Graph &                  graph = context.graph;
    const GraphNode *              node  = context.root_node;
    if (node == nullptr || node->op != GGML_OP_MUL_MAT || node->inputs.size() != 2 || !graph.has_index()) {
        return match;
    }

    const Value * weight            = graph_value(graph, node->inputs[0]);
    const Value * input             = graph_value(graph, node->inputs[1]);
    const Value * projection_output = graph_value(graph, node->output);
    if (weight == nullptr || input == nullptr || projection_output == nullptr || !is_2d(*weight) || !is_2d(*input) ||
        !is_2d(*projection_output)) {
        return {};
    }
    if (weight->type != GGML_TYPE_Q4_K || input->type != GGML_TYPE_F32 || projection_output->type != GGML_TYPE_F32 ||
        !weight->contiguous || !input->contiguous || !projection_output->contiguous) {
        return {};
    }

    const int64_t input_size  = weight->ne[0];
    const int64_t output_size = weight->ne[1];
    const int64_t token_count = input->ne[1];
    if (token_count != 1 || input->ne[0] != input_size || projection_output->ne[0] != output_size ||
        projection_output->ne[1] != token_count || input_size != 4096 || output_size != kQwenHiddenSize) {
        return {};
    }

    const CommandPlanAlternateValue * input_alternate = find_alternate_value(
        graph, context.plan, input->id, GGML_TYPE_Q8_1, q8_1_x4_byte_count(token_count, input_size));
    if (input_alternate == nullptr) {
        return {};
    }

    const Value *     selected_projection = projection_output;
    const GraphNode * projection_get_rows = nullptr;
    const GraphNode * add_node            = find_single_consumer_with_op(graph, projection_output->id, GGML_OP_ADD);
    if (add_node == nullptr) {
        projection_get_rows = find_single_consumer_with_op(graph, projection_output->id, GGML_OP_GET_ROWS);
        if (get_rows_source_with_same_layout(graph, projection_get_rows) != projection_output) {
            return {};
        }
        selected_projection = graph_value(graph, projection_get_rows->output);
        add_node            = find_single_consumer_with_op(graph, projection_get_rows->output, GGML_OP_ADD);
    }
    if (add_node == nullptr || add_node->inputs.size() != 2) {
        return {};
    }
    const Value *     residual_input    = nullptr;
    const Value *     selected_residual = nullptr;
    const GraphNode * residual_get_rows = nullptr;
    const GraphNode * residual_consumer = add_node;
    if (add_node->inputs[0] == selected_projection->id) {
        selected_residual = graph_value(graph, add_node->inputs[1]);
    } else if (add_node->inputs[1] == selected_projection->id) {
        selected_residual = graph_value(graph, add_node->inputs[0]);
    }
    if (selected_residual == nullptr) {
        return {};
    }
    const GraphNode * selected_residual_producer = graph.index().producer(selected_residual->id);
    residual_input                               = get_rows_source_with_same_layout(graph, selected_residual_producer);
    if (residual_input != nullptr) {
        residual_get_rows = selected_residual_producer;
        residual_consumer = residual_get_rows;
    } else {
        residual_input = selected_residual;
    }
    const Value * residual_output = graph_value(graph, add_node->output);
    if (residual_input == nullptr || residual_output == nullptr || residual_input->type != GGML_TYPE_F32 ||
        residual_output->type != GGML_TYPE_F32 || !same_value_layout(*selected_projection, *selected_residual) ||
        !same_value_layout(*selected_projection, *residual_input) ||
        !same_value_layout(*selected_projection, *residual_output) ||
        !value_has_no_uncovered_consumers_except(context, residual_input->id, residual_consumer)) {
        return {};
    }

    const GraphNode * rms_node = find_single_consumer_with_op(graph, residual_output->id, GGML_OP_RMS_NORM);
    if (rms_node == nullptr || rms_node->inputs.size() != 1) {
        return {};
    }
    const Value *     rms_output = graph_value(graph, rms_node->output);
    const GraphNode * mul_node   = find_single_consumer_with_op(graph, rms_node->output, GGML_OP_MUL);
    if (rms_output == nullptr || mul_node == nullptr || mul_node->inputs.size() != 2) {
        return {};
    }

    const Value * norm_weight = nullptr;
    if (mul_node->inputs[0] == rms_node->output) {
        norm_weight = graph_value(graph, mul_node->inputs[1]);
    } else if (mul_node->inputs[1] == rms_node->output) {
        norm_weight = graph_value(graph, mul_node->inputs[0]);
    }
    const Value * normalized_output = graph_value(graph, mul_node->output);
    if (norm_weight == nullptr || normalized_output == nullptr || norm_weight->type != GGML_TYPE_F32 ||
        normalized_output->type != GGML_TYPE_F32 || !norm_weight->contiguous || !normalized_output->contiguous ||
        norm_weight->ne[0] != output_size || normalized_output->ne[0] != output_size ||
        normalized_output->ne[1] != token_count) {
        return {};
    }

    match.input               = input;
    match.input_alternate     = input_alternate;
    match.weight              = weight;
    match.projection_output   = projection_output;
    match.residual_input      = residual_input;
    match.residual_output     = residual_output;
    match.norm_weight         = norm_weight;
    match.normalized_output   = normalized_output;
    match.projection_get_rows = projection_get_rows;
    match.residual_get_rows   = residual_get_rows;
    match.add_node            = add_node;
    match.rms_node            = rms_node;
    match.mul_node            = mul_node;
    match.input_size          = input_size;
    match.output_size         = output_size;
    match.token_count         = token_count;
    return match;
}

}  // namespace

bool qwen_accepts_q6k_q8_input(const DispatchMatchContext & context,
                               const GraphNode &            consumer,
                               const Value &                input) {
    if (consumer.op != GGML_OP_MUL_MAT || consumer.inputs.size() != 2 || consumer.inputs[1] != input.id) {
        return false;
    }
    const Value * weight = graph_value(context.graph, consumer.inputs[0]);
    const Value * output = graph_value(context.graph, consumer.output);
    return weight != nullptr && output != nullptr && weight->type == GGML_TYPE_Q6_K &&
           input.type == GGML_TYPE_F32 && output->type == GGML_TYPE_F32 && weight->contiguous && input.contiguous &&
           output->contiguous && input.ne[0] == kQwenHiddenSize && input.ne[1] == 1 && input.ne[2] == 1 &&
           input.ne[3] == 1 && weight->ne[0] == kQwenHiddenSize && weight->ne[1] == kQwenVocabularyCount &&
           weight->ne[2] == 1 && weight->ne[3] == 1 && output->ne[0] == kQwenVocabularyCount && output->ne[1] == 1 &&
           output->ne[2] == 1 && output->ne[3] == 1;
}

static void build_qwen_matmul_dispatch(const QwenMatmulMatch & match,
                                       DispatchMatch &         dispatch_match,
                                       size_t                  root_index) {
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(match.kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.integer_parameters.emplace("input_size", match.input_size);
    dispatch.kernel.integer_parameters.emplace("output_size", match.output_size);
    dispatch.kernel.compile_parameters.emplace("ggml.linear_q6k_q8_1_x4.token_capacity",
                                               to_config_value(match.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.linear_q6k_q8_1_x4.output_capacity",
                                               to_config_value(match.output_size));
    dispatch.bindings.push_back({ match.input_value, 0, match.input_bytes });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(root_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
}

static bool match_qwen_q6k_q8_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const QwenMatmulMatch match = match_qwen_q6k_q8_matmul(context.graph, context.root_node, context.plan);
    if (!match.matched()) {
        return false;
    }
    build_qwen_matmul_dispatch(match, dispatch_match, context.root_index);
    return true;
}

static bool match_qwen_attention_output_next_q8_dispatch(const DispatchMatchContext & context,
                                                         DispatchMatch &              dispatch_match) {
    const QwenAttentionOutputNextQ8Match match = match_qwen_attention_output_next_q8(context);
    if (!match.matched()) {
        return false;
    }

    const CommonActivationPublicationCapabilities capabilities = {
        DispatchActivationInputQ8_1X4,
        DispatchActivationInputQ8_1X4,
    };
    const CommonActivationPublicationPlan publication_plan =
        common_select_activation_publication_plan(context, *match.normalized_output, capabilities);
    const CommonActivationPublicationDemand q8_demand =
        publication_plan.matched() ? publication_plan.preferred() : CommonActivationPublicationDemand{};
    if (!q8_demand.matched()) {
        return false;
    }

    const ValueId completion_counter(context.next_plan_value.value);
    dispatch_match.completion_counter_requests.push_back({
        completion_counter,
        "qwen.decode.attention_output.completion_counter",
        1,
    });
    CommonActivationPublication q8_publication;
    constexpr const char * q8_publication_name = "qwen.decode.attention_output.next_q8_output";
    if (!common_reserve_activation_publication(context, dispatch_match, *match.normalized_output, q8_demand,
                                               q8_publication_name, q8_publication) ||
        !common_append_activation_publication(dispatch_match, q8_publication, q8_publication_name)) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kQwenDenseLinearQ4KQ8NextQ8Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.input_size",
                                               to_config_value(match.input_size));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.output_size",
                                               to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.dense_quantized.output_accumulation", "1");
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.hidden_size", to_config_value(match.output_size));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.model.rms_epsilon", "0.000001");
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity", to_config_value(match.token_count));
    dispatch.bindings.push_back({ match.input_alternate->alternate_value, 0, match.input_alternate->byte_count });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ match.residual_output->id, 0, match.residual_output->byte_count });
    dispatch.bindings.push_back({ match.norm_weight->id, 0, match.norm_weight->byte_count });
    dispatch.bindings.push_back({ match.normalized_output->id, 0, match.normalized_output->byte_count });
    dispatch.bindings.push_back({ completion_counter, 0, sizeof(int32_t) });
    dispatch.bindings.push_back(q8_publication.binding());

    dispatch_match.value_aliases.push_back({ match.residual_input->id, match.residual_output->id });
    if (!append_covered_node_index_once(context.graph, context.covered_nodes, context.root_node,
                                        dispatch_match.covered_nodes) ||
        (match.projection_get_rows != nullptr &&
         !append_covered_node_index_once(context.graph, context.covered_nodes, match.projection_get_rows,
                                         dispatch_match.covered_nodes)) ||
        (match.residual_get_rows != nullptr &&
         !append_covered_node_index_once(context.graph, context.covered_nodes, match.residual_get_rows,
                                         dispatch_match.covered_nodes)) ||
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.add_node,
                                        dispatch_match.covered_nodes) ||
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.rms_node,
                                        dispatch_match.covered_nodes) ||
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.mul_node,
                                        dispatch_match.covered_nodes)) {
        return false;
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

void register_qwen_matmul_dispatches(DispatchRegistryBuilder & registry) {
    registry.add_activation_consumer({
        "qwen.matmul.input.q8_1_x4.q6k_vocab",
        GGML_OP_MUL_MAT,
        DispatchActivationInputQ8_1X4,
        DispatchActivationConsumerUse::BandwidthLimited,
        DispatchActivationProducerRequirementNone,
        DispatchSource::Qwen,
        qwen_accepts_q6k_q8_input,
    });
    registry.add({
        "qwen.matmul.attention_output_q4k_q8_1_x4_next_q8",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        300,
        DispatchSource::Qwen,
        match_qwen_attention_output_next_q8_dispatch,
    });
    registry.add({
        "qwen.matmul.q6k_q8_1_x4",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        305,
        DispatchSource::Qwen,
        match_qwen_q6k_q8_dispatch,
    });
}

}  // namespace ggml::hrx
