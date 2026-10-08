#include "dispatch-gather-add.h"

#include "dispatch-binary-common.h"
#include "dispatch-mul-mat-common.h"
#include "ggml.h"
#include "graph/op-params.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kGatherAddF32Kernel = GGML_HRX_KERNEL_REF("hrx", "ggml_gather_add_f32");
static constexpr KernelCatalogRef kGatherAddRmsNormBinaryF32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_gather_add_rmsnorm_binary_f32");

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool same_shape(const Value & lhs, const Value & rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i]) {
            return false;
        }
    }
    return true;
}

static bool is_2d_f32(const Value & value) {
    return value.type == GGML_TYPE_F32 && value.contiguous && value.ne[0] > 0 && value.ne[1] > 0 && value.ne[2] == 1 &&
           value.ne[3] == 1;
}

static bool is_row_id_tensor(const Value & value, int64_t output_token_count) {
    return value.type == GGML_TYPE_I32 && value.contiguous && value.element_count == output_token_count &&
           output_token_count > 0;
}

static bool is_supported_hidden_size(int64_t hidden_size) {
    return hidden_size >= 128 && hidden_size <= 32768 && hidden_size % 128 == 0;
}

static bool is_supported_token_count(int64_t token_count) {
    return token_count >= 1 && token_count <= 2048;
}

static bool value_is_available(const Graph & graph, ValueId value, const std::vector<bool> & covered_nodes) {
    const GraphNode * producer = graph.index().producer(value);
    if (producer == nullptr) {
        return true;
    }
    size_t producer_index = 0;
    return graph.index().node_index(producer, producer_index) && producer_index < covered_nodes.size() &&
           covered_nodes[producer_index];
}

struct GatherAddMatch {
    const GraphNode * first_get_rows        = nullptr;
    const GraphNode * second_get_rows       = nullptr;
    const GraphNode * add_node              = nullptr;
    const Value *     first_source          = nullptr;
    const Value *     second_source         = nullptr;
    const Value *     row_ids               = nullptr;
    const Value *     output                = nullptr;
    size_t            first_get_rows_index  = 0;
    size_t            second_get_rows_index = 0;
    size_t            add_node_index        = 0;
    int64_t           source_token_count    = 0;
    int64_t           output_token_count    = 0;
    int64_t           hidden_size           = 0;
    BinaryKind        op                    = BinaryKind::Add;

    bool matched() const {
        return first_get_rows != nullptr && second_get_rows != nullptr && add_node != nullptr &&
               first_source != nullptr && second_source != nullptr && row_ids != nullptr && output != nullptr;
    }
};

struct GatherAddRmsNormBinaryMatch {
    GatherAddMatch   gather_add;
    const GraphNode *rms_node          = nullptr;
    const GraphNode *binary_node       = nullptr;
    const Value *    weight            = nullptr;
    const Value *    normalized_output = nullptr;
    size_t           rms_node_index    = 0;
    size_t           binary_node_index = 0;
    float            epsilon           = 0.0f;

    bool matched() const {
        return gather_add.matched() && rms_node != nullptr && binary_node != nullptr && weight != nullptr &&
               normalized_output != nullptr;
    }
};

static const GraphNode * find_single_binary_consumer(const Graph & graph, const GraphNode & get_rows) {
    const std::vector<const GraphNode *> & consumers = graph.index().consumers(get_rows.output);
    if (consumers.size() != 1) {
        return nullptr;
    }
    const GraphNode * binary = consumers.front();
    return common_match_binary_consumer(binary, get_rows.output, kCommonArithmeticBinaryKinds).matched() ? binary :
                                                                                                           nullptr;
}

static const GraphNode * peer_get_rows_input(const Graph &     graph,
                                             const GraphNode & add,
                                             const GraphNode & root_get_rows) {
    ValueId peer_output;
    if (add.inputs[0] == root_get_rows.output) {
        peer_output = add.inputs[1];
    } else if (add.inputs[1] == root_get_rows.output) {
        peer_output = add.inputs[0];
    } else {
        return nullptr;
    }

    const GraphNode * peer = graph.index().producer(peer_output);
    return peer != nullptr && peer->op == GGML_OP_GET_ROWS && peer->inputs.size() == 2 ? peer : nullptr;
}

static GatherAddMatch match_gather_add_f32(const Graph & graph, const GraphNode * node, size_t node_index) {
    GatherAddMatch match;
    if (node == nullptr || node->op != GGML_OP_GET_ROWS || node->inputs.size() != 2 || !graph.has_index()) {
        return match;
    }

    const GraphNode * add_node = find_single_binary_consumer(graph, *node);
    const GraphNode * peer     = add_node != nullptr ? peer_get_rows_input(graph, *add_node, *node) : nullptr;
    if (add_node == nullptr || peer == nullptr) {
        return {};
    }

    size_t peer_index = 0;
    size_t add_index  = 0;
    if (!graph.index().node_index(peer, peer_index) || !graph.index().node_index(add_node, add_index)) {
        return {};
    }

    const GraphNode * first_get_rows        = add_node->inputs[0] == node->output ? node : peer;
    const GraphNode * second_get_rows       = first_get_rows == node ? peer : node;
    const size_t      first_get_rows_index  = first_get_rows == node ? node_index : peer_index;
    const size_t      second_get_rows_index = first_get_rows == node ? peer_index : node_index;
    const Value *     first_source          = graph_value(graph, first_get_rows->inputs[0]);
    const Value *     first_ids             = graph_value(graph, first_get_rows->inputs[1]);
    const Value *     first_output          = graph_value(graph, first_get_rows->output);
    const Value *     second_source         = graph_value(graph, second_get_rows->inputs[0]);
    const Value *     second_ids            = graph_value(graph, second_get_rows->inputs[1]);
    const Value *     second_output         = graph_value(graph, second_get_rows->output);
    const Value * output        = graph_value(graph, add_node->output);
    if (first_source == nullptr || first_ids == nullptr || first_output == nullptr || second_source == nullptr ||
        second_ids == nullptr || second_output == nullptr || output == nullptr) {
        return {};
    }
    if (first_get_rows->inputs[1] != second_get_rows->inputs[1]) {
        return {};
    }
    if (!is_2d_f32(*first_source) || !is_2d_f32(*second_source) || !is_2d_f32(*first_output) ||
        !is_2d_f32(*second_output) || !is_2d_f32(*output)) {
        return {};
    }
    if (!same_shape(*first_source, *second_source) || !same_shape(*first_output, *second_output) ||
        !same_shape(*first_output, *output)) {
        return {};
    }

    const int64_t hidden_size        = first_source->ne[0];
    const int64_t source_token_count = first_source->ne[1];
    const int64_t output_token_count = first_output->ne[1];
    if (first_output->ne[0] != hidden_size || !is_row_id_tensor(*first_ids, output_token_count) ||
        !is_row_id_tensor(*second_ids, output_token_count)) {
        return {};
    }
    if (!is_supported_hidden_size(hidden_size) || !is_supported_token_count(source_token_count) ||
        !is_supported_token_count(output_token_count)) {
        return {};
    }

    const CommonBinaryMatch binary =
        common_match_binary_consumer(add_node, first_get_rows->output, kCommonArithmeticBinaryKinds);
    if (!binary.matched() || !binary.producer_is_lhs) {
        return {};
    }

    match.first_get_rows        = first_get_rows;
    match.second_get_rows       = second_get_rows;
    match.add_node              = add_node;
    match.first_source          = first_source;
    match.second_source         = second_source;
    match.row_ids               = first_ids;
    match.output                = output;
    match.first_get_rows_index  = first_get_rows_index;
    match.second_get_rows_index = second_get_rows_index;
    match.add_node_index        = add_index;
    match.source_token_count    = source_token_count;
    match.output_token_count    = output_token_count;
    match.hidden_size           = hidden_size;
    match.op                    = binary.kind;
    return match;
}

static bool is_packed_q8_consumer(const Graph & graph, const GraphNode * consumer, const Value & input) {
    if (consumer == nullptr || consumer->op != GGML_OP_MUL_MAT || consumer->inputs.size() != 2 ||
        consumer->inputs[1] != input.id) {
        return false;
    }
    const Value * weight = graph_value(graph, consumer->inputs[0]);
    const Value * output = graph_value(graph, consumer->output);
    if (weight == nullptr || output == nullptr || input.type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
        !input.contiguous || !weight->contiguous || !output->contiguous) {
        return false;
    }
    const bool prefill = (weight->type == GGML_TYPE_Q5_K || weight->type == GGML_TYPE_IQ4_XS) && input.ne[1] >= 256 &&
                         input.ne[1] <= 2048 && input.ne[1] % 256 == 0;
    const bool decode = input.ne[1] >= 1 && input.ne[1] <= 5 && weight->alias_source.value < 0 &&
                        (weight->type == GGML_TYPE_Q4_K ||
                         (weight->type == GGML_TYPE_Q6_K && !graph.index().consumers(output->id).empty()));
    return (prefill || decode) && input.ne[0] >= 256 && input.ne[0] <= 32768 && input.ne[0] % 256 == 0 &&
           input.ne[2] == 1 && input.ne[3] == 1 && weight->ne[0] == input.ne[0] && weight->ne[1] >= 64 &&
           weight->ne[1] <= 262144 && weight->ne[1] % 64 == 0 && weight->ne[2] == 1 && weight->ne[3] == 1 &&
           output->ne[0] == weight->ne[1] && output->ne[1] == input.ne[1] && output->ne[2] == 1 && output->ne[3] == 1;
}

static bool has_packed_q8_consumer(const Graph & graph, const Value & value) {
    for (const GraphNode * consumer : graph.index().consumers(value.id)) {
        if (is_packed_q8_consumer(graph, consumer, value)) {
            return true;
        }
    }
    return false;
}

template <size_t N> static bool pairwise_distinct_storage_roots(const std::array<const Value *, N> & values) {
    for (size_t lhs = 0; lhs < values.size(); ++lhs) {
        for (size_t rhs = lhs + 1; rhs < values.size(); ++rhs) {
            if (values[lhs]->storage_root == values[rhs]->storage_root) {
                return false;
            }
        }
    }
    return true;
}

static GatherAddRmsNormBinaryMatch match_gather_add_rmsnorm_binary_f32(const Graph &     graph,
                                                                        const GraphNode * node,
                                                                        size_t            node_index) {
    GatherAddRmsNormBinaryMatch match;
    match.gather_add = match_gather_add_f32(graph, node, node_index);
    if (!match.gather_add.matched()) {
        return {};
    }

    const GraphNode * rms_node = nullptr;
    for (const GraphNode * consumer : graph.index().consumers(match.gather_add.output->id)) {
        if (consumer != nullptr && consumer->op == GGML_OP_RMS_NORM && consumer->inputs.size() == 1) {
            if (rms_node != nullptr) {
                return {};
            }
            rms_node = consumer;
        }
    }
    size_t rms_node_index = 0;
    if (rms_node == nullptr || !graph.index().node_index(rms_node, rms_node_index)) {
        return {};
    }
    const RmsNormParams * rms_params = op_params_as<RmsNormParams>(rms_node->params);
    const Value *         rms_output = graph_value(graph, rms_node->output);
    if (rms_params == nullptr || !std::isfinite(rms_params->eps) || rms_params->eps <= 0.0f || rms_output == nullptr ||
        rms_output->type != GGML_TYPE_F32 || !rms_output->contiguous ||
        !same_shape(*match.gather_add.output, *rms_output)) {
        return {};
    }
    const std::vector<const GraphNode *> & rms_consumers = graph.index().consumers(rms_output->id);
    if (rms_consumers.size() != 1 || rms_consumers.front() == nullptr || rms_consumers.front()->op != GGML_OP_MUL ||
        rms_consumers.front()->inputs.size() != 2) {
        return {};
    }
    const GraphNode * binary_node = rms_consumers.front();
    size_t            binary_node_index = 0;
    if (!graph.index().node_index(binary_node, binary_node_index)) {
        return {};
    }
    const ValueId weight_id = binary_node->inputs[0] == rms_output->id ? binary_node->inputs[1] :
                                      binary_node->inputs[1] == rms_output->id ? binary_node->inputs[0] :
                                                                                 ValueId{};
    const Value * weight            = graph_value(graph, weight_id);
    const Value * normalized_output = graph_value(graph, binary_node->output);
    if (weight == nullptr || normalized_output == nullptr || weight->type != GGML_TYPE_F32 ||
        normalized_output->type != GGML_TYPE_F32 || !weight->contiguous || !normalized_output->contiguous ||
        weight->ne[0] != match.gather_add.hidden_size || weight->ne[1] != 1 || weight->ne[2] != 1 ||
        weight->ne[3] != 1 || !same_shape(*match.gather_add.output, *normalized_output) ||
        has_packed_q8_consumer(graph, *normalized_output) ||
        common_has_symmetric_i4_lowrow_consumer(graph, *normalized_output) ||
        !pairwise_distinct_storage_roots(std::array<const Value *, 6>{
            match.gather_add.first_source, match.gather_add.second_source, match.gather_add.row_ids,
            match.gather_add.output, weight, normalized_output })) {
        return {};
    }

    match.rms_node          = rms_node;
    match.binary_node       = binary_node;
    match.weight            = weight;
    match.normalized_output = normalized_output;
    match.rms_node_index    = rms_node_index;
    match.binary_node_index = binary_node_index;
    match.epsilon           = rms_params->eps;
    return match;
}

static std::string to_config_value(float value) {
    std::ostringstream out;
    out.precision(9);
    out << value;
    return out.str();
}

}  // namespace

static bool match_gather_add_rmsnorm_binary_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GatherAddRmsNormBinaryMatch fused =
        match_gather_add_rmsnorm_binary_f32(context.graph, context.root_node, context.root_index);
    if (!fused.matched() || fused.gather_add.first_get_rows_index >= context.covered_nodes.size() ||
        fused.gather_add.second_get_rows_index >= context.covered_nodes.size() ||
        fused.gather_add.add_node_index >= context.covered_nodes.size() ||
        fused.rms_node_index >= context.covered_nodes.size() ||
        fused.binary_node_index >= context.covered_nodes.size() ||
        context.covered_nodes[fused.gather_add.first_get_rows_index] ||
        context.covered_nodes[fused.gather_add.second_get_rows_index] ||
        context.covered_nodes[fused.gather_add.add_node_index] || context.covered_nodes[fused.rms_node_index] ||
        context.covered_nodes[fused.binary_node_index]) {
        return false;
    }
    if (!value_is_available(context.graph, fused.gather_add.first_source->id, context.covered_nodes) ||
        !value_is_available(context.graph, fused.gather_add.second_source->id, context.covered_nodes) ||
        !value_is_available(context.graph, fused.gather_add.row_ids->id, context.covered_nodes) ||
        !value_is_available(context.graph, fused.weight->id, context.covered_nodes)) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kGatherAddRmsNormBinaryF32Kernel);
    dispatch.kernel.integer_parameters.emplace("source_token_count", fused.gather_add.source_token_count);
    dispatch.kernel.integer_parameters.emplace("output_token_count", fused.gather_add.output_token_count);
    dispatch.kernel.compile_parameters.emplace("ggml.gather_add_rmsnorm_binary_f32.hidden_size",
                                               std::to_string(fused.gather_add.hidden_size));
    dispatch.kernel.compile_parameters.emplace("ggml.gather_add_rmsnorm_binary_f32.rms_epsilon",
                                               to_config_value(fused.epsilon));
    common_set_binary_compile_parameters(dispatch.kernel, "ggml.gather_add_rmsnorm_binary_f32.gather_op", nullptr,
                                         fused.gather_add.op, true);
    dispatch.bindings.push_back({ fused.gather_add.first_source->id, 0, fused.gather_add.first_source->byte_count });
    dispatch.bindings.push_back({ fused.gather_add.second_source->id, 0, fused.gather_add.second_source->byte_count });
    dispatch.bindings.push_back({ fused.gather_add.row_ids->id, 0, fused.gather_add.row_ids->byte_count });
    dispatch.bindings.push_back({ fused.gather_add.output->id, 0, fused.gather_add.output->byte_count });
    dispatch.bindings.push_back({ fused.weight->id, 0, fused.weight->byte_count });
    dispatch.bindings.push_back({ fused.normalized_output->id, 0, fused.normalized_output->byte_count });

    match.covered_nodes.push_back(fused.gather_add.first_get_rows_index);
    match.covered_nodes.push_back(fused.gather_add.second_get_rows_index);
    match.covered_nodes.push_back(fused.gather_add.add_node_index);
    match.covered_nodes.push_back(fused.rms_node_index);
    match.covered_nodes.push_back(fused.binary_node_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_gather_add_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GatherAddMatch gather_add = match_gather_add_f32(context.graph, context.root_node, context.root_index);
    if (!gather_add.matched() || gather_add.first_get_rows_index >= context.covered_nodes.size() ||
        gather_add.second_get_rows_index >= context.covered_nodes.size() ||
        gather_add.add_node_index >= context.covered_nodes.size() ||
        context.covered_nodes[gather_add.first_get_rows_index] ||
        context.covered_nodes[gather_add.second_get_rows_index] || context.covered_nodes[gather_add.add_node_index]) {
        return false;
    }
    if (!value_is_available(context.graph, gather_add.first_source->id, context.covered_nodes) ||
        !value_is_available(context.graph, gather_add.second_source->id, context.covered_nodes) ||
        !value_is_available(context.graph, gather_add.row_ids->id, context.covered_nodes)) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kGatherAddF32Kernel);
    dispatch.kernel.integer_parameters.emplace("source_token_count", gather_add.source_token_count);
    dispatch.kernel.integer_parameters.emplace("output_token_count", gather_add.output_token_count);
    dispatch.kernel.integer_parameters.emplace("hidden_size", gather_add.hidden_size);
    common_set_binary_compile_parameters(dispatch.kernel, "ggml.gather_add_f32.binary_op", nullptr, gather_add.op,
                                         true);
    dispatch.bindings.push_back({ gather_add.first_source->id, 0, gather_add.first_source->byte_count });
    dispatch.bindings.push_back({ gather_add.second_source->id, 0, gather_add.second_source->byte_count });
    dispatch.bindings.push_back({ gather_add.row_ids->id, 0, gather_add.row_ids->byte_count });
    dispatch.bindings.push_back({ gather_add.output->id, 0, gather_add.output->byte_count });

    match.covered_nodes.push_back(gather_add.first_get_rows_index);
    match.covered_nodes.push_back(gather_add.second_get_rows_index);
    match.covered_nodes.push_back(gather_add.add_node_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

void register_gather_add_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.gather_add_rmsnorm_binary_f32",
        GGML_OP_GET_ROWS,
        DispatchMatchKind::Fused,
        1100,
        DispatchSource::Common,
        match_gather_add_rmsnorm_binary_f32_dispatch,
    });
    registry.add({
        "common.gather_add_f32",
        GGML_OP_GET_ROWS,
        DispatchMatchKind::Fused,
        1000,
        DispatchSource::Common,
        match_gather_add_f32_dispatch,
    });
}

}  // namespace ggml::hrx
