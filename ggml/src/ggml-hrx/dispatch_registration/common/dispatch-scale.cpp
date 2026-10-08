#include "dispatch-scale.h"

#include "dispatch-binary-common.h"
#include "dispatch-layout-utils.h"
#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kScaleF32Kernel    = GGML_HRX_KERNEL_REF("loom_libs", "ggml_scale_f32");
static constexpr KernelCatalogRef kScaleAddF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_scale_add_f32");

static bool same_shape(const Value & lhs, const Value & rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i]) {
            return false;
        }
    }
    return true;
}

static bool positive_shape(const Value & value) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (value.ne[i] <= 0) {
            return false;
        }
    }
    return value.element_count > 0;
}

static bool packed_f32_layout(const Value & value) {
    size_t expected_stride = sizeof(float);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (value.nb[i] != expected_stride) {
            return false;
        }
        expected_stride *= static_cast<size_t>(value.ne[i]);
    }
    return true;
}

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool distinct_storage(const Value & input, const Value & output) {
    return input.storage != output.storage;
}

static bool storage_ranges_disjoint(const Value & lhs,
                                    size_t        lhs_byte_count,
                                    const Value & rhs,
                                    size_t        rhs_byte_count) {
    if (lhs.storage != rhs.storage) {
        return true;
    }
    if (lhs.storage_offset > std::numeric_limits<size_t>::max() - lhs_byte_count ||
        rhs.storage_offset > std::numeric_limits<size_t>::max() - rhs_byte_count) {
        return false;
    }
    return lhs.storage_offset + lhs_byte_count <= rhs.storage_offset ||
           rhs.storage_offset + rhs_byte_count <= lhs.storage_offset;
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

static bool value_is_available_or_layout_alias(const Graph &             graph,
                                               ValueId                   value,
                                               const std::vector<bool> & covered_nodes) {
    if (value_is_available(graph, value, covered_nodes)) {
        return true;
    }
    const GraphNode * producer = graph.index().producer(value);
    return producer != nullptr && is_layout_alias_node(graph, *producer) && producer->inputs.size() == 1 &&
           value_is_available(graph, producer->inputs[0], covered_nodes);
}

static bool supported_source_layout(const Graph & graph, const Value & value) {
    if (value.alias_source.value < 0) {
        return true;
    }
    if (value.storage_offset != 0) {
        return false;
    }
    const GraphNode * producer = graph.index().producer(value.id);
    return producer != nullptr && (producer->op == GGML_OP_RESHAPE || producer->op == GGML_OP_CLAMP);
}

static std::string format_float_config(float value) {
    std::ostringstream out;
    out.precision(9);
    out << value;
    return out.str();
}

static void add_strided_source_config(Dispatch &    dispatch,
                                      const char *  prefix,
                                      const Value & source,
                                      size_t        source_byte_count) {
    dispatch.kernel.compile_parameters.emplace(std::string(prefix) + "stride1",
                                               std::to_string(source.nb[1] / sizeof(float)));
    dispatch.kernel.compile_parameters.emplace(std::string(prefix) + "stride2",
                                               std::to_string(source.nb[2] / sizeof(float)));
    dispatch.kernel.compile_parameters.emplace(std::string(prefix) + "stride3",
                                               std::to_string(source.nb[3] / sizeof(float)));
    dispatch.kernel.compile_parameters.emplace(std::string(prefix) + "span",
                                               std::to_string(source_byte_count / sizeof(float)));
}

static bool match_scale_add_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * scale_node = context.root_node;
    if (scale_node == nullptr || scale_node->op != GGML_OP_SCALE || scale_node->inputs.size() != 1 ||
        !context.graph.has_index() || context.root_index >= context.covered_nodes.size() ||
        context.covered_nodes[context.root_index]) {
        return false;
    }

    const ScaleParams * scale_params = op_params_as<ScaleParams>(scale_node->params);
    if (scale_params == nullptr || !context.graph.index().has_single_consumer(scale_node->output)) {
        return false;
    }
    const GraphNode *       binary_node = context.graph.index().consumers(scale_node->output).front();
    const CommonBinaryMatch binary =
        common_match_binary_consumer(binary_node, scale_node->output, kCommonArithmeticBinaryKinds);
    if (!binary.matched()) {
        return false;
    }

    size_t binary_index = 0;
    if (!context.graph.index().node_index(binary_node, binary_index) || binary_index >= context.covered_nodes.size() ||
        context.covered_nodes[binary_index]) {
        return false;
    }

    const Value * input    = graph_value(context.graph, scale_node->inputs[0]);
    const Value * scaled   = graph_value(context.graph, scale_node->output);
    const Value * residual = graph_value(context.graph, binary.operand);
    const Value * output   = graph_value(context.graph, binary_node->output);
    if (input == nullptr || scaled == nullptr || residual == nullptr || output == nullptr ||
        input->type != GGML_TYPE_F32 || scaled->type != GGML_TYPE_F32 || residual->type != GGML_TYPE_F32 ||
        output->type != GGML_TYPE_F32 || !same_shape(*input, *scaled) || !same_shape(*input, *residual) ||
        !same_shape(*input, *output) || !positive_shape(*output) || !output->contiguous ||
        !packed_f32_layout(*output) || scaled->alias_source.value >= 0 || output->alias_source.value >= 0 ||
        static_cast<uint64_t>(output->element_count) > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    size_t input_byte_count    = 0;
    size_t residual_byte_count = 0;
    if (!strided_f32_storage_span_bytes(*input, input_byte_count) ||
        !strided_f32_storage_span_bytes(*residual, residual_byte_count) ||
        !storage_ranges_disjoint(*input, input_byte_count, *output, output->byte_count) ||
        !storage_ranges_disjoint(*residual, residual_byte_count, *output, output->byte_count) ||
        !value_is_available_or_layout_alias(context.graph, input->id, context.covered_nodes) ||
        !value_is_available_or_layout_alias(context.graph, residual->id, context.covered_nodes)) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kScaleAddF32Kernel);
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.kernel.compile_parameters.emplace("ggml.scale_add_f32.ne0", std::to_string(output->ne[0]));
    dispatch.kernel.compile_parameters.emplace("ggml.scale_add_f32.ne1", std::to_string(output->ne[1]));
    dispatch.kernel.compile_parameters.emplace("ggml.scale_add_f32.ne2", std::to_string(output->ne[2]));
    dispatch.kernel.compile_parameters.emplace("ggml.scale_add_f32.scale", format_float_config(scale_params->scale));
    dispatch.kernel.compile_parameters.emplace("ggml.scale_add_f32.bias", format_float_config(scale_params->bias));
    common_set_binary_compile_parameters(dispatch.kernel, "ggml.scale_add_f32.binary_op",
                                         "ggml.scale_add_f32.scaled_lhs", binary.kind, binary.producer_is_lhs);
    add_strided_source_config(dispatch, "ggml.scale_add_f32.input_", *input, input_byte_count);
    add_strided_source_config(dispatch, "ggml.scale_add_f32.residual_", *residual, residual_byte_count);
    dispatch.bindings.push_back({ input->storage_root, input->storage_offset, input_byte_count });
    dispatch.bindings.push_back({ residual->storage_root, residual->storage_offset, residual_byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.covered_nodes.push_back(binary_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_scale_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_SCALE || node->inputs.size() != 1) {
        return false;
    }

    const ScaleParams * params = op_params_as<ScaleParams>(node->params);
    if (params == nullptr) {
        return false;
    }

    const Value * output = graph_value(context.graph, node->output);
    const Value * input  = graph_value(context.graph, node->inputs[0]);
    if (output == nullptr || input == nullptr) {
        return false;
    }

    if (output->type != GGML_TYPE_F32 || input->type != GGML_TYPE_F32 || !same_shape(*output, *input) ||
        !positive_shape(*output) || !output->contiguous || !input->contiguous || !packed_f32_layout(*output) ||
        !packed_f32_layout(*input) || output->alias_source.value >= 0 ||
        !supported_source_layout(context.graph, *input) || !distinct_storage(*input, *output) ||
        static_cast<uint64_t>(output->element_count) > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kScaleF32Kernel);
    dispatch.kernel.workload_specialization = WorkloadSpecialization::Dynamic;
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.kernel.compile_parameters.emplace("ggml.scale_f32.scale", format_float_config(params->scale));
    dispatch.kernel.compile_parameters.emplace("ggml.scale_f32.bias", format_float_config(params->bias));
    dispatch.bindings.push_back({ input->id, 0, input->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_scale_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.scale_add_f32",
        GGML_OP_SCALE,
        DispatchMatchKind::Fused,
        1000,
        DispatchSource::Common,
        match_scale_add_f32_dispatch,
    });
    registry.add({
        "common.scale_f32",
        GGML_OP_SCALE,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_scale_f32_dispatch,
    });
}

}  // namespace ggml::hrx
