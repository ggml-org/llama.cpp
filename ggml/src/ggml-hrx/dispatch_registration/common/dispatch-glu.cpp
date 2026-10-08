#include "dispatch-glu.h"

#include "dispatch-layout-utils.h"
#include "dispatch-mul-mat-common.h"
#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kBinaryF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_binary_f32");

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
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

static bool storage_ranges_disjoint(const Value & lhs,
                                    size_t        lhs_offset,
                                    size_t        lhs_byte_count,
                                    const Value & rhs,
                                    size_t        rhs_offset,
                                    size_t        rhs_byte_count) {
    if (lhs.storage != rhs.storage) {
        return true;
    }
    if (lhs.storage_offset > std::numeric_limits<size_t>::max() - lhs_offset ||
        rhs.storage_offset > std::numeric_limits<size_t>::max() - rhs_offset) {
        return false;
    }
    const size_t lhs_start = lhs.storage_offset + lhs_offset;
    const size_t rhs_start = rhs.storage_offset + rhs_offset;
    if (lhs_start > std::numeric_limits<size_t>::max() - lhs_byte_count ||
        rhs_start > std::numeric_limits<size_t>::max() - rhs_byte_count) {
        return false;
    }
    return lhs_start + lhs_byte_count <= rhs_start || rhs_start + rhs_byte_count <= lhs_start;
}

static bool packed_glu_half_span_bytes(const Value & input,
                                       const Value & output,
                                       size_t        half_offset,
                                       size_t &      byte_count) {
    if (input.type != GGML_TYPE_F32 || input.nb[0] != sizeof(float)) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (input.nb[i] % sizeof(float) != 0) {
            return false;
        }
    }

    size_t max_offset = 0;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (output.ne[i] <= 0) {
            return false;
        }
        const size_t extent = static_cast<size_t>(output.ne[i] - 1);
        if (extent != 0 && input.nb[i] > std::numeric_limits<size_t>::max() / extent) {
            return false;
        }
        const size_t dim_offset = extent * input.nb[i];
        if (max_offset > std::numeric_limits<size_t>::max() - dim_offset) {
            return false;
        }
        max_offset += dim_offset;
    }
    if (max_offset > std::numeric_limits<size_t>::max() - sizeof(float)) {
        return false;
    }
    byte_count = max_offset + sizeof(float);
    if (input.storage_offset > std::numeric_limits<size_t>::max() - half_offset ||
        input.storage_offset + half_offset > std::numeric_limits<size_t>::max() - byte_count) {
        return false;
    }
    return input.storage_offset + half_offset + byte_count <= input.storage_byte_count;
}

static void add_packed_glu_parameters(Dispatch &    dispatch,
                                      const Value & input,
                                      const Value & output,
                                      size_t        half_span) {
    dispatch.kernel.integer_parameters.emplace("element_count", output.element_count);
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.ne0", std::to_string(output.ne[0]));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.ne1", std::to_string(output.ne[1]));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.ne2", std::to_string(output.ne[2]));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.src0_stride1",
                                               std::to_string(input.nb[1] / sizeof(float)));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.src0_stride2",
                                               std::to_string(input.nb[2] / sizeof(float)));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.src0_stride3",
                                               std::to_string(input.nb[3] / sizeof(float)));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.src1_stride1",
                                               std::to_string(input.nb[1] / sizeof(float)));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.src1_stride2",
                                               std::to_string(input.nb[2] / sizeof(float)));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.src1_stride3",
                                               std::to_string(input.nb[3] / sizeof(float)));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.src0_span", std::to_string(half_span / sizeof(float)));
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.src1_span", std::to_string(half_span / sizeof(float)));
}

static void bind_packed_glu_source(Dispatch & dispatch, const Value & input, size_t offset, size_t byte_count) {
    dispatch.bindings.push_back({ input.storage_root, input.storage_offset + offset, byte_count });
}

static bool match_packed_glu_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_GLU || node->inputs.size() != 1) {
        return false;
    }

    BinaryKind op;
    if (!common_fused_binary_kind_from_params(node->params, op)) {
        return false;
    }

    const GluParams * params = op_params_as<GluParams>(node->params);
    if (params == nullptr) {
        return false;
    }

    const Value * input  = graph_value(context.graph, node->inputs[0]);
    const Value * output = graph_value(context.graph, node->output);
    if (input == nullptr || output == nullptr || input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
        !positive_shape(*output) || input->ne[0] != 2 * output->ne[0] || output->alias_source.value >= 0 ||
        !output->contiguous || !packed_f32_layout(*output) ||
        static_cast<uint64_t>(output->element_count) > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (input->ne[i] != output->ne[i]) {
            return false;
        }
    }

    const size_t half_offset = static_cast<size_t>(output->ne[0]) * sizeof(float);
    size_t       half_span   = 0;
    size_t       input_span  = 0;
    if (!packed_glu_half_span_bytes(*input, *output, 0, half_span) ||
        !packed_glu_half_span_bytes(*input, *output, half_offset, half_span) ||
        !strided_f32_storage_span_bytes(*input, input_span) ||
        !storage_ranges_disjoint(*input, 0, input_span, *output, 0, output->byte_count)) {
        return false;
    }

    const size_t gate_offset = params->swapped ? half_offset : 0;
    const size_t up_offset   = params->swapped ? 0 : half_offset;

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kBinaryF32Kernel);
    add_packed_glu_parameters(dispatch, *input, *output, half_span);
    dispatch.kernel.compile_parameters.emplace("ggml.binary_f32.op", std::to_string(binary_kind_config_value(op)));
    bind_packed_glu_source(dispatch, *input, gate_offset, half_span);
    bind_packed_glu_source(dispatch, *input, up_offset, half_span);
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_glu_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.packed_glu_f32",
        GGML_OP_GLU,
        DispatchMatchKind::SingleOp,
        10,
        DispatchSource::Common,
        match_packed_glu_f32_dispatch,
    });
}

}  // namespace ggml::hrx
