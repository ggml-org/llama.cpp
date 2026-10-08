#include "dispatch-get-rows.h"

#include "../qwen/dispatch-llm-profiles.h"
#include "dispatch-activation-publication.h"
#include "dispatch-mul-mat-common.h"
#include "dispatch-mul-mat-weight-format.h"
#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kGetRowsF32Kernel     = GGML_HRX_KERNEL_REF("loom_libs", "ggml_get_rows_f32");
static constexpr KernelCatalogRef kGetRowsF32NextKernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_get_rows_f32_next");
static constexpr KernelCatalogRef kGetRowsScaleF32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_get_rows_scale_f32");
static constexpr KernelCatalogRef kGetRowsRmsNormBinaryQ8_1X4F16Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_get_rows_rmsnorm_binary_q8_1_x4_f16");
static constexpr int64_t          kMaximumHiddenElements = int64_t{ 1 } << 30;
static constexpr int64_t          kQ1_0GetRowsFormat    = 10;
static constexpr int64_t          kMaxGetRowsRowCount   = 262208;

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool is_1d_or_2d_column(const Value & value) {
    return value.ne[0] > 0 && value.ne[1] == 1 && value.ne[2] == 1 && value.ne[3] == 1;
}

static bool is_2d(const Value & value) {
    return value.ne[0] > 0 && value.ne[1] > 0 && value.ne[2] == 1 && value.ne[3] == 1;
}

static bool is_supported_hidden_size(ggml_type type, int64_t hidden_size) {
    if (hidden_size < 4 || hidden_size > kMaximumHiddenElements || hidden_size % 4 != 0) {
        return false;
    }

    const int64_t block_size = ggml_blck_size(type);
    return block_size > 0 && hidden_size % block_size == 0;
}

static bool is_supported_token_count(int64_t token_count) {
    return token_count >= 1 && token_count <= 2048;
}

static int64_t token_capacity_class(int64_t token_count) {
    int64_t capacity = 1;
    while (capacity < token_count) {
        capacity *= 2;
    }
    return capacity;
}

static bool is_supported_row_count(int64_t row_count) {
    return row_count >= 1 && row_count <= kMaxGetRowsRowCount;
}

static std::string to_config_value(int64_t value) {
    return std::to_string(value);
}

static std::string to_config_value(float value) {
    std::ostringstream out;
    out.precision(9);
    out << value;
    return out.str();
}

static bool get_rows_format_for_type(ggml_type type, int64_t & format) {
    if (type == GGML_TYPE_Q1_0) {
        format = kQ1_0GetRowsFormat;
        return true;
    }

    CommonMulMatWeightFormat common_format;
    if (!common_mul_mat_format_for_type(type, common_format)) {
        return false;
    }

    format = common_mul_mat_format_config_value(common_format);
    return true;
}

struct GetRowsMatch {
    const Value * ids                 = nullptr;
    const Value * weight              = nullptr;
    const Value * output              = nullptr;
    int64_t       weight_format_value = -1;
    int64_t       token_count         = 0;
    int64_t       row_count           = 0;
    int64_t       hidden_size         = 0;

    bool matched() const {
        return ids != nullptr && weight != nullptr && output != nullptr && weight_format_value >= 0;
    }
};

struct GetRowsRmsNormMatch {
    GetRowsMatch      rows;
    const GraphNode * rms_node          = nullptr;
    const GraphNode * binary_node       = nullptr;
    const Value *     norm_weight       = nullptr;
    const Value *     normalized_output = nullptr;
    size_t            rms_node_index    = 0;
    size_t            binary_node_index = 0;
    float             epsilon           = 0.0f;

    bool matched() const {
        return rows.matched() && rms_node != nullptr && binary_node != nullptr && norm_weight != nullptr &&
               normalized_output != nullptr;
    }
};

struct GetRowsScaleMatch {
    GetRowsMatch      rows;
    const GraphNode * scale_node  = nullptr;
    const Value *     output      = nullptr;
    size_t            scale_index = 0;
    float             scale       = 0.0f;
    float             bias        = 0.0f;

    bool matched() const {
        return rows.matched() && scale_node != nullptr && output != nullptr;
    }
};

static void add_common_compile_parameters(Dispatch & dispatch, const GetRowsMatch & match) {
    dispatch.kernel.compile_parameters.emplace("ggml.get_rows_f32.token_capacity", to_config_value(match.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.get_rows_f32.hidden_capacity", to_config_value(match.hidden_size));
    dispatch.kernel.compile_parameters.emplace("ggml.get_rows_f32.weight_format",
                                               to_config_value(match.weight_format_value));
}

static void add_common_integer_parameters(Dispatch & dispatch, const GetRowsMatch & match) {
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.integer_parameters.emplace("row_count", match.row_count);
    dispatch.kernel.integer_parameters.emplace("hidden_size", match.hidden_size);
}

static void add_primary_bindings(Dispatch & dispatch, const GetRowsMatch & match) {
    dispatch.bindings.push_back({ match.ids->id, 0, match.ids->byte_count });
    dispatch.bindings.push_back({ match.weight->id, 0, match.weight->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
}

static GetRowsMatch match_get_rows_f32(const Graph & graph, const GraphNode * node) {
    GetRowsMatch match;
    if (node == nullptr || node->op != GGML_OP_GET_ROWS || node->inputs.size() != 2) {
        return match;
    }

    const Value *            weight = graph_value(graph, node->inputs[0]);
    const Value *            ids    = graph_value(graph, node->inputs[1]);
    const Value *            output = graph_value(graph, node->output);
    if (weight == nullptr || ids == nullptr || output == nullptr || weight->type == GGML_TYPE_IQ3_S ||
        weight->type == GGML_TYPE_IQ4_NL || ids->type != GGML_TYPE_I32 || output->type != GGML_TYPE_F32 ||
        !weight->contiguous || !ids->contiguous || !output->contiguous || !is_2d(*weight) ||
        !is_1d_or_2d_column(*ids) || !is_2d(*output)) {
        return {};
    }

    int64_t weight_format_value = 0;
    if (!get_rows_format_for_type(weight->type, weight_format_value)) {
        return {};
    }

    const int64_t hidden_size = weight->ne[0];
    const int64_t row_count   = weight->ne[1];
    const int64_t token_count = ids->ne[0];
    if (output->ne[0] != hidden_size || output->ne[1] != token_count ||
        !is_supported_hidden_size(weight->type, hidden_size) || !is_supported_row_count(row_count) ||
        !is_supported_token_count(token_count)) {
        return {};
    }

    match.ids                 = ids;
    match.weight              = weight;
    match.output              = output;
    match.weight_format_value = weight_format_value;
    match.token_count         = token_count;
    match.row_count           = row_count;
    match.hidden_size         = hidden_size;
    return match;
}

static GetRowsScaleMatch match_get_rows_scale_f32(const DispatchMatchContext & context) {
    GetRowsScaleMatch match;
    match.rows = match_get_rows_f32(context.graph, context.root_node);
    if (!match.rows.matched() || !context.graph.has_index()) {
        return {};
    }

    const bool supported_weight = match.rows.weight->type == GGML_TYPE_F32 ||
                                  (match.rows.weight->type == GGML_TYPE_Q5_1 && match.rows.token_count > 1);
    if (!supported_weight) {
        return {};
    }

    const std::vector<const GraphNode *> & consumers = context.graph.index().consumers(match.rows.output->id);
    if (consumers.size() != 1 || consumers.front() == nullptr || consumers.front()->op != GGML_OP_SCALE ||
        consumers.front()->inputs.size() != 1 || consumers.front()->inputs[0] != match.rows.output->id) {
        return {};
    }

    const GraphNode * scale_node = consumers.front();
    const ScaleParams * params   = op_params_as<ScaleParams>(scale_node->params);
    const Value * output         = graph_value(context.graph, scale_node->output);
    if (params == nullptr || !std::isfinite(params->scale) || !std::isfinite(params->bias) || output == nullptr ||
        output->type != GGML_TYPE_F32 || !output->contiguous || output->alias_source.value >= 0 ||
        output->ne != match.rows.output->ne || output->byte_count != match.rows.output->byte_count) {
        return {};
    }

    const std::array<const Value *, 4> buffers = { match.rows.ids, match.rows.weight, match.rows.output, output };
    for (size_t lhs = 0; lhs < buffers.size(); ++lhs) {
        for (size_t rhs = lhs + 1; rhs < buffers.size(); ++rhs) {
            if (buffers[lhs]->storage_root == buffers[rhs]->storage_root) {
                return {};
            }
        }
    }
    if (!context.graph.index().node_index(scale_node, match.scale_index)) {
        return {};
    }

    match.scale_node = scale_node;
    match.output     = output;
    match.scale      = params->scale;
    match.bias       = params->bias;
    return match;
}

static GetRowsRmsNormMatch match_get_rows_rmsnorm_binary(const DispatchMatchContext & context) {
    GetRowsRmsNormMatch match;
    match.rows = match_get_rows_f32(context.graph, context.root_node);
    if (!match.rows.matched() || !context.graph.has_index() || match.rows.hidden_size < 128 ||
        match.rows.hidden_size > 32768 || match.rows.hidden_size % 128 != 0) {
        return {};
    }

    const GraphNode * rms = nullptr;
    for (const GraphNode * consumer : context.graph.index().consumers(match.rows.output->id)) {
        if (consumer != nullptr && consumer->op == GGML_OP_RMS_NORM && consumer->inputs.size() == 1 &&
            consumer->inputs[0] == match.rows.output->id) {
            if (rms != nullptr) {
                return {};
            }
            rms = consumer;
        }
    }
    if (rms == nullptr) {
        return {};
    }

    const RmsNormParams * params     = op_params_as<RmsNormParams>(rms->params);
    const Value *         rms_output = graph_value(context.graph, rms->output);
    if (params == nullptr || !std::isfinite(params->eps) || params->eps <= 0.0f || rms_output == nullptr ||
        rms_output->type != GGML_TYPE_F32 || !rms_output->contiguous || rms_output->ne != match.rows.output->ne) {
        return {};
    }

    const std::vector<const GraphNode *> & rms_consumers = context.graph.index().consumers(rms_output->id);
    if (rms_consumers.size() != 1 || rms_consumers.front() == nullptr || rms_consumers.front()->op != GGML_OP_MUL ||
        rms_consumers.front()->inputs.size() != 2) {
        return {};
    }
    const GraphNode * binary      = rms_consumers.front();
    const ValueId     weight_id   = binary->inputs[0] == rms_output->id ? binary->inputs[1] :
                                    binary->inputs[1] == rms_output->id ? binary->inputs[0] :
                                                                          ValueId();
    const Value *     norm_weight = graph_value(context.graph, weight_id);
    const Value *     output      = graph_value(context.graph, binary->output);
    if (weight_id.value < 0 || norm_weight == nullptr || output == nullptr || norm_weight->type != GGML_TYPE_F32 ||
        output->type != GGML_TYPE_F32 || !norm_weight->contiguous || !output->contiguous ||
        norm_weight->ne[0] != match.rows.hidden_size || norm_weight->ne[1] != 1 || norm_weight->ne[2] != 1 ||
        norm_weight->ne[3] != 1 || output->ne != match.rows.output->ne ||
        match.rows.output->storage_root == output->storage_root || norm_weight->storage_root == output->storage_root) {
        return {};
    }

    if (!context.graph.index().node_index(rms, match.rms_node_index) ||
        !context.graph.index().node_index(binary, match.binary_node_index)) {
        return {};
    }
    match.rms_node          = rms;
    match.binary_node       = binary;
    match.norm_weight       = norm_weight;
    match.normalized_output = output;
    match.epsilon           = params->eps;
    return match;
}

static Dispatch make_get_rows_dispatch(const GetRowsMatch & match) {
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kGetRowsF32Kernel);
    dispatch.kernel.workload_specialization = WorkloadSpecialization::Dynamic;
    add_common_integer_parameters(dispatch, match);
    dispatch.kernel.compile_parameters.emplace("ggml.get_rows_f32.token_capacity",
                                               to_config_value(token_capacity_class(match.token_count)));
    dispatch.kernel.compile_parameters.emplace("ggml.get_rows_f32.hidden_capacity", to_config_value(match.hidden_size));
    dispatch.kernel.compile_parameters.emplace("ggml.get_rows_f32.weight_format",
                                               to_config_value(match.weight_format_value));
    add_primary_bindings(dispatch, match);
    return dispatch;
}

static Dispatch make_get_rows_next_dispatch(const GetRowsMatch &     match,
                                            CommonMulMatWeightFormat next_format,
                                            ValueId                  next_value,
                                            size_t                   next_byte_count) {
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kGetRowsF32NextKernel);
    add_common_integer_parameters(dispatch, match);
    add_common_compile_parameters(dispatch, match);
    dispatch.kernel.compile_parameters.emplace("ggml.get_rows_f32.next_format",
                                               to_config_value(common_mul_mat_format_config_value(next_format)));
    add_primary_bindings(dispatch, match);
    dispatch.bindings.push_back({ next_value, 0, next_byte_count });
    return dispatch;
}

static bool match_get_rows_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const GetRowsMatch match = match_get_rows_f32(context.graph, context.root_node);
    if (!match.matched()) {
        return false;
    }

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(make_get_rows_dispatch(match));
    return true;
}

static bool match_get_rows_scale_f32_dispatch(const DispatchMatchContext & context,
                                              DispatchMatch &              dispatch_match) {
    const GetRowsScaleMatch match = match_get_rows_scale_f32(context);
    if (!match.matched() || match.scale_index >= context.covered_nodes.size() ||
        context.covered_nodes[context.root_index] || context.covered_nodes[match.scale_index]) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kGetRowsScaleF32Kernel);
    add_common_integer_parameters(dispatch, match.rows);
    add_common_compile_parameters(dispatch, match.rows);
    dispatch.kernel.compile_parameters.emplace("ggml.get_rows_scale_f32.scale", to_config_value(match.scale));
    dispatch.kernel.compile_parameters.emplace("ggml.get_rows_scale_f32.bias", to_config_value(match.bias));
    dispatch.bindings.push_back({ match.rows.ids->id, 0, match.rows.ids->byte_count });
    dispatch.bindings.push_back({ match.rows.weight->id, 0, match.rows.weight->byte_count });
    dispatch.bindings.push_back({ match.rows.output->id, 0, match.rows.output->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.covered_nodes.push_back(match.scale_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_get_rows_f32_next_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const GetRowsMatch match = match_get_rows_f32(context.graph, context.root_node);
    if (!match.matched()) {
        return false;
    }

    const CommonActivationPublicationCapabilities capabilities = {
        DispatchActivationInputQ8_1X4,
        DispatchActivationInputNone,
        DispatchActivationProducerRequirementNone,
        DispatchActivationInputNone,
        DispatchActivationConsumerSourceQwen,
    };
    const CommonActivationPublicationPlan publication_plan =
        common_select_activation_publication_plan(context, *match.output, capabilities);
    if (!publication_plan.matched()) {
        return false;
    }
    const CommonActivationPublicationDemand & demand = publication_plan.preferred();

    CommonMulMatWeightFormat next_format;
    if (!common_mul_mat_alternate_format_for_type(GGML_TYPE_Q8_1, next_format)) {
        return false;
    }
    CommonActivationPublication publication;
    constexpr const char * publication_name = "common.get_rows.q8_1_x4";
    if (!common_reserve_activation_publication(context, dispatch_match, *match.output, demand,
                                               publication_name, publication) ||
        !common_append_activation_publication(dispatch_match, publication, publication_name)) {
        return false;
    }
    dispatch_match.dispatches.push_back(
        make_get_rows_next_dispatch(match, next_format, publication.alternate_value, publication.byte_count));
    dispatch_match.covered_nodes.push_back(context.root_index);
    return dispatch_match.status.success();
}

static bool match_get_rows_rmsnorm_binary_dispatch(const DispatchMatchContext & context,
                                                   DispatchMatch &              dispatch_match) {
    const GetRowsRmsNormMatch match = match_get_rows_rmsnorm_binary(context);
    if (!match.matched() || match.rms_node_index >= context.covered_nodes.size() ||
        match.binary_node_index >= context.covered_nodes.size() || context.covered_nodes[context.root_index] ||
        context.covered_nodes[match.rms_node_index] || context.covered_nodes[match.binary_node_index]) {
        return false;
    }

    const CommonActivationPublicationCapabilities capabilities = {
        DispatchActivationInputQ8_1X4 | DispatchActivationInputF16Row,
        DispatchActivationInputQ8_1X4 | DispatchActivationInputF16Row,
        DispatchActivationProducerRequirementNone,
        DispatchActivationInputQ8_1X4 | DispatchActivationInputF16Row,
    };
    const CommonActivationPublicationPlan publication_plan =
        common_select_activation_publication_plan(context, *match.normalized_output, capabilities);
    const CommonActivationPublicationDemand * q8_demand =
        publication_plan.find(CommonActivationPublicationFormat::Q8_1X4);
    const CommonActivationPublicationDemand * f16_demand =
        publication_plan.find(CommonActivationPublicationFormat::F16Row);
    const bool publish = publication_plan.matched();
    if (publish && (q8_demand == nullptr || f16_demand == nullptr)) {
        return false;
    }
    CommonActivationPublication q8_publication;
    CommonActivationPublication f16_publication;
    const bool q8_reserved = publish ?
        common_reserve_activation_publication(context, dispatch_match, *match.normalized_output, *q8_demand,
                                              "common.get_rows_rmsnorm.q8_1_x4", q8_publication) :
        common_reserve_private_activation_output(context, dispatch_match, *match.normalized_output,
                                                 CommonActivationPublicationFormat::Q8_1X4,
                                                 "common.get_rows_rmsnorm.private_q8_1_x4", q8_publication);
    const bool f16_reserved = publish ?
        common_reserve_activation_publication(context, dispatch_match, *match.normalized_output, *f16_demand,
                                              "common.get_rows_rmsnorm.f16", f16_publication) :
        common_reserve_private_activation_output(context, dispatch_match, *match.normalized_output,
                                                 CommonActivationPublicationFormat::F16Row,
                                                 "common.get_rows_rmsnorm.private_f16", f16_publication);
    if (!q8_reserved || !f16_reserved ||
        (publish && (!common_append_activation_publication(dispatch_match, q8_publication,
                                                           "common.get_rows_rmsnorm.q8_1_x4") ||
                     !common_append_activation_publication(dispatch_match, f16_publication,
                                                           "common.get_rows_rmsnorm.f16")))) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kGetRowsRmsNormBinaryQ8_1X4F16Kernel);
    add_common_integer_parameters(dispatch, match.rows);
    add_common_compile_parameters(dispatch, match.rows);
    dispatch.kernel.compile_parameters.emplace("ggml.get_rows_rmsnorm.rms_epsilon", to_config_value(match.epsilon));
    dispatch.bindings.push_back({ match.rows.ids->id, 0, match.rows.ids->byte_count });
    dispatch.bindings.push_back({ match.rows.weight->id, 0, match.rows.weight->byte_count });
    dispatch.bindings.push_back({ match.norm_weight->id, 0, match.norm_weight->byte_count });
    dispatch.bindings.push_back({ match.rows.output->id, 0, match.rows.output->byte_count });
    dispatch.bindings.push_back({ match.normalized_output->id, 0, match.normalized_output->byte_count });
    dispatch.bindings.push_back(q8_publication.binding());
    dispatch.bindings.push_back(f16_publication.binding());

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.covered_nodes.push_back(match.rms_node_index);
    dispatch_match.covered_nodes.push_back(match.binary_node_index);
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return dispatch_match.status.success();
}

}  // namespace

void register_get_rows_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.get_rows_scale.f32",
        GGML_OP_GET_ROWS,
        DispatchMatchKind::Fused,
        300,
        DispatchSource::Common,
        match_get_rows_scale_f32_dispatch,
    });
    registry.add({
        "common.get_rows_rmsnorm_binary.q8_1_x4_f16",
        GGML_OP_GET_ROWS,
        DispatchMatchKind::Fused,
        330,
        DispatchSource::Common,
        match_get_rows_rmsnorm_binary_dispatch,
    });
    registry.add({
        "common.get_rows.f32_next",
        GGML_OP_GET_ROWS,
        DispatchMatchKind::Fused,
        150,
        DispatchSource::Common,
        match_get_rows_f32_next_dispatch,
    });
    registry.add({
        "common.get_rows.f32",
        GGML_OP_GET_ROWS,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Common,
        match_get_rows_f32_dispatch,
    });
}

}  // namespace ggml::hrx
