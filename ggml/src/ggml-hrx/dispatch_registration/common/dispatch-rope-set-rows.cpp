#include "dispatch-rope-set-rows.h"

#include "dispatch-layout-utils.h"
#include "dispatch-rope-utils.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kRopeF32Kernel       = GGML_HRX_KERNEL_REF("loom_libs", "ggml_rope_f32");
static constexpr KernelCatalogRef kRopeConcatF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_rope_concat_f32");
static constexpr KernelCatalogRef kRopeScaleF32Kernel  = GGML_HRX_KERNEL_REF("loom_libs", "ggml_rope_scale_f32");
static constexpr KernelCatalogRef kRopeTokenScaleF32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_rope_token_scale_f32");
static constexpr KernelCatalogRef kSetRowsKernel        = GGML_HRX_KERNEL_REF("loom_libs", "ggml_set_rows");
static constexpr KernelCatalogRef kRopeSetRowsF32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_rope_set_rows_f32");

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static std::string to_config_value(int64_t value) {
    return std::to_string(value);
}

static bool same_shape(const Value & lhs, const Value & rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i]) {
            return false;
        }
    }
    return true;
}

static bool is_1d_shape(const Value & value, int64_t ne0) {
    return value.ne[0] == ne0 && value.ne[1] == 1 && value.ne[2] == 1 && value.ne[3] == 1;
}

static bool is_2d_shape(const Value & value, int64_t ne0, int64_t ne1) {
    return value.ne[0] == ne0 && value.ne[1] == ne1 && value.ne[2] == 1 && value.ne[3] == 1;
}

static std::string value_layout_for_log(const Value * value) {
    if (value == nullptr) {
        return "null";
    }
    std::ostringstream out;
    out << ggml_type_name(value->type) << " ne=[" << value->ne[0] << "," << value->ne[1] << "," << value->ne[2] << ","
        << value->ne[3] << "] nb=[" << value->nb[0] << "," << value->nb[1] << "," << value->nb[2] << "," << value->nb[3]
        << "] contiguous=" << (value->contiguous ? 1 : 0);
    if (value->alias_source.value >= 0) {
        out << " alias=" << value->alias_source.value << " storage_offset=" << value->storage_offset;
    }
    return out.str();
}

static void log_rope_reject(Status *      status,
                            const char *  reason,
                            const Value * input,
                            const Value * positions,
                            const Value * output,
                            const Value * freq_factors = nullptr) {
    if (status == nullptr) {
        return;
    }
    status->log("ROPE matcher rejected node: %s input=%s positions=%s output=%s freq_factors=%s", reason,
                value_layout_for_log(input).c_str(), value_layout_for_log(positions).c_str(),
                value_layout_for_log(output).c_str(), value_layout_for_log(freq_factors).c_str());
}

static void log_set_rows_reject(Status *      status,
                                const char *  reason,
                                const Value * rows,
                                const Value * indices,
                                const Value * cache,
                                const Value * output) {
    if (status == nullptr) {
        return;
    }
    status->log("SET_ROWS matcher rejected node: %s rows=%s indices=%s cache=%s output=%s", reason,
                value_layout_for_log(rows).c_str(), value_layout_for_log(indices).c_str(),
                value_layout_for_log(cache).c_str(), value_layout_for_log(output).c_str());
}

static bool is_rope_shape(const Value & value) {
    return value.ne[0] >= 4 && value.ne[0] <= 1024 && value.ne[0] % 4 == 0 && value.ne[1] >= 1 && value.ne[1] <= 64 &&
           value.ne[2] >= 1 && value.ne[2] <= 2048 && value.ne[3] == 1;
}

static bool is_packed_f32_rope_layout(const Value & value) {
    if (value.type != GGML_TYPE_F32 || !is_rope_shape(value) || value.nb[0] != sizeof(float)) {
        return false;
    }
    return value.nb[1] == static_cast<size_t>(value.ne[0]) * sizeof(float) &&
           value.nb[2] == static_cast<size_t>(value.ne[1]) * value.nb[1] &&
           value.nb[3] == static_cast<size_t>(value.ne[2]) * value.nb[2];
}

static bool is_supported_rope_input_layout(const Value & value, size_t & span_elements) {
    if (!is_rope_shape(value) || value.nb[1] % sizeof(float) != 0 || value.nb[2] % sizeof(float) != 0 ||
        !strided_f32_storage_span_elements(value, span_elements)) {
        return false;
    }

    const size_t stride1 = value.nb[1] / sizeof(float);
    const size_t stride2 = value.nb[2] / sizeof(float);
    if (stride1 < static_cast<size_t>(value.ne[0]) || stride2 < static_cast<size_t>(value.ne[1]) * stride1) {
        return false;
    }

    const size_t token_span = static_cast<size_t>(value.ne[2] - 1) * stride2;
    const size_t head_span  = static_cast<size_t>(value.ne[1] - 1) * stride1;
    const size_t required   = token_span + head_span + static_cast<size_t>(value.ne[0]);
    return required <= span_elements;
}

static bool is_supported_rope_params(const RopeParams & params, int64_t head_size) {
    const bool supported_mode = params.mode == GGML_ROPE_TYPE_NORMAL || params.mode == GGML_ROPE_TYPE_NEOX;
    return params.n_dims >= 4 && params.n_dims <= head_size && params.n_dims % 4 == 0 && supported_mode &&
           std::isfinite(params.freq_base) && params.freq_base > 0.0f && std::isfinite(params.freq_scale) &&
           params.freq_scale > 0.0f && std::isfinite(params.ext_factor) && std::isfinite(params.attn_factor);
}

static bool build_rope_theta_table(const GraphNode &      rope,
                                   int64_t                n_dims,
                                   std::vector<uint8_t> & data,
                                   float &                mscale) {
    const RopeParams * params = op_params_as<RopeParams>(rope.params);
    if (params == nullptr) {
        return false;
    }

    RopeFrequencyTable table;
    if (!build_rope_frequency_table(*params, n_dims, table)) {
        return false;
    }
    data   = std::move(table.data);
    mscale = table.mscale;
    return true;
}

static void build_unit_frequency_factors(int64_t n_dims, std::vector<uint8_t> & data) {
    data.resize(static_cast<size_t>(n_dims / 2) * sizeof(float));
    const float one = 1.0f;
    for (int64_t i = 0; i < n_dims / 2; ++i) {
        std::memcpy(data.data() + static_cast<size_t>(i) * sizeof(float), &one, sizeof(one));
    }
}

static bool format_value(ggml_type type, int64_t & value) {
    switch (type) {
        case GGML_TYPE_F16:
            value = 16;
            return true;
        case GGML_TYPE_F32:
            value = 32;
            return true;
        default:
            return false;
    }
}

static bool supported_token_count(int64_t token_count) {
    return token_count >= 1 && token_count <= 2048;
}

static int64_t token_capacity_class(int64_t token_count) {
    int64_t capacity = 1;
    while (capacity < token_count) {
        capacity *= 2;
    }
    return capacity;
}

static bool supported_cache_row_count(int64_t row_count) {
    return row_count >= 1 && row_count <= 1048576;
}

static bool supported_hidden_size(int64_t hidden_size) {
    return hidden_size >= 4 && hidden_size <= 32768 && hidden_size % 4 == 0;
}

static bool set_rows_supported_row_layout(const Value & rows,
                                          int64_t       hidden_size,
                                          int64_t       token_count,
                                          int64_t &     input_stride) {
    if (!is_2d_shape(rows, hidden_size, token_count)) {
        return false;
    }

    const size_t element_size = ggml_type_size(rows.type);
    if (rows.nb[0] != element_size || rows.nb[1] % element_size != 0) {
        return false;
    }

    input_stride = static_cast<int64_t>(rows.nb[1] / element_size);
    return input_stride >= hidden_size && input_stride <= 1048576;
}

static bool supported_set_rows_input_layout(const Value & rows,
                                            int64_t       hidden_size,
                                            int64_t       token_count,
                                            int64_t &     input_stride,
                                            size_t &      rows_span_bytes) {
    if (!set_rows_supported_row_layout(rows, hidden_size, token_count, input_stride)) {
        return false;
    }

    if (rows.contiguous) {
        rows_span_bytes = rows.byte_count;
        return true;
    }

    if (rows.type != GGML_TYPE_F32) {
        return false;
    }

    return strided_f32_storage_span_bytes(rows, rows_span_bytes);
}

static ValueId next_match_transient_value(const DispatchMatchContext & context, const DispatchMatch & dispatch_match) {
    return ValueId(context.next_plan_value.value + static_cast<int32_t>(dispatch_match.transients.size()) +
                   static_cast<int32_t>(dispatch_match.completion_counter_requests.size()));
}

struct RopeMatch {
    const GraphNode *    node               = nullptr;
    const Value *        input              = nullptr;
    const Value *        positions          = nullptr;
    const Value *        freq_factors       = nullptr;
    const Value *        output             = nullptr;
    int64_t              token_count        = 0;
    int64_t              head_count         = 0;
    int64_t              head_size          = 0;
    int64_t              n_dims             = 0;
    int64_t              input_stride1      = 0;
    int64_t              input_stride2      = 0;
    size_t               input_span         = 0;
    size_t               input_span_bytes   = 0;
    float                rope_mscale        = 1.0f;
    int64_t              mode               = 0;
    size_t               theta_bytes        = 0;
    size_t               freq_factors_bytes = 0;
    std::vector<uint8_t> theta_data;
    std::vector<uint8_t> freq_factors_data;

    bool matched() const { return node != nullptr && input != nullptr && positions != nullptr && output != nullptr; }
};

struct SetRowsMatch {
    const GraphNode * node            = nullptr;
    const Value *     rows            = nullptr;
    const Value *     indices         = nullptr;
    const Value *     cache           = nullptr;
    const Value *     output          = nullptr;
    size_t            rows_span_bytes = 0;
    int64_t           input_stride    = 0;
    int64_t           row_format      = 0;
    int64_t           output_format   = 0;
    int64_t           token_count     = 0;
    int64_t           cache_row_count = 0;
    int64_t           hidden_size     = 0;

    bool matched() const { return node != nullptr && rows != nullptr && indices != nullptr && output != nullptr; }
};

struct RopeSetRowsMatch {
    RopeMatch         rope;
    SetRowsMatch      set_rows;
    const GraphNode * layout         = nullptr;
    const Value *     published_rows = nullptr;
    size_t            set_rows_idx   = 0;
    size_t            layout_idx     = 0;

    bool matched() const { return rope.matched() && set_rows.matched() && published_rows != nullptr; }
};

struct RopeScaleMatch {
    RopeMatch         rope;
    const GraphNode * scale_node = nullptr;
    const Value *     scales     = nullptr;
    const Value *     output     = nullptr;
    float             scalar     = 1.0f;
    bool              per_token  = false;

    bool matched() const { return rope.matched() && scale_node != nullptr && output != nullptr; }
};

struct RopeConcatMatch {
    RopeMatch         rope;
    const GraphNode * concat           = nullptr;
    const Value *     prefix           = nullptr;
    const Value *     output           = nullptr;
    size_t            concat_idx       = 0;
    size_t            input_span_bytes = 0;

    bool matched() const { return rope.matched() && concat != nullptr && prefix != nullptr && output != nullptr; }
};

static RopeMatch match_rope_f32(const Graph & graph, const GraphNode * node, Status * status = nullptr) {
    RopeMatch match;
    if (node == nullptr || node->op != GGML_OP_ROPE || node->inputs.size() < 2 || node->inputs.size() > 3) {
        log_rope_reject(status, "root is not a supported ROPE arity", nullptr, nullptr, nullptr);
        return match;
    }

    const Value * input      = graph_value(graph, node->inputs[0]);
    const Value * positions  = graph_value(graph, node->inputs[1]);
    const Value * output     = graph_value(graph, node->output);
    size_t        input_span = 0;
    if (input == nullptr || positions == nullptr || output == nullptr || input->type != GGML_TYPE_F32 ||
        output->type != GGML_TYPE_F32 || positions->type != GGML_TYPE_I32 ||
        !is_supported_rope_input_layout(*input, input_span) || !is_packed_f32_rope_layout(*output) ||
        !positions->contiguous || !same_shape(*input, *output)) {
        log_rope_reject(status, "input/output/positions shape or layout is unsupported", input, positions, output);
        return {};
    }

    const int64_t      head_size   = input->ne[0];
    const int64_t      head_count  = input->ne[1];
    const int64_t      token_count = input->ne[2];
    const RopeParams * params      = op_params_as<RopeParams>(node->params);
    if (params == nullptr || !is_supported_rope_params(*params, head_size) || !is_1d_shape(*positions, token_count)) {
        std::string reason = "ROPE params or positions shape is unsupported";
        if (params == nullptr) {
            reason += " params=missing";
        } else {
            std::ostringstream out;
            out << reason << " params={n_dims=" << params->n_dims << ", mode=" << params->mode
                << ", n_ctx_orig=" << params->n_ctx_orig << ", freq_base=" << params->freq_base
                << ", freq_scale=" << params->freq_scale << ", ext_factor=" << params->ext_factor
                << ", attn_factor=" << params->attn_factor << ", beta_fast=" << params->beta_fast
                << ", beta_slow=" << params->beta_slow << "}";
            reason = out.str();
        }
        log_rope_reject(status, reason.c_str(), input, positions, output);
        return {};
    }
    const int64_t n_dims = params->n_dims;

    std::vector<uint8_t> theta_data;
    float                rope_mscale = 1.0f;
    if (!build_rope_theta_table(*node, n_dims, theta_data, rope_mscale)) {
        log_rope_reject(status, "failed to build supported ROPE theta table", input, positions, output);
        return {};
    }

    const Value *        freq_factors       = nullptr;
    size_t               freq_factors_bytes = 0;
    std::vector<uint8_t> freq_factors_data;
    if (node->inputs.size() == 3) {
        freq_factors = graph_value(graph, node->inputs[2]);
        if (freq_factors == nullptr || freq_factors->type != GGML_TYPE_F32 || !freq_factors->contiguous ||
            !is_1d_shape(*freq_factors, n_dims / 2)) {
            log_rope_reject(status, "explicit frequency factors shape or layout is unsupported", input, positions,
                            output, freq_factors);
            return {};
        }
        freq_factors_bytes = freq_factors->byte_count;
    } else {
        build_unit_frequency_factors(n_dims, freq_factors_data);
        freq_factors_bytes = freq_factors_data.size();
    }

    match.node               = node;
    match.input              = input;
    match.positions          = positions;
    match.freq_factors       = freq_factors;
    match.output             = output;
    match.token_count        = token_count;
    match.head_count         = head_count;
    match.head_size          = head_size;
    match.n_dims             = n_dims;
    match.input_stride1      = static_cast<int64_t>(input->nb[1] / sizeof(float));
    match.input_stride2      = static_cast<int64_t>(input->nb[2] / sizeof(float));
    match.input_span         = input_span;
    match.input_span_bytes   = input_span * sizeof(float);
    match.rope_mscale        = rope_mscale;
    match.mode               = params->mode;
    match.theta_bytes        = theta_data.size();
    match.freq_factors_bytes = freq_factors_bytes;
    match.theta_data         = std::move(theta_data);
    match.freq_factors_data  = std::move(freq_factors_data);
    return match;
}

static SetRowsMatch match_set_rows_2d(const Graph & graph, const GraphNode * node, Status * status = nullptr) {
    SetRowsMatch match;
    if (node == nullptr || node->op != GGML_OP_SET_ROWS || node->inputs.size() != 3) {
        log_set_rows_reject(status, "root is not a supported SET_ROWS arity", nullptr, nullptr, nullptr, nullptr);
        return match;
    }

    const Value * rows    = graph_value(graph, node->inputs[0]);
    const Value * indices = graph_value(graph, node->inputs[1]);
    const Value * cache   = graph_value(graph, node->inputs[2]);
    const Value * output  = graph_value(graph, node->output);
    if (rows == nullptr || indices == nullptr || cache == nullptr || output == nullptr || !indices->contiguous ||
        !cache->contiguous || indices->type != GGML_TYPE_I64 || output->type != cache->type ||
        !same_shape(*cache, *output) || !graph.values().same_storage(cache->id, output->id)) {
        log_set_rows_reject(status, "input/output shape or storage is unsupported", rows, indices, cache, output);
        return {};
    }

    int64_t row_format    = 0;
    int64_t output_format = 0;
    if (!format_value(rows->type, row_format) || !format_value(output->type, output_format)) {
        log_set_rows_reject(status, "input or output format is unsupported", rows, indices, cache, output);
        return {};
    }
    if (rows->type == GGML_TYPE_F16 && output->type != GGML_TYPE_F16) {
        log_set_rows_reject(status, "f16 input requires f16 output", rows, indices, cache, output);
        return {};
    }

    const int64_t hidden_size     = rows->ne[0];
    const int64_t token_count     = rows->ne[1];
    const int64_t cache_row_count = cache->ne[1];
    int64_t       input_stride    = 0;
    size_t        rows_span_bytes = 0;
    if (!supported_set_rows_input_layout(*rows, hidden_size, token_count, input_stride, rows_span_bytes) ||
        !is_1d_shape(*indices, token_count) || !is_2d_shape(*cache, hidden_size, cache_row_count) ||
        !supported_hidden_size(hidden_size) || !supported_token_count(token_count) ||
        !supported_cache_row_count(cache_row_count)) {
        log_set_rows_reject(status, "shape or rows layout is unsupported", rows, indices, cache, output);
        return {};
    }

    match.node            = node;
    match.rows            = rows;
    match.indices         = indices;
    match.cache           = cache;
    match.output          = output;
    match.rows_span_bytes = rows_span_bytes;
    match.input_stride    = input_stride;
    match.row_format      = row_format;
    match.output_format   = output_format;
    match.token_count     = token_count;
    match.cache_row_count = cache_row_count;
    match.hidden_size     = hidden_size;
    return match;
}

static void add_rope_compile_parameters(Dispatch & dispatch,
                                        const RopeMatch & match,
                                        const char *      prefix,
                                        bool              include_input_span = true) {
    dispatch.kernel.compile_parameters.emplace(std::string(prefix) + ".head_size", to_config_value(match.head_size));
    dispatch.kernel.compile_parameters.emplace(std::string(prefix) + ".n_dims", to_config_value(match.n_dims));
    dispatch.kernel.compile_parameters.emplace(std::string(prefix) + ".head_count", to_config_value(match.head_count));
    dispatch.kernel.compile_parameters.emplace(std::string(prefix) + ".token_capacity",
                                               to_config_value(match.token_count));
    dispatch.kernel.compile_parameters.emplace(std::string(prefix) + ".input_stride1",
                                               to_config_value(match.input_stride1));
    dispatch.kernel.compile_parameters.emplace(std::string(prefix) + ".input_stride2",
                                               to_config_value(match.input_stride2));
    if (include_input_span) {
        dispatch.kernel.compile_parameters.emplace(std::string(prefix) + ".input_span",
                                                   to_config_value(match.input_span));
    }
    dispatch.kernel.compile_parameters.emplace(std::string(prefix) + ".mscale",
                                               rope_mscale_config_value(match.rope_mscale));
    dispatch.kernel.compile_parameters.emplace(std::string(prefix) + ".mode", to_config_value(match.mode));
}

static void add_set_rows_compile_parameters(Dispatch & dispatch, const SetRowsMatch & match) {
    dispatch.kernel.compile_parameters.emplace("ggml.set_rows.token_capacity", to_config_value(match.token_count));
    dispatch.kernel.compile_parameters.emplace("ggml.set_rows.hidden_capacity", to_config_value(match.hidden_size));
    dispatch.kernel.compile_parameters.emplace("ggml.set_rows.input_format", to_config_value(match.row_format));
    dispatch.kernel.compile_parameters.emplace("ggml.set_rows.output_format", to_config_value(match.output_format));
    dispatch.kernel.compile_parameters.emplace("ggml.set_rows.input_stride", to_config_value(match.input_stride));
}

static ValueId add_constant_binding(const DispatchMatchContext & context,
                                    DispatchMatch &              dispatch_match,
                                    const char *                 name,
                                    size_t                       byte_count,
                                    const std::vector<uint8_t> & data) {
    const ValueId value = next_match_transient_value(context, dispatch_match);
    dispatch_match.transients.push_back({ value, name, byte_count, 256 });
    dispatch_match.constant_initializations.push_back({
        value,
        name,
        0,
        data,
    });
    return value;
}

static std::pair<ValueId, ValueId> add_rope_frequency_bindings(const DispatchMatchContext & context,
                                                               const RopeMatch &            match,
                                                               DispatchMatch &              dispatch_match,
                                                               const char *                 prefix) {
    const std::string theta_name = std::string(prefix) + ".theta";
    const ValueId     theta =
        add_constant_binding(context, dispatch_match, theta_name.c_str(), match.theta_bytes, match.theta_data);
    if (match.freq_factors != nullptr) {
        return { theta, match.freq_factors->id };
    }

    const std::string factors_name = std::string(prefix) + ".freq_factors";
    const ValueId     freq_factors = add_constant_binding(context, dispatch_match, factors_name.c_str(),
                                                          match.freq_factors_bytes, match.freq_factors_data);
    return { theta, freq_factors };
}

static Dispatch make_rope_dispatch(const DispatchMatchContext & context,
                                   const RopeMatch &            match,
                                   DispatchMatch &              dispatch_match) {
    const auto [theta, freq_factors] = add_rope_frequency_bindings(context, match, dispatch_match, "common.rope");
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kRopeF32Kernel);
    dispatch.kernel.workload_specialization = WorkloadSpecialization::Dynamic;
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.integer_parameters.emplace("input_span", match.input_span);
    add_rope_compile_parameters(dispatch, match, "ggml.rope_f32", false);
    dispatch.kernel.compile_parameters["ggml.rope_f32.token_capacity"] =
        to_config_value(token_capacity_class(match.token_count));
    dispatch.bindings.push_back({ match.positions->id, 0, match.positions->byte_count });
    dispatch.bindings.push_back({ match.input->storage_root, match.input->storage_offset, match.input_span_bytes });
    dispatch.bindings.push_back({ theta, 0, match.theta_bytes });
    dispatch.bindings.push_back({ freq_factors, 0, match.freq_factors_bytes });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    return dispatch;
}

static Dispatch make_rope_scale_dispatch(const DispatchMatchContext & context,
                                         const RopeScaleMatch &       match,
                                         DispatchMatch &              dispatch_match) {
    const auto [theta, freq_factors] =
        add_rope_frequency_bindings(context, match.rope, dispatch_match, "common.rope_scale");
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(match.per_token ? kRopeTokenScaleF32Kernel : kRopeScaleF32Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.rope.token_count);
    add_rope_compile_parameters(dispatch, match.rope, "ggml.rope_f32");
    if (!match.per_token) {
        dispatch.kernel.compile_parameters.emplace("ggml.rope_scale_f32.scale", rope_mscale_config_value(match.scalar));
    }
    dispatch.bindings.push_back({ match.rope.positions->id, 0, match.rope.positions->byte_count });
    dispatch.bindings.push_back(
        { match.rope.input->storage_root, match.rope.input->storage_offset, match.rope.input_span_bytes });
    dispatch.bindings.push_back({ theta, 0, match.rope.theta_bytes });
    dispatch.bindings.push_back({ freq_factors, 0, match.rope.freq_factors_bytes });
    if (match.per_token) {
        dispatch.bindings.push_back(
            { match.scales->storage_root, match.scales->storage_offset, match.scales->byte_count });
    }
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    return dispatch;
}

static Dispatch make_set_rows_dispatch(const SetRowsMatch & match) {
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kSetRowsKernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.token_count);
    dispatch.kernel.integer_parameters.emplace("cache_row_count", match.cache_row_count);
    dispatch.kernel.integer_parameters.emplace("hidden_size", match.hidden_size);
    add_set_rows_compile_parameters(dispatch, match);
    dispatch.bindings.push_back({ match.rows->storage_root, match.rows->storage_offset, match.rows_span_bytes });
    dispatch.bindings.push_back({ match.indices->id, 0, match.indices->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    return dispatch;
}

static Dispatch make_rope_set_rows_dispatch(const DispatchMatchContext & context,
                                            const RopeSetRowsMatch &     match,
                                            DispatchMatch &              dispatch_match) {
    const auto [theta, freq_factors] =
        add_rope_frequency_bindings(context, match.rope, dispatch_match, "common.rope_set_rows");
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kRopeSetRowsF32Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.rope.token_count);
    dispatch.kernel.integer_parameters.emplace("cache_row_count", match.set_rows.cache_row_count);
    add_rope_compile_parameters(dispatch, match.rope, "ggml.rope_set_rows_f32");
    dispatch.kernel.compile_parameters.emplace("ggml.rope_set_rows_f32.output_format",
                                               to_config_value(match.set_rows.output_format));
    dispatch.bindings.push_back({ match.rope.positions->id, 0, match.rope.positions->byte_count });
    dispatch.bindings.push_back({ match.set_rows.indices->id, 0, match.set_rows.indices->byte_count });
    dispatch.bindings.push_back(
        { match.rope.input->storage_root, match.rope.input->storage_offset, match.rope.input_span_bytes });
    dispatch.bindings.push_back({ theta, 0, match.rope.theta_bytes });
    dispatch.bindings.push_back({ freq_factors, 0, match.rope.freq_factors_bytes });
    dispatch.bindings.push_back({ match.set_rows.output->id, 0, match.set_rows.output->byte_count });
    return dispatch;
}

static Dispatch make_rope_concat_dispatch(const DispatchMatchContext & context,
                                          const RopeConcatMatch &      match,
                                          DispatchMatch &              dispatch_match) {
    const auto [theta, freq_factors] =
        add_rope_frequency_bindings(context, match.rope, dispatch_match, "common.rope_concat");
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kRopeConcatF32Kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.rope.token_count);
    const char * prefix = "ggml.rope_concat_f32.";
    dispatch.kernel.compile_parameters.emplace(std::string(prefix) + "prefix_size",
                                               to_config_value(match.prefix->ne[0]));
    dispatch.kernel.compile_parameters.emplace(std::string(prefix) + "rope_size",
                                               to_config_value(match.rope.head_size));
    dispatch.kernel.compile_parameters.emplace(std::string(prefix) + "n_dims", to_config_value(match.rope.n_dims));
    dispatch.kernel.compile_parameters.emplace(std::string(prefix) + "head_count",
                                               to_config_value(match.rope.head_count));
    dispatch.kernel.compile_parameters.emplace(std::string(prefix) + "token_capacity",
                                               to_config_value(match.rope.token_count));
    dispatch.kernel.compile_parameters.emplace(std::string(prefix) + "input_stride1",
                                               to_config_value(match.rope.input_stride1));
    dispatch.kernel.compile_parameters.emplace(std::string(prefix) + "input_stride2",
                                               to_config_value(match.rope.input_stride2));
    dispatch.kernel.compile_parameters.emplace(std::string(prefix) + "input_span",
                                               to_config_value(match.input_span_bytes / sizeof(float)));
    dispatch.kernel.compile_parameters.emplace(std::string(prefix) + "mscale",
                                               rope_mscale_config_value(match.rope.rope_mscale));
    dispatch.kernel.compile_parameters.emplace(std::string(prefix) + "mode", to_config_value(match.rope.mode));
    dispatch.bindings.push_back({ match.rope.positions->id, 0, match.rope.positions->byte_count });
    dispatch.bindings.push_back({ match.prefix->storage_root, match.prefix->storage_offset, match.input_span_bytes });
    dispatch.bindings.push_back({ theta, 0, match.rope.theta_bytes });
    dispatch.bindings.push_back({ freq_factors, 0, match.rope.freq_factors_bytes });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });
    return dispatch;
}

static RopeConcatMatch match_rope_concat_f32(const Graph & graph, const GraphNode * node) {
    RopeConcatMatch match;
    match.rope = match_rope_f32(graph, node);
    if (!match.rope.matched() || match.rope.mode != GGML_ROPE_TYPE_NEOX || !graph.has_index() ||
        !graph.index().has_single_consumer(node->output)) {
        return {};
    }

    const GraphNode * concat = graph.index().consumers(node->output).front();
    if (concat == nullptr || concat->op != GGML_OP_CONCAT || concat->inputs.size() != 2 ||
        concat->inputs[1] != node->output || !graph.index().node_index(concat, match.concat_idx)) {
        return {};
    }

    const Value * prefix = graph_value(graph, concat->inputs[0]);
    const Value * output = graph_value(graph, concat->output);
    if (prefix == nullptr || output == nullptr || prefix->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
        prefix->ne[0] < 4 || prefix->ne[0] > 1024 || prefix->ne[0] % 4 != 0 ||
        prefix->ne[0] + match.rope.head_size > 1024 || !is_packed_f32_rope_layout(*output) ||
        output->ne[0] != prefix->ne[0] + match.rope.head_size || prefix->nb[0] != sizeof(float) ||
        prefix->storage_root != match.rope.input->storage_root || prefix->storage_root == output->storage_root ||
        match.rope.input->storage_offset !=
            prefix->storage_offset + static_cast<size_t>(prefix->ne[0]) * sizeof(float)) {
        return {};
    }
    for (int dim = 1; dim < GGML_MAX_DIMS; ++dim) {
        if (prefix->ne[dim] != match.rope.input->ne[dim] || prefix->ne[dim] != output->ne[dim] ||
            prefix->nb[dim] != match.rope.input->nb[dim]) {
            return {};
        }
    }

    size_t prefix_span_bytes = 0;
    if (!strided_f32_storage_span_bytes(*prefix, prefix_span_bytes)) {
        return {};
    }
    const size_t rope_offset_bytes = match.rope.input->storage_offset - prefix->storage_offset;
    match.input_span_bytes         = std::max(prefix_span_bytes, rope_offset_bytes + match.rope.input_span_bytes);
    match.concat                   = concat;
    match.prefix                   = prefix;
    match.output                   = output;
    return match;
}

static bool match_rope_concat_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const RopeConcatMatch match = match_rope_concat_f32(context.graph, context.root_node);
    if (!match.matched() ||
        !append_covered_node_index_once(context.graph, context.covered_nodes, context.root_node,
                                        dispatch_match.covered_nodes) ||
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.concat,
                                        dispatch_match.covered_nodes)) {
        return false;
    }

    dispatch_match.dispatches.push_back(make_rope_concat_dispatch(context, match, dispatch_match));
    return true;
}

static bool find_rope_set_rows_consumer(const Graph & graph, const GraphNode & rope, RopeSetRowsMatch & match) {
    if (!graph.has_index() || !graph.index().has_single_consumer(rope.output)) {
        return false;
    }

    const GraphNode * consumer = graph.index().consumers(rope.output).front();
    if (consumer == nullptr) {
        return false;
    }

    if (consumer->op == GGML_OP_SET_ROWS) {
        match.set_rows_idx = 0;
        match.layout       = nullptr;
        return graph.index().node_index(consumer, match.set_rows_idx) &&
               (match.set_rows = match_set_rows_2d(graph, consumer)).matched();
    }

    if (!is_layout_alias_node(graph, *consumer) || !graph.index().has_single_consumer(consumer->output)) {
        return false;
    }

    const GraphNode * set_rows = graph.index().consumers(consumer->output).front();
    if (set_rows == nullptr || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    match.layout = consumer;
    return graph.index().node_index(consumer, match.layout_idx) &&
           graph.index().node_index(set_rows, match.set_rows_idx) &&
           (match.set_rows = match_set_rows_2d(graph, set_rows)).matched();
}

static RopeSetRowsMatch match_rope_set_rows_f32(const Graph & graph, const GraphNode * node) {
    RopeSetRowsMatch match;
    match.rope = match_rope_f32(graph, node);
    if (!match.rope.matched() || !find_rope_set_rows_consumer(graph, *node, match)) {
        return {};
    }

    match.published_rows = match.set_rows.rows;
    if (match.set_rows.row_format != 32 || match.set_rows.hidden_size != match.rope.head_size * match.rope.head_count ||
        match.set_rows.token_count != match.rope.token_count || match.published_rows->type != GGML_TYPE_F32) {
        return {};
    }
    return match;
}

static RopeScaleMatch match_rope_scale_f32(const Graph & graph, const GraphNode * node) {
    RopeScaleMatch match;
    match.rope = match_rope_f32(graph, node);
    if (!match.rope.matched() || !graph.has_index() || !graph.index().has_single_consumer(node->output)) {
        return {};
    }

    const GraphNode * scale_node = graph.index().consumers(node->output).front();
    if (scale_node == nullptr) {
        return {};
    }
    const Value * output = graph_value(graph, scale_node->output);
    if (output == nullptr || output->type != GGML_TYPE_F32 || !output->contiguous || output->alias_source.value >= 0 ||
        !same_shape(*match.rope.output, *output)) {
        return {};
    }

    if (scale_node->op == GGML_OP_SCALE && scale_node->inputs.size() == 1 &&
        scale_node->inputs.front() == node->output) {
        const ScaleParams * params = op_params_as<ScaleParams>(scale_node->params);
        if (params == nullptr || params->bias != 0.0f || !std::isfinite(params->scale)) {
            return {};
        }
        match.scalar = params->scale;
    } else if (scale_node->op == GGML_OP_MUL && scale_node->inputs.size() == 2) {
        const BinaryParams * params = op_params_as<BinaryParams>(scale_node->params);
        if (params == nullptr || params->op != BinaryKind::Mul) {
            return {};
        }
        ValueId scale_id;
        if (scale_node->inputs[0] == node->output) {
            scale_id = scale_node->inputs[1];
        } else if (scale_node->inputs[1] == node->output) {
            scale_id = scale_node->inputs[0];
        } else {
            return {};
        }
        const Value * scales = graph_value(graph, scale_id);
        if (scales == nullptr || scales->type != GGML_TYPE_F32 || !scales->contiguous || scales->ne[0] != 1 ||
            scales->ne[1] != 1 || scales->ne[2] != match.rope.token_count || scales->ne[3] != 1) {
            return {};
        }
        match.scales    = scales;
        match.per_token = true;
    } else {
        return {};
    }

    match.scale_node = scale_node;
    match.output     = output;
    return match;
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

static bool match_rope_scale_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const RopeScaleMatch match = match_rope_scale_f32(context.graph, context.root_node);
    if (!match.matched() ||
        (match.per_token && !value_is_available(context.graph, match.scales->id, context.covered_nodes)) ||
        !append_covered_node_index_once(context.graph, context.covered_nodes, context.root_node,
                                        dispatch_match.covered_nodes) ||
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.scale_node,
                                        dispatch_match.covered_nodes)) {
        return false;
    }

    dispatch_match.dispatches.push_back(make_rope_scale_dispatch(context, match, dispatch_match));
    return true;
}

static bool match_rope_set_rows_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const RopeSetRowsMatch match = match_rope_set_rows_f32(context.graph, context.root_node);
    if (!match.matched()) {
        return false;
    }

    if (!append_covered_node_index_once(context.graph, context.covered_nodes, context.root_node,
                                        dispatch_match.covered_nodes) ||
        (match.layout != nullptr && !append_covered_node_index_once(context.graph, context.covered_nodes, match.layout,
                                                                    dispatch_match.covered_nodes)) ||
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.set_rows.node,
                                        dispatch_match.covered_nodes)) {
        return false;
    }

    dispatch_match.dispatches.push_back(make_rope_set_rows_dispatch(context, match, dispatch_match));
    return true;
}

static bool match_rope_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const RopeMatch match = match_rope_f32(context.graph, context.root_node, &dispatch_match.status);
    if (!match.matched()) {
        return false;
    }

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(make_rope_dispatch(context, match, dispatch_match));
    return true;
}

static bool match_set_rows_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const SetRowsMatch match = match_set_rows_2d(context.graph, context.root_node, &dispatch_match.status);
    if (!match.matched()) {
        return false;
    }

    dispatch_match.covered_nodes.push_back(context.root_index);
    dispatch_match.dispatches.push_back(make_set_rows_dispatch(match));
    return true;
}

}  // namespace

void register_rope_set_rows_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.rope_scale.f32",
        GGML_OP_ROPE,
        DispatchMatchKind::Fused,
        300,
        DispatchSource::Common,
        match_rope_scale_f32_dispatch,
    });
    registry.add({
        "common.rope_concat.f32",
        GGML_OP_ROPE,
        DispatchMatchKind::Fused,
        300,
        DispatchSource::Common,
        match_rope_concat_f32_dispatch,
    });
    registry.add({
        "common.rope_set_rows.f32",
        GGML_OP_ROPE,
        DispatchMatchKind::Fused,
        200,
        DispatchSource::Common,
        match_rope_set_rows_f32_dispatch,
    });
    registry.add({
        "common.rope.f32",
        GGML_OP_ROPE,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Common,
        match_rope_f32_dispatch,
    });
    registry.add({
        "common.set_rows",
        GGML_OP_SET_ROWS,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Common,
        match_set_rows_dispatch,
    });
}

}  // namespace ggml::hrx
