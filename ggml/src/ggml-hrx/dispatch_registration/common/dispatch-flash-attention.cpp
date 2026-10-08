#include "dispatch-flash-attention.h"

#include "dispatch-activation-publication.h"
#include "dispatch-mul-mat-common.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kFlashAttentionF32F16WmmaKernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_flash_attention_f32_f16_wmma");
static constexpr KernelCatalogRef kFlashAttentionF32F16WmmaPublishF16Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_flash_attention_f32_f16_wmma_publish_f16");
static constexpr KernelCatalogRef kFlashAttentionDecodeSplitNextQ8Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_flash_attention_decode_split_f32_f16_wmma_next_q8");
static constexpr KernelCatalogRef kCopyTransposeF16Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_copy_transpose_f16");
static constexpr int64_t kDecodeRowCapacity         = 16;
static constexpr int64_t kDecodeKvTileSize          = 64;
static constexpr int64_t kPrefillQkHeadSizeBlock    = 16;
static constexpr int64_t kPrefillValueHeadSizeBlock = 64;
static constexpr int64_t kPrefillMinQkHeadSize      = kPrefillQkHeadSizeBlock;
static constexpr int64_t kPrefillMinValueHeadSize   = kPrefillValueHeadSizeBlock;
// Current cap keeps full-head Q/K staging within the kernel's fixed LDS budget.
static constexpr int64_t kPrefillMaxHeadSize        = 512;

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool nearly_equal(float lhs, float rhs) {
    return std::fabs(lhs - rhs) <= 1.0e-6f;
}

static bool use_exact_prefill_workload(int64_t query_token_count, int64_t key_value_token_count) {
    // Preserve the measured hot prefill case; other supported totals share dynamic executables.
    return query_token_count == 256 && key_value_token_count == 256;
}

static bool is_supported_token_count(int64_t token_count) {
    return token_count >= 1 && token_count <= 2048;
}

static bool is_supported_key_value_token_count(int64_t token_count) {
    return token_count >= 1 && token_count <= 32768;
}

static bool is_supported_decode_key_value_token_count(int64_t token_count) {
    return token_count >= 1 && token_count <= 2048;
}

static bool is_supported_decode_query_length(int64_t query_length) {
    return query_length >= 1 && query_length < kDecodeRowCapacity;
}

static bool is_supported_head_count(int64_t head_count) {
    return head_count >= 1 && head_count <= 64;
}

static bool is_supported_qk_head_size(int64_t head_size) {
    return head_size >= kPrefillMinQkHeadSize && head_size <= kPrefillMaxHeadSize &&
           head_size % kPrefillQkHeadSizeBlock == 0;
}

static bool is_supported_value_head_size(int64_t head_size) {
    return head_size >= kPrefillMinValueHeadSize && head_size <= kPrefillMaxHeadSize &&
           head_size % kPrefillValueHeadSizeBlock == 0;
}

static bool has_query_layout(const Value & value, int64_t query_head_count, int64_t head_size) {
    const size_t element_size = sizeof(float);
    return value.nb[0] == element_size &&
           value.nb[1] == static_cast<size_t>(query_head_count * head_size) * element_size &&
           (value.ne[2] == 1 || value.nb[2] == static_cast<size_t>(head_size) * element_size);
}

static bool has_key_value_layout(const Value & value, int64_t key_value_head_count, int64_t head_size) {
    const size_t element_size = sizeof(ggml_fp16_t);
    return value.nb[0] == element_size &&
           value.nb[1] == static_cast<size_t>(key_value_head_count * head_size) * element_size &&
           (value.ne[2] == 1 || value.nb[2] == static_cast<size_t>(head_size) * element_size);
}

static bool has_mask_layout(const Value & value, int64_t key_value_token_count) {
    const size_t element_size = sizeof(ggml_fp16_t);
    return value.nb[0] == element_size && value.nb[1] == static_cast<size_t>(key_value_token_count) * element_size;
}

static bool has_output_layout(const Value & value, int64_t query_head_count, int64_t head_size) {
    const size_t element_size = sizeof(float);
    return value.nb[0] == element_size && value.nb[1] == static_cast<size_t>(head_size) * element_size &&
           value.nb[2] == static_cast<size_t>(query_head_count * head_size) * element_size;
}

static bool has_flash_attention_params(const GraphNode & node) {
    const FlashAttnExtParams * params = op_params_as<FlashAttnExtParams>(node.params);
    if (params == nullptr) {
        return false;
    }
    return nearly_equal(params->max_bias, 0.0f) && nearly_equal(params->logit_softcap, 0.0f) &&
           (params->prec == GGML_PREC_DEFAULT || params->prec == GGML_PREC_F32);
}

static size_t attention_mask_byte_count(int64_t query_token_count, int64_t key_value_token_count) {
    if (query_token_count <= 0 || key_value_token_count <= 0) {
        return 0;
    }
    return static_cast<size_t>(query_token_count) * static_cast<size_t>(key_value_token_count) * sizeof(ggml_fp16_t);
}

static std::string to_config_value(int64_t value) {
    return std::to_string(value);
}

static int64_t ceil_div(int64_t value, int64_t divisor) {
    return (value + divisor - 1) / divisor;
}

static int64_t decode_key_value_capacity(int64_t token_count) {
    int64_t capacity = kDecodeKvTileSize;
    while (capacity < token_count) {
        capacity *= 2;
    }
    return capacity;
}

static ValueId match_value(const DispatchMatchContext & context, const DispatchMatch & dispatch_match, int32_t offset) {
    return ValueId(context.next_plan_value.value + static_cast<int32_t>(dispatch_match.transients.size()) +
                   static_cast<int32_t>(dispatch_match.completion_counter_requests.size()) + offset);
}

struct FlashAttentionMatch {
    const Value *     query         = nullptr;
    const Value *     key           = nullptr;
    const Value *     value         = nullptr;
    const Value *     mask          = nullptr;
    const Value *     output        = nullptr;
    const GraphNode * output_layout = nullptr;
    ValueId           mask_binding_value;
    size_t            mask_binding_bytes    = 0;
    int64_t           query_token_count     = 0;
    int64_t           key_value_token_count = 0;
    int64_t           query_head_count      = 0;
    int64_t           key_value_head_count  = 0;
    int64_t           qk_head_size          = 0;
    int64_t           value_head_size       = 0;
    float             attention_scale       = 0.0f;

    bool matched() const {
        return query != nullptr && key != nullptr && value != nullptr && mask != nullptr && output != nullptr;
    }
};

struct DecodeSplitFlashAttentionMatch {
    const Value *     query                 = nullptr;
    const Value *     key                   = nullptr;
    const Value *     value                 = nullptr;
    const Value *     mask                  = nullptr;
    const Value *     output                = nullptr;
    const GraphNode * output_layout         = nullptr;
    int64_t           query_token_count     = 0;
    int64_t           key_value_token_count = 0;
    int64_t           key_value_capacity    = 0;
    int64_t           query_head_count      = 0;
    int64_t           key_value_head_count  = 0;
    int64_t           qk_head_size          = 0;
    int64_t           value_head_size       = 0;
    float             attention_scale       = 0.0f;

    bool matched() const {
        return query != nullptr && key != nullptr && value != nullptr && mask != nullptr && output != nullptr;
    }
};

static FlashAttentionMatch match_flash_attention_f32_f16(const Graph &       graph,
                                                         const CommandPlan & plan,
                                                         const GraphNode *   node) {
    FlashAttentionMatch match;
    if (node == nullptr || node->op != GGML_OP_FLASH_ATTN_EXT || node->inputs.size() != 4) {
        return match;
    }

    const Value * query  = graph_value(graph, node->inputs[0]);
    const Value * key    = graph_value(graph, node->inputs[1]);
    const Value * value  = graph_value(graph, node->inputs[2]);
    const Value * mask   = graph_value(graph, node->inputs[3]);
    const Value * output = graph_value(graph, node->output);
    if (query == nullptr || key == nullptr || value == nullptr || mask == nullptr || output == nullptr) {
        return {};
    }
    if (query->type != GGML_TYPE_F32 || key->type != GGML_TYPE_F16 || value->type != GGML_TYPE_F16 ||
        mask->type != GGML_TYPE_F16 || output->type != GGML_TYPE_F32) {
        return {};
    }
    const int64_t qk_head_size    = query->ne[0];
    const int64_t value_head_size = value->ne[0];
    if (!is_supported_qk_head_size(qk_head_size) || !is_supported_value_head_size(value_head_size) ||
        !has_flash_attention_params(*node)) {
        return {};
    }
    if (key->ne[0] != qk_head_size || output->ne[0] != value_head_size) {
        return {};
    }
    if (query->ne[3] != 1 || key->ne[3] != 1 || value->ne[3] != 1 || output->ne[3] != 1 || mask->ne[2] != 1 ||
        mask->ne[3] != 1) {
        return {};
    }

    const int64_t query_token_count    = query->ne[1];
    const int64_t query_head_count     = query->ne[2];
    const int64_t key_value_capacity   = key->ne[1];
    const int64_t key_value_head_count = key->ne[2];
    if (!is_supported_token_count(query_token_count) || key_value_capacity < query_token_count ||
        !is_supported_head_count(query_head_count) || !is_supported_head_count(key_value_head_count) ||
        query_head_count % key_value_head_count != 0) {
        return {};
    }
    if (value->ne[1] != key_value_capacity || value->ne[2] != key_value_head_count ||
        mask->ne[0] > key_value_capacity || mask->ne[1] != query_token_count || output->ne[1] != query_head_count ||
        output->ne[2] != query_token_count) {
        return {};
    }

    int64_t      key_value_token_count     = key_value_capacity;
    ValueId      mask_binding_value        = mask->id;
    size_t       mask_binding_bytes        = mask->byte_count;
    bool         mask_binding_is_alternate = false;
    const size_t compact_mask_bytes        = attention_mask_byte_count(query_token_count, query_token_count);
    const auto * compact_mask              = find_alternate_value(plan, mask->id, GGML_TYPE_F16, compact_mask_bytes);
    const bool   mask_is_compact           = mask->ne[0] == query_token_count;
    const bool   mask_is_capacity          = mask->ne[0] == key_value_capacity;
    if (compact_mask != nullptr && mask_is_capacity && mask->ne[0] > query_token_count) {
        key_value_token_count     = query_token_count;
        mask_binding_value        = compact_mask->alternate_value;
        mask_binding_bytes        = compact_mask->byte_count;
        mask_binding_is_alternate = true;
    } else if (!mask_is_compact && !mask_is_capacity) {
        return {};
    }
    if (!is_supported_key_value_token_count(key_value_token_count)) {
        return {};
    }

    if (!has_query_layout(*query, query_head_count, qk_head_size) ||
        !has_key_value_layout(*key, key_value_head_count, qk_head_size) ||
        !has_key_value_layout(*value, key_value_head_count, value_head_size) ||
        !has_output_layout(*output, query_head_count, value_head_size)) {
        return {};
    }
    if (!mask_binding_is_alternate && !has_mask_layout(*mask, key_value_token_count)) {
        return {};
    }

    match.query                 = query;
    match.key                   = key;
    match.value                 = value;
    match.mask                  = mask;
    match.output                = output;
    match.output_layout         = find_single_layout_alias_consumer(graph, output->id);
    match.mask_binding_value    = mask_binding_value;
    match.mask_binding_bytes    = mask_binding_bytes;
    match.query_token_count     = query_token_count;
    match.key_value_token_count = key_value_token_count;
    match.query_head_count      = query_head_count;
    match.key_value_head_count  = key_value_head_count;
    match.qk_head_size          = qk_head_size;
    match.value_head_size       = value_head_size;
    match.attention_scale       = op_params_as<FlashAttnExtParams>(node->params)->scale;
    return match;
}

static DecodeSplitFlashAttentionMatch match_decode_split_flash_attention_f32_f16(const Graph &     graph,
                                                                                 const GraphNode * node) {
    DecodeSplitFlashAttentionMatch match;
    if (node == nullptr || node->op != GGML_OP_FLASH_ATTN_EXT || node->inputs.size() != 4) {
        return match;
    }

    const Value * query  = graph_value(graph, node->inputs[0]);
    const Value * key    = graph_value(graph, node->inputs[1]);
    const Value * value  = graph_value(graph, node->inputs[2]);
    const Value * mask   = graph_value(graph, node->inputs[3]);
    const Value * output = graph_value(graph, node->output);
    if (query == nullptr || key == nullptr || value == nullptr || mask == nullptr || output == nullptr) {
        return {};
    }
    if (query->type != GGML_TYPE_F32 || key->type != GGML_TYPE_F16 || value->type != GGML_TYPE_F16 ||
        mask->type != GGML_TYPE_F16 || output->type != GGML_TYPE_F32) {
        return {};
    }

    const int64_t qk_head_size    = query->ne[0];
    const int64_t value_head_size = value->ne[0];
    if (!is_supported_qk_head_size(qk_head_size) || !is_supported_value_head_size(value_head_size) ||
        !has_flash_attention_params(*node)) {
        return {};
    }
    if (key->ne[0] != qk_head_size || output->ne[0] != value_head_size) {
        return {};
    }
    if (query->ne[3] != 1 || key->ne[3] != 1 || value->ne[3] != 1 || output->ne[3] != 1 || mask->ne[2] != 1 ||
        mask->ne[3] != 1) {
        return {};
    }

    const int64_t query_token_count     = query->ne[1];
    const int64_t query_head_count      = query->ne[2];
    const int64_t key_value_capacity    = key->ne[1];
    const int64_t key_value_head_count  = key->ne[2];
    const int64_t key_value_token_count = mask->ne[0];
    if (!is_supported_decode_query_length(query_token_count) ||
        !is_supported_decode_key_value_token_count(key_value_token_count) ||
        key_value_capacity < key_value_token_count || !is_supported_head_count(query_head_count) ||
        !is_supported_head_count(key_value_head_count) || query_head_count % key_value_head_count != 0) {
        return {};
    }
    if (value->ne[1] != key_value_capacity || value->ne[2] != key_value_head_count ||
        mask->ne[1] != query_token_count || output->ne[1] != query_head_count || output->ne[2] != query_token_count) {
        return {};
    }
    if (!has_query_layout(*query, query_head_count, qk_head_size) ||
        !has_key_value_layout(*key, key_value_head_count, qk_head_size) ||
        !has_key_value_layout(*value, key_value_head_count, value_head_size) ||
        !has_mask_layout(*mask, key_value_token_count) ||
        !has_output_layout(*output, query_head_count, value_head_size)) {
        return {};
    }

    match.query                 = query;
    match.key                   = key;
    match.value                 = value;
    match.mask                  = mask;
    match.output                = output;
    match.output_layout         = find_single_layout_alias_consumer(graph, output->id);
    match.query_token_count     = query_token_count;
    match.key_value_token_count = key_value_token_count;
    match.key_value_capacity    = decode_key_value_capacity(key_value_token_count);
    match.query_head_count      = query_head_count;
    match.key_value_head_count  = key_value_head_count;
    match.qk_head_size          = qk_head_size;
    match.value_head_size       = value_head_size;
    match.attention_scale       = op_params_as<FlashAttnExtParams>(node->params)->scale;
    return match;
}

static std::string to_config_value(float value) {
    std::ostringstream out;
    out.precision(9);
    out << value;
    return out.str();
}

static void add_flash_attention_decode_compile_parameters(KernelSpecialization & kernel,
                                                          int64_t                query_head_count,
                                                          int64_t                key_value_head_count,
                                                          int64_t                qk_head_size,
                                                          int64_t                value_head_size,
                                                          float                  attention_scale) {
    kernel.compile_parameters.emplace("ggml.flash_attention.query_head_count", to_config_value(query_head_count));
    kernel.compile_parameters.emplace("ggml.flash_attention.key_value_head_count",
                                      to_config_value(key_value_head_count));
    kernel.compile_parameters.emplace("ggml.flash_attention.qk_head_size", to_config_value(qk_head_size));
    kernel.compile_parameters.emplace("ggml.flash_attention.value_head_size", to_config_value(value_head_size));
    kernel.compile_parameters.emplace("ggml.flash_attention.attention_scale", to_config_value(attention_scale));
}

static void add_flash_attention_compile_parameters(KernelSpecialization & kernel,
                                                   int64_t                query_head_count,
                                                   int64_t                key_value_head_count,
                                                   int64_t                qk_head_size,
                                                   int64_t                value_head_size,
                                                   float                  attention_scale,
                                                   bool                   apply_gate,
                                                   int64_t                gate_stride_head,
                                                   int64_t                gate_stride_token) {
    kernel.compile_parameters.emplace("ggml.flash_attention.query_head_count", to_config_value(query_head_count));
    kernel.compile_parameters.emplace("ggml.flash_attention.key_value_head_count",
                                      to_config_value(key_value_head_count));
    kernel.compile_parameters.emplace("ggml.flash_attention.qk_head_size", to_config_value(qk_head_size));
    kernel.compile_parameters.emplace("ggml.flash_attention.value_head_size", to_config_value(value_head_size));
    kernel.compile_parameters.emplace("ggml.flash_attention.attention_scale", to_config_value(attention_scale));
    kernel.compile_parameters.emplace("ggml.flash_attention.apply_gate", apply_gate ? "1" : "0");
    kernel.compile_parameters.emplace("ggml.flash_attention.gate_stride_head", to_config_value(gate_stride_head));
    kernel.compile_parameters.emplace("ggml.flash_attention.gate_stride_token", to_config_value(gate_stride_token));
}

static const GraphNode * single_consumer_with_op(const Graph & graph, ValueId value, ggml_op op) {
    const std::vector<const GraphNode *> & consumers = graph.index().consumers(value);
    return consumers.size() == 1 && consumers.front() != nullptr && consumers.front()->op == op ? consumers.front() :
                                                                                                  nullptr;
}

static DispatchBinding prepare_flash_attention_value(const DispatchMatchContext & context,
                                                     const FlashAttentionMatch & match,
                                                     DispatchMatch & dispatch_match,
                                                     KernelSpecialization & attention) {
    if (match.query_token_count < 256 || match.key_value_token_count < 512 ||
        match.key_value_token_count % 32 != 0) {
        return { match.value->id, 0, match.value->byte_count };
    }

    const int64_t columns = match.key_value_head_count * match.value_head_size;
    const size_t bytes = static_cast<size_t>(match.key_value_token_count * columns) * sizeof(ggml_fp16_t);
    const ValueId transposed = match_value(context, dispatch_match, 0);
    dispatch_match.transients.push_back({ transposed, "common.flash_attention.transposed_value", bytes, 256 });

    Dispatch copy;
    copy.kernel = make_kernel_specialization(kCopyTransposeF16Kernel);
    copy.kernel.workload_specialization = WorkloadSpecialization::Dynamic;
    copy.kernel.integer_parameters.emplace("row_count", match.key_value_token_count);
    copy.kernel.integer_parameters.emplace("column_count", columns);
    copy.bindings.push_back({ match.value->id, 0, bytes });
    copy.bindings.push_back({ transposed, 0, bytes });
    dispatch_match.dispatches.push_back(std::move(copy));
    attention.compile_parameters.emplace("ggml.flash_attention.value_layout", "1");
    return { transposed, 0, bytes };
}

static bool match_flash_attention_gate_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const FlashAttentionMatch match = match_flash_attention_f32_f16(context.graph, context.plan, context.root_node);
    if (!match.matched() || match.output_layout == nullptr || match.output_layout->op != GGML_OP_RESHAPE) {
        return false;
    }

    const GraphNode * reshape  = match.output_layout;
    const Value *     reshaped = graph_value(context.graph, reshape->output);
    const GraphNode * mul =
        reshaped != nullptr ? single_consumer_with_op(context.graph, reshaped->id, GGML_OP_MUL) : nullptr;
    if (reshape->inputs.size() != 1 || reshaped == nullptr || mul == nullptr || mul->inputs.size() != 2 ||
        reshaped->type != GGML_TYPE_F32 || !reshaped->contiguous ||
        reshaped->ne[0] != match.value_head_size * match.query_head_count ||
        reshaped->ne[1] != match.query_token_count || reshaped->ne[2] != 1 || reshaped->ne[3] != 1) {
        return false;
    }

    ValueId gate_value_id;
    if (mul->inputs[0] == reshape->output) {
        gate_value_id = mul->inputs[1];
    } else if (mul->inputs[1] == reshape->output) {
        gate_value_id = mul->inputs[0];
    } else {
        return false;
    }

    const GraphNode *   sigmoid        = context.graph.index().producer(gate_value_id);
    const UnaryParams * sigmoid_params = sigmoid != nullptr ? op_params_as<UnaryParams>(sigmoid->params) : nullptr;
    if (sigmoid == nullptr || sigmoid->op != GGML_OP_UNARY || sigmoid->inputs.size() != 1 ||
        sigmoid_params == nullptr || sigmoid_params->op != UnaryKind::Sigmoid ||
        single_consumer_with_op(context.graph, sigmoid->output, GGML_OP_MUL) != mul) {
        return false;
    }

    const GraphNode * cont = context.graph.index().producer(sigmoid->inputs[0]);
    if (cont == nullptr || cont->op != GGML_OP_CONT || cont->inputs.size() != 1 ||
        single_consumer_with_op(context.graph, cont->output, GGML_OP_UNARY) != sigmoid) {
        return false;
    }

    const GraphNode * gate_view = context.graph.index().producer(cont->inputs[0]);
    const Value *     raw_gate  = gate_view != nullptr ? graph_value(context.graph, gate_view->output) : nullptr;
    const Value *     output    = graph_value(context.graph, mul->output);
    if (gate_view == nullptr || gate_view->op != GGML_OP_VIEW || gate_view->inputs.size() != 1 || raw_gate == nullptr ||
        output == nullptr || raw_gate->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
        raw_gate->ne[0] != match.value_head_size || raw_gate->ne[1] != match.query_head_count ||
        raw_gate->ne[2] != match.query_token_count || raw_gate->ne[3] != 1 || raw_gate->nb[0] != sizeof(float) ||
        output->ne != reshaped->ne || !output->contiguous ||
        single_consumer_with_op(context.graph, gate_view->output, GGML_OP_CONT) != cont) {
        return false;
    }
    if (output->storage_root == match.query->storage_root || output->storage_root == match.key->storage_root ||
        output->storage_root == match.value->storage_root || output->storage_root == match.mask->storage_root ||
        output->storage_root == raw_gate->storage_root) {
        return false;
    }

    dispatch_match.covered_nodes.push_back(context.root_index);
    for (const GraphNode * covered : { reshape, gate_view, cont, sigmoid, mul }) {
        if (!append_covered_node_index_once(context.graph, context.covered_nodes, covered,
                                            dispatch_match.covered_nodes)) {
            return false;
        }
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kFlashAttentionF32F16WmmaKernel);
    if (!use_exact_prefill_workload(match.query_token_count, match.key_value_token_count)) {
        dispatch.kernel.workload_specialization = WorkloadSpecialization::Dynamic;
    }
    dispatch.kernel.integer_parameters.emplace("query_token_count", match.query_token_count);
    dispatch.kernel.integer_parameters.emplace("key_value_token_count", match.key_value_token_count);
    add_flash_attention_compile_parameters(dispatch.kernel, match.query_head_count, match.key_value_head_count,
                                           match.qk_head_size, match.value_head_size, match.attention_scale, true,
                                           static_cast<int64_t>(raw_gate->nb[1] / sizeof(float)),
                                           static_cast<int64_t>(raw_gate->nb[2] / sizeof(float)));
    dispatch.bindings.push_back({ match.query->id, 0, match.query->byte_count });
    dispatch.bindings.push_back({ match.key->id, 0, match.key->byte_count });
    dispatch.bindings.push_back(prepare_flash_attention_value(context, match, dispatch_match, dispatch.kernel));
    dispatch.bindings.push_back({ match.mask_binding_value, 0, match.mask_binding_bytes });
    dispatch.bindings.push_back({ raw_gate->id, 0, raw_gate->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_flash_attention_f32_f16_dispatch(const DispatchMatchContext & context,
                                                   DispatchMatch &              dispatch_match) {
    const FlashAttentionMatch match = match_flash_attention_f32_f16(context.graph, context.plan, context.root_node);
    if (!match.matched()) {
        return false;
    }

    const CommonActivationPublicationCapabilities capabilities = {
        match.query_token_count >= 128 ? DispatchActivationInputF16Row : DispatchActivationInputNone,
        DispatchActivationInputF16Row,
    };
    const CommonActivationPublicationPlan publication_plan =
        common_select_activation_publication_plan(context, *match.output, capabilities);
    const CommonActivationPublicationDemand * demand =
        publication_plan.find(CommonActivationPublicationFormat::F16Row);
    const bool publish_f16 = demand != nullptr;
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(publish_f16 ? kFlashAttentionF32F16WmmaPublishF16Kernel :
                                                               kFlashAttentionF32F16WmmaKernel);
    if (!use_exact_prefill_workload(match.query_token_count, match.key_value_token_count)) {
        dispatch.kernel.workload_specialization = WorkloadSpecialization::Dynamic;
    }
    dispatch.kernel.integer_parameters.emplace("query_token_count", match.query_token_count);
    dispatch.kernel.integer_parameters.emplace("key_value_token_count", match.key_value_token_count);
    add_flash_attention_compile_parameters(dispatch.kernel, match.query_head_count, match.key_value_head_count,
                                           match.qk_head_size, match.value_head_size, match.attention_scale, false, 1,
                                           1);
    dispatch.bindings.push_back({ match.query->id, 0, match.query->byte_count });
    dispatch.bindings.push_back({ match.key->id, 0, match.key->byte_count });
    dispatch.bindings.push_back(prepare_flash_attention_value(context, match, dispatch_match, dispatch.kernel));
    dispatch.bindings.push_back({ match.mask_binding_value, 0, match.mask_binding_bytes });
    dispatch.bindings.push_back({ match.query->id, 0, match.query->byte_count });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

    if (publish_f16) {
        constexpr const char * name = "common.flash_attention.f16";
        CommonActivationPublication publication;
        if (!common_reserve_activation_publication(context, dispatch_match, *match.output, *demand, name, publication) ||
            !common_append_activation_publication(dispatch_match, publication, name)) {
            return false;
        }
        dispatch.bindings.push_back(publication.binding());
    }

    dispatch_match.covered_nodes.push_back(context.root_index);
    if (match.output_layout != nullptr) {
        if (!append_covered_node_index_once(context.graph, context.covered_nodes, match.output_layout,
                                            dispatch_match.covered_nodes)) {
            return false;
        }
    }
    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_flash_attention_decode_split_next_q8_dispatch(const DispatchMatchContext & context,
                                                                DispatchMatch &              dispatch_match) {
    const DecodeSplitFlashAttentionMatch match =
        match_decode_split_flash_attention_f32_f16(context.graph, context.root_node);
    if (!match.matched()) {
        return false;
    }

    const Value * publication_subject =
        match.output_layout != nullptr ? graph_value(context.graph, match.output_layout->output) : nullptr;
    if (match.output_layout != nullptr &&
        (match.output_layout->op != GGML_OP_RESHAPE || publication_subject == nullptr ||
         !same_full_ordered_value_range(*match.output, *publication_subject))) {
        return false;
    }
    const CommonActivationPublicationCapabilities capabilities = {
        DispatchActivationInputQ8_1X4,
    };
    const CommonActivationPublicationPlan publication_plan = publication_subject != nullptr ?
        common_select_activation_publication_plan(context, *publication_subject, capabilities) :
        CommonActivationPublicationPlan{};
    const CommonActivationPublicationDemand * demand =
        publication_plan.find(CommonActivationPublicationFormat::Q8_1X4);

    const int64_t key_value_block_count = ceil_div(match.key_value_capacity, kDecodeKvTileSize);
    const size_t  partial_scalar_count  = static_cast<size_t>(match.key_value_head_count) *
                                        static_cast<size_t>(key_value_block_count) *
                                        static_cast<size_t>(kDecodeRowCapacity);
    const size_t  partial_value_count  = partial_scalar_count * static_cast<size_t>(match.value_head_size);
    const size_t  partial_scalar_bytes = partial_scalar_count * sizeof(float);
    const size_t  partial_output_bytes = partial_value_count * sizeof(ggml_fp16_t);
    const int64_t query_hidden_size    = match.query_head_count * match.qk_head_size;
    const int64_t output_hidden_size   = match.query_head_count * match.value_head_size;
    if (partial_scalar_bytes == 0 || partial_output_bytes == 0) {
        return false;
    }

    const ValueId partial_max        = match_value(context, dispatch_match, 0);
    const ValueId partial_sum        = match_value(context, dispatch_match, 1);
    const ValueId partial_output     = match_value(context, dispatch_match, 2);
    const ValueId completion_counter = match_value(context, dispatch_match, 3);

    dispatch_match.transients.push_back(
        { partial_max, "common.decode.flash_attention.partial_max", partial_scalar_bytes, 256 });
    dispatch_match.transients.push_back(
        { partial_sum, "common.decode.flash_attention.partial_sum", partial_scalar_bytes, 256 });
    dispatch_match.transients.push_back(
        { partial_output, "common.decode.flash_attention.partial_output", partial_output_bytes, 256 });
    dispatch_match.completion_counter_requests.push_back({
        completion_counter,
        "common.decode.flash_attention.completion_counter",
        static_cast<uint32_t>(match.key_value_head_count),
    });

    CommonActivationPublication publication;
    constexpr const char * publication_name = "common.decode.flash_attention.next_q8_output";
    Value private_layout = publication_subject != nullptr ? *publication_subject : *match.output;
    if (publication_subject == nullptr) {
        private_layout.ne = { output_hidden_size * match.query_token_count, 1, 1, 1 };
    }
    const bool reserved = demand != nullptr ?
        common_reserve_activation_publication(context, dispatch_match, *publication_subject, *demand,
                                              publication_name, publication) :
        common_reserve_private_activation_output(context, dispatch_match, private_layout,
                                                 CommonActivationPublicationFormat::Q8_1X4,
                                                 "common.decode.flash_attention.private_q8_1_x4", publication);
    if (!reserved ||
        (demand != nullptr && !common_append_activation_publication(dispatch_match, publication, publication_name))) {
        return false;
    }
    const size_t q8_row_bytes = publication.byte_count / static_cast<size_t>(match.query_token_count);

    const size_t query_row_bytes  = static_cast<size_t>(query_hidden_size) * sizeof(float);
    const size_t mask_row_bytes   = static_cast<size_t>(match.key_value_token_count) * sizeof(ggml_fp16_t);
    const size_t output_row_bytes = static_cast<size_t>(output_hidden_size) * sizeof(float);
    for (int64_t row = 0; row < match.query_token_count; ++row) {
        Dispatch dispatch;
        dispatch.kernel = make_kernel_specialization(kFlashAttentionDecodeSplitNextQ8Kernel);
        dispatch.kernel.workload_specialization = WorkloadSpecialization::Dynamic;
        dispatch.kernel.integer_parameters.emplace("key_value_token_count", match.key_value_token_count);
        add_flash_attention_decode_compile_parameters(dispatch.kernel, match.query_head_count,
                                                      match.key_value_head_count, match.qk_head_size,
                                                      match.value_head_size, match.attention_scale);
        dispatch.kernel.compile_parameters.emplace("ggml.flash_attention.decode.key_value_token_capacity",
                                                   to_config_value(match.key_value_capacity));
        dispatch.bindings.push_back(
            { match.query->id, static_cast<size_t>(row) * match.query->nb[1], query_row_bytes });
        dispatch.bindings.push_back({ match.key->id, 0, match.key->byte_count });
        dispatch.bindings.push_back({ match.value->id, 0, match.value->byte_count });
        dispatch.bindings.push_back({ match.mask->id, static_cast<size_t>(row) * match.mask->nb[1], mask_row_bytes });
        dispatch.bindings.push_back({ partial_max, 0, partial_scalar_bytes });
        dispatch.bindings.push_back({ partial_sum, 0, partial_scalar_bytes });
        dispatch.bindings.push_back({ partial_output, 0, partial_output_bytes });
        dispatch.bindings.push_back(
            { completion_counter, 0, static_cast<size_t>(match.key_value_head_count) * sizeof(int32_t) });
        dispatch.bindings.push_back(
            { match.output->id, static_cast<size_t>(row) * match.output->nb[2], output_row_bytes });
        dispatch.bindings.push_back(publication.binding(static_cast<size_t>(row) * q8_row_bytes, q8_row_bytes));
        dispatch_match.dispatches.push_back(std::move(dispatch));
    }

    dispatch_match.covered_nodes.push_back(context.root_index);
    if (match.output_layout != nullptr) {
        if (!append_covered_node_index_once(context.graph, context.covered_nodes, match.output_layout,
                                            dispatch_match.covered_nodes)) {
            return false;
        }
    }
    return true;
}

}  // namespace

void register_flash_attention_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "common.flash_attention_f32_f16_gate",
        GGML_OP_FLASH_ATTN_EXT,
        DispatchMatchKind::Fused,
        100,
        DispatchSource::Common,
        match_flash_attention_gate_dispatch,
    });
    registry.add({
        "common.flash_attention_decode_split_next_q8",
        GGML_OP_FLASH_ATTN_EXT,
        DispatchMatchKind::SingleOp,
        75,
        DispatchSource::Common,
        match_flash_attention_decode_split_next_q8_dispatch,
    });
    registry.add({
        "common.flash_attention_f32_f16_wmma",
        GGML_OP_FLASH_ATTN_EXT,
        DispatchMatchKind::SingleOp,
        50,
        DispatchSource::Common,
        match_flash_attention_f32_f16_dispatch,
    });
}

}  // namespace ggml::hrx
