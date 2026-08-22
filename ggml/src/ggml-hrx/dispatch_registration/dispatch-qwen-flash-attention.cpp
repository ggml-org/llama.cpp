#include "dispatch-qwen-flash-attention.h"

#include "dispatch-llm-shapes.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kQwenFlashAttentionF32F16WmmaKernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_flash_attention_f32_f16_wmma");
static constexpr KernelCatalogRef kQwenFlashAttentionDecodeSplitNextQ8Kernel =
    GGML_HRX_KERNEL_REF("qwen3_moe", "qwen3_moe_flash_attention_decode_split_f32_f16_wmma_next_q8");
static constexpr int64_t kQwenAttentionHeadSize = 128;
static constexpr int64_t kQwenQueryHeadCount    = 32;
static constexpr int64_t kQwenKeyValueHeadCount = 4;
static constexpr int64_t kQwenDecodeRowCapacity = 16;
static constexpr int64_t kQwenDecodeKvTileSize  = 64;

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool nearly_equal(float lhs, float rhs) {
    return std::fabs(lhs - rhs) <= 1.0e-6f;
}

static bool is_supported_key_value_token_count(int64_t token_count) {
    return token_count >= 1 && token_count <= 32768;
}

static bool is_supported_decode_key_value_token_count(int64_t token_count) {
    return token_count >= 1 && token_count <= 2048;
}

static bool is_qwen_flash_decode_query_length(int64_t query_length) {
    return query_length >= 1 && query_length < kQwenDecodeRowCapacity;
}

static bool is_supported_head_count(int64_t head_count) {
    return head_count >= 1 && head_count <= 64;
}

static bool has_query_layout(const Value & value, int64_t query_head_count) {
    const size_t element_size = sizeof(float);
    return value.nb[0] == element_size &&
           value.nb[1] == static_cast<size_t>(query_head_count * kQwenAttentionHeadSize) * element_size &&
           (value.ne[2] == 1 || value.nb[2] == static_cast<size_t>(kQwenAttentionHeadSize) * element_size);
}

static bool has_key_value_layout(const Value & value, int64_t key_value_head_count) {
    const size_t element_size = sizeof(ggml_fp16_t);
    return value.nb[0] == element_size &&
           value.nb[1] == static_cast<size_t>(key_value_head_count * kQwenAttentionHeadSize) * element_size &&
           (value.ne[2] == 1 || value.nb[2] == static_cast<size_t>(kQwenAttentionHeadSize) * element_size);
}

static bool has_mask_layout(const Value & value, int64_t key_value_token_count) {
    const size_t element_size = sizeof(ggml_fp16_t);
    return value.nb[0] == element_size && value.nb[1] == static_cast<size_t>(key_value_token_count) * element_size;
}

static bool has_output_layout(const Value & value, int64_t query_head_count) {
    const size_t element_size = sizeof(float);
    return value.nb[0] == element_size && value.nb[1] == static_cast<size_t>(kQwenAttentionHeadSize) * element_size &&
           value.nb[2] == static_cast<size_t>(query_head_count * kQwenAttentionHeadSize) * element_size;
}

static std::string to_config_value(int64_t value) {
    return std::to_string(value);
}

static size_t attention_mask_byte_count(int64_t query_token_count, int64_t key_value_token_count) {
    if (query_token_count <= 0 || key_value_token_count <= 0) {
        return 0;
    }
    return static_cast<size_t>(query_token_count) * static_cast<size_t>(key_value_token_count) * sizeof(ggml_fp16_t);
}

static size_t q8_1_x4_byte_count(int64_t row_count, int64_t hidden_size) {
    if (row_count <= 0 || hidden_size <= 0) {
        return 0;
    }
    return static_cast<size_t>(row_count) * ggml_row_size(GGML_TYPE_Q8_1, hidden_size);
}

static int64_t ceil_div(int64_t value, int64_t divisor) {
    return (value + divisor - 1) / divisor;
}

static ValueId match_value(const DispatchMatchContext & context, const DispatchMatch & dispatch_match, int32_t offset) {
    return ValueId(context.next_plan_value.value + static_cast<int32_t>(dispatch_match.transients.size()) +
                   static_cast<int32_t>(dispatch_match.completion_counter_requests.size()) + offset);
}

struct QwenFlashAttentionMatch {
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

    bool matched() const {
        return query != nullptr && key != nullptr && value != nullptr && mask != nullptr && output != nullptr;
    }
};

struct QwenDecodeSplitFlashAttentionMatch {
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

    bool matched() const {
        return query != nullptr && key != nullptr && value != nullptr && mask != nullptr && output != nullptr;
    }
};

static bool has_qwen_flash_attention_params(const GraphNode & node) {
    const FlashAttnExtParams * params = op_params_as<FlashAttnExtParams>(node.params);
    if (params == nullptr) {
        return false;
    }
    const float expected_scale = 1.0f / std::sqrt(static_cast<float>(kQwenAttentionHeadSize));
    return nearly_equal(params->scale, expected_scale) && nearly_equal(params->max_bias, 0.0f) &&
           nearly_equal(params->logit_softcap, 0.0f) &&
           (params->prec == GGML_PREC_DEFAULT || params->prec == GGML_PREC_F32);
}

static QwenFlashAttentionMatch match_qwen_flash_attention(const Graph &       graph,
                                                          const CommandPlan & plan,
                                                          const GraphNode *   node) {
    QwenFlashAttentionMatch match;
    if (node == nullptr || node->op != GGML_OP_FLASH_ATTN_EXT || node->inputs.size() != 4 ||
        !has_qwen_flash_attention_params(*node)) {
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
    if (query->ne[0] != kQwenAttentionHeadSize || key->ne[0] != kQwenAttentionHeadSize ||
        value->ne[0] != kQwenAttentionHeadSize || output->ne[0] != kQwenAttentionHeadSize) {
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
    if (!is_qwen_prefill_query_length(query_token_count) || key_value_capacity < query_token_count ||
        !is_supported_head_count(query_head_count) || !is_supported_head_count(key_value_head_count) ||
        query_head_count % key_value_head_count != 0) {
        return {};
    }
    if (value->ne[1] != key_value_capacity || value->ne[2] != key_value_head_count ||
        mask->ne[0] > key_value_capacity || mask->ne[1] != query_token_count || output->ne[1] != query_head_count ||
        output->ne[2] != query_token_count) {
        return {};
    }

    int64_t      key_value_token_count = key_value_capacity;
    ValueId      mask_binding_value    = mask->id;
    size_t       mask_binding_bytes    = mask->byte_count;
    const size_t compact_mask_bytes    = attention_mask_byte_count(query_token_count, query_token_count);
    const auto * compact_mask          = find_alternate_value(plan, mask->id, GGML_TYPE_F16, compact_mask_bytes);
    const bool   mask_is_compact       = mask->ne[0] == query_token_count;
    const bool   mask_is_capacity      = mask->ne[0] == key_value_capacity;
    if (compact_mask != nullptr && mask_is_capacity && mask->ne[0] > query_token_count) {
        key_value_token_count = query_token_count;
        mask_binding_value    = compact_mask->alternate_value;
        mask_binding_bytes    = compact_mask->byte_count;
    } else if (!mask_is_compact && !mask_is_capacity) {
        return {};
    }
    if (!is_supported_key_value_token_count(key_value_token_count)) {
        return {};
    }

    if (!has_query_layout(*query, query_head_count) || !has_key_value_layout(*key, key_value_head_count) ||
        !has_key_value_layout(*value, key_value_head_count) || !has_output_layout(*output, query_head_count)) {
        return {};
    }
    if (mask_binding_value == mask->id && !has_mask_layout(*mask, key_value_token_count)) {
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
    return match;
}

static QwenDecodeSplitFlashAttentionMatch match_qwen_decode_split_flash_attention(const Graph &     graph,
                                                                                  const GraphNode * node) {
    QwenDecodeSplitFlashAttentionMatch match;
    if (node == nullptr || node->op != GGML_OP_FLASH_ATTN_EXT || node->inputs.size() != 4 ||
        !has_qwen_flash_attention_params(*node)) {
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
    if (query->ne[0] != kQwenAttentionHeadSize || key->ne[0] != kQwenAttentionHeadSize ||
        value->ne[0] != kQwenAttentionHeadSize || output->ne[0] != kQwenAttentionHeadSize) {
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
    if (!is_qwen_flash_decode_query_length(query_token_count) ||
        !is_supported_decode_key_value_token_count(key_value_token_count) || query_head_count != kQwenQueryHeadCount ||
        key_value_head_count != kQwenKeyValueHeadCount || key_value_capacity < key_value_token_count ||
        value->ne[1] != key_value_capacity || value->ne[2] != key_value_head_count ||
        mask->ne[1] != query_token_count || output->ne[1] != query_head_count || output->ne[2] != query_token_count) {
        return {};
    }
    if (!has_query_layout(*query, query_head_count) || !has_key_value_layout(*key, key_value_head_count) ||
        !has_key_value_layout(*value, key_value_head_count) || !has_mask_layout(*mask, key_value_token_count) ||
        !has_output_layout(*output, query_head_count)) {
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
    match.key_value_capacity    = ceil_div(key_value_token_count, kQwenDecodeKvTileSize) * kQwenDecodeKvTileSize;
    match.query_head_count      = query_head_count;
    match.key_value_head_count  = key_value_head_count;
    return match;
}

}  // namespace

static bool match_qwen_decode_split_flash_attention_next_q8_dispatch(const DispatchMatchContext & context,
                                                                     DispatchMatch &              dispatch_match) {
    const QwenDecodeSplitFlashAttentionMatch match =
        match_qwen_decode_split_flash_attention(context.graph, context.root_node);
    if (!match.matched()) {
        return false;
    }

    const int64_t key_value_block_count = ceil_div(match.key_value_capacity, kQwenDecodeKvTileSize);
    const size_t  partial_scalar_count  = static_cast<size_t>(match.key_value_head_count) *
                                        static_cast<size_t>(key_value_block_count) *
                                        static_cast<size_t>(kQwenDecodeRowCapacity);
    const size_t  partial_value_count  = partial_scalar_count * static_cast<size_t>(kQwenAttentionHeadSize);
    const size_t  partial_scalar_bytes = partial_scalar_count * sizeof(float);
    const size_t  partial_output_bytes = partial_value_count * sizeof(ggml_fp16_t);
    const int64_t hidden_size          = match.query_head_count * kQwenAttentionHeadSize;
    const size_t  q8_row_bytes         = q8_1_x4_byte_count(1, hidden_size);
    const size_t  q8_output_bytes      = q8_1_x4_byte_count(match.query_token_count, hidden_size);
    if (partial_scalar_bytes == 0 || partial_output_bytes == 0 || q8_row_bytes == 0 || q8_output_bytes == 0) {
        return false;
    }

    const ValueId partial_max        = match_value(context, dispatch_match, 0);
    const ValueId partial_sum        = match_value(context, dispatch_match, 1);
    const ValueId partial_output     = match_value(context, dispatch_match, 2);
    const ValueId completion_counter = match_value(context, dispatch_match, 3);
    const ValueId q8_output          = match_value(context, dispatch_match, 4);

    dispatch_match.transients.push_back(
        { partial_max, "qwen.decode.flash_attention.partial_max", partial_scalar_bytes, 256 });
    dispatch_match.transients.push_back(
        { partial_sum, "qwen.decode.flash_attention.partial_sum", partial_scalar_bytes, 256 });
    dispatch_match.transients.push_back(
        { partial_output, "qwen.decode.flash_attention.partial_output", partial_output_bytes, 256 });
    dispatch_match.transients.push_back(
        { q8_output, "qwen.decode.flash_attention.next_q8_output", q8_output_bytes, 256 });
    dispatch_match.completion_counter_requests.push_back({
        completion_counter,
        "qwen.decode.flash_attention.completion_counter",
        static_cast<uint32_t>(match.key_value_head_count),
    });

    Status metadata_status;
    if (!dispatch_match.metadata.append_alternate_value({ match.output->id, q8_output, GGML_TYPE_Q8_1, q8_output_bytes,
                                                          "qwen.decode.flash_attention.next_q8_output" },
                                                        metadata_status)) {
        dispatch_match.status.append(metadata_status);
        return false;
    }

    const size_t query_row_bytes  = static_cast<size_t>(hidden_size) * sizeof(float);
    const size_t mask_row_bytes   = static_cast<size_t>(match.key_value_token_count) * sizeof(ggml_fp16_t);
    const size_t output_row_bytes = query_row_bytes;
    for (int64_t row = 0; row < match.query_token_count; ++row) {
        Dispatch dispatch;
        dispatch.kernel = make_kernel_specialization(kQwenFlashAttentionDecodeSplitNextQ8Kernel);
        dispatch.kernel.integer_parameters.emplace("key_value_token_count", match.key_value_token_count);
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.attention.query_head_count",
                                                   to_config_value(match.query_head_count));
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.attention.key_value_head_count",
                                                   to_config_value(match.key_value_head_count));
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.attention.key_value_token_capacity",
                                                   to_config_value(match.key_value_capacity));
        dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity", "1");
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
        dispatch.bindings.push_back({ q8_output, static_cast<size_t>(row) * q8_row_bytes, q8_row_bytes });
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

static bool match_qwen_flash_attention_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    const QwenFlashAttentionMatch match = match_qwen_flash_attention(context.graph, context.plan, context.root_node);
    if (!match.matched()) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(kQwenFlashAttentionF32F16WmmaKernel);
    dispatch.kernel.integer_parameters.emplace("query_token_count", match.query_token_count);
    dispatch.kernel.integer_parameters.emplace("key_value_token_count", match.key_value_token_count);
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.attention.query_head_count",
                                               to_config_value(match.query_head_count));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.attention.key_value_head_count",
                                               to_config_value(match.key_value_head_count));
    dispatch.kernel.compile_parameters.emplace("qwen3_moe.workload.token_capacity",
                                               to_config_value(match.query_token_count));
    dispatch.bindings.push_back({ match.query->id, 0, match.query->byte_count });
    dispatch.bindings.push_back({ match.key->id, 0, match.key->byte_count });
    dispatch.bindings.push_back({ match.value->id, 0, match.value->byte_count });
    dispatch.bindings.push_back({ match.mask_binding_value, 0, match.mask_binding_bytes });
    dispatch.bindings.push_back({ match.output->id, 0, match.output->byte_count });

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

void register_qwen_flash_attention_dispatches(DispatchRegistryBuilder & registry) {
    registry.add({
        "qwen.flash_attention_decode_split_next_q8",
        GGML_OP_FLASH_ATTN_EXT,
        DispatchMatchKind::SingleOp,
        200,
        DispatchSource::Qwen,
        match_qwen_decode_split_flash_attention_next_q8_dispatch,
    });
    registry.add({
        "qwen.flash_attention_f32_f16_wmma",
        GGML_OP_FLASH_ATTN_EXT,
        DispatchMatchKind::SingleOp,
        100,
        DispatchSource::Qwen,
        match_qwen_flash_attention_dispatch,
    });
}

}  // namespace ggml::hrx
