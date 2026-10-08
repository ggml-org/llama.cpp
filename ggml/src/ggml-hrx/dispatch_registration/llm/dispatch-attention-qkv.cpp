#include "dispatch-attention-qkv.h"

#include "../common/dispatch-mul-mat-common.h"
#include "ggml.h"
#include "graph/graph-matcher.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace ggml::hrx {
namespace {

static constexpr KernelCatalogRef kAttentionQMatMulRopeTiledF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_attention_q_matmul_rope_tiled_f32_f32");
static constexpr KernelCatalogRef kAttentionKMatMulRopeSetRowsTiledF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_attention_k_matmul_rope_set_rows_tiled_f32_f32");
static constexpr KernelCatalogRef kAttentionVMatMulSetRowsTiledF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_attention_v_matmul_set_rows_tiled_f32_f32");
static constexpr KernelCatalogRef kAttentionQMatMulRopeVectorF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_attention_q_matmul_rope_vector_f32_f32");
static constexpr KernelCatalogRef kAttentionKMatMulRopeSetRowsVectorF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_attention_k_matmul_rope_set_rows_vector_f32_f32");
static constexpr KernelCatalogRef kAttentionVMatMulSetRowsVectorF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_attention_v_matmul_set_rows_vector_f32_f32");
static constexpr KernelCatalogRef kAttentionQMatMulRopeLegacyF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_attention_q_matmul_rope_legacy_f32_f32");
static constexpr KernelCatalogRef kAttentionKMatMulRopeSetRowsLegacyF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_attention_k_matmul_rope_set_rows_legacy_f32_f32");
static constexpr KernelCatalogRef kAttentionVMatMulSetRowsLegacyF32F32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "llm_attention_v_matmul_set_rows_legacy_f32_f32");

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool is_1d_shape(const Value & value, int64_t ne0) {
    return value.ne[0] == ne0 && value.ne[1] == 1 && value.ne[2] == 1 && value.ne[3] == 1;
}

static bool is_2d_shape(const Value & value, int64_t ne0, int64_t ne1) {
    return value.ne[0] == ne0 && value.ne[1] == ne1 && value.ne[2] == 1 && value.ne[3] == 1;
}

static bool is_2d(const Value & value) {
    return value.ne[0] > 0 && value.ne[1] > 0 && value.ne[2] == 1 && value.ne[3] == 1;
}

static bool is_attention_rope_shape(const Value & value) {
    return value.ne[0] >= 4 && value.ne[0] <= 1024 && value.ne[0] % 4 == 0 && value.ne[1] >= 1 && value.ne[1] <= 64 &&
           value.ne[2] >= 1 && value.ne[2] <= 2048 && value.ne[3] == 1;
}

static bool is_supported_dense_input_size(int64_t input_size) {
    return input_size >= 256 && input_size <= 32768 && input_size % 256 == 0;
}

static bool is_supported_decode_input_size(int64_t input_size) {
    return input_size >= 256 && input_size <= 32768 && input_size % 32 == 0;
}

static bool is_supported_dense_output_size(int64_t output_size) {
    return output_size >= 1 && output_size <= 262144;
}

static bool is_supported_decode_output_size(int64_t output_size) {
    return output_size >= 1 && output_size <= 32768;
}

static bool is_supported_prefill_token_count(int64_t token_count) {
    return token_count > 1 && token_count <= 2048;
}

static bool is_supported_decode_token_count(int64_t token_count) {
    return token_count == 1;
}

static bool is_supported_attention_token_count(int64_t token_count) {
    return is_supported_decode_token_count(token_count) || is_supported_prefill_token_count(token_count);
}

static bool is_supported_attention_input_size(int64_t input_size, int64_t token_count) {
    return is_supported_decode_token_count(token_count) ? is_supported_decode_input_size(input_size) :
                                                          is_supported_dense_input_size(input_size);
}

static bool is_supported_attention_output_size(int64_t output_size, int64_t token_count) {
    return is_supported_decode_token_count(token_count) ? is_supported_decode_output_size(output_size) :
                                                          is_supported_dense_output_size(output_size);
}

static bool is_supported_attention_cache_row_count(int64_t row_count) {
    return row_count >= 1 && row_count <= 1048576;
}

static bool same_shape(const Value & lhs, const Value & rhs) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (lhs.ne[i] != rhs.ne[i]) {
            return false;
        }
    }
    return true;
}

static std::string to_config_value(int64_t value) {
    return std::to_string(value);
}

static bool is_supported_attention_rope_params(const RopeParams & params, int64_t head_size) {
    return params.n_dims == head_size && params.mode == GGML_ROPE_TYPE_NORMAL && std::isfinite(params.freq_base) &&
           params.freq_base > 0.0f && std::isfinite(params.freq_scale) && params.freq_scale > 0.0f &&
           params.ext_factor == 0.0f && params.attn_factor == 1.0f;
}

static bool build_rope_theta_table(const GraphNode & rope, int64_t head_size, std::vector<uint8_t> & data) {
    const RopeParams * params = op_params_as<RopeParams>(rope.params);
    if (params == nullptr || !is_supported_attention_rope_params(*params, head_size)) {
        return false;
    }

    data.resize(static_cast<size_t>(head_size / 2) * sizeof(float));
    const float theta_scale = std::pow(params->freq_base, -2.0f / static_cast<float>(head_size));
    float       theta       = params->freq_scale;
    for (int64_t i = 0; i < head_size / 2; ++i) {
        std::memcpy(data.data() + static_cast<size_t>(i) * sizeof(float), &theta, sizeof(theta));
        theta *= theta_scale;
    }
    return true;
}

static void build_unit_frequency_factors(int64_t head_size, std::vector<uint8_t> & data) {
    data.resize(static_cast<size_t>(head_size / 2) * sizeof(float));
    const float one = 1.0f;
    for (int64_t i = 0; i < head_size / 2; ++i) {
        std::memcpy(data.data() + static_cast<size_t>(i) * sizeof(float), &one, sizeof(one));
    }
}

static bool cache_output_format_value(ggml_type type, int64_t & value) {
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

static ValueId next_match_transient_value(const DispatchMatchContext & context, const DispatchMatch & dispatch_match) {
    return ValueId(context.next_plan_value.value + static_cast<int32_t>(dispatch_match.transients.size()) +
                   static_cast<int32_t>(dispatch_match.completion_counter_requests.size()));
}

static bool node_is_available(const DispatchMatchContext & context, const GraphNode * node) {
    size_t node_index = 0;
    return node != nullptr && context.graph.index().node_index(node, node_index) &&
           node_index < context.covered_nodes.size() && !context.covered_nodes[node_index];
}

static const GraphNode * find_only_available_consumer(const DispatchMatchContext & context, ValueId value) {
    if (!context.graph.has_index()) {
        return nullptr;
    }
    const std::vector<const GraphNode *> & consumers = context.graph.index().consumers(value);
    if (consumers.size() != 1 || !node_is_available(context, consumers.front())) {
        return nullptr;
    }
    return consumers.front();
}

static const GraphNode * find_only_consumer_after_layout_aliases(const DispatchMatchContext &     context,
                                                                 const Value *                    start,
                                                                 std::vector<const GraphNode *> & layouts,
                                                                 const Value *&                   current,
                                                                 size_t                           max_layouts) {
    current = start;
    for (;;) {
        if (current == nullptr) {
            return nullptr;
        }
        const GraphNode * consumer = find_only_available_consumer(context, current->id);
        if (consumer == nullptr) {
            return nullptr;
        }
        if (!is_layout_alias_node(context.graph, *consumer)) {
            return consumer;
        }
        if (layouts.size() >= max_layouts) {
            return nullptr;
        }
        const Value * output = graph_value(context.graph, consumer->output);
        if (output == nullptr || output->type != current->type || !output->contiguous) {
            return nullptr;
        }
        layouts.push_back(consumer);
        current = output;
    }
}

struct AttentionMatMulMatch {
    const Value *            input         = nullptr;
    const Value *            weight        = nullptr;
    const Value *            output        = nullptr;
    int64_t                  input_size    = 0;
    int64_t                  output_size   = 0;
    int64_t                  token_count   = 0;
    CommonMulMatWeightFormat weight_format = CommonMulMatWeightFormat::Q4K;

    bool matched() const { return input != nullptr && weight != nullptr && output != nullptr; }
};

struct AttentionRopeMatch {
    const GraphNode *    node               = nullptr;
    const Value *        input              = nullptr;
    const Value *        positions          = nullptr;
    const Value *        freq_factors       = nullptr;
    const Value *        output             = nullptr;
    int64_t              token_count        = 0;
    int64_t              head_count         = 0;
    int64_t              head_size          = 0;
    size_t               theta_bytes        = 0;
    size_t               freq_factors_bytes = 0;
    std::vector<uint8_t> theta_data;
    std::vector<uint8_t> freq_factors_data;

    bool matched() const { return node != nullptr && input != nullptr && positions != nullptr && output != nullptr; }
};

struct AttentionSetRowsMatch {
    const GraphNode * node            = nullptr;
    const Value *     rows            = nullptr;
    const Value *     indices         = nullptr;
    const Value *     output          = nullptr;
    int64_t           output_format   = 0;
    int64_t           token_count     = 0;
    int64_t           cache_row_count = 0;
    int64_t           output_size     = 0;

    bool matched() const { return node != nullptr && rows != nullptr && indices != nullptr && output != nullptr; }
};

enum class AttentionQkvProjectionKind {
    Query,
    Key,
    Value,
};

struct AttentionQkvMatch {
    AttentionMatMulMatch           root;
    AttentionRopeMatch             rope;
    AttentionSetRowsMatch          set_rows;
    std::vector<const GraphNode *> layout_nodes;
    KernelCatalogRef               kernel = {};
    AttentionQkvProjectionKind     kind   = AttentionQkvProjectionKind::Query;
    bool                           use_f16_activation = false;

    bool matched() const { return root.matched() && kernel.id != kUncatalogedKernelId; }
};

static bool prefill_v_cache_fusion_disabled() {
    const char * value = std::getenv("GGML_HRX_DISABLE_PREFILL_V_CACHE_FUSION");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 && std::strcmp(value, "FALSE") != 0 &&
           std::strcmp(value, "off") != 0 && std::strcmp(value, "OFF") != 0;
}

static bool distinct_storage(const Graph & graph, const Value & lhs, const Value & rhs) {
    return !graph.values().same_storage(lhs.id, rhs.id);
}

static bool is_q6_f16_prefill_v_cache_cell(const DispatchMatchContext & context,
                                           const AttentionMatMulMatch & root,
                                           const AttentionSetRowsMatch & set_rows) {
    if (prefill_v_cache_fusion_disabled() || root.weight->type != GGML_TYPE_Q6_K || root.input_size != 1536 ||
        root.output_size != 256 || root.token_count != 64 || set_rows.output_format != 16 ||
        set_rows.cache_row_count != 512 || root.output->kind != ValueKind::Transient ||
        root.weight->alias_source.value >= 0 || !distinct_storage(context.graph, *root.input, *root.weight) ||
        !distinct_storage(context.graph, *root.input, *set_rows.indices) ||
        !distinct_storage(context.graph, *root.input, *set_rows.output) ||
        !distinct_storage(context.graph, *root.weight, *set_rows.indices) ||
        !distinct_storage(context.graph, *root.weight, *set_rows.output) ||
        !distinct_storage(context.graph, *set_rows.indices, *set_rows.output)) {
        return false;
    }

    return true;
}

static bool can_fuse_q6_f16_prefill_v_cache_cell(const DispatchMatchContext & context,
                                                 const AttentionMatMulMatch & root,
                                                 const AttentionSetRowsMatch & set_rows) {
    if (!is_q6_f16_prefill_v_cache_cell(context, root, set_rows)) {
        return false;
    }

    const size_t activation_bytes = static_cast<size_t>(root.input_size * root.token_count) * sizeof(ggml_fp16_t);
    return find_alternate_value(context.graph, context.plan, root.input->id, GGML_TYPE_F16, activation_bytes) !=
           nullptr;
}

static bool can_use_packed_lowtoken_attention_matmul(const AttentionMatMulMatch & root) {
    if (root.token_count < 1 || root.token_count > 5 || root.output_size % 64 != 0 ||
        root.weight->alias_source.value >= 0) {
        return false;
    }
    if (root.weight_format != CommonMulMatWeightFormat::Q4K &&
        root.weight_format != CommonMulMatWeightFormat::Q6K) {
        return false;
    }
    return common_is_supported_dense_input_size(root.weight_format, root.input_size);
}

static CommonMulMatWeightFormat packed_attention_lowtoken_format(CommonMulMatWeightFormat format) {
    return format == CommonMulMatWeightFormat::Q4K ? CommonMulMatWeightFormat::Q4KRow64 :
                                                     CommonMulMatWeightFormat::Q6KRow64;
}

static AttentionMatMulMatch match_attention_matmul_any_format(const Graph & graph, const GraphNode * node) {
    AttentionMatMulMatch match;
    if (node == nullptr || node->op != GGML_OP_MUL_MAT || node->inputs.size() != 2) {
        return match;
    }

    const Value * weight = graph_value(graph, node->inputs[0]);
    const Value * input  = graph_value(graph, node->inputs[1]);
    const Value * output = graph_value(graph, node->output);
    if (weight == nullptr || input == nullptr || output == nullptr || !is_2d(*weight) || !is_2d(*input) ||
        !is_2d(*output) || !weight->contiguous || !input->contiguous || !output->contiguous ||
        input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32) {
        return {};
    }

    CommonMulMatWeightFormat format = CommonMulMatWeightFormat::Q4K;
    if (!common_mul_mat_format_for_type(weight->type, format)) {
        return {};
    }

    const int64_t input_size  = weight->ne[0];
    const int64_t output_size = weight->ne[1];
    const int64_t token_count = input->ne[1];
    if (input->ne[0] != input_size || output->ne[0] != output_size || output->ne[1] != token_count ||
        !is_supported_attention_token_count(token_count) ||
        !is_supported_attention_input_size(input_size, token_count) ||
        !is_supported_attention_output_size(output_size, token_count)) {
        return {};
    }

    match.input         = input;
    match.weight        = weight;
    match.output        = output;
    match.input_size    = input_size;
    match.output_size   = output_size;
    match.token_count   = token_count;
    match.weight_format = format;
    return match;
}

static AttentionRopeMatch match_attention_rope(const Graph &     graph,
                                               const GraphNode * node,
                                               const Value *     expected_input) {
    AttentionRopeMatch match;
    if (node == nullptr || node->op != GGML_OP_ROPE || node->inputs.size() < 2 || node->inputs.size() > 3) {
        return match;
    }

    const Value * input     = graph_value(graph, node->inputs[0]);
    const Value * positions = graph_value(graph, node->inputs[1]);
    const Value * output    = graph_value(graph, node->output);
    if (input == nullptr || positions == nullptr || output == nullptr || expected_input == nullptr ||
        input->id != expected_input->id || input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
        positions->type != GGML_TYPE_I32 || !input->contiguous || !output->contiguous || !positions->contiguous ||
        !is_attention_rope_shape(*input) || !same_shape(*input, *output)) {
        return {};
    }

    const int64_t      head_size   = input->ne[0];
    const int64_t      head_count  = input->ne[1];
    const int64_t      token_count = input->ne[2];
    const RopeParams * params      = op_params_as<RopeParams>(node->params);
    if (params == nullptr || !is_supported_attention_rope_params(*params, head_size) ||
        !is_1d_shape(*positions, token_count)) {
        return {};
    }

    std::vector<uint8_t> theta_data;
    if (!build_rope_theta_table(*node, head_size, theta_data)) {
        return {};
    }

    const Value *        freq_factors       = nullptr;
    size_t               freq_factors_bytes = 0;
    std::vector<uint8_t> freq_factors_data;
    if (node->inputs.size() == 3) {
        freq_factors = graph_value(graph, node->inputs[2]);
        if (freq_factors == nullptr || freq_factors->type != GGML_TYPE_F32 || !freq_factors->contiguous ||
            !is_1d_shape(*freq_factors, head_size / 2)) {
            return {};
        }
        freq_factors_bytes = freq_factors->byte_count;
    } else {
        build_unit_frequency_factors(head_size, freq_factors_data);
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
    match.theta_bytes        = theta_data.size();
    match.freq_factors_bytes = freq_factors_bytes;
    match.theta_data         = std::move(theta_data);
    match.freq_factors_data  = std::move(freq_factors_data);
    return match;
}

static AttentionSetRowsMatch match_attention_set_rows(const Graph &     graph,
                                                      const GraphNode * node,
                                                      const Value *     expected_rows) {
    AttentionSetRowsMatch match;
    if (node == nullptr || node->op != GGML_OP_SET_ROWS || node->inputs.size() != 3) {
        return match;
    }

    const Value * rows    = graph_value(graph, node->inputs[0]);
    const Value * indices = graph_value(graph, node->inputs[1]);
    const Value * cache   = graph_value(graph, node->inputs[2]);
    const Value * output  = graph_value(graph, node->output);
    if (rows == nullptr || indices == nullptr || cache == nullptr || output == nullptr || expected_rows == nullptr ||
        rows->id != expected_rows->id || rows->type != GGML_TYPE_F32 || indices->type != GGML_TYPE_I64 ||
        output->type != cache->type || !rows->contiguous || !indices->contiguous || !cache->contiguous ||
        !same_shape(*cache, *output) || !graph.values().same_storage(cache->id, output->id)) {
        return {};
    }

    int64_t output_format = 0;
    if (!cache_output_format_value(output->type, output_format)) {
        return {};
    }

    const int64_t output_size     = rows->ne[0];
    const int64_t token_count     = rows->ne[1];
    const int64_t cache_row_count = cache->ne[1];
    if (!is_2d_shape(*rows, output_size, token_count) || !is_1d_shape(*indices, token_count) ||
        !is_2d_shape(*cache, output_size, cache_row_count) ||
        !is_supported_attention_output_size(output_size, token_count) ||
        !is_supported_attention_token_count(token_count) || !is_supported_attention_cache_row_count(cache_row_count)) {
        return {};
    }

    match.node            = node;
    match.rows            = rows;
    match.indices         = indices;
    match.output          = output;
    match.output_format   = output_format;
    match.token_count     = token_count;
    match.cache_row_count = cache_row_count;
    match.output_size     = output_size;
    return match;
}

static bool accepts_q6_f16_prefill_v_cache_cell(const DispatchMatchContext & context,
                                                const GraphNode &            consumer,
                                                const Value &                input) {
    AttentionMatMulMatch root = match_attention_matmul_any_format(context.graph, &consumer);
    if (!root.matched() || root.input->id != input.id || !context.graph.has_index()) {
        return false;
    }

    std::vector<const GraphNode *> layouts;
    const Value *     after_projection = nullptr;
    const GraphNode * set_rows_node =
        find_only_consumer_after_layout_aliases(context, root.output, layouts, after_projection, 2);
    if (set_rows_node == nullptr || set_rows_node->op != GGML_OP_SET_ROWS) {
        return false;
    }
    const AttentionSetRowsMatch set_rows = match_attention_set_rows(context.graph, set_rows_node, after_projection);
    return set_rows.matched() && set_rows.token_count == root.token_count &&
           set_rows.output_size == root.output_size && is_q6_f16_prefill_v_cache_cell(context, root, set_rows);
}

static AttentionQkvMatch match_attention_qkv_projection(const DispatchMatchContext & context) {
    AttentionQkvMatch    match;
    AttentionMatMulMatch root = match_attention_matmul_any_format(context.graph, context.root_node);
    if (!root.matched() || !context.graph.has_index()) {
        return match;
    }

    const Value *     after_projection = nullptr;
    const GraphNode * consumer =
        find_only_consumer_after_layout_aliases(context, root.output, match.layout_nodes, after_projection, 2);
    if (consumer == nullptr) {
        return {};
    }

    if (consumer->op == GGML_OP_ROPE) {
        match.rope = match_attention_rope(context.graph, consumer, after_projection);
        if (!match.rope.matched() || match.rope.token_count != root.token_count ||
            match.rope.head_size * match.rope.head_count != root.output_size) {
            return {};
        }

        const Value *                  after_rope = nullptr;
        std::vector<const GraphNode *> set_rows_layouts;
        const GraphNode *              rope_consumer =
            find_only_consumer_after_layout_aliases(context, match.rope.output, set_rows_layouts, after_rope, 2);
        if (rope_consumer != nullptr && rope_consumer->op == GGML_OP_SET_ROWS) {
            match.set_rows = match_attention_set_rows(context.graph, rope_consumer, after_rope);
            if (!match.set_rows.matched() || match.set_rows.token_count != root.token_count ||
                match.set_rows.output_size != root.output_size) {
                return {};
            }
            match.layout_nodes.insert(match.layout_nodes.end(), set_rows_layouts.begin(), set_rows_layouts.end());
            match.root   = root;
            match.kernel = is_supported_decode_token_count(root.token_count) ?
                               kAttentionKMatMulRopeSetRowsVectorF32F32Kernel :
                               kAttentionKMatMulRopeSetRowsTiledF32F32Kernel;
            if (can_use_packed_lowtoken_attention_matmul(root)) {
                match.root.weight_format = packed_attention_lowtoken_format(root.weight_format);
                match.kernel = kAttentionKMatMulRopeSetRowsLegacyF32F32Kernel;
            }
            match.kind   = AttentionQkvProjectionKind::Key;
            return match;
        }

        match.root   = root;
        match.kernel = is_supported_decode_token_count(root.token_count) ? kAttentionQMatMulRopeVectorF32F32Kernel :
                                                                           kAttentionQMatMulRopeTiledF32F32Kernel;
        if (can_use_packed_lowtoken_attention_matmul(root)) {
            match.root.weight_format = packed_attention_lowtoken_format(root.weight_format);
            match.kernel = kAttentionQMatMulRopeLegacyF32F32Kernel;
        }
        match.kind   = AttentionQkvProjectionKind::Query;
        return match;
    }

    if (consumer->op == GGML_OP_SET_ROWS) {
        match.set_rows = match_attention_set_rows(context.graph, consumer, after_projection);
        if (!match.set_rows.matched() || match.set_rows.token_count != root.token_count ||
            match.set_rows.output_size != root.output_size) {
            return {};
        }
        const bool quantized_prefill = !is_supported_decode_token_count(root.token_count) &&
                                       !common_mul_mat_dense_float_format(root.weight_format);
        if (quantized_prefill && !can_fuse_q6_f16_prefill_v_cache_cell(context, root, match.set_rows)) {
            return {};
        }
        match.root   = root;
        match.kernel = is_supported_decode_token_count(root.token_count) ? kAttentionVMatMulSetRowsVectorF32F32Kernel :
                                                                           kAttentionVMatMulSetRowsTiledF32F32Kernel;
        if (can_use_packed_lowtoken_attention_matmul(root)) {
            match.root.weight_format = packed_attention_lowtoken_format(root.weight_format);
            match.kernel = kAttentionVMatMulSetRowsLegacyF32F32Kernel;
        }
        match.kind   = AttentionQkvProjectionKind::Value;
        match.use_f16_activation = quantized_prefill;
        return match;
    }

    return {};
}

static ValueId add_attention_constant_binding(const DispatchMatchContext & context,
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

static std::pair<ValueId, ValueId> add_attention_rope_frequency_bindings(const DispatchMatchContext & context,
                                                                         const AttentionRopeMatch &   match,
                                                                         DispatchMatch &              dispatch_match) {
    const ValueId theta = add_attention_constant_binding(context, dispatch_match, "llm.attention_qkv.theta",
                                                         match.theta_bytes, match.theta_data);
    if (match.freq_factors != nullptr) {
        return { theta, match.freq_factors->id };
    }

    const ValueId freq_factors = add_attention_constant_binding(
        context, dispatch_match, "llm.attention_qkv.freq_factors", match.freq_factors_bytes, match.freq_factors_data);
    return { theta, freq_factors };
}

static void add_attention_qkv_compile_parameters(Dispatch & dispatch, const AttentionQkvMatch & match) {
    dispatch.kernel.compile_parameters.emplace("ggml.workload.token_capacity", to_config_value(match.root.token_count));
    dispatch.kernel.compile_parameters.emplace("llm.attention_qkv.input_size", to_config_value(match.root.input_size));
    dispatch.kernel.compile_parameters.emplace("llm.attention_qkv.output_size",
                                               to_config_value(match.root.output_size));
    dispatch.kernel.compile_parameters.emplace(
        "llm.attention_qkv.weight_format",
        to_config_value(common_mul_mat_format_config_value(match.root.weight_format)));
    if (match.kind == AttentionQkvProjectionKind::Query || match.kind == AttentionQkvProjectionKind::Key) {
        dispatch.kernel.compile_parameters.emplace("llm.attention_qkv.head_size",
                                                   to_config_value(match.rope.head_size));
        dispatch.kernel.compile_parameters.emplace("llm.attention_qkv.head_count",
                                                   to_config_value(match.rope.head_count));
    }
    if (match.kind == AttentionQkvProjectionKind::Key || match.kind == AttentionQkvProjectionKind::Value) {
        dispatch.kernel.compile_parameters.emplace("llm.attention_qkv.cache_row_count",
                                                   to_config_value(match.set_rows.cache_row_count));
        dispatch.kernel.compile_parameters.emplace("llm.attention_qkv.cache_output_format",
                                                   to_config_value(match.set_rows.output_format));
    }
}

static bool cover_attention_qkv_nodes(const DispatchMatchContext & context,
                                      const AttentionQkvMatch &    match,
                                      DispatchMatch &              dispatch_match) {
    if (!append_covered_node_index_once(context.graph, context.covered_nodes, context.root_node,
                                        dispatch_match.covered_nodes)) {
        return false;
    }
    for (const GraphNode * layout : match.layout_nodes) {
        if (!append_covered_node_index_once(context.graph, context.covered_nodes, layout,
                                            dispatch_match.covered_nodes)) {
            return false;
        }
    }
    if ((match.kind == AttentionQkvProjectionKind::Query || match.kind == AttentionQkvProjectionKind::Key) &&
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.rope.node,
                                        dispatch_match.covered_nodes)) {
        return false;
    }
    if ((match.kind == AttentionQkvProjectionKind::Key || match.kind == AttentionQkvProjectionKind::Value) &&
        !append_covered_node_index_once(context.graph, context.covered_nodes, match.set_rows.node,
                                        dispatch_match.covered_nodes)) {
        return false;
    }
    return true;
}

static bool match_attention_qkv_dispatch(const DispatchMatchContext & context, DispatchMatch & dispatch_match) {
    AttentionQkvMatch match = match_attention_qkv_projection(context);
    if (!match.matched() || !cover_attention_qkv_nodes(context, match, dispatch_match)) {
        return false;
    }

    Dispatch dispatch;
    bool pack_q4 = match.root.weight_format == CommonMulMatWeightFormat::Q4KRow64;
    bool pack_q6 = match.root.weight_format == CommonMulMatWeightFormat::Q6KRow64;
    bool use_q8_activation = false;
    DispatchBinding activation = { match.root.input->id, 0, match.root.input->byte_count };
    if (match.use_f16_activation) {
        const size_t activation_bytes =
            static_cast<size_t>(match.root.input_size * match.root.token_count) * sizeof(ggml_fp16_t);
        const CommandPlanAlternateValue * alternate =
            find_alternate_value(context.graph, context.plan, match.root.input->id, GGML_TYPE_F16, activation_bytes);
        if (alternate == nullptr) {
            return false;
        }
        activation = { alternate->alternate_value, 0, activation_bytes };
    }
    if (pack_q4 || pack_q6) {
        if (common_prepare_q8_1_x4_input(context, *match.root.input, match.root.input_size, match.root.token_count,
                                         dispatch_match, activation,
                                         CommonQ8ActivationPolicy::ExistingAlternateOnly)) {
            use_q8_activation = true;
        } else {
            CommonMulMatWeightFormat fallback_format = CommonMulMatWeightFormat::Q4K;
            if (!common_mul_mat_format_for_type(match.root.weight->type, fallback_format)) {
                return false;
            }
            match.root.weight_format = fallback_format;
            pack_q4 = false;
            pack_q6 = false;
            switch (match.kind) {
                case AttentionQkvProjectionKind::Query:
                    match.kernel = is_supported_decode_token_count(match.root.token_count) ?
                                       kAttentionQMatMulRopeVectorF32F32Kernel :
                                       kAttentionQMatMulRopeTiledF32F32Kernel;
                    break;
                case AttentionQkvProjectionKind::Key:
                    match.kernel = is_supported_decode_token_count(match.root.token_count) ?
                                       kAttentionKMatMulRopeSetRowsVectorF32F32Kernel :
                                       kAttentionKMatMulRopeSetRowsTiledF32F32Kernel;
                    break;
                case AttentionQkvProjectionKind::Value:
                    match.kernel = is_supported_decode_token_count(match.root.token_count) ?
                                       kAttentionVMatMulSetRowsVectorF32F32Kernel :
                                       kAttentionVMatMulSetRowsTiledF32F32Kernel;
                    break;
            }
        }
    }
    dispatch.kernel = make_kernel_specialization(match.kernel);
    dispatch.kernel.integer_parameters.emplace("token_count", match.root.token_count);
    add_attention_qkv_compile_parameters(dispatch, match);
    if (use_q8_activation) {
        dispatch.kernel.compile_parameters.emplace("ggml.mul_mat.activation_format", std::to_string(GGML_TYPE_Q8_1));
    } else if (match.use_f16_activation) {
        dispatch.kernel.compile_parameters.emplace("ggml.mul_mat.activation_format", std::to_string(GGML_TYPE_F16));
    }
    dispatch.bindings.push_back(activation);
    if (pack_q4 || pack_q6) {
        const char * layout = pack_q4 ? kQ4KPackedK256Row64Layout : kQ6KPackedK256Row64ScaleRowLayout;
        dispatch.bindings.push_back({ match.root.weight->id, 0, match.root.weight->byte_count, layout,
                                      match.root.weight->type, match.root.input_size, match.root.output_size,
                                      match.root.weight->byte_count });
    } else {
        dispatch.bindings.push_back({ match.root.weight->id, 0, match.root.weight->byte_count });
    }

    if (match.kind == AttentionQkvProjectionKind::Query) {
        const auto [theta, freq_factors] = add_attention_rope_frequency_bindings(context, match.rope, dispatch_match);
        dispatch.bindings.push_back({ match.rope.positions->id, 0, match.rope.positions->byte_count });
        dispatch.bindings.push_back({ theta, 0, match.rope.theta_bytes });
        dispatch.bindings.push_back({ freq_factors, 0, match.rope.freq_factors_bytes });
        dispatch.bindings.push_back({ match.rope.output->id, 0, match.rope.output->byte_count });
    } else if (match.kind == AttentionQkvProjectionKind::Key) {
        const auto [theta, freq_factors] = add_attention_rope_frequency_bindings(context, match.rope, dispatch_match);
        dispatch.bindings.push_back({ match.rope.positions->id, 0, match.rope.positions->byte_count });
        dispatch.bindings.push_back({ match.set_rows.indices->id, 0, match.set_rows.indices->byte_count });
        dispatch.bindings.push_back({ theta, 0, match.rope.theta_bytes });
        dispatch.bindings.push_back({ freq_factors, 0, match.rope.freq_factors_bytes });
        dispatch.bindings.push_back({ match.set_rows.output->id, 0, match.set_rows.output->byte_count });
    } else {
        dispatch.bindings.push_back({ match.set_rows.indices->id, 0, match.set_rows.indices->byte_count });
        dispatch.bindings.push_back({ match.set_rows.output->id, 0, match.set_rows.output->byte_count });
    }

    dispatch_match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_llm_attention_qkv_dispatches(DispatchRegistryBuilder & registry) {
    registry.add_activation_consumer({
        "llm.attention_v_cache.input.f16_row.prefill",
        GGML_OP_MUL_MAT,
        DispatchActivationInputF16Row,
        DispatchActivationConsumerUse::Tiled,
        DispatchActivationProducerRequirementNone,
        DispatchSource::Llm,
        accepts_q6_f16_prefill_v_cache_cell,
    });
    registry.add({
        "llm.attention_qkv_matmul_postprocess.tiled_vector_f32_f32",
        GGML_OP_MUL_MAT,
        DispatchMatchKind::Fused,
        210,
        DispatchSource::Llm,
        match_attention_qkv_dispatch,
    });
}

}  // namespace ggml::hrx
